mod hrtf;
mod interp;
mod sample;


use core::cell::UnsafeCell;
use core::f32::consts::TAU;
use hrtf::{Speaker, HRTF_TAPS, NUM_SPEAKERS};
use sample::SampleBuffer;

pub const BLOCK_SIZE: usize = 128;
pub const MAX_VOICES: usize = 64;
pub const MAX_SAMPLES: usize = 256;
const CONV_LEN: usize = BLOCK_SIZE + HRTF_TAPS - 1;
const REAL_VOICE_BUDGET: usize = 24;
const GAIN_SMOOTH_TIME: f32 = 0.005;
const SPATIAL_SMOOTH: f32 = 0.999;
const REFERENCE_DISTANCE: f32 = 3.0;
const MAX_DISTANCE: f32 = 100.0;
const ROLLOFF: f32 = 1.0;
const FADE_SAMPLES: i32 = 240;
const GAIN_SETTLE_EPSILON: f32 = 1e-5;

struct Voice {
    active: bool,
    spatial: bool,
    spatial_snap: bool,
    clip_id: u32,
    clip_position: f64,
    clip_looping: bool,
    clip_playing: bool,
    clip_stopping: bool,
    clip_paused: bool,
    pause_gain: f32,
    clip_gain: f32,
    clip_gain_target: f32,
    buffers: [[f32; BLOCK_SIZE]; 2],
    azimuth: f32,
    elevation: f32,
    distance: f32,
    azimuth_target: f32,
    elevation_target: f32,
    distance_target: f32,
    virtual_voice: bool,
    audibility: f32,
    fade_samples: i32,
}

fn default_voice() -> Voice {
    Voice {
        active: false,
        spatial: false,
        spatial_snap: true,
        clip_id: 0,
        clip_position: 0.0,
        clip_looping: false,
        clip_playing: false,
        clip_stopping: false,
        clip_paused: false,
        pause_gain: 1.0,
        clip_gain: 0.0,
        clip_gain_target: 0.0,
        buffers: [[0.0; BLOCK_SIZE]; 2],
        azimuth: 0.0,
        elevation: 0.0,
        distance: MAX_DISTANCE,
        azimuth_target: 0.0,
        elevation_target: 0.0,
        distance_target: MAX_DISTANCE,
        virtual_voice: false,
        audibility: 0.0,
        fade_samples: 0,
    }
}

fn boxed_fn<T, const N: usize>(f: impl Fn() -> T) -> Box<[T; N]> {
    let values: Vec<T> = (0..N).map(|_| f()).collect();
    values.into_boxed_slice().try_into().ok().unwrap()
}

fn distance_gain(distance: f32) -> f32 {
    if distance <= REFERENCE_DISTANCE {
        1.0
    } else {
        REFERENCE_DISTANCE / (REFERENCE_DISTANCE + ROLLOFF * (distance - REFERENCE_DISTANCE))
    }
}

#[cfg(target_feature = "simd128")]
#[inline]
fn dot2(window: &[f32], left_rev: &[f32; HRTF_TAPS], right_rev: &[f32; HRTF_TAPS]) -> (f32, f32) {
    use core::arch::wasm32::{f32x4_add, f32x4_extract_lane, f32x4_mul, f32x4_splat, v128_load};
    // SAFETY: callers provide a window at least HRTF_TAPS long; every 4-float
    // load is in bounds. simd128 is enabled for the shipped wasm target.
    unsafe {
        let (wp, lp, rp) = (window.as_ptr(), left_rev.as_ptr(), right_rev.as_ptr());
        let (mut left, mut right) = (f32x4_splat(0.0), f32x4_splat(0.0));
        let mut tap = 0;
        while tap < HRTF_TAPS {
            let samples = v128_load(wp.add(tap).cast());
            left = f32x4_add(left, f32x4_mul(samples, v128_load(lp.add(tap).cast())));
            right = f32x4_add(right, f32x4_mul(samples, v128_load(rp.add(tap).cast())));
            tap += 4;
        }
        (
            f32x4_extract_lane::<0>(left)
                + f32x4_extract_lane::<1>(left)
                + f32x4_extract_lane::<2>(left)
                + f32x4_extract_lane::<3>(left),
            f32x4_extract_lane::<0>(right)
                + f32x4_extract_lane::<1>(right)
                + f32x4_extract_lane::<2>(right)
                + f32x4_extract_lane::<3>(right),
        )
    }
}

#[cfg(not(target_feature = "simd128"))]
#[inline]
fn dot2(window: &[f32], left_rev: &[f32; HRTF_TAPS], right_rev: &[f32; HRTF_TAPS]) -> (f32, f32) {
    let (mut left, mut right) = ([0.0f32; 4], [0.0f32; 4]);
    for tap in 0..HRTF_TAPS {
        let lane = tap & 3;
        left[lane] += window[tap] * left_rev[tap];
        right[lane] += window[tap] * right_rev[tap];
    }
    (
        left[0] + left[1] + left[2] + left[3],
        right[0] + right[1] + right[2] + right[3],
    )
}

fn convolve_speaker(
    delay: &mut [f32; CONV_LEN],
    block: &[f32; BLOCK_SIZE],
    left_rev: &[f32; HRTF_TAPS],
    right_rev: &[f32; HRTF_TAPS],
    out_left: &mut [f32; BLOCK_SIZE],
    out_right: &mut [f32; BLOCK_SIZE],
) {
    delay.copy_within(BLOCK_SIZE.., 0);
    delay[HRTF_TAPS - 1..].copy_from_slice(block);
    for i in 0..BLOCK_SIZE {
        let (left, right) = dot2(&delay[i..i + HRTF_TAPS], left_rev, right_rev);
        out_left[i] = left;
        out_right[i] = right;
    }
}

pub struct AudioEngine {
    gain_smooth_coeff: f32,
    spatial_smooth_block: f32,
    voices: Box<[Voice; MAX_VOICES]>,
    samples: Vec<SampleBuffer>,
    foa_bus: [[f32; BLOCK_SIZE]; 4],
    output: [f32; BLOCK_SIZE * 2],
    speakers: [Speaker; NUM_SPEAKERS],
    hrtf_norm: f32,
    hrtf_left_rev: [[f32; HRTF_TAPS]; NUM_SPEAKERS],
    hrtf_right_rev: [[f32; HRTF_TAPS]; NUM_SPEAKERS],
    conv_buf: [[f32; CONV_LEN]; NUM_SPEAKERS],
    sorted_indices: [u8; MAX_VOICES],

}

impl AudioEngine {
    pub fn new(sample_rate: f32) -> Self {
        let mut engine = AudioEngine {
            gain_smooth_coeff: 1.0 - (-1.0 / (GAIN_SMOOTH_TIME * sample_rate)).exp(),
            spatial_smooth_block: SPATIAL_SMOOTH.powi(BLOCK_SIZE as i32),
            voices: boxed_fn(default_voice),
            samples: (0..MAX_SAMPLES).map(|_| SampleBuffer::default()).collect(),
            foa_bus: [[0.0; BLOCK_SIZE]; 4],
            output: [0.0; BLOCK_SIZE * 2],
            speakers: unsafe { core::mem::zeroed() },
            hrtf_norm: 1.0 / NUM_SPEAKERS as f32,
            hrtf_left_rev: [[0.0; HRTF_TAPS]; NUM_SPEAKERS],
            hrtf_right_rev: [[0.0; HRTF_TAPS]; NUM_SPEAKERS],
            conv_buf: [[0.0; CONV_LEN]; NUM_SPEAKERS],
            sorted_indices: [0; MAX_VOICES],

        };
        engine.speakers = hrtf::init_speakers(sample_rate);
        for speaker in 0..NUM_SPEAKERS {
            for tap in 0..HRTF_TAPS {
                engine.hrtf_left_rev[speaker][tap] =
                    engine.speakers[speaker].left[HRTF_TAPS - 1 - tap];
                engine.hrtf_right_rev[speaker][tap] =
                    engine.speakers[speaker].right[HRTF_TAPS - 1 - tap];
            }
        }
        engine
    }

    pub fn reset_state(&mut self) {
        for delay in &mut self.conv_buf {
            delay.fill(0.0);
        }
    }

    fn classify_voices(&mut self) {
        let mut count = 0usize;
        for voice_id in 0..MAX_VOICES {
            let voice = &mut self.voices[voice_id];
            if !voice.active || (voice.clip_paused && voice.pause_gain == 0.0) {
                continue;
            }
            let gain = if voice.spatial {
                distance_gain(voice.distance)
            } else {
                1.0
            };
            voice.audibility = voice.clip_gain * gain;
            self.sorted_indices[count] = voice_id as u8;
            count += 1;
        }

        if count <= REAL_VOICE_BUDGET {
            for rank in 0..count {
                let voice = &mut self.voices[self.sorted_indices[rank] as usize];
                if voice.virtual_voice {
                    voice.virtual_voice = false;
                    if !voice.clip_stopping {
                        voice.fade_samples = FADE_SAMPLES;
                    }
                }
            }
        } else {
            let ranked = &mut self.sorted_indices[..count];
            for i in 1..count {
                let key = ranked[i];
                let score = self.voices[key as usize].audibility;
                let mut j = i;
                while j > 0 && self.voices[ranked[j - 1] as usize].audibility < score {
                    ranked[j] = ranked[j - 1];
                    j -= 1;
                }
                ranked[j] = key;
            }
            let below_boundary = self.voices[ranked[REAL_VOICE_BUDGET] as usize].audibility;
            for (rank, &index) in ranked.iter().enumerate() {
                let voice = &mut self.voices[index as usize];
                if rank < REAL_VOICE_BUDGET {
                    if voice.virtual_voice
                        && (rank < REAL_VOICE_BUDGET - 1 || voice.audibility > below_boundary * 1.2)
                    {
                        voice.virtual_voice = false;
                        if !voice.clip_stopping {
                            voice.fade_samples = if voice.fade_samples < 0 {
                                -voice.fade_samples
                            } else {
                                FADE_SAMPLES
                            };
                        }
                    }
                } else if !voice.virtual_voice {
                    voice.fade_samples = if voice.fade_samples > 0 {
                        -voice.fade_samples
                    } else {
                        -FADE_SAMPLES
                    };
                    voice.virtual_voice = true;
                }
            }
        }

        let block_coeff = self.spatial_smooth_block;
        let one_minus = 1.0 - block_coeff;
        for voice in self.voices.iter_mut() {
            if !voice.active || !voice.virtual_voice || !voice.spatial {
                continue;
            }
            let mut azimuth_delta = voice.azimuth_target - voice.azimuth;
            if azimuth_delta > core::f32::consts::PI {
                azimuth_delta -= TAU;
            } else if azimuth_delta < -core::f32::consts::PI {
                azimuth_delta += TAU;
            }
            voice.azimuth += one_minus * azimuth_delta;
            voice.elevation = block_coeff * voice.elevation + one_minus * voice.elevation_target;
            voice.distance = block_coeff * voice.distance + one_minus * voice.distance_target;
        }
    }

    pub fn set_spatial(&mut self, voice_id: u32, azimuth: f32, elevation: f32, distance: f32) {
        let index = voice_id as usize;
        if index >= MAX_VOICES {
            return;
        }
        let voice = &mut self.voices[index];
        voice.azimuth_target = azimuth;
        voice.elevation_target = elevation;
        voice.distance_target = distance.min(MAX_DISTANCE);
        if voice.spatial_snap {
            voice.azimuth = voice.azimuth_target;
            voice.elevation = voice.elevation_target;
            voice.distance = voice.distance_target;
            voice.spatial_snap = false;
        }
    }

    pub fn set_voice_spatial(&mut self, voice_id: u32, spatial: u32) {
        let index = voice_id as usize;
        if index < MAX_VOICES {
            self.voices[index].spatial = spatial != 0;
        }
    }

    pub fn voice_active(&mut self, voice_id: u32, active: u32) {
        let index = voice_id as usize;
        if index >= MAX_VOICES {
            return;
        }
        let voice = &mut self.voices[index];
        if active != 0 && !voice.active {
            voice.virtual_voice = false;
            voice.spatial_snap = true;
            voice.distance = MAX_DISTANCE;
            voice.azimuth = 0.0;
            voice.elevation = 0.0;
        }
        if active == 0 {
            voice.clip_playing = false;
            voice.clip_stopping = false;
            voice.fade_samples = 0;
        }
        voice.active = active != 0;
    }

    pub fn voice_idle(&self, voice_id: u32) -> u32 {
        let index = voice_id as usize;
        if index >= MAX_VOICES {
            return 0;
        }
        let voice = &self.voices[index];
        if !voice.active {
            return 0;
        }
        u32::from(!voice.clip_playing && (!voice.clip_stopping || voice.fade_samples == 0))
    }

    pub fn play_clip(&mut self, voice_id: u32, clip_id: u32, looping: bool, gain: f32) -> bool {
        let voice_index = voice_id as usize;
        let sample_index = clip_id as usize;
        if voice_index >= MAX_VOICES || sample_index >= MAX_SAMPLES || !gain.is_finite() {
            return false;
        }
        let sample = &self.samples[sample_index];
        if sample.count == 0 || sample.channels[0].is_empty() {
            return false;
        }

        self.voice_active(voice_id, 1);
        let voice = &mut self.voices[voice_index];
        voice.clip_id = clip_id;
        voice.clip_position = 0.0;
        voice.clip_paused = false;
        voice.pause_gain = 1.0;
        voice.clip_looping = looping;
        voice.clip_playing = true;
        voice.clip_stopping = false;
        voice.clip_gain = 0.0;
        voice.clip_gain_target = gain.max(0.0);
        voice.fade_samples = FADE_SAMPLES;
        voice.virtual_voice = false;
        true
    }

    pub fn stop_clip(&mut self, voice_id: u32) {
        let index = voice_id as usize;
        if index >= MAX_VOICES {
            return;
        }
        let voice = &mut self.voices[index];
        if voice.clip_playing {
            voice.clip_stopping = true;
            voice.fade_samples = -FADE_SAMPLES;
        }
    }

    pub fn set_clip_paused(&mut self, voice_id: u32, paused: bool) {
        if let Some(voice) = self.voices.get_mut(voice_id as usize) {
            voice.clip_paused = paused;
        }
    }

    pub fn set_clip_gain(&mut self, voice_id: u32, gain: f32) {
        let index = voice_id as usize;
        if index < MAX_VOICES && gain.is_finite() && self.voices[index].active {
            self.voices[index].clip_gain_target = gain.max(0.0);
        }
    }

    pub fn set_clip_loop(&mut self, voice_id: u32, looping: bool) {
        let index = voice_id as usize;
        if index < MAX_VOICES && self.voices[index].active {
            self.voices[index].clip_looping = looping;
        }
    }

    pub fn sample_alloc(&mut self, id: u32, channel: u32, channels: u32, len: u32) -> *mut f32 {
        let index = id as usize;
        let channel = channel as usize;
        if index >= MAX_SAMPLES || channel >= 2 {
            return core::ptr::null_mut();
        }
        let sample = &mut self.samples[index];
        sample.count = channels.min(2) as u8;
        sample.channels[channel] = vec![0.0; len as usize];
        sample.channels[channel].as_mut_ptr()
    }

    pub fn clear_sample(&mut self, id: u32) {
        let index = id as usize;
        if index >= MAX_SAMPLES {
            return;
        }
        let sample = &mut self.samples[index];
        sample.channels[0].clear();
        sample.channels[1].clear();
        sample.count = 0;
    }

    fn render_clip_voice(&mut self, voice_id: usize) -> bool {
        let smooth_coeff = self.gain_smooth_coeff;
        let virtual_skip =
            self.voices[voice_id].virtual_voice && self.voices[voice_id].fade_samples >= 0;
        let samples = &self.samples;
        let voice = &mut self.voices[voice_id];
        voice.buffers[0].fill(0.0);
        voice.buffers[1].fill(0.0);

        if voice.clip_paused && voice.pause_gain == 0.0 && !voice.clip_stopping {
            return true;
        }
        let sample = &samples[voice.clip_id as usize];
        let left = &sample.channels[0];
        if !voice.clip_playing || left.is_empty() {
            voice.clip_playing = false;
            return virtual_skip;
        }
        let right = if sample.count > 1 {
            &sample.channels[1]
        } else {
            left
        };

        for frame in 0..BLOCK_SIZE {
            if voice.clip_paused && voice.pause_gain == 0.0 && !voice.clip_stopping {
                break;
            }
            let pause_step = 1.0 / FADE_SAMPLES as f32;
            voice.pause_gain = if voice.clip_paused {
                (voice.pause_gain - pause_step).max(0.0)
            } else {
                (voice.pause_gain + pause_step).min(1.0)
            };
            let delta = voice.clip_gain_target - voice.clip_gain;
            let next_gain = voice.clip_gain + smooth_coeff * delta;
            voice.clip_gain = if delta.abs() <= GAIN_SETTLE_EPSILON || next_gain == voice.clip_gain
            {
                voice.clip_gain_target
            } else {
                next_gain
            };
            if virtual_skip {
                voice.clip_position += 1.0;
            } else if voice.clip_looping {
                let length = left.len() as f64;
                voice.clip_position = voice.clip_position.rem_euclid(length);
                voice.buffers[0][frame] =
                    sample::sample_read_loop(left, voice.clip_position) * voice.clip_gain;
                voice.buffers[1][frame] =
                    sample::sample_read_loop(right, voice.clip_position) * voice.clip_gain;
                voice.clip_position += 1.0;
            } else if voice.clip_position < left.len() as f64 {
                voice.buffers[0][frame] =
                    sample::sample_read(left, voice.clip_position) * voice.clip_gain;
                voice.buffers[1][frame] =
                    sample::sample_read(right, voice.clip_position) * voice.clip_gain;
                voice.clip_position += 1.0;
            } else {
                voice.clip_playing = false;
                break;
            }

            voice.buffers[0][frame] *= voice.pause_gain;
            voice.buffers[1][frame] *= voice.pause_gain;
            if voice.clip_looping {
                voice.clip_position = voice.clip_position.rem_euclid(left.len() as f64);
            } else if voice.clip_position >= left.len() as f64 {
                voice.clip_playing = false;
            }
        }
        virtual_skip
    }

    fn render_voices(&mut self) {
        let block_coeff = self.spatial_smooth_block;
        let one_minus_block = 1.0 - block_coeff;
        let inv_block = 1.0 / BLOCK_SIZE as f32;

        for voice_id in 0..MAX_VOICES {
            if !self.voices[voice_id].active || self.render_clip_voice(voice_id) {
                continue;
            }
            let voice = &mut self.voices[voice_id];
            if voice.spatial {
                let gain_start = distance_gain(voice.distance);
                let cos_az_start = voice.azimuth.cos();
                let sin_az_start = voice.azimuth.sin();
                let cos_el_start = voice.elevation.cos();
                let sin_el_start = voice.elevation.sin();

                let mut azimuth_delta = voice.azimuth_target - voice.azimuth;
                if azimuth_delta > core::f32::consts::PI {
                    azimuth_delta -= TAU;
                } else if azimuth_delta < -core::f32::consts::PI {
                    azimuth_delta += TAU;
                }
                voice.azimuth += one_minus_block * azimuth_delta;
                voice.elevation =
                    block_coeff * voice.elevation + one_minus_block * voice.elevation_target;
                voice.distance =
                    block_coeff * voice.distance + one_minus_block * voice.distance_target;

                let gain_end = distance_gain(voice.distance);
                let mut mono = [0.0f32; BLOCK_SIZE];
                for (frame, output) in mono.iter_mut().enumerate() {
                    let t = frame as f32 * inv_block;
                    let gain = gain_start + (gain_end - gain_start) * t;
                    *output = 0.5 * (voice.buffers[0][frame] + voice.buffers[1][frame]) * gain;
                }
                let fade_samples = voice.fade_samples;
                apply_fade(fade_samples, &mut mono);
                advance_fade(&mut voice.fade_samples);

                let azimuth_end = voice.azimuth;
                let elevation_end = voice.elevation;
                let (sin_az_end, cos_az_end) = (azimuth_end.sin(), azimuth_end.cos());
                let (sin_el_end, cos_el_end) = (elevation_end.sin(), elevation_end.cos());
                for (frame, &sample) in mono.iter().enumerate() {
                    let t = frame as f32 * inv_block;
                    let sin_az = sin_az_start + (sin_az_end - sin_az_start) * t;
                    let cos_az = cos_az_start + (cos_az_end - cos_az_start) * t;
                    let sin_el = sin_el_start + (sin_el_end - sin_el_start) * t;
                    let cos_el = cos_el_start + (cos_el_end - cos_el_start) * t;
                    let value = sample;
                    self.foa_bus[0][frame] += value;
                    self.foa_bus[1][frame] += value * sin_az * cos_el;
                    self.foa_bus[2][frame] += value * sin_el;
                    self.foa_bus[3][frame] += value * cos_az * cos_el;
                }
            } else {
                let fade_samples = voice.fade_samples;
                apply_fade(fade_samples, &mut voice.buffers[0]);
                apply_fade(fade_samples, &mut voice.buffers[1]);
                advance_fade(&mut voice.fade_samples);
                for frame in 0..BLOCK_SIZE {
                    self.output[frame] += voice.buffers[0][frame];
                    self.output[BLOCK_SIZE + frame] += voice.buffers[1][frame];
                }
            }
            if voice.clip_stopping && voice.fade_samples == 0 {
                voice.clip_playing = false;
                voice.clip_stopping = false;
            }
        }
    }

    fn render_binaural(&mut self) {
        if !self
            .voices
            .iter()
            .any(|voice| voice.active && voice.spatial)
        {
            return;
        }
        let mut output = [0.0f32; BLOCK_SIZE * 2];
        let mut left = [0.0f32; BLOCK_SIZE];
        let mut right = [0.0f32; BLOCK_SIZE];
        for speaker in 0..NUM_SPEAKERS {
            let direction = &self.speakers[speaker];
            let mut block = [0.0f32; BLOCK_SIZE];
            for (frame, value) in block.iter_mut().enumerate() {
                const MAX_RE: f32 = 0.775;
                *value = self.foa_bus[0][frame]
                    + MAX_RE
                        * (direction.x * self.foa_bus[1][frame]
                            + direction.y * self.foa_bus[2][frame]
                            + direction.z * self.foa_bus[3][frame]);
            }
            convolve_speaker(
                &mut self.conv_buf[speaker],
                &block,
                &self.hrtf_left_rev[speaker],
                &self.hrtf_right_rev[speaker],
                &mut left,
                &mut right,
            );
            for frame in 0..BLOCK_SIZE {
                output[frame] += left[frame];
                output[BLOCK_SIZE + frame] += right[frame];
            }
        }
        for (output, spatial) in self.output.iter_mut().zip(output) {
            *output += spatial * self.hrtf_norm;
        }
    }

    pub fn process_pre_limiter(&mut self) -> *const f32 {
        self.classify_voices();
        self.output.fill(0.0);
        for channel in &mut self.foa_bus {
            channel.fill(0.0);
        }
        self.render_voices();
        self.render_binaural();

        self.output.as_ptr()
    }

    pub fn process(&mut self) -> *const f32 {
        let output = self.process_pre_limiter();
        for sample in &mut self.output {
            *sample = sample.tanh();
        }
        output
    }
}

fn apply_fade(fade_samples: i32, output: &mut [f32; BLOCK_SIZE]) {
    if fade_samples == 0 {
        return;
    }
    let fade_len = FADE_SAMPLES as f32;
    for (frame, sample) in output.iter_mut().enumerate() {
        let gain = if fade_samples > 0 {
            (1.0 - (fade_samples as f32 - frame as f32) / fade_len).clamp(0.0, 1.0)
        } else {
            ((-fade_samples as f32 - frame as f32) / fade_len).clamp(0.0, 1.0)
        };
        *sample *= gain;
    }
}

fn advance_fade(fade_samples: &mut i32) {
    if *fade_samples > 0 {
        *fade_samples = (*fade_samples - BLOCK_SIZE as i32).max(0);
    } else if *fade_samples < 0 {
        *fade_samples = (*fade_samples + BLOCK_SIZE as i32).min(0);
    }
}

unsafe impl Sync for Wrap {}
struct Wrap(UnsafeCell<core::mem::MaybeUninit<AudioEngine>>);
static STATE: Wrap = Wrap(UnsafeCell::new(core::mem::MaybeUninit::uninit()));

fn engine() -> &'static mut AudioEngine {
    unsafe { (*STATE.0.get()).assume_init_mut() }
}

#[no_mangle]
pub extern "C" fn audio_init(sample_rate: f32) {
    unsafe { (*STATE.0.get()).write(AudioEngine::new(sample_rate)) };
}

#[no_mangle]
pub extern "C" fn audio_reset() {
    engine().reset_state();
}

#[no_mangle]
pub extern "C" fn audio_set_spatial(voice_id: u32, azimuth: f32, elevation: f32, distance: f32) {
    engine().set_spatial(voice_id, azimuth, elevation, distance);
}

#[no_mangle]
pub extern "C" fn audio_set_voice_spatial(voice_id: u32, spatial: u32) {
    engine().set_voice_spatial(voice_id, spatial);
}

#[no_mangle]
pub extern "C" fn audio_set_clip_paused(voice_id: u32, paused: u32) {
    engine().set_clip_paused(voice_id, paused != 0);
}

#[no_mangle]
pub extern "C" fn audio_play_clip(voice_id: u32, clip_id: u32, looping: u32, gain: f32) {
    engine().play_clip(voice_id, clip_id, looping != 0, gain);
}

#[no_mangle]
pub extern "C" fn audio_stop_clip(voice_id: u32) {
    engine().stop_clip(voice_id);
}

#[no_mangle]
pub extern "C" fn audio_set_clip_gain(voice_id: u32, gain: f32) {
    engine().set_clip_gain(voice_id, gain);
}

#[no_mangle]
pub extern "C" fn audio_set_clip_loop(voice_id: u32, looping: u32) {
    engine().set_clip_loop(voice_id, looping != 0);
}

#[no_mangle]
pub extern "C" fn audio_voice_active(voice_id: u32, active: u32) {
    engine().voice_active(voice_id, active);
}

#[no_mangle]
pub extern "C" fn audio_voice_idle(voice_id: u32) -> u32 {
    engine().voice_idle(voice_id)
}

#[no_mangle]
pub extern "C" fn audio_sample_alloc(id: u32, channel: u32, channels: u32, len: u32) -> *mut f32 {
    engine().sample_alloc(id, channel, channels, len)
}

#[no_mangle]
pub extern "C" fn audio_clear_sample(id: u32) {
    engine().clear_sample(id);
}

#[no_mangle]
pub extern "C" fn audio_process() -> *const f32 {
    engine().process()
}

#[no_mangle]
pub extern "C" fn audio_process_pre_limiter() -> *const f32 {
    engine().process_pre_limiter()
}
