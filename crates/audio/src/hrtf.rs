//! First-order ambisonic (FOA) decoding to a virtual speaker ring. Each speaker
//! is convolved with a parametric Brown-Duda structural HRTF (Brown and Duda,
//! 1998); the model avoids
//! embedding a measured impulse-response set in the audio module.

use core::f32::consts::PI;

pub const NUM_SPEAKERS: usize = 8;
pub const HRTF_TAPS: usize = 128;

pub struct Speaker {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub left: [f32; HRTF_TAPS],
    pub right: [f32; HRTF_TAPS],
}

fn sinc(x: f32) -> f32 {
    if x.abs() < 1e-6 {
        1.0
    } else {
        (PI * x).sin() / (PI * x)
    }
}

fn hann(i: usize, n: usize) -> f32 {
    0.5 * (1.0 - (2.0 * PI * i as f32 / (n - 1) as f32).cos())
}

fn build_delay_filter(delay_samples: f32, gain: f32, buf: &mut [f32; HRTF_TAPS]) {
    let center = (HRTF_TAPS / 2) as f32;
    let d = center + delay_samples;
    for (i, value) in buf.iter_mut().enumerate() {
        *value = gain * sinc(i as f32 - d) * hann(i, HRTF_TAPS);
    }
}

fn brown_duda_alpha(theta: f32) -> f32 {
    const ALPHA_MIN: f32 = 0.1;
    const THETA_MIN: f32 = 5.0 * PI / 6.0;
    if theta < THETA_MIN {
        (1.0 + ALPHA_MIN) / 2.0 + (1.0 - ALPHA_MIN) / 2.0 * (theta * PI / THETA_MIN).cos()
    } else {
        ALPHA_MIN
    }
}

fn apply_head_shadow(buf: &mut [f32; HRTF_TAPS], alpha: f32, tau: f32, sample_rate: f32) {
    let c = 2.0 * sample_rate;
    let denom = tau * c + 1.0;
    let b0 = (alpha * tau * c + 1.0) / denom;
    let b1 = (1.0 - alpha * tau * c) / denom;
    let a1 = (1.0 - tau * c) / denom;

    let mut prev_in: f32 = 0.0;
    let mut prev_out: f32 = 0.0;
    for value in buf.iter_mut() {
        let x = *value;
        let y = b0 * x + b1 * prev_in - a1 * prev_out;
        *value = y;
        prev_in = x;
        prev_out = y;
    }
}

fn compute_speaker(az_deg: f32, el_deg: f32, sample_rate: f32) -> Speaker {
    let az = az_deg * PI / 180.0;
    let el = el_deg * PI / 180.0;

    let x = az.sin() * el.cos();
    let y = el.sin();
    let z = az.cos() * el.cos();

    let head_radius: f32 = 0.0875;
    let speed_of_sound: f32 = 343.0;
    let tau = 2.0 * head_radius / speed_of_sound;

    let angle_left = az - PI * 0.5;
    let angle_right = az + PI * 0.5;

    let itd_left = head_radius / speed_of_sound * (angle_left.cos().clamp(-1.0, 1.0) + 1.0);
    let itd_right = head_radius / speed_of_sound * (angle_right.cos().clamp(-1.0, 1.0) + 1.0);

    let delay_left = itd_left * sample_rate;
    let delay_right = itd_right * sample_rate;

    let theta_left = (-x).acos();
    let theta_right = x.acos();
    let alpha_left = brown_duda_alpha(theta_left);
    let alpha_right = brown_duda_alpha(theta_right);

    let mut left = [0.0f32; HRTF_TAPS];
    let mut right = [0.0f32; HRTF_TAPS];
    build_delay_filter(delay_left, 1.0, &mut left);
    build_delay_filter(delay_right, 1.0, &mut right);
    apply_head_shadow(&mut left, alpha_left, tau, sample_rate);
    apply_head_shadow(&mut right, alpha_right, tau, sample_rate);

    Speaker {
        x,
        y,
        z,
        left,
        right,
    }
}


pub fn init_speakers(sample_rate: f32) -> [Speaker; NUM_SPEAKERS] {
    let dirs: [(f32, f32); NUM_SPEAKERS] = [
        (45.0, 35.26),
        (135.0, 35.26),
        (225.0, 35.26),
        (315.0, 35.26),
        (45.0, -35.26),
        (135.0, -35.26),
        (225.0, -35.26),
        (315.0, -35.26),
    ];

    let mut speakers: [Speaker; NUM_SPEAKERS] = unsafe { core::mem::zeroed() };
    for (i, &(az, el)) in dirs.iter().enumerate() {
        speakers[i] = compute_speaker(az, el, sample_rate);
    }
    speakers
}
