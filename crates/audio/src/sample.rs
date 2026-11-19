use crate::interp::hermite4;

/// Planar sample data: count 0 is unallocated and reads as silence; count 1 is
/// mono and channel-1 reads use channel 0; count 2 is stereo.
/// The host allocates one channel at a time through `sample_alloc`.
#[derive(Default)]
pub struct SampleBuffer {
    pub channels: [Vec<f32>; 2],
    pub count: u8,
}

/// Position is in samples. Keep f64 position precision: f32 loses fractional
/// positions in long buffers and cannot resolve adjacent indices past 2²⁴ samples.
/// Empty buffers and negative positions return silence. For nonempty buffers,
/// positions that cast to usize::MAX panic; other positions at or beyond the
/// buffer length return silence. NaN is not validated.
/// Holds the last value through its final fractional interval; interpolation
/// taps clamp at the buffer ends.
pub fn sample_read(buffer: &[f32], position: f64) -> f32 {
    let len = buffer.len();
    if len == 0 || position < 0.0 {
        return 0.0;
    }
    let i = position as usize;
    if i + 1 >= len {
        return if i < len { buffer[i] } else { 0.0 };
    }
    let frac = (position - i as f64) as f32;
    let ym1 = if i >= 1 { buffer[i - 1] } else { buffer[0] };
    let y2 = if i + 2 < len {
        buffer[i + 2]
    } else {
        buffer[i + 1]
    };
    hermite4(ym1, buffer[i], buffer[i + 1], y2, frac)
}

/// Position is in samples, with the same precision requirement as `sample_read`.
/// Wraps positions and interpolation taps across both ends so a seam-matched
/// loop stays C¹-continuous. Empty buffers read as silence.
pub fn sample_read_loop(buffer: &[f32], position: f64) -> f32 {
    let len = buffer.len();
    if len == 0 {
        return 0.0;
    }
    let pos = position.rem_euclid(len as f64);
    let i = (pos as usize).min(len - 1);
    let frac = (pos - i as f64) as f32;
    let ym1 = buffer[(i + len - 1) % len];
    let y1 = buffer[(i + 1) % len];
    let y2 = buffer[(i + 2) % len];
    hermite4(ym1, buffer[i], y1, y2, frac)
}
