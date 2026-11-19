/// 4-point, 3rd-order cubic Hermite interpolation, x-form (Niemitalo,
/// "Polynomial Interpolators for High-Quality Resampling of Oversampled Audio").
/// `t` is the fractional position in [0, 1] between `y0` and `y1`; `ym1` and
/// `y2` are the flanking samples. Catmull-Rom tangents (central differences):
/// interpolates `y0` at t=0, `y1` at t=1, and reproduces any polynomial up to
/// degree 2 exactly. The extra two taps reduce the aliasing linear interpolation
/// leaves in fractional-position clip reads.
pub fn hermite4(ym1: f32, y0: f32, y1: f32, y2: f32, t: f32) -> f32 {
    let c0 = y0;
    let c1 = 0.5 * (y1 - ym1);
    let c2 = ym1 - 2.5 * y0 + 2.0 * y1 - 0.5 * y2;
    let c3 = 0.5 * (y2 - ym1) + 1.5 * (y0 - y1);
    ((c3 * t + c2) * t + c1) * t + c0
}
