/** A NAMED construction endpoint's whole native ink interface. This is not
 * anonymous nearby ink, a stretched glyph or a substitute for source paper.
 * Pixel centres are half-integers; anchors may be projected fractional points.
 * The8×8window holds every pixel centre within the existing3px anchor radius. */
export type NativeInterfacePoint = readonly [number, number];
export function nativeInterfacePoints(anchor: NativeInterfacePoint, ink: readonly NativeInterfacePoint[]): NativeInterfacePoint[] {
  if (![...anchor, ...ink.flat()].every(Number.isFinite)) throw Error('Nonfinite native interface');
  return ink.filter(p => {
    if (!Number.isInteger(p[0] - .5) || !Number.isInteger(p[1] - .5)) throw Error('Native interface needs whole pixel centres');
    return (p[0] - anchor[0]) ** 2 + (p[1] - anchor[1]) ** 2 <= 9;
  });
}
export function nativeInterfacesMeet(a: readonly NativeInterfacePoint[], b: readonly NativeInterfacePoint[]): boolean {
  return a.some(p => b.some(q => (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 <= 9));
}
export const NATIVE_SOURCE_INCIDENCE_WGSL = /* wgsl */`
// Caller establishes the SAME admitted source occurrence and named members.
// A source ID alone never donates a missing row/return relationship.
fn nativeSourceRowsMeet(pa:u32, ra:u32, xa:u32, pb:u32, rb:u32, xb:u32) -> bool {
  if (pa == pb) { return ra == rb && xa == xb; }
  let dp = i32(pb) - i32(pa); let dr = i32(rb) - i32(ra);
  return abs(dp) == 1 && dr == dp;
}
`;
export const NATIVE_INTERFACE_WGSL = NATIVE_SOURCE_INCIDENCE_WGSL + /* wgsl */`
fn nativeInterfaceOrigin(anchor: vec2<f32>) -> vec2<i32> { return vec2<i32>(floor(anchor)) - vec2(3); }
fn nativeInterfacePoint(anchor: vec2<f32>, index: u32) -> vec2<f32> {
  return vec2<f32>(nativeInterfaceOrigin(anchor) + vec2(i32(index & 7u), i32(index >> 3u))) + vec2(0.5);
}
fn nativeInterfaceAdd(mask: vec2<u32>, anchor: vec2<f32>, ink: vec2<f32>) -> vec2<u32> {
  let delta = ink - anchor; if (dot(delta, delta) > 9.0) { return mask; }
  let cell = vec2<i32>(floor(ink)) - nativeInterfaceOrigin(anchor);
  if (any(cell < vec2(0)) || any(cell >= vec2(8))) { return mask; }
  let bit = u32(cell.y * 8 + cell.x); var out = mask;
  out[bit / 32u] |= 1u << (bit % 32u); return out;
}
fn nativeInterfacesMeet(a: vec2<u32>, aa: vec2<f32>, b: vec2<u32>, ba: vec2<f32>) -> bool {
  if (!any(a != vec2(0u)) || !any(b != vec2(0u))) { return false; }
  let delta = nativeInterfaceOrigin(aa) - nativeInterfaceOrigin(ba);
  if (any(abs(delta) > vec2(10))) { return false; }
  // Exact integer-pixel disk, NOT all pairs of glyph pixels. Both masks use
  // native half-integer centres, so dx²+dy²<=9 has these seven row reaches.
  for (var y = 0u; y < 8u; y++) {
    let row = (a[y / 4u] >> ((y % 4u) * 8u)) & 255u; if (row == 0u) { continue; }
    for (var dy = -3; dy <= 3; dy++) {
      let by = i32(y) + delta.y + dy; if (by < 0 || by >= 8) { continue; }
      let other = (b[u32(by) / 4u] >> ((u32(by) % 4u) * 8u)) & 255u;
      var spread = row << 3u;
      if (abs(dy) <= 2) { spread |= (row << 1u) | (row << 2u) | (row << 4u) | (row << 5u); }
      if (dy == 0) { spread |= row | (row << 6u); }
      let shift = delta.x - 3;
      if (shift < 0) { spread >>= u32(-shift); } else { spread <<= u32(shift); }
      if ((spread & other) != 0u) { return true; }
    }
  } return false;
}
// A closest point remains only a SOFT fit metric. It never decides whether the
// whole interfaces are compatible.64is the explicit empty-interface sentinel.
fn nativeInterfaceClosest(mask: vec2<u32>, anchor: vec2<f32>) -> u32 {
  var best = 1e30; var index = 64u;
  for (var w = 0u; w < 2u; w++) { var bits = mask[w];
    loop { if (bits == 0u) { break; } let bit = firstTrailingBit(bits); bits &= bits - 1u;
      let i = w * 32u + bit; let d = nativeInterfacePoint(anchor, i) - anchor;
      if (dot(d, d) < best) { best = dot(d, d); index = i; }
    }
  } return index;
}
`;
