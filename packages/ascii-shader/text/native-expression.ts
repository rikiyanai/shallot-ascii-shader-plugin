/** Depictive functions, not geometric classes or a glyph alphabet. These admit
 * source constructions; current relationships, masks and paper must still fit. */
export const NATIVE_FUNCTIONS = ['continuation', 'opening', 'band-turn', 'surface-return', 'rim-turn'] as const;
export type NativeFunction = typeof NATIVE_FUNCTIONS[number];
export function nativeExpressionKey(id: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < id.length; i++) {h ^= id.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0;}
  if (!h) throw Error('Native expression key is reserved');
  return h;
}
export function nativeFunctionMask(functions: readonly NativeFunction[]): number {
  let mask = 0;
  for (const f of functions) {
    const i = NATIVE_FUNCTIONS.indexOf(f);
    if (i < 0 || (mask & (1 << i))) throw Error(`Invalid or duplicate native function: ${f}`);
    mask |= 1 << i;
  }
  if (!mask) throw Error('Missing native visual function');
  return mask;
}
