import type {NativeGlyph} from './native-page.ts';

/** The record written by the compiled GPU native placement pass.
 *
 * These are deliberately the placement records, not a second glyph page:
 * word.data[3] is the native x pen, word.data[4] is the row, word.data[5]
 * is an address in the resident compiled font and word.data[10] is the
 * packed foreground colour.  The GPU publication shader consumes these
 * fields without rebuilding a scalar page on the host.
 */
export const GPU_NATIVE_WORD_STRIDE = 32 as const;
export const GPU_NATIVE_WORD_BYTES = GPU_NATIVE_WORD_STRIDE * 4;

/**
 * A seal is deliberately opaque. The only production path should issue it
 * after the GPU canonical-seal stage has validated cursor and scalar
 * publication status. In particular, this is not a collection of caller-set
 * booleans that can accidentally bless proposal records for display.
 */
const GPU_NATIVE_TAPE_SEAL: unique symbol = Symbol('MAIN GPU native tape seal');
export interface GpuNativeTapeSeal {
  readonly [GPU_NATIVE_TAPE_SEAL]: true;
}

/** The small status tuple copied from the GPU canonical-seal header. */
export interface GpuNativeTapeSealStatus {
  readonly flags: number;
  readonly scalarCount: number;
  readonly capacity: number;
}

/**
 * Issue the transport seal only after the canonical GPU status has been
 * observed.  This checks cursor/scalar publication validity; it intentionally
 * does not claim that the renderer's visual construction has been accepted.
 */
export function issueGpuNativeTapeSeal(status:GpuNativeTapeSealStatus):GpuNativeTapeSeal {
  if(!Number.isSafeInteger(status.flags)||status.flags!==0)throw Error(`GPU native tape seal rejected flags ${status.flags}`);
  if(!Number.isSafeInteger(status.scalarCount)||!Number.isSafeInteger(status.capacity)||status.scalarCount<0||status.capacity<1||status.scalarCount>status.capacity)
    throw Error(`GPU native tape seal rejected scalar count ${status.scalarCount}/${status.capacity}`);
  return Object.freeze({[GPU_NATIVE_TAPE_SEAL]:true}) as GpuNativeTapeSeal;
}

/** A completed GPU construction tape that is safe to display directly.
 *
 * A tape produced before the canonical GPU pass has sealed its cursor and
 * scalar-publication status cannot be painted directly and then exported
 * differently.  The opaque `seal` is issued by that GPU stage; it is not
 * constructible by a publication caller.  This seal is a transport/parity
 * receipt, not a visual-quality or whole-construction acceptance verdict.
 */
export interface GpuNativeTape {
  readonly words: GPUBuffer;
  /** Buffer containing the committed word count at countOffsetBytes. */
  readonly count: GPUBuffer;
  readonly countOffsetBytes: number;
  readonly capacity: number;
  readonly width: number;
  readonly height: number;
  /** Number of entries in the compiled-font address space used by words[5]. */
  readonly fontAddressCount: number;
  readonly seal: GpuNativeTapeSeal;
}

export interface GpuNativeFontAddress {
  readonly text: string;
  readonly codepoint: number;
  readonly advance: number;
  /** Atlas code assigned by NativePublication. */
  readonly atlasCode: number;
}

export interface GpuNativeWord {
  readonly x: number;
  readonly row: number;
  readonly fontAddress: number;
  readonly text: string;
  readonly codepoint: number;
  readonly advance: number;
  readonly foreground: number;
}

export function validateGpuNativeTape(tape: GpuNativeTape): void {
  if (!tape.seal) {
    throw Error('GPU native tape is not a sealed shared construction');
  }
  if (!Number.isSafeInteger(tape.countOffsetBytes) || tape.countOffsetBytes < 0 ||
      tape.countOffsetBytes % 4 !== 0 ||
      !Number.isSafeInteger(tape.capacity) || tape.capacity < 1 ||
      !Number.isSafeInteger(tape.width) || tape.width < 1 ||
      !Number.isSafeInteger(tape.height) || tape.height < 1 ||
      !Number.isSafeInteger(tape.fontAddressCount) || tape.fontAddressCount < 1) {
    throw Error('GPU native tape has an invalid extent');
  }
  if (tape.words.size < tape.capacity * GPU_NATIVE_WORD_BYTES) {
    throw Error('GPU native tape word storage is truncated');
  }
  if (tape.count.size < tape.countOffsetBytes + 4) {
    throw Error('GPU native tape count storage is truncated');
  }
}

/** Explicit export path for a displayed GPU tape.  It is created only when
 * the operator asks for text; display never maps the tape.  The full payload
 * copy is acceptable here because export is an explicit, non-frame operation
 * and the count is used to decode only the committed prefix. */
export class GpuNativeTapeReadback {
  private readonly countRead: GPUBuffer;
  private readonly wordsRead: GPUBuffer;
  private encoded = false;
  private disposed = false;
  private result?: Promise<readonly GpuNativeWord[]>;

  constructor(private readonly device: GPUDevice, private readonly tape: GpuNativeTape,
              private readonly font: readonly GpuNativeFontAddress[]) {
    validateGpuNativeTape(tape);
    if (font.length !== tape.fontAddressCount) throw Error('GPU native tape/font address count mismatch');
    this.countRead = device.createBuffer({label: 'MAIN GPU native tape export count', size: 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ});
    this.wordsRead = device.createBuffer({label: 'MAIN GPU native tape export words', size: tape.capacity * GPU_NATIVE_WORD_BYTES, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ});
  }

  encode(encoder: GPUCommandEncoder): void {
    if (this.disposed) throw Error('GPU native tape export is disposed');
    if (this.encoded) throw Error('GPU native tape export was encoded twice');
    encoder.copyBufferToBuffer(this.tape.count, this.tape.countOffsetBytes, this.countRead, 0, 4);
    encoder.copyBufferToBuffer(this.tape.words, 0, this.wordsRead, 0, this.wordsRead.size);
    this.encoded = true;
  }

  async read(): Promise<readonly GpuNativeWord[]> {
    if (this.disposed) throw Error('GPU native tape export is disposed');
    if (!this.encoded) throw Error('GPU native tape export was not encoded');
    return this.result ??= this.readImpl();
  }

  private async readImpl(): Promise<readonly GpuNativeWord[]> {
    await this.countRead.mapAsync(GPUMapMode.READ);
    const count = new Uint32Array(this.countRead.getMappedRange())[0]!;
    this.countRead.unmap();
    if (count > this.tape.capacity) throw Error(`GPU native tape export count ${count} exceeds capacity ${this.tape.capacity}`);
    if (!count) return [];
    await this.wordsRead.mapAsync(GPUMapMode.READ);
    try {
      const words = new Uint32Array(this.wordsRead.getMappedRange());
      const result: GpuNativeWord[] = [];
      for (let index = 0; index < count; index++) {
        const at = index * GPU_NATIVE_WORD_STRIDE;
        const x = words[at + 3]!, row = words[at + 4]!, fontAddress = words[at + 5]!;
        const address = this.font[fontAddress];
        if (!address) throw Error(`GPU native tape export has unknown font address ${fontAddress}`);
        if (x + address.advance > this.tape.width || row * 17 + 17 > this.tape.height) {
          throw Error(`GPU native tape export has an out-of-page word at ${x},${row}`);
        }
        result.push({x, row, fontAddress, text: address.text, codepoint: address.codepoint, advance: address.advance, foreground: words[at + 10]!});
      }
      return Object.freeze(result);
    } finally {
      this.wordsRead.unmap();
    }
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    if (this.countRead.mapState === 'mapped') this.countRead.unmap();
    if (this.wordsRead.mapState === 'mapped') this.wordsRead.unmap();
    this.countRead.destroy();
    this.wordsRead.destroy();
  }
}

/** Readback is intentionally separate from display.  Callers use this only
 * from an explicit export request after encoding the request on the host's
 * normal frame encoder. */
export function requestGpuNativeTapeExport(device: GPUDevice, encoder: GPUCommandEncoder,
                                           tape: GpuNativeTape, font: readonly GpuNativeFontAddress[]): GpuNativeTapeReadback {
  const readback = new GpuNativeTapeReadback(device, tape, font);
  readback.encode(encoder);
  return readback;
}

/** Build the address records shared by the GPU publication and its lazy
 * export decoder.  `glyphs` must be the same ordered bank passed to the
 * compiled grammar packer; no runtime reordering is allowed. */
export function gpuNativeFontAddresses(glyphs: readonly NativeGlyph[], atlasCode: ReadonlyMap<string, number>): readonly GpuNativeFontAddress[] {
  return Object.freeze(glyphs.map(glyph => {
    const scalars = [...glyph.text];
    while (scalars.length > 1 && scalars.at(-1) === ' ') scalars.pop();
    const text = scalars.length === 1 ? scalars[0]! : glyph.text;
    const codepoint = text.codePointAt(0);
    const code = codepoint === undefined ? undefined : atlasCode.get(text);
    if (codepoint === undefined || code === undefined) throw Error(`GPU native tape has no atlas code for ${JSON.stringify(glyph.text)}`);
    return {text, codepoint, advance: glyph.advance - (glyph.text.length - text.length) * 5, atlasCode: code};
  }));
}
