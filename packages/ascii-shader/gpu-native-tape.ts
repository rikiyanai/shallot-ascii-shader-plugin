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

/** Context carried by a normal GPU readback owner.  Values are deliberately
 * scalar so a driver object cannot smuggle an unbounded string into the UI. */
export type GpuReadbackFailureContext = Readonly<Record<string, string|number|undefined>>;

const GPU_READBACK_TEXT_LIMIT=256;
function boundedReadbackText(value:unknown,limit=GPU_READBACK_TEXT_LIMIT):string{
  try{
    const text=typeof value==='string'?value:String(value);
    return text.length>limit?`${text.slice(0,limit)}…`:text;
  }catch{return '<unavailable>';}
}
function readbackErrorField(error:unknown,key:'name'|'message'):string{
  try{
    if(typeof error==='string')return key==='message'?boundedReadbackText(error):'UnknownError';
    if(error instanceof Error){
      const value=error[key];
      if(typeof value==='string')return boundedReadbackText(value);
    }
    if(typeof error==='object'&&error!==null){
      const value=(error as Record<string,unknown>)[key];
      if(typeof value==='string')return boundedReadbackText(value);
    }
  }catch{/* A hostile cross-realm/driver error must not mask the cause. */}
  return key==='name'?'UnknownError':'<unavailable>';
}

/** Format a bounded, owner-specific GPU readback failure without guessing a
 * shader.  The original rejection is retained as `cause` for callers that
 * need the platform error object. */
export function gpuReadbackFailure(stage:string,error:unknown,context:GpuReadbackFailureContext={}):Error{
  const fields:string[]=[];
  try{
    for(const [key,value] of Object.entries(context)){
      if(value===undefined)continue;
      const rendered=typeof value==='number'&&Number.isFinite(value)?String(value):boundedReadbackText(value);
      fields.push(`${boundedReadbackText(key,64)}=${rendered}`);
    }
  }catch{/* Context is internal, but diagnostics must remain non-throwing. */}
  const suffix=fields.length?` (${fields.join('; ')})`:'';
  return Error(`GPU readback ${boundedReadbackText(stage,96)} failed${suffix}: ${readbackErrorField(error,'name')}: ${readbackErrorField(error,'message')}`,{cause:error});
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

  private readFailure(stage:'count header'|'scalar tape', error:unknown):Error {
    return gpuReadbackFailure(`GPU native tape export ${stage} mapAsync`,error,{lifecycle:this.disposed?'disposed during readback':'active',frame:`${this.tape.width}x${this.tape.height}`,capacity:this.tape.capacity,countOffsetBytes:this.tape.countOffsetBytes,countReadBytes:this.countRead.size,wordsReadBytes:this.wordsRead.size});
  }

  private async readImpl(): Promise<readonly GpuNativeWord[]> {
    try {
      await this.countRead.mapAsync(GPUMapMode.READ);
    } catch(error) {
      throw this.readFailure('count header',error);
    }
    let countMapped=true;
    try {
      const count = new Uint32Array(this.countRead.getMappedRange())[0]!;
      if (count > this.tape.capacity) throw Error(`GPU native tape export count ${count} exceeds capacity ${this.tape.capacity}`);
      if (!count) return [];
      if (this.countRead.mapState==='mapped') this.countRead.unmap();
      countMapped=false;
      try {
        await this.wordsRead.mapAsync(GPUMapMode.READ);
      } catch(error) {
        throw this.readFailure('scalar tape',error);
      }
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
        if (this.wordsRead.mapState==='mapped') this.wordsRead.unmap();
      }
    } finally {
      if (countMapped&&this.countRead.mapState==='mapped') this.countRead.unmap();
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
