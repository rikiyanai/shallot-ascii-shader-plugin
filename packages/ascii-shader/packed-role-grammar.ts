import type {NativeGlyph} from './native-page.ts';
import {GRAMMAR_DIRECTIONS,GRAMMAR_PHASES,GRAMMAR_RECORD_WORDS,WORD_STRIDE,type PackedRoleGrammar,type GrammarRoleEntry,type CompiledRolePart} from './role-grammar-format.ts';
import type {NativeWhitespacePolicy} from './text/native-whitespace.ts';

export const PACKED_ROLE_GRAMMAR_SCHEMA='trueff.packed_role_grammar.v1' as const;
export const PACKED_ROLE_GRAMMAR_VERSION=1 as const;
export const PACKED_ROLE_GRAMMAR_MAGIC='TSHGRM01' as const;

const HEADER_BYTES=64,ENTRY_BYTES=16,ALIGNMENT=4;
const POLICY_CODE={preformatted:0,constrained:1} as const;
const POLICY_BY_CODE=['preformatted','constrained'] as const;
const ARRAY_SECTIONS=[
  ['instructions',1,Uint32Array],['articulatedInstructions',1,Uint32Array],
  ['hardLineInstructions',1,Uint32Array],['words',1,Uint32Array],
  ['glyphs',1,Uint32Array],['points',2,Int32Array],['partAtoms',1,Uint32Array],
  ['anchorContexts',1,Uint32Array],['partPaper',1,Uint32Array],
  ['circleProgram',1,Uint32Array],['gaps',1,Uint32Array],['farStrip',1,Uint32Array],
  ['hardJunctions',1,Uint32Array],['terrainInstructions',1,Uint32Array],
  ['pairedFlanks',1,Uint32Array],['foliageEdges',1,Uint32Array],
] as const;
type ArraySectionName=typeof ARRAY_SECTIONS[number][0];
type ArrayValue=Uint32Array|Int32Array;
type Policy=NativeWhitespacePolicy;

export interface PackedRoleGrammarCodecOptions {
  readonly fontSha256:string;
}
export interface PackedRoleGrammarDecodeOptions extends PackedRoleGrammarCodecOptions {
  readonly policy:Policy;
  readonly maxWidth:number;
  /** Expected compiler source identity; required for runtime loads. */
  readonly expectedSourceSha256?:string;
}
export interface PackedRoleGrammarLoadOptions extends PackedRoleGrammarDecodeOptions {
  /** SHA-256 of the decompressed canonical artifact. */
  readonly expectedArtifactSha256:string;
  /** SHA-256 of the gzip transport bytes when `compressed` is true. */
  readonly expectedCompressedSha256?:string;
  readonly compressed?:boolean;
}
export interface PackedRoleGrammarManifest {
  readonly schema:typeof PACKED_ROLE_GRAMMAR_SCHEMA;
  readonly version:typeof PACKED_ROLE_GRAMMAR_VERSION;
  readonly policy:Policy;
  readonly maxWidth:number;
  readonly fontSha256:string;
  readonly sourceSha256:string;
  readonly cursorThreshold:number;
  readonly toneWords:number;
  readonly byteLength:number;
  readonly typedLengths:Readonly<Record<ArraySectionName,number>>;
}
interface Section {readonly name:string;readonly type:number;readonly offset:number;readonly byteLength:number;}
interface Metadata {
 schema:typeof PACKED_ROLE_GRAMMAR_SCHEMA;version:typeof PACKED_ROLE_GRAMMAR_VERSION;
 policy:Policy;maxWidth:number;fontSha256:string;sourceSha256:string;
 cursorThreshold:number;toneWords:number;gapWidth:number;wordStride:number;
 recordWords:number;directions:number;phases:number;
 parts:readonly CompiledRolePart[];partEntries:readonly GrammarRoleEntry[];
 typedLengths:Readonly<Record<ArraySectionName,number>>;
}

const align=(value:number)=>Math.ceil(value/ALIGNMENT)*ALIGNMENT;
const ascii=(value:string)=>new TextEncoder().encode(value);
const sectionType=(name:ArraySectionName)=>ARRAY_SECTIONS.find(entry=>entry[0]===name)![1];
const isCanonicalArtifact=(bytes:Uint8Array):boolean=>bytes.byteLength>=8&&new TextDecoder().decode(bytes.subarray(0,8))===PACKED_ROLE_GRAMMAR_MAGIC;
const isGzip=(bytes:Uint8Array):boolean=>bytes.byteLength>=2&&bytes[0]===0x1f&&bytes[1]===0x8b;

function assertPolicy(policy:Policy):void{
 if(!(policy in POLICY_CODE))throw new Error(`Unsupported grammar whitespace policy: ${policy}`);
}
function assertFinite(value:number,label:string):void{
 if(!Number.isSafeInteger(value)||value<0)throw new Error(`Invalid packed grammar ${label}`);
}
function typedBytes(value:ArrayValue):Uint8Array{
 return new Uint8Array(value.buffer,value.byteOffset,value.byteLength);
}

export function packedRoleGrammarManifest(grammar:PackedRoleGrammar,options:PackedRoleGrammarCodecOptions,byteLength=0):PackedRoleGrammarManifest{
 const policy=(grammar.cursorThreshold===40?'preformatted':'constrained') as Policy;
 return {schema:PACKED_ROLE_GRAMMAR_SCHEMA,version:PACKED_ROLE_GRAMMAR_VERSION,policy,maxWidth:grammar.gapWidth,
   fontSha256:options.fontSha256,sourceSha256:grammar.sourceSha256,cursorThreshold:grammar.cursorThreshold,
   toneWords:grammar.toneWords,byteLength,typedLengths:Object.fromEntries(ARRAY_SECTIONS.map(([name])=>[name,(grammar[name as ArraySectionName] as ArrayValue).length])) as Record<ArraySectionName,number>};
}

/** Encode the incumbent PackedRoleGrammar without changing a single array or scalar. */
export function encodePackedRoleGrammar(grammar:PackedRoleGrammar,options:PackedRoleGrammarCodecOptions):Uint8Array{
 assertPolicy(grammar.cursorThreshold===40?'preformatted':'constrained');
 assertFinite(grammar.gapWidth,'gap width');
 const metadata:Metadata={...packedRoleGrammarManifest(grammar,options),gapWidth:grammar.gapWidth,wordStride:WORD_STRIDE,
   recordWords:GRAMMAR_RECORD_WORDS,directions:GRAMMAR_DIRECTIONS,phases:GRAMMAR_PHASES,
   parts:grammar.parts,partEntries:grammar.partEntries,
   typedLengths:Object.fromEntries(ARRAY_SECTIONS.map(([name])=>[name,(grammar[name as ArraySectionName] as ArrayValue).length])) as Record<ArraySectionName,number>};
 const metadataBytes=ascii(JSON.stringify(metadata));
 const sections:string[]=[...ARRAY_SECTIONS.map(([name])=>name),'__metadata'];
 const sectionBytes=new Map<string,Uint8Array>();
 for(const [name] of ARRAY_SECTIONS)sectionBytes.set(name,typedBytes(grammar[name as ArraySectionName] as ArrayValue));
 sectionBytes.set('__metadata',metadataBytes);
 const sectionCount=sections.length,tableOffset=HEADER_BYTES,tableEnd=tableOffset+sectionCount*ENTRY_BYTES;
 let cursor=align(tableEnd);const entries:Section[]=[];
 for(const name of sections){const bytes=sectionBytes.get(name)!;cursor=align(cursor);entries.push({name,type:name==='__metadata'?3:sectionType(name as ArraySectionName),offset:cursor,byteLength:bytes.byteLength});cursor+=bytes.byteLength;}
 const output=new Uint8Array(cursor);const view=new DataView(output.buffer);
 output.set(ascii(PACKED_ROLE_GRAMMAR_MAGIC),0);
 view.setUint32(8,PACKED_ROLE_GRAMMAR_VERSION,true);view.setUint32(12,HEADER_BYTES,true);
 view.setUint32(16,sectionCount,true);view.setUint32(20,POLICY_CODE[metadata.policy],true);
 view.setUint32(24,metadata.maxWidth,true);view.setUint32(28,metadata.cursorThreshold,true);
 view.setUint32(32,WORD_STRIDE,true);view.setUint32(36,GRAMMAR_RECORD_WORDS,true);
 view.setUint32(40,GRAMMAR_DIRECTIONS,true);view.setUint32(44,GRAMMAR_PHASES,true);
 view.setUint32(48,metadata.toneWords,true);view.setUint32(52,metadata.gapWidth,true);
 view.setUint32(56,0,true);view.setUint32(60,0,true);
 for(const [i,entry]of entries.entries()){
  const at=tableOffset+i*ENTRY_BYTES;view.setUint32(at,i,true);view.setUint32(at+4,entry.type,true);
  view.setUint32(at+8,entry.offset,true);view.setUint32(at+12,entry.byteLength,true);
  output.set(sectionBytes.get(entry.name)!,entry.offset);
 }
 return output;
}

function readSections(data:Uint8Array):{view:DataView;sections:Section[]}{
 if(data.byteLength<HEADER_BYTES)throw new Error('Packed grammar is truncated');
 const view=new DataView(data.buffer,data.byteOffset,data.byteLength);
 const magic=new TextDecoder().decode(data.subarray(0,8));
 if(magic!==PACKED_ROLE_GRAMMAR_MAGIC)throw new Error(`Packed grammar magic mismatch: ${magic}`);
 if(view.getUint32(8,true)!==PACKED_ROLE_GRAMMAR_VERSION)throw new Error('Unsupported packed grammar version');
 if(view.getUint32(12,true)!==HEADER_BYTES)throw new Error('Packed grammar header size mismatch');
 const count=view.getUint32(16,true);if(count!==ARRAY_SECTIONS.length+1)throw new Error('Packed grammar section count mismatch');
 const sections:Section[]=[];let previous=0;
 for(let i=0;i<count;i++){
  const at=HEADER_BYTES+i*ENTRY_BYTES;if(at+ENTRY_BYTES>data.byteLength)throw new Error('Packed grammar section table is truncated');
  const offset=view.getUint32(at+8,true),byteLength=view.getUint32(at+12,true);
  if(offset%ALIGNMENT||offset<HEADER_BYTES+count*ENTRY_BYTES||byteLength<0||offset+byteLength>data.byteLength)throw new Error('Packed grammar section range is invalid');
  if(offset<previous)throw new Error('Packed grammar sections are not ordered');previous=offset+byteLength;
  sections.push({name:i===ARRAY_SECTIONS.length?'__metadata':ARRAY_SECTIONS[i]![0],type:view.getUint32(at+4,true),offset,byteLength});
 }
 return {view,sections};
}

function decodeView<T extends ArrayValue>(data:Uint8Array,section:Section,kind:'u32'|'i32'):T{
 const expected=kind==='u32'?1:2;if(section.type!==expected)throw new Error(`Packed grammar section type mismatch: ${section.name}`);
 if(section.byteLength%4||((data.byteOffset+section.offset)%4))throw new Error(`Packed grammar section alignment mismatch: ${section.name}`);
 const length=section.byteLength/4;
 return (kind==='u32'?new Uint32Array(data.buffer,data.byteOffset+section.offset,length):new Int32Array(data.buffer,data.byteOffset+section.offset,length)) as T;
}

function validateFont(font:readonly NativeGlyph[],metrics:Uint32Array,points:Int32Array,expectedSha:string,actualSha:string):void{
 if(actualSha!==expectedSha)throw new Error(`Packed grammar font hash mismatch: ${actualSha}`);
 if(metrics.length!==font.length*4)throw new Error('Packed grammar glyph table length mismatch');
 let pointAt=0;
 for(const [index,glyph]of font.entries()){
  if([...glyph.text].length!==1)throw new Error(`Native font glyph ${index} is not one scalar`);
  const at=index*4,pointCount=glyph.points.length;
  if(metrics[at]!==glyph.text.codePointAt(0)||metrics[at+1]!==glyph.advance||metrics[at+2]!==pointAt/2||metrics[at+3]!==pointCount)throw new Error(`Packed grammar glyph metadata mismatch at ${index}`);
  for(const point of glyph.points){if(points[pointAt++]!==point[0]||points[pointAt++]!==point[1])throw new Error(`Packed grammar glyph mask mismatch at ${index}`);}
 }
}

/** Decode typed-array views and reject any stale, truncated, or mismatched artifact. */
export function decodePackedRoleGrammar(bytes:Uint8Array|ArrayBuffer,font:readonly NativeGlyph[],options:PackedRoleGrammarDecodeOptions):PackedRoleGrammar{
 const data=bytes instanceof Uint8Array?bytes:new Uint8Array(bytes),{view,sections}=readSections(data),headerPolicy=view.getUint32(20,true),artifactMaxWidth=view.getUint32(24,true);
 if(headerPolicy>1||POLICY_BY_CODE[headerPolicy]!==options.policy)throw new Error('Packed grammar whitespace policy mismatch');
 if(!Number.isSafeInteger(options.maxWidth)||options.maxWidth<1||options.maxWidth>artifactMaxWidth)throw new Error('Packed grammar maxWidth mismatch');
 const sectionMap=new Map(sections.map(section=>[section.name,section]));const metaSection=sectionMap.get('__metadata')!;
 if(metaSection.type!==3)throw new Error('Packed grammar metadata section type mismatch');
 let metadata:Metadata;try{metadata=JSON.parse(new TextDecoder().decode(data.subarray(metaSection.offset,metaSection.offset+metaSection.byteLength)))}catch{throw new Error('Packed grammar metadata is invalid');}
 if(metadata.schema!==PACKED_ROLE_GRAMMAR_SCHEMA||metadata.version!==PACKED_ROLE_GRAMMAR_VERSION)throw new Error('Packed grammar metadata schema mismatch');
 if(metadata.policy!==options.policy||metadata.maxWidth!==artifactMaxWidth)throw new Error('Packed grammar metadata identity mismatch');
 if(metadata.fontSha256!==options.fontSha256)throw new Error('Packed grammar font identity mismatch');
 if(options.expectedSourceSha256!==undefined&&metadata.sourceSha256!==options.expectedSourceSha256)throw new Error('Packed grammar source identity mismatch');
 if(metadata.wordStride!==WORD_STRIDE||metadata.recordWords!==GRAMMAR_RECORD_WORDS||metadata.directions!==GRAMMAR_DIRECTIONS||metadata.phases!==GRAMMAR_PHASES)throw new Error('Packed grammar ABI mismatch');
 const arrays={} as Record<ArraySectionName,ArrayValue>;
 for(const [name]of ARRAY_SECTIONS){const section=sectionMap.get(name);if(!section)throw new Error(`Packed grammar section missing: ${name}`);const kind=name==='points'?'i32':'u32';arrays[name]=decodeView(data,section,kind);if(arrays[name].length!==metadata.typedLengths[name])throw new Error(`Packed grammar ${name} length mismatch`);}
 const typed=arrays as Record<ArraySectionName,Uint32Array|Int32Array>;
 validateFont(font,typed.glyphs as Uint32Array,typed.points as Int32Array,options.fontSha256,metadata.fontSha256);
 const expectedArtifactGaps=(artifactMaxWidth+1)*2+2*(metadata.cursorThreshold*2+1)**2;
 if(typed.gaps.length!==expectedArtifactGaps||metadata.gapWidth!==artifactMaxWidth||metadata.cursorThreshold<0)throw new Error('Packed grammar gap table mismatch');
 let gaps=typed.gaps as Uint32Array;
 if(options.maxWidth<artifactMaxWidth){
  const side=metadata.cursorThreshold*2+1,transitionOffset=(artifactMaxWidth+1)*2,transitionBytes=2*side*side;
  gaps=new Uint32Array((options.maxWidth+1)*2+transitionBytes);
  for(let leading=0;leading<2;leading++){
   const source=leading*(artifactMaxWidth+1),target=leading*(options.maxWidth+1),length=options.maxWidth+1;
   gaps.set((typed.gaps as Uint32Array).subarray(source,source+length),target);
  }
  gaps.set((typed.gaps as Uint32Array).subarray(transitionOffset,transitionOffset+transitionBytes),(options.maxWidth+1)*2);
 }
 if(!Array.isArray(metadata.parts)||!Array.isArray(metadata.partEntries))throw new Error('Packed grammar metadata parts are invalid');
 const result={...(typed as unknown as Record<string,unknown>),gaps,toneWords:metadata.toneWords,
   parts:metadata.parts,partEntries:metadata.partEntries,gapWidth:options.maxWidth,cursorThreshold:metadata.cursorThreshold,
   font,sourceSha256:metadata.sourceSha256} as unknown as PackedRoleGrammar;
 if(result.words.length%WORD_STRIDE!==0||result.glyphs.length%4!==0)throw new Error('Packed grammar record alignment mismatch');
 return result;
}

async function sha256(bytes:Uint8Array):Promise<string>{
 const subtle=globalThis.crypto?.subtle;if(!subtle)throw new Error('Packed grammar integrity requires crypto.subtle');
 const digest=await subtle.digest('SHA-256',bytes as unknown as BufferSource);return [...new Uint8Array(digest)].map(byte=>byte.toString(16).padStart(2,'0')).join('');
}
async function decompressGzip(bytes:Uint8Array):Promise<Uint8Array>{
 if(typeof DecompressionStream==='undefined')throw new Error('Packed grammar gzip requires DecompressionStream');
 const stream=new Blob([bytes as unknown as BlobPart]).stream().pipeThrough(new DecompressionStream('gzip'));
 return new Uint8Array(await new Response(stream).arrayBuffer());
}
export async function loadPackedRoleGrammar(url:string|URL,font:readonly NativeGlyph[],options:PackedRoleGrammarLoadOptions):Promise<PackedRoleGrammar>{
 const response=await fetch(url);if(!response.ok)throw new Error(`Packed grammar download failed: ${response.status}`);
 const transport=new Uint8Array(await response.arrayBuffer());
 const encoding=response.headers.get('content-encoding')?.toLowerCase()??'';
 // Fetch transparently decodes Content-Encoding:gzip in browsers. A valid
 // compressed URL therefore has two legal bodies: gzip bytes, or the
 // canonical artifact with the gzip header still visible. Detect the bytes,
 // not only the option/header, so integrity is never weakened by either path.
 let bytes:Uint8Array;
 if(isGzip(transport)){
  if(options.expectedCompressedSha256===undefined)throw new Error('Packed grammar compressed load requires compressed artifact hash');
  if(await sha256(transport)!==options.expectedCompressedSha256)throw new Error('Packed grammar compressed artifact hash mismatch');
  bytes=await decompressGzip(transport);
 }else if(isCanonicalArtifact(transport)){
  bytes=transport;
 }else if(options.compressed||encoding.includes('gzip')){
  throw new Error('Packed grammar response is neither gzip nor canonical artifact bytes');
 }else bytes=transport;
 if(await sha256(bytes)!==options.expectedArtifactSha256)throw new Error('Packed grammar artifact hash mismatch');
 return decodePackedRoleGrammar(bytes,font,options);
}
