import type {ConstructionType,ToneRegions} from './composition.ts';
import {decodeSourceWholeConstructionTemplate,encodeSourceWholeConstructionTemplate} from './source-whole-construction-codec.ts';
import type {SourceWholeConstructionTemplate} from './source-lod.ts';
import type {SourceSmallParent,SourceSmallParentEnd,SourceSmallParentKind} from './source-small-parent.ts';
export {decodeSourceWholeConstructionTemplate,encodeSourceWholeConstructionTemplate} from './source-whole-construction-codec.ts';

/** Static source decisions only. Original geometry, indexed incidence, fans,
 * contact lists and expanded run packets are deliberately NOT this format.
 * The GPU source owner derives those local numeric fields from the original
 * source. Welded topology is consumed by the build's semantic compiler; it
 * is not rebuilt in a displayed-frame GPU packet with no component reader. */
export interface SourceSemanticResidual {
 readonly vertices:number;readonly corners:number;readonly indexedEdges:number;
 readonly type:ConstructionType;readonly styleWord:number;
 readonly foliage:boolean;readonly organic:boolean;readonly planeRegions:number;
 readonly publishedAdjacency:boolean;
 readonly boundaries:number;readonly contacts:number;readonly contactMembers:number;
 /** Two bits per ORIGINAL halfedge address, identical to the existing class. */
 readonly edgeClasses:Uint32Array;
 /** Existing section/band/representative context; empty keeps the old fallback. */
 readonly vertexContexts:Uint32Array;
 readonly criticalVertices:Uint32Array;
 /** Sparse [original edge address, existing boundary context] decisions.
  * Boundary walking is nonlocal; do not replace pinched/open paths by a
  * transitive GPU union which would merge the compiler's distinct paths. */
 readonly boundaryContexts:Uint32Array;
 readonly faceMaterials:Uint32Array;
 readonly toneRegions?:ToneRegions;
 /** Compiled construction levels (compiled-construction-topology.ts compileEdgeLevels), header words 28/29; absent keeps all. */
 readonly edgeLevels?:Uint32Array;
 /** perf-impl-108 parent member adequacy (compileMeshProfileSafe), compiled with
  * this source revision; header word 30: 0 unknown (every older residual: the
  * member keeps its incumbent single representation), 1 safe, 2 not safe. */
 readonly profileSafe?:boolean;
 /** perf-impl-109 compiled limb-vertex table (compiled-construction-topology
  * compileLimbVertices), header words 31/32 after the levels; absent: no limb
  * (the GPU-derived run program then carries no limb table, the incumbent). */
 readonly limbVertices?:Uint32Array;
 /** perf-impl-110 source-owned soft-part packet, header words 33/34 after
  * the limb table. The packet contains one record per section/band plus a
  * one-word original-halfedge membership table. Older packets omit this
  * section and keep incumbent soft playback; records are source-revision-owned,
  * not projected runs. */
 readonly partPrograms?:Uint32Array;
 /** Source-owned whole parent construction for baked/numeric residency,
  * header words 35/36. Older packets omit it and remain unknown/incumbent. */
 readonly wholeConstruction?:SourceWholeConstructionTemplate;
 /** Smallest-tier source class for baked/numeric residency. Header words
  * 37/38 carry its optional payload and word 39 carries its schema. Older
  * packets leave all three words zero and remain unknown/incumbent. */
 readonly smallParent?:SourceSmallParent;
}

export const SOURCE_SEMANTIC_MAGIC=0x53454d52,SOURCE_SEMANTIC_VERSION=2;
const HEADER_WORDS=48;
/** Header words 37..39 are the reserved tail of the existing 48-word ABI;
 * identity remains at 40..47. The payload is deliberately fixed-width so a
 * future/unknown extension cannot hide in a decoded resident packet. */
export const SOURCE_SMALL_PARENT_SCHEMA=1,SOURCE_SMALL_PARENT_WORDS=15;
const SMALL_PARENT_AT=37,SMALL_PARENT_LENGTH=38,SMALL_PARENT_SCHEMA=39;
const TYPES:readonly ConstructionType[]=['paper','path-led','planar','rounded','far-strip','terrain'];
const wordsOf=(value:Uint32Array|Float32Array)=>new Uint32Array(value.buffer,value.byteOffset,value.length);
const SMALL_KINDS:readonly SourceSmallParentKind[]=['round','limb','box'];
const SMALL_ENDS:readonly SourceSmallParentEnd[]=['cap','contact','open'];
const MAX_UINT32=0xffffffff;
const float32=(value:number):number=>new Float32Array([value])[0]!;
const floatBits=(value:number):number=>new Uint32Array(new Float32Array([value]).buffer)[0]!;
const bitsFloat=(value:number):number=>new Float32Array(Uint32Array.of(value).buffer)[0]!;

/** Validate the source-owned class before it enters either the baked packet or
 * a live worker result. This is metadata validation only: it does not compile
 * geometry, inspect a camera, or choose a displayed representation. */
export function validateSourceSmallParent(value:SourceSmallParent|undefined):void{
 if(value===undefined)return;
 if(!value||typeof value!=='object'||!Array.isArray(value.axis)||value.axis.length!==3||
   !value.axis.every(component=>typeof component==='number'&&Number.isFinite(component)&&Number.isFinite(float32(component))))throw Error('Invalid source small-parent axis');
 const norm=Math.hypot(value.axis[0]!,value.axis[1]!,value.axis[2]!);
 if(!Number.isFinite(norm)||Math.abs(norm-1)>1e-3)throw Error('Invalid source small-parent axis extent');
 if(!SMALL_KINDS.includes(value.kind)||!SMALL_ENDS.includes(value.positiveEnd)||!SMALL_ENDS.includes(value.negativeEnd))throw Error('Invalid source small-parent kind or end');
 if(typeof value.along!=='number'||!Number.isFinite(value.along)||!(value.along>0)||!(float32(value.along)>0)||typeof value.across!=='number'||!Number.isFinite(value.across)||!(value.across>0)||!(float32(value.across)>0))throw Error('Invalid source small-parent extent');
 if(!Number.isSafeInteger(value.sections)||value.sections<=0||value.sections>MAX_UINT32||!Number.isSafeInteger(value.rectangular)||value.rectangular<0||value.rectangular>MAX_UINT32||value.rectangular>value.sections)throw Error('Invalid source small-parent evidence counts');
 const relations=value.relations;
 if(!relations||typeof relations!=='object'||!Number.isSafeInteger(relations.forks)||relations.forks<0||relations.forks>MAX_UINT32||!Number.isSafeInteger(relations.openings)||relations.openings<0||relations.openings>MAX_UINT32||!Number.isSafeInteger(relations.contacts)||relations.contacts<0||relations.contacts>MAX_UINT32||!Number.isSafeInteger(relations.seams)||relations.seams<0||relations.seams>MAX_UINT32||!Number.isSafeInteger(relations.components)||relations.components<1||relations.components>MAX_UINT32||!Number.isSafeInteger(relations.defects)||relations.defects<0||relations.defects>MAX_UINT32||typeof value.replaceable!=='boolean'||value.replaceable!==(relations.forks===0&&relations.openings===0&&relations.contacts===0&&relations.components===1&&relations.defects===0))throw Error('Invalid source small-parent relationships');
}

/** Flat, zero-object-graph asset codec. Eight identity words are supplied by
 * the build's source-content/semantic-compiler fingerprint, never an ECS id. */
export function encodeSourceSemanticResidual(source:SourceSemanticResidual,identity:Uint32Array):Uint8Array{
 if(identity.length!==8)throw Error('Source semantic identity requires 256 bits');
 validateSourceSmallParent(source.smallParent);
 const sections=[source.edgeClasses,source.vertexContexts,source.criticalVertices,source.boundaryContexts,source.faceMaterials,
  source.toneRegions?.regions??new Float32Array(),source.toneRegions?.edges??new Float32Array()];
 const levels=source.edgeLevels??new Uint32Array(),limbs=source.limbVertices??new Uint32Array(),parts=source.partPrograms??new Uint32Array(),whole=source.wholeConstruction?encodeSourceWholeConstructionTemplate(source.wholeConstruction):new Uint32Array(),small=source.smallParent?new Uint32Array(SOURCE_SMALL_PARENT_WORDS):new Uint32Array();
 if(source.smallParent){const parent=source.smallParent,packed=SMALL_KINDS.indexOf(parent.kind)|(SMALL_ENDS.indexOf(parent.positiveEnd)<<2)|(SMALL_ENDS.indexOf(parent.negativeEnd)<<4);small[0]=packed;small[1]=floatBits(parent.axis[0]!);small[2]=floatBits(parent.axis[1]!);small[3]=floatBits(parent.axis[2]!);small[4]=floatBits(parent.along);small[5]=floatBits(parent.across);small[6]=parent.sections;small[7]=parent.rectangular;small[8]=parent.relations.forks;small[9]=parent.relations.openings;small[10]=parent.relations.contacts;small[11]=parent.relations.seams;small[12]=parent.relations.components;small[13]=parent.relations.defects;small[14]=Number(parent.replaceable);}
 const size=HEADER_WORDS+sections.reduce((n,section)=>n+section.length,0)+levels.length+limbs.length+parts.length+whole.length+small.length,words=new Uint32Array(size);
 words.set([SOURCE_SEMANTIC_MAGIC,SOURCE_SEMANTIC_VERSION,size,source.vertices,source.corners,source.indexedEdges,
  TYPES.indexOf(source.type),source.styleWord,Number(source.foliage)|(Number(source.organic)<<1)|(Number(source.publishedAdjacency)<<2),
  source.planeRegions,source.boundaries,source.contacts]);
 let at=HEADER_WORDS;
 for(const [index,section]of sections.entries()){words[12+index*2]=at;words[13+index*2]=section.length;words.set(wordsOf(section),at);at+=section.length;}
 if(levels.length){words[28]=at;words[29]=levels.length;words.set(levels,at);at+=levels.length;}
 if(limbs.length){words[31]=at;words[32]=limbs.length;words.set(limbs,at);at+=limbs.length;}
 if(parts.length){words[33]=at;words[34]=parts.length;words.set(parts,at);at+=parts.length;}
 if(whole.length){words[35]=at;words[36]=whole.length;words.set(whole,at);at+=whole.length;}
 if(small.length){words[SMALL_PARENT_AT]=at;words[SMALL_PARENT_LENGTH]=small.length;words[SMALL_PARENT_SCHEMA]=SOURCE_SMALL_PARENT_SCHEMA;words.set(small,at);at+=small.length;}
 words[26]=source.contactMembers;words[30]=source.profileSafe===undefined?0:source.profileSafe?1:2;words.set(identity,40);return new Uint8Array(words.buffer);
}

/** perf-impl-108: the compiled parent member verdict alone (header word 30),
 * O(1): true safe, false not safe, undefined unknown (an older residual, or not
 * a semantic packet); the caller keeps the incumbent for undefined. */
export function semanticProfileSafe(bytes:Uint8Array|undefined):boolean|undefined{
 if(!bytes||bytes.byteOffset%4||bytes.byteLength<HEADER_WORDS*4)return undefined;
 const words=new Uint32Array(bytes.buffer,bytes.byteOffset,HEADER_WORDS);
 if(words[0]!==SOURCE_SEMANTIC_MAGIC||words[1]!==SOURCE_SEMANTIC_VERSION)return undefined;
 return words[30]===1?true:words[30]===2?false:undefined;
}

/** Views share the asset bytes. Residency may upload them directly; no
 * deserialized halfedge/plane/section objects are reconstructed on the phone. */
export function decodeSourceSemanticResidual(bytes:Uint8Array,identity?:Uint32Array):SourceSemanticResidual{
 if(bytes.byteOffset%4||bytes.byteLength<HEADER_WORDS*4||bytes.byteLength%4)throw Error('Invalid source semantic byte extent');
 const words=new Uint32Array(bytes.buffer,bytes.byteOffset,bytes.byteLength/4);
 if(words[0]!==SOURCE_SEMANTIC_MAGIC||words[1]!==SOURCE_SEMANTIC_VERSION||words[2]!==words.length||words[6]!>=TYPES.length)throw Error('Unsupported source semantic packet');
 if(identity&&(identity.length!==8||identity.some((word,i)=>word!==words[40+i])))throw Error('Source semantic identity mismatch');
 let end=HEADER_WORDS;
 const section=(index:number):Uint32Array=>{const at=words[12+index*2]!,length=words[13+index*2]!;
  if(at!==end||at+length>words.length)throw Error('Invalid source semantic section');end=at+length;return words.subarray(at,end);};
 const edgeClasses=section(0),vertexContexts=section(1),criticalVertices=section(2),boundaryContexts=section(3),faceMaterials=section(4),regions=section(5),edges=section(6);
 let edgeLevels:Uint32Array|undefined;
 if(words[29]){if(words[28]!==end||end+words[29]!>words.length||words[29]!<Math.ceil(words[4]!/2))throw Error('Invalid source semantic level extent');edgeLevels=words.subarray(end,end+words[29]!);end+=words[29]!;}
 let limbVertices:Uint32Array|undefined;
 if(words[32]){if(words[31]!==end||end+words[32]!>words.length)throw Error('Invalid source semantic limb extent');limbVertices=words.subarray(end,end+words[32]!);end+=words[32]!;}
 let partPrograms:Uint32Array|undefined;
 if(words[34]){if(words[33]!==end||end+words[34]!>words.length)throw Error('Invalid source semantic part-program extent');partPrograms=words.subarray(end,end+words[34]!);end+=words[34]!;}
 let wholeConstruction:SourceWholeConstructionTemplate|undefined;
 if(words[36]){if(words[35]!==end||end+words[36]!>words.length)throw Error('Invalid source semantic whole-construction extent');wholeConstruction=decodeSourceWholeConstructionTemplate(words.subarray(end,end+words[36]!));end+=words[36]!;}
 let smallParent:SourceSmallParent|undefined;
 const smallAt=words[SMALL_PARENT_AT]!,smallLength=words[SMALL_PARENT_LENGTH]!,smallSchema=words[SMALL_PARENT_SCHEMA]!;
 if((smallAt===0)!==(smallLength===0))throw Error('Invalid source semantic small-parent extent');
 if(smallLength===0){if(smallSchema!==0)throw Error('Unsupported source semantic small-parent schema');}
 else{
  if(smallSchema!==SOURCE_SMALL_PARENT_SCHEMA||smallAt!==end||smallLength!==SOURCE_SMALL_PARENT_WORDS||end+smallLength>words.length)throw Error('Invalid source semantic small-parent extent');
  const packed=words[smallAt]!,kind=SMALL_KINDS[packed&3],positiveEnd=SMALL_ENDS[(packed>>>2)&3],negativeEnd=SMALL_ENDS[(packed>>>4)&3];
  if((packed&0xffffffc0)!==0||kind===undefined||positiveEnd===undefined||negativeEnd===undefined)throw Error('Invalid source semantic small-parent kind');
  smallParent={kind,axis:[bitsFloat(words[smallAt+1]!),bitsFloat(words[smallAt+2]!),bitsFloat(words[smallAt+3]!)],along:bitsFloat(words[smallAt+4]!),across:bitsFloat(words[smallAt+5]!),positiveEnd,negativeEnd,sections:words[smallAt+6]!,rectangular:words[smallAt+7]!,relations:{forks:words[smallAt+8]!,openings:words[smallAt+9]!,contacts:words[smallAt+10]!,seams:words[smallAt+11]!,components:words[smallAt+12]!,defects:words[smallAt+13]!},replaceable:words[smallAt+14]!==0};
  if(words[smallAt+14]!>1)throw Error('Invalid source semantic small-parent relationships');
  validateSourceSmallParent(smallParent);end+=smallLength;
 }
 if(end!==words.length||edgeClasses.length!==Math.ceil(words[4]!/16)||criticalVertices.length%2||boundaryContexts.length%2||(faceMaterials.length>0&&faceMaterials.length!==Math.ceil(words[4]!/24))||regions.length%8||edges.length%12)throw Error('Invalid source semantic field extent');
 const floats=(values:Uint32Array)=>new Float32Array(values.buffer,values.byteOffset,values.length);
 return {vertices:words[3]!,corners:words[4]!,indexedEdges:words[5]!,type:TYPES[words[6]!]!,styleWord:words[7]!,
  foliage:(words[8]!&1)!==0,organic:(words[8]!&2)!==0,publishedAdjacency:(words[8]!&4)!==0,planeRegions:words[9]!,boundaries:words[10]!,contacts:words[11]!,contactMembers:words[26]!,
  edgeClasses,vertexContexts,criticalVertices,boundaryContexts,faceMaterials,...(regions.length?{toneRegions:{regions:floats(regions),edges:floats(edges)}}:{}),...(edgeLevels?{edgeLevels}:{}),...(limbVertices?{limbVertices}:{}),...(partPrograms?{partPrograms}:{}),...(wholeConstruction?{wholeConstruction}:{}),...(smallParent?{smallParent}:{}),
  ...(words[30]===1?{profileSafe:true}:words[30]===2?{profileSafe:false}:{})};
}
