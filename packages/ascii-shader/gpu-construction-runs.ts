
import type {RelationVisibility} from './relation-visibility.ts';
import type {CompiledConstructionGraph,ConstructionType} from './composition.ts';
import type {MeshSourceComplex} from './source-complex.ts';
import {FACE_HIDDEN,PACKED_RUN_CONTEXT,TOPOLOGY_FACE_MAGIC,TOPOLOGY_FACE_MAGIC_WORD,TOPOLOGY_LIMB_MAGIC,TOPOLOGY_LIMB_MAGIC_WORD,TOPOLOGY_PART_MAGIC,TOPOLOGY_PART_MAGIC_WORD,TOPOLOGY_PART_PROGRAM_WORDS,
 TOPOLOGY_PART_PACKET_MAGIC,TOPOLOGY_PART_CANONICAL_MAGIC,TOPOLOGY_PART_CANONICAL_VERSION,TOPOLOGY_PART_JUNCTION_MAGIC,TOPOLOGY_PART_JUNCTION_VERSION,TOPOLOGY_PART_JUNCTION_RECORD_WORDS,TOPOLOGY_PART_JUNCTION_ARM_WORDS,TOPOLOGY_PART_JUNCTION_EDGE_WORDS,
 TOPOLOGY_PART_JUNCTION_FLAG_COMPLETE,TOPOLOGY_PART_JUNCTION_FLAG_MISSING_ARM,TOPOLOGY_PART_JUNCTION_ARM_COMPLETE} from './compiled-construction-topology.ts';
import {SOURCE_JUNCTION_RECORD_LIMIT,SOURCE_JUNCTION_ARM_LIMIT} from './source-junction.ts';
import {GPU_INDEXED_CONSTRUCTION_SHADER,gpuConstructionTypeCode,gpuIndexedConstructionSlot,gpuIndexedQuadContext,QUAD_RUN_CONTEXT} from './gpu-indexed-source-construction.ts';
import type {GpuIndexedConstructionInput} from './gpu-indexed-source-construction.ts';
import {storageBinding} from './source-storage.ts';
import {DeviceComputePrograms,type ComputePipelineProgress} from './gpu-pipeline-preparation.ts';
import {isolateComputeEntry} from './gpu-compute-program.ts';
import {RELATION_RECORD_WGSL,CONSTRUCTION_POINT_WGSL,CONSTRUCTION_POINT_WORDS,POINT_SOURCE_CHART_WGSL,SPAN_SOURCE_TAG,WHOLE_EDGE_DESCRIPTOR} from './source-coordinates.ts';
import {GpuSourceFarForms,type SourceFarForms} from './gpu-source-far-forms.ts';
import {FAR_FORM_MERGE_PIXELS} from './form-depths.ts';
import type {SurfaceField} from './surface-fields.ts';

const LINK_WORDS=14;
const JUMP_WORDS=4;
const STYLE_WORDS=5;
const EMPTY_ROW=0xffffffff,RETIRED_ROW=0xfffffffe;
const ARENA_LABEL='MAIN resident compiled construction topology and work';
// A face source vertex is emitted as a dedicated joint run. This bit is
// outside the ordinary boundary flags and does not change the four-point face
// packet or its canonical source ordering.
export const FACE_JUNCTION_RUN=0x00800000;

/**
 * Decode the producer-owned address used to index a source part's original
 * halfedge membership table.  Relation.key.w remains the complete source
 * feature identity; this value is only the separate topology lookup address.
 *
 * The relation producers use different namespaces for the same kind word:
 * boundaries/creases carry halfedges, smooth and suggestive contours carry
 * faces, while ridges carry a tagged face.  Contacts and compiled hull/far
 * records are not source-halfedge members and must retain their incumbent
 * consumers.  Rejecting an unrecognised full-width word is safer than
 * truncating it into another namespace.
 */
export function decodeSourceMembershipAddress(kind:number,key:number,sourceChartFlags=0):number|undefined{
 if(!Number.isInteger(kind)||kind<0||kind>0xffffffff||!Number.isInteger(key)||key<0||key>0xffffffff||!Number.isInteger(sourceChartFlags)||sourceChartFlags<0||sourceChartFlags>0xffffffff)return undefined;
 const k=kind>>>0,w=key>>>0,flags=sourceChartFlags>>>0;
 // bit 1 = held far part; bit 2 = compiled L2 connected outline.
 if(k===4||(flags&6)!==0)return undefined;
 const maxFace=Math.floor(0xffffffff/3);
 if(k===0||k===2){
  // Original boundary/crease addresses are untagged halfedges.  Tagged
  // smooth/ridge/hull addresses must not alias a halfedge here.
  return (w&0x80000000)===0?w:undefined;
 }
 if(k===1){
  // The edge pass marks a smooth contour's canonical halfedge explicitly;
  // the face pass leaves its face number untagged.
  if((w&0x80000000)!==0)return w&0x7fffffff;
  return w<=maxFace?w*3:undefined;
 }
 if(k===3){
  const namespace=w&0xc0000000;
  if(namespace===0x40000000){
   const face=w&0x3fffffff;
   return face<=maxFace?face*3:undefined;
  }
  return namespace===0&&w<=maxFace?w*3:undefined;
 }
 return undefined;
}

/** Residency counters since page load (read through physical-world.ts
 * diagnostics). A settled view writes nothing: every count stays still. */
export const constructionArena={
 /** Resident source slots and the arena they live in. */
 sources:0,arenaBytes:0,usedBytes:0,freeBytes:0,
 /** Sources placed (arrived or changed) and their slot uploads. */
 placed:0,slotBytes:0,
 /** Table rows written (placement, retirement) and whole-table rebuilds. */
 rows:0,tables:0,retired:0,
 /** Arena growths (by half), repacks (same size, fragments closed) and the
  * bytes the GPU copied into the new buffer. */
 growths:0,repacks:0,copiedBytes:0,
};

/** First-fit allocator of arena words. Released ranges coalesce; a range
 * that reaches the end of the used extent shortens it. */
class ArenaHeap {
 end=0;private readonly free:{offset:number;words:number}[]=[];
 constructor(public capacity:number){}
 allocate(words:number):number|undefined{
  for(let i=0;i<this.free.length;i++){
   const range=this.free[i]!;if(range.words<words)continue;
   const offset=range.offset;if(range.words===words)this.free.splice(i,1);else{range.offset+=words;range.words-=words;}return offset;
  }
  if(this.end+words>this.capacity)return undefined;
  const offset=this.end;this.end+=words;return offset;
 }
 release(offset:number,words:number):void{
  if(words<=0)return;
  let i=0;while(i<this.free.length&&this.free[i]!.offset<offset)i++;
  this.free.splice(i,0,{offset,words});
  const next=this.free[i+1];if(next&&offset+words===next.offset){this.free[i]!.words+=next.words;this.free.splice(i+1,1);}
  const before=this.free[i-1];if(before&&before.offset+before.words===offset){before.words+=this.free[i]!.words;this.free.splice(i,1);}
  const last=this.free[this.free.length-1];if(last&&last.offset+last.words===this.end){this.end=last.offset;this.free.pop();}
 }
 get freeWords():number{let n=this.capacity-this.end;for(const range of this.free)n+=range.words;return n;}
}

/** Where the shader and claimRow start probing for a source address. */
const rowHash=(domain:number,capacity:number):number=>(Math.imul(domain,0x9e3779b9)>>>0)&(capacity-1);


/** Geometry/context compiled at source residency. No projected path, glyph
 * candidate, visible pixel or native pen belongs in this table. */
export interface CpuSourceConstructionClass extends CompiledConstructionGraph {
 readonly domain:number;readonly topologyRevision?:number;readonly complex?:MeshSourceComplex;readonly surface?:'liquid'|'heightfield';
 readonly farForms?:SourceFarForms;
}
export interface GpuIndexedSourceConstructionClass {
 readonly domain:number;readonly type:ConstructionType;readonly styleWord?:number;readonly topologyRevision?:number;
 readonly gpuIndexed:GpuIndexedConstructionInput;readonly runProgram?:never;readonly complex?:never;readonly surface?:'liquid'|'heightfield';
}
export type SourceConstructionClass=CpuSourceConstructionClass|GpuIndexedSourceConstructionClass;
export interface ResidentConstructionRuns {
 readonly runs:GPUBuffer;readonly points:GPUBuffer;readonly status:GPUBuffer;
 readonly sourceStatus:GPUBuffer;
 readonly capacity:number;readonly pointCapacity:number;readonly statusOffset:number;
 dispose():void;
}

/** Source incidence → complete current-view runs. The hash indexes ORIGINAL
 * ports (domain, owner, class, source vertex/edge crossing); it is not a screen
 * weld, glyph search, nearest-neighbour fit, or image-to-text algorithm.
 * Direction changes/extrema split runs; visibility only annotates them.
 * Whole rounded boundaries remain one paired source-section construction. */
export const GPU_CONSTRUCTION_RUNS_SHADER=/* wgsl */`
struct Params {extent:vec4u,rect:vec4f,sourceExtent:vec4u,table:vec4u}
${RELATION_RECORD_WGSL}
struct Interval {relation:u32,lo:f32,hi:f32,cause:u32}
struct Run {range:vec4u,identity:vec4u,part:vec4u,paper:vec4u,correspondence:vec4u}
${CONSTRUCTION_POINT_WGSL}
${POINT_SOURCE_CHART_WGSL}
struct Node {first:atomic<u32>,second:atomic<u32>,degree:atomic<u32>,least:atomic<u32>,minX:atomic<i32>,minY:atomic<i32>,maxX:atomic<i32>,maxY:atomic<i32>}
struct SourceKey {identity:vec4u,parameter:u32,contact:vec3u}
struct Link {ends:vec2u,jump:vec2u,root:vec2u,owner:u32,enabled:u32}
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read> relations:array<Relation>;
@group(0) @binding(2) var<storage,read> intervals:array<Interval>;
@group(0) @binding(3) var<storage,read> partitions:array<u32>;
@group(0) @binding(4) var<storage,read> sourceStatus:array<u32>;
@group(0) @binding(5) var<storage,read_write> runs:array<Run>;
@group(0) @binding(6) var<storage,read_write> points:array<Point>;
@group(0) @binding(7) var<storage,read_write> nodes:array<Node>;
// The persistent arena: the anchor program table (p.table.w), the source
// table (p.sourceExtent.x), one topology slot per resident source, and this
// encode's full links and four-word alternating jump/root state
// (p.table.y, p.extent.w). Immutable link fields are never ping-ponged.
@group(0) @binding(8) var<storage,read_write> work:array<u32>;
const NONE=0xffffffffu;
const LINE_NODE=0u;
const OWNER_NODE=1u;
const JUNCTION_NODE=2u;
const PAIR_NODE=3u;
const STYLE_WORDS=5u;
// The domain word of a welded terrain seam node (key): never a source domain.
const SEAM_DOMAIN=0x1ffffffdu;

const CONTACT_DOMAIN=0x1ffffffcu;
fn count()->u32{return min(sourceStatus[2],p.extent.x);}
fn kind(r:Relation)->u32{return r.key.z&0x1fffffffu;}
// The resident source table: p.table.x rows (a power of two) of STYLE_WORDS
// at work word p.sourceExtent.x, open-addressed by ORIGINAL source address
// (GpuConstructionRuns.claimRow uses the same hash and probe). A row is
// [domain, form, critical list address, critical count, slot address]; an
// empty row holds NONE, a retired one RETIRED_SOURCE. Historical domain IDs
// are identities, never buffer extents; a source arriving writes its own row
// and slot only. This lookup chooses no glyph/pen.
const RETIRED_SOURCE=0xfffffffeu;
fn sourceClass(domain:u32)->u32{
 let mask=p.table.x-1u;var at=(domain*0x9e3779b9u)&mask;
 for(var probe=0u;probe<p.table.x;probe++){
  let row=p.sourceExtent.x+at*STYLE_WORDS;let key=work[row];
  if(key==domain){return row;}if(key==NONE){return NONE;}
  at=(at+1u)&mask;
 }
 return NONE;
}
fn shape(r:Relation)->u32{let at=sourceClass(r.key.x);if(at==NONE){return 0u;}return work[at+1u]&255u;}
// Authored FACE classes override the whole-source class, including unknown.
// A mixed tree's wood must not inherit foliage merely because its crown has
// leaves. The registration hint "contains foliage" is not a surface style.
fn foliageRelation(r:Relation)->bool{
 if(kind(r)==4u){return false;}
 let face=u32(r.sourceChart.z);if(face!=0u){return face==7u;}
 let at=sourceClass(r.key.x);if(at==NONE){return false;}
 return ((work[at+1u]>>8u)&15u)==6u;
}
fn critical(r:Relation,end:u32)->u32{
 if(kind(r)==4u){return 0u;}
 let port=r.ports[end];if(port>=0x80000000u){return 0u;}
 let info=sourceClass(r.key.x);if(info==NONE){return 0u;}
 let first=work[info+2u];let count=work[info+3u];var low=0u;var high=count;
 // Source-address lookup, never a glyph candidate or ink fit.
 loop{if(low>=high){break;}let middle=(low+high)/2u;if(work[first+middle*2u]<port){low=middle+1u;}else{high=middle;}}
 if(low<count&&work[first+low*2u]==port){return work[first+low*2u+1u];}return 0u;
}
// The source's topology slot (compiled-construction-topology.ts): its header
// address, which is also the base every range in the slot is relative to.
fn topologyHeader(r:Relation)->u32{let at=sourceClass(r.key.x);if(at==NONE){return NONE;}return work[at+4u];}
// A planar form with at most 24 compiled plane regions is a volume (box, room,
// block); more planes is a planar surface (terrain, voxel landscape).
fn planarVolume(r:Relation)->bool{let header=topologyHeader(r);return header==NONE||work[header+16u]<=24u;}
// The compiled class of a source edge (compiled-construction-topology.ts,
// planeEdges): 0 a soft edge, 1 an authored feature edge (an open boundary, a
// crease beside a plane of two or more coplanar faces, a region border of a
// volume), 2 a facet crease between two single triangles of a triangle soup
// (canopy, rock, terrain).
fn edgeClass(r:Relation)->u32{
 if(kind(r)!=0u&&kind(r)!=2u){return 0u;}
 let header=topologyHeader(r);if(header==NONE||r.key.w>=work[header+43u]){return 0u;}
 if(work[header+37u]==${QUAD_RUN_CONTEXT}u){
  let corner=r.key.w%6u;if(corner==0u||corner==1u||corner==4u||corner==5u){return 1u;}
  if(corner!=2u){return 0u;}let quad=r.key.w/6u;
  return (work[header+work[header+42u]+quad/16u]>>((quad%16u)*2u))&3u;
 }
 let value=work[header+work[header+42u]+r.key.w];
 return select(value,value&3u,work[header+37u]==${PACKED_RUN_CONTEXT}u);
}
// THE FACE OWNER (face construction slice 1, compiled-construction-topology.ts
// revision 4): the face word of a path-led source edge, 0 when the edge
// borders no wall face (roof, ridge, eave sheets, foliage, terrain). Every
// silhouette or crease of a face-owned edge is ONE edge of the owner's face
// construction (roundReserve, assemble), as every edge of a box is of its box.

// (the second side of a strip, a frame member's outer side, a return's
// edge): it is counted, reserved, assembled and linked by nothing.
const FACE_MAGIC=${TOPOLOGY_FACE_MAGIC}u;const FACE_HIDDEN=${FACE_HIDDEN}u;
fn faceHidden(word:u32)->bool{return (word&FACE_HIDDEN)!=0u;}
// A FACE edge owns one source passage. Visibility partitions annotate that
// passage; they never become its endpoints, lattice origin or feature ID.
// Several relation records can describe that same source feature/port pair
// after visibility and clipping. They are joined in the existing node arena
// before count/reserve, so the packet is counted and assembled once while

fn faceVisible(id:u32,r:Relation)->bool{
 let span=front(r);var part=partitions[id*2u];
 loop{if(part==0u){break;}let interval=intervals[part-1u];
  if(finiteInterval(r,interval)&&(interval.cause&3u)==0u&&min(interval.hi,span.y)>max(interval.lo,span.x)){return true;}
  part=partitions[p.extent.x*2u+(part-1u)*2u];}
 return false;
}
fn faceEdge(r:Relation)->u32{
 let k=kind(r);if(k!=0u&&k!=2u){return 0u;}
 if(shape(r)!=0u){return 0u;}
 let header=topologyHeader(r);if(header==NONE||work[header+${TOPOLOGY_FACE_MAGIC_WORD}u]!=FACE_MAGIC){return 0u;}
 if(r.key.w>=work[header+${TOPOLOGY_FACE_MAGIC_WORD+2}u]){return 0u;}
 return work[header+work[header+${TOPOLOGY_FACE_MAGIC_WORD+1}u]+r.key.w];
}
const FACE_GROUP_FAMILY=9u;
const FACE_VERTEX_FAMILY=8u;
const FACE_GROUP_VISIBLE=1u;
fn faceGroupKey(r:Relation)->SourceKey{
 let low=min(r.ports.x,r.ports.y);let high=max(r.ports.x,r.ports.y);
 return SourceKey(vec4u(r.key.x,r.key.y,(r.key.z&0x80000000u)|FACE_GROUP_FAMILY,faceEdge(r)),low,vec3u(high,r.ports.z,r.ports.w));
}
// Face groups share the already allocated source-node dictionary. This is a
// producer key, not a second fixed-size private hash. A full dictionary is a
// real source-arena overflow and causes the normal hash growth retry; it never
// falls back to the conflicting per-relation packet.
fn facePacketGroup(id:u32,r:Relation)->u32{
 // Ordinary owner witnesses always use endpoint zero. The unused endpoint
 // one under that tag identifies a face-group witness atomically, so every
 // probing reader reconstructs the actual held key, not its requested key
 // family. Reinterpreting an ordinary junction witness as a face group would
 // falsely merge them whenever their hashes collide.
 return nodeFor(id*2u+1u,OWNER_NODE);
}
// Every edge of a planar VOLUME is a hard line, and so is a contact. Any other
// source is hard edge by edge, and only an authored feature edge is. A facet
// crease on the silhouette of a triangle soup (terrain skyline, crown, rock)
// is one triangle side of a CURVE: its chain is a soft articulated run, one
// unit per native row with the source's shift. Calling it hard sent every
// such outline to the box program, which spells each source edge alone with
// the box units (the stacked ＼ ＿ | staircases on hills and crowns).
// A terrain source (form 3) has no hard edge but its contacts: a ridge, a
// skyline and a shelf edge are slopes, played back from the terrain line
// table one row at a time (compiled-role-grammar.ts terrainInstructions).
fn hardEdge(r:Relation)->bool{
 if(kind(r)==4u){return true;}
 // Compiled outline spans and their joined plane-turn creases are actual
 // straight construction edges (chart flag bit 2), not facet curve samples.
 if((kind(r)==0u||kind(r)==2u)&&(u32(r.sourceChart.w)&4u)!=0u){return true;}
 // A plane-class source's occluding contour or interior turn (kept where
 // its posed mesh bends, incidence) is a soft line of a curved surface.
 let form=shape(r);if(form==1u&&planarVolume(r)&&kind(r)!=1u&&kind(r)!=3u){return true;}
 if(form==2u||form==3u){return false;}
 return edgeClass(r)==1u;
}
// A silhouette edge of a path-led source that is not an authored feature
// edge: one triangle side of a tessellated outline. Every silhouette and
// crease edge of a terrain source is one too.
// The rule holds for EVERY outline class and the interior lines (line repair
// step 1): an occluding contour (kind 1) and an interior turn (kind 3) cross a
// tessellated surface face by face and jog at every face exactly as a facet
// silhouette does; held to the 18 degree straight rule they broke at almost

// 42 interior turns, median 2 px, every one under one row and deselected).
fn tessellated(r:Relation)->bool{
 let form=shape(r);let k=kind(r);
 if(form==3u){return k<=3u;}
 return (form==0u&&((k==0u&&edgeClass(r)!=1u)||k==1u||k==3u))||(form==1u&&(k==1u||k==3u));
}
// A CONTACT IS A JUNCTION TO DRAW. Where exactly one of the two sources of a
// contact is path-led (form 0) and the other is a surface (planar 1 or
// terrain 3), the standing form has a foot on that surface. Returns 0 when
// the relation is not such a contact, 1 when its own source is the standing
// form, 2 when the other source is.
fn footContact(r:Relation)->u32{
 if(kind(r)!=4u){return 0u;}
 let own=sourceClass(r.key.x);let other=sourceClass(r.ports.z&0x1fffffffu);if(own==NONE||other==NONE){return 0u;}
 let a=work[own+1u]&255u;let b=work[other+1u]&255u;
 if(a==0u&&(b==1u||b==3u)){return 1u;}
 if(b==0u&&(a==1u||a==3u)){return 2u;}
 return 0u;
}
// Residency compiled the source-address indices. No frame scans entire
// contact, boundary or section arrays. Candidate context is not a posed weld.
fn topologyGroup(r:Relation,port:u32)->u32{
 if(kind(r)==4u){return r.ports.z^(r.ports.w*0x9e3779b9u);}
 let header=topologyHeader(r);if(header==NONE){return 0u;}
 if(work[header+37u]==${QUAD_RUN_CONTEXT}u){
  if((kind(r)==0u||kind(r)==2u)&&r.key.w<work[header+39u]){
   let corner=r.key.w%6u;if(corner==0u||corner==1u||corner==4u||corner==5u){return 0x20000000u|(r.key.w/6u*5u);}
  }
  return select(0u,port,port<work[header+41u]);
 }
 if(kind(r)==0u||kind(r)==2u){
  let count=work[header+39u];if(r.key.w<count){
   var group=work[header+work[header+38u]+r.key.w];
   if(work[header+37u]==${PACKED_RUN_CONTEXT}u){group=(group&0x60000000u)|((group>>2u)&0x07ffffffu);}
   if(group!=0u){return group;}
  }
 }
 if(port<work[header+41u]){return work[header+work[header+40u]+port];}
 return 0u;
}

// compiled-construction-topology TOPOLOGY_LIMB_MAGIC): the chain label of an
// original vertex port, 0 when the slot has no table or the vertex no limb.
fn limbVertex(r:Relation,port:u32)->u32{
 let header=topologyHeader(r);if(header==NONE||port>=0x80000000u){return 0u;}
 if(work[header+${TOPOLOGY_LIMB_MAGIC_WORD}u]!=${TOPOLOGY_LIMB_MAGIC}u||port>=work[header+${TOPOLOGY_LIMB_MAGIC_WORD+2}u]){return 0u;}
 return work[header+work[header+${TOPOLOGY_LIMB_MAGIC_WORD+1}u]+port];
}
// Convert only the source producers' declared namespaces to the original
// halfedge membership address. key.w itself remains the complete feature
// identity used by correspondence. Contacts, held-far parts, connected L2
// hulls and unknown full-width words deliberately keep the incumbent path.
fn sourceMembershipAddress(r:Relation)->u32{
 let k=kind(r);let w=r.key.w;
 if(k==4u||(u32(r.sourceChart.w)&6u)!=0u){return NONE;}
 if(k==0u||k==2u){return select(w,NONE,(w&0x80000000u)!=0u);}
 if(k==1u){if((w&0x80000000u)!=0u){return w&0x7fffffffu;}if(w<=0x55555555u){return w*3u;}return NONE;}
 if(k==3u){let addressNamespace=w&0xc0000000u;if(addressNamespace==0x40000000u){let face=w&0x3fffffffu;if(face<=0x55555555u){return face*3u;}}else if(addressNamespace==0u&&w<=0x55555555u){return w*3u;}}
 return NONE;
}
// One whole-part program is selected through the decoded original-halfedge
// membership word. The relation's visible interval is never consulted here;
// a run split still carries the same source feature descriptor. vec4 =
// [feature id, role/flags, direction, source labels].
fn sourcePartProgram(r:Relation)->vec4u{
 let header=topologyHeader(r);if(header==NONE||work[header+${TOPOLOGY_PART_MAGIC_WORD}u]!=${TOPOLOGY_PART_MAGIC}u){return vec4u(0);}
 let count=work[header+${TOPOLOGY_PART_MAGIC_WORD+2}u];let memberCount=work[header+${TOPOLOGY_PART_MAGIC_WORD+4}u];let member=sourceMembershipAddress(r);if(member==NONE||member>=memberCount){return vec4u(0);}
 let memberAt=header+work[header+${TOPOLOGY_PART_MAGIC_WORD+3}u];let program=work[memberAt+member];if(program==NONE||program>=count){return vec4u(0);}
 let at=header+work[header+${TOPOLOGY_PART_MAGIC_WORD+1}u]+program*${TOPOLOGY_PART_PROGRAM_WORDS}u;
 // Source correspondence and whole-entry morphology are separate contracts.
 // This packet still owns the feature/role/lineage even when a new retained
 // native-part program is ineligible; the articulation owner gates that
 // replacement before emission. Never disable stable identity transport to
 // reject an incomplete literal-entry producer.
 return vec4u(work[at],work[at+1u],work[at+2u],work[at+3u]);
}
fn page(q:vec4f)->vec4f{
 let w=max(q.w,.00001);let xy=(q.xy/w*vec2f(.5,-.5)+vec2f(.5))*p.rect.zw+p.rect.xy;
 return vec4f(xy,q.z/w,1.0/w);
}
fn front(r:Relation)->vec2f{
 var span=vec2f(0,1);
 if(r.a.w<.00001){span.x=(.00001-r.a.w)/(r.b.w-r.a.w);}
 if(r.b.w<.00001){span.y=(.00001-r.a.w)/(r.b.w-r.a.w);}
 return clamp(span,vec2f(0),vec2f(1));
}
fn finiteInterval(r:Relation,i:Interval)->bool{
 let span=front(r);return min(i.hi,span.y)>max(i.lo,span.x);
}
// key.w is the original source/topology feature address. It is carried on
// every point of the one whole-edge packet; it is never regenerated from a
// clipped interval or from the local native ordinal.
fn originalFeatureId(r:Relation)->u32{return r.key.w;}
fn sourcePoint(r:Relation,t:f32,parameter:f32,visible:bool)->Point{
 let q=page(mix(r.a,r.b,t));
 // Only an original endpoint owns a corner. A visibility/near-plane cut
 // inside the edge is not a new vertex, even if it projects onto another
 // edge's endpoint. Keep the source port separate from its projected pen.
 let endpoint=t==0.0||t==1.0;let port=r.ports[select(0u,1u,t==1.0)];
 return Point(vec4f(q.xyz,parameter),vec4u(sourceChartFlags(visible,mix(r.sourceChart.x,r.sourceChart.y,t),(u32(r.sourceChart.w)&1u)!=0u),bitcast<u32>(q.w),select(NONE,port,endpoint),originalFeatureId(r)));
}
// A face packet keeps its original homogeneous endpoints in the same resident
// point arena as its drawable clipped span. The descriptor pair is tagged by
// the disjoint source-chart sentinel in visible.x and is never a glyph/row
// input; articulation reconstructs its projected anchors for the native
// lattice while the following pair remains the finite draw span. Ports belong
// only to these original endpoints and visible.w retains the full feature ID.
fn wholeEdgeDescriptorPoint(r:Relation,end:u32)->Point{
 let q=select(r.a,r.b,end==1u);let w=select(select(-.00001,.00001,q.w>=0.0),q.w,abs(q.w)>=.00001);
 return Point(q,vec4u(${WHOLE_EDGE_DESCRIPTOR}u,bitcast<u32>(1.0/w),r.ports[end],originalFeatureId(r)));
}
fn hash(key:SourceKey)->u32{
 let v=key.identity;
 var h=(v.x*0x9e3779b9u)^(v.y*0x85ebca6bu)^(v.z*0xc2b2ae35u)^v.w^(key.parameter*0x27d4eb2du)^key.contact.x^(key.contact.y*0x85ebca6bu)^(key.contact.z*0xc2b2ae35u);
 h=(h^(h>>16u))*0x7feb352du;return (h^(h>>15u))&(p.extent.z-1u);
}
fn key(endpoint:u32,mode:u32)->SourceKey{
 let r=relations[endpoint/2u];var port=r.ports[endpoint&1u];
 if(mode==OWNER_NODE&&(endpoint&1u)==1u){return faceGroupKey(r);}
 // One node per ordered pair of contacting source owners: the exact pair, no
 // port, parameter or face, so every contact segment of the pair reaches it.

 // owner, pair group) in the disjoint family 7 (authoredPairKey).
 if(mode==PAIR_NODE){if(authoredPaired(r)){return authoredPairKey(r);}return SourceKey(vec4u(r.key.x,r.key.y,(r.key.z&0x80000000u)|6u,NONE),0u,vec3u(r.ports.z,r.ports.w,0u));}
 // Smooth crossings at different parameters on the same source edge must
 // not acquire a false junction. Keep the canonical source-edge parameter
 // separate from the port: XOR is a hash, not an equality comparison.
 // Quantization only reconciles the adjoining triangles' f32 evaluation
 // of that same crossing; there is no screen-coordinate weld.
 let parameter=r.coordinates[endpoint&1u];
 // Face-owned edges use a dedicated source-vertex incidence family. This
 // keeps their joint index out of ordinary outline continuation while still
 // reusing the existing LINE_NODE linked-list arena.
 var family=kind(r);
 if(mode==LINE_NODE&&faceEdge(r)!=0u){
  // Visibility fragments of one source edge have changing coordinates but
  // the same source endpoint. Keep one incidence node per source vertex;
  // facePacketGroup/least below removes duplicate edge groups at consumption.
  return SourceKey(vec4u(r.key.x,r.key.y,(r.key.z&0x80000000u)|FACE_VERTEX_FAMILY,port),0u,vec3u(0u));
 }
 else if(mode==OWNER_NODE){port=NONE;family=4u;}else if(mode==JUNCTION_NODE){family=5u;}
 // ONE OUTLINE FAMILY (line repair step 1, 2026-11-12). A silhouette and an
 // occluding contour are one outline; which of them a segment is, is a label
 // of that segment (assemble: the run's class), not a reason to start a new
 // line at a source vertex both pass through. Interior turns, creases and
 // contacts keep their own families.
 else if(family==1u){family=0u;}
 // ONE HEIGHT SURFACE ACROSS ITS CHUNKS. A terrain chunk is its own source
 // domain, so the same heightfield vertex on the border of two chunks had two
 // keys and every outline crossing a chunk border ended there (the near rock
 // at spot-frames spans chunks 396 and 397). At an ORIGINAL vertex port of a
 // terrain source the line node is that vertex's posed position itself: the
 // exact bits of its homogeneous clip coordinate, which the visibility pass
 // computes from the same world vertex with the same matrix for both chunks
 // (apps/game/physical-world.ts terrainGeometry: ox + x * cell, the same
 // height sample). Exact equality, no tolerance: it is the source compiler's
 // rest-position weld carried across chunk sources, not a screen weld; two
 // different world points cannot share all four words.

 // crease end whose canonical vertex port is a host-supplied contact CORNER of
 // its owner (surface-relations contactCorner: explicit solver incidence of a
 // touching pair, the same id listed for both owners) is keyed by that point,
 // not by its own (owner, port). Both bodies' edge ends at the shared corner
 // then meet in ONE node, and the existing continuation rules decide: the
 // mutual straightest same-hardness pair within 45 degrees (straightMate), so
 // two stacked crates' collinear sides are one run and their joint edges,
 // leaving the corner the same way, are not joined. An offset crate's
 // corner lies on no contact point and keeps its own node. The terrain seam
 // below is the same rule with the vertex's exact posed bits as the identity.
 let solverPoint=(u32(r.sourceChart.w)>>select(3u,13u,(endpoint&1u)==1u))&1023u;
 if((mode==LINE_NODE||mode==JUNCTION_NODE)&&kind(r)!=4u&&solverPoint!=0u){
  return SourceKey(vec4u(CONTACT_DOMAIN,solverPoint,(r.key.z&0x80000000u)|family,mode),0u,vec3u(0u));
 }
 if(mode==LINE_NODE&&kind(r)!=4u&&port<0x80000000u&&shape(r)==3u){
  let q=bitcast<vec4u>(select(r.a,r.b,(endpoint&1u)==1u));
  return SourceKey(vec4u(SEAM_DOMAIN,q.x,(r.key.z&0x80000000u)|family,q.y),q.z,vec3u(q.w,0u,0u));
 }
 // Contact endpoint = original edge of one surface x face of the other.
 // Keep the complete pair for equality; hashes never establish incidence.
 var contact=vec3u(0);var address=u32(round(parameter*1048576.0));
 if(kind(r)==4u&&mode!=OWNER_NODE){
  contact=vec3u(r.ports.zw,select(u32(r.coordinates.z),r.key.w,(port&0x80000000u)!=0u));
  // Where an original edge of EACH surface passes through the endpoint
  // (the producer sets that end's bit in w and carries the other edge's
  // address as the parameter), the node is that edge pair. The two contact
  // segments that meet there lie on different faces of both surfaces, so a
  // face in the key would leave every such crossing an open end.
  if((u32(r.coordinates.w)&(1u<<(endpoint&1u)))!=0u){address=u32(parameter);contact.z=NONE;}
 }
 return SourceKey(vec4u(r.key.x,r.key.y,(r.key.z&0x80000000u)|family,port),select(address,0u,mode==OWNER_NODE),contact);
}
fn nodeFor(endpoint:u32,mode:u32)->u32{
 let wanted=key(endpoint,mode);var at=hash(wanted);
 // Publish the namespace WITH the endpoint in the single atomic claim.
 // Reinterpreting an occupied junction as an owner (or vice versa) makes
 // unrelated hash collisions look equal and corrupts every downstream run.
 let tag=mode<<30u;
 // Collision resolution in an integer-address dictionary is not a drawing
 // decision. No glyphs, geometry distances, costs or candidate pens occur.
 for(var probe=0u;probe<min(p.extent.z,256u);probe++){
  let old=atomicCompareExchangeWeak(&nodes[at].first,0u,(endpoint+1u)|tag);
  if(old.exchanged){return at;}
  if(old.old_value==0u){continue;}
  if((old.old_value&0xc0000000u)==tag){
   let prior=key((old.old_value&0x3fffffffu)-1u,mode);
   if(all(prior.identity==wanted.identity)&&prior.parameter==wanted.parameter&&all(prior.contact==wanted.contact)){return at;}
  }
  at=(at+1u)&(p.extent.z-1u);
 }
 atomicOr(&nodes[p.extent.z].degree,1u);return NONE;
}
fn ordered(value:f32)->i32{let b=bitcast<i32>(value);return select(b^0x7fffffff,b,b>=0);}
fn real(value:i32)->f32{return bitcast<f32>(select(value^0x7fffffff,value,value>=0));}
// An owner's SOURCE page rows (its node's bounds over every relation of the
// owner, occluded ones included): top and bottom + 16384 in 16 bits each, the
// run's paper.z. Articulation's standing-form rows read exactly these words.
fn ownerRows(low:vec2f,high:vec2f)->u32{return u32(clamp(low.y+16384.0,0.0,65535.0))|(u32(clamp(high.y+16384.0,0.0,65535.0))<<16u);}

// takes whole: an outline (silhouette or occluding contour) of a field-class
// owner whose source rows stand under FAR_FORM_MERGE_PIXELS, the same rows and
// bound articulation's farUnderGlyph reads.
fn standingRetained(r:Relation,low:vec2f,high:vec2f)->bool{
 if(shape(r)!=0u||kind(r)>1u){return false;}
 let rows=ownerRows(low,high);return f32(rows>>16u)-f32(rows&0xffffu)<${FAR_FORM_MERGE_PIXELS};
}
fn linkAt(index:u32)->u32{return p.table.y+index*${LINK_WORDS}u;}
fn other(endpoint:u32)->u32{
 let at=linkAt(endpoint/2u);return work[at+(endpoint&1u)];
}
fn choose(a:u32,b:u32)->u32{
 if(a==NONE){return b;}if(b==NONE){return a;}
 let x=relations[a/2u];let y=relations[b/2u];
 if(x.key.w!=y.key.w){return select(b,a,x.key.w<y.key.w);}
 if(kind(x)==4u&&kind(y)==4u){
  if(x.ports.z!=y.ports.z){return select(b,a,x.ports.z<y.ports.z);}
  if(x.ports.w!=y.ports.w){return select(b,a,x.ports.w<y.ports.w);}
  if(x.coordinates.z!=y.coordinates.z){return select(b,a,x.coordinates.z<y.coordinates.z);}
 }
 // Shared junctions can contain different feature families with the same
 // numerical source address. Their order must not depend on which atomic
 // relation emission arrived first this frame.
 if(kind(x)!=kind(y)){return select(b,a,kind(x)<kind(y));}
 return select(b,a,endpointLess(a,b));
}
// A relation's index is the order of this frame's atomic emissions, which
// changes between frames of a still view (crawl, 2026-11-11). Ties between
// endpoints are settled by their source content; the index decides only
// between records that are identical in every field.
fn endpointLess(a:u32,b:u32)->bool{
 let x=relations[a/2u];let y=relations[b/2u];
 for(var i=0u;i<4u;i++){if(x.key[i]!=y.key[i]){return x.key[i]<y.key[i];}}
 for(var i=0u;i<4u;i++){if(x.ports[i]!=y.ports[i]){return x.ports[i]<y.ports[i];}}
 if((a&1u)!=(b&1u)){return (a&1u)<(b&1u);}
 let xa=bitcast<vec4u>(x.a);let ya=bitcast<vec4u>(y.a);let xb=bitcast<vec4u>(x.b);let yb=bitcast<vec4u>(y.b);let xc=bitcast<vec4u>(x.coordinates);let yc=bitcast<vec4u>(y.coordinates);
 for(var i=0u;i<4u;i++){if(xa[i]!=ya[i]){return xa[i]<ya[i];}}
 for(var i=0u;i<4u;i++){if(xb[i]!=yb[i]){return xb[i]<yb[i];}}
 for(var i=0u;i<4u;i++){if(xc[i]!=yc[i]){return xc[i]<yc[i];}}
 return a<b;
}
// A tessellated silhouette follows source mesh edges and staircases along
// them: consecutive edges jog, and reverse, by less than the drawing can
// show. Its direction at a vertex is therefore also read along the chain,
// until the reach is two native rows (34 page px) or eight edges. Source
// incidence only, and the same walk from either relation that meets at the
// vertex, so both reach the same verdict.
fn endpointPage(endpoint:u32)->vec2f{
 let q=relations[endpoint/2u];let span=front(q);
 return page(mix(q.a,q.b,select(span.x,span.y,(endpoint&1u)==1u))).xy;
}
fn chord(origin:vec2f,endpoint:u32)->vec2f{
 var at=endpoint;var reach=endpointPage(at^1u)-origin;
 for(var step=0u;step<8u;step++){
  if(dot(reach,reach)>=1156.0){break;}
  let far=at^1u;let q=relations[far/2u];
  if(select(q.a.w,q.b.w,(far&1u)==1u)<.00001){break;}
  let node=nodeFor(far,LINE_NODE);if(node==NONE||atomicLoad(&nodes[node].degree)!=2u){break;}
  let first=atomicLoad(&nodes[node].second)-1u;
  let second=work[linkAt(first/2u)+10u+(first&1u)]-1u;
  at=select(first,second,far==first);reach=endpointPage(at^1u)-origin;
 }
 return reach;
}
fn junctionRepresentative(node:u32,endpoint:u32){
 // Source-address minimum, not the first atomic claimant. The dictionary's
 // first field remains its key witness; this independent reduction owns
 // which incident relation emits the ONE shared anchor.
 var before=atomicLoad(&nodes[node].least);
 loop{let after=choose(before,endpoint);if(after==before){return;}
  let claim=atomicCompareExchangeWeak(&nodes[node].least,before,after);if(claim.exchanged){return;}before=claim.old_value;
 }
}

// TASK,00:09UTC"). surface-relations authoredPathAt emits each authored path
// segment with address 0xb0000000 | role<<26 (1 pair A, 2 pair B) |
// group<<13 (pair) | phase (segment index from A's start; reversed B already
// runs from A's start) and ports 0xd8000000|end-port (authored path END) or
// 0xd0000000|site. One PAIR_NODE per exact (domain, owner, pair group), in a
// disjoint identity family (7; foot contacts use 6), counts both rails:
// second = nA | nB<<16, maxX/maxY = highest phase of A/B, degree bits 1/2/4/8
// = A head-end, A tail-end, B head-end, B tail-end present, 16 = a segment cut
// by the near plane, 0x80000000 = reserved; minX = the reserved point base.
// A COMPLETE pair (every segment of both rails present once, both ends of
// each rail present, no near-plane cut, nA == nB so phases align) reserves ONE
// run (roundReserve) holding both rails in phase order; its relations then
// neither link nor assemble per edge. Any other case keeps the incumbent.
const AUTHORED_PAIR_FLAG=0x1000000u;
const AUTHORED_PAIR_FAMILY=7u;
const AUTHORED_PAIR_PHASE_FAMILY=8u;
const AUTHORED_PAIR_INVALID=0x20u;
const AUTHORED_PAIR_ATTEMPTED=0x40u;
fn authoredPaired(r:Relation)->bool{
 let role=(r.key.w>>26u)&3u;return kind(r)==0u&&(r.key.w&0xf0000000u)==0xb0000000u&&(role==1u||role==2u);
}
fn authoredPairKey(r:Relation)->SourceKey{
 return SourceKey(vec4u(r.key.x,r.key.y,(r.key.z&0x80000000u)|AUTHORED_PAIR_FAMILY,(r.key.w>>13u)&8191u),0u,vec3u(0u));
}
// Every phase gets its own node in the existing source-node arena.  The pair
// node remains the aggregate; phase nodes carry the exact two ports for one
// rail/phase, so adjacent phases can validate their source chain without a
// pair-wide scan or a second storage binding.
fn authoredPairPhaseKey(r:Relation,role:u32,phase:u32)->SourceKey{
 let group=(r.key.w>>13u)&8191u;
 return SourceKey(vec4u(r.key.x,r.key.y,(r.key.z&0x80000000u)|AUTHORED_PAIR_PHASE_FAMILY,(group<<16u)|(role<<14u)|phase),0u,vec3u(0u));
}
fn authoredPairPhaseKeyEqual(a:SourceKey,b:SourceKey)->bool{
 return a.identity[0]==b.identity[0]&&a.identity[1]==b.identity[1]&&a.identity[2]==b.identity[2]&&a.identity[3]==b.identity[3]&&a.parameter==b.parameter&&a.contact[0]==b.contact[0]&&a.contact[1]==b.contact[1]&&a.contact[2]==b.contact[2];
}
fn authoredPairPhaseLookup(r:Relation,role:u32,phase:u32)->u32{
 if(phase>8191u){return NONE;}
 let wanted=authoredPairPhaseKey(r,role,phase);var at=hash(wanted);let tag=PAIR_NODE<<30u;
 for(var probe=0u;probe<min(p.extent.z,256u);probe++){
  let old=atomicLoad(&nodes[at].first);
  if(old==0u){return NONE;}
  if((old&0xc0000000u)==tag){let prior=relations[((old&0x3fffffffu)-1u)/2u];if(authoredPairPhaseKeyEqual(authoredPairPhaseKey(prior,(prior.key.w>>26u)&3u,prior.key.w&8191u),wanted)){return at;}}
  at=(at+1u)&(p.extent.z-1u);
 }
 return NONE;
}
fn authoredPairPhaseNode(id:u32,r:Relation)->u32{
 let wanted=authoredPairPhaseKey(r,(r.key.w>>26u)&3u,r.key.w&8191u);var at=hash(wanted);let tag=PAIR_NODE<<30u;
 for(var probe=0u;probe<min(p.extent.z,256u);probe++){
  let old=atomicCompareExchangeWeak(&nodes[at].first,0u,(id*2u+1u)|tag);
  if(old.exchanged){return at;}
  if(old.old_value==0u){continue;}
  if((old.old_value&0xc0000000u)==tag){let prior=relations[((old.old_value&0x3fffffffu)-1u)/2u];if(authoredPairPhaseKeyEqual(authoredPairPhaseKey(prior,(prior.key.w>>26u)&3u,prior.key.w&8191u),wanted)){return at;}}
  at=(at+1u)&(p.extent.z-1u);
 }
 return NONE;
}
fn authoredPairInvalid(id:u32){let pair=nodeFor(id*2u,PAIR_NODE);if(pair!=NONE){atomicOr(&nodes[pair].degree,AUTHORED_PAIR_INVALID);}}
fn authoredPairPhaseReady(node:u32)->bool{return node!=NONE&&atomicLoad(&nodes[node].maxX)==1;}
fn authoredPairPhaseEnd(port:u32)->bool{return (port&0xf8000000u)==0xd8000000u;}
fn authoredPairPhaseCheck(id:u32,r:Relation,node:u32){
 let role=(r.key.w>>26u)&3u;let phase=r.key.w&8191u;
 if((phase==0u)!=authoredPairPhaseEnd(r.ports.x)){authoredPairInvalid(id);}
 if(phase!=0u&&authoredPairPhaseEnd(r.ports.x)){authoredPairInvalid(id);}
 if(role!=1u&&role!=2u){authoredPairInvalid(id);return;}
 if(phase>0u){let previous=authoredPairPhaseLookup(r,role,phase-1u);if(authoredPairPhaseReady(previous)){
   let previousTail=atomicLoad(&nodes[previous].degree)-1u;
   if(previousTail!=r.ports.x||authoredPairPhaseEnd(previousTail)||authoredPairPhaseEnd(r.ports.x)){authoredPairInvalid(id);}
  }}
 if(phase<8191u){let next=authoredPairPhaseLookup(r,role,phase+1u);if(authoredPairPhaseReady(next)){
   let nextHead=atomicLoad(&nodes[next].second)-1u;
   if(r.ports.y!=nextHead||authoredPairPhaseEnd(r.ports.y)||authoredPairPhaseEnd(nextHead)){authoredPairInvalid(id);}
  }}
}
fn authoredPairCount(id:u32,r:Relation){
 let node=nodeFor(id*2u,PAIR_NODE);if(node==NONE){return;}
 let role=(r.key.w>>26u)&3u;let phase=i32(r.key.w&8191u);
 let old=atomicAdd(&nodes[node].second,select(1u,0x10000u,role==2u));
 if(select(old&0xffffu,(old>>16u)&0xffffu,role==2u)==0xffffu){atomicOr(&nodes[node].degree,AUTHORED_PAIR_INVALID);}
 if(role==1u){atomicMax(&nodes[node].maxX,phase);}else{atomicMax(&nodes[node].maxY,phase);}
 var bits=0u;
 if((r.ports.x&0xf8000000u)==0xd8000000u){bits|=select(1u,4u,role==2u);}
 if((r.ports.y&0xf8000000u)==0xd8000000u){bits|=select(2u,8u,role==2u);}
 if(r.a.w<.00001||r.b.w<.00001){bits|=16u;}
 atomicOr(&nodes[node].degree,bits);junctionRepresentative(node,id*2u);
 let phaseNode=authoredPairPhaseNode(id,r);if(phaseNode==NONE){atomicOr(&nodes[node].degree,AUTHORED_PAIR_INVALID);return;}
 let phaseClaim=(id*2u+1u)|(PAIR_NODE<<30u);if(atomicLoad(&nodes[phaseNode].first)!=phaseClaim){atomicOr(&nodes[node].degree,AUTHORED_PAIR_INVALID);return;}
 if(atomicLoad(&nodes[phaseNode].second)!=0u||atomicLoad(&nodes[phaseNode].maxX)!=bitcast<i32>(0x80000000u)){atomicOr(&nodes[node].degree,AUTHORED_PAIR_INVALID);return;}
 atomicStore(&nodes[phaseNode].second,r.ports.x+1u);atomicStore(&nodes[phaseNode].degree,r.ports.y+1u);atomicStore(&nodes[phaseNode].maxX,1);
}
fn authoredPairComplete(node:u32)->bool{
 let d=atomicLoad(&nodes[node].degree);let counts=atomicLoad(&nodes[node].second);let a=counts&0xffffu;let b=counts>>16u;
 if((d&(31u|AUTHORED_PAIR_INVALID))!=15u||a==0u||a!=b||i32(a)!=atomicLoad(&nodes[node].maxX)+1||i32(b)!=atomicLoad(&nodes[node].maxY)+1){return false;}
 let endpoint=(atomicLoad(&nodes[node].first)&0x3fffffffu)-1u;let r=relations[endpoint/2u];
 let firstA=authoredPairPhaseLookup(r,1u,0u);let firstB=authoredPairPhaseLookup(r,2u,0u);let lastA=authoredPairPhaseLookup(r,1u,u32(atomicLoad(&nodes[node].maxX)));let lastB=authoredPairPhaseLookup(r,2u,u32(atomicLoad(&nodes[node].maxY)));
 if(!authoredPairPhaseReady(firstA)||!authoredPairPhaseReady(firstB)||!authoredPairPhaseReady(lastA)||!authoredPairPhaseReady(lastB)){return false;}
 return authoredPairPhaseEnd(atomicLoad(&nodes[lastA].degree)-1u)&&authoredPairPhaseEnd(atomicLoad(&nodes[lastB].degree)-1u);
}
// The relation at this endpoint belongs to a complete authored pair (decided
// once incidence has counted every relation).
fn authoredPairOwned(endpoint:u32)->bool{
 if(!authoredPaired(relations[endpoint/2u])){return false;}
 let node=nodeFor(endpoint,PAIR_NODE);return node!=NONE&&(atomicLoad(&nodes[node].degree)&0x80000000u)!=0u&&authoredPairComplete(node);
}
fn authoredPairReserve(id:u32,r:Relation){
 let node=nodeFor(id*2u,PAIR_NODE);
 if(node==NONE||atomicLoad(&nodes[node].least)!=id*2u||!authoredPairComplete(node)){return;}
 let claimed=atomicOr(&nodes[node].degree,AUTHORED_PAIR_ATTEMPTED);if((claimed&AUTHORED_PAIR_ATTEMPTED)!=0u){return;}
 let n=atomicLoad(&nodes[node].second)&0xffffu;let total=4u*n;
 let base=atomicAdd(&nodes[p.extent.z].first,total);let slot=atomicAdd(&nodes[p.extent.z].second,1u);
 if(base+total>p.extent.y){atomicOr(&nodes[p.extent.z].degree,2u);}
 if(slot>=p.sourceExtent.z){atomicOr(&nodes[p.extent.z].degree,4u);}
 if(base+total>p.extent.y||slot>=p.sourceExtent.z){return;}
 var size=1u;let owner=work[linkAt(id)+6u];
 if(owner!=NONE){let rows=real(atomicLoad(&nodes[owner].maxY))-real(atomicLoad(&nodes[owner].minY));if(rows>=0.0){size=select(select(2u,1u,rows<119.0),0u,rows<51.0);}}
 let group=(r.key.w>>13u)&8191u;
 let identity=(r.key.x*0x9e3779b9u)^(r.key.y*0x85ebca6bu)^(group*0xc2b2ae35u)^0x50000000u;
 runs[slot]=Run(vec4u(base,total,AUTHORED_PAIR_FLAG,size),vec4u(identity,r.key.x|(r.key.z&0x80000000u),r.key.y,identity),vec4u(0u),vec4u(0u),vec4u(0u));
 atomicStore(&nodes[node].minX,bitcast<i32>(base));atomicOr(&nodes[node].degree,0x80000000u);
}
// Fill the reserved run: rail A segment s at base+2s, rail B segment s at
// base+2n+2s; each segment writes its two ORIGINAL endpoints (ports, feature
// address, projected position, parameter = source phase + t).
fn authoredPairFill(r:Relation,node:u32){
 let base=bitcast<u32>(atomicLoad(&nodes[node].minX));let n=atomicLoad(&nodes[node].second)&0xffffu;
 let role=(r.key.w>>26u)&3u;let phase=r.key.w&8191u;
 let at=base+select(2u*n,0u,role==1u)+2u*phase;
 points[at]=sourcePoint(r,0.0,f32(phase),true);points[at+1u]=sourcePoint(r,1.0,f32(phase)+1.0,false);
}
// Incidence only publishes phase-node records. Validation is a separate
// dispatch so every phase node's ports are visible before a neighbor check;
// no relaxed cross-workgroup read is used as a readiness protocol.
@compute @workgroup_size(64) fn authoredPairValidate(@builtin(global_invocation_id) gid:vec3u){
 let id=gid.x+gid.y*p.table.z*64u;if(id>=count()||sourceStatus[1]!=0u){return;}
 let r=relations[id];if(!authoredPaired(r)){return;}
 let node=authoredPairPhaseLookup(r,(r.key.w>>26u)&3u,r.key.w&8191u);if(authoredPairPhaseReady(node)){authoredPairPhaseCheck(id,r,node);}
}
// Reservation is deliberately before connect. Valid pairs suppress the
// incumbent only after their whole run has actually been reserved; capacity
// overflow leaves ATTEMPTED without the reserved bit, so connect and assemble
// retain the complete per-edge fallback.
@compute @workgroup_size(64) fn authoredPairReserveStage(@builtin(global_invocation_id) gid:vec3u){
 let id=gid.x+gid.y*p.table.z*64u;if(id>=count()||sourceStatus[1]!=0u){return;}
 let at=linkAt(id);if(work[at+7u]==0u){return;}let r=relations[id];if(authoredPaired(r)){authoredPairReserve(id,r);}
}
@compute @workgroup_size(64) fn initialize(@builtin(global_invocation_id) gid:vec3u){
 let id=gid.x+gid.y*p.table.z*64u;
 if(id<p.extent.z){atomicStore(&nodes[id].first,0u);atomicStore(&nodes[id].second,0u);atomicStore(&nodes[id].degree,0u);atomicStore(&nodes[id].least,NONE);atomicStore(&nodes[id].minX,0x7fffffff);atomicStore(&nodes[id].minY,0x7fffffff);atomicStore(&nodes[id].maxX,bitcast<i32>(0x80000000u));atomicStore(&nodes[id].maxY,bitcast<i32>(0x80000000u));}
 if(id<p.sourceExtent.z){runs[id]=Run(vec4u(0),vec4u(0),vec4u(0),vec4u(0),vec4u(0));}
}
@compute @workgroup_size(64) fn incidence(@builtin(global_invocation_id) gid:vec3u){
 let id=gid.x+gid.y*p.table.z*64u;if(id>=count()||sourceStatus[1]!=0u){return;}
 let r=relations[id];let at=linkAt(id);let k=kind(r);let form=shape(r);
 for(var i=0u;i<${LINK_WORDS}u;i++){work[at+i]=NONE;}work[at+7u]=0u;
 // A plane has no radial-curvature feature. A convex round has ONE mesh
 // boundary; its normal-zero alias is not a second interior silhouette.
 // A plane-class source lifted into a curved surface by its material
 // (snow-trails: a flat grid at rest, hills when posed) keeps the radial
 // features where its POSED mesh bends (surface-relations.ts bends, ports.w
 // bit 0x40000000); a flat plane or box face has none (lines4).
 if((form==1u&&k!=0u&&k!=2u&&k!=4u&&!((k==1u||k==3u)&&(r.ports.w&0x40000000u)!=0u))||(form==2u&&k==1u)){return;}
 // A camera-plane cut is visibility, not permission to drop the visible
 // half of a source edge. Original ports/parameters remain unchanged.
 if(r.a.w<.00001&&r.b.w<.00001){return;}
 work[at+7u]=1u;

 if(authoredPaired(r)){authoredPairCount(id,r);}
 let owner=nodeFor(id*2u,OWNER_NODE);work[at+6u]=owner;
 if(owner!=NONE){
  // Terrain line nodes join across source owners. Reserve one source-context
  // record per owner before assembly, so every joined interval keeps its own
  // contributor without splitting the construction at a chunk border.
  if(form==3u&&k!=4u){junctionRepresentative(owner,id*2u);}
  // Owners aggregate bounds only. Boundary traversal follows original port
  // incidence below; an atomic owner-wide list has neither cycle identity
  // nor a stable source order and must not own rounded correspondence.
  let span=front(r);let a=page(mix(r.a,r.b,span.x)).xy;let b=page(mix(r.a,r.b,span.y)).xy;
  atomicMin(&nodes[owner].minX,ordered(min(a.x,b.x)));atomicMin(&nodes[owner].minY,ordered(min(a.y,b.y)));
  atomicMax(&nodes[owner].maxX,ordered(max(a.x,b.x)));atomicMax(&nodes[owner].maxY,ordered(max(a.y,b.y)));
  // ONE rounded form has ONE whole-form run from its COMPLETE source
  // boundary, exactly as the compiled construction did: a source seam,
  // camera cut or occlusion that divides the boundary into open walks does
  // not turn a sphere into soft path words. Count its finite interval
  // points on the owner now; roundReserve allocates, assemble fills.
  // A planar edge is represented by ONE endpoint pair in its whole-form
  // packet. Visibility partitions annotate that edge; they are not more
  // edges. Count the same footprint that assemble writes, so fragmented
  // self-visibility cannot change the form's producer or exhaust its packet.
  // A face-owned edge counts as one box edge (faceEdge); a compiled-out one
  // counts nothing.
  let faceWord=faceEdge(r);let faced=faceWord!=0u&&!faceHidden(faceWord);
  if(faced){
   let group=facePacketGroup(id,r);work[at+8u]=group;
   if(group!=NONE){
    junctionRepresentative(group,id*2u);
    if(faceVisible(id,r)){atomicOr(&nodes[group].degree,FACE_GROUP_VISIBLE);}
   }
  }

  // never into the owner's round/box/face packet.
  if(!authoredPaired(r)&&!faceHidden(faceWord)&&((form==2u&&k==0u)||(form==1u&&(k==0u||k==2u))||faced)){
   var n=0u;var visible=false;var part=partitions[id*2u];
   loop{if(part==0u){break;}let interval=intervals[part-1u];if(finiteInterval(r,interval)){n+=2u;visible=visible||(interval.cause&3u)==0u;}part=partitions[p.extent.x*2u+(part-1u)*2u];}
   // A box edge is a source construction even when its current projected
   // visibility is zero or only a sliver. Reserve the complete original
   // packet; visibility is carried by point flags and clips emitted units
   // later. A face packet has its descriptor pair plus one draw pair.
   if(faced){n=0u;}
   else if(form==1u){n=2u;}
   // The owner's representative is the least endpoint by SOURCE CONTENT
   // (choose, as a junction's), never the lowest relation index: the index is
   // the order this frame's relations were appended, so fusing or reordering
   // the relation dispatches changed which relation reserved the whole-form run.
   if(n>0u){atomicAdd(&nodes[owner].second,n);junctionRepresentative(owner,id*2u);if(visible||form==1u){atomicOr(&nodes[owner].degree,1u);}}
  }
 }
 // The foot of a standing form: the VISIBLE extent of every contact segment
 // of one owner pair, and the pair's one representative by source address.
 if(footContact(r)!=0u){
  var lowest=vec2f(3.402823e38);var highest=vec2f(-3.402823e38);var part=partitions[id*2u];
  loop{if(part==0u){break;}let interval=intervals[part-1u];
   if(finiteInterval(r,interval)&&(interval.cause&3u)==0u){
    let span=front(r);let a=page(mix(r.a,r.b,max(interval.lo,span.x))).xy;let b=page(mix(r.a,r.b,min(interval.hi,span.y))).xy;
    lowest=min(lowest,min(a,b));highest=max(highest,max(a,b));
   }
   part=partitions[p.extent.x*2u+(part-1u)*2u];}
  if(highest.x>=lowest.x){
   let pair=nodeFor(id*2u,PAIR_NODE);
   if(pair!=NONE){
    atomicMin(&nodes[pair].minX,ordered(lowest.x));atomicMin(&nodes[pair].minY,ordered(lowest.y));
    atomicMax(&nodes[pair].maxX,ordered(highest.x));atomicMax(&nodes[pair].maxY,ordered(highest.y));
    junctionRepresentative(pair,id*2u);
   }
  }
 }
 for(var end=0u;end<2u;end++){
  if(select(r.a.w,r.b.w,end==1u)<.00001){continue;}
  let endpoint=id*2u+end;let node=nodeFor(endpoint,LINE_NODE);if(node==NONE){continue;}
  // Complete SOURCE-junction incidence, not a screen-neighbour weld.
  // The list is immutable after this dispatch and follows the run links
  // through pointer doubling, so a fork never scans the whole owner.
  work[at+10u+end]=atomicExchange(&nodes[node].second,endpoint+1u);
  atomicAdd(&nodes[node].degree,1u);
  // The face-family line node is only the source-vertex index. Its minimum
  // endpoint is the single producer representative consumed by anchors().
  if(faceEdge(r)!=0u&&!faceHidden(faceEdge(r))){junctionRepresentative(node,endpoint);}
  // A silhouette, crease and interior turn retain separate continuation
  // rules, but their ORIGINAL common vertex owns one real junction. Do not
  // manufacture a join from coincident screen coordinates or edge crossings.
  if(form==0u&&k!=4u&&r.ports[end]<0x80000000u){
   let junction=nodeFor(endpoint,JUNCTION_NODE);if(junction==NONE){continue;}
   work[at+12u+end]=atomicExchange(&nodes[junction].second,endpoint+1u);
   atomicAdd(&nodes[junction].degree,1u);
   junctionRepresentative(junction,endpoint);
  }
 }
}
// A LINE CONTINUES THROUGH A VERTEX WHERE SEVERAL EDGES MEET. A fold or ridge
// of a finely tessellated surface (a cliff of thin triangles, a rock, a far
// ridge) passes vertices where a third and fourth crease meet it; stopping
// there cut every fold into pieces under three rows, which selection then
// dropped (view 06 cliff: the folds of the underdrawing missing from the text;

// such a vertex each edge end pairs with the end that continues it most
// straightly, of the same hardness, when that end chooses it back (mutual, so
// the links stay one consistent chain) and the turn is 45 degrees or less.
// The other edges start their own lines. Fixed rule, nothing is scored.
fn leaving(e:u32)->vec2f{let n=relations[e/2u];return select(page(n.b).xy-page(n.a).xy,page(n.a).xy-page(n.b).xy,(e&1u)==1u);}
fn straightMate(e:u32,node:u32)->u32{
 let d=leaving(e);let ld=length(d);if(ld<1e-5){return NONE;}
 let hard=hardEdge(relations[e/2u]);
 var best=NONE;var score=.70710678;var f=atomicLoad(&nodes[node].second)-1u;
 for(var k=0u;k<16u&&f!=NONE;k++){
  if(f!=e&&hardEdge(relations[f/2u])==hard&&faceEdge(relations[f/2u])==0u){
   let o=leaving(f);let lo=length(o);
   if(lo>1e-5){let c=-dot(d,o)/(ld*lo);if(c>score||(c==score&&best!=NONE&&endpointLess(f,best))){score=c;best=f;}}
  }
  f=work[linkAt(f/2u)+10u+(f&1u)]-1u;
 }
 return best;
}
@compute @workgroup_size(64) fn connect(@builtin(global_invocation_id) gid:vec3u){
 let id=gid.x+gid.y*p.table.z*64u;if(id>=count()){return;}
 let at=linkAt(id);if(work[at+7u]==0u){return;}let r=relations[id];
 let span=front(r);let a=page(mix(r.a,r.b,span.x)).xy;let b=page(mix(r.a,r.b,span.y)).xy;
 for(var end=0u;end<2u;end++){
  let endpoint=id*2u+end;let node=nodeFor(endpoint,LINE_NODE);var continuation=NONE;
  let roundBoundary=shape(r)==2u&&kind(r)==0u;
  if(select(r.a.w,r.b.w,end==1u)>=.00001&&node!=NONE&&atomicLoad(&nodes[node].degree)==2u&&(roundBoundary||critical(r,end)==0u)){
   let first=atomicLoad(&nodes[node].second)-1u;
   let second=work[linkAt(first/2u)+10u+(first&1u)]-1u;
   let next=select(first,second,endpoint==first);let n=relations[next/2u];
   let incoming=select(a-b,b-a,end==1u);let outgoing=select(page(n.b).xy-page(n.a).xy,page(n.a).xy-page(n.b).xy,(next&1u)==1u);
   let denominator=length(incoming)*length(outgoing);
   let extrema=any(incoming*outgoing<vec2f(0));
   // An interior turn (a cliff fold) bends at every facet; it continues through
   // bends up to 30 degrees, so one fold is one run, not pieces under three
   // rows that selection drops (crop-review-20261110 crop-6 A2, A3).
   // A soft OUTLINE (silhouette or occluding contour of a tessellated
   // surface) continues through 30 degrees as well: a rock's outline turns a
   // few tens of degrees at each facet and is one stroke, whose direction
   // segments articulation spells (line repair step 1).
   let outline=kind(r)<=1u&&kind(n)<=1u&&tessellated(r)&&tessellated(n);
   let cosine=select(select(.95,.8660254,(kind(r)==3u&&kind(n)==3u)||outline),.99999,hardEdge(r)&&hardEdge(n));
   // A rounded boundary is a complete ORIGINAL connected cycle. Screen
   // extrema and shoulders change its articulation, not its connectivity.
   // Other classes still split runs at real turns/critical source ports.
   // A hard path retains both legs of a degree-two source turn. It is not
   // a box: articulation keeps open-path and enclosed-volume semantics apart.
   // Two facet edges of a tessellated outline are not such a path: their
   // turn is judged below.
   var hardBoundary=false;
   if(shape(r)==0u&&hardEdge(r)&&hardEdge(n)&&!(tessellated(r)&&tessellated(n))){
    let junction=nodeFor(endpoint,JUNCTION_NODE);
    hardBoundary=junction!=NONE&&atomicLoad(&nodes[junction].degree)==2u;
   }
   var aligned=hardEdge(r)==hardEdge(n)&&denominator>1e-10&&!extrema&&dot(incoming,outgoing)>=cosine*denominator;
   // A run is bounded by a turn of the outline, not by the triangles that
   // tessellate it. Where two tessellated silhouette edges meet and one is
   // shorter than two native rows, the outline continues unless it turns
   // by more than 45 degrees both edge to edge and chord to chord (a
   // corner), or it reverses on a screen axis edge to edge with the chords
   // leaning opposite ways by at least one in ten each (a screen extremum).
   // A staircase jog turns edge to edge but not on the chords; a small
   // round arc turns on neither until its extremum. Soft and facet edges of
   // one outline join; the run is hard only when every edge of it is.
   if(!aligned&&tessellated(r)&&tessellated(n)){
    let here=select(a,b,end==1u);let leg=endpointPage(next^1u)-endpointPage(next);
    if(min(dot(incoming,incoming),dot(leg,leg))<1156.0){
     let back=-chord(here,endpoint);let ahead=chord(here,next);let scale=length(back)*length(ahead);
     if(scale>1e-10){
      let i=normalize(back);let o=normalize(ahead);let flip=incoming*leg<vec2f(0);
      let reversed=(flip.x&&i.x*o.x<0.0&&min(abs(i.x),abs(o.x))>=.1)||(flip.y&&i.y*o.y<0.0&&min(abs(i.y),abs(o.y))>=.1);
      aligned=!reversed&&(dot(i,o)>=.70710678||dot(incoming,leg)>=.70710678*length(incoming)*length(leg));
     }
    }
   }
   if(roundBoundary||hardBoundary||(kind(r)==4u&&kind(n)==4u)||aligned){continuation=next;}
  }else if(select(r.a.w,r.b.w,end==1u)>=.00001&&node!=NONE&&atomicLoad(&nodes[node].degree)>=3u&&atomicLoad(&nodes[node].degree)<=16u){
   let mate=straightMate(endpoint,node);
   if(mate!=NONE&&straightMate(mate,node)==endpoint){continuation=mate;}
  }
  // A face-owned edge is one edge of its owner's face construction: it
  // continues into nothing, and nothing continues into it (faceEdge).
  if(continuation!=NONE&&(faceEdge(r)!=0u||faceEdge(relations[continuation/2u])!=0u)){continuation=NONE;}

  // segments neither continue into, nor receive, any per-edge line.
  if(continuation!=NONE&&(authoredPairOwned(endpoint)||authoredPairOwned(continuation))){continuation=NONE;}
  work[at+end]=continuation;
  work[at+2u+end]=select(continuation^1u,endpoint,continuation==NONE);
  work[at+4u+end]=endpoint;
 }
}
// FACE PACKET RESERVE. Incidence has published all source feature/port groups
// and their aggregate visibility. This separate dispatch picks one stable
// representative per group and adds exactly its descriptor+draw footprint to
// the owning construction. Hidden-first/later-live therefore remains one
// packet, while the live bit is the union of every relation in that group.
@compute @workgroup_size(64) fn facePacketReserve(@builtin(global_invocation_id) gid:vec3u){
 let id=gid.x+gid.y*p.table.z*64u;if(id>=count()||sourceStatus[1]!=0u){return;}
 let at=linkAt(id);if(work[at+7u]==0u){return;}let r=relations[id];let faceWord=faceEdge(r);
 if(faceWord==0u||faceHidden(faceWord)){return;}
 let group=work[at+8u];let owner=work[at+6u];if(group==NONE||owner==NONE){return;}
 if(atomicLoad(&nodes[group].least)!=id*2u){return;}
 atomicAdd(&nodes[owner].second,4u);
 junctionRepresentative(owner,id*2u);
 if((atomicLoad(&nodes[group].degree)&FACE_GROUP_VISIBLE)!=0u){atomicOr(&nodes[owner].degree,1u);}
}

// Doubled source pointers find a run's terminal endpoints/component identity
// in logarithmic dispatches. No serial CPU walk or world-size readback.
// CONVERGED ROUNDS RETURN (frame cost, 2026-11-13). Round r (p.sourceExtent.w,
// 1-based) marks bit r-1 of the status node's spare minX word when a link's
// jump or root words change. When round r-1 changed nothing, those four
// words match in both tables, so this and every later round are no-ops.
fn jumpWord(id:u32,field:u32,alternate:bool)->u32{
 if(alternate){return work[p.extent.w+id*${JUMP_WORDS}u+field];}
 return work[linkAt(id)+2u+field];
}
fn writeJump(id:u32,field:u32,value:u32,alternate:bool){
 if(alternate){work[p.extent.w+id*${JUMP_WORDS}u+field]=value;}
 else{work[linkAt(id)+2u+field]=value;}
}
@compute @workgroup_size(64) fn jump(@builtin(global_invocation_id) gid:vec3u){
 let round=p.sourceExtent.w;
 if(round>1u&&(bitcast<u32>(atomicLoad(&nodes[p.extent.z].minX))&(1u<<((round-2u)&31u)))==0u){return;}
 let id=gid.x+gid.y*p.table.z*64u;if(id>=count()){return;}
 let source=linkAt(id);let alternate=(round&1u)==0u;
 for(var k=0u;k<${JUMP_WORDS}u;k++){writeJump(id,k,jumpWord(id,k,alternate),!alternate);}
 if(work[source+7u]==0u){return;}
 var changed=false;
 for(var end=0u;end<2u;end++){
  let next=jumpWord(id,end,alternate);
  let jumped=jumpWord(next/2u,next&1u,alternate);let root=choose(jumpWord(id,2u+end,alternate),jumpWord(next/2u,2u+(next&1u),alternate));
  changed=changed||jumped!=jumpWord(id,end,alternate)||root!=jumpWord(id,2u+end,alternate);
  writeJump(id,end,jumped,!alternate);
  writeJump(id,2u+end,root,!alternate);
 }
 if(changed&&round>0u){atomicOr(&nodes[p.extent.z].minX,bitcast<i32>(1u<<((round-1u)&31u)));}
}
// An odd number of rounds leaves the result in the compact side. Restore
// only its four words before the existing anchor/assembly readers run.
@compute @workgroup_size(64) fn finishJumps(@builtin(global_invocation_id) gid:vec3u){
 let id=gid.x+gid.y*p.table.z*64u;if(id>=count()){return;}
 for(var k=0u;k<${JUMP_WORDS}u;k++){writeJump(id,k,jumpWord(id,k,true),false);}
}
fn appendInterval(r:Relation,i:Interval,reverse:bool,ordinal:u32,total:u32,base:u32)->u32{
 if(!finiteInterval(r,i)){return base;}
 let span=front(r);let low=max(i.lo,span.x);let high=min(i.hi,span.y);
 let lo=select(low,high,reverse);let hi=select(high,low,reverse);
 var a=sourcePoint(r,lo,(f32(ordinal)+select(lo,1.0-lo,reverse))/f32(total),(i.cause&3u)==0u);
 var b=sourcePoint(r,hi,(f32(ordinal)+select(hi,1.0-hi,reverse))/f32(total),false);
 if(kind(r)==4u){
  // A crossing edge is not itself a corner: two receiver faces, or two
  // parameters on that edge, can establish different contacts. The exact
  // incidence dictionary already carries both surfaces and the parameter.
  // Its handle is used ONLY for same-frame corner equality, never exported
  // as persistent identity. Interior visibility cuts remain non-corners.
  if(a.visible.z!=NONE){a.visible.z=nodeFor(i.relation*2u+select(0u,1u,lo==1.0),LINE_NODE);}
  if(b.visible.z!=NONE){b.visible.z=nodeFor(i.relation*2u+select(0u,1u,hi==1.0),LINE_NODE);}
 }else if(shape(r)==3u){
  let owner=work[linkAt(i.relation)+6u];
  if(owner!=NONE){
   let source=atomicLoad(&nodes[owner].second);
   if(source!=0u){a.visible.z=0x80000000u|(source-1u);b.visible.z=a.visible.z;}
  }
 }
 points[base]=a;points[base+1u]=b;return base+2u;
}
fn contactContinuation(a:Point,b:Point,c:Point,d:Point)->bool{
 // Incidence, not a screen-space weld. Only uninterrupted visible contact
 // spans can lose a tessellation vertex. Visibility cuts keep NONE ports.
 if(!pointVisible(a)||!pointVisible(c)||b.visible.z==NONE||b.visible.z!=c.visible.z){return false;}
 let delta=d.position.xy-a.position.xy;let squared=dot(delta,delta);
 if(squared<=1e-10){return false;}
 let t=dot(b.position.xy-a.position.xy,delta)/squared;
 if(t<=0.0||t>=1.0){return false;}
 // Tolerance covers only f32 projection arithmetic, not a glyph-sized
 // geometric simplification. A depth or reciprocal-W bend remains a bend
 // even if its two spans happen to align in this camera's image.
 let magnitude=max(vec2f(1),max(max(abs(a.position.xy),abs(b.position.xy)),max(abs(c.position.xy),abs(d.position.xy))));
 let epsilon=magnitude*0.000002;
 if(any(abs(b.position.xy-c.position.xy)>epsilon)||any(abs(mix(a.position.xy,d.position.xy,t)-b.position.xy)>epsilon)){return false;}
 if(abs(b.position.z-c.position.z)>0.000001||abs(mix(a.position.z,d.position.z,t)-b.position.z)>0.000001){return false;}
 let aw=bitcast<f32>(a.visible.y);let bw=bitcast<f32>(b.visible.y);let cw=bitcast<f32>(c.visible.y);let dw=bitcast<f32>(d.visible.y);
 let wTolerance=max(max(abs(aw),abs(dw)),max(abs(bw),abs(cw)))*0.000002;
 return abs(bw-cw)<=wTolerance&&abs(mix(aw,dw,t)-bw)<=wTolerance;
}
fn compactContact(base:u32,end:u32)->u32{
 var output=base;
 for(var input=base;input+1u<end;input+=2u){
  let a=points[input];let b=points[input+1u];
  if(output>=base+2u&&contactContinuation(points[output-2u],points[output-1u],a,b)){
   points[output-1u]=b;
  }else{points[output]=a;points[output+1u]=b;output+=2u;}
 }
 return output;
}
fn direction(delta:vec2f)->u32{return u32(round(fract(atan2(delta.y,delta.x)/6.28318530718+1.0)*8.0))&7u;}
fn direction16(delta:vec2f)->u32{return u32(round(fract(atan2(delta.y,delta.x)/6.28318530718+1.0)*16.0))&15u;}
// JUNCTION INCIDENCE (compiled-role-grammar.ts compileJunctions). A run end
// joins the other lines at its source vertex when that vertex is visible on
// every leg; the legs read the same source chains (chord) from every run that
// meets there, so every run reaches the same record. Source incidence only.
fn endVisible(member:u32)->bool{
 let t=f32(member&1u);var piece=partitions[(member/2u)*2u];
 loop{if(piece==0u){break;}let item=intervals[piece-1u];if((item.cause&3u)==0u&&item.lo<=t&&item.hi>=t){return true;}piece=partitions[p.extent.x*2u+(piece-1u)*2u];}
 return false;
}
// An end that takes part in a junction: an active, projectable, non-contact
// edge end of the given hardness.
fn junctionEnd(e:u32,hard:bool)->bool{
 let q=relations[e/2u];
 return kind(q)!=4u&&work[linkAt(e/2u)+7u]!=0u&&select(q.a.w,q.b.w,(e&1u)==1u)>=.00001&&hardEdge(q)==hard;
}
// The two other ends at a vertex of degree three (NONE when there are not two).
fn otherEnds(node:u32,own:u32)->vec2u{
 var others=vec2u(NONE);var n=0u;var f=atomicLoad(&nodes[node].second)-1u;
 for(var k=0u;k<3u&&f!=NONE;k++){if(f!=own){if(n<2u){others[n]=f;}n++;}f=work[linkAt(f/2u)+10u+(f&1u)]-1u;}
 if(n!=2u){return vec2u(NONE);}return others;
}
// THREE LEGS: a T (the other two ends continue each other: one line passes
// through the vertex and this run ends on it) or a Y (all three end). Bits as
// the two-leg record plus the third leg in bits 11-14 and bit 15; the ending
// run of a T spells the glyph, the source-address least end of a Y. 0: none.
fn threeLegs(own:u32,corner:u32)->u32{
 let hard=hardEdge(relations[own/2u]);let others=otherEnds(corner,own);
 if(others.x==NONE||others.x/2u==own/2u||others.y/2u==own/2u||others.x/2u==others.y/2u){return 0u;}
 if(!junctionEnd(others.x,hard)||!junctionEnd(others.y,hard)){return 0u;}
 let lx=work[linkAt(others.x/2u)+(others.x&1u)];let ly=work[linkAt(others.y/2u)+(others.y&1u)];
 let through=lx==others.y&&ly==others.x;
 if(!through&&(lx!=NONE||ly!=NONE)){return 0u;}
 if(!endVisible(own)||!endVisible(others.x)||!endVisible(others.y)){return 0u;}
 let here=endpointPage(own);let a=chord(here,own);let b=chord(here,others.x);let c=chord(here,others.y);
 if(min(dot(a,a),min(dot(b,b),dot(c,c)))<289.0){return 0u;}
 let owner=through||choose(choose(own,others.x),others.y)==own;
 return 1u|select(0u,2u,owner)|(direction16(a)<<2u)|(direction16(b)<<6u)|select(0x400u,0u,hard)|(direction16(c)<<11u)|0x8000u;
}
// A T ON THIS RUN'S OWN LINE: where the run passes through a vertex of
// degree three (its two ends there continue each other) and the third end
// ends there (the ending run spells the glyph), the run records the point
// index of the vertex and the three legs so it holds back its own units around
// the glyph. Word: bit 31 valid, bit 28 natural, legs in bits 16-27 (ending
// leg, then the two through legs), point index in the low 16 bits. 0: none.
fn throughT(far:u32,next:u32,index:u32)->u32{
 let node=nodeFor(far,LINE_NODE);if(node==NONE||atomicLoad(&nodes[node].degree)!=3u||index>0xffffu){return 0u;}
 let hard=hardEdge(relations[far/2u]);let others=otherEnds(node,far);
 let stub=select(others.x,others.y,others.x==next);
 if(others.x==NONE||(others.x!=next&&others.y!=next)||stub/2u==far/2u||stub/2u==next/2u){return 0u;}
 if(!junctionEnd(stub,hard)||work[linkAt(stub/2u)+(stub&1u)]!=NONE||hardEdge(relations[next/2u])!=hard){return 0u;}
 if(!endVisible(stub)||!endVisible(far)||!endVisible(next)){return 0u;}
 let here=endpointPage(stub);let a=chord(here,stub);let b=chord(here,far);let c=chord(here,next);
 if(min(dot(a,a),min(dot(b,b),dot(c,c)))<289.0){return 0u;}
 return 0x80000000u|select(0x10000000u,0u,hard)|(direction16(a)<<16u)|(direction16(b)<<20u)|(direction16(c)<<24u)|index;
}
// The foot anchor: ONE retained contact_junction part per owner pair, emitted
// by the pair's representative relation at the centre of the visible contact
// on its lowest page line. Fixed conditions, no fit: the visible contact is
// wider than tall (the form stands above a ground-like line; a form meeting a
// wall has no retained part), and its width lies between CONTACT_PART_LEAST
// and CONTACT_PART_MOST of the part's native width, so the part's own
// standing rows are about as wide as the foot they replace. The part is the
// compiled (contact_junction, N in, E out) context; the retained entries of
// this context are all size band m. The contact line itself is still drawn
// by its run; a foot the part does not fit keeps that run alone.
// The owner node of a source named by (domain | producer bit, owner), as
// incidence keyed it (key, OWNER_NODE). Lookup only: an absent owner is NONE
// and nothing is claimed.
fn ownerNodeOf(source:vec2u)->u32{
 let wanted=SourceKey(vec4u(source.x&0x1fffffffu,source.y,(source.x&0x80000000u)|4u,NONE),0u,vec3u(0u));
 var at=hash(wanted);let tag=OWNER_NODE<<30u;
 for(var probe=0u;probe<min(p.extent.z,256u);probe++){
  let held=atomicLoad(&nodes[at].first);if(held==0u){return NONE;}
  if((held&0xc0000000u)==tag){let prior=key((held&0x3fffffffu)-1u,OWNER_NODE);if(all(prior.identity==wanted.identity)){return at;}}
  at=(at+1u)&(p.extent.z-1u);
 }
 return NONE;
}

// The fork producer's junction side table (compiled-construction-topology.ts
// TOPOLOGY_PART_JUNCTION_*) is resident in this source's topology words right
// after the part packet's canonical table; its offsets are relative to the
// packet start. A missing, foreign or malformed table is NONE, and the anchor
// keeps its incumbent decision.
fn junctionPacket(header:u32)->u32{
 let total=arrayLength(&work);
 if(header==NONE||header+${TOPOLOGY_PART_MAGIC_WORD+1}u>=total||work[header+${TOPOLOGY_PART_MAGIC_WORD}u]!=${TOPOLOGY_PART_MAGIC}u){return NONE;}
 let offset=work[header+${TOPOLOGY_PART_MAGIC_WORD+1}u];if(offset<5u||header+offset+5u>total){return NONE;}
 let packet=header+offset-5u;if(work[packet]!=${TOPOLOGY_PART_PACKET_MAGIC}u){return NONE;}
 return packet;
}
fn junctionTable(packet:u32)->u32{
 if(packet==NONE){return NONE;}
 let total=arrayLength(&work);let side=packet+work[packet+3u]+work[packet+4u];if(side+16u>total){return NONE;}
 var at=side;
 if(work[side]==${TOPOLOGY_PART_CANONICAL_MAGIC}u&&work[side+1u]==${TOPOLOGY_PART_CANONICAL_VERSION}u){at=packet+work[side+4u]+work[side+5u];}
 if(at+16u>total||work[at]!=${TOPOLOGY_PART_JUNCTION_MAGIC}u||work[at+1u]!=${TOPOLOGY_PART_JUNCTION_VERSION}u){return NONE;}
 return at;
}
fn junctionArmBit(armAt:u32,arms:u32,label:u32)->u32{
 for(var i=0u;i<arms;i++){if(work[armAt+i*${TOPOLOGY_PART_JUNCTION_ARM_WORDS}u]==label){return 1u<<i;}}
 return 0u;
}
// A record the consumer may use: COMPLETE, no missing arm, 2..32 arms each
// COMPLETE, its arm and edge ranges inside the words.
fn junctionUsable(packet:u32,row:u32)->bool{
 let total=arrayLength(&work);if(row+${TOPOLOGY_PART_JUNCTION_RECORD_WORDS}u>total){return false;}
 let flags=work[row+2u];if((flags&${TOPOLOGY_PART_JUNCTION_FLAG_COMPLETE}u)==0u||(flags&${TOPOLOGY_PART_JUNCTION_FLAG_MISSING_ARM}u)!=0u){return false;}
 let arms=work[row+4u];let armAt=packet+work[row+3u];if(arms<2u||arms>${SOURCE_JUNCTION_ARM_LIMIT}u||armAt+arms*${TOPOLOGY_PART_JUNCTION_ARM_WORDS}u>total){return false;}
 for(var i=0u;i<arms;i++){if((work[armAt+i*${TOPOLOGY_PART_JUNCTION_ARM_WORDS}u+1u]&${TOPOLOGY_PART_JUNCTION_ARM_COMPLETE}u)==0u){return false;}}
 return packet+work[row+5u]+work[row+6u]*${TOPOLOGY_PART_JUNCTION_EDGE_WORDS}u<=total;
}
// Does every arm section of this record meet at packet vertex v (its own label
// plus the labels across its band edges)? Packet-internal: no relation port.
fn junctionMeets(packet:u32,row:u32,v:u32)->bool{
 let arms=work[row+4u];let armAt=packet+work[row+3u];let edges=work[row+6u];let edgeAt=packet+work[row+5u];
 var met=0u;
 for(var e=0u;e<edges;e++){
  let at=edgeAt+e*${TOPOLOGY_PART_JUNCTION_EDGE_WORDS}u;if(work[at+3u]!=v&&work[at+4u]!=v){continue;}
  met|=junctionArmBit(armAt,arms,work[at+5u])|junctionArmBit(armAt,arms,work[at+6u]);
 }
 return met==select(0xffffffffu,(1u<<arms)-1u,arms<32u);
}
// The packet vertices of relation r's endpoint (end) and of its far end, with
// the far and own labels: (vertex, far vertex, far label, own label), or NONE
// in x. The SOURCE-PORT NAMESPACE problem is avoided entirely: a band relation
// (kind 0/2) carries its index edge's id in key.w (sourceEdge(id).x, the
// lowest incidence halfedge, surface-relations) with its ports ordered by that
// halfedge, and every packet edge names that same index edge id plus its own
// halfedge. The packet's representative ids are compared only with packet ids.
fn forkEnd(packet:u32,row:u32,r:Relation,end:u32)->vec4u{
 let k=kind(r);if(k!=0u&&k!=2u){return vec4u(NONE);}let edge=r.key.w;if((edge&0x80000000u)!=0u){return vec4u(NONE);}
 let edges=work[row+6u];let edgeAt=packet+work[row+5u];var other=NONE;var others=0u;
 for(var e=0u;e<edges;e++){
  let at=edgeAt+e*${TOPOLOGY_PART_JUNCTION_EDGE_WORDS}u;if(work[at]!=edge){continue;}
  // The relation's own halfedge: end 0 is its from (packet a).
  if(work[at+1u]==edge){return select(vec4u(work[at+3u],work[at+4u],work[at+6u],work[at+5u]),vec4u(work[at+4u],work[at+3u],work[at+5u],work[at+6u]),end==1u);}
  other=at;others++;
 }
 // Only the opposite incidence lies in the band (a band boundary edge): its
 // from is the relation's to. Two or more foreign incidences are ambiguous.
 if(others!=1u){return vec4u(NONE);}
 return select(vec4u(work[other+4u],work[other+3u],work[other+5u],work[other+6u]),vec4u(work[other+3u],work[other+4u],work[other+6u],work[other+5u]),end==1u);
}
fn junctionDirection(origin:vec4f,away:vec4f)->u32{return direction(away.xy-origin.xy);}
// The arm an endpoint at the joint reaches, by source ownership: the far end's
// arm when the far end carries one (the edge leads into that arm), else the
// joint's own arm when this edge is owned by it, else none (interior).
fn forkArm(armAt:u32,arms:u32,e:vec4u)->u32{
 if(junctionArmBit(armAt,arms,e.z)!=0u){return e.z;}
 if(junctionArmBit(armAt,arms,e.w)!=0u){return e.w;}
 return NONE;
}
const FORK_UNION=4u;

// source JOINT when it maps (forkEnd) to a packet vertex at which every arm of
// a usable fork record meets. Every usable record meeting at that joint takes
// part (their union, in table order, at most FORK_UNION; a larger union keeps
// the incumbent), and EVERY arm of every one of them must be reached by an
// incident relation (the first in stable source order per arm, forkArm) and
// be visible at the joint. In each record the last arm (an upper section) is
// the trunk and every other arm, in record order, one continued branch: one
// retained branch_entry whole part (role 2) per branch, entering from the
// trunk. A k-arm fork is k-1 cooperating retained parts at ONE source anchor,
// each keyed by (fork record, branch section), so neither table nor incident
// order changes any part's identity; every incident run keeps playing its own
// arm passage up to that anchor. Every part of the union is legal (retained
// entry present, fits the owner) before any is emitted; otherwise false and
// the caller's incumbent decision runs unchanged, so nothing is replaced
// partially.
fn junctionSourceFork(r:Relation,end:u32,node:u32,size:u32,owner:u32,low:f32,high:f32)->bool{
 let packet=junctionPacket(topologyHeader(r));let table=junctionTable(packet);if(table==NONE){return false;}
 let records=min(work[table+3u],${SOURCE_JUNCTION_RECORD_LIMIT}u);let recordAt=packet+work[table+2u];
 var joint=NONE;
 for(var k=0u;k<records;k++){
  let row=recordAt+k*${TOPOLOGY_PART_JUNCTION_RECORD_WORDS}u;if(!junctionUsable(packet,row)){continue;}
  let e=forkEnd(packet,row,r,end);if(e.x!=NONE&&junctionMeets(packet,row,e.x)){joint=e.x;break;}
 }
 if(joint==NONE){return false;}
 var forks:array<u32,FORK_UNION>;var count=0u;
 for(var k=0u;k<records;k++){
  let row=recordAt+k*${TOPOLOGY_PART_JUNCTION_RECORD_WORDS}u;if(!junctionUsable(packet,row)||!junctionMeets(packet,row,joint)){continue;}
  if(count==FORK_UNION){return false;}forks[count]=row;count++;
 }
 var armEnd:array<u32,${SOURCE_JUNCTION_ARM_LIMIT*4}>;for(var i=0u;i<${SOURCE_JUNCTION_ARM_LIMIT*4}u;i++){armEnd[i]=NONE;}
 var visible=false;var member=atomicLoad(&nodes[node].second);
 loop{if(member==0u){break;}let endpoint=member-1u;member=work[linkAt(endpoint/2u)+12u+(endpoint&1u)];
  let x=relations[endpoint/2u];var reached=false;
  for(var c=0u;c<count;c++){
   let row=forks[c];let e=forkEnd(packet,row,x,endpoint&1u);if(e.x!=joint){continue;}
   let arms=work[row+4u];let armAt=packet+work[row+3u];let arm=forkArm(armAt,arms,e);if(arm==NONE){continue;}
   for(var i=0u;i<arms;i++){let slot=c*${SOURCE_JUNCTION_ARM_LIMIT}u+i;if(work[armAt+i*${TOPOLOGY_PART_JUNCTION_ARM_WORDS}u]==arm&&(armEnd[slot]==NONE||choose(armEnd[slot],endpoint)==endpoint)){armEnd[slot]=endpoint;}}
   reached=true;
  }
  if(!reached){continue;}
  var interval=partitions[(endpoint/2u)*2u];
  loop{if(interval==0u){break;}let item=intervals[interval-1u];let t=f32(endpoint&1u);visible=visible||((item.cause&3u)==0u&&item.lo<=t&&item.hi>=t);interval=partitions[p.extent.x*2u+(interval-1u)*2u];}
 }
 if(!visible){return false;}
 let q=page(select(r.a,r.b,end==1u));
 let rows=u32(max(1.0,ceil((high-low)/17.0)));let width=real(atomicLoad(&nodes[owner].maxX))-real(atomicLoad(&nodes[owner].minX));
 var parts=0u;
 for(var c=0u;c<count;c++){
  let arms=work[forks[c]+4u];
  for(var i=0u;i<arms;i++){if(armEnd[c*${SOURCE_JUNCTION_ARM_LIMIT}u+i]==NONE){return false;}}
  let trunk=armEnd[c*${SOURCE_JUNCTION_ARM_LIMIT}u+arms-1u];let t=relations[trunk/2u];let incoming=junctionDirection(q,page(select(t.b,t.a,(trunk&1u)==1u)));
  for(var i=0u;i+1u<arms;i++){
   let branch=armEnd[c*${SOURCE_JUNCTION_ARM_LIMIT}u+i];let b=relations[branch/2u];let outgoing=junctionDirection(q,page(select(b.b,b.a,(branch&1u)==1u)));
   let program=p.table.w+((2u*3u+size)*64u+incoming*8u+outgoing)*8u;
   if(work[program+1u]==0u||work[program+6u]>rows||f32(work[program+7u])>max(11.0,ceil(width/11.0)*11.0)){return false;}
   parts++;
  }
 }
 // All parts or none: one reservation for the whole union.
 let base=atomicAdd(&nodes[p.extent.z].first,parts);let slot=atomicAdd(&nodes[p.extent.z].second,parts);
 if(base+parts>p.extent.y){atomicOr(&nodes[p.extent.z].degree,2u);}
 if(slot+parts>p.sourceExtent.z){atomicOr(&nodes[p.extent.z].degree,4u);}
 if(base+parts>p.extent.y||slot+parts>p.sourceExtent.z){return true;}
 let group=topologyGroup(r,r.ports[end]);var at=0u;
 for(var c=0u;c<count;c++){
  let row=forks[c];let arms=work[row+4u];let armAt=packet+work[row+3u];let fork=work[row+1u];
  let trunk=armEnd[c*${SOURCE_JUNCTION_ARM_LIMIT}u+arms-1u];let t=relations[trunk/2u];let incoming=junctionDirection(q,page(select(t.b,t.a,(trunk&1u)==1u)));
  for(var i=0u;i+1u<arms;i++){
   let branch=armEnd[c*${SOURCE_JUNCTION_ARM_LIMIT}u+i];let b=relations[branch/2u];let outgoing=junctionDirection(q,page(select(b.b,b.a,(branch&1u)==1u)));
   let program=p.table.w+((2u*3u+size)*64u+incoming*8u+outgoing)*8u;let section=work[armAt+i*${TOPOLOGY_PART_JUNCTION_ARM_WORDS}u];
   points[base+at]=sourcePoint(r,f32(end),0.0,true);
   // The joint's source key, the fork record and the branch SECTION: a stable
   // identity across frames, cuts, table order and incident-relation order.
   let identity=(r.key.x*0x9e3779b9u)^(r.key.y*0x85ebca6bu)^r.ports[end]^0x10000000u^(fork*0x85ebca6bu)^(section*0x27d4eb2fu);
   let flags=2u|0x10000u|(incoming<<20u)|(outgoing<<23u);
   runs[slot+at]=Run(vec4u(base+at,1,flags,size),vec4u(identity,r.key.x|(r.key.z&0x80000000u),r.key.y,identity^group),vec4u(work[program],work[program+1u],work[program+2u],work[program+3u]),vec4u(work[program+4u],work[program+5u],r.ports[end],0x10000000u),vec4u(0u));
   at++;
  }
 }
 return true;
}
const CONTACT_PART_LEAST=.4;
const CONTACT_PART_MOST=1.0;
fn contactAnchor(id:u32,r:Relation){
 let foot=footContact(r);if(foot==0u){return;}
 let pair=nodeFor(id*2u,PAIR_NODE);if(pair==NONE||atomicLoad(&nodes[pair].least)!=id*2u){return;}
 let low=vec2f(real(atomicLoad(&nodes[pair].minX)),real(atomicLoad(&nodes[pair].minY)));
 let high=vec2f(real(atomicLoad(&nodes[pair].maxX)),real(atomicLoad(&nodes[pair].maxY)));
 let extent=high-low;if(extent.x<0.0||extent.y>extent.x){return;}
 let program=p.table.w+((3u*3u+1u)*64u+6u*8u)*8u;
 if(work[program+1u]==0u){return;}
 let native=f32(work[program+7u]);
 if(extent.x<CONTACT_PART_LEAST*native||extent.x>CONTACT_PART_MOST*native){return;}
 // The representative's first visible point carries the contact's depth.
 var depth=vec4f(0.0);var found=false;var part=partitions[id*2u];
 loop{if(part==0u||found){break;}let interval=intervals[part-1u];
  if(finiteInterval(r,interval)&&(interval.cause&3u)==0u){depth=page(mix(r.a,r.b,max(interval.lo,front(r).x)));found=true;}
  part=partitions[p.extent.x*2u+(part-1u)*2u];}
 if(!found){return;}
 let base=atomicAdd(&nodes[p.extent.z].first,1u);let slot=atomicAdd(&nodes[p.extent.z].second,1u);
 if(base>=p.extent.y){atomicOr(&nodes[p.extent.z].degree,2u);}
 if(slot>=p.sourceExtent.z){atomicOr(&nodes[p.extent.z].degree,4u);}
 if(base>=p.extent.y||slot>=p.sourceExtent.z){return;}
 points[base]=Point(vec4f((low.x+high.x)*.5,high.y,depth.z,0.0),vec4u(1u,bitcast<u32>(depth.w),NONE,r.key.w));
 // The anchor belongs to the STANDING form (its rows above the contact lie
 // on that form); paper.z names the surface owner, which articulation does
 // not read as an occluder of its own contact. paper.w 0x10000001 keeps the
 // correspondence key apart from a fork or cap at a source vertex.
 let standing=select(vec2u((r.ports.z&0x1fffffffu)|(r.ports.z&0x80000000u),r.ports.w),vec2u(r.key.x|(r.key.z&0x80000000u),r.key.y),foot==1u);
 let surface=select(r.key.y,r.ports.w,foot==1u);
 let identity=(standing.x*0x9e3779b9u)^(standing.y*0x85ebca6bu)^(surface*0xc2b2ae35u)^0x30000000u;
 let flags=3u|0x10000u|(6u<<20u);
 // The anchor carries the STANDING form's size band, the one value every
 // other run of that owner carries (assemble, anchors: the page rows the
 // owner occupies), so articulation selects the foot with its form. The part
 // program above stays the context's one retained band. A standing form with
 // no owner node (nothing of it entered incidence) keeps band 1.
 var band=1u;let node=select(ownerNodeOf(standing),work[linkAt(id)+6u],foot==1u);
 if(node!=NONE){let rows=real(atomicLoad(&nodes[node].maxY))-real(atomicLoad(&nodes[node].minY));if(rows>=0.0){band=select(select(2u,1u,rows<119.0),0u,rows<51.0);}}
  runs[slot]=Run(vec4u(base,1,flags,band),vec4u(identity,standing.x,standing.y,identity),vec4u(work[program],work[program+1u],work[program+2u],work[program+3u]),vec4u(work[program+4u],work[program+5u],surface,0x10000001u),vec4u(0u));
}
// FACE SOURCE-VERTEX JOINT INDEX. Face packets can contain more than 128
// source edges, so their joints cannot be recovered by the native reader's
// fixed edge bitset. The face-family LINE_NODE chain is the compiled source
// incidence index. One representative endpoint per source vertex walks that
// chain once, counts its face contributors, projects their posed legs on the
// GPU, and emits a joint-only run. No all-source scan or screen weld occurs.
fn faceVertexCandidate(endpoint:u32,owner:u32)->bool{
 let edge=endpoint/2u;let at=linkAt(edge);let q=relations[edge];let group=work[at+8u];
 // A visibility fragment is not another source leg. The packet-group
 // representative is the one source edge admitted to this vertex; its
 // group's visibility bit is the union of all retained fragments.
 return work[at+7u]!=0u&&work[at+6u]==owner&&group!=NONE&&atomicLoad(&nodes[group].least)==edge*2u&&faceEdge(q)!=0u&&!faceHidden(faceEdge(q));
}
fn faceVertexCount(node:u32,owner:u32)->u32{
 var total=0u;var member=atomicLoad(&nodes[node].second);
 loop{if(member==0u){break;}let endpoint=member-1u;
  if(faceVertexCandidate(endpoint,owner)){total++;}
  member=work[linkAt(endpoint/2u)+10u+(endpoint&1u)];
 }
 return total;
}
fn faceVertexAnchor(id:u32,r:Relation,at:u32){
 let owner=work[at+6u];if(owner==NONE){return;}
 let low=real(atomicLoad(&nodes[owner].minY));let high=real(atomicLoad(&nodes[owner].maxY));
 let size=select(select(2u,1u,high-low<119.0),0u,high-low<51.0);
 for(var end=0u;end<2u;end++){
  if(select(r.a.w,r.b.w,end==1u)<.00001){continue;}
  let endpoint=id*2u+end;let node=nodeFor(endpoint,LINE_NODE);
  if(node==NONE||atomicLoad(&nodes[node].least)!=endpoint){continue;}
  let edges=faceVertexCount(node,owner);if(edges<2u){continue;}
  let total=edges*2u;let base=atomicAdd(&nodes[p.extent.z].first,total);let slot=atomicAdd(&nodes[p.extent.z].second,1u);
  if(base+total>p.extent.y){atomicOr(&nodes[p.extent.z].degree,2u);}
  if(slot>=p.sourceExtent.z){atomicOr(&nodes[p.extent.z].degree,4u);}
  if(base+total>p.extent.y||slot>=p.sourceExtent.z){continue;}
  var output=base;var member=atomicLoad(&nodes[node].second);
  loop{if(member==0u){break;}let incident=member-1u;
   if(faceVertexCandidate(incident,owner)){
    let edge=incident/2u;let q=relations[edge];let side=incident&1u;let group=work[linkAt(edge)+8u];let live=(atomicLoad(&nodes[group].degree)&FACE_GROUP_VISIBLE)!=0u;
    points[output]=sourcePoint(q,f32(side),0.0,live);
    points[output+1u]=sourcePoint(q,f32(1u-side),1.0,false);output+=2u;
   }
   member=work[linkAt(incident/2u)+10u+(incident&1u)];
  }
  // The source vertex port, not this frame's relation index, is the joint's
  // retained identity. Duplicate visibility fragments therefore publish the
  // same native passage even when their dispatch order changes.
  let vertex=r.ports[end];let identity=(r.key.x*0x9e3779b9u)^(r.key.y*0x85ebca6bu)^vertex^0x40000000u;
  runs[slot]=Run(vec4u(base,total,${FACE_JUNCTION_RUN}u|0x100000u,size),vec4u(identity,r.key.x|(r.key.z&0x80000000u),r.key.y,vertex),vec4u(0u),vec4u(0u),vec4u(0u));
 }
}
@compute @workgroup_size(64) fn anchors(@builtin(global_invocation_id) gid:vec3u){
 let id=gid.x+gid.y*p.table.z*64u;if(id>=count()||sourceStatus[1]!=0u){return;}
 let r=relations[id];let at=linkAt(id);if(work[at+7u]==0u){return;}
 if(kind(r)==4u){contactAnchor(id,r);return;}

 if(authoredPaired(r)){let pair=nodeFor(id*2u,PAIR_NODE);if(pair!=NONE&&(atomicLoad(&nodes[pair].degree)&0x80000000u)!=0u){return;}}
 let faceWord=faceEdge(r);if(faceWord!=0u){if(!faceHidden(faceWord)){faceVertexAnchor(id,r,at);}return;}
 if(shape(r)!=0u){return;}
 let owner=work[at+6u];if(owner==NONE){return;}
 let low=real(atomicLoad(&nodes[owner].minY));let high=real(atomicLoad(&nodes[owner].maxY));
 let size=select(select(2u,1u,high-low<119.0),0u,high-low<51.0);
 for(var end=0u;end<2u;end++){
  let criticalKind=critical(r,end);if(criticalKind==0u||select(r.a.w,r.b.w,end==1u)<.00001){continue;}
  let node=nodeFor(id*2u+end,JUNCTION_NODE);if(node==NONE||atomicLoad(&nodes[node].least)!=id*2u+end){continue;}

  // source arms (junctionSourceFork); anything else keeps the incumbent below.
  if(junctionSourceFork(r,end,node,size,owner,low,high)){continue;}
  let degree=atomicLoad(&nodes[node].degree);let fork=(criticalKind&2u)!=0u&&degree>=3u;let cap=degree==1u;
  if(!fork&&!cap){continue;}
  let q=page(select(r.a,r.b,end==1u));
  // Context is the original source junction's incident branches, never a
  // nearest glyph/tangent fit. Stable source order selects the two ports.
  var first=NONE;var second=NONE;var visible=false;var member=atomicLoad(&nodes[node].second);
  loop{if(member==0u){break;}let endpoint=member-1u;member=work[linkAt(endpoint/2u)+12u+(endpoint&1u)];
   if(first==NONE||choose(first,endpoint)==endpoint){second=first;first=endpoint;}else if(second==NONE||choose(second,endpoint)==endpoint){second=endpoint;}
   var interval=partitions[(endpoint/2u)*2u];
   loop{if(interval==0u){break;}let item=intervals[interval-1u];let t=f32(endpoint&1u);visible=visible||((item.cause&3u)==0u&&item.lo<=t&&item.hi>=t);interval=partitions[p.extent.x*2u+(interval-1u)*2u];}
  }
  if(first==NONE||!visible){continue;}
  let a=relations[first/2u];var incoming=page(select(a.b,a.a,(first&1u)==1u)).xy-q.xy;
  var outgoing=-incoming;
  if(second!=NONE){let b=relations[second/2u];outgoing=page(select(b.b,b.a,(second&1u)==1u)).xy-q.xy;}
  let role=select(select(1u,0u,q.y<(low+high)*.5),2u,fork);
  let program=p.table.w+((role*3u+size)*64u+direction(incoming)*8u+direction(outgoing))*8u;
  if(work[program+1u]==0u){continue;}
  // Coarse size-band fallback is not permission to paste a multirow part
  // larger than its entire projected owner. The original source runs remain
  // the class construction when no intact native part fits this row tier.
  let width=real(atomicLoad(&nodes[owner].maxX))-real(atomicLoad(&nodes[owner].minX));
  if(work[program+6u]>u32(max(1.0,ceil((high-low)/17.0)))||f32(work[program+7u])>max(11.0,ceil(width/11.0)*11.0)){continue;}
  let base=atomicAdd(&nodes[p.extent.z].first,1u);let slot=atomicAdd(&nodes[p.extent.z].second,1u);
  if(base>=p.extent.y){atomicOr(&nodes[p.extent.z].degree,2u);}
  if(slot>=p.sourceExtent.z){atomicOr(&nodes[p.extent.z].degree,4u);}
  if(base>=p.extent.y||slot>=p.sourceExtent.z){continue;}
  // Its parameter belongs to the junction, not endpoint 0/1 of whichever
  // incident edge happened to carry the representative this frame.
  points[base]=sourcePoint(r,f32(end),0.0,true);
  // Keep the source-port key, but include its resident construction group so
  // a shared contact/section cannot collapse into an unrelated run during
  // publication or correspondence.
  let group=topologyGroup(r,r.ports[end]);
  let identity=(r.key.x*0x9e3779b9u)^(r.key.y*0x85ebca6bu)^r.ports[end]^0x10000000u;
  // Declared branch-port context survives in the anchor record. Scale may
  // insert run units, but cannot alone replace an enduring whole role part.
  // paper.z is deliberately the SOURCE JUNCTION PORT. node.least above
  // admits exactly one representative of its incident source relations;
  // an incident-edge address would make that shared join camera-order-owned.
  let flags=select(select(3u,0u,role==0u),2u,role==2u)|0x10000u|(direction(incoming)<<20u)|(direction(outgoing)<<23u);
  runs[slot]=Run(vec4u(base,1,flags,size),vec4u(identity,r.key.x|(r.key.z&0x80000000u),r.key.y,identity^group),vec4u(work[program],work[program+1u],work[program+2u],work[program+3u]),vec4u(work[program+4u],work[program+5u],r.ports[end],0x10000000u),vec4u(0u));
 }
}
// The owner's smallest boundary relation reserves the ONE whole-form run of
// a rounded source. Its paired cap/shoulder/flank/return program reads these
// points as unordered source pairs, so no cycle walk or seam order is needed.
@compute @workgroup_size(64) fn roundReserve(@builtin(global_invocation_id) gid:vec3u){
 let id=gid.x+gid.y*p.table.z*64u;if(id>=count()||sourceStatus[1]!=0u){return;}
 let at=linkAt(id);if(work[at+7u]==0u){return;}let r=relations[id];

 if(authoredPaired(r)){let pair=nodeFor(id*2u,PAIR_NODE);if(pair!=NONE&&(atomicLoad(&nodes[pair].degree)&AUTHORED_PAIR_ATTEMPTED)!=0u){return;}authoredPairReserve(id,r);return;}
 if(shape(r)==3u&&kind(r)!=4u){
  let owner=work[at+6u];if(owner==NONE||atomicLoad(&nodes[owner].least)!=id*2u){return;}
  let slot=atomicAdd(&nodes[p.extent.z].second,1u);
  if(slot>=p.sourceExtent.z){atomicOr(&nodes[p.extent.z].degree,4u);return;}
  // Zero points: never a drawable run, never a second terrain producer.
  runs[slot]=Run(vec4u(0u),vec4u(0u,r.key.x|(r.key.z&0x80000000u),r.key.y,${SPAN_SOURCE_TAG}u),vec4u(0u),vec4u(0u),vec4u(0u));
  atomicStore(&nodes[owner].second,slot+1u);return;
 }
 let form=shape(r);let round=form==2u&&kind(r)==0u;let box=form==1u&&(kind(r)==0u||kind(r)==2u);
 let faceWord=faceEdge(r);let face=faceWord!=0u&&!faceHidden(faceWord);
 if(!round&&!box&&!face){return;}
 let owner=work[at+6u];if(owner==NONE||atomicLoad(&nodes[owner].least)!=id*2u){return;}
 let total=atomicLoad(&nodes[owner].second);if(total==0u||((atomicLoad(&nodes[owner].degree)&1u)==0u&&!(box||face))){return;}
 let low=real(atomicLoad(&nodes[owner].minY));let high=real(atomicLoad(&nodes[owner].maxY));let height=high-low;
 // Every planar VOLUME at every size is ONE edge construction: a cube, a
 // pedestal, a room, a building block. The class is the compiled topology
 // (plane regions), never the projected size: a many-plane surface (terrain,
 // a voxel landscape) is not a box, its drawing is ranked silhouettes and
 // collapsed facet creases (finding 4), which the per-edge runs own.
 // total counts endpoint pairs of source relations, NOT visibility fragments.
 // A volume has few edges (a cuboid 12, a room with an opening about 30); a
 // deformed field or a chunk carries hundreds of interior creases, and those
 // are its drawing. The unreserved owner keeps the per-edge runs (assemble).
 if(box&&(!planarVolume(r)||total>128u)){return;}
 // THE FACE OWNER: one reserved construction of every face-owned source
 // feature/port group (all its wall faces: a corner two faces share is one
 // vertex of one packet), read by articulation's box program with the face
 // bit 0x200000. There is no fixed face-count escape hatch here: if the
 // source-sized run/point arena cannot hold the complete packet, the normal
 // allocation status requests growth and the incumbent remains intact.
 let base=atomicAdd(&nodes[p.extent.z].first,total);let slot=atomicAdd(&nodes[p.extent.z].second,1u);
 if(base+total>p.extent.y){atomicOr(&nodes[p.extent.z].degree,2u);}
 if(slot>=p.sourceExtent.z){atomicOr(&nodes[p.extent.z].degree,4u);}
 if(base+total>p.extent.y||slot>=p.sourceExtent.z){return;}
 let size=select(select(2u,1u,height<119.0),0u,height<51.0);
 let identity=(r.key.x*0x9e3779b9u)^(r.key.y*0x85ebca6bu)^select(select(0x10000000u,0x40000000u,face),0x20000000u,round);
 // The compact box is a hard construction: its | ￣ ＿ units keep their
 // visible source interval like every hard edge, so the base on the ground
 // contact is not read as occluded by the ground in front of it.
 // This record names the complete boundary, not whichever silhouette or
 // crease happened to win its atomic reservation in this frame.
 runs[slot]=Run(vec4u(base,total,2u|select(select(0x140000u,0x340000u,face),0x20000u,round),size),vec4u(identity,r.key.x|(r.key.z&0x80000000u),r.key.y,identity),vec4u(0u),vec4u(0),vec4u(0u));
 atomicStore(&nodes[owner].least,base);atomicStore(&nodes[owner].second,0u);atomicOr(&nodes[owner].degree,0x80000000u);
}
// CANONICAL PACKET ORDER (load-to-load stability, 2026-11-13). assemble fills
// a reserved whole-form packet at least+atomicAdd, so its endpoint pairs came
// in thread order, and articulation reads a packet's first and last point. The
// pairs are put in content order: source relation address, then source
// parameter, then page position. Shell sort, one thread per packet (a box holds
// at most 128 edges; a round's boundary can hold more).
fn pairBefore(a:u32,b:u32)->bool{
 let x=points[a];let y=points[b];
 if(pointFeatureId(x)!=pointFeatureId(y)){return pointFeatureId(x)<pointFeatureId(y);}
 let px=bitcast<vec4u>(x.position);let py=bitcast<vec4u>(y.position);
 if(px.w!=py.w){return px.w<py.w;}
 if(px.x!=py.x){return px.x<py.x;}
 if(px.y!=py.y){return px.y<py.y;}
 let qx=points[a+1u];let qy=points[b+1u];
 let rx=bitcast<vec4u>(qx.position);let ry=bitcast<vec4u>(qy.position);
 if(rx.w!=ry.w){return rx.w<ry.w;}
 if(rx.x!=ry.x){return rx.x<ry.x;}
 return rx.y<ry.y;
}
fn facePacketBefore(a:u32,b:u32)->bool{
 let x=points[a];let y=points[b];
 if(pointFeatureId(x)!=pointFeatureId(y)){return pointFeatureId(x)<pointFeatureId(y);}
 let xa=points[a+1u];let ya=points[b+1u];
 let xl=min(x.visible.z,xa.visible.z);let yl=min(y.visible.z,ya.visible.z);
 if(xl!=yl){return xl<yl;}
 return max(x.visible.z,xa.visible.z)<max(y.visible.z,ya.visible.z);
}
@compute @workgroup_size(64) fn canonPackets(@builtin(global_invocation_id) gid:vec3u){
 let id=gid.x+gid.y*p.table.z*64u;
 if(id>=min(atomicLoad(&nodes[p.extent.z].second),p.sourceExtent.z)||sourceStatus[1]!=0u){return;}
 let run=runs[id];let flags=run.range.z;
 // Programmed soft runs already carry their source-root pair first. Their
 // whole-part correspondence must not be re-ordered by a visibility-derived
 // parameter sort; legacy packets retain canonical sorting below.
 if((run.range.w>>5u)!=0u){return;}
 // A face descriptor and finite draw pair are ONE four-point packet. Their
 // position.w values have different meanings (clip w versus source phase).
 // Sort whole packets by source feature/ports, never the two pairs apart.
 // Retaining atomic assembly order would also make bounded joint readers
 // change their source subset between otherwise identical frames.
 let face=(flags&0x200000u)!=0u;let stride=select(2u,4u,face);
 if((flags&0x10000u)!=0u||(flags&0x120000u)==0u){return;}
 let base=run.range.x;let pairs=run.range.y/stride;if(pairs<2u||base+pairs*stride>p.extent.y){return;}
 var gap=1u;while(gap*3u+1u<pairs){gap=gap*3u+1u;}
 loop{
  for(var i=gap;i<pairs;i++){
   let a=points[base+i*stride];let b=points[base+i*stride+1u];
   var c=a;var d=b;if(face){c=points[base+i*stride+2u];d=points[base+i*stride+3u];}
   var j=i;
   loop{
    if(j<gap){break;}
    let k=j-gap;
    // Compare the held pair with the pair at k (both written back below).
    points[base+j*stride]=a;points[base+j*stride+1u]=b;
    if(face){points[base+j*stride+2u]=c;points[base+j*stride+3u]=d;}
    var before=false;if(face){before=facePacketBefore(base+j*stride,base+k*stride);}else{before=pairBefore(base+j*stride,base+k*stride);}
    if(!before){break;}
    points[base+j*stride]=points[base+k*stride];points[base+j*stride+1u]=points[base+k*stride+1u];
    if(face){points[base+j*stride+2u]=points[base+k*stride+2u];points[base+j*stride+3u]=points[base+k*stride+3u];}
    j=k;
   }
   points[base+j*stride]=a;points[base+j*stride+1u]=b;
   if(face){points[base+j*stride+2u]=c;points[base+j*stride+3u]=d;}
  }
  if(gap==1u){break;}gap=(gap-1u)/3u;
 }
}
@compute @workgroup_size(64) fn assemble(@builtin(global_invocation_id) gid:vec3u){
 let id=gid.x+gid.y*p.table.z*64u;if(id>=count()||sourceStatus[1]!=0u){return;}
 let at=linkAt(id);if(work[at+7u]==0u){return;}let r=relations[id];let form=shape(r);

 // and nothing else; an unreserved one falls through to per-edge assembly
 // (never into an owner round/box/face packet, which it was not counted in).
 let authored=authoredPaired(r);
 if(authored){let pair=nodeFor(id*2u,PAIR_NODE);if(pair!=NONE&&(atomicLoad(&nodes[pair].degree)&0x80000000u)!=0u){authoredPairFill(r,pair);return;}}
 // A face-owned edge fills its owner's reserved face construction as a box
 // edge; a compiled-out face edge is drawn by nothing (slice 1b).
 let faceWord=faceEdge(r);if(faceHidden(faceWord)){return;}
 let faceGroup=select(NONE,work[at+8u],faceWord!=0u);
 if(faceWord!=0u&&(faceGroup==NONE||atomicLoad(&nodes[faceGroup].least)!=id*2u)){return;}
 let round=form==2u&&kind(r)==0u;let box=(form==1u&&(kind(r)==0u||kind(r)==2u))||faceWord!=0u;
 let reserved=(round||box)&&!authored;let owner0=work[at+6u];
 if(reserved&&(owner0==NONE||(atomicLoad(&nodes[owner0].degree)&0x80000000u)!=0u)){
  // Fill the owner's reserved whole-form run. The parameter is the ORIGINAL
  // source edge address, stable across frames and independent of which
  // relation this frame's atomics placed first.
  let owner=owner0;if(owner==NONE){return;}
  var n=0u;var part=partitions[id*2u];
  loop{if(part==0u){break;}let interval=intervals[part-1u];if(finiteInterval(r,interval)){n+=2u;}part=partitions[p.extent.x*2u+(part-1u)*2u];}
  // A face edge reserves one descriptor pair and one draw pair for its whole
  // source passage. The partition list is visibility data, not an endpoint
  // stream. A volume edge always reserves its original endpoint pair too.
  var faceHasVisible=false;if(faceWord!=0u){faceHasVisible=(atomicLoad(&nodes[faceGroup].degree)&FACE_GROUP_VISIBLE)!=0u;n=4u;}
  else if(box){n=2u;}
  if(n==0u){return;}
  var output=atomicLoad(&nodes[owner].least)+atomicAdd(&nodes[owner].second,n);
  if(box){
   // A planar form's edge is ONE unit of its box construction, so the box
   // reads one whole edge per source relation. Original endpoints and
   // corners survive visibility cuts; only a true near-plane cut changes
   // the projectable span. Native articulation clips against foreground
   // visibility per intact glyph, not by replacing the construction graph
   // with the hull of its visible intervals. An edge on a crease or ground is
   // cut by the depth test into slivers; those slivers are not corners,
   // lids or depth edges.
   let end=output+n;let span=front(r);
   var edgeHasVisible=false;part=partitions[id*2u];
   loop{if(part==0u){break;}let interval=intervals[part-1u];
    if(finiteInterval(r,interval)&&(interval.cause&3u)==0u){edgeHasVisible=true;}
    part=partitions[p.extent.x*2u+(part-1u)*2u];}
   // A face packet is retained for ANY real visible interval. Its whole-edge
   // passage is then filtered per native scalar by the face consumer; a
   // fragment-size cutoff would turn visibility into a producer-side drop.
   // The 12% source-drop rule is intentionally gone. The complete original
   // box edge remains in the packet at 0%, 11%, 13% and 100% visibility;
   // only the point visibility flags and downstream native-unit visibility
   // decide whether ink is emitted.
   let ordinal=f32(r.key.w&0xffffu);
    if(faceWord!=0u){
     // Keep the original homogeneous endpoints/ports once, then the finite
     // display span. The original feature address stays in visible.w and the
     // draw pair's canonical source parameter stays in position.w. A
     // visibility cut is never a corner; sourcePoint marks ports only for
     // t==0/1. This is one packet per source feature, not
     // (edgeOrdinal+t)/65536 as its identity.
     points[output]=wholeEdgeDescriptorPoint(r,0u);points[output+1u]=wholeEdgeDescriptorPoint(r,1u);output+=2u;
     if(faceHasVisible){
      points[output]=sourcePoint(r,span.x,span.x,true);points[output+1u]=sourcePoint(r,span.y,span.y,false);output+=2u;
     }
    }else{
     points[output]=sourcePoint(r,span.x,(ordinal+span.x)/65536.0,edgeHasVisible);points[output+1u]=sourcePoint(r,span.y,(ordinal+span.y)/65536.0,false);output+=2u;
    }
   for(;output<end;output++){points[output]=Point(vec4f(0.0),vec4u(0u));}
   return;
  }
  part=partitions[id*2u];
  loop{if(part==0u){break;}let interval=intervals[part-1u];output=appendInterval(r,interval,false,r.key.w&0xffffu,65536u,output);part=partitions[p.extent.x*2u+(part-1u)*2u];}
  return;
 }
 // An unreserved rounded boundary (no visible interval) draws nothing. An
 // unreserved planar owner is a surface (a field, a chunk: many planes or
 // many edges), whose ranked creases are the per-edge runs below.
 if(round&&!authored){return;}
 // The root is chosen by ORIGINAL source address, not atomic emission order.
 var start=choose(work[at+4u],work[at+5u]);
 if(start/2u!=id){return;}
 // Pointer doubling identifies one component. Open walks start at their
 // terminal endpoint; the rounded whole form was reserved above.
 start=work[at+2u];
 var edgeCount=0u;var pointCount=0u;var anyVisible=false;
 let owner=work[at+6u];if(owner==NONE){return;}
 let low=vec2f(real(atomicLoad(&nodes[owner].minX)),real(atomicLoad(&nodes[owner].minY)));
 let high=vec2f(real(atomicLoad(&nodes[owner].maxX)),real(atomicLoad(&nodes[owner].maxY)));
 var endpoint=start;
 // Count FIRST, then reserve the exact native source footprint. Allocation
 // failure sets a real required count; it never publishes a truncated run.
 for(var step=0u;step<count();step++){
  let edge=endpoint/2u;let q=relations[edge];
  edgeCount++;
  var part=partitions[edge*2u];
  loop{if(part==0u){break;}let interval=intervals[part-1u];if(finiteInterval(q,interval)){pointCount+=2u;anyVisible=anyVisible||(interval.cause&3u)==0u;}part=partitions[p.extent.x*2u+(part-1u)*2u];}
  let next=work[linkAt(edge)+(1u-(endpoint&1u))];if(next==NONE||next==start){break;}endpoint=next;
 }

 // standing form the far strip takes whole (standingRetained) is still that
 // owner's source: reserve it with every point hidden, so the kind-3 profile
 // reads the owner's complete outline before foreground visibility, never the
 // walks that happen to survive it. Any other unseen walk is not reserved.
 if(pointCount==0u||(!anyVisible&&!standingRetained(r,low,high))){return;}
 let base=atomicAdd(&nodes[p.extent.z].first,pointCount);let slot=atomicAdd(&nodes[p.extent.z].second,1u);
 if(base+pointCount>p.extent.y){atomicOr(&nodes[p.extent.z].degree,2u);}
 if(slot>=p.sourceExtent.z){atomicOr(&nodes[p.extent.z].degree,4u);}
 if(base+pointCount>p.extent.y||slot>=p.sourceExtent.z){return;}
 var output=base;endpoint=start;var ordinal=0u;var upright=true;var hard=true;
 var hasTurn=false;var previousDirection=vec2f(0);var teeWords=vec3u(0u);var tees=0u;var silhouette=false;var foliage=true;var partFar=true;var compiledOutline=true;
 for(var step=0u;step<count();step++){
  let edge=endpoint/2u;let q=relations[edge];silhouette=silhouette||kind(q)==0u;foliage=foliage&&foliageRelation(q);partFar=partFar&&(u32(q.sourceChart.w)&2u)!=0u;compiledOutline=compiledOutline&&(u32(q.sourceChart.w)&4u)!=0u;
  let reverse=(endpoint&1u)!=0u;upright=upright&&(q.key.z&0x20000000u)!=0u;hard=hard&&hardEdge(q);
  let span=front(q);let direction=(page(mix(q.a,q.b,span.y)).xy-page(mix(q.a,q.b,span.x)).xy)*select(1.0,-1.0,reverse);
  let scale=length(previousDirection)*length(direction);
  if(ordinal>0u&&scale>1e-10){hasTurn=hasTurn||dot(previousDirection,direction)<.99999*scale;}
  previousDirection=direction;
  var part=partitions[edge*2u+select(0u,1u,reverse)];
  loop{if(part==0u){break;}let interval=intervals[part-1u];output=appendInterval(q,interval,reverse,ordinal,edgeCount,output);part=partitions[p.extent.x*2u+(part-1u)*2u+select(0u,1u,reverse)];}
  ordinal++;
  let next=work[linkAt(edge)+(1u-(endpoint&1u))];
  // Up to three T vertices on this run's own line (throughT); the vertex is
  // the last point written (the edge is visible there, or there is no T).
  if(next!=NONE&&kind(r)!=4u&&output>base&&tees<3u){let tee=throughT(endpoint^1u,next,output-1u-base);if(tee!=0u){teeWords[tees]=tee;tees++;}}
  if(next==NONE||next==start){break;}endpoint=next;
 }
 // Native runs are bounded by actual turns/cuts, not the triangles that
 // happened to tessellate either participant. Keep the reserved allocation
 // upper bound, but publish only the compacted span count to articulation.
 if(kind(r)==4u){output=compactContact(base,output);}
 let first=points[base].position;let last=points[output-1u].position;
 let delta=last.xy-first.xy;let height=high.y-low.y;let size=select(select(2u,1u,height<119.0),0u,height<51.0);
 var role=select(1u,2u,abs(delta.y)>abs(delta.x));
 if(role!=2u){let middle=(first.y+last.y)*.5;if(middle-low.y<height*.15){role=0u;}else if(middle-low.y>height*.78){role=3u;}}
 // A terrain run takes its role from what the line IS, never from where it
 // lies in the owner's screen box (a hill has no rim or base by position):
 // a contact or foot line is the return_base, a shelf edge (crease) the
 // rim_cap unless it runs steeper than 45 degrees, a ridge or skyline the
 // shoulder, and any line steeper than 45 degrees the flank.
 if(form==3u){
  let steep=abs(delta.y)>abs(delta.x);
  role=select(select(select(1u,2u,steep),select(0u,2u,steep),kind(r)==2u),3u,kind(r)==4u);
 }
 // The class of the run is the class of its segments, the outline taking the
 // silhouette's when any segment is one (one outline family, key).
 let lineClass=select(select(select(select(4u,select(1u,0u,silhouette),kind(r)==1u),3u,kind(r)==2u),0u,kind(r)==0u),2u,kind(r)==4u);
 // 0x8000000: the run's source is field-class (form 0), the class the tone
 // producer's texture edge acts on (tone-field.ts textureEdges, styleOf==0),
 // so the line producer's region-led clause agrees with it exactly.
 // 0x10000000: the run's source is terrain (form 3). Articulation selects it
 // by the terrain rows of its tier table and plays it from the terrain line
 // table. It is not field-class: a skyline is drawn, never left to texture.
 // A connected mixed-material line keeps ONE structural playback. Splitting
 // its incidence at a material change exposes new cap/flank runs and loses
 // the original construction. Only an entirely foliage-owned run can use
 // the foliage-specific flank passage; no source span is dropped.
 let flags=role|(lineClass<<8u)|select(0u,0x40000u,hard)|select(0u,0x80000u,upright)|select(0u,0x4000000u,(hard&&(shape(r)==0u||compiledOutline)&&hasTurn)||kind(r)==4u)|select(0u,0x8000000u,shape(r)==0u)|select(0u,0x10000000u,form==3u)|select(0u,0x20000000u,foliage)|select(0u,0x40000000u,partFar);
 let group=topologyGroup(r,r.ports[0]);
 let identity=(r.key.x*0x9e3779b9u)^(r.key.y*0x85ebca6bu)^r.key.w^group;
 // A run that is not a contact carries its owner's page rows (top, bottom,
 // each + 16384 in 16 bits) so articulation can tell which part of the form
 // each row of the run lies on. A contact keeps its two ports here.
 let rows=ownerRows(low,high);
 var contact=select(vec4u(0u,0u,rows,0u),vec4u(r.ports.zw,u32(r.coordinates.z),r.ports.x),kind(r)==4u);
 // A source vertex where exactly two hard edges of one source and one feature
 // family end unjoined is ONE junction, not two colliding run ends. Each run
 // records per end (first point in the low 16 bits, last point in the next):
 // bit 0 the junction exists, bit 1 this run spells it (one owner by original
 // source address), then the sixteenths in which its own leg and the other leg
 // leave the vertex (bits 2-5, 6-9), bit 10 natural, and for three legs the
 // third leg (bits 11-14) and bit 15. Every run at the node reads the same
 // node, so they agree.
 var joins=0u;
 for(var side=0u;side<2u;side++){
  let own=select(start,endpoint^1u,side==1u);let ownEdge=relations[own/2u];
  // A natural corner (two lines without the hard flag: terrain, rock) joins
  // too, through the natural join table (bit 10; compiled-role-grammar.ts
  // NATURAL_JUNCTIONS); a hard line joins only a hard line.
  if(kind(ownEdge)==4u||work[linkAt(own/2u)+(own&1u)]!=NONE||select(ownEdge.a.w,ownEdge.b.w,(own&1u)==1u)<.00001){continue;}
  let corner=nodeFor(own,LINE_NODE);if(corner==NONE){continue;}
  let degree=atomicLoad(&nodes[corner].degree);
  // Three legs (a T or a Y): ONE glyph from the three-leg table.
  if(degree==3u){joins|=threeLegs(own,corner)<<(side*16u);continue;}
  if(degree!=2u){continue;}
  let head=atomicLoad(&nodes[corner].second)-1u;let tail=work[linkAt(head/2u)+10u+(head&1u)]-1u;
  let mate=select(head,tail,own==head);let mateEdge=relations[mate/2u];
  // A face-owned mate belongs to the face construction: the runs simply meet (D1).
  if(mate/2u==own/2u||work[linkAt(mate/2u)+7u]==0u||work[linkAt(mate/2u)+(mate&1u)]!=NONE||hardEdge(mateEdge)!=hardEdge(ownEdge)||faceEdge(mateEdge)!=0u){continue;}
  if(!endVisible(own)||!endVisible(mate)){continue;}
  let here=endpointPage(own);var ownLeg=chord(here,own);var mateLeg=chord(here,mate);
  // A STEP: a level line ends where a parallel line one row away begins and
  // the mesh joins them by a near-upright edge shorter than one native row.

  // Y8 view-02 call-out 2026-11-12). The step is ONE corner glyph at the
  // connector's LOWER end, its upright reaching up into the upper line's row
  // (∟ ｣ by the compiled two-leg table); the connector's upper end spells
  // nothing. The connector's leg is its own edge, not its chain (the chain
  // runs on into the upper line and leans). Same rule from both runs.
  let ownSpan=endpointPage(own^1u)-here;let mateSpan=endpointPage(mate^1u)-here;
  let ownStep=dot(ownSpan,ownSpan)<289.0&&abs(ownSpan.x)<=.5*abs(ownSpan.y)&&abs(mateLeg.y)<=.5*abs(mateLeg.x)&&dot(mateLeg,mateLeg)>=289.0;
  let mateStep=dot(mateSpan,mateSpan)<289.0&&abs(mateSpan.x)<=.5*abs(mateSpan.y)&&abs(ownLeg.y)<=.5*abs(ownLeg.x)&&dot(ownLeg,ownLeg)>=289.0;
  if(ownStep||mateStep){
   // Page y grows down: the connector climbs from its lower end.
   if(select(mateSpan.y,ownSpan.y,ownStep)>=0.0){continue;}
   if(ownStep){ownLeg=ownSpan;}else{mateLeg=mateSpan;}
  }
  // A corner exists only with its two lines: a leg shorter than one native
  // unit spells nothing, and the glyph alone was a stray mark on every small
  // form. Then the two runs simply meet.
  else if(min(dot(ownLeg,ownLeg),dot(mateLeg,mateLeg))<289.0){continue;}
  joins|=(1u|select(0u,2u,choose(own,mate)==own)|(direction16(ownLeg)<<2u)|(direction16(mateLeg)<<6u)|select(0x400u,0u,hardEdge(ownEdge)))<<(side*16u);
 }
 // A T on this run's own line: part.w and, for a run that is not a contact,
 // paper.x and paper.y carry throughT words (articulation hardJunction).
 if(kind(r)!=4u){contact.x=teeWords.y;contact.y=teeWords.z;}

 // size in bits 0..1 and carries the selected source feature/role in the
 // high bits. Articulation reads it once for the whole run; absent/legacy
 // records leave the incumbent role path unchanged.

 // interval) joins no whole-part group and no limb pair: it can lead no
 // part and pair no tube, exactly as when it was not reserved at all.
 var sourceProgram=vec4u(0u);if(anyVisible){sourceProgram=sourcePartProgram(r);}var programmedSize=size;var correspondence=vec4u(0u);
 if(sourceProgram.x!=0u&&sourceProgram.x<=0x07ffffffu){
  let authored=sourceProgram.y;
  // The compact feature token is a collision-free source-program ordinal.
  // Its nonzero value is the presence bit; the full source ID and kind remain
  // in the resident packet for CPU/debug consumers.
  programmedSize=size|((authored&3u)<<2u)|select(0u,1u<<4u,(authored&0x100u)!=0u)|(sourceProgram.x<<5u);
  // Preserve source identity alongside the projected point packet. The
  // original halfedge is the stable subfeature; the two ports distinguish
  // source sides, and the part labels carry the compiled source lineage.
  // Projected contour coordinates are view-dependent and never enter this
  // temporal key. Occlusion may remove visible words but cannot rename it.
  correspondence=vec4u(r.key.w,r.ports.x,r.ports.y,sourceProgram.w);
 }

 // occluding contour on a compiled LIMB part carries that part's chain label
 // in paper.w (0 = no limb, the incumbent). Articulation pairs a tube's two
 // silhouettes by this identity (limbPairSource), not by field probes.

 // edge from a limb onto a fork, a cap or another chain (or onto a vertex
 // with no limb word) carries 0: its run keeps the incumbent pairing rather
 // than taking whichever endpoint happened to answer first.
 if(anyVisible&&kind(r)<=1u&&shape(r)==0u){let a=limbVertex(r,r.ports.x);let b=limbVertex(r,r.ports.y);contact.w=select(0u,a,a!=0u&&a==b);}
 runs[slot]=Run(vec4u(base,output-base,flags,programmedSize),vec4u(identity,r.key.x|(r.key.z&0x80000000u),r.key.y,identity),vec4u(r.key.w,r.key.z,joins,teeWords.x),contact,correspondence);
}
`;












/** The shader's source form of a compiled construction type: 1 planar, 2
 * rounded, 3 terrain (a height surface), 0 every other (path-led). Compared
 * as a string, so this holds whether or not the composition's type union
 * names 'terrain' yet. */
export function sourceForm(type:string):0|1|2|3{return type==='planar'?1:type==='rounded'?2:type==='terrain'?3:0;}
export const CONSTRUCTION_RUN_WORDS=20;
function sourceRunPipelinePlan(device:GPUDevice){
 const layout=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:'uniform'}},...[1,2,3,4].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:'read-only-storage' as const}})),...[5,6,7,8].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:'storage' as const}}))]});
 const module=(entryPoint:string)=>device.createShaderModule({label:'MAIN retained source construction runs '+entryPoint,code:isolateComputeEntry(GPU_CONSTRUCTION_RUNS_SHADER,entryPoint)}),pipeline=device.createPipelineLayout({bindGroupLayouts:[layout]});
 return {layout,plans:new Map(['initialize','incidence','facePacketReserve','authoredPairValidate','authoredPairReserveStage','connect','jump','finishJumps','roundReserve','anchors','assemble','canonPackets'].map(entryPoint=>[entryPoint,()=>({layout:pipeline,compute:{module:module(entryPoint),entryPoint}})]))};
}
function indexedRunPipelinePlan(device:GPUDevice){
 const layout=device.createBindGroupLayout({entries:[0,1,2,3,4,5].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:binding===0?'uniform':binding===4?'storage':'read-only-storage'} as const}))});
 const module=(entryPoint:string)=>device.createShaderModule({label:'MAIN GPU indexed source construction '+entryPoint,code:isolateComputeEntry(GPU_INDEXED_CONSTRUCTION_SHADER,entryPoint)}),pipeline=device.createPipelineLayout({bindGroupLayouts:[layout]});
 return {layout,plans:new Map(['clear','unionBoundaryPass','finalizeBoundary','classify'].map(entryPoint=>[entryPoint,()=>({layout:pipeline,compute:{module:module(entryPoint),entryPoint}})]))};
}
interface PreparedRunFamily {readonly layout:GPUBindGroupLayout;readonly pipelines:ReadonlyMap<string,GPUComputePipeline>}
interface PreparedSourceRunPipelines extends PreparedRunFamily {readonly indexed:PreparedRunFamily}
// Only immutable programs/layouts have device lifetime. Each instance still
// owns its topology arena, empty-plane buffer, pending work and disposal.
const sourceRunPrograms=new DeviceComputePrograms((device:GPUDevice)=>{
 const {layout,plans}=sourceRunPipelinePlan(device);return {resources:layout,plans};
});
const indexedRunPrograms=new DeviceComputePrograms((device:GPUDevice)=>{
 const {layout,plans}=indexedRunPipelinePlan(device);return {resources:layout,plans};
});
/** Owns GPU source topology scratch, never native output/publication. */
export class GpuConstructionRuns {
 private runExtent=65536;private pointExtent=262144;private hashExtent=0;
 private readonly layout:GPUBindGroupLayout;private readonly pipelines:ReadonlyMap<string,GPUComputePipeline>;




 private readonly spare=new Map<string,GPUBuffer>();private readonly active=new Set<GPUBuffer>();
 private readonly retiredWork:GPUBuffer[]=[];
 private encoding?:GPUCommandEncoder;
 private readonly styleSnapshot=new Map<number,{type:string;runProgram?:unknown;topologyRevision?:number;styleWord?:number}>();
 private readonly domainHeader=new Uint32Array(1);
 private readonly gpuIndexedParams:GPUBuffer[]=[];
 private readonly gpuIndexedClearPipeline:GPUComputePipeline;private readonly gpuIndexedUnionPipeline:GPUComputePipeline;
 private readonly gpuIndexedFinalizePipeline:GPUComputePipeline;private readonly gpuIndexedClassifyPipeline:GPUComputePipeline;
 private readonly gpuIndexedEmptyPlaneFaces:GPUBuffer;
 // INCREMENTAL GPU RESIDENCY (2026-11-12). One persistent arena holds the
 // anchor programs, the source table, one slot per resident source and each
 // encode's link scratch. A source that arrives or changes is compiled alone
 // and written to its own new slot and its own table row; its previous slot is
 // released after the frame that may still read it is submitted. Unchanged
 // sources are never repacked, copied or uploaded. Before, any change rebuilt
 // the whole packet and replaced the whole arena: on Y8 (439 sources) a 140 to
 // 165 MB buffer created and uploaded every ~6 s while part graphs arrived,
 // two arenas alive at once (load probe 2026-11-12).
 // When an allocation finds no room, the GPU copies every resident slot,
 // packed, into a new buffer: of the same size when the live words fill at
 // most three quarters of it (the room was fragments of replaced slots),
 // otherwise twice as large. Slot words are relative to their own base, so a
 // moved slot is copied as it is; only the table rows that address it change.
 // Nothing resident is uploaded again from the CPU.
 private arena?:{buffer:GPUBuffer;heap:ArenaHeap;anchors:number;table:{offset:number;capacity:number;used:number;rows:Uint32Array}};
 private readonly slots=new Map<number,{offset:number;words:number;row:number}>();
 private readonly retiring=new Set<number>();
 /** Slots replaced or retired this frame; reusable once it is submitted. */
 private readonly released:{offset:number;words:number}[]=[];
 private readonly gpuIndexedLayout:GPUBindGroupLayout;
 private limit():number{return Math.min(this.device.limits.maxStorageBufferBindingSize,this.device.limits.maxBufferSize);}
 private createArena(words:number):GPUBuffer{
  return this.device.createBuffer({label:ARENA_LABEL,size:Math.max(16,words*4),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
 }
 /** Words for one allocation; repacks, or grows the arena by half (at least
  * to live data plus a third), when no room. Doubling raised Y8's steady
  * MAIN GPU memory from about 694 MB to 869 MB (arena 342 MB). */
 private reserve(words:number):number{
  const arena=this.arena!,found=arena.heap.allocate(words);if(found!==undefined)return found;
  const table=arena.table,live=this.anchorContexts.length+table.capacity*STYLE_WORDS+[...this.slots.values()].reduce((n,slot)=>n+slot.words,0);
  const limit=Math.floor(this.limit()/4),before=arena.heap.capacity;let capacity=before;
  if((live+words)*4>capacity*3)capacity=Math.max(Math.ceil(capacity*3/2),Math.ceil((live+words)*4/3));
  capacity=Math.min(limit,capacity);
  if(live+words>capacity)throw Error('Resident construction graph needs source-sized GPU chunks');
  const heap=new ArenaHeap(capacity),buffer=this.createArena(capacity);
  const copy=this.encoding??this.device.createCommandEncoder({label:'MAIN construction arena repack'});let copied=0;
  const move=(from:number,count:number):number=>{const to=heap.allocate(count)!;if(count){copy.copyBufferToBuffer(arena.buffer,from*4,buffer,to*4,count*4);copied+=count*4;}return to;};
  arena.anchors=move(arena.anchors,this.anchorContexts.length);
  table.offset=heap.allocate(table.capacity*STYLE_WORDS)!;
  for(const slot of this.slots.values()){
   const at=slot.row*STYLE_WORDS,critical=table.rows[at+2]!-slot.offset;
   slot.offset=move(slot.offset,slot.words);table.rows[at+2]=slot.offset+critical;table.rows[at+4]=slot.offset;
  }
  // GPU-derived source slots may have been written earlier in THIS host
  // encoder. Their relocation must follow those writes in that encoder,
  // not race ahead of it in a private submission. The table and arriving
  // CPU slots have disjoint ranges and retain their normal queue uploads.
  if(!this.encoding)this.device.queue.submit([copy.finish()]);
  this.retiredWork.push(arena.buffer);arena.buffer=buffer;arena.heap=heap;
  // Ranges of the old buffer: released with it. A link slice still held by
  // an encode of the old buffer is dropped by its dispose (heap identity).
  this.released.length=0;
  this.writeWords(table.offset,table.rows);
  if(capacity>before)constructionArena.growths++;else constructionArena.repacks++;
  constructionArena.copiedBytes+=copied;
  return heap.allocate(words)!;
 }
 private writeWords(offset:number,data:Uint32Array,from=0,count=data.length-from):void{
  if(count>0)this.device.queue.writeBuffer(this.arena!.buffer,offset*4,data,from,count);
 }
 private writeRow(row:number):void{const table=this.arena!.table;this.writeWords(table.offset+row*STYLE_WORDS,table.rows,row*STYLE_WORDS,STYLE_WORDS);constructionArena.rows++;}
 /** The table row of a source address not yet in the table (same hash and
 * probe as the shader's sourceClass). The table is rebuilt, twice as large
 * when half its live rows would exceed it, before it is half full of live
 * and retired rows. */
 private claimRow(domain:number):number{
  let table=this.arena!.table;
  if((table.used+1)*2>table.capacity){
   let capacity=table.capacity;while((this.slots.size+1)*4>capacity)capacity*=2;
   const offset=this.reserve(capacity*STYLE_WORDS),rows=new Uint32Array(capacity*STYLE_WORDS);
   for(let row=0;row<capacity;row++)rows[row*STYLE_WORDS]=EMPTY_ROW;
   const old=table;table={offset,capacity,used:0,rows};this.arena!.table=table;
   for(const [resident,slot]of this.slots){
    let at=rowHash(resident,capacity);while(rows[at*STYLE_WORDS]!==EMPTY_ROW)at=(at+1)&(capacity-1);
    rows.set(old.rows.subarray(slot.row*STYLE_WORDS,slot.row*STYLE_WORDS+STYLE_WORDS),at*STYLE_WORDS);slot.row=at;table.used++;
   }
   this.writeWords(offset,rows);this.released.push({offset:old.offset,words:old.capacity*STYLE_WORDS});constructionArena.tables++;
  }
  let at=rowHash(domain,table.capacity);
  while(table.rows[at*STYLE_WORDS]!==EMPTY_ROW&&table.rows[at*STYLE_WORDS]!==RETIRED_ROW)at=(at+1)&(table.capacity-1);
  if(table.rows[at*STYLE_WORDS]===EMPTY_ROW)table.used++;
  return at;
 }
 /** Place already-compiled source programs; never visit source object graphs. */
 private reside(changed:readonly SourceConstructionClass[],dynamicWords:number):void{
  const compiled=changed.map(item=>{
   if('gpuIndexed' in item){const slot=gpuIndexedConstructionSlot(item.gpuIndexed);return {item,topology:new Uint32Array(slot.words),criticals:new Uint32Array(0)};}
   if(!item.runProgram)throw Error('Source runtime program is not registered; compile the source before drawing');
   const program=item.runProgram;
   return {item,topology:'topology' in program?program.topology:program.gpuTopology,criticals:program.criticalVertices};
  });
  if(!this.arena){
   // First size: what the arriving sources need, a quarter more for the part
   // graphs that follow them, and the link scratch of one encode.
   let rows=64;while((compiled.length+1)*4>rows)rows*=2;
   const needed=this.anchorContexts.length+rows*STYLE_WORDS+compiled.reduce((n,c)=>n+('length' in c.topology?c.topology.length:c.topology.words)+c.criticals.length,0)+dynamicWords,limit=Math.floor(this.limit()/4);
   if(needed>limit)throw Error('Resident construction graph needs source-sized GPU chunks');
   const sized=Math.min(limit,Math.ceil(needed*1.25)),heap=new ArenaHeap(sized),buffer=this.createArena(sized);
   const anchors=heap.allocate(this.anchorContexts.length)!,offset=heap.allocate(rows*STYLE_WORDS)!,table=new Uint32Array(rows*STYLE_WORDS);
   for(let row=0;row<rows;row++)table[row*STYLE_WORDS]=EMPTY_ROW;
   this.arena={buffer,heap,anchors,table:{offset,capacity:rows,used:0,rows:table}};
   this.writeWords(anchors,this.anchorContexts);this.writeWords(offset,table);
  }
  const table=()=>this.arena!.table;
  for(const domain of this.retiring){
   const slot=this.slots.get(domain);if(!slot||this.styleSnapshot.has(domain))continue;
   table().rows[slot.row*STYLE_WORDS]=RETIRED_ROW;this.writeRow(slot.row);
   this.released.push({offset:slot.offset,words:slot.words});this.slots.delete(domain);constructionArena.retired++;
  }
  this.retiring.clear();
  for(const {item,topology,criticals}of compiled){
   // The row first: a table rebuild may repack the arena, which moves only
   // slots already in this.slots.
   const row=this.slots.get(item.domain)?.row??this.claimRow(item.domain);
   const topologyWords='length' in topology?topology.length:topology.words;
   const words=topologyWords+criticals.length,offset=this.reserve(words),old=this.slots.get(item.domain);
   // The program is shared source geometry, independent of engine domains.
   // Bind only its scalar identity; all compiled words stay untouched.
   this.domainHeader[0]=item.domain;this.writeWords(offset,this.domainHeader);
   if('length' in topology)this.writeWords(offset+1,topology,1);
   else{
    const copy=this.encoding??this.device.createCommandEncoder({label:'MAIN compiled source program adoption'});
    copy.copyBufferToBuffer(topology.binding.buffer,(topology.binding.offset??0)+4,this.arena!.buffer,(offset+1)*4,(topologyWords-1)*4);
    if(!this.encoding)this.device.queue.submit([copy.finish()]);
   }
   this.writeWords(offset+topologyWords,criticals);
   if(old)this.released.push({offset:old.offset,words:old.words});
   this.slots.set(item.domain,{offset,words,row});
   // Form word: low byte construction form; bits8-11 WHOLE-source material
   // class. Per-face roles arrive on each relation, not this source-wide row.
   table().rows.set([item.domain,sourceForm(item.type)|((((item.styleWord??0)>>>6)&15)<<8),offset+topologyWords,criticals.length/2,offset],row*STYLE_WORDS);this.writeRow(row);
   constructionArena.placed++;constructionArena.slotBytes+=words*4;
  }
  const heap=this.arena!.heap;
  Object.assign(constructionArena,{sources:this.slots.size,arenaBytes:heap.capacity*4,usedBytes:(heap.capacity-heap.freeWords)*4,freeBytes:heap.freeWords*4});
 }
 static async create(device:GPUDevice,anchorContexts:Uint32Array,progress?:ComputePipelineProgress):Promise<GpuConstructionRuns>{
  const sourceProgress=progress===undefined?undefined:(p:number,stage?:string)=>progress(p*2/3,stage);
  const indexedProgress=progress===undefined?undefined:(p:number,stage?:string)=>progress(2/3+p/3,stage);
  const source=await sourceRunPrograms.prepare(device,sourceProgress);
  const indexed=await indexedRunPrograms.prepare(device,indexedProgress);
  return new GpuConstructionRuns(device,anchorContexts,{layout:source.resources,pipelines:source.pipelines,indexed:{layout:indexed.resources,pipelines:indexed.pipelines}});
 }
 constructor(private readonly device:GPUDevice,private readonly anchorContexts:Uint32Array,prepared?:PreparedSourceRunPipelines){
  // Resolve BOTH banks before allocating instance state. A failed source
  // bank must not trigger synchronous indexed compilation or leak a buffer.
  const source=prepared??(()=>{const {resources:layout,pipelines}=sourceRunPrograms.get(device);return {layout,pipelines};})();
  const indexed=prepared?.indexed??(()=>{const {resources:layout,pipelines}=indexedRunPrograms.get(device);return {layout,pipelines};})();
  this.layout=source.layout;this.pipelines=source.pipelines;
  this.gpuIndexedLayout=indexed.layout;this.gpuIndexedClearPipeline=indexed.pipelines.get('clear')!;this.gpuIndexedUnionPipeline=indexed.pipelines.get('unionBoundaryPass')!;
  this.gpuIndexedFinalizePipeline=indexed.pipelines.get('finalizeBoundary')!;this.gpuIndexedClassifyPipeline=indexed.pipelines.get('classify')!;
  this.gpuIndexedEmptyPlaneFaces=device.createBuffer({label:'MAIN GPU indexed source empty plane table',size:4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});device.queue.writeBuffer(this.gpuIndexedEmptyPlaneFaces,0,new Uint32Array([0xffffffff]));
 }
 private encodeGpuIndexed(pass:GPUComputePassEncoder,item:GpuIndexedSourceConstructionClass,slot:number,initialize:boolean):void{
   const input=item.gpuIndexed;if(!input)return;const compact=gpuIndexedQuadContext(input),p=this.device.createBuffer({label:'MAIN GPU indexed source construction params',size:64,usage:GPUBufferUsage.UNIFORM,mappedAtCreation:true});
   new Uint32Array(p.getMappedRange()).set([slot,input.domain,gpuConstructionTypeCode(input.kind),input.incidence.layout.edges,input.topology.indexBase,input.topology.indexCount,input.topology.vertexCapacity,input.incidence.layout.cornerEdge,input.incidence.layout.edgeOffsets,input.incidence.layout.edgeCorners,input.incidence.layout.fanOffsets,input.incidence.layout.fanCorners,input.planeRegionCount??0,input.planeFaces?1:0,compact?2:Number(input.indexPattern==='independent-quads'),0]);p.unmap();this.gpuIndexedParams.push(p);
   const group=this.device.createBindGroup({layout:this.gpuIndexedLayout,entries:[{binding:0,resource:{buffer:p}},{binding:1,resource:storageBinding(input.topology.indices)},{binding:2,resource:storageBinding(input.incidence.binding)},{binding:3,resource:storageBinding(input.pose.vertices)},{binding:4,resource:{buffer:this.arena!.buffer}},{binding:5,resource:input.planeFaces?storageBinding(input.planeFaces):{buffer:this.gpuIndexedEmptyPlaneFaces}}]});
   const dispatch=(pipeline:GPUComputePipeline,count:number)=>{pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.max(1,Math.ceil(count/64)));};
   if(initialize){
    dispatch(this.gpuIndexedClearPipeline,compact?Math.ceil(input.topology.indexCount/6/16):Math.max(input.topology.indexCount,input.topology.vertexCapacity));
    if(input.indexPattern!=='independent-quads'){
     const rounds=Math.ceil(Math.log2(Math.max(1,input.incidence.layout.edges)))+1;for(let i=0;i<rounds;i++)dispatch(this.gpuIndexedUnionPipeline,input.incidence.layout.edges);
     dispatch(this.gpuIndexedFinalizePipeline,input.topology.indexCount);
    }
   }
   // Posed crease classes can change, but boundary connectivity belongs to
   // the source generation and must never be rebuilt by a camera frame.
   dispatch(this.gpuIndexedClassifyPipeline,compact?input.topology.indexCount/6:input.topology.indexCount);
  }
  encode(encoder:GPUCommandEncoder,source:RelationVisibility,classes:readonly SourceConstructionClass[],rect:{x:number;y:number;width:number;height:number},width:number,height:number,extraRuns=0,field?:SurfaceField):ResidentConstructionRuns{
  this.encoding=encoder;
  const far=classes.flatMap(item=>'farForms' in item&&item.farForms?[item.farForms]:[]);
  classes=classes.filter(item=>!('farForms' in item&&item.farForms));
  const capacity=source.capacity,runCapacity=this.runExtent,pointCapacity=this.pointExtent;
  // Source capacity is NOT emitted construction size. Reserving three runs
  // per possible relation exceeded a binding on voxel before a single run
  // was emitted. GPU allocation counters grow complete compact output; an
  // exhausted attempt is never published or treated as a geometry quota.
  const limit=Math.min(this.device.limits.maxStorageBufferBindingSize,this.device.limits.maxBufferSize);
  let maxHash=1;while((maxHash*2+1)*32<=limit)maxHash*=2;
  let hashCapacity=1;while(hashCapacity<Math.max(capacity*2,this.hashExtent))hashCapacity*=2;
  hashCapacity=Math.min(hashCapacity,maxHash);
  // Residency, not view membership, owns the table. Alternating cameras or
  // material draw order cannot recompile unchanged source graphs each frame.
  // A part graph arriving from the compile worker for a resident source
  // changes that source's slot only, so it is placed at once (before, such
  // arrivals waited up to 45 frames for a whole-packet rebuild).
  const changed:SourceConstructionClass[]=[];
  for(const item of classes){const old=this.styleSnapshot.get(item.domain);if(item.type!==old?.type||item.runProgram!==old?.runProgram||item.topologyRevision!==old?.topologyRevision||item.styleWord!==old?.styleWord)changed.push(item);}
  const dynamicWords=capacity*(LINK_WORDS+JUMP_WORDS)+4;
  if(!this.arena||changed.length||this.retiring.size)this.reside(changed,dynamicWords);
  // A registration error must not mark a missing program as already placed.
  for(const item of changed)this.styleSnapshot.set(item.domain,{type:item.type,runProgram:item.runProgram,topologyRevision:item.topologyRevision,styleWord:item.styleWord});
  const workOffset=this.reserve(dynamicWords),arena=this.arena!,heap=arena.heap;
  const arrived=new Set(changed.map(item=>item.domain));
  // Registration initializes only arrivals. All sources' current-pose edge
  // classes share one pass; dispatch order and original source addresses stay
  // unchanged, without a Metal encoder boundary for every visible chunk.
  let indexedPass:GPUComputePassEncoder|undefined;
  try{for(const item of classes){const slot=this.slots.get(item.domain);if('gpuIndexed' in item&&slot){
    indexedPass??=encoder.beginComputePass({label:'MAIN GPU indexed source construction'});
    this.encodeGpuIndexed(indexedPass,item,slot.offset,arrived.has(item.domain));
  }}}finally{indexedPass?.end();}
  const linkA=workOffset,linkB=linkA+capacity*LINK_WORDS;
  const buffers:GPUBuffer[]=[];
  const allocate=(label:string,size:number)=>{if(size>this.device.limits.maxStorageBufferBindingSize)throw Error(`${label} needs source-sized chunks`);let b=this.spare.get(label);this.spare.delete(label);if(b&&b.size<size){b.destroy();b=undefined;}b??=this.device.createBuffer({label,size:Math.max(16,size),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});buffers.push(b);this.active.add(b);return b;};
  const runs=allocate('MAIN resident GPU construction runs',(runCapacity+extraRuns)*CONSTRUCTION_RUN_WORDS*4),points=allocate('MAIN resident GPU construction points',pointCapacity*CONSTRUCTION_POINT_WORDS*4),nodes=allocate('MAIN source-port incidence dictionary',(hashCapacity+1)*32),work=arena.buffer;
  const sourceStatus=allocate('MAIN retained source construction status',16);encoder.copyBufferToBuffer(source.status,0,sourceStatus,0,16);
  encoder.clearBuffer(nodes,hashCapacity*32,32);
  const groups=Math.max(1,Math.ceil(Math.max(hashCapacity,capacity,runCapacity)/64)),columns=Math.min(groups,this.device.limits.maxComputeWorkgroupsPerDimension);
  const dispatch=(name:string,read:number,write:number,amount:number,round=0)=>{
   const uniform=this.device.createBuffer({label:'MAIN source run dispatch',size:64,usage:GPUBufferUsage.UNIFORM,mappedAtCreation:true});buffers.push(uniform);
   const u=new Uint32Array(uniform.getMappedRange()),f=new Float32Array(u.buffer);u.set([capacity,pointCapacity,hashCapacity,write]);f.set([rect.x,rect.y,rect.width,rect.height],4);u.set([arena.table.offset,0,runCapacity,round],8);u.set([arena.table.capacity,read,columns,arena.anchors],12);uniform.unmap();
   const list=[uniform,source.lines,source.intervals,source.partitions,source.status,runs,points,nodes,work];
   const pipeline=this.pipelines.get(name)!,group=this.device.createBindGroup({layout:this.layout,entries:list.map((buffer,binding)=>({binding,resource:{buffer}}))});
   const pass=encoder.beginComputePass({label:`MAIN GPU construction ${name}`});pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.min(columns,Math.ceil(amount/64)),Math.ceil(Math.ceil(amount/64)/columns));pass.end();
  };
  dispatch('initialize',linkA,linkB,Math.max(hashCapacity,runCapacity));dispatch('incidence',linkA,linkB,capacity);dispatch('facePacketReserve',linkA,linkB,capacity);dispatch('authoredPairValidate',linkA,linkB,capacity);dispatch('authoredPairReserveStage',linkA,linkB,capacity);dispatch('connect',linkA,linkB,capacity);





  let rounds=0;for(let jump=1;jump<capacity*2;jump*=2){dispatch('jump',linkA,linkB,capacity,++rounds);}
  if(rounds&1)dispatch('finishJumps',linkA,linkB,capacity);
  dispatch('roundReserve',linkA,linkB,capacity);dispatch('anchors',linkA,linkB,capacity);dispatch('assemble',linkA,linkB,capacity);dispatch('canonPackets',linkA,linkB,runCapacity);
  if(far.length){
   if(!field)throw Error('Smallest source forms require their same-pose native field');
   new GpuSourceFarForms(this.device).encode(encoder,far,field,{runs,points,status:nodes,statusOffset:hashCapacity*32,capacity:runCapacity,pointCapacity},rect,buffer=>buffers.push(buffer));
  }
  let disposed=false;return {runs,points,status:nodes,statusOffset:hashCapacity*32,sourceStatus,capacity:runCapacity,pointCapacity,dispose:()=>{if(disposed)return;disposed=true;if(this.arena?.heap===heap)heap.release(workOffset,dynamicWords);for(const buffer of buffers){if(buffer===work)continue;if(!this.active.delete(buffer)){buffer.destroy();continue;}const before=this.spare.get(buffer.label);if(before&&before.size>=buffer.size)buffer.destroy();else{before?.destroy();this.spare.set(buffer.label,buffer);}}}};
 }
 /** Compact GPU counts, never source geometry or font fitting on the CPU. */
 grow(flags:number,points:number,runs:number,hashCapacity:number):void{
  const limit=Math.min(this.device.limits.maxStorageBufferBindingSize,this.device.limits.maxBufferSize);
  if(flags&~7)throw Error(`Unknown construction allocation flags ${flags}`);
  if(flags&1){const next=hashCapacity*2;if((next+1)*32>limit)throw Error('Source-port incidence exceeds the device binding extent');this.hashExtent=Math.max(this.hashExtent,next);}
  if(flags&2){const next=Math.ceil(points/256)*256;if(next*CONSTRUCTION_POINT_WORDS*4>limit)throw Error('Emitted construction points exceed the device binding extent');this.pointExtent=Math.max(this.pointExtent,next);}
  if(flags&4){const next=Math.ceil(runs/256)*256;if(next*CONSTRUCTION_RUN_WORDS*4>limit)throw Error('Emitted construction runs exceed the device binding extent');this.runExtent=Math.max(this.runExtent,next);}
 }
 /** Retire actual mesh revisions, never an offscreen camera's membership. */
 /** The slot and row are released at the next encode, not here: a pass
  * already recorded this frame may still read them. */
 retainSources(domains:ReadonlySet<number>):void{for(const domain of this.styleSnapshot.keys())if(!domains.has(domain)){this.styleSnapshot.delete(domain);this.retiring.add(domain);}}
 submitted():void{
  for(const buffer of this.retiredWork.splice(0))buffer.destroy();for(const buffer of this.gpuIndexedParams.splice(0))buffer.destroy();
  const heap=this.arena?.heap;for(const range of this.released.splice(0))heap?.release(range.offset,range.words);
  if(heap)Object.assign(constructionArena,{usedBytes:(heap.capacity-heap.freeWords)*4,freeBytes:heap.freeWords*4});

 }
 dispose():void{this.submitted();this.gpuIndexedEmptyPlaneFaces.destroy();this.arena?.buffer.destroy();this.arena=undefined;this.slots.clear();this.retiring.clear();this.released.length=0;this.styleSnapshot.clear();for(const buffer of this.spare.values())buffer.destroy();this.spare.clear();for(const buffer of this.active)buffer.destroy();this.active.clear();}
}
