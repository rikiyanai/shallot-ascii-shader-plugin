import {storageBinding,sameStorageBinding,type SourceStorageBinding} from './source-storage.ts';
import type {MaterialGeometryBatch,MaterialGeometryDraw} from './geometry.ts';
export type PosedSourceBatch=Pick<MaterialGeometryBatch,'vertices'|'attributes'|'metadata'|'gpuPose'>&{readonly capacity?:number;readonly storage?:{readonly identity:object;readonly vertices:GPUBufferBinding;readonly attributes:GPUBufferBinding;readonly metadata:GPUBufferBinding};readonly source:Pick<MaterialGeometryDraw,'vertexCapacity'|'indexBase'|'cullMode'|'instanceCapacity'|'ownerAuthority'|'nativeVisibleInstanceFrontier'>&{readonly material?:unknown;readonly bounds?:GPUBuffer;readonly visibleInstances?:GPUBuffer;readonly visibleInstanceOffset?:number;readonly visibleInstanceFrontier?:import('./source-instance-arena.ts').SourceInstanceFrontier;readonly posedRevision?:number;readonly physicalUniformProducerRows?:true}};
/** The engine's materialType publishes an explicit `vertex` member, including
 * undefined for its ordinary affine instance transform. A rest-planar mesh
 * alone is NOT this certificate: skinning, morphs and custom vertex functions
 * may bend its straight boundary runs. Unknown material owners stay fine. */
export function sourceMaterialIsAffine(material:unknown):boolean{
 return typeof material==='object'&&material!==null&&'vertex' in material&&material.vertex===undefined;
}
export function posedBatchIdentity(batch:PosedSourceBatch):object{return batch.storage?.identity??batch.vertices;}
/** The anchor projection dispatch enumerates the immutable camera frontier when
 * one exists. Otherwise it must cover the live transport slice, not the
 * producer's reusable instance-table capacity. `batch.capacity` is the number
 * of posed vertex records exposed by this same-encoder slice; dividing by the
 * source vertex stride gives the instance range the GPU can actually visit.
 * The GPU indirect count may be smaller and is still checked by the shader.
 * The declared capacity is only a ceiling for malformed/legacy batches. */
export function partAnchorInstanceCount(batch:Pick<PosedSourceBatch,'source'>&Partial<Pick<PosedSourceBatch,'capacity'>>):number{
 const frontier=batch.source.visibleInstanceFrontier;
 if(frontier)return frontier.indices.length;
 // A transport batch is already bounded to the source slice the current
 // encoder exposes. Never substitute the producer's reusable instance-table
 // capacity for that bounded extent.
 const vertexCapacity=batch.source.vertexCapacity,capacity=batch.capacity;
 const live=capacity!==undefined&&Number.isSafeInteger(capacity)&&capacity>0&&Number.isSafeInteger(vertexCapacity)&&vertexCapacity>0
  ?Math.max(1,Math.ceil(capacity/vertexCapacity)):undefined;
 const declared=batch.source.instanceCapacity;
 return live===undefined?Math.max(1,declared??1):Math.max(1,declared===undefined?live:Math.min(declared,live));
}
/** The number of native instance ordinals exposed by this transport slice.
 * Geometry transport capacity is a vertex-record count; derive the slice
 * extent from the same vertex stride rather than the renderer's reusable
 * instance-table capacity or a CPU visibility census. */
function nativeTransportInstanceCapacity(batch:PosedSourceBatch):number{
 const vertexCapacity=batch.source.vertexCapacity,capacity=batch.capacity;
 if(capacity!==undefined&&Number.isSafeInteger(capacity)&&capacity>0&&Number.isSafeInteger(vertexCapacity)&&vertexCapacity>0){
  return Math.max(1,Math.ceil(capacity/vertexCapacity));
 }
 return Math.max(1,batch.source.instanceCapacity??1);
}
/** Bytes for the complete owner-keyed anchor table, including its head words,
 * extent records and projected canonical sample rows. */
export function partAnchorTableBytes(headSlots:number,recordCapacity:number,canonicalCapacity:number):number{
 const canonicalBase=2+headSlots+recordCapacity*PART_ANCHOR_RECORD_WORDS;
 return (canonicalBase+canonicalCapacity*PART_ANCHOR_CANONICAL_WORDS)*4;
}
/** Bytes for the prepared canonical native-row arena. The bitset and prefix
 * are group×row bits/blocks; only `touchedRows` receive edge-ordinal payloads. */
export function partAnchorRowTableBytes(runCount:number,groupCount:number,rowCount:number,touchedRows:number):number{
 return sourcePartRowTableLayout(runCount,groupCount,rowCount,touchedRows).bytes;
}
/** Reusable row-payload capacity.  The rounded reserve is an allocation
 * policy, not a drawing quota: callers still report the exact required count
 * separately, and an over-limit reserve falls back to that exact count. */
export function partAnchorRowCapacity(required:number,runCount:number,groupCount:number,rowCount:number,maxBytes:number):number{
 for(const value of [required,runCount,groupCount,rowCount,maxBytes])if(!Number.isSafeInteger(value)||value<0)throw Error('Invalid native row capacity extent');
 if(required===0)return 0;
 const rounded=Math.max(required,Math.ceil(required*1.25/256)*256);
 return sourcePartRowTableLayout(runCount,groupCount,rowCount,rounded).bytes<=maxBytes?rounded:required;
}
import {isGpuIndexedResident,type ResidentRelationSource,type ResidentGpuIndexedSource} from './source-cache.ts';
import type {GpuIndexedRestCellChart,GpuIndexedCellOccupancy} from './gpu-indexed-source.ts';
import type {SurfaceField} from './surface-fields.ts';
import {SurfaceIntersections} from './surface-intersections.ts';
import {SubmissionExpiry} from './submission-expiry.ts';
import {sourceIncidenceReaders} from './source-incidence-residual.ts';
import {sourceResidencyEdgeCount,sourceResidencyVertexCount,sourceResidencyFaceCount} from './source-transfer.ts';
import {compileComputePipelines,prepareComputePipelines,type ComputePipelinePlan} from './gpu-pipeline-preparation.ts';
import {isolateComputeEntry} from './gpu-compute-program.ts';
import {FORM_DEPTH_WGSL,FORM_DEPTH_HEADER_ROW,FAR_STRIP_BELOW_ROWS,LOD_TIER_HOLD,NEAR_FROM_ROWS,formDepths} from './form-depths.ts';
import {SOURCE_PATH_REVERSED,SOURCE_PATH_PAIR_B,EDGE_LEVEL_UNIT,EDGE_LEVEL_DETAIL,EDGE_LEVEL_CHAINED,FACET_CREASE_LEAST_ROWS,TOPOLOGY_PART_CANONICAL_MAGIC,TOPOLOGY_PART_CANONICAL_VERSION,TOPOLOGY_PART_CANONICAL_HEADER_WORDS,TOPOLOGY_PART_CANONICAL_RECORD_WORDS,TOPOLOGY_PART_CANONICAL_SAMPLE_WORDS,TOPOLOGY_PART_CANONICAL_FLAG_COMPLETE} from './compiled-construction-topology.ts';
import {FAR_AGGREGATE_MEMBER_WGSL} from './gpu-source-far-forms.ts';
import {sourcePartRowTableLayout} from './gpu-playback-program.ts';
/** Current GPU-indexed CSRI packets carry incidence, not the MAIN planar/hull
 * bank directories consumed by sourcePlanarSpanAt/sourceHullAt. Their voxel
 * chart hierarchy remains the complete producer; an instance-bank prefix for
 * their registration capacity would allocate memory but select no coarse bank. */
export function supportsCompiledInstanceLevels(source:ResidentRelationSource):boolean{return !isGpuIndexedResident(source);}
export interface SurfaceGraphView {
  readonly clipFromWorld:Float32Array;readonly eye:readonly number[];
  readonly direction:readonly number[];readonly perspective:boolean;
  /** Native page extent, not the reduced underdrawing field extent. */
  readonly pageExtent?:readonly [number,number];
}

/** Complete posed relations precede camera visibility. A relation keeps its
 * two source ports and adjacent face addresses even when both ends are hidden. */
export {RELATION_WORDS} from './source-coordinates.ts';
import {RELATION_WORDS,RELATION_RECORD_WGSL,SOURCE_CHART_TAG,VOXEL_REST_CORNER_WGSL,VOXEL_REST_CORNER_ORDER} from './source-coordinates.ts';

export const LEVEL_PREFIX_WORDS_PER_RELATION=8;
export function levelPrefixRecordCount(instances:number):number{
 if(!Number.isSafeInteger(instances)||instances<0)throw Error('Invalid level prefix instance count');
 return Math.max(1,Math.ceil((3+instances*5)/LEVEL_PREFIX_WORDS_PER_RELATION));
}
export interface LevelPrefixReservation {readonly base:number;readonly records:number;readonly nextUsed:number;readonly nextCapacity:number}
export function levelPrefixReservation(lineCapacity:number,capacity:number,used:number,instances:number,locked:boolean):LevelPrefixReservation|undefined{
 for(const value of [lineCapacity,capacity,used,instances])if(!Number.isSafeInteger(value)||value<0)throw Error('Invalid level prefix reservation extent');
 const records=levelPrefixRecordCount(instances),nextUsed=used+records;
 if(nextUsed>capacity&&locked)return undefined;
 return {base:lineCapacity+used,records,nextUsed,nextCapacity:Math.max(capacity,nextUsed)};
}






export const VOXEL_CLUSTER_ROWS=1;



export const CONTACT_PATCH_POINTS=1023;










export interface ContactPatchPoint {readonly domain:number;readonly eid:number;readonly vertex:number;readonly id:number}
/** The texture rows of a contact corner table: one rgba32uint texel per row,
 * row 0 .x = the count, row i+1 = [domain, eid, vertex, id], sorted by
 * (domain, eid, vertex). Invalid ids and non-integer fields are dropped. */
export function contactPatchRows(points:readonly ContactPatchPoint[]):Uint32Array{
 const ok=(v:number)=>Number.isInteger(v)&&v>=0&&v<=0xffffffff;
 const used=points.filter(p=>ok(p.domain)&&ok(p.eid)&&ok(p.vertex)&&p.vertex<0x80000000&&ok(p.id)&&p.id<CONTACT_PATCH_POINTS)
  .sort((a,b)=>a.domain-b.domain||a.eid-b.eid||a.vertex-b.vertex||a.id-b.id);
 const rows=new Uint32Array((used.length+1)*4);rows[0]=used.length;
 used.forEach((p,i)=>rows.set([p.domain,p.eid,p.vertex,p.id],(i+1)*4));
 return rows;
}
export const SURFACE_RELATIONS_SHADER=/* wgsl */`
${VOXEL_REST_CORNER_WGSL}
struct Params {clip:mat4x4f,eye:vec4f,direction:vec4f,capacity:u32,domain:u32,vertexCapacity:u32,indexBase:u32,selection:vec4u,owners:array<vec4u,4>,frontier:vec4u,levelDispatch:vec4u}
${RELATION_RECORD_WGSL}
struct Status {count:atomic<u32>,overflow:atomic<u32>,nonmanifold:atomic<u32>,reserved:atomic<u32>}
@group(0) @binding(0) var<uniform> params:Params;
@group(0) @binding(1) var<storage,read> source:array<u32>;
@group(0) @binding(2) var<storage,read> positions:array<vec4u>;
@group(0) @binding(3) var<storage,read> attributes:array<vec4u>;
@group(0) @binding(4) var<storage,read> native:array<u32>;
@group(0) @binding(5) var<storage,read_write> relations:array<Relation>;
@group(0) @binding(6) var<storage,read_write> status:Status;
@group(0) @binding(7) var<storage,read_write> dispatch:array<u32>;
@group(0) @binding(8) var<storage,read_write> selectedOwners:array<u32>;
@group(0) @binding(10) var formDepths:texture_2d<u32>;
@group(0) @binding(11) var<uniform> detailWorldFromClip:mat4x4f;
${FORM_DEPTH_WGSL}
${FAR_AGGREGATE_MEMBER_WGSL}
// Temporary posed surface tensors, then vertex radial curvature. The same
// buffer is reused for ordered source-instance chunks; it is not a new
// scene graph, pixel image, glyph fitter or CPU readback.
@group(0) @binding(9) var<storage,read_write> differentials:array<vec4f>;
// Explicit producer-authorized, current-POSE quad contacts. This scratch is
// ordered per source; original topology, indices and vertex addresses remain.
@group(0) @binding(12) var<storage,read_write> contactWork:array<atomic<u32>>;
@group(0) @binding(13) var<storage,read> partCodes:array<u32>;
@group(0) @binding(14) var<storage,read_write> partLevels:array<atomic<u32>>;

// SOURCE CORNER, supplied by the host (SurfaceRelations.setContactPatches);
// never a screen, world or pose weld. rgba32uint, one texel a row; row 0 .x =
// the count n; row i+1 = [owner domain, owner eid, source vertex port, corner
// id], sorted by (domain, eid, vertex). An exact key lookup: the edge end's own
// canonical vertex port, no distance and no drawn pose.
@group(0) @binding(15) var contactPatches:texture_2d<u32>;

// parts, 6

// table after those vertices; its projected samples are written into the
// trailing region of the same owner table by the existing posed dispatch.
// The owner table record is the exact directory link for each projected path;
// no consumer is allowed to recover a path by a global sample range or page
// proximity.
@group(0) @binding(16) var<storage,read_write> partAnchorVertices:array<u32>;
@group(0) @binding(17) var<storage,read_write> partAnchorTable:array<atomic<u32>>;
// [records, canonical rows, overflow, candidates] for the live projected
// frontier. This is separate from relation Status because both producers run
// in the same compute pass and must not alias their counters.
@group(0) @binding(19) var<storage,read_write> partAnchorAlloc:array<atomic<u32>>;

// (restCellChart.occupancy, 'voxel-column-occupancy-v1'): W=ceil(Y/32) words
// per column, cell (x,y,z) is word (z*X+x)*W+(y>>5), bit (y&31), set = SOLID.
// Bound (and owners[3].y bit 32 set) only when the producer published it.
@group(0) @binding(18) var<storage,read> cellOccupancy:array<u32>;
// Native standard-renderer ownership is a same-submit read-only pair: the
// packed EID/producer-row list and the exact indirect record that gives its
// live count and firstInstance. The whole indirect buffer is bound because a
// storage binding offset need not satisfy the device's storage alignment;
// frontier.x carries the exact record word offset. MAIN never guesses an
// owner from geometry.
@group(0) @binding(20) var<storage,read> nativeMembers:array<vec4u>;
@group(0) @binding(21) var<storage,read> nativeIndirect:array<u32>;
fn contactCorner(owner:vec2u,port:u32)->u32{
 if(port>=0x80000000u){return 0u;}
 let n=textureLoad(contactPatches,vec2i(0,0),0).x;if(n==0u){return 0u;}
 var lo=0u;var hi=n;
 loop{if(lo>=hi){break;}let mid=(lo+hi)/2u;let k=textureLoad(contactPatches,vec2i(0,i32(mid+1u)),0);
  if(k.x<owner.x||(k.x==owner.x&&(k.y<owner.y||(k.y==owner.y&&k.z<port)))){lo=mid+1u;}else{hi=mid;}}
 if(lo>=n){return 0u;}
 let k=textureLoad(contactPatches,vec2i(0,i32(lo+1u)),0);
 if(k.x!=owner.x||k.y!=owner.y||k.z!=port){return 0u;}
 return min(k.w,${CONTACT_PATCH_POINTS-1}u)+1u;
}
struct Triangle {p:mat3x3f,n:mat3x3f,face:vec3f,owner:u32,producer:u32,chart:vec3f,charted:bool,material:u32}
${sourceIncidenceReaders('source')}
fn slot(v:u32,instance:u32)->u32{return instance*params.vertexCapacity+u32(i32(v)+bitcast<i32>(native[3]));}
fn position(v:u32,instance:u32)->vec3f{return bitcast<vec3f>(positions[slot(v,instance)].xyz);}
fn normal(v:u32,instance:u32)->vec3f{
 let oct=unpack2x16snorm(attributes[slot(v,instance)].x);
 var n=vec3f(oct,1.0-abs(oct.x)-abs(oct.y));
 if(n.z<0.0){n=vec3f((vec2f(1.0)-abs(n.yx))*select(vec2f(-1.0),vec2f(1.0),n.xy>=vec2f(0.0)),n.z);}
 return normalize(n);
}
fn halfedge(h:u32)->vec3u{return sourceHalfedge(h);}
fn same(a:vec3f,b:vec3f)->bool{return all(abs(a-b)<=vec3f(0.000001));}
fn vertexInDraw(v:u32)->bool{
 let shifted=i32(v)+bitcast<i32>(native[3]);if(shifted<0||shifted>=i32(params.vertexCapacity)){return false;}
 let incidence=sourceVertexRecord(v);for(var i=0u;i<incidence.w;i++){if(inDraw(sourceVertexCorner(v,i))){return true;}}return false;
}
fn vertexPort(v:u32,instance:u32)->u32{
 let incidence=sourceVertexRecord(v);var result=v;let p=position(v,instance);
 for(var i=0u;i<incidence.y;i++){let other=sourceVertexCandidate(v,i);if(other<result&&vertexInDraw(other)&&same(p,position(other,instance))){result=other;}}
 return result;
}
fn selected(instance:u32)->bool{
 if(params.selection.z!=0u){return selectedOwners[params.owners[3].z+instance]!=0u;}
 if(params.selection.y==0u){return true;}
 let first=halfedge(native[2]-params.indexBase).x;let owner=positions[slot(first,instance)].w;
 if(owner==0xffffffffu){return true;}
 var lo=0u;var hi=params.selection.x;
 while(lo<hi){let mid=(lo+hi)/2u;let value=selectedOwners[mid];if(value<owner){lo=mid+1u;}else{hi=mid;}}
 return lo<params.selection.x&&selectedOwners[lo]==owner;
}
// The source frontier is compacted after final form-depth parent admission.
// Direct sources use the whole filtered list; batched tensor chunks carry an
// ORDINAL start in frontier.y. The list pointer remains owners[2].w, and the
// compacted count remains at frontier.x; never compare the absolute list
// address with that count.
fn selectedCount()->u32{
 if(params.owners[3].x==0u){return native[1];}
 if(params.frontier.w==0u){return params.owners[2].z;}
 let total=selectedOwners[select(params.frontier.x,0u,nativeFrontier())];let ordinalStart=params.frontier.y;
 if(ordinalStart>=total){return 0u;}return min(params.owners[2].z,total-ordinalStart);
}
fn selectedInstance(ordinal:u32)->u32{if(params.owners[3].x!=0u){return selectedOwners[params.owners[2].w+ordinal];}return ordinal;}
const LEVEL_BANK_FINE=0u;const LEVEL_BANK_PRINCIPAL_FINE=1u;const LEVEL_BANK_PLANAR=2u;const LEVEL_BANK_OUTLINE=3u;const LEVEL_BANK_UNKNOWN_FINE=4u;
fn levelDispatchMode()->u32{return params.levelDispatch.y;}
fn levelDispatchBase()->u32{return params.levelDispatch.x;}
fn levelDispatchCapacity()->u32{return params.levelDispatch.z;}
fn levelOwnerCertificate()->u32{return params.levelDispatch.w;}
fn levelDispatchRelationWord(k:u32)->u32{
 let record=relations[levelDispatchBase()+k/8u];let lane=k%8u;
 if(lane<4u){if(lane==0u){return record.key.x;}if(lane==1u){return record.key.y;}if(lane==2u){return record.key.z;}return record.key.w;}
 if(lane==4u){return record.ports.x;}if(lane==5u){return record.ports.y;}if(lane==6u){return record.ports.z;}return record.ports.w;
}
fn levelDispatchEnabled()->bool{return levelDispatchMode()!=0u&&levelDispatchRelationWord(0u)!=0xffffffffu;}
fn levelDispatchWord(k:u32)->u32{return levelDispatchRelationWord(k);}
fn levelDispatchRecordCount()->u32{if(!levelDispatchEnabled()){return 0u;}return levelDispatchWord(0u);}
fn levelDispatchTotal()->u32{if(!levelDispatchEnabled()){return 0u;}return levelDispatchWord(1u);}
fn levelDispatchRecord(k:u32)->vec4u{let at=2u+k*4u;return vec4u(levelDispatchWord(at),levelDispatchWord(at+1u),levelDispatchWord(at+2u),levelDispatchWord(at+3u));}
fn levelContourBase()->u32{return 2u+levelDispatchRecordCount()*4u;}
fn levelContourStart(k:u32)->u32{return levelDispatchWord(levelContourBase()+k);}
fn levelContourTotal()->u32{return levelDispatchWord(levelContourBase()+levelDispatchRecordCount());}
fn levelContourFor(id:u32)->vec3u{
 let count=levelDispatchRecordCount();var lo=0u;var hi=count;
 while(lo<hi){let mid=(lo+hi)/2u;if(levelContourStart(mid)<=id){lo=mid+1u;}else{hi=mid;}}
 if(lo==0u){return vec3u(0xffffffffu);}
 let ordinal=lo-1u;let start=levelContourStart(ordinal);var end=levelContourTotal();if(ordinal+1u<count){end=levelContourStart(ordinal+1u);}
 if(id>=end){return vec3u(0xffffffffu);}
 let record=levelDispatchRecord(ordinal);return vec3u(record.x,id-start,record.w);
}
fn levelDispatchFor(id:u32)->vec4u{
 let count=levelDispatchRecordCount();var lo=0u;var hi=count;
 while(lo<hi){let mid=(lo+hi)/2u;let r=levelDispatchRecord(mid);if(r.z<=id){if(id<r.z+levelBankCount(r.y)){return r;}lo=mid+1u;}else{hi=mid;}}
 if(lo>0u){let r=levelDispatchRecord(lo-1u);if(id>=r.z&&id<r.z+levelBankCount(r.y)){return r;}}
 return vec4u(0xffffffffu);
}
fn levelDispatchStore(k:u32,value:u32){
 let at=levelDispatchBase()+k/8u;let lane=k%8u;var record=relations[at];
 if(lane<4u){if(lane==0u){record.key.x=value;}else if(lane==1u){record.key.y=value;}else if(lane==2u){record.key.z=value;}else{record.key.w=value;}}
 else if(lane==4u){record.ports.x=value;}else if(lane==5u){record.ports.y=value;}else if(lane==6u){record.ports.z=value;}else{record.ports.w=value;}
 relations[at]=record;
}
fn levelOwner(instance:u32)->vec2u{if(sourceEdgeCount()==0u){return vec2u(params.domain,0xffffffffu);}let e=edge(0u);return edgeOwner(halfedge(e.x).x,instance);}
fn levelOwnerCertified(instance:u32)->bool{
 let certificate=levelOwnerCertificate();
 if(certificate==0u||sourceEdgeCount()==0u){return false;}
 let owner=levelOwner(instance);let tagged=(owner.x&0x80000000u)!=0u;
 if(certificate==1u){return !tagged;}
 if(certificate==2u){return tagged&&nativeProducerRows();}
 return false;
}
fn levelBankFor(instance:u32)->u32{
 if(native[0]==0u||!levelOwnerCertified(instance)){return LEVEL_BANK_UNKNOWN_FINE;}
 if(planarProgramLevel(instance)){return LEVEL_BANK_PLANAR;}
 let owner=levelOwner(instance);
 if(outlineLevel(owner)){return LEVEL_BANK_OUTLINE;}
 if(principalLevel(owner)){return LEVEL_BANK_PRINCIPAL_FINE;}
 return LEVEL_BANK_FINE;
}
fn levelBankCount(bank:u32)->u32{
 var count=sourceBoundarySlots();
 if(bank==LEVEL_BANK_PLANAR){count=sourcePlanarSpanCount()+sourcePanelJoinCount();}
 if(bank==LEVEL_BANK_OUTLINE){count=sourceHullEdgeCount();}
 if(sourcePathSegmentCount()!=0u){count=max(count,sourceBoundarySlots());}
 return count;
}
fn prepareLevelDispatch(){
 if(levelDispatchMode()==0u){return;}
 let count=selectedCount();if(count>levelDispatchCapacity()){levelDispatchStore(0u,0xffffffffu);return;}
 var total=0u;var contourTotal=0u;
 for(var ordinal=0u;ordinal<count;ordinal++){
  let instance=selectedInstance(ordinal);let bank=levelBankFor(instance);levelDispatchStore(2u+ordinal*4u,instance);levelDispatchStore(3u+ordinal*4u,bank);levelDispatchStore(4u+ordinal*4u,total);levelDispatchStore(5u+ordinal*4u,ordinal);total+=levelBankCount(bank);
  levelDispatchStore(2u+count*4u+ordinal,contourTotal);
  if(bank!=LEVEL_BANK_PLANAR&&bank!=LEVEL_BANK_OUTLINE){contourTotal+=sourceFaceCount();}
 }
 levelDispatchStore(0u,count);levelDispatchStore(1u,total);
 levelDispatchStore(2u+count*5u,contourTotal);
 let limit=params.owners[3].w;let boundaryGroups=(total+63u)/64u;dispatch[0]=min(boundaryGroups,limit);dispatch[1]=max(1u,(boundaryGroups+limit-1u)/limit);dispatch[2]=1u;
 let contourGroups=(contourTotal+63u)/64u;dispatch[3]=min(contourGroups,limit);dispatch[4]=max(1u,(contourGroups+limit-1u)/limit);dispatch[5]=1u;
}
fn nativeFrontier()->bool{return (params.owners[3].y&64u)!=0u;}
fn nativeProducerRows()->bool{return (params.owners[3].y&128u)!=0u;}
fn nativeIndirectBase()->u32{return select(0u,params.frontier.x,nativeFrontier());}
fn nativeIndirectWord(index:u32)->u32{return nativeIndirect[nativeIndirectBase()+index];}
fn nativeOwnerAt(ordinal:u32)->vec2u{
 let member=nativeMembers[nativeIndirectWord(4u)+ordinal];let producer=nativeProducerRows();
 return vec2u(params.domain|select(0u,0x80000000u,producer),select(member.x,member.y,producer));
}
fn frontierOwner(instance:u32)->vec2u{
 if(nativeFrontier()){return nativeOwnerAt(native[4u]-nativeIndirectWord(4u)+instance);}
 let first=halfedge(native[2]-params.indexBase).x;let at=slot(first,instance);let entity=positions[at].w;let producer=entity==0xffffffffu;
 return vec2u(params.domain|select(0u,0x80000000u,producer),select(entity,attributes[at].w,producer));
}
// Final representation admission. The producer's frontier is the only list
// mutated: a parent-selected owner removes its replaced child instance, while
// an unadmitted/unknown owner remains in the incumbent list. The count lives
// in the existing word immediately before the list, so all later consumers
// (contacts, anchors, lines and interior tensors) read the same result.
@compute @workgroup_size(1) fn admitFrontier(){
 if(params.owners[3].x==0u||params.frontier.w==0u||native[0u]<3u||native[2u]<params.indexBase){return;}
 let countAt=select(params.frontier.x,0u,nativeFrontier());let listAt=params.owners[2].w;var write=0u;
 if(nativeFrontier()){
  let first=nativeIndirectWord(4u);let slice=native[4u];let total=nativeIndirectWord(1u);
  if(slice<first||slice-first>total){return;}
  let available=min(total-(slice-first),params.owners[2].z);
  for(var read=0u;read<available;read++){if(!formParentSelected(nativeOwnerAt(slice-first+read))){selectedOwners[listAt+write]=read;write++;}}
 }else{
  let count=selectedOwners[countAt];
  for(var read=0u;read<count;read++){
   let instance=selectedOwners[listAt+read];if(!formParentSelected(frontierOwner(instance))){selectedOwners[listAt+write]=instance;write++;}
  }
 }
 selectedOwners[countAt]=write;
}
fn edge(id:u32)->vec4u{return sourceEdge(id);}
const NO_CONTACT=0xffffffffu;
fn posedContacts()->bool{return (params.owners[3].y&1u)!=0u&&selectedCount()!=0u&&native[1]==1u&&native[2]==params.indexBase&&native[5]==0u;}
fn contactSlots()->u32{return params.owners[2].y;}
fn cellParts()->bool{return (params.owners[3].y&2u)!=0u;}
fn partBase()->u32{return contactSlots()*3u+sourceEdgeCount();}
fn edgePartBase()->u32{return contactSlots()*6u+sourceEdgeCount();}
fn partFlagsBase()->u32{return contactSlots()*6u+sourceEdgeCount()*2u;}

// the contact and part regions, cleared with them (contactClear).
fn clusterBase()->u32{return select(contactSlots()*3u+sourceEdgeCount(),contactSlots()*7u+sourceEdgeCount()*2u,cellParts());}
// Page position, in native ROWS, of a posed point (the same row the part pass
// uses); x is measured in rows too. Negative: behind the eye.
fn pageRowsAt(q:vec3f)->vec2f{
 let c=params.clip*vec4f(q,1.0);if(!(c.w>0.00001)){return vec2f(-1e30);}
 return vec2f((.5+.5*c.x/c.w)*f32(params.owners[1].w),(.5-.5*c.y/c.w)*f32(params.owners[2].x))/bitcast<f32>(params.owners[1].y);
}
// The cluster block, in cells (1, 2, 4, 8 or 16): the largest power of two
// whose cluster still projects within VOXEL_CLUSTER_ROWS of a native row at
// this point (the largest of the cell's three axis projections). A cell of a
// third of a row or more keeps its own edges (block 1).
fn voxelBlock(a:vec3f,s:f32,held:f32)->f32{
 if((params.owners[1].z&1u)==0u||params.eye.w==0.0||!(s>0.0)){return 1.0;}
 let p=pageRowsAt(a);if(p.x<-1e29){return 1.0;}
 var cell=0.0;
 for(var k=0u;k<3u;k++){var v=vec3f(0.0);v[k]=s;let q=pageRowsAt(a+v);if(q.x<-1e29){return 1.0;}cell=max(cell,length(q-p));}
 if(!(cell>0.0)){return 1.0;}

 // at a threshold; the rule is monotone in the projected cell size).
 if(held>=1.0&&voxelRuleBlock(cell/${LOD_TIER_HOLD})<=held&&held<=voxelRuleBlock(cell*${LOD_TIER_HOLD})){return held;}
 return voxelRuleBlock(cell);
}
fn voxelRuleBlock(cell:f32)->f32{
 var block=1.0;
 loop{if(block>=16.0||cell*block*2.0>${VOXEL_CLUSTER_ROWS}){break;}block*=2.0;}
 return block;
}
// The chunk's level of its previous build (persistent chart home header word 2;
// 0 = none: a fresh home, a scratch home, no chart).
fn heldChunkBlock()->f32{
 if(!persistentHome()||!chartOn()){return 0.0;}

 // not in a level home: during the vote the voted home is not yet known.
 let w=atomicLoad(&contactWork[params.owners[0].w+3u]);if(w==0u||w>5u){return 0.0;}
 return exp2(f32(5u-w));
}
// A cluster corner's port: never a source vertex id (bit 30), equal for every
// cluster edge meeting at that corner at the same block.
fn clusterPort(q:vec3f,block:f32)->u32{
 let bits=bitcast<vec3u>(q);return 0x40000000u|(contactHash(bits.x^contactHash(bits.y^contactHash(bits.z^bitcast<u32>(block))))&0x3fffffffu);
}
// The CLUSTER EDGE of a voxel chunk's silhouette or crease a-b: both ends
// snapped to the block grid of cell size |b-a| (a voxel edge is one cell).
// A collapsed edge (both ends one corner) is not produced; of the fine edges
// that snap to one cluster edge of one kind, exactly ONE is produced (a claim
// in the cluster words): the cluster edge is drawn once. Returns true when it
// handled the edge (emitted or dropped); block 1 is the source edge itself.
fn voxelClusterEmit(a:vec3f,b:vec3f,t:Triangle,kind:u32,address:u32,faces:vec2u,partFar:bool)->bool{
 let s=length(b-a);let block=voxelBlock((a+b)*.5,s,0.0);if(block<=1.0){return false;}
 let grid=block*s;let qa=floor(a/grid+vec3f(.5))*grid;let qb=floor(b/grid+vec3f(.5))*grid;
 if(all(qa==qb)){return true;}
 let pa=clusterPort(qa,block);let pb=clusterPort(qb,block);
 let key=(contactHash(min(pa,pb)^contactHash(max(pa,pb)^(kind*0x9e3779b9u)))|1u);
 let slots=contactSlots();let base=clusterBase();
 for(var probe=0u;probe<slots;probe++){
  let at=base+((key+probe)&(slots-1u));var held=atomicLoad(&contactWork[at]);
  // The same claim loop as addPartVertex: retry only a spurious weak failure.
  loop{if(held!=0u){break;}let claim=atomicCompareExchangeWeak(&contactWork[at],0u,key);held=claim.old_value;
   if(claim.exchanged){emit(qa,qb,t,kind,address,vec4u(pa,pb,faces),vec4f(0),partFar);return true;}}
  if(held==key){return true;}
 }
 return true;
}

// world-y columns, which read a radially deformed chunk's terraces and caves
// in the wrong frame). ONE level per chunk, selected BEFORE fine incidence.
// The chart is the producer's REST cell lattice (closed-cell parts: each
// vertex's cell x/y/z and face direction, all six faces of every cell). Per
// frame (pose-dependent): each quad stores its posed centroid under its cell
// and face; a cell's posed frame is its six face centroids (axis k = +k face
// - -k face, centre = their mean); a vertex's rest LATTICE POINT is its cell
// plus, per axis, the side of the centre it lies on (no mesher convention);
// the posed position of every lattice point that is a vertex is recorded.

// the CLUSTER occupancy, a cluster (block^3 cells) being occupied when any
// declared cell lies in it, with a representative cell and face. The coarse
// construction is the edges of that occupancy along all three chart axes,
// each classified from its four surrounding clusters (convex, concave, both
// for a diagonal pair, nothing when flat), its ends at the ACTUAL posed
// lattice points (a lattice point with no vertex: the affine step from the
// frame of the first occupied cluster around it, the same for every edge),
// its faces turned by the posed cell axes. Voids, caves and overhangs of a
// cluster or more stay empty clusters and draw their own edges. A fine edge
// is owned by the chart (it produces nothing; the first one reaching each
// cluster emits that cluster's share once) when every cluster around it is
// interior (not on the chunk's x/z border, where the chunk joins its
// neighbour); every other fine edge keeps the incumbent fine producer, as does
// a chunk at block 1, a source without parts, an unbounded chart and an
// overflow or an incomplete cell frame (chartUnsupported).
// Scratch, after the cluster claim words: 4 chunk words [block vote, cell
// size, retained, unsupported], then (chart) cells x 19 face words, corners x
// 4 lattice words, then the scratch home. A home: 4 header words [cluster
// capacity, stored token, stored block, spare], then clusters x 4 words
// [~least cell, ~least incidence, flags (bit 0 emitted), spare].
fn chartOn()->bool{return (params.owners[3].y&4u)!=0u;}
fn chunkBlockWord()->u32{return clusterBase()+contactSlots();}
fn retainedWord()->u32{return chunkBlockWord()+2u;}
fn overflowWord()->u32{return chunkBlockWord()+3u;}
fn chartCells()->u32{let e=partExtent();return select(0u,e.x*e.y*e.z,chartOn());}
fn chartCornerCount()->u32{let e=partExtent()+vec3u(1u);return select(0u,e.x*e.y*e.z,chartOn());}
fn chartClusters()->u32{let n=(partExtent()+vec3u(1u))/2u;return select(0u,n.x*n.y*n.z,chartOn());}
fn faceBase()->u32{return chunkBlockWord()+4u;}
fn cornerBase()->u32{return faceBase()+chartCells()*19u;}
fn scratchHome()->u32{return cornerBase()+chartCornerCount()*4u;}
fn persistentHome()->bool{return params.owners[0].w!=0u;}

// level homes from params.owners[0].w: blocks 2, 4, 8, 16, each 4 header words
// + 4 words per cluster of its OWN level (the host's chartLevelClusters /
// chartHomeWords). Each home keeps its own [capacity, token, block] header, so
// a level transition lands on that level's retained home and its resident
// records instead of rebuilding a single shared home. The base home's header
// word 3 holds the HELD level (the previous vote), read during the vote.
fn levelClusters(b:u32)->u32{let n=(partExtent()+vec3u(b-1u))/b;return n.x*n.y*n.z;}
fn levelHomeOffset(b:u32)->u32{var off=0u;var level=2u;loop{if(level>=b||level>16u){break;}off+=4u+levelClusters(level)*4u;level*=2u;}return off;}
// Generation-owned per-part frontier membership follows the source packet's
// complete canonical table. Each word is a certified level-node index (high
// bit set) or zero for mixed/border/seam/unsupported content. The host appends
// exactly partCount*5 words (four level nodes plus one frame marker), so this
// address never depends on a partial draw
// or a camera-sized home.
fn partFrontierCacheBase()->u32{return arrayLength(&partAnchorVertices)-partAnchorVertices[1u]*5u;}
fn partFrontierCacheAt(part:u32,ordinal:u32)->u32{return partFrontierCacheBase()+part*5u+ordinal;}
fn partFrontierMarkerAt(part:u32)->u32{return partFrontierCacheBase()+part*5u+4u;}
// The voted cluster level (vote code w: 4 = block 2 ... 1 = block 16); 0 for
// no cluster level this frame (no vote, block 1, an overflow is not read here).
fn votedClusterBlock()->u32{let w=atomicLoad(&contactWork[chunkBlockWord()]);if(w<1u||w>4u){return 0u;}return 1u<<(5u-w);}
fn homeBase()->u32{
 if(!persistentHome()){return scratchHome();}
 let b=votedClusterBlock();return params.owners[0].w+select(0u,levelHomeOffset(b),b>=2u);
}
fn clusterCapacity()->u32{if(persistentHome()){return atomicLoad(&contactWork[homeBase()]);}return chartClusters();}

// block b, not the voted one, so all four level homes are built (occupancy,
// classification, compiled records) when content changes, each kept or rebuilt
// under its own header. The scratch home (no persistent home) holds only the
// voted level. levelOrdinal: block 2 -> 0 ... 16 -> 3.
fn levelOrdinal(b:u32)->u32{return countTrailingZeros(b)-1u;}
fn levelHomeBase(b:u32)->u32{if(!persistentHome()){return scratchHome();}return params.owners[0].w+levelHomeOffset(b);}
fn levelCapacity(b:u32)->u32{if(persistentHome()){return atomicLoad(&contactWork[levelHomeBase(b)]);}return chartClusters();}
// A home exists for block b this draw: every level of a persistent source, only
// the voted level of a scratch home.
fn levelHomed(b:u32)->bool{return b>=2u&&b<=16u&&(persistentHome()||b==votedClusterBlock());}
// Per-draw retained decision of each level home (voxelHome), in scratch words
// after the selected list (cleared by contactClear every draw).
fn levelRetainedWord(b:u32)->u32{return selectedListBase()+2u+selectedListCapacity()+levelOrdinal(b);}
fn levelRetained(b:u32)->bool{return levelHomed(b)&&atomicLoad(&contactWork[levelRetainedWord(b)])!=0u;}
// The block code a level home stores (the vote code of that block: 2 -> 4 ... 16 -> 1).
fn levelCode(b:u32)->u32{return 5u-countTrailingZeros(b);}

// implementation"; CPU reference voxel-ancestor-cache.ts selectVoxelFrontier).
// Replaces the chunk-wide finest vote for a persistent source (every level is

// for a block (voxelBlock, its region's held level as hysteresis); chartQuad
// records the FINEST ask of every node containing it, one atomicMax per level
// into per-draw scratch words (cleared by contactClear). A node's requirement
// only gets finer as the node grows, so the levels L whose node admits L form
// a down-set: a cell's level is unique, the LARGEST such L, and cluster C is
// SELECTED at L exactly when its node admits L and its parent does not admit
// 2L. A node therefore takes its complete record XOR its children, with two
// word reads and no per-frame topology walk, and only quads inside a node can
// refine it: a near region never promotes unrelated far regions. A scratch
// home keeps the single voted level.
fn frontierOn()->bool{return persistentHome()&&chartOn();}
fn requirementBase()->u32{return selectedListBase()+2u+selectedListCapacity()+4u;}
fn requirementCount()->u32{return select(0u,levelClusters(2u)+levelClusters(4u)+levelClusters(8u)+levelClusters(16u),chartOn());}
fn requirementWord(b:u32,index:u32)->u32{var off=0u;var l=2u;loop{if(l>=b){break;}off+=levelClusters(l);l*=2u;}return requirementBase()+off+index;}
fn levelIndex(c:vec3i,b:u32)->u32{let n=vec3i((partExtent()+vec3u(b-1u))/b);return u32((c.z*n.y+c.y)*n.x+c.x);}
fn levelCoord(index:u32,b:u32)->vec3i{let n=(partExtent()+vec3u(b-1u))/b;return vec3i(i32(index%n.x),i32((index/n.x)%n.y),i32(index/(n.x*n.y)));}
fn levelInside(c:vec3i,b:u32)->bool{let n=vec3i((partExtent()+vec3u(b-1u))/b);return all(c>=vec3i(0))&&all(c<n);}
// The finest block any quad of node C (level b) asked for this draw; 32 = no ask.
// A forced node (unknown/cave/border in range, stamped at build) reads as
// requiring block 0: finer than every ask, so clusterAtLevel/cellLevel
// recurse exactly like the CPU reference (voxel-ancestor-cache.ts).
fn nodeRequired(c:vec3i,b:u32)->u32{
 if(!levelInside(c,b)){return 32u;}
 if(clusterForced(c,b)){return 0u;}
 let w=atomicLoad(&contactWork[requirementWord(b,levelIndex(c,b))]);if(w==0u||w>5u){return 32u;}return 1u<<(5u-w);
}
fn clusterAtLevel(c:vec3i,b:u32)->bool{
 if(!frontierOn()){return b>=2u&&b==votedClusterBlock();}
 if(b<2u||b>16u||!levelInside(c,b)||nodeRequired(c,b)<b){return false;}
 return b==16u||nodeRequired(c/2,2u*b)<2u*b;
}
// Node C admits its own level b (the first half of clusterAtLevel; no parent read).
fn clusterOwnLevel(c:vec3i,b:u32)->bool{
 if(!frontierOn()){return b>=2u&&b==votedClusterBlock();}
 return b>=2u&&b<=16u&&levelInside(c,b)&&nodeRequired(c,b)>=b;
}
// The frontier's level at an in-grid cell (0 = fine: block 1).
fn cellLevel(cell:vec3i)->u32{
 if(!frontierOn()){return select(0u,votedClusterBlock(),votedClusterBlock()>=2u);}
 for(var b=16u;b>=2u;b/=2u){if(nodeRequired(cell/i32(b),b)>=b){return b;}}
 return 0u;
}
// Region seams: a declared cluster belongs to the level only when the frontier
// selected it there; an undeclared (empty) cluster is level-agnostic, so a
// silhouette against empty space still coarsens. Declared clusters of two
// different levels meeting at a seam keep their fine segments on BOTH sides
// and emit no coarse edge there: never drawn twice, never joined by fiction.
fn clusterLevelOk(c:vec3i,b:u32)->bool{return !clusterDeclared(c,b)||clusterAtLevel(c,b);}
// HOLD per region: the level-2 cluster word 3 bits 28..30 keep the level code
// its cells took last draw (1 block 16 ... 4 block 2, 5 fine; 0 none), written
// by chartCertify after the frontier is complete and read by the next draw's
// asks (LOD_TIER_HOLD band, voxelBlock). Cleared with a rebuilt home.
const CLUSTER_HELD_SHIFT=28u;const CLUSTER_HELD_MASK=0x70000000u;
fn heldCellBlock(cell:vec3i)->f32{
 if(!frontierOn()){return heldChunkBlock();}
 let s=clusterSlot(cell/2,2u);if(s==NONE_CLUSTER){return 0.0;}
 let code=(atomicLoad(&contactWork[s+3u])&CLUSTER_HELD_MASK)>>CLUSTER_HELD_SHIFT;if(code==0u||code>5u){return 0.0;}
 return exp2(f32(5u-code));
}

// that stores holds, one thread per cluster index across all levels, so on a
// home-rebuild draw an ancestor's bit may not be stamped yet (no ordering
// between workgroups). Reading it there gave a schedule-dependent, too-coarse
// hold. Store a hold only when every ancestor node cellLevel reads is stamped
// (EDGES_DONE and FORCED are set by one atomicOr); otherwise clear it (code 0 =
// no hold: the next draw's ask has no hysteresis, then holds resume).
fn storeHeld(s:u32,c2:vec3i){
 for(var b=4u;b<=16u;b*=2u){let a=clusterSlot((c2*2)/i32(b),b);if(a!=NONE_CLUSTER&&(atomicLoad(&contactWork[a+3u])&CLUSTER_EDGES_DONE)==0u){atomicAnd(&contactWork[s+3u],~CLUSTER_HELD_MASK);return;}}
 let level=cellLevel(c2*2);let code=select(5u,levelCode(level),level>=2u);
 atomicAnd(&contactWork[s+3u],~CLUSTER_HELD_MASK);atomicOr(&contactWork[s+3u],code<<CLUSTER_HELD_SHIFT);
}
// The level a fine segment va-vb would be represented at: the frontier level
// of the first DECLARED cell whose closure holds its midpoint (0: none / fine).
fn segmentLevel(va:u32,vb:u32)->u32{
 if(!chartOn()||atomicLoad(&contactWork[overflowWord()])!=0u){return 0u;}
 let la=storedLattice(va);let lb=storedLattice(vb);if(la.w==0||lb.w==0){return 0u;}
 let cells=segmentClusters(la.xyz,lb.xyz,1u);
 for(var k=0;k<8;k++){if(cells[k].w!=0&&faceMask(cells[k].xyz)!=0u){return cellLevel(cells[k].xyz);}}
 return 0u;
}

// (x | y<<9 | z<<18, plus 1; 0 = none), written by chartCorners (which binds
// the cell codes) so the boundary stage reads lattice points WITHOUT binding
// partCodes (it stays within 8 storage buffers).
fn chartVertices()->u32{return select(0u,sourceFaceCount()*2u,chartOn());}
fn latticeBase()->u32{return scratchHome()+4u+chartClusters()*4u;}

// the rest of the scratch): [count, overflow, then chartClusters() cluster
// indices]. chartSelect appends a cluster exactly once, when its own atomicOr
// first sets CLUSTER_SELECTED; chartEmitAt visits only these clusters instead
// of scanning every cluster of the selected level. An index past the list's
// capacity (a persistent home larger than this frame's chart) sets overflow,
// and chartEmitAt then falls back to the full scan: never a dropped cluster.
fn selectedListBase()->u32{return latticeBase()+chartVertices();}
fn selectedListCapacity()->u32{return chartClusters();}
// The last scratch word is a per-draw completion marker for the ordered
// chartSelect -> chartEmit pair. It lives in the same contact arena that
// contactClear owns, so a chart cannot be mistaken for the previous view.
fn chartEarlyEmittedWord()->u32{return selectedListBase()+select(0u,2u+selectedListCapacity()+4u+requirementCount(),chartOn());}
fn chartEarlyEmitted()->bool{return chartOn()&&atomicLoad(&contactWork[chartEarlyEmittedWord()])!=0u;}
fn columnEnd()->u32{return chartEarlyEmittedWord()+select(0u,1u,chartOn());}
// R51 residual domain: the chart-cluster list is deliberately not reused for
// fine source IDs. Its bounded tail holds [edgeCount,vertexCount,overflow,
// faceCount], a compact edge list, a vertex membership bitset, and a compact
// vertex list. The edge/vertex output capacities are the chart's cluster
// count; overflow keeps the complete incumbent consumer domains.
fn residualHeader()->u32{return columnEnd();}
// Bit 6 of the producer's chart flags authorizes the optional residual suffix.
fn residualOn()->bool{return chartOn()&&(params.owners[3].y&64u)!=0u;}
fn residualEdgeCapacity()->u32{return select(0u,chartClusters(),residualOn());}
fn residualVertexCapacity()->u32{return select(0u,chartClusters(),residualOn());}
fn residualEdgeBase()->u32{return residualHeader()+4u;}
fn residualVertexBitsBase()->u32{return residualEdgeBase()+residualEdgeCapacity();}
fn residualVertexBits()->u32{return select(0u,(sourceVertexCount()+31u)/32u,residualOn());}
fn residualVertexBase()->u32{return residualVertexBitsBase()+residualVertexBits();}
fn residualFaceBitsBase()->u32{return residualVertexBase()+residualVertexCapacity();}
fn residualFaceBits()->u32{return select(0u,(sourceFaceCount()+31u)/32u,residualOn());}
fn residualFaceBase()->u32{return residualFaceBitsBase()+residualFaceBits();}
fn residualFaceCapacity()->u32{return select(0u,chartClusters(),residualOn());}
fn residualEnd()->u32{return select(columnEnd(),residualFaceBase()+residualFaceCapacity(),residualOn());}
fn residualEdgeCount()->u32{return atomicLoad(&contactWork[residualHeader()]);}
fn residualVertexCount()->u32{return atomicLoad(&contactWork[residualHeader()+1u]);}
fn residualOverflow()->bool{return atomicLoad(&contactWork[residualHeader()+2u])!=0u;}
fn residualReady()->bool{return residualOn()&&chartEarlyEmitted()&&!residualOverflow();}
fn residualEdgeAt(i:u32)->u32{return atomicLoad(&contactWork[residualEdgeBase()+i]);}
fn residualVertexAt(i:u32)->u32{return atomicLoad(&contactWork[residualVertexBase()+i]);}
fn residualFaceCount()->u32{return atomicLoad(&contactWork[residualHeader()+3u]);}
fn residualFaceAt(i:u32)->u32{return atomicLoad(&contactWork[residualFaceBase()+i]);}
fn residualOverflowMark(){atomicStore(&contactWork[residualHeader()+2u],1u);}
fn residualEdgeAppend(id:u32){if(!residualOn()){return;}let at=atomicAdd(&contactWork[residualHeader()],1u);if(at<residualEdgeCapacity()){atomicStore(&contactWork[residualEdgeBase()+at],id);}else{residualOverflowMark();}}
fn residualVertexAppend(v:u32){if(!residualOn()||v>=sourceVertexCount()){if(residualOn()){residualOverflowMark();}return;}let word=residualVertexBitsBase()+v/32u;let mask=1u<<(v%32u);let old=atomicOr(&contactWork[word],mask);if((old&mask)==0u){let at=atomicAdd(&contactWork[residualHeader()+1u],1u);if(at<residualVertexCapacity()){atomicStore(&contactWork[residualVertexBase()+at],v);}else{residualOverflowMark();}}}
fn residualFaceAppend(face:u32){
 if(!residualOn()||face>=sourceFaceCount()){if(residualOn()){residualOverflowMark();}return;}
 let word=residualFaceBitsBase()+face/32u;let mask=1u<<(face%32u);let old=atomicOr(&contactWork[word],mask);
 if((old&mask)==0u){let at=atomicAdd(&contactWork[residualHeader()+3u],1u);if(at<residualFaceCapacity()){atomicStore(&contactWork[residualFaceBase()+at],face);}else{residualOverflowMark();}}
}
fn residualDispatchBase()->u32{return 0u;}
fn storedLattice(v:u32)->vec4i{
 if(v>=chartVertices()){return vec4i(0);}let w=atomicLoad(&contactWork[latticeBase()+v]);if(w==0u){return vec4i(0);}
 let p=w-1u;return vec4i(i32(p&511u),i32((p>>9u)&511u),i32((p>>18u)&511u),1);
}
fn chartUnsupported(){atomicStore(&contactWork[overflowWord()],1u);}
fn voxelChunkBlock()->f32{
 if(!chartOn()||atomicLoad(&contactWork[overflowWord()])!=0u){return 1.0;}
 let w=atomicLoad(&contactWork[chunkBlockWord()]);if(w==0u){return 1.0;}return exp2(f32(5u-min(w,5u)));
}
// Per quad (its first vertex): the chunk's finest block and its cell size.
fn voxelColumnVertex(v:u32){
 if(v%4u!=0u){return;}
 let a=position(v,0u);let b=position(v+1u,0u);let s=distance(a,b);if(!(s>0.0)){return;}
 let centre=(a+position(v+2u,0u))*.5;let block=voxelBlock(centre,s,heldChunkBlock());
 atomicMax(&contactWork[chunkBlockWord()],5u-u32(round(log2(block))));
 atomicMax(&contactWork[chunkBlockWord()+1u],formOrdered(s));
}

// CLOSED (parts.kind 'closed-cells', unchanged): every declared cell has all
// six faces, so a cluster with no declared cell is empty, as is space outside
// the grid. SURFACE (restCellChart of a surface-only mesh: it does NOT promise
// six faces): a declared cell (one emitting a face) is solid and the cell
// beyond each emitted face is air; nothing else is declared. An undeclared
// cluster is uniform (a solid cell beside an air cell would have emitted a
// face) and so is every straight run of undeclared cells, so the first
// declared cell q that a run from the cluster reaches decides it: AIR when q
// emits the face back toward the run, SOLID when it does not. chartClassify
// stores that state (cluster word 3: 1 air, 2 solid, 3 unknown) when the
// home is rebuilt; it is generation-owned like the claims. A cluster no run
// resolves, whose runs disagree, or outside the grid is UNKNOWN, never empty:
// the chart owns no fine edge whose replacement would read it (the fine
// producer keeps it). A surface cell's posed frame uses its opposite face
// pairs, else the same face of its neighbour along that axis; one missing axis
// (the face normal of a flat run) is the cross product of the other two
// (orientation-preserving pose); two missing: no frame, its vertices keep the
// fine producer. A surface vertex's offset along its own face axis is the
// face's side, exactly.
fn chartSurface()->bool{return (params.owners[3].y&8u)!=0u;}
const CLUSTER_AIR=1u;const CLUSTER_SOLID=2u;const CLUSTER_UNKNOWN=3u;
fn faceMask(c:vec3i)->u32{if(any(c<vec3i(0))||any(c>=vec3i(partExtent()))){return 0u;}return atomicLoad(&contactWork[faceBase()+cellIndex(vec3u(c))*19u+18u]);}
fn faceStep(g:u32)->vec3i{return unit(g/2u)*select(-1,1,g%2u==0u);}
fn faceCentre(c:vec3i,g:u32)->vec4f{
 if((faceMask(c)&(1u<<g))==0u){return vec4f(0.0);}let at=faceBase()+cellIndex(vec3u(c))*19u+g*3u;
 return vec4f(bitcast<f32>(atomicLoad(&contactWork[at])),bitcast<f32>(atomicLoad(&contactWork[at+1u])),bitcast<f32>(atomicLoad(&contactWork[at+2u])),1.0);
}
// The posed image of chart axis k at surface cell c (w 0: none).
fn surfaceAxis(c:vec3i,k:u32)->vec4f{
 let p=faceCentre(c,2u*k);let m=faceCentre(c,2u*k+1u);if(p.w!=0.0&&m.w!=0.0){return vec4f(p.xyz-m.xyz,1.0);}
 let e=unit(k);
 for(var g=0u;g<6u;g++){
  if(g/2u==k){continue;}let a=faceCentre(c,g);if(a.w==0.0){continue;}
  let up=faceCentre(c+e,g);if(up.w!=0.0){return vec4f(up.xyz-a.xyz,1.0);}
  let down=faceCentre(c-e,g);if(down.w!=0.0){return vec4f(a.xyz-down.xyz,1.0);}
 }
 return vec4f(0.0);
}
fn surfaceFrame(c:vec3u)->Frame{
 let ci=vec3i(c);let mask=faceMask(ci);if(mask==0u){return Frame(false,vec3f(0.0),mat3x3f());}
 var a:array<vec3f,3>;var known=0u;
 for(var k=0u;k<3u;k++){let x=surfaceAxis(ci,k);if(x.w!=0.0){a[k]=x.xyz;known|=1u<<k;}}
 if(known!=7u){
  var missing=3u;for(var k=0u;k<3u;k++){if((known&(1u<<k))==0u){if(missing!=3u){return Frame(false,vec3f(0.0),mat3x3f());}missing=k;}}
  let n=cross(a[(missing+1u)%3u],a[(missing+2u)%3u]);let l=length(n);if(!(l>0.0)){return Frame(false,vec3f(0.0),mat3x3f());}
  a[missing]=n/l*sqrt(l);
 }
 var centre=vec3f(0.0);var count=0.0;
 for(var g=0u;g<6u;g++){let f=faceCentre(ci,g);if(f.w==0.0){continue;}centre+=f.xyz-a[g/2u]*select(-.5,.5,g%2u==0u);count+=1.0;}
 return Frame(true,centre/count,mat3x3f(a[0],a[1],a[2]));
}
// One straight run from cell m along face direction g: the state of m's run.
fn runState(m:vec3i,g:u32)->u32{
 let e=vec3i(partExtent());let step=faceStep(g);var q=m;
 for(var n=0;n<256;n++){
  q+=step;if(any(q<vec3i(0))||any(q>=e)){return CLUSTER_UNKNOWN;}
  let mask=faceMask(q);if(mask!=0u){return select(CLUSTER_SOLID,CLUSTER_AIR,(mask&(1u<<(g^1u)))!=0u);}
 }
 return CLUSTER_UNKNOWN;
}
fn classifyCluster(index:u32,b:u32){
 let n=(partExtent()+vec3u(b-1u))/b;let s=levelHomeBase(b)+4u+index*4u;if(atomicLoad(&contactWork[s])!=0u){return;}
 let m=vec3i(i32(index%n.x),i32((index/n.x)%n.y),i32(index/(n.x*n.y)))*i32(b);
 var state=0u;
 for(var g=0u;g<6u;g++){let r=runState(m,g);if(r==CLUSTER_UNKNOWN){continue;}if(state==0u){state=r;}else if(state!=r){state=CLUSTER_UNKNOWN;break;}}
 atomicStore(&contactWork[s+3u],select(CLUSTER_UNKNOWN,state,state!=0u));
}
// In chartCorners (after voxelBin completed every face mask and claim), by
// every vertex thread over the clusters; skipped when the home is retained.
fn chartClassifyAt(t:u32,threads:u32){
 if(!chartSurface()){return;}

 for(var b=2u;b<=16u;b*=2u){
  if(!levelHomed(b)||levelRetained(b)){continue;}
  let total=min(levelClusters(b),levelCapacity(b));
  for(var i=t;i<total;i+=max(threads,1u)){classifyCluster(i,b);}
 }
}
fn cellOf(v:u32)->vec3u{let c=partCodes[v];return vec3u(c&255u,(c>>8u)&255u,(c>>16u)&255u);}
fn faceOf(v:u32)->u32{return (partCodes[v]>>24u)&255u;}
fn cellIndex(c:vec3u)->u32{let e=partExtent();return (c.z*e.y+c.y)*e.x+c.x;}
fn latticeIndex(l:vec3i)->u32{let e=vec3i(partExtent())+vec3i(1);return u32((l.z*e.y+l.y)*e.x+l.x);}
fn latticeInside(l:vec3i)->bool{return all(l>=vec3i(0))&&all(l<=vec3i(partExtent()));}
// Per quad (its first edge; after the vote): its posed centroid under its cell
// and face (every frame: it is pose-dependent), and, unless the home is
// retained, its cell's cluster.
fn chartQuad(id:u32){
 if(!chartOn()){return;}
 let v=(halfedge(edge(id).x).x/4u)*4u;let c=cellOf(v);let f=faceOf(v);
 if(any(c>=partExtent())||f>5u){chartUnsupported();return;}
 let centre=(position(v,0u)+position(v+1u,0u)+position(v+2u,0u)+position(v+3u,0u))*.25;
 let at=faceBase()+cellIndex(c)*19u;
 atomicStore(&contactWork[at+f*3u],bitcast<u32>(centre.x));atomicStore(&contactWork[at+f*3u+1u],bitcast<u32>(centre.y));atomicStore(&contactWork[at+f*3u+2u],bitcast<u32>(centre.z));
 atomicOr(&contactWork[at+18u],1u<<f);

 // retained (a persistent source prebuilds all four, whatever this frame's
 // vote; a scratch home builds only the voted level).

 // region's last level) refines every node that contains it, every draw.
 if(frontierOn()){
  let s=distance(position(v,0u),position(v+1u,0u));
  let ask=voxelBlock(centre,s,heldCellBlock(vec3i(c)));let w=5u-u32(round(log2(ask)));
  for(var b=2u;b<=16u;b*=2u){atomicMax(&contactWork[requirementWord(b,levelIndex(vec3i(c/vec3u(b)),b))],w);}
 }
 let corner=~sourceEdgeCorner(halfedge(edge(id).x).z,0u);
 for(var b=2u;b<=16u;b*=2u){
  if(!levelHomed(b)||levelRetained(b)){continue;}
  let slot=clusterSlot(vec3i(c/vec3u(b)),b);if(slot==NONE_CLUSTER){chartUnsupported();return;}
  atomicMax(&contactWork[slot],~cellIndex(c));atomicMax(&contactWork[slot+1u],corner);
 }
}
const NONE_CLUSTER=0xffffffffu;
// The words of cluster C at block b (dense over the cluster grid), or NONE
// outside the grid (open space above or below a chunk is simply empty).
fn clusterSlot(c:vec3i,b:u32)->u32{
 if(!levelHomed(b)){return NONE_CLUSTER;}
 let n=vec3i((partExtent()+vec3u(b-1u))/b);if(any(c<vec3i(0))||any(c>=n)){return NONE_CLUSTER;}
 let index=u32((c.z*n.y+c.y)*n.x+c.x);if(index>=levelCapacity(b)){return NONE_CLUSTER;}
 return levelHomeBase(b)+4u+index*4u;
}
fn clusterState(c:vec3i,b:u32)->u32{
 let s=clusterSlot(c,b);if(s==NONE_CLUSTER){return select(CLUSTER_AIR,CLUSTER_UNKNOWN,chartSurface());}
 if(atomicLoad(&contactWork[s])!=0u){return CLUSTER_SOLID;}if(!chartSurface()){return CLUSTER_AIR;}
 let w=atomicLoad(&contactWork[s+3u])&3u;return select(CLUSTER_UNKNOWN,w,w==CLUSTER_AIR||w==CLUSTER_SOLID);
}
fn clusterOccupied(c:vec3i,b:u32)->bool{return clusterState(c,b)==CLUSTER_SOLID;}
// Holds a declared cell (closed: the same as occupied).
fn clusterDeclared(c:vec3i,b:u32)->bool{let s=clusterSlot(c,b);return s!=NONE_CLUSTER&&atomicLoad(&contactWork[s])!=0u;}

fn clusterEmits(c:vec3i,b:u32)->bool{let s=clusterSlot(c,b);return s!=NONE_CLUSTER&&atomicLoad(&contactWork[s])!=0u&&(atomicLoad(&contactWork[s+2u])&CLUSTER_SELECTED)!=0u;}
fn clusterKnown(c:vec3i,b:u32)->bool{return clusterState(c,b)!=CLUSTER_UNKNOWN;}

// TOPOLOGY (word 3; generation + selected block: cleared with the home, kept
// with a retained one): bit 4 computed, bit 5 every cluster of the 3x3x3
// neighbourhood has a known state (what this cluster's emission reads).
// POSE (word 2; this draw: reset with the emission flags): bit 1 computed,
// bit 2 the least cell places its posed frame; bit 3 SELECTED (an owned fine
// edge reaches it; contactEdges). Both certificates are written by ONE

const CLUSTER_FRAME_DONE=2u;const CLUSTER_FRAME_OK=4u;const CLUSTER_SELECTED=8u;const CLUSTER_NEAR_DONE=16u;const CLUSTER_NEAR_KNOWN=32u;
fn nearKnown(c:vec3i,b:u32)->bool{
 for(var z=-1;z<=1;z++){for(var y=-1;y<=1;y++){for(var x=-1;x<=1;x++){if(!clusterKnown(c+vec3i(x,y,z),b)){return false;}}}}
 return true;
}

fn clusterNearKnown(s:u32)->bool{let w=atomicLoad(&contactWork[s+3u]);return (w&CLUSTER_NEAR_DONE)!=0u&&(w&CLUSTER_NEAR_KNOWN)!=0u;}
fn clusterFramePlaced(s:u32)->bool{let w=atomicLoad(&contactWork[s+2u]);return (w&CLUSTER_FRAME_DONE)!=0u&&(w&CLUSTER_FRAME_OK)!=0u;}
// Surface: the chart may own a fine edge around this cluster (closed: always).
fn clusterCertified(c:vec3i,b:u32)->bool{
 if(!chartSurface()){return true;}
 let s=clusterSlot(c,b);if(s==NONE_CLUSTER||!clusterNearKnown(s)){return false;}
 return atomicLoad(&contactWork[s])==0u||clusterFramePlaced(s);
}
// On the chunk's x/z border: the chunk meets its neighbour there.
fn clusterBorder(c:vec3i,b:u32)->bool{let e=vec3i(partExtent());let n=i32(b);return c.x<=0||c.z<=0||(c.x+1)*n>=e.x||(c.z+1)*n>=e.z;}

// A node whose box holds an unknown, cave or x/z-border cell must recurse
// (complete fine children), exactly the CPU reference selectVoxelFrontier.
// The decision is stamped once per home build into the cluster's own word 3
// bit 7 (CLUSTER_FORCED) by compileClusterEdges, alongside the resident edge
// topology under the same EDGES_DONE guard, so it is kept with a retained
// home for free: voxelHome's retained reset only clears word 2 bits 0..3,
// and every other word-3 writer touches only its own bits (classification
// 0..1, near certificates 4..5, edge records 8..27, hold code 28..30).
// nodeRequired reads a forced node as requiring block 0 (finer than every
// ask), which keeps the frontier's down-set: clusterAtLevel/cellLevel then
// recurse exactly like the reference, and the region seams keep fine
// segments on both sides (clusterLevelOk) with no other change.
const CLUSTER_FORCED=128u;
const CAVE_WALK_BOUND=128u;
fn clusterForced(c:vec3i,b:u32)->bool{let s=clusterSlot(c,b);return s!=NONE_CLUSTER&&(atomicLoad(&contactWork[s+3u])&CLUSTER_FORCED)!=0u;}

// quads, so faceMask==0 is not air: a solid interior has no face. Air is
// certified ONLY by the producer's occupancy; without it nothing is air here
// and clusterForceBox keeps complete fine children instead.
fn occupancyOn()->bool{return (params.owners[3].y&32u)!=0u;}
fn cellSolid(c:vec3i)->bool{
 let e=partExtent();let w=(e.y+31u)/32u;let y=u32(c.y);
 return ((cellOccupancy[(u32(c.z)*e.x+u32(c.x))*w+y/32u]>>(y%32u))&1u)!=0u;
}
// The cell's column above it holds no solid cell up to the top of the grid
// (open to the exterior; exact for any occupancy, every cell for a heightfield).
fn cellOpenAbove(c:vec3i)->bool{
 let e=partExtent();let w=(e.y+31u)/32u;let base=(u32(c.z)*e.x+u32(c.x))*w;let y=u32(c.y)+1u;
 if(y>=e.y){return true;}
 for(var k=y/32u;k<w;k++){var bits=cellOccupancy[base+k];if(k==y/32u){bits&=~((1u<<(y%32u))-1u);}if(bits!=0u){return false;}}
 return true;
}
// A cell the cave walk may step on: in the grid and certified AIR.
fn caveWalkable(cell:vec3i)->bool{
 if(any(cell<vec3i(0))||any(cell>=vec3i(partExtent()))){return false;}
 if(!occupancyOn()){return false;}
 return !cellSolid(cell);
}
// Whether start's air reaches the open exterior: bounded depth-first walk
// over walkable cells in fixed neighbor order (+x,-x,+y,-y,+z,-z), with a
// linear visited set. Reaching outside the grid is open (true); exhausting
// the component, overflowing the visit arrays, or hitting CAVE_WALK_BOUND
// visits without leaving is a cave (false). The bound only ever over-forces
// (safe): a truly enclosed region is correctly a cave at any bound, since
// only boundary contact returns open. Terminates: every iteration pops one
// entry or returns, visits are capped, arrays are never overrun.
fn caveWalkOpen(start:vec3i)->bool{
 var seen:array<u32,CAVE_WALK_BOUND>;var count=0u;
 var stack:array<vec3i,CAVE_WALK_BOUND>;var top=0u;
 stack[0]=start;top=1u;seen[0]=cellIndex(vec3u(start));count=1u;
 var visits=0u;
 let e=vec3i(partExtent());
 loop{
  if(top==0u){return false;}
  if(visits>=CAVE_WALK_BOUND){return false;}
  visits+=1u;top-=1u;let c=stack[top];
  for(var k=0u;k<6u;k++){
   let n=c+vec3i(select(0,1,k==0u)-select(0,1,k==1u),select(0,1,k==2u)-select(0,1,k==3u),select(0,1,k==4u)-select(0,1,k==5u));
   if(any(n<vec3i(0))||any(n>=e)){return true;}
   if(!caveWalkable(n)){continue;}
   let li=cellIndex(vec3u(n));
   var known=false;for(var i=0u;i<count;i++){if(seen[i]==li){known=true;break;}}
   if(known){continue;}
   if(count>=CAVE_WALK_BOUND||top>=CAVE_WALK_BOUND){return false;}
   seen[count]=li;count+=1u;
   stack[top]=n;top+=1u;
  }
 }
 return false;// Unreachable: the loop only exits via return. Naga requires it.
}
// Whether node (c,b) must recurse: its box holds an unknown, cave or
// x/z-border cell. Border is exact arithmetic (clusterBorder already tests
// box/strip intersection); unknown reads the node's own state plus every
// level-2 state in range (both rebuild-stable here); caves walk from
// non-boundary walkable cells adjacent to non-walkable ones (every enclosed
// component has one: its maximum-x cell's +x neighbour is in-grid and
// non-member, hence non-walkable). Reads only rebuild-stable words.
fn clusterForceBox(c:vec3i,b:u32)->bool{
 if(clusterBorder(c,b)){return true;}
 if(clusterState(c,b)==CLUSTER_UNKNOWN){return true;}
 let e=vec3i(partExtent());let n=i32(b);
 let lo=c*n;let hi=min(lo+vec3i(n),e);
 for(var qz=lo.z/2;qz*2<hi.z;qz++){for(var qy=lo.y/2;qy*2<hi.y;qy++){for(var qx=lo.x/2;qx*2<hi.x;qx++){
  if(clusterState(vec3i(qx,qy,qz),2u)==CLUSTER_UNKNOWN){return true;}
 }}}

 if(!occupancyOn()){return true;}
 for(var z=lo.z;z<hi.z;z++){for(var y=lo.y;y<hi.y;y++){for(var x=lo.x;x<hi.x;x++){
  let cell=vec3i(x,y,z);
  if(x==0||y==0||z==0||x+1>=e.x||y+1>=e.y||z+1>=e.z){continue;}
  if(!caveWalkable(cell)){continue;}
  // Open to the top of the grid: exterior air, no walk (exact).
  if(cellOpenAbove(cell)){continue;}
  var edge=false;
  if(!caveWalkable(cell+vec3i(1,0,0))){edge=true;}
  if(!caveWalkable(cell+vec3i(-1,0,0))){edge=true;}
  if(!caveWalkable(cell+vec3i(0,1,0))){edge=true;}
  if(!caveWalkable(cell+vec3i(0,-1,0))){edge=true;}
  if(!caveWalkable(cell+vec3i(0,0,1))){edge=true;}
  if(!caveWalkable(cell+vec3i(0,0,-1))){edge=true;}
  if(!edge){continue;}
  if(!caveWalkOpen(cell)){return true;}
 }}}
 return false;
}
struct Frame {ok:bool,centre:vec3f,axis:mat3x3f}
fn cellFrame(c:vec3u)->Frame{if(chartSurface()){return surfaceFrame(c);}return closedFrame(c);}
fn closedFrame(c:vec3u)->Frame{
 let at=faceBase()+cellIndex(c)*19u;
 if(any(c>=partExtent())||atomicLoad(&contactWork[at+18u])!=63u){return Frame(false,vec3f(0.0),mat3x3f());}
 var f:array<vec3f,6>;
 for(var k=0u;k<6u;k++){f[k]=vec3f(bitcast<f32>(atomicLoad(&contactWork[at+k*3u])),bitcast<f32>(atomicLoad(&contactWork[at+k*3u+1u])),bitcast<f32>(atomicLoad(&contactWork[at+k*3u+2u])));}
 return Frame(true,(f[0]+f[1]+f[2]+f[3]+f[4]+f[5])/6.0,mat3x3f(f[0]-f[1],f[2]-f[3],f[4]-f[5]));
}
// A vertex's rest lattice point (w 0: its cell frame is incomplete).

// bit 4: restCellChart.cornerOrder 'shallot-voxel-quad-v1', the mesher's
// quad), the point is the shared direct lookup of (cell, face, v % 4): no
// posed frame and no inference. Otherwise the posed cell-frame inversion.
fn cornersCertified()->bool{return (params.owners[3].y&16u)!=0u;}
fn vertexLattice(v:u32)->vec4i{
 if(cornersCertified()){let cell=cellOf(v);let g=faceOf(v);if(any(cell>=partExtent())||g>5u){return vec4i(0);}return vec4i(voxelRestLatticeCorner(cell,g,v%4u),1);}
 let c=cellOf(v);let frame=cellFrame(c);if(!frame.ok){return vec4i(0);}
 // Its coordinates in the cell's own posed frame (d = A o): exact for a
 // sheared or twisted cell, where dot(d, a_k) is not.
 let d=position(v,0u)-frame.centre;let a=frame.axis;let det=dot(a[0],cross(a[1],a[2]));if(!(abs(det)>0.0)){return vec4i(0);}
 let o=vec3i(select(0,1,dot(d,cross(a[1],a[2]))*det>0.0),select(0,1,dot(d,cross(a[2],a[0]))*det>0.0),select(0,1,dot(d,cross(a[0],a[1]))*det>0.0));
 var lo=o;if(chartSurface()){let g=faceOf(v);if(g>5u){return vec4i(0);}lo[g/2u]=select(0,1,g%2u==0u);}
 return vec4i(vec3i(c)+lo,1);
}
// The cell that places lattice point p when no vertex sits there: the least
// cell of the first occupied cluster around p (fixed order), the same for
// every edge that reaches p.
fn latticeCell(p:vec3i,b:u32)->vec4i{
 let n=i32(b);
 for(var k=0;k<8;k++){
  let c=vec3i(select(p.x/n-1,p.x/n,(k&1)!=0),select(p.y/n-1,p.y/n,(k&2)!=0),select(p.z/n-1,p.z/n,(k&4)!=0));
  let s=clusterSlot(c,b);if(s==NONE_CLUSTER){continue;}let w=atomicLoad(&contactWork[s]);
  if(w!=0u){let i=~w;let e=partExtent();let cell=vec4i(i32(i%e.x),i32((i/e.x)%e.y),i32(i/(e.x*e.y)),1);if(chartSurface()&&!clusterFramePlaced(s)){continue;}return cell;}
 }
 return vec4i(0);
}
// The posed position of lattice point p: the vertex recorded there, else the
// affine step from latticeCell's frame.
fn latticePoint(p:vec3i,b:u32)->vec4f{
 if(latticeInside(p)){let at=cornerBase()+latticeIndex(p)*4u;if(atomicLoad(&contactWork[at])!=0u){return vec4f(bitcast<f32>(atomicLoad(&contactWork[at+1u])),bitcast<f32>(atomicLoad(&contactWork[at+2u])),bitcast<f32>(atomicLoad(&contactWork[at+3u])),1.0);}}
 let c=latticeCell(p,b);if(c.w==0){return vec4f(0.0);}
 let frame=cellFrame(vec3u(c.xyz));if(!frame.ok){return vec4f(0.0);}
 return vec4f(frame.centre+frame.axis*(vec3f(p)-vec3f(c.xyz)-vec3f(.5)),1.0);
}
fn toEye(q:vec3f)->vec3f{return select(-params.direction.xyz,params.eye.xyz-q,params.eye.w!=0.0);}
fn columnFace(n:vec3f,q:vec3f)->bool{let f=dot(n,toEye(q))>0.0;return select(f,!f,params.direction.w==2.0);}
// One coarse edge between two coarse faces of normals na, nb: crease when both
// face the eye, silhouette when one does, nothing when neither.
fn columnEdge(a:vec3f,b:vec3f,na:vec3f,nb:vec3f,t:Triangle,address:u32,block:f32,face:u32){
 let m=(a+b)*.5;let fa=columnFace(na,m);let fb=columnFace(nb,m);if(!fa&&!fb){return;}
 emit(a,b,t,select(0u,2u,fa&&fb),address,vec4u(clusterPort(a,block),clusterPort(b,block),face,0xffffffffu),vec4f(0),false);
}
// The edge(s) of one 2x2 cross-section pattern around a grid edge a-b. Cell i
// lies at (-1 or +1 along U, -1 or +1 along V): i = 0 (-,-), 1 (+,-), 2 (-,+),
// 3 (+,+). One occupied cell: a convex edge between its two outward faces.
// Three: a concave edge between the two faces bounding the empty cell. Two
// diagonal: both convex edges. Two adjacent, none or all four: flat, nothing.
fn patternEdges(p:vec4<bool>,a:vec3f,b:vec3f,U:vec3f,V:vec3f,t:Triangle,address:u32,block:f32,face:u32){
 let n=u32(p.x)+u32(p.y)+u32(p.z)+u32(p.w);
 let signs=array<vec2f,4>(vec2f(-1.0,-1.0),vec2f(1.0,-1.0),vec2f(-1.0,1.0),vec2f(1.0,1.0));
 if(n==1u||(n==2u&&((p.x&&p.w)||(p.y&&p.z)))){
  for(var i=0u;i<4u;i++){if(!p[i]){continue;}let s=signs[i];columnEdge(a,b,U*(-s.x),V*(-s.y),t,address,block,face);}
  return;
 }
 if(n==3u){for(var i=0u;i<4u;i++){if(p[i]){continue;}let s=signs[i];columnEdge(a,b,U*s.x,V*s.y,t,address,block,face);}}
}
fn unit(k:u32)->vec3i{return vec3i(select(0,1,k==0u),select(0,1,k==1u),select(0,1,k==2u));}

// implementation"). Cluster C owns 12 candidate coarse edges: along each chart
// axis k (0..2) its four corner edges (corner 0..3), edge e = k*4 + corner.
// The four clusters around edge e:
fn edgeClusters(c:vec3i,e:u32)->array<vec3i,4>{
 let k=e/4u;let corner=e%4u;let ei=unit((k+1u)%3u);let ej=unit((k+2u)%3u);let oi=i32(corner&1u);let oj=i32(corner>>1u);
 return array<vec3i,4>(c+ei*(oi-1)+ej*(oj-1),c+ei*oi+ej*(oj-1),c+ei*(oi-1)+ej*oj,c+ei*oi+ej*oj);
}
// The TOPOLOGY of edge e of cluster C at block b, independent of camera,
// pose, lighting and this draw's selection: the 4-bit occupancy pattern
// (bit i = cluster i of edgeClusters occupied) when the edge produces a coarse
// edge (one occupied, three occupied, or a diagonal pair), every surrounding
// cluster is known (surface semantics) and none lies on the chunk's x/z border;
// else 0 (flat, unknown or border: never a coarse edge). It reads only the
// generation-owned occupancy/classification words, so it is compiled ONCE per
// home build and retained with the home (generation token + block).
fn clusterEdgePattern(c:vec3i,b:u32,e:u32)->u32{
 let q=edgeClusters(c,e);
 if(chartSurface()&&!(clusterKnown(q[0],b)&&clusterKnown(q[1],b)&&clusterKnown(q[2],b)&&clusterKnown(q[3],b))){return 0u;}
 if(clusterBorder(q[0],b)||clusterBorder(q[1],b)||clusterBorder(q[2],b)||clusterBorder(q[3],b)){return 0u;}
 let p=u32(clusterOccupied(q[0],b))|(u32(clusterOccupied(q[1],b))<<1u)|(u32(clusterOccupied(q[2],b))<<2u)|(u32(clusterOccupied(q[3],b))<<3u);
 let n=countOneBits(p);
 return select(0u,p,n==1u||n==3u||(n==2u&&(p==9u||p==6u)));
}
// Resident storage, in the cluster's EXISTING home words (no layout or binding
// change): word 2 bits 4..31 hold edges 0..6 (4 bits each; voxelHome resets
// only bits 0..3 of word 2 on a retained home), word 3 bits 8..27 hold edges
// 7..11, and word 3 bit 6 (CLUSTER_EDGES_DONE) marks the cluster compiled.
// Both words are zeroed with the rest of the home on a rebuild, so a content
// (generation) or level change recompiles every cluster of that home.
const CLUSTER_EDGES_DONE=64u;
fn compileClusterEdges(s:u32,c:vec3i,b:u32){
 if((atomicLoad(&contactWork[s+3u])&CLUSTER_EDGES_DONE)!=0u){return;}
 var lo=0u;var hi=0u;
 for(var e=0u;e<12u;e++){let p=clusterEdgePattern(c,b,e);if(e<7u){lo|=p<<(4u+4u*e);}else{hi|=p<<(8u+4u*(e-7u));}}
 var forced=0u;if(frontierOn()&&clusterForceBox(c,b)){forced=CLUSTER_FORCED;}
 atomicOr(&contactWork[s+2u],lo);atomicOr(&contactWork[s+3u],hi|CLUSTER_EDGES_DONE|forced);
}
fn residentEdgePattern(s:u32,e:u32)->u32{
 if(e<7u){return (atomicLoad(&contactWork[s+2u])>>(4u+4u*e))&15u;}
 return (atomicLoad(&contactWork[s+3u])>>(8u+4u*(e-7u)))&15u;
}
// Cluster C's share of the coarse edges: along each chart axis k, the four
// edges of C, each only where C is the first selected declared cluster of its

// is a coarse edge. Ends at the posed lattice points, faces turned by the
// posed axes of C's least cell: projection, pose, facing and this draw's
// selection stay dynamic; the occupancy/known/border topology is READ from the

// compiled record (no chartCertify this build) computes the same rule live.
fn clusterEmit(c:vec3i,b:u32,t:Triangle,address:u32,face:u32){
 let rep=latticeCell(c*i32(b)+vec3i(i32(b)),b);let frame=cellFrame(vec3u(max(rep.xyz,vec3i(0))));if(rep.w==0||!frame.ok){return;}
 let block=f32(b);let s=clusterSlot(c,b);
 let resident=s!=NONE_CLUSTER&&(atomicLoad(&contactWork[s+3u])&CLUSTER_EDGES_DONE)!=0u;
 for(var k=0u;k<3u;k++){
  let i=(k+1u)%3u;let j=(k+2u)%3u;let ek=unit(k);let ei=unit(i);let ej=unit(j);
  let U=normalize(frame.axis[i]);let V=normalize(frame.axis[j]);
  for(var corner=0u;corner<4u;corner++){
   let e=k*4u+corner;
   var pattern=0u;if(resident){pattern=residentEdgePattern(s,e);}else{pattern=clusterEdgePattern(c,b,e);}
   if(pattern==0u){continue;}
   let q=edgeClusters(c,e);

   // its clusters belong to level b; otherwise its fine segments stay fine.
   if(!(clusterLevelOk(q[0],b)&&clusterLevelOk(q[1],b)&&clusterLevelOk(q[2],b)&&clusterLevelOk(q[3],b))){continue;}

   // a coarse edge beside an owned (suppressed) fine edge is always drawn,
   // exactly once, by a cluster that runs clusterEmit (only selected ones do).
   let d=vec4<bool>(clusterEmits(q[0],b),clusterEmits(q[1],b),clusterEmits(q[2],b),clusterEmits(q[3],b));
   var owner=4u;if(d.w){owner=3u;}if(d.z){owner=2u;}if(d.y){owner=1u;}if(d.x){owner=0u;}
   if(owner!=3u-corner){continue;}
   let oi=i32(corner&1u);let oj=i32(corner>>1u);
   let o=(c+ei*oi+ej*oj)*i32(b);let a=latticePoint(o,b);let z=latticePoint(o+ek*i32(b),b);if(a.w==0.0||z.w==0.0){continue;}
   patternEdges(vec4<bool>((pattern&1u)!=0u,(pattern&2u)!=0u,(pattern&4u)!=0u,(pattern&8u)!=0u),a.xyz,z.xyz,U,V,t,address,block,face);
  }
 }
}
// The clusters around the segment between lattice points A and B: the
// clusters of the cells whose closure holds its midpoint.
fn segmentClusters(a:vec3i,b:vec3i,block:u32)->array<vec4i,8>{
 var out:array<vec4i,8>;let m=a+b;let n=i32(block);
 for(var k=0;k<8;k++){
  var c=vec3i(0);var used=true;
  for(var axis=0u;axis<3u;axis++){
   let bit=(k>>axis)&1;
   if((m[axis]&1)!=0){if(bit!=0){used=false;}c[axis]=(m[axis]-1)/2;}else{c[axis]=m[axis]/2-1+bit;}
  }
  let cl=vec3i(select(c.x/n,(c.x-n+1)/n,c.x<0),select(c.y/n,(c.y-n+1)/n,c.y<0),select(c.z/n,(c.z-n+1)/n,c.z<0));
  out[k]=vec4i(cl,select(0,1,used));
 }
 return out;
}
// THE OWNERSHIP TEST, one predicate for every pass: the chart owns the fine
// segment va-vb (two posed vertices) when the chunk is at a cluster level and
// every cluster around it is interior.
fn chartSegmentCoarse(va:u32,vb:u32)->bool{

 // vote; every cluster around it must belong to that level (clusterLevelOk).
 let b=segmentLevel(va,vb);if(b<2u){return false;}
 let la=storedLattice(va);let lb=storedLattice(vb);
 let around=segmentClusters(la.xyz,lb.xyz,b);
 for(var k=0;k<8;k++){if(around[k].w!=0&&(clusterBorder(around[k].xyz,b)||!clusterCertified(around[k].xyz,b)||!clusterLevelOk(around[k].xyz,b))){return false;}}
 return true;
}
fn voxelEdgeCoarse(id:u32)->bool{if(!chartOn()){return false;}let h=halfedge(edge(id).x);return chartSegmentCoarse(h.x,h.y);}
fn voxelVertexCoarse(v:u32)->bool{
 if(!chartOn()){return false;}let q=(v/4u)*4u;let k=v%4u;
 return chartSegmentCoarse(v,q+(k+1u)%4u)&&chartSegmentCoarse(q+(k+3u)%4u,v);
}
// contactEdges: an owned fine edge marks every declared cluster around it.
fn chartSelect(id:u32){
 let h=halfedge(edge(id).x);let b=segmentLevel(h.x,h.y);if(b<2u){return;}
 let around=segmentClusters(storedLattice(h.x).xyz,storedLattice(h.y).xyz,b);
 for(var k=0;k<8;k++){
  if(around[k].w==0){continue;}let s=clusterSlot(around[k].xyz,b);
  if(s!=NONE_CLUSTER&&atomicLoad(&contactWork[s])!=0u){
   let old=atomicOr(&contactWork[s+2u],CLUSTER_SELECTED);
   if((old&CLUSTER_SELECTED)==0u){

    let index=(s-levelHomeBase(b)-4u)/4u;let list=selectedListBase();let at=atomicAdd(&contactWork[list],1u);

    if(at<selectedListCapacity()){atomicStore(&contactWork[list+2u+at],index|(levelOrdinal(b)<<28u));}else{atomicStore(&contactWork[list+1u],1u);}
   }
  }
 }
}
// boundaryAt's voxel branch. True: the chart owns this fine edge; it produces
// nothing (its clusters are emitted by chartEmitAt). False: fine path.
fn voxelColumnEdge(id:u32,t:Triangle,face:u32)->bool{return voxelEdgeCoarse(id);}
fn residualFacesForEdge(id:u32){
 let e=edge(id);
 for(var i=0u;i<e.z;i++){let incidence=sourceEdgeCorner(id,i);residualFaceAppend(incidence/3u);}
}
// Voxel levels are selected from the already-certified resident chart before
// any fine contact slots are built. This is the same ownership predicate used
// by boundaryAtFine; the stage only moves the existing cluster claim earlier,
// it does not invent a second topology or a fallback geometry.
fn chartSelectAt(id:u32){
 if(!posedContacts()||!chartOn()||id>=sourceEdgeCount()||!edgeInDraw(id)){return;}
 let h=halfedge(edge(id).x);let b=segmentLevel(h.x,h.y);var coarse=b>=2u;
 let around=segmentClusters(storedLattice(h.x).xyz,storedLattice(h.y).xyz,max(b,1u));
 if(coarse){for(var k=0u;k<8u;k++){if(around[k].w!=0&&(clusterBorder(around[k].xyz,b)||!clusterCertified(around[k].xyz,b)||!clusterLevelOk(around[k].xyz,b))){coarse=false;break;}}}
 let bankFallback=sourcePathSegmentCount()!=0u||sourcePlanarSpanCount()!=0u||sourcePanelJoinCount()!=0u||sourceHullEdgeCount()!=0u;
 if(coarse&&!bankFallback){
  for(var k=0u;k<8u;k++){
   if(around[k].w==0){continue;}let s=clusterSlot(around[k].xyz,b);
   if(s!=NONE_CLUSTER&&atomicLoad(&contactWork[s])!=0u){
    let old=atomicOr(&contactWork[s+2u],CLUSTER_SELECTED);
    if((old&CLUSTER_SELECTED)==0u){let index=(s-levelHomeBase(b)-4u)/4u;let list=selectedListBase();let at=atomicAdd(&contactWork[list],1u);if(at<selectedListCapacity()){atomicStore(&contactWork[list+2u+at],index|(levelOrdinal(b)<<28u));}else{atomicStore(&contactWork[list+1u],1u);}}
   }
  }
  return;
 }
 residualEdgeAppend(id);residualVertexAppend(h.x);residualVertexAppend(h.y);residualFacesForEdge(id);
}
@compute @workgroup_size(64) fn chartSelectStage(@builtin(global_invocation_id) gid:vec3u){chartSelectAt(gid.x+gid.y*params.owners[3].w*64u);}
// Cluster-count work in the boundary stage: thread i (strided) emits every
// SELECTED cluster once, from its own least quad (triangle, address, owner
// gates) - never re-expanded from the fine edges that own it.
fn chartEmitAt(id:u32,threads:u32){
 if(!posedContacts()||!chartOn()||selectedCount()==0u){return;}
 if(atomicLoad(&contactWork[overflowWord()])!=0u){return;}
 let instance=selectedInstance(0u);if(!selected(instance)){return;}

 // ITS frontier level), not every cluster; after a list overflow, the full
 // scan covers every homed level in turn.
 let list=selectedListBase();let overflow=atomicLoad(&contactWork[list+1u])!=0u;
 let listed=min(atomicLoad(&contactWork[list]),selectedListCapacity());
 var scan=0u;for(var l=2u;l<=16u;l*=2u){if(levelHomed(l)){scan+=levelClusters(l);}}
 let span=select(listed,scan,overflow);
 for(var j=id;j<span;j+=max(threads,1u)){
  var b=0u;var i=0u;
  if(!overflow){let entry=atomicLoad(&contactWork[list+2u+j]);b=2u<<(entry>>28u);i=entry&0x0fffffffu;}
  else{var r=j;for(var l=2u;l<=16u;l*=2u){if(!levelHomed(l)){continue;}let m=levelClusters(l);if(r<m){b=l;i=r;break;}r-=m;}}
  if(!levelHomed(b)){continue;}
  let n=(partExtent()+vec3u(b-1u))/b;if(i>=min(n.x*n.y*n.z,levelCapacity(b))){continue;}
  let s=levelHomeBase(b)+4u+i*4u;if(atomicLoad(&contactWork[s])==0u||(atomicLoad(&contactWork[s+2u])&CLUSTER_SELECTED)==0u){continue;}
  let inc=~atomicLoad(&contactWork[s+1u]);if(!inDraw(inc)||!validFace(inc)){continue;}
  let v=halfedge(inc).x;let owner=edgeOwner(v,instance);
  if(ownerUnseen(v,instance)||ownerMerged(owner)||outlineLevel(owner)){continue;}
  clusterEmit(vec3i(i32(i%n.x),i32((i/n.x)%n.y),i32(i/(n.x*n.y))),b,triangle(inc,0u),inc,inc/3u);
 }
}
fn residualDispatchWrite(){
 let base=residualDispatchBase();let limit=max(1u,params.owners[3].w);let usable=residualOn()&&!residualOverflow();
 var boundary=sourceBoundarySlots()*selectedCount();
 var contour=sourceFaceCount()*selectedCount();
 var vertices=min(sourceFaceCount()*2u,native[0]/6u*4u);
 var edges=min(sourceEdgeCount(),native[0]/6u*5u);
 if(usable){boundary=residualEdgeCount();contour=residualFaceCount()*selectedCount();vertices=max(residualVertexCount(),prepareAnchorParts());edges=residualEdgeCount();}
 let bg=(boundary+63u)/64u;dispatch[base+0u]=min(bg,limit);dispatch[base+1u]=max(1u,(bg+limit-1u)/limit);dispatch[base+2u]=1u;
 let cg=(contour+63u)/64u;dispatch[base+3u]=min(cg,limit);dispatch[base+4u]=max(1u,(cg+limit-1u)/limit);dispatch[base+5u]=1u;
 let vg=(vertices+63u)/64u;dispatch[base+9u]=min(vg,limit);dispatch[base+10u]=max(1u,(vg+limit-1u)/limit);dispatch[base+11u]=1u;
 let eg=(edges+63u)/64u;dispatch[base+12u]=min(eg,limit);dispatch[base+13u]=max(1u,(eg+limit-1u)/limit);dispatch[base+14u]=1u;
}
// The chart is a complete source-owned surface representation. Emit its
// selected clusters as a separate ordered stage, before contactVertices and
// contactEdges can expand the fine incidence graph. The marker is written by
// invocation zero only after this dispatch; the following boundary dispatch is
// therefore the only place that can observe it.
@compute @workgroup_size(64) fn chartEmit(@builtin(global_invocation_id) gid:vec3u){
 let id=gid.x+gid.y*params.owners[3].w*64u;let threads=max(1u,chartClusters());
 chartEmitAt(id,threads);if(id==0u&&posedContacts()&&chartOn()){atomicStore(&contactWork[chartEarlyEmittedWord()],1u);}
}
@compute @workgroup_size(64) fn chartPublish(@builtin(global_invocation_id) gid:vec3u){
 let id=gid.x+gid.y*params.owners[3].w*64u;if(id==0u&&posedContacts()&&chartOn()){residualDispatchWrite();}
}
fn partCode(v:u32)->u32{return (partCodes[v]&0xffffffu)+1u;}
fn partExtent()->vec3u{return ((vec3u(params.owners[3].y)>>vec3u(8u,16u,24u))&vec3u(255u))+vec3u(1u);}
fn contactHash(x:u32)->u32{var h=x;h=(h^(h>>16u))*0x7feb352du;h=(h^(h>>15u))*0x846ca68bu;return h^(h>>16u);}
fn contactPositionHash(q:vec3f)->u32{
 let bits=bitcast<vec3u>(select(q,vec3f(0),q==vec3f(0)));
 return contactHash(bits.x^contactHash(bits.y)^contactHash(bits.z+0x9e3779b9u));
}
fn partSlot(code:u32)->u32{
 let slots=contactSlots();let base=partBase();let hash=contactHash(code);
 for(var probe=0u;probe<slots;probe++){let at=(hash+probe)&(slots-1u);let found=atomicLoad(&contactWork[base+at]);if(found==code){return at;}if(found==0u){break;}}
 return NO_CONTACT;
}
fn connectParts(a:u32,b:u32){
 if(a==b){return;}let first=partSlot(a);let second=partSlot(b);
 if(first!=NO_CONTACT){atomicOr(&contactWork[partFlagsBase()+first],2u);}
 if(second!=NO_CONTACT){atomicOr(&contactWork[partFlagsBase()+second],2u);}
}
fn addPartVertex(v:u32,q:vec3f){
 if(!cellParts()){return;}let code=partCode(v);let slots=contactSlots();let hash=contactHash(code);let base=partBase();
 for(var probe=0u;probe<slots;probe++){
  let at=(hash+probe)&(slots-1u);var first=atomicLoad(&contactWork[base+at]);
  loop{if(first!=0u){break;}let claim=atomicCompareExchangeWeak(&contactWork[base+at],0u,code);first=claim.old_value;if(claim.exchanged){first=code;break;}}
  if(first!=code){continue;}
  let clip=params.clip*vec4f(q,1);var low=0u;var high=0xffffffffu;
  if(clip.w>0.00001&&clip.z>=0.0&&clip.z<=clip.w){
   let row=(.5-.5*clip.y/clip.w)*f32(params.owners[2].x)/bitcast<f32>(params.owners[1].y);
   if(row==row&&abs(row)<3.402823e38){low=formOrdered(row);high=low;}
  }
  atomicMin(&contactWork[base+slots+at],low);atomicMax(&contactWork[base+slots*2u+at],high);return;
 }
}
fn smallClosedPart(code:u32)->bool{
 let cell=(vec3u(code-1u)>>vec3u(0u,8u,16u))&vec3u(255u);let extent=partExtent();
 if(any(cell>=extent)){return false;}
 let slots=contactSlots();let hash=contactHash(code);let base=partBase();
 for(var probe=0u;probe<slots;probe++){
  let at=(hash+probe)&(slots-1u);let first=atomicLoad(&contactWork[base+at]);if(first==0u){break;}if(first!=code){continue;}
  let lo=atomicLoad(&contactWork[base+slots+at]);let hi=atomicLoad(&contactWork[base+slots*2u+at]);
  if(lo==0u||hi==0xffffffffu||lo==0xffffffffu){return false;}
  let rows=formUnordered(hi)-formUnordered(lo);let address=(cell.z*extent.y+cell.y)*extent.x+cell.x;
  let held=atomicLoad(&partLevels[address]);
  let far=rows<select(${FAR_STRIP_BELOW_ROWS*LOD_TIER_HOLD},${FAR_STRIP_BELOW_ROWS/LOD_TIER_HOLD},held==2u);
  atomicStore(&partLevels[address],select(1u,2u,far));return far;
 }
 return false;
}
fn posedVertexPort(v:u32)->u32{
 let q=position(v,0u);let slots=contactSlots();let hash=contactPositionHash(q);
 for(var probe=0u;probe<slots;probe++){
  let first=atomicLoad(&contactWork[(hash+probe)&(slots-1u)]);
  if(first==0u){break;}if(all(position(first-1u,0u)==q)){return first-1u;}
 }
 return v;
}
fn posedEdgePorts(id:u32)->vec2u{
 let h=halfedge(edge(id).x);let a=posedVertexPort(h.x);let b=posedVertexPort(h.y);return vec2u(min(a,b),max(a,b));
}
fn posedEdgeSlot(id:u32)->u32{
 if(!posedContacts()){return NO_CONTACT;}let key=posedEdgePorts(id);if(key.x==key.y){return NO_CONTACT;}
 let slots=contactSlots();let hash=contactHash(key.x^contactHash(key.y));
 for(var probe=0u;probe<slots;probe++){
  let slot=(hash+probe)&(slots-1u);let first=atomicLoad(&contactWork[slots+slot]);
  if(first==0u){break;}if(all(posedEdgePorts(first-1u)==key)){return slot;}
 }
 return NO_CONTACT;
}
fn boundaryPort(v:u32,instance:u32)->u32{if(posedContacts()){return posedVertexPort(v);}return vertexPort(v,instance);}

// Projection and clearing are separate dispatches. A record is published (its
// index exchanged into a bucket head) only after the same projection invocation
// wrote its complete key and extent; readers walk the bucket list and compare
// the full key (domain|producer, native owner, instance, part ordinal). The
// owner is edgeOwner of the anchor vertex, exactly a run's (identity.y,
// identity.z); instance is retained separately in the record and row.
// Visibility is never read.
const PART_ANCHOR_RECORD_WORDS=16u;
const PART_ANCHOR_CANONICAL_WORDS=8u;
const PART_ANCHOR_CANONICAL_HEADER_WORDS=10u;
fn partAnchorHeads()->u32{return atomicLoad(&partAnchorTable[0]);}
fn partAnchorRecord(record:u32)->u32{return atomicLoad(&partAnchorTable[1])+record*PART_ANCHOR_RECORD_WORDS;}
fn partAnchorBucket(owner:vec2u,part:u32,instance:u32)->u32{
 var h=(owner.x*0x9e3779b9u)^(owner.y*0x85ebca6bu)^(part*0xc2b2ae35u)^(instance*0x27d4eb2fu);h=(h^(h>>16u))*0x7feb352du;h=h^(h>>15u);
 return 2u+(h&(partAnchorHeads()-1u));
}
// An empty same-pose selection is a real empty result.  Do not manufacture one
// anchor instance merely because the source has a declared part bank.
// The native indirect count/frontier is the live candidate extent. The packet's
// first word is the source edge count used only by the anchor-only chart reader
// above; it is not a capacity or a source-instance census.
fn partAnchorCount()->u32{return partAnchorVertices[1]*selectedCount();}
// Batch frontier preparation can carry the registered part count in the
// frontier packet; the anchor consumer still uses partAnchorCount() and its
// declared source bank.
fn prepareAnchorParts()->u32{return params.frontier.z;}
fn partAnchorCanonicalBase(parts:u32)->u32{
 // All source packet kinds share this directory boundary. The old fallback
 // to table length let legacy records overwrite canonical payload storage.
 return partAnchorVertices[0u];
}
fn partAnchorTableCapacity(parts:u32)->u32{
 let total=arrayLength(&partAnchorTable);let base=atomicLoad(&partAnchorTable[1]);let canonical=partAnchorCanonicalBase(parts);
 return select(0u,(canonical-base)/PART_ANCHOR_RECORD_WORDS,base<canonical&&canonical<=total);
}
fn partAnchorReserve(parts:u32,samples:u32)->vec2u{
 let record=atomicAdd(&partAnchorAlloc[0],1u);
 let canonical=atomicAdd(&partAnchorAlloc[1],samples);
 let recordCapacity=partAnchorTableCapacity(parts);let canonicalBase=partAnchorCanonicalBase(parts);
 let recordAt=partAnchorRecord(record);let canonicalAt=canonicalBase+canonical*PART_ANCHOR_CANONICAL_WORDS;
 // Legacy six-anchor packets have no canonical payload: their canonical
 // base is the table end. A prior source may already have advanced the shared
 // sample cursor, but that cannot exhaust an empty range or reject its record.
 if(record>=recordCapacity||(samples>0u&&canonicalAt+samples*PART_ANCHOR_CANONICAL_WORDS>arrayLength(&partAnchorTable))){
  atomicStore(&partAnchorAlloc[2],1u);return vec2u(0xffffffffu,0xffffffffu);
 }
 atomicAdd(&partAnchorAlloc[3],1u);return vec2u(recordAt,canonicalAt);
}
fn partAnchorInvalidate(at:u32,instance:u32){
 atomicStore(&partAnchorTable[at+2u],0u);atomicStore(&partAnchorTable[at+3u],0u);
 atomicStore(&partAnchorTable[at+8u],0u);atomicStore(&partAnchorTable[at+9u],0u);atomicStore(&partAnchorTable[at+10u],0u);atomicStore(&partAnchorTable[at+11u],0u);
 atomicStore(&partAnchorTable[at+12u],instance);atomicStore(&partAnchorTable[at+13u],0u);atomicStore(&partAnchorTable[at+14u],0u);atomicStore(&partAnchorTable[at+15u],0u);
}
fn partAnchorCanonicalAt(parts:u32)->u32{return 2u+parts*6u;}
fn partAnchorCanonicalReady(parts:u32)->bool{
 let at=partAnchorCanonicalAt(parts);return at+PART_ANCHOR_CANONICAL_HEADER_WORDS<=arrayLength(&partAnchorVertices)&&
  partAnchorVertices[at]==${TOPOLOGY_PART_CANONICAL_MAGIC}u&&partAnchorVertices[at+1u]==${TOPOLOGY_PART_CANONICAL_VERSION}u&&
  partAnchorVertices[at+2u]>=at+PART_ANCHOR_CANONICAL_HEADER_WORDS&&partAnchorVertices[at+3u]==parts&&
  partAnchorVertices[at+4u]>=partAnchorVertices[at+2u]+parts*${TOPOLOGY_PART_CANONICAL_RECORD_WORDS}u&&
  partAnchorVertices[at+5u]%${TOPOLOGY_PART_CANONICAL_SAMPLE_WORDS}u==0u&&
  partAnchorVertices[at+6u]==${TOPOLOGY_PART_CANONICAL_RECORD_WORDS}u&&partAnchorVertices[at+7u]==${TOPOLOGY_PART_CANONICAL_SAMPLE_WORDS}u&&
  partAnchorVertices[0u]>0u&&partAnchorVertices[at+9u]>0u&&
  partAnchorVertices[at+2u]+parts*${TOPOLOGY_PART_CANONICAL_RECORD_WORDS}u<=arrayLength(&partAnchorVertices)&&
  partAnchorVertices[at+4u]+partAnchorVertices[at+5u]<=arrayLength(&partAnchorVertices);
}
// Build the source-owned cache only when the generation-owned homes are being
// rebuilt. A canonical part is admitted at a level only when every complete
// source segment's exact lattice-cell closure is in one interior node. The
// source endpoint/cell table is the producer's original topology; no native
// draw count, projected bounds, or six-anchor approximation participates.
fn partFrontierCacheBuild(part:u32){
 let parts=partAnchorVertices[1u];
 for(var ordinal=0u;ordinal<4u;ordinal++){partAnchorVertices[partFrontierCacheAt(part,ordinal)]=0u;}
 // Packet lifetime is distinct from the geometry home's lifetime. A newly
 // registered packet starts at zero even when that home is retained. Mark
 // complete inspection, including unsupported content, separately from the
 // current camera's ownership bit so that case is built once, not never.
 partAnchorVertices[partFrontierMarkerAt(part)]=0x40000000u;
 if(!persistentHome()||!chartOn()||chartSurface()||!cornersCertified()||!partAnchorCanonicalReady(parts)||part>=parts){return;}
 let metadata=partAnchorCanonicalAt(parts);let record=metadata+PART_ANCHOR_CANONICAL_HEADER_WORDS+part*${TOPOLOGY_PART_CANONICAL_RECORD_WORDS}u;
 let flags=partAnchorVertices[record+3u];let sampleCount=partAnchorVertices[record+5u];
 if((flags&${TOPOLOGY_PART_CANONICAL_FLAG_COMPLETE}u)==0u||sampleCount==0u){return;}
 var nodes:array<vec3i,4>;var valid:array<bool,4>;var seen:array<bool,4>;
 for(var ordinal=0u;ordinal<4u;ordinal++){nodes[ordinal]=vec3i(0);valid[ordinal]=true;seen[ordinal]=false;}
 let sampleAt=partAnchorVertices[record+4u];
 for(var sample=0u;sample<sampleCount;sample++){
  let at=sampleAt+sample*${TOPOLOGY_PART_CANONICAL_SAMPLE_WORDS}u;let a=partAnchorVertices[at];let b=partAnchorVertices[at+1u];
  if(a>=arrayLength(&partCodes)||b>=arrayLength(&partCodes)){for(var ordinal=0u;ordinal<4u;ordinal++){valid[ordinal]=false;}break;}
  let ca=cellOf(a);let cb=cellOf(b);if(any(ca>=partExtent())||any(cb>=partExtent())||faceOf(a)>5u||faceOf(b)>5u||faceMask(vec3i(ca))==0u||faceMask(vec3i(cb))==0u){for(var ordinal=0u;ordinal<4u;ordinal++){valid[ordinal]=false;}break;}
  let la=voxelRestLatticeCorner(ca,faceOf(a),a%4u);let lb=voxelRestLatticeCorner(cb,faceOf(b),b%4u);let around=segmentClusters(la,lb,1u);
  for(var ordinal=0u;ordinal<4u;ordinal++){
   if(!valid[ordinal]){continue;}let block=2u<<ordinal;var nodeSet=seen[ordinal];var node=nodes[ordinal];
   for(var k=0u;k<8u;k++){
    if(around[k].w==0){continue;}let c=around[k].xyz;let candidate=c/i32(block);
    if(any(c<vec3i(0))||any(c>=vec3i(partExtent()))||!levelInside(candidate,block)||clusterBorder(candidate,block)){valid[ordinal]=false;break;}
    if(!nodeSet){node=candidate;nodeSet=true;}else if(any(node!=candidate)){valid[ordinal]=false;break;}
   }
   if(valid[ordinal]){nodes[ordinal]=node;seen[ordinal]=nodeSet;if(!nodeSet){valid[ordinal]=false;}}
  }
 }
 for(var ordinal=0u;ordinal<4u;ordinal++){
  if(valid[ordinal]&&seen[ordinal]){let block=2u<<ordinal;let slot=clusterSlot(nodes[ordinal],block);valid[ordinal]=slot!=NONE_CLUSTER&&clusterDeclared(nodes[ordinal],block)&&clusterCertified(nodes[ordinal],block);}
  if(valid[ordinal]){partAnchorVertices[partFrontierCacheAt(part,ordinal)]=0x80000000u|levelIndex(nodes[ordinal],2u<<ordinal);}
 }
}
fn partFrontierMarker(part:u32){
 let parts=partAnchorVertices[1u];if(part>=parts||!partAnchorCanonicalReady(parts)){return;}
 var owned=false;
 if(persistentHome()&&frontierOn()&&!chartSurface()&&cornersCertified()){
  for(var ordinal=0u;ordinal<4u;ordinal++){
   let cached=partAnchorVertices[partFrontierCacheAt(part,ordinal)];if((cached&0x80000000u)==0u){continue;}
   let block=2u<<ordinal;let index=cached&0x7fffffffu;if(index>=levelClusters(block)){continue;}
   let node=levelCoord(index,block);let slot=clusterSlot(node,block);
   if(slot!=NONE_CLUSTER&&clusterAtLevel(node,block)&&clusterCertified(node,block)){owned=true;break;}
  }
 }
 let built=partAnchorVertices[partFrontierMarkerAt(part)]&0x40000000u;
 partAnchorVertices[partFrontierMarkerAt(part)]=built|select(0u,0x80000000u,owned);
}
// The post-certification contactVertices stage writes one current-frame marker
// into the reserved source packet word. The anchor entry reads only that
// marker, so it needs no contact/home binding and performs no sample walk.
fn partAnchorCachedFrontierOwned(parts:u32,part:u32)->bool{
 if(!chartOn()||!persistentHome()||chartSurface()||!cornersCertified()||!partAnchorCanonicalReady(parts)||part>=parts){return false;}
 return (partAnchorVertices[partFrontierMarkerAt(part)]&0x80000000u)!=0u;
}
fn partAnchorCanonicalSample(parts:u32,part:u32,index:u32,destinationBase:u32,instance:u32)->u32{
 let metadata=partAnchorCanonicalAt(parts);let record=metadata+PART_ANCHOR_CANONICAL_HEADER_WORDS+part*${TOPOLOGY_PART_CANONICAL_RECORD_WORDS}u;
 let sourceAt=partAnchorVertices[record+4u]+index*${TOPOLOGY_PART_CANONICAL_SAMPLE_WORDS}u;
 let destination=destinationBase+index*PART_ANCHOR_CANONICAL_WORDS;
 if(destination+PART_ANCHOR_CANONICAL_WORDS>arrayLength(&partAnchorTable)){return destination;}
 let a=partAnchorVertices[sourceAt];let b=partAnchorVertices[sourceAt+1u];let t=bitcast<f32>(partAnchorVertices[sourceAt+2u]);
 let ca=params.clip*vec4f(mix(position(a,instance),position(b,instance),clamp(t,0.0,1.0)),1.0);
 // Keep homogeneous clip data, not only page x/y.  The consumer can recover
 // perspective phase through z/w and the exact source u remains independent
 // of near-plane clipping; a clipped row is marked invalid, never re-anchored.
 atomicStore(&partAnchorTable[destination],bitcast<u32>(ca.x));atomicStore(&partAnchorTable[destination+1u],bitcast<u32>(ca.y));
 atomicStore(&partAnchorTable[destination+2u],bitcast<u32>(ca.z));atomicStore(&partAnchorTable[destination+3u],bitcast<u32>(ca.w));
 atomicStore(&partAnchorTable[destination+4u],partAnchorVertices[sourceAt+3u]);atomicStore(&partAnchorTable[destination+5u],a);
 atomicStore(&partAnchorTable[destination+6u],b);atomicStore(&partAnchorTable[destination+7u],partAnchorVertices[sourceAt+2u]);
 // Owner, instance, feature, context and flags are already held ONCE in
 // this sample range's directory record. The consumer checks that exact
 // directory before reading any sample. Clip.w supplies the validity bit.
 return destination;
}
// Projection: each thread writes ONLY its own record.
@compute @workgroup_size(64) fn partAnchors(@builtin(global_invocation_id) gid:vec3u){partAnchorAt(gid.x+gid.y*params.owners[3].w*64u);}
fn partAnchorAt(id:u32){
 let parts=partAnchorVertices[1];if(parts==0u||id>=partAnchorCount()){return;}
 let ordinal=id/parts;let part=id%parts;let instance=selectedInstance(ordinal);
 var lo=vec2f(3.402823e38);var hi=vec2f(-3.402823e38);var found=false;var owner=vec2u(0u);var ownerVertex=0xffffffffu;var owned=false;
 for(var k=0u;k<6u;k++){
  let v=partAnchorVertices[2u+part*6u+k];if(v==0xffffffffu||v>=params.vertexCapacity){continue;}
  if(!owned){owner=edgeOwner(v,instance);ownerVertex=v;owned=true;}
  let c=params.clip*vec4f(position(v,instance),1.0);if(!(c.w>0.00001)){continue;}
  let ndc=c.xy/c.w;lo=min(lo,ndc);hi=max(hi,ndc);found=true;
 }
 // Do not reserve a table record until this candidate has passed the same
 // owner/selection/merge/clip predicates that decide whether it contributes
 // to the current camera frontier. Selection is keyed by the exact owner of
 // this part's anchor, avoiding the first-mesh-vertex/source-incidence read
 // that the general relation path needs. The source packet remains complete;
 // only live projected parts consume the bounded owner table.
 if(!owned||!found||ownerVertex==0xffffffffu){return;}
 if(params.selection.z!=0u&&selectedOwners[params.owners[3].z+instance]==0u){return;}
 if(params.selection.z==0u&&params.selection.y!=0u&&owner.y!=0xffffffffu){
  var selectedLo=0u;var selectedHi=params.selection.x;
  loop{if(selectedLo>=selectedHi){break;}let mid=(selectedLo+selectedHi)/2u;let value=selectedOwners[mid];if(value<owner.y){selectedLo=mid+1u;}else{selectedHi=mid;}}
  if(selectedLo>=params.selection.x||selectedOwners[selectedLo]!=owner.y){return;}
 }
 if(ownerUnseen(ownerVertex,instance)||ownerMerged(owner)){return;}
 // The source-owned cache was built once with complete canonical source
 // segments. A selected certified ancestor suppresses only the exact part it
 // covers; every unsupported/mixed/seam part keeps this incumbent fine path.
 if(partAnchorCachedFrontierOwned(parts,part)){return;}
 let metadata=partAnchorCanonicalAt(parts);let canonical=partAnchorCanonicalReady(parts);
 var sourceRecord=0u;var sampleCount=0u;
 if(canonical){sourceRecord=metadata+PART_ANCHOR_CANONICAL_HEADER_WORDS+part*${TOPOLOGY_PART_CANONICAL_RECORD_WORDS}u;sampleCount=partAnchorVertices[sourceRecord+5u];}
 let allocation=partAnchorReserve(parts,sampleCount);if(allocation.x==0xffffffffu){return;}
 let at=allocation.x;let destinationBase=allocation.y;
 atomicStore(&partAnchorTable[at],owner.x);atomicStore(&partAnchorTable[at+1u],owner.y);atomicStore(&partAnchorTable[at+2u],select(0u,part+1u,owned&&found));
 atomicStore(&partAnchorTable[at+4u],bitcast<u32>(lo.x));atomicStore(&partAnchorTable[at+5u],bitcast<u32>(lo.y));
 atomicStore(&partAnchorTable[at+6u],bitcast<u32>(hi.x));atomicStore(&partAnchorTable[at+7u],bitcast<u32>(hi.y));
 if(canonical){
  atomicStore(&partAnchorTable[at+8u],select(0u,destinationBase, sampleCount>0u));
  atomicStore(&partAnchorTable[at+9u],sampleCount);
  atomicStore(&partAnchorTable[at+10u],PART_ANCHOR_CANONICAL_WORDS);
  atomicStore(&partAnchorTable[at+11u],partAnchorVertices[sourceRecord+3u]);
  atomicStore(&partAnchorTable[at+12u],instance);atomicStore(&partAnchorTable[at+13u],partAnchorVertices[sourceRecord+1u]);
  atomicStore(&partAnchorTable[at+14u],partAnchorVertices[sourceRecord]);atomicStore(&partAnchorTable[at+15u],sourceRecord);
  for(var sample=0u;sample<sampleCount;sample++){partAnchorCanonicalSample(parts,part,sample,destinationBase,instance);}
 }else{
  atomicStore(&partAnchorTable[at+8u],0u);atomicStore(&partAnchorTable[at+9u],0u);atomicStore(&partAnchorTable[at+10u],0u);atomicStore(&partAnchorTable[at+11u],0u);
  atomicStore(&partAnchorTable[at+12u],instance);atomicStore(&partAnchorTable[at+13u],0u);atomicStore(&partAnchorTable[at+14u],0u);atomicStore(&partAnchorTable[at+15u],0u);
 }
 let bucket=partAnchorBucket(owner,part,instance);let record=(at-atomicLoad(&partAnchorTable[1]))/PART_ANCHOR_RECORD_WORDS;let next=atomicExchange(&partAnchorTable[bucket],record+1u);atomicStore(&partAnchorTable[at+3u],0x80000000u|next);
}
// Stale invalidation: zero every head and link word once per frame before the
// first projection (a record of a source not drawn this frame is unreachable).
@compute @workgroup_size(64) fn partAnchorClear(@builtin(global_invocation_id) gid:vec3u){partAnchorClearAt(gid.x+gid.y*params.owners[3].w*64u);}
fn partAnchorClearAt(id:u32){
 let heads=partAnchorHeads();let total=arrayLength(&partAnchorTable);
 if(id<heads){atomicStore(&partAnchorTable[2u+id],0u);return;}
 let record=id-heads;let at=partAnchorRecord(record);if(at+PART_ANCHOR_RECORD_WORDS<=total){atomicStore(&partAnchorTable[at+3u],0u);}
}
@compute @workgroup_size(64) fn contactClear(@builtin(global_invocation_id) gid:vec3u){contactClearAt(gid.x+gid.y*params.owners[3].w*64u);}
fn contactClearAt(id:u32){
 if(!posedContacts()||native[0]==0u){return;}
 let slots=contactSlots();let count=residualEnd();
 if(id<count){let base=partBase();let low=cellParts()&&id>=base+slots&&id<base+slots*2u;atomicStore(&contactWork[id],select(0u,0xffffffffu,low));}
}

// chunk's block and cell size, and the part cells the part flags read),
// voxelHome, voxelBin (every quad: its cell face; the cluster occupancy unless
// retained), chartCorners (every vertex's lattice point), then contactVertices and
// contactEdges build the fine posed incidence ONLY for a vertex or edge the
// selected coarse representation does not own (voxelSpanCoarse on the
// completed table). A chunk at block 1, a non-voxel source and every vertex or
// edge touching an unrepresentable or border column keep the full fine work.
@compute @workgroup_size(64) fn voxelVote(@builtin(global_invocation_id) gid:vec3u){voxelVoteAt(gid.x+gid.y*params.owners[3].w*64u);}
fn voxelVoteAt(v:u32){
 if(!posedContacts()){return;}
 voxelColumnVertex(v);addPartVertex(v,position(v,0u));
}
@compute @workgroup_size(64) fn voxelBin(@builtin(global_invocation_id) gid:vec3u){voxelBinAt(gid.x+gid.y*params.owners[3].w*64u);}
fn voxelBinAt(id:u32){
 if(!posedContacts()||id>=sourceEdgeCount()||!edgeInDraw(id)){return;}
 if(id%5u==0u){chartQuad(id);}
}

// (after voxelBin wrote every cell's faces; before any fine incidence).
@compute @workgroup_size(64) fn chartCorners(@builtin(global_invocation_id) gid:vec3u){chartCornerAt(gid.x+gid.y*params.owners[3].w*64u);}
fn chartCornerAt(v:u32){
 if(!posedContacts()||!chartOn()||v>=min(sourceFaceCount()*2u,native[0]/6u*4u)){return;}
 chartClassifyAt(v,min(sourceFaceCount()*2u,native[0]/6u*4u));
 let lattice=vertexLattice(v);if(lattice.w==0){if(!chartSurface()){chartUnsupported();}return;}
 let at=cornerBase()+latticeIndex(lattice.xyz)*4u;let p=position(v,0u);
 atomicStore(&contactWork[at+1u],bitcast<u32>(p.x));atomicStore(&contactWork[at+2u],bitcast<u32>(p.y));atomicStore(&contactWork[at+3u],bitcast<u32>(p.z));
 atomicStore(&contactWork[at],1u);
 if(v<chartVertices()){let l=vec3u(lattice.xyz);atomicStore(&contactWork[latticeBase()+v],(l.x|(l.y<<9u)|(l.z<<18u))+1u);}
}

// chartCorners (claims, face centroids and surface states complete) and
// before contactVertices / contactEdges / boundaries read the flags. Surface
// charts only (closed clusters are certified by construction). TOPOLOGY
// certificate: computed once per home rebuild (kept with a retained home).
// POSE certificate: every draw (voxelHome resets it in a retained home).
@compute @workgroup_size(64) fn chartCertify(@builtin(global_invocation_id) gid:vec3u){chartCertifyAt(gid.x+gid.y*params.owners[3].w*64u);}
fn chartCertifyAt(i:u32){
 if(!posedContacts()||native[0]==0u||!chartOn()){return;}

 // level once per home build (closed and surface charts alike), after
 // voxelBin/chartCorners completed occupancy and classification; a retained
 // level keeps it (CLUSTER_EDGES_DONE). Level 2 has the most clusters, so the
 // dispatch over chart.clusters covers every level's index range.
 for(var lb=2u;lb<=16u;lb*=2u){
  if(!levelHomed(lb)){continue;}
  let ln=(partExtent()+vec3u(lb-1u))/lb;if(i>=min(ln.x*ln.y*ln.z,levelCapacity(lb))){continue;}
  let s=levelHomeBase(lb)+4u+i*4u;let c=vec3i(i32(i%ln.x),i32((i/ln.x)%ln.y),i32(i/(ln.x*ln.y)));
  compileClusterEdges(s,c,lb);

  if(lb==2u&&frontierOn()){storeHeld(s,c);}
  if(!chartSurface()){continue;}
  // TOPOLOGY certificate (any cluster a segment of this level may read; kept with the home).
  if((atomicLoad(&contactWork[s+3u])&CLUSTER_NEAR_DONE)==0u){atomicOr(&contactWork[s+3u],CLUSTER_NEAR_DONE|select(0u,CLUSTER_NEAR_KNOWN,nearKnown(c,lb)));}
  // POSE certificate: declared clusters whose OWN node admits this level, a

  // the parent's FORCED bit, which another thread of this same pass stamps,
  // so on a home-rebuild draw a selected cluster could miss its certificate
  // (a one-draw gap after every content change). The own node's bit was
  // stamped by this thread just above. An extra certificate is never emitted:
  // every reader (segment ownership, clusterEmit) also requires clusterLevelOk.
  let w=atomicLoad(&contactWork[s]);
  if(w!=0u&&clusterOwnLevel(c,lb)){let k=~w;let e=partExtent();let ok=cellFrame(vec3u(k%e.x,(k/e.x)%e.y,k/(e.x*e.y))).ok;atomicOr(&contactWork[s+2u],CLUSTER_FRAME_DONE|select(0u,CLUSTER_FRAME_OK,ok));}
 }
}

// KEPT when the home's stored token equals this source's generation token
// (params.selection.w > 1: the cell chart is generation-owned, so posing and
// moving never invalidate it) and its stored block equals this frame's vote.
// Kept: only the per-view emission claims (bit 0 of each cluster's flags word)
// are reset. Otherwise every cluster word is cleared and voxelBin rebuilds it. Thread 0 publishes the decision
// (retainedWord); the header itself is rewritten only in contactVertices, a
// later pass, so every thread here reads the same header.
@compute @workgroup_size(64) fn voxelHome(@builtin(global_invocation_id) gid:vec3u){voxelHomeAt(gid.x+gid.y*params.owners[3].w*64u);}
fn voxelHomeAt(id:u32){
 if(!posedContacts()||native[0]==0u||!persistentHome()||!chartOn()){return;}

 // dispatches chartHomeWords(levels)). Each level home is kept or rebuilt
 // under its OWN header (token and its level code), whatever this frame's vote,
 // so every level is built when content changes and is resident before a
 // transition selects it.
 var lb=0u;var local=id;
 for(var b=2u;b<=16u;b*=2u){let size=4u+levelClusters(b)*4u;if(local<size){lb=b;break;}local-=size;}
 if(lb==0u){return;}
 let h=levelHomeBase(lb);if(local>=4u+levelCapacity(lb)*4u){return;}
 let token=params.selection.w;
 let kept=token>1u&&atomicLoad(&contactWork[h+1u])==token&&atomicLoad(&contactWork[h+2u])==levelCode(lb);
 if(local==0u){
  atomicStore(&contactWork[levelRetainedWord(lb)],select(0u,1u,kept));
  if(lb==votedClusterBlock()){atomicStore(&contactWork[retainedWord()],select(0u,1u,kept));}
  return;
 }
 if(local<4u){return;}
 if(kept){if((local-4u)%4u==2u){atomicAnd(&contactWork[h+local],~15u);}return;}
 atomicStore(&contactWork[h+local],0u);
}
@compute @workgroup_size(64) fn contactVertices(@builtin(global_invocation_id) gid:vec3u){contactVertexAt(gid.x+gid.y*params.owners[3].w*64u);}
fn contactVertexAt(v:u32){
 // This dispatch follows chartCertify, so current-frame forced bits and
 // certificates are complete before the marker is read by partAnchors. Its
 // extra threads are source-part markers, not topology/instance records.
 if(v<partAnchorVertices[1u]){
  if(persistentHome()&&(!levelRetained(2u)||(partAnchorVertices[partFrontierMarkerAt(v)]&0x40000000u)==0u)){partFrontierCacheBuild(v);}
  partFrontierMarker(v);
 }
 if(!posedContacts()||v>=min(sourceFaceCount()*2u,native[0]/6u*4u)){return;}

 // this content token and block, or nothing after an overflow (never kept).
 if(v==0u&&persistentHome()){
  let vote=atomicLoad(&contactWork[chunkBlockWord()]);

  atomicStore(&contactWork[params.owners[0].w+3u],vote);

  // of which voxelBin built or kept this draw; an overflow certifies none.
  let complete=atomicLoad(&contactWork[overflowWord()])==0u;
  for(var b=2u;b<=16u;b*=2u){
   let h=levelHomeBase(b);
   atomicStore(&contactWork[h+1u],select(0u,params.selection.w,complete));atomicStore(&contactWork[h+2u],levelCode(b));
  }
 }
 let residual=residualReady();var sourceV=v;
 if(residual){if(v>=residualVertexCount()){return;}sourceV=residualVertexAt(v);}
 if(sourceV>=min(sourceFaceCount()*2u,native[0]/6u*4u)){return;}
 // Owned by the selected coarse representation: no fine port is ever read.
 if(!residual&&voxelVertexCoarse(sourceV)){return;}
 let q=position(sourceV,0u);let slots=contactSlots();let hash=contactPositionHash(q);
 for(var probe=0u;probe<slots;probe++){
  let at=(hash+probe)&(slots-1u);var first=atomicLoad(&contactWork[at]);
  // Retry only a spurious EMPTY exchange. Never wait for another invocation
  // to publish a key: the claimed word is already a complete source address.
  loop{if(first!=0u){break;}let claim=atomicCompareExchangeWeak(&contactWork[at],0u,sourceV+1u);first=claim.old_value;if(claim.exchanged){first=sourceV+1u;break;}}
  if(all(position(first-1u,0u)==q)){atomicMin(&contactWork[at],sourceV+1u);break;}
 }
}
@compute @workgroup_size(64) fn contactEdges(@builtin(global_invocation_id) gid:vec3u){contactEdgeAt(gid.x+gid.y*params.owners[3].w*64u);}
fn contactEdgeAt(id:u32){
 if(!posedContacts()){return;}
 let residual=residualReady();var edgeId=id;if(residual){if(id>=residualEdgeCount()){return;}edgeId=residualEdgeAt(id);}else if(id>=sourceEdgeCount()||!edgeInDraw(id)){return;}
 if(cellParts()){
  let code=partCode(halfedge(edge(edgeId).x).x);let base=edgePartBase();
  atomicStore(&contactWork[base+edgeId],code);
  // One edge per quad contributes its held whole-part level; all vertices
  // have completed the bound reduction. Flags commute with contact marking.
  if(edgeId%5u==0u&&smallClosedPart(code)){let at=partSlot(code);if(at!=NO_CONTACT){atomicOr(&contactWork[partFlagsBase()+at],1u);}}
 }
 // Owned by the selected coarse representation: its contact list is never read;

 if(!residual&&voxelEdgeCoarse(edgeId)){return;}
 let key=posedEdgePorts(edgeId);if(key.x==key.y){return;}let slots=contactSlots();let hash=contactHash(key.x^contactHash(key.y));
 for(var probe=0u;probe<slots;probe++){
  let slot=(hash+probe)&(slots-1u);let at=slots+slot;var first=atomicLoad(&contactWork[at]);
  loop{if(first!=0u){break;}let claim=atomicCompareExchangeWeak(&contactWork[at],0u,edgeId+1u);first=claim.old_value;if(claim.exchanged){first=edgeId+1u;break;}}
  if(all(posedEdgePorts(first-1u)==key)){
   atomicMin(&contactWork[at],edgeId+1u);
   let previous=atomicExchange(&contactWork[slots*2u+slot],edgeId+1u);atomicStore(&contactWork[slots*3u+edgeId],previous);
   // Mark the WHOLE part as connected, not just this edge. Read the original
   // immutable codes, not another invocation's not-yet-published work record.
   if(cellParts()&&previous!=0u){connectParts(partCode(halfedge(edge(edgeId).x).x),partCode(halfedge(edge(previous-1u).x).x));}return;
  }
 }
}
// contactEdges links each coincident source edge by atomic insertion order.
// The sort dispatch uses the SAME full/residual edge domain and indirect
// extent as contactEdges. Only the canonical minimum source edge of each
// bucket sorts its existing chain, so this remains one bounded pass over the
// producer's edge work rather than a scan of every hash bucket.
@compute @workgroup_size(64) fn contactSort(@builtin(global_invocation_id) gid:vec3u){contactSortAt(gid.x+gid.y*params.owners[3].w*64u);}
fn contactSortAt(id:u32){
 if(!posedContacts()||native[0]==0u){return;}
 let residual=residualReady();var edgeId=id;
 if(residual){if(id>=residualEdgeCount()){return;}edgeId=residualEdgeAt(id);}
 else if(id>=sourceEdgeCount()||!edgeInDraw(id)){return;}
 // The existing canonical bucket header is the membership certificate. An
 // edge that contactEdges excluded cannot own a bucket; do not recertify its
 // chart geometry here or pull the chart classifier into this integer stage.
 let key=posedEdgePorts(edgeId);if(key.x==key.y){return;}
 let slots=contactSlots();let slot=posedEdgeSlot(edgeId);if(slot==NO_CONTACT){return;}
 // contactEdges atomically keeps the minimum source edge in the bucket
 // header. Exactly that invocation owns the in-place sort.
 if(atomicLoad(&contactWork[slots+slot])!=edgeId+1u){return;}
 var current=atomicLoad(&contactWork[slots*2u+slot]);var sorted=0u;
 loop{
  if(current==0u){break;}
  let next=atomicLoad(&contactWork[slots*3u+(current-1u)]);
  if(sorted==0u||current<sorted){
   atomicStore(&contactWork[slots*3u+(current-1u)],sorted);sorted=current;
  }else{
   var previous=sorted;var after=atomicLoad(&contactWork[slots*3u+(previous-1u)]);
   loop{if(after==0u||after>current){break;}previous=after;after=atomicLoad(&contactWork[slots*3u+(previous-1u)]);}
   atomicStore(&contactWork[slots*3u+(current-1u)],after);atomicStore(&contactWork[slots*3u+(previous-1u)],current);
  }
  current=next;
 }
 // Boundary traversal starts at the per-bucket tail word. It now walks the
 // same ascending source-edge order as the historical nested selector.
 atomicStore(&contactWork[slots*2u+slot],sorted);
}
fn contactMatches(a:u32,b:u32,instance:u32)->bool{
 let x=halfedge(a);let y=halfedge(b);let p=position(x.x,instance);let q=position(x.y,instance);
 if(!edgeInDraw(x.z)||!edgeInDraw(y.z)){return false;}
 return (same(p,position(y.x,instance))&&same(q,position(y.y,instance)))||(same(p,position(y.y,instance))&&same(q,position(y.x,instance)));
}
fn canonicalEdge(id:u32,instance:u32)->u32{
 let e=edge(id);var result=e.x;
 if(e.w>0u){let group=source[9]+(e.w-1u)*2u;let at=source[group];
  for(var i=0u;i<source[group+1u];i++){let other=source[at+i*2u];if(other<result&&contactMatches(e.x,other,instance)){result=other;}}
 }
 return result;
}
fn inDraw(h:u32)->bool{let index=h/3u*3u;let start=native[2]-params.indexBase;return index>=start&&index<start+native[0];}
fn edgeInDraw(id:u32)->bool{let e=edge(id);for(var i=0u;i<e.z;i++){if(inDraw(sourceEdgeCorner(id,i))){return true;}}return false;}
fn validFace(h:u32)->bool{
 for(var k=0u;k<3u;k++){let v=i32(halfedge(h/3u*3u+k).x)+bitcast<i32>(native[3]);if(v<0||v>=i32(params.vertexCapacity)){atomicOr(&status.overflow,8u);return false;}}
 return true;
}
fn triangle(h:u32,instance:u32)->Triangle{
 let face=h/3u;let a=halfedge(face*3u).x;let b=halfedge(face*3u+1u).x;let c=halfedge(face*3u+2u).x;
 let p=mat3x3f(position(a,instance),position(b,instance),position(c,instance));let n=mat3x3f(normal(a,instance),normal(b,instance),normal(c,instance));
 let crossN=cross(p[1]-p[0],p[2]-p[0]);let faceN=crossN/max(length(crossN),0.000000000000000000000000000001);
 let identity=positions[slot(a,instance)].w;
 let owner=select(identity,attributes[slot(a,instance)].w,identity==0xffffffffu);
 let aa=attributes[slot(a,instance)];let ab=attributes[slot(b,instance)];let ac=attributes[slot(c,instance)];
 return Triangle(p,n,faceN*select(-1.0,1.0,dot(faceN,n[0]+n[1]+n[2])>=0.0),owner,select(0u,0x80000000u,identity==0xffffffffu),bitcast<vec3f>(vec3u(aa.y,ab.y,ac.y)),aa.z==${SOURCE_CHART_TAG}u&&ab.z==${SOURCE_CHART_TAG}u&&ac.z==${SOURCE_CHART_TAG}u,sourceFaceMaterial(face));
}
fn front(t:Triangle)->bool{
 let facing=dot(t.face,select(-params.direction.xyz,params.eye.xyz-(t.p[0]+t.p[1]+t.p[2])/3.0,params.eye.w!=0.0))>0.0;
 return select(facing,!facing,params.direction.w==2.0);
}
fn field(p:vec3f,n:vec3f)->f32{
 let toward=select(-params.direction.xyz,params.eye.xyz-p,params.eye.w!=0.0);
 let value=dot(n,toward/max(length(toward),0.000000001));return select(value,0.0,abs(value)<=0.000001);
}
fn slope(t:Triangle)->vec2f{
 let a=params.clip*vec4f(t.p[0],1);let b=params.clip*vec4f(t.p[1],1);let c=params.clip*vec4f(t.p[2],1);
 let n=cross(b.xyz/b.w-a.xyz/a.w,c.xyz/c.w-a.xyz/a.w);
 // Native fragment depth is affine over a projected TRIANGLE, not over
 // distance along the feature. Keep its signed gradient. An edge-on face
 // has no raster plane and cannot supply a fabricated finite slope.
 if(abs(n.z)<=1e-20){return vec2f(0);}
 return -n.xy/n.z;
}
fn isSmooth(t:Triangle)->bool{return min(min(dot(t.n[0],t.n[1]),dot(t.n[1],t.n[2])),dot(t.n[2],t.n[0]))<0.9999;}
// THE POSED MESH BENDS HERE (lines4, 2026-11-13). An occluding contour of a
// smooth surface (kind 1) and an interior turn (kind 3) belong to the surface
// the drawing shows, so they exist only where the POSED mesh itself turns: an
// edge neighbour of the face turns against it by more than float noise and
// less than a crease. A flat face with interpolated normals (a smooth-normal
// box) has normal-zero and kr-zero aliases with no turn; a plane-class source
// lifted into hills by its material (snow-trails' snowfield) has real ones.
// The answer rides in ports.w bit BENT; the run pass reads it for form 1 only.
const BENT=0x40000000u;
fn restingNormal(face:u32,instance:u32)->vec3f{
 let a=position(halfedge(face*3u).x,instance);let n=cross(position(halfedge(face*3u+1u).x,instance)-a,position(halfedge(face*3u+2u).x,instance)-a);
 return n/max(length(n),1e-30);
}
fn bends(face:u32,instance:u32)->u32{
 // Only a source whose REST mesh is plane-class (owners[1].x, planarRest)
 // reads the tag; every other source skips the neighbour reads (h11 cost).
 if(params.owners[1].x==0u){return 0u;}
 let n=restingNormal(face,instance);
 for(var k=0u;k<3u;k++){
  let e=edge(halfedge(face*3u+k).z);
  var members=1u;var membersAt=0u;if(e.w>0u){let group=source[9]+(e.w-1u)*2u;membersAt=source[group];members=source[group+1u];}
  for(var member=0u;member<members;member++){
   let h=select(e.x,source[membersAt+member*2u],e.w>0u);if(!contactMatches(e.x,h,instance)){continue;}
   let item=edge(halfedge(h).z);
   for(var i=0u;i<item.z;i++){
    let incidence=sourceEdgeCorner(halfedge(h).z,i);if(incidence/3u==face||!inDraw(incidence)){continue;}
    let turn=abs(dot(restingNormal(incidence/3u,instance),n));
    if(turn>.70710678&&turn<.9999999){return BENT;}
   }
  }
 }
 return 0u;
}
fn triangleChart(t:Triangle,point:vec3f)->f32{
 let a=t.p[1]-t.p[0];let b=t.p[2]-t.p[0];let q=point-t.p[0];
 let aa=dot(a,a);let ab=dot(a,b);let bb=dot(b,b);let denominator=aa*bb-ab*ab;
 if(abs(denominator)<1e-20){return t.chart.x;}
 let v=(bb*dot(q,a)-ab*dot(q,b))/denominator;let w=(aa*dot(q,b)-ab*dot(q,a))/denominator;
 return dot(vec3f(1.0-v-w,v,w),t.chart);
}
fn emit(a:vec3f,b:vec3f,t:Triangle,kind:u32,address:u32,ports:vec4u,coordinates:vec4f,partFar:bool){emitLevel(a,b,t,kind,address,ports,coordinates,partFar,false);}
fn emitLevel(a:vec3f,b:vec3f,t:Triangle,kind:u32,address:u32,ports:vec4u,coordinates:vec4f,partFar:bool,outline:bool){
 // A subpixel or zero-length span can be the connector between two different
 // source-edge crossings. Keep its incidence; glyph articulation may contract
 // it as a relationship, but deleting it here opens otherwise closed forms.
 let at=atomicAdd(&status.count,1u);
 if(at>=params.capacity){atomicOr(&status.overflow,2u);return;}
 let gradient=slope(t);
 // The mesh definition stays intrinsic; this support comes from the ACTUAL
 // material vertex result. Presence is explicit for older untagged records.
 let center=params.clip*vec4f((t.p[0]+t.p[1]+t.p[2])/3.0,1);
 var detail=coordinates;var tag=kind|t.producer;
 // Keep the posed feature's world-up meaning through perspective. This is
 // geometry, not a material/subject recipe or a screen-angle guess.
 let axis=b-a;
 if(kind==0u||kind==2u){if(length(axis.xz)<=length(axis)*.00001){tag|=0x20000000u;}}
 if(abs(center.w)>.000001){detail.z=center.x/center.w;detail.w=center.y/center.w;tag=tag|0x40000000u;}
 var chart=vec4f(0.0,0.0,f32(t.material),0.0);if(t.charted){chart=vec4f(triangleChart(t,a),triangleChart(t,b),f32(t.material),1.0);}
 chart.w+=select(0.0,2.0,partFar)+select(0.0,4.0,outline);

 // supplied contact corner of its owner carries that corner (id+1) in chart.w bits 3-12 (end a) and 13-22
 // (end b); the run stage keys that end's line node by the point
 // (gpu-construction-runs key: CONTACT_DOMAIN), so the two owners' edges meet.
 if(kind==0u||kind==2u){let owner=vec2u(params.domain,t.owner);chart.w+=f32((contactCorner(owner,ports.x)<<3u)|(contactCorner(owner,ports.y)<<13u));}
 relations[at]=Relation(vec4f(a,gradient.x),vec4f(b,gradient.y),vec4u(params.domain,t.owner,tag,address),ports,detail,chart);
}
fn prepareIn(){
 let invalid=native[5]!=0u||sourceVertexCount()>params.vertexCapacity||native[2]<params.indexBase||native[2]-params.indexBase+native[0]>sourceFaceCount()*3u||(native[2]-params.indexBase)%3u!=0u||native[0]%3u!=0u;
 // A source/instance range is not capped at one dispatch dimension. Use
 // the device's actual X extent, then continue the SAME source IDs in Y.
 let limit=params.owners[3].w;
 let edges=select((sourceBoundarySlots()*selectedCount()+63u)/64u,0u,invalid);let faces=select((sourceFaceCount()*selectedCount()+63u)/64u,0u,invalid);
 dispatch[0]=min(edges,limit);dispatch[1]=max(1u,(edges+limit-1u)/limit);dispatch[2]=1u;
 dispatch[3]=min(faces,limit);dispatch[4]=max(1u,(faces+limit-1u)/limit);dispatch[5]=1u;
 // Native GPU visibility/counts own the contact frontier too. Previously
 // every hidden chunk still cleared its complete hash and launched contact
 // work from registration capacity before the kernels noticed instanceCount0.
 let contactActive=!invalid&&posedContacts()&&native[0]!=0u;

 // chunk words (columnEnd), the same extent contactClearAt clears and the host
 // allocates (contactWords); short of it, column claims, tops/floors and the
 // chunk level vote would persist into the next frame.
 // The indirect clear must reach the SAME complete optional suffix that
 // contactClearAt clears and the host allocates. Stopping at columnEnd left
 // previous-frame residual headers, face membership and compact IDs live.
 let words=residualEnd();
 let clearGroups=select(0u,(words+63u)/64u,contactActive);
 let vertexWork=max(min(sourceFaceCount()*2u,native[0]/6u*4u),prepareAnchorParts());
 let vertexGroups=select(0u,(vertexWork+63u)/64u,contactActive);
 let edgeGroups=select(0u,(min(sourceEdgeCount(),native[0]/6u*5u)+63u)/64u,contactActive);
 dispatch[6]=min(clearGroups,limit);dispatch[7]=max(1u,(clearGroups+limit-1u)/limit);dispatch[8]=1u;
 dispatch[9]=min(vertexGroups,limit);dispatch[10]=max(1u,(vertexGroups+limit-1u)/limit);dispatch[11]=1u;
 dispatch[12]=min(edgeGroups,limit);dispatch[13]=max(1u,(edgeGroups+limit-1u)/limit);dispatch[14]=1u;
 // The anchor passes use the same native count and workgroup limit.  The
 // source input is bound even when this draw has no declared part bank, so a
 // missing registration cannot read an unbound storage slot.
 let anchorCount=select(0u,prepareAnchorParts()*selectedCount(),!invalid);
 let anchorGroups=(anchorCount+63u)/64u;
 dispatch[18]=min(anchorGroups,limit);dispatch[19]=max(1u,(anchorGroups+limit-1u)/limit);dispatch[20]=1u;
 // The admitted frontier is the only live source extent.  Publish the
 // remaining relation/contact arguments from that same count so consumers do
 // not launch the old host reservation and then clear holes.
 let voxelWork=select(0u,min(sourceFaceCount()*2u,native[0]/6u*4u),contactActive);
 let edgeWork=select(0u,min(sourceEdgeCount(),native[0]/6u*5u),contactActive);
 let homeWork=select(0u,4u+levelClusters(2u)*4u+4u+levelClusters(4u)*4u+4u+levelClusters(8u)*4u+4u+levelClusters(16u)*4u,contactActive&&persistentHome()&&chartOn());
 let chartWork=select(0u,max(chartClusters(),prepareAnchorParts()),contactActive);
 dispatch[21]=min((voxelWork+63u)/64u,limit);dispatch[22]=max(1u,((voxelWork+63u)/64u+limit-1u)/limit);dispatch[23]=1u;
 dispatch[24]=min((homeWork+63u)/64u,limit);dispatch[25]=max(1u,((homeWork+63u)/64u+limit-1u)/limit);dispatch[26]=1u;
 dispatch[27]=min((edgeWork+63u)/64u,limit);dispatch[28]=max(1u,((edgeWork+63u)/64u+limit-1u)/limit);dispatch[29]=1u;
 dispatch[30]=min((voxelWork+63u)/64u,limit);dispatch[31]=max(1u,((voxelWork+63u)/64u+limit-1u)/limit);dispatch[32]=1u;
 dispatch[33]=min((chartWork+63u)/64u,limit);dispatch[34]=max(1u,((chartWork+63u)/64u+limit-1u)/limit);dispatch[35]=1u;
 if(!invalid){prepareLevelDispatch();}
 if(invalid){atomicOr(&status.overflow,1u);}
}
@compute @workgroup_size(1) fn prepare(){prepareIn();}
@compute @workgroup_size(64) fn boundaries(@builtin(global_invocation_id) gid:vec3u){boundaryAt(gid.x+gid.y*params.owners[3].w*64u);}
// Planar construction is compiled from actual panel boundary walks. Unlike a
// hull it retains inner openings and unmatched join tails. Only the engine's
// explicit ordinary (affine) vertex path can reuse those straight source
// spans; arbitrary material deformation keeps the complete fine producer.
fn sourceBoundarySlots()->u32{return sourceEdgeCount();}

// TASK,23:49UTC"). Reads the SOURCE_PATH level section (compileSourcePathProgram).
// Each site resolves against the ACTUAL posed source: a vertex, or the posed
// face's corners with its authored barycentric weights. Path segment g is drawn
// by the boundary thread of source edge g mod E (the principal-chain pattern);
// no dispatch, binding or host extent changes. Pair B flagged reversed is
// emitted from A's start, so both rails progress in one source phase. Ends use
// the authored port table (a shared port is one authored join); interior sites
// own their site port. Only faces inside an authored covered range give up
// their fine edges (authoredOwnsEdge); every other face keeps its incumbent.
fn authoredSitePoint(k:u32,instance:u32)->vec3f{
 let s=sourcePathSite(k);if(s[0]==0u){return position(s[1],instance);}
 let f=s[1]*3u;
 return position(halfedge(f).x,instance)*bitcast<f32>(s[2])+position(halfedge(f+1u).x,instance)*bitcast<f32>(s[3])+position(halfedge(f+2u).x,instance)*bitcast<f32>(s[4]);
}
fn authoredSitePort(p:array<u32,9>,local:u32)->u32{
 let last=p[5]-1u;
 if(local==0u||local==last){let port=select(p[1],p[2],local==last);return 0xd8000000u|(sourcePathPortFirst(port)&0x07ffffffu);}
 return 0xd0000000u|((p[4]+local)&0x07ffffffu);
}
fn authoredPathAt(g:u32,instance:u32){
 let k=sourcePathOfSegment(g);let p=sourcePath(k);let s=g-(p[4]-k);let last=p[5]-1u;
 if(s>=last||p[6]==0xffffffffu){return;}
 let face=p[6]*3u;if(!inDraw(face)||!validFace(face)){return;}
 let v=halfedge(face).x;if(ownerUnseen(v,instance)||ownerMerged(edgeOwner(v,instance))){return;}
 let t=triangle(face,instance);if(dot(t.face,t.face)<0.1||(!front(t)&&params.direction.w!=1.0)){return;}
 let reversed=(p[8]&${SOURCE_PATH_REVERSED}u)!=0u;
 let head=select(s,last-s,reversed);let tail=select(s+1u,last-s-1u,reversed);

 // ONE source phase to the run stage (which binds no source packet):
 // 0xb0000000 | role<<26 (0 unpaired, 1 pair A, 2 pair B) | group<<13 (the
 // pair, or the unpaired path) | phase (segment index from A's start). A
 // group or phase beyond 13 bits keeps a unique plain address (role 3, g).
 let paired=p[7]!=0xffffffffu;let role=select(0u,select(1u,2u,(p[8]&${SOURCE_PATH_PAIR_B}u)!=0u),paired);
 let group=select(k,p[7],paired);
 var address=0xbc000000u|(g&0x03ffffffu);
 if(group<8192u&&s<8192u){address=0xb0000000u|(role<<26u)|(group<<13u)|s;}
 emitLevel(authoredSitePoint(p[4]+head,instance),authoredSitePoint(p[4]+tail,instance),t,0u,address,
  vec4u(authoredSitePort(p,head),authoredSitePort(p,tail),p[6],0xffffffffu),vec4f(0),false,false);
}
fn authoredPathsAt(ordinal:u32,instance:u32){
 let total=sourcePathSegmentCount();let stride=sourceBoundarySlots();if(total==0u||stride==0u){return;}
 var g=ordinal;loop{if(g>=total){break;}authoredPathAt(g,instance);g+=stride;}
}
// An incidence (halfedge index) is replaced only when its face lies in an
// authored covered range whose path is emitted by this same draw.
fn authoredOwnsFace(incidence:u32)->bool{
 if(incidence==0xffffffffu){return true;}
 let k=sourcePathCoveredBy(incidence/3u);if(k==0xffffffffu){return false;}
 let owner=sourcePath(k)[6];return owner!=0xffffffffu&&inDraw(owner*3u);
}
fn authoredOwnsEdge(a:u32,b:u32,c:u32,d:u32)->bool{
 if(sourcePathCoverCount()==0u||(a==0xffffffffu&&b==0xffffffffu&&c==0xffffffffu&&d==0xffffffffu)){return false;}
 return authoredOwnsFace(a)&&authoredOwnsFace(b)&&authoredOwnsFace(c)&&authoredOwnsFace(d);
}
fn planarProgramLevel(instance:u32)->bool{
 if((params.owners[1].z&32u)==0u||native[0]==0u||sourcePlanarSpanAt()==0u||!sourcePlanarComplete()){return false;}
 // Both direct and batched dispatches own one slot per original source edge.
 // An unusually fragmented replacement that needs more slots keeps its
 // complete fine producer; it must never truncate the final instance.
 if(sourcePlanarSpanCount()+sourcePanelJoinCount()>sourceBoundarySlots()){return false;}
 if(native[2]!=params.indexBase||native[0]!=sourceFaceCount()*3u){return false;}
 if(!levelOwnerCertified(instance)){return false;}
 let owner=edgeOwner(halfedge(0u).x,instance);
 return formLevel(owner)>=1u;
}
fn coarseFaceOwnerAdmitted(face:u32,instance:u32)->bool{
 if(face>=sourceFaceCount()){return false;}
 let v=halfedge(face*3u).x;let owner=edgeOwner(v,instance);
 return !ownerUnseen(v,instance)&&!ownerMerged(owner);
}
fn planarSpanAt(k:u32,instance:u32){
 if(k>=sourcePlanarSpanCount()){return;}
 let s=sourcePlanarSpan(k);let face=s[2]*3u;

 if(!coarseFaceOwnerAdmitted(s[2],instance)){return;}
 if(sourcePathCoveredBy(s[2])!=0xffffffffu){return;}
 if(!inDraw(face)||!validFace(face)){return;}
 let t=triangle(face,instance);if(dot(t.face,t.face)<0.1||(!front(t)&&params.direction.w!=1.0)){return;}
 let a=position(s[0],instance);let b=position(s[1],instance);
 emitLevel(a,b,t,0u,0xa0000000u|k,vec4u(boundaryPort(s[0],instance),boundaryPort(s[1],instance),s[2],0xffffffffu),vec4f(0),false,true);
}
fn planarJoinAt(k:u32,instance:u32){
 if(k>=sourcePanelJoinCount()){return;}
 let j=sourcePanelJoin(k);let fa=j[2]*3u;let fb=j[3]*3u;
 if(!coarseFaceOwnerAdmitted(j[2],instance)||!coarseFaceOwnerAdmitted(j[3],instance)){return;}
 if(sourcePathCoveredBy(j[2])!=0xffffffffu&&sourcePathCoveredBy(j[3])!=0xffffffffu){return;}
 if(!inDraw(fa)||!validFace(fa)||!inDraw(fb)||!validFace(fb)){return;}
 let a=triangle(fa,instance);let b=triangle(fb,instance);
 if(dot(a.face,a.face)<0.1||dot(b.face,b.face)<0.1){return;}
 let af=front(a);let bf=front(b);var t=a;var kind=0u;
 if(af==bf){
  if(!af&&params.direction.w!=1.0){return;}
  // A coplanar shared border is interior paper, not two false silhouettes.
  // A genuine plane turn keeps the same crease threshold as boundaryAt.
  if(dot(a.face,b.face)>=0.70710678){return;}kind=2u;
 }else if(!af){t=b;}
 emitLevel(position(j[0],instance),position(j[1],instance),t,kind,0xa8000000u|k,vec4u(boundaryPort(j[0],instance),boundaryPort(j[1],instance),j[2],j[3]),vec4f(0),false,true);
}
fn boundaryAtFine(ordinal:u32,instance:u32,selectedOrdinal:u32,residual:bool){
 if(native[0]==0u||!selected(instance)){return;}

 authoredPathsAt(ordinal,instance);
 if(planarProgramLevel(instance)){
  let v=halfedge(0u).x;if(ownerUnseen(v,instance)||ownerMerged(edgeOwner(v,instance))){return;}
  if(ordinal<sourcePlanarSpanCount()){planarSpanAt(ordinal,instance);}else{planarJoinAt(ordinal-sourcePlanarSpanCount(),instance);}return;
 }
 if(ordinal>=sourceEdgeCount()){return;}
 let sourceId=selectedOrdinal*sourceEdgeCount()+ordinal;
 if(!chartEarlyEmitted()){chartEmitAt(sourceId,sourceEdgeCount()*selectedCount());}let e=edge(ordinal);if(ownerUnseen(halfedge(e.x).x,instance)||ownerMerged(edgeOwner(halfedge(e.x).x,instance))){return;}
 // L2 (outlineLevel): this owner is drawn by its COMPILED outline construction
 // only. The thread of source edge k draws hull edge k; no fine boundary,
 // crease, fold or contour of the owner is produced.
 if(outlineLevel(edgeOwner(halfedge(e.x).x,instance))){hullOutlineAt(ordinal,instance);return;}
 // L1 (principalLevel): the thread of source edge k also draws chain k.
 let principal=principalLevel(edgeOwner(halfedge(e.x).x,instance));
 if(principal){principalChainAt(ordinal,instance);}
 if(!edgeInDraw(ordinal)){return;}

 // representation; this fine edge is then owned by it (not expanded), BEFORE

 if(!residual&&posedContacts()){let inc=sourceEdgeCorner(halfedge(e.x).z,0u);if(inDraw(inc)&&validFace(inc)&&voxelColumnEdge(ordinal,triangle(inc,0u),inc/3u)){return;}}
 let contact=posedEdgeSlot(ordinal);var canonical=canonicalEdge(ordinal,instance);
 if(contact!=NO_CONTACT){canonical=edge(atomicLoad(&contactWork[contactSlots()+contact])-1u).x;}
 if(e.x!=canonical){return;}
 var first=0xffffffffu;var second=0xffffffffu;var anyFace=0xffffffffu;var faces=0u;var frontCount=0u;var backCount=0u;var allSmooth=true;var crease=false;var firstNormal=vec3f(0.0);var nonzeroField=false;
 // Two back faces of a two-sided source (both render, turned to the eye).
 var backFirst=0xffffffffu;var backSecond=0xffffffffu;var backNormal=vec3f(0.0);var backCrease=false;
 let start=halfedge(e.x);
 var members=1u;var membersAt=0u;if(e.w>0u){let group=source[9]+(e.w-1u)*2u;membersAt=source[group];members=source[group+1u];}
 var member=0u;var contactHead=0u;if(contact!=NO_CONTACT){contactHead=atomicLoad(&contactWork[contactSlots()*2u+contact]);}
 var samePart=true;var ownPart=0u;var smallPart=false;
 if(cellParts()){
  ownPart=atomicLoad(&contactWork[edgePartBase()+ordinal]);
  let at=partSlot(ownPart);if(ownPart!=0u&&at!=NO_CONTACT){smallPart=(atomicLoad(&contactWork[partFlagsBase()+at])&3u)==1u;}
 }
 loop{
  var h=e.x;
  if(contact!=NO_CONTACT){
   if(contactHead==0u){break;}
   let current=contactHead;contactHead=atomicLoad(&contactWork[contactSlots()*3u+(current-1u)]);h=edge(current-1u).x;
  }
  else{if(member>=members){break;}h=select(e.x,source[membersAt+member*2u],e.w>0u);member++;}
  if(!contactMatches(e.x,h,instance)){continue;}
  if(cellParts()){samePart=samePart&&atomicLoad(&contactWork[edgePartBase()+halfedge(h).z])==ownPart;}
  let item=edge(halfedge(h).z);
  for(var i=0u;i<item.z;i++){
   let incidence=sourceEdgeCorner(halfedge(h).z,i);if(!inDraw(incidence)||!validFace(incidence)){continue;}let t=triangle(incidence,instance);if(dot(t.face,t.face)<0.1){continue;}
   anyFace=min(anyFace,incidence);
   for(var k=0u;k<3u;k++){nonzeroField=nonzeroField||field(t.p[k],t.n[k])!=0.0;}
   faces++;allSmooth=allSmooth&&isSmooth(t);
   if(front(t)){frontCount++;if(first==0xffffffffu){first=incidence;firstNormal=t.face;}else{crease=crease||dot(firstNormal,t.face)<0.70710678;second=incidence;}}
   else{backCount++;second=incidence;
    if(backFirst==0xffffffffu){backFirst=incidence;backNormal=t.face;}else{backCrease=backCrease||dot(backNormal,t.face)<0.70710678;backSecond=incidence;}}
  }
 }
 if(faces>2u){atomicAdd(&status.nonmanifold,1u);}

 // drawn by its path construction (authoredPathAt), not as a fine edge.
 if(faces<=2u&&authoredOwnsEdge(first,second,backFirst,backSecond)){return;}
 let edgeA=position(start.x,instance);let edgeB=position(start.y,instance);
 // A normal-zero curve may run exactly through a source vertex or along a
 // source edge. Give that event its VERTEX ports and emit the edge once;
 // labeling every incidence as a different edge crossing opens the round.
 if(anyFace!=0xffffffffu&&allSmooth&&nonzeroField&&field(edgeA,normal(start.x,instance))==0.0&&field(edgeB,normal(start.y,instance))==0.0){
  let t=triangle(anyFace,instance);emit(edgeA,edgeB,t,1u,e.x|0x80000000u,vec4u(boundaryPort(start.x,instance),boundaryPort(start.y,instance),anyFace/3u,(anyFace/3u)|bends(anyFace/3u,instance)),vec4f(0),smallPart&&samePart);
  // This smooth-contour candidate does not consume the mesh edge. It may
  // also be a front/back boundary needed to close the source silhouette.
  // Rounded construction suppresses the smooth alias, not that boundary.
 }
 // A TWO-SIDED SOURCE'S FOLD SEEN FROM ITS BACK (line repair 2026-11-13).
 // The physical raster draws every face turned toward the eye
 // (physical-frame.ts cullMode none, normal flipped toward the eye), so two
 // faces wound away from the eye that turn 45 degrees or more against each
 // other are a crease the drawing shows (the near rock's right Y at
 // spot-frames, y8-instance-322 vertex 56: all six faces back-facing). The
 // depth visibility pass removes such a crease where a nearer face hides it
 // (the far side of a closed mesh); a culled source keeps the old rule.
 if(frontCount==0u&&params.direction.w==1.0&&faces==2u&&backCount==2u&&backCrease){
  if(smallPart&&samePart){return;}
  let h=halfedge(e.x);let t=triangle(backFirst,instance);
  if(levelDrops(e.x,position(h.x,instance),position(h.y,instance),instance,false)){return;}
  emit(position(h.x,instance),position(h.y,instance),t,2u,e.x,vec4u(boundaryPort(h.x,instance),boundaryPort(h.y,instance),backFirst/3u,backSecond/3u),vec4f(0),false);
  return;
 }
 if(frontCount==0u){
  // An open two-sided surface still has its source boundary when viewed
  // from behind. Do not turn every back edge of a closed mesh into ink.
  if(faces!=1u||params.direction.w!=1.0){return;}
  first=anyFace;second=0xffffffffu;
 }
 // Retain the actual posed mesh's occluding boundary even on a smooth
 // surface. Interpolated normal-zero curves are a SEPARATE candidate family;
 // they can terminate at normal/material seams and cannot replace the closed
 // source silhouette. Interior coplanar edges are still not emitted.
 // Coverage, not index winding, determines a silhouette. Same-direction
 // halfedges on a fully front-facing patch are still an INTERIOR edge;
 // treating them as boundaries filled source contour graphs with seams.
 // Keep genuine plane turns as creases and front/back transitions as the
 // silhouette, including nonmanifold incidence. No source port is welded.
 var kind=0u;if(faces>1u){if(backCount==0u){if(!crease){return;}kind=2u;}}
 // A whole closed part below the far threshold loses INTERIOR creases, not
 // its connected silhouette. Contact with another cell preserves the ledge:
 // a connected terrain mass is not classified by one tiny constituent face.
 if(kind==2u&&smallPart&&samePart){return;}
 // A chained principal crease at L1 is drawn by its chain (principalChainAt).
 if(kind==2u&&principal&&sourceEdgeLevel(e.x)==${EDGE_LEVEL_CHAINED}u){return;}
 let h=halfedge(e.x);let t=triangle(first,instance);
 if((kind==2u||kind==0u)&&levelDrops(e.x,position(h.x,instance),position(h.y,instance),instance,kind==0u)){return;}
 // A voxel chunk (posed independent quads) whose cells are too small to read
 // is drawn by its CLUSTER edges (voxelClusterEmit), each once.
 if(posedContacts()&&voxelClusterEmit(position(h.x,instance),position(h.y,instance),t,kind,e.x,vec2u(first/3u,select(0xffffffffu,second/3u,second!=0xffffffffu)),smallPart&&samePart)){return;}
 emit(position(h.x,instance),position(h.y,instance),t,kind,e.x,vec4u(boundaryPort(h.x,instance),boundaryPort(h.y,instance),first/3u,select(0xffffffffu,second/3u,second!=0xffffffffu)),vec4f(0),smallPart&&samePart);
}
fn boundaryAt(id:u32){
 if(levelDispatchEnabled()){
  if(id>=levelDispatchTotal()){return;}
  let record=levelDispatchFor(id);if(record.x==0xffffffffu){return;}
  let ordinal=id-record.z;let instance=record.x;if(!selected(instance)){return;}
  if(record.y==LEVEL_BANK_OUTLINE){authoredPathsAt(ordinal,instance);if(ordinal<sourceHullEdgeCount()){hullOutlineAt(ordinal,instance);}return;}
  if(record.y==LEVEL_BANK_PLANAR){authoredPathsAt(ordinal,instance);let spans=sourcePlanarSpanCount();if(ordinal<spans){planarSpanAt(ordinal,instance);}else if(ordinal<spans+sourcePanelJoinCount()){planarJoinAt(ordinal-spans,instance);}return;}
  boundaryAtFine(ordinal,instance,record.w,false);return;
 }
 if(residualReady()){
  let residualEdges=residualEdgeCount();if(residualEdges==0u){return;}
  let total=residualEdges*selectedCount();if(id>=total){return;}
  let selectedOrdinal=id/residualEdges;let residualOrdinal=id%residualEdges;boundaryAtFine(residualEdgeAt(residualOrdinal),selectedInstance(selectedOrdinal),selectedOrdinal,true);return;
 }
 let slots=sourceBoundarySlots();if(id>=slots*selectedCount()||native[0]==0u){return;}
 boundaryAtFine(id%slots,selectedInstance(id/slots),id/slots,false);
}
// THE L2 CONNECTED OUTLINE CONSTRUCTION (compileOutlineHull). Hull edge k is
// an outline when its two compiled hull faces, posed from the SAME vertex
// positions as every other relation, turn opposite ways to the eye: the
// emitted edges form the closed outline of the posed form. Ports are the
// source vertex ports (boundaryPort), the anchors L0 and L1 spell at those
// corners. Drawn once, by the draw that holds the edge's source face.
// Address 0xa0000000|k: never an original halfedge (edge classes and face
// words read none), stable for the edge across frames.
fn hullOutlineAt(k:u32,instance:u32){
 if(k>=sourceHullEdgeCount()){return;}
 let w=sourceHullEdge(k);let face=sourceHullFace(k);

 if(!coarseFaceOwnerAdmitted(face,instance)){return;}
 if(sourcePathCoveredBy(face)!=0xffffffffu){return;}
 if(!inDraw(face*3u)||!validFace(face*3u)){return;}
 let a=position(w.x,instance);let b=position(w.y,instance);
 let n0=cross(b-a,position(w.z,instance)-a);let n1=cross(a-b,position(w.w,instance)-b);
 var view=params.direction.xyz;if(params.eye.w==1.0){view=(a+b)*.5-params.eye.xyz;}
 let s0=dot(n0,view);let s1=dot(n1,view);
 if(!((s0<0.0&&s1>=0.0)||(s0>=0.0&&s1<0.0))){return;}
 let t=triangle(face*3u,instance);
 emitLevel(a,b,t,0u,0xa0000000u|k,vec4u(boundaryPort(w.x,instance),boundaryPort(w.y,instance),face,0xffffffffu),vec4f(0),false,true);
}
@compute @workgroup_size(64) fn contours(@builtin(global_invocation_id) gid:vec3u){contourAt(gid.x+gid.y*params.owners[3].w*64u);}
fn contourAt(id:u32){
 var instance=0u;var face=0u;
 if(levelDispatchEnabled()){
  if(id>=levelContourTotal()){return;}
  let mapped=levelContourFor(id);if(mapped.x==0xffffffffu){return;}instance=mapped.x;face=mapped.y;
 }else if(residualReady()){
  let faces=residualFaceCount();if(faces==0u||id>=faces*selectedCount()){return;}
  instance=selectedInstance(id/faces);face=residualFaceAt(id%faces);
 }else{
  let faces=sourceFaceCount();if(faces==0u||id>=faces*selectedCount()){return;}instance=selectedInstance(id/faces);face=id%faces;
 }
 if(!selected(instance)||planarProgramLevel(instance)){return;}
 if(ownerUnseen(halfedge(face*3u).x,instance)||ownerMerged(edgeOwner(halfedge(face*3u).x,instance))||outlineLevel(edgeOwner(halfedge(face*3u).x,instance))){return;}if(!inDraw(face*3u)||!validFace(face*3u)){return;}
 let t=triangle(face*3u,instance);var points:array<vec3f,3>;var ports:array<u32,3>;var parameters:array<f32,3>;var count=0u;var zeros=0u;
 var values:array<f32,3>;
 for(var k=0u;k<3u;k++){
  values[k]=field(t.p[k],t.n[k]);if(values[k]==0.0){points[count]=t.p[k];ports[count]=vertexPort(halfedge(face*3u+k).x,instance);parameters[count]=0.0;count++;zeros++;}
 }
 if(zeros>=2u){return;}
 for(var k=0u;k<3u;k++){
  let next=(k+1u)%3u;let a=values[k];let b=values[next];
  if(a!=0.0&&b!=0.0&&(a<0.0)!=(b<0.0)){
   points[count]=mix(t.p[k],t.p[next],a/(a-b));let canonical=canonicalEdge(halfedge(face*3u+k).z,instance);ports[count]=canonical|0x80000000u;
   let edge=halfedge(canonical);let origin=position(edge.x,instance);let delta=position(edge.y,instance)-origin;
   parameters[count]=dot(points[count]-origin,delta)/max(dot(delta,delta),0.000000000001);count++;
  }
 }
 if(count==2u){emit(points[0],points[1],t,1u,face,vec4u(ports[0],ports[1],face,face|bends(face,instance)),vec4f(parameters[0],parameters[1],0,0),false);}
}

// instance of a source that draws this frame's field, owning no texel of
// that field, is not seen at this camera: its boundaries, creases and
// contours are not produced (they could only reach the page through the
// one-texel depth halo). Selected here, before any relation exists, from
// the same frame's form table. Producer rows (one draw, many owners) and
// sources the field does not attribute (flag bit 4 unset) keep everything.

// vertices; a GPU producer row (voxel, batched static rows: entity word
// 0xffffffff) owns the vertices its row wrote. Gates read the owner of the
// edge or face they produce, never the draw's first vertex.
fn edgeOwner(v:u32,instance:u32)->vec2u{
 let at=slot(v,instance);let entity=positions[at].w;let producer=entity==0xffffffffu;
 return vec2u(params.domain|select(0u,0x80000000u,producer),select(entity,attributes[at].w,producer));
}
fn ownerUnseen(v:u32,instance:u32)->bool{
 if((params.owners[1].z&17u)!=17u||native[0]==0u){return false;}
 return !formExtent(edgeOwner(v,instance)).found;
}

// owners: voxel detail props, batched static rows) whose own visible form fits
// one native cell with the hold (farAggregateMember, gpu-source-far-forms.ts)
// produces no boundary, crease or occluding contour (boundaryAt, contourAt).
// The interior stages (suggestive, ridge, view-face lines) are NOT gated here:

// failed validation when emit() read it). It is a
// member of the producer far aggregate, which the far aggregate emitter draws
// once from the same field (its units on the merged far skyline where the
// members are the column's outline). The emitter reads the SAME test on the
// same form table and lists only the domains gated here (flag bits 1 and 16,
// a perspective view), so a row is drawn by exactly one of the two. Entity
// instances are never members here: their parent aggregates are selected
// before any relation work (source-lod parents).
fn ownerMerged(owner:vec2u)->bool{
 if(formParentSelected(owner)){return true;}
 if((params.owners[1].z&17u)!=17u||native[0]==0u||params.eye.w==0.0||(owner.x&0x80000000u)==0u){return false;}
 return farAggregateMember(owner,formExtent(owner),vec2f(f32(params.owners[1].w),f32(params.owners[2].x)),bitcast<f32>(params.owners[1].y));
}
fn instanceUnseen(instance:u32)->bool{
 if((params.owners[1].z&17u)!=17u||native[0]==0u){return false;}
 let first=halfedge(native[2]-params.indexBase).x;let entity=positions[slot(first,instance)].w;
 if(entity==0xffffffffu){return false;}
 return !formExtent(vec2u(params.domain,entity)).found;
}

// 2026-11-17). A facet crease (its compiled level code, sourceEdgeLevel, is not
// 0) of a VISIBLE instance whose form is 中景 or 遠景 is produced only when its
// compiled feature chain could reach the tier's least run (TIER_SELECTION
// column 3) with the hold: chain rows = this edge's POSED length (the
// instance's world scale) x code / EDGE_LEVEL_UNIT, over the world units one
// row covers at the form's nearest visible depth (tierRowUnits, the frame's own
// matrix) - an upper bound of its projected length. The instance's tier is the
// form table's, as needsViewCurvature reads it: rows with LOD_TIER_HOLD; a cut,
// spanning, enclosed, parallel, terrain or unknown form is near and keeps all.
// Principal creases have code 0. A silhouette (kind 0) is limited only when its
// edge is class 1 DETAIL (code bit 15: a window or panel recess side); facet
// silhouettes and smooth contours are never limited, so every form keeps its
// connected outline (D9). No search, no probe.
fn levelDrops(address:u32,a:vec3f,b:vec3f,instance:u32,silhouette:bool)->bool{
 let code=sourceEdgeLevel(address);if(code==0u||(silhouette&&(code&${EDGE_LEVEL_DETAIL}u)==0u)){return false;}
 let flags=params.owners[1].z;
 if((flags&1u)==0u||(flags&2u)!=0u||params.eye.w==0.0||native[0]==0u){return false;}
 if(!formOpen()){return false;}
 let owner=edgeOwner(halfedge(address).x,instance);

 // measured by THAT extent (the form-table writer's, the same one its held
 // level came from), never by the pixels that survive the foreground. No
 // bound: the visible extent, as before.
 var form=formExtent(owner);let sourceExtent=formSourceExtent(owner);if(sourceExtent.found){form=sourceExtent;}
 let extent=vec2f(f32(params.owners[1].w),f32(params.owners[2].x));
 if(!form.found||formCut(form.box,extent)||formSpansDepth(form.near,form.far,extent,detailWorldFromClip)){return false;}
 let row=bitcast<f32>(params.owners[1].y);
 // The L1 subset applies from the held principal level (formLevel).
 if(formLevel(owner)==0u){return false;}
 let unit=tierRowUnits(vec2f(form.box.z+form.box.w,form.box.x+form.box.y)*.5,form.near,extent,row,detailWorldFromClip);
 if(!(unit>0.0)){return false;}
 return length(b-a)*f32(code&${EDGE_LEVEL_DETAIL-1}u)/${EDGE_LEVEL_UNIT}.0<unit*${FACET_CREASE_LEAST_ROWS*LOD_TIER_HOLD};
}
fn interiorInstances()->u32{
 let available=min(params.owners[0].y,native[1]-min(native[1],params.owners[0].x));
 return select(available,selectedCount(),params.owners[3].x!=0u);
}
fn interiorCandidates()->u32{if(params.owners[3].x!=0u){return selectedCount();}return interiorInstances();}
fn interiorCandidate(ordinal:u32)->u32{if(params.owners[3].x!=0u){return selectedInstance(ordinal);}return params.owners[0].x+ordinal;}
fn detailBase()->u32{return (sourceFaceCount()*3u+sourceVertexCount()*3u)*interiorInstances();}
fn viewInteriorInstances()->u32{if((params.owners[1].z&8u)!=0u){return bitcast<u32>(differentials[detailBase()].x);}return interiorCandidates();}
fn viewInteriorInstance(ordinal:u32)->u32{if((params.owners[1].z&8u)!=0u){return bitcast<u32>(differentials[detailBase()+1u+ordinal].x);}return interiorCandidate(ordinal);}
fn radialBase()->u32{return sourceFaceCount()*interiorInstances()*3u;}
// Early structural LOD, using the final drawing's SAME form table, extent,
// compact-form rule and hysteresis boundary. Unknown/cut/spanning forms,
// terrain and enclosed/parallel views keep the complete interior producer.
fn needsViewCurvature(instance:u32)->bool{
 if(planarProgramLevel(instance)){return false;}
 if(selected(instance)&&native[0]!=0u){
  let first=halfedge(native[2]-params.indexBase).x;let at=slot(first,instance);
  let entity=positions[at].w;let producer=entity==0xffffffffu;
  if(formParentSelected(vec2u(params.domain|select(0u,0x80000000u,producer),select(entity,attributes[at].w,producer)))){return false;}
 }
 let flags=params.owners[1].z;
 if((flags&1u)==0u||(flags&2u)!=0u||params.eye.w==0.0){return true;}
 if(!formOpen()){return true;}
 if(!selected(instance)||native[0]==0u){return false;}
 let first=halfedge(native[2]-params.indexBase).x;let at=slot(first,instance);
 let entity=positions[at].w;let producer=entity==0xffffffffu;
 let owner=vec2u(params.domain|select(0u,0x80000000u,producer),select(entity,attributes[at].w,producer));
 return !farStripForm(owner);
}
// THE FORM LEVEL RULE shared by the interior gate (needsViewCurvature) and the
// L2 outline (outlineLevel): the owner's same-pose form-table rows are under
// the far-strip tier with the hold. Unknown, cut, depth-spanning, terrain,
// enclosed or parallel views are never L2.
fn levelRows(owner:vec2u)->f32{
 let flags=params.owners[1].z;
 if((flags&1u)==0u||(flags&2u)!=0u||params.eye.w==0.0||!formOpen()){return -1.0;}
 let form=formExtent(owner);let extent=vec2f(f32(params.owners[1].w),f32(params.owners[2].x));
 if(!form.found||formCut(form.box,extent)||formSpansDepth(form.near,form.far,extent,detailWorldFromClip)){return -1.0;}
 return (form.bottom-form.top)/bitcast<f32>(params.owners[1].y);
}


// form and camera with separate entry/leave bands, read from the SAME form
// texture (no new binding). The header texel's w word says the writer
// published held levels this frame; without it every reader keeps the

fn formLodActive()->bool{return textureLoad(formDepths,vec2i(0,${FORM_DEPTH_HEADER_ROW}),0).w!=0u;}
fn formLevel(owner:vec2u)->u32{

 // complete posed source bound (form-depths resolveSource/resolveLod) takes
 // that held level as is; its surviving pixel box (levelRows' visible extent,
 // cut and depth span) never re-sizes the representation. A cut or spanning
 // source holds level 0 there. Every other owner keeps the incumbent read.
 let flags=params.owners[1].z;
 if(formLodActive()&&(flags&1u)!=0u&&(flags&2u)==0u&&params.eye.w!=0.0&&formOpen()&&(formSourceFlags(owner)&1u)!=0u){return formHeldLod(owner,true);}
 let rows=levelRows(owner);if(rows<0.0){return 0u;}
 if(formLodActive()){return formHeldLod(owner,true);}
 return select(select(1u,2u,rows<${FAR_STRIP_BELOW_ROWS*LOD_TIER_HOLD}),0u,!(rows<${NEAR_FROM_ROWS*LOD_TIER_HOLD}));
}
fn farStripForm(owner:vec2u)->bool{return formLevel(owner)==2u;}

// 中景 owner (under NEAR_FROM_ROWS with the hold, the levelDrops boundary)
// draws each compiled straight principal chain as ONE crease edge between its
// end anchors, instead of the chain's member creases (EDGE_LEVEL_CHAINED).
fn principalLevel(owner:vec2u)->bool{
 // Producer rows (including PhysicalFrame/Y8) use the same compiled CSR
 // ladder. Capability comes from that packet, not the owner's identity tag.
 if(native[0]==0u||sourceChainCount()==0u){return false;}
 return formLevel(owner)>=1u;
}
fn principalChainAt(k:u32,instance:u32){
 if(k>=sourceChainCount()){return;}
 let c=sourceChain(k);

 if(sourcePathCoveredBy(c[2])!=0xffffffffu&&sourcePathCoveredBy(c[3])!=0xffffffffu){return;}
 if(!inDraw(c[2]*3u)||!validFace(c[2]*3u)||!inDraw(c[3]*3u)||!validFace(c[3]*3u)){return;}
 let t0=triangle(c[2]*3u,instance);let t1=triangle(c[3]*3u,instance);
 // The same crease rule as boundaryAt: both planes turned to the eye and
 // turning 45 degrees or more; a chain on the silhouette is drawn by its
 // members' silhouettes (kind 0 is never chained).
 if(dot(t0.face,t0.face)<0.1||dot(t1.face,t1.face)<0.1||!front(t0)||!front(t1)||dot(t0.face,t1.face)>=0.70710678){return;}
 emit(position(c[0],instance),position(c[1],instance),t0,2u,c[4],vec4u(boundaryPort(c[0],instance),boundaryPort(c[1],instance),c[2],c[3]),vec4f(0),false);
}

// far-strip level (farStripForm) but still a unit or more (smaller ones are far
// aggregate members, selected before this pass) is drawn as ONE connected
// outline (D9): only its silhouette edges are produced, tagged (chart flag
// bit 2), and the run stage reserves them as one whole-form construction
// whatever its plane count (gpu-construction-runs outlineRelation). L1 is the
// principal subset (levelDrops), L0 everything. Every level keeps the same
// source vertex ports, so the anchors are shared and a level change adds or
// removes whole edges between them; it never respells a corner.
fn outlineLevel(owner:vec2u)->bool{return native[0]!=0u&&sourceHullEdgeCount()!=0u&&farStripForm(owner);}
var<workgroup> detailLimit:u32;
var<workgroup> detailCount:u32;
var<workgroup> detailFlags:array<u32,64>;
@compute @workgroup_size(64) fn prepareInterior(@builtin(local_invocation_index) local:u32){prepareInteriorIn(local);}
fn prepareInteriorIn(local:u32){
 var count=select(interiorInstances(),0u,native[5]!=0u||sourceVertexCount()>params.vertexCapacity||(atomicLoad(&status.overflow)&9u)!=0u);
 if(local==0u){detailLimit=min(interiorCandidates(),count);detailCount=0u;}
 workgroupBarrier();
 // One near member must not promote every far member of a shared source.
 // Compact in original candidate order, preserving source/tensor addresses.
 // The workgroup-uniform bound keeps every lane at every block barrier.
 let candidates=workgroupUniformLoad(&detailLimit);
 for(var first=0u;first<candidates;first+=64u){
  let ordinal=first+local;var needed=false;
  if(ordinal<candidates){needed=needsViewCurvature(interiorCandidate(ordinal));}
  detailFlags[local]=select(0u,1u,needed);workgroupBarrier();
  if(local==0u){
   for(var k=0u;k<min(64u,candidates-first);k++){
    if(detailFlags[k]!=0u){differentials[detailBase()+1u+detailCount]=bitcast<vec4f>(vec4u(interiorCandidate(first+k),0u,0u,0u));detailCount++;}
   }
  }
  workgroupBarrier();
 }
 if(local!=0u){return;}
 let viewCount=detailCount;differentials[detailBase()]=bitcast<vec4f>(vec4u(viewCount,0u,0u,0u));
 // Retained shape tensors are view-independent: even a far first view must
 // initialize them before its revision can be reused by an approaching view.
 if((params.owners[1].z&4u)==0u&&viewCount==0u){count=0u;}
 let limit=params.owners[0].z;let faces=(sourceFaceCount()*count+63u)/64u;let vertices=(sourceVertexCount()*count+63u)/64u;
 dispatch[6]=min(faces,limit);dispatch[7]=max(1u,(faces+limit-1u)/limit);dispatch[8]=1u;
 dispatch[9]=min(vertices,limit);dispatch[10]=max(1u,(vertices+limit-1u)/limit);dispatch[11]=1u;
 let viewFaces=(sourceFaceCount()*viewCount+63u)/64u;let viewVertices=(sourceVertexCount()*viewCount+63u)/64u;
 dispatch[12]=min(viewFaces,limit);dispatch[13]=max(1u,(viewFaces+limit-1u)/limit);dispatch[14]=1u;
 dispatch[15]=min(viewVertices,limit);dispatch[16]=max(1u,(viewVertices+limit-1u)/limit);dispatch[17]=1u;
}
// Port of tools/hatch_studio_export.py:49-82: symmetric second fundamental
// form from the three edge/normal differences, then area-weighted at each
// source vertex. This evaluates the tensor along the current radial direction
// directly, without eigenvector selection. Normals and positions are the REAL
// material vertex result, so deformation/nonuniform scaling are not guessed
// from a rest-mesh curvature or an asset name.
// Method: https://gfx.cs.princeton.edu/pubs/Rusinkiewicz_2004_ECA/
@compute @workgroup_size(64) fn faceDifferentials(@builtin(global_invocation_id) gid:vec3u){faceDifferentialAt(gid.x+gid.y*params.owners[0].z*64u);}
fn faceDifferentialAt(id:u32){
 if(id>=sourceFaceCount()*interiorInstances()){return;}
 let ordinal=id/sourceFaceCount();let instance=interiorCandidate(ordinal);let face=id%sourceFaceCount();let at=id*3u;
 differentials[at]=vec4f(0);differentials[at+1u]=vec4f(0);differentials[at+2u]=vec4f(0);
 if((params.selection.w==0u&&!selected(instance))||!inDraw(face*3u)||!validFace(face*3u)){return;}
 let tri=triangle(face*3u,instance);
 let e=array<vec3f,3>(tri.p[2]-tri.p[1],tri.p[0]-tri.p[2],tri.p[1]-tri.p[0]);
 let dn=array<vec3f,3>(tri.n[2]-tri.n[1],tri.n[0]-tri.n[2],tri.n[1]-tri.n[0]);
 let area=length(cross(e[2],-e[1]))*.5;let scale=max(length(e[0]),max(length(e[1]),length(e[2])));
 if(area<=0.0||scale<=0.0){return;}
 let tangent=e[0]/max(length(e[0]),1e-30);let binormal=cross(tri.face,tangent);
 var a=0.0;var b=0.0;var c=0.0;var rhs=vec3f(0);
 for(var k=0u;k<3u;k++){
  let u=dot(e[k],tangent)/scale;let v=dot(e[k],binormal)/scale;
  let du=dot(dn[k],tangent);let dv=dot(dn[k],binormal);
  a+=u*u;b+=u*v;c+=v*v;rhs+=vec3f(u*du,v*du+u*dv,v*dv);
 }
 // Cholesky solve of [a b 0; b a+c b; 0 b c]. Geometric scale
 // normalization keeps the arithmetic independent of world-unit magnitudes.
 if(a<=1e-12||a*c-b*b<=1e-10*a*c){return;}
 let l0=sqrt(a);let l10=b/l0;let l1=sqrt(max(a+c-l10*l10,0.0));
 if(l1<=1e-12){return;}let l21=b/l1;let last=c-l21*l21;if(last<=1e-12){return;}
 let l2=sqrt(last);let y0=rhs.x/l0;let y1=(rhs.y-l10*y0)/l1;let y2=(rhs.z-l21*y1)/l2;
 let n=y2/l2;let m=(y1-l21*n)/l1;let l=(y0-l10*m)/l0;
 differentials[at]=vec4f(tangent,area);differentials[at+1u]=vec4f(binormal,0);
 differentials[at+2u]=vec4f(vec3f(l,m,n)/scale,1);
}
@compute @workgroup_size(64) fn radialCurvature(@builtin(global_invocation_id) gid:vec3u){radialCurvatureAt(gid.x+gid.y*params.owners[0].z*64u);}
fn radialCurvatureAt(id:u32){
 if(id>=sourceVertexCount()*viewInteriorInstances()){return;}
 let ordinal=id/sourceVertexCount();let instance=viewInteriorInstance(ordinal);let local=ordinal;let vertex=id%sourceVertexCount();let out=radialBase()+local*sourceVertexCount()+vertex;
 differentials[out]=vec4f(0);
 if(!selected(instance)||!vertexInDraw(vertex)){return;}
 let P=position(vertex,instance);let N=normal(vertex,instance);
 let toward=select(-params.direction.xyz,params.eye.xyz-P,params.eye.w!=0.0);
 let V=toward/max(length(toward),1e-30);let ndv=dot(N,V);let w=V-N*ndv;let lengthW=length(w);
 if(lengthW<=1e-6){return;}let radial=w/lengthW;
 let incidence=sourceVertexRecord(vertex);var sum=0.0;var area=0.0;
 // ORIGINAL vertex incidence only. Coincident coordinates alone cannot
 // mix the curvature of another part, sheet or contacting subject.
 for(var k=0u;k<incidence.w;k++){
  let h=sourceVertexCorner(vertex,k);if(!inDraw(h)){continue;}
  let face=(local*sourceFaceCount()+h/3u)*3u;let t=differentials[face];let b=differentials[face+1u];let form=differentials[face+2u];
  if(form.w==0.0){continue;}
  let u=dot(radial,t.xyz);let v=dot(radial,b.xyz);
  sum+=(form.x*u*u+2.0*form.y*u*v+form.z*v*v)*t.w;area+=t.w;
 }
 // Preserve the source-normal sign across the silhouette. Flipping each
 // vertex independently would fabricate a zero set on a convex surface.
 // Visible-side orientation belongs to the derivative test per segment.
 if(area<=0.0){return;}let kr=sum/area;
 if(kr!=kr||abs(kr)>=3.402823e38){return;}
 differentials[out]=vec4f(kr,abs(ndv),0,1);
}
// Source-surface kr=0, restricted to positive D_w kr on its visible side.
// Unlike the image-space n·v-valley witness, this produces actual interior
// source relations which join across shared mesh-edge crossing parameters.
// Definition: https://gfx.cs.princeton.edu/pubs/DeCarlo_2003_SCF/
@compute @workgroup_size(64) fn suggestiveContours(@builtin(global_invocation_id) gid:vec3u){suggestiveAt(gid.x+gid.y*params.owners[0].z*64u);}
fn suggestiveAt(id:u32){
 if(id>=sourceFaceCount()*viewInteriorInstances()){return;}
 let ordinal=id/sourceFaceCount();let instance=viewInteriorInstance(ordinal);let local=ordinal;let face=id%sourceFaceCount();
 if(!selected(instance)||!inDraw(face*3u)||!validFace(face*3u)){return;}
 let tri=triangle(face*3u,instance);if(!front(tri)&&params.direction.w!=1.0){return;}
 var values:array<f32,3>;
 for(var k=0u;k<3u;k++){let vertex=halfedge(face*3u+k).x;let radial=differentials[radialBase()+local*sourceVertexCount()+vertex];if(radial.w==0.0){return;}values[k]=radial.x;}
 var points:array<vec3f,3>;var ports:array<u32,3>;var parameters:array<f32,3>;var count=0u;
 // Zero belongs to the nonnegative side. This half-open marching rule
 // retains a zero set lying exactly on an edge from its negative side,
 // instead of dropping every triangle with two zero vertices. A whole
 // zero triangle emits nothing; a zero vertex tangent is not a segment.
 for(var k=0u;k<3u;k++){let next=(k+1u)%3u;let a=values[k];let b=values[next];if((a<0.0)!=(b<0.0)){
  points[count]=mix(tri.p[k],tri.p[next],a/(a-b));
  if(a==0.0||b==0.0){let vertex=select(next,k,a==0.0);ports[count]=vertexPort(halfedge(face*3u+vertex).x,instance);parameters[count]=0.0;}
  else{let canonical=canonicalEdge(halfedge(face*3u+k).z,instance);ports[count]=canonical|0x80000000u;
   let edge=halfedge(canonical);let origin=position(edge.x,instance);let delta=position(edge.y,instance)-origin;
   // ONE CROSSING PER SHARED EDGE (line repair, 2026-11-12): read from the
   // canonical edge's own corner values in its own direction, so both faces
   // of the edge compute the same point and join parameter bit for bit.
   // Per face, from world positions, the two halves of one crossing differed
   // by more than the join key's quantum and a fold ended at every face.
   let va=differentials[radialBase()+local*sourceVertexCount()+edge.x];let vb=differentials[radialBase()+local*sourceVertexCount()+edge.y];
   if(va.w!=0.0&&vb.w!=0.0&&va.x!=0.0&&vb.x!=0.0&&(va.x<0.0)!=(vb.x<0.0)){let s=va.x/(va.x-vb.x);points[count]=mix(origin,position(edge.y,instance),s);parameters[count]=s;}
   else{parameters[count]=dot(points[count]-origin,delta)/max(dot(delta,delta),1e-30);}
  }count++;
 }}
 if(count!=2u||all(points[0]==points[1])){return;}
 let e1=tri.p[1]-tri.p[0];let e2=tri.p[2]-tri.p[0];let crossN=cross(e1,e2);let norm=dot(crossN,crossN);if(norm<=1e-30){return;}
 let gradient=((values[1]-values[0])*cross(e2,crossN)+(values[2]-values[0])*cross(crossN,e1))/norm;
 let P=(points[0]+points[1])*.5;let normalSum=tri.n[0]+tri.n[1]+tri.n[2];let normalLength=length(normalSum);if(normalLength<=1e-6){return;}let N=normalSum/normalLength;
 let toward=select(-params.direction.xyz,params.eye.xyz-P,params.eye.w!=0.0);let V=toward/max(length(toward),1e-30);let ndv=dot(N,V);let w=V-N*ndv;
 if(abs(ndv)<=1e-6||dot(gradient,w)*select(-1.0,1.0,ndv>=0.0)<=0.0){return;}
 emit(points[0],points[1],tri,3u,face,vec4u(ports[0],ports[1],face,face|bends(face,instance)),vec4f(parameters[0],parameters[1],0,0),false);
}

// lines joined into whole curves on the mesh before selection; the Studio
// export's per-vertex curv/pdir1 is the input). The posed tensor above gives,
// per source vertex, the principal curvatures k1 (largest magnitude) and k2
// and the direction e1 of k1: the live-mesh equivalent of tools/
// hatch_studio_export.py's curv and pdir1, computed on the REAL posed vertices.
// A ridge (k1 > 0) or valley (k1 < 0) line is where |k1| is extreme along its
// own direction: the zero set of e = grad(k1).e1 crossed with e falling along
// e1 (Ohtake, Belyaev, Seidel 2004, "Ridge-valley lines on meshes via implicit
// surface fitting"). Its crossings sit on canonical source edges, so the run
// pass links consecutive faces through the shared port into one curve: a brim,
// a jaw, a fold is one run, not a fragment per face. View independent except
// the visible-side test, so a retained posed tensor is not recomputed.
fn principalBase()->u32{return radialBase()+sourceVertexCount()*interiorInstances();}
fn ridgeBase()->u32{return principalBase()+sourceVertexCount()*interiorInstances();}
fn outer(a:vec3f,b:vec3f)->mat3x3f{return mat3x3f(a*b.x,a*b.y,a*b.z);}
// A line is drawn where the surface turns by RIDGE_TURN radians or more across
// one incident edge of the mesh's own sampling (|k1| times the mean incident
// edge length) and k1 dominates k2 by RIDGE_ANISOTROPY: an umbilic (a sphere,
// a flat) has no ridge. Fixed rule, no score.
const RIDGE_TURN=.08;
const RIDGE_ANISOTROPY=1.5;
@compute @workgroup_size(64) fn principalCurvature(@builtin(global_invocation_id) gid:vec3u){principalCurvatureAt(gid.x+gid.y*params.owners[0].z*64u);}
fn principalCurvatureAt(id:u32){
 if(id>=sourceVertexCount()*interiorInstances()){return;}
 let local=id/sourceVertexCount();let instance=interiorCandidate(local);let vertex=id%sourceVertexCount();
 differentials[principalBase()+id]=vec4f(0);differentials[ridgeBase()+id]=vec4f(0);
 // Like faceDifferentials, retained shape data is initialized for every
 // instance of this revision, not only those this camera happens to see.
 if((params.selection.w==0u&&!selected(instance))||!vertexInDraw(vertex)){return;}
 let N=normal(vertex,instance);let P=position(vertex,instance);
 let incidence=sourceVertexRecord(vertex);var II=mat3x3f(vec3f(0),vec3f(0),vec3f(0));var area=0.0;var reach=0.0;var edges=0.0;
 for(var k=0u;k<incidence.w;k++){
  let h=sourceVertexCorner(vertex,k);if(!inDraw(h)){continue;}
  let face=(local*sourceFaceCount()+h/3u)*3u;let t=differentials[face];let b=differentials[face+1u];let form=differentials[face+2u];
  if(form.w==0.0){continue;}
  II+=(outer(t.xyz,t.xyz)*form.x+(outer(t.xyz,b.xyz)+outer(b.xyz,t.xyz))*form.y+outer(b.xyz,b.xyz)*form.z)*t.w;area+=t.w;
  reach+=length(position(halfedge(h).y,instance)-P);edges+=1.0;
 }
 if(area<=0.0||edges==0.0){return;}
 II=II*(1.0/area);
 var u=cross(N,vec3f(0,0,1));if(dot(u,u)<1e-6){u=cross(N,vec3f(0,1,0));}u=normalize(u);let v=cross(N,u);
 let a=dot(u,II*u);let c=dot(v,II*v);let s=dot(u,II*v);
 let mean=(a+c)*.5;let spread=sqrt(max((a-c)*(a-c)*.25+s*s,0.0));
 let theta=.5*atan2(2.0*s,a-c);let along=cos(theta)*u+sin(theta)*v;
 var k1=mean+spread;var k2=mean-spread;var e1=along;
 if(abs(k2)>abs(k1)){k1=mean-spread;k2=mean+spread;e1=cross(N,along);}
 if(k1!=k1||abs(k1)>=3.4e38||k1==0.0){return;}
 differentials[principalBase()+id]=vec4f(e1,k1);differentials[ridgeBase()+id]=vec4f(k2,0.0,reach/edges,0.0);
}
@compute @workgroup_size(64) fn ridgeDerivative(@builtin(global_invocation_id) gid:vec3u){ridgeDerivativeAt(gid.x+gid.y*params.owners[0].z*64u);}
fn ridgeDerivativeAt(id:u32){
 if(id>=sourceVertexCount()*interiorInstances()){return;}
 let local=id/sourceVertexCount();let vertex=id%sourceVertexCount();let instance=interiorCandidate(local);
 let principal=differentials[principalBase()+id];if(principal.w==0.0){return;}
 let incidence=sourceVertexRecord(vertex);var gradient=vec3f(0);var area=0.0;
 for(var k=0u;k<incidence.w;k++){
  let h=sourceVertexCorner(vertex,k);if(!inDraw(h)){continue;}let face=h/3u;
  var p:array<vec3f,3>;var value:array<f32,3>;var whole=true;
  for(var j=0u;j<3u;j++){let w=halfedge(face*3u+j).x;p[j]=position(w,instance);let q=differentials[principalBase()+local*sourceVertexCount()+w];if(q.w==0.0){whole=false;}value[j]=q.w;}
  if(!whole){continue;}
  let e1=p[1]-p[0];let e2=p[2]-p[0];let crossN=cross(e1,e2);let norm=dot(crossN,crossN);if(norm<=1e-30){continue;}
  let weight=sqrt(norm)*.5;
  gradient+=((value[1]-value[0])*cross(e2,crossN)+(value[2]-value[0])*cross(crossN,e1))/norm*weight;area+=weight;
 }
 if(area<=0.0){return;}
 let out=ridgeBase()+id;let r=differentials[out];
 differentials[out]=vec4f(r.x,dot(gradient/area,principal.xyz),r.z,1.0);
}
@compute @workgroup_size(64) fn ridgeLines(@builtin(global_invocation_id) gid:vec3u){ridgeLineAt(gid.x+gid.y*params.owners[0].z*64u);}
fn ridgeLineAt(id:u32){
 if(id>=sourceFaceCount()*viewInteriorInstances()){return;}
 let ordinal=id/sourceFaceCount();let instance=viewInteriorInstance(ordinal);let local=ordinal;let face=id%sourceFaceCount();
 if(!selected(instance)||!inDraw(face*3u)||!validFace(face*3u)){return;}
 let tri=triangle(face*3u,instance);if(!front(tri)&&params.direction.w!=1.0){return;}
 var e:array<f32,3>;var k1:array<f32,3>;var k2:array<f32,3>;var dirs:array<vec3f,3>;var reach=0.0;
 for(var j=0u;j<3u;j++){
  let w=halfedge(face*3u+j).x;let q=differentials[principalBase()+local*sourceVertexCount()+w];let r=differentials[ridgeBase()+local*sourceVertexCount()+w];
  if(q.w==0.0||r.w==0.0){return;}
  dirs[j]=q.xyz;k1[j]=q.w;k2[j]=r.x;e[j]=r.y;reach+=r.z/3.0;
 }
 // One principal family over the face (a ridge and a valley never share
 // one), and one sense of e1: a principal direction has no sign.
 if(k1[0]*k1[1]<=0.0||k1[0]*k1[2]<=0.0){return;}
 for(var j=1u;j<3u;j++){if(dot(dirs[j],dirs[0])<0.0){e[j]=-e[j];}}
 var points:array<vec3f,3>;var ports:array<u32,3>;var parameters:array<f32,3>;var strong=true;var count=0u;
 for(var k=0u;k<3u;k++){let next=(k+1u)%3u;let a=e[k];let b=e[next];if((a<0.0)!=(b<0.0)){
  let t=a/(a-b);points[count]=mix(tri.p[k],tri.p[next],t);
  let kc=mix(k1[k],k1[next],t);let kc2=mix(k2[k],k2[next],t);
  strong=strong&&abs(kc)*reach>=RIDGE_TURN&&abs(kc)>=RIDGE_ANISOTROPY*abs(kc2);
  if(a==0.0||b==0.0){let vertex=select(next,k,a==0.0);ports[count]=vertexPort(halfedge(face*3u+vertex).x,instance);parameters[count]=0.0;}
  else{let canonical=canonicalEdge(halfedge(face*3u+k).z,instance);ports[count]=canonical|0x80000000u;
   let edge=halfedge(canonical);let origin=position(edge.x,instance);let delta=position(edge.y,instance)-origin;
   // One crossing per shared edge, as in suggestiveContours: e at the
   // canonical corners, the second turned to the first's sense of e1.
   let qa=differentials[principalBase()+local*sourceVertexCount()+edge.x];let qb=differentials[principalBase()+local*sourceVertexCount()+edge.y];
   let ra=differentials[ridgeBase()+local*sourceVertexCount()+edge.x];let rb=differentials[ridgeBase()+local*sourceVertexCount()+edge.y];
   let ea=ra.y;let eb=select(rb.y,-rb.y,dot(qa.xyz,qb.xyz)<0.0);
   if(qa.w!=0.0&&qb.w!=0.0&&ra.w!=0.0&&rb.w!=0.0&&ea!=0.0&&eb!=0.0&&(ea<0.0)!=(eb<0.0)){let s=ea/(ea-eb);points[count]=mix(origin,position(edge.y,instance),s);parameters[count]=s;}
   else{parameters[count]=dot(points[count]-origin,delta)/max(dot(delta,delta),1e-30);}
  }count++;
 }}
 if(count!=2u||!strong||all(points[0]==points[1])){return;}
 // An extremum of |k1|, not a trough of it: e falls along e1 across a ridge
 // and rises across a valley.
 let e1=tri.p[1]-tri.p[0];let e2=tri.p[2]-tri.p[0];let crossN=cross(e1,e2);let norm=dot(crossN,crossN);if(norm<=1e-30){return;}
 let gradient=((e[1]-e[0])*cross(e2,crossN)+(e[2]-e[0])*cross(crossN,e1))/norm;
 if(sign(k1[0])*dot(gradient,dirs[0])>=0.0){return;}
 // The address keeps a ridge apart from a suggestive contour of the same face.
 emit(points[0],points[1],tri,3u,face|0x40000000u,vec4u(ports[0],ports[1],face,face|bends(face,instance)),vec4f(parameters[0],parameters[1],0,0),false);
}

// relation dispatch costs ~15 us of fixed time and the stages of ~137 in-view
// sources ran as ~815 serial dispatches (14 ms of a 30 ms job). Stages of ONE
// source that read nothing another of them writes share one dispatch: each
// thread runs the same per-id body as the separate entry above, so the set of
// emitted relations is unchanged; only their append order interleaves.
// surfaceLines: boundaries over edges x instances, then contours over faces x
// instances, as one id range (workgroup g covers ids 64g..64g+63).
@compute @workgroup_size(64) fn surfaceLines(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) local:u32){
 let id=(wid.x+wid.y*params.owners[3].w)*64u+local;let edges=sourceBoundarySlots()*selectedCount();
 if(id<edges){boundaryAt(id);}else{contourAt(id-edges);}
}
// viewFaceLines: suggestive contours and ridge lines over the same view faces
// (both read radial/principal/ridge words written by EARLIER dispatches).
@compute @workgroup_size(64) fn viewFaceLines(@builtin(global_invocation_id) gid:vec3u){
 let id=gid.x+gid.y*params.owners[0].z*64u;suggestiveAt(id);ridgeLineAt(id);
}
`;

// 2026-11-17). The ~541 per-source dispatches of a Y8 view (~17 us each) become
// one dispatch per stage per group of sources that share storage pages. The
// stage bodies are the SAME functions as the per-source entries above; only
// the binding offsets of the per-source path become per-source word bases:
// each source-range array (source incidence, posed positions/attributes,
// batch metadata, contact scratch, posed tensor, dispatch words) is bound as
// its whole storage page and indexed at base + the same index. A thread finds
// its source in a uniform table by its id (sources are contiguous id ranges,
// one thread per relation candidate exactly as in the per-source dispatch).
export const SURFACE_RELATION_BATCH_ITEMS=255;
const BATCH_BASES:Record<string,string>={source:'B0.x',positions:'B0.y',attributes:'B0.z',native:'B0.w',contactWork:'B1.x',differentials:'B1.y',dispatch:'B1.z'};
function stripWgslComments(code:string):string{return code.replace(/\/\*[\s\S]*?\*\//g,'').replace(/\/\/[^\n]*/g,'');}
/** `name[index]` becomes `name[BASE+(index)]` for the source-range arrays. */
function rebaseSourceArrays(code:string):string{
 const pattern=/\b(source|positions|attributes|native|contactWork|differentials|dispatch)\[/g;let out='',at=0;
 for(;;){
  pattern.lastIndex=at;const match=pattern.exec(code);if(!match)return out+code.slice(at);
  const open=match.index+match[0].length-1;let depth=1,end=open+1;
  for(;depth;end++){if(end>=code.length)throw Error('Unbalanced WGSL index');if(code[end]==='[')depth++;else if(code[end]===']')depth--;}
  out+=code.slice(at,match.index)+match[1]+'['+BATCH_BASES[match[1]]+'+('+rebaseSourceArrays(code.slice(open+1,end-1))+')]';at=end;
 }
}
const PARAMS_BINDING='@group(0) @binding(0) var<uniform> params:Params;';
if(!SURFACE_RELATIONS_SHADER.includes(PARAMS_BINDING))throw Error('Surface relation params binding changed');
const batchEntry=(name:string,body:string)=>`@compute @workgroup_size(64) fn ${name}(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) local:u32){let id=batchLocal(wid,local);if(id==BATCH_NONE){return;}${body}}\n`;
/** Replace one WGSL function body without changing its signature or any other
 * declaration. This is used only to build the explicit physical-host shader;
 * generic and GPU-indexed sources keep the original bodies byte-for-byte. */
function rewriteWgslFunction(source:string,name:string,rewrite:(body:string)=>string):string{
 const marker=`fn ${name}(`,start=source.indexOf(marker);if(start<0)throw Error(`Missing WGSL function ${name}`);
 const open=source.indexOf('{',start);if(open<0)throw Error(`Missing WGSL body ${name}`);
 let depth=1,lineComment=false,blockComment=false,close=-1;
 for(let i=open+1;i<source.length;i++){
  const c=source[i]!,next=source[i+1];
  if(lineComment){if(c==='\n')lineComment=false;continue;}
  if(blockComment){if(c==='*'&&next==='/'){blockComment=false;i++;}continue;}
  if(c==='/'&&next==='/'){lineComment=true;i++;continue;}
  if(c==='/'&&next==='*'){blockComment=true;i++;continue;}
  if(c==='{')depth++;
  else if(c==='}'&&--depth===0){close=i;break;}
 }
 if(close<0)throw Error(`Unbalanced WGSL function ${name}`);
 return source.slice(0,open+1)+rewrite(source.slice(open+1,close))+source.slice(close);
}
function replaceWgslFunctionBody(source:string,name:string,body:string):string{
 return rewriteWgslFunction(source,name,()=>body);
}
function replaceWgslFunctionSnippet(source:string,name:string,from:string,to:string):string{
 return rewriteWgslFunction(source,name,body=>{
  const next=body.replaceAll(from,to);if(next===body)throw Error(`Physical relation shader snippet changed in ${name}`);
  return next;
 });
}
/** PhysicalSourceTransport's source packet has no GPU-indexed voxel chart.
 * Remove the chart/frontier hooks from only the physical batch shader so
 * isolateComputeEntry cannot retain their cave/frontier call graph. The
 * generic shader remains the producer-capable implementation. */
function physicalSurfaceRelationsShader(source:string):string{
 let specialized=replaceWgslFunctionBody(source,'smallClosedPart','return false;');
 specialized=replaceWgslFunctionBody(specialized,'partAnchorCachedFrontierOwned','return false;');
 specialized=replaceWgslFunctionBody(specialized,'prepareAnchorParts','return params.frontier.z;');
 for(const hook of [
  'if(!chartEarlyEmitted()){chartEmitAt(sourceId,sourceEdgeCount()*selectedCount());}',
  'if(!residual&&posedContacts()){let inc=sourceEdgeCorner(halfedge(e.x).z,0u);if(inDraw(inc)&&validFace(inc)&&voxelColumnEdge(ordinal,triangle(inc,0u),inc/3u)){return;}}',
  'if(posedContacts()&&voxelClusterEmit(position(h.x,instance),position(h.y,instance),t,kind,e.x,vec2u(first/3u,select(0xffffffffu,second/3u,second!=0xffffffffu)),smallPart&&samePart)){return;}',
 ]){
  const next=specialized.replace(hook,'');if(next===specialized)throw Error('Physical relation shader hook changed');specialized=next;
 }
 specialized=replaceWgslFunctionBody(specialized,'boundaryPort','return vertexPort(v,instance);');
 for(const [from,to] of [
  ['let contact=posedEdgeSlot(ordinal);var canonical=canonicalEdge(ordinal,instance);','var canonical=canonicalEdge(ordinal,instance);'],
  ['if(contact!=NO_CONTACT){canonical=edge(atomicLoad(&contactWork[contactSlots()+contact])-1u).x;}',''],
  ['var member=0u;var contactHead=0u;if(contact!=NO_CONTACT){contactHead=atomicLoad(&contactWork[contactSlots()*2u+contact]);}','var member=0u;'],
  [`var samePart=true;var ownPart=0u;var smallPart=false;
 if(cellParts()){
  ownPart=atomicLoad(&contactWork[edgePartBase()+ordinal]);
  let at=partSlot(ownPart);if(ownPart!=0u&&at!=NO_CONTACT){smallPart=(atomicLoad(&contactWork[partFlagsBase()+at])&3u)==1u;}
 }`,'var smallPart=false;'],
  ['  if(contact!=NO_CONTACT){\n   if(contactHead==0u){break;}\n   let current=contactHead;contactHead=atomicLoad(&contactWork[contactSlots()*3u+(current-1u)]);h=edge(current-1u).x;\n  }\n  else{if(member>=members){break;}h=select(e.x,source[membersAt+member*2u],e.w>0u);member++;}','  if(member>=members){break;}h=select(e.x,source[membersAt+member*2u],e.w>0u);member++;'],
  ['  if(cellParts()){samePart=samePart&&atomicLoad(&contactWork[edgePartBase()+halfedge(h).z])==ownPart;}\n',''],
  ['smallPart&&samePart','false'],
 ])specialized=replaceWgslFunctionSnippet(specialized,'boundaryAtFine',from,to);
 // The physical host has no chart/contact arena. Remove the generic residual
 // branch from the boundary caller itself, rather than relying on a false
 // predicate that would keep binding 12 in the isolated call graph.
 specialized=replaceWgslFunctionBody(specialized,'boundaryAt',`if(levelDispatchEnabled()){
  if(id>=levelDispatchTotal()){return;}
  let record=levelDispatchFor(id);if(record.x==0xffffffffu){return;}
  let ordinal=id-record.z;let instance=record.x;if(!selected(instance)){return;}
  if(record.y==3u){authoredPathsAt(ordinal,instance);if(ordinal<sourceHullEdgeCount()){hullOutlineAt(ordinal,instance);}return;}
  if(record.y==2u){authoredPathsAt(ordinal,instance);let spans=sourcePlanarSpanCount();if(ordinal<spans){planarSpanAt(ordinal,instance);}else if(ordinal<spans+sourcePanelJoinCount()){planarJoinAt(ordinal-spans,instance);}return;}
  boundaryAtFine(ordinal,instance,record.w,false);return;
 }
 let slots=sourceBoundarySlots();if(id>=slots*selectedCount()||native[0]==0u){return;}
 boundaryAtFine(id%slots,selectedInstance(id/slots),id/slots,false);`);
 // Physical source packets cannot publish a chart suffix. Keep their contour
 // closure chart-free while generic direct consumers retain the remap.
 specialized=replaceWgslFunctionSnippet(specialized,'contourAt',` }else if(residualReady()){
  let faces=residualFaceCount();if(faces==0u||id>=faces*selectedCount()){return;}
  instance=selectedInstance(id/faces);face=residualFaceAt(id%faces);
 }else{`,` }else{`);
 return specialized;
}
export const SURFACE_RELATIONS_PHYSICAL_SHADER=physicalSurfaceRelationsShader(SURFACE_RELATIONS_SHADER);
/** Ordinary direct playback is a fine-only fallback for a producer that was
 * not present during warm-time admission. Keep the posed contact/canonical
 * correspondence and authored/planar/hull banks, but remove chart, residual
 * and voxel-cluster consumers from the line call graph. */
function ordinaryDirectSurfaceRelationsShader(source:string):string{
 let specialized=source;
 for(const hook of [
  'if(!chartEarlyEmitted()){chartEmitAt(sourceId,sourceEdgeCount()*selectedCount());}',
  'if(!residual&&posedContacts()){let inc=sourceEdgeCorner(halfedge(e.x).z,0u);if(inDraw(inc)&&validFace(inc)&&voxelColumnEdge(ordinal,triangle(inc,0u),inc/3u)){return;}}',
  'if(posedContacts()&&voxelClusterEmit(position(h.x,instance),position(h.y,instance),t,kind,e.x,vec2u(first/3u,select(0xffffffffu,second/3u,second!=0xffffffffu)),smallPart&&samePart)){return;}',
 ] as const){
  const next=specialized.replace(hook,'');if(next===specialized)throw Error('Ordinary fine relation shader hook changed');specialized=next;
 }
 specialized=replaceWgslFunctionBody(specialized,'boundaryAt',`if(levelDispatchEnabled()){
  if(id>=levelDispatchTotal()){return;}
  let record=levelDispatchFor(id);if(record.x==0xffffffffu){return;}
  let ordinal=id-record.z;let instance=record.x;if(!selected(instance)){return;}
  if(record.y==LEVEL_BANK_OUTLINE){authoredPathsAt(ordinal,instance);if(ordinal<sourceHullEdgeCount()){hullOutlineAt(ordinal,instance);}return;}
  if(record.y==LEVEL_BANK_PLANAR){authoredPathsAt(ordinal,instance);let spans=sourcePlanarSpanCount();if(ordinal<spans){planarSpanAt(ordinal,instance);}else if(ordinal<spans+sourcePanelJoinCount()){planarJoinAt(ordinal-spans,instance);}return;}
  boundaryAtFine(ordinal,instance,record.w,false);return;
 }
 let slots=sourceBoundarySlots();if(id>=slots*selectedCount()||native[0]==0u){return;}
 boundaryAtFine(id%slots,selectedInstance(id/slots),id/slots,false);`);
 specialized=replaceWgslFunctionSnippet(specialized,'contourAt',` }else if(residualReady()){
  let faces=residualFaceCount();if(faces==0u||id>=faces*selectedCount()){return;}
  instance=selectedInstance(id/faces);face=residualFaceAt(id%faces);
 }else{`,` }else{`);
 specialized=replaceWgslFunctionSnippet(specialized,'contactVertexAt',` let residual=residualReady();var sourceV=v;
 if(residual){if(v>=residualVertexCount()){return;}sourceV=residualVertexAt(v);}
 if(sourceV>=min(sourceFaceCount()*2u,native[0]/6u*4u)){return;}
 // Owned by the selected coarse representation: no fine port is ever read.
 if(!residual&&voxelVertexCoarse(sourceV)){return;}`,` var sourceV=v;
 if(sourceV>=min(sourceFaceCount()*2u,native[0]/6u*4u)){return;}`);
 specialized=replaceWgslFunctionSnippet(specialized,'contactEdgeAt',` let residual=residualReady();var edgeId=id;if(residual){if(id>=residualEdgeCount()){return;}edgeId=residualEdgeAt(id);}else if(id>=sourceEdgeCount()||!edgeInDraw(id)){return;}`,` var edgeId=id;if(id>=sourceEdgeCount()||!edgeInDraw(id)){return;}`);
 specialized=replaceWgslFunctionSnippet(specialized,'contactEdgeAt','if(!residual&&voxelEdgeCoarse(edgeId)){return;}','');
 specialized=replaceWgslFunctionSnippet(specialized,'contactSortAt',` let residual=residualReady();var edgeId=id;
 if(residual){if(id>=residualEdgeCount()){return;}edgeId=residualEdgeAt(id);}
 else if(id>=sourceEdgeCount()||!edgeInDraw(id)){return;}`,` var edgeId=id;if(id>=sourceEdgeCount()||!edgeInDraw(id)){return;}`);
 return specialized;
}
export const SURFACE_RELATIONS_GENERIC_DIRECT_ORDINARY_SHADER=ordinaryDirectSurfaceRelationsShader(SURFACE_RELATIONS_SHADER);
const SURFACE_RELATIONS_GENERIC_DIRECT_SHADER=SURFACE_RELATIONS_PHYSICAL_SHADER;
function voxelSurfaceRelationsShader(source:string):string{
 let specialized=source;
 specialized=replaceWgslFunctionSnippet(specialized,'boundaryAtFine','if(!chartEarlyEmitted()){chartEmitAt(sourceId,sourceEdgeCount()*selectedCount());}','');
 for(const [name,body] of [
  ['authoredPathsAt','return;'],['authoredOwnsFace','return false;'],['authoredOwnsEdge','return false;'],
  ['planarProgramLevel','return false;'],['planarSpanAt','return;'],['planarJoinAt','return;'],['hullOutlineAt','return;'],['outlineLevel','return false;'],
  ['levelBankFor','if(native[0]==0u||!levelOwnerCertified(instance)){return LEVEL_BANK_UNKNOWN_FINE;}if(principalLevel(levelOwner(instance))){return LEVEL_BANK_PRINCIPAL_FINE;}return LEVEL_BANK_FINE;'],
  ['levelBankCount','return sourceBoundarySlots();'],
 ] as const)specialized=replaceWgslFunctionBody(specialized,name,body);
 for(const snippet of [
  'authoredPathsAt(ordinal,instance);',
  `if(planarProgramLevel(instance)){
  let v=halfedge(0u).x;if(ownerUnseen(v,instance)||ownerMerged(edgeOwner(v,instance))){return;}
  if(ordinal<sourcePlanarSpanCount()){planarSpanAt(ordinal,instance);}else{planarJoinAt(ordinal-sourcePlanarSpanCount(),instance);}return;
 }`,
  'if(outlineLevel(edgeOwner(halfedge(e.x).x,instance))){hullOutlineAt(ordinal,instance);return;}',
  'if(faces<=2u&&authoredOwnsEdge(first,second,backFirst,backSecond)){return;}',
 ] as const)specialized=replaceWgslFunctionSnippet(specialized,'boundaryAtFine',snippet,'');
 specialized=replaceWgslFunctionSnippet(specialized,'chartSelectAt','let bankFallback=sourcePathSegmentCount()!=0u||sourcePlanarSpanCount()!=0u||sourcePanelJoinCount()!=0u||sourceHullEdgeCount()!=0u;','let bankFallback=false;');
 specialized=replaceWgslFunctionBody(specialized,'boundaryAt',`if(levelDispatchEnabled()){
  if(id>=levelDispatchTotal()){return;}
  let record=levelDispatchFor(id);if(record.x==0xffffffffu){return;}
  let ordinal=id-record.z;let instance=record.x;if(!selected(instance)){return;}
  boundaryAtFine(ordinal,instance,record.w,false);return;
 }
 if(residualReady()){
  let residualEdges=residualEdgeCount();if(residualEdges==0u){return;}
  let total=residualEdges*selectedCount();if(id>=total){return;}
  let selectedOrdinal=id/residualEdges;let residualOrdinal=id%residualEdges;boundaryAtFine(residualEdgeAt(residualOrdinal),selectedInstance(selectedOrdinal),selectedOrdinal,true);return;
 }
 let slots=sourceBoundarySlots();if(id>=slots*selectedCount()||native[0]==0u){return;}
 boundaryAtFine(id%slots,selectedInstance(id/slots),id/slots,false);`);
 const marker='@compute @workgroup_size(64) fn boundaries(';const contourMarker='@compute @workgroup_size(64) fn contours(';
 if(!specialized.includes(marker)||!specialized.includes(contourMarker))throw Error('Voxel relation line entry changed');
 return specialized.replace(marker,'@compute @workgroup_size(64) fn boundariesVoxel(').replace(contourMarker,'@compute @workgroup_size(64) fn contoursVoxel(');
}
export const SURFACE_RELATIONS_GENERIC_DIRECT_VOXEL_SHADER=voxelSurfaceRelationsShader(SURFACE_RELATIONS_SHADER);
function makeBatchShader(source:string,disableChartStages=false):string{
 const chartEntry=(name:string,body:string)=>batchEntry(name,disableChartStages?'return;':body);
 const batchSource=source.replace('fn residualDispatchBase()->u32{return 0u;}','fn residualDispatchBase()->u32{return B1.z;}');
 return rebaseSourceArrays(stripWgslComments(batchSource).replace(PARAMS_BINDING,''))+/* wgsl */`
struct BatchItem {p:Params,base:vec4u,base2:vec4u,span:vec4u}
struct BatchTable {head:vec4u,items:array<BatchItem,${SURFACE_RELATION_BATCH_ITEMS}>}
@group(0) @binding(0) var<uniform> table:BatchTable;
var<private> params:Params;
var<private> B0:vec4u;
var<private> B1:vec4u;
const BATCH_NONE=0xffffffffu;
fn batchEnter(i:u32){params=table.items[i].p;B0=table.items[i].base;B1=table.items[i].base2;}
fn batchLocal(wid:vec3u,local:u32)->u32{
 let global=(wid.x+wid.y*table.head.y)*64u+local;if(global>=table.head.z){return BATCH_NONE;}
 var lo=0u;var hi=table.head.x;
 loop{if(hi-lo<=1u){break;}let mid=(lo+hi)/2u;if(table.items[mid].span.x<=global){lo=mid;}else{hi=mid;}}
 batchEnter(lo);let id=global-table.items[lo].span.x;if(id>=table.items[lo].span.y){return BATCH_NONE;}return id;
}
// The per-source path sizes these three tensor stages from prepareInterior's
// dispatch words (zero when the source is invalid or its tensor is skipped).
fn batchGate(word:u32)->bool{return B1.w==0u||dispatch[B1.z+word]!=0u;}
${chartEntry('batchContactClear','contactClearAt(id);')}
${chartEntry('batchVoxelVote','voxelVoteAt(id);')}
${chartEntry('batchVoxelBin','voxelBinAt(id);')}
${chartEntry('batchChartCorners','chartCornerAt(id);')}
${chartEntry('batchChartCertify','chartCertifyAt(id);')}
${chartEntry('batchChartSelectStage','chartSelectAt(id);')}
${chartEntry('batchChartEmit','chartEmitAt(id,max(1u,chartClusters()));if(id==0u&&posedContacts()&&chartOn()){atomicStore(&contactWork[chartEarlyEmittedWord()],1u);}')}
${chartEntry('batchChartPublish','if(id==0u&&posedContacts()&&chartOn()){residualDispatchWrite();}')}
${batchEntry('batchPartAnchorClear','partAnchorClearAt(id);')}
${batchEntry('batchPartAnchors','partAnchorAt(id);')}
${batchEntry('batchPrepareFrontier','if(local==0u){prepareIn();}')}
${chartEntry('batchContactVertices','contactVertexAt(id);')}
${chartEntry('batchContactEdges','contactEdgeAt(id);')}
${chartEntry('batchContactSort','contactSortAt(id);')}
${batchEntry('batchBoundaryLines','boundaryAt(id);')}
${batchEntry('batchContourLines','contourAt(id);')}
@compute @workgroup_size(64) fn batchPrepareInterior(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) local:u32){batchEnter(wid.x);prepareInteriorIn(local);}
${batchEntry('batchFaceDifferentials','faceDifferentialAt(id);')}
${batchEntry('batchRadialCurvature','radialCurvatureAt(id);')}
${batchEntry('batchSuggestiveContours','suggestiveAt(id);')}
${batchEntry('batchPrincipalCurvature','principalCurvatureAt(id);')}
${batchEntry('batchRidgeDerivative','ridgeDerivativeAt(id);')}
${batchEntry('batchRidgeLines','ridgeLineAt(id);')}
${batchEntry('batchViewFaceLines','suggestiveAt(id);ridgeLineAt(id);')}
`;
}
export const SURFACE_RELATIONS_BATCH_SHADER=makeBatchShader(SURFACE_RELATIONS_SHADER);
/** Physical batches retain the stage entry ABI, but the chart/contact stage
 * bodies are inert because the guarded physical source contract cannot supply
 * their producer-owned chart. No chart or cave/frontier helper is reachable
 * from the line or anchor stages. */
export const SURFACE_RELATIONS_PHYSICAL_BATCH_SHADER=makeBatchShader(SURFACE_RELATIONS_PHYSICAL_SHADER,true);
type BatchStage='partAnchorClear'|'partAnchors'|'prepareFrontier'|'contactClear'|'voxelVote'|'voxelBin'|'chartCorners'|'chartCertify'|'chartSelectStage'|'chartEmit'|'chartPublish'|'contactVertices'|'contactEdges'|'contactSort'|'boundaryLines'|'contourLines'|'prepareInterior'|'faceDifferentials'|'radialCurvature'|'suggestiveContours'|'principalCurvature'|'ridgeDerivative'|'ridgeLines'|'viewFaceLines';
/** Global stage order. Per source it is the per-source order (contacts, lines,
 * prepareInterior, tensor, view lines); sources write disjoint ranges. */
const BATCH_ORDER:readonly BatchStage[]=['partAnchorClear','prepareFrontier','contactClear','voxelVote','voxelBin','chartCorners','chartCertify','chartSelectStage','chartEmit','chartPublish','contactVertices','contactEdges','contactSort','boundaryLines','contourLines','partAnchors','prepareInterior','faceDifferentials','radialCurvature','suggestiveContours','principalCurvature','ridgeDerivative','ridgeLines','viewFaceLines'];
const batchEntryName=(stage:BatchStage)=>'batch'+stage[0]!.toUpperCase()+stage.slice(1);
// Direct and batched anchor dispatches execute the same projection body. It
// links each complete record atomically; there is no second insertion owner.
export const SURFACE_PART_ANCHOR_BINDINGS={
 partAnchorClear:[0,17],
 // The generation-owned cache and current-frame marker live in the source
 // packet. Projection reads that marker and never re-walks canonical samples.
 partAnchors:[0,2,3,4,8,10,16,17,19],
} as const;
// Direct and batched stages share their bodies and resource contract. Frontier
// compilation moved partCodes/anchor input from certification to vertex marking.
export const SURFACE_CONTACT_BINDINGS={
 contactClear:[0,1,4,8,12],voxelVote:[0,1,2,4,8,12,13],voxelHome:[0,1,4,8,12],
 voxelBin:[0,1,2,4,8,12,13],chartCorners:[0,1,2,4,8,12,13],
 chartCertify:[0,1,4,8,12,18],contactVertices:[0,1,2,4,8,12,13,16],
 contactEdges:[0,1,2,4,8,12,13,14],contactSort:[0,1,2,4,8,12],chartSelectStage:[0,1,4,8,12],chartEmit:[0,1,2,3,4,5,6,8,10,11,12,15],chartPublish:[0,1,4,7,8,12],
} as const;
const BATCH_BINDINGS:Record<BatchStage,readonly number[]>={...SURFACE_PART_ANCHOR_BINDINGS,prepareFrontier:[0,1,2,3,4,5,6,7,8,10,11],...SURFACE_CONTACT_BINDINGS,boundaryLines:[0,1,2,3,4,5,6,8,10,11,12,15],contourLines:[0,1,2,3,4,5,6,8,10,11,12,15],prepareInterior:[0,1,2,3,4,6,7,8,9,10,11],faceDifferentials:[0,1,2,3,4,6,8,9],radialCurvature:[0,1,2,3,4,8,9],suggestiveContours:[0,1,2,3,4,5,6,8,9,15],principalCurvature:[0,1,2,3,4,8,9],ridgeDerivative:[0,1,2,4,8,9],ridgeLines:[0,1,2,3,4,5,6,8,9,15],viewFaceLines:[0,1,2,3,4,5,6,8,9,15]};
const BATCH_INDIRECT_OFFSETS:Partial<Record<BatchStage,number>>={boundaryLines:0,contourLines:12,contactClear:24,contactVertices:36,contactEdges:48,contactSort:48,partAnchors:72,voxelVote:84,voxelBin:108,chartCorners:120,chartCertify:132,faceDifferentials:24,radialCurvature:36,suggestiveContours:48,principalCurvature:36,ridgeDerivative:36,ridgeLines:48,viewFaceLines:48};
/** The physical shader's boundary no longer reaches partCodes or the chart;
 * binding 13 is therefore omitted from its host bind group. Generic and
 * GPU-indexed stages keep the producer-capable map above. */
const PHYSICAL_BATCH_BINDINGS:Record<BatchStage,readonly number[]>={...BATCH_BINDINGS,prepareFrontier:[0,1,2,3,4,5,6,7,8,10,11],contactClear:[0],voxelVote:[0],voxelBin:[0],chartCorners:[0],chartCertify:[0],chartSelectStage:[0],chartEmit:[0],chartPublish:[0],contactVertices:[0],contactEdges:[0],contactSort:[0],boundaryLines:BATCH_BINDINGS.boundaryLines.filter(binding=>binding!==12&&binding!==13),contourLines:BATCH_BINDINGS.contourLines.filter(binding=>binding!==12)};
// Params grew by one vec4u for the per-source level prefix. The batch table
// keeps its existing source/base/span trailer after the complete Params.
const BATCH_ITEM_WORDS=68,BATCH_DISPATCH_WORDS=36;
/** The header is part of the uniform binding too. Select a complete table
 * once per device, then use its identical item count in WGSL and host splits.
 * Smaller devices get more batches, never fewer source constructions. */
function relationBatchItems(device:GPUDevice):number{
 const limit=device.limits?.maxUniformBufferBindingSize??16384;
 const count=Math.min(SURFACE_RELATION_BATCH_ITEMS,Math.floor((limit-16)/(BATCH_ITEM_WORDS*4)));
 if(count<1)throw Error(`MAIN relation table cannot fit one complete source in ${limit}-byte uniform binding`);
 return count;
}
/** Retained posed tensors of batched sources, as ranges of a few large pages
 * (every source of one page shares one dispatch). Pages grow 32 MB, 64 MB ...
 * up to the device binding limit; a retired range is reused only after the
 * submission that retired it (its readers are ordered before any new writer). */
interface TensorRange{readonly page:TensorPage;readonly offset:number;readonly size:number}
interface TensorPage{readonly buffer:GPUBuffer;readonly capacity:number;readonly free:{offset:number;size:number}[];live:number}
class TensorPages{
 private readonly pages:TensorPage[]=[];private retiring:TensorRange[]=[];private next=32*1024*1024;
 constructor(private readonly device:GPUDevice,private readonly maximum:number){}
 allocate(bytes:number):TensorRange{
  const size=Math.ceil(bytes/256)*256;if(size>this.maximum)throw Error('Posed tensor range exceeds the device binding size');
  let page:TensorPage|undefined,hole=-1,best=Infinity;
  for(const candidate of this.pages)candidate.free.forEach((span,i)=>{if(span.size>=size&&span.size<best){page=candidate;hole=i;best=span.size;}});
  if(!page){
   const capacity=Math.min(this.maximum,Math.max(size,this.next));this.next=Math.min(this.maximum,capacity*2);
   page={buffer:this.device.createBuffer({label:'MAIN retained posed shape tensors (batched)',size:capacity,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),capacity,free:[{offset:0,size:capacity}],live:0};this.pages.push(page);hole=0;
  }
  const span=page.free[hole]!,offset=span.offset;span.offset+=size;span.size-=size;if(!span.size)page.free.splice(hole,1);page.live++;
  return {page,offset,size};
 }
 retire(range:TensorRange):void{this.retiring.push(range);}
 submitted():void{
  for(const {page,offset,size} of this.retiring.splice(0)){
   page.live--;page.free.push({offset,size});page.free.sort((a,b)=>a.offset-b.offset);
   for(let i=1;i<page.free.length;){const a=page.free[i-1]!,b=page.free[i]!;if(a.offset+a.size===b.offset){a.size+=b.size;page.free.splice(i,1);}else i++;}
   if(!page.live){page.buffer.destroy();this.pages.splice(this.pages.indexOf(page),1);}
  }
 }
 dispose():void{for(const page of this.pages)page.buffer.destroy();this.pages.length=0;this.retiring=[];}
}
interface BatchItem{readonly stage:BatchStage;readonly threads:number;readonly words:Uint32Array;readonly base:readonly number[];readonly base2:readonly number[];readonly bound:RelationBinding[];readonly dynamic?:boolean}
interface BatchQueue{readonly items:BatchItem[];readonly clears:GPUBufferBinding[];readonly seen:Set<object>;contactWords:number;dispatchWords:number;levelPrefixRecords:number}

/** A physical batch's static draw extent: index count and instance count
 * (its metadata words 0 and 1, written once with the batch). */
export interface SourceDrawExtent {readonly indices:number;readonly instances:number}
type RelationBinding=SourceStorageBinding|{readonly view:GPUTextureView};
type RelationGroups=Map<string,{buffers:RelationBinding[];group:GPUBindGroup}>;
type RetainedBindings={params:GPUBuffer;selection?:GPUBuffer;ownerKey:string;groups:RelationGroups;ranges:Map<number,{uniform:GPUBuffer;groups:RelationGroups}>;serial:number};
/** Uses original incidence and current posed candidate contacts. No spatial
 * hash redefines source topology; visibility cannot alter its port addresses. */
/** LINES4 (snow-trails): the run pass keeps a plane-class source's radial
 * features where its posed mesh bends (surface-relations shader bends). The
 * tag is computed only for a source whose REST mesh is plane-class by the
 * composition rule's plane count (composition.ts:278, at most 24 quantised
 * planes), so no other source pays for the neighbour reads. Once per
 * resident geometry. */
const planarRestCache=new WeakMap<object,boolean>();
function planarRest(geometry:{readonly positions:Float32Array;readonly indices:Uint32Array;readonly surface?:string}):boolean{
  const cached=planarRestCache.get(geometry);if(cached!==undefined)return cached;
  // A liquid sheet's posed waves are region-led (the tone lane's sea field),
  // not interior lines: it keeps the plane rule (Y8 view-02 water, h15).
  if(geometry.surface==='liquid'){planarRestCache.set(geometry,false);return false;}
  const p=geometry.positions,ix=geometry.indices;let lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];
  for(let i=0;i+2<p.length;i+=3)for(let k=0;k<3;k++){lo[k]=Math.min(lo[k]!,p[i+k]!);hi[k]=Math.max(hi[k]!,p[i+k]!);}
  const radius=Math.max(1e-9,Math.hypot(hi[0]!-lo[0]!,hi[1]!-lo[1]!,hi[2]!-lo[2]!)/2),planes=new Set<string>();let planar=true;
  for(let f=0;f+2<ix.length;f+=3){
    const a=ix[f]!*3,b=ix[f+1]!*3,c=ix[f+2]!*3;
    const ux=p[b]!-p[a]!,uy=p[b+1]!-p[a+1]!,uz=p[b+2]!-p[a+2]!,vx=p[c]!-p[a]!,vy=p[c+1]!-p[a+1]!,vz=p[c+2]!-p[a+2]!;
    let nx=uy*vz-uz*vy,ny=uz*vx-ux*vz,nz=ux*vy-uy*vx;const l=Math.hypot(nx,ny,nz);if(l<=0)continue;nx/=l;ny/=l;nz/=l;
    const d=nx*p[a]!+ny*p[a+1]!+nz*p[a+2]!;
    planes.add([nx,ny,nz].map(v=>Math.round(v*10000)).join('/')+'/'+Math.round(d/radius*10000));
    if(planes.size>24){planar=false;break;}
  }
  planarRestCache.set(geometry,planar);return planar;
}




export const CHART_CELLS=32768;





export const PART_ANCHOR_RECORD_WORDS=16;




/** Homogeneous clip(4), source parameter, original edge endpoints and t.
 * The exact directory owns the repeated owner/instance/feature/context. */
export const PART_ANCHOR_CANONICAL_WORDS=8;









/** The prepared row table is packed into the articulation counts arena.  The
 * relation owner publishes only its immutable layout; GPU storage is owned by
 * the articulation job so no ninth bind-group storage slot is needed. */
export interface SourcePartRowTable {readonly touchedRows:number}
export interface SourcePartAnchorTable {readonly buffer:GPUBuffer;readonly headSlots:number;readonly headBase:number;readonly recordBase:number;readonly recordWords:number;readonly canonicalBase?:number;readonly canonicalRecords?:number;readonly canonicalWords?:number;readonly rowTable?:SourcePartRowTable}
/** Same-submission live anchor demand. The GPU cursor is copied after the
 * relation pass and read only after that submission's fence; it never drives
 * a partial publication or a previous-frame estimate. */
export interface PartAnchorDemand {readonly records:number;readonly canonicalRows:number;readonly overflow:boolean;readonly selected:number}
export interface PartAnchorDemandReadback {collect():Promise<PartAnchorDemand>;dispose():void}
const samePartAnchorVertices=(a:Uint32Array|undefined,b:Uint32Array|undefined):boolean=>a===b||(!!a&&!!b&&a.length===b.length&&a.every((value,index)=>value===b[index]));
interface PartAnchorCanonicalLayout {readonly parts:number;readonly meta:number;readonly recordAt:number;readonly sampleAt:number;readonly sampleWords:number;readonly sampleCount:number;readonly instanceStride:number}
/** Find the source packet's trailing canonical table without interpreting an
 * old six-word anchor array as metadata. This is residency-time parsing only;
 * no topology scan occurs on a camera frame. */
const partAnchorCanonicalLayout=(vertices:Uint32Array|undefined):PartAnchorCanonicalLayout|undefined=>{
 if(!vertices)return;
 for(let parts=1;parts*6+TOPOLOGY_PART_CANONICAL_HEADER_WORDS<=vertices.length;parts++){
  const meta=parts*6;
  if(vertices[meta]!==TOPOLOGY_PART_CANONICAL_MAGIC||vertices[meta+1]!==TOPOLOGY_PART_CANONICAL_VERSION||vertices[meta+3]!==parts||vertices[meta+6]!==TOPOLOGY_PART_CANONICAL_RECORD_WORDS||vertices[meta+7]!==TOPOLOGY_PART_CANONICAL_SAMPLE_WORDS)continue;
  const recordAt=vertices[meta+2]!,sampleAt=vertices[meta+4]!,sampleWords=vertices[meta+5]!;
  if(recordAt<meta+TOPOLOGY_PART_CANONICAL_HEADER_WORDS||recordAt+parts*TOPOLOGY_PART_CANONICAL_RECORD_WORDS>vertices.length||sampleWords%TOPOLOGY_PART_CANONICAL_SAMPLE_WORDS!==0||sampleAt+sampleWords>vertices.length)continue;
  let sampleCount=0;for(let part=0;part<parts;part++){const at=recordAt+part*TOPOLOGY_PART_CANONICAL_RECORD_WORDS;sampleCount+=vertices[at+5]!;}
  return {parts,meta,recordAt,sampleAt,sampleWords,sampleCount,instanceStride:sampleCount};
 }
 return;
};
const partAnchorPartCount=(vertices:Uint32Array|undefined):number=>{
 const canonical=partAnchorCanonicalLayout(vertices);
 return canonical?.parts??(vertices&&vertices.length>=6&&vertices.length%6===0?vertices.length/6:0);
};

export function currentRestCellChart(indexed:{readonly generation:number;readonly restCellChart?:GpuIndexedRestCellChart}|undefined):GpuIndexedRestCellChart|undefined{
  const chart=indexed?.restCellChart;return chart&&chart.kind==='voxel-rest-cell-chart'&&chart.sourceGeneration===indexed!.generation?chart:undefined;
}



export function chartOccupancy(indexed:{readonly generation:number;readonly restCellChart?:GpuIndexedRestCellChart}|undefined,extent:readonly [number,number,number]):GpuIndexedCellOccupancy|undefined{
  const chart=currentRestCellChart(indexed),occupancy=chart?.occupancy;
  return occupancy?.kind==='voxel-column-occupancy-v1'&&chart!.extent.every((n,i)=>n===extent[i])?occupancy:undefined;
}
export function cellChart(extent:readonly [number,number,number]):{cells:number;corners:number;clusters:number}|undefined{
  const [x,y,z]=extent;if(![x,y,z].every(n=>Number.isInteger(n)&&n>=1&&n<=256))return undefined;
  const cells=x*y*z;if(cells>CHART_CELLS)return undefined;
  return {cells,corners:(x+1)*(y+1)*(z+1),clusters:Math.ceil(x/2)*Math.ceil(y/2)*Math.ceil(z/2)};
}
/** Choose an optional producer-owned contact suffix without changing the
 * required incumbent arena. `growth` is 1 for direct contactWork and 1.25 for
 * the batched owner, matching its existing aggregate allocation policy. */
export function chooseOptionalContactArena(baselineWords:number,residualWords:number,maximumBytes:number,usedWords=0,growth=1):{readonly words:number;readonly enabled:boolean}{
 for(const value of [baselineWords,residualWords,maximumBytes,usedWords,growth])if(!Number.isFinite(value)||value<0)throw Error('Invalid optional contact arena budget');
 if(!Number.isInteger(baselineWords)||!Number.isInteger(residualWords)||!Number.isInteger(usedWords)||growth<=0)throw Error('Optional contact arena extents must be integral');
 const maximumWords=Math.floor(maximumBytes/4);
 const fits=(words:number)=>words<=maximumWords&&Math.min(Math.ceil(words*growth),maximumWords)*4<=maximumBytes;
 if(!fits(usedWords+baselineWords))throw Error(`Complete contact arena (${usedWords+baselineWords} words) exceeds device binding capacity ${maximumBytes} bytes`);
 const full=usedWords+baselineWords+residualWords;
 return {words:baselineWords+(residualWords>0&&fits(full)?residualWords:0),enabled:residualWords>0&&fits(full)};
}
export interface ContactArenaHomePlan {readonly retainHome:boolean;readonly scratchWords:number;readonly homeWords:number}
/** Select the optional suffix only after accounting for the direct allocator's
 * retained-home reserve and its second 1.25 growth. The transient plan is the
 * safe fallback when a new persistent home would not fit. */
export function chooseOptionalContactArenaWithHome(baselineWords:number,residualWords:number,maximumBytes:number,full:ContactArenaHomePlan,baselinePlan:ContactArenaHomePlan,transient?:ContactArenaHomePlan):{readonly words:number;readonly enabled:boolean;readonly plan:ContactArenaHomePlan}{
 const values=[baselineWords,residualWords,maximumBytes,full.scratchWords,full.homeWords,baselinePlan.scratchWords,baselinePlan.homeWords,...(transient?[transient.scratchWords,transient.homeWords]:[])];
 for(const value of values)if(!Number.isFinite(value)||value<0||!Number.isInteger(value))throw Error('Invalid retained contact-home budget');
 const bytes=(words:number,plan:ContactArenaHomePlan)=>{
  const required=plan.retainHome?Math.max(words,plan.scratchWords+plan.homeWords):words;
  return (plan.retainHome?Math.max(required,Math.min(Math.ceil(required*1.25),Math.floor(maximumBytes/4))):required)*4;
 };
 if(bytes(baselineWords+residualWords,full)<=maximumBytes)return {words:baselineWords+residualWords,enabled:residualWords>0,plan:full};
 if(bytes(baselineWords,baselinePlan)<=maximumBytes)return {words:baselineWords,enabled:false,plan:baselinePlan};
 if(transient&&bytes(baselineWords,transient)<=maximumBytes)return {words:baselineWords,enabled:false,plan:transient};
 throw Error(`Complete retained contact arena exceeds device binding capacity ${maximumBytes} bytes`);
}
/** params.selection.w: 0 without a posed revision, 1 with one (unchanged meaning, tested ==0u), and for a cell-chart
 * source 2 + its exact source GENERATION (0..POSED_CONTENT_MAX): the chart's occupancy is generation-owned, so a
 * retained home survives every pose. Outside that range: 1, never retained. */
export function chartToken(batch:{readonly source:{readonly posedRevision?:number}},source:unknown,chart:boolean):number{
  if(batch.source.posedRevision===undefined)return 0;
  const generation=chart&&isGpuIndexedResident(source as ResidentRelationSource)?(source as {gpu:{generation:number}}).gpu.generation:undefined;
  if(generation===undefined||!Number.isSafeInteger(generation)||generation<0||generation>POSED_CONTENT_MAX)return 1;
  return generation+2;
}
interface ColumnHome {readonly offset:number;readonly slots:number;readonly levels:readonly number[];readonly anchorParts:number;source:unknown;fresh:boolean}


export function chartLevelClusters(extent:readonly [number,number,number]):readonly number[]{
  return [2,4,8,16].map(b=>extent.reduce((n,e)=>n*Math.ceil(e/b),1));
}
/** Words of one source's four contiguous level homes (each: 4 header words + 4 words per cluster). */
export function chartHomeWords(levels:readonly number[]):number{return levels.reduce((n,clusters)=>n+4+clusters*4,0);}


export function chartRequirementWords(extent:readonly [number,number,number]):number{return chartLevelClusters(extent).reduce((n,clusters)=>n+clusters,0);}
/** Home words of all chunks together; past it every home restarts (bounded memory, 16 MB). */
const COLUMN_HOME_WORDS=1<<22;
/** The largest posed content revision with its own token (POSED_CONTENT_MAX+2 = 0xffffffff). */
export const POSED_CONTENT_MAX=0xfffffffd;
/** params.selection.w: 0 without a posed revision (unchanged meaning, tested ==0u), 1 with one, and
 * 2 + the producer's posed CONTENT revision when it certifies one (only then can occupancy persist).
 * EXACT, never a hash or modulo: revisions 0..POSED_CONTENT_MAX map one-to-one onto 2..0xffffffff, so two
 * distinct revisions never share a token. Anything else (out of range, negative, not an integer) is 1:
 * never retained, the chunk rebuilds every frame. */
export function posedContentToken(batch:{readonly source:{readonly posedRevision?:number};readonly gpuPose?:{readonly posedContentRevision?:number}}):number{
  if(batch.source.posedRevision===undefined)return 0;
  const content=batch.gpuPose?.posedContentRevision;
  if(content===undefined||!Number.isSafeInteger(content)||content<0||content>POSED_CONTENT_MAX)return 1;
  return content+2;
}







export function tensorRevision(batch:{readonly source:{readonly posedRevision?:number;readonly visibleInstanceFrontier?:unknown;readonly nativeVisibleInstanceFrontier?:unknown};readonly gpuPose?:{readonly posedContentRevision?:number}},frontierAdmitted=false):number|undefined{
  if(batch.source.posedRevision===undefined)return undefined;
  // Parent admission compacts the GPU frontier after form-depth resolution;
  // its final membership is not available to the CPU cache key. Rebuild the
  // selected tensor layout rather than reusing a buffer indexed by a prior
  // camera's compacted ordinals.
 if(frontierAdmitted&&(batch.source.visibleInstanceFrontier!==undefined||batch.source.nativeVisibleInstanceFrontier!==undefined))return undefined;
  if(!batch.gpuPose)return batch.source.posedRevision;
  const token=posedContentToken(batch);return token>1?token:undefined;
}
const SURFACE_RELATION_ENTRIES=['admitFrontier','prepare','partAnchorClear','partAnchors','contactClear','voxelVote','voxelHome','voxelBin','chartCorners','chartCertify','chartSelectStage','chartEmit','chartPublish','contactVertices','contactEdges','contactSort','boundaries','contours','surfaceLines'] as const;
const SURFACE_RELATION_GENERIC_DIRECT_ENTRIES=[...SURFACE_RELATION_ENTRIES.filter(entry=>entry!=='surfaceLines'),'boundariesVoxel','contoursVoxel'];
const SURFACE_RELATION_GENERIC_DIRECT_ORDINARY_ENTRIES=['admitFrontier','prepare','partAnchorClear','partAnchors','contactClear','contactVertices','contactEdges','contactSort','boundaries','contours'] as const;
const SURFACE_RELATION_INTERIOR_ENTRIES=['prepareInterior','faceDifferentials','radialCurvature','suggestiveContours','principalCurvature','ridgeDerivative','ridgeLines','viewFaceLines'] as const;
// The batched entries exist only with the interior stages (the physical path).
const SURFACE_RELATION_BATCH_ENTRIES=BATCH_ORDER.map(batchEntryName);
type SurfaceRelationEntry=typeof SURFACE_RELATION_ENTRIES[number]|typeof SURFACE_RELATION_INTERIOR_ENTRIES[number]|string;
/** Preparation contract for the shared relation owner. `generic` remains the
 * public compatibility profile for consumers that may use both direct and
 * batched append contracts. MAIN's ordinary example caller uses
 * `generic-direct`: it retains every direct GPU-indexed, voxel/contact and
 * interior producer, but has no extent/frontier batch contract and therefore
 * must not compile the batch-only entry points. The compiled physical host
 * supplies the complete batch contract (extent + visibility frontier + posed
 * revision) and queues every source into flushBatched(), so it uses the
 * physical profile instead of guessing from a scene name. */
export type SurfaceRelationPreparation='generic'|'generic-direct'|'generic-direct-ordinary'|'physical-batched';
const surfaceRelationEntries=(interior:boolean,preparation:SurfaceRelationPreparation='generic'):readonly SurfaceRelationEntry[]=>
 preparation==='physical-batched'?(interior?['admitFrontier',...SURFACE_RELATION_BATCH_ENTRIES]:['admitFrontier']):
 preparation==='generic-direct-ordinary'?(interior?[...SURFACE_RELATION_GENERIC_DIRECT_ORDINARY_ENTRIES,...SURFACE_RELATION_INTERIOR_ENTRIES]:SURFACE_RELATION_GENERIC_DIRECT_ORDINARY_ENTRIES):
 preparation==='generic-direct'?(interior?[...SURFACE_RELATION_GENERIC_DIRECT_ENTRIES,...SURFACE_RELATION_INTERIOR_ENTRIES]:SURFACE_RELATION_GENERIC_DIRECT_ENTRIES):
 interior?[...SURFACE_RELATION_ENTRIES,...SURFACE_RELATION_INTERIOR_ENTRIES,...SURFACE_RELATION_BATCH_ENTRIES]:SURFACE_RELATION_ENTRIES;
interface PreparedSurfaceRelations{
 readonly plans:ReadonlyMap<SurfaceRelationEntry,ComputePipelinePlan>;
 readonly pipelines:ReadonlyMap<SurfaceRelationEntry,GPUComputePipeline>;
 readonly preparation:SurfaceRelationPreparation;
}
function surfaceRelationPipelinePlans(device:GPUDevice,interior:boolean,preparation:SurfaceRelationPreparation='generic'):ReadonlyMap<SurfaceRelationEntry,ComputePipelinePlan>{
 const plans=new Map<SurfaceRelationEntry,ComputePipelinePlan>();
 const batchSource=(preparation==='physical-batched'?SURFACE_RELATIONS_PHYSICAL_BATCH_SHADER:SURFACE_RELATIONS_BATCH_SHADER).replace(`items:array<BatchItem,${SURFACE_RELATION_BATCH_ITEMS}>`,`items:array<BatchItem,${relationBatchItems(device)}>`);
 for(const entry of surfaceRelationEntries(interior,preparation)){
  const batched=entry.startsWith('batch');
  const voxelEntry=entry==='boundariesVoxel'||entry==='contoursVoxel';
  const directSource=preparation==='generic-direct'&&voxelEntry?SURFACE_RELATIONS_GENERIC_DIRECT_VOXEL_SHADER:preparation==='generic-direct-ordinary'&&!batched?SURFACE_RELATIONS_GENERIC_DIRECT_ORDINARY_SHADER:(preparation==='generic-direct'||preparation==='generic-direct-ordinary')&&(entry==='boundaries'||entry==='contours')?SURFACE_RELATIONS_GENERIC_DIRECT_SHADER:SURFACE_RELATIONS_SHADER;
  const entryPoint=entry==='boundariesVoxel'?'boundariesVoxel':entry==='contoursVoxel'?'contoursVoxel':entry;
  const code=batched?isolateComputeEntry(batchSource,entry):isolateComputeEntry(directSource,entryPoint);
  plans.set(entry,()=>{const module=device.createShaderModule({label:'MAIN attributed source relations '+entry,code});
   return {layout:'auto',compute:{module,entryPoint}};});
 }
 return plans;
}
export class SurfaceRelations {
  private readonly batchItems:number;
  lines:GPUBuffer;readonly status:GPUBuffer;
  private dispatch:GPUBuffer;
  /** Relation records reserved after the complete emitted-line capacity. The
   * level prefix is written here, never into binding 7 or an externally-owned
   * visibility list. */
  private levelPrefixRecords:number;
  /** Per-frame suffix cursor and lifetime lock. Once a direct source has
   * captured `lines`, replacing that buffer would make later consumers read a
   * different arena and lose the earlier source's complete relations. */
  private levelPrefixUsed=0;
  private levelPrefixWriting=false;
  /** Complete prefix demand observed in this frame, including sources that
   * had to fall back because the current suffix was already locked. It is
   * consumed by the next begin() before any writer binds the new arena. */
  private levelPrefixDemand=0;
  private readonly preparation:SurfaceRelationPreparation;
  private readonly pipelines:Record<string,GPUComputePipeline>={};
  private pending:GPUBuffer[]=[];
  private retired:GPUBuffer[]=[];
  private differentials?:GPUBuffer;
  private contactWork?:GPUBuffer;



  private readonly partAnchorSources=new Map<unknown,{source:unknown;vertices:Uint32Array;input:GPUBuffer;base:number;parts:number;instances:number;canonicalBase:number;canonicalSamples:number;canonicalStride:number}>();
  /** Per-frame live anchor cursors: [records, canonical words, overflow, selected]. */
  private readonly partAnchorAlloc:GPUBuffer;
  private partAnchorBuffer?:GPUBuffer;private partAnchorHeadSlots=0;private partAnchorRecords=0;private partAnchorRecordCapacity=0;
  private partAnchorCanonicalBase=0;private partAnchorCanonicalSamples=0;private partAnchorCanonicalCapacity=0;
  private partAnchorRowLayout?:SourcePartRowTable;
  /** Exact maximum demand observed from the producer, separate from the
   * rounded payload reserve used by the next articulation job. */
  private partAnchorRowRequired=0;private partAnchorRowCapacity=0;
  private partAnchorFrameCleared=false;private partAnchorGrowTo=0;private partAnchorGrowCanonicalTo=0;
  /** The owner-keyed extent table (undefined until a source registers). Consumers re-read it each frame. */
  get partAnchorTable():SourcePartAnchorTable|undefined{
    return this.partAnchorBuffer?{buffer:this.partAnchorBuffer,headSlots:this.partAnchorHeadSlots,headBase:2,recordBase:2+this.partAnchorHeadSlots,recordWords:PART_ANCHOR_RECORD_WORDS,canonicalBase:this.partAnchorCanonicalBase,canonicalRecords:this.partAnchorCanonicalSamples,canonicalWords:PART_ANCHOR_CANONICAL_WORDS,rowTable:this.partAnchorRowLayout}:undefined;
  }
  /** Allocate the prepared row arena alongside the existing owner-keyed
   * anchor arena. Its descriptor/bitset space is sized from the live source
   * records and page rows; its payload is sized only from the exact touched-row
   * demand returned by the native producer. A zero-demand first attempt is a
   * measured grow probe, not a permanent quota or incumbent replacement. */
  private ensurePartAnchorRowTable():void{
    // Candidate anchor records are NOT native source programmes. In
    // particular, a GPU voxel source can declare millions of legacy anchors
    // with no canonical passage. Carry only reserve demand here; articulation
    // owns the selected source-run domain and sizes its row metadata there.
    this.partAnchorRowLayout=this.partAnchorBuffer&&this.detailPage&&this.partAnchorCanonicalSamples>0?{touchedRows:this.partAnchorRowCapacity}:undefined;
  }
  /** Register (or keep) a batch's declared anchors from its residency source; returns the entry, or undefined. */
  private partAnchorsFor(batch:PosedSourceBatch,source:ResidentRelationSource):{input:GPUBuffer;parts:number;instances:number}|undefined{
    const vertices=(source as {partAnchorVertices?:Uint32Array}).partAnchorVertices;const key=posedBatchIdentity(batch);
    const instances=partAnchorInstanceCount(batch);
    // An empty prepared frontier is a valid no-draw result. Do not allocate a
    // source registration or canonical side table for a dispatch with no
    // selected ordinals; an older entry remains fenced by the existing expiry
    // path so an in-flight retry keeps its range valid.
    if(instances===0)return undefined;
    const held=this.partAnchorSources.get(key);

    // CONTENTS + demand. A batched producer may expose a shorter final slice
    // under the same identity; its existing larger range remains safe because
    // the GPU count check exits before visiting the unused records.
    const retainedDemand=held&&samePartAnchorVertices(held.vertices,vertices)&&
      (held.instances===instances||(batch.source.visibleInstanceFrontier===undefined&&held.instances>instances));
    if(retainedDemand){held.source=source;this.expiry.use(held);return held;}
    if(held){this.partAnchorSources.delete(key);this.retired.push(held.input);}
    const canonical=partAnchorCanonicalLayout(vertices);
    const parts=canonical?.parts??(vertices&&vertices.length>=6&&vertices.length%6===0?vertices.length/6:0);
    if(!vertices||parts===0)return undefined;
    const canonicalSamples=canonical?.sampleCount??0,canonicalStride=canonical?.instanceStride??0;
    if(!this.partAnchorBuffer)this.growPartAnchorTable(parts*instances,canonicalSamples*instances);
    // Canonical rows are compacted by the GPU live cursor. A source may have
    // more candidate samples than the current device-sized table can hold;
    // that is reported by partAnchorAlloc and retried at the same pose, not
    // by refusing this source during registration.
    const base=this.partAnchorRecords;this.partAnchorRecords+=parts*instances;this.partAnchorCanonicalSamples+=canonicalSamples*instances;
    // Four level-cache words plus one current-frame marker per source part
    // follow the immutable packet. The
    // chart-certify producer writes them only on a content-generation rebuild;
    // camera frames only read them, so the packet remains the source-owned
    // lifetime rather than a per-frame topology allocation.
    // Word 0 used to carry the unused candidate-prefix offset. Live records
    // are cursor-allocated; use that word for the SAME directory boundary in
    // canonical and legacy packets, without growing either packet/header ABI.
    const data=new Uint32Array(2+vertices.length+parts*5);data[0]=this.partAnchorCanonicalBase;data[1]=parts;data.set(vertices,2);
    if(canonical){
      const meta=2+canonical.meta,recordAt=2+canonical.recordAt,sampleAt=2+canonical.sampleAt;
      // The GPU allocator assigns compact rows after live owner/LOD tests;
      // The shared packet header owns the directory boundary; metadata keeps
      // only immutable source addresses and its instance stride.
      data[meta+2]=recordAt;data[meta+4]=sampleAt;data[meta+9]=canonicalStride;
      for(let part=0;part<parts;part++){const record=recordAt+part*TOPOLOGY_PART_CANONICAL_RECORD_WORDS;if(data[record+5]!>0)data[record+4]=data[record+4]!+2;}
    }
    const input=this.device.createBuffer({label:'MAIN source-declared part anchors',size:data.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
    this.device.queue.writeBuffer(input,0,data);
    const entry={source,vertices,input,base,parts,instances,canonicalBase:0,canonicalSamples,canonicalStride};this.partAnchorSources.set(key,entry);
    this.expiry.use(entry,()=>{if(this.partAnchorSources.get(key)!==entry)return;this.partAnchorSources.delete(key);this.retired.push(entry.input);});
    return entry;
  }
  /** A larger table restarts every registration (records are re-allocated lazily by later draws). */



  reservePartAnchors(batches:Iterable<{readonly batch:PosedSourceBatch;readonly source:ResidentRelationSource}>):void{
    let additional=0,canonicalAdditional=0;const seen=new Set<object>();
    for(const {batch,source}of batches){
      const key=posedBatchIdentity(batch);if(seen.has(key))continue;seen.add(key);
      const vertices=(source as {partAnchorVertices?:Uint32Array}).partAnchorVertices,canonical=partAnchorCanonicalLayout(vertices);const parts=canonical?.parts??(vertices&&vertices.length>=6&&vertices.length%6===0?vertices.length/6:0);if(!vertices||parts===0)continue;
      const instances=partAnchorInstanceCount(batch),held=this.partAnchorSources.get(key);
      if(instances===0)continue;
      if(held&&samePartAnchorVertices(held.vertices,vertices)&&
        (held.instances===instances||(batch.source.visibleInstanceFrontier===undefined&&held.instances>instances)))continue;
      additional+=parts*instances;canonicalAdditional+=(canonical?.sampleCount??0)*instances;
    }
    if(additional===0){this.ensurePartAnchorRowTable();return;}
    if(this.partAnchorBuffer){this.ensurePartAnchorRowTable();return;}
    // The first table is only a device-sized arena. Candidate parts do not
    // reserve one record each; the projection cursor measures selected demand
    // and schedules an exact same-pose grow/replay if this arena is too small.
    this.growPartAnchorTable(Math.max(1,additional),Math.max(0,canonicalAdditional));
    this.ensurePartAnchorRowTable();
  }
  private livePartAnchorRecords():number{
    let records=0;for(const entry of this.partAnchorSources.values())records+=entry.parts*entry.instances;return records;
  }
  private livePartAnchorCanonicalSamples():number{
    let samples=0;for(const entry of this.partAnchorSources.values())samples+=entry.canonicalSamples*entry.instances;return samples;
  }
  private growPartAnchorTable(records:number,canonicalSamples=0):void{
    const maximum=Math.min(this.device.limits?.maxBufferSize??Infinity,this.device.limits?.maxStorageBufferBindingSize??Infinity);
    const powerOfTwoAtLeast=(value:number,min:number)=>{let result=Math.max(1,min);while(result<value)result*=2;return result;};
    // Geometric slack is useful for ordinary churn, but it is not allowed to
    // turn a complete source demand into a device-limit failure. The record
    // and canonical extents are independent: only the bucket heads need a
    // power-of-two extent. Prefer the incumbent 2x growth when it fits, then
    // retry with the exact complete demand (not a cap or a dropped producer).
    const exactRecords=Math.max(1,records),exactCanonical=Math.max(0,canonicalSamples);
    const exactHeads=powerOfTwoAtLeast(exactRecords,1024);
    const preferredRecords=powerOfTwoAtLeast(exactRecords*2,1024),preferredCanonical=exactCanonical===0?0:powerOfTwoAtLeast(exactCanonical*2,1024),preferredHeads=powerOfTwoAtLeast(preferredRecords,1024);
    const preferredBytes=partAnchorTableBytes(preferredHeads,preferredRecords,preferredCanonical);
    const usePreferred=preferredBytes<=maximum;
    let capacity=usePreferred?preferredRecords:exactRecords;
    let heads=usePreferred?preferredHeads:exactHeads;
    const maximumWords=Math.floor(maximum/4);
    // Initial registration may describe every producer candidate, while the
    // live camera selects only a subset. Never let that candidate census make
    // the first table impossible: reserve a device-sized arena and let the
    // completed cursor demand drive an exact same-pose repartition.
    if(2+heads+capacity*PART_ANCHOR_RECORD_WORDS>maximumWords){
      heads=1024;capacity=Math.floor((maximumWords-2-heads)/PART_ANCHOR_RECORD_WORDS);
      if(capacity<1)throw Error(`Owner-keyed part-anchor record arena cannot fit device binding capacity ${maximum} bytes`);
    }
    const canonicalBase=2+heads+capacity*PART_ANCHOR_RECORD_WORDS;
    // The canonical cursor consumes only live projected rows. Keep the
    // complete record directory, then use every remaining device-sized word
    // for canonical rows; an over-demand is surfaced by the GPU overflow
    // counter for same-pose replay rather than failing before first text.
    const availableCanonical=Math.max(0,Math.floor((maximumWords-canonicalBase)/PART_ANCHOR_CANONICAL_WORDS));
    const canonicalCapacity=Math.min(exactCanonical,availableCanonical);
    const bytes=partAnchorTableBytes(heads,capacity,canonicalCapacity);
    if(bytes>maximum)throw Error(`Owner-keyed part anchor records (${bytes} bytes; ${records} complete records) exceed device binding capacity ${maximum} bytes`);
    // Allocate and initialize the replacement before retiring the current
    // table or clearing its registrations. A device-limit/create/write failure
    // must leave the old table usable for the existing frame and retry, rather
    // than multiplying the requested size against an already-cleared cache.
    const previous=this.partAnchorBuffer,previousInputs=[...this.partAnchorSources.values()];
    let replacement:GPUBuffer|undefined;
    try{
      replacement=this.device.createBuffer({label:'MAIN owner-keyed part anchor extents and source passages',size:bytes,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
      this.device.queue.writeBuffer(replacement,0,new Uint32Array([heads,2+heads]));
    }catch(error){replacement?.destroy();throw error;}
    if(previous)this.retired.push(previous);
    for(const entry of previousInputs)this.retired.push(entry.input);
    this.partAnchorSources.clear();this.partAnchorRecords=0;this.partAnchorCanonicalSamples=0;
    this.partAnchorRecordCapacity=capacity;this.partAnchorCanonicalCapacity=canonicalCapacity;this.partAnchorHeadSlots=heads;this.partAnchorCanonicalBase=canonicalBase;
    this.partAnchorBuffer=replacement;this.partAnchorRowLayout=undefined;this.partAnchorFrameCleared=false;
  }
  private readonly homes=new Map<unknown,ColumnHome>();private homeScratch=0;private homeWords=0;
  /** Only a certified native owner authority or the legacy physical producer
   * row may compact a shared instance frontier. Custom/GPU-indexed and
   * multi-owner sources retain their complete incumbent list. */
  private readonly admittedFrontiers=new Set<object>();
  /** Per-frame native compact lists. They are produced before contacts and
   * reused by every same-pose consumer; the retained binding owns their
   * submission lifetime. */
  private readonly frontierSelections=new Map<object,GPUBuffer>();
  private intersections?:SurfaceIntersections;
  private contactsActive=false;
  private detailField?:SurfaceField;
  private detailPage?:readonly [number,number];
  private emptyDetail?:GPUTexture;
  private emptyDetailView?:GPUTextureView;
  private emptyPartAnchorVertices?:GPUBuffer;



  fieldDomains?:ReadonlySet<number>;
  /** The form table the line stages read (binding 10); a stable empty
   * view when this frame has none (flag bit 0 unset, never read). */
  private lineDetailView(earlyDetail:boolean):GPUTextureView{
    if(earlyDetail)return formDepths(this.device).view(this.detailField!);
    return this.emptyDetailView??=(this.emptyDetail??=this.device.createTexture({label:'MAIN unselected detail table',size:[1,1],format:'rgba32uint',usage:GPUTextureUsage.TEXTURE_BINDING})).createView();
  }
  private readonly retainedDifferentials=new Map<object,Map<number,{revision:number;buffer:GPUBuffer}>>();
 static async create(device:GPUDevice,lineCapacity=32768,interior=false,progress?:(fraction:number)=>void,preparation:SurfaceRelationPreparation='generic'):Promise<SurfaceRelations>{
   const plans=surfaceRelationPipelinePlans(device,interior,preparation),pipelines=await prepareComputePipelines(device,plans,progress);
   return new SurfaceRelations(device,lineCapacity,interior,{plans,pipelines,preparation});
  }
  constructor(private readonly device:GPUDevice,public lineCapacity=32768,private readonly interior=false,prepared?:PreparedSurfaceRelations){
    // Start with one reserved Relation record and grow the suffix from the
    // admitted instance count. Reserving four words per line-capacity entry
    // would turn a large relation arena into an unnecessary second arena before
    // a levelled source is ever submitted.
    this.levelPrefixRecords=Math.min(1,this.levelPrefixBudget());
    this.lines=device.createBuffer({size:(lineCapacity+this.levelPrefixRecords)*RELATION_WORDS*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
    this.status=device.createBuffer({size:16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
    // prepare publishes relation/contact arguments through word 35 (the
    // compact frontier stages), while prepareInterior reuses words 6..17.
    // Binding 7 remains the ordinary 36-word indirect-argument arena. Level
    // records do not live here; they live in the reserved Relation tail.
    this.dispatch=device.createBuffer({size:144,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.INDIRECT});
    this.partAnchorAlloc=device.createBuffer({label:'MAIN live part-anchor allocation cursors',size:16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
    this.emptyPartAnchorVertices=device.createBuffer({label:'MAIN empty source part anchors',size:8,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
    this.device.queue.writeBuffer(this.emptyPartAnchorVertices,0,new Uint32Array([0,0]));
    this.preparation=prepared?.preparation??'generic';this.batchItems=relationBatchItems(device);
    const plans=prepared?.plans??surfaceRelationPipelinePlans(device,interior,this.preparation),pipelines=prepared?.pipelines??compileComputePipelines(device,plans);
    for(const entryPoint of plans.keys())this.pipelines[entryPoint]=pipelines.get(entryPoint)!;
  }
  /** Grow storage for an already counted, retained source submission. This
   * changes allocation only: it never limits source owners or source ranges.
   * The caller owns the submitted-frame fence before replaying its SAME pose. */
  ensureCapacity(count:number):void{
    if(!Number.isSafeInteger(count)||count<0)throw Error('Invalid complete source relation count');
    if(count<=this.lineCapacity)return;
    const capacity=Math.ceil(count/256)*256,maximum=Math.floor(Math.min(this.device.limits.maxBufferSize,this.device.limits.maxStorageBufferBindingSize)/(RELATION_WORDS*4));
    if(capacity>maximum)throw Error(`Complete source relation range (${count}) needs device-sized chunks`);
    const prefix=Math.min(this.levelPrefixRecords,maximum-capacity),size=(capacity+prefix)*RELATION_WORDS*4;
    const next=this.device.createBuffer({label:'MAIN complete counted source relations',size,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
    this.retired.push(this.lines);this.lines=next;this.lineCapacity=capacity;this.levelPrefixRecords=prefix;
  }
  private levelPrefixBudget():number{return Math.max(0,Math.floor(Math.min(this.device.limits.maxBufferSize,this.device.limits.maxStorageBufferBindingSize)/(RELATION_WORDS*4))-this.lineCapacity);}
  private ensureLevelPrefixRecords(records:number):void{
    if(!Number.isSafeInteger(records)||records<1)throw Error('Invalid complete level prefix record count');
    if(records<=this.levelPrefixRecords)return;
    if(this.levelPrefixWriting)throw Error('Cannot grow level prefix after relation writers are bound');
    const size=(this.lineCapacity+records)*RELATION_WORDS*4;
    if(size>Math.min(this.device.limits.maxBufferSize,this.device.limits.maxStorageBufferBindingSize))throw Error(`Complete level prefix (${records} records) exceeds relation arena capacity`);
    const next=this.device.createBuffer({label:'MAIN complete source level prefix relation arena',size,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
    this.retired.push(this.lines);this.lines=next;this.levelPrefixRecords=records;
  }
  private reserveLevelPrefix(instances:number):LevelPrefixReservation|undefined{
    const records=levelPrefixRecordCount(instances);
    const budget=this.levelPrefixBudget();if(records>budget)return undefined;
    this.levelPrefixDemand=Math.min(budget,this.levelPrefixDemand+records);
    const reservation=levelPrefixReservation(this.lineCapacity,this.levelPrefixRecords,this.levelPrefixUsed,instances,this.levelPrefixWriting);
    if(!reservation)return undefined;
    if(reservation.nextCapacity>this.levelPrefixRecords)this.ensureLevelPrefixRecords(reservation.nextCapacity);
    this.levelPrefixUsed=reservation.nextUsed;this.levelPrefixWriting=true;
    return reservation;
  }
  private noteLevelPrefixDemand(instances:number):number{
    const records=levelPrefixRecordCount(instances),budget=this.levelPrefixBudget();if(records>budget)return 0;
    this.levelPrefixDemand=Math.min(budget,this.levelPrefixDemand+records);return records;
  }
  /** `contactScale`: passed to SurfaceIntersections.begin (id raster columns per field column). */
  begin(encoder:GPUCommandEncoder,view?:SurfaceGraphView,field?:SurfaceField,contactScale=1):void{
    // A previous frame may have recorded requests that fell back after its
    // suffix was locked. Grow here, before any current-frame writer can bind
    // `lines`; this makes fallback temporary and preserves the full incumbent
    // output while the demand is learned.
    this.levelPrefixUsed=0;this.levelPrefixWriting=false;
    const wanted=Math.min(this.levelPrefixDemand,this.levelPrefixBudget());
    if(wanted>this.levelPrefixRecords)this.ensureLevelPrefixRecords(wanted);
    this.levelPrefixDemand=0;
    encoder.clearBuffer(this.status);encoder.clearBuffer(this.partAnchorAlloc);this.admittedFrontiers.clear();this.frontierSelections.clear();this.contactsActive=!!view&&!!field;
    this.detailField=field;this.detailPage=view?.pageExtent;this.fieldDomains=undefined;
    if(view&&field)(this.intersections??=new SurfaceIntersections(this.device)).begin(encoder,view,field,this.status,contactScale);

    // and dispatched together in finish(), before the contact-triangle finish.
    this.queue=this.interior?{items:[],clears:[],seen:new Set(),contactWords:0,dispatchWords:0,levelPrefixRecords:0}:undefined;
  }
  ensureContactCapacity(count:number):void{(this.intersections??=new SurfaceIntersections(this.device)).ensureCapacity(count);}
  finish(encoder:GPUCommandEncoder):PartAnchorDemandReadback{
    this.flushBatched(encoder);if(this.contactsActive)this.intersections!.finish(encoder,this.lines,this.status,this.lineCapacity);this.contactsActive=false;
    const readback=this.device.createBuffer({label:'MAIN live part-anchor demand readback',size:16,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
    encoder.copyBufferToBuffer(this.partAnchorAlloc,0,readback,0,16);
    let consumed=false;
    return {
      async collect():Promise<PartAnchorDemand>{
        if(consumed)throw Error('Part-anchor demand readback was already consumed');consumed=true;
        await readback.mapAsync(GPUMapMode.READ);
        const words=new Uint32Array(readback.getMappedRange()).slice();readback.unmap();readback.destroy();
        return {records:words[0]!,canonicalRows:words[1]!,overflow:words[2]!==0,selected:words[3]!};
      },
      dispose():void{if(consumed)return;consumed=true;if(readback.mapState==='mapped')readback.unmap();readback.destroy();},
    };
  }
  /** Grow only after a completed same-pose demand readback. If the complete
   * selected demand cannot fit the device binding, fail explicitly rather than
   * publishing a table with missing owners or canonical rows. */
  ensurePartAnchorDemand(demand:PartAnchorDemand):void{
    if(!demand.overflow)return;
    const maximum=Math.min(this.device.limits.maxBufferSize,this.device.limits.maxStorageBufferBindingSize);
    const heads=Math.max(1024,2**Math.ceil(Math.log2(Math.max(1,demand.records))));
    const bytes=partAnchorTableBytes(heads,Math.max(1,demand.records),demand.canonicalRows);
    if(bytes>maximum)throw Error(`Complete live part-anchor demand (${bytes} bytes; ${demand.records} records, ${demand.canonicalRows} canonical rows) exceeds device binding capacity ${maximum} bytes`);
    if(demand.records>this.partAnchorRecordCapacity||demand.canonicalRows>this.partAnchorCanonicalCapacity){
      this.growPartAnchorTable(Math.max(1,demand.records),demand.canonicalRows);
    }
  }
  /** Grow the counts-arena row payload from the exact source-group demand
   * reported by the articulation header.  This is metadata only; the next
   * same-pose articulation job allocates the bytes in its existing scratch. */
  ensurePartAnchorRowDemand(rows:number):void{
    if(!Number.isSafeInteger(rows)||rows<0)throw Error('Invalid complete native row demand');
    if(rows>this.partAnchorRowRequired)this.partAnchorRowRequired=rows;
    if(rows>this.partAnchorRowCapacity){
      const maximum=Math.min(this.device.limits.maxBufferSize,this.device.limits.maxStorageBufferBindingSize);
      this.partAnchorRowCapacity=Math.max(this.partAnchorRowCapacity,partAnchorRowCapacity(rows,0,0,1,maximum));
      this.ensurePartAnchorRowTable();
    }
  }
  /** Current rounded payload reserve and exact source demand are observable so
   * callers can distinguish a growth hint from a structural source failure. */
  partAnchorRowCapacityWords():number{return this.partAnchorRowCapacity;}
  partAnchorRowRequiredWords():number{return this.partAnchorRowRequired;}
  /** The contact producer's passes for one source (append does this itself
   * when it opens its own pass). */
  appendContacts(encoder:GPUCommandEncoder,batch:PosedSourceBatch,source:ResidentRelationSource,domain:number):void{
    if(this.contactsActive)this.intersections!.append(encoder,batch,source,domain,this.admittedFrontiers.has(batch),this.frontierSelections.get(batch));
  }
  /** Every source's contact passes as two passes (surface-intersections appendAll). */

  appendContactsAll(encoder:GPUCommandEncoder,items:readonly {batch:PosedSourceBatch;source:ResidentRelationSource;domain:number;extent?:SourceDrawExtent}[]):void{
    if(this.contactsActive){this.intersections!.fieldDomains=this.fieldDomains;this.intersections!.page=this.detailPage;this.intersections!.appendAll(encoder,items.map(item=>({...item,frontierAdmitted:this.admittedFrontiers.has(item.batch),selection:this.frontierSelections.get(item.batch)})));}
  }
  /** Admit the final GPU parent frontier before contact packing or any source
   * relation stage. The octree/source selector supplies candidates, while the
   * same-pose form table is the authority that decides whether a complete
   * parent group actually replaced its children. The compacted list is reused
   * by every later consumer through its existing count word. */
  admitFrontiers(encoder:GPUCommandEncoder,items:readonly {batch:PosedSourceBatch;source:ResidentRelationSource;domain:number}[]):void{
    if(!this.detailField||!this.detailPage)return;
    const candidates=items.filter(item=>{
      if(isGpuIndexedResident(item.source))return false;
      const batch=item.batch,source=batch.source,native=source.nativeVisibleInstanceFrontier,authority=source.ownerAuthority;
      const nativeOwner=native?.kind==='native-indirect'&&!!authority&&((authority.kind==='producer-row'&&native.owner==='producer-row')||(authority.kind==='standard-entity'&&native.owner==='entity'));
      const physical=!!source.visibleInstanceFrontier&&!!source.visibleInstances&&/\/producer-row\//.test(item.source.id);
      return nativeOwner||physical;
    });
    if(!candidates.length)return;
    const pass=encoder.beginComputePass({label:'MAIN admitted source representation frontiers'});
    try{
      for(const {batch,source,domain}of candidates){
        const native=batch.source.nativeVisibleInstanceFrontier;
        const frontier=batch.source.visibleInstanceFrontier;
        this.admittedFrontiers.add(batch);
        const entry=this.bindings(source,batch),indexed=isGpuIndexedResident(source)?source.gpu:undefined;
        const floats=new Float32Array(56),u=new Uint32Array(floats.buffer);
        u.set([this.lineCapacity,domain,batch.source.vertexCapacity,batch.source.indexBase??0],24);
        u.set([0,0,1,Number(batch.source.posedRevision!==undefined)],28);
        u[36]=indexed?Number(indexed.constructionKind==='planar'):0;
        u[38]=Number(!!this.detailField&&!!this.detailPage&&this.detailPage[0]>0&&this.detailPage[1]>0);
        const nativeOwner=native?.kind==='native-indirect'&&!!batch.source.ownerAuthority&&((batch.source.ownerAuthority.kind==='producer-row'&&native.owner==='producer-row')||(batch.source.ownerAuthority.kind==='standard-entity'&&native.owner==='entity'));
        let frontierBuffers:RelationBinding[];
        if(nativeOwner){
          // Keep the admitted owner list at its existing base (word 1). The
          // compact count is written at the producer's indirect word offset;
          // the level prefix is no longer stored in this owner buffer.
          const capacity=nativeTransportInstanceCapacity(batch),countWord=Math.floor(native!.indirectOffset/4),words=Math.max(2,capacity+1,countWord+1),bytes=words*4;
          if(!entry.selection||entry.ownerKey!=='native-frontier'||entry.selection.size<bytes){
            if(entry.selection)this.retired.push(entry.selection);
            entry.selection=this.device.createBuffer({label:'MAIN native admitted instance frontier',size:bytes,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
            entry.ownerKey='native-frontier';
          }
          u[42]=capacity;u[43]=1;u[44]=1;u[45]=64|((batch.source.ownerAuthority!.kind==='producer-row')?128:0);u[46]=0;u[47]=this.device.limits.maxComputeWorkgroupsPerDimension;u[48]=Math.floor(native!.indirectOffset/4);u[49]=0;u[50]=0;u[51]=1;
          this.frontierSelections.set(batch,entry.selection);
          frontierBuffers=[entry.selection,native!.members,native!.indirect];
        }else if(frontier){
          u[42]=frontier.indices.length;u[43]=frontier.offset;u[44]=1;u[46]=batch.source.visibleInstanceOffset??0;u[47]=this.device.limits.maxComputeWorkgroupsPerDimension;
          u[48]=frontier.offset-1;u[51]=1;
          // The legacy branch never reads nativeMembers/nativeIndirect, but
          // auto-layout still requires their resources. Do not alias the
          // writable compact frontier (binding 8) as a read-only member buffer
          // (binding 20) in the same dispatch; metadata is read-only throughout.
          frontierBuffers=[batch.source.visibleInstances!,batch.metadata,batch.metadata];
        }else continue;
        // Admission and the later relation pass are encoded before one submit;
        // a queue write to the retained relation uniform would otherwise
        // overwrite the admission parameters before its dispatch executes.
        const admissionParams=this.device.createBuffer({label:'MAIN native frontier admission parameters',size:224,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
        this.pending.push(admissionParams);this.device.queue.writeBuffer(admissionParams,0,floats);
        const buffers=[admissionParams,indexed?.incidence.binding??source.bufferBinding??source.buffer,batch.storage?.vertices??batch.vertices,batch.storage?.attributes??batch.attributes,batch.storage?.metadata??batch.metadata,this.lines,this.status,this.dispatch,frontierBuffers[0]] as RelationBinding[];
        buffers[10]={view:this.lineDetailView(u[38]!==0)};
        buffers[20]=frontierBuffers[1];buffers[21]=frontierBuffers[2];
        const pipeline=this.pipelines.admitFrontier!;
        pass.setPipeline(pipeline);pass.setBindGroup(0,this.group(entry.groups,'admitFrontier',pipeline,[0,1,2,3,4,8,10,20,21],buffers));pass.dispatchWorkgroups(1);
      }
    }finally{pass.end();}
  }
  /** `shared`: a compute pass the caller opened for EVERY source of the
   * frame (and ends itself), after calling appendContacts for each source.
   * ONE PASS PER FRAME (frame cost, 2026-11-11): a pass per source was about
   * 455 Metal command encoders per Y8 text frame (with the contact passes
   * between them, ~1375); the GPU idles at every encoder boundary, and a
   * frame took ~178 ms from submit to completion while its timed passes
   * summed to ~16 ms. Dispatches of one pass run in order and each sees the
   * previous one's writes, so the per-source stage order is unchanged. */
  /** `extent`: the batch's static draw extent (physical batches). With it the
   * one-thread prepare and prepareInterior dispatches are not encoded: the
   * host sizes each stage's direct dispatch from the same counts they read
   * (HOST-SIZED STAGES below). */
  append(encoder:GPUCommandEncoder,batch:PosedSourceBatch,source:ResidentRelationSource,view:SurfaceGraphView,domain:number,selectedOwners?:readonly number[],shared?:GPUComputePassEncoder,extent?:SourceDrawExtent):void{
    if(!shared)this.appendContacts(encoder,batch,source,domain);
    const frontierAdmitted=this.admittedFrontiers.has(batch),nativeFrontier=batch.source.nativeVisibleInstanceFrontier;
    const nativeAdmitted=frontierAdmitted&&!!nativeFrontier;
    // Native packed membership is already the engine's live owner list. Do
    // not rebuild/sort a CPU owner census for that path; the GPU compact list
    // below is the sole membership input.
    const owners=nativeAdmitted?new Uint32Array(0):Uint32Array.from([...new Set((selectedOwners??[]).map(owner=>owner>>>0))].sort((a,b)=>a-b));
    // RETAINED PER-SOURCE BINDINGS (frame cost, 2026-11-11): the parameter
    // block, owner selection and bind groups of a source are kept between
    // frames; only the 208 parameter bytes are rewritten. Rebuilding them was
    // ~4300 bind groups and ~1900 buffers per Y8 text frame, each validated in
    // the GPU process, which held a frame ~130 ms from submit to completion
    // while the GPU itself was busy 29% of the time.

    // GPU instance visibility and a retained posed tensor is queued; finish()
    // dispatches every queued source's stage at once. A second use of the same
    // batch in one frame keeps the per-source path (its tensor would be shared).
    const batched=!!shared&&!!extent&&this.interior&&!!this.queue&&!!batch.source.visibleInstances&&batch.source.posedRevision!==undefined&&!this.queue.seen.has(posedBatchIdentity(batch));
    if((this.preparation==='generic-direct'||this.preparation==='generic-direct-ordinary')&&extent)throw Error('Generic-direct relation preparation does not accept a host extent');
    if(this.preparation==='physical-batched'&&(isGpuIndexedResident(source)||!batched))throw Error('Physical-batched relation preparation requires a non-GPU-indexed host batch with extent, visibility frontier and posed revision');
    const entry=batched?undefined:this.bindings(source,batch);
    const ownerKey=nativeAdmitted?'native-frontier':selectedOwners===undefined?'':owners.join(',');
    let selection=batch.source.visibleInstances??(nativeAdmitted?this.frontierSelections.get(batch):undefined);
    if(!selection&&entry){
      // The level prefix now lives in the relation arena. This buffer carries
      // only the producer-owned owner selection list used by selected().
      const ownerWords=Math.max(1,owners.length);
      if(!entry.selection||entry.ownerKey!==ownerKey||entry.selection.size<ownerWords*4){if(entry.selection)this.retired.push(entry.selection);entry.selection=this.device.createBuffer({size:ownerWords*4,usage:GPUBufferUsage.STORAGE,mappedAtCreation:true});new Uint32Array(entry.selection.getMappedRange()).set(owners);entry.selection.unmap();entry.ownerKey=ownerKey;}
      selection=entry.selection;
    }
    const floats=new Float32Array(56);floats.set(view.clipFromWorld);floats.set([...view.eye.slice(0,3),Number(view.perspective)],16);floats.set([...view.direction.slice(0,3),batch.source.cullMode??0],20);
    new Uint32Array(floats.buffer).set([this.lineCapacity,domain,batch.source.vertexCapacity,batch.source.indexBase??0],24);new Uint32Array(floats.buffer).set(nativeAdmitted?[0,0,0,Number(batch.source.posedRevision!==undefined)]:[owners.length,Number(selectedOwners!==undefined),Number(!!batch.source.visibleInstances),Number(batch.source.posedRevision!==undefined)],28);
    new Uint32Array(floats.buffer)[46]=batch.source.visibleInstances?(batch.source.visibleInstanceOffset??0):0;new Uint32Array(floats.buffer)[47]=this.device.limits.maxComputeWorkgroupsPerDimension;
    new Uint32Array(floats.buffer)[36]=isGpuIndexedResident(source)?Number(source.gpu.constructionKind==='planar'):planarRest(source.geometry)?1:0;
    const anchorParts=partAnchorPartCount((source as {partAnchorVertices?:Uint32Array}).partAnchorVertices);
    const page=this.detailPage,field=this.detailField,frontier=batch.source.visibleInstanceFrontier;
    // This is the existing producer-issued source owner certificate. It is
    // deliberately separate from the native frontier bit: standard entity
    // sources can be direct, while producer rows may use either native or the
    // retained physical frontier transport.
    const levelOwnerCertificate=batch.source.physicalUniformProducerRows?2:batch.source.ownerAuthority?.kind==='standard-entity'?1:batch.source.ownerAuthority?.kind==='producer-row'?2:0;
    const terrain=isGpuIndexedResident(source)?source.gpu.constructionKind==='terrain':source.geometry.surface==='heightfield'||('analysis' in source&&(source.analysis as {type?:string}).type==='terrain');
    const earlyDetail=!!field&&!!page&&page[0]>0&&page[1]>0;
    if(earlyDetail){floats[37]=17*field!.height/page![1];new Uint32Array(floats.buffer)[39]=field!.width;new Uint32Array(floats.buffer)[40]=field!.height;}
    new Uint32Array(floats.buffer)[38]=Number(earlyDetail)|(Number(terrain)<<1)|(Number(tensorRevision(batch,frontierAdmitted)!==undefined)<<2)|(Number(earlyDetail&&!terrain)<<3)|(Number(earlyDetail&&!!this.fieldDomains?.has(domain))<<4)|(Number(!isGpuIndexedResident(source)&&(batch.source.physicalUniformProducerRows||sourceMaterialIsAffine(batch.source.material)))<<5);
    // R48's prefix is written by prepareIn before boundaries. Owned selection
    // buffers use their suffix; external visibility lists use the existing
    // indirect dispatch storage and retain their producer-owned list intact.
    const levelCapacity=nativeAdmitted?nativeTransportInstanceCapacity(batch):Math.max(1,frontier?.indices.length??batch.source.instanceCapacity??owners.length,owners.length);
    const earlyLevels=earlyDetail&&levelOwnerCertificate!==0&&supportsCompiledInstanceLevels(source);
    const levelReservation=!batched&&earlyLevels?this.reserveLevelPrefix(levelCapacity):undefined;
    if(!batched)this.levelPrefixWriting=true;
    // Every complete prefix lives in the relation output arena. Its base is a
    // relation-record index, not a word offset into the producer's visibility
    // list or the indirect argument buffer.
    const levelMode=levelReservation?1:0;
    const levelBase=levelReservation?.base??0;
    {const u=new Uint32Array(floats.buffer);u[52]=levelBase;u[53]=levelMode;u[54]=levelCapacity;u[55]=levelOwnerCertificate;}
    new Uint32Array(floats.buffer)[50]=anchorParts;
    if(frontier){const u=new Uint32Array(floats.buffer);u[42]=frontier.indices.length;u[43]=frontier.offset;u[44]=1;u[48]=frontier.offset-1;u[49]=0;u[50]=anchorParts;u[51]=Number(frontierAdmitted);}
    if(nativeAdmitted){
      const u=new Uint32Array(floats.buffer),capacity=nativeTransportInstanceCapacity(batch);
      u[42]=capacity;u[43]=1;u[44]=1;u[45]=64|((batch.source.ownerAuthority?.kind==='producer-row')?128:0);u[46]=0;u[47]=this.device.limits.maxComputeWorkgroupsPerDimension;u[48]=Math.floor(nativeFrontier!.indirectOffset/4);u[49]=0;u[50]=anchorParts;u[51]=1;
    }
    const indexed=isGpuIndexedResident(source)?source.gpu:undefined;
    const posedContacts=indexed?.edgeContactClass==='posed-coincident'&&indexed.topology.indexPattern==='independent-quads'&&batch.source.instanceCapacity===1;
    // Compatibility callers also own posed contacts/charts. Only the
    // explicitly fine-only profile disables them; specialized voxel entries
    // belong to generic-direct, not to the compatibility map.
    const producerCapable=this.preparation!=='generic-direct-ordinary'&&this.preparation!=='physical-batched';
    const voxelSpecialized=this.preparation==='generic-direct';
    const fineOnlyOrdinary=this.preparation==='generic-direct-ordinary';
    const boundaryStage=voxelSpecialized&&posedContacts?'boundariesVoxel':'boundaries';
    const contourStage=voxelSpecialized&&posedContacts?'contoursVoxel':'contours';
    const contactEdges=posedContacts?sourceResidencyEdgeCount(source):0;
    const parts=producerCapable&&posedContacts&&earlyDetail?indexed?.parts:undefined;
    let contactSlots=0,contactWords=1,baselineContactWords=1,residualWords=0,residualEnabled=false;
    let allowPersistentHome=true;
    let plannedHomeLevels:readonly number[]|undefined;
    let plannedHomeKey:unknown;
    let plannedHomeWords=0;

    // rest-cell chart of THIS generation), when its dense tables are bounded. Never both: parts win.
    const surfaceChart=producerCapable&&posedContacts&&earlyDetail&&!parts?currentRestCellChart(indexed):undefined;
    const chart=producerCapable&&posedContacts?(parts?cellChart(parts.extent):surfaceChart?cellChart(surfaceChart.extent):undefined):undefined;
    const chartCodes:SourceStorageBinding|undefined=parts?.cellCodes??(chart?surfaceChart?.vertexCellCodes:undefined);
    const occupancy=chart?chartOccupancy(indexed,(parts?parts.extent:surfaceChart!.extent) as readonly [number,number,number]):undefined;
    new Uint32Array(floats.buffer)[31]=chartToken(batch,source,!!chart);
    if(posedContacts){
      contactSlots=2**Math.ceil(Math.log2(Math.max(2,contactEdges*2,sourceResidencyFaceCount(source)*4)));


      baselineContactWords=(parts?contactSlots*7+contactEdges*2:contactSlots*3+contactEdges)+contactSlots+4+(chart?chart.cells*19+chart.corners*4+4+chart.clusters*4+sourceResidencyFaceCount(source)*2+2+chart.clusters+4+chartRequirementWords((parts?parts.extent:surfaceChart!.extent) as readonly [number,number,number]):4);
      residualWords=chart?4+chart.clusters+Math.ceil(sourceResidencyVertexCount(source)/32)+chart.clusters+Math.ceil(sourceResidencyFaceCount(source)/32)+chart.clusters:0;
      const residualProducerEnabled=!!chart&&(!!parts||!!surfaceChart);
      const maximumBytes=Math.min(this.device.limits.maxBufferSize,this.device.limits.maxStorageBufferBindingSize);
      const homeRequested=!!chart&&chartToken(batch,source,true)>1;
      const homeLevels=homeRequested?chartLevelClusters((parts?parts.extent:surfaceChart!.extent) as readonly [number,number,number]):undefined;
      const homeKey=homeRequested?posedBatchIdentity(batch):undefined;
      const heldHome=homeRequested?this.homes.get(homeKey):undefined;
      const sameHomeLayout=!!heldHome&&heldHome.anchorParts===anchorParts&&heldHome.levels.length===homeLevels!.length&&heldHome.levels.every((clusters,i)=>clusters===homeLevels![i]);
      plannedHomeLevels=homeLevels;plannedHomeKey=homeKey;plannedHomeWords=homeLevels?chartHomeWords(homeLevels):0;
      const homePlan=(words:number):ContactArenaHomePlan=>{
        const reset=words>this.homeScratch;let scratchWords=reset?Math.ceil(words*1.25):this.homeScratch;let homeWords=reset?0:this.homeWords;let homes=reset?0:this.homes.size;
        if(homeRequested&&(reset||!sameHomeLayout)){if(homeWords+plannedHomeWords>COLUMN_HOME_WORDS){homeWords=0;homes=0;}homeWords+=plannedHomeWords;homes=1;}
        return {retainHome:homes>0,scratchWords,homeWords};
      };
      const fullHomePlan=homePlan(baselineContactWords+(residualProducerEnabled?residualWords:0));
      const baselineHomePlan=homePlan(baselineContactWords);
      const transientHomePlan={retainHome:false,scratchWords:0,homeWords:0};
      const arena=chooseOptionalContactArenaWithHome(baselineContactWords,residualProducerEnabled?residualWords:0,maximumBytes,fullHomePlan,baselineHomePlan,transientHomePlan);
      contactWords=arena.words;residualEnabled=residualProducerEnabled&&arena.enabled;allowPersistentHome=arena.plan.retainHome;
      const words=new Uint32Array(floats.buffer);words[41]=contactSlots;words[45]=1;
      if(parts){const [x,y,z]=parts.extent;words[45]=3|(chart?4:0)|((x-1)<<8)|((y-1)<<16)|((z-1)<<24);}
      // Surface: the chart (4) with explicit surface semantics (8); NEVER the closed-cell part bit (2).
      else if(chart&&surfaceChart){const [x,y,z]=surfaceChart.extent;words[45]=1|4|8|((x-1)<<8)|((y-1)<<16)|((z-1)<<24);}

      if(chart&&currentRestCellChart(indexed)?.cornerOrder===VOXEL_REST_CORNER_ORDER)words[45]|=16;

      if(occupancy)words[45]|=32;
      if(residualEnabled)words[45]|=64;
    }
    if(levelOwnerCertificate===2)new Uint32Array(floats.buffer)[45]|=128;
    if(batched){this.queueBatched(batch,source,new Uint32Array(floats.buffer),extent!,{posedContacts,contactWords,baselineContactWords,residualWords,contactEdges,parts,chartCodes,occupancy,chartClusters:chart?.clusters??0,anchorParts,earlyDetail,terrain,selection:selection!,frontierAdmitted});return;}
    const held=entry!,params=held.params;

    // grow-only) | homes]. A source whose producer certifies its posed content
    // (posedContentToken > 1) gets a home sized from its column extent: the
    // closed-cell extent at the coarsest-but-one block (2), or one column per
    // 4 quads; twice that in slots. A full table is an overflow (the chunk
    // keeps its fine producer). Growing the scratch or the buffer, or a changed
    // source, restarts the homes (a new header, an invalid token).
    let home:ColumnHome|undefined;
    if(!allowPersistentHome&&(this.homes.size>0||this.homeScratch>0)){
      if(this.contactWork)this.retired.push(this.contactWork);
      this.contactWork=undefined;this.homes.clear();this.homeWords=0;this.homeScratch=0;
    }
    if(allowPersistentHome&&posedContacts&&contactWords>this.homeScratch){this.homeScratch=Math.ceil(contactWords*1.25);this.homes.clear();this.homeWords=0;}
    if(allowPersistentHome&&chart&&chartToken(batch,source,true)>1){

      // home.offset, each sized for its own level and retained under its own (generation token, block)
      // header. A level transition selects that level's retained home instead of rebuilding one shared home.
      const slots=chart.clusters,levels=plannedHomeLevels??chartLevelClusters((parts?parts.extent:surfaceChart!.extent) as readonly [number,number,number]),key=plannedHomeKey??posedBatchIdentity(batch);
      const words=plannedHomeWords||chartHomeWords(levels);
      home=this.homes.get(key);

      // level-2 slots can carry different 4/8/16 sizes and hence different level-home offsets). A changed
      // layout never reuses the old offsets: it gets a fresh allocation appended after them, so storage still
      // referenced by in-flight GPU work keeps its old contents until the bounded home space restarts.
      const sameLayout=!!home&&home.anchorParts===anchorParts&&home.levels.length===levels.length&&home.levels.every((clusters,i)=>clusters===levels[i]);
      if(home&&(home.source!==source||!sameLayout)){if(sameLayout){home.source=source;home.fresh=true;}else home=undefined;}
      if(!home){
        if(this.homeWords+words>COLUMN_HOME_WORDS){this.homes.clear();this.homeWords=0;}
        home={offset:this.homeScratch+this.homeWords,slots,levels,anchorParts,source,fresh:true};this.homeWords+=words;this.homes.set(key,home);
      }
    }
    const required=Math.max(contactWords,this.homes.size?this.homeScratch+this.homeWords:0);
    if(!this.contactWork||this.contactWork.size<required*4){
      if(this.contactWork)this.retired.push(this.contactWork);
      const maximumWords=Math.floor(Math.min(this.device.limits.maxBufferSize,this.device.limits.maxStorageBufferBindingSize)/4);
      if(required>maximumWords)throw Error(`Complete contact arena (${required} words) exceeds device binding capacity`);
      const size=this.homes.size?Math.min(maximumWords,Math.ceil(required*1.25)):required;
      this.contactWork=this.device.createBuffer({label:'MAIN source-local posed quad contacts',size:size*4,usage:GPUBufferUsage.STORAGE});
      // A new buffer is zero: every home's header is written again before use.
      for(const record of this.homes.values())record.fresh=true;
    }
    if(home?.fresh){
      // Every level home gets a fresh header [capacity, no token, no block, held level 0].
      let at=home.offset;for(const clusters of home.levels){this.device.queue.writeBuffer(this.contactWork,at*4,new Uint32Array([clusters,0,0,0]));at+=4+clusters*4;}
      home.fresh=false;
    }
    new Uint32Array(floats.buffer)[35]=home?.offset??0;
    const parameterWords=this.interior?new Uint32Array(floats.buffer).slice():undefined;this.device.queue.writeBuffer(params,0,floats);
    if(!selection)throw Error('Admitted source frontier has no compact selection buffer');
    const buffers=[params,indexed?.incidence.binding??source.bufferBinding??source.buffer,batch.storage?.vertices??batch.vertices,batch.storage?.attributes??batch.attributes,batch.storage?.metadata??batch.metadata,this.lines,this.status,this.dispatch,selection] as SourceStorageBinding[];
    buffers[12]=this.contactWork;
    buffers[13]=chartCodes??batch.storage?.metadata??batch.metadata;
    buffers[14]=parts?.levels??this.status;
    buffers[18]=occupancy?.columns??this.status;
    buffers[19]=this.partAnchorAlloc;
    (buffers as RelationBinding[])[17]=this.partAnchorBuffer??this.partAnchorAlloc;
    const used:Record<string,readonly number[]>={...SURFACE_PART_ANCHOR_BINDINGS,...SURFACE_CONTACT_BINDINGS,prepare:[0,1,2,3,4,5,6,7,8,10,11],boundaries:this.preparation==='generic-direct-ordinary'?[0,1,2,3,4,5,6,8,10,11,12,15]:this.preparation==='generic-direct'?[0,1,2,3,4,5,6,8,10,11,15]:[0,1,2,3,4,5,6,8,10,11,12,15],boundariesVoxel:[0,1,2,3,4,5,6,8,10,11,12,15],contours:(this.preparation==='generic-direct'||this.preparation==='generic-direct-ordinary')?[0,1,2,3,4,5,6,8,10,11,15]:[0,1,2,3,4,5,6,8,10,11,12,15],contoursVoxel:[0,1,2,3,4,5,6,8,10,12,15]};
    (buffers as RelationBinding[])[10]={view:this.lineDetailView(earlyDetail)};
    (buffers as RelationBinding[])[15]={view:this.contactPatchView()};
    const anchors=this.partAnchorsFor(batch,source);
    this.ensurePartAnchorRowTable();
    (buffers as RelationBinding[])[16]=anchors?.input??this.emptyPartAnchorVertices!;
    if(anchors&&this.partAnchorBuffer)(buffers as RelationBinding[])[17]=this.partAnchorBuffer;

    (buffers as RelationBinding[])[11]=earlyDetail?formDepths(this.device).matrix(field!):params;
    // ONE PASS PER SOURCE (frame cost, 2026-11-10): the seven stages of one
    // source are dispatches of one compute pass, in the same order. WebGPU
    // makes each dispatch's writes visible to the next dispatch of the pass
    // (and the indirect arguments `prepare` writes to the dispatch after it),
    // so the result is unchanged; a separate pass per stage cost about 3200
    // passes per Y8 text frame in driver work alone.
    const pass=shared??encoder.beginComputePass({label:'MAIN source relations'});
    const end=():void=>{if(!shared)pass.end();};
    // HOST-SIZED STAGES (frame cost, 2026-11-12). With a static draw extent
    // the stages are the same dispatches in the same order; only the
    // one-thread prepare and prepareInterior passes, which wrote each stage's
    // workgroup count from the source words and the batch metadata, are
    // replaced by that count computed here (edges, faces and vertices of the
    // source, instances of the batch; a physical batch's metadata is always
    // valid). Relations are appended by the same dispatches in the same order.
    // Two stages per source (and per instance range) are gone, ~350 of the
    // ~1200 dispatches of a Y8 view, each a serial step of the shared pass.
    const host=this.interior&&extent?extent:undefined,limit=this.device.limits.maxComputeWorkgroupsPerDimension;
    const vertices=sourceResidencyVertexCount(source),faces=sourceResidencyFaceCount(source),edges=host?sourceResidencyEdgeCount(source):0;
    const direct=(stage:string,threads:number,groups:RelationGroups,bindings:readonly number[],bound:readonly RelationBinding[]):void=>{
      if(threads<=0)return;const pipeline=this.pipelines[stage]!,count=Math.ceil(threads/64);
      pass.setPipeline(pipeline);pass.setBindGroup(0,this.group(groups,stage,pipeline,bindings,bound));pass.dispatchWorkgroups(Math.min(count,limit),Math.ceil(count/limit));
    };
    const indirect=(stage:string,offset?:number)=>{const pipeline=this.pipelines[stage]!,bindings=used[stage]!;pass.setPipeline(pipeline);pass.setBindGroup(0,this.group(held.groups,stage,pipeline,bindings,buffers));if(offset===undefined)pass.dispatchWorkgroups(1);else pass.dispatchWorkgroupsIndirect(this.dispatch,offset);};
    if((posedContacts&&!host)||frontierAdmitted){
      indirect('prepare');
      if(posedContacts&&producerCapable){
        indirect('contactClear',24);indirect('voxelVote',84);indirect('voxelHome',96);indirect('voxelBin',108);indirect('chartCorners',120);indirect('chartCertify',132);indirect('chartSelectStage',48);indirect('chartEmit',132);direct('chartPublish',1,held.groups,used.chartPublish!,buffers);indirect('contactVertices',36);indirect('contactEdges',48);indirect('contactSort',48);
      }else if(posedContacts&&fineOnlyOrdinary){
        // Fine-only late producers still need posed contact/canonical ports,
        // but have no chart, residual or voxel stages in this profile.
        indirect('contactClear',24);indirect('contactVertices',36);indirect('contactEdges',48);indirect('contactSort',48);
      }
    }else if(posedContacts){
      if(producerCapable){
        direct('contactClear',contactWords,held.groups,used.contactClear!,buffers);
        direct('voxelVote',faces*2,held.groups,used.voxelVote!,buffers);
        direct('voxelHome',home?chartHomeWords(home.levels):0,held.groups,used.voxelHome!,buffers);
        direct('voxelBin',contactEdges,held.groups,used.voxelBin!,buffers);
        direct('chartCorners',faces*2,held.groups,used.chartCorners!,buffers);
        direct('chartCertify',chart?Math.max(chart.clusters,anchorParts):0,held.groups,used.chartCertify!,buffers);
        direct('chartSelectStage',contactEdges,held.groups,used.chartSelectStage!,buffers);
        direct('chartEmit',chart?Math.max(chart.clusters,anchorParts):0,held.groups,used.chartEmit!,buffers);
        direct('chartPublish',1,held.groups,used.chartPublish!,buffers);
        if(chart){
          // chartPublish writes the compact/full producer count into the same
          // indirect segment used by contactEdges. Sorting consumes that exact
          // domain and offset; it must not fall back to hash-slot capacity.
          indirect('contactVertices',36);indirect('contactEdges',48);indirect('contactSort',48);
        }else{
          // Without a chart there is no chartPublish writer. Use the complete
          // host/native source extent directly so contact stages never consume
          // unwritten indirect arguments.
          direct('contactVertices',Math.max(faces*2,anchorParts),held.groups,used.contactVertices!,buffers);
          direct('contactEdges',contactEdges,held.groups,used.contactEdges!,buffers);
          direct('contactSort',contactEdges,held.groups,used.contactSort!,buffers);
        }
      }else if(fineOnlyOrdinary){
        direct('contactClear',contactWords,held.groups,used.contactClear!,buffers);
        direct('contactVertices',Math.max(faces*2,anchorParts),held.groups,used.contactVertices!,buffers);
        direct('contactEdges',contactEdges,held.groups,used.contactEdges!,buffers);direct('contactSort',contactEdges,held.groups,used.contactSort!,buffers);
      }
    }

    // R48: once the form table is available, prepareIn writes the exact mixed
    // fine/planar/hull prefix and its boundary indirect count. The old fused
    // E*instances surfaceLines path remains only for sources without early
    // levels; it cannot consume a compact mixed-bank mapping safely.
    if(host&&!frontierAdmitted){
      if(earlyLevels){indirect('prepare');indirect(boundaryStage,0);indirect(contourStage,12);}
      else if(posedContacts){indirect(boundaryStage,0);indirect(contourStage,12);}
      else{const instances=frontier?.indices.length??host.instances;direct('surfaceLines',(edges+faces)*instances,held.groups,used.boundaries!,buffers);}
    }else{if(!posedContacts&&!frontierAdmitted)indirect('prepare');indirect(boundaryStage,0);indirect(contourStage,12);}
    // A host-sized posed-contact source skips prepare above; run it here so
    // its anchor dispatch uses the same native count and no capacity-derived
    // direct extent. All other paths already issued prepare before lines.
    if(host&&posedContacts&&!frontierAdmitted&&!earlyLevels)indirect('prepare');

    // link each complete record from the prepare-written indirect extent.
    if(anchors&&this.partAnchorBuffer){
      if(!this.partAnchorFrameCleared){direct('partAnchorClear',this.partAnchorHeadSlots+this.partAnchorRecordCapacity,held.groups,used.partAnchorClear!,buffers);this.partAnchorFrameCleared=true;}
      indirect('partAnchors',72);
    }
    if(!this.interior){end();return;}
    // A producer may certify its MATERIAL'S posed normal field as constant
    // inside every face. Its normal differences and therefore all smooth
    // curvature tensors are identically zero, at every level/pose. Hard
    // edges, silhouettes and contacts above still use the complete geometry.
    if(isGpuIndexedResident(source)&&source.gpu.posedNormalClass==='face-constant'){end();return;}
    if(!vertices||!faces){end();return;}
    const capacity=Math.min(batch.source.instanceCapacity??Infinity,batch.gpuPose?.singleInstanceVertexCount!==undefined?1:Math.floor((batch.storage?.vertices.size??batch.vertices.size)/16/batch.source.vertexCapacity));
    if(!capacity){end();return;}
    const perInstance=(faces*3+vertices*3)*16,maximum=Math.min(this.device.limits.maxStorageBufferBindingSize,this.device.limits.maxBufferSize);
    // A device binding limit is not a sensible working-set target. Keep every
    // instance, but reuse a small tensor across complete ordered chunks. A
    // single large mesh still gets its full tensor (no face/detail reduction).
    // Retained posed tensors keep their established cache layout and lifetime.
    const workingBytes=batch.source.posedRevision===undefined?Math.min(maximum,Math.max(perInstance+32,4*1024*1024)):maximum;
    const chunk=Math.min(capacity,Math.floor((workingBytes-16)/(perInstance+16)));
    if(!chunk){end();throw Error('Posed surface differentials require source-face chunks');}
    const bytes=perInstance*chunk+(chunk+1)*16;
    if(batch.source.posedRevision===undefined&&(!this.differentials||this.differentials.size<bytes)){
      if(this.differentials)this.retired.push(this.differentials);
      this.differentials=this.device.createBuffer({label:'MAIN reusable posed surface differentials',size:bytes,usage:GPUBufferUsage.STORAGE});
    }
    // Reuse scratch only across ORDERED GPU passes. Source instances are
    // chunked to the device binding extent, never silently capped or dropped.
    // Only emitted source relations survive these passes; no tensor readback.
    const interiorUsed:Record<string,number[]>={prepareInterior:[0,1,2,3,4,6,7,8,9,10,11],faceDifferentials:[0,1,2,3,4,6,8,9],radialCurvature:[0,1,2,3,4,8,9],suggestiveContours:[0,1,2,3,4,5,6,8,9,15],principalCurvature:[0,1,2,3,4,8,9],ridgeDerivative:[0,1,2,4,8,9],ridgeLines:[0,1,2,3,4,5,6,8,9,15],viewFaceLines:[0,1,2,3,4,5,6,8,9,15]};
    const hostInterior=host&&(!earlyDetail||terrain)&&!frontierAdmitted;
    const detailView=earlyDetail?formDepths(this.device).view(field!):(this.emptyDetail??=this.device.createTexture({label:'MAIN unselected detail table',size:[1,1],format:'rgba32uint',usage:GPUTextureUsage.TEXTURE_BINDING})).createView();
    const detailMatrix=earlyDetail?formDepths(this.device).matrix(field!):params;
    for(let first=0;first<capacity;first+=chunk){
      let differential=this.differentials,refresh=true;
      if(batch.source.posedRevision!==undefined){
        let chunks=this.retainedDifferentials.get(posedBatchIdentity(batch));if(!chunks){chunks=new Map();this.retainedDifferentials.set(posedBatchIdentity(batch),chunks);}
        let cached=chunks.get(first);
        if(!cached||cached.buffer.size<bytes){if(cached)this.retired.push(cached.buffer);cached={revision:-1,buffer:this.device.createBuffer({label:'MAIN retained posed shape tensor',size:bytes,usage:GPUBufferUsage.STORAGE})};chunks.set(first,cached);}
        differential=cached.buffer;const key=tensorRevision(batch,frontierAdmitted);refresh=key===undefined||cached.revision!==key;cached.revision=key??-1;
      }
      let range=held.ranges.get(first);
      if(!range){range={uniform:this.device.createBuffer({label:'MAIN posed differential instance range',size:224,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),groups:new Map()};held.ranges.set(first,range);}
      const uniform=range.uniform;
      const localCount=Math.min(chunk,capacity-first),data=parameterWords!.slice();data.set([first,localCount,this.device.limits.maxComputeWorkgroupsPerDimension],32);
      if(frontier){
        // The admission pass compacts the frontier by ordinal. Chunk tensor
        // ranges therefore use compacted-list offsets, not old source values.
        data[42]=Math.min(frontier.indices.length-first,localCount);data[43]=frontier.offset+first;data[48]=frontier.offset-1;data[49]=first;data[50]=localCount;data[51]=1;
      }
      this.device.queue.writeBuffer(uniform,0,data);
      const interiorBuffers:RelationBinding[]=[uniform,...buffers.slice(1,9),differential!,{view:detailView},detailMatrix];interiorBuffers[15]={view:this.contactPatchView()};
      if(hostInterior){
        // prepareInterior's extent (WGSL interiorInstances), then the same
        // stages in the same order as the indirect path below.
        const local=Math.min(Math.min(chunk,capacity-first),host.instances-Math.min(host.instances,first));
        const visible=frontier?data[42]!:local;
        const counts:Record<string,number>={faceDifferentials:faces,radialCurvature:vertices,suggestiveContours:faces,principalCurvature:vertices,ridgeDerivative:vertices,ridgeLines:faces};
        // A retained tensor (no refresh) leaves suggestive contours and ridge
        // lines adjacent with the same face extent: one viewFaceLines dispatch.
        const stages=refresh?['faceDifferentials','radialCurvature','suggestiveContours','principalCurvature','ridgeDerivative','ridgeLines']:['radialCurvature','viewFaceLines'];
        for(const stage of stages){
          if((stage==='faceDifferentials'||stage==='principalCurvature'||stage==='ridgeDerivative')&&!refresh)continue;
          const instances=stage==='radialCurvature'||stage==='suggestiveContours'||stage==='ridgeLines'||stage==='viewFaceLines'?visible:local;
          direct(stage,(stage==='viewFaceLines'?faces:counts[stage]!)*instances,range.groups,interiorUsed[stage]!,interiorBuffers);
        }
        continue;
      }
      const stages=refresh?['prepareInterior','faceDifferentials','radialCurvature','suggestiveContours','principalCurvature','ridgeDerivative','ridgeLines']:['prepareInterior','radialCurvature','viewFaceLines'];
      for(const stage of stages){
        // The tensor, its principal frame and its ridge derivative do not
        // depend on the view: a retained posed tensor keeps them until the
        // pose changes. Only the visible-side line passes run every frame.
        if((stage==='faceDifferentials'||stage==='principalCurvature'||stage==='ridgeDerivative')&&!refresh)continue;
        const pipeline=this.pipelines[stage]!,group=this.group(range.groups,stage,pipeline,interiorUsed[stage]!,interiorBuffers);
        pass.setPipeline(pipeline);pass.setBindGroup(0,group);
        if(stage==='prepareInterior')pass.dispatchWorkgroups(1);else pass.dispatchWorkgroupsIndirect(this.dispatch,stage==='radialCurvature'?60:stage==='suggestiveContours'||stage==='ridgeLines'||stage==='viewFaceLines'?48:stage==='principalCurvature'||stage==='ridgeDerivative'?36:24);
      }
    }
    end();
  }

  private queue?:BatchQueue;
  /** Retained posed tensors of queued sources live in shared pages, so one
   * dispatch can address every source's tensor at its own base. */
  private tensors?:TensorPages;
  private readonly tensorRanges=new Map<object,Map<number,{revision:number;range:TensorRange;bytes:number}>>();
  private batchContact?:GPUBuffer;private batchDispatch?:GPUBuffer;private emptyMatrix?:GPUBuffer;
  private readonly batchTables:{buffer:GPUBuffer;groups:RelationGroups}[]=[];private batchCursor=0;
  private readonly bindingIds=new WeakMap<object,number>();private bindingSerial=0;
  private bindingId(value:RelationBinding):string{
    if('view' in value){let id=this.bindingIds.get(value.view);if(id===undefined){id=++this.bindingSerial;this.bindingIds.set(value.view,id);}return 'v'+id;}
    const b=storageBinding(value);let id=this.bindingIds.get(b.buffer);if(id===undefined){id=++this.bindingSerial;this.bindingIds.set(b.buffer,id);}
    return b.offset===undefined&&b.size===undefined?String(id):`${id}@${b.offset??0}:${b.size??''}`;
  }
  /** The per-source stages of `append` for one source, as queued items: the
   * same thread counts (host-sized, or prepareInterior's upper bound with its
   * GPU count read by the stage body), the same parameter words, and word
   * bases in place of binding offsets. */
  private queueBatched(batch:PosedSourceBatch,source:ResidentRelationSource,words:Uint32Array,host:SourceDrawExtent,{posedContacts,contactWords,baselineContactWords,residualWords,contactEdges,parts,chartCodes,occupancy,chartClusters,anchorParts,earlyDetail,terrain,selection,frontierAdmitted}:{posedContacts:boolean;contactWords:number;baselineContactWords:number;residualWords:number;contactEdges:number;parts:{readonly cellCodes:SourceStorageBinding;readonly levels:SourceStorageBinding}|undefined;chartCodes:SourceStorageBinding|undefined;occupancy:GpuIndexedCellOccupancy|undefined;chartClusters:number;anchorParts:number;earlyDetail:boolean;terrain:boolean;selection:GPUBuffer;frontierAdmitted:boolean}):void{
    const q=this.queue!;q.seen.add(posedBatchIdentity(batch));
    const indexed=isGpuIndexedResident(source)?source.gpu:undefined;
    const whole=(value:SourceStorageBinding,unit:number):[GPUBuffer,number]=>{const b=storageBinding(value),offset=b.offset??0;if(offset%unit)throw Error('Batched source relation range is not word aligned');return [b.buffer,offset/unit];};
    const [incidence,incidenceBase]=whole(indexed?.incidence.binding??source.bufferBinding??source.buffer,4);
    const [vertices,vertexBase]=whole(batch.storage?.vertices??batch.vertices,16);
    const [attributes,attributeBase]=whole(batch.storage?.attributes??batch.attributes,16);
    const [metadata,metadataBase]=whole(batch.storage?.metadata??batch.metadata,4);
    if(posedContacts){
      const maximumBytes=Math.min(this.device.limits.maxBufferSize,this.device.limits.maxStorageBufferBindingSize);
      const residualRequested=(words[45]!&64)!==0;
      const arena=chooseOptionalContactArena(baselineContactWords,residualRequested?residualWords:0,maximumBytes,q.contactWords,1.25);
      contactWords=arena.words;
      if(residualRequested&&!arena.enabled)words[45]&=~64;
    }
    const contactBase=posedContacts?q.contactWords:0;if(posedContacts)q.contactWords+=contactWords;
    const base=[incidenceBase,vertexBase,attributeBase,metadataBase];
    // Bindings 5, 6, 7 and 12 are this object's shared buffers, bound at flush.
    const bound:RelationBinding[]=[];bound[1]=incidence;bound[2]=vertices;bound[3]=attributes;bound[4]=metadata;bound[8]=selection;bound[17]=this.partAnchorBuffer??this.partAnchorAlloc;bound[19]=this.partAnchorAlloc;
    // Anchor projection and source lines consume the SAME camera detail
    // view. Publish it before copying anchorBound; setting only lineBound
    // later leaves binding10 undefined in the actual batched anchor stage.
    bound[10]={view:this.lineDetailView(earlyDetail)};
    bound[11]=earlyDetail?formDepths(this.device).matrix(this.detailField!):(this.emptyMatrix??=this.device.createBuffer({label:'MAIN unselected detail matrix',size:64,usage:GPUBufferUsage.UNIFORM}));
    bound[13]=chartCodes??metadata;bound[14]=parts?.levels??this.status;bound[15]={view:this.contactPatchView()};bound[18]=occupancy?.columns??this.status;
    const push=(stage:BatchStage,threads:number,itemWords:Uint32Array,base2:readonly number[],itemBound:RelationBinding[],dynamic=false)=>{if(threads>0)q.items.push({stage,threads,words:itemWords,base,base2,bound:itemBound,dynamic});};
    const vertexCount=sourceResidencyVertexCount(source),faces=sourceResidencyFaceCount(source),edges=sourceResidencyEdgeCount(source);

    const anchorEntry=this.partAnchorsFor(batch,source);
    this.ensurePartAnchorRowTable();
    const frontier=batch.source.visibleInstanceFrontier,selectedInstances=frontier?.indices.length??host.instances;
    // R50's per-source representation prefix is a reserved tail of the shared
    // relation output arena. The ordinary indirect words remain separate and
    // are still written in the existing ordered prepare pass.
    const earlyLevels=earlyDetail&&supportsCompiledInstanceLevels(source)&&!!(batch.source.physicalUniformProducerRows||batch.source.ownerAuthority);
    const requestedPrefixRecords=earlyLevels?this.noteLevelPrefixDemand(selectedInstances):0;
    // A direct source may already have captured `lines` in this frame. If its
    // reserved suffix leaves no room for this queued source, preserve the
    // complete incumbent path instead of growing the live arena at flush.
    const prefixEnd=this.levelPrefixUsed+q.levelPrefixRecords+requestedPrefixRecords;
    const useLevelPrefix=earlyLevels&&requestedPrefixRecords>0&&prefixEnd<=this.levelPrefixBudget()&&(!this.levelPrefixWriting||prefixEnd<=this.levelPrefixRecords);
    const levelPrefixWords=useLevelPrefix?2+selectedInstances*4:0;
    const levelPrefixBase=this.lineCapacity+q.levelPrefixRecords;
    if(levelPrefixWords)q.levelPrefixRecords+=requestedPrefixRecords;
    const frontierDispatchBase=(posedContacts||frontierAdmitted||earlyDetail)?q.dispatchWords:0;
    if(posedContacts||frontierAdmitted||earlyDetail)q.dispatchWords+=BATCH_DISPATCH_WORDS;
    if(levelPrefixWords){words[52]=levelPrefixBase;words[53]=1;words[54]=selectedInstances;words[55]=batch.source.physicalUniformProducerRows?2:batch.source.ownerAuthority?.kind==='standard-entity'?1:batch.source.ownerAuthority?.kind==='producer-row'?2:0;}
    if(batch.source.physicalUniformProducerRows||batch.source.ownerAuthority?.kind==='producer-row')words[45]|=128;
    if(frontierAdmitted||earlyDetail){words[49]=0;words[50]=anchorParts;push('prepareFrontier',64,words,[contactBase,0,frontierDispatchBase,Number(frontierAdmitted)],bound);}
    // Registration may grow the shared table; refresh the chart-certify
    // binding after that transaction before queueing any stage that writes the
    // current-frame suppression marker.
    bound[17]=this.partAnchorBuffer??this.partAnchorAlloc;
    bound[16]=anchorEntry?.input??this.emptyPartAnchorVertices!;
    const relationBase2=[contactBase,0,frontierDispatchBase,Number(frontierAdmitted)];
    if(anchorEntry&&this.partAnchorBuffer){const anchorBound=[...bound];anchorBound[16]=anchorEntry.input;anchorBound[17]=this.partAnchorBuffer;push('partAnchors',anchorEntry.parts*anchorEntry.instances,words,relationBase2,anchorBound,frontierAdmitted);}
    if(posedContacts){const compact=chartClusters>0;push('contactClear',contactWords,words,relationBase2,bound,frontierAdmitted);push('voxelVote',faces*2,words,relationBase2,bound,frontierAdmitted);push('voxelBin',contactEdges,words,relationBase2,bound,frontierAdmitted);push('chartCorners',faces*2,words,relationBase2,bound,frontierAdmitted);push('chartCertify',Math.max(chartClusters,anchorParts),words,relationBase2,bound,frontierAdmitted);push('chartSelectStage',contactEdges,words,relationBase2,bound,frontierAdmitted);push('chartEmit',chartClusters,words,relationBase2,bound,frontierAdmitted);push('chartPublish',1,words,relationBase2,bound);push('contactVertices',Math.max(faces*2,anchorParts),words,relationBase2,bound,frontierAdmitted||compact);push('contactEdges',contactEdges,words,relationBase2,bound,frontierAdmitted||compact);push('contactSort',contactEdges,words,relationBase2,bound,frontierAdmitted||compact);}
    const lineBound=[...bound];
    lineBound[11]=earlyDetail?formDepths(this.device).matrix(this.detailField!):(this.emptyMatrix??=this.device.createBuffer({label:'MAIN unselected detail matrix',size:64,usage:GPUBufferUsage.UNIFORM}));
    // Keep the original fused operation's boundary-before-contour order, but
    // give each body its own entry so physical startup does not compile the
    // large combined batchSurfaceLines call graph. Each stage still groups the
    // same queued source items; only one additional dispatch per existing group
    // is introduced, never one dispatch per source.
    const dynamicLines=frontierAdmitted||earlyDetail;
    // Dynamic lines use the producer-written indirect count. Keep a wide
    // table span so a complete planar bank can never be clipped by the old
    // fine E*instance host estimate; boundaryAt/contourAt still reject only
    // their own out-of-range IDs.
    push('boundaryLines',dynamicLines?0xffffffff:edges*selectedInstances,words,relationBase2,lineBound,dynamicLines);
    push('contourLines',dynamicLines?0xffffffff:faces*selectedInstances,words,relationBase2,lineBound,dynamicLines);
    if(indexed?.posedNormalClass==='face-constant'||!vertexCount||!faces)return;
    const capacity=Math.min(batch.source.instanceCapacity??Infinity,batch.gpuPose?.singleInstanceVertexCount!==undefined?1:Math.floor((batch.storage?.vertices.size??batch.vertices.size)/16/batch.source.vertexCapacity));
    if(!capacity)return;
    const perInstance=(faces*3+vertexCount*3)*16,maximum=Math.min(this.device.limits.maxStorageBufferBindingSize,this.device.limits.maxBufferSize);
    const chunk=Math.min(capacity,Math.floor((maximum-16)/(perInstance+16)));
    if(!chunk)throw Error('Posed surface differentials require source-face chunks');
    const bytes=perInstance*chunk+(chunk+1)*16;
    const hostInterior=!earlyDetail||terrain;
    const detailView=earlyDetail?formDepths(this.device).view(this.detailField!):(this.emptyDetail??=this.device.createTexture({label:'MAIN unselected detail table',size:[1,1],format:'rgba32uint',usage:GPUTextureUsage.TEXTURE_BINDING})).createView();
    // Bound only by prepareInterior, which runs only with the early detail table.
    const detailMatrix=earlyDetail?formDepths(this.device).matrix(this.detailField!):(this.emptyMatrix??=this.device.createBuffer({label:'MAIN unselected detail matrix',size:64,usage:GPUBufferUsage.UNIFORM}));
    const tensors=this.tensors??=new TensorPages(this.device,Math.floor(maximum/256)*256);
    let chunks=this.tensorRanges.get(posedBatchIdentity(batch));if(!chunks){chunks=new Map();this.tensorRanges.set(posedBatchIdentity(batch),chunks);}
    for(let first=0;first<capacity;first+=chunk){
      let cached=chunks.get(first);
      if(!cached||cached.bytes<bytes){
        if(cached)tensors.retire(cached.range);
        cached={revision:-1,range:tensors.allocate(bytes),bytes};chunks.set(first,cached);
        // A new range starts zeroed, as a new retained tensor buffer did.
        q.clears.push({buffer:cached.range.page.buffer,offset:cached.range.offset,size:cached.range.size});
      }
      const key=tensorRevision(batch,frontierAdmitted),refresh=key===undefined||cached.revision!==key;cached.revision=key??-1;
      const tensor={buffer:cached.range.page.buffer,offset:cached.range.offset};
      const localCount=Math.min(chunk,capacity-first),data=words.slice();data.set([first,localCount,this.device.limits.maxComputeWorkgroupsPerDimension],32);
      if(frontier){
        data[42]=Math.min(frontier.indices.length-first,localCount);data[43]=frontier.offset+first;data[48]=frontier.offset-1;data[49]=first;data[50]=localCount;data[51]=1;
      }
      const interiorBound=[...bound];interiorBound[9]=tensor.buffer;interiorBound[10]={view:detailView};interiorBound[11]=detailMatrix;
      // WGSL interiorInstances and interiorCandidates for this chunk.
      const local=Math.min(localCount,host.instances-Math.min(host.instances,first)),visible=frontier?data[42]!:local;
      const gated=!hostInterior||frontierAdmitted;
      // Interior chunks each publish their own counts: unlike the boundary
      // prefix, these arguments are rewritten by every prepareInterior item.
      // Keep a distinct 36-word segment per chunk, including when the source
      // also owns the level prefix, so later chunks cannot overwrite earlier
      // tensor dispatches before the globally ordered consumer stage.
      const dispatchBase=gated?(()=>{const at=q.dispatchWords;q.dispatchWords+=BATCH_DISPATCH_WORDS;return at;})():0;
      const base2=[contactBase,(tensor.offset??0)/16,dispatchBase,Number(gated)];
      if(gated)q.items.push({stage:'prepareInterior',threads:64,words:data,base,base2,bound:interiorBound,dynamic:true});
      const counts:Partial<Record<BatchStage,number>>={faceDifferentials:faces*local,radialCurvature:vertexCount*visible,suggestiveContours:faces*visible,principalCurvature:vertexCount*local,ridgeDerivative:vertexCount*local,ridgeLines:faces*visible,viewFaceLines:faces*visible};
      const stages:BatchStage[]=refresh?['faceDifferentials','radialCurvature','suggestiveContours','principalCurvature','ridgeDerivative','ridgeLines']:['radialCurvature','viewFaceLines'];
      for(const stage of stages)push(stage,counts[stage]!,data,base2,interiorBound,gated);
    }
  }
  /** Every queued source's stages: one dispatch per stage per group of sources
   * whose bound pages agree (at most SURFACE_RELATION_BATCH_ITEMS per dispatch). */
  private flushBatched(encoder:GPUCommandEncoder):void{
    const q=this.queue;this.queue=undefined;if(!q)return;
    if(q.levelPrefixRecords){
      const prefixStart=this.levelPrefixUsed,prefixEnd=prefixStart+q.levelPrefixRecords;
      // Queue construction disables a prefix that cannot fit after a direct
      // writer. This second check protects future queue producers and makes
      // the no-live-replacement invariant explicit at the allocation owner.
      if(prefixEnd>this.levelPrefixRecords){
        if(this.levelPrefixWriting){
          for(const item of q.items)if(item.words[53]===1){item.words[52]=0;item.words[53]=0;}
          q.levelPrefixRecords=0;
        }else this.ensureLevelPrefixRecords(prefixEnd);
      }
      if(q.levelPrefixRecords){
        for(const item of q.items)if(item.words[53]===1)item.words[52]=item.words[52]!+prefixStart;
        this.levelPrefixUsed=prefixEnd;this.levelPrefixWriting=true;
      }
    }
    if(!q.items.length)return;
    const maximumContactWords=Math.floor(Math.min(this.device.limits.maxBufferSize,this.device.limits.maxStorageBufferBindingSize)/4);
    if(q.contactWords>maximumContactWords)throw Error(`Complete batched contact arena (${q.contactWords} words) exceeds device binding capacity`);
    if(!this.batchContact||this.batchContact.size<Math.max(4,q.contactWords*4)){if(this.batchContact)this.retired.push(this.batchContact);this.batchContact=this.device.createBuffer({label:'MAIN batched posed quad contacts',size:Math.max(4,Math.min(maximumContactWords,Math.ceil(q.contactWords*1.25))*4),usage:GPUBufferUsage.STORAGE});}
    const dispatchBytes=Math.max(1,q.dispatchWords)*4;
    if(!this.batchDispatch||this.batchDispatch.size<dispatchBytes){if(this.batchDispatch)this.retired.push(this.batchDispatch);this.batchDispatch=this.device.createBuffer({label:'MAIN batched source dispatch words',size:dispatchBytes*2,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.INDIRECT});}
    for(const clear of q.clears)encoder.clearBuffer(clear.buffer,clear.offset,clear.size);

    // header [head slots, record base] survives).
    if(!this.partAnchorFrameCleared&&this.partAnchorBuffer&&q.items.some(item=>item.stage==='partAnchors')){encoder.clearBuffer(this.partAnchorBuffer,8);this.partAnchorFrameCleared=true;}
    const limit=this.device.limits.maxComputeWorkgroupsPerDimension;
    const pass=encoder.beginComputePass({label:'MAIN source relations'});
    try{
      for(const stage of BATCH_ORDER){
        const items=q.items.filter(item=>item.stage===stage);if(!items.length)continue;
        const bindings=(this.preparation==='physical-batched'?PHYSICAL_BATCH_BINDINGS:BATCH_BINDINGS)[stage],groups=new Map<string,BatchItem[]>();
        for(const item of items){
          item.bound[5]=this.lines;item.bound[6]=this.status;item.bound[7]=this.batchDispatch;item.bound[12]=this.batchContact;
          const key=bindings.filter(b=>b!==0).map(b=>this.bindingId(item.bound[b]!)).join('|');
          let list=groups.get(key);if(!list){list=[];groups.set(key,list);}list.push(item);
        }
        const pipeline=this.pipelines[batchEntryName(stage)]!,indirectOffset=BATCH_INDIRECT_OFFSETS[stage];
        for(const list of groups.values()){
          // An admitted source owns its compact dispatch arguments. Keep the
          // existing page-grouped path for incumbent sources, but never put a
          // compact source back into a capacity-sized aggregate dispatch.
          const parts:BatchItem[][]=[];
          const staticItems=list.filter(item=>!item.dynamic||indirectOffset===undefined);
          for(let at=0;at<staticItems.length;at+=this.batchItems)parts.push(staticItems.slice(at,at+this.batchItems));
          if(indirectOffset!==undefined)for(const item of list)if(item.dynamic)parts.push([item]);
          for(const part of parts){
            const data=new Uint32Array(4+part.length*BATCH_ITEM_WORDS);let total=0;
            part.forEach((item,k)=>{const o=4+k*BATCH_ITEM_WORDS;data.set(item.words,o);data.set(item.base,o+56);data.set(item.base2,o+60);data.set([total,item.threads],o+64);total+=item.threads;});
            data.set([part.length,limit,total,0],0);
            let table=this.batchTables[this.batchCursor];
            if(!table){table={buffer:this.device.createBuffer({label:'MAIN batched source relation table',size:16+this.batchItems*BATCH_ITEM_WORDS*4,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),groups:new Map()};this.batchTables.push(table);}
            this.batchCursor++;this.device.queue.writeBuffer(table.buffer,0,data);
            const bound=[...part[0]!.bound];bound[0]=table.buffer;
            pass.setPipeline(pipeline);pass.setBindGroup(0,this.group(table.groups,stage,pipeline,bindings,bound));
            if(stage==='prepareInterior')pass.dispatchWorkgroups(part.length);
            else if(part.length===1&&part[0]!.dynamic&&indirectOffset!==undefined)pass.dispatchWorkgroupsIndirect(this.batchDispatch!,part[0]!.base2[2]!+indirectOffset);
            else{const count=Math.ceil(total/64);pass.dispatchWorkgroups(Math.min(count,limit),Math.ceil(count/limit));}
          }
        }
      }
    }finally{pass.end();}
  }
  private serial=0;
  private readonly retained=new Map<ResidentRelationSource,Map<object,RetainedBindings>>();
  private readonly expiry=new SubmissionExpiry<object>();
  /** This source+batch's retained bindings. A second use before the frame is
   * submitted gets a fresh one-frame set: its parameter write would otherwise
   * replace the first use's before either ran. */
  private bindings(source:ResidentRelationSource,batch:PosedSourceBatch):RetainedBindings{
    let byBatch=this.retained.get(source);if(!byBatch){byBatch=new Map();this.retained.set(source,byBatch);}
    const key=posedBatchIdentity(batch);let entry=byBatch.get(key);
    const fresh=():RetainedBindings=>({params:this.device.createBuffer({label:'MAIN source relation parameters',size:224,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),ownerKey:'',groups:new Map(),ranges:new Map(),serial:this.serial});
    if(entry&&entry.serial===this.serial){const once=fresh();this.oneFrame.push(once);return once;}
    if(!entry){
      entry=fresh();byBatch.set(key,entry);const owned=entry;
      this.expiry.use(entry,()=>{if(byBatch.get(key)!==owned)return;this.release(owned);byBatch.delete(key);if(!byBatch.size)this.retained.delete(source);});
    }else this.expiry.use(entry);
    entry.serial=this.serial;return entry;
  }
  private oneFrame:RetainedBindings[]=[];
  /** A bind group is reused while every buffer it names is the same object. */

  // (contactPatchRows). Rewritten only when `revision` changes; an empty table
  // (count 0) keeps every edge end on its own node (the incumbent).
  private contactPatchTexture?:GPUTexture;private contactPatchRevision=-1;private retiredContactPatches?:GPUTexture;
  setContactPatches(revision:number,points:readonly ContactPatchPoint[]):void{
    if(revision===this.contactPatchRevision)return;this.contactPatchRevision=revision;
    const rows=contactPatchRows(points),height=rows.length/4;
    if(!this.contactPatchTexture||this.contactPatchTexture.height<height){
      this.retiredContactPatches?.destroy();this.retiredContactPatches=this.contactPatchTexture;
      this.contactPatchTexture=this.device.createTexture({label:'MAIN contact patch points',size:[1,Math.max(height,(this.contactPatchTexture?.height??0)*2,1)],format:'rgba32uint',usage:GPUTextureUsage.COPY_DST|GPUTextureUsage.TEXTURE_BINDING});
      this.contactPatchViewed=undefined;
    }
    this.device.queue.writeTexture({texture:this.contactPatchTexture},rows,{bytesPerRow:16,rowsPerImage:height},[1,height]);
  }
  private contactPatchViewed?:GPUTextureView;
  private contactPatchView():GPUTextureView{
    // The empty default needs no write: a new texture is zero (count 0).
    if(!this.contactPatchTexture)this.contactPatchTexture=this.device.createTexture({label:'MAIN contact patch points',size:[1,1],format:'rgba32uint',usage:GPUTextureUsage.COPY_DST|GPUTextureUsage.TEXTURE_BINDING});
    return this.contactPatchViewed??=this.contactPatchTexture!.createView();
  }
  private group(groups:RelationGroups,stage:string,pipeline:GPUComputePipeline,bindings:readonly number[],buffers:readonly RelationBinding[]):GPUBindGroup{
    const named=bindings.map(binding=>buffers[binding]!);const held=groups.get(stage);
    if(held&&held.buffers.every((buffer,index)=>{const other=named[index]!;return 'view' in buffer?'view' in other&&buffer.view===other.view:!('view' in other)&&sameStorageBinding(buffer,other);}))return held.group;
    const group=this.device.createBindGroup({label:'MAIN surface relations/'+stage,layout:pipeline.getBindGroupLayout(0),entries:bindings.map((binding,index)=>{const resource=named[index]!;return {binding,resource:'view' in resource?resource.view:storageBinding(resource)};})});
    groups.set(stage,{buffers:named,group});return group;
  }
  private release(entry:RetainedBindings):void{this.retired.push(entry.params);if(entry.selection)this.retired.push(entry.selection);for(const range of entry.ranges.values())this.retired.push(range.uniform);}
  retainPosedBatches(live:ReadonlySet<object>):void{for(const [key,chunks]of this.retainedDifferentials)if(!live.has(key)){for(const entry of chunks.values())this.retired.push(entry.buffer);this.retainedDifferentials.delete(key);}
    for(const [key,chunks]of this.tensorRanges)if(!live.has(key)){for(const entry of chunks.values())this.tensors!.retire(entry.range);this.tensorRanges.delete(key);}
    for(const [key,entry]of this.partAnchorSources)if(!live.has(key as object)){this.retired.push(entry.input);this.partAnchorSources.delete(key);}
  }
  submitted():void{
    this.partAnchorFrameCleared=false;
    this.frontierSelections.clear();
    if(this.partAnchorGrowTo>0||this.partAnchorGrowCanonicalTo>0){const records=this.partAnchorGrowTo,canonical=this.partAnchorGrowCanonicalTo;this.partAnchorGrowTo=0;this.partAnchorGrowCanonicalTo=0;this.growPartAnchorTable(records,canonical);}
    this.intersections?.submitted();
    // Batched tables are rewritten by queue writes ordered after this submission.
    this.batchCursor=0;this.tensors?.submitted();
    for(const entry of this.oneFrame)this.release(entry);this.oneFrame=[];
    // A source+batch unused for 240 frames has left the world (or its batch
    // was replaced): its retained bindings go with it.
    this.serial++;
    this.expiry.submitted();
    for(const buffer of [...this.pending,...this.retired])buffer.destroy();this.pending=[];this.retired=[];
  }
  dispose():void{this.homes.clear();this.homeWords=0;this.expiry.clear();for(const byBatch of this.retained.values())for(const entry of byBatch.values())this.release(entry);this.retained.clear();this.retainPosedBatches(new Set());this.partAnchorGrowTo=0;if(this.partAnchorBuffer)this.retired.push(this.partAnchorBuffer);this.partAnchorBuffer=undefined;this.partAnchorRowLayout=undefined;this.partAnchorRowRequired=0;this.partAnchorRowCapacity=0;this.partAnchorRecords=0;this.partAnchorRecordCapacity=0;this.partAnchorHeadSlots=0;this.submitted();this.intersections?.dispose();this.lines.destroy();this.status.destroy();this.dispatch.destroy();this.partAnchorAlloc.destroy();this.differentials?.destroy();this.contactWork?.destroy();this.emptyDetail?.destroy();this.emptyMatrix?.destroy();this.emptyPartAnchorVertices?.destroy();this.tensors?.dispose();this.batchContact?.destroy();this.batchDispatch?.destroy();for(const table of this.batchTables)table.buffer.destroy();}
}
