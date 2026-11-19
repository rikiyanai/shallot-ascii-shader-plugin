import {storageBinding,sameStorageBinding,type SourceStorageBinding} from './source-storage.ts';
import type {MaterialGeometryBatch,MaterialGeometryDraw} from './geometry.ts';
export type PosedSourceBatch=Pick<MaterialGeometryBatch,'vertices'|'attributes'|'metadata'|'gpuPose'>&{readonly storage?:{readonly identity:object;readonly vertices:GPUBufferBinding;readonly attributes:GPUBufferBinding;readonly metadata:GPUBufferBinding};readonly source:Pick<MaterialGeometryDraw,'vertexCapacity'|'indexBase'|'cullMode'|'instanceCapacity'>&{readonly bounds?:GPUBuffer;readonly visibleInstances?:GPUBuffer;readonly visibleInstanceOffset?:number;readonly visibleInstanceFrontier?:import('./source-instance-arena.ts').SourceInstanceFrontier;readonly posedRevision?:number}};
export function posedBatchIdentity(batch:PosedSourceBatch):object{return batch.storage?.identity??batch.vertices;}
import {isGpuIndexedResident,type ResidentRelationSource,type ResidentGpuIndexedSource} from './source-cache.ts';
import type {GpuIndexedRestCellChart} from './gpu-indexed-source.ts';
import type {SurfaceField} from './surface-fields.ts';
import {SurfaceIntersections} from './surface-intersections.ts';
import {SubmissionExpiry} from './submission-expiry.ts';
import {sourceIncidenceReaders} from './source-incidence-residual.ts';
import {sourceResidencyEdgeCount,sourceResidencyVertexCount,sourceResidencyFaceCount} from './source-transfer.ts';
import {compileComputePipelines,prepareComputePipelines,type ComputePipelinePlan} from './gpu-pipeline-preparation.ts';
import {isolateComputeEntry} from './gpu-compute-program.ts';
import {FORM_DEPTH_WGSL,FORM_DEPTH_HEADER_ROW,FAR_STRIP_BELOW_ROWS,LOD_TIER_HOLD,NEAR_FROM_ROWS,formDepths} from './form-depths.ts';
import {EDGE_LEVEL_UNIT,EDGE_LEVEL_DETAIL,EDGE_LEVEL_CHAINED,FACET_CREASE_LEAST_ROWS} from './compiled-construction-topology.ts';
import {FAR_AGGREGATE_MEMBER_WGSL} from './gpu-source-far-forms.ts';
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
/** A voxel cluster's largest projected size, in native rows (perf-impl-106;
 * perf-impl-108: one glyph row). Steps inside one cluster snap to one level,
 * so contours of one terrace closer than a glyph are ONE line (AGENTS.md
 * finding 4; articulation FACET_COLLAPSE: two parallel lines closer than one
 * glyph cannot both be spelled). At a third of a row (perf-106) a contour
 * every third of a row still reached the page as two or three marks per cell
 * (the voxel island's interior clutter, perf-106 session voxel frame 0). */
export const VOXEL_CLUSTER_ROWS=1;
/** perf-impl-108 contact corner table bound: corner ids per frame (10-bit ids in
 * Relation.sourceChart.w). Beyond it a corner is not used: its edge ends keep
 * their own nodes (the incumbent). */
export const CONTACT_PATCH_POINTS=1023;
/** One supplied contact CORNER (source-corner incidence, perf-impl-108; Codex
 * review: a world point at the drawn pose cannot be fed from fixed-step solver
 * uploads, because the renderer draws GPU-interpolated transforms). An owner
 * (relation key: source domain, entity eid), the owner's SOURCE VERTEX on that
 * contact (the weld-canonical port its edge ends carry: the least original
 * vertex index among the coincident copies in the draw, surface-relations
 * vertexPort), and the corner id shared by both owners of the touching pair
 * (0..CONTACT_PATCH_POINTS-1). The host resolves the solver's body-local
 * contact point to that rest-space vertex once per completed step; it does not
 * move with the drawn pose, so no interpolation or tolerance enters the frame. */
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
struct Params {clip:mat4x4f,eye:vec4f,direction:vec4f,capacity:u32,domain:u32,vertexCapacity:u32,indexBase:u32,selection:vec4u,owners:array<vec4u,4>}
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
@group(0) @binding(8) var<storage,read> selectedOwners:array<u32>;
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
// CONTACT CORNERS (perf-impl-108): explicit two-owner solver incidence by
// SOURCE CORNER, supplied by the host (SurfaceRelations.setContactPatches);
// never a screen, world or pose weld. rgba32uint, one texel a row; row 0 .x =
// the count n; row i+1 = [owner domain, owner eid, source vertex port, corner
// id], sorted by (domain, eid, vertex). An exact key lookup: the edge end's own
// canonical vertex port, no distance and no drawn pose.
@group(0) @binding(15) var contactPatches:texture_2d<u32>;
// perf-impl-137/140: a source's declared part anchors [record base, parts, 6 vertex ids per part] and the ONE
// owner-keyed extent table (heads + records).
@group(0) @binding(16) var<storage,read> partAnchorVertices:array<u32>;
@group(0) @binding(17) var<storage,read_write> partAnchorTable:array<atomic<u32>>;
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
fn selectedCount()->u32{if(params.owners[3].x!=0u){return params.owners[2].z;}return native[1];}
fn selectedInstance(ordinal:u32)->u32{if(params.owners[3].x!=0u){return selectedOwners[params.owners[2].w+ordinal];}return ordinal;}
fn edge(id:u32)->vec4u{return sourceEdge(id);}
const NO_CONTACT=0xffffffffu;
fn posedContacts()->bool{return (params.owners[3].y&1u)!=0u&&native[1]==1u&&native[2]==params.indexBase&&native[5]==0u;}
fn contactSlots()->u32{return params.owners[2].y;}
fn cellParts()->bool{return (params.owners[3].y&2u)!=0u;}
fn partBase()->u32{return contactSlots()*3u+sourceEdgeCount();}
fn edgePartBase()->u32{return contactSlots()*6u+sourceEdgeCount();}
fn partFlagsBase()->u32{return contactSlots()*6u+sourceEdgeCount()*2u;}
// THE VOXEL CLUSTER LEVEL (perf-impl-106): contactSlots() claim words after
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
 // perf-impl-127: keep the held level inside the LOD_TIER_HOLD band (no flip
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
 let w=atomicLoad(&contactWork[homeBase()+2u]);if(w==0u||w>5u){return 0.0;}
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
// THE VOXEL CELL-CHART REPRESENTATION (perf-impl-116; replaces the 110/111
// world-y columns, which read a radially deformed chunk's terraces and caves
// in the wrong frame). ONE level per chunk, selected BEFORE fine incidence.
// The chart is the producer's REST cell lattice (closed-cell parts: each
// vertex's cell x/y/z and face direction, all six faces of every cell). Per
// frame (pose-dependent): each quad stores its posed centroid under its cell
// and face; a cell's posed frame is its six face centroids (axis k = +k face
// - -k face, centre = their mean); a vertex's rest LATTICE POINT is its cell
// plus, per axis, the side of the centre it lies on (no mesher convention);
// the posed position of every lattice point that is a vertex is recorded.
// Generation-owned (retained per generation and block, perf-impl-114 home):
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
fn homeBase()->u32{return select(scratchHome(),params.owners[0].w,persistentHome());}
fn clusterCapacity()->u32{if(persistentHome()){return atomicLoad(&contactWork[homeBase()]);}return chartClusters();}
// perf-impl-117: one word per source vertex, its packed rest lattice point
// (x | y<<9 | z<<18, plus 1; 0 = none), written by chartCorners (which binds
// the cell codes) so the boundary stage reads lattice points WITHOUT binding
// partCodes (it stays within 8 storage buffers).
fn chartVertices()->u32{return select(0u,sourceFaceCount()*2u,chartOn());}
fn latticeBase()->u32{return scratchHome()+4u+chartClusters()*4u;}
fn columnEnd()->u32{return latticeBase()+chartVertices();}
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
// perf-impl-118 OCCUPANCY SEMANTICS, explicit per source (owners[3].y bit 3).
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
 let n=(partExtent()+vec3u(b-1u))/b;let s=homeBase()+4u+index*4u;if(atomicLoad(&contactWork[s])!=0u){return;}
 let m=vec3i(i32(index%n.x),i32((index/n.x)%n.y),i32(index/(n.x*n.y)))*i32(b);
 var state=0u;
 for(var g=0u;g<6u;g++){let r=runState(m,g);if(r==CLUSTER_UNKNOWN){continue;}if(state==0u){state=r;}else if(state!=r){state=CLUSTER_UNKNOWN;break;}}
 atomicStore(&contactWork[s+3u],select(CLUSTER_UNKNOWN,state,state!=0u));
}
// In chartCorners (after voxelBin completed every face mask and claim), by
// every vertex thread over the clusters; skipped when the home is retained.
fn chartClassifyAt(t:u32,threads:u32){
 if(!chartSurface()||atomicLoad(&contactWork[retainedWord()])!=0u){return;}
 let block=voxelChunkBlock();if(block<=1.0){return;}let b=u32(block);
 let n=(partExtent()+vec3u(b-1u))/b;let total=min(n.x*n.y*n.z,clusterCapacity());
 for(var i=t;i<total;i+=max(threads,1u)){classifyCluster(i,b);}
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
 if(atomicLoad(&contactWork[retainedWord()])!=0u){return;}
 let block=voxelChunkBlock();if(block<=1.0){return;}
 let slot=clusterSlot(vec3i(c/vec3u(u32(block))),u32(block));if(slot==NONE_CLUSTER){chartUnsupported();return;}
 atomicMax(&contactWork[slot],~cellIndex(c));atomicMax(&contactWork[slot+1u],~sourceEdgeCorner(halfedge(edge(id).x).z,0u));
}
const NONE_CLUSTER=0xffffffffu;
// The words of cluster C at block b (dense over the cluster grid), or NONE
// outside the grid (open space above or below a chunk is simply empty).
fn clusterSlot(c:vec3i,b:u32)->u32{
 let n=vec3i((partExtent()+vec3u(b-1u))/b);if(any(c<vec3i(0))||any(c>=n)){return NONE_CLUSTER;}
 let index=u32((c.z*n.y+c.y)*n.x+c.x);if(index>=clusterCapacity()){return NONE_CLUSTER;}
 return homeBase()+4u+index*4u;
}
fn clusterState(c:vec3i,b:u32)->u32{
 let s=clusterSlot(c,b);if(s==NONE_CLUSTER){return select(CLUSTER_AIR,CLUSTER_UNKNOWN,chartSurface());}
 if(atomicLoad(&contactWork[s])!=0u){return CLUSTER_SOLID;}if(!chartSurface()){return CLUSTER_AIR;}
 let w=atomicLoad(&contactWork[s+3u])&3u;return select(CLUSTER_UNKNOWN,w,w==CLUSTER_AIR||w==CLUSTER_SOLID);
}
fn clusterOccupied(c:vec3i,b:u32)->bool{return clusterState(c,b)==CLUSTER_SOLID;}
// Holds a declared cell (closed: the same as occupied).
fn clusterDeclared(c:vec3i,b:u32)->bool{let s=clusterSlot(c,b);return s!=NONE_CLUSTER&&atomicLoad(&contactWork[s])!=0u;}
// perf-impl-124: declared and selected this draw (contactEdges marks; chartEmitAt emits).
fn clusterEmits(c:vec3i,b:u32)->bool{let s=clusterSlot(c,b);return s!=NONE_CLUSTER&&atomicLoad(&contactWork[s])!=0u&&(atomicLoad(&contactWork[s+2u])&CLUSTER_SELECTED)!=0u;}
fn clusterKnown(c:vec3i,b:u32)->bool{return clusterState(c,b)!=CLUSTER_UNKNOWN;}
// perf-impl-119 CERTIFICATES, two explicit lifetimes, in the existing cluster words.
// TOPOLOGY (word 3; generation + selected block: cleared with the home, kept
// with a retained one): bit 4 computed, bit 5 every cluster of the 3x3x3
// neighbourhood has a known state (what this cluster's emission reads).
// POSE (word 2; this draw: reset with the emission flags): bit 1 computed,
// bit 2 the least cell places its posed frame; bit 3 SELECTED (an owned fine
// edge reaches it; contactEdges). Both certificates are written by ONE
// thread per cluster in chartCertify (perf-impl-121), never by readers.
const CLUSTER_FRAME_DONE=2u;const CLUSTER_FRAME_OK=4u;const CLUSTER_SELECTED=8u;const CLUSTER_NEAR_DONE=16u;const CLUSTER_NEAR_KNOWN=32u;
fn nearKnown(c:vec3i,b:u32)->bool{
 for(var z=-1;z<=1;z++){for(var y=-1;y<=1;y++){for(var x=-1;x<=1;x++){if(!clusterKnown(c+vec3i(x,y,z),b)){return false;}}}}
 return true;
}
// Readers (perf-impl-121): the flags chartCertify wrote; absent = not certified.
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
// perf-impl-121: with the producer's CERTIFIED corner order (owners[3].y
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
// Cluster C's share of the coarse edges: along each chart axis k, the four
// edges of C, each only where C is the first occupied of its four surrounding
// clusters and all four are interior. Ends at the posed lattice points, faces
// turned by the posed axes of C's least cell.
fn clusterEmit(c:vec3i,b:u32,t:Triangle,address:u32,face:u32){
 let rep=latticeCell(c*i32(b)+vec3i(i32(b)),b);let frame=cellFrame(vec3u(max(rep.xyz,vec3i(0))));if(rep.w==0||!frame.ok){return;}
 let block=f32(b);
 for(var k=0u;k<3u;k++){
  let i=(k+1u)%3u;let j=(k+2u)%3u;let ek=unit(k);let ei=unit(i);let ej=unit(j);
  let U=normalize(frame.axis[i]);let V=normalize(frame.axis[j]);
  for(var corner=0u;corner<4u;corner++){
   let oi=i32(corner&1u);let oj=i32(corner>>1u);
   let q0=c+ei*(oi-1)+ej*(oj-1);let q1=c+ei*oi+ej*(oj-1);let q2=c+ei*(oi-1)+ej*oj;let q3=c+ei*oi+ej*oj;
   let p=vec4<bool>(clusterOccupied(q0,b),clusterOccupied(q1,b),clusterOccupied(q2,b),clusterOccupied(q3,b));
   if(chartSurface()&&!(clusterKnown(q0,b)&&clusterKnown(q1,b)&&clusterKnown(q2,b)&&clusterKnown(q3,b))){continue;}
   // perf-impl-124: the first declared AND SELECTED cluster owns the edge, so
   // a coarse edge beside an owned (suppressed) fine edge is always drawn,
   // exactly once, by a cluster that runs clusterEmit (only selected ones do).
   let d=vec4<bool>(clusterEmits(q0,b),clusterEmits(q1,b),clusterEmits(q2,b),clusterEmits(q3,b));
   var owner=4u;if(d.w){owner=3u;}if(d.z){owner=2u;}if(d.y){owner=1u;}if(d.x){owner=0u;}
   if(owner!=3u-corner){continue;}
   if(clusterBorder(q0,b)||clusterBorder(q1,b)||clusterBorder(q2,b)||clusterBorder(q3,b)){continue;}
   let o=(c+ei*oi+ej*oj)*i32(b);let a=latticePoint(o,b);let z=latticePoint(o+ek*i32(b),b);if(a.w==0.0||z.w==0.0){continue;}
   patternEdges(p,a.xyz,z.xyz,U,V,t,address,block,face);
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
 let block=voxelChunkBlock();if(block<=1.0){return false;}let b=u32(block);
 let la=storedLattice(va);let lb=storedLattice(vb);if(la.w==0||lb.w==0){return false;}
 let around=segmentClusters(la.xyz,lb.xyz,b);
 for(var k=0;k<8;k++){if(around[k].w!=0&&(clusterBorder(around[k].xyz,b)||!clusterCertified(around[k].xyz,b))){return false;}}
 return true;
}
fn voxelEdgeCoarse(id:u32)->bool{if(!chartOn()){return false;}let h=halfedge(edge(id).x);return chartSegmentCoarse(h.x,h.y);}
fn voxelVertexCoarse(v:u32)->bool{
 if(!chartOn()){return false;}let q=(v/4u)*4u;let k=v%4u;
 return chartSegmentCoarse(v,q+(k+1u)%4u)&&chartSegmentCoarse(q+(k+3u)%4u,v);
}
// contactEdges: an owned fine edge marks every declared cluster around it.
fn chartSelect(id:u32){
 let b=u32(voxelChunkBlock());let h=halfedge(edge(id).x);
 let around=segmentClusters(storedLattice(h.x).xyz,storedLattice(h.y).xyz,b);
 for(var k=0;k<8;k++){
  if(around[k].w==0){continue;}let s=clusterSlot(around[k].xyz,b);
  if(s!=NONE_CLUSTER&&atomicLoad(&contactWork[s])!=0u){atomicOr(&contactWork[s+2u],CLUSTER_SELECTED);}
 }
}
// boundaryAt's voxel branch. True: the chart owns this fine edge; it produces
// nothing (its clusters are emitted by chartEmitAt). False: fine path.
fn voxelColumnEdge(id:u32,t:Triangle,face:u32)->bool{return voxelEdgeCoarse(id);}
// Cluster-count work in the boundary stage: thread i (strided) emits every
// SELECTED cluster once, from its own least quad (triangle, address, owner
// gates) - never re-expanded from the fine edges that own it.
fn chartEmitAt(id:u32,threads:u32){
 if(!posedContacts()||!chartOn()||selectedCount()==0u){return;}
 let block=voxelChunkBlock();if(block<=1.0){return;}let b=u32(block);
 let n=(partExtent()+vec3u(b-1u))/b;let total=min(n.x*n.y*n.z,clusterCapacity());
 let instance=selectedInstance(0u);if(!selected(instance)){return;}
 for(var i=id;i<total;i+=max(threads,1u)){
  let s=homeBase()+4u+i*4u;if(atomicLoad(&contactWork[s])==0u||(atomicLoad(&contactWork[s+2u])&CLUSTER_SELECTED)==0u){continue;}
  let inc=~atomicLoad(&contactWork[s+1u]);if(!inDraw(inc)||!validFace(inc)){continue;}
  let v=halfedge(inc).x;let owner=edgeOwner(v,instance);
  if(ownerUnseen(v,instance)||ownerMerged(owner)||outlineLevel(owner)){continue;}
  clusterEmit(vec3i(i32(i%n.x),i32((i/n.x)%n.y),i32(i/(n.x*n.y))),b,triangle(inc,0u),inc,inc/3u);
 }
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
// perf-impl-137/140 SOURCE-DECLARED PART ANCHOR PROJECTION (crawl brief item C).
// Projection, insertion and clearing are separate dispatches. A record is
// published (its index exchanged into a bucket head) only after a previous
// dispatch wrote its complete key and extent; readers walk the bucket list and
// compare the full key (domain|producer, native owner, part ordinal). The
// owner is edgeOwner of the anchor vertex, exactly a run's (identity.y,
// identity.z); never the selected-instance ordinal. Visibility is never read.
const PART_ANCHOR_RECORD_WORDS=8u;
fn partAnchorHeads()->u32{return atomicLoad(&partAnchorTable[0]);}
fn partAnchorRecord(record:u32)->u32{return atomicLoad(&partAnchorTable[1])+record*PART_ANCHOR_RECORD_WORDS;}
fn partAnchorBucket(owner:vec2u,part:u32)->u32{
 var h=(owner.x*0x9e3779b9u)^(owner.y*0x85ebca6bu)^(part*0xc2b2ae35u);h=(h^(h>>16u))*0x7feb352du;h=h^(h>>15u);
 return 2u+(h&(partAnchorHeads()-1u));
}
fn partAnchorCount()->u32{return max(partAnchorVertices[1],0u)*max(selectedCount(),1u);}
// Projection: each thread writes ONLY its own record.
@compute @workgroup_size(64) fn partAnchors(@builtin(global_invocation_id) gid:vec3u){partAnchorAt(gid.x+gid.y*params.owners[3].w*64u);}
fn partAnchorAt(id:u32){
 let parts=partAnchorVertices[1];if(parts==0u||id>=partAnchorCount()){return;}
 let part=id%parts;let instance=selectedInstance(id/parts);let at=partAnchorRecord(partAnchorVertices[0]+id);
 var lo=vec2f(3.402823e38);var hi=vec2f(-3.402823e38);var found=false;var owner=vec2u(0u);var owned=false;
 for(var k=0u;k<6u;k++){
  let v=partAnchorVertices[2u+part*6u+k];if(v==0xffffffffu||v>=params.vertexCapacity){continue;}
  if(!owned){owner=edgeOwner(v,instance);owned=true;}
  let c=params.clip*vec4f(position(v,instance),1.0);if(!(c.w>0.00001)){continue;}
  let ndc=c.xy/c.w;lo=min(lo,ndc);hi=max(hi,ndc);found=true;
 }
 atomicStore(&partAnchorTable[at],owner.x);atomicStore(&partAnchorTable[at+1u],owner.y);atomicStore(&partAnchorTable[at+2u],select(0u,part+1u,owned&&found));
 atomicStore(&partAnchorTable[at+4u],bitcast<u32>(lo.x));atomicStore(&partAnchorTable[at+5u],bitcast<u32>(lo.y));
 atomicStore(&partAnchorTable[at+6u],bitcast<u32>(hi.x));atomicStore(&partAnchorTable[at+7u],bitcast<u32>(hi.y));
}
// Insertion (a later dispatch): link each complete record once per frame.
@compute @workgroup_size(64) fn partAnchorInsert(@builtin(global_invocation_id) gid:vec3u){partAnchorInsertAt(gid.x+gid.y*params.owners[3].w*64u);}
fn partAnchorInsertAt(id:u32){
 if(partAnchorVertices[1]==0u||id>=partAnchorCount()||partAnchorHeads()==0u){return;}
 let record=partAnchorVertices[0]+id;let at=partAnchorRecord(record);let part=atomicLoad(&partAnchorTable[at+2u]);if(part==0u){return;}
 var held=atomicLoad(&partAnchorTable[at+3u]);
 loop{if(held!=0u){return;}let claim=atomicCompareExchangeWeak(&partAnchorTable[at+3u],0u,0x80000000u);held=claim.old_value;if(claim.exchanged){break;}}
 let bucket=partAnchorBucket(vec2u(atomicLoad(&partAnchorTable[at]),atomicLoad(&partAnchorTable[at+1u])),part);
 let next=atomicExchange(&partAnchorTable[bucket],record+1u);atomicStore(&partAnchorTable[at+3u],0x80000000u|next);
}
// Stale invalidation: zero every head and link word once per frame before the
// first insertion (a record of a source not drawn this frame is unreachable).
@compute @workgroup_size(64) fn partAnchorClear(@builtin(global_invocation_id) gid:vec3u){partAnchorClearAt(gid.x+gid.y*params.owners[3].w*64u);}
fn partAnchorClearAt(id:u32){
 let heads=partAnchorHeads();let total=arrayLength(&partAnchorTable);
 if(id<heads){atomicStore(&partAnchorTable[2u+id],0u);return;}
 let record=id-heads;let at=partAnchorRecord(record);if(at+PART_ANCHOR_RECORD_WORDS<=total){atomicStore(&partAnchorTable[at+3u],0u);}
}
@compute @workgroup_size(64) fn contactClear(@builtin(global_invocation_id) gid:vec3u){contactClearAt(gid.x+gid.y*params.owners[3].w*64u);}
fn contactClearAt(id:u32){
 if(!posedContacts()||native[0]==0u){return;}
 let slots=contactSlots();let count=columnEnd();
 if(id<count){let base=partBase();let low=cellParts()&&id>=base+slots&&id<base+slots*2u;atomicStore(&contactWork[id],select(0u,0xffffffffu,low));}
}
// perf-impl-113/116 EARLY-WORK ORDER: clear, voxelVote (every vertex: the
// chunk's block and cell size, and the part cells the part flags read),
// voxelHome, voxelBin (every quad: its cell face; the cluster occupancy unless
// retained), chartCorners (every vertex's lattice point), then contactVertices and
// contactEdges build the fine posed incidence ONLY for a vertex or edge the
// selected coarse representation does not own (voxelSpanCoarse on the
// completed table). A chunk at block 1, a non-voxel source and every vertex or
// edge touching an unrepresentable or border column keep the full fine work.
@compute @workgroup_size(64) fn voxelVote(@builtin(global_invocation_id) gid:vec3u){voxelVoteAt(gid.x+gid.y*params.owners[3].w*64u);}
fn voxelVoteAt(v:u32){
 if(!posedContacts()||v>=min(sourceFaceCount()*2u,native[0]/6u*4u)){return;}
 voxelColumnVertex(v);addPartVertex(v,position(v,0u));
}
@compute @workgroup_size(64) fn voxelBin(@builtin(global_invocation_id) gid:vec3u){voxelBinAt(gid.x+gid.y*params.owners[3].w*64u);}
fn voxelBinAt(id:u32){
 if(!posedContacts()||id>=sourceEdgeCount()||!edgeInDraw(id)){return;}
 if(id%5u==0u){chartQuad(id);}
}
// perf-impl-116: every vertex's rest lattice point and its posed position
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
// perf-impl-121 CLUSTER CERTIFICATION STAGE, one thread per cluster, after
// chartCorners (claims, face centroids and surface states complete) and
// before contactVertices / contactEdges / boundaries read the flags. Surface
// charts only (closed clusters are certified by construction). TOPOLOGY
// certificate: computed once per home rebuild (kept with a retained home).
// POSE certificate: every draw (voxelHome resets it in a retained home).
@compute @workgroup_size(64) fn chartCertify(@builtin(global_invocation_id) gid:vec3u){chartCertifyAt(gid.x+gid.y*params.owners[3].w*64u);}
fn chartCertifyAt(i:u32){
 if(!posedContacts()||native[0]==0u||!chartOn()||!chartSurface()){return;}
 let block=voxelChunkBlock();if(block<=1.0){return;}let b=u32(block);
 let n=(partExtent()+vec3u(b-1u))/b;if(i>=min(n.x*n.y*n.z,clusterCapacity())){return;}
 let s=homeBase()+4u+i*4u;let c=vec3i(i32(i%n.x),i32((i/n.x)%n.y),i32(i/(n.x*n.y)));
 if((atomicLoad(&contactWork[s+3u])&CLUSTER_NEAR_DONE)==0u){atomicOr(&contactWork[s+3u],CLUSTER_NEAR_DONE|select(0u,CLUSTER_NEAR_KNOWN,nearKnown(c,b)));}
 let w=atomicLoad(&contactWork[s]);
 if(w!=0u){let k=~w;let e=partExtent();let ok=cellFrame(vec3u(k%e.x,(k/e.x)%e.y,k/(e.x*e.y))).ok;atomicOr(&contactWork[s+2u],CLUSTER_FRAME_DONE|select(0u,CLUSTER_FRAME_OK,ok));}
}
// perf-impl-114/116: between the vote and the bin, over the home's words.
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
 let h=homeBase();let slots=clusterCapacity();if(id>=4u+slots*4u){return;}
 let block=atomicLoad(&contactWork[chunkBlockWord()]);let token=params.selection.w;
 let kept=token>1u&&block!=0u&&atomicLoad(&contactWork[h+1u])==token&&atomicLoad(&contactWork[h+2u])==block;
 if(id==0u){atomicStore(&contactWork[retainedWord()],select(0u,1u,kept));return;}
 if(id<4u){return;}
 if(kept){if((id-4u)%4u==2u){atomicAnd(&contactWork[h+id],~15u);}return;}
 atomicStore(&contactWork[h+id],0u);
}
@compute @workgroup_size(64) fn contactVertices(@builtin(global_invocation_id) gid:vec3u){contactVertexAt(gid.x+gid.y*params.owners[3].w*64u);}
fn contactVertexAt(v:u32){
 if(!posedContacts()||v>=min(sourceFaceCount()*2u,native[0]/6u*4u)){return;}
 // perf-impl-114: the table is complete (voxelBin ran). Record what it holds:
 // this content token and block, or nothing after an overflow (never kept).
 if(v==0u&&persistentHome()){
  let h=homeBase();let complete=atomicLoad(&contactWork[overflowWord()])==0u;
  atomicStore(&contactWork[h+1u],select(0u,params.selection.w,complete));atomicStore(&contactWork[h+2u],atomicLoad(&contactWork[chunkBlockWord()]));
 }
 // Owned by the selected coarse representation: no fine port is ever read.
 if(voxelVertexCoarse(v)){return;}
 let q=position(v,0u);let slots=contactSlots();let hash=contactPositionHash(q);
 for(var probe=0u;probe<slots;probe++){
  let at=(hash+probe)&(slots-1u);var first=atomicLoad(&contactWork[at]);
  // Retry only a spurious EMPTY exchange. Never wait for another invocation
  // to publish a key: the claimed word is already a complete source address.
  loop{if(first!=0u){break;}let claim=atomicCompareExchangeWeak(&contactWork[at],0u,v+1u);first=claim.old_value;if(claim.exchanged){first=v+1u;break;}}
  if(all(position(first-1u,0u)==q)){atomicMin(&contactWork[at],v+1u);break;}
 }
}
@compute @workgroup_size(64) fn contactEdges(@builtin(global_invocation_id) gid:vec3u){contactEdgeAt(gid.x+gid.y*params.owners[3].w*64u);}
fn contactEdgeAt(id:u32){
 if(!posedContacts()||id>=sourceEdgeCount()||!edgeInDraw(id)){return;}
 if(cellParts()){
  let code=partCode(halfedge(edge(id).x).x);let base=edgePartBase();
  atomicStore(&contactWork[base+id],code);
  // One edge per quad contributes its held whole-part level; all vertices
  // have completed the bound reduction. Flags commute with contact marking.
  if(id%5u==0u&&smallClosedPart(code)){let at=partSlot(code);if(at!=NO_CONTACT){atomicOr(&contactWork[partFlagsBase()+at],1u);}}
 }
 // Owned by the selected coarse representation: its contact list is never read;
 // it selects its clusters for chartEmitAt (perf-impl-119).
 if(voxelEdgeCoarse(id)){chartSelect(id);return;}
 let key=posedEdgePorts(id);if(key.x==key.y){return;}let slots=contactSlots();let hash=contactHash(key.x^contactHash(key.y));
 for(var probe=0u;probe<slots;probe++){
  let slot=(hash+probe)&(slots-1u);let at=slots+slot;var first=atomicLoad(&contactWork[at]);
  loop{if(first!=0u){break;}let claim=atomicCompareExchangeWeak(&contactWork[at],0u,id+1u);first=claim.old_value;if(claim.exchanged){first=id+1u;break;}}
  if(all(posedEdgePorts(first-1u)==key)){
   atomicMin(&contactWork[at],id+1u);
   let previous=atomicExchange(&contactWork[slots*2u+slot],id+1u);atomicStore(&contactWork[slots*3u+id],previous);
   // Mark the WHOLE part as connected, not just this edge. Read the original
   // immutable codes, not another invocation's not-yet-published work record.
   if(cellParts()&&previous!=0u){connectParts(partCode(halfedge(edge(id).x).x),partCode(halfedge(edge(previous-1u).x).x));}return;
  }
 }
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
 // perf-impl-108: a silhouette or crease end whose canonical vertex port is a
 // supplied contact corner of its owner carries that corner (id+1) in chart.w bits 3-12 (end a) and 13-22
 // (end b); the run stage keys that end's line node by the point
 // (gpu-construction-runs key: CONTACT_DOMAIN), so the two owners' edges meet.
 if(kind==0u||kind==2u){let owner=vec2u(params.domain,t.owner);chart.w+=f32((contactCorner(owner,ports.x)<<3u)|(contactCorner(owner,ports.y)<<13u));}
 relations[at]=Relation(vec4f(a,gradient.x),vec4f(b,gradient.y),vec4u(params.domain,t.owner,tag,address),ports,detail,chart);
}
@compute @workgroup_size(1) fn prepare(){
 let invalid=native[5]!=0u||sourceVertexCount()>params.vertexCapacity||native[2]<params.indexBase||native[2]-params.indexBase+native[0]>sourceFaceCount()*3u||(native[2]-params.indexBase)%3u!=0u||native[0]%3u!=0u;
 // A source/instance range is not capped at one dispatch dimension. Use
 // the device's actual X extent, then continue the SAME source IDs in Y.
 let limit=params.owners[3].w;
 let edges=select((sourceEdgeCount()*selectedCount()+63u)/64u,0u,invalid);let faces=select((sourceFaceCount()*selectedCount()+63u)/64u,0u,invalid);
 dispatch[0]=min(edges,limit);dispatch[1]=max(1u,(edges+limit-1u)/limit);dispatch[2]=1u;
 dispatch[3]=min(faces,limit);dispatch[4]=max(1u,(faces+limit-1u)/limit);dispatch[5]=1u;
 // Native GPU visibility/counts own the contact frontier too. Previously
 // every hidden chunk still cleared its complete hash and launched contact
 // work from registration capacity before the kernels noticed instanceCount0.
 let contactActive=!invalid&&posedContacts()&&native[0]!=0u;
 // perf-impl-110: the clear reaches the voxel cluster columns and the two
 // chunk words (columnEnd), the same extent contactClearAt clears and the host
 // allocates (contactWords); short of it, column claims, tops/floors and the
 // chunk level vote would persist into the next frame.
 let words=columnEnd();
 let clearGroups=select(0u,(words+63u)/64u,contactActive);
 let vertexGroups=select(0u,(min(sourceFaceCount()*2u,native[0]/6u*4u)+63u)/64u,contactActive);
 let edgeGroups=select(0u,(min(sourceEdgeCount(),native[0]/6u*5u)+63u)/64u,contactActive);
 dispatch[6]=min(clearGroups,limit);dispatch[7]=max(1u,(clearGroups+limit-1u)/limit);dispatch[8]=1u;
 dispatch[9]=min(vertexGroups,limit);dispatch[10]=max(1u,(vertexGroups+limit-1u)/limit);dispatch[11]=1u;
 dispatch[12]=min(edgeGroups,limit);dispatch[13]=max(1u,(edgeGroups+limit-1u)/limit);dispatch[14]=1u;
 if(invalid){atomicOr(&status.overflow,1u);}
}
@compute @workgroup_size(64) fn boundaries(@builtin(global_invocation_id) gid:vec3u){boundaryAt(gid.x+gid.y*params.owners[3].w*64u);}
fn boundaryAt(id:u32){
 if(id>=sourceEdgeCount()*selectedCount()){return;}
 chartEmitAt(id,sourceEdgeCount()*selectedCount());let instance=selectedInstance(id/sourceEdgeCount());if(!selected(instance)){return;}let e=edge(id%sourceEdgeCount());if(ownerUnseen(halfedge(e.x).x,instance)||ownerMerged(edgeOwner(halfedge(e.x).x,instance))){return;}
 // L2 (outlineLevel): this owner is drawn by its COMPILED outline construction
 // only. The thread of source edge k draws hull edge k; no fine boundary,
 // crease, fold or contour of the owner is produced.
 if(outlineLevel(edgeOwner(halfedge(e.x).x,instance))){hullOutlineAt(id%sourceEdgeCount(),instance);return;}
 // L1 (principalLevel): the thread of source edge k also draws chain k.
 let principal=principalLevel(edgeOwner(halfedge(e.x).x,instance));
 if(principal){principalChainAt(id%sourceEdgeCount(),instance);}
 if(!edgeInDraw(id%sourceEdgeCount())){return;}
 // perf-impl-110/113: a voxel chunk at a cluster level is drawn by its column
 // representation; this fine edge is then owned by it (not expanded), BEFORE
 // any posed contact lookup (its incidence was never built, perf-impl-113).
 if(posedContacts()){let inc=sourceEdgeCorner(halfedge(e.x).z,0u);if(inDraw(inc)&&validFace(inc)&&voxelColumnEdge(id%sourceEdgeCount(),triangle(inc,0u),inc/3u)){return;}}
 let contact=posedEdgeSlot(id%sourceEdgeCount());var canonical=canonicalEdge(id%sourceEdgeCount(),instance);
 if(contact!=NO_CONTACT){canonical=edge(atomicLoad(&contactWork[contactSlots()+contact])-1u).x;}
 if(e.x!=canonical){return;}
 var first=0xffffffffu;var second=0xffffffffu;var anyFace=0xffffffffu;var faces=0u;var frontCount=0u;var backCount=0u;var allSmooth=true;var crease=false;var firstNormal=vec3f(0.0);var nonzeroField=false;
 // Two back faces of a two-sided source (both render, turned to the eye).
 var backFirst=0xffffffffu;var backSecond=0xffffffffu;var backNormal=vec3f(0.0);var backCrease=false;
 let start=halfedge(e.x);
 var members=1u;var membersAt=0u;if(e.w>0u){let group=source[9]+(e.w-1u)*2u;membersAt=source[group];members=source[group+1u];}
 var member=0u;var contactHead=0u;var lastContact=NO_CONTACT;if(contact!=NO_CONTACT){contactHead=atomicLoad(&contactWork[contactSlots()*2u+contact]);}
 var samePart=true;var ownPart=0u;var smallPart=false;
 if(cellParts()){
  ownPart=atomicLoad(&contactWork[edgePartBase()+id%sourceEdgeCount()]);
  let at=partSlot(ownPart);if(ownPart!=0u&&at!=NO_CONTACT){smallPart=(atomicLoad(&contactWork[partFlagsBase()+at])&3u)==1u;}
 }
 loop{
  var h=e.x;
  if(contact!=NO_CONTACT){
   // Atomic list insertion order is not drawing order. Visit source edges
   // ascending so face ports and the normal reference cannot flip by race.
   var cursor=contactHead;var next=NO_CONTACT;
   loop{if(cursor==0u){break;}let candidate=cursor-1u;if(lastContact==NO_CONTACT||candidate>lastContact){next=min(next,candidate);}cursor=atomicLoad(&contactWork[contactSlots()*3u+candidate]);}
   if(next==NO_CONTACT){break;}lastContact=next;h=edge(next).x;
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
 if(id>=sourceFaceCount()*selectedCount()){return;}let instance=selectedInstance(id/sourceFaceCount());if(!selected(instance)){return;}let face=id%sourceFaceCount();if(ownerUnseen(halfedge(face*3u).x,instance)||ownerMerged(edgeOwner(halfedge(face*3u).x,instance))||outlineLevel(edgeOwner(halfedge(face*3u).x,instance))){return;}if(!inDraw(face*3u)||!validFace(face*3u)){return;}
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
// REPRESENTATION FRONTIER, stage 1 (perf-impl-92, 2026-11-17). An ENTITY
// instance of a source that draws this frame's field, owning no texel of
// that field, is not seen at this camera: its boundaries, creases and
// contours are not produced (they could only reach the page through the
// one-texel depth halo). Selected here, before any relation exists, from
// the same frame's form table. Producer rows (one draw, many owners) and
// sources the field does not attribute (flag bit 4 unset) keep everything.
// THE EDGE'S OWN OWNER (perf-impl-95). An entity instance owns all its
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
// FAR AGGREGATE MEMBER (perf-impl-96). A GPU producer row (one draw, many
// owners: voxel detail props, batched static rows) whose own visible form fits
// one native cell with the hold (farAggregateMember, gpu-source-far-forms.ts)
// produces no boundary, crease or occluding contour (boundaryAt, contourAt).
// The interior stages (suggestive, ridge, view-face lines) are NOT gated here:
// their bind groups do not bind the form table (perf-impl-96 first install
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
// COMPILED CONSTRUCTION LEVELS (representation frontier stage 2, perf-impl-93,
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
 let form=formExtent(owner);let extent=vec2f(f32(params.owners[1].w),f32(params.owners[2].x));
 if(!form.found||formCut(form.box,extent)||formSpansDepth(form.near,form.far,extent,detailWorldFromClip)){return false;}
 let row=bitcast<f32>(params.owners[1].y);
 // The L1 subset applies from the held principal level (formLevel).
 if(formLevel(owner)==0u){return false;}
 let unit=tierRowUnits(vec2f(form.box.z+form.box.w,form.box.x+form.box.y)*.5,form.near,extent,row,detailWorldFromClip);
 if(!(unit>0.0)){return false;}
 return length(b-a)*f32(code&${EDGE_LEVEL_DETAIL-1}u)/${EDGE_LEVEL_UNIT}.0<unit*${FACET_CREASE_LEAST_ROWS*LOD_TIER_HOLD};
}
fn interiorInstances()->u32{return min(params.owners[0].y,native[1]-min(native[1],params.owners[0].x));}
fn interiorCandidates()->u32{if(params.owners[3].x!=0u){return params.owners[2].z;}return interiorInstances();}
fn interiorCandidate(ordinal:u32)->u32{if(params.owners[3].x!=0u){return selectedInstance(ordinal);}return params.owners[0].x+ordinal;}
fn detailBase()->u32{return (sourceFaceCount()*3u+sourceVertexCount()*3u)*interiorInstances();}
fn viewInteriorInstances()->u32{if((params.owners[1].z&8u)!=0u){return bitcast<u32>(differentials[detailBase()].x);}return interiorCandidates();}
fn viewInteriorInstance(ordinal:u32)->u32{if((params.owners[1].z&8u)!=0u){return bitcast<u32>(differentials[detailBase()+1u+ordinal].x);}return interiorCandidate(ordinal);}
fn radialBase()->u32{return sourceFaceCount()*interiorInstances()*3u;}
// Early structural LOD, using the final drawing's SAME form table, extent,
// compact-form rule and hysteresis boundary. Unknown/cut/spanning forms,
// terrain and enclosed/parallel views keep the complete interior producer.
fn needsViewCurvature(instance:u32)->bool{
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
// THE ONE HELD LEVEL (perf-impl-107 reader hookup; writer: form-depths.ts
// resolveLod, Codex): 0 near/full, 1 principal, 2 connected outline, held per
// form and camera with separate entry/leave bands, read from the SAME form
// texture (no new binding). The header texel's w word says the writer
// published held levels this frame; without it every reader keeps the
// perf-106 plain rule (rows against the lower hold edge), unchanged.
fn formLodActive()->bool{return textureLoad(formDepths,vec2i(0,${FORM_DEPTH_HEADER_ROW}),0).w!=0u;}
fn formLevel(owner:vec2u)->u32{
 let rows=levelRows(owner);if(rows<0.0){return 0u;}
 if(formLodActive()){return formHeldLod(owner,true);}
 return select(select(1u,2u,rows<${FAR_STRIP_BELOW_ROWS*LOD_TIER_HOLD}),0u,!(rows<${NEAR_FROM_ROWS*LOD_TIER_HOLD}));
}
fn farStripForm(owner:vec2u)->bool{return formLevel(owner)==2u;}
// L1, THE PRINCIPAL CONSTRUCTION (compilePrincipalChains, perf-impl-106): a
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
 if(!inDraw(c[2]*3u)||!validFace(c[2]*3u)||!inDraw(c[3]*3u)||!validFace(c[3]*3u)){return;}
 let t0=triangle(c[2]*3u,instance);let t1=triangle(c[3]*3u,instance);
 // The same crease rule as boundaryAt: both planes turned to the eye and
 // turning 45 degrees or more; a chain on the silhouette is drawn by its
 // members' silhouettes (kind 0 is never chained).
 if(dot(t0.face,t0.face)<0.1||dot(t1.face,t1.face)<0.1||!front(t0)||!front(t1)||dot(t0.face,t1.face)>=0.70710678){return;}
 emit(position(c[0],instance),position(c[1],instance),t0,2u,c[4],vec4u(boundaryPort(c[0],instance),boundaryPort(c[1],instance),c[2],c[3]),vec4f(0),false);
}
// L2, THE CONNECTED OUTLINE (structural ladder, perf-impl-106). An owner at the
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
 let instance=params.owners[0].x+id/sourceFaceCount();let face=id%sourceFaceCount();let at=id*3u;
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
 let instance=viewInteriorInstance(id/sourceVertexCount());let local=instance-params.owners[0].x;let vertex=id%sourceVertexCount();let out=radialBase()+local*sourceVertexCount()+vertex;
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
 let instance=viewInteriorInstance(id/sourceFaceCount());let local=instance-params.owners[0].x;let face=id%sourceFaceCount();
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
// RIDGES AND VALLEYS (operator order 2026-11-11, GOAL-EXECUTOR ledger: interior
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
 let local=id/sourceVertexCount();let instance=params.owners[0].x+local;let vertex=id%sourceVertexCount();
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
 let local=id/sourceVertexCount();let vertex=id%sourceVertexCount();let instance=params.owners[0].x+local;
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
 let instance=viewInteriorInstance(id/sourceFaceCount());let local=instance-params.owners[0].x;let face=id%sourceFaceCount();
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
// FUSED SOURCE STAGES (perf-impl-76, frame cost 2026-11-17). On the M3 Pro a
// relation dispatch costs ~15 us of fixed time and the stages of ~137 in-view
// sources ran as ~815 serial dispatches (14 ms of a 30 ms job). Stages of ONE
// source that read nothing another of them writes share one dispatch: each
// thread runs the same per-id body as the separate entry above, so the set of
// emitted relations is unchanged; only their append order interleaves.
// surfaceLines: boundaries over edges x instances, then contours over faces x
// instances, as one id range (workgroup g covers ids 64g..64g+63).
@compute @workgroup_size(64) fn surfaceLines(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) local:u32){
 let id=(wid.x+wid.y*params.owners[3].w)*64u+local;let edges=sourceEdgeCount()*selectedCount();
 if(id<edges){boundaryAt(id);}else{contourAt(id-edges);}
}
// viewFaceLines: suggestive contours and ridge lines over the same view faces
// (both read radial/principal/ridge words written by EARLIER dispatches).
@compute @workgroup_size(64) fn viewFaceLines(@builtin(global_invocation_id) gid:vec3u){
 let id=gid.x+gid.y*params.owners[0].z*64u;suggestiveAt(id);ridgeLineAt(id);
}
`;
// ONE DISPATCH PER STAGE OVER EVERY SOURCE (perf-impl-79, frame cost
// 2026-11-17). The ~541 per-source dispatches of a Y8 view (~17 us each) become
// one dispatch per stage per group of sources that share storage pages. The
// stage bodies are the SAME functions as the per-source entries above; only
// the binding offsets of the per-source path become per-source word bases:
// each source-range array (source incidence, posed positions/attributes,
// batch metadata, contact scratch, posed tensor, dispatch words) is bound as
// its whole storage page and indexed at base + the same index. A thread finds
// its source in a uniform table by its id (sources are contiguous id ranges,
// one thread per relation candidate exactly as in the per-source dispatch).
export const SURFACE_RELATION_BATCH_ITEMS=256;
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
export const SURFACE_RELATIONS_BATCH_SHADER=rebaseSourceArrays(stripWgslComments(SURFACE_RELATIONS_SHADER).replace(PARAMS_BINDING,''))+/* wgsl */`
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
${batchEntry('batchContactClear','contactClearAt(id);')}
${batchEntry('batchVoxelVote','voxelVoteAt(id);')}
${batchEntry('batchVoxelBin','voxelBinAt(id);')}
${batchEntry('batchChartCorners','chartCornerAt(id);')}
${batchEntry('batchChartCertify','chartCertifyAt(id);')}
${batchEntry('batchPartAnchorClear','partAnchorClearAt(id);')}
${batchEntry('batchPartAnchors','partAnchorAt(id);')}
${batchEntry('batchPartAnchorInsert','partAnchorInsertAt(id);')}
${batchEntry('batchContactVertices','contactVertexAt(id);')}
${batchEntry('batchContactEdges','contactEdgeAt(id);')}
${batchEntry('batchSurfaceLines','let edges=sourceEdgeCount()*selectedCount();if(id<edges){boundaryAt(id);}else{contourAt(id-edges);}')}
@compute @workgroup_size(64) fn batchPrepareInterior(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) local:u32){batchEnter(wid.x);prepareInteriorIn(local);}
${batchEntry('batchFaceDifferentials','if(batchGate(6u)){faceDifferentialAt(id);}')}
${batchEntry('batchRadialCurvature','radialCurvatureAt(id);')}
${batchEntry('batchSuggestiveContours','suggestiveAt(id);')}
${batchEntry('batchPrincipalCurvature','if(batchGate(9u)){principalCurvatureAt(id);}')}
${batchEntry('batchRidgeDerivative','if(batchGate(9u)){ridgeDerivativeAt(id);}')}
${batchEntry('batchRidgeLines','ridgeLineAt(id);')}
${batchEntry('batchViewFaceLines','suggestiveAt(id);ridgeLineAt(id);')}
`;
type BatchStage='partAnchorClear'|'partAnchors'|'partAnchorInsert'|'contactClear'|'voxelVote'|'voxelBin'|'chartCorners'|'chartCertify'|'contactVertices'|'contactEdges'|'surfaceLines'|'prepareInterior'|'faceDifferentials'|'radialCurvature'|'suggestiveContours'|'principalCurvature'|'ridgeDerivative'|'ridgeLines'|'viewFaceLines';
/** Global stage order. Per source it is the per-source order (contacts, lines,
 * prepareInterior, tensor, view lines); sources write disjoint ranges. */
const BATCH_ORDER:readonly BatchStage[]=['partAnchorClear','partAnchors','partAnchorInsert','contactClear','voxelVote','voxelBin','chartCorners','chartCertify','contactVertices','contactEdges','surfaceLines','prepareInterior','faceDifferentials','radialCurvature','suggestiveContours','principalCurvature','ridgeDerivative','ridgeLines','viewFaceLines'];
const batchEntryName=(stage:BatchStage)=>'batch'+stage[0]!.toUpperCase()+stage.slice(1);
const BATCH_BINDINGS:Record<BatchStage,readonly number[]>={partAnchorClear:[0,17],partAnchors:[0,2,3,4,8,16,17],partAnchorInsert:[0,4,16,17],contactClear:[0,1,4,12],voxelVote:[0,1,2,4,12,13],voxelBin:[0,1,2,4,12,13],chartCorners:[0,1,2,4,12,13],chartCertify:[0,1,4,12],contactVertices:[0,1,2,4,12],contactEdges:[0,1,2,4,12,13,14],surfaceLines:[0,1,2,3,4,5,6,8,10,11,12,15],prepareInterior:[0,1,2,3,4,6,7,8,9,10,11],faceDifferentials:[0,1,2,3,4,6,7,8,9],radialCurvature:[0,1,2,3,4,8,9],suggestiveContours:[0,1,2,3,4,5,6,8,9,15],principalCurvature:[0,1,2,3,4,7,8,9],ridgeDerivative:[0,1,2,4,7,9],ridgeLines:[0,1,2,3,4,5,6,8,9,15],viewFaceLines:[0,1,2,3,4,5,6,8,9,15]};
const BATCH_ITEM_WORDS=60,BATCH_TABLE_BYTES=16+SURFACE_RELATION_BATCH_ITEMS*BATCH_ITEM_WORDS*4,BATCH_DISPATCH_WORDS=18;
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
interface BatchItem{readonly stage:BatchStage;readonly threads:number;readonly words:Uint32Array;readonly base:readonly number[];readonly base2:readonly number[];readonly bound:RelationBinding[]}
interface BatchQueue{readonly items:BatchItem[];readonly clears:GPUBufferBinding[];readonly seen:Set<object>;contactWords:number;dispatchItems:number}

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
/** perf-impl-114: one chunk's persistent occupancy home (word offset in contactWork). */
/** perf-impl-116: a closed-cell source's CELL CHART sizes (dense tables over its cell extent), or undefined when
 * they would exceed CHART_CELLS (that source keeps its fine producer). cells: per-cell face centroids (19 words);
 * corners: lattice points (4 words); clusters: at block 2, the largest table any block needs (4 words). */
export const CHART_CELLS=32768;
/** perf-impl-140: words per part anchor record [owner.x, owner.y, part ordinal (0 = none), link (0x80000000|next record+1), min.x, min.y, max.x, max.y]. */
export const PART_ANCHOR_RECORD_WORDS=8;
/** perf-impl-140 descriptor: the owner-keyed NDC extent table of source-declared part anchors. Word 0 = head slot
 * count, word 1 = record word base; head h (at headBase+h) holds record+1 of a bucket list (0 = empty). Key =
 * (domain|producer, native owner, part ordinal = packet program index + 1): a run's (identity.y, identity.z) and its
 * program ordinal. Bucket = 2 + (hash & (headSlots-1)) with the WGSL partAnchorBucket hash. Valid for the frame
 * after SurfaceRelations' dispatches; readers walk the list and compare the full key. */
export interface SourcePartAnchorTable {readonly buffer:GPUBuffer;readonly headSlots:number;readonly headBase:number;readonly recordBase:number;readonly recordWords:number}
const samePartAnchorVertices=(a:Uint32Array|undefined,b:Uint32Array|undefined):boolean=>a===b||(!!a&&!!b&&a.length===b.length&&a.every((value,index)=>value===b[index]));
/** perf-impl-118: a surface source's rest-cell chart, only for the topology generation it names (stale: absent). */
export function currentRestCellChart(indexed:{readonly generation:number;readonly restCellChart?:GpuIndexedRestCellChart}|undefined):GpuIndexedRestCellChart|undefined{
  const chart=indexed?.restCellChart;return chart&&chart.kind==='voxel-rest-cell-chart'&&chart.sourceGeneration===indexed!.generation?chart:undefined;
}
export function cellChart(extent:readonly [number,number,number]):{cells:number;corners:number;clusters:number}|undefined{
  const [x,y,z]=extent;if(![x,y,z].every(n=>Number.isInteger(n)&&n>=1&&n<=256))return undefined;
  const cells=x*y*z;if(cells>CHART_CELLS)return undefined;
  return {cells,corners:(x+1)*(y+1)*(z+1),clusters:Math.ceil(x/2)*Math.ceil(y/2)*Math.ceil(z/2)};
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
interface ColumnHome {readonly offset:number;readonly slots:number;source:unknown;fresh:boolean}
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
/** perf-impl-115: the key under which a source's view-independent shape tensors (faceDifferentials,
 * principalCurvature, ridgeDerivative) may be REUSED, or undefined when they can never be. A GPU-indexed pose's
 * posedRevision is a per-capture serial (geometry.ts), so only the producer's exact posed CONTENT token
 * (posedContentToken > 1) can repeat; without it the tensor refreshes every frame and is computed only for
 * instances that consume it this frame. Every other source keeps its posedRevision (unchanged). */
export function tensorRevision(batch:{readonly source:{readonly posedRevision?:number};readonly gpuPose?:{readonly posedContentRevision?:number}}):number|undefined{
  if(batch.source.posedRevision===undefined)return undefined;
  if(!batch.gpuPose)return batch.source.posedRevision;
  const token=posedContentToken(batch);return token>1?token:undefined;
}
const SURFACE_RELATION_ENTRIES=['prepare','partAnchorClear','partAnchors','partAnchorInsert','contactClear','voxelVote','voxelHome','voxelBin','chartCorners','chartCertify','contactVertices','contactEdges','boundaries','contours','surfaceLines'] as const;
const SURFACE_RELATION_INTERIOR_ENTRIES=['prepareInterior','faceDifferentials','radialCurvature','suggestiveContours','principalCurvature','ridgeDerivative','ridgeLines','viewFaceLines'] as const;
// The batched entries exist only with the interior stages (the physical path).
const SURFACE_RELATION_BATCH_ENTRIES=BATCH_ORDER.map(batchEntryName);
type SurfaceRelationEntry=typeof SURFACE_RELATION_ENTRIES[number]|typeof SURFACE_RELATION_INTERIOR_ENTRIES[number]|string;
const surfaceRelationEntries=(interior:boolean):readonly SurfaceRelationEntry[]=>interior?[...SURFACE_RELATION_ENTRIES,...SURFACE_RELATION_INTERIOR_ENTRIES,...SURFACE_RELATION_BATCH_ENTRIES]:SURFACE_RELATION_ENTRIES;
interface PreparedSurfaceRelations{
 readonly plans:ReadonlyMap<SurfaceRelationEntry,ComputePipelinePlan>;
 readonly pipelines:ReadonlyMap<SurfaceRelationEntry,GPUComputePipeline>;
}
function surfaceRelationPipelinePlans(device:GPUDevice,interior:boolean):ReadonlyMap<SurfaceRelationEntry,ComputePipelinePlan>{
 const plans=new Map<SurfaceRelationEntry,ComputePipelinePlan>();
 for(const entry of surfaceRelationEntries(interior)){
  const batched=entry.startsWith('batch');
  plans.set(entry,()=>{const module=device.createShaderModule({label:'MAIN attributed source relations '+entry,code:isolateComputeEntry(batched?SURFACE_RELATIONS_BATCH_SHADER:SURFACE_RELATIONS_SHADER,entry)});
   return {layout:'auto',compute:{module,entryPoint:entry}};});
 }
 return plans;
}
export class SurfaceRelations {
  lines:GPUBuffer;readonly status:GPUBuffer;
  private readonly dispatch:GPUBuffer;
  private readonly pipelines:Record<string,GPUComputePipeline>={};
  private pending:GPUBuffer[]=[];
  private retired:GPUBuffer[]=[];
  private differentials?:GPUBuffer;
  private contactWork?:GPUBuffer;
  /** perf-impl-114: persistent occupancy homes in contactWork, after homeScratch words. */
  /** perf-impl-137/140: lazily registered source-declared part anchors per material batch (posedBatchIdentity),
   * retained while the residency source and its vertex array are unchanged, and the ONE owner-keyed extent table. */
  private readonly partAnchorSources=new Map<unknown,{source:unknown;vertices:Uint32Array;input:GPUBuffer;base:number;parts:number;instances:number}>();
  private partAnchorBuffer?:GPUBuffer;private partAnchorHeadSlots=0;private partAnchorRecords=0;private partAnchorRecordCapacity=0;
  private partAnchorFrameCleared=false;private partAnchorGrowTo=0;
  /** The owner-keyed extent table (undefined until a source registers). Consumers re-read it each frame. */
  get partAnchorTable():SourcePartAnchorTable|undefined{
    return this.partAnchorBuffer?{buffer:this.partAnchorBuffer,headSlots:this.partAnchorHeadSlots,headBase:2,recordBase:2+this.partAnchorHeadSlots,recordWords:PART_ANCHOR_RECORD_WORDS}:undefined;
  }
  /** Register (or keep) a batch's declared anchors from its residency source; returns the entry, or undefined. */
  private partAnchorsFor(batch:PosedSourceBatch,source:ResidentRelationSource):{input:GPUBuffer;parts:number;instances:number}|undefined{
    const vertices=(source as {partAnchorVertices?:Uint32Array}).partAnchorVertices;const key=posedBatchIdentity(batch);
    const instances=Math.max(1,batch.source.instanceCapacity??1);
    const held=this.partAnchorSources.get(key);
    // perf-impl-142: physical wrapper sources change with pose; retain by anchor CONTENTS + instance capacity.
    if(held&&samePartAnchorVertices(held.vertices,vertices)&&held.instances===instances){held.source=source;this.expiry.use(held);return held;}
    if(held){this.partAnchorSources.delete(key);this.retired.push(held.input);}
    if(!vertices||vertices.length<6||vertices.length%6!==0)return undefined;
    const parts=vertices.length/6;
    if(this.partAnchorRecords+parts*instances>this.partAnchorRecordCapacity){
      // perf-impl-142: never grow mid-frame (earlier sources' records and queued references stay valid);
      // the first table has no references yet. Otherwise skip this source until the frame boundary.
      if(this.partAnchorBuffer){this.partAnchorGrowTo=Math.max(this.partAnchorGrowTo,this.livePartAnchorRecords()+parts*instances);return undefined;}
      this.growPartAnchorTable(parts*instances);
    }
    const base=this.partAnchorRecords;this.partAnchorRecords+=parts*instances;
    const data=new Uint32Array(2+vertices.length);data[0]=base;data[1]=parts;data.set(vertices,2);
    const input=this.device.createBuffer({label:'MAIN source-declared part anchors',size:data.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
    this.device.queue.writeBuffer(input,0,data);
    const entry={source,vertices,input,base,parts,instances};this.partAnchorSources.set(key,entry);
    this.expiry.use(entry,()=>{if(this.partAnchorSources.get(key)!==entry)return;this.partAnchorSources.delete(key);this.retired.push(entry.input);});
    return entry;
  }
  /** A larger table restarts every registration (records are re-allocated lazily by later draws). */
  /** perf-impl-144: pre-reserve the part anchor table for every batch a frame will draw, BEFORE its first draw.
   * Grows immediately when nothing this frame references the table yet (no anchor dispatch encoded or queued);
   * otherwise the growth waits for the frame boundary, so earlier sources and queued references stay valid. */
  reservePartAnchors(batches:Iterable<{readonly batch:PosedSourceBatch;readonly source:ResidentRelationSource}>):void{
    let additional=0,replaced=0;const seen=new Set<object>();
    for(const {batch,source}of batches){
      const key=posedBatchIdentity(batch);if(seen.has(key))continue;seen.add(key);
      const vertices=(source as {partAnchorVertices?:Uint32Array}).partAnchorVertices;if(!vertices||vertices.length<6||vertices.length%6!==0)continue;
      const instances=Math.max(1,batch.source.instanceCapacity??1),held=this.partAnchorSources.get(key);
      if(held&&samePartAnchorVertices(held.vertices,vertices)&&held.instances===instances)continue;
      additional+=vertices.length/6*instances;replaced+=held?held.parts*held.instances:0;
    }
    const needed=this.partAnchorRecords+additional;
    if(needed<=this.partAnchorRecordCapacity)return;
    // Only at exhaustion: compact dead slots instead of growing from the
    // historical high-water mark. Ordinary frames never scan this cache.
    const liveNeeded=this.livePartAnchorRecords()+additional-replaced;
    const referenced=this.partAnchorFrameCleared||(this.queue?.items.some(item=>item.stage==='partAnchors')??false);
    if(!this.partAnchorBuffer||!referenced)this.growPartAnchorTable(liveNeeded);else this.partAnchorGrowTo=Math.max(this.partAnchorGrowTo,liveNeeded);
  }
  private livePartAnchorRecords():number{
    let records=0;for(const entry of this.partAnchorSources.values())records+=entry.parts*entry.instances;return records;
  }
  private growPartAnchorTable(records:number):void{
    let capacity=1024;while(capacity<records*2)capacity*=2;
    let heads=1024;while(heads<capacity)heads*=2;
    if(this.partAnchorBuffer)this.retired.push(this.partAnchorBuffer);
    for(const entry of this.partAnchorSources.values())this.retired.push(entry.input);
    this.partAnchorSources.clear();this.partAnchorRecords=0;this.partAnchorRecordCapacity=capacity;this.partAnchorHeadSlots=heads;
    this.partAnchorBuffer=this.device.createBuffer({label:'MAIN owner-keyed part anchor extents',size:(2+heads+capacity*PART_ANCHOR_RECORD_WORDS)*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
    this.device.queue.writeBuffer(this.partAnchorBuffer,0,new Uint32Array([heads,2+heads]));this.partAnchorFrameCleared=false;
  }
  private readonly homes=new Map<unknown,ColumnHome>();private homeScratch=0;private homeWords=0;
  private intersections?:SurfaceIntersections;
  private contactsActive=false;
  private detailField?:SurfaceField;
  private detailPage?:readonly [number,number];
  private emptyDetail?:GPUTexture;
  private emptyDetailView?:GPUTextureView;
  /** perf-impl-92: source domains whose material draws own texels of the
   * current field (set by the caller after begin; cleared by begin). Only
   * their entity instances may be found unseen in the form table. */
  fieldDomains?:ReadonlySet<number>;
  /** The form table the line stages read (binding 10); a stable empty
   * view when this frame has none (flag bit 0 unset, never read). */
  private lineDetailView(earlyDetail:boolean):GPUTextureView{
    if(earlyDetail)return formDepths(this.device).view(this.detailField!);
    return this.emptyDetailView??=(this.emptyDetail??=this.device.createTexture({label:'MAIN unselected detail table',size:[1,1],format:'rgba32uint',usage:GPUTextureUsage.TEXTURE_BINDING})).createView();
  }
  private readonly retainedDifferentials=new Map<object,Map<number,{revision:number;buffer:GPUBuffer}>>();
  static async create(device:GPUDevice,lineCapacity=32768,interior=false,progress?:(fraction:number)=>void):Promise<SurfaceRelations>{
   const plans=surfaceRelationPipelinePlans(device,interior),pipelines=await prepareComputePipelines(device,plans,progress);
   return new SurfaceRelations(device,lineCapacity,interior,{plans,pipelines});
  }
  constructor(private readonly device:GPUDevice,public lineCapacity=32768,private readonly interior=false,prepared?:PreparedSurfaceRelations){
    this.lines=device.createBuffer({size:lineCapacity*RELATION_WORDS*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
    this.status=device.createBuffer({size:16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
    this.dispatch=device.createBuffer({size:72,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.INDIRECT});
    const plans=prepared?.plans??surfaceRelationPipelinePlans(device,interior),pipelines=prepared?.pipelines??compileComputePipelines(device,plans);
    for(const entryPoint of surfaceRelationEntries(interior))this.pipelines[entryPoint]=pipelines.get(entryPoint)!;
  }
  /** Grow storage for an already counted, retained source submission. This
   * changes allocation only: it never limits source owners or source ranges.
   * The caller owns the submitted-frame fence before replaying its SAME pose. */
  ensureCapacity(count:number):void{
    if(!Number.isSafeInteger(count)||count<0)throw Error('Invalid complete source relation count');
    if(count<=this.lineCapacity)return;
    const capacity=Math.ceil(count/256)*256,size=capacity*RELATION_WORDS*4;
    if(size>Math.min(this.device.limits.maxBufferSize,this.device.limits.maxStorageBufferBindingSize))throw Error(`Complete source relation range (${count}) needs device-sized chunks`);
    const next=this.device.createBuffer({label:'MAIN complete counted source relations',size,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
    this.retired.push(this.lines);this.lines=next;this.lineCapacity=capacity;
  }
  /** `contactScale`: passed to SurfaceIntersections.begin (id raster columns per field column). */
  begin(encoder:GPUCommandEncoder,view?:SurfaceGraphView,field?:SurfaceField,contactScale=1):void{
    encoder.clearBuffer(this.status);this.contactsActive=!!view&&!!field;
    this.detailField=field;this.detailPage=view?.pageExtent;this.fieldDomains=undefined;
    if(view&&field)(this.intersections??=new SurfaceIntersections(this.device)).begin(encoder,view,field,this.status,contactScale);
    // perf-impl-79: host-sized physical sources appended after this are queued
    // and dispatched together in finish(), before the contact-triangle finish.
    this.queue=this.interior?{items:[],clears:[],seen:new Set(),contactWords:0,dispatchItems:0}:undefined;
  }
  ensureContactCapacity(count:number):void{(this.intersections??=new SurfaceIntersections(this.device)).ensureCapacity(count);}
  finish(encoder:GPUCommandEncoder):void{this.flushBatched(encoder);if(this.contactsActive)this.intersections!.finish(encoder,this.lines,this.status,this.lineCapacity);this.contactsActive=false;}
  /** The contact producer's passes for one source (append does this itself
   * when it opens its own pass). */
  appendContacts(encoder:GPUCommandEncoder,batch:PosedSourceBatch,source:ResidentRelationSource,domain:number):void{
    if(this.contactsActive)this.intersections!.append(encoder,batch,source,domain);
  }
  /** Every source's contact passes as two passes (surface-intersections appendAll). */
  /** perf-impl-95: the contact producer reads the same field attribution. */
  appendContactsAll(encoder:GPUCommandEncoder,items:readonly {batch:PosedSourceBatch;source:ResidentRelationSource;domain:number;extent?:SourceDrawExtent}[]):void{
    if(this.contactsActive){this.intersections!.fieldDomains=this.fieldDomains;this.intersections!.page=this.detailPage;this.intersections!.appendAll(encoder,items);}
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
    const owners=Uint32Array.from([...new Set((selectedOwners??[]).map(owner=>owner>>>0))].sort((a,b)=>a-b));
    // RETAINED PER-SOURCE BINDINGS (frame cost, 2026-11-11): the parameter
    // block, owner selection and bind groups of a source are kept between
    // frames; only the 192 parameter bytes are rewritten. Rebuilding them was
    // ~4300 bind groups and ~1900 buffers per Y8 text frame, each validated in
    // the GPU process, which held a frame ~130 ms from submit to completion
    // while the GPU itself was busy 29% of the time.
    // ONE DISPATCH PER STAGE (perf-impl-79): a host-sized physical source with
    // GPU instance visibility and a retained posed tensor is queued; finish()
    // dispatches every queued source's stage at once. A second use of the same
    // batch in one frame keeps the per-source path (its tensor would be shared).
    const batched=!!shared&&!!extent&&this.interior&&!!this.queue&&!!batch.source.visibleInstances&&batch.source.posedRevision!==undefined&&!this.queue.seen.has(posedBatchIdentity(batch));
    const entry=batched?undefined:this.bindings(source,batch);
    const ownerKey=selectedOwners===undefined?'':owners.join(',');
    let selection=batch.source.visibleInstances;
    if(!selection&&entry){
      if(!entry.selection||entry.ownerKey!==ownerKey){if(entry.selection)this.retired.push(entry.selection);entry.selection=this.device.createBuffer({size:Math.max(4,owners.byteLength),usage:GPUBufferUsage.STORAGE,mappedAtCreation:true});new Uint32Array(entry.selection.getMappedRange()).set(owners);entry.selection.unmap();entry.ownerKey=ownerKey;}
      selection=entry.selection;
    }
    const floats=new Float32Array(48);floats.set(view.clipFromWorld);floats.set([...view.eye.slice(0,3),Number(view.perspective)],16);floats.set([...view.direction.slice(0,3),batch.source.cullMode??0],20);
    new Uint32Array(floats.buffer).set([this.lineCapacity,domain,batch.source.vertexCapacity,batch.source.indexBase??0],24);new Uint32Array(floats.buffer).set([owners.length,Number(selectedOwners!==undefined),Number(!!batch.source.visibleInstances),Number(batch.source.posedRevision!==undefined)],28);
    new Uint32Array(floats.buffer)[46]=batch.source.visibleInstances?(batch.source.visibleInstanceOffset??0):0;new Uint32Array(floats.buffer)[47]=this.device.limits.maxComputeWorkgroupsPerDimension;
    new Uint32Array(floats.buffer)[36]=isGpuIndexedResident(source)?Number(source.gpu.constructionKind==='planar'):planarRest(source.geometry)?1:0;
    const page=this.detailPage,field=this.detailField;
    const terrain=isGpuIndexedResident(source)?source.gpu.constructionKind==='terrain':source.geometry.surface==='heightfield'||('analysis' in source&&(source.analysis as {type?:string}).type==='terrain');
    const earlyDetail=!!field&&!!page&&page[0]>0&&page[1]>0;
    if(earlyDetail){floats[37]=17*field!.height/page![1];new Uint32Array(floats.buffer)[39]=field!.width;new Uint32Array(floats.buffer)[40]=field!.height;}
    new Uint32Array(floats.buffer)[38]=Number(earlyDetail)|(Number(terrain)<<1)|(Number(tensorRevision(batch)!==undefined)<<2)|(Number(earlyDetail&&!terrain)<<3)|(Number(earlyDetail&&!!this.fieldDomains?.has(domain))<<4);
    const frontier=batch.source.visibleInstanceFrontier;
    if(frontier){const u=new Uint32Array(floats.buffer);u[42]=frontier.indices.length;u[43]=frontier.offset;u[44]=1;}
    const indexed=isGpuIndexedResident(source)?source.gpu:undefined;
    const posedContacts=indexed?.edgeContactClass==='posed-coincident'&&indexed.topology.indexPattern==='independent-quads'&&batch.source.instanceCapacity===1;
    const contactEdges=posedContacts?sourceResidencyEdgeCount(source):0;
    const parts=posedContacts&&earlyDetail?indexed?.parts:undefined;
    let contactSlots=0,contactWords=1;
    // perf-impl-116/118: the cell chart of a closed-cell source (parts) or of a surface-only source (the producer's
    // rest-cell chart of THIS generation), when its dense tables are bounded. Never both: parts win.
    const surfaceChart=posedContacts&&earlyDetail&&!parts?currentRestCellChart(indexed):undefined;
    const chart=posedContacts?(parts?cellChart(parts.extent):surfaceChart?cellChart(surfaceChart.extent):undefined):undefined;
    const chartCodes:SourceStorageBinding|undefined=parts?.cellCodes??(chart?surfaceChart?.vertexCellCodes:undefined);
    new Uint32Array(floats.buffer)[31]=chartToken(batch,source,!!chart);
    if(posedContacts){
      contactSlots=2**Math.ceil(Math.log2(Math.max(2,contactEdges*2,sourceResidencyFaceCount(source)*4)));
      // + contactSlots: the voxel cluster claim words (clusterBase, perf-impl-106).
      // perf-impl-110: + the voxel cluster columns (columnSlots x 8) and the seven chunk words (WGSL columnEnd).
      contactWords=(parts?contactSlots*7+contactEdges*2:contactSlots*3+contactEdges)+contactSlots+4+(chart?chart.cells*19+chart.corners*4+4+chart.clusters*4+sourceResidencyFaceCount(source)*2:4);
      const words=new Uint32Array(floats.buffer);words[41]=contactSlots;words[45]=1;
      if(parts){const [x,y,z]=parts.extent;words[45]=3|(chart?4:0)|((x-1)<<8)|((y-1)<<16)|((z-1)<<24);}
      // Surface: the chart (4) with explicit surface semantics (8); NEVER the closed-cell part bit (2).
      else if(chart&&surfaceChart){const [x,y,z]=surfaceChart.extent;words[45]=1|4|8|((x-1)<<8)|((y-1)<<16)|((z-1)<<24);}
      // perf-impl-121: the producer certified its mesher quad corner order (surface and closed charts).
      if(chart&&currentRestCellChart(indexed)?.cornerOrder===VOXEL_REST_CORNER_ORDER)words[45]|=16;
    }
    if(batched){this.queueBatched(batch,source,new Uint32Array(floats.buffer),extent!,{posedContacts,contactWords,contactEdges,parts,chartCodes,chartClusters:chart?.clusters??0,earlyDetail,terrain,selection:selection!});return;}
    const held=entry!,params=held.params;
    // perf-impl-114 OCCUPANCY HOMES. The buffer is [scratch (homeScratch words,
    // grow-only) | homes]. A source whose producer certifies its posed content
    // (posedContentToken > 1) gets a home sized from its column extent: the
    // closed-cell extent at the coarsest-but-one block (2), or one column per
    // 4 quads; twice that in slots. A full table is an overflow (the chunk
    // keeps its fine producer). Growing the scratch or the buffer, or a changed
    // source, restarts the homes (a new header, an invalid token).
    let home:ColumnHome|undefined;
    if(posedContacts&&contactWords>this.homeScratch){this.homeScratch=Math.ceil(contactWords*1.25);this.homes.clear();this.homeWords=0;}
    if(chart&&chartToken(batch,source,true)>1){
      const slots=chart.clusters,key=posedBatchIdentity(batch);
      home=this.homes.get(key);
      if(home&&(home.source!==source||home.slots!==slots)){if(home.slots===slots){home.source=source;home.fresh=true;}else home=undefined;}
      if(!home){
        if(this.homeWords+4+slots*4>COLUMN_HOME_WORDS){this.homes.clear();this.homeWords=0;}
        home={offset:this.homeScratch+this.homeWords,slots,source,fresh:true};this.homeWords+=4+slots*4;this.homes.set(key,home);
      }
    }
    const required=Math.max(contactWords,this.homes.size?this.homeScratch+this.homeWords:0);
    if(!this.contactWork||this.contactWork.size<required*4){
      if(this.contactWork)this.retired.push(this.contactWork);
      const size=this.homes.size?Math.ceil(required*1.25):required;
      this.contactWork=this.device.createBuffer({label:'MAIN source-local posed quad contacts',size:size*4,usage:GPUBufferUsage.STORAGE});
      // A new buffer is zero: every home's header is written again before use.
      for(const record of this.homes.values())record.fresh=true;
    }
    if(home?.fresh){this.device.queue.writeBuffer(this.contactWork,home.offset*4,new Uint32Array([home.slots,0,0,0]));home.fresh=false;}
    new Uint32Array(floats.buffer)[35]=home?.offset??0;
    const parameterWords=this.interior?new Uint32Array(floats.buffer).slice():undefined;this.device.queue.writeBuffer(params,0,floats);
    const buffers=[params,indexed?.incidence.binding??source.bufferBinding??source.buffer,batch.storage?.vertices??batch.vertices,batch.storage?.attributes??batch.attributes,batch.storage?.metadata??batch.metadata,this.lines,this.status,this.dispatch,selection] as SourceStorageBinding[];
    buffers[12]=this.contactWork;
    buffers[13]=chartCodes??batch.storage?.metadata??batch.metadata;
    buffers[14]=parts?.levels??this.status;
    const used:Record<string,number[]>={prepare:[0,1,4,6,7],partAnchorClear:[0,17],partAnchors:[0,2,3,4,8,16,17],partAnchorInsert:[0,4,16,17],contactClear:[0,1,4,12],voxelVote:[0,1,2,4,12,13],voxelHome:[0,1,4,12],voxelBin:[0,1,2,4,12,13],chartCorners:[0,1,2,4,12,13],chartCertify:[0,1,4,12],contactVertices:[0,1,2,4,12],contactEdges:[0,1,2,4,12,13,14],boundaries:[0,1,2,3,4,5,6,8,10,11,12,15],contours:[0,1,2,3,4,5,6,8,10,11,15]};
    (buffers as RelationBinding[])[10]={view:this.lineDetailView(earlyDetail)};
    (buffers as RelationBinding[])[15]={view:this.contactPatchView()};
    // perf-impl-93: the line stages read the same-pose form matrix (levelDrops).
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
    const indirect=(stage:string,offset?:number)=>{const pipeline=this.pipelines[stage]!;pass.setPipeline(pipeline);pass.setBindGroup(0,this.group(held.groups,stage,pipeline,used[stage]!,buffers));if(offset===undefined)pass.dispatchWorkgroups(1);else pass.dispatchWorkgroupsIndirect(this.dispatch,offset);};
    if(posedContacts&&!host){
      indirect('prepare');indirect('contactClear',24);indirect('voxelVote',36);direct('voxelHome',home?4+home.slots*4:0,held.groups,used.voxelHome!,buffers);indirect('voxelBin',48);indirect('chartCorners',36);direct('chartCertify',chart?chart.clusters:0,held.groups,used.chartCertify!,buffers);indirect('contactVertices',36);indirect('contactEdges',48);
    }else if(posedContacts){
      direct('contactClear',contactWords,held.groups,used.contactClear!,buffers);
      direct('voxelVote',faces*2,held.groups,used.voxelVote!,buffers);
      direct('voxelHome',home?4+home.slots*4:0,held.groups,used.voxelHome!,buffers);
      direct('voxelBin',contactEdges,held.groups,used.voxelBin!,buffers);
      direct('chartCorners',faces*2,held.groups,used.chartCorners!,buffers);
      direct('chartCertify',chart?chart.clusters:0,held.groups,used.chartCertify!,buffers);
      direct('contactVertices',faces*2,held.groups,used.contactVertices!,buffers);
      direct('contactEdges',contactEdges,held.groups,used.contactEdges!,buffers);
    }
    // perf-impl-140: project this source's declared part anchors, then insert them (separate dispatches).
    const anchors=this.partAnchorsFor(batch,source);
    if(anchors&&this.partAnchorBuffer){
      (buffers as RelationBinding[])[16]=anchors.input;(buffers as RelationBinding[])[17]=this.partAnchorBuffer;
      if(!this.partAnchorFrameCleared){direct('partAnchorClear',this.partAnchorHeadSlots+this.partAnchorRecordCapacity,held.groups,used.partAnchorClear!,buffers);this.partAnchorFrameCleared=true;}
      direct('partAnchors',anchors.parts*anchors.instances,held.groups,used.partAnchors!,buffers);
      direct('partAnchorInsert',anchors.parts*anchors.instances,held.groups,used.partAnchorInsert!,buffers);
    }
    // FUSED SOURCE STAGES (perf-impl-76): with a host-sized extent, boundaries
    // and contours are ONE dispatch (WGSL surfaceLines: the same per-id bodies
    // over one id range, edges x instances then faces x instances).
    if(host){const instances=frontier?.indices.length??host.instances;direct('surfaceLines',(edges+faces)*instances,held.groups,used.boundaries!,buffers);}
    else{if(!posedContacts)indirect('prepare');indirect('boundaries',0);indirect('contours',12);}
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
    const interiorUsed:Record<string,number[]>={prepareInterior:[0,1,2,3,4,6,7,8,9,10,11],faceDifferentials:[0,1,2,3,4,6,8,9],radialCurvature:[0,1,2,3,4,8,9],suggestiveContours:[0,1,2,3,4,5,6,8,9,15],principalCurvature:[0,1,2,3,4,8,9],ridgeDerivative:[0,1,2,4,9],ridgeLines:[0,1,2,3,4,5,6,8,9,15],viewFaceLines:[0,1,2,3,4,5,6,8,9,15]};
    const hostInterior=host&&(!earlyDetail||terrain);
    const detailView=earlyDetail?formDepths(this.device).view(field!):(this.emptyDetail??=this.device.createTexture({label:'MAIN unselected detail table',size:[1,1],format:'rgba32uint',usage:GPUTextureUsage.TEXTURE_BINDING})).createView();
    const detailMatrix=earlyDetail?formDepths(this.device).matrix(field!):params;
    for(let first=0;first<capacity;first+=chunk){
      let differential=this.differentials,refresh=true;
      if(batch.source.posedRevision!==undefined){
        let chunks=this.retainedDifferentials.get(posedBatchIdentity(batch));if(!chunks){chunks=new Map();this.retainedDifferentials.set(posedBatchIdentity(batch),chunks);}
        let cached=chunks.get(first);
        if(!cached||cached.buffer.size<bytes){if(cached)this.retired.push(cached.buffer);cached={revision:-1,buffer:this.device.createBuffer({label:'MAIN retained posed shape tensor',size:bytes,usage:GPUBufferUsage.STORAGE})};chunks.set(first,cached);}
        differential=cached.buffer;const key=tensorRevision(batch);refresh=key===undefined||cached.revision!==key;cached.revision=key??-1;
      }
      let range=held.ranges.get(first);
      if(!range){range={uniform:this.device.createBuffer({label:'MAIN posed differential instance range',size:192,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),groups:new Map()};held.ranges.set(first,range);}
      const uniform=range.uniform;
      const localCount=Math.min(chunk,capacity-first),data=parameterWords!.slice();data.set([first,localCount,this.device.limits.maxComputeWorkgroupsPerDimension],32);
      if(frontier){
        const lower=(value:number)=>{let lo=0,hi=frontier.indices.length;while(lo<hi){const mid=(lo+hi)>>>1;if(frontier.indices[mid]!<value)lo=mid+1;else hi=mid;}return lo;};
        const from=lower(first),to=lower(first+localCount);data[42]=to-from;data[43]=frontier.offset+from;
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
  // ---- ONE DISPATCH PER STAGE OVER EVERY SOURCE (perf-impl-79) ----
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
  private queueBatched(batch:PosedSourceBatch,source:ResidentRelationSource,words:Uint32Array,host:SourceDrawExtent,{posedContacts,contactWords,contactEdges,parts,chartCodes,chartClusters,earlyDetail,terrain,selection}:{posedContacts:boolean;contactWords:number;contactEdges:number;parts:{readonly cellCodes:SourceStorageBinding;readonly levels:SourceStorageBinding}|undefined;chartCodes:SourceStorageBinding|undefined;chartClusters:number;earlyDetail:boolean;terrain:boolean;selection:GPUBuffer}):void{
    const q=this.queue!;q.seen.add(posedBatchIdentity(batch));
    const indexed=isGpuIndexedResident(source)?source.gpu:undefined;
    const whole=(value:SourceStorageBinding,unit:number):[GPUBuffer,number]=>{const b=storageBinding(value),offset=b.offset??0;if(offset%unit)throw Error('Batched source relation range is not word aligned');return [b.buffer,offset/unit];};
    const [incidence,incidenceBase]=whole(indexed?.incidence.binding??source.bufferBinding??source.buffer,4);
    const [vertices,vertexBase]=whole(batch.storage?.vertices??batch.vertices,16);
    const [attributes,attributeBase]=whole(batch.storage?.attributes??batch.attributes,16);
    const [metadata,metadataBase]=whole(batch.storage?.metadata??batch.metadata,4);
    const contactBase=posedContacts?q.contactWords:0;if(posedContacts)q.contactWords+=contactWords;
    const base=[incidenceBase,vertexBase,attributeBase,metadataBase];
    // Bindings 5, 6, 7 and 12 are this object's shared buffers, bound at flush.
    const bound:RelationBinding[]=[];bound[1]=incidence;bound[2]=vertices;bound[3]=attributes;bound[4]=metadata;bound[8]=selection;
    bound[13]=chartCodes??metadata;bound[14]=parts?.levels??this.status;bound[15]={view:this.contactPatchView()};
    const push=(stage:BatchStage,threads:number,itemWords:Uint32Array,base2:readonly number[],itemBound:RelationBinding[])=>{if(threads>0)q.items.push({stage,threads,words:itemWords,base,base2,bound:itemBound});};
    const vertexCount=sourceResidencyVertexCount(source),faces=sourceResidencyFaceCount(source),edges=sourceResidencyEdgeCount(source);
    const lineBase2=[contactBase,0,0,0];
    // perf-impl-140: Y8's batched sources project and insert their declared part anchors too.
    const anchorEntry=this.partAnchorsFor(batch,source);
    if(anchorEntry&&this.partAnchorBuffer){const anchorBound=[...bound];anchorBound[16]=anchorEntry.input;anchorBound[17]=this.partAnchorBuffer;push('partAnchors',anchorEntry.parts*anchorEntry.instances,words,lineBase2,anchorBound);push('partAnchorInsert',anchorEntry.parts*anchorEntry.instances,words,lineBase2,anchorBound);}
    if(posedContacts){push('contactClear',contactWords,words,lineBase2,bound);push('voxelVote',faces*2,words,lineBase2,bound);push('voxelBin',contactEdges,words,lineBase2,bound);push('chartCorners',faces*2,words,lineBase2,bound);push('chartCertify',chartClusters,words,lineBase2,bound);push('contactVertices',faces*2,words,lineBase2,bound);push('contactEdges',contactEdges,words,lineBase2,bound);}
    const frontier=batch.source.visibleInstanceFrontier;
    const lineBound=[...bound];lineBound[10]={view:this.lineDetailView(earlyDetail)};
    lineBound[11]=earlyDetail?formDepths(this.device).matrix(this.detailField!):(this.emptyMatrix??=this.device.createBuffer({label:'MAIN unselected detail matrix',size:64,usage:GPUBufferUsage.UNIFORM}));
    push('surfaceLines',(edges+faces)*(frontier?.indices.length??host.instances),words,lineBase2,lineBound);
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
      const key=tensorRevision(batch),refresh=key===undefined||cached.revision!==key;cached.revision=key??-1;
      const tensor={buffer:cached.range.page.buffer,offset:cached.range.offset};
      const localCount=Math.min(chunk,capacity-first),data=words.slice();data.set([first,localCount,this.device.limits.maxComputeWorkgroupsPerDimension],32);
      if(frontier){
        const lower=(value:number)=>{let lo=0,hi=frontier.indices.length;while(lo<hi){const mid=(lo+hi)>>>1;if(frontier.indices[mid]!<value)lo=mid+1;else hi=mid;}return lo;};
        const from=lower(first),to=lower(first+localCount);data[42]=to-from;data[43]=frontier.offset+from;
      }
      const interiorBound=[...bound];interiorBound[9]=tensor.buffer;interiorBound[10]={view:detailView};interiorBound[11]=detailMatrix;
      // WGSL interiorInstances and interiorCandidates for this chunk.
      const local=Math.min(localCount,host.instances-Math.min(host.instances,first)),visible=frontier?data[42]!:local;
      const gated=!hostInterior,dispatchBase=gated?q.dispatchItems++*BATCH_DISPATCH_WORDS:0;
      const base2=[contactBase,(tensor.offset??0)/16,dispatchBase,Number(gated)];
      if(gated)q.items.push({stage:'prepareInterior',threads:64,words:data,base,base2,bound:interiorBound});
      const counts:Partial<Record<BatchStage,number>>={faceDifferentials:faces*local,radialCurvature:vertexCount*visible,suggestiveContours:faces*visible,principalCurvature:vertexCount*local,ridgeDerivative:vertexCount*local,ridgeLines:faces*visible,viewFaceLines:faces*visible};
      const stages:BatchStage[]=refresh?['faceDifferentials','radialCurvature','suggestiveContours','principalCurvature','ridgeDerivative','ridgeLines']:['radialCurvature','viewFaceLines'];
      for(const stage of stages)push(stage,counts[stage]!,data,base2,interiorBound);
    }
  }
  /** Every queued source's stages: one dispatch per stage per group of sources
   * whose bound pages agree (at most SURFACE_RELATION_BATCH_ITEMS per dispatch). */
  private flushBatched(encoder:GPUCommandEncoder):void{
    const q=this.queue;this.queue=undefined;if(!q||!q.items.length)return;
    if(!this.batchContact||this.batchContact.size<Math.max(4,q.contactWords*4)){if(this.batchContact)this.retired.push(this.batchContact);this.batchContact=this.device.createBuffer({label:'MAIN batched posed quad contacts',size:Math.max(4,Math.ceil(q.contactWords*1.25)*4),usage:GPUBufferUsage.STORAGE});}
    const dispatchBytes=Math.max(1,q.dispatchItems)*BATCH_DISPATCH_WORDS*4;
    if(!this.batchDispatch||this.batchDispatch.size<dispatchBytes){if(this.batchDispatch)this.retired.push(this.batchDispatch);this.batchDispatch=this.device.createBuffer({label:'MAIN batched source dispatch words',size:dispatchBytes*2,usage:GPUBufferUsage.STORAGE});}
    for(const clear of q.clears)encoder.clearBuffer(clear.buffer,clear.offset,clear.size);
    // perf-impl-140: stale invalidation before the first anchor insertion of the frame (heads + link words; the
    // header [head slots, record base] survives).
    if(!this.partAnchorFrameCleared&&this.partAnchorBuffer&&q.items.some(item=>item.stage==='partAnchorInsert')){encoder.clearBuffer(this.partAnchorBuffer,8);this.partAnchorFrameCleared=true;}
    const limit=this.device.limits.maxComputeWorkgroupsPerDimension;
    const pass=encoder.beginComputePass({label:'MAIN source relations'});
    try{
      for(const stage of BATCH_ORDER){
        const items=q.items.filter(item=>item.stage===stage);if(!items.length)continue;
        const bindings=BATCH_BINDINGS[stage],groups=new Map<string,BatchItem[]>();
        for(const item of items){
          item.bound[5]=this.lines;item.bound[6]=this.status;item.bound[7]=this.batchDispatch;item.bound[12]=this.batchContact;
          const key=bindings.filter(b=>b!==0).map(b=>this.bindingId(item.bound[b]!)).join('|');
          let list=groups.get(key);if(!list){list=[];groups.set(key,list);}list.push(item);
        }
        const pipeline=this.pipelines[batchEntryName(stage)]!;
        for(const list of groups.values())for(let at=0;at<list.length;at+=SURFACE_RELATION_BATCH_ITEMS){
          const part=list.slice(at,at+SURFACE_RELATION_BATCH_ITEMS),data=new Uint32Array(4+part.length*BATCH_ITEM_WORDS);let total=0;
          part.forEach((item,k)=>{const o=4+k*BATCH_ITEM_WORDS;data.set(item.words,o);data.set(item.base,o+48);data.set(item.base2,o+52);data.set([total,item.threads],o+56);total+=item.threads;});
          data.set([part.length,limit,total,0],0);
          let table=this.batchTables[this.batchCursor];
          if(!table){table={buffer:this.device.createBuffer({label:'MAIN batched source relation table',size:BATCH_TABLE_BYTES,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),groups:new Map()};this.batchTables.push(table);}
          this.batchCursor++;this.device.queue.writeBuffer(table.buffer,0,data);
          const bound=[...part[0]!.bound];bound[0]=table.buffer;
          pass.setPipeline(pipeline);pass.setBindGroup(0,this.group(table.groups,stage,pipeline,bindings,bound));
          if(stage==='prepareInterior')pass.dispatchWorkgroups(part.length);
          else{const count=Math.ceil(total/64);pass.dispatchWorkgroups(Math.min(count,limit),Math.ceil(count/limit));}
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
    const fresh=():RetainedBindings=>({params:this.device.createBuffer({label:'MAIN source relation parameters',size:192,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),ownerKey:'',groups:new Map(),ranges:new Map(),serial:this.serial});
    if(entry&&entry.serial===this.serial){const once=fresh();this.oneFrame.push(once);return once;}
    if(!entry){
      entry=fresh();byBatch.set(key,entry);const owned=entry;
      this.expiry.use(entry,()=>{if(byBatch.get(key)!==owned)return;this.release(owned);byBatch.delete(key);if(!byBatch.size)this.retained.delete(source);});
    }else this.expiry.use(entry);
    entry.serial=this.serial;return entry;
  }
  private oneFrame:RetainedBindings[]=[];
  /** A bind group is reused while every buffer it names is the same object. */
  // perf-impl-108 CONTACT PATCHES: the host's explicit two-owner solver incidence
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
    const group=this.device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:bindings.map((binding,index)=>{const resource=named[index]!;return {binding,resource:'view' in resource?resource.view:storageBinding(resource)};})});
    groups.set(stage,{buffers:named,group});return group;
  }
  private release(entry:RetainedBindings):void{this.retired.push(entry.params);if(entry.selection)this.retired.push(entry.selection);for(const range of entry.ranges.values())this.retired.push(range.uniform);}
  retainPosedBatches(live:ReadonlySet<object>):void{for(const [key,chunks]of this.retainedDifferentials)if(!live.has(key)){for(const entry of chunks.values())this.retired.push(entry.buffer);this.retainedDifferentials.delete(key);}
    for(const [key,chunks]of this.tensorRanges)if(!live.has(key)){for(const entry of chunks.values())this.tensors!.retire(entry.range);this.tensorRanges.delete(key);}
    for(const [key,entry]of this.partAnchorSources)if(!live.has(key as object)){this.retired.push(entry.input);this.partAnchorSources.delete(key);}
  }
  submitted():void{
    this.partAnchorFrameCleared=false;
    if(this.partAnchorGrowTo>0){const records=this.partAnchorGrowTo;this.partAnchorGrowTo=0;this.growPartAnchorTable(records);}
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
  dispose():void{this.homes.clear();this.homeWords=0;this.expiry.clear();for(const byBatch of this.retained.values())for(const entry of byBatch.values())this.release(entry);this.retained.clear();this.retainPosedBatches(new Set());this.partAnchorGrowTo=0;if(this.partAnchorBuffer)this.retired.push(this.partAnchorBuffer);this.partAnchorBuffer=undefined;this.partAnchorRecords=0;this.partAnchorRecordCapacity=0;this.partAnchorHeadSlots=0;this.submitted();this.intersections?.dispose();this.lines.destroy();this.status.destroy();this.dispatch.destroy();this.differentials?.destroy();this.contactWork?.destroy();this.emptyDetail?.destroy();this.emptyMatrix?.destroy();this.tensors?.dispose();this.batchContact?.destroy();this.batchDispatch?.destroy();for(const table of this.batchTables)table.buffer.destroy();}
}
