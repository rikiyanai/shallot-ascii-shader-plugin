import type {GpuIndexedSourcePose} from './gpu-indexed-source.ts';
import {INDEXED_INCIDENCE_MAGIC,INDEXED_QUAD_INCIDENCE_VERSION} from './gpu-source-incidence.ts';
import type {ConstructionType} from './composition.ts';
import {TOPOLOGY_HEADER_WORDS,TOPOLOGY_TYPE_TERRAIN} from './compiled-construction-topology.ts';
/** Original quad contexts by fixed address arithmetic; only the posed
 * diagonal's two-bit crease class needs storage. */
export const QUAD_RUN_CONTEXT=0x51554144;
export function gpuIndexedQuadContext(input:GpuIndexedConstructionInput):boolean{return input.indexPattern==='independent-quads'&&!input.planeFaces&&!input.planeRegionCount;}

/** GPU-only input for a resident source construction slot.  The topology
 * output is written into GpuConstructionRuns' existing arena; this module
 * owns no source or relation storage. */
export interface GpuIndexedConstructionInput {
 readonly domain:number;
 readonly revision:number;
 readonly generation:number;
 readonly kind:ConstructionType;
 readonly indexPattern?:'independent-quads';
 readonly topology:{readonly indices:GPUBufferBinding;readonly indexBase:number;readonly indexCount:number;readonly vertexCapacity:number};
 /** The existing CSR binding starts at its magic word. It is not a copied
  * source packet and remains owned by GpuIndexedSourceResidency. */
 readonly incidence:{readonly binding:GPUBufferBinding;readonly layout:{readonly edges:number;readonly cornerEdge:number;readonly edgeOffsets:number;readonly edgeCorners:number;readonly fanOffsets:number;readonly fanCorners:number}};
 /** Optional authored plane-region count. Zero means no region table. */
 readonly planeRegionCount?:number;
 /** Optional GPU face->plane-region ids. 0xffffffff means no authored plane. */
 readonly planeFaces?:GPUBufferBinding;
 /** Construction reads positions only, after the material batch's other
  * readers finish. Do not extend attributes/metadata lifetime to this stage. */
 readonly pose:Pick<GpuIndexedSourcePose,'vertices'>;
}

export interface GpuIndexedConstructionSlot {
 readonly words:number;
 /** Header 38/39: one entry per source halfedge address. */
 readonly edgeWords:number;
 /** Header 40/41: one entry per source vertex address. */
 readonly vertexWords:number;
 /** Header 42/43: one entry per source halfedge address. */
 readonly edgeClassWords:number;
 readonly generation:number;
 readonly revision:number;
}

export interface IndexedConstructionOracleInput {
 readonly domain?:number;readonly kind:ConstructionType;readonly positions:Float32Array;readonly indices:Uint32Array;
 readonly planeRegionCount?:number;readonly facePlane?:readonly number[];
}
export interface IndexedConstructionOracle {
 readonly header:Uint32Array;readonly edgeGroups:Uint32Array;readonly vertexGroups:Uint32Array;readonly edgeClasses:Uint32Array;
 readonly words:Uint32Array;readonly canonicalEdges:readonly {readonly first:number;readonly corners:readonly number[]}[];
}
const CREASE_COS=0.70710678,COPLANAR_COS=0.99862953;
const typeCode=(kind:ConstructionType):number=>kind==='terrain'?TOPOLOGY_TYPE_TERRAIN:kind==='planar'?2:kind==='rounded'?3:kind==='paper'?0:kind==='far-strip'?4:1;
const edgeKey=(a:number,b:number):string=>a<b?`${a}:${b}`:`${b}:${a}`;
const faceNormal=(positions:Float32Array,indices:Uint32Array,face:number):[number,number,number]=>{
 const a=indices[face*3]!*3,b=indices[face*3+1]!*3,c=indices[face*3+2]!*3;
 const ax=positions[b]!-positions[a]!,ay=positions[b+1]!-positions[a+1]!,az=positions[b+2]!-positions[a+2]!;
 const bx=positions[c]!-positions[a]!,by=positions[c+1]!-positions[a+1]!,bz=positions[c+2]!-positions[a+2]!;
 const nx=ay*bz-az*by,ny=az*bx-ax*bz,nz=ax*by-ay*bx,len=Math.hypot(nx,ny,nz);
 return len>1e-20?[nx/len,ny/len,nz/len]:[0,0,0];
};
const dot=(a:readonly number[],b:readonly number[]):number=>a[0]! * b[0]!+a[1]! * b[1]!+a[2]! * b[2]!;

/** CPU oracle for the GPU packet. Context groups, vertex groups and edge
 * classes are intentionally separate arrays. */
export function deriveIndexedConstructionOracle(input:IndexedConstructionOracleInput):IndexedConstructionOracle{
 const {indices,positions,kind}=input;
 if(indices.length<3||indices.length%3||positions.length%3)throw Error('indexed topology fixture has invalid extent');
 const vertices=positions.length/3,corners=indices.length,faces=corners/3;
 for(const index of indices)if(index>=vertices)throw Error('indexed topology fixture index is out of bounds');
 const normals=Array.from({length:faces},(_,face)=>faceNormal(positions,indices,face));
 const groups=new Map<string,number[]>();
 for(let h=0;h<corners;h++){const next=Math.floor(h/3)*3+(h+1)%3,key=edgeKey(indices[h]!,indices[next]!);(groups.get(key)??(groups.set(key,[]),groups.get(key)!)).push(h);}
 const canonicalEdges=[...groups.values()].map(corners=>({first:corners[0]!,corners}));
 const edgeAtCorner=new Uint32Array(corners);for(const [edge,candidate]of canonicalEdges.entries())for(const h of candidate.corners)edgeAtCorner[h]=edge;
 const nextCorner=(h:number)=>Math.floor(h/3)*3+(h+1)%3;
 const boundarySuccessor=(h:number):number=>{let cursor=nextCorner(h);for(let step=0;step<corners;step++){const edge=edgeAtCorner[cursor]!,degree=canonicalEdges[edge]!.corners.length;if(degree===1)return canonicalEdges[edge]!.first;if(degree!==2)return -1;const members=canonicalEdges[edge]!.corners,cursor2=members[0]===cursor?members[1]!:members[0]!;cursor=nextCorner(cursor2);}return -1;};
 const faceNeighbours=Array.from({length:faces},()=>[] as number[]);
 for(const edge of canonicalEdges)for(const a of edge.corners)for(const b of edge.corners){const fa=Math.floor(a/3),fb=Math.floor(b/3);if(fa!==fb&&!faceNeighbours[fa]!.includes(fb))faceNeighbours[fa]!.push(fb);}
 const inPlane=(face:number):boolean=>faceNeighbours[face]!.some(other=>dot(normals[face]!,normals[other]!)>=COPLANAR_COS);
 const faceRegion=input.facePlane??[],hasPlane=(face:number):boolean=>faceRegion[face]!==undefined;
 const edgeClasses=new Uint32Array(corners),edgeGroups=new Uint32Array(corners);
 const parent=Uint32Array.from({length:canonicalEdges.length},(_,edge)=>edge);
 const root=(edge:number):number=>{let at=edge;while(parent[at]!==at)at=parent[at]!;while(parent[edge]!==edge){const next=parent[edge]!;parent[edge]=at;edge=next;}return at;};
 const union=(a:number,b:number)=>{a=root(a);b=root(b);if(a!==b)parent[Math.max(a,b)]=Math.min(a,b);};
 for(let a=0;a<canonicalEdges.length;a++)if(canonicalEdges[a]!.corners.length===1){const successor=boundarySuccessor(canonicalEdges[a]!.first);if(successor>=0)union(a,[...canonicalEdges.entries()].findIndex(([,candidate])=>candidate.first===successor));}
 for(const [edgeIndex,edge]of canonicalEdges.entries()){
  const first=edge.first,degree=edge.corners.length,facesForEdge=[...new Set(edge.corners.map(h=>Math.floor(h/3)))];let klass=0;
  if(degree!==2)klass=1;else{
   const a=facesForEdge[0]!,b=facesForEdge[1]!,all=facesForEdge.every(hasPlane),any=facesForEdge.some(hasPlane),distinct=all&&faceRegion[a]!==faceRegion[b];
   if(all&&(facesForEdge.length===1||((input.planeRegionCount??0)<=24&&distinct)))klass=1;
   else if(dot(normals[a]!,normals[b]!)<CREASE_COS)klass=(kind==='planar'||any||(inPlane(a)&&inPlane(b)))?1:2;
  }
  // The GPU source has no authored boundary-loop table. A conservative
  // edge-local marker for degree-one boundaries never joins unrelated edges;
  // nonmanifold edges remain ungrouped and are represented by class 1.
  if(degree===1)edgeGroups[first]=(0x20000000|root(edgeIndex))>>>0;
  edgeClasses[first]=klass;
 }
 const vertexGroups=Uint32Array.from({length:vertices},(_,vertex)=>vertex),header=new Uint32Array(TOPOLOGY_HEADER_WORDS);
 header[0]=input.domain??0;header[1]=typeCode(kind);header[16]=input.planeRegionCount??0;
 header[38]=TOPOLOGY_HEADER_WORDS;header[39]=corners;header[40]=TOPOLOGY_HEADER_WORDS+corners;header[41]=vertices;header[42]=TOPOLOGY_HEADER_WORDS+corners+vertices;header[43]=corners;
 const words=new Uint32Array(TOPOLOGY_HEADER_WORDS+corners+vertices+corners);words.set(header);words.set(edgeGroups,header[38]);words.set(vertexGroups,header[40]);words.set(edgeClasses,header[42]);
 return {header,edgeGroups,vertexGroups,edgeClasses,words,canonicalEdges};
}

/** No CPU positions, normals, mesh objects, or edge arrays cross this seam.
 * The pass reads the real material-stage pose plus the existing GPU CSR and
 * writes three separate context ranges that GpuConstructionRuns already
 * consumes: edge groups, vertex groups, and edge classes. */
export const GPU_INDEXED_CONSTRUCTION_SHADER=/* wgsl */`
struct Params {slot:u32,domain:u32,kind:u32,edges:u32,indexBase:u32,indexCount:u32,vertexCapacity:u32,cornerEdge:u32,edgeOffsets:u32,edgeCorners:u32,fanOffsets:u32,fanCorners:u32,planeRegionCount:u32,planeFacesPresent:u32,_pass:u32,_pad:u32}
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read> indices:array<u32>;
@group(0) @binding(2) var<storage,read> csr:array<u32>;
@group(0) @binding(3) var<storage,read> posed:array<vec4u>;
@group(0) @binding(4) var<storage,read_write> topology:array<atomic<u32>>;
@group(0) @binding(5) var<storage,read> planeFaces:array<u32>;
fn loadWord(at:u32)->u32{return atomicLoad(&topology[at]);}
fn storeWord(at:u32,value:u32){atomicStore(&topology[at],value);}
fn minWord(at:u32,value:u32){atomicMin(&topology[at],value);}
fn indexAt(corner:u32)->u32{return indices[p.indexBase+corner];}
fn point(vertex:u32)->vec3f{return bitcast<vec3f>(posed[vertex].xyz);}
fn faceNormal(face:u32)->vec3f{
 let a=point(indexAt(face*3u));let b=point(indexAt(face*3u+1u));let c=point(indexAt(face*3u+2u));
 return normalize(cross(b-a,c-a));
}
fn quadProgram()->bool{return csr[0u]==${INDEXED_INCIDENCE_MAGIC}u&&csr[1u]==${INDEXED_QUAD_INCIDENCE_VERSION}u;}
fn edgeAtCorner(corner:u32)->u32{if(quadProgram()){let local=corner%6u;return corner/6u*5u+select(local,local-1u,local>=3u);}return csr[p.cornerEdge+corner];}
fn edgeFirst(edge:u32)->u32{if(quadProgram()){var starts=array<u32,5>(0u,1u,2u,4u,5u);return edge/5u*6u+starts[edge%5u];}return csr[p.edgeOffsets+edge];}
fn edgeLast(edge:u32)->u32{if(quadProgram()){return edgeFirst(edge)+select(1u,2u,edge%5u==2u);}return csr[p.edgeOffsets+edge+1u];}
fn edgeMember(at:u32)->u32{if(quadProgram()){return at;}return csr[p.edgeCorners+at];}
fn edgeCanonical(edge:u32)->u32{return edgeMember(edgeFirst(edge));}
fn faceRegion(face:u32)->u32{if(p.planeFacesPresent==0u){return 0xffffffffu;}return planeFaces[face];}
fn faceInPlane(face:u32)->bool{
 for(var k=0u;k<3u;k++){
  let edge=edgeAtCorner(face*3u+k);let first=edgeFirst(edge);let last=edgeLast(edge);
  for(var at=first;at<last;at++){
   let other=edgeMember(at)/3u;if(other!=face&&dot(faceNormal(face),faceNormal(other))>=0.99862953){return true;}
  }
 }
 return false;
}
fn boundaryEdge(edge:u32)->bool{return edgeLast(edge)==edgeFirst(edge)+1u;}
fn nextCorner(h:u32)->u32{return h/3u*3u+(h+1u)%3u;}
fn boundarySuccessor(h:u32)->u32{
 var cursor=nextCorner(h);
 for(var step=0u;step<p.indexCount;step++){
  let edge=edgeAtCorner(cursor);let degree=edgeLast(edge)-edgeFirst(edge);
  if(degree==1u){return edgeCanonical(edge);}
  if(degree!=2u){return 0xffffffffu;}
  let first=edgeFirst(edge);let a=edgeMember(first);let b=edgeMember(first+1u);cursor=nextCorner(select(a,b,a==cursor));
 }
 return 0xffffffffu;
}
fn boundaryParent(edge:u32)->u32{return p.slot+${TOPOLOGY_HEADER_WORDS}u+edgeCanonical(edge)+p.indexCount+p.vertexCapacity;}
fn boundaryRoot(edge:u32)->u32{
 var root=edge;
 for(var hop=0u;hop<32u;hop++){
  let next=loadWord(boundaryParent(root));if(next==0xffffffffu||next==root){return root;}root=next;
 }
 return root;
}
fn unionBoundary(id:vec3u){
 let edge=id.x;if(edge>=p.edges||!boundaryEdge(edge)){return;}
 minWord(boundaryParent(edge),edge);let successor=boundarySuccessor(edgeCanonical(edge));if(successor!=0xffffffffu){minWord(boundaryParent(successor),successor);let a=boundaryRoot(edge);let b=boundaryRoot(successor);if(a!=b){let low=min(a,b);let high=max(a,b);minWord(boundaryParent(high),low);}}
}
@compute @workgroup_size(64) fn clear(@builtin(global_invocation_id) id:vec3u){
 let work=id.x;
 if(p._pass==2u){
  if(work==0u){
   storeWord(p.slot,p.domain);storeWord(p.slot+1u,p.kind);storeWord(p.slot+16u,0u);storeWord(p.slot+37u,${QUAD_RUN_CONTEXT}u);
   storeWord(p.slot+38u,0u);storeWord(p.slot+39u,p.indexCount);storeWord(p.slot+40u,0u);storeWord(p.slot+41u,p.vertexCapacity);
   storeWord(p.slot+42u,${TOPOLOGY_HEADER_WORDS}u);storeWord(p.slot+43u,p.indexCount);
  }
  if(work<(p.indexCount/6u+15u)/16u){storeWord(p.slot+${TOPOLOGY_HEADER_WORDS}u+work,0u);}return;
 }
 if(work==0u){
  storeWord(p.slot+0u,p.domain);storeWord(p.slot+1u,p.kind);storeWord(p.slot+16u,p.planeRegionCount);
  storeWord(p.slot+38u,${TOPOLOGY_HEADER_WORDS}u);storeWord(p.slot+39u,p.indexCount);
  storeWord(p.slot+40u,${TOPOLOGY_HEADER_WORDS}u+p.indexCount);storeWord(p.slot+41u,p.vertexCapacity);
  storeWord(p.slot+42u,${TOPOLOGY_HEADER_WORDS}u+p.indexCount+p.vertexCapacity);storeWord(p.slot+43u,p.indexCount);
 }
 if(work<p.indexCount){
  var group=0u;if(p._pass==1u){let h=work%6u;if(h==0u||h==1u||h==4u||h==5u){group=0x20000000u|(work/6u*5u);}}
  storeWord(p.slot+${TOPOLOGY_HEADER_WORDS}u+work,group);storeWord(p.slot+${TOPOLOGY_HEADER_WORDS}u+p.indexCount+p.vertexCapacity+work,0xffffffffu);
 }
 if(work<p.vertexCapacity){storeWord(p.slot+${TOPOLOGY_HEADER_WORDS}u+p.indexCount+work,work);}
}
@compute @workgroup_size(64) fn unionBoundaryPass(@builtin(global_invocation_id) id:vec3u){unionBoundary(id);}
@compute @workgroup_size(64) fn finalizeBoundary(@builtin(global_invocation_id) id:vec3u){
 let work=id.x;if(work>=p.indexCount){return;}let edge=edgeAtCorner(work);if(edgeCanonical(edge)!=work){return;}
 if(boundaryEdge(edge)){storeWord(p.slot+${TOPOLOGY_HEADER_WORDS}u+work,0x20000000u|boundaryRoot(edge));}else{storeWord(p.slot+${TOPOLOGY_HEADER_WORDS}u+work,0u);}
}
@compute @workgroup_size(64) fn classify(@builtin(global_invocation_id) id:vec3u){
 if(p._pass==2u){
  let quad=id.x;if(quad>=p.indexCount/6u){return;}
  var klass=0u;if(dot(faceNormal(quad*2u),faceNormal(quad*2u+1u))<0.70710678){klass=select(2u,1u,p.kind==2u);}
  let at=p.slot+${TOPOLOGY_HEADER_WORDS}u+quad/16u;let shift=(quad%16u)*2u;
  atomicAnd(&topology[at],~(3u<<shift));atomicOr(&topology[at],klass<<shift);return;
 }
 let work=id.x;if(work>=p.indexCount){return;}let edge=edgeAtCorner(work);if(edgeCanonical(edge)!=work){storeWord(p.slot+${TOPOLOGY_HEADER_WORDS}u+p.indexCount+p.vertexCapacity+work,0u);return;}
 let first=edgeFirst(edge);let last=edgeLast(edge);let degree=last-first;var klass=0u;
 if(degree!=2u){klass=1u;}else{
  let a=edgeMember(first)/3u;let b=edgeMember(first+1u)/3u;let ra=faceRegion(a);let rb=faceRegion(b);let all=ra!=0xffffffffu&&rb!=0xffffffffu;let any=ra!=0xffffffffu||rb!=0xffffffffu;let distinct=all&&ra!=rb;
  if(all&&p.planeRegionCount<=24u&&distinct){klass=1u;}else if(dot(faceNormal(a),faceNormal(b))<0.70710678){klass=select(2u,1u,p.kind==2u||any||(faceInPlane(a)&&faceInPlane(b)));}
 }
 storeWord(p.slot+${TOPOLOGY_HEADER_WORDS}u+p.indexCount+p.vertexCapacity+work,klass);
}`;

export function gpuIndexedConstructionSlot(input:GpuIndexedConstructionInput):GpuIndexedConstructionSlot{
 if(gpuIndexedQuadContext(input)){
  const edgeClassWords=Math.ceil(input.topology.indexCount/6/16);
  return {words:TOPOLOGY_HEADER_WORDS+edgeClassWords,edgeWords:0,vertexWords:0,edgeClassWords,generation:input.generation,revision:input.revision};
 }
 const edgeWords=input.topology.indexCount,vertexWords=input.topology.vertexCapacity,edgeClassWords=input.topology.indexCount;
 return {words:TOPOLOGY_HEADER_WORDS+edgeWords+vertexWords+edgeClassWords,edgeWords,vertexWords,edgeClassWords,generation:input.generation,revision:input.revision};
}

export function gpuConstructionTypeCode(kind:GpuIndexedConstructionInput['kind']):number{
 return typeCode(kind);
}

/** The source-row form is intentionally separate from the packet type code:
 * the construction shader stores packet identity (terrain=5), while the
 * existing run shader's table row uses its compact source form (terrain=3). */
export function gpuConstructionSourceForm(kind:GpuIndexedConstructionInput['kind']):0|1|2|3{
 return kind==='planar'?1:kind==='rounded'?2:kind==='terrain'?3:0;
}
