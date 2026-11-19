import {GpuSourceIncidence,indexedIncidenceLayout,INDEXED_INCIDENCE_MAGIC,INDEXED_INCIDENCE_VERSION,INDEXED_QUAD_INCIDENCE_VERSION,type IndexedIncidenceLayout} from './gpu-source-incidence.ts';
import type {MeshSource,MeshSourceComplex} from './source-complex.ts';
import {SourceStorage,type SourceStorageRange} from './source-storage.ts';
import {OUTLINE_HULL_MAGIC,OUTLINE_HULL_WORDS,PRINCIPAL_CHAIN_MAGIC,PRINCIPAL_CHAIN_WORDS} from './compiled-construction-topology.ts';

/** V2 remains the legacy adapter. V3 has explicit CSR readers, not an
 * interpretation of V2's old halfedge/position offsets. */
export const COMPACT_RELATION_SOURCE_VERSION=3;
/** Numeric registration derives the runtime's contact lists on the GPU.
 * Welded components belong to semantic compilation/explicit CPU inspection. */
export const GPU_NUMERIC_RELATION_SOURCE_VERSION=5;
export interface CompiledSourceIncidencePlan {
 readonly kind:'indexed-csr-v1';readonly prefix:Uint32Array;
 readonly csr:IndexedIncidenceLayout;readonly csrOffset:number;readonly words:number;
 readonly indexOffset:number;readonly edges:number;
}
export interface UploadedSourceIncidence {
 readonly range:SourceStorageRange;readonly binding:GPUBufferBinding;readonly indices:GPUBufferBinding;
}
/** The registration seam for both engine adapters: one complete source
 * packet, built in its final arena range. Readiness covers the CSR fence.
 * No CPU reconstruction or copy-back of the GPU-built incidence occurs. */
export class GpuRelationSourceCompiler {
 private readonly indexed:GpuSourceIncidence;private disposed=false;
 private readonly pending=new Set<Promise<UploadedSourceIncidence>>();
 static async create(device:GPUDevice,storage:SourceStorage,progress?:(fraction:number)=>void):Promise<GpuRelationSourceCompiler>{
  const indexed=await GpuSourceIncidence.create(device,storage,8*1024*1024,progress);
  return new GpuRelationSourceCompiler(device,storage,indexed);
 }
 constructor(device:GPUDevice,private readonly storage:SourceStorage,prepared?:GpuSourceIncidence){this.indexed=prepared??new GpuSourceIncidence(device,storage);}
 build(indices:GPUBufferBinding,vertices:number,corners:number,edgeCapacity=corners,indexPattern?:'independent-quads'){return this.indexed.buildWithIndices(indices,vertices,corners,edgeCapacity,indexPattern);}
 buildRaw(indices:GPUBufferBinding,vertices:number,corners:number,edges:number){return this.indexed.build(indices,vertices,corners,edges);}
 upload(plan:CompiledSourceIncidencePlan):Promise<UploadedSourceIncidence>{
  if(this.disposed)return Promise.reject(Error('Relation source compiler is disposed'));
  const task=this.run(plan);this.pending.add(task);void task.then(()=>this.pending.delete(task),()=>this.pending.delete(task));return task;
 }
 private async run(plan:CompiledSourceIncidencePlan):Promise<UploadedSourceIncidence>{
  if(plan.kind!=='indexed-csr-v1'||plan.prefix[0]!==0x4d41494e||plan.prefix[1]!==COMPACT_RELATION_SOURCE_VERSION||plan.prefix.length!==plan.csrOffset||plan.words!==plan.csrOffset+plan.csr.words||plan.prefix[4]!==plan.indexOffset||plan.prefix[5]!==plan.csrOffset||plan.prefix[6]!==plan.edges||plan.csr.edgeCapacity!==plan.edges)throw Error('Invalid compact relation source plan');
  const range=this.storage.allocate(plan.words*4);
  try{
   this.storage.write(range,plan.prefix);const binding=this.storage.binding(range),offset=binding.offset??0;
   const indices={buffer:binding.buffer,offset:offset+plan.indexOffset*4,size:plan.csr.corners*4};
   const target={range,binding:{buffer:binding.buffer,offset:offset+plan.csrOffset*4,size:plan.csr.words*4}};
   const result=await this.indexed.buildInto(indices,plan.csr,target);
   if(result.edges!==plan.edges)throw Error('GPU/source residual edge identities disagree');
   if(this.disposed)throw Error('Relation source retired during GPU upload');
   return {range,binding,indices};
  }catch(error){this.storage.retire(range);throw error;}
 }
 /** The storage owner waits for this before destroying its pages. */
 async dispose():Promise<void>{this.disposed=true;this.indexed.dispose();await Promise.allSettled([...this.pending]);}
}
/** Small authoritative residual, not an expanded halfedge/fan packet. Exact
 * published contacts and connected-component decisions are preserved here;
 * raw indexed edge membership and ordered vertex fans are GPU-built.
 * CPU inspection keeps the original complete complex, including issues. */
export function compileSourceIncidencePlan(source:MeshSource,complex:MeshSourceComplex,attributes:{materialClasses?:Uint32Array;twoSided?:Uint32Array;edgeLevels?:Uint32Array}={}):CompiledSourceIncidencePlan {
 if(source.id!==complex.id||source.revision!==complex.revision||source.indices.length!==complex.halfedges.length||complex.faceComponents.length*3!==source.indices.length)throw Error('Source incidence revision/extent mismatch');
 const vertices=source.positions.length/3,faces=source.indices.length/3,csr=indexedIncidenceLayout(vertices,source.indices.length,complex.edges.length);
 const data:number[]=Array(16).fill(0),reserve=(n:number)=>{const at=data.length;for(let i=0;i<n;i++)data.push(0);return at;};
 const contacts=reserve(complex.contacts.length*2),edgeContact=reserve(complex.edges.length);
 for(const [i,contact]of complex.contacts.entries()){
  data[contacts+i*2]=data.length;data[contacts+i*2+1]=contact.members.length;
  for(const member of contact.members){
   const edge=complex.halfedges[member.edge];if(!edge||complex.edges[edge.edge]?.id!==member.edge)throw Error('Source contact lacks its canonical edge address');
   data.push(member.edge,Number(member.forward));data[edgeContact+edge.edge]=i+1;
  }
 }
 const components=data.length;for(const component of complex.faceComponents)data.push(component);
 const candidateRecords=reserve(vertices*2),groups=new Map<string,number[]>();
 if(!complex.vertexContacts){
  const bits=new Uint32Array(source.positions.buffer,source.positions.byteOffset,source.positions.length);
  for(let vertex=0;vertex<vertices;vertex++){
   const key=source.contactKeys?`${source.contactKeys[vertex*2]}:${source.contactKeys[vertex*2+1]}`:[0,1,2].map(axis=>source.positions[vertex*3+axis]===0?0:bits[vertex*3+axis]).join(':');
   const group=groups.get(key)??[];group.push(vertex);groups.set(key,group);
  }
 }
 for(const group of complex.vertexContacts??groups.values()){
  const at=data.length;for(const vertex of group)data.push(vertex);
  for(const vertex of group){data[candidateRecords+vertex*2]=at;data[candidateRecords+vertex*2+1]=group.length;}
 }
 const pack=(values:Uint32Array|undefined,perWord:number,maximum:number)=>{
  if(!values)return 0;
  if(values.length!==Math.ceil(faces/perWord))throw Error('Source attribute does not cover its face extent');
  const bits=32/perWord,mask=2**bits-1;
  if(faces%perWord&&(values.at(-1)!>>>((faces%perWord)*bits))!==0)throw Error('Source attribute has nonzero padding');
  for(let face=0;face<faces;face++)if(((values[Math.floor(face/perWord)]!>>>((face%perWord)*bits))&mask)>maximum)throw Error('Source attribute contains an invalid class');
  if(!values.some(value=>value!==0))return 0;const at=data.length;for(const value of values)data.push(value);return at;
 };
 const classes=pack(attributes.materialClasses,8,8),twoSided=pack(attributes.twoSided,32,1);
 // Compiled construction levels (compiled-construction-topology.ts
 // compileConstructionLevels), prefix word 8: the L1 codes, then the L2 hull.
 let levels=0;const edgeLevels=attributes.edgeLevels;
 if(edgeLevels?.length){if(edgeLevels.length<Math.ceil(source.indices.length/2))throw Error('Source edge levels do not cover its halfedge extent');levels=data.length;for(const value of edgeLevels)data.push(value);}
 while(data.length%64)data.push(0);const indexOffset=data.length;for(const index of source.indices)data.push(index);
 while(data.length%64)data.push(0);const csrOffset=data.length;
 // Original indices are also the physical field's index binding. No second
 // retained index copy is needed when that adapter migrates to this packet.
 const prefix=Uint32Array.from(data);
 prefix.set([0x4d41494e,COMPACT_RELATION_SOURCE_VERSION,vertices,faces,indexOffset,csrOffset,complex.edges.length,components,levels,contacts,complex.contacts.length,edgeContact,classes,source.revision,candidateRecords,twoSided]);
 return {kind:'indexed-csr-v1',prefix,csr,csrOffset,words:csrOffset+csr.words,indexOffset,edges:complex.edges.length};
}

/** Shared relation/contact consumer. Compile-time binding-name substitution
 * only; no source selection, fitting or graph decoding in a frame. */
export function sourceIncidenceReaders(stream:'source'|'topology'):string {
 return /* wgsl */`
fn sourceQuadProgram()->bool{return $source[0u]==${INDEXED_INCIDENCE_MAGIC}u&&$source[1u]==${INDEXED_QUAD_INCIDENCE_VERSION}u;}
fn sourceNumericProgram()->bool{return $source[0u]==0x4d41494eu&&$source[1u]==${GPU_NUMERIC_RELATION_SOURCE_VERSION}u;}
fn sourceGpuIndexed()->bool{return $source[0u]==${INDEXED_INCIDENCE_MAGIC}u&&($source[1u]==${INDEXED_INCIDENCE_VERSION}u||$source[1u]==${INDEXED_QUAD_INCIDENCE_VERSION}u);}
fn sourceVertexCount()->u32{return $source[2u];}
fn sourceFaceCount()->u32{return select($source[3u],$source[3u]/3u,sourceGpuIndexed());}
fn sourceEdgeCount()->u32{return select($source[6u],$source[4u],sourceGpuIndexed());}
fn sourceIndexAt(h:u32)->u32{if(sourceQuadProgram()){var vertices=array<u32,6>(0u,1u,2u,0u,2u,3u);return h/6u*4u+vertices[h%6u];}if(sourceGpuIndexed()){return $source[$source[12u]+h];}return $source[$source[4u]+h];}
// Zero means no authored face table. Otherwise class+1, so an explicitly
// authored unknown face is distinct from a whole-source material fallback.
// COMPILED CONSTRUCTION LEVEL of original halfedge h (compiled-construction-topology.ts
// compileEdgeLevels): 0 = produced at every tier. Compact packets carry the table at
// prefix word 8, numeric packets at header word 20; every other packet has none.
fn sourceEdgeLevel(h:u32)->u32{
 var at=0u;if($source[0u]==0x4d41494eu){if($source[1u]==${COMPACT_RELATION_SOURCE_VERSION}u){at=$source[8u];}else if($source[1u]==${GPU_NUMERIC_RELATION_SOURCE_VERSION}u){at=$source[20u];}}
 if(at==0u){return 0u;}return ($source[at+(h>>1u)]>>((h&1u)*16u))&0xffffu;
}
// THE COMPILED L2 OUTLINE HULL (compileOutlineHull, perf-impl-106): the
// section after the ceil(H/2) code words of the same table, H = faces x 3.
// 0 = this source has no outline construction (and is never drawn at L2).
fn sourceHullAt()->u32{
 var at=0u;if($source[0u]==0x4d41494eu){if($source[1u]==${COMPACT_RELATION_SOURCE_VERSION}u){at=$source[8u];}else if($source[1u]==${GPU_NUMERIC_RELATION_SOURCE_VERSION}u){at=$source[20u];}}
 if(at==0u){return 0u;}let base=at+(sourceFaceCount()*3u+1u)/2u;
 if($source[base]!=${OUTLINE_HULL_MAGIC}u){return 0u;}return base;
}
fn sourceHullEdgeCount()->u32{let base=sourceHullAt();if(base==0u){return 0u;}return $source[base+1u];}
// [a, b, c, d]: hull edge a-b, tri0 (a,b,c) and tri1 (b,a,d) outward.
fn sourceHullEdge(k:u32)->vec4u{let at=sourceHullAt()+2u+k*${OUTLINE_HULL_WORDS}u;return vec4u($source[at],$source[at+1u],$source[at+2u],$source[at+3u]);}
fn sourceHullFace(k:u32)->u32{return $source[sourceHullAt()+2u+k*${OUTLINE_HULL_WORDS}u+4u];}
// THE COMPILED L1 PRINCIPAL CHAINS (compilePrincipalChains), after the hull section.
fn sourceChainAt()->u32{
 let hull=sourceHullAt();if(hull==0u){return 0u;}let base=hull+2u+$source[hull+1u]*${OUTLINE_HULL_WORDS}u;
 if($source[base]!=${PRINCIPAL_CHAIN_MAGIC}u){return 0u;}return base;
}
fn sourceChainCount()->u32{let base=sourceChainAt();if(base==0u){return 0u;}return $source[base+1u];}
// [a, b, face0, face1, first member halfedge].
fn sourceChain(k:u32)->array<u32,5>{let at=sourceChainAt()+2u+k*${PRINCIPAL_CHAIN_WORDS}u;return array<u32,5>($source[at],$source[at+1u],$source[at+2u],$source[at+3u],$source[at+4u]);}
fn sourceFaceMaterial(face:u32)->u32{
 if(sourceGpuIndexed()){return 0u;}
 let at=$source[12];if(at==0u){return 0u;}
 return 1u+(($source[at+(face>>3u)]>>((face&7u)*4u))&15u);
}
// Explicit three-word runtime contract: original endpoints + indexed edge.
// Neither consumer reads a component id. Do not derive a welded union on the
// GPU merely to fill an unused fourth lane of this interface.
fn sourceHalfedge(h:u32)->vec3u{
 if(sourceQuadProgram()){let n=h/3u*3u+(h+1u)%3u;let local=h%6u;return vec3u(sourceIndexAt(h),sourceIndexAt(n),h/6u*5u+select(local,local-1u,local>=3u));}
 if(sourceGpuIndexed()){let n=h/3u*3u+(h+1u)%3u;return vec3u(sourceIndexAt(h),sourceIndexAt(n),$source[$source[6u]+h]);}
 if($source[1]==${COMPACT_RELATION_SOURCE_VERSION}u||sourceNumericProgram()){
  let c=$source[5];let n=h/3u*3u+(h+1u)%3u;
  return vec3u($source[$source[4]+h],$source[$source[4]+n],$source[c+$source[c+6u]+h]);
 }
 let at=$source[4]+h*8u;return vec3u($source[at],$source[at+1u],$source[at+4u]);
}
fn sourceEdge(id:u32)->vec4u{
 if(sourceQuadProgram()){var starts=array<u32,5>(0u,1u,2u,4u,5u);return vec4u(id/5u*6u+starts[id%5u],0u,select(1u,2u,id%5u==2u),0u);}
 if(sourceGpuIndexed()){let at=$source[7u]+id;let first=$source[at];let end=$source[at+1u];return vec4u($source[$source[8u]+first],$source[8u]+first,end-first,0u);}
 if($source[1]==${COMPACT_RELATION_SOURCE_VERSION}u||sourceNumericProgram()){
  let c=$source[5];let at=c+$source[c+7u]+id;let first=$source[at];let end=$source[at+1u];let members=c+$source[c+8u]+first;
  return vec4u($source[members],members,end-first,$source[$source[11]+id]);
 }
 let at=$source[5]+id*4u;return vec4u($source[at],$source[at+1u],$source[at+2u],$source[at+3u]);
}
fn sourceVertexRecord(vertex:u32)->vec4u{
 if(sourceQuadProgram()){if(vertex>=$source[3u]/6u*4u){return vec4u(0u);}return vec4u(0u,0u,0u,select(1u,2u,vertex%4u==0u||vertex%4u==2u));}
 if(sourceGpuIndexed()){let at=$source[9u]+vertex;let first=$source[at];let end=$source[at+1u];return vec4u(0u,0u,$source[10u]+first,end-first);}
 if(sourceNumericProgram()){
  let c=$source[5u];let at=c+$source[c+9u]+vertex;let first=$source[at];let end=$source[at+1u];let root=$source[$source[14u]+vertex];
  return vec4u(root,$source[$source[16u]+root],c+$source[c+10u]+first,end-first);
 }
 if($source[1]==${COMPACT_RELATION_SOURCE_VERSION}u){
  let c=$source[5];let at=c+$source[c+9u]+vertex;let first=$source[at];let end=$source[at+1u];let group=$source[14]+vertex*2u;
  return vec4u($source[group],$source[group+1u],c+$source[c+10u]+first,end-first);
 }
 let at=$source[14]+vertex*4u;return vec4u($source[at],$source[at+1u],$source[at+2u],$source[at+3u]);
}
fn sourceEdgeCorner(edge:u32,ordinal:u32)->u32{let record=sourceEdge(edge);if(sourceQuadProgram()){return record.x+ordinal;}return $source[record.y+ordinal];}
fn sourceVertexCandidate(vertex:u32,ordinal:u32)->u32{
 let record=sourceVertexRecord(vertex);if(!sourceNumericProgram()){return $source[record.x+ordinal];}
 var other=record.x;for(var i=0u;i<ordinal;i++){other=$source[$source[8u]+other]-1u;}return other;
}
fn sourceVertexCorner(vertex:u32,ordinal:u32)->u32{
 if(sourceQuadProgram()){
  let q=vertex/4u*6u;let local=vertex%4u;
  if(local==0u){return q+select(0u,3u,ordinal!=0u);}if(local==1u){return q+1u;}if(local==2u){return q+select(2u,4u,ordinal!=0u);}return q+5u;
 }
 return $source[sourceVertexRecord(vertex).z+ordinal];
}
`.replaceAll('$source',stream);
}
