import type {ResidentIndexedIncidence} from './gpu-source-incidence.ts';
import type {ConstructionType} from './composition.ts';
import {compileGpuIndexedSource,type GpuIndexedSourceCompiler} from './gpu-indexed-source-compiler.ts';
import {sameStorageBinding,SourceStorage,type SourceStorageRange} from './source-storage.ts';

/**
 * Contract for a source whose indexed topology is produced by a GPU pass.
 *
 * `sourceId` and `generation` are producer-owned.  A generation must advance
 * whenever the index contents or the vertex/index capacities change. A draw
 * count may change inside one topology only when the registered CSR extent
 * already covers that range; changing the compiled extent is a new generation. The
 * GPU count binding is retained for draw/visibility consumers and is never
 * read back to manufacture a CPU mesh.
 */
export const GPU_INDEXED_SOURCE_API_VERSION=1;
export type GpuIndexFormat='uint32';






export interface GpuIndexedClosedCellParts {
 readonly kind:'closed-cells';
 readonly extent:readonly [number,number,number];
 readonly cellCodes:GPUBufferBinding;
}
/** Producer-authored rest-cell chart for voxel material vertices. The packed
 * word is the original `voxel` attribute: low 24 bits are the source cell
 * x/y/z and the high byte is the emitted face direction. Unlike `parts`, this
 * chart does not claim that all six faces of every cell exist, so surface-only
 * terrain may publish it safely. */
export interface GpuIndexedRestCellChartInput {
 readonly kind:'voxel-rest-cell-chart';
 readonly cornerOrder:'shallot-voxel-quad-v1';
 readonly extent:readonly [number,number,number];
 readonly vertexCellCodes:GPUBufferBinding;
}
export interface GpuIndexedRestCellChart extends GpuIndexedRestCellChartInput {
 /** Exact topology generation whose vertex-code binding is still valid. */
 readonly sourceGeneration:number;
}
/** Producer-owned certificate for a material vertex stage whose posed content
 * is known from CPU inputs.  The certificate is intentionally typed rather
 * than inferred from GPUBuffer identity: the voxel producer supplies the
 * exact source emission revision, chunk instance Xform and TerrainMaterial
 * `inner`/`outer` values used by its vertex function.  Unknown or deforming
 * stages omit it. */
export interface GpuIndexedSourcePoseContentInput {
 readonly kind:'voxel-terrain-material';
 readonly vertexRevision:number;
 readonly instanceTransform:readonly [number,number,number,number,number,number,number,number,number,number];
 readonly materialParameters:readonly [number,number];
}
export interface GpuIndexedSourcePoseContent extends GpuIndexedSourcePoseContentInput {
 /** Source topology generation is assigned by the producer runtime. */
 readonly sourceGeneration:number;
 /** Monotonic producer revision; changes whenever any certified input changes. */
 readonly revision:number;
}
export interface GpuIndexedSourcePose {
 readonly sourceId:string;
 readonly generation:number;
 readonly posedRevision:number;




 readonly posedContentRevision?:number;
 readonly vertices:GPUBufferBinding;
 readonly attributes:GPUBufferBinding;
 readonly metadata:GPUBufferBinding;
 readonly vertexCapacity:number;
 /** Exact referenced prefix for a certified single-instance independent-quad
  * draw. Capacity remains the original instance stride, not a compact stride. */
 readonly singleInstanceVertexCount?:number;
}
export interface GpuIndexedSourceDescriptor {
 readonly apiVersion:typeof GPU_INDEXED_SOURCE_API_VERSION;
 readonly sourceId:string;
 readonly revision:number;
 readonly generation:number;
 readonly topology:{
  readonly indexPattern?:'independent-quads';
  readonly indexFormat:GpuIndexFormat;
  readonly indices:GPUBufferBinding;
  readonly indexCapacity:number;
  readonly vertexCapacity:number;
 };
 /** The producer's real DrawIndexedIndirect indexCount word. */
 readonly liveIndexCount:GPUBufferBinding;
 /**
  * Optional producer snapshot used only to size the registration-time CSR
  * build.  It is not a second source of truth: the live GPU binding above is
  * what draw and relation consumers use.  If absent, registration waits
  * rather than guessing from capacity or reading back a whole mesh.
  */
 readonly registrationIndexCount?:number;
 /** Genuine producer classification; voxel supplies terrain. */
 readonly constructionKind?:ConstructionType;
 /** Material contract over EVERY posed state, not a rest-normal inference.
  * Equal vertex normals within each face give zero smooth curvature. */
 readonly posedNormalClass?:'face-constant';
 /** The producer authorizes contacts between coincident quad edges of this
  * ONE source. Actual posed equality, not rest proximity, decides contact. */
 readonly edgeContactClass?:'posed-coincident';
 readonly parts?:GpuIndexedClosedCellParts;
 /** Optional source-cell chart; unlike `parts`, it does not imply closed cells. */
 readonly restCellChart?:GpuIndexedRestCellChart;
 /** Optional producer certificate for this source's material vertex pose. */
 readonly poseContent?:GpuIndexedSourcePoseContent;
}
export interface GpuIndexedSourceEmpty {
 readonly state:'empty'; readonly sourceId:string; readonly revision:number; readonly generation:number;
 readonly topology:GpuIndexedSourceDescriptor['topology']; readonly liveIndexCount:GPUBufferBinding;
}
export interface GpuIndexedSourceWaiting {
 readonly state:'waiting-for-live-index-count';
 readonly sourceId:string;readonly generation:number;
 readonly reason:'producer-registration-count-not-published';
}
export interface GpuIndexedSourceReady {
 readonly state:'ready';
 readonly apiVersion:typeof GPU_INDEXED_SOURCE_API_VERSION;
 readonly sourceId:string;readonly revision:number;readonly generation:number;
 readonly topology:GpuIndexedSourceDescriptor['topology'];
 readonly liveIndexCount:GPUBufferBinding;
 readonly incidence:ResidentIndexedIncidence;
 readonly constructionKind?:ConstructionType;
 readonly posedNormalClass?:'face-constant';
 readonly edgeContactClass?:'posed-coincident';
 readonly parts? :GpuIndexedClosedCellParts&{readonly levels:GPUBufferBinding;readonly range:SourceStorageRange};
 readonly restCellChart?:GpuIndexedRestCellChart;
}
export interface GpuIndexedSourceBatch {
 readonly sourceId:string;readonly revision:number;readonly generation:number;
 readonly topology:GpuIndexedSourceDescriptor['topology'];
 readonly liveIndexCount:GPUBufferBinding;
 readonly incidence:ResidentIndexedIncidence;
 readonly pose:GpuIndexedSourcePose;
 readonly constructionKind?:ConstructionType;
 readonly posedNormalClass?:'face-constant';
 readonly edgeContactClass?:'posed-coincident';
 readonly parts?:GpuIndexedSourceReady['parts'];
 readonly restCellChart?:GpuIndexedRestCellChart;
}

type BufferLike={readonly buffer:GPUBuffer;readonly offset?:number;readonly size?:number};
const integer=(value:number,name:string,min=0)=>{
 if(!Number.isSafeInteger(value)||value<min)throw Error(`GPU indexed source ${name} must be a non-negative integer`);
 return value;
};
const finite=(value:number,name:string):number=>{
 if(!Number.isFinite(value))throw Error(`GPU indexed source ${name} must be finite`);
 return value;
};
function validatePoseContent(value:GpuIndexedSourcePoseContent):void{
 if(value.kind!=='voxel-terrain-material')throw Error('Unsupported GPU indexed source pose-content certificate');
 integer(value.sourceGeneration,'pose-content source generation');
 integer(value.revision,'pose-content revision');
 integer(value.vertexRevision,'pose-content vertex revision');
 if(value.instanceTransform.length!==10||value.instanceTransform.some((n)=>!Number.isFinite(n)))throw Error('GPU indexed source pose-content transform must contain ten finite values');
 if(value.materialParameters.length!==2||value.materialParameters.some((n)=>!Number.isFinite(n)))throw Error('GPU indexed source pose-content material parameters must contain two finite values');
}
function bindingBytes(binding:BufferLike,name:string,minBytes:number):void{
 const offset=integer(binding.offset??0,`${name} offset`);
 if(offset%4)throw Error(`GPU indexed source ${name} offset must be 4-byte aligned`);
 const available=binding.size??binding.buffer.size-offset;
 if(!Number.isSafeInteger(available)||available<minBytes||offset+available>binding.buffer.size)
  throw Error(`GPU indexed source ${name} binding is shorter than its declared extent`);
}
function sameBinding(a:GPUBufferBinding,b:GPUBufferBinding):boolean{return sameStorageBinding(a,b);}

/** CPU-side validation at the producer/MAIN seam.  It deliberately does not
 * accept rest positions, CPU indices, scene names or an empty complex. */
export function validateGpuIndexedSourceDescriptor(value:GpuIndexedSourceDescriptor):void{
 if(value.apiVersion!==GPU_INDEXED_SOURCE_API_VERSION)throw Error(`Unsupported GPU indexed source API ${value.apiVersion}`);
 if(typeof value.sourceId!=='string'||value.sourceId.length===0)throw Error('GPU indexed source requires a stable producer sourceId');
 integer(value.revision,'revision');integer(value.generation,'generation');
 if(value.posedNormalClass!==undefined&&value.posedNormalClass!=='face-constant')throw Error('Unsupported GPU indexed source posed-normal class');
 const topology=value.topology;
 if(topology.indexFormat!=='uint32')throw Error('GPU indexed source requires uint32 indexed topology');
 const indexCapacity=integer(topology.indexCapacity,'index capacity',3),vertexCapacity=integer(topology.vertexCapacity,'vertex capacity',1);
 if(indexCapacity%3)throw Error('GPU indexed source index capacity must contain whole triangles');
 bindingBytes(topology.indices,'index topology',indexCapacity*4);
 bindingBytes(value.liveIndexCount,'live index count',4);
 if(value.parts){
  if(value.parts.kind!=='closed-cells'||value.parts.extent.length!==3||value.parts.extent.some(n=>!Number.isInteger(n)||n<1||n>256))throw Error('GPU indexed source cell parts require an integer 1..256 extent');
  bindingBytes(value.parts.cellCodes,'source cell ownership',vertexCapacity*4);
 }
 if(value.restCellChart){
  if(value.restCellChart.kind!=='voxel-rest-cell-chart'||value.restCellChart.cornerOrder!=='shallot-voxel-quad-v1'||value.restCellChart.extent.length!==3||value.restCellChart.extent.some(n=>!Number.isInteger(n)||n<1||n>256))throw Error('GPU indexed source rest-cell chart has an unsupported kind, corner order or extent');
  bindingBytes(value.restCellChart.vertexCellCodes,'source rest-cell vertex codes',vertexCapacity*4);
  integer(value.restCellChart.sourceGeneration,'rest-cell chart source generation');
  if(value.restCellChart.sourceGeneration!==value.generation)throw Error('GPU indexed source rest-cell chart names the wrong topology generation');
 }
 if(value.poseContent){
  validatePoseContent(value.poseContent);
  if(value.poseContent.sourceGeneration!==value.generation)throw Error('GPU indexed source pose-content certificate names the wrong topology generation');
 }
 if(value.registrationIndexCount!==undefined){
  integer(value.registrationIndexCount,'registration index count');
  if(value.registrationIndexCount!==0&&(value.registrationIndexCount<3||value.registrationIndexCount%3||value.registrationIndexCount>indexCapacity))throw Error('GPU indexed source registration count must contain whole triangles within topology capacity');
 }
}

function validatePose(source:GpuIndexedSourceReady,pose:GpuIndexedSourcePose):void{
 if(pose.sourceId!==source.sourceId||pose.generation!==source.generation)throw Error('GPU indexed source pose is for a different topology generation');
 integer(pose.posedRevision,'posed revision');if(pose.posedContentRevision!==undefined)integer(pose.posedContentRevision,'posed content revision');integer(pose.vertexCapacity,'posed vertex capacity',1);
 if(pose.vertexCapacity<source.topology.vertexCapacity)throw Error('GPU indexed source pose capacity is smaller than topology capacity');
 const count=pose.singleInstanceVertexCount??pose.vertexCapacity;
 if(pose.singleInstanceVertexCount!==undefined){
  integer(count,'single-instance posed vertex count',1);
  if(source.topology.indexPattern!=='independent-quads'||source.incidence.layout.corners%6||count!==source.incidence.layout.corners/6*4||count>pose.vertexCapacity)
   throw Error('Compact posed prefix must cover the complete single-instance quad source');
 }
 bindingBytes(pose.vertices,'posed vertices',count*16);
 bindingBytes(pose.attributes,'posed attributes',count*16);
 bindingBytes(pose.metadata,'posed draw metadata',32);
}

/** One residency owner over the existing SourceStorage/GpuSourceIncidence.
 * It retains only GPU bindings and the CSR range; topology is never copied to
 * CPU and no parallel source graph is allocated. */
export class GpuIndexedSourceResidency{
 private readonly ready=new Map<string,GpuIndexedSourceReady>();
 private readonly pending=new Map<string,Promise<GpuIndexedSourceReady|GpuIndexedSourceWaiting|GpuIndexedSourceEmpty>>();
 private disposed=false;
 private partState?:SourceStorage;
 constructor(private readonly storage:SourceStorage,private readonly incidence:GpuIndexedSourceCompiler){}
 async register(descriptor:GpuIndexedSourceDescriptor):Promise<GpuIndexedSourceReady|GpuIndexedSourceWaiting|GpuIndexedSourceEmpty>{
  if(this.disposed)throw Error('GPU indexed source residency is disposed');
  validateGpuIndexedSourceDescriptor(descriptor);
  const key=`${descriptor.sourceId}@${descriptor.generation}`;
  const queued=this.pending.get(key);if(queued)return queued;
  const task=this.registerRevision(descriptor);this.pending.set(key,task);void task.then(()=>this.pending.delete(key),()=>this.pending.delete(key));return task;
 }
 private async registerRevision(descriptor:GpuIndexedSourceDescriptor):Promise<GpuIndexedSourceReady|GpuIndexedSourceWaiting|GpuIndexedSourceEmpty>{
  const previous=this.ready.get(descriptor.sourceId);
  if(previous){
   if(descriptor.generation<previous.generation)throw Error('GPU indexed source generation moved backwards');
   if(descriptor.generation===previous.generation){
    const a=descriptor.restCellChart,b=previous.restCellChart;
    if(descriptor.revision!==previous.revision||descriptor.posedNormalClass!==previous.posedNormalClass||descriptor.edgeContactClass!==previous.edgeContactClass||!sameBinding(descriptor.topology.indices,previous.topology.indices)||!!a!==!!b||!!a&&!!b&&(a.sourceGeneration!==b.sourceGeneration||a.cornerOrder!==b.cornerOrder||a.extent.some((n,i)=>n!==b.extent[i])||!sameBinding(a.vertexCellCodes,b.vertexCellCodes)))throw Error('GPU indexed source changed without advancing generation');
    if(descriptor.registrationIndexCount===undefined)return previous;
    if(descriptor.registrationIndexCount!==previous.incidence.layout.corners)throw Error('GPU indexed source live topology changed without advancing generation');
    return previous;
   }
  }
  if(descriptor.registrationIndexCount===undefined)return {state:'waiting-for-live-index-count',sourceId:descriptor.sourceId,generation:descriptor.generation,reason:'producer-registration-count-not-published'};
  if(descriptor.registrationIndexCount===0){if(previous)this.retireOwned(previous);this.ready.delete(descriptor.sourceId);return {state:'empty',sourceId:descriptor.sourceId,revision:descriptor.revision,generation:descriptor.generation,topology:descriptor.topology,liveIndexCount:descriptor.liveIndexCount};}
  const incidence=await compileGpuIndexedSource(this.incidence,{indices:descriptor.topology.indices,vertexCapacity:descriptor.topology.vertexCapacity,indexCount:descriptor.registrationIndexCount,edgeCapacity:descriptor.registrationIndexCount,indexPattern:descriptor.topology.indexPattern});
  if(this.disposed){this.storage.retire(incidence.range);throw Error('GPU indexed source residency retired during registration');}
  // The mesh may have retired/reappeared while its GPU build was pending.
  // Displace the CURRENT owner, never retire a captured range a second time.
  const current=this.ready.get(descriptor.sourceId);
  if(current&&current.generation>=descriptor.generation){this.storage.retire(incidence.range);return current;}
  let parts:GpuIndexedSourceReady['parts'];
  if(descriptor.parts){
   try{
    const state=this.partState??=this.storage.separate('MAIN source part LOD state');
    const count=descriptor.parts.extent.reduce((n,v)=>n*v,1),range=state.allocate(count*4);
    state.write(range,new Uint32Array(count));parts={...descriptor.parts,range,levels:state.binding(range)};
   }catch(error){this.storage.retire(incidence.range);throw error;}
  }
  if(current)this.retireOwned(current);
  const installed:GpuIndexedSourceReady={state:'ready',apiVersion:GPU_INDEXED_SOURCE_API_VERSION,sourceId:descriptor.sourceId,revision:descriptor.revision,generation:descriptor.generation,topology:descriptor.topology,liveIndexCount:descriptor.liveIndexCount,incidence,constructionKind:descriptor.constructionKind,posedNormalClass:descriptor.posedNormalClass,edgeContactClass:descriptor.edgeContactClass,parts,restCellChart:descriptor.restCellChart};
  this.ready.set(descriptor.sourceId,installed);return installed;
 }
 /** Attach the material vertex-stage pose without copying it or changing the
  * registered topology identity. The returned batch is a frame-local view. */
 pose(source:GpuIndexedSourceReady,pose:GpuIndexedSourcePose):GpuIndexedSourceBatch{
  if(this.disposed)throw Error('GPU indexed source residency is disposed');
  if(this.ready.get(source.sourceId)!==source)throw Error('GPU indexed source is not the current resident revision');
  validatePose(source,pose);return {sourceId:source.sourceId,revision:source.revision,generation:source.generation,topology:source.topology,liveIndexCount:source.liveIndexCount,incidence:source.incidence,pose,constructionKind:source.constructionKind,posedNormalClass:source.posedNormalClass,edgeContactClass:source.edgeContactClass,parts:source.parts,restCellChart:source.restCellChart};
 }
 current(sourceId:string):GpuIndexedSourceReady|undefined{return this.ready.get(sourceId);}
 private retireOwned(source:GpuIndexedSourceReady):void{this.storage.retire(source.incidence.range);if(source.parts)this.partState!.retire(source.parts.range);}
 retire(sourceId:string,generation?:number):void{const source=this.ready.get(sourceId);if(!source||generation!==undefined&&source.generation!==generation)return;this.retireOwned(source);this.ready.delete(sourceId);}
 submitted():void{this.storage.submitted();this.partState?.submitted();}
 dispose():void{if(this.disposed)return;this.disposed=true;this.pending.clear();for(const source of this.ready.values())this.retireOwned(source);this.ready.clear();this.partState?.dispose();}
}

/** CPU fixture oracle for tests and integration assertions: an indexed source
 * has no rest-position/empty-complex fallback and its only identity is the
 * producer's stable source id plus generation. */
export function gpuIndexedSourceIdentity(source:Pick<GpuIndexedSourceDescriptor,'sourceId'|'generation'|'revision'>):string{
 validateProducerIdentity(source);return `${source.sourceId}@${source.generation}#${source.revision}`;
}
function validateProducerIdentity(source:Pick<GpuIndexedSourceDescriptor,'sourceId'|'generation'|'revision'>):void{
 if(typeof source.sourceId!=='string'||source.sourceId.length===0)throw Error('GPU indexed source requires a stable producer sourceId');
 integer(source.generation,'generation');integer(source.revision,'revision');
}
