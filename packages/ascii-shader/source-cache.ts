import type {MaterialGeometryDraw} from './geometry.ts';
import type {MeshSourceComplex} from './source-complex.ts';
import type {CompositionGeometry,SourceAnalysis} from './composition.ts';
import type {CompiledSourceProgram,SourceComplexRetention,PlaybackSourceAnalysis} from './source-program.ts';
import type {SourceResidencySummary} from './source-transfer.ts';
import {SourceProgramCompiler,type SourceCompilation} from './source-program-compiler.ts';
import {SourceStorage,type SourceStorageRange} from './source-storage.ts';
import {GpuRelationSourceCompiler} from './source-incidence-residual.ts';
import {GpuIndexedSourceResidency,type GpuIndexedSourceDescriptor,type GpuIndexedSourcePose,type GpuIndexedSourceReady,type GpuIndexedSourceBatch} from './gpu-indexed-source.ts';
import {GpuSourceNumerics,type NumericSourceReady,type NumericSourceCompilation} from './gpu-source-numerics.ts';
import {decodeSourceSemanticResidual,semanticProfileSafe,type SourceSemanticResidual} from './source-semantic-residual.ts';
import {sourcePartAnchorVertices} from './compiled-construction-topology.ts';
import {bindSourceWholeConstruction,type SourceLodSelection,type SourceWholeConstructionTemplate} from './source-lod.ts';
import type {SourceCornerIndexData} from './source-contact-corners.ts';
import type {SourceSmallParent} from './source-small-parent.ts';

type SourceData=NonNullable<MaterialGeometryDraw['sourceGeometry']>&Pick<CompositionGeometry,'lineage'|'surface'|'materialSource'|'construction'>;
interface ResidentSourceData {
 readonly id:string;readonly domain:number;readonly revision:number;
 readonly geometry:SourceData;readonly buffer:GPUBuffer;readonly bufferBinding?:GPUBufferBinding;
 /** Compiled source-revision parent adequacy; absent is unknown, never safe. */
 readonly profileSafe?:boolean;
 /** Compiled once per source revision; absent for legacy/numeric residency. */
 readonly cornerIndex?:SourceCornerIndexData;
 /** `empty` is a completed opt-in for a zero-vertex source; it must not retry. */
 readonly cornerIndexState?:'ready'|'empty';
 /** Source-revision parent graph emitted by the production worker. */
 readonly wholeConstruction?:SourceWholeConstructionTemplate;
 readonly smallParent?:SourceSmallParent;
 /** Original source vertices for declared part extents, compiled on change. */
 readonly partAnchorVertices?:Uint32Array;
}
export interface ResidentSourceComplex extends ResidentSourceData {readonly complex:MeshSourceComplex}
export interface ResidentSourceSummary extends ResidentSourceData {readonly summary:SourceResidencySummary;readonly complex?:never}
export interface ResidentGpuIndexedSource {
 readonly kind:'gpu-indexed';readonly state:'ready';readonly id:string;readonly domain:number;readonly revision:number;readonly generation:number;
 readonly gpu:GpuIndexedSourceReady;readonly buffer:GPUBuffer;readonly bufferBinding:GPUBufferBinding;readonly indexBinding:GPUBufferBinding;
 readonly summary:{readonly schema:'shallot.source-residency-summary.v1';readonly edgeCount:number;readonly vertexCount:number;readonly faceCount:number};
}
export interface ResidentGpuIndexedEmpty {readonly kind:'gpu-indexed';readonly state:'empty';readonly id:string;readonly domain:number;readonly revision:number;readonly generation:number;}
/** Compiled smallest-tier source form, not a fake/empty incidence packet.
 * The complete sub-cell form joins the native skyline regardless of its
 * original class. Larger forms retain their original typed source program. */
export interface ResidentCompiledFarForm {
 readonly kind:'compiled-far-form';readonly id:string;readonly domain:number;readonly revision:number;readonly geometry:SourceData;
 /** Present for baked assets; live sources retain their worker analysis. */
 readonly semantic?:SourceSemanticResidual;readonly complex?:never;
 readonly analysis:Pick<PlaybackSourceAnalysis,'type'|'styleWord'|'foliage'|'organic'|'toneRegions'>;
 readonly wholeConstruction?:SourceWholeConstructionTemplate;
 readonly smallParent?:SourceSmallParent;
}
export function isCompiledFarForm(source:unknown):source is ResidentCompiledFarForm{return !!source&&typeof source==='object'&&'kind' in source&&source.kind==='compiled-far-form';}
 export type ResidentRelationSource=ResidentSourceComplex|ResidentSourceSummary|ResidentGpuIndexedSource;
 type CpuRegisteredSource<R extends SourceComplexRetention='full'>=R extends 'full'
  ?ResidentSourceComplex&{readonly analysis:SourceAnalysis}
  :ResidentSourceSummary&{readonly analysis:PlaybackSourceAnalysis};
 export type RegisteredSource<R extends SourceComplexRetention='full'>=CpuRegisteredSource<R>|ResidentGpuIndexedSource|ResidentGpuIndexedEmpty;
export function isGpuIndexedResident(source:ResidentRelationSource):source is ResidentGpuIndexedSource;
export function isGpuIndexedResident(source:ResidentRelationSource|ResidentGpuIndexedEmpty|ResidentCompiledFarForm):source is ResidentGpuIndexedSource|ResidentGpuIndexedEmpty;
export function isGpuIndexedResident(source:ResidentRelationSource|ResidentGpuIndexedEmpty|ResidentCompiledFarForm):source is ResidentGpuIndexedSource|ResidentGpuIndexedEmpty{return 'kind' in source&&source.kind==='gpu-indexed';}
/** Standard Shallot registration also supplies its completed classification.
 * The physical adapter's existing resident interface is unchanged. */
export type RegisteredSourceComplex<R extends SourceComplexRetention='full'>=CpuRegisteredSource<R>;
interface Registration {
 readonly id:number;readonly mesh?:number;readonly data:SourceData;readonly revision:number;
 readonly promise:Promise<void>;readonly resolve:()=>void;readonly reject:(error:unknown)=>void;
 error?:Error;
 compilation?:SourceCompilation;
 numeric?:NumericSourceCompilation;
 deferred?:boolean;
 contactCorners:boolean;
}
/** Source revision work belongs to the registration worker, not acquire/draw.
 * Incidence, plane analysis and part graphs share one source compilation. */
export class MeshSourceCache<R extends SourceComplexRetention='full'> {
 private readonly meshSources=new Map<number,{data:object;resident:RegisteredSource<R>}>();
 private readonly unregisteredSources=new WeakMap<object,RegisteredSource<R>>();
 private readonly meshWork=new Map<number,Registration>();
 private readonly unregisteredWork=new WeakMap<object,Registration>();
 private readonly work=new Map<number,Registration>();
 private readonly buffers=new Set<GPUBuffer>();
 private readonly domains=new Set<number>();
 private readonly byDomain=new Map<number,RegisteredSource<R>>();
 private readonly compiled:RegisteredSource<R>[]=[];
 private retired:GPUBuffer[]=[];private serial=0;private disposed=false;
 private readonly compiler=new SourceProgramCompiler();
 private readonly storage?:SourceStorage;private readonly incidence?:GpuRelationSourceCompiler;
 private readonly gpuResidency?:GpuIndexedSourceResidency;
 private numerics?:GpuSourceNumerics;
 private readonly numericSources=new WeakMap<ResidentRelationSource,NumericSourceReady>();
 private readonly sourceForms=new WeakMap<object,ResidentCompiledFarForm>();
 private readonly meshForms=new Map<number,{data:object;form:ResidentCompiledFarForm}>();
 /** Host-selected solver meshes. This set is intentionally independent of LOD. */
 private readonly contactCornerMeshes=new Set<number>();
 private readonly gpuSources=new Map<string,{mesh?:number;resident:ResidentGpuIndexedSource|ResidentGpuIndexedEmpty}>();
 private readonly gpuOwners=new Map<string,number|undefined>();
 private readonly gpuWork=new Map<string,Promise<void>>();
 private readonly gpuErrors=new Map<string,Error>();
 private readonly ranges=new WeakMap<ResidentRelationSource,SourceStorageRange>();
 private preparation?:{readonly waits:ReadonlySet<Promise<void>>;readonly promise:Promise<void>};
 static async create<R extends SourceComplexRetention='full'>(device:GPUDevice,backend:'legacy'|'gpu-csr'='legacy',retention:R='full' as R,progress?:(fraction:number)=>void):Promise<MeshSourceCache<R>>{
  if(backend!=='gpu-csr'){const result=new MeshSourceCache(device,backend,retention);progress?.(1);return result;}
  const storage=new SourceStorage(device,'MAIN source incidence');let incidence:GpuRelationSourceCompiler|undefined;
  try{incidence=await GpuRelationSourceCompiler.create(device,storage,progress);return new MeshSourceCache(device,backend,retention,{storage,incidence});}
  catch(error){if(incidence)await incidence.dispose();storage.dispose();throw error;}
 }
 constructor(private readonly device:GPUDevice,private readonly backend:'legacy'|'gpu-csr'='legacy',private readonly retention:R='full' as R,prepared?:{readonly storage:SourceStorage;readonly incidence:GpuRelationSourceCompiler}){
  if(retention==='summary'&&backend!=='gpu-csr')throw Error('Summary residency requires compiled GPU incidence');
  if(backend==='gpu-csr'){this.storage=prepared?.storage??new SourceStorage(device,'MAIN source incidence');this.incidence=prepared?.incidence??new GpuRelationSourceCompiler(device,this.storage);this.gpuResidency=new GpuIndexedSourceResidency(this.storage,this.incidence);}
 }
 private resident(draw:MaterialGeometryDraw):RegisteredSource<R>|undefined{
  const descriptor=draw.gpuIndexedSource;
  if(descriptor){const found=this.gpuSources.get(descriptor.sourceId)?.resident;if(found&&found.revision===descriptor.revision&&found.generation===descriptor.generation)return found;return;}
  const data=draw.sourceGeometry;if(!data)return;
  const registered=draw.mesh===undefined?undefined:this.meshSources.get(draw.mesh);
  const previous=registered?.resident??(draw.mesh===undefined?this.unregisteredSources.get(data):undefined);
  if(previous?.revision===data.revision&&(!registered||registered.data===data))return previous;
 }
 private numericPath(data:SourceData):boolean{
  return this.retention==='summary'&&!!this.storage&&!!this.incidence&&!!data.semantic&&!data.lineage?.adjacency;
 }
 /** Both baked and worker-compiled sources feed the same parent adequacy
  * contract. This lookup never compiles geometry on the camera/draw path. */
 profileSafe(draw:MaterialGeometryDraw):boolean|undefined{
  if(draw.gpuIndexedSource)return;
  const baked=semanticProfileSafe(draw.sourceGeometry?.semantic);if(baked!==undefined)return baked;
  const source=this.resident(draw);return source&&!isGpuIndexedResident(source)?source.profileSafe:undefined;
 }
 /** Bind the resident source graph to the concrete instance identity used by
  * the octree. This is a lookup over immutable metadata, never a camera-time
  * geometry scan or a second compiler. */
 wholeConstruction(draw:MaterialGeometryDraw,mesh:number,eid:number,worldFromSource?:ArrayLike<number>){
  if(draw.gpuIndexedSource)return;
  const source=this.resident(draw);
  // Baked metadata is already compiled, even while numeric GPU residency is
  // pending. Use the same cached immutable owner as farForm; otherwise the
  // octree frontier sees no construction that the emitter already holds.
  const template=source&&!isGpuIndexedResident(source)&&source.wholeConstruction?source.wholeConstruction:this.form(draw)?.wholeConstruction;
  return bindSourceWholeConstruction(template,mesh,eid,worldFromSource);
 }
 /** Source-revision metadata only. The posed LOD consumer owns whether its
  * small construction can preserve the current form; this never compiles. */
 smallParent(draw:MaterialGeometryDraw):SourceSmallParent|undefined{
  if(draw.gpuIndexedSource)return;
  const source=this.resident(draw);
  return source&&!isGpuIndexedResident(source)&&source.smallParent?source.smallParent:this.form(draw)?.smallParent;
 }
 acquire(draw:MaterialGeometryDraw):RegisteredSource<R>|undefined;
 acquire(draw:MaterialGeometryDraw,selection:SourceLodSelection|undefined):RegisteredSource<R>|ResidentCompiledFarForm|undefined;
 acquire(draw:MaterialGeometryDraw,selection?:SourceLodSelection):RegisteredSource<R>|ResidentCompiledFarForm|undefined{
  if(this.disposed)throw Error('Source registration is disposed');
  if(!draw.sourceGeometry&&!draw.gpuIndexedSource)return;
  if(draw.gpuIndexedSource)return this.resident(draw);
  if(draw.mesh!==undefined&&selection?.outside?.has(draw.mesh))return;
  // A MIXED mesh (perf-impl-94) returns its full resident; farForm draws its far members.
  if(draw.mesh!==undefined&&selection?.far.has(draw.mesh)&&!selection.full?.has(draw.mesh)){const form=this.form(draw);if(form)return form;}
  const resident=this.resident(draw);if(!resident)throw Error('Source revision is not registered; prepare the complete camera sources before drawing');
  return resident;
 }
 /**
  * Request one source-revision-owned corner index for a mesh before the next
  * prepare. The host calls this only for an actual Shape + MeshInstance
  * pairing. Existing residents are recompiled asynchronously by prepare;
  * acquire/draw never builds the index.
  */
 requestContactCorners(mesh:number):void{
  if(!Number.isSafeInteger(mesh)||mesh<0)throw new RangeError('contact corner request requires a mesh handle');
  this.contactCornerMeshes.add(mesh);
 }
 /** The compiled far form of a MIXED mesh's far members (perf-impl-94);
  * undefined when the mesh has none (its members then all stay full). */
 farForm(draw:MaterialGeometryDraw):ResidentCompiledFarForm|undefined{return this.form(draw);}
 private form(draw:MaterialGeometryDraw):ResidentCompiledFarForm|undefined{
  const data=draw.sourceGeometry as SourceData|undefined;if(this.retention!=='summary'||!data||data.surface||data.lineage?.adjacency)return;
  const prior=this.sourceForms.get(data);if(prior?.revision===data.revision)return prior;
  const semantic=data.semantic?decodeSourceSemanticResidual(data.semantic):undefined;
  const registered=this.resident(draw);
  // Ordinary imported examples reach the same frontier after their one
  // source-revision worker compilation. Never compile or scan on acquire.
  const analysis=semantic??(registered&&!isGpuIndexedResident(registered)?registered.analysis:undefined);
  if(!analysis||semantic?.publishedAdjacency||analysis.type==='terrain'||analysis.type==='paper'||analysis.type==='far-strip')return;
  const job=draw.mesh===undefined?this.unregisteredWork.get(data):this.meshWork.get(draw.mesh);
  const domain=registered?.domain??job?.id??++this.serial;if(domain>=0x20000000)throw Error('Source identity capacity exhausted');
 const wholeConstruction=registered&&!isGpuIndexedResident(registered)&&registered.wholeConstruction?registered.wholeConstruction:semantic?.wholeConstruction;
 const smallParent=registered&&!isGpuIndexedResident(registered)&&registered.smallParent?registered.smallParent:semantic?.smallParent;
 const form:ResidentCompiledFarForm={kind:'compiled-far-form',id:`mesh-source-${domain}`,domain,revision:data.revision,geometry:registered&&!isGpuIndexedResident(registered)?registered.geometry:data,...(wholeConstruction?{wholeConstruction}:{}),...(smallParent?{smallParent}:{}),...(semantic?{semantic}:{}),
   analysis:{type:analysis.type,styleWord:analysis.styleWord,foliage:analysis.foliage,organic:analysis.organic}};
  this.sourceForms.set(data,form);this.domains.add(domain);
  if(draw.mesh!==undefined){const previous=this.meshForms.get(draw.mesh);if(previous&&previous.form.domain!==domain&&!this.byDomain.has(previous.form.domain))this.domains.delete(previous.form.domain);this.meshForms.set(draw.mesh,{data,form});}
  return form;
 }
 /** Pending means retain the previous target, never omit this source. */
 prepare(draws:readonly MaterialGeometryDraw[],selection?:SourceLodSelection):Promise<void>|undefined{
  if(this.disposed)throw Error('Source registration is disposed');
  const waits=new Set<Promise<void>>();
  for(const draw of draws){
   const descriptor=draw.gpuIndexedSource;
   if(descriptor){
    const error=this.gpuErrors.get(descriptor.sourceId);if(error)throw error;
    // The existing field producer keeps drawing this form while its new
    // topology registers. A streaming source must not stop every other
    // ready source or hold first publication behind a whole-world barrier.
    const task=this.prepareGpu(draw,descriptor);
    if(task)void task.catch(error=>{if(!this.disposed)this.gpuErrors.set(descriptor.sourceId,error instanceof Error?error:Error(String(error)));});
    continue;
   }
   const data=draw.sourceGeometry;if(!data)continue;
   if(draw.mesh!==undefined&&selection?.outside?.has(draw.mesh))continue;
   const far=draw.mesh!==undefined&&selection?.far.has(draw.mesh)&&!selection.full?.has(draw.mesh),defer=draw.mesh!==undefined&&selection?.defer.has(draw.mesh);
   const form=far||defer?this.form(draw):undefined;
   let job=draw.mesh===undefined?this.unregisteredWork.get(data):this.meshWork.get(draw.mesh);
   if(job&&(job.data!==data||job.revision!==data.revision)){this.cancel(job);job=undefined;}
   if(far&&form){if(job?.deferred)this.cancel(job);continue;}
   const resident=this.resident(draw);
   const wantsCorners=draw.mesh!==undefined&&this.contactCornerMeshes.has(draw.mesh)&&!this.numericPath(data);
   const cornerState=resident&&'cornerIndexState' in resident?resident.cornerIndexState:undefined;
   const cornerMissing=wantsCorners&&!!resident&&cornerState===undefined;
   if(job&&wantsCorners&&!job.contactCorners){this.cancel(job);job=undefined;}
   if(resident&&!cornerMissing&&!job)continue;
   if(job?.error)throw job.error;
   if(!job)job=this.register(draw,data,!!defer&&!!form,resident?.domain,wantsCorners);
   else if(job.deferred&&!(defer&&form))this.start(job);
   waits.add(job.promise);
  }
  if(!waits.size){if(this.preparation)for(const job of this.work.values()){job.compilation?.prioritize(1);job.numeric?.setRequired(false);}this.preparation=undefined;return;}
  const previous=this.preparation;
  if(previous&&previous.waits.size===waits.size&&[...waits].every(wait=>previous.waits.has(wait)))return previous.promise;
  // Match the physical adapter's required cohort. A small weight difference
  // lets every untouched background generator overtake a required source
  // after its first expensive phase under the worker's service fairness.
  for(const job of this.work.values()){const required=waits.has(job.promise);job.compilation?.prioritize(required?1_000_000:1);job.numeric?.setRequired(required);}
  // One dependency barrier per actual required-source set, not another
  // Promise.all subscription for every source on every displayed host tick.
  // Every wait remains present; an arrival/revision/retirement changes the set.
  const promise=Promise.all(waits).then(()=>undefined);
  this.preparation={waits,promise};
  const clear=()=>{if(this.preparation?.promise===promise)this.preparation=undefined;};
  void promise.then(clear,clear);return promise;
 }
 private prepareGpu(draw:MaterialGeometryDraw,descriptor:GpuIndexedSourceDescriptor):Promise<void>|undefined{
   if(!this.gpuResidency)throw Error('GPU indexed source requires the existing gpu-csr source arena');
   const old=this.gpuSources.get(descriptor.sourceId)?.resident;
   if(old&&old.revision===descriptor.revision&&old.generation===descriptor.generation)return;
   const pending=this.gpuWork.get(descriptor.sourceId);if(pending)return pending;
   this.gpuOwners.set(descriptor.sourceId,draw.mesh);
   const task=this.gpuResidency.register(descriptor).then(result=>{
     if(result.state==='waiting-for-live-index-count')return;
     if(this.disposed||!this.gpuOwners.has(descriptor.sourceId)){this.gpuResidency!.retire(descriptor.sourceId);return;}
     // A compact domain identifies a source GENERATION, not a recyclable
     // producer slot. Reusing it kept retired edge addresses in correspondence
     // forever and confused them with a later chunk's unrelated topology.
     const previous=this.gpuSources.get(descriptor.sourceId)?.resident;
     if(previous&&previous.revision===result.revision&&previous.generation===result.generation)return;
     const domain=++this.serial;if(domain>=0x20000000)throw Error('Source identity capacity exhausted');
     if(previous)this.retire(previous);
     const resident:ResidentGpuIndexedSource|ResidentGpuIndexedEmpty=result.state==='empty'
       ?{kind:'gpu-indexed',state:'empty',id:result.sourceId,domain,revision:result.revision,generation:result.generation}
       :{kind:'gpu-indexed',state:'ready',id:result.sourceId,domain,revision:result.revision,generation:result.generation,gpu:result,buffer:result.incidence.binding.buffer,bufferBinding:result.incidence.binding,indexBinding:result.topology.indices,summary:{schema:'shallot.source-residency-summary.v1',edgeCount:result.incidence.edges,vertexCount:result.topology.vertexCapacity,faceCount:result.incidence.layout.corners/3}};
     this.gpuSources.set(descriptor.sourceId,{mesh:draw.mesh,resident});this.domains.add(domain);this.byDomain.set(domain,resident as RegisteredSource<R>);this.compiled.push(resident as RegisteredSource<R>);
   }).finally(()=>{if(this.gpuWork.get(descriptor.sourceId)===task)this.gpuWork.delete(descriptor.sourceId);});
   this.gpuWork.set(descriptor.sourceId,task);return task;
  }
  pose(source:ResidentGpuIndexedSource,pose:GpuIndexedSourcePose):GpuIndexedSourceBatch{if(!this.gpuResidency)throw Error('GPU indexed source residency is disposed');return this.gpuResidency.pose(source.gpu,pose);}
  /** Drain once per revision, not once per camera/frame. */
 takeCompiled():readonly RegisteredSource<R>[]{return this.compiled.splice(0).filter(source=>this.domains.has(source.domain));}
 private register(draw:MaterialGeometryDraw,data:SourceData,deferred=false,domain?:number,contactCorners=false):Registration{
  const id=domain??this.sourceForms.get(data)?.domain??++this.serial;if(id>=0x20000000)throw Error('Source identity capacity exhausted');
  let resolve!:()=>void,reject!:(error:unknown)=>void;
  const promise=new Promise<void>((yes,no)=>{resolve=yes;reject=no;});
  const job:Registration={id,mesh:draw.mesh,data,revision:data.revision,promise,resolve,reject,deferred,contactCorners};
  this.work.set(id,job);
  if(draw.mesh===undefined)this.unregisteredWork.set(data,job);else this.meshWork.set(draw.mesh,job);
  return deferred?job:this.start(job);
 }
 private start(job:Registration):Registration{
  job.deferred=false;const {id,data}=job;
  const adjacency=data.lineage?.adjacency;
  try{
   // Static source decisions arrive with the immutable authored revision.
   // Do not send these meshes through full CPU source/sheet/part expansion
   // before the first displayed page. Full inspection and uncompiled live
   // sources retain their original worker producer.
   if(this.retention==='summary'&&this.storage&&this.incidence&&data.semantic&&!adjacency){
    const semantic=decodeSourceSemanticResidual(data.semantic);
    if(!semantic.publishedAdjacency){
     this.numerics??=new GpuSourceNumerics(this.device,this.storage,this.incidence);
     const identity=new Uint32Array(data.semantic.buffer,data.semantic.byteOffset+40*4,8);
     job.numeric=this.numerics.register(data,semantic,id,identity);
     void job.numeric.result.then(result=>{
      if(this.work.get(id)!==job||this.disposed){this.numerics!.retire(result);return;}
      const resident={id:`mesh-source-${id}`,domain:id,revision:job.revision,buffer:result.binding.buffer,bufferBinding:result.binding,geometry:data,profileSafe:semantic.profileSafe,...(semantic.wholeConstruction?{wholeConstruction:semantic.wholeConstruction}:{}),...(semantic.smallParent?{smallParent:semantic.smallParent}:{}),
       ...(semantic.partPrograms?{partAnchorVertices:sourcePartAnchorVertices(semantic.partPrograms,data.indices)}:{}),
       summary:{schema:'shallot.source-residency-summary.v1' as const,edgeCount:semantic.indexedEdges,vertexCount:semantic.vertices,faceCount:semantic.corners/3},
       analysis:{type:semantic.type,styleWord:semantic.styleWord,foliage:semantic.foliage,organic:semantic.organic,toneRegions:semantic.toneRegions,runProgram:result.runProgram}} as RegisteredSourceComplex<R>;
      this.numericSources.set(resident,result);this.publish(job,resident);
      this.work.delete(id);this.forget(job);job.resolve();
     }).catch(error=>{if(!this.disposed&&this.work.get(id)===job){this.work.delete(id);job.error=error instanceof Error?error:Error(String(error));job.reject(job.error);}});
     return job;
    }
   }
   job.compilation=this.compiler.compile({id:`mesh-source-${id}`,revision:data.revision,positions:data.positions,indices:data.indices,contactKeys:data.contactKeys,partAdjacency:adjacency,complexMode:this.retention,incidence:this.backend==='gpu-csr'?'gpu-csr':undefined,contactCorners:job.contactCorners,definition:{normals:data.normals,lineage:data.lineage,surface:data.surface,materialSource:data.materialSource,construction:data.construction}});
   void job.compilation.result.then(async program=>{
    if(!this.work.has(id)||this.disposed)return;
    if(program)await this.install(job,program);
    if(!this.work.delete(id)||this.disposed)return;
    this.forget(job);job.resolve();
   }).catch(error=>{if(!this.disposed){this.work.delete(id);job.error=error instanceof Error?error:Error(String(error));job.reject(job.error);}});
  }catch(error){this.work.delete(id);job.error=error instanceof Error?error:Error(String(error));job.reject(job.error);}
  return job;
 }
 private async install(job:Registration,program:CompiledSourceProgram):Promise<void>{
  if(program.complexMode!==this.retention)throw Error('Source compiler returned the wrong residency contract');
  const {packet}=program;let buffer:GPUBuffer,bufferBinding:GPUBufferBinding|undefined,range:SourceStorageRange|undefined;
  if(packet.kind==='gpu-csr-v1'){
   if(!this.incidence)throw Error('Compact source submitted to a legacy registration owner');
   const uploaded=await this.incidence.upload(packet.plan);range=uploaded.range;bufferBinding=uploaded.binding;buffer=bufferBinding.buffer;
   if(this.disposed)return;
   if(this.work.get(job.id)!==job){this.storage!.retire(range);return;}
  }else{
   const words=packet.words;
   if(words.byteLength>this.device.limits.maxStorageBufferBindingSize)throw Error(`Source complex mesh-source-${job.id} exceeds the device storage binding capacity`);
   buffer=this.device.createBuffer({label:`MAIN mesh-source-${job.id} source incidence`,size:Math.max(64,words.byteLength),usage:GPUBufferUsage.STORAGE,mappedAtCreation:true});
   new Uint32Array(buffer.getMappedRange()).set(words);buffer.unmap();this.buffers.add(buffer);
  }
  const base={id:`mesh-source-${job.id}`,domain:job.id,revision:job.revision,buffer,bufferBinding,geometry:program.geometry,profileSafe:program.profileSafe,...(program.wholeConstruction?{wholeConstruction:program.wholeConstruction}:{}),...(program.smallParent?{smallParent:program.smallParent}:{}),...(program.partAnchorVertices?{partAnchorVertices:program.partAnchorVertices}:{}),...(program.cornerIndex?{cornerIndex:program.cornerIndex}:{}),...(program.cornerIndexState?{cornerIndexState:program.cornerIndexState}:{})};
  const installed:RegisteredSourceComplex<SourceComplexRetention>=program.complexMode==='full'
   ?{...base,complex:program.complex,analysis:program.analysis}
   :{...base,summary:program.summary,analysis:program.analysis};
  // The mode check above establishes this cache's generic retention contract.
  const resident=installed as RegisteredSourceComplex<R>;
  if(range)this.ranges.set(resident,range);
  this.publish(job,resident);
 }
 private publish(job:Registration,resident:RegisteredSourceComplex<R>):void{
  const previous=job.mesh===undefined?this.unregisteredSources.get(job.data):this.meshSources.get(job.mesh)?.resident;
  if(previous){this.retire(previous);this.domains.delete(previous.domain);this.byDomain.delete(previous.domain);}
  this.domains.add(job.id);this.byDomain.set(job.id,resident);this.compiled.push(resident);
  // A baked/pending far form may have been acquired before its worker
  // metadata was ready, at this SAME geometry revision. Retire that cached
  // metadata owner before the resident-backed replacement is acquired.
  const form=this.sourceForms.get(job.data);
  if(form){
   this.sourceForms.delete(job.data);
   if(!this.byDomain.has(form.domain))this.domains.delete(form.domain);
   if(job.mesh!==undefined&&this.meshForms.get(job.mesh)?.form===form)this.meshForms.delete(job.mesh);
  }
  if(job.mesh===undefined)this.unregisteredSources.set(job.data,resident);else this.meshSources.set(job.mesh,{data:job.data,resident});
 }
 private forget(job:Registration):void{
  if(job.mesh===undefined){if(this.unregisteredWork.get(job.data)===job)this.unregisteredWork.delete(job.data);}
  else if(this.meshWork.get(job.mesh)===job)this.meshWork.delete(job.mesh);
 }
 private cancel(job:Registration):void{this.work.delete(job.id);this.forget(job);job.compilation?.cancel();job.numeric?.cancel();job.resolve();}
 private retire(source:ResidentRelationSource|ResidentGpuIndexedEmpty):void{
   if(isGpuIndexedResident(source)){this.gpuResidency?.retire(source.id,source.generation);if(this.gpuSources.get(source.id)?.resident===source)this.gpuSources.delete(source.id);if(this.byDomain.get(source.domain)===source){this.domains.delete(source.domain);this.byDomain.delete(source.domain);}return;}
   const numeric=this.numericSources.get(source);if(numeric){this.numericSources.delete(source);this.numerics!.retire(numeric);return;}
   const range=this.ranges.get(source);if(range){this.ranges.delete(source);this.storage!.retire(range);}else this.retired.push(source.buffer);
  }
 collect(isLiveMesh:(mesh:number)=>boolean):void{
  for(const [mesh,{form}]of this.meshForms)if(!isLiveMesh(mesh)){this.meshForms.delete(mesh);this.contactCornerMeshes.delete(mesh);if(!this.byDomain.has(form.domain))this.domains.delete(form.domain);}
  for(const mesh of this.contactCornerMeshes)if(!isLiveMesh(mesh)&&!this.meshSources.has(mesh)&&!this.meshWork.has(mesh))this.contactCornerMeshes.delete(mesh);
  for(const [id,mesh]of this.gpuOwners)if(mesh!==undefined&&!isLiveMesh(mesh)){
   this.gpuOwners.delete(id);const source=this.gpuSources.get(id)?.resident;
   if(source)this.retire(source);
  }
  for(const [mesh,job]of this.meshWork)if(!isLiveMesh(mesh))this.cancel(job);
  for(const [mesh,{resident}]of this.meshSources)if(!isLiveMesh(mesh)){this.meshSources.delete(mesh);this.domains.delete(resident.domain);this.byDomain.delete(resident.domain);this.retire(resident);}
 }
 submitted():void{for(const buffer of this.retired){buffer.destroy();this.buffers.delete(buffer);}this.retired=[];this.numerics?.submitted();if(this.gpuResidency)this.gpuResidency.submitted();else this.storage?.submitted();}
 isLive(domain:number):boolean{return this.domains.has(domain);}
 component(domain:number,face:number):number|undefined{
  const source=this.byDomain.get(domain);if(!source)return;
  if(isGpuIndexedResident(source))throw Error('GPU indexed source has no CPU component table; use GPU-derived topology');
  if(!source.complex)throw Error('CPU component inspection requires full source residency');
  return source.complex.faceComponents[face];
 }
 liveDomains():ReadonlySet<number>{return new Set(this.domains);}
 dispose():void{
  this.gpuOwners.clear();this.gpuSources.clear();this.gpuResidency?.dispose();
  this.disposed=true;
  for(const source of this.gpuSources.values())if(source.resident.state==='ready')this.gpuResidency?.retire(source.resident.id);
  this.gpuSources.clear();for(const job of this.work.values())job.resolve();this.work.clear();this.meshWork.clear();this.compiler.dispose();
  if(this.incidence)void (async()=>{await this.numerics?.dispose();await this.incidence!.dispose();this.storage!.dispose();})();
  for(const buffer of this.buffers)buffer.destroy();this.buffers.clear();this.domains.clear();this.byDomain.clear();this.meshSources.clear();this.meshForms.clear();this.contactCornerMeshes.clear();this.retired=[];this.compiled.length=0;
 }
}
