import {
 GPU_INDEXED_SOURCE_API_VERSION,
 type GpuIndexedSourceDescriptor,
 type GpuIndexedSourcePoseContent,
 type GpuIndexedSourcePoseContentInput,
 type GpuIndexedRestCellChart,
 type GpuIndexedRestCellChartInput,
 type GpuIndexFormat,
 validateGpuIndexedSourceDescriptor,
} from './gpu-indexed-source.ts';
import {sameStorageBinding} from './source-storage.ts';

/** The producer-side identity is deliberately separate from Mesh.name.  A
 * voxel slot may reuse the display label `chunk` while its source identity
 * remains stable for the slot's lifetime. */
export interface GpuIndexedSourceProducerInput {
 readonly sourceId:string;
 /** Canonical producer key for the emitted topology.  Include every input
  * that changes indexed topology (for voxel this includes x, z and allFaces). */
 readonly topologyKey:string;
 readonly indexCapacity:number;
 readonly vertexCapacity:number;
 readonly indices:GPUBufferBinding;
 readonly liveIndexCount:GPUBufferBinding;
 /** A registration snapshot is optional until the producer has published its
  * compact count.  It is never inferred from capacity. */
 readonly registrationIndexCount?:number;
 readonly constructionKind?:import('./composition.ts').ConstructionType;
 readonly indexPattern?:'independent-quads';
 readonly posedNormalClass?:'face-constant';
 readonly edgeContactClass?:'posed-coincident';
 readonly parts?:import('./gpu-indexed-source.ts').GpuIndexedClosedCellParts;
 readonly restCellChart?:GpuIndexedRestCellChartInput;
 /** Optional explicit material-pose certificate. Unknown/deforming producers
  * leave this absent; it is never inferred from buffers or scene names. */
 readonly poseContent?:GpuIndexedSourcePoseContentInput;
}

export interface GpuIndexedSourceProducerMetadata extends GpuIndexedSourceDescriptor {
 readonly topologyKey:string;
}

export type GpuIndexedSourceRegistration =
 | {readonly state:'ready';readonly source:GpuIndexedSourceProducerMetadata}
 | {readonly state:'empty';readonly source:GpuIndexedSourceProducerMetadata}
 | {readonly state:'waiting-for-live-index-count';readonly source:GpuIndexedSourceProducerMetadata};

const integer=(value:number,name:string,min=0)=>{
 if(!Number.isSafeInteger(value)||value<min)throw Error(`GPU indexed source producer ${name} must be a non-negative integer`);
 return value;
};
const finite=(value:number,name:string):number=>{
 if(!Number.isFinite(value))throw Error(`GPU indexed source producer ${name} must be finite`);
 return value;
};
function validatePoseContentInput(value:GpuIndexedSourcePoseContentInput):void{
 if(value.kind!=='voxel-terrain-material')throw Error('Unsupported GPU indexed source producer pose-content certificate');
 integer(value.vertexRevision,'pose-content vertex revision');
 if(value.instanceTransform.length!==10||value.instanceTransform.some((n)=>!Number.isFinite(n)))throw Error('GPU indexed source producer pose-content transform must contain ten finite values');
 if(value.materialParameters.length!==2||value.materialParameters.some((n)=>!Number.isFinite(n)))throw Error('GPU indexed source producer pose-content material parameters must contain two finite values');
}
function validateRestCellChartInput(value:GpuIndexedRestCellChartInput,vertexCapacity:number):void{
 if(value.kind!=='voxel-rest-cell-chart'||value.cornerOrder!=='shallot-voxel-quad-v1')throw Error('Unsupported GPU indexed source producer rest-cell chart corner order');
 if(value.extent.length!==3||value.extent.some(n=>!Number.isInteger(n)||n<1||n>256))throw Error('GPU indexed source producer rest-cell chart requires an integer 1..256 extent');
 const offset=value.vertexCellCodes.offset??0,size=value.vertexCellCodes.size??value.vertexCellCodes.buffer.size-offset;
 if(!Number.isSafeInteger(offset)||offset<0||offset%4||!Number.isSafeInteger(size)||size<vertexCapacity*4||offset+size>value.vertexCellCodes.buffer.size)
  throw Error('GPU indexed source producer rest-cell chart binding is shorter than vertex capacity');

 if(value.occupancy){
  if(value.occupancy.kind!=='voxel-column-occupancy-v1')throw Error('Unsupported GPU indexed source producer rest-cell occupancy kind');
  const at=value.occupancy.columns.offset??0,bytes=value.occupancy.columns.size??value.occupancy.columns.buffer.size-at,need=value.extent[0]*value.extent[2]*Math.ceil(value.extent[1]/32)*4;
  if(!Number.isSafeInteger(at)||at<0||at%4||!Number.isSafeInteger(bytes)||bytes<need||at+bytes>value.occupancy.columns.buffer.size)
   throw Error('GPU indexed source producer rest-cell occupancy binding is shorter than its extent');
 }
}
function cloneRestCellChartInput(value:GpuIndexedRestCellChartInput):GpuIndexedRestCellChartInput{
 return {kind:value.kind,cornerOrder:value.cornerOrder,extent:[...value.extent] as GpuIndexedRestCellChartInput['extent'],vertexCellCodes:value.vertexCellCodes,...(value.occupancy?{occupancy:{kind:value.occupancy.kind,columns:value.occupancy.columns}}:{})};
}
function sameRestCellChart(a:GpuIndexedRestCellChart|GpuIndexedRestCellChartInput,b:GpuIndexedRestCellChart|GpuIndexedRestCellChartInput):boolean{
 return a.kind===b.kind&&a.cornerOrder===b.cornerOrder&&a.extent.every((value,index)=>value===b.extent[index])&&sameStorageBinding(a.vertexCellCodes,b.vertexCellCodes)&&
  !!a.occupancy===!!b.occupancy&&(!a.occupancy||!b.occupancy||a.occupancy.kind===b.occupancy.kind&&sameStorageBinding(a.occupancy.columns,b.occupancy.columns));
}
function samePoseContent(a:GpuIndexedSourcePoseContent,b:GpuIndexedSourcePoseContentInput,sourceGeneration:number):boolean{
 return a.sourceGeneration===sourceGeneration&&a.kind===b.kind&&a.vertexRevision===b.vertexRevision&&
  a.instanceTransform.every((value,index)=>value===b.instanceTransform[index])&&
  a.materialParameters.every((value,index)=>value===b.materialParameters[index]);
}
const bindingSize=(binding:GPUBufferBinding):number=>{
 const offset=binding.offset??0,size=binding.size??binding.buffer.size-offset;
 if(!Number.isSafeInteger(offset)||offset<0||offset%4||!Number.isSafeInteger(size)||size<4||offset+size>binding.buffer.size)
  throw Error('GPU indexed source producer binding is not an aligned in-range buffer extent');
 return size;
};
/** Mesh.indirectIndexCount intentionally omits `size`: it names a u32 word
 * inside a producer-owned counter buffer.  Validate the word extent, then
 * compare only the buffer identity and aligned word offset.  The topology
 * index binding remains strict (`sameStorageBinding`) because its extent is
 * part of the registered topology. */
function sameCountWordBinding(a:GPUBufferBinding,b:GPUBufferBinding):boolean{
 bindingSize(a);bindingSize(b);
 return a.buffer===b.buffer&&(a.offset??0)===(b.offset??0);
}
function validateInput(input:GpuIndexedSourceProducerInput):void{
 if(typeof input.sourceId!=='string'||input.sourceId.length===0)throw Error('GPU indexed source producer requires a stable sourceId');
 if(typeof input.topologyKey!=='string'||input.topologyKey.length===0)throw Error('GPU indexed source producer requires a topologyKey');
 if(input.posedNormalClass!==undefined&&input.posedNormalClass!=='face-constant')throw Error('Unsupported GPU indexed source producer posed-normal class');
 const indices=integer(input.indexCapacity,'index capacity',3),vertices=integer(input.vertexCapacity,'vertex capacity',1);
 if(indices%3)throw Error('GPU indexed source producer index capacity must contain whole triangles');
 if(bindingSize(input.indices)<indices*4)throw Error('GPU indexed source producer index binding is too short');
 if(bindingSize(input.liveIndexCount)<4)throw Error('GPU indexed source producer live count binding is too short');
 if(input.registrationIndexCount!==undefined)validateRegistrationCount(input.registrationIndexCount,indices);
 if(input.restCellChart)validateRestCellChartInput(input.restCellChart,vertices);
}
function validateRegistrationCount(count:number,capacity:number):void{
 integer(count,'registration index count');
 if(count>capacity||count%3)throw Error('GPU indexed source producer registration count must be a whole-triangle extent within capacity');
}

/** Stable, order-independent encoding for producer inputs.  The producer
 * decides the fields; this helper prevents object insertion order from
 * becoming a hidden generation source. */
export function gpuIndexedSourceTopologyKey(fields:Readonly<Record<string,string|number|boolean>>):string{
 return JSON.stringify(Object.entries(fields).sort(([a],[b])=>a.localeCompare(b)));
}

type RecordState={
 topologyKey:string;
 source:GpuIndexedSourceProducerMetadata;
};

type PoseContentState={
 readonly generation:number;
 readonly input:GpuIndexedSourcePoseContentInput|undefined;
 readonly revision:number;
};

function clonePoseContentInput(value:GpuIndexedSourcePoseContentInput):GpuIndexedSourcePoseContentInput{
 return {
  kind:value.kind,
  vertexRevision:value.vertexRevision,
  instanceTransform:[...value.instanceTransform] as GpuIndexedSourcePoseContentInput['instanceTransform'],
  materialParameters:[...value.materialParameters] as GpuIndexedSourcePoseContentInput['materialParameters'],
 };
}

/** Tracks producer topology without owning a second mesh or source graph.
 * Pose changes and indirect draw-count changes retain a generation. A new
 * topology key, index binding, or capacity starts a new generation. */
export class GpuIndexedSourceTopologyTracker{
 private readonly records=new Map<string,RecordState>();

 observe(input:GpuIndexedSourceProducerInput):GpuIndexedSourceRegistration{
  validateInput(input);
  const previous=this.records.get(input.sourceId);
  const a=previous?.source.parts,b=input.parts;
  const partsChanged=!!a!==!!b||!!a&&!!b&&(a.extent.some((n,i)=>n!==b.extent[i])||!sameStorageBinding(a.cellCodes,b.cellCodes));
  const oldChart=previous?.source.restCellChart,newChart=input.restCellChart;
  const chartChanged=!!oldChart!==!!newChart||!!oldChart&&!!newChart&&(!sameRestCellChart(oldChart,newChart)||oldChart.sourceGeneration!==previous!.source.generation);
  const changed=!previous||previous.topologyKey!==input.topologyKey||
   partsChanged||chartChanged||
   previous.source.topology.indexCapacity!==input.indexCapacity||
   previous.source.topology.vertexCapacity!==input.vertexCapacity||
   previous.source.topology.indexPattern!==input.indexPattern||
   previous.source.posedNormalClass!==input.posedNormalClass||
   previous.source.edgeContactClass!==input.edgeContactClass||
   !sameStorageBinding(previous.source.topology.indices,input.indices);
  if(!changed){
   // The count binding is a live producer binding. It may be updated by the
   // producer without implying a topology change; the registration snapshot
   // remains the first authoritative count for this generation.
   if(previous.source.registrationIndexCount!==undefined&&input.registrationIndexCount!==undefined&&
    previous.source.registrationIndexCount!==input.registrationIndexCount)
    throw Error('GPU indexed source producer registration count changed within a topology generation');
   if(previous.source.registrationIndexCount===undefined&&input.registrationIndexCount!==undefined)
    validateRegistrationCount(input.registrationIndexCount,input.indexCapacity);
   const source={...previous.source,liveIndexCount:input.liveIndexCount,
    ...(previous.source.registrationIndexCount===undefined&&input.registrationIndexCount!==undefined?{registrationIndexCount:input.registrationIndexCount}:{}),};
   const updated={topologyKey:previous.topologyKey,source:Object.freeze(source)};
   this.records.set(input.sourceId,updated);
   return registrationState(updated.source);
  }
  const generation=(previous?.source.generation??0)+1;
  const source:GpuIndexedSourceProducerMetadata=Object.freeze({
   apiVersion:GPU_INDEXED_SOURCE_API_VERSION,
   sourceId:input.sourceId,
   revision:generation,
   generation,
   topologyKey:input.topologyKey,
   topology:{indexFormat:'uint32' as GpuIndexFormat,indices:input.indices,indexCapacity:input.indexCapacity,vertexCapacity:input.vertexCapacity,indexPattern:input.indexPattern},
   liveIndexCount:input.liveIndexCount,
   registrationIndexCount:input.registrationIndexCount,
   constructionKind:input.constructionKind,
   posedNormalClass:input.posedNormalClass,
   edgeContactClass:input.edgeContactClass,
   parts:input.parts,
   restCellChart:input.restCellChart?{...cloneRestCellChartInput(input.restCellChart),sourceGeneration:generation}:undefined,
  });
  // The producer contract accepts zero as a real empty publication. The
  // existing CSR residency is only called for a `ready` non-empty source.
  if(source.registrationIndexCount!==undefined&&source.registrationIndexCount>0)
   validateGpuIndexedSourceDescriptor(source);
  this.records.set(input.sourceId,{topologyKey:input.topologyKey,source});
  return registrationState(source);
 }

 /** Install a compact count obtained from the producer's count word. It is
  * allowed exactly once per topology generation; later reads are rejected so
  * a changing draw count cannot silently redefine the CSR registration. */
 setRegistrationIndexCount(sourceId:string,generation:number,count:number):GpuIndexedSourceRegistration{
  const record=this.records.get(sourceId);
  if(!record||record.source.generation!==generation)throw Error('GPU indexed source producer count belongs to a stale generation');
  validateRegistrationCount(count,record.source.topology.indexCapacity);
  const prior=record.source.registrationIndexCount;
  if(prior!==undefined){if(prior!==count)throw Error('GPU indexed source producer registration count changed within a topology generation');return registrationState(record.source);}
  const source:GpuIndexedSourceProducerMetadata=Object.freeze({...record.source,registrationIndexCount:count});
  this.records.set(sourceId,{topologyKey:record.topologyKey,source});
  if(count>0)validateGpuIndexedSourceDescriptor(source);
  return registrationState(source);
 }

 current(sourceId:string):GpuIndexedSourceProducerMetadata|undefined{return this.records.get(sourceId)?.source;}
 clear():void{this.records.clear();}
}

function registrationState(source:GpuIndexedSourceProducerMetadata):GpuIndexedSourceRegistration{
 if(source.registrationIndexCount===undefined)return {state:'waiting-for-live-index-count',source};
 if(source.registrationIndexCount===0)return {state:'empty',source};
 return {state:'ready',source};
}

export interface GpuIndexedSourceCountReader{
 /** Read only the producer's four-byte indirect index-count word. */
 readIndexCount(binding:GPUBufferBinding):Promise<number>;
}

/** Deduplicates the optional count snapshot by source generation. This is the
 * only CPU readback this contract permits; vertex/index bytes never cross the
 * boundary. */
export class GpuIndexedSourceCountSnapshots{
 private readonly values=new Map<string,number>();
 private readonly pending=new Map<string,Promise<number>>();
 private epoch=0;

 async read(source:GpuIndexedSourceProducerMetadata,reader:GpuIndexedSourceCountReader):Promise<number>{
  const key=`${source.sourceId}@${source.generation}`;
  const value=this.values.get(key);if(value!==undefined)return value;
  const queued=this.pending.get(key);if(queued)return queued;
  const epoch=this.epoch;
  const task=(async()=>{
   const count=await reader.readIndexCount(source.liveIndexCount);
   validateRegistrationCount(count,source.topology.indexCapacity);
   if(this.epoch===epoch)this.values.set(key,count);return count;
  })();
  this.pending.set(key,task);
  void task.then(()=>{if(this.pending.get(key)===task)this.pending.delete(key);},()=>{if(this.pending.get(key)===task)this.pending.delete(key);});
  return task;
 }
 forgetBefore(sourceId:string,generation:number):void{
  for(const key of this.values.keys())if(key.startsWith(`${sourceId}@`)&&Number(key.slice(sourceId.length+1))<generation)this.values.delete(key);
 }
 clear():void{this.epoch++;this.values.clear();this.pending.clear();}
}

export interface GpuIndexedSourceProducerMeshInput{
 readonly indexPattern?:'independent-quads';
 readonly posedNormalClass?:'face-constant';
 readonly edgeContactClass?:'posed-coincident';
 readonly parts?:import('./gpu-indexed-source.ts').GpuIndexedClosedCellParts;
 readonly restCellChart?:GpuIndexedRestCellChartInput;
 readonly poseContent?:GpuIndexedSourcePoseContentInput;
 readonly sourceId:string;
 readonly topologyKey:string;
 readonly indexCapacity:number;
 readonly vertexCapacity:number;
 readonly indices:GPUBufferBinding;
 readonly liveIndexCount:GPUBufferBinding;
 readonly constructionKind?:import('./composition.ts').ConstructionType;
}

/** The small part of World used by the transformed producer. Keeping this
 * structural avoids making the producer depend on the engine package while
 * making frame-submit and disposal ownership explicit. */
export interface GpuIndexedSourceFrameOwner{
 onFrameSubmit(callback:()=>void):void;
 onDispose(callback:()=>void):void;
}

type CountCopy={
 readonly key:string;
 readonly source:GpuIndexedSourceProducerMetadata;
 readonly target:GPUBuffer;
 readonly word:number;
};

/** Small count-copy owner used by a transformed producer. `encode()` only
 * appends COPY_SRC -> MAP_READ commands to the caller's existing frame
 * encoder. The host must call `onFrameSubmit()` after its normal queue submit;
 * this class never submits privately and never maps before that boundary. */
export class GpuIndexedSourceCountCopyQueue{
 private readonly copies=new Map<string,CountCopy>();
 private readonly inFlight=new Set<string>();
 private readonly targets=new Set<GPUBuffer>();
 constructor(private readonly device:GPUDevice){}
 encode(encoder:GPUCommandEncoder,source:GpuIndexedSourceProducerMetadata):void{
  this.encodeMany(encoder,[source]);
 }
 /** One compact mapping per dirty cohort, not one buffer/map per source.
  * Contiguous words of one original producer counter are copied together.
  * Capacity splits bound scratch, never discard any source or count. */
 encodeMany(encoder:GPUCommandEncoder,sources:readonly GpuIndexedSourceProducerMetadata[]):void{
  const fresh=new Map<string,GpuIndexedSourceProducerMetadata>();
  for(const source of sources){const key=`${source.sourceId}@${source.generation}`;if(!this.copies.has(key)&&!this.inFlight.has(key))fresh.set(key,source);}
  const pending=[...fresh.values()];
  const limit=Math.max(1,Math.floor(Math.min(this.device.limits?.maxBufferSize??65536,65536)/4));
  for(let start=0;start<pending.length;start+=limit){
   const cohort=pending.slice(start,start+limit);
   const target=this.device.createBuffer({label:'MAIN indexed source count cohort',size:cohort.length*4,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
   this.targets.add(target);
   try{
    for(let word=0;word<cohort.length;){
     const first=cohort[word]!,offset=first.liveIndexCount.offset??0;let end=word+1;
     while(end<cohort.length&&cohort[end]!.liveIndexCount.buffer===first.liveIndexCount.buffer&&
       (cohort[end]!.liveIndexCount.offset??0)===offset+(end-word)*4)end++;
     encoder.copyBufferToBuffer(first.liveIndexCount.buffer,offset,target,word*4,(end-word)*4);word=end;
    }
    cohort.forEach((source,word)=>{const key=`${source.sourceId}@${source.generation}`;this.copies.set(key,{key,source,target,word});});
   }catch(error){this.release(target);throw error;}
  }
 }
 async afterSubmit():Promise<{readonly source:GpuIndexedSourceProducerMetadata;readonly count:number}[]>{
  const current=[...this.copies.values()];this.copies.clear();
  for(const copy of current)this.inFlight.add(copy.key);
  const targets=new Map<GPUBuffer,CountCopy[]>();
  for(const copy of current){const group=targets.get(copy.target)??[];group.push(copy);targets.set(copy.target,group);}
  const result:{source:GpuIndexedSourceProducerMetadata;count:number}[]=[];
  try{
   for(const [target,cohort]of targets){
    await target.mapAsync(GPUMapMode.READ);
    const counts=new Uint32Array(target.getMappedRange());
    for(const copy of cohort){const count=counts[copy.word]!;validateRegistrationCount(count,copy.source.topology.indexCapacity);result.push({source:copy.source,count});}
    this.release(target);
   }
   return result;
  }finally{for(const target of targets.keys())this.release(target);for(const copy of current)this.inFlight.delete(copy.key);}
 }
 private release(target:GPUBuffer):void{if(!this.targets.delete(target))return;try{target.unmap();}catch{}target.destroy();}
 dispose():void{for(const target of this.targets)this.release(target);this.copies.clear();this.inFlight.clear();}
}

/** Complete producer-side runtime used by the build transform. It owns only
 * source identity/generation and tiny count-copy targets; Mesh, vertex/index,
 * pose, and CSR storage remain owned by their existing systems. */
export class GpuIndexedSourceProducerRuntime{
 readonly topology=new GpuIndexedSourceTopologyTracker();
 readonly counts:GpuIndexedSourceCountCopyQueue;
 private readonly meshes=new Map<string,{readonly indirectIndexCount?:GPUBufferBinding;gpuIndexedSource?:GpuIndexedSourceProducerMetadata}>();
 private readonly waitingCounts=new Map<string,GpuIndexedSourceProducerMetadata>();
 private readonly poseContents=new Map<string,PoseContentState>();
 private frameSubmitError:{readonly error:unknown}|undefined;
 constructor(device:GPUDevice,private readonly onCount?: (registration:GpuIndexedSourceRegistration)=>void){this.counts=new GpuIndexedSourceCountCopyQueue(device);}
 observeMesh<T extends {readonly indirectIndexCount?:GPUBufferBinding}>(mesh:T,input:GpuIndexedSourceProducerMeshInput):GpuIndexedSourceRegistration{
  const registration=this.topology.observe(input);
  const source=this.publishPoseContent(registration.source,input.poseContent);
  const published={...registration,source};
  this.meshes.set(input.sourceId,mesh as T & {gpuIndexedSource?:GpuIndexedSourceProducerMetadata});
  if(published.state==='waiting-for-live-index-count')this.waitingCounts.set(input.sourceId,published.source);
  else this.waitingCounts.delete(input.sourceId);
  attachGpuIndexedSourceMetadata(mesh,source);
  return published;
 }
 /** Update the certified material-pose inputs without pretending that a GPU
  * buffer object's identity is content authority. The actual producer calls
  * this for every source whose instance transform may have changed. */
 observePoseContent(sourceId:string,input:GpuIndexedSourcePoseContentInput|undefined):GpuIndexedSourceProducerMetadata|undefined{
  const source=this.topology.current(sourceId);if(!source)return;
  const published=this.publishPoseContent(source,input);
  const mesh=this.meshes.get(sourceId);if(mesh)attachGpuIndexedSourceMetadata(mesh,published);
  if(this.waitingCounts.has(sourceId))this.waitingCounts.set(sourceId,published);
  return published;
 }
 encodeCountCopies(encoder:GPUCommandEncoder):void{
  this.counts.encodeMany(encoder,[...this.waitingCounts.values()]);
 }
 /** Call after the host's ordinary queue.submit. It is deliberately not safe
  * to call while the frame encoder is still open. */
 async onFrameSubmit():Promise<readonly GpuIndexedSourceRegistration[]>{
  const registrations:GpuIndexedSourceRegistration[]=[];
  for(const {source,count} of await this.counts.afterSubmit()){
   const current=this.topology.current(source.sourceId);
   if(!current||current.generation!==source.generation)continue;
   const registration=this.topology.setRegistrationIndexCount(source.sourceId,source.generation,count);
   this.waitingCounts.delete(source.sourceId);
   const mesh=this.meshes.get(source.sourceId);
   const published={...registration,source:this.publishPoseContent(registration.source,this.poseContents.get(source.sourceId)?.input)};
   if(mesh)attachGpuIndexedSourceMetadata(mesh,published.source);
   registrations.push(published);this.onCount?.(published);
  }
  return registrations;
 }
 /** The engine callback cannot await a readback and must not throw. It stores
  * the failure; the next draw-system call rethrows it synchronously through
  * World.scheduler, the engine's normal error owner. */
 captureFrameSubmitError(error:unknown):void{this.frameSubmitError??={error};}
 rethrowFrameSubmitError():void{
  const failure=this.frameSubmitError;if(!failure)return;
  this.frameSubmitError=undefined;throw failure.error;
 }
 private publishPoseContent(source:GpuIndexedSourceProducerMetadata,input:GpuIndexedSourcePoseContentInput|undefined):GpuIndexedSourceProducerMetadata{
  const previous=this.poseContents.get(source.sourceId);
  if(input===undefined){
   // Keep a same-generation tombstone. Clearing the certificate must not make
   // the next identical pose look like revision 1: a persistent GPU home may
   // still hold the prior revision under this source/generation key.
   this.poseContents.set(source.sourceId,previous?.generation===source.generation
    ? {...previous,input:undefined}
    : {generation:source.generation,input:undefined,revision:0});
   return source;
  }
  validatePoseContentInput(input);
  const cloned=clonePoseContentInput(input);
  const same=previous?.generation===source.generation&&previous.input!==undefined&&samePoseContent({
   ...previous.input,sourceGeneration:previous.generation,revision:previous.revision,
  },cloned,source.generation);
  const revision=same?previous!.revision:previous?.generation===source.generation?previous.revision+1:1;
  const poseContent:GpuIndexedSourcePoseContent={...cloned,sourceGeneration:source.generation,revision};
  this.poseContents.set(source.sourceId,{generation:source.generation,input:cloned,revision});
  return {...source,poseContent};
 }
 dispose():void{this.counts.dispose();this.topology.clear();this.waitingCounts.clear();this.meshes.clear();this.poseContents.clear();this.frameSubmitError=undefined;}
}

/** Generic producer seam used by setup/integration code. The metadata is
 * attached to the GPU-resident Mesh object; no CPU positions or empty complex
 * are introduced, and the real indirect count binding remains authoritative. */
export function attachGpuIndexedSourceMetadata<T extends {readonly indirectIndexCount?:GPUBufferBinding}>(mesh:T,source:GpuIndexedSourceProducerMetadata):T & {readonly gpuIndexedSource:GpuIndexedSourceProducerMetadata}{
 validateInput({sourceId:source.sourceId,topologyKey:source.topologyKey,indexCapacity:source.topology.indexCapacity,vertexCapacity:source.topology.vertexCapacity,indices:source.topology.indices,liveIndexCount:source.liveIndexCount,registrationIndexCount:source.registrationIndexCount});
 if(!mesh.indirectIndexCount||!sameCountWordBinding(mesh.indirectIndexCount,source.liveIndexCount))
  throw Error('GPU indexed source producer metadata must use the Mesh indirectIndexCount binding');
 return Object.assign(mesh,{gpuIndexedSource:source});
}
