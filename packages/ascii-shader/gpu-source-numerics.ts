import {SOURCE_NUMERICS_SHADER} from './source-numeric-program.ts';
import {encodeSourceSemanticResidual,type SourceSemanticResidual} from './source-semantic-residual.ts';
import {GpuRelationSourceCompiler,GPU_NUMERIC_RELATION_SOURCE_VERSION} from './source-incidence-residual.ts';
import {indexedIncidenceLayout,type ResidentIndexedIncidence} from './gpu-source-incidence.ts';
import {TOPOLOGY_HEADER_WORDS,type GpuCompiledSourceRunProgram} from './compiled-construction-topology.ts';
import {prepareComputePipelines} from './gpu-pipeline-preparation.ts';
import {isolateComputeEntry} from './gpu-compute-program.ts';
import {SourceStorage,type SourceStorageRange} from './source-storage.ts';

const ENTRIES=['seedRuntime','applyBoundaryContexts','hashVertices','resolveVertices','linkVertices','sortVertices',
 'hashContactEdges','resolveContactEdges','linkContactEdges','prefixContacts','assembleContacts'] as const;
type Entry=typeof ENTRIES[number];
interface NumericRequest {
 readonly input:NumericSourceInput;readonly semantic:SourceSemanticResidual;readonly domain:number;readonly identity:Uint32Array;
 readonly resolve:(value:NumericSourceReady)=>void;readonly reject:(error:unknown)=>void;
 required:boolean;cancelled:boolean;
}
export interface NumericSourceCompilation {
 readonly result:Promise<NumericSourceReady>;
 /** Unsubmitted work follows the current camera frontier, not arrival order. */
 setRequired(required:boolean):void;
 cancel():void;
}
interface NumericCohort {
 readonly encoder:GPUCommandEncoder;readonly descriptor:GPUBuffer;
 readonly transient:GPUBuffer[];readonly indexed:ResidentIndexedIncidence[];
}
export interface NumericSourceInput {
 readonly positions:Float32Array;readonly indices:Uint32Array;readonly contactKeys?:Uint32Array;
 readonly revision:number;
}
export interface NumericSourceReady {
 readonly range:SourceStorageRange;readonly programRange:SourceStorageRange;
 readonly binding:GPUBufferBinding;readonly runProgram:GpuCompiledSourceRunProgram;
 readonly semantic:SourceSemanticResidual;
}

/** Original source + build-time semantic decisions -> numeric GPU residency.
 * No MeshSourceComplex, sheet analysis or part graph is reconstructed at load.
 * The old worker remains the owner for genuinely uncompiled/dynamic inputs.
 * A source becomes ready only after the complete registration submission.
 * Queue ordering serializes bounded scratch across concurrent mesh arrivals. */
export class GpuSourceNumerics {
 private readonly source:SourceStorage;private readonly programs:SourceStorage;
 private pipelines?:Promise<{layout:GPUBindGroupLayout;pipelines:ReadonlyMap<Entry,GPUComputePipeline>}>;
 private tail:Promise<unknown>=Promise.resolve();private disposed=false;private scratch?:GPUBuffer;
 private readonly queued:NumericRequest[]=[];private scheduled=false;
 constructor(private readonly device:GPUDevice,private readonly inputStorage:SourceStorage,private readonly incidence:GpuRelationSourceCompiler){
  this.source=inputStorage.separate('MAIN numeric source');this.programs=inputStorage.separate('MAIN semantic run program');
 }
 private prepare(){
  return this.pipelines??=(async()=>{
   const layout=this.device.createBindGroupLayout({label:'MAIN numeric source registration',entries:[
    {binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:'uniform'}},
    ...[1,2,3].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:'read-only-storage' as const}})),
    ...[4,5,6].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:'storage' as const}})),
   ]});
   const pipelineLayout=this.device.createPipelineLayout({bindGroupLayouts:[layout]});
   const plans=new Map(ENTRIES.map(entry=>[entry,()=>({label:'MAIN numeric source '+entry,layout:pipelineLayout,
    compute:{entryPoint:entry,module:this.device.createShaderModule({label:'MAIN numeric source '+entry,code:isolateComputeEntry(SOURCE_NUMERICS_SHADER,entry)})}})]));
   return {layout,pipelines:await prepareComputePipelines(this.device,plans)};
  })();
 }
 register(input:NumericSourceInput,semantic:SourceSemanticResidual,domain:number,identity:Uint32Array):NumericSourceCompilation{
  if(this.disposed)throw Error('Numeric source registration is disposed');
  let request!:NumericRequest;
  const result=new Promise<NumericSourceReady>((resolve,reject)=>{request={input,semantic,domain,identity,resolve,reject,required:true,cancelled:false};this.queued.push(request);});
  this.schedule();
  return {result,setRequired:required=>{if(request.cancelled||this.disposed||request.required===required)return;request.required=required;if(required)this.schedule();},cancel:()=>{
   if(request.cancelled)return;request.cancelled=true;request.required=false;
   const index=this.queued.indexOf(request);if(index>=0)this.queued.splice(index,1);
   // An already encoded cohort keeps its resources through its GPU fence.
   // Cancellation retires the revision; it never invents a ready source.
   request.reject(new DOMException('Numeric source revision retired','AbortError'));
  }};
 }
 private schedule():void{
  if(this.scheduled||this.disposed)return;this.scheduled=true;
  queueMicrotask(()=>{
   this.tail=this.tail.then(async()=>{
    while(!this.disposed){
     const batch:NumericRequest[]=[],held:NumericRequest[]=[];let bytes=0;
     for(const request of this.queued){
      if(request.cancelled)continue;
      if(!request.required){held.push(request);continue;}
      const {input,semantic:s}=request,size=input.positions.byteLength+(input.contactKeys?.byteLength??0)+
       indexedIncidenceLayout(s.vertices,s.corners,s.indexedEdges).words*4+input.indices.byteLength+s.edgeClasses.byteLength+s.vertexContexts.byteLength+s.boundaryContexts.byteLength;
      // Keep the existing transfer window, not a geometry quota. Revisit
      // live demand after EACH completed cohort instead of draining the old
      // world's whole queue ahead of a moved camera. Dormant jobs stay held.
      if(batch.length&&bytes+size>8*1024*1024){held.push(request);continue;}
      batch.push(request);bytes+=size;
     }
     this.queued.splice(0,this.queued.length,...held);
     if(!batch.length)break;
     await this.cohort(batch);
    }
   }).catch(error=>{for(const request of this.queued.splice(0))request.reject(error);}).finally(()=>{
    this.scheduled=false;
    if(this.queued.some(request=>request.required&&!request.cancelled))this.schedule();
   });
  });
 }
 private async cohort(requests:readonly NumericRequest[]):Promise<void>{
  const batch:NumericCohort={encoder:this.device.createCommandEncoder({label:'MAIN numeric source registration cohort'}),
   descriptor:this.device.createBuffer({label:'MAIN numeric registration completion',size:requests.length*128,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),transient:[],indexed:[]};
  let results:PromiseSettledResult<NumericSourceReady>[]=[];let submitted=false;
  try{
   // Starting together also feeds ONE cohort to the existing CSR owner.
   // Separate source uniforms remain immutable while scratch is reused in
   // encoded order. No fence/pass storm is hidden behind smaller buffers.
   results=await Promise.allSettled(requests.map((r,i)=>this.run(r.input,r.semantic,r.domain,r.identity,batch,i*128)));
   const failed=results.find(r=>r.status==='rejected');if(failed?.status==='rejected')throw failed.reason;
   this.device.queue.submit([batch.encoder.finish()]);submitted=true;await batch.descriptor.mapAsync(GPUMapMode.READ);
   const words=new Uint32Array(batch.descriptor.getMappedRange()).slice();batch.descriptor.unmap();
   if(this.disposed)throw Error('Numeric source retired before publication');
   for(let i=0;i<requests.length;i++)if(words[i*32+15]!==0)throw Error('Numeric source registration failed (flags '+words[i*32+15]+')');
   results.forEach((result,i)=>{if(result.status==='fulfilled'){
    if(requests[i]!.cancelled)this.retire(result.value);else requests[i]!.resolve(result.value);
   }});
  }catch(error){
   if(submitted)await this.device.queue.onSubmittedWorkDone().catch(()=>{});
   for(const result of results)if(result.status==='fulfilled')this.retire(result.value);
   for(const request of requests)request.reject(error);
  }finally{
   for(const buffer of batch.transient)buffer.destroy();batch.descriptor.destroy();
   for(const indexed of batch.indexed)this.inputStorage.retire(indexed.range);
  }
 }
 private async run(input:NumericSourceInput,s:SourceSemanticResidual,domain:number,identity:Uint32Array,batch:NumericCohort,descriptorOffset:number):Promise<NumericSourceReady>{
  if(this.disposed)throw Error('Numeric source registration is disposed');
  const {positions,indices,contactKeys}=input,V=positions.length/3,H=indices.length,E=s.indexedEdges,C=s.contacts,M=s.contactMembers;
  if(s.publishedAdjacency||s.vertices!==V||s.corners!==H||!V||!H||H%3||contactKeys&&contactKeys.length!==V*2)throw Error('Numeric source does not match its semantic revision');
  const ready=await this.prepare();if(this.disposed)throw Error('Numeric source retired during preparation');
  const csr=indexedIncidenceLayout(V,H,E),align=(n:number)=>Math.ceil(n/64)*64;
  let at=32;const take=(n:number)=>{const from=at;at+=n;return from;};
  at=align(at);const indexOffset=take(H);at=align(at);const csrOffset=take(csr.words);
  const vertexCanon=take(V),vertexNext=take(V),vertexCounts=take(V);
  const contactRecords=take(C*2),edgeContact=take(E),contactMembers=take(M*2),faceMaterials=s.faceMaterials.length?take(s.faceMaterials.length):0;

  const edgeLevels=s.edgeLevels?.length?take(s.edgeLevels.length):0;

  const limbWords=s.limbVertices?.length??0,partWords=s.partPrograms?.length??0;
  const words=at,runWords=TOPOLOGY_HEADER_WORDS+H+s.vertexContexts.length+limbWords+partWords;
  let hashSlots=1;while(hashSlots<2*Math.max(V,H,E))hashSlots*=2;
  const N=Math.max(V,E),heads=hashSlots,next=heads+N,counts=next+N,ordinals=counts+N;
  const scratchBytes=(ordinals+N)*4;
  if(scratchBytes>Math.min(this.device.limits.maxBufferSize,this.device.limits.maxStorageBufferBindingSize))throw Error('Numeric source requires source-sized registration chunks');
  if(!this.scratch||this.scratch.size<scratchBytes){this.scratch?.destroy();this.scratch=this.device.createBuffer({label:'MAIN numeric registration scratch',size:scratchBytes,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});}
  const range=this.source.allocate(words*4);let programRange:SourceStorageRange;
  try{programRange=this.programs.allocate(runWords*4);}catch(error){this.source.retire(range);throw error;}
  const binding=this.source.binding(range),programBinding=this.programs.binding(programRange);
  const transient:GPUBuffer[]=[];let indexed:ResidentIndexedIncidence|undefined,handedOff=false;
  const make=(label:string,data:ArrayBufferView,usage:number)=>{const b=this.device.createBuffer({label,size:Math.max(4,data.byteLength),usage:usage|GPUBufferUsage.COPY_DST});transient.push(b);if(data.byteLength)this.device.queue.writeBuffer(b,0,data.buffer,data.byteOffset,data.byteLength);return b;};
  try{
   const header=new Uint32Array(32);header.set([0x4d41494e,GPU_NUMERIC_RELATION_SOURCE_VERSION,V,H/3,indexOffset,csrOffset,E,0,
    vertexNext,contactRecords,C,edgeContact,faceMaterials,input.revision,vertexCanon,0,vertexCounts,contactMembers,M,runWords,edgeLevels]);header.set(identity,24);
   this.source.write(range,header);this.source.write(range,indices,indexOffset*4);
   const sourceIndices={buffer:binding.buffer,offset:(binding.offset??0)+indexOffset*4,size:H*4};
   indexed=await this.incidence.buildRaw(sourceIndices,V,H,E);
   if(indexed.edges!==E)throw Error('Numeric source and semantic indexed edges disagree');
   if(this.disposed)throw Error('Numeric source retired during incidence registration');
   const keys=contactKeys??new Uint32Array(positions.buffer,positions.byteOffset,positions.length);
   const keyBuffer=make('MAIN original contact/position keys',keys,GPUBufferUsage.STORAGE);
   const semanticBuffer=make('MAIN build-time semantic decisions',encodeSourceSemanticResidual(s,identity),GPUBufferUsage.STORAGE);
   const groupsX=this.device.limits.maxComputeWorkgroupsPerDimension;
   const params=make('MAIN numeric source registration parameters',new Uint32Array([
    V,H,E,C,M,contactKeys?2:3,hashSlots-1,domain,indexOffset,csrOffset,0,vertexCanon,
    vertexNext,vertexCounts,contactRecords,edgeContact,contactMembers,faceMaterials,input.revision,runWords,
    heads,next,counts,ordinals,0,0,0,groupsX,
   ]),GPUBufferUsage.UNIFORM);
   const group=this.device.createBindGroup({layout:ready.layout,entries:[{binding:0,resource:{buffer:params}},{binding:1,resource:{buffer:keyBuffer}},
    {binding:2,resource:{...indexed.binding,size:csr.words*4}},{binding:3,resource:{buffer:semanticBuffer}},
    {binding:4,resource:{buffer:this.scratch}},{binding:5,resource:binding},{binding:6,resource:programBinding}]});
   const encoder=batch.encoder,base=binding.offset??0;
   // The original indices were already read by the indexed producer. Do not
   // clear them along with the derived fields before this submission.
   encoder.clearBuffer(binding.buffer,base+128,(indexOffset-32)*4);
   encoder.clearBuffer(binding.buffer,base+(indexOffset+H)*4,(words-indexOffset-H)*4);
   if(edgeLevels)encoder.copyBufferToBuffer(make('MAIN compiled construction levels',s.edgeLevels!,GPUBufferUsage.COPY_SRC),0,binding.buffer,base+edgeLevels*4,s.edgeLevels!.byteLength);
   encoder.clearBuffer(programBinding.buffer,programBinding.offset??0,runWords*4);encoder.clearBuffer(this.scratch);
   let pass=encoder.beginComputePass({label:'MAIN numeric source exact contacts'});
   const dispatch=(entry:Entry,n:number)=>{if(!n)return;pass.setPipeline(ready.pipelines.get(entry)!);pass.setBindGroup(0,group);
    const groups=entry==='prefixContacts'?1:Math.ceil(n/64),x=Math.min(groupsX,groups);pass.dispatchWorkgroups(x,Math.ceil(groups/x));};
   dispatch('seedRuntime',Math.max(csr.words,H,s.vertexContexts.length,s.faceMaterials.length,limbWords,partWords));dispatch('applyBoundaryContexts',s.boundaryContexts.length/2);
   dispatch('hashVertices',V);dispatch('resolveVertices',V);dispatch('linkVertices',V);dispatch('sortVertices',V);pass.end();
   encoder.clearBuffer(this.scratch);pass=encoder.beginComputePass({label:'MAIN numeric source boundary contacts'});
   dispatch('hashContactEdges',E);dispatch('resolveContactEdges',E);dispatch('linkContactEdges',E);dispatch('prefixContacts',1);dispatch('assembleContacts',E);pass.end();
   encoder.copyBufferToBuffer(binding.buffer,base,batch.descriptor,descriptorOffset,128);
   batch.transient.push(...transient);batch.indexed.push(indexed);handedOff=true;
   return {range,programRange,binding,semantic:s,runProgram:{gpuTopology:{binding:programBinding,words:runWords},criticalVertices:s.criticalVertices}};
  }catch(error){this.source.retire(range);this.programs.retire(programRange);throw error;}
  finally{if(!handedOff){for(const b of transient)b.destroy();if(indexed)this.inputStorage.retire(indexed.range);}}
 }
 retire(source:NumericSourceReady):void{this.source.retire(source.range);this.programs.retire(source.programRange);}
 submitted():void{this.source.submitted();this.programs.submitted();}
 async dispose():Promise<void>{this.disposed=true;for(const request of this.queued.splice(0))request.reject(Error('Numeric source disposed'));await this.tail;this.scratch?.destroy();this.source.dispose();this.programs.dispose();}
}
