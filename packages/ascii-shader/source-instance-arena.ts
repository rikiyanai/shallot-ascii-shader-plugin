import {prepareComputePipelines} from './gpu-pipeline-preparation.ts';
/** Source residency owns the shared bounds/visibility subset of the arena
 * interface. Geometry/posed-output buffers keep their existing owners until
 * their range-aware consumers migrate. No buffer is a batch identity. */
/** Shared source/posed arena contract. Cull1 connects the instance subset;
 * SourceResidencyFrame is exposed only once all its ranges are populated.
 * There is no partial snapshot whose missing ranges stand in for geometry. */
export const SOURCE_RESIDENCY_API_VERSION=1;
export const SOURCE_RECORD_WORDS=32,BATCH_RECORD_WORDS=16,INSTANCE_RECORD_WORDS=32;
export interface SourceResidencyFrame {
 readonly version:typeof SOURCE_RESIDENCY_API_VERSION;
 /** Resource replacement, NOT source content or pose revision. */
 readonly epoch:number;
 readonly catalogWords:GPUBuffer;
 readonly sourcePositions:GPUBuffer;readonly sourceNormals:GPUBuffer;
 /** Original indices and tagged incidence; all offsets in u32 words. */
 readonly sourceWords:GPUBuffer;
 readonly posedVertices:GPUBuffer;readonly posedAttributes:GPUBuffer;
 readonly posedBounds:GPUBuffer;readonly visibleInstances:GPUBuffer;
 readonly sourceTableBase:number;readonly batchTableBase:number;readonly instanceTableBase:number;
 readonly instanceCount:number;
}
/** Stable cache identity + explicit arena range. Never key a batch solely by
 * its GPUBuffer after it shares a resource with other batches. */
export interface SourceArenaBatchReference {
 readonly slot:number;readonly generation:number;
 readonly sourceSlot:number;readonly sourceGeneration:number;readonly posedRevision:number;
 readonly vertexBase:number;readonly vertexCapacity:number;readonly attributeBase:number;
 readonly boundsBase:number;readonly visibilityBase:number;readonly instanceCount:number;
}
export interface SourceInstanceRange {readonly base:number;readonly count:number;readonly generation:number}
/** Original batch-local instance addresses, never renumbered geometry. The
 * count immediately precedes offset in the same visibility binding. */
export interface SourceInstanceFrontier {readonly offset:number;readonly indices:Uint32Array}
interface CullSlot {readonly params:GPUBuffer;readonly image:ArrayBuffer;busy:boolean;group?:GPUBindGroup;catalog?:GPUBuffer;bounds?:GPUBuffer;visibility?:GPUBuffer}
const CULL=/* wgsl */`
struct P {clip:mat4x4f,count:vec4u}
@group(0) @binding(0) var<uniform> p:P;
@group(0) @binding(1) var<storage,read> bounds:array<vec4f>;
@group(0) @binding(2) var<storage,read_write> visible:array<u32>;
@group(0) @binding(3) var<storage,read> instances:array<vec4u>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3u){
 let ordinal=id.x+id.y*p.count.y*64u;if(ordinal>=p.count.x){return;}
 let instance=instances[ordinal];
 // A previous frontier member left the view. Clear its OLD direct flag in
 // queue order; never leave stale visibility when the octree skips its work.
 if(instance.w==0xffffffffu){visible[instance.y]=0u;return;}
 let low=bounds[instance.x*2u].xyz;let high=bounds[instance.x*2u+1u].xyz;var outside=127u;
 for(var corner=0u;corner<8u;corner++){
  let q=p.clip*vec4f(select(low,high,vec3u(corner&1u,corner&2u,corner&4u)!=vec3u(0)),1);
  let planes=array<f32,7>(q.x+q.w,q.w-q.x,q.y+q.w,q.w-q.y,q.z,q.w-q.z,q.w-.00001);var mask=0u;
  for(var plane=0u;plane<7u;plane++){if(planes[plane]<0.0){mask|=1u<<plane;}}outside&=mask;
 }
 visible[instance.y]=select(1u,0u,outside!=0u);
}`;

export class SourceInstanceArena {
 private readonly pipeline:GPUComputePipeline;
 private capacity=0;private end=0;private serial=0;
 private boundsBuffer?:GPUBuffer;private visibilityBuffer?:GPUBuffer;
 private catalog?:GPUBuffer;private catalogImage=new Uint32Array(0);private catalogCount=0;
 private active=new Uint32Array(0);private activeCount=0;private catalogDirty=true;
 private readonly activeSlots=new Set<number>();
 private readonly frontierMembers=new Map<SourceInstanceRange,number[]>();
 private readonly frontiers=new Map<SourceInstanceRange,SourceInstanceFrontier>();
 private frontierCursor=0;private frontierBase=0;
 private readonly slots:CullSlot[]=[];private readonly retired:GPUBuffer[]=[];
 private readonly free:SourceInstanceRange[]=[];private readonly retiring:SourceInstanceRange[]=[];
 epoch=0;
 static async create(device:GPUDevice,progress?:(fraction:number)=>void):Promise<SourceInstanceArena>{
  const descriptor=()=>({layout:'auto' as const,compute:{module:device.createShaderModule({label:'MAIN global instance frustum',code:CULL}),entryPoint:'main'}});
  const pipelines=await prepareComputePipelines(device,new Map([['global-instance-frustum',descriptor]]),progress);
  return new SourceInstanceArena(device,pipelines.get('global-instance-frustum')!);
 }
 constructor(private readonly device:GPUDevice,prepared?:GPUComputePipeline){this.pipeline=prepared??device.createComputePipeline({layout:'auto',compute:{module:device.createShaderModule({label:'MAIN global instance frustum',code:CULL}),entryPoint:'main'}});}
 get bounds():GPUBuffer{if(!this.boundsBuffer)throw Error('Source instance arena was not reserved');return this.boundsBuffer;}
 get visibility():GPUBuffer{if(!this.visibilityBuffer)throw Error('Source instance arena was not reserved');return this.visibilityBuffer;}
 /** Must precede every pose dispatch of this encode. Preserve unchanged
  * bounds on growth; old resources stay valid through submitted(). */
 reserve(encoder:GPUCommandEncoder,additional:number|readonly number[]):void{
  let end=this.end;
  if(typeof additional==='number')end+=additional;
  else {const free=this.free.map(range=>range.count);for(const count of additional){const index=free.findIndex(size=>size>=count);if(index<0)end+=count;else {const size=free.splice(index,1)[0]!;if(size>count)free.push(size-count);}}}
  const required=Math.max(1,end);if(required<=this.capacity){this.reserveFrontier();return;}
  const capacity=2**Math.ceil(Math.log2(required));
  if(capacity*32>Math.min(this.device.limits.maxBufferSize,this.device.limits.maxStorageBufferBindingSize))throw Error('Shared source instance bounds exceed device binding capacity');
  const usage=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST;
  const bounds=this.device.createBuffer({label:'MAIN source residency instance bounds',size:capacity*32,usage});
  const visibility=this.device.createBuffer({label:'MAIN source residency instance visibility and frontier',size:capacity*12,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
  if(this.boundsBuffer){encoder.copyBufferToBuffer(this.boundsBuffer,0,bounds,0,this.end*32);this.retired.push(this.boundsBuffer,this.visibilityBuffer!);}
  this.boundsBuffer=bounds;this.visibilityBuffer=visibility;this.capacity=capacity;this.frontierCursor=0;this.epoch++;this.catalogDirty=true;
 }
 private reserveFrontier():void{
  // A second prepared view before submitted() gets a different immutable
  // payload range. Queue writes must not replace an earlier encoder's list.
  const words=this.capacity+this.frontierCursor+this.capacity*2;
  if(this.visibilityBuffer&&this.visibilityBuffer.size>=words*4)return;
  const bytes=2**Math.ceil(Math.log2(Math.max(16,words*4)));
  if(bytes>Math.min(this.device.limits.maxStorageBufferBindingSize,this.device.limits.maxBufferSize))throw Error('Complete instance frontier exceeds device binding capacity');
  const next=this.device.createBuffer({label:'MAIN source residency instance visibility and frontier',size:bytes,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
  if(this.visibilityBuffer)this.retired.push(this.visibilityBuffer);
  this.visibilityBuffer=next;this.frontierCursor=0;this.catalogDirty=true;this.epoch++;
 }
 allocate(count:number):SourceInstanceRange{
  const index=this.free.findIndex(range=>range.count>=count);let base:number;
  if(index>=0){const range=this.free.splice(index,1)[0]!;base=range.base;if(range.count>count)this.free.push({base:base+count,count:range.count-count,generation:0});}
  else {base=this.end;this.end+=count;}
  if(this.end>this.capacity)throw Error('Source instance ranges were not reserved before posing');
  this.catalogDirty=true;return {base,count,generation:++this.serial};
 }
 retire(range:SourceInstanceRange):void{this.retiring.push(range);}
 begin():void{this.activeCount=0;this.activeSlots.clear();this.frontierMembers.clear();this.frontiers.clear();this.frontierBase=this.capacity+this.frontierCursor;this.frontierCursor+=this.capacity*2;}
 private append(bounds:number,visible:number,generation:number,instance:number):void{
  const required=(this.activeCount+1)*4;
  if(this.active.length<required){const next=new Uint32Array(2**Math.ceil(Math.log2(Math.max(4,required))));next.set(this.active);this.active=next;}
  const at=this.activeCount++*4;this.active[at]=bounds;this.active[at+1]=visible;this.active[at+2]=generation;this.active[at+3]=instance;
 }
 includeInstance(range:SourceInstanceRange,instance:number):void{
  if(!Number.isSafeInteger(instance)||instance<0||instance>=range.count)throw Error('Spatial frontier instance is outside its source range');
  const slot=range.base+instance;if(this.activeSlots.has(slot))return;
  this.activeSlots.add(slot);this.append(slot,slot,range.generation,instance);
  let members=this.frontierMembers.get(range);if(!members){members=[];this.frontierMembers.set(range,members);}members.push(instance);
 }
 include(range:SourceInstanceRange):void{
  for(let instance=0;instance<range.count;instance++)this.includeInstance(range,instance);
 }
 frontier(range:SourceInstanceRange):SourceInstanceFrontier|undefined{return this.frontiers.get(range);}
 private uploadFrontiers():void{
  const data=new Uint32Array(this.activeSlots.size+this.frontierMembers.size);let at=0;
  for(const [range,members]of this.frontierMembers){
   members.sort((a,b)=>a-b);data[at++]=members.length;
   const indices=Uint32Array.from(members);this.frontiers.set(range,{offset:this.frontierBase+at,indices});data.set(indices,at);at+=indices.length;
  }
  if(data.length)this.device.queue.writeBuffer(this.visibility,this.frontierBase*4,data);
 }
 /** Direct range flags, NOT an atomic append stream. Active ordinal order
  * remains host batch/instance order, and consumers keep their own order. */
 encode(pass:()=>GPUComputePassEncoder,clip:Float32Array,changed:boolean):boolean{
  this.uploadFrontiers();
  // Cull only the current candidate frontier plus one-frame clears for
  // departed candidates. Each address is written once in this dispatch.
  for(let at=0;at<this.catalogImage.length;at+=4)if(this.catalogImage[at+3]!==0xffffffff&&!this.activeSlots.has(this.catalogImage[at+1]!))this.append(this.catalogImage[at]!,this.catalogImage[at+1]!,this.catalogImage[at+2]!,0xffffffff);
  if(!this.activeCount)return false;
  let different=this.catalogCount!==this.activeCount;
  for(let i=0;i<this.activeCount*4&&!different;i++)different=this.catalogImage[i]!==this.active[i];
  if(different||!this.catalog){
   const image=this.active.slice(0,this.activeCount*4),buffer=this.device.createBuffer({label:'MAIN source residency active instance table',size:Math.max(16,image.byteLength),usage:GPUBufferUsage.STORAGE,mappedAtCreation:true});new Uint32Array(buffer.getMappedRange()).set(image);buffer.unmap();
   if(this.catalog)this.retired.push(this.catalog);this.catalog=buffer;this.catalogImage=image;this.catalogCount=this.activeCount;this.catalogDirty=true;
  }
  if(!this.activeCount||(!changed&&!this.catalogDirty))return false;
  let slot=this.slots.find(value=>!value.busy);
  if(!slot){slot={params:this.device.createBuffer({label:'MAIN leased global frustum input',size:80,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),image:new ArrayBuffer(80),busy:false};this.slots.push(slot);}
  slot.busy=true;new Float32Array(slot.image).set(clip);const groups=Math.ceil(this.activeCount/64),columns=Math.min(groups,this.device.limits.maxComputeWorkgroupsPerDimension);new Uint32Array(slot.image).set([this.activeCount,columns,0,0],16);this.device.queue.writeBuffer(slot.params,0,slot.image);
  if(!slot.group||slot.catalog!==this.catalog||slot.bounds!==this.bounds||slot.visibility!==this.visibility){slot.group=this.device.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries:[slot.params,this.bounds,this.visibility,this.catalog].map((buffer,binding)=>({binding,resource:{buffer}}))});slot.catalog=this.catalog;slot.bounds=this.bounds;slot.visibility=this.visibility;}
  const compute=pass();compute.setPipeline(this.pipeline);compute.setBindGroup(0,slot.group);compute.dispatchWorkgroups(columns,Math.ceil(groups/columns));this.catalogDirty=false;return true;
 }
 submitted():void{
  this.frontierCursor=0;
  for(const slot of this.slots)slot.busy=false;this.free.push(...this.retiring.splice(0));this.free.sort((a,b)=>a.base-b.base);for(let i=1;i<this.free.length;){const a=this.free[i-1]!,b=this.free[i]!;if(a.base+a.count===b.base){this.free.splice(i-1,2,{base:a.base,count:a.count+b.count,generation:0});}else i++;}while(this.free.at(-1)&&this.free.at(-1)!.base+this.free.at(-1)!.count===this.end)this.end=this.free.pop()!.base;for(const buffer of this.retired.splice(0))buffer.destroy();
  if(!this.end){for(const slot of this.slots)slot.params.destroy();this.slots.length=0;this.catalog?.destroy();this.boundsBuffer?.destroy();this.visibilityBuffer?.destroy();this.catalog=this.boundsBuffer=this.visibilityBuffer=undefined;this.capacity=this.end=this.catalogCount=0;this.catalogImage=new Uint32Array(0);this.activeSlots.clear();this.free.length=0;this.catalogDirty=true;this.epoch++;}
 }
 dispose():void{this.submitted();for(const slot of this.slots)slot.params.destroy();this.catalog?.destroy();this.boundsBuffer?.destroy();this.visibilityBuffer?.destroy();this.slots.length=0;}
}
