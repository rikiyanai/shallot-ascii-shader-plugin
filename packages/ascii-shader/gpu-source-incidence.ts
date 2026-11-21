import {SourceStorage,type SourceStorageRange} from './source-storage.ts';
import {isolateComputeEntry} from './gpu-compute-program.ts';
import {compileComputePipelines,prepareComputePipelines,type ComputePipelinePlan,type ComputePipelineProgress} from './gpu-pipeline-preparation.ts';

/** Indexed incidence only. Rest-position welding and published P0 contacts
 * are different relations, not permission to merge these vertex addresses.
 * Every member list is ordered by ORIGINAL corner, including nonmanifolds.
 * This format is not the v2 relation packet and must never be bound as one. */
export const INDEXED_INCIDENCE_MAGIC=0x43535249;
export const INDEXED_INCIDENCE_VERSION=1;
/** Certified independent-quad incidence is a fixed address program. */
export const INDEXED_QUAD_INCIDENCE_VERSION=2;
export interface IndexedIncidenceLayout {
 readonly vertices:number;readonly corners:number;readonly edgeCapacity:number;
 readonly cornerEdge:number;readonly edgeOffsets:number;readonly edgeCorners:number;
 readonly fanOffsets:number;readonly fanCorners:number;readonly words:number;
}
export function indexedIncidenceLayout(vertices:number,corners:number,edgeCapacity=corners):IndexedIncidenceLayout {
 if(!Number.isSafeInteger(vertices)||vertices<1||!Number.isSafeInteger(corners)||corners<3||corners%3||
    !Number.isSafeInteger(edgeCapacity)||edgeCapacity<1||edgeCapacity>corners)throw Error('Invalid indexed incidence extent');
 const cornerEdge=16,edgeOffsets=cornerEdge+corners,edgeCorners=edgeOffsets+edgeCapacity+1;
 const fanOffsets=edgeCorners+corners,fanCorners=fanOffsets+vertices+1,words=fanCorners+corners;
 if(words>0xffffffff)throw Error('Indexed incidence exceeds word addressing');
 return {vertices,corners,edgeCapacity,cornerEdge,edgeOffsets,edgeCorners,fanOffsets,fanCorners,words};
}
export interface ResidentIndexedIncidence {
 readonly layout:IndexedIncidenceLayout;readonly edges:number;readonly range:SourceStorageRange;readonly binding:GPUBufferBinding;
}
export type IndexedIncidenceTarget=Pick<ResidentIndexedIncidence,'range'|'binding'>;
interface IndexedBuildRequest {indices:GPUBufferBinding;layout:IndexedIncidenceLayout;target?:IndexedIncidenceTarget;retainIndices:boolean;indexPattern?:'independent-quads';resolve:(value:ResidentIndexedIncidence)=>void;reject:(error:unknown)=>void}
interface IndexedBuildBatch {encoder:GPUCommandEncoder;summary:GPUBuffer;offset:number;uniforms:GPUBuffer[]}

/** Hashing finishes before CSR assembly. Its slots borrow the final output;
 * only a possible tail and the error word need external hash storage. */
export function indexedIncidenceScratch(layout:IndexedIncidenceLayout){
 const groups=Math.ceil(layout.corners/128);let slots=1;while(slots<layout.corners*2)slots*=2;
 const resident=Math.min(slots,layout.words),spill=slots-resident;
 const sizes=[layout.corners*8,layout.corners*8,(spill+1)*4,groups*16*4,groups*16*4,9*256];
 return {groups,slots,resident,spill,sizes,bytes:sizes.reduce((a,b)=>a+b,64)};
}

/** Registration-time compute, never a displayed-frame pass. Hashing only
 * identifies exact index pairs. A stable radix sort preserves corner order;
 * atomics never choose publication or reduction order. */
export const INDEXED_INCIDENCE_SHADER=/* wgsl */`
struct Params {dims:vec4u,offsets:vec4u,tail:vec4u,origin:vec4u}
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read> indices:array<u32>;
@group(0) @binding(2) var<storage,read_write> slots:array<atomic<u32>>;
@group(0) @binding(3) var<storage,read> input:array<vec2u>;
@group(0) @binding(4) var<storage,read_write> output:array<vec2u>;
@group(0) @binding(5) var<storage,read_write> histogram:array<u32>;
@group(0) @binding(6) var<storage,read_write> prefix:array<u32>;
@group(0) @binding(7) var<storage,read_write> csr:array<atomic<u32>>;
var<workgroup> counts:array<atomic<u32>,16>;
var<workgroup> digits:array<u32,128>;
var<workgroup> starts:array<u32,128>;
fn sourceIndex(h:u32)->u32{return indices[h];}
@compute @workgroup_size(128) fn retainIndices(@builtin(global_invocation_id) id:vec3u){
 let h=id.x;if(h<p.dims.y){storeCSR(p.tail.w+h,sourceIndex(h));}if(h==0u){storeCSR(12u,p.tail.w);}
}
fn loadCSR(at:u32)->u32{return atomicLoad(&csr[p.origin.x+at]);}
fn storeCSR(at:u32,value:u32){atomicStore(&csr[p.origin.x+at],value);}
fn loadSlot(at:u32)->u32{if(at<p.origin.z){return atomicLoad(&csr[p.origin.x+at]);}return atomicLoad(&slots[at-p.origin.z]);}
fn claimSlot(at:u32,value:u32)->vec2u{
 if(at<p.origin.z){let c=atomicCompareExchangeWeak(&csr[p.origin.x+at],0u,value);return vec2u(c.old_value,select(0u,1u,c.exchanged));}
 let c=atomicCompareExchangeWeak(&slots[at-p.origin.z],0u,value);return vec2u(c.old_value,select(0u,1u,c.exchanged));
}
fn minSlot(at:u32,value:u32){if(at<p.origin.z){atomicMin(&csr[p.origin.x+at],value);}else{atomicMin(&slots[at-p.origin.z],value);}}
fn next(h:u32)->u32{return h/3u*3u+(h+1u)%3u;}
fn edgeKey(h:u32)->vec2u{let a=sourceIndex(h);let b=sourceIndex(next(h));return vec2u(min(a,b),max(a,b));}
fn hash(key:vec2u)->u32{return ((key.x*0x9e3779b9u)^(key.y*0x85ebca6bu))&p.dims.w;}
@compute @workgroup_size(1) fn initialize(){
 storeCSR(0u,${INDEXED_INCIDENCE_MAGIC}u);storeCSR(1u,${INDEXED_INCIDENCE_VERSION}u);
 storeCSR(2u,p.dims.x);storeCSR(3u,p.dims.y);storeCSR(5u,p.dims.z);
 storeCSR(6u,p.offsets.x);storeCSR(7u,p.offsets.y);storeCSR(8u,p.offsets.z);
 storeCSR(9u,p.offsets.w);storeCSR(10u,p.tail.x);
}
@compute @workgroup_size(128) fn hashEdges(@builtin(global_invocation_id) id:vec3u){
 let h=id.x;if(h>=p.dims.y){return;}let key=edgeKey(h);var at=hash(key);
 if(key.y>=p.dims.x||key.x==key.y){atomicOr(&slots[p.origin.w],2u);return;}
 // There are at most H occupied slots in a table of at least 2H slots.
 for(var probe=0u;probe<=p.dims.w;probe++){
  let claim=claimSlot(at,h+1u);
  if(claim.y!=0u){return;}
  if(claim.x==0u){continue;}
  if(all(edgeKey(claim.x-1u)==key)){minSlot(at,h+1u);return;}
  at=(at+1u)&p.dims.w;
 }
}
@compute @workgroup_size(128) fn seedEdges(@builtin(global_invocation_id) id:vec3u){
 let h=id.x;if(h>=p.dims.y){return;}let key=edgeKey(h);var at=hash(key);
 if(key.y>=p.dims.x||key.x==key.y){output[h]=vec2u(0u,h);return;}
 for(var probe=0u;probe<=p.dims.w;probe++){
  let first=loadSlot(at);
  if(first!=0u&&all(edgeKey(first-1u)==key)){output[h]=vec2u(first-1u,h);return;}
  at=(at+1u)&p.dims.w;
 }
}
@compute @workgroup_size(128) fn seedFans(@builtin(global_invocation_id) id:vec3u){
 if(id.x<p.dims.y){output[id.x]=vec2u(sourceIndex(id.x),id.x);}
}
@compute @workgroup_size(128) fn radixHistogram(@builtin(global_invocation_id) id:vec3u,@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) group:vec3u){
 if(lane<16u){atomicStore(&counts[lane],0u);}workgroupBarrier();
 if(id.x<p.dims.y){atomicAdd(&counts[(input[id.x].x>>p.tail.z)&15u],1u);}workgroupBarrier();
 if(lane<16u){histogram[lane*p.tail.y+group.x]=atomicLoad(&counts[lane]);}
}
@compute @workgroup_size(16) fn radixPrefix(@builtin(local_invocation_index) digit:u32){
 var sum=0u;
 for(var d=0u;d<digit;d++){for(var g=0u;g<p.tail.y;g++){sum+=histogram[d*p.tail.y+g];}}
 for(var g=0u;g<p.tail.y;g++){let at=digit*p.tail.y+g;prefix[at]=sum;sum+=histogram[at];}
}
@compute @workgroup_size(128) fn radixScatter(@builtin(global_invocation_id) id:vec3u,@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) group:vec3u){
 let valid=id.x<p.dims.y;var digit=16u;if(valid){digit=(input[id.x].x>>p.tail.z)&15u;}
 digits[lane]=digit;workgroupBarrier();
 if(valid){var rank=0u;for(var i=0u;i<lane;i++){if(digits[i]==digit){rank++;}}
  output[prefix[digit*p.tail.y+group.x]+rank]=input[id.x];}
}
fn groupStart(i:u32)->u32{if(i>=p.dims.y){return 0u;}if(i==0u){return 1u;}return select(0u,1u,input[i-1u].x!=input[i].x);}
@compute @workgroup_size(128) fn edgeGroups(@builtin(global_invocation_id) id:vec3u,@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) group:vec3u){
 starts[lane]=groupStart(id.x);workgroupBarrier();
 if(lane==0u){var sum=0u;for(var i=0u;i<128u;i++){sum+=starts[i];}histogram[group.x]=sum;}
}
@compute @workgroup_size(1) fn edgePrefix(){
 var sum=0u;for(var g=0u;g<p.tail.y;g++){prefix[g]=sum;sum+=histogram[g];}
 storeCSR(4u,sum);storeCSR(11u,atomicLoad(&slots[p.origin.w])|select(0u,1u,sum>p.dims.z));
}
@compute @workgroup_size(128) fn assembleEdges(@builtin(global_invocation_id) id:vec3u,@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) group:vec3u){
 starts[lane]=groupStart(id.x);workgroupBarrier();
 if(id.x>=p.dims.y||loadCSR(11u)!=0u){return;}
 var rank=0u;for(var i=0u;i<=lane;i++){rank+=starts[i];}let edge=prefix[group.x]+rank-1u;
 storeCSR(p.offsets.x+input[id.x].y,edge);storeCSR(p.offsets.z+id.x,input[id.x].y);
 if(starts[lane]!=0u){storeCSR(p.offsets.y+edge,id.x);}
 if(id.x+1u==p.dims.y){storeCSR(p.offsets.y+loadCSR(4u),p.dims.y);}
}
@compute @workgroup_size(128) fn assembleFanCorners(@builtin(global_invocation_id) id:vec3u){
 if(id.x<p.dims.y){storeCSR(p.tail.x+id.x,input[id.x].y);}
}
@compute @workgroup_size(128) fn assembleFanOffsets(@builtin(global_invocation_id) id:vec3u){
 let v=id.x;if(v>p.dims.x){return;}var lo=0u;var hi=p.dims.y;
 while(lo<hi){let mid=(lo+hi)/2u;if(input[mid].x<v){lo=mid+1u;}else{hi=mid;}}
 storeCSR(p.offsets.w+v,lo);
}
`;

const ENTRIES=['initialize','hashEdges','seedEdges','seedFans','radixHistogram','radixPrefix','radixScatter','edgeGroups','edgePrefix','assembleEdges','assembleFanCorners','assembleFanOffsets','retainIndices'] as const;
type Entry=typeof ENTRIES[number];
const ENTRY_BINDINGS:Record<Entry,readonly number[]>={initialize:[0,7],hashEdges:[0,1,2,7],seedEdges:[0,1,2,4,7],seedFans:[0,1,4,7],radixHistogram:[0,3,5],radixPrefix:[0,5,6],radixScatter:[0,3,4,6],edgeGroups:[0,3,5],edgePrefix:[0,2,5,6,7],assembleEdges:[0,3,6,7],assembleFanCorners:[0,3,7],assembleFanOffsets:[0,3,7],retainIndices:[0,1,7]};
/** A producer-certified index primitive, not a scene or asset recipe. Validate
 * every live input word while writing the same CSR in one GPU dispatch. */
export const INDEPENDENT_QUAD_INCIDENCE_SHADER=/* wgsl */`
struct P {extent:vec4u,offsets:vec4u,tail:vec4u,pad:vec4u}
@group(0) @binding(0) var<uniform> p:P;
@group(0) @binding(1) var<storage,read> indices:array<u32>;
@group(0) @binding(2) var<storage,read_write> csr:array<atomic<u32>>;
fn put(at:u32,value:u32){atomicStore(&csr[at],value);}
@compute @workgroup_size(128) fn main(@builtin(global_invocation_id) id:vec3u){
 let h=id.x;let corners=p.extent.y;let quads=corners/6u;let edges=quads*5u;
 if(p.pad.x==${INDEXED_QUAD_INCIDENCE_VERSION}u){
  if(h==0u){put(0u,${INDEXED_INCIDENCE_MAGIC}u);put(1u,${INDEXED_QUAD_INCIDENCE_VERSION}u);put(2u,p.extent.x);put(3u,corners);put(4u,edges);put(5u,p.extent.z);}
  // Keep the original GPU certification and failure surface. Only its
  // expanded copies/fan tables disappear, never an unchecked producer fact.
  if(h<corners){var vertices=array<u32,6>(0u,1u,2u,0u,2u,3u);if(indices[h]!=h/6u*4u+vertices[h%6u]){atomicOr(&csr[11u],4u);}}
  return;
 }
 if(h==0u){put(0u,${INDEXED_INCIDENCE_MAGIC}u);put(1u,${INDEXED_INCIDENCE_VERSION}u);put(2u,p.extent.x);put(3u,corners);put(4u,edges);put(5u,p.extent.z);put(6u,p.offsets.x);put(7u,p.offsets.y);put(8u,p.offsets.z);put(9u,p.offsets.w);put(10u,p.tail.x);put(12u,p.tail.y);}
 if(h<corners){
  var vertices=array<u32,6>(0u,1u,2u,0u,2u,3u);let expected=h/6u*4u+vertices[h%6u];
  if(indices[h]!=expected){atomicOr(&csr[11u],4u);}
  put(p.tail.y+h,indices[h]);
  var edge=array<u32,6>(0u,1u,2u,2u,3u,4u);put(p.offsets.x+h,h/6u*5u+edge[h%6u]);put(p.offsets.z+h,h);
  var fan=array<u32,6>(0u,3u,1u,2u,4u,5u);put(p.tail.x+h,h/6u*6u+fan[h%6u]);
 }
 if(h<=edges){var starts=array<u32,5>(0u,1u,2u,4u,5u);put(p.offsets.y+h,select(corners,h/5u*6u+starts[h%5u],h<edges));}
 if(h<=p.extent.x){var starts=array<u32,4>(0u,2u,3u,5u);put(p.offsets.w+h,select(corners,h/4u*6u+starts[h%4u],h<quads*4u));}
}`;
/** One bounded upload scratch owner, shared by every source. The caller owns
 * the resulting range in its EXISTING source arena. Builds serialize only
 * registration, not displayed frames; no source graph is read back to JS. */
interface PreparedIncidence {
 readonly quad:GPUComputePipeline;
 readonly layouts:Record<'shared'|'separate',Record<Entry,GPUBindGroupLayout>>;
 readonly pipelines:Record<'shared'|'separate',Record<Entry,GPUComputePipeline>>;
}
function incidencePipelinePlan(device:GPUDevice){
 const plans=new Map<string,ComputePipelinePlan>();
 plans.set('independent-quads',()=>({label:'MAIN certified independent-quad incidence',layout:'auto',compute:{module:device.createShaderModule({code:isolateComputeEntry(INDEPENDENT_QUAD_INCIDENCE_SHADER,'main')}),entryPoint:'main'}}));
 // Binding subsets and their shared layouts are the original synchronization
 // contract. Preparing driver programs must not widen those subsets.
 const cached=new Map<string,GPUBindGroupLayout>();
 const layouts={} as PreparedIncidence['layouts'];
 for(const mode of ['shared','separate'] as const){
  layouts[mode]=Object.fromEntries(ENTRIES.map(entry=>{
   const bindings=ENTRY_BINDINGS[entry].filter(b=>mode!=='shared'||b!==1),key=bindings.join(',');let layout=cached.get(key);
   if(!layout){layout=device.createBindGroupLayout({entries:bindings.map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:binding===0?'uniform':binding===1||binding===3?'read-only-storage':'storage'}}))});cached.set(key,layout);}
   return [entry,layout];
  })) as Record<Entry,GPUBindGroupLayout>;
  const code=mode==='shared'?INDEXED_INCIDENCE_SHADER.replace('return indices[h];','return atomicLoad(&csr[p.origin.y+h]);'):INDEXED_INCIDENCE_SHADER;
  for(const entryPoint of ENTRIES){
   plans.set(`${mode}:${entryPoint}`,()=>{const module=device.createShaderModule({label:`MAIN upload indexed CSR ${mode} ${entryPoint}`,code:isolateComputeEntry(code,entryPoint)});
    return {label:`MAIN upload CSR ${entryPoint}`,layout:device.createPipelineLayout({bindGroupLayouts:[layouts[mode][entryPoint]]}),compute:{module,entryPoint}};});
  }
 }
 return {layouts,plans};
}
function preparedIncidence(layouts:PreparedIncidence['layouts'],pipelines:ReadonlyMap<string,GPUComputePipeline>):PreparedIncidence{
 return {layouts,quad:pipelines.get('independent-quads')!,pipelines:Object.fromEntries((['shared','separate'] as const).map(mode=>[mode,Object.fromEntries(ENTRIES.map(entry=>[entry,pipelines.get(`${mode}:${entry}`)!]))])) as PreparedIncidence['pipelines']};
}
export class GpuSourceIncidence {
 private readonly quadPipeline:GPUComputePipeline;
 private readonly pipelines:Record<'shared'|'separate',Record<Entry,GPUComputePipeline>>;
 private readonly layouts:Record<'shared'|'separate',Record<Entry,GPUBindGroupLayout>>;
 private pending:Promise<void>=Promise.resolve();private disposed=false;
 private readonly queuedIndexed:IndexedBuildRequest[]=[];private indexedScheduled=false;
 private scratch?:{buffers:GPUBuffer[];summary:GPUBuffer};
 static async create(device:GPUDevice,storage:SourceStorage,scratchBudget=8*1024*1024,progress?:ComputePipelineProgress):Promise<GpuSourceIncidence>{
  const {layouts,plans}=incidencePipelinePlan(device),pipelines=await prepareComputePipelines(device,plans,progress);
  return new GpuSourceIncidence(device,storage,scratchBudget,preparedIncidence(layouts,pipelines));
 }
 constructor(private readonly device:GPUDevice,private readonly storage:SourceStorage,private readonly scratchBudget=8*1024*1024,prepared?:PreparedIncidence){
  const ready=prepared??(()=>{const {layouts,plans}=incidencePipelinePlan(device);return preparedIncidence(layouts,compileComputePipelines(device,plans));})();
  this.quadPipeline=ready.quad;this.layouts=ready.layouts;this.pipelines=ready.pipelines;
 }
 build(indices:GPUBufferBinding,vertices:number,corners:number,edgeCapacity=corners):Promise<ResidentIndexedIncidence>{
  const layout=indexedIncidenceLayout(vertices,corners,edgeCapacity);
  return this.enqueue({indices,layout,retainIndices:false});
 }
 /** GPU-only sources retain their original indices in this same resident
  * range once per generation, keeping drawing passes inside eight bindings. */
 buildWithIndices(indices:GPUBufferBinding,vertices:number,corners:number,edgeCapacity=corners,indexPattern?:'independent-quads'):Promise<ResidentIndexedIncidence>{
  const layout=indexedIncidenceLayout(vertices,corners,edgeCapacity);
  if(indexPattern==='independent-quads'&&(corners%6||vertices<corners/6*4||edgeCapacity<corners/6*5))throw Error('Independent quad source has an invalid primitive extent');
  return this.enqueue({indices,layout,indexPattern,retainIndices:true});
 }
 private enqueue(request:Omit<IndexedBuildRequest,'resolve'|'reject'>):Promise<ResidentIndexedIncidence>{
  if(this.disposed)return Promise.reject(Error('GPU source incidence is disposed'));
  const result=new Promise<ResidentIndexedIncidence>((resolve,reject)=>this.queuedIndexed.push({...request,resolve,reject}));
  if(!this.indexedScheduled){this.indexedScheduled=true;queueMicrotask(()=>{
   const work=this.pending.then(()=>{
    // Worker results arrive in different event tasks. Take the cohort only
    // when the prior GPU upload finishes, not when the first result arrives;
    // otherwise every worker message still creates a one-source fence.
    this.indexedScheduled=false;
    return this.runIndexedBatch(this.queuedIndexed.splice(0));
   });
   this.pending=work.then(()=>undefined,()=>undefined);
  });}
  return result;
 }
 private async runIndexedBatch(requests:IndexedBuildRequest[]):Promise<void>{
  if(!requests.length)return;
  const uniforms:GPUBuffer[]=[],results:ResidentIndexedIncidence[]=[];let summary:GPUBuffer|undefined,submitted=false;
  try{
   if(this.disposed)throw Error('GPU source incidence is disposed');
   // One bounded scratch set is reused in command order. Parameters must
   // remain distinct: queue writes to one uniform would alias every source.
   const sizes=requests.filter(r=>!r.indexPattern).reduce<number[]>((max,r)=>{const s=indexedIncidenceScratch(r.layout);if(s.bytes>this.scratchBudget)throw Error('Source incidence exceeds bounded scratch budget');return s.sizes.map((n,i)=>Math.max(n,max[i]??0));},[]);
   if(sizes.length)this.uploadScratch(sizes);
   summary=this.device.createBuffer({label:'MAIN indexed source cohort descriptors',size:requests.length*64,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
   const encoder=this.device.createCommandEncoder({label:'MAIN indexed source cohort registration'});
   for(const [i,r]of requests.entries()){
    const batch={encoder,summary,offset:i*64,uniforms};
    results.push(r.indexPattern==='independent-quads'?this.encodeIndependentQuads(r.indices,r.layout,batch):await this.run(r.indices,r.layout,r.target,r.retainIndices,batch));
   }
   this.device.queue.submit([encoder.finish()]);submitted=true;await summary.mapAsync(GPUMapMode.READ);
   const words=new Uint32Array(summary.getMappedRange()).slice();summary.unmap();
   if(this.disposed)throw Error('GPU source incidence retired during cohort registration');
   for(let i=0;i<results.length;i++)if(words[i*16]!==INDEXED_INCIDENCE_MAGIC||words[i*16+1]!==(requests[i]!.indexPattern==='independent-quads'?INDEXED_QUAD_INCIDENCE_VERSION:INDEXED_INCIDENCE_VERSION)||words[i*16+11]!==0)throw Error(`GPU indexed cohort source ${i} failed (flags ${words[i*16+11]})`);
   results.forEach((value,i)=>requests[i]!.resolve({...value,edges:words[i*16+4]!}));
  }catch(error){
   if(submitted)await this.device.queue.onSubmittedWorkDone().catch(()=>{});
   // Caller-owned packet ranges are retired by their existing upload owner.
   // Only ranges allocated here belong to this batch's failure cleanup.
   for(const [i,result]of results.entries())if(!requests[i]!.target)this.storage.retire(result.range);
   for(const request of requests)request.reject(error);
  }finally{for(const uniform of uniforms)uniform.destroy();summary?.destroy();}
 }
 private encodeIndependentQuads(indices:GPUBufferBinding,layout:IndexedIncidenceLayout,batch:IndexedBuildBatch):ResidentIndexedIncidence{
  const range=this.storage.allocate(64);
  try{
   const binding=this.storage.binding(range),params=this.device.createBuffer({label:'MAIN independent-quad incidence parameters',size:64,usage:GPUBufferUsage.UNIFORM,mappedAtCreation:true});batch.uniforms.push(params);
   new Uint32Array(params.getMappedRange()).set([layout.vertices,layout.corners,layout.edgeCapacity,0,layout.cornerEdge,layout.edgeOffsets,layout.edgeCorners,layout.fanOffsets,layout.fanCorners,layout.words,0,0,INDEXED_QUAD_INCIDENCE_VERSION,0,0,0]);params.unmap();
   batch.encoder.clearBuffer(binding.buffer,binding.offset??0,binding.size);
   const group=this.device.createBindGroup({layout:this.quadPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:params}},{binding:1,resource:indices},{binding:2,resource:binding}]});
   const pass=batch.encoder.beginComputePass({label:'MAIN certified quad incidence program'});pass.setPipeline(this.quadPipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(layout.corners/128));pass.end();
   batch.encoder.copyBufferToBuffer(binding.buffer,binding.offset??0,batch.summary,batch.offset,64);
   return {layout:{...layout,words:16},range,binding,edges:layout.corners/6*5};
  }catch(error){this.storage.retire(range);throw error;}
 }
 /** Write directly into a caller-owned source-packet range. No intermediate
  * resident CSR, copy bridge or second lifetime owner. On failure the caller
  * retains ownership of its complete packet and retires it at its fence. */
 buildInto(indices:GPUBufferBinding,layout:IndexedIncidenceLayout,target:IndexedIncidenceTarget):Promise<ResidentIndexedIncidence>{
  const expected=indexedIncidenceLayout(layout.vertices,layout.corners,layout.edgeCapacity);
  if(Object.keys(expected).some(key=>expected[key as keyof IndexedIncidenceLayout]!==layout[key as keyof IndexedIncidenceLayout]))throw Error('Inconsistent indexed incidence layout');
  const owner=this.storage.binding(target.range),offset=target.binding.offset??0,size=target.binding.size??target.binding.buffer.size-offset;
  if(owner.buffer!==target.binding.buffer||offset<(owner.offset??0)||offset+size>(owner.offset??0)+owner.size!||size!==layout.words*4)throw Error('Indexed incidence target is not its caller-owned source range');
  return this.enqueue({indices,layout,target,retainIndices:false});
 }
 private freeScratch():void{if(this.scratch){for(const buffer of this.scratch.buffers)buffer.destroy();this.scratch.summary.destroy();this.scratch=undefined;}}
 private uploadScratch(sizes:readonly number[]):{buffers:GPUBuffer[];summary:GPUBuffer}{
  if(this.scratch&&sizes.every((size,i)=>this.scratch!.buffers[i]!.size>=size))return this.scratch;
  // run() is serialized through the preceding upload's terminal fence.
  // Destroy before growth, so two scratch capacities never overlap.
  this.freeScratch();const made:GPUBuffer[]=[];
  try{
   for(const [i,size]of sizes.entries()){
    if(size>Math.min(this.device.limits.maxBufferSize,this.device.limits.maxStorageBufferBindingSize))throw Error('Source incidence scratch exceeds a device binding limit');
    made.push(this.device.createBuffer({label:`MAIN upload CSR scratch ${i}`,size,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}));
   }
   const summary=this.device.createBuffer({label:'MAIN upload CSR summary',size:64,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
   return this.scratch={buffers:made,summary};
  }catch(error){for(const buffer of made)buffer.destroy();throw error;}
 }
 private async run(indices:GPUBufferBinding,layout:IndexedIncidenceLayout,target?:IndexedIncidenceTarget,retainIndices=false,batch?:IndexedBuildBatch):Promise<ResidentIndexedIncidence>{
  if(this.disposed)throw Error('GPU source incidence is disposed');
  const {vertices,corners,edgeCapacity}=layout,{groups,slots,resident,spill,sizes,bytes}=indexedIncidenceScratch(layout);
  if(bytes>this.scratchBudget)throw Error('Source incidence needs a chunked upload larger than the bounded scratch budget');
  if(groups>this.device.limits.maxComputeWorkgroupsPerDimension||Math.ceil((vertices+1)/128)>this.device.limits.maxComputeWorkgroupsPerDimension)throw Error('Source incidence exceeds dispatch extent');
  const available=indices.size??indices.buffer.size-(indices.offset??0);
  if(available<corners*4)throw Error('Source incidence index binding is too short');
  let range:SourceStorageRange|undefined=target?.range;let submitted=false;
  try{
   const scratch=this.uploadScratch(sizes),buffers=[...scratch.buffers],summary=batch?.summary??scratch.summary;
   if(batch){buffers[5]=this.device.createBuffer({label:'MAIN indexed cohort immutable parameters',size:9*256,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});batch.uniforms.push(buffers[5]!);}
   range??=this.storage.allocate((layout.words+(retainIndices?corners:0))*4);const binding=target?.binding??this.storage.binding(range);
   const mode=indices.buffer===binding.buffer?'shared':'separate';
   // One RW binding for a shared page: WebGPU does not permit binding the
   // same buffer as both readonly indices and writable CSR in one pass.
   const workingBinding=mode==='shared'?{buffer:binding.buffer}:binding;
   const csrBase=mode==='shared'?(binding.offset??0)/4:0,indexBase=mode==='shared'?(indices.offset??0)/4:0;
   const parameters=new Uint32Array(9*64);
   for(let i=0;i<9;i++)parameters.set([vertices,corners,edgeCapacity,slots-1,layout.cornerEdge,layout.edgeOffsets,layout.edgeCorners,layout.fanOffsets,layout.fanCorners,groups,i<8?i*4:0,retainIndices?layout.words:0,csrBase,indexBase,resident,spill],i*64);
   this.device.queue.writeBuffer(buffers[5]!,0,parameters);
   const a=buffers[0]!,b=buffers[1]!,hash=buffers[2]!,hist=buffers[3]!,prefix=buffers[4]!,uniform=buffers[5]!;
   const group=(input:GPUBuffer,output:GPUBuffer,step=8)=>({input,output,step,groups:new Map<Entry,GPUBindGroup>()});
   const ba=group(b,a),ab=group(a,b),radix=Array.from({length:8},(_,i)=>i%2?group(a,b,i):group(b,a,i));
   const encoder=batch?.encoder??this.device.createCommandEncoder({label:'MAIN source indexed incidence upload'});
   encoder.clearBuffer(hash);encoder.clearBuffer(binding.buffer,binding.offset??0,binding.size);
   // Compute resource usage is scoped per dispatch. Keep the ordered stages
   // together instead of opening ~60 passes per registered source. The one
   // encoder clear between hash and assembly is the required pass boundary:
   // the borrowed CSR hash must be dead before its final header is written.
   let pass=encoder.beginComputePass({label:'MAIN indexed source hash'});
   const dispatch=(entry:Entry,g:ReturnType<typeof group>,n=groups)=>{
    let bound=g.groups.get(entry);
    if(!bound){const resources=[{buffer:uniform,offset:g.step*256,size:64},indices,{buffer:hash},{buffer:g.input},{buffer:g.output},{buffer:hist},{buffer:prefix},workingBinding];bound=this.device.createBindGroup({layout:this.layouts[mode][entry],entries:ENTRY_BINDINGS[entry].filter(b=>mode!=='shared'||b!==1).map(binding=>({binding,resource:resources[binding]!}))});g.groups.set(entry,bound);}
    pass.setPipeline(this.pipelines[mode][entry]);pass.setBindGroup(0,bound);pass.dispatchWorkgroups(n);
   };
   const sort=()=>{for(let i=0;i<8;i++){dispatch('radixHistogram',radix[i]!);dispatch('radixPrefix',radix[i]!,1);dispatch('radixScatter',radix[i]!);}};
   dispatch('hashEdges',ba);dispatch('seedEdges',ab);
   pass.end();
   // The borrowed hash is dead now. Header/CSR assembly may overwrite it.
   // Clear it again so unused capacity/padding remains byte-identical too.
   encoder.clearBuffer(binding.buffer,binding.offset??0,binding.size);
   pass=encoder.beginComputePass({label:'MAIN indexed source assembly'});
   dispatch('initialize',ba,1);sort();
   dispatch('edgeGroups',ba);dispatch('edgePrefix',ba,1);dispatch('assembleEdges',ba);
   dispatch('seedFans',ab);sort();dispatch('assembleFanCorners',ba);dispatch('assembleFanOffsets',ba,Math.ceil((vertices+1)/128));
   if(retainIndices)dispatch('retainIndices',ba);
   pass.end();
   // Only the fixed descriptor comes back. It guards capacity/input errors
   // before a range can become resident; no incidence/geometry is decoded.
   encoder.copyBufferToBuffer(binding.buffer,binding.offset??0,summary,batch?.offset??0,64);
   if(batch)return {layout,edges:layout.edgeCapacity,range,binding};
   this.device.queue.submit([encoder.finish()]);submitted=true;await summary.mapAsync(GPUMapMode.READ);
   const descriptor=new Uint32Array(summary.getMappedRange()).slice();summary.unmap();
   if(descriptor[0]!==INDEXED_INCIDENCE_MAGIC||descriptor[1]!==INDEXED_INCIDENCE_VERSION||descriptor[11]!==0)throw Error(`GPU indexed incidence build failed (flags ${descriptor[11]})`);
   if(this.disposed)throw Error('GPU source incidence retired during upload');
   return {layout,edges:descriptor[4]!,range,binding};
  }catch(error){if(submitted)await this.device.queue.onSubmittedWorkDone().catch(()=>{});if(range&&!target)this.storage.retire(range);throw error;}
 }
 dispose():void{if(this.disposed)return;this.disposed=true;void this.pending.then(()=>this.freeScratch());}
}
