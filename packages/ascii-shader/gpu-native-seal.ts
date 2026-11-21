import {WORD_STRIDE} from './role-grammar-format.ts';
import type {GpuNativeTape} from './gpu-native-tape.ts';
import {DeviceComputePrograms,type ComputePipelinePlan,type ComputePipelineProgress} from './gpu-pipeline-preparation.ts';
import {isolateComputeEntry} from './gpu-compute-program.ts';
/** Canonicalize the committed native cursors, not the proposal words.
 * Shared glyphs have one cursor and therefore one displayed scalar. The
 * original word records remain available separately for contribution export.
 * No glyph choice, displacement, fitting or construction repair occurs here.
 */
export const GPU_NATIVE_SEAL_SHADER=/* wgsl */`
const WS=${WORD_STRIDE}u;
struct Params {width:u32,rows:u32,wordCapacity:u32,scalarCapacity:u32,metrics:u32,columns:u32,relaxed:u32}
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read> placed:array<u32>;
@group(0) @binding(2) var<storage,read> grammar:array<u32>;
@group(0) @binding(3) var<storage,read> words:array<u32>;
@group(0) @binding(4) var<storage,read> gaps:array<u32>;
@group(0) @binding(5) var<storage,read_write> scalars:array<u32>;
// DrawIndirect followed by the seal's error flag. Word-placement overflow
// is checked independently before this snapshot can become a publication.
@group(0) @binding(6) var<storage,read_write> status:array<atomic<u32>>;
@group(0) @binding(7) var<storage,read_write> arguments:array<u32>;
fn cursorBase(row:u32)->u32{return p.wordCapacity*32u+row*(p.width*2u+1u);}
fn total(base:u32)->u32{
 var x=p.width;var result=0u;
 while(x>0u){result+=placed[base+x];x-=x&(~x+1u);}
 return result;
}
fn referenceAt(base:u32,ordinal:u32)->u32{
 var bit=1u;while(bit<=p.width/2u){bit*=2u;}
 var index=0u;var remaining=ordinal;
 while(bit>0u){let next=index+bit;if(next<=p.width&&placed[base+next]<remaining){remaining-=placed[base+next];index=next;}bit/=2u;}
 return placed[base+p.width+1u+index];
}
// The articulation owner's existing skyline/correspondence header permits
// at most 508 rows. This 512-entry scan covers that whole interface, not a
// quota on forms, words or glyphs. It never walks a row's scalar records.
var<workgroup> offsets:array<u32,512>;
@compute @workgroup_size(128) fn sealRows(@builtin(local_invocation_index) lane:u32){
 for(var row=lane;row<512u;row+=128u){
  var count=0u;if(row<p.rows){count=total(cursorBase(row));
   if(count>p.width){atomicOr(&status[4],8u);atomicMax(&status[5],count);count=0u;}
  }
  offsets[row]=count;
 }
 workgroupBarrier();
 // Work-efficient exclusive scan. Each level has disjoint writes and a
 // barrier, unlike reading a partly updated prefix from another lane.
 for(var span=1u;span<512u;span*=2u){
  for(var k=lane;k<512u/(span*2u);k+=128u){let at=(k+1u)*span*2u-1u;offsets[at]+=offsets[at-span];}
  workgroupBarrier();
 }
 if(lane==0u){
  let count=offsets[511];atomicStore(&status[1],count);atomicStore(&status[8u+p.rows],count);
  if(count>p.scalarCapacity){atomicOr(&status[4],1u);}
  let groups=(min(count,p.scalarCapacity)+63u)/64u;
  arguments[0]=min(groups,p.columns);arguments[1]=max(1u,(groups+p.columns-1u)/p.columns);arguments[2]=1u;
  offsets[511]=0u;
 }
 workgroupBarrier();
 for(var span=256u;span>0u;span/=2u){
  for(var k=lane;k<512u/(span*2u);k+=128u){let at=(k+1u)*span*2u-1u;let left=offsets[at-span];offsets[at-span]=offsets[at];offsets[at]+=left;}
  workgroupBarrier();
 }
 for(var row=lane;row<p.rows;row+=128u){atomicStore(&status[8u+row],offsets[row]);}
}
struct Scalar {at:u32,scalar:u32,font:u32,x:u32,advance:u32}
fn scalarAt(base:u32,ordinal:u32)->Scalar{
  let reference=referenceAt(base,ordinal);let at=(reference>>4u)*32u;let scalar=reference&15u;
  var font=placed[at+5u];var x=placed[at+3u];
  if(placed[at+9u]==0u||placed[at+9u]==3u){let word=font;
   let start=select(0u,placed[at+13u],placed[at+9u]==0u);
   for(var i=0u;i<scalar;i++){x+=grammar[p.metrics+words[word*WS+2u+start+i]*4u+1u];}
   font=words[word*WS+2u+start+scalar];
  }
  let advance=grammar[p.metrics+font*4u+1u];
  return Scalar(at,scalar,font,x,advance);
}
@compute @workgroup_size(64) fn seal(@builtin(global_invocation_id) id:vec3u){
 let index=id.x+id.y*p.columns*64u;
 if(index>=min(atomicLoad(&status[1]),p.scalarCapacity)){return;}
 // Locate the scalar's row in the compact prefix table. This is an address
 // lookup, not a glyph/pen choice; duplicate offsets skip empty rows.
 var lo=0u;var hi=p.rows;
 while(lo<hi){let mid=lo+(hi-lo)/2u;if(atomicLoad(&status[8u+mid+1u])<=index){lo=mid+1u;}else{hi=mid;}}
 let row=lo;let first=atomicLoad(&status[8u+row]);let end=atomicLoad(&status[8u+row+1u]);
 // Preserve the incumbent capacity policy: never seal a partial row.
 if(end>p.scalarCapacity){return;}
 let ordinal=index-first+1u;let base=cursorBase(row);let current=scalarAt(base,ordinal);
 let at=current.at;let scalar=current.scalar;let font=current.font;let x=current.x;let advance=current.advance;
 var previousEnd=0u;
 if(ordinal>1u){let prior=scalarAt(base,ordinal-1u);previousEnd=prior.x+prior.advance;}
  if(x<previousEnd||x+advance>p.width){atomicOr(&status[4],2u);}
  else{let gap=x-previousEnd;let leading=select(0u,p.width+1u,previousEnd==0u);
   // relaxed: a camera covered by a stacked camera; its gaps may span that
   // camera's glyphs, whose merged page gaps the export commit checks.
   if(p.relaxed==0u&&gaps[leading+gap]!=gap){atomicOr(&status[4],4u);}
  }
  let out=index*32u;
  for(var k=0u;k<32u;k++){scalars[out+k]=placed[at+k];}
  scalars[out+3u]=x;scalars[out+4u]=row;scalars[out+5u]=font;
  scalars[out+6u]=1u;scalars[out+7u]=advance;scalars[out+9u]=4u;
  scalars[out+13u]=select(placed[at+13u],placed[at+13u]+scalar,placed[at+9u]==0u);
}
`;

/** One publication operation owns the complete count/scan/scalar execution.
 * Allocation ownership is supplied by the frame lease: immutable tape/count
 * buffers transfer to publication; parameters and arguments retire with the job.
 */
interface PreparedGpuNativeSealer{
 readonly pipelines:ReadonlyMap<'sealRows'|'seal',GPUComputePipeline>;
}
function nativeSealPipelinePlans(device:GPUDevice):ReadonlyMap<'sealRows'|'seal',ComputePipelinePlan>{
 const plans=new Map<'sealRows'|'seal',ComputePipelinePlan>();
 for(const entryPoint of ['sealRows','seal'] as const){
  plans.set(entryPoint,()=>{const module=device.createShaderModule({label:'MAIN parallel authoritative native scalar tape '+entryPoint,code:isolateComputeEntry(GPU_NATIVE_SEAL_SHADER,entryPoint)});
   return {layout:'auto',compute:{module,entryPoint}};});
 }
 return plans;
}
const nativeSealPrograms=new DeviceComputePrograms((device:GPUDevice)=>({resources:undefined,plans:nativeSealPipelinePlans(device)}));
export class GpuNativeSealer {
 private readonly rows:GPUComputePipeline;
 private readonly scalars:GPUComputePipeline;
 static async create(device:GPUDevice,progress?:ComputePipelineProgress):Promise<GpuNativeSealer>{
  return new GpuNativeSealer(device,await nativeSealPrograms.prepare(device,progress));
 }
 constructor(private readonly device:GPUDevice,prepared?:PreparedGpuNativeSealer){
  const pipelines=prepared?.pipelines??nativeSealPrograms.get(device).pipelines;
  this.rows=pipelines.get('sealRows')!;
  this.scalars=pipelines.get('seal')!;
 }
 encode(encoder:GPUCommandEncoder,input:{width:number;height:number;rows:number;wordCapacity:number;scalarCapacity:number;metrics:number;fontAddressCount:number;placed:GPUBuffer;grammar:GPUBuffer;words:GPUBuffer;gaps:GPUBuffer;relaxedGaps?:boolean},own:(buffer:GPUBuffer)=>GPUBuffer):Omit<GpuNativeTape,'seal'>{
  // This is already enforced more tightly by articulation's 512-word
  // skyline header. Keep the module's own interface explicit for callers.
  if(input.rows>512)throw Error('Native seal exceeds the articulation row extent');
  const bytes=input.scalarCapacity*128;
  if(bytes>Math.min(this.device.limits.maxStorageBufferBindingSize,this.device.limits.maxBufferSize))throw Error('Native scalar tape needs device-sized page chunks');
  const words=own(this.device.createBuffer({label:'MAIN sealed native scalar tape',size:Math.max(16,bytes),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}));
  const count=own(this.device.createBuffer({label:'MAIN native scalar count and row offsets',size:(9+input.rows)*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC|GPUBufferUsage.INDIRECT}));
  this.device.queue.writeBuffer(count,0,new Uint32Array([6,0,0,0,0,0,0,0]));
  const data=new Uint32Array(8);data.set([input.width,input.rows,input.wordCapacity,input.scalarCapacity,input.metrics,this.device.limits.maxComputeWorkgroupsPerDimension,input.relaxedGaps?1:0]);
  const parameters=own(this.device.createBuffer({label:'MAIN native seal parameters',size:data.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}));this.device.queue.writeBuffer(parameters,0,data);
  // Indirect arguments cannot alias a storage binding of the scalar pass.
  const argumentsBuffer=own(this.device.createBuffer({label:'MAIN active scalar seal arguments',size:16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.INDIRECT}));
  const bind=(pipeline:GPUComputePipeline,entries:readonly (readonly [number,GPUBuffer])[])=>this.device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:entries.map(([binding,buffer])=>({binding,resource:{buffer}}))});
  const rowGroup=bind(this.rows,[[0,parameters],[1,input.placed],[6,count],[7,argumentsBuffer]]);
  const scalarGroup=bind(this.scalars,[parameters,input.placed,input.grammar,input.words,input.gaps,words,count].map((buffer,binding)=>[binding,buffer] as const));
  const rows=encoder.beginComputePass({label:'MAIN parallel native row offsets'});rows.setPipeline(this.rows);rows.setBindGroup(0,rowGroup);rows.dispatchWorkgroups(1);rows.end();
  const scalars=encoder.beginComputePass({label:'MAIN one invocation per displayed scalar'});scalars.setPipeline(this.scalars);scalars.setBindGroup(0,scalarGroup);scalars.dispatchWorkgroupsIndirect(argumentsBuffer,0);scalars.end();
  return {words,count,countOffsetBytes:4,capacity:input.scalarCapacity,width:input.width,height:input.height,fontAddressCount:input.fontAddressCount};
 }
}
