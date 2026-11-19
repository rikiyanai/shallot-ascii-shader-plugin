import {RELATION_WORDS,type SurfaceRelations} from './surface-relations.ts';
import type {SurfaceField} from './surface-fields.ts';
import {RELATION_RECORD_WGSL} from './source-coordinates.ts';
import {compileComputePipelines,prepareComputePipelines,type ComputePipelinePlan} from './gpu-pipeline-preparation.ts';
import {isolateComputeEntry} from './gpu-compute-program.ts';

/** Same packed decoder format, with actual counts as its record extents. */
export interface ProjectedRelationMemory {readonly memory:ArrayBuffer;readonly capacity:number;readonly intervalCapacity:number}
export interface ProjectedRelationReadback {collect():Promise<ProjectedRelationMemory>;dispose():void}

export const RELATION_SNAPSHOT_SHADER=/* wgsl */`
struct Params {extent:vec4u}
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read> status:array<u32>;
@group(0) @binding(2) var<storage,read> lines:array<vec4u>;
@group(0) @binding(3) var<storage,read> intervals:array<vec4u>;
@group(0) @binding(4) var<storage,read_write> retainedLines:array<vec4u>;
@group(0) @binding(5) var<storage,read_write> retainedIntervals:array<vec4u>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid:vec3u){
 if(status[1]!=0u){return;}
 let id=gid.x+gid.y*p.extent.z*64u;
 // Copy complete emitted records only. Unused allocation tails never
 // cross the GPU/CPU interface, nor get copied into a per-job snapshot.
 if(id<min(status[2],p.extent.x)){for(var k=0u;k<${RELATION_WORDS/4}u;k++){retainedLines[id*${RELATION_WORDS/4}u+k]=lines[id*${RELATION_WORDS/4}u+k];}}
 if(id<min(status[0],p.extent.y)){retainedIntervals[id]=intervals[id];}
}`;

/** A source-pose snapshot owns its storage until its compact copy completes.
 * The host's normal encoder/submission drives both stages; no private submit,
 * live-buffer reread, geometry change or selection occurs during collection. */
class CountedRelationReadback implements ProjectedRelationReadback {
 private readonly header:GPUBuffer;
 readonly lines:GPUBuffer;readonly intervals:GPUBuffer;
 private payload?:GPUBuffer;private counts?:Uint32Array;
 private result?:Promise<ProjectedRelationMemory>;
 private ready=false;private encoded=false;private mapping=false;private disposed=false;
 private resolve?:((value:ProjectedRelationMemory)=>void);private reject?:((error:unknown)=>void);
 constructor(private readonly device:GPUDevice,readonly capacity:number,readonly intervalCapacity:number,private readonly retire:()=>void){
  this.header=device.createBuffer({label:'MAIN relation count header',size:16,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  this.lines=device.createBuffer({label:'MAIN retained posed relation records',size:Math.max(16,capacity*RELATION_WORDS*4),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  this.intervals=device.createBuffer({label:'MAIN retained posed visibility intervals',size:Math.max(16,intervalCapacity*16),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
 }
 encodeHeader(encoder:GPUCommandEncoder,status:GPUBuffer):void{encoder.copyBufferToBuffer(status,0,this.header,0,16);}
 collect():Promise<ProjectedRelationMemory>{return this.result??=this.read().finally(()=>this.dispose());}
 private async read():Promise<ProjectedRelationMemory>{
  if(this.disposed)throw Error('Relation readback was retired');
  await this.header.mapAsync(GPUMapMode.READ);
  if(this.disposed)throw Error('Relation readback was retired');
  const counts=this.counts=new Uint32Array(this.header.getMappedRange().slice(0));this.header.unmap();
  const relations=counts[2]!,intervals=counts[0]!;
  // Return exhausted counts to the existing SAME-pose allocation retry.
  // Never copy or decode the partial records of an overflowed submission.
  if(counts[1])return {memory:counts.buffer as ArrayBuffer,capacity:0,intervalCapacity:0};
  if(relations>this.capacity||intervals>this.intervalCapacity)throw Error('Invalid complete relation snapshot counts');
  const bytes=relations*RELATION_WORDS*4+intervals*16;
  if(!bytes)return {memory:counts.buffer as ArrayBuffer,capacity:0,intervalCapacity:0};
  if(bytes>this.device.limits.maxBufferSize)throw Error('Counted relation readback needs device-sized chunks');
  this.payload=this.device.createBuffer({label:'MAIN count-sized relation readback',size:bytes,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  return new Promise((resolve,reject)=>{this.resolve=resolve;this.reject=reject;this.ready=true;});
 }
 encode(encoder:GPUCommandEncoder):void{
  if(!this.ready||this.disposed)return;this.ready=false;this.encoded=true;
  const relations=this.counts![2]!,intervals=this.counts![0]!;
  if(relations)encoder.copyBufferToBuffer(this.lines,0,this.payload!,0,relations*RELATION_WORDS*4);
  if(intervals)encoder.copyBufferToBuffer(this.intervals,0,this.payload!,relations*RELATION_WORDS*4,intervals*16);
 }
 submitted():void{
  if(!this.encoded||this.mapping||this.disposed)return;this.mapping=true;
  const payload=this.payload!;
  void payload.mapAsync(GPUMapMode.READ).then(()=>{
   if(this.disposed)return;
   const data=payload.getMappedRange(),memory=new ArrayBuffer(16+data.byteLength);
   new Uint32Array(memory,0,4).set(this.counts!);new Uint8Array(memory,16).set(new Uint8Array(data));payload.unmap();
   this.resolve!({memory,capacity:this.counts![2]!,intervalCapacity:this.counts![0]!});
  }).catch(error=>this.reject?.(error));
 }
 dispose():void{
  if(this.disposed)return;this.disposed=true;this.ready=false;
  this.reject?.(Error('Relation readback was retired'));
  for(const buffer of [this.header,this.lines,this.intervals,this.payload])if(buffer){if(buffer.mapState==='mapped')buffer.unmap();buffer.destroy();}
  this.retire();
 }
}
const SHADER=/* wgsl */`
struct Params {clip:mat4x4f,extent:vec4f,capacity:vec4u}
${RELATION_RECORD_WGSL}
struct Interval {relation:u32,lo:f32,hi:f32,cause:u32}
struct Status {count:atomic<u32>,overflow:atomic<u32>,relations:atomic<u32>,reserved:atomic<u32>}
@group(0) @binding(0) var<uniform> params:Params;
@group(0) @binding(1) var<storage,read> source:array<Relation>;
@group(0) @binding(2) var<storage,read> sourceStatus:array<u32>;
@group(0) @binding(3) var depth:texture_depth_multisampled_2d;
@group(0) @binding(4) var<storage,read_write> projected:array<Relation>;
@group(0) @binding(5) var<storage,read_write> intervals:array<Interval>;
@group(0) @binding(6) var<storage,read_write> status:Status;
@group(0) @binding(7) var layerDepth:texture_2d<f32>;
@group(0) @binding(8) var layerOwner:texture_2d<u32>;
@group(0) @binding(9) var layerColor:texture_2d<f32>;
// GPU construction consumes each source relation's ordered partitions directly.
// Heads/tails and next/previous links use index+1 (zero is the empty list).
@group(0) @binding(10) var<storage,read_write> partitions:array<u32>;
var<private> previousInterval:u32;
fn screen(p:vec4f)->vec3f{return vec3f((p.xy/p.w*vec2f(.5,-.5)+vec2f(.5))*params.extent.xy,p.z/p.w);}
// A screen-space sample is mapped back to the original homogeneous edge
// parameter. Do not use a linear clip-space mix for this: under perspective,
// that would move every probe when the visible interval is clipped.
fn edgeTFromScreenLambda(lambda:f32,wa:f32,wb:f32)->f32{
 let denominator=(1.0-lambda)*wb+lambda*wa;
 return clamp(lambda*wa/max(denominator,1e-20),0.0,1.0);
}
fn screenLambdaFromEdgeT(t:f32,wa:f32,wb:f32)->f32{
 let denominator=(1.0-t)*wa+t*wb;
 return clamp(t*wb/max(denominator,1e-20),0.0,1.0);
}
// Invert a projected page point on the ORIGINAL homogeneous edge. This is
// used for a crossing edge whose far endpoint has no finite page projection:
// the lattice still advances by a fixed page distance from its finite source
// anchor, rather than by a fraction of the currently clipped span.
fn edgeTFromPagePoint(pagePoint:vec2f,a:vec4f,b:vec4f)->f32{
 let ndc=vec2f((pagePoint.x/params.extent.x-.5)*2.0,(pagePoint.y/params.extent.y-.5)*-2.0);
 let delta=b.xy-a.xy;let deltaW=b.w-a.w;
 let denominatorX=ndc.x*deltaW-delta.x;let denominatorY=ndc.y*deltaW-delta.y;
 let useX=abs(denominatorX)>=abs(denominatorY);
 let numerator=select(a.y-ndc.y*a.w,a.x-ndc.x*a.w,useX);
 let denominator=select(denominatorY,denominatorX,useX);
 let safe=select(denominator,select(-1e-20,1e-20,denominator>=0.0),abs(denominator)<1e-20);
 return clamp(numerator/safe,0.0,1.0);
}
fn visible(p:vec4f,key:vec4u,gradient:vec2f)->bool{
 let q=screen(p);let page=vec2i(q.xy);if(any(page<vec2i(0))||any(page>=vec2i(params.extent.xy))){return false;}
 // CELL-RESOLUTION DEPTH (2026-11-12): the depth texture may be the glyph
 // grid's field, not the page. Its texels are addressed in its own units and
 // the halo below is one of ITS texels; a page-sized texture reads as before.
 let scale=vec2f(textureDimensions(depth))/params.extent.xy;let d=q.xy*scale;let xy=vec2i(d);
 // Conservative geometric candidate visibility has a finite ONE-TEXEL
 // footprint. It cannot turn an unbounded source depth slope into permission
 // to see through an unrelated foreground. Native ink admission must still
 // test the receiver at its own samples; this halo is not glyph visibility.
 for(var y=-1;y<=1;y++){for(var x=-1;x<=1;x++){
  let pixel=xy+vec2i(x,y);if(any(pixel<vec2i(0))||any(pixel>=vec2i(textureDimensions(depth)))){continue;}
  if(any(abs(vec2f(pixel)+vec2f(.5)-d)>vec2f(1.0))){continue;}
  // Reverse-Z: the nearest covered MSAA sample owns visibility. Taking
  // the minimum treated an uncovered sample (clear depth 0) as permission
  // for a back edge to pass through an otherwise opaque surface.
  var z=0.0;for(var sample=0u;sample<textureNumSamples(depth);sample++){z=max(z,textureLoad(depth,pixel,i32(sample)));}
  if(q.z+0.0000001>=z){return true;}
  if(params.capacity.z!=0u){
   // A sloping source edge and the depth texel centre are DIFFERENT points.
   // Comparing their depths directly creates alternating self-occlusion on
   // a perfectly visible plane. Recover equality on that same source plane,
   // not a slope-sized depth tolerance through arbitrary foreground.
   let size=textureDimensions(layerDepth);
   let fieldPixel=vec2i((vec2f(pixel)+vec2f(.5))/vec2f(textureDimensions(depth))*vec2f(size));
   let owner=textureLoad(layerOwner,fieldPixel,0).xy;
   let same=(owner.x&0x5fffffffu)==key.x&&owner.y==key.y&&(owner.x&0x80000000u)==(key.z&0x80000000u);
   if(same&&textureLoad(layerColor,fieldPixel,0).a>=.99){
    let at=(vec2f(fieldPixel)+vec2f(.5))/vec2f(size)*params.extent.xy;
    let predicted=q.z+dot(gradient,(at-q.xy)*vec2f(2.0,-2.0)/params.extent.xy);
    let actual=textureLoad(layerDepth,fieldPixel,0).x;
    // Equality is evaluated at THIS field texel centre. The native depth
    // above is the nearest MSAA sample, at a different subpixel position;
    // requiring it to equal the centre depth rejects sloping visible faces
    // and their contacts. The field's opaque source owner plus plane-depth
    // equality establishes a visible contribution at the centre. A foreign
    // foreground or a nearer face of the same mesh still fails this test.
    // The raster places that plane's vertices on a subpixel grid, so the
    // stored depth differs from the analytic plane by the plane's own depth
    // change over one grid step. On a face seen at a grazing angle that
    // exceeds the fixed epsilon: the base edge of a volume standing on a
    // surface, and its contact, then read hidden on every sample (the box
    // side above is nearer, the floor below is nearer) and the volume is
    // drawn open at the bottom. The allowance is 1/16 pixel of this
    // relation's own plane slope (4 subpixel bits, the least a WebGPU
    // backend provides). It applies only on a texel the same source owns.
    // In texels of the field (1/16 of one field texel of this plane's slope).
    if(abs(predicted-actual)<=0.0000001+length(gradient*vec2f(2.0)/vec2f(size))*.0625){return true;}
   }
  }
 }}return false;
}
fn classify(p:vec4f,line:Relation,gradient:vec2f)->vec3u{
 let key=line.key;
 let cause=select(1u,0u,visible(p,key,gradient));if(params.capacity.z==0u){return vec3u(cause,0u,0u);}
 let q=screen(p);let size=textureDimensions(layerDepth);let pixel=vec2i(q.xy/params.extent.xy*vec2f(size));
 if(any(pixel<vec2i(0))||any(pixel>=vec2i(size))){return vec3u(cause,0u,0u);}
 let owner=textureLoad(layerOwner,pixel,0).xy;let source=owner.x&0x5fffffffu;
 let same=source==key.x&&owner.y==key.y&&(owner.x&0x80000000u)==(key.z&0x80000000u);
 // A finite surface intersection belongs to BOTH participants. Crossing
 // their raster ownership boundary must not split a visible stroke into
 // alternating foreground-attribution intervals. This does not change
 // visible(): a nearer face (including either participant) still occludes.
 let participant=(key.z&255u)==4u&&(owner.x&0x9fffffffu)==line.ports.z&&owner.y==line.ports.w;
 if(source!=0u&&!same&&!participant&&textureLoad(layerColor,pixel,0).a>.01&&textureLoad(layerDepth,pixel,0).x>q.z+.0000001){
  // Foreground coverage is independent of opaque visibility. Transparent
  // geometry keeps a visible source interval while its native ink can own
  // a scalar span. Owner transitions split the interval for exact attribution.
  return vec3u(cause|4u,owner.x,owner.y);
 }
 return vec3u(cause,0u,0u);
}
fn emit(id:u32,lo:f32,hi:f32,cause:u32){
 if(hi<=lo){return;}let at=atomicAdd(&status.count,1u);if(at>=params.capacity.y){atomicOr(&status.overflow,2u);return;}
 intervals[at]=Interval(id,lo,hi,cause);
 let base=params.capacity.x*2u+at*2u;
 partitions[base]=0u;partitions[base+1u]=previousInterval;
 if(previousInterval==0u){partitions[id*2u]=at+1u;}
 else{partitions[params.capacity.x*2u+(previousInterval-1u)*2u]=at+1u;}
 previousInterval=at+1u;partitions[id*2u+1u]=previousInterval;
}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid:vec3u){
 let id=gid.x+gid.y*params.capacity.w*64u;
 // An exhausted contact atlas reports its complete required triangle count
 // through the same compact retry header, never as a partial drawing.
 if(id==0u){atomicStore(&status.reserved,select(sourceStatus[0],sourceStatus[3],(sourceStatus[1]&16u)!=0u));}
 if(sourceStatus[1]!=0u){if(id==0u){atomicOr(&status.overflow,sourceStatus[1]);}return;}
 if(id>=sourceStatus[0]){return;}if(id==0u){atomicStore(&status.relations,sourceStatus[0]);}
 partitions[id*2u]=0u;partitions[id*2u+1u]=0u;previousInterval=0u;
 let line=source[id];let a=params.clip*vec4f(line.a.xyz,1);let b=params.clip*vec4f(line.b.xyz,1);
 // The complete relation is retained. Visibility is an annotation on its
 // original homogeneous interval; clipped endpoints never become source ports.
 projected[id]=Relation(a,b,line.key,line.ports,line.coordinates,line.sourceChart);
 let da=array<f32,7>(a.x+a.w,a.w-a.x,a.y+a.w,a.w-a.y,a.z,a.w-a.z,a.w-.00001);
 let db=array<f32,7>(b.x+b.w,b.w-b.x,b.y+b.w,b.w-b.y,b.z,b.w-b.z,b.w-.00001);
 var lo=0.0;var hi=1.0;
 for(var plane=0u;plane<7u;plane++){
  if(da[plane]<0.0&&db[plane]<0.0){emit(id,0,1,2u);return;}
  if((da[plane]<0.0)!=(db[plane]<0.0)){let t=da[plane]/(da[plane]-db[plane]);if(da[plane]<0.0){lo=max(lo,t);}else{hi=min(hi,t);}}
 }
 if(hi<=lo){emit(id,0,1,2u);return;}emit(id,0,lo,2u);
 // Anchor the bounded probe lattice to an ORIGINAL finite endpoint. A
 // near-plane intersection is only a clipping limit; it is not a source
 // identity or a new sampling origin. Samples are fixed page distances from
 // that endpoint. Clipping computes an integer sample range; it never
 // changes the distance or phase of any surviving sample.
 let finiteA=a.w>=.00001;let finiteB=b.w>=.00001;
 var anchorT=lo;var endT=hi;
 if(finiteA){anchorT=0.0;endT=select(hi,1.0,finiteB);}
 else if(finiteB){anchorT=1.0;endT=lo;}
 let anchor=screen(mix(a,b,anchorT)).xy;
 let derivative=(b.xy*a.w-a.xy*b.w)/max(abs(a.w*a.w),1e-20)*vec2f(.5,-.5)*params.extent.xy;
 let directed=select(derivative,-derivative,anchorT==1.0);
 let direction=select(vec2f(1.0,0.0),normalize(directed),length(directed)>1e-6);
 let pageDiagonal=length(vec2f(params.extent.xy));
 // A finite edge has a fixed whole-edge length. A near-plane crossing has no
 // finite original far endpoint, so its bounded lattice uses the page
 // diagonal; the clipped interval only selects which of those fixed samples
 // participate. Both cases are independent of lo/hi rephasing.
 let projectedLength=select(pageDiagonal,distance(anchor,screen(mix(a,b,1.0-anchorT)).xy),finiteA&&finiteB);
 let pitch=max(2.0,pageDiagonal/2048.0);
 let sampleCount=u32(clamp(ceil(projectedLength/max(pitch,1e-20)),1.0,2048.0));
 let maxSample=sampleCount-1u;
 let loPage=screen(mix(a,b,lo)).xy;let hiPage=screen(mix(a,b,hi)).xy;
 let loDistance=dot(loPage-anchor,direction);let hiDistance=dot(hiPage-anchor,direction);
 let firstDistance=max(0.0,min(loDistance,hiDistance));let lastDistance=min(projectedLength,max(loDistance,hiDistance));
 let firstSample=u32(clamp(ceil(firstDistance/max(pitch,1e-20)),0.0,f32(maxSample)));
 let lastSample=u32(clamp(floor(lastDistance/max(pitch,1e-20)),f32(firstSample),f32(maxSample)));
 let anchorW=mix(a,b,anchorT).w;let endW=mix(a,b,endT).w;
 let reverse=endT<anchorT;let gradient=vec2f(line.a.w,line.b.w);
 var start=select(lo,hi,reverse);var prior=classify(mix(a,b,start),line,gradient);
 for(var sample=firstSample;sample<=lastSample;sample++){
  // Each sample is exactly sample*pitch from the original finite anchor.
  // For finite endpoints this is the required perspective inverse
  // t=lambda*wa/((1-lambda)*wb+lambda*wa); crossing edges invert the same
  // original homogeneous line at the page point directly.
  let distance=f32(sample)*pitch;var t=0.0;
  if(finiteA&&finiteB){let lambda=distance/max(projectedLength,1e-20);let segmentT=edgeTFromScreenLambda(lambda,anchorW,endW);t=mix(anchorT,endT,segmentT);}
  else{t=edgeTFromPagePoint(anchor+direction*distance,a,b);}
  if(t<lo||t>hi){continue;}
  let show=classify(mix(a,b,t),line,gradient);
  if(any(show!=prior)){
   let boundaryDistance=max(0.0,distance-.5*pitch);var boundary=0.0;
   if(finiteA&&finiteB){let boundaryLambda=boundaryDistance/max(projectedLength,1e-20);let boundarySegment=edgeTFromScreenLambda(boundaryLambda,anchorW,endW);boundary=clamp(mix(anchorT,endT,boundarySegment),lo,hi);}
   else{boundary=clamp(edgeTFromPagePoint(anchor+direction*boundaryDistance,a,b),lo,hi);}
   if(reverse){emit(id,boundary,start,prior.x);}else{emit(id,start,boundary,prior.x);}
   start=boundary;prior=show;
  }
 }
 if(reverse){emit(id,lo,start,prior.x);}else{emit(id,start,hi,prior.x);}emit(id,hi,1,2u);
}
`;
/** Exported for the source-level/Naga gate. Runtime pipeline creation still
 * isolates the requested entry point from this same shader string; no second
 * visibility implementation is introduced for tests. */
export const RELATION_VISIBILITY_SHADER=SHADER;
/** Complete projected source relations plus a partitions of visible, occluded
 * and outside-view intervals. No reconnection or feature selection occurs here. */
interface PreparedRelationVisibility{
 readonly plans:ReadonlyMap<'main',ComputePipelinePlan>;
 readonly pipelines:ReadonlyMap<'main',GPUComputePipeline>;
}
function relationVisibilityPipelinePlans(device:GPUDevice,multisampled:boolean):ReadonlyMap<'main',ComputePipelinePlan>{
 const code=multisampled?SHADER:SHADER.replace('texture_depth_multisampled_2d','texture_depth_2d').replace('for(var sample=0u;sample<textureNumSamples(depth);sample++){z=max(z,textureLoad(depth,pixel,i32(sample)));}','z=textureLoad(depth,pixel,0);');
 const layout=device.createBindGroupLayout({entries:[
  {binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:'uniform'}},
  ...[1,2].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:'read-only-storage' as const}})),
  {binding:3,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:'depth',multisampled}},
  ...[4,5,6].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:'storage' as const}})),
  {binding:7,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:'unfilterable-float'}},
  {binding:8,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:'uint'}},
  {binding:9,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:'float'}},
  {binding:10,visibility:GPUShaderStage.COMPUTE,buffer:{type:'storage'}},
 ]});
 const pipelineLayout=device.createPipelineLayout({bindGroupLayouts:[layout]});
 return new Map([['main',()=>{const module=device.createShaderModule({label:'MAIN source interval visibility main',code:isolateComputeEntry(code,'main')});
  return {layout:pipelineLayout,compute:{module,entryPoint:'main'}};}]]);
}
export class RelationVisibility {
  lines:GPUBuffer;intervals:GPUBuffer;readonly status:GPUBuffer;
  partitions:GPUBuffer;
  private readonly pipeline:GPUComputePipeline;
  private readonly fallback:GPUTexture[];
  private pending:GPUBuffer[]=[];
  private retired:GPUBuffer[]=[];
  private snapshotPipeline?:GPUComputePipeline;
  private readonly readbacks=new Set<CountedRelationReadback>();
  static async create(device:GPUDevice,capacity=32768,intervalCapacity=131072,multisampled=true,progress?:(fraction:number)=>void):Promise<RelationVisibility>{
   const plans=relationVisibilityPipelinePlans(device,multisampled),pipelines=await prepareComputePipelines(device,plans,progress);
   return new RelationVisibility(device,capacity,intervalCapacity,multisampled,{plans,pipelines});
  }
  constructor(private readonly device:GPUDevice,public capacity=32768,public intervalCapacity=131072,multisampled=true,prepared?:PreparedRelationVisibility){
    this.lines=device.createBuffer({size:capacity*RELATION_WORDS*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
    this.intervals=device.createBuffer({size:intervalCapacity*16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
    this.partitions=device.createBuffer({label:'MAIN GPU source partitions incidence',size:(capacity+intervalCapacity)*8,usage:GPUBufferUsage.STORAGE});
    this.status=device.createBuffer({size:16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
    this.fallback=['r32float','rg32uint','rgba16float'].map(format=>device.createTexture({size:[1,1],format:format as GPUTextureFormat,usage:GPUTextureUsage.TEXTURE_BINDING}));
    const plans=prepared?.plans??relationVisibilityPipelinePlans(device,multisampled),pipelines=prepared?.pipelines??compileComputePipelines(device,plans);
    this.pipeline=pipelines.get('main')!;
  }
  /** Exact exhausted counts come from the SAME submitted source pose. Buffer
   * growth is not a visibility/geometry quota and cannot truncate a range. */
  ensureCapacity(relations:number,intervals:number):void{
    if(![relations,intervals].every(n=>Number.isSafeInteger(n)&&n>=0))throw Error('Invalid complete visibility extent');
    const maximum=Math.min(this.device.limits.maxBufferSize,this.device.limits.maxStorageBufferBindingSize);
    const before=this.capacity+this.intervalCapacity;
    for(const [kind,count,stride]of [['lines',relations,RELATION_WORDS*4],['intervals',intervals,16]] as const){
      const prior=kind==='lines'?this.capacity:this.intervalCapacity;if(count<=prior)continue;
      const capacity=Math.ceil(count/256)*256,size=capacity*stride;
      if(size>maximum)throw Error(`Complete ${kind} range (${count}) needs device-sized chunks`);
      const next=this.device.createBuffer({label:`MAIN complete counted ${kind}`,size,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
      this.retired.push(this[kind]);this[kind]=next;if(kind==='lines')this.capacity=capacity;else this.intervalCapacity=capacity;
    }
    if(before!==this.capacity+this.intervalCapacity){const size=(this.capacity+this.intervalCapacity)*8;if(size>maximum)throw Error('Source partitions incidence requires device-sized chunks');this.retired.push(this.partitions);this.partitions=this.device.createBuffer({label:'MAIN GPU source partitions incidence',size,usage:GPUBufferUsage.STORAGE});}
  }
  encode(encoder:GPUCommandEncoder,source:SurfaceRelations,depth:GPUTextureView,clip:Float32Array,width:number,height:number,field?:SurfaceField):void{
    if(source.lineCapacity>this.capacity)throw Error('Relation visibility capacity is smaller than its complete source graph');
    const groups=Math.max(1,Math.ceil(source.lineCapacity/64)),columns=Math.min(groups,this.device.limits.maxComputeWorkgroupsPerDimension);
    const params=this.device.createBuffer({size:96,usage:GPUBufferUsage.UNIFORM,mappedAtCreation:true});const floats=new Float32Array(params.getMappedRange());
    floats.set(clip);floats.set([width,height,0,0],16);new Uint32Array(floats.buffer).set([this.capacity,this.intervalCapacity,Number(!!field),columns],20);params.unmap();this.pending.push(params);encoder.clearBuffer(this.status);
    const buffers=[params,source.lines,source.status,undefined,this.lines,this.intervals,this.status];
    const entries:GPUBindGroupEntry[]=buffers.flatMap((buffer,binding)=>buffer?[{binding,resource:{buffer}}]:[]);entries.push({binding:3,resource:depth});
    [field?.textures[0]??this.fallback[0]!,field?.textures[3]??this.fallback[1]!,field?.textures[4]??this.fallback[2]!].forEach((texture,i)=>entries.push({binding:7+i,resource:texture.createView()}));
    entries.push({binding:10,resource:{buffer:this.partitions}});
    const group=this.device.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries});const pass=encoder.beginComputePass({label:'MAIN source visibility intervals'});
    pass.setPipeline(this.pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(columns,Math.ceil(groups/columns));pass.end();
  }
  /** Snapshot THIS camera before shared visibility storage is reused. Only
   * the 16-byte count header is initially read back; the GPU retains exact
   * emitted records for a later count-sized host-encoder copy. */
  readback(encoder:GPUCommandEncoder):ProjectedRelationReadback{
    const pipeline=this.snapshotPipeline??=this.device.createComputePipeline({layout:'auto',compute:{module:this.device.createShaderModule({label:'MAIN emitted relation snapshot',code:RELATION_SNAPSHOT_SHADER}),entryPoint:'main'}});
    const readback=new CountedRelationReadback(this.device,this.capacity,this.intervalCapacity,()=>this.readbacks.delete(readback));this.readbacks.add(readback);
    const groups=Math.ceil(Math.max(this.capacity,this.intervalCapacity)/64),columns=Math.min(groups,this.device.limits.maxComputeWorkgroupsPerDimension);
    const params=this.device.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM,mappedAtCreation:true});new Uint32Array(params.getMappedRange()).set([this.capacity,this.intervalCapacity,columns,0]);params.unmap();this.pending.push(params);
    const buffers=[params,this.status,this.lines,this.intervals,readback.lines,readback.intervals];
    const group=this.device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:buffers.map((buffer,binding)=>({binding,resource:{buffer}}))});
    const pass=encoder.beginComputePass({label:'MAIN retain emitted same-pose relations'});pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(columns,Math.ceil(groups/columns));pass.end();
    readback.encodeHeader(encoder,this.status);return readback;
  }
  encodeReadbacks(encoder:GPUCommandEncoder):void{for(const readback of this.readbacks)readback.encode(encoder);}
  submitted():void{for(const readback of this.readbacks)readback.submitted();for(const buffer of [...this.pending,...this.retired])buffer.destroy();this.pending=[];this.retired=[];}
  dispose():void{for(const readback of [...this.readbacks])readback.dispose();this.submitted();this.lines.destroy();this.intervals.destroy();this.partitions.destroy();this.status.destroy();this.fallback.forEach(texture=>texture.destroy());}
}
