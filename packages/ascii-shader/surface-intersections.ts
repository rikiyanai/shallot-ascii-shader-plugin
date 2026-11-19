import {DeviceComputePrograms,type ComputePipelinePlan} from './gpu-pipeline-preparation.ts';
import {isolateComputeEntry} from './gpu-compute-program.ts';
import {storageBinding,sameStorageBinding,type SourceStorageBinding} from './source-storage.ts';
import {posedBatchIdentity,type PosedSourceBatch,type SurfaceGraphView} from './surface-relations.ts';
import {isGpuIndexedResident,type ResidentRelationSource} from './source-cache.ts';
import type {SurfaceField} from './surface-fields.ts';
import {sourceIncidenceReaders} from './source-incidence-residual.ts';
import {RELATION_RECORD_WGSL} from './source-coordinates.ts';
import {SubmissionExpiry} from './submission-expiry.ts';
import {FORM_DEPTH_WGSL,FORM_DEPTH_HEADER_ROW,FAR_STRIP_BELOW_ROWS,LOD_TIER_HOLD,formDepths} from './form-depths.ts';

// Rasterization supplies candidate pairs only. The emitted contact is the
// finite triangle/triangle intersection, with original source-edge ports.
// No pixel chain, inferred infinite plane, glyph search or CPU geometry walk.
const TYPES=/* wgsl */`
struct P {clip:mat4x4f,extent:vec4u,source:vec4u,range:vec4u,form:vec4u}
struct Tri {a:vec4f,b:vec4f,c:vec4f,key:vec4u,edges:vec4u}
`;
// Four exact u32 texels per triangle: nine position floats, domain/owner/face
// and four edge words. Homogeneous position w is always 1; key.w (the pack
// invocation's instance ordinal) has no reader. No coordinate is quantized.
// A texture atlas removes the single storage-buffer-binding size ceiling.
const TRI_ADDRESS=/* wgsl */`
fn triangleAddress(index:u32,word:u32)->vec2i{
 let width=textureDimensions(triangles).x;let at=index*4u+word;
 return vec2i(i32(at%width),i32(at/width));
}
`;
const TRI_READ=TRI_ADDRESS+/* wgsl */`
fn readTriangle(index:u32)->Tri{
 let a=textureLoad(triangles,triangleAddress(index,0u),0);
 let b=textureLoad(triangles,triangleAddress(index,1u),0);
 let c=textureLoad(triangles,triangleAddress(index,2u),0);
 let edges=textureLoad(triangles,triangleAddress(index,3u),0);
 return Tri(vec4f(bitcast<vec3f>(a.xyz),1),vec4f(bitcast<vec3f>(vec3u(a.w,b.xy)),1),
  vec4f(bitcast<vec3f>(vec3u(b.zw,c.x)),1),vec4u(c.yzw,0u),edges);
}
`;
const PACK=TYPES+/* wgsl */`
@group(0) @binding(0) var<uniform> p:P;
@group(0) @binding(1) var<storage,read> topology:array<u32>;
@group(0) @binding(2) var<storage,read> vertices:array<vec4u>;
@group(0) @binding(3) var<storage,read> attributes:array<vec4u>;
@group(0) @binding(4) var<storage,read> native:array<u32>;
@group(0) @binding(5) var triangles:texture_storage_2d<rgba32uint,write>;
@group(0) @binding(6) var<storage,read_write> args:array<u32>;
@group(0) @binding(7) var<storage,read> selected:array<u32>;
struct Status {count:atomic<u32>,overflow:atomic<u32>,nonmanifold:atomic<u32>,triangles:atomic<u32>}
@group(0) @binding(8) var<storage,read_write> status:Status;
@group(0) @binding(9) var<storage,read> drawArgs:array<u32>;
@group(0) @binding(10) var formDepths:texture_2d<u32>;
@group(0) @binding(11) var<uniform> formWorldFromClip:mat4x4f;
${FORM_DEPTH_WGSL}
// CONTACT SELECTION BEFORE CONTACTS (perf-impl-95). p.form: x bit 0 the frame's
// form table is bound, bit 1 this source draws the field; y one text row in
// field pixels (f32 bits); z, w the field extent. An owner with no texel of
// the field is unseen; an owner whose compact form is 遠景 (rows under
// FAR_STRIP_BELOW_ROWS with the hold, not cut, not spanning, receding view)
// draws no contact (TIER_SELECTION column 2). Either way its triangles take
// part in no contact. Anything unknown keeps its contacts.
fn contactSkipped(owner:vec2u)->bool{
 if(formParentSelected(owner)){return true;}
 if((p.form.x&3u)!=3u){return false;}
 let form=formExtent(owner);if(!form.found){return true;}
 let m=formWorldFromClip;let extent=vec2f(f32(p.form.z),f32(p.form.w));
 if(!(abs(m[0][3])+abs(m[1][3])+abs(m[2][3])>1e-6*abs(m[3][3])&&formOpen())){return false;}
 if(formCut(form.box,extent)||formSpansDepth(form.near,form.far,extent,m)){return false;}
 // perf-impl-107: the one held form level when the writer published it
 // (header texel w), so a form's contacts follow its relation level exactly.
 if(textureLoad(formDepths,vec2i(0,${FORM_DEPTH_HEADER_ROW}),0).w!=0u){return formHeldLod(owner,true)==2u;}
 return (form.bottom-form.top)/bitcast<f32>(p.form.y)<${FAR_STRIP_BELOW_ROWS*LOD_TIER_HOLD};
}
${sourceIncidenceReaders('topology')}
${TRI_ADDRESS}
// The frontier stores original batch-local indices. Its count precedes the
// list in the existing selection binding; native/unknown callers keep the
// original indirect traversal and the uniform ABI does not change.
fn instanceCount()->u32{if(p.range.w!=0u){return selected[p.range.w-1u];}return native[1];}
fn instanceAt(ordinal:u32)->u32{if(p.range.w!=0u){return selected[p.range.w+ordinal];}return ordinal;}
@compute @workgroup_size(1) fn prepare(){
 let count=select(native[0]/3u*instanceCount(),0u,native[5]!=0u);
 let base=atomicAdd(&status.triangles,count);let fits=base+count<=p.extent.z;
 if(!fits){atomicOr(&status.overflow,16u);}
 let groups=select(0u,(count+63u)/64u,fits);
 args[0]=min(groups,p.source.w);args[1]=max(1u,(groups+p.source.w-1u)/p.source.w);args[2]=1u;
 args[4]=select(0u,count*3u,fits);args[5]=1u;args[6]=base*3u;args[7]=0u;
}
@compute @workgroup_size(64) fn pack(@builtin(global_invocation_id) id:vec3u){
 let k=id.x+id.y*p.source.w*64u;let faces=native[0]/3u;if(k>=faces*instanceCount()||native[5]!=0u){return;}
 let instance=instanceAt(k/faces);let face=(native[2]-p.source.z)/3u+k%faces;
 var positions:array<vec4u,3>;var edges=vec4u(0);var slots:array<u32,3>;
 for(var i=0u;i<3u;i++){
  let h=sourceHalfedge(face*3u+i);let v=h.x;
  slots[i]=instance*p.source.y+u32(i32(v)+bitcast<i32>(native[3]));
  positions[i]=vertices[slots[i]];edges[i]=h.z;edges.w|=select(0u,1u,v>h.y)<<i;
 }
 let entity=positions[0].w;let domain=p.source.x|select(0u,0x80000000u,entity==0xffffffffu);
 let owner=select(entity,attributes[slots[0]].w,entity==0xffffffffu);
 let enabled=(p.extent.w==0u||selected[p.range.z+instance]!=0u)&&!contactSkipped(vec2u(domain,owner));
 // range.y: the host reserved this source's range (range.x), no prepare ran.
 let out=select(drawArgs[6]/3u,p.range.x,p.range.y!=0u)+k;
 textureStore(triangles,triangleAddress(out,0u),vec4u(positions[0].xyz,positions[1].x));
 textureStore(triangles,triangleAddress(out,1u),vec4u(positions[1].yz,positions[2].xy));
 textureStore(triangles,triangleAddress(out,2u),vec4u(positions[2].z,select(0u,domain,enabled),owner,face));
 textureStore(triangles,triangleAddress(out,3u),edges);
}
`;
const RASTER=TYPES+/* wgsl */`
@group(0) @binding(0) var<uniform> p:P;
@group(0) @binding(1) var triangles:texture_2d<u32>;
${TRI_READ}
struct V {@builtin(position) clip:vec4f,@location(0) @interpolate(flat) address:u32}
@vertex fn vertex(@builtin(vertex_index) index:u32)->V{
 let t=readTriangle(index/3u);let points=array<vec4f,3>(t.a,t.b,t.c);
 var clip=p.clip*points[index%3u];if(t.key.x==0u){clip=vec4f(2,2,2,1);}
 return V(clip,index/3u+1u);
}
@fragment fn fragment(v:V)->@location(0) u32{return v.address;}
`;
const INTERSECT=TYPES+/* wgsl */`
${RELATION_RECORD_WGSL}
struct Status {count:atomic<u32>,overflow:atomic<u32>,nonmanifold:atomic<u32>,reserved:atomic<u32>}
@group(0) @binding(0) var<uniform> p:P;
@group(0) @binding(1) var triangles:texture_2d<u32>;
${TRI_READ}
@group(0) @binding(2) var ids:texture_2d<u32>;
@group(0) @binding(3) var owners:texture_2d<u32>;
@group(0) @binding(4) var<storage,read_write> pairs:array<atomic<u32>>;
@group(0) @binding(5) var<storage,read_write> relations:array<Relation>;
@group(0) @binding(6) var<storage,read_write> status:Status;
fn pair(address:u32)->vec2u{
 let pixel=address/2u;let at=vec2i(i32(pixel%p.extent.x),i32(pixel/p.extent.x));
 let next=at+select(vec2i(1,0),vec2i(0,1),(address&1u)!=0u);
 if(any(next>=vec2i(p.extent.xy))){return vec2u(0);}
 let a=textureLoad(ids,at,0).x;let b=textureLoad(ids,next,0).x;
 if(a==0u||b==0u||a==b){return vec2u(0);}
 let ta=readTriangle(a-1u);let tb=readTriangle(b-1u);
 // The original fragment field owns alpha/discard and surface identity.
 // This separate triangle address raster never overrides those decisions.
 // The id raster may be finer than the owner field (cell-resolution field):
 // each id texel reads the field texel under its own centre.
 let fs=vec2f(textureDimensions(owners))/vec2f(p.extent.xy);
 let oa=textureLoad(owners,vec2i((vec2f(at)+vec2f(.5))*fs),0).xy&vec2u(0x9fffffffu,0xffffffffu);
 let ob=textureLoad(owners,vec2i((vec2f(next)+vec2f(.5))*fs),0).xy&vec2u(0x9fffffffu,0xffffffffu);
 if(any(oa!=ta.key.xy)||any(ob!=tb.key.xy)||all(ta.key.xy==tb.key.xy)){return vec2u(0);}
 return vec2u(min(a,b),max(a,b));
}
struct Cut {low:vec3f,high:vec3f,ports:vec2u,parameter:vec2f,valid:u32}
fn cut(t:Tri,origin:vec3f,normal:vec3f,axis:vec3f)->Cut{
 let points=array<vec3f,3>(t.a.xyz,t.b.xyz,t.c.xyz);var result=Cut(vec3f(0),vec3f(0),vec2u(0),vec2f(0),0u);
 var low=3.402823e38;var high=-3.402823e38;
 for(var i=0u;i<3u;i++){
  let a=points[i];let b=points[(i+1u)%3u];let da=dot(a-origin,normal);let db=dot(b-origin,normal);
  if((da<0.0&&db<0.0)||(da>0.0&&db>0.0)||da==db){continue;}
  let u=clamp(da/(da-db),0.0,1.0);let q=mix(a,b,u);let along=dot(q-origin,axis);
  // Canonical edge parameter uses original vertex order, independent of
  // the pose. Adjoining faces traverse their shared edge oppositely.
  let reverse=(t.edges.w&(1u<<i))!=0u;
  let parameter=select(u,1.0-u,reverse);
  if(along<low){low=along;result.low=q;result.ports.x=t.edges[i];result.parameter.x=parameter;}
  if(along>high){high=along;result.high=q;result.ports.y=t.edges[i];result.parameter.y=parameter;}
 }
 result.valid=select(0u,1u,high>low);return result;
}
@compute @workgroup_size(8,8) fn contacts(@builtin(global_invocation_id) id:vec3u){
 if((atomicLoad(&status.overflow)&16u)!=0u){return;}
 if(any(id.xy>=p.extent.xy)){return;}
 for(var direction=0u;direction<2u;direction++){
  let address=(id.y*p.extent.x+id.x)*2u+direction;let found=pair(address);if(found.x==0u){continue;}
  var hash=((found.x*0x9e3779b9u)^(found.y*0x85ebca6bu))&(p.extent.w-1u);var own=false;
  for(var probe=0u;probe<p.extent.w;probe++){
   let claim=atomicCompareExchangeWeak(&pairs[hash],0u,address+1u);
   if(claim.exchanged){own=true;break;}
   if(claim.old_value==0u){continue;}
   if(all(pair(claim.old_value-1u)==found)){break;}hash=(hash+1u)&(p.extent.w-1u);
  }
  if(!own){continue;}
  var a=readTriangle(found.x-1u);var b=readTriangle(found.y-1u);
  // Canonical pair orientation is independent of draw order and the pixel
  // that discovered it. Source/owner order is stable while the camera moves.
  if(a.key.x>b.key.x||(a.key.x==b.key.x&&a.key.y>b.key.y)){let tmp=a;a=b;b=tmp;}
  let na=cross(a.b.xyz-a.a.xyz,a.c.xyz-a.a.xyz);let nb=cross(b.b.xyz-b.a.xyz,b.c.xyz-b.a.xyz);
  let crossing=cross(na,nb);let magnitude=length(crossing);
  if(magnitude<=length(na)*length(nb)*1e-6){continue;}
  let axis=crossing/magnitude;
  let ca=cut(a,b.a.xyz,nb,axis);let cb=cut(b,a.a.xyz,na,axis);if(ca.valid==0u||cb.valid==0u){continue;}
  let origin=a.a.xyz;
  let loA=dot(ca.low-origin,axis);let hiA=dot(ca.high-origin,axis);
  let loB=dot(cb.low-origin,axis);let hiB=dot(cb.high-origin,axis);
  // Both cuts lie on the one intersection line, so equal positions along it
  // are one point: an original edge of EACH surface passes through it. Two
  // sources tessellated on common lines (a level sheet over the ground it
  // floods) end every contact segment that way. Such an end is named by
  // the edge pair, which both adjoining segments share; the f32 order of
  // the two cuts there is noise and must not pick the port or drop a face
  // pair that only touches at that point into the node. The slack is the
  // f32 resolution of the posed positions, not a screen or world weld.
  let slack=(length(origin)+max(hiA,hiB)-min(loA,loB))*8e-6;
  if(min(hiA,hiB)-max(loA,loB)<=slack){continue;}
  let tie=vec2<bool>(abs(loB-loA)<=slack,abs(hiB-hiA)<=slack);
  let fromB=select(vec2<bool>(loB>loA,hiB<hiA),vec2<bool>(false,false),tie);
  let low=select(ca.low,cb.low,fromB.x);let high=select(ca.high,cb.high,fromB.y);
  let ports=select(ca.ports,cb.ports|vec2u(0x80000000u),fromB);
  // A tied end carries the other surface's edge address in its parameter
  // slot (same f32 carriage as the face address) and its bit in w.
  let parameter=select(select(ca.parameter,cb.parameter,fromB),vec2f(cb.ports),tie);
  let tied=f32(select(0u,1u,tie.x)|select(0u,2u,tie.y));
  let x=p.clip*a.a;let y=p.clip*a.b;let z=p.clip*a.c;let plane=cross(y.xyz/y.w-x.xyz/x.w,z.xyz/z.w-x.xyz/x.w);
  let gradient=-plane.xy/select(1e-20,plane.z,abs(plane.z)>1e-20);
  let out=atomicAdd(&status.count,1u);if(out>=p.extent.z){atomicOr(&status.overflow,2u);continue;}
  relations[out]=Relation(vec4f(low,gradient.x),vec4f(high,gradient.y),vec4u(a.key.x&0x1fffffffu,a.key.y,4u|(a.key.x&0x80000000u),a.key.z),vec4u(ports,b.key.xy),vec4f(parameter,f32(b.key.z),tied),vec4f(0.0));
 }
}
`;

/** Shared by engine material draws and the physical-world adapter. Posed
 * scratch is copied before reuse; no rest-pose transform substitutes for it. */
type ContactBinding=SourceStorageBinding|{readonly view:GPUTextureView};
type RetainedContact={uniform:GPUBuffer;args:GPUBuffer;groups:Map<string,{buffers:ContactBinding[];group:GPUBindGroup}>;serial:number};
function sameContactBinding(a:ContactBinding,b:ContactBinding):boolean{
 return 'view' in a||'view' in b?'view' in a&&'view' in b&&a.view===b.view:sameStorageBinding(a,b);
}
/** A bind group reused while every buffer it names is the same object. */
function retainedGroup(device:GPUDevice,groups:Map<string,{buffers:ContactBinding[];group:GPUBindGroup}>,name:string,pipeline:GPUComputePipeline|GPURenderPipeline,buffers:ContactBinding[],bindings:readonly number[]=buffers.map((_,index)=>index)):GPUBindGroup{
 const held=groups.get(name);if(held&&held.buffers.length===buffers.length&&held.buffers.every((buffer,index)=>sameContactBinding(buffer,buffers[index]!)))return held.group;
 const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:buffers.map((buffer,index)=>({binding:bindings[index]!,resource:'view' in buffer?buffer.view:storageBinding(buffer)}))});
 groups.set(name,{buffers,group});return group;
}

const contactPrograms=new DeviceComputePrograms(device=>{
 const plans=new Map<string,ComputePipelinePlan>();
 for(const entryPoint of ['prepare','pack','contacts']){
  plans.set(entryPoint,()=>{const module=device.createShaderModule({label:'MAIN posed contact triangles '+entryPoint,code:isolateComputeEntry(entryPoint==='contacts'?INTERSECT:PACK,entryPoint)});
   return {label:'MAIN posed contact triangles '+entryPoint,layout:'auto',compute:{module,entryPoint}};});
 }
 return {resources:undefined,plans};
});
export async function prepareSurfaceIntersectionPrograms(device:GPUDevice,progress?:(fraction:number)=>void):Promise<void>{await contactPrograms.prepare(device,progress);}

export class SurfaceIntersections {
 private triangles?:GPUTexture;private triangleBinding?:{readonly view:GPUTextureView};private ids?:GPUTexture;private depth?:GPUTexture;private pairs?:GPUBuffer;
 private width=0;private height=0;private used=0;private field?:SurfaceField;private view?:SurfaceGraphView;
 private triangleCapacity=32768;private sourceStatus?:GPUBuffer;
 /** Triangles reserved by the host this frame (appendAll's host ranges). */
 private reserved=0;private cleared=true;private readonly reservedWord:GPUBuffer;
 private pending:GPUBuffer[]=[];private retired:(GPUBuffer|GPUTexture)[]=[];
 private readonly prepare:GPUComputePipeline;private readonly pack:GPUComputePipeline;private readonly contacts:GPUComputePipeline;
 private readonly raster:GPURenderPipeline[];
 constructor(private readonly device:GPUDevice){
  const ready=contactPrograms.get(device).pipelines;
  this.reservedWord=device.createBuffer({label:'MAIN host-reserved contact triangles',size:4,usage:GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
  this.prepare=ready.get('prepare')!;this.pack=ready.get('pack')!;this.contacts=ready.get('contacts')!;
  const raster=device.createShaderModule({code:RASTER});
  this.raster=(['back','none','front'] as const).map(cullMode=>device.createRenderPipeline({layout:'auto',vertex:{module:raster,entryPoint:'vertex'},fragment:{module:raster,entryPoint:'fragment',targets:[{format:'r32uint'}]},primitive:{cullMode},depthStencil:{format:'depth32float',depthWriteEnabled:true,depthCompare:'greater-equal'}}));
 }
 /** perf-impl-95: set by SurfaceRelations before appendAll. */
 fieldDomains?:ReadonlySet<number>;page?:readonly number[];private emptyForm?:GPUTexture;
 ensureCapacity(count:number):void{
  if(!Number.isSafeInteger(count)||count<0)throw Error('Invalid contact triangle count');
  const capacity=Math.ceil(Math.max(this.triangleCapacity,count)/256)*256;
  const texels=capacity*4,maximum=this.device.limits.maxTextureDimension2D;
  if(texels>maximum*maximum)throw Error('Counted contact triangles need device-sized texture pages');
  if(this.triangles&&capacity===this.triangleCapacity)return;
  if(this.triangles)this.retired.push(this.triangles);
  this.triangleCapacity=capacity;
  const width=Math.min(maximum,Math.max(4,2**Math.ceil(Math.log2(Math.sqrt(texels)))));
  this.triangles=this.device.createTexture({label:'MAIN counted posed contact triangles',size:[width,Math.ceil(texels/width)],format:'rgba32uint',usage:GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.TEXTURE_BINDING});
  this.triangleBinding={view:this.triangles.createView()};
 }
 /** `scale`: id raster columns per field column (CELL-RESOLUTION FIELD: the
  * physical frame's field is the glyph grid; contact seams are found on a
  * raster twice as fine across). */
 begin(encoder:GPUCommandEncoder,view:SurfaceGraphView,field:SurfaceField,status:GPUBuffer,scale=1):void{
  this.used=0;this.reserved=0;this.view=view;this.field=field;this.sourceStatus=status;this.ensureCapacity(0);
  const width=field.width*scale,height=field.height;
  if(this.width!==width||this.height!==height){
   for(const resource of [this.ids,this.depth,this.pairs])if(resource)this.retired.push(resource);
   this.width=width;this.height=height;
   this.ids=this.device.createTexture({size:[this.width,this.height],format:'r32uint',usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING});
   this.depth=this.device.createTexture({size:[this.width,this.height],format:'depth32float',usage:GPUTextureUsage.RENDER_ATTACHMENT});
   const slots=2**Math.ceil(Math.log2(Math.max(256,this.width*this.height*2)));
   this.pairs=this.device.createBuffer({size:slots*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
  }
  encoder.clearBuffer(this.pairs!);
  // Cleared by the first id draw pass's load op (appendAll), not a pass of its own.
  this.cleared=false;
 }
 private uniform(values:readonly number[],source:readonly number[]=[]):GPUBuffer{
  const buffer=this.device.createBuffer({size:128,usage:GPUBufferUsage.UNIFORM,mappedAtCreation:true});
  const data=new Float32Array(buffer.getMappedRange());data.set(this.view!.clipFromWorld);new Uint32Array(data.buffer).set(values,16);new Uint32Array(data.buffer).set(source,20);buffer.unmap();this.pending.push(buffer);return buffer;
 }
 append(encoder:GPUCommandEncoder,batch:PosedSourceBatch,source:ResidentRelationSource,domain:number):void{
  this.appendAll(encoder,[{batch,source,domain}]);
 }
 /** Every source of a frame: all packs in ONE compute pass, then all draws in
  * ONE render pass (frame cost, 2026-11-11; was a compute and a render pass
  * per source, ~910 passes per Y8 text frame). prepare reserves each
  * source's triangle range with an atomic on the shared counter and its draw
  * starts at that range, so a later source's pack never overwrites an
  * earlier one; the draws keep source order against the same depth test, so
  * the id/depth result is the one the per-source passes made. */
 appendAll(encoder:GPUCommandEncoder,items:readonly {batch:PosedSourceBatch;source:ResidentRelationSource;domain:number;extent?:{readonly indices:number;readonly instances:number}}[]):void{
  if(!this.field||!items.length)return;
  // HOST RESERVATION (frame cost, 2026-11-12). When every source states its
  // static draw extent, the host reserves each source's triangle range in
  // item order, exactly the ranges prepare's atomic made dispatch by
  // dispatch; pack is then one direct dispatch per source with no one-thread
  // prepare, and the id raster is one draw per run of one cull mode over
  // those contiguous ranges (primitive order is draw order). A frame whose
  // triangles exceed the buffer grows it here instead of failing for a retry.
  let host=items.every(item=>item.extent);const bases:number[]=[];let total=this.reserved;
  if(host){
   for(const item of items){bases.push(total);total+=Math.floor(item.extent!.indices/3)*(item.batch.source.visibleInstanceFrontier?.indices.length??item.extent!.instances);}
   if(total>this.triangleCapacity){try{this.ensureCapacity(total);}catch{host=false;}}
  }
  const draws:{pipeline:GPURenderPipeline;entry:RetainedContact;base:number;count:number}[]=[];
  const packing=encoder.beginComputePass({label:'MAIN contact triangles'});
  for(const {batch,source,domain}of items){
   // Native indirect counts own the atlas extent. Scratch/producer capacity
   // is not the number of posed faces in this draw. Overflow records the full
   // required count for replay; no partial contact set is published.
   this.used++;
   // RETAINED PER-SOURCE BINDINGS (frame cost, 2026-11-11), as in
   // surface-relations.ts: the uniform, indirect arguments and bind groups of
   // a source stay between frames and only the 96 uniform bytes are rewritten.
   const entry=this.retainedFor(source,batch);
   const index=draws.length,count=host?Math.floor(items[index]!.extent!.indices/3)*(batch.source.visibleInstanceFrontier?.indices.length??items[index]!.extent!.instances):0;
   const data=new Float32Array(32);data.set(this.view!.clipFromWorld);new Uint32Array(data.buffer).set([this.width,this.height,this.triangleCapacity,Number(!!batch.source.visibleInstances)],16);new Uint32Array(data.buffer).set([domain,batch.source.vertexCapacity,batch.source.indexBase??0,this.device.limits.maxComputeWorkgroupsPerDimension],20);
   new Uint32Array(data.buffer).set([host?bases[index]!:0,Number(host),batch.source.visibleInstanceOffset??0,batch.source.visibleInstanceFrontier?.offset??0],24);
   const page=this.page,early=!!page&&page[0]>0&&page[1]>0;
   if(early){new Uint32Array(data.buffer).set([1|(Number(!!this.fieldDomains?.has(domain))<<1),0,this.field!.width,this.field!.height],28);data[29]=17*this.field!.height/page![1];}
   this.device.queue.writeBuffer(entry.uniform,0,data);
   const formView=early?formDepths(this.device).view(this.field!):(this.emptyForm??=this.device.createTexture({label:'MAIN contact unselected form table',size:[1,1],format:'rgba32uint',usage:GPUTextureUsage.TEXTURE_BINDING})).createView();
   const formMatrix=early?formDepths(this.device).matrix(this.field!):entry.uniform;
   const indexed=isGpuIndexedResident(source)?source.gpu:undefined;
   const buffers=[entry.uniform,indexed?.incidence.binding??source.bufferBinding??source.buffer,batch.storage?.vertices??batch.vertices,batch.storage?.attributes??batch.attributes,batch.storage?.metadata??batch.metadata,this.triangleBinding!,entry.args,batch.source.visibleInstances??batch.metadata,this.sourceStatus!,entry.args,{view:formView},formMatrix] as ContactBinding[];
   for(const [pipeline,used,name]of [[this.prepare,[0,4,6,7,8],'prepare'],[this.pack,[0,1,2,3,4,5,7,9,10,11],'pack']] as const){
    if(host&&pipeline===this.prepare)continue;
    if(host&&!count)continue;
    packing.setPipeline(pipeline);packing.setBindGroup(0,retainedGroup(this.device,entry.groups,name,pipeline,used.map(binding=>buffers[binding]!),used));
    if(pipeline===this.prepare)packing.dispatchWorkgroups(1);
    else if(host){const groups=Math.ceil(count/64),limit=this.device.limits.maxComputeWorkgroupsPerDimension;packing.dispatchWorkgroups(Math.min(groups,limit),Math.ceil(groups/limit));}
    else packing.dispatchWorkgroupsIndirect(entry.args,0);
   }
   draws.push({pipeline:this.raster[batch.source.cullMode??0]!,entry,base:host?bases[index]!:0,count});
  }
  packing.end();
  const load:GPULoadOp=this.cleared?'load':'clear';this.cleared=true;
  const pass=encoder.beginRenderPass({label:'MAIN contact triangle ids',colorAttachments:[{view:this.ids!.createView(),loadOp:load,storeOp:'store'}],depthStencilAttachment:{view:this.depth!.createView(),depthClearValue:0,depthLoadOp:load,depthStoreOp:'store'}});
  if(host){
   // The status triangle word is what prepare's atomics left: every range.
   this.reserved=total;this.device.queue.writeBuffer(this.reservedWord,0,new Uint32Array([total]));
   for(let at=0;at<draws.length;){
    const pipeline=draws[at]!.pipeline,first=draws[at]!.base;let end=at,triangles=0;
    while(end<draws.length&&draws[end]!.pipeline===pipeline){triangles+=draws[end]!.count;end++;}
    if(triangles){const entry=draws[at]!.entry;pass.setPipeline(pipeline);pass.setBindGroup(0,retainedGroup(this.device,entry.groups,`draw ${this.raster.indexOf(pipeline)}`,pipeline,[entry.uniform,this.triangleBinding!]));pass.draw(triangles*3,1,first*3,0);}
    at=end;
   }
  }else for(const draw of draws){
   const name=`draw ${this.raster.indexOf(draw.pipeline)}`;
   pass.setPipeline(draw.pipeline);pass.setBindGroup(0,retainedGroup(this.device,draw.entry.groups,name,draw.pipeline,[draw.entry.uniform,this.triangleBinding!]));pass.drawIndirect(draw.entry.args,16);
  }
  pass.end();
  if(host)encoder.copyBufferToBuffer(this.reservedWord,0,this.sourceStatus!,12,4);
 }
 private serial=0;
 private readonly retained=new Map<ResidentRelationSource,Map<object,RetainedContact>>();
 private readonly expiry=new SubmissionExpiry<RetainedContact>();
 /** A source+batch's retained contact bindings; a second use within one
  * frame gets its own one-frame set (its uniform write would replace the
  * first use's before either ran). */
 private retainedFor(source:ResidentRelationSource,batch:PosedSourceBatch):RetainedContact{
  let byBatch=this.retained.get(source);if(!byBatch){byBatch=new Map();this.retained.set(source,byBatch);}
  const fresh=():RetainedContact=>({uniform:this.device.createBuffer({label:'MAIN contact source parameters',size:128,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),args:this.device.createBuffer({label:'MAIN contact source arguments',size:32,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.INDIRECT}),groups:new Map(),serial:this.serial});
  const key=posedBatchIdentity(batch);let entry=byBatch.get(key);
  if(entry&&entry.serial===this.serial){const once=fresh();this.pending.push(once.uniform,once.args);return once;}
  if(!entry){
   entry=fresh();byBatch.set(key,entry);const owned=entry;
   this.expiry.use(entry,()=>{if(byBatch.get(key)!==owned)return;this.retired.push(owned.uniform,owned.args);byBatch.delete(key);if(!byBatch.size)this.retained.delete(source);});
  }else this.expiry.use(entry);
  entry.serial=this.serial;return entry;
 }
 finish(encoder:GPUCommandEncoder,lines:GPUBuffer,status:GPUBuffer,capacity:number):void{
  if(!this.field||!this.triangles||!this.used)return;
  const uniform=this.uniform([this.width,this.height,capacity,this.pairs!.size/4]);
  const resources:GPUBindingResource[]=[{buffer:uniform},this.triangleBinding!.view,this.ids!.createView(),this.field.textures[3]!.createView(),{buffer:this.pairs!},{buffer:lines},{buffer:status}];
  const pass=encoder.beginComputePass({label:'MAIN finite source-surface contacts'});pass.setPipeline(this.contacts);pass.setBindGroup(0,this.device.createBindGroup({layout:this.contacts.getBindGroupLayout(0),entries:resources.map((resource,binding)=>({binding,resource}))}));pass.dispatchWorkgroups(Math.ceil(this.width/8),Math.ceil(this.height/8));pass.end();this.field=undefined;
 }
 submitted():void{
  // A serial is a submitted frame, not one appendAll call. The native
  // adapter appends each scratch slice separately: advancing per append
  // expired earlier sources within that SAME frame, scanned the retained
  // map once per source, and reused uniforms written by earlier slices.
  // Keep those slices/cameras distinct until their shared encoder settles.
  this.serial++;
  this.expiry.submitted();
  for(const resource of [...this.pending,...this.retired])resource.destroy();this.pending=[];this.retired=[];
 }
 dispose():void{this.expiry.clear();for(const byBatch of this.retained.values())for(const entry of byBatch.values())this.retired.push(entry.uniform,entry.args);this.retained.clear();this.submitted();for(const resource of [this.triangles,this.ids,this.depth,this.pairs,this.reservedWord])resource?.destroy();}
}
