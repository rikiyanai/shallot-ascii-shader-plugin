import type {RelationVisibility} from './relation-visibility.ts';
import type {SurfaceField} from './surface-fields.ts';
import {D10_LINE_CLASSES} from './d10-line-classes.ts';
import {RELATION_RECORD_WGSL} from './source-coordinates.ts';
import {compileComputePipelines,prepareComputePipelines,type ComputePipelinePlan,type ComputePipelineProgress} from './gpu-pipeline-preparation.ts';
import {isolateComputeEntry} from './gpu-compute-program.ts';
const SHADER=/* wgsl */`
struct Params {extent:vec4f,rect:vec4f}
${RELATION_RECORD_WGSL}
struct Interval {relation:u32,lo:f32,hi:f32,cause:u32}
@group(0) @binding(0) var<uniform> params:Params;
@group(0) @binding(1) var<storage,read> relations:array<Relation>;
@group(0) @binding(2) var<storage,read> intervals:array<Interval>;
@group(0) @binding(3) var<storage,read> status:array<u32>;
@group(0) @binding(4) var<storage,read_write> indirect:array<u32>;
struct Vertex {@builtin(position) position:vec4f,@location(0) @interpolate(flat) kind:u32}
@compute @workgroup_size(1) fn prepare(){indirect[0]=2u;indirect[1]=select(min(status[0],u32(params.extent.z)),0u,status[1]!=0u);indirect[2]=0u;indirect[3]=0u;}
@vertex fn vertex(@builtin(vertex_index) vertex:u32,@builtin(instance_index) instance:u32)->Vertex{
 let interval=intervals[instance];let line=relations[interval.relation];let p=mix(line.a,line.b,select(interval.lo,interval.hi,vertex==1u));
 let xy=(p.xy/p.w*vec2f(.5,-.5)+vec2f(.5))*params.rect.zw+params.rect.xy;
 let ndc=xy/params.extent.xy*vec2f(2,-2)+vec2f(-1,1);
 // FIXED CLASS ORDER: intervals arrive in the visibility pass's append (thread)
 // order. Draw 0 (extent.w 0) takes the relation intervals, draw 1 the
 // suggestive ones over them, so a texel two classes share is always the
 // suggestive line, whatever order the threads appended them in.
 let kind=line.key.z&0x1fffffffu;let draw=select(0u,1u,kind==3u);
 return Vertex(vec4f(select(vec2f(2),ndc,(interval.cause&3u)==0u&&draw==u32(params.extent.w)),0,1),kind);
}
// Retain the full D10 field below. Source-surface interior turns use its
// suggestive-contour colour; other source intervals remain cyan.
@fragment fn fragment(v:Vertex)->@location(0) vec4f{return select(vec4f(0.0,.62,.9,1),vec4f(.85,.05,.55,1),v.kind==3u);}
`;
/** D10 line classes from the frame's own fields (depth, world normal, owner).
 * Image-space, per pixel, same pose as the text:
 *  silhouette / depth step  near side of a depth discontinuity (second
 *                           difference of reverse-Z depth; zero on planes)
 *  occluding contour        n·v = 0 on the visible side
 *  crease                   normal turn > 40° inside one owner
 *  suggestive contour       valley of n·v (DeCarlo et al. 2003 image-space
 *                           form of kr = 0 with D_w kr > 0; Hatch Studio
 *                           sugThresh .044)
 *  contact / opening        owner change at continuous depth
 */
export const D10_CONSTRUCTION_MASK_SHADER=/* wgsl */`
struct FeatureParams {worldFromClip:mat4x4f,rect:vec4f,extent:vec4f,light:vec4f}
@group(0) @binding(0) var<uniform> fp:FeatureParams;
@group(0) @binding(1) var depthField:texture_2d<f32>;
@group(0) @binding(2) var normalField:texture_2d<f32>;
@group(0) @binding(3) var ownerField:texture_2d<u32>;
${D10_LINE_CLASSES}
struct Sample {d:f32,n:vec3f,o:vec2u,ndv:f32,on:bool}
fn octDecode(e:vec2f)->vec3f{
 var n=vec3f(e,1.0-abs(e.x)-abs(e.y));
 if(n.z<0.0){n=vec3f((vec2f(1.0)-abs(n.yx))*select(vec2f(-1.0),vec2f(1.0),n.xy>=vec2f(0.0)),n.z);}
 return normalize(n);
}
fn unproject(q:vec2f,d:f32)->vec3f{
 let clip=vec4f(q/fp.extent.xy*vec2f(2,-2)+vec2f(-1,1),d,1);
 let p=fp.worldFromClip*clip;return p.xyz/p.w;
}
fn at(q:vec2i)->Sample{
 let size=vec2i(fp.extent.xy);
 if(any(q<vec2i(0))||any(q>=size)){return Sample(0.0,vec3f(0),vec2u(0),0.0,false);}
 let o=textureLoad(ownerField,q,0).xy;let d=textureLoad(depthField,q,0).x;
 if((o.x|o.y)==0u||d<=0.0){return Sample(0.0,vec3f(0),vec2u(0),0.0,false);}
 let raw=octDecode(textureLoad(normalField,q,0).xy);
 let c=vec2f(q)+vec2f(.5);
 // Reverse-Z: depth 1 is the near plane, so this points at the camera for
 // perspective and orthographic views alike.
 let v=normalize(unproject(c,1.0)-unproject(c,d));
 // The visible side: two-sided and flipped-normal sources shade the same way.
 let s=dot(raw,v);let n=select(raw,-raw,s<0.0);
 return Sample(d,n,o,abs(s),true);
}
@vertex fn vertex(@builtin(vertex_index) vertex:u32)->@builtin(position) vec4f{return vec4f(f32(vertex&1u)*4-1,f32(vertex>>1u)*4-1,0,1);}
@fragment fn fragment(@builtin(position) p:vec4f)->@location(0) vec4f{
 let q=vec2i(floor(p.xy-fp.rect.xy));
 let c=at(q);if(!c.on){return vec4f(0);}
 let classes=d10Classes(q,fp.extent.xy,fp.worldFromClip);
 if((classes&1u)!=0u){return vec4f(.05,.05,.08,1);}
 if((classes&2u)!=0u){return vec4f(.12,.22,.62,1);}
 if((classes&4u)!=0u){return vec4f(0,.55,.25,1);}
 if((classes&8u)!=0u){return vec4f(.95,.45,0,1);}
 if((classes&16u)!=0u){return vec4f(.85,.05,.55,1);}
 // Region value: n·L toward the scene's brightest directional light (camera
 // light when the scene has none) in four bands with fixed thresholds. Alpha
 // .5 marks a tone pixel for the composite.
 let P=unproject(vec2f(q)+vec2f(.5),c.d);
 let L=select(normalize(unproject(vec2f(q)+vec2f(.5),1.0)-P),normalize(fp.light.xyz),fp.light.w>0.0);
 let nl=max(dot(c.n,L),0.0);
 let band=select(select(select(.45,.62,nl>=.15),.8,nl>=.4),1.0,nl>=.7);
 return vec4f(vec3f(band),.5);
}
`;








interface PreparedConstructionMask{
 readonly prepare:GPUComputePipeline;
 readonly lines:GPURenderPipeline;readonly features:GPURenderPipeline;readonly featureLayout:GPUBindGroupLayout;
}
const plansOf=(device:GPUDevice):ReadonlyMap<'prepare',ComputePipelinePlan>=>new Map([['prepare',()=>{
 const module=device.createShaderModule({label:'MAIN construction mask intervals prepare',code:isolateComputeEntry(SHADER,'prepare')});
 return {layout:'auto',compute:{module,entryPoint:'prepare'}};}]]);
const linesDescriptor=(device:GPUDevice):GPURenderPipelineDescriptor=>{
 const module=device.createShaderModule({label:'MAIN construction mask intervals',code:SHADER});
 return {label:'MAIN construction mask intervals',layout:'auto',vertex:{module,entryPoint:'vertex'},fragment:{module,entryPoint:'fragment',targets:[{format:'rgba8unorm'}]},primitive:{topology:'line-list'}};
};
const featureLayoutOf=(device:GPUDevice):GPUBindGroupLayout=>device.createBindGroupLayout({label:'MAIN construction mask fields',entries:[
 {binding:0,visibility:GPUShaderStage.FRAGMENT,buffer:{type:'uniform'}},
 {binding:1,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:'unfilterable-float'}},
 {binding:2,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:'unfilterable-float'}},
 {binding:3,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:'uint'}}]});
const featuresDescriptor=(device:GPUDevice,layout:GPUBindGroupLayout):GPURenderPipelineDescriptor=>{
 const module=device.createShaderModule({label:'MAIN construction mask line classes',code:D10_CONSTRUCTION_MASK_SHADER});
 return {label:'MAIN construction mask line classes',layout:device.createPipelineLayout({bindGroupLayouts:[layout]}),vertex:{module,entryPoint:'vertex'},fragment:{module,entryPoint:'fragment',targets:[{format:'rgba8unorm'}]}};
};
const maskProgramSlots=new WeakMap<GPUDevice,{pending?:Promise<PreparedConstructionMask>;ready?:PreparedConstructionMask;failure?:unknown}>();
async function prepareMaskPrograms(device:GPUDevice,progress?:ComputePipelineProgress):Promise<PreparedConstructionMask>{
 progress?.(0,'construction-mask');let slot=maskProgramSlots.get(device);
 if(!slot){slot={};maskProgramSlots.set(device,slot);}
 if(slot.ready){progress?.(1);return slot.ready;}
 if('failure' in slot)throw slot.failure;
 if(!slot.pending){
  let pending!:Promise<PreparedConstructionMask>;
  pending=(async()=>{
   try{
    const pipelines=await prepareComputePipelines(device,plansOf(device));
    const lines=await device.createRenderPipelineAsync(linesDescriptor(device));
    const featureLayout=featureLayoutOf(device),features=await device.createRenderPipelineAsync(featuresDescriptor(device,featureLayout));
    const prepared={prepare:pipelines.get('prepare')!,lines,features,featureLayout};slot!.ready=prepared;return prepared;
   }catch(error){slot!.failure=error;throw error;}
   finally{if(slot!.pending===pending)delete slot!.pending;}
  })();
  slot.pending=pending;
 }
 const prepared=await slot.pending;progress?.(1);return prepared;
}
export class ConstructionMask {
 private readonly prepare:GPUComputePipeline;private readonly lines:GPURenderPipeline;
 private readonly features:GPURenderPipeline;private readonly featureLayout:GPUBindGroupLayout;
 private readonly indirect:GPUBuffer;private pending:GPUBuffer[]=[];
 /** Startup factory: every program is first-frame tone input, so all of them
  * compile here, asynchronously and one after another; none is deferred. */
 static async create(device:GPUDevice,progress?:ComputePipelineProgress):Promise<ConstructionMask>{
  const prepared=await prepareMaskPrograms(device,progress);return new ConstructionMask(device,prepared);
 }
 /** Without `prepared`: the explicit synchronous inspection adapter. */
 constructor(private readonly device:GPUDevice,prepared?:PreparedConstructionMask){
  const pipelines=prepared?undefined:compileComputePipelines(device,plansOf(device));
  this.prepare=prepared?.prepare??pipelines!.get('prepare')!;
  this.lines=prepared?.lines??device.createRenderPipeline(linesDescriptor(device));
  this.featureLayout=prepared?.featureLayout??featureLayoutOf(device);
  this.features=prepared?.features??device.createRenderPipeline(featuresDescriptor(device,this.featureLayout));
  this.indirect=device.createBuffer({size:16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.INDIRECT});
 }
 /** `light`: unit direction toward the scene's brightest directional light, or undefined. */
 encode(encoder:GPUCommandEncoder,graph:RelationVisibility,target:GPUTexture,width:number,height:number,rect:{x:number;y:number;width:number;height:number},clear:boolean,field?:SurfaceField,light?:readonly number[]):void{
  // The target may be smaller than the page (the glyph-grid construction
  // image): the page rect in target texels, for the scissors and the D10 read.
  const sx=target.width/width,sy=target.height/height;
  const scissor=[Math.floor(rect.x*sx),Math.floor(rect.y*sy),Math.min(target.width,Math.ceil(rect.width*sx)),Math.min(target.height,Math.ceil(rect.height*sy))] as const;
  if(field){
   const fp=this.device.createBuffer({size:112,usage:GPUBufferUsage.UNIFORM,mappedAtCreation:true});
   const f=new Float32Array(fp.getMappedRange());f.set(field.worldFromClip);f.set([scissor[0],scissor[1],scissor[2],scissor[3],field.width,field.height,0,0,...(light?[light[0]!,light[1]!,light[2]!,1]:[0,0,0,0])],16);fp.unmap();this.pending.push(fp);
   const group=this.device.createBindGroup({layout:this.featureLayout,entries:[{binding:0,resource:{buffer:fp}},{binding:1,resource:field.textures[0]!.createView()},{binding:2,resource:field.textures[1]!.createView()},{binding:3,resource:field.textures[3]!.createView()}]});
   const pass=encoder.beginRenderPass({label:'MAIN construction mask line classes',colorAttachments:[{view:target.createView(),loadOp:clear?'clear':'load',storeOp:'store',clearValue:{r:0,g:0,b:0,a:0}}]});
   pass.setScissorRect(...scissor);pass.setPipeline(this.features);pass.setBindGroup(0,group);pass.draw(3);pass.end();
   clear=false;
  }
  const params=this.device.createBuffer({size:32,usage:GPUBufferUsage.UNIFORM,mappedAtCreation:true});
  new Float32Array(params.getMappedRange()).set([width,height,graph.intervalCapacity,0,rect.x,rect.y,rect.width,rect.height]);params.unmap();this.pending.push(params);
  const prepareGroup=this.device.createBindGroup({layout:this.prepare.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:params}},{binding:3,resource:{buffer:graph.status}},{binding:4,resource:{buffer:this.indirect}}]});
  const compute=encoder.beginComputePass({label:'MAIN construction mask interval count'});compute.setPipeline(this.prepare);compute.setBindGroup(0,prepareGroup);compute.dispatchWorkgroups(1);compute.end();
  // Two draws in a fixed class order (WGSL vertex: extent.w selects the class).
  const second=this.device.createBuffer({size:32,usage:GPUBufferUsage.UNIFORM,mappedAtCreation:true});
  new Float32Array(second.getMappedRange()).set([width,height,graph.intervalCapacity,1,rect.x,rect.y,rect.width,rect.height]);second.unmap();this.pending.push(second);
  const groups=[params,second].map(buffer=>this.device.createBindGroup({layout:this.lines.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer}},{binding:1,resource:{buffer:graph.lines}},{binding:2,resource:{buffer:graph.intervals}}]}));
  const pass=encoder.beginRenderPass({label:'MAIN construction mask intervals',colorAttachments:[{view:target.createView(),loadOp:clear?'clear':'load',storeOp:'store',clearValue:{r:0,g:0,b:0,a:0}}]});
  pass.setScissorRect(...scissor);
  pass.setPipeline(this.lines);for(const group of groups){pass.setBindGroup(0,group);pass.drawIndirect(this.indirect,0);}pass.end();
 }
 submitted():void{for(const buffer of this.pending)buffer.destroy();this.pending=[];}
 dispose():void{this.submitted();this.indirect.destroy();}
}
