import type {RelationVisibility} from './relation-visibility.ts';
import type {SurfaceField} from './surface-fields.ts';
import {D10_LINE_CLASSES} from './d10-line-classes.ts';
import {RELATION_RECORD_WGSL} from './source-coordinates.ts';
import {compileComputePipelines,prepareComputePipelines,type ComputePipelinePlan} from './gpu-pipeline-preparation.ts';
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
export const D10_UNDERDRAWING_SHADER=/* wgsl */`
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
export const UNDERDRAWING_COMPOSITE_SHADER=/* wgsl */`
@group(0) @binding(0) var native:texture_2d<f32>;
@group(0) @binding(1) var drawing:texture_2d<f32>;
struct PresentationVertex {@builtin(position) position:vec4f,@location(0) uv:vec2f}
@vertex fn vertex(@builtin(vertex_index) vertex:u32)->PresentationVertex{
 let xy=vec2f(f32(vertex&1u)*2,f32(vertex>>1u)*2);
 return PresentationVertex(vec4f(xy*2-1,0,1),vec2f(xy.x,1-xy.y));
}
fn quantize(l:f32)->f32{return select(select(select(select(.42,.58,l>=.2),.72,l>=.4),.85,l>=.6),1.0,l>=.8);}
// Value bands with thresholds fixed across frames (agreed design :34-37,
// toon :43-48). Surfaces: the n·L band from the line-class pass, mixed with
// the displayed frame's own value (cast shadow, fog, exposure) and quantized
// again. Sky and empty space: the displayed value alone (atmosphere).
// The value bands are the inspection backdrop: faint on white paper so the
// native glyph pass, drawn afterward, remains the foreground authority.

// line classes and the construction intervals (alpha 1) keep their own legend
// colour at full strength; from 2026-11-08 (449fe04) they were mixed to 20 %
// with the bands, a pale tint no reader could find under the glyphs.
// The original D10 texture and its line/band alpha are NOT modified.
@fragment fn fragment(p:PresentationVertex)->@location(0) vec4f{
 let base=textureLoad(native,vec2i(p.uv*vec2f(textureDimensions(native))),0);let ink=textureLoad(drawing,vec2i(p.uv*vec2f(textureDimensions(drawing))),0);
 let l=dot(base.rgb,vec3f(.2126,.7152,.0722));
 if(ink.a>.75){return vec4f(ink.rgb,1);}
 var guide=vec3f(quantize(l));
 if(ink.a>.25){guide=vec3f(quantize(mix(ink.r,l,.35)));}
 return vec4f(mix(vec3f(1),guide,.2),1);
}
@fragment fn nativeView(p:PresentationVertex)->@location(0) vec4f{
 let size=vec2i(textureDimensions(native));return textureLoad(native,clamp(vec2i(p.uv*vec2f(size)),vec2i(0),size-1),0);
}
`;
export const UNDERDRAWING_LEGEND='Grey bands: n·L toward the scene light with its shadow, fog and exposure, fixed thresholds. Black: silhouette / depth step. Blue: occluding contour (n·v = 0). Orange: crease. Magenta: suggestive contour. Green: contact. Cyan: construction intervals.';
interface PreparedUnderdrawing{
 readonly plans:ReadonlyMap<'prepare',ComputePipelinePlan>;
 readonly pipelines:ReadonlyMap<'prepare',GPUComputePipeline>;
}
function underdrawingPipelinePlans(device:GPUDevice):ReadonlyMap<'prepare',ComputePipelinePlan>{
 return new Map([['prepare',()=>{const module=device.createShaderModule({label:'MAIN underdrawing intervals prepare',code:isolateComputeEntry(SHADER,'prepare')});
  return {layout:'auto',compute:{module,entryPoint:'prepare'}};}]]);
}
/** The D10 underdrawing over the same-frame target: value bands from the
 * scene light, the line classes from the frame's fields, and the admitted
 * construction intervals. It never substitutes for the native text page. */
export class Underdrawing {
  private readonly prepare:GPUComputePipeline;private readonly lines:GPURenderPipeline;
  private readonly features:GPURenderPipeline;private readonly featureLayout:GPUBindGroupLayout;
  private readonly indirect:GPUBuffer;private readonly compositeModule:GPUShaderModule;
  private readonly composites=new Map<GPUTextureFormat,GPURenderPipeline>();
  private readonly snapshots=new Map<GPUTextureFormat,GPURenderPipeline>();
  private pending:GPUBuffer[]=[];
 static async create(device:GPUDevice,progress?:(fraction:number)=>void):Promise<Underdrawing>{
  const plans=underdrawingPipelinePlans(device),pipelines=await prepareComputePipelines(device,plans,progress);
  return new Underdrawing(device,{plans,pipelines});
 }
 constructor(private readonly device:GPUDevice,prepared?:PreparedUnderdrawing){
  const plans=prepared?.plans??underdrawingPipelinePlans(device),pipelines=prepared?.pipelines??compileComputePipelines(device,plans);
  this.prepare=pipelines.get('prepare')!;
    const module=device.createShaderModule({label:'MAIN underdrawing intervals',code:SHADER});
    this.lines=device.createRenderPipeline({layout:'auto',vertex:{module,entryPoint:'vertex'},fragment:{module,entryPoint:'fragment',targets:[{format:'rgba8unorm'}]},primitive:{topology:'line-list'}});
    this.featureLayout=device.createBindGroupLayout({label:'MAIN underdrawing fields',entries:[
      {binding:0,visibility:GPUShaderStage.FRAGMENT,buffer:{type:'uniform'}},
      {binding:1,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:'unfilterable-float'}},
      {binding:2,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:'unfilterable-float'}},
      {binding:3,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:'uint'}}]});
    const features=device.createShaderModule({label:'MAIN underdrawing line classes',code:D10_UNDERDRAWING_SHADER});
    this.features=device.createRenderPipeline({label:'MAIN underdrawing line classes',layout:device.createPipelineLayout({bindGroupLayouts:[this.featureLayout]}),vertex:{module:features,entryPoint:'vertex'},fragment:{module:features,entryPoint:'fragment',targets:[{format:'rgba8unorm'}]}});
    this.indirect=device.createBuffer({size:16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.INDIRECT});
    this.compositeModule=device.createShaderModule({label:'MAIN same-pose underdrawing composite',code:UNDERDRAWING_COMPOSITE_SHADER});
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
      const pass=encoder.beginRenderPass({label:'MAIN underdrawing line classes',colorAttachments:[{view:target.createView(),loadOp:clear?'clear':'load',storeOp:'store',clearValue:{r:0,g:0,b:0,a:0}}]});
      pass.setScissorRect(...scissor);pass.setPipeline(this.features);pass.setBindGroup(0,group);pass.draw(3);pass.end();
      clear=false;
    }
    const params=this.device.createBuffer({size:32,usage:GPUBufferUsage.UNIFORM,mappedAtCreation:true});
    new Float32Array(params.getMappedRange()).set([width,height,graph.intervalCapacity,0,rect.x,rect.y,rect.width,rect.height]);params.unmap();this.pending.push(params);
    const prepareGroup=this.device.createBindGroup({layout:this.prepare.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:params}},{binding:3,resource:{buffer:graph.status}},{binding:4,resource:{buffer:this.indirect}}]});
    const compute=encoder.beginComputePass({label:'MAIN underdrawing interval count'});compute.setPipeline(this.prepare);compute.setBindGroup(0,prepareGroup);compute.dispatchWorkgroups(1);compute.end();
    // Two draws in a fixed class order (WGSL vertex: extent.w selects the class).
    const second=this.device.createBuffer({size:32,usage:GPUBufferUsage.UNIFORM,mappedAtCreation:true});
    new Float32Array(second.getMappedRange()).set([width,height,graph.intervalCapacity,1,rect.x,rect.y,rect.width,rect.height]);second.unmap();this.pending.push(second);
    const groups=[params,second].map(buffer=>this.device.createBindGroup({layout:this.lines.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer}},{binding:1,resource:{buffer:graph.lines}},{binding:2,resource:{buffer:graph.intervals}}]}));
    const pass=encoder.beginRenderPass({label:'MAIN source relation underdrawing',colorAttachments:[{view:target.createView(),loadOp:clear?'clear':'load',storeOp:'store',clearValue:{r:0,g:0,b:0,a:0}}]});
    pass.setScissorRect(...scissor);
    pass.setPipeline(this.lines);for(const group of groups){pass.setBindGroup(0,group);pass.drawIndirect(this.indirect,0);}pass.end();
  }
  composite(encoder:GPUCommandEncoder,native:GPUTexture,drawing:GPUTexture,target:GPUTextureView,format:GPUTextureFormat):void{
    let pipeline=this.composites.get(format);if(!pipeline){pipeline=this.device.createRenderPipeline({layout:'auto',vertex:{module:this.compositeModule,entryPoint:'vertex'},fragment:{module:this.compositeModule,entryPoint:'fragment',targets:[{format}]}});this.composites.set(format,pipeline);}
    const group=this.device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:native.createView()},{binding:1,resource:drawing.createView()}]});
    const pass=encoder.beginRenderPass({label:'MAIN same-frame mesh/underdrawing comparison',colorAttachments:[{view:target,loadOp:'load',storeOp:'store'}]});pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.draw(3);pass.end();
  }
  snapshot(encoder:GPUCommandEncoder,native:GPUTexture,target:GPUTextureView,format:GPUTextureFormat):void{
    let pipeline=this.snapshots.get(format);if(!pipeline){pipeline=this.device.createRenderPipeline({layout:'auto',vertex:{module:this.compositeModule,entryPoint:'vertex'},fragment:{module:this.compositeModule,entryPoint:'nativeView',targets:[{format}]}});this.snapshots.set(format,pipeline);}
    const group=this.device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:native.createView()}]});
    const pass=encoder.beginRenderPass({label:'MAIN latched original mesh comparison',colorAttachments:[{view:target,loadOp:'load',storeOp:'store'}]});pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.draw(3);pass.end();
  }
  submitted():void{for(const buffer of this.pending)buffer.destroy();this.pending=[];}
  dispose():void{this.submitted();this.indirect.destroy();}
}
