import type {PhaseGeometryDraw} from '@dylanebert/shallot/rendering';
import {bindGroupSource,renderPipelineSource} from '@dylanebert/shallot/runtime';
import {fieldStage,type FieldStage} from './field-stage.ts';
import {formDepths} from './form-depths.ts';

export const FIELD_FORMATS:readonly GPUTextureFormat[]=['r32float','rg16float','rg32float','rg32uint','rgba16float'];
/** Optional attachment authority, supplied only by a producer whose actual
 * vertex stage is this source/model transform. Arbitrary material deformation
 * must not borrow an ECS/rest transform as if it posed the rendered surface. */
export interface SurfaceAttachment {
 readonly source:readonly [number,number];
 readonly worldFromSource:Float32Array;readonly sourceFromWorld:Float32Array;
}
export interface SurfaceField {
 readonly width:number;readonly height:number;
 /** Native fragment depth, oct normal, exact UV, exact source/owner, HDR RGBA. */
 readonly textures:readonly GPUTexture[];
 /** Primary raster owns native radiance; bitset word 0 is its format marker. */
 readonly primary?:true;readonly emissiveCoverage?:GPUBuffer;
 readonly worldFromClip:Float32Array;
 readonly attachments?:readonly SurfaceAttachment[];
 readonly sources:ReadonlyMap<number,{readonly kind:'entity'|'producer-row'|'mixed';readonly hasUV:boolean;readonly stage:PhaseGeometryDraw['stage']}>;
}
interface Compiled {pipeline:GPURenderPipeline;stage:FieldStage;groups:Map<number,WeakMap<GPUBindGroup,GPUBindGroup>>}
interface Resident {width:number;height:number;textures:GPUTexture[];depth:GPUTexture;users:number;primary:boolean;coverage?:GPUBuffer}

/** The same plugin observes native draws from standard materials, procedural
 * GPU meshes, and custom render phases. No example/asset name selects a path.
 * Fields preserve source identities and fragment behavior for the construction
 * stage. They are not themselves the published drawing.
 */
export class SurfaceFields {
 private readonly pipelines=new WeakMap<object,Compiled>();
 private readonly primaryPipelines=new WeakMap<object,Compiled>();
 private resident?:Resident;private retired:Resident[]=[];private uniforms:GPUBuffer[]=[];
 private readonly retained=new Map<readonly GPUTexture[],Resident>();
 private readonly producerDomains=new WeakMap<object,Map<string,number>>();
 private readonly identities=new WeakMap<object,number>();private nextIdentity=0;private nextDomain=0x40000000;
 constructor(private readonly device:GPUDevice){}
 begin(encoder:GPUCommandEncoder,width:number,height:number,worldFromClip:Float32Array,options:{primary?:boolean;clearValue?:GPUColor}={}):SurfaceField{
  const primary=options.primary===true;
  if(!this.resident||this.resident.width!==width||this.resident.height!==height||this.resident.primary!==primary||this.resident.users>0){
   if(this.resident)this.retired.push(this.resident);
   const textures=FIELD_FORMATS.map((format,i)=>this.device.createTexture({label:`MAIN native field ${i}`,size:[width,height],format,usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_SRC}));
   let coverage:GPUBuffer|undefined;
   if(primary){
    coverage=this.device.createBuffer({label:'MAIN primary emissive bitset',size:4+Math.ceil(width*height/32)*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,mappedAtCreation:true});
    new Uint32Array(coverage.getMappedRange())[0]=1;coverage.unmap();
   }
   this.resident={width,height,textures,primary,coverage,depth:this.device.createTexture({label:'MAIN field visibility',size:[width,height],format:'depth32float',usage:GPUTextureUsage.RENDER_ATTACHMENT}),users:0};
   this.retained.set(textures,this.resident);
  }
  const r=this.resident;
  if(r.coverage)encoder.clearBuffer(r.coverage,4);
  const clear=encoder.beginRenderPass({label:'MAIN clear current native fields',colorAttachments:r.textures.map((texture,i)=>({view:texture.createView(),loadOp:'clear' as const,storeOp:'store' as const,clearValue:i===4&&primary?options.clearValue??[0,0,0,0]:[0,0,0,0]})),depthStencilAttachment:{view:r.depth.createView(),depthLoadOp:'clear',depthStoreOp:'store',depthClearValue:0}});clear.end();
  return {width,height,textures:r.textures,primary:primary?true:undefined,emissiveCoverage:r.coverage,worldFromClip:new Float32Array(worldFromClip),sources:new Map()};
 }
 private identity(object:object):number{let id=this.identities.get(object);if(id===undefined){id=++this.nextIdentity;this.identities.set(object,id);}return id;}
 private domain(draw:PhaseGeometryDraw):number{
  const sources=this.producerDomains.get(draw.renderer)??new Map<string,number>();this.producerDomains.set(draw.renderer,sources);
  const args=draw.arguments.map(value=>typeof value==='number'?value:this.identity(value)).join('/');
  const key=[draw.stage,draw.stableId,draw.stablePart,this.identity((draw.recipe??draw.pipeline)!),draw.index?this.identity(draw.index.buffer):0,...draw.vertices.map(v=>`${v.slot}:${this.identity(v.buffer)}:${v.offset}`),args].join('/');
  let id=sources.get(key);if(id===undefined){id=++this.nextDomain;if(id>=0x60000000)throw Error('Native field source capacity exhausted');sources.set(key,id);}return id;
 }
 /** One ordered field pass owns the complete draw list. Reopening five
  * attachments for every mesh forced a full tile store/load per source. */
 appendAll(encoder:GPUCommandEncoder,field:SurfaceField,draws:Iterable<{draw:PhaseGeometryDraw;sourceDomain?:number}>):void{
  const resident=this.retained.get(field.textures);if(!resident)throw Error('Cannot append to a retired native field');
  const open=()=>encoder.beginRenderPass({label:'MAIN exact native fragment field',colorAttachments:field.textures.map(texture=>({view:texture.createView(),loadOp:'load' as const,storeOp:'store' as const})),depthStencilAttachment:{view:resident.depth.createView(),depthLoadOp:'load',depthStoreOp:'store'}});
  let pass:GPURenderPassEncoder|undefined;
  try{for(const {draw,sourceDomain}of draws){
  const key=draw.recipe??draw.pipeline;if(!key)throw Error('Native field draw has no pipeline source');
  // A screen-space pass (a fullscreen triangle or quad: a non-indexed draw
  // of at most six vertices, one instance) is an effect over the image, not
  // a surface. Recorded as a surface it relabels every pixel's owner and
  // depth (the fog composite erased the room from its own field).
  const screen=draw.renderer.geometryAdapter==='screen-effect';
  if(screen&&!field.primary)continue;
  if(draw.command==='bundles')throw Error('Native field render bundles require their recorded draw sources');
  const source=draw.recipe??renderPipelineSource(draw.pipeline!);if(!source?.fragmentCode||!source.descriptor.fragment)throw Error('Native field pipeline has no resolved fragment source');
  const pipelines=field.primary?this.primaryPipelines:this.pipelines;
  let compiled=pipelines.get(key);
  if(!compiled){
   // tone2: an additive draw (colour blend one/one) is an emissive effect, not a surface.
   const additive=source.descriptor.fragment.targets?.[0]?.blend?.color;const emissive=additive?.srcFactor==='one'&&additive?.dstFactor==='one';
   const stage=fieldStage(source.vertexCode,source.fragmentCode,source.descriptor.vertex.entryPoint,source.descriptor.fragment.entryPoint,emissive,field.primary===true);
   const vertex=this.device.createShaderModule({label:'MAIN native source vertex',code:stage.vertex}),fragment=this.device.createShaderModule({label:'MAIN native source fragment',code:stage.fragment});
   const depth=source.descriptor.depthStencil;
   const pipeline=this.device.createRenderPipeline({label:'MAIN native surface/field input',layout:'auto',vertex:{...source.descriptor.vertex,module:vertex,entryPoint:'mainFieldVertex'},fragment:{...source.descriptor.fragment,module:fragment,entryPoint:'mainFieldFragment',targets:FIELD_FORMATS.map((format,i)=>field.primary?
     i===4?{...Array.from(source.descriptor.fragment!.targets)[0]!,format}:
      {format,...(emissive||screen?{writeMask:0}:{})}:
     !emissive?{format}:i===4?{format,writeMask:GPUColorWrite.ALPHA,blend:{color:{srcFactor:'zero',dstFactor:'one',operation:'add'},alpha:{srcFactor:'one',dstFactor:'one',operation:'add'}}}:{format,writeMask:0})},primitive:source.descriptor.primitive,
    // Native equal-depth forward draws were preceded by the engine's prepass.
    // This independent field has its own visibility buffer, populated here.
    depthStencil:{format:'depth32float',depthCompare:screen&&!depth?'always':!depth||depth.depthCompare==='equal'?'greater-equal':depth.depthCompare,depthWriteEnabled:emissive||screen?false:!depth||depth.depthCompare==='equal'?true:depth.depthWriteEnabled,depthBias:depth?.depthBias,depthBiasSlopeScale:depth?.depthBiasSlopeScale,depthBiasClamp:depth?.depthBiasClamp},multisample:{count:1}});
   compiled={pipeline,stage,groups:new Map()};pipelines.set(key,compiled);
  }
  const domain=sourceDomain??this.domain(draw),stage=compiled.stage;
  (field.sources as Map<number,SurfaceField['sources'] extends ReadonlyMap<number,infer T>?T:never>).set(domain,{kind:stage.ownerKind,hasUV:stage.hasUV,stage:draw.stage});
  const params=this.device.createBuffer({label:'MAIN native field source/view',size:96,usage:GPUBufferUsage.UNIFORM,mappedAtCreation:true});
  const floats=new Float32Array(params.getMappedRange());floats.set(field.worldFromClip);floats.set([field.width,field.height,0,0],16);
  // Domain bit30 distinguishes raw producers; bits31/29 carry the actual
  // invocation's identity/UV contract. A shared mesh's last material cannot
  // relabel another material's fragments.
  // Producer identity comes from each actual vertex invocation. A standard
  // material can mix valid ECS eids with eid=0xffffffff producer rows.
  new Uint32Array(floats.buffer)[20]=(domain|(stage.hasUV?0x20000000:0))>>>0;params.unmap();this.uniforms.push(params);
  pass??=open();
  pass.setPipeline(compiled.pipeline);
  for(const native of draw.groups){
   if(native.index===stage.group)continue;
   const used=stage.resources.get(native.index);if(!used)continue;
   let cache=compiled.groups.get(native.index);if(!cache){cache=new WeakMap();compiled.groups.set(native.index,cache);}
   let group=native.offsets.length?undefined:cache.get(native.group);
   if(!group){
    const original=bindGroupSource(native.group);if(!original)throw Error('Native field source bind group is unavailable');
    const offsets=new Map<number,number>();
    if(native.offsets.length){
     if(!original.layout)throw Error('Native field dynamic offsets lack their original layout');
     const dynamic=Array.from(original.layout.entries).filter(e=>e.buffer?.hasDynamicOffset).sort((a,b)=>a.binding-b.binding);
     if(dynamic.length!==native.offsets.length)throw Error('Native field dynamic offset count changed');
     dynamic.forEach((entry,i)=>offsets.set(entry.binding,native.offsets[i]!));
    }
    const entries=Array.from(original.descriptor.entries).filter(entry=>used.has(entry.binding)).map(entry=>{
     const offset=offsets.get(entry.binding);if(offset===undefined)return entry;
     if(!('buffer'in entry.resource)||entry.resource.size===undefined)throw Error('Native field dynamic source needs an explicit range');
     return {...entry,resource:{...entry.resource,offset:(entry.resource.offset??0)+offset}};
    });
    group=this.device.createBindGroup({layout:compiled.pipeline.getBindGroupLayout(native.index),entries});if(!native.offsets.length)cache.set(native.group,group);
   }
   pass.setBindGroup(native.index,group);
  }
  pass.setBindGroup(stage.group,this.device.createBindGroup({layout:compiled.pipeline.getBindGroupLayout(stage.group),entries:[{binding:0,resource:{buffer:params}},...(stage.resources.get(stage.group)?.has(1)?[{binding:1,resource:{buffer:field.emissiveCoverage!}}]:[])]}));
  for(const v of draw.vertices)pass.setVertexBuffer(v.slot,v.buffer,v.offset,v.size);
  if(draw.index)pass.setIndexBuffer(draw.index.buffer,draw.index.format,draw.index.offset,draw.index.size);
  // A new pass used to supply these defaults implicitly. Reset them for
  // each draw so a clipped/viewport draw cannot leak state to its successor.
  pass.setViewport(...(draw.viewport??[0,0,field.width,field.height,0,1]) as [number,number,number,number,number,number]);
  pass.setScissorRect(...(draw.scissor??[0,0,field.width,field.height]) as [number,number,number,number]);
  if(draw.blendConstant)pass.setBlendConstant(draw.blendConstant);
  if(draw.stencilReference!==undefined)pass.setStencilReference(draw.stencilReference);
  const args=draw.arguments;
  if(draw.command==='draw')pass.draw(...args as [number,number,number,number]);
  else if(draw.command==='draw-indexed')pass.drawIndexed(...args as [number,number,number,number,number]);
  else if(draw.command==='draw-indirect')pass.drawIndirect(...args as [GPUBuffer,number]);
  else if(draw.command==='draw-indexed-indirect')pass.drawIndexedIndirect(...args as [GPUBuffer,number]);
  }}finally{pass?.end();}
 }
 /** Keep a pending GPU native job's SAME-pose masks alive. A camera may
  * change target/resolution while another target still awaits compact tone;
  * neither a clear nor a resize may overwrite that job's visibility field. */
 retain(field:SurfaceField):()=>void{
  const resident=this.retained.get(field.textures);if(!resident)throw Error('Cannot retain a foreign native field');
  resident.users++;let released=false;
  return ()=>{if(released)return;released=true;resident.users--;};
 }
 private destroy(resident:Resident):void{formDepths(this.device).release(resident.textures);resident.textures.forEach(t=>t.destroy());resident.depth.destroy();resident.coverage?.destroy();this.retained.delete(resident.textures);}
 submitted():void{
  for(const uniform of this.uniforms)uniform.destroy();this.uniforms=[];
  this.retired=this.retired.filter(resident=>{if(resident.users)return true;this.destroy(resident);return false;});
 }
 dispose():void{
  this.submitted();for(const resident of this.retired)this.destroy(resident);this.retired=[];
  if(this.resident)this.destroy(this.resident);this.resident=undefined;
 }
}

/**
 * Drop field owners for views which have left the world.  The plugin owns the
 * map, so a removed camera otherwise keeps its five field textures and depth
 * texture until the whole World is disposed.  `pending` is supplied by the
 * caller from the target queue: while any text job is pending, retain every
 * map entry so a SAME-pose job can release its field and its submission can
 * still reach `submitted()` safely.  The next non-pending frame performs the
 * actual owner destruction.
 */
export function pruneInactiveSurfaceFields(
 fields:Map<number,SurfaceFields>,
 active:ReadonlySet<number>,
 pending:boolean,
):void{
 if(pending)return;
 for(const [eid,owner] of fields){
  if(active.has(eid))continue;
  fields.delete(eid);
  owner.dispose();
 }
}
