import type {PhaseGeometryDraw} from '@dylanebert/shallot/rendering';
import {bindGroupSource,renderPipelineSource} from '@dylanebert/shallot/runtime';
import {WgslReflect} from 'wgsl_reflect';
import type {MaterialGeometryBatch,MaterialGeometryDraw} from './geometry.ts';

interface RawStage {code:string;entry:string;position:string;uv:string|undefined;arguments:readonly string[];group:number;groups:readonly number[]}
function withoutComments(code:string):string{
  let output='',depth=0,line=false;
  for(let i=0;i<code.length;i++){
    const pair=code.slice(i,i+2),ch=code[i]!;
    if(line){if(ch==='\n'){line=false;output+='\n';}else output+=' ';continue;}
    if(depth){if(pair==='/*'){depth++;output+='  ';i++;}else if(pair==='*/'){depth--;output+='  ';i++;}else output+=ch==='\n'?'\n':' ';continue;}
    if(pair==='//'){line=true;output+='  ';i++;}else if(pair==='/*'){depth=1;output+='  ';i++;}else output+=ch;
  }
  if(depth)throw Error('Unterminated WGSL comment');return output;
}
function rawStage(code:string,entryPoint:string|undefined):RawStage{
  const clean=withoutComments(code),entries=[...clean.matchAll(/@vertex\s+fn\s+(\w+)\s*\(/g)].filter(match=>!entryPoint||match[1]===entryPoint);
  if(entries.length!==1)throw Error('Raw geometry requires one recognized vertex entry');
  const match=entries[0]!,open=match.index!+match[0].length-1;let depth=1,end=open+1;
  for(;end<clean.length&&depth;end++){if(clean[end]==='(')depth++;if(clean[end]===')')depth--;}
  if(depth)throw Error('Invalid raw vertex parameter list');
  const parameters=clean.slice(open+1,end-1),parts=parameters.trim()?parameters.split(',').map(value=>value.trim()).filter(Boolean):[];
  const args:string[]=[],plain:string[]=[];
  for(const part of parts){
    const parameter=part.match(/^@builtin\((vertex_index|instance_index)\)\s+(\w+)\s*:\s*u32$/);
    if(!parameter)throw Error('Raw vertex inputs require an explicit vertex-buffer/attribute adapter');
    plain.push(`${parameter[2]}:u32`);args.push(parameter[1]==='vertex_index'?'vidx+mainRaw.info.y':'iid');
  }
  const returned=clean.slice(end).match(/^\s*->\s*(\w+)\s*\{/);
  if(!returned)throw Error('Raw geometry requires a structured vertex output');
  const fields=clean.match(new RegExp(`struct\\s+${returned[1]}\\s*\\{([\\s\\S]*?)\\}`))?.[1];
  const position=fields?.match(/@builtin\(position\)\s+(\w+)\s*:\s*vec4(?:f|<f32>)/)?.[1];
  if(!position)throw Error('Raw geometry has no declared clip position');
  const uv=fields?.match(/\b(uv)\s*:\s*vec2(?:f|<f32>)/)?.[1];
  const groups=[...new Set([...clean.matchAll(/@group\s*\(\s*(\d+)u?\s*\)/g)].map(m=>Number(m[1])))];
  const group=[0,1,2,3].find(g=>!groups.includes(g));
  if(group===undefined)throw Error('Raw geometry needs a free bind group or an explicit source adapter');
  code=code.slice(0,match.index!)+`fn ${match[1]}(${plain.join(',')})`+code.slice(end);
  return{code,entry:match[1]!,position,uv,arguments:args,group,groups};
}

/** Transport actual non-indexed render-phase vertex invocations. This adapter
 * retains native instance identity and clip support; it does not infer that an
 * arbitrary shader is a material, particle or physical attachment. */
export class RawGeometryTransport {
  private readonly pipelines=new WeakMap<GPURenderPipeline,{pipeline:GPURenderPipeline;stage:RawStage;bindings:Map<number,Set<number>>;groups:Map<number,WeakMap<GPUBindGroup,GPUBindGroup>>}>();
  private resources?:{capacity:number;width:number;height:number;vertices:GPUBuffer;attributes:GPUBuffer;targets:GPUTexture[]};
  private retired:NonNullable<RawGeometryTransport['resources']>[]=[];
  private readonly indices=new Map<number,GPUBuffer>();
  private readonly pending:GPUBuffer[]=[];
  private readonly owners=new WeakMap<object,number>();
  private ownerCount=0;
  constructor(private readonly device:GPUDevice){}
  encode(encoder:GPUCommandEncoder,draw:PhaseGeometryDraw,worldFromClip:Float32Array,normal:readonly number[],consume:(batch:MaterialGeometryBatch)=>void):void{
    if(draw.command!=='draw'||!draw.pipeline)throw Error(`Raw geometry source requires an adapter for ${draw.command}`);
    const [vertexCount,instanceCount,firstVertex,firstInstance]=draw.arguments as readonly number[];
    if(![vertexCount,instanceCount,firstVertex,firstInstance].every(v=>Number.isSafeInteger(v)&&v!>=0))throw Error('Invalid native raw draw arguments');
    if(!vertexCount||!instanceCount)return;
    const source=renderPipelineSource(draw.pipeline);
    if(!source)throw Error('Native raw pipeline has no retained shader source');
    if(source.descriptor.primitive?.topology&&source.descriptor.primitive.topology!=='triangle-list')throw Error('Raw topology requires its declared primitive adapter');
    if(vertexCount!%3)throw Error('Triangle-list raw source has an incomplete primitive');
    if(draw.vertices.length)throw Error('Raw vertex buffers need an explicit source stream adapter');
    let compiled=this.pipelines.get(draw.pipeline);
    if(!compiled){
      const stage=rawStage(source.vertexCode,source.descriptor.vertex.entryPoint);
      const code=stage.code+`
struct MainRawParameters {worldFromClip:mat4x4f,normal:vec4f,info:vec4u,shape:vec4u}
@group(${stage.group}) @binding(0) var<uniform> mainRaw:MainRawParameters;
struct MainRawVertex { @builtin(position) position:vec4f,@location(0) @interpolate(flat) world:vec3f,@location(1) @interpolate(flat) attributes:vec4u }
fn mainRawOct(value:vec3f)->vec2f{
 let n=value/max(dot(abs(value),vec3f(1.0)),0.000000001);
 return select((vec2f(1)-abs(n.yx))*select(vec2f(-1),vec2f(1),n.xy>=vec2f(0)),n.xy,n.z>=0.0);
}
@vertex fn mainRawVertex(@builtin(vertex_index) vidx:u32,@builtin(instance_index) iid:u32)->MainRawVertex{
 let original=${stage.entry}(${stage.arguments.join(',')});
 let clip=original.${stage.position};let world=mainRaw.worldFromClip*clip;
 let slot=vidx+(iid-mainRaw.info.z)*mainRaw.info.x;
 let pixel=vec2u(slot%mainRaw.info.w,slot/mainRaw.info.w);
 let size=vec2f(f32(mainRaw.info.w),f32(mainRaw.shape.x));
 let ndc=(vec2f(pixel)+vec2f(.5))/size*2.0-1.0;
 let uv=${stage.uv?'original.'+stage.uv:'vec2f(0)'};
 return MainRawVertex(vec4f(ndc.x,-ndc.y,.5,1),world.xyz/world.w,vec4u(pack2x16snorm(mainRawOct(mainRaw.normal.xyz)),bitcast<vec2u>(uv),iid));
}
struct MainRawOutput { @location(0) world:vec4u,@location(1) attributes:vec4u }
@fragment fn mainRawFragment(v:MainRawVertex)->MainRawOutput{return MainRawOutput(vec4u(bitcast<vec3u>(v.world),0xffffffffu),v.attributes);}
`;
      const module=this.device.createShaderModule({label:'MAIN actual raw vertex transport',code});
      const reflection=new WgslReflect(code),bindings=new Map<number,Set<number>>();
      for(const entry of [...reflection.entry.vertex.filter(e=>e.name==='mainRawVertex'),...reflection.entry.fragment.filter(e=>e.name==='mainRawFragment')])for(const resource of entry.resources){
        const used=bindings.get(resource.group)??new Set<number>();used.add(resource.binding);bindings.set(resource.group,used);
      }
      const pipeline=this.device.createRenderPipeline({layout:'auto',vertex:{module,entryPoint:'mainRawVertex',constants:source.descriptor.vertex.constants},fragment:{module,entryPoint:'mainRawFragment',targets:[{format:'rgba32uint'},{format:'rgba32uint'}]},primitive:{topology:'point-list'}});
      compiled={pipeline,stage,bindings,groups:new Map()};this.pipelines.set(draw.pipeline,compiled);
    }
    const required=vertexCount!*instanceCount!;
    if(!this.resources||this.resources.capacity<required){
      if(this.resources)this.retired.push(this.resources);
      const width=Math.min(this.device.limits.maxTextureDimension2D,Math.ceil(Math.sqrt(required)/16)*16),height=Math.ceil(required/width),capacity=width*height;
      if(height>this.device.limits.maxTextureDimension2D||capacity*16>this.device.limits.maxStorageBufferBindingSize)throw Error('Raw geometry capacity exceeds device limits');
      const make=()=>this.device.createBuffer({size:capacity*16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
      this.resources={capacity,width,height,vertices:make(),attributes:make(),targets:[0,1].map(()=>this.device.createTexture({size:[width,height],format:'rgba32uint',usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.COPY_SRC}))};
    }
    const r=this.resources;
    // The source count is part of the immutable per-draw parameter image, never
    // captured in a pipeline variant: instance counts may change every frame.
    const params=this.device.createBuffer({size:112,usage:GPUBufferUsage.UNIFORM,mappedAtCreation:true});
    const values=new Float32Array(params.getMappedRange());values.set(worldFromClip);values.set([...normal.slice(0,3),0],16);new Uint32Array(values.buffer).set([vertexCount!,firstVertex!,firstInstance!,r.width,r.height,0,0,0],20);params.unmap();this.pending.push(params);
    const extra=this.device.createBindGroup({layout:compiled.pipeline.getBindGroupLayout(compiled.stage.group),entries:[{binding:0,resource:{buffer:params}}]});
    const pass=encoder.beginRenderPass({label:'MAIN native raw geometry',colorAttachments:r.targets.map(texture=>({view:texture.createView(),loadOp:'clear' as const,storeOp:'store' as const,clearValue:{r:0,g:0,b:0,a:0}}))});
    pass.setPipeline(compiled.pipeline);
    for(const group of draw.groups)if(group.index!==compiled.stage.group&&compiled.bindings.has(group.index)){
      let cache=compiled.groups.get(group.index);if(!cache){cache=new WeakMap();compiled.groups.set(group.index,cache);}
      let cloned=group.offsets.length?undefined:cache.get(group.group);
      if(!cloned){
        const original=bindGroupSource(group.group);if(!original)throw Error('Native raw bindings have no retained source descriptor');
        const dynamic=new Map<number,number>();
        if(group.offsets.length){
          if(!original.layout)throw Error('Raw dynamic bindings require their explicit original layout');
          const entries=Array.from(original.layout.entries).filter(e=>e.buffer?.hasDynamicOffset).sort((a,b)=>a.binding-b.binding);
          if(entries.length!==group.offsets.length)throw Error('Raw dynamic offset count changed');
          entries.forEach((entry,i)=>dynamic.set(entry.binding,group.offsets[i]!));
        }
        const entries=Array.from(original.descriptor.entries).filter(entry=>compiled!.bindings.get(group.index)!.has(entry.binding)).map(entry=>{
          const offset=dynamic.get(entry.binding);if(offset===undefined)return entry;
          if(!('buffer' in entry.resource)||entry.resource.size===undefined)throw Error('Raw dynamic buffer needs an explicit source range');
          return{...entry,resource:{...entry.resource,offset:(entry.resource.offset??0)+offset}};
        });
        cloned=this.device.createBindGroup({layout:compiled.pipeline.getBindGroupLayout(group.index),entries});if(!group.offsets.length)cache.set(group.group,cloned);
      }
      pass.setBindGroup(group.index,cloned);
    }
    pass.setBindGroup(compiled.stage.group,extra);pass.draw(vertexCount!,instanceCount!,0,firstInstance!);pass.end();
    encoder.copyTextureToBuffer({texture:r.targets[0]!},{buffer:r.vertices,bytesPerRow:r.width*16},[r.width,r.height]);
    encoder.copyTextureToBuffer({texture:r.targets[1]!},{buffer:r.attributes,bytesPerRow:r.width*16},[r.width,r.height]);
    let indices=this.indices.get(vertexCount!);
    if(!indices){indices=this.device.createBuffer({size:vertexCount!*4,usage:GPUBufferUsage.INDEX|GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});this.device.queue.writeBuffer(indices,0,Uint32Array.from({length:vertexCount!},(_,i)=>i));this.indices.set(vertexCount!,indices);}
    const metadata=this.device.createBuffer({size:32,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.INDIRECT|GPUBufferUsage.COPY_SRC,mappedAtCreation:true});new Uint32Array(metadata.getMappedRange()).set([vertexCount!,instanceCount!,0,0,firstInstance!,0,0,0]);metadata.unmap();this.pending.push(metadata);
    let owner=this.owners.get(draw.renderer);if(owner===undefined){owner=++this.ownerCount;this.owners.set(draw.renderer,owner);}
    const materialSource:MaterialGeometryDraw={name:`phase:${owner}:${draw.stableId}:${draw.stablePart}:${draw.ordinal}`,vertex:draw.pipeline,vertexCode:source.vertexCode,layouts:[],groups:[],indices,vertexCapacity:vertexCount!,indirect:metadata,indirectOffset:0,bindings:{},material:source};
    consume({vertices:r.vertices,attributes:r.attributes,metadata,capacity:r.capacity,source:materialSource});
  }
  submitted():void{for(const b of this.pending)b.destroy();this.pending.length=0;for(const r of this.retired){r.vertices.destroy();r.attributes.destroy();r.targets.forEach(t=>t.destroy());}this.retired.length=0;}
  dispose():void{this.submitted();if(this.resources){this.resources.vertices.destroy();this.resources.attributes.destroy();this.resources.targets.forEach(t=>t.destroy());}for(const b of this.indices.values())b.destroy();this.indices.clear();}
}
