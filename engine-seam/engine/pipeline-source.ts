/* SHALLOT_MATERIAL_RECIPES_0116 */
/** A resolved material description is not an executable GPU descriptor.
 * Native modules are absent until the optional native consumer asks for them. */
export type RenderPipelineSourceDescriptor=Omit<GPURenderPipelineDescriptor,'vertex'|'fragment'>&{
 readonly vertex:Omit<GPUVertexState,'module'>&{readonly module?:GPUShaderModule};
 readonly fragment?:Omit<GPUFragmentState,'module'>&{readonly module?:GPUShaderModule};
};
export interface RenderPipelineRecipe extends RenderPipelineSource {materialize():GPURenderPipeline;}
export function prepareRenderPipelineRecipe(pipeline:object):RenderPipelineRecipe|undefined{
 const prepare=(pipeline as Record<PropertyKey,unknown>)[Symbol.for('shallot.material.prepare-recipe')];
 if(typeof prepare!=='function')throw Error('The TypeGPU material-recipe build transform is missing');
 return prepare.call(pipeline);
}
export interface RenderPipelineSource {
  readonly descriptor:RenderPipelineSourceDescriptor;
  readonly vertexCode:string;
  readonly fragmentCode:string|undefined;
}
const sources=new WeakMap<GPURenderPipeline,RenderPipelineSource>();
const layouts=new WeakMap<GPUBindGroupLayout,GPUBindGroupLayoutDescriptor>();
const groups=new WeakMap<GPUBindGroup,GPUBindGroupDescriptor>();
export function retainBindGroupLayoutSource(layout:GPUBindGroupLayout,descriptor:GPUBindGroupLayoutDescriptor):GPUBindGroupLayout{
  layouts.set(layout,{...descriptor,entries:Array.from(descriptor.entries,e=>({...e,buffer:e.buffer?{...e.buffer}:undefined,sampler:e.sampler?{...e.sampler}:undefined,texture:e.texture?{...e.texture}:undefined,storageTexture:e.storageTexture?{...e.storageTexture}:undefined,externalTexture:e.externalTexture?{...e.externalTexture}:undefined}))});return layout;
}
export function retainBindGroupSource(group:GPUBindGroup,descriptor:GPUBindGroupDescriptor):GPUBindGroup{
  groups.set(group,{...descriptor,entries:Array.from(descriptor.entries,e=>({...e,resource:'buffer' in e.resource?{...e.resource}:e.resource}))});return group;
}
export function bindGroupSource(group:GPUBindGroup):{descriptor:GPUBindGroupDescriptor;layout:GPUBindGroupLayoutDescriptor|undefined}|undefined{
  const descriptor=groups.get(group);return descriptor?{descriptor,layout:layouts.get(descriptor.layout)}:undefined;
}
export function retainRenderPipelineSource(pipeline:GPURenderPipeline,descriptor:GPURenderPipelineDescriptor,shaderSources:WeakMap<GPUShaderModule,string>,device?:GPUDevice):GPURenderPipeline{
  const vertexCode=shaderSources.get(descriptor.vertex.module);
  if(vertexCode===undefined)return pipeline;
  const saved:GPURenderPipelineDescriptor={...descriptor,
    vertex:{...descriptor.vertex,constants:descriptor.vertex.constants?{...descriptor.vertex.constants}:undefined,buffers:descriptor.vertex.buffers?Array.from(descriptor.vertex.buffers,buffer=>buffer?{...buffer,attributes:Array.from(buffer.attributes,attribute=>({...attribute}))}:null):undefined},
    fragment:descriptor.fragment?{...descriptor.fragment,constants:descriptor.fragment.constants?{...descriptor.fragment.constants}:undefined,targets:Array.from(descriptor.fragment.targets,target=>target?{...target,blend:target.blend?{color:{...target.blend.color},alpha:{...target.blend.alpha}}:undefined}:null)}:undefined,
    primitive:descriptor.primitive?{...descriptor.primitive}:undefined,
    multisample:descriptor.multisample?{...descriptor.multisample}:undefined,
    depthStencil:descriptor.depthStencil?{...descriptor.depthStencil,stencilFront:descriptor.depthStencil.stencilFront?{...descriptor.depthStencil.stencilFront}:undefined,stencilBack:descriptor.depthStencil.stencilBack?{...descriptor.depthStencil.stencilBack}:undefined}:undefined,
  };
  const fragmentCode=descriptor.fragment?shaderSources.get(descriptor.fragment.module):undefined;
  const retained={descriptor:saved,vertexCode,fragmentCode};
  sources.set(pipeline,retained);
  return pipeline;
}
export function renderPipelineSource(pipeline:GPURenderPipeline):RenderPipelineSource|undefined{return sources.get(pipeline);}
