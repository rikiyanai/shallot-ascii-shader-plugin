                                                                              
                                                                             
                                                                            
                                                                          
                                                                               
const SHADER=/* wgsl */`
@group(0) @binding(0) var source:texture_2d<f32>;
struct V {@builtin(position) position:vec4f,@location(0) uv:vec2f}
@vertex fn vertex(@builtin(vertex_index) vertex:u32)->V{
 let xy=vec2f(f32(vertex&1u)*2,f32(vertex>>1u)*2);
 return V(vec4f(xy*2-1,0,1),vec2f(xy.x,1-xy.y));
}
@fragment fn fragment(p:V)->@location(0) vec4f{
 let size=vec2i(textureDimensions(source));return textureLoad(source,clamp(vec2i(p.uv*vec2f(size)),vec2i(0),size-1),0);
}
`;
export class TextureBlit {
 private module?:GPUShaderModule;
 private readonly pipelines=new Map<GPUTextureFormat,GPURenderPipeline>();
 constructor(private readonly device:GPUDevice){}
 snapshot(encoder:GPUCommandEncoder,source:GPUTexture,target:GPUTextureView,format:GPUTextureFormat):void{
  let pipeline=this.pipelines.get(format);
  if(!pipeline){
   this.module??=this.device.createShaderModule({label:'MAIN texture blit',code:SHADER});
   pipeline=this.device.createRenderPipeline({label:'MAIN texture blit',layout:'auto',vertex:{module:this.module,entryPoint:'vertex'},fragment:{module:this.module,entryPoint:'fragment',targets:[{format}]}});
   this.pipelines.set(format,pipeline);
  }
  const group=this.device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:source.createView()}]});
  const pass=encoder.beginRenderPass({label:'MAIN texture blit',colorAttachments:[{view:target,loadOp:'load',storeOp:'store'}]});pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.draw(3);pass.end();
 }
}
