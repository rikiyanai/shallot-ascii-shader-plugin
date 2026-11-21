import type {NativeGlyph} from './native-page.ts';
import {nativeScalarFont} from './native-font.ts';
import {displayCharacters,assertPaintEqualsExport,OCCLUDED_FLAG,type TextFrame,type TextCharacter} from './text/text-frame.ts';
import {gpuNativeFontAddresses,requestGpuNativeTapeExport,validateGpuNativeTape,GPU_NATIVE_WORD_STRIDE,type GpuNativeFontAddress,type GpuNativeTape,type GpuNativeTapeReadback} from './gpu-native-tape.ts';

const SHADER=/* wgsl */`
struct Params {extent:vec4f,pageShift:vec4f}
struct Glyph {pen:vec2f,code:u32,foreground:u32}
@group(0) @binding(0) var<uniform> params:Params;
@group(0) @binding(1) var<storage,read> glyphs:array<Glyph>;
@group(0) @binding(2) var atlas:texture_2d<f32>;
@group(0) @binding(3) var appearance:texture_2d<f32>;
struct V { @builtin(position) position:vec4f,@location(0) uv:vec2f,@location(1) @interpolate(flat) code:u32,@location(2) @interpolate(flat) foreground:u32 }
@vertex fn glyphVertex(@builtin(vertex_index) vertex:u32,@builtin(instance_index) instance:u32)->V{
 let corners=array<vec2f,6>(vec2f(0,0),vec2f(1,0),vec2f(0,1),vec2f(0,1),vec2f(1,0),vec2f(1,1));
 let g=glyphs[instance];let uv=corners[vertex]*vec2f(32,17);let p=(g.pen+params.pageShift.xy+vec2f(-8,0)+uv)/params.extent.xy*vec2f(2,-2)+vec2f(-1,1);
 return V(vec4f(p,0,1),uv,g.code,g.foreground);
}
@fragment fn glyphFragment(v:V)->@location(0) vec4f{
 let cell=vec2u(v.code%32u,v.code/32u)*32u;
 let ink=textureLoad(atlas,vec2i(cell+vec2u(v.uv)),0).r;if(ink<0.5){discard;}
 return unpack4x8unorm(v.foreground);
}
@vertex fn pageVertex(@builtin(vertex_index) vertex:u32)->@builtin(position) vec4f{
 let xy=vec2f(f32(vertex&1u)*4.0-1.0,f32(vertex>>1u)*4.0-1.0);return vec4f(xy,0,1);
}
@fragment fn pageFragment(@builtin(position) p:vec4f)->@location(0) vec4f{
 if(params.extent.z<0.5&&params.extent.w<0.5){return vec4f(1);}
 let color=textureLoad(appearance,vec2i(p.xy),0);
 if(params.extent.z>0.5){return color;}
 let alpha=select(1.0,color.a,params.extent.w>0.5);
 return vec4f(vec3f(alpha),alpha);
}
`;

/**
 * Direct presentation of the canonical GPU tape.  Unlike SHADER, this path
 * has no host-built instance array: one 32-word sealed record is one intact
 * native occurrence, and the count is copied into an indirect draw argument
 * on the same command encoder.  The tape's font address is resolved through
 * the resident atlas-code table so the compiled grammar and publication use
 * exactly the same scalar font order.
 */
const GPU_TAPE_SHADER=/* wgsl */`
struct Params {extent:vec4f,pageShift:vec4f}
@group(0) @binding(0) var<uniform> params:Params;
@group(0) @binding(1) var<storage,read> tape:array<u32>;
@group(0) @binding(2) var<storage,read> fontCodes:array<u32>;
@group(0) @binding(3) var atlas:texture_2d<f32>;
@group(0) @binding(4) var appearance:texture_2d<f32>;
struct V { @builtin(position) position:vec4f,@location(0) uv:vec2f,@location(1) @interpolate(flat) code:u32,@location(2) @interpolate(flat) foreground:u32 }
@vertex fn glyphVertex(@builtin(vertex_index) vertex:u32,@builtin(instance_index) instance:u32)->V{
 let corners=array<vec2f,6>(vec2f(0,0),vec2f(1,0),vec2f(0,1),vec2f(0,1),vec2f(1,0),vec2f(1,1));
 let at=instance*${GPU_NATIVE_WORD_STRIDE}u;
 let pen=vec2f(f32(tape[at+3u]),f32(tape[at+4u]*17u));
 let uv=corners[vertex]*vec2f(32,17);let p=(pen+params.pageShift.xy+vec2f(-8,0)+uv)/params.extent.xy*vec2f(2,-2)+vec2f(-1,1);
 return V(vec4f(p,0,1),uv,fontCodes[tape[at+5u]],tape[at+10u]);
}
@fragment fn glyphFragment(v:V)->@location(0) vec4f{
 let cell=vec2u(v.code%32u,v.code/32u)*32u;let ink=textureLoad(atlas,vec2i(cell+vec2u(v.uv)),0).r;if(ink<0.5){discard;}
 return unpack4x8unorm(v.foreground);
}
@vertex fn pageVertex(@builtin(vertex_index) vertex:u32)->@builtin(position) vec4f{
 let xy=vec2f(f32(vertex&1u)*4.0-1.0,f32(vertex>>1u)*4.0-1.0);return vec4f(xy,0,1);
}
@fragment fn pageFragment(@builtin(position) p:vec4f)->@location(0) vec4f{
 if(params.extent.z<0.5&&params.extent.w<0.5){return vec4f(1);}
 let color=textureLoad(appearance,vec2i(p.xy),0);if(params.extent.z>0.5){return color;}
 let alpha=select(1.0,color.a,params.extent.w>0.5);return vec4f(vec3f(alpha),alpha);
}
`;

/** One publication owns the scalar text, native pens and exact glyph masks.
 * Display never has a separate bitmap approximation of exported characters. */
export class NativePublication<T extends TextFrame=TextFrame> {
  private readonly atlas:GPUTexture;
  private readonly glyphIds=new Map<string,{code:number;advance:number}>();
  private readonly pipelines=new Map<GPUTextureFormat,{glyph:GPURenderPipeline;page:GPURenderPipeline}>();
  private readonly module:GPUShaderModule;
  private readonly layout:GPUBindGroupLayout;
  private readonly gpuTapeModule:GPUShaderModule;
  private readonly gpuTapeLayout:GPUBindGroupLayout;
  private readonly gpuTapePipelines=new Map<GPUTextureFormat,{glyph:GPURenderPipeline;page:GPURenderPipeline}>();
  private readonly gpuFontAddresses:readonly GpuNativeFontAddress[];
  private readonly gpuFontCodes:GPUBuffer;
  private glyphs:GPUBuffer;
  private count=0;
  private current?:T;
  private parameters:GPUBuffer[]=[];
  constructor(private readonly device:GPUDevice,glyphs:readonly NativeGlyph[]){
    const compiledFont=glyphs;
    glyphs=nativeScalarFont(glyphs);
    const height=Math.ceil(glyphs.length/32)*32;
    this.atlas=device.createTexture({label:'MAIN intact Saitamaar glyphs',size:[1024,height],format:'r8unorm',usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});
    const image=new Uint8Array(1024*height);
    glyphs.forEach((g,code)=>{
      this.glyphIds.set(g.text,{code,advance:g.advance});
      for(const p of g.points){
        const x=p[0]!,y=p[1]!;
        if(x< -8||x>=24||y<0||y>=17)throw Error(`Native publication would clip the intact glyph ${g.text}`);
        image[(Math.floor(code/32)*32+y)*1024+(code%32)*32+x+8]=255;
      }
    });
    device.queue.writeTexture({texture:this.atlas},image,{bytesPerRow:1024},[1024,height]);
    const atlasCodes=new Map([...this.glyphIds].map(([text,glyph])=>[text,glyph.code]));
    // The tape's address is the exact `packFeatureGrammar` input index. The
    // atlas is scalarised for rendering, but the indirection table must retain
    // the original compiled-bank order rather than silently renumbering it.
    this.gpuFontAddresses=gpuNativeFontAddresses(compiledFont,atlasCodes);
    const fontCodes=new Uint32Array(this.gpuFontAddresses.map(address=>address.atlasCode));
    this.gpuFontCodes=device.createBuffer({label:'MAIN compiled native font atlas addresses',size:Math.max(4,fontCodes.byteLength),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
    if(fontCodes.byteLength)device.queue.writeBuffer(this.gpuFontCodes,0,fontCodes);
    this.glyphs=device.createBuffer({size:16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
    this.module=device.createShaderModule({label:'MAIN single native publication',code:SHADER});
    this.layout=device.createBindGroupLayout({entries:[
      {binding:0,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:'uniform'}},
      {binding:1,visibility:GPUShaderStage.VERTEX,buffer:{type:'read-only-storage'}},
      {binding:2,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:'float'}},
      {binding:3,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:'float'}},
    ]});
    this.gpuTapeModule=device.createShaderModule({label:'MAIN direct sealed native tape publication',code:GPU_TAPE_SHADER});
    this.gpuTapeLayout=device.createBindGroupLayout({label:'MAIN direct sealed native tape bindings',entries:[
      {binding:0,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:'uniform'}},
      {binding:1,visibility:GPUShaderStage.VERTEX,buffer:{type:'read-only-storage'}},
      {binding:2,visibility:GPUShaderStage.VERTEX,buffer:{type:'read-only-storage'}},
      {binding:3,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:'float'}},
      {binding:4,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:'float'}},
    ]});
  }
  publish(frame:T):void{
    const instances:number[]=[];
    const painted:TextCharacter[]=[];
    for(const char of displayCharacters(frame)){
      const [cp,x,y,advance,flags,foreground]=char;
      if(flags&OCCLUDED_FLAG)continue;
      const ch=String.fromCodePoint(cp),glyph=this.glyphIds.get(ch);
      if(!glyph||glyph.advance!==advance)throw Error(`Native publication missing glyph/advance ${ch}`);
      const pen:TextCharacter=[cp,Math.fround(x),Math.fround(y),advance,flags,foreground,char[6]];
      painted.push(pen);
      if(ch!==' '&&ch!=='　')instances.push(pen[1],pen[2],glyph.code,foreground);
    }
    assertPaintEqualsExport(frame,painted);
    const bytes=Math.max(16,instances.length*4);
    if(this.glyphs.size<bytes){this.glyphs.destroy();this.glyphs=this.device.createBuffer({size:bytes,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});}
    const floats=new Float32Array(instances),words=new Uint32Array(floats.buffer);
    for(let i=2;i<words.length;i+=4){words[i]=instances[i]!;words[i+1]=instances[i+1]!;}
    if(instances.length)this.device.queue.writeBuffer(this.glyphs,0,floats);
    this.count=instances.length/4;this.current=frame;
  }
  get frame():T|undefined{return this.current;}
  encode(encoder:GPUCommandEncoder,appearance:GPUTextureView|undefined,target:GPUTextureView,format:GPUTextureFormat,width:number,height:number,ascii:boolean,stacked:boolean,overlay=false):void{
    appearance??=this.atlas.createView(); // Unread binding in opaque ASCII mode.
    let pipelines=this.pipelines.get(format);
    if(!pipelines){
      const layout=this.device.createPipelineLayout({bindGroupLayouts:[this.layout]});
      pipelines={
        glyph:this.device.createRenderPipeline({layout,vertex:{module:this.module,entryPoint:'glyphVertex'},fragment:{module:this.module,entryPoint:'glyphFragment',targets:[{format}]}}),
        page:this.device.createRenderPipeline({layout,vertex:{module:this.module,entryPoint:'pageVertex'},fragment:{module:this.module,entryPoint:'pageFragment',targets:[{format}]}}),
      };this.pipelines.set(format,pipelines);
    }
    const params=this.device.createBuffer({size:32,usage:GPUBufferUsage.UNIFORM,mappedAtCreation:true});
    const shift=this.current?.page_shift_px??[0,0];
    new Float32Array(params.getMappedRange()).set([width,height,Number(!ascii||overlay),Number(stacked),shift[0],shift[1],0,0]);params.unmap();this.parameters.push(params);
    const group=this.device.createBindGroup({layout:this.layout,entries:[{binding:0,resource:{buffer:params}},{binding:1,resource:{buffer:this.glyphs}},{binding:2,resource:this.atlas.createView()},{binding:3,resource:appearance}]});
    const pass=encoder.beginRenderPass({label:'MAIN published native text',colorAttachments:[{view:target,loadOp:overlay?'load':'clear',storeOp:'store',clearValue:{r:0,g:0,b:0,a:0}}]});
    pass.setBindGroup(0,group);
    if(!overlay){pass.setPipeline(pipelines.page);pass.draw(3);}
    if(ascii&&this.current&&this.count){pass.setPipeline(pipelines.glyph);pass.draw(6,this.count);}
    pass.end();
  }
  /**
   * Present an already sealed GPU tape without materialising a NativeFrame or
   * uploading a host instance array.  Callers must keep the tape buffers alive
   * until the host submission fence retires them; `submitted()` only retires
   * publication-owned transient parameters and never the tape itself.
   */
  publishGpu(encoder:GPUCommandEncoder,tapes:GpuNativeTape|readonly GpuNativeTape[],appearance:GPUTextureView|undefined,target:GPUTextureView,format:GPUTextureFormat,width:number,height:number,ascii:boolean,stacked:boolean,overlay=false):void{
    appearance??=this.atlas.createView(); // Unread binding in opaque ASCII mode.
    const records:readonly GpuNativeTape[]=Array.isArray(tapes)?tapes:[tapes];
    for(const tape of records){
      validateGpuNativeTape(tape);
      if(tape.width!==width||tape.height!==height)throw Error(`GPU native tape extent ${tape.width}x${tape.height} does not match publication ${width}x${height}`);
      if(tape.fontAddressCount!==this.gpuFontAddresses.length)throw Error('GPU native tape font address space does not match publication');
    }
    let pipelines=this.gpuTapePipelines.get(format);
    if(!pipelines){
      const layout=this.device.createPipelineLayout({bindGroupLayouts:[this.gpuTapeLayout]});
      pipelines={
        glyph:this.device.createRenderPipeline({layout,vertex:{module:this.gpuTapeModule,entryPoint:'glyphVertex'},fragment:{module:this.gpuTapeModule,entryPoint:'glyphFragment',targets:[{format}]}}),
        page:this.device.createRenderPipeline({layout,vertex:{module:this.gpuTapeModule,entryPoint:'pageVertex'},fragment:{module:this.gpuTapeModule,entryPoint:'pageFragment',targets:[{format}]}}),
      };
      this.gpuTapePipelines.set(format,pipelines);
    }
    const params=this.device.createBuffer({label:'MAIN sealed native tape parameters',size:32,usage:GPUBufferUsage.UNIFORM,mappedAtCreation:true});
    // Canonical seal pens are already absolute native-page coordinates. A
    // legacy TextFrame page shift would move the GPU tape and break same-tape
    // export parity, so sealed presentation always uses zero shift.
    new Float32Array(params.getMappedRange()).set([width,height,Number(!ascii||overlay),Number(stacked),0,0,0,0]);params.unmap();this.parameters.push(params);
    const pageTape=records[0]?.words??this.glyphs;
    const pageGroup=this.device.createBindGroup({layout:this.gpuTapeLayout,entries:[
      {binding:0,resource:{buffer:params}},{binding:1,resource:{buffer:pageTape}},{binding:2,resource:{buffer:this.gpuFontCodes}},
      {binding:3,resource:this.atlas.createView()},{binding:4,resource:appearance},
    ]});
    const draws=records.map((tape,index)=>{
      const indirect=this.device.createBuffer({label:`MAIN sealed native tape indirect count ${index}`,size:16,usage:GPUBufferUsage.INDIRECT|GPUBufferUsage.COPY_DST,mappedAtCreation:true});
      // drawIndirect reads vertexCount, instanceCount, firstVertex, first
      // instance. The tape owns only instanceCount; initialize the immutable
      // six-vertex quad before copying that count at offset 4.
      new Uint32Array(indirect.getMappedRange()).set([6,0,0,0]);indirect.unmap();
      // DrawIndirect's instanceCount is the second u32. The canonical seal
      // owns this count, so display never waits for or reconstructs it on CPU.
      encoder.copyBufferToBuffer(tape.count,tape.countOffsetBytes,indirect,4,4);this.parameters.push(indirect);
      const group=this.device.createBindGroup({layout:this.gpuTapeLayout,entries:[
        {binding:0,resource:{buffer:params}},{binding:1,resource:{buffer:tape.words}},{binding:2,resource:{buffer:this.gpuFontCodes}},
        {binding:3,resource:this.atlas.createView()},{binding:4,resource:appearance},
      ]});
      return {group,indirect};
    });
    const pass=encoder.beginRenderPass({label:'MAIN published sealed native tape',colorAttachments:[{view:target,loadOp:overlay?'load':'clear',storeOp:'store',clearValue:{r:0,g:0,b:0,a:0}}]});
    if(!overlay){pass.setBindGroup(0,pageGroup);pass.setPipeline(pipelines.page);pass.draw(3);}
    if(ascii){pass.setPipeline(pipelines.glyph);for(const draw of draws){pass.setBindGroup(0,draw.group);pass.drawIndirect(draw.indirect,0);}}
    pass.end();
  }
  /** Backwards-compatible name for callers that treat publication as an
   * encoder operation. Both names use the same sealed-tape path. */
  encodeGpu(encoder:GPUCommandEncoder,tapes:GpuNativeTape|readonly GpuNativeTape[],appearance:GPUTextureView|undefined,target:GPUTextureView,format:GPUTextureFormat,width:number,height:number,ascii:boolean,stacked:boolean,overlay=false):void{
    this.publishGpu(encoder,tapes,appearance,target,format,width,height,ascii,stacked,overlay);
  }
  /** Export the same sealed tape that was displayed. This intentionally
   * schedules a readback and is never part of the display frame. */
  requestGpuExport(encoder:GPUCommandEncoder,tape:GpuNativeTape):GpuNativeTapeReadback{
    return requestGpuNativeTapeExport(this.device,encoder,tape,this.gpuFontAddresses);
  }
  submitted():void{for(const b of this.parameters)b.destroy();this.parameters.length=0;}
  dispose():void{this.submitted();this.glyphs.destroy();this.gpuFontCodes.destroy();this.atlas.destroy();}
}
