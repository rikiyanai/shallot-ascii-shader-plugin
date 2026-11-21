import type {SourceMaterialKey} from './source-material.ts';
import type {SourceConstruction,SourceAssemblyColumn} from './source-construction.ts';
import type {GpuIndexedSourceDescriptor,GpuIndexedSourcePose} from './gpu-indexed-source.ts';
import {prepareComputePipelines} from './gpu-pipeline-preparation.ts';
import {sameStorageBinding} from './source-storage.ts';
function validateGeometryPreparation(device:GPUDevice,capacity:number):void{
  if(!device.features.has('indirect-first-instance'))throw Error('MAIN geometry requires Shallot StandardRendering indirect-first-instance support');
  if(!Number.isSafeInteger(capacity)||capacity<3)throw Error('MAIN geometry capacity must cover a triangle');
}
/** GPU transport of the exact resolved Shallot material vertex stage.
 * Synthetic pixels address vertex invocations, never camera imagery. */
export interface MaterialGeometryDraw {
  readonly name:string;
  readonly vertex:object;
  readonly vertexCode:string;
  readonly layouts:readonly GPUBindGroupLayout[];
  readonly groups:readonly GPUBindGroup[];
  readonly indices:GPUBuffer;
  readonly vertexCapacity:number;
  /** Authoritative engine instance-table bound, not a subject quota. Live GPU counts
   * still decide each slice; absent bounds retain the guarded single draw. */
  readonly instanceCapacity?:number;
  /** Exact engine-resolved CullModeCode: Back0, None1, Front2. */
  readonly cullMode?:number;
  /** Absolute first vertex in a shared packed family, zero for unsliced streams. */
  readonly vertexBase?:number;
  readonly mesh?:number;
  readonly indexBase?:number;
  /** Index incidence before the material stage. Positions establish candidate
   * contacts only; the actual posed geometry always comes from the GPU stage. */
  readonly sourceGeometry?:{readonly positions:Float32Array;readonly indices:Uint32Array;readonly contactKeys?:Uint32Array;readonly normals?:Float32Array;readonly revision:number;readonly materialSource?:SourceMaterialKey;readonly surface?:'liquid'|'heightfield';readonly semantic?:Uint8Array;readonly construction?:SourceConstruction};
  /** Same original instance rows as the producer draw, not a CPU row census.
   * Parent identity is the authored tuple in the column, never row order. */
  readonly sourceAssembly?:SourceAssemblyColumn;
  /** Producer-issued ownership certificate.  A source may only enter the
   * early representation frontier when the engine has identified the owner
   * domain at the draw boundary; uncertified custom/GPU-indexed draws leave
   * this absent and retain the complete incumbent. */
  readonly ownerAuthority?:MaterialGeometryOwnerAuthority;
  /** Native live membership for this exact view's indirect draw.  The
   * membership buffer is the engine's packed GPU list and the indirect record
   * supplies its live count and firstInstance in the same submission.  No
   * CPU readback or census is implied by this descriptor.  It is kept
   * under a separate name until the relation consumer joins GPU indirect
   * membership with the existing CPU-indexed frontier type. */
  readonly nativeVisibleInstanceFrontier?:MaterialGeometryVisibleInstanceFrontier;
  /** Engine-owned indexed topology. It is mutually exclusive with sourceGeometry. */
  readonly gpuIndexedSource?:GpuIndexedSourceDescriptor;
  readonly indirect:GPUBuffer;
  readonly indirectOffset:number;
  /** Actual resolved eids, meshInstances, material parameters and custom bindings. */
  readonly bindings:Readonly<Record<string,unknown>>;
  readonly material:unknown;
}
export type MaterialGeometryOwnerAuthority=
 | {readonly kind:'standard-entity';readonly mesh:number;readonly identity:'eid';readonly posedBounds?:'engine-deformed-cull-sphere-v1'|'material-source-cull-sphere-v1'}
 | {readonly kind:'producer-row';readonly namespace:string;readonly records:GPUBufferBinding;readonly stride:24;readonly identity:'row'};
export interface MaterialGeometryVisibleInstanceFrontier {
  readonly kind:'native-indirect';
  readonly members:GPUBuffer;
  readonly membersStride:16;
  /** Exact DrawIndexedIndirect record used by this visitor item and slot. */
  readonly indirect:GPUBuffer;
  readonly indirectOffset:number;
  readonly owner:'entity'|'producer-row';
}
export interface MaterialGeometryBatch {
  /** Each 16-byte vertex is world.xyz as f32, then the exact u32 source entity.
   * 0xffffffff names a producer row. Stages without identities expose iid as
   * a producer row; they cannot borrow an unrelated ECS entity's lifetime.
   * Attributes are four u32: packed oct normal, bitcast UV.x, bitcast UV.y, row.
   * Integer attachments preserve all identity and chart bits without float IDs.
   * Address: vertex_index - vertexBase + (instance_index - firstInstance) * vertexCapacity.
   * Only vertices referenced by this batch's original indexed draw are initialized. */
  readonly vertices:GPUBuffer;
  readonly attributes:GPUBuffer;
  /** Eight u32: this slice's DrawIndexedIndirect five words, overflow, reserved, reserved.
   * firstInstance is the original source row plus the slice offset.
   * On overflow there is no geometry; the consumer must reject the completed frame. */
  readonly metadata:GPUBuffer;
  readonly capacity:number;
  readonly source:MaterialGeometryDraw;
  /** Same-encoder posed material output; never retained after submission. */
  readonly gpuPose?:GpuIndexedSourcePose;
}




function posedContentRevision(draw:MaterialGeometryDraw):number|undefined{
 const source=draw.gpuIndexedSource,certificate=source?.poseContent;
 if(!source||!certificate||certificate.sourceGeneration!==source.generation)return;
 return certificate.revision;
}
function validateVisibleInstanceFrontier(draw:MaterialGeometryDraw):void{
 const frontier=draw.nativeVisibleInstanceFrontier;if(!frontier)return;
 if(frontier.kind!=='native-indirect'||frontier.membersStride!==16||
    !Number.isSafeInteger(frontier.indirectOffset)||frontier.indirectOffset<0||
    frontier.indirect!==draw.indirect||frontier.indirectOffset!==draw.indirectOffset)
  throw Error('MAIN native instance frontier is not the draw\'s same-submit indirect slice');
 if(!(frontier.members.usage&GPUBufferUsage.STORAGE))throw Error('MAIN native instance frontier requires a storage membership buffer');
 if(draw.ownerAuthority?.kind==='producer-row'&&frontier.owner!=='producer-row')throw Error('MAIN producer frontier owner kind mismatch');
 if(draw.ownerAuthority?.kind==='standard-entity'&&frontier.owner!=='entity')throw Error('MAIN entity frontier owner kind mismatch');
}
/** Read-only same-encoder ranges, retained until submission. Separate ranges
 * may share a page; consumers must bind these ranges, never the whole page. */
export interface RetainedMaterialGeometryBatch extends MaterialGeometryBatch {
  readonly storage:{readonly identity:object;readonly vertices:GPUBufferBinding;readonly attributes:GPUBufferBinding;readonly metadata:GPUBufferBinding};
}
const PREPARE=`
struct Parameters { offset:u32, capacity:u32, width:u32, height:u32, vertexCapacity:u32, a:u32,b:u32,c:u32 }
@group(0) @binding(0) var<storage,read> source:array<u32>;
@group(0) @binding(1) var<storage,read_write> indirect:array<u32>;
@group(0) @binding(2) var<storage,read_write> metadata:array<u32>;
@group(0) @binding(3) var<uniform> params:Parameters;
@compute @workgroup_size(1) fn prepare(){
  let at=params.offset;let instances=source[at+1u];
  let perSlice=params.capacity/params.vertexCapacity;
  let bound=select(perSlice,params.c,params.c!=0u);
  let take=min(instances-min(instances,params.b),perSlice);
  var overflow=0u;
  if(instances>bound){overflow=1u;}
  for(var k=0u;k<5u;k++){metadata[k]=source[at+k];indirect[k]=source[at+k];}
  metadata[1]=take;indirect[1]=take;
  metadata[4]=source[at+4u]+params.b;indirect[4]=metadata[4];
  metadata[5]=overflow;metadata[6]=0u;metadata[7]=0u;
  if(overflow!=0u){indirect[0]=0u;indirect[1]=0u;}
}`;
/** Same count/slice operation as prepare, with one invocation per independent
 * destination. Source words stay live on the GPU; only address/topology facts
 * are retained in the registration table. No geometry or draw is culled here. */
const PREPARE_BATCH=`
@group(0) @binding(0) var<storage,read> source:array<u32>;
@group(0) @binding(1) var<storage,read_write> destination:array<u32>;
@group(0) @binding(2) var<storage,read> records:array<vec4u>;
@compute @workgroup_size(64) fn prepareBatch(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) grid:vec3u){
 let record=gid.x+gid.y*grid.x*64u;
 if(record>=arrayLength(&records)/2u){return;}
 let a=records[record*2u];let b=records[record*2u+1u];
 let instances=source[a.x+1u];let perSlice=a.y/a.z;
 let bound=select(perSlice,b.x,b.x!=0u);
 let take=min(instances-min(instances,a.w),perSlice);
 let overflow=select(0u,1u,instances>bound);
 for(var k=0u;k<5u;k++){destination[b.y+k]=source[a.x+k];destination[b.z+k]=source[a.x+k];}
 destination[b.y+1u]=take;destination[b.z+1u]=take;
 destination[b.y+4u]=source[a.x+4u]+a.w;destination[b.z+4u]=destination[b.y+4u];
 destination[b.y+5u]=overflow;destination[b.y+6u]=0u;destination[b.y+7u]=0u;
 if(overflow!=0u){destination[b.z]=0u;destination[b.z+1u]=0u;}
}`;
export function geometryExtractionShader(code:string,storage=false):string{
  const direct=[...code.matchAll(/@vertex\s+fn\s+(\w+)\s*\(\s*@builtin\(vertex_index\)\s+vidx\s*:\s*u32\s*,\s*@builtin\(instance_index\)\s+iid\s*:\s*u32\s*\)\s*->\s*(\w+)/g)];
  const entries=[...code.matchAll(/@vertex\s+fn\s+(\w+)\s*\(\s*(\w+)\s*:\s*(\w+)\s*\)\s*->\s*(\w+)/g)];
  if(entries.length+direct.length!==1)throw Error('MAIN geometry requires one recognized Shallot vertex entry');
  let invoke:string;
  const output=direct.length?direct[0]![2]!:entries[0]?.[4];
  const fields=output?code.match(new RegExp(`struct\\s+${output}\\s*\\{([\\s\\S]*?)\\}`))?.[1]??'':'';
  const hasEntity=/\beid\s*:\s*u32/.test(fields),hasRow=/\brow\s*:\s*u32/.test(fields);
  const identity=hasEntity?'vertex.eid':'0xffffffffu';
  const row=hasRow?'vertex.row':'iid';
  const uv=/\buv\s*:\s*vec2(?:f|<f32>)/.test(fields)?'vertex.uv':'vec2f(0.0)';
  if(direct.length){
    const [declaration,entry,output]=direct[0]!;
    code=code.replace(declaration,`fn ${entry}(vidx:u32,iid:u32)->${output}`);
    invoke=`${entry}(vidx,iid)`;
  }else{
    const [,entry,,input]=entries[0]!;
    const fields=code.match(new RegExp(`struct\\s+${input}\\s*\\{([\\s\\S]*?)\\}`))?.[1];
    if(!fields||!/@builtin\(vertex_index\)\s+vidx\s*:\s*u32/.test(fields)||!/@builtin\(instance_index\)\s+iid\s*:\s*u32/.test(fields)||fields.indexOf('vidx')>fields.indexOf('iid'))
      throw Error('MAIN geometry refuses an unknown vertex input contract');
    code=code.replace(/@vertex\s+fn/,'fn');invoke=`${entry}(${input}(vidx,iid))`;
  }
  if(/@group\s*\(\s*3\s*\)/.test(code))throw Error('MAIN geometry extraction requires group 3');
  return code+`
struct MainGeometryParameters { offset:u32, capacity:u32, width:u32, height:u32, vertexCapacity:u32, a:u32,b:u32,c:u32 }
struct MainGeometryMetadata { indexCount:u32,instanceCount:u32,firstIndex:u32,baseVertex:i32,firstInstance:u32,overflow:u32,a:u32,b:u32 }
@group(3) @binding(0) var<uniform> mainParameters:MainGeometryParameters;
@group(3) @binding(1) var<uniform> mainMetadata:MainGeometryMetadata;
struct MainGeometryVarying {
 @builtin(position) position:vec4f,
 @location(0) @interpolate(flat) world:vec3f,
 @location(1) @interpolate(flat) attributes:vec4u,
 @location(2) @interpolate(flat) entity:u32,
 ${storage?'@location(3) @interpolate(flat) slot:u32,':''}
}
fn mainOctEncode(value:vec3f)->vec2f {
 let n=value/max(dot(abs(value),vec3f(1.0)),0.000000001);
 return select((vec2f(1.0)-abs(n.yx))*select(vec2f(-1.0),vec2f(1.0),n.xy>=vec2f(0.0)),n.xy,n.z>=0.0);
}
@vertex fn mainGeometryVertex(@builtin(vertex_index) vidx:u32,@builtin(instance_index) iid:u32)->MainGeometryVarying {
 let vertex=${invoke};
 let slot=vidx-mainParameters.a+(iid-mainMetadata.firstInstance)*mainParameters.vertexCapacity;
 let pixel=vec2u(slot%mainParameters.width,${storage?'(slot/mainParameters.width)%mainParameters.height':'slot/mainParameters.width'});
 let size=vec2f(f32(mainParameters.width),f32(mainParameters.height));
 let ndc=(vec2f(pixel)+vec2f(0.5))/size*2.0-1.0;
 let uv=${uv};
 return MainGeometryVarying(vec4f(ndc.x,-ndc.y,0.5,1.0),vertex.world,
   vec4u(pack2x16snorm(mainOctEncode(vertex.worldNormal)),bitcast<vec2u>(uv),${row}),${identity}${storage?',slot':''});
}
${storage?`
@group(3) @binding(2) var<storage,read_write> mainPositions:array<atomic<u32>>;
@group(3) @binding(3) var<storage,read_write> mainAttributes:array<atomic<u32>>;
@fragment fn mainGeometryFragment(input:MainGeometryVarying)->@location(0) vec4f {
 let world=vec4u(bitcast<vec3u>(input.world),input.entity);let at=input.slot*4u;
 // An indexed draw can invoke a shared vertex more than once. Atomic stores
 // of the identical record avoid a writable-storage data race between those
 // fragments; no vertex stage writes storage and no value is quantized.
 for(var k=0u;k<4u;k++){atomicStore(&mainPositions[at+k],world[k]);atomicStore(&mainAttributes[at+k],input.attributes[k]);}
 return vec4f(0.0);
}`:`
struct MainGeometryOutput { @location(0) world:vec4u,@location(1) attributes:vec4u }
@fragment fn mainGeometryFragment(input:MainGeometryVarying)->MainGeometryOutput {
 return MainGeometryOutput(vec4u(bitcast<vec3u>(input.world),input.entity),input.attributes);
}`}`;
}
/** Consumers encode their GPU work inside consume, before scratch is reused by
 * the next draw. Indexed points preserve shared vertex identity, original native
 * firstIndex/baseVertex/firstInstance and live GPU instance/index counts. */
type GeometryScratch={vertices:GPUBuffer;normals:GPUBuffer;targets:GPUTexture[];width:number;height:number;capacity:number};
type GeometryParameters={params:GPUBuffer;prepare:GPUBindGroup;extract:GPUBindGroup;serial:number;identities:object[];storage?:{bindings:readonly GPUBufferBinding[];extract:GPUBindGroup}};
type GeometryCountProgram={source:GPUBuffer;target:GPUBuffer;words:Uint32Array;buffer:GPUBuffer;group:GPUBindGroup};
export class MaterialGeometryTransport{
  private readonly scratch=new Map<number,GeometryScratch>();
  private largeScratch?:GeometryScratch;
  private retired:GeometryScratch[]=[];
  private readonly metadata:GPUBuffer;
  private readonly indirect:GPUBuffer;
  private readonly extractionLayout:GPUBindGroupLayout;
  private readonly prepare:GPUComputePipeline;
  private readonly prepareBatch:GPUComputePipeline;
  private readonly countPrograms:GeometryCountProgram[][]=[];
  private readonly pipelines=new WeakMap<object,GPURenderPipeline>();
  private parameterImages:GPUBuffer[]=[];
  private readonly parameters=new Map<GPUBuffer,Map<string,GeometryParameters>>();
  private submissionSerial=0;
  private disposed=false;
  private poseSerial=0;
  private retainedCursor=0;
  private retainedPage=0;
  private retainedOffset=0;
  private readonly retainedPages:GPUBuffer[]=[];
  private readonly retainedIdentities:object[]=[];
  private constructionPage=0;
  private constructionOffset=0;
  private readonly constructionPages:GPUBuffer[]=[];
  private attributePage=0;
  private attributeOffset=0;
  private readonly attributePages:GPUBuffer[]=[];
  private metadataCursor=0;
  private readonly metadataPages:GPUBuffer[]=[];
  private storageLayout?:GPUBindGroupLayout;
  private storageTarget?:GPUTexture;
  private readonly storagePipelines=new WeakMap<object,GPURenderPipeline>();
  static async create(device:GPUDevice,capacity:number,progress?:(fraction:number)=>void):Promise<MaterialGeometryTransport>{
    validateGeometryPreparation(device,capacity);
    const pipelines=await prepareComputePipelines(device,new Map([
      ['material-geometry-prepare',()=>({layout:'auto' as const,compute:{module:device.createShaderModule({code:PREPARE}),entryPoint:'prepare'}})],
      ['material-geometry-prepare-batch',()=>({layout:'auto' as const,compute:{module:device.createShaderModule({code:PREPARE_BATCH}),entryPoint:'prepareBatch'}})],
    ]),progress);
    return new MaterialGeometryTransport(device,capacity,pipelines.get('material-geometry-prepare')!,pipelines.get('material-geometry-prepare-batch')!);
  }
  constructor(private readonly device:GPUDevice,readonly capacity:number,prepared?:GPUComputePipeline,preparedBatch?:GPUComputePipeline){
    validateGeometryPreparation(device,capacity);
    this.metadata=device.createBuffer({label:'MAIN geometry status',size:32,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_SRC});
    this.indirect=device.createBuffer({label:'MAIN guarded indexed count',size:20,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.INDIRECT});
    this.extractionLayout=device.createBindGroupLayout({entries:[
      {binding:0,visibility:GPUShaderStage.VERTEX,buffer:{type:'uniform'}},
      {binding:1,visibility:GPUShaderStage.VERTEX,buffer:{type:'uniform'}},
    ]});
    this.prepare=prepared??device.createComputePipeline({layout:'auto',compute:{module:device.createShaderModule({code:PREPARE}),entryPoint:'prepare'}});
    this.prepareBatch=preparedBatch??device.createComputePipeline({layout:'auto',compute:{module:device.createShaderModule({code:PREPARE_BATCH}),entryPoint:'prepareBatch'}});
  }
  /** Capacity is a batching preference, not a reservation for every scene.
   * One complete vertex stream always fits; declared instance ranges are played
   * in ordered slices. An unknown producer bound keeps the old guarded extent. */
  private workingSet(draw:MaterialGeometryDraw){
    const required=draw.instanceCapacity===undefined?Math.max(this.capacity,draw.vertexCapacity):
      Math.max(draw.vertexCapacity,Math.min(this.capacity,65536,draw.vertexCapacity*draw.instanceCapacity));
    if(required>65536&&this.largeScratch&&this.largeScratch.capacity>=required)return this.largeScratch;
    const capacity=Math.min(Math.max(this.capacity,draw.vertexCapacity),2**Math.ceil(Math.log2(required)));
    const retained=this.scratch.get(capacity);if(retained)return retained;
    const width=Math.min(this.device.limits.maxTextureDimension2D,Math.ceil(Math.sqrt(capacity)/16)*16),height=Math.ceil(capacity/width),bytes=width*height*16;
    if(height>this.device.limits.maxTextureDimension2D||bytes>this.device.limits.maxStorageBufferBindingSize||bytes>this.device.limits.maxBufferSize)
      throw Error('MAIN geometry capacity exceeds device buffer or texture limits');
    const partial:{destroy():void}[]=[];
    try{
      const vertices=this.device.createBuffer({label:'MAIN posed world positions',size:bytes,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});partial.push(vertices);
      const normals=this.device.createBuffer({label:'MAIN posed world normals',size:bytes,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});partial.push(normals);
      const targets=['world positions and entity','normal, chart and row'].map(label=>{
        const texture=this.device.createTexture({label:`MAIN invocation ${label}`,size:[width,height],format:'rgba32uint',usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.COPY_SRC});partial.push(texture);return texture;
      });
      // A large terrain must not turn every later small source into a clear
      // of that terrain's entire render attachment. Small geometric tiers
      // total at most 131072 invocation slots. Larger sources share one
      // high-water allocation; they are not retained once per source size.
      const scratch={vertices,normals,targets,width,height,capacity};
      if(required<=65536)this.scratch.set(capacity,scratch);
      else{if(this.largeScratch)this.retired.push(this.largeScratch);this.largeScratch=scratch;}
      return scratch;
    }catch(error){for(const resource of partial)resource.destroy();throw error;}
  }
  private parameterSet(draw:MaterialGeometryDraw,scratch:Pick<GeometryScratch,'capacity'|'width'|'height'>,first:number):GeometryParameters{
    const words=[draw.indirectOffset/4,scratch.capacity,scratch.width,scratch.height,draw.vertexCapacity,draw.vertexBase??0,first,draw.instanceCapacity??0];
    const key=words.join('/');let byDraw=this.parameters.get(draw.indirect);
    if(!byDraw){byDraw=new Map();this.parameters.set(draw.indirect,byDraw);}
    let entry=byDraw.get(key);
    if(!entry){
      // Immutable contents: separate draws/cameras can reuse an identical
      // image without queue.writeBuffer replacing an earlier command's data.
      // Any changed native offset, extent or slice gets a distinct image.
      const params=this.device.createBuffer({size:32,usage:GPUBufferUsage.UNIFORM,mappedAtCreation:true});
      try{
        new Uint32Array(params.getMappedRange()).set(words);params.unmap();
        const prepare=this.device.createBindGroup({layout:this.prepare.getBindGroupLayout(0),entries:[
          {binding:0,resource:{buffer:draw.indirect}},{binding:1,resource:{buffer:this.indirect}},
          {binding:2,resource:{buffer:this.metadata}},{binding:3,resource:{buffer:params}},
        ]});
        const extract=this.device.createBindGroup({layout:this.extractionLayout,entries:[{binding:0,resource:{buffer:params}},{binding:1,resource:{buffer:this.metadata}}]});
        entry={params,prepare,extract,serial:this.submissionSerial,identities:[]};byDraw.set(key,entry);
      }catch(error){params.destroy();throw error;}
    }
    entry.serial=this.submissionSerial;return entry;
  }
  encode(encoder:GPUCommandEncoder,draw:MaterialGeometryDraw,consume:(batch:MaterialGeometryBatch)=>void):void{
    if(this.disposed)throw Error('MAIN geometry transport is disposed');
    if(draw.layouts.length!==3||draw.groups.length!==3)throw Error('MAIN geometry requires exact material groups');
    if(!(draw.indirect.usage&GPUBufferUsage.STORAGE)||!(draw.indices.usage&GPUBufferUsage.INDEX))throw Error('MAIN geometry requires STORAGE indirect and INDEX topology');
    if(!Number.isSafeInteger(draw.vertexCapacity)||draw.vertexCapacity<=0)throw Error('MAIN geometry requires the real vertex stream capacity');
    if(draw.indirectOffset%4||draw.indirectOffset<0||draw.indirectOffset+20>draw.indirect.size)throw Error('MAIN geometry indirect record is out of range');
    if(draw.instanceCapacity!==undefined&&(!Number.isSafeInteger(draw.instanceCapacity)||draw.instanceCapacity<1))throw Error('Invalid declared instance extent');
    validateVisibleInstanceFrontier(draw);
    const scratch=this.workingSet(draw),{vertices,normals,targets,width,height,capacity}=scratch;
    let pipeline=this.pipelines.get(draw.vertex);
    if(!pipeline){
      const module=this.device.createShaderModule({label:`MAIN geometry ${draw.name}`,code:geometryExtractionShader(draw.vertexCode)});
      pipeline=this.device.createRenderPipeline({layout:this.device.createPipelineLayout({bindGroupLayouts:[...draw.layouts,this.extractionLayout]}),vertex:{module,entryPoint:'mainGeometryVertex'},fragment:{module,entryPoint:'mainGeometryFragment',targets:[{format:'rgba32uint'},{format:'rgba32uint'}]},primitive:{topology:'point-list'}});
      this.pipelines.set(draw.vertex,pipeline);
    }
    const perSlice=Math.floor(capacity/draw.vertexCapacity);
    if(!perSlice)throw Error('MAIN source vertex extent needs vertex-range chunks');
    const bound=draw.instanceCapacity??perSlice;
    // All slices remain in this SAME encoder/source pose. The GPU clamps
    // each to its actual native count; CPU bounds never manufacture rows.
    // Consumers finish using a slice before this scratch is reused.
    for(let first=0;first<bound;first+=perSlice){
    const parameters=this.parameterSet(draw,scratch,first);
    const compute=encoder.beginComputePass({label:'MAIN live geometry count'});
    compute.setPipeline(this.prepare);compute.setBindGroup(0,parameters.prepare);compute.dispatchWorkgroups(1);compute.end();
    const pass=encoder.beginRenderPass({label:'MAIN post-vertex geometry transport',colorAttachments:targets.map(texture=>({view:texture.createView(),loadOp:'clear' as const,storeOp:'store' as const,clearValue:{r:0,g:0,b:0,a:0}}))});
    pass.setPipeline(pipeline);draw.groups.forEach((binding,i)=>pass.setBindGroup(i,binding));pass.setBindGroup(3,parameters.extract);pass.setIndexBuffer(draw.indices,'uint32');pass.drawIndexedIndirect(this.indirect,0);pass.end();
    // The last slice may use fewer rows than its size tier. Every legal native invocation
    // lies in this source/slice's declared range; metadata rejects a native
    // count beyond that bound. Unknown producer extents keep the full copy.
    const rows=draw.instanceCapacity===undefined?height:
      Math.min(height,Math.ceil(draw.vertexCapacity*Math.min(perSlice,bound-first)/width));
    for(const [i,buffer] of [vertices,normals].entries())encoder.copyTextureToBuffer({texture:targets[i]!},{buffer,bytesPerRow:width*16,rowsPerImage:rows},[width,rows]);
    const contentRevision=posedContentRevision(draw);
    consume({vertices,attributes:normals,metadata:this.metadata,capacity,source:draw,
      ...(draw.gpuIndexedSource?{gpuPose:{sourceId:draw.gpuIndexedSource.sourceId,
        generation:draw.gpuIndexedSource.generation,posedRevision:this.poseSerial++,
        ...(contentRevision===undefined?{}:{posedContentRevision:contentRevision}),
        vertices:{buffer:vertices,size:vertices.size},attributes:{buffer:normals,size:normals.size},
        metadata:{buffer:this.metadata,size:32},vertexCapacity:draw.vertexCapacity}}:{})});
    }
  }
  /** Every native draw still executes its exact vertex function and original
   * indirect slice. Prepare their counts together, then write the same posed
   * records directly from synthetic point fragments into bounded work windows.
   * Empty GPU slices issue no points and never clear/copy a capacity-sized MRT. */
  encodeAll(encoder:GPUCommandEncoder,draws:readonly MaterialGeometryDraw[],consume:(batches:readonly RetainedMaterialGeometryBatch[])=>void):void{
    if(this.disposed)throw Error('MAIN geometry transport is disposed');
    const specs:{draw:MaterialGeometryDraw;capacity:number;first:number;bound:number}[]=[];
    for(const draw of draws){
      if(draw.layouts.length!==3||draw.groups.length!==3)throw Error('MAIN geometry requires exact material groups');
      if(!(draw.indirect.usage&GPUBufferUsage.STORAGE)||!(draw.indices.usage&GPUBufferUsage.INDEX))throw Error('MAIN geometry requires STORAGE indirect and INDEX topology');
      if(!Number.isSafeInteger(draw.vertexCapacity)||draw.vertexCapacity<=0)throw Error('MAIN geometry requires the real vertex stream capacity');
      if(draw.indirectOffset%4||draw.indirectOffset<0||draw.indirectOffset+20>draw.indirect.size)throw Error('MAIN geometry indirect record is out of range');
      if(draw.instanceCapacity!==undefined&&(!Number.isSafeInteger(draw.instanceCapacity)||draw.instanceCapacity<1))throw Error('Invalid declared instance extent');
      validateVisibleInstanceFrontier(draw);
      const required=draw.instanceCapacity===undefined?Math.max(this.capacity,draw.vertexCapacity):Math.max(draw.vertexCapacity,Math.min(this.capacity,65536,draw.vertexCapacity*draw.instanceCapacity));
      const capacity=Math.min(Math.max(this.capacity,draw.vertexCapacity),2**Math.ceil(Math.log2(required))),perSlice=Math.floor(capacity/draw.vertexCapacity);
      if(!perSlice)throw Error('MAIN source vertex extent needs vertex-range chunks');
      const bound=draw.instanceCapacity??perSlice;
      for(let first=0;first<bound;first+=perSlice)specs.push({draw,capacity,first,bound});
    }
    if(!specs.length)return;
    const alignment=Math.max(this.device.limits.minStorageBufferOffsetAlignment,this.device.limits.minUniformBufferOffsetAlignment),stride=alignment*2;
    const maximum=Math.min(this.device.limits.maxBufferSize,this.device.limits.maxStorageBufferBindingSize),bytes=specs.length*stride;
    // Metadata has no subject quota either. Extremely large command sets keep
    // the existing ordered source-sized transport rather than omit a draw.
    if(bytes>maximum){for(const draw of draws)this.encode(encoder,draw,batch=>{consume([this.retain(encoder,batch)]);this.releaseBatches();});return;}
    const lease=this.metadataCursor++;let metadataPage=this.metadataPages[lease];
    if(!metadataPage||metadataPage.size<bytes){if(metadataPage)this.parameterImages.push(metadataPage);metadataPage=this.device.createBuffer({label:'MAIN native geometry batch metadata and indirect records',size:Math.min(maximum,2**Math.ceil(Math.log2(Math.max(stride,bytes)))),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.UNIFORM|GPUBufferUsage.INDIRECT});this.metadataPages[lease]=metadataPage;}
    const layout=this.storageLayout??=this.device.createBindGroupLayout({entries:[
      {binding:0,visibility:GPUShaderStage.VERTEX,buffer:{type:'uniform'}},{binding:1,visibility:GPUShaderStage.VERTEX,buffer:{type:'uniform'}},
      {binding:2,visibility:GPUShaderStage.FRAGMENT,buffer:{type:'storage'}},{binding:3,visibility:GPUShaderStage.FRAGMENT,buffer:{type:'storage'}},
    ]});
    const target=this.storageTarget??=this.device.createTexture({label:'MAIN synthetic point storage transport target',size:[128,128],format:'rgba8unorm',usage:GPUTextureUsage.RENDER_ATTACHMENT});
    const targetView=target.createView();
    type Record={batch:RetainedMaterialGeometryBatch;pipeline:GPURenderPipeline;groups:NonNullable<GeometryParameters['storage']>;indirectOffset:number};
    const windows:Record[][]=[];let window:Record[]=[],windowBytes=0;
    const endWindow=()=>{if(window.length)windows.push(window);window=[];windowBytes=0;this.releaseBatches();};
    for(let ordinal=0;ordinal<specs.length;ordinal++){
      const {draw,capacity,first,bound}=specs[ordinal]!;
      let pipeline=this.storagePipelines.get(draw.vertex);
      if(!pipeline){const module=this.device.createShaderModule({label:'MAIN storage geometry '+draw.name,code:geometryExtractionShader(draw.vertexCode,true)});pipeline=this.device.createRenderPipeline({layout:this.device.createPipelineLayout({bindGroupLayouts:[...draw.layouts,layout]}),vertex:{module,entryPoint:'mainGeometryVertex'},fragment:{module,entryPoint:'mainGeometryFragment',targets:[{format:'rgba8unorm',writeMask:0}]},primitive:{topology:'point-list'}});this.storagePipelines.set(draw.vertex,pipeline);}
      const input=draw.gpuIndexedSource;
      const compact=draw.instanceCapacity===1&&(draw.indexBase??0)===0&&(draw.vertexBase??0)===0&&input?.topology.indexPattern==='independent-quads'&&input.registrationIndexCount!==undefined&&input.registrationIndexCount>0&&input.registrationIndexCount%6===0;
      const count=compact?input!.registrationIndexCount!/6*4:Math.min(capacity,draw.vertexCapacity*(bound-first));
      const vertices=this.outputRange(count*16,input?'construction':'batch'),attributes=this.outputRange(count*16,'attributes');
      const metadata:GPUBufferBinding={buffer:metadataPage,offset:ordinal*stride,size:32};
      const indirect:GPUBufferBinding={buffer:metadataPage,offset:ordinal*stride+alignment,size:20};
      const parameters=this.parameterSet(draw,{capacity,width:128,height:128},first),bindings=[vertices,attributes,metadata];
      if(!parameters.storage||!parameters.storage.bindings.every((b,i)=>sameStorageBinding(b,bindings[i]!))){
        parameters.storage={bindings,extract:this.device.createBindGroup({layout,entries:[{binding:0,resource:{buffer:parameters.params}},{binding:1,resource:metadata},{binding:2,resource:vertices},{binding:3,resource:attributes}]})};
      }
      // Residency belongs to the source/slice parameter image, NOT its ordinal
      // in this frame's visible draw list. Reordering sources otherwise creates
      // another source x global-slot cache entry on each walk/turn. Cameras
      // retain separate identities; duplicate uses still take the established
      // one-frame uniform path in contact/relation consumers.
      const identity=parameters.identities[lease]??(parameters.identities[lease]={});
      const contentRevision=posedContentRevision(draw);
      const batch:RetainedMaterialGeometryBatch={vertices:vertices.buffer,attributes:attributes.buffer,metadata:metadata.buffer,capacity:count,source:draw,storage:{identity,vertices,attributes,metadata},
        ...(input?{gpuPose:{sourceId:input.sourceId,generation:input.generation,posedRevision:this.poseSerial++,...(contentRevision===undefined?{}:{posedContentRevision:contentRevision}),vertices,attributes,metadata,vertexCapacity:draw.vertexCapacity,...(compact?{singleInstanceVertexCount:count}:{})}}:{})};
      window.push({batch,pipeline,groups:parameters.storage,indirectOffset:indirect.offset!});
      windowBytes+=attributes.size!+32+(input?0:vertices.size!);
      if(windowBytes>=8*1024*1024)endWindow();
    }
    endWindow();
    // All metadata is immutable for this encoder after this pass. The two
    // writable ranges of each record are disjoint and aligned; render binds
    // the same buffer only for uniform/indirect reads, never as writable.
    this.encodeBatchCounts(encoder,specs,metadataPage,stride,lease);
    for(let index=0;index<windows.length;index++){
      const records=windows[index]!;
      const pass=encoder.beginRenderPass({label:'MAIN batched post-vertex storage transport',colorAttachments:[{view:targetView,loadOp:index?'load':'clear',storeOp:'store',clearValue:{r:0,g:0,b:0,a:0}}]});
      try{for(const record of records){const draw=record.batch.source;pass.setPipeline(record.pipeline);draw.groups.forEach((group,i)=>pass.setBindGroup(i,group));pass.setBindGroup(3,record.groups.extract);pass.setIndexBuffer(draw.indices,'uint32');pass.drawIndexedIndirect(metadataPage,record.indirectOffset);}}finally{pass.end();}
      // Consume before a later window overwrites its transient positions or
      // attributes. Indexed construction positions keep their frame lease.
      consume(records.map(record=>record.batch));
    }
  }
  private encodeBatchCounts(encoder:GPUCommandEncoder,specs:readonly {draw:MaterialGeometryDraw;capacity:number;first:number;bound:number}[],target:GPUBuffer,stride:number,lease:number):void{
    const cohorts=new Map<GPUBuffer,number[]>();
    for(let i=0;i<specs.length;i++){
      const {draw,capacity,first}=specs[i]!;
      let words=cohorts.get(draw.indirect);if(!words){words=[];cohorts.set(draw.indirect,words);}
      words.push(draw.indirectOffset/4,capacity,draw.vertexCapacity,first,draw.instanceCapacity??0,i*stride/4,(i*stride+stride/2)/4,0);
    }
    const previous=this.countPrograms[lease]??[],next:GeometryCountProgram[]=[];
    try{
      for(const [source,values]of cohorts){
        const old=previous[next.length];let program=old;
        if(!old||old.source!==source||old.target!==target||old.words.length!==values.length||!old.words.every((word,i)=>word===values[i])){
          const words=Uint32Array.from(values),buffer=this.device.createBuffer({label:'MAIN native geometry count program',size:words.byteLength,usage:GPUBufferUsage.STORAGE,mappedAtCreation:true});
          try{
            new Uint32Array(buffer.getMappedRange()).set(words);buffer.unmap();
            const group=this.device.createBindGroup({layout:this.prepareBatch.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:source}},{binding:1,resource:{buffer:target}},{binding:2,resource:{buffer}}]});
            program={source,target,words,buffer,group};
          }catch(error){buffer.destroy();throw error;}
        }
        next.push(program!);
      }
    }catch(error){for(let i=0;i<next.length;i++)if(next[i]!==previous[i])next[i]!.buffer.destroy();throw error;}
    // A changed layout gets an immutable table until the host submission seam;
    // another camera/encode in this frame must not overwrite earlier commands.
    for(let i=0;i<previous.length;i++)if(previous[i]!==next[i])this.parameterImages.push(previous[i]!.buffer);
    this.countPrograms[lease]=next;
    const pass=encoder.beginComputePass({label:'MAIN all native geometry counts'});pass.setPipeline(this.prepareBatch);
    try{for(const program of next){
      const count=Math.ceil(program.words.length/8/64),limit=this.device.limits.maxComputeWorkgroupsPerDimension;
      pass.setBindGroup(0,program.group);pass.dispatchWorkgroups(Math.min(count,limit),Math.ceil(count/limit));
    }}finally{pass.end();}
  }
  /** Keep the exact material output until all sources can share the contact
   * and relation passes. Frame pages replace per-source position-only buffers;
   * they are reused in queue order only after the encoder is submitted. */
  retain(encoder:GPUCommandEncoder,batch:MaterialGeometryBatch):RetainedMaterialGeometryBatch{
    if(this.disposed)throw Error('MAIN geometry transport is disposed');
    const draw=batch.source,input=draw.gpuIndexedSource;
    // Independent quads address precisely [0, indexCount/6*4). Only the
    // certified one-instance, zero-base draw may retain this shorter prefix;
    // unknown/multi-instance draws preserve every declared slice invocation.
    const compact=draw.instanceCapacity===1&&(draw.indexBase??0)===0&&(draw.vertexBase??0)===0&&input?.topology.indexPattern==='independent-quads'&&
      input.registrationIndexCount!==undefined&&input.registrationIndexCount>0&&input.registrationIndexCount%6===0;
    const count=compact?input!.registrationIndexCount!/6*4:
      Math.min(batch.capacity,draw.instanceCapacity===undefined?batch.capacity:draw.vertexCapacity*draw.instanceCapacity);
    const bytes=count*16;
    if(!Number.isSafeInteger(bytes)||bytes<=0||bytes>batch.vertices.size||bytes>batch.attributes.size)throw Error('Retained material pose exceeds its extracted vertex range');
    const copy=(from:GPUBuffer,size:number,construction=false):GPUBufferBinding=>{const range=this.outputRange(size,construction?'construction':'batch');encoder.copyBufferToBuffer(from,0,range.buffer,range.offset!,size);return range;};
    // Only indexed construction reads positions after contact/relation work.
    // Other material streams may have enormous producer-table bounds: retain
    // them for one ordered batch, not the entire world's declared capacity.
    const vertices=copy(batch.vertices,bytes,!!batch.gpuPose),attributes=copy(batch.attributes,bytes),metadata=copy(batch.metadata,32);
    const slot=this.retainedCursor++,identity=this.retainedIdentities[slot]??(this.retainedIdentities[slot]={});
    return {...batch,vertices:vertices.buffer,attributes:attributes.buffer,metadata:metadata.buffer,capacity:count,
      storage:{identity,vertices,attributes,metadata},
      ...(batch.gpuPose?{gpuPose:{...batch.gpuPose,vertices,attributes,metadata,...(compact?{singleInstanceVertexCount:count}:{})}}:{})};
  }
  /** After the caller encodes ALL contact/relation readers of a batch, its
   * transient pages may be overwritten by later commands in the same encoder.
   * Indexed construction positions remain distinct until submission. */
  releaseBatches():void{this.retainedPage=0;this.retainedOffset=0;this.attributePage=0;this.attributeOffset=0;}
  private outputRange(size:number,kind:'batch'|'construction'|'attributes'):GPUBufferBinding{
    const alignment=this.device.limits.minStorageBufferOffsetAlignment,maximum=Math.min(this.device.limits.maxBufferSize,this.device.limits.maxStorageBufferBindingSize);
    if(!Number.isSafeInteger(size)||size<=0||size>maximum)throw Error('Material output range needs source-sized chunks');
    const pages=kind==='construction'?this.constructionPages:kind==='attributes'?this.attributePages:this.retainedPages;
    let index=kind==='construction'?this.constructionPage:kind==='attributes'?this.attributePage:this.retainedPage;
    const used=kind==='construction'?this.constructionOffset:kind==='attributes'?this.attributeOffset:this.retainedOffset;
    let offset=Math.ceil(used/alignment)*alignment,page=pages[index];
    if(page&&offset+size>page.size){index++;offset=0;page=pages[index];}
    if(!page||page.size<size){if(page)this.parameterImages.push(page);page=this.device.createBuffer({label:'MAIN '+kind+' material output page',size:Math.max(size,Math.min(8*1024*1024,maximum)),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});pages[index]=page;}
    if(kind==='construction'){this.constructionPage=index;this.constructionOffset=offset+size;}
    else if(kind==='attributes'){this.attributePage=index;this.attributeOffset=offset+size;}
    else{this.retainedPage=index;this.retainedOffset=offset+size;}
    return {buffer:page,offset,size};
  }
  /** Call after submitting or discarding the encoder containing encode. */
  submitted():void{
    this.retainedCursor=0;this.retainedPage=0;this.retainedOffset=0;
    this.constructionPage=0;this.constructionOffset=0;
    this.attributePage=0;this.attributeOffset=0;this.metadataCursor=0;
    this.submissionSerial++;
    for(const [indirect,entries]of this.parameters){for(const [key,entry]of entries)if(this.submissionSerial-entry.serial>240){entry.params.destroy();entries.delete(key);}if(!entries.size)this.parameters.delete(indirect);}
    for(const buffer of this.parameterImages)buffer.destroy();this.parameterImages=[];
    for(const scratch of this.retired){scratch.vertices.destroy();scratch.normals.destroy();for(const texture of scratch.targets)texture.destroy();}this.retired=[];
  }
  dispose():void{if(this.disposed)return;this.disposed=true;this.retired.push(...this.scratch.values());this.scratch.clear();if(this.largeScratch)this.retired.push(this.largeScratch);this.largeScratch=undefined;this.submitted();for(const entries of this.parameters.values())for(const entry of entries.values())entry.params.destroy();this.parameters.clear();for(const programs of this.countPrograms)for(const program of programs)program.buffer.destroy();this.countPrograms.length=0;for(const buffer of [...this.retainedPages,...this.constructionPages,...this.attributePages,...this.metadataPages])buffer.destroy();this.retainedPages.length=0;this.constructionPages.length=0;this.attributePages.length=0;this.metadataPages.length=0;this.retainedIdentities.length=0;this.storageTarget?.destroy();this.metadata.destroy();this.indirect.destroy();}
}
