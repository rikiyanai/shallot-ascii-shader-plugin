import {SourceStorage,allocateStorageRanges,type SourceStorageRange} from './source-storage.ts';
import {prepareComputePipelines,type ComputePipelineProgress} from './gpu-pipeline-preparation.ts';
function validatePhysicalPreparation(backend:'legacy'|'gpu-csr',retention:SourceComplexRetention):void{
 if(retention==='summary'&&backend!=='gpu-csr')throw Error('Physical summary residency requires compiled GPU incidence');
}
import type {CompositionGeometry,SourceAnalysis} from './composition.ts';
import {SourceInstanceArena,type SourceInstanceRange} from './source-instance-arena.ts';
import type {MeshSourceComplex} from './source-complex.ts';
import {SourceProgramCompiler,type SourceCompilation} from './source-program-compiler.ts';
import type {CompiledSourceProgram,SourceComplexRetention,PlaybackSourceAnalysis,SourceProgramInput} from './source-program.ts';
import type {RegisteredSourceComplex} from './source-cache.ts';
import type {SourceResidencySummary} from './source-transfer.ts';
import type {PosedSourceBatch} from './surface-relations.ts';
import {GpuRelationSourceCompiler} from './source-incidence-residual.ts';
import {GpuSourceNumerics,canRegisterNumericSource,type NumericSourceCompilation,type NumericSourceReady} from './gpu-source-numerics.ts';
import {decodeSourceSemanticResidual,type SourceSemanticResidual} from './source-semantic-residual.ts';
import {sourcePartAnchorVertices} from './compiled-construction-topology.ts';
import {SOURCE_CHART_TAG,REST_SOURCE_CHART_WGSL} from './source-coordinates.ts';
import {MATERIAL_FOLIAGE_FACES_BIT} from './source-material.ts';
import {SpatialIndex,type SpatialBounds} from './spatial-index.ts';
import {bindSourceWholeConstruction,fitsParentTier,selectSourceLod,type SourceLodInstance,type SourceLodParent,type SourceLodSelection,ParentGroupCache,type SourceWholeConstructionTemplate,type SmallConstructionAvailability,type SourceWholeNativePartAdmission} from './source-lod.ts';
import type {SourceSmallParent} from './source-small-parent.ts';

export type PhysicalSourceGeometry=CompositionGeometry&{readonly contactKeys?:Uint32Array;readonly semantic?:Uint8Array};
/** Exact dynamic-worker input shared by runtime registration and build-time
 * semantic identity. Keep source id, P0 adjacency, contact keys, material and
 * producer lineage in one factory; baked assets use this same model authority
 * before their semantic residual is attached. */
export function physicalSourceProgramInput(geometry:PhysicalSourceGeometry,backend:'legacy'|'gpu-csr',retention:SourceComplexRetention):SourceProgramInput{
 return {id:geometry.lineage?.source??geometry.owner,revision:geometry.revision,positions:geometry.positions,indices:geometry.indices,contactKeys:geometry.contactKeys,adjacency:geometry.lineage?.adjacency,incidence:backend==='gpu-csr'?'gpu-csr':undefined,complexMode:retention,physical:{lineage:geometry.lineage,surface:geometry.surface,materialSource:geometry.materialSource,construction:geometry.construction}};
}
export interface PhysicalSourceSemanticAttributes {readonly semantic:Uint8Array;readonly normals?:Float32Array}
export type PhysicalSourceSemanticResolver=(geometry:PhysicalSourceGeometry,input:SourceProgramInput)=>Promise<PhysicalSourceSemanticAttributes|undefined>;
interface PhysicalSourceData {
 readonly id:string;readonly revision:number;readonly positions:GPUBuffer;readonly normals:GPUBuffer;
 readonly indices:GPUBuffer;
 readonly incidence:GPUBuffer;readonly geometry:PhysicalSourceGeometry;
 readonly bounds:Float32Array;retired?:boolean;
 readonly positionsBinding:GPUBufferBinding;readonly normalsBinding:GPUBufferBinding;readonly indicesBinding:GPUBufferBinding;readonly incidenceBinding:GPUBufferBinding;
 readonly ranges:{positions:SourceStorageRange;normals:SourceStorageRange;indices?:SourceStorageRange;incidence:SourceStorageRange};
 readonly numeric?:NumericSourceReady;
 readonly wholeConstruction?:SourceWholeConstructionTemplate;
 readonly smallParent?:SourceSmallParent;
 readonly partAnchorVertices?:Uint32Array;
}
export type PhysicalSource=PhysicalSourceData&(
 {readonly complex:MeshSourceComplex;readonly summary?:never;readonly analysis:SourceAnalysis;readonly profileSafe:boolean}|
 {readonly summary:SourceResidencySummary;readonly complex?:never;readonly analysis:PlaybackSourceAnalysis;readonly profileSafe:boolean});
interface Registration {readonly id:number;readonly bounds:Float32Array;priority?:number;readonly geometry:PhysicalSourceGeometry;resolvedGeometry?:PhysicalSourceGeometry;resolving?:Promise<void>;compilation?:SourceCompilation;numeric?:NumericSourceCompilation;readonly promise:Promise<void>;readonly resolve:()=>void;readonly reject:(error:Error)=>void;error?:Error}
export interface PhysicalSourceInstance {readonly geometry:CompositionGeometry;readonly domain:number;readonly owner:number}
export interface PhysicalResidentBatch {readonly batch:PosedSourceBatch;readonly source:RegisteredSourceComplex<SourceComplexRetention>;readonly domain:number}
export interface PhysicalProducerOwner {readonly mesh:number;readonly eid:number}
/** Parse the retained per-instance producer identity, not the shared source's
 * first-owner id. A bank mesh can have many copies in one retained batch. */
export function parsePhysicalProducerOwner(owner:string):PhysicalProducerOwner|undefined{
 const match=/^camera-0\/(\d+)\/producer-row\/(\d+)$/.exec(owner);if(!match)return;
 const mesh=Number(match[1]),eid=Number(match[2]);
 return Number.isSafeInteger(mesh)&&Number.isSafeInteger(eid)?{mesh,eid}:undefined;
}
interface RetainedBatch {
 readonly mesh:PhysicalSource;readonly batch:PosedSourceBatch;readonly range:SourceInstanceRange;
 readonly storageRanges:{vertices:SourceStorageRange;attributes:SourceStorageRange;metadata:SourceStorageRange};
 readonly members:string[];readonly matrices:Float32Array;revision:number;
 /** Inputs are leased until submitted(), never overwritten while an encoder
  * or native job can still read them. Outputs keep their source-owned identity. */
 readonly slots:SourceFrameSlot[];published?:PhysicalResidentBatch;
 /** A changed root is not a certified LOD candidate until it settles. */
 readonly settling:Set<number>;
 /** The view of this batch's last frustum cull. */
 culledClip?:Float32Array;
}
interface SourceFrameSlot {
 readonly roots:GPUBuffer;readonly params:GPUBuffer;group:GPUBindGroup;boundBounds:GPUBuffer;
 readonly memory:ArrayBuffer;readonly floats:Float32Array;readonly words:Uint32Array;
 readonly parameterImage:ArrayBuffer;readonly parameters:Uint32Array;busy:boolean;
}
type SpatialInstance={readonly batch:RetainedBatch;readonly instance:number};
const IDENTITY=new Float32Array([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]);
/** Retained batches of one domain are addressed `domain*BATCHES+index`. */
const BATCHES=0x100000;
const SHADER=/* wgsl */`
struct Params {extent:vec4u,low:vec4f,high:vec4f}
struct Root {world:mat4x4f,identity:vec4u}
@group(0) @binding(0) var<uniform> params:Params;
@group(0) @binding(1) var<storage,read> source:array<vec4f>;
@group(0) @binding(2) var<storage,read> normals:array<vec4f>;
@group(0) @binding(3) var<storage,read_write> vertices:array<vec4u>;
@group(0) @binding(4) var<storage,read_write> attributes:array<vec4u>;
@group(0) @binding(5) var<storage,read_write> bounds:array<vec4f>;
@group(0) @binding(6) var<storage,read> roots:array<Root>;
${REST_SOURCE_CHART_WGSL}
fn oct(value:vec3f)->vec2f{
 let n=value/max(dot(abs(value),vec3f(1)),1e-20);
 return select((vec2f(1)-abs(n.yx))*select(vec2f(-1),vec2f(1),n.xy>=vec2f(0)),n.xy,n.z>=0);
}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3u){
 let idInBatch=id.x+id.y*params.extent.z*64u;if(idInBatch>=params.extent.x*params.extent.y){return;}
 let i=idInBatch%params.extent.x;let root=roots[idInBatch/params.extent.x];let out=root.identity.y*params.extent.x+i;
 let p=root.world*source[i];let normal=normals[i].xyz;
 let worldNormal=normal.x*cross(root.world[1].xyz,root.world[2].xyz)+normal.y*cross(root.world[2].xyz,root.world[0].xyz)+normal.z*cross(root.world[0].xyz,root.world[1].xyz);
 vertices[out]=vec4u(bitcast<vec3u>(p.xyz),0xffffffffu);
 // The two previously unused words carry a tagged rest-source chart. The
 // chart follows the posed source, including every instance, without a
 // per-frame CPU producer or another buffer/bind-group.
 var chart=vec2u(0u);if(params.low.w!=0.0){chart=vec2u(bitcast<u32>(sourceHeightChart(source[i].xyz,params.low.xyz,params.high.xyz)),${SOURCE_CHART_TAG}u);}
 attributes[out]=vec4u(pack2x16snorm(oct(worldNormal)),chart,root.identity.x);
 if(i==0u){var low=vec3f(3.402823e38);var high=vec3f(-3.402823e38);
  for(var corner=0u;corner<8u;corner++){let q=(root.world*vec4f(select(params.low.xyz,params.high.xyz,vec3u(corner&1u,corner&2u,corner&4u)!=vec3u(0)),1)).xyz;low=min(low,q);high=max(high,q);}
  bounds[(params.extent.w+root.identity.y)*2u]=vec4f(low,0);bounds[(params.extent.w+root.identity.y)*2u+1u]=vec4f(high,0);
 }
}`;
/** The CULL test below on the host, for a whole batch: may this instance's
 * bounds touch the view? Same corners and planes, with a margin so that f64
 * here never rejects what the f32 shader keeps. A batch with no such instance
 * has every instance culled on the GPU, so its relation and contact stages
 * emit nothing and are not encoded (frame cost, 2026-11-11). */
// PREDICTIVE REGISTRATION (speed, 2026-11-17). Unseen sources near the camera
// start compiling before they are seen, so a camera turn or a few steps of
// walking finds them ready instead of blocking publication. Only the current
// frustum still BLOCKS a job (no visible form is ever silently omitted); the
// predicted ring only starts work, at a priority below the required cohort.
// Ring: within PREDICT_RADIUS world units of the eye, or inside the frustum
// widened PREDICT_WIDEN times in x and y.
const PREDICT_RADIUS=60,PREDICT_WIDEN=3;
/** The eye of a perspective clip matrix (column-major): clip x = y = w = 0. */
function clipEye(clip:Float32Array):[number,number,number]|undefined{
 const a=(r:number,c:number)=>clip[c*4+r]!,rows=[0,1,3];
 const m=rows.map(r=>[a(r,0),a(r,1),a(r,2)]),b=rows.map(r=>-a(r,3));
 const det=(q:number[][])=>q[0]![0]!*(q[1]![1]!*q[2]![2]!-q[1]![2]!*q[2]![1]!)-q[0]![1]!*(q[1]![0]!*q[2]![2]!-q[1]![2]!*q[2]![0]!)+q[0]![2]!*(q[1]![0]!*q[2]![1]!-q[1]![1]!*q[2]![0]!);
 const d=det(m);if(!Number.isFinite(d)||Math.abs(d)<1e-12)return undefined;
 const column=(k:number)=>m.map((row,i)=>row.map((v,j)=>j===k?b[i]!:v));
 return [det(column(0))/d,det(column(1))/d,det(column(2))/d];
}
function widenedClip(clip:Float32Array,factor:number):Float32Array{const wide=clip.slice();for(let c=0;c<4;c++){wide[c*4]=clip[c*4]!/factor;wide[c*4+1]=clip[c*4+1]!/factor;}return wide;}
/** Distance from a point to the world bounds of a transformed local box. */
function boundsDistance(local:Float32Array,m:Float32Array,point:readonly number[]):number{
 const low=[Infinity,Infinity,Infinity],high=[-Infinity,-Infinity,-Infinity];
 for(let corner=0;corner<8;corner++){
  const x=corner&1?local[4]!:local[0]!,y=corner&2?local[5]!:local[1]!,z=corner&4?local[6]!:local[2]!;
  for(let k=0;k<3;k++){const v=m[k]!*x+m[4+k]!*y+m[8+k]!*z+m[12+k]!;low[k]=Math.min(low[k]!,v);high[k]=Math.max(high[k]!,v);}
 }
 let sum=0;for(let k=0;k<3;k++){const p=point[k]!,d=p<low[k]!?low[k]!-p:p>high[k]!?p-high[k]!:0;sum+=d*d;}return Math.sqrt(sum);
}
function instanceMayShow(local:Float32Array,matrices:Float32Array,index:number,clip:Float32Array):boolean{
 const m=matrices.subarray(index*16,index*16+16);
 const low=[Infinity,Infinity,Infinity],high=[-Infinity,-Infinity,-Infinity];
 for(let corner=0;corner<8;corner++){
  const x=corner&1?local[4]!:local[0]!,y=corner&2?local[5]!:local[1]!,z=corner&4?local[6]!:local[2]!;
  for(let k=0;k<3;k++){const v=m[k]!*x+m[4+k]!*y+m[8+k]!*z+m[12+k]!;low[k]=Math.min(low[k]!,v);high[k]=Math.max(high[k]!,v);}
 }
 let outside=127;
 for(let corner=0;corner<8;corner++){
  const x=corner&1?high[0]!:low[0]!,y=corner&2?high[1]!:low[1]!,z=corner&4?high[2]!:low[2]!;
  const q=[0,1,2,3].map(k=>clip[k]!*x+clip[4+k]!*y+clip[8+k]!*z+clip[12+k]!);
  const w=q[3]!,margin=1e-3*(Math.abs(w)+1);
  const planes=[q[0]!+w,w-q[0]!,q[1]!+w,w-q[1]!,q[2]!,w-q[2]!,w-.00001];
  let mask=0;for(let plane=0;plane<7;plane++)if(planes[plane]!<-margin)mask|=1<<plane;outside&=mask;
 }
 return outside===0;
}
function instanceBounds(local:Float32Array,matrices:Float32Array,index:number):SpatialBounds{
 const at=index*16,low=[Infinity,Infinity,Infinity],high=[-Infinity,-Infinity,-Infinity];
 for(let corner=0;corner<8;corner++){
  const x=corner&1?local[4]!:local[0]!,y=corner&2?local[5]!:local[1]!,z=corner&4?local[6]!:local[2]!;
  for(let k=0;k<3;k++){
   const a=matrices[at+k]!*x,b=matrices[at+4+k]!*y,c=matrices[at+8+k]!*z,d=matrices[at+12+k]!;
   const value=a+b+c+d,error=1e-5*(1+Math.abs(a)+Math.abs(b)+Math.abs(c)+Math.abs(d));
   low[k]=Math.min(low[k]!,value-error);high[k]=Math.max(high[k]!,value+error);
  }
 }
 return [low[0]!,low[1]!,low[2]!,high[0]!,high[1]!,high[2]!];
}
function clipPlanes(clip:Float32Array,out:Float32Array):void{
 for(let plane=0;plane<6;plane++)for(let k=0;k<4;k++){
  const row=plane<2?0:plane<4?1:2,sign=plane%2===0?1:-1;
  out[plane*4+k]=plane===4?clip[k*4+2]!:clip[k*4+3]!+sign*clip[k*4+row]!;
 }
}



/** Cached source incidence + GPU root posing for engine-neutral physical
 * triangles. It emits the same attributed batch consumed by Shallot's own
 * SurfaceRelations; it does not construct contours or characters on the CPU. */
export class PhysicalSourceTransport {
 private readonly pipeline:GPUComputePipeline;
 private readonly arena:SourceInstanceArena;
 private readonly sourcePositions:SourceStorage;private readonly sourceNormals:SourceStorage;private readonly sourceWords:SourceStorage;
 private readonly posedVertices:SourceStorage;private readonly posedAttributes:SourceStorage;private readonly catalog:SourceStorage;
 private readonly cache=new WeakMap<Float32Array,Map<Uint32Array,PhysicalSource>>();
 private readonly sourceKeys=new WeakMap<PhysicalSource,PhysicalSourceGeometry>();
 private readonly activeSources=new Set<PhysicalSource>();
 private readonly compiler=new SourceProgramCompiler();
 private numerics?:GpuSourceNumerics;
 private serial=0;
 private readonly registrations=new WeakMap<Float32Array,Map<Uint32Array,Registration>>();
 private demand?:{readonly jobs:ReadonlySet<Registration>;readonly promise:Promise<void>};
 private readonly work=new Set<Registration>();private readonly compiled:PhysicalSource[]=[];private disposed=false;
 private readonly batches=new Map<number,RetainedBatch>();
 private readonly spatial=new SpatialIndex<number,SpatialInstance>();
 private readonly spatialPlanes=new Float32Array(24);
 // Frame scratch, emptied by every encode; nothing here outlives the call.
 private readonly groups=new Map<number,PhysicalSourceInstance[]>();private readonly lists:PhysicalSourceInstance[][]=[];
 private readonly live=new Set<number>();private readonly changed:number[]=[];
 /** The batches reached by the last camera's conservative octree query.
  * Native GPU flags still decide exact per-instance visibility. */
 inView:ReadonlySet<PhysicalResidentBatch>&Set<PhysicalResidentBatch>=new Set();
 private readonly buffers=new Set<GPUBuffer>();private readonly leasedSlots=new Set<SourceFrameSlot>();private readonly retired:GPUBuffer[]=[];
 private readonly incidenceCompiler?:GpuRelationSourceCompiler;
 private semanticResolver?:PhysicalSourceSemanticResolver;
 static async create(device:GPUDevice,backend:'legacy'|'gpu-csr'='legacy',retention:SourceComplexRetention='full',progress?:ComputePipelineProgress):Promise<PhysicalSourceTransport>{
  validatePhysicalPreparation(backend,retention);
  const sourceWords=new SourceStorage(device,'MAIN source words');let arena:SourceInstanceArena|undefined,incidenceCompiler:GpuRelationSourceCompiler|undefined;
  try{
   const pipelines=await prepareComputePipelines(device,new Map([['physical-source-posing',()=>({layout:'auto' as const,compute:{module:device.createShaderModule({label:'MAIN physical source posing',code:SHADER}),entryPoint:'main'}})]]),(p,stage)=>progress?.(p/4,stage));
   arena=await SourceInstanceArena.create(device,(p,stage)=>progress?.(1/4+p/4,stage));
   if(backend==='gpu-csr')incidenceCompiler=await GpuRelationSourceCompiler.create(device,sourceWords,(p,stage)=>progress?.(1/2+p/2,stage));
   else progress?.(1);
   return new PhysicalSourceTransport(device,backend,retention,{pipeline:pipelines.get('physical-source-posing')!,arena,sourceWords,incidenceCompiler});
  }catch(error){if(incidenceCompiler)await incidenceCompiler.dispose();arena?.dispose();sourceWords.dispose();throw error;}
 }
 constructor(private readonly device:GPUDevice,private readonly backend:'legacy'|'gpu-csr'='legacy',private readonly retention:SourceComplexRetention='full',prepared?:{readonly pipeline:GPUComputePipeline;readonly arena:SourceInstanceArena;readonly sourceWords:SourceStorage;readonly incidenceCompiler?:GpuRelationSourceCompiler}){
  validatePhysicalPreparation(backend,retention);
  this.pipeline=prepared?.pipeline??device.createComputePipeline({layout:'auto',compute:{module:device.createShaderModule({label:'MAIN physical source posing',code:SHADER}),entryPoint:'main'}});this.arena=prepared?.arena??new SourceInstanceArena(device);
  this.sourcePositions=new SourceStorage(device,'MAIN source positions');this.sourceNormals=new SourceStorage(device,'MAIN source normals');this.sourceWords=prepared?.sourceWords??new SourceStorage(device,'MAIN source words');
  this.posedVertices=new SourceStorage(device,'MAIN posed vertices');this.posedAttributes=new SourceStorage(device,'MAIN posed attributes');this.catalog=new SourceStorage(device,'MAIN source catalog');
  if(backend==='gpu-csr')this.incidenceCompiler=prepared?.incidenceCompiler??new GpuRelationSourceCompiler(device,this.sourceWords);}
 /** Attach the optional build-time semantic asset owner. It is consulted only
  * by startRegistration, after prepare has admitted a demanded source. A
  * resolver miss deliberately falls through to the complete worker producer. */
 setSemanticResolver(resolver?:PhysicalSourceSemanticResolver):void{this.semanticResolver=resolver;}
 acquire(g:PhysicalSourceGeometry):PhysicalSource{
  if(this.disposed)throw Error('Physical source transport is disposed');
  const source=this.cache.get(g.positions)?.get(g.indices);
  if(!source||source.retired||source.revision!==g.revision)throw Error('Physical source revision is not registered; prepare sources before encoding');
  return source;
 }
 /** Register complete scene membership before a frame, not selected visible
  * instances inside encode. Camera/frustum changes do not retire a mesh. */
 /** With no clip this preserves the existing all-source contract. With a
  * clip, only CURRENT-view dependencies delay publication. Full scene
  * membership remains registered, but an unseen revision owns only its
  * bounds until demanded: no copied vertex snapshot or worker graph yet.
  * Already-started work is not cancelled merely because the camera turns.
  * Physical fields have source-model
  * vertex authority, so transformed local bounds enclose every raster vertex.
  * Original engine shadow/caster input remains outside this readiness gate. */
 prepare(geometry:readonly PhysicalSourceGeometry[],clip?:Float32Array):Promise<void>|undefined{
  if(this.disposed)throw Error('Physical source transport is disposed');
  const requested=new Set<Registration>(),required=new Set<Registration>(),predicted=new Set<Registration>(),live=new Set<PhysicalSource>();
  const eye=clip&&clipEye(clip),wide=clip&&widenedClip(clip,PREDICT_WIDEN);
  for(const g of geometry){
   if(!g.indices.length)continue;
   const source=this.cache.get(g.positions)?.get(g.indices);
   if(source&&!source.retired&&source.revision===g.revision){live.add(source);continue;}
   let byIndex=this.registrations.get(g.positions);
   if(!byIndex){byIndex=new Map();this.registrations.set(g.positions,byIndex);}
   let job=byIndex.get(g.indices);
   if(job&&job.geometry.revision!==g.revision){this.cancelRegistration(job);byIndex.delete(g.indices);job=undefined;}
   if(!job){
    const definition:PhysicalSourceGeometry={owner:g.owner,positions:g.positions,indices:g.indices,revision:g.revision,normals:g.normals,contactKeys:g.contactKeys,semantic:g.semantic,lineage:g.lineage,surface:g.surface,materialSource:g.materialSource,construction:g.construction};
    // Bounds are registered once per source revision, not recomputed per
    // camera or instance. The worker's completed bounds replace this scratch.
    const bounds=new Float32Array([Infinity,Infinity,Infinity,0,-Infinity,-Infinity,-Infinity,0]);
    for(let i=0;i<definition.positions.length;i+=3)for(let k=0;k<3;k++){bounds[k]=Math.min(bounds[k]!,definition.positions[i+k]!);bounds[k+4]=Math.max(bounds[k+4]!,definition.positions[i+k]!);}
    let resolve!:()=>void,reject!:(error:Error)=>void;
    const promise=new Promise<void>((yes,no)=>{resolve=yes;reject=no;});
    void promise.catch(()=>{});
    job={id:++this.serial,geometry:definition,bounds,promise,resolve,reject};byIndex.set(definition.indices,job);this.work.add(job);
   }
   requested.add(job);
   // OR across ALL instances of one source, regardless of enumeration order.
   if(!clip||!job.bounds.every(Number.isFinite)||instanceMayShow(job.bounds,g.worldFromSource??IDENTITY,0,clip))required.add(job);
   else if(wide&&(instanceMayShow(job.bounds,g.worldFromSource??IDENTITY,0,wide)||(eye&&boundsDistance(job.bounds,g.worldFromSource??IDENTITY,eye)<=PREDICT_RADIUS)))predicted.add(job);
  }
  for(const job of this.work)if(!requested.has(job)){this.cancelRegistration(job);const byIndex=this.registrations.get(job.geometry.positions);if(byIndex?.get(job.geometry.indices)===job)byIndex.delete(job.geometry.indices);}
  for(const job of requested){
   // Demand is OR-ed across every instance BEFORE starting the shared source.
   // Priority alone cannot avoid cloning and expanding an unseen revision.
   if(required.has(job)||predicted.has(job))this.startRegistration(job);
   // Keep the worker's service fairness WITHIN the required cohort. This
   // scheduler priority does not change the resulting compiled program.
   // Predicted (near, not yet seen) sources come after every required one.
   const priority=required.has(job)?1_000_000:predicted.has(job)?1_000:1;
   if(priority!==job.priority){job.priority=priority;job.compilation?.prioritize(priority);}
   job.numeric?.setRequired(required.has(job));
   if(required.has(job)&&job.error)throw job.error;
  }
  for(const source of this.activeSources)if(!live.has(source))this.retireSource(source);
  if(!required.size){this.demand=undefined;return;}
  if(this.demand?.jobs.size===required.size&&[...required].every(job=>this.demand!.jobs.has(job)))return this.demand.promise;
  const promise=Promise.all([...required].map(job=>job.promise)).then(()=>undefined);
  this.demand={jobs:required,promise};return promise;
 }
 /** The current-view sources the last prepare() still waits for: in view and
  * not yet compiled. A caller that publishes without them reports them. */
 demanded():readonly CompositionGeometry[]{return this.demand?[...this.demand.jobs].map(job=>job.geometry):[];}
 private cancelRegistration(job:Registration):void{this.work.delete(job);job.compilation?.cancel();job.numeric?.cancel();job.resolve();}
 private bakedSemantic(geometry:PhysicalSourceGeometry):{semantic:SourceSemanticResidual;identity:Uint32Array}|undefined{
  if(this.retention!=='summary'||this.backend!=='gpu-csr'||!this.incidenceCompiler||!geometry.semantic)return;
  // Physical posing must retain the producer's authored normal field. A
  // missing/incorrect normal stream is an unbaked source, so it deliberately
  // stays on the complete worker path instead of inventing a replacement.
  const vertices=geometry.positions.length/3;if(!geometry.normals||geometry.normals.length!==vertices*4)return;
  if(geometry.semantic.byteOffset%4||geometry.semantic.byteLength<48*4)throw Error('Physical baked source semantic packet is truncated');
  const words=new Uint32Array(geometry.semantic.buffer,geometry.semantic.byteOffset,geometry.semantic.byteLength/4),identity=words.subarray(40,48);
  const semantic=decodeSourceSemanticResidual(geometry.semantic,identity);
  if(semantic.vertices!==vertices||semantic.corners!==geometry.indices.length)throw Error('Physical baked source semantic revision does not match positions or indices');
  if(semantic.publishedAdjacency!==Boolean(geometry.lineage?.adjacency))return;
  if(!canRegisterNumericSource({positions:geometry.positions,indices:geometry.indices,contactKeys:geometry.contactKeys,revision:geometry.revision},semantic))return;
  return {semantic,identity:new Uint32Array(identity)};
 }
 private startRegistration(job:Registration):void{
  if(job.compilation||job.numeric||job.resolving||job.error||this.disposed)return;
  const definition=job.resolvedGeometry??job.geometry;
  const fail=(error:unknown)=>{
   if(this.disposed||!this.work.has(job))return;
   job.error=error instanceof Error?error:Error(String(error));job.reject(job.error);
  };
  // Asset lookup is intentionally after prepare's current-view admission. A
  // catalog miss, blocked storage or cancellation must never make every scene
  // geometry a startup dependency or prevent the normal worker route.
  // Published P0 is loaded through the same exact source-content identity.
  // Only its complete authority packet can pass bakedSemantic; legacy/missing
  // metadata still takes the original complete worker producer below.
  if(this.semanticResolver&&this.retention==='summary'&&this.backend==='gpu-csr'&&!definition.semantic){
   const input=physicalSourceProgramInput(definition,this.backend,this.retention),resolver=this.semanticResolver;
   job.resolving=Promise.resolve().then(()=>resolver(definition,input)).then(attributes=>{
    job.resolving=undefined;
    if(this.disposed||!this.work.has(job))return;
    const resolved=attributes?{...definition,semantic:attributes.semantic,...(attributes.normals?{normals:attributes.normals}:{})}:definition;
    job.resolvedGeometry=resolved;this.dispatchRegistration(job,resolved);
   }).catch(fail);
   return;
  }
  this.dispatchRegistration(job,definition);
 }
 private dispatchRegistration(job:Registration,definition:PhysicalSourceGeometry):void{
  if(job.compilation||job.numeric||job.error||this.disposed||!this.work.has(job))return;
  const fail=(error:unknown)=>{
   if(this.disposed||!this.work.has(job))return;
   job.error=error instanceof Error?error:Error(String(error));job.reject(job.error);
  };
  try{
   const baked=this.bakedSemantic(definition);
   if(baked){
    this.numerics??=new GpuSourceNumerics(this.device,this.sourceWords,this.incidenceCompiler!);
    job.numeric=this.numerics.register({positions:definition.positions,indices:definition.indices,contactKeys:definition.contactKeys,revision:definition.revision},baked.semantic,job.id,baked.identity);
    job.numeric.setRequired(job.priority===1_000_000);
    void job.numeric.result.then(async result=>{
     if(!this.work.has(job)||this.disposed){this.numerics!.retire(result);return;}
     await this.installNumeric(job,result,baked.semantic);
     this.work.delete(job);const byIndex=this.registrations.get(definition.positions);if(byIndex?.get(definition.indices)===job)byIndex.delete(definition.indices);job.resolve();
    }).catch(fail);
    return;
   }
   const compilation=job.compilation=this.compiler.compile(physicalSourceProgramInput(definition,this.backend,this.retention));
   // Asset resolution may finish after prepare returned. Preserve that
   // camera's required/predicted priority before the worker's first slice.
   if(job.priority!==undefined)compilation.prioritize(job.priority);
   void compilation.result.then(async program=>{
    if(!this.work.has(job)||this.disposed)return;
    if(program)await this.install(job,program);
    this.work.delete(job);
    const byIndex=this.registrations.get(definition.positions);
    if(byIndex?.get(definition.indices)===job)byIndex.delete(definition.indices);
    job.resolve();
   }).catch(fail);
  }catch(error){fail(error);}
 }
 isReady(geometry:CompositionGeometry):boolean{
  if(this.disposed)return false;
  const source=this.cache.get(geometry.positions)?.get(geometry.indices);
  return !!source&&!source.retired&&source.revision===geometry.revision;
 }
 takeCompiled():readonly PhysicalSource[]{return this.compiled.splice(0).filter(source=>!source.retired);}
 private async installNumeric(registration:Registration,result:NumericSourceReady,semantic:SourceSemanticResidual):Promise<void>{
  const g=registration.resolvedGeometry??registration.geometry,vertices=g.positions.length/3,positions=new Float32Array(vertices*4);
  for(let vertex=0;vertex<vertices;vertex++){positions.set(g.positions.subarray(vertex*3,vertex*3+3),vertex*4);positions[vertex*4+3]=1;}
  let ranges:{positions:SourceStorageRange;normals:SourceStorageRange};
  try{ranges=allocateStorageRanges({positions:{storage:this.sourcePositions,data:positions},normals:{storage:this.sourceNormals,data:g.normals!}});}
  catch(error){this.numerics!.retire(result);throw error;}
  const indicesBinding=result.indices,incidenceBinding=result.binding;
  if(this.disposed||!this.work.has(registration)){this.sourcePositions.retire(ranges.positions);this.sourceNormals.retire(ranges.normals);this.numerics!.retire(result);return;}
  const geometry=g,numericRanges={...ranges,indices:result.range,incidence:result.range},positionsBinding=this.sourcePositions.binding(ranges.positions),normalsBinding=this.sourceNormals.binding(ranges.normals);
  const base={id:geometry.owner,revision:g.revision,positions:positionsBinding.buffer,normals:normalsBinding.buffer,indices:indicesBinding.buffer,incidence:incidenceBinding.buffer,positionsBinding,normalsBinding,indicesBinding,incidenceBinding,ranges:numericRanges,geometry,bounds:registration.bounds,profileSafe:semantic.profileSafe===true,numeric:result,...(semantic.wholeConstruction?{wholeConstruction:semantic.wholeConstruction}:{}),...(semantic.smallParent?{smallParent:semantic.smallParent}:{}),...(semantic.partPrograms?{partAnchorVertices:sourcePartAnchorVertices(semantic.partPrograms,g.indices)}:{})};
  const resident:PhysicalSource={...base,summary:{schema:'shallot.source-residency-summary.v1',edgeCount:semantic.indexedEdges},analysis:{type:semantic.type,styleWord:semantic.styleWord,foliage:semantic.foliage,organic:semantic.organic,toneRegions:semantic.toneRegions,runProgram:result.runProgram}};
  let byIndex=this.cache.get(g.positions);if(!byIndex){byIndex=new Map();this.cache.set(g.positions,byIndex);}
  const previous=byIndex.get(g.indices);if(previous&&!previous.retired)this.retireSource(previous);
  byIndex.set(g.indices,resident);this.sourceKeys.set(resident,g);this.activeSources.add(resident);this.compiled.push(resident);
 }
 private async install(registration:Registration,program:CompiledSourceProgram):Promise<void>{
  if(program.complexMode!==this.retention)throw Error('Physical source returned the wrong topology retention');
  const g=registration.resolvedGeometry??registration.geometry;
  const physical=program.physical;if(!physical)throw Error('Physical source compiler returned no vertex inputs');
  // The worker's physical projection computes normals and retains the source
  // lineage; merge the admitted input fields so a semantic asset cannot vanish
  // merely because this source took the complete fallback route.
  const geometry:PhysicalSourceGeometry={...physical.geometry,contactKeys:g.contactKeys,semantic:g.semantic};
  const vertices=allocateStorageRanges({positions:{storage:this.sourcePositions,data:physical.positions},normals:{storage:this.sourceNormals,data:geometry.normals!}});
  let incidence:SourceStorageRange|undefined,indices:SourceStorageRange|undefined,indicesBinding:GPUBufferBinding,incidenceBinding:GPUBufferBinding;
  try{
   if(program.packet.kind==='gpu-csr-v1'){
    if(!this.incidenceCompiler)throw Error('Compact source submitted to a legacy physical owner');
    const uploaded=await this.incidenceCompiler.upload(program.packet.plan);incidence=uploaded.range;indicesBinding=uploaded.indices;incidenceBinding=uploaded.binding;
   }else{
    const legacy=allocateStorageRanges({indices:{storage:this.sourceWords,data:geometry.indices},incidence:{storage:this.sourceWords,data:program.packet.words}});
    indices=legacy.indices;incidence=legacy.incidence;indicesBinding=this.sourceWords.binding(indices);incidenceBinding=this.sourceWords.binding(incidence);
   }
  }catch(error){this.sourcePositions.retire(vertices.positions);this.sourceNormals.retire(vertices.normals);throw error;}
  const ranges={...vertices,indices,incidence};
  if(this.disposed)return;
  if(!this.work.has(registration)){this.sourcePositions.retire(ranges.positions);this.sourceNormals.retire(ranges.normals);if(indices)this.sourceWords.retire(indices);this.sourceWords.retire(incidence);return;}
  const positionsBinding=this.sourcePositions.binding(ranges.positions),normalsBinding=this.sourceNormals.binding(ranges.normals);
  // The parent selector consumes the source-revision verdict produced by
  // sourceProgramWork. Older cached programs omit it and stay conservative.
  const profileSafe=program.profileSafe===true;
  const base={id:geometry.owner,revision:g.revision,positions:positionsBinding.buffer,normals:normalsBinding.buffer,indices:indicesBinding.buffer,incidence:incidenceBinding.buffer,positionsBinding,normalsBinding,indicesBinding,incidenceBinding,ranges,geometry,bounds:physical.bounds,profileSafe,...(program.wholeConstruction?{wholeConstruction:program.wholeConstruction}:{}),...(program.smallParent?{smallParent:program.smallParent}:{}),...(program.partAnchorVertices?{partAnchorVertices:program.partAnchorVertices}:{})};
  const result:PhysicalSource=program.complexMode==='full'
   ?{...base,complex:program.complex,analysis:program.analysis}
   :{...base,summary:program.summary,analysis:program.analysis};
  let byIndex=this.cache.get(g.positions);if(!byIndex){byIndex=new Map();this.cache.set(g.positions,byIndex);}
  const previous=byIndex.get(g.indices);if(previous&&!previous.retired)this.retireSource(previous);
  byIndex.set(g.indices,result);this.sourceKeys.set(result,g);this.activeSources.add(result);this.compiled.push(result);
 }
 private retireSource(source:PhysicalSource):void{
  if(source.retired)return;source.retired=true;this.activeSources.delete(source);
  const key=this.sourceKeys.get(source),byIndex=key&&this.cache.get(key.positions);
  if(key&&byIndex?.get(key.indices)===source)byIndex.delete(key.indices);
  this.sourceKeys.delete(source);
  this.sourcePositions.retire(source.ranges.positions);this.sourceNormals.retire(source.ranges.normals);if(source.numeric)this.numerics?.retire(source.numeric);else{if(source.ranges.indices)this.sourceWords.retire(source.ranges.indices);this.sourceWords.retire(source.ranges.incidence);}
 }
 private frameSlot(retained:RetainedBatch,source:PhysicalSource,size:number):SourceFrameSlot{
  let slot=retained.slots.find(value=>!value.busy);
  if(!slot){
   const partial:GPUBuffer[]=[];
   try{
    const roots=this.device.createBuffer({label:'MAIN retained source roots',size:size*80,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});partial.push(roots);
    const params=this.device.createBuffer({label:'MAIN retained source pose parameters',size:48,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});partial.push(params);
    const group=this.device.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries:[ {buffer:params},source.positionsBinding,source.normalsBinding,retained.batch.storage!.vertices,retained.batch.storage!.attributes,{buffer:this.arena.bounds},{buffer:roots}].map((resource,binding)=>({binding,resource}))});
    const memory=new ArrayBuffer(size*80),parameterImage=new ArrayBuffer(48);
    new Float32Array(parameterImage).set(source.bounds,4);
    // Only the foliage-side consumer needs this chart today. Registration
    // supplies the class; non-foliage relations keep the old math path.
    new Float32Array(parameterImage)[7]=(source.analysis.styleWord&MATERIAL_FOLIAGE_FACES_BIT)!==0?1:0;
    slot={roots,params,group,boundBounds:this.arena.bounds,memory,floats:new Float32Array(memory),words:new Uint32Array(memory),parameterImage,parameters:new Uint32Array(parameterImage),busy:false};
    retained.slots.push(slot);for(const buffer of partial)this.buffers.add(buffer);
   }catch(error){for(const buffer of partial)buffer.destroy();throw error;}
  }
  if(slot.boundBounds!==this.arena.bounds){slot.group=this.device.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries:[{buffer:slot.params},source.positionsBinding,source.normalsBinding,retained.batch.storage!.vertices,retained.batch.storage!.attributes,{buffer:this.arena.bounds},{buffer:slot.roots}].map((resource,binding)=>({binding,resource}))});slot.boundBounds=this.arena.bounds;}
  slot.busy=true;this.leasedSlots.add(slot);return slot;
 }
 /** Retained mesh instances are posed DIRECTLY into their shared GPU batch.
  * There is no second per-owner vertex/attribute allocation or full-world
  * copy. CPU work is source residency and the changed root matrices only. */
 encode(encoder:GPUCommandEncoder,instances:readonly PhysicalSourceInstance[],clipFromWorld:Float32Array):readonly PhysicalResidentBatch[]{
  const groups=this.groups,lists=this.lists,live=this.live,changed=this.changed;
  if(groups.size){for(const list of groups.values()){list.length=0;lists.push(list);}groups.clear();}// an encode that threw
  for(const instance of instances){let list=groups.get(instance.domain);if(!list){list=lists.pop()??[];groups.set(instance.domain,list);}list.push(instance);}
  // Preflight all new ranges before opening a pass: arena growth copies the
  // unchanged bounds and every pose of this encode must bind the NEW buffer.
  const additional:number[]=[];
  for(const [domain,list]of groups){const g=list[0]!.geometry,source=this.acquire(g),count=source.geometry.positions.length/3,stride=count*16,maximum=Math.min(this.device.limits.maxStorageBufferBindingSize,this.device.limits.maxBufferSize),chunk=Math.max(1,Math.floor(maximum/Math.max(stride,80)));
   for(let offset=0,batch=0;offset<list.length;offset+=chunk,batch++){const size=Math.min(chunk,list.length-offset),old=this.batches.get(domain*BATCHES+batch);if(!old||old.mesh!==source||old.batch.storage!.vertices.size!==stride*size)additional.push(size);}
  }
  this.arena.reserve(encoder,additional);this.arena.begin();let cullChanged=false;
  const result:PhysicalResidentBatch[]=[],ranges:SourceInstanceRange[]=[];live.clear();this.inView=new Set();
  // ONE PASS FOR EVERY BATCH (frame cost, 2026-11-10): each batch's changed
  // roots precede one global cull in this pass. Dispatch ordering makes all
  // current bounds visible to that cull, instead of one
  // pass per batch (about 455 per Y8 frame).
  let shared:GPUComputePassEncoder|undefined;const pass=()=>shared??=encoder.beginComputePass({label:'MAIN retained mesh instances'});
  try{
  for(const [domain,list]of groups){
   const first=list[0]!,g=first.geometry,source=this.acquire(g);
   for(const item of list)if(item.geometry.positions!==g.positions||item.geometry.indices!==g.indices||item.geometry.revision!==g.revision)throw Error('Physical domain changed its immutable mesh within one frame');
   const count=source.geometry.positions.length/3,stride=count*16,maximum=Math.min(this.device.limits.maxStorageBufferBindingSize,this.device.limits.maxBufferSize);
   if(!count||Math.max(stride,80)>maximum)throw Error('Physical posed source requires device-sized geometry chunks');
   // A root is 80 bytes even for a three-vertex source. Every binding, not
   // just the posed vertex stream, must fit the same device-sized slice.
   const chunk=Math.max(1,Math.floor(maximum/Math.max(stride,80)));
   for(let offset=0,batch=0;offset<list.length;offset+=chunk,batch++){
    if(batch>=BATCHES)throw Error('Physical domain exceeds its retained batch addressing');
    const size=Math.min(chunk,list.length-offset),key=domain*BATCHES+batch,bytes=stride*size;live.add(key);let retained=this.batches.get(key);
    if(!retained||retained.mesh!==source||retained.batch.storage!.vertices.size!==bytes){
     if(retained)this.retireBatch(retained);
     const storageRanges=allocateStorageRanges({vertices:{storage:this.posedVertices,data:bytes},attributes:{storage:this.posedAttributes,data:bytes},metadata:{storage:this.catalog,data:new Uint32Array([g.indices.length,size,0,0,0,0,0,0])}});
     const storage={identity:{},vertices:this.posedVertices.binding(storageRanges.vertices),attributes:this.posedAttributes.binding(storageRanges.attributes),metadata:this.catalog.binding(storageRanges.metadata)};
     const range=this.arena.allocate(size),bounds=this.arena.bounds,visibleInstances=this.arena.visibility;
     retained={mesh:source,storageRanges,range,batch:{vertices:storage.vertices.buffer,attributes:storage.attributes.buffer,metadata:storage.metadata.buffer,storage,source:{vertexCapacity:count,indexBase:0,cullMode:1,instanceCapacity:size,bounds,visibleInstances,visibleInstanceOffset:range.base,physicalUniformProducerRows:true}},members:[],matrices:new Float32Array(size*16),revision:0,slots:[],settling:new Set()};this.batches.set(key,retained);
    }
    changed.length=0;
    for(let index=0;index<size;index++){
     const member=list[offset+index]!,matrix=member.geometry.worldFromSource??IDENTITY;let dirty=retained.members[index]!==member.geometry.owner;
     for(let k=0;k<16&&!dirty;k++)dirty=retained.matrices[index*16+k]!==matrix[k];
     if(dirty){changed.push(index);retained.settling.add(index);retained.members[index]=member.geometry.owner;retained.matrices.set(matrix,index*16);this.spatial.set(retained.range.base+index,instanceBounds(source.bounds,retained.matrices,index),{batch:retained,instance:index});}
     else retained.settling.delete(index);
    }
    let slot:SourceFrameSlot|undefined;
    if(changed.length){
     retained.revision++;
     slot=this.frameSlot(retained,source,size);const f=slot.floats,u=slot.words;
     for(let record=0;record<changed.length;record++){const index=changed[record]!,at=record*20;for(let k=0;k<16;k++)f[at+k]=retained.matrices[index*16+k]!;u[at+16]=list[offset+index]!.owner;u[at+17]=index;u[at+18]=0;u[at+19]=0;}
     const groups=Math.ceil(count*changed.length/64),columns=Math.min(groups,this.device.limits.maxComputeWorkgroupsPerDimension);
     slot.parameters.set([count,changed.length,columns,retained.range.base]);this.device.queue.writeBuffer(slot.roots,0,slot.memory,0,changed.length*80);this.device.queue.writeBuffer(slot.params,0,slot.parameterImage);
     const roots_=pass();roots_.setPipeline(this.pipeline);roots_.setBindGroup(0,slot.group);roots_.dispatchWorkgroups(columns,Math.ceil(groups/columns));
    }
    // Preserve the host's coarse in-view decision and exact held-view test.
    // A changed pose or camera requests the single global cull below; its
    // shared range is read directly by relations and contact packing.
    const held=retained.culledClip;let unchanged=!changed.length&&!!held;
    for(let k=0;k<16&&unchanged;k++)unchanged=held![k]===clipFromWorld[k];
    if(!unchanged){
     cullChanged=true;
     (retained.culledClip??=new Float32Array(16)).set(clipFromWorld);
    }
    // An unchanged source at an unchanged posed revision publishes the SAME
    // record: same domain, same `complex` object, same revision.
    if(retained.published?.batch.source.posedRevision!==retained.revision||retained.published.batch.source.visibleInstances!==this.arena.visibility){
     const data={id:source.id,domain,revision:g.revision,buffer:source.incidence,bufferBinding:source.incidenceBinding,geometry:{...source.geometry,positions:source.geometry.positions,indices:source.geometry.indices,normals:source.geometry.normals,revision:g.revision},...(source.partAnchorVertices?{partAnchorVertices:source.partAnchorVertices}:{})};
     const relation:RegisteredSourceComplex<SourceComplexRetention>=source.complex
      ?{...data,complex:source.complex,analysis:source.analysis}
      :{...data,summary:source.summary,analysis:source.analysis};
     retained.published={domain,source:relation,batch:{...retained.batch,source:{...retained.batch.source,bounds:this.arena.bounds,visibleInstances:this.arena.visibility,visibleInstanceOffset:retained.range.base,posedRevision:retained.revision}}};
    }
    result.push(retained.published);ranges.push(retained.range);
   }
  }
  // A removed range must leave the hierarchy BEFORE this camera's query.
  // Remaining posed/source ranges retain their original addresses and order.
  for(const [key,retained]of this.batches)if(!live.has(key)){this.retireBatch(retained);this.batches.delete(key);}
  clipPlanes(clipFromWorld,this.spatialPlanes);
  this.spatial.query(this.spatialPlanes,({batch,instance})=>{this.arena.includeInstance(batch.range,instance);this.inView.add(batch.published!);});
  this.arena.encode(pass,clipFromWorld,cullChanged);
  }finally{shared?.end();}
  // A retained job/retry keeps THIS view's immutable traversal list. A live
  // getter into the next camera's arena frontier would cross pose ownership.
  for(let i=0;i<result.length;i++){
   const base=result[i]!;if(!this.inView.has(base))continue;
   const frontier=this.arena.frontier(ranges[i]!);if(!frontier)throw Error('Visible physical batch has no complete instance frontier');
   const current={...base,batch:{...base.batch,source:{...base.batch.source,visibleInstanceFrontier:frontier}}};
   result[i]=current;this.inView.delete(base);this.inView.add(current);
  }
  // Membership belongs to prepare, not camera-dependent selection above.
  for(const list of groups.values()){list.length=0;lists.push(list);}groups.clear();
  return result;
 }
 /** Select the representation frontier from the already-posed octree. This
  * runs after roots/bounds are updated but before relations, contacts or runs;
  * the spatial index remains the culler and its `select` traversal supplies
  * the same node membership to the LOD compiler. */
 selectLod(clip:ArrayLike<number>,width:number,height:number,pageWidth:number,pageHeight:number,exterior:boolean,perspective:boolean,held:ReadonlySet<number>=new Set(),cache:ParentGroupCache=new ParentGroupCache(),smallAvailable?:SmallConstructionAvailability,heldSmall:ReadonlySet<string>=new Set(),nativePart?:SourceWholeNativePartAdmission):SourceLodSelection{
  const instances:SourceLodInstance[]=[],parents:SourceLodParent[]=[],profiles=new Map<number,boolean>();
  const toInstance=(entry:SpatialInstance):SourceLodInstance|undefined=>{
   const identity=parsePhysicalProducerOwner(entry.batch.members[entry.instance]??'');if(!identity)return;
   const {mesh,eid}=identity;profiles.set(mesh,entry.batch.mesh.profileSafe);
   const worldFromSource=entry.batch.matrices.subarray(entry.instance*16,entry.instance*16+16);
   return {eid,mesh,bounds:instanceBounds(entry.batch.mesh.bounds,entry.batch.matrices,entry.instance),settling:entry.batch.settling.has(entry.instance)||!entry.batch.mesh.profileSafe,sourceRevision:entry.batch.mesh.revision,poseRevision:entry.batch.revision,worldFromSource,construction:bindSourceWholeConstruction(entry.batch.mesh.wholeConstruction,mesh,eid,worldFromSource),smallParent:entry.batch.mesh.smallParent};
  };
  const adequate=(bounds:SpatialBounds)=>exterior&&perspective&&fitsParentTier(bounds,clip,width,height,pageWidth,pageHeight);
  const planes=new Float32Array(24);clipPlanes(clip as Float32Array,planes);
  this.spatial.select(planes,adequate,(bounds,values,key)=>{
   const members=values.map(toInstance).filter((value):value is SourceLodInstance=>!!value);
   // Physical/Y8 entries are producer rows. Their larger node representation
   // is the merged producer carrier; only a one-cell node may use the shared
   // far aggregate after its producer owner keys are preserved below. Entity-
   // shaped physical sources may use the shared parent selector directly.
   // SpatialIndex.select gives the complete node membership to this callback.
   // If any member is moving or lacks the compiled whole-form proof, expand
   // that membership here instead of handing an unsafe node to source-lod's
   // parent tier. No child is dropped; each member then takes its own full or
   // far decision below.
   const certified=members.length>1&&members.every(member=>!member.settling&&profiles.get(member.mesh)===true);
   // The shared form-depth parent flag now gates every downstream reader, so
   // certified producer nodes can use the same coherent parent representation
   // at the full parent tier; rejected groups fall back to their members.
   if(certified)parents.push({bounds,instances:members,key});else instances.push(...members);
  },value=>{const member=toInstance(value);if(member)instances.push(member);});
  return selectSourceLod(instances,clip,width,height,pageWidth,pageHeight,exterior,perspective,parents,held,mesh=>profiles.get(mesh)===true,eid=>eid,cache,false,smallAvailable,heldSmall,nativePart);
 }
 private retireBatch(retained:RetainedBatch):void{for(let i=0;i<retained.range.count;i++)this.spatial.delete(retained.range.base+i);this.arena.retire(retained.range);this.posedVertices.retire(retained.storageRanges.vertices);this.posedAttributes.retire(retained.storageRanges.attributes);this.catalog.retire(retained.storageRanges.metadata);for(const slot of retained.slots)this.retired.push(slot.roots,slot.params);}
 /** Host releases only at the native job's terminal submission fence, or
  * after discarding its encoder. The next live tick is not a release. */
 submitted():void{this.arena.submitted();this.numerics?.submitted();for(const store of [this.sourcePositions,this.sourceNormals,this.sourceWords,this.posedVertices,this.posedAttributes,this.catalog])store.submitted();for(const slot of this.leasedSlots)slot.busy=false;this.leasedSlots.clear();for(const buffer of this.retired.splice(0)){buffer.destroy();this.buffers.delete(buffer);}}
 dispose():void{this.disposed=true;this.demand=undefined;for(const job of this.work)job.resolve();this.compiler.dispose();this.work.clear();this.compiled.length=0;this.submitted();this.activeSources.clear();this.batches.clear();this.spatial.clear();this.arena.dispose();const stores=[this.sourcePositions,this.sourceNormals,this.sourceWords,this.posedVertices,this.posedAttributes,this.catalog];if(this.incidenceCompiler)void (async()=>{await this.numerics?.dispose();await this.incidenceCompiler!.dispose();for(const store of stores)store.dispose();})();else{void this.numerics?.dispose();for(const store of stores)store.dispose();}for(const buffer of this.buffers)buffer.destroy();this.buffers.clear();}
}
