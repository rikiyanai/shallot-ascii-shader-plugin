import nativeFont from './native-glyphs.json' with {type:'json'};
import {inverseMatrix,type CompositionGeometry} from './composition.ts';
import {NativeConstruction,type ConstructionMetrics} from './native-construction.ts';
import {NativePresentation,type PresentedNativeFrame} from './native-presentation.ts';
import {NativePublication} from './publication.ts';
import {constructionSupports} from './construction-support.ts';
import {FieldReadback,ResidentConstructionField,type ConstructionFieldReadback,type DrawingFieldView} from './field-readback.ts';
import type {SurfaceField,SurfaceAttachment} from './surface-fields.ts';
import {formDepths,type FormParentGroup,type FormSourceBound} from './form-depths.ts';
import {FormLodLifetimes} from './form-lod-lifetimes.ts';
import type {NativeLabel} from './semantic-text.ts';
import type {PresentationMode} from './text/text-frame.ts';
import {PhysicalSourceTransport,type PhysicalSourceInstance,type PhysicalResidentBatch} from './physical-source.ts';
import {preparePhysicalGpuServices,type PhysicalGpuServices} from './gpu-startup.ts';
import type {ComputePipelineProgress} from './gpu-pipeline-preparation.ts';
import {SurfaceRelations,posedBatchIdentity,type SurfaceGraphView,type SourcePartAnchorTable,type PartAnchorDemand,type PartAnchorDemandReadback} from './surface-relations.ts';
import {RELATION_WORDS} from './source-coordinates.ts';
import {GpuArticulation,type ArticulationResult,type ArticulationLiteral,type ArticulationLiteralFeed} from './gpu-articulation.ts';
import type {SourceConstructionClass} from './gpu-construction-runs.ts';
import {toneField,TONE_ROW,type ToneReadback,type ToneMember,type TonedFieldView,type GpuToneMembers,type ToneSunShadow} from './tone-field.ts';
import {RelationVisibility,type ProjectedRelationReadback} from './relation-visibility.ts';
import {decodePosedGraph,type PosedGraph} from './posed-graph.ts';
import {TextureBlit} from './texture-blit.ts';
import {ConstructionMask} from './construction-mask.ts';
import {compileSourceFaceMaterialClasses} from './source-material.ts';
import type {MeshSourceComplex} from './source-complex.ts';
import {sourceResidencyEdgeCount} from './source-transfer.ts';
import {exportResidentFrame,type ResidentFrameState,type ResidentNativeSnapshot,residentFrameState} from './resident-native-frame.ts';
import {ParentGroupCache,ParentHolds,remapSourceWholeConstruction,type SourceLodSelection} from './source-lod.ts';
import type {SourceFarForms} from './gpu-source-far-forms.ts';
import {discardPartAnchorAttempt} from './part-anchor-replay.ts';

/** Retain this exact pose for deferred/relabelled native playback. The
 * direct snapshot pack reads the retained table as storage, not a copy-only
 * staging buffer. The job remains the buffer's lifetime owner. */
export function retainPhysicalPartAnchors(device:GPUDevice,encoder:GPUCommandEncoder,anchors:SourcePartAnchorTable):SourcePartAnchorTable{
 const buffer=device.createBuffer({label:'MAIN physical pose declared anchors',size:anchors.buffer.size,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
 encoder.copyBufferToBuffer(anchors.buffer,0,buffer,0,buffer.size);
 return {...anchors,buffer};
}

export interface PhysicalGeometry extends CompositionGeometry {
 /** Source-revision bounds supplied at asset registration; camera admission
  * transforms eight corners, never scans vertices on its frame path. */
 readonly sourceBounds?:readonly [number,number,number,number,number,number];
 /** Visibility flags only: source triangles and lineage remain complete. */
 readonly cameraCleared?:Uint8Array;
 readonly albedo?:readonly [number,number,number];
}




export function formSourceBoundOf(geometry:PhysicalGeometry,owner:readonly [number,number]):FormSourceBound[]{
 const b=geometry.sourceBounds,m=geometry.worldFromSource;
 if(!b||!m||m.length<16||geometry.poseAuthority!=='source-model')return [];
 const low=[Infinity,Infinity,Infinity],high=[-Infinity,-Infinity,-Infinity];
 for(let corner=0;corner<8;corner++){
  const x=corner&1?b[3]:b[0],y=corner&2?b[4]:b[1],z=corner&4?b[5]:b[2];
  for(let axis=0;axis<3;axis++){const v=m[axis]!*x+m[4+axis]!*y+m[8+axis]!*z+m[12+axis]!;low[axis]=Math.min(low[axis]!,v);high[axis]=Math.max(high[axis]!,v);}
 }
 if(![...low,...high].every(Number.isFinite))return [];
 return [{owner,low:[low[0]!,low[1]!,low[2]!],high:[high[0]!,high[1]!,high[2]!]}];
}
export interface PhysicalFrameInput {
 readonly encoder:GPUCommandEncoder;readonly target?:GPUTextureView;
 readonly targetWidth:number;readonly targetHeight:number;
 readonly width:number;readonly height:number;readonly engineFrame:number;
 /** Native WebGPU projection: positive-up clip Y and reversed [0,1] depth. */
 readonly clipFromWorld:Float32Array;readonly worldFromClip:Float32Array;
 readonly pose:NonNullable<DrawingFieldView['pose']>;
 readonly geometry:readonly PhysicalGeometry[];readonly labels?:readonly NativeLabel[];
 readonly mode:PresentationMode;
 /** Exterior admission is explicit. An unknown/interior composition keeps the
  * incumbent complete source; the Y8 host passes true for its open world. */
 readonly exterior?:boolean;
 readonly light?:readonly [number,number,number];
 /** The host engine's sun caster for this frame, in the Shallot SunShadow
  * layout plugin.ts passes (sunShadowView / sunShadowParams / shadowSampler).
  * Without it region tone has n.L bands only, no cast shadow. */
 readonly sunShadow?:Pick<ToneSunShadow,'view'|'sampler'|'params'>;
 /** The host's completed same-pose beauty target, when the adapter owns it. */
 readonly appearance?:GPUTexture;
}
/** Prepared inputs used to decide whether the next host step needs a canvas
 * presentation ticket. They intentionally exclude the per-step encoder,
 * target and engine frame number: readiness is decided before BeginFrame can
 * acquire a canvas texture. */
export type PhysicalPresentationInput=Omit<PhysicalFrameInput,'encoder'|'target'|'engineFrame'>;
export interface PhysicalTextFrame extends PresentedNativeFrame {
 readonly construction:{readonly sourceLines:number;readonly omitted:readonly string[];readonly pending?:readonly string[];readonly metrics:ConstructionMetrics};
}
interface Targets {width:number;height:number;field:GPUTexture[];depth:GPUTexture;comparison?:GPUTexture;mask:GPUTexture;inspection?:GPUTexture}

// MATRIX RESOLUTION THE RESOLUTION?"). Every field consumer reads it through
// the field extent (tone toField, articulation fieldPoint, visibility and the
// form table by their texture size): one sample per CELL_STEP page pixels
// across (the narrowest native advance, '.' ',' ':' 'i' 'l', is 3 px) and
// ROW_SAMPLES samples down each TONE_ROW-pixel text row (tone reads three
// heights per row). 3488x2252 is 1163x399 samples, not 7.85 M pixels.
const CELL_STEP=3,ROW_SAMPLES=3;
function fieldExtent(width:number,height:number):[number,number]{return [Math.ceil(width/CELL_STEP),Math.ceil(height/TONE_ROW)*ROW_SAMPLES];}
/** The field channels and their depth: written and read by one text job only. */
interface FieldSet {width:number;height:number;field:GPUTexture[];depth:GPUTexture}
interface SourceBuffers {positions:Float32Array;indices:Uint32Array;revision:number;vertex:GPUBuffer;index:GPUBuffer;vertexBinding:GPUBufferBinding;indexBinding:GPUBufferBinding;flags:GPUBuffer;clearance?:Uint8Array;classes?:Uint32Array;attachment?:SurfaceAttachment;
 binding?:{uniform:GPUBuffer;group:GPUBindGroup};
}
interface Job {
 played?:Promise<ArticulationResult>;
 toneLiterals?:{resolve:(literals:readonly ArticulationLiteral[],tone?:GpuToneMembers)=>void;reject:(error:unknown)=>void};
 toneMembers?:readonly ToneMember[];
 sunShadow?:Pick<ToneSunShadow,'view'|'sampler'|'params'>;
 serial:number;engineFrame:number;targets:Targets;read:ConstructionFieldReadback;staging?:GPUBuffer;relationReadback?:ProjectedRelationReadback;tone?:ToneReadback;
 components:ReadonlyMap<number,MeshSourceComplex>;view:DrawingFieldView;geometry:readonly PhysicalGeometry[];
 labels:readonly NativeLabel[];mode:PresentationMode;cpuEncodeMs:number;
 sourceView:SurfaceGraphView;sourceField:SurfaceField;
 /** Same-pose anchor table retained with tone for label-only publication. */
 partAnchorTable?:SourcePartAnchorTable;
 partAnchorDemand?:PartAnchorDemandReadback;
 sources:readonly PhysicalResidentBatch[];inView:ReadonlySet<PhysicalResidentBatch>;
 capacity:number;intervalCapacity:number;field?:DrawingFieldView['field'];
 retry?:{relations:number;intervals:number;triangles?:number;anchor?:PartAnchorDemand;standingOwnerDemand?:number};
 /** The full-resolution inspection field and D10 image, while an inspection view is on. */
 inspection?:{field:SurfaceField;target:GPUTexture};
 /** This job's field slot (and inspection slot): read by its own same-pose
  * retries after later jobs have started, so never shared with them. */
 fieldSet?:FieldSet;inspectionSet?:FieldSet;
 /** The posed-source state this job encoded (a label-only job shares its base's). */
 pose?:object;
 /** Terminal: the sealed snapshot to publish in serial order, or none. */
 settled?:boolean;snapshot?:ResidentNativeSnapshot;
 /** In-view sources still compiling when this job started (admit). The
  * published frame reports them; each joins the first job after it is ready. */
 pending?:readonly CompositionGeometry[];
 /** The same-view representation frontier consumed by articulation. */
 lod?:SourceLodSelection;
}
interface DisplayedResident {snapshot:ResidentNativeSnapshot;targets:Targets;}
interface DisplayedFrame {frame?:PhysicalTextFrame;resident?:DisplayedResident;targets:Targets;}
interface RetirementBatch {buffers:GPUBuffer[];targets:Targets[];snapshots:ResidentNativeSnapshot[];fields:FieldSet[];outputs:GPUTexture[]}
/** Original-scene pixels are an explicit inspection/presentation choice.
 * MAIN's normal ASCII page must not fall back to the host beauty image while
 * text is pending or when a previous inspection pair is still retained. */
export function originalScenePresentationRequested(flags:Pick<PhysicalFrame['controller'],'ascii'|'underlay'|'overlay'>):boolean{
 return !flags.ascii||flags.underlay||flags.overlay;
}
// Field channels: depth, oct normal, UV, owner, colour. The physical source
// shader writes UV as 0 and no reader samples it (the attachment keeps the
// field layout's indices): one byte, not eight (memory, 2026-11-12).
const FORMATS:GPUTextureFormat[]=['r32float','rg16float','r8unorm','rg32uint','rgba16float'];
const IDENTITY=new Float32Array([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]);
/** The source owners a job's frame reports pending (still compiling at its start). */
function pendingOwners(job:Pick<Job,'pending'>):readonly string[]{return (job.pending??[]).map(source=>source.owner);}
/** How far the sun turns before the drawing takes the new sun (holdLight). */
const LIGHT_STEP_DEGREES=3;
const PHYSICAL=/* wgsl */`
struct Parameters {clip:mat4x4f,world:mat4x4f,identity:vec4u,albedo:vec4f,eye:vec4f,light:vec4f}
@group(0) @binding(0) var<uniform> params:Parameters;
@group(0) @binding(1) var<storage,read> positions:array<vec4f>;
@group(0) @binding(2) var<storage,read> indices:array<u32>;
@group(0) @binding(3) var<storage,read> flags:array<u32>;
struct V {@builtin(position) clip:vec4f,@location(0) world:vec3f,@location(1) @interpolate(flat) normal:vec3f,@location(2) @interpolate(flat) flag:u32}
@vertex fn vertex(@builtin(vertex_index) index:u32)->V{
 let first=index/3u*3u;
 let a=(params.world*positions[indices[first]]).xyz;
 let b=(params.world*positions[indices[first+1u]]).xyz;
 let c=(params.world*positions[indices[first+2u]]).xyz;
 let world=(params.world*positions[indices[index]]).xyz;
 var clip=params.clip*vec4f(world,1);
 let n=cross(b-a,c-a);
 if((flags[first/3u]&255u)==1u||length(n)<1e-20){clip=vec4f(2,2,2,1);}
 return V(clip,world,n/max(length(n),1e-20),flags[first/3u]);
}
struct F {@location(0) depth:f32,@location(1) normal:vec2f,@location(2) uv:vec2f,@location(3) owner:vec2u,@location(4) color:vec4f}
fn oct(value:vec3f)->vec2f{
 let n=value/max(dot(abs(value),vec3f(1)),1e-20);
 return select((vec2f(1)-abs(n.yx))*select(vec2f(-1),vec2f(1),n.xy>=vec2f(0)),n.xy,n.z>=0);
}
@fragment fn fragment(v:V)->F{
 if((v.flag&255u)==2u&&distance(v.world,params.eye.xyz)<.4){discard;}
 var normal=normalize(v.normal);
 if(dot(normal,params.eye.xyz-v.world)<0){normal=-normal;}
 let diffuse=max(0,dot(normal,normalize(params.light.xyz)));
 let light=.3+.7*floor(diffuse*3.0+.5)/3.0;
 // Location 2 (r8unorm): the face's authored material class / 255 (tone).
 return F(v.clip.z,oct(normal),vec2f(f32((v.flag>>8u)&15u)/255.0,0),params.identity.xy,vec4f(params.albedo.xyz*light,1));
}`;
const BLIT=/* wgsl */`
struct Params {source:vec2f,outputExtent:vec2f,unscaled:vec4f}
@group(0) @binding(0) var image:texture_2d<f32>;
@group(0) @binding(1) var<uniform> params:Params;
@vertex fn vertex(@builtin(vertex_index) i:u32)->@builtin(position) vec4f{
 return vec4f(f32(i&1u)*4.0-1.0,f32(i>>1u)*4.0-1.0,0,1);
}
@fragment fn fragment(@builtin(position) p:vec4f)->@location(0) vec4f{
 // unscaled: the source pixel under the target pixel, paper beyond it.
 if(params.unscaled.x>.5){if(any(p.xy>=params.source)){return vec4f(1);}let color=textureLoad(image,vec2i(p.xy),0);return select(vec4f(1),color,color.a>0);}
 let at=min(vec2i(params.source)-vec2i(1),vec2i(p.xy/params.outputExtent*params.source));
 let color=textureLoad(image,at,0);return select(vec4f(1),color,color.a>0);
}`;

/** Engine-neutral physical source adapter. It encodes only into the host's
 * current transaction. Native composition and publication are the SAME core
 * used by the standard Shallot collector; no old glyph buffer is consumed. */
export class PhysicalFrame {
 private readonly lodLifetimes=new FormLodLifetimes();
 private readonly parentHolds=new ParentHolds();
 private readonly parentGroups=new ParentGroupCache();
 readonly controller={ascii:true,underlay:false,overlay:false,paused:false,pending:false,mode:'text' as PresentationMode,
  frames:new Map<number,PhysicalTextFrame>(),residentFrames:new Map<number,ResidentFrameState>(),graphs:new Map<number,readonly PosedGraph[]>(),errors:[] as string[],pause:(_paused:boolean)=>{},onState:()=>{}};
 private readonly construction=new NativeConstruction(nativeFont.glyphs,'preformatted');
 private readonly playback?:GpuArticulation;
 private readonly gpuServices?:PhysicalGpuServices;
 private readonly presentation=new NativePresentation(nativeFont.glyphs,nativeFont.fontSha256);
 private readonly publication:NativePublication<PhysicalTextFrame>;
 private readonly pipeline:GPURenderPipeline;private readonly blit:GPURenderPipeline;
 private readonly fieldLayout:GPUBindGroupLayout;
 private fieldUniforms?:{buffer:GPUBuffer;memory:Float32Array;words:Uint32Array};
 private readonly textureBlit:TextureBlit;private readonly mask:ConstructionMask;
 private readonly transport:PhysicalSourceTransport;private readonly relations:SurfaceRelations;private readonly visibility:RelationVisibility;
 private readonly sources=new Map<string,SourceBuffers>();
 private readonly resources=new Set<Targets>();private readonly waiting=new Set<{resolve:(frame:PhysicalTextFrame)=>void;reject:(error:Error)=>void}>();
 private readonly retiredBuffers:GPUBuffer[]=[];private readonly retiredTargets:Targets[]=[];
 private readonly retiredSnapshots:ResidentNativeSnapshot[]=[];
 // encode transfers retirement work to a host transaction. The host may
 // discard its callback, so that closure must never be the sole owner.
 private readonly retirementBatches=new Set<RetirementBatch>();
 private lastEncode?:RetirementBatch;
 private shownSnapshot?:ResidentNativeSnapshot;
 private registration?:Promise<void>;private serial=0;private displayed?:DisplayedFrame;
 // THE DRAWING'S SUN STEPS (crawl, 2026-11-11). A game's day cycle turns the
 // sun a little every frame (Y8: about 0.2 degrees a second). Every text
 // decision that reads the light (n.L bands, tone regions, shade sides) was
 // re-made from the new sun each frame, so a stationary view crawled: camera,
 // poses and vertices identical across 29 jobs, 2-7% of the ink changed every
 // half second. The drawing keeps the sun it was made with until the sun has
 // turned LIGHT_STEP_DEGREES, then takes the new one: a shadow side moves in
 // deliberate steps, as cascaded shadow maps snap to texels.
 private held?:readonly [number,number,number];
 private holdLight(light:readonly [number,number,number]|undefined):readonly [number,number,number]|undefined{
  if(!light){this.held=undefined;return undefined;}
  const held=this.held,length=(v:readonly number[])=>Math.hypot(v[0]!,v[1]!,v[2]!)||1;
  if(held){const cosine=(held[0]*light[0]+held[1]*light[1]+held[2]*light[2])/(length(held)*length(light));if(cosine>=Math.cos(LIGHT_STEP_DEGREES*Math.PI/180))return held;}
  this.held=[light[0],light[1],light[2]];return this.held;
 }
 // TWO TEXT JOBS IN FLIGHT (speed, 2026-11-17; ownership table in

 // Job N+1 is encoded and submitted while job N's GPU work or header map is
 // still pending. Publication stays after the header seal and in serial order
 // (drain): a later serial is never displayed before an earlier one, and a
 // complete later snapshot waits for an earlier job's same-pose retry.
 // Shared buffers a job reads after its own host encoder (the field set and the
 // inspection field) have one slot per in-flight job; every other shared buffer
 // is consumed by ordered passes of the job's own encoder. A SOURCE-level retry
 // replays its exact pose only while the shared posed batches still hold that
 // pose (posed); once a newer job has re-posed them, the retry is superseded:
 // its measured extents still grow capacity and the newer pose publishes.
 private static readonly IN_FLIGHT=2;
 private readonly inflight:Job[]=[];
 private posed?:object;
 private growth?:{relations:number;intervals:number;triangles?:number};
 // LABEL-ONLY JOBS: the last full job whose relations, D10 image and tone the
 // retained buffers still hold, with its sorted tone members (relabel).
 private kept?:{job:Job;tone:ToneReadback};private lastStartedJob?:Job;private readonly relabels=new WeakSet<Job>();
 // STILL VIEW (frame cost, 2026-11-12). The inputs of the last started text
 // job. When every one of them is EXACTLY equal again (no tolerance): the clip
 // matrix and pose, page size, mode, held light (holdLight), labels, the
 // host's appearance target and every source in order (owner, revision, mesh
 // arrays, worldFromSource, camera-clearance flags, albedo), a new job would
 // draw the same page from the same sources. The host target's extent and the
 // Rows setting are in the key too: a resize or a Rows change always starts a
 // job, never re-presents the old page. None starts; the displayed tape
 // keeps being presented, and the GPU does only that presentation. A failed
 // job clears this, so a failed view retries every frame as before.
 private started?:{clip:Float32Array;eye:readonly number[];forward:readonly number[];focal:number;perspective:boolean;exterior?:boolean;width:number;height:number;targetWidth:number;targetHeight:number;rows:unknown;inspect:boolean;mode:PresentationMode;
  light?:readonly [number,number,number];labels:string;appearance?:GPUTexture;
  geometry:{source:PhysicalGeometry;world?:Float32Array;cleared?:Uint8Array;albedo?:readonly number[]}[];
  pending:readonly CompositionGeometry[]};
 private startKey(input:PhysicalFrameInput,light:readonly [number,number,number]|undefined,pending:readonly CompositionGeometry[]=[]):NonNullable<PhysicalFrame['started']>{
  return {pending,clip:input.clipFromWorld.slice(),eye:[...input.pose.eye],forward:[...input.pose.forward],focal:input.pose.focalPixels,perspective:input.pose.perspective,exterior:input.exterior,width:input.width,height:input.height,targetWidth:input.targetWidth,targetHeight:input.targetHeight,rows:(this.controller as {rows?:unknown}).rows,inspect:originalScenePresentationRequested(this.controller),mode:input.mode,
   light,labels:JSON.stringify(input.labels??[]),appearance:input.appearance,
   geometry:input.geometry.map(source=>({source,world:source.worldFromSource?.slice(),cleared:source.cameraCleared?.slice(),albedo:source.albedo&&[...source.albedo]}))};
 }
 private still(input:PhysicalPresentationInput,ignoreLabels=false):boolean{
  const last=this.started;if(!last||!this.displayed)return false;
  // A source the displayed page reports pending joins once it is ready.
  if(last.pending.some(source=>this.transport.isReady(source)))return false;
  if(this.holdLight(input.light)!==last.light)return false;
  if(last.width!==input.width||last.height!==input.height||last.targetWidth!==input.targetWidth||last.targetHeight!==input.targetHeight||last.rows!==(this.controller as {rows?:unknown}).rows||last.inspect!==originalScenePresentationRequested(this.controller)||last.mode!==input.mode||last.appearance!==input.appearance||last.perspective!==input.pose.perspective||last.exterior!==input.exterior||last.focal!==input.pose.focalPixels)return false;
  const same=(a:ArrayLike<number>|undefined,b:ArrayLike<number>|undefined)=>{if(a===b)return true;if(!a||!b||a.length!==b.length)return false;for(let i=0;i<a.length;i++)if(a[i]!==b[i])return false;return true;};
  if(!same(last.clip,input.clipFromWorld)||!same(last.eye,input.pose.eye)||!same(last.forward,input.pose.forward))return false;
  if(last.geometry.length!==input.geometry.length)return false;
  for(let i=0;i<input.geometry.length;i++){
   const g=input.geometry[i]!,k=last.geometry[i]!,s=k.source;
   if(g.owner!==s.owner||g.revision!==s.revision||g.positions!==s.positions||g.indices!==s.indices||g.surface!==s.surface||g.primary!==s.primary||g.index!==s.index||g.normals!==s.normals||g.lineage!==s.lineage)return false;
   if(!same(g.worldFromSource,k.world)||!same(g.cameraCleared,k.cleared)||!same(g.albedo,k.albedo))return false;
  }
  return ignoreLabels||JSON.stringify(input.labels??[])===last.labels;
 }
 /** The host supplies build-time source decisions; registration still owns
  * demand, revision identity, numeric readiness and the dynamic producer. */
 setSourceSemanticResolver(resolver:Parameters<PhysicalSourceTransport['setSemanticResolver']>[0]):void{
  if(this.disposed)throw Error('MAIN physical renderer is disposed');
  this.transport.setSemanticResolver(resolver);
 }
 /** Decide the engine ticket before BeginFrame can acquire a canvas target.
  * A ready tape that is not the currently presented pair is still a reason to
  * acquire once; a stable presented pair is not. Source work and simulation
  * continue in either case. */
 readyPresentation(input:PhysicalPresentationInput):boolean{
  if(this.disposed)throw Error('MAIN physical renderer is disposed');
  if((this.presentationRequestEpoch??0)>(this.presentationConsumedEpoch??0))return true;
  if(this.waiting.size)return true;
  const shown=this.presented,displayed=this.displayed;
  if(!shown||!displayed||displayed.targets!==shown.targets)return true;
  if(displayed.resident?.snapshot!==shown.snapshot||displayed.frame!==shown.frame)return true;
  const overlay=this.controller.ascii&&(this.controller.underlay||this.controller.overlay);
  if(shown.ascii!==this.controller.ascii||shown.overlay!==overlay)return true;
  return !this.still(input);
 }
 /** Request one fresh canvas publication without changing current-page export. */
 requestPresentation():void{
  if(this.disposed)throw Error('MAIN physical renderer is disposed');
  this.presentationRequestEpoch=(this.presentationRequestEpoch??0)+1;
 }
 private meshOnly?:Targets;
 private spareTargets?:Targets;
 private disposed=false;private sink:((frame:PhysicalTextFrame)=>void)|null=null;
 // A request can arrive while the fence for the previous encode is pending.
 // Keep request ownership monotonic so that completing that older fence only
 // consumes the epoch it captured, never a request made for the next frame.
 private presentationRequestEpoch=0;
 private presentationConsumedEpoch=0;
 static async create(device:GPUDevice,format:GPUTextureFormat,progress?:ComputePipelineProgress):Promise<PhysicalFrame>{
  const compiled=typeof location!=='undefined'&&new URL(location.href).searchParams.get('grammar')!=='legacy';
  const playback=compiled?await GpuArticulation.create(device,nativeFont.glyphs,'preformatted',8192,(p,stage)=>progress?.(p*2/3,stage)):undefined;
  let services:PhysicalGpuServices|undefined;
  try{services=await preparePhysicalGpuServices(device,compiled,(p,stage)=>progress?.((compiled?2/3:0)+p*(compiled?1/3:1),stage));return new PhysicalFrame(device,format,{playback,services});}
  catch(error){services?.dispose();playback?.dispose();throw error;}
 }
 constructor(private readonly device:GPUDevice,private readonly format:GPUTextureFormat,prepared?:{readonly playback?:GpuArticulation;readonly services?:PhysicalGpuServices}){
  if(prepared)this.playback=prepared.playback;
  else if(typeof location!=='undefined'&&new URL(location.href).searchParams.get('grammar')!=='legacy')throw Error('Compiled PhysicalFrame requires await PhysicalFrame.create()');
  // MEASUREMENT ONLY (T2(a) sizing, 2026-11-12): called by a review tool, never per frame.
  Object.assign(this.controller,{instanceCensus:()=>this.instanceCensus()});
  Object.defineProperty(this.controller,'pending',{enumerable:true,get:()=>!this.disposed&&((this.controller.ascii&&!this.presented)||this.inflight.length>0||!!this.registration)});
  Object.defineProperty(this.controller,'pendingStage',{enumerable:true,get:()=>this.registration?'Compiling visible sources':this.inflight.length?'Building text frame':!this.presented?'Waiting for first text frame':undefined});
  this.controller.pause=paused=>{this.controller.paused=paused;};
  this.gpuServices=prepared?.services;
  this.transport=this.gpuServices?.transport??new PhysicalSourceTransport(device,'gpu-csr',this.playback?'summary':'full');this.relations=this.gpuServices?.relations??new SurfaceRelations(device,32768,!!this.playback);this.visibility=this.gpuServices?.visibility??new RelationVisibility(device,32768,131072,false);
  const module=device.createShaderModule({label:'MAIN physical source triangles',code:PHYSICAL});
  this.fieldLayout=device.createBindGroupLayout({label:'MAIN retained physical field bindings',entries:[
   {binding:0,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:'uniform',hasDynamicOffset:true,minBindingSize:192}},
   ...[1,2,3].map(binding=>({binding,visibility:GPUShaderStage.VERTEX,buffer:{type:'read-only-storage' as const}}))]});
  this.pipeline=device.createRenderPipeline({layout:device.createPipelineLayout({bindGroupLayouts:[this.fieldLayout]}),vertex:{module,entryPoint:'vertex'},fragment:{module,entryPoint:'fragment',targets:FORMATS.map(format=>({format}))},primitive:{topology:'triangle-list',cullMode:'none'},depthStencil:{format:'depth32float',depthCompare:'greater',depthWriteEnabled:true}});
  const image=device.createShaderModule({label:'MAIN native presentation magnification',code:BLIT});
  this.blit=device.createRenderPipeline({layout:'auto',vertex:{module:image,entryPoint:'vertex'},fragment:{module:image,entryPoint:'fragment',targets:[{format}]}});
  this.publication=new NativePublication(device,nativeFont.glyphs);
  this.textureBlit=new TextureBlit(device);
  this.mask=this.gpuServices?.mask??new ConstructionMask(device);
 }
 // ONE FIELD SLOT PER IN-FLIGHT JOB (memory 2026-11-12, speed 2026-11-17). A
 // text job writes and reads its field (depth, normal, UV, owner, colour and
 // the depth test) until it completes, including its same-pose retries in later
 // host encoders; presentation reads only the pair's comparison, underdrawing
 // and output. At most IN_FLIGHT jobs are in flight, so the ring holds at most
 // that many sets of one extent (cell resolution: ~13.5 MB at 3488x2252). A set
 // of another extent retires at this frame's fence once no job holds it.
 private readonly fieldRing:FieldSet[]=[];private readonly retiredFields:FieldSet[]=[];
 private readonly inspectionRing:FieldSet[]=[];
 private texture(width:number,height:number,format:GPUTextureFormat):GPUTexture{return this.device.createTexture({size:[width,height],format,usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_SRC});}
 private slot(ring:FieldSet[],held:ReadonlySet<FieldSet|undefined>,width:number,height:number):FieldSet{
  const free=ring.find(set=>!held.has(set)&&set.width===width&&set.height===height);if(free)return free;
  for(const set of ring.filter(set=>!held.has(set)&&(set.width!==width||set.height!==height))){ring.splice(ring.indexOf(set),1);this.retiredFields.push(set);}
  const set={width,height,field:FORMATS.map(format=>this.texture(width,height,format)),depth:this.texture(width,height,'depth32float')};ring.push(set);return set;
 }
 private fields(width:number,height:number):FieldSet{return this.slot(this.fieldRing,new Set(this.inflight.map(job=>job.fieldSet)),width,height);}
 /** Inspection slots no in-flight job holds retire at this frame's fence. */
 private retireInspection():void{const held=new Set(this.inflight.map(job=>job.inspectionSet));for(const set of this.inspectionRing.filter(set=>!held.has(set))){this.inspectionRing.splice(this.inspectionRing.indexOf(set),1);this.retiredFields.push(set);}}
 private destroyFields(set:FieldSet):void{formDepths(this.device).release(set.field);set.field.forEach(t=>t.destroy());set.depth.destroy();}
 /** How many instances of the in-view batches the last job's relation stages
  * ran for, and how many may be in view (conservative corner test). */
 private lastCensus?:{geometry:readonly PhysicalGeometry[];clip:Float32Array;inView:ReadonlySet<number>;edges:ReadonlyMap<number,number>;width:number;height:number};
 private instanceCensus():object|undefined{
  const c=this.lastCensus;if(!c)return undefined;
  const sizeBands:Record<string,{instances:number;faces:number;edges:number}>={};const bounds=new Map<Float32Array,number[]>();let instances=0,visible=0,faceThreads=0,faceVisible=0,edgeThreads=0,edgeVisible=0;
  for(const g of c.geometry){
   const match=/^camera-0\/(\d+)\//.exec(g.owner);if(!match)continue;const domain=Number(match[1]);if(!c.inView.has(domain))continue;
   let b=bounds.get(g.positions);if(!b){b=[Infinity,Infinity,Infinity,-Infinity,-Infinity,-Infinity];for(let i=0;i<g.positions.length;i+=3)for(let k=0;k<3;k++){b[k]=Math.min(b[k]!,g.positions[i+k]!);b[k+3]=Math.max(b[k+3]!,g.positions[i+k]!);}bounds.set(g.positions,b);}
   const m=g.worldFromSource??IDENTITY,faces=g.indices.length/3,edges=c.edges.get(domain)??faces*1.5;let outside=127;const box=[Infinity,Infinity,-Infinity,-Infinity];let near=false;
   for(let corner=0;corner<8;corner++){
    const x=corner&1?b[3]!:b[0]!,y=corner&2?b[4]!:b[1]!,z=corner&4?b[5]!:b[2]!;
    const w=[0,1,2].map(k=>m[k]!*x+m[4+k]!*y+m[8+k]!*z+m[12+k]!);
    const q=[0,1,2,3].map(k=>c.clip[k]!*w[0]!+c.clip[4+k]!*w[1]!+c.clip[8+k]!*w[2]!+c.clip[12+k]!);
    if(q[3]!<=1e-6)near=true;else{const sx=(q[0]!/q[3]!*.5+.5)*c.width,sy=(.5-q[1]!/q[3]!*.5)*c.height;box[0]=Math.min(box[0]!,sx);box[1]=Math.min(box[1]!,sy);box[2]=Math.max(box[2]!,sx);box[3]=Math.max(box[3]!,sy);}
    const planes=[q[0]!+q[3]!,q[3]!-q[0]!,q[1]!+q[3]!,q[3]!-q[1]!,q[2]!,q[3]!-q[2]!];let mask=0;for(let k=0;k<6;k++)if(planes[k]!<0)mask|=1<<k;outside&=mask;
   }
   const seen=outside===0;instances++;faceThreads+=faces;edgeThreads+=edges;if(seen){visible++;faceVisible+=faces;edgeVisible+=edges;
    const extent=near?Math.max(c.width,c.height):Math.max(Math.min(box[2]!,c.width)-Math.max(box[0]!,0),Math.min(box[3]!,c.height)-Math.max(box[1]!,0),0);
    const rows=extent/17,band=[1,2,4,8,16].findIndex(limit=>rows<=limit),key=band<0?'>16':'<='+[1,2,4,8,16][band];
    const e=sizeBands[key]??={instances:0,faces:0,edges:0};e.instances++;e.faces+=faces;e.edges+=edges;}
  }
  return {instances,visible,faceThreads,faceVisible,edgeThreads,edgeVisible,culledShareOfFaceThreads:faceThreads?1-faceVisible/faceThreads:0,page:[c.width,c.height],sizeBands};
 }
 private targets(width:number,height:number):Targets{
  const [fieldWidth,fieldHeight]=fieldExtent(width,height),{field,depth}=this.fields(fieldWidth,fieldHeight);
  // The pair the last completed job replaced is retired in THIS encode and
  // no command of this frame presents it: the new job takes it now (the
  // displayed pair plus one per in-flight job). Its earlier reads were
  // earlier submissions. An in-flight job's pair is never retired.
  const now=this.retiredTargets.findIndex(target=>target.width===width&&target.height===height&&target!==this.displayed?.targets&&target!==this.meshOnly);
  if(now>=0){const [target]=this.retiredTargets.splice(now,1);target!.field=field;target!.depth=depth;return target!;}
  const spare=this.spareTargets;this.spareTargets=undefined;
  if(spare){if(spare.width===width&&spare.height===height){spare.field=field;spare.depth=depth;return spare;}this.destroy(spare);}
  const result:Targets={width,height,field,depth,mask:this.texture(fieldWidth,fieldHeight,'rgba8unorm')};
  this.resources.add(result);return result;
 }
 /** A presentation pair only: the shared field set is retired by fields(). */
 // A 1x1 paper image where a pair has no comparison (no view can show one).
 private paperTexture?:GPUTexture;
 private paper():GPUTexture{
  if(!this.paperTexture){this.paperTexture=this.device.createTexture({label:'MAIN paper appearance',size:[1,1],format:this.format,usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});
   const white=new Uint8Array([255,255,255,255]);this.device.queue.writeTexture({texture:this.paperTexture},white,{bytesPerRow:4},[1,1]);}
  return this.paperTexture;
 }
 private destroy(target:Targets):void{target.comparison?.destroy();target.mask.destroy();target.inspection?.destroy();this.resources.delete(target);}
 // The full-resolution inspection field (only while an inspection view is on),
 // one slot per in-flight job like the cell field.
 private inspectionFields(width:number,height:number):FieldSet{return this.slot(this.inspectionRing,new Set(this.inflight.map(job=>job.inspectionSet)),width,height);}
 // ONE OUTPUT, REDRAWN ONLY WHEN IT CHANGES (frame cost and memory,
 // 2026-11-12). Only the presented pair's page is ever composed, so one
 // output texture serves every pair (8 B/px per pair, not 12). The composed
 // page depends only on the presented pair (its comparison and underdrawing),
 // its sealed snapshot or legacy frame and the ASCII and overlay state; while
 // none changes, a host frame only copies the retained output to its target.
 // Recomposing an unchanged page every host frame was ~2.2 ms of GPU per host
 // frame at 3488x2252 (composite, tape and page passes).
 private shownOutput?:GPUTexture;private readonly retiredOutputs:GPUTexture[]=[];
 private presented?:{targets:Targets;snapshot?:ResidentNativeSnapshot;frame?:PhysicalTextFrame;ascii:boolean;overlay:boolean};
 // Host submissions normally resolve in queue order, but a discarded or
 // externally-held completion callback must never regress the state published
 // by a newer successful presentation.
 private presentationSerial=0;private presentedSerial=0;
 private outputFor(width:number,height:number):GPUTexture{
  const have=this.shownOutput;if(have&&have.width===width&&have.height===height)return have;
  if(have)this.retiredOutputs.push(have);this.presented=undefined;
  return this.shownOutput=this.texture(width,height,this.format);
 }
 private recycle(target:Targets):void{
  if(this.disposed){this.destroy(target);return;}
  // Only a completed host fence releases this coherent mesh/field/text pair.
  // Keep one spare, rather than allocating nine full-size render targets for
  // every text frame or building an unbounded viewport-size cache.
  if(this.spareTargets)this.destroy(this.spareTargets);this.spareTargets=target;
 }
 private retireBatch(batch:RetirementBatch|undefined,reuse:boolean):void{
  if(!batch||!this.retirementBatches.delete(batch))return;
  batch.buffers.forEach(buffer=>buffer.destroy());
  batch.targets.forEach(target=>reuse?this.recycle(target):this.destroy(target));
  batch.fields.forEach(set=>this.destroyFields(set));
  batch.outputs.forEach(texture=>texture.destroy());
  for(const snapshot of batch.snapshots)for(const result of snapshot.results)result.resident?.release();
 }
 private release(source:SourceBuffers):void{
  // Vertex/index residency has one owner, PhysicalSourceTransport. Only
  // camera/instance flags belong to the field adapter.
  this.retiredBuffers.push(source.flags);
 }
 /** Each write follows the previous host submit, before this frame's draws.
  * Asynchronous native jobs use retained field textures, never this arena.
  * Only a mesh revision or arena growth replaces a source's field bindings. */
 private fieldParameters(count:number):{buffer:GPUBuffer;memory:Float32Array;words:Uint32Array;stride:number}{
  const alignment=this.device.limits.minUniformBufferOffsetAlignment,stride=Math.ceil(192/alignment)*alignment;
  const size=Math.max(stride,count*stride);
  if(size>this.device.limits.maxBufferSize)throw Error('Physical field parameters exceed the device buffer extent');
  if(!this.fieldUniforms||this.fieldUniforms.buffer.size<size){
   if(this.fieldUniforms)this.retiredBuffers.push(this.fieldUniforms.buffer);
   const extent=Math.min(Math.floor(this.device.limits.maxBufferSize/stride)*stride,stride*2**Math.ceil(Math.log2(size/stride)));
   const memory=new Float32Array(extent/4),buffer=this.device.createBuffer({label:'MAIN retained physical field parameters',size:extent,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
   this.fieldUniforms={buffer,memory,words:new Uint32Array(memory.buffer)};
  }
  return {...this.fieldUniforms,stride};
 }
 private source(geometry:PhysicalGeometry):SourceBuffers{
  const key=geometry.owner,previous=this.sources.get(key);
  if(previous&&previous.positions===geometry.positions&&previous.indices===geometry.indices&&previous.revision===geometry.revision)return previous;
  if(previous)this.release(previous);
  const mesh=this.transport.acquire(geometry);
  const flags=new Uint32Array(Math.max(1,geometry.indices.length/3));
  const buffer=this.device.createBuffer({label:'MAIN physical instance field flags',size:flags.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
  // AUTHORED FACE MATERIAL (tone lane): the class of face f in bits 8-11 of
  // its flag word, beside the camera clearance in bits 0-7. No authored table
  // entry (or a lineage that does not cover the faces): no class bits.
  let classes:Uint32Array|undefined;
  try{
   const faces=geometry.indices.length/3,nibbles=compileSourceFaceMaterialClasses(geometry as Parameters<typeof compileSourceFaceMaterialClasses>[0],faces);
   if(nibbles){classes=new Uint32Array(flags.length);for(let f=0;f<faces;f++)classes[f]=((nibbles[f>>>3]!>>>((f&7)*4))&15)<<8;flags.set(classes);this.device.queue.writeBuffer(buffer,0,flags);}
  }catch{classes=undefined;}
  const result={positions:geometry.positions,indices:geometry.indices,revision:geometry.revision,vertex:mesh.positions,index:mesh.indices,vertexBinding:mesh.positionsBinding,indexBinding:mesh.indicesBinding,flags:buffer,classes};this.sources.set(key,result);return result;
 }
 private copy(encoder:GPUCommandEncoder,source:GPUTexture,target:GPUTextureView,width:number,height:number,uniforms:GPUBuffer[],unscaled=false):void{
  const params=this.device.createBuffer({size:32,usage:GPUBufferUsage.UNIFORM,mappedAtCreation:true});new Float32Array(params.getMappedRange()).set([source.width,source.height,width,height,Number(unscaled),0,0,0]);params.unmap();uniforms.push(params);
  const group=this.device.createBindGroup({layout:this.blit.getBindGroupLayout(0),entries:[{binding:0,resource:source.createView()},{binding:1,resource:{buffer:params}}]});
  const pass=encoder.beginRenderPass({colorAttachments:[{view:target,loadOp:'clear',storeOp:'store',clearValue:{r:1,g:1,b:1,a:1}}]});pass.setPipeline(this.blit);pass.setBindGroup(0,group);pass.draw(3);pass.end();
 }
 private sourceClasses(job:Job):SourceConstructionClass[]{
  if(!this.playback)return [];
  return job.sources.flatMap(source=>{
   const base={domain:source.domain,revision:source.source.geometry.revision,topologyRevision:source.source.geometry.revision,surface:source.source.geometry.surface,...source.source.analysis};
   // Physical Y8 rows carry a producer-tagged owner in the field. Their
   // selected coarse representation is the producer-column carrier below;
   // attaching per-instance far records here would make the same row eligible
   // for both the entity emitter and the producer emitter. Keep the complete
   // source class so the relation stage's ownerMerged gate can suppress only
   // the selected producer rows. Entity-shaped physical sources (if a future
   // host supplies one) retain the source-lod far-form path.
   const producer=/\/producer-row\//.test(source.source.id);
   if(producer)return [base];
   const instances=job.lod?.far.get(source.domain);if(!instances?.length)return [base];
   const far={...base,farForms:{domain:source.domain,instances}};
   // A shared physical source can have both near and far producer rows. Keep
   // the complete source class for its full members and add the far carrier;
   // never let one far member demote a near member of the same source.
   return job.lod?.full?.has(source.domain)?[base,far]:[far];
  });
 }



  private pushFarAggregate(classes:unknown[],job:Job):void{
   const perspective=job.sourceView.perspective,lod=job.lod;if(!perspective||!job.sources.length||!lod)return;
   const farDomains=[...lod.far.keys()].filter(domain=>job.sources.some(source=>source.domain===domain));
   const producerSourceDomains=new Set(farDomains.filter(domain=>job.sources.some(source=>source.domain===domain&&/\/producer-row\//.test(source.source.id))));
   const fieldOwner=(mesh:number,eid:number)=>{
    if(!farDomains.includes(mesh))return;
    const producer=producerSourceDomains.has(mesh);
    // Producer owners retain their high-bit identity in the aggregate key;
    // the far-form consumer resolves the same tagged owner from the field.
    return [producer?((mesh|0x80000000)>>>0):mesh,eid] as const;
   };
   const aggregates=(lod.parents??[]).map(node=>({key:node.key??0,box:node.box,construction:remapSourceWholeConstruction(node.construction,fieldOwner),members:node.members.flatMap(member=>{const owner=fieldOwner(member.mesh,member.eid);return owner?[owner]:[];})})).filter(node=>node.members.length);
   // Keep every far producer domain. The form-depth parent flag removes only
   // members whose complete group was certified on the GPU; mixed domains and
   // rejected groups must remain eligible for the legacy column carrier.
   const producerDomains=farDomains.filter(domain=>producerSourceDomains.has(domain));
   if(!producerDomains.length&&!aggregates.length)return;
   const farForms:SourceFarForms={domain:0,instances:[],aggregates,producers:producerDomains.length?{domains:producerDomains,perspective}:undefined};
   classes.push({domain:0,type:'far-aggregate',topologyRevision:0,farForms});
  }
 private encodeRelations(encoder:GPUCommandEncoder,job:Job):void{
  this.relations.begin(encoder,job.sourceView,job.sourceField,2);

  // sources, so the owner gates (no-texel owner, far producer row) run here too.
  this.relations.fieldDomains=new Set(job.sources.map(source=>source.domain));
  this.relations.retainPosedBatches(new Set(job.sources.map(source=>posedBatchIdentity(source.batch))));
  // Contact passes of every source first, then every source's relation
  // stages in ONE compute pass (surface-relations.ts append, frame cost).
  // Batches with every instance outside the view emit nothing on the GPU
  // (physical-source.ts instanceMayShow): they are not encoded at all.
  const seen=job.sources.filter(source=>job.inView.has(source));
  this.lastCensus={geometry:job.geometry,clip:job.sourceView.clipFromWorld,inView:new Set(seen.map(source=>source.domain)),edges:new Map(seen.map(source=>[source.domain,sourceResidencyEdgeCount(source.source)])),width:job.targets.width,height:job.targets.height};
  // A physical batch's draw extent is static: its metadata words 0 and 1 are
  // the source's index count and the batch's instance count (physical-source
  // retained batch metadata), so relation and contact stages are sized on
  // the host (surface-relations.ts HOST-SIZED STAGE SETS).
  const extent=(source:PhysicalResidentBatch)=>({indices:source.source.geometry.indices.length,instances:source.batch.source.instanceCapacity!});
  this.relations.reservePartAnchors(seen);
  // Final same-pose parent admission compacts each source frontier before
  // contacts are packed. Rejected candidates leave their incumbent children
  // in place; all later relation/anchor/interior consumers reuse this list.
  this.relations.admitFrontiers(encoder,seen);
  this.relations.appendContactsAll(encoder,seen.map(source=>({...source,extent:extent(source)})));
  if(seen.length){const pass=encoder.beginComputePass({label:'MAIN source relations'});try{for(const source of seen)this.relations.append(encoder,source.batch,source.source,job.sourceView,source.domain,undefined,pass,extent(source));}finally{pass.end();}}
  job.partAnchorDemand=this.relations.finish(encoder);
  // Relabeling reuses this job's pose after later relation jobs may have
  // overwritten the shared table. Retain it with the already-kept tone data.
  if(job.partAnchorTable)this.retiredBuffers.push(job.partAnchorTable.buffer);
  const anchors=this.relations.partAnchorTable;
  job.partAnchorTable=undefined;
  if(anchors){
   job.partAnchorTable=retainPhysicalPartAnchors(this.device,encoder,anchors);
  }
  const {width,height}=job.targets;
  this.visibility.encode(encoder,this.relations,job.targets.depth.createView(),job.sourceView.clipFromWorld,width,height,job.sourceField);

  this.mask.encode(encoder,this.visibility,job.targets.mask,width,height,job.view.rect,true,job.sourceField,job.view.light);
  // D6 consumes THIS retained pose's D10 boundaries. Encode after the mask,
  // not against an uninitialized/stale target before relation extraction.
  const classes=this.sourceClasses(job);
  // Carry the selected parent/producer carrier to the same GPU articulation
  // call. This is deliberately before tone and enqueue, not a side-channel
  // diagnostic or an unconsumed selection flag.
  if(this.playback)this.pushFarAggregate(classes,job);
  // The host's sun caster map, when it has one, with THIS job's pose (the
  // cascade split reads the eye and forward the fields were drawn from).
  const sun=job.sunShadow&&{...job.sunShadow,eye:job.sourceView.eye,forward:job.sourceView.direction};
  if(this.playback&&!job.tone)job.tone=toneField(this.device).encode(encoder,job.sourceField,job.sourceView.clipFromWorld,job.view.rect,width,height,job.view.light,job.targets.comparison,true,sun,job.targets.mask,classes as unknown as Parameters<ReturnType<typeof toneField>['encode']>[11],job.view.fog);
  job.capacity=this.visibility.capacity;job.intervalCapacity=this.visibility.intervalCapacity;
  // Retain the current-view GPU runs before any later source extraction.
  // Compact tone then joins the SAME fixed-priority native placement; no
  // posed graph, relation readback or CPU construction is reintroduced.
  if(this.playback){
   this.playback.retainSources(new Set(job.sources.map(source=>source.domain)),new Set(job.geometry.map(source=>Number(source.owner.slice(source.owner.lastIndexOf('/')+1)))));
   const literals:readonly ArticulationLiteral[]|ArticulationLiteralFeed=(job.tone?.immediate??job.toneMembers)?.length===0
    ?this.construction.playbackLiterals(job.labels,[],[])
    :{subscribe(resolve,reject){job.toneLiterals={resolve,reject};}};
   job.played=this.playback.enqueueSource(encoder,this.visibility,classes,job.sourceField,width,height,'camera-0',job.view.owners,job.view.rect,literals,job.tone?.gpu,[],undefined,job.partAnchorTable);void job.played.catch(()=>{});return;
  }
  const size=16+job.capacity*RELATION_WORDS*4+job.intervalCapacity*16;
  if(size>this.device.limits.maxBufferSize)throw Error('Complete physical source readback needs device-sized chunks');
  const staging=this.device.createBuffer({label:'MAIN retained physical source relation readback',size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  encoder.copyBufferToBuffer(this.visibility.status,0,staging,0,16);
  encoder.copyBufferToBuffer(this.visibility.lines,0,staging,16,job.capacity*RELATION_WORDS*4);
  encoder.copyBufferToBuffer(this.visibility.intervals,0,staging,16+job.capacity*RELATION_WORDS*4,job.intervalCapacity*16);
  job.staging=staging;
 }
 // LABEL-ONLY JOB (frame cost, 2026-11-13). Every still-view input but the
 // labels equal (still(input,true)) and the kept full job displayed: only
 // articulation runs, from that job's retained field, visibility, D10 image and
 // tone members, with the new labels as its literals, on the same targets.
 private relabel(input:PhysicalFrameInput):Job|undefined{
  const base=this.kept;if(!base||!this.playback||this.displayed?.targets!==base.job.targets||!this.still(input,true))return undefined;
  const tone:ToneReadback={...base.tone,dispose(){}};
  const job:Job={...base.job,serial:++this.serial,engineFrame:input.engineFrame,labels:input.labels??[],mode:input.mode,cpuEncodeMs:0,tone,
   played:undefined,toneLiterals:undefined,toneMembers:undefined,staging:undefined,relationReadback:undefined,retry:undefined,field:undefined,settled:false,snapshot:undefined};
  this.relabels.add(job);
  const classes=this.sourceClasses(job);this.pushFarAggregate(classes,job);
  this.playback.retainSources(new Set(job.sources.map(source=>source.domain)),new Set(job.geometry.map(source=>Number(source.owner.slice(source.owner.lastIndexOf('/')+1)))));
  job.played=this.playback.enqueueSource(input.encoder,this.visibility,classes,job.sourceField,job.targets.width,job.targets.height,'camera-0',job.view.owners,job.view.rect,this.construction.playbackLiterals(job.labels,[],[]),tone.gpu,[],undefined,job.partAnchorTable);void job.played.catch(()=>{});
  this.inflight.push(job);this.started=this.startKey(input,this.holdLight(input.light),job.pending);this.lastStartedJob=job;
  return job;
 }

 // A new job used to start only after every in-view source had compiled. A
 // camera turn at 0.8 rad/s brings uncompiled sources 60-320 units away into
 // view faster than the worker installs them (~25/s), so on Y8 no job started
 // for 5-7 s. After the first publication, a job starts with the ready sources.
 // The in-view sources still compiling travel with it: the published frame
 // reports them pending in its status (never a silent omission), and each
 // joins the first job after it is ready (still). The first publication still
 // waits for its whole in-view cohort.
 /** Returns the pending in-view sources the new job goes without ([] when all
  * are ready), or undefined when no job may start yet. */
 private admit(input:PhysicalFrameInput):readonly CompositionGeometry[]|undefined{
  const waiting=this.transport.prepare(input.geometry,input.clipFromWorld);
  if(waiting&&waiting!==this.registration){
   const registration=this.registration=waiting;
   void registration.catch(error=>{if(!this.disposed){const message=error instanceof Error?error.message:String(error);if(this.controller.errors.at(-1)!==message)this.controller.errors.push(message);this.controller.onState();}}).finally(()=>{if(this.registration===registration)this.registration=undefined;});
  }else if(!waiting)this.registration=undefined;
  if(waiting&&!this.displayed)return undefined;
  for(const source of this.transport.takeCompiled())if(!this.playback){
   if(!source.complex)throw Error('Legacy construction requires full source inspection');
   this.construction.installSource(source.geometry,source.analysis);
  }
  return waiting?this.transport.demanded():[];
 }
 /** A job reached its terminal state (snapshot, failure or supersession).
  * Publication follows serial order: drain publishes every settled job at the
  * head, so a later serial waits for an earlier job's retry and is never
  * shown before it. Two ready at once: the later replaces the earlier before
  * the next host frame presents. */
 private settle(job:Job):void{if(job.settled)return;job.settled=true;this.drain();}
 private drain():void{
  while(!this.disposed&&this.inflight[0]?.settled){
   const job=this.inflight.shift()!,snapshot=job.snapshot;job.snapshot=undefined;
   if(snapshot){
    const old=this.displayed;this.displayed={resident:{snapshot,targets:job.targets},targets:job.targets};
    this.controller.errors.length=0;
    if(old){if(old.targets!==job.targets)this.retiredTargets.push(old.targets);if(old.resident)this.retiredSnapshots.push(old.resident.snapshot);}
   }else if(job.targets!==this.displayed?.targets&&job.targets!==this.meshOnly&&!this.inflight.some(other=>other.targets===job.targets)&&!this.retiredTargets.includes(job.targets))this.retiredTargets.push(job.targets);
   this.finish(job);
  }
 }
 private finish(job:Job):void{
  job.toneLiterals?.reject(Error('Native physical job retired'));job.toneLiterals=undefined;
  // The displayed full job keeps its tone members for label-only jobs.
  if(job.tone?.gpu&&!this.relabels.has(job)&&job===this.lastStartedJob&&this.displayed?.targets===job.targets&&this.kept?.job!==job){this.releaseKept();this.kept={job,tone:job.tone};}
  else if(this.kept?.job!==job)job.tone?.dispose();
  if(!this.relabels.has(job)&&this.kept?.job!==job){job.partAnchorTable?.buffer.destroy();job.partAnchorTable=undefined;}
  job.tone=undefined;
  job.read.dispose();
  job.relationReadback?.dispose();job.relationReadback=undefined;
  job.partAnchorDemand?.dispose();job.partAnchorDemand=undefined;
  if(job.staging){if(job.staging.mapState==='mapped')job.staging.unmap();job.staging.destroy();job.staging=undefined;}
  // The source-pose buffers survive ALL allocation stages of this one job.
  // Only the terminal source fence releases them, never a new live tick.
  this.transport.submitted();
 }
 private releaseKept():void{
  if(!this.kept)return;
  this.kept.tone.dispose();this.kept.job.partAnchorTable?.buffer.destroy();this.kept.job.partAnchorTable=undefined;this.kept=undefined;
 }
 encode(input:PhysicalFrameInput):(submitted:Promise<void>)=>void{
  if(this.disposed)throw Error('MAIN physical renderer is disposed');
  this.visibility.encodeReadbacks(input.encoder);
  toneField(this.device).encodeReadbacks(input.encoder);
  this.playback?.encode(input.encoder);
  const started=performance.now();
  const uniforms:GPUBuffer[]=[];const jobs:Job[]=[];let job:Job|undefined;
  // Own presentation allocations immediately, including an encode that
  // throws before it can return the host's submission callback.
  const retirement:RetirementBatch={buffers:uniforms,targets:[],snapshots:[],fields:[],outputs:[]};
  this.retirementBatches.add(retirement);
  this.lastEncode=retirement;
  const report=(error:unknown)=>{const message=error instanceof Error?error.message:String(error);if(this.controller.errors.at(-1)!==message)this.controller.errors.push(message);this.controller.onState();};
  // Extents measured by a superseded source retry: grown before any encode reads them.
  if(this.growth){
   const extent=this.growth;this.growth=undefined;
   try{this.relations.ensureCapacity(extent.relations);if(extent.triangles)this.relations.ensureContactCapacity(extent.triangles);this.visibility.ensureCapacity(this.relations.lineCapacity,extent.intervals);}catch(error){report(error);}
  }
  for(const retained of this.inflight.filter(other=>other.retry)){
   const extent=retained.retry!;retained.retry=undefined;
   try{
    if(extent.anchor)this.relations.ensurePartAnchorDemand(extent.anchor);
    if(extent.standingOwnerDemand!==undefined)this.playback?.ensureStandingOwnerCapacity(extent.standingOwnerDemand);
    this.relations.ensureCapacity(extent.relations);
    if(extent.triangles)this.relations.ensureContactCapacity(extent.triangles);
    this.visibility.ensureCapacity(this.relations.lineCapacity,extent.intervals);
    // SUPERSEDED SOURCE RETRY: a newer job re-posed the shared batches, so
    // this exact pose is no longer resident. Its extents grew above; the
    // newer pose publishes, or a fresh job starts when none is in flight.
    if(this.posed!==retained.pose){this.started=undefined;this.settle(retained);continue;}
    this.encodeRelations(input.encoder,retained);jobs.push(retained);
   }catch(error){report(error);this.started=undefined;this.settle(retained);}
  }
  let sourcesReady=false,pending:readonly CompositionGeometry[]=[];
  // A new job starts while fewer than IN_FLIGHT jobs are in flight and none
  // awaits a source retry. Label-only jobs read the kept job's shared
  // visibility buffers, so they start only with nothing in flight; the legacy
  // CPU producer keeps one job at a time (its relation readbacks are shared).

  // re-encoded above already holds this encoder. GpuArticulation links every
  // same-camera job of one encoder into one cohort, and a member's overflow or
  // failure releases every resident of that cohort, so a new job here destroyed
  // the retry's published tape ("MAIN sealed native scalar tape" used in submit
  // while destroyed). The new job starts at the next host frame.
  const busy=jobs.length>0||this.inflight.length>=(this.playback?PhysicalFrame.IN_FLIGHT:1)||this.inflight.some(other=>other.retry||this.relabels.has(other));
  if(!busy&&!this.controller.paused&&!this.still(input)&&!(this.inflight.length===0&&(job=this.relabel(input)))){
   try{const admitted=this.admit(input);if(admitted){pending=admitted;sourcesReady=true;}}catch(error){const message=error instanceof Error?error.message:String(error);if(this.controller.errors.at(-1)!==message)this.controller.errors.push(message);this.controller.onState();}
  }
  if(sourcesReady){
   // Only a NEW text job snapshots source/model authority. Presenting the
   // retained page while that job is pending must not recopy every source.
   // Other material adapters cannot assume this physical vertex contract.
   const geometries:PhysicalGeometry[]=input.geometry.filter(source=>this.transport.isReady(source)).map(source=>({...source,poseAuthority:'source-model'}));
   const live=new Set(geometries.map(source=>source.owner));
   for(const [owner,buffers]of this.sources)if(!live.has(owner)){this.release(buffers);this.sources.delete(owner);}
   const light=this.holdLight(input.light);   const serial=++this.serial,targets=this.targets(input.width,input.height),attachments:SurfaceAttachment[]=[],source:SurfaceField={width:targets.field[0]!.width,height:targets.field[0]!.height,textures:targets.field,worldFromClip:input.worldFromClip,sources:new Map(),attachments};
   const parameters=this.fieldParameters(geometries.length);let parameterIndex=0;const draws:[GPUBindGroup,number,number][]=[];
   const pass=input.encoder.beginRenderPass({label:'MAIN physical world fields',colorAttachments:targets.field.map(texture=>({view:texture.createView(),loadOp:'clear' as const,storeOp:'store' as const,clearValue:[0,0,0,0]})),depthStencilAttachment:{view:targets.depth.createView(),depthLoadOp:'clear',depthStoreOp:'store',depthClearValue:0}});
   pass.setPipeline(this.pipeline);
   for(const geometry of geometries){
    if(!geometry.indices.length)continue;
    const match=/^camera-0\/(\d+)\/producer-row\/(\d+)$/.exec(geometry.owner);if(!match)throw Error(`Invalid physical source identity ${geometry.owner}`);
    const domain=Number(match[1]),owner=Number(match[2]);if(domain<=0||domain>=0x20000000||owner<0||owner>0xffffffff)throw Error('Physical source identity exceeds integer field range');
    const buffers=this.source(geometry);
    if(this.playback){
     // This physical vertex stage is certified source/model posing. Snapshot
     // that exact authority for GPU tone; do not make a moving actor travel
     // through a stationary world-space pattern. Singular/flattened sources
     // retain the world fallback rather than rejecting the whole frame.
     const pose=geometry.worldFromSource??IDENTITY,cached=buffers.attachment;
     try{
      // Cache with the existing source lifetime. Held/static poses do not
      // rebuild inverse matrices or acquire a second growing owner registry.
      if(cached&&cached.worldFromSource.every((value,index)=>value===pose[index]))attachments.push(cached);
      else{const worldFromSource=new Float32Array(pose),attachment:SurfaceAttachment={source:[(domain|0x80000000)>>>0,owner],worldFromSource,sourceFromWorld:inverseMatrix(worldFromSource)};buffers.attachment=attachment;attachments.push(attachment);}
     }
     catch(error){if(!(error instanceof Error)||error.message!=='Noninvertible composition source transform')throw error;}
    }
    const clearance=geometry.cameraCleared;
    if(clearance){if(clearance.length!==geometry.indices.length/3)throw Error('Physical camera-clearance extent changed');
     if(!buffers.clearance||clearance.some((value,index)=>value!==buffers.clearance![index])){this.device.queue.writeBuffer(buffers.flags,0,Uint32Array.from(clearance,(value,index)=>value|(buffers.classes?.[index]??0)));buffers.clearance=clearance.slice();}
    // Cleared through the queue like the write above: the frame encoder is
    // locked while this render pass is open, and an encoder clear here
    // invalidated the whole frame (Y8 walk 2026-11-11: "Recording in
    // shallot-frame which is locked while MAIN physical world fields is open",
    // no text frame for ~50 s).
    }else if(buffers.clearance){this.device.queue.writeBuffer(buffers.flags,0,buffers.classes?buffers.classes.slice(0,buffers.clearance.length):new Uint32Array(buffers.clearance.length));buffers.clearance=undefined;}
    const offset=parameterIndex++*parameters.stride,at=offset/4,values=parameters.memory,words=parameters.words;
    values.set(input.clipFromWorld,at);values.set(geometry.worldFromSource??IDENTITY,at+16);words[at+32]=(domain|0x80000000)>>>0;words[at+33]=owner;
    values[at+36]=geometry.albedo?.[0]??.8;values[at+37]=geometry.albedo?.[1]??.8;values[at+38]=geometry.albedo?.[2]??.8;values[at+39]=1;
    values[at+40]=input.pose.eye[0]!;values[at+41]=input.pose.eye[1]!;values[at+42]=input.pose.eye[2]!;
    values[at+44]=light?.[0]??.4;values[at+45]=light?.[1]??1;values[at+46]=light?.[2]??.25;
    if(buffers.binding?.uniform!==parameters.buffer)buffers.binding={uniform:parameters.buffer,group:this.device.createBindGroup({layout:this.fieldLayout,entries:[{binding:0,resource:{buffer:parameters.buffer,size:192}},{binding:1,resource:buffers.vertexBinding},{binding:2,resource:buffers.indexBinding},{binding:3,resource:{buffer:buffers.flags}}]})};
    pass.setBindGroup(0,buffers.binding.group,[offset]);pass.draw(geometry.indices.length);draws.push([buffers.binding.group,offset,geometry.indices.length]);
    (source.sources as Map<number,{kind:'producer-row';hasUV:boolean;stage:'opaque'}>).set(domain,{kind:'producer-row',hasUV:false,stage:'opaque'});
   }
   pass.end();if(parameterIndex)this.device.queue.writeBuffer(parameters.buffer,0,parameters.memory.buffer,0,parameterIndex*parameters.stride);
   // INSPECTION AT HALF RESOLUTION (each axis; frame cost 2026-11-13: about
   // 33 ms per update at 3488 at device resolution), ONLY WHILE SHOWN. The mesh comparison and
   // the D10 inspection image are what a viewer compares the text against: with
   // ASCII off or a mesh/underdrawing view on, the same draws are replayed into a
   // full-resolution field for them (encodeRelations draws its D10 image). With
   // no inspection view the comparison is the cell field's colour, enlarged.
   const inspect=originalScenePresentationRequested(this.controller);
   let full:FieldSet|undefined;
   if(inspect){
    const inspectWidth=Math.ceil(input.width/2),inspectHeight=Math.ceil(input.height/2);
    full=this.inspectionFields(inspectWidth,inspectHeight);
    const replay=input.encoder.beginRenderPass({label:'MAIN physical world fields (inspection, half resolution)',colorAttachments:full.field.map(texture=>({view:texture.createView(),loadOp:'clear' as const,storeOp:'store' as const,clearValue:[0,0,0,0]})),depthStencilAttachment:{view:full.depth.createView(),depthLoadOp:'clear',depthStoreOp:'store',depthClearValue:0}});
    replay.setPipeline(this.pipeline);for(const [group,offset,count]of draws){replay.setBindGroup(0,group,[offset]);replay.draw(count);}replay.end();
    if(targets.inspection&&(targets.inspection.width!==inspectWidth||targets.inspection.height!==inspectHeight)){this.retiredOutputs.push(targets.inspection);targets.inspection=undefined;}
    targets.inspection??=this.texture(inspectWidth,inspectHeight,'rgba8unorm');
   }else{
    this.retireInspection();
    if(targets.inspection){this.retiredOutputs.push(targets.inspection);targets.inspection=undefined;}
   }
   // THE COMPARISON IMAGE ONLY WHEN EXPLICITLY REQUESTED: ASCII-off, mesh /
   // underdrawing, or overlay. A normal ASCII page never allocates or copies
   // the original scene, including before its first text publication; its
   // pending presentation is paper. Tone still consumes the source field and
   // the required D10 construction mask above.
   if(inspect){targets.comparison??=this.texture(input.width,input.height,this.format);this.copy(input.encoder,(full??targets).field[4]!,targets.comparison.createView(),input.width,input.height,uniforms);}
   else if(targets.comparison){this.retiredOutputs.push(targets.comparison);targets.comparison=undefined;}
   // Source presentation must survive a failed text job. This pair has NO
   // character record and is labelled NO FRAME/ERROR by the front door; it
   // is not a successful fallback drawing or a text publication.
   // An in-flight job's pair is not retired here: it is published or, when
   // that job ends without a snapshot, retired by drain.
   if(this.meshOnly&&this.meshOnly!==this.displayed?.targets&&!this.inflight.some(other=>other.targets===this.meshOnly))this.retiredTargets.push(this.meshOnly);
   this.meshOnly=targets;
   // Early shared composition limits relation extraction, while EVERY actual
   // physical triangle above remains in the same depth/ownership occluder pass.
   const selected=this.construction.selectSources(geometries),components=new Map<number,MeshSourceComplex>();
   const instances:PhysicalSourceInstance[]=[],snapshots=new Map<string,PhysicalGeometry>();
   for(const geometry of geometries){
    if(!selected.has(geometry.owner)||!geometry.indices.length)continue;
    const match=/^camera-0\/(\d+)\/producer-row\/(\d+)$/.exec(geometry.owner)!,domain=Number(match[1]),owner=Number(match[2]);
    instances.push({geometry,domain,owner});
   }
   const instanceBatches=this.transport.encode(input.encoder,instances,input.clipFromWorld),posed={};this.posed=posed;
   // REPRESENTATION FRONTIER (Y8 physical adapter): the transport has now
   // posed the complete source cohort and updated its spatial index. Select
   // the certified exterior frontier before any relation/contact/run stage.
   // Producer rows use the certified parent aggregate; the domain carrier
   // remains present for mixed or rejected groups and is filtered by the
   // shared form-depth parent flag.
   const lodScope=this.lodLifetimes.scope('camera-0',this,0,false);
   // The physical adapter must use the same resident packed-bank admission as
   // the examples adapter. Keep the bound transport call intact: its spatial
   // index is the owner of the selector's `this` state.
   const nativePart=this.playback?.wholePartAvailable.bind(this.playback);
   const lod=this.playback?this.transport.selectLod(input.clipFromWorld,input.width,input.height,input.width,input.height,input.exterior===true,input.pose.perspective,this.parentHolds.get(lodScope),this.parentGroups,
    (parent,rows,width)=>this.playback!.smallConstructionAvailable(parent,rows,width),this.parentHolds.getSmall(lodScope),nativePart):undefined;
   if(lod)this.parentHolds.set(lodScope,lod.heldParents??new Set(),lod.heldSmall??new Set());
   // The field is complete: reduce its one per-form nearest-depth table after
   // the representation frontier is known, so resolveParents receives the
   // exact producer-tagged owner groups that this same pose may aggregate.
   // Unknown/interior views pass no groups and remain on the incumbent.
   const parentDomains=new Set(instances.map(instance=>instance.domain));
   const parentGroups:FormParentGroup[]=(lod?.parents??[]).map(node=>({members:node.members.map(member=>[
    (parentDomains.has(member.mesh)?((member.mesh|0x80000000)>>>0):member.mesh),member.eid
   ] as const)}));
   const formScope=this.playback?this.lodLifetimes.scope('camera-0',this,TONE_ROW*source.height/input.height):undefined;

   // source-revision bound under its certified source/model pose (all eight
   // corners), the same bound camera admission uses. No bound: incumbent.
   const sourceBounds:FormSourceBound[]=this.playback?instances.flatMap(({geometry,domain,owner})=>formSourceBoundOf(geometry,[(domain|0x80000000)>>>0,owner])):[];
   formDepths(this.device).encode(input.encoder,source,formScope,parentGroups,sourceBounds);
   if(!this.playback)for(const posed of instanceBatches){
    if(!posed.source.complex)throw Error('Legacy CPU inspection requires full source topology');
    components.set(posed.domain,posed.source.complex);
   }
   if(!this.playback){const definitions=new Map(instanceBatches.map(item=>[item.domain,item.source.geometry]));for(const instance of instances){const definition=definitions.get(instance.domain)!;snapshots.set(instance.geometry.owner,{...instance.geometry,positions:definition.positions,indices:definition.indices,normals:definition.normals,worldFromSource:instance.geometry.worldFromSource?.slice()});}}
   const read=this.playback?new ResidentConstructionField(source):new FieldReadback(this.device,input.encoder,source);
   const rect={x:0,y:0,width:input.width,height:input.height},appearance=input.appearance??targets.comparison??targets.field[4]!;
   read.completeAppearance(this.device,input.encoder,appearance,rect);
   const tone=this.playback?undefined:toneField(this.device).encode(input.encoder,source,input.clipFromWorld,rect,input.width,input.height,light,appearance,false);
   const pose={...input.pose,eye:[...input.pose.eye],forward:[...input.pose.forward]};
   job={serial,engineFrame:input.engineFrame,targets,read,tone,sunShadow:input.sunShadow,components,sources:instanceBatches,inView:this.transport.inView,sourceField:source,sourceView:{clipFromWorld:input.clipFromWorld.slice(),eye:pose.eye,direction:pose.forward,perspective:pose.perspective,pageExtent:[input.width,input.height]},capacity:0,intervalCapacity:0,
    inspection:full&&targets.inspection?{target:targets.inspection,field:{width:full.width,height:full.height,textures:full.field,worldFromClip:input.worldFromClip,sources:new Map(),attachments:[]}}:undefined,
    fieldSet:this.fieldRing.find(set=>set.field===targets.field),inspectionSet:full,pose:posed,pending,
    geometry:geometries.map(g=>snapshots.get(g.owner)??{...g,worldFromSource:g.worldFromSource?.slice()}),labels:input.labels??[],mode:input.mode,cpuEncodeMs:0,view:{field:undefined as never,camera:'camera-0',rect,owners:new Map(),pose,light},lod};
   // A full job overwrites the visibility buffers: label-only jobs of the kept job end here.
   this.releaseKept();
   this.encodeRelations(input.encoder,job);this.inflight.push(job);this.started=this.startKey(input,light,pending);this.lastStartedJob=job;
  }
  if(job)jobs.push(job);
  const shown=this.displayed;
  let nextPresented=this.presented;
  // EXTENT (resize, 2026-11-12). The blit scales its source to the host
  // target. A pair drawn for another extent (the page before a resize, until
  // the first job of the new extent publishes) is shown at 1:1 from the page
  // origin instead: its glyphs keep their exact size and place, and only the
  // part of the new target it does not reach is paper. Never scaled.
  const stale=(target:Targets)=>target.width!==input.width||target.height!==input.height;
  if(shown){
   const overlay=this.controller.ascii&&(this.controller.underlay||this.controller.overlay);
   const output=input.target?this.outputFor(shown.targets.width,shown.targets.height):undefined,last=this.presented,snapshot=shown.resident?.snapshot;
   if(input.target&&output&&(!last||last.targets!==shown.targets||last.snapshot!==snapshot||last.frame!==shown.frame||last.ascii!==this.controller.ascii||last.overlay!==overlay)){
    const appearance=originalScenePresentationRequested(this.controller)?shown.targets.comparison??this.paper():this.paper();
    if(shown.resident){
     const tapes=shown.resident.snapshot.results.flatMap(result=>result.resident?[result.resident.tape]:[]);
     this.publication.encodeGpu(input.encoder,tapes,appearance.createView(),output.createView(),this.format,shown.targets.width,shown.targets.height,this.controller.ascii,false,overlay);
    }else if(shown.frame){
     if(this.publication.frame!==shown.frame)this.publication.publish(shown.frame);
     this.publication.encode(input.encoder,appearance.createView(),output.createView(),this.format,shown.targets.width,shown.targets.height,this.controller.ascii,false,overlay);
    }
    nextPresented={targets:shown.targets,snapshot,frame:shown.frame,ascii:this.controller.ascii,overlay};
   }
   if(input.target&&output)this.copy(input.encoder,output,input.target,input.targetWidth,input.targetHeight,uniforms,stale(shown.targets));
  }else if(this.meshOnly){
   const mesh=this.meshOnly;nextPresented=undefined;
   const inspect=originalScenePresentationRequested(this.controller);
   if(input.target&&this.controller.ascii&&(this.controller.underlay||this.controller.overlay)){const output=this.outputFor(mesh.width,mesh.height);this.textureBlit.snapshot(input.encoder,mesh.comparison??this.paper(),output.createView(),this.format);this.copy(input.encoder,output,input.target,input.targetWidth,input.targetHeight,uniforms,stale(mesh));}
   else if(input.target)this.copy(input.encoder,inspect?mesh.comparison??this.paper():this.paper(),input.target,input.targetWidth,input.targetHeight,uniforms,stale(mesh));
  }
  else if(input.target){const pass=input.encoder.beginRenderPass({colorAttachments:[{view:input.target,loadOp:'clear',storeOp:'store',clearValue:{r:1,g:1,b:1,a:1}}]});pass.end();}
  const retiredBuffers=this.retiredBuffers.splice(0),retiredTargets=this.retiredTargets.splice(0),retiredSnapshots=this.retiredSnapshots.splice(0),retiredFields=this.retiredFields.splice(0),retiredOutputs=this.retiredOutputs.splice(0);
  for(const buffer of retiredBuffers)retirement.buffers.push(buffer);
  retirement.targets=retiredTargets;retirement.snapshots=retiredSnapshots;retirement.fields=retiredFields;retirement.outputs=retiredOutputs;
  if(!retirement.buffers.length&&!retiredTargets.length&&!retiredSnapshots.length&&!retiredFields.length&&!retiredOutputs.length)this.retirementBatches.delete(retirement);
  for(const encoded of jobs)encoded.cpuEncodeMs+=performance.now()-started;
  // This encoder's header maps are requested right after its own submit (the
  // callback below), before a later job is queued behind it.
  const maps=this.playback?.takeSubmissionMaps()??[],presentationSerial=input.target===undefined?0:(this.presentationSerial??0)+1,presentationRequestEpoch=this.presentationRequestEpoch??0;
  if(input.target!==undefined)this.presentationSerial=presentationSerial;
  return submitted=>{
   if(this.lastEncode===retirement)this.lastEncode=undefined;
   const hasTarget=input.target!==undefined;
   // Request header maps immediately after the owning queue submission, but
   // do not publish the canvas/export snapshot until that submission's fence
   // fulfills. A rejected fence/device loss must leave the prior presentation
   // authoritative and retryable.
   for(const map of maps)map();
   const commitPresentation=():void=>{
    if(this.disposed||!hasTarget||presentationSerial<(this.presentedSerial??0))return;
    this.presentedSerial=presentationSerial;this.presented=nextPresented;
    if(presentationRequestEpoch>(this.presentationConsumedEpoch??0))this.presentationConsumedEpoch=presentationRequestEpoch;
    if(shown?.resident){
     const snapshot=shown.resident.snapshot,changed=this.shownSnapshot!==snapshot;this.shownSnapshot=snapshot;
     this.controller.residentFrames.set(0,residentFrameState(snapshot));
     if(changed)this.controller.frames.delete(0);
     this.controller.onState();this.flushResidentExports(snapshot);
     // A subscribed recorder is an explicit continuous export consumer.
     // Without one, drawing never schedules a glyph readback.
     if(changed&&this.sink)void this.exportResident(snapshot).catch(error=>{if(!this.disposed){this.controller.errors.push(String(error));this.controller.onState();}});
    }
    if(shown?.frame&&!shown.resident){
     this.controller.frames.set(0,shown.frame);this.controller.onState();
     for(const request of this.waiting)request.resolve(shown.frame);this.waiting.clear();this.sink?.(shown.frame);
    }
   };
   // Each job of this frame collects independently: one job's failure ends
   // that job only (drain retires its pair), never the other in flight.
   const failed=(encoded:Job,error:unknown)=>{
    if(!this.disposed){const message=error instanceof Error?error.message:String(error);if(this.controller.errors.at(-1)!==message)this.controller.errors.push(message);this.controller.onState();}
    this.started=undefined;this.settle(encoded);
   };
   void submitted.then(()=>{
    commitPresentation();
    this.playback?.submitted();
    this.visibility.submitted();
    toneField(this.device).submitted();
    this.retireBatch(retirement,!this.disposed);this.publication.submitted();
    if(jobs.length){this.relations.submitted();this.mask.submitted();}
   }).then(()=>{for(const encoded of jobs)void this.collect(encoded).catch(error=>failed(encoded,error));},error=>{
    this.retireBatch(retirement,false);
    if(!jobs.length&&!this.disposed){const message=error instanceof Error?error.message:String(error);if(this.controller.errors.at(-1)!==message)this.controller.errors.push(message);this.controller.onState();}
    for(const encoded of jobs)failed(encoded,error);
   });
  };
 }
 /** The engine drops the open draw group without invoking its submit
  * callback. Retire only the most recently encoded, not-yet-owned batch; all
  * earlier batches already have their own submission fence. */
 discardFrame():void{
  const batch=this.lastEncode;this.lastEncode=undefined;
  if(batch)this.retireBatch(batch,false);
 }
 private async collect(job:Job):Promise<void>{
  const staging=job.staging,relationReadback=job.relationReadback;
  try{
   if(this.playback){await this.collectRetained(job);return;}
   if(!staging&&!relationReadback)throw Error('Physical text job has no source graph readback');
   const [field,packed]=await Promise.all([job.field?Promise.resolve(job.field):job.read.collect(),relationReadback?relationReadback.collect():staging!.mapAsync(GPUMapMode.READ).then(()=>undefined)]);if(this.disposed)return;job.field=field;
   const memory=packed?.memory??staging!.getMappedRange(),status=new Uint32Array(memory,0,4);
   if(status[1]){
    const sourceCount=status[3]!,intervalCount=status[0]!;
    if((status[1]&16)!==0&&(status[1]&~18)===0){
     job.retry={relations:job.capacity,intervals:Math.max(job.intervalCapacity,intervalCount),triangles:sourceCount};return;
    }
    if(status[1]===2&&(sourceCount>job.capacity||intervalCount>job.intervalCapacity)){
     // Counters continue through exhausted storage. Retain this EXACT source
     // pose/field/root geometry and finish it in the next HOST encoder using
     // those actual extents; no source quota, abandoned frame or own submit.
     job.retry={relations:Math.max(job.capacity,sourceCount),intervals:Math.max(job.intervalCapacity,intervalCount)};
     return;
    }
    throw Error(`Source graph/visibility failure ${status[1]} (${sourceCount} relations, ${intervalCount} intervals)`);
   }
   const cpuStarted=performance.now();
   const graph=decodePosedGraph(memory,packed?.capacity??job.capacity,packed?.intervalCapacity??job.intervalCapacity,'camera-0',job.view.rect,job.targets.width,job.targets.height,undefined,(domain,face)=>job.components.get(domain)?.faceComponents[face],field.visibleOwners?.());
   const graphs=[graph];let visibleSpans=0;
   for(const relation of graph.relations)for(const part of relation.intervals)if(part.cause==='visible')visibleSpans++;
   this.controller.graphs.set(job.serial,graphs);if(this.controller.graphs.size>4)this.controller.graphs.delete(this.controller.graphs.keys().next().value!);
   const tone=job.tone?await job.tone.collect():undefined;job.tone=undefined;
   const view:TonedFieldView={...job.view,field,tone};
   const playback=undefined;// legacy CPU producer only; compiled playback returned above
   const page=this.construction.construct(visibleSpans,job.targets.width,job.targets.height,job.serial,graphs,[view],job.labels,job.geometry,playback);
   const presented=this.presentation.present(page,constructionSupports(page,graphs),job.serial,job.engineFrame,job.mode);
   const frame:PhysicalTextFrame={...presented,construction:{sourceLines:page.sourceLines,omitted:page.omitted,pending:pendingOwners(job),metrics:{...page.metrics,cpuFrameScope:'physical source encoding + graph decode + native composition/presentation CPU; GPU/readback/engine excluded',cpuFrameMs:job.cpuEncodeMs+performance.now()-cpuStarted}}};
   const old=this.displayed;this.displayed={frame,targets:job.targets};
   this.controller.errors.length=0;
   // A later host fence retires the old pair after every already encoded
   // presentation of it. Collection never submits an independent encoder.
   if(old)this.retiredTargets.push(old.targets);
  }finally{
   relationReadback?.dispose();if(job.relationReadback===relationReadback)job.relationReadback=undefined;
   if(staging){if(staging.mapState==='mapped')staging.unmap();staging.destroy();if(job.staging===staging)job.staging=undefined;}
   if(!job.retry)this.settle(job);
  }
 }
 private async collectRetained(job:Job):Promise<void>{
  const playback=this.playback!;
  // Only the compact tone member list returns before articulation; the source
  // graph is read by GPU run assembly from the resident buffers.
  // A count-driven retry retains the SAME pose, including its compact tone.
  // Never silently replace that input with [] after the first collection.
  // GPU region packets remain owned by this source job through articulation
  // and same-pose retries. Only legacy compact CPU packets are collected.
  if(job.tone&&!job.tone.gpu){job.toneMembers=await job.tone.collect();job.tone=undefined;}if(this.disposed)return;
  let played=job.played;
  if(job.partAnchorDemand){
   const demand=await job.partAnchorDemand.collect();job.partAnchorDemand=undefined;
   if(demand.overflow){
    // The table was copied only for this exact pose. Grow from the complete
    // selected demand, then encodeRelations will replay the retained pose
    // before this job can enter the publication drain.
    if(!played)throw Error('Part-anchor demand has no retained articulation job');
    const [discarded]=await discardPartAnchorAttempt([{played,cancel(){job.toneLiterals?.reject(Error('Part-anchor demand retry discarded incomplete physical playback'));job.toneLiterals=undefined;}}]);
    const discardedResult=discarded?.status==='fulfilled'?discarded.value:undefined;
    const overflow=discardedResult?.sourceOverflow;
    if(overflow&&(overflow.flags&~22))throw Error(`GPU physical source failure ${overflow.flags}`);
    if(discardedResult?.sourceRowDemand!==undefined)this.relations.ensurePartAnchorRowDemand(discardedResult.sourceRowDemand);
    // Anchor and relation/contact demand belong to the same attempt. Do not
    // throw the latter away while retiring its incomplete anchor snapshot.
    job.retry={relations:Math.max(job.capacity,overflow?.relations??0),intervals:Math.max(job.intervalCapacity,overflow?.intervals??0),triangles:overflow?.triangles,anchor:demand};return;
   }
  }
  const tone=job.toneMembers;
  const view={...job.view,tone} as TonedFieldView;
  job.toneLiterals?.resolve(this.construction.playbackLiterals(job.labels,[view],[]));
  if(!played){
   const classes=this.sourceClasses(job);this.pushFarAggregate(classes,job);
   played=playback.enqueueSourceNext(this.visibility,classes,job.sourceField,job.targets.width,job.targets.height,'camera-0',job.view.owners,job.view.rect,this.construction.playbackLiterals(job.labels,[view],[]),job.tone?.gpu,job.partAnchorTable);
  }
  const result=await played;if(this.disposed){result.resident?.release();return;}
  // The hint proves THIS job used an undersized row table. Another in-flight
  // job may already have grown the shared reserve; that does not upgrade this
  // job's immutable layout. A latest stale job still needs a real rebuild.
  if(result.sourceRowDemand!==undefined){this.relations.ensurePartAnchorRowDemand(result.sourceRowDemand);if(this.lastStartedJob===job)this.started=undefined;}
  if(result.sourceOverflow){const overflow=result.sourceOverflow;if(overflow.flags&~22)throw Error(`GPU physical source failure ${overflow.flags}`);if(result.standingOwnerDemand!==undefined)this.playback?.ensureStandingOwnerCapacity(result.standingOwnerDemand);job.retry={relations:Math.max(job.capacity,overflow.relations),intervals:Math.max(job.intervalCapacity,overflow.intervals),triangles:overflow.triangles,standingOwnerDemand:result.standingOwnerDemand};return;}
  if(result.standingOwnerDemand!==undefined){
   // The missing compact carrier kept the complete incumbent. Publish that
   // valid tape now; grow and invalidate this particular old layout for the
   // next job instead of turning a harmless reserve hint into no first frame.
   if(result.standingOwnerDemand<0)throw Error('Negative standing-owner demand');
   this.playback?.ensureStandingOwnerCapacity(result.standingOwnerDemand);
   if(this.lastStartedJob===job)this.started=undefined;
  }
  const resident=result.resident;if(!resident)throw Error('Compiled physical playback returned no resident GPU tape');
  // The header proved this tape complete and sealed; drain publishes it in
  // serial order (after any earlier job's same-pose retry).
  job.snapshot={results:[result],width:job.targets.width,height:job.targets.height,serial:job.serial,engineFrame:job.engineFrame,mode:job.mode,cpuEncodeMs:job.cpuEncodeMs,pending:pendingOwners(job)};
 }
 private async exportResident(snapshot:ResidentNativeSnapshot):Promise<PhysicalTextFrame>{
  const frame=await exportResidentFrame(snapshot);if(this.disposed)throw Error('MAIN physical renderer is disposed');
  if(this.shownSnapshot===snapshot)this.controller.frames.set(0,frame);this.controller.onState();this.sink?.(frame);return frame;
 }
 private flushResidentExports(snapshot:ResidentNativeSnapshot):void{
  if(!this.waiting.size)return;
  const requests=[...this.waiting];this.waiting.clear();
  for(const request of requests)void this.exportResident(snapshot).then(request.resolve,request.reject);
 }
 /** Export the currently presented page; this does not request a new canvas frame. */
 nextTextFrame():Promise<PhysicalTextFrame>{
  if(this.disposed)return Promise.reject(Error('MAIN physical renderer is disposed'));
  if(this.playback){const snapshot=this.shownSnapshot;return snapshot?this.exportResident(snapshot):new Promise((resolve,reject)=>this.waiting.add({resolve,reject}));}
  const frame=this.controller.frames.get(0);return frame?Promise.resolve(frame):new Promise((resolve,reject)=>this.waiting.add({resolve,reject}));
 }
 setTextSink(sink:((frame:PhysicalTextFrame)=>void)|null):void{this.sink=sink;}
 dispose():void{this.disposed=true;this.releaseKept();for(const batch of [...this.retirementBatches])this.retireBatch(batch,false);this.displayed?.resident?.snapshot.results.forEach(result=>result.resident?.release());this.playback?.dispose();this.construction.dispose();for(const job of this.inflight.splice(0)){job.snapshot?.results.forEach(result=>result.resident?.release());job.snapshot=undefined;this.finish(job);}this.sources.forEach(source=>this.release(source));this.sources.clear();this.fieldUniforms?.buffer.destroy();this.fieldUniforms=undefined;this.retiredBuffers.forEach(buffer=>buffer.destroy());for(const target of [...this.resources])this.destroy(target);for(const set of [...this.retiredFields.splice(0),...this.fieldRing.splice(0),...this.inspectionRing.splice(0)])this.destroyFields(set);this.paperTexture?.destroy();this.paperTexture=undefined;for(const texture of [...this.retiredOutputs.splice(0),...(this.shownOutput?[this.shownOutput]:[])])texture.destroy();this.shownOutput=undefined;if(this.gpuServices)this.gpuServices.dispose();else{this.transport.dispose();this.relations.dispose();this.visibility.dispose();this.mask.dispose();}this.publication.dispose();for(const request of this.waiting)request.reject(Error('MAIN physical source epoch was retired'));this.waiting.clear();}
}
