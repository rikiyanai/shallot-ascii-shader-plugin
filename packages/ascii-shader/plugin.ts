import {type Plugin,type World,Camera,Meshes,MeshInstance,MeshMaterial,Materials,Text,GlobalTransform,DirectionalLight,Shape,type MeshHandle,unpackColor} from '@dylanebert/shallot';
import {DepthPrepassRequests,FinalPresentationPasses,MainPassSystem,PrepareViewsSystem,RenderContext,TonemappingSystem,VIEW_BYTES,Views,PhaseGeometryCapture,type View} from '@dylanebert/shallot/rendering';
import {BeginFrameSystem,requestCanvasPresentationTicket} from '@dylanebert/shallot/rendering';
import {PresentationLedger} from './presentation-ledger.ts';
import {Fog} from '@dylanebert/shallot/extras';
import {cellMatrix,type CellMatrix} from './cell-matrix.ts';
import {MeshRenderPlugin,StandardRenderingPlugin,visitMaterialGeometry,sourceViewWorkset,sunShadowView,sunShadowParams,shadowSampler} from '@dylanebert/shallot/standard/rendering';
import {composeGlobalTransform} from '@dylanebert/shallot/transform';
import type {CompositionGeometry} from './composition.ts';
import nativeFont from './native-glyphs.json' with {type:'json'};
import {MaterialGeometryTransport,type MaterialGeometryDraw,type RetainedMaterialGeometryBatch} from './geometry.ts';
import {prepareExampleGpuServices,type ExampleGpuRelationProfile} from './gpu-startup.ts';
import type {SourceConstructionClass} from './gpu-construction-runs.ts';
import type {SourceFarForms} from './gpu-source-far-forms.ts';
import {SurfaceRelations,RELATION_WORDS,type PartAnchorDemandReadback} from './surface-relations.ts';
import {discardPartAnchorAttempt} from './part-anchor-replay.ts';
import {MeshSourceCache,isGpuIndexedResident,isCompiledFarForm,type ResidentGpuIndexedSource,type ResidentRelationSource} from './source-cache.ts';
import {selectSourceLod,fitsParentTier,ParentHolds,ParentGroupCache,mergeSourceLod,remapSourceWholeConstruction,type SourceLodSelection} from './source-lod.ts';
import type {SourceComplexRetention} from './source-program.ts';
import {decodePosedGraph,posedOwnerId,type PosedGraph} from './posed-graph.ts';
import {RelationVisibility,type ProjectedRelationReadback} from './relation-visibility.ts';
import {NativeConstruction,type NativeFrame} from './native-construction.ts';
import {NativePublication} from './publication.ts';
import {TextureBlit} from './texture-blit.ts';
import {ConstructionMask} from './construction-mask.ts';
import {toneField,toneClasses,type ToneReadback,type TonedFieldView,type GpuToneMembers,type SourceToneClass} from './tone-field.ts';
import {GpuArticulation,materialInk,materialInkTable,type ArticulationResult,type ArticulationLiteral,type ArticulationLiteralFeed} from './gpu-articulation.ts';
import {NativePresentation,type PresentedNativeFrame} from './native-presentation.ts';
import {constructionSupports} from './construction-support.ts';
import {plainText,type PresentationMode} from './text/text-frame.ts';
import {SurfaceFields,pruneInactiveSurfaceFields,type SurfaceField} from './surface-fields.ts';
import {formDepths,type FormParentGroup,type FormSourceBound} from './form-depths.ts';
import {FormLodLifetimes} from './form-lod-lifetimes.ts';
import {PhysicsContactHost} from './physics-contact-host.ts';
import {FieldReadback,type ConstructionFieldReadback,type DrawingFieldView} from './field-readback.ts';
import type {NativeWhitespacePolicy} from './text/native-whitespace.ts';
import {SEMANTIC_TEXT_DOMAIN,type NativeLabel} from './semantic-text.ts';
import {collectNativeLabels} from './shallot-text.ts';
import {createFrameStateNotice,disposeFrameState,publishFrameState,publishFrameStateNotice} from './frame-state.js';
import {applyMobileUiAppearance,supportsTouchControls} from './input/controls.ts';
import {exportResidentFrame,residentFrameState,type ResidentNativeSnapshot,type ResidentFrameState} from './resident-native-frame.ts';
import {AppearanceHash,StillParts,StillWorld} from './still-view.ts';

interface ComparisonView {readonly camera:string;readonly rect:{x:number;y:number;width:number;height:number};readonly extent:readonly [number,number];readonly clipFromWorld:readonly number[];readonly viewFromWorld:readonly number[];readonly sources:readonly number[]}
interface ComparisonSource {readonly world:number;readonly target:number;readonly serial:number;readonly engineFrame:number;readonly entities:readonly (readonly [number,number])[];readonly views:readonly ComparisonView[]}
export interface MainTextFrame extends PresentedNativeFrame {readonly comparison:ComparisonSource;readonly construction:{readonly sourceLines:number;readonly omitted:readonly string[];readonly metrics:NativeFrame['metrics']}}
type ResidentMainSnapshot=ResidentNativeSnapshot&{readonly comparison:ComparisonSource};

type SourceDefinition=Pick<CompositionGeometry,'positions'|'indices'|'revision'>;
interface CameraRecord {played?:Promise<ArticulationResult>;toneLiterals?:{resolve:(literals:readonly ArticulationLiteral[],tone?:GpuToneMembers)=>void;reject:(error:unknown)=>void};tone?:ToneReadback;toneSource?:{field:SurfaceField;clip:Float32Array;release?:()=>void};classes?:readonly SourceToneClass[];geometry?:readonly CompositionGeometry[];definitions?:ReadonlyMap<number,SourceDefinition>;pose?:DrawingFieldView['pose'];light?:readonly number[];fog?:readonly number[];inks?:Uint32Array;labels?:readonly NativeLabel[];field?:ConstructionFieldReadback;unresolved?:readonly string[];staging?:GPUBuffer;relationReadback?:ProjectedRelationReadback;partAnchorDemand?:PartAnchorDemandReadback;camera:string;cameraEntity:number;depth:GPUTextureView;rect:{x:number;y:number;width:number;height:number};width:number;height:number;capacity:number;intervalCapacity:number;visibility:number;owners:Map<number,number>}
interface TargetState {publication:NativePublication<MainTextFrame>;presentation:NativePresentation;resident?:ResidentMainSnapshot;still?:{parts:StillParts;world:StillWorld;frame:number;serial:number};appearance?:AppearanceHash;native?:GPUTexture;scaledPage?:GPUTexture;pending:boolean;serial:number;width:number;height:number;frame?:NativeFrame;error?:string}
interface TargetJob {cpuEncodeMs?:number;state:TargetState;source:GPUTexture;native?:GPUTexture;mask:GPUTexture;records:CameraRecord[];width:number;height:number;serial:number;engineFrame:number;error?:string;comparison:ComparisonSource}
export interface MainRendererController {
  ascii:boolean;underlay:boolean;overlay:boolean;mode:PresentationMode;
  paused:boolean;pause(paused:boolean):void;
  rows:number|undefined;
  readonly pending:boolean;
  readonly pendingStage?:string;
  readonly pendingSince?:number;
  onState?:()=>void;
  readonly frames:ReadonlyMap<number,MainTextFrame>;
  readonly residentFrames:ReadonlyMap<number,ResidentFrameState>;
  exportText():Promise<readonly MainTextFrame[]>;


  requestPresentation():Promise<void>;
  readonly errors:readonly string[];
  readonly graphs:ReadonlyMap<number,readonly PosedGraph[]>;
}
interface MainState {
  matrix:CellMatrix;
  status?:HTMLElement;notice?:HTMLElement;
  transport:MaterialGeometryTransport;graph:SurfaceRelations;sources:MeshSourceCache<SourceComplexRetention>;visibility:RelationVisibility[];
  relationExtent:number;intervalExtents:number[];
  construction:NativeConstruction;playback?:GpuArticulation;fields:Map<number,SurfaceFields>;blit:TextureBlit;mask:ConstructionMask;targets:Map<number,TargetState>;jobs:TargetJob[];
  displayedSnapshots:Map<number,ResidentMainSnapshot>;retiredSnapshots:ResidentMainSnapshot[];
  registration?:Promise<void>;
  primaryFields:WeakMap<View,{field:SurfaceField;frame:number;ready:boolean;materials:readonly MaterialGeometryDraw[];selection:SourceLodSelection|undefined}>;
  lodLifetimes:FormLodLifetimes;
  physicsContacts?:PhysicsContactHost;contactSourcesFrame?:number;
  controller:{ascii:boolean;underlay:boolean;overlay:boolean;mode:PresentationMode;paused:boolean;rows:number|undefined;readonly pending:boolean;readonly pendingStage?:string;readonly pendingSince?:number;onState?:()=>void;pause(paused:boolean):void;exportText():Promise<readonly MainTextFrame[]>;requestPresentation():Promise<void>;frames:Map<number,MainTextFrame>;residentFrames:Map<number,ResidentFrameState>;graphs:Map<number,readonly PosedGraph[]>;errors:string[]};world:number;serial:number;disposed:boolean;liveOwners:Set<number>;semanticOwners:Set<number>;pauseLatched:boolean;
}
let nextWorld=0;
const stateKey={create:()=>({current:undefined as MainState|undefined})};
const glyphs=nativeFont.glyphs;

/** Admit the relation capability from registered producer metadata after all
 * scene/plugin initialize hooks and before MAIN prepares its GPU closures. */
export function exampleGpuRelationProfile(world:World):ExampleGpuRelationProfile{
 for(const mesh of world.resource(Meshes).values()){
  const source=mesh.gpuIndexedSource;
  if(source?.edgeContactClass==='posed-coincident'&&source.topology.indexPattern==='independent-quads')return 'producer-capable';
 }
 return 'ordinary';
}

function fail(state:MainState,error:unknown):void{
  const message=error instanceof Error?error.message:String(error);if(state.controller.errors.at(-1)===message)return;state.controller.errors.push(message);if(state.controller.errors.length>64)state.controller.errors.shift();console.error('MAIN construction:',message);
  if(state.status)publishFrameState(state.controller,state.status,state.notice);
}
/** Resolve phase identity once from the same material cohort for both the
 * primary appearance and the construction consumer. No scene-specific path. */
function fieldInputs(world:World,draws:readonly import('@dylanebert/shallot/rendering').PhaseGeometryDraw[],resident:readonly {draw:MaterialGeometryDraw;source?:ReturnType<MeshSourceCache<SourceComplexRetention>['acquire']>}[]){
 const byMesh=new Map<number,(typeof resident)[number]>();
 const byIndirect=new Map<GPUBuffer,Map<number,(typeof resident)[number]>>(),byIndices=new Map<GPUBuffer,Map<number,(typeof resident)[number]>>();
 for(const item of resident){
  const draw=item.draw;if(draw.mesh!==undefined&&!byMesh.has(draw.mesh))byMesh.set(draw.mesh,item);
  for(const [table,buffer,offset]of [[byIndirect,draw.indirect,draw.indirectOffset],[byIndices,draw.indices,draw.indexBase??0]] as const){
   let offsets=table.get(buffer);if(!offsets){offsets=new Map();table.set(buffer,offsets);}if(!offsets.has(offset))offsets.set(offset,item);
  }
 }
 return draws.map(draw=>{
  const standard=draw.renderer.geometryAdapter!=='standard-material'?undefined:draw.stage==='transparent'&&world.has(draw.stableId,MeshInstance)
   ?byMesh.get(world.storage(MeshInstance).mesh.get(draw.stableId))
   :draw.command==='draw-indexed-indirect'?byIndirect.get(draw.arguments[0] as GPUBuffer)?.get(draw.arguments[1] as number)
   :draw.command==='draw-indexed'&&draw.index?byIndices.get(draw.index.buffer)?.get(Number(draw.arguments[2])+(draw.index.offset??0)/(draw.index.format==='uint16'?2:4)):undefined;
  return {draw,sourceDomain:standard?.source?.domain};
 });
}
function displayView(view:Pick<View,'canvas'|'context'>):boolean{
 // attachTexture is scene data (reflections, trails, height maps), not a
 // second user presentation. Only attachCanvas opts into MAIN replacement.
 // Stacked/viewport cameras inherit the same canvas/context from Core.
 return !!view.canvas&&!!view.context;
}
function frameOwners(world:World,state:MainState):Map<number,number>{
 return state.lodLifetimes.ownersForFrame(world.gpu.frame,()=>new Map(world.entities().map(eid=>[eid,world.ref(eid)])));
}
/** Physics ownership opts sources into worker compilation, never into a
 * render-time source scan. Shared meshes need only one revision-owned index. */
function prepareContactSources(world:World,state:MainState):void{
 if(state.contactSourcesFrame===world.gpu.frame)return;state.contactSourcesFrame=world.gpu.frame;
 for(const eid of world.query([Shape,MeshInstance])){
  state.physicsContacts??=new PhysicsContactHost();
  state.sources.requestContactCorners(world.storage(MeshInstance).mesh.get(eid));
 }
}
/** Selection precedes source incidence and material extraction. The engine
 * owns conservative posed bounds; the native field still owns visibility.
 * Unknown/custom sources cannot certify an exterior view. */
function sourceLod(world:World,view:View,materials:readonly MaterialGeometryDraw[],state:MainState,eid:number):SourceLodSelection|undefined{
 if(!state.playback)return;
 frameOwners(world,state);
 const staging=world.resource(RenderContext).viewStaging,offset=view.slot*VIEW_BYTES/4;
  const clip=staging.subarray(offset,offset+16),rect0=view.viewport??{width:view.surfaceTexture?.width??view.width,height:view.surfaceTexture?.height??view.height};

  // native cell is ONE far representation; otherwise its children are visited.
  const workset=sourceViewWorkset(world,view.slot,staging.subarray(offset+32,offset+35),bounds=>fitsParentTier(bounds,clip,view.width,view.height,rect0.width,rect0.height));if(!workset)return;
 const phases=world.resource(PhaseGeometryCapture).views.get(view)??[];
 const known=materials.every(draw=>draw.mesh!==undefined&&!!draw.sourceGeometry&&!draw.gpuIndexedSource)&&phases.every(draw=>draw.renderer.geometryAdapter==='standard-material');
 const rect=view.viewport??{width:view.surfaceTexture?.width??view.width,height:view.surfaceTexture?.height??view.height};

 // semantic residual, this revision); unknown keeps the incumbent (not safe).
 const profiles=new Map(materials.flatMap(draw=>draw.mesh!==undefined?[[draw.mesh,state.sources.profileSafe(draw)===true] as const]:[]));
 const profileSafe=(mesh:number)=>profiles.get(mesh)===true;
 // Groups and their exact ids are cached per world; the hold is per host LOD scope.
 let cache=parentCaches.get(world);if(!cache){cache=new ParentGroupCache();parentCaches.set(world,cache);}
 const holdScope=state.lodLifetimes.scope(eid,world.ref(eid),0,false);
 // This workset names entity placements, so split-straddling singles may
 // join certified parents. Producer-row adapters keep the selector's default.
 const drawsByMesh=new Map<number,MaterialGeometryDraw>();for(const draw of materials)if(draw.mesh!==undefined&&!drawsByMesh.has(draw.mesh))drawsByMesh.set(draw.mesh,draw);
 const decorate=(value:typeof workset.instances[number])=>{const draw=value.mesh===undefined?undefined:drawsByMesh.get(value.mesh);return {...value,construction:draw?state.sources.wholeConstruction(draw,value.mesh,value.eid,value.worldFromSource):undefined,smallParent:draw?state.sources.smallParent(draw):undefined};};
 const instances=workset.instances.map(decorate),parents=workset.parents?.map(parent=>({...parent,instances:parent.instances.map(decorate)}));
 // The source selector asks the already-prepared resident grammar, never
 // rebuilds/searches the authoring role bank. GpuArticulation owns this seam.
 const nativePart=state.playback?.wholePartAvailable.bind(state.playback);

 // the complete source size the form-table writer resolves held levels from.
 const bounds=workset.instances.flatMap(instance=>instance.bounds&&instance.bounds.length>=6&&!instance.boundsPending?[{mesh:instance.mesh,eid:instance.eid,bounds:instance.bounds}]:[]);
 const selection=selectSourceLod(instances,staging.subarray(offset,offset+16),view.width,view.height,rect.width,rect.height,workset.exterior&&known,staging[offset+30]!==0,parents,parentHolds.get(holdScope),profileSafe,member=>world.ref(member),cache,true,
  (parent,rows,width)=>state.playback?.smallConstructionAvailable(parent,rows,width)===true,parentHolds.getSmall(holdScope),nativePart);
 parentHolds.set(holdScope,selection.heldParents??new Set(),selection.heldSmall??new Set());

 // lifetime, engine frame); a path that returns no selection supplies none.
 sourceBoundsBySelection.set(selection,{world,camera:world.ref(eid),frame:world.gpu.frame,bounds});
 return selection;
}

const parentHolds=new ParentHolds();






interface SourceBoundsCohort {readonly world:World;readonly camera:number;readonly frame:number;readonly bounds:readonly {readonly mesh:number;readonly eid:number;readonly bounds:readonly number[]}[]}
const sourceBoundsBySelection=new WeakMap<SourceLodSelection,SourceBoundsCohort>();
/** The current cohort's bounds for this world/camera/frame, or none (every
 * owner then keeps the incumbent visible read). */
function currentSourceBounds(world:World,eid:number,selection:SourceLodSelection|undefined):SourceBoundsCohort['bounds']{
 const cohort=selection?sourceBoundsBySelection.get(selection):undefined;
 if(!cohort||cohort.world!==world||cohort.camera!==world.ref(eid)||cohort.frame!==world.gpu.frame)return [];
 return cohort.bounds;
}

const parentCaches=new WeakMap<World,ParentGroupCache>();

/** Runs inside the engine colour phase, BEFORE its postFX. The original
 * MainPass records commands but does not raster a second beauty image. */
function primaryAppearance(world:World,view:View,eid:number,state:MainState):void{
 // The target already retains the previous publication while this job runs.
 // No construction consumer can use another field until pending is cleared
 // (the same synchronous World step checks it again in build). Still record
 // the phases, and keep primary selected: returning to native beauty here
 // would reinstate the duplicate renderer. Comparison takes its own branch.
 if(state.controller.paused||(view.target!==undefined&&state.targets.get(view.target)?.pending))return;
 prepareContactSources(world,state);
 const encoder=world.frameEncoder()!,materials:MaterialGeometryDraw[]=[];
 visitMaterialGeometry(world,view.slot,draw=>materials.push(draw));
 const selection=sourceLod(world,view,materials,state,eid);
 const registration=state.sources.prepare(materials,selection);
 if(registration&&state.registration!==registration){
  state.registration=registration;
  void registration.then(()=>{if(state.registration===registration)state.registration=undefined;},error=>{if(state.registration===registration)state.registration=undefined;if(!state.disposed)fail(state,error);});
 }
 // Pending sources have no registered construction domain yet. Their exact
 // material still draws into the primary field using producer identity; no
 // text job consumes that field until a later, fully registered cohort.
 const resident=materials.map(draw=>({draw,source:registration?undefined:state.sources.acquire(draw,selection)}));
 let fields=state.fields.get(eid);if(!fields){fields=new SurfaceFields(world.gpu.device!);state.fields.set(eid,fields);}
 const offset=view.slot*VIEW_BYTES/4,staging=world.resource(RenderContext).viewStaging;
 const color=unpackColor(world.storage(Camera).clearColor.get(eid));
 const field=fields.begin(encoder,view.width,view.height,new Float32Array(staging.slice(offset+36,offset+52)),{primary:true,clearValue:view.sortedIndexForTarget?[0,0,0,0]:[color.r,color.g,color.b,1]});
 fields.appendAll(encoder,field,fieldInputs(world,world.resource(PhaseGeometryCapture).views.get(view)??[],resident));
 // Compiled MAIN consumes the retained material field directly. It does not
 // need a full-canvas colour image or the original post-processing chain.
 // The explicit legacy CPU adapter still reads completed appearance; keep
 // that input until its consumer is replaced, not as a default render.
 if(!state.playback)state.blit.snapshot(encoder,field.textures[4]!,view.framebuffer!,world.resource(RenderContext).format);
 state.primaryFields.set(view,{field,frame:world.gpu.frame,ready:!registration,materials,selection});
}

function build(world:World):void{
  const state=world.resource(stateKey).current;if(!state)return;
  // World.pause freezes virtual time, not draw. Stop producing jobs while the
  // already submitted job drains; otherwise rAF observers see pending forever.
  if(state.controller.paused){state.jobs=[];return;}
  prepareContactSources(world,state);
  const cpuStart=performance.now(),encoder=world.frameEncoder()!,device=world.gpu.device!,serial=++state.serial;
  const owners=frameOwners(world,state);
  state.playback?.retainCameras(new Set([...world.query([Camera])].map(eid=>`camera-${world.ref(eid)}`)));
  state.liveOwners=new Set(owners.values());
  state.semanticOwners=new Set([...world.query([Text])].map(eid=>world.ref(eid)));
  state.sources.collect(mesh=>world.resource(Meshes).has(mesh as MeshHandle));
  state.construction.retire(owner=>state.sources.isLive(owner.domain)&&(owner.kind==='producer-row'||state.liveOwners.has(owner.lifetime)));

  // field every publication; it owns no registered source to outlive.
  state.targets.forEach(target=>target.presentation.retire(owner=>owner.source===SEMANTIC_TEXT_DOMAIN?state.semanticOwners.has(owner.reference):owner.source===0||state.sources.isLive(owner.source)&&(owner.kind==='producer-row'||state.liveOwners.has(owner.reference))));
  const jobs=new Map<number,TargetJob>();
  if(!state.controller.ascii){
    state.jobs=[];
    pruneInactiveSurfaceFields(state.fields,new Set<number>(),[...state.targets.values()].some(target=>target.pending));
    return;
  }
  // Register every source needed by every camera before encoding any of a
  // target's views. An arrival never compiles on the drawing/event thread,
  // and an unready source never becomes a partial camera publication.
  const materialViews=new Map<number,readonly MaterialGeometryDraw[]>(),sourceLods=new Map<number,SourceLodSelection>();let installed=0;
  for(const [eid,view]of world.resource(Views)){
    if(!displayView(view)||!view.present||!view.depth||!view.surfaceTexture||view.target===undefined)continue;
    const primary=state.primaryFields.get(view),same=primary?.frame===world.gpu.frame;
    // Relations consume the exact source cohort and representation decision
    // used by their field, not a second traversal/hold update in this frame.
    let materials:readonly MaterialGeometryDraw[],selection:SourceLodSelection|undefined;
    if(same){materials=primary.materials;selection=primary.selection;}
    else{const current:MaterialGeometryDraw[]=[];visitMaterialGeometry(world,view.slot,draw=>current.push(draw));materials=current;selection=sourceLod(world,view,materials,state,eid);}
    materialViews.set(eid,materials);if(selection)sourceLods.set(eid,selection);
  }
  // A view/camera can leave the World without ending the World itself.  Do not
  // leave its five native field textures owned forever; a pending target keeps
  // all owners until its SAME-pose job releases and submits them.
  pruneInactiveSurfaceFields(state.fields,new Set(materialViews.keys()),[...state.targets.values()].some(target=>target.pending));
  try{
    const selection=sourceLods.size?mergeSourceLod([...materialViews].map(([eid,materials])=>({selection:sourceLods.get(eid)??{far:new Map(),defer:new Set(),exterior:false},meshes:materials.flatMap(draw=>draw.mesh===undefined?[]:[draw.mesh])}))):undefined;
    const registration=state.sources.prepare([...materialViews.values()].flat(),selection);
    if(registration){
      if(state.registration!==registration){
        state.registration=registration;
        void registration.then(()=>{if(state.registration===registration)state.registration=undefined;},error=>{if(state.registration===registration)state.registration=undefined;if(!state.disposed)fail(state,error);});
      }
      state.jobs=[];return;
    }
    // A moved camera owns its current dependencies, not the previous view's
    // still-running background compilation. prepare keeps that work resident.
    state.registration=undefined;
    for(const source of state.sources.takeCompiled()){
      installed++;
      if(isGpuIndexedResident(source)||state.playback)continue;
      if(!source.complex)throw Error('Legacy construction requires full source inspection');
      state.construction.installSource({...source.geometry,owner:source.id},source.analysis);
    }
  }catch(error){state.jobs=[];fail(state,error);return;}
  // Upload ONE contact table for this encoder's entire camera cohort. Updating
  // a shared texture per view would make every encoded view read the last
  // camera's contacts when this encoder is finally submitted.
  try{if(state.physicsContacts){
    const contactSources=new Map<number,import('./source-cache.ts').ResidentSourceComplex|import('./source-cache.ts').ResidentSourceSummary>();
    for(const [eid,materials]of materialViews)for(const draw of materials){
      if(draw.mesh===undefined)continue;
      const source=state.sources.acquire(draw,sourceLods.get(eid));
      if(source&&!isGpuIndexedResident(source)&&!isCompiledFarForm(source)&&source.cornerIndex)contactSources.set(draw.mesh,source);
    }
    const patches=state.physicsContacts.update(world,mesh=>contactSources.get(mesh));
    state.graph.setContactPatches(patches.revision,patches.points);
  }}catch(error){state.jobs=[];fail(state,error);return;}
  // GPU counters keep counting through exhausted storage. Allocate from those
  // real extents before encoding a NEW coherent scene snapshot. Never pass a
  // truncated range (or an invented empty graph) to construction. Each job's
  // mesh, field, text and exported camera metadata still name its OWN pose.
  // STILL EXAMPLE VIEW (frame cost, 2026-11-13; still-view.ts). A target whose
  // every input equals its last started, published job's starts no job: the
  // displayed tape keeps being presented. Any difference, including one only
  // the engine's GPU work shows (appearance hash), starts a job as before.
  // A blank page (no glyph) or one with unresolved forms is never still:
  // blank is not an output (AGENTS finding 7); it keeps starting jobs.
  const frame=world.gpu.frame,stillParts=new Map<number,StillParts>(),stillTargets=new Set<number>(),viewStaging=world.resource(RenderContext).viewStaging;
  for(const [eid,view]of world.resource(Views)){
    if(!displayView(view)||!view.present||!view.depth||!view.surfaceTexture||view.target===undefined)continue;
    let parts=stillParts.get(view.target);
    if(!parts){parts=new StillParts();stillParts.set(view.target,parts);parts.push('controller',state.controller.ascii,state.controller.underlay,state.controller.overlay,state.controller.mode,state.matrix.rows,installed);parts.pushArray('light',keyLight(world));parts.pushArray('fog',sceneFog(world));parts.pushArray('inks',sceneMaterialInks(world));}
    parts.push('contacts',state.physicsContacts?.value.revision??0);
    const offset=view.slot*VIEW_BYTES/4,surface=view.surfaceTexture;
    parts.push('view',eid,view.slot,view.camera,view.width,view.height,surface.width,surface.height,surface.format,view.viewport?.x,view.viewport?.y,view.viewport?.width,view.viewport?.height,world.storage(Camera).antialias.get(eid));
    parts.pushArray('view',viewStaging.subarray(offset,offset+VIEW_BYTES/4));parts.pushArray('view',view.viewFromWorld);
    parts.push('labels',JSON.stringify(collectNativeLabels(world,eid,new Float32Array(viewStaging.slice(offset,offset+16)),view.viewport?{...view.viewport}:{x:0,y:0,width:surface.width,height:surface.height})));
    for(const draw of materialViews.get(eid)??[]){parts.push('materials',draw.name,draw.indices,draw.indirect,draw.indirectOffset,draw.vertexBase,draw.indexBase,draw.vertexCapacity,draw.instanceCapacity,draw.cullMode,draw.mesh,draw.sourceGeometry?.revision,draw.sourceGeometry?.positions,draw.vertexCode);parts.push('material-groups',...draw.groups);}
    for(const draw of world.resource(PhaseGeometryCapture).views.get(view)??[]){
      parts.push('phase',draw.renderer,draw.stage,draw.stableId,draw.stablePart,draw.ordinal,draw.pipeline,draw.recipe,draw.command,draw.index?.buffer,draw.index?.format,draw.index?.offset,draw.index?.size);
      for(const argument of draw.arguments)if(Array.isArray(argument))parts.push('phase',argument.length,...argument);else parts.push('phase',argument);
      for(const vertex of draw.vertices)parts.push('phase',vertex.slot,vertex.buffer,vertex.offset,vertex.size);
      parts.pushArray('phase',draw.viewport);parts.pushArray('phase',draw.scissor);
      for(const group of draw.groups){parts.push('phase-groups',group.index,group.group);parts.pushArray('phase-groups',group.offsets);}
    }
  }
  for(const [target,parts]of stillParts){
    const targetState=state.targets.get(target),last=targetState?.still;
    if(targetState?.pending)continue;
    const reason=!targetState||!last?'first':targetState.resident?.serial!==last.serial?'unpublished':!targetState.resident!.results.some(result=>(result.resident?.glyphCount??0)>0)||targetState.resident!.results.some(result=>result.unresolved.length>0)?'blank':parts.differs(last.parts)||last.world.differs(world)||(targetState.appearance?.state(frame)??'appearance-pending');
    if(reason===''||reason==='appearance-wait'||reason==='appearance-blip')stillTargets.add(target);
  }
  if([...world.resource(Views).values()].some(view=>{const primary=state.primaryFields.get(view);return primary?.frame===world.gpu.frame&&!primary.ready;})){state.jobs=[];return;}
  state.graph.ensureCapacity(state.relationExtent);
  state.visibility.forEach((visibility,index)=>visibility.ensureCapacity(state.graph.lineCapacity,state.intervalExtents[index]!));
  // A target, including all of its stacked/viewport cameras, owns one coherent
  // publication. Character size is independent of camera render resolution.
  // STACKED CAMERAS (patch_vmocclude): each target's views highest Camera.order
  // first, so a camera's articulation is sealed before the cameras it covers;
  // targets keep their first-seen order. One camera per target is unchanged.
  const cameraOrder=(eid:number)=>world.storage(Camera).order.get(eid);
  const byTarget=new Map<number,[number,View][]>();
  for(const [eid,view]of world.resource(Views)){if(!displayView(view)||!view.present||!view.depth||!view.surfaceTexture||view.target===undefined)continue;let list=byTarget.get(view.target);if(!list){list=[];byTarget.set(view.target,list);}list.push([eid,view]);}
  const presenting=[...byTarget.values()].flatMap(list=>list.length>1?[...list].sort(([a],[b])=>cameraOrder(b)-cameraOrder(a)):list);
  for(const [eid,view]of presenting){
    if(!displayView(view)||!view.present||!view.depth||!view.surfaceTexture||view.target===undefined)continue;
    const target=view.target;let targetState=state.targets.get(target);
    if(!targetState){targetState={publication:new NativePublication<MainTextFrame>(device,glyphs),presentation:new NativePresentation(glyphs,nativeFont.fontSha256),pending:false,serial:0,width:0,height:0};state.targets.set(target,targetState);}
    targetState.width=view.surfaceTexture.width;targetState.height=view.surfaceTexture.height;
    if(targetState.pending||stillTargets.has(target))continue;
    let job=jobs.get(target);
    if(!job){
      const source=view.surfaceTexture,width=source.width,height=source.height;
      const native=state.controller.underlay||state.controller.overlay?device.createTexture({label:'MAIN same-frame original comparison',size:[width,height],format:source.format,usage:GPUTextureUsage.COPY_DST|GPUTextureUsage.TEXTURE_BINDING}):undefined;
      const mask=device.createTexture({label:'MAIN source construction mask',size:[width,height],format:'rgba8unorm',usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING});
      job={state:targetState,source,native,mask,records:[],width,height,serial,engineFrame:world.gpu.frame,
        comparison:{world:state.world,target,serial,engineFrame:world.gpu.frame,entities:[...owners],views:[]}};jobs.set(target,job);
    }
    try{
    const offset=view.slot*VIEW_BYTES/4;
    const staging=world.resource(RenderContext).viewStaging;
    const clip=new Float32Array(staging.slice(offset,offset+16));
    const matrix=view.viewFromWorld!;
    const materials=materialViews.get(eid)!;
    const resident=materials.map(draw=>({draw,source:state.sources.acquire(draw,sourceLods.get(eid))}));
    const geometry:CompositionGeometry[]=[];
    const meshSources=new Map(resident.filter(item=>item.source&&item.draw.mesh!==undefined).map(item=>[item.draw.mesh,item.source!]));
    if(!state.playback)for(const entity of world.query([MeshInstance,GlobalTransform])){
      const source=meshSources.get(world.storage(MeshInstance).mesh.get(entity));if(!source||isGpuIndexedResident(source)||isCompiledFarForm(source))continue;
      const transform=new Float32Array(16);composeGlobalTransform(world,entity,transform);
      geometry.push({index:entity,owner:posedOwnerId({camera:`camera-${view.camera}`,domain:source.domain,kind:'entity',lifetime:world.ref(entity)}),positions:source.geometry.positions,indices:source.geometry.indices,revision:source.revision,worldFromSource:transform});
    }
    const selectedSources=state.playback?new Set<string>():state.construction.selectSources(geometry);
    for(let i=0;i<geometry.length;i++)geometry[i]={...geometry[i]!,primary:selectedSources.has(geometry[i]!.owner)};
    const selectedEntities=geometry.filter(g=>g.primary).map(g=>g.index!);
    const pose={eye:Array.from(staging.slice(offset+32,offset+35)),forward:[-matrix[2]!,-matrix[6]!,-matrix[10]!],focalPixels:Math.hypot(clip[1]!,clip[5]!,clip[9]!)*view.height/2,perspective:staging[offset+30]!==0};
    let fields=state.fields.get(eid);if(!fields){fields=new SurfaceFields(device);state.fields.set(eid,fields);}
    const primary=state.primaryFields.get(view);
    if(primary&&(primary.frame!==world.gpu.frame||!primary.ready))throw Error('MAIN primary material field is not ready for this same-frame source cohort');
    const field=primary?.field??fields.begin(encoder,view.width,view.height,new Float32Array(staging.slice(offset+36,offset+52)));
    if(!primary)fields.appendAll(encoder,field,fieldInputs(world,world.resource(PhaseGeometryCapture).views.get(view)??[],resident));
    // The field is complete: reduce its one per-form nearest-depth table
    // before the relation, underdrawing, tone and articulation passes read it.

    // scoped to this camera lifetime and world, with the same rows-per-pixel
    // the relation stage measures (17 native pixels per exported row).
    const lodPage=view.viewport?[view.viewport.width,view.viewport.height]:[job.width,job.height];
    // Publish the exact far-copy frontier to fine tone/texture readers too.
    // Visibility cannot promote an already-replaced child or demote a near copy.
    const coarseOwners=new Map<string,readonly [number,number]>(),viewLod=sourceLods.get(eid);
    if(state.playback&&viewLod?.exterior)for(const {draw,source}of resident){
      if(!source||draw.mesh===undefined||isGpuIndexedResident(source))continue;
      const members=viewLod.far.get(draw.mesh);if(!members?.length)continue;
      const form=isCompiledFarForm(source)?source:viewLod.full?.has(draw.mesh)?state.sources.farForm(draw):undefined;
      if(!form)continue;
      for(const member of members)coarseOwners.set(`${form.domain}/${member.eid}`,[form.domain,member.eid]);
    }
    const coarseGroups:FormParentGroup[]=coarseOwners.size?[{members:[...coarseOwners.values()],sourceSelected:true}]:[];

    // source domain, its eid) with a current posed engine bound. Owners whose
    // source or bound is unknown keep the visible incumbent.

    // sourceDomain | UV bit, the material's eid output = the ECS entity id), and
    // fieldInputs resolves the domain per drawn material entry, not per mesh.
    // The bound is therefore offered under EVERY resident domain of its mesh; a
    // key the field never wrote holds no pixel and no reader asks for it.
    const domains=new Map<number,Set<number>>();for(const {draw,source}of resident)if(source&&draw.mesh!==undefined&&!isGpuIndexedResident(source)){let set=domains.get(draw.mesh);if(!set){set=new Set();domains.set(draw.mesh,set);}set.add(source.domain);}
    const sourceBounds:FormSourceBound[]=(state.playback?currentSourceBounds(world,eid,viewLod):[]).flatMap(({mesh,eid:owner,bounds})=>[...(domains.get(mesh)??[])].map(domain=>
     ({owner:[domain,owner] as const,low:[bounds[0]!,bounds[1]!,bounds[2]!] as const,high:[bounds[3]!,bounds[4]!,bounds[5]!] as const})));
    formDepths(device).encode(encoder,field,!state.playback||!(lodPage[1]!>0)?undefined:state.lodLifetimes.scope(eid,world.ref(eid),17*field.height/lodPage[1]!),coarseGroups,sourceBounds);
    const unresolved:string[]=[];const gpuPoses=new Map<number,GPUBufferBinding>();
    const relationView={clipFromWorld:clip,eye:pose.eye,direction:pose.forward,perspective:pose.perspective};
    state.graph.begin(encoder,{...relationView,pageExtent:view.viewport?[view.viewport.width,view.viewport.height]:[job.width,job.height]},field);

    // material draws own texels of THIS field. Their unseen entity instances
    // produce no relations (surface-relations.ts instanceUnseen).
    const relationFieldDomains=new Set(fieldInputs(world,world.resource(PhaseGeometryCapture).views.get(view)??[],resident).flatMap(input=>input.sourceDomain===undefined?[]:[input.sourceDomain]));
    state.graph.fieldDomains=relationFieldDomains;
    const materialSources=new Map<MaterialGeometryDraw,ResidentRelationSource>();
    const mixedFar=(draw:MaterialGeometryDraw)=>{const lod=sourceLods.get(eid);if(draw.mesh===undefined||!lod?.full?.has(draw.mesh))return;const members=lod.far.get(draw.mesh);const form=members?state.sources.farForm(draw):undefined;return members&&form?{members,form}:undefined;};
    const fullMembers=(draw:MaterialGeometryDraw)=>mixedFar(draw)?sourceLods.get(eid)!.full!.get(draw.mesh!)!:undefined;
    for(const {draw,source}of resident){
      if(!source){
        if(draw.mesh===undefined||!sourceLods.get(eid)?.outside?.has(draw.mesh))unresolved.push('GPU topology pending; incumbent field construction retained');
        continue;
      }
      if(isCompiledFarForm(source))continue; // emitted whole from this SAME native field below
      if(isGpuIndexedResident(source)&&source.state==='empty')continue;
      materialSources.set(draw,source);
    }
    state.transport.encodeAll(encoder,[...materialSources.keys()],batches=>{
      const posedBatches:{batch:RetainedMaterialGeometryBatch;source:ResidentRelationSource;domain:number}[]=[];
      for(const retained of batches){
        const source=materialSources.get(retained.source)!;
        if(isGpuIndexedResident(source)){
          if(!retained.gpuPose)throw Error('GPU indexed source '+source.id+' has no material-stage pose');
          if(!gpuPoses.has(source.domain)){
            gpuPoses.set(source.domain,state.sources.pose(source,retained.gpuPose).pose.vertices);
          }
        }
        posedBatches.push({batch:retained,source,domain:source.domain});
      }
    // Native GPU counts still own every draw. Unlike physical static batches,
    // these must not host-reserve faces from instance/vertex capacities.
      state.graph.reservePartAnchors(posedBatches);
      // The engine's packed membership and exact indirect slice are the only
      // certified native frontier. Compact it before contacts, relations,
      // anchors and interior work; those consumers reuse the same list.
      state.graph.admitFrontiers(encoder,posedBatches);
      state.graph.appendContactsAll(encoder,posedBatches);
      const pass=encoder.beginComputePass({label:'MAIN source relations'});

      // only for its full members (GPU owner selection); its far members are
      // the far-form class below. Without a far form every member stays full.
      try{for(const item of posedBatches){const selectedOwners=item.batch.source.nativeVisibleInstanceFrontier?undefined:state.playback?fullMembers(item.batch.source):selectedEntities;state.graph.append(encoder,item.batch,item.source,relationView,item.domain,selectedOwners,pass);}}
      finally{pass.end();}
    });
    const partAnchorDemand=state.graph.finish(encoder);
    const multisampled=world.storage(Camera).antialias.get(eid)!==0;
    const visible=state.visibility[Number(multisampled)]!;
    visible.encode(encoder,state.graph,view.depth,clip,view.width,view.height,field);
    const rect=view.viewport?{...view.viewport}:{x:0,y:0,width:job.width,height:job.height};
    const camera=`camera-${view.camera}`;
    job.comparison={...job.comparison,views:[...job.comparison.views,{camera,rect:{...rect},extent:[view.width,view.height],
      clipFromWorld:Array.from(clip),viewFromWorld:Array.from(matrix),sources:resident.flatMap(({source})=>source?[source.domain]:[])}]};
    // Snapshot the geometry construction for EVERY text job, not the current
    // UI flag. The compiled tone producer reads this texture's D10 boundaries
    // through `markAt`; it is semantic input, not an optional overlay. The

    // original scene (ASCII off) or an underdrawing/overlay view; normal
    // compiled ASCII keeps its required field mask without presenting it.

    state.mask.encode(encoder,visible,job.mask,job.width,job.height,rect,job.records.length===0,state.playback?field:undefined,keyLight(world));
    const labels=collectNativeLabels(world,eid,clip,rect);
    // One material-ink table for this view's lines and tone (MATERIAL INK).
    const inks=sceneMaterialInks(world);state.playback?.setMaterialInks(inks);
    const classes=state.playback?resident.flatMap<SourceConstructionClass>(({draw,source})=>{
      if(!source)return [];
      if(isCompiledFarForm(source)){
        const instances=sourceLods.get(eid)?.far.get(draw.mesh!);if(!instances)throw Error('Smallest source form lost its same-view selection');
        return [{domain:source.domain,topologyRevision:source.revision,...source.analysis,farForms:{domain:source.domain,instances}}];
      }
      if(isGpuIndexedResident(source)){
        if(source.state==='empty')return [];
        const posed=gpuPoses.get(source.domain);if(!posed)throw Error('GPU indexed source '+source.id+' has no same-encoder pose');
        const kind=source.gpu.constructionKind;if(!kind)throw Error('GPU indexed source '+source.id+' has no producer construction kind');
        return [{domain:source.domain,type:kind,topologyRevision:source.generation,gpuIndexed:{domain:source.domain,revision:source.revision,generation:source.generation,kind,indexPattern:source.gpu.topology.indexPattern,topology:{indices:source.gpu.topology.indices,indexBase:0,indexCount:source.gpu.incidence.layout.corners,vertexCapacity:source.gpu.topology.vertexCapacity},incidence:{binding:source.gpu.incidence.binding,layout:{...source.gpu.incidence.layout,edges:source.gpu.incidence.edges}},pose:{vertices:posed}}}];
      }
      const mixed=mixedFar(draw);
      return [{domain:source.domain,topologyRevision:source.geometry.revision,...source.analysis},...(mixed?[{domain:mixed.form.domain,topologyRevision:mixed.form.revision,...mixed.form.analysis,farForms:{domain:mixed.form.domain,instances:mixed.members}}]:[])];
    }):[];

    // adequate octree node, its members listed by the domain their pixels carry
    // (a wholly far mesh's compiled far form, or a mixed mesh's far form). A
    // member whose mesh resolved to the full producer draws itself and is not
    // listed. Plus the producer aggregate over the domains whose producer rows
    // the relation stage gated this frame (ownerMerged: the field's own domains
    // in a perspective view). The far aggregate emitter (gpu-source-far-forms)
    // writes each as ONE far run; articulation's farKind/farSkyline draws it.
    const toneSourceClasses=[...classes]; // the carrier below is not a tone class
    if(state.playback){
      const farDomain=new Map<number,number>();
      for(const {draw,source}of resident){if(!source||draw.mesh===undefined)continue;if(isCompiledFarForm(source))farDomain.set(draw.mesh,source.domain);else{const mixed=mixedFar(draw);if(mixed)farDomain.set(draw.mesh,mixed.form.domain);}}
      const fieldOwner=(mesh:number,eid:number)=>{const domain=farDomain.get(mesh);return domain===undefined?undefined:[domain,eid] as const;};
      const aggregates=(sourceLods.get(eid)?.parents??[]).map(node=>({key:node.key,box:node.box,construction:remapSourceWholeConstruction(node.construction,fieldOwner),members:node.members.flatMap(member=>{const owner=fieldOwner(member.mesh,member.eid);return owner?[owner]:[];})})).filter(node=>node.members.length);
      const producerDomains=pose.perspective?[...state.graph.fieldDomains??[]]:[];
      if(aggregates.length||producerDomains.length){
        const farForms:SourceFarForms={domain:0,instances:[],aggregates,producers:{domains:producerDomains,perspective:pose.perspective}};
        (classes as SourceConstructionClass[]).push({domain:0,type:'far-aggregate',topologyRevision:0,farForms} as unknown as SourceConstructionClass);
      }
    }
    let toneLiterals:CameraRecord['toneLiterals'];
    const literals:ArticulationLiteralFeed={subscribe(resolve,reject){toneLiterals={resolve,reject};}};
    // Retain this camera's GPU runs now. Its compact same-pose tone joins
    // their fixed-priority placement after the appearance pass completes.
    const above=presenting.filter(([other,v])=>v.target===view.target&&cameraOrder(other)>cameraOrder(eid)).map(([,v])=>`camera-${v.camera}`);
    // A retry belongs to this target's exact prepared pose, not just a camera
    // or command encoder reused by another target/frame.
    const cohortKey=`world-${state.world}/target-${target}/serial-${job.serial}`;
    const played=state.playback?.enqueueSource(encoder,visible,classes,field,job.width,job.height,camera,owners,rect,literals,undefined,above,cohortKey,state.graph.partAnchorTable);
    // Rejection is awaited from publish after the host submit, never through
    // a private queue/fence. Attach a handler now to avoid unhandled failures.
    void played?.catch(()=>{});
    const relationReadback=undefined;
    let read:GPUBuffer|undefined;
    if(!state.playback){
      read=device.createBuffer({label:'MAIN visible geometry relations',size:16+visible.capacity*RELATION_WORDS*4+visible.intervalCapacity*16,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
      encoder.copyBufferToBuffer(visible.status,0,read,0,16);encoder.copyBufferToBuffer(visible.lines,0,read,16,visible.capacity*RELATION_WORDS*4);encoder.copyBufferToBuffer(visible.intervals,0,read,16+visible.capacity*RELATION_WORDS*4,visible.intervalCapacity*16);
    }
    const definitions=state.playback?undefined:new Map<number,SourceDefinition>();
    if(definitions)for(const {source}of resident)if(source&&!isGpuIndexedResident(source))definitions.set(source.domain,{positions:source.geometry.positions,indices:source.geometry.indices,revision:source.revision});
    // D6 region tone from the D10 bands: classified per page cell on the GPU;
    // only the compact member list comes back.
    const frameLight=keyLight(world),record:CameraRecord={played,toneLiterals:state.playback?toneLiterals:undefined,toneSource:{field,clip:new Float32Array(clip),release:state.playback?fields.retain(field):undefined},classes:toneClasses(world,toneSourceClasses,resident),geometry,definitions,pose,labels,field:state.playback?undefined:new FieldReadback(device,encoder,field),unresolved,staging:read,relationReadback,partAnchorDemand,camera,cameraEntity:eid,depth:view.depth,rect,light:frameLight,fog:sceneFog(world),inks,width:view.width,height:view.height,capacity:visible.capacity,intervalCapacity:visible.intervalCapacity,visibility:Number(multisampled),owners};
    job.records.push(record);
    }catch(error){job.error=error instanceof Error?error.message:String(error);job.state.error=job.error;fail(state,error);}
  }
  state.playback?.retainSources(state.sources.liveDomains());
  state.jobs=[...jobs.values()];
  for(const job of state.jobs){
    job.cpuEncodeMs=performance.now()-cpuStart;job.state.pending=true;
    const parts=stillParts.get(job.comparison.target);
    if(parts&&!job.error){job.state.still={parts,world:StillWorld.snapshot(world),frame,serial:job.serial};(job.state.appearance??=new AppearanceHash(device)).watch(frame);}else job.state.still=undefined;
  }
}

/** Unit direction toward the scene's brightest directional light (its
 * GlobalTransform local +Z, the reverse of the engine's travel direction). */
function keyLight(world:World):number[]|undefined{
  const rotation=world.storage(GlobalTransform).rotation,lights=world.storage(DirectionalLight);let best:number[]|undefined,brightest=-Infinity;
  for(const eid of world.query([DirectionalLight,GlobalTransform])){
    const illuminance=lights.illuminance.get(eid);if(illuminance<=brightest)continue;brightest=illuminance;
    const x=rotation.x.get(eid),y=rotation.y.get(eid),z=rotation.z.get(eid),w=rotation.w.get(eid);
    const t=[-2*(x*z+w*y),-2*(y*z-w*x),-1+2*(x*x+y*y)],length=Math.hypot(t[0]!,t[1]!,t[2]!)||1;
    best=[-t[0]!/length,-t[1]!/length,-t[2]!/length];
  }
  return best;
}

/** The scene's own fog medium (engine `Fog`): extinction density
 * (density × (absorption + scattering)), height base and height falloff. The
 * thickest medium when a scene has several; absent when the scene is clear.
 * The drawing's atmosphere (D3) thins region tone with view distance by it. */
function sceneFog(world:World):number[]|undefined{
  const fog=world.storage(Fog);let best:number[]|undefined;
  for(const eid of world.query([Fog])){
    const extinction=fog.density.get(eid)*(fog.absorption.get(eid)+fog.scattering.get(eid));
    if(extinction<=0||(best&&extinction<=best[0]!))continue;
    best=[extinction,fog.heightBase.get(eid),fog.heightFalloff.get(eid)];
  }
  return best;
}

/** THE FRAME'S MATERIAL INKS (examples-89; gpu-articulation.ts MATERIAL INK):
 * every drawn entity's own authored StandardMaterial base colour, reduced
 * once to a readable ink (none for a neutral material). Keyed by the field
 * owner's entity word, so a form's lines and tone take ONE hue: its material,
 * never the lit colour of one facet. A material of another type (no
 * StandardMaterial row) has no entry. */
function sceneMaterialInks(world:World):Uint32Array{
  const store=world.storage(MeshMaterial),rows=store.material,types=store.type,materials=world.resource(Materials),pairs:[number,number][]=[];
  // MeshMaterial.material is a row of the entity's OWN material type; Materials
  // holds StandardMaterial rows (type 0, the engine default). Another type's
  // row (checker, wave, extended) has no entry here (examples-98).
  const byRow=new Map<number,number>();
  for(const eid of world.query([MeshInstance,MeshMaterial])){
    if(types.get(eid)!==0)continue;
    const row=rows.get(eid);let ink=byRow.get(row);
    if(ink===undefined){
      let values:{baseColor?:ArrayLike<number>}|undefined;
      try{values=materials.get(row) as {baseColor?:ArrayLike<number>}|undefined;}catch{values=undefined;}
      const base=values?.baseColor;ink=base?materialInk([base[0]!,base[1]!,base[2]!]):0;byRow.set(row,ink);
    }
    if(ink)pairs.push([eid>>>0,ink]);
  }
  return materialInkTable(pairs);
}


// acquisition). Before BeginFrame acquires any canvas, MAIN/PresentationGate
// requests ONE ticket per canvas context only when what that canvas shows
// would change: a view MAIN does not replace (scene data, other canvases), a
// live-scene mode (ASCII off, mesh + underdrawing, overlay), no page or an
// error yet, a changed backing size / presentation mode, or a published page
// that is not the displayed one. Otherwise the canvas is not acquired, copied
// or presented (it keeps the page it shows, so exported text still equals the
// display); its original cameras, simulation and scene rendering continue on
// the engine's targetless stand-in, MAIN's jobs keep publishing there, and a
// new page earns the next ticket (no no-ticket deadlock). Stacked/viewport
// cameras share their canvas's one ticket; separate canvases own their own.
type CanvasPresentationView=View&{presentationTicket?:unknown;presentationTargetless?:boolean};
const presentedCanvas=new WeakMap<GPUCanvasContext,string>();

// request epochs, demands; presentation-ledger.ts, PhysicalFrame's mechanics).
const presentationLedgers=new WeakMap<object,PresentationLedger<GPUCanvasContext>>();
const presentationLedger=(controller:object)=>{let ledger=presentationLedgers.get(controller);if(!ledger){ledger=new PresentationLedger<GPUCanvasContext>();presentationLedgers.set(controller,ledger);}return ledger;};

// overlay), not only its size and text mode, so returning from a live-scene
// mode re-presents the page instead of keeping the mesh on screen.
const canvasKey=(state:MainState,view:View)=>{const c=state.controller;return `${view.canvas!.width}x${view.canvas!.height}|${c.mode}|${c.paused}|${c.ascii}|${c.underlay}|${c.overlay}`;};
function canvasChange(state:MainState|undefined,view:View):string{
  if(!state)return 'starting';
  if(presentationLedger(state.controller).demanded)return 'demand';
  if(!displayView(view))return 'scene-view';
  const controller=state.controller;
  if(!controller.ascii||controller.underlay||controller.overlay)return 'live-scene';
  if(presentedCanvas.get(view.context!)!==canvasKey(state,view))return 'target';
  if(view.target===undefined)return 'unbound';
  const target=state.targets.get(view.target);
  if(!target||target.error||!target.resident&&!target.frame)return 'first';
  if(target.resident)return state.displayedSnapshots.get(view.target)!==target.resident?'page':'';
  return target.publication.frame!==controller.frames.get(view.target)?'page':'';
}
function presentationGate(world:World):void{
  const state=world.resource(stateKey).current,reasons=new Map<GPUCanvasContext,string>();
  for(const view of world.resource(Views).values()){
    // A context without a canvas element is never MAIN-replaced ('scene-view'):
    // it gets its ticket every frame (it has no stand-in), as before management.
    const context=view.context;if(!context)continue;
    const reason=canvasChange(state,view);if(reason&&!reasons.has(context))reasons.set(context,reason);
  }
  for(const [context,reason]of reasons)requestCanvasPresentationTicket(world,context,reason);
}
function present(world:World,views:readonly View[]):void{
  const state=world.resource(stateKey).current;if(!state)return;const encoder=world.frameEncoder()!;

  // was honoured shows this frame's key, its targets' pages become displayed and
  // exported, and the demands up to this epoch resolve, ONLY when this
  // submission's frame fence fulfils (never at encode or submission).
  const ledger=presentationLedger(state.controller),{serial,epoch}=ledger.encode();
  const acknowledged:[GPUCanvasContext,string][]=[];
  for(const view of views)if(view.context&&view.canvas&&(view as CanvasPresentationView).presentationTicket)acknowledged.push([view.context,canvasKey(state,view)]);
  state.visibility.forEach(visibility=>visibility.encodeReadbacks(encoder));
  toneField(world.gpu.device!).encodeReadbacks(encoder);
  for(const job of state.jobs)if(!job.error){
    if(job.native)encoder.copyTextureToTexture({texture:job.source},{texture:job.native},[job.width,job.height]);
    for(const record of job.records){
      record.field?.completeAppearance(world.gpu.device!,encoder,job.source,record.rect);
      // The completed SAME-frame appearance includes the engine's cast
      // shadows, fog and exposure; classification must not run before it.
      const shadowView=state.playback&&sunShadowView(world),shadowParams=state.playback&&sunShadowParams(world),sampler=state.playback&&shadowSampler(world);
      const shadow=shadowView&&shadowParams&&sampler&&record.pose?{view:shadowView,params:shadowParams,sampler,eye:record.pose.eye,forward:record.pose.forward}:undefined;
      // Compiled tone uses the source field and light/caster data, not a
      // completed beauty-frame copy. A canvas texture is not sampleable.
      if(record.toneSource)record.tone=toneField(world.gpu.device!).encode(encoder,record.toneSource.field,record.toneSource.clip,record.rect,job.width,job.height,record.light,job.native,!!state.playback,shadow,job.mask,record.classes,record.fog,record.inks);
      if(record.toneLiterals&&record.tone?.immediate?.length===0){
        // GPU region records join the retained source runs on THIS host
        // encoder, with no count/member readback or next-frame detour.
        record.toneLiterals.resolve(state.construction.playbackLiterals(record.labels??[],[],[]),record.tone.gpu);record.toneLiterals=undefined;
      }
    }
  }
  state.playback?.encode(encoder);
  const seen=new Set<number>();
  const submittedFrames:[number,MainTextFrame][]=[];
  const submittedResident:[number,ResidentMainSnapshot][]=[];
  for(const view of views){
    if(!displayView(view)||view.target===undefined||seen.has(view.target)||!view.surfaceTexture||!view.surfaceView||!state.controller.ascii&&!state.controller.paused)continue;
    seen.add(view.target);const target=state.targets.get(view.target);
    // The finished appearance (before the page is drawn on it) of the frames
    // a still decision can need: the job's own frame and every other
    // non-pending frame (a change is confirmed by frames N, N+2, N+4).
    if(target?.appearance&&(target.appearance.watched?.frame===world.gpu.frame||!target.pending&&((world.gpu.frame-(target.appearance.watched?.frame??0))&1)===0))target.appearance.encode(encoder,view.surfaceTexture,world.gpu.frame);

    // keeps the page it shows; nothing is rasterized, copied or exported.
    if((view as CanvasPresentationView).presentationTargetless)continue;
    if(!target||target.error||!target.resident&&!target.frame){
      if(state.controller.ascii&&!state.controller.underlay&&!state.controller.overlay){
        const paper=encoder.beginRenderPass({label:'MAIN awaiting first text publication',colorAttachments:[{view:view.surfaceView,loadOp:'clear',storeOp:'store',clearValue:[1,1,1,1]}]});paper.end();
      }
      continue;
    }
    if(!state.controller.ascii){if(target.native)state.blit.snapshot(encoder,target.native,view.surfaceView,view.surfaceTexture.format);if(target.publication.frame)submittedFrames.push([view.target,target.publication.frame]);continue;}
    // Rasterize the exact exported page at its native dimensions, then present
    // it nearest-neighbour. During a resolution change the old coherent pair
    // remains visible until the new same-pose publication is ready.
    const width=target.resident?.width??target.frame!.width,height=target.resident?.height??target.frame!.height;
    const scaled=width!==view.surfaceTexture.width||height!==view.surfaceTexture.height;
    if(scaled&&(!target.scaledPage||target.scaledPage.width!==width||target.scaledPage.height!==height||target.scaledPage.format!==view.surfaceTexture.format)){
      target.scaledPage?.destroy();target.scaledPage=world.gpu.device!.createTexture({label:'MAIN native text before nearest presentation',size:[width,height],format:view.surfaceTexture.format,usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING});
    }
    const destination=scaled?target.scaledPage!.createView():view.surfaceView;
    const overlay=!!(target.native&&(state.controller.underlay||state.controller.overlay));
    if(overlay)state.blit.snapshot(encoder,target.native!,destination,view.surfaceTexture.format);
    if(target.resident){
      target.publication.encodeGpu(encoder,target.resident.results.map(result=>result.resident!.tape),overlay?target.native!.createView():undefined,destination,view.surfaceTexture.format,width,height,true,false,overlay);
      submittedResident.push([view.target,target.resident]);
    }else target.publication.encode(encoder,overlay?target.native!.createView():undefined,destination,view.surfaceTexture.format,width,height,true,false,overlay);
    if(scaled)state.blit.snapshot(encoder,target.scaledPage!,view.surfaceView,view.surfaceTexture.format);
    if(target.publication.frame)submittedFrames.push([view.target,target.publication.frame]);
  }
  // A candidate can be retired by a newer publication before this submission's
  // fence settles. Keep every such candidate alive through the owning fence,
  // not only snapshots that have already become DISPLAYED. The ledger's lease
  // is identity-refcounted so two pending submissions may share one snapshot.
  const releaseRetired=(snapshot:ResidentMainSnapshot):void=>{
    const displayed=new Set(state.displayedSnapshots.values());
    if(displayed.has(snapshot)||ledger.isSnapshotRetained(snapshot))return;
    const index=state.retiredSnapshots.indexOf(snapshot);if(index<0)return;
    state.retiredSnapshots.splice(index,1);
    for(const result of snapshot.results)result.resident?.release();
  };
  const snapshotLease=ledger.retainSnapshots(submittedResident.map(([,snapshot])=>snapshot),releaseRetired);
  const jobs=state.jobs;state.jobs=[];
  world.onFrameSubmit(()=>{

    // fence publishes nothing (the prior page stays displayed and exported,
    // demands stay pending), and an older fence never overwrites a newer serial.
    const commit=():void=>{
      if(state.disposed)return;
      for(const [context,key]of acknowledged)if(ledger.commitCanvas(context,serial))presentedCanvas.set(context,key);
      for(const [id,frame]of submittedFrames)if(ledger.commitTarget(id,serial))state.controller.frames.set(id,frame);
      for(const [id,snapshot]of submittedResident)if(ledger.commitTarget(id,serial)){state.displayedSnapshots.set(id,snapshot);state.controller.residentFrames.set(id,residentFrameState(snapshot));state.controller.frames.delete(id);}
      if(acknowledged.length)ledger.answer(epoch);
      if(state.status)publishFrameState(state.controller,state.status,state.notice);
    };
    const releaseRetiredSnapshots=():void=>{
      const displayed=new Set(state.displayedSnapshots.values()),kept:ResidentMainSnapshot[]=[];
      for(const snapshot of state.retiredSnapshots.splice(0)){
        if(displayed.has(snapshot)||ledger.isSnapshotRetained(snapshot)){kept.push(snapshot);continue;}
        for(const result of snapshot.results)result.resident?.release();
      }
      state.retiredSnapshots.push(...kept);
    };
    const settle=(accepted:boolean):void=>{
      try{if(accepted&&!state.disposed)commit();}
      finally{snapshotLease.release();releaseRetiredSnapshots();}
    };
    const fence=world.frameFence;
    if(fence)void fence.then(()=>settle(true),()=>settle(false));else settle(true);
    state.playback?.submitted();
    // Resource bookkeeping stays at submission, but a retired snapshot that is
    // still DISPLAYED or owned by a pending fence is released later.
    releaseRetiredSnapshots();
    state.fields.forEach(field=>field.submitted());state.transport.submitted();state.graph.submitted();state.sources.submitted();state.mask.submitted();toneField(world.gpu.device!).submitted();state.visibility.forEach(v=>v.submitted());state.targets.forEach(t=>{t.publication.submitted();t.appearance?.submitted();});
    for(const job of jobs)void publish(state,job).catch(error=>fail(state,error));
    if(state.controller.paused&&[...state.targets.values()].every(target=>!target.pending&&!!(target.resident||target.frame||target.error||state.controller.errors.length)))state.pauseLatched=true;
  });
}

async function publish(state:MainState,job:TargetJob):Promise<void>{
  let visibleSpans=0;
  const graphs:PosedGraph[]=[],fields:DrawingFieldView[]=[],geometry:CompositionGeometry[]=[];
  let allocationRetry=false;
  try{
    if(job.error)throw Error(job.error);
    if(state.playback){await publishRetained(state,job);return;}
    await Promise.all(job.records.map(async record=>{
      if(record.partAnchorDemand){const demand=await record.partAnchorDemand.collect();record.partAnchorDemand=undefined;if(demand.overflow){state.graph.ensurePartAnchorDemand(demand);allocationRetry=true;return;}}
      const staging=record.staging;
      const packed=record.relationReadback?await record.relationReadback.collect():undefined;
      if(!packed){if(!staging)throw Error('Text job has no source graph readback');await staging.mapAsync(GPUMapMode.READ);}
      const tone=record.tone?await record.tone.collect():undefined;
      const memory=packed?.memory??staging!.getMappedRange(),status=new Uint32Array(memory,0,4);
      if(status[1]===2&&(status[3]!>record.capacity||status[0]!>record.intervalCapacity)){
        state.relationExtent=Math.max(state.relationExtent,status[3]!);
        state.intervalExtents[record.visibility]=Math.max(state.intervalExtents[record.visibility]!,status[0]!);
        allocationRetry=true;return;
      }
      const field=record.field&&await record.field.collect();
      if(field)fields.push({field,camera:record.camera,rect:record.rect,owners:record.owners,pose:record.pose,light:record.light,fog:record.fog,tone} as TonedFieldView);
      const graph=decodePosedGraph(memory,packed?.capacity??record.capacity,packed?.intervalCapacity??record.intervalCapacity,
        record.camera,record.rect,record.width,record.height,record.owners,(domain,face)=>state.sources.component(domain,face),field?.visibleOwners?.());
      graphs.push(graph);
      const boundGeometry=new Map((record.geometry??[]).map(source=>[source.owner,source]));
      // Bind EVERY actual posed owner to the immutable definition named by
      // its decoded domain, including ordinary/default-mesh entities. The
      // early ECS mesh lookup is useful pose context, not the authority for
      // whether a visible box gets its compiled planar/hard construction.
      for(const [owner,reference]of graph.owners??[]){
        const definition=record.definitions?.get(reference.domain);if(!definition)continue;
        const posed=boundGeometry.get(owner);
        boundGeometry.set(owner,posed?{...posed,...definition,owner}:{...definition,owner,sourceOnly:true});
      }
      // An unmatched GPU instance gets NO invented identity CPU transform;
      // its real current pose remains in the graph used by articulation.
      for(const source of boundGeometry.values())geometry.push(source);
      // Construction consumes the complete posed graph. The withdrawn line
      // preview was used only for this count, not drawing; do not allocate a
      // second object/string copy of hundreds of thousands of source spans.
      for(const relation of graph.relations)for(const part of relation.intervals)if(part.cause==='visible')visibleSpans++;
    }));
    // This allocation attempt is NOT a text frame or a blank success. No
    // camera of a multi-view target may publish while another is incomplete.
    // All records have settled before their readbacks are retired in finally.
    if(allocationRetry){job.native?.destroy();job.mask.destroy();return;}
    if(state.disposed||job.state.width!==job.width||job.state.height!==job.height){job.native?.destroy();job.mask.destroy();return;}
    state.controller.graphs.set(job.serial,graphs);if(state.controller.graphs.size>4)state.controller.graphs.delete(state.controller.graphs.keys().next().value!);
    for(const graph of graphs)for(const owner of graph.owners?.values()??[]){
      if(!state.sources.isLive(owner.domain)||(owner.kind==='entity'&&!state.liveOwners.has(owner.lifetime))){job.native?.destroy();job.mask.destroy();return;}
    }
    const cpuPublishStart=performance.now(),labels=job.records.flatMap(record=>record.labels??[]);
    if(labels.some(label=>!state.semanticOwners.has(label.reference))){job.native?.destroy();job.mask.destroy();return;}
    // This branch belongs only to ?grammar=legacy. Compiled frames return
    // above; no CPU run compiler remains alongside their GPU playback.
    const frame=state.construction.construct(visibleSpans,job.width,job.height,job.serial,graphs,fields,labels,geometry);
    const presented=job.state.presentation.present(frame,constructionSupports(frame,graphs),job.serial,job.engineFrame,state.controller.mode);
    const published:MainTextFrame={...presented,comparison:job.comparison,construction:{sourceLines:frame.sourceLines,omitted:frame.omitted,metrics:{...frame.metrics,cpuFrameScope:'MAIN encode + native composition/publication CPU; GPU wait and other engine work excluded',cpuFrameMs:(job.cpuEncodeMs??0)+performance.now()-cpuPublishStart}}};
    job.state.publication.publish(published);job.state.frame=frame;job.state.serial=job.serial;job.state.error=undefined;
    job.state.native?.destroy();job.state.native=job.native;job.mask.destroy();
  }catch(error){
    job.native?.destroy();job.mask.destroy();
    // Disposing an embedded World destroys its GPU device and cancels pending
    // mapAsync calls. Retire that job; never report its cancellation as a live
    // replacement World's renderer failure. Real errors while alive still fail.
    if(!state.disposed)throw error;
  }
  finally{for(const record of job.records){record.toneLiterals?.reject(Error('Native camera job retired'));record.relationReadback?.dispose();record.partAnchorDemand?.dispose();record.tone?.dispose();record.field?.dispose();record.toneSource?.release?.();if(record.staging?.mapState==='mapped')record.staging.unmap();record.staging?.destroy();}job.state.pending=false;}
}

/** Compact native-tape publication only. The source graph, visibility,
 * run construction, rank and grammar playback have already run on the GPU. */
async function publishRetained(state:MainState,job:TargetJob):Promise<void>{
 const demands=await Promise.all(job.records.map(async record=>{if(!record.partAnchorDemand)return undefined;const demand=await record.partAnchorDemand.collect();record.partAnchorDemand=undefined;return demand;}));
 const anchorOverflow=demands.find(demand=>demand?.overflow);
 if(anchorOverflow){
  await discardPartAnchorAttempt(job.records.map(record=>({played:record.played,cancel(){record.toneLiterals?.reject(Error('Part-anchor demand retry discarded incomplete playback'));record.toneLiterals=undefined;}})));
  // Material transport ranges and engine depth are reusable after submission.
  // They cannot replay an old pose. Grow from its complete measured demand,
  // retire this incomplete cohort, and build the next coherent live pose.
  // The last complete page stays published until that cohort passes its gate.
  state.graph.ensurePartAnchorDemand(anchorOverflow);
  job.native?.destroy();job.mask.destroy();return;
 }
 const results=await Promise.all(job.records.map(async record=>{
  if(!record.played)throw Error('Retained construction has no GPU result');
  if(record.toneLiterals){
   // No compiled-frame fallback through a CPU field/tone reconstruction.
   record.toneLiterals.resolve(state.construction.playbackLiterals(record.labels??[],[],[]),record.tone?.gpu);
  }
  return record.played;
 }));
 let rowGrowth=false;
 // A sibling target can grow the shared reserve before this result settles.
 // Its old immutable layout still needs a next job even when allocation no
 // longer grows globally; the complete incumbent is published below.
 for(const result of results)if(result.sourceRowDemand!==undefined){state.graph.ensurePartAnchorRowDemand(result.sourceRowDemand);rowGrowth=true;}
 let standingGrowth=false;
 for(const result of results)if(result.standingOwnerDemand!==undefined){
  if(result.standingOwnerDemand<0)throw Error('Negative standing-owner demand');
  state.playback?.ensureStandingOwnerCapacity(result.standingOwnerDemand);standingGrowth=true;
 }
 for(const [index,result]of results.entries())if(result.sourceOverflow){
  const overflow=result.sourceOverflow,record=job.records[index]!;
  if(overflow.flags&~22)throw Error(`GPU source construction failure ${overflow.flags}`);
  if(overflow.triangles)state.graph.ensureContactCapacity(overflow.triangles);
  if(overflow.flags===2){state.relationExtent=Math.max(state.relationExtent,overflow.relations);state.intervalExtents[record.visibility]=Math.max(state.intervalExtents[record.visibility]!,overflow.intervals);}
  for(const result of results)result.resident?.release();
  if(rowGrowth||standingGrowth)job.state.still=undefined;
  job.native?.destroy();job.mask.destroy();return;
 }
 if(rowGrowth||standingGrowth){
  // Both demand hints preserve a complete incumbent. Publish it while making
  // this target's next job consume the measured reserve; another target's
  // allocation growth does not upgrade this immutable layout.
  job.state.still=undefined;
 }
 if(state.disposed||job.state.width!==job.width||job.state.height!==job.height){for(const result of results)result.resident?.release();job.native?.destroy();job.mask.destroy();return;}
 if(results.some(result=>!result.resident))throw Error('Compiled frame has no sealed resident publication');
 const snapshot:ResidentMainSnapshot={results,width:job.width,height:job.height,serial:job.serial,engineFrame:job.engineFrame,
  mode:state.controller.mode,cpuEncodeMs:job.cpuEncodeMs??0,comparison:job.comparison};
 if(job.state.resident)state.retiredSnapshots.push(job.state.resident);
 job.state.resident=snapshot;job.state.serial=job.serial;job.state.error=undefined;
 job.state.native?.destroy();job.state.native=job.native;job.mask.destroy();
}

/** Install in the existing App's plugins. The original world, simulation,
 * cameras, controls and native effects continue to run once. */
export function mainRenderer(options:{ascii?:boolean;rows?:number;geometryCapacity?:number;whitespace?:NativeWhitespacePolicy;compiledPlayback?:boolean}={}):Plugin{
  return {name:'Shallot MAIN',gpu:{features:['indirect-first-instance']},dependencies:[StandardRenderingPlugin,MeshRenderPlugin],
    systems:[{name:'MAIN/CellMatrix',group:'draw',first:true,before:[PrepareViewsSystem],update(world){world.resource(stateKey).current?.matrix.apply(world);}},

      {name:'MAIN/PresentationGate',group:'draw',first:true,after:[PrepareViewsSystem],before:[BeginFrameSystem],update:presentationGate},
      {name:'MAIN/Geometry',group:'draw',after:[MainPassSystem],before:[TonemappingSystem],update:build}],
    async warm(world,onProgress){
      const device=world.gpu.device!;
      const startupNotice=createFrameStateNotice(document.body,applyMobileUiAppearance),startupSince=Date.now();
      let startupDisposed=false,startupFailed=false,startupHandedOff=false;
      let startupTimer:ReturnType<typeof setInterval>|undefined,statusTimer:ReturnType<typeof setInterval>|undefined,display:HTMLDetailsElement|undefined;
      const startupAbort=new AbortController();
      let lastStartupStage:string|undefined;
      const startupStage=(stage?:string)=>{const name=stage?.trim();if(name)lastStartupStage=name;return lastStartupStage?`Preparing MAIN GPU program ${lastStartupStage}`:'Preparing MAIN GPU programs';};
      const stopStartupRefresh=()=>{if(startupTimer!==undefined){clearInterval(startupTimer);startupTimer=undefined;}};
      const stopStatusRefresh=()=>{if(statusTimer!==undefined){clearInterval(statusTimer);statusTimer=undefined;}};
      const disposeControls=()=>{stopStartupRefresh();stopStatusRefresh();display?.remove();if(!startupFailed)disposeFrameState(startupNotice);};
      const publishStartup=(stage?:string)=>{if(!startupDisposed&&!startupHandedOff)publishFrameStateNotice(startupNotice,{state:'PENDING',stage:startupStage(stage),startedAt:startupSince});};
      const publishStartupError=(error:unknown)=>{if(startupDisposed){disposeControls();return;}startupFailed=true;disposeControls();publishFrameStateNotice(startupNotice,{state:'ERROR',error});};
      // Keep the pre-init notice through a failed plugin hook so the engine's
      // world cleanup cannot turn a real initialization error back into a
      // blank/loading page. A normal disposal still removes this owner.
      world.onDispose(()=>{startupDisposed=true;startupAbort.abort(Error('MAIN world retired during GPU preparation'));if(startupFailed||!startupHandedOff)disposeControls();});
      publishStartup();
      startupTimer=setInterval(()=>publishStartup(),250);
      try{
      const matrix=cellMatrix(world);
      // An application's presentation default applies before GPU preparation
      // and the first page. Explicit URL Rows (including Auto) still wins.
      if(options.rows!==undefined&&!new URL(location.href).searchParams.has('rows'))matrix.rows=options.rows;

      // ?grammar=legacy keeps the CPU producer reachable until it is deleted.
      const compiled=options.compiledPlayback??new URL(location.href).searchParams.get('grammar')!=='legacy';
      // All scene/plugin initialize hooks have completed before warm. Admit
      // the relation capability from registered mesh producers now, so the
      // required shader closure is compiled before the first frame.
      const relationProfile=exampleGpuRelationProfile(world);
      const services=await prepareExampleGpuServices(device,options.geometryCapacity??1048576,compiled,(p,stage)=>{publishStartup(stage);onProgress?.(p/2);},startupAbort.signal,relationProfile);
      let state:MainState;
      try{state={matrix,displayedSnapshots:new Map(),retiredSnapshots:[],world:++nextWorld,transport:services.transport,graph:services.graph,sources:services.sources,visibility:services.visibility,relationExtent:32768,intervalExtents:[131072,131072],construction:new NativeConstruction(glyphs,options.whitespace??'preformatted'),primaryFields:new WeakMap(),lodLifetimes:new FormLodLifetimes(),fields:new Map(),blit:new TextureBlit(device),mask:services.mask,targets:new Map(),jobs:[],controller:{ascii:options.ascii??true,overlay:new URL(location.href).searchParams.get('overlay')==='1',mode:'text',pendingSince:startupSince,get rows(){return matrix.rows;},set rows(rows){if(rows!==undefined&&(!Number.isInteger(rows)||rows<1||rows>matrix.maxRows))throw Error(`Rows must be an integer from 1 to ${matrix.maxRows}`);matrix.rows=rows;},get pending(){const published=this.frames.size>0||this.residentFrames.size>0;return !state.disposed&&(!published||!!state.registration||this.paused&&!state.pauseLatched||[...state.targets.values()].some(target=>target.pending));},get pendingStage(){if(state.registration)return 'Compiling visible sources';if([...state.targets.values()].some(target=>target.pending))return 'Publishing text frame';return undefined;},underlay:new URL(location.href).searchParams.get('underdrawing')==='1',paused:false,pause(paused){if(paused===this.paused)return;state.pauseLatched=false;this.paused=paused;if(paused)world.pause();else world.resume();},frames:new Map(),residentFrames:new Map(),async exportText(){const entries=[...state.displayedSnapshots];if(!entries.length)return [...this.frames.values()];return Promise.all(entries.map(async([target,snapshot])=>{const frame={...await exportResidentFrame(snapshot),comparison:snapshot.comparison};if(state.displayedSnapshots.get(target)===snapshot)this.frames.set(target,frame);return frame;}));},requestPresentation(){return presentationLedger(this).demand();},graphs:new Map(),errors:[]},serial:0,disposed:false,liveOwners:new Set(),semanticOwners:new Set(),pauseLatched:false};}
      catch(error){services.dispose();throw error;}
      world.resource(stateKey).current=state;

      world.onDispose(()=>presentationLedger(state.controller).dispose());
      // Register lifetime cleanup BEFORE awaiting driver compilation: an
      // initialization failure or retired world cannot strand its resources.
      world.onDispose(()=>{state.disposed=true;state.playback?.dispose();state.construction.dispose();disposeControls?.();state.fields.forEach(field=>field.dispose());services.dispose();state.targets.forEach(t=>{t.publication.dispose();t.appearance?.dispose();t.native?.destroy();t.scaledPage?.destroy();});});
      // Both scene and Y8 defaults use retained GPU construction. The CPU
      // producer is reachable only through the explicit legacy override.
      if(compiled){
        const playback=await GpuArticulation.create(device,glyphs,options.whitespace??'preformatted',8192,(progress,stage)=>{
          if(state.disposed)throw Error('MAIN world retired during pipeline preparation');
          publishStartup(stage);
          onProgress?.(1/2+progress/2);
        });
        if(state.disposed){playback.dispose();throw Error('MAIN world retired during pipeline preparation');}
        state.playback=playback;
      }else onProgress?.(1);
      world.resource(PhaseGeometryCapture).enabled=true;world.resource(PhaseGeometryCapture).includeStandard=true;
      const capture=world.resource(PhaseGeometryCapture);
      const primarySelected=()=>state.controller.ascii&&!state.controller.underlay&&!state.controller.overlay;
      capture.isPrimary=primarySelected;
      const primary=(world:World,view:View,eid:number)=>{
        state.primaryFields.delete(view);
        if(!displayView(view)||!primarySelected())return;
        return ()=>primaryAppearance(world,view,eid,state);
      };
      capture.primary=primary;
      world.onDispose(()=>{if(capture.primary===primary)capture.primary=undefined;if(capture.isPrimary===primarySelected)capture.isPrimary=undefined;});
      world.resource(DepthPrepassRequests).push((_world,_eid,view)=>displayView(view));world.resource(FinalPresentationPasses).push(present);
      (globalThis as typeof globalThis&{__shallotMain?:MainRendererController}).__shallotMain=state.controller;
      const controls=document.createElement('div');controls.setAttribute('aria-label','Display controls');
      const touchDisplay=supportsTouchControls();display=document.createElement('details');const summary=document.createElement('summary');
      display.setAttribute('aria-label','Renderer comparison');
      display.open=!touchDisplay;summary.textContent='Display';
      display.style.cssText='position:fixed;left:max(8px,env(safe-area-inset-left));top:max(72px,calc(env(safe-area-inset-top) + 56px));max-width:calc(100vw - 16px);box-sizing:border-box;z-index:10000;background:#fff;color:#000;border:2px solid #000;border-radius:0';
      applyMobileUiAppearance(display);
      summary.style.cssText='min-height:44px;padding:0 10px;display:flex;align-items:center;gap:10px;cursor:pointer;box-sizing:border-box';
      controls.style.cssText='padding:8px 10px;display:flex;flex-wrap:wrap;gap:12px;align-items:center;max-height:calc(100dvh - 100px);overflow:auto;box-sizing:border-box';
      display.append(summary,controls);
      const label=document.createElement('label'),toggle=document.createElement('input');toggle.type='checkbox';toggle.checked=state.controller.ascii;
      toggle.addEventListener('change',()=>{state.controller.ascii=toggle.checked;});label.append(toggle,' ASCII');
      const diagnosticLabel=document.createElement('label'),diagnostic=document.createElement('input');diagnostic.type='checkbox';diagnostic.checked=state.controller.underlay;diagnostic.addEventListener('change',()=>{state.controller.underlay=diagnostic.checked;});diagnosticLabel.append(diagnostic,' Mesh');
      const pauseLabel=document.createElement('label'),pause=document.createElement('input');pause.type='checkbox';pause.addEventListener('change',()=>state.controller.pause(pause.checked));pauseLabel.append(pause,' Pause comparison');
      const rowsLabel=matrix.control();
      state.controller.underlay=new URL(location.href).searchParams.get('underdrawing')==='1';diagnostic.checked=state.controller.underlay;
      const status=document.createElement('span');status.setAttribute('role','status');state.status=status;
      const notice=startupNotice;state.notice=notice;
      publishFrameState(state.controller,status,notice);
      statusTimer=setInterval(()=>{toggle.checked=state.controller.ascii;diagnostic.checked=state.controller.underlay;pause.checked=state.controller.paused;mode.value=state.controller.mode;publishFrameState(state.controller,status,notice);},250);
      const mode=document.createElement('select');mode.setAttribute('aria-label','Text presentation');
      for(const [value,label]of [['text','Text keyframes + moving runs'],['floating','Floating runs']]){const option=document.createElement('option');option.value=value!;option.textContent=label!;mode.append(option);}
      mode.addEventListener('change',()=>{state.controller.mode=mode.value as PresentationMode;});
      const exportButton=document.createElement('button');exportButton.textContent='Export text';
      exportButton.addEventListener('click',async()=>{
        exportButton.disabled=true;
        try{for(const frame of await state.controller.exportText()){const target=frame.comparison.target,exact=frame.presentation?.kind==='exact_text'&&frame.page_shift_px.every(value=>value===0),blob=new Blob([exact?plainText(frame,frame.nativeRows)+'\n':JSON.stringify(frame,null,2)+'\n'],{type:exact?'text/plain;charset=utf-8':'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=`shallot-${target}-${frame.frame}.${exact?'txt':'json'}`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}}
        catch(error){fail(state,error);}finally{exportButton.disabled=false;}
      });controls.append(label,diagnosticLabel,pauseLabel,rowsLabel,mode,exportButton);summary.append(status);
      status.style.cssText='font-size:11px;max-width:min(65vw,600px);white-space:normal';
      if(touchDisplay){
        for(const input of controls.querySelectorAll<HTMLElement>('input,select,button'))Object.assign(input.style,{minHeight:'44px',fontSize:'16px'});
        for(const label of controls.querySelectorAll<HTMLElement>('label'))Object.assign(label.style,{display:'inline-flex',alignItems:'center',minHeight:'44px'});
      }
      document.body.append(display);
      stopStartupRefresh();
      startupHandedOff=true;
      }catch(error){publishStartupError(error);throw error;}
    },
  };
}
