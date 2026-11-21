import {prepareToneFieldPrograms} from './tone-field.ts';
import {prepareFormDepthPrograms} from './form-depths.ts';
import {prepareSurfaceIntersectionPrograms} from './surface-intersections.ts';
import {MaterialGeometryTransport} from './geometry.ts';
import {MeshSourceCache} from './source-cache.ts';
import {PhysicalSourceTransport} from './physical-source.ts';
import {SurfaceRelations} from './surface-relations.ts';
import {RelationVisibility} from './relation-visibility.ts';
import {ConstructionMask} from './construction-mask.ts';
import type {SourceComplexRetention} from './source-program.ts';
import {prepareSourceFarForms} from './gpu-source-far-forms.ts';
import type {ComputePipelineProgress} from './gpu-pipeline-preparation.ts';

interface Disposable {dispose():void}
type Factories<T>={[K in keyof T]:(progress:ComputePipelineProgress)=>Promise<T[K]>};
/** One sequencing and lifetime owner for the shared adapters' startup. No
 * frame can enqueue compute work while a synchronous constructor floods the
 * Metal compiler with other MAIN programs. Constructors still support the
 * explicit synchronous inspection adapter; normal worlds use these factories. */
async function prepareParts<T extends {[K in keyof T]:Disposable}>(factories:Factories<T>,progress?:ComputePipelineProgress,signal?:AbortSignal):Promise<{parts:T;dispose:()=>void}>{
 const parts={} as T,owned:Disposable[]=[];let disposed=false,abortCleanup:unknown;
 const check=()=>{if(disposed||signal?.aborted)throw signal?.reason??Error('MAIN world retired during GPU preparation');};
 // An abort event must not throw into the window's global error handler.
 // Report cleanup failure through the preparation promise or explicit owner.
 const abort=()=>{try{dispose();}catch(error){abortCleanup=error;}};
 const dispose=()=>{
  if(disposed){if(abortCleanup!==undefined){const error=abortCleanup;abortCleanup=undefined;throw error;}return;}
  disposed=true;signal?.removeEventListener('abort',abort);
  const errors:unknown[]=[];for(const item of owned.splice(0).reverse())try{item.dispose();}catch(error){errors.push(error);}
  if(errors.length)throw new AggregateError(errors,'MAIN GPU startup resource cleanup failed');
 };
 signal?.addEventListener('abort',abort,{once:true});
 try{
  check();const keys=Object.keys(factories) as (keyof T)[];progress?.(0);
  for(const [index,key] of keys.entries()){
   check();const item=await factories[key]((fraction,stage)=>{check();progress?.((index+fraction)/keys.length,stage);});
   if(disposed||signal?.aborted){item.dispose();check();}
   parts[key]=item;owned.push(item);
  }
  check();progress?.(1);return {parts,dispose};
 }catch(error){try{dispose();}catch(cleanup){throw new AggregateError([error,cleanup],'MAIN GPU preparation and cleanup failed',{cause:error});}throw error;}
}


/** Warm immutable device programs only. Existing field/contact instances
 * keep their own buffer/texture lifetimes; startup owns no duplicate image. */
type FieldFactory=(device:GPUDevice,progress?:ComputePipelineProgress)=>Promise<void>;
async function prepareFieldPrograms(device:GPUDevice,progress?:ComputePipelineProgress,signal?:AbortSignal):Promise<void>{
 const check=()=>{if(signal?.aborted)throw signal.reason??Error('MAIN world retired during field-program preparation');};
 const factories:readonly [string,FieldFactory][]= [
  ['surface-intersections',prepareSurfaceIntersectionPrograms],
  ['tone-field',prepareToneFieldPrograms],
  ['form-depths',prepareFormDepthPrograms],
 ];
 check();progress?.(0);for(const [index,[name,prepare]]of factories.entries()){
  check();const base=index/factories.length;
  // Name the factory before its await even if an inner cache exposes no
  // callback. Inner keyed plans replace this with their actual stage.
  progress?.(base,name);
  await prepare(device,(fraction,stage)=>{check();progress?.((index+fraction)/factories.length,stage??name);});
  check();progress?.((index+1)/factories.length);
 }
}

interface ExampleParts {
 transport:MaterialGeometryTransport;graph:SurfaceRelations;sources:MeshSourceCache<SourceComplexRetention>;
 visibilitySingle:RelationVisibility;visibilityMultisampled:RelationVisibility;mask:ConstructionMask;
}
export interface ExampleGpuServices {
 readonly transport:MaterialGeometryTransport;readonly graph:SurfaceRelations;readonly sources:MeshSourceCache<SourceComplexRetention>;
 readonly visibility:[RelationVisibility,RelationVisibility];readonly mask:ConstructionMask;dispose():void;
}
export type ExampleGpuRelationProfile='ordinary'|'producer-capable';
/** Map warm-time source capability admission to one closed GPU preparation
 * graph. Ordinary worlds omit producer-only boundary entries; producer-
 * capable worlds retain both ordinary and posed-quad closures. */
export function surfaceRelationPreparation(profile:ExampleGpuRelationProfile):'generic-direct'|'generic-direct-ordinary'{
 return profile==='producer-capable'?'generic-direct':'generic-direct-ordinary';
}
export async function prepareExampleGpuServices(device:GPUDevice,geometryCapacity:number,compiled:boolean,progress?:ComputePipelineProgress,signal?:AbortSignal,relationProfile:ExampleGpuRelationProfile='ordinary'):Promise<ExampleGpuServices>{
 await prepareFieldPrograms(device,(p,stage)=>progress?.(p/4,stage),signal);
 if(compiled){
  if(signal?.aborted)throw signal.reason??Error('MAIN world retired during source-far preparation');
  progress?.(1/4,'source-far-forms');
  await prepareSourceFarForms(device,(p,stage)=>{if(signal?.aborted)throw signal.reason??Error('MAIN world retired during source-far preparation');progress?.(1/4+p/4,stage??'source-far-forms');});
 }
 const partsBase=compiled?1/2:1/4,partsWeight=compiled?1/2:3/4;
 const prepared=await prepareParts<ExampleParts>({
  transport:p=>MaterialGeometryTransport.create(device,geometryCapacity,p),
  // MAIN's shared example caller appends direct sources without the
  // host-sized extent/frontier contract. Keep the public generic default for
  // other consumers, but explicitly avoid preparing unused batch kernels here.
  graph:p=>SurfaceRelations.create(device,32768,compiled,p,surfaceRelationPreparation(relationProfile)),
  sources:p=>MeshSourceCache.create(device,'gpu-csr',compiled?'summary':'full',p),
  visibilitySingle:p=>RelationVisibility.create(device,32768,131072,false,p),
  visibilityMultisampled:p=>RelationVisibility.create(device,32768,131072,true,p),

  mask:p=>ConstructionMask.create(device,p),
 },(p,stage)=>progress?.(partsBase+partsWeight*p,stage),signal);
 const p=prepared.parts;return {transport:p.transport,graph:p.graph,sources:p.sources,visibility:[p.visibilitySingle,p.visibilityMultisampled],mask:p.mask,dispose:prepared.dispose};
}
interface PhysicalParts {transport:PhysicalSourceTransport;relations:SurfaceRelations;visibility:RelationVisibility;mask:ConstructionMask}
export interface PhysicalGpuServices extends PhysicalParts {dispose():void}
export async function preparePhysicalGpuServices(device:GPUDevice,compiled:boolean,progress?:ComputePipelineProgress,signal?:AbortSignal):Promise<PhysicalGpuServices>{
 await prepareFieldPrograms(device,(p,stage)=>progress?.(p/4,stage),signal);
 const prepared=await prepareParts<PhysicalParts>({
  transport:p=>PhysicalSourceTransport.create(device,'gpu-csr',compiled?'summary':'full',p),
  // The compiled physical frame always supplies the host-sized batch contract
  // (extent, GPU visibility frontier and posed revision) and consumes only the
  // queued batch entries. Legacy physical inspection retains the generic
  // direct stages because it does not satisfy that contract.
  relations:p=>SurfaceRelations.create(device,32768,compiled,p,compiled?'physical-batched':'generic'),
  visibility:p=>RelationVisibility.create(device,32768,131072,false,p),

  mask:p=>ConstructionMask.create(device,p),
 },(p,stage)=>progress?.(1/4+3*p/4,stage),signal);
 return {...prepared.parts,dispose:prepared.dispose};
}
