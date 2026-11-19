import {prepareToneFieldPrograms} from './tone-field.ts';
import {prepareFormDepthPrograms} from './form-depths.ts';
import {prepareSurfaceIntersectionPrograms} from './surface-intersections.ts';
import {MaterialGeometryTransport} from './geometry.ts';
import {MeshSourceCache} from './source-cache.ts';
import {PhysicalSourceTransport} from './physical-source.ts';
import {SurfaceRelations} from './surface-relations.ts';
import {RelationVisibility} from './relation-visibility.ts';
import {Underdrawing} from './underdrawing.ts';
import type {SourceComplexRetention} from './source-program.ts';
import {prepareSourceFarForms} from './gpu-source-far-forms.ts';

interface Disposable {dispose():void}
type Factories<T>={[K in keyof T]:(progress:(fraction:number)=>void)=>Promise<T[K]>};
/** One sequencing and lifetime owner for the shared adapters' startup. No
 * frame can enqueue compute work while a synchronous constructor floods the
 * Metal compiler with other MAIN programs. Constructors still support the
 * explicit synchronous inspection adapter; normal worlds use these factories. */
async function prepareParts<T extends {[K in keyof T]:Disposable}>(factories:Factories<T>,progress?:(fraction:number)=>void,signal?:AbortSignal):Promise<{parts:T;dispose:()=>void}>{
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
   check();const item=await factories[key](fraction=>{check();progress?.((index+fraction)/keys.length);});
   if(disposed||signal?.aborted){item.dispose();check();}
   parts[key]=item;owned.push(item);
  }
  check();progress?.(1);return {parts,dispose};
 }catch(error){try{dispose();}catch(cleanup){throw new AggregateError([error,cleanup],'MAIN GPU preparation and cleanup failed',{cause:error});}throw error;}
}


/** Warm immutable device programs only. Existing field/contact instances
 * keep their own buffer/texture lifetimes; startup owns no duplicate image. */
async function prepareFieldPrograms(device:GPUDevice,progress?:(fraction:number)=>void,signal?:AbortSignal):Promise<void>{
 const check=()=>{if(signal?.aborted)throw signal.reason??Error('MAIN world retired during field-program preparation');};
 const factories=[prepareSurfaceIntersectionPrograms,prepareToneFieldPrograms,prepareFormDepthPrograms];
 check();progress?.(0);for(const [index,prepare]of factories.entries()){check();await prepare(device);check();progress?.((index+1)/factories.length);}
}

interface ExampleParts {
 transport:MaterialGeometryTransport;graph:SurfaceRelations;sources:MeshSourceCache<SourceComplexRetention>;
 visibilitySingle:RelationVisibility;visibilityMultisampled:RelationVisibility;underlay:Underdrawing;
}
export interface ExampleGpuServices {
 readonly transport:MaterialGeometryTransport;readonly graph:SurfaceRelations;readonly sources:MeshSourceCache<SourceComplexRetention>;
 readonly visibility:[RelationVisibility,RelationVisibility];readonly underlay:Underdrawing;dispose():void;
}
export async function prepareExampleGpuServices(device:GPUDevice,geometryCapacity:number,compiled:boolean,progress?:(fraction:number)=>void,signal?:AbortSignal):Promise<ExampleGpuServices>{
 await prepareFieldPrograms(device,p=>progress?.(p/4),signal);
 if(compiled)await prepareSourceFarForms(device);
 const prepared=await prepareParts<ExampleParts>({
  transport:p=>MaterialGeometryTransport.create(device,geometryCapacity,p),
  graph:p=>SurfaceRelations.create(device,32768,compiled,p),
  sources:p=>MeshSourceCache.create(device,'gpu-csr',compiled?'summary':'full',p),
  visibilitySingle:p=>RelationVisibility.create(device,32768,131072,false,p),
  visibilityMultisampled:p=>RelationVisibility.create(device,32768,131072,true,p),
  underlay:p=>Underdrawing.create(device,p),
 },p=>progress?.(1/4+3*p/4),signal);
 const p=prepared.parts;return {transport:p.transport,graph:p.graph,sources:p.sources,visibility:[p.visibilitySingle,p.visibilityMultisampled],underlay:p.underlay,dispose:prepared.dispose};
}
interface PhysicalParts {transport:PhysicalSourceTransport;relations:SurfaceRelations;visibility:RelationVisibility;underdrawing:Underdrawing}
export interface PhysicalGpuServices extends PhysicalParts {dispose():void}
export async function preparePhysicalGpuServices(device:GPUDevice,compiled:boolean,progress?:(fraction:number)=>void,signal?:AbortSignal):Promise<PhysicalGpuServices>{
 await prepareFieldPrograms(device,p=>progress?.(p/4),signal);
 const prepared=await prepareParts<PhysicalParts>({
  transport:p=>PhysicalSourceTransport.create(device,'gpu-csr',compiled?'summary':'full',p),
  relations:p=>SurfaceRelations.create(device,32768,compiled,p),
  visibility:p=>RelationVisibility.create(device,32768,131072,false,p),
  underdrawing:p=>Underdrawing.create(device,p),
 },p=>progress?.(1/4+3*p/4),signal);
 return {...prepared.parts,dispose:prepared.dispose};
}
