import type {SourceConstructionClass,ResidentConstructionRuns} from './gpu-construction-runs.ts';
import {markCoherentParentRejected} from './coherent-parent-union.ts';
import {COHERENT_PARENT_STATUS_REJECT,COHERENT_PARENT_STATUS_OVERFLOW} from './coherent-parent-gpu.ts';
export {COHERENT_PARENT_STATUS_REJECT,COHERENT_PARENT_STATUS_OVERFLOW};
export const COHERENT_PARENT_FAILURE_MASK=COHERENT_PARENT_STATUS_REJECT|COHERENT_PARENT_STATUS_OVERFLOW;

/** Called by the actual mapped source-run header before any native tape can
 * publish. Allocation exhaustion retries the same product with larger storage;
 * a malformed product invalidates its generation so host adapters reselect its
 * complete children. Neither condition exposes a partially emitted group. */
export function retryCoherentSourceRuns(flags:number,points:number,runs:number,resident:Pick<ResidentConstructionRuns,'statusOffset'>,classes:readonly SourceConstructionClass[],grow:(flags:number,points:number,runs:number,hashCapacity:number)=>void):void{
 if(flags&~(7|COHERENT_PARENT_FAILURE_MASK))throw Error(`Unknown construction allocation flags ${flags}`);
 const allocation=flags&7;
 if(flags&COHERENT_PARENT_STATUS_OVERFLOW){
  if(!allocation)throw Error('Coherent parent overflow has no allocation demand');
 }else if(flags&COHERENT_PARENT_STATUS_REJECT){
  let marked=false;
  for(const source of classes)if('farForms' in source)for(const aggregate of source.farForms?.aggregates??[]){
   marked=markCoherentParentRejected(aggregate.construction)||marked;
  }
  if(!marked)throw Error('Coherent parent rejection has no retained source product');
 }
 if(allocation)grow(allocation,points,runs,resident.statusOffset/32);
}
