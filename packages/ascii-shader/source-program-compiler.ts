import {assertSourcePacketVersion} from './source-packet.ts';
import type {CompiledSourceProgram,FullCompiledSourceProgram,SummaryCompiledSourceProgram,SourceProgramInput} from './source-program.ts';
import {decodeSourceComplex,type SourceComplexColumns} from './source-transfer.ts';
import {sourceWholeCompoundReady,type SourceWholeConstructionTemplate, type SourceWholePartRole} from './source-lod.ts';
import {validateSourceSmallParent} from './source-semantic-residual.ts';
export {validateSourceSmallParent} from './source-semantic-residual.ts';

export interface SourceCompilation {
 readonly result:Promise<CompiledSourceProgram|undefined>;
 cancel():void;
 /** Compilation scheduling only, never native source selection. */
 prioritize(priority:number):void;
}
interface Request {
 readonly resolve:(program:CompiledSourceProgram|undefined)=>void;
 readonly reject:(error:Error)=>void;
 readonly complexMode:'full'|'summary';readonly physical:boolean;
}


export function validateWholeConstruction(value:SourceWholeConstructionTemplate|undefined,revision:number):void{
 if(value===undefined)return;
 if(value.revision!==revision||!value.parts.length)throw Error('Invalid source whole construction revision');
 const roles=new Set(value.parts.map(part=>part.role));
 const required:readonly SourceWholePartRole[]=['cap','turn','flank','return'];
 const completeRoles=required.every(role=>roles.has(role));
 const check=(items:readonly {feature:number;ports:readonly number[];sourceContributors:readonly number[];geometry?:readonly (readonly number[])[];closed?:boolean;entryRef?:number;context?:number}[])=>{for(const item of items){if(!Number.isSafeInteger(item.feature)||item.feature<=0||item.ports.length<2||item.sourceContributors.length===0||item.ports.some(port=>!Number.isSafeInteger(port)||port<0)||item.sourceContributors.some(id=>!Number.isSafeInteger(id)||id<0))throw Error('Invalid source whole construction passage');if(item.geometry!==undefined&&(item.geometry.length<2||item.geometry.some(point=>point.length!==3||point.some(value=>!Number.isFinite(value)))))throw Error('Invalid source whole construction passage geometry');if(item.closed!==undefined&&typeof item.closed!=='boolean')throw Error('Invalid source whole construction passage closure');if(item.entryRef!==undefined&&(!Number.isSafeInteger(item.entryRef)||item.entryRef<1)||item.context!==undefined&&(!Number.isSafeInteger(item.context)||item.context===0))throw Error('Invalid source whole construction passage context');}};
 for(const part of value.parts){if(!Number.isSafeInteger(part.feature)||part.feature<=0||!Number.isSafeInteger(part.entryRef)||part.entryRef<1||!Number.isSafeInteger(part.context)||part.context===0||part.ports.length!==2||part.ports.some(port=>!Number.isSafeInteger(port)||port<0)||part.sourceContributors.length===0||part.sourceContributors.some(id=>!Number.isSafeInteger(id)||id<0)||!part.geometry?.length||part.geometry.some(point=>point.length!==3||point.some(value=>!Number.isFinite(value))))throw Error('Invalid source whole construction part');if(part.paper)check(part.paper);}
 check(value.openings);check(value.contacts);check(value.protectedVoids);if(value.paper)check(value.paper);
 // A partial role set is legal only for the one bounded planar compound whose
 // source graph carries the complete outer/opening/contact identity ledger.
 // Do not let four-role presence or a marker turn relation metadata into a
 // parent replacement: every compound passage must still be drawable by its
 // own native entry and source geometry.
 const nativePassage=(item:{feature:number;ports:readonly number[];sourceContributors:readonly number[];geometry?:readonly (readonly number[])[];closed?:boolean;entryRef?:number;context?:number;role?:SourceWholePartRole}):boolean=>
  !!item.geometry&&item.geometry.length>=2&&typeof item.closed==='boolean'&&Number.isSafeInteger(item.entryRef)&&item.entryRef!>=1&&Number.isSafeInteger(item.context)&&item.context!==0&&item.role!==undefined;
 const compoundComplete=!!value.compound&&sourceWholeCompoundReady(value)&&
  value.parts.every(part=>nativePassage(part))&&
  [...value.openings,...value.contacts,...value.protectedVoids,...(value.paper??[])].every(item=>nativePassage(item));
 if(value.compound&&!compoundComplete)throw Error('Invalid source whole construction compound');
 if(!completeRoles&&!compoundComplete)throw Error('Incomplete source whole construction roles');
}
type TransferredSourceProgram=(Omit<FullCompiledSourceProgram,'complex'>&{readonly complex:SourceComplexColumns})|SummaryCompiledSourceProgram;
/** Owns immutable revision snapshots and compilation lifetimes for both
 * engine adapters. No source compilation is allowed on their draw paths. */
export class SourceProgramCompiler {
 private worker?:Worker;private serial=0;private disposed=false;private error?:Error;
 private readonly requests=new Map<number,Request>();
 private start():Worker{
  if(this.worker)return this.worker;
  const worker=this.worker=new Worker(new URL('./mesh-part-worker.ts',import.meta.url),{type:'module'});
  worker.onmessage=({data}:{data:{id:number;program?:TransferredSourceProgram;error?:string}})=>{
   const request=this.requests.get(data.id);if(!request||this.disposed)return;
   this.requests.delete(data.id);
   if(data.program){try{
    const packet=data.program.packet;
    if(packet?.kind==='legacy-v2')assertSourcePacketVersion(packet.words);
    else if(packet?.kind==='gpu-csr-v1'){
     const p=packet.plan;if(p.kind!=='indexed-csr-v1'||p.prefix[0]!==0x4d41494e||p.prefix[1]!==3||p.prefix.length!==p.csrOffset||p.words!==p.csrOffset+p.csr.words)throw Error('Invalid compact source program');
    }else throw Error('Unsupported source program packet');
    const program=data.program;
    validateWholeConstruction(program.wholeConstruction,program.geometry.revision);
    // The worker result is a source-revision snapshot. Validate the compact
    // smallest-tier metadata before publishing it; do not silently discard it
    // from either summary or full retention.
    validateSourceSmallParent(program.smallParent);
    if(program.complexMode!==request.complexMode)throw Error('Source compiler returned the wrong topology retention');
    if(!!program.physical!==request.physical)throw Error('Source compiler returned the wrong physical vertex contract');
    if(program.physical){
     const p=program.physical,vertices=program.geometry.positions.length/3;
     if(p.geometry!==program.geometry||!(p.positions instanceof Float32Array)||p.positions.length!==vertices*4||
       !(p.geometry.normals instanceof Float32Array)||p.geometry.normals.length!==vertices*4||
       !(p.bounds instanceof Float32Array)||p.bounds.length!==8)throw Error('Invalid physical source vertex inputs');
    }
    if(program.complexMode==='summary'){
     const summary=program.summary;
     if(packet.kind!=='gpu-csr-v1'||Object.hasOwn(program,'complex')||summary?.schema!=='shallot.source-residency-summary.v1'||!Number.isSafeInteger(summary.edgeCount)||summary.edgeCount<0||summary.edgeCount!==packet.plan.edges)throw Error('Invalid compiled source residency summary');
     request.resolve(program);
    }else if(program.complexMode==='full')request.resolve({...program,complex:decodeSourceComplex(program.complex)});
    else throw Error('Unsupported source complex retention contract');
   }catch(error){request.reject(error instanceof Error?error:Error(String(error)));}}else request.reject(Error(data.error??'Source compiler returned no program'));
  };
  worker.onerror=event=>{
   const error=this.error=Error(event.message||'Source compiler worker failed');
   for(const request of this.requests.values())request.reject(error);
   this.requests.clear();worker.terminate();
  };
  return worker;
 }
 compile(input:SourceProgramInput):SourceCompilation{
  if(this.disposed)throw Error('Source compiler is disposed');if(this.error)throw this.error;
  const worker=this.start(),id=++this.serial;
  let resolve!:Request['resolve'],reject!:Request['reject'];
  const result=new Promise<CompiledSourceProgram|undefined>((yes,no)=>{resolve=yes;reject=no;});
  this.requests.set(id,{resolve,reject,complexMode:input.complexMode??'full',physical:!!input.physical});
  try{
   // A source's arrays belong to its simulation. Transfer only this revision
   // snapshot, including lineage, never the caller's buffers.
   const source=structuredClone(input),buffers=new Set<ArrayBuffer>();
   const visit=(value:unknown):void=>{
    if(ArrayBuffer.isView(value)){if(value.buffer instanceof ArrayBuffer)buffers.add(value.buffer);}
    else if(value&&typeof value==='object')for(const member of Object.values(value))visit(member);
   };
   visit(source);worker.postMessage({id,priority:1,source},[...buffers]);
  }catch(error){this.requests.delete(id);reject(error instanceof Error?error:Error(String(error)));}
  let priority=1;
  return {result,prioritize:(value:number)=>{
   if(!Number.isFinite(value)||value<=0)throw Error('Invalid source compilation priority');
   if(value!==priority&&this.requests.has(id)){priority=value;worker.postMessage({id,priority:value});}
  },cancel:()=>{if(this.requests.delete(id)){worker.postMessage({id,cancel:true});resolve(undefined);}}};
 }
 dispose():void{
  this.disposed=true;for(const request of this.requests.values())request.resolve(undefined);
  this.requests.clear();this.worker?.terminate();this.worker=undefined;
 }
}
