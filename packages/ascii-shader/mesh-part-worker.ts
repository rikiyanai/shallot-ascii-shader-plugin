import {meshPartWork,type MeshPartGraph,type PartMesh} from './mesh-part-graph.ts';
import {sourceProgramWork,type CompiledSourceProgram,type SourceProgramInput,type SummaryCompiledSourceProgram} from './source-program.ts';
import {encodeSourceComplex} from './source-transfer.ts';
import {IndexedSourceProgramStore,PersistentSourcePrograms,SOURCE_PROGRAM_ENTRY_BYTES,SOURCE_PROGRAM_STORE_BYTES,sourceProgramStoreNamespace,sourceProgramStoredBytes} from './source-program-store.ts';

export type SourceWorkerRequest={id:number;priority:number;mesh?:PartMesh;source?:SourceProgramInput;cancel?:boolean};
type Result={graph:MeshPartGraph}|{program:CompiledSourceProgram};
type SourcePersistence={
 find(input:SourceProgramInput):Promise<{key:string;program?:SummaryCompiledSourceProgram}|undefined>;
 remember(key:string|undefined,program:SummaryCompiledSourceProgram):void;
};
type SourceWorkerScope={postMessage:(value:unknown,transfer?:Transferable[])=>void;onmessage?:(event:MessageEvent<SourceWorkerRequest>)=>void};
type SourceJob={priority:number;service:number;work?:Generator<void,Result>;cacheKey?:string;cachePending?:boolean;cacheProgram?:SummaryCompiledSourceProgram;cacheProgramBytes?:number;completed?:boolean;cancelled?:boolean};
const MAX_RETAINED_CACHE_ENTRIES=4096; // source-program-store.ts MAX_ENTRIES

/** Keep the visible cohort responsive without changing which sources or stages
 * are compiled. Background work retains the original fairness quantum. */
export const REQUIRED_WORKER_QUANTUM_MS=12;
export const BACKGROUND_WORKER_QUANTUM_MS=4;

export interface SourceWorkerOptions {
 /** Production uses a zero-delay timer; tests can drive the actual handler one slice at a time. */
 readonly schedule?: (task:()=>void)=>void;
 /** Production uses the worker clock; tests can force one generator step per slice. */
 readonly now?: ()=>number;
}
export type SourceWorkerHandler=((event:MessageEvent<SourceWorkerRequest>)=>void)&{cacheRetention:()=>{bytes:number;entries:number}};

function* parts(mesh:PartMesh):Generator<void,Result>{return {graph:yield* meshPartWork(mesh)};}
function* source(mesh:SourceProgramInput):Generator<void,Result>{return {program:yield* sourceProgramWork(mesh)};}
function transfers(value:unknown,buffers=new Set<ArrayBuffer>()):ArrayBuffer[]{
 if(ArrayBuffer.isView(value)){if(value.buffer instanceof ArrayBuffer)buffers.add(value.buffer);}
 else if(value&&typeof value==='object')for(const member of Object.values(value))transfers(member,buffers);
 return [...buffers];
}

/**
 * The production worker and the startup regression tests share this handler.
 * Persistence is deliberately injected only at this boundary: a cache lookup
 * may accelerate a source, but it is never allowed to gate the complete
 * source-program producer or to publish detached buffers.
 */
export function createSourceWorkerHandler(scope:SourceWorkerScope,stored:SourcePersistence,options:SourceWorkerOptions={}):SourceWorkerHandler{
 const jobs=new Map<number,SourceJob>();
 const now=options.now??(()=>performance.now());
 const enqueue=options.schedule??(task=>{setTimeout(task,0);});
 let scheduled=false;
 let retainedCacheBytes=0,retainedCacheEntries=0;
 const releaseCache=(job:SourceJob):void=>{if(job.cacheProgramBytes!==undefined){retainedCacheBytes-=job.cacheProgramBytes;retainedCacheEntries--;job.cacheProgramBytes=undefined;}job.cacheProgram=undefined;};
 const retainCache=(job:SourceJob,program:SummaryCompiledSourceProgram):void=>{
  const bytes=sourceProgramStoredBytes(program);
  if(bytes>SOURCE_PROGRAM_ENTRY_BYTES||retainedCacheEntries>=MAX_RETAINED_CACHE_ENTRIES||retainedCacheBytes+bytes>SOURCE_PROGRAM_STORE_BYTES)return;
  try{job.cacheProgram=structuredClone(program);job.cacheProgramBytes=bytes;retainedCacheBytes+=bytes;retainedCacheEntries++;}catch{/* Cache is optional; never fail publication. */}
 };
 const schedule=():void=>{if(!scheduled&&[...jobs.values()].some(job=>job.work)){scheduled=true;enqueue(advance);}};
 const cacheResult=(pending:SourceJob,hit:{key:string;program?:SummaryCompiledSourceProgram}|undefined):void=>{
  if(pending.cancelled)return;
  pending.cachePending=false;
  pending.cacheKey=hit?.program?undefined:hit?.key;
  // A hit is allowed to replace a source only before its first worker slice.
  // Once service began, the complete generator remains authoritative.
  if(hit?.program&&pending.service===0&&!pending.completed){
   pending.work=(function*():Generator<void,Result>{return {program:hit.program!};})();
  }
  // A cold compile may have completed while the content-addressed lookup was
  // still hashing. The posted program's buffers are transferred below, so use
  // the retained pre-transfer clone to preserve the eventual cache miss.
  if(pending.cacheProgram){
   if(pending.cacheKey)stored.remember(pending.cacheKey,pending.cacheProgram);
   releaseCache(pending);
  }
  if(!pending.completed)schedule();
 };
 function advance():void{
  scheduled=false;
  // Compile source geometry independently of the main render/event thread.
  // No frame, glyph, camera snapshot or text publisher lives in this worker.
  // Read importance orders initial work, but a complex receiver cannot keep
  // every other source pending until it finishes. Service changes only work
  // order, never source selection, coverage, native vocabulary or output.
  let selected:number|undefined;for(const [id,job]of jobs){if(!job.work)continue;const previous=selected===undefined?undefined:jobs.get(selected)!;if(!previous||job.priority/(1+job.service)>previous.priority/(1+previous.service))selected=id;}
  if(selected===undefined)return;
  const job=jobs.get(selected)!,started=now();
  // A required visible source is still part of the first-publication cohort;
  // it must not be published early or replaced with a field-only placeholder.
  // Give that bounded cohort a larger worker quantum so the compiler spends
  // less time yielding between its already-yielding stages. Background work
  // keeps the old 4 ms slice and fairness rule.
  const until=started+(job.priority>=1_000_000?REQUIRED_WORKER_QUANTUM_MS:BACKGROUND_WORKER_QUANTUM_MS);let operations=0;
  try{
   do{const next=job.work!.next();operations++;if(next.done){
    const program='program' in next.value?next.value.program:undefined;
    if(program?.complexMode==='summary'){
     if(job.cacheKey)stored.remember(job.cacheKey,program);
     else if(job.cachePending)retainCache(job,program);
    }
    job.completed=true;job.work=undefined;jobs.delete(selected);
    const value=program?{program:program.complexMode==='full'?{...program,complex:encodeSourceComplex(program.complex)}:program}:next.value;
    scope.postMessage({id:selected,...value},transfers(value));break;
   }}
   while(operations<4096&&now()<until);
  }catch(error){job.completed=true;job.work=undefined;jobs.delete(selected);scope.postMessage({id:selected,error:error instanceof Error?error.message:String(error)});}
  job.service+=now()-started;schedule();
 }
 const handle=(
  {data}:MessageEvent<SourceWorkerRequest>
 )=>{
  const job=jobs.get(data.id);
  if(data.cancel){if(job){job.cancelled=true;releaseCache(job);job.work?.return(undefined as never);job.work=undefined;}jobs.delete(data.id);return;}
  if(data.source){
   const sourceInput=data.source,pending:SourceJob={priority:data.priority,service:0,cachePending:true};jobs.set(data.id,pending);
   // A source's complete compiler is the post-init first-publication producer.
   // Do not serialize its first worker slice behind the IndexedDB/hash lookup:
   // on a cold phone that lookup is guaranteed to miss but still hashes the
   // complete geometry and lineage. Start the same complete compiler
   // immediately, while retaining the cache lookup as an opportunistic warm-
   // hit replacement before the job receives service. No partial program is
   // ever posted.
   pending.work=source(sourceInput);
   void stored.find(sourceInput).then(hit=>cacheResult(pending,hit),()=>{pending.cachePending=false;releaseCache(pending);if(!pending.completed)schedule();});
   schedule();
  }
  else if(data.mesh)jobs.set(data.id,{priority:data.priority,service:0,work:parts(data.mesh)});
  else if(job)job.priority=data.priority;
  schedule();
 };
 return Object.assign(handle,{cacheRetention:()=>({bytes:retainedCacheBytes,entries:retainedCacheEntries})});
}

const scope=globalThis as unknown as SourceWorkerScope;
const stored=new PersistentSourcePrograms(sourceProgramStoreNamespace(import.meta.url),typeof indexedDB==='undefined'?undefined:new IndexedSourceProgramStore(indexedDB));
scope.onmessage=createSourceWorkerHandler(scope,stored);
