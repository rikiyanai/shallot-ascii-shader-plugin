import {meshPartWork,type MeshPartGraph,type PartMesh} from './mesh-part-graph.ts';
import {sourceProgramWork,type CompiledSourceProgram,type SourceProgramInput} from './source-program.ts';
import {encodeSourceComplex} from './source-transfer.ts';
import {IndexedSourceProgramStore,PersistentSourcePrograms,sourceProgramStoreNamespace} from './source-program-store.ts';

type Request={id:number;priority:number;mesh?:PartMesh;source?:SourceProgramInput;cancel?:boolean};
type Result={graph:MeshPartGraph}|{program:CompiledSourceProgram};
const scope=globalThis as unknown as {onmessage:(event:MessageEvent<Request>)=>void;postMessage:(value:unknown,transfer?:Transferable[])=>void};
const jobs=new Map<number,{priority:number;service:number;work?:Generator<void,Result>;cacheKey?:string}>();
const stored=new PersistentSourcePrograms(sourceProgramStoreNamespace(import.meta.url),typeof indexedDB==='undefined'?undefined:new IndexedSourceProgramStore(indexedDB));
function* parts(mesh:PartMesh):Generator<void,Result>{return {graph:yield* meshPartWork(mesh)};}
function* source(mesh:SourceProgramInput):Generator<void,Result>{return {program:yield* sourceProgramWork(mesh)};}
function transfers(value:unknown,buffers=new Set<ArrayBuffer>()):ArrayBuffer[]{
 if(ArrayBuffer.isView(value)){if(value.buffer instanceof ArrayBuffer)buffers.add(value.buffer);}
 else if(value&&typeof value==='object')for(const member of Object.values(value))transfers(member,buffers);
 return [...buffers];
}
let scheduled=false;
function schedule():void{if(!scheduled&&[...jobs.values()].some(job=>job.work)){scheduled=true;setTimeout(advance,0);}}
function advance():void{
 scheduled=false;
 // Compile source geometry independently of the main render/event thread.
 // No frame, glyph, camera snapshot or text publisher lives in this worker.
 // Read importance orders initial work, but a complex receiver cannot keep
 // every other source pending until it finishes. Service changes only work
 // order, never source selection, coverage, native vocabulary or output.
 let selected:number|undefined;for(const [id,job]of jobs){if(!job.work)continue;const previous=selected===undefined?undefined:jobs.get(selected)!;if(!previous||job.priority/(1+job.service)>previous.priority/(1+previous.service))selected=id;}
 if(selected===undefined)return;
 const job=jobs.get(selected)!,started=performance.now(),until=started+4;let operations=0;
 try{
  do{const next=job.work!.next();operations++;if(next.done){jobs.delete(selected);
   const program='program' in next.value?next.value.program:undefined;
   if(program?.complexMode==='summary')stored.remember(job.cacheKey,program);
   const value=program?{program:program.complexMode==='full'?{...program,complex:encodeSourceComplex(program.complex)}:program}:next.value;
   scope.postMessage({id:selected,...value},transfers(value));break;
  }}
  while(operations<4096&&performance.now()<until);
 }catch(error){jobs.delete(selected);scope.postMessage({id:selected,error:error instanceof Error?error.message:String(error)});}
 job.service+=performance.now()-started;schedule();
}
scope.onmessage=({data})=>{
 const job=jobs.get(data.id);
 if(data.cancel){job?.work?.return(undefined as never);jobs.delete(data.id);return;}
 if(data.source){
  const sourceInput=data.source,pending={priority:data.priority,service:0} as {priority:number;service:number;work?:Generator<void,Result>;cacheKey?:string};jobs.set(data.id,pending);
  void stored.find(sourceInput).then(hit=>{
   if(jobs.get(data.id)!==pending)return;
   // A hit is already persisted. Do not clone/write the whole program again.
   pending.cacheKey=hit?.program?undefined:hit?.key;
   pending.work=hit?.program?(function*():Generator<void,Result>{return {program:hit.program!};})():source(sourceInput);
   schedule();
  });
 }
 else if(data.mesh)jobs.set(data.id,{priority:data.priority,service:0,work:parts(data.mesh)});
 else if(job)job.priority=data.priority;
 schedule();
};
