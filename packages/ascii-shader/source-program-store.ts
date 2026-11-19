import type {SourceProgramInput,SummaryCompiledSourceProgram} from './source-program.ts';

const SCHEMA='shallot.source-program-cache.v1';
export const SOURCE_PROGRAM_STORE_BYTES=64*1024*1024;
export const SOURCE_PROGRAM_ENTRY_BYTES=8*1024*1024;
const MAX_ENTRIES=4096;
type Entry={readonly key:string;readonly schema:typeof SCHEMA;readonly program:SummaryCompiledSourceProgram};
type Metadata={key:string;bytes:number;touched:number};

/** Persistence is optional revision work, never a second source authority.
 * put must clone synchronously: the worker transfers its buffers afterwards. */
export interface SourceProgramStore {
 get(key:string):Promise<unknown>;
 put(key:string,entry:Entry,bytes:number):void;
 remove(key:string):void;
}

/** Vite's content-addressed worker includes the compiler and authored material
 * table. Unbundled/dev workers deliberately miss instead of reusing stale code. */
export function sourceProgramStoreNamespace(workerUrl:string):string|undefined{
 const name=new URL(workerUrl).pathname.split('/').at(-1)!;
 return /^mesh-part-worker-[A-Za-z0-9_-]{8,}\.js$/.test(name)?`${SCHEMA}:${name}`:undefined;
}

/** Hash ALL revision inputs in the worker, including P0 incidence, normals,
 * lineage, material metadata and the requested retention/backend. No view,
 * light or frame result is cached. Typed views hash their own bytes only. */
export async function sourceProgramInputKey(namespace:string,input:SourceProgramInput):Promise<string>{
 const hashes=new WeakMap<object,Promise<string>>(),hex=(bytes:ArrayBuffer)=>Array.from(new Uint8Array(bytes),b=>b.toString(16).padStart(2,'0')).join('');
 const digest=async(bytes:Uint8Array<ArrayBuffer>)=>hex(await crypto.subtle.digest('SHA-256',bytes));
 const visit=async(value:unknown):Promise<unknown>=>{
  if(ArrayBuffer.isView(value)){
   if(!(value.buffer instanceof ArrayBuffer))throw Error('Shared source memory is not persistable');
   let hash=hashes.get(value);if(!hash){hash=digest(new Uint8Array(value.buffer,value.byteOffset,value.byteLength));hashes.set(value,hash);}
   return [value.constructor.name,value.byteLength,await hash];
  }
  if(value===undefined)return ['undefined'];
  if(typeof value==='number')return ['number',Object.is(value,-0)?'-0':String(value)];
  if(value===null||typeof value==='string'||typeof value==='boolean')return value;
  if(Array.isArray(value))return ['array',await Promise.all(value.map(visit))];
  if(value&&typeof value==='object'&&Object.getPrototypeOf(value)===Object.prototype)
   return ['object',await Promise.all(Object.keys(value).sort().map(async key=>[key,await visit((value as Record<string,unknown>)[key])]))];
  throw Error('Unsupported persistent source input');
 };
 return `${namespace}:${await digest(new TextEncoder().encode(JSON.stringify(await visit(input))))}`;
}

/** Conservative storage accounting counts each backing buffer once and
 * charges scalar/object overhead too. Oversize entries compile normally. */
export function sourceProgramStoredBytes(value:unknown,seen=new Set<object>()):number{
 if(value===null||value===undefined)return 8;
 if(typeof value==='string')return 16+value.length*2;
 if(typeof value!=='object')return 16;
 if(seen.has(value))return 0;seen.add(value);
 if(ArrayBuffer.isView(value))return 32+sourceProgramStoredBytes(value.buffer,seen);
 if(value instanceof ArrayBuffer)return value.byteLength;
 return 64+Object.entries(value).reduce((sum,[key,item])=>sum+key.length*2+sourceProgramStoredBytes(item,seen),0);
}

function valid(entry:unknown,key:string):entry is Entry{
 try{
 const item=entry as Entry|undefined,p=item?.program;
 if(item?.key!==key||item.schema!==SCHEMA||p?.complexMode!=='summary'||Object.hasOwn(p,'complex')||p.packet?.kind!=='gpu-csr-v1')return false;
 const plan=p.packet.plan,g=p.geometry,run=p.analysis?.runProgram;
 return p.summary?.schema==='shallot.source-residency-summary.v1'&&p.summary.edgeCount===plan.edges&&
  plan.prefix instanceof Uint32Array&&plan.prefix[0]===0x4d41494e&&plan.prefix[1]===3&&plan.prefix.length===plan.csrOffset&&plan.words===plan.csrOffset+plan.csr.words&&
  g?.positions instanceof Float32Array&&g.indices instanceof Uint32Array&&!!run&&'topology' in run&&run.topology instanceof Uint32Array&&run.criticalVertices instanceof Uint32Array&&
  (!p.physical||(p.physical.geometry===g&&p.physical.positions instanceof Float32Array&&p.physical.positions.length===g.positions.length/3*4&&
   g.normals instanceof Float32Array&&g.normals.length===p.physical.positions.length&&p.physical.bounds instanceof Float32Array&&p.physical.bounds.length===8));
 }catch{return false;}
}

export class PersistentSourcePrograms {
 private readonly revisions=new Map<string,{revision:number;key?:string;volatile:boolean}>();
 private readonly revoked=new Set<string>();
 constructor(private readonly namespace:string|undefined,private readonly store:SourceProgramStore|undefined){}
 async find(input:SourceProgramInput):Promise<{key:string;program?:SummaryCompiledSourceProgram}|undefined>{
  if(!this.namespace||!this.store||input.complexMode!=='summary'||input.incidence!=='gpu-csr')return;
  try{
   let revision=this.revisions.get(input.id);
   const retire=()=>{revision!.volatile=true;if(revision!.key){this.revoked.add(revision!.key);this.store!.remove(revision!.key);}};
   if(revision?.volatile)return;
   if(revision&&revision.revision!==input.revision){retire();return;}
   if(!revision){
    if(this.revisions.size>=MAX_ENTRIES)return;
    revision={revision:input.revision,volatile:false};this.revisions.set(input.id,revision);
   }
   const key=await sourceProgramInputKey(this.namespace,input);
   if(revision.volatile)return;
   if(revision.key&&revision.key!==key){retire();return;}
   revision.key=key;
   const entry=await this.store.get(key);
   if(revision.volatile)return;
   if(entry===undefined)return {key};
   if(!valid(entry,key)){this.store.remove(key);return {key};}
   return {key,program:entry.program};
  }catch{return;}
 }
 remember(key:string|undefined,program:SummaryCompiledSourceProgram):void{
  if(!key||!this.store||this.revoked.has(key))return;
  const entry:Entry={key,schema:SCHEMA,program},bytes=sourceProgramStoredBytes(entry);
  if(bytes>SOURCE_PROGRAM_ENTRY_BYTES)return;
  try{this.store.put(key,entry,bytes);}catch{/* Storage failure must not fail source compilation. */}
 }
}

/** One origin-wide bounded store. Metadata has no source arrays: eviction
 * never deserializes every old program or duplicates them in worker memory. */
export class IndexedSourceProgramStore implements SourceProgramStore {
 private db?:IDBDatabase;private unavailable=false;private opening?:Promise<IDBDatabase|undefined>;
 private writing=0;
 constructor(private readonly factory:IDBFactory){}
 private open():Promise<IDBDatabase|undefined>{
  if(this.unavailable)return Promise.resolve(undefined);
  if(this.db)return Promise.resolve(this.db);
  if(this.opening)return this.opening;
  return this.opening=new Promise(resolve=>{
   let done=false;const finish=(db?:IDBDatabase)=>{if(done){db?.close();return;}done=true;clearTimeout(timer);this.db=db;if(!db)this.unavailable=true;resolve(db);};
   // Blocked/private storage must not become a new loader barrier.
   const timer=setTimeout(()=>finish(),250);
   try{
    const request=this.factory.open(SCHEMA,1);
    request.onupgradeneeded=()=>{const db=request.result;db.createObjectStore('programs');db.createObjectStore('metadata',{keyPath:'key'});};
    request.onsuccess=()=>{const db=request.result;db.onversionchange=()=>{db.close();this.db=undefined;this.unavailable=true;};finish(db);};
    request.onerror=()=>finish();request.onblocked=()=>finish();
   }catch{finish();}
  });
 }
 async get(key:string):Promise<unknown>{
  const db=await this.open();if(!db)return;
  return new Promise(resolve=>{
   let done=false;const finish=(value?:unknown)=>{if(done)return;done=true;clearTimeout(timer);resolve(value);};
   let tx:IDBTransaction|undefined;
   const timer=setTimeout(()=>{this.unavailable=true;finish();try{tx?.abort();}catch{}},250);
   try{
    tx=db.transaction(['programs','metadata'],'readwrite');const request=tx.objectStore('programs').get(key),metadata=tx.objectStore('metadata');
    request.onsuccess=()=>{if(done)return;const value=request.result;finish(value);if(value!==undefined){const read=metadata.get(key);read.onsuccess=()=>{if(read.result)metadata.put({...read.result,touched:Date.now()});};}};
    request.onerror=()=>finish();tx.onabort=()=>finish();
   }catch{finish();}
  });
 }
 put(key:string,entry:Entry,bytes:number):void{
  const db=this.db;if(!db||this.unavailable||this.writing+bytes>SOURCE_PROGRAM_ENTRY_BYTES)return;
  const tx=db.transaction(['programs','metadata'],'readwrite'),programs=tx.objectStore('programs'),metadata=tx.objectStore('metadata');
  this.writing+=bytes;let settled=false;
  const release=()=>{if(settled)return;settled=true;this.writing-=bytes;};
  tx.oncomplete=release;tx.onabort=()=>{release();this.unavailable=true;};
  // IDB clones here, before postMessage detaches the compiled buffers.
  programs.put(entry,key);metadata.put({key,bytes,touched:Date.now()} satisfies Metadata);
  const request=metadata.getAll();request.onsuccess=()=>{
   const records=request.result as Metadata[];let total=records.reduce((n,item)=>n+item.bytes,0),count=records.length;
   for(const item of records.sort((a,b)=>a.touched-b.touched||a.key.localeCompare(b.key))){
    if(total<=SOURCE_PROGRAM_STORE_BYTES&&count<=MAX_ENTRIES)break;
    programs.delete(item.key);metadata.delete(item.key);total-=item.bytes;count--;
   }
  };
 }
 remove(key:string):void{
  if(!this.db||this.unavailable)return;
  try{const tx=this.db.transaction(['programs','metadata'],'readwrite');tx.objectStore('programs').delete(key);tx.objectStore('metadata').delete(key);}catch{}
 }
}
