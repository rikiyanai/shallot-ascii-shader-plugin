/** Source residency's storage owner. Ranges, not GPUBuffer identity, identify
 * data. Pages never move live ranges; retirement waits for the caller's fence.
 * A page is bounded by the device storage binding limit, not scene size. */
export interface SourceStorageRange {
 readonly slot:number;readonly generation:number;readonly page:number;
 readonly byteOffset:number;readonly byteLength:number;
}
export type SourceStorageBinding=GPUBuffer|GPUBufferBinding;
export function storageBinding(value:SourceStorageBinding):GPUBufferBinding {
 return 'buffer' in value?value:{buffer:value};
}
export function sameStorageBinding(a:SourceStorageBinding,b:SourceStorageBinding):boolean {
 const x=storageBinding(a),y=storageBinding(b);
 return x.buffer===y.buffer&&(x.offset??0)===(y.offset??0)&&
  (x.size??x.buffer.size-(x.offset??0))===(y.size??y.buffer.size-(y.offset??0));
}
/** A revision/batch is registered as a whole. Failed allocation retires only
 * this transaction's new ranges; unrelated residents keep their identities. */
export function allocateStorageRanges<T extends Record<string,{storage:SourceStorage;data:ArrayBufferView|number}>>(inputs:T):{[K in keyof T]:SourceStorageRange} {
 const ranges={} as {[K in keyof T]:SourceStorageRange},made:{storage:SourceStorage;range:SourceStorageRange}[]=[];
 try{
  for(const key in inputs){const {storage,data}=inputs[key]!,range=typeof data==='number'?storage.allocate(data):storage.upload(data);ranges[key]=range;made.push({storage,range});}
  return ranges;
 }catch(error){for(const {storage,range} of made)storage.retire(range);throw error;}
}
interface Page {
 readonly id:number;readonly buffer:GPUBuffer;readonly capacity:number;
 readonly free:{offset:number;size:number}[];live:number;
}
interface Allocation {readonly range:SourceStorageRange;readonly page:Page;readonly reserved:number;retiring:boolean}
export class SourceStorage {
 private readonly pages=new Map<number,Page>();
 private readonly allocations=new Map<number,Allocation>();
 private readonly retired:Allocation[]=[];private serial=0;private pageSerial=0;private disposed=false;
 private readonly alignment:number;private readonly maximum:number;private nextPageBytes:number;
 constructor(private readonly device:GPUDevice,private readonly label:string,private readonly pageBytes=256*1024){
  this.alignment=Math.max(16,device.limits.minStorageBufferOffsetAlignment??256);
  this.maximum=Math.floor(Math.min(device.limits.maxBufferSize,device.limits.maxStorageBufferBindingSize)/this.alignment)*this.alignment;
  if(this.maximum<this.alignment)throw Error('Source storage device cannot bind one aligned range');
  this.nextPageBytes=Math.min(this.maximum,Math.ceil(Math.max(this.alignment,pageBytes)/this.alignment)*this.alignment);
 }
 /** A physically separate pool for a different pass usage class. Disjoint
  * ranges of ONE buffer cannot mix storage-read and storage-write in a pass. */
 separate(label:string):SourceStorage{return new SourceStorage(this.device,label,this.pageBytes);}
 allocate(bytes:number):SourceStorageRange {
  if(this.disposed)throw Error('Source storage is disposed');
  if(!Number.isSafeInteger(bytes)||bytes<1||bytes%4)throw Error('Source storage requires a positive whole-word byte count');
  const reserved=Math.ceil(bytes/this.alignment)*this.alignment;
  if(reserved>this.maximum)throw Error('Source storage range exceeds device binding capacity; split source before residency');
  let page:Page|undefined,hole=-1,best=Infinity;
  for(const candidate of this.pages.values())for(let i=0;i<candidate.free.length;i++){
   const size=candidate.free[i]!.size;if(size>=reserved&&size<best){page=candidate;hole=i;best=size;}
  }
  if(!page){
   const capacity=Math.min(this.maximum,Math.max(reserved,this.nextPageBytes));
   const id=++this.pageSerial,buffer=this.device.createBuffer({label:`${this.label} page ${id}`,size:capacity,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
   page={id,buffer,capacity,free:[{offset:0,size:capacity}],live:0};this.pages.set(id,page);hole=0;
   this.nextPageBytes=Math.min(this.maximum,8*1024*1024,capacity*2);
  }
  const span=page.free[hole]!,offset=span.offset;span.offset+=reserved;span.size-=reserved;if(!span.size)page.free.splice(hole,1);
  // A monotonically increasing slot is a handle, never an address or revision.
  const slot=++this.serial,range=Object.freeze({slot,generation:slot,page:page.id,byteOffset:offset,byteLength:bytes});
  this.allocations.set(slot,{range,page,reserved,retiring:false});page.live++;return range;
 }
 upload(data:ArrayBufferView):SourceStorageRange {
  const range=this.allocate(data.byteLength);try{this.write(range,data);return range;}catch(error){this.retire(range);throw error;}
 }
 private allocation(range:SourceStorageRange):Allocation {
  const allocation=this.allocations.get(range.slot);
  if(!allocation||allocation.range!==range||allocation.retiring)throw Error('Source storage range is retired, foreign or stale');
  return allocation;
 }
 binding(range:SourceStorageRange):GPUBufferBinding {
  const {page}=this.allocation(range);return {buffer:page.buffer,offset:range.byteOffset,size:range.byteLength};
 }
 write(range:SourceStorageRange,data:ArrayBufferView,offset=0):void {
  const {page}=this.allocation(range);
  if(!Number.isSafeInteger(offset)||offset<0||offset%4||data.byteLength%4||offset+data.byteLength>range.byteLength)throw Error('Source storage write exceeds its registered range');
  this.device.queue.writeBuffer(page.buffer,range.byteOffset+offset,data.buffer,data.byteOffset,data.byteLength);
 }
 retire(range:SourceStorageRange):void {
  const allocation=this.allocation(range);allocation.retiring=true;this.retired.push(allocation);
 }
 /** Call ONLY after every submitted reader/writer of retired ranges completes,
  * or its encoder is discarded. A host tick is not a submission fence. */
 submitted():void {
  for(const allocation of this.retired.splice(0)){
   this.allocations.delete(allocation.range.slot);const page=allocation.page;page.live--;
   page.free.push({offset:allocation.range.byteOffset,size:allocation.reserved});page.free.sort((a,b)=>a.offset-b.offset);
   for(let i=1;i<page.free.length;){const a=page.free[i-1]!,b=page.free[i]!;if(a.offset+a.size===b.offset){a.size+=b.size;page.free.splice(i,1);}else i++;}
   if(!page.live){page.buffer.destroy();this.pages.delete(page.id);}
  }
 }
 /** Residency-time descriptors for merged GPU consumers. No vertex copying,
  * topology decoding or per-frame concatenation is performed here. */
 descriptors():readonly {readonly page:number;readonly buffer:GPUBuffer;readonly bytes:number}[]{
  return [...this.pages.values()].map(page=>({page:page.id,buffer:page.buffer,bytes:page.capacity}));
 }
 get bytes():number{let bytes=0;for(const page of this.pages.values())bytes+=page.capacity;return bytes;}
 dispose():void{if(this.disposed)return;this.disposed=true;for(const page of this.pages.values())page.buffer.destroy();this.pages.clear();this.allocations.clear();this.retired.length=0;}
}
