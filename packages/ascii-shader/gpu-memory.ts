/**
 * EVERY MAIN GPU BUFFER AND TEXTURE, COUNTED WHERE IT IS MADE (2026-11-12).
 * The memoryMiB allowance used to count only source positions and indices;
 * the renderer's own buffers, textures and an arena replaced in the same frame
 * as its successor were outside it. MAIN is given an accounting view of the
 * host device: createBuffer and createTexture add the resource's bytes, its
 * destroy() (or its collection) removes them, so a replaced buffer stays in
 * the total until it is destroyed. queue.writeBuffer is counted per buffer
 * label, which is how the arena's uploads after load are read.
 * The view forwards every other member to the host device unchanged; the host
 * engine's own resources on the same device are not MAIN's and are not counted.
 */
export interface GpuMemoryLedger {
 /** The device MAIN creates its resources on. */
 readonly device:GPUDevice;
 /** Bytes of live MAIN buffers and textures. */
 readonly bytes:number;
 /** Highest `bytes` since the last takePeak(). */
 takePeak():number;
 readonly peakBytes:number;
 readonly buffers:number;readonly textures:number;
 /** Live bytes per resource label. */
 readonly live:ReadonlyMap<string,{count:number;bytes:number}>;
 /** queue.writeBuffer calls and bytes per destination label, since creation. */
 readonly written:ReadonlyMap<string,{writes:number;bytes:number}>;
 /** Resources created per label, since creation (a replaced arena adds one). */
 readonly created:ReadonlyMap<string,{count:number;bytes:number}>;
}

const COMPRESSED=/^(bc|etc2|eac|astc)/;
/** Bytes per texel of an uncompressed format (channels × bits); compressed
 * formats count one byte per texel, an upper bound for BC/ETC/ASTC. */
export function texelBytes(format:GPUTextureFormat):number{
 if(COMPRESSED.test(format))return 1;
 switch(format){
  case 'depth16unorm':return 2;case 'stencil8':return 1;
  case 'depth24plus':case 'depth32float':case 'depth24plus-stencil8':return 4;case 'depth32float-stencil8':return 8;
  case 'rgb10a2unorm':case 'rgb10a2uint':case 'rg11b10ufloat':case 'rgb9e5ufloat':return 4;
 }
 const match=/^(rgba|bgra|rg|r)(8|16|32)/.exec(format);if(!match)return 4;
 return match[1]!.length*Number(match[2])/8;
}

function textureBytes(descriptor:GPUTextureDescriptor):number{
 const size=descriptor.size as GPUExtent3DDict|number[];
 const [width,height,layers]=Array.isArray(size)?[size[0]??1,size[1]??1,size[2]??1]:[size.width,size.height??1,size.depthOrArrayLayers??1];
 const three=descriptor.dimension==='3d',texel=texelBytes(descriptor.format)*(descriptor.sampleCount??1);
 let bytes=0;
 for(let level=0;level<(descriptor.mipLevelCount??1);level++)bytes+=Math.max(1,width>>level)*Math.max(1,height>>level)*(three?Math.max(1,layers>>level):layers)*texel;
 return bytes;
}

export function accountGpuMemory(host:GPUDevice):GpuMemoryLedger{
 let bytes=0,peak=0,peakBytes=0,buffers=0,textures=0;
 const live=new Map<string,{count:number;bytes:number}>(),created=new Map<string,{count:number;bytes:number}>(),written=new Map<string,{writes:number;bytes:number}>();
 type Held={size:number;label:string;texture:boolean;released:boolean};
 const release=(held:Held)=>{
  if(held.released)return;held.released=true;bytes-=held.size;if(held.texture)textures--;else buffers--;
  const row=live.get(held.label);if(row){row.count--;row.bytes-=held.size;if(!row.count)live.delete(held.label);}
 };
 // A resource collected without destroy() is released by the collector.
 const collected=new FinalizationRegistry<Held>(release);
 const admit=<T extends GPUBuffer|GPUTexture>(resource:T,size:number,texture:boolean):T=>{
  const held:Held={size,label:resource.label||'(unlabelled)',texture,released:false};
  bytes+=size;if(texture)textures++;else buffers++;peak=Math.max(peak,bytes);peakBytes=Math.max(peakBytes,bytes);
  for(const table of [live,created]){const row=table.get(held.label);if(row){row.count++;row.bytes+=size;}else table.set(held.label,{count:1,bytes:size});}
  const destroy=resource.destroy;collected.register(resource,held,held);
  Object.defineProperty(resource,'destroy',{configurable:true,value:()=>{collected.unregister(held);release(held);destroy.call(resource);}});
  return resource;
 };
 const bound=(target:object,cache:Map<PropertyKey,unknown>,key:PropertyKey):unknown=>{
  if(cache.has(key))return cache.get(key);
  const value=Reflect.get(target,key,target);
  if(typeof value!=='function')return value;
  const call=(value as (...args:unknown[])=>unknown).bind(target);cache.set(key,call);return call;
 };
 const queueCalls=new Map<PropertyKey,unknown>();
 const writeBuffer=(buffer:GPUBuffer,offset:number,data:BufferSource|SharedArrayBuffer,dataOffset?:number,size?:number):void=>{
  const element=ArrayBuffer.isView(data)?(data as ArrayBufferView&{BYTES_PER_ELEMENT?:number}).BYTES_PER_ELEMENT??1:1;
  const total=size!==undefined?size*element:(data as ArrayBuffer).byteLength-(dataOffset??0)*element;
  const row=written.get(buffer.label||'(unlabelled)');if(row){row.writes++;row.bytes+=total;}else written.set(buffer.label||'(unlabelled)',{writes:1,bytes:total});
  (host.queue.writeBuffer as (...args:unknown[])=>void).call(host.queue,buffer,offset,data,...(dataOffset===undefined?[]:[dataOffset]),...(size===undefined?[]:[size]));
 };
 const queue=new Proxy(host.queue,{get:(target,key)=>key==='writeBuffer'?writeBuffer:bound(target,queueCalls,key),set:(target,key,value)=>Reflect.set(target,key,value)});
 const deviceCalls=new Map<PropertyKey,unknown>();
 const createBuffer=(descriptor:GPUBufferDescriptor):GPUBuffer=>admit(host.createBuffer(descriptor),descriptor.size,false);
 const createTexture=(descriptor:GPUTextureDescriptor):GPUTexture=>admit(host.createTexture(descriptor),textureBytes(descriptor),true);
 const device=new Proxy(host,{
  get:(target,key)=>key==='createBuffer'?createBuffer:key==='createTexture'?createTexture:key==='queue'?queue:bound(target,deviceCalls,key),
  set:(target,key,value)=>Reflect.set(target,key,value),
 });
 return {
  device,get bytes(){return bytes;},get peakBytes(){return peakBytes;},get buffers(){return buffers;},get textures(){return textures;},live,written,created,
  takePeak(){const value=peak;peak=bytes;return value;},
 };
}
