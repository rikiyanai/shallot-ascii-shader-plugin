import {type World,GlobalTransform,MeshInstance,Text,Camera,DirectionalLight} from '@dylanebert/shallot';

/** STILL EXAMPLE VIEW (frame cost, 2026-11-13). The plugin started a new text
 * job for every host frame in which its target was not pending, so a standing
 * example (city at 640: about 6 jobs a second, 78 ms of GPU each) redrew the
 * same page from the same inputs. A target starts no job while EVERY input of
 * its last started job is exactly equal again (no tolerance) and that job was
 * published; the displayed tape keeps being presented, as PhysicalFrame does
 * for Y8 (still()/startKey()).
 *
 * The inputs are of two kinds.
 * 1. CPU values the job reads or that pose its sources, compared bit for bit
 *    (StillKey): the controller state the job depends on, each view's slot,
 *    size, viewport and whole ViewUniforms slice (clip, eye, worldFromClip,
 *    exposure ...), its labels, the key light and fog, every material draw and
 *    phase-captured draw (buffers, bind groups, arguments, source revisions),
 *    and the entities the drawing reads (MeshInstance, Text, Camera,
 *    DirectionalLight: identity and GlobalTransform). Other entities (events,
 *    helpers created and destroyed every frame) reach the drawing only through
 *    what the engine draws, which the appearance hash covers.
 * 2. What the engine computed on the GPU from them (GPU particles, animated
 *    shaders, interpolated poses, shadows): one exact hash over every texel of
 *    the target's finished appearance, the same image the job copies
 *    (AppearanceHash, read back asynchronously). A view is still only when the
 *    newest hash read back is from a frame AFTER the job's frame and equals the
 *    hash of the job's own frame. A change therefore starts a job at most the
 *    readback latency (a few frames) later; motion is never frozen. */

export type StillReason=''|'appearance-wait'|'appearance-blip'|'first'|'unpublished'|'views'|'controller'|'view'|'labels'|'light'|'fog'|'materials'|'phase'|'contacts'|'entities'|'transforms'|'appearance-pending'|'appearance';

/** Plain values compared with Object.is, in push order, one list per part. */
export class StillParts {
 readonly parts=new Map<string,unknown[]>();
 push(part:string,...values:unknown[]):void{let list=this.parts.get(part);if(!list){list=[];this.parts.set(part,list);}for(const value of values)list.push(value);}
 pushArray(part:string,values:ArrayLike<number>|undefined):void{if(!values){this.push(part,undefined);return;}this.push(part,values.length);for(let i=0;i<values.length;i++)this.push(part,values[i]);}
 /** The first part (in insertion order of `last`) whose values differ, or ''. */
 differs(last:StillParts):string{
  if(last.parts.size!==this.parts.size)return 'views';
  for(const [part,values]of last.parts){
   const now=this.parts.get(part);if(!now||now.length!==values.length)return part;
   for(let i=0;i<values.length;i++)if(!Object.is(now[i],values[i]))return part;
  }
  return '';
 }
}

const COLUMNS=['translation','rotation','scale','linearVelocity'] as const,LANES=4;
type Column=ArrayLike<number>&{buffer:ArrayBufferLike;byteOffset:number;length:number};
function words(column:Column):Uint32Array{return new Uint32Array(column.buffer,column.byteOffset,column.length);}
function liveColumns(world:World):Uint32Array[]{const storage=world.storage(GlobalTransform) as unknown as Record<string,{column:Column}>;return COLUMNS.map(name=>words(storage[name]!.column));}
/** Entities the drawing reads, in a stable order (eid ascending, no duplicates). */
function drawn(world:World):number[]{
 const set=new Set<number>();
 for(const component of [MeshInstance,Text,Camera,DirectionalLight])for(const eid of world.query([component]))set.add(eid);
 return [...set].sort((a,b)=>a-b);
}

/** The world part shared by every target: the entities the drawing reads (eid
 * and world.ref, generation * 2^32 + eid, hence Float64Array) and
 * their GlobalTransform slots (bit-exact, so NaN and -0 compare). */
export class StillWorld {
 private constructor(readonly entities:Float64Array,readonly poses:Uint32Array){}
 static snapshot(world:World):StillWorld{
  const eids=drawn(world),entities=new Float64Array(eids.length*2),poses=new Uint32Array(eids.length*COLUMNS.length*LANES),columns=liveColumns(world);
  let i=0,k=0;
  for(const eid of eids){entities[i++]=eid;entities[i++]=world.ref(eid);for(const column of columns)for(let lane=0;lane<LANES;lane++)poses[k++]=column[eid*LANES+lane]??0;}
  return new StillWorld(entities,poses);
 }
 differs(world:World):''|'entities'|'transforms'{
  const eids=drawn(world);if(this.entities.length!==eids.length*2)return 'entities';
  let i=0;for(const eid of eids){if(this.entities[i++]!==eid||this.entities[i++]!==world.ref(eid))return 'entities';}
  const columns=liveColumns(world);let k=0;
  for(const eid of eids)for(const column of columns)for(let lane=0;lane<LANES;lane++)if(this.poses[k++]!==(column[eid*LANES+lane]??0))return 'transforms';
  return '';
 }
}

export const APPEARANCE_HASH_SHADER=/* wgsl */`
struct Band {y0:u32,rows:u32,width:u32,index:u32}
@group(0) @binding(0) var image:texture_2d<f32>;
@group(0) @binding(1) var<storage,read_write> sums:array<atomic<u32>>;
@group(0) @binding(2) var<uniform> band:Band;
var<workgroup> local:array<atomic<u32>,4>;
fn mix32(v:u32)->u32{var x=v;x^=x>>16u;x*=0x7feb352du;x^=x>>15u;x*=0x846ca68bu;x^=x>>16u;return x;}
// Every texel of one band of the finished appearance, 4x4 texels per thread,
// summed into that band's four words.
// Order-independent: wrapping sums and a xor of position-keyed texel hashes.
@compute @workgroup_size(16,16) fn main(@builtin(global_invocation_id) id:vec3u,@builtin(local_invocation_index) li:u32){
 if(li<4u){atomicStore(&local[li],0u);}
 workgroupBarrier();
 var a=0u;var b=0u;var c=0u;var n=0u;
 for(var dy=0u;dy<4u;dy++){
  for(var dx=0u;dx<4u;dx++){
   let x=id.x*4u+dx;let y=id.y*4u+dy;
   if(x<band.width&&y<band.rows){
    let t=bitcast<vec4u>(textureLoad(image,vec2i(i32(x),i32(y)),0));
    let at=(band.y0+y)*band.width+x;
    let h=mix32(t.x^mix32(t.y^mix32(t.z^mix32(t.w^mix32(at+0x9e3779b9u)))));
    a+=h;b^=mix32(h+at);c+=mix32(h^0x85ebca6bu);n+=1u;
   }
  }
 }
 atomicAdd(&local[0],a);atomicXor(&local[1],b);atomicAdd(&local[2],c);atomicAdd(&local[3],n);
 workgroupBarrier();
 let o=band.index*4u;
 if(li==0u){atomicAdd(&sums[o],atomicLoad(&local[0]));atomicXor(&sums[o+1u],atomicLoad(&local[1]));atomicAdd(&sums[o+2u],atomicLoad(&local[2]));atomicAdd(&sums[o+3u],atomicLoad(&local[3]));}
}`;

const BAND_ROWS=256,STAGING=6,MAX_BANDS=64,WAIT_FRAMES=30;
/** One target's appearance hash per host frame, copied band by band through a
 * small texture (no second full-size image), read back without stalling. */
export class AppearanceHash {
 private pipeline?:GPUComputePipeline;private band?:GPUTexture;private bands?:GPUBuffer;private bandCount=0;
 private readonly sums:GPUBuffer;private readonly free:GPUBuffer[]=[];private pending:{buffer:GPUBuffer;frame:number;count:number}[]=[];private submittedReads:{buffer:GPUBuffer;frame:number;count:number}[]=[];
 private disposed=false;private waitFrom?:number;private blipFrom?:number;
 /** Consecutive resolved frames after the job frame whose hash differs from it. */
 private streak=0;
 /** Newest hash read back, and the hash of the watched (job) frame. */
 latest?:{frame:number;hash:string};watched?:{frame:number;hash?:string};
 constructor(private readonly device:GPUDevice){
  this.sums=device.createBuffer({label:'MAIN still appearance hash',size:16*MAX_BANDS,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
  for(let i=0;i<STAGING;i++)this.free.push(device.createBuffer({label:'MAIN still appearance hash readback',size:16*MAX_BANDS,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}));
 }
 /** The job started on `frame`: still needs that frame's hash. */
 watch(frame:number):void{this.watched={frame};this.waitFrom=undefined;this.blipFrom=undefined;this.streak=0;}







 state(frame:number):''|'appearance-wait'|'appearance-blip'|'appearance-pending'|'appearance'{
  const watched=this.watched,latest=this.latest;
  if(!watched)return 'appearance-pending';
  if(watched.hash===undefined||!latest||latest.frame<=watched.frame){this.waitFrom??=frame;return frame-this.waitFrom<WAIT_FRAMES?'appearance-wait':'appearance-pending';}
  if(latest.hash===watched.hash){this.blipFrom=undefined;return '';}
  if(this.streak>=3)return 'appearance';
  this.blipFrom??=frame;return frame-this.blipFrom<WAIT_FRAMES?'appearance-blip':'appearance';
 }
 encode(encoder:GPUCommandEncoder,source:GPUTexture,frame:number):void{
  const buffer=this.free.pop();if(!buffer)return;
  const device=this.device,width=source.width,height=source.height;
  if(!this.band||this.band.width!==width||this.band.format!==source.format){
   this.band?.destroy();this.band=device.createTexture({label:'MAIN still appearance band',size:[width,BAND_ROWS],format:source.format,usage:GPUTextureUsage.COPY_DST|GPUTextureUsage.TEXTURE_BINDING});
  }
  const count=Math.ceil(height/BAND_ROWS);if(count>MAX_BANDS){this.free.push(buffer);return;}
  if(!this.bands||this.bandCount<count){
   this.bands?.destroy();this.bandCount=count;
   this.bands=device.createBuffer({label:'MAIN still appearance bands',size:256*count,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
  }
  const words=new Uint32Array(64*count);for(let i=0;i<count;i++){words[i*64]=i*BAND_ROWS;words[i*64+1]=Math.min(BAND_ROWS,height-i*BAND_ROWS);words[i*64+2]=width;words[i*64+3]=i;}
  device.queue.writeBuffer(this.bands,0,words);
  this.pipeline??=device.createComputePipeline({label:'MAIN still appearance hash',layout:'auto',compute:{module:device.createShaderModule({label:'MAIN still appearance hash',code:APPEARANCE_HASH_SHADER}),entryPoint:'main'}});
  encoder.clearBuffer(this.sums);
  for(let i=0;i<count;i++){
   const rows=Math.min(BAND_ROWS,height-i*BAND_ROWS);
   encoder.copyTextureToTexture({texture:source,origin:[0,i*BAND_ROWS,0]},{texture:this.band},[width,rows]);
   const pass=encoder.beginComputePass({label:'MAIN still appearance hash'});pass.setPipeline(this.pipeline);
   pass.setBindGroup(0,device.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:this.band.createView()},{binding:1,resource:{buffer:this.sums}},{binding:2,resource:{buffer:this.bands,offset:256*i,size:16}}]}));
   pass.dispatchWorkgroups(Math.ceil(width/64),Math.ceil(rows/64));pass.end();
  }
  encoder.copyBufferToBuffer(this.sums,0,buffer,0,16*count);
  this.pending.push({buffer,frame,count});
 }
 /** After the host submit that carried encode(). */
 submitted():void{
  for(const read of this.pending.splice(0)){
   this.submittedReads.push(read);
   read.buffer.mapAsync(GPUMapMode.READ).then(()=>{
    if(this.disposed)return;
    const words=new Uint32Array(read.buffer.getMappedRange(),0,read.count*4),bands:string[]=[];
    for(let i=0;i<read.count;i++)bands.push(Array.from(words.subarray(i*4,i*4+4),word=>word.toString(16)).join('.'));
    const hash=bands.join('|');read.buffer.unmap();
    if(this.watched?.frame===read.frame){this.watched.hash=hash;this.streak=0;}
    else if(this.watched?.hash!==undefined&&read.frame>this.watched.frame)this.streak=hash===this.watched.hash?0:this.streak+1;
    if(!this.latest||read.frame>this.latest.frame)this.latest={frame:read.frame,hash};
    this.submittedReads=this.submittedReads.filter(item=>item!==read);this.free.push(read.buffer);
   },()=>{});
  }
 }
 dispose():void{this.disposed=true;this.sums.destroy();this.band?.destroy();this.bands?.destroy();for(const buffer of this.free)buffer.destroy();for(const read of [...this.pending,...this.submittedReads])read.buffer.destroy();}
}
