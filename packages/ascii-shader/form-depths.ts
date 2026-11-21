import {DeviceComputePrograms,type ComputePipelineProgress} from './gpu-pipeline-preparation.ts';
import {isolateComputeEntry} from './gpu-compute-program.ts';
import type {SurfaceField} from './surface-fields.ts';
import {FORM_LOD_STATE_WORDS,conservativeFormLodScope,formLodScopeKey,validateFormLodScope} from './form-lod-state.ts';
import type {FormLodScope} from './form-lod-state.ts';












































export const FORM_DEPTH_SLOTS=65536,FORM_DEPTH_COLUMNS=256,FORM_DEPTH_PROBES=32,FORM_SLOT_WORDS=16;
export const FORM_DEPTH_HEADER_ROW=FORM_DEPTH_SLOTS*FORM_SLOT_WORDS/4/FORM_DEPTH_COLUMNS;





export const FORM_SOURCE_ROW=FORM_DEPTH_HEADER_ROW+1,FORM_SOURCE_TEXELS=2;
export const FORM_DEPTH_ROWS=FORM_SOURCE_ROW+FORM_DEPTH_SLOTS*FORM_SOURCE_TEXELS/FORM_DEPTH_COLUMNS;
/** The table's final word is written after reduction, never by pixel reduce. */
export const FORM_LOD_WORD=15;
/** Writer-only state; readers still bind only the existing form texture. */
export const FORM_LOD_STATE_BYTES=FORM_DEPTH_SLOTS*FORM_LOD_STATE_WORDS*4;





export const NEAR_FROM_ROWS=7;
/** Rows under which a form is 遠景 far, a place on the merged strip: the
 * markup's super-far marks are one to three rows; the strip's compiled words
 * span two rows at most (个 over a | stem), and a cap, a side and a return on
 * rows of their own need three. The run pass's part size band uses 51 px. */
export const FAR_STRIP_BELOW_ROWS=3;
/** Smallest whole forms below this extent join the existing far silhouette.
 * Shared by early source work selection and final far construction. */
export const FAR_FORM_MERGE_PIXELS=25.5;
/** Shared lower edge of the existing LOD hysteresis band. Early work
 * selection must not discard detail the final held tier could still draw. */
export const LOD_TIER_HOLD=.8;
/** An owner SPANS DEPTH when its farthest visible point is this many times as
 * far as its nearest (world units per row grow by this factor across it): one
 * reference length then holds NEAR_FROM_ROWS rows at its near end and under
 * FAR_STRIP_BELOW_ROWS at its far end, so no single tier describes it. */
export const FORM_DEPTH_SPAN=NEAR_FROM_ROWS/FAR_STRIP_BELOW_ROWS;
const FORM_DEPTH_KEY=/* wgsl */`
fn formDepthKey(owner:vec2u)->vec2u{return vec2u((owner.x&0xdfffffffu)|((owner.y>>31u)<<29u),(owner.y<<1u)|1u);}
fn formDepthHash(key:vec2u)->u32{var h=(key.x*0x9e3779b9u)^(key.y*0x85ebca6bu);h=(h^(h>>16u))*0x7feb352du;return h^(h>>15u);}
// World units one text row (row field pixels) covers at field point f and
// depth, unprojected through the frame's own matrix.
fn tierRowUnits(f:vec2f,depth:f32,extent:vec2f,row:f32,worldFromClip:mat4x4f)->f32{
 let a=worldFromClip*vec4f(f/extent*vec2f(2,-2)+vec2f(-1,1),depth,1);
 let b=worldFromClip*vec4f((f+vec2f(0.0,row))/extent*vec2f(2,-2)+vec2f(-1,1),depth,1);
 return length(b.xyz/b.w-a.xyz/a.w);
}
// The owner spans depth (FORM_DEPTH_SPAN), read at the field centre.
fn formSpansDepth(near:f32,far:f32,extent:vec2f,worldFromClip:mat4x4f)->bool{
 let c=extent*.5;
 return tierRowUnits(c,far,extent,1.0,worldFromClip)>=tierRowUnits(c,near,extent,1.0,worldFromClip)*${FORM_DEPTH_SPAN};
}
// The owner's visible box (top, bottom, left, right in field pixels, the
// ends exclusive) touches an edge of the field: the page cuts it, so its
// visible rows and its visible world box are only lower bounds of its size.
fn formCut(box:vec4f,extent:vec2f)->bool{return box.x<=0.0||box.z<=0.0||box.y>=extent.y||box.w>=extent.x;}
fn formOrdered(v:f32)->u32{let b=bitcast<u32>(v);return select(b|0x80000000u,~b,(b&0x80000000u)!=0u);}
fn formUnordered(o:u32)->f32{return bitcast<f32>(select(~o,o&0x7fffffffu,(o&0x80000000u)!=0u));}
`;
/** Reader side. The including module declares `formDepths:texture_2d<u32>`
 * bound to FormDepths.view(field) of the SAME field as its other channels. */
export const FORM_DEPTH_WGSL=/* wgsl */`
${FORM_DEPTH_KEY}
// One owner's slot this frame (field pixels; top and left inclusive, bottom
// and right exclusive; box = top, bottom, left, right).
struct FormExtent {found:bool,near:f32,far:f32,top:f32,bottom:f32,box:vec4f}
fn formExtent(owner:vec2u)->FormExtent{
 let none=FormExtent(false,0.0,0.0,0.0,0.0,vec4f(0.0));
 if((owner.x&0xdfffffffu)==0u){return none;}
 let key=formDepthKey(owner);let h=formDepthHash(key);
 for(var probe=0u;probe<${FORM_DEPTH_PROBES}u;probe++){
  let texel=((h+probe)&${FORM_DEPTH_SLOTS-1}u)*${FORM_SLOT_WORDS/4}u;
  let a=textureLoad(formDepths,vec2i(i32(texel%${FORM_DEPTH_COLUMNS}u),i32(texel/${FORM_DEPTH_COLUMNS}u)),0);
  if(a.x==0u){return none;}
  if(a.x==key.x&&a.y==key.y){
   let b=textureLoad(formDepths,vec2i(i32((texel+1u)%${FORM_DEPTH_COLUMNS}u),i32((texel+1u)/${FORM_DEPTH_COLUMNS}u)),0);
   let box=vec4f(f32(~b.x),f32(b.y),f32(~b.z),f32(b.w));
   return FormExtent(a.z!=0u,bitcast<f32>(a.z),bitcast<f32>(~a.w),box.x,box.y,box);
  }
 }
 return none;
}
// The owner runs up to open background somewhere in view (slot word 14).
fn formUnderSky(owner:vec2u)->bool{
 if((owner.x&0xdfffffffu)==0u){return false;}
 let key=formDepthKey(owner);let h=formDepthHash(key);
 for(var probe=0u;probe<${FORM_DEPTH_PROBES}u;probe++){
  let texel=((h+probe)&${FORM_DEPTH_SLOTS-1}u)*${FORM_SLOT_WORDS/4}u;
  let a=textureLoad(formDepths,vec2i(i32(texel%${FORM_DEPTH_COLUMNS}u),i32(texel/${FORM_DEPTH_COLUMNS}u)),0);
  if(a.x==0u){return false;}
  if(a.x==key.x&&a.y==key.y){return (textureLoad(formDepths,vec2i(i32((texel+3u)%${FORM_DEPTH_COLUMNS}u),i32((texel+3u)/${FORM_DEPTH_COLUMNS}u)),0).z&1u)!=0u;}
 }
 return false;
}
// The owner belongs to an all-or-none compiled parent representation. This
// uses the exact same owner hash/key lookup as formExtent/formUnderSky.
fn formParentSelected(owner:vec2u)->bool{
 if((owner.x&0xdfffffffu)==0u){return false;}
 let key=formDepthKey(owner);let h=formDepthHash(key);
 for(var probe=0u;probe<${FORM_DEPTH_PROBES}u;probe++){
  let texel=((h+probe)&${FORM_DEPTH_SLOTS-1}u)*${FORM_SLOT_WORDS/4}u;
  let a=textureLoad(formDepths,vec2i(i32(texel%${FORM_DEPTH_COLUMNS}u),i32(texel/${FORM_DEPTH_COLUMNS}u)),0);
  if(a.x==0u){return false;}
  if(a.x==key.x&&a.y==key.y){return (textureLoad(formDepths,vec2i(i32((texel+3u)%${FORM_DEPTH_COLUMNS}u),i32((texel+3u)/${FORM_DEPTH_COLUMNS}u)),0).z&2u)!=0u;}
 }
 return false;
}

// result, the writer resolved this owner's held level from its COMPLETE posed
// source bound (resolveSource/resolveLod), not from its surviving pixels; bit 1,
// that source is cut by the page or spans depth (its level is then 0 and its
// lines keep their own local part scale). Every reader of the level (relation
// selection, native articulation, tone) takes the held level of such an owner
// as is, with no visible-extent gate. 0: the incumbent visible read.
fn formSourceFlags(owner:vec2u)->u32{
 if((owner.x&0xdfffffffu)==0u){return 0u;}
 let key=formDepthKey(owner);let h=formDepthHash(key);
 for(var probe=0u;probe<${FORM_DEPTH_PROBES}u;probe++){
  let texel=((h+probe)&${FORM_DEPTH_SLOTS-1}u)*${FORM_SLOT_WORDS/4}u;
  let a=textureLoad(formDepths,vec2i(i32(texel%${FORM_DEPTH_COLUMNS}u),i32(texel/${FORM_DEPTH_COLUMNS}u)),0);
  if(a.x==0u){return 0u;}
  if(a.x==key.x&&a.y==key.y){return (textureLoad(formDepths,vec2i(i32((texel+3u)%${FORM_DEPTH_COLUMNS}u),i32((texel+3u)/${FORM_DEPTH_COLUMNS}u)),0).z>>2u)&3u;}
 }
 return 0u;
}

// header row), in formExtent's form: field-pixel box and reverse-Z depth range.
// found=false: no complete source bound (every reader keeps its visible read).
fn formSourceExtent(owner:vec2u)->FormExtent{
 let none=FormExtent(false,0.0,0.0,0.0,0.0,vec4f(0.0));
 if((owner.x&0xdfffffffu)==0u){return none;}
 let key=formDepthKey(owner);let h=formDepthHash(key);
 for(var probe=0u;probe<${FORM_DEPTH_PROBES}u;probe++){
  let slot=(h+probe)&${FORM_DEPTH_SLOTS-1}u;let texel=slot*${FORM_SLOT_WORDS/4}u;
  let a=textureLoad(formDepths,vec2i(i32(texel%${FORM_DEPTH_COLUMNS}u),i32(texel/${FORM_DEPTH_COLUMNS}u)),0);
  if(a.x==0u){return none;}
  if(a.x==key.x&&a.y==key.y){
   let at=${FORM_SOURCE_ROW*FORM_DEPTH_COLUMNS}u+slot*${FORM_SOURCE_TEXELS}u;
   let b=textureLoad(formDepths,vec2i(i32(at%${FORM_DEPTH_COLUMNS}u),i32(at/${FORM_DEPTH_COLUMNS}u)),0);
   let d=textureLoad(formDepths,vec2i(i32((at+1u)%${FORM_DEPTH_COLUMNS}u),i32((at+1u)/${FORM_DEPTH_COLUMNS}u)),0);
   if(d.z!=1u){return none;}
   let box=bitcast<vec4f>(b);
   return FormExtent(true,bitcast<f32>(d.x),bitcast<f32>(d.y),box.x,box.y,box);
  }
 }
 return none;
}
// The held level is metadata in the same per-form texture, not a second
// reader binding.  eligible is supplied by the existing producer's class
// checks: false for unknown/cut/spanning/terrain/interior/orthographic forms
// preserves the conservative full producer. Producer-row owners (bit 31) may
// opt in when the caller has established that this is a compact physical form.
fn formHeldLod(owner:vec2u,eligible:bool)->u32{
 if(!eligible||(owner.x&0xdfffffffu)==0u){return 0u;}
 let key=formDepthKey(owner);let h=formDepthHash(key);
 for(var probe=0u;probe<${FORM_DEPTH_PROBES}u;probe++){
  let texel=((h+probe)&${FORM_DEPTH_SLOTS-1}u)*${FORM_SLOT_WORDS/4}u;
  let a=textureLoad(formDepths,vec2i(i32(texel%${FORM_DEPTH_COLUMNS}u),i32(texel/${FORM_DEPTH_COLUMNS}u)),0);
  if(a.x==0u){return 0u;}
  if(a.x==key.x&&a.y==key.y){
   let d=textureLoad(formDepths,vec2i(i32((texel+3u)%${FORM_DEPTH_COLUMNS}u),i32((texel+3u)/${FORM_DEPTH_COLUMNS}u)),0);
   return min(d.w,2u);
  }
 }
 return 0u;
}
// The view is OPEN: it shows background (sky) in at least two grid patches
// (reduce), so it can reach a horizon. A view without is enclosed: one room.
fn formOpen()->bool{return textureLoad(formDepths,vec2i(0,${FORM_DEPTH_HEADER_ROW}),0).z>=2u;}
`;
/** Writer side, three dispatches of one pass. reduce: one thread per field
 * pixel; each slot word is raised only when the pixel improves it, so most
 * pixels of a large owner read and do not write. vote: one thread per slot.
 * reference: one thread, one cumulative walk of the histogram (a reduction,
 * nothing is tried or scored). */
const FORM_HEADER=FORM_DEPTH_SLOTS*FORM_SLOT_WORDS;
const FORM_LOD_NEAR_ENTER=NEAR_FROM_ROWS*0.8;
const FORM_LOD_NEAR_LEAVE=NEAR_FROM_ROWS/0.8;
const FORM_LOD_FAR_ENTER=FAR_STRIP_BELOW_ROWS*0.8;
const FORM_LOD_FAR_LEAVE=FAR_STRIP_BELOW_ROWS/0.8;
export const FORM_SOURCE_BOUND_MAGIC=0x53524342;




export interface FormSourceBound {readonly owner:readonly [number,number];readonly low:readonly [number,number,number];readonly high:readonly [number,number,number]}
/** The form table's exact key of an owner pair (formDepthKey). */
export function formDepthKeyOf(x:number,y:number):[number,number]{return [(((x>>>0)&0xdfffffff)|(((y>>>0)>>>31)<<29))>>>0,(((y>>>0)<<1)|1)>>>0];}
function invert4(m:ArrayLike<number>):Float64Array|undefined{
 const a=Float64Array.from({length:16},(_,i)=>m[i]!),inv=new Float64Array(16);
 inv[0]=a[5]!*a[10]!*a[15]!-a[5]!*a[11]!*a[14]!-a[9]!*a[6]!*a[15]!+a[9]!*a[7]!*a[14]!+a[13]!*a[6]!*a[11]!-a[13]!*a[7]!*a[10]!;
 inv[4]=-a[4]!*a[10]!*a[15]!+a[4]!*a[11]!*a[14]!+a[8]!*a[6]!*a[15]!-a[8]!*a[7]!*a[14]!-a[12]!*a[6]!*a[11]!+a[12]!*a[7]!*a[10]!;
 inv[8]=a[4]!*a[9]!*a[15]!-a[4]!*a[11]!*a[13]!-a[8]!*a[5]!*a[15]!+a[8]!*a[7]!*a[13]!+a[12]!*a[5]!*a[11]!-a[12]!*a[7]!*a[9]!;
 inv[12]=-a[4]!*a[9]!*a[14]!+a[4]!*a[10]!*a[13]!+a[8]!*a[5]!*a[14]!-a[8]!*a[6]!*a[13]!-a[12]!*a[5]!*a[10]!+a[12]!*a[6]!*a[9]!;
 inv[1]=-a[1]!*a[10]!*a[15]!+a[1]!*a[11]!*a[14]!+a[9]!*a[2]!*a[15]!-a[9]!*a[3]!*a[14]!-a[13]!*a[2]!*a[11]!+a[13]!*a[3]!*a[10]!;
 inv[5]=a[0]!*a[10]!*a[15]!-a[0]!*a[11]!*a[14]!-a[8]!*a[2]!*a[15]!+a[8]!*a[3]!*a[14]!+a[12]!*a[2]!*a[11]!-a[12]!*a[3]!*a[10]!;
 inv[9]=-a[0]!*a[9]!*a[15]!+a[0]!*a[11]!*a[13]!+a[8]!*a[1]!*a[15]!-a[8]!*a[3]!*a[13]!-a[12]!*a[1]!*a[11]!+a[12]!*a[3]!*a[9]!;
 inv[13]=a[0]!*a[9]!*a[14]!-a[0]!*a[10]!*a[13]!-a[8]!*a[1]!*a[14]!+a[8]!*a[2]!*a[13]!+a[12]!*a[1]!*a[10]!-a[12]!*a[2]!*a[9]!;
 inv[2]=a[1]!*a[6]!*a[15]!-a[1]!*a[7]!*a[14]!-a[5]!*a[2]!*a[15]!+a[5]!*a[3]!*a[14]!+a[13]!*a[2]!*a[7]!-a[13]!*a[3]!*a[6]!;
 inv[6]=-a[0]!*a[6]!*a[15]!+a[0]!*a[7]!*a[14]!+a[4]!*a[2]!*a[15]!-a[4]!*a[3]!*a[14]!-a[12]!*a[2]!*a[7]!+a[12]!*a[3]!*a[6]!;
 inv[10]=a[0]!*a[5]!*a[15]!-a[0]!*a[7]!*a[13]!-a[4]!*a[1]!*a[15]!+a[4]!*a[3]!*a[13]!+a[12]!*a[1]!*a[7]!-a[12]!*a[3]!*a[5]!;
 inv[14]=-a[0]!*a[5]!*a[14]!+a[0]!*a[6]!*a[13]!+a[4]!*a[1]!*a[14]!-a[4]!*a[2]!*a[13]!-a[12]!*a[1]!*a[6]!+a[12]!*a[2]!*a[5]!;
 inv[3]=-a[1]!*a[6]!*a[11]!+a[1]!*a[7]!*a[10]!+a[5]!*a[2]!*a[11]!-a[5]!*a[3]!*a[10]!-a[9]!*a[2]!*a[7]!+a[9]!*a[3]!*a[6]!;
 inv[7]=a[0]!*a[6]!*a[11]!-a[0]!*a[7]!*a[10]!-a[4]!*a[2]!*a[11]!+a[4]!*a[3]!*a[10]!+a[8]!*a[2]!*a[7]!-a[8]!*a[3]!*a[6]!;
 inv[11]=-a[0]!*a[5]!*a[11]!+a[0]!*a[7]!*a[9]!+a[4]!*a[1]!*a[11]!-a[4]!*a[3]!*a[9]!-a[8]!*a[1]!*a[7]!+a[8]!*a[3]!*a[5]!;
 inv[15]=a[0]!*a[5]!*a[10]!-a[0]!*a[6]!*a[9]!-a[4]!*a[1]!*a[10]!+a[4]!*a[2]!*a[9]!+a[8]!*a[1]!*a[6]!-a[8]!*a[2]!*a[5]!;
 const det=a[0]!*inv[0]!+a[1]!*inv[4]!+a[2]!*inv[8]!+a[3]!*inv[12]!;if(!Number.isFinite(det)||Math.abs(det)<1e-30)return;
 for(let i=0;i<16;i++)inv[i]=inv[i]!/det;return inv;
}
const sourceBits=new Float32Array(1),sourceWords=new Uint32Array(sourceBits.buffer);
const f32Bits=(value:number)=>{sourceBits[0]=value;return sourceWords[0]!;};
/** Project each complete posed bound over all EIGHT corners with the field's
 * own clip matrix (the inverse of its worldFromClip) into the field-pixel box
 * and reverse-Z depth range resolveLod measures. A corner behind the eye or
 * outside the depth range makes the bound unusable for that owner (it keeps
 * the incumbent visible read): a clipped bound never supplies a falsely small
 * form. The f64 box is padded by the native f32/raster roundoff, as the source
 * LOD selector does. Several bounds of one owner are joined. Sorted by key. */
export function packFormSourceBounds(bounds:readonly FormSourceBound[],worldFromClip:ArrayLike<number>,width:number,height:number):Uint32Array{
 const clip=bounds.length?invert4(worldFromClip):undefined;const owners=new Map<string,{key:[number,number];box:number[];valid:boolean}>();
 for(const bound of bounds){
  const key=formDepthKeyOf(bound.owner[0],bound.owner[1]);if(key[0]===0)continue;const id=`${key[0]}/${key[1]}`;
  let owner=owners.get(id);if(!owner){owner={key,box:[Infinity,-Infinity,Infinity,-Infinity,-Infinity,Infinity],valid:true};owners.set(id,owner);}
  if(!clip){owner.valid=false;continue;}
  for(let corner=0;corner<8;corner++){
   const x=corner&1?bound.high[0]:bound.low[0],y=corner&2?bound.high[1]:bound.low[1],z=corner&4?bound.high[2]:bound.low[2];
   const w=clip[3]!*x+clip[7]!*y+clip[11]!*z+clip[15]!,depth=clip[2]!*x+clip[6]!*y+clip[10]!*z+clip[14]!;
   if(!(w>1e-5)||!(depth>=0&&depth<=w)){owner.valid=false;break;}
   const px=((clip[0]!*x+clip[4]!*y+clip[8]!*z+clip[12]!)/w*.5+.5)*width,py=(.5-(clip[1]!*x+clip[5]!*y+clip[9]!*z+clip[13]!)/w*.5)*height,d=depth/w;
   const b=owner.box;b[0]=Math.min(b[0]!,py);b[1]=Math.max(b[1]!,py);b[2]=Math.min(b[2]!,px);b[3]=Math.max(b[3]!,px);b[4]=Math.max(b[4]!,d);b[5]=Math.min(b[5]!,d);
  }
 }
 const entries=[...owners.values()].filter(owner=>owner.valid&&owner.box.every(Number.isFinite)).sort((a,b)=>a.key[0]-b.key[0]||a.key[1]-b.key[1]);
 const out=new Uint32Array(4+entries.length*8);out[0]=entries.length;out[3]=FORM_SOURCE_BOUND_MAGIC;
 entries.forEach((owner,i)=>{const b=owner.box,pad=(v:number,sign:number)=>v+sign*(1+1e-5*Math.abs(v)),at=4+i*8;
  out.set([owner.key[0],owner.key[1],f32Bits(pad(b[0]!,-1)),f32Bits(pad(b[1]!,1)),f32Bits(pad(b[2]!,-1)),f32Bits(pad(b[3]!,1)),f32Bits(b[4]!),f32Bits(b[5]!)],at);});
 return out;
}
export const FORM_DEPTH_SHADER=/* wgsl */`
@group(0) @binding(0) var depthField:texture_2d<f32>;
@group(0) @binding(1) var ownerField:texture_2d<u32>;
@group(0) @binding(2) var<storage,read_write> table:array<atomic<u32>>;
struct FormDepthParams {worldFromClip:mat4x4f,scope:vec4u,options:vec4u}
@group(0) @binding(3) var<uniform> depthParams:FormDepthParams;
// Writer-only persistent state. Readers still consume only the copied table
// texture, so this does not consume a relation/run/articulation binding.
@group(0) @binding(4) var<storage,read_write> lodState:array<atomic<u32>>;
// Writer-only parent representation: [groupCount, memberCount, 0, magic],
// then group [offset,count] ranges and exact owner.x/owner.y pairs.
@group(0) @binding(5) var<storage,read> parentGroups:array<u32>;
const PARENT_GROUP_HEADER=4u;const PARENT_GROUP_WORDS=2u;const PARENT_GROUP_MAGIC=0x50475250u;

// (sorted by key) [key.x, key.y, top, bottom, left, right, near, far]: the
// field-pixel box and reverse-Z depth range of its complete current posed
// bound, projected on the host over all eight corners (packFormSourceBounds).
@group(0) @binding(6) var<storage,read> sourceBounds:array<u32>;
const SOURCE_BOUND_HEADER=4u;const SOURCE_BOUND_WORDS=8u;const SOURCE_BOUND_MAGIC=${FORM_SOURCE_BOUND_MAGIC}u;
fn sourceBoundCount()->u32{if(sourceBounds[3u]!=SOURCE_BOUND_MAGIC){return 0u;}return sourceBounds[0];}
fn sourceBoundKeyLess(at:u32,key:vec2u)->bool{let x=sourceBounds[at];return x<key.x||(x==key.x&&sourceBounds[at+1u]<key.y);}
// The owner's source bound entry (word address), 0xffffffff when it has none.
fn sourceBoundOf(key:vec2u)->u32{
 var lo=0u;var hi=sourceBoundCount();
 loop{if(lo>=hi){break;}let mid=(lo+hi)/2u;if(sourceBoundKeyLess(SOURCE_BOUND_HEADER+mid*SOURCE_BOUND_WORDS,key)){lo=mid+1u;}else{hi=mid;}}
 let at=SOURCE_BOUND_HEADER+lo*SOURCE_BOUND_WORDS;
 if(lo<sourceBoundCount()&&sourceBounds[at]==key.x&&sourceBounds[at+1u]==key.y){return at;}
 return 0xffffffffu;
}
${FORM_DEPTH_KEY}
fn raise(i:u32,v:u32){if(atomicLoad(&table[i])<v){atomicMax(&table[i],v);}}
// The owner's slot in the table (claimed if free), or 0xffffffff when its
// probe window holds no slot for it (the LIMIT above).
fn slotOf(key:vec2u)->u32{
 let h=formDepthHash(key);
 for(var probe=0u;probe<${FORM_DEPTH_PROBES}u;probe++){
  let at=((h+probe)&${FORM_DEPTH_SLOTS-1}u)*${FORM_SLOT_WORDS}u;
  // Claim the two key words in order. A weak exchange may fail with the word
  // still empty: four fixed attempts each. A slot another owner holds in
  // either word is passed over; no thread waits on another.
  var x=atomicLoad(&table[at]);
  for(var attempt=0u;attempt<4u&&x==0u;attempt++){let claim=atomicCompareExchangeWeak(&table[at],0u,key.x);x=select(claim.old_value,key.x,claim.exchanged);}
  if(x!=key.x){continue;}
  var y=atomicLoad(&table[at+1u]);
  for(var attempt=0u;attempt<4u&&y==0u;attempt++){let claim=atomicCompareExchangeWeak(&table[at+1u],0u,key.y);y=select(claim.old_value,key.y,claim.exchanged);}
  if(y!=key.y){continue;}
  return at;
 }
 return 0xffffffffu;
}
fn lodHash(scope:vec3u,key:vec2u)->u32{
 var h=(scope.x*0x9e3779b9u)^(scope.y*0x85ebca6bu)^(scope.z*0xc2b2ae35u)^key.x^(key.y*0x27d4eb2fu);
 h=(h^(h>>16u))*0x7feb352du;return h^(h>>15u);
}
// State is open-addressed and compares every key word. A hash collision is
// therefore only extra probes, never another form's held tier.
fn lodStateSlot(scope:vec3u,key:vec2u)->u32{
 let h=lodHash(scope,key);
 for(var probe=0u;probe<${FORM_DEPTH_PROBES}u;probe++){
  let slot=(h+probe)&${FORM_DEPTH_SLOTS-1}u;let at=slot*${FORM_LOD_STATE_WORDS}u;
  let valid=atomicLoad(&lodState[at+5u]);
  if(valid==0u){
   let claim=atomicCompareExchangeWeak(&lodState[at+5u],0u,1u);
   if(claim.exchanged){
    atomicStore(&lodState[at],scope.x);atomicStore(&lodState[at+1u],scope.y);
    atomicStore(&lodState[at+2u],scope.z);atomicStore(&lodState[at+3u],key.x);
    atomicStore(&lodState[at+4u],key.y);atomicStore(&lodState[at+6u],0u);
    return at;
   }
   continue;
  }
  if(atomicLoad(&lodState[at])==scope.x&&atomicLoad(&lodState[at+1u])==scope.y&&
     atomicLoad(&lodState[at+2u])==scope.z&&atomicLoad(&lodState[at+3u])==key.x&&
     atomicLoad(&lodState[at+4u])==key.y){return at;}
 }
 return 0xffffffffu;
}
fn writerOpen()->bool{return atomicLoad(&table[${FORM_HEADER+2}u])>=2u;}
fn writerRecedes()->bool{
 let m=depthParams.worldFromClip;
 return abs(m[0][3])+abs(m[1][3])+abs(m[2][3])>1e-6*abs(m[3][3]);
}
fn writerPlain(rows:f32)->u32{
 if(!(rows>=0.0)){return 0u;}
 return select(select(1u,2u,rows<${FAR_STRIP_BELOW_ROWS}.0),0u,rows>=${NEAR_FROM_ROWS}.0);
}
fn writerHeld(rows:f32,held:u32,hasHeld:bool)->u32{
 if(!hasHeld){return writerPlain(rows);}
 if(held==0u){if(rows>=${FORM_LOD_NEAR_ENTER}){return 0u;}return select(2u,1u,rows>=${FORM_LOD_FAR_ENTER});}
 if(held==1u){if(rows>=${FORM_LOD_NEAR_LEAVE}){return 0u;}return select(2u,1u,rows>=${FORM_LOD_FAR_ENTER});}
 if(rows>=${FORM_LOD_NEAR_LEAVE}){return 0u;}
 return select(2u,1u,rows>=${FORM_LOD_FAR_LEAVE});
}
fn parentSlot(key:vec2u)->u32{
 if(key.x==0u||key.y==0u){return 0xffffffffu;}
 let h=formDepthHash(key);
 for(var probe=0u;probe<${FORM_DEPTH_PROBES}u;probe++){
  let at=((h+probe)&${FORM_DEPTH_SLOTS-1}u)*${FORM_SLOT_WORDS}u;
  let x=atomicLoad(&table[at]);if(x==0u){return 0xffffffffu;}
  if(x==key.x&&atomicLoad(&table[at+1u])==key.y){return at;}
 }
 return 0xffffffffu;
}
fn parentMemberEligible(key:vec2u)->bool{
 let at=parentSlot(key);if(at==0xffffffffu){return false;}
 let flags=depthParams.options.y;let rowPixels=bitcast<f32>(depthParams.options.x);
 if((flags&1u)==0u||rowPixels<=0.0||!writerRecedes()||!writerOpen()){return false;}
 if(atomicLoad(&table[at+2u])==0u){return false;}
 let top=f32(~atomicLoad(&table[at+4u]));let bottom=f32(atomicLoad(&table[at+5u]));
 let left=f32(~atomicLoad(&table[at+6u]));let right=f32(atomicLoad(&table[at+7u]));
 let extent=vec2f(f32(depthParams.options.z),f32(depthParams.options.w));
 if(extent.x<=0.0||extent.y<=0.0||formCut(vec4f(top,bottom,left,right),extent)){return false;}
 let near=bitcast<f32>(atomicLoad(&table[at+2u]));let far=bitcast<f32>(~atomicLoad(&table[at+3u]));
 return !formSpansDepth(near,far,extent,depthParams.worldFromClip);
}

// with a complete source bound holds a slot (claimed here when no pixel of it
// survives the foreground: its visible words stay empty, so formExtent still
// reports it absent) and is marked source-sized (word 14 bit 2).
@compute @workgroup_size(64) fn resolveSource(@builtin(global_invocation_id) id:vec3u){
 if(id.x>=sourceBoundCount()){return;}
 let entry=SOURCE_BOUND_HEADER+id.x*SOURCE_BOUND_WORDS;
 let at=slotOf(vec2u(sourceBounds[entry],sourceBounds[entry+1u]));if(at==0xffffffffu){return;}
 atomicOr(&table[at+14u],4u);

 let region=${FORM_SOURCE_ROW*FORM_DEPTH_COLUMNS*4}u+(at/${FORM_SLOT_WORDS}u)*${FORM_SOURCE_TEXELS*4}u;
 for(var k=0u;k<6u;k++){atomicStore(&table[region+k],sourceBounds[entry+2u+k]);}
 atomicStore(&table[region+6u],1u);
}
// One invocation per reduced form slot. It resolves the held level before the
// table is copied to the reader texture; no readback or frame search occurs.
@compute @workgroup_size(256) fn resolveLod(@builtin(global_invocation_id) id:vec3u){
 if(id.x>=${FORM_DEPTH_SLOTS}u){return;}
 if(id.x==0u){
  let rowPixels=bitcast<f32>(depthParams.options.x);let flags=depthParams.options.y;
  atomicStore(&table[${FORM_HEADER+3}u],select(0u,1u,(flags&1u)!=0u&&rowPixels>0.0&&writerRecedes()&&writerOpen()));
 }
 let at=id.x*${FORM_SLOT_WORDS}u;let key=vec2u(atomicLoad(&table[at]),atomicLoad(&table[at+1u]));
 if(key.x==0u||key.y==0u){return;}
 // A disabled scope, an orthographic view, an enclosed view, a cut/spanning
// form keeps level 0 (the existing full producer). Producer/class eligibility
// is supplied by readers; compact physical producer owners may opt in there.
 let flags=depthParams.options.y;let rowPixels=bitcast<f32>(depthParams.options.x);
 var level=0u;
  if((flags&1u)!=0u&&rowPixels>0.0&&writerRecedes()&&writerOpen()){
  var top=f32(~atomicLoad(&table[at+4u]));var bottom=f32(atomicLoad(&table[at+5u]));
  var left=f32(~atomicLoad(&table[at+6u]));var right=f32(atomicLoad(&table[at+7u]));
  let extent=vec2f(f32(depthParams.options.z),f32(depthParams.options.w));
  var near=bitcast<f32>(atomicLoad(&table[at+2u]));
  var far=bitcast<f32>(~atomicLoad(&table[at+3u]));

  // source bound, never by the pixels that survive the foreground; the same
  // camera/world/epoch history holds its level.
  let source=sourceBoundOf(key);
  if(source!=0xffffffffu){
   top=bitcast<f32>(sourceBounds[source+2u]);bottom=bitcast<f32>(sourceBounds[source+3u]);
   left=bitcast<f32>(sourceBounds[source+4u]);right=bitcast<f32>(sourceBounds[source+5u]);
   near=bitcast<f32>(sourceBounds[source+6u]);far=bitcast<f32>(sourceBounds[source+7u]);
  }
  let box=vec4f(top,bottom,left,right);
  let compact=extent.x>0.0&&extent.y>0.0&&!formCut(box,extent)&&!formSpansDepth(near,far,extent,depthParams.worldFromClip);
  if(source!=0xffffffffu&&!compact){atomicOr(&table[at+14u],8u);}
  if(compact){
   let rows=(bottom-top)/rowPixels;let scope=depthParams.scope.xyz;let stateAt=lodStateSlot(scope,key);
   var held=0u;var hasHeld=false;
   if(stateAt!=0xffffffffu){let encoded=atomicLoad(&lodState[stateAt+6u]);if(encoded>=1u&&encoded<=3u){held=encoded-1u;hasHeld=true;}}
   level=writerHeld(rows,held,hasHeld);
   if(stateAt!=0xffffffffu){atomicStore(&lodState[stateAt+6u],level+1u);}
  }
 }
 atomicStore(&table[at+${FORM_LOD_WORD}u],level);
}
// Physical producer parents are all-or-none: missing/cut/spanning members keep
// the incumbent. Entity source-selected groups (count bit31) only annotate
// the frontier already selected from complete posed bounds by their adapter.
@compute @workgroup_size(64) fn resolveParents(@builtin(global_invocation_id) id:vec3u){
 let groups=parentGroups[0];if(parentGroups[3u]!=PARENT_GROUP_MAGIC||id.x>=groups){return;}
 let range=PARENT_GROUP_HEADER+id.x*PARENT_GROUP_WORDS;let offset=parentGroups[range];let taggedCount=parentGroups[range+1u];
 let count=taggedCount&0x7fffffffu;let sourceSelected=(taggedCount&0x80000000u)!=0u;
 if(count==0u||offset+count>parentGroups[1u]){return;}
 // Entity adapters have already removed these fine members using complete
 // posed source bounds. Visibility annotates, never reverses that frontier.
 // Physical producer groups keep the all-or-none field certification.
 if(sourceSelected){if((depthParams.options.y&1u)==0u||!writerRecedes()){return;}}
 for(var i=0u;i<count&&!sourceSelected;i++){
  let member=PARENT_GROUP_HEADER+groups*PARENT_GROUP_WORDS+(offset+i)*2u;
  let owner=vec2u(parentGroups[member],parentGroups[member+1u]);
  if(!parentMemberEligible(formDepthKey(owner))){return;}
 }
 for(var i=0u;i<count;i++){
  let member=PARENT_GROUP_HEADER+groups*PARENT_GROUP_WORDS+(offset+i)*2u;
  let at=parentSlot(formDepthKey(vec2u(parentGroups[member],parentGroups[member+1u])));
  if(at!=0xffffffffu){atomicOr(&table[at+14u],2u);}
 }
}
// TILE REDUCTION (frame cost, 2026-11-12). One thread per field pixel did a
// global probe and up to 13 atomic max operations on the one slot every pixel

// owners: its pixels first raise a slot of a 32-slot table in workgroup memory
// (same words, same max), then one thread per tile slot raises the global
// slot once per word. A max of maxima is the max: every owner's words are the
// ones the per-pixel reduce made. A pixel whose tile table has no slot for its
// owner raises the global table itself, as before.
const TILE_SLOTS=32u;
var<workgroup> tileKeys:array<atomic<u32>,64>;
var<workgroup> tileWords:array<atomic<u32>,416>;
@compute @workgroup_size(16,16) fn reduce(@builtin(global_invocation_id) id:vec3u,@builtin(local_invocation_index) li:u32){
 for(var i=li;i<64u;i+=256u){atomicStore(&tileKeys[i],0u);}
 for(var i=li;i<416u;i+=256u){atomicStore(&tileWords[i],0u);}
 workgroupBarrier();
 let size=textureDimensions(depthField);
 if(id.x<size.x&&id.y<size.y){pixel(id.xy,size);}
 workgroupBarrier();
 if(li<TILE_SLOTS){
  let key=vec2u(atomicLoad(&tileKeys[li*2u]),atomicLoad(&tileKeys[li*2u+1u]));
  if(key.x!=0u&&key.y!=0u){
   let at=slotOf(key);
   if(at!=0xffffffffu){for(var k=0u;k<13u;k++){let v=atomicLoad(&tileWords[li*13u+k]);if(v!=0u){raise(at+2u+k,v);}}}
  }
 }
}
fn pixel(xy:vec2u,size:vec2u){
 let q=vec2i(xy);let d=textureLoad(depthField,q,0).x;let owner=textureLoad(ownerField,q,0).xy;
 // OPEN BACKGROUND (formOpen): on an 8 px grid, a background pixel whose four
 // neighbours 4 px away are background too, so a patch of sky at least 9 px
 // across, never a one- or two-pixel crack between meshes.
 if(!(d>0.0)){
  if((xy.x&7u)==0u&&(xy.y&7u)==0u){
   let hi=vec2i(size)-1;
   let around=max(max(textureLoad(depthField,clamp(q+vec2i(4,0),vec2i(0),hi),0).x,textureLoad(depthField,clamp(q-vec2i(4,0),vec2i(0),hi),0).x),
    max(textureLoad(depthField,clamp(q+vec2i(0,4),vec2i(0),hi),0).x,textureLoad(depthField,clamp(q-vec2i(0,4),vec2i(0),hi),0).x));
   if(!(around>0.0)){atomicAdd(&table[${FORM_HEADER+2}u],1u);}
  }
  return;
 }
 if((owner.x&0xdfffffffu)==0u){return;}
 let key=formDepthKey(owner);
 let w=depthParams.worldFromClip*vec4f((vec2f(xy)+.5)/vec2f(size)*vec2f(2,-2)+vec2f(-1,1),d,1);let world=w.xyz/w.w;
 // Words 2..14 of the slot: depth near/far, rows, columns, world box, and
 // word 14, UNDER THE SKY: open background two pixels above (it runs up to a
 // horizon; a room's floor ends at a wall).
 var v:array<u32,13>;
 v[0]=bitcast<u32>(d);v[1]=~bitcast<u32>(d);v[2]=~xy.y;v[3]=xy.y+1u;v[4]=~xy.x;v[5]=xy.x+1u;
 for(var axis=0u;axis<3u;axis++){let o=formOrdered(world[axis]);v[6u+axis*2u]=~o;v[7u+axis*2u]=o;}
 v[12]=select(0u,1u,xy.y>=2u&&!(textureLoad(depthField,q-vec2i(0,2),0).x>0.0));
 let h=formDepthHash(key);
 for(var probe=0u;probe<TILE_SLOTS;probe++){
  let s=(h+probe)%TILE_SLOTS;
  var x=atomicLoad(&tileKeys[s*2u]);
  for(var attempt=0u;attempt<4u&&x==0u;attempt++){let claim=atomicCompareExchangeWeak(&tileKeys[s*2u],0u,key.x);x=select(claim.old_value,key.x,claim.exchanged);}
  if(x!=key.x){continue;}
  var y=atomicLoad(&tileKeys[s*2u+1u]);
  for(var attempt=0u;attempt<4u&&y==0u;attempt++){let claim=atomicCompareExchangeWeak(&tileKeys[s*2u+1u],0u,key.y);y=select(claim.old_value,key.y,claim.exchanged);}
  if(y!=key.y){continue;}
  for(var k=0u;k<13u;k++){if(v[k]!=0u){atomicMax(&tileWords[s*13u+k],v[k]);}}
  return;
 }
 let at=slotOf(key);if(at==0xffffffffu){return;}
 for(var k=0u;k<13u;k++){if(v[k]!=0u){raise(at+2u+k,v[k]);}}
}
`;
interface FormLodHistory {readonly key:string;readonly cameraId:number;readonly worldId:number;epoch:number;state:GPUBuffer;refs:number}
export interface FormParentGroup {
 readonly members:readonly (readonly [number,number])[];
 /** Entity adapter already selected from complete posed bounds and removed
  * its fine members. Not a producer-row opt-in. */
 readonly sourceSelected?:boolean;
}
interface FormDepthTable {
 buffer:GPUBuffer;params:GPUBuffer;texture:GPUTexture;view:GPUTextureView;group?:GPUBindGroup;history?:FormLodHistory;
 parentBuffer:GPUBuffer;parentBytes:number;retiredParents:GPUBuffer[];boundParent?:GPUBuffer;
 sourceBuffer:GPUBuffer;sourceBytes:number;boundSource?:GPUBuffer;
}
const PARENT_GROUP_MAGIC=0x50475250;
function parentWord(value:number,label:string):number{
 if(!Number.isSafeInteger(value)||value<0||value>0xffffffff)throw new RangeError(`${label} must be a uint32`);
 return value;
}
export function packFormParentGroups(groups:readonly FormParentGroup[]):Uint32Array{
 const owners=new Set<string>();let members=0;
 for(const [gi,group]of groups.entries()){
  if(!Array.isArray(group.members)||!group.members.length)throw new RangeError(`parent group ${gi} must have members`);
  if(group.members.length>0x7fffffff)throw new RangeError('parent group member count exceeds tagged range');
  members+=group.members.length;if(members>0xffffffff)throw new RangeError('parent group member count exceeds uint32');
  for(const [mi,owner]of group.members.entries()){
   if(!Array.isArray(owner)||owner.length!==2)throw new RangeError(`parent group ${gi} member ${mi} must be an owner pair`);
   const x=parentWord(owner[0],`parent group ${gi} owner.x`),y=parentWord(owner[1],`parent group ${gi} owner.y`),key=`${x}/${y}`;
   if(group.sourceSelected&&(x&0x80000000)!==0)throw new RangeError('source-selected groups require entity owners');
   if(owners.has(key))throw new Error(`parent owner appears in more than one group: ${key}`);owners.add(key);
  }
 }
 const words=4+groups.length*2+members*2,result=new Uint32Array(words);result[0]=parentWord(groups.length,'parent group count');result[1]=parentWord(members,'parent member count');result[3]=PARENT_GROUP_MAGIC;
 let offset=0,at=4+groups.length*2;
 for(let gi=0;gi<groups.length;gi++){
  const group=groups[gi]!;result[4+gi*2]=offset;result[5+gi*2]=(group.members.length|(group.sourceSelected?0x80000000:0))>>>0;
  for(const owner of group.members){result[at++]=parentWord(owner[0],'parent owner.x');result[at++]=parentWord(owner[1],'parent owner.y');}offset+=group.members.length;
 }
 return result;
}

const formPrograms=new DeviceComputePrograms(device=>{
 const layout=device.createBindGroupLayout({label:'MAIN form table',entries:[
   {binding:0,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:'unfilterable-float'}},
   {binding:1,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:'uint'}},
   {binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:'storage'}},
   {binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:'uniform'}},
   {binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:'storage'}},
   {binding:5,visibility:GPUShaderStage.COMPUTE,buffer:{type:'read-only-storage'}},
   {binding:6,visibility:GPUShaderStage.COMPUTE,buffer:{type:'read-only-storage'}}]});
 const pipeline=device.createPipelineLayout({bindGroupLayouts:[layout]});
 return {resources:layout,plans:new Map([
  ['reduce',()=>{const module=device.createShaderModule({label:'MAIN form table',code:isolateComputeEntry(FORM_DEPTH_SHADER,'reduce')});
   return {label:'MAIN form table reduce',layout:pipeline,compute:{module,entryPoint:'reduce'}};}],
  ['resolveSource',()=>{const module=device.createShaderModule({label:'MAIN form source bounds',code:isolateComputeEntry(FORM_DEPTH_SHADER,'resolveSource')});
   return {label:'MAIN form source bounds',layout:pipeline,compute:{module,entryPoint:'resolveSource'}};}],
  ['resolveLod',()=>{const module=device.createShaderModule({label:'MAIN form held LOD',code:isolateComputeEntry(FORM_DEPTH_SHADER,'resolveLod')});
   return {label:'MAIN form held LOD',layout:pipeline,compute:{module,entryPoint:'resolveLod'}}; }],
  ['resolveParents',()=>{const module=device.createShaderModule({label:'MAIN form parent groups',code:isolateComputeEntry(FORM_DEPTH_SHADER,'resolveParents')});
   return {label:'MAIN form parent groups',layout:pipeline,compute:{module,entryPoint:'resolveParents'}};}]
 ])};
});
export async function prepareFormDepthPrograms(device:GPUDevice,progress?:ComputePipelineProgress):Promise<void>{await formPrograms.prepare(device,progress);}

export class FormDepths {
 private readonly reduce:GPUComputePipeline;
 private readonly resolveLod:GPUComputePipeline;
 private readonly resolveParents:GPUComputePipeline;
 private readonly resolveSource:GPUComputePipeline;
 private readonly layout:GPUBindGroupLayout;
 // Keyed by the field's channel set: the table is reused for as long as those
 // textures are, across frames, and never shared between two live fields.
 private readonly tables=new WeakMap<readonly GPUTexture[],FormDepthTable>();
 private readonly histories=new Map<string,FormLodHistory>();
 constructor(private readonly device:GPUDevice){
  const ready=formPrograms.get(device);this.layout=ready.resources;this.reduce=ready.pipelines.get('reduce')!;this.resolveLod=ready.pipelines.get('resolveLod')!;this.resolveParents=ready.pipelines.get('resolveParents')!;
  this.resolveSource=ready.pipelines.get('resolveSource')!;
 }
 private historyFor(scope:FormLodScope,encoder:GPUCommandEncoder):FormLodHistory{
  const key=formLodScopeKey(scope);let history=this.histories.get(key);
  if(!history){
   const state=this.device.createBuffer({label:`MAIN form held LOD state ${key}`,size:FORM_LOD_STATE_BYTES,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
   history={key,cameraId:scope.cameraId,worldId:scope.worldId,epoch:scope.epoch,state,refs:0};this.histories.set(key,history);encoder.clearBuffer(state);
  }else if(history.epoch!==scope.epoch){
   if(scope.epoch<history.epoch){throw Error(`Form LOD scope epoch regressed for ${key}`);}
   // A source/owner index may now name a new lifetime. Reuse the same
   // camera/world history resource, but clear it once before this epoch's
   // first field reduction. All retained field rings share this resource.
   history.epoch=scope.epoch;encoder.clearBuffer(history.state);
  }
  return history;
 }
 private unrefHistory(history:FormLodHistory):void{
  history.refs--;
  if(history.refs===0&&this.histories.get(history.key)===history){history.state.destroy();this.histories.delete(history.key);}
 }
 private bind(table:FormDepthTable,field:SurfaceField,history:FormLodHistory):void{
  if(table.history===history&&table.boundParent===table.parentBuffer&&table.boundSource===table.sourceBuffer)return;
  if(table.history!==history){if(table.history)this.unrefHistory(table.history);table.history=history;history.refs++;}
  table.group=this.device.createBindGroup({layout:this.layout,entries:[
   {binding:0,resource:field.textures[0]!.createView()},{binding:1,resource:field.textures[3]!.createView()},
   {binding:2,resource:{buffer:table.buffer}},{binding:3,resource:{buffer:table.params}},{binding:4,resource:{buffer:history.state}},{binding:5,resource:{buffer:table.parentBuffer}},
   {binding:6,resource:{buffer:table.sourceBuffer}},
  ]});
  table.boundParent=table.parentBuffer;table.boundSource=table.sourceBuffer;
 }





 encode(encoder:GPUCommandEncoder,field:SurfaceField,scope?:FormLodScope,parentGroups:readonly FormParentGroup[]=[],sourceBounds:readonly FormSourceBound[]=[]):void{
  const active=validateFormLodScope(scope??conservativeFormLodScope());
  const packedParents=packFormParentGroups(parentGroups);
  const packedSource=packFormSourceBounds(sourceBounds,field.worldFromClip,field.width,field.height);
  let table=this.tables.get(field.textures);
  const rows=FORM_DEPTH_ROWS;
  if(!table){
   const buffer=this.device.createBuffer({label:'MAIN form table slots',size:rows*FORM_DEPTH_COLUMNS*16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
   const params=this.device.createBuffer({label:'MAIN form table matrix and LOD scope',size:96,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
   const texture=this.device.createTexture({label:'MAIN form table',size:[FORM_DEPTH_COLUMNS,rows],format:'rgba32uint',usage:GPUTextureUsage.COPY_DST|GPUTextureUsage.TEXTURE_BINDING});
   const parentBuffer=this.device.createBuffer({label:'MAIN form parent groups',size:Math.max(16,packedParents.byteLength),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
   const sourceBuffer=this.device.createBuffer({label:'MAIN form source bounds',size:Math.max(16,packedSource.byteLength),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
   table={buffer,params,texture,view:texture.createView(),parentBuffer,parentBytes:parentBuffer.size,retiredParents:[],sourceBuffer,sourceBytes:sourceBuffer.size};this.tables.set(field.textures,table);
  }
  if(packedSource.byteLength>table.sourceBytes){
   const next=this.device.createBuffer({label:'MAIN form source bounds',size:packedSource.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
   table.retiredParents.push(table.sourceBuffer);table.sourceBuffer=next;table.sourceBytes=next.size;table.group=undefined;table.boundSource=undefined;
  }
  this.device.queue.writeBuffer(table.sourceBuffer,0,packedSource);
  if(packedParents.byteLength>table.parentBytes){
   const next=this.device.createBuffer({label:'MAIN form parent groups',size:packedParents.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
   table.retiredParents.push(table.parentBuffer);table.parentBuffer=next;table.parentBytes=next.size;table.group=undefined;table.boundParent=undefined;
  }
  this.device.queue.writeBuffer(table.parentBuffer,0,packedParents);
  const history=this.historyFor(active,encoder);this.bind(table,field,history);
  const paramBytes=new ArrayBuffer(96);new Float32Array(paramBytes,0,16).set(field.worldFromClip.subarray(0,16));
  const words=new Uint32Array(paramBytes);words[16]=active.cameraId>>>0;words[17]=active.worldId>>>0;words[18]=active.epoch>>>0;words[19]=0;
  words[21]=active.allowCompact===true?1:0;words[22]=field.width;words[23]=field.height;
  new DataView(paramBytes).setFloat32(80,Number.isFinite(active.rowPixels)?active.rowPixels:0,true);
  this.device.queue.writeBuffer(table.params,0,paramBytes);
  encoder.clearBuffer(table.buffer);
  const pass=encoder.beginComputePass({label:'MAIN form table'});pass.setBindGroup(0,table.group!);
  pass.setPipeline(this.reduce);pass.dispatchWorkgroups(Math.ceil(field.width/16),Math.ceil(field.height/16));
  if(packedSource[0]!>0){pass.setPipeline(this.resolveSource);pass.dispatchWorkgroups(Math.ceil(packedSource[0]!/64));}
  pass.setPipeline(this.resolveLod);pass.dispatchWorkgroups(Math.ceil(FORM_DEPTH_SLOTS/256));
  if(packedParents[0]!>0){pass.setPipeline(this.resolveParents);pass.dispatchWorkgroups(Math.ceil(packedParents[0]!/64));}
  pass.end();
  encoder.copyBufferToTexture({buffer:table.buffer,bytesPerRow:FORM_DEPTH_COLUMNS*16,rowsPerImage:rows},{texture:table.texture},[FORM_DEPTH_COLUMNS,rows]);
 }
 /** The table every reader of this field binds. A field that was never
  * reduced has no table: that is an encode-order fault, never a default. */
 view(field:SurfaceField):GPUTextureView{
  const table=this.tables.get(field.textures);if(!table)throw Error('Surface field has no form table');
  return table.view;
 }
 /** Same-pose inverse used by the table's writer. Consumers must not
  * substitute another camera's matrix when selecting source work. */
 matrix(field:SurfaceField):GPUBuffer{
  const table=this.tables.get(field.textures);if(!table)throw Error('Surface field has no form table');
  return table.params;
 }
 release(textures:readonly GPUTexture[]):void{
  const table=this.tables.get(textures);if(!table)return;
  table.buffer.destroy();table.params.destroy();table.texture.destroy();
  table.parentBuffer.destroy();table.sourceBuffer.destroy();for(const buffer of table.retiredParents)buffer.destroy();
  const history=table.history;if(history){this.unrefHistory(history);}
  this.tables.delete(textures);
 }
}
const formDepthOwners=new WeakMap<GPUDevice,FormDepths>();
export function formDepths(device:GPUDevice):FormDepths{let owner=formDepthOwners.get(device);if(!owner){owner=new FormDepths(device);formDepthOwners.set(device,owner);}return owner;}
