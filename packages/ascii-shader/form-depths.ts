import {DeviceComputePrograms} from './gpu-pipeline-preparation.ts';
import {isolateComputeEntry} from './gpu-compute-program.ts';
import type {SurfaceField} from './surface-fields.ts';
import {FORM_LOD_STATE_WORDS,conservativeFormLodScope,formLodScopeKey,validateFormLodScope} from './form-lod-state.ts';
import type {FormLodScope} from './form-lod-state.ts';

/** THE ONE PER-FORM TABLE of a field (a form = one source owner, the owner
 * field's id pair without the UV bit), the input of the distance tier
 * (gpu-articulation.ts "THE TIER IS THE ROWS THE DRAWING HAS").
 *
 * Owner: the surface-field stage, because the table is a reduction of the
 * field's own owner and depth channels and must live exactly as long as they
 * do: a retained same-pose field keeps its table, a cleared field gets a new
 * one. FormDepths.encode runs ONCE per field, right after the field's last
 * draw and before the relation, underdrawing, tone and articulation passes.
 * Every reader binds view(field) as a texture (read-only by construction) and
 * reads it through formExtent and formReference (FORM_DEPTH_WGSL); the
 * distance tier is distanceTierOf (gpu-articulation.ts DISTANCE_TIER_WGSL)
 * for every reader.
 *
 * Per owner, from ONE pass over its visible field pixels (reduce): its
 * nearest and farthest visible depth, its top and bottom field row (the page
 * rows it occupies), its page columns and the world box of its visible
 * surface. A compact form is an owner that does not span depth
 * (formSpansDepth) and is not cut by an edge of the field (formCut); its
 * level is its page rows. No scene reference length is derived (the
 * reference length L and its vote were deleted 2026-11-10 on operator
 * order: a world line across the ground was a guess).
 *
 * Layout: FORM_DEPTH_SLOTS slots of FORM_SLOT_WORDS u32 (four texels) in an
 * open-addressed hash, as a storage buffer for the one writer and copied to
 * an rgba32uint texture of FORM_DEPTH_COLUMNS columns for the readers (texel
 * t of the buffer is texel (t % columns, t / columns)). Slot = [key.x, key.y,
 * bits of the nearest visible depth, ~bits of the farthest] [~top field row,
 * bottom field row + 1, ~left field column, right field column + 1]
 * [~ordered lowest world x, ordered highest world x, the same for y]
 * [the same for z, 0, held LOD level]: every word but the keys is an atomic max, so a
 * cleared buffer is the empty table. key.x = owner.x & 0xdfffffff with
 * owner.y's top bit in bit 29; key.y = owner.y << 1 | 1, so both words of a
 * used slot are nonzero and the 64-bit owner is compared exactly. Depth is
 * reverse Z and positive: larger is nearer and the bit pattern orders as the
 * value. A slot is found within FORM_DEPTH_PROBES linear probes of the hash.
 * The texture row FORM_DEPTH_HEADER_ROW holds [bits of L, vote weight,
 * open background patches, 0] in its first texel and the vote histogram
 * after it.
 *
 * LIMIT, stated: an owner that finds no slot (more visible owners than the
 * table holds in one probe window) is read like an owner that spans depth:
 * at the point each reader asks about, by L. */
export const FORM_DEPTH_SLOTS=65536,FORM_DEPTH_COLUMNS=256,FORM_DEPTH_PROBES=32,FORM_SLOT_WORDS=16;
export const FORM_DEPTH_HEADER_ROW=FORM_DEPTH_SLOTS*FORM_SLOT_WORDS/4/FORM_DEPTH_COLUMNS;
/** The table's final word is written after reduction, never by pixel reduce. */
export const FORM_LOD_WORD=15;
/** Writer-only state; readers still bind only the existing form texture. */
export const FORM_LOD_STATE_BYTES=FORM_DEPTH_SLOTS*FORM_LOD_STATE_WORDS*4;
/** Rows a form occupies from which it is 近景 near: the practitioner's smallest
 * scale is the 7-row drawing, which has no tone, no gaps and no interior
 * detail (GOAL-EXECUTOR practitioner rules, "The smallest tier changes
 * vocabulary"); the operator's two near-tree constructions are 8 and 13 rows.
 * The run pass's compiled part size band uses the same 119 px. */
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
  let top=f32(~atomicLoad(&table[at+4u]));let bottom=f32(atomicLoad(&table[at+5u]));
  let left=f32(~atomicLoad(&table[at+6u]));let right=f32(atomicLoad(&table[at+7u]));
  let extent=vec2f(f32(depthParams.options.z),f32(depthParams.options.w));
  let box=vec4f(top,bottom,left,right);
  let near=bitcast<f32>(atomicLoad(&table[at+2u]));
  let far=bitcast<f32>(~atomicLoad(&table[at+3u]));
  if(extent.x>0.0&&extent.y>0.0&&!formCut(box,extent)&&!formSpansDepth(near,far,extent,depthParams.worldFromClip)){
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
// of its owner shares (16.6 ms at 3488x2252, Codex). A 16x16 tile holds a few
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
   {binding:5,visibility:GPUShaderStage.COMPUTE,buffer:{type:'read-only-storage'}}]});
 const pipeline=device.createPipelineLayout({bindGroupLayouts:[layout]});
 return {resources:layout,plans:new Map([
  ['reduce',()=>{const module=device.createShaderModule({label:'MAIN form table',code:isolateComputeEntry(FORM_DEPTH_SHADER,'reduce')});
   return {label:'MAIN form table reduce',layout:pipeline,compute:{module,entryPoint:'reduce'}};}],
  ['resolveLod',()=>{const module=device.createShaderModule({label:'MAIN form held LOD',code:isolateComputeEntry(FORM_DEPTH_SHADER,'resolveLod')});
   return {label:'MAIN form held LOD',layout:pipeline,compute:{module,entryPoint:'resolveLod'}}; }],
  ['resolveParents',()=>{const module=device.createShaderModule({label:'MAIN form parent groups',code:isolateComputeEntry(FORM_DEPTH_SHADER,'resolveParents')});
   return {label:'MAIN form parent groups',layout:pipeline,compute:{module,entryPoint:'resolveParents'}};}]
 ])};
});
export async function prepareFormDepthPrograms(device:GPUDevice,progress?:(fraction:number)=>void):Promise<void>{await formPrograms.prepare(device,progress);}

export class FormDepths {
 private readonly reduce:GPUComputePipeline;
 private readonly resolveLod:GPUComputePipeline;
 private readonly resolveParents:GPUComputePipeline;
 private readonly layout:GPUBindGroupLayout;
 // Keyed by the field's channel set: the table is reused for as long as those
 // textures are, across frames, and never shared between two live fields.
 private readonly tables=new WeakMap<readonly GPUTexture[],FormDepthTable>();
 private readonly histories=new Map<string,FormLodHistory>();
 constructor(private readonly device:GPUDevice){
  const ready=formPrograms.get(device);this.layout=ready.resources;this.reduce=ready.pipelines.get('reduce')!;this.resolveLod=ready.pipelines.get('resolveLod')!;this.resolveParents=ready.pipelines.get('resolveParents')!;
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
  if(table.history===history&&table.boundParent===table.parentBuffer)return;
  if(table.history!==history){if(table.history)this.unrefHistory(table.history);table.history=history;history.refs++;}
  table.group=this.device.createBindGroup({layout:this.layout,entries:[
   {binding:0,resource:field.textures[0]!.createView()},{binding:1,resource:field.textures[3]!.createView()},
   {binding:2,resource:{buffer:table.buffer}},{binding:3,resource:{buffer:table.params}},{binding:4,resource:{buffer:history.state}},{binding:5,resource:{buffer:table.parentBuffer}},
  ]});
  table.boundParent=table.parentBuffer;
 }
 /** Reduce this field's owner and depth channels into its table, then the
  * reference length. Call once, after the field's last draw and before any
  * reader's pass. */
 encode(encoder:GPUCommandEncoder,field:SurfaceField,scope?:FormLodScope,parentGroups:readonly FormParentGroup[]=[]):void{
  const active=validateFormLodScope(scope??conservativeFormLodScope());
  const packedParents=packFormParentGroups(parentGroups);
  let table=this.tables.get(field.textures);
  const rows=FORM_DEPTH_HEADER_ROW+1;
  if(!table){
   const buffer=this.device.createBuffer({label:'MAIN form table slots',size:rows*FORM_DEPTH_COLUMNS*16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
   const params=this.device.createBuffer({label:'MAIN form table matrix and LOD scope',size:96,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
   const texture=this.device.createTexture({label:'MAIN form table',size:[FORM_DEPTH_COLUMNS,rows],format:'rgba32uint',usage:GPUTextureUsage.COPY_DST|GPUTextureUsage.TEXTURE_BINDING});
   const parentBuffer=this.device.createBuffer({label:'MAIN form parent groups',size:Math.max(16,packedParents.byteLength),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
   table={buffer,params,texture,view:texture.createView(),parentBuffer,parentBytes:parentBuffer.size,retiredParents:[]};this.tables.set(field.textures,table);
  }
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
  table.parentBuffer.destroy();for(const buffer of table.retiredParents)buffer.destroy();
  const history=table.history;if(history){this.unrefHistory(history);}
  this.tables.delete(textures);
 }
}
const formDepthOwners=new WeakMap<GPUDevice,FormDepths>();
export function formDepths(device:GPUDevice):FormDepths{let owner=formDepthOwners.get(device);if(!owner){owner=new FormDepths(device);formDepthOwners.set(device,owner);}return owner;}
