import type {SurfaceField} from './surface-fields.ts';
import {PARENT_MAX_COLUMNS,SOURCE_COARSE_CONTOUR_FLAG,SOURCE_CONTACT_BOUNDARY_FLAG,SOURCE_CONTACT_LINE_CLASS,sourceWholeConstructionRenderable,type FarFormInstance,type SourceWholeConstruction,type SourceWholeOpening,type SourceWholePartRef,type SourceWholePartRole} from './source-lod.ts';
import {DeviceComputePrograms,type ComputePipelineProgress} from './gpu-pipeline-preparation.ts';
import {CONSTRUCTION_POINT_WGSL} from './source-coordinates.ts';
import {FORM_DEPTH_WGSL,LOD_TIER_HOLD,formDepths} from './form-depths.ts';
import {inverseMatrix} from './composition.ts';
import type {SourceSmallParent} from './source-small-parent.ts';
import {packSmallConstructionKindAndEnds} from './role-grammar-format.ts';
import {withCoherentSourceParts} from './coherent-parent-gpu.ts';
import {coherentParentProductOf} from './coherent-parent-union.ts';
import {coherentCorrespondenceEntries,encodeCoherentCorrespondence,type CoherentCorrespondencePart} from './coherent-parent-correspondence.ts';
























export interface SourceFarAggregate {readonly key:number;readonly box:readonly [number,number,number,number];readonly members:readonly (readonly [number,number])[];readonly construction?:SourceWholeConstruction}
export interface SourceFarProducers {readonly domains:readonly number[];readonly perspective:boolean}
export interface SourceFarForms {readonly domain:number;readonly instances:readonly FarFormInstance[];readonly aggregates?:readonly SourceFarAggregate[];readonly producers?:SourceFarProducers}
export const COMPILED_FAR_PROFILE_BIT=0x80000000;
/** Page columns a parent box under PARENT_MAX_COLUMNS columns can meet (it straddles column edges). */
const PARENT_PROFILE_COLUMNS=PARENT_MAX_COLUMNS+2;
/** The producer-row member test, shared by the relation stage (surface-relations
 * ownerMerged: its relations are not produced) and this emitter (its pixels join
 * the aggregate), so a row is in exactly one of them. A producer owner (owner.x
 * bit 31) whose visible form fits one native cell with the hold (the entity rule
 * fitsNativeCell, in field pixels: `row` field pixels per text row, a column
 * 16/17 of a row), not cut by the field edge, in an open view. Requires
 * FORM_DEPTH_WGSL. */
export const FAR_AGGREGATE_MEMBER_WGSL=/* wgsl */`
fn farAggregateMember(owner:vec2u,form:FormExtent,extent:vec2f,row:f32)->bool{
 if(formParentSelected(owner)){return false;}

 // member by THAT extent (the form-table writer's), so a foreground cut that
 // leaves a few pixels can neither merge a large form into the aggregate nor
 // split a small one out of it. No bound: the visible extent, as before.
 var sized=form;let sourceExtent=formSourceExtent(owner);if(sourceExtent.found){sized=sourceExtent;}
 if((owner.x&0x80000000u)==0u||!sized.found||!(row>0.0)||!formOpen()||formCut(sized.box,extent)){return false;}
 return sized.box.y-sized.box.x<row*${LOD_TIER_HOLD}&&sized.box.w-sized.box.z<row*${16*LOD_TIER_HOLD/17};
}
`;
export const SOURCE_FAR_FORMS_SHADER=/* wgsl */`
struct Params {extent:vec4u,rect:vec4f,field:vec4f,clipFromWorld:mat4x4f}
struct Input {identity:vec2u,members:vec2u,box:vec4f,mode:vec4u,program:vec4u}
struct Run {range:vec4u,identity:vec4u,part:vec4u,paper:vec4u,correspondence:vec4u}
${CONSTRUCTION_POINT_WGSL}
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read> inputs:array<Input>;
@group(0) @binding(2) var ownerField:texture_2d<u32>;
@group(0) @binding(3) var depthField:texture_2d<f32>;
@group(0) @binding(4) var<storage,read_write> status:array<atomic<u32>>;
@group(0) @binding(5) var<storage,read_write> runs:array<Run>;
@group(0) @binding(6) var<storage,read_write> points:array<Point>;
@group(0) @binding(7) var formDepths:texture_2d<u32>;
@group(0) @binding(8) var<storage,read> memberKeys:array<vec2u>;
${FORM_DEPTH_WGSL}
${FAR_AGGREGATE_MEMBER_WGSL}
fn keyLess(a:vec2u,b:vec2u)->bool{return a.x<b.x||(a.x==b.x&&a.y<b.y);}
fn programWord(input:Input,word:u32)->u32{
 let at=input.program.x+word;let pair=memberKeys[at/2u];return select(pair.x,pair.y,(at&1u)!=0u);
}
fn programPartRole(value:u32)->u32{return value&3u;}
// An entity aggregate's member: its sorted (domain, eid) list, one lookup.
fn listed(input:Input,owner:vec2u)->bool{
 // Producer parents and fine producers consume the same all-or-none table
 // decision. An ineligible parent must not draw beside its retained children.
 if((owner.x&0x80000000u)!=0u&&!formParentSelected(owner)){return false;}
 let key=vec2u(owner.x&0xdfffffffu,owner.y);let end=input.members.x+input.members.y;
 var lo=input.members.x;var hi=end;
 loop{if(lo>=hi){break;}let mid=(lo+hi)/2u;if(keyLess(memberKeys[mid],key)){lo=mid+1u;}else{hi=mid;}}
 return lo<end&&all(memberKeys[lo]==key);
}
// A producer aggregate's member: a producer pixel of a listed domain (the
// domains whose relations the relation stage gated this frame) whose own form
// passes farAggregateMember.
fn producerMember(input:Input,owner:vec2u)->bool{
 if((owner.x&0x80000000u)==0u||p.field.z==0.0){return false;}
 let domain=owner.x&0x1fffffffu;let end=input.members.x+input.members.y;
 var lo=input.members.x;var hi=end;
 loop{if(lo>=hi){break;}let mid=(lo+hi)/2u;if(memberKeys[mid].x<domain){lo=mid+1u;}else{hi=mid;}}
 if(lo>=end||memberKeys[lo].x!=domain){return false;}
 return farAggregateMember(owner,formExtent(owner),p.field.xy,17.0*p.field.y/p.rect.w);
}
fn reserve(pointCount:u32,runCount:u32)->vec2u{
 let base=atomicAdd(&status[p.extent.w],pointCount);let slot=atomicAdd(&status[p.extent.w+1u],runCount);
 if(base+pointCount>p.extent.z){atomicOr(&status[p.extent.w+2u],2u);}
 if(slot+runCount>p.extent.y){atomicOr(&status[p.extent.w+2u],4u);}
 if(base+pointCount>p.extent.z||slot+runCount>p.extent.y){return vec2u(0xffffffffu);}
 return vec2u(base,slot);
}
// One WORKGROUP per record: an entity aggregate is one thread's pass over its
// box; a producer column is split over the 64 lanes (a column strip of the
// full-resolution field is thousands of pixels).
var<workgroup> firstRow:atomic<u32>;
var<workgroup> memberDepth:atomic<u32>;
var<workgroup> memberSeen:atomic<u32>;
@compute @workgroup_size(64) fn emit(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) local:u32){
 let id=group.x+group.y*65535u;if(id>=p.extent.x){return;}let input=inputs[id];
 if(input.mode.x==1u){producerColumn(input,local);return;}
 if(input.mode.w==2u){sourceParts(input,local);return;}
 entityAggregate(input,local);
}
// Actual source-part playback input. The source program owns these samples,
// their closure and their pose; the native owner/depth field clips words later.
// Never substitute the parent's visibility skyline for a part's construction.
fn sourceFloat(input:Input,word:u32)->f32{return bitcast<f32>(programWord(input,word));}
fn sourceVector(input:Input,word:u32)->vec4f{return vec4f(sourceFloat(input,word),sourceFloat(input,word+1u),sourceFloat(input,word+2u),sourceFloat(input,word+3u));}
fn sourcePoint(input:Input,pose:mat4x4f,geometry:u32,index:u32,count:u32,feature:u32)->Point{
 let at=geometry+index*3u;let rest=vec4f(sourceFloat(input,at),sourceFloat(input,at+1u),sourceFloat(input,at+2u),1.0);
 let clip=p.clipFromWorld*pose*rest;let valid=clip.w>1e-5&&clip.z>=0.0&&clip.z<=clip.w;
 let ndc=clip.xyz/max(clip.w,1e-5);let xy=p.rect.xy+(vec2f(ndc.x,-ndc.y)*.5+.5)*p.rect.zw;
 return Point(vec4f(xy,ndc.z,f32(index)/f32(max(count-1u,1u))),vec4u(u32(valid),bitcast<u32>(1.0/max(clip.w,1e-5)),0xffffffffu,feature));
}
// The Run keeps the source part/owner identity. Point.visible.w is the
// compile-owned edge ordinal consumed by hardBoundaryConstruction's retained
// source-edge closure: both endpoints of one source edge share it, while
// adjacent edges remain distinct. It is not a camera fragment or glyph id.
fn sourceEdgeOrdinal(segment:u32)->u32{return segment+1u;}
// Relations have their own lineage namespace. A raw contact feature ordinal
// can equal an ordinary part ordinal; that must never alias the actual memo.
fn sourcePartLineage(feature:u32,ordinal:u32,relation:bool,contact:bool)->u32{
 if(!relation){return ordinal;}
 return feature|select(0x80000000u,0xc0000000u,contact);
}
fn sourceParts(input:Input,local:u32){
 for(var i=local;i<input.program.y;i+=64u){
  let base=4u+i*8u;let feature=programWord(input,base);let entry=programWord(input,base+1u);let context=programWord(input,base+2u);let roleWord=programWord(input,base+3u);let role=programPartRole(roleWord);
  // Preserve the original 15-bit geometry-count ABI for ordinary parts. The
  // relation flags live in the unused high role bits, never in the count.
  let first=programWord(input,base+6u);let packed=programWord(input,base+7u);let contributors=packed&0xffffu;let count=(packed>>16u)&0x7fffu;let closed=(packed&0x80000000u)!=0u;let relation=(roleWord&0x80000000u)!=0u;let contact=(roleWord&0x40000000u)!=0u;let relationParent=(roleWord&0x20000000u)!=0u;let compoundParent=(roleWord&0x10000000u)!=0u;
  let contour=(roleWord&${SOURCE_COARSE_CONTOUR_FLAG}u)!=0u&&closed&&(roleWord&0xf0000000u)==0u;
  let poseAt=first+contributors*2u;let pose=mat4x4f(sourceVector(input,poseAt),sourceVector(input,poseAt+4u),sourceVector(input,poseAt+8u),sourceVector(input,poseAt+12u));
  let geometry=poseAt+16u;let segments=count-select(1u,0u,closed);let pointCount=segments*2u;
  for(var contributor=0u;contributor<contributors;contributor++){
   let owner=vec2u(programWord(input,first+contributor*2u),programWord(input,first+contributor*2u+1u));
   let at=reserve(pointCount,1u);if(at.x==0xffffffffu){return;}
   var top=1e30;var bottom=-1e30;
   for(var segment=0u;segment<segments;segment++){
    var a=sourcePoint(input,pose,geometry,segment,count,feature);var b=sourcePoint(input,pose,geometry,(segment+1u)%count,count,feature);
    if(contour){
     a.visible.w=sourceEdgeOrdinal(segment);b.visible.w=sourceEdgeOrdinal(segment);
     a.visible.z=segment;b.visible.z=(segment+1u)%count;
     a.position.w=f32(segment)/f32(segments);b.position.w=f32(segment+1u)/f32(segments);
    }
    if(closed&&segment+1u==count){b.position.w=1.0;}
    points[at.x+segment*2u]=a;points[at.x+segment*2u+1u]=b;
    top=min(top,min(a.position.y,b.position.y));bottom=max(bottom,max(a.position.y,b.position.y));
   }
   let rows=u32(clamp(floor(top)+16384.0,0.0,65535.0))|(u32(clamp(ceil(bottom)+16384.0,0.0,65535.0))<<16u);
   // Required contacts are ordinary open-hard source paths. They stay in the
   // normal planar queue and retain their source samples/feature identity;
   // only closed parts and protected/opening relations use the measured-far
   // parent consumer. This prevents a far profile from taking ownership of a
   // one-row contact and lets hardBoundaryConstruction spell its complete
   // source edge with the existing visibility path.
   let pathFlags=select(${COMPILED_FAR_PROFILE_BIT}u|0x40000002u,${SOURCE_CONTACT_BOUNDARY_FLAG}u|(${SOURCE_CONTACT_LINE_CLASS}u<<8u)|role,contact);
   runs[at.y]=Run(vec4u(at.x,pointCount,pathFlags,(feature<<5u)|(role<<2u)),vec4u(feature,owner,feature),vec4u(entry,context,programWord(input,base+4u),programWord(input,base+5u)),vec4u(0u,count,rows,role|select(0u,0x80000000u,relation)|select(0u,0x40000000u,contact)|select(0u,0x20000000u,relationParent)|select(0u,0x10000000u,compoundParent)|select(0u,${SOURCE_COARSE_CONTOUR_FLAG}u,contour)),vec4u(owner,input.program.z,sourcePartLineage(feature,i,relation,contact)));
  }
 }
}

// form or an octree node under the connected-outline tier (source-lod
// PARENT_MAX_COLUMNS columns at most, so the box meets at most
// PARENT_PROFILE_COLUMNS page columns). The 64 lanes split its page columns;
// each lane inspects every native pixel of its columns' strips of the box.
var<workgroup> profileTop:array<f32,${PARENT_PROFILE_COLUMNS}>;
var<workgroup> aggregateLow:atomic<u32>;
var<workgroup> aggregateHigh:atomic<u32>;
var<workgroup> aggregateDepth:atomic<u32>;
fn entityAggregate(input:Input,local:u32){
 let start=max(vec2i(0),vec2i(floor(input.box.xy)));let end=min(vec2i(p.field.xy),vec2i(ceil(input.box.zw)));
 if(local==0u){atomicStore(&aggregateLow,0xffffffffu);atomicStore(&aggregateHigh,0u);atomicStore(&aggregateDepth,0u);}
 let toPage=p.rect.z/p.field.x;
 let firstColumn=i32(floor((f32(start.x)*toPage+p.rect.x)/16.0));
 let lastColumn=i32(floor((f32(end.x)*toPage+p.rect.x)/16.0));
 let columns=u32(clamp(lastColumn-firstColumn+1,0,${PARENT_PROFILE_COLUMNS}));
 workgroupBarrier();
 for(var k=local;k<${PARENT_PROFILE_COLUMNS}u;k+=64u){
  var top=1e30;
  if(k<columns){
   let c=firstColumn+i32(k);
   let x0=max(start.x,i32(floor((f32(c)*16.0-p.rect.x)/toPage))-1);let x1=min(end.x,i32(ceil((f32(c+1)*16.0-p.rect.x)/toPage))+1);
   // Every native visible pixel of the column's strip of the box: a thin
   // form is never missed by a probe quota.
   for(var y=start.y;y<end.y;y++){for(var x=x0;x<x1;x++){
    let at=vec2i(x,y);let owner=textureLoad(ownerField,at,0).xy;let d=textureLoad(depthField,at,0).x;
    if(d<=0.0||!listed(input,owner)){continue;}
    let q=(vec2f(at)+.5)/p.field.xy*p.rect.zw+p.rect.xy;if(i32(floor(q.x/16.0))!=c){continue;}
    top=min(top,q.y);atomicMin(&aggregateLow,u32(y));atomicMax(&aggregateHigh,u32(y+1));atomicMax(&aggregateDepth,bitcast<u32>(d));
   }}
  }
  profileTop[k]=top;
 }
 workgroupBarrier();
 if(local!=0u){return;}
 let lowY=atomicLoad(&aggregateLow);if(lowY==0xffffffffu){return;} // fully hidden/discarded by the original material
 let depth=bitcast<f32>(atomicLoad(&aggregateDepth));
 let ay=f32(lowY)/p.field.y*p.rect.w+p.rect.y;let by=f32(atomicLoad(&aggregateHigh))/p.field.y*p.rect.w+p.rect.y;
 var profilePoints=0u;for(var k=0u;k<columns;k++){if(profileTop[k]<1e29){profilePoints+=2u;}}
 // Older residents without usable source geometry/pose keep exactly ONE
 // incumbent profile, not one duplicate skyline per metadata part.
 let small=input.mode.w==3u;
 // Keep the incumbent profile intact until the complete small record fits.
 // Its two source-owned layout anchors are extra reserved points, not a
 // visibility fragment or part of the published skyline's drawable range.
 let at=reserve(profilePoints+select(0u,2u,small),1u);if(at.x==0xffffffffu){return;}
 let identity=(input.identity.x*0x9e3779b9u)^(input.identity.y*0x85ebca6bu)^0x20000000u;
 let top=u32(clamp(floor(ay)+16384.0,0.0,65535.0));let bottom=u32(clamp(ceil(by)+16384.0,0.0,65535.0));
 var output=at.x;
  for(var k=0u;k<columns;k++){let tk=profileTop[k];if(tk>=1e29){continue;}let left=f32(firstColumn+i32(k))*16.0;
   points[output]=Point(vec4f(left,tk,depth,f32(k)/f32(columns)),vec4u(1u,0x3f800000u,0xffffffffu,identity));
   points[output+1u]=Point(vec4f(left+15.999,tk,depth,f32(k+1u)/f32(columns)),vec4u(1u,0x3f800000u,0xffffffffu,identity));output+=2u;
  }
  // Unknown source programs retain the incumbent parent profile. Do not
  // fabricate a role, entry, context or contributor identity.
  var part=vec4u(0u);var paper=vec4u(0u,0u,top|(bottom<<16u),0u);var correspondence=vec4u(0u);
  if(small){
   let sourceLow=input.box.xy/p.field.xy*p.rect.zw+p.rect.xy;
   let sourceHigh=input.box.zw/p.field.xy*p.rect.zw+p.rect.xy;
   // Size, not which page row a translated endpoint happens to cross, owns
   // the construction level. Translation moves the anchor only.
   let sourceRows=u32(max(1.0,round((sourceHigh.y-sourceLow.y)/17.0)));let sourceWidth=u32(max(0.0,round(sourceHigh.x-sourceLow.x)));
   points[at.x+profilePoints]=Point(vec4f(sourceLow,depth,0.0),vec4u(1u,0x3f800000u,0xffffffffu,identity));
   points[at.x+profilePoints+1u]=Point(vec4f(sourceHigh,depth,1.0),vec4u(1u,0x3f800000u,0xffffffffu,identity));
   part=vec4u(input.mode.z,sourceWidth,profilePoints/2u,sourceRows);
   // paper.z remains the incumbent packed top/bottom. The far-stage bridge
   // supplies the small helper a local paper-layout view only after the
   // source marker is checked; it never mutates the fallback run.
   correspondence=vec4u(input.identity,input.program.z,0x534d414cu);
  }
  runs[at.y]=Run(vec4u(at.x,profilePoints,${COMPILED_FAR_PROFILE_BIT}u|0x40000002u,0u),vec4u(identity,input.identity,identity),part,paper,correspondence);
}
// THE PRODUCER AGGREGATE, one page column (input.box = the column's field
// strip). The column's topmost ink row: when a member pixel is in it, the
// members ARE the merged outline there and the column gets one far skyline unit
// at that row; otherwise the members lie under another form's outline (props on
// the ground they stand on) and add nothing.
fn inked(x:i32,y:i32)->bool{let at=vec2i(x,y);let owner=textureLoad(ownerField,at,0).xy;return textureLoad(depthField,at,0).x>0.0&&(owner.x|owner.y)!=0u;}
fn producerColumn(input:Input,local:u32){
 let x0=max(0,i32(floor(input.box.x)));let x1=min(i32(p.field.x),i32(ceil(input.box.z)));
 if(local==0u){atomicStore(&firstRow,0xffffffffu);atomicStore(&memberDepth,0u);atomicStore(&memberSeen,0u);}
 workgroupBarrier();
 // The column's topmost inked field row: lanes take rows local, local+64, ...
 // and stop at the first inked one (or below a row another lane found).
 for(var y=i32(local);y<i32(p.field.y);y+=64){
  if(u32(y)>atomicLoad(&firstRow)){break;}
  var hit=false;for(var x=x0;x<x1;x++){if(inked(x,y)){hit=true;break;}}
  if(hit){atomicMin(&firstRow,u32(y));break;}
 }
 workgroupBarrier();
 let row=atomicLoad(&firstRow);
 // Members in that row: the lanes split its pixels.
 if(row!=0xffffffffu){
  for(var x=x0+i32(local);x<x1;x+=64){
   let at=vec2i(x,i32(row));let owner=textureLoad(ownerField,at,0).xy;let d=textureLoad(depthField,at,0).x;
   if(d>0.0&&producerMember(input,owner)){atomicStore(&memberSeen,1u);atomicMax(&memberDepth,bitcast<u32>(d));}
  }
 }
 workgroupBarrier();
 if(local!=0u||row==0xffffffffu||atomicLoad(&memberSeen)==0u){return;}
 let depth=bitcast<f32>(atomicLoad(&memberDepth));
 let at=reserve(2u,1u);if(at.x==0xffffffffu){return;}
 let column=input.mode.y;let left=f32(column)*16.0;
 let top=(f32(row)+.5)/p.field.y*p.rect.w+p.rect.y;
 let identity=(input.identity.x*0x9e3779b9u)^(input.identity.y*0x85ebca6bu)^0x10000000u;
 points[at.x]=Point(vec4f(left,top,depth,0.0),vec4u(1u,0x3f800000u,0xffffffffu,identity));
 points[at.x+1u]=Point(vec4f(left+15.999,top,depth,1.0),vec4u(1u,0x3f800000u,0xffffffffu,identity));
 let rows=u32(clamp(floor(top)+16384.0,0.0,65535.0))|(u32(clamp(ceil(top)+16384.0,0.0,65535.0))<<16u);
 // Terrain form (0x10000000), a whole far part (0x40000000: its form rows are
 // already measured, far), role 1 (skyline shoulder), line class 0 (outline).
 runs[at.y]=Run(vec4u(at.x,2u,0x50000001u,0u),vec4u(identity,input.identity,identity),vec4u(0u),vec4u(0u,0u,rows,0u),vec4u(0u));
}
`;

/** Exact assembled module consumed by the runtime pipeline and validators. */
export const SOURCE_FAR_FORMS_PIPELINE_SHADER=withCoherentSourceParts(SOURCE_FAR_FORMS_SHADER);
const programs=new DeviceComputePrograms(device=>{
 const layout=device.createBindGroupLayout({label:'MAIN compiled far aggregates',entries:[
  {binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:'uniform'}},
  {binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:'read-only-storage'}},
  {binding:2,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:'uint'}},
  {binding:3,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:'unfilterable-float'}},
  ...[4,5,6].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:'storage' as const}})),
  {binding:7,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:'uint'}},
  {binding:8,visibility:GPUShaderStage.COMPUTE,buffer:{type:'read-only-storage'}},
 ]});
 return {resources:layout,plans:new Map([['emit',()=>({label:'MAIN compiled far aggregates',layout:device.createPipelineLayout({bindGroupLayouts:[layout]}),
  compute:{entryPoint:'emit',module:device.createShaderModule({label:'MAIN source far forms',code:SOURCE_FAR_FORMS_PIPELINE_SHADER})}})]])};
});
export async function prepareSourceFarForms(device:GPUDevice,progress?:ComputePipelineProgress):Promise<void>{await programs.prepare(device,progress);}
const pairLess=(a:readonly [number,number],b:readonly [number,number])=>a[0]-b[0]||a[1]-b[1];
const roleCode=(role:SourceWholePartRole):number=>role==='cap'?0:role==='turn'?1:role==='flank'?2:3;
/** A relation is a source-owned boundary, not a screen-space paper mask. It
 * enters the same native whole-part consumer as ordinary parts, with its
 * immutable source samples and instance pose. The opening and protected-void
 * lists intentionally overlap in the compiler; collapse that duplicate here
 * so the parent is emitted once, never parent plus a second copy of its hole. */
const RELATION_ROLE_BIT=0x80000000;
const RELATION_CONTACT_BIT=0x40000000;
const RELATION_PARENT_BIT=0x20000000;
/** Explicit source-product marker. RELATION_PARENT_BIT also appears on
 * legacy ordinary parts whenever a construction has relations. */
const COMPOUND_PARENT_BIT=0x10000000;
const relationFloatValue=new Float32Array(1),relationFloatWord=new Uint32Array(relationFloatValue.buffer);
const relationFloatBits=(value:number):number=>{relationFloatValue[0]=value;return relationFloatWord[0]!;};
const relationKey=(item:SourceWholeOpening):string=>`${item.feature}:${item.contributors.map(pair=>`${pair[0]}:${pair[1]}`).join(',')}:${item.worldFromSource?Array.from(item.worldFromSource,relationFloatBits).join(','):''}:${item.geometry?.map(point=>point.map(relationFloatBits).join(',')).join(';')??''}`;
const relationListKey=(kind:SourceWholePartRef['relationKind'],item:SourceWholeOpening):string=>`${kind==='contact'?'contact':'boundary'}:${relationKey(item)}`;
const relationLists=(value:SourceWholeConstruction):readonly (readonly [readonly SourceWholeOpening[],SourceWholePartRef['relationKind']])[]=>[[value.openings,'opening'],[value.contacts,'contact'],[value.protectedVoids,'protected-void'],[value.paper??[],'paper']];
function relationParts(value:SourceWholeConstruction):SourceWholePartRef[]|undefined{
 const seen=new Set<string>(),result:SourceWholePartRef[]=[];
 for(const [items,relationKind] of relationLists(value))for(const relation of items){
  const key=relationListKey(relationKind,relation);if(seen.has(key))continue;seen.add(key);
  // No silent seed turn/pose inheritance. Unsupported relation metadata keeps
  // the incumbent children instead of drawing a different native passage.
  if(!relation.geometry||relation.geometry.length<2||relation.entryRef===undefined||relation.context===undefined||relation.role===undefined||!relation.worldFromSource)return undefined;
  const contributorKeys=new Set<string>(),contributors=relation.contributors.filter(pair=>{const key=`${pair[0]>>>0}:${pair[1]>>>0}`;if(contributorKeys.has(key))return false;contributorKeys.add(key);return true;});
  if(!contributors.length)return undefined;
  result.push({feature:relation.feature,entryRef:relation.entryRef,context:relation.context,role:relation.role,ports:[relation.ports[0]??0,relation.ports.at(-1)??relation.ports[0]??0],contributors,geometry:relation.geometry,closed:relation.closed??false,worldFromSource:relation.worldFromSource,relationKind});
 }
 return result;
}
const floatWord=new Float32Array(1),uintWord=new Uint32Array(floatWord.buffer);
const floatBits=(value:number):number=>{floatWord[0]=value;return uintWord[0]!;};
const IDENTITY_POSE=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1] as const;
const coherentCorrespondencePackets=new WeakMap<SourceWholeConstruction,Uint32Array>();
interface PackedCoherentProgram {readonly words:Uint32Array;readonly parts:number;readonly revision:number;readonly signature:number;readonly correspondence:number;readonly mode:number;}
// Only generation-owned coherent products qualify. Legacy callers may mutate
// their ordinary construction objects and retain their existing packing path.
const coherentSourcePrograms=new WeakMap<SourceWholeConstruction,PackedCoherentProgram>();
const partCounts=(part:SourceWholeConstruction['parts'][number]):number=>(part.contributors.length|((part.geometry?.length??0)<<16)|(part.closed?0x80000000:0))>>>0;
/** Missing pose is older metadata, never permission to project rest geometry
 * at the world origin. Keep the incumbent until the owning adapter is wired. */
function posedProgram(value:SourceWholeConstruction|undefined):boolean{
 const relations=value?relationParts(value):undefined;
 const relationKeys=new Set<string>();if(value)for(const [items,kind] of relationLists(value))for(const item of items)if(item.geometry&&item.geometry.length>=2)relationKeys.add(relationListKey(kind,item));
 const relationCount=relationKeys.size;
 return !!value&&!!relations&&relations.length===relationCount&&[...value.parts,...relations].every(part=>part.feature<=0x07ffffff&&part.entryRef>0&&typeof part.closed==='boolean'&&!!part.geometry&&part.geometry.length>=2&&part.geometry.length<=0x7fff&&part.contributors.length>0&&part.contributors.length<=0xffff&&!!part.worldFromSource&&part.worldFromSource.length===16);
}
/** The compiled aggregate records of one camera: input rows and their sorted
 * member keys. Pure (no device), so the record layout is testable. */
export function farAggregateRecords(sources:readonly SourceFarForms[],rect:{x:number;y:number;width:number;height:number},field:{width:number;height:number}):{inputs:Uint32Array;keys:Uint32Array;count:number;perspective:boolean}{
 const rows:{identity:[number,number];members:(readonly [number,number])[];box:readonly number[];mode:number;column:number;construction?:SourceWholeConstruction;smallParent?:SourceSmallParent;sourceRevision?:number}[]=[];
 let producers:SourceFarProducers|undefined;
 for(const source of sources){
  // A single far form is the one-member aggregate; an octree node's members are
  // its aggregate's, never emitted singly (FarFormInstance.parent).
  for(const instance of source.instances)if(instance.parent===undefined)rows.push({identity:[source.domain,instance.eid],members:[[source.domain,instance.eid]],box:instance.box,mode:0,column:0,...(instance.smallParent?.replaceable?{smallParent:instance.smallParent,sourceRevision:instance.sourceRevision}:{})});
  for(const aggregate of source.aggregates??[]){
   // A known partial construction is never silently downgraded to a profile;
   // the selector descends those members. An absent construction is older
   // residency, so keep the incumbent parent profile until its replacement is
   // actually populated and connected.
   if(aggregate.construction&&!sourceWholeConstructionRenderable(aggregate.construction))continue;
   // Source-Lod owns aggregate members as named records. Lower them once at
   // this producer seam to the tuple layout used by the resident member-key
   // table; treating the records as iterables throws before the first frame.
   const members=aggregate.members.map(member=>{
    // The local far packet type is the resident tuple layout, while the
    // source-lod adapter supplies its named `{mesh,eid}` records. Accept both
    // at this lowering seam so the producer cannot throw on a valid selector
    // result and the type remains honest for direct far callers.
    if(Array.isArray(member))return [member[0]!,member[1]!] as const;
    const named=member as unknown as {readonly mesh:number;readonly eid:number};
    return [named.mesh,named.eid] as const;
   }).sort(pairLess);
   rows.push({identity:[0,aggregate.key>>>0],members,box:aggregate.box,mode:0,column:0,construction:aggregate.construction});
  }
  if(source.producers?.domains.length)producers=source.producers;
 }
 if(producers){
  const domains=[...new Set(producers.domains)].sort((a,b)=>a-b).map(d=>[d,0] as const),columns=Math.ceil(rect.width/16);
  for(let c=0;c<columns;c++){const x0=(c*16-rect.x)/rect.width*field.width,x1=((c+1)*16-rect.x)/rect.width*field.width;
   rows.push({identity:[0,(0x60000000|c)>>>0],members:domains,box:[x0,0,x1,field.height],mode:1,column:c});}
 }
 const inputs=new Uint32Array(Math.max(1,rows.length)*16),floats=new Float32Array(inputs.buffer),keyList:number[]=[];
 // Every column record of the producer aggregate reads ONE shared domain list.
 const placed=new Map<object,number>();
 rows.forEach((row,i)=>{let offset=placed.get(row.members);if(offset===undefined){if(keyList.length&1)keyList.push(0);offset=keyList.length/2;placed.set(row.members,offset);for(const [a,b]of row.members)keyList.push(a>>>0,b>>>0);}
  const cachedProgram=row.construction?coherentSourcePrograms.get(row.construction):undefined;
  const base=i*16;inputs.set([row.identity[0]>>>0,row.identity[1]>>>0,offset,row.members.length],base);floats.set(row.box.slice(0,4),base+4);inputs.set([row.mode,row.column,0,cachedProgram?.mode??(posedProgram(row.construction)?2:row.construction?1:0)],base+8);
  if(row.smallParent&&row.smallParent.positiveEnd!=='open'&&row.smallParent.negativeEnd!=='open'){
   const kindEnds=packSmallConstructionKindAndEnds(row.smallParent.kind,row.smallParent.positiveEnd,row.smallParent.negativeEnd);
   inputs[base+10]=kindEnds;inputs[base+11]=3;inputs[base+14]=row.sourceRevision??0;
  }
  if(cachedProgram){
   const programOffset=keyList.length;for(const word of cachedProgram.words)keyList.push(word);
   inputs.set([programOffset,cachedProgram.parts,cachedProgram.revision,cachedProgram.signature],base+12);inputs[base+10]=cachedProgram.correspondence;return;
  }
  if(row.construction){const sourceParts=row.construction.parts,relations=relationParts(row.construction);if(!relations)return;const parts=[...sourceParts,...relations],programOffset=keyList.length,hasRelations=relations.length>0,compound=!!row.construction.compound;
   const contourProduct=row.construction.principalComplete===true&&!hasRelations&&!compound;
   const signature=()=>{let h=2166136261>>>0;const mix=(word:number)=>{h=Math.imul(h^(word>>>0),16777619)>>>0;};mix(row.construction!.revision);mix(Number(row.construction!.principalComplete===true));mix(parts.length);for(const [index,part] of parts.entries()){mix(part.feature);mix(part.entryRef);mix(part.context);mix(roleCode(part.role));mix((part.relationKind==='contact'?RELATION_CONTACT_BIT|RELATION_ROLE_BIT:part.relationKind?RELATION_ROLE_BIT:hasRelations?RELATION_PARENT_BIT:0)|(compound?COMPOUND_PARENT_BIT:0)|(contourProduct&&!part.relationKind&&part.closed===true?SOURCE_COARSE_CONTOUR_FLAG:0));mix(part.ports[0]);mix(part.ports[1]);const geometry=part.geometry??[],packed=partCounts(part);mix(packed);for(const contributor of part.contributors){mix(contributor[0]);mix(contributor[1]);}if(part.worldFromSource)for(const value of Array.from(part.worldFromSource))mix(floatBits(value));for(const point of geometry){mix(floatBits(point[0]));mix(floatBits(point[1]));mix(floatBits(point[2]));}}return h||1;};
   const programSignature=signature();keyList.push(0x50524731,row.construction.revision>>>0,parts.length>>>0,programSignature);
   // Each part's contributor pairs occupy one contiguous source-owned slice.
   // The shader hashes/reads `count` pairs from `first`; a globally deduped
   // table would make a part's slice run into the next part whenever their
   // contributor sets differ.
   const contributorOffset=4+parts.length*8;
   let payloadCursor=0;const partWords:number[]=[],payloadWords:number[]=[];
   for(const [index,part] of parts.entries()){
    // programWord adds programOffset exactly once, so payload references
    // here are RELATIVE to the program header, never absolute key addresses.
    const contributorBase=contributorOffset+payloadCursor;
    for(const contributor of part.contributors){payloadWords.push(contributor[0]>>>0,contributor[1]>>>0);payloadCursor+=2;}
    const geometry=part.geometry??[],packed=partCounts(part);
    const pose=part.worldFromSource??IDENTITY_POSE;
    for(let axis=0;axis<16;axis++){payloadWords.push(floatBits(pose[axis]!));payloadCursor++;}
    // Pose precedes the immutable rest samples and is part of this posed
    // contributor's program identity; equal source geometry at a different
    // pose must not collapse into the same relation record.
    for(const point of geometry){payloadWords.push(floatBits(point[0]),floatBits(point[1]),floatBits(point[2]));payloadCursor+=3;}
    const relationFlags=(part.relationKind==='contact'?RELATION_ROLE_BIT|RELATION_CONTACT_BIT:part.relationKind?RELATION_ROLE_BIT:hasRelations?RELATION_PARENT_BIT:0)|(compound?COMPOUND_PARENT_BIT:0)|
      (contourProduct&&!part.relationKind&&part.closed===true?SOURCE_COARSE_CONTOUR_FLAG:0);
    partWords.push(part.feature>>>0,part.entryRef>>>0,part.context>>>0,(roleCode(part.role)|relationFlags)>>>0,part.ports[0]>>>0,part.ports[1]>>>0,contributorBase>>>0,packed>>>0);
   }
   // A real octree parent may contain many source parts. Do not use an
   // argument spread whose engine stack limit depends on scene density.
   for(const word of partWords)keyList.push(word);
   for(const word of payloadWords)keyList.push(word);
   const product=coherentParentProductOf(row.construction);
   let coherentWords=product?coherentCorrespondencePackets.get(row.construction):undefined;
   if(product&&!coherentWords){
    const correspondenceParts:CoherentCorrespondencePart[]=parts.map(part=>({feature:part.feature,kind:part.relationKind??'part',contributors:part.contributors}));
    coherentWords=encodeCoherentCorrespondence(coherentCorrespondenceEntries(product,row.members.map(([mesh,eid])=>({mesh,eid,construction:row.construction!})),row.construction),correspondenceParts);
    coherentCorrespondencePackets.set(row.construction,coherentWords);
   }
   const coherentOffset=coherentWords?keyList.length-programOffset:0;if(coherentWords)for(const word of coherentWords)keyList.push(word);
   inputs.set([programOffset,parts.length,row.construction.revision>>>0,programSignature],base+12);
   inputs[base+10]=coherentOffset;
   if(product)coherentSourcePrograms.set(row.construction,{words:Uint32Array.from(keyList.slice(programOffset)),parts:parts.length,revision:row.construction.revision>>>0,signature:programSignature,correspondence:coherentOffset,mode:inputs[base+11]!});
  }
 });
 if(keyList.length&1)keyList.push(0);
 return {inputs,keys:Uint32Array.from(keyList.length?keyList:[0,0]),count:rows.length,perspective:!!producers?.perspective};
}
export class GpuSourceFarForms {
 private readonly ready;
 constructor(private readonly device:GPUDevice){this.ready=programs.get(device);}
 encode(encoder:GPUCommandEncoder,sources:readonly SourceFarForms[],field:SurfaceField,output:{runs:GPUBuffer;points:GPUBuffer;status:GPUBuffer;statusOffset:number;capacity:number;pointCapacity:number},rect:{x:number;y:number;width:number;height:number},retain:(buffer:GPUBuffer)=>void):void{
  const records=farAggregateRecords(sources,rect,field);if(!records.count)return;
  const buffer=(label:string,words:Uint32Array)=>{const b=this.device.createBuffer({label,size:words.byteLength,usage:GPUBufferUsage.STORAGE,mappedAtCreation:true});new Uint32Array(b.getMappedRange()).set(words);b.unmap();retain(b);return b;};
  const inputs=buffer('MAIN far aggregate records',records.inputs),keys=buffer('MAIN far aggregate member keys',records.keys);
  const uniform=this.device.createBuffer({label:'MAIN far aggregate bounds',size:112,usage:GPUBufferUsage.UNIFORM,mappedAtCreation:true});
  const u=new Uint32Array(uniform.getMappedRange()),f=new Float32Array(u.buffer);u.set([records.count,output.capacity,output.pointCapacity,output.statusOffset/4]);
  f.set([rect.x,rect.y,rect.width,rect.height,field.width,field.height,records.perspective?1:0,0],4);f.set(inverseMatrix(field.worldFromClip),12);uniform.unmap();retain(uniform);
  const group=this.device.createBindGroup({layout:this.ready.resources,entries:[{binding:0,resource:{buffer:uniform}},{binding:1,resource:{buffer:inputs}},
   {binding:2,resource:field.textures[3]!.createView()},{binding:3,resource:field.textures[0]!.createView()},
   {binding:4,resource:{buffer:output.status}},{binding:5,resource:{buffer:output.runs}},{binding:6,resource:{buffer:output.points}},
   {binding:7,resource:formDepths(this.device).view(field)},{binding:8,resource:{buffer:keys}}]});
  const pass=encoder.beginComputePass({label:'MAIN compiled far aggregates'});pass.setPipeline(this.ready.pipelines.get('emit')!);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.min(records.count,65535),Math.ceil(records.count/65535));pass.end();
 }
}
