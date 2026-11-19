import {PLAYBACK_FAMILIES,familyPlayback} from './gpu-playback-program.ts';
import {buildFarKindStages,FAR_KIND_ENTRIES,FAR_PLAN_WORDS,standingDispatch} from './gpu-far-program.ts';
import {isolateComputeEntry} from './gpu-compute-program.ts';
import type {SourcePartAnchorTable} from './surface-relations.ts';
import {WORD_STRIDE,PAIRED_FLANK_LEANS,FOLIAGE_EDGES_AT,HARD_SLOPE_PHASES,JUNCTION_TWO,JUNCTION_THREE,SOFT_PART_TABLE_VERSION,SOFT_PART_CONTEXTS,SOFT_PART_TABLE_RECORD_WORDS,SOFT_PART_TABLE_RECORDS,SOFT_PART_TABLE_AT,
 SOFT_D5_TABLE_AT,SOFT_D5_TABLE_VERSION,SOFT_D5_TABLE_TABLES,SOFT_D5_TABLE_COLUMNS,SOFT_D5_TABLE_PHASES,SOFT_D5_TABLE_VARIANTS,SOFT_D5_TABLE_ENTRY_WORDS,SOFT_D5_TABLE_HEADER_WORDS,SOFT_D5_TABLE_NONE,SOFT_D5_TABLE_HARD,SOFT_D5_TABLE_UPRIGHT,SOFT_D5_TABLE_WORDS,
 type PackedRoleGrammar,type GrammarRole} from './role-grammar-format.ts';
import {loadRuntimeRoleGrammar} from './runtime-role-grammar.ts';
import {decodeSmallConstructionTable,lookupSmallConstructionRecord,type PackedSmallConstructionTable} from './role-grammar-format.ts';
import type {SourceSmallParent} from './source-small-parent.ts';
import {compileComputePipelines,prepareComputePipelines,type ComputePipelinePlan} from './gpu-pipeline-preparation.ts';
import {CONSTRUCTION_POINT_WGSL,POINT_SOURCE_CHART_WGSL,SPAN_SOURCE_TAG,NATIVE_SPAN_SOURCE} from './source-coordinates.ts';
import {TONE_PASSAGES,TEXTURE_EDGE_MEASURE} from './compiled-tone-passages.ts';
import {nativeWholeInk,type NativeGlyph,type NativeNeighborhood,type NativeOccurrence} from './native-page.ts';
import type {SurfaceField} from './surface-fields.ts';
import {FORM_DEPTH_WGSL,FORM_DEPTH_HEADER_ROW,NEAR_FROM_ROWS,FAR_STRIP_BELOW_ROWS,LOD_TIER_HOLD,formDepths} from './form-depths.ts';
import type {NativeSupport} from './native-presentation.ts';
import {D10_LINE_CLASSES} from './d10-line-classes.ts';
import {CONSTRUCTION_RUN_WORDS,GpuConstructionRuns,type ResidentConstructionRuns,type SourceConstructionClass} from './gpu-construction-runs.ts';
import {SOURCE_PART_CONTEXT_CLASS_SHIFT,SOURCE_PART_CONTEXT_CLASS_MASK,SOURCE_PART_FEATURE_ORDINAL_MASK} from './compiled-construction-topology.ts';
import type {RelationVisibility} from './relation-visibility.ts';
import type {NativeWhitespacePolicy} from './text/native-whitespace.ts';
import type {GpuToneMembers} from './tone-field.ts';
import {GpuNativeSealer} from './gpu-native-seal.ts';
import {GPU_PLANAR_FAN} from './gpu-planar-fan.ts';
import {issueGpuNativeTapeSeal,type GpuNativeTape} from './gpu-native-tape.ts';


export interface ArticulationPoint {
 readonly x:number;readonly y:number;readonly depth:number;readonly parameter:number;readonly visible:boolean;
 /** Reciprocal homogeneous clip W at this SOURCE point. Affine callers
  * may omit it; projected mesh points carry their actual perspective. */
 readonly reciprocalW?:number;
}
export interface ArticulationRun {
 readonly id:string;readonly owner:string;readonly sources:readonly string[];
 readonly role:GrammarRole;readonly size:0|1|2;readonly lineClass:0|1|2|3|4;
 readonly key:number;readonly generation?:number;
 /** Non-anchor source intervals are consecutive [start,end] pairs. Only
  * each start carries visibility; cuts never manufacture connecting spans. */
 readonly points:readonly ArticulationPoint[];
 readonly source?:readonly [number,number];
 /** A retained whole multirow role part at an actual graph anchor. */
 readonly part?:string;
 readonly anchorOnly?:boolean;
 /** One complete rounded source cycle, not independent local cap stamps. */
 readonly round?:boolean;
 /** Genuine hard-plane edge, with its own clean-line run program. */
 readonly hard?:boolean;
 readonly upright?:boolean;
}
export interface ArticulationResult {
 readonly groups:readonly {readonly group:NativeNeighborhood;readonly owners:readonly string[];readonly sources:readonly string[];readonly support?:NativeSupport}[];
 readonly unresolved:readonly string[];
 /** Words that lost placement to another word of a drawn owner. */
 readonly unresolvedWords?:number;
 readonly sourceLines?:number;
 readonly sourceOverflow?:{readonly flags:number;readonly relations:number;readonly intervals:number;readonly triangles?:number};
 /** Display owns the sealed GPU tape. Materializing its source annotations
  * and scalar page is an explicit export operation, not a frame dependency. */
 readonly resident?:ResidentNativeConstruction;
}
export interface ResidentNativeConstruction {
 readonly tape:GpuNativeTape;
 readonly glyphCount:number;
 read():Promise<ArticulationResult>;
 release():void;
}
export interface ArticulationLiteral {readonly camera:string;readonly group:NativeNeighborhood;readonly support:NativeSupport;readonly tone?:boolean;readonly owners?:readonly string[];readonly sources?:readonly string[];
 /** Authored world labels use their original projected depth. Overlay labels
  * omit it. The same native-mask visibility rule runs on the GPU, not in a
  * CPU surface sample or a second image-to-text decision. */
 readonly depth?:number;
}
/** A same-pose compact input may become ready later in the host frame.
 * Synchronous delivery can still use that frame's encoder, without a fence. */
export interface ArticulationLiteralFeed {
 subscribe(ready:(literals:readonly ArticulationLiteral[],tone?:GpuToneMembers)=>void,failed:(error:unknown)=>void):void;
}
// State + exact five-word source key + nine-word retained articulation + five
// exact source-correspondence words used to disambiguate hash collisions.
// Owner/domain/address already live in the key: copying them into every
// payload made a valid large scene exceed the storage binding limit.
/** Loss-probe instrumentation in the placement shader: off unless ?mainProbes=1. */
const MAIN_PROBES=typeof location!=='undefined'&&new URLSearchParams(location.search).has('mainProbes');
const ROW_CAP=2048,WORD_WORDS=32,ROW_WORDS=18,MEMO_AT=2056,MEMO_WORDS=20,MEMO_HEADER_WORDS=6;
const MEMO_CORRESPONDENCE_AT=15;



const LIMB_SLOTS=4096,LIMB_WORDS=8;
/** One source-program seed record per relation run. Records are established
 * before reduction, then a later ordered pass walks exact-key bucket chains.
 * [domain, instance, lifetime, program, next, leader, minX, minY, maxX, maxY,
 * contributor count, ready, prepared, native record, layout x/y/width/rows]. */
export const SOURCE_PART_GROUP_WORDS=18;
export const SOURCE_PART_HEAD_WORDS=1;
/** Job-owned descriptor followed by the camera's immutable anchor table. */
export const PART_ANCHOR_HEADER_WORDS=8;
export const PART_ANCHOR_TABLE_RECORD_WORDS=8;
export function sourcePartGroupSlots(runCount:number):number{
 if(!Number.isSafeInteger(runCount)||runCount<0)throw Error('Invalid source-part group extent');
 let slots=1,needed=Math.max(2,runCount*2);while(slots<needed)slots*=2;
 return slots;
}
const WORD_PROPOSAL_WORDS=25;
export const RESIDENT_RUN_WORDS=CONSTRUCTION_RUN_WORDS;
const ACTIVE_QUEUE_FAMILIES=9,ACTIVE_QUEUE_META_WORDS=ACTIVE_QUEUE_FAMILIES*2,ACTIVE_SOURCE_QUEUE_FAMILIES=7,ACTIVE_QUEUE_LABEL=7,ACTIVE_QUEUE_TONE=8,ACTIVE_ARGUMENT_WORDS=ACTIVE_QUEUE_FAMILIES*3;
export const FAR_TEXTURE_PROBE_BOUNDS=[3,48,6,0] as const;
export interface MainTransientLayoutInput {readonly runCount:number;readonly epochWords:number;readonly paperWords:number;readonly gridWords:number;readonly orderWords:number;readonly rowLayoutWords:number;readonly proposalCapacity:number;readonly farPlanWords:number;readonly activeQueueWords:number;readonly partAnchorWords?:number;}
export interface MainTransientLayout {readonly transientWords:number;readonly proposalWords:number;readonly proposalBase:number;readonly proposalStatus:number;readonly farPlanBase:number;readonly activeQueueBase:number;readonly rowLayoutBase:number;readonly orderBase:number;readonly partAnchorBase:number;readonly intervals:Readonly<Record<'state'|'partAnchors'|'order'|'rows'|'proposals'|'farPlan'|'activeQueues',readonly [number,number]>>;}
export function mainTransientLayout(input:MainTransientLayoutInput):MainTransientLayout{
 const {runCount,epochWords,paperWords,gridWords,orderWords,rowLayoutWords,proposalCapacity,farPlanWords,activeQueueWords}=input;
 const partAnchorWords=input.partAnchorWords??0;
 for(const value of [runCount,epochWords,paperWords,gridWords,orderWords,rowLayoutWords,proposalCapacity,farPlanWords,activeQueueWords,partAnchorWords])if(!Number.isSafeInteger(value)||value<0)throw Error('Invalid MAIN transient extent');
 const stateWords=runCount+epochWords+paperWords+epochWords+runCount*3+gridWords,orderSpan=orderWords*2,proposalWords=proposalCapacity*WORD_PROPOSAL_WORDS+4;
 const transientWords=stateWords+partAnchorWords+orderSpan+rowLayoutWords+proposalWords+farPlanWords+activeQueueWords;
 const proposalBase=transientWords-proposalWords-farPlanWords-activeQueueWords;
 const rowLayoutBase=proposalBase-rowLayoutWords,orderBase=rowLayoutBase-orderSpan,farPlanBase=proposalBase+proposalWords,activeQueueBase=farPlanBase+farPlanWords;
 const partAnchorBase=stateWords;
 const intervals={state:[0,stateWords] as const,partAnchors:[partAnchorBase,partAnchorBase+partAnchorWords] as const,order:[orderBase,orderBase+orderSpan] as const,rows:[rowLayoutBase,rowLayoutBase+rowLayoutWords] as const,proposals:[proposalBase,proposalBase+proposalWords] as const,farPlan:[farPlanBase,farPlanBase+farPlanWords] as const,activeQueues:[activeQueueBase,activeQueueBase+activeQueueWords] as const};
 return {transientWords,proposalWords,proposalBase,proposalStatus:proposalBase+proposalCapacity*WORD_PROPOSAL_WORDS,farPlanBase,activeQueueBase,rowLayoutBase,orderBase,partAnchorBase,intervals};
}
export function activeDispatchArgs(count:number,stride:number):readonly[number,number,number]{
 if(!Number.isSafeInteger(count)||count<0||!Number.isSafeInteger(stride)||stride<1)throw Error('Invalid active dispatch extent');
 const groups=Math.ceil(count/64),x=Math.min(groups,stride),y=Math.ceil(groups/Math.max(1,x));return [x,y,1];
}
const ARTICULATION_PARAMETERS=/* wgsl */`struct Params {extent:vec4u,rect:vec4f,field:vec4f,worldFromClip:mat4x4f,playback:vec4u,proposals:vec4u,farLoops:vec4u,queue:vec4u,partAnchors:vec4u}`;
const ACTIVE_QUEUE_CONSTANTS=/* wgsl */`
const QUEUE_MEMO=0u;const QUEUE_PATH=1u;const QUEUE_PLANAR=2u;const QUEUE_ROUND=3u;
const QUEUE_FAR_PREPARE=4u;const QUEUE_FAR_TERRAIN=5u;const QUEUE_FAR_STANDING=6u;
const QUEUE_LABEL=7u;const QUEUE_TONE=8u;const ACTIVE_QUEUE_FAMILIES=9u;
`;
const ACTIVE_QUEUE_WGSL=/* wgsl */`${ACTIVE_QUEUE_CONSTANTS}
fn queueBase(family:u32)->u32{return atomicLoad(&counts[p.queue.x+family*2u]);}
fn queueCount(family:u32)->u32{return atomicLoad(&counts[p.queue.x+family*2u+1u]);}
fn scheduledRunFamily(id:vec3u,family:u32)->u32{
 let ordinal=id.x+id.y*p.playback.y*64u;
 if(p.queue.x==0u){return ordinal;}
 let queued=queueCount(family);if(ordinal>=queued){return 0xffffffffu;}
 return atomicLoad(&counts[queueBase(family)+ordinal]);
}
fn scheduledRun(id:vec3u)->u32{return scheduledRunFamily(id,QUEUE_MEMO);}
fn scheduledRunWorkgroup(group:vec3u,family:u32)->u32{
 let queued=select(p.extent.z,queueCount(family),p.queue.x!=0u);
 let rows=max(1u,(queued+p.proposals.w-1u)/p.proposals.w);
 let columns=max(1u,(queued+rows-1u)/rows);
 let ordinal=group.x+group.y*columns;if(ordinal>=queued){return 0xffffffffu;}
 if(p.queue.x==0u){return ordinal;}
 return atomicLoad(&counts[queueBase(family)+ordinal]);
}
// Labels are seeded into the native row table and tone members are emitted by
// toneSeed; these consumers still validate the typed producer ranges instead
// of treating their queues as allocation-only bookkeeping.
fn typedQueueActive(runID:u32,kind:u32)->bool{
 if(p.queue.x==0u){return true;}
 if(kind==2u){let base=p.queue.y;let count=queueCount(QUEUE_LABEL);return runID>=base&&runID<base+count;}
 if(kind==3u){let base=p.queue.y+p.queue.z;let count=queueCount(QUEUE_TONE);return runID>=base&&runID<base+count;}
 return true;
}
`;
/** Compact the resident source-run slots into the existing correspondence
 * scratch. Source run IDs stay the values in the queue, so memo keys,
 * contribution records and diagnostics keep their original addresses. Labels
 * and tone slots remain typed tails: labels are always active, while tone is
 * bounded by the producer's count header, not its allocation capacity. */
export const ACTIVE_RUNS_SHADER=/* wgsl */`${ARTICULATION_PARAMETERS}${ACTIVE_QUEUE_CONSTANTS}
// ResidentConstructionRuns is five vec4u records (range, identity, part,
// paper, source correspondence), 80 bytes per source slot. Keeping the
// complete ABI here preserves the old fields while carrying source ownership.
struct Run {range:vec4u,identity:vec4u,part:vec4u,paper:vec4u,correspondence:vec4u}
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read> runs:array<Run>;
@group(0) @binding(2) var<storage,read_write> counts:array<atomic<u32>>;
@group(0) @binding(3) var<storage,read> tone:array<u32>;
fn queueBase(family:u32)->u32{return atomicLoad(&counts[p.queue.x+family*2u]);}
fn appendQueue(family:u32,id:u32,capacity:u32){
 let countAt=p.queue.x+family*2u+1u;var slot=atomicLoad(&counts[countAt]);
 loop{
  if(slot>=capacity){return;}
  let claim=atomicCompareExchangeWeak(&counts[countAt],slot,slot+1u);
  if(claim.exchanged){atomicStore(&counts[queueBase(family)+slot],id);return;}
  slot=claim.old_value;
 }
}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3u){
 let groups=max(1u,(p.extent.z+63u)/64u);
 let rows=max(1u,(groups+p.proposals.w-1u)/p.proposals.w);
 let columns=max(1u,(groups+rows-1u)/rows);
 let i=id.x+id.y*columns*64u;if(i>=p.extent.z){return;}
 let sourceCount=p.queue.y;let labelCount=p.queue.z;let toneCapacity=p.queue.w;
 if(i<sourceCount){
  if(runs[i].range.y==0u){return;}
  // Anchor payload occupies the same raw range word as boundary flags. The
  // playback decoder treats an anchor as a path, so compaction must clear the
  // payload before selecting the specialized planar/round families too.
  let raw=runs[i].range.z;let anchor=(raw&0x10000u)!=0u;
  let boundary=select(raw&0x4120000u,0u,anchor);
  appendQueue(QUEUE_MEMO,i,sourceCount);appendQueue(QUEUE_FAR_PREPARE,i,sourceCount);
  let family=select(select(QUEUE_PATH,QUEUE_ROUND,(boundary&0x20000u)!=0u),QUEUE_PLANAR,(boundary&0x4100000u)!=0u);
  appendQueue(family,i,sourceCount);
  let farKind=atomicLoad(&counts[p.farLoops.w+i*8u]);
  if(farKind==1u){appendQueue(QUEUE_FAR_TERRAIN,i,sourceCount);}
  if(farKind==3u){appendQueue(QUEUE_FAR_STANDING,i,sourceCount);}
  return;
 }
 if(i<sourceCount+labelCount){appendQueue(QUEUE_LABEL,i,labelCount);return;}
 let toneIndex=i-sourceCount-labelCount;if(toneIndex<min(tone[0],toneCapacity)){appendQueue(QUEUE_TONE,i,toneCapacity);}
}`;
export const ACTIVE_RUN_ARGS_SHADER=/* wgsl */`${ARTICULATION_PARAMETERS}${ACTIVE_QUEUE_CONSTANTS}
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read> counts:array<u32>;
@group(0) @binding(2) var<storage,read_write> arguments:array<u32>;
fn queueCount(family:u32)->u32{return counts[p.queue.x+family*2u+1u];}
fn writeArgs(family:u32,offset:u32,standing:bool){
 let count=queueCount(family);let groups=(count+63u)/64u;
 if(standing){let rows=max(1u,(count+p.proposals.w-1u)/p.proposals.w);let columns=max(1u,(count+rows-1u)/rows);arguments[offset]=columns;arguments[offset+1u]=rows;}
 else{let x=min(groups,p.playback.y);let divisor=max(1u,x);arguments[offset]=x;arguments[offset+1u]=(groups+divisor-1u)/divisor;}
 arguments[offset+2u]=1u;
}
@compute @workgroup_size(1) fn main(){
 for(var family=0u;family<ACTIVE_QUEUE_FAMILIES;family++){
  let offset=family*3u;if(family==QUEUE_FAR_STANDING){writeArgs(family,offset,true);}else{writeArgs(family,offset,false);}
 }
}`;
/** Dispatch only the active proposal prefix without mapping its count. The
 * indirect arguments have their own buffer: binding the writable proposal
 * store as an indirect buffer in the same pass would alias usage scopes. */
export const WORD_VISIBILITY_DISPATCH_SHADER=/* wgsl */`
${ARTICULATION_PARAMETERS}
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read> counts:array<u32>;
@group(0) @binding(2) var<storage,read_write> arguments:array<u32>;
@compute @workgroup_size(1) fn main(){
 let groups=(min(counts[p.proposals.z],p.proposals.y)+63u)/64u;
 arguments[0]=min(groups,p.proposals.w);arguments[1]=max(1u,(groups+p.proposals.w-1u)/p.proposals.w);arguments[2]=1u;
}`;


/** Source-owned region records join the SAME native priority/cursor pass.
 * This merely transports the tone producer's decisions, not an image fit,
 * CPU member readback, second page, or a post-publication tone overlay. */
export const GPU_TONE_SEED_SHADER=/* wgsl */`
struct P {extent:vec4u,seed:vec4u,rect:vec4f,sources:vec4u}
struct Run {range:vec4u,identity:vec4u,part:vec4u,paper:vec4u,correspondence:vec4u}
struct Word {data:array<u32,${ROW_WORDS}>}
@group(0) @binding(0) var<uniform> p:P;
@group(0) @binding(1) var<storage,read> tone:array<u32>;
@group(0) @binding(2) var<storage,read_write> runs:array<Run>;
@group(0) @binding(3) var<storage,read_write> rowWords:array<Word>;
@group(0) @binding(4) var<storage,read_write> counts:array<atomic<u32>>;
@group(0) @binding(5) var<storage,read> domains:array<u32>;
// The grammar's whole-word table: region passages are its words toneWords..
@group(0) @binding(6) var<storage,read> words:array<u32>;
const WS=${WORD_STRIDE}u;
fn retained(domain:u32)->bool{
 var lo=0u;var hi=p.sources.x;
 while(lo<hi){let mid=(lo+hi)/2u;if(domains[mid]<domain){lo=mid+1u;}else{hi=mid;}}
 return lo<p.sources.x&&domains[lo]==domain;
}
// CANONICAL SEED (2026-11-12). The member list arrives in canonical order
// (tone-field.ts TONE_ORDER_SHADER: rows ascending, each row by its surface
// key). The thread of a row's first member seeds that row's members in list
// order; the others return. A row's words, their slots after its authored
// literal words, and every run index (extent.w + member) are then the same
// however the producer's threads were scheduled.
@compute @workgroup_size(64) fn seedTone(@builtin(global_invocation_id) id:vec3u){
 let first=id.x+id.y*p.seed.w*64u;let total=min(tone[0],p.seed.x);if(first>=total){return;}
 let row=tone[4u+first*11u+1u];
 if(first>0u&&tone[4u+(first-1u)*11u+1u]==row){return;}
 for(var member=first;member<total&&tone[4u+member*11u+1u]==row;member++){seedMember(member);}
}
fn seedMember(member:u32){
 let at=4u+member*11u;let row=tone[at+1u];
 // Member column 2 is the region's passage index (tone-field.ts
 // TONE_PASSAGES); the word is that whole retained passage, one constant
 // pattern per region (D6), placed at the producer's centre.
 let passage=tone[at+2u];if(passage==0u||passage>=p.sources.y){return;}
 let font=p.seed.y+passage-1u;let count=words[font*WS];let advance=words[font*WS+1u];
 let x=i32(round(bitcast<f32>(tone[at])-f32(advance)*.5));
 let rows=(p.extent.y+16u)/17u;
 if(row>=rows||x<0||u32(x)+advance>p.extent.x||row*17u+17u>p.extent.y){return;}
 if(f32(x)<p.rect.x||f32(x)+f32(advance)>p.rect.x+p.rect.z||f32(row*17u)<p.rect.y||f32(row*17u+17u)>p.rect.y+p.rect.w){return;}
 let run=p.extent.w+member;let source=tone[at+8u];let owner=tone[at+9u];
 // Field-only producers have not acquired a construction/source lifetime.
 // Their dark pixels cannot establish a new dots-only depicted owner, or
 // invalidate an otherwise publishable frame with untracked provenance.
 // The field's owner carries the identity (bit 31), raw-producer (bit 30)
 // and UV-contract (bit 29) flags; the retained source domain has none.
 // tone2: a particle word (tone-field.ts particleCell) is a producer row of
 // domain 0 (source 0x80000000, owner 0); no surface word has domain 0, so
 // only it passes unretained.
 if(!retained(source&0x1fffffffu)&&!(source==0x80000000u&&owner==0u)){return;}
 let key=(source*0x9e3779b9u)^(owner*0x85ebca6bu)^tone[at+4u]^(tone[at+5u]*0xc2b2ae35u)^tone[at+6u]^tone[at+7u];
 runs[run]=Run(vec4u(0),vec4u(key,source&0xdfffffffu,owner,key),vec4u(0),vec4u(0,0,0,0x08000000u|(tone[at+2u]<<24u)),vec4u(0));
 let slot=atomicAdd(&counts[row],1u);atomicMax(&counts[rows+3u],slot+1u);
 let limit=atomicLoad(&counts[p.sources.z+rows+row]);
 if(slot>=limit){atomicOr(&counts[rows],1u);return;}
 var word:Word;word.data[0]=run;word.data[3]=u32(x);word.data[4]=row;word.data[5]=font;word.data[6]=count;word.data[7]=advance;
 word.data[9]=3u;word.data[10]=tone[at+10u];word.data[11]=tone[at+3u];word.data[12]=1u;
 word.data[14]=tone[at+4u];word.data[15]=tone[at+5u];word.data[16]=tone[at+6u];word.data[17]=tone[at+7u];
 // A changed light band is a legitimate region-context change, not a
 // retained old spelling merely because its source lattice node survives.
 rowWords[atomicLoad(&counts[p.sources.z+row])+slot]=word;
}
`;
const MEMO_HASH=/* wgsl */`
fn memoHash(key:array<u32,5>)->u32{
 var hash=0x9e3779b9u;
 for(var i=0u;i<5u;i++){hash=(hash^key[i])*0x85ebca6bu;hash=hash^(hash>>16u);}
 return hash;
}
`;
/** Readdress retained SOURCE keys on the GPU when storage grows. This is
 * integer dictionary maintenance, never a glyph/pen/fit decision. */
export const MEMO_REHASH_SHADER=/* wgsl */`
struct Params {oldSlots:u32,newSlots:u32,base:u32,columns:u32,domains:u32,epochs:u32,producers:u32,flags:u32}
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read> before:array<u32>;
@group(0) @binding(2) var<storage,read_write> after:array<atomic<u32>>;
@group(0) @binding(3) var<storage,read> lifetimes:array<u32>;
${MEMO_HASH}
fn hasLifetime(first:u32,count:u32,value:u32)->bool{
 var lo=0u;var hi=count;
 while(lo<hi){let mid=(lo+hi)/2u;if(lifetimes[first+mid]<value){lo=mid+1u;}else{hi=mid;}}
 return lo<count&&lifetimes[first+lo]==value;
}
@compute @workgroup_size(64) fn rehash(@builtin(global_invocation_id) id:vec3u){
 let slot=id.x+id.y*p.columns*64u;if(slot>=p.oldSlots){return;}
 let old=p.base+slot*${MEMO_WORDS}u;if(before[old]!=3u){return;}
 let key=array<u32,5>(before[old+1u],before[old+2u],before[old+3u],before[old+4u],before[old+5u]);
 if((p.flags&1u)!=0u&&!hasLifetime(0u,p.domains,key[0]&0x5fffffffu)){return;}
 if((key[0]&0x80000000u)!=0u){
  if(p.producers!=0xffffffffu&&!hasLifetime(p.domains+p.epochs,p.producers,key[1])){return;}
 }else if((p.flags&2u)!=0u){
  if(key[1]>=p.epochs||key[4]==0xffffffffu||lifetimes[p.domains+key[1]]!=key[4]){return;}
 }
 let hash=memoHash(key);
 for(var probe=0u;probe<p.newSlots;probe++){
  let destination=p.base+((hash+probe)&(p.newSlots-1u))*${MEMO_WORDS}u;
  let claim=atomicCompareExchangeWeak(&after[destination],0u,1u);
  if(!claim.exchanged){if(claim.old_value==0u){probe--;}continue;}
  for(var word=1u;word<${MEMO_WORDS}u;word++){atomicStore(&after[destination+word],before[old+word]);}
  atomicAdd(&after[p.base-1u],1u);
  atomicStore(&after[destination],before[old]);return;
 }
}
`;

export {NEAR_FROM_ROWS,FAR_STRIP_BELOW_ROWS};



















































































export const CHROMA_LEAST=.45,INK_LUMA=.35,INK_VALUE=.85;
export function materialInk(rgb:readonly number[]):number{
 const c=[0,1,2].map(i=>Math.max(0,rgb[i]??0)),top=Math.max(...c);
 if(!(top>.002)||!Number.isFinite(top))return 0;
 const hue=c.map(v=>v/top);if(1-Math.min(...hue)<CHROMA_LEAST)return 0;
 const luma=hue[0]!*.2126+hue[1]!*.7152+hue[2]!*.0722,k=Math.min(INK_VALUE,INK_LUMA/Math.max(luma,.0001));
 const [r,g,b]=hue.map(v=>Math.round(Math.min(1,v*k)*255));
 return ((255<<24)|(b!<<16)|(g!<<8)|r!)>>>0;
}
/** Sorted (entity, ink) pairs for binary search, pair 0 = (count, 0); entity
 * is the field owner's second word (the engine's u32 eid). Uploaded as an
 * rg32uint texture (MATERIAL_INK_WIDTH texels per row): the articulation
 * stage already uses every storage-buffer slot a stage may hold (8). */
export const MATERIAL_INK_WIDTH=1024;
export function materialInkTable(pairs:Iterable<readonly [number,number]>):Uint32Array{
 const sorted=[...pairs].filter(([,ink])=>ink!==0).sort((a,b)=>(a[0]>>>0)-(b[0]>>>0));
 const out=new Uint32Array(2+sorted.length*2);out[0]=sorted.length;
 sorted.forEach(([entity,ink],i)=>{out[2+i*2]=entity>>>0;out[3+i*2]=ink>>>0;});
 return out;
}
/** The table's texture: width/height and the texel data padded to whole rows. */
export function materialInkTexels(table:Uint32Array):{width:number;height:number;data:Uint32Array}{
 const texels=Math.max(1,table.length/2),width=Math.min(MATERIAL_INK_WIDTH,texels),height=Math.ceil(texels/width);
 const data=new Uint32Array(width*height*2);data.set(table.length?table:[0,0]);
 return {width,height,data};
}
/** Needs `materialInks:texture_2d<u32>` (rg32uint) bound by the including shader. */
export const MATERIAL_INK_WGSL=/* wgsl */`
fn materialInkPair(i:u32)->vec2u{let w=textureDimensions(materialInks).x;return textureLoad(materialInks,vec2i(i32(i%w),i32(i/w)),0).xy;}
fn materialInk(owner:vec2u)->u32{
 // A producer row (owner bit 31) has no entity material, nor has a far

 // owner word is an octree or column key, not an entity).
 if((owner.x&0x80000000u)!=0u||owner.x==0u){return 0u;}
 let n=materialInkPair(0u).x;var lo=0u;var hi=n;
 while(lo<hi){let mid=(lo+hi)/2u;if(materialInkPair(1u+mid).x<owner.y){lo=mid+1u;}else{hi=mid;}}
 if(lo<n){let pair=materialInkPair(1u+lo);if(pair.x==owner.y){return pair.y;}}
 return 0u;
}
`;
export const DISTANCE_TIER_WGSL=/* wgsl */`
${FORM_DEPTH_WGSL}
// Near from NEAR_FROM_ROWS rows, far under FAR_STRIP_BELOW_ROWS (an unread
// count, NaN, is near).
fn tierOfRows(rows:f32)->u32{return select(select(1u,2u,rows<${FAR_STRIP_BELOW_ROWS}.0),0u,!(rows<${NEAR_FROM_ROWS}.0));}
// THE ONE LEVEL READ of every producer: SCREEN-SIZE LOD. owner: the owner
// field's id pair. terrain: the owner is a terrain source. extent: the field
// size. row: one text row in field pixels. 0 = 近景, 1 = 中景, 2 = 遠景.
// Parallel and enclosed views are near; a compact form reads the rows it
// occupies on the page. No world distance, no reference length, no line

// was a guess).

// (header texel w, form-depths.ts resolveLod).
fn formLodActive()->bool{return textureLoad(formDepths,vec2i(0,${FORM_DEPTH_HEADER_ROW}),0).w!=0u;}
fn distanceTierOf(owner:vec2u,terrain:bool,f:vec2f,depth:f32,extent:vec2f,row:f32,worldFromClip:mat4x4f)->u32{
 let m=worldFromClip;
 if(!viewRecedes(m)){return 0u;}
 if(!terrain){
  let form=formExtent(owner);
  if(form.found&&!formSpansDepth(form.near,form.far,extent,m)){
   let rows=(form.bottom-form.top)/row;

   // selection of game engines, the line-density control of Cole et al. 2006
   // and Markosian et al. 2000): a compact form's level is the rows it
   // occupies on the page and nothing else. No world line, no reference
   // length. Every level draws the form's OWN edges, fewer of them (the
   // selection table is the fixed priority order, and each level's lines are
   // a subset of the level above); see farKind for the one merge.

   if(!formCut(form.box,extent)){if(formLodActive()){return formHeldLod(owner,true);}return tierOfRows(rows);}
  }
 }
 // A SURFACE LARGER THAN THE PAGE SHOWS (terrain, an owner spanning depth, an
 // owner the page cuts, an owner without a slot) is not one form and holds
 // no level of its own. Its TONE is one field on its surface lattice, whose
 // dyadic node spacing is already the continuous screen-space level (rows
 // close up as the surface recedes; nothing switches). Its LINES are each a
 // form by their own page size (distanceTier). So this read is near.
 return 0u;
}
// The owner is such a surface (see distanceTierOf), in a receding view.
fn spanningSurface(owner:vec2u,terrain:bool,extent:vec2f,worldFromClip:mat4x4f)->bool{
 if(!viewRecedes(worldFromClip)){return false;}
 if(terrain){return true;}
 let form=formExtent(owner);
 return !(form.found&&!formSpansDepth(form.near,form.far,extent,worldFromClip)&&!formCut(form.box,extent));
}
// The view recedes to a horizon: a perspective projection of an open view
// (not parallel, not enclosed). Otherwise everything is 近景.
fn viewRecedes(m:mat4x4f)->bool{return abs(m[0][3])+abs(m[1][3])+abs(m[2][3])>1e-6*abs(m[3][3])&&formOpen();}
`;
/** Page width up to which a far form is ONE mark: two cells of the strip's
 * 16 px lattice. Three cells are the smallest profile (shoulder, peak,
 * shoulder), so a wider form is spelled cell by cell. */
export const FAR_MARK_WIDTH=32;
/** Rows an occluding contour inside a soft form must run to be drawn at the
 * outline tier (tierSelects): two, the least a unit-plus-shift run occupies. */
export const FAR_CONTOUR_LEAST_ROWS=2;
/** MERGED FAR SKYLINE profile (farSkyline): three tables of SKY_COLUMNS words,
 * one word per column of the strip's 16 px page lattice, in the free header
 * words of the persistent per-camera correspondence buffer (binding 7) from
 * word SKY_AT; the row counts before them end at rows + 4 and the memo use
 * counter is word MEMO_AT - 1. 512 columns are the 8192 px the compiled cursor
 * table allows (GpuArticulation maxWidth). Cleared in encoder order every
 * frame, written by skylineReduce, read by featurecurves; no buffer is made. */
export const SKY_AT=512,SKY_COLUMNS=512;
/** A line class that is not drawn at a tier. */
export const TIER_NEVER=0xffffffff;
/** The run's hard bit (0x40000) separates a principal crease from a facet
 * seam: the construction runs set it on a crease only for a compiled authored
 * feature edge (edge class 1), never for a facet crease of a triangle soup
 * (edge class 2, gpu-construction-runs.ts hardEdge). */
export const PRINCIPAL_CREASE_FROM_HARD_BIT=true;

























export const TIER_SELECTION:readonly (readonly number[])[]=[
 [0,0,0,0,2,0], // 近景: the full graph; a facet crease only across a tone change; interior curves of two rows or more
 [0,0,0,1,3,0], // 中景: outline, contacts, every principal crease, and a facet crease of one row or more that is a plane boundary (tierSelects: a tone-band change or a 45 degree turn)
 [0,0,TIER_NEVER,1,TIER_NEVER,0], // 遠景: the connected outline, its principal creases and a plane boundary of one row or more (second in the priority order; a two-row rock is its outline and its top edge)
];



















export const TERRAIN_SELECTION:readonly (readonly number[])[]=[
 [0,0,0,TIER_NEVER,1,3], // 近景: skyline and ridge silhouettes, interior turns of 1 row or more (a fold one glyph long on the magenta line draws; line repair 2026-11-12, was 2), principal slope turns, contacts
 [0,0,0,TIER_NEVER,2,4], // 中景: ridge outlines, interior turns of 2 rows or more (was 3), shelf edges, contacts
 [0,0,TIER_NEVER,TIER_NEVER,TIER_NEVER,TIER_NEVER], // 遠景: the outline against sky or a farther form (far strip); its base line is the ground line
];
/** cos 45°: the dihedral turn from which a terrain crease is a slope turn, the
 * same turn at which surface-relations.ts emits a crease at all; read here on
 * the frame's own normals so a smooth-shaded seam does not qualify. */
export const TERRAIN_TURN_COS=.70710678;
/** World-up component of a shelf's top face normal (within about 37° of level). */
export const TERRAIN_SHELF_TOP=.8;
/** World-up component at or below which the other face is the shelf's drop (60° or steeper). */
export const TERRAIN_SHELF_FACE=.5;
/** A path-led limb whose two silhouettes lie between these page widths in one
 * row is ONE paired flank column (compiled-role-grammar.ts pairedFlanks).
 * Narrower is one stroke; wider keeps two independent outlines. The far edge
 * is read in LIMB_PAIR_STEPS steps of 4 page px. */
export const LIMB_PAIR_LEAST=9,LIMB_PAIR_STEPS=32;

/** Unit ordinal of the smallest-tier whole-form glyph of a planar volume (口 ロ
 * ﾛ, hardBoundaryConstruction). It is the form, not a join between runs, so
 * the typesetter gives it D1(1), its nearest legal pen, and a second stamp of
 * the same word on the same cells is one shared mark (typesetWord). */
export const STAMP_ORDINAL=0x7fe00000;
/** Ports of old MAIN featurecurves/featureplace. Expressive decisions are
 * immutable grammar records. A run looks up its instruction ONCE; its units
 * carry the same word through the complete source parameterization. */
const ARTICULATION_SOURCE=/* wgsl */`

const WS=${WORD_STRIDE}u;
${ARTICULATION_PARAMETERS}
struct Run {range:vec4u,identity:vec4u,part:vec4u,paper:vec4u,correspondence:vec4u}
// range.w is a compact source-program packet when a producer supplied one:
// bits 0..1 remain the incumbent size band, bits 2..3 the source role, bit 4
// the source D5 sequence flag, and bits 5..31 the nonzero source feature key.
// Legacy runs retain the old size-only word and therefore have no feature key.
fn runSize(run:Run)->u32{return run.range.w&3u;}
fn runProgramFeature(run:Run)->u32{return run.range.w>>5u;}
// A source-program feature token is [9-bit source context | 18-bit packet
// ordinal].  The whole token remains the correspondence feature identity;
// only this immutable context prefix selects a retained native entry.
fn runProgramContext(run:Run)->u32{
 // Source parents retain their packet feature identity, but an admitted
 // passage may have another legal side than the packet's first contour.
 // sourceParts carries that admitted context in part.y for every consumer.
 if((run.range.z&0x80000000u)!=0u&&runProgramFeature(run)!=0u){return run.part.y;}
 return runProgramFeature(run)>>18u;
}
fn runProgramRole(run:Run)->u32{return (run.range.w>>2u)&3u;}
fn runProgramD5(run:Run)->bool{return runProgramFeature(run)!=0u&&(run.range.w&16u)!=0u;}
fn runRole(run:Run)->u32{return select(run.range.z&255u,runProgramRole(run),runProgramFeature(run)!=0u);}
${CONSTRUCTION_POINT_WGSL}
struct Glyph {scalar:u32,advance:u32,first:u32,count:u32}
struct Word {data:array<u32,${WORD_WORDS}>}
// Candidate rows reference resident run provenance instead of copying the
// same source/endpoint packet into every repeated unit. Only published
// words expand to the self-contained export record.
struct RowWord {data:array<u32,${ROW_WORDS}>}
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read> runs:array<Run>;
@group(0) @binding(2) var<storage,read> points:array<Point>;
@group(0) @binding(3) var<storage,read> grammar:array<u32>;
@group(0) @binding(4) var<storage,read> words:array<u32>;
@group(0) @binding(6) var<storage,read> gaps:array<u32>;
@group(0) @binding(7) var<storage,read_write> counts:array<atomic<u32>>;
@group(0) @binding(8) var<storage,read_write> rowWords:array<RowWord>;
@group(0) @binding(9) var<storage,read_write> output:array<u32>;
@group(0) @binding(10) var depthField:texture_2d<f32>;
@group(0) @binding(11) var normalField:texture_2d<f32>;
@group(0) @binding(12) var ownerField:texture_2d<u32>;
@group(0) @binding(13) var radianceField:texture_2d<f32>;
// The field's per-form nearest-depth table (form-depths.ts FormDepths),
// read-only here and in the tone producer through distanceTierOf.
@group(0) @binding(5) var formDepths:texture_2d<u32>;
// STACKED CAMERAS: page cells (x, glyph row) covered by a camera stacked above
// this one (its rendered coverage and its glyph cells, gpu-articulation
// occluderFor). A 1x1 zero texture when nothing is stacked above.
@group(0) @binding(14) var occluder:texture_2d<u32>;
// The frame's material inks (MATERIAL INK; materialInkTable/materialInkTexels): rg32uint (entity, ink) pairs, pair 0 = (count, 0).
@group(0) @binding(15) var materialInks:texture_2d<u32>;
${ACTIVE_QUEUE_WGSL}
fn occludedCell(x:i32,row:i32)->bool{let d=vec2i(textureDimensions(occluder));if(x<0||row<0||x>=d.x||row>=d.y){return false;}return textureLoad(occluder,vec2i(x,row),0).x!=0u;}
${D10_LINE_CLASSES}
${MEMO_HASH}
${MATERIAL_INK_WGSL}
// A line word's ink is its own FORM's material ink (examples-89 rework,
// orchestrator review: a line's colour comes from its own form's material,
// never from the lit colour of the facet beside it or a neighbouring form).
// One table read by the run's owner entity, so one form's outline is one
// hue. Black when the form has no chromatic material (materialInk 0).
fn lineInk(runID:u32,x:i32,row:i32,advance:u32)->u32{
 let identity=runs[runID].identity;
 let ink=materialInk(vec2u(identity.y,identity.z));
 return select(0xff000000u,ink,ink!=0u);
}
const NONE=0xffffffffu;
const STAMP_ORDINAL=${STAMP_ORDINAL}u;
fn rowCount()->u32{return (p.extent.y+16u)/17u;}
// Prefix addresses and capacities belong to THIS page's allocation, not to
// its drawing decisions. The table immediately precedes the proposal store.
fn rowLayoutBase()->u32{return p.proposals.x-rowCount()*2u;}
fn rowBase(row:u32)->u32{return atomicLoad(&counts[rowLayoutBase()+row]);}
fn rowLimit(row:u32)->u32{return atomicLoad(&counts[rowLayoutBase()+rowCount()+row]);}
// The line keeps its root identity for whole-construction correspondence.
// Only surface/visibility reads use the interval's actual contributor.
// A tagged handle is same-frame storage, never an exported feature address.
fn spanSource(run:Run,context:u32)->Run{
 var source=run;
 if((run.range.z&0x10010000u)!=0x10000000u||context==NONE||(context&0x80000000u)==0u){return source;}
 let slot=context&0x7fffffffu;if(slot>=arrayLength(&runs)){return source;}
 let record=runs[slot];
 if(record.range.y==0u&&record.identity.w==${SPAN_SOURCE_TAG}u){source.identity=vec4u(run.identity.x,record.identity.yz,run.identity.w);}
 return source;
}
fn wordHasSpanSource(word:Word)->bool{
 // Bit24 is an output-only tag for non-anchor terrain words. Anchor direction
 // flags occupy the same high bits, and tone22/23 are region coordinates.
 return word.data[9]!=2u&&word.data[9]!=3u&&(word.data[20]&(0x10010000u|${NATIVE_SPAN_SOURCE}u))==(0x10000000u|${NATIVE_SPAN_SOURCE}u);
}
fn rowSource(word:RowWord)->vec4u{
 let run=runs[word.data[0]];
 if(word.data[9]==2u){return vec4u(0);}
 if(word.data[9]==3u){return vec4u(run.identity.yz,0u,run.paper.w);}
 let anchor=(run.range.z&0x10000u)!=0u;
 return vec4u(run.identity.yz,select(run.part.x,run.paper.z,anchor),select(run.part.y,run.paper.w,anchor));
}
fn expandWord(row:RowWord)->Word{
 var word:Word;for(var i=0u;i<16u;i++){word.data[i]=row.data[i];}
 let source=rowSource(row);for(var i=0u;i<4u;i++){word.data[16u+i]=source[i];}
 let run=runs[row.data[0]];word.data[22]=row.data[16];word.data[23]=row.data[17];
 if(row.data[9]==3u){
  word.data[24]=bitcast<u32>(f32(row.data[3]));word.data[25]=bitcast<u32>(f32(row.data[4]*17u));word.data[26]=word.data[24];word.data[27]=word.data[25];
 }else if(row.data[9]!=2u){
  word.data[20]=run.range.z;word.data[21]=run.identity.w;
  if(row.data[9]==0u&&row.data[16]!=0u){
   let source=spanSource(run,row.data[16]);word.data[22]=source.identity.y;word.data[23]=source.identity.z;
   word.data[20]|=${NATIVE_SPAN_SOURCE}u;
  }
  if((run.part.y&0x1fffffffu)==4u){for(var i=0u;i<4u;i++){word.data[28u+i]=run.paper[i];}}
  if(run.range.y>0u){let a=finiteRunStart(run).position;let b=finiteRunEnd(run).position;
   word.data[24]=bitcast<u32>(a.x);word.data[25]=bitcast<u32>(a.y);word.data[26]=bitcast<u32>(b.x);word.data[27]=bitcast<u32>(b.y);}
 }
 return word;
}
fn interpolatePoint(a:Point,b:Point,t:f32)->vec4f{
 // Screen XY and post-divide depth interpolate affinely. The source
 // parameter does not: retain its exact perspective mapping through cuts,
 // rather than treating native-page progress as progress along the mesh.
 var point=mix(a.position,b.position,t);
 let weights=vec2f(1.0-t,t)*vec2f(bitcast<f32>(a.visible.y),bitcast<f32>(b.visible.y));
 let denominator=weights.x+weights.y;
 if(denominator>0.0){point.w=dot(weights,vec2f(pointSourceParameter(a),pointSourceParameter(b)))/denominator;}
 return point;
}
${POINT_SOURCE_CHART_WGSL}
// Face packets begin with a non-drawable homogeneous descriptor pair. Generic
// consumers must skip only that tagged pair; an ordinary invisible Point is
// still a legitimate finite endpoint and must remain part of the run. These
// accessors therefore remove descriptor interiors without applying visibility
// as an endpoint predicate.
fn finiteRunStart(run:Run)->Point{
 for(var i=0u;i<run.range.y;i++){let point=points[run.range.x+i];if(!pointWholeEdgeDescriptor(point)){return point;}}
 return Point(vec4f(0.0),vec4u(0u));
}
fn finiteRunEnd(run:Run)->Point{
 for(var offset=0u;offset<run.range.y;offset++){
  let i=run.range.y-1u-offset;let point=points[run.range.x+i];if(!pointWholeEdgeDescriptor(point)){return point;}
 }
 return Point(vec4f(0.0),vec4u(0u));
}
fn nativeGap(wanted:u32,leading:bool)->u32{return gaps[select(0u,p.extent.x+1u,leading)+min(wanted,p.extent.x)];}
fn boundedGap(wanted:u32,budget:u32,leading:bool)->u32{
 let desired=min(wanted,budget);let t=grammar[5];let side=t*2u+1u;
 let at=(p.extent.x+1u)*2u+(select(0u,side,leading)+min(desired,t*2u))*side+min(budget-desired,t*2u);
 let step=gaps[at];if(step==0u){return NONE;}return u32(i32(desired)+i32(step)-i32(t)-1);
}
fn fieldPixel(q:vec2f)->vec2i{return vec2i((q-p.rect.xy)/p.rect.zw*p.field.xy);}
fn paperStride()->u32{return (p.extent.x+31u)/32u;}
fn memoMapBase()->u32{return u32(p.field.z)+u32(p.field.w)*${MEMO_WORDS}u;}
fn epochBase()->u32{return memoMapBase()+p.extent.z;}
fn paperBase()->u32{return epochBase()+p.playback.z;}
// Per source owner: bit 31 = a visible run exists; low bits = committed words.
// The frame's true state names every visible owner that drew nothing.
fn drawnBase()->u32{return paperBase()+paperStride()*rowCount();}
fn drawnSlot(run:Run)->u32{return drawnBase()+min(run.identity.z,p.playback.z-1u);}
// Programmed runs carry source correspondence in the producer-owned tail:
// [original subfeature, source port-a, source port-b, source-program lineage].
// The source feature is range.w>>5. The hash below is only a probe bucket; the
// exact tuple is stored in the memo slot's bounded correspondence tail and is
// compared before a slot is reused. No projected/visibility-fragment value is
// a temporal correspondence identity.
fn programmedCorrespondenceHash(run:Run)->u32{
 let c=run.correspondence;var h=runProgramFeature(run)^(c.x*0x85ebca6bu);
 h^=c.y*0xc2b2ae35u;h^=c.z*0x27d4eb2du;h^=c.w*0x165667b1u;
 h=(h^(h>>16u))*0x7feb352du;return h^(h>>15u);
}
fn programmedCorrespondenceTuple(run:Run)->array<u32,5>{
 return array<u32,5>(runProgramFeature(run),run.correspondence.x,run.correspondence.y,run.correspondence.z,run.correspondence.w);
}
fn memoCorrespondenceMatches(run:Run,at:u32)->bool{
 let wanted=programmedCorrespondenceTuple(run);var same=true;
 for(var word=0u;word<5u;word++){same=same&&atomicLoad(&counts[at+${MEMO_CORRESPONDENCE_AT}u+word])==wanted[word];}
 return same;
}
fn memoKey(run:Run)->array<u32,5>{
 let anchor=(run.range.z&0x10000u)!=0u;
 // Runs name an original relation; anchors name an ORIGINAL JUNCTION.
 // The source pass emits one anchor per junction/family, not one per edge
 // incident on it. Replacing its port with an edge ID duplicates the join.
 let programmed=runProgramFeature(run)!=0u;
 let address=select(select(run.part.x,runProgramFeature(run),programmed),run.paper.z,anchor);
 let rawKind=select(run.part.y,run.paper.w,anchor)&0x1fffffffu;
 // The run builder joins silhouette/occluding segments as ONE outline.
 // That segment label is not a new feature or an articulation lifetime.
 // Normalize the incumbent legacy outline label first. A programmed source
 // relation then owns the complete kind word, including when its raw source
 // relation happened to use the old outline label (1); applying the legacy
 // collapse after the programmed branch would erase that source identity.
 let kind=select(select(rawKind,0u,!anchor&&rawKind==1u),programmedCorrespondenceHash(run),programmed);
 var epoch=select(NONE,0u,(run.identity.y&0x80000000u)!=0u);
 if((run.identity.y&0x80000000u)==0u&&run.identity.z<p.playback.z){epoch=atomicLoad(&counts[epochBase()+run.identity.z]);}
 return array<u32,5>(run.identity.y,run.identity.z,address,kind,epoch);
}
// range.z is a tagged payload: anchors use bits 20..25 for their declared
// incoming/outgoing ports. Bit 20 is a box flag ONLY for non-anchor runs.
// Decode the tag before both memo allocation and producer dispatch. Otherwise
// half the anchor contexts skip allocation, write their memo over row counts,
// and then enter the box producer with a single point instead of drawing a join.
fn wholeBoundaryFlags(run:Run)->u32{
 if((run.range.z&0x10000u)!=0u){return 0u;}
 return run.range.z&0x4120000u;
}
@compute @workgroup_size(64) fn claimMemo(@builtin(global_invocation_id) id:vec3u){
 let runID=scheduledRun(id);if(runID==0xffffffffu||runID>=p.extent.z){return;}
 let map=memoMapBase()+runID;atomicStore(&counts[map],NONE);
 let run=runs[runID];if(run.range.y==0u){return;}
 if(wholeBoundaryFlags(run)!=0u){return;}
 // First correspondence belongs to a drawing, not every hidden source
 // fragment. Existing offscreen entries remain resident and are recovered
 // by the exact key on reveal; this does not evict or rephase them.
 var visible=false;
 for(var point=0u;point<run.range.y;point++){if(pointVisible(points[run.range.x+point])){visible=true;break;}}
 if(!visible){return;}
 let key=memoKey(run);let hash=memoHash(key);let slots=u32(p.field.w);
 for(var attempt=0u;attempt<slots;attempt++){
  var free=NONE;var expected=0u;
  for(var probe=0u;probe<slots;probe++){
   let slot=(hash+probe)&(slots-1u);let at=u32(p.field.z)+slot*${MEMO_WORDS}u;
   let state=atomicLoad(&counts[at]);
   if(state==0u||state==4u){if(free==NONE){free=slot;expected=state;}if(state==0u){break;}continue;}
   // Never wait on an unfinished lane. The resolver reconciles racing
   // aliases once every claimed key has been installed.
   if(state<2u){continue;}
   var same=true;for(var word=0u;word<5u;word++){same=same&&atomicLoad(&counts[at+1u+word])==key[word];}
   if(same&&memoCorrespondenceMatches(run,at)){atomicStore(&counts[map],slot);return;}
  }
  if(free==NONE){break;}
  // Finish the exact lookup BEFORE reusing a tombstone. Otherwise a new
  // empty payload could shadow this key's surviving spelling further down
  // the probe chain, even though the original feature has not changed.
  let used=atomicAdd(&counts[u32(p.field.z)-1u],1u);
  if(used>=slots/2u){atomicSub(&counts[u32(p.field.z)-1u],1u);atomicOr(&counts[rowCount()],4u);return;}
  let at=u32(p.field.z)+free*${MEMO_WORDS}u;
  let claim=atomicCompareExchangeWeak(&counts[at],expected,1u);
  if(claim.exchanged){
   for(var word=0u;word<5u;word++){atomicStore(&counts[at+1u+word],key[word]);}
   let correspondence=programmedCorrespondenceTuple(run);
   for(var word=0u;word<5u;word++){atomicStore(&counts[at+${MEMO_CORRESPONDENCE_AT}u+word],correspondence[word]);}
   atomicStore(&counts[at],2u);atomicStore(&counts[map],free);return;
  }
  atomicSub(&counts[u32(p.field.z)-1u],1u);
 }
 // Capacity, not visibility/admission. Retry this exact source pose after
 // a GPU rehash; never publish a partial cache or substitute another key.
 atomicOr(&counts[rowCount()],4u);
}
@compute @workgroup_size(64) fn resolveMemo(@builtin(global_invocation_id) id:vec3u){
 let runID=scheduledRun(id);if(runID==0xffffffffu||runID>=p.extent.z){return;}
 let map=memoMapBase()+runID;if(atomicLoad(&counts[map])==NONE){return;}
 let run=runs[runID];let key=memoKey(run);let hash=memoHash(key);let slots=u32(p.field.w);
 // A repeated source relation can race its own initial claim. Resolve all
 // such aliases to the FIRST exact key after claims have completed, before
 // any spelling payload is read/written. No spinlocks or trial spellings.
 for(var probe=0u;probe<slots;probe++){
  let slot=(hash+probe)&(slots-1u);let at=u32(p.field.z)+slot*${MEMO_WORDS}u;
  let state=atomicLoad(&counts[at]);if(state<2u||state==4u){continue;}
  var same=true;for(var word=0u;word<5u;word++){same=same&&atomicLoad(&counts[at+1u+word])==key[word];}
  if(same&&memoCorrespondenceMatches(run,at)){if(state==2u){atomicStore(&counts[at],5u);}atomicStore(&counts[map],slot);return;}
 }
 atomicOr(&counts[rowCount()],4u);
}
@compute @workgroup_size(64) fn reclaimMemo(@builtin(global_invocation_id) id:vec3u){
 let slot=id.x+id.y*p.playback.y*64u;if(slot>=u32(p.field.w)){return;}
 let at=u32(p.field.z)+slot*${MEMO_WORDS}u;
 // Resolve marked each canonical new slot 5. Unreferenced race aliases
 // remain 2: reclaim them WITHOUT opening holes in a probe chain. State 4
 // is a reusable tombstone, never a ready key or a spelling payload.
 if(atomicLoad(&counts[at])==2u){atomicStore(&counts[at],4u);atomicSub(&counts[u32(p.field.z)-1u],1u);}
}
// featureplace reads its own row's protected paper (written only by

// paperWord tested every ink pixel of every word with a device atomic load.
var<workgroup> localPaper:array<u32,256>;
var<private> paperLocal:bool=false;
var<private> paperRow:u32=0u;
fn protectedPaper(x:i32,row:i32)->bool{
 if(x<0||x>=i32(p.extent.x)||row<0||row>=i32(rowCount())){return false;}
 if(paperLocal&&u32(row)==paperRow){
  let local=(localPaper[u32(x)/32u]&(1u<<(u32(x)%32u)))!=0u;
  if(PLACE_CHECK){placeCheck(0u);if(local!=((atomicLoad(&counts[paperBase()+u32(row)*paperStride()+u32(x)/32u])&(1u<<(u32(x)%32u)))!=0u)){placeCheck(1u);}}
  return local;
 }
 return (atomicLoad(&counts[paperBase()+u32(row)*paperStride()+u32(x)/32u])&(1u<<(u32(x)%32u)))!=0u;
}
fn paperWord(x:u32,row:u32,word:u32,count:u32,part:u32,sourceScalar:u32)->bool{
 var pen=i32(x);
 for(var scalar=0u;scalar<count;scalar++){
  var glyph=word;if(part==0u||part==3u){glyph=words[word*WS+2u+scalar+select(0u,sourceScalar,part==0u)];}
  let at=grammar[2]+glyph*4u;let first=grammar[at+2u];let pixels=grammar[at+3u];
  for(var pixel=0u;pixel<pixels;pixel++){
   let point=grammar[3]+(first+pixel)*2u;
   // Every protected source span covers its native row, not a screen fill.
   if(protectedPaper(pen+bitcast<i32>(grammar[point]),i32(row))){return true;}
  }
  pen+=i32(grammar[at+1u]);
 }
 return false;
}
// Distance tiers (GOAL-EXECUTOR responsibility 1): HOW MUCH of a visible
// form is drawn, never whether. A fixed rule over the frame's own fields;
// nothing is scored by trying a spelling and no count is budgeted.
const TIER_NEVER=${TIER_NEVER}u;
// Two faces share a tone band while the darker keeps this fraction of the
// brighter: the step between adjacent band ceilings of the tone field.
const TONE_BAND_STEP=.8;
// One full-width glyph: two parallel lines closer than this cannot both be spelled.
const FACET_COLLAPSE=16.0;
struct TierSample {on:bool,d:f32,n:vec3f,luma:f32}
fn fieldPoint(q:vec2f)->vec2f{return (q-p.rect.xy)/p.rect.zw*p.field.xy;}
fn tierSample(run:Run,f:vec2f)->TierSample{
 let q=vec2i(floor(f));
 if(any(q<vec2i(0))||any(q>=vec2i(p.field.xy))){return TierSample(false,0.0,vec3f(0),0.0);}
 let owner=textureLoad(ownerField,q,0).xy;let d=textureLoad(depthField,q,0).x;
 if((owner.x&0xdfffffffu)!=run.identity.y||owner.y!=run.identity.z||d<=0.0){return TierSample(false,0.0,vec3f(0),0.0);}
 let e=textureLoad(normalField,q,0).xy;var n=vec3f(e,1.0-abs(e.x)-abs(e.y));
 if(n.z<0.0){n=vec3f((vec2f(1.0)-abs(n.yx))*select(vec2f(-1.0),vec2f(1.0),n.xy>=vec2f(0)),n.z);}
 return TierSample(true,d,normalize(n),dot(textureLoad(radianceField,q,0).rgb,vec3f(.2126,.7152,.0722)));
}
fn tierWorld(f:vec2f,d:f32)->vec3f{return d10Unproject(f,d,p.field.xy,p.worldFromClip);}
${DISTANCE_TIER_WGSL}
// A run of a terrain source (form 3; gpu-construction-runs.ts assemble).
fn terrainRun(run:Run)->bool{return (run.range.z&0x10010000u)==0x10000000u;}
// THE FORM'S ONE TIER: 0 = 近景 near, 1 = 中景 middle, 2 = 遠景 far, by
// distanceTierOf (the one tier read, shared with the tone producer): every
// run of a compact source owner reads the owner's visible rows from the
// field's form table, so one tree or one house is never drawn with the
// constructions of two tiers. The point this producer asks about, used for
// terrain, a spanning owner and an owner without a slot, is the run's own
// nearest point (largest 1/w): a height surface runs through every distance
// and its form is the ridge or the shelf edge.
fn distanceTier(run:Run)->u32{
 return tierOfRows(formRows(run));
}
// THE ROWS THE FORM HAS, the one measure every tier of a run is read from:
// a compact form's visible rows from the form table (distanceTierOf's read),
// a line of a spanning surface the rows below. 1e9 (near) in a parallel or
// enclosed view.
fn formRows(run:Run)->f32{
 // The geometry owner already measured and held this COMPLETE closed part.
 // Do not override its far level with a spanning source owner's extent.
 if((run.range.z&0x40000000u)!=0u){return 0.0;}
 let terrain=terrainRun(run);
 if(!spanningSurface(run.identity.yz,terrain,p.field.xy,p.worldFromClip)){
  if(!viewRecedes(p.worldFromClip)){return 1.0e9;}
  let form=formExtent(run.identity.yz);
  return (form.bottom-form.top)/(17.0*p.field.y/p.rect.w);
 }
 // A LINE OF A SPANNING SURFACE (a ridge, a shelf edge, a cliff fold, a wall
 // edge) is itself the form: its level is the rows of the longer side of its
 // own visible page box. A ridge across the page is near wherever it lies; a
 // three-row bump on the horizon is far. Same thresholds as every form.
 var low=vec2f(3.402823e38);var high=vec2f(-3.402823e38);var seen=false;
 for(var i=0u;i<run.range.y;i++){
  let point=points[run.range.x+i];if(!pointVisible(point)){continue;}
  low=min(low,point.position.xy);high=max(high,point.position.xy);seen=true;
  if(i+1u<run.range.y){let next=points[run.range.x+i+1u].position.xy;low=min(low,next);high=max(high,next);}
 }
 if(!seen){return 1.0e9;}
 var own=max(high.x-low.x,high.y-low.y)/17.0;let face=faceRows(run);
 // A FORM THAT IS NOT TERRAIN is as near as its own rows on the page say,
 // whichever of its lines is asked: a rock in front of the viewer whose line
 // shows only a few pixels was classed far and drawn as a strip profile
 // (crop-review-20261110 crops 1 F, 3 B, 5 E). Its rows are the form table's
 // visible extent. A height surface has no such size: its lines keep theirs.
 if(!terrainRun(run)){let extent=formExtent(run.identity.yz);if(extent.found){own=max(own,(extent.bottom-extent.top)/p.field.y*p.rect.w/17.0);}}

 return max(own,face);
}
// ONE TIER PER FORM, HELD (line repair step 2, 2026-11-12). A form's tier
// changes only when its rows pass the threshold by a margin: a near form stays
// near until it falls under NEAR_FROM_ROWS * TIER_HOLD rows, a middle one
// turns near only from NEAR_FROM_ROWS / TIER_HOLD, and the same about
// FAR_STRIP_BELOW_ROWS (the LOD hysteresis band of game engines; skill
// temporal reference: "no threshold chatter"). Before, the hard 3- and 7-row
// thresholds switched a form's construction whenever its rows wobbled across
// them by a pixel. The previous tier is the run's own correspondence record
// (memo payload word 5, keyed by the run's identity); a run without one (an
// anchor, a whole box or round, which read their owner's rows) keeps the
// plain tier. The tone producer reads the plain tier (tone-field.ts); the two
// can differ only inside the band.
const TIER_HOLD=${LOD_TIER_HOLD};
fn tierWord(runID:u32,run:Run)->u32{
 if((run.range.z&0x10000u)!=0u||wholeBoundaryFlags(run)!=0u){return NONE;}
 let slot=atomicLoad(&counts[memoMapBase()+runID]);if(slot==NONE){return NONE;}
 return u32(p.field.z)+slot*${MEMO_WORDS}u+${MEMO_HEADER_WORDS}u+5u;
}
fn tierHold(rows:f32,held:u32,plain:u32)->u32{
 if(held==0u&&plain==1u&&rows>=${NEAR_FROM_ROWS}.0*TIER_HOLD){return 0u;}
 if(held==1u&&plain==0u&&rows<${NEAR_FROM_ROWS}.0/TIER_HOLD){return 1u;}
 if(held==1u&&plain==2u&&rows>=${FAR_STRIP_BELOW_ROWS}.0*TIER_HOLD){return 1u;}
 if(held==2u&&plain==1u&&rows<${FAR_STRIP_BELOW_ROWS}.0/TIER_HOLD){return 2u;}
 return plain;
}

// spanning, cut or unslotted owner) takes the form table's ONE held level,
// the same level its relations, contacts and outline were produced at
// (surface-relations formLevel). Spanning-surface lines keep the run hold.
fn compactHeldLevel(run:Run)->u32{
 if(!formLodActive()||(run.range.z&0x40000000u)!=0u||terrainRun(run)||!viewRecedes(p.worldFromClip)){return NONE;}
 let owner=run.identity.yz;let form=formExtent(owner);
 if(!form.found||formCut(form.box,p.field.xy)||formSpansDepth(form.near,form.far,p.field.xy,p.worldFromClip)){return NONE;}
 return formHeldLod(owner,true);
}
fn heldTier(runID:u32,run:Run)->u32{
 let rows=formRows(run);let plain=tierOfRows(rows);var tier=plain;
 let at=tierWord(runID,run);
 let compact=compactHeldLevel(run);
 if(compact!=NONE){return compact;}
 if(at!=NONE){let stored=atomicLoad(&counts[at]);if((stored&3u)!=0u&&(stored&~3u)==(run.identity.w&~3u)){tier=tierHold(rows,(stored&3u)-1u,plain);}}

 return tier;
}
// featurecurves records the tier it drew with; skylineReduce, earlier in the
// same frame, read the same previous record, so both agree.
fn keepTier(runID:u32,run:Run,tier:u32){let at=tierWord(runID,run);if(at!=NONE){atomicStore(&counts[at],(run.identity.w&~3u)|(min(tier,2u)+1u));}}
// THE ROWS OF THE SURFACE AT THE LINE'S DISTANCE. A line of a spanning
// surface is short on the page wherever the surface is folded finely, also on
// a cliff right in front of the viewer, so its own length is not its size: a
// near cliff's folds read as far and the far tier dropped them all (probe

// 2026-11-10 "the walls of cliffs missing edges"). The surface's size at the
// line is the page extent, up and down from the line, over which the same
// surface stays at the line's own distance within half again (the scale at
// which a form of that surface is drawn, as the form table gives a compact
// form). Under the horizon the distance falls away at once, so the ground
// line stays far; a cliff face holds its distance and is many rows.
fn eyeWorld()->vec3f{let e=p.worldFromClip*vec4f(0.0,0.0,1.0,0.0);if(abs(e.w)<1e-9){return vec3f(0.0);}return e.xyz/e.w;}
fn faceRows(run:Run)->f32{
 var at=NONE;var best=-1.0;
 for(var i=0u;i+1u<run.range.y;i++){
  let a=points[run.range.x+i];if(!pointVisible(a)){continue;}
  let l=length(points[run.range.x+i+1u].position.xy-a.position.xy);if(l>best){best=l;at=i;}
 }
 if(at==NONE){return 0.0;}
 let a=points[run.range.x+at].position.xy;let b=points[run.range.x+at+1u].position.xy;let m=(a+b)*.5;
 let side=normalize(vec2f(b.y-a.y,a.x-b.x)+vec2f(1e-6,0.0));
 // The surface continues while each 4 px step stays on surface and moves less
 // than a quarter of its own distance (no depth step). Two corrections

 // camera" drew almost nothing: 176 of its 280 terrain lines read far with
 // face rows 0-1): a terrain line's surface is every terrain chunk, not the
 // one chunk owner its line lies on (the rock spans chunks 396/397, and the
 // walk stopped at the chunk seam); and the walk compares each step with the
 // previous one, not with the start, because a near face's distance changes
 // by half again within a row or two although it is one continuous face.
 // Ground near the horizon still stops at once (its distance jumps a large
 // fraction per pixel), so far ground lines stay far.
 let terrain=terrainRun(run);
 let eye=eyeWorld();var rows=0.0;
 for(var k=0;k<2;k++){
  let start=m+side*select(-3.0,3.0,k==0);let s0=faceSample(run,fieldPoint(start),terrain);if(!s0.on){continue;}
  let r0=length(tierWorld(fieldPoint(start),s0.d)-eye);if(r0<=1e-6){continue;}
  var reach=0.0;
  for(var dir=-1.0;dir<=1.0;dir+=2.0){
   var previous=r0;
   for(var j=1;j<=48;j++){
    let q=start+vec2f(0.0,dir*4.0*f32(j));let s=faceSample(run,fieldPoint(q),terrain);if(!s.on){break;}
    let r=length(tierWorld(fieldPoint(q),s.d)-eye);if(abs(r-previous)>previous*.25){break;}
    previous=r;reach+=4.0;
   }
  }
  rows=max(rows,reach/17.0);
 }
 return rows;
}
// A terrain line's surface is any surface sample (all terrain chunks are one
// height surface; a depth step ends it in faceRows); other lines keep their owner.
fn faceSample(run:Run,f:vec2f,terrain:bool)->TierSample{
 if(!terrain){return tierSample(run,f);}
 let q=vec2i(floor(f));
 if(any(q<vec2i(0))||any(q>=vec2i(p.field.xy))){return TierSample(false,0.0,vec3f(0),0.0);}
 let d=textureLoad(depthField,q,0).x;
 return TierSample(d>0.0,d,vec3f(0),0.0);
}
var<private> lineTier:u32;
// The compiled size band a tier may use at most: far 0, middle 1, near 2.
fn tierBand(tier:u32)->u32{return 2u-min(tier,2u);}
// x: stations read on the run's own surface. y: stations where the two
// faces fall in different tone bands of the scene light. z: stations with a
// second crease inside one glyph width on the canonical side (down the page
// for a flat run, right for a steep one).
fn creaseRead(run:Run)->vec3u{
 let scale=p.field.xy/p.rect.zw;let stride=max(1u,run.range.y/8u)*2u;var read=vec3u(0u);
 for(var k=0u;k<4u;k++){
  let i=k*stride;if(i+1u>=run.range.y){break;}
  let a=points[run.range.x+i];let b=points[run.range.x+i+1u];if(!pointVisible(a)){continue;}
  let source=spanSource(run,a.visible.z);
  let along=b.position.xy-a.position.xy;if(dot(along,along)<16.0){continue;}
  var side=normalize(vec2f(-along.y,along.x));
  let flat=abs(side.y)>=abs(side.x);
  if((flat&&(side.y<0.0))||(!flat&&(side.x<0.0))){side=-side;}
  let lean=side*scale;let px=length(lean);let unit=lean/px;let inner=max(2.0*px,1.5);let pace=max(2.0*px,1.0);
  for(var s=1u;s<4u;s++){
   let f=fieldPoint(mix(a.position.xy,b.position.xy,f32(s)*.25));
   let plus=tierSample(source,f+unit*inner);let minus=tierSample(source,f-unit*inner);
   if(!plus.on||!minus.on){continue;}
   read.x++;
   let bright=max(plus.luma,minus.luma);
   if(bright>.0001&&min(plus.luma,minus.luma)<TONE_BAND_STEP*bright){read.y++;}
   var prior=plus;var reach=inner;
   for(var w=0u;w<8u;w++){
    reach+=pace;if(reach>FACET_COLLAPSE*px){break;}
    let next=tierSample(source,f+unit*reach);if(!next.on){break;}
    if(dot(next.n,prior.n)<.766){read.z++;break;}
    prior=next;
   }
  }
 }
 return read;
}
// The two silhouettes of one thin form each decide alone which of them is
// drawn. The canonical side flips at one run direction, so the side rule has
// a dead band about it where both are kept: .368 keeps silhouettes up to 29
// degrees from parallel from ever both yielding.
const LIMB_SIDE_MARGIN=.368;
// Reads a silhouette or occluding run whose own form lies opposite the
// canonical side (canonical: below a flat run, right of a steep one) at
// every station it can read. x: stations read. y: stations where the form
// ends or drops behind within one glyph width while it runs on at least one
// glyph width along the line, which is a stroke and not a small closed form.
// Zero when any station finds the form on the canonical side or inside the
// dead band: that run is the one that stays.
fn limbRead(run:Run)->vec2u{
 let scale=p.field.xy/p.rect.zw;let stride=max(1u,run.range.y/8u)*2u;var read=vec2u(0u);
 for(var k=0u;k<4u;k++){
  let i=k*stride;if(i+1u>=run.range.y){break;}
  let a=points[run.range.x+i];let b=points[run.range.x+i+1u];if(!pointVisible(a)){continue;}
  let source=spanSource(run,a.visible.z);
  let along=b.position.xy-a.position.xy;if(dot(along,along)<16.0){continue;}
  var side=normalize(vec2f(-along.y,along.x));
  let flat=abs(side.y)>=abs(side.x);
  if((flat&&(side.y<0.0))||(!flat&&(side.x<0.0))){side=-side;}
  let decisive=side.x+side.y>LIMB_SIDE_MARGIN;
  let lean=side*scale;let px=length(lean);let unit=lean/px;let inner=max(2.0*px,1.5);let pace=max(2.0*px,1.0);
  let beside=normalize(along)*scale*FACET_COLLAPSE;
  for(var s=1u;s<4u;s++){
   let f=fieldPoint(mix(a.position.xy,b.position.xy,f32(s)*.25));
   // The form is the only side that reads the owner, or the nearer side
   // where the owner occludes itself (same 3% depth relation as the D10
   // classifier). Half the offset first, to reach forms thinner than it.
   var own=0.0;var start=inner;var prior=TierSample(false,0.0,vec3f(0),0.0);
   for(var h=0u;h<2u&&own==0.0;h++){
    start=inner*select(1.0,.5,h==0u);
    let plus=tierSample(source,f+unit*start);let minus=tierSample(source,f-unit*start);
    if(plus.on&&(!minus.on||plus.d>minus.d*1.03)){own=1.0;prior=plus;}
    else if(minus.on&&(!plus.on||minus.d>plus.d*1.03)){own=-1.0;prior=minus;}
   }
   if(own==0.0){continue;}
   if(own>0.0||!decisive){return vec2u(0u);}
   read.x++;var reach=start;var thin=false;
   for(var w=0u;w<8u;w++){
    reach+=pace;if(reach>FACET_COLLAPSE*px){break;}
    let next=tierSample(source,f-unit*reach);
    if(!next.on||next.d<prior.d*.97){thin=true;break;}
    prior=next;
   }
   if(thin&&(tierSample(source,f-unit*start+beside).on||tierSample(source,f-unit*start-beside).on)){read.y++;}
  }
 }
 return read;
}
// PATH-LED OR REGION-LED (compiled-tone-passages.ts TEXTURE_EDGE_MEASURE, the
// same function the tone producer reads): x stations read, y stations where
// the run's own form has a boundary finer than a glyph and hangs over open
// space. The tone producer spells that envelope as texture-edge words.
${TEXTURE_EDGE_MEASURE}
fn leafRead(run:Run)->vec2u{
 let scale=p.field.xy/p.rect.zw;let stride=max(1u,run.range.y/8u)*2u;var read=vec2u(0u);
 for(var k=0u;k<4u;k++){
  let i=k*stride;if(i+1u>=run.range.y){break;}
  let a=points[run.range.x+i];let b=points[run.range.x+i+1u];if(!pointVisible(a)){continue;}
  let along=b.position.xy-a.position.xy;if(dot(along,along)<16.0){continue;}
  let lean=normalize(vec2f(-along.y,along.x))*scale;let px=length(lean);let unit=lean/px;let inner=max(2.0*px,1.5);
  let f=fieldPoint(mix(a.position.xy,b.position.xy,.5));
  // The form is the side that reads the owner, or the nearer side where the
  // owner occludes itself (limbRead's relation).
  let plus=tierSample(run,f+unit*inner);let minus=tierSample(run,f-unit*inner);
  var own=f;var depth=0.0;
  if(plus.on&&(!minus.on||plus.d>minus.d*1.03)){own=f+unit*inner;depth=plus.d;}
  else if(minus.on&&(!plus.on||minus.d>plus.d*1.03)){own=f-unit*inner;depth=minus.d;}
  else{continue;}
  read.x++;
  if(textureEdge(own,vec2u(run.identity.y,run.identity.z),depth)){read.y++;}
 }
 return read;
}
// Selection table [tier][column] (TIER_SELECTION): least visible projected
// length in text rows, TIER_NEVER = not drawn at that tier.
// The slope turn across a terrain crease, read on the frame's own normals at
// the stations creaseRead uses. x: stations where both faces read the run's
// surface. y: stations whose faces turn by 45 degrees or more
// (TERRAIN_TURN_COS). z: stations that are a shelf edge, one face near level
// and the other a drop (TERRAIN_SHELF_TOP, TERRAIN_SHELF_FACE; world up is +y,
// as surface-relations.ts reads it). Fixed thresholds, nothing is scored.
fn terrainCreaseRead(run:Run)->vec3u{
 let scale=p.field.xy/p.rect.zw;let stride=max(1u,run.range.y/8u)*2u;var read=vec3u(0u);
 for(var k=0u;k<4u;k++){
  let i=k*stride;if(i+1u>=run.range.y){break;}
  let j=min(i+max(stride/2u,1u),run.range.y-1u);
  let a=points[run.range.x+i];let b=points[run.range.x+j];if(!pointVisible(a)){continue;}
  let along=b.position.xy-a.position.xy;if(dot(along,along)<16.0){continue;}
  let lean=normalize(vec2f(-along.y,along.x))*scale;let px=length(lean);let unit=lean/px;let inner=max(2.0*px,1.5);
  for(var s=1u;s<4u;s++){
   let f=fieldPoint(mix(a.position.xy,b.position.xy,f32(s)*.25));
   let plus=tierSample(run,f+unit*inner);let minus=tierSample(run,f-unit*inner);
   if(!plus.on||!minus.on){continue;}
   read.x++;
   if(dot(plus.n,minus.n)<=${TERRAIN_TURN_COS}){read.y++;}
   if(max(plus.n.y,minus.n.y)>=${TERRAIN_SHELF_TOP}&&min(plus.n.y,minus.n.y)<=${TERRAIN_SHELF_FACE}){read.z++;}
  }
 }
 return read;
}
// TERRAIN OUTLINE OR INTERIOR LINE. A silhouette or occluding contour of a
// height surface separates the near side (larger field depth) from what lies
// beyond it. Per station, two page px to each side of the line, and a third
// sample two page px further on the near side: the near surface's depth is
// extrapolated across the line (post-projective depth is affine in the page
// over a plane, so a plane continues exactly) and compared, in world units, to
// the surface actually beyond. The station is an OUTLINE when the beyond side
// is paper (sky) or lies farther behind the continued near surface than one
// text row spans at the near point (tierRowUnits: the page's own scale); it is
// INTERIOR when the beyond surface continues the near one within that span
// (any owner: a road or field mesh lying on the ground is the ground). x:
// stations read, y: outline stations by that depth rule, z: outline stations
// by the depth rule OR a turn of the two sides' normals by TERRAIN_TURN_COS or
// more (a crease is continuous in depth and still a line: only a coplanar
// seam is not). Fixed rule, nothing is scored.
fn fieldSurface(f:vec2f)->f32{
 let q=vec2i(floor(f));
 if(any(q<vec2i(0))||any(q>=vec2i(p.field.xy))){return -1.0;}
 let owner=textureLoad(ownerField,q,0).xy;if((owner.x|owner.y)==0u){return 0.0;}
 return max(textureLoad(depthField,q,0).x,0.0);
}
fn fieldNormal(f:vec2f)->vec3f{
 let e=textureLoad(normalField,vec2i(floor(f)),0).xy;var n=vec3f(e,1.0-abs(e.x)-abs(e.y));
 if(n.z<0.0){n=vec3f((vec2f(1.0)-abs(n.yx))*select(vec2f(-1.0),vec2f(1.0),n.xy>=vec2f(0)),n.z);}
 return normalize(n);
}
// Three stations on one page segment a..b (page px).
fn surfaceOutlineRead(a:vec2f,b:vec2f)->vec3u{
 var read=vec3u(0u);
 let along=b-a;if(dot(along,along)<16.0){if(probeOutline){probeAdd(86u);}return read;}
 let scale=p.field.xy/p.rect.zw;let row=17.0*p.field.y/p.rect.w;
 let lean=normalize(vec2f(-along.y,along.x))*scale;let px=length(lean);let unit=lean/px;let inner=max(2.0*px,1.5);
 for(var s=1u;s<4u;s++){
  let f=fieldPoint(mix(a,b,f32(s)*.25));
  let plus=fieldSurface(f+unit*inner);let minus=fieldSurface(f-unit*inner);
  if(plus<0.0||minus<0.0||(plus==0.0&&minus==0.0)){if(probeOutline){probeAdd(87u);}continue;}
  if(plus==0.0||minus==0.0){read+=vec3u(1u);continue;}
  let side=select(-1.0,1.0,plus>=minus);
  let nearF=f+unit*inner*side;let nearD=max(plus,minus);let farF=f-unit*inner*side;let farD=min(plus,minus);
  let next=fieldSurface(f+unit*inner*3.0*side);if(next<=0.0){if(probeOutline){probeAdd(88u);}continue;}
  let continued=2.0*nearD-next;if(continued<=0.0){if(probeOutline){probeAdd(89u);}continue;}
  read.x++;
  // The step is measured across the beyond surface (along its normal): at a
  // grazing view a low kerb on flat ground opens a long gap along the ray
  // but stands only its own height off the ground.
  let farN=fieldNormal(farF);
  let gap=abs(dot(tierWorld(farF,farD)-tierWorld(farF,continued),farN));
  let stepped=gap>tierRowUnits(nearF,nearD,p.field.xy,row,p.worldFromClip);
  if(stepped){read.y++;}
  if(stepped||dot(fieldNormal(nearF),farN)<=${TERRAIN_TURN_COS}){read.z++;}
 }
 return read;
}
fn terrainOutlineRead(run:Run)->vec3u{
 let stride=max(1u,run.range.y/8u)*2u;var read=vec3u(0u);
 for(var k=0u;k<4u;k++){
  let i=k*stride;if(i+1u>=run.range.y){break;}
  // The chord to the next station: neighbouring run points lie closer than
  // the four page px a station needs (a run that read no station kept its
  // selection, which let every seam lying on open ground through).
  let j=min(i+max(stride/2u,1u),run.range.y-1u);
  let a=points[run.range.x+i];let b=points[run.range.x+j];if(!pointVisible(a)){if(probeOutline){probeAdd(90u);}continue;}
  read+=surfaceOutlineRead(a.position.xy,b.position.xy);
 }
 return read;
}
var<private> probeOutline:bool;// PROBE
// tier: the form's one tier (distanceTier), the row of TIER_SELECTION.
fn tierSelects(run:Run,tier:u32)->bool{
 var table=array<u32,36>(${[...TIER_SELECTION,...TERRAIN_SELECTION].flat().map(rows=>rows+'u').join(',')});
 let line=min((run.range.z>>8u)&255u,4u);
 let terrain=terrainRun(run);
 // A terrain crease is a candidate only as a principal slope turn (column 5).
 let principal=line==3u&&(terrain||(${PRINCIPAL_CREASE_FROM_HARD_BIT}&&(run.range.z&0x40000u)!=0u));
 let least=table[(select(0u,3u,terrain)+min(tier,2u))*6u+select(line,5u,principal)];
 if(least==TIER_NEVER){return false;}
 if(terrain&&line==3u){
  var projected=0.0;
  for(var i=0u;i+1u<run.range.y;i++){
   if(pointVisible(points[run.range.x+i])){projected+=length(points[run.range.x+i+1u].position.xy-points[run.range.x+i].position.xy);}
  }
  if(projected<f32(least)*17.0){return false;}
  // Parallel slope turns inside one glyph width are one line (creaseRead z).
  let facets=creaseRead(run);if(facets.x>0u&&facets.z==facets.x){return false;}
  let turn=terrainCreaseRead(run);
  if(turn.x==0u){return false;}
  return select(turn.z,turn.y,tier==0u)*2u>=turn.x;
 }
 // What a silhouette or occluding contour of a SURFACE contributes (a height
 // surface, or a hard edge that is not a whole box or open-path construction;
 // terrainOutlineRead): an outline against sky or a farther surface carries
 // the read (the horizon, a ridge before a farther slope, a wall against the
 // ground behind it); a line whose two sides are one continuous surface (a
 // seam or mesh border lying on flat ground, a fold inside one cliff face, at
 // the page's own scale) is no outline, and the surface's tone carries it
 // (skill 11 falsifier "all detected edges were spelled"). Every tier: what
 // a row cannot show is decided by the row's world span. Majority of the
 // stations read; a run that reads none keeps its selection.
 // A hard edge also keeps a crease turn (z); a height surface's folds go by
 // depth alone (y), so a picket of facet contours down one cliff face yields.
 if((terrain||((run.range.z&0x40000u)!=0u&&wholeBoundaryFlags(run)==0u))&&line<=1u){
  probeOutline=!terrain;let outline=terrainOutlineRead(run);probeOutline=false;let kept=select(outline.z,outline.y,terrain);
  if(!terrain){if(outline.x==0u){probeAdd(67u);}else if(kept*2u>outline.x){probeAdd(84u);if(outline.y*2u>outline.x){probeAdd(85u);}}}// PROBE
  if(outline.x>0u&&kept*2u<=outline.x){
   // A FOLD OF A STEEP FACE (a cliff's facets turning 45 degrees or more
   // against each other) is the face's construction, not a seam: it stays
   // when it runs three rows or more and no second fold lies inside one
   // glyph width of it (creaseRead z), so a face is a few fold strokes with
   // its tone between them and never a picket.
   // A fold whose two faces fall in different tone bands of the scene light

   // call-out 2026-11-10, view 06 cliff); a gentle fold inside one band is
   // carried by the face's tone.
   let band=creaseRead(run);
   if(!terrain||(outline.z*2u<=outline.x&&!(band.x>0u&&band.y*2u>=band.x))){return false;}
   var folded=0.0;
   for(var i=0u;i+1u<run.range.y;i++){
    if(pointVisible(points[run.range.x+i])){folded+=length(points[run.range.x+i+1u].position.xy-points[run.range.x+i].position.xy);}
   }
   if(folded<51.0){return false;}
   let near=creaseRead(run);if(near.x>0u&&near.z==near.x){return false;}
  }
 }
 // A CONTACT is a junction of two surfaces that turn against each other (a
 // wall on the ground, a trunk on a slope). Two surfaces that cross nearly
 // coplanar (two ground layers through each other, a slab flush with the
 // terrain) meet in a seam lying on the ground: no turn, no depth step, no line.
 if(line==2u&&wholeBoundaryFlags(run)==0u){let outline=terrainOutlineRead(run);if(outline.x>0u&&outline.z*2u<=outline.x){return false;}}
 // A form thinner than one glyph is one stroke, not two outlines. Of its two
 // silhouettes the one whose form lies opposite the canonical side yields at
 // every tier, so the other always draws. Forms one glyph wide or more, and
 // runs that cannot read their own form, keep both sides.
 // A whole round or box (0x120000) is ONE closed run: no twin outline to
 // yield to; a thin visible crescent keeps its arc (lines4, D1).
 if(line<=1u&&(run.range.z&0x120000u)==0u){let limb=limbRead(run);if(limb.x>0u&&limb.y*2u>limb.x){return false;}}
 // A boundary finer than a glyph that hangs over open space is region-led:
 // in the near and middle views the tone producer spells it as texture-edge
 // words (tone-field.ts textureEdges), so its outline is not also spelled.
 // The far view keeps the connected outline (D9). Only a field-class source
 // (0x8000000, the class the tone producer's texture edge acts on) qualifies;
 // hard edges and whole round or box constructions are never texture.
 if(line<=1u&&tier<2u&&(run.range.z&0x8060000u)==0x8000000u){let leaf=leafRead(run);if(leaf.x>0u&&leaf.y*2u>leaf.x){return false;}}
 // At the outline tier an occluding contour INSIDE a soft form (one tier of
 // a conifer's crown over the next, a lobe over a lobe) is drawn only when it
 // runs FAR_CONTOUR_LEAST_ROWS rows: a run is one unit and its shift per row,
 // so a shorter contour is one stray glyph inside the outline, not a line.
 // The silhouette (line 0) is drawn at any length.
 if(line==1u&&tier==2u&&(run.range.z&0x8060000u)==0x8000000u){
  var length1=0.0;
  for(var i=0u;i+1u<run.range.y;i++){
   if(pointVisible(points[run.range.x+i])){length1+=length(points[run.range.x+i+1u].position.xy-points[run.range.x+i].position.xy);}
  }
  if(length1<f32(${FAR_CONTOUR_LEAST_ROWS})*17.0){return false;}
 }
 // A terrain interior turn (column 4) yields to a parallel turn inside one
 // glyph width (creaseRead z), as a crease does, before its length is read.
 if(terrain&&line==4u){
  let near=creaseRead(run);if(near.x>0u&&near.z==near.x){return false;}
  // No light or turn test: a gentle fold inside one tone band is a fold of

  // runs 2895, 2563, 2896, all near, were dropped here by a "light separates
  // the faces or 45 degrees" rule that no reference supports; withdrawn).
 }
 if(line!=3u&&least==0u){return true;}
 var projected=0.0;
 for(var i=0u;i+1u<run.range.y;i++){
  if(pointVisible(points[run.range.x+i])){projected+=length(points[run.range.x+i+1u].position.xy-points[run.range.x+i].position.xy);}
 }
 if(projected<f32(least)*17.0){return false;}
 if(line!=3u){return true;}
 // Interior creases are not all one weight. Parallel creases inside one
 // glyph width collapse into the last one on the canonical side. A facet
 // seam (both faces in one tone band) is not a line; a principal crease is
 // always a line, at every tier (TIER_SELECTION). No crease angle and no
 // count decides this.
 let read=creaseRead(run);
 if(read.x>0u&&read.z==read.x){return false;}
 if(principal){return true;}
 if(read.x>0u&&read.y*2u>=read.x){return true;}
 // A PLANE BOUNDARY is a line whether or not the light separates its two
 // faces: a faceted rock's top against its side, a roof plane against a wall.
 // A triangle soup carries no authored hard bit, so the turn is read on the
 // frame's own normals (terrainCreaseRead y: 45 degrees or more).
 let turn=terrainCreaseRead(run);
 return turn.x>0u&&turn.y*2u>=turn.x;
}
fn rank(run:Run,a:vec2f,b:vec2f)->u32{
 // Fixed contribution-to-read order. This is NOT a glyph-fit score. The line
 // CLASS decides first (bits 28-30: outline above contact above crease above
 // interior, AGENTS finding 4 and the skill's "decide importance first, then
 // draw; cut from the bottom"), and length x contrast orders lines only
 // within one class. As a product, a long interior crease outranked a short
 // silhouette piece and took its cells: the nearest rock's right silhouette
 // and the bridge edges at spot-frames (loss split 2026-11-12: 433 of 1389
 // losses were a line blocked by another line of the same form).
 let weights=array<f32,5>(8.0,7.0,5.0,4.0,3.0);let cls=min((run.range.z>>8u)&255u,4u);let weight=weights[cls];
 // A compiled whole-part program already selected its source feature and its
 // stable order. Its priority must not be re-arbitrated by this frame's
 // projected length, contrast or visible-fragment count. Preserve the same
 // line-class precedence, then use the source feature ordinal as a deterministic
 // tie-break (lower source ordinal wins). Legacy runs keep the incumbent field
 // contribution calculation below.
 let programmed=runProgramFeature(run);
 if(programmed!=0u){return ((4u-cls)<<28u)|max(1u,0x07ffffffu-min(programmed,0x07fffffeu));}
 let size=vec2i(p.field.xy);var contrast=.125;var projected=0.0;
 for(var i=0u;i<run.range.y;i++){
  let point=points[run.range.x+i];if(!pointVisible(point)){continue;}
  if(i+1u<run.range.y){projected+=length(points[run.range.x+i+1u].position.xy-point.position.xy);}
  let center=fieldPixel(point.position.xy);
  if(any(center<=vec2i(0))||any(center>=size-vec2i(1))){continue;}
  let owner=textureLoad(ownerField,center,0).xy;
  if((owner.x&0xdfffffffu)!=run.identity.y||owner.y!=run.identity.z){continue;}
  let d=textureLoad(depthField,center,0).x;
  for(var axis=0u;axis<2u;axis++){
   let delta=select(vec2i(1,0),vec2i(0,1),axis==1u);
   let left=textureLoad(depthField,center-delta,0).x;let right=textureLoad(depthField,center+delta,0).x;
   let na=textureLoad(normalField,center-delta,0).xy;let nb=textureLoad(normalField,center+delta,0).xy;
   contrast=max(contrast,max(abs(left-right)/max(d,.000001),length(na-nb)));
  }
 }
 // NO PROMOTION (line repair step 3). The run's class was raised to the
 // highest D10 class found under any of its points (cls=min(cls,k)): an
 // interior crease crossing a silhouette pixel ranked as a silhouette and the
 // class-first order of the comment above was undone. The class is the run's
 // own line class; the field supplies contrast only.
 let within=u32(clamp(weight*max(projected,length(b-a))*contrast*256.0/16.0,1.0,268435455.0));
 return ((4u-cls)<<28u)|within;
}
// THE SELECTION CELL GRID (line repair step 3, 2026-11-12). Every selected
// line run that featurecurves draws (not an anchor, not a whole box or round,
// not a run the far strip spells) writes its key into each text cell its
// visible path crosses: key = rank (bits 11-30 of the run's fixed rank, class
// first) << 12 | owner tag << 4 | undirected direction (16 bins of 180
// degrees). Two grids, the second shifted by half a cell, so two lines closer
// than 8 px always share a cell of one of them. A cell's atomic maximum is the
// highest-ranked line through it. A word of a lower-ranked run whose cell is
// held by a higher-ranked PARALLEL line (two bins, 22 degrees) of the SAME
// owner (terrain: the one height surface) is that line, one glyph width away:

// 55), instead of being found as a copy among the committed words
// (terrainCopy, duplicateWord, which stay until this draws as well).
// Fixed rule, decided once per frame from selection, no pens tried.
// PER-FEATURE PLACEMENT RECORD (line repair step 5): two words per run after
// the drawn table, cleared every frame: [0] units of the run that found no
// placement (losses), [1] units merged as copies of a committed word
// (terrainCopy, duplicateWord, levelTwin). A copy is recorded as a copy of
// its run, never silently as a placed unit.
fn featureRecord(runID:u32)->u32{return drawnBase()+p.playback.z+min(runID,p.extent.z-1u)*2u;}
fn gridColumns()->u32{return (p.extent.x+15u)/16u+1u;}
fn gridBase()->u32{return drawnBase()+p.playback.z+p.extent.z*3u;}
fn gridCell(grid:u32,x:f32,row:i32)->u32{
 if(row<0||row>=i32(rowCount())){return NONE;}
 let c=i32(floor((x+f32(grid)*8.0)/16.0));if(c<0||c>=i32(gridColumns())){return NONE;}
 return gridBase()+(grid*rowCount()+u32(row))*gridColumns()+u32(c);
}
fn gridDir(d:vec2f)->u32{return u32(round(fract(atan2(d.y,d.x)/3.14159265359+1.0)*16.0))&15u;}
var<private> gridKeyBase:u32;
var<private> collapseDir:u32;
fn gridWrite(run:Run,rank:u32){
 let terrain=terrainRun(run);
 let tag=select((((run.identity.y*0x9e3779b9u)^(run.identity.z*0x85ebca6bu))>>24u)&0xfeu,0xffu,terrain);
 gridKeyBase=((rank>>11u)<<12u)|(tag<<4u);if((rank>>11u)==0u){gridKeyBase=0u;return;}
 var samples=0u;
 for(var k=0u;k+1u<run.range.y&&samples<512u;k+=2u){
  let a=points[run.range.x+k];if(!pointVisible(a)){continue;}let b=points[run.range.x+k+1u];
  let d=b.position.xy-a.position.xy;let l=length(d);if(l<.01){continue;}
  let key=gridKeyBase|gridDir(d);let n=u32(ceil(l/8.0))+1u;
  for(var i=0u;i<n&&samples<512u;i++){
   let q=mix(a.position.xy,b.position.xy,f32(i)/f32(max(n-1u,1u)));let row=i32(floor(q.y/17.0));samples++;
   for(var g=0u;g<2u;g++){let at=gridCell(g,q.x,row);if(at!=NONE){atomicMax(&counts[at],key);}}
  }
 }
}
// RIPPLE THINNING (lines4, 2026-11-13): see l4ripple. Interior curves of a
// surface source that is not terrain, field-class, a round or an anchor lie
// in families of parallel ripples closer than a glyph; one is kept every two
// to three glyphs. skylineReduce writes each selected one into two
// half-offset grids of 32 px x 2-row cells (after the line grids) with a key
// ordered by a hash of its SOURCE address (root face part.x, owner, domain):
// the kept set belongs to the surface, not to this frame's lengths or atomic
// order. A run yields once, whole, when at most of its stations a parallel
// same-owner ripple of higher key holds the cell.
fn rippleRun(run:Run)->bool{return ((run.range.z>>8u)&255u)==4u&&(run.range.z&0x18030000u)==0u;}
fn rippleRows()->u32{return (rowCount()+1u)/2u+2u;}
fn rippleColumns()->u32{return (p.extent.x+31u)/32u+2u;}
fn rippleCell(g:u32,q:vec2f)->u32{
 let c=i32(floor((q.x+f32(g)*16.0)/32.0));let r=i32(floor((q.y+f32(g)*17.0)/34.0));
 if(c<0||r<0||c>=i32(rippleColumns())||r>=i32(rippleRows())){return NONE;}
 return gridBase()+2u*rowCount()*gridColumns()+(g*rippleRows()+u32(r))*rippleColumns()+u32(c);
}
fn rippleKey(run:Run)->u32{
 var h=(run.part.x*0x9e3779b9u)^(run.identity.z*0x85ebca6bu)^(run.identity.y*0xc2b2ae35u);
 h=(h^(h>>16u))*0x7feb352du;h=h^(h>>15u);
 let tag=(((run.identity.y*0x9e3779b9u)^(run.identity.z*0x85ebca6bu))>>24u)&0xffu;
 return (((h>>12u)|1u)<<12u)|(tag<<4u);
}
fn rippleWrite(run:Run){
 let key=rippleKey(run);var samples=0u;
 for(var k=0u;k+1u<run.range.y&&samples<512u;k+=2u){
  let a=points[run.range.x+k];if(!pointVisible(a)){continue;}let b=points[run.range.x+k+1u];
  let d=b.position.xy-a.position.xy;let l=length(d);if(l<.01){continue;}
  let n=u32(ceil(l/8.0))+1u;let dir=gridDir(d);
  for(var i=0u;i<n&&samples<512u;i++){
   let q=mix(a.position.xy,b.position.xy,f32(i)/f32(max(n-1u,1u)));samples++;
   for(var g=0u;g<2u;g++){let at=rippleCell(g,q);if(at!=NONE){atomicMax(&counts[at],key|dir);}}
  }
 }
}
fn rippleYields(run:Run)->bool{
 let own=rippleKey(run);var read=0u;var held=0u;
 let stride=max(1u,run.range.y/16u)*2u;
 for(var k=0u;k+1u<run.range.y;k+=stride){
  let a=points[run.range.x+k];if(!pointVisible(a)){continue;}let b=points[run.range.x+k+1u];
  let d=b.position.xy-a.position.xy;if(dot(d,d)<.0001){continue;}
  let dir=gridDir(d);let q=(a.position.xy+b.position.xy)*.5;read++;
  for(var g=0u;g<2u;g++){
   let at=rippleCell(g,q);if(at==NONE){continue;}
   let cell=atomicLoad(&counts[at]);let turn=((cell&15u)+16u-dir)&15u;
   if((cell>>12u)>(own>>12u)&&((cell>>4u)&255u)==((own>>4u)&255u)&&min(turn,16u-turn)<=2u){held++;break;}
  }
 }
 return read>0u&&held*2u>read;
}
fn gridCollapsed(own:u32,x:i32,row:i32,advance:u32)->bool{
 if(own==0u){return false;}
 let centre=f32(x)+f32(advance)*.5;
 for(var g=0u;g<2u;g++){
  let at=gridCell(g,centre,row);if(at==NONE){continue;}
  let held=atomicLoad(&counts[at]);let turn=((held&15u)+16u-(own&15u))&15u;

  // owner. Numabe draws a road, its curb and its walk as separate placements;

  // Numabe frame), and two parallel lines closer than one glyph cannot both be
  // spelled (FACET_COLLAPSE). The owner tag still orders equal ranks.
  if((held>>12u)>(own>>12u)&&min(turn,16u-turn)<=2u){return true;}
 }
 return false;
}
fn visibleWord(run:Run,x:i32,row:i32,word:u32,count:u32,part:u32,depth:f32,sourceScalar:u32)->bool{
 var total=0u;var covered=0u;var pen=x;
 // A foot anchor (gpu-construction-runs.ts contactAnchor) lies on the contact
 // of its standing form with a surface; paper.z names that surface's owner.
 // The surface is the other half of the junction, not an occluder of it.
 let foot=(run.range.z&0x10000u)!=0u&&run.paper.w==0x10000001u;
 for(var scalar=0u;scalar<count;scalar++){
  var glyph=word;if(part==0u||part==3u){glyph=words[word*WS+2u+scalar+select(0u,sourceScalar,part==0u)];}
  let at=grammar[2]+glyph*4u;let first=grammar[at+2u];let pixels=grammar[at+3u];
  for(var pixel=0u;pixel<pixels;pixel++){
   let point=grammar[3]+(first+pixel)*2u;
   let q=fieldPixel(vec2f(f32(pen+bitcast<i32>(grammar[point]))+.5,f32(row*17+bitcast<i32>(grammar[point+1u]))+.5));
   total++;if(occludedCell(pen+bitcast<i32>(grammar[point]),row)){return false;}if(any(q<vec2i(0))||any(q>=vec2i(p.field.xy))){continue;}
   let owner=textureLoad(ownerField,q,0).xy;let same=((owner.x&0xdfffffffu)==run.identity.y&&owner.y==run.identity.z)||(foot&&owner.y==run.paper.z);
   // Transparent material keeps the background visible. Duplicate edge
   // contraction belongs to the shared hard-run placement, not occlusion.
   let foreground=(owner.x|owner.y)!=0u&&!same&&textureLoad(radianceField,q,0).a>=.99&&textureLoad(depthField,q,0).x>depth+.000001;
   if(!foreground){covered++;}
  }
  pen+=i32(grammar[at+1u]);
 }
 return covered*2u>total;
}
// Native row alignment may displace a contact onto its receiver. Visibility
// follows the contributing SOURCE span, not that displaced ink position.
// The four pixel centres surrounding the source retain a boundary with
// paper/own surface on either side. Starting at floor(q) instead biases the
// footprint down/right and discards bottom/right contacts against a receiver.
// An opaque foreign surface covering all four is real occlusion.
fn sourcePointVisible(run:Run,q:vec4f)->bool{
 if(occludedCell(i32(floor(q.x)),i32(floor(q.y/17.0)))){return false;}
 let origin=vec2i(floor((q.xy-p.rect.xy)/p.rect.zw*p.field.xy-vec2f(.5)));let size=vec2i(p.field.xy);
 for(var y=0;y<2;y++){for(var x=0;x<2;x++){
  let pixel=origin+vec2i(x,y);
  if(any(pixel<vec2i(0))||any(pixel>=size)){continue;}
  let owner=textureLoad(ownerField,pixel,0).xy;
  let same=(owner.x&0xdfffffffu)==run.identity.y&&owner.y==run.identity.z;
  if((owner.x|owner.y)==0u||same||textureLoad(radianceField,pixel,0).a<.99||textureLoad(depthField,pixel,0).x<=q.z+.000001){return true;}
 }}
 return false;
}
fn sourceGlyphVisible(run:Run,x:i32,row:i32,glyph:u32,a:Point,b:Point)->bool{
 let at=grammar[2]+glyph*4u;let first=grammar[at+2u];let pixels=grammar[at+3u];
 let delta=b.position.xy-a.position.xy;let lengthSquared=dot(delta,delta);var visible=0u;
 for(var pixel=0u;pixel<pixels;pixel++){
  let point=grammar[3]+(first+pixel)*2u;
  let ink=vec2f(f32(x+bitcast<i32>(grammar[point]))+.5,f32(row*17+bitcast<i32>(grammar[point+1u]))+.5);
  let t=clamp(dot(ink-a.position.xy,delta)/max(lengthSquared,.000001),0.0,1.0);
  if(sourcePointVisible(run,interpolatePoint(a,b,t))){visible++;}
 }
 // Same majority-ink policy as other native words. Each glyph remains intact;
 // visible neighbors retain the selected word's pens and scalar addresses.
 return visible*2u>pixels;
}
fn emitRecord(runID:u32,ordinal:u32,parameter:f32,x:i32,row:i32,word:u32,count:u32,advance:u32,priority:u32,part:u32,depth:f32,sourceScalar:u32,sourceEdge:u32,sourceContext:u32){
 let slot=atomicAdd(&counts[u32(row)],1u);
 atomicMax(&counts[rowCount()+3u],slot+1u);
 if(slot>=rowLimit(u32(row))){atomicOr(&counts[rowCount()],1u);return;}
 let at=rowBase(u32(row))+slot;
 var w:RowWord;
 w.data[0]=runID;w.data[1]=ordinal;w.data[2]=bitcast<u32>(parameter);w.data[3]=u32(x);
 w.data[4]=u32(row);w.data[5]=word;w.data[6]=count;w.data[7]=advance;w.data[8]=priority;w.data[9]=part;w.data[10]=lineInk(runID,x,row,advance);
 w.data[11]=bitcast<u32>(depth);w.data[12]=select(1u,0u,(runs[runID].range.z&0x40000u)!=0u);w.data[13]=sourceScalar;
 // Whole planar runs are assembled atomically. Their point-buffer slot is
 // NOT an edge identity. Carry the exact source address beside the local
 // unit ordinal through scalar cuts, publication and explicit export.
 // These two fields are region coordinates for part 3, edge/tag for part 0.
 w.data[14]=select(sourceEdge,0u,sourceEdge==NONE);w.data[15]=u32(sourceEdge!=NONE);
 w.data[16]=sourceContext;
 rowWords[at]=w;
}
fn emit(runID:u32,ordinal:u32,parameter:f32,x:i32,row:i32,word:u32,count:u32,advance:u32,priority:u32,part:u32,depth:f32){
 let absent=Point(vec4f(0),vec4u(0));
 emitMapped(runID,ordinal,parameter,x,row,word,count,advance,priority,part,depth,absent,absent,false);
}
fn sourceWord(id:u32,ordinal:u32,q:vec4f,x:i32,row:i32,word:u32,count:u32,advance:u32,priority:u32,a:Point,b:Point){
 emitMapped(id,ordinal,q.w,x,row,word,count,advance,priority,0u,q.z,a,b,true);
}
fn spanWord(id:u32,ordinal:u32,q:vec4f,x:i32,row:i32,word:u32,count:u32,advance:u32,priority:u32,pair:u32){
 spanPartWord(id,ordinal,q,x,row,word,count,advance,priority,pair,0u);
}
fn spanPartWord(id:u32,ordinal:u32,q:vec4f,x:i32,row:i32,word:u32,count:u32,advance:u32,priority:u32,pair:u32,part:u32){
 let run=runs[id];
 // Keep the compiled word and its source span together. The shared mapped
 // visibility pass clips whole native scalars on this span; it never asks the
 // grammar for a replacement word after an occlusion cut.
 emitMapped(id,ordinal,q.w,x,row,word,count,advance,priority,part,q.z,points[run.range.x+pair*2u],points[run.range.x+pair*2u+1u],true);
}
// A reconstructed span carries its SOURCE endpoints into the shared scalar
// visibility cut. Native row alignment is depiction, not a change of depth or
// ownership. Clip hidden members of the chosen word, never the whole word by
// one sample and never choose a replacement glyph independently.
fn emitMapped(runID:u32,ordinal:u32,parameter:f32,x:i32,row:i32,word:u32,count:u32,advance:u32,priority:u32,part:u32,depth:f32,a:Point,b:Point,mapped:bool){
 if(row<0||row>=i32(rowCount())||row*17+17>i32(p.extent.y)){return;}
 // A run ends at its hard-junction glyph (hardJunction): its own units centred on that glyph are the glyph.
 for(var side=0u;side<JUNCTION_HOLDS;side++){let held=junctionHold[side];let centre=2*x+i32(advance);if(held.x<held.y&&(row==held.z||row==held.w)&&centre>=held.x&&centre<=held.y){return;}}
 for(var k=0u;k<boxHolds;k++){let held=boxHold[k];let centre=2*x+i32(advance);if(held.x<held.y&&(row==held.z||row==held.w)&&centre>=held.x&&centre<=held.y){return;}}
 // The run producer owns its private junction holds. Visibility does not:
 // retain just the current word and its exact projected source span, then
 // evaluate words in parallel instead of nesting glyph-ink walks in a run.
 let slot=atomicAdd(&counts[p.proposals.z],1u);if(slot>=p.proposals.y){return;}
 let at=p.proposals.x+slot*${WORD_PROPOSAL_WORDS}u;
 let collapse=select(0u,gridKeyBase|(collapseDir&15u),gridKeyBase!=0u&&collapseDir!=NONE&&part==0u);
 let data=array<u32,${WORD_PROPOSAL_WORDS}>(runID,ordinal,bitcast<u32>(parameter),bitcast<u32>(x),u32(row),word,count,advance,priority,part,bitcast<u32>(depth),u32(mapped),a.visible.y,b.visible.y,a.visible.w,collapse,
  bitcast<u32>(a.position.x),bitcast<u32>(a.position.y),bitcast<u32>(a.position.z),bitcast<u32>(a.position.w),
  bitcast<u32>(b.position.x),bitcast<u32>(b.position.y),bitcast<u32>(b.position.z),bitcast<u32>(b.position.w),a.visible.z);
 for(var i=0u;i<${WORD_PROPOSAL_WORDS}u;i++){atomicStore(&counts[at+i],data[i]);}
}
// The incumbent whole-word visibility algorithm, unchanged. The only move
// is its execution owner: one candidate word, not a whole source run.
fn realizeMapped(runID:u32,ordinal:u32,parameter:f32,x:i32,row:i32,word:u32,count:u32,advance:u32,priority:u32,part:u32,depth:f32,a:Point,b:Point,mapped:bool,collapse:u32){

 // The selection grid (gridWrite): a parallel same-owner line of higher rank
 // holds this cell. The word is that line; recorded, not placed.
 if(gridCollapsed(collapse,x,row,advance)){probeAdd(34u);return;}
 let context=select(0u,a.visible.z,terrainRun(runs[runID])&&a.visible.z!=NONE&&(a.visible.z&0x80000000u)!=0u);
 let sourceEdge=select(NONE,pointFeatureId(a),context!=0u||(mapped&&(wholeBoundaryFlags(runs[runID])&0x100000u)!=0u));
 let source=spanSource(runs[runID],context);
 if(part!=0u){
  if(x>=0&&x+i32(advance)<=i32(p.extent.x)&&visibleWord(source,x,row,word,count,part,depth,0u)){
   emitRecord(runID,ordinal,parameter,x,row,word,count,advance,priority,part,depth,0u,NONE,0u);
  }
  return;
 }
 // Visibility and viewport cuts act on INTACT SCALARS of the already
 // selected whole word. A hidden/off-page member cannot erase its visible
 // neighbors, nor does a majority-visible word authorize foreground ink.
 // Visibility cuts are the one legitimate scalar clip: an occluded/off-page
 // scalar is removed, but every surviving scalar keeps its original ordinal,
 // pen and source parameter. Collision placement never enters this path.
 // A hard-line unit is already emitted only on a VISIBLE source interval.
 // Its ink is a whole native line that sits up to half a row from the
 // source edge; at a contact (a box foot on the ground, a wall base) that
 // displaced ink lands on the touching surface, which is nearer than the
 // edge itself and would read as an occluder. The source interval owns a
 // hard unit's visibility; only the page bounds clip it here.
 let hardUnit=(runs[runID].range.z&0x40000u)!=0u;
 var mask=0u;var pen=x;var complete=x>=0&&x+i32(advance)<=i32(p.extent.x);
 for(var scalar=0u;scalar<count;scalar++){
  let font=words[word*WS+2u+scalar];let at=grammar[2]+font*4u;let width=i32(grammar[at+1u]);
  if(grammar[at+3u]!=0u){
   var visible=hardUnit;
   if(mapped){visible=sourceGlyphVisible(source,pen,row,font,a,b);}
   else if(!hardUnit){visible=visibleWord(source,pen,row,font,1u,1u,depth,0u);}
   let keep=pen>=0&&pen+width<=i32(p.extent.x)&&visible;
   if(keep){mask|=1u<<scalar;}else{complete=false;}
  }
  pen+=width;
 }
 if(mask==0u){return;}
 if(complete){emitRecord(runID,ordinal,parameter,x,row,word,count,advance,priority,0u,depth,0u,sourceEdge,context);return;}
 // A visibility cut changes the visible range of a passage, not its
 // vocabulary or internal spacing. Keep each contiguous visible range as
 // one word-table slice through admission, sealing and export. Empty
 // scalars between surviving ink remain part of that same passage.
 pen=x;var start=0u;var startPen=x;var span=0u;var spanAdvance=0u;
 for(var scalar=0u;scalar<=count;scalar++){
  var width=0u;var hasInk=true;var keep=false;
  if(scalar<count){
   let font=words[word*WS+2u+scalar];let metric=grammar[2]+font*4u;
   width=grammar[metric+1u];hasInk=grammar[metric+3u]!=0u;keep=(mask&(1u<<scalar))!=0u;
  }
  if(keep){
   if(span==0u){start=scalar;startPen=pen;}
   span=scalar-start+1u;spanAdvance=u32(pen-startPen)+width;
  }else if(hasInk&&span>0u){
   emitRecord(runID,ordinal,parameter,startPen,row,word,span,spanAdvance,priority,0u,depth,start,sourceEdge,context);
   span=0u;
  }
  pen+=i32(width);
 }
}
@compute @workgroup_size(64) fn wordvisibility(@builtin(global_invocation_id) id:vec3u){
 let slot=id.x+id.y*p.proposals.w*64u;
 if(slot>=min(atomicLoad(&counts[p.proposals.z]),p.proposals.y)){return;}
 var data:array<u32,${WORD_PROPOSAL_WORDS}>;let at=p.proposals.x+slot*${WORD_PROPOSAL_WORDS}u;
 for(var i=0u;i<${WORD_PROPOSAL_WORDS}u;i++){data[i]=atomicLoad(&counts[at+i]);}
 let a=Point(bitcast<vec4f>(vec4u(data[16],data[17],data[18],data[19])),vec4u(0u,data[12],data[24],data[14]));
 let b=Point(bitcast<vec4f>(vec4u(data[20],data[21],data[22],data[23])),vec4u(0u,data[13],0u,0u));

 realizeMapped(data[0],data[1],bitcast<f32>(data[2]),bitcast<i32>(data[3]),i32(data[4]),data[5],data[6],data[7],data[8],data[9],bitcast<f32>(data[10]),a,b,data[11]!=0u,data[15]);

}
struct CircleSection {left:vec4f,right:vec4f,visible:u32}
fn circleSection(run:Run,y:f32)->CircleSection{
 var section=CircleSection(vec4f(3.402823e38,0,0,0),vec4f(-3.402823e38,0,0,0),0u);
 // The source's exact retained edge/visibility pairs, not an ellipse fitted
 // to a screen rectangle. Visibility never creates new contour anchors.
 for(var i=0u;i+1u<run.range.y;i+=2u){let a=points[run.range.x+i];let b=points[run.range.x+i+1u];
  if((a.position.y<=y&&b.position.y>y)||(b.position.y<=y&&a.position.y>y)){
   let q=interpolatePoint(a,b,(y-a.position.y)/(b.position.y-a.position.y));
   if(q.x<section.left.x){section.left=q;section.visible=(section.visible&2u)|select(0u,1u,pointVisible(a));}
   if(q.x>section.right.x){section.right=q;section.visible=(section.visible&1u)|select(0u,2u,pointVisible(a));}
  }
 }
 return section;
}
fn circleSupport(run:Run,x:f32,row:i32,width:f32)->bool{
 // A whole native role occupies a row, not one infinitesimal silhouette
 // sample. Keep it when its footprint meets a visible interval of this
 // source cycle; emit still clips ink hidden by another visible owner.
 // Sampling only the row's section point erased a whole shoulder/flank
 // when that point fell in a short self-occluded raster partition.
 let low=vec2f(x,f32(row*17));let high=low+vec2f(width,17.0);
 for(var i=0u;i+1u<run.range.y;i+=2u){let a=points[run.range.x+i];let b=points[run.range.x+i+1u];
  if(!pointVisible(a)){continue;}
  // Exact segment/role-rectangle intersection. Bounding-box overlap alone
  // can falsely support a role across a diagonal gap in the source.
  let delta=b.position.xy-a.position.xy;var lo=0.0;var hi=1.0;
  for(var axis=0u;axis<2u;axis++){
   if(abs(delta[axis])<.000001){
    if(a.position[axis]<low[axis]||a.position[axis]>high[axis]){hi=-1.0;}
   }else{
    let t0=(low[axis]-a.position[axis])/delta[axis];let t1=(high[axis]-a.position[axis])/delta[axis];
    lo=max(lo,min(t0,t1));hi=min(hi,max(t0,t1));
   }
  }
  if(lo<=hi){return true;}
 }
 return false;
}
fn circleWord(id:u32,ordinal:u32,q:vec4f,x:i32,row:i32,word:u32,priority:u32,side:bool){
 // The construction already attached this word to q on its source cycle.
 // Row alignment and the retained passage's native offsets need not put
 // its advance rectangle across that cycle again. Requiring intersection
 // here erased complete visible shoulders before shared ink visibility
 // could run. The shared emitter still clips each intact scalar against
 // actual foreground depth/ownership and the page; no occlusion bypass.
 emit(id,ordinal,q.w,x,row,word,words[word*WS],words[word*WS+1u],priority,0u,q.z);
}
fn circleSpan(a:vec4f,b:vec4f,t:f32)->vec4f{
 var q=mix(a,b,t);var delta=b.w-a.w;
 // A closed source cycle has a seam at 0/1. A cap across that seam belongs
 // there, not halfway around the back of the loop. The native span itself
 // is unchanged; only its circular source coordinate is interpolated.
 if(delta>.5){delta-=1.0;}else if(delta<-.5){delta+=1.0;}
 q.w=fract(a.w+delta*t);return q;
}
fn circlePhase(sourceRow:u32)->f32{if(sourceRow<3u){return .99;}if(sourceRow>4u){return .01;}return .5;}
// One side's unit for an interior round row, from the projected chord's own
// horizontal travel across the row (17 px): ／ ＼ is a 45° unit, / ｀、 ､ /
// travel about a third of a cell, l is upright. The nearest unit is the
// construction; nothing is fitted to pixels and no row is a repeat of another.
fn circleSideRow(above:f32,below:f32,open:bool,upper:bool)->u32{
 let slope=select(abs(below-above)/17.0,1.0,open);
 if(slope>=.65){return select(5u,1u,upper);}
 if(slope>=.2){return select(4u,2u,upper);}
 return 3u;
}
// The program row a large round takes at a row (the same rule as the loop in
// circleConstruction: cap, shoulders, base, else the chord's slope).
fn circleRowUnit(run:Run,row:u32,height:u32,low:vec2f,high:vec2f)->u32{
 if(row>=height){return 0xffffffffu;}
 if(row==0u){return 0u;}if(row+1u==height){return 6u;}if(row==1u){return 1u;}if(row+2u==height){return 5u;}
 let above=circleSection(run,mix(low.y,high.y,f32(row)/f32(height)));let below=circleSection(run,mix(low.y,high.y,f32(row+1u)/f32(height)));
 let open=above.left.x>above.right.x||below.left.x>below.right.x;
 let travel=abs((below.right.x-below.left.x)-(above.right.x-above.left.x))*.5;
 return circleSideRow(0.0,travel,open,row*2u+1u<height);
}
fn circleConstruction(id:u32,run:Run,priority:u32){
 var low=vec2f(3.402823e38);var high=vec2f(-3.402823e38);
 for(var i=0u;i<run.range.y;i++){let q=points[run.range.x+i].position.xy;low=min(low,q);high=max(high,q);}
 let height=max(2u,u32(round((high.y-low.y)/17.0)));let top=i32(round(low.y/17.0));let program=grammar[7];
 if(height<=4u){
  // Compact circles are the original COMPLETE library entries, selected
  // by height, once per form. They are never placed at contour cuts.
  let at=program+(height-2u)*4u;let first=grammar[at];let count=grammar[at+1u];let width=grammar[at+2u];
  let origin=i32(round((low.x+high.x-f32(width))*.5));
  for(var i=0u;i<count;i++){let atom=grammar[1]+(first+i)*4u;let row=grammar[atom+2u];
   let section=circleSection(run,mix(low.y,high.y,(f32(row)+.5)/f32(height)));
   if(!circleSupport(run,low.x,top+i32(row),high.x-low.x)){continue;}
   let q=circleSpan(section.left,section.right,.5);
   emit(id,i,q.w,origin+bitcast<i32>(grammar[atom+1u]),top+i32(row),grammar[atom],1u,grammar[atom+3u],priority,1u,q.z);
  }
  return;
 }
 // The paired cap/shoulder/curved-flank/return roles form ONE height program.
 // More height inserts complete upper/lower side units between enduring
 // cap/shoulder/return anchors. No glyph is stretched or chosen by a fit.
 let start=u32(max(0,-top));let end=min(height,u32(max(0,i32(rowCount())-top)));
 for(var row=start;row<end;row++){

  // stands on the same side in two consecutive rows. Where the contour stays
  // within the upright band for two rows or more (a large round's equator:

  // is OPEN between its entry and return turns (γ⌒ヽ / {　} / 乂__乂), never
  // a stack of uprights; a lone upright-band row keeps its l.
  if(circleRowUnit(run,row,height,low,high)==3u&&((row>0u&&circleRowUnit(run,row-1u,height,low,high)==3u)||circleRowUnit(run,row+1u,height,low,high)==3u)){continue;}
  // aa[57] / hand-authored target rows: 0 cap, 1 ／ ＼ shoulder, 2 the / ｀、
  // entry, 3 l l flank, 4 the ､ / return entry, 5 ＼ ／, 6 the separated
  // base. The cap, its shoulder, the lower shoulder and the base are the
  // enduring anchors; every other row takes, PER SIDE, the unit whose slope
  // matches this row's projected chord: the shoulder near 45°, the entries
  // when steeper, l only where the contour is within ~9° of vertical. No row
  // is repeated for height: a taller or foreshortened round gets more entry
  // rows, never a stack of uprights (an ellipse differs on its two sides).
  var leftRow=3u;var rightRow=3u;
  if(row==0u){leftRow=0u;rightRow=0u;}else if(row+1u==height){leftRow=6u;rightRow=6u;}
  else if(row==1u){leftRow=1u;rightRow=1u;}else if(row+2u==height){leftRow=5u;rightRow=5u;}
  else{
   let above=circleSection(run,mix(low.y,high.y,f32(row)/f32(height)));let below=circleSection(run,mix(low.y,high.y,f32(row+1u)/f32(height)));
   let upper=row*2u+1u<height;let open=above.left.x>above.right.x||below.left.x>below.right.x;
   // The chord's width change across the row, shared by both sides: a round
   // form's two flanks are one turn, and a faceted source silhouette must
   // not flip one side's unit a row early (the targets are symmetric).
   let travel=abs((below.right.x-below.left.x)-(above.right.x-above.left.x))*.5;
   leftRow=circleSideRow(0.0,travel,open,upper);rightRow=leftRow;
  }
  let at=program+12u+leftRow*8u;let atR=program+12u+rightRow*8u;let left=grammar[at];let right=grammar[atR+1u];let fill=grammar[at+4u];
  // Sample each side's chord where its OUTERMOST ink lies: the cap marks and
  // the entry slashes sit low in their cells, the returns and the base's
  // ｀ ´ start at the row top, the flanks and the ､ / entry at mid-row.
  let section=circleSection(run,mix(low.y,high.y,(f32(row)+circlePhase(leftRow))/f32(height)));
  var sectionR=section;if(rightRow!=leftRow){sectionR=circleSection(run,mix(low.y,high.y,(f32(row)+circlePhase(rightRow))/f32(height)));}
  if(section.left.x>section.right.x||sectionR.left.x>sectionR.right.x){continue;}
  let nativeRow=top+i32(row);let anchor=(leftRow==0u||leftRow==1u||leftRow==5u||leftRow==6u);
  let weight=select(priority,0x80000000u|min(priority,0x7fffffffu),anchor);
  let unit=select(select(height-1u-row,row,leftRow<3u),row,leftRow==3u);
  let ordinal=(leftRow<<24u)|(rightRow<<20u)|(unit<<12u);
  if(right==0u){
   let advance=words[left*WS+1u];let count=max(1u,u32(floor((section.right.x-section.left.x)/f32(advance))));
   let origin=i32(round((section.left.x+section.right.x-f32(count*advance))*.5));
   for(var i=0u;i<count;i++){let q=circleSpan(section.left,section.right,(f32(i)+.5)/f32(count));circleWord(id,ordinal+i,q,origin+i32(i*advance),nativeRow,left,weight,false);}
  }else{
   let leftX=i32(round(section.left.x))-bitcast<i32>(grammar[at+2u]);let rightX=i32(round(sectionR.right.x))-bitcast<i32>(grammar[atR+3u]);
   circleWord(id,ordinal,section.left,leftX,nativeRow,left,weight,true);
   circleWord(id,ordinal|0x800u,sectionR.right,rightX,nativeRow,right,weight,true);
   // Whole fill units (⌒ on the cap, ＿ on the base) inserted between the
   // enduring turn ends as the form widens; none when the ends already meet.
   if(fill!=0u){
    let start=leftX+bitcast<i32>(grammar[at+6u]);let end=rightX+bitcast<i32>(grammar[at+5u]);let advance=i32(words[fill*WS+1u]);
    if(end-start>=advance){
     let count=u32((end-start)/advance);let origin=start+(end-start-i32(count)*advance)/2;
     for(var i=0u;i<count;i++){let q=circleSpan(section.left,section.right,(f32(i)+.5)/f32(count));circleWord(id,ordinal|0x400u|i,q,origin+i32(i)*advance,nativeRow,fill,weight,false);}
    }
   }
  }
 }
}
// A planar volume (a cuboid, a pedestal, a slab) is ONE native box built from
// its own projected edges and the vertices they share, as the hand-authored
// targets spell it:  ＿＿＿  (back top edge)  /  ∠＿＿＿／|  (front top edge
// with the two depth edges leaving its corners, the back upright)  /
// |  :;:  |  |  (front and back uprights)  /  |＿＿＿_|/  (base with the half
// unit remainder, the narrow depth edge beside the upright). Every unit is
// placed from the edge it draws; corners come from shared source vertices,
// never from a silhouette span or a fitted glyph.
// 1 upright, 2 lid/base (the complete edge stays within half a native row),
// 3 a stepped edge: one unit per row, chosen by the edge's
// horizontal travel (/ ／ ＼ ＼､ ｀ヽ､､. _,.-‐´).
fn boxKind(d:vec2f)->u32{let ad=abs(d);if(ad.x<.25*ad.y){return 1u;}if(ad.y<8.5){return 2u;}return 3u;}
// Open hard paths keep the source turn, but do not inherit a box's upright
// approximation or its closed smallest tier. Near-vertical slants use the
// existing compiled straight-edge program, once for the complete edge.
fn boundaryKind(run:Run,d:vec2f)->u32{
 if((run.range.z&0x4000000u)==0u){return boxKind(d);}
 let ad=abs(d);
 if(ad.x<=.5){return 1u;}
 if(ad.y<8.5){return 2u;}
 if(ad.x<.25*ad.y){return 4u;}
 return 3u;
}
// Rows owned by a source span. A sub-row span may cross no row centre;
// it still owns one intact unit on its containing row, not an empty loop.
// Visibility and viewport clipping remain the sourceWord consumer's job.
// Clip execution, not construction. Source corners and unit ordinals remain
// those of the complete passage even when a near-plane endpoint is far off
// screen. Iterating that invisible prefix can otherwise exhaust the GPU.
fn pageRows(rows:vec2i)->vec2i{
 return vec2i(max(0,rows.x),min(i32(p.extent.y/17u)-1,rows.y));
}
fn pageUnits(origin:i32,step:i32,count:u32,width:u32)->vec2u{
 if(step==0||count==0u||width>p.extent.x){return vec2u(0);}
 let distance=abs(f32(step));let last=f32(p.extent.x-width);
 let low=select((f32(origin)-last)/distance,-f32(origin)/distance,step>0);
 let high=select(f32(origin)/distance,(last-f32(origin))/distance,step>0);
 let first=u32(clamp(ceil(low),0.0,f32(count)));
 let end=u32(clamp(floor(high)+1.0,f32(first),f32(count)));
 return vec2u(first,end);
}
fn boxRows(lowY:f32,highY:f32)->vec2i{
 let rows=vec2i(i32(round(lowY/17.0)),i32(round(highY/17.0))-1);
 if(rows.y>=rows.x){return rows;}
 let row=i32(floor((lowY+highY)*.5/17.0));return vec2i(row,row);
}
// All incident runs use the SAME native row for a projected source corner.
// Keep the source Point intact: depicting a corner on a row boundary must
// not move the geometry used by sourceWord's depth/ownership visibility.
fn boxNativeY(q:Point)->f32{return round(q.position.y/17.0)*17.0;}
fn boxNativeDelta(a:Point,b:Point)->vec2f{
 let native=vec2f(b.position.x-a.position.x,boxNativeY(b)-boxNativeY(a));
 // A visible sub-row edge still owns its one-row terminal unit. Shared
 // corner projection may collapse its native endpoints, not erase the span.
 if(dot(native,native)<1.0){return b.position.xy-a.position.xy;}
 return native;
}
fn boxEdgeAt(a:Point,b:Point,y:f32)->vec4f{
 let start=boxNativeY(a);let delta=boxNativeY(b)-start;
 var t=.5;if(abs(delta)>.001){t=clamp((y-start)/delta,0.0,1.0);}
 let q=interpolatePoint(a,b,t);
 return vec4f(q.x,mix(start,boxNativeY(b),t),q.zw);
}
// Face packets carry a tagged descriptor pair in the same resident point
// range as their finite draw pair. Re-project the original homogeneous
// endpoints here for row/corner/lattice decisions; the finite pair remains
// the only pair passed to sourceWord and scalar visibility.
struct WholeEdgeAnchors {a:Point,b:Point,found:bool}
fn wholeAnchorPage(point:Point,parameter:f32)->Point{
 let q=point.position;let w=select(select(-.00001,.00001,q.w>=0.0),q.w,abs(q.w)>=.00001);
 let xy=(q.xy/w*vec2f(.5,-.5)+vec2f(.5))*p.rect.zw+p.rect.xy;
 return Point(vec4f(xy,q.z/w,parameter),vec4u(0u,bitcast<u32>(1.0/w),point.visible.z,point.visible.w));
}
fn wholeEdgeAnchors(run:Run,feature:u32)->WholeEdgeAnchors{
 var out=WholeEdgeAnchors(Point(vec4f(0.0),vec4u(0u)),Point(vec4f(0.0),vec4u(0u)),false);
 for(var i=0u;i+1u<run.range.y;i+=2u){
  let a=points[run.range.x+i];if(!pointWholeEdgeDescriptor(a)||pointFeatureId(a)!=feature){continue;}
  out=WholeEdgeAnchors(wholeAnchorPage(a,0.0),wholeAnchorPage(points[run.range.x+i+1u],1.0),true);return out;
 }
 return out;
}
fn faceEdgeOrdinal(point:Point)->u32{return pointFeatureId(point);}
// One shallow edge of a planar construction. The retained turn is the whole row passage;
// width is added with whole baseline units on its low end. Visibility clips
// the selected passage against the source span, never reseeds its members.
// A STRAIGHT edge flatter than the whole-cell slash is spelled glyph by glyph
// along its row at the edge's own height there (HARD_SLOPE_PHASES: a rising
// edge climbs _ , ‐ ´ ￣, a falling one ￣ ｀ ‐ ､ _), read from the compiled
// clean-line table (grammar[8]) by the edge's direction and each glyph's row
// phase. The rise is spread over the whole row, so the edge reads straight.
// It replaces one turn word (_,.-‐´ / ｀ヽ､､.) plus a flat _ fill, which put the
// whole rise at one end of every row: a straight box edge read as a wavy
// staircase and a cube as a pillow (moving-b minimal, the rock island). Each
// row starts where the edge enters it, so consecutive rows chain.
fn shallowEdgeRow(id:u32,ordinal:u32,a:Point,b:Point,wholeA:Point,wholeB:Point,row:i32,priority:u32,size:u32){
 if(row<0||row>=i32(p.extent.y/17u)){return;}
 let d=boxNativeDelta(wholeA,wholeB);
 let screenDelta=b.position.xy-a.position.xy;
 let screenAdvance=select(1.0,screenDelta.x,abs(screenDelta.x)>=.001);
 // The graded column of the edge's side, also for an edge under
 // TERRAIN_STEP_SLOPE: that column's level rule (＿ low, ￣ high) left a
 // near-level edge as two flat runs a whole row apart with nothing joining
 // them (the slab's front edge). Graded, the step reads ＿＿_,‐´￣￣.
 let rising=d.x*d.y<0.0;
 let dir=select(select(3u,35u,d.x<0.0),select(61u,29u,d.x<0.0),rising);
 let lid=grammar[grammar[7]+12u+7u*8u+2u];
 let qb=boxEdgeAt(a,b,f32(row+1)*17.0);
 let qt=boxEdgeAt(a,b,f32(row)*17.0);
 let left=min(qb.x,qt.x);let right=max(qb.x,qt.x);
 // The row's one turn (_ , ‐ ´ climbing, ￣ ｀ ‐ ､ falling) is centred where
 // the edge crosses the middle of the row, over at most 48 px; the edge lies
 // on the row's lower half before it (＿) and its upper half after it (￣), so
 // a near-level edge is ＿＿_,‐´￣￣ (AAHub's flat runs and one compact turn),
 // not a run of mid-height ‐‐‐‐､､､､ where it lingers between the two edges.
 // A 24 px turn left a 13 degree rim (75 px a row) as separate flat dashes.
 let turnWidth=min(right-left,48.0);
 // In the box's native frame (endpoints on row boundaries), as left/right.
 let middle=boxEdgeAt(a,b,f32(row)*17.0+8.5).x;
 let turnStart=middle-turnWidth*.5;
 let lowEnd=select(right,turnStart+2.0,rising);
 var x=i32(round(left));
 for(var k=0u;k<160u;k++){
  // The phase is read a narrow glyph's half advance into the cell.
  let probe=f32(x)+3.0;
  var y=select(16.0,0.0,rising);
  if(probe<turnStart){y=select(0.0,16.0,rising);}
  else if(probe<=turnStart+turnWidth){let s=(probe-turnStart)/max(turnWidth,1.0);y=select(16.0*s,16.0*(1.0-s),rising);}
  let phase=u32(clamp(floor(y),0.0,16.0));
  let instruction=grammar[8]+(((2u*3u+size)*64u+dir)*17u+phase)*12u;
  var word=grammar[instruction];
  // A whole cell on the row's lower edge is the full-width ＿ (the lid unit),
  // not three half-width _.
  if(y>=${HARD_SLOPE_PHASES[3]}.0&&f32(x)+16.0<=lowEnd){word=lid;}
  let advance=i32(words[word*WS+1u]);
  if(k>0u&&f32(x)+f32(advance)*.5>right){break;}
  let q=interpolatePoint(a,b,clamp((f32(x)+f32(advance)*.5-a.position.x)/screenAdvance,0.0,1.0));
  sourceWord(id,select(ordinal|0x400u|k,ordinal,k==0u),q,x,row,word,words[word*WS],u32(advance),priority,a,b);
  x+=advance;
 }
}
// 1: a visible depth edge rises to the right from the vertex q (／ or ∠),
// 2: one rises to the left (＼), 0: none.
fn boxSameCorner(a:Point,b:Point)->bool{
 return a.visible.z!=NONE&&a.visible.z==b.visible.z;
}
fn boxDiagonalFrom(run:Run,q:Point)->u32{
 for(var i=0u;i+1u<run.range.y;i+=2u){let a=points[run.range.x+i];let b=points[run.range.x+i+1u];if(!pointVisible(a)){continue;}
  let anchors=wholeEdgeAnchors(run,pointFeatureId(a));var wholeA=a;var wholeB=b;if(anchors.found){wholeA=anchors.a;wholeB=anchors.b;}
  let d=boxNativeDelta(wholeA,wholeB);if(dot(d,d)<1.0||boundaryKind(run,d)<3u){continue;}
  var lower=a;var upper=b.position.xy;if(b.position.y>a.position.y){lower=b;upper=a.position.xy;}
  if(!boxSameCorner(lower,q)){continue;}
  return select(2u,1u,upper.x>lower.position.x);
 }
 return 0u;
}
// A visible lid/base edge leaves the vertex q to the right.
fn boxHorizontalFrom(run:Run,q:Point)->bool{
 for(var i=0u;i+1u<run.range.y;i+=2u){let a=points[run.range.x+i];let b=points[run.range.x+i+1u];if(!pointVisible(a)){continue;}
  let anchors=wholeEdgeAnchors(run,pointFeatureId(a));var wholeA=a;var wholeB=b;if(anchors.found){wholeA=anchors.a;wholeB=anchors.b;}
  let d=boxNativeDelta(wholeA,wholeB);if(dot(d,d)<1.0||boundaryKind(run,d)!=2u){continue;}
  var left=a;if(b.position.x<a.position.x){left=b;}
  if(boxSameCorner(left,q)){return true;}
 }
 return false;
}
// A visible upright of this box occupies the cell at x on this row.
fn boxUprightAt(run:Run,x:f32,row:i32)->bool{
 for(var i=0u;i+1u<run.range.y;i+=2u){let a=points[run.range.x+i];let b=points[run.range.x+i+1u];if(!pointVisible(a)){continue;}
  let anchors=wholeEdgeAnchors(run,pointFeatureId(a));var wholeA=a;var wholeB=b;if(anchors.found){wholeA=anchors.a;wholeB=anchors.b;}
  let d=boxNativeDelta(wholeA,wholeB);if(dot(d,d)<1.0||boundaryKind(run,d)!=1u){continue;}
  let rows=boxRows(min(wholeA.position.y,wholeB.position.y),max(wholeA.position.y,wholeB.position.y));
  if(row<rows.x||row>rows.y){continue;}
  let at=boxEdgeAt(a,b,f32(row)*17.0+8.5).x;
  if(abs(at-x)<3.0){return true;}
 }
 return false;
}
// BOX JUNCTIONS (compiled-role-grammar.ts compileJunctions, built vocabulary).
// Where two or three drawn edges of one planar construction meet at a source
// vertex (a lid and an upright, a depth edge and a lid, the corner of a step),
// or where an edge's free end stops on the interior of another of its edges
// (a post under a lid: a T), the meeting is ONE compiled glyph read by the
// legs' sixteenths, emitted once at the vertex; every edge then holds back its
// own units centred on it (boxHold, read in emitMapped), the contract of
// hardJunction. A leg's direction is the drawn class of its edge: an upright
// is north or south, a lid east or west, any other edge its own sixteenth.
// Kept as before: the ∠ corner the lid loop places where a rising depth edge
// leaves the lid's left end. A step (a lid, an upright shorter than one row,
// the next lid) is spelled at the upright's LOWER end only, its glyph's upright
// reaching up to the upper lid. Nothing is ranked or tried: one record lookup
// per vertex, in edge order.
const BOX_HOLDS=24u;const BOX_EDGES=256u;
var<private> boxHold:array<vec4i,BOX_HOLDS>;
var<private> boxHolds:u32;
var<private> boxDrawn:array<u32,8>;
fn boxDrawnEdge(i:u32)->bool{return i<BOX_EDGES&&(boxDrawn[i/32u]&(1u<<(i%32u)))!=0u;}
fn boxLeg(run:Run,i:u32,end:u32)->u32{
 let a=points[run.range.x+2u*i];let b=points[run.range.x+2u*i+1u];
 let anchors=wholeEdgeAnchors(run,pointFeatureId(a));var wholeA=a;var wholeB=b;if(anchors.found){wholeA=anchors.a;wholeB=anchors.b;}
 let d=select(wholeB.position.xy-wholeA.position.xy,wholeA.position.xy-wholeB.position.xy,end==1u);
 let kind=boundaryKind(run,boxNativeDelta(wholeA,wholeB));
 if(kind==1u){return select(12u,4u,d.y>0.0);}
 if(kind==2u){return select(8u,0u,d.x>0.0);}
 return u32(round(fract(atan2(d.y,d.x)/6.28318530718+1.0)*16.0))&15u;
}
fn boxJunctions(id:u32,run:Run,weight:u32){
 boxHolds=0u;
 let edges=min(run.range.y/2u,BOX_EDGES);
 for(var k=0u;k<8u;k++){boxDrawn[k]=0u;}
 for(var i=0u;i<edges;i++){
  let a=points[run.range.x+2u*i];let b=points[run.range.x+2u*i+1u];if(!pointVisible(a)){continue;}
  let anchors=wholeEdgeAnchors(run,pointFeatureId(a));var wholeA=a;var wholeB=b;if(anchors.found){wholeA=anchors.a;wholeB=anchors.b;}
  let d=boxNativeDelta(wholeA,wholeB);if(dot(d,d)<1.0){continue;}
  let seam=surfaceOutlineRead(a.position.xy,b.position.xy);if(seam.x>0u&&seam.z*2u<=seam.x){continue;}
  boxDrawn[i/32u]|=1u<<(i%32u);
 }
 for(var i=0u;i<edges;i++){
  if(!boxDrawnEdge(i)){continue;}
  for(var e=0u;e<2u;e++){
   let q=points[run.range.x+2u*i+e];let vertex=q.visible.z;if(vertex==NONE){continue;}
   // The vertex is spelled from its first drawn edge; count its drawn legs.
   var first=true;var count=0u;var legs=0u;var shortDown=false;
   for(var j=0u;j<edges&&first;j++){
    if(!boxDrawnEdge(j)){continue;}
    for(var f=0u;f<2u;f++){
     if(points[run.range.x+2u*j+f].visible.z!=vertex){continue;}
     // The vertex is spelled from the drawn edge with the least SOURCE edge
     // address (point word visible.w), not the first in the reserved point
     // order, which is the atomic fill order of assemble.
     let ej=points[run.range.x+2u*j].visible.w;let ei=points[run.range.x+2u*i].visible.w;
     if(ej<ei||(ej==ei&&j<i)){first=false;break;}
     let leg=boxLeg(run,j,f);if(count<3u){legs|=leg<<(4u*count);}count++;
     let a=points[run.range.x+2u*j];let b=points[run.range.x+2u*j+1u];
     let anchors=wholeEdgeAnchors(run,pointFeatureId(a));var wholeA=a;var wholeB=b;if(anchors.found){wholeA=anchors.a;wholeB=anchors.b;}
     if(leg==4u&&boundaryKind(run,boxNativeDelta(wholeA,wholeB))==1u&&abs(wholeA.position.y-wholeB.position.y)<17.0){shortDown=true;}
    }
   }
   if(!first||count>3u){continue;}
   var node=q.position;var three=count==3u;
   if(count==1u){
    // A free end on the interior of another drawn edge of this form: a T.
    let own=points[run.range.x+2u*i+(1u-e)].position.xy-q.position.xy;var found=false;
    for(var j=0u;j<edges&&!found;j++){
     if(j==i||!boxDrawnEdge(j)){continue;}
     let a=points[run.range.x+2u*j];let b=points[run.range.x+2u*j+1u];let s=b.position.xy-a.position.xy;let l2=dot(s,s);
     if(l2<289.0||dot(own,own)<289.0){continue;}
     let t=dot(q.position.xy-a.position.xy,s)/l2;if(t<.1||t>.9){continue;}
     let across=abs((q.position.x-a.position.x)*s.y-(q.position.y-a.position.y)*s.x)/sqrt(l2);
     if(across>3.0||abs(dot(normalize(own),s/sqrt(l2)))>.70710678){continue;}
     legs|=(boxLeg(run,j,1u)<<4u)|(boxLeg(run,j,0u)<<8u);node=interpolatePoint(a,b,t);three=true;found=true;
    }
    if(!found){continue;}
   }else if(count==2u){
    if(shortDown){continue;}
    if(boxDiagonalFrom(run,q)==1u&&boxHorizontalFrom(run,q)){continue;}
   }
   let at=junctionRecord(legs,three,false);let word=grammar[at];if(word==0u){continue;}
   let held=junctionAt(at,node);
   emit(id,0x7fd00000u|(2u*i+e),node.w,i32(round(node.x))-i32(grammar[at+2u]),held.z,word,words[word*WS],grammar[at+1u],weight,0u,node.z);
   if(boxHolds<BOX_HOLDS){boxHold[boxHolds]=held;boxHolds++;}
  }
 }
}
${GPU_PLANAR_FAN}
// A FACE LID BORDERS ITS OWN SURFACE (slice 1b, the stray lid): a whole
// source face edge is the outline of its face only where the face's source
// lies beside it on the page. Visibility intervals do not create new edge
// endpoints or restart the native passage. A unit whose line has another
// source's pixels both above and below (a wall top running across another
// house's gable) is not placed. Off-field samples count as bordering (no

fn faceBorders(run:Run,x:f32,y:f32)->bool{
 for(var s=0u;s<2u;s++){
  let q=vec2i(floor(fieldPoint(vec2f(x,y+select(-5.0,5.0,s==1u)))));
  if(any(q<vec2i(0))||any(q>=vec2i(p.field.xy))){return true;}
  let owner=textureLoad(ownerField,q,0).xy;if((owner.x&0xdfffffffu)==run.identity.y&&owner.y==run.identity.z){return true;}
 }
 return false;
}
fn hardBoundaryConstruction(id:u32,run:Run,priority:u32,volume:bool){
 // THE FACE OWNER (slice 1b): a face run (bit 0x200000) is the box program over
 // its face's compiled edges, not the whole-object corner template.
 let faceRun=(run.range.z&0x200000u)!=0u;
 if(volume&&!faceRun&&planarFanConstruction(id,run,priority)){return;}
 let record=grammar[7]+12u+7u*8u;
 let upright=grammar[record];let lid=grammar[record+2u];let corner=grammar[record+3u];let rise=grammar[record+4u];let fall=grammar[record+5u];let narrow=grammar[record+6u];let remainder=grammar[record+7u];
 let fallTail=grammar[record+8u];
 let uprightAdvance=i32(words[upright*WS+1u]);let fillAdvance=i32(words[lid*WS+1u]);let restAdvance=i32(words[remainder*WS+1u]);let cornerAdvance=i32(words[corner*WS+1u]);let riseAdvance=i32(words[rise*WS+1u]);
 let anchorWeight=0x80000000u|min(priority,0x7fffffffu);
 // Smallest tier: a box under a row and a half, or one that cannot hold two
 // | cells and a lid unit and stands under FAR_STRIP_BELOW_ROWS rows, is ONE
 // connected outline glyph (口 ロ ﾛ by width) at its centre (D9: never dots,
 // never a lone upright), as a compact circle is one library entry. A
 // narrow box of three rows or more (a pillar, a post) is built from its
 // edges below over its own rows: one glyph never stands for a form taller

 var low=vec2f(3.402823e38);var high=vec2f(-3.402823e38);var seen=false;
 for(var i=0u;i+1u<run.range.y;i+=2u){let a=points[run.range.x+i];if(!pointVisible(a)){continue;}let b=points[run.range.x+i+1u];
  let anchors=wholeEdgeAnchors(run,pointFeatureId(a));var wholeA=a;var wholeB=b;if(anchors.found){wholeA=anchors.a;wholeB=anchors.b;}
  low=min(low,min(wholeA.position.xy,wholeB.position.xy));high=max(high,max(wholeA.position.xy,wholeB.position.xy));seen=true;}
 if(!seen){return;}
 let extent=high-low;
 // LINES3 STEP 2: under a row and a half tall, only a box too narrow to hold
 // a lid run of three full-width units (48 px) is the one stamp; a flat wide
 // box (an edge-on platform) is drawn by its edges below (D9 connected outline).
 if(volume&&((extent.y<25.5&&extent.x<48.0)||(extent.x<28.0&&extent.y<f32(${FAR_STRIP_BELOW_ROWS*17})))){
  var tiny=grammar[record+13u];if(extent.x>=14.0){tiny=grammar[record+11u];}else if(extent.x>=9.0){tiny=grammar[record+12u];}
  let advance=words[tiny*WS+1u];let centre=(low+high)*.5;let first=finiteRunStart(run);
  // The stamp ordinal tells the typesetter this anchor is a whole form's one
  // glyph, not a join: it keeps D1(1), its nearest legal pen (typesetWord).
  emit(id,STAMP_ORDINAL,first.position.w,i32(round(centre.x-f32(advance)*.5)),i32(floor(centre.y/17.0)),tiny,1u,advance,anchorWeight,0u,first.position.z);
  return;
 }
 boxJunctions(id,run,anchorWeight);
 for(var i=0u;i+1u<run.range.y;i+=2u){
  let a=points[run.range.x+i];let b=points[run.range.x+i+1u];if(!pointVisible(a)){continue;}
  let anchors=wholeEdgeAnchors(run,pointFeatureId(a));var wholeA=a;var wholeB=b;if(anchors.found){wholeA=anchors.a;wholeB=anchors.b;}
  let d=boxNativeDelta(wholeA,wholeB);if(dot(d,d)<1.0){continue;}
  // An edge whose two sides are one coplanar surface (two ground slabs or
  // tiles meeting flush, a platform's border lying on the ground) is a seam,
  // not an outline: same depth rule and crease turn as surfaceOutlineRead.
  // A box's own silhouettes and creases always read as outline.
  let seam=surfaceOutlineRead(a.position.xy,b.position.xy);if(seam.x>0u&&seam.z*2u<=seam.x){continue;}
  let kindOf=boundaryKind(run,d);let ordinal=select(select(i<<12u,0u,volume),faceEdgeOrdinal(a),faceRun);
  let rows=boxRows(min(wholeA.position.y,wholeB.position.y),max(wholeA.position.y,wholeB.position.y));
  if(kindOf==4u){
   var dir=u32(round(fract(atan2(d.y,d.x)/6.28318530718+1.0)*64.0))&63u;
   // Quantization cannot turn a visibly slanted path into an upright.
   if(dir==16u){dir=select(17u,15u,d.x>0.0);}
   if(dir==48u){dir=select(47u,49u,d.x>0.0);}
   let phase=0u;
   var instruction=grammar[8]+(((2u*3u+runSize(run))*64u+dir)*17u+phase)*12u;
   var word=grammar[instruction];var offset=bitcast<i32>(grammar[instruction+8u]);
   // D5: a CURVED chain never takes an upright. Within HARD_UPRIGHT_LEAN of
   // vertical the clean-line table spells a leaning post or jamb with |, which
   // on a curved chain (an arch, a dome, a vault's flank) stacked one | per row
   // down the curve (loop 351 view 02, the arch flank: eight lone |). An edge
   // on a curve (chainCurves) reads the natural-form column of the same direction
   // instead: its best-fitting natural unit, one per row (compiled-role-grammar
   // SLOPE FIT), never an upright; a straight edge keeps its upright run.
   // A near-upright piece leaning under two dots per row (a stone tower's
   // slightly kinked side) is still the practitioner upright run (featurecurves'
   // straight soft edge: at most one dot per row), curve or not.
   if(abs(d.x)*17.0>=2.0*abs(d.y)&&chainCurves(run,i)){
    instruction=grammar[11]+${(PAIRED_FLANK_LEANS.length+1)*4}u+(((2u*3u+runSize(run))*64u+dir)*17u+phase)*12u;
    word=grammar[instruction];offset=-(i32(grammar[instruction+6u]&255u)-128);
   }
   let visibleRows=pageRows(rows);
   for(var row=visibleRows.x;row<=visibleRows.y;row++){
    let q=boxEdgeAt(a,b,f32(row)*17.0+8.5);
    sourceWord(id,ordinal|u32(row-rows.x),q,i32(round(q.x))+offset,row,word,words[word*WS],words[word*WS+1u],priority,a,b);
   }
   continue;
  }
  if(kindOf==1u){
   // An upright on every row whose centre the edge crosses. Two uprights of
   // one form inside one full-width cell (a thin block's front and back edge,
   // a jamb's two faces: the || of view 02, 9 px apart; AGENTS finding 4,
   // parallels inside one glyph width are one) are one upright: the nearer
   // edge draws it.
   let visibleRows=pageRows(rows);
   for(var row=visibleRows.x;row<=visibleRows.y;row++){let q=boxEdgeAt(a,b,f32(row)*17.0+8.5);
    var doubled=false;
    for(var j=0u;j+1u<run.range.y&&!doubled;j+=2u){
     if(j==i){continue;}let c=points[run.range.x+j];let e=points[run.range.x+j+1u];if(!pointVisible(c)){continue;}
     let anchors=wholeEdgeAnchors(run,pointFeatureId(c));var wholeC=c;var wholeE=e;if(anchors.found){wholeC=anchors.a;wholeE=anchors.b;}
     let dj=boxNativeDelta(wholeC,wholeE);if(dot(dj,dj)<1.0||boundaryKind(run,dj)!=1u){continue;}
     let other=boxRows(min(wholeC.position.y,wholeE.position.y),max(wholeC.position.y,wholeE.position.y));if(row<other.x||row>other.y){continue;}
     let at=boxEdgeAt(c,e,f32(row)*17.0+8.5);let depth=at.z;
     if(abs(at.x-q.x)>=16.0){continue;}
     doubled=depth>q.z+.0000001||(abs(depth-q.z)<=.0000001&&j<i);
    }
    if(doubled){continue;}
    sourceWord(id,ordinal|u32(row-rows.x),q,i32(round(q.x))-1,row,upright,1u,u32(uprightAdvance),priority,a,b);}
   continue;
  }
  if(kindOf==2u){
   // A lid/base on the cell edge nearest the source line, so it meets the
   // uprights and the depth edges at their shared corner vertices.
   let row=i32(round((wholeA.position.y+wholeB.position.y)*.5/17.0))-1;
   if(row<0||row>=i32(p.extent.y/17u)){continue;}
   var leftEnd=a;var rightEnd=b;if(b.position.x<a.position.x){leftEnd=b;rightEnd=a;}
   // A top face thinner than half a row has no row of its own: its far edge
   // lands on the lid's row over the same cells, and the nearer edge is the
   // lid (|￣￣￣| when a box is seen edge-on). The farther one is not drawn.
   var covered=false;
   for(var j=0u;j+1u<run.range.y&&!covered;j+=2u){
    if(j==i){continue;}let c=points[run.range.x+j];let e=points[run.range.x+j+1u];if(!pointVisible(c)){continue;}
    let anchors=wholeEdgeAnchors(run,pointFeatureId(c));var wholeC=c;var wholeE=e;if(anchors.found){wholeC=anchors.a;wholeE=anchors.b;}
    let dj=boxNativeDelta(wholeC,wholeE);if(dot(dj,dj)<1.0||boundaryKind(run,dj)!=2u){continue;}
    if(i32(round((wholeC.position.y+wholeE.position.y)*.5/17.0))-1!=row){continue;}
    let lo=max(min(a.position.x,b.position.x),min(c.position.x,e.position.x));let hi=min(max(a.position.x,b.position.x),max(c.position.x,e.position.x));
    if(hi-lo<.5*abs(d.x)){continue;}
    let mine=(wholeA.position.z+wholeB.position.z)*.5;let theirs=(wholeC.position.z+wholeE.position.z)*.5;
    covered=theirs>mine+.0000001||(abs(theirs-mine)<=.0000001&&j<i);
   }
   if(covered){continue;}
   // One lid is one chain: the whole source edge is one native passage. Its
   // visibility intervals may hide individual native units, but they cannot
   // restart the lattice at a cut or turn a cut end into a corner. Collinear
   // source edges that touch end to end may still share a ＿ run under the
   // declared source-corner rule.
   let edge=pointFeatureId(a);
   var owner=true;
   for(var sweep=0u;sweep<4u&&owner;sweep++){
    for(var j=0u;j+1u<run.range.y;j+=2u){
     if(j==i){continue;}let c=points[run.range.x+j];let e=points[run.range.x+j+1u];if(!pointVisible(c)){continue;}
     let anchors=wholeEdgeAnchors(run,pointFeatureId(c));var wholeC=c;var wholeE=e;if(anchors.found){wholeC=anchors.a;wholeE=anchors.b;}
     let dj=boxNativeDelta(wholeC,wholeE);if(dot(dj,dj)<1.0||boundaryKind(run,dj)!=2u){continue;}
     if(i32(round((wholeC.position.y+wholeE.position.y)*.5/17.0))-1!=row||abs((wholeC.position.y+wholeE.position.y)*.5-(wholeA.position.y+wholeB.position.y)*.5)>2.5){continue;}
     var cl=c;var cr=e;if(e.position.x<c.position.x){cl=e;cr=c;}
     let sameEdge=pointFeatureId(c)==edge;
     // A face run's lid is one whole source edge (slice 1b): nothing joins it
     // merely because a visibility interval happened to be adjacent.
     if(faceRun||(!sameEdge&&!boxSameCorner(rightEnd,cl)&&!boxSameCorner(cr,leftEnd))){continue;}
     if(cl.position.x<leftEnd.position.x-.5||(abs(cl.position.x-leftEnd.position.x)<=.5&&j<i)){owner=false;break;}
     if(cr.position.x>rightEnd.position.x){rightEnd=cr;}
    }
   }
   if(!owner){continue;}
   let l=leftEnd.position.xy;let r=rightEnd.position.xy;
   var begin=i32(round(l.x));var finish=i32(round(r.x));
   let dl=boxDiagonalFrom(run,leftEnd);let dr=boxDiagonalFrom(run,rightEnd);
   if(dl==1u){sourceWord(id,ordinal|0x800u,leftEnd.position,begin,row,corner,1u,u32(cornerAdvance),anchorWeight,leftEnd,rightEnd);begin+=cornerAdvance;}
   else if(boxUprightAt(run,l.x,row)){begin+=uprightAdvance-1;}
   if(dr==2u){finish-=riseAdvance;}else if(dr==0u&&boxUprightAt(run,r.x,row)){finish-=uprightAdvance-1;}
   // Whole ＿ units from the left corner, then the half-width _ remainder at
   // the right (∠＿＿＿／, |＿＿＿_|): the targets never stretch a lid, leave
   // a corner open or float the run between its corners.
   let width=finish-begin;if(width<restAdvance){continue;}
   let count=u32(width/fillAdvance);let rest=width-i32(count)*fillAdvance;let extra=select(0u,1u,rest>=restAdvance);
   let origin=begin;let span=max(1.0,r.x-l.x);
   let visibleUnits=pageUnits(origin,fillAdvance,count,u32(fillAdvance));
   for(var k=visibleUnits.x;k<visibleUnits.y;k++){let x=origin+i32(k)*fillAdvance;let q=interpolatePoint(leftEnd,rightEnd,clamp((f32(x)+f32(fillAdvance)*.5-l.x)/span,0.0,1.0));
    if(faceRun&&!faceBorders(run,f32(x)+f32(fillAdvance)*.5,q.y)){continue;}
    sourceWord(id,ordinal|0x400u|k,q,x,row,lid,1u,u32(fillAdvance),anchorWeight,leftEnd,rightEnd);}
   if(extra==1u){let x=origin+i32(count)*fillAdvance;let q=interpolatePoint(leftEnd,rightEnd,clamp((f32(x)+f32(restAdvance)*.5-l.x)/span,0.0,1.0));
    sourceWord(id,ordinal|0x400u|count,q,x,row,remainder,1u,u32(restAdvance),anchorWeight,leftEnd,rightEnd);}
   continue;
  }
  // A depth edge: ／ rising to the right or ＼ falling, one unit per row it
  // crosses. At its lower vertex the ∠ corner already carries it when a lid
  // leaves that vertex to the right; beside an upright in the same row it is
  // the narrow / (|＿＿＿_|/); elsewhere the whole-cell slash.
  let rising=d.x*d.y<0.0;var lowerEnd=a;if(b.position.y>a.position.y){lowerEnd=b;}
  // A depth edge shorter than a row (a shallow top face) is still one unit,
  // in the row that holds it: the ／ that ends the lid, the / beside the base.
  var span=rows;if(span.y<span.x){let middle=i32(round(lowerEnd.position.y/17.0))-1;span=vec2i(middle,middle);}
  let visibleRows=pageRows(span);
  for(var row=visibleRows.x;row<=visibleRows.y;row++){
   if(row==span.y&&rising&&boxHorizontalFrom(run,lowerEnd)){continue;}
   let qb=boxEdgeAt(a,b,f32(row+1)*17.0);let qt=boxEdgeAt(a,b,f32(row)*17.0);
   // The row's horizontal travel chooses the stepped unit: a whole-cell slash
   // near 45°, ＼､ when the edge recedes faster than a cell per row, the
   // ｀ヽ､､. / _,.-‐´ passages when it is flatter still (fog target's room and
   // floor edges). The unit starts at the row's upper end for a falling edge
   // and at its lower end for a rising one, so consecutive rows chain.
   // Choose the passage once for the EDGE, not again at its clipped end
   // rows. A long shallow edge still crosses rows; its width grows by whole
   // baseline units beside the retained turn, never by flattening the edge.
   // Lowering this bound (to 21/27 px, 2026-11-11) sent the fog and
   // lit-interior ＼ chains through the row walk as ＿/￣ staircases and lost

   let travel=abs(d.x)*17.0/abs(d.y);
   if(travel>=30.0){
    shallowEdgeRow(id,ordinal|(u32(row-span.x)<<20u),a,b,wholeA,wholeB,row,priority,runSize(run));
    continue;
   }
   if(!rising){
    var word=fall;if(travel>=20.0){word=fallTail;}
    sourceWord(id,ordinal|u32(row-span.x),qt,i32(round(qt.x)),row,word,words[word*WS],words[word*WS+1u],priority,a,b);continue;
   }
   if(boxUprightAt(run,qb.x,row)){sourceWord(id,ordinal|u32(row-span.x),qb,i32(round(qb.x))+uprightAdvance-1,row,narrow,1u,u32(words[narrow*WS+1u]),priority,a,b);}
   else{sourceWord(id,ordinal|u32(row-span.x),qb,i32(round(qb.x)),row,rise,1u,u32(riseAdvance),priority,a,b);}
  }
 }
}
// 遠景 MERGED FAR SKYLINE (D3 exterior far field: ONE merged skyline strip;

// one base line with peaks and marks standing on it, medium-far keeps the
// hills' and masses' own rows). Far forms and the far terrain silhouette
// CONTRIBUTE to shared profiles on the page's 16 px column lattice and the
// strip is spelled once from them; no far form spells itself.
//  skylineReduce, one thread per run, before featurecurves. A selected 遠景
//  run does one atomic max per column it covers:
//   table 0 terrain: the highest far terrain silhouette of the column, and
//           the top of a far BOX (a planar volume's whole-box run): adjacent
//           small boxes are ONE mass whose skyline and shelf lines this
//           table spells (flat runs, rises, falls, ∧ peaks), never one box
//           stamp per form (bit 19: a run taller than two rows, which keeps
//           its own construction and only occupies the column);
//   table 1 mass:    the highest far standing soft or round form of the column;
//   table 2 mark:    a form taller than wide, no wider than FAR_MARK_WIDTH
//           and under FAR_STRIP_BELOW_ROWS rows, is ONE mark at its own
//           centre column (the tallest of the column).
//  A word is (4095 - page y of the top) << 20 | flag << 19 | run id, so the
//  maximum is the highest surface and names the run that spells the column.
//  farSkyline, in featurecurves: the run a column names emits that column's
//  unit, read from the compiled table (grammar[9] + 99) by class and by which
//  neighbour columns are lower: flat run, rise, fall or peak on the column's
//  level row, then one flank unit per row down to the lower neighbour. A
//  level is the row whose top edge is nearest the column's top. A mass falls
//  to its neighbour mass or to the ground: the terrain level beside it when
//  that is within a row of the form's foot, else its own foot, so standing
//  forms sit ON the terrain line and the line runs on beneath their flanks.
//  A mark stands in the row above the same ground level. A terrain run that
//  is not the highest of its column is the same line when it lies within one
//  level of it (it draws nothing) and a separate lower line otherwise (it
//  spells its own columns by the same table from its own neighbours); a mass
//  under a higher mass draws nothing there.

// LOD", opened at full size: one base line, single marks y 人 个 standing on
// it, tufts and commas under it, dotted texture; no outlined cone, no stacked
// arc rows. The dotted texture is the tone producer's, tone-field.ts):
//  MARK  a standing form that is taller than wide: one compiled mark by its
//        height (grammar[9] + 67) standing on the base line at the form's
//        own centre pen, over one | stem unit per further row it stands (a
//        trunk is a straight upright, D5): the mark spans the form's own
//        visible rows, at least two. A form reaches the mark only under
//        FAR_STRIP_BELOW_ROWS rows (the far tier is under three rows; a
//        spanning far form of three rows or more is a mass), so the mark never
//        draws a form in fewer rows than it occupies, and a far tree is 个
//        over its stem, never a hollow cone.
//  LOBE  a wider mass: a LOW BOUNDED RUN from the skill's 15.5 idioms. Cap
//        row: ／ ⌒ ... ⌒ ヽ (the puff lobe ⌒ヽ); under a side that falls
//        three rows or more, ONE flank unit on the next row; on the last row
//        of a side that falls two rows or more, the base turn ヽ_ (left) or
//        _ノ (right). The rows between are left to the tone producer: the
//        far band never holds an outlined cone. A mass that stands lower than
//        another in its columns keeps its turns and base only, never a second
//        ⌒ row under the first.
//  TUFT  under a flat unit of the far terrain line, every fourth column,
//        where the ground one row below the line is the same terrain and is
//        not level: ,, (15.5 base flicker).
//         reading: rough or sloping ground at the foot of the horizon line
//         (level ground there holds the tone producer's dot rows instead);
//         organization: one short group a row under the base line at a fixed
//         column interval; owner: the terrain silhouette run of the column.
//         Removing it leaves sloping far ground blank under the line.
// Nothing is tried, scored or fitted: every unit is one table read addressed
// by two level comparisons, every pen a column or a compiled offset.
// STABILITY: a mark's pen follows its form's own centre and the base line
// under it does not move with it. Profile units are whole lattice cells: a
// level changes when a column's top crosses half a row and a peak moves by
// one column when its apex crosses a column edge. No state is kept between
// frames, so neither has hysteresis.
const SKY_AT=${SKY_AT}u;const SKY_COLUMNS=${SKY_COLUMNS}u;
struct SkyCell {top:f32,apex:f32,at:u32,lean:f32}
struct Sky {on:bool,level:i32,flag:u32,id:u32}
struct FarForm {narrow:bool,centre:f32,height:f32,base:f32,foot:u32}
fn skyColumns()->i32{return i32(min((p.extent.x+15u)/16u,SKY_COLUMNS));}
fn skyLevel(y:f32)->i32{return i32(floor(y/17.0+.5));}
// The run's visible page box (low.xy, high.xy); low.x > high.x when nothing of it is visible.
fn skySpan(run:Run)->vec4f{
 var low=vec2f(3.402823e38);var high=vec2f(-3.402823e38);
 for(var i=0u;i+1u<run.range.y;i++){
  let a=points[run.range.x+i];if(!pointVisible(a)){continue;}let b=points[run.range.x+i+1u];
  low=min(low,min(a.position.xy,b.position.xy));high=max(high,max(a.position.xy,b.position.xy));
 }
 return vec4f(low,high);
}
// What one lattice column holds of the run: its highest visible point, where
// it is, and the segment carrying it (NONE: the run is not in the column).
fn skyCell(run:Run,x0:f32)->SkyCell{
 var cell=SkyCell(3.402823e38,x0+8.0,NONE,0.0);
 for(var i=0u;i+1u<run.range.y;i++){
  let a=points[run.range.x+i];if(!pointVisible(a)){continue;}let b=points[run.range.x+i+1u];
  let lo=max(min(a.position.x,b.position.x),x0);let hi=min(max(a.position.x,b.position.x),x0+16.0);
  if(hi<lo||lo>=x0+16.0){continue;}
  let dx=b.position.x-a.position.x;var ya=min(a.position.y,b.position.y);var yb=ya;
  if(abs(dx)>.0001){ya=mix(a.position.y,b.position.y,(lo-a.position.x)/dx);yb=mix(a.position.y,b.position.y,(hi-a.position.x)/dx);}
  if(cell.at==NONE||min(ya,yb)<cell.top){cell.top=min(ya,yb);cell.apex=select(hi,lo,ya<=yb);cell.at=i;}
 }
 return cell;
}
// The column owner's tag in the skyline key. Not the run's slot: slots are
// handed out by this frame's atomic order, so on a level tie the winner of a
// column changed between frames of a still view (crawl, 2026-11-11). The tag
// is the run's form (high 10 bits) and its own end points (low 9 bits).
fn skyMix(x:u32)->u32{var h=x*0x9e3779b1u;h^=h>>15u;h*=0x85ebca77u;h^=h>>13u;return h;}
fn skyID(run:Run)->u32{
 let a=bitcast<vec2u>(finiteRunStart(run).position.xy);let b=bitcast<vec2u>(finiteRunEnd(run).position.xy);
 let form=skyMix(run.identity.y^skyMix(run.identity.z))&0x3ffu;
 let own=skyMix(a.x^skyMix(a.y^skyMix(b.x^skyMix(b.y^skyMix(run.identity.w^run.range.y)))))&0x1ffu;
 return (form<<9u)|own;
}
fn skyKey(top:f32,flag:u32,id:u32)->u32{
 return ((4095u-u32(clamp(i32(round(top))+1024,0,4094)))<<20u)|(flag<<19u)|(id&0x7ffffu);
}
fn skyRead(table:u32,c:i32)->Sky{
 if(c<0||c>=skyColumns()){return Sky(false,0,0u,0u);}
 let key=atomicLoad(&counts[SKY_AT+table*SKY_COLUMNS+u32(c)]);
 if(key==0u){return Sky(false,0,0u,0u);}
 return Sky(true,skyLevel(f32(i32(4095u-(key>>20u))-1024)),(key>>19u)&1u,key&0x7ffffu);
}
// 0: not a far strip contributor. 1: far terrain silhouette or far box top
// of two rows or less. 2: taller far terrain silhouette or box (occupies,
// spells itself). 3: far standing field-class form (path-led or region-led
// source) or far round, any size band.
// THE ONE MERGE (the hierarchical LOD of game engines: neighbours too small
// to be drawn singly become one joint silhouette). A standing form joins the
// merged strip only when it stands under FAR_MERGE_BELOW page px, where it
// cannot hold a drawing of its own; from there up a far form is spelled from
// its own outline by its own construction, never replaced by a strip mark

// call-out 2026-11-10).
const FAR_MERGE_BELOW=25.5;
fn farStands(run:Run)->f32{
 var top=f32(run.paper.z&0xffffu)-16384.0;var base=f32(run.paper.z>>16u)-16384.0;
 let extent=formExtent(run.identity.yz);
 if(extent.found){top=extent.top/p.field.y*p.rect.w+p.rect.y;base=extent.bottom/p.field.y*p.rect.w+p.rect.y;}
 else if(run.paper.z==0u){return 1.0e6;}
 return base-top;
}
fn farUnderGlyph(run:Run)->bool{return farStands(run)<FAR_MERGE_BELOW;}
fn farKind(run:Run,away:u32,span:vec4f)->u32{
 if(away==0u||grammar[9]==0u||span.x>span.z||(run.range.z&0x10000u)!=0u){return 0u;}
 // The early source owner certifies the complete form below ONE native
 // cell, exterior only. Its native profile is class-neutral, never a round
 // substitution for a house/rail. All larger sources keep the paths below.
 if((run.range.z&0x80000000u)!=0u){return 3u;}
 // 中景: the OUTLINE of a middle standing form is the same profile on the same
 // base line; its interior lines stay its own (detail returns from the inside
 // as the form grows). Terrain lines and boxes join the strip only when far.
 // A middle form keeps lines of its own only where it stands on the rows to
 // hold them (five): under that an open, curved or interior-lined form is the
 // same profile as a solid one (a three-row tree spelled by its own runs was

 if(away==1u&&(terrainRun(run)||(run.range.z&0x100000u)!=0u||(run.range.z&0x20000u)!=0u)){return 0u;}
 // A MIDDLE FORM KEEPS ITS OWN LINES (line repair step 2; TIER_SELECTION
 // row 1: outline, occluding contours, principal creases, joins). A middle
 // form under five rows was its column profile on the strip, so a bridge, a
 // wall or a boulder at middle distance lost most of its edges (bridge-turn
 // owner 4: 25 of 31 lines; roofs wall 300: 28 of 41; boulder 272: 20 of 24).
 // Only a form too small to hold one unit (FAR_MERGE_BELOW) joins the strip.
 if(away==1u&&!farUnderGlyph(run)){return 0u;}
 if(terrainRun(run)){if(wholeBoundaryFlags(run)!=0u){return 0u;}return select(2u,1u,span.w-span.y<=34.0);}
 // A box keeps its own closed construction at every size (the smallest is one
 // box glyph, D9) and only occupies its columns of the strip.
 if((run.range.z&0x100000u)!=0u){return 2u;}
 // A far standing form of a unit or more is spelled from its own outline by
 // its own construction (TIER_SELECTION row 2, D9); the strip takes the

 // strip mark; only forms too small to hold a drawing merge).
 if((run.range.z&0x18000000u)==0x8000000u&&run.paper.z!=0u){return select(0u,3u,farUnderGlyph(run));}
 if((run.range.z&0x20000u)!=0u){return select(0u,3u,farUnderGlyph(run));}
 return 0u;
}
// The FORM of a far standing run, the same for every run of it: its visible
// page rows from the form table (the rows its tier was read from; the run
// pass's relation rows run.paper.z when the owner has no slot) and its width
// across its mid height, read on the owner field on a 4 px page grid six
// lattice cells to each side of the run's own centre (three cells did not
// reach across a far tree from a run on its edge, and that run spelled a

// view 06). foot: the run's lowest visible segment (the source span of a mark).
fn farForm(run:Run,span:vec4f)->FarForm{
 var top=f32(run.paper.z&0xffffu)-16384.0;var base=f32(run.paper.z>>16u)-16384.0;
 let extent=formExtent(run.identity.yz);
 if(extent.found){top=extent.top/p.field.y*p.rect.w+p.rect.y;base=extent.bottom/p.field.y*p.rect.w+p.rect.y;}
 let height=base-top;
 var foot=NONE;var lowest=-3.402823e38;
 for(var i=0u;i+1u<run.range.y;i++){
  let a=points[run.range.x+i];if(!pointVisible(a)){continue;}let b=points[run.range.x+i+1u];
  let y=max(a.position.y,b.position.y);if(foot==NONE||y>lowest){foot=i;lowest=y;}
 }
 let y=(top+base)*.5;let origin=4.0*round((span.x+span.z)*.125);var left=25;var right=-25;
 for(var k=-24;k<=24;k++){if(tierSample(run,fieldPoint(vec2f(origin+4.0*f32(k),y))).on){left=min(left,k);right=max(right,k);}}
 let width=4.0*f32(right-left)+4.0;
 // ONE MARK: the form is taller than wide, its whole width was read, it is
 // no wider than two lattice cells and stands under a far band's rows (a
 // spanning far form of three rows or more is a mass in its own rows).
 let narrow=left<=right&&left>-24&&right<24&&height>width&&width<=f32(${FAR_MARK_WIDTH})&&height<f32(${FAR_STRIP_BELOW_ROWS*17});
 return FarForm(narrow,origin+2.0*f32(left+right),height,base,foot);
}
// THE FORM PROFILE (the column data of a far region in a game's distant-terrain
// LOD: one height per column for the whole form, not whichever of its line
// runs survived selection and visibility). A far standing form's top in one
// lattice column is read on the owner field: three probes across the column,
// each walked down the form's own page rows to the first pixel the form owns.
// Every run of the form reads the same cells, so the strip holds the form's
// whole silhouette. A form without a table slot keeps the run's own points.
fn formOwns(run:Run,q:vec2f)->bool{
 let f=vec2i(floor(fieldPoint(q)));
 if(any(f<vec2i(0))||any(f>=vec2i(p.field.xy))){return false;}
 let owner=textureLoad(ownerField,f,0).xy;
 return (owner.x&0xdfffffffu)==run.identity.y&&owner.y==run.identity.z&&textureLoad(depthField,f,0).x>0.0;
}
fn base17()->f32{return 1.0e6;}
// THE FORM'S OWN EDGE on one side at one page height: from x, outward while
// the form owns the pixel, or inward to its first owned pixel (2 px steps,
// bounded). The side of a profile is set on this edge row by row.
fn formEdge(run:Run,x:f32,y:f32,side:u32)->f32{
 let out=select(-2.0,2.0,side==1u);
 if(formOwns(run,vec2f(x,y))){var e=x;for(var k=1;k<=10;k++){let q=x+out*f32(k);if(!formOwns(run,vec2f(q,y))){break;}e=q;}return e;}
 for(var k=1;k<=8;k++){let q=x-out*f32(k);if(formOwns(run,vec2f(q,y))){return q;}}
 return x;
}
// The far base row the forms of one group share. Table 2 holds each column's
// lowest standing-form base. A group is the occupied columns reached from c
// across gaps of two columns or less (neighbours merge: the distant-terrain
// LOD of games draws a region's far forms as one silhouette on one line). A
// form whose own base is within a row of the group's stands on the group's.
fn skyFoot(c:i32)->i32{
 if(c<0||c>=skyColumns()){return -1;}
 let key=atomicLoad(&counts[SKY_AT+2u*SKY_COLUMNS+u32(c)]);if(key==0u){return -1;}
 return i32(key)-1025;
}
fn groupFoot(c:i32,foot:i32)->i32{
 var low=foot;
 for(var side=-1;side<=1;side+=2){
  var gap=0;
  for(var k=1;k<=24&&gap<=2;k++){let f=skyFoot(c+side*k);if(f<0){gap++;continue;}gap=0;low=max(low,f);}
 }
 return select(foot,low,low-foot<=1);
}
fn formCell(run:Run,extent:FormExtent,c:i32)->SkyCell{
 var cell=SkyCell(3.402823e38,f32(c)*16.0+8.0,NONE,0.0);var tops=array<f32,3>(base17(),base17(),base17());
 var at=NONE;for(var i=0u;i+1u<run.range.y;i++){if(pointVisible(points[run.range.x+i])){at=i;break;}}
 if(at==NONE){return cell;}
 let scale=p.rect.w/p.field.y;let top=extent.top*scale+p.rect.y;let base=extent.bottom*scale+p.rect.y;
 for(var k=0;k<i32(p.farLoops.x);k++){
  let x=f32(c)*16.0+2.5+5.5*f32(k);
  for(var j=0;j<i32(p.farLoops.y);j++){
   let y=top+.5+f32(j)*2.5;if(y>=base){break;}
   if(formOwns(run,vec2f(x,y))){tops[k]=y-top;if(y<cell.top){cell.top=y;cell.apex=x;}break;}
  }
 }
 if(cell.top<3.0e38){cell.at=at;cell.lean=clamp(tops[0]-tops[2],-17.0,17.0);}
 return cell;
}
// A MASS OR A BRANCHING FORM. A profile is the outline of a form that fills
// its own page box (a rock, a bush, a crown, a hut). A form that owns under
// half of a 6 x 6 grid of its box is open (a bare tree, a fence, a frame): its
// column tops are a block that is not its shape, so it keeps its own lines.
fn formSolid(run:Run)->bool{
 let extent=formExtent(run.identity.yz);if(!extent.found){return false;}
 let scale=p.rect.zw/p.field.xy;var owned=0u;
 for(var j=0;j<i32(p.farLoops.z);j++){for(var i=0;i<i32(p.farLoops.z);i++){
  let f=vec2f(mix(extent.box.z,extent.box.w,(f32(i)+.5)/6.0),mix(extent.box.x,extent.box.y,(f32(j)+.5)/6.0));
  if(formOwns(run,f*scale+p.rect.xy)){owned++;}
 }}
 return owned>=18u;
}
// A CURVED OUTLINE keeps its own spelling. A run whose points leave its own
// chord by more than a quarter row is a chained curve (a sphere, a dome, a
// crown lobe): the column profile would cut it into a polygon. A straight
// facet edge (a rock, a slab) is what the profile replaces.
fn runBent(run:Run)->bool{
 let a=finiteRunStart(run).position.xy;let b=finiteRunEnd(run).position.xy;
 let d=b-a;let length=max(length(d),.0001);
 for(var i=1u;i+1u<run.range.y;i++){let point=points[run.range.x+i];if(pointWholeEdgeDescriptor(point)){continue;}let q=point.position.xy-a;if(abs(q.x*d.y-q.y*d.x)/length>4.25){return true;}}
 return false;
}
// The edge (pair i) of a hard chain lies on a CURVE when the next real turn
// along its own chain, on either side, is gentle: a faceted arch, vault or
// dome turns a few tens of degrees at each vertex. Pieces of one straight
// edge (visibility cuts, collinear splits, under 4 degrees) are walked over;
// a corner of 30 degrees or more (a post meeting a lintel, a wall meeting a
// roof) is a construction corner, not a curve. Whole-chain bending (runBent)
// cannot tell a curve from a post-and-lintel chain.
fn chainTurn(run:Run,i:u32,step:i32)->f32{
 let aPoint=points[run.range.x+i];let bPoint=points[run.range.x+i+1u];
 if(pointWholeEdgeDescriptor(aPoint)||pointWholeEdgeDescriptor(bPoint)){return 0.0;}
 let a=aPoint.position.xy;let b=bPoint.position.xy;let d=normalize(b-a+vec2f(1e-6,0.0));
 var join=select(a,b,step>0);var k=i32(i);
 for(var guard=0;guard<4;guard++){
  k+=step*2;if(k<0||u32(k)+1u>=run.range.y){return 0.0;}
  let cPoint=points[run.range.x+u32(k)];let ePoint=points[run.range.x+u32(k)+1u];
  if(pointWholeEdgeDescriptor(cPoint)||pointWholeEdgeDescriptor(ePoint)){return 0.0;}
  let c=cPoint.position.xy;let e=ePoint.position.xy;
  let near=select(e,c,step>0);let far=select(c,e,step>0);
  if(distance(near,join)>1.5){return 0.0;}
  let span=far-near;if(length(span)<1.0){join=far;continue;}
  let turn=acos(clamp(abs(dot(d,normalize(span))),0.0,1.0));
  if(turn<.0698){join=far;continue;}
  return turn;
 }
 return 0.0;
}
fn chainCurves(run:Run,i:u32)->bool{
 let before=chainTurn(run,i,-1);let after=chainTurn(run,i,1);
 return (before>0.0&&before<.5236)||(after>0.0&&after<.5236);
}
fn farCell(run:Run,kind:u32,c:i32)->SkyCell{
 if((run.range.z&0x80000000u)!=0u){return skyCell(run,f32(c)*16.0);}
 if(kind==3u){let extent=formExtent(run.identity.yz);if(extent.found){return formCell(run,extent,c);}}
 return skyCell(run,f32(c)*16.0);
}
fn farColumns(run:Run,kind:u32,span:vec4f)->vec2i{
 var x=span.xz;
 if(kind==3u){if((run.range.z&0x80000000u)==0u){let extent=formExtent(run.identity.yz);if(extent.found){x=extent.box.zw/p.field.x*p.rect.z+vec2f(p.rect.x,p.rect.x-1.0);}}}
 return vec2i(max(i32(floor(x.x/16.0)),0),min(i32(floor(x.y/16.0)),skyColumns()-1));
}
fn farMarkKey(run:Run,form:FarForm)->u32{return (u32(clamp(form.height,0.0,4095.0))<<20u)|(run.identity.z&0xfffffu);}
@compute @workgroup_size(64) fn skylineReduce(@builtin(global_invocation_id) id:vec3u){
 let runID=scheduledRun(id);if(runID==0xffffffffu||runID>=p.extent.z){return;}
 let run=runs[runID];if(run.range.y<2u||(run.range.z&0x10000u)!=0u){return;}
 let away=heldTier(runID,run);
 // This ordered pass already precedes all construction dispatches. Prepare
 // one complete whole-part verdict here; no member yields speculatively.
 prepareWholeSourcePart(runID,run,away);

 if(away==0u){limbWrite(run);}
 if(rippleRun(run)&&tierSelects(run,away)){rippleWrite(run);}
 if(away==0u){return;}
 let span=skySpan(run);let kind=farKind(run,away,span);if(kind==0u){return;}
 if(!tierSelects(run,away)){return;}
 let table=select(0u,1u,kind==3u);
 let columns=farColumns(run,kind,span);let c0=columns.x;let c1=columns.y;
 var footKey=0u;if(kind==3u){footKey=u32(clamp(skyLevel(farForm(run,span).base),-1024,3000)+1025);}
 for(var c=c0;c<=c1;c++){
  let cell=farCell(run,kind,c);if(cell.at==NONE){continue;}
  atomicMax(&counts[SKY_AT+table*SKY_COLUMNS+u32(c)],skyKey(cell.top,u32(kind==2u),skyID(run)));
  if(kind==3u){atomicMax(&counts[SKY_AT+2u*SKY_COLUMNS+u32(c)],footKey);}
 }
}
fn skyWord(id:u32,run:Run,ordinal:u32,at:u32,pen:i32,row:i32,word:u32,priority:u32){
 let a=points[run.range.x+at];let b=points[run.range.x+at+1u];
 sourceWord(id,ordinal,a.position,pen,row,word,words[word*WS],words[word*WS+1u],priority,a,b);
}
// The ground a standing form meets beside column c: the far terrain level
// there when it is within one row of the form's own foot level, else the foot.
fn skyGround(c:i32,foot:i32)->i32{
 let terrain=skyRead(0u,c);
 if(terrain.on&&abs(terrain.level-foot)<=1){return terrain.level;}
 return foot;
}
// One column: the cap on its level row by class and lower sides. A peak
// keeps its cell against a neighbour's flat unit and is set on its apex
// inside the cell. Under the cap, by class:
//  terrain: each lower side's flank unit, one per row, shifted by its lean.
//  mass (THE FAR UNITS, LOBE): one flank unit on the next row of a side that
//   falls three rows or more, and the compiled base turn (grammar[9] + 163)
//   on the last row of a side that falls two rows or more; no other row.
//   lower: the mass stands under a higher one in this column; its flat cap
//   would be a second arc row under the first and is not drawn.

// slanted rise, a cap, a slanted fall, closed on the base line). Cap: ￣ where
// the form's top is level across the column, ／ or ＼ where the top itself
// leans across the column toward a lower side, ∧ where the form is under one
// cell wide at its top. Sides: one unit per row from the cap down to the lower
// neighbour's level, each read from the form's own edge in that row (the
// compiled side units), so a steep rock keeps an upright side, a leaning one a
// slant, and no unit stands outside the silhouette.
fn profileSide(id:u32,run:Run,c:i32,cell:SkyCell,row:i32,side:u32,j:i32,beside:bool,priority:u32){
 let t=grammar[9];let x0=f32(c)*16.0+select(0.0,15.0,side==1u);
 let xa=formEdge(run,x0,f32(row)*17.0+3.0,side);let xb=formEdge(run,x0,f32(row)*17.0+14.0,side);
 let dx=xb-xa;let steep=f32(grammar[t+176u]);let slant=f32(grammar[t+177u]);
 var k=0u;if(dx<-slant){k=2u;}else if(dx<-steep){k=1u;}else if(dx>steep){k=3u;}
 let word=grammar[t+168u+k*2u];if(word==0u){return;}
 var pen=i32(round((xa+xb)*.5))-bitcast<i32>(grammar[t+169u+k*2u]);
 if(beside){if(side==0u){pen=min(pen,c*16-i32(words[word*WS+1u]));}else{pen=max(pen,c*16+16);}}
 skyWord(id,run,(u32(c)<<12u)|(u32(j+1)<<1u)|side,cell.at,pen,row,word,priority);
}
// THE SHADE BAND (D6: one constant pattern per region, bounded by lines).
// The form's lit face is its brightest owned sample; a row cell of the form
// under its cap whose own light is a tone band below that face is shade and
// holds the compiled unit, set inside the cell so the outline keeps the edge.
fn formBright(run:Run)->f32{
 let extent=formExtent(run.identity.yz);if(!extent.found){return 0.0;}
 var bright=0.0;
 for(var j=0;j<i32(p.farLoops.z);j++){for(var i=0;i<i32(p.farLoops.z);i++){
  let f=vec2f(mix(extent.box.z,extent.box.w,(f32(i)+.5)/6.0),mix(extent.box.x,extent.box.y,(f32(j)+.5)/6.0));
  let s=tierSample(run,f);if(s.on){bright=max(bright,s.luma);}
 }}
 return bright;
}
fn shadeCell(id:u32,run:Run,c:i32,row:i32,at:u32,bright:f32,priority:u32){
 let t=grammar[9];let word=grammar[t+178u];if(word==0u||bright<=.0001){return;}
 let advance=i32(grammar[t+179u]);
 // The whole unit must lie on the form in its row: both ends and the middle.
 let y=f32(row)*17.0+8.5;let x0=f32(c*16+8-advance/2);
 var dark=0u;
 for(var k=0;k<3;k++){
  let s=tierSample(run,fieldPoint(vec2f(x0+f32(advance)*f32(k)*.5,y)));
  if(!s.on){return;}
  if(s.luma<TONE_BAND_STEP*bright){dark++;}
 }
 if(dark<2u){return;}
 skyWord(id,run,(u32(c)<<12u)|0x400u|u32(row&0x3ff),at,i32(x0),row,word,priority>>1u);
}
fn profileUnits(id:u32,run:Run,c:i32,cell:SkyCell,level:i32,left:i32,right:i32,priority:u32,lower:bool){
 let rows=i32(rowCount());
 let drop=vec2i(clamp(left-level,0,rows),clamp(right-level,0,rows));
 var shape=0u;
 if(drop.x>0&&drop.y>0){
  if(cell.lean>6.0){shape=1u;}else if(cell.lean<-6.0){shape=2u;}
  else{let y=cell.top+4.0;if(formEdge(run,cell.apex,y,1u)-formEdge(run,cell.apex,y,0u)<12.0){shape=3u;}}
 }else if(drop.x>0&&cell.lean>6.0){shape=1u;}
 else if(drop.y>0&&cell.lean<-6.0){shape=2u;}
 if(lower&&drop.x==0&&drop.y==0){return;}
 let record=grammar[9]+99u+(4u+shape)*8u;let cap=grammar[record];if(cap==0u){return;}
 var pen=c*16;var weight=priority;
 if(shape==3u){
  pen=clamp(i32(round(cell.apex))-bitcast<i32>(grammar[record+1u]),c*16,c*16+max(16-i32(words[cap*WS+1u]),0));
  weight=0x80000000u|min(priority,0x7fffffffu);
 }
 skyWord(id,run,u32(c)<<12u,cell.at,pen,level,cap,weight);
 for(var side=0u;side<2u;side++){
  // The cap row's own side is drawn beside the cap where the cap does not
  // carry it (a level cap, or the far side of a leaning one).
  let covered=shape==3u||shape==side+1u;
  for(var j=select(0,1,covered);j<drop[side];j++){profileSide(id,run,c,cell,level+j,side,j,j==0,priority);}
 }
}
fn skyUnits(id:u32,run:Run,cls:u32,c:i32,cell:SkyCell,level:i32,left:i32,right:i32,priority:u32,lower:bool){
 if(cls==1u){profileUnits(id,run,c,cell,level,left,right,priority,lower);return;}
 let rows=i32(rowCount());
 let drop=vec2i(clamp(left-level,0,rows),clamp(right-level,0,rows));
 let shape=u32(drop.x>0)|(u32(drop.y>0)<<1u);
 if(lower&&cls==1u&&shape==0u){return;}
 let record=grammar[9]+99u+shape*8u;let cap=grammar[record];if(cap==0u){return;}
 var pen=c*16;var weight=priority;
 if(shape==3u){
  pen=clamp(i32(round(cell.apex))-bitcast<i32>(grammar[record+1u]),c*16,c*16+max(16-i32(words[cap*WS+1u]),0));
  weight=0x80000000u|min(priority,0x7fffffffu);
 }
 skyWord(id,run,u32(c)<<12u,cell.at,pen,level,cap,weight);
 for(var side=0u;side<2u;side++){
  let flank=grammar[record+2u+side*3u];if(flank==0u){continue;}
  let offset=bitcast<i32>(grammar[record+3u+side*3u]);let lean=bitcast<i32>(grammar[record+4u+side*3u]);
  for(var j=1;j<drop[side];j++){
   skyWord(id,run,(u32(c)<<12u)|(u32(j)<<1u)|side,cell.at,pen+offset+lean*(j-1),level+j,flank,priority);
  }
 }
}
// Returns false only for a far terrain run that holds the horizon in none of
// its columns: it is a line in front of the horizon, and featurecurves spells
// it whole through the terrain line table (one unit plus a per-row shift),
// never as column-lattice fragments under the strip.
fn farSkyline(id:u32,run:Run,kind:u32,span:vec4f,priority:u32)->bool{
 let table=grammar[9];var own=0;var bright=0.0;
 if(kind==3u){
  own=skyLevel(farForm(run,span).base);bright=formBright(run);
 }
 let cls=select(0u,1u,kind==3u);
 let columns=farColumns(run,kind,span);let c0=columns.x;let c1=columns.y;
 // THE STRIP IS THE HORIZON ONLY (D3 one merged skyline strip). A far terrain
 // line spells strip units only in the columns where it is the horizon (the
 // highest far outline of the column) or within one level of it. A line that
 // is the horizon in none of its columns lies in front of it (a nearer far
 // ridge): it is not cut into column cells but spelled whole by the line table.
 if(cls==0u&&terrainRun(run)){
  var held=false;
  for(var c=c0;c<=c1&&!held;c++){
   let cell=skyCell(run,f32(c)*16.0);if(cell.at==NONE){continue;}
   let top=skyRead(0u,c);
   held=!top.on||(top.flag==0u&&top.id==skyID(run))||skyLevel(cell.top)-top.level<=1;
  }
  if(!held){return false;}
 }
 for(var c=c0;c<=c1;c++){
  let cell=farCell(run,kind,c);if(cell.at==NONE){continue;}
  var foot=own;if(cls==1u){foot=groupFoot(c,own);}
  let top=skyRead(cls,c);var level=skyLevel(cell.top);var left=level;var right=level;
  // A mass keeps its cap run on its TRUE top (a wide far mass, a cliff or a
  // stand, is a hill outline: the markup's medium-far hills are simple
  // outline strokes). A trial that held every lobe within three rows of its

  // was withdrawn.
  if(top.on&&top.flag==0u&&top.id==skyID(run)){
   level=top.level;
   let l=skyRead(cls,c-1);let r=skyRead(cls,c+1);
   if(cls==0u){
    // A line that ends, ends flat.
    left=select(level,l.level,l.on);right=select(level,r.level,r.on);
   }else{
    left=skyGround(c-1,foot);right=skyGround(c+1,foot);
    if(l.on){left=min(left,l.level);}
    if(r.on){right=min(right,r.level);}
   }
  }else{
   // Not the highest of its column. Within one level of the highest it is
   // the same line; a contour of the SAME form under its own top is not
   // drawn; anything else is a separate lower line (a nearer far ridge, a
   // far form standing in front of a higher one) and spells its own columns.
   if(top.on&&level-top.level<=1){continue;}
   // A terrain line under the horizon in this column is in front of it: the
   // strip does not spell it here (no stepped lattice fragments on the ground).
   if(cls==0u&&terrainRun(run)){continue;}
   if(top.on&&cls==1u&&(top.id>>9u)==(skyID(run)>>9u)){continue;}
   // A lower MASS ends on its ground (its turn and base are what it keeps);
   // a lower terrain line that ends, ends flat.
   let l=farCell(run,kind,c-1);let r=farCell(run,kind,c+1);
   if(cls==1u){left=skyGround(c-1,foot);right=skyGround(c+1,foot);}
   if(l.at!=NONE){left=skyLevel(l.top);}
   if(r.at!=NONE){right=skyLevel(r.top);}
   skyUnits(id,run,cls,c,cell,level,left,right,priority,true);
   if(cls==1u&&foot>level){skyWord(id,run,(u32(c)<<12u)|0x800u,cell.at,c*16,foot,grammar[table+99u],priority);}
   continue;
  }
  skyUnits(id,run,cls,c,cell,level,left,right,priority,false);
  if(cls==1u){for(var row=level+1;row<foot;row++){shadeCell(id,run,c,row,cell.at,bright,priority);}}
  // THE SHARED BASE LINE. A far standing form stands on one line with its
  // neighbours: under every column it tops, the flat unit of its ground level.
  // The line runs on through a gap of two columns or less to the next form.
  if(cls==1u&&foot>level){
   skyWord(id,run,(u32(c)<<12u)|0x800u,cell.at,c*16,foot,grammar[table+99u],priority);
   if(skyFoot(c+1)<0){
    var next=0;for(var g=2;g<=3;g++){if(skyFoot(c+g)>=0){next=g;break;}}
    for(var g=1;g<next;g++){skyWord(id,run,(u32(c)<<12u)|0x800u|u32(g),cell.at,(c+g)*16,foot,grammar[table+99u],priority);}
   }
  }
  // TUFT (THE FAR UNITS): under a flat unit of the far terrain line, every
  // fourth column, where the ground a row below the line is this terrain
  // and is not level. One field read; the word and its pen are compiled. A
  // far box's top line (table 0) stands over a wall, not rough ground.
  if(cls==0u&&terrainRun(run)&&left==level&&right==level&&(c&3)==1){
   let below=tierSample(run,fieldPoint(vec2f(f32(c)*16.0+8.0,f32(level+1)*17.0+8.5)));
   if(below.on&&below.n.y<.9){skyWord(id,run,(u32(c)<<12u)|1u,cell.at,c*16+5,level+1,grammar[table+167u],priority);}
  }
 }
 return true;
}
// JUNCTION PLAYBACK. Where runs meet at one source vertex the join is ONE
// compiled glyph (compiled-role-grammar.ts compileJunctions, grammar[10]: the
// unit measured once on the native font whose ink best carries every leg out
// to its edge with the least ink off the legs), not run ends that collide or
// stop against each other. The run builder recorded per run end that the
// junction exists, which run spells it and the sixteenths of the two or three
// legs (a corner, a peak, a side join, a T, a Y, the lower corner of a step),
// and per run that passes THROUGH a T the vertex's point and its three legs.
// The record is read by those legs and carries the word, its advance, the ink
// column of its vertex and its row offset. The owner emits the word once as
// an anchor at the vertex. Every run at the vertex then holds back its own
// units centred on the glyph in the glyph's row and the vertex's row, so each
// ends at (or passes into) the glyph; a flank shorter than one unit leaves the
// glyph alone. Only line classes drawn at every distance tier take part, so
// the runs are selected together. Nothing is fitted or tried in a frame.
const JUNCTION_HOLDS=5u;
var<private> junctionHold:array<vec4i,JUNCTION_HOLDS>;
fn junctionRecord(legs:u32,three:bool,natural:bool)->u32{
 if(three){return grammar[10]+(${JUNCTION_THREE}u+select(0u,4096u,natural)+(legs&15u)*256u+((legs>>4u)&15u)*16u+((legs>>8u)&15u))*4u;}
 return grammar[10]+(${JUNCTION_TWO}u+select(0u,256u,natural)+(legs&15u)*16u+((legs>>4u)&15u))*4u;
}
// The hold of one junction glyph at a page point (node), or none.
fn junctionAt(at:u32,node:vec4f)->vec4i{
 let word=grammar[at];if(word==0u){return vec4i(0);}
 let advance=i32(grammar[at+1u]);let vertex=i32(grammar[at+2u]);let centre=i32(round(node.x));
 let row=i32(floor((node.y-f32(bitcast<i32>(grammar[at+3u]))*.5)/17.0));let reach=2*max(vertex,advance-vertex);
 return vec4i(2*centre-reach,2*centre+reach,row,i32(floor(node.y/17.0)));
}
fn hardJunction(runID:u32,run:Run,rank:u32){
 let joins=run.part.z;let line=min((run.range.z>>8u)&255u,4u);
 // Contacts (line class 2) keep their ports in paper.x/y; others carry T words.
 var tees=vec3u(run.part.w,0u,0u);if(line!=2u){tees.y=run.paper.x;tees.z=run.paper.y;}
 let runStart=finiteRunStart(run);let runEnd=finiteRunEnd(run);
 if((joins==0u&&all(tees==vec3u(0u)))||grammar[10]==0u||((${[0,1,2,3,4].reduce((mask,column)=>TIER_SELECTION.every(tier=>tier[column]===0)?mask|1<<column:mask,0)}u>>line)&1u)==0u){return;}
 var hold=array<vec4i,JUNCTION_HOLDS>(vec4i(0),vec4i(0),vec4i(0),vec4i(0),vec4i(0));
 for(var side=0u;side<2u;side++){
  let bits=(joins>>(side*16u))&0xffffu;if((bits&1u)==0u){continue;}
  let node=select(runStart.position,runEnd.position,side==1u);
  let three=(bits&0x8000u)!=0u;
  let at=junctionRecord(((bits>>2u)&15u)|(((bits>>6u)&15u)<<4u)|(((bits>>11u)&15u)<<8u),three,(bits&0x400u)!=0u);
  hold[side]=junctionAt(at,node);let word=grammar[at];if(word==0u){continue;}
  if((bits&2u)!=0u){emit(runID,0x7ff00000u|side,node.w,i32(round(node.x))-i32(grammar[at+2u]),hold[side].z,word,words[word*WS],grammar[at+1u],0x80000000u|min(rank,0x7fffffffu),0u,node.z);}
 }
 // A T on this run's own line: the ending run spells the glyph; this run
 // only holds back its units in the glyph's own row (a lid one row above a
 // 丁 stays whole and the T reads ＿ over 丁).
 for(var k=0u;k<3u;k++){
  let tee=tees[k];if((tee&0x80000000u)==0u||(tee&0xffffu)>=run.range.y){continue;}
  let teePoint=points[run.range.x+(tee&0xffffu)];if(pointWholeEdgeDescriptor(teePoint)){continue;}
  var held=junctionAt(junctionRecord((tee>>16u)&0xfffu,true,(tee&0x10000000u)!=0u),teePoint.position);
  held.w=held.z;hold[2u+k]=held;
 }
 junctionHold=hold;
}
// SOFT RUN PLAYBACK. A soft run (the silhouette or crease of a curved or
// triangle-soup form) is drawn one native row at a time. Its own points are
// cut where they cross a row; consecutive pieces inside one row are a row
// portion with a path-order entry and exit. A portion that crosses most of
// its row within one and a half unit widths takes the compiled whole-row unit
// of ITS direction, set on the path at mid-row: the unit repeats down the
// rows and the source supplies the shift (practitioner run rule), so a curve
// changes unit where its direction changes class and nowhere else. Every
// other portion is shallow, a crest or a stub: it advances by whole words
// and reads, from the same direction column, the record of the path's height
// in the row, so a hill top is spelled by its own heights. The former soft
// path read ONE record for the whole run at the accident of its first
// point's phase and chord. No word is ranked, tried or fitted in a frame.
fn chordClass(d:vec2f)->u32{return u32(round(fract(atan2(d.y,d.x)/6.28318530718+1.0)*64.0))&63u;}
// A column whose records are all paired brackets has no line unit; the
// upright column of the same role and size stands in for it.
fn softColumn(table:u32,role:u32,size:u32,dir:u32)->u32{
 let column=table+(((role*3u+size)*64u+dir)*17u)*12u;
 if(grammar[column+11u]!=0u){return column;}
 return table+(((role*3u+size)*64u+select(48u,16u,dir<32u))*17u)*12u;
}
// The source-part table is stored in the reserved prefix of the far-strip
// bank (grammar[9]).  A zero/old version is an explicit conservative fallback
// for baked banks compiled before whole-part playback existed.
fn softPartRecord(role:u32,size:u32,context:u32)->u32{
 if(role>=4u||size>=3u||context>=${SOFT_PART_CONTEXTS}u){return NONE;}
 let base=grammar[9]+${SOFT_PART_TABLE_AT}u;
 if(base+3u>=arrayLength(&grammar)||grammar[base]!=${SOFT_PART_TABLE_VERSION}u||grammar[base+1u]!=${SOFT_PART_TABLE_RECORD_WORDS}u||grammar[base+2u]!=${SOFT_PART_TABLE_RECORDS}u){return NONE;}
 let at=base+4u+(role*3u*${SOFT_PART_CONTEXTS}u+size*${SOFT_PART_CONTEXTS}u+context)*${SOFT_PART_TABLE_RECORD_WORDS}u;
 return select(NONE,at,at+7u<arrayLength(&grammar)&&grammar[at+1u]!=0u&&grammar[at+6u]==(context>>6u));
}
fn markSoftPartPaper(record:u32,origin:i32,row:i32,partRow:i32,growth:vec4u){
 if(record==NONE||partRow<0){return;}
 let first=grammar[record+2u];let count=grammar[record+3u];
 for(var i=0u;i<count;i++){
  let at=grammar[6]+(first+i)*4u;
  if(bitcast<i32>(grammar[at+1u])!=partRow){continue;}
  for(var repeat=0u;repeat<=growth.w;repeat++){
    let start=origin+i32(repeat*growth.z)+bitcast<i32>(grammar[at]);let end=start+i32(grammar[at+2u]);
    for(var x=max(0,start);x<min(i32(p.extent.x),end);x++){
     atomicOr(&counts[paperBase()+u32(row)*paperStride()+u32(x)/32u],1u<<(u32(x)%32u));
    }
  }
 }
}
fn sourcePartDirection(delta:vec2f)->u32{return u32(round(fract(atan2(delta.y,delta.x)/6.28318530718+1.0)*8.0))&7u;}
fn sourcePartEndpointDirections(source:Run)->vec2u{
 var incoming=NONE;var outgoing=NONE;
 for(var i=0u;i+1u<source.range.y;i+=2u){
  let a=points[source.range.x+i];let b=points[source.range.x+i+1u];
  if(pointWholeEdgeDescriptor(a)||pointWholeEdgeDescriptor(b)){continue;}
  if(length(b.position.xy-a.position.xy)<.00001){continue;}
  incoming=sourcePartDirection(a.position.xy-b.position.xy);break;
 }
 // Read only the end pairs, rather than walking the complete curved run on
 // every lookup. Descriptor and degenerate pairs are the only skipped ones.
 for(var offset=0u;offset+1u<source.range.y;offset+=2u){
  let i=source.range.y-2u-offset;let a=points[source.range.x+i];let b=points[source.range.x+i+1u];
  if(pointWholeEdgeDescriptor(a)||pointWholeEdgeDescriptor(b)){continue;}
  if(length(b.position.xy-a.position.xy)<.00001){continue;}
  outgoing=sourcePartDirection(b.position.xy-a.position.xy);break;
 }
 return vec2u(incoming,outgoing);
}
// The source graph owns the context family; the projected direction is read
// from its source-owned endpoint pair for this pose. It is not derived from
// a visibility fragment or a row-selected glyph.
fn sourcePartContextKey(run:Run)->u32{
 let contextClass=(runProgramContext(run)>>${SOURCE_PART_CONTEXT_CLASS_SHIFT}u)&${SOURCE_PART_CONTEXT_CLASS_MASK}u;if(contextClass==0u){return 0u;}
 let leader=sourcePartLeader(run);var source=run;if(leader!=NONE){source=runs[leader];}
 let directions=sourcePartEndpointDirections(source);if(directions.x==NONE){return 0u;}
 return directions.x|(directions.y<<3u)|(contextClass<<${SOURCE_PART_CONTEXT_CLASS_SHIFT}u);
}
// A context-valid source part is selected once from the packet's immutable
// context family and the posed source-owned tangent pair. The source part
// aggregate supplies one complete contributor span across all relation runs.
fn wholePartContextReady(run:Run,role:u32,size:u32)->u32{
 let prepared=sourcePartPrepared(run);if(prepared!=NONE){return atomicLoad(&counts[prepared+13u]);}
 let feature=runProgramFeature(run);let context=sourcePartContextKey(run);
 if(feature==0u||context==0u||run.correspondence.x==NONE||run.correspondence.y==NONE||run.correspondence.z==NONE){return NONE;}
 return softPartRecord(role,size,context);
}
fn wholePartLayout(run:Run,record:u32)->vec4i{
 if(record==NONE){return vec4i(0,0,0,-1);}
 let span=sourcePartAggregate(run);if(span.z<=span.x||span.w<span.y){return vec4i(0,0,0,-1);}
 let rows=i32(floor((span.w-span.y)/17.0)+1.0);let partRows=i32(grammar[record+5u]);
 // A related retained entry may be shorter than the posed source span: its
 // enduring first/last rows remain anchors while the middle row construction
 // is inserted by programmedWholePart. A taller entry cannot fit without
 // dropping rows, so the run falls to the next smaller declared band (w=-2);
 // only validity rejections stay on the incumbent.
 if(rows<=0||partRows<2){return vec4i(0,0,0,-1);}
 if(rows<partRows){return vec4i(0,0,0,-2);}
 let width=i32(grammar[record+4u]&0xffffu);if(width<=0){return vec4i(0,0,0,-1);}
 let spanWidth=i32(round(span.z-span.x));let origin=i32(round(span.x));
 if(spanWidth<width){return vec4i(0,0,0,-2);}
 if(origin<0||origin+spanWidth>i32(p.extent.x)){return vec4i(0,0,0,-1);}
 // Preflight the complete entry before any row is published. A bad atom
 // rejects the whole program and leaves D1's incumbent responsible for the
 // complete source span; no partial stamp plus legacy tail is possible.
 let first=grammar[record];let count=grammar[record+1u];
 for(var i=0u;i<count;i++){
  let at=grammar[1]+(first+i)*4u;let atomRow=bitcast<i32>(grammar[at+2u]);let advance=i32(grammar[at+3u]);
  if(atomRow<0||atomRow>=partRows||advance<=0){return vec4i(0,0,0,-1);}
  let x=origin+bitcast<i32>(grammar[at+1u]);if(x<0||x+advance>i32(p.extent.x)){return vec4i(0,0,0,-1);}
 }
 return vec4i(origin,i32(floor(span.y/17.0)),spanWidth,rows);
}

// invocation plays back one run's rows; the whole-part layout of (run, record)
// and its all-row preflight verdict are computed once and reused by every row
// (the leader's row loop and the soft per-row path), instead of re-running the
// complete preflight per row (rows^2). Keyed by (runID, record): a per-row
// role change selects another record and misses the cache.
var<private> wholeKey:vec2u=vec2u(0xffffffffu,0xffffffffu);
var<private> wholeCached:vec4i;
var<private> wholeVerdict:u32;// 0 unchecked, 1 every row fits, 2 rejected
fn wholePartLayoutFor(runID:u32,run:Run,record:u32)->vec4i{
 if(wholeKey.x==runID&&wholeKey.y==record){return wholeCached;}
 let prepared=sourcePartPrepared(run);
 if(prepared!=NONE&&atomicLoad(&counts[prepared+13u])==record){
  wholeKey=vec2u(runID,record);wholeCached=vec4i(bitcast<i32>(atomicLoad(&counts[prepared+14u])),bitcast<i32>(atomicLoad(&counts[prepared+15u])),bitcast<i32>(atomicLoad(&counts[prepared+16u])),bitcast<i32>(atomicLoad(&counts[prepared+17u])));wholeVerdict=1u;return wholeCached;
 }
 wholeKey=vec2u(runID,record);wholeCached=wholePartLayout(run,record);wholeVerdict=0u;return wholeCached;
}
// A row's repeat unit is the native interval BETWEEN its two end anchors,
// including its authored paper. The end glyph itself is not part of that
// interval. Preserve native x coordinates relative to the one part origin.
// An unsupported extension stays on the incumbent; it never stretches a glyph.
fn wholePartRowGrowth(record:u32,nativeRow:i32,extra:u32)->vec4u{
 let first=grammar[record];let count=grammar[record+1u];var firstX=NONE;var lastX=0u;var firstAdvance=0u;
 for(var atom=0u;atom<count;atom++){
  let at=grammar[1]+(first+atom)*4u;if(bitcast<i32>(grammar[at+2u])!=nativeRow){continue;}
  let x=grammar[at+1u];if(x<firstX){firstX=x;firstAdvance=grammar[at+3u];}lastX=max(lastX,x);
 }
 if(extra==0u){return vec4u(firstX,lastX,0u,0u);}
 if(firstX==NONE||lastX<=firstX+firstAdvance){return vec4u(firstX,lastX,0u,NONE);}
 let unitWidth=lastX-(firstX+firstAdvance);let inserts=min(8u,extra/unitWidth);
 if(inserts*unitWidth!=extra){return vec4u(firstX,lastX,unitWidth,NONE);}
 return vec4u(firstX,lastX,unitWidth,inserts);
}
fn wholePartAtomX(x:u32,growth:vec4u,repeat:u32)->u32{
 if(x==growth.x){return x;}
 if(x==growth.y){return x+growth.w*growth.z;}
 return x+repeat*growth.z;
}
// The complete-entry preflight (every source row, every inserted middle unit)
// of the cached layout, evaluated once.
fn wholePartRowsFit(record:u32,whole:vec4i)->bool{
 if(wholeVerdict!=0u){return wholeVerdict==1u;}
 wholeVerdict=2u;
 let first=grammar[record];let count=grammar[record+1u];let width=grammar[record+4u]&0xffffu;let partRows=i32(grammar[record+5u]);let spanWidth=u32(whole.z);let extra=spanWidth-width;
 for(var sourceRow=0;sourceRow<whole.w;sourceRow++){
  let nativeRow=wholePartNativeRow(sourceRow,whole.w,partRows);let growth=wholePartRowGrowth(record,nativeRow,extra);
  if(growth.w==NONE){return false;}
  for(var atom=0u;atom<count;atom++){
   let at=grammar[1]+(first+atom)*4u;if(bitcast<i32>(grammar[at+2u])!=nativeRow){continue;}
   let x=grammar[at+1u];
   // End anchors have one canonical member. Only intervening atoms repeat;
   // replaying anchors at the same pen adds duplicate proposals, not width.
   let repeats=select(growth.w+1u,1u,x==growth.x||x==growth.y);
   for(var repeat=0u;repeat<repeats;repeat++){
    let pageX=whole.x+i32(wholePartAtomX(x,growth,repeat));
    if(pageX<0||pageX+i32(grammar[at+3u])>i32(p.extent.x)){return false;}
   }
  }
 }
 wholeVerdict=1u;return true;
}

// band does not fit the posed source span, select the DECLARED RELATED
// VARIANT, the same role and context at the next smaller size band, down to
// band 0. The first band whose entry fits wins, deterministically. Validity
// rejections (no span, a degenerate entry, the origin outside the page, a bad
// atom) return no entry without falling through. Selected once per (runID,

// path select the same record for the same run.
var<private> wholeVariantKey:vec3u=vec3u(0xffffffffu,0xffffffffu,0xffffffffu);
var<private> wholeVariantCached:u32;
fn wholePartVariantRecord(runID:u32,run:Run,role:u32,size:u32)->u32{
 if(wholeVariantKey.x==runID&&wholeVariantKey.y==role&&wholeVariantKey.z==size){return wholeVariantCached;}
 wholeVariantKey=vec3u(runID,role,size);wholeVariantCached=NONE;
 if(size>=3u){return NONE;}
 let feature=runProgramFeature(run);let context=sourcePartContextKey(run);
 if(feature==0u||context==0u||run.correspondence.x==NONE||run.correspondence.y==NONE||run.correspondence.z==NONE){return NONE;}
 for(var step=0u;step<=size;step++){
  let record=softPartRecord(role,size-step,context);
  if(record==NONE){continue;}
  let whole=wholePartLayoutFor(runID,run,record);
  if(whole.w>=0){wholeVariantCached=record;return record;}
  if(whole.w!=-2){return NONE;}
 }
 return NONE;
}

// is the first anchor and the last source row is the last anchor; rows
// 1..partRows-2 map one-to-one, top-aligned; every further source row is an
// inserted whole unit repeating native row partRows-2 (row 1 when partRows==2
// has no middle). Growing the span moves only the last anchor down.
fn wholePartNativeRow(sourceRow:i32,sourceRows:i32,partRows:i32)->i32{
 if(sourceRows<=1||partRows<=1||sourceRow<=0){return 0;}
 if(sourceRow+1>=sourceRows){return partRows-1;}
 if(partRows==2){return 1;}
 if(sourceRow<=partRows-2){return sourceRow;}
 return partRows-2;
}
// Emit the selected native entry's complete row. The shared origin and whole
// unit insertion count come from the entire source span, while source phase is
// interpolated only between the source endpoints' scalar parameters.
fn programmedWholePart(runID:u32,run:Run,role:u32,size:u32,row:i32,portion:u32,seen:bool,sourcePair:u32,sourceQ:vec4f,priority:u32)->bool{
 var record=wholePartContextReady(run,role,size);if(record==NONE){return false;}
 var whole=wholePartLayoutFor(runID,run,record);
 if(whole.w==-2){record=wholePartVariantRecord(runID,run,role,size);if(record==NONE){return false;}whole=wholePartLayoutFor(runID,run,record);}
 if(whole.w<0){return false;}
 let partRow=row-whole.y;if(partRow<0||partRow>=whole.w){return false;}
 // A whole entry owns the row even when occlusion hides it. This prevents a
 // visibility cut from reselecting a smaller family or appending a legacy tail.
 if(!seen||portion>0u){return true;}
 let first=grammar[record];let count=grammar[record+1u];let width=grammar[record+4u]&0xffffu;let partRows=i32(grammar[record+5u]);let spanWidth=u32(whole.z);let extra=spanWidth-width;
 // Preflight every source row and every inserted middle unit before emitting

 // and last native atoms remain anchors; only the interior sequence repeats.
 if(!wholePartRowsFit(record,whole)){return false;}
 let startPhase=finiteRunStart(run).position.w;let endPhase=finiteRunEnd(run).position.w;
 let phase=mix(startPhase,endPhase,min(f32(partRow)/max(f32(partRows-1),1.0),1.0));var q=sourceQ;q.w=phase;
 let nativeRow=wholePartNativeRow(partRow,whole.w,partRows);let growth=wholePartRowGrowth(record,nativeRow,extra);
 for(var i=0u;i<count;i++){
  let at=grammar[1]+(first+i)*4u;if(bitcast<i32>(grammar[at+2u])!=nativeRow){continue;}
  let x=grammar[at+1u];
  let repeats=select(growth.w+1u,1u,x==growth.x||x==growth.y);
  for(var repeat=0u;repeat<repeats;repeat++){
   if(x!=growth.x&&x!=growth.y&&repeat>0u){q=sourceQ;q.w=phase;}
   let pageX=whole.x+i32(wholePartAtomX(x,growth,repeat));

   // the first anchor row is 0, the LAST anchor row is a fixed 0x3ff (it keeps
   // its name when whole units are inserted above it), middle and inserted rows
   // their top-aligned source row; low 16 bits = repeat ordinal x atoms + atom.
   // The visibility portion (always 0 here) is not identity.
   let memberRow=select(u32(partRow),0x3ffu,partRow==whole.w-1&&partRow>0);
   spanPartWord(runID,(memberRow<<16u)|(repeat*count+i),q,pageX,row,grammar[at],1u,grammar[at+3u],priority,sourcePair,1u);softWords++;
  }
 }
 markSoftPartPaper(record,whole.x,row,nativeRow,growth);return true;
}
// A word that repeats one glyph is cut to whole repeats of it (width by
// whole repeat units); any other word stays whole. Returns (scalars, advance).
fn repeatFit(word:u32,count:u32,advance:u32,room:f32)->vec2u{
 if(count<2u){return vec2u(count,advance);}
 let font=words[word*WS+2u];
 for(var scalar=1u;scalar<count;scalar++){if(words[word*WS+2u+scalar]!=font){return vec2u(count,advance);}}
 let width=grammar[grammar[2]+font*4u+1u];if(width==0u){return vec2u(count,advance);}
 let units=clamp(u32(round(room/f32(width))),1u,count);return vec2u(units,units*width);
}
// Words a soft run handed to placement, and whether a row of it was yielded
// to its pair: a visible soft run with neither is a loss (featurecurves).
var<private> softWords:u32;
var<private> softYield:bool;
// A silhouette or occluding run of a field-class (path-led, form 0) source
// that is neither hard, round, terrain nor an anchor.
fn limbPairable(run:Run)->bool{return (run.range.z&0x18070000u)==0x8000000u&&((run.range.z>>8u)&255u)<=1u;}
// The form beside an outline point, read along the row on the frame's own
// owner and depth fields. x: +1 the form lies right of the point, -1 left, 0
// unread or on both sides. y: the page x where that form ends, within
// LIMB_PAIR_STEPS steps of 4 px, against paper, a farther surface or its own
// farther part. A nearer foreign surface hiding the far edge is not an edge
// of this form: no pair.
fn limbScan(run:Run,at:vec2f)->vec2f{
 let right=tierSample(run,fieldPoint(at+vec2f(3.0,0.0)));let left=tierSample(run,fieldPoint(at-vec2f(3.0,0.0)));
 var side=0.0;var prior=right;
 if(right.on&&(!left.on||right.d>left.d*1.03)){side=1.0;}
 else if(left.on&&(!right.on||left.d>right.d*1.03)){side=-1.0;prior=left;}
 if(side==0.0){return vec2f(0.0);}
 for(var k=1u;k<=${LIMB_PAIR_STEPS}u;k++){
  let x=at.x+side*(3.0+4.0*f32(k));let f=fieldPoint(vec2f(x,at.y));let next=tierSample(run,f);
  if(next.on&&next.d>=prior.d*.97){prior=next;continue;}
  let pixel=vec2i(floor(f));
  if(all(pixel>=vec2i(0))&&all(pixel<vec2i(p.field.xy))){
   let owner=textureLoad(ownerField,pixel,0).xy;
   let same=(owner.x&0xdfffffffu)==run.identity.y&&owner.y==run.identity.z;
   if(!same&&(owner.x|owner.y)!=0u&&textureLoad(depthField,pixel,0).x>prior.d*1.03){return vec2f(0.0);}
  }
  // The edge lies inside this last 4 px step: read it to the page pixel.
  var edge=x-side*4.0;
  for(var s=1u;s<4u;s++){
   let inside=tierSample(run,fieldPoint(vec2f(edge+side,at.y)));
   if(!inside.on||inside.d<prior.d*.97){break;}
   edge+=side;
  }
  return vec2f(side,edge+side*.5);
 }
 return vec2f(0.0);
}
// One row of a limb is a PAIR when the same side reads at the row's top and
// bottom and the other edge crosses the row inside one and a half full-width
// units (24 px), the condition under which that edge's own run takes its
// one-unit-per-row branch. Both silhouettes evaluate this same rule on the
// same row, so one spells the pair and the other yields. The two edges must
// also run alongside each other: their horizontal travel between the two
// reads (12 px of height) differs by at most LIMB_PAIR_PARALLEL px, about 22
// degrees. A lobe or a body whose sides turn apart is not a limb and keeps
// its two outlines. x: side as limbScan, 0 no pair. y: the other edge's page
// x at mid-row.
const LIMB_PAIR_PARALLEL=5.0;

// compiled LIMB chain label (gpu-construction-runs limbVertex) is one of that
// tube's silhouettes. skylineReduce (earlier in the same frame) records, per
// (owner, chain, row), the least and greatest page x of the chain's VISIBLE
// silhouettes at the row's top and bottom sample heights; featurecurves pairs
// from that record. The pair is the chain's own two outlines, by source
// identity: no owner/depth field is read (limbScan stays only for sources
// without a compiled limb table, the incumbent).

// [row word, chain label, owner domain word, owner eid], then four ordered x
// words (least top, greatest top, least bottom, greatest bottom). The hash only
// picks where probing starts; a slot is this tuple's only when all four words
// equal it. Row word = (row + 1) | bit 30 (domain word is 0) | bit 31 (eid is 0),
// and the chain label always has bit 0 (PART_LIMB): both are never 0, so they
// are claimed first, and a zero owner word is never claimed (its tuple's row
// word already says so; any tuple reaching it has the same mask, so the same 0).
// The table is per frame (cleared with the grids) and keyed by this frame's
// run owners, so no lifetime outlives one frame's owner identities.
const LIMB_SLOTS=${LIMB_SLOTS}u;const LIMB_WORDS=${LIMB_WORDS}u;
fn limbBase()->u32{return gridBase()+2u*rowCount()*gridColumns()+2u*rippleRows()*rippleColumns();}
const SOURCE_PART_GROUP_WORDS=${SOURCE_PART_GROUP_WORDS}u;
fn sourcePartGroupSlots()->u32{
 var slots=1u;let needed=max(2u,p.extent.z*2u);
 loop{if(slots>=needed){break;}slots*=2u;}
 return slots;
}
fn sourcePartHeadBase()->u32{return limbBase()+LIMB_SLOTS*LIMB_WORDS;}
fn sourcePartRecordBase()->u32{return sourcePartHeadBase()+sourcePartGroupSlots();}
fn sourcePartOrder(value:f32)->u32{
 let bits=bitcast<u32>(value);return select(bits^0x80000000u,~bits,(bits&0x80000000u)!=0u);
}
fn sourcePartReal(value:u32)->f32{
 let bits=select(value^0x80000000u,~value,(value&0x80000000u)!=0u);return bitcast<f32>(bits);
}
// The hash only locates a bucket.  The four words are compared in full before
// a record is considered the same source part: domain, instance, actual host
// instance epoch and packet-local part program. No screen coordinate or
// visible fragment participates in this identity.
fn sourcePartKey(run:Run)->vec4u{
 // Use the existing exact lifetime column, also consumed by memoKey. The
 // legacy run hash contains topologyGroup and is NOT a source lifetime:
 // XOR-cancelling its feature still splits one part by unrelated edge groups.
 var lifetime=select(NONE,0u,(run.identity.y&0x80000000u)!=0u);
 if((run.identity.y&0x80000000u)==0u&&run.identity.z<p.playback.z){lifetime=atomicLoad(&counts[epochBase()+run.identity.z]);}
 return vec4u(run.identity.y,run.identity.z,lifetime,runProgramFeature(run));
}
fn sourcePartHash(key:vec4u)->u32{
 var h=(key.x*0x9e3779b9u)^(key.y*0x85ebca6bu)^(key.z*0xc2b2ae35u)^(key.w*0x27d4eb2fu);
 h=(h^(h>>16u))*0x7feb352du;return (h^(h>>15u))&(sourcePartGroupSlots()-1u);
}
fn sourcePartRecordAddress(runID:u32)->u32{return sourcePartRecordBase()+runID*SOURCE_PART_GROUP_WORDS;}
fn sourcePartKeyMatches(at:u32,key:vec4u)->bool{
 return atomicLoad(&counts[at])==key.x&&atomicLoad(&counts[at+1u])==key.y&&atomicLoad(&counts[at+2u])==key.z&&atomicLoad(&counts[at+3u])==key.w;
}
// Stage 1: each run owns its record.  The record is completely initialized
// before its run ID is exchanged into the bucket head; no other workgroup can
// observe a half-written key or bounds record.  Stage 2 is a separate dispatch
// and is the only place that reduces records, so there is no cross-workgroup
// spin or leader/bounds publication race.
fn sourcePartRecord(runID:u32,run:Run){
 let at=sourcePartRecordAddress(runID);
 for(var word=0u;word<SOURCE_PART_GROUP_WORDS;word++){atomicStore(&counts[at+word],0u);}
 if(run.range.y==0u||runProgramFeature(run)==0u||runProgramContext(run)==0u){return;}
 let key=sourcePartKey(run);
 atomicStore(&counts[at],key.x);atomicStore(&counts[at+1u],key.y);atomicStore(&counts[at+2u],key.z);atomicStore(&counts[at+3u],key.w);
 let bucket=sourcePartHeadBase()+sourcePartHash(key);
 let next=atomicExchange(&counts[bucket],runID+1u);atomicStore(&counts[at+4u],next);
}
@compute @workgroup_size(64) fn sourcePartInit(@builtin(global_invocation_id) id:vec3u){
 let runID=scheduledRun(id);if(runID==0xffffffffu||runID>=p.extent.z){return;}
 sourcePartRecord(runID,runs[runID]);
}
fn sourcePartRecordFor(run:Run)->u32{
 let feature=runProgramFeature(run);if(feature==0u||runProgramContext(run)==0u){return NONE;}
 let key=sourcePartKey(run);var link=atomicLoad(&counts[sourcePartHeadBase()+sourcePartHash(key)]);
 for(var step=0u;step<p.extent.z;step++){
  if(link==0u){break;}
  let at=sourcePartRecordAddress(link-1u);if(sourcePartKeyMatches(at,key)){return at;}
  link=atomicLoad(&counts[at+4u]);
 }
 return NONE;
}
const SOURCE_PART_FEATURE_ORDINAL_MASK=${SOURCE_PART_FEATURE_ORDINAL_MASK}u;
fn partAnchorRegionBase()->u32{return p.partAnchors.x;}
fn partAnchorBucketOf(ownerX:u32,ownerY:u32,part:u32,heads:u32)->u32{
 var h=(ownerX*0x9e3779b9u)^(ownerY*0x85ebca6bu)^(part*0xc2b2ae35u);h=(h^(h>>16u))*0x7feb352du;h=h^(h>>15u);
 return h&(heads-1u);
}
// The descriptor and table are copied from this job's camera before another
// camera can reuse SurfaceRelations. Missing metadata keeps member bounds.
fn sourcePartAnchorExtent(ownerX:u32,ownerY:u32,part:u32)->vec4f{
 let absent=vec4f(0.0,0.0,-1.0,-1.0);let base=partAnchorRegionBase();
 if(p.partAnchors.y<${PART_ANCHOR_HEADER_WORDS}u||base>arrayLength(&counts)||p.partAnchors.y>arrayLength(&counts)-base){return absent;}
 if(atomicLoad(&counts[base])==0u||part==0u){return absent;}
 let words=atomicLoad(&counts[base]);let heads=atomicLoad(&counts[base+1u]);let headBase=atomicLoad(&counts[base+2u]);let recordBase=atomicLoad(&counts[base+3u]);let stride=atomicLoad(&counts[base+4u]);
 if(words>p.partAnchors.y-${PART_ANCHOR_HEADER_WORDS}u||heads==0u||(heads&(heads-1u))!=0u||headBase>words||heads>words-headBase||recordBase>words||stride!=${PART_ANCHOR_TABLE_RECORD_WORDS}u){return absent;}
 let table=base+${PART_ANCHOR_HEADER_WORDS}u;let records=(words-recordBase)/stride;
 var link=atomicLoad(&counts[table+headBase+partAnchorBucketOf(ownerX,ownerY,part,heads)]);
 for(var step=0u;step<records;step++){
  let next=link&0x7fffffffu;if(next==0u||next>records){break;}
  let at=table+recordBase+(next-1u)*stride;
  if(atomicLoad(&counts[at])==ownerX&&atomicLoad(&counts[at+1u])==ownerY&&atomicLoad(&counts[at+2u])==part){
   let minX=bitcast<f32>(atomicLoad(&counts[at+4u]));let minY=bitcast<f32>(atomicLoad(&counts[at+5u]));let maxX=bitcast<f32>(atomicLoad(&counts[at+6u]));let maxY=bitcast<f32>(atomicLoad(&counts[at+7u]));
   if(!(minX<=maxX&&minY<=maxY)){return absent;}
   let rx=p.rect.x;let ry=p.rect.y;let rw=p.rect.z;let rh=p.rect.w;
   return vec4f((minX*0.5+0.5)*rw+rx,(-maxY*0.5+0.5)*rh+ry,(maxX*0.5+0.5)*rw+rx,(-minY*0.5+0.5)*rh+ry);
  }
  link=atomicLoad(&counts[at+3u]);
 }
 return absent;
}
@compute @workgroup_size(64) fn sourcePartReduce(@builtin(global_invocation_id) id:vec3u){
 let runID=scheduledRun(id);if(runID==0xffffffffu||runID>=p.extent.z){return;}
 let run=runs[runID];let at=sourcePartRecordAddress(runID);if(runProgramFeature(run)==0u||runProgramContext(run)==0u){return;}
 // The initialized bucket chain has one first exact-key representative.
 // Reduce there once, not once per member (quadratic for a dense part).
 // All consumers already resolve that same representative through this
 // lookup; the native leader is still selected by original source feature.
 if(sourcePartRecordFor(run)!=at){return;}
 let key=sourcePartKey(run);var link=atomicLoad(&counts[sourcePartHeadBase()+sourcePartHash(key)]);
 var leader=NONE;var leaderFeature=NONE;var minX=sourcePartOrder(3.402823e38);var minY=sourcePartOrder(3.402823e38);var maxX=sourcePartOrder(-3.402823e38);var maxY=sourcePartOrder(-3.402823e38);var members=0u;
 for(var step=0u;step<p.extent.z;step++){
  if(link==0u){break;}
  let candidateID=link-1u;let candidateAt=sourcePartRecordAddress(candidateID);
  if(sourcePartKeyMatches(candidateAt,key)){
   let candidate=runs[candidateID];let candidateFeature=candidate.correspondence.x;
   if(leader==NONE||candidateFeature<leaderFeature||(candidateFeature==leaderFeature&&candidateID<leader)){leader=candidateID;leaderFeature=candidateFeature;}
   let a=finiteRunStart(candidate).position;let b=finiteRunEnd(candidate).position;
   minX=min(minX,sourcePartOrder(min(a.x,b.x)));minY=min(minY,sourcePartOrder(min(a.y,b.y)));maxX=max(maxX,sourcePartOrder(max(a.x,b.x)));maxY=max(maxY,sourcePartOrder(max(a.y,b.y)));members++;
  }
  link=atomicLoad(&counts[candidateAt+4u]);
 }
 let declared=sourcePartAnchorExtent(run.identity.y,run.identity.z,runProgramFeature(run)&SOURCE_PART_FEATURE_ORDINAL_MASK);
 if(declared.x<=declared.z&&declared.y<=declared.w){minX=sourcePartOrder(declared.x);minY=sourcePartOrder(declared.y);maxX=sourcePartOrder(declared.z);maxY=sourcePartOrder(declared.w);}
 atomicStore(&counts[at+5u],leader);atomicStore(&counts[at+6u],minX);atomicStore(&counts[at+7u],minY);atomicStore(&counts[at+8u],maxX);atomicStore(&counts[at+9u],maxY);atomicStore(&counts[at+10u],members);atomicStore(&counts[at+11u],select(0u,1u,members!=0u));
}
fn sourcePartAggregate(run:Run)->vec4f{
 let at=sourcePartRecordFor(run);if(at==NONE||atomicLoad(&counts[at+11u])==0u){return vec4f(0.0);}
 return vec4f(sourcePartReal(atomicLoad(&counts[at+6u])),sourcePartReal(atomicLoad(&counts[at+7u])),sourcePartReal(atomicLoad(&counts[at+8u])),sourcePartReal(atomicLoad(&counts[at+9u])));
}
fn sourcePartLeader(run:Run)->u32{
 let at=sourcePartRecordFor(run);if(at==NONE||atomicLoad(&counts[at+11u])==0u){return NONE;}return atomicLoad(&counts[at+5u]);
}
// Words 12..17 of the existing job-owned group record carry the committed
// preparation state, selected native record and layout. Only the canonical
// leader writes them, during skylineReduce; all readers run in later passes.
fn sourcePartPrepared(run:Run)->u32{
 let leader=sourcePartLeader(run);if(leader==NONE){return NONE;}
 let at=sourcePartRecordAddress(leader);return select(NONE,at,atomicLoad(&counts[at+12u])==1u);
}
fn prepareWholeSourcePart(runID:u32,run:Run,tier:u32){
 if(runProgramFeature(run)==0u||runProgramContext(run)==0u||sourcePartLeader(run)!=runID){return;}
 // A leader must reach the soft whole-part consumer in featurecurves.
 // If another family owns it, every member keeps its incumbent instead.
 if(wholeBoundaryFlags(run)!=0u||rippleRun(run)||!tierSelects(run,tier)){return;}
 if(atomicLoad(&counts[memoMapBase()+runID])==NONE){return;}
 let first=finiteRunStart(run).position.xy;let last=finiteRunEnd(run).position.xy;let delta=last-first;
 if(length(delta)<.00001){return;}
 if((run.range.z&0x40000u)!=0u){
  var bent=false;
  for(var i=0u;i<run.range.y;i++){
   let point=points[run.range.x+i];if(pointWholeEdgeDescriptor(point)){continue;}
   let q=point.position.xy;bent=bent||abs((q.x-first.x)*delta.y-(q.y-first.y)*delta.x)>.25*max(1.0,length(delta));
  }
  if(!bent){return;}
 }
 if((run.range.y&1u)!=0u){return;}
 if(tier!=0u){let kind=farKind(run,tier,skySpan(run));if(kind==1u||kind==3u){return;}}
 let size=min(runSize(run),tierBand(tier));var record=wholePartContextReady(run,runRole(run),size);
 if(record==NONE){return;}
 var whole=wholePartLayoutFor(runID,run,record);
 if(whole.w==-2){record=wholePartVariantRecord(runID,run,runRole(run),size);if(record==NONE){return;}whole=wholePartLayoutFor(runID,run,record);}
 if(whole.w<0||!wholePartRowsFit(record,whole)){return;}
 let at=sourcePartRecordAddress(runID);
 atomicStore(&counts[at+13u],record);atomicStore(&counts[at+14u],bitcast<u32>(whole.x));atomicStore(&counts[at+15u],bitcast<u32>(whole.y));atomicStore(&counts[at+16u],bitcast<u32>(whole.z));atomicStore(&counts[at+17u],bitcast<u32>(whole.w));
 atomicStore(&counts[at+12u],1u);
}
fn limbTuple(run:Run,row:i32)->vec4u{
 let zeros=select(0u,0x40000000u,run.identity.y==0u)|select(0u,0x80000000u,run.identity.z==0u);
 return vec4u((u32(row)+1u)|zeros,run.paper.w,run.identity.y,run.identity.z);
}
fn limbAddress(t:vec4u)->u32{
 var h=(t.x*0x27d4eb2fu)^(t.y*0xc2b2ae35u)^(t.z*0x9e3779b9u)^(t.w*0x85ebca6bu);
 h=(h^(h>>16u))*0x7feb352du;return h^(h>>15u);
}
// True when word at holds v once this writer is done with it: an empty word
// is claimed (retrying only a spurious weak failure, as addPartVertex); a held
// word must already be v. A zero component is never written (see the mask).
fn limbClaim(at:u32,v:u32)->bool{
 if(v==0u){return true;}
 var held=atomicLoad(&counts[at]);
 loop{if(held!=0u){break;}let c=atomicCompareExchangeWeak(&counts[at],0u,v);held=c.old_value;if(c.exchanged){return true;}}
 return held==v;
}
// Writers claim the tuple word by word in a fixed order: a different tuple
// stops at its first mismatching word and probes on; a slot whose words mix
// two tuples matches neither and is never read. Readers (a later dispatch)
// compare all four words. Full (no matching slot within 32 probes): the writer
// records nothing and the reader finds no pair, the single-outline incumbent.
fn limbSlot(t:vec4u,claim:bool)->u32{
 let h=limbAddress(t);
 for(var probe=0u;probe<32u;probe++){
  let at=limbBase()+((h+probe*0x9e3779b9u)%LIMB_SLOTS)*LIMB_WORDS;
  if(!claim){
   let k=vec4u(atomicLoad(&counts[at]),atomicLoad(&counts[at+1u]),atomicLoad(&counts[at+2u]),atomicLoad(&counts[at+3u]));
   if(all(k==t)){return at;}if(k.x==0u){return NONE;}continue;
  }
  if(limbClaim(at,t.x)&&limbClaim(at+1u,t.y)&&limbClaim(at+2u,t.z)&&limbClaim(at+3u,t.w)){return at;}
 }
 return NONE;
}
fn limbOrdered(x:f32)->u32{return u32(clamp((x+16384.0)*16.0,0.0,4294967040.0))+1u;}
fn limbUnordered(v:u32)->f32{return f32(v-1u)/16.0-16384.0;}
fn limbWrite(run:Run){
 if((run.paper.w&1u)==0u||(run.range.z&0x10000u)!=0u||!limbPairable(run)){return;}
 for(var k=0u;k+1u<run.range.y;k+=2u){
  let a=points[run.range.x+k];if(!pointVisible(a)){continue;}let b=points[run.range.x+k+1u];
  let lo=min(a.position.y,b.position.y);let hi=max(a.position.y,b.position.y);let dy=b.position.y-a.position.y;
  if(abs(dy)<.00001){continue;}
  for(var row=i32(floor(lo/17.0));row<=i32(floor(hi/17.0));row++){
   if(row<0||row>=i32(rowCount())){continue;}
   for(var s=0u;s<2u;s++){
    let y=f32(row*17)+select(2.5,14.5,s==1u);if(y<lo||y>hi){continue;}
    let x=mix(a.position.x,b.position.x,(y-a.position.y)/dy);
    let at=limbSlot(limbTuple(run,row),true);if(at==NONE){continue;}
    let v=limbOrdered(x);
    atomicMax(&counts[at+4u+s*2u],0xffffffffu-v);atomicMax(&counts[at+5u+s*2u],v);
   }
  }
 }
}
// The limbPair contract from the compiled chain: x = +1 this run is the LEFT
// silhouette (it spells both flanks), -1 the RIGHT (it yields), 0 no pair;
// y = the other silhouette's page x at mid-row. top/bottom: this run's own x
// at the two sample heights. The same width and parallel tests as limbPair.
fn limbPairSource(run:Run,row:i32,top:f32,bottom:f32)->vec2f{
 let at=limbSlot(limbTuple(run,row),false);if(at==NONE){return vec2f(0.0);}
 let w=vec4u(atomicLoad(&counts[at+4u]),atomicLoad(&counts[at+5u]),atomicLoad(&counts[at+6u]),atomicLoad(&counts[at+7u]));
 if(any(w==vec4u(0u))){return vec2f(0.0);}
 let leastTop=limbUnordered(0xffffffffu-w.x);let mostTop=limbUnordered(w.y);
 let leastBottom=limbUnordered(0xffffffffu-w.z);let mostBottom=limbUnordered(w.w);
 var side=0.0;var otherTop=0.0;var otherBottom=0.0;
 if(abs(top-leastTop)<=1.5&&abs(bottom-leastBottom)<=1.5&&mostTop-leastTop>1.5){side=1.0;otherTop=mostTop;otherBottom=mostBottom;}
 else if(abs(top-mostTop)<=1.5&&abs(bottom-mostBottom)<=1.5&&mostTop-leastTop>1.5){side=-1.0;otherTop=leastTop;otherBottom=leastBottom;}
 if(side==0.0){return vec2f(0.0);}
 if(abs(otherBottom-otherTop)>24.0){return vec2f(0.0);}
 if(abs((otherBottom-otherTop)-(bottom-top))>LIMB_PAIR_PARALLEL){return vec2f(0.0);}
 return vec2f(side,(otherTop+otherBottom)*.5);
}
fn limbPair(run:Run,top:vec2f,bottom:vec2f)->vec2f{
 let a=limbScan(run,top);if(a.x==0.0){return vec2f(0.0);}
 let b=limbScan(run,bottom);if(b.x!=a.x||abs(b.y-a.y)>24.0){return vec2f(0.0);}
 if(abs((b.y-a.y)-(bottom.x-top.x))>LIMB_PAIR_PARALLEL){return vec2f(0.0);}
 return vec2f(a.x,(a.y+b.y)*.5);
}

// shallow pen walk is spelled only where, inside its own x-span and this row
// portion, the run's VISIBLE intervals (pair start flag) are not outweighed by its
// HIDDEN ones (D1(6): a glyph at a visibility cut keeps its place when most of its
// ink lies on the visible portion). A 2 px visible crumb of an 8 px terrain piece

// A fully visible run has no hidden interval and keeps every unit.
fn softUnitStands(run:Run,startPair:u32,startT:f32,endPair:u32,endT:f32,lo:f32,hi:f32)->bool{
 var shown=0.0;var hidden=0.0;
 for(var k=startPair;k<=endPair;k++){
  let pa=points[run.range.x+k*2u];let pb=points[run.range.x+k*2u+1u];
  let xa=mix(pa.position.x,pb.position.x,select(0.0,startT,k==startPair));let xb=mix(pa.position.x,pb.position.x,select(1.0,endT,k==endPair));
  let c=max(0.0,min(hi,max(xa,xb))-max(lo,min(xa,xb)));
  if(pointVisible(pa)){shown+=c;}else{hidden+=c;}
 }
 return shown>=hidden;
}
// A visibility cut is not a new source passage. The run packet retains every
// projected interval (visible and hidden) and carries the original source-edge
// address in Point.visible.w. Resolve the portion's source span from that
// address and the native row, never from the first interval that happens to be
// visible. A source edge may have several clipped pieces; the piece crossing
// the row centre is the stable source span for this row. If an edge has no
// address, retain the caller's pair and stay on the conservative generic path.
fn softSourcePair(run:Run,seed:u32,row:i32)->u32{
 let base=run.range.x;let edge=points[base+seed*2u].visible.w;
 if(edge==NONE){return seed;}
 let centre=f32(row)*17.0+8.5;var best=seed;var score=3.402823e38;
 for(var k=0u;k<run.range.y/2u;k++){
  let a=points[base+k*2u];let b=points[base+k*2u+1u];if(a.visible.w!=edge){continue;}
  let low=min(a.position.y,b.position.y);let high=max(a.position.y,b.position.y);
  let inside=centre>=low&&centre<=high;
  let candidate=select(abs((a.position.y+b.position.y)*.5-centre),0.0,inside)+f32(k)*.000001;
  if(candidate<score){score=candidate;best=k;}
  // At the first crossing the best score is <= k's tie-break. Every
  // later score is a nonnegative distance plus a no-smaller tie-break,
  // so none can win. Keep best: an earlier near miss may still beat k.
  if(inside){break;}
 }
 return best;
}

// pairs are its source edges' visible AND hidden finite intervals in source
// relation order, so the source point at a row centre is the same however
// visibility splits the edge. The row reads the first pair (source order) with
// a chord of at least 1 px whose source y-span contains the row centre, else
// the nearest such pair, else pair 0. Visibility is never read: moving the
// anchor to the first VISIBLE fragment would restore the rephasing failure.

// one run) and looked up per row, instead of scanning every pair for every row.
const ROW_PAIR_ROWS=64u;
var<private> rowPairRun:u32=0xffffffffu;
var<private> rowPairBase:i32;
var<private> rowPairTable:array<u32,64>;
fn rowPairQualifies(run:Run,k:u32)->bool{
 let a=points[run.range.x+k*2u].position;let b=points[run.range.x+k*2u+1u].position;return length(b.xy-a.xy)>=1.0;
}
fn prepareRowPairs(runID:u32,run:Run){
 rowPairRun=runID;let pairs=run.range.y/2u;let base=run.range.x;var lowest=3.402823e38;
 for(var r=0u;r<ROW_PAIR_ROWS;r++){rowPairTable[r]=0xffffffffu;}
 for(var k=0u;k<pairs;k++){if(!rowPairQualifies(run,k)){continue;}lowest=min(lowest,min(points[base+k*2u].position.y,points[base+k*2u+1u].position.y));}
 if(lowest>=3.0e38){rowPairBase=0;return;}
 rowPairBase=i32(ceil((lowest-8.5)/17.0));
 // Source order; the first qualifying pair containing a row centre claims it.
 for(var k=0u;k<pairs;k++){
  if(!rowPairQualifies(run,k)){continue;}
  let a=points[base+k*2u].position.y;let b=points[base+k*2u+1u].position.y;
  let first=i32(ceil((min(a,b)-8.5)/17.0));let last=i32(floor((max(a,b)-8.5)/17.0));
  for(var row=max(first,rowPairBase);row<=last;row++){
   let at=row-rowPairBase;if(at>=i32(ROW_PAIR_ROWS)){break;}
   if(rowPairTable[at]==0xffffffffu){rowPairTable[at]=k;}
  }
 }
}
// The 133 rule, for a row the table does not hold: nearest qualifying pair.
fn softProgramRowPairScan(run:Run,row:i32)->u32{
 let pairs=run.range.y/2u;if(pairs==0u){return 0u;}
 let base=run.range.x;let rowY=f32(row)*17.0+8.5;var best=0u;var bestDistance=3.402823e38;
 for(var k=0u;k<pairs;k++){
  let a=points[base+k*2u].position;let b=points[base+k*2u+1u].position;
  if(length(b.xy-a.xy)<1.0){continue;}
  let lo=min(a.y,b.y);let hi=max(a.y,b.y);
  if(rowY>=lo&&rowY<=hi){return k;}
  let d=min(abs(rowY-lo),abs(rowY-hi));if(d<bestDistance){bestDistance=d;best=k;}
 }
 return best;
}
fn softProgramRowPair(runID:u32,run:Run,row:i32)->u32{
 if(run.range.y/2u==0u){return 0u;}
 if(rowPairRun!=runID){prepareRowPairs(runID,run);}
 let at=row-rowPairBase;
 if(at>=0&&at<i32(ROW_PAIR_ROWS)&&rowPairTable[at]!=0xffffffffu){return rowPairTable[at];}
 return softProgramRowPairScan(run,row);
}

// inside the row's band. A vertical source segment, or a pen whose line point
// leaves the row, reads the row-centre point (softSourceAt) as before.
fn softSourceAtPen(run:Run,pair:u32,x:f32,row:i32)->vec4f{
 let base=run.range.x;let a=points[base+pair*2u];let b=points[base+pair*2u+1u];
 let dx=b.position.x-a.position.x;if(abs(dx)<=.0001){return softSourceAt(run,pair,row);}
 let q=interpolatePoint(a,b,clamp((x-a.position.x)/dx,0.0,1.0));
 let top=f32(row)*17.0;if(q.y<top||q.y>top+17.0){return softSourceAt(run,pair,row);}
 return q;
}
fn softSourceAt(run:Run,pair:u32,row:i32)->vec4f{
 let base=run.range.x;let a=points[base+pair*2u];let b=points[base+pair*2u+1u];
 let dy=b.position.y-a.position.y;let rowY=f32(row)*17.0+8.5;
 let t=select(.5,clamp((rowY-a.position.y)/dy,0.0,1.0),abs(dy)>.00001);
 return interpolatePoint(a,b,t);
}
fn softSourceChart(run:Run,pair:u32,row:i32)->vec2f{
 let base=run.range.x;let a=points[base+pair*2u];let b=points[base+pair*2u+1u];
 let dy=b.position.y-a.position.y;let rowY=f32(row)*17.0+8.5;
 let t=select(.5,clamp((rowY-a.position.y)/dy,0.0,1.0),abs(dy)>.00001);
 return sourceChartAt(a,b,t);
}

// lattice unit is named by the source-anchored position it is placed at: its
// row relative to the anchor point's row (14 bits) and its exact integer pen
// offset from the anchor along the row direction (15 bits), never by the
// visibility portion or a count from the row's first visible pen. At most
// 0x3fff7fff: below STAMP_ORDINAL and the circle/edge ordinal encodings'
// special values.
fn softMemberOrdinal(anchor:Point,row:i32,offset:i32)->u32{
 let anchorRow=i32(floor(anchor.position.y/17.0));
 return ((u32(clamp(row-anchorRow+8192,0,16383))&0x3fffu)<<16u)|(u32(clamp(offset+16384,0,32767))&0x7fffu);
}

// paired flanks k=1/2, a conifer whole k=3) at fixed offsets past any lattice
// pen offset (pages narrower than about 16000 px): never the visibility portion.
fn softRowOrdinal(anchor:Point,row:i32,k:u32)->u32{return softMemberOrdinal(anchor,row,16383-i32(k));}
// T11 COMPILED NEAREST-PHASE INDEX (crawl brief item F): the T9-selected phase
// for this column and phase from the compiled table (role-grammar-format.ts
// SOFT_PHASE_TABLE_*, literals pinned by main-soft-phase-table.test.ts), or
// NONE when the loaded bank predates the table; 0xff is a compiled absence
// of qualifying phases, not permission to scan again. table is grammar[0],
// grammar[4] or the terrain table; column is the softColumn result.
fn softPhaseWant(table:u32,column:u32,phase:u32)->u32{
 let base=grammar[9]+49338u;
 if(base+3u>=arrayLength(&grammar)||grammar[base]!=1u||grammar[base+1u]!=3264u||grammar[base+2u]!=3u){return NONE;}
 var sel=2u;if(table==grammar[0]){sel=0u;}else if(table==grammar[4]){sel=1u;}
 let byte=((sel*768u+(column-table)/204u)*17u+phase);
 return (grammar[base+4u+(byte>>2u)]>>((byte&3u)*8u))&0xffu;
}
// The D5 progression is compiled with the role-bank grammar.  An entry's low
// 30 bits are an offset in the selected soft table or hard-line table; bit 30
// selects the hard table and bit 31 reports the pre-swap record's upright
// classification.  Old banks stay on the incumbent runtime helpers below.
fn softD5TableReady()->bool{
 let base=grammar[9]+${SOFT_D5_TABLE_AT}u;
 return base+${SOFT_D5_TABLE_WORDS-1}u<arrayLength(&grammar)&&grammar[base]==${SOFT_D5_TABLE_VERSION}u&&grammar[base+1u]==${SOFT_D5_TABLE_ENTRY_WORDS}u&&grammar[base+2u]==${SOFT_D5_TABLE_TABLES}u&&grammar[base+3u]==${SOFT_D5_TABLE_COLUMNS}u&&grammar[base+4u]==${SOFT_D5_TABLE_PHASES}u&&grammar[base+5u]==${SOFT_D5_TABLE_VARIANTS}u;
}
fn softD5Want(table:u32,role:u32,size:u32,dir:u32,phase:u32,chord:vec2f,whole:bool)->u32{
 let base=grammar[9]+${SOFT_D5_TABLE_AT}u;var sel=2u;if(table==grammar[0]){sel=0u;}else if(table==grammar[4]){sel=1u;}
 let variant=select(0u,1u,(dir==16u||dir==48u)&&chord.x<0.0);let column=(role*3u+size)*64u+dir;let slot=select(0u,1u,whole);
 let at=base+${SOFT_D5_TABLE_HEADER_WORDS}u+((((variant*${SOFT_D5_TABLE_TABLES}u+sel)*${SOFT_D5_TABLE_COLUMNS}u+column)*${SOFT_D5_TABLE_PHASES}u+phase)*${SOFT_D5_TABLE_ENTRY_WORDS}u)+slot;
 return grammar[at];
}
fn softD5Upright(entry:u32)->bool{return (entry&${SOFT_D5_TABLE_UPRIGHT}u)!=0u;}
fn softD5Record(table:u32,entry:u32)->u32{
 let offset=entry&${SOFT_D5_TABLE_NONE|SOFT_D5_TABLE_HARD}u;
 if(offset==${SOFT_D5_TABLE_NONE}u){return NONE;}
 return select(table+offset,grammar[8]+(offset&${SOFT_D5_TABLE_NONE}u),(entry&${SOFT_D5_TABLE_HARD}u)!=0u);
}
fn softPortion(runID:u32,run:Run,table:u32,size:u32,row:i32,portion:u32,startPair:u32,startT:f32,endPair:u32,endT:f32,seen:bool,markPair:u32,markT:f32,priority:u32){
 if(row<0||row>=i32(rowCount())){return;}
 let base=run.range.x;let programFeature=runProgramFeature(run);var role=runRole(run);
 var sourcePair=0u;
 if(programFeature==0u){sourcePair=softSourcePair(run,startPair,row);}else{sourcePair=softProgramRowPair(runID,run,row);}
 let sourceQ=softSourceAt(run,sourcePair,row);
 let enter=interpolatePoint(points[base+startPair*2u],points[base+startPair*2u+1u],startT);
 let leave=interpolatePoint(points[base+endPair*2u],points[base+endPair*2u+1u],endT);
 // Glyph vocabulary and direction belong to the source span; enter and leave
 // remain projected interval bounds used only for placement geometry.
 let sourceA=points[base+sourcePair*2u];let sourceB=points[base+sourcePair*2u+1u];

 // identities are anchored at the RUN's own first source point, as the lattice
 // rule states. The row's source pair start (sourceA) changed whenever a row
 // crossed into the next source pair, shifting every pen of that row.
 let runAnchor=finiteRunStart(run);
 let chord=sourceB.position.xy-sourceA.position.xy;if(length(chord)<1.0){probeAdd(91u);return;}// PROBE
 collapseDir=gridDir(chord);
 // The PART of the form this row of the outline lies on, by the run pass's
 // own rule applied per row instead of once per chain: steep is a flank;
 // otherwise the top 15 per cent of the owner is its cap, below 78 per cent
 // its return, between them a shoulder. One silhouette chain runs through
 // several parts (a rock's outline is cap, shoulder, flank and return); one
 // role for the whole chain spelled all of them from one part's columns.
 // Outline classes only: an interior crease keeps its chain's role.
 // A terrain run has no cap or base by its place in the owner's box (a hill
 // is not a round): its rows keep the run pass's context role, and only a
 // row of an outline steeper than 45 degrees is a flank.
 let terrain=terrainRun(run);
 if(terrain&&programFeature==0u){if(((run.range.z>>8u)&255u)<=1u){role=select(1u,2u,abs(chord.y)>abs(chord.x));}}
 else if(programFeature==0u&&((run.range.z>>8u)&255u)<=1u&&(run.range.z&0x10000u)==0u&&(run.part.y&0x1fffffffu)!=4u&&run.paper.z!=0u){
  let low=f32(run.paper.z&0xffffu)-16384.0;let height=f32(run.paper.z>>16u)-16384.0-low;let middle=f32(row)*17.0+8.5;
  role=select(1u,2u,abs(chord.y)>abs(chord.x));
  if(role!=2u){if(middle-low<height*.15){role=0u;}else if(middle-low>height*.78){role=3u;}}
 }
 // Programmed source parts consume the selected retained native entry before
 // any per-row direction/phase logic. Unsupported or legacy banks return to
 // the incumbent D1 path below; they are never dropped or respelled here.
 if(programFeature!=0u&&sourcePartPrepared(run)!=NONE&&programmedWholePart(runID,run,role,size,row,portion,seen,sourcePair,sourceQ,priority)){return;}
 // ONE UNIT PER RUN (practitioner run rule): a terrain run's rows read the
 // run's own direction (runDirection, its first-to-last chord) unless the
 // row's direction turns more than three bins (about 17 degrees) away from
 // it, a genuine turn of the fold. A row-by-row direction class re-chose the
 // unit wherever a straight fold wobbled across a bin edge.
 var dir=select(chordClass(chord),runDirection,programFeature!=0u);
 // Only a portion that crosses its row steeply (the whole-row unit branch)
 // reads its segment's direction. A shallow portion is spelled along the row
 // by its own chord: the segment of a steep fold read there repeated the fold's
 // upright unit across the row (||||| pickets, lines-4 spot-frames).
 if(programFeature==0u&&abs(chord.y)>=9.0){let own=segmentDirection(startPair);let turn=(dir+64u-own)%64u;if(min(turn,64u-turn)<=3u){dir=own;}}
 let column=softColumn(table,role,size,dir);
 var unit=column+(max(grammar[column+11u],1u)-1u)*12u;
 let d5Ready=softD5TableReady();
 if(d5Ready){
  let d5=softD5Want(table,role,size,dir,0u,chord,true);
  let upright=softD5Upright(d5);
  if((programFeature!=0u&&runProgramD5(run))||(programFeature==0u&&curvedRun&&abs(row-lastUprightRow)==1&&upright)||(noUprightRun&&upright)){
   let chosen=softD5Record(table,d5);if(chosen!=NONE){unit=chosen;}
  }
 }else if(((programFeature!=0u&&runProgramD5(run))||(programFeature==0u&&d5Swap(grammar[unit],row)))||(noUprightRun&&uprightWord(grammar[unit]))){
  // Old banks retain the incumbent progression until the versioned table is
  // present; this branch is never reached for a current compiled bank.
  let lean=slashNeighbour(dir,chord);let other=softColumn(table,role,size,lean);let swapped=other+(max(grammar[other+11u],1u)-1u)*12u;
  // The table's own neighbouring column; when that is an upright too (a
  // steep fold of the terrain table, whose 70-85 degree columns are the
  // ずらし run of l), the clean-line built unit of that direction.
  let built=grammar[8]+(((role*3u+size)*64u+lean)*17u+8u)*12u;
  if(grammar[swapped]!=0u&&!uprightWord(grammar[swapped])){unit=swapped;}
  else if(grammar[built]!=0u&&!uprightWord(grammar[built])){unit=built;}
 }
 let unitAdvance=grammar[unit+2u];
 if(abs(chord.y)>=9.0&&abs(chord.x)<=1.5*f32(unitAdvance)&&grammar[unit]!=0u){
  if(!seen){probeAdd(92u);return;}// PROBE
  let a=grammar[unit+6u];let b=grammar[unit+7u];let ink=(f32(a&255u)+f32(b&255u))*.5-128.0;
  var middle=sourceQ.x;
  var q=sourceQ;
  if(terrain){
   // The source row point remains the correspondence owner even when that
   // interval is hidden. The shared visibility pass clips the whole word.
   middle=sourceQ.x;
  }
  // PAIRED FLANK COLUMN. On a row where a path-led limb's other silhouette
  // also crosses the row steeply, the two are one construction: the LEFT
  // silhouette spells both flanks with the one record of this row's lean
  // (grammar[11], pairedFlanks), each centred on its own edge, and the right
  // silhouette spells nothing on that row. The interior is left to tone. A
  // pair narrower than LIMB_PAIR_LEAST is one stroke: the left unit alone.

  // red line; beyond it a trunk is its plain outline, then the far strip).
  // lineTier is the FORM's selection tier, set once per run in featurecurves:
  // every limb of one tree is paired, or none is.
  if(limbPairable(run)&&lineTier==0u){
   let top=f32(row*17)+2.5;let bottom=f32(row*17)+14.5;
   let topAt=vec2f(enter.x+(top-enter.y)*chord.x/chord.y,top);let bottomAt=vec2f(enter.x+(bottom-enter.y)*chord.x/chord.y,bottom);

   // (limbPairSource); a source without the compiled table keeps limbPair.
   var pair=vec2f(0.0);
   if((run.paper.w&1u)!=0u&&(run.range.z&0x10000u)==0u){pair=limbPairSource(run,row,topAt.x,bottomAt.x);}else{pair=limbPair(run,topAt,bottomAt);}
   if(pair.x<0.0){softYield=true;return;}
   if(pair.x>0.0&&pair.y-middle>=f32(${LIMB_PAIR_LEAST})){
    let lean=chord.x*sign(chord.y);
    var leanClass=2u;
    if(lean<=f32(${PAIRED_FLANK_LEANS[0]})){leanClass=0u;}else if(lean<=f32(${PAIRED_FLANK_LEANS[1]})){leanClass=1u;}else if(lean>=f32(${PAIRED_FLANK_LEANS[3]})){leanClass=4u;}else if(lean>=f32(${PAIRED_FLANK_LEANS[2]})){leanClass=3u;}
    let record=grammar[11]+leanClass*4u;let word=grammar[record];let advance=grammar[record+1u];let centre=bitcast<i32>(grammar[record+2u]);
    emit(runID,softRowOrdinal(runAnchor,row,1u),q.w,i32(round(middle))-centre,row,word,1u,advance,priority,0u,q.z);
    emit(runID,softRowOrdinal(runAnchor,row,2u),q.w,i32(round(pair.y))-centre,row,word,1u,advance,priority,0u,q.z);
    softWords+=2u;return;
   }
  }
  // FOLIAGE SIDES: the resK-41 sequence belongs to the REST source chart,
  // not to a screen-row count or a renormalized visible chain. Occlusion,
  // extending a run and a camera-only row crossing do not restart it.
  // Uprights and sources without this chart keep the generic run above.
  if((run.range.z&0x20000000u)!=0u&&((run.range.z>>8u)&255u)<=1u&&abs(chord.x)>=.15*abs(chord.y)){
   let chart=softSourceChart(run,sourcePair,row);
   if(chart.y!=0.0){
    let rec=grammar[11]+${FOLIAGE_EDGES_AT}u+(select(4u,0u,chord.x*chord.y<0.0)+foliageChartBand(chart.x))*4u;
    let conifer=grammar[rec];let centre=bitcast<i32>(grammar[rec+2u]);
    if(conifer!=0u){spanWord(runID,softRowOrdinal(runAnchor,row,3u),q,i32(round(middle))-centre,row,conifer,words[conifer*WS],words[conifer*WS+1u],priority,sourcePair);softWords++;return;}
   }
  }
  // 抜き (implied continuation): a near-tier ずらし run (record word 9, its gap
  // period) leaves out one row in each period, counted from the run's own
  // first source point so the gaps ride the fold instead of the page. Only on
  // an outline (line classes 0 and 1, the outer part of the form): an
  // interior fold is a dense, front part and stays continuous (GOAL-EXECUTOR
  // practitioner rules: gaps only on outer, lighter parts).
  let gapPeriod=select(0u,grammar[unit+9u],role<=1u);
  if(gapPeriod!=0u){let anchorRow=i32(floor(finiteRunStart(run).position.y/17.0));if(u32(abs(row-anchorRow))%gapPeriod==gapPeriod-1u){softWords++;return;}}
  spanWord(runID,softRowOrdinal(runAnchor,row,0u),q,i32(round(middle-ink)),row,grammar[unit],grammar[unit+1u],unitAdvance,priority,sourcePair);
  if(uprightWord(grammar[unit])){lastUprightRow=row;}
  softWords++;return;
 }
 let way=select(-1.0,1.0,chord.x>=0.0);
 // WORD BOUNDARIES ARE ATTACHED TO THE LINE, NOT TO THE ROW CUT. The row
 // boundary crosses a shallow line at a point that slides far along it when
 // the line moves a pixel up or down, so pens counted from that crossing
 // swam along the surface (glyph crawl). The first pen is the lattice point
 // at or before the crossing on a lattice of the run's whole-row unit pitch
 // anchored at the run's own first source point (projected now, so it moves
 // with the form); words are set at their lattice pens, and the line point a
 // pen still short of the crossing reads is the crossing itself.
 let anchorX=runAnchor.position.x;
 let pitch=max(f32(unitAdvance),8.0);
 let lattice=anchorX+floor((enter.x-anchorX)*way/pitch)*pitch*way;
 // Start at the page edge when the portion enters from outside it.
 var pen=select(min(lattice,f32(p.extent.x)+64.0),max(lattice,-64.0),way>0.0);
 var cursor=startPair;var index=0u;var priorUpright=false;
 for(var guard=0u;guard<512u;guard++){
  let room=(leave.x-pen)*way;
  if((guard>0u&&room<=0.0)||(way>0.0&&pen>f32(p.extent.x))||(way<0.0&&pen<0.0)){break;}
  var found=false;var q=enter;
  let probe=select(pen,enter.x,(pen-enter.x)*way<0.0);
  for(var k=cursor;k<=endPair;k++){
   let pa=points[base+k*2u];let pb=points[base+k*2u+1u];
   let t0=select(0.0,startT,k==startPair);let t1=select(1.0,endT,k==endPair);
   let xa=mix(pa.position.x,pb.position.x,t0);let xb=mix(pa.position.x,pb.position.x,t1);
   if((probe-xa)*way>=-.5&&(xb-probe)*way>=-.5){
    let f=select(0.0,clamp((probe-xa)/(xb-xa),0.0,1.0),abs(xb-xa)>.0001);
   cursor=k;
   if(programFeature==0u){sourcePair=softSourcePair(run,cursor,row);}

   // this lattice pen reads the source line AT THE PEN, so its phase (the
   // line's height within the row) and source scalar follow the line across
   // the row; the row-centre point gave every unit of the row one phase.
   q=softSourceAtPen(run,sourcePair,probe,row);found=true;break;
   }
  }
  if(!found){if(guard==0u){probeAdd(94u);}break;}// PROBE
  let phase=u32(clamp(floor(q.y)-f32(row*17),0.0,16.0));
  // D1(3), a fixed order: a word wider than twice the room left takes the
  // same role, direction and height one size band smaller. The smallest
  // band's first word is always drawn.
  var band=size;var record=softColumn(table,role,band,dir)+phase*12u;
  var fit=repeatFit(grammar[record],grammar[record+1u],grammar[record+2u],max(room,1.0));
  // The source program owns the part's size family. Remaining visible room
  // may shorten the repeated unit, but it never selects a smaller grammar
  // band or a different source family at a visibility cut.
  // NO COMPILED WORD IS NOT A BLANK (D1: never drop a word while a legal
  // alternative exists; AGENTS finding 7). A direction column of the pinned
  // table holds a word only at some row phases, so a selected line whose
  // path stood at an empty phase drew nothing (loop 351 view 06: 452 selected
  // runs with a visible span and no word, the rock and slope edges the
  // underdrawing shows and the text did not). Fixed order: the column's own
  // whole-row unit; then the clean-line unit of the same direction and
  // phase, which every direction has. A curved or slanted run never takes
  // the upright (D5): its near-vertical directions read the neighbouring
  // slash column.
  if(grammar[record]==0u){
   let column=softColumn(table,role,band,dir);let whole=column+(max(grammar[column+11u],1u)-1u)*12u;

   // whole-unit / clean-line fallback (a different vocabulary at a phase
   // threshold), take the nearest non-empty phase record of the SAME direction
   // column in fixed order phase-1, phase+1, phase-2, phase+2, ... (lower first
   // on a tie). A neighbouring-row record (grammar[r+3u]!=0u) never qualifies.
   // T11 reads the compiled index first (softPhaseWant); the scan below is
   // the older-artifact fallback and selects identically.
   var near=NONE;
   let want=softPhaseWant(table,column,phase);
   if(want<17u){near=column+want*12u;}
   else if(want==NONE){
    for(var d=1u;d<17u;d++){
     if(phase>=d){let r=column+(phase-d)*12u;if(grammar[r]!=0u&&grammar[r+3u]==0u){near=r;break;}}
     if(phase+d<17u){let r=column+(phase+d)*12u;if(grammar[r]!=0u&&grammar[r+3u]==0u){near=r;break;}}
    }
   }
   if(near!=NONE){record=near;}
   else if(grammar[whole]!=0u&&grammar[whole+3u]==0u){record=whole;}
   else{
    var clean=dir;let lean=dir%32u;
    if(table!=grammar[0]&&lean>=14u&&lean<=18u){clean=dir-lean+select(19u,13u,lean<=16u);}
    record=grammar[8]+(((role*3u+band)*64u+clean)*17u+phase)*12u;
   }
   fit=repeatFit(grammar[record],grammar[record+1u],grammar[record+2u],max(room,1.0));
  }
  // D5 (d5Swap): the second consecutive upright row takes the slash column.
  // Current banks read the compiled progression; old banks preserve the
  // incumbent helper path until their ABI is refreshed.
  let d5Ready=softD5TableReady();
  if(d5Ready){
   let d5=softD5Want(table,role,band,dir,phase,chord,false);
   let upright=softD5Upright(d5);
   if((programFeature!=0u&&runProgramD5(run))||(programFeature==0u&&curvedRun&&abs(row-lastUprightRow)==1&&upright)||(noUprightRun&&upright)){
    let chosen=softD5Record(table,d5);if(chosen!=NONE){record=chosen;fit=repeatFit(grammar[record],grammar[record+1u],grammar[record+2u],max(room,1.0));}
   }
  }else if(((programFeature!=0u&&runProgramD5(run))||(programFeature==0u&&d5Swap(grammar[record],row)))||(noUprightRun&&uprightWord(grammar[record]))){
   let lean=slashNeighbour(dir,chord);let swapped=softColumn(table,role,band,lean)+phase*12u;
   let built=grammar[8]+(((role*3u+band)*64u+lean)*17u+phase)*12u;
   if(grammar[swapped]!=0u&&grammar[swapped+3u]==0u&&!uprightWord(grammar[swapped])){record=swapped;fit=repeatFit(grammar[record],grammar[record+1u],grammar[record+2u],max(room,1.0));}
   else if(grammar[built]!=0u&&!uprightWord(grammar[built])){record=built;fit=repeatFit(grammar[record],grammar[record+1u],grammar[record+2u],max(room,1.0));}
  }
  var step=8.0;
  // A record that belongs to the neighbouring row is that row's own portion.
  if(grammar[record]!=0u&&grammar[record+3u]==0u){
   if(index>0u&&room*2.0<f32(fit.y)){break;}
   // An upright unit is one mark per row: one directly after another upright
   // of the same row portion is skipped (no ||||| along a row); the row goes on.
   let upright=uprightWord(grammar[record]);
   if(!(upright&&priorUpright)&&softUnitStands(run,startPair,startT,endPair,endT,select(pen-f32(fit.y),pen,way>0.0),select(pen,pen+f32(fit.y),way>0.0))){spanWord(runID,softMemberOrdinal(runAnchor,row,i32(round((pen-anchorX)*way))),q,i32(round(pen))-select(i32(fit.y),0,way>0.0),row,grammar[record],fit.x,fit.y,priority,sourcePair);softWords++;if(upright){lastUprightRow=row;}}else{probeAdd(95u);}// PROBE
   priorUpright=upright;
   index++;step=f32(fit.y);
  }else{probeAdd(93u);}// PROBE
  pen+=way*step;
 }
}
var<private> runDirection:u32;
// ONE UNIT PER DIRECTION SEGMENT OF THE WHOLE STROKE (practitioner rule: a run
// is one unit plus a per-row shift, never re-chosen glyph by glyph; line
// repair step 4). The joined stroke (step 1) is cut into direction segments
// once, on its own path: a segment keeps the direction its chord holds after
// one row of length, and ends where the chord from its start turns more than
// three bins (about 17 degrees) from that. Every row of a segment reads the
// segment's direction column; the source supplies the shift. A row whose own
// chord turns more than three bins from its segment is a genuine local turn
// and reads its own (the former terrain-only rule, now every soft run).
const SEGMENTS=16u;
var<private> segmentEnd:array<u32,SEGMENTS>;
var<private> segmentDir:array<u32,SEGMENTS>;
var<private> segmentCount:u32;
fn directionSegments(run:Run){
 let base=run.range.x;let pairs=run.range.y/2u;segmentCount=0u;
 var open=false;var origin=vec2f(0.0);var held=NONE;var last=0u;var lastEnd=vec2f(0.0);
 for(var k=0u;k<pairs;k++){
  let a=points[base+k*2u].position.xy;let b=points[base+k*2u+1u].position.xy;if(length(b-a)<.01){continue;}
  if(!open){open=true;origin=a;held=NONE;}
  let reach=b-origin;
  if(held==NONE){if(dot(reach,reach)>=289.0){held=chordClass(reach);}}
  else{
   let turn=(chordClass(reach)+64u-held)%64u;
   if(min(turn,64u-turn)>3u&&segmentCount+1u<SEGMENTS){
    segmentEnd[segmentCount]=last;segmentDir[segmentCount]=held;segmentCount++;
    origin=lastEnd;let again=b-origin;held=select(NONE,chordClass(again),dot(again,again)>=289.0);
   }
  }
  last=k;lastEnd=b;
 }
 if(open){segmentEnd[segmentCount]=0xfffffffeu;segmentDir[segmentCount]=select(held,runDirection,held==NONE);segmentCount++;}
}
fn segmentDirection(pair:u32)->u32{
 for(var i=0u;i<segmentCount;i++){if(pair<=segmentEnd[i]){return segmentDir[i];}}
 return runDirection;
}
// D5 ON A CURVED FORM (line repair step 4): no upright or bracket unit on the
// same side in two consecutive rows. A run is one side; a curved run is a bent
// chain (runBent). A straight steep fold keeps its ずらし run (terrain table):
// alternating it with a slash would zigzag across a straight line. A unit all of whose ink is
// | ｜ l ｌ ! ( ) （ ） is an upright; on the second consecutive row it takes
// the direction's fitted non-upright unit, the neighbouring slash column on
// the side the chord leans to (the clean-line rule of the empty-phase
// fallback). A straight post or trunk side is not bent and keeps its run.
var<private> curvedRun:bool;
// A terrain run, or a crease or interior turn of any source: never an upright.
var<private> noUprightRun:bool;
var<private> lastUprightRow:i32;
fn uprightWord(word:u32)->bool{
 let n=words[word*WS];var inked=0u;
 for(var i=0u;i<n;i++){
  let font=words[word*WS+2u+i];let at=grammar[2]+font*4u;if(grammar[at+3u]==0u){continue;}inked++;
  let c=grammar[at];
  if(c!=0x7cu&&c!=0xff5cu&&c!=0x6cu&&c!=0xff4cu&&c!=0x21u&&c!=0x28u&&c!=0x29u&&c!=0xff08u&&c!=0xff09u){return false;}
 }
 return inked>0u;
}
fn slashNeighbour(dir:u32,chord:vec2f)->u32{
 let lean=dir%32u;let downward=dir<32u;
 if(lean==16u){return dir-lean+select(select(13u,19u,chord.x>=0.0),select(19u,13u,chord.x>=0.0),downward);}
 return dir-lean+select(max(lean,19u),min(lean,13u),lean<16u);
}
fn d5Swap(word:u32,row:i32)->bool{return curvedRun&&word!=0u&&abs(row-lastUprightRow)==1&&uprightWord(word);}
// A source program may cover many relation runs. Once the group aggregate
// has passed the complete-entry preflight, its canonical source member emits
// every native row and all other members yield; they never replay the same
// retained part independently from their local edge span.
fn programmedWholeSourcePart(runID:u32,run:Run,table:u32,size:u32,priority:u32)->bool{
 if(runProgramFeature(run)==0u||runProgramContext(run)==0u||run.range.y<2u){return false;}
 if(sourcePartPrepared(run)==NONE){return false;}
 let record=wholePartContextReady(run,runRole(run),size);if(record==NONE){return false;}
 var whole=wholePartLayoutFor(runID,run,record);
 if(whole.w==-2){let variant=wholePartVariantRecord(runID,run,runRole(run),size);if(variant==NONE){return false;}whole=wholePartLayoutFor(runID,run,variant);}
 if(whole.w<0){return false;}
 let leader=sourcePartLeader(run);if(leader==NONE){return false;}
 if(leader!=runID){return true;}
 for(var sourceRow=0;sourceRow<whole.w;sourceRow++){
  let row=whole.y+sourceRow;let q=softSourceAt(run,softProgramRowPair(runID,run,row),row);
  if(!programmedWholePart(runID,run,runRole(run),size,row,0u,true,0u,q,priority)){return false;}
 }
 return true;
}
fn softRunPlayback(runID:u32,run:Run,table:u32,size:u32,priority:u32){
 if(programmedWholeSourcePart(runID,run,table,size,priority)){return;}
 let base=run.range.x;let pairs=run.range.y/2u;
 runDirection=chordClass(finiteRunEnd(run).position.xy-finiteRunStart(run).position.xy);
 if(runProgramFeature(run)==0u){directionSegments(run);}curvedRun=runBent(run);lastUprightRow=-1000000;
 noUprightRun=terrainRun(run)||((run.range.z>>8u)&255u)>=3u;
 var open=false;var row=0;var portion=0u;var startPair=0u;var startT=0.0;var endPair=0u;var endT=0.0;
 var seen=false;var markPair=0u;var markT=0.0;
 for(var k=0u;k<pairs;k++){
  let pa=points[base+k*2u];let pb=points[base+k*2u+1u];
  let a=pa.position.xy;let b=pb.position.xy;let span=length(b-a);if(span<.01){continue;}
  // Rows off the page are one row each side: their pieces are never drawn.
  let rowA=clamp(i32(floor(a.y/17.0)),-1,i32(rowCount()));let rowB=clamp(i32(floor(b.y/17.0)),-1,i32(rowCount()));
  let pieces=u32(abs(rowB-rowA))+1u;let down=rowB>=rowA;
  for(var j=0u;j<pieces;j++){
   let r=select(rowA-i32(j),rowA+i32(j),down);
   var t0=0.0;var t1=1.0;
   if(j>0u){t0=clamp((f32(select(r+1,r,down))*17.0-a.y)/(b.y-a.y),0.0,1.0);}
   if(j+1u<pieces){t1=clamp((f32(select(r,r+1,down))*17.0-a.y)/(b.y-a.y),0.0,1.0);}
   if((t1-t0)*span<.01){continue;}
   if(open&&r!=row){softPortion(runID,run,table,size,row,portion,startPair,startT,endPair,endT,seen,markPair,markT,priority);open=false;}
   if(!open){open=true;row=r;portion++;startPair=k;startT=t0;seen=false;}
   endPair=k;endT=t1;
   if(!seen&&pointVisible(pa)){seen=true;markPair=k;markT=(t0+t1)*.5;}
  }
 }
 if(open){softPortion(runID,run,table,size,row,portion,startPair,startT,endPair,endT,seen,markPair,markT,priority);}
}

// item H). A rowLed unit of a straight hard run is spelled wherever the run's
// VISIBLE intervals (pair start flag, the run pass's occlusion: interval.cause)
// cover ANY part of the span the unit stands for in row [r0,r1): its advance
// along x (shallow) or the row's y-span [lo,hi] (steep). Only a unit with no
// visible part is skipped; a partially visible unit is emitted and the mapped
// pass (realizeMapped) clips its hidden members per scalar. The per-glyph
// depth test (sourceGlyphVisible) passes a
// self-occluded span, since its own surface counts as visible, so a hidden wall
// top was spelled a row-height above its line, across another house's gable

// already skip hidden pairs; this is the same rule for the row walk.
fn rowUnitVisible(run:Run,r0:f32,r1:f32,shallow:bool,lo:f32,hi:f32,fallback:bool)->bool{
 if(hi-lo<.00001){return fallback;}
 var covered=0.0;
 for(var k=0u;k+1u<run.range.y;k+=2u){
  let a=points[run.range.x+k];if(!pointVisible(a)){continue;}
  let pa=a.position.xy;let pb=points[run.range.x+k+1u].position.xy;var t=vec2f(0.0,1.0);
  if(abs(pb.y-pa.y)<.00001){if(pa.y<r0||pa.y>=r1){continue;}}
  else{let u=(r0-pa.y)/(pb.y-pa.y);let v=(r1-pa.y)/(pb.y-pa.y);t=vec2f(max(0.0,min(u,v)),min(1.0,max(u,v)));if(t.y<=t.x){continue;}}
  let s=select(vec2f(mix(pa.y,pb.y,t.x),mix(pa.y,pb.y,t.y)),vec2f(mix(pa.x,pb.x,t.x),mix(pa.x,pb.x,t.y)),shallow);
  covered+=max(0.0,min(hi,max(s.x,s.y))-max(lo,min(s.x,s.y)));
 }
 return covered>0.0;
}
@compute @workgroup_size(64) fn featurecurves(@builtin(global_invocation_id) id:vec3u){
 let runID=scheduledRun(id);
 if(runID==0xffffffffu||runID>=p.extent.z){return;}
 var run=runs[runID];if(run.range.y==0u){return;}
 gridKeyBase=0u;collapseDir=NONE;
 let first=finiteRunStart(run).position;let last=finiteRunEnd(run).position;
 let anchorOnly=(run.range.z&0x10000u)!=0u;
 let boundary=wholeBoundaryFlags(run);
 for(var i=0u;i<run.range.y;i++){if(pointVisible(points[run.range.x+i])){atomicOr(&counts[drawnSlot(run)],0x80000000u);break;}}
 var memo=0u;
 if(boundary==0u){
  let slot=atomicLoad(&counts[memoMapBase()+runID]);if(slot==NONE){return;}
  memo=u32(p.field.z)+slot*${MEMO_WORDS}u+${MEMO_HEADER_WORDS}u;
 }
 if(anchorOnly){
  // Whole anchors keep their native spelling as the camera approaches.
  // Only a changed role/declared-port context or source lifetime replaces
  // the part. The intervening run, not this part, absorbs added row units.
  let same=atomicLoad(&counts[memo-${MEMO_HEADER_WORDS}u])==3u&&atomicLoad(&counts[memo])==run.identity.w&&atomicLoad(&counts[memo+1u])==run.range.z;
  if(same){
   run.part=vec4u(atomicLoad(&counts[memo+5u]),atomicLoad(&counts[memo+6u]),atomicLoad(&counts[memo+4u]),atomicLoad(&counts[memo+3u]));
   run.paper.x=atomicLoad(&counts[memo+7u]);run.paper.y=atomicLoad(&counts[memo+8u]);
  }else{
   atomicStore(&counts[memo],run.identity.w);atomicStore(&counts[memo+1u],run.range.z);
   atomicStore(&counts[memo+3u],run.part.w);atomicStore(&counts[memo+4u],run.part.z);
   atomicStore(&counts[memo+5u],run.part.x);atomicStore(&counts[memo+6u],run.part.y);atomicStore(&counts[memo+7u],run.paper.x);atomicStore(&counts[memo+8u],run.paper.y);
   atomicStore(&counts[memo-${MEMO_HEADER_WORDS}u],3u);
  }
 }
 // Selection by distance tier, the ONE gate for hard and generic runs. A
 // deselected run emits nothing and is not a word-placement loss; anchors
 // (joins, caps) stay at every distance, except on a form that is a place on
 // the far strip: a 遠景 field-class form is spelled by the merged skyline and
 // nothing else (anchors exist only on form-0 sources; a foot anchor names
 // its own class).
 // CONSTRUCTION BY TIER, one row per form class; tier = the form's ONE tier,
 // its rows (distanceTier; THE TIER IS THE ROWS THE DRAWING HAS):
 //  path-led / region-led (field-class soft runs)
 //   tier 0: both outlines, paired flank columns on limbs (softPortion), the
 //           crown envelope left to the tone producer, creases across a tone
 //           change, joins and contacts; band-2 words at most.
 //   tier 1: outline, a thin limb as one stroke, no paired column, principal
 //           creases, joins, contact as the foot; band-1 words at most.
 //   tier 2: its place on the merged far skyline and nothing else (farSkyline
 //           THE FAR UNITS: one mark on its stem when taller than wide, else
 //           a low lobe with its turn and base).
 //  planar volume / hard-edged: the closed edge construction with its
 //           principal creases in 近景 and 中景 (hardBoundaryConstruction);
 //           a far whole box of two rows or less is its top line on the
 //           merged skyline (TERRAIN table, so adjacent boxes are one mass),
 //           a taller far box keeps its construction and occupies the strip.
 //  round:   the circle by its height in rows in 近景 and 中景
 //           (circleConstruction); far, a mark or a lobe on the strip.
 //  terrain: per run. tier 0 skyline, slope turns, contacts; tier 1 ridges,
 //           shelf edges, contacts; tier 2 the merged skyline for a run of two
 //           rows or less (farSkyline), joined slope strokes above.
 let away=heldTier(runID,run);let tier=away;lineTier=tier;keepTier(runID,run,tier);
 // No anchor under a 遠景 form, foot junctions included: the form's distance
 // tier (the one its strip reads) decides, so a form on the strip never also
 // draws a join or a foot (从, the trunk column); 近景 and 中景 forms keep theirs.
 // A foot passage is detail of a form that holds lines of its own. A form
 // that stands on under five rows away from the viewer is its profile on the
 // base line (farKind); the three-row passage stamped at its foot was as tall

 // 中景 keeps "principal features and joins" (agreed design 1): only a form
 // the strip takes (too small for one unit), or a 遠景 form under five rows,
 // has no anchor of its own.
 if(anchorOnly&&away!=0u&&(farUnderGlyph(run)||(away==2u&&farStands(run)<85.0))){return;}
 if(!anchorOnly&&!tierSelects(run,tier)){return;}
 if(!anchorOnly&&rippleRun(run)&&rippleYields(run)){return;}
 if(!anchorOnly&&terrainRun(run)){probeAdd(35u+min(tier,2u)*5u+min((run.range.z>>8u)&255u,4u));}
 if(!anchorOnly&&!terrainRun(run)&&(run.range.z&0x40000u)!=0u&&wholeBoundaryFlags(run)==0u){probeAdd(52u+min(tier,2u)*5u+min((run.range.z>>8u)&255u,4u));}
 let baseRow=i32(floor(first.y/17.0));let baseRank=min(rank(run,first.xy,last.xy),0x7fffffffu);let priority=select(baseRank,0x80000000u|baseRank,anchorOnly);
 // 遠景: a far terrain silhouette of two rows or less and every far standing
 // field-class form are spelled by the merged skyline and by nothing else.
 if(away!=0u&&!anchorOnly){
  let span=skySpan(run);let far=farKind(run,away,span);
  if((far==1u||far==3u)&&farSkyline(runID,run,far,span,priority)){return;}
 }
 // A crease or interior turn (class 3, 4) yields to a parallel line once per
 // run (tierSelects, creaseRead z), never word by word: the per-word grid cut
 // folds into broken columns wherever a neighbouring fold, itself cut, held
 // the cell (view-06 cliff, line repair 2026-11-12).
 if(!anchorOnly&&boundary==0u&&run.range.y>=2u&&((run.range.z>>8u)&255u)<3u){gridWrite(run,baseRank);}
 if(!anchorOnly){hardJunction(runID,run,baseRank);}
 if((boundary&0x100000u)!=0u){hardBoundaryConstruction(runID,run,priority,true);return;}
 if((boundary&0x4000000u)!=0u){hardBoundaryConstruction(runID,run,priority,false);return;}
 if((boundary&0x20000u)!=0u){circleConstruction(runID,run,priority);return;}
 if(anchorOnly&&pointVisible(finiteRunStart(run))){for(var i=0u;i<run.paper.y;i++){
  let at=grammar[6]+(run.paper.x+i)*4u;let start=i32(round(first.x))+bitcast<i32>(run.part.z)+i32(grammar[at]);
  let row=baseRow+bitcast<i32>(run.part.w)+i32(grammar[at+1u]);let end=start+i32(grammar[at+2u]);
  if(row<0||row>=i32(rowCount())){continue;}
  for(var x=max(0,start);x<min(i32(p.extent.x),end);x++){
   let q=fieldPixel(vec2f(f32(x)+.5,f32(row*17)+8.5));if(any(q<vec2i(0))||any(q>=vec2i(p.field.xy))){continue;}
   let owner=textureLoad(ownerField,q,0).xy;let same=(owner.x&0xdfffffffu)==run.identity.y&&owner.y==run.identity.z;
   if((owner.x|owner.y)!=0u&&!same&&textureLoad(radianceField,q,0).a>=.99&&textureLoad(depthField,q,0).x>first.z+.000001){continue;}
   atomicOr(&counts[paperBase()+u32(row)*paperStride()+u32(x)/32u],1u<<(u32(x)%32u));
  }
 }}
 // Whole retained parts are independent of the continuing-word table.
 if(anchorOnly&&run.part.y!=0u&&pointVisible(finiteRunStart(run))){for(var i=0u;i<run.part.y;i++){
  let at=grammar[1]+(run.part.x+i)*4u;let atom=vec4u(grammar[at],grammar[at+1u],grammar[at+2u],grammar[at+3u]);
  let row=baseRow+bitcast<i32>(run.part.w)+i32(atom.z);let x=i32(round(first.x))+bitcast<i32>(run.part.z)+bitcast<i32>(atom.y);
  emit(runID,i,first.w,x,row,atom.x,1u,atom.w,priority,1u,first.z);
 }}
 if(anchorOnly){if(run.part.y==0u&&pointVisible(finiteRunStart(run))){atomicAdd(&counts[rowCount()+1u],1u);probeAdd(0u);}return;}
 if(run.range.y<2u){return;}
 let delta=last.xy-first.xy;if(length(delta)<.00001){return;}
 var straightVertical=true;var bent=false;var monotone=true;var havePrevious=false;var previousY=first.y;
 for(var i=0u;i<run.range.y;i++){
  let point=points[run.range.x+i];if(pointWholeEdgeDescriptor(point)){continue;}
  let q=point.position.xy;straightVertical=straightVertical&&abs(q.x-first.x)<.5;
  bent=bent||abs((q.x-first.x)*delta.y-(q.y-first.y)*delta.x)>.25*max(1.0,length(delta));
  if(havePrevious){monotone=monotone&&(q.y-previousY)*sign(delta.y)>=-.00001;}
  previousY=q.y;havePrevious=true;
 }
 // Practitioner upright run: a posed WORLD-UP edge keeps its identity while
 // its perspective lean is below half a native cell advance per row.
 // Perspective lean accumulates over a tall edge; it is not a reason to
 // turn the box corner into the soft table's colon/slash family. Curves and
 // genuinely diagonal hard edges do not acquire upright permission.
 if((run.range.z&0x40000u)!=0u&&!bent){
  let lean=select(1.0,5.5,(run.range.z&0x80000u)!=0u);
  straightVertical=abs(delta.x)*17.0<=abs(delta.y)*lean;
 }
 // A straight SOFT edge (the side of a post or a trunk) leaning at most one
 // dot per row is the practitioner upright run too; the playback moves the
 // unit by that shift. A curved run never qualifies (UPRIGHT_RUN).
 if((run.range.z&0x40000u)==0u&&!bent){straightVertical=abs(delta.x)*17.0<=abs(delta.y);}
 // Current geometry decides the grammar, not silhouette vs occluding label.
 let articulated=!straightVertical||bent;
 let direction=normalize(delta);let angle=atan2(direction.y,direction.x);
 var dir=u32(round(fract(angle/6.28318530718+1.0)*64.0))&63u;
 let hardLine=(run.range.z&0x40000u)!=0u&&!bent;
 let horizontal=hardLine&&abs(delta.y)<=.5;
 if((run.range.z&0x40000u)!=0u&&straightVertical&&!bent){dir=select(48u,16u,delta.y>0.0);}
 if(horizontal){dir=select(32u,0u,delta.x>0.0);}
 // Direction quantization must not turn a visibly shallow slant into the
 // flat underscore program. Keep it in the shallow-line family instead.
 if(hardLine&&!horizontal&&(dir==0u||dir==32u)){
  dir=select(select(33u,31u,delta.y>0.0),select(63u,1u,delta.y>0.0),delta.x>0.0);
 }
 let phase=u32(clamp(floor(first.y)-f32(baseRow*17),0.0,16.0));
 let roleContext=runRole(run)*3u+runSize(run);let context=roleContext+select(0u,12u,articulated)+select(0u,24u,hardLine);
 var instruction=select(select(grammar[0],grammar[4],articulated),grammar[8],hardLine)+((roleContext*64u+dir)*17u+phase)*12u;
 let previous=atomicLoad(&counts[memo+2u]);let turn=min((dir+64u-previous)%64u,(previous+64u-dir)%64u);
 let retained=atomicLoad(&counts[memo-${MEMO_HEADER_WORDS}u])==3u&&atomicLoad(&counts[memo])==run.identity.w&&atomicLoad(&counts[memo+1u])==context&&turn<=select(1u,0u,hardLine);
 if(retained){instruction=atomicLoad(&counts[memo+4u]);}
 let soft=!hardLine&&(run.range.y&1u)==0u;
 let word=grammar[instruction];if(word==0u&&!soft){atomicAdd(&counts[rowCount()+1u],1u);probeAdd(1u);return;}
 // Retain a surviving run's COMPLETE unit. Row-phase drift is not a reason
 // to respell it. A genuine role/size/direction context change does one new
 // lookup; recycled source slots carry a different lifetime epoch.
 if(!retained){atomicStore(&counts[memo],run.identity.w);atomicStore(&counts[memo+1u],context);atomicStore(&counts[memo+2u],dir);atomicStore(&counts[memo+3u],phase);atomicStore(&counts[memo+4u],instruction);atomicStore(&counts[memo-${MEMO_HEADER_WORDS}u],3u);}
 // Soft runs are played back row by row from the compiled direction columns
 // at the size band of the run's distance tier. The run-level record above
 // is only the memo's context key for them.
 // A terrain run reads the terrain line table (grammar[11], after the
 // paired-flank records) by the same address.
 if(soft){
  let terrainTable=grammar[11]+${(PAIRED_FLANK_LEANS.length+1)*4}u;
  softWords=0u;softYield=false;
  softRunPlayback(runID,run,select(select(grammar[0],grammar[4],articulated),terrainTable,terrainRun(run)),min(runSize(run),tierBand(tier)),priority);
  // Loss accounting: a selected run with a visible span that handed no word
  // to placement, and did not yield a row to its pair, drew nothing.
  if(softWords==0u&&!softYield){
   for(var i=0u;i<run.range.y;i+=2u){if(pointVisible(points[run.range.x+i])){atomicAdd(&counts[rowCount()+1u],1u);probeAdd(2u);break;}}
  }
  return;
 }
 let count=grammar[instruction+1u];let advance=grammar[instruction+2u];
 let shift=bitcast<i32>(grammar[instruction+3u]);let offset=bitcast<i32>(grammar[instruction+8u]);
 let a=grammar[instruction+6u];let b=grammar[instruction+7u];
 let entry=vec2f(f32(a&255u)-128.0,f32((a>>8u)&255u)-128.0);
 let exit=vec2f(f32(b&255u)-128.0,f32((b>>8u)&255u)-128.0);
 let rowLed=hardLine&&!horizontal&&monotone&&(run.range.y&1u)==0u;
 if(rowLed){
  // One WHOLE selected unit per native row on a straight hard line. The
  // source run owns the shift; native rows own repetition. Soft runs have
  // their own row playback above.
  let low=min(first.y,last.y);let high=max(first.y,last.y);
  let firstRow=i32(floor(low/17.0));let endRow=i32(ceil(high/17.0));
  // Matches the compiled hard-line table: below 45° a hard edge is a stepped
  // run of whole ― units across each row, never a curved kana stroke.
  let shallow=abs(direction.y)<.70710678;
  let wholeSlope=grammar[instruction+9u]==1u;collapseDir=gridDir(direction);
  let unitWord=select(word,grammar[instruction+4u],wholeSlope);
  let unitCount=words[unitWord*WS];let unitAdvance=words[unitWord*WS+1u];
  let unitOffset=select(offset,bitcast<i32>(grammar[instruction+5u]),wholeSlope);
  var segment=select(run.range.y-2u,0u,delta.y>0.0);
  for(var row=max(0,firstRow);row<min(i32(rowCount()),endRow);row++){
   let lower=max(low,f32(row)*17.0);let upper=min(high,f32(row+1)*17.0);
   let y=clamp(f32(row)*17.0+8.5,lower,upper);
   // Walk actual [start,end] source pairs, not the zero-length connector
   // between two adjacent visibility intervals. Internal upper endpoints
   // are half-open, so a hidden→visible cut belongs to the visible interval
   // that starts there. The outermost endpoint remains included.
   var found=false;
   loop{
    let a=points[run.range.x+segment].position.y;let b=points[run.range.x+segment+1u].position.y;
    let lastSpan=select(segment==0u,segment+2u>=run.range.y,delta.y>0.0);
    if(abs(b-a)>.00001&&y>=min(a,b)&&(y<max(a,b)||(lastSpan&&y<=max(a,b)))){found=true;break;}
    if(lastSpan){break;}if(delta.y>0.0){segment+=2u;}else{segment-=2u;}
   }
   if(!found){continue;}
   let pa=points[run.range.x+segment];let pb=points[run.range.x+segment+1u];
   // A straight hard span can cross an occluder inside this native row.
   // Its midpoint's interval cannot own the visibility of every repeat
   // unit in the row. Keep the selected passage and cut its members against
   // the source span below; soft paths retain their interval semantics.
   let t=clamp((y-pa.position.y)/(pb.position.y-pa.position.y),0.0,1.0);let q=interpolatePoint(pa,pb,t);
   let x0=first.x+(lower-first.y)*delta.x/delta.y;let x1=first.x+(upper-first.y)*delta.x/delta.y;
   // Shallow HARD edges retain their horizontal repeat span. A steep path
   // carries exactly one complete grammar unit, never an ink-length quota.
   let units=select(1u,max(1u,u32(ceil(abs(x1-x0)/f32(unitAdvance)))),shallow&&!wholeSlope);
   var origin=select(q.x,(x0+x1)*.5-f32(units-1u)*f32(unitAdvance)*.5,shallow&&!wholeSlope);
   // A graded unit carries the WHOLE row's rise. Keep its row-centre pen
   // even when only an endpoint fragment survives: clipping changes visible
   // members, not the phase/placement of the surviving construction.
   if(wholeSlope){origin=first.x+(f32(row)*17.0+8.5-first.y)*delta.x/delta.y;}
   for(var unit=0u;unit<units;unit++){

    // unit is named by its row relative to the run's first source point and its
    // integer pen offset from that point (softMemberOrdinal), not by a count from
    // the run's current top row and the row-centred origin.
    let pen=i32(round(origin))+i32(unit*unitAdvance)+unitOffset;
    let ordinal=softMemberOrdinal(finiteRunStart(run),row,pen-i32(round(finiteRunStart(run).position.x)));
    // Slice 1c: the unit's source must be visible (rowUnitVisible), as in the repeat program.
    let centre=origin+f32(unit*unitAdvance);
    if(!rowUnitVisible(run,f32(row)*17.0,f32(row+1)*17.0,shallow,select(lower,centre-f32(unitAdvance)*.5,shallow),select(upper,centre+f32(unitAdvance)*.5,shallow),pointVisible(pa))){continue;}
    sourceWord(runID,ordinal,q,pen,row,unitWord,unitCount,unitAdvance,priority,finiteRunStart(run),finiteRunEnd(run));
   }
  }
  return;
 }
 // A straight repeat program advances by a WHOLE native word. Multiplying
 // by direction.x made shallow runs advance LESS than their own word width
 // on the page (advance * dx * dx). Successive units consequently collided
 // with their own run and disappeared in placement. Invert the projection
 // instead: one arc step projects to exactly one measured native advance.
 // Bent paths keep their source-span progression; steep monotone paths have
 // already returned through the one-unit-per-row program above.
 let wholeAdvance=hardLine||(!bent&&abs(direction.x)>=.92387953);
 let step=select(max(1.0,max(abs(direction.x)*f32(advance),abs(dot(exit-entry,direction)))),f32(advance)/max(abs(direction.x),.00001),wholeAdvance);
 var distance=0.0;collapseDir=gridDir(direction);
 for(var i=0u;i+1u<run.range.y;i++){
  let pa=points[run.range.x+i];let pb=points[run.range.x+i+1u];let span=length(pb.position.xy-pa.position.xy);
  if(span<.00001){continue;}
  if(pointVisible(pa)){
   // Skip the invisible prefix arithmetically. Ordinals still belong to
   // the COMPLETE feature; neither clipping nor rows restart its phase.
   let lo=u32(max(0.0,ceil(distance/step)));let hi=u32(max(0.0,ceil((distance+span)/step)));
   for(var unit=lo;unit<hi;unit++){
    let t=clamp((f32(unit)*step-distance)/span,0.0,1.0);let q=interpolatePoint(pa,pb,t);
    let row=i32(floor(q.y/17.0))+shift;let x=i32(round(q.x))+offset;
    emit(runID,unit,q.w,x,row,word,count,advance,priority,0u,q.z);
   }
  }
  distance+=span;
 }
}
// ONE LEVEL LINE, ONE SPELLING. Two edges a few pixels apart on either side
// of a row boundary were ＿ in the upper row and ￣ in the lower one: two

// call-out 2026-11-10, view 06: "double stacked horizontal lines"; AGENTS
// finding 4: parallel lines inside one glyph width collapse). A level unit
// gives way to the opposite level unit of another run across the boundary
// that overlaps it by half its width and stands before it in the fixed
// priority order. Both rows read the same emitted lists, so the decision is
// the same from either side and one of the two always stays.
// The twin test reads only the emitted lists, never the placement, so the
// row's 128 lanes evaluate it for every word in parallel after the sort
// (levelTwinScan into twinBits) and the one placing lane reads the bit. The
// scan of the neighbouring row per level word ran inside that one lane:
// quadratic per row, the walk-time growth of featureplace (30 to 103 ms per
// update at view 06, gpu-profile-20261112b).
var<workgroup> twinBits:array<atomic<u32>,${ROW_CAP/32}>;
var<private> twinsReady:bool=false;
var<private> twinBase:u32=0u;
fn levelTwin(index:u32,row:u32,lows:u32,highs:u32)->bool{
 let twin=levelTwinBit(index,row,lows,highs);
 if(twin&&rowWords[index].data[9]!=3u&&rowWords[index].data[9]!=2u){atomicAdd(&counts[featureRecord(rowWords[index].data[0])+1u],1u);}
 return twin;
}
fn levelTwinBit(index:u32,row:u32,lows:u32,highs:u32)->bool{
 if(twinsReady&&index>=twinBase&&index-twinBase<${ROW_CAP}u){let i=index-twinBase;return (atomicLoad(&twinBits[i/32u])&(1u<<(i%32u)))!=0u;}
 return levelTwinScan(index,row,lows,highs);
}
fn levelTwinScan(index:u32,row:u32,lows:u32,highs:u32)->bool{
 let x=rowWords[index].data;if(x[9]!=0u||(x[5]!=lows&&x[5]!=highs)||lows==0u){return false;}
 var other=row+1u;var want=highs;if(x[5]==highs){if(row==0u){return false;}other=row-1u;want=lows;}
 if(other>=rowCount()){return false;}
 let n=min(atomicLoad(&counts[other]),rowLimit(other));let start=rowBase(other);
 let a0=i32(x[3]);let a1=a0+i32(x[7]);
 for(var k=0u;k<n;k++){
  let y=rowWords[start+k].data;if(y[9]!=0u||y[5]!=want){continue;}
  let b0=i32(y[3]);let b1=b0+i32(y[7]);let overlap=min(a1,b1)-max(a0,b0);
  if(overlap*2<min(a1-a0,b1-b0)){continue;}
  // Equal priority: the row's own total order by content (before), never the
  // run slot, which is this frame's atomic append order.
  if(y[8]>x[8]||(y[8]==x[8]&&before(start+k,index))){return true;}
 }
 return false;
}
var<workgroup> order:array<u32,${ROW_CAP}>;
var<workgroup> itemCount:u32;
fn before(a:u32,b:u32)->bool{
 if(a==NONE){return false;}if(b==NONE){return true;}
 let x=rowWords[a].data;let y=rowWords[b].data;
 if(x[8]!=y[8]){return x[8]>y[8];}
 // Region tone words of one row commit left to right. The tape's native gap
 // is checked against the committed neighbour, so an abutting field is legal
 // only in that order: a unit whose right-hand neighbour was still unplaced
 // saw an unspellable gap to the unit beyond it and was dropped (fog, loop
 // 346: two of every three wall units lost).
 if(x[9]==3u&&y[9]==3u&&x[3]!=y[3]){return x[3]<y[3];}
 let kx=runs[x[0]].identity.x;let ky=runs[y[0]].identity.x;
 if(kx!=ky){return kx<ky;}
 // The incumbent hash ordering stays unchanged. A hash collision is not
 // a source identity tie: ORIGINAL source addresses break those collisions,
 // independent of atomic emission or the row sort's initial permutation.
 let sx=rowSource(rowWords[a]);let sy=rowSource(rowWords[b]);
 for(var i=0u;i<4u;i++){if(sx[i]!=sy[i]){return sx[i]<sy[i];}}
 if(x[9]==0u&&y[9]==0u){
  let cx=spanSource(runs[x[0]],x[16]).identity.yz;let cy=spanSource(runs[y[0]],y[16]).identity.yz;
  if(cx.x!=cy.x){return cx.x<cy.x;}if(cx.y!=cy.y){return cx.y<cy.y;}
 }
 // Within the SAME selected source passage, commit native words in reading
 // order. Source ordinals may run right-to-left; using them as placement
 // order mistakes an as-yet-unplaced word for an unspellable whitespace
 // gap. Keep identity/priority ordering between sources and keep ordinals
 // as correspondence, not as the row's serialization direction.
 if(x[3]!=y[3]){return x[3]<y[3];}
 if(x[9]==0u&&y[9]==0u&&x[15]==1u&&y[15]==1u&&x[14]!=y[14]){return x[14]<y[14];}
 if(x[1]!=y[1]){return x[1]<y[1];}
 if(x[13]!=y[13]){return x[13]<y[13];}
 // A TOTAL order (crawl, 2026-11-11): two different words equal on every
 // key above came out in emission order, which is GPU thread timing; the
 // one placed first takes the cells, so a stationary Y8 view changed 4-10%
 // of its ink every half second. The rest of the word's content decides
 // (word 0 is the run slot, itself emission order, and is left out).
 for(var i=1u;i<${ROW_WORDS}u;i++){
  // A part0 source-context slot is an atomic allocation, not content.
  if(i==16u&&(x[9]==0u||y[9]==0u)){continue;}
  if(x[i]!=y[i]){return x[i]<y[i];}
 }
 return false;
}
// Dense rows use the SAME comparator as ordinary rows, but sort references,
// never RowWord records. Adjacent rows read those records in levelTwin.
// Tile-local bitonic ordering is parallel; larger rows merge ordered tiles
// with one invocation per word. No source word is copied, respelled or lost.
// Scratch is in the camera-owned transient tail of counts, not a ninth
// storage binding or a second publication record.
@compute @workgroup_size(128) fn featureorder(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let row=group.y;
 if(lane==0u){itemCount=min(atomicLoad(&counts[row]),rowLimit(row));}
 let total=workgroupUniformLoad(&itemCount);
 let tile=group.x*${ROW_CAP}u;if(total<=${ROW_CAP}u||tile>=total){return;}
 let count=min(${ROW_CAP}u,total-tile);let start=rowBase(row);
 var span=1u;while(span<count){span*=2u;}
 for(var i=lane;i<span;i+=128u){order[i]=select(NONE,start+tile+i,i<count);}
 workgroupBarrier();
 for(var k=2u;k<=span;k*=2u){for(var j=k/2u;j>0u;j/=2u){
  for(var i=lane;i<span;i+=128u){let other=i^j;if(other>i){let a=order[i];let b=order[other];if(before(b,a)==((i&k)==0u)){order[i]=b;order[other]=a;}}}
  workgroupBarrier();
 }}
 for(var i=lane;i<count;i+=128u){atomicStore(&counts[p.playback.w+start+tile+i],order[i]);}
}
@compute @workgroup_size(64) fn featureorderMerge(@builtin(global_invocation_id) id:vec3u){
 let row=id.y;let i=id.x;let count=min(atomicLoad(&counts[row]),rowLimit(row));
 if(count<=${ROW_CAP}u||i>=count){return;}
 // In this pass only: extent.z is the input reference address, extent.w
 // the merge span; playback.w is the output reference address.
 let span=p.extent.w;let start=rowBase(row);
 let base=(i/(span*2u))*(span*2u);let right=i-base>=span;
 let own=select(base,base+span,right);let other=select(base+span,base,right);
 let end=min(other+span,count);let index=atomicLoad(&counts[p.extent.z+start+i]);
 var lo=min(other,count);var hi=end;
 // Binary partition of an already ordered tile by the unchanged priority
 // comparator, not glyph/pen search. Equal keys from the left tile stay
 // first, so the merge is stable even for duplicate identical words.
 while(lo<hi){let mid=lo+(hi-lo)/2u;let candidate=atomicLoad(&counts[p.extent.z+start+mid]);
  let earlier=select(before(candidate,index),!before(index,candidate),right);
  if(earlier){lo=mid+1u;}else{hi=mid;}
 }
 let destination=base+(i-own)+(lo-min(other,count));
 atomicStore(&counts[p.playback.w+start+destination],index);
}
// Row-local order-statistic index of accepted scalar intervals. Only lane
// zero writes its row. A prefix count and rank-select give the exact prior
// and next interval in O(log(page width)), without rescanning every accepted
// word, trying pens, or changing the fixed priority decision.
fn cursorBase(row:u32)->u32{return p.extent.w*${WORD_WORDS}u+row*(p.extent.x*2u+1u);}
fn storedCount(base:u32,end:u32)->u32{
 var i=end;var count=0u;
 while(i>0u){count+=output[base+i];i-=i&(~i+1u);}
 return count;
}
fn storedIndex(base:u32,ordinal:u32)->u32{
 var bit=1u;while(bit<=p.extent.x/2u){bit*=2u;}
 var index=0u;var remaining=ordinal;
 while(bit>0u){let next=index+bit;if(next<=p.extent.x&&output[base+next]<remaining){remaining-=output[base+next];index=next;}bit/=2u;}
 return index;
}
fn storedInsert(base:u32,pen:u32,address:u32){
 output[base+p.extent.x+1u+pen]=address;
 var i=pen+1u;while(i<=p.extent.x){output[base+i]++;i+=i&(~i+1u);}
}

// cursor queries were chains of dependent device-memory loads on its one lane
// (about 23 us per word). featureplace keeps the row's occupancy as a pen
// bitmask plus a Fenwick tree over its 256 word popcounts in workgroup memory
// and answers the same two queries from it; addresses stay in the stored
// table. A bitmask holds counts 0 and 1 only: a row whose stored cursor holds
// any other count, a page wider than 8192 px, or an insert on an occupied pen
// stays on (or returns to) the stored Fenwick, so the answers never differ.
const LOCAL_WORDS=256u;
const PLACE_CHECK=false;
var<workgroup> localBits:array<u32,256>;
var<workgroup> localSums:array<u32,257>;
var<workgroup> localBroken:atomic<u32>;
var<private> localCursor:bool=false;
var<private> localBase:u32=0u;
var<private> localTotal:u32=0u;
fn placeCheck(k:u32){if(PLACE_CHECK&&rowCount()+98u<=512u){atomicAdd(&counts[rowCount()+94u+k],1u);}}
fn localPrefix(words:u32)->u32{var i=words;var n=0u;while(i>0u){n+=localSums[i];i-=i&(~i+1u);}return n;}
fn localCount(end:u32)->u32{let w=end/32u;let b=end%32u;var n=localPrefix(w);if(b!=0u){n+=countOneBits(localBits[w]&((1u<<b)-1u));}return n;}
// The stored rank-select's index: the pen of the ordinal-th scalar, 0 for
// ordinal 0, extent.x past the last scalar.
fn localIndex(ordinal:u32)->u32{
 if(ordinal==0u){return 0u;}
 if(ordinal>localTotal){return p.extent.x;}
 var bit=LOCAL_WORDS;var w=0u;var remaining=ordinal;
 while(bit>0u){let next=w+bit;if(next<=LOCAL_WORDS&&localSums[next]<remaining){remaining-=localSums[next];w=next;}bit/=2u;}
 let word=localBits[min(w,LOCAL_WORDS-1u)];var pen=0u;
 for(var half=16u;half>0u;half/=2u){let c=countOneBits((word>>pen)&((1u<<half)-1u));if(c<remaining){remaining-=c;pen+=half;}}
 return w*32u+pen;
}
fn localInsert(pen:u32)->bool{
 let w=pen/32u;let m=1u<<(pen%32u);if((localBits[w]&m)!=0u){return false;}
 localBits[w]|=m;localSummary[w/32u]|=1u<<(w%32u);var j=w+1u;while(j<=LOCAL_WORDS){localSums[j]+=1u;j+=j&(~j+1u);}
 localTotal+=1u;return true;
}
// Bit w%32 of localSummary[w/32]: localBits[w] holds a scalar.
var<workgroup> localSummary:array<u32,8>;
// The last committed pen < e (e <= 8192), or NONE.
fn localPrev(e:u32)->u32{
 let w=e/32u;let b=e%32u;
 if(b!=0u){let m=localBits[w]&((1u<<b)-1u);if(m!=0u){return w*32u+31u-countLeadingZeros(m);}}
 var s=w/32u;var m=0u;if(s<8u){m=localSummary[s]&((1u<<(w%32u))-1u);}else{s=8u;}
 while(m==0u&&s>0u){s-=1u;m=localSummary[s];}
 if(m==0u){return NONE;}
 let word=s*32u+31u-countLeadingZeros(m);return word*32u+31u-countLeadingZeros(localBits[word]);
}
// The first committed pen >= e, or NONE.
fn localNext(e:u32)->u32{
 if(e>=LOCAL_WORDS*32u){return NONE;}
 let w=e/32u;let m0=localBits[w]&~((1u<<(e%32u))-1u);
 if(m0!=0u){return w*32u+countTrailingZeros(m0);}
 var s=w/32u;var m=localSummary[s]&~((2u<<(w%32u))-1u);
 while(m==0u&&s<7u){s+=1u;m=localSummary[s];}
 if(m==0u){return NONE;}
 let word=s*32u+countTrailingZeros(m);return word*32u+countTrailingZeros(localBits[word]);
}
fn cursorAddr(base:u32,pen:u32)->u32{return output[base+p.extent.x+1u+pen];}
// The same neighbours from the stored Fenwick: the scalar at rank count(e),
// and the one at rank count(e)+1 (exactly the rank pairs they replace).
fn storedLast(base:u32,e:u32)->u32{let n=storedCount(base,e);if(n==0u){return NONE;}return storedIndex(base,n);}
fn storedFirst(base:u32,e:u32)->u32{let n=storedCount(base,e);if(n>=storedCount(base,p.extent.x)){return NONE;}return storedIndex(base,n+1u);}
fn cursorLast(base:u32,e:u32)->u32{
 if(localCursor&&base==localBase){let pen=localPrev(e);if(PLACE_CHECK){placeCheck(0u);if(pen!=storedLast(base,e)){placeCheck(2u);}}return pen;}
 return storedLast(base,e);
}
fn cursorFirst(base:u32,e:u32)->u32{
 if(localCursor&&base==localBase){let pen=localNext(e);if(PLACE_CHECK){placeCheck(0u);if(pen!=storedFirst(base,e)){placeCheck(2u);}}return pen;}
 return storedFirst(base,e);
}
// The stored Fenwick node c (pens c-lowbit(c) .. c-1) of the local bitmask.
fn localNode(c:u32)->u32{return localCount(c)-localCount(c-(c&(~c+1u)));}
// Leave the local copy (an insert on an occupied pen): the stored Fenwick is
// written whole from it, and the row continues on the stored cursor.
fn localLeave(){
 placeCheck(3u);
 if(!PLACE_CHECK){for(var c=1u;c<=p.extent.x;c++){output[localBase+c]=localNode(c);}}
 localCursor=false;atomicStore(&localBroken,1u);
}
fn cursorCount(base:u32,end:u32)->u32{
 if(localCursor&&base==localBase){
  let n=localCount(end);
  if(PLACE_CHECK){placeCheck(0u);if(n!=storedCount(base,end)){placeCheck(1u);}}
  return n;
 }
 return storedCount(base,end);
}
fn cursorAt(base:u32,ordinal:u32)->u32{
 var index=0u;
 if(localCursor&&base==localBase){
  index=localIndex(ordinal);
  if(PLACE_CHECK){placeCheck(0u);if(index!=storedIndex(base,ordinal)){placeCheck(2u);}}
 }else{index=storedIndex(base,ordinal);}
 // Fenwick coordinate is pen+1; rank-select's predecessor index is pen.
 return output[base+p.extent.x+1u+index];
}
fn cursorInsert(base:u32,pen:u32,address:u32){
 fitArmed=false;
 if(localCursor&&base==localBase&&pen<p.extent.x){
  if(localInsert(pen)){
   if(PLACE_CHECK){storedInsert(base,pen,address);}else{output[base+p.extent.x+1u+pen]=address;}
   return;
  }
  localLeave();
 }
 storedInsert(base,pen,address);
}
// Role words (part 0) and region tone passages (part 3) are whole words of
// the grammar table; every other part addresses one native glyph directly.
fn scalarFont(word:Word,scalar:u32)->u32{
 if(word.data[9]==0u||word.data[9]==3u){return words[word.data[5]*WS+2u+scalar+select(0u,word.data[13],word.data[9]==0u)];}return word.data[5];
}
// The interval index addresses INTACT SCALARS in the committed GPU tape.
// Different whole words may share one scalar without claiming one another's
// other strokes or silently deleting a construction's contribution.
var<workgroup> glyphTags:array<u32,128>;
var<workgroup> glyphValues:array<vec4u,128>;
fn storedGlyph(reference:u32)->vec4u{
 if(!paperLocal||reference==NONE){return storedGlyphRecord(reference);}
 let slot=(reference^(reference>>7u))&127u;
 if(glyphTags[slot]==reference){
  let v=glyphValues[slot];
  if(PLACE_CHECK){placeCheck(0u);if(any(v!=storedGlyphRecord(reference))){placeCheck(2u);}}
  return v;
 }
 let v=storedGlyphRecord(reference);glyphTags[slot]=reference;glyphValues[slot]=v;return v;
}
fn storedGlyphRecord(reference:u32)->vec4u{
 let at=(reference>>4u)*${WORD_WORDS}u;let scalar=reference&15u;
 var font=output[at+5u];var x=output[at+3u];
 if(output[at+9u]==0u||output[at+9u]==3u){
  let word=font;let first=select(0u,output[at+13u],output[at+9u]==0u);
  for(var i=0u;i<scalar;i++){let prior=words[word*WS+2u+first+i];x+=grammar[grammar[2]+prior*4u+1u];}
  font=words[word*WS+2u+first+scalar];
 }
 return vec4u(x,font,grammar[grammar[2]+font*4u+1u],output[at+10u]);
}
struct NativeSlot {left:u32,right:u32,sameGlyph:bool}
fn nativeSlotStored(cursor:u32,x:u32,advance:u32,font:u32,color:u32)->NativeSlot{
 let total=cursorCount(cursor,p.extent.x);let before=cursorCount(cursor,min(p.extent.x,x+advance));
 var previous=before;var sameGlyph=false;
 if(previous>0u){let prior=storedGlyph(cursorAt(cursor,previous));sameGlyph=prior.x==x&&prior.y==font&&prior.z==advance&&prior.w==color;if(sameGlyph){previous--;}}
 var left=0u;var right=p.extent.x;
 if(previous>0u){let prior=storedGlyph(cursorAt(cursor,previous));left=prior.x+prior.z;}
 if(before<total){right=storedGlyph(cursorAt(cursor,before+1u)).x;}
 return NativeSlot(left,right,sameGlyph);
}
fn nativeSlotLocal(cursor:u32,x:u32,advance:u32,font:u32,color:u32)->NativeSlot{
 // The scalar before the slot (the last pen < x+advance), the one before it
 // when that one is this same glyph, and the scalar after (the first pen >=).
 let e=min(p.extent.x,x+advance);var previous=cursorLast(cursor,e);var sameGlyph=false;
 var left=0u;var right=p.extent.x;
 if(previous!=NONE){let prior=storedGlyph(cursorAddr(cursor,previous));sameGlyph=prior.x==x&&prior.y==font&&prior.z==advance&&prior.w==color;
  if(sameGlyph){previous=cursorLast(cursor,previous);}else{left=prior.x+prior.z;}}
 if(sameGlyph&&previous!=NONE){let prior=storedGlyph(cursorAddr(cursor,previous));left=prior.x+prior.z;}
 let next=cursorFirst(cursor,e);if(next!=NONE){right=storedGlyph(cursorAddr(cursor,next)).x;}
 return NativeSlot(left,right,sameGlyph);
}
fn nativeSlot(cursor:u32,x:u32,advance:u32,font:u32,color:u32)->NativeSlot{
 if(localCursor&&cursor==localBase){
  let a=nativeSlotLocal(cursor,x,advance,font,color);
  if(PLACE_CHECK){placeCheck(0u);let b=nativeSlotStored(cursor,x,advance,font,color);if(a.left!=b.left||a.right!=b.right||a.sameGlyph!=b.sameGlyph){placeCheck(2u);}}
  return a;
 }
 return nativeSlotStored(cursor,x,advance,font,color);
}
fn intervalPen(left:u32,right:u32,wanted:u32,advance:u32)->u32{
 if(right<left+advance){return NONE;}
 let desired=select(0u,wanted-left,wanted>=left);
 let gap=boundedGap(desired,right-left-advance,left==0u);
 return select(left+gap,NONE,gap==NONE);
}
var<private> fitArmed:bool=false;
var<private> fitSame:u32=0u;
var<private> fitKey:vec4u=vec4u(0u);
var<private> fitKey2:vec2u=vec2u(0u);
fn tapeFits(word:Word,pen:u32,row:u32,cursor:u32)->bool{
 fitArmed=false;
 if(pen+word.data[7]>p.extent.x){return false;}
 var x=pen;var siblingEnd=0u;var same=0u;var reusable=paperLocal&&word.data[6]<=32u;
 for(var scalar=0u;scalar<word.data[6];scalar++){
  let font=scalarFont(word,scalar);let metric=grammar[2]+font*4u;let advance=grammar[metric+1u];
  if(grammar[metric+3u]==0u){x+=advance;continue;}
  let slot=nativeSlot(cursor,x,advance,font,word.data[10]);
  if(slot.sameGlyph){same|=1u<<min(scalar,31u);}if(advance==0u){reusable=false;}
  let left=max(slot.left,siblingEnd);var right=slot.right;var next=x+advance;
  for(var i=scalar+1u;i<word.data[6];i++){let other=grammar[2]+scalarFont(word,i)*4u;if(grammar[other+3u]!=0u){right=min(right,next);break;}next+=grammar[other+1u];}
  if(x<left||x+advance>right||nativeGap(x-left,left==0u)!=x-left){return false;}
  if(right<p.extent.x&&nativeGap(right-x-advance,false)!=right-x-advance){return false;}
  // Region tone's clearance from strokes is the producer's decision
  // (tone-field.ts: 11 px of paper inside a face's bounding lines); here a
  // tone word only needs a legal native gap, so its tiles may abut.
  siblingEnd=x+advance;x+=advance;
 }
 let fits=word.data[9]==2u||!paperWord(pen,row,word.data[5],word.data[6],word.data[9],word.data[13]);
 if(fits&&reusable){fitArmed=true;fitSame=same;fitKey=vec4u(pen,word.data[5],word.data[6],word.data[9]);fitKey2=vec2u(word.data[13],word.data[10]);}
 return fits;
}
fn commitWord(word:Word,row:u32,cursor:u32)->bool{
 let out=atomicAdd(&counts[rowCount()+2u],1u);
 if(out>=p.extent.w){atomicOr(&counts[rowCount()],2u);return true;}
 if(word.data[9]!=3u){
  let root=runs[word.data[0]];atomicAdd(&counts[drawnSlot(root)],1u);
  if(wordHasSpanSource(word)&&word.data[23]!=root.identity.z){atomicAdd(&counts[drawnBase()+min(word.data[23],p.playback.z-1u)],1u);}
 }
 for(var k=0u;k<${WORD_WORDS}u;k++){output[out*${WORD_WORDS}u+k]=word.data[k];}
 var x=word.data[3];
 let reuse=fitArmed&&all(fitKey==vec4u(word.data[3],word.data[5],word.data[6],word.data[9]))&&all(fitKey2==vec2u(word.data[13],word.data[10]));let same=fitSame;fitArmed=false;
 for(var scalar=0u;scalar<word.data[6];scalar++){
  let font=scalarFont(word,scalar);let metric=grammar[2]+font*4u;let advance=grammar[metric+1u];
  var held=false;
  if(grammar[metric+3u]!=0u){
   if(reuse){held=(same&(1u<<min(scalar,31u)))!=0u;if(PLACE_CHECK){placeCheck(0u);if(held!=nativeSlot(cursor,x,advance,font,word.data[10]).sameGlyph){placeCheck(2u);}}}
   else{held=nativeSlot(cursor,x,advance,font,word.data[10]).sameGlyph;}
  }
  if(grammar[metric+3u]!=0u&&!held){cursorInsert(cursor,x,(out<<4u)|scalar);}
  x+=advance;
 }
 return true;
}

// word gets one bounded candidate pair from the row's fixed priority order;
// admission confirms the nearest candidate, then the other existing interval
// only when the real paper or source-visibility check rejects the first. No
// score, font search, or unbounded pen walk is performed, and a collision
// never respells a word; the higher rank keeps the cell. GOAL-EXECUTOR D1
// here: D1(1) is the compiled gap lookup and D1(6) is the bounded terminal
// confirmation. D1(2),(3), respelling with a neighbouring phase or size
// record, were removed by that order. D1(5) belongs to construction dispatch,
// not to this stage.
fn wordSeen(word:Word,pen:u32,row:u32)->bool{
 if(word.data[12]==0u){return true;}
 let depth=bitcast<f32>(word.data[11]);
 var source=runs[word.data[0]];
 if(wordHasSpanSource(word)){source.identity=vec4u(source.identity.x,word.data[22],word.data[23],source.identity.w);}
 if(word.data[9]==2u){return visibleWord(source,i32(pen),i32(row),word.data[5],1u,1u,depth,0u);}
 return visibleWord(source,i32(pen),i32(row),word.data[5],word.data[6],word.data[9],depth,word.data[13]);
}
// Offsets of the word's first and last ink from its pen; y==0 means no ink.
fn inkSpan(word:Word)->vec2u{
 var x=0u;var lead=NONE;var end=0u;
 for(var scalar=0u;scalar<word.data[6];scalar++){
  let metric=grammar[2]+scalarFont(word,scalar)*4u;let advance=grammar[metric+1u];
  if(grammar[metric+3u]!=0u){if(lead==NONE){lead=x;}end=x+advance;}
  x+=advance;
 }
 if(lead==NONE){return vec2u(0u,0u);}
 return vec2u(lead,end);
}
// The free span holding the point, or the two free spans beside the committed
// scalar that covers it. xy and zw are [left,right); x==NONE marks no span.
fn freeSpansStored(cursor:u32,point:u32)->vec4u{
 let total=cursorCount(cursor,p.extent.x);let ordinal=cursorCount(cursor,min(p.extent.x,point+1u));
 if(ordinal==0u){
  var right=p.extent.x;if(total>0u){right=storedGlyph(cursorAt(cursor,1u)).x;}
  return vec4u(0u,right,NONE,0u);
 }
 let prior=storedGlyph(cursorAt(cursor,ordinal));
 var next=p.extent.x;if(ordinal<total){next=storedGlyph(cursorAt(cursor,ordinal+1u)).x;}
 if(prior.x+prior.z<=point){return vec4u(prior.x+prior.z,next,NONE,0u);}
 var before=0u;if(ordinal>1u){let earlier=storedGlyph(cursorAt(cursor,ordinal-1u));before=earlier.x+earlier.z;}
 return vec4u(before,prior.x,prior.x+prior.z,next);
}
fn freeSpansLocal(cursor:u32,point:u32)->vec4u{
 let e=min(p.extent.x,point+1u);let last=cursorLast(cursor,e);
 if(last==NONE){
  var right=p.extent.x;let first=cursorFirst(cursor,0u);if(first!=NONE){right=storedGlyph(cursorAddr(cursor,first)).x;}
  return vec4u(0u,right,NONE,0u);
 }
 let prior=storedGlyph(cursorAddr(cursor,last));
 var next=p.extent.x;let after=cursorFirst(cursor,e);if(after!=NONE){next=storedGlyph(cursorAddr(cursor,after)).x;}
 if(prior.x+prior.z<=point){return vec4u(prior.x+prior.z,next,NONE,0u);}
 var before=0u;let earlier=cursorLast(cursor,last);if(earlier!=NONE){let glyph=storedGlyph(cursorAddr(cursor,earlier));before=glyph.x+glyph.z;}
 return vec4u(before,prior.x,prior.x+prior.z,next);
}
fn freeSpans(cursor:u32,point:u32)->vec4u{
 if(localCursor&&cursor==localBase){
  let a=freeSpansLocal(cursor,point);
  if(PLACE_CHECK){placeCheck(0u);let b=freeSpansStored(cursor,point);if(any(a!=b)){placeCheck(2u);}}
  return a;
 }
 return freeSpansStored(cursor,point);
}
fn spanFits(span:vec2u,a:u32,b:u32)->bool{
 if(span.x==NONE||a<span.x||b>span.y){return false;}
 if(nativeGap(a-span.x,span.x==0u)!=a-span.x){return false;}
 return span.y>=p.extent.x||nativeGap(span.y-b,false)==span.y-b;
}
// A line word may move up to half its advance, and at least 6 px: spaces
// spell 0, 5, 11, 16, 21 ... px (skill 15.4), so a word whose neighbour
// leaves 1-4, 6-10 or 12-15 px needs up to 3 px on one side, more when its
// left and right gaps must both be legal. With 2 px a 4 px | could not reach
// one (loss split 2026-11-12, spot-frames: 317 of 1389 losses were a free
// span with an unspellable gap). Tone stays on its lattice node (2 px).
fn penReach(word:Word)->u32{
 if(word.data[9]==3u){return 2u;}
 return clamp(word.data[7]/2u,6u,48u);
}
// D1(1), one table lookup, no pen walk. The free span holding the word's ink
// centre yields a bounded candidate: the declared ink start projected into
// that span through the compiled two-sided gap table (boundedGap: nearest
// left gap whose left and right gaps are both spellable, resolved once at
// residency; distance zero first, so a legal declared pen is returned
// unchanged). A span that ends at the page edge has no right-hand gap to spell
// (the table's all-legal tail), as in tapeFits. If a committed scalar covers
// the centre, freeSpans supplies the two adjacent legal intervals. The pair is
// ordered by distance from the declared pen, left on a tie; admission below
// confirms each at most once with tapeFits and wordSeen. A pen further than
// reach is no pen.
fn tablePenSpan(word:Word,wanted:u32,span:vec2u)->u32{
 if(span.x==NONE||span.y<span.x){return NONE;}
 let ink=inkSpan(word);if(ink.y==0u){return wanted;}
 var right=span.y;if(right>=p.extent.x){right=p.extent.x+grammar[5]*4u;}
 let left=intervalPen(span.x,right,wanted+ink.x,ink.y-ink.x);
 if(left==NONE||left<ink.x){return NONE;}
 let pen=left-ink.x;
 return select(pen,NONE,abs(i32(pen)-i32(wanted))>i32(penReach(word)));
}
fn penDistance(pen:u32,wanted:u32)->u32{
 if(pen==NONE){return NONE;}
 return u32(abs(i32(pen)-i32(wanted)));
}
fn tablePenCandidates(word:Word,cursor:u32)->vec2u{
 let wanted=word.data[3];let ink=inkSpan(word);if(ink.y==0u){return vec2u(wanted,NONE);}
 let spans=freeSpans(cursor,min(p.extent.x,wanted+(ink.x+ink.y)/2u));
 if(spans.z!=NONE){
  let left=tablePenSpan(word,wanted,spans.xy);let right=tablePenSpan(word,wanted,spans.zw);
  if(left==NONE){return vec2u(right,NONE);}if(right==NONE){return vec2u(left,NONE);}
  return select(vec2u(left,right),vec2u(right,left),penDistance(right,wanted)<penDistance(left,wanted));
 }
 return vec2u(tablePenSpan(word,wanted,spans.xy),NONE);
}
fn tablePen(word:Word,cursor:u32)->u32{
 return tablePenCandidates(word,cursor).x;
}
// The one confirmation at that pen: tape (paper mask, final gap legality)
// and visibility. The producer already cut a whole word at its declared pen
// against visibility; a displaced word and every clipped piece are checked.
fn setWord(word:Word,pen:u32,row:u32,cursor:u32,cut:bool)->bool{
 if(pen==NONE||!tapeFits(word,pen,row,cursor)){return false;}
 if((cut||pen!=word.data[3])&&!wordSeen(word,pen,row)){return false;}
 var ready=word;ready.data[3]=pen;return commitWord(ready,row,cursor);
}
// D1 admission is bounded to the two existing intervals from the native
// cursor. Candidate order is nearest legal pen, left on a tie; a candidate
// that fails the real paper or source-visibility check does not reject the
// unit while the other existing interval remains legal.
fn setTableWord(word:Word,row:u32,cursor:u32,cut:bool)->bool{
 let candidates=tablePenCandidates(word,cursor);
 if(candidates.x!=NONE&&setWord(word,candidates.x,row,cursor,cut)){return true;}
 if(candidates.y!=NONE&&candidates.y!=candidates.x&&setWord(word,candidates.y,row,cursor,cut)){return true;}
 return false;
}
fn duplicateWord(word:Word,cursor:u32)->bool{
 if(word.data[9]!=0u){return false;}
 let centre=min(p.extent.x,word.data[3]+word.data[7]/2u);
 let last=cursorLast(cursor,min(p.extent.x,centre+1u));if(last==NONE){return false;}
 let reference=cursorAddr(cursor,last);let glyph=storedGlyph(reference);
 if(glyph.x+glyph.z<=centre){return false;}
 let overlap=min(word.data[3]+word.data[7],glyph.x+glyph.z)-max(word.data[3],glyph.x);
 if(2u*overlap<word.data[7]){return false;}
 let at=(reference>>4u)*${WORD_WORDS}u;
 if(output[at+9u]!=0u&&output[at+9u]!=4u){return false;}
 return output[at+8u]>=word.data[8]&&((runs[output[at]].range.z^runs[word.data[0]].range.z)&0xffffu)==0u;
}
fn setPiece(original:Word,first:u32,count:u32,x:u32,advance:u32,row:u32,cursor:u32)->bool{
 var piece=original;piece.data[3]=x;piece.data[6]=count;piece.data[7]=advance;piece.data[13]=original.data[13]+first;
 return setTableWord(piece,row,cursor,true);
}
// D1(4),(6) as ONE left-to-right walk over the word's own scalars at their
// declared positions, taken immediately in priority order; no piece lengths
// are tried and nothing is retried. Each inked scalar reads the free span at
// its own ink centre. Consecutive scalars that lie whole inside the same free
// span stay one piece at their declared spacing, so a clipped word still
// reads as a run of whole glyphs; a scalar that overhangs its span or whose
// centre is covered is a piece of one. Each piece gets the same bounded lookup
// and admission as a whole word (tablePenCandidates, setTableWord); a piece
// that has no legal existing interval is left out.
fn clipWord(original:Word,row:u32,cursor:u32)->bool{
 let count=min(original.data[6],16u);var drawn=false;var x=original.data[3];
 var first=NONE;var key=0u;var start=0u;var taken=0u;var width=0u;var blank=0u;var blankAdvance=0u;
 for(var scalar=0u;scalar<count;scalar++){
  let metric=grammar[2]+scalarFont(original,scalar)*4u;let advance=grammar[metric+1u];
  if(grammar[metric+3u]==0u){blank++;blankAdvance+=advance;x+=advance;continue;}
  let spans=freeSpans(cursor,min(p.extent.x,x+advance/2u));
  let inPlace=spans.z==NONE&&x>=spans.x&&x+advance<=spans.y;
  if(first!=NONE&&inPlace&&spans.x==key){taken+=blank+1u;width+=blankAdvance+advance;}
  else{
   if(first!=NONE){if(setPiece(original,first,taken,start,width,row,cursor)){drawn=true;}first=NONE;}
   if(inPlace){first=scalar;key=spans.x;start=x;taken=1u;width=advance;}
   else if(setPiece(original,scalar,1u,x,advance,row,cursor)){drawn=true;}
  }
  blank=0u;blankAdvance=0u;x+=advance;
 }
 if(first!=NONE){if(setPiece(original,first,taken,start,width,row,cursor)){drawn=true;}}
 return drawn;
}
// The committed glyph nearest before page x (its x, font, advance, part), or
// x == NONE when the row holds none before x.
fn glyphBefore(cursor:u32,x:u32)->vec4u{
 let last=cursorLast(cursor,min(p.extent.x,x));if(last==NONE){return vec4u(NONE,0u,0u,0u);}
 let reference=cursorAddr(cursor,last);let glyph=storedGlyph(reference);
 return vec4u(glyph.x,glyph.y,glyph.z,output[(reference>>4u)*${WORD_WORDS}u+9u]);
}
// One half space (5 px) is the narrowest spellable gap (skill 15.4); a tone
// word nearer than that to a stroke, or under one, is cleared around it.
fn toneCleared(word:Word,cursor:u32)->bool{
 var x=word.data[3];
 for(var scalar=0u;scalar<word.data[6];scalar++){
  let metric=grammar[2]+scalarFont(word,scalar)*4u;let advance=grammar[metric+1u];
  if(grammar[metric+3u]!=0u){
   let left=glyphBefore(cursor,x+advance+5u);
   if(left.x!=NONE&&left.w!=3u&&left.x+left.z+5u>x){return true;}
  }
  x+=advance;
 }
 return false;
}
fn sharedStamp(word:Word,cursor:u32)->bool{
 let centre=min(p.extent.x,word.data[3]+word.data[7]/2u);
 let last=cursorLast(cursor,min(p.extent.x,centre+1u));if(last==NONE){return false;}
 let reference=cursorAddr(cursor,last);let glyph=storedGlyph(reference);let at=(reference>>4u)*${WORD_WORDS}u;
 return glyph.x+glyph.z>centre&&output[at+1u]==STAMP_ORDINAL&&glyph.y==scalarFont(word,0u);
}
// Returns false only for a word of which no scalar was committed and that is
// neither hidden, shared nor a duplicate: the caller counts exactly those.
// ONE TERRAIN EDGE, ONE STROKE. A height surface's edge often arrives as
// several runs of one owner (its LOD pieces and overlapping chunks: ten runs
// of the view 06 cliff's right edge, crop-review-20261110 crop-2 A). Each
// spelled the same unit in the same row, and placement set them side by side
// 3 px apart: "|||". A terrain word whose committed neighbour in the row is
// the same word of another run of the same owner, within one advance of it,
// is that stroke: it is shared, not placed again.
// ONE EDGE, ONE STROKE, WHICHEVER SURFACES REPORT IT (2026-11-11): the edge
// two surfaces share arrives once per surface. In the fog room the wall's
// silhouette (run 0, owner 1/7, class 0) and the floor's contact (run 33,
// owner 1/1, class 2) both spelled the corner as "＼､" 24 px apart in every

// owner test is gone, and a copy is any same word of another run in the same
// row within one and a half advances: parallels inside one glyph width
// collapse (AGENTS.md finding 4).
fn straightUprightGlyph(font:u32)->bool{let c=grammar[grammar[2]+font*4u];return c==0x7cu||c==0xff5cu||c==0x6cu||c==0xff4cu||c==0x21u;}
fn straightUprightWord(word:u32)->bool{
 let n=words[word*WS];var inked=0u;
 for(var i=0u;i<n;i++){let font=words[word*WS+2u+i];if(grammar[grammar[2]+font*4u+3u]==0u){continue;}inked++;if(!straightUprightGlyph(font)){return false;}}
 return inked>0u;
}
fn terrainCopy(word:Word,cursor:u32)->bool{
 // Every edge, hard or not: 12 hard runs of one ledge (flags 0xc040200,

 // crop-review-20261110 crop-2 J) and the hard contraction below, which
 // compares whole-run endpoint directions, missed them.
 if(word.data[9]!=0u){return false;}
 let wanted=word.data[3];let q=min(p.extent.x,wanted+word.data[7]/2u);
 for(var side=0u;side<2u;side++){
  var pen=NONE;if(side==0u){pen=cursorLast(cursor,q);}else{pen=cursorFirst(cursor,q);}if(pen==NONE){continue;}
  let reference=cursorAddr(cursor,pen);let at=(reference>>4u)*${WORD_WORDS}u;
  if(output[at]==word.data[0]||output[at+9u]!=0u){continue;}
  let glyph=storedGlyph(reference);

  // different runs with at most a half space (5 px, the narrowest spellable gap)
  // of paper between them are one stroke (AGENTS finding 4), whatever word each

  if(output[at+5u]!=word.data[5]){
   if(!straightUprightWord(word.data[5])||!straightUprightGlyph(glyph.y)){continue;}
   let gap=select(i32(wanted)-i32(glyph.x+glyph.z),i32(glyph.x)-i32(wanted+word.data[7]),glyph.x>=wanted);
   if(gap<=5){return true;}
   continue;
  }
  // A terrain line's parallels merge inside one full-width cell (16 px,
  // AGENTS finding 4: parallel creases inside one glyph width are one), not
  // one and a half of their own advance: a 3 px l merged only folds 4 px
  // apart, and a cliff face of near-vertical folds stood as l||| clusters
  // (combined-g view 06). Built edges keep the advance rule (a post's or a
  // trunk's two sides can lie closer than a cell).
  // Built edges merge inside 10 px: two runs' uprights 9 px apart drew as a
  // doubled || at one jamb (junction-fit-b view 02); a post narrower than
  // that is one stroke.
  // A whole passage's advance is its LENGTH, not the width of one native
  // cell. Using 1.5 * passage advance merged separate rails 50px apart
  // when both read _,.-‐´ (35px), although the largest scalar is only 8px.
  // The same contraction rule keeps its native-cell footprint when a
  // scalar unit becomes a multi-glyph construction. No glyph is re-chosen.
  var cell=0u;
  for(var scalar=0u;scalar<word.data[6];scalar++){
   cell=max(cell,grammar[grammar[2]+scalarFont(word,scalar)*4u+1u]);
  }
  let reach=max(max(cell,glyph.z)*3u/2u,select(10u,16u,terrainRun(runs[word.data[0]])));
  if(abs(i32(glyph.x)-i32(wanted))<=i32(reach)){return true;}
 }
 return false;
}
// A single glyph whose cell already holds the same glyph is drawn by it.
fn sameGlyphShared(word:Word,cursor:u32)->bool{
 if(word.data[6]!=1u){return false;}
 let ink=inkSpan(word);if(ink.y==0u){return false;}
 let centre=min(p.extent.x,word.data[3]+(ink.x+ink.y)/2u);
 let last=cursorLast(cursor,min(p.extent.x,centre+1u));if(last==NONE){return false;}
 let glyph=storedGlyph(cursorAddr(cursor,last));
 return glyph.x+glyph.z>centre&&glyph.y==scalarFont(word,0u)&&abs(i32(glyph.x)-i32(word.data[3]))*2<=i32(glyph.z);
}
// An anchor of several glyphs keeps its authored scalar order and source
// correspondence, but a collision is still a per-scalar admission failure,
// not a reason to reject the whole native part. Each scalar first keeps its
// declared pen; setTableWord then takes the same bounded existing-interval
// alternative used by the ordinary path when paper or source visibility
// rejects that pen. No glyph is respelled and no source span is re-derived.
fn clipDeclared(original:Word,row:u32,cursor:u32)->bool{
 let count=min(original.data[6],16u);var drawn=false;var x=original.data[3];
 for(var scalar=0u;scalar<count;scalar++){
  let metric=grammar[2]+scalarFont(original,scalar)*4u;let advance=grammar[metric+1u];
  if(grammar[metric+3u]!=0u){
   var piece=original;piece.data[3]=x;piece.data[6]=1u;piece.data[7]=advance;piece.data[13]=original.data[13]+scalar;
   if(setTableWord(piece,row,cursor,true)){drawn=true;}
  }
  x+=advance;
 }
 return drawn;
}
fn typesetWord(original:Word,row:u32,cursor:u32)->bool{
 let placed=typesetUnit(original,row,cursor);
 if(!placed&&original.data[9]!=3u&&original.data[9]!=2u){atomicAdd(&counts[featureRecord(original.data[0])],1u);}
 return placed;
}
// Contraction may preserve a requested native scalar, never choose another
// word's vocabulary or paint. A construction that wants a shared junction
// must request that compiled entry itself before reaching native placement.
fn sharesRequestedScalar(word:Word,glyph:vec4u)->bool{
 return word.data[6]==1u&&scalarFont(word,0u)==glyph.y&&word.data[10]==glyph.w;
}
fn typesetUnit(original:Word,row:u32,cursor:u32)->bool{
 // A region tone word on a cell covered by a camera stacked above is not placed
 // (rows over TONE_CAP words keep tone in this serial path; toneplace does the same).
 if(original.data[9]==3u&&toneOccluded(original,original.data[3],row)){return true;}
 var word=original;let wanted=word.data[3];let advance=word.data[7];
 if(terrainCopy(word,cursor)){atomicAdd(&counts[featureRecord(word.data[0])+1u],1u);return true;}
 if(word.data[9]==0u&&word.data[6]==1u&&(runs[word.data[0]].range.z&0x40000u)!=0u){
  // Fixed-rank geometry contraction: two parallel same-owner edges within
  // one native unit share that unit, not two displaced columns. Only the
  // two adjacent committed intervals can overlap this word's footprint.
  let run=runs[word.data[0]];let a=finiteRunStart(run).position.xy;let b=finiteRunEnd(run).position.xy;
  let delta=b-a;let span=length(delta);let total=cursorCount(cursor,p.extent.x);
  let before=cursorCount(cursor,min(p.extent.x,wanted+advance));
  for(var side=0u;side<2u;side++){
   let ordinal=select(before,before+1u,side==1u);if(ordinal==0u||ordinal>total){continue;}
   let reference=cursorAt(cursor,ordinal);let address=(reference>>4u)*${WORD_WORDS}u;
   // Adjacent units of THIS run extend its line; they are not duplicate
   // edges. Contracting them erased every other horizontal repeat unit.
   if(output[address]==word.data[0]){continue;}
   let prior=runs[output[address]];
   if((prior.range.z&0x40000u)==0u||!all(prior.identity.yz==run.identity.yz)||prior.range.y<2u){continue;}
   let c=finiteRunStart(prior).position.xy;let d=finiteRunEnd(prior).position.xy;let other=d-c;
   if(span<.00001||length(other)<.00001){continue;}
   let uprightPair=(run.range.z&prior.range.z&0x80000u)!=0u;
   if(abs(dot(normalize(delta),normalize(other)))<select(.9999,.99,uprightPair)){continue;}
   let axis=delta/span;let ca=dot(c-a,axis);let da=dot(d-a,axis);
   if(min(span,max(ca,da))<=max(0.0,min(ca,da))){continue;}
   // Perspective-parallel world-up edges may converge slightly. BOTH ends
   // must stay within one cell's normal footprint: a crossing or another
   // wide box face is not a duplicate. Glyph advance alone is only 4px for
   // |; it cannot define the width of the containing text column.
   let separation=max(abs((c.x-a.x)*delta.y-(c.y-a.y)*delta.x),abs((d.x-a.x)*delta.y-(d.y-a.y)*delta.x))/span;
   let unitWidth=length(vec2f(delta.y*11.0,delta.x*17.0))/span;
   let existing=storedGlyph(reference);if(separation>unitWidth){continue;}
   // Keep same-scalar contraction and its original source contributions.
   // Different requested ink takes the unchanged native spacing/clip path
   // below; proximity is not permission to borrow the neighbour's glyph.
   if(!sharesRequestedScalar(word,existing)){continue;}
   // Preserve both source contributions to the one intact shared scalar.
   word.data[3]=existing.x;word.data[5]=scalarFont(word,0u);word.data[6]=1u;word.data[7]=existing.z;word.data[9]=4u;
   return commitWord(word,row,cursor);
  }
 }
 if(word.data[9]==2u&&word.data[12]!=0u&&!visibleWord(runs[word.data[0]],i32(wanted),i32(row),word.data[5],1u,1u,bitcast<f32>(word.data[11]),0u)){return true;}
 // Anchors keep their declared relationship: the declared pen is their only
 // pen, and they are never clipped. A whole-form stamp is not a join: it is
 // the form's one glyph and takes D1(1), the one table pen inside its reach.
 let stamp=word.data[9]==0u&&word.data[1]==STAMP_ORDINAL;
 let anchor=!stamp&&((word.data[8]&0x80000000u)!=0u||(word.data[9]!=2u&&word.data[9]!=3u&&(runs[word.data[0]].range.z&0x10000u)!=0u));
 var pen=wanted;
 if(anchor){if(setWord(word,pen,row,cursor,false)){return true;}}
 else if(setTableWord(word,row,cursor,false)){return true;}
 // D6: region tone is cleared around the strokes that cross it. A tone word
 // whose ink a committed stroke covers, or that lies closer to one than a
 // half space, is that clearing, not a lost word.
 if(word.data[9]==3u){return toneCleared(word,cursor);}
 // Two small forms whose stamps fall on the same cells: the committed stamp of
 // the same word is the mark of both (the cell cannot hold two), not a loss.
 if(stamp){return sharedStamp(word,cursor);}
 // Only an authored label glyph takes its table pen; a join keeps its declared
 // pen (moved joins landed beside every crossing of a level grid line and
 // refused that line's own units their gaps: post-processing f0, lines-5).
 if(anchor){if(word.data[9]==0u&&word.data[6]>=2u){return clipDeclared(word,row,cursor);}if(word.data[9]!=2u){return false;}return setTableWord(word,row,cursor,false);}
 // The same cells already carry a higher-rank word of the same role and
 // line class: a duplicate, neither shoved sideways nor counted as a loss.
 if(duplicateWord(word,cursor)){atomicAdd(&counts[featureRecord(word.data[0])+1u],1u);return true;}
// Region tone keeps its +-2 px pen and is never clipped; a single scalar
// has nothing shorter to clip to. A compiled whole-part word may contain
// several repeated native scalars. If its group pen is rejected, split only
// that word at the scalar boundary and admit each member through the same
// bounded interval path; a legal member must not be lost with its neighbour.
 if(word.data[9]!=0u||word.data[6]<2u){
  // One glyph: the same glyph already on its cell draws it (a copy, patch_d1b).
  if(word.data[6]==1u&&word.data[9]!=3u&&word.data[9]!=2u&&sameGlyphShared(word,cursor)){atomicAdd(&counts[featureRecord(word.data[0])+1u],1u);return true;}
  if(word.data[9]==1u&&word.data[6]>=2u){return clipWord(word,row,cursor);}
  return false;
 }
 return clipWord(word,row,cursor);
}
// PROBE-BEGIN temporary loss instrumentation. Off unless the page URL carries
// ?mainProbes=1 (PROBES, a compile-time constant the shader compiler folds
// away): the per-word atomics and the loss re-checks ran in featureplace's
// one placing lane every frame (gpu-profile-20261112c).
const PROBES=${MAIN_PROBES};
fn probeAdd(k:u32){if(PROBES){atomicAdd(&counts[rowCount()+4u+k],1u);}}
fn probeClass(run:Run)->u32{
 if(terrainRun(run)){return 0u;}
 let b=wholeBoundaryFlags(run);
 if((b&0x100000u)!=0u){return 1u;}
 if((b&0x4000000u)!=0u){return 2u;}
 if((b&0x20000u)!=0u){return 3u;}
 if((run.range.z&0x10000u)!=0u){return 4u;}
 if((run.range.z&0x40000u)!=0u){return 5u;}
 return 6u;
}
fn probeLoss(word:Word,row:u32,cursor:u32){
 if(!PROBES){return;}
 let part=word.data[9];
 if(part==3u){probeAdd(3u);return;}
 if(part==2u){probeAdd(4u);return;}
 let run=runs[word.data[0]];
 probeAdd(18u+probeClass(run));
 if(terrainRun(run)){probeAdd(select(51u,50u,distanceTier(run)==2u));}
 let anchor=(word.data[8]&0x80000000u)!=0u||(run.range.z&0x10000u)!=0u;
 if(anchor){probeAdd(5u);}
 let ink=inkSpan(word);if(ink.y==0u){probeAdd(17u);return;}
 let centre=min(p.extent.x,word.data[3]+(ink.x+ink.y)/2u);
 var mine=0u;var mineX=word.data[3];var x=word.data[3];
 for(var s=0u;s<word.data[6];s++){let f=scalarFont(word,s);let a=grammar[grammar[2]+f*4u+1u];if(grammar[grammar[2]+f*4u+3u]!=0u&&(mine==0u||x<=centre)){mine=f;mineX=x;}x+=a;}
 let spans=freeSpans(cursor,centre);
 if(spans.z!=NONE){
  let ordinal=cursorCount(cursor,min(p.extent.x,centre+1u));
  let reference=cursorAt(cursor,ordinal);let at=(reference>>4u)*${WORD_WORDS}u;let glyph=storedGlyph(reference);
  let bpart=output[at+9u];
  if(bpart==3u){probeAdd(12u);return;}
  if(bpart==2u){probeAdd(13u);return;}
  let other=runs[output[at]];
  probeAdd(25u+probeClass(other));
  let same=glyph.y==mine&&abs(i32(glyph.x)-i32(mineX))*2<=i32(glyph.z);
  if(output[at]==word.data[0]){probeAdd(select(7u,6u,same));return;}
  if(all(other.identity.yz==run.identity.yz)){probeAdd(select(9u,8u,same));return;}
  probeAdd(select(11u,10u,same));return;
 }
 var overlap=false;x=word.data[3];
 for(var s=0u;s<word.data[6];s++){let f=scalarFont(word,s);let a=grammar[grammar[2]+f*4u+1u];
  if(grammar[grammar[2]+f*4u+3u]!=0u){let slot=nativeSlot(cursor,x,a,f,word.data[10]);if(x<slot.left||x+a>slot.right){overlap=true;}}
  x+=a;}
 if(overlap){probeAdd(14u);return;}
 if(paperWord(word.data[3],row,word.data[5],word.data[6],word.data[9],word.data[13])){probeAdd(15u);return;}
 if(!tapeFits(word,word.data[3],row,cursor)){probeAdd(16u);return;}
 probeAdd(17u);
}
// PROBE-END
// PARALLEL REGION TONE (frame cost, 2026-11-13). Region tone words (part 3)
// are lattice-placed and certified clear of the construction by the producer
// (tone-field.ts unitClear); they are not typeset in the serial row loop. After
// the row's other words are committed, each tone word is tested on its own
// against the committed non-tone glyphs (the cursor, read-only here): the same
// table pen, tape fit and visibility as before, then the producer's half-space
// clearance. Two kept neighbours that overlap or leave an unspellable gap keep
// the earlier by (pen, lattice key). The kept words are committed by lane 0
// with commitWord, as the serial loop did: the displayed tape is read from the
// row's cursor, so every committed scalar must enter it.
const TONE_CAP=1024u;
// A region tone word whose ink touches a cell covered by a stacked camera above.
fn toneOccluded(word:Word,pen:u32,row:u32)->bool{
 var x=pen;
 for(var scalar=0u;scalar<word.data[6];scalar++){
  let metric=grammar[2]+scalarFont(word,scalar)*4u;let advance=grammar[metric+1u];
  if(grammar[metric+3u]!=0u&&(occludedCell(i32(x),i32(row))||occludedCell(i32(x+advance/2u),i32(row))||occludedCell(i32(x+max(advance,1u)-1u),i32(row)))){return true;}
  x+=advance;
 }
 return false;
}
// commitWord for a tone word whose cells are known free (no glyph of the row
// overlaps it, checked above): the same tape record and cursor inserts, without
// the shared-glyph lookups per scalar (a shared glyph needs an overlap).
fn commitToneWord(word:Word,cursor:u32){
 let out=atomicAdd(&counts[rowCount()+2u],1u);
 if(out>=p.extent.w){atomicOr(&counts[rowCount()],2u);return;}
 for(var k=0u;k<${WORD_WORDS}u;k++){output[out*${WORD_WORDS}u+k]=word.data[k];}
 var x=word.data[3];
 for(var scalar=0u;scalar<word.data[6];scalar++){
  let font=scalarFont(word,scalar);let metric=grammar[2]+font*4u;let advance=grammar[metric+1u];
  if(grammar[metric+3u]!=0u){cursorInsert(cursor,x,(out<<4u)|scalar);}
  x+=advance;
 }
}
var<workgroup> tonePens:array<u32,1024>;
var<workgroup> toneOrder:array<u32,1024>;
var<workgroup> toneCount:u32;
fn toneTablePen(word:Word,row:u32,cursor:u32)->u32{
 let candidates=tablePenCandidates(word,cursor);var pen=candidates.x;
 if(pen!=NONE&&!toneOccluded(word,pen,row)){
  var fits=tapeFits(word,pen,row,cursor);if(fits&&pen!=word.data[3]){fits=wordSeen(word,pen,row);}
  if(fits){return pen;}
 }
 pen=candidates.y;
 if(pen!=NONE&&!toneOccluded(word,pen,row)){
  var fits=tapeFits(word,pen,row,cursor);if(fits&&pen!=word.data[3]){fits=wordSeen(word,pen,row);}
  if(fits){return pen;}
 }
 return NONE;
}
// Pen, then the lattice key (row words 14..17), then the row slot: a total
// order, independent of thread timing. NONE (not a kept tone word) sorts last.
fn toneFirst(a:u32,b:u32)->bool{
 if(a==NONE){return false;}if(b==NONE){return true;}
 if(tonePens[a]!=tonePens[b]){return tonePens[a]<tonePens[b];}
 let x=rowWords[rowBase(toneRow)+a].data;let y=rowWords[rowBase(toneRow)+b].data;
 for(var k=14u;k<18u;k++){if(x[k]!=y[k]){return x[k]<y[k];}}
 return a<b;
}
var<workgroup> toneRow:u32;
@compute @workgroup_size(64) fn toneplace(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let row=group.x;if(row>=rowCount()){return;}
 // A row over TONE_CAP words keeps its tone in the serial loop.
 if(lane==0u){let all=min(atomicLoad(&counts[row]),rowLimit(row));toneCount=select(all,0u,all>TONE_CAP);toneRow=row;}
 let count=workgroupUniformLoad(&toneCount);if(count==0u){return;}
 let start=rowBase(row);let cursor=cursorBase(row);
 for(var i=lane;i<count;i+=64u){
  tonePens[i]=NONE;let raw=rowWords[start+i];if(raw.data[9]!=3u||!typedQueueActive(raw.data[0],3u)){continue;}
  let word=expandWord(raw);let pen=toneTablePen(word,row,cursor);
  var fits=pen!=NONE;
  if(!fits){if(!toneCleared(word,cursor)){atomicAdd(&counts[rowCount()+1u],1u);probeAdd(3u);}continue;}
  let left=glyphBefore(cursor,min(p.extent.x,pen+word.data[7]+5u));
  if(left.x!=NONE&&left.w!=3u&&left.x+left.z+5u>pen){continue;}
  tonePens[i]=pen;
 }
 workgroupBarrier();
 // 2: the kept words in (pen, key) order (bitonic, this row's power of two).
 var span=1u;while(span<count){span*=2u;}
 for(var i=lane;i<span;i+=64u){toneOrder[i]=select(NONE,i,i<count&&tonePens[min(i,count-1u)]!=NONE);}
 workgroupBarrier();
 for(var k=2u;k<=span;k*=2u){for(var j=k/2u;j>0u;j/=2u){
  for(var i=lane;i<span;i+=64u){let other=i^j;if(other>i){let a=toneOrder[i];let b=toneOrder[other];if(toneFirst(b,a)==((i&k)==0u)){toneOrder[i]=b;toneOrder[other]=a;}}}
  workgroupBarrier();
 }}
 // 3: left to right, as the serial typesetter met them: a word overlapping
 // the last kept tone word, or leaving an unspellable gap to it with no
 // committed glyph between, is dropped; a kept word is committed (cursor
 // insert). A glyph found between two tone words is a non-tone glyph: the
 // previous tone word's scalars all start before its ink end.
 if(lane!=0u){return;}
 // Ends and gaps are INK ends, as the tape measures them (a passage's blank
 // scalars are not glyphs: its first ink may follow leading blanks).
 var lastEnd=0u;var any=false;
 for(var n=0u;n<count;n++){
  let i=toneOrder[n];if(i==NONE){break;}
  let word=expandWord(rowWords[start+i]);let ink=inkSpan(word);if(ink.y==0u){continue;}
  let pen=tonePens[i];let first=pen+ink.x;let end=pen+ink.y;
  if(any){
   if(lastEnd>first){continue;}
   let between=glyphBefore(cursor,min(p.extent.x,first));
   let separated=between.x!=NONE&&between.x>=lastEnd;
   if(!separated&&nativeGap(first-lastEnd,false)!=first-lastEnd){continue;}
  }
  var placed=word;placed.data[3]=pen;commitToneWord(placed,cursor);lastEnd=end;any=true;
 }
}
@compute @workgroup_size(128) fn featureplace(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let row=group.x;if(row>=rowCount()){return;}
 // Broadcast the count through workgroup memory so all barrier-bearing
 // sort loops have uniform bounds. Empty rows do no sort/placement work.
 if(lane==0u){itemCount=min(atomicLoad(&counts[row]),rowLimit(row));}
 let count=workgroupUniformLoad(&itemCount);if(count==0u){return;}
 let start=rowBase(row);
 if(count<=${ROW_CAP}u){
 var span=1u;while(span<count){span*=2u;}
 // Bitonic padding ends at THIS row's next power of two, not at the
 // allocation capacity. Priority, source identity and ordinal ties are
 // unchanged; no word is removed or respelled by this bound.
 for(var i=lane;i<span;i+=128u){order[i]=select(NONE,start+i,i<count);}
 workgroupBarrier();
 for(var k=2u;k<=span;k*=2u){for(var j=k/2u;j>0u;j/=2u){
  for(var i=lane;i<span;i+=128u){let other=i^j;if(other>i){let a=order[i];let b=order[other];if(before(b,a)==((i&k)==0u)){order[i]=b;order[other]=a;}}}
  workgroupBarrier();
 }}
 }
 let lows=grammar[grammar[9]+180u];let highs=grammar[grammar[9]+181u];
 for(var w=lane;w<${ROW_CAP/32}u;w+=128u){atomicStore(&twinBits[w],0u);}
 workgroupBarrier();
 if(count<=${ROW_CAP}u){for(var i=lane;i<count;i+=128u){if(levelTwinScan(start+i,row,lows,highs)){atomicOr(&twinBits[i/32u],1u<<(i%32u));}}}
 workgroupBarrier();
 let cursor=cursorBase(row);

 // of the occupancy bitmask from the stored Fenwick (a pen's count is its node
 // minus its children), then the word popcount Fenwick.
 let narrow=p.extent.x<=LOCAL_WORDS*32u;
 for(var w=lane;w<LOCAL_WORDS;w+=128u){
  var bits=0u;
  if(narrow){
   for(var b=0u;b<32u;b++){
    let pen=w*32u+b;if(pen>=p.extent.x){break;}
    let c=pen+1u;let low=c&(~c+1u);var v=output[cursor+c];
    for(var k=1u;k<low;k*=2u){v-=output[cursor+c-k];}
    if(v==1u){bits|=1u<<b;}else if(v!=0u){atomicStore(&localBroken,1u);}
   }
  }
  localBits[w]=bits;
  localPaper[w]=select(0u,atomicLoad(&counts[paperBase()+row*paperStride()+min(w,paperStride()-1u)]),narrow&&w<paperStride());
 }
 if(!narrow&&lane==0u){atomicStore(&localBroken,1u);}
 workgroupBarrier();
 for(var j=lane+1u;j<=LOCAL_WORDS;j+=128u){let low=j&(~j+1u);var n=0u;for(var w=j-low;w<j;w++){n+=countOneBits(localBits[w]);}localSums[j]=n;}
 for(var i=lane;i<128u;i+=128u){glyphTags[i]=NONE;}
 for(var s=lane;s<8u;s+=128u){var m=0u;for(var k=0u;k<32u;k++){if(localBits[s*32u+k]!=0u){m|=1u<<k;}}localSummary[s]=m;}
 workgroupBarrier();
 if(lane==0u){
 twinBase=start;twinsReady=count<=${ROW_CAP}u;
 localCursor=atomicLoad(&localBroken)==0u;localBase=cursor;localTotal=localPrefix(LOCAL_WORDS);
 paperLocal=narrow;paperRow=row;
 // One evaluation per word in the fixed priority order; no second pass. The
 // loss count is every word that drew no scalar and was not hidden, shared
 // or a duplicate. A clipped word that drew at least one glyph is not a loss.
 for(var i=0u;i<count;i++){
  var index=start+i;if(count<=${ROW_CAP}u){index=order[i];}else{index=atomicLoad(&counts[p.playback.w+start+i]);}
  // Region tone words are placed by toneplace after this loop. Labels and
  // tone members must still belong to their typed compacted queues; the
  // queue is a consumer contract, not only an allocation tail.
  let record=rowWords[index].data;
  if((record[9]==2u||record[9]==3u)&&!typedQueueActive(record[0],record[9])){continue;}
  if(count<=1024u&&record[9]==3u){continue;}
  if(levelTwin(index,row,lows,highs)){probeAdd(33u);continue;}
  let word=expandWord(rowWords[index]);probeAdd(32u);if(!typesetWord(word,row,cursor)){atomicAdd(&counts[rowCount()+1u],1u);probeLoss(word,row,cursor);}
  else if(row*2u>rowCount()&&word.data[9]!=3u&&word.data[9]!=2u){let r=runs[word.data[0]];probeAdd(68u+probeClass(r));probeAdd(75u+min((r.range.z>>8u)&255u,4u));if(terrainRun(r)){probeAdd(80u+min(distanceTier(r),2u));}}
 }
 }
 // The stored Fenwick again, from the local copy, for toneplace, the stacked
 // camera's foreignRemove and the seal (unless lane 0 left the copy).
 workgroupBarrier();
 if(atomicLoad(&localBroken)==0u){
  for(var c=lane+1u;c<=p.extent.x;c+=128u){
   let v=localNode(c);
   if(PLACE_CHECK){if(output[cursor+c]!=v){placeCheck(1u);}}else{output[cursor+c]=v;}
  }
 }
}
`;
/** STACKED CAMERA OCCLUDER passes (patch_vmocclude; gpu-articulation occluderFor). */
export const OCCLUDER_SHADER=/* wgsl */`
// STACKED CAMERA OCCLUDER (patch_vmocclude). One texel per (page x, glyph row).
struct OP {pageW:u32,rows:u32,mode:u32,margin:u32,rect:vec4f,field:vec2f,capacity:u32,pad:u32}
@group(0) @binding(0) var<uniform> o:OP;
@group(0) @binding(1) var upperOwner:texture_2d<u32>;
@group(0) @binding(2) var mask:texture_storage_2d<r32uint,write>;
@group(0) @binding(3) var<storage,read> scalars:array<u32>;
@group(0) @binding(4) var<storage,read> sealCount:array<u32>;
// The upper camera's rendered coverage: any owner in this row's 17 px band within the margin.
@compute @workgroup_size(64) fn coverage(@builtin(global_invocation_id) id:vec3u){
 let x=id.x;let row=id.y;if(x>=o.pageW||row>=o.rows){return;}
 let size=vec2i(textureDimensions(upperOwner));var hit=false;
 for(var dy=0u;dy<17u;dy++){
  for(var dx=0u;dx<=2u*o.margin;dx++){
   let page=vec2f(f32(i32(x)+i32(dx)-i32(o.margin))+.5,f32(row*17u+dy)+.5);
   let f=vec2i(floor((page-o.rect.xy)/o.rect.zw*o.field));
   if(any(f<vec2i(0))||any(f>=size)){continue;}
   let w=textureLoad(upperOwner,f,0).xy;if((w.x|w.y)!=0u){hit=true;}
  }
 }
 if(hit){textureStore(mask,vec2i(i32(x),i32(row)),vec4u(1u,0u,0u,0u));}
 else if(o.mode==0u){textureStore(mask,vec2i(i32(x),i32(row)),vec4u(0u,0u,0u,0u));}
}
// The upper camera's sealed glyph cells (scalar record: x at 3, row at 4, advance at 7).
@compute @workgroup_size(64) fn glyphs(@builtin(global_invocation_id) id:vec3u){
 let index=id.x;if(index>=min(sealCount[1],o.capacity)){return;}
 let at=index*32u;let x=scalars[at+3u];let row=scalars[at+4u];let advance=scalars[at+7u];
 if(row>=o.rows){return;}
 let lo=max(i32(x)-i32(o.margin),0);let hi=min(i32(x+advance+o.margin),i32(o.pageW));
 for(var px=lo;px<hi;px++){textureStore(mask,vec2i(px,i32(row)),vec4u(1u,0u,0u,0u));}
}
// STACKED CAMERAS: the upper camera's sealed glyphs of each row become committed
// neighbours of the covered camera's placement, written into that job's own
// output records and row cursor (the articulation's layout: record = 32 words,
// cursor = tape*32 + row*(width*2+1), Fenwick over pen+1, address slot at
// width+1+pen). One thread per row: the Fenwick counts are not atomic.
struct FP {width:u32,rows:u32,tape:u32,pad:u32}
@group(0) @binding(5) var<uniform> fp:FP;
@group(0) @binding(6) var<storage,read_write> output:array<u32>;
@group(0) @binding(7) var<storage,read_write> counts:array<atomic<u32>>;
fn foreignCount(base:u32,end:u32)->u32{var i=end;var n=0u;while(i>0u){n+=output[base+i];i-=i&(~i+1u);}return n;}
fn foreignCursorInsert(base:u32,pen:u32,address:u32){output[base+fp.width+1u+pen]=address;var i=pen+1u;while(i<=fp.width){output[base+i]++;i+=i&(~i+1u);}}
@compute @workgroup_size(64) fn foreignInsert(@builtin(global_invocation_id) id:vec3u){
 let row=id.x;if(row>=fp.rows||arrayLength(&sealCount)<9u+fp.rows){return;}
 let limit=min(arrayLength(&scalars)/32u,sealCount[1]);let base=fp.tape*32u+row*(fp.width*2u+1u);
 for(var i=sealCount[8u+row];i<min(sealCount[9u+row],limit);i++){
  let x=scalars[i*32u+3u];if(x>=fp.width||scalars[i*32u+4u]!=row){continue;}
  if(foreignCount(base,x+1u)!=foreignCount(base,x)){continue;}
  let out=atomicAdd(&counts[fp.rows+2u],1u);
  if(out>=fp.tape){atomicOr(&counts[fp.rows],2u);return;}
  for(var k=0u;k<32u;k++){output[out*32u+k]=scalars[i*32u+k];}
  output[out*32u+9u]=5u;
  foreignCursorInsert(base,x,out<<4u);
 }
}
// After the covered camera's placement and before its seal: the foreign
// neighbours (part 5) leave its cursor; its records stay but are not exported.
@compute @workgroup_size(64) fn foreignRemove(@builtin(global_invocation_id) id:vec3u){
 let row=id.x;if(row>=fp.rows||arrayLength(&sealCount)<9u+fp.rows){return;}
 let limit=min(arrayLength(&scalars)/32u,sealCount[1]);let base=fp.tape*32u+row*(fp.width*2u+1u);
 for(var i=sealCount[8u+row];i<min(sealCount[9u+row],limit);i++){
  let x=scalars[i*32u+3u];if(x>=fp.width||scalars[i*32u+4u]!=row){continue;}
  if(foreignCount(base,x+1u)==foreignCount(base,x)){continue;}
  let address=output[base+fp.width+1u+x];
  if(output[(address>>4u)*32u+9u]!=5u){continue;}
  var j=x+1u;while(j<=fp.width){output[base+j]--;j+=j&(~j+1u);}
 }
}
`;

export const ARTICULATION_SHADER=ARTICULATION_SOURCE;










interface Pending {
 readonly runs:readonly ArticulationRun[];readonly width:number;readonly height:number;readonly field:SurfaceField;
 readonly literals:ReadonlyMap<number,ArticulationLiteral>;
 readonly rect:{x:number;y:number;width:number;height:number};
 readonly resolve:(result:ArticulationResult)=>void;readonly reject:(error:unknown)=>void;
 /** Count-driven allocation retry retains this exact source/view snapshot. */
 outputCapacity?:number;
 /** Sealed scalar tape capacity grown from a reported scalar count. */
 scalarCapacity?:number;
 proposalCapacity?:number;
 /** Captured before deferred tone/labels or another camera reuse the table. */
 partAnchorSnapshot?:GPUBuffer;
 /** Cameras of the same target stacked above this one (their coverage occludes it). */
 occludedBy?:readonly string[];
 rowCapacities?:Uint32Array;
 sourceLines?:number;
 memoSlots?:number;
 gpuTone?:GpuToneMembers;
 gpu?:{readonly runs:ResidentConstructionRuns;readonly source:RelationVisibility;readonly camera:string;readonly owners:ReadonlyMap<number,number>;readonly count:number};
 deferredSource?:{readonly source:RelationVisibility;readonly classes:readonly SourceConstructionClass[];readonly camera:string;readonly owners:ReadonlyMap<number,number>;readonly partAnchorTable?:SourcePartAnchorTable};
 ready?:boolean;settled?:boolean;cohort?:CameraCohort;readonly cohortEncoder?:GPUCommandEncoder;
}
interface CameraMemo {readonly buffer:GPUBuffer;readonly slots:number;readonly lifecycle:number;readonly epochs:Uint32Array;readonly hasOwners:boolean}
interface CameraCohort {readonly key:string;readonly explicit:boolean;readonly jobs:Set<Pending>;readonly residents:Map<Pending,ResidentNativeConstruction>}
interface FrameSeal {readonly encoder:GPUCommandEncoder;readonly sealed:Omit<GpuNativeTape,'seal'>;readonly field:SurfaceField;readonly rect:{x:number;y:number;width:number;height:number};readonly cohort?:CameraCohort}

function articulationPipelinePlan(device:GPUDevice){
 const storage=(binding:number,type:GPUBufferBindingType='read-only-storage'):GPUBindGroupLayoutEntry=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type}});
 const layout=device.createBindGroupLayout({label:'MAIN compiled native grammar',entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:'uniform'}},...[1,2,3,4,6].map(binding=>storage(binding)),...[7,8,9].map(binding=>storage(binding,'storage')),...[10,11,13].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:'unfilterable-float' as const}})),...[5,12,14,15].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:'uint' as const}}))]});
 const pipeline=device.createPipelineLayout({bindGroupLayouts:[layout]});
 const plans=new Map<string,ComputePipelinePlan>();
 const stage=(key:string,entryPoint:string,code:string)=>{const label='MAIN '+key;plans.set(key,()=>{const module=device.createShaderModule({label,code});return {label,layout:pipeline,compute:{module,entryPoint}};});};
 for(const entryPoint of ['featureplace','toneplace','featureorder','featureorderMerge','wordvisibility','sourcePartInit','sourcePartReduce','claimMemo','resolveMemo','reclaimMemo','skylineReduce'])stage(entryPoint,entryPoint,isolateComputeEntry(ARTICULATION_SHADER,entryPoint));
 for(const {entryPoint,code}of buildFarKindStages(ARTICULATION_SHADER).entries)stage(entryPoint,entryPoint,code);
 for(const family of PLAYBACK_FAMILIES)stage('featurecurves:'+family,'featurecurves',familyPlayback(ARTICULATION_SHADER,family));
 plans.set('visibilityDispatch',()=>({layout:'auto',compute:{module:device.createShaderModule({label:'MAIN active word visibility dispatch',code:WORD_VISIBILITY_DISPATCH_SHADER}),entryPoint:'main'}}));
 plans.set('activeRuns',()=>({layout:'auto',compute:{module:device.createShaderModule({label:'MAIN active source-run compaction',code:ACTIVE_RUNS_SHADER}),entryPoint:'main'}}));
 plans.set('activeRunArgs',()=>({layout:'auto',compute:{module:device.createShaderModule({label:'MAIN active source-run indirect arguments',code:ACTIVE_RUN_ARGS_SHADER}),entryPoint:'main'}}));
 plans.set('rehashMemo',()=>({layout:'auto',compute:{module:device.createShaderModule({label:'MAIN exact source-key correspondence rehash',code:MEMO_REHASH_SHADER}),entryPoint:'rehash'}}));
 plans.set('toneSeed',()=>({layout:'auto',compute:{module:device.createShaderModule({label:'MAIN resident region words',code:GPU_TONE_SEED_SHADER}),entryPoint:'seedTone'}}));
 return {layout,plans};
}
interface PreparedArticulation {
 readonly packed:PackedRoleGrammar;readonly sourceRuns?:GpuConstructionRuns;readonly seal?:GpuNativeSealer;
 readonly layout?:GPUBindGroupLayout;readonly pipelines?:ReadonlyMap<string,GPUComputePipeline>;
}

/** Host-encoder-owned compute. enqueue never submits a GPU queue. Both the
 * header and its COUNT-SIZED member copy ride the engine's normal encoder
 * and submit fence; the same decoded tape feeds paint and native TXT. */
export class GpuArticulation {
 private readonly sourceRuns:GpuConstructionRuns;
 private readonly packed:PackedRoleGrammar;private readonly layout:GPUBindGroupLayout;
 private readonly smallTable?:PackedSmallConstructionTable;
 /** The frontier and shader address the SAME compiled table. No source
  * compilation, glyph matching or candidate scan runs in a camera frame. */
 smallConstructionAvailable(parent:SourceSmallParent,rows:number,width:number):boolean{
  const table=this.smallTable;
  if(!table||!parent.replaceable||parent.positiveEnd==='open'||parent.negativeEnd==='open'||
   !Number.isInteger(rows)||!Number.isFinite(width)||width<0||width>table.widthLimit)return false;
  const record=lookupSmallConstructionRecord(table,parent.kind,parent.positiveEnd,parent.negativeEnd,rows,Math.round(width));
  return !!record&&record.rows===rows&&record.width<=Math.round(width)&&Math.round(width)-record.width<16&&record.atomCount>0;
 }
 // MATERIAL INK table (binding 15, an rg32uint texture). A replaced texture
 // is kept until dispose, since an encoded job may still bind it.
 private inkTexture?:GPUTexture;private inkBytes=new Uint32Array([0,0]);private inkDirty=true;private readonly inkRetired:GPUTexture[]=[];
 /** The frame's material inks (materialInkTable); absent means none (black lines). */
 setMaterialInks(table:Uint32Array):void{
  if(table.length===this.inkBytes.length&&table.every((v,i)=>v===this.inkBytes[i]))return;
  this.inkBytes=table.slice();this.inkDirty=true;
 }
 private materialInkView():GPUTextureView{
  const {width,height,data}=materialInkTexels(this.inkBytes);
  if(!this.inkTexture||this.inkTexture.width!==width||this.inkTexture.height!==height){if(this.inkTexture)this.inkRetired.push(this.inkTexture);this.inkTexture=this.device.createTexture({label:'MAIN material inks',size:[width,height],format:'rg32uint',usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});this.inkDirty=true;}
  if(this.inkDirty){this.device.queue.writeTexture({texture:this.inkTexture},data,{bytesPerRow:width*8},[width,height]);this.inkDirty=false;}
  return this.inkTexture.createView();
 }
 private readonly curves:readonly GPUComputePipeline[];private readonly far:readonly GPUComputePipeline[];private readonly place:GPUComputePipeline;private readonly tonePlace:GPUComputePipeline;private readonly sourcePartInit:GPUComputePipeline;private readonly sourcePartReduce:GPUComputePipeline;private readonly claimMemo:GPUComputePipeline;private readonly resolveMemo:GPUComputePipeline;private readonly reclaimMemo:GPUComputePipeline;
 private readonly order:GPUComputePipeline;private readonly orderMerge:GPUComputePipeline;
 private readonly wordVisibility:GPUComputePipeline;private readonly visibilityDispatch:GPUComputePipeline;private readonly activeRuns:GPUComputePipeline;private readonly activeRunArgs:GPUComputePipeline;
 /** skylineReduce: the merged far skyline profile, after the memo passes and before featurecurves. */
 private readonly skyline:GPUComputePipeline;





 private wordText(word:number):string{const at=word*WORD_STRIDE,count=this.packed.words[at]??0;let text='';for(let s=0;s<Math.min(count,32);s++){const glyph=this.packed.font[this.packed.words[at+2+s]!];text+=glyph?.text??'?';}return text;}
 private readonly rehashMemo:GPUComputePipeline;
 private readonly toneSeed:GPUComputePipeline;
 private readonly seal:GpuNativeSealer;
 private readonly exports:{encode:(encoder:GPUCommandEncoder)=>void;submitted:()=>void;cancel:()=>void}[]=[];
 private readonly cancelExports=new Set<()=>void>();
 private readonly tapes=new Set<ResidentNativeConstruction>();
 private readonly resident:GPUBuffer[];private pending:Pending[]=[]; private readonly gapTables=new Map<number,GPUBuffer>();
 private readonly fontAddresses:ReadonlyMap<string,number>;
 private readonly parts:ReadonlyMap<string,PackedRoleGrammar['parts'][number]>;
 private readonly active=new Map<Pending,Set<GPUBuffer>>();
 // Only completed jobs donate storage. Keep one unused buffer per purpose,
 // not one cache entry for every viewport size encountered while playing.
 private readonly spareScratch=new Map<'rows'|'tape',GPUBuffer>();
 private readonly scratchKinds=new Map<GPUBuffer,'rows'|'tape'>();
 // Allocation hint only: a dense held view must not repeat its count/grow
 // round trip every frame. The exact current count still owns admission.
 private readonly rowHints=new Map<string,{width:number;height:number;capacities:Uint32Array}>();
 // Native packet allocation, not a world/relation readback. Retain the
 // observed tape extent so publication does not wait for another host
 // frame merely to copy a few already-selected words after their header.
 // STACKED CAMERAS: camera jobs form a target cohort. A covered job cannot
 // encode until every declared upper job is ready and has a valid seal. Seals
 // survive a later lower-camera allocation retry, but never cross cohorts.
 private readonly frameSeals=new Map<string,FrameSeal>();
 private readonly cameraJobs=new Map<string,Set<Pending>>();
 private readonly cohorts=new Set<CameraCohort>();
 private cohortSequence=0;
 private cameraOf(job:Pending):string|undefined{return job.gpu?.camera??job.deferredSource?.camera;}
 private sealKey(job:Pending,camera:string):string{return `${job.cohort!.key}\u0000${camera}`;}
 private attachCohort(job:Pending,requested?:string):void{
  const camera=this.cameraOf(job);if(!camera)return;
  const linked=new Set<CameraCohort>(),candidates=[...(this.cameraJobs.get(camera)??[])];
  for(const above of job.occludedBy??[])candidates.push(...(this.cameraJobs.get(above)??[]));
  // The normal adapter creates the upper camera first. The inverse scan keeps
  // the contract valid if a producer delivers a covered camera first.
  for(const members of this.cameraJobs.values())for(const candidate of members)if(candidate.occludedBy?.includes(camera))candidates.push(candidate);
  for(const candidate of candidates)if(candidate!==job&&this.active.has(candidate)&&candidate.cohort&&
    (requested!==undefined?candidate.cohort.key===requested:!candidate.cohort.explicit&&candidate.cohortEncoder===job.cohortEncoder))linked.add(candidate.cohort);
  let cohort=requested!==undefined?[...this.cohorts].find(value=>value.key===requested):undefined;
  if(!cohort)cohort=[...linked][0];
  if(!cohort){cohort={key:requested??`camera:${camera}:${++this.cohortSequence}`,explicit:requested!==undefined,jobs:new Set(),residents:new Map()};this.cohorts.add(cohort);}
  for(const other of linked)if(other!==cohort){for(const member of other.jobs){cohort.jobs.add(member);member.cohort=cohort;}other.jobs.clear();other.residents.clear();this.cohorts.delete(other);}
  cohort.jobs.add(job);job.cohort=cohort;
  let jobs=this.cameraJobs.get(camera);if(!jobs){jobs=new Set();this.cameraJobs.set(camera,jobs);}jobs.add(job);
 }
 private detachCohortJob(job:Pending):void{
  const camera=this.cameraOf(job),jobs=camera?this.cameraJobs.get(camera):undefined;jobs?.delete(job);if(jobs&&!jobs.size)this.cameraJobs.delete(camera!);
  const cohort=job.cohort;if(!cohort)return;cohort.jobs.delete(job);cohort.residents.delete(job);
  if(!cohort.jobs.size){for(const [name,seal]of this.frameSeals)if(seal.cohort===cohort)this.frameSeals.delete(name);this.cohorts.delete(cohort);}
 }
 private completeCohort(cohort:CameraCohort):void{
  if([...cohort.jobs].some(job=>this.active.has(job)||this.pending.includes(job)))return;
  for(const [name,seal]of this.frameSeals)if(seal.cohort===cohort)this.frameSeals.delete(name);
  for(const job of [...cohort.jobs])this.detachCohortJob(job);
 }
 private abortCohort(job:Pending,error:unknown):void{
  const cohort=job.cohort;
  if(!cohort){if(!job.settled){job.settled=true;job.reject(error);}this.detachCohortJob(job);return;}
  for(const resident of cohort.residents.values())resident.release();cohort.residents.clear();
  this.pending=this.pending.filter(candidate=>!cohort.jobs.has(candidate));
  for(const member of [...cohort.jobs]){
   if(this.active.has(member)){member.gpu?.runs.dispose();this.release(member);}
   if(!member.settled){member.settled=true;member.reject(error);}
   this.detachCohortJob(member);
  }
  for(const [name,seal]of this.frameSeals)if(seal.cohort===cohort)this.frameSeals.delete(name);
  this.cohorts.delete(cohort);
 }
 private dependenciesReady(job:Pending,encoder:GPUCommandEncoder):boolean{
  const cohort=job.cohort;
  return (job.occludedBy??[]).every(camera=>{const seal=this.frameSeals.get(this.sealKey(job,camera));return !!seal&&seal.cohort===cohort;});
 }
 private retryJob(job:Pending):void{
  const camera=this.cameraOf(job);if(camera&&job.cohort)this.frameSeals.delete(this.sealKey(job,camera));
  this.release(job,true);this.active.set(job,new Set());this.pending.push(job);
 }
 private resolveCohortOverflow(job:Pending,result:ArticulationResult):void{
  const cohort=job.cohort;if(!cohort){job.settled=true;job.resolve(result);job.gpu?.runs.dispose();this.release(job);return;}
  this.pending=this.pending.filter(candidate=>!cohort.jobs.has(candidate));
  for(const resident of cohort.residents.values())resident.release();cohort.residents.clear();
  for(const member of [...cohort.jobs]){
   if(this.active.has(member)){member.gpu?.runs.dispose();this.release(member);}
   if(!member.settled){member.settled=true;member.resolve(result);}
   this.detachCohortJob(member);
  }
  for(const [name,seal]of this.frameSeals)if(seal.cohort===cohort)this.frameSeals.delete(name);
  this.cohorts.delete(cohort);
 }
 private emptyOccluder?:GPUTexture;private readonly occluders=new Map<string,GPUTexture>();private occluderPipelines?:{coverage:GPUComputePipeline;glyphs:GPUComputePipeline;foreign:GPUComputePipeline;foreignRemove:GPUComputePipeline};
 /** Covered jobs whose upper camera had no complete seal in its cohort. */
 occluderMissing=0;
 private occluderFor(encoder:GPUCommandEncoder,job:Pending,rows:number,own:(buffer:GPUBuffer)=>GPUBuffer):{mask:GPUTexture;upper?:{words:GPUBuffer;count:GPUBuffer}}{
  const empty=()=>this.emptyOccluder??=this.device.createTexture({label:'MAIN no stacked occluder',size:[1,1],format:'r32uint',usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING});
  if(!job.occludedBy?.length)return {mask:empty()};
  const above=job.occludedBy.map(camera=>this.frameSeals.get(this.sealKey(job,camera))).filter(entry=>entry?.cohort===job.cohort) as FrameSeal[];
  if(above.length<job.occludedBy.length)this.occluderMissing++;
  if(!above.length)return {mask:empty()};
  const key=job.gpu?.camera??'';let texture=this.occluders.get(key);
  if(!texture||texture.width!==job.width||texture.height!==rows){texture?.destroy();texture=this.device.createTexture({label:'MAIN stacked camera occluder',size:[job.width,rows],format:'r32uint',usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING});this.occluders.set(key,texture);}
  if(!this.occluderPipelines){const module=this.device.createShaderModule({label:'MAIN stacked camera occluder',code:OCCLUDER_SHADER});this.occluderPipelines={coverage:this.device.createComputePipeline({label:'MAIN stacked occluder coverage',layout:'auto',compute:{module,entryPoint:'coverage'}}),glyphs:this.device.createComputePipeline({label:'MAIN stacked occluder glyphs',layout:'auto',compute:{module,entryPoint:'glyphs'}}),foreign:this.device.createComputePipeline({label:'MAIN stacked camera foreign neighbours',layout:'auto',compute:{module,entryPoint:'foreignInsert'}}),foreignRemove:this.device.createComputePipeline({label:'MAIN stacked camera foreign neighbours removed',layout:'auto',compute:{module,entryPoint:'foreignRemove'}})};}
  const {coverage,glyphs}=this.occluderPipelines;const view=texture.createView();
  above.forEach((upper,index)=>{
   const data=new ArrayBuffer(48),u=new Uint32Array(data),f=new Float32Array(data);
   u[0]=job.width;u[1]=rows;u[2]=index===0?0:1;u[3]=5;f[4]=upper.rect.x;f[5]=upper.rect.y;f[6]=upper.rect.width;f[7]=upper.rect.height;f[8]=upper.field.width;f[9]=upper.field.height;u[10]=upper.sealed.capacity;
   const uniform=own(this.device.createBuffer({label:'MAIN stacked occluder parameters',size:48,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}));this.device.queue.writeBuffer(uniform,0,data);
   const a=encoder.beginComputePass({label:'MAIN stacked occluder coverage'});a.setPipeline(coverage);
   a.setBindGroup(0,this.device.createBindGroup({layout:coverage.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:uniform}},{binding:1,resource:upper.field.textures[3]!.createView()},{binding:2,resource:view}]}));
   a.dispatchWorkgroups(Math.ceil(job.width/64),rows);a.end();
   const b=encoder.beginComputePass({label:'MAIN stacked occluder glyphs'});b.setPipeline(glyphs);
   b.setBindGroup(0,this.device.createBindGroup({layout:glyphs.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:uniform}},{binding:2,resource:view},{binding:3,resource:{buffer:upper.sealed.words}},{binding:4,resource:{buffer:upper.sealed.count}}]}));
   b.dispatchWorkgroups(Math.max(1,Math.ceil(upper.sealed.capacity/64)));b.end();
  });
  // The nearest stacked camera (encoded last of those above) supplies the foreign neighbours.
  const nearest=above[above.length-1]!;
  return {mask:texture,upper:{words:nearest.sealed.words,count:nearest.sealed.count}};
 }
 private tapeCapacity=4096;
 // Largest sealed scalar count seen (grown from the count, never a quota).
 private scalarCapacity=0;
 private proposalCapacity=4096;
 private readonly retiredMemo=new Set<GPUBuffer>();
 private readonly cameraMemos=new Map<string,CameraMemo>();
 private liveCameras?:ReadonlySet<string>;
 private sourceLife?:{readonly domains:Uint32Array;readonly producers?:Uint32Array};private lifecycle=0;
 private submittedTasks:(()=>void)[]=[];private readonly submissionMaps:(()=>void)[]=[];private disposed=false;
 static async create(device:GPUDevice,font:readonly NativeGlyph[],whitespace:NativeWhitespacePolicy='preformatted',maxWidth=8192,progress?:(fraction:number)=>void):Promise<GpuArticulation>{
  const packed=await loadRuntimeRoleGrammar(font,whitespace,maxWidth);
  const sourceRuns=await GpuConstructionRuns.create(device,packed.anchorContexts,p=>progress?.(p/3));
  try{
   const {layout,plans}=articulationPipelinePlan(device),pipelines=await prepareComputePipelines(device,plans,p=>progress?.((1+p)/3));
   const seal=await GpuNativeSealer.create(device,p=>progress?.((2+p)/3));
   return new GpuArticulation(device,font,whitespace,maxWidth,{packed,sourceRuns,layout,pipelines,seal});
  }catch(error){sourceRuns.dispose();throw error;}
 }
 constructor(private readonly device:GPUDevice,font:readonly NativeGlyph[],whitespace:NativeWhitespacePolicy='preformatted',private readonly maxWidth=8192,prepared?:PreparedArticulation){
  if(!prepared)throw Error('GpuArticulation requires a prepared grammar artifact; use await GpuArticulation.create()');
  this.packed=prepared.packed;
  this.smallTable=decodeSmallConstructionTable(this.packed.farStrip);
  this.sourceRuns=prepared?.sourceRuns??new GpuConstructionRuns(device,this.packed.anchorContexts);
  this.fontAddresses=new Map(font.map((glyph,index)=>[glyph.text,index]));
  this.parts=new Map(this.packed.parts.map(part=>[part.id,part]));
  const plan=prepared.layout&&prepared.pipelines?{layout:prepared.layout,pipelines:prepared.pipelines}:(()=>{const {layout,plans}=articulationPipelinePlan(device);return {layout,pipelines:compileComputePipelines(device,plans)};})();
  this.layout=plan.layout;const pipelines=plan.pipelines;
  this.curves=PLAYBACK_FAMILIES.map(family=>pipelines.get('featurecurves:'+family)!);this.far=FAR_KIND_ENTRIES.map(entry=>pipelines.get(entry)!);this.place=pipelines.get('featureplace')!;this.tonePlace=pipelines.get('toneplace')!;
  this.order=pipelines.get('featureorder')!;this.orderMerge=pipelines.get('featureorderMerge')!;
  this.wordVisibility=pipelines.get('wordvisibility')!;this.visibilityDispatch=pipelines.get('visibilityDispatch')!;
  this.activeRuns=pipelines.get('activeRuns')!;this.activeRunArgs=pipelines.get('activeRunArgs')!;
  this.sourcePartInit=pipelines.get('sourcePartInit')!;this.sourcePartReduce=pipelines.get('sourcePartReduce')!;this.claimMemo=pipelines.get('claimMemo')!;this.resolveMemo=pipelines.get('resolveMemo')!;this.reclaimMemo=pipelines.get('reclaimMemo')!;
  this.skyline=pipelines.get('skylineReduce')!;this.rehashMemo=pipelines.get('rehashMemo')!;this.toneSeed=pipelines.get('toneSeed')!;
  this.seal=prepared.seal??new GpuNativeSealer(device);
  const articulatedAt=12+this.packed.instructions.length,hardAt=articulatedAt+this.packed.articulatedInstructions.length,partsAt=hardAt+this.packed.hardLineInstructions.length,paperAt=partsAt+this.packed.partAtoms.length,circleAt=paperAt+this.packed.partPaper.length,metricsAt=circleAt+this.packed.circleProgram.length,pointsAt=metricsAt+this.packed.glyphs.length;
  // The far strip table is appended after every existing section (header
  // slot 9), so no earlier bank address moves.
  const farAt=pointsAt+this.packed.points.length;
  // The hard-junction table follows it (header slot 10), again at the end.
  const junctionAt=farAt+this.packed.farStrip.length;
  // The class tables follow it (header slot 11, the last free slot): the
  // paired-flank records, then the terrain line table. Again at the end.
  const classAt=junctionAt+this.packed.hardJunctions.length,terrainAt=classAt+this.packed.pairedFlanks.length;
  // The conifer side units follow the terrain table, at the fixed offset the
  // shader reads (FOLIAGE_EDGES_AT from grammar[11]).
  const foliageAt=terrainAt+this.packed.terrainInstructions.length;if(foliageAt-classAt!==FOLIAGE_EDGES_AT)throw Error('Foliage edge table offset changed');
const bank=new Uint32Array(foliageAt+this.packed.foliageEdges.length);bank.set(this.packed.foliageEdges,foliageAt);bank.set([12,partsAt,metricsAt,pointsAt,articulatedAt,this.packed.cursorThreshold,paperAt,circleAt,hardAt,farAt,junctionAt,classAt]);bank.set(this.packed.pairedFlanks,classAt);bank.set(this.packed.terrainInstructions,terrainAt);bank.set(this.packed.farStrip,farAt);bank.set(this.packed.hardJunctions,junctionAt);bank.set(this.packed.instructions,12);bank.set(this.packed.articulatedInstructions,articulatedAt);bank.set(this.packed.hardLineInstructions,hardAt);bank.set(this.packed.partAtoms,partsAt);bank.set(this.packed.partPaper,paperAt);bank.set(this.packed.circleProgram,circleAt);bank.set(this.packed.glyphs,metricsAt);bank.set(new Uint32Array(this.packed.points.buffer,this.packed.points.byteOffset,this.packed.points.length),pointsAt);
  this.resident=[this.upload(bank),this.upload(this.packed.words)];
 }
 private upload(data:ArrayBufferView,usage=GPUBufferUsage.STORAGE):GPUBuffer{
  const buffer=this.device.createBuffer({size:Math.max(4,data.byteLength),usage:usage|GPUBufferUsage.COPY_DST});
  if(data.byteLength)this.device.queue.writeBuffer(buffer,0,data.buffer,data.byteOffset,data.byteLength);return buffer;
 }
 enqueue(runs:readonly ArticulationRun[],field:SurfaceField,width:number,height:number,rect={x:0,y:0,width,height},literals:readonly ArticulationLiteral[]=[]):Promise<ArticulationResult>{
  if(this.disposed)return Promise.reject(Error('Native GPU playback is disposed'));
  if(width>this.maxWidth)return Promise.reject(Error('Native GPU playback width exceeds its compiled cursor table'));
  const records=[...runs],labels=new Map<number,ArticulationLiteral>();
  for(const literal of literals){labels.set(records.length,literal);records.push({id:literal.group.id,owner:literal.group.id,sources:[],role:'flank',size:0,lineClass:0,key:0xffffffff-records.length,points:[],anchorOnly:true});}
  return new Promise((resolve,reject)=>{const job={runs:records,literals:labels,field,width,height,rect,resolve,reject};this.active.set(job,new Set());this.pending.push(job);});
 }
 /** Default retained path: the source graph and its visibility never leave
  * the GPU. CPU input is immutable source classification and authored text;
  * CPU output is only the already-selected native word tape. */
 /** `cohortKey` is the adapter's target+pose identity. The same key must be
  * supplied to every stacked camera; the encoder fallback only protects
  * legacy callers that enqueue all cameras from one host frame. */
 enqueueSource(encoder:GPUCommandEncoder,source:RelationVisibility,classes:readonly SourceConstructionClass[],field:SurfaceField,width:number,height:number,camera:string,owners:ReadonlyMap<number,number>,rect={x:0,y:0,width,height},literals:readonly ArticulationLiteral[]|ArticulationLiteralFeed=[],gpuTone?:GpuToneMembers,occludedBy:readonly string[]=[],cohortKey?:string,partAnchorTable?:SourcePartAnchorTable):Promise<ArticulationResult>{
  if(this.disposed)return Promise.reject(Error('Native GPU playback is disposed'));
  if(width>this.maxWidth)return Promise.reject(Error('Native GPU playback width exceeds its compiled cursor table'));
  // Assemble NOW, before another camera reuses the relation/visibility
  // buffers. Tone may arrive later; the generated runs/status are job-owned.
  const deferred=!Array.isArray(literals);
  const generated=this.sourceRuns.encode(encoder,source,classes,rect,width,height,deferred?0:(literals as readonly ArticulationLiteral[]).length,field);
  const records:ArticulationRun[]=[],labels=new Map<number,ArticulationLiteral>();
  return new Promise((resolve,reject)=>{
   const job:Pending={runs:records,literals:labels,field,width,height,rect,resolve,reject,gpu:{runs:generated,source,camera,owners,count:generated.capacity},occludedBy,cohortEncoder:encoder};
   this.active.set(job,new Set());this.attachCohort(job,cohortKey);
   try{this.capturePartAnchors(encoder,job,partAnchorTable);}catch(error){generated.dispose();this.release(job);this.abortCohort(job,error);return;}
   let delivered=false;
   const ready=(input:readonly ArticulationLiteral[],tone?:GpuToneMembers)=>{
    if(delivered||this.disposed||!this.active.has(job))return;delivered=true;
    // A delayed tone/label packet belongs to the camera snapshot that
    // retained these source runs, never the next camera to reuse its fields.
    if(input.some(literal=>literal.camera!==camera)){failed(Error('Native literals belong to another camera'));return;}
    if(tone&&tone.field!==field){failed(Error('Native region members belong to another source pose'));return;}
    for(const literal of input){const index=generated.capacity+records.length;labels.set(index,literal);records.push({id:literal.group.id,owner:literal.group.id,sources:[],role:'flank',size:0,lineClass:0,key:0xffffffff-index,points:[],anchorOnly:true});}
    job.gpuTone=tone;job.gpu={...job.gpu!,count:generated.capacity+records.length+(tone?.capacity??0)};
    job.ready=true;if(deferred||occludedBy.length||cohortKey)this.pending.push(job);else this.encodeJob(encoder,job);
   };
   const failed=(error:unknown)=>{if(!this.active.has(job))return;generated.dispose();this.release(job);this.abortCohort(job,error);};
   if(deferred)try{(literals as ArticulationLiteralFeed).subscribe(ready,failed);}catch(error){failed(error);}
   else try{ready(literals as readonly ArticulationLiteral[],gpuTone);}catch(error){failed(error);}
  });
 }
 /** Same retained path, encoded at the NEXT host encoder (after the frame's
  * compact tone members are known). The relation/visibility buffers are held
  * by the pending physical job until this resolves, so no CPU graph decode,
  * relation readback or CPU run compilation occurs in a displayed frame. */
 enqueueSourceNext(source:RelationVisibility,classes:readonly SourceConstructionClass[],field:SurfaceField,width:number,height:number,camera:string,owners:ReadonlyMap<number,number>,rect={x:0,y:0,width,height},literals:readonly ArticulationLiteral[]=[],gpuTone?:GpuToneMembers,partAnchorTable?:SourcePartAnchorTable):Promise<ArticulationResult>{
  if(this.disposed)return Promise.reject(Error('Native GPU playback is disposed'));
  if(width>this.maxWidth)return Promise.reject(Error('Native GPU playback width exceeds its compiled cursor table'));
  if(literals.some(literal=>literal.camera!==camera))return Promise.reject(Error('Native literals belong to another camera'));
  if(gpuTone&&gpuTone.field!==field)return Promise.reject(Error('Native region members belong to another source pose'));
  const records:ArticulationRun[]=[],labels=new Map<number,ArticulationLiteral>();
  for(const literal of literals){const index=records.length;labels.set(index,literal);records.push({id:literal.group.id,owner:literal.group.id,sources:[],role:'flank',size:0,lineClass:0,key:0xffffffff-index,points:[],anchorOnly:true});}
  // The physical adapter already owns each prepared camera transaction. It
  // supplies no stacked-camera dependencies; concurrent poses must stay
  // independent instead of being grouped by their reusable camera name.
  // Unlike enqueueSource's live table, this optional table must be the
  // physical caller's retained immutable pose, held through the next encode.
  return new Promise((resolve,reject)=>{const job:Pending={runs:records,literals:labels,gpuTone,field,width,height,rect,resolve,reject,deferredSource:{source,classes,camera,owners,partAnchorTable},ready:true};this.active.set(job,new Set());this.pending.push(job);});
 }
 private capturePartAnchors(encoder:GPUCommandEncoder,job:Pending,table:SourcePartAnchorTable|undefined):void{
  if(!table||job.partAnchorSnapshot)return;
  const words=table.buffer.size/4;
  const snapshot=this.own(job,this.device.createBuffer({label:'MAIN camera-owned declared part anchors',size:(PART_ANCHOR_HEADER_WORDS+words)*4,usage:GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST}));
  this.device.queue.writeBuffer(snapshot,0,Uint32Array.of(words,table.headSlots,table.headBase,table.recordBase,table.recordWords,0,0,0));
  encoder.copyBufferToBuffer(table.buffer,0,snapshot,PART_ANCHOR_HEADER_WORDS*4,table.buffer.size);
  job.partAnchorSnapshot=snapshot;
 }
 /** Real source retirement only. Offscreen sources remain in this inventory;
  * a camera cut never clears another surviving feature's correspondence. */
 retainSources(domains:ReadonlySet<number>,producers?:ReadonlySet<number>):void{
  const same=(before:Uint32Array|undefined,after:readonly number[])=>before?.length===after.length&&after.every((value,index)=>before[index]===value);
  const next=[...domains].sort((a,b)=>a-b),rows=producers&&[...producers].sort((a,b)=>a-b);
  if(this.sourceLife&&same(this.sourceLife.domains,next)&&((!rows&&!this.sourceLife.producers)||(rows&&same(this.sourceLife.producers,rows))))return;
  this.sourceRuns.retainSources(domains);
  this.sourceLife={domains:Uint32Array.from(next),producers:rows&&Uint32Array.from(rows)};this.lifecycle++;
 }
 /** Inventory of actual camera lifetimes, including offscreen cameras.
  * A removed camera's submitted jobs drain before its dictionary retires. */
 retainCameras(cameras:ReadonlySet<string>):void{
  this.liveCameras=new Set(cameras);this.retireCameraMemos();
  for(const camera of this.rowHints.keys())if(!cameras.has(camera))this.rowHints.delete(camera);
 }
 private retireCameraMemos():void{
  if(!this.liveCameras)return;
  for(const [camera,memo]of this.cameraMemos){
   if(this.liveCameras.has(camera)||[...this.active.keys()].some(job=>(job.gpu?.camera??job.deferredSource?.camera)===camera))continue;
   this.cameraMemos.delete(camera);this.retiredMemo.add(memo.buffer);
   this.submittedTasks.push(()=>{memo.buffer.destroy();this.retiredMemo.delete(memo.buffer);});
  }
 }
 private ensureMemo(encoder:GPUCommandEncoder,camera:string,needed:number,transientWords:number,epochs:Uint32Array,hasOwners:boolean):CameraMemo{
  const previous=this.cameraMemos.get(camera);let slots=previous?.slots??256;while(slots<needed)slots*=2;
  const sameLife=previous?.lifecycle===this.lifecycle&&previous.hasOwners===hasOwners&&previous.epochs.length===epochs.length&&previous.epochs.every((value,index)=>epochs[index]===value);
  const size=(MEMO_AT+slots*MEMO_WORDS+transientWords)*4;
  if(size>Math.min(this.device.limits.maxStorageBufferBindingSize,this.device.limits.maxBufferSize))throw Error(`Source correspondence exceeds the device storage extent (${slots} slots, ${needed} requested, ${transientWords} transient words)`);
  if(previous&&slots===previous.slots&&size<=previous.buffer.size&&sameLife)return previous;
  const buffer=this.device.createBuffer({label:`MAIN exact source correspondence ${camera}`,size,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
  const result={buffer,slots,lifecycle:this.lifecycle,epochs:epochs.slice(),hasOwners};this.cameraMemos.set(camera,result);
  if(previous){
   if(slots===previous.slots&&sameLife)encoder.copyBufferToBuffer(previous.buffer,0,buffer,0,(MEMO_AT+slots*MEMO_WORDS)*4);
   else{
    const groups=Math.ceil(previous.slots/64),columns=Math.min(groups,this.device.limits.maxComputeWorkgroupsPerDimension);
    const domains=this.sourceLife?.domains??new Uint32Array(),producers=this.sourceLife?.producers;
    const life=new Uint32Array(Math.max(1,domains.length+epochs.length+(producers?.length??0)));life.set(domains);life.set(epochs,domains.length);if(producers)life.set(producers,domains.length+epochs.length);
    const liveBuffer=this.upload(life),uniform=this.upload(new Uint32Array([previous.slots,slots,MEMO_AT,columns,domains.length,epochs.length,producers?.length??0xffffffff,Number(!!this.sourceLife)|Number(hasOwners)*2]),GPUBufferUsage.UNIFORM);
    const group=this.device.createBindGroup({layout:this.rehashMemo.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:uniform}},{binding:1,resource:{buffer:previous.buffer}},{binding:2,resource:{buffer}},{binding:3,resource:{buffer:liveBuffer}}]});
    const pass=encoder.beginComputePass({label:'MAIN retained source-key dictionary growth'});pass.setPipeline(this.rehashMemo);pass.setBindGroup(0,group);pass.dispatchWorkgroups(columns,Math.ceil(groups/columns));pass.end();
    for(const buffer of [uniform,liveBuffer]){this.retiredMemo.add(buffer);this.submittedTasks.push(()=>{buffer.destroy();this.retiredMemo.delete(buffer);});}
   }
   this.retiredMemo.add(previous.buffer);this.submittedTasks.push(()=>{previous.buffer.destroy();this.retiredMemo.delete(previous.buffer);});
  }
  return result;
 }
 private own(job:Pending,buffer:GPUBuffer):GPUBuffer{this.active.get(job)!.add(buffer);return buffer;}
 private scratch(job:Pending,kind:'rows'|'tape',size:number,usage:GPUBufferUsageFlags):GPUBuffer{
  let buffer=this.spareScratch.get(kind);this.spareScratch.delete(kind);
  if(buffer&&buffer.size<size){buffer.destroy();this.scratchKinds.delete(buffer);buffer=undefined;}
  if(!buffer){buffer=this.device.createBuffer({label:`MAIN reusable native ${kind}`,size,usage});this.scratchKinds.set(buffer,kind);}
  return this.own(job,buffer);
 }
 private release(job:Pending,reusable=false):void{
  for(const buffer of this.active.get(job)??[]){
   const kind=this.scratchKinds.get(buffer),spare=kind&&this.spareScratch.get(kind);
   if(kind&&reusable&&!this.disposed&&(!spare||spare.size<buffer.size)){
    if(spare){spare.destroy();this.scratchKinds.delete(spare);}this.spareScratch.set(kind,buffer);
   }else{buffer.destroy();this.scratchKinds.delete(buffer);}
  }
  this.active.delete(job);
  this.retireCameraMemos();
 }
 encode(encoder:GPUCommandEncoder):void{
  if(this.disposed)return;
  for(const request of this.exports.splice(0)){request.encode(encoder);this.submittedTasks.push(request.submitted);}
  const queue=this.pending.splice(0),groups=new Map<CameraCohort,Pending[]>(),single:Pending[]=[];
  for(const job of queue){if(!this.active.has(job))continue;const cohort=job.cohort;if(cohort&&cohort.jobs.size>1){const list=groups.get(cohort)??[];list.push(job);groups.set(cohort,list);}else single.push(job);}
  const requeue=(job:Pending)=>{if(this.active.has(job)&&!this.pending.includes(job))this.pending.push(job);};
  const run=(job:Pending)=>{try{
   if(job.deferredSource&&!job.gpu){
    const d=job.deferredSource;this.capturePartAnchors(encoder,job,d.partAnchorTable);
    const generated=this.sourceRuns.encode(encoder,d.source,d.classes,job.rect,job.width,job.height,job.runs.length,job.field);
    // Compact source runs no longer reserve three slots for every relation.
    // Semantic labels follow this job's actual allocation, including growth
    // between enqueue and the host encoder that services it.
    const labels=job.literals as Map<number,ArticulationLiteral>,values=[...labels.values()];labels.clear();
    values.forEach((literal,index)=>labels.set(generated.capacity+index,literal));
    job.gpu={runs:generated,source:d.source,camera:d.camera,owners:d.owners,count:generated.capacity+job.runs.length+(job.gpuTone?.capacity??0)};
   }
   this.encodeJob(encoder,job);
  }catch(error){job.gpu?.runs.dispose();this.release(job);this.abortCohort(job,error);}};
  for(const job of single){if(this.dependenciesReady(job,encoder))run(job);else requeue(job);}
  for(const [cohort,candidates] of groups){
   const members=[...cohort.jobs].filter(job=>this.active.has(job));
   // Literal readiness is part of the cohort contract. Do not seal an upper
   // camera before a lower camera's same-pose literal packet has arrived.
   if(!members.every(job=>job.ready)){for(const job of candidates)requeue(job);continue;}
   for(const job of [...candidates].sort((a,b)=>(a.occludedBy?.length??0)-(b.occludedBy?.length??0))){
    if(!this.dependenciesReady(job,encoder)){requeue(job);continue;}
    run(job);
   }
  }
 }
 /** Allocation hints follow this camera's current row distribution. They
  * never suppress a word; overflow reports complete counts and replays the
  * same retained pose. Do not retain a global worst row for every row. */
 private rememberRows(camera:string,width:number,height:number,observed:Uint32Array,minima:Uint32Array):Uint32Array{
  const exact=new Uint32Array(observed.length),room=new Uint32Array(observed.length);
  let bytes=0;
  for(let row=0;row<observed.length;row++){
   exact[row]=Math.max(1,observed[row]!,minima[row]!);
   room[row]=Math.max(ROW_CAP,Math.ceil(exact[row]!*1.25/256)*256);bytes+=room[row]!*ROW_WORDS*4;
  }
  const limit=Math.min(this.device.limits.maxStorageBufferBindingSize,this.device.limits.maxBufferSize);
  const capacities=bytes<=limit?room:exact;
  this.rowHints.set(camera,{width,height,capacities});return capacities;
 }
 private encodeJob(encoder:GPUCommandEncoder,job:Pending):void{
  if(!job.runs.length&&!job.gpu){this.release(job);job.settled=true;job.resolve({groups:[],unresolved:[]});if(job.cohort)this.completeCohort(job.cohort);return;}
  const own=(buffer:GPUBuffer)=>this.own(job,buffer);
  const rows=Math.ceil(job.height/17),capacity=job.outputCapacity??this.tapeCapacity,runData=new Uint32Array(job.runs.length*RESIDENT_RUN_WORDS),pointData:number[]=[];
  const camera=job.gpu?.camera??job.runs[0]?.owner.split('/')[0]??'native-input';
  // Authored words are already countable before dispatch. Generated words
  // report their exact peak in the header and retry the same pose if needed.
  const literalCounts=new Uint32Array(rows);
  for(const literal of job.literals.values())for(const atom of literal.group.occurrences)if(atom.row>=0&&atom.row<rows)literalCounts[atom.row]++;
  const storageLimit=Math.min(this.device.limits.maxStorageBufferBindingSize,this.device.limits.maxBufferSize);
  const hint=this.rowHints.get(camera),previous=job.rowCapacities??(hint?.width===job.width&&hint.height===job.height?hint.capacities:undefined);
  const rowOffsets=new Uint32Array(rows),rowCapacities=new Uint32Array(rows);
  const initial=Math.max(1,Math.min(ROW_CAP,Math.floor(storageLimit/(rows*ROW_WORDS*4))));
  let rowCapacity=1,rowSlots=0;
  for(let row=0;row<rows;row++){
   rowOffsets[row]=rowSlots;const count=Math.max(1,literalCounts[row]!,previous?.[row]??initial);
   rowCapacities[row]=count;rowSlots+=count;rowCapacity=Math.max(rowCapacity,count);
  }
  const rowBytes=rowSlots*ROW_WORDS*4;
  if(rowBytes>storageLimit)throw Error(`Complete native row records need device-sized page chunks (${rows} rows, ${rowSlots} total words)`);
  for(const [i,run]of job.runs.entries()){
   const at=i*RESIDENT_RUN_WORDS,role=['rim_cap','shoulder','flank','return_base'].indexOf(run.role)|(run.lineClass<<8)|(run.anchorOnly?0x10000:0)|(run.round?0x20000:0)|(run.hard?0x40000:0)|(run.upright?0x80000:0);runData.set([pointData.length/8,run.points.length,role,run.size,run.key,...(run.source??[0,0]),run.generation??run.key],at);
   if(!run.anchorOnly)runData.set([run.key,run.lineClass],at+8);
   else runData.set([0,0,run.key,0x10000000|run.lineClass],at+12);
   const part=run.anchorOnly&&run.part&&this.parts.get(run.part);if(part){runData.set([part.first,part.count,Math.round(-part.width/2)>>>0,run.role==='return_base'?1-part.rows:0],at+8);runData.set([part.paperFirst,part.paperCount,run.key,0x10000000|run.lineClass],at+12);}
   for(const p of run.points)pointData.push(p.x,p.y,p.depth,p.parameter,Number(p.visible),p.reciprocalW??1,0,0);
  }
  const values=Float32Array.from(pointData),bits=new Uint32Array(values.buffer);for(let i=4;i<bits.length;i+=8)bits[i]=pointData[i]!;
  // Dictionary capacity depends on how many features exist, never the
  // numeric magnitude of an original source address/hash.
  let needed=1;for(const run of job.runs)if(run.points.length&&!run.round)needed+=2;
  const paperWords=rows*Math.ceil(job.width/32),runCount=job.gpu?.count??job.runs.length;
  let epochExtent=1;for(const owner of job.gpu?.owners.keys()??[])epochExtent=Math.max(epochExtent,owner+1);
  const epochs=new Uint32Array(epochExtent).fill(0xffffffff);for(const [owner,reference]of job.gpu?.owners??[])epochs[owner]=Math.floor(reference/4294967296)>>>0;








  const orderWords=rowCapacity>ROW_CAP?rowSlots:0;
  const proposalCapacity=job.proposalCapacity??this.proposalCapacity;
  const farPlanWords=runCount*FAR_PLAN_WORDS;
  // After the drawn table: two loss/copy words and one far-playback status
  // word per run, then the selection
  // cell grids (gridWrite), cleared every frame.
  // Then the ripple grids (rippleCell): two of 32 px x 2-row cells.

 const gridWords=2*rows*(Math.ceil(job.width/16)+1)+2*(Math.ceil(rows/2)+2)*(Math.ceil(job.width/32)+2)+LIMB_SLOTS*LIMB_WORDS+sourcePartGroupSlots(runCount)*SOURCE_PART_HEAD_WORDS+runCount*SOURCE_PART_GROUP_WORDS;
  const rowLayoutWords=rows*2;
  // Seven source families each retain original source IDs; labels and tone
  // have their own typed tails. Metadata stores one base/count pair per
  // family in the same correspondence scratch, not a new binding.
  const activeQueueWords=job.gpu?ACTIVE_QUEUE_META_WORDS+job.gpu.runs.capacity*ACTIVE_SOURCE_QUEUE_FAMILIES+job.runs.length+(job.gpuTone?.capacity??0):0;
  const partAnchorWords=(job.partAnchorSnapshot?.size??0)/4;
  const layout=mainTransientLayout({runCount,epochWords:epochs.length,paperWords,gridWords,orderWords,rowLayoutWords,proposalCapacity,farPlanWords,activeQueueWords,partAnchorWords});
  const {transientWords,proposalWords}=layout;
  const memo=this.ensureMemo(encoder,camera,Math.max(job.memoSlots??0,job.gpu?8192:needed),transientWords,epochs,!!job.gpu);
  if(rows+4>MEMO_AT-1)throw Error('Native page exceeds the correspondence header extent');
  if(rows+4>SKY_AT||SKY_AT+SKY_COLUMNS*3>MEMO_AT-1||job.width>SKY_COLUMNS*16)throw Error('Native page exceeds the far skyline profile extent');
  const curveGroups=Math.max(1,Math.ceil(runCount/64)),curveColumns=Math.min(curveGroups,this.device.limits.maxComputeWorkgroupsPerDimension);
  const params=new Uint32Array(48),floats=new Float32Array(params.buffer);params.set([job.width,job.height,runCount,capacity]);floats.set([job.rect.x,job.rect.y,job.rect.width,job.rect.height,job.field.width,job.field.height,MEMO_AT,memo.slots],4);floats.set(job.field.worldFromClip,12);params[28]=rowCapacity;params[29]=curveColumns;params[30]=epochs.length;
  const transientBase=MEMO_AT+memo.slots*MEMO_WORDS,proposalBase=transientBase+layout.proposalBase,proposalStatus=transientBase+layout.proposalStatus;
  params.set([proposalBase,proposalCapacity,proposalStatus,this.device.limits.maxComputeWorkgroupsPerDimension],32);
  params.set(FAR_TEXTURE_PROBE_BOUNDS,36);
  // Word39 was unused. The compact per-run plan follows the proposal status;
  // it does not overlap proposal records or move any earlier scratch owner.
  params[39]=transientBase+layout.farPlanBase;
  const activeQueueBase=job.gpu?transientBase+layout.activeQueueBase:0;
  params[40]=activeQueueBase;params[41]=job.gpu?.runs.capacity??0;params[42]=job.gpu?job.runs.length:0;params[43]=job.gpuTone?.capacity??0;
  params[44]=transientBase+layout.partAnchorBase;params[45]=partAnchorWords;
  const rowLayoutBase=transientBase+layout.rowLayoutBase,orderBase=transientBase+layout.orderBase;
  let ordered=orderBase;for(let span=ROW_CAP;span<rowCapacity;span*=2)ordered=orderBase+(ordered===orderBase?orderWords:0);
  params[31]=orderWords?ordered:0;
  // Rebase the leading-state cursor table to THIS page stride; the pinned
  // grammar and native decisions remain resident across page sizes.
  let gaps=this.gapTables.get(job.width);
  if(!gaps){const extra=this.packed.gaps.subarray((this.maxWidth+1)*2),gapData=new Uint32Array((job.width+1)*2+extra.length);gapData.set(this.packed.gaps.subarray(0,job.width+1));gapData.set(this.packed.gaps.subarray(this.maxWidth+1,this.maxWidth+job.width+2),job.width+1);gapData.set(extra,(job.width+1)*2);gaps=this.upload(gapData);this.gapTables.set(job.width,gaps);}
  const uniform=own(this.upload(params,GPUBufferUsage.UNIFORM)),points=job.gpu?.runs.points??own(this.upload(values));
  let runs=job.gpu?.runs.runs??own(this.upload(runData));
  // Region membership and its count stay on the GPU. The complete cell bound
  // reserves source-address slots, not a glyph quota or a CPU readback.
  const runBytes=runCount*RESIDENT_RUN_WORDS*4;
  if(runs.size<runBytes){
   if(runBytes>storageLimit)throw Error('Complete native literals need device-sized page chunks');
   const expanded=own(this.device.createBuffer({label:'MAIN retained runs plus compact literals',size:runBytes,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}));
   encoder.copyBufferToBuffer(runs,0,expanded,0,job.gpu!.runs.capacity*RESIDENT_RUN_WORDS*4);runs=expanded;
  }
  if(job.gpu&&runData.length)this.device.queue.writeBuffer(runs,job.gpu.runs.capacity*RESIDENT_RUN_WORDS*4,runData);
  if(job.gpuTone)encoder.clearBuffer(runs,(job.gpu!.runs.capacity+job.runs.length)*RESIDENT_RUN_WORDS*4,job.gpuTone.capacity*RESIDENT_RUN_WORDS*4);
  const initialCounts=new Uint32Array(rows+4);
  const rowWords=this.scratch(job,'rows',rowBytes,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST);
  const seededRows=new Map<number,number[]>();
  // Authored native text is source input, not OCR or a glyph decision. Seed
  // its immutable pens first so the SAME GPU priority pass protects labels.
  for(const [runIndex,literal]of job.literals)for(const [ordinal,atom]of literal.group.occurrences.entries()){
   const glyph=this.fontAddresses.get(atom.text)!;if(atom.row<0||atom.row>=rows||atom.x<0||atom.x+this.packed.font[glyph]!.advance>job.width)continue;
   initialCounts[atom.row]!++;
   const record=new Uint32Array(ROW_WORDS);record.set([runIndex,ordinal,0,atom.x,atom.row,glyph,1,this.packed.font[glyph]!.advance,literal.tone?0:0xffffffff,literal.tone?3:2,atom.foreground??0xff000000]);
   if(literal.depth!==undefined){new Float32Array(record.buffer)[11]=literal.depth;record[12]=1;}
   const seed=seededRows.get(atom.row)??[];seed.push(...record);seededRows.set(atom.row,seed);
  }
  // A compact upload per occupied row, not one driver call per tone glyph.
  for(const [row,records]of seededRows)this.device.queue.writeBuffer(rowWords,rowOffsets[row]!*ROW_WORDS*4,Uint32Array.from(records));
  for(let row=0;row<rows;row++)initialCounts[rows+3]=Math.max(initialCounts[rows+3]!,initialCounts[row]!);
  const counts=memo.buffer;
  if(job.gpu){
   // The queue count and original run IDs live in this camera's existing
   // correspondence scratch. No new bind-group storage slot is introduced.
   encoder.clearBuffer(counts,activeQueueBase*4,activeQueueWords*4);
  }
  encoder.clearBuffer(counts,proposalStatus*4,16);
  // These commands, unlike queue.writeBuffer, remain ordered separately for
  // every camera job sharing this persistent correspondence buffer.
  const seed=own(this.upload(initialCounts,GPUBufferUsage.COPY_SRC));encoder.copyBufferToBuffer(seed,0,counts,0,initialCounts.byteLength);
  const rowLayout=new Uint32Array(rowLayoutWords);rowLayout.set(rowOffsets);rowLayout.set(rowCapacities,rows);
  const layoutSeed=own(this.upload(rowLayout,GPUBufferUsage.COPY_SRC));encoder.copyBufferToBuffer(layoutSeed,0,counts,rowLayoutBase*4,rowLayout.byteLength);
  const epochSeed=own(this.upload(epochs,GPUBufferUsage.COPY_SRC));
  encoder.copyBufferToBuffer(epochSeed,0,counts,(MEMO_AT+memo.slots*MEMO_WORDS+runCount)*4,epochs.byteLength);
  // Protected paper, then the per-owner drawn table (visible run bit, words).
  encoder.clearBuffer(counts,(MEMO_AT+memo.slots*MEMO_WORDS+runCount+epochs.length)*4,(paperWords+epochs.length)*4);
  encoder.clearBuffer(counts,(MEMO_AT+memo.slots*MEMO_WORDS+runCount+epochs.length+paperWords+epochs.length)*4,(runCount*3+gridWords)*4);
  if(job.partAnchorSnapshot)encoder.copyBufferToBuffer(job.partAnchorSnapshot,0,counts,params[44]!*4,job.partAnchorSnapshot.size);
  // The merged far skyline profile of this frame (three column tables in the header words).
  encoder.clearBuffer(counts,SKY_AT*4,SKY_COLUMNS*3*4);
  if(MAIN_PROBES)encoder.clearBuffer(counts,(rows+4)*4,96*4);
  if(job.gpuTone){
   const tone=job.gpuTone;if(tone.format!=='whole-passages-v1')throw Error('Native region members must be whole passages');
   const groups=Math.max(1,Math.ceil(tone.capacity/64)),columns=Math.min(groups,this.device.limits.maxComputeWorkgroupsPerDimension);
   const domains=this.sourceLife?.domains??new Uint32Array(),data=new Uint32Array(16),floats=new Float32Array(data.buffer);data.set([job.width,job.height,rowCapacity,job.gpu!.runs.capacity+job.runs.length,tone.capacity,this.packed.toneWords,0,columns]);floats.set([job.rect.x,job.rect.y,job.rect.width,job.rect.height],8);data.set([domains.length,TONE_PASSAGES.length,rowLayoutBase],12);
   const uniform=own(this.upload(data,GPUBufferUsage.UNIFORM)),sources=own(this.upload(domains)),group=this.device.createBindGroup({layout:this.toneSeed.getBindGroupLayout(0),entries:[uniform,tone.buffer,runs,rowWords,counts,sources,this.resident[1]!].map((buffer,binding)=>({binding,resource:{buffer}}))});
   const pass=encoder.beginComputePass({label:'MAIN resident caster-region words into native selection'});pass.setPipeline(this.toneSeed);pass.setBindGroup(0,group);pass.dispatchWorkgroups(columns,Math.ceil(groups/columns));pass.end();
  }
  let activeArguments:GPUBuffer|undefined;
  let rebuildActiveQueues:(()=>void)|undefined;
  if(job.gpu){
   const sourceQueueBase=activeQueueBase+ACTIVE_QUEUE_META_WORDS,labelQueueBase=sourceQueueBase+job.gpu.runs.capacity*ACTIVE_SOURCE_QUEUE_FAMILIES,toneQueueBase=labelQueueBase+job.runs.length;
   const queueMeta=new Uint32Array(ACTIVE_QUEUE_META_WORDS);
   for(let family=0;family<ACTIVE_SOURCE_QUEUE_FAMILIES;family++)queueMeta[family*2]=sourceQueueBase+family*job.gpu.runs.capacity;
   queueMeta[ACTIVE_QUEUE_LABEL*2]=labelQueueBase;queueMeta[ACTIVE_QUEUE_TONE*2]=toneQueueBase;
   const queueSeed=own(this.upload(queueMeta,GPUBufferUsage.COPY_SRC));
   activeArguments=own(this.device.createBuffer({label:'MAIN typed source-run indirect arguments',size:ACTIVE_ARGUMENT_WORDS*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.INDIRECT|GPUBufferUsage.COPY_DST}));
   const tone=job.gpuTone?.buffer??own(this.device.createBuffer({label:'MAIN empty tone member list',size:4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}));
   if(!job.gpuTone)this.device.queue.writeBuffer(tone,0,new Uint32Array([0]));
   const compactGroup=this.device.createBindGroup({layout:this.activeRuns.getBindGroupLayout(0),entries:[
    {binding:0,resource:{buffer:uniform}},{binding:1,resource:{buffer:runs}},{binding:2,resource:{buffer:counts}},{binding:3,resource:{buffer:tone}},
   ]});
   const argsGroup=this.device.createBindGroup({layout:this.activeRunArgs.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:uniform}},{binding:1,resource:{buffer:counts}},{binding:2,resource:{buffer:activeArguments}}]});
   const compactGroups=Math.max(1,Math.ceil(runCount/64)),compactLimit=this.device.limits.maxComputeWorkgroupsPerDimension;
   const compactRows=Math.max(1,Math.ceil(compactGroups/compactLimit)),compactColumns=Math.max(1,Math.ceil(compactGroups/compactRows));
   if(compactRows>compactLimit||compactColumns>compactLimit)throw Error('Active source-run compaction exceeds device workgroup limit');
   rebuildActiveQueues=()=>{
    encoder.clearBuffer(counts,activeQueueBase*4,activeQueueWords*4);
    encoder.copyBufferToBuffer(queueSeed,0,counts,activeQueueBase*4,queueMeta.byteLength);
    const compact=encoder.beginComputePass({label:'MAIN typed source-run queue compaction'});compact.setPipeline(this.activeRuns);compact.setBindGroup(0,compactGroup);compact.dispatchWorkgroups(compactColumns,compactRows);compact.end();
    const args=encoder.beginComputePass({label:'MAIN typed source-run indirect arguments'});args.setPipeline(this.activeRunArgs);args.setBindGroup(0,argsGroup);args.dispatchWorkgroups(1);args.end();
   };
   rebuildActiveQueues();
  }
  const outputBytes=(capacity*WORD_WORDS+rows*(job.width*2+1))*4;
  if(outputBytes>Math.min(this.device.limits.maxStorageBufferBindingSize,this.device.limits.maxBufferSize))throw Error('Complete native tape needs device-sized page chunks');
  const output=this.scratch(job,'tape',outputBytes,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST);
  // Record slots are written before use and count-sized on export. Only
  // the cursor/Fenwick region needs zeroing when a completed tape is reused;
  // clear THIS page's range in encoder order, including after width changes.
  encoder.clearBuffer(output,capacity*WORD_WORDS*4,rows*(job.width*2+1)*4);
  const proposalStatusOffset=job.gpu?80+epochs.length*4:16;
  const rowCountsOffset=proposalStatusOffset+16,headerBytes=rowCountsOffset+rows*4;
  // Diagnostic counters are not a publication dependency. Their shader is
  // compiled out in normal play; do not clear/copy/map/materialize them there.
  const probeBytes=job.gpu&&MAIN_PROBES?96*4:0;
  const header=own(this.device.createBuffer({label:'MAIN compact native result packet',size:headerBytes+(job.gpu?probeBytes:capacity*WORD_WORDS*4),usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}));
  if(job.gpu)encoder.copyBufferToBuffer(counts,(MEMO_AT+memo.slots*MEMO_WORDS+runCount+epochs.length+paperWords)*4,header,80,epochs.length*4);
  const buffers=[uniform,runs,points,this.resident[0]!,this.resident[1]!,gaps,counts,rowWords,output];
  const bindings=[0,1,2,3,4,6,7,8,9];
  const entries:GPUBindGroupEntry[]=buffers.map((buffer,index)=>({binding:bindings[index]!,resource:{buffer}}));entries.push({binding:10,resource:job.field.textures[0]!.createView()},{binding:11,resource:job.field.textures[1]!.createView()},{binding:12,resource:job.field.textures[3]!.createView()},{binding:13,resource:job.field.textures[4]!.createView()},{binding:5,resource:formDepths(this.device).view(job.field)});
  const stacked=this.occluderFor(encoder,job,rows,own);entries.push({binding:14,resource:stacked.mask.createView()},{binding:15,resource:this.materialInkView()});
  const group=this.device.createBindGroup({layout:this.layout,entries});


  const visibilityArguments=own(this.device.createBuffer({label:'MAIN active word visibility arguments',size:16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.INDIRECT}));
  const visibilityDispatchGroup=this.device.createBindGroup({layout:this.visibilityDispatch.getBindGroupLayout(0),entries:[uniform,counts,visibilityArguments].map((buffer,binding)=>({binding,resource:{buffer}}))});
  const construction=[...this.far,...this.curves];
  const activeRunPipelines=new Set<GPUComputePipeline>([this.sourcePartInit,this.sourcePartReduce,this.claimMemo,this.resolveMemo,this.skyline,...this.far,...this.curves]);
  const standingGroups=standingDispatch(Math.max(1,runCount),this.device.limits.maxComputeWorkgroupsPerDimension);
  const activeArgumentOffset=(pipeline:GPUComputePipeline):number|undefined=>{

   if(pipeline===this.sourcePartInit||pipeline===this.sourcePartReduce||pipeline===this.claimMemo||pipeline===this.resolveMemo||pipeline===this.skyline)return 0;
   const farIndex=this.far.indexOf(pipeline);if(farIndex>=0)return (4+farIndex)*12;
   const curveIndex=this.curves.indexOf(pipeline);if(curveIndex>=0)return (1+curveIndex)*12;
   return undefined;
  };
  for(const [pipeline,x,y]of [[this.sourcePartInit,curveColumns,Math.ceil(curveGroups/curveColumns)],[this.sourcePartReduce,curveColumns,Math.ceil(curveGroups/curveColumns)],[this.claimMemo,curveColumns,Math.ceil(curveGroups/curveColumns)],[this.resolveMemo,curveColumns,Math.ceil(curveGroups/curveColumns)],[this.reclaimMemo,Math.min(curveColumns,Math.ceil(memo.slots/64)),Math.ceil(memo.slots/(curveColumns*64))],[this.skyline,curveColumns,Math.ceil(curveGroups/curveColumns)],...construction.map(pipeline=>{const [x,y]=pipeline===this.far[2]?standingGroups:[curveColumns,Math.ceil(curveGroups/curveColumns)];return [pipeline,x,y] as const;}),[this.wordVisibility,0,0],[this.place,rows,1]] as const){
   if(pipeline===this.wordVisibility){
    const pass=encoder.beginComputePass({label:'MAIN count-sized word visibility dispatch'});pass.setPipeline(this.visibilityDispatch);pass.setBindGroup(0,visibilityDispatchGroup);pass.dispatchWorkgroups(1);pass.end();
   }
   if(pipeline===this.place&&orderWords){
    const orderPass=(pipeline:GPUComputePipeline,data:Uint32Array,x:number,label:string)=>{
     const uniform=own(this.upload(data,GPUBufferUsage.UNIFORM));
     const bindings=this.device.createBindGroup({layout:this.layout,entries:entries.map(entry=>entry.binding===0?{binding:0,resource:{buffer:uniform}}:entry)});
     const pass=encoder.beginComputePass({label});pass.setPipeline(pipeline);pass.setBindGroup(0,bindings);pass.dispatchWorkgroups(x,rows);pass.end();
    };
    const seed=params.slice();seed[31]=orderBase;
    orderPass(this.order,seed,Math.ceil(rowCapacity/ROW_CAP),'MAIN parallel dense native row tiles');
    let from=orderBase;
    for(let span=ROW_CAP;span<rowCapacity;span*=2){
     const to=orderBase+(from===orderBase?orderWords:0),merge=params.slice();merge[2]=from;merge[3]=span;merge[31]=to;
     orderPass(this.orderMerge,merge,Math.ceil(rowCapacity/64),'MAIN parallel dense native row merge');from=to;
    }
   }
   if(pipeline===this.place&&stacked.upper){
    const fpData=new Uint32Array([job.width,rows,capacity,0]),fpBuffer=own(this.device.createBuffer({label:'MAIN stacked camera foreign parameters',size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}));this.device.queue.writeBuffer(fpBuffer,0,fpData);
    const foreign=this.occluderPipelines!.foreign,f=encoder.beginComputePass({label:'MAIN stacked camera foreign neighbours'});f.setPipeline(foreign);
    f.setBindGroup(0,this.device.createBindGroup({layout:foreign.getBindGroupLayout(0),entries:[{binding:5,resource:{buffer:fpBuffer}},{binding:6,resource:{buffer:output}},{binding:7,resource:{buffer:counts}},{binding:3,resource:{buffer:stacked.upper.words}},{binding:4,resource:{buffer:stacked.upper.count}}]}));
    f.dispatchWorkgroups(Math.ceil(rows/64));f.end();
   }
   const pass=encoder.beginComputePass({label:pipeline===this.sourcePartInit?'MAIN source-part key initialization':pipeline===this.sourcePartReduce?'MAIN source-part exact-key reduction':pipeline===this.skyline?'MAIN merged far skyline profile':pipeline===this.claimMemo||pipeline===this.resolveMemo||pipeline===this.reclaimMemo?'MAIN exact source-key correspondence':construction.includes(pipeline)?'MAIN featurecurves compiled playback':pipeline===this.wordVisibility?'MAIN parallel whole-word visibility':'MAIN featureplace fixed native cursors'});pass.setPipeline(pipeline);pass.setBindGroup(0,group);const argumentOffset=activeArgumentOffset(pipeline);if(pipeline===this.wordVisibility)pass.dispatchWorkgroupsIndirect(visibilityArguments,0);else if(job.gpu&&activeArguments&&activeRunPipelines.has(pipeline)&&argumentOffset!==undefined)pass.dispatchWorkgroupsIndirect(activeArguments,argumentOffset);else pass.dispatchWorkgroups(x,y);pass.end();
   if(pipeline===this.far[0])rebuildActiveQueues?.();
   if(pipeline===this.place){const tone=encoder.beginComputePass({label:'MAIN parallel region tone words'});tone.setPipeline(this.tonePlace);tone.setBindGroup(0,group);tone.dispatchWorkgroups(x,y);tone.end();}
  }

  let sealed:Omit<GpuNativeTape,'seal'>|undefined;
  if(job.gpu){
   // Same metrics address as the resident grammar header; no second bank.
   const metrics=12+this.packed.instructions.length+this.packed.articulatedInstructions.length+this.packed.hardLineInstructions.length+this.packed.partAtoms.length+this.packed.partPaper.length+this.packed.circleProgram.length;
   if(stacked.upper){
    const fpData=new Uint32Array([job.width,rows,capacity,0]),fpBuffer=own(this.device.createBuffer({label:'MAIN stacked camera foreign parameters',size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}));this.device.queue.writeBuffer(fpBuffer,0,fpData);
    const remove=this.occluderPipelines!.foreignRemove,f=encoder.beginComputePass({label:'MAIN stacked camera foreign neighbours removed'});f.setPipeline(remove);
    f.setBindGroup(0,this.device.createBindGroup({layout:remove.getBindGroupLayout(0),entries:[{binding:5,resource:{buffer:fpBuffer}},{binding:6,resource:{buffer:output}},{binding:3,resource:{buffer:stacked.upper.words}},{binding:4,resource:{buffer:stacked.upper.count}}]}));
    f.dispatchWorkgroups(Math.ceil(rows/64));f.end();
   }
   sealed=this.seal.encode(encoder,{width:job.width,height:job.height,rows,wordCapacity:capacity,scalarCapacity:Math.min(Math.max(capacity*6,job.scalarCapacity??this.scalarCapacity),job.width*rows),metrics,fontAddressCount:this.packed.font.length,placed:output,grammar:this.resident[0]!,words:this.resident[1]!,gaps,relaxedGaps:!!stacked.upper},own);
   encoder.copyBufferToBuffer(sealed.count,0,header,48,32);
  }
  encoder.copyBufferToBuffer(counts,rows*4,header,0,16);
  encoder.copyBufferToBuffer(counts,proposalStatus*4,header,proposalStatusOffset,16);
  // Existing allocation retry receives each row's complete requirement,
  // rather than making every row reserve the busiest one's requirement.
  encoder.copyBufferToBuffer(counts,0,header,rowCountsOffset,rows*4);
  if(probeBytes)encoder.copyBufferToBuffer(counts,(rows+4)*4,header,headerBytes,probeBytes);
  if(job.gpu){encoder.copyBufferToBuffer(job.gpu.runs.status,job.gpu.runs.statusOffset,header,16,16);encoder.copyBufferToBuffer(job.gpu.runs.sourceStatus,0,header,32,16);}
  // One host submission/fence publishes the compact native tape. Source
  // geometry/fields/relations never come back, and no second live tick is
  // needed to schedule a count-then-copy command. Overflow retains this pose.
  if(!job.gpu)encoder.copyBufferToBuffer(output,0,header,headerBytes,capacity*WORD_WORDS*4);
  // HEADER MAP AT SUBMISSION (speed, 2026-11-17): a caller that owns the
  // submission (takeSubmissionMaps) requests this map right after the queue
  // submit, so it resolves with THIS job's work rather than after a later
  // job already queued behind it. Publication still waits for the header.
  let mapping:Promise<void>|undefined;
  this.submissionMaps.push(()=>{if(!this.disposed&&!mapping){mapping=header.mapAsync(GPUMapMode.READ);mapping.catch(()=>{});}});
  this.submittedTasks.push(()=>{void (async()=>{try{if(this.disposed)return;await (mapping??=header.mapAsync(GPUMapMode.READ));if(this.disposed)return;const memory=header.getMappedRange(),h=new Uint32Array(memory),overflow=h[0]!,unresolved=h[1]!,count=h[2]!;
   if(probeBytes&&job.gpu){const g=globalThis as unknown as {__mainLossSplit?:Record<string,number[]>};g.__mainLossSplit={...(g.__mainLossSplit??{}),[job.gpu.camera]:Array.from(new Uint32Array(memory,headerBytes,probeBytes/4))};}
   if(job.gpu){
    job.sourceLines=h[10]!;
    const flags=h[9]!;
    if(flags){this.resolveCohortOverflow(job,{groups:[],unresolved:[],sourceOverflow:{flags,relations:(flags&16)?0:h[11]!,intervals:h[8]!,triangles:(flags&16)?h[11]!:undefined}});return;}
    if(h[6]){
     this.sourceRuns.grow(h[6]!,h[4]!,h[5]!,job.gpu.runs.statusOffset/32);
     // The retained source/visibility owner retries through its normal host
     // scheduling. Never expose this exhausted attempt as partial drawing.
     this.resolveCohortOverflow(job,{groups:[],unresolved:[],sourceOverflow:{flags:4,relations:0,intervals:0}});return;
    }
   }
   const proposed=new Uint32Array(memory,proposalStatusOffset,4)[0]!;
   if(proposed>proposalCapacity){
    job.proposalCapacity=Math.ceil(proposed*1.25/256)*256;this.proposalCapacity=Math.max(this.proposalCapacity,job.proposalCapacity);
    // Same source pose and grammar, a larger numeric packet. The exhausted
    // prefix is never published as though it were a complete drawing.
    this.retryJob(job);return;
   }
   if((overflow&~7)===0&&overflow!==0){
    // The counter names the required range. An allocation attempt is never
    // published as a partial drawing or a quota on source contributions.
    const observed=new Uint32Array(memory,rowCountsOffset,rows);
    if((overflow&1)!==0&&!observed.some((count,row)=>count>rowCapacities[row]!))throw Error('Native row overflow did not report its required extent');
    if((overflow&2)!==0&&count<=capacity)throw Error('Native tape overflow did not report its required extent');
    if((overflow&1)!==0)job.rowCapacities=this.rememberRows(camera,job.width,job.height,observed,literalCounts);
    if((overflow&2)!==0){job.outputCapacity=Math.ceil(count*1.25/256)*256;this.tapeCapacity=Math.max(this.tapeCapacity,job.outputCapacity);}
    // A full dictionary grows from its own demand. Total source runs include
    // hidden fragments which did not claim a key; using that unrelated upper
    // bound jumped directly beyond a binding before trying a fitting table.
    // Rehash and retry the SAME pose, with no key or drawing quota.
    if((overflow&4)!==0)job.memoSlots=memo.slots*2;
    this.retryJob(job);return;
   }
   if(overflow)throw Error(`Native GPU playback capacity overflow ${overflow}`);
   this.rememberRows(camera,job.width,job.height,new Uint32Array(memory,rowCountsOffset,rows),literalCounts);
   if(sealed){
    // Scalar tape capacity only (flag 1): grow from the reported count and
    // requeue the same snapshot, as the row/tape/memo capacities do. A
    // partial row is never sealed or published.
    if(h[16]===1&&h[13]!>sealed.capacity&&h[13]!<=job.width*Math.ceil(job.height/17)){
     job.scalarCapacity=Math.ceil(h[13]!*1.25/256)*256;this.scalarCapacity=Math.max(this.scalarCapacity,job.scalarCapacity);
     this.retryJob(job);return;
    }
    if(h[16])throw Error(`Native scalar tape seal failed ${h[16]}`);
    const scalarCount=h[13]!;
    // Ownership passes to the immutable publication. Do not recycle its raw
    // provenance or scalar tape while the display/export still references it.
    for(const buffer of [output,sealed.words,sealed.count])this.active.get(job)!.delete(buffer);
    this.scratchKinds.delete(output);
    const tape:GpuNativeTape={...sealed,seal:issueGpuNativeTapeSeal({flags:h[16]!,scalarCount,capacity:sealed.capacity})};
    const resident=this.keepTape(job,tape,output,count,scalarCount,unresolved);
    // A seal becomes a dependency only after its header proved complete. The
    // entry is retained for a lower camera that must retry on a later encoder.
    if(job.gpu!.camera&&job.cohort)this.frameSeals.set(this.sealKey(job,job.gpu!.camera),{encoder,sealed:tape,field:job.field,rect:job.rect,cohort:job.cohort});
    if(job.cohort)job.cohort.residents.set(job,resident);
    // The frame's true state: every source owner with a visible run and no
    // committed word is an omitted form. Placement losses between words of
    // drawn owners are cooperation, not omission, and stay out of this list.
    const omitted:string[]=[];
    for(let owner=0;owner<epochs.length;owner++){const drawn=h[20+owner]!;if((drawn&0x80000000)!==0&&(drawn&0x7fffffff)===0)omitted.push(`${job.gpu!.camera}/entity/${owner}: visible source drew no glyph`);}
    job.settled=true;job.resolve({groups:[],sourceLines:job.sourceLines,unresolved:omitted,unresolvedWords:unresolved,resident});
   }else{job.settled=true;job.resolve(this.decode(job,new Uint32Array(memory,headerBytes,count*WORD_WORDS),count,unresolved));}
   header.unmap();job.gpu?.runs.dispose();this.release(job,true);if(job.cohort)this.completeCohort(job.cohort);
  }catch(error){job.gpu?.runs.dispose();this.release(job);this.abortCohort(job,error);}finally{if(header.mapState==='mapped')header.unmap();header.destroy();}})();});
 }
 /** Header maps of the jobs encoded since the last call. The caller runs them
  * right after the queue.submit of the encoder that holds those jobs, and
  * never for an encoder it discards; unrun maps fall back to submitted(). */
 takeSubmissionMaps():(()=>void)[]{return this.submissionMaps.splice(0);}
 submitted():void{this.submissionMaps.length=0;this.sourceRuns.submitted();for(const task of this.submittedTasks.splice(0))task();}
 private keepTape(job:Pending,tape:GpuNativeTape,words:GPUBuffer,count:number,glyphCount:number,unresolved:number):ResidentNativeConstruction{
  let retired=false,reading=false,destroyed=false,result:Promise<ArticulationResult>|undefined;
  const destroy=()=>{if(destroyed||reading)return;destroyed=true;words.destroy();tape.words.destroy();tape.count.destroy();this.tapes.delete(resident);};
  const resident:ResidentNativeConstruction={tape,glyphCount,
   release:()=>{retired=true;destroy();},
   read:()=>{
    if(result)return result;
    if(retired||this.disposed)return Promise.reject(Error('Native display snapshot has retired'));
    reading=true;
    result=new Promise<ArticulationResult>((resolve,reject)=>{
     const staging=this.device.createBuffer({label:'MAIN requested native contribution export',size:Math.max(4,count*WORD_WORDS*4),usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
     let done=false;
     const finish=(error?:unknown,value?:ArticulationResult)=>{if(done)return;done=true;this.cancelExports.delete(cancel);if(staging.mapState==='mapped')staging.unmap();staging.destroy();reading=false;if(error)reject(error);else resolve(value!);if(retired)destroy();};
     const cancel=()=>finish(Error('Native export source was disposed'));this.cancelExports.add(cancel);
     this.exports.push({cancel,encode:encoder=>{if(!done&&count)encoder.copyBufferToBuffer(words,0,staging,0,count*WORD_WORDS*4);},submitted:()=>{
      if(done)return;
      void (async()=>{try{await staging.mapAsync(GPUMapMode.READ);if(done)return;finish(undefined,this.decode(job,new Uint32Array(staging.getMappedRange(),0,count*WORD_WORDS),count,unresolved));}catch(error){finish(error);}})();
     }});
    });
    return result;
   },
  };
  this.tapes.add(resident);return resident;
 }
 private decode(job:Pending,memory:Uint32Array,count:number,unresolved:number):ArticulationResult{
  if(job.gpu)return this.decodeSource(job,memory,count,unresolved);
  const groups=new Map<number,NativeOccurrence[]>(),values=new Float32Array(memory.buffer,memory.byteOffset,memory.length);
  for(let i=0;i<count;i++){
   const at=i*WORD_WORDS,runIndex=memory[at]!,run=job.runs[runIndex]!;let x=memory[at+3]!,row=memory[at+4]!;
   const kind=memory[at+9]!,part=kind!==0,word=memory[at+5]!,length=memory[at+6]!,atoms=groups.get(runIndex)??[];groups.set(runIndex,atoms);
   const literal=job.literals.get(runIndex);
   if(literal){const atom=literal.group.occurrences[memory[at+1]!]!;atoms.push({...atom,x,row});continue;}
   for(let scalar=0;scalar<length;scalar++){
    const sourceScalar=kind===4?memory[at+13]!:scalar+(kind===0?memory[at+13]!:0);
    const fontIndex=part&&kind!==3?word:this.packed.words[word*WORD_STRIDE+2+sourceScalar]!,glyph=this.packed.font[fontIndex]!;
    if(glyph.points.length)atoms.push({id:`${run.id}/unit-${memory[at+1]}/${kind===1?'anchor':'run'}/${sourceScalar}`,text:glyph.text,x,row,foreground:memory[at+10]!,contributions:[{source:run.id,kind:'path',pixels:glyph.points.map((_,index)=>index),parameter:[values[at+2]!,values[at+2]!]}]});x+=glyph.advance;
   }
  }
  const obligations=unresolved?[`${unresolved} native GPU words require the next construction alternative`]:[];
  const complete=[...groups].sort(([a],[b])=>a-b).flatMap<ArticulationResult['groups'][number]>(([index,occurrences])=>{
   const run=job.runs[index]!;
   const literal=job.literals.get(index);if(literal)return [{owners:literal.owners??[],sources:literal.sources??[],support:literal.support,group:{...literal.group,occurrences}}];
   if(run.anchorOnly&&run.part){
    const part=this.parts.get(run.part)!;const point=run.points[0]!;
    const x=Math.round(point.x)+Math.round(-part.width/2),row=Math.floor(point.y/17)+(run.role==='return_base'?1-part.rows:0);let expected=0;
    if(point.visible)for(let i=0;i<part.count;i++){const at=(part.first+i)*4,px=x+(this.packed.partAtoms[at+1]!|0),py=row+this.packed.partAtoms[at+2]!;
     if(px>=0&&px+this.packed.partAtoms[at+3]!<=job.width&&py>=0&&py*17+17<=job.height)expected++;}
    if(occurrences.length!==expected)obligations.push(`${run.id}:retained anchor requires its complete native part`);
   }
   return [{owners:[run.owner],sources:run.sources,group:{id:run.id,occurrences,contacts:[],paper:[],requiredSources:[run.id]}}];
  });
  return {groups:complete,unresolved:obligations};
 }
 private decodeSource(job:Pending,memory:Uint32Array,count:number,unresolved:number):ArticulationResult{
  const context=job.gpu!,values=new Float32Array(memory.buffer,memory.byteOffset,memory.length);
  const groups=new Map<string,{owners:readonly string[];sources:readonly string[];support:NativeSupport;group:NativeNeighborhood;atoms:NativeOccurrence[];points:Map<string,readonly [number,number]>}>();
  for(let index=0;index<count;index++){
   const at=index*WORD_WORDS,runIndex=memory[at]!,literal=job.literals.get(runIndex);
   // A stacked camera's glyph seeded as a placement neighbour (part 5) belongs to that camera's export.
   if(memory[at+9]===5)continue;
   if(literal){let entry=groups.get(literal.group.id);if(!entry){const atoms:NativeOccurrence[]=[];entry={owners:literal.owners??[],sources:literal.sources??[],support:literal.support,group:{...literal.group,occurrences:atoms},atoms,points:new Map()};groups.set(literal.group.id,entry);}const original=literal.group.occurrences[memory[at+1]!]!;entry.atoms.push({...original,x:memory[at+3]!,row:memory[at+4]!,foreground:memory[at+10]!});continue;}
   const encoded=memory[at+16]!,domain=encoded&0x5fffffff,ownerIndex=memory[at+17]!,producer=!!(encoded&0x80000000);

   // domain 0, never a registered source's domain): its owner is the parent
   // construction's stable key (an octree node id or a producer skyline column),
   // not an entity the camera tracks. Its lifetime is that key.
   const aggregate=domain===0;
   const lifetime=producer||aggregate?ownerIndex:context.owners.get(ownerIndex);
   if(lifetime===undefined)throw Error('GPU native word lost its source lifetime');
   const owner=`${context.camera}/${domain}/${aggregate?'far-aggregate':producer?'producer-row':'entity'}/${lifetime}`;
   const rawKind=memory[at+19]!&0x1fffffff;
   // Match the builder/memo outline family without reinterpreting anchors.
   const kind=(memory[at+20]!&0x10000)===0&&rawKind===1?0:rawKind;
   // A surface intersection belongs to BOTH source faces. Different
   // receiver triangles can meet the same source face; source face alone
   // is not a run identity. Keep the exact tuple, not a 32-bit hash or a
   // snapshot-local allocation index, through native publication/export.
   const owners=[owner];
   const supportOwners:NativeSupport['owners'][number][]=[{reference:lifetime,source:domain,kind:producer?'producer-row':'entity'}];
   let contact='';
   if(kind===4){
    const encodedReceiver=memory[at+28]!,receiverDomain=encodedReceiver&0x1fffffff;
    const receiverProducer=!!(encodedReceiver&0x80000000),receiverIndex=memory[at+29]!;
    const receiverLifetime=receiverProducer?receiverIndex:context.owners.get(receiverIndex);
    if(receiverLifetime===undefined)throw Error('GPU native contact lost its receiver lifetime');
    const receiver=`${context.camera}/${receiverDomain}/${receiverProducer?'producer-row':'entity'}/${receiverLifetime}`;
    owners.push(receiver);supportOwners.push({reference:receiverLifetime,source:receiverDomain,kind:receiverProducer?'producer-row':'entity'});
    contact=`/receiver-${receiver}/face-${memory[at+30]}/port-${memory[at+31]}`;
   }
   // A run with no source address (0: a far form or far aggregate run of the

   // of one owner on one page keep distinct occurrence identities (tonemapping

   const id=`${owner}/gpu-run/${kind}/${memory[at+18]}${memory[at+18]===0?`/run-${runIndex}`:''}${contact}`;
   // A joined construction remains ONE group. Its words can be supported by
   // other source chunks; retain both root membership and the actual source
   // owner without exporting the transient metadata-run handle.
   let contributor:string|undefined;
   if(memory[at+9]!==2&&memory[at+9]!==3&&(memory[at+20]!&(0x10010000|NATIVE_SPAN_SOURCE))===(0x10000000|NATIVE_SPAN_SOURCE)){
    const encodedSpan=memory[at+22]!,spanDomain=encodedSpan&0x5fffffff,spanIndex=memory[at+23]!;
    const spanProducer=!!(encodedSpan&0x80000000),spanLifetime=spanProducer?spanIndex:context.owners.get(spanIndex);
    if(spanLifetime===undefined)throw Error('GPU native span lost its contributor lifetime');
    const spanOwner=`${context.camera}/${spanDomain}/${spanProducer?'producer-row':'entity'}/${spanLifetime}`;
    contributor=`${spanOwner}/source-edge/${memory[at+14]}`;
    if(!owners.includes(spanOwner)){owners.push(spanOwner);supportOwners.push({reference:spanLifetime,source:spanDomain,kind:spanProducer?'producer-row':'entity'});}
   }
   let entry=groups.get(id);
   if(!entry){const atoms:NativeOccurrence[]=[],points=new Map<string,readonly [number,number]>();const support:NativeSupport={configuration:id,points,owners:supportOwners};entry={owners,sources:[id],support,group:{id,occurrences:atoms,contacts:[],paper:[],requiredSources:[id]},atoms,points};groups.set(id,entry);}
   else{
    entry.owners=[...new Set([...entry.owners,...owners])].sort();
    const merged=new Map(entry.support.owners.map(o=>[`${o.kind}/${o.source}/${o.reference}`,o]));
    for(const o of supportOwners)merged.set(`${o.kind}/${o.source}/${o.reference}`,o);
    entry.support={...entry.support,owners:[...merged.values()].sort((a,b)=>a.source-b.source||a.reference-b.reference)};
   }
   if(contributor&&!entry.sources.includes(contributor))entry.sources=[...entry.sources,contributor].sort();
   entry.points.set('source-start',[values[at+24]!,values[at+25]!]);entry.points.set('source-end',[values[at+26]!,values[at+27]!]);
   const form=memory[at+9]!,word=memory[at+5]!,length=memory[at+6]!;let x=memory[at+3]!;const row=memory[at+4]!;
   for(let scalar=0;scalar<length;scalar++){
    const sourceScalar=form===4?memory[at+13]!:scalar+(form===0?memory[at+13]!:0);
    const glyph=this.packed.font[form!==0&&form!==3?word:this.packed.words[word*WORD_STRIDE+2+sourceScalar]!]!;
    // A source address is not a unique occurrence: disjoint visible regions
    // can emit the same source-local tone tile. The immutable committed-word
    // address distinguishes those occurrences without changing their source,
    // pen, or ink. It is snapshot-local, NOT a temporal correspondence key.
   const edge=form===0&&memory[at+15]===1?`/edge-${memory[at+14]}`:'';
   // An address-0 run (far form / far aggregate) can commit one unit ordinal on

   // is told apart by its committed-word address (snapshot-local).
   const ordinal=form===3?`${id}/node-${memory[at+14]}/${memory[at+15]}/${memory[at+22]}/${memory[at+23]}/${row}/word-${index}/${scalar}`:`${id}${edge}/unit-${memory[at+1]}/${sourceScalar}${memory[at+18]===0?`/word-${index}`:''}`;
    // Publication shades from this committed word. Export must retain the
    // same packed RGBA value; omitting it makes NativePresentation default
    // every decoded GPU occurrence to black.
    if(glyph.points.length){
     const contributions:NativeOccurrence['contributions'][number][]=[form===3?{source:id,kind:'region',pixels:nativeWholeInk(glyph)}:{source:id,kind:'path',pixels:nativeWholeInk(glyph),parameter:[values[at+2]!,values[at+2]!]}];
     if(contributor)contributions.push({source:contributor,kind:'path',pixels:nativeWholeInk(glyph)});
     entry.atoms.push({id:ordinal,text:glyph.text,x,row,foreground:memory[at+10]!,contributions});
    }
    x+=glyph.advance;
   }
  }
  return {groups:[...groups.values()].map(({atoms,points,...entry})=>entry),sourceLines:job.sourceLines,unresolved:unresolved?[`${unresolved} native GPU words require the next construction alternative`]:[]};
 }
 dispose():void{if(this.disposed)return;this.disposed=true;this.inkTexture?.destroy();for(const t of this.inkRetired)t.destroy();this.emptyOccluder?.destroy();this.emptyOccluder=undefined;for(const texture of this.occluders.values())texture.destroy();this.occluders.clear();this.rowHints.clear();for(const cancel of this.cancelExports)cancel();this.exports.length=0;for(const tape of this.tapes)tape.release();for(const cohort of this.cohorts)for(const resident of cohort.residents.values())resident.release();this.frameSeals.clear();this.cameraJobs.clear();this.cohorts.clear();const error=Error('Native GPU playback disposed');for(const job of [...this.active.keys()]){if(!job.settled){job.settled=true;job.reject(error);}job.gpu?.runs.dispose();this.release(job);}this.sourceRuns.dispose();this.spareScratch.forEach(buffer=>buffer.destroy());this.spareScratch.clear();this.scratchKinds.clear();this.pending=[];this.submittedTasks=[];this.retiredMemo.forEach(buffer=>buffer.destroy());this.retiredMemo.clear();this.resident.forEach(buffer=>buffer.destroy());for(const memo of this.cameraMemos.values())memo.buffer.destroy();this.cameraMemos.clear();this.gapTables.forEach(buffer=>buffer.destroy());this.gapTables.clear();}
}
