import {COHERENT_CORRESPONDENCE_MAGIC,COHERENT_CORRESPONDENCE_VERSION,COHERENT_CORRESPONDENCE_NONE} from './coherent-parent-correspondence.ts';
import {COHERENT_PARENT_RUN_FLAG,COHERENT_PARENT_SECONDARY_FLAG} from './coherent-parent-native-gpu.ts';

/** Existing far-status word bits reserved by the private integration seam. */
export const COHERENT_PARENT_STATUS_REJECT=0x8;
export const COHERENT_PARENT_STATUS_OVERFLOW=0x10;

/**
 * MAIN generated consumer for the in-band parent packet.  It keeps the
 * existing far Input/Run ABI: the primary owner remains identity.yz and a
 * second owner is carried in correspondence.xy.  One source passage emits
 * one Run; membership and source-feature aliases are resolved from the table,
 * not by emitting the passage once per owner.
 */
export const COHERENT_SOURCE_PARTS_WGSL=/* wgsl */`
const COHERENT_PARENT_MAGIC=${COHERENT_CORRESPONDENCE_MAGIC}u;
const COHERENT_PARENT_VERSION=${COHERENT_CORRESPONDENCE_VERSION}u;
const COHERENT_PARENT_NONE=${COHERENT_CORRESPONDENCE_NONE}u;
const COHERENT_PARENT_STATUS_REJECT=${COHERENT_PARENT_STATUS_REJECT}u;
const COHERENT_PARENT_STATUS_OVERFLOW=${COHERENT_PARENT_STATUS_OVERFLOW}u;
var<workgroup> coherentInvalid:atomic<u32>;
var<workgroup> coherentPointTotal:atomic<u32>;
var<workgroup> coherentRunTotal:atomic<u32>;
var<workgroup> coherentPointBase:atomic<u32>;
var<workgroup> coherentRunBase:atomic<u32>;
var<workgroup> coherentPointCursor:atomic<u32>;
var<workgroup> coherentRunCursor:atomic<u32>;
fn coherentWord(input:Input,word:u32)->u32{
 let words=arrayLength(&memberKeys)*2u;let program=input.program.x;
 if(program>words||input.mode.z>words-program||word>words-program-input.mode.z){return 0u;}
 return programWord(input,input.mode.z+word);
}
fn coherentHeaderValid(input:Input)->bool{
 let words=arrayLength(&memberKeys)*2u;let program=input.program.x;
 if(program>words||input.mode.z<6u||input.mode.z>words-program){return false;}
 if(input.members.x>arrayLength(&memberKeys)||input.members.y>arrayLength(&memberKeys)-input.members.x){return false;}
 let side=program+input.mode.z;if(words-side<6u){return false;}
 let entries=coherentWord(input,2u);let contributorWords=coherentWord(input,3u);let relatedWords=coherentWord(input,4u);let partCount=coherentWord(input,5u);let remaining=words-side-6u;
 if(entries>remaining/7u){return false;}
 let table=entries*7u;if(contributorWords>remaining-table){return false;}
 if(relatedWords>remaining-table-contributorWords||partCount!=input.program.y){return false;}
 let indexWords=partCount+entries+relatedWords;if(indexWords>remaining-table-contributorWords-relatedWords){return false;}
 if(input.program.y>input.mode.z/8u){return false;}
 return coherentWord(input,0u)==COHERENT_PARENT_MAGIC&&coherentWord(input,1u)==COHERENT_PARENT_VERSION;
}
fn coherentEntryCount(input:Input)->u32{return coherentWord(input,2u);}
fn coherentEntryBase(index:u32)->u32{return 6u+index*7u;}
fn coherentIndexBase(input:Input)->u32{return 6u+coherentEntryCount(input)*7u+coherentWord(input,3u)+coherentWord(input,4u);}
fn coherentPartEntry(input:Input,part:u32)->u32{return select(COHERENT_PARENT_NONE,coherentWord(input,coherentIndexBase(input)+part),part<input.program.y);}
fn coherentEntryPart(input:Input,entry:u32)->u32{return select(COHERENT_PARENT_NONE,coherentWord(input,coherentIndexBase(input)+input.program.y+entry),entry<coherentEntryCount(input));}
fn coherentRelatedEntry(input:Input,related:u32)->u32{return select(COHERENT_PARENT_NONE,coherentWord(input,coherentIndexBase(input)+input.program.y+coherentEntryCount(input)+related),related<coherentWord(input,4u));}
fn coherentFirstAlias(input:Input,part:u32)->u32{
 let entry=coherentPartEntry(input,part);if(entry>=coherentEntryCount(input)){return 0xffffffffu;}
 return coherentWord(input,coherentEntryBase(entry));
}
fn coherentFirstRelatedFeature(input:Input,part:u32,feature:u32)->u32{
 let entry=coherentPartEntry(input,part);if(entry>=coherentEntryCount(input)){return feature;}
 let at=coherentEntryBase(entry);let count=coherentWord(input,at+6u);if(count==0u){return feature;}
 let relatedBase=6u+coherentEntryCount(input)*7u+coherentWord(input,3u);return coherentWord(input,relatedBase+coherentWord(input,at+5u));
}
fn coherentEntryKindMatches(kind:u32,relation:bool,contact:bool)->bool{
 return select(kind==0u,kind!=0u,relation)&&(!contact||kind==2u);
}
fn coherentMemberVisible(input:Input,owner:vec2u)->bool{return listed(input,owner);}
fn sourceVisibilityOwnerMatches(primary:vec2u,secondary:vec2u,owner:vec2u,hasSecondary:bool)->bool{return all(owner==primary)||(hasSecondary&&all(owner==secondary));}
fn coherentSourceVisibility(input:Input,primary:vec2u,secondary:vec2u,hasSecondary:bool)->bool{
 return coherentMemberVisible(input,primary)&&(!hasSecondary||coherentMemberVisible(input,secondary))&&sourceVisibilityOwnerMatches(primary,secondary,primary,hasSecondary);
}
fn coherentExpectedContributorsExact(input:Input,entryIndex:u32,first:u32,count:u32)->bool{
 let at=coherentEntryBase(entryIndex);let contributorWords=coherentWord(input,3u);let contributorsBase=6u+coherentEntryCount(input)*7u;let expectedOffset=coherentWord(input,at+3u);let expectedCount=coherentWord(input,at+4u);
 if(expectedCount!=count||expectedOffset>contributorWords||expectedCount*2u>contributorWords-expectedOffset){return false;}
 let expectedBase=contributorsBase+expectedOffset;
 for(var actual=0u;actual<count;actual++){
  let owner=vec2u(programWord(input,first+actual*2u)&0xdfffffffu,programWord(input,first+actual*2u+1u));
  for(var prior=0u;prior<actual;prior++){let previous=vec2u(programWord(input,first+prior*2u)&0xdfffffffu,programWord(input,first+prior*2u+1u));if(all(owner==previous)){return false;}}
  var found=false;
  for(var expected=0u;expected<expectedCount;expected++){let candidate=vec2u(coherentWord(input,expectedBase+expected*2u)&0xdfffffffu,coherentWord(input,expectedBase+expected*2u+1u));if(all(owner==candidate)){found=true;}}
  if(!found){return false;}
 }
 for(var expected=0u;expected<expectedCount;expected++){
  let candidate=vec2u(coherentWord(input,expectedBase+expected*2u)&0xdfffffffu,coherentWord(input,expectedBase+expected*2u+1u));var found=false;
  for(var actual=0u;actual<count;actual++){let owner=vec2u(programWord(input,first+actual*2u)&0xdfffffffu,programWord(input,first+actual*2u+1u));if(all(owner==candidate)){found=true;}}
  if(!found){return false;}
 }
 return true;
}
fn coherentPartContributorsExact(input:Input,part:u32,feature:u32,relation:bool,contact:bool,first:u32,count:u32)->bool{
 let index=coherentPartEntry(input,part);if(index>=coherentEntryCount(input)){return false;}
 let at=coherentEntryBase(index);let kind=coherentWord(input,at+2u);return coherentWord(input,at+1u)==feature&&coherentEntryKindMatches(kind,relation,contact)&&coherentExpectedContributorsExact(input,index,first,count);
}
fn coherentEntryMatchesPart(input:Input,entry:u32,part:u32)->bool{
 if(entry>=coherentEntryCount(input)||part>=input.program.y){return false;}
 let entryAt=coherentEntryBase(entry);let partAt=4u+part*8u;let roleWord=programWord(input,partAt+3u);let packed=programWord(input,partAt+7u);let relation=(roleWord&0x80000000u)!=0u;let contact=(roleWord&0x40000000u)!=0u;let kind=coherentWord(input,entryAt+2u);
 return coherentWord(input,entryAt+1u)==programWord(input,partAt)&&coherentEntryKindMatches(kind,relation,contact)&&coherentExpectedContributorsExact(input,entry,programWord(input,partAt+6u),packed&0xffffu);
}
fn coherentCorrespondenceCoverage(input:Input)->bool{
 for(var entryIndex=0u;entryIndex<coherentEntryCount(input);entryIndex++){
  let partIndex=coherentEntryPart(input,entryIndex);if(!coherentEntryMatchesPart(input,entryIndex,partIndex)){return false;}
  let entry=coherentEntryBase(entryIndex);let relatedBase=6u+coherentEntryCount(input)*7u+coherentWord(input,3u);let relatedOffset=coherentWord(input,entry+5u);let relatedCount=coherentWord(input,entry+6u);
  for(var related=0u;related<relatedCount;related++){
   let relatedIndex=relatedOffset+related;let relatedEntry=coherentRelatedEntry(input,relatedIndex);if(relatedEntry>=coherentEntryCount(input)){return false;}
   let relatedFeature=coherentWord(input,relatedBase+relatedIndex);let relatedAt=coherentEntryBase(relatedEntry);if(relatedFeature!=coherentWord(input,relatedAt)&&relatedFeature!=coherentWord(input,relatedAt+1u)){return false;}
  }
 }
 return true;
}
fn coherentIndexValid(input:Input)->bool{
 for(var part=0u;part<input.program.y;part++){if(coherentPartEntry(input,part)>=coherentEntryCount(input)){return false;}}
 let contributorWords=coherentWord(input,3u);let relatedWords=coherentWord(input,4u);
 for(var entry=0u;entry<coherentEntryCount(input);entry++){
  if(coherentEntryPart(input,entry)>=input.program.y){return false;}
  let at=coherentEntryBase(entry);let contributorOffset=coherentWord(input,at+3u);let contributorCount=coherentWord(input,at+4u);let relatedOffset=coherentWord(input,at+5u);let relatedCount=coherentWord(input,at+6u);
  if(contributorOffset>contributorWords||contributorCount*2u>contributorWords-contributorOffset||relatedOffset>relatedWords||relatedCount>relatedWords-relatedOffset){return false;}
 }
 for(var related=0u;related<coherentWord(input,4u);related++){if(coherentRelatedEntry(input,related)>=coherentEntryCount(input)){return false;}}
 return true;
}
fn coherentPartValid(input:Input,index:u32)->bool{
 if(index>=input.program.y||input.mode.z<4u||input.program.y>(input.mode.z-4u)/8u){return false;}
 let base=4u+index*8u;let feature=programWord(input,base);let roleWord=programWord(input,base+3u);let first=programWord(input,base+6u);let packed=programWord(input,base+7u);let contributors=packed&0xffffu;let count=(packed>>16u)&0x7fffu;let closed=(packed&0x80000000u)!=0u;let relation=(roleWord&0x80000000u)!=0u;let contact=(roleWord&0x40000000u)!=0u;
 if(contributors==0u||contributors>2u||count<2u||first>input.mode.z||contributors*2u>input.mode.z-first){return false;}
 let poseAt=first+contributors*2u;if(poseAt>input.mode.z||16u>input.mode.z-poseAt){return false;}
 let geometry=poseAt+16u;if(geometry>input.mode.z||count>(input.mode.z-geometry)/3u){return false;}
 if(!coherentPartContributorsExact(input,index,feature,relation,contact,first,contributors)){return false;}
 for(var ownerIndex=0u;ownerIndex<contributors;ownerIndex++){
  let owner=vec2u(programWord(input,first+ownerIndex*2u)&0xdfffffffu,programWord(input,first+ownerIndex*2u+1u));if(!coherentMemberVisible(input,owner)){return false;}
 }
 return true;
}
fn coherentSourceParts(input:Input,local:u32)->bool{
 if(local==0u){atomicStore(&coherentInvalid,0u);atomicStore(&coherentPointTotal,0u);atomicStore(&coherentRunTotal,0u);atomicStore(&coherentPointBase,0xffffffffu);atomicStore(&coherentRunBase,0xffffffffu);atomicStore(&coherentPointCursor,0u);atomicStore(&coherentRunCursor,0u);}
 workgroupBarrier();
 let sourceMode=input.mode.w==2u&&input.mode.z!=0u;
 // The header and complete index are uniform for this source workgroup.
 // Validate them once, not once in each of its64 lanes. Only current part
 // visibility/geometry validation is distributed over those lanes.
 if(sourceMode&&local==0u&&(!coherentHeaderValid(input)||!coherentIndexValid(input))){atomicOr(&status[p.extent.w+2u],COHERENT_PARENT_STATUS_REJECT);atomicStore(&coherentInvalid,1u);}
 workgroupBarrier();
 if(sourceMode&&atomicLoad(&coherentInvalid)==0u){for(var check=local;check<input.program.y;check+=64u){if(!coherentPartValid(input,check)){atomicOr(&status[p.extent.w+2u],COHERENT_PARENT_STATUS_REJECT);atomicStore(&coherentInvalid,1u);}}}
 workgroupBarrier();
 if(sourceMode&&atomicLoad(&coherentInvalid)==0u&&local==0u&&!coherentCorrespondenceCoverage(input)){atomicOr(&status[p.extent.w+2u],COHERENT_PARENT_STATUS_REJECT);atomicStore(&coherentInvalid,1u);}
 workgroupBarrier();
 if(sourceMode&&atomicLoad(&coherentInvalid)==0u){for(var total=local;total<input.program.y;total+=64u){
  let totalBase=4u+total*8u;let totalPacked=programWord(input,totalBase+7u);let totalCount=(totalPacked>>16u)&0x7fffu;let totalClosed=(totalPacked&0x80000000u)!=0u;let totalSegments=totalCount-select(1u,0u,totalClosed);
  atomicAdd(&coherentPointTotal,totalSegments*2u);atomicAdd(&coherentRunTotal,1u);
 }}
 workgroupBarrier();
 if(sourceMode&&local==0u&&atomicLoad(&coherentInvalid)==0u){let allocation=reserve(atomicLoad(&coherentPointTotal),atomicLoad(&coherentRunTotal));if(allocation.x==0xffffffffu){atomicOr(&status[p.extent.w+2u],COHERENT_PARENT_STATUS_REJECT|COHERENT_PARENT_STATUS_OVERFLOW);atomicStore(&coherentInvalid,1u);}else{atomicStore(&coherentPointBase,allocation.x);atomicStore(&coherentRunBase,allocation.y);}}
 workgroupBarrier();
 if(sourceMode&&atomicLoad(&coherentInvalid)==0u){for(var i=local;i<input.program.y;i+=64u){
  let base=4u+i*8u;let feature=programWord(input,base);let entry=programWord(input,base+1u);let context=programWord(input,base+2u);let roleWord=programWord(input,base+3u);let role=programPartRole(roleWord);
  let first=programWord(input,base+6u);let packed=programWord(input,base+7u);let contributors=packed&0xffffu;let count=(packed>>16u)&0x7fffu;let closed=(packed&0x80000000u)!=0u;let relation=(roleWord&0x80000000u)!=0u;let contact=(roleWord&0x40000000u)!=0u;let relationParent=(roleWord&0x20000000u)!=0u;let compoundParent=(roleWord&0x10000000u)!=0u;
  let primary=vec2u(programWord(input,first)&0xdfffffffu,programWord(input,first+1u));let hasSecondary=contributors==2u;var secondary=vec2u(0u);if(hasSecondary){secondary=vec2u(programWord(input,first+2u)&0xdfffffffu,programWord(input,first+3u));}
  if(!coherentSourceVisibility(input,primary,secondary,hasSecondary)){continue;}
  let poseAt=first+contributors*2u;let pose=mat4x4f(sourceVector(input,poseAt),sourceVector(input,poseAt+4u),sourceVector(input,poseAt+8u),sourceVector(input,poseAt+12u));
  let geometry=poseAt+16u;let segments=count-select(1u,0u,closed);let pointCount=segments*2u;let pointAt=atomicAdd(&coherentPointCursor,pointCount);let runAt=atomicAdd(&coherentRunCursor,1u);let at=vec2u(atomicLoad(&coherentPointBase)+pointAt,atomicLoad(&coherentRunBase)+runAt);
  var top=1e30;var bottom=-1e30;
  for(var segment=0u;segment<segments;segment++){
   let a=sourcePoint(input,pose,geometry,segment,count,feature);var b=sourcePoint(input,pose,geometry,(segment+1u)%count,count,feature);if(closed&&segment+1u==count){b.position.w=1.0;}
   points[at.x+segment*2u]=a;points[at.x+segment*2u+1u]=b;top=min(top,min(a.position.y,b.position.y));bottom=max(bottom,max(a.position.y,b.position.y));
  }
  let rows=u32(clamp(floor(top)+16384.0,0.0,65535.0))|(u32(clamp(ceil(bottom)+16384.0,0.0,65535.0))<<16u);
  let pathFlags=select(0x80000000u|0x40000002u,0x04000000u|(2u<<8u)|role,contact)|${COHERENT_PARENT_RUN_FLAG}u|select(0u,${COHERENT_PARENT_SECONDARY_FLAG}u,hasSecondary);
  let lineage=select(coherentFirstAlias(input,i),coherentFirstRelatedFeature(input,i,feature),relation);
  runs[at.y]=Run(vec4u(at.x,pointCount,pathFlags,(feature<<5u)|(role<<2u)),vec4u(feature,primary,feature),vec4u(entry,context,programWord(input,base+4u),programWord(input,base+5u)),vec4u(0u,count,rows,role|select(0u,0x80000000u,relation)|select(0u,0x40000000u,contact)|select(0u,0x20000000u,relationParent)|select(0u,0x10000000u,compoundParent)),vec4u(secondary,input.program.z,lineage));
 }}
 return sourceMode;
}`;

/** Replace only the existing sourceParts body; all other generated producer
 * and consumer functions remain the current MAIN implementation. */
function replaceWgslFunction(source:string,name:string,replacement:string):string{
 const marker=`fn ${name}(`,start=source.indexOf(marker);if(start<0)throw Error(`missing WGSL function ${name}`);
 const brace=source.indexOf('{',start);if(brace<0)throw Error(`missing WGSL body ${name}`);
 let depth=0,end=brace;
 for(;end<source.length;end++){if(source[end]==='{')depth++;else if(source[end]==='}'&&--depth===0){end++;break;}}
 if(depth!==0)throw Error(`unclosed WGSL body ${name}`);
 return source.slice(0,start)+replacement+source.slice(end);
}

/** Keep the incumbent source-part producer for old packets; coherent packets
 * enter the new producer only when their in-band sidecar offset is nonzero. */
export function withCoherentSourceParts(shader:string):string{
 const legacy=shader.replace('fn sourceParts(','fn sourcePartsLegacy(');
 const coherent=COHERENT_SOURCE_PARTS_WGSL;
 const assembled=`${legacy}\n${coherent}\nfn sourceParts(input:Input,local:u32)->bool{
 let coherentHandled=coherentSourceParts(input,local);
 if(coherentHandled){return true;}
 if(input.mode.w==2u){sourcePartsLegacy(input,local);return true;}
 return false;
}`;
 const marker='fn emit(',start=assembled.indexOf(marker),brace=assembled.indexOf('{',start);if(start<0||brace<0)throw Error('missing WGSL function emit');
 let depth=0,end=brace;for(;end<assembled.length;end++){if(assembled[end]==='{')depth++;else if(assembled[end]==='}'&&--depth===0){end++;break;}}
 if(depth!==0)throw Error('unclosed WGSL function emit');
 return assembled.slice(0,start)+`fn emit(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) local:u32){
 let id=group.x+group.y*65535u;if(id>=p.extent.x){return;}let input=inputs[id];
 let sourceHandled=sourceParts(input,local);if(sourceHandled){return;}
 if(input.mode.x==1u){producerColumn(input,local);return;}
 entityAggregate(input,local);
}`+assembled.slice(end);
}
export const COHERENT_SOURCE_PARTS_ENTRY='emit' as const;
