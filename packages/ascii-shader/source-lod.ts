import {FAR_STRIP_BELOW_ROWS,LOD_TIER_HOLD} from './form-depths.ts';
import {latticeOpening} from './compiled-construction-topology.ts';
import type {SourceSmallParent} from './source-small-parent.ts';

/** THE PARENT TIER (perf-impl-108). An octree node of two or more placements
 * whose WHOLE posed box stands under the connected-outline tier (the L2 band,
 * FAR_STRIP_BELOW_ROWS with the hold) is ONE parent construction: its members
 * produce no fine relations, contacts, runs or placements, and the far
 * aggregate emitter draws the node once as its upper profile on the merged far
 * skyline (articulation farSkyline: one mass, its cap, flanks and base turn).
 * perf-99/105 traced the city stamp field to Numabe composing one building
 * from several placements: each sheet was its own owner under three rows and
 * took the box program's one-glyph stamp. A parent node is the building (or
 * the block) as one form. Entry at FAR_STRIP_BELOW_ROWS*LOD_TIER_HOLD rows;
 * a node that was a parent last frame stays one until FAR_STRIP_BELOW_ROWS/
 * LOD_TIER_HOLD (the same separated bands as the held per-form level). */
export const PARENT_ENTER_ROWS=FAR_STRIP_BELOW_ROWS*LOD_TIER_HOLD,PARENT_LEAVE_ROWS=FAR_STRIP_BELOW_ROWS/LOD_TIER_HOLD;
/** Page columns one parent profile may span (the emitter's workgroup table). */
export const PARENT_MAX_COLUMNS=64;

export interface SourceLodInstance {
 readonly eid:number;readonly mesh:number;readonly bounds?:readonly number[];readonly settling:boolean;
 /** Bounded initial native-history warmup, never an exemption for motion. */
 readonly boundsPending?:boolean;
 /** Immutable source revision used by the parent representation cache. */
 readonly sourceRevision?:number;
 /** Pose revision for the resident bound; it is not a camera revision. */
 readonly poseRevision?:number;
 /** Actual settled source pose, cached by its owning workset, not a camera. */
 readonly worldFromSource?:ArrayLike<number>;
 /** The source-owned whole construction, retained by source compilation. */
 readonly construction?:SourceWholeConstruction;
 /** Immutable source metadata; not itself a parent-eligibility certificate. */
 readonly smallParent?:SourceSmallParent;
}
/** A complete source-owner extent, not a selected edge or an image glyph. */
/** `parent`: the key of the octree aggregate this member belongs to (perf-impl-96).
 * Such a member is far for residency and relations, but it is drawn ONLY by its
 * parent's one aggregate record, never as a far form of its own. */
export interface FarFormInstance {
 readonly eid:number;readonly box:readonly [number,number,number,number];readonly parent?:number;
 readonly smallParent?:SourceSmallParent;
 readonly sourceRevision?:number;
}
/** One compiled parent construction (perf-impl-96): an adequate octree node's
 * members, drawn once through the far aggregate emitter (gpu-source-far-forms). */
export type SourceWholePartRole='cap'|'turn'|'flank'|'return';
/** Rest-source construction samples. These are compiled once from the source
 * graph; they are not camera fragments or glyph coordinates. */
export type SourceWholePoint=readonly [number,number,number];
/** One exact retained source-part passage. `entryRef` is an opaque reference
 * into the resident role/program table; selection never chooses a glyph. */
export interface SourceWholePartRef {
 readonly feature:number;readonly entryRef:number;readonly context:number;
 readonly role:SourceWholePartRole;readonly ports:readonly [number,number];
 readonly contributors:readonly (readonly [number,number])[];
 /** Source-space section/band samples for the far consumer. The consumer
  * poses these with its existing source transform; it must not reuse one
  * parent skyline for every part. */
 readonly geometry?:readonly SourceWholePoint[];
 readonly closed?:boolean;
 /** Pose for the immutable source samples. Never inferred from a screen box. */
 readonly worldFromSource?:ArrayLike<number>;
 /** Source-paper passages carried with the part. The native consumer owns
  * their page interpretation; the parent selector must not erase them. */
 readonly paper?:readonly SourceWholeOpening[];
}
export interface SourceWholeOpening {
 readonly feature:number;readonly ports:readonly number[];
 readonly contributors:readonly (readonly [number,number])[];
}
/** Source-owned parent construction. The optional-looking arrays are required
 * by the contract so an empty value means "known empty", not "unknown". */
export interface SourceWholeConstruction {
 readonly revision:number;
 readonly parts:readonly SourceWholePartRef[];
 readonly openings:readonly SourceWholeOpening[];
 readonly contacts:readonly SourceWholeOpening[];
 readonly protectedVoids:readonly SourceWholeOpening[];
 /** Explicit paper/air passages owned by the compiled source graph. */
 readonly paper?:readonly SourceWholeOpening[];
}

/** Source-revision form emitted by the production compiler before an instance
 * identity exists. `sourceContributors` are compiled source-part ordinals,
 * never glyphs or projected fragments. The engine adapter binds them to the
 * concrete (mesh,eid) instance only after the immutable resident is ready. */
export interface SourceWholePartTemplate extends Omit<SourceWholePartRef,'contributors'|'paper'> {
 readonly sourceContributors:readonly number[];
 readonly paper?:readonly SourceWholeOpeningTemplate[];
}
export interface SourceWholeOpeningTemplate extends Omit<SourceWholeOpening,'contributors'> {
 readonly sourceContributors:readonly number[];
}
export interface SourceWholeConstructionTemplate {
 readonly revision:number;
 readonly parts:readonly SourceWholePartTemplate[];
 readonly openings:readonly SourceWholeOpeningTemplate[];
 readonly contacts:readonly SourceWholeOpeningTemplate[];
 readonly protectedVoids:readonly SourceWholeOpeningTemplate[];
 readonly paper?:readonly SourceWholeOpeningTemplate[];
}

const bindOpening=(value:SourceWholeOpeningTemplate,mesh:number,eid:number):SourceWholeOpening|undefined=>value.sourceContributors.length?({
 feature:value.feature,ports:value.ports,contributors:value.sourceContributors.map(()=>[mesh,eid] as const)
}):undefined;
/** Bind one immutable source template to one posed instance. This is a pure
 * identity adapter: no camera, mesh scan, glyph lookup or source recompilation
 * occurs here. */
export function bindSourceWholeConstruction(template:SourceWholeConstructionTemplate|undefined,mesh:number,eid:number,worldFromSource?:ArrayLike<number>):SourceWholeConstruction|undefined{
 if(!template||!Number.isSafeInteger(mesh)||!Number.isSafeInteger(eid))return;
 if(template.parts.some(part=>part.sourceContributors.length===0)||[...template.openings,...template.contacts,...template.protectedVoids,...(template.paper??[])].some(value=>value.sourceContributors.length===0))return;
 const bindParts=template.parts.map(part=>({feature:part.feature,entryRef:part.entryRef,context:part.context,role:part.role,ports:part.ports,contributors:[[mesh,eid] as const],...(part.geometry?{geometry:part.geometry,closed:part.closed}:{}),...(worldFromSource?{worldFromSource}:{}),...(part.paper?{paper:part.paper.flatMap(value=>{const bound=bindOpening(value,mesh,eid);return bound?[bound]:[];})}: {})}));
 const bindList=(values:readonly SourceWholeOpeningTemplate[])=>values.flatMap(value=>{const bound=bindOpening(value,mesh,eid);return bound?[bound]:[];});
 return {revision:template.revision,parts:bindParts,openings:bindList(template.openings),contacts:bindList(template.contacts),protectedVoids:bindList(template.protectedVoids),...(template.paper?{paper:bindList(template.paper)}:{})};
}
/** Selection uses mesh handles; native fields use resident source domains
 * (and producer owner tags). Convert identities at that adapter seam, without
 * changing source features, geometry, pose, or anchor correspondence. */
export function remapSourceWholeConstruction(value:SourceWholeConstruction|undefined,owner:(mesh:number,eid:number)=>readonly [number,number]|undefined):SourceWholeConstruction|undefined{
 if(!value)return;
 const pairs=(contributors:SourceWholePartRef['contributors'])=>{
  const result:(readonly [number,number])[]=[];
  for(const [mesh,eid] of contributors){const mapped=owner(mesh,eid);if(!mapped)return;result.push(mapped);}
  return result;
 };
 const openings=(values:readonly SourceWholeOpening[])=>{
  const result:SourceWholeOpening[]=[];
  for(const item of values){const contributors=pairs(item.contributors);if(!contributors)return;result.push({...item,contributors});}
  return result;
 };
 const parts:SourceWholePartRef[]=[];
 for(const part of value.parts){
  const contributors=pairs(part.contributors),paper=part.paper?openings(part.paper):undefined;
  if(!contributors||(part.paper&&!paper))return;
  parts.push({...part,contributors,...(paper?{paper}:{})});
 }
 const knownOpenings=openings(value.openings),contacts=openings(value.contacts),protectedVoids=openings(value.protectedVoids),paper=value.paper?openings(value.paper):undefined;
 if(!knownOpenings||!contacts||!protectedVoids||(value.paper&&!paper))return;
 return {...value,parts,openings:knownOpenings,contacts,protectedVoids,...(paper?{paper}:{})};
}
export interface SourceLodAggregate {
 readonly key:number;readonly box:readonly [number,number,number,number];
 readonly members:readonly {readonly mesh:number;readonly eid:number}[];
 /** Exact source-owned construction consumed by the far-program emitter.
  * Absent is an older/unknown resident and deliberately keeps the incumbent
  * parent profile until the replacement producer is connected. It is not a
  * claim that the profile is a complete whole construction. */
 readonly construction?:SourceWholeConstruction;
}
export interface SourceLodSelection {
 readonly far:ReadonlyMap<number,readonly FarFormInstance[]>;
 /** The parent aggregates of this camera (perf-impl-96); their members are in `far` with `parent`. */
 readonly parents?:readonly SourceLodAggregate[];
 /** The node keys drawn as parents this frame: next frame's held set (perf-impl-108). */
 readonly heldParents?:ReadonlySet<number>;
 readonly heldSmall?:ReadonlySet<string>;
 readonly defer:ReadonlySet<number>;
 /** Every instance is outside this camera's padded, same-pose page box. */
 readonly outside?:ReadonlySet<number>;
 /** PER-INSTANCE LEVELS (perf-impl-94): a mesh in BOTH `far` and `full` is
  * mixed. `full` names the member eids this camera draws with the complete
  * producer; its `far` members are compiled far forms. A mesh only in `far`
  * is wholly far; a mesh in neither is wholly full (as before). In a merged
  * (residency) selection the list is empty: full program AND far form needed. */
 readonly full?:ReadonlyMap<number,readonly number[]>;
 readonly exterior:boolean;
}
/** Project ALL eight corners of the authoritative native bound. A clipped
 * near plane or unknown bound never supplies a falsely small form. Page rows,
 * not world metres or scene extent, select the compiled smallest tier. */
function projectedBox(bounds:readonly number[],m:ArrayLike<number>,width:number,height:number):[number,number,number,number]|undefined{
 const box:[number,number,number,number]=[Infinity,Infinity,-Infinity,-Infinity];
 for(let corner=0;corner<8;corner++){
  const x=bounds[corner&1?3:0]!,y=bounds[corner&2?4:1]!,z=bounds[corner&4?5:2]!;
  const w=m[3]!*x+m[7]!*y+m[11]!*z+m[15]!,depth=m[2]!*x+m[6]!*y+m[10]!*z+m[14]!;
  if(!(w>1e-5)||!(depth>=0&&depth<=w))return;
  const px=((m[0]!*x+m[4]!*y+m[8]!*z+m[12]!)/w*.5+.5)*width;
  const py=(.5-(m[1]!*x+m[5]!*y+m[9]!*z+m[13]!)/w*.5)*height;
  box[0]=Math.min(box[0],px);box[1]=Math.min(box[1],py);box[2]=Math.max(box[2],px);box[3]=Math.max(box[3],py);
 }
 // The bound was evaluated on the CPU in f64; include native f32/raster
 // roundoff instead of shrinking it to a count of currently visible pixels.
 for(let k=0;k<4;k++)box[k]!+=(k<2?-1:1)*(1+1e-5*Math.abs(box[k]!));
 return box.every(Number.isFinite)?box:undefined;
}

/** Source demand before numeric incidence, material extraction and
 * relationships, PER INSTANCE (perf-impl-94). Each admitted instance takes its
 * own level: a complete form smaller than one native cell is a compiled far
 * form, any other (or unknown, clipped, moving) instance keeps the full
 * producer. One near copy no longer promotes every copy of its mesh: the mesh
 * is then MIXED (`full` lists its full members). No instance is removed. Only
 * bounded initial native-history warmup may defer work. */
/** The far-form rule as a test on any posed bound: its complete projected
 * box fits ONE native cell with the hold (perf-impl-95: the octree asks it of
 * a node's tight aggregate box). */
export function fitsNativeCell(bounds:readonly number[],clip:ArrayLike<number>,width:number,height:number,pageWidth:number,pageHeight:number):boolean{
 const box=projectedBox(bounds,clip,width,height);if(!box)return false;
 return box[2]-box[0]<16*LOD_TIER_HOLD*width/Math.max(1,pageWidth)&&box[3]-box[1]<17*LOD_TIER_HOLD*height/Math.max(1,pageHeight);
}
/** A projected field box under `rows` text rows and PARENT_MAX_COLUMNS columns. */
function parentBox(box:readonly number[],rows:number,width:number,height:number,pageWidth:number,pageHeight:number):boolean{
 return box[3]!-box[1]!<rows*17*height/Math.max(1,pageHeight)&&box[2]!-box[0]!<PARENT_MAX_COLUMNS*16*width/Math.max(1,pageWidth);
}
/** The octree's adequacy test for the parent tier: the LEAVE band, so a held
 * parent is still offered whole; selectSourceLod applies the entry band to a
 * node that was not a parent last frame. */
export function fitsParentTier(bounds:readonly number[],clip:ArrayLike<number>,width:number,height:number,pageWidth:number,pageHeight:number):boolean{
 const box=projectedBox(bounds,clip,width,height);
 return !!box&&parentBox(box,PARENT_LEAVE_ROWS,width,height,pageWidth,pageHeight);
}
/** An octree node selected as ONE representation (engine source-workset). */
/** `key`: the node's stable identity (spatial-index node id, perf-impl-96); the
 * aggregate keeps it while the node exists, so its skyline units keep their tag. */
export interface SourceLodParent {readonly bounds:readonly number[];readonly instances:readonly SourceLodInstance[];readonly key?:number}
/** Members of one candidate node that touch in the WORLD: their bounds meet
 * within PARENT_TOUCH of the smaller member's largest side (view-independent,
 * so a group does not change with the camera). A member without bounds is a
 * group of its own. Groups in least-eid order.
 * perf-impl-112: only PROFILE-SAFE members join. One member whose own upper
 * profile would close an opening (an arch sheet) is a group of its own and
 * keeps its own producer; it no longer returns every touching safe member of
 * the building to its per-sheet stamp (whole-group rejection on one member). */
export const PARENT_TOUCH=.02;
const touchSide=(b:readonly number[])=>Math.max(b[3]!-b[0]!,b[4]!-b[1]!,b[5]!-b[2]!);
/** Two member bounds meet within PARENT_TOUCH of the smaller one's largest side. */
function membersTouch(a:readonly number[],b:readonly number[]):boolean{
 const tol=PARENT_TOUCH*Math.min(touchSide(a),touchSide(b));
 return [0,1,2].every(k=>a[k]!<=b[k+3]!+tol&&b[k]!<=a[k+3]!+tol);
}
/** Member index groups (indices into `members`), least member first. */
function parentGroups(members:readonly SourceLodInstance[],safe:readonly boolean[]=members.map(()=>true)):number[][]{
 const order=members.map((_,i)=>i).sort((a,b)=>members[a]!.eid-members[b]!.eid),parent=order.map((_,i)=>i);
 const find=(i:number):number=>{while(parent[i]!==i){parent[i]=parent[parent[i]!]!;i=parent[i]!;}return i;};
 for(let i=0;i<order.length;i++){const a=members[order[i]!]!.bounds;if(!a||!safe[order[i]!])continue;
  for(let j=i+1;j<order.length;j++){const b=members[order[j]!]!.bounds;if(!b||!safe[order[j]!])continue;
   if(membersTouch(a,b)){const x=find(i),y=find(j);if(x!==y)parent[Math.max(x,y)]=Math.min(x,y);}}}
 const groups=new Map<number,number[]>();
 order.forEach((index,i)=>{const r=find(i);const g=groups.get(r)??[];g.push(index);groups.set(r,g);});
 return [...groups.values()];
}
/** Allocated group ids carry this bit, so they never meet an octree node id
 * (the one-cell aggregate's key, perf-impl-96) in the held set or a run tag. */
export const PARENT_GROUP_ID_BIT=0x80000000;
export interface ParentGroup {readonly id:number;readonly members:readonly SourceLodInstance[]}
interface CachedMember {readonly eid:number;readonly ref:number;readonly mesh:number;readonly safe:boolean;readonly sourceRevision?:number;readonly poseRevision?:number;readonly bounds?:readonly number[]}
const sameBounds=(a:readonly number[]|undefined,b:readonly number[]|undefined)=>a===b||(!!a&&!!b&&a.length===b.length&&a.every((v,i)=>v===b[i]));
const instanceSourceRevision=(instance:SourceLodInstance):number|undefined=>instance.sourceRevision??instance.construction?.revision;
/** THE CACHED PARENT REPRESENTATION (perf-impl-108). Per octree node (a
 * SpatialIndex node id never recycles within its world), the node's world
 * touching groups are compiled when its membership, a member's lifetime ref,
 * mesh or bounds change, and only then; an unchanged node costs one O(N)
 * comparison per view. A group's id is ALLOCATED once per exact member
 * content (the sorted lifetime ref and mesh of every member, compared as
 * exact keys, never a hash), so a hold or an aggregate tag is inherited only
 * by the identical group. One cache per host world. */
export class ParentGroupCache{
 private readonly nodes=new Map<number,{members:readonly CachedMember[];groups:readonly (readonly number[])[];ids:readonly number[]}>();
 private readonly ids=new Map<string,number>();private next=1;
 /** Entries beyond these bounds are dropped wholesale: ids then re-allocate,
  * so a hold restarts at the entry band (conservative), never inherits. */
 static readonly MAX_NODES=16384;static readonly MAX_IDS=65536;
 /** The allocated id of this exact member content (sorted lifetime ref and mesh). */
 contentId(members:readonly {readonly ref:number;readonly mesh:number;readonly revision?:number;readonly sourceRevision?:number}[]):number{
  if(this.ids.size>ParentGroupCache.MAX_IDS)this.ids.clear();
  const content=members.map(m=>`${m.ref}:${m.mesh}:${m.revision??m.sourceRevision??0}`).sort().join('|');
  let id=this.ids.get(content);if(id===undefined){id=(PARENT_GROUP_ID_BIT|(this.next++&0x7fffffff))>>>0;this.ids.set(content,id);}
  return id;
 }
 groups(node:number,members:readonly SourceLodInstance[],identity:(eid:number)=>number,profileSafe:(mesh:number)=>boolean=()=>true):readonly ParentGroup[]{
  let cached=this.nodes.get(node);
  const same=!!cached&&cached.members.length===members.length&&members.every((m,i)=>{const c=cached!.members[i]!;return c.eid===m.eid&&c.mesh===m.mesh&&c.ref===identity(m.eid)&&c.safe===profileSafe(m.mesh)&&c.sourceRevision===instanceSourceRevision(m)&&c.poseRevision===m.poseRevision&&sameBounds(c.bounds,m.bounds);});
  if(!same){
   const record=members.map(m=>({eid:m.eid,ref:identity(m.eid),mesh:m.mesh,safe:profileSafe(m.mesh),sourceRevision:instanceSourceRevision(m),poseRevision:m.poseRevision,bounds:m.bounds?[...m.bounds]:undefined}));
   const groups=parentGroups(members,record.map(m=>m.safe));
   const ids=groups.map(group=>this.contentId(group.map(i=>record[i]!)));
   if(this.nodes.size>ParentGroupCache.MAX_NODES)this.nodes.clear();
   cached={members:record,groups,ids};this.nodes.set(node,cached);
  }
  return cached!.groups.map((group,g)=>({id:cached!.ids[g]!,members:group.map(i=>members[i]!)}));
 }
}
/** THE PARENT HOLD across frames (perf-impl-108), per host LOD scope: the
 * camera lifetime, world and owner epoch of FormLodLifetimes.scope, never a
 * View object or slot that the engine can recycle. Bounded: the least
 * recently written scopes beyond PARENT_HOLD_SCOPES are dropped (a dropped or
 * new scope starts with no parents held, the conservative entry band). */
export const PARENT_HOLD_SCOPES=64;
export interface ParentHoldScope {readonly cameraId:number;readonly worldId:number;readonly epoch:number}
export class ParentHolds{
 private readonly table=new Map<string,{readonly parents:ReadonlySet<number>;readonly small:ReadonlySet<string>}>();
 private static key(scope:ParentHoldScope):string{return `${scope.worldId}:${scope.cameraId}:${scope.epoch}`;}
 get(scope:ParentHoldScope):ReadonlySet<number>{return this.table.get(ParentHolds.key(scope))?.parents??new Set();}
 getSmall(scope:ParentHoldScope):ReadonlySet<string>{return this.table.get(ParentHolds.key(scope))?.small??new Set();}
 set(scope:ParentHoldScope,held:ReadonlySet<number>,small:ReadonlySet<string>=new Set()):void{
  const key=ParentHolds.key(scope);this.table.delete(key);this.table.set(key,{parents:held,small});
  while(this.table.size>PARENT_HOLD_SCOPES)this.table.delete(this.table.keys().next().value!);
 }
}
function unionBox(boxes:readonly ([number,number,number,number]|undefined)[]):[number,number,number,number]|undefined{
 const out:[number,number,number,number]=[Infinity,Infinity,-Infinity,-Infinity];
 for(const b of boxes){if(!b)return undefined;out[0]=Math.min(out[0],b[0]);out[1]=Math.min(out[1],b[1]);out[2]=Math.max(out[2],b[2]);out[3]=Math.max(out[3],b[3]);}
 return out;
}

const WHOLE_PART_ROLES:readonly SourceWholePartRole[]=['cap','turn','flank','return'];
const U32_MAX=0xffffffff;
const pairKey=(pair:readonly [number,number]):string=>`${pair[0]>>>0}:${pair[1]>>>0}`;
/** Validate only the resident source metadata. This never inspects mesh data,
 * camera pixels or the glyph bank. */
export function sourceWholeConstructionReady(value:SourceWholeConstruction|undefined):value is SourceWholeConstruction{
 if(!value||!Number.isSafeInteger(value.revision)||value.revision<=0||!value.parts.length)return false;
 const roles=new Set(value.parts.map(part=>part.role));if(WHOLE_PART_ROLES.some(role=>!roles.has(role)))return false;
 for(const part of value.parts){
  if(!WHOLE_PART_ROLES.includes(part.role))return false;
  if(!Number.isSafeInteger(part.feature)||part.feature<=0||part.feature>U32_MAX||!Number.isSafeInteger(part.entryRef)||part.entryRef<0||part.entryRef>U32_MAX||!Number.isSafeInteger(part.context)||part.context<0||part.context>U32_MAX)return false;
  if(part.ports.length!==2||part.ports.some(port=>!Number.isSafeInteger(port)||port<0||port>U32_MAX)||!part.contributors.length)return false;
  if(part.geometry&&(!part.geometry.length||part.geometry.some(point=>point.length!==3||point.some(value=>!Number.isFinite(value)))))return false;
  if(part.worldFromSource&&(part.worldFromSource.length!==16||Array.from(part.worldFromSource).some(value=>!Number.isFinite(value))))return false;
  if(part.contributors.some(pair=>pair.length!==2||pair.some(word=>!Number.isSafeInteger(word)||word<0||word>U32_MAX)))return false;
  if(part.paper&&part.paper.some(value=>!Number.isSafeInteger(value.feature)||value.feature<0||value.ports.length<2||value.ports.some(port=>!Number.isSafeInteger(port)||port<0||port>U32_MAX)||!value.contributors.length||value.contributors.some(pair=>pair.length!==2||pair.some(word=>!Number.isSafeInteger(word)||word<0||word>U32_MAX))))return false;
 }
 for(const opening of [...value.openings,...value.contacts,...value.protectedVoids,...(value.paper??[])]){
  if(!Number.isSafeInteger(opening.feature)||opening.feature<0||opening.feature>U32_MAX||opening.ports.length<2||opening.ports.some(port=>!Number.isSafeInteger(port)||port<0||port>U32_MAX)||!opening.contributors.length)return false;
  if(opening.contributors.some(pair=>pair.length!==2||pair.some(word=>!Number.isSafeInteger(word)||word<0||word>U32_MAX)))return false;
 }
 return true;
}
/** The current far-program consumer can preserve its four connected passage
 * roles and contributor ids. It cannot reconstruct an opaque opening/contact
 * ref in page space, so callers must descend to complete children instead. */
export function sourceWholeConstructionRenderable(value:SourceWholeConstruction|undefined):value is SourceWholeConstruction{
 return sourceWholeConstructionReady(value)&&value.openings.length===0&&value.contacts.length===0&&value.protectedVoids.length===0;
}
/** Combine immutable source programs without introducing a scene recipe. A
 * program can only represent the listed member identities; a missing member,
 * revision or protected relation therefore rejects the parent. */
export function composeParentConstruction(members:readonly SourceLodInstance[]):SourceWholeConstruction|undefined{
 if(!members.length||members.some(member=>!sourceWholeConstructionRenderable(member.construction)))return undefined;
 const identities=members.map(member=>[member.mesh,member.eid] as const),known=new Set(identities.map(pairKey));
 if(known.size!==identities.length)return undefined;
 const programs=members.map(member=>member.construction!);
 const validContributor=(pairs:readonly (readonly [number,number])[])=>pairs.length>0&&pairs.every(pair=>known.has(pairKey(pair)));
 if(programs.some(program=>program.parts.some(part=>!validContributor(part.contributors)||(part.paper??[]).some(item=>!validContributor(item.contributors)))||[...program.openings,...program.contacts,...program.protectedVoids,...(program.paper??[])].some(item=>!validContributor(item.contributors))))return undefined;
 const parts=programs.flatMap(program=>program.parts);
 const openings=programs.flatMap(program=>program.openings),contacts=programs.flatMap(program=>program.contacts),protectedVoids=programs.flatMap(program=>program.protectedVoids);
 // All source program refs remain in the payload, but this consumer cannot
 // project a protected opening/contact from an opaque ref. Descending is the
 // legal representation when such a relation cannot be preserved exactly.
 if(openings.length||contacts.length||protectedVoids.length)return undefined;
 const revision=programs.reduce((hash,program)=>{
  let next=(hash^program.revision)>>>0;for(const part of program.parts){next=Math.imul(next^part.feature,16777619)>>>0;next=Math.imul(next^part.entryRef,16777619)>>>0;next=Math.imul(next^part.context,16777619)>>>0;for(const paper of part.paper??[])next=Math.imul(next^paper.feature,16777619)>>>0;}return next;
 },2166136261)>>>0;
 const paper=programs.flatMap(program=>program.paper??[]);
 return {revision:revision||1,parts,openings,contacts,protectedVoids,...(paper.length?{paper}: {})};
}
/** NO OPENING (perf-impl-108): on the glyph-cell lattice over the group's
 * projected box, no empty cell is covered on both sides in its row and above
 * it (latticeOpening: an arch, a gap between piers under a lintel) that the
 * upper profile would close. Cells are judged at their centres from the
 * members' complete projected bounds; each member's OWN openings are its
 * compiled compileMeshProfileSafe (the host's `profileSafe`). */
function parentAdequate(group:readonly SourceLodInstance[],clip:ArrayLike<number>,width:number,height:number,pageWidth:number,pageHeight:number,box:readonly number[]):boolean{
 const cw=16*width/Math.max(1,pageWidth),rh=17*height/Math.max(1,pageHeight);
 const boxes=group.map(instance=>projectedBox(instance.bounds!,clip,width,height));if(boxes.some(b=>!b))return false;
 const columns=Math.max(1,Math.ceil((box[2]!-box[0]!)/cw)),rows=Math.max(1,Math.ceil((box[3]!-box[1]!)/rh));
 const grid=new Uint8Array(columns*rows);
 for(let c=0;c<columns;c++)for(let r=0;r<rows;r++){
  const x=Math.min(box[2]!,box[0]!+(c+.5)*cw),y=Math.min(box[3]!,box[1]!+(r+.5)*rh);
  // Lattice row 0 is the LOW side (the page's bottom row).
  if(boxes.some(b=>x>=b![0]&&x<=b![2]&&y>=b![1]&&y<=b![3]))grid[(rows-1-r)*columns+c]=1;
 }
 return !latticeOpening(grid,columns,false,rows);
}
export type SmallConstructionAvailability=(parent:SourceSmallParent,rows:number,width:number)=>boolean;
/** Straight native side rows cannot depict a rotated/slanted source tube.
 * The current small family also has its positive end at the top. Use the
 * actual instance pose; an unsupported orientation retains its source work. */
export function smallParentPoseSupported(instance:SourceLodInstance,clip:ArrayLike<number>,pageWidth=1,pageHeight=1):boolean{
 const parent=instance.smallParent,pose=instance.worldFromSource,bounds=instance.bounds;
 if(!parent||!pose||pose.length!==16||!bounds)return false;
 const center=[(bounds[0]!+bounds[3]!)/2,(bounds[1]!+bounds[4]!)/2,(bounds[2]!+bounds[5]!)/2];
 const axis=parent.axis.map((_,row)=>pose[row]!*parent.axis[0]+pose[row+4]!*parent.axis[1]+pose[row+8]!*parent.axis[2]);
 const project=(point:readonly number[])=>{
  const w=clip[3]!*point[0]!+clip[7]!*point[1]!+clip[11]!*point[2]!+clip[15]!;
  if(!(w>1e-5))return undefined;
  return [(clip[0]!*point[0]!+clip[4]!*point[1]!+clip[8]!*point[2]!+clip[12]!)/w,
   -(clip[1]!*point[0]!+clip[5]!*point[1]!+clip[9]!*point[2]!+clip[13]!)/w];
 };
 const a=project(center),b=project(center.map((value,index)=>value+axis[index]!));
 if(!a||!b)return false;
 // Clip x and y each span [-1,1], but native horizontal advances and row
 // pitch live on the actual page. A wide viewport must not certify a visibly
 // slanted side as vertical merely because its NDC delta is small.
 const dx=(b[0]!-a[0]!)*Math.max(1,pageWidth),dy=(b[1]!-a[1]!)*Math.max(1,pageHeight);
 return dy<-1e-6&&Math.abs(dx)<=Math.abs(dy)*.25;
}
export function selectSourceLod(instances:readonly SourceLodInstance[],clip:ArrayLike<number>,width:number,height:number,pageWidth:number,pageHeight:number,exterior:boolean,perspective:boolean,parents:readonly SourceLodParent[]=[],held:ReadonlySet<number>=new Set(),profileSafe:(mesh:number)=>boolean=()=>false,identity:(eid:number)=>number=eid=>eid,cache:ParentGroupCache=new ParentGroupCache(),joinSingles=false,smallAvailable?:SmallConstructionAvailability,heldSmall:ReadonlySet<string>=new Set()):SourceLodSelection{
 const far=new Map<number,FarFormInstance[]>(),defer=new Set<number>(),full=new Map<number,number[]>(),seen=new Set<number>(),aggregates:SourceLodAggregate[]=[],heldParents=new Set<number>();
 const smallOut=new Set<string>();
 // Only a complete form smaller than ONE native cell may merge. A narrow
 // two-row house still holds its own construction, whatever its class.
 const maximumX=16*LOD_TIER_HOLD*width/Math.max(1,pageWidth),maximumY=17*LOD_TIER_HOLD*height/Math.max(1,pageHeight);
 // PARENT CONSTRUCTION (perf-impl-96; perf-impl-95 selected the node): an
 // adequate octree node is ONE aggregate. Its members are far (residency, no
 // relations) and carry the node's key; the far aggregate emitter draws the
 // node once from all their pixels. A node that no longer fits (or an unknown
 // view) lists its members singly.
 const adopt=(groupKey:number,groupBox:[number,number,number,number],group:readonly SourceLodInstance[],construction?:SourceWholeConstruction)=>{
  heldParents.add(groupKey);
  for(const instance of group){seen.add(instance.mesh);const members=far.get(instance.mesh)??[];members.push({eid:instance.eid,box:groupBox,parent:groupKey});far.set(instance.mesh,members);}
  aggregates.push({key:groupKey,box:groupBox,members:group.map(instance=>({mesh:instance.mesh,eid:instance.eid})),construction});
 };
 // perf-impl-108 PARENT TIER: the octree node is only the candidate set, never
 // the form. Its members split into groups of placements that TOUCH in the
 // world (parentGroups); a group is one parent only when its upper profile
 // loses nothing (every member compileMeshProfileSafe, and parentAdequate: no
 // opening between members on the page). Unknown members are not safe. Every other
 // member is listed singly, so its own producer draws its openings/contacts.
 const pieces:{id?:number;members:readonly SourceLodInstance[]}[]=[],rest:SourceLodInstance[]=[];
 for(const node of parents){
  const box=exterior&&perspective?projectedBox(node.bounds,clip,width,height):undefined;
  const key=node.key??Math.min(...node.instances.map(instance=>instance.eid));
  const pending=node.instances.some(instance=>instance.boundsPending);
  // perf-impl-96, unchanged: a node whose whole box fits ONE native cell is one
  // aggregate (nothing inside one cell can be spelled as an opening).
  if(box&&!pending&&box[2]-box[0]<maximumX&&box[3]-box[1]<maximumY){
   const construction=composeParentConstruction(node.instances);
   // An absent construction is the incumbent source contract, not evidence
   // that the old one-cell parent is unsafe. Only a known replacement that
   // cannot preserve its relations forces the legal child fallback.
   if(construction||node.instances.every(instance=>instance.construction===undefined)){adopt(key,box,node.instances,construction);continue;}
   rest.push(...node.instances);continue;
  }
  if(box&&!pending&&parentBox(box,PARENT_LEAVE_ROWS,width,height,pageWidth,pageHeight))for(const group of cache.groups(key,node.instances,identity,profileSafe))pieces.push(group);
  else rest.push(...node.instances);
 }
 // perf-impl-112 ACROSS NODES. The octree keeps a placement that crosses a
 // split plane in the node ABOVE the split; when that node is too large to be
 // a parent, the placement used to be listed singly (its own one-glyph stamp)
 // and the sheets beside it, in the child nodes, could never group with it, so
 // a building straddling a split became two parents and a stamp. Now every
 // small, settled, profile-safe single is a candidate piece too, and pieces
 // that touch in the WORLD (membersTouch, the same rule) merge into one group,
 // whatever node they came from. A merged group's id is allocated by its exact
 // content, so a hold carries over only to the identical group.
 // OPT-IN (`joinSingles`): only a host whose singles are ENTITY placements
 // (the plugin's octree workset) passes true. The physical/Y8 selector lists
 // its producer rows as singles on purpose (their carrier owns the row-level
 // gate and its owner identity is producer-tagged): with the default they
 // never join an entity aggregate. Node groups still merge across nodes.
 const singles:SourceLodInstance[]=[];
 for(const instance of instances){
  const box=joinSingles&&exterior&&perspective&&!instance.settling&&!instance.boundsPending&&instance.bounds&&profileSafe(instance.mesh)?projectedBox(instance.bounds,clip,width,height):undefined;
  if(box&&parentBox(box,PARENT_LEAVE_ROWS,width,height,pageWidth,pageHeight))pieces.push({members:[instance]});else singles.push(instance);
 }
 // Sweep on world x (sorted piece bounds, the widest member tolerance), then
 // the exact member test. Unsafe members are pieces of their own and never join.
 const piece=pieces.map(p=>{
  const b=[Infinity,Infinity,Infinity,-Infinity,-Infinity,-Infinity];let tol=0;
  const joins=p.members.every(m=>!!m.bounds&&profileSafe(m.mesh));
  for(const m of p.members)if(m.bounds){for(let k=0;k<3;k++){b[k]=Math.min(b[k]!,m.bounds[k]!);b[k+3]=Math.max(b[k+3]!,m.bounds[k+3]!);}tol=Math.max(tol,PARENT_TOUCH*touchSide(m.bounds));}
  return {b,tol,joins};
 });
 const link=pieces.map((_,i)=>i),find=(i:number):number=>{while(link[i]!==i){link[i]=link[link[i]!]!;i=link[i]!;}return i;};
 const order=pieces.map((_,i)=>i).filter(i=>piece[i]!.joins).sort((a,c)=>piece[a]!.b[0]!-piece[c]!.b[0]!);
 for(let s=0;s<order.length;s++){const i=order[s]!,a=piece[i]!;
  for(let t=s+1;t<order.length;t++){const j=order[t]!,c=piece[j]!;
   if(c.b[0]!>a.b[3]!+Math.max(a.tol,c.tol))break;
   if(find(i)===find(j)||![0,1,2].every(k=>a.b[k]!<=c.b[k+3]!+Math.max(a.tol,c.tol)&&c.b[k]!<=a.b[k+3]!+Math.max(a.tol,c.tol)))continue;
   if(pieces[i]!.members.some(m=>pieces[j]!.members.some(n=>membersTouch(m.bounds!,n.bounds!)))){const x=find(i),y=find(j);link[Math.max(x,y)]=Math.min(x,y);}
  }
 }
 const merged=new Map<number,number[]>();pieces.forEach((_,i)=>{const r=find(i);const g=merged.get(r)??[];g.push(i);merged.set(r,g);});
 const tryParent=(group:readonly SourceLodInstance[],groupKey:number):boolean=>{
  if(group.length<2)return false;
  const groupBox=unionBox(group.map(instance=>projectedBox(instance.bounds!,clip,width,height)));
  if(!groupBox||!parentBox(groupBox,held.has(groupKey)?PARENT_LEAVE_ROWS:PARENT_ENTER_ROWS,width,height,pageWidth,pageHeight)||
   !group.every(instance=>profileSafe(instance.mesh))||!parentAdequate(group,clip,width,height,pageWidth,pageHeight,groupBox))return false;
  const construction=composeParentConstruction(group);
  // Keep the certified incumbent parent while source-program compilation is
  // still old/unknown. A known partial/invalid replacement remains ineligible
  // and descends instead of silently losing openings or contacts.
  if(!construction&&!group.every(instance=>instance.construction===undefined))return false;
  adopt(groupKey,groupBox,group,construction);return true;
 };
 for(const parts of merged.values()){
  const group=parts.flatMap(i=>pieces[i]!.members);
  const only=parts.length===1?pieces[parts[0]!]!:undefined;
  if(group.length>1&&tryParent(group,only?.id??cache.contentId(group.map(m=>({ref:identity(m.eid),mesh:m.mesh,revision:instanceSourceRevision(m)})))))continue;
  // The first legal alternative, in fixed order: a merged group that fails
  // (too tall, or an opening between its pieces) falls back to its node
  // groups exactly as perf-108 judged them; merged singles stay single.
  if(parts.length>1){for(const i of parts){const p=pieces[i]!;if(p.id===undefined||!tryParent(p.members,p.id))rest.push(...p.members);}}
  else rest.push(...group);
 }
 instances=[...singles,...rest];
 for(const instance of instances){
  const mesh=instance.mesh;
  if(instance.boundsPending){defer.add(mesh);seen.add(mesh);continue;}
  const box=!instance.settling&&instance.bounds?projectedBox(instance.bounds,clip,width,height):undefined;
  // The draw sphere frontier still reserves EVERY native indirect instance.
  // Its false positives need no topology before this camera can draw. Test
  // the complete posed box, including the pixel/roundoff pad above. Clipped,
  // unknown or moving bounds never certify absence.
  if(box&&(box[2]<0||box[0]>=width||box[3]<0||box[1]>=height))continue;
  seen.add(mesh);
  const smallKey=`${mesh}:${identity(instance.eid)}:${instanceSourceRevision(instance)??0}`;
  const nativeHeight=box?(box[3]-box[1])*pageHeight/Math.max(1,height)/17:Infinity;
  const nativeWidth=box?Math.round((box[2]-box[0])*pageWidth/Math.max(1,width)):0;
  const small=!!(exterior&&perspective&&box&&box[0]>=0&&box[1]>=0&&box[2]<=width&&box[3]<=height&&instance.smallParent?.replaceable&&profileSafe(mesh)&&smallParentPoseSupported(instance,clip,pageWidth,pageHeight)&&
   nativeHeight<(heldSmall.has(smallKey)?PARENT_LEAVE_ROWS:PARENT_ENTER_ROWS)&&
   smallAvailable?.(instance.smallParent,Math.max(1,Math.round(nativeHeight)),nativeWidth));
  if(!exterior||!perspective||!box||(!small&&(box[2]-box[0]>=maximumX||box[3]-box[1]>=maximumY))){const members=full.get(mesh)??[];members.push(instance.eid);full.set(mesh,members);continue;}
  if(small)smallOut.add(smallKey);
  const members=far.get(mesh)??[];members.push({eid:instance.eid,box,...(small?{smallParent:instance.smallParent,sourceRevision:instance.sourceRevision}:{})});far.set(mesh,members);
 }
 const outside=new Set<number>();for(const instance of instances)if(!seen.has(instance.mesh))outside.add(instance.mesh);
 // A deferred mesh keeps its bounded warmup; any full member ends it.
 for(const mesh of defer){if(full.has(mesh))defer.delete(mesh);else far.delete(mesh);}
 // Wholly full meshes stay out of both maps (unchanged meaning).
 for(const mesh of [...full.keys()])if(!far.has(mesh))full.delete(mesh);
 // A deferred mesh's members left `far` above (bounded warmup); the aggregate
 // keeps every member that is still far, so none is drawn by nobody.
 const parentsOut=aggregates.map(node=>({...node,members:node.members.filter(member=>
  far.get(member.mesh)?.some(candidate=>candidate.eid===member.eid&&candidate.parent===node.key)===true
 )})).filter(node=>node.members.length);
 return {far,parents:parentsOut,heldParents,heldSmall:smallOut,defer,outside,full,exterior};
}

/** RESIDENCY across cameras (perf-impl-94): a mesh needs its full program when
 * any camera draws a member at full, and its compiled far form when any camera
 * draws a far member. Each camera's own selection still decides its drawing;
 * this merge never promotes one camera's far members to full detail. */
export function mergeSourceLod(selections:readonly {readonly selection:SourceLodSelection;readonly meshes:readonly number[]}[]):SourceLodSelection{
 const far=new Map<number,FarFormInstance[]>(),defer=new Set<number>(),needFull=new Set<number>(),outside=new Set<number>(),shown=new Set<number>();
 for(const {selection,meshes}of selections)for(const mesh of meshes){
  if(selection.outside?.has(mesh))continue;
  shown.add(mesh);
  if(selection.defer.has(mesh)){defer.add(mesh);continue;}
  const members=selection.far.get(mesh);
  if(members){const all=far.get(mesh)??[];all.push(...members);far.set(mesh,all);if(selection.full?.has(mesh))needFull.add(mesh);}
  else needFull.add(mesh);
 }
 for(const {selection,meshes}of selections)for(const mesh of meshes)if(!shown.has(mesh)&&selection.outside?.has(mesh))outside.add(mesh);
 for(const mesh of defer){if(needFull.has(mesh))defer.delete(mesh);else far.delete(mesh);}
 const full=new Map<number,readonly number[]>();
 for(const mesh of needFull)if(far.has(mesh))full.set(mesh,[]);
 return {far,defer,outside,full,exterior:selections.every(item=>item.selection.exterior)};
}
