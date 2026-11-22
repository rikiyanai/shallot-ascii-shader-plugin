import {FAR_STRIP_BELOW_ROWS,NEAR_FROM_ROWS,LOD_TIER_HOLD} from './form-depths.ts';
import {latticeOpening} from './compiled-construction-topology.ts';
import type {SourceSmallParent} from './source-small-parent.ts';
import {aliasCoherentParentProduct,CoherentParentCache,isCoherentParentRejected,type CoherentParentMember} from './coherent-parent-union.ts';













export const PARENT_ENTER_ROWS=FAR_STRIP_BELOW_ROWS*LOD_TIER_HOLD,PARENT_LEAVE_ROWS=FAR_STRIP_BELOW_ROWS/LOD_TIER_HOLD;
/** A complete source principal product replaces the fine producer in the
 * existing middle tier too, not just below one cell. The hold belongs to the
 * exact source lifetime/revision and uses the same native-page size bands. */
export const PRINCIPAL_ENTER_ROWS=NEAR_FROM_ROWS*LOD_TIER_HOLD,PRINCIPAL_LEAVE_ROWS=NEAR_FROM_ROWS/LOD_TIER_HOLD;
/** Shared CPU/GPU native plan capacity covers both held tiers and the row
 * intersected by an endpoint. A parent-only cap rejects valid principals. */
export const SOURCE_WHOLE_NATIVE_ROW_PLAN_MAX=Math.ceil(Math.max(PARENT_LEAVE_ROWS,PRINCIPAL_LEAVE_ROWS))+1;
/** Page columns one parent profile may span (the emitter's workgroup table). */
export const PARENT_MAX_COLUMNS=64;

export interface SourceLodInstance {
 readonly eid:number;readonly mesh:number;readonly bounds?:readonly number[];readonly settling:boolean;
 /** Producer authority for the complete posed bound consumed by page-scale
  * level selection. Undefined is never a license to use the rest mesh. */
 readonly boundsAuthority?:'native-affine'|'entity-cull-sphere'|'engine-deformed-cull-sphere'|'material-source-cull-sphere';
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
/** Resident source-owner identity returned by a camera-local native refusal.
 * The first word is the resident domain (including the producer high bit when
 * the field owner uses it); the second is the retained source owner. */
export type SourceContourRetryOwner=readonly [number,number];
/** Stable key for one camera/job refusal scope. This is not a product or
 * construction identity and must never be cached globally. */
export function sourceContourRetryKey(mesh:number,eid:number):string{return `${mesh>>>0}:${eid>>>0}`;}
export function sourceContourRetrySet(owners:readonly SourceContourRetryOwner[]):ReadonlySet<string>{
 const result=new Set<string>();for(const [mesh,eid] of owners)result.add(sourceContourRetryKey(mesh,eid));return result;
}
/** A complete source-owner extent, not a selected edge or an image glyph. */



export interface FarFormInstance {
 readonly eid:number;readonly box:readonly [number,number,number,number];readonly parent?:number;
 readonly smallParent?:SourceSmallParent;
 readonly sourceRevision?:number;
}


export type SourceWholePartRole='cap'|'turn'|'flank'|'return';
/** Internal far-lowering tag. It is derived from which source relation list a
 * passage came from; it is not a new source/scene-specific role. */
export type SourceWholeRelationKind='opening'|'contact'|'protected-void'|'paper';
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
 /** Present only on the far lowering's relation pseudo-part. */
 readonly relationKind?:SourceWholeRelationKind;
}
export interface SourceWholeOpening {
 readonly feature:number;readonly ports:readonly number[];
 readonly contributors:readonly (readonly [number,number])[];
 /** Ordered source-space boundary/centerline samples.  Relation metadata
  * without this immutable geometry is not a drawable parent replacement. */
 readonly geometry?:readonly SourceWholePoint[];
 readonly closed?:boolean;
 /** Native whole-part context for the relation boundary.  This is compiled
  * once from the same source graph; it is never elected from page fragments. */
 readonly entryRef?:number;
 readonly context?:number;
 readonly role?:SourceWholePartRole;
 /** Instance pose for this relation's source samples. */
 readonly worldFromSource?:ArrayLike<number>;
}
/** One bounded compound-parent class.  The lists are stable source feature and
 * vertex identities, not camera fragments or glyph choices.  A product is
 * admitted only when its planar outer passages, every protected opening loop,
 * and every required contact/ground anchor are all present in the same source
 * revision.  Other relation classes deliberately retain their children. */
export interface SourceWholeCompound {
 readonly kind:'planar-openings-contact'|'planar-contact';
 readonly outerFeatures:readonly number[];
 readonly openingFeatures:readonly number[];
 readonly contactFeatures:readonly number[];
 readonly anchorVertices:readonly number[];
}
/** Paper attached to a part has that part's pose unless it names its own.
 * Every selector/lowerer must see this same list; union compilation is not
 * guaranteed to precede direct principal/far playback. */
export function sourceWholePaperOf(value:SourceWholeConstruction):readonly SourceWholeOpening[]{
 return [...(value.paper??[]),...value.parts.flatMap(part=>(part.paper??[]).map(paper=>({...paper,worldFromSource:paper.worldFromSource??part.worldFromSource})))];
}
/** Source-owned parent construction. The optional-looking arrays are required
 * by the contract so an empty value means "known empty", not "unknown". */
export interface SourceWholeConstruction {
 readonly revision:number;
 /** Source compiler certified every component/section in this principal
  * product. Legacy four-role presence alone is not a coverage certificate. */
 readonly principalComplete?:true;
 readonly parts:readonly SourceWholePartRef[];
 readonly openings:readonly SourceWholeOpening[];
 readonly contacts:readonly SourceWholeOpening[];
 readonly protectedVoids:readonly SourceWholeOpening[];
 /** Explicit paper/air passages owned by the compiled source graph. */
 readonly paper?:readonly SourceWholeOpening[];
 /** Complete compound-parent product, present only for the bounded class. */
 readonly compound?:SourceWholeCompound;
}

/**
 * A required contact is an open, source-owned path.  It is not a closed native
 * part: when it occupies one page row the resident whole-part bank quite
 * correctly has no two-anchor height variant, but the far source-part emitter
 * can still carry the ordered contact segment into the shared path consumer.
 * Keep this predicate at the source/far seam so the selector cannot admit a
 * one-row contact unless that producer contract is actually present.
 */
export function sourceWholeContactPathReady(value:SourceWholeOpening|undefined):boolean{
 const entryRef=value?.entryRef,context=value?.context,pose=value?.worldFromSource;
 return !!value&&value.closed===false&&!!value.geometry&&value.geometry.length>=2&&
  typeof entryRef==='number'&&Number.isSafeInteger(entryRef)&&entryRef>=1&&
  typeof context==='number'&&Number.isSafeInteger(context)&&context>0&&
  !!value.role&&!!pose&&pose.length===16&&
  Array.from(pose).every(number=>Number.isFinite(number));
}

/** Shared source/far marker for the existing open-hard contact consumer. */
export const SOURCE_CONTACT_BOUNDARY_FLAG=0x04000000;
export const SOURCE_CONTACT_LINE_CLASS=2;
/** Run.paper.w marker for a closed source-owned contour that may use the
 * canonical hard-boundary grammar when the retained whole entry cannot fit. */
export const SOURCE_COARSE_CONTOUR_FLAG=0x08000000;

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
 readonly principalComplete?:true;
 readonly parts:readonly SourceWholePartTemplate[];
 readonly openings:readonly SourceWholeOpeningTemplate[];
 readonly contacts:readonly SourceWholeOpeningTemplate[];
 readonly protectedVoids:readonly SourceWholeOpeningTemplate[];
 readonly paper?:readonly SourceWholeOpeningTemplate[];
 readonly compound?:SourceWholeCompound;
}

const bindOpening=(value:SourceWholeOpeningTemplate,mesh:number,eid:number,worldFromSource?:ArrayLike<number>):SourceWholeOpening|undefined=>value.sourceContributors.length?({
 feature:value.feature,ports:value.ports,contributors:value.sourceContributors.map(()=>[mesh,eid] as const),
 ...(value.geometry?{geometry:value.geometry}:{}),...(value.closed!==undefined?{closed:value.closed}:{}),
 ...(value.entryRef!==undefined?{entryRef:value.entryRef}:{}),...(value.context!==undefined?{context:value.context}:{}),
 ...(value.role!==undefined?{role:value.role}:{}),...(worldFromSource?{worldFromSource}:{}),
}):undefined;
/** Bind one immutable source template to one posed instance. This is a pure
 * identity adapter: no camera, mesh scan, glyph lookup or source recompilation
 * occurs here. */
const boundWholeConstructions=new WeakMap<SourceWholeConstructionTemplate,Map<string,{readonly pose:readonly number[]|undefined;readonly construction:SourceWholeConstruction}>>();
export function bindSourceWholeConstruction(template:SourceWholeConstructionTemplate|undefined,mesh:number,eid:number,worldFromSource?:ArrayLike<number>):SourceWholeConstruction|undefined{
 if(!template||!Number.isSafeInteger(mesh)||!Number.isSafeInteger(eid))return;
 let bindings=boundWholeConstructions.get(template);if(!bindings){bindings=new Map();boundWholeConstructions.set(template,bindings);}
 const key=`${mesh}:${eid}:${template.revision}`,cached=bindings.get(key);
 const incomingPose=worldFromSource;
 const samePose=cached&&(incomingPose===undefined?cached.pose===undefined:!!cached.pose&&cached.pose.length===incomingPose.length&&cached.pose.every((value,index)=>value===incomingPose[index]));
 if(samePose)return cached.construction;
 if(template.parts.some(part=>part.sourceContributors.length===0)||[...template.openings,...template.contacts,...template.protectedVoids,...(template.paper??[])].some(value=>value.sourceContributors.length===0))return;
 // Engine matrix views are mutable. The generation-owned bound construction
 // must retain its own pose, otherwise the cache and an in-flight native job
 // would silently observe a later pose through the same array identity.
 const pose=worldFromSource===undefined?undefined:Array.from(worldFromSource);
 worldFromSource=pose;
 const bindParts=template.parts.map(part=>({feature:part.feature,entryRef:part.entryRef,context:part.context,role:part.role,ports:part.ports,contributors:[[mesh,eid] as const],...(part.geometry?{geometry:part.geometry,closed:part.closed}:{}),...(worldFromSource?{worldFromSource}:{}),...(part.paper?{paper:part.paper.flatMap(value=>{const bound=bindOpening(value,mesh,eid,worldFromSource);return bound?[bound]:[];})}: {})}));
 const bindList=(values:readonly SourceWholeOpeningTemplate[])=>values.flatMap(value=>{const bound=bindOpening(value,mesh,eid,worldFromSource);return bound?[bound]:[];});
 const construction={revision:template.revision,...(template.principalComplete?{principalComplete:true as const}:{}),parts:bindParts,openings:bindList(template.openings),contacts:bindList(template.contacts),protectedVoids:bindList(template.protectedVoids),...(template.paper?{paper:bindList(template.paper)}:{}),...(template.compound?{compound:template.compound}:{})};
 if(bindings.size>=65536)bindings.delete(bindings.keys().next().value!);
 bindings.set(key,{pose,construction});return construction;
}
/** Selection uses mesh handles; native fields use resident source domains
 * (and producer owner tags). Convert identities at that adapter seam, without
 * changing source features, geometry, pose, or anchor correspondence. */
const remappedWholeConstructions=new WeakMap<SourceWholeConstruction,{readonly contributors:readonly (readonly [number,number])[];readonly residents:Map<string,SourceWholeConstruction>}>();
export function remapSourceWholeConstruction(value:SourceWholeConstruction|undefined,owner:(mesh:number,eid:number)=>readonly [number,number]|undefined):SourceWholeConstruction|undefined{
 if(!value)return;
 let memo=remappedWholeConstructions.get(value);
 if(!memo){
  const unique=new Map<string,readonly [number,number]>();
  const collect=(items:readonly {readonly contributors:readonly (readonly [number,number])[]}[])=>{for(const item of items)for(const pair of item.contributors)unique.set(`${pair[0]}:${pair[1]}`,pair);};
  collect(value.parts);for(const part of value.parts)collect(part.paper??[]);
  collect([...value.openings,...value.contacts,...value.protectedVoids,...(value.paper??[])]);
  memo={contributors:[...unique.values()],residents:new Map()};remappedWholeConstructions.set(value,memo);
 }
 const mappedOwners=new Map<string,readonly [number,number]>();
 for(const [mesh,eid] of memo.contributors){const mapped=owner(mesh,eid);if(!mapped)return;mappedOwners.set(`${mesh}:${eid}`,mapped);}
 const mappingKey=[...mappedOwners.values()].map(pair=>`${pair[0]}:${pair[1]}`).join('|'),cached=memo.residents.get(mappingKey);
 if(cached)return cached;
 const pairs=(contributors:SourceWholePartRef['contributors'])=>{
  const result:(readonly [number,number])[]=[];
  for(const [mesh,eid] of contributors){const mapped=mappedOwners.get(`${mesh}:${eid}`);if(!mapped)return;result.push(mapped);}
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
 const resident={...value,parts,openings:knownOpenings,contacts,protectedVoids,...(paper?{paper}:{}),...(value.compound?{compound:value.compound}:{})};
 aliasCoherentParentProduct(value,resident);
 if(memo.residents.size>=8)memo.residents.delete(memo.residents.keys().next().value!);
 memo.residents.set(mappingKey,resident);return resident;
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

 readonly parents?:readonly SourceLodAggregate[];

 readonly heldParents?:ReadonlySet<number>;
 readonly heldSmall?:ReadonlySet<string>;
 readonly defer:ReadonlySet<number>;
 /** Every instance is outside this camera's padded, same-pose page box. */
 readonly outside?:ReadonlySet<number>;





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


export interface SourceLodParent {readonly bounds:readonly number[];readonly instances:readonly SourceLodInstance[];readonly key?:number}








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


export const PARENT_GROUP_ID_BIT=0x80000000;
export interface ParentGroup {readonly id:number;readonly members:readonly SourceLodInstance[]}
interface CachedMember {readonly eid:number;readonly ref:number;readonly mesh:number;readonly safe:boolean;readonly sourceRevision?:number;readonly poseRevision?:number;readonly bounds?:readonly number[]}
const sameBounds=(a:readonly number[]|undefined,b:readonly number[]|undefined)=>a===b||(!!a&&!!b&&a.length===b.length&&a.every((v,i)=>v===b[i]));
const instanceSourceRevision=(instance:SourceLodInstance):number|undefined=>instance.sourceRevision??instance.construction?.revision;








export class ParentGroupCache{
 private readonly nodes=new Map<number,{members:readonly CachedMember[];groups:readonly (readonly number[])[];ids:readonly number[]}>();
 private readonly ids=new Map<string,number>();private next=1;
 /** Source unions are generation-owned; camera selection reuses them. */
 readonly coherentParents=new CoherentParentCache();
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
/**
 * Admission at the source/renderer seam.  The selector owns projection and
 * source geometry; the renderer owns the packed resident table and the exact
 * band-2 -> band-0 fallback (record, growth rows, repeats and page bounds).
 * Keeping this as one callback prevents the source selector from importing the
 * authoring role bank or searching native glyphs on every camera frame.
 */
export type SourceWholeNativePartAdmission=(role:SourceWholePartRole,context:number,rows:number,width:number,origin:number)=>boolean;
/** Validate the bounded compound-parent product independently of camera size.
 * This is the source/product gate; projection and native row admission remain
 * the selector's separate executable gate below. */
export function sourceWholeCompoundReady(value:SourceWholeConstruction|SourceWholeConstructionTemplate|undefined):value is (SourceWholeConstruction|SourceWholeConstructionTemplate)&{readonly compound:SourceWholeCompound}{
 const compound=value?.compound;
 const contactOnly=compound?.kind==='planar-contact';
 if(!compound||(compound.kind!=='planar-openings-contact'&&!contactOnly)||!compound.outerFeatures.length||!compound.contactFeatures.length||!compound.anchorVertices.length||(!contactOnly&&!compound.openingFeatures.length)||(contactOnly&&compound.openingFeatures.length!==0))return false;
 const u32=(word:number)=>Number.isSafeInteger(word)&&word>=0&&word<=U32_MAX;
 const distinct=(values:readonly number[])=>values.every(u32)&&new Set(values).size===values.length;
 if(!distinct(compound.outerFeatures)||!distinct(compound.openingFeatures)||!distinct(compound.contactFeatures)||!distinct(compound.anchorVertices))return false;
 const parts=new Set(value.parts.map(part=>part.feature)),openings=new Set(value.openings.map(item=>item.feature)),contacts=new Set(value.contacts.map(item=>item.feature));
 const exact=(declared:readonly number[],actual:ReadonlySet<number>)=>declared.length===actual.size&&declared.every(feature=>actual.has(feature));
 const contactPorts=new Set(value.contacts.flatMap(contact=>contact.ports));
 if(contactOnly&&(value.openings.length>0||value.protectedVoids.length>0||!!value.paper?.length||value.parts.some(part=>part.paper?.length)))return false;
 return exact(compound.outerFeatures,parts)&&exact(compound.openingFeatures,openings)&&exact(compound.contactFeatures,contacts)&&exact(compound.anchorVertices,contactPorts);
}
/** Validate only the resident source metadata. This never inspects mesh data,
 * camera pixels or the glyph bank. */
export function sourceWholeConstructionReady(value:SourceWholeConstruction|undefined):value is SourceWholeConstruction{
 // Source revisions are unsigned counters, beginning at zero in both the
 // static semantic baker and physical producers. Zero is not an absent
 // product; identity/coverage/native admission are checked independently.
 if(!value||!Number.isSafeInteger(value.revision)||value.revision<0||value.revision>U32_MAX||!value.parts.length)return false;
 const roles=new Set(value.parts.map(part=>part.role));
 // The legacy whole product still requires all four retained role families.
 // The bounded planar compound is different: its outer boundary may be a
 // single native flank/return family plus the explicit opening/contact
 // passages, so requiring absent cap/turn records would reject a legal source
 // construction before the product reaches the consumer.
 if(!value.compound&&WHOLE_PART_ROLES.some(role=>!roles.has(role)))return false;
 if(value.compound&&!roles.size)return false;
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
  if(opening.geometry&&(!opening.geometry.length||opening.geometry.some(point=>point.length!==3||point.some(value=>!Number.isFinite(value)))))return false;
  if(opening.closed!==undefined&&typeof opening.closed!=='boolean')return false;
  if(opening.entryRef!==undefined&&(!Number.isSafeInteger(opening.entryRef)||opening.entryRef<1||opening.entryRef>U32_MAX))return false;
  if(opening.context!==undefined&&(!Number.isSafeInteger(opening.context)||opening.context===0||opening.context>U32_MAX))return false;
  if(opening.role!==undefined&&!WHOLE_PART_ROLES.includes(opening.role))return false;
  if(opening.worldFromSource&&(opening.worldFromSource.length!==16||Array.from(opening.worldFromSource).some(value=>!Number.isFinite(value))))return false;
 }
 if(value.compound&&!sourceWholeCompoundReady(value))return false;
 return true;
}
/** A relation-bearing parent is drawable only when each relation carries its
 * ordered source samples, native context and posed identity.  Older baked
 * metadata remains conservative and descends to children. */
export function sourceWholeConstructionRenderable(value:SourceWholeConstruction|undefined):value is SourceWholeConstruction{
 if(!sourceWholeConstructionReady(value))return false;
 // A resident product may have passed CPU admission and still be rejected by
 // the actual far producer (coverage, native lowering, or allocation). Keep
 // that product out of the ordinary selection path so its complete incumbent
 // children remain available; a camera-local contour retry supplies an
 // explicit owner scope rather than weakening this malformed-generation guard.
 if(isCoherentParentRejected(value))return false;
 const relation=[...value.openings,...value.contacts,...value.protectedVoids,...sourceWholePaperOf(value)];
 // A relation is a replacement construction, not a hint to inherit the first
 // part's native turn.  If its own compiled passage is incomplete, the caller
 // must retain/descend to the incumbent children rather than silently drawing
 // a different part's grammar through the opening.
 if(relation.length&&value.parts.some(part=>!part.geometry||part.geometry.length<2||typeof part.closed!=='boolean'||!part.worldFromSource))return false;
 return relation.every(item=>!!item.geometry&&item.geometry.length>=2&&typeof item.closed==='boolean'&&
  Number.isSafeInteger(item.entryRef)&&Number.isSafeInteger(item.context)&&!!item.role&&!!item.worldFromSource);
}
/**
 * A source-owned terminal contour for a complete principal product.  This is
 * deliberately stricter than "has points": the contour must be a closed,
 * ordered source passage with a real edge at every step.  It is the bounded
 * fallback for a valid principal part whose projected height is one native
 * row; it is not a box, a dot, an eye-lid recipe, or permission to pick edges
 * again during playback.
 */
export function sourceWholeCoarseContourReady(part:SourceWholePartRef|undefined):boolean{
 if(!part||part.closed!==true||!part.geometry||part.geometry.length<3||part.paper?.length)return false;
 for(let index=0;index<part.geometry.length;index++){
  const a=part.geometry[index]!,b=part.geometry[(index+1)%part.geometry.length]!;
  if(a.some(value=>!Number.isFinite(value))||b.some(value=>!Number.isFinite(value)))return false;
  if(a[0]===b[0]&&a[1]===b[1]&&a[2]===b[2])return false;
 }
 return true;
}
/** Stable product fingerprint used by the far-form cache.  Source revision is
 * part of the identity, but geometry/pose/relation edits must also retire a
 * same-revision resident instead of reusing stale parent bytes. */
export function sourceWholeConstructionProductHash(value:SourceWholeConstruction|SourceWholeConstructionTemplate|undefined):number|undefined{
 if(!value)return;
 let hash=2166136261>>>0;const mix=(word:number)=>{hash=Math.imul(hash^(word>>>0),16777619)>>>0;};
 const floatValue=new Float32Array(1),floatWord=new Uint32Array(floatValue.buffer),mixFloat=(number:number)=>{floatValue[0]=number;mix(floatWord[0]!);};
 const contributorsOf=(item:{readonly contributors?:readonly (readonly [number,number])[];readonly sourceContributors?:readonly number[]})=>item.contributors??item.sourceContributors??[];
 const list=(items:readonly (SourceWholeOpening|SourceWholeOpeningTemplate)[])=>{mix(items.length);for(const item of items){mix(item.feature);for(const port of item.ports)mix(port);for(const contributor of contributorsOf(item))typeof contributor==='number'?mix(contributor):(()=>{mix(contributor[0]);mix(contributor[1]);})();for(const point of item.geometry??[])for(const number of point)mixFloat(number);mix(Number(item.closed===true));mix(item.entryRef??0);mix(item.context??0);mix(item.role?WHOLE_PART_ROLES.indexOf(item.role):0);for(const number of Array.from(item.worldFromSource??[]))mixFloat(number);}};
 const partList=value.parts;mix(value.revision);mix(Number(value.principalComplete===true));mix(partList.length);for(const part of partList){mix(part.feature);mix(part.entryRef);mix(part.context);mix(WHOLE_PART_ROLES.indexOf(part.role));for(const port of part.ports)mix(port);for(const contributor of contributorsOf(part))typeof contributor==='number'?mix(contributor):(()=>{mix(contributor[0]);mix(contributor[1]);})();for(const point of part.geometry??[])for(const number of point)mixFloat(number);mix(Number(part.closed===true));for(const number of Array.from(part.worldFromSource??[]))mixFloat(number);for(const paper of part.paper??[])list([paper]);}
 list(value.openings);list(value.contacts);list(value.protectedVoids);list(value.paper??[]);
 if(value.compound){mix(1);for(const feature of value.compound.outerFeatures)mix(feature);for(const feature of value.compound.openingFeatures)mix(feature);for(const feature of value.compound.contactFeatures)mix(feature);for(const vertex of value.compound.anchorVertices)mix(vertex);}
 return hash||1;
}
/**
 * Camera-bound eligibility for the far native consumer.  `renderable` proves
 * that relation metadata exists; this second gate proves that every retained
 * part and relation will produce the minimum native passage that
 * `farNativePart` actually executes at THIS projected size.  A parent that
 * fails this gate stays represented by its complete children; it is never
 * emitted as a skyline with its opening missing.
 */
export function sourceWholeConstructionExecutable(value:SourceWholeConstruction|undefined,clip:ArrayLike<number>,width:number,height:number,pageWidth:number,pageHeight:number,nativePart?:SourceWholeNativePartAdmission,requireNative=false):value is SourceWholeConstruction{
 if(!sourceWholeConstructionRenderable(value))return false;
 const relation=[...value.openings,...value.contacts,...value.protectedVoids,...sourceWholePaperOf(value)];
 // A relation-bearing parent must lower every ordinary part as well: a
 // profile-only parent would close the relation while its children are gone.
 // Relation-free legacy residents may still keep their incumbent profile.
 if(relation.length&&value.parts.some(part=>!part.geometry||part.geometry.length<2||typeof part.closed!=='boolean'||!part.worldFromSource))return false;
 if(!relation.length&&!requireNative)return true;
 // Relation metadata without the complete bounded product is not a parent
 // replacement.  Keep every incumbent child rather than closing an opening or
 // inventing a contact from whichever part happened to be first.
 if(relation.length&&!sourceWholeCompoundReady(value))return false;
 const project=(point:SourceWholePoint,pose:ArrayLike<number>):readonly [number,number]|undefined=>{
  const x=point[0]!,y=point[1]!,z=point[2]!,wx=pose[0]!*x+pose[4]!*y+pose[8]!*z+pose[12]!,wy=pose[1]!*x+pose[5]!*y+pose[9]!*z+pose[13]!,wz=pose[2]!*x+pose[6]!*y+pose[10]!*z+pose[14]!,ww=pose[3]!*x+pose[7]!*y+pose[11]!*z+pose[15]!;
  const cx=clip[0]!*wx+clip[4]!*wy+clip[8]!*wz+clip[12]!,cy=clip[1]!*wx+clip[5]!*wy+clip[9]!*wz+clip[13]!,cw=clip[3]!*wx+clip[7]!*wy+clip[11]!*wz+clip[15]!;
  if(!(cw>1e-5))return;return [((cx/cw)*.5+.5)*pageWidth,(.5-(cy/cw)*.5)*pageHeight];
 };
 const parts:{readonly part:SourceWholePartRef|SourceWholeOpening;readonly contact:boolean;readonly sourcePart:boolean}[] = [];
 parts.push(...value.parts.map(part=>({part,contact:false,sourcePart:true})),...value.openings.map(part=>({part,contact:false,sourcePart:false})),
  ...value.contacts.map(part=>({part,contact:true,sourcePart:false})),...value.protectedVoids.map(part=>({part,contact:false,sourcePart:false})),
  ...sourceWholePaperOf(value).map(part=>({part,contact:false,sourcePart:false})));
 const direction=(delta:readonly [number,number]):number=>Math.round((((Math.atan2(delta[1]!,delta[0]!)/(Math.PI*2))+1)%1)*8)&7;
 const rowPlanComplete=(projected:readonly (readonly [number,number])[],closed:boolean,low:number,high:number):boolean=>{
  const covered=new Uint8Array(Math.max(0,high-low+1)),segments=closed?projected.length:projected.length-1;
  for(let index=0;index<segments;index++){
   const a=projected[index]!,b=projected[(index+1)%projected.length]!;
   const first=Math.max(low,Math.floor(Math.min(a[1]!,b[1]!)/17)),last=Math.min(high,Math.floor(Math.max(a[1]!,b[1]!)/17));
   for(let row=first;row<=last;row++)if(row>=low&&row<=high)covered[row-low]=1;
  }
  return covered.every(value=>value!==0);
 };
 return parts.every(({part,contact,sourcePart})=>{
  const geometry=part.geometry,pose=part.worldFromSource;if(!geometry||geometry.length<2||!pose||pose.length!==16)return false;
  const projected=geometry.map(point=>project(point,pose));if(projected.some(point=>!point))return false;
  const projectedPoints=projected as readonly (readonly [number,number])[];
  const xs=projectedPoints.map(point=>point[0]!),ys=projectedPoints.map(point=>point[1]!);
  const minX=Math.min(...xs),maxX=Math.max(...xs),minY=Math.min(...ys),maxY=Math.max(...ys);
  // The native far consumer writes into the current page extent. A clipped
  // relation cannot prove its complete row plan here, so keep the incumbent
  // children rather than selecting a parent that may lose its opening at an
  // edge of the page.
  const lowRow=Math.floor(minY/17),highRow=Math.floor(maxY/17),origin=Math.round(minX),
   // A contact/ground anchor is a real source line and may project to one
   // column even when both endpoints share the same x. Reserve that column;
   // row coverage and the segment count below still reject a collapsed point.
   sourceWidth=Math.max(1,Math.round(maxX-minX)),rowCount=Math.ceil(pageHeight/17);
  // GPU farNativePart rounds the source origin and span before its atom/page
  // preflight. Float bounds alone can therefore admit a .5-tie one-pixel
  // overrun or an endpoint on the first invalid integer row.
  if(minX<0||maxX>pageWidth||origin<0||origin+sourceWidth>pageWidth||minY<0||maxY>pageHeight||lowRow<0||highRow>=rowCount)return false;
  const rows=highRow-lowRow+1,closed=part.closed===true;
  if(rows>SOURCE_WHOLE_NATIVE_ROW_PLAN_MAX)return false;
  // A required contact uses the existing open-hard boundary consumer, not a
  // fabricated whole-part bank entry. Its source geometry, pose and relation
  // identity remain in the packet; the ordinary planar queue owns spelling.
  const contactSpan=contact&&sourceWholeContactPathReady(part)&&projectedPoints.some((a,index)=>{
   if(index+1>=projectedPoints.length)return false;const b=projectedPoints[index+1]!;
   return Math.hypot(b[0]!-a[0]!,b[1]!-a[1]!)>=1;
  });
  // A complete principal source part may use the source-owned ordered contour
  // consumer when the resident whole-part bank cannot represent one row. All
  // relation/opening records remain on their existing native paths.
  const coarseContour=nativePart!==undefined&&value.principalComplete===true&&sourcePart&&relation.length===0&&sourceWholeCoarseContourReady(part as SourceWholePartRef)&&
   minX<maxX&&minY<maxY&&projectedPoints.some((a,index)=>{const b=projectedPoints[(index+1)%projectedPoints.length]!;return Math.hypot(b[0]-a[0],b[1]-a[1])>=1;});
  if(rows<2&&!contactSpan&&!coarseContour)return false;
  if(rows<2&&coarseContour)return true;
  if(!rowPlanComplete(projectedPoints,closed,lowRow,highRow))return false;
  const segments=closed?projectedPoints.length:projectedPoints.length-1;if(segments<1)return false;
  const firstDelta:[number,number]=[projectedPoints[0]![0]!-projectedPoints[1]![0]!,projectedPoints[0]![1]!-projectedPoints[1]![1]!];
  const lastIndex=segments-1,lastA=projectedPoints[lastIndex]!,lastB=projectedPoints[(lastIndex+1)%projectedPoints.length]!;
  const lastDelta:[number,number]=[lastB[0]!-lastA[0]!,lastB[1]!-lastA[1]!];
  const nativeContext=((part.context!>>6)&7)<<6|direction(firstDelta)|direction(lastDelta)<<3;
  if(contactSpan)return true;
  if(!part.role||!nativePart)return false;
  return nativePart(part.role,nativeContext,rows,sourceWidth,origin)||coarseContour;
 });
}
/** Combine immutable source programs without introducing a scene recipe. A
 * program can only represent the listed member identities; a missing member,
 * revision or protected relation therefore rejects the parent. */
export function composeParentConstruction(members:readonly SourceLodInstance[]):SourceWholeConstruction|undefined{
 if(!members.length||members.some(member=>!sourceWholeConstructionRenderable(member.construction)))return undefined;
 const identities=members.map(member=>[member.mesh,member.eid] as const),known=new Set(identities.map(pairKey));
 if(known.size!==identities.length)return undefined;
 const programs=members.map(member=>member.construction!);
 const relationPrograms=programs.filter(program=>program.openings.length||program.contacts.length||program.protectedVoids.length||sourceWholePaperOf(program).length);
 const validContributor=(pairs:readonly (readonly [number,number])[])=>pairs.length>0&&pairs.every(pair=>known.has(pairKey(pair)));
 if(programs.some(program=>program.parts.some(part=>!validContributor(part.contributors)||(part.paper??[]).some(item=>!validContributor(item.contributors)))||[...program.openings,...program.contacts,...program.protectedVoids,...sourceWholePaperOf(program)].some(item=>!validContributor(item.contributors))))return undefined;
 const parts=programs.flatMap(program=>program.parts);
 const openings=programs.flatMap(program=>program.openings),contacts=programs.flatMap(program=>program.contacts),protectedVoids=programs.flatMap(program=>program.protectedVoids);
 const floatValue=new Float32Array(1),floatWord=new Uint32Array(floatValue.buffer),bits=(value:number)=>{floatValue[0]=value;return floatWord[0]!;};
 const revision=programs.reduce((hash,program)=>{
  let next=(hash^program.revision)>>>0;const mix=(word:number)=>{next=Math.imul(next^(word>>>0),16777619)>>>0;};
  for(const part of program.parts){mix(part.feature);mix(part.entryRef);mix(part.context);mix(WHOLE_PART_ROLES.indexOf(part.role));mix(Number(part.closed===true));for(const value of Array.from(part.worldFromSource??[]))mix(bits(value));for(const point of part.geometry??[])for(const value of point)mix(bits(value));for(const paper of part.paper??[]){mix(paper.feature);mix(paper.entryRef??0);mix(paper.context??0);for(const point of paper.geometry??[])for(const value of point)mix(bits(value));}}
  for(const relation of [...program.openings,...program.contacts,...program.protectedVoids,...(program.paper??[])]){mix(relation.feature);mix(relation.entryRef??0);mix(relation.context??0);mix(relation.role?WHOLE_PART_ROLES.indexOf(relation.role):0);mix(Number(relation.closed===true));for(const value of Array.from(relation.worldFromSource??[]))mix(bits(value));for(const port of relation.ports)mix(port);for(const point of relation.geometry??[])for(const value of point)mix(bits(value));}
  if(program.compound){mix(1);for(const feature of program.compound.outerFeatures)mix(feature);for(const feature of program.compound.openingFeatures)mix(feature);for(const feature of program.compound.contactFeatures)mix(feature);for(const vertex of program.compound.anchorVertices)mix(vertex);}
  return next;
 },2166136261)>>>0;
 const paper=programs.flatMap(program=>sourceWholePaperOf(program));
 const compounds=programs.map(program=>program.compound).filter((value):value is SourceWholeCompound=>!!value);
 const contactOnly=!openings.length&&contacts.length>0&&!protectedVoids.length&&!paper.length&&compounds.every(value=>value.kind==='planar-contact');
 const compound=relationPrograms.length&&compounds.length===programs.length&&(openings.length>0||contactOnly)?{
  kind:(contactOnly?'planar-contact':'planar-openings-contact') as SourceWholeCompound['kind'],
  outerFeatures:[...new Set(compounds.flatMap(value=>value.outerFeatures))],
  openingFeatures:[...new Set(compounds.flatMap(value=>value.openingFeatures))],
  contactFeatures:[...new Set(compounds.flatMap(value=>value.contactFeatures))],
  anchorVertices:[...new Set(compounds.flatMap(value=>value.anchorVertices))],
 }:undefined;
 return {revision:revision||1,parts,openings,contacts,protectedVoids,...(paper.length?{paper}:{}),...(compound?{compound}:{})};
}






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
export function selectSourceLod(instances:readonly SourceLodInstance[],clip:ArrayLike<number>,width:number,height:number,pageWidth:number,pageHeight:number,exterior:boolean,perspective:boolean,parents:readonly SourceLodParent[]=[],held:ReadonlySet<number>=new Set(),profileSafe:(mesh:number)=>boolean=()=>false,identity:(eid:number)=>number=eid=>eid,cache:ParentGroupCache=new ParentGroupCache(),joinSingles=false,smallAvailable?:SmallConstructionAvailability,heldSmall:ReadonlySet<string>=new Set(),nativePart?:SourceWholeNativePartAdmission,refusedOwners?:ReadonlySet<string>):SourceLodSelection{
 // A workset may refer to one source through a node and a straddling single.
 // Canonicalize that producer identity ONCE, before choosing any frontier;
 // deduplicating only the emitted parent leaves duplicate singleton packets.
 const inputs=new Map<string,SourceLodInstance>();
 const once=(instance:SourceLodInstance)=>{
  const key=`${instance.mesh}:${instance.eid}`,prior=inputs.get(key);if(!prior){inputs.set(key,instance);return true;}
  const samePose=prior.worldFromSource===instance.worldFromSource||!!prior.worldFromSource&&!!instance.worldFromSource&&sameBounds(Array.from(prior.worldFromSource),Array.from(instance.worldFromSource));
  if(prior.sourceRevision!==instance.sourceRevision||prior.poseRevision!==instance.poseRevision||!sameBounds(prior.bounds,instance.bounds)||!samePose||
   prior.settling!==instance.settling||prior.boundsPending!==instance.boundsPending||sourceWholeConstructionProductHash(prior.construction)!==sourceWholeConstructionProductHash(instance.construction))
   throw Error(`MAIN source frontier has conflicting source instance mesh=${instance.mesh}, owner=${instance.eid}; one retained owner must name one source revision and pose`);
  return false;
 };
 instances=instances.filter(once);parents=parents.map(parent=>({...parent,instances:parent.instances.filter(once)})).filter(parent=>parent.instances.length>0);
 const isRefused=(instance:SourceLodInstance):boolean=>refusedOwners?.has(sourceContourRetryKey(instance.mesh,instance.eid))===true;
 const forcedFull=new Set<string>();
 const forceGroupFull=(group:readonly SourceLodInstance[]):void=>{for(const member of group)forcedFull.add(sourceContourRetryKey(member.mesh,member.eid));};
 const far=new Map<number,FarFormInstance[]>(),defer=new Set<number>(),full=new Map<number,number[]>(),seen=new Set<number>(),aggregates:SourceLodAggregate[]=[],heldParents=new Set<number>();
 const smallOut=new Set<string>();
 const compileSourceParent=(group:readonly SourceLodInstance[]):SourceWholeConstruction|undefined=>{
  // A union may simplify a complete source product, not repair an invalid
  // opening/contact certificate by rebuilding a different compound ledger.
  if(group.some(instance=>isRefused(instance)||!sourceWholeConstructionRenderable(instance.construction)||
   instance.construction.principalComplete!==true&&!sourceWholeCompoundReady(instance.construction)))return;
  const members:CoherentParentMember[]=group.map(instance=>({mesh:instance.mesh,eid:instance.eid,construction:instance.construction!,sourceRevision:instance.sourceRevision,poseRevision:instance.poseRevision}));
  const result=cache.coherentParents.getOrCompile(members);return result.status==='parent'?result.product.construction:undefined;
 };
 // Only a complete form smaller than ONE native cell may merge. A narrow
 // two-row house still holds its own construction, whatever its class.
 const maximumX=16*LOD_TIER_HOLD*width/Math.max(1,pageWidth),maximumY=17*LOD_TIER_HOLD*height/Math.max(1,pageHeight);

 // adequate octree node is ONE aggregate. Its members are far (residency, no
 // relations) and carry the node's key; the far aggregate emitter draws the
 // node once from all their pixels. A node that no longer fits (or an unknown
 // view) lists its members singly.
 const adopt=(groupKey:number,groupBox:[number,number,number,number],group:readonly SourceLodInstance[],construction?:SourceWholeConstruction)=>{
  heldParents.add(groupKey);
  for(const instance of group){seen.add(instance.mesh);const members=far.get(instance.mesh)??[];members.push({eid:instance.eid,box:groupBox,parent:groupKey});far.set(instance.mesh,members);}
  aggregates.push({key:groupKey,box:groupBox,members:group.map(instance=>({mesh:instance.mesh,eid:instance.eid})),construction});
 };

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

  // aggregate (nothing inside one cell can be spelled as an opening).
  if(box&&!pending&&box[2]-box[0]<maximumX&&box[3]-box[1]<maximumY){
   // A camera-local native refusal invalidates this node only for this job.
   // Expand every member to its complete child producer; do not let the
   // refused member re-enter another coarse representation below.
   if(node.instances.some(isRefused)){forceGroupFull(node.instances);rest.push(...node.instances);continue;}
   const construction=compileSourceParent(node.instances);
   const executable=construction&&sourceWholeConstructionExecutable(construction,clip,width,height,pageWidth,pageHeight,nativePart,true);

   // that the old one-cell parent is unsafe. Only a known replacement that
   // cannot preserve its relations forces the legal child fallback.
   if(executable||node.instances.every(instance=>instance.construction===undefined)){adopt(key,box,node.instances,executable?construction:undefined);continue;}
   rest.push(...node.instances);continue;
  }
  if(box&&!pending&&parentBox(box,PARENT_LEAVE_ROWS,width,height,pageWidth,pageHeight))for(const group of cache.groups(key,node.instances,identity,profileSafe))pieces.push(group);
  else rest.push(...node.instances);
 }

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
  if(group.some(isRefused)){forceGroupFull(group);return false;}
  const groupBox=unionBox(group.map(instance=>projectedBox(instance.bounds!,clip,width,height)));
  if(!groupBox||!parentBox(groupBox,held.has(groupKey)?PARENT_LEAVE_ROWS:PARENT_ENTER_ROWS,width,height,pageWidth,pageHeight)||
   !group.every(instance=>profileSafe(instance.mesh))||!parentAdequate(group,clip,width,height,pageWidth,pageHeight,groupBox))return false;
  const construction=compileSourceParent(group);
  const executable=construction&&sourceWholeConstructionExecutable(construction,clip,width,height,pageWidth,pageHeight,nativePart,true);
  // Keep the certified incumbent parent while source-program compilation is
  // still old/unknown. A known partial/invalid replacement remains ineligible
  // and descends instead of silently losing openings or contacts.
  if(!executable&&!group.every(instance=>instance.construction===undefined))return false;
  adopt(groupKey,groupBox,group,executable?construction:undefined);return true;
 };
 for(const parts of merged.values()){
  const group=parts.flatMap(i=>pieces[i]!.members);
  const only=parts.length===1?pieces[parts[0]!]!:undefined;
  if(group.length>1&&tryParent(group,only?.id??cache.contentId(group.map(m=>({ref:identity(m.eid),mesh:m.mesh,revision:instanceSourceRevision(m)})))))continue;
  // The first legal alternative, in fixed order: a merged group that fails
  // (too tall, or an opening between its pieces) falls back to its node

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
  if(forcedFull.has(sourceContourRetryKey(instance.mesh,instance.eid))||isRefused(instance)){
   const members=full.get(mesh)??[];members.push(instance.eid);full.set(mesh,members);continue;
  }
  // A compiled whole principal product is already a source-owned drawing.
  // Previously only tiny box/round metadata or a sub-cell singleton could
  // bypass fine expansion; a multi-component principal product was never
  // consumed at its middle tier. Give THIS copy one aggregate packet, with
  // its exact features/pose/relations, before any fine work. Native admission
  // is mandatory here even for relation-free products: the legacy skyline
  // exception must not authorize a new replacement that cannot draw.
  if(exterior&&perspective&&box&&!instance.settling&&!instance.boundsPending&&nativePart&&instance.construction?.principalComplete===true&&
   parentBox(box,PRINCIPAL_LEAVE_ROWS,width,height,pageWidth,pageHeight)){
   const key=cache.contentId([{ref:identity(instance.eid),mesh,revision:instanceSourceRevision(instance)}]);
   if(parentBox(box,held.has(key)?PRINCIPAL_LEAVE_ROWS:PRINCIPAL_ENTER_ROWS,width,height,pageWidth,pageHeight)){
    // Use the same generation/correspondence/status owner as multi-instance
    // parents. Passing raw template parts directly would bypass GPU rejection
    // retirement and could repeat a failed singleton generation forever.
    const construction=compileSourceParent([instance]);
    if(sourceWholeConstructionExecutable(construction,clip,width,height,pageWidth,pageHeight,nativePart,true)){
     adopt(key,box,[instance],construction);continue;
    }
   }
  }
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
