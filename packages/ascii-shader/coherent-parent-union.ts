import type {
 SourceWholeConstruction,
 SourceWholeOpening,
 SourceWholePartRef,
 SourceWholePartRole,
 SourceWholePoint,
} from './source-lod.ts';
import {
 unionSourceComponentContours,
 type SourceComponentContour,
} from './source-component-union.ts';

/**
 * MAIN source-owned coherent parent compiler.
 *
 * A coherent parent is a source-owned boundary graph, not an AABB or a
 * skyline.  Child source samples are posed into one parent coordinate space;
 * exact coplanar shared segments are owned once, while every relation remains
 * a separate source passage.  The provenance ledgers preserve the source
 * feature/member/pose identities that the current SourceWholePartRef ABI
 * cannot carry on a single de-duplicated run.
 */

export interface CoherentParentMember {
 readonly mesh:number;
 readonly eid:number;
 readonly construction:SourceWholeConstruction;
 /** Generation-owned invalidation seams; camera projection is not a key. */
 readonly sourceRevision?:number;
 readonly poseRevision?:number;
}

export interface CoherentPoseBinding {
 readonly feature:number;
 readonly mesh:number;
 readonly eid:number;
 readonly worldFromSource:readonly number[];
 readonly relation:boolean;
}

export interface CoherentPassage {
 readonly feature:number;
 readonly sourceFeatures:readonly number[];
 readonly role:SourceWholePartRole;
 readonly entryRef:number;
 readonly context:number;
 readonly ports:readonly [number,number];
 readonly contributors:readonly (readonly [number,number])[];
 readonly geometry:readonly SourceWholePoint[];
 readonly closed:boolean;
 readonly removedSharedJoin:boolean;
}

export interface CoherentParentProduct {
 readonly construction:SourceWholeConstruction;
 /** Every original part feature, including a feature aliased by a shared join. */
 readonly sourceFeatures:readonly number[];
 /** Source feature aliases retained when one shared edge becomes one run. */
 readonly featureAliases:readonly {readonly retained:number;readonly aliases:readonly number[]}[];
 readonly poseBindings:readonly CoherentPoseBinding[];
 readonly passages:readonly CoherentPassage[];
 readonly connectedComponents:number;
 readonly removedSharedJoins:number;
 readonly expandedPartCount:number;
}

export type CoherentParentResult=
 | {readonly status:'parent';readonly product:CoherentParentProduct}
 | {readonly status:'children';readonly reason:'missing-geometry'|'invalid-pose'|'invalid-contributor'|'invalid-relation'|'unsupported-plane'};

/**
 * Generation/member cache for source unions. The source construction is
 * immutable, so object identity plus its explicit source/pose revisions is a
 * cheaper and safer key than walking all geometry on every camera frame.
 * Projection, native-row admission and the parent hold remain outside this
 * cache and are evaluated by source-lod for the current camera.
 */
export class CoherentParentCache{
 private readonly entries=new Map<string,CoherentParentResult>();
 private readonly constructionIds=new WeakMap<SourceWholeConstruction,number>();
 private nextConstructionId=1;
 static readonly MAX_ENTRIES=16384;
 private constructionId(value:SourceWholeConstruction):number{
  const found=this.constructionIds.get(value);if(found!==undefined)return found;
  const id=this.nextConstructionId++;this.constructionIds.set(value,id);return id;
 }
 private key(members:readonly CoherentParentMember[]):string{
  return members.map(member=>`${member.mesh>>>0}:${member.eid>>>0}:${this.constructionId(member.construction)}:${member.sourceRevision??member.construction.revision}:${member.poseRevision??0}`).sort().join('|');
 }
 getOrCompile(members:readonly CoherentParentMember[]):CoherentParentResult{
  const key=this.key(members),cached=this.entries.get(key);if(cached)return cached;
  const result=compileCoherentParent(members);
  if(this.entries.size>=CoherentParentCache.MAX_ENTRIES)this.entries.delete(this.entries.keys().next().value!);
  this.entries.set(key,result);return result;
 }
}

/** Machine-readable private manifest. These classes stay complete children. */
export const COHERENT_PARENT_UNSUPPORTED_MANIFEST={
 curvedOrNonCoplanarNearCoincidentBoundaries:'endpoint equality is not a shared-join certificate',
 multiPlaneJunctionsAndTJunctions:'no junction ownership or split correspondence rule',
 overlappingNonCoincidentSpans:'requires new source samples or tolerance-dependent welding',
 incompleteParentProducts:'missing geometry, pose, contributor, relation, native context, or correspondence',
 productionOctreeAndGpuWiring:'source-lod and far packing call this compiler; native host status fallback is owned by the articulation seam',
} as const;

const coherentProducts=new WeakMap<SourceWholeConstruction,CoherentParentProduct>();
export function coherentParentProductOf(value:SourceWholeConstruction|undefined):CoherentParentProduct|undefined{return value?coherentProducts.get(value):undefined;}
/**
 * A parent product can be rejected after its GPU producer has inspected the
 * real packet.  The rejection belongs to the generation-owned product, not
 * to one resident construction object: remapping a source owner creates a new
 * immutable construction, but aliasCoherentParentProduct deliberately points
 * it at this same product.  Keeping the mark on the product therefore makes
 * every resident alias take the same atomic child fallback.
 */
const coherentRejectedProducts=new WeakSet<CoherentParentProduct>();
const coherentTargetProduct=(value:SourceWholeConstruction|CoherentParentProduct|undefined):CoherentParentProduct|undefined=>{
 if(!value)return;
 const candidate=value as Partial<CoherentParentProduct>;
 if(candidate.construction&&Array.isArray(candidate.passages)&&Array.isArray(candidate.sourceFeatures))return value as CoherentParentProduct;
 return coherentProducts.get(value as SourceWholeConstruction);
};
export function markCoherentParentRejected(value:SourceWholeConstruction|CoherentParentProduct|undefined):boolean{
 const product=coherentTargetProduct(value);if(!product)return false;
 coherentRejectedProducts.add(product);return true;
}
export function isCoherentParentRejected(value:SourceWholeConstruction|CoherentParentProduct|undefined):boolean{
 const product=coherentTargetProduct(value);return !!product&&coherentRejectedProducts.has(product);
}
/** The field adapter remaps source owners into resident-domain owners by
 * returning a new immutable construction object. Preserve the compiled
 * product across that adapter seam; correspondence generation will remap its
 * contributor pairs against the aliased resident construction. */
export function aliasCoherentParentProduct(source:SourceWholeConstruction|undefined,resident:SourceWholeConstruction|undefined):void{
 const product=source?coherentProducts.get(source):undefined;if(product&&resident)coherentProducts.set(resident,product);
}

const IDENTITY:readonly number[]=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
const EPS=1e-5;
const pairKey=(pair:readonly [number,number]):string=>`${pair[0]>>>0}:${pair[1]>>>0}`;
const pointKey=(point:SourceWholePoint):string=>point.map(value=>Math.round(value/EPS)).join(',');
const sortedSegmentKey=(a:SourceWholePoint,b:SourceWholePoint,entryRef:number,context:number):string=>{
 const left=pointKey(a),right=pointKey(b);
 return `${left<right?left:right}|${left<right?right:left}|${entryRef}|${context}`;
};
const finitePose=(pose:ArrayLike<number>|undefined):pose is ArrayLike<number>=>!!pose&&pose.length===16&&Array.from(pose).every(value=>Number.isFinite(value));
const posed=(point:SourceWholePoint,pose:ArrayLike<number>):SourceWholePoint=>[
 pose[0]!*point[0]!+pose[4]!*point[1]!+pose[8]!*point[2]!+pose[12]!,
 pose[1]!*point[0]!+pose[5]!*point[1]!+pose[9]!*point[2]!+pose[13]!,
 pose[2]!*point[0]!+pose[6]!*point[1]!+pose[10]!*point[2]!+pose[14]!,
];
const distance=(a:SourceWholePoint,b:SourceWholePoint):number=>Math.hypot(a[0]!-b[0]!,a[1]!-b[1]!,a[2]!-b[2]!);
const uniquePairs=(pairs:readonly (readonly [number,number])[]):readonly (readonly [number,number])[]=>{
 const seen=new Set<string>(),out:(readonly [number,number])[]=[];
 for(const pair of pairs){const key=pairKey(pair);if(!seen.has(key)){seen.add(key);out.push(pair);}}
 return out;
};
/** The current resident correspondence ABI carries one primary and one
 * optional secondary owner. Groups needing more owners remain complete
 * children; silently collapsing a duplicate or truncating a third owner would
 * publish a parent whose native visibility cannot be exact. */
const exactContributorSet=(pairs:readonly (readonly [number,number])[],known:ReadonlySet<string>):readonly (readonly [number,number])[]|undefined=>{
 if(!pairs.length||pairs.length>2)return;
 const seen=new Set<string>();
 for(const pair of pairs){const key=pairKey(pair);if(seen.has(key)||!known.has(key))return;seen.add(key);}
 return pairs;
};
const mix=(hash:number,word:number):number=>Math.imul((hash^(word>>>0))>>>0,16777619)>>>0;
const featureHash=(features:readonly number[],revision:number):number=>{
 let hash=2166136261>>>0;hash=mix(hash,revision);for(const feature of features)hash=mix(hash,feature);return hash||1;
};
const canonicalFeatureAliases=(values:readonly {readonly retained:number;readonly aliases:readonly number[]}[]):{readonly retained:number;readonly aliases:readonly number[]}[]=>{
 const groups:number[][]=[];
 for(const value of values){
  const members=[value.retained,...value.aliases].sort((left,right)=>left-right),matches:number[]=[];
  for(let index=0;index<groups.length;index++)if(members.some(member=>groups[index]!.includes(member)))matches.push(index);
  if(!matches.length){groups.push([...new Set(members)]);continue;}
  const merged=[...members];for(const index of matches)merged.push(...groups[index]!);
  const keep=matches[0]!;groups[keep]=[...new Set(merged)].sort((left,right)=>left-right);
  for(let index=matches.length-1;index>0;index--)groups.splice(matches[index]!,1);
 }
 return groups.filter(group=>group.length>1).map(group=>({retained:group[0]!,aliases:group.slice(1)}));
};

interface SegmentRecord {
 readonly first:SourceWholePoint;
 readonly last:SourceWholePoint;
 readonly feature:number;
 readonly role:SourceWholePartRole;
 readonly entryRef:number;
 readonly context:number;
 readonly ports:readonly [number,number];
 readonly contributors:readonly (readonly [number,number])[];
 readonly sourceFeatures:readonly number[];
 readonly removedSharedJoin:boolean;
 readonly plane?:Plane;
 readonly patch?:ClosedPatch;
 readonly references:PartSegment[];
 readonly shared:boolean;
}

interface PartSegment {
 readonly key:string;
 readonly first:SourceWholePoint;
 readonly last:SourceWholePoint;
 keep:boolean;
}

interface PartWork {
 readonly part:SourceWholePartRef;
 readonly member:CoherentParentMember;
 readonly pose:ArrayLike<number>;
 readonly segments:readonly PartSegment[];
 readonly plane?:Plane;
}

interface Plane {
 readonly origin:SourceWholePoint;
 readonly normal:SourceWholePoint;
}

interface ClosedPatch {
 readonly points:readonly SourceWholePoint[];
 readonly plane:Plane;
}

const vector=(a:SourceWholePoint,b:SourceWholePoint):SourceWholePoint=>[a[0]!-b[0]!,a[1]!-b[1]!,a[2]!-b[2]!];
const cross=(a:SourceWholePoint,b:SourceWholePoint):SourceWholePoint=>[a[1]!*b[2]!-a[2]!*b[1]!,a[2]!*b[0]!-a[0]!*b[2]!,a[0]!*b[1]!-a[1]!*b[0]!];
const dot=(a:SourceWholePoint,b:SourceWholePoint):number=>a[0]!*b[0]!+a[1]!*b[1]!+a[2]!*b[2]!;
const normalize=(value:SourceWholePoint):SourceWholePoint=>{const length=Math.hypot(value[0]!,value[1]!,value[2]!);return [value[0]!/length,value[1]!/length,value[2]!/length];};
const planeFromPoints=(points:readonly SourceWholePoint[]):Plane|undefined=>{
 const origin=points[0];if(!origin)return;
 let second:SourceWholePoint|undefined;
 for(let index=1;index<points.length;index++)if(distance(origin,points[index]!)>EPS){second=points[index]!;break;}
 if(!second)return;
 const base=vector(origin,second);
 for(let index=1;index<points.length;index++){
  const normal=cross(base,vector(origin,points[index]!));
  if(Math.hypot(normal[0]!,normal[1]!,normal[2]!)>EPS)return {origin,normal:normalize(normal)};
 }
 return;
};
const planeContains=(plane:Plane,point:SourceWholePoint):boolean=>dot(plane.normal,vector(plane.origin,point))===0;
const planesCompatible=(left:Plane|undefined,right:Plane|undefined):boolean=>{
 if(!left||!right)return false;
 const normalCross=cross(left.normal,right.normal);
 return normalCross[0]===0&&normalCross[1]===0&&normalCross[2]===0&&planeContains(left,right.origin)&&planeContains(right,left.origin);
};
const planeKey=(plane:Plane|undefined):string|undefined=>{
 if(!plane)return;
 let normal=plane.normal;
 for(const value of normal)if(Math.abs(value)>EPS){if(value<0)normal=[-normal[0]!, -normal[1]!, -normal[2]!];break;}
 return [...normal,dot(normal,plane.origin)].map(value=>Math.round(value/1e-4)).join(',');
};
const projectionAxis=(normal:SourceWholePoint):number=>{
 const values=[Math.abs(normal[0]!),Math.abs(normal[1]!),Math.abs(normal[2]!)];
 if(values[0]!>=values[1]!&&values[0]!>=values[2]!)return 0;
 if(values[1]!>=values[2]!)return 1;
 return 2;
};
const project=(point:SourceWholePoint,axis:number):readonly [number,number]=>axis===0?[point[1]!,point[2]!]:axis===1?[point[0]!,point[2]!]:[point[0]!,point[1]!];
const orient2=(a:readonly [number,number],b:readonly [number,number],c:readonly [number,number]):number=>(b[0]!-a[0]!)*(c[1]!-a[1]!)-(b[1]!-a[1]!)*(c[0]!-a[0]!);
const polygonArea=(points:readonly (readonly [number,number])[]):number=>points.reduce((sum,point,index)=>{const next=points[(index+1)%points.length]!;return sum+point[0]!*next[1]!-point[1]!*next[0]!;},0)*.5;
const same2=(a:readonly [number,number],b:readonly [number,number]):boolean=>Math.hypot(a[0]!-b[0]!,a[1]!-b[1]!)<=EPS;
const between2=(a:readonly [number,number],b:readonly [number,number],point:readonly [number,number]):boolean=>Math.abs(orient2(a,b,point))<=1e-4&&point[0]!>=Math.min(a[0]!,b[0]!)-EPS&&point[0]!<=Math.max(a[0]!,b[0]!)+EPS&&point[1]!>=Math.min(a[1]!,b[1]!)-EPS&&point[1]!<=Math.max(a[1]!,b[1]!)+EPS;
const segmentsIntersect=(a:readonly [number,number],b:readonly [number,number],c:readonly [number,number],d:readonly [number,number]):boolean=>{
 const ab=orient2(a,b,c),ab2=orient2(a,b,d),cd=orient2(c,d,a),cd2=orient2(c,d,b);
 if(((ab>1e-4&&ab2<-1e-4)||(ab<-1e-4&&ab2>1e-4))&&((cd>1e-4&&cd2<-1e-4)||(cd<-1e-4&&cd2>1e-4)))return true;
 return between2(a,b,c)||between2(a,b,d)||between2(c,d,a)||between2(c,d,b);
};
const closedPatchCertificate=(points:readonly SourceWholePoint[],closed:boolean,expectedPlane:Plane|undefined):ClosedPatch|undefined=>{
 if(!closed||points.length<3)return;
 const plane=planeFromPoints(points);if(!plane||expectedPlane&&!planesCompatible(plane,expectedPlane))return;
 const axis=projectionAxis(plane.normal),projected=points.map(point=>project(point,axis));
 if(Math.abs(polygonArea(projected))<=1e-4)return;
 for(let left=0;left<projected.length;left++)for(let right=left+1;right<projected.length;right++){
  if(right===left+1||(left===0&&right===projected.length-1))continue;
  if(segmentsIntersect(projected[left]!,projected[(left+1)%projected.length]!,projected[right]!,projected[(right+1)%projected.length]!))return;
 }
 return {points,plane};
};
const oppositeInteriorSides=(left:ClosedPatch|undefined,right:ClosedPatch|undefined,a:SourceWholePoint,b:SourceWholePoint):boolean=>{
 if(!left||!right||!planesCompatible(left.plane,right.plane))return false;
 const axis=projectionAxis(left.plane.normal),edgeA=project(a,axis),edgeB=project(b,axis),side=(patch:ClosedPatch):number|undefined=>{
  const points=patch.points.map(point=>project(point,axis)),area=polygonArea(points);if(Math.abs(area)<=1e-4)return;
  const sign=area>0?1:-1;
  for(let index=0;index<points.length;index++){
   const next=points[(index+1)%points.length]!;
   if(same2(points[index]!,edgeA)&&same2(next,edgeB))return sign;
   if(same2(points[index]!,edgeB)&&same2(next,edgeA))return -sign;
  }
  return;
 };
 const leftSide=side(left),rightSide=side(right);return leftSide!==undefined&&rightSide!==undefined&&leftSide!==rightSide;
};
const sharesOwner=(left:readonly (readonly [number,number])[],right:readonly (readonly [number,number])[]):boolean=>{
 const owners=new Set(left.map(pairKey));return right.some(pair=>owners.has(pairKey(pair)));
};

function relationWorld(value:SourceWholeOpening,member:CoherentParentMember):SourceWholeOpening|undefined{
 const pose=value.worldFromSource??member.construction.parts.find(part=>part.contributors.some(pair=>pair[0]===member.mesh&&pair[1]===member.eid))?.worldFromSource;
 if(!finitePose(pose)||!value.geometry||value.geometry.length<2||typeof value.closed!=='boolean'||value.entryRef===undefined||value.context===undefined||!value.role)return;
 return {...value,geometry:value.geometry.map(point=>posed(point,pose)),worldFromSource:IDENTITY,contributors:uniquePairs(value.contributors)};
}

function sourcePose(part:SourceWholePartRef,member:CoherentParentMember):ArrayLike<number>|undefined{return part.worldFromSource??member.construction.parts.find(candidate=>candidate.contributors.some(pair=>pair[0]===member.mesh&&pair[1]===member.eid))?.worldFromSource;}

/**
 * Compile one bounded coherent parent.  The returned parent is eligible only
 * when every source part and relation has posed geometry.  Children remain
 * the caller's fallback for any unsupported/partial source product.
 */
export function compileCoherentParent(members:readonly CoherentParentMember[]):CoherentParentResult{
 if(!members.length)return {status:'children',reason:'missing-geometry'};
 const known=new Set(members.map(member=>`${member.mesh}:${member.eid}`));
 if(known.size!==members.length)return {status:'children',reason:'invalid-contributor'};
 if(members.some(member=>!member.construction.parts.length))return {status:'children',reason:'missing-geometry'};
 const poseBindings:CoherentPoseBinding[]=[];
 const sourceFeatures:number[]=[];
 const featureSeen=new Set<number>();
 const segments=new Map<string,SegmentRecord>();
 const featureAliases:{retained:number;aliases:number[]}[]=[];
 const partWorks:PartWork[]=[];
 const partSegmentGroups:PartSegment[][]=[];
 let expandedPartCount=0,removedSharedJoins=0;
 // A complete single-instance principal is already one compiled drawing.
 // Passing its ordered 3D passages through does not require a planar union
 // certificate. Exact planes still gate ALL shared-edge removal/merging;
 // never round a swept/rounded source onto a plane to make it eligible.
 const retainedPrincipal=members.length===1&&members[0]!.construction.principalComplete===true;

 for(const [memberIndex,member] of members.entries()){
  for(const [partIndex,part] of member.construction.parts.entries()){
   expandedPartCount++;
   if(!part.geometry||part.geometry.length<2||typeof part.closed!=='boolean')return {status:'children',reason:'missing-geometry'};
   const pose=sourcePose(part,member);if(!finitePose(pose))return {status:'children',reason:'invalid-pose'};
   const partContributors=exactContributorSet(part.contributors,known);
   if(!partContributors)return {status:'children',reason:'invalid-contributor'};
   if(!featureSeen.has(part.feature)){featureSeen.add(part.feature);sourceFeatures.push(part.feature);}
   poseBindings.push({feature:part.feature,mesh:member.mesh,eid:member.eid,worldFromSource:Array.from(pose),relation:false});
   const end=part.closed?part.geometry.length:part.geometry.length-1;
   const posedGeometry=part.geometry.map(value=>posed(value,pose)),partPlane=planeFromPoints(posedGeometry),planar=partPlane===undefined||posedGeometry.every(point=>planeContains(partPlane,point));
   if(part.closed&&partPlane&&!planar&&!retainedPrincipal)return {status:'children',reason:'unsupported-plane'};
   const usablePlane=planar?partPlane:undefined,patch=planar?closedPatchCertificate(posedGeometry,part.closed,usablePlane):undefined;
   const partSegments:PartSegment[]=[];
   for(let index=0;index<end;index++){
    const first=posedGeometry[index]!,last=posedGeometry[(index+1)%part.geometry.length]!;if(distance(first,last)<EPS)return {status:'children',reason:'missing-geometry'};
    const baseKey=sortedSegmentKey(first,last,part.entryRef,part.context),initialKey=`${baseKey}|${planeKey(usablePlane)??`uncertified:${member.mesh>>>0}:${member.eid>>>0}:${partIndex}:${index}`}`;
    let key=initialKey,existing=segments.get(key);
    const canMerge=!!(existing&&planesCompatible(existing.plane,usablePlane)&&!sharesOwner(existing.contributors,partContributors)&&oppositeInteriorSides(existing.patch,patch,first,last));
    if(existing&&!canMerge){key=`${initialKey}|collision:${memberIndex}:${partIndex}:${index}`;while(segments.has(key))key+=':x';existing=segments.get(key);}
    if(existing&&planesCompatible(existing.plane,usablePlane)&&!sharesOwner(existing.contributors,partContributors)&&oppositeInteriorSides(existing.patch,patch,first,last)){
     const mergedContributors=exactContributorSet(uniquePairs([...existing.contributors,...partContributors]),known);
     if(!mergedContributors)return {status:'children',reason:'invalid-contributor'};
     const current={key,first,last,keep:true};
     segments.set(key,{...existing,contributors:mergedContributors,sourceFeatures:[...new Set([...existing.sourceFeatures,part.feature])],removedSharedJoin:true,references:[...existing.references,current],shared:true});
     partSegments.push(current);
    }else{
     const current={key,first,last,keep:true};
     segments.set(key,{first,last,feature:part.feature,role:part.role,entryRef:part.entryRef,context:part.context,ports:part.ports,contributors:partContributors,sourceFeatures:[part.feature],removedSharedJoin:false,plane:usablePlane,patch,references:[current],shared:false});
     partSegments.push(current);
    }
   }
   partWorks.push({part,member,pose,segments:partSegments,plane:usablePlane});
   partSegmentGroups.push(partSegments);
  }
 }
 const referenceGroup=new Map<PartSegment,PartSegment[]>();for(const group of partSegmentGroups)for(const reference of group)referenceGroup.set(reference,group);
 const sharedKeys=new Set([...segments.values()].filter(record=>record.shared).map(record=>record.references[0]!.key));
 for(const record of segments.values()){
  if(!record.shared||record.references.length!==2)continue;
  const groups=record.references.map(reference=>referenceGroup.get(reference));
  if(!groups[0]||!groups[1]||groups[0]===groups[1])continue;
  // A complete overlapping copy (all of both patches is shared) is not an
  // internal boundary. Keep both complete outlines rather than erasing them.
  if(!groups[0].some(reference=>!sharedKeys.has(reference.key))||!groups[1].some(reference=>!sharedKeys.has(reference.key)))continue;
  for(const reference of record.references)reference.keep=false;
  removedSharedJoins++;
  const aliases=featureAliases.find(value=>value.retained===record.feature);
  for(const feature of record.sourceFeatures)if(feature!==record.feature){if(aliases){if(!aliases.aliases.includes(feature))aliases.aliases.push(feature);}else featureAliases.push({retained:record.feature,aliases:[feature]});}
 }
 if(!segments.size)return {status:'children',reason:'missing-geometry'};

 const list=(kind:'openings'|'contacts'|'protectedVoids'|'paper'):SourceWholeOpening[]=>{
  const values:SourceWholeOpening[]=[];
  for(const member of members){
   const nested=kind==='paper'?member.construction.parts.flatMap(part=>(part.paper??[]).map(value=>({...value,worldFromSource:value.worldFromSource??part.worldFromSource}))):[];
   for(const value of [...(member.construction[kind]??[]),...nested]){
    const relationContributors=exactContributorSet(value.contributors,known);if(!relationContributors)return [];
    const mapped=relationWorld(value,member);if(!mapped)return [];
    values.push({...mapped,contributors:relationContributors});
    const pose=value.worldFromSource??member.construction.parts.find(part=>part.contributors.some(pair=>pair[0]===member.mesh&&pair[1]===member.eid))?.worldFromSource;
    if(!finitePose(pose)){return [];}poseBindings.push({feature:value.feature,mesh:member.mesh,eid:member.eid,worldFromSource:Array.from(pose),relation:true});
   }
  }
  return values;
 };
 const openings=list('openings'),contacts=list('contacts'),protectedVoids=list('protectedVoids'),paperSource=list('paper');
 if((members.some(member=>member.construction.openings.length>0)&&openings.length===0)||
    (members.some(member=>member.construction.contacts.length>0)&&contacts.length===0)||
    (members.some(member=>member.construction.protectedVoids.length>0)&&protectedVoids.length===0)||
    (members.some(member=>member.construction.paper?.length||member.construction.parts.some(part=>part.paper?.length))&&paperSource.length===0))return {status:'children',reason:'invalid-relation'};

 const passages:CoherentPassage[]=[];let passageContributorFailure=false;
 const addPassage=(work:PartWork,geometry:readonly SourceWholePoint[],refs:readonly PartSegment[],full:boolean)=>{
  const sourceFeatures=[...new Set(refs.flatMap(ref=>segments.get(ref.key)?.sourceFeatures??[]))];
  const contributors=exactContributorSet(uniquePairs([...work.part.contributors,...refs.flatMap(ref=>segments.get(ref.key)?.contributors??[])]),known);
  if(!contributors){passageContributorFailure=true;return;}
  passages.push({feature:work.part.feature,sourceFeatures:sourceFeatures.length?sourceFeatures:[work.part.feature],role:work.part.role,entryRef:work.part.entryRef,context:work.part.context,ports:work.part.ports,contributors,geometry,closed:full&&work.part.closed===true,removedSharedJoin:refs.some(ref=>!ref.keep)});
 };
 const addRuns=(work:PartWork)=>{
  const refs=work.segments;
  if(refs.every(ref=>ref.keep)){
   addPassage(work,work.part.geometry!.map(value=>posed(value,work.pose)),refs,true);return;
  }
  const runs:{refs:PartSegment[];points:SourceWholePoint[]}[]=[];
  const pushRun=(run:{refs:PartSegment[];points:SourceWholePoint[]})=>{if(run.refs.length)runs.push(run);};
  const ordered=work.part.closed?(()=>{const breakAt=refs.findIndex(ref=>!ref.keep);return [...refs.slice(breakAt+1),...refs.slice(0,breakAt+1)];})():[...refs];
  let current:{refs:PartSegment[];points:SourceWholePoint[]}={refs:[],points:[]};
  for(const ref of ordered){
   if(!ref.keep){pushRun(current);current={refs:[],points:[]};continue;}
   if(!current.refs.length){current.refs.push(ref);current.points.push(ref.first,ref.last);continue;}
   if(distance(current.points.at(-1)!,ref.first)>EPS){pushRun(current);current={refs:[ref],points:[ref.first,ref.last]};continue;}
   current.refs.push(ref);current.points.push(ref.last);
  }
  pushRun(current);for(const run of runs)addPassage(work,run.points,run.refs,false);
 };
 for(const work of partWorks)addRuns(work);

 // The resident segment map above is still the incumbent for unsupported
 // source geometry.  This second, source-owned product is deliberately
 // narrower: only closed parts with one identical native role/context/ports
 // contract and disjoint owner contributors may be replaced by an ordered
 // component union.  A missing or conflicting contract leaves every original
 // passage in place; no part donates its native entry to another part.
 const nativeSignature=(part:SourceWholePartRef):string=>`${part.role}|${part.entryRef}|${part.context}|${part.ports[0]}|${part.ports[1]}`;
 const samePoints=(left:readonly SourceWholePoint[],right:readonly SourceWholePoint[]):boolean=>left.length===right.length&&left.every((point,index)=>point.every((value,axis)=>value===right[index]![axis]));
 const unionGroups=new Map<string,PartWork[]>();
 for(const work of partWorks){
  if(work.part.closed!==true||!work.part.geometry||work.part.geometry.length<3)continue;
  if(!Number.isSafeInteger(work.part.entryRef)||!Number.isSafeInteger(work.part.context))continue;
  const group=unionGroups.get(nativeSignature(work.part))??[];group.push(work);unionGroups.set(nativeSignature(work.part),group);
 }
 type ComponentUnionPlan={readonly retained:number;readonly features:readonly number[];readonly geometry:readonly SourceWholePoint[];readonly template:SourceWholePartRef;readonly contributors:readonly (readonly [number,number])[]};
 const unionPlans:ComponentUnionPlan[]=[];
 for(const group of unionGroups.values()){
  if(group.length<2)continue;
  const ownerKeys=new Set<string>();let ownerConflict=false;
  for(const work of group){
   for(const pair of work.part.contributors){const key=pairKey(pair);if(ownerKeys.has(key)){ownerConflict=true;break;}ownerKeys.add(key);}
   if(ownerConflict)break;
  }
  // Same-owner repeated role parts are separate native members, even when
  // their source coordinates happen to coincide.
  if(ownerConflict)continue;
  const protectedByFeature=new Map<number,SourceComponentContour>();let protectedConflict=false;
  for(const work of group){
   const relationLists:[SourceComponentContour['kind'],readonly SourceWholeOpening[]][]=[
    ['hole',work.member.construction.openings],
    ['hole',work.member.construction.protectedVoids],
    ['paper',work.member.construction.paper??[]],
    ['paper',(work.part.paper??[]).map(value=>({...value,worldFromSource:value.worldFromSource??work.pose}))],
   ];
   for(const [kind,values] of relationLists)for(const value of values){
    const mapped=relationWorld(value,work.member);
    if(!mapped||mapped.closed!==true||!mapped.geometry||mapped.geometry.length<3){protectedConflict=true;break;}
    const contour:SourceComponentContour={feature:value.feature,contributors:[value.feature],kind,points:mapped.geometry};
    const existing=protectedByFeature.get(contour.feature);
    if(existing&&!samePoints(existing.points,contour.points)){protectedConflict=true;break;}
    // The same source aperture may be listed in both the opening/void and
    // paper ledgers.  Keep one protected contour for the union certificate;
    // the original relation ledgers remain intact on the resulting parent.
    if(existing&&existing.kind==='paper'&&contour.kind==='hole')protectedByFeature.set(contour.feature,contour);
    else if(!existing)protectedByFeature.set(contour.feature,contour);
   }
   if(protectedConflict)break;
  }
  if(protectedConflict)continue;
  // Source feature IDs are bank-local and may repeat for different instances.
  // Give the temporary union graph owner-scoped identities, then recover the
  // original feature IDs only after the topology is certified.  This avoids
  // aliasing two copies merely because their bank feature number is equal.
  const sortedGroup=[...group].sort((left,right)=>{
   const leftOwner=left.part.contributors.map(pairKey).sort().join('|'),rightOwner=right.part.contributors.map(pairKey).sort().join('|');
   return leftOwner.localeCompare(rightOwner)||left.part.feature-right.part.feature;
  });
  const reserved=new Set(protectedByFeature.keys()),temporaryToWork=new Map<number,PartWork>();
  let temporaryFeature=-1;const contours:SourceComponentContour[]=[];
  for(const work of sortedGroup){while(reserved.has(temporaryFeature))temporaryFeature--;reserved.add(temporaryFeature);temporaryToWork.set(temporaryFeature,work);contours.push({feature:temporaryFeature,contributors:[temporaryFeature],kind:'outer',points:work.part.geometry!.map(value=>posed(value,work.pose))});temporaryFeature--;}
  contours.push(...protectedByFeature.values());
  const union=unionSourceComponentContours(contours);if(union.status!=='union'||!union.featureAliases.length)continue;
  const groupPlans:ComponentUnionPlan[]=[];let invalidPlan=false;
  for(const alias of union.featureAliases){
   const temporaryFeatures=[alias.retained,...alias.aliases].sort((left,right)=>left-right);
   const sourceParts=temporaryFeatures.map(feature=>temporaryToWork.get(feature)).filter((work):work is PartWork=>!!work);
   if(sourceParts.length!==temporaryFeatures.length){invalidPlan=true;break;}
   const loop=union.loops.find(candidate=>candidate.kind==='outer'&&candidate.sourceFeatures.some(feature=>temporaryFeatures.includes(feature)));
   if(!loop||loop.points.length<3){invalidPlan=true;break;}
   const contributors=exactContributorSet(uniquePairs(sourceParts.flatMap(work=>work.part.contributors)).slice().sort((left,right)=>pairKey(left).localeCompare(pairKey(right))),known);
   if(!contributors){invalidPlan=true;break;}
   const sourceFeatures=[...new Set(sourceParts.map(work=>work.part.feature))].sort((left,right)=>left-right);
   const template=sourceParts.slice().sort((left,right)=>left.part.feature-right.part.feature||left.part.contributors.map(pairKey).join('|').localeCompare(right.part.contributors.map(pairKey).join('|')))[0]!.part;
   groupPlans.push({retained:sourceFeatures[0]!,features:sourceFeatures,geometry:loop.points,template,contributors});
  }
  if(!invalidPlan)unionPlans.push(...groupPlans);
 }
 if(unionPlans.length){
  // Bank-local features repeat across copies. Only a passage belonging to
  // THIS union's source owners/native contract is replaced; a disjoint copy
  // with the same feature number must retain its complete construction.
  for(let index=passages.length-1;index>=0;index--){
   const passage=passages[index]!;
   if(unionPlans.some(plan=>nativeSignature(passage)===nativeSignature(plan.template)&&
    passage.sourceFeatures.every(feature=>plan.features.includes(feature))&&
    passage.contributors.every(pair=>plan.contributors.some(owner=>owner[0]===pair[0]&&owner[1]===pair[1]))))passages.splice(index,1);
  }
  for(const plan of unionPlans){
   passages.push({feature:plan.retained,sourceFeatures:plan.features,role:plan.template.role,entryRef:plan.template.entryRef,context:plan.template.context,ports:plan.template.ports,contributors:plan.contributors,geometry:plan.geometry,closed:true,removedSharedJoin:true});
   const existing=featureAliases.find(value=>value.retained===plan.retained);
   const missingAliases=plan.features.filter(feature=>feature!==plan.retained&&!(existing?.aliases.includes(feature)??false));
   if(existing){for(const feature of missingAliases)existing.aliases.push(feature);}
   else featureAliases.push({retained:plan.retained,aliases:plan.features.filter(feature=>feature!==plan.retained)});
   removedSharedJoins+=missingAliases.length;
  }
 }
 const normalizedAliases=canonicalFeatureAliases(featureAliases);
 featureAliases.splice(0,featureAliases.length,...normalizedAliases.map(value=>({retained:value.retained,aliases:[...value.aliases]})));
 if(passageContributorFailure)return {status:'children',reason:'invalid-contributor'};
 const parts:SourceWholePartRef[]=passages.map(passage=>({feature:passage.feature,entryRef:passage.entryRef,context:passage.context,role:passage.role,ports:passage.ports,contributors:passage.contributors,geometry:passage.geometry,closed:passage.closed,worldFromSource:IDENTITY}));
 const allPaper=[...paperSource,...openings,...protectedVoids];
 // The compound ledger names emitted parts. Absorbed source features remain
 // in passage provenance/featureAliases, not as nonexistent emitted parts.
 const outerFeatures=[...new Set(parts.map(part=>part.feature))];
 const openingFeatures=[...new Set(openings.map(value=>value.feature))];
 const contactFeatures=[...new Set(contacts.map(value=>value.feature))];
 const anchors=[...new Set(contacts.flatMap(value=>value.ports))];
 const hasRelations=openingFeatures.length>0||contactFeatures.length>0||allPaper.length>0;
 const contactOnly=!openingFeatures.length&&contactFeatures.length>0&&!protectedVoids.length&&!paperSource.length;
 const compound=hasRelations&&outerFeatures.length&&contactFeatures.length&&anchors.length&&(openingFeatures.length>0||contactOnly)?{
  kind:(contactOnly?'planar-contact':'planar-openings-contact') as 'planar-contact'|'planar-openings-contact',outerFeatures,openingFeatures,contactFeatures,anchorVertices:anchors,
 }:undefined;
 if(hasRelations&&!compound)return {status:'children',reason:'invalid-relation'};
 const revision=featureHash(sourceFeatures,members.reduce((hash,member)=>mix(hash,member.construction.revision),2166136261));
 const construction:SourceWholeConstruction={revision,...(members.every(member=>member.construction.principalComplete===true)?{principalComplete:true as const}:{}),parts,openings,contacts,protectedVoids,...(allPaper.length?{paper:allPaper}:{}),...(compound?{compound}:{})};
 const endpoints=new Map<string,number>(),parent=Array.from({length:passages.length},(_,index)=>index);
 const root=(index:number):number=>{while(parent[index]!==index){parent[index]=parent[parent[index]!]!;index=parent[index]!;}return index;};
 const join=(a:number,b:number)=>{const x=root(a),y=root(b);if(x!==y)parent[Math.max(x,y)]=Math.min(x,y);};
 passages.forEach((passage,index)=>{const first=passage.geometry[0]!,last=passage.geometry[passage.geometry.length-1]!;for(const point of [first,last]){const key=pointKey(point),other=endpoints.get(key);if(other===undefined)endpoints.set(key,index);else join(index,other);}});
 const connectedComponents=new Set(passages.map((_,index)=>root(index))).size;
 const product={construction,sourceFeatures,featureAliases,poseBindings,passages,connectedComponents,removedSharedJoins,expandedPartCount};
 coherentProducts.set(construction,product);
 return {status:'parent',product};
}
