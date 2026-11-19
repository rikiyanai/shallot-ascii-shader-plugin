import {relationPoint,type PosedGraph,type PosedRelation,type PosedOwner} from './posed-graph.ts';
import type {NativeFrame} from './native-construction.ts';
import type {NativeSupport} from './native-presentation.ts';

/** Evaluate every relation of each contributing drawing neighborhood, including hidden spans.
 * A bounding box or camera delta cannot certify a deforming native drawing.
 * These are geometry supports; the grammar still owes surface/field admission. */
export function constructionSupports(page:NativeFrame,graphs:readonly PosedGraph[]):ReadonlyMap<string,NativeSupport>{
 const owners=new Map<string,PosedOwner>(),relations=new Map<string,PosedRelation[]>();
 for(const graph of graphs){
  for(const [key,owner]of graph.owners??[])owners.set(key,owner);
  for(const relation of graph.relations){const list=relations.get(relation.id)??[];list.push(relation);relations.set(relation.id,list);}
 }
 // Invert native membership once, preserving the ORIGINAL page order in
 // every group. A scan/sort in group order would change the support origin
 // (native[0]) and therefore its configuration/correspondence contract.
 const nativeByGroup=new Map<string,NativeFrame['occurrences'][number][]>(),memberships=new Map<string,string[]>();
 for(const group of page.constructionGroups){
  if(page.directSupports?.get(group.id))continue;
  nativeByGroup.set(group.id,[]);
  for(const id of new Set(group.occurrences)){const list=memberships.get(id)??[];list.push(group.id);memberships.set(id,list);}
 }
 for(const atom of page.occurrences)for(const group of memberships.get(atom.id)??[])nativeByGroup.get(group)!.push(atom);
 // The same source relation often supports several cooperating runs. Its
 // hidden intervals remain dependencies, but project their exact samples
 // only once in THIS immutable source/view snapshot, never across frames.
 const evaluated=new Map<PosedRelation,{points:ReadonlyMap<string,readonly [number,number]>;depths:readonly (number|null)[];cameraPlane:readonly number[]}>();
 const evaluate=(relation:PosedRelation)=>{
  let support=evaluated.get(relation);if(support)return support;
  const samples=[...new Set([0,.5,1,...relation.intervals.flatMap(part=>[part.lo,part.hi])])].sort((a,b)=>a-b);
  const points=new Map<string,readonly [number,number]>(),depths:(number|null)[]=[],cameraPlane:number[]=[];
  for(const t of samples){
   const point=relationPoint(relation,t);
   if(!point.every(Number.isFinite)){
    const homogeneous=relation.a.map((value,axis)=>value*(1-t)+relation.b[axis]!*t);
    // A finite, nonzero projective point at W=0 is not a corrupt source.
    // It has no native-page coordinate, but remains a geometry dependency.
    // Preserve its parameter and the complete homogeneous endpoints in the
    // configuration below; never substitute an arbitrary finite XY point.
    if(relation.a.every(Number.isFinite)&&relation.b.every(Number.isFinite)&&homogeneous.every(Number.isFinite)&&homogeneous[3]===0&&homogeneous.slice(0,3).some(value=>value!==0)){
     cameraPlane.push(t);depths.push(null);continue;
    }
    throw Error(`Nonprojectable construction support ${relation.id}`);
   }
   points.set(`${relation.id}/${t}`,[point[0],point[1]]);depths.push(point[2]);
  }
  support={points,depths,cameraPlane};evaluated.set(relation,support);return support;
 };
 const supports=new Map<string,NativeSupport>();
 for(const group of page.constructionGroups){
  const direct=page.directSupports?.get(group.id);if(direct){supports.set(group.id,direct);continue;}
  const members=page.supportOwners.get(group.id);if(!members?.length)throw Error(`Construction ${group.id} has no source ownership`);
  const native=nativeByGroup.get(group.id)!;
  const points=new Map<string,readonly [number,number]>(),configuration:unknown[]=[native.map(atom=>[atom.id,atom.text,atom.x-(native[0]?.x??0),atom.row-(native[0]?.row??0),atom.contributions.map(c=>[c.source,c.kind,c.pixels,c.parameter,c.support?.id,c.support?.configuration])])],lifetimes:NativeSupport['owners'][number][]=[];
  configuration.push(group.paper.map(paper=>[paper.id,paper.rect[0]-(native[0]?.x??0),paper.rect[1]-(native[0]?.row??0)*17,paper.rect[2],paper.rect[3],paper.sources]));
  // Resolve only this group's declared relation IDs. A part's two edges
  // must not rescan the complete terrain/figure owning those edges.
  const selected=new Map<string,PosedRelation[]>();
  for(const id of new Set(page.supportRelations.get(group.id)))for(const relation of relations.get(id)??[]){const list=selected.get(relation.owner)??[];list.push(relation);selected.set(relation.owner,list);}
  for(const member of [...members].sort()){
   const owner=owners.get(member),list=selected.get(member);
   if(!owner||!list?.length)throw Error(`Construction ${group.id} lost its live source ${member}`);
   lifetimes.push({reference:owner.lifetime,source:owner.domain,kind:owner.kind});
   for(const relation of [...list].sort((a,b)=>a.id.localeCompare(b.id))){
    const support=evaluate(relation);for(const [key,point]of support.points)points.set(key,point);
    const relationConfiguration:unknown[]=[relation.id,relation.kind,relation.ports,relation.regions,relation.intervals,support.depths];
    if(support.cameraPlane.length)relationConfiguration.push({cameraPlane:support.cameraPlane,homogeneous:[relation.a,relation.b]});
    configuration.push(relationConfiguration);
   }
  }
  for(const atom of native)for(const contribution of atom.contributions)if(contribution.support){
   const support=contribution.support;points.set(support.id,support.point);configuration.push([support.id,support.configuration]);
  }
  supports.set(group.id,{configuration:JSON.stringify(configuration),points,owners:lifetimes});
 }
 return supports;
}
