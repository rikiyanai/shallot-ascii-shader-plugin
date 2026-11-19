import type {CompositionSubject} from './composition.ts';
import {relationPoint,type PosedRelation} from './posed-graph.ts';

type Point=readonly [number,number,number];
const face=(region:string|undefined)=>region===undefined?undefined:Number(region.slice(region.lastIndexOf('/face-')+6));
const cross=(a:Point,b:Point,c:Point)=>(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);
function hull(points:Point[]):Point[]{
 points.sort((a,b)=>a[0]-b[0]||a[1]-b[1]);
 const half=(list:readonly Point[])=>{const kept:Point[]=[];for(const point of list){while(kept.length>=2&&cross(kept.at(-2)!,kept.at(-1)!,point)<=0)kept.pop();kept.push(point);}return kept;};
 return [...half(points).slice(0,-1),...half([...points].reverse()).slice(0,-1)];
}
function principal(points:Point[],unit:number,rowPitch:number):boolean{
 if(points.length<3)return false;
 const outline=hull(points);if(outline.length<3)return false;
 // Rest-plane membership alone is not enough: a material can bend it. Check
 // its actual post-material homogeneous projection's affine depth plane.
 const a=outline[0]!,b=outline[1]!,c=outline[2]!,det=cross(a,b,c);
 if(!det)return false;
 const ux=b[0]-a[0],uy=b[1]-a[1],vx=c[0]-a[0],vy=c[1]-a[1],zu=b[2]-a[2],zv=c[2]-a[2];
 if(points.some(q=>Math.abs(q[2]-a[2]-(zu*((q[0]-a[0])*vy-vx*(q[1]-a[1]))+zv*(ux*(q[1]-a[1])-(q[0]-a[0])*uy))/det)>1e-5))return false;
 let twiceArea=0,perimeter=0;
 for(let i=0;i<outline.length;i++){const p=outline[i]!,q=outline[(i+1)%outline.length]!;twiceArea+=p[0]*q[1]-q[0]*p[1];perimeter+=Math.hypot(p[0]-q[0],p[1]-q[1]);}
 const area=Math.abs(twiceArea)/2;
 // Text-scale read hierarchy, not a glyph/angle fit: a long sub-unit facet
 // remains part of its enclosing form, rather than becoming another wire.
 return area>=unit*rowPitch&&2*area/perimeter>=unit/2;
}

/** Principal flat parts inside ANY source mesh. Source incidence partitions
 * disconnected coplanar patches; current posed depth verifies they stayed
 * flat. Only turns shared by two principal parts become interior features.
 * Nothing here chooses text, inspects an image or keys on an asset/scene. */
export function planeParts(subject:CompositionSubject,unit:number,rowPitch:number):{angular:ReadonlySet<string>;interiors:readonly PosedRelation[]}{
 const byFace=new Map<number,number>();for(const region of subject.planeRegions??[])for(const f of region.faces)byFace.set(f,region.id);
 const regions=new Map<number,PosedRelation[]>();
 for(const relation of subject.relations){
  if(relation.kind==='smooth-contour')continue;
  for(const id of new Set(relation.regions.flatMap(region=>{const f=face(region),plane=f===undefined?undefined:byFace.get(f);return plane===undefined?[]:[plane];}))){const edges=regions.get(id)??[];edges.push(relation);regions.set(id,edges);}
 }
 const membership=new Map<string,Set<number>>();
 for(const [region,edges]of regions){
  const byPort=new Map<string,number[]>();for(const [i,edge]of edges.entries())for(const port of edge.ports){const incident=byPort.get(port)??[];incident.push(i);byPort.set(port,incident);}
  const seen=new Set<number>();
  for(let first=0;first<edges.length;first++){
   if(seen.has(first))continue;const pending=[first],members:PosedRelation[]=[],vertices=new Map<string,Point>();seen.add(first);
   while(pending.length){const at=pending.pop()!,edge=edges[at]!;members.push(edge);for(const [end,port]of edge.ports.entries()){
    const point=relationPoint(edge,end);if(point.every(Number.isFinite))vertices.set(port,point);
    for(const next of byPort.get(port)??[])if(!seen.has(next)){seen.add(next);pending.push(next);}
   }}
   if(!principal([...vertices.values()],unit,rowPitch))continue;
   for(const member of members){const owners=membership.get(member.id)??new Set<number>();owners.add(region);membership.set(member.id,owners);}
  }
 }
 return {angular:new Set(membership.keys()),interiors:subject.relations.filter(relation=>relation.kind==='crease'&&(membership.get(relation.id)?.size??0)>=2)};
}
