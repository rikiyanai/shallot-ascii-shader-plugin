import {relationPoint,type PosedRelation} from './posed-graph.ts';

/** Source incidence establishes a single convex plane before any native
 * program is selected. Homogeneous projection preserves its affine depth
 * plane; a screen-shaped cycle alone cannot establish surface ownership. */
export function planeBoundary(relations:readonly PosedRelation[]):Map<string,readonly [number,number,number]>|undefined{
 const owner=relations[0]?.owner;if(!owner||relations.length<3||relations.some(r=>r.owner!==owner||r.kind!=='boundary'))return;
 const adjacent=new Map<string,string[]>(),points=new Map<string,readonly [number,number,number]>();
 for(const relation of relations)for(const [i,port]of relation.ports.entries()){
  const point=relationPoint(relation,i);if(!point.every(Number.isFinite))return;points.set(port,point);
  const list=adjacent.get(port)??[];list.push(relation.ports[1-i]!);adjacent.set(port,list);
 }
 if(points.size!==relations.length||[...adjacent.values()].some(list=>list.length!==2))return;
 const cycle:string[]=[],start=points.keys().next().value!;let at=start,previous='';
 do{if(cycle.includes(at))return;cycle.push(at);const next=adjacent.get(at)!.find(port=>port!==previous)!;previous=at;at=next;}while(at!==start);
 if(cycle.length!==points.size)return;
 const p=cycle.map(port=>points.get(port)!);let turn=0;
 for(let i=0;i<p.length;i++){
  const a=p[i]!,b=p[(i+1)%p.length]!,c=p[(i+2)%p.length]!,cross=(b[0]-a[0])*(c[1]-b[1])-(b[1]-a[1])*(c[0]-b[0]);
  if(Math.abs(cross)<1e-6||turn&&cross*turn<0)return;turn=cross;
 }
 const [a,b,c]=p as [typeof p[number],typeof p[number],typeof p[number]],ux=b[0]-a[0],uy=b[1]-a[1],vx=c[0]-a[0],vy=c[1]-a[1],det=ux*vy-vx*uy;
 const zu=b[2]-a[2],zv=c[2]-a[2];
 if(p.some(q=>Math.abs(q[2]-a[2]-(zu*((q[0]-a[0])*vy-vx*(q[1]-a[1]))+zv*(ux*(q[1]-a[1])-(q[0]-a[0])*uy))/det)>1e-5))return;
 return new Map(cycle.map(port=>[port,points.get(port)!]));
}
