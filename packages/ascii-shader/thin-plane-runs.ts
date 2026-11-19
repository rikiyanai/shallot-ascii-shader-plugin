import type {SourcePlaneRegion} from './composition.ts';
import type {ArticulationRun} from './gpu-articulation.ts';
import {relationPoint,type PosedRelation} from './posed-graph.ts';

const planeFaces=new WeakMap<readonly SourcePlaneRegion[],ReadonlyMap<number,number>>();

/** A ribbon without room for separating paper is one construction, not two independently quantized
 * borders. Only a real four-sided source plane with no opening qualifies;
 * proximity alone never welds unrelated edges, owners or overlapping forms.
 * This is bounded source-topology LOD, not a glyph/pen/ink search. */
export function shareThinPlaneRuns(runs:readonly ArticulationRun[],relations:readonly PosedRelation[],planes:readonly SourcePlaneRegion[]):ArticulationRun[]{
 const owner=relations[0]?.owner;if(!owner||!planes.length)return [...runs];
 let facePlane=planeFaces.get(planes);if(!facePlane){const compiled=new Map<number,number>();for(const plane of planes)for(const face of plane.faces)compiled.set(face,plane.id);planeFaces.set(planes,compiled);facePlane=compiled;}
 const borders=new Map<number,Map<string,PosedRelation>>();
 for(const relation of relations){
  if(relation.kind!=='boundary'&&relation.kind!=='crease')continue;
  if(relation.a[3]<=.00001||relation.b[3]<=.00001)continue;
  const key=[...relation.ports].sort().join('>');
  for(const region of relation.regions){
   const match=region&&/\/face-(\d+)$/u.exec(region),plane=match?facePlane.get(Number(match[1])):undefined;if(plane===undefined)continue;
   const edges=borders.get(plane)??new Map<string,PosedRelation>(),prior=edges.get(key);
   if(!prior||relation.kind==='boundary'&&prior.kind!=='boundary')edges.set(key,relation);
   borders.set(plane,edges);
  }
 }
 const byRelation=new Map<string,number[]>();runs.forEach((run,index)=>{if(run.owner!==owner||!run.hard||run.anchorOnly)return;for(const source of run.sources){const list=byRelation.get(source)??[];list.push(index);byRelation.set(source,list);}});
 const parent=runs.map((_,index)=>index),root=(index:number):number=>{while(parent[index]!==index){parent[index]=parent[parent[index]!]!;index=parent[index]!;}return index;};
 const lengths=runs.map(run=>{const a=run.points[0],b=run.points.at(-1);return a&&b?Math.hypot(b.x-a.x,b.y-a.y):0;});
 const weight=[8,7,5,4,3];
 const preferred=(a:number,b:number)=>weight[runs[a]!.lineClass]!*lengths[a]!==weight[runs[b]!.lineClass]!*lengths[b]!
  ?weight[runs[a]!.lineClass]!*lengths[a]!>weight[runs[b]!.lineClass]!*lengths[b]!:runs[a]!.id<runs[b]!.id;
 for(const edges of borders.values()){
  if(edges.size!==4)continue;
  const border=[...edges.values()],degree=new Map<string,number>();for(const edge of border)for(const port of edge.ports)degree.set(port,(degree.get(port)??0)+1);
  if(degree.size!==4||[...degree.values()].some(n=>n!==2))continue;
  // Four real edges give exactly two opposite pairs. No all-pairs pass over
  // the world, no screen hash, and no assumption that a face is a subject.
  const first=border[0]!,opposite=border.find(edge=>!edge.ports.some(port=>first.ports.includes(port)));if(!opposite)continue;
  const others=border.filter(edge=>edge!==first&&edge!==opposite);
  for(const [a,b]of [[first,opposite],others] as const){
   if(!a||!b)continue;
   // A transparent foreground is not a change to this plane's topology or
   // native scale. Use the visible source intervals below, exactly as with
   // no glass in front. Opaque hidden spans do not enter that extent; intact
   // glyph visibility still belongs to articulation after sharing.
   // Resolve scale over the FRAMED source span. A near offscreen corner
   // must not prevent sharing the sub-row portion actually on the page.
   // The run itself keeps its complete endpoints and original phase.
   const av=a.intervals.filter(part=>part.cause==='visible'),bv=b.intervals.filter(part=>part.cause==='visible');if(!av.length||!bv.length)continue;
   const p=relationPoint(a,av[0]!.lo),q=relationPoint(a,av.at(-1)!.hi),r=relationPoint(b,bv[0]!.lo),s=relationPoint(b,bv.at(-1)!.hi);
   const dx=q[0]-p[0],dy=q[1]-p[1],ex=s[0]-r[0],ey=s[1]-r[1],length=Math.hypot(dx,dy),other=Math.hypot(ex,ey);
   if(length<51||other<51||Math.abs((dx*ex+dy*ey)/(length*other))<.999)continue;
   const distance=Math.max(Math.abs((r[0]-p[0])*dy-(r[1]-p[1])*dx),Math.abs((s[0]-p[0])*dy-(s[1]-p[1])*dx))/length;
   const cell=Math.hypot(dy/length*11,dx/length*17);
   // Two borders need an intervening paper unit to articulate a face.
   // Adjacent border rows/columns with no such channel read as a duplicate
   // stroke instead. Longitudinal extent and source incidence still gate it.
   if(distance>=cell*2||Math.min(length,other)<distance*4)continue;
   const start=((r[0]-p[0])*dx+(r[1]-p[1])*dy)/length,end=((s[0]-p[0])*dx+(s[1]-p[1])*dy)/length;
   if(Math.min(length,Math.max(start,end))-Math.max(0,Math.min(start,end))<Math.min(length,other)*.9)continue;
   const candidates=[...new Set([...byRelation.get(a.id)??[],...byRelation.get(b.id)??[]])];if(candidates.length<2)continue;
   let master=root(candidates[0]!);for(const index of candidates){const next=root(index);if(preferred(next,master))master=next;}
   const sources=new Set(border.map(edge=>edge.id));
   // The short connectors collapse as relationships into the shared unit.
   // A run extending beyond this face remains separate; never erase its
   // continuation, a neighboring opening, or another plane's construction.
   for(const edge of border)for(const index of byRelation.get(edge.id)??[])if(runs[index]!.sources.every(source=>sources.has(source))){const from=root(index);if(from!==master)parent[from]=master;}
  }
 }
 const sources=new Map<number,Set<string>>();runs.forEach((run,index)=>{const into=root(index),members=sources.get(into)??new Set<string>();for(const source of run.sources)members.add(source);sources.set(into,members);});
 return runs.flatMap((run,index)=>root(index)===index?[{...run,sources:[...sources.get(index)!]}]:[]);
}
