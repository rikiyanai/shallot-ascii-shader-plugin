import {relationPoint,type PosedGraph,type PosedRelation,type FeatureInterval} from './posed-graph.ts';
/** A path is a selected source relationship. Its source incidence survives
 * visibility changes; visible intervals do not become independent paths. */
export interface FeatureWalk {
  readonly id:string;readonly owner:string;readonly closed:boolean;
  readonly steps:readonly {readonly relation:PosedRelation;readonly reverse:boolean;readonly intervals:readonly FeatureInterval[]}[];
  readonly ports:readonly [string,string];
}
export interface FeatureJunction {readonly id:string;readonly ports:readonly string[];readonly walks:readonly string[]}
export interface FeatureGraph {readonly walks:readonly FeatureWalk[];readonly junctions:readonly FeatureJunction[];readonly regions:ReadonlyMap<string,readonly string[]>}

/** Boundary of an oriented current-view surface. At an actual multiway
 * source port, use its rotation system with surface ink on the left, rather
 * than cutting every contour into independent strokes. Angle orders SOURCE
 * halfedges only; it never chooses a glyph or manufactures a screen contact.
 */
export function selectContourGraph(graph:PosedGraph,interiorSide:(relation:PosedRelation)=>number):FeatureGraph{
 const edges=graph.relations.map(relation=>({relation,reverse:interiorSide(relation)<0})),incoming=new Map<string,number[]>(),outgoing=new Map<string,number[]>(),regions=new Map<string,string[]>();
 for(const [i,edge]of edges.entries()){
  const start=edge.relation.ports[Number(edge.reverse)],end=edge.relation.ports[Number(!edge.reverse)];
  const out=outgoing.get(start)??[];out.push(i);outgoing.set(start,out);
  const into=incoming.get(end)??[];into.push(i);incoming.set(end,into);
  for(const region of edge.relation.regions)if(region){const list=regions.get(region)??[];list.push(edge.relation.id);regions.set(region,list);}
 }
 const next=new Map<number,number>(),ports=new Set([...incoming.keys(),...outgoing.keys()]);
 for(const port of ports){
  const into=incoming.get(port)??[],out=outgoing.get(port)??[];
  if(into.length!==out.length)continue;
  if(into.length===1){next.set(into[0]!,out[0]!);continue;}
  const ray=(index:number,entering:boolean)=>{
   const edge=edges[index]!,from=relationPoint(edge.relation,edge.reverse?1:0),to=relationPoint(edge.relation,edge.reverse?0:1),a=entering?to:from,b=entering?from:to;
   return {index,entering,angle:Math.atan2(b[1]-a[1],b[0]-a[0])};
  };
  const rays=[...into.map(index=>ray(index,true)),...out.map(index=>ray(index,false))].sort((a,b)=>a.angle-b.angle||Number(a.entering)-Number(b.entering)||edges[a.index]!.relation.id.localeCompare(edges[b.index]!.relation.id));
  // Cyclic, noncrossing halfedge pairing. A source port's complete incident
  // set is considered together, not a nearest-piece continuation picked
  // independently for each line. Unbalanced source incidence stays explicit.
  let balance=0,minimum=0,start=0;
  for(let i=0;i<rays.length;i++){balance+=rays[i]!.entering?-1:1;if(balance<minimum){minimum=balance;start=i+1;}}
  const stack:number[]=[];
  for(let i=0;i<rays.length;i++){const item=rays[(start+i)%rays.length]!;if(!item.entering)stack.push(item.index);else{const match=stack.pop();if(match!==undefined)next.set(item.index,match);}}
 }
 // Continuation identity is not a physical vertex weld. Two contour fans
 // may pass through the same real vertex without becoming a four-arm pen
 // junction. Preserve both original relation IDs and their paired paths.
 for(const [from,to]of next){
  const a=edges[from]!,b=edges[to]!,end=Number(!a.reverse),start=Number(b.reverse),port=a.relation.ports[end];
  if((incoming.get(port)?.length??0)+(outgoing.get(port)?.length??0)<=2)continue;
  const label=`${port}/continuation-${[a.relation.id,b.relation.id].sort().join('+')}`;
  const ap:[string,string]=[...a.relation.ports],bp:[string,string]=from===to?ap:[...b.relation.ports];ap[end]=label;bp[start]=label;
  a.relation={...a.relation,ports:ap};b.relation={...b.relation,ports:bp};
 }
 const visited=new Set<number>(),walks:FeatureWalk[]=[],hasPredecessor=new Set(next.values());
 const order=[...edges.keys()].sort((a,b)=>Number(hasPredecessor.has(a))-Number(hasPredecessor.has(b))||a-b);
 for(const first of order){
  if(visited.has(first))continue;
  const steps:FeatureWalk['steps'][number][]=[],start=edges[first]!.relation.ports[Number(edges[first]!.reverse)];let at=first,end=start,closed=false;
  while(!visited.has(at)){
   const edge=edges[at]!,relation=edge.relation,reverse=edge.reverse;visited.add(at);
   steps.push({relation,reverse,intervals:reverse?[...relation.intervals].reverse().map(part=>({...part,lo:1-part.hi,hi:1-part.lo})):relation.intervals});
   end=relation.ports[Number(!reverse)];const following=next.get(at);if(following===first){closed=true;break;}if(following===undefined)break;at=following;
  }
  const seed=steps.reduce((a,b)=>a.relation.id<b.relation.id?a:b).relation;
  walks.push({id:`${seed.owner}/contour-${seed.id.slice(seed.owner.length+1)}`,owner:seed.owner,closed,steps,ports:[start,end]});
 }
 return {walks,regions,junctions:[...ports].filter(port=>(incoming.get(port)?.length??0)+(outgoing.get(port)?.length??0)!==2).map(port=>({id:port,ports:[port],walks:walks.filter(walk=>walk.ports.includes(port)).map(walk=>walk.id)}))};
}

/** Maximal walks are split at actual source junctions, not screen crossings.
 * Degree-two bends remain within their walk so articulation can choose caps,
 * shoulders and returns together. Source face-region incidences remain attached.
 * This selection does not claim that a triangulated face is a finished AA region. */
export function selectFeatureGraph(graph:PosedGraph):FeatureGraph{
  const incidence=new Map<string,number[]>(),regions=new Map<string,string[]>();
  graph.relations.forEach((relation,index)=>{
    for(const port of relation.ports){const list=incidence.get(port)??[];list.push(index);incidence.set(port,list);}
    for(const region of relation.regions)if(region){const list=regions.get(region)??[];list.push(relation.id);regions.set(region,list);}
  });
  const visited=new Set<number>(),walks:FeatureWalk[]=[];
  const walk=(start:string,first:number)=>{
    let port=start,at=first;const steps:FeatureWalk['steps'][number][]=[];let end=start;
    while(!visited.has(at)){
      const relation=graph.relations[at]!;const reverse=relation.ports[1]===port;visited.add(at);
      steps.push({relation,reverse,intervals:reverse?[...relation.intervals].reverse().map(part=>({...part,lo:1-part.hi,hi:1-part.lo})):relation.intervals});
      end=relation.ports[reverse?0:1];if(end===start||(incidence.get(end)?.length??0)!==2)break;
      const next=incidence.get(end)!.find(index=>!visited.has(index));if(next===undefined)break;port=end;at=next;
    }
    const seed=steps.reduce((a,b)=>a.relation.id<b.relation.id?a:b).relation;
    walks.push({id:`${seed.owner}/walk-${seed.id.slice(seed.owner.length+1)}`,owner:seed.owner,closed:end===start,steps,ports:[start,end]});
  };
  for(const [port,edges]of incidence)if(edges.length!==2)for(const edge of edges)if(!visited.has(edge))walk(port,edge);
  for(let edge=0;edge<graph.relations.length;edge++)if(!visited.has(edge))walk(graph.relations[edge]!.ports[0],edge);
  const junctions:FeatureJunction[]=[];
  for(const [port,edges]of incidence)if(edges.length!==2)junctions.push({id:port,ports:[port],walks:walks.filter(w=>w.ports.includes(port)).map(w=>w.id)});
  return {walks,junctions,regions};
}
