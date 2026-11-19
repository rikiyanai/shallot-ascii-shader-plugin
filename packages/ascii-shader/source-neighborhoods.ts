import type {PosedGraph,PosedRelation} from './posed-graph.ts';

export interface SourceNeighborhood {readonly id:string;readonly owner:string;readonly graph:PosedGraph}
interface Descriptor {id:string;relations:Set<string>;ports:Set<string>;components:Set<number>}
interface Owner {next:number;parts:Descriptor[]}
function postings<T>(parts:readonly Descriptor[],values:(part:Descriptor)=>ReadonlySet<T>):Map<T,number[]>{
 const index=new Map<T,number[]>();
 for(const [i,part]of parts.entries())for(const value of values(part)){const list=index.get(value)??[];list.push(i);index.set(value,list);}
 return index;
}
function uniqueParticipant<T>(values:ReadonlySet<T>,index:ReadonlyMap<T,readonly number[]>):number|undefined{
 let only:number|undefined;
 for(const value of values)for(const i of index.get(value)??[]){if(only===undefined)only=i;else if(only!==i)return;}
 return only;
}

/** Drawing neighborhoods are not mesh lifetimes. Source incidence separates
 * disconnected features, while immutable source ports/regions carry their
 * correspondence through posing and visibility changes. Screen coordinates,
 * cells and codepoints never name a neighborhood. */
export class SourceNeighborhoods {
 private readonly owners=new Map<string,Owner>();
 retire(owner:string):void{this.owners.delete(owner);}
 select(graphs:readonly PosedGraph[]):SourceNeighborhood[]{
  const result:SourceNeighborhood[]=[];
  for(const graph of graphs){
   // Preserve the original port order, but visit this complete inventory
   // only ONCE. A part then looks up its own actual incident ports directly;
   // copying/filtering the world map for every small part was quadratic.
   const portOrder=new Map([...graph.ports.keys()].map((id,i)=>[id,i]));
   const byOwner=new Map<string,PosedRelation[]>();
   for(const relation of graph.relations){const list=byOwner.get(relation.owner)??[];list.push(relation);byOwner.set(relation.owner,list);}
   for(const [owner,relations]of byOwner){
    const incidence=new Map<string,number[]>();
    relations.forEach((r,index)=>{for(const port of r.ports){const list=incidence.get(port)??[];list.push(index);incidence.set(port,list);}});
    const visited=new Set<number>(),parts:{relations:PosedRelation[];descriptor:Descriptor}[]=[];
    for(let first=0;first<relations.length;first++){
     if(visited.has(first))continue;const pending=[first],members:PosedRelation[]=[];visited.add(first);
     while(pending.length){const relation=relations[pending.pop()!]!;members.push(relation);
      for(const port of relation.ports)for(const next of incidence.get(port)??[])if(!visited.has(next)){visited.add(next);pending.push(next);}
     }
     parts.push({relations:members,descriptor:{id:'',relations:new Set(members.map(r=>r.id)),ports:new Set(members.flatMap(r=>r.ports)),components:new Set(members.flatMap(r=>r.components??[]))}});
    }
    const state=this.owners.get(owner)??{next:0,parts:[]};this.owners.set(owner,state);
    const pairs:{now:number;before:number;score:number}[]=[];
    const oldRelations=postings(state.parts,p=>p.relations),oldPorts=postings(state.parts,p=>p.ports),oldComponents=postings(state.parts,p=>p.components);
    const currentComponents=postings(parts.map(p=>p.descriptor),p=>p.components);
    for(const [now,part]of parts.entries()){
     const scores=new Map<number,number>();
     for(const relation of part.descriptor.relations)for(const before of oldRelations.get(relation)??[])scores.set(before,(scores.get(before)??0)+4);
     for(const port of part.descriptor.ports)for(const before of oldPorts.get(port)??[])scores.set(before,(scores.get(before)??0)+1);
     // A migrating contour may lose all old edge crossings. A UNIQUE source
     // component can still establish correspondence; ambiguous loops cannot
     // silently borrow another loop's identity merely by sharing a mesh.
     // Inverted incidence computes the SAME overlap counts, uniqueness and
     // tie order without comparing unrelated current/previous parts.
     const before=uniqueParticipant(part.descriptor.components,oldComponents);
     if(before!==undefined){
      if(uniqueParticipant(state.parts[before]!.components,currentComponents)===now)scores.set(before,(scores.get(before)??0)+1);
     }
     for(const [before,score]of scores)pairs.push({now,before,score});
    }
    pairs.sort((a,b)=>b.score-a.score||a.before-b.before||a.now-b.now);
    const used=new Set<number>();
    for(const pair of pairs)if(!parts[pair.now]!.descriptor.id&&!used.has(pair.before)){parts[pair.now]!.descriptor.id=state.parts[pair.before]!.id;used.add(pair.before);}
    for(const part of parts){
     if(!part.descriptor.id){const serial=state.next++;part.descriptor.id=serial?`${owner}/neighborhood-${serial}`:owner;}
     const ports=[...part.descriptor.ports].flatMap(id=>{const port=graph.ports.get(id);return port?[[id,port] as const]:[];})
      .sort((a,b)=>portOrder.get(a[0])!-portOrder.get(b[0])!);
     result.push({id:part.descriptor.id,owner,graph:{...graph,relations:part.relations,ports:new Map(ports)}});
    }
    const current=new Map(parts.map(p=>[p.descriptor.id,p.descriptor]));
    // Invisible is not retired. Keep absent source neighborhoods until the
    // owning geometry revision/entity actually ends.
    state.parts=[...current.values(),...state.parts.filter(old=>!current.has(old.id))];
   }
  }
  return result;
 }
}
