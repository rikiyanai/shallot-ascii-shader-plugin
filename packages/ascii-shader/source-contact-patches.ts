import type {ContactPatchPoint} from './surface-relations.ts';

/** A resolved original-source corner, not a world/screen proximity key. */
export interface SourceContactCorner {
 readonly domain:number;readonly eid:number;readonly lifetime:number;readonly vertex:number;
}
export interface SourceContactLink {readonly a:SourceContactCorner;readonly b:SourceContactCorner}
export interface SourceContactPatches {
 readonly revision:number;readonly points:readonly ContactPatchPoint[];
 /** Unrepresented joins retain their separate incumbent edge nodes. */
 readonly capacityLimited:number;
}
const CAPACITY=1023;
const valid=(c:SourceContactCorner)=>[c.domain,c.eid,c.lifetime,c.vertex].every(v=>Number.isSafeInteger(v)&&v>=0)&&c.domain<0x20000000&&c.eid<=0xffffffff&&c.vertex<0x80000000;
const key=(c:SourceContactCorner)=>`${c.domain}/${c.eid}/${c.lifetime}/${c.vertex}`;

/** Joins only explicit two-owner solver incidence. Full corner tuples own
 * identity; a hash never establishes incidence. A corner touching several
 * bodies belongs to ONE connected component, not conflicting texture rows.
 * IDs never recycle during this renderer lifetime: CONTACT_DOMAIN keys can
 * survive in retained correspondence after a solver contact disappears.
 */
export class SourceContactPatchTable {
 private readonly ids=new Map<string,number>();
 private current:SourceContactPatches={revision:0,points:[],capacityLimited:0};
 private signature='';
 get value():SourceContactPatches{return this.current;}
 update(links:readonly SourceContactLink[]):SourceContactPatches{
  const corners=new Map<string,SourceContactCorner>(),parents=new Map<string,string>();
  const root=(k:string):string=>{let at=k;while(parents.get(at)!==at)at=parents.get(at)!;while(k!==at){const next=parents.get(k)!;parents.set(k,at);k=next;}return at;};
  for(const {a,b}of links){
   if(!valid(a)||!valid(b)||(a.domain===b.domain&&a.lifetime===b.lifetime))continue;
   const ka=key(a),kb=key(b);
   for(const [k,c]of [[ka,a],[kb,b]] as const)if(!parents.has(k)){parents.set(k,k);corners.set(k,c);}
   const ra=root(ka),rb=root(kb);if(ra!==rb)parents.set(ra<rb?rb:ra,ra<rb?ra:rb);
  }
  const groups=new Map<string,string[]>();
  for(const k of corners.keys()){const r=root(k),group=groups.get(r)??[];group.push(k);groups.set(r,group);}
  const components=[...groups.values()].filter(g=>g.length>1).map(g=>g.sort()).sort((a,b)=>a[0]!.localeCompare(b[0]!));
  const signature=JSON.stringify(components);
  if(signature===this.signature)return this.current;
  const points:ContactPatchPoint[]=[];let capacityLimited=0;
  for(const members of components){
   const identity=JSON.stringify(members);let id=this.ids.get(identity);
   if(id===undefined){if(this.ids.size===CAPACITY){capacityLimited+=members.length;continue;}id=this.ids.size;this.ids.set(identity,id);}
   for(const member of members){const {domain,eid,vertex}=corners.get(member)!;points.push({domain,eid,vertex,id});}
  }
  // Shader lookup cannot distinguish generations of an eid. The host supplies
  // only currently resolved owners; reject contradictory input conservatively.
  const rows=new Map<string,ContactPatchPoint|undefined>();
  for(const point of points){const k=`${point.domain}/${point.eid}/${point.vertex}`;if(rows.has(k)){if(rows.get(k)?.id!==point.id)rows.set(k,undefined);}else rows.set(k,point);}
  this.signature=signature;
  return this.current={revision:this.current.revision+1,points:[...rows.values()].filter((p):p is ContactPatchPoint=>p!==undefined),capacityLimited};
 }
}
