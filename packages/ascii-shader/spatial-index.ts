/** Incremental broad phase. Bounds choose work, never glyphs or visibility
 * acceptance. A caller's exact culler still decides individual instances. */
export type SpatialBounds=readonly [number,number,number,number,number,number];
type Entry<T>={bounds:SpatialBounds;value:T;node?:Node<T>};



type Node<T>={id:number;low:[number,number,number];size:number;parent?:Node<T>;children:(Node<T>|undefined)[];entries:Set<Entry<T>>;count:number;revision:number;box?:{revision:number;bounds:SpatialBounds};values?:{revision:number;list:T[]}};
function contains<T>(node:Node<T>,b:SpatialBounds):boolean{
 return b[0]>=node.low[0]&&b[1]>=node.low[1]&&b[2]>=node.low[2]&&
  b[3]<=node.low[0]+node.size&&b[4]<=node.low[1]+node.size&&b[5]<=node.low[2]+node.size;
}
function outside(b:SpatialBounds,planes:ArrayLike<number>,offset:number):boolean{
 for(let i=0;i<6;i++){
  const at=offset+i*4,nx=planes[at]!,ny=planes[at+1]!,nz=planes[at+2]!,d=planes[at+3]!;
  const x=nx*(nx>=0?b[3]:b[0]),y=ny*(ny>=0?b[4]:b[1]),z=nz*(nz>=0?b[5]:b[2]);
  // The original culler runs f32; broad-phase rejection is deliberately
  // looser than it, including at large world coordinates and plane tangency.
  if(x+y+z+d < -1e-5*(1+Math.abs(x)+Math.abs(y)+Math.abs(z)+Math.abs(d)))return true;
 }
 return false;
}
/** No world-distance cells or scene extent. The root grows by powers of two;
 * only the changed entry is reinserted. Crossing a split never loses a form:
 * an entry which spans children stays in their parent. */
export class SpatialIndex<K,T>{
 private root?:Node<T>;
 private readonly entries=new Map<K,Entry<T>>();
 private readonly unbounded=new Set<Entry<T>>();
 private readonly stack:Node<T>[]=[];

 private nodeIds=0;
 private node(low:[number,number,number],size:number,parent?:Node<T>):Node<T>{return {id:++this.nodeIds,low,size,parent,children:[],entries:new Set(),count:0,revision:0};}
 /** Reports changed membership, bounds or value identity. Unchanged notifications must not force
  * every camera to traverse the hierarchy again, including unbounded rows. */
 set(key:K,bounds:SpatialBounds|undefined,value:T):boolean{
  let entry=this.entries.get(key);
  const valid=!!bounds&&bounds.every(Number.isFinite)&&!bounds.some((n,i)=>i<3&&n>bounds![i+3]!);
  if(entry&&entry.value===value){
   if(entry.node&&valid&&entry.bounds.every((n,i)=>n===bounds![i]))return false;
   if(!valid&&this.unbounded.has(entry))return false;
  }
  if(entry)this.detach(entry);else{entry={bounds:[0,0,0,0,0,0],value};this.entries.set(key,entry);}
  entry.value=value;
  if(!valid){
   this.unbounded.add(entry);return true;
  }
  // valid establishes the finite ordered extent; keep this narrowing local.
  bounds=bounds!;
  entry.bounds=[...bounds];
  if(!this.root){
   const side=2**Math.ceil(Math.log2(Math.max(bounds[3]-bounds[0],bounds[4]-bounds[1],bounds[5]-bounds[2],1)));
   if(!Number.isFinite(side)){this.unbounded.add(entry);return true;}
   this.root=this.node([(bounds[0]+bounds[3]-side)/2,(bounds[1]+bounds[4]-side)/2,(bounds[2]+bounds[5]-side)/2],side);
  }
  while(!contains(this.root,bounds)){
   const old=this.root,size=old.size*2;
   if(!Number.isFinite(size)){this.unbounded.add(entry);return true;}
   const low=old.low.map((n,k)=>bounds[k]!<n?n-old.size:n) as [number,number,number];
   const root=this.node(low,size),slot=(old.low[0]>low[0]?1:0)|(old.low[1]>low[1]?2:0)|(old.low[2]>low[2]?4:0);
   root.children[slot]=old;old.parent=root;root.count=old.count;root.revision=old.revision+1;this.root=root;
  }
  let node=this.root,depth=0;
  while(depth++<32){
   const half=node.size/2;if(half===0||node.low.some(n=>n+half===n))break;
   const mid=node.low.map(n=>n+half);
   let slot=0,spans=false;
   for(let k=0;k<3;k++){if(bounds[k]!>=mid[k]!)slot|=1<<k;else if(bounds[k+3]!>mid[k]!)spans=true;}
   if(spans)break;
   let child=node.children[slot];
   if(!child){
    if(node.entries.size<8)break;
    const low=node.low.map((n,k)=>n+((slot&(1<<k))?half:0)) as [number,number,number];
    child=this.node(low,half,node);node.children[slot]=child;
   }
   node=child;
  }
  node.entries.add(entry);entry.node=node;
  for(let current:Node<T>|undefined=node;current;current=current.parent){current.count++;current.revision++;}
  return true;
 }
 private detach(entry:Entry<T>):void{
  this.unbounded.delete(entry);const node=entry.node;if(!node)return;
  node.entries.delete(entry);entry.node=undefined;
  for(let current:Node<T>|undefined=node;current;current=current.parent){current.count--;current.revision++;}
  let empty:Node<T>|undefined=node;
  while(empty?.parent&&!empty.count){const parent:Node<T>=empty.parent;parent.children[parent.children.indexOf(empty)]=undefined;empty=parent;}
  if(!this.root?.count)this.root=undefined;
 }
 delete(key:K):void{const entry=this.entries.get(key);if(!entry)return;this.detach(entry);this.entries.delete(key);}
 query(planes:ArrayLike<number>,visit:(value:T)=>void,offset=0):void{
  for(const entry of this.unbounded)visit(entry.value);
  const stack=this.stack;stack.length=0;if(this.root)stack.push(this.root);
  while(stack.length){
   const node=stack.pop()!,lo=node.low,size=node.size;
   if(!node.count||outside([lo[0],lo[1],lo[2],lo[0]+size,lo[1]+size,lo[2]+size],planes,offset))continue;
   for(const entry of node.entries)if(!outside(entry.bounds,planes,offset))visit(entry.value);
   for(const child of node.children)if(child?.count)stack.push(child);
  }
 }
 /** The subtree's tight aggregate bounds (entries and live children). */
 private tight(node:Node<T>):SpatialBounds{
  if(node.box?.revision===node.revision)return node.box.bounds;
  const b:[number,number,number,number,number,number]=[Infinity,Infinity,Infinity,-Infinity,-Infinity,-Infinity];
  const grow=(o:SpatialBounds)=>{for(let k=0;k<3;k++){b[k]=Math.min(b[k]!,o[k]!);b[k+3]=Math.max(b[k+3]!,o[k+3]!);}};
  for(const entry of node.entries)grow(entry.bounds);
  for(const child of node.children)if(child?.count)grow(this.tight(child));
  node.box={revision:node.revision,bounds:b};return b;
 }
 private members(node:Node<T>):readonly T[]{
  if(node.values?.revision===node.revision)return node.values.list;
  const list:T[]=[];for(const entry of node.entries)list.push(entry.value);
  for(const child of node.children)if(child?.count)list.push(...this.members(child));
  node.values={revision:node.revision,list};return list;
 }




 select(planes:ArrayLike<number>,adequate:(bounds:SpatialBounds)=>boolean,parent:(bounds:SpatialBounds,values:readonly T[],key:number)=>void,visit:(value:T)=>void,offset=0):void{
  for(const entry of this.unbounded)visit(entry.value);
  const stack:Node<T>[]=[];if(this.root)stack.push(this.root);
  while(stack.length){
   const node=stack.pop()!,lo=node.low,size=node.size;
   if(!node.count||outside([lo[0],lo[1],lo[2],lo[0]+size,lo[1]+size,lo[2]+size],planes,offset))continue;
   const bounds=this.tight(node);if(outside(bounds,planes,offset))continue;
   if(node.count>1&&adequate(bounds)){parent(bounds,this.members(node),node.id);continue;}
   for(const entry of node.entries)if(!outside(entry.bounds,planes,offset))visit(entry.value);
   for(const child of node.children)if(child?.count)stack.push(child);
  }
 }
 clear():void{this.root=undefined;this.entries.clear();this.unbounded.clear();this.stack.length=0;}
}
