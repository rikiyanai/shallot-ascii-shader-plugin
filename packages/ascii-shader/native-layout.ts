import {nativeSpacePlan,type NativeWhitespacePolicy} from './text/native-whitespace.ts';
import {NativePage,NativeCommitError,type NativeGlyph,type NativeNeighborhood,type NativeOccurrence,type NativeFrame} from './native-page.ts';

type Point=readonly [number,number];
export interface NativeSurface {
 readonly id:string;readonly owner:string;
 readonly kind?:'surface'|'opening';
  /** A chart domain whose body coverage is the mapped source receiver,
   * never the rectangular outline itself. Existing occlusion samples must
   * verify owner/depth; native ink and protected paper still come from parts. */
  readonly coverage?:'source-receiver';
 /** The body constructed by the native passage, including its paper. */
 readonly outline:readonly Point[];
 /** Native-to-source surface charts. They sample current receiver ownership,
  * depth and opacity; no image or chart chooses a glyph. */
 readonly charts:readonly {readonly native:readonly [Point,Point,Point];readonly source:readonly [Point,Point,Point]}[];
}
export interface NativeContinuation {readonly from:string;readonly to:string;readonly fromAnchor:Point;readonly toAnchor:Point;readonly distance:number}
export interface NativePattern extends NativeNeighborhood {
  /** One source program's origin. Same-row members remain one scalar tape;
   * declared interfaces constrain its native row-paper parameters. Masks and
   * glyph order stay intact throughout admission and moving presentation. */
  readonly origin:readonly [number,number];
  /** Native cursor donors identified by actual source occlusion, rather
   * than a scalar depth assigned to an entire possibly intersecting object. */
  readonly after?:readonly string[];
  readonly surfaces?:readonly NativeSurface[];
  /** Actual source-part domains carried by a composite construction. These
   * bind its selected paths to their own receiver, not another copy of the
   * entire mesh body when independent complete parts need separate pens. */
  readonly sourceParts?:readonly {readonly paths:readonly string[];readonly surface:NativeSurface}[];
  readonly continuations?:readonly NativeContinuation[];
  /** Full selected feature paths. A shared mesh cannot donate another
   * disconnected feature's nearest edge or visibility interval. */
  readonly paths?:readonly {readonly id:string;readonly relations:readonly string[]}[];
  /** Per-ink geometric correspondence on an already selected construction.
   * Endpoints are ordered like the real source relation's ports. */
  readonly edgeCharts?:readonly {readonly source:string;readonly native:readonly [Point,Point]}[];
  /** Complete, phase-retained repeat runs. Visibility may omit units at a
   * cut; native paper can grow there by omitting further WHOLE end units.
   * Physical interfaces remain protected, and no interior unit is reseeded. */
  readonly repeatRuns?:readonly {readonly id:string;readonly members:readonly string[]}[];
}
export interface NativeLayoutResult {readonly frame:NativeFrame;readonly rejected:readonly {id:string;reason:string}[]}

export interface NativeCursorConstraint {readonly gap:number;readonly leading:boolean;readonly direction:1|-1}
const SLOT_RADIUS=16,SLOT_ALL=(1n<<33n)-1n;
// Finite native-whitespace language, compiled independently of any glyph,
// bitmap, scene or candidate placement. Runtime only intersects these masks.
const slotMasks=(policy:NativeWhitespacePolicy)=>Array.from({length:2},(_,leading)=>Array.from({length:2},(_,reverse)=>
 Array.from({length:160},(_,at)=>{
  const gap=at-SLOT_RADIUS;let mask=0n;
  for(let phase=-SLOT_RADIUS;phase<=SLOT_RADIUS;phase++)if(nativeSpacePlan(gap+(reverse?-phase:phase),0,!!leading,false,policy))mask|=1n<<BigInt(phase+SLOT_RADIUS);
  return mask;
 })));
const SLOT_MASKS={constrained:slotMasks('constrained'),preformatted:slotMasks('preformatted')};
/** Choose one shared native slot parameter for a COMPLETE function. This
 * never retries glyph candidates or moves individual members of the function. */
export function nativeProgramPhase(constraints:readonly NativeCursorConstraint[],previous?:number,policy:NativeWhitespacePolicy='constrained',legal?:(phase:number)=>boolean):number|undefined{
 let mask=SLOT_ALL;
 for(const constraint of constraints){
  if(constraint.gap>=144)continue;
  if(constraint.gap< -SLOT_RADIUS)return;
  mask&=SLOT_MASKS[policy][Number(constraint.leading)]![Number(constraint.direction===-1)]![constraint.gap+SLOT_RADIUS]!;
 }
 if(previous!==undefined&&Math.abs(previous)<=SLOT_RADIUS&&(mask&(1n<<BigInt(previous+SLOT_RADIUS)))&&(!legal||legal(previous)))return previous;
 for(let distance=0;distance<=SLOT_RADIUS;distance++){
  if(-distance!==previous&&(mask&(1n<<BigInt(SLOT_RADIUS-distance)))&&(!legal||legal(-distance)))return -distance;
  if(distance&&distance!==previous&&(mask&(1n<<BigInt(SLOT_RADIUS+distance)))&&(!legal||legal(distance)))return distance;
 }
}

// Every gap >=128 is legal under both actual native whitespace contracts.
// Below that finite threshold, layout intersects a precompiled language mask;
// it never searches glyphs, scores pixels or repairs independent row words.
const language=(policy:NativeWhitespacePolicy)=>Array.from({length:2},(_,leading)=>{
 let bits=0n;for(let gap=0;gap<128;gap++)if(nativeSpacePlan(gap,0,leading===1,false,policy))bits|=1n<<BigInt(gap);return bits;
});
const LANGUAGE={constrained:language('constrained'),preformatted:language('preformatted')};
const RADIUS=SLOT_RADIUS,WIDTH=RADIUS*2+1,ALL=(1n<<BigInt(WIDTH))-1n;
function phaseMask(gap:number,leading:boolean,reverse=false,policy:NativeWhitespacePolicy='constrained'):bigint{
 const start=gap-RADIUS;
 if(start>=128)return ALL;if(start+WIDTH<=0)return 0n;
 let bits=start<0?LANGUAGE[policy][Number(leading)]!<<BigInt(-start):LANGUAGE[policy][Number(leading)]!>>BigInt(start);
 // Positions beyond the finite language prefix are unconditionally legal.
 const high=Math.max(0,128-start);if(high<WIDTH)bits|=ALL&~((1n<<BigInt(high))-1n);
 bits&=ALL;if(!reverse)return bits;
 let reversed=0n;for(let i=0;i<WIDTH;i++)if(bits&(1n<<BigInt(i)))reversed|=1n<<BigInt(WIDTH-1-i);return reversed;
}

/** Native-language admission of WHOLE cooperating programs. Each actual
 * native contact group keeps one rigid phase. Occlusion dependencies remain
 * atomic without falsely attaching their geometry. The explicit text
 * policy controls only the serialization of actual 5px/11px spaces. */
export class NativeLayout {
 private readonly page:NativePage;private readonly font:Map<string,NativeGlyph>;
 private readonly phases=new Map<string,number>();
 private readonly whitespace:NativeWhitespacePolicy;
 private readonly margin:number;private readonly rowMargin:number;
 constructor(glyphs:readonly NativeGlyph[],whitespace:NativeWhitespacePolicy='constrained',viewportMargin=0){
  this.whitespace=whitespace;this.page=new NativePage(glyphs,whitespace);this.font=new Map(glyphs.map(g=>[g.text,g]));
  this.margin=viewportMargin;this.rowMargin=Math.ceil(viewportMargin/17);
 }
 private commit(groups:readonly NativeNeighborhood[],width:number,height:number,serial:number){
  const x=this.margin,y=this.rowMargin*17;
  const normalized=x||y?groups.map(group=>({...group,
   occurrences:group.occurrences.map(atom=>({...atom,x:atom.x+x,row:atom.row+this.rowMargin})),
   contacts:group.contacts.map(contact=>({...contact,fromAnchor:[contact.fromAnchor[0]+x,contact.fromAnchor[1]+y] as const,toAnchor:[contact.toAnchor[0]+x,contact.toAnchor[1]+y] as const})),
   paper:group.paper.map(paper=>({...paper,rect:[paper.rect[0]+x,paper.rect[1]+y,paper.rect[2],paper.rect[3]] as const}))})):groups;
  const frame=this.page.commit(normalized,width+2*x,height+2*y,serial);
  return x||y?Object.freeze({...frame,pageShift:[-x,-y] as const}):frame;
 }
 private admitPhases(patterns:readonly NativePattern[],placed:readonly NativeOccurrence[],width:number,height:number,live:Set<string>):Map<string,number>|undefined{
  const indices=new Map(patterns.map((p,i)=>[p.id,i])),parent=patterns.map((_,i)=>i);
  const root=(i:number):number=>parent[i]===i?i:parent[i]=root(parent[i]!);
  const union=(a:number,b:number)=>{parent[root(a)]=root(b);};
  for(const [i,p]of patterns.entries())for(const id of p.phaseWith??[]){const j=indices.get(id);if(j!==undefined)union(i,j);}
  const positions=new Map<string,{index:number;text:string}>();
  // Sharing an intact scalar fixes its donors' relative phase. This is a
  // drawing interface, distinct from a foreground body passing in front.
  for(const [i,p]of patterns.entries())for(const atom of p.occurrences){
   const key=`${atom.row+p.origin[1]}/${atom.x+p.origin[0]}`,other=positions.get(key);
   if(other&&other.text===atom.text)union(i,other.index);else positions.set(key,{index:i,text:atom.text});
  }
  const clusters=new Map<number,number>();for(let i=0;i<patterns.length;i++){const r=root(i);if(!clusters.has(r))clusters.set(r,clusters.size);}
  const count=clusters.size,fixed=count,domains:bigint[]=Array(count).fill(ALL);domains.push(1n<<BigInt(RADIUS));
  const cluster=(i:number)=>clusters.get(root(i))!;
  const keys=Array.from({length:count},(_,c)=>patterns.filter((_,i)=>cluster(i)===c).map(p=>p.id).sort().join('\n'));
  for(const key of keys)live.add(key);
  interface Member {x:number;advance:number;text:string;variable:number}
  const rows=new Map<number,Member[]>();
  const add=(atom:NativeOccurrence,variable:number)=>{
   const glyph=this.font.get(atom.text);if(!glyph||atom.row< -this.rowMargin||atom.row*17+17>height+this.rowMargin*17)return false;
   let mask=0n;for(let phase=-RADIUS;phase<=RADIUS;phase++)if(atom.x+phase>= -this.margin&&atom.x+phase+glyph.advance<=width+this.margin)mask|=1n<<BigInt(phase+RADIUS);
   domains[variable]=domains[variable]!&mask;
   const list=rows.get(atom.row)??[];list.push({x:atom.x,advance:glyph.advance,text:atom.text,variable});rows.set(atom.row,list);return true;
  };
  for(const [i,p]of patterns.entries())for(const atom of p.occurrences)if(!add({...atom,x:atom.x+p.origin[0],row:atom.row+p.origin[1]},cluster(i)))return;
  for(const atom of placed)if(!add(atom,fixed))return;
  interface Pair {a:number;b:number;allowed:bigint[]}
  const pairs=new Map<string,Pair>();
  const constrain=(left:Member,right:Member)=>{
   const gap=right.x-left.x-left.advance;
   if(left.variable===right.variable)return left.x===right.x&&left.text===right.text||!!nativeSpacePlan(gap,0,false,false,this.whitespace);
   if(gap>=128+2*RADIUS)return true;
   const a=Math.min(left.variable,right.variable),b=Math.max(left.variable,right.variable),key=`${a}/${b}`;
   const pair=pairs.get(key)??{a,b,allowed:Array(WIDTH).fill(ALL)};
   for(let ai=0;ai<WIDTH;ai++){
    let mask=0n;
    for(let bi=0;bi<WIDTH;bi++){
     const lp=(left.variable===a?ai:bi)-RADIUS,rp=(right.variable===a?ai:bi)-RADIUS;
     const shared=left.x===right.x&&left.text===right.text;
     if(shared?lp===rp:!!nativeSpacePlan(gap+rp-lp,0,false,false,this.whitespace))mask|=1n<<BigInt(bi);
    }
    pair.allowed[ai]=pair.allowed[ai]!&mask;
   }
   pairs.set(key,pair);return true;
  };
  for(const members of rows.values()){
   members.sort((a,b)=>a.x-b.x||a.variable-b.variable);
   const unique=members.filter((m,i)=>!i||m.x!==members[i-1]!.x||m.text!==members[i-1]!.text||m.variable!==members[i-1]!.variable);
   const first=unique[0]!;domains[first.variable]=domains[first.variable]!&phaseMask(first.x+this.margin,true,false,this.whitespace);
   for(let i=1;i<unique.length;i++)if(!constrain(unique[i-1]!,unique[i]!))return;
  }
  // Finite scalar-language propagation, with no glyph candidates, bitmap
  // objective, beam or backtracking. A whole program's offset can narrow;
  // none of its rows or members can move independently. An unresolved cyclic
  // language context remains rejected rather than triggering page fitting.
  const propagate=()=>{
   let changed=true;
   while(changed){changed=false;
    for(const {a,b,allowed}of pairs.values()){
     let da=0n,db=0n;
     for(let ai=0;ai<WIDTH;ai++)if(domains[a]!&(1n<<BigInt(ai))){const support=allowed[ai]!&domains[b]!;if(support){da|=1n<<BigInt(ai);db|=support;}}
     if(da!==domains[a]||db!==domains[b])changed=true;domains[a]=da;domains[b]=db;
     if(!da||!db)return false;
    }
   }
   return domains.every(mask=>!!mask);
  };
  if(!propagate())return;
  for(let c=0;c<count;c++){
   const previous=this.phases.get(keys[c]!);let chosen:number|undefined;
   if(previous!==undefined&&domains[c]!&(1n<<BigInt(previous+RADIUS)))chosen=previous;
   for(let distance=0;chosen===undefined&&distance<=RADIUS;distance++)for(const phase of distance?[-distance,distance]:[0])if(domains[c]!&(1n<<BigInt(phase+RADIUS))){chosen=phase;break;}
   if(chosen===undefined)return;
   domains[c]=1n<<BigInt(chosen+RADIUS);if(!propagate())return;
  }
  const phases=domains.slice(0,count).map(mask=>{for(let i=0;i<WIDTH;i++)if(mask&(1n<<BigInt(i)))return i-RADIUS;throw Error('Empty native phase');});
  for(const [i,key]of keys.entries())this.phases.set(key,phases[i]!);
  return new Map(patterns.map((p,i)=>[p.id,phases[cluster(i)]!]));
 }
 private admitCutRuns(patterns:readonly NativePattern[],placed:readonly NativeOccurrence[],width:number,height:number,key:string):{patterns:NativePattern[];phases:Map<string,number>}|undefined{
  const atoms=new Map(patterns.flatMap(pattern=>pattern.occurrences.map(atom=>[atom.id,atom] as const)));
  const protectedIDs=new Set(patterns.flatMap(pattern=>[...pattern.contacts.flatMap(contact=>[contact.from,contact.to]),...(pattern.continuations??[]).flatMap(link=>[link.from,link.to])]));
  interface Chunk {ids:string[];leftCut:boolean;rightCut:boolean}
  const membership=new Map<string,Chunk>();
  for(const pattern of patterns)for(const run of pattern.repeatRuns??[]){
   for(let at=0;at<run.members.length;){
    if(!atoms.has(run.members[at]!)){at++;continue;}
    const start=at;while(at<run.members.length&&atoms.has(run.members[at]!))at++;
    const chunk={ids:run.members.slice(start,at),leftCut:start>0,rightCut:at<run.members.length};
    for(const id of chunk.ids)membership.set(id,chunk);
   }
  }
  // A finite native writer state changes only the common program origin.
  // Within that state, terminal omission is monotone: it can enlarge paper
  // at an existing source cut, never remove an interior or named contact.
  const previous=this.phases.get(key),states=[...new Set([...(previous===undefined?[]:[previous]),0,...Array.from({length:RADIUS},(_,i)=>[-i-1,i+1]).flat()])];
  for(const phase of states){
   const removed=new Set<string>(),viewportRemoved=new Set<string>();
   const canRemove=(id:string,left:boolean)=>{
    const chunk=membership.get(id);if(!chunk||protectedIDs.has(id)||!(left?chunk.leftCut:chunk.rightCut))return false;
    const active=chunk.ids.filter(id=>!removed.has(id));return (left?active[0]:active.at(-1))===id;
   };
   const rows=new Map<number,{atom:NativeOccurrence;moving:boolean}[]>();
   for(const pattern of patterns)for(const atom of pattern.occurrences){
    const value={...atom,x:atom.x+pattern.origin[0]+phase,row:atom.row+pattern.origin[1]},list=rows.get(value.row)??[];
    const glyph=this.font.get(atom.text)!;
    if(value.row< -this.rowMargin||value.row*17+17>height+this.rowMargin*17||value.x< -this.margin||value.x+glyph.advance>width+this.margin){removed.add(atom.id);viewportRemoved.add(atom.id);continue;}
    list.push({atom:value,moving:true});rows.set(value.row,list);
   }
   for(const atom of placed){const list=rows.get(atom.row)??[];list.push({atom,moving:false});rows.set(atom.row,list);}
   let valid=true;
   for(const [row,list]of rows){
    list.sort((a,b)=>a.atom.x-b.atom.x||Number(a.moving)-Number(b.moving));
    if(row< -this.rowMargin||row*17+17>height+this.rowMargin*17){valid=false;break;}
    for(let i=0;i<list.length;){
     const current=list[i]!,atom=current.atom,glyph=this.font.get(atom.text)!;
     if(atom.x< -this.margin||atom.x+glyph.advance>width+this.margin){
      if(current.moving&&canRemove(atom.id,atom.x<0)){removed.add(atom.id);list.splice(i,1);continue;}
      valid=false;break;
     }
     const prior=list[i-1],shared=prior&&prior.atom.x===atom.x&&prior.atom.text===atom.text;
     const gap=prior?atom.x-prior.atom.x-this.font.get(prior.atom.text)!.advance:atom.x+this.margin;
     if(!shared&&!nativeSpacePlan(gap,0,!prior,false,this.whitespace)){
      if(prior?.moving&&canRemove(prior.atom.id,false)){removed.add(prior.atom.id);list.splice(i-1,1);i=Math.max(0,i-1);continue;}
      if(current.moving&&canRemove(atom.id,true)){removed.add(atom.id);list.splice(i,1);continue;}
      valid=false;break;
     }
     i++;
    }
    if(!valid)break;
   }
   if(!valid)continue;
   const adjusted=patterns.map(pattern=>{
    const beforeCut=new Set(pattern.occurrences.filter(atom=>!viewportRemoved.has(atom.id)).flatMap(atom=>atom.contributions.map(c=>c.source)));
    return {...pattern,occurrences:pattern.occurrences.filter(atom=>!removed.has(atom.id)),requiredSources:pattern.requiredSources.filter(source=>beforeCut.has(source)),
     contacts:pattern.contacts.filter(contact=>!viewportRemoved.has(contact.from)&&!viewportRemoved.has(contact.to)),
     continuations:pattern.continuations?.filter(link=>!viewportRemoved.has(link.from)&&!viewportRemoved.has(link.to))};
   });
   if(adjusted.some(pattern=>{const covered=new Set(pattern.occurrences.flatMap(atom=>atom.contributions.map(c=>c.source)));return !pattern.occurrences.length||pattern.requiredSources.some(source=>!covered.has(source));}))continue;
   this.phases.set(key,phase);return {patterns:adjusted,phases:new Map(patterns.map(pattern=>[pattern.id,phase]))};
  }
 }
 construct(patterns:readonly NativePattern[],width:number,height:number,serial:number):NativeLayoutResult{
  const accepted:NativeNeighborhood[]=[],placed:NativeOccurrence[]=[],rejected:{id:string;reason:string}[]=[];
  // Neighboring complete drawings share the SAME row-paper language even
  // when they have no physical contact. Allocate their whole origins together
  // before freezing a left neighbor's phase. This is the finite whitespace
  // propagation above: no glyph choices, raster objective, or backtracking.
  // A failed allocation falls through to directed admission below, so an
  // invalid receiver cannot remove otherwise valid foreground construction.
  if(patterns.length>1&&!patterns.some(pattern=>pattern.repeatRuns?.length)){
   const live=new Set<string>(),phases=this.admitPhases(patterns,[],width,height,live);
   if(phases){
    const groups=patterns.map(pattern=>this.place(pattern,phases.get(pattern.id)!));
    try{
     const frame=this.commit(groups,width,height,serial);
     for(const key of this.phases.keys())if(!live.has(key))this.phases.delete(key);
     return {frame,rejected};
    }catch(error){if(!(error instanceof NativeCommitError))throw error;}
   }
  }
  // Occlusion is DIRECTED: a receiver may need a foreground cursor donor,
  // but a failed receiver cannot withdraw that foreground construction.
  // Actual interfaces are bidirectional. Only a strongly connected set of
  // dependencies requires one atomic admission; this is not a page solver.
  const indices=new Map(patterns.map((pattern,index)=>[pattern.id,index])),edges=patterns.map(()=>new Set<number>());
  const atomOwners=new Map(patterns.flatMap((pattern,index)=>pattern.occurrences.map(atom=>[atom.id,index] as const)));
  const join=(a:number,b:number)=>{edges[a]!.add(b);edges[b]!.add(a);};
  for(const [index,pattern]of patterns.entries()){
   for(const id of pattern.after??[]){const other=indices.get(id);if(other!==undefined)edges[index]!.add(other);}
   for(const id of pattern.phaseWith??[]){const other=indices.get(id);if(other!==undefined)join(index,other);}
   for(const contact of pattern.contacts)for(const id of [contact.from,contact.to]){const other=atomOwners.get(id);if(other!==undefined&&other!==index)join(index,other);}
  }
  const components:NativePattern[][]=[],stack:number[]=[],active=new Set<number>(),number=new Map<number,number>(),low=new Map<number,number>();let next=0;
  const visit=(index:number)=>{
   number.set(index,next);low.set(index,next++);stack.push(index);active.add(index);
   for(const other of edges[index]!){
    if(!number.has(other)){visit(other);low.set(index,Math.min(low.get(index)!,low.get(other)!));}
    else if(active.has(other))low.set(index,Math.min(low.get(index)!,number.get(other)!));
   }
   if(low.get(index)===number.get(index)){
    const members:NativePattern[]=[];let member:number;
    do{member=stack.pop()!;active.delete(member);members.push(patterns[member]!);}while(member!==index);
    // Stable native program order within the actual dependency cycle.
    members.sort((a,b)=>indices.get(a.id)!-indices.get(b.id)!);components.push(members);
   }
  };
  patterns.forEach((_,index)=>{if(!number.has(index))visit(index);});
  const live=new Set<string>();
  for(const component of components){
   const key=component.map(pattern=>pattern.id).sort().join('\n');live.add(key);
   const reject=(reason:string)=>{for(const pattern of component)rejected.push({id:pattern.id,reason});};
   const cutRuns=component.some(pattern=>pattern.repeatRuns?.length)?this.admitCutRuns(component,placed,width,height,key):undefined;
   const phases=cutRuns?.phases??(component.some(pattern=>pattern.repeatRuns?.length)?undefined:this.admitPhases(component,placed,width,height,live));
   if(!phases){reject('cooperating source programs have no complete native cursor phase');continue;}
   const groups:NativeNeighborhood[]=(cutRuns?.patterns??component).map(pattern=>this.place(pattern,phases.get(pattern.id)!));
   try{this.commit([...accepted,...groups],width,height,serial);}
   catch(error){if(!(error instanceof NativeCommitError))throw error;reject(error.obligation);continue;}
   accepted.push(...groups);placed.push(...groups.flatMap(group=>group.occurrences));
  }
  for(const key of this.phases.keys())if(!live.has(key))this.phases.delete(key);
  return {frame:this.commit(accepted,width,height,serial),rejected};
 }
 private place(pattern:NativePattern,phase:number):NativeNeighborhood{
  const dx=pattern.origin[0]+phase,dy=pattern.origin[1]*17;
  return {...pattern,occurrences:pattern.occurrences.map(atom=>({...atom,x:atom.x+dx,row:atom.row+pattern.origin[1]})),
   contacts:pattern.contacts.map(contact=>({...contact,fromAnchor:[contact.fromAnchor[0]+dx,contact.fromAnchor[1]+dy] as const,toAnchor:[contact.toAnchor[0]+dx,contact.toAnchor[1]+dy] as const})),
   paper:pattern.paper.map(paper=>({...paper,rect:[paper.rect[0]+dx,paper.rect[1]+dy,paper.rect[2],paper.rect[3]] as const}))};
 }
}
