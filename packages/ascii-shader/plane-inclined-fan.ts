import sources from './plane-source-programs.json' with {type:'json'};
import {relationPoint,relationRange,type PosedRelation} from './posed-graph.ts';
import type {NativeGlyph,NativeOccurrence,NativeContact} from './native-page.ts';
import type {NativeContinuation,NativePattern} from './native-layout.ts';
import {nativeProgramPhase,type NativeCursorConstraint} from './native-layout.ts';
import type {FanResult,FanState,FanSlice} from './plane-cap-fan.ts';
import {nativeSpacePlan} from './text/native-whitespace.ts';
import {planeBoundary} from './plane-boundary.ts';

type Point=readonly [number,number];
type Vertex='apex'|'left'|'right'|'center'|'leftFoot'|'rightFoot'|'foot';
type Role='back'|'leftSlope'|'front'|'topSlope'|'backPost'|'leftPost'|'frontPost'|'sideReturn'|'base';
const roles:Record<Role,readonly [Vertex,Vertex]>={back:['apex','right'],leftSlope:['apex','left'],front:['left','center'],topSlope:['center','right'],
 backPost:['right','rightFoot'],leftPost:['left','leftFoot'],frontPost:['center','foot'],sideReturn:['foot','rightFoot'],base:['leftFoot','foot']};
interface Bound {owner:string;ports:Record<Vertex,string>;points:Map<Vertex,Point>;edges:Partial<Record<Role,PosedRelation>>;boundary?:boolean}
interface Template {pattern:Omit<NativePattern,'origin'>;vertices:Map<Vertex,Point>;admissible:boolean}

/** The inclined THREE-FACE source function. Its Y, shoulders and returns are
 * one neighborhood, not independently selected tangent glyphs. Entire source
 * words enter between its retained corners as the two inclined axes grow.
 * Intrinsic native tapes compile once per integer parameter tuple; current
 * source incidence, pose and visibility are bound separately each frame. */
export class PlaneInclinedFan {
 private readonly font:Map<string,NativeGlyph>;private readonly cache=new Map<string,Template>();
 constructor(font:readonly NativeGlyph[]){this.font=new Map(font.map(g=>[g.text,g]));}
 private bind(relations:readonly PosedRelation[],compact=false):Bound|undefined{
  if(relations.length===6)return this.bindBoundary(relations);
  if(relations.length!==9||relations.some(r=>r.kind==='smooth-contour'||r.owner!==relations[0]!.owner))return;
  const adjacency=new Map<string,string[]>(),points=new Map<string,Point>();
  for(const r of relations)for(const [i,p]of r.ports.entries()){
   const q=relationPoint(r,i);if(!q.every(Number.isFinite))return;points.set(p,[q[0],q[1]]);
   const list=adjacency.get(p)??[];list.push(r.ports[1-i]!);adjacency.set(p,list);
  }
  const centers=[...adjacency].filter(([p,list])=>list.length===3&&[...adjacency].every(([q,ns])=>q===p||ns.filter(n=>n!==p).length===2)).map(([p])=>p);
  if(points.size!==7||centers.length!==1)return;
  const center=centers[0]!,c=points.get(center)!,arms=adjacency.get(center)!;
  const above=arms.filter(p=>compact?points.get(p)![1]<=c[1]+.001:points.get(p)![1]<c[1]-8.5).sort((a,b)=>points.get(a)![0]-points.get(b)![0]),below=arms.filter(p=>points.get(p)![1]>c[1]+(compact?17:0));
  if(above.length!==2||below.length!==1||points.get(above[0]!)![0]>=c[0]||points.get(above[1]!)![0]<=c[0])return;
  const left=above[0]!,right=above[1]!,foot=below[0]!;
  const between=(a:string,b:string)=>adjacency.get(a)!.filter(p=>p!==center&&adjacency.get(b)!.includes(p));
  const apex=between(left,right),leftFoot=between(left,foot),rightFoot=between(right,foot);
  if(apex.length!==1||leftFoot.length!==1||rightFoot.length!==1)return;
  const ports:Record<Vertex,string>={center,left,right,foot,apex:apex[0]!,leftFoot:leftFoot[0]!,rightFoot:rightFoot[0]!};
  if(new Set(Object.values(ports)).size!==7)return;
  const edges={} as Record<Role,PosedRelation>;
  for(const [role,[a,b]]of Object.entries(roles) as [Role,readonly [Vertex,Vertex]][]){
   const edge=relations.find(r=>r.ports.includes(ports[a])&&r.ports.includes(ports[b]));if(!edge)return;edges[role]=edge;
  }
  return {owner:relations[0]!.owner,ports,points:new Map((Object.entries(ports) as [Vertex,string][]).map(([key,p])=>[key,points.get(p)!])),edges};
 }
 /** The same whole source perimeter can bound ONE plane without its three
  * internal face spokes. Only source-connected, coplanar convex six-port
  * boundaries qualify; projected overlap cannot invent a plane or a join. */
 private bindBoundary(relations:readonly PosedRelation[]):Bound|undefined{
  const owner=relations[0]?.owner,positions=planeBoundary(relations);if(!owner||!positions)return;
  const adjacent=new Map<string,string[]>();
  for(const relation of relations)for(const [i,port]of relation.ports.entries()){
   const list=adjacent.get(port)??[];list.push(relation.ports[1-i]!);adjacent.set(port,list);
  }
  if(positions.size!==6||[...adjacent.values()].some(list=>list.length!==2))return;
  const sorted=[...positions].sort((a,b)=>a[1][1]!-b[1][1]!),apex=sorted[0]![0],foot=sorted.at(-1)![0];
  if(apex===foot)return;
  const sides=(port:string)=>[...adjacent.get(port)!].sort((a,b)=>positions.get(a)![0]!-positions.get(b)![0]!);
  const [left,right]=sides(apex),[leftFoot,rightFoot]=sides(foot);
  const perimeter=[apex,right!,rightFoot!,foot,leftFoot!,left!];if(new Set(perimeter).size!==6)return;
  if(perimeter.some((port,i)=>!adjacent.get(port)!.includes(perimeter[(i+1)%6]!)))return;
  const ports:Record<Vertex,string>={apex,left:left!,right:right!,leftFoot:leftFoot!,rightFoot:rightFoot!,foot,center:`${owner}/plane-interior`};
  const edges:Partial<Record<Role,PosedRelation>>={};
  for(const role of ['back','leftSlope','backPost','leftPost','sideReturn','base'] as Role[]){
   const [a,b]=roles[role],edge=relations.find(r=>r.ports.includes(ports[a])&&r.ports.includes(ports[b]));if(!edge)return;edges[role]=edge;
  }
  return {owner,ports,edges,boundary:true,points:new Map((Object.entries(ports) as [Vertex,string][]).filter(([key])=>key!=='center').map(([key,port])=>[key,[positions.get(port)![0]!,positions.get(port)![1]!] as Point]))};
 }
 matches(relations:readonly PosedRelation[]):boolean{return this.bind(relations)!==undefined;}
 /** One complete native low-cap neighborhood. When the ENTIRE projected
  * cap is subrow, its two nearly coincident boundary chains share a rail;
  * they are not four independently placed short slashes. The three actual
  * straight vertical posts and the coupled return retain their own ports.
  * This provisional family does not claim an AAHub retained-entry identity.
  * Slanted posts and caps/returns deeper than one row remain outside its
  * domain; they must not acquire uprights by passing a loose fit threshold. */
 compact(relations:readonly PosedRelation[],neighborhood=relations[0]?.owner??''):FanResult{
  const owner=relations[0]?.owner??'',fail=(obligation:string):FanResult=>({ok:false,owner,obligation});
  const bound=this.bind(relations,true);if(!bound||bound.boundary)return fail('compact cap requires its complete incident solid');
  const p=bound.points,l=p.get('left')!,c=p.get('center')!,r=p.get('right')!,a=p.get('apex')!,lf=p.get('leftFoot')!,f=p.get('foot')!,rf=p.get('rightFoot')!;
  const sourcePoints=new Map(relations.flatMap(edge=>edge.ports.map((port,i)=>[port,relationPoint(edge,i)] as const)));
  for(const face of [['apex','right','center','left'],['left','center','foot','leftFoot'],['center','right','rightFoot','foot']] as Vertex[][]){
   const [a,b,c,d]=face.map(vertex=>sourcePoints.get(bound.ports[vertex])!);
   const u=b!.map((q,k)=>q-a![k]!),v=c!.map((q,k)=>q-a![k]!);
   const n=[u[1]!*v[2]!-u[2]!*v[1]!,u[2]!*v[0]!-u[0]!*v[2]!,u[0]!*v[1]!-u[1]!*v[0]!],length=Math.hypot(...n);
   if(length&&Math.abs(n.reduce((sum,q,k)=>sum+q*(d![k]!-a![k]!),0))/length>1e-5)return fail('compact cap cannot flatten a nonplanar source face');
  }
  const same=(x:number,y:number)=>Math.abs(x-y)<.001;
  const cap=[a,l,c,r],ys=cap.map(q=>q[1]);
  const feet=[lf,f,rf],footYs=feet.map(q=>q[1]);
  if(Math.max(...ys)-Math.min(...ys)>17||Math.max(...footYs)-Math.min(...footYs)>17||a[0]<l[0]||a[0]>r[0])return fail('cap or return needs its noncompact shared-axis family');
  const heights=[lf[1]-l[1],f[1]-c[1],rf[1]-r[1]];
  if(!same(l[0],lf[0])||!same(c[0],f[0])||!same(r[0],rf[0]))return fail('compact upright family cannot replace projected slants');
  if(this.font.get('|')?.advance!==4||this.font.get('_')?.advance!==5)return fail('compact cap needs its pinned native4/5 units');
  if(c[0]-l[0]<9||r[0]-c[0]<9||heights.some(h=>h<34))return fail('compact cap needs its complete smaller solid variant');
  // One common integer axis system supplies BOTH rails and every post. Each
  // added WHOLE5px horizontal unit changes the incident rails together. The
  // cap is one underline word, not three overhanging upright cap fragments.
  const leftUnits=Math.max(1,Math.round((c[0]-l[0]-4)/5)),rightUnits=Math.max(1,Math.round((r[0]-c[0]-4)/5));
  const columns=[0,4+5*leftUnits,8+5*(leftUnits+rightUnits)],height=Math.max(1,Math.round(heights.reduce((n,h)=>n+h,0)/3/17));
  const origin=Math.round((l[0]-1+c[0]-columns[1]!-1+r[0]-columns[2]!-1)/3),row=Math.round(((l[1]+c[1]+r[1])/3-15)/17);
  const id=`${neighborhood}/compact-plane-cap`,atoms:NativeOccurrence[]=[],contacts:NativeContact[]=[],continuations:NativeContinuation[]=[];
  const capXs:number[]=[];
  for(let k=0;k<leftUnits+rightUnits+2;k++)for(const q of this.font.get('_')!.points)capXs.push(1+5*k+q[0]!);
  // The source apex survives as a port on the coalesced rail. Locate it on
  // the ALREADY fixed native tape; this never elects a glyph or an angle.
  const wanted=1+(a[0]-l[0])/(r[0]-l[0])*(columns[2]!-columns[0]!);
  const apex=capXs.reduce((x,q)=>Math.abs(q-wanted)<Math.abs(x-wanted)?q:x);
  const native=new Map<Vertex,Point>([['left',[1,15]],['center',[columns[1]!+1,15]],['right',[columns[2]!+1,15]],['apex',[apex,15]],
   ['leftFoot',[1,height*17+15]],['foot',[columns[1]!+1,height*17+15]],['rightFoot',[columns[2]!+1,height*17+15]]]);
  const capRoles:Role[]=['leftSlope','back','front','topSlope'],returnRoles:Role[]=['base','sideReturn'];
  const add=(text:string,x:number,y:number,key:string,post?:Role)=>{
   const glyph=this.font.get(text)!;
   const assigned=new Map<Role,{pixels:number[];values:number[]}>();
   const assign=(role:Role,index:number,t:number)=>{const value=assigned.get(role)??{pixels:[],values:[]};value.pixels.push(index);value.values.push(t);assigned.set(role,value);};
   for(const [index,q]of glyph.points.entries()){
    const px=x+q[0]!,py=y*17+q[1]!,bar=y===0&&q[1]===15?capRoles:y===height&&q[1]===15?returnRoles:undefined;
    if(bar){
     for(const role of bar){const [u,v]=roles[role],from=native.get(u)![0],to=native.get(v)![0];
      if(px<Math.min(from,to)||px>Math.max(from,to))continue;
      assign(role,index,from===to?0:Math.max(0,Math.min(1,(px-from)/(to-from))));
     }
    }else if(post)assign(post,index,(py-15)/(height*17));
   }
   const contributions=[...assigned].map(([role,value])=>{
    const edge=bound.edges[role]!,[from,to]=roles[role],collapsed=native.get(from)![0]===native.get(to)![0]&&!role.endsWith('Post');
    const range:readonly [number,number]=collapsed?[0,1]:[Math.min(...value.values),Math.max(...value.values)];
    return {source:edge.id,kind:'path' as const,pixels:value.pixels,parameter:relationRange(edge,bound.ports[from],range)};
   });
   atoms.push({id:`${id}/${key}`,text,x,row:y,contributions});
  };
  for(const [i,role]of (['leftPost','frontPost','backPost'] as Role[]).entries())for(let y=1;y<=height;y++)add('|',columns[i]!,y,`post/${role}/${y}`,role);
  for(let k=0;k<leftUnits+rightUnits+2;k++)add('_',1+5*k,0,`cap/${k}`);
  for(const [side,start,count]of [['left',4,leftUnits],['right',columns[1]!+4,rightUnits]] as const)
   for(let k=0;k<count;k++)add('_',start+5*k,height,`return/${side}/${k}`);
  const covered=new Set(atoms.flatMap(atom=>atom.contributions.map(part=>part.source)));
  if(relations.some(relation=>!covered.has(relation.id)))return fail('compact neighborhood lost an incident source part');
  // Interface measurement annotates the completed word, never repairs it.
  const ink=(atom:NativeOccurrence,source:string)=>atom.contributions.filter(part=>part.source===source).flatMap(part=>part.pixels.map(index=>{
   const q=this.font.get(atom.text)!.points[index]!;return [atom.x+q[0]!+.5,atom.row*17+q[1]!+.5] as Point;
  }));
  for(const [vertex,point]of native){
   const incident=(Object.keys(roles) as Role[]).filter(role=>roles[role].includes(vertex));
   for(let i=1;i<incident.length;i++){
    const first=bound.edges[incident[0]!]!,second=bound.edges[incident[i]!]!;
    const at=(source:string)=>atoms.flatMap(atom=>ink(atom,source).filter(q=>Math.hypot(q[0]-point[0]-.5,q[1]-point[1]-.5)<=3).map(q=>({atom,q})));
    let join:{a:NativeOccurrence;b:NativeOccurrence;p:Point;q:Point;distance:number}|undefined;
    for(const a of at(first.id))for(const b of at(second.id)){const distance=Math.hypot(a.q[0]-b.q[0],a.q[1]-b.q[1]);if(!join||distance<join.distance)join={a:a.atom,b:b.atom,p:a.q,q:b.q,distance};}
    if(!join||join.distance>3)return fail('compact cap/side/return has an unresolved native joint');
    contacts.push({id:`${id}/joint/${vertex}/${i}`,from:join.a.id,to:join.b.id,fromSource:first.id,toSource:second.id,fromPort:bound.ports[vertex],toPort:bound.ports[vertex],fromAnchor:join.p,toAnchor:join.q,relation:'whole-compact-plane-cap-side-return'});
   }
  }
  const faces:Vertex[][]=[['left','center','foot','leftFoot'],['center','right','rightFoot','foot']];
  const sourceVertices=new Map([...p].map(([v,q])=>[bound.ports[v],q])),nativeVertices=new Map([...native].map(([v,q])=>[bound.ports[v],q]));
  return {ok:true,neighborhood,pattern:{id,origin:[origin,row],occurrences:atoms,contacts,continuations,paper:[],requiredSources:relations.map(relation=>relation.id),
   edgeCharts:relations.map(relation=>({source:relation.id,native:[nativeVertices.get(relation.ports[0])!,nativeVertices.get(relation.ports[1])!] as const})),
   surfaces:faces.map((face,i)=>({id:`${id}/face-${i}`,owner,outline:face.map(v=>native.get(v)!),charts:[[0,1,2],[0,2,3]].map(indices=>({native:indices.map(k=>native.get(face[k]!)!) as [Point,Point,Point],source:indices.map(k=>p.get(face[k]!)!) as [Point,Point,Point]}))}))},
   state:{width:columns[2]!,origin,row,depthRows:0,wallRows:height,context:'inclined'},slices:atoms.flatMap(atom=>atom.contributions.map(part=>({occurrence:atom.id,source:part.source,parameter:part.parameter!,pixels:part.pixels,role:'complete-compact-plane'}))),fullEdges:relations,sourceVertices,nativeVertices,regionSockets:[],implied:[],
   production:{status:'provisional-whole-fan-growth',capWitness:'independent-plane-axis-r14/compact-cap-side-return',returnWitnesses:['independent-plane-axis-r14/compact-cap-side-return']}};
 }
 /** A source-axis variant of the complete three-face neighborhood. Unlike
  * template(), horizontal growth and rise are independent parameters. One
  * cap/side/return grammar owns all nine axes and all seven shared corners.
  * Native words are never scaled, scored, harvested or selected per sample.
  * Parallel source axes share quantization; a converging projection retains
  * its actual seven corners and uses a whole slanted-post variant instead of
  * forcing those posts upright. */
 private sourceAxes(bound:Bound,neighborhood:string):FanResult{
  const fail=(obligation:string):FanResult=>({ok:false,owner:bound.owner,obligation});
  if(bound.boundary)return fail('source-axis solid needs its incident side faces');
  const p=bound.points,c=p.get('center')!,f=p.get('foot')!,l=p.get('left')!,r=p.get('right')!,a=p.get('apex')!,lf=p.get('leftFoot')!,rf=p.get('rightFoot')!;
  const same=(x:number,y:number)=>Math.abs(x-y)<.001;
  const parallel=same(l[0],lf[0])&&same(r[0],rf[0])&&same(c[0],f[0])&&same(a[0]+c[0],l[0]+r[0])&&same(a[1]+c[1],l[1]+r[1])&&same(lf[1]-l[1],f[1]-c[1])&&same(rf[1]-r[1],f[1]-c[1]);
  const Q=(x:number)=>16*Math.round(x/16),R=(y:number)=>Math.round(y/17);
  // Quantize THREE shared source axes, not seven unrelated corners. When
  // thickness occupies at most one native row, cap and return use ONE whole
  // plate variant with shared contributions. Inflating it into a taller box
  // would invent a side wall; overlapping two unrelated joint words would
  // instead reject the entire floor. Neither is a compact construction.
  const thin=[f[1]-c[1],lf[1]-l[1],rf[1]-r[1]].every(height=>R(height)<=1);
  const cx=Q(c[0]),cy=17*R(c[1]),ux=Q(c[0]-l[0]),uy=17*R(c[1]-l[1]),vx=Q(r[0]-c[0]),vy=17*R(c[1]-r[1]),height=thin?0:17*R(f[1]-c[1]);
  const points=new Map<Vertex,Point>(parallel?[['center',[cx,cy]],['left',[cx-ux,cy-uy]],['right',[cx+vx,cy-vy]],['apex',[cx-ux+vx,cy-uy-vy]],
   ['foot',[cx,cy+height]],['leftFoot',[cx-ux,cy-uy+height]],['rightFoot',[cx+vx,cy-vy+height]]]:[...p].map(([v,q])=>[v,[Q(q[0]),17*R(q[1])]]));
  if(thin)for(const [foot,cap]of [['leftFoot','left'],['rightFoot','right'],['foot','center']] as const)points.set(foot,points.get(cap)!);
  // One profile per COMPLETE source post, carried through its two joints
  // and every repeat. This is not an angle table or a per-row election.
  // Only a genuinely vertical projected feature can select the upright.
  const postProfiles=new Map<Role,{text:string;anchor:number}>();
  for(const [role,top,bottom]of [['leftPost',l,lf],['backPost',r,rf],['frontPost',c,f]] as const){const vertical=same(top[0],bottom[0]),right=bottom[0]>top[0];postProfiles.set(role,{text:vertical?'|':right?'＼':'/',anchor:vertical?1:right?7:3});}
  const id=`${neighborhood}/source-axis-plane`,atoms:NativeOccurrence[]=[],contacts:NativeContact[]=[],continuations:NativeContinuation[]=[];
  const joints=new Map<Vertex,{x:number;row:number;members:string[]}>();
  const add=(text:string,x:number,row:number,key:string,parts:readonly {role:Role;range:readonly [number,number];select?:(q:readonly number[])=>boolean}[])=>{
   const glyph=this.font.get(text);if(!glyph)throw Error(`Source-axis plane lacks native ${text}`);
   const shared=thin?parts.flatMap(part=>[part,...(part.role==='front'?[{...part,role:'base' as const}]:part.role==='topSlope'?[{...part,role:'sideReturn' as const}]:[])]):parts;
   const contributions=shared.flatMap(part=>{const edge=bound.edges[part.role]!,pixels=glyph.points.flatMap((q,i)=>!part.select||part.select(q)?[i]:[]);return pixels.length?[{source:edge.id,kind:'path' as const,pixels,parameter:relationRange(edge,bound.ports[roles[part.role][0]],part.range)}]:[];});
   atoms.push({id:`${id}/${key}`,text,x,row,contributions});return atoms.at(-1)!.id;
  };
  const compactVertex=(vertex:Vertex):Vertex=>thin?vertex==='leftFoot'?'left':vertex==='rightFoot'?'right':vertex==='foot'?'center':vertex:vertex;
  const endRange=(role:Role,vertex:Vertex):readonly [number,number]=>thin&&role.endsWith('Post')?[0,1]:compactVertex(roles[role][0])===vertex?[0,0]:[1,1];
  const word=(vertex:Vertex,text:string,anchorX:number,anchorY:number,owners:readonly (Role|readonly {role:Role;select?:(q:readonly number[])=>boolean}[])[])=>{
   const q=points.get(vertex)!,x=q[0]-anchorX,row=R(q[1]-anchorY),members:string[]=[];let pen=x;
   for(const [i,ch]of [...text].entries()){
    const assignment=owners[i]!,parts=typeof assignment==='string'?[{role:assignment}]:assignment;
    members.push(add(ch,pen,row,`joint/${vertex}/${i}`,parts.map(part=>({...part,range:endRange(part.role,vertex)}))));pen+=this.font.get(ch)!.advance;
   }
   joints.set(vertex,{x,row,members});
  };
  // These are complete, independently authored incident words, not retained
  // entry excerpts. Extra stem ink at the outside corners is on the back/
  // outside of the solid, never a replacement for either cap arm.
  word('apex','／＼',15,0,['leftSlope','back']);
  if(thin){
   word('left','<',0,7,[[{role:'leftSlope',select:q=>q[1]!<=7},{role:'front',select:q=>q[1]!>=7},{role:'leftPost',select:q=>q[0]!<=2}]]);
   word('right','>',this.font.get('>')!.advance-1,7,[[{role:'back',select:q=>q[1]!<=7},{role:'topSlope',select:q=>q[1]!>=7},{role:'backPost',select:q=>q[0]!>=this.font.get('>')!.advance-3}]]);
   word('center','＼／',15,14,[[{role:'front'},{role:'frontPost',select:q=>q[1]!>=12}],[{role:'topSlope'},{role:'frontPost',select:q=>q[1]!>=12}]]);
   for(const vertex of ['leftFoot','rightFoot','foot'] as const)joints.set(vertex,joints.get(compactVertex(vertex))!);
  }else{
   const leftPost=postProfiles.get('leftPost')!,rightPost=postProfiles.get('backPost')!,middlePost=postProfiles.get('frontPost')!;
   word('left',leftPost.text+'く',leftPost.anchor,7,['leftPost',[{role:'leftSlope',select:q=>q[1]!<=7},{role:'front',select:q=>q[1]!>=7}]]);
   word('right','>'+rightPost.text,this.font.get('>')!.advance+rightPost.anchor,7,[[{role:'back',select:q=>q[1]!<=7},{role:'topSlope',select:q=>q[1]!>=7}], 'backPost']);
   const prefix=this.font.get('.')!.advance+this.font.get('｀')!.advance;
   word('center','.｀Y´.',prefix+4,7,['front','front',[{role:'front',select:q=>q[1]!<=8&&q[0]!<=4},{role:'topSlope',select:q=>q[1]!<=8&&q[0]!>=4},{role:'frontPost',select:q=>q[1]!>=7}],'topSlope','topSlope']);
   word('leftFoot',leftPost.text+'＼',leftPost.anchor,7,['leftPost','base']);
   word('rightFoot','／'+rightPost.text,16+rightPost.anchor,7,['sideReturn','backPost']);
   word('foot','＼'+middlePost.text+'_／',16+middlePost.anchor,14,['base','frontPost','sideReturn','sideReturn']);
  }
  const endpoint=(vertex:Vertex,role:Role,exit:boolean):{x:number;row:number}=>{
   const joint=joints.get(vertex)!,members=atoms.filter(atom=>joint.members.includes(atom.id)&&atom.contributions.some(part=>part.source===bound.edges[role]!.id));
   if(!members.length)throw Error(`Source-axis plane joint lost ${vertex}/${role}`);
   // A part's scalar boundary, not a fitted ink location. All horizontal
   // lengths below are therefore exact equations in existing native advances.
   const left=Math.min(...members.map(atom=>atom.x)),right=Math.max(...members.map(atom=>atom.x+this.font.get(atom.text)!.advance));
   return {x:exit?right:left,row:joint.row};
  };
  const run=(role:Role,from:Vertex,to:Vertex):string|undefined=>{
   const u=points.get(from)!,v=points.get(to)!,right=v[0]>u[0],start=endpoint(from,role,right),finish=endpoint(to,role,!right);
   const first=start.row+1,last=finish.row-1,count=last-first+1;
   const span=right?finish.x-start.x:start.x-finish.x;
   if(count<1){
    // A cap axis with no intervening row is not an absent face. Its two
    // incident words already fix BOTH ends. Complete the same-row rail by
    // its exact scalar equation; every +16px width step adds one whole unit
    // without changing the enduring short suffix or either corner.
    if(start.row===finish.row&&span>0){
     const short=((span%16)*13)%16,full=(span-5*short)/16;
     if(full>=0){
      const text='＿'.repeat(full)+'_'.repeat(short),left=Math.min(start.x,finish.x);let pen=left;
      for(const [i,ch]of [...text].entries()){
       const glyph=this.font.get(ch)!,xs=glyph.points.map(q=>q[0]!),t=(x:number)=>(x-u[0])/(v[0]-u[0]);
       const values=[t(pen+Math.min(...xs)),t(pen+Math.max(...xs))],lo=Math.max(0,Math.min(...values)),hi=Math.min(1,Math.max(...values));
       const range:readonly [number,number]=roles[role][0]===from?[lo,hi]:[1-hi,1-lo];
       add(ch,pen,start.row,`run/${role}/rail/${i}`,[{role,range}]);pen+=glyph.advance;
      }
      if(pen!==left+span)throw Error('Compact source-axis rail failed its cursor equation');
      return;
     }
    }
    // Adjacent complete corner words sometimes ALREADY continue this short
    // axis. Verify only their named source ink, not an unrelated glyph's
    // convenient stroke. This measures an existing interface; it never
    // selects a glyph, moves a corner or fills an unconnected gap.
    if(finish.row===start.row+1||start.row===finish.row&&span===0){
     const edge=bound.edges[role]!,at=(vertex:Vertex)=>atoms.filter(atom=>joints.get(vertex)!.members.includes(atom.id))
      .flatMap(atom=>atom.contributions.filter(part=>part.source===edge.id).flatMap(part=>part.pixels.map(index=>{
       const q=this.font.get(atom.text)!.points[index]!;return {atom,p:[atom.x+q[0]!+.5,atom.row*17+q[1]!+.5] as Point};
      })));
     let join:{a:NativeOccurrence;b:NativeOccurrence;p:Point;q:Point;distance:number}|undefined;
     for(const a of at(from))for(const b of at(to)){const distance=Math.hypot(a.p[0]-b.p[0],a.p[1]-b.p[1]);if(!join||distance<join.distance)join={a:a.atom,b:b.atom,p:a.p,q:b.p,distance};}
     if(join&&join.distance<=3){
      const members=new Set([...joints.get(from)!.members,...joints.get(to)!.members]);
      for(let i=0;i<atoms.length;i++)if(members.has(atoms[i]!.id))atoms[i]={...atoms[i]!,contributions:atoms[i]!.contributions.map(part=>part.source===edge.id?{...part,parameter:[0,1] as const}:part)};
      continuations.push({from:join.a.id,to:join.b.id,fromAnchor:join.p,toAnchor:join.q,distance:3});return;
     }
    }
    return `${role}:source-axis interval needs a compact complete incident variant`;
   }
   // 5 and16 are real scalar advances. The short suffix is constant under
   // every +16px scale step; growing the form adds whole long units instead
   // of respelling a rail every fifth step as the old modulo-five fit did.
   // Fixed complete-run width variants, in authored order. A variant is
   // selected once for the entire run by its exact scalar cursor equation;
   // no row/sample chooses a glyph, angle or ink-fit winner.
   let selected:{diagonal:string;step:number;short:number;full:number}|undefined;
   // Full and half slash are separate complete run productions (the
   // practitioner's 16-dot and 8-dot shift units). Try the narrower family
   // only when the serving full-width words cannot close this exact axis.
   // Once chosen, EVERY intervening row keeps that unit and its ordinals.
   for(const diagonal of right?['＼','ヽ','\\']:['／','ノ','/']){
    const step=this.font.get(diagonal)!.advance,remaining=span-step*count;
    if(remaining<0)continue;
    const short=((remaining%16)*13)%16,full=(remaining-5*short)/16;
    if(full>=0){selected={diagonal,step,short,full};break;}
   }
   if(!selected)return `${role}:source-axis interval needs an adjacent complete width variant`;
   const {diagonal,step,short,full}=selected;
   let pen=start.x;
   for(let k=0;k<count;k++){
    const longs=Math.floor((k+1)*full/count)-Math.floor(k*full/count),shorts=Math.floor((k+1)*short/count)-Math.floor(k*short/count);
    const text=right?diagonal+'＿'.repeat(longs)+'_'.repeat(shorts):'_'.repeat(shorts)+'＿'.repeat(longs)+diagonal;
    const advance=step+16*longs+5*shorts,at=right?pen:pen-advance;let cursor=at;
    const localRange:readonly [number,number]=[(k+.5)/(count+1),(k+1.5)/(count+1)],range:readonly [number,number]=roles[role][0]===from?localRange:[1-localRange[1],1-localRange[0]];
    for(const [i,ch]of [...text].entries()){add(ch,cursor,first+k,`run/${role}/${k}/${i}`,[{role,range}]);cursor+=this.font.get(ch)!.advance;}
    pen+=right?advance:-advance;
   }
   if(pen!==finish.x)throw Error('Source-axis native rail failed its cursor equation');
  };
  const capRuns:readonly (readonly [Role,Vertex,Vertex])[]=[['leftSlope','apex','left'],['back','apex','right'],['front','left','center'],['topSlope','right','center']];
  const returnRuns:readonly (readonly [Role,Vertex,Vertex])[]=thin?[]:[['base','leftFoot','foot'],['sideReturn','rightFoot','foot']];
  for(const [role,from,to]of [...capRuns,...returnRuns]){const problem=run(role,from,to);if(problem)return fail(problem);}
  const uprights:readonly (readonly [Role,Vertex,Vertex])[]=thin?[]:[['leftPost','left','leftFoot'],['backPost','right','rightFoot'],['frontPost','center','foot']];
  for(const [role,from,to]of uprights){
   const begin=joints.get(from)!,end=joints.get(to)!,profile=postProfiles.get(role)!,top=points.get(from)!,bottom=points.get(to)!;
   for(let row=begin.row+1;row<end.row;row++){
    const t=(row-begin.row)/(end.row-begin.row),x=Math.round(top[0]+t*(bottom[0]-top[0]))-profile.anchor;
    add(profile.text,x,row,`post/${role}/${row-begin.row}`,[{role,range:[t,(row-begin.row+1)/(end.row-begin.row)]}]);
   }
  }
  // D1 stage1 acts on an entire source-owned run, with all other parts and
  // its incident joints already present. It is NOT a per-glyph phase repair:
  // every repeat keeps its scalar, ordinal and relative placement. The
  // existing finite whitespace language supplies nearest-legal pen order;
  // there are no glyph candidates, costs or solver frontier.
  for(const [role]of uprights){
   const members=new Set(atoms.filter(atom=>atom.id.startsWith(`${id}/post/${role}/`)).map(atom=>atom.id));
   if(!members.size)continue;
   const constraints:NativeCursorConstraint[]=[],partRows=new Map<number,NativeOccurrence[]>();
   for(const atom of atoms){const row=partRows.get(atom.row)??[];row.push(atom);partRows.set(atom.row,row);}
   for(const row of partRows.values()){
    row.sort((a,b)=>a.x-b.x);
    for(let i=0;i<row.length;i++){
     const atom=row[i]!,before=row[i-1],mine=members.has(atom.id),prior=!!before&&members.has(before.id);
     if(!before){if(mine)constraints.push({gap:atom.x,leading:true,direction:1});continue;}
     if(mine===prior)continue;
     constraints.push({gap:atom.x-before.x-this.font.get(before.text)!.advance,leading:false,direction:mine?1:-1});
    }
   }
   const phase=nativeProgramPhase(constraints,undefined,'preformatted');
   if(phase===undefined)return fail(`${role}:complete post run needs the next whole-part D1 variant`);
   if(phase)for(let i=0;i<atoms.length;i++)if(members.has(atoms[i]!.id))atoms[i]={...atoms[i]!,x:atoms[i]!.x+phase};
  }
  // This is a complete source construction, not yet the visible scalar tape.
  // Front/back parts can occupy the same row before geometric visibility;
  // rejecting their cursor gap here discards the VISIBLE solid as well. Keep
  // both source-bound parts for ConstructionVisibility, then let the existing
  // whole-part placement and NativePage enforce the final scalar language.
  // No hidden unit is made visible and no glyph is trimmed or substituted.
  // Establish actual native contacts only AFTER the entire source program is
  // fixed. Measuring an existing interface never elects or substitutes ink.
  for(const [vertex,joint]of joints){
   const incident=(Object.keys(roles) as Role[]).filter(role=>roles[role].includes(vertex));
   for(let i=1;i<incident.length;i++){
    const left=bound.edges[incident[0]!]!,right=bound.edges[incident[i]!]!;let closest:{a:NativeOccurrence;b:NativeOccurrence;p:Point;q:Point;distance:number}|undefined;
    const ink=(atom:NativeOccurrence,source:string)=>atom.contributions.filter(part=>part.source===source).flatMap(part=>part.pixels.map(index=>{const q=this.font.get(atom.text)!.points[index]!;return [atom.x+q[0]!+.5,17*atom.row+q[1]!+.5] as Point;}));
    const members=atoms.filter(atom=>joint.members.includes(atom.id));
    for(const a of members)for(const b of members)for(const p of ink(a,left.id))for(const q of ink(b,right.id)){const distance=Math.hypot(p[0]-q[0],p[1]-q[1]);if(!closest||distance<closest.distance)closest={a,b,p,q,distance};}
    if(!closest)continue;
    if(closest.distance<=3)contacts.push({id:`${id}/joint/${vertex}/${i}`,from:closest.a.id,to:closest.b.id,fromSource:left.id,toSource:right.id,fromPort:bound.ports[vertex],toPort:bound.ports[vertex],fromAnchor:closest.p,toAnchor:closest.q,relation:'complete-source-axis-plane'});
    else continuations.push({from:closest.a.id,to:closest.b.id,fromAnchor:closest.p,toAnchor:closest.q,distance:closest.distance});
   }
  }
  const faces:Vertex[][]=[['apex','right','center','left'],['left','center','foot','leftFoot'],['center','right','rightFoot','foot']];
  const surfaces=faces.map((face,i)=>({id:`${id}/face-${i}`,owner:bound.owner,outline:face.map(v=>points.get(v)!),charts:[[0,1,2],[0,2,3]].map(indices=>({native:indices.map(k=>points.get(face[k]!)!) as [Point,Point,Point],source:indices.map(k=>p.get(face[k]!)!) as [Point,Point,Point]}))}));
  const sourceVertices=new Map([...p].map(([v,q])=>[bound.ports[v],q])),nativeVertices=new Map([...points].map(([v,q])=>[bound.ports[v],q]));
  return {ok:true,neighborhood,pattern:{id,origin:[0,0],occurrences:atoms,contacts,continuations,paper:[],requiredSources:Object.values(bound.edges).map(edge=>edge.id),surfaces,
   edgeCharts:Object.values(bound.edges).map(edge=>({source:edge.id,native:[nativeVertices.get(edge.ports[0])!,nativeVertices.get(edge.ports[1])!] as const})),
   repeatRuns:[...capRuns,...returnRuns,...uprights].map(([role])=>({id:`${id}/part/${role}`,members:atoms.filter(atom=>atom.id.startsWith(`${id}/run/${role}/`)||atom.id.startsWith(`${id}/post/${role}/`)).map(atom=>atom.id)}))},
   state:{width:ux+vx,origin:0,row:0,depthRows:(uy+vy)/17,wallRows:height/17,context:'inclined'},slices:atoms.flatMap(atom=>atom.contributions.map(part=>({occurrence:atom.id,source:part.source,parameter:part.parameter!,pixels:part.pixels,role:'complete-source-axis-plane'}))),fullEdges:Object.values(bound.edges),sourceVertices,nativeVertices,regionSockets:[],implied:[],
   production:{status:'provisional-whole-fan-growth',capWitness:'independent-plane-axis-r14/cap-side-return',returnWitnesses:['independent-plane-axis-r14/cap-side-return']}};
 }
 private template(a:number,b:number,h:number,boundary=false):Template{
  const key=`${boundary?'boundary':'fan'}/${a}/${b}/${h}`,cached=this.cache.get(key);if(cached)return cached;
  const pitch=44,dy=17,right=46+sources.inclined.rightPaperAdvance;
  // One shared paper axis solves three simultaneous scalar equations:
  // right−46=10, right−15=41, right−34=22. Whole44px words preserve
  // all three native languages when more units enter between anchors.
  const vertices=new Map<Vertex,Point>([['center',[30,24]],['apex',[29+pitch*(b-a),-dy*(a+b)]],
   ['left',[12-pitch*a,17-dy*a]],['right',[right+1+pitch*b,17-dy*b]],
   ['leftFoot',[12-pitch*a,32-dy*a+dy*h]],['rightFoot',[right+1+pitch*b,32-dy*b+dy*h]],['foot',[boundary?31:30,47+dy*h]]]);
  const atoms:NativeOccurrence[]=[],participation=new Map<string,{role:Role;pixels:number[]}[]>();
  const add=(text:string,x:number,row:number,id:string,role:Role,select?:(p:readonly number[])=>boolean)=>{
   if(boundary&&['front','topSlope','frontPost'].includes(role))return;
   const glyph=this.font.get(text);if(!glyph)throw Error(`Inclined plane source lacks native ${text}`);
   const pixels=glyph.points.flatMap((p,i)=>!select||select(p)?[i]:[]);if(!pixels.length)return;
   const prior=atoms.find(atom=>atom.id===id);if(!prior)atoms.push({id,text,x,row,contributions:[]});
   const list=participation.get(id)??[];list.push({role,pixels});participation.set(id,list);
  };
  const word=(text:string,x:number,row:number,id:string,role:Role)=>{
   for(const [cp,c]of [...text].entries()){add(c,x,row,`${id}/cp-${cp}`,role);x+=this.font.get(c)!.advance;}
  };
  // Retained compact corner/shoulder units. Their counterparts never move
  // independently: all positions are functions of the SAME three axes.
  add('.',11-pitch*a,-a,'apex/left-dot','leftSlope');
  add('／',14+pitch*(b-a),-a-b,'apex/left','leftSlope');
  add('＼',30+pitch*(b-a),-a-b,'apex/right','back');
  word('.｀',15,1,'fork/left','front');
  add('Y',26,1,'fork/shared','front',p=>p[1]!<=8&&p[0]!<=4);
  add('Y',26,1,'fork/shared','topSlope',p=>p[1]!<=8&&p[0]!>=4);
  add('Y',26,1,'fork/shared','frontPost',p=>p[1]!>=7);
  word('´.',35,1,'fork/right','topSlope');
  add('.',11-pitch*a,2-a+h,'foot/left-dot','base');
  add('＼',14-pitch*a,2-a+h,'foot/left','base');
  add('!',30,2+h,'foot/shared','frontPost');
  // Removing the interior ! also removes its4px scalar cursor. The boundary
  // version uses one actual5px space between the two retained return arms.
  add('／',boundary?35:34,2+h,'foot/right','sideReturn');
  for(let r=0;r<=h;r++){
   add('|',11-pitch*a,1-a+r,`wall/left-${r}`,'leftPost');
   add('|',right+pitch*b,1-b+r,`wall/right-${r}`,'backPost');
  }
  for(let r=0;r<h;r++)add('|',30,2+r,`wall/center-${r}`,'frontPost');
  const {ascending,descending}=sources.inclined.growth;
  for(let k=1;k<=a;k++){
   word(descending,15-pitch*k,1-k,`front/unit-${k}`,'front');
   word(descending,46+pitch*(b-a)+pitch*(k-1),-a-b+k,`back/unit-${k}`,'back');
   word(descending,30-pitch*a+pitch*(k-1),3-a+h+(k-1),`base/unit-${k}`,'base');
  }
  for(let k=1;k<=b;k++){
   word(ascending,right+pitch*(k-1),1-k,`top/unit-${k}`,'topSlope');
   word(ascending,14-pitch*a+pitch*(k-1),-a-(k-1),`left/unit-${k}`,'leftSlope');
   word(ascending,49+pitch*(k-1),1+h-(k-1),`side/unit-${k}`,'sideReturn');
  }
  // Lower the immutable whole program to native source-u supports. This is
  // attribution of already selected ink, never a glyph or image fit.
  for(let i=0;i<atoms.length;i++){
   const atom=atoms[i]!,glyph=this.font.get(atom.text)!;
   const contributions=participation.get(atom.id)!.map(({role,pixels})=>{
    const [from,to]=roles[role],p=vertices.get(from)!,q=vertices.get(to)!,dx=q[0]-p[0],dy=q[1]-p[1],length=dx*dx+dy*dy;
    const parameters=pixels.map(index=>{const point=glyph.points[index]!;return Math.max(0,Math.min(1,((atom.x+point[0]!-p[0])*dx+(atom.row*17+point[1]!-p[1])*dy)/length));});
    return {source:role,kind:'path' as const,pixels,parameter:[Math.min(...parameters),Math.max(...parameters)] as const};
   });
   atoms[i]={...atom,contributions};
  }
  const contacts:NativeContact[]=[],continuations:NativeContinuation[]=[];
  const ink=(atom:NativeOccurrence,role:Role)=>atom.contributions.filter(c=>c.source===role).flatMap(c=>c.pixels.map(i=>{const p=this.font.get(atom.text)!.points[i]!;return [atom.x+p[0]!+.5,atom.row*17+p[1]!+.5] as Point;}));
  for(const [port,point]of vertices){
   const incident=(Object.keys(roles) as Role[]).filter(role=>roles[role].includes(port)&&(!boundary||!['front','topSlope','frontPost'].includes(role)));
   for(let i=1;i<incident.length;i++){
    const left=incident[0]!,right=incident[i]!;
    // Only each source arm's END interfaces can participate at this corner.
    const end=(role:Role)=>atoms.filter(atom=>atom.contributions.some(c=>c.source===role&&c.parameter&&(roles[role][0]===port?c.parameter[0]<=1e-6:c.parameter[1]>=1-1e-6)));
    const nearest=(role:Role)=>{const list=end(role);return list.length?list:atoms.filter(atom=>atom.contributions.some(c=>c.source===role)).sort((a,b)=>Math.min(...ink(a,role).map(p=>(p[0]-point[0])**2+(p[1]-point[1])**2))-Math.min(...ink(b,role).map(p=>(p[0]-point[0])**2+(p[1]-point[1])**2))).slice(0,2);};
    let best:{a:NativeOccurrence;b:NativeOccurrence;p:Point;q:Point;distance:number}|undefined;
    for(const a of nearest(left))for(const b of nearest(right))for(const p of ink(a,left))for(const q of ink(b,right)){
     const distance=(p[0]-q[0])**2+(p[1]-q[1])**2;if(!best||distance<best.distance)best={a,b,p,q,distance};
    }
    if(!best)throw Error('Inclined native program lost an incidence');
    if(best.distance<=9)contacts.push({id:`contact/${port}/${i}`,from:best.a.id,to:best.b.id,fromSource:left,toSource:right,fromPort:port,toPort:port,fromAnchor:best.p,toAnchor:best.q,relation:'whole-inclined-three-face-plane'});
    else continuations.push({from:best.a.id,to:best.b.id,fromAnchor:best.p,toAnchor:best.q,distance:Math.sqrt(best.distance)});
   }
  }
  const rows=new Map<number,NativeOccurrence[]>();for(const atom of atoms){const row=rows.get(atom.row)??[];row.push(atom);rows.set(atom.row,row);}
  const admissible=[...rows.values()].every(row=>{row.sort((a,b)=>a.x-b.x);return row.every((atom,i)=>i===0||!!nativeSpacePlan(atom.x-row[i-1]!.x-this.font.get(row[i-1]!.text)!.advance,0,false,false,'preformatted'));});
  if(boundary)vertices.delete('center');
  const template:Template={vertices,admissible,pattern:{id:'inclined-plane',occurrences:atoms,contacts,continuations,paper:[],requiredSources:Object.keys(roles).filter(role=>!boundary||!['front','topSlope','frontPost'].includes(role))}};
  this.cache.set(key,template);return template;
 }
 grow(relations:readonly PosedRelation[],previous?:FanState,neighborhood=relations[0]?.owner??''):FanResult{
  const bound=this.bind(relations),owner=relations[0]?.owner??'',fail=(obligation:string):FanResult=>({ok:false,owner,obligation});
  if(!bound)return fail('source is outside the inclined three-face context');
  const nextFamily=(obligation:string):FanResult=>{const next=this.sourceAxes(bound,neighborhood);return next.ok?next:fail(`${obligation}; ${next.obligation}`);};
  const p=bound.points,c=p.get('center'),l=p.get('left')!,r=p.get('right')!,f=p.get('foot')!,apex=p.get('apex')!;
  const na=Math.max(0,Math.round((bound.boundary?r[0]-apex[0]-28:c![0]-l[0]-18)/44));
  const nb=Math.max(0,Math.round((bound.boundary?apex[0]-l[0]-17:r[0]-c![0]-17-sources.inclined.rightPaperAdvance)/44));
  const nh=Math.max(0,Math.round((bound.boundary?((p.get('leftFoot')![1]-l[1])+(p.get('rightFoot')![1]-r[1]))/2-15:f[1]-c![1]-23)/17));
  // Quantize the three shared axes ONCE. No nearby integer tuples, costs,
  // prior-frame preference or glyph candidates elect this construction.
  const template=this.template(na,nb,nh,!!bound.boundary);
  if(!template.admissible)return nextFamily('inclined source family has no legal complete scalar tape');
  const origin=Math.round([...p].reduce((sum,[id,q])=>sum+q[0]-template.vertices.get(id)![0],0)/p.size),row=Math.round([...p].reduce((sum,[id,q])=>sum+q[1]-template.vertices.get(id)![1],0)/p.size/17);
   // Bounds follow the quantization of this WHOLE function. Each inserted
   // source word changes an axis by44x17; its half-step uncertainty travels
   // to every dependent corner. A single17px scalar gate incorrectly deleted
   // the full construction between legitimate whole-word size alternatives.
   const uncertainty:Record<Vertex,Point>={center:[17,17],left:[39,25.5],right:[39,25.5],apex:[61,34],
    leftFoot:[39,34],rightFoot:[39,34],foot:[17,25.5]};
   const residuals=[...p].map(([id,q])=>{const n=template.vertices.get(id)!;return {id,dx:origin+n[0]-q[0],dy:row*17+n[1]-q[1]};});
  if(residuals.some(({id,dx,dy})=>Math.abs(dx)>uncertainty[id][0]||Math.abs(dy)>uncertainty[id][1]))return nextFamily('inclined source axes exceed the whole-word growth coverage');
  const id=`${neighborhood}/inclined-plane`,prefix=(local:string)=>`${id}/${local}`;
  const contribution=(role:string)=>bound.edges[role as Role]!,parameter=(role:string,range:readonly [number,number])=>relationRange(contribution(role),bound.ports[roles[role as Role][0]],range);
  const occurrences=template.pattern.occurrences.map(atom=>({...atom,id:prefix(atom.id),contributions:atom.contributions.map(c=>({...c,source:contribution(c.source).id,parameter:parameter(c.source,c.parameter!)}))}));
  const contacts=template.pattern.contacts.map(contact=>({...contact,id:prefix(contact.id),from:prefix(contact.from),to:prefix(contact.to),fromSource:contribution(contact.fromSource).id,toSource:contribution(contact.toSource).id,fromPort:bound.ports[contact.fromPort as Vertex],toPort:bound.ports[contact.toPort as Vertex]}));
  const continuations=template.pattern.continuations?.map(link=>({...link,from:prefix(link.from),to:prefix(link.to)}));
  const faces:Vertex[][]=bound.boundary?[['apex','right','rightFoot','foot','leftFoot','left']]:[['apex','right','center','left'],['left','center','foot','leftFoot'],['center','right','rightFoot','foot']];
  const charts=faces.flatMap(face=>Array.from({length:face.length-2},(_,i)=>[0,i+1,i+2]).map(indices=>({native:indices.map(i=>template.vertices.get(face[i]!)!) as [Point,Point,Point],source:indices.map(i=>p.get(face[i]!)!) as [Point,Point,Point]})));
  const slices:FanSlice[]=occurrences.flatMap(atom=>atom.contributions.map(c=>({occurrence:atom.id,source:c.source,parameter:c.parameter!,pixels:c.pixels,role:'inclined-source'})));
  const nativeVertices=new Map([...template.vertices].map(([v,p])=>[bound.ports[v],p]));
  return {ok:true,neighborhood,pattern:{id,origin:[origin,row],occurrences,contacts,continuations,paper:[],requiredSources:relations.map(r=>r.id),
   edgeCharts:relations.map(relation=>({source:relation.id,native:[nativeVertices.get(relation.ports[0])!,nativeVertices.get(relation.ports[1])!] as const})),
   surfaces:[{id:`${id}/body`,owner,outline:(['apex','right','rightFoot','foot','leftFoot','left'] as Vertex[]).map(v=>template.vertices.get(v)!),charts}]},
   state:{width:Math.round(r[0]-l[0]),origin,row,context:'inclined'},slices,fullEdges:relations,
   sourceVertices:new Map([...p].map(([v,p])=>[bound.ports[v],p])),nativeVertices:new Map([...template.vertices].map(([v,p])=>[bound.ports[v],p])),regionSockets:[],implied:[],
   production:{status:'provisional-whole-fan-growth',capWitness:`${sources.inclined.source}/entry2`,returnWitnesses:[`${sources.inclined.source}/entry11`]}};
 }
}
