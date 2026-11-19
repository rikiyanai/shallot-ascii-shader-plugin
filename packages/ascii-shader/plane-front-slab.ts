import {relationPoint,relationRange,type PosedRelation} from './posed-graph.ts';
import type {NativeGlyph,NativeOccurrence,NativeContact} from './native-page.ts';
import type {NativeContinuation} from './native-layout.ts';
import type {FanResult,FanState} from './plane-cap-fan.ts';
import {nativeSpacePlan} from './text/native-whitespace.ts';
import {planeBoundary} from './plane-boundary.ts';

type Point=readonly [number,number];
/** Two visible plane regions with a shared crease. The complete native
 * |＿＿| / |￣￣| plane-and-post relationship is witnessed in the MLT
 * geometry collection's Menger construction (a7065b14…, entry157).
 * Repeat/row growth is a provisional function of that relationship. This
 * accepts SOURCE incidence, including hidden spans, never a mesh/scene name.
 */
export class PlaneFrontSlab {
 private readonly font:Map<string,NativeGlyph>;
 constructor(font:readonly NativeGlyph[]){this.font=new Map(font.map(g=>[g.text,g]));}
 private units(span:number):{short:number;long:number}|undefined{
  const long=((span%5)+5)%5,short=(span-long*16)/5;
  if(Number.isInteger(short)&&short>=0)return {short,long};
 }
 grow(relations:readonly PosedRelation[],previous?:FanState,axes?:{width:number;origin:number;row?:number},neighborhood=relations[0]?.owner??''):FanResult{
  if(relations.length===4)return this.panel(relations,neighborhood);
  const owner=relations[0]?.owner??'',fail=(obligation:string):FanResult=>({ok:false,owner,obligation});
  if(relations.length!==7||relations.some(r=>r.owner!==owner||r.kind==='smooth-contour'))return fail('not a complete two-face crease');
  const adjacency=new Map<string,string[]>(),source=new Map<string,Point>();
  for(const r of relations)for(const [i,port]of r.ports.entries()){
   const p=relationPoint(r,i);if(!p.every(Number.isFinite))return fail('nonprojectable plane source');
   source.set(port,[p[0],p[1]]);const neighbors=adjacency.get(port)??[];neighbors.push(r.ports[1-i]!);adjacency.set(port,neighbors);
  }
  const middle=relations.filter(r=>r.ports.every(p=>adjacency.get(p)?.length===3));
  if(source.size!==6||middle.length!==1||[...adjacency.values()].some(n=>n.length!==2&&n.length!==3))return fail('source is not one two-face crease');
  const crease=middle[0]!,ends=[...crease.ports].sort((a,b)=>source.get(a)![0]-source.get(b)![0]),left=ends[0]!,right=ends[1]!;
  const leftNeighbors=adjacency.get(left)!.filter(p=>p!==right).sort((a,b)=>source.get(a)![1]-source.get(b)![1]);
  const rightNeighbors=adjacency.get(right)!.filter(p=>p!==left).sort((a,b)=>source.get(a)![1]-source.get(b)![1]);
  const [backLeft,baseLeft]=leftNeighbors as [string,string],[backRight,baseRight]=rightNeighbors as [string,string];
  const edge=(a:string,b:string)=>relations.find(r=>r.ports.includes(a)&&r.ports.includes(b));
  const back=edge(backLeft,backRight),base=edge(baseLeft,baseRight);
  if(!back||!base||source.get(backLeft)![1]>=source.get(left)![1]||source.get(baseLeft)![1]<=source.get(left)![1]||source.get(backRight)![1]>=source.get(right)![1]||source.get(baseRight)![1]<=source.get(right)![1])return fail('two-face orientation needs a different whole plane function');
  const xLeft=(source.get(left)![0]+source.get(baseLeft)![0])/2;
  const xRight=(source.get(right)![0]+source.get(baseRight)![0])/2;
  const yBack=(source.get(backLeft)![1]+source.get(backRight)![1])/2,yFront=(source.get(left)![1]+source.get(right)![1])/2,yBase=(source.get(baseLeft)![1]+source.get(baseRight)![1])/2;
  // Retained rails own the row count. Rounding their SEPARATION and then
  // independently centering the whole slab loses a row between legitimate
  // native anchor positions. Grow complete paired rows between these rails.
  const row=axes?.row??Math.floor((yBack-15)/17),depthRows=Math.max(1,Math.round((yFront-15)/17)-row);
  const wallRows=Math.max(0,Math.round((yBase-15)/17)-row-depthRows);
  // A projected plane has TWO rails. Their source corners supply one coupled
  // taper, rather than forcing a rectangle or spelling unrelated mesh edges.
  // Paired sloping side rows grow between the retained rails; ／ and ＼ are
  // the intact plane-side words of box entries20/18 in d00be527… . Mirroring
  // the complete pair and varying its taper is a provisional growth rule.
  const leftTaper=Math.round(source.get(backLeft)![0]-xLeft),rightSeed=Math.round(source.get(backRight)![0]-xRight);
  if(Math.max(Math.abs(leftTaper),Math.abs(rightSeed))>32*depthRows)return fail('projected plane taper exceeds the paired source-side growth');
  // Ordinary width growth inserts the native5px underscore unit between
  // the retained |￣| interfaces. A joint return can supply its own coupled
  // width parameter instead of independently moving either rail endpoint.
  const chosenWidth=axes?.width??4+16*Math.max(1,Math.round((xRight-xLeft-4)/16));
  const railUnits=this.units(chosenWidth-4),backUnits=this.units(chosenWidth+rightSeed-leftTaper+2);
  if(!railUnits||!backUnits)return fail('complete plane axes need a different whole native rail family');
  const choice={width:chosenWidth,origin:axes?.origin??Math.round((xLeft-1+xRight-chosenWidth-1)/2),rightTaper:rightSeed,back:backUnits,rail:railUnits};
  const {width,origin,rightTaper}=choice,frontRow=depthRows,baseRow=depthRows+wallRows;
  const native=new Map<string,Point>([[backLeft,[1+leftTaper,15]],[backRight,[width+1+rightTaper,15]],[left,[1,17*frontRow+15]],[right,[width+1,17*frontRow+15]],
   [baseLeft,[1,17*baseRow+15]],[baseRight,[width+1,17*baseRow+15]]]);
  if([...native].some(([id,p])=>Math.hypot(origin+p[0]-source.get(id)![0],row*17+p[1]-source.get(id)![1])>17))return fail('source plane axes exceed the frontal native growth envelope');
  const occurrences:NativeOccurrence[]=[],slices:Extract<FanResult,{ok:true}>['slices']=[];
  const add=(text:string,x:number,r:number,role:string,relation:PosedRelation,from:string,range:readonly [number,number])=>{
   const glyph=this.font.get(text)!;const id=`${neighborhood}/plane-slab/${role}`,pixels=glyph.points.map((_,i)=>i);
   const parameter=relationRange(relation,from,range);
   const contribution={source:relation.id,kind:'path' as const,pixels,parameter},shared=occurrences.find(atom=>atom.text===text&&atom.x===x&&atom.row===r);
   if(shared){const index=occurrences.indexOf(shared);occurrences[index]={...shared,contributions:[...shared.contributions,contribution]};}
   else occurrences.push({id,text,x,row:r,contributions:[contribution]});
   slices.push({occurrence:shared?.id??id,source:relation.id,pixels,parameter,role});
  };
  const rail=(words:{short:number;long:number},start:number,r:number,role:string,relation:PosedRelation,from:string,total:number)=>{
   let x=start;const text=role==='back'&&words.short?'_'+'＿'.repeat(words.long)+'_'.repeat(words.short-1):'＿'.repeat(words.long)+'_'.repeat(words.short);
   for(const [i,c]of [...text].entries()){
    const glyph=this.font.get(c)!,points=glyph.points.map(p=>p[0]!);
    const sourceStart=native.get(from)![0];
    add(c,x,r,`${role}/${i}`,relation,from,[Math.max(0,(x+Math.min(...points)-sourceStart)/total),Math.min(1,(x+Math.max(...points)-sourceStart)/total)]);x+=glyph.advance;
   }
  };
  rail(choice.back,1+leftTaper,0,'back',back,backLeft,width+rightTaper-leftTaper);
  rail(choice.rail,4,frontRow,'front',crease,left,width);
  rail(choice.rail,4,baseRow,'base',base,baseLeft,width);
  for(const [side,upper,front,lower,x,taper]of [['left',backLeft,left,baseLeft,0,leftTaper],['right',backRight,right,baseRight,width,rightTaper]] as const){
   const topSide=edge(upper,front)!,post=edge(front,lower)!;
   const word=Math.abs(taper)<=4?'|':taper>0?'／':'＼';
   const points=this.font.get(word)!.points,headY=Math.min(...points.map(p=>p[1]!)),tailY=Math.max(...points.map(p=>p[1]!));
   const headX=points.filter(p=>p[1]===headY).reduce((sum,p)=>sum+p[0]!,0)/points.filter(p=>p[1]===headY).length;
   const tailX=points.filter(p=>p[1]===tailY).reduce((sum,p)=>sum+p[0]!,0)/points.filter(p=>p[1]===tailY).length;
   const first=x+1+taper-headX,last=x+1-tailX;
   for(let k=1;k<=depthRows;k++){
    // All side-row coordinates derive from the SAME two rail axes. The
    // terminal post retains the shared crease corner; there is no local
    // glyph candidate, pixel score or independent per-edge row phase.
    const text=k===depthRows?'|':word;
    const at=k===depthRows?x:word==='|'?x:Math.round(depthRows>2?first+(last-first)*(k-1)/(depthRows-2):(first+last)/2);
    add(text,at,k,`top/${side}-${k}`,topSide,upper,[(k-1)/depthRows,k/depthRows]);
   }
   for(let k=1;k<=wallRows;k++)add('|',x,frontRow+k,`wall/${side}-${k}`,post,front,[(k-1)/wallRows,k/wallRows]);
   if(!wallRows)add('|',x,frontRow,`wall/${side}-collapsed`,post,front,[0,1]);
  }
  const rows=new Map<number,NativeOccurrence[]>();for(const atom of occurrences){const list=rows.get(atom.row)??[];list.push(atom);rows.set(atom.row,list);}
  if([...rows.values()].some(atoms=>{atoms.sort((a,b)=>a.x-b.x);return atoms.some((atom,i)=>i>0&&!nativeSpacePlan(atom.x-atoms[i-1]!.x-this.font.get(atoms[i-1]!.text)!.advance,0,false,false,'preformatted'));}))
   return fail('coupled plane rails cannot share one native scalar tape');
  const contacts:NativeContact[]=[],continuations:NativeContinuation[]=[];
  for(const port of source.keys()){
   const incident=relations.filter(r=>r.ports.includes(port));
   for(let i=1;i<incident.length;i++){
    const a=incident[0]!,b=incident[i]!;let best:{a:NativeOccurrence;b:NativeOccurrence;p:Point;q:Point;distance:number}|undefined;
    for(const aa of occurrences.filter(o=>o.contributions.some(c=>c.source===a.id)))for(const bb of occurrences.filter(o=>o.contributions.some(c=>c.source===b.id)))
     for(const p of this.font.get(aa.text)!.points)for(const q of this.font.get(bb.text)!.points){
      const pa:Point=[aa.x+p[0]!+.5,aa.row*17+p[1]!+.5],pb:Point=[bb.x+q[0]!+.5,bb.row*17+q[1]!+.5],distance=(pa[0]-pb[0])**2+(pa[1]-pb[1])**2;
      if(!best||distance<best.distance)best={a:aa,b:bb,p:pa,q:pb,distance};
     }
    if(!best)return fail('unrealized plane source incidence');
    if(best.distance<=9)contacts.push({id:`${neighborhood}/plane-slab/contact-${port}-${i}`,from:best.a.id,to:best.b.id,fromSource:a.id,toSource:b.id,fromPort:port,toPort:port,fromAnchor:best.p,toAnchor:best.q,relation:'whole-two-face-plane'});
    else continuations.push({from:best.a.id,to:best.b.id,fromAnchor:best.p,toAnchor:best.q,distance:Math.sqrt(best.distance)+2});
   }
  }
  const faces=[[backLeft,backRight,right,left],[left,right,baseRight,baseLeft]],charts=faces.flatMap(face=>[[0,1,2],[0,2,3]].map(indices=>({
   native:indices.map(i=>native.get(face[i]!)!) as [Point,Point,Point],source:indices.map(i=>source.get(face[i]!)!) as [Point,Point,Point]})));
  return {ok:true,neighborhood,pattern:{id:`${neighborhood}/plane-slab`,origin:[origin,row],occurrences,contacts,continuations,paper:[],requiredSources:relations.map(r=>r.id),
   repeatRuns:['back','front','base'].map(role=>({id:`${neighborhood}/plane-slab/${role}`,members:occurrences.filter(atom=>slices.some(slice=>slice.occurrence===atom.id&&slice.role.startsWith(`${role}/`))).sort((a,b)=>a.x-b.x).map(atom=>atom.id)})),
   edgeCharts:relations.map(relation=>({source:relation.id,native:[native.get(relation.ports[0])!,native.get(relation.ports[1])!] as const})),
   surfaces:[{id:`${neighborhood}/plane-slab/body`,owner,outline:[backLeft,backRight,right,baseRight,baseLeft,left].map(id=>native.get(id)!),charts}]},
   state:{width,origin,row,depthRows,wallRows},slices,fullEdges:relations,sourceVertices:source,nativeVertices:native,regionSockets:[],implied:[],
   production:{status:'provisional-whole-fan-growth',capWitness:'a7065b14a8fb85fd61e0a2e01e91501b/entry157:|￣￣|',returnWitnesses:['a7065b14a8fb85fd61e0a2e01e91501b/entry157:|＿＿|']}};
 }
 private panel(relations:readonly PosedRelation[],neighborhood:string):FanResult{
  const owner=relations[0]!.owner,fail=(obligation:string):FanResult=>({ok:false,owner,obligation}),boundary=planeBoundary(relations);
  if(!boundary)return fail('one-face native production requires a complete convex source plane');
  const byX=[...boundary].sort((a,b)=>a[1][0]-b[1][0]);
  const [tl,bl]=byX.slice(0,2).sort((a,b)=>a[1][1]-b[1][1]).map(([id])=>id) as [string,string];
  const [tr,br]=byX.slice(2).sort((a,b)=>a[1][1]-b[1][1]).map(([id])=>id) as [string,string];
  const edge=(a:string,b:string)=>relations.find(r=>r.ports.includes(a)&&r.ports.includes(b));
  const top=edge(tl,tr),left=edge(tl,bl),right=edge(tr,br),base=edge(bl,br);if(!top||!left||!right||!base)return fail('one-face orientation requires a different complete source function');
  const heightSource=(boundary.get(bl)![1]+boundary.get(br)![1]-boundary.get(tl)![1]-boundary.get(tr)![1])/2;
  const leftSteps=Math.round((boundary.get(bl)![0]-boundary.get(tl)![0])/16),rightSteps=Math.round((boundary.get(br)![0]-boundary.get(tr)![0])/16);
  const count=Math.max(1,Math.round((boundary.get(tr)![0]-boundary.get(tl)![0]-4)/16)),baseCount=count+rightSteps-leftSteps;
  if(baseCount<1)return fail('panel return requires another complete native cap function');
  const width=4+16*count,h=Math.max(1,Math.round((heightSource-15)/17),Math.abs(leftSteps)+1,Math.abs(rightSteps)+1);
  const native=new Map<string,Point>([[tl,[1,0]],[tr,[width+1,0]],[bl,[1+16*leftSteps,h*17+15]],[br,[width+1+16*rightSteps,h*17+15]]]);
  const origin=Math.round([...native].reduce((sum,[id,p])=>sum+boundary.get(id)![0]-p[0],0)/4),row=Math.round([...native].reduce((sum,[id,p])=>sum+boundary.get(id)![1]-p[1],0)/4/17);
  if([...native].some(([id,p])=>Math.hypot(origin+p[0]-boundary.get(id)![0],row*17+p[1]-boundary.get(id)![1])>17))return fail('projected plane needs a complete inclined or perspective panel production');
  const id=`${neighborhood}/plane-panel`,occurrences:NativeOccurrence[]=[],slices:Extract<FanResult,{ok:true}>['slices']=[];
  const postRole=(side:string,r:number)=>`${side}/${r===0?'cap':r===h?'return':`wall-${r}`}`;
  const add=(text:string,x:number,r:number,role:string,relation:PosedRelation,from:string,range:readonly [number,number],select?:(p:readonly number[])=>boolean)=>{
   const glyph=this.font.get(text)!,pixels=glyph.points.flatMap((point,i)=>!select||select(point)?[i]:[]);if(!pixels.length)return;
   const key=`${id}/${role}`,parameter=relationRange(relation,from,range),contribution={source:relation.id,kind:'path' as const,parameter,pixels};
   const prior=occurrences.find(atom=>atom.id===key);
   if(prior)occurrences[occurrences.indexOf(prior)]={...prior,contributions:[...prior.contributions,contribution]};else occurrences.push({id:key,text,x,row:r,contributions:[contribution]});
   slices.push({occurrence:key,source:relation.id,parameter,pixels,role});
  };
  // One paired side-run production distributes WHOLE post/diagonal units
  // between the retained corners. Both rails own taper and height together. No
  // tangent samples choose glyphs or independently reposition row members.
  for(const [side,relation,from,x,steps]of [['left',left,tl,0,leftSteps],['right',right,tr,width,rightSteps]] as const){
   const turns=new Set(Array.from({length:Math.abs(steps)},(_,i)=>1+Math.floor((i+.5)*(h-1)/Math.abs(steps)))),direction=Math.sign(steps);let shift=0;
   for(let r=0;r<=h;r++){
    const turning=r>0&&r<h&&turns.has(r),text=turning?(direction>0?'＼':'／'):'|';
    const at=x+shift+(turning?(direction>0?1:-13):0);
    add(text,at,r,postRole(side,r),relation,from,[r*17/(h*17+15),(r*17+15)/(h*17+15)]);
    if(turning)shift+=16*direction;
   }
  }
  for(const [role,relation,from,r,text,units,start,span]of [['cap',top,tl,0,'￣',count,0,width],['return',base,bl,h,'＿',baseCount,leftSteps*16,width+16*(rightSteps-leftSteps)]] as const){
   const y=role==='cap'?0:15;
   add('|',start,r,postRole('left',r),relation,from,[0,3/span],p=>p[1]===y);
   for(let k=0;k<units;k++)add(text,start+4+16*k,r,`${role}/unit-${k}`,relation,from,[(3+16*k)/span,(18+16*k)/span]);
   add('|',start+span,r,postRole('right',r),relation,from,[1-2/span,1],p=>p[1]===y);
  }
  const contacts:NativeContact[]=[];
  for(const [port,side,r,horizontal]of [[tl,'left',0,top],[tr,'right',0,top],[bl,'left',h,base],[br,'right',h,base]] as const){
   const post=`${id}/${postRole(side,r)}`,point=native.get(port)!,anchor=[point[0]+.5,point[1]+.5] as const;
   contacts.push({id:`${id}/corner/${port}`,from:post,to:post,fromSource:(side==='left'?left:right).id,toSource:horizontal.id,fromPort:port,toPort:port,fromAnchor:anchor,toAnchor:anchor,relation:'whole-one-face-cap-side-return'});
  }
  const continuations:NativeContinuation[]=[];
  const interfacePoint=(atom:NativeOccurrence,end:boolean):Point=>{
   const local:Point=atom.text==='|'?[1.5,end?15.5:.5]:atom.text==='＼'?(end?[14.5,14.5]:[.5,.5]):(end?[.5,14.5]:[14.5,.5]);
   return [atom.x+local[0],atom.row*17+local[1]];
  };
  for(const side of [left,right]){
   const run=occurrences.filter(atom=>atom.contributions.some(c=>c.source===side.id)).sort((a,b)=>a.row-b.row);
   for(let i=1;i<run.length;i++){
    const a=run[i-1]!,b=run[i]!,p=interfacePoint(a,true),q=interfacePoint(b,false),distance=Math.hypot(p[0]-q[0],p[1]-q[1]);
    if(distance>4)return fail('paired plane-side program lost its retained native continuation');
    continuations.push({from:a.id,to:b.id,fromAnchor:p,toAnchor:q,distance});
   }
  }
  const corners=[tl,tr,br,bl],charts=[[0,1,2],[0,2,3]].map(indices=>({native:indices.map(i=>native.get(corners[i]!)!) as [Point,Point,Point],source:indices.map(i=>boundary.get(corners[i]!)!.slice(0,2) as unknown as Point) as [Point,Point,Point]}));
  const source=new Map([...boundary].map(([id,p])=>[id,[p[0],p[1]] as Point]));
  return {ok:true,neighborhood,pattern:{id,origin:[origin,row],occurrences,contacts,continuations,paper:[],requiredSources:relations.map(r=>r.id),
   repeatRuns:['cap','return'].map(role=>({id:`${id}/${role}`,members:occurrences.filter(atom=>atom.id.startsWith(`${id}/${role}/`)).map(atom=>atom.id)})),
   edgeCharts:relations.map(relation=>({source:relation.id,native:[native.get(relation.ports[0])!,native.get(relation.ports[1])!] as const})),
   surfaces:[{id:`${id}/body`,owner,outline:corners.map(port=>native.get(port)!),charts}]},
   state:{width,origin,row,wallRows:h},slices,fullEdges:relations,sourceVertices:source,nativeVertices:native,regionSockets:[],implied:[],
   production:{status:'provisional-whole-fan-growth',capWitness:'a7065b14a8fb85fd61e0a2e01e91501b/entry157:|￣￣|',returnWitnesses:['a7065b14a8fb85fd61e0a2e01e91501b/entry157:|＿＿|']}};
 }
}
