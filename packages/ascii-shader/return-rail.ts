import {relationPoint,relationParameter,type PosedGraph,type PosedRelation} from './posed-graph.ts';
import type {CurvePrograms} from './curve-programs.ts';
import {PlaneCapFan,type FanResult} from './plane-cap-fan.ts';
import type {GrowthTerminal} from './whole-role-growth.ts';
import type {DrawingFieldView} from './field-readback.ts';
import type {NativeGlyph,NativeOccurrence,NativeContact} from './native-page.ts';
import type {NativePattern} from './native-layout.ts';

type Plane=Extract<FanResult,{ok:true}>;
type Point=readonly [number,number];
type Box=readonly [number,number,number,number];
interface CurveContext {pattern:NativePattern;owner:string;order:number;box:Box;edges:readonly {a:readonly number[];b:readonly number[]}[]}
interface CurveNode {box:Box;items?:readonly CurveContext[];left?:CurveNode;right?:CurveNode}
function curveTree(items:CurveContext[]):CurveNode|undefined{
 if(!items.length)return;
 const box:[number,number,number,number]=[Infinity,Infinity,-Infinity,-Infinity];
 for(const item of items){box[0]=Math.min(box[0],item.box[0]);box[1]=Math.min(box[1],item.box[1]);box[2]=Math.max(box[2],item.box[2]);box[3]=Math.max(box[3],item.box[3]);}
 if(items.length<=8)return {box,items};
 const axis=box[2]-box[0]>=box[3]-box[1]?0:1;
 items.sort((a,b)=>a.box[axis]!+a.box[axis+2]!-b.box[axis]!-b.box[axis+2]!||a.order-b.order);
 const middle=items.length>>>1;return {box,left:curveTree(items.slice(0,middle)),right:curveTree(items.slice(middle))};
}
function intersectingCurves(node:CurveNode|undefined,box:Box,result:CurveContext[]):void{
 if(!node||node.box[0]>box[2]||node.box[2]<box[0]||node.box[1]>box[3]||node.box[3]<box[1])return;
 if(node.items){for(const item of node.items)result.push(item);return;}
 intersectingCurves(node.left,box,result);intersectingCurves(node.right,box,result);
}
export interface ReturnRailPlan {readonly planes:readonly Plane[];readonly terminals:ReadonlyMap<string,GrowthTerminal>}

/** Compose a lower curved return with an occluding horizontal source edge.
 * The foreground return is the WHOLE ゝ __ノ role from lib66; the external
 * rail intervals follow the contiguous-return/rail construction witnessed in
 * sh-mlt03-compact-circle. Combining/growing those functions is provisional.
 * All axes and repeat counts are chosen before either scalar tape is emitted.
 * No mesh name, material recipe, scene switch, or glyph/image fit is involved.
 */
export function returnRailPlans(planes:readonly Plane[],curves:CurvePrograms,graphs:readonly PosedGraph[],views:readonly DrawingFieldView[],factory:PlaneCapFan,font:readonly NativeGlyph[]):ReturnRailPlan{
 const glyphs=new Map(font.map(glyph=>[glyph.text,glyph])),terminals=new Map<string,GrowthTerminal>(),result:Plane[]=[];
 const leftWidth=glyphs.get('ゝ')!.advance,gap=glyphs.get(' ')!.advance,rightWidth=glyphs.get('ノ')!.advance,unit=glyphs.get('_')!.advance,longUnit=glyphs.get('＿')!.advance;
 const base=leftWidth+gap+rightWidth,portSpan=leftWidth+gap+9-2;
 if(!planes.length||!curves.patterns.length)return {planes:[...planes],terminals};
 // Source addressing and projected geometry are shared by all rail queries
 // in this pose. The old nested plane × curve loop filtered the COMPLETE
 // world relation list for every pair, blocking city/Y8 before publication.
 // These indices change neither the retained return nor its join decision.
 const bySource=new Map<string,{relation:PosedRelation;order:number}[]>();let order=0;
 for(const graph of graphs)for(const relation of graph.relations){const id=relation.id,list=bySource.get(id)??[];list.push({relation,order:order++});bySource.set(id,list);}
 const bounded:CurveContext[]=[],unbounded:CurveContext[]=[];
 for(const [order,pattern]of curves.patterns.entries()){
  if(pattern.surfaces?.every(surface=>surface.kind==='opening'))continue;
  const owner=curves.patternOwners.get(pattern.id)?.[0];if(!owner)continue;
  const ids=new Set(pattern.paths?.flatMap(path=>path.relations));
  const source=[...ids].flatMap(id=>bySource.get(id)??[]).filter(item=>item.relation.kind==='smooth-contour').sort((a,b)=>a.order-b.order);
  if(!source.length)continue;
  const edges=source.map(({relation})=>({a:relationPoint(relation,0),b:relationPoint(relation,1)}));
  const box:[number,number,number,number]=[Infinity,Infinity,-Infinity,-Infinity];
  for(const edge of edges)for(const p of [edge.a,edge.b]){box[0]=Math.min(box[0],p[0]!);box[1]=Math.min(box[1],p[1]!);box[2]=Math.max(box[2],p[0]!);box[3]=Math.max(box[3],p[1]!);}
  const context={pattern,owner,order,edges,box};(box.every(Number.isFinite)?bounded:unbounded).push(context);
 }
 const tree=curveTree(bounded);
 const nativeRail=(plane:Plane,rail:PosedRelation)=>{
  const ids=new Set(plane.slices.filter(slice=>slice.source===rail.id&&slice.role.startsWith('back/')).map(slice=>slice.occurrence));
  const atoms=plane.pattern.occurrences.filter(atom=>ids.has(atom.id));if(!atoms.length)return;
  const start=plane.pattern.origin[0]+Math.min(...atoms.map(atom=>atom.x)),end=plane.pattern.origin[0]+Math.max(...atoms.map(atom=>atom.x+glyphs.get(atom.text)!.advance));
  const a=plane.nativeVertices.get(rail.ports[0]),b=plane.nativeVertices.get(rail.ports[1]);if(!a||!b)return;
  const sourceStart=plane.pattern.origin[0]+Math.min(a[0],b[0]),sourceEnd=plane.pattern.origin[0]+Math.max(a[0],b[0]);
  return {start,end,sourceStart,sourceEnd};
 };
 for(const original of planes){
  if(original.state.context==='inclined'){result.push(original);continue;}
  const source=original.slices.find(slice=>slice.role.startsWith('back/'))?.source,rail=original.fullEdges.find(edge=>edge.id===source);
  if(!rail){result.push(original);continue;}
  const ra=relationPoint(rail,0),rb=relationPoint(rail,1),dx=rb[0]-ra[0],dy=rb[1]-ra[1];
  if(Math.abs(dy)>8){result.push(original);continue;}
  const contexts:{id:string;left:number;right:number;bottom:number;transparent:boolean}[]=[];
  const candidates=[...unbounded];
  intersectingCurves(tree,[Math.min(ra[0],rb[0]),Math.min(ra[1],rb[1]),Math.max(ra[0],rb[0]),Math.max(ra[1],rb[1])],candidates);
  candidates.sort((a,b)=>a.order-b.order);
  for(const {pattern,owner,edges,box}of candidates){
   if(terminals.has(pattern.id))continue;
   const cuts:number[]=[];
   for(const {a,b}of edges){
    const ex=b[0]!-a[0]!,ey=b[1]!-a[1]!,det=dx*ey-dy*ex;
    if(Math.abs(det)<1e-9)continue;
    const qx=a[0]-ra[0],qy=a[1]-ra[1],t=(qx*ey-qy*ex)/det,u=(qx*dy-qy*dx)/det;
    if(t>=0&&t<=1&&u>=0&&u<=1&&!cuts.some(prior=>Math.abs(prior-t)<1e-5))cuts.push(t);
   }
   if(cuts.length!==2)continue;cuts.sort((a,b)=>a-b);
   const y=ra[1]+dy*(cuts[0]!+cuts[1]!)/2,lo=box[1],hi=box[3];
   if(y<(lo+hi)/2||hi-y>17||original.state.row-pattern.origin[1]<1)continue;
   const middle=(cuts[0]!+cuts[1]!)/2,p=[ra[0]+dx*middle,ra[1]+dy*middle,ra[2]+(rb[2]-ra[2])*middle];let transparent=false;
   const covered=views.some(view=>{
     if(view.camera!==owner.split('/')[0])return false;
     const sample=view.field.sample((p[0]-view.rect.x)/view.rect.width*view.field.width,(p[1]-view.rect.y)/view.rect.height*view.field.height);
     if(!sample||sample.radiance[3]<=.01||sample.depth<=p[2]!+1e-7)return false;

     const reference=sample.kind==='entity'?view.owners.get(sample.owner):sample.owner;
     if(`${view.camera}/${sample.source}/${sample.kind}/${reference}`!==owner)return false;
     transparent=sample.radiance[3]<.99;return true;
   });
   if(covered){const xs=cuts.map(t=>ra[0]+dx*t).sort((a,b)=>a-b);contexts.push({id:pattern.id,left:xs[0]!,right:xs[1]!,bottom:hi,transparent});}
  }
  if(contexts.length!==1){result.push(original);continue;}
  const curve=contexts[0]!,floor=Math.max(1,Math.round((curve.right-curve.left-portSpan)/unit));
  let best:{plane:Plane;leftShort:number;leftLong:number;rightShort:number;rightLong:number;J:number;cost:number}|undefined;
  for(let width=original.state.width-8;width<=original.state.width+8;width++){
   const grown=factory.grow(original.fullEdges,original.state,{width,origin:original.state.origin,row:Math.round(((ra[1]+rb[1])/2-15)/17)},original.neighborhood);if(!grown.ok)continue;
   // Compose the PRODUCED scalar tape, not a guessed width+constant. A
   // two-face passage and a three-face cap have different native endpoints.
   const run=nativeRail(grown,rail);if(!run)continue;
   const span=run.end-run.start,long=((span-base)%unit+unit)%unit,short=(span-base-longUnit*long)/unit;
   if(!Number.isInteger(short)||short<floor)continue;
   const A=run.start,target=(curve.left-2+curve.right-leftWidth-gap-9-unit*floor)/2;
   for(let leftLong=0;leftLong<=long;leftLong++){
    const leftShort=Math.max(0,Math.min(short-floor,Math.round((target-A-longUnit*leftLong)/unit))),J=A+unit*leftShort+longUnit*leftLong;
    const cost=(width-original.state.width)**2+(J+2-curve.left)**2+(J+leftWidth+gap+unit*floor+9-curve.right)**2;
    if(!best||cost<best.cost)best={plane:grown,leftShort,leftLong,rightShort:short-floor-leftShort,rightLong:long-leftLong,J,cost};
   }
  }
  if(!best){result.push(original);continue;}
  const {plane,J}=best,run=nativeRail(plane,rail)!;const A=run.start;
  if(Math.abs(J+2-curve.left)>8||Math.abs(J+leftWidth+gap+unit*floor+9-curve.right)>8||Math.abs(plane.state.row*17+15-curve.bottom)>17){result.push(original);continue;}
  const oldBack=new Set(plane.slices.filter(slice=>slice.source===rail.id&&slice.role.startsWith('back/')).map(slice=>slice.occurrence));
  const retained=plane.pattern.occurrences.filter(atom=>!oldBack.has(atom.id)),railAtoms:NativeOccurrence[]=[];
  const u=(x:number)=>Math.max(0,Math.min(1,(x-run.sourceStart)/(run.sourceEnd-run.sourceStart))),reverse=ra[0]>rb[0];
  const repeat=(text:string,count:number,start:number,role:string)=>{
   let x=start;for(let i=0;i<count;i++){
    const glyph=glyphs.get(text)!,x0=x+Math.min(...glyph.points.map(p=>p[0]!)),x1=x+Math.max(...glyph.points.map(p=>p[0]!)),range=[u(x0),u(x1)] as const;
    railAtoms.push({id:`${plane.pattern.id}/shared-return/${role}-${i}`,text,x:x-plane.state.origin,row:0,
     contributions:[{source:rail.id,kind:'path',pixels:glyph.points.map((_,pixel)=>pixel),parameter:(reverse?[1-range[1],1-range[0]]:range).map(s=>relationParameter(rail,s)) as unknown as readonly [number,number]}]});x+=glyph.advance;
   }return x;
  };
  let pen=repeat('_',best.leftShort,A,'left-short');pen=repeat('＿',best.leftLong,pen,'left-long');
  pen=J+base+unit*floor;pen=repeat('＿',best.rightLong,pen,'right-long');repeat('_',best.rightShort,pen,'right-short');
  const occurrences=[...retained,...railAtoms],contacts=plane.pattern.contacts.filter(contact=>!oldBack.has(contact.from)&&!oldBack.has(contact.to)),
   continuations=(plane.pattern.continuations??[]).filter(link=>!oldBack.has(link.from)&&!oldBack.has(link.to));
  // The physical source corners survive the run rewrite. Associate their
  // fixed native interfaces; this never selects a replacement glyph.
  for(const port of rail.ports)for(const adjacent of plane.fullEdges.filter(edge=>edge!==rail&&edge.ports.includes(port))){
   const from=occurrences.filter(atom=>atom.contributions.some(c=>c.source===rail.id)),to=occurrences.filter(atom=>atom.contributions.some(c=>c.source===adjacent.id));
   let closest:{a:NativeOccurrence;b:NativeOccurrence;p:Point;q:Point;distance:number}|undefined;
   for(const a of from)for(const b of to)for(const p of glyphs.get(a.text)!.points)for(const q of glyphs.get(b.text)!.points){
    const pa:Point=[a.x+p[0]!+.5,a.row*17+p[1]!+.5],pb:Point=[b.x+q[0]!+.5,b.row*17+q[1]!+.5],distance=(pa[0]-pb[0])**2+(pa[1]-pb[1])**2;
    if(!closest||distance<closest.distance)closest={a,b,p:pa,q:pb,distance};
   }
   if(closest){if(closest.distance<=9)contacts.push({id:`${plane.pattern.id}/shared-return/corner-${port}/${adjacent.id}`,from:closest.a.id,to:closest.b.id,
    fromSource:rail.id,toSource:adjacent.id,fromPort:port,toPort:port,fromAnchor:closest.p,toAnchor:closest.q,relation:'native-source-corner'} satisfies NativeContact);
    else continuations.push({from:closest.a.id,to:closest.b.id,fromAnchor:closest.p,toAnchor:closest.q,distance:Math.sqrt(closest.distance)+2});}
  }
  result.push({...plane,pattern:{...plane.pattern,occurrences,contacts,continuations,after:[...(plane.pattern.after??[]),curve.id],phaseWith:[...(plane.pattern.phaseWith??[]),curve.id]}});
  terminals.set(curve.id,{row:plane.state.row,left:J,floor,...(curve.transparent?{rail:{edge:rail,start:run.sourceStart,end:run.sourceEnd,reverse}}:{})});
 }
 return {planes:result,terminals};
}
