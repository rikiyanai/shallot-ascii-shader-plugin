import {selectFeatureGraph} from './feature-graph.ts';
import {relationPoint} from './posed-graph.ts';
import type {CompositionSubject} from './composition.ts';
import type {NativeGlyph,NativeOccurrence,NativePaper} from './native-page.ts';
import type {NativePattern} from './native-layout.ts';
import {nativeSpacePlan} from './text/native-whitespace.ts';

type Point=readonly [number,number];
type Result={ok:true;pattern:NativePattern;configuration:string}|{ok:false;reason:string};

/** A complete convex round's cap, two turning sides and return. The source
 * cycle supplies the shape and the four shared turning anchors; visibility
 * never restarts a side. This is the independently authored smoothClosed
 * realization from r14, NOT a retained library entry or a fitted glyph image.
 * Small retained circles remain the incumbent in NativeConstruction.
 */
export class RoundConstruction {
 private readonly font:Map<string,NativeGlyph>;
 constructor(glyphs:readonly NativeGlyph[]){this.font=new Map(glyphs.map(g=>[g.text,g]));}
 construct(subject:CompositionSubject):Result{
  const fail=(reason:string):Result=>({ok:false,reason});
  if(subject.openings.length)return fail('round opening needs its complete rim/inside neighborhood');
  const walks=selectFeatureGraph({ports:new Map(),relations:subject.relations.filter(r=>r.kind!=='crease')}).walks;
  if(walks.length!==1||!walks[0]!.closed)return fail('round requires a complete closed source cycle');
  const walk=walks[0]!,source=walk.steps.map(step=>relationPoint(step.relation,step.reverse?1:0));
  if(source.some(p=>!p.every(Number.isFinite)))return fail('round has a nonprojectable source port');
  const [x0,y0,x1,y1]=subject.bounds,h=Math.round((y1-y0)/17),top=Math.round(y0/17);
  if(h<4)return fail('compact round needs its whole native size-band construction');
  const id=`${subject.id}/smooth-closed`,path=`${id}/boundary`;
  const atoms:NativeOccurrence[]=[],paper:NativePaper[]=[],parts=new Map<string,string[]>();
  // Intersections observe the COMPLETE source cycle, not its current visible
  // fragments. They only locate an already selected whole side function.
  const section=(y:number):readonly [number,number]=>{
   const xs:number[]=[];
   for(let i=0;i<source.length;i++){
    const a=source[i]!,b=source[(i+1)%source.length]!;
    if((a[1]<=y&&b[1]>y)||(b[1]<=y&&a[1]>y))xs.push(a[0]+(y-a[1])*(b[0]-a[0])/(b[1]-a[1]));
   }
   if(xs.length<2)return [(x0+x1)/2,(x0+x1)/2];
   let lo=Infinity,hi=-Infinity;for(const x of xs){lo=Math.min(lo,x);hi=Math.max(hi,x);}return [lo,hi];
  };
  const nativeY=(row:number,within:number)=>y0+(17*row+within)/(17*h)*(y1-y0);
  const add=(text:string,x:number,row:number,part:string,key:string)=>{
   const members=parts.get(part)??[];let pen=x;
   for(const [scalar,ch]of [...text].entries()){
    const glyph=this.font.get(ch);if(!glyph)throw Error(`Round program lacks native ${ch}`);
    if(glyph.points.length){const occurrence={id:`${id}/${part}/${key}/${scalar}`,text:ch,x:pen,row:top+row,
     contributions:[{source:path,kind:'path' as const,pixels:glyph.points.map((_,i)=>i)}]};atoms.push(occurrence);members.push(occurrence.id);}
    pen+=glyph.advance;
   }
   parts.set(part,members);return pen;
  };
  const legalPen=(wanted:number)=>{
   const at=Math.round(wanted);
   // D1 nearest legal pen for this WHOLE selected side unit. This is the
   // native whitespace language, not glyph/angle/pixel candidate ranking.
   for(let d=0;d<=16;d++)for(const x of d?[at-d,at+d]:[at])if(nativeSpacePlan(x,0,true,false,'preformatted'))return x;
   return undefined;
  };
  const pair=(left:string,right:string,row:number,part:string)=>{
   const [lo,hi]=section(nativeY(row,8));
   const lp=legalPen(lo-3),wanted=Math.round(hi-3),advance=[...left].reduce((n,ch)=>n+this.font.get(ch)!.advance,0);
   if(lp===undefined)return false;
   let rp:number|undefined;
   for(let d=0;d<=16&&rp===undefined;d++)for(const value of d?[wanted-d,wanted+d]:[wanted])
    if(nativeSpacePlan(value-lp-advance,0,false,false,'preformatted')){rp=value;break;}
   if(rp===undefined)return false;
   add(left,lp,row,`${part}/left`,String(row));add(right,rp,row,`${part}/right`,String(row));
   const leftInkEnd=lp+[...left].reduce((n,ch)=>n+this.font.get(ch)!.advance,0),rightInkStart=rp;
   if(rightInkStart-leftInkEnd>10)paper.push({id:`${id}/inside/${row}`,rect:[leftInkEnd+3,(top+row)*17,rightInkStart-leftInkEnd-6,17],sources:[path]});
   return true;
  };
  // Both cap rows and both return rows are authored TOGETHER. Width changes
  // insert whole horizontal units between their enduring shoulder anchors.
  const crown=(row:number,text:string,within:number,part:string)=>{
   const [lo,hi]=section(nativeY(row,within)),unit=this.font.get(text)!.advance;
   const count=Math.max(1,Math.round((hi-lo)/unit)),pen=legalPen((lo+hi-count*unit)/2);
   if(pen===undefined)return false;add(text.repeat(count),pen,row,part,'span');return true;
  };
  if(!crown(0,'＿',15,'cap')||!pair('／','＼',1,'cap-shoulders'))return fail('complete round cap needs its next legal width variant');
  const equator=Math.max(2,Math.min(h-2,Math.round(h/2)));
  // Choose these complete units ONCE per source run. Their placement shift
  // follows source curvature; no sample chooses a scalar or a tangent family.
  for(let row=2;row<h-2;row++)if(!pair(row<equator?'ノ':'ヽ、',row<equator?'ヽ、':'ノ',row,row<equator?'upper-flanks':'lower-flanks'))
   return fail('complete round sides need their next legal width variant');
  if(!pair('＼','／',h-2,'return-shoulders')||!crown(h-1,'￣',1,'return'))return fail('complete round return needs its next legal width variant');
  const native=source.map(p=>[p[0],top*17+(p[1]-y0)/(y1-y0)*h*17] as Point);
  // The shape is the source cycle, not this chart rectangle. Its native ↔
  // source map is affine, so TWO triangles suffice regardless of mesh detail.
  // A fan per mesh vertex repeated the same map for every protected pixel.
  const n=[[x0,top*17],[x1,top*17],[x1,(top+h)*17],[x0,(top+h)*17]] as const;
  const q=[[x0,y0],[x1,y0],[x1,y1],[x0,y1]] as const;
  const surfaces=[{id:`${id}/body`,owner:subject.owner,outline:native,charts:[[0,1,2],[0,2,3]].map(t=>({native:[n[t[0]!]!,n[t[1]!]!,n[t[2]!]!] as const,source:[q[t[0]!]!,q[t[1]!]!,q[t[2]!]!] as const}))}];
  return {ok:true,configuration:`independent/smoothClosed/${h}`,pattern:{id,origin:[0,0],occurrences:atoms,contacts:[],paper,requiredSources:[path],surfaces,
   paths:[{id:path,relations:walk.steps.map(step=>step.relation.id)}],repeatRuns:[...parts].map(([part,members])=>({id:`${id}/${part}`,members}))}};
 }
}
