import roles from '../../assets/grammar/aahub-role-lookup-3493185.v1.json' with {type:'json'};
import supplement from './grammar/role-rounds-supplement.json' with {type:'json'};
import {WholeRoleGrowth,type GrowthState,type GrowthTerminal} from './whole-role-growth.ts';
import {selectFeatureGraph,type FeatureWalk} from './feature-graph.ts';
import {relationPoint} from './posed-graph.ts';
import type {SourceNeighborhood} from './source-neighborhoods.ts';
import {SourceNeighborhoods} from './source-neighborhoods.ts';
import type {DrawingFieldView} from './field-readback.ts';
import type {NativeGlyph,NativeOccurrence,NativePaper} from './native-page.ts';
import type {NativePattern} from './native-layout.ts';

interface SourceBlock {
 readonly id:string;readonly native:{readonly rows:readonly string[];readonly row_pen_x:readonly number[]};
}
interface Program {readonly id:string;readonly atoms:readonly {text:string;x:number;row:number;cp:number}[];readonly paper:readonly NativePaper[];readonly bounds:readonly [number,number,number,number]}
export interface CurvePrograms {readonly patterns:readonly NativePattern[];readonly patternOwners:ReadonlyMap<string,readonly string[]>;readonly unresolved:readonly {id:string;reason:string}[];readonly owners:ReadonlySet<string>}
const IDS=new Set(['sh-lib66-circle-2row','sh-lib60-circle-3row','sh-lib64-circle-4row','sh-lib57-circle-8row-whole','rx-lib99-plain-round-7row']);

/** Finite SOURCE blocks, lowered without the old compiler's per-word nudges.
 * They are provisional native size alternatives for one isolated convex round
 * relationship, not a general curvature grammar or runtime subject approval.
 * Unsupported scale, occlusion, contacts and interior structure stay unresolved. */
export class CurveProgramBank {
 private readonly growth:WholeRoleGrowth;private readonly states=new Map<string,GrowthState>();
 private readonly programs:Program[]=[];private readonly font:Map<string,NativeGlyph>;
 private readonly boundaries=new Map<string,{owner:string;history:SourceNeighborhoods}>();
 constructor(glyphs:readonly NativeGlyph[]){
  this.font=new Map(glyphs.map(g=>[g.text,g]));this.growth=new WholeRoleGrowth(glyphs);
  const entries=[...roles.entries,...supplement.entries].filter(e=>IDS.has(e.id)) as SourceBlock[];
  for(const entry of entries){
   const atoms:Program['atoms'][number][]=[],paper:NativePaper[]=[];let minX=Infinity,minY=Infinity,maxX=-Infinity,maxY=-Infinity;
   entry.native.rows.forEach((text,row)=>{
    let x=entry.native.row_pen_x[row]!;let priorInkEnd:number|undefined;
    for(const [cp,ch]of [...text].entries()){
     const glyph=this.font.get(ch);if(!glyph)throw Error(`Source block ${entry.id} has unavailable native scalar ${ch}`);
     if(glyph.points.length){
      const left=x+Math.min(...glyph.points.map(p=>p[0]!)),right=x+Math.max(...glyph.points.map(p=>p[0]!))+1;
      if(priorInkEnd!==undefined&&left-priorInkEnd>=11)paper.push({id:`${entry.id}/paper-${row}-${cp}`,rect:[priorInkEnd,row*17,left-priorInkEnd,17],sources:[entry.id]});
      atoms.push({text:ch,x,row,cp});priorInkEnd=right;
      for(const p of glyph.points){minX=Math.min(minX,x+p[0]!);maxX=Math.max(maxX,x+p[0]!);minY=Math.min(minY,row*17+p[1]!);maxY=Math.max(maxY,row*17+p[1]!);}
     }
     x+=glyph.advance;
    }
   });
   this.programs.push({id:entry.id,atoms,paper,bounds:[minX,minY,maxX,maxY]});
  }
  this.programs.sort((a,b)=>(a.bounds[3]-a.bounds[1])-(b.bounds[3]-b.bounds[1]));
 }
 private surface(pattern:NativePattern,owner:string,source:readonly (readonly number[])[],kind:'surface'|'opening'):NativePattern{
  const points=pattern.occurrences.flatMap(atom=>this.font.get(atom.text)!.points.map(p=>[atom.x+p[0]!,atom.row*17+p[1]!] as const))
   .sort((a,b)=>a[0]-b[0]||a[1]-b[1]);
  const cross=(a:readonly number[],b:readonly number[],c:readonly number[])=>(b[0]!-a[0]!)*(c[1]!-a[1]!)-(b[1]!-a[1]!)*(c[0]!-a[0]!);
  const half=(list:typeof points)=>{const hull:typeof points=[];for(const point of list){while(hull.length>=2&&cross(hull.at(-2)!,hull.at(-1)!,point)<=0)hull.pop();hull.push(point);}return hull;};
  const outline=[...half(points).slice(0,-1),...half([...points].reverse()).slice(0,-1)];
  const x0=Math.min(...points.map(p=>p[0])),x1=Math.max(...points.map(p=>p[0])),y0=Math.min(...points.map(p=>p[1])),y1=Math.max(...points.map(p=>p[1]));
  const a=Math.min(...source.map(p=>p[0]!)),b=Math.max(...source.map(p=>p[0]!)),c=Math.min(...source.map(p=>p[1]!)),d=Math.max(...source.map(p=>p[1]!));
  const native=[[x0,y0],[x1,y0],[x1,y1],[x0,y1]] as const,projected=[[a,c],[b,c],[b,d],[a,d]] as const;
  return {...pattern,surfaces:[{id:`${pattern.id}/body`,owner,kind,outline,charts:[{native:[native[0],native[1],native[2]],source:[projected[0],projected[1],projected[2]]},
   {native:[native[0],native[2],native[3]],source:[projected[0],projected[2],projected[3]]}]}]};
 }
 private opening(walk:FeatureWalk,points:readonly (readonly number[])[],views:readonly DrawingFieldView[]):boolean{
  const area=points.reduce((sum,p,i)=>{const q=points[(i+1)%points.length]!;return sum+p[0]!*q[1]!-q[0]!*p[1]!;},0);
  let filledInside=0,filledOutside=0;
  for(let i=0;i<walk.steps.length;i+=Math.max(1,Math.floor(walk.steps.length/12))){
   const step=walk.steps[i]!,a=relationPoint(step.relation,step.reverse?1:0),b=relationPoint(step.relation,step.reverse?0:1);
   if(!step.relation.intervals.some(part=>part.cause==='visible'&&part.lo<=.5&&part.hi>=.5))continue;
   const dx=b[0]-a[0],dy=b[1]-a[1],length=Math.hypot(dx,dy);if(!length)continue;
   const p=relationPoint(step.relation,.5);
   for(const view of views){
    if(view.camera!==walk.owner.split('/')[0])continue;
    const delta=2*Math.max(view.rect.width/view.field.width,view.rect.height/view.field.height);
    const world=(x:number,y:number,z:number)=>{
     const v=[(x-view.rect.x)/view.rect.width*2-1,1-(y-view.rect.y)/view.rect.height*2,z,1],m=view.field.source.worldFromClip;
     const q=[0,1,2,3].map(row=>v.reduce((sum,value,k)=>sum+value*m[k*4+row]!,0));return q.slice(0,3).map(value=>value/q[3]!);
    };
    const center=world(p[0],p[1],p[2]),next=world(p[0]+delta,p[1],p[2]),vertical=world(p[0],p[1]+delta,p[2]);
    const footprint=Math.max(Math.hypot(...next.map((v,i)=>v-center[i]!)),Math.hypot(...vertical.map((v,i)=>v-center[i]!)));
    if(!Number.isFinite(footprint))continue;
    for(const side of [-1,1]){
     const x=p[0]-dy/length*delta*side,y=p[1]+dx/length*delta*side;
     const sample=view.field.sample((x-view.rect.x)/view.rect.width*view.field.width,(y-view.rect.y)/view.rect.height*view.field.height);if(!sample)continue;
     const reference=sample.kind==='entity'?view.owners.get(sample.owner):sample.owner;
     if(`${view.camera}/${sample.source}/${sample.kind}/${reference}`!==walk.owner)continue;
     const point=world(x,y,sample.depth),distance=Math.abs(sample.normal.reduce((sum,value,i)=>sum+value*(point[i]!-center[i]!),0));
     // A same-owner back surface seen THROUGH an opening cannot masquerade
     // as the local face. The bound is one source-texel geometric footprint.
     if(!Number.isFinite(distance)||distance>footprint)continue;
     if((side>0)===(area>0))filledInside++;else filledOutside++;
    }
   }
  }
  return filledOutside>=3&&filledInside===0;
 }
 retire(owner:string):void{
  for(const key of this.states.keys())if(key.startsWith(`${owner}/`))this.states.delete(key);
  for(const [key,part]of this.boundaries)if(part.owner===owner)this.boundaries.delete(key);
 }
 private apertures(input:readonly NativePattern[]):NativePattern[]{
  let patterns=[...input];
  const bounds=(pattern:NativePattern)=>{
   const points=pattern.surfaces![0]!.charts.flatMap(chart=>chart.source);
   return [Math.min(...points.map(p=>p[0])),Math.min(...points.map(p=>p[1])),Math.max(...points.map(p=>p[0])),Math.max(...points.map(p=>p[1]))] as const;
  };
  for(const opening of input.filter(pattern=>pattern.surfaces?.[0]?.kind==='opening')){
   const hole=bounds(opening),owner=opening.surfaces![0]!.owner;
   const containers=patterns.filter(pattern=>pattern.id!==opening.id&&pattern.surfaces?.[0]?.owner===owner&&pattern.surfaces[0].kind!=='opening')
    .filter(pattern=>{const b=bounds(pattern);return hole[0]>=b[0]&&hole[1]>=b[1]&&hole[2]<=b[2]&&hole[3]<=b[3];})
    .sort((a,b)=>{const x=bounds(a),y=bounds(b);return (x[2]-x[0])*(x[3]-x[1])-(y[2]-y[0])*(y[3]-y[1]);});
   const parent=containers[0];if(!parent)continue;
   const ink=opening.occurrences.flatMap(atom=>this.font.get(atom.text)!.points.map(p=>[opening.origin[0]+atom.x+p[0]!,opening.origin[1]*17+atom.row*17+p[1]!] as const));
   const left=Math.min(...ink.map(p=>p[0]))-parent.origin[0],top=Math.min(...ink.map(p=>p[1]))-parent.origin[1]*17;
   const right=Math.max(...ink.map(p=>p[0]))+1-parent.origin[0],bottom=Math.max(...ink.map(p=>p[1]))+1-parent.origin[1]*17;
   // Parent lit paper cannot prohibit the aperture's boundary or the world
   // visible through it. This cuts a declared relation before publication;
   // no glyph bitmap is erased. Both constructions retain ONE common phase.
   patterns=patterns.map(pattern=>pattern.id===opening.id?{...pattern,paper:[],phaseWith:[...(pattern.phaseWith??[]),parent.id]}:pattern.id!==parent.id?pattern:{...pattern,paper:pattern.paper.flatMap(paper=>{
    const [x,y,w,h]=paper.rect,a=Math.max(x,left),b=Math.max(y,top),c=Math.min(x+w,right),d=Math.min(y+h,bottom);
    if(a>=c||b>=d)return [paper];
    const rects:NativePaper['rect'][]=[[x,y,w,b-y],[x,d,w,y+h-d],[x,b,a-x,d-b],[c,b,x+w-c,d-b]];
    return rects.filter(([, ,width,height])=>width>0&&height>0).map((rect,i)=>({...paper,id:`${paper.id}/aperture-${opening.id}-${i}`,rect}));
   })});
  }
  return patterns;
 }
 compile(neighborhoods:readonly SourceNeighborhood[],terminals:ReadonlyMap<string,GrowthTerminal>=new Map(),views:readonly DrawingFieldView[]=[]):CurvePrograms{
  const patternOwners=new Map<string,readonly string[]>();
  const patterns:NativePattern[]=[],unresolved:{id:string;reason:string}[]=[],owners=new Set<string>();
  for(const {id:neighborhood,owner,graph} of neighborhoods){
   if(!graph.relations.some(relation=>relation.kind==='smooth-contour'))continue;
   for(const relation of graph.relations)if(relation.kind==='crease')unresolved.push({id:relation.id,reason:'internal crease requires a cooperating native feature program'});
   let part=this.boundaries.get(neighborhood);if(!part){part={owner,history:new SourceNeighborhoods()};this.boundaries.set(neighborhood,part);}
   const boundaries=part.history.select([{...graph,relations:graph.relations.filter(relation=>relation.kind!=='crease')}]);
   for(const boundary of boundaries){
   const features=selectFeatureGraph(boundary.graph);
   for(const selectedWalk of features.walks){
    const walk={...selectedWalk,id:boundary.id===owner?`${neighborhood}/closed-form`:`${neighborhood}/boundary/${boundary.id}/closed-form`};
    const paths=[{id:walk.id,relations:walk.steps.map(step=>step.relation.id)}];
    if(!walk.steps.some(step=>step.relation.kind==='smooth-contour'))continue;owners.add(walk.owner);
    const fail=(reason:string)=>unresolved.push({id:walk.id,reason});
    if(!walk.closed){fail('no complete native production for this open curved neighborhood');continue;}
    if(features.junctions.some(junction=>junction.walks.includes(selectedWalk.id)&&junction.walks.length>1)){fail('branching contour requires a joint native junction production');continue;}
    const points=walk.steps.map(step=>relationPoint(step.relation,step.reverse?1:0));
    if(points.some(p=>!p.every(Number.isFinite))){fail('curve crosses a nonprojectable source endpoint');continue;}
    const xs=points.map(p=>p[0]),ys=points.map(p=>p[1]),x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys),width=x1-x0,height=y1-y0;
    if(width<1||height<1){fail('curved source has no projected area');continue;}
    // Whole growth builds one row per 17 px of projected height and one unit
    // per repeat width. A contour projected near the eye plane (Y8 terrain)
    // spans thousands of pages, froze the main thread and crashed the tab
    // (2026-11-08). It is not one whole form on this page.
    const page=views.reduce((most,view)=>[Math.max(most[0]!,view.rect.width),Math.max(most[1]!,view.rect.height)],[0,0]);
    if(page[0]!>0&&(width>4*page[0]!||height>4*page[1]!)){fail(`projected curve ${Math.round(width)}x${Math.round(height)} px exceeds four pages; not one whole form on this page`);continue;}
    const kind=this.opening(walk,points,views)?'opening':'surface';
    const cx=(x0+x1)/2,cy=(y0+y1)/2,rx=width/2,ry=height/2;
    const ellipseCompatible=!points.some(p=>Math.abs(Math.hypot((p[0]-cx)/rx,(p[1]-cy)/ry)-1)>.18);
    // The full feature, including hidden intervals, owns the role sequence.
    // The whole source function grows in native repeat/paper slots; it never
    // chooses each row's glyph from its tangent or fits a font to an image.
    const terminal=terminals.get(walk.id),stateKey=`${walk.id}/${terminal?'rail-return':'closed'}`,grown=this.growth.grow(walk,this.states.get(stateKey),terminal);
    if(grown.ok){
     const rail=terminal?.rail,edgeCharts=rail?[{source:rail.edge.id,native:(rail.reverse?[[rail.end,terminal.row*17+15],[rail.start,terminal.row*17+15]]:[[rail.start,terminal.row*17+15],[rail.end,terminal.row*17+15]]).map(p=>[p[0]!-grown.pattern.origin[0],p[1]!-grown.pattern.origin[1]*17]) as unknown as readonly [readonly [number,number],readonly [number,number]]}]:undefined;
     this.states.set(stateKey,grown.state);patterns.push(this.surface({...grown.pattern,paths,edgeCharts},walk.owner,points,kind));patternOwners.set(walk.id,rail?[walk.owner,rail.edge.owner]:[walk.owner]);continue;
    }
    if(terminal){fail(grown.obligation);continue;}
    if(!ellipseCompatible){fail('non-round curvature requires a complete source-role growth function');continue;}
    // Finite intact source profiles remain alternatives until the growing
    // function covers their contexts. They are not per-piece fallbacks.
    // One direct height lookup chooses the complete source spelling. We do
    // not choose each shoulder, side or return from tangent glyph candidates.
    let program=this.programs[0]!;
    for(let i=1;i<this.programs.length;i++){
     const previous=this.programs[i-1]!,next=this.programs[i]!;
     if(height>=((previous.bounds[3]-previous.bounds[1])+(next.bounds[3]-next.bounds[1]))/2)program=next;
    }
    const [a,b,c,d]=program.bounds;
    if(Math.abs((c-a)-width)>17||Math.abs((d-b)-height)>17){fail('projected size exceeds the finite source-profile coverage; no frozen-size fallback');continue;}
    const occurrences:NativeOccurrence[]=program.atoms.map(atom=>({id:`${walk.id}/${program.id}/row-${atom.row}/cp-${atom.cp}`,text:atom.text,x:atom.x,row:atom.row,
      contributions:[{source:walk.id,kind:'path',pixels:this.font.get(atom.text)!.points.map((_,index)=>index)}]}));
    patternOwners.set(walk.id,[walk.owner]);
    patterns.push(this.surface({id:walk.id,origin:[Math.round(cx-(a+c)/2),Math.round((cy-(b+d)/2)/17)],occurrences,contacts:[],
      paper:program.paper.map(paper=>({...paper,sources:[walk.id]})),requiredSources:[walk.id],paths},walk.owner,points,kind));
   }
   }
  }
  return {patterns:this.apertures(patterns),unresolved,owners,patternOwners};
 }
}
