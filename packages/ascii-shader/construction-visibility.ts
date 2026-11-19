import {relationPoint,relationParameter,type PosedGraph,type PosedRelation} from './posed-graph.ts';
import type {DrawingFieldView} from './field-readback.ts';
import type {NativePattern} from './native-layout.ts';
import {sourceSurfacePoint as chart,NativeSurfaceVisibility} from './native-surface.ts';
import type {NativeGlyph,NativePaper,NativeOccurrence} from './native-page.ts';
import {SourceSupportIndex,type SourceSupport as Support} from './source-support.ts';

type Point=readonly [number,number];
function sourcePoint(pattern:NativePattern,x:number,y:number):Point|undefined{
 for(const surface of pattern.surfaces??[]){const point=chart([x-pattern.origin[0],y-pattern.origin[1]*17],surface);if(point)return point;}
}

/** Visibility is applied to an ALREADY complete native source function. It
 * never restarts its row enumeration, chooses another glyph, clips a mask or
 * changes its native phase. D1 retains an intact scalar at a visibility cut
 * when most of its ink is visible. A minority-visible unit leaves its visible
 * source span for the shared terminal run, never a cropped display mask.
 */
export class ConstructionVisibility {
 private readonly font:Map<string,NativeGlyph>;
 private readonly support=new SourceSupportIndex();
 private readonly frames=new WeakMap<readonly PosedGraph[],{relations:ReadonlyMap<string,PosedRelation>;byOwner:ReadonlyMap<string,readonly PosedRelation[]>}>();
 constructor(font:readonly NativeGlyph[]){this.font=new Map(font.map(glyph=>[glyph.text,glyph]));}
 select(patterns:readonly NativePattern[],owners:ReadonlyMap<string,readonly string[]>,graphs:readonly PosedGraph[],views:readonly DrawingFieldView[],width:number,height:number,surfaces=new NativeSurfaceVisibility(patterns,views,[...this.font.values()],graphs)):{
  patterns:NativePattern[];unresolved:string[];couplings:readonly (readonly [string,string])[];
  terminalPatterns:readonly NativePattern[];
  terminalVisibility:ReadonlyMap<string,(pen:number)=>boolean>;
 }{
  // Graphs are immutable completed GPU snapshots. All producers in this
  // NativeConstruction frame share that same snapshot. Re-indexing the whole
  // world for each one-part call made restored coverage quadratic in scene
  // size (City's live stack stopped here). The index dies with its snapshot;
  // no previous pose, visibility interval or owner lifetime can leak forward.
  let frame=this.frames.get(graphs);
  if(!frame){const relations=new Map<string,PosedRelation>(),byOwner=new Map<string,PosedRelation[]>();
   for(const graph of graphs)for(const relation of graph.relations){relations.set(relation.id,relation);const list=byOwner.get(relation.owner)??[];list.push(relation);byOwner.set(relation.owner,list);}
   frame={relations,byOwner};this.frames.set(graphs,frame);
  }
  const {relations,byOwner}=frame;
  const unresolved:string[]=[],visible:NativePattern[]=[],terminalPatterns:NativePattern[]=[],terminalVisibility=new Map<string,(pen:number)=>boolean>(),couplings=new Map<string,readonly [string,string]>();
  const paths=new Map(patterns.flatMap(pattern=>(pattern.paths??[]).map(path=>[path.id,path.relations.flatMap(id=>relations.has(id)?[relations.get(id)!]:[])] as const)));
  const coupled=(a:string,b:string)=>{if(a!==b)couplings.set(`${a}\n${b}`,[a,b]);};
  const blocked=(x:number,y:number,support:Support,patternID:string)=>{
   const foreground=surfaces.occluder(x,y,support.relation.owner,support.depth,support.relation.id);
   if(foreground!==undefined){coupled(patternID,foreground);return true;}
   // Original-scene visibility is already on the COMPLETE source interval.
   // Re-testing a native pen against the camera image would attach geometry
   // to font pixels and let an unrelated ground sample erase a displaced
   // native return. Only constructed foreground bodies occlude HERE.
   return false;
  };
  for(const pattern of patterns){
   const ownerIDs=owners.get(pattern.id)??[],selected=new Set(pattern.paths?.flatMap(path=>path.relations));
   const whole=ownerIDs.flatMap(owner=>byOwner.get(owner)??[]).filter(relation=>!pattern.paths||selected.has(relation.id)),smooth=whole.filter(relation=>relation.kind==='smooth-contour');
   const cutOccurrences:NativeOccurrence[]=[];
   const occurrences=pattern.occurrences.flatMap((atom):NativeOccurrence[]=>{
    const glyph=this.font.get(atom.text)!,y=(pattern.origin[1]+atom.row)*17;
    // The terminal owner may move this SAME intact scalar along its source
    // run. Visibility is a legality condition, not a new glyph fit/score.
    const classify=(x:number)=>{
    const coverage=new Set<number>(),contributions:NativeOccurrence['contributions'][number][]=[];
    for(const contribution of atom.contributions){
     const exact=relations.get(contribution.source),support=exact?[exact]:paths.get(contribution.source)??(contribution.kind==='region'?whole:smooth);
     if(!support.length)continue;
     const pixels:number[]=[];
     for(const pixel of contribution.pixels){
      const p=glyph.points[pixel]!,px=x+p[0]!+.5,py=y+p[1]!+.5;
      if(px<0||py<0||px>=width||py>=height)continue;
      const mapped=sourcePoint(pattern,px,py);
      let at=this.support.nearest(mapped?.[0]??px,mapped?.[1]??py,support);
      if(at&&exact&&contribution.parameter){
       // The source-u range belongs to this intact native unit. Its current
       // visible/hidden decision must not depend on which ORIGINAL screen
       // edge happens to be closest to a quantized native pen.
       const [lo,hi]=contribution.parameter,edge=pattern.edgeCharts?.find(chart=>chart.source===exact.id);
       let parameter=Math.max(lo,Math.min(hi,at.parameter));
       if(edge){
        const [a,b]=edge.native,dx=b[0]-a[0],dy=b[1]-a[1],length=dx*dx+dy*dy;
        const s=length?Math.max(0,Math.min(1,((px-pattern.origin[0]-a[0])*dx+(py-pattern.origin[1]*17-a[1])*dy)/length)):0;
        parameter=Math.max(lo,Math.min(hi,relationParameter(exact,s)));
       }
       const point=relationPoint(exact,parameter);
       at={...at,parameter,depth:point[2]};
      }
      if(at){
       const foreground=surfaces.cursorOccluder(x,x+glyph.advance,pattern.origin[1]+atom.row,at.relation.owner,at.depth,at.relation.id);
       if(foreground!==undefined){coupled(pattern.id,foreground);continue;}
      }
      if(at&&contribution.kind==='region'){
       let receiver=false;
       for(const view of views){
        if(view.camera!==at.relation.owner.split('/')[0])continue;
        const sample=view.field.sample(((mapped?.[0]??px)-view.rect.x)/view.rect.width*view.field.width,((mapped?.[1]??py)-view.rect.y)/view.rect.height*view.field.height);
        if(!sample)continue;
        const reference=sample.kind==='entity'?view.owners.get(sample.owner):sample.owner;
        if(`${view.camera}/${sample.source}/${sample.kind}/${reference}`===at.relation.owner){at={...at,depth:sample.depth};receiver=true;break;}
       }
       if(!receiver)continue;
      }
      const interval=at?.relation.intervals.find(part=>at.parameter>=part.lo-1e-6&&at.parameter<=part.hi+1e-6);
      if((contribution.kind==='region'||interval?.cause==='visible')&&at&&!blocked(px,py,at,pattern.id)){pixels.push(pixel);coverage.add(pixel);}
     }
     if(pixels.length)contributions.push({...contribution,pixels});
    }
    return {coverage,contributions};
    };
    const {coverage,contributions}=classify(pattern.origin[0]+atom.x);
    // D1 keeps the INTACT scalar when most of its ink belongs to visible
    // construction. Do not erase a word because a few native ink pixels
    // cross a geometric cut. Those leftover strokes belong to the surviving
    // depicted part, never to a hidden source or to a cropped display mask.
    if(!coverage.size)return [];
    if(coverage.size*2<=glyph.points.length){
     // A minority-visible span is a pending terminal request, NOT a lost
     // occurrence. The run's existing scalar is retried at legal pens; the
     // full glyph can survive only where its majority is actually visible.
     terminalVisibility.set(atom.id,pen=>classify(pen).coverage.size*2>glyph.points.length);
     cutOccurrences.push(atom);
     unresolved.push(`${atom.id}:visible source span needs its terminal run at the cut`);return [];
    }
    if(coverage.size!==glyph.points.length){
     let donor=0;for(let i=1;i<contributions.length;i++)if(contributions[i]!.pixels.length>contributions[donor]!.pixels.length)donor=i;
     const missing=glyph.points.flatMap((_,i)=>coverage.has(i)?[]:[i]);
     contributions[donor]={...contributions[donor]!,pixels:[...contributions[donor]!.pixels,...missing]};
    }
    return [{...atom,contributions}];
   });
   if(cutOccurrences.length){
    const candidates=new Map([...occurrences,...cutOccurrences].map(atom=>[atom.id,atom]));
    const full=pattern.occurrences.flatMap(atom=>candidates.has(atom.id)?[candidates.get(atom.id)!]:[]);
    const ids=new Set(full.map(atom=>atom.id)),covered=new Set(full.flatMap(atom=>atom.contributions.map(c=>c.source)));
    terminalPatterns.push({...pattern,occurrences:full,requiredSources:pattern.requiredSources.filter(source=>covered.has(source)),
     contacts:pattern.contacts.filter(contact=>ids.has(contact.from)&&ids.has(contact.to)),
     continuations:pattern.continuations?.filter(link=>ids.has(link.from)&&ids.has(link.to)),
     paper:pattern.paper.flatMap(paper=>this.paper(paper,pattern,whole,new Set(ownerIDs),views,(x,y,at)=>blocked(x,y,at,pattern.id)))});
   }
   if(!occurrences.length)continue;
   const ids=new Set(occurrences.map(atom=>atom.id)),covered=new Set(occurrences.flatMap(atom=>atom.contributions.map(contribution=>contribution.source))),byID=new Map(occurrences.map(atom=>[atom.id,atom]));
   visible.push({...pattern,occurrences,contacts:pattern.contacts.filter(contact=>byID.get(contact.from)?.contributions.some(c=>c.source===contact.fromSource)&&byID.get(contact.to)?.contributions.some(c=>c.source===contact.toSource)),
    continuations:pattern.continuations?.filter(link=>ids.has(link.from)&&ids.has(link.to)),
    requiredSources:pattern.requiredSources.filter(source=>covered.has(source)),
    paper:pattern.paper.flatMap(paper=>this.paper(paper,pattern,whole,new Set(ownerIDs),views,(x,y,at)=>blocked(x,y,at,pattern.id)))});
  }
  return {patterns:visible,unresolved,couplings:[...couplings.values()],terminalPatterns,terminalVisibility};
 }
 private paper(paper:NativePaper,pattern:NativePattern,relations:readonly PosedRelation[],owners:ReadonlySet<string>,views:readonly DrawingFieldView[],blocked:(x:number,y:number,at:Support)=>boolean):NativePaper[]{
  const [x,y]=paper.rect,dx=pattern.origin[0],dy=pattern.origin[1]*17;
  const foreground=(x:number,y:number)=>{
   const mapped=sourcePoint(pattern,x,y),sx=mapped?.[0]??x,sy=mapped?.[1]??y;
   const support=this.support.nearest(sx,sy,relations);if(support&&blocked(x,y,support))return true;
   for(const view of views){
    const sample=view.field.sample((sx-view.rect.x)/view.rect.width*view.field.width,(sy-view.rect.y)/view.rect.height*view.field.height);
    if(!sample||!sample.depth||sample.radiance[3]<.99)continue;

    const reference=sample.kind==='entity'?view.owners.get(sample.owner):sample.owner;
    const owner=`${view.camera}/${sample.source}/${sample.kind}/${reference}`;
    if(owners.has(owner))return false;
    const at=support;
    if(at&&sample.depth>at.depth+1e-6)return true;
   }
   return false;
  };
  // Enforce the SAME source visibility at the actual conflicting ink, not
  // at every pixel of a potentially screen-filling empty region. Relative
  // coordinates preserve this constraint through legal native pen shifts.
  // Integer pixel centres match the previous eager visibility partition.
  return [{...paper,visibleAt:(u,v)=>(!paper.visibleAt||paper.visibleAt(u,v))&&
   !foreground(Math.floor(x+u)+dx+.5,Math.floor(y+v)+dy+.5)}];
 }
}
