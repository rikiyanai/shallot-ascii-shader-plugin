import {sourceContourGraph} from './contour-fans.ts';
import {selectContourGraph,selectFeatureGraph} from './feature-graph.ts';
import {planeParts} from './plane-parts.ts';
import {relationPoint,type PosedRelation} from './posed-graph.ts';
import {fieldOwner,type CompositionGeometry,type CompositionSubject} from './composition.ts';
import type {DrawingFieldView} from './field-readback.ts';
import type {NativeGlyph,NativeOccurrence,NativePaper} from './native-page.ts';
import type {NativePattern,NativeSurface} from './native-layout.ts';

type Point=readonly [number,number];
interface Anchor {index:number;kind:'cap'|'return'|'left'|'right'|'junction';point:Point;port?:string}
interface Arc {id:string;points:readonly Point[];relations:readonly PosedRelation[];from:Anchor;to:Anchor;angular:boolean}
type Result={ok:true;pattern:NativePattern;configuration:string}|{ok:false;reason:string};

/** Source-part construction, not a bank of subject drawings. A complete
 * current-view boundary is articulated as incident turns and whole runs;
 * neither mesh triangles nor newly visible fragments elect glyphs. The
 * section graph qualifies a volume/branched part and retains its fork/void
 * lineage. Its section loops are NEVER published as rows of circles.
 *
 * This is the independent LP source program, not retained AAHub playback or
 * a claim of visual acceptance. Existing planar/rounded programs still own
 * their serving domains. Receiver/effect programs are separate obligations.
 */
export class LimbConstruction {
 private readonly font:Map<string,NativeGlyph>;
 constructor(glyphs:readonly NativeGlyph[]){this.font=new Map(glyphs.map(g=>[g.text,g]));}
 /** D1's generic class run uses the actual posed source graph even while a
  * static part compiler is pending, or for an open receiver. This is the same
  * articulation owner, not an image-to-character fallback. */
 terminal(subject:CompositionSubject,geometry:CompositionGeometry|undefined,view:DrawingFieldView|undefined):Result{
  return this.construct(subject,geometry,view,true);
 }
 construct(subject:CompositionSubject,geometry:CompositionGeometry|undefined,view:DrawingFieldView|undefined,terminal=false):Result{
  const fail=(reason:string):Result=>({ok:false,reason});
  if(!terminal&&!subject.structure)return fail(subject.structureError??'source part graph is still compiling');
  if(!view||!terminal&&!geometry)return fail('source part has no same-pose geometry/view');
  if(!terminal&&!subject.structure!.sections.some(section=>section.closed))return fail('open receiver needs its complete receiver construction');
  const id=`${subject.id}/${terminal?'class-run':'limb-complex'}`,occurrences:NativeOccurrence[]=[],paper:NativePaper[]=[],surfaces:NativeSurface[]=[];
  const paths:NonNullable<NativePattern['paths']>[number][]=[],repeatRuns:NonNullable<NativePattern['repeatRuns']>[number][]=[];
  const sourceParts:NonNullable<NativePattern['sourceParts']>[number][]=[];
  const input={ports:new Map(),relations:subject.relations};
  const boundaryGraph=geometry?sourceContourGraph(input,geometry,view):selectContourGraph({...input,relations:subject.relations.filter(r=>r.kind==='boundary')},relation=>{
   const c=relation.frontFacePoint;if(!c)return 0;const a=relationPoint(relation,0),b=relationPoint(relation,1);return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);
  });
  // A mixed mesh keeps its actual plane parts. The former whole-mesh
  // `path-led` dispatch made a multi-plane house/brick use round flanks.
  // Retain the enclosing form while articulating principal flat parts and
  // their shared turns together; do not transcribe every mesh facet.
  const planes=planeParts(subject,this.font.get('＿')!.advance,17);
  const interiors=selectFeatureGraph({...input,relations:planes.interiors});
  const graph={...boundaryGraph,walks:[...boundaryGraph.walks,...interiors.walks]},boundary=graph.walks.flatMap(walk=>walk.steps.map(step=>step.relation));
  if(!graph.walks.length)return fail('source part has no boundary construction');
  const degree=new Map<string,number>();for(const relation of boundary)for(const port of relation.ports)degree.set(port,(degree.get(port)??0)+1);
  // A perforated shell has genuine degree-three boundary ports. Requiring
  // every walk to close rejected its ENTIRE figure. Preserve that graph and
  // coauthor all incident arcs at one native junction instead of inventing a
  // closed cap or discarding the other valid source parts.
  const nodes=new Map<string,{anchor:Anchor;arcs:Arc[];neighbors:Point[]}>();
  let sourceLeft=Infinity,sourceTop=Infinity,sourceRight=-Infinity,sourceBottom=-Infinity;
  for(const relation of boundary)for(const t of [0,1]){const point=relationPoint(relation,t);if(!point.every(Number.isFinite))continue;sourceLeft=Math.min(sourceLeft,point[0]);sourceTop=Math.min(sourceTop,point[1]);sourceRight=Math.max(sourceRight,point[0]);sourceBottom=Math.max(sourceBottom,point[1]);}
  if(!Number.isFinite(sourceTop))return fail('part has no projectable complete source anchor');
  // A foreground cut must not restart the part's row phase at the top of its
  // remaining visible fragment. Phase and chart belong to the COMPLETE
  // source feature; only whole units are subsequently tested for visibility.
  // A real attached foot is less movable than the top of a growing body.
  // Bind by ORIGINAL vertex incidence, not the closest projected glyph/path.
  // Only an adapter-certified source/model contact qualifies. Read its
  // CURRENT posed source port directly; unprojecting float depth and then
  // comparing it to a double-precision model point adds no identity proof.
  // Arbitrary GPU deformation cannot certify this model contact.
  let footY:number|undefined;
  const contacts=[...(subject.supportContacts??[])].sort((a,b)=>a.sourceVertex-b.sourceVertex||a.receiver.localeCompare(b.receiver)||a.receiverFace-b.receiverFace);
  for(const contact of contacts){
   if(!contact.posed)continue;
   const port=`${subject.owner}/port-${contact.sourceVertex}`;
   for(const relation of boundary)for(const end of [0,1])if(relation.ports[end]===port||relation.ports[end]!.startsWith(port+'/')){
    const p=relationPoint(relation,end);if(p.every(Number.isFinite)){footY=p[1];break;}
   }
   if(footY!==undefined)break;
  }
  const phase=footY===undefined?Math.round(sourceTop/17)*17-sourceTop:Math.round((footY-16)/17)*17+16-footY;
  const row=(y:number)=>Math.round((y+phase-8)/17);
  const add=(text:string,x:number,row:number,key:string,sources:readonly string[])=>{
   const members:string[]=[];let pen=x;
   // Materialize only whole units intersecting this camera's page. Ordinals
   // still come from the COMPLETE source run, not a clipped/restarted path.
   // Near-plane projection can put a floor endpoint millions of rows away;
   // iterating that invisible prefix is neither construction nor coverage.
   const advance=[...text].reduce((sum,ch)=>sum+this.font.get(ch)!.advance,0);
   if(row*17+17<=view.rect.y||row*17>=view.rect.y+view.rect.height||x+advance<=view.rect.x||x>=view.rect.x+view.rect.width)return members;
   for(const [scalar,ch]of [...text].entries()){
    const glyph=this.font.get(ch);if(!glyph)throw Error(`LP program lacks native ${ch}`);
    if(glyph.points.length){const atom={id:`${key}/${scalar}`,text:ch,x:pen,row,contributions:sources.map(source=>({source,kind:'path' as const,pixels:glyph.points.map((_,i)=>i)}))};occurrences.push(atom);members.push(atom.id);}
    pen+=glyph.advance;
   }return members;
  };
  for(const walk of graph.walks){
   const source=walk.steps.map(step=>{const p=relationPoint(step.relation,step.reverse?1:0);return [p[0],p[1]] as Point;});
   const cyclic=walk.closed&&degree.get(walk.ports[0])===2;
   if(!cyclic){const step=walk.steps.at(-1)!,p=relationPoint(step.relation,step.reverse?0:1);source.push([p[0],p[1]]);}
   if(source.some(p=>!p.every(Number.isFinite)))return fail('part crosses a nonprojectable source port');
   const angular=terminal&&subject.type==='planar';
   let anchors=this.anchors(source,cyclic,angular,angular?17:2);
   if(!angular&&walk.steps.some(step=>planes.angular.has(step.relation.id))){
    const principalAt=(index:number)=>planes.angular.has(walk.steps[Math.min(index,walk.steps.length-1)]!.relation.id)||index>0&&planes.angular.has(walk.steps[index-1]!.relation.id);
    // Only the flat part takes text-scale angular simplification. A planar
    // patch on a mixed form must not erase its curved/forked neighbors.
    anchors=[...anchors.filter(anchor=>!principalAt(anchor.index)),...this.anchors(source,cyclic,true,17).filter(anchor=>principalAt(anchor.index))].sort((a,b)=>a.index-b.index);
   }
   for(const anchor of anchors){
    anchor.port=anchor.index===walk.steps.length?walk.ports[1]:walk.steps[anchor.index]!.relation.ports[Number(walk.steps[anchor.index]!.reverse)];
    if((degree.get(anchor.port)??0)>2)anchor.kind='junction';
   }
   if(anchors.length<2){if(terminal)continue;return fail('part requires its complete compact native construction');}
   const arcs:Arc[]=anchors.slice(0,cyclic?anchors.length:-1).map((from,i)=>{
    const to=anchors[(i+1)%anchors.length]!,points:Point[]=[],relations:PosedRelation[]=[];
    for(let index=from.index;index!==to.index;index=(index+1)%source.length){points.push(source[index]!);relations.push(walk.steps[index]!.relation);}
    points.push(to.point);return {id:`${id}/${walk.id}/arc-${walk.steps[from.index]!.relation.ports[Number(walk.steps[from.index]!.reverse)]}`,points,relations,from,to,angular:relations.length>0&&relations.every(relation=>planes.angular.has(relation.id))};
   });
   // Keep the full source domain, including hidden stretches, with this
   // connected part. Its actual receiver mask remains authoritative; these
   // charts do not turn a rectangular transport window into painted ink.
   let left=Infinity,right=-Infinity,top=Infinity,bottom=-Infinity;for(const p of source){left=Math.min(left,p[0]);right=Math.max(right,p[0]);top=Math.min(top,p[1]);bottom=Math.max(bottom,p[1]);}
   const q=[[left-24,top-24],[right+24,top-24],[right+24,bottom+24],[left-24,bottom+24]] as const,n=q.map(p=>[p[0],p[1]+phase] as Point);
   sourceParts.push({paths:arcs.map(arc=>arc.id),surface:{id:`${id}/${walk.id}/source-domain`,owner:subject.owner,coverage:'source-receiver',
    outline:walk.closed&&source.length>=3?source.map(p=>[p[0],p[1]+phase] as Point):n,
    charts:[[0,1,2],[0,2,3]].map(t=>({native:[n[t[0]!]!,n[t[1]!]!,n[t[2]!]!] as const,source:[q[t[0]!]!,q[t[1]!]!,q[t[2]!]!] as const}))}});
   for(const arc of arcs)paths.push({id:arc.id,relations:arc.relations.map(relation=>relation.id)});
   for(const arc of arcs)for(const [anchor,neighbor]of [[arc.from,arc.points[1]!],[arc.to,arc.points.at(-2)!]] as const){
    const key=anchor.port!,node=nodes.get(key)??{anchor,arcs:[],neighbors:[]};node.arcs.push(arc);node.neighbors.push(neighbor);nodes.set(key,node);
   }
   for(const arc of arcs){
    const a=arc.from.point,b=arc.to.point,first=row(a[1]),last=row(b[1]),direction=Math.sign(last-first),members:string[]=[];
    // Choose the WHOLE run unit once. Curvature changes the source shift,
    // not the scalar. No sample has a glyph candidate list or ink objective.
    const straightVertical=arc.points.every(p=>Math.abs(p[0]-a[0])<1e-5);
    if(Math.abs(b[0]-a[0])>Math.abs(b[1]-a[1])&&direction){
     const points=a[0]<b[0]?arc.points:[...arc.points].reverse(),descending=points.at(-1)![1]>points[0]![1];
     if(terminal||arc.angular){
      // One complete shallow-rise/return word per crossed native row. The
      // former fallback repeated a 43px rising word along a 2000px nearly
      // horizontal edge: every repetition invented another hill. Grow the
      // three coupled horizontal runs INSIDE one word instead. Vocabulary is
      // fixed for the entire arc; no glyph is elected by a sampled tangent.
      const low=Math.max(Math.min(first,last),Math.floor(view.rect.y/17)),high=Math.min(Math.max(first,last),Math.ceil((view.rect.y+view.rect.height)/17)-1);
      const minY=Math.min(a[1],b[1]),maxY=Math.max(a[1],b[1]);
      for(let r=low;r<=high;r++){
       const y0=Math.max(minY,Math.min(maxY,r*17-phase)),y1=Math.max(minY,Math.min(maxY,(r+1)*17-phase));if(y0===y1)continue;
       const start=this.crossing(arc.points,y0),end=this.crossing(arc.points,y1);if(start===undefined||end===undefined)continue;
       const left=Math.min(start,end),right=Math.max(start,end),width=right-left;if(width<=0)continue;
       const available=Math.max(0,width-30),count=(ch:string)=>Math.max(1,Math.round(available/3/this.font.get(ch)!.advance));
       const word:readonly (readonly [string,number])[]=width<30?[[descending?'＼':'／',1]]:width<59?[[descending?'｀‐-.._':'_..-‐´',1]]:
        descending?[['￣',count('￣')],['｀',1],["''",1],['‐',1],['-',count('-')],['..',1],['_',count('_')]]:
        [['_',count('_')],['..',1],['-',count('-')],['‐',1],["''",1],['´',1],['￣',count('￣')]];
       // Counts and phase belong to the UNCLIPPED source word. Materialize
       // its visible repeat ordinals algebraically; viewport cropping must
       // not restart a hill or change the whole word's growth parameters.
       let pen=Math.round(left);for(const [part,[unit,repeats]]of word.entries()){
        const advance=[...unit].reduce((sum,ch)=>sum+this.font.get(ch)!.advance,0),begin=Math.max(0,Math.floor((view.rect.x-pen)/advance)),end=Math.min(repeats-1,Math.floor((view.rect.x+view.rect.width-pen)/advance));
        for(let k=begin;k<=end;k++)members.push(...add(unit,pen+k*advance,r,`${arc.id}/whole-row-${r-first}/part-${part}/unit-${k}`,[arc.id]));
        pen+=advance*repeats;
       }
      }
     }else{
     const unit=descending?'｀‐--.._':'_..--‐´',glyphs=[...unit].map(ch=>this.font.get(ch)!);
     const advance=glyphs.reduce((sum,glyph)=>sum+glyph.advance,0),lengths=[0];
     for(let i=1;i<points.length;i++)lengths.push(lengths.at(-1)!+Math.hypot(points[i]![0]-points[i-1]![0],points[i]![1]-points[i-1]![1]));
     const total=lengths.at(-1)!,count=Math.max(0,Math.round(total/advance)-1),entryY=descending?0:15;
     const intervals=this.visibleOrdinals(points,lengths,advance,count,view,phase);
     for(const [first,last]of intervals)for(let k=first;k<=last;k++){
      const distance=advance*(k+.5);let at=1;while(at+1<lengths.length&&lengths[at]!<distance)at++;
      const from=points[at-1]!,to=points[at]!,t=(distance-lengths[at-1]!)/(lengths[at]!-lengths[at-1]!||1),x=from[0]+t*(to[0]-from[0]),y=from[1]+t*(to[1]-from[1]);
      members.push(...add(unit,Math.round(x),Math.round((y+phase-entryY)/17),`${arc.id}/sweep-${k}`,[arc.id]));
     }
     }
    }else if(!direction){
     const unit='＿',advance=this.font.get(unit)!.advance,left=Math.min(a[0],b[0]),right=Math.max(a[0],b[0]);
     const count=Math.max(0,Math.round((right-left)/advance)-1);
     const begin=Math.max(0,Math.ceil((view.rect.x-left)/advance-1.5)),end=Math.min(count-1,Math.floor((view.rect.x+view.rect.width-left)/advance-.5));
     for(let k=begin;k<=end;k++)members.push(...add(unit,Math.round(left+advance*(k+.5)),first,`${arc.id}/horizontal-${k}`,[arc.id]));
    }else{
     const unit=straightVertical&&subject.type!=='rounded'?'|':arc.angular||terminal&&subject.type==='planar'?((b[0]-a[0])*(b[1]-a[1])>0?'＼':'／'):(b[0]-a[0])*(b[1]-a[1])>0?'ヽ、':'ノ';
     const glyphs=[...unit].map(ch=>this.font.get(ch)!),ink=glyphs.flatMap((glyph,i)=>{const offset=glyphs.slice(0,i).reduce((sum,g)=>sum+g.advance,0);return glyph.points.map(p=>p[0]!+offset);});
     const anchorX=(Math.min(...ink)+Math.max(...ink))/2;
     const low=Math.max(Math.min(first,last)+1,Math.floor(view.rect.y/17)),high=Math.min(Math.max(first,last)-1,Math.ceil((view.rect.y+view.rect.height)/17)-1);
     for(let r=direction>0?low:high;direction>0?r<=high:r>=low;r+=direction){
      const y=17*r+8-phase,x=this.crossing(arc.points,y);
      if(x===undefined)return fail('complete source run lost its row crossing');
      members.push(...add(unit,Math.round(x-anchorX),r,`${arc.id}/repeat-${Math.abs(r-first)}`,[arc.id]));
     }
    }
    repeatRuns.push({id:arc.id,members});
   }
  }
  for(const [port,node]of nodes){
   const count=node.arcs.length;if(count===1)continue;
   const a=node.anchor,up=node.neighbors.filter(p=>p[1]<a.point[1]).length,down=node.neighbors.filter(p=>p[1]>a.point[1]).length;
   const angular=node.arcs.some(arc=>arc.angular)||terminal&&subject.type==='planar';
   const text=count===2?(a.kind==='cap'?'∧':a.kind==='return'?'∨':a.kind==='left'?(angular?'<':'('):(angular?'>':')')):
    count===3?(up===2&&down===1?'Y':up===1&&down===2?'人':node.neighbors.reduce((sum,p)=>sum+p[0]-a.point[0],0)>=0?'ト':'┤'):
    count===4?'Ｘ':count===5?'大':count===6?'＊':undefined;
   if(!text){if(terminal)continue;return fail('source junction needs its complete higher-valence native neighborhood');}
   const glyph=this.font.get(text);if(!glyph)return fail(`source junction lacks native whole ${text}`);
   const xs=glyph.points.map(p=>p[0]!),source=`${id}/junction/${port}`;
   paths.push({id:source,relations:[...new Set(node.arcs.flatMap(arc=>arc.relations.map(relation=>relation.id)))]});
   add(text,Math.round(a.point[0]-(Math.min(...xs)+Math.max(...xs))/2),row(a.point[1]),source,[source]);
  }
  if(!occurrences.length)return fail('part has no complete native footprint');
  // This is an affine source CHART, not a box-shaped body or envelope.
  // NativeSurfaceVisibility tests the exact receiver owner/depth at every
  // mapped point; holes and finger/branch gaps remain genuine empty space.
  const [left,top,right,bottom]=[sourceLeft,sourceTop,sourceRight,sourceBottom],q=[[left-24,top-24],[right+24,top-24],[right+24,bottom+24],[left-24,bottom+24]] as const;
  const n=q.map(p=>[p[0],p[1]+phase] as Point);
  surfaces.push({id:`${id}/source-chart`,owner:subject.owner,coverage:'source-receiver',outline:n,
   charts:[[0,1,2],[0,2,3]].map(t=>({native:[n[t[0]!]!,n[t[1]!]!,n[t[2]!]!] as const,source:[q[t[0]!]!,q[t[1]!]!,q[t[2]!]!] as const}))});
  // Fork channels are protected in the SAME parent as all its incident
  // turns/flanks. They are actual void intervals, not a blanket lit rectangle
  // through the body that would prohibit the later structural/light passage.
  const byRow=new Map<number,NativeOccurrence[]>();for(const atom of occurrences){const list=byRow.get(atom.row)??[];list.push(atom);byRow.set(atom.row,list);}
  for(const [r,atoms]of byRow){
   atoms.sort((a,b)=>a.x-b.x);for(let i=1;i<atoms.length;i++){
    const a=atoms[i-1]!,b=atoms[i]!,start=a.x+this.font.get(a.text)!.advance+2,end=b.x-2;
    if(end-start<5)continue;const center:Point=[(start+end)/2,r*17+8];
    const sample=view.field.sample((center[0]-view.rect.x)/view.rect.width*view.field.width,(center[1]-phase-view.rect.y)/view.rect.height*view.field.height);
    if(sample&&fieldOwner(view,sample)===subject.owner)continue;
    paper.push({id:`${id}/channel/${r}/${i}`,rect:[start,r*17,end-start,17],sources:[...new Set([...a.contributions,...b.contributions].map(c=>c.source))]});
   }
  }
  return {ok:true,configuration:`source-part/${geometry?.revision??'posed'}/${terminal?'terminal':subject.structure!.bands.map(b=>`${b.lower.join('.')}>${b.upper.join('.')}`).join(';')}`,
   pattern:{id,origin:[0,0],occurrences,contacts:[],paper,requiredSources:paths.map(path=>path.id),paths,repeatRuns,surfaces,sourceParts}};
 }


 private visibleOrdinals(points:readonly Point[],lengths:readonly number[],advance:number,count:number,view:DrawingFieldView,phase:number):[number,number][]{
  const ranges:[number,number][]=[],left=view.rect.x-advance,right=view.rect.x+view.rect.width,top=view.rect.y-phase-17,bottom=view.rect.y+view.rect.height-phase+17;
  for(let i=1;i<points.length;i++){
   const a=points[i-1]!,b=points[i]!,dx=b[0]-a[0],dy=b[1]-a[1];let lo=0,hi=1;
   for(const [p,q]of [[-dx,a[0]-left],[dx,right-a[0]],[-dy,a[1]-top],[dy,bottom-a[1]]] as const){if(!p){if(q<0){hi=-1;break;}continue;}const t=q/p;if(p<0)lo=Math.max(lo,t);else hi=Math.min(hi,t);}
   if(lo>hi)continue;
   const length=lengths[i]!-lengths[i-1]!,first=Math.max(0,Math.ceil((lengths[i-1]!+lo*length)/advance-.5)),last=Math.min(count-1,Math.floor((lengths[i-1]!+hi*length)/advance-.5));
   if(first>last)continue;const prior=ranges.at(-1);if(prior&&first<=prior[1]+1)prior[1]=Math.max(prior[1],last);else ranges.push([first,last]);
  }
  return ranges;
 }
 private crossing(points:readonly Point[],y:number):number|undefined{
  for(let i=1;i<points.length;i++){const a=points[i-1]!,b=points[i]!;if(y<Math.min(a[1],b[1])-1e-6||y>Math.max(a[1],b[1])+1e-6)continue;if(a[1]===b[1])continue;return a[0]+(y-a[1])*(b[0]-a[0])/(b[1]-a[1]);}
 }
 private anchors(points:readonly Point[],closed:boolean,planeCorners=false,resolution=2):Anchor[]{
  // Source-only line economy: contract <=2px facet wiggles, retaining the
  // original source indices/ranges of every resulting arc. This does not
  // fit a glyph to the polyline or delete a source loop/part.
  let top=0,bottom=0;for(let i=1;i<points.length;i++){if(points[i]![1]<points[top]![1])top=i;if(points[i]![1]>points[bottom]![1])bottom=i;}
  if(top===bottom){for(let i=1;i<points.length;i++)if(points[i]![0]>points[bottom]![0])bottom=i;}
  const keep=new Set(closed?[top,bottom]:[0,points.length-1]),pending:[number,number][]=closed?[[top,bottom],[bottom,top]]:[[0,points.length-1]];
  while(pending.length){const [first,last]=pending.pop()!,a=points[first]!,b=points[last]!,dx=b[0]-a[0],dy=b[1]-a[1],length=dx*dx+dy*dy;let distance=resolution*resolution,selected=-1;
   for(let i=(first+1)%points.length;i!==last;i=(i+1)%points.length){const p=points[i]!,t=length?Math.max(0,Math.min(1,((p[0]-a[0])*dx+(p[1]-a[1])*dy)/length)):0,d=(p[0]-a[0]-t*dx)**2+(p[1]-a[1]-t*dy)**2;if(d>distance){distance=d;selected=i;}}
   if(selected>=0){keep.add(selected);pending.push([first,selected],[selected,last]);}
  }
  const selected=[...keep].sort((a,b)=>a-b),anchors:Anchor[]=[];
  const direction=(axis:number,at:number,step:number)=>{const p=points[selected[at]!]!;for(let n=1;n<selected.length;n++){const index=at+step*n;if(!closed&&(index<0||index>=selected.length))break;const q=points[selected[(index+selected.length*n)%selected.length]!]!,delta=q[axis]!-p[axis]!;if(Math.abs(delta)>1e-5)return Math.sign(delta);}return 0;};
  for(let at=0;at<selected.length;at++){
   const index=selected[at]!,beforeY=direction(1,at,-1),afterY=direction(1,at,1),beforeX=direction(0,at,-1),afterX=direction(0,at,1);
   const kind=!closed&&(at===0||at===selected.length-1)?'junction':beforeY>0&&afterY>0?'cap':beforeY<0&&afterY<0?'return':beforeX>0&&afterX>0?'left':beforeX<0&&afterX<0?'right':undefined;
   if(kind||planeCorners)anchors.push({index,kind:kind??'junction',point:points[index]!});
  }return anchors;
 }
}
