import {selectFeatureGraph} from './feature-graph.ts';
import {relationPoint,type PosedGraph,type PosedRelation} from './posed-graph.ts';
import type {CompiledConstructionGraph} from './composition.ts';
import {ROLE_ENTRIES} from './grammar/role-bank.ts';
import type {ArticulationPoint,ArticulationRun} from './gpu-articulation.ts';
import type {GrammarRole} from './role-grammar-format.ts';
import {shareThinPlaneRuns} from './thin-plane-runs.ts';

/** Topology only. No glyph masks, fit objectives, candidate words or legal-pen
 * trials occur here. The mesh worker supplies intrinsic critical vertices;
 * current-view degree/extrema split its posed features into run spans. Such
 * a split is not itself an anchor: only source corners/forks/caps get parts.
 * Glyph choice, line-class rank and repetition are GPU responsibilities. */
export class ConstructionRuns {
 private readonly keys=new Map<string,{owner:string;slot:number;epoch:number}>();private next=0;private epoch=0;
 private readonly free:number[]=[];
 private readonly parts=new Map<string,string>();
 private readonly retainedAnchors=new Map<string,{context:string;part:string|undefined}>();
 private readonly outlineOnly=new Set<string>();
 constructor(){
  // Whole anchor parts retain their DECLARED ports. A whole-circle entry's
  // secondary cap tag cannot turn the complete circle into a corner stamp.
  const exact=new Map<string,string>(),roles=['rim_cap','shoulder','flank','return_base','contact_junction','branch_entry'],bands=['s','m','l'],directions=['E','SE','S','SW','W','NW','N','NE'];
  for(const entry of ROLE_ENTRIES){if(!roles.includes(entry.role))continue;
   const key=`${entry.role}/${entry.size_band.band}/${entry.context.tangent_in}/${entry.context.tangent_out}`;if(!exact.has(key))exact.set(key,entry.id);
  }
  // D1's size-band order is compiled into this direct context lookup once;
  // no displayed frame enumerates anchor alternatives or measures ink.
  for(const role of roles)for(let size=0;size<3;size++)for(const incoming of directions)for(const outgoing of directions){
   for(const band of [bands[size],bands[Math.max(0,size-1)],bands[Math.min(2,size+1)]]){const part=exact.get(`${role}/${band}/${incoming}/${outgoing}`);if(part){this.parts.set(`${role}/${bands[size]}/${incoming}/${outgoing}`,part);break;}}
  }
 }
 retire(isLive:(owner:string)=>boolean):void{for(const [id,key]of this.keys)if(!isLive(key.owner)){this.free.push(key.slot);this.keys.delete(id);this.retainedAnchors.delete(id);}}
 private key(id:string,owner:string):number{let key=this.keys.get(id);if(key===undefined){if(this.epoch===0xffffffff)throw Error('Construction feature epoch extent exhausted');key={owner,slot:this.free.pop()??++this.next,epoch:++this.epoch};this.keys.set(id,key);}else if(key.owner!==owner)throw Error('Construction feature changed its source owner');return key.slot;}
 /** Current whole-form tier, not the height of one face or visible fragment. */
 allowsTone(owner:string):boolean{return !this.outlineOnly.has(owner);}
 compile(graphs:readonly PosedGraph[],parts:ReadonlyMap<string,CompiledConstructionGraph>=new Map()):readonly ArticulationRun[]{
  this.outlineOnly.clear();
  const result:ArticulationRun[]=[],anchors=new Set<string>();
  for(const graph of graphs){
   const byOwner=new Map<string,PosedRelation[]>();
   for(const relation of graph.relations){const list=byOwner.get(relation.owner)??[];list.push(relation);byOwner.set(relation.owner,list);}
   for(const [owner,relations]of byOwner){
    const ownerStart=result.length;
    const posedOwner=graph.owners?.get(owner),ownerSource:readonly [number,number]|undefined=posedOwner?[(posedOwner.domain|(posedOwner.kind==='producer-row'?0x80000000:0))>>>0,posedOwner.index]:undefined;
    const construction=parts.get(owner),rounded=construction?.type==='rounded';
    const planar=construction?.type==='planar';
    let ownerTop=Infinity,ownerBottom=-Infinity,ownerLeft=Infinity,ownerRight=-Infinity;
    for(const relation of relations)for(const t of [0,1]){const [x,y]=relationPoint(relation,t);if(Number.isFinite(y)){ownerTop=Math.min(ownerTop,y);ownerBottom=Math.max(ownerBottom,y);}if(Number.isFinite(x)){ownerLeft=Math.min(ownerLeft,x);ownerRight=Math.max(ownerRight,x);}}
    // D9 / petit-AA tier: no surface tone on a genuinely miniature form.
    // Use BOTH dimensions of its complete source contour: a long low wall
    // or far ridge is not a tiny subject merely because it is shallow.
    // A near-plane crossing has no finite projected bound; do not infer a
    // miniature tier from its unclipped/behind-camera endpoints.
    const extent=Math.max(ownerBottom-ownerTop,ownerRight-ownerLeft);
    if(Number.isFinite(extent)&&extent>=0&&Math.round(extent/17)<=7&&relations.every(relation=>relation.a[3]>.00001&&relation.b[3]>.00001))this.outlineOnly.add(owner);
    const critical=new Set(construction?.structure?.bands.flatMap(band=>band.criticalVertices)??[]);
    const forks=new Set(construction?.structure?.bands.filter(band=>band.lower.length>1||band.upper.length>1).flatMap(band=>band.criticalVertices)??[]);
    const incidence=new Map<string,PosedRelation[]>();for(const relation of relations)for(const port of relation.ports){const list=incidence.get(port)??[];list.push(relation);incidence.set(port,list);}
    // Separate line classes for the read hierarchy, while preserving actual
    // junction incidence across classes in the common anchor table.
    for(const kind of ['boundary','crease','smooth-contour','suggestive-contour'] as const){
     const interior=kind==='suggestive-contour',smooth=kind==='smooth-contour'||interior;
     // A compiled plane has no radial-curvature turn. Its real boundaries
     // and shared creases already own the cap/side/return graph; normal-zero
     // aliases must not add dotted second edges or columns across its face.
     if(planar&&smooth)continue;
     // A convex rounded solid has one mesh silhouette construction. Its
     // normal-zero family is not a second set of caps inside that solid.
     if(rounded&&kind==='smooth-contour')continue;
     if(rounded&&kind==='boundary'){
       // The mesh compiler has identified ONE rounded form. Its complete
       // source boundary feeds one paired section/height program, even when
       // a source seam divides the boundary graph into open walks. The GPU
       // already consumes independent edge pairs and their real visibility;
       // it neither needs nor fits an ellipse to a screen-space rectangle.
       // Never send an unpaired flank back through the soft-word lookup.
       const source=relations.filter(relation=>relation.kind==='boundary'),points:ArticulationPoint[]=[];
       for(const [index,item]of source.entries()){
        for(const interval of item.intervals){
         const point=(t:number,visible:boolean):ArticulationPoint|undefined=>{const w=item.a[3]*(1-t)+item.b[3]*t;if(w<=.00001)return;const p=relationPoint(item,t);if(!p.every(Number.isFinite))return;return {x:p[0],y:p[1],depth:p[2],parameter:(index+t)/source.length,reciprocalW:1/w,visible};};
         const a=point(interval.lo,interval.cause==='visible'),b=point(interval.hi,false);if(a&&b)points.push(a,b);
        }
       }
       if(points.some(point=>point.visible)){
        const height=ownerBottom-ownerTop,size:0|1|2=height<51?0:height<119?1:2;
        const id=`${owner}/round/whole-form`;result.push({id,owner,sources:source.map(relation=>relation.id),role:'flank',size,lineClass:0,key:this.key(id,owner),generation:this.keys.get(id)!.epoch,points,round:true,source:ownerSource});
       }
       continue;
      }
     const selected=selectFeatureGraph({...graph,relations:relations.filter(relation=>relation.kind===kind)});
     for(const walk of selected.walks){
      const nodes=walk.steps.map(step=>relationPoint(step.relation,step.reverse?1:0));
      if(!walk.closed){const last=walk.steps.at(-1)!;nodes.push(relationPoint(last.relation,last.reverse?0:1));}
      if(nodes.some(point=>!point.every(Number.isFinite)))continue;
      let top=Infinity,bottom=-Infinity;for(const point of nodes){top=Math.min(top,point[1]);bottom=Math.max(bottom,point[1]);}
      const size:0|1|2=bottom-top<51?0:bottom-top<119?1:2,band=['s','m','l'][size]!;
      const cuts:number[]=[];
      const actualAnchors=new Map<number,'corner'|'fork'|'cap'>();
      for(let i=0;i<nodes.length;i++){
       const previous=nodes[(i+nodes.length-1)%nodes.length]!,point=nodes[i]!,next=nodes[(i+1)%nodes.length]!;
       const step=walk.steps[Math.min(i,walk.steps.length-1)]!,port=i===walk.steps.length?walk.ports[1]:step.relation.ports[Number(step.reverse)];
       const match=/\/port-(\d+)/u.exec(port),criticalPort=match&&critical.has(Number(match[1]));
       const a=[point[0]-previous[0],point[1]-previous[1]],b=[next[0]-point[0],next[1]-point[1]];
       const turn=(a[0]!*b[0]!+a[1]!*b[1]!)/Math.max(1e-12,Math.hypot(...a)*Math.hypot(...b));
       const extrema=a[0]!*b[0]!<0||a[1]!*b[1]!<0;
       const vertex=match?Number(match[1]):undefined,sourceVertex=vertex!==undefined&&vertex<0x80000000;
       // A parameterization cut, normal-zero edge crossing, visibility end
       // or degree mismatch is not a fork/corner/cap in the construction.
       const corner=!smooth&&!rounded&&turn<.95&&(walk.closed||i>0&&i+1<nodes.length);
       // A section-graph split alone is not a drawable fork. It must also
       // have distinct incident source branches here; coincident line
       // classes at one edge do not create extra branches. Rounded solids
       // use their circle program, never interior branch-entry stamps.
       const branches=new Set((incidence.get(port)??[]).map(relation=>relation.ports[0]===port?relation.ports[1]:relation.ports[0]));
       const fork=!rounded&&sourceVertex&&forks.has(vertex!)&&branches.size>=3;
       const cap=sourceVertex&&criticalPort&&kind==='boundary'&&!walk.closed&&(i===0||i===nodes.length-1);
       // Interior zero-set endpoints/cuts do not acquire caps or fork
       // stamps merely by passing through a critical source vertex.
       if(!interior&&sourceVertex&&(corner||fork||cap))actualAnchors.set(i,fork?'fork':corner?'corner':'cap');
       // Hard-plane runs end at each real change of edge direction. A
       // shallow corner/chamfer is not a bent soft-contour word just because
       // its angle is smaller than the nonplanar shoulder threshold.
       if(!walk.closed&&(i===0||i===nodes.length-1)||criticalPort||(incidence.get(port)?.length??0)!==2||extrema||!smooth&&turn<(planar?1-1e-5:.95))cuts.push(i);
      }
      if(!cuts.length)cuts.push(0,Math.floor(nodes.length/2));
      else if(walk.closed&&cuts.length===1)cuts.push((cuts[0]!+Math.floor(nodes.length/2))%nodes.length);
      cuts.sort((a,b)=>a-b);
      const appendAnchor=(index:number,lineClass:ArticulationRun['lineClass'],source:readonly PosedRelation[])=>{
       const kind=actualAnchors.get(index);if(!kind)return;
       // A hard corner is not a biological root/contact junction. The role
       // bank's contact blocks include an entire trunk, foliage and ground.
       // Hard corners are the shared endpoints of their cap/side/return
       // runs; only an actual path-part fork/cap can take a whole block.
       if(kind==='corner'||construction?.type!=='path-led')return;
       const final=!walk.closed&&index===walk.steps.length,step=walk.steps[final?index-1:index]!,point=nodes[index]!;
       const t=final?Number(!step.reverse):Number(step.reverse),port=step.relation.ports[t],endpoint=t?step.relation.b:step.relation.a;
       const id=`${owner}/anchor/${port}`;if(anchors.has(id))return;
       const anchorRole=kind==='fork'?'branch_entry':point[1]<(top+bottom)*.5?'rim_cap':'return_base',directions=['E','SE','S','SW','W','NW','N','NE'];
       // Open endpoints have one actual tangent, not a fabricated connection
       // to the other end of the walk.
       const previous=index>0?nodes[index-1]!:walk.closed?nodes.at(-1)!:[2*point[0]-nodes[1]![0],2*point[1]-nodes[1]![1]];
       const next=index+1<nodes.length?nodes[index+1]!:walk.closed?nodes[0]!:[2*point[0]-previous[0]!,2*point[1]-previous[1]!];
       const direction=(p:readonly number[])=>directions[(Math.round(Math.atan2(p[1]!-point[1],p[0]!-point[0])*4/Math.PI)+8)%8]!;
       const context=`${anchorRole}/${direction(previous)}/${direction(next)}`;
       let retained=this.retainedAnchors.get(id);
       // Scale inserts run units between anchors. It does not re-spell a
       // surviving whole anchor merely because a size cut passed.
       if(!retained||retained.context!==context){retained={context,part:this.parts.get(`${anchorRole}/${band}/${direction(previous)}/${direction(next)}`)};this.retainedAnchors.set(id,retained);}
       const anchor:ArticulationPoint={x:point[0],y:point[1],depth:point[2],parameter:index/Math.max(1,walk.steps.length),visible:endpoint[3]>.00001&&step.relation.intervals.some(part=>part.cause==='visible'&&part.lo<=t&&part.hi>=t)};
       // Anchor placement belongs to the actual cap/foot, not the role of
       // its neighboring run. A foot keeps its above-anchor row offset even
       // when the tall continuing side correctly takes the flank grammar.
       const placementRole:GrammarRole=anchorRole==='return_base'?'return_base':anchorRole==='rim_cap'?'rim_cap':'flank';
       anchors.add(id);result.push({id,owner,sources:(incidence.get(port)??source).map(relation=>relation.id),role:placementRole,size,lineClass,key:this.key(id,owner),generation:this.keys.get(id)!.epoch,points:[anchor],part:retained.part,anchorOnly:true,source:ownerSource});
      };
      for(let c=0;c<(walk.closed?cuts.length:cuts.length-1);c++){
       const first=cuts[c]!,last=cuts[(c+1)%cuts.length]!,step=walk.steps[Math.min(first,walk.steps.length-1)]!;
       const from=step.relation.ports[Number(step.reverse)],endStep=walk.steps[(last+walk.steps.length-1)%walk.steps.length]!,to=endStep.relation.ports[Number(!endStep.reverse)];
       const id=`${owner}/run/${kind}/${from}>${to}`,source:PosedRelation[]=[],points:ArticulationPoint[]=[];
       for(let i=first;i!==last;i=(i+1)%nodes.length){
        const item=walk.steps[i];if(!item)break;source.push(item.relation);
        const intervals=item.reverse?[...item.relation.intervals].reverse().map(part=>({...part,lo:1-part.hi,hi:1-part.lo})):item.relation.intervals;
        for(const part of intervals){
         const point=(u:number,visible:boolean):ArticulationPoint|undefined=>{
          const t=item.reverse?1-u:u,w=item.relation.a[3]*(1-t)+item.relation.b[3]*t;if(w<=.00001)return;
          const p=relationPoint(item.relation,t);if(!p.every(Number.isFinite))return;
          return {x:p[0],y:p[1],depth:p[2],parameter:(source.length-1+u)/Math.max(1,walk.steps.length),reciprocalW:1/w,visible};
         };
         const a=point(part.lo,part.cause==='visible'),b=point(part.hi,false);if(a&&b)points.push(a,b);
        }
       }
       if(points.length<2||!points.some(point=>point.visible))continue;
       const a=nodes[first]!,b=nodes[last]!,height=bottom-top;
       const side:GrammarRole=Math.abs(b[1]-a[1])>Math.abs(b[0]-a[0])?'flank':'shoulder';
       const middle=(a[1]+b[1])*.5,ownerHeight=ownerBottom-ownerTop;
       // This is a WHOLE run between anchors, regardless of geometry class.
       // A tall limb/trunk side starting at a cap or foot is still a flank;
       // it must not repeat a cap/return word through the part's height.
       // Non-steep spans use their middle in the complete part's context,
       // not whichever endpoint happened to begin source traversal.
       const contextTop=planar?ownerTop:top,contextHeight=planar?ownerHeight:height;
       const role:GrammarRole=interior||side==='flank'?side:middle-contextTop<contextHeight*.15?'rim_cap':middle-contextTop>contextHeight*.78?'return_base':'shoulder';
       const sources=source.map(relation=>relation.id),lineClass=kind==='boundary'?0:kind==='smooth-contour'?1:interior?4:3;
       const runSize:0|1|2=planar?ownerHeight<51?0:ownerHeight<119?1:2:size;
       result.push({id,owner,sources,role,size:runSize,lineClass,key:this.key(id,owner),generation:this.keys.get(id)!.epoch,points,source:ownerSource,hard:planar&&!smooth,upright:source.every(relation=>relation.upright)});
       appendAnchor(first,lineClass,source);
       if(!walk.closed&&c===cuts.length-2)appendAnchor(last,lineClass,source);
      }
     }
    }
    if(planar&&construction?.planeRegions){const shared=shareThinPlaneRuns(result.slice(ownerStart),relations,construction.planeRegions);result.length=ownerStart;for(const run of shared)result.push(run);}
   }
  }
  return result;
 }
}
