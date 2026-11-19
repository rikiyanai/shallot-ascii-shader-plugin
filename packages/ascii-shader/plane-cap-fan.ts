import {relationPoint,relationRange,type PosedRelation} from './posed-graph.ts';
import type {NativePattern,NativeContinuation} from './native-layout.ts';
import type {NativeGlyph,NativeOccurrence,NativeContact} from './native-page.ts';
import {PlaneFrontSlab} from './plane-front-slab.ts';
import {PlaneInclinedFan} from './plane-inclined-fan.ts';

type Point=readonly [number,number];
interface Edge {relation:PosedRelation;from:string;to:string}
interface Fan {owner:string;center:string;left:string;upper:string;lower:string;
  upperLeft:string;upperLower:string;lowerLeft:string;
  points:Map<string,Point>;edges:Record<string,Edge>;reflection?:number}
export interface FanState {width:number;origin:number;row:number;depthRows?:number;wallRows?:number;mirrored?:boolean;context?:'inclined'}
export interface FanSlice {occurrence:string;source:string;parameter:readonly [number,number];
  /** Named source participation; hidden intervals never renumber these. */
  pixels:readonly number[];role:string}
export type FanResult={ok:true;neighborhood:string;pattern:NativePattern;state:FanState;slices:FanSlice[];
  fullEdges:readonly PosedRelation[];sourceVertices:ReadonlyMap<string,Point>;
  nativeVertices:ReadonlyMap<string,Point>;regionSockets:readonly {id:string;owner:string;bounds:readonly [number,number,number,number];sources:readonly string[]}[];
  implied:readonly {from:string;to:string;sourcePort:string;gap:number}[];
  production:{status:'provisional-whole-fan-growth';capWitness:string;returnWitnesses:readonly string[]}}
  |{ok:false;owner:string;obligation:string};

/** One generic THREE-FACE CORNER context, selected by source incidence. It is
 * not a box classifier: no8vertex/12edge mesh, normals, object name or scene
 * test exists. More complex plane maps need another whole source function.
 * Source cap: tetsudou-haikei/resK-99 r10 ∠__／|; r11–15 support context.
 * Coupled return: C104 |＿ plus C99 |／. Assembly/growth are provisional.
 */
export class PlaneCapFan {
  private readonly font:Map<string,NativeGlyph>;
  private readonly frontal:PlaneFrontSlab;
  private readonly inclined:PlaneInclinedFan;
  constructor(font:readonly NativeGlyph[]) {this.font=new Map(font.map(g=>[g.text,g]));this.frontal=new PlaneFrontSlab(font);this.inclined=new PlaneInclinedFan(font);}
  private bind(relations:readonly PosedRelation[]):Fan|string {
    if(!relations.length)return 'empty source component';
    const owner=relations[0]!.owner;
    if(relations.some(e=>e.owner!==owner||e.kind==='smooth-contour'))return 'fan function requires one complete hard-edge source component';
    const adjacency=new Map<string,string[]>(),points=new Map<string,Point>();
    for(const e of relations) {
      for(const [i,p]of e.ports.entries()) {
        const q=relationPoint(e,i);if(!q.every(Number.isFinite))return 'nonprojectable source port';
        points.set(p,[q[0],q[1]]);const list=adjacency.get(p)??[];list.push(e.ports[1-i]!);adjacency.set(p,list);
      }
    }
    // Remove one actual3way source junction: its three spokes must leave one
    // exterior cycle. This is a planar face-incidence context, not a silhouette
    // recognizer, and does not look at any visibility interval.
    const centers=[...adjacency].filter(([,neighbors])=>neighbors.length===3).map(([id])=>id).filter(center=>
      [...adjacency].every(([id,neighbors])=>id===center||neighbors.filter(n=>n!==center).length===2));
    if(centers.length!==1)return 'source plane map is not one three-face corner fan';
    const center=centers[0]!,arms=[...adjacency.get(center)!].sort((a,b)=>points.get(a)![1]-points.get(b)![1]||a.localeCompare(b));
    const upper=arms[0]!,left=arms[1]!,lower=arms[2]!,c=points.get(center)!;
    const reflection=points.get(left)![0]>c[0]&&points.get(upper)![0]<c[0]
      ?Math.round(Math.min(...[...points.values()].map(p=>p[0]))+Math.max(...[...points.values()].map(p=>p[0]))):undefined;
    if(reflection!==undefined)for(const [id,p]of points)points.set(id,[reflection-p[0],p[1]]);
    const normalized=points.get(center)!;
    if(!(points.get(left)![0]<normalized[0]&&points.get(upper)![0]>normalized[0]&&points.get(upper)![1]<normalized[1]&&points.get(lower)![1]>normalized[1]))
      return 'source fan handedness needs its own whole cap/return production';
    const between=(a:string,b:string)=>adjacency.get(a)!.filter(n=>n!==center&&adjacency.get(b)!.includes(n));
    const ul=between(upper,left),ud=between(upper,lower),ld=between(left,lower);
    if(ul.length!==1||ud.length!==1||ld.length!==1||new Set([center,upper,left,lower,ul[0],ud[0],ld[0]]).size!==7||points.size!==7||relations.length!==9)
      return 'fan exterior has additional source bends/regions; a larger whole-plane function is required';
    const edge=(from:string,to:string):Edge=>{
      const found=relations.filter(e=>e.ports.includes(from)&&e.ports.includes(to));
      if(found.length!==1)throw Error('Ambiguous source fan incidence');return {relation:found[0]!,from,to};
    };
    return {owner,center,left,upper,lower,upperLeft:ul[0]!,upperLower:ud[0]!,lowerLeft:ld[0]!,points,reflection,
      edges:{back:edge(ul[0]!,upper),leftSlope:edge(ul[0]!,left),front:edge(left,center),topSlope:edge(center,upper),
        backPost:edge(upper,ud[0]!),leftPost:edge(left,ld[0]!),frontPost:edge(center,lower),sideReturn:edge(lower,ud[0]!),base:edge(ld[0]!,lower)}};
  }
  /** Exact semilinear native width, no font enumeration or word fitting. */
  private units(length:number):{half:number;full:number}|undefined {
    const full=length%5,half=(length-16*full)/5;
    return length>=0&&half>=0&&Number.isInteger(half)?{half,full}:undefined;
  }
  grow(relations:readonly PosedRelation[],previous?:FanState,axes?:{readonly width:number;readonly origin:number;readonly row?:number},neighborhood=relations[0]?.owner??''):FanResult {
    const incumbent=this.growPreferred(relations,previous,axes,neighborhood);
    if(incumbent.ok)return incumbent;
    const compact=this.inclined.compact(relations,neighborhood);
    return compact.ok?compact:{...incumbent,obligation:`${incumbent.obligation}; ${compact.obligation}`};
  }
  private growPreferred(relations:readonly PosedRelation[],previous?:FanState,axes?:{readonly width:number;readonly origin:number;readonly row?:number},neighborhood=relations[0]?.owner??''):FanResult {
    if(relations.length===4||relations.length===7)return this.frontal.grow(relations,previous,axes,neighborhood);
    if(this.inclined.matches(relations))return this.inclined.grow(relations,previous,neighborhood);
    const owner=relations[0]?.owner??'',fail=(obligation:string):FanResult=>({ok:false,owner,obligation});
    const bound=this.bind(relations);if(typeof bound==='string')return fail(bound);const f=bound;
    for(const c of ['∠','_','＿','／','|'])if(!this.font.get(c)?.points.length)return fail(`native source scalar ${c} unavailable`);
    if(this.font.get('∠')!.advance!==16||this.font.get('_')!.advance!==5||this.font.get('＿')!.advance!==16||this.font.get('／')!.advance!==16||this.font.get('|')!.advance!==4)
      return fail('this source function requires the pinned native16/5/16/16/4 metrics');
    const l=f.points.get(f.left)!,c=f.points.get(f.center)!,targetWidth=c[0]-l[0];
    // These are shared FACE axes, not independently spelled edge fragments.
    // New whole side rows enter between the retained cap and return. Actual
    // straight architectural posts remain straight; the rounded-flank rule
    // is enforced by WholeRoleGrowth, where upright dominance was the fault.
    const sourceDepth=c[1]-f.points.get(f.upper)![1],sourceHeight=f.points.get(f.lower)![1]-c[1];
    const seedDepth=Math.max(1,Math.round(sourceDepth/17)),seedWall=Math.max(1,Math.round(sourceHeight/17));
    const depthRows=seedDepth,wallRows=seedWall;
    // One coupled native family. Every width increment inserts one full
    // unit in ALL three rails; modulo-based short/full conversion would
    // respell the entire rail every fifth increment.
    const repeats=Math.max(0,Math.round((targetWidth-34)/16));
    const width=axes?.width??34+16*repeats;
    if(width<34||(width-34)%16)return fail('shared axes require the complete fan width family');
    const m=(width-34)/16,extra=16*(depthRows-1);
    const localX=new Map<string,number>([[f.upperLeft,12+extra],[f.upper,width+16+extra],[f.left,2],[f.center,width+2],
      [f.upperLower,width+19+extra],[f.lowerLeft,2],[f.lower,width+4]]);
    const targetOrigin=Math.round([...localX].reduce((sum,[id,x])=>sum+f.points.get(id)![0]-x,0)/localX.size);
    const origin=axes?(f.reflection===undefined?axes.origin:f.reflection-axes.origin-(width+36+extra)):targetOrigin;
    const localY=new Map([[f.upperLeft,15],[f.upper,15],[f.left,17*depthRows+12],[f.center,17*depthRows+14],
      [f.upperLower,17*(wallRows+1)],[f.lowerLeft,17*(depthRows+wallRows)+15],[f.lower,17*(depthRows+wallRows)+15]]);
    const shape={row:Math.round([...localY].reduce((sum,[id,y])=>sum+f.points.get(id)![1]-y,0)/localY.size/17)};
    const choice={width,origin,n:4,m,back:{half:8,full:m},base:{half:0,full:m+2}};
    const {n,back,base}=choice,S=width+2,baseRow=depthRows+wallRows;
    // The native return/post interface is S+2. Its cap interface is S;
    // the two-pixel continuation remains perceptually aligned. This also
    // makes the side-wall gap 16D−5 = 11D+5(D−1), so every added wall row
    // is serializable with REAL native spaces, including overlapping posts.
    const native=new Map<string,Point>([[f.upperLeft,[12+extra,15]],[f.upper,[S+14+extra,15]],
      [f.left,[2,17*depthRows+12]],[f.center,[S,17*depthRows+14]],
      [f.upperLower,[S+17+extra,17*(wallRows+1)]],[f.lowerLeft,[2,17*baseRow+15]],[f.lower,[S+2,17*baseRow+15]]]);
    const row=axes?.row??shape!.row;
    if([...native].some(([id,p])=>Math.hypot(origin+p[0]-f.points.get(id)![0],row*17+p[1]-f.points.get(id)![1])>17))
      return fail('source face axes exceed the native fan growth envelope');
    let occurrences:NativeOccurrence[]=[];const slices:FanSlice[]=[];
    const add=(text:string,x:number,r:number,role:string,edgeName:string,range:readonly [number,number],select?:(p:readonly number[])=>boolean)=>{
      const glyph=this.font.get(text)!,e=f.edges[edgeName]!;
      const pixels=glyph.points.flatMap((p,i)=>!select||select(p)?[i]:[]);
      if(!pixels.length)return;
      const id=`${neighborhood}/plane-fan/${role}`;
      const parameter=relationRange(e.relation,e.from,range);
      const old=occurrences.find(a=>a.id===id),contribution={source:e.relation.id,kind:'path' as const,pixels,parameter};
      if(old)occurrences[occurrences.indexOf(old)]={...old,contributions:[...old.contributions,contribution]};
      else occurrences.push({id,text,x,row:r,contributions:[contribution]});
      slices.push({occurrence:id,source:e.relation.id,parameter,pixels,role});
    };
    const run=(text:string,start:number,r:number,role:string,edge:string,total:number,phaseOffset=0)=>{
      let pen=start;const ordinals=new Map<string,number>();
      for(const c of text) {
        const glyph=this.font.get(c)!,xs=glyph.points.map(p=>p[0]!);
        const ordinal=ordinals.get(c)??0;ordinals.set(c,ordinal+1);
        // INK parameter support, not a clipped scalar advance. Native paper
        // between glyphs remains part of the complete source feature phase.
        add(c,pen,r,`${role}/${c==='_'?'half':'full'}-${ordinal}`,edge,[(phaseOffset+pen-start+Math.min(...xs))/total,(phaseOffset+pen-start+Math.max(...xs))/total]);
        pen+=glyph.advance;
      }
      return pen;
    };
    // One short source unit fixes the back-rail start phase; long whole units
    // grow next; the remaining short continuation is one enduring suffix.
    // This full word is independent of visibility and is never restarted at a
    // clipping cut. A straddling occurrence remains an explicit obligation.
    // Last short unit's ink ends2px before its scalar cursor. Therefore the
    // scalar span is W+6 while the SOURCE endpoint span is W+4. Confusing the
    // two unnecessarily forced long units across the foreground cut.
    const backWord='_'+'＿'.repeat(back.full)+'_'.repeat(back.half-1);
    run(backWord,12+extra,0,'back','back',width+4);
    for(let r=1;r<depthRows;r++)add('／',16*(depthRows-r),r,`cap/left-depth-${r}`,'leftSlope',[(r-1)/depthRows,r/depthRows]);
    add('∠',0,depthRows,'cap/left-corner','leftSlope',[(depthRows-1)/depthRows,1],p=>p[1]!==12||p[0]===2);
    add('∠',0,depthRows,'cap/left-corner','front',[0,14/width],p=>p[1]===12);
    run('_'.repeat(n)+'＿'.repeat(m),16,depthRows,'cap/front','front',width,14);
    for(let r=1;r<=depthRows;r++)add('／',S+16*(depthRows-r),r,`cap/top-slope-${r}`,'topSlope',[1-r/depthRows,1-(r-1)/depthRows]);
    for(let r=1;r<=wallRows;r++)add('|',S+16+extra,r,`cap/back-post-${r}`,'backPost',[(r-1)/wallRows,r/wallRows]);
    for(let k=1;k<=wallRows;k++){
      const r=depthRows+k;
      add('|',1,r,k===wallRows?'return/left-post':`wall/left-${k}`,'leftPost',[(k-1)/wallRows,k/wallRows]);
      add('|',S+1,r,k===wallRows?'return/front-post':`wall/front-${k}`,'frontPost',[(k-1)/wallRows,k/wallRows]);
    }
    add('|',1,baseRow,'return/left-post','base',[0,3/(width+2)],p=>p[1]===15);
    run('_'.repeat(base.half)+'＿'.repeat(base.full),5,baseRow,'return/base','base',width+2,3);
    add('|',S+1,baseRow,'return/front-post','base',[1-3/(width+2),1],p=>p[1]===15);
    for(let k=1;k<=depthRows;k++)add('／',S+5+16*(depthRows-k),wallRows+k,`return/side-slope-${k}`,'sideReturn',[1-k/depthRows,1-(k-1)/depthRows]);
    let actualOrigin=origin;
    if(f.reflection!==undefined){
      // A complete opposite-handed production uses native ＼, never a
      // reflected/stretched glyph bitmap. The ∠ horizontal arm has no
      // counterpart in ＼; the front rail owns its own continuation.
      const axis=width+36+extra;actualOrigin=f.reflection-origin-axis;
      occurrences=occurrences.map(atom=>{
        const replacement=atom.text==='／'||atom.text==='∠'?'＼':atom.text,glyph=this.font.get(replacement)!;
        const contributions=atom.contributions.filter(c=>atom.text!=='∠'||c.source!==f.edges.front!.relation.id)
          .map(c=>({...c,pixels:atom.text==='∠'?glyph.points.map((_,i)=>i):c.pixels}));
        return {...atom,text:replacement,x:axis-atom.x-this.font.get(atom.text)!.advance,contributions};
      });
      for(let i=slices.length-1;i>=0;i--){
        const atom=occurrences.find(a=>a.id===slices[i]!.occurrence)!,contribution=atom.contributions.find(c=>c.source===slices[i]!.source);
        if(!contribution)slices.splice(i,1);else slices[i]={...slices[i]!,pixels:contribution.pixels};
      }
      for(const [id,p]of native)native.set(id,[axis-p[0],p[1]]);
      for(const [id,p]of f.points)f.points.set(id,[f.reflection-p[0],p[1]]);
    }
    const contacts:NativeContact[]=[],continuations:NativeContinuation[]=[],implied:{from:string;to:string;sourcePort:string;gap:number}[]=[];
    // Verify named source junction interfaces only AFTER the fixed full word
    // function is generated. No glyph or pen is selected by this calculation.
    for(const [port]of native) {
      const incident=relations.filter(e=>e.ports.includes(port));
      for(let i=1;i<incident.length;i++) {
        const a=incident[0]!,b=incident[i]!;let best:{a:NativeOccurrence;b:NativeOccurrence;p:Point;q:Point;d:number}|undefined;
        for(const sa of slices.filter(s=>s.source===a.id))for(const sb of slices.filter(s=>s.source===b.id)) {
          const aa=occurrences.find(o=>o.id===sa.occurrence)!,bb=occurrences.find(o=>o.id===sb.occurrence)!;
          for(const ia of sa.pixels)for(const ib of sb.pixels) {
            const pa=this.font.get(aa.text)!.points[ia]!,pb=this.font.get(bb.text)!.points[ib]!;
            const p:Point=[aa.x+pa[0]!+.5,aa.row*17+pa[1]!+.5],q:Point=[bb.x+pb[0]!+.5,bb.row*17+pb[1]!+.5],d=(p[0]-q[0])**2+(p[1]-q[1])**2;
            if(!best||d<best.d)best={a:aa,b:bb,p,q,d};
          }
        }
        if(!best)return fail(`unrealized source incidence ${port}`);
        if(best.d<=9)contacts.push({id:`${neighborhood}/plane-fan/contact/${port}/${i}`,from:best.a.id,to:best.b.id,
          fromSource:a.id,toSource:b.id,fromPort:port,toPort:port,fromAnchor:best.p,toAnchor:best.q,relation:'provisional-whole-plane-fan'});
        else {implied.push({from:best.a.id,to:best.b.id,sourcePort:port,gap:Math.sqrt(best.d)});
          continuations.push({from:best.a.id,to:best.b.id,fromAnchor:best.p,toAnchor:best.q,distance:Math.sqrt(best.d)+2});}
      }
    }
    const faces=[[f.upperLeft,f.upper,f.center,f.left],[f.left,f.center,f.lower,f.lowerLeft],[f.center,f.upper,f.upperLower,f.lower]];
    const charts=faces.flatMap(face=>[[0,1,2],[0,2,3]].map(indices=>({native:indices.map(i=>native.get(face[i]!)!) as [Point,Point,Point],
      source:indices.map(i=>f.points.get(face[i]!)!) as [Point,Point,Point]})));
    return {ok:true,neighborhood,pattern:{id:`${neighborhood}/plane-fan`,origin:[actualOrigin,row],occurrences,contacts,continuations,paper:[],requiredSources:relations.map(e=>e.id),
      repeatRuns:['back','cap/front','return/base'].map(role=>({id:`${neighborhood}/plane-fan/${role}`,members:occurrences.filter(atom=>slices.some(slice=>slice.occurrence===atom.id&&slice.role.startsWith(`${role}/`))).sort((a,b)=>a.x-b.x).map(atom=>atom.id)})),
      edgeCharts:relations.map(relation=>({source:relation.id,native:[native.get(relation.ports[0])!,native.get(relation.ports[1])!] as const})),
      surfaces:[{id:`${neighborhood}/plane-fan/body`,owner,outline:[f.upperLeft,f.upper,f.upperLower,f.lower,f.lowerLeft,f.left].map(id=>native.get(id)!),charts}]},
      state:{width,origin:actualOrigin,row,depthRows,wallRows,mirrored:f.reflection!==undefined},slices,fullEdges:relations,sourceVertices:f.points,nativeVertices:native,implied,
      regionSockets:[{id:`${neighborhood}/plane-fan/front-region`,owner,bounds:[5,17*(depthRows+1),Math.max(0,width-6),17*wallRows-3],sources:[f.edges.front!.relation.id,f.edges.leftPost!.relation.id,f.edges.frontPost!.relation.id,f.edges.base!.relation.id]}],
      production:{status:'provisional-whole-fan-growth',capWitness:'tetsudou-haikei/resK-99.txt#r10-r15:∠__／|',
        returnWitnesses:['C104 r21:|＿','C99 r47-r51:|／']}};
  }
}
