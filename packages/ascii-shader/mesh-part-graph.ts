import {weldSourceVertices} from './source-complex.ts';

type Point=readonly [number,number,number];
/** Part label word, the same layout on a section, a band and a chain:
 *  bit 0 PART_LIMB  the chain is an elongated tube: every section closed and
 *                   un-nested, length >= PART_LIMB_ELONGATION x mean width;
 *  bit 1 PART_FORK  (bands only) the band joins three or more sections;
 *  bit 2 PART_ROOT  a free chain end that reaches the source's support plane
 *                   (lowest source Y, within PART_ROOT_PLANE of the Y extent);
 *  bit 3 PART_CAP   any other free chain end;
 *  bits 8..31       chain id + 1 (0: no chain, e.g. a fork band).
 * A chain is the run of sections joined by one-below/one-above bands, so it
 * ends at a fork, an arch or a free end. A section whose two ends are both
 * free (a post, a stump) carries both end bits it earns. The resident packet
 * stores the word unchanged (compiled-construction-topology.ts,
 * TOPOLOGY_SECTION_LABEL_WORD / TOPOLOGY_BAND_LABEL_WORD). */
export const PART_LIMB=1,PART_FORK=2,PART_ROOT=4,PART_CAP=8,PART_CHAIN_SHIFT=8;
/** A tube twice as long as it is wide is a limb. A cube measures about .8 and
 * a sphere about 1 by the same measure, so neither becomes one. */
export const PART_LIMB_ELONGATION=2;
/** A free end within this share of the source's Y extent above its lowest
 * point stands on the support plane (root flares do not end at one height). */
export const PART_ROOT_PLANE=.02;
export const partChain=(labels:number):number=>(labels>>>PART_CHAIN_SHIFT)-1;
export interface PartChain {
 readonly id:number;readonly sections:readonly number[];readonly bands:readonly number[];
 /** LIMB plus the end bits of its sections. */
 readonly labels:number;readonly length:number;readonly width:number;
}
export interface PartMesh {
 readonly positions:Float32Array;readonly indices:Uint32Array;
 /** Published ORIGINAL face adjacency. This is not a proximity weld. */
 readonly adjacency?:{readonly offsets:Uint32Array;readonly faces:Uint32Array};
}
export interface PartSample {readonly a:number;readonly b:number;readonly t:number}
export interface PartSection {
 readonly id:number;readonly component:number;readonly level:number;readonly samples:readonly PartSample[];
 readonly segments:readonly (readonly [number,number,number])[];
 readonly faces:readonly number[];readonly closed:boolean;readonly center:Point;
 /** Preserve nested holes/islands in one surface component. Nesting alone
  * does not make this section a publishable closed solid or a separate limb. */
 readonly inside?:number;
 /** Part label word (PART_LIMB | PART_ROOT | PART_CAP | chain). */
 readonly labels?:number;
}
export interface PartBand {
 readonly id:number;readonly lower:readonly number[];readonly upper:readonly number[];
 readonly faces:readonly number[];readonly criticalVertices:readonly number[];
 /** Part label word (PART_LIMB | PART_FORK | PART_ROOT | PART_CAP | chain). */
 readonly labels?:number;
}
export interface MeshPartGraph {
 readonly axis:Point;readonly origin:Point;
 readonly sections:readonly PartSection[];readonly bands:readonly PartBand[];
 readonly levels:readonly number[];readonly vertexRepresentatives:Uint32Array;
 readonly chains?:readonly PartChain[];
}
const dot=(a:readonly number[],b:readonly number[])=>a[0]!*b[0]!+a[1]!*b[1]!+a[2]!*b[2]!;
const sub=(a:Point,b:Point):Point=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]];
const cross=(a:Point,b:Point):Point=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const length=(a:Point)=>Math.hypot(...a);
const unit=(a:Point):Point=>{const n=length(a);return n?[a[0]/n,a[1]/n,a[2]/n]:[0,1,0];};
export function partPoint(mesh:Pick<PartMesh,'positions'>,sample:PartSample):Point{
 const p=mesh.positions;return [0,1,2].map(k=>p[sample.a*3+k]!*(1-sample.t)+p[sample.b*3+k]!*sample.t) as unknown as Point;
}
function linkComponents(neighbors:ReadonlySet<number>,edges:readonly (readonly [number,number])[],keep:(v:number)=>boolean):number{
 const parent=new Map([...neighbors].filter(keep).map(v=>[v,v]));
 const root=(v:number):number=>{let p=v;while(parent.get(p)!==p)p=parent.get(p)!;return p;};
 for(const [a,b]of edges)if(parent.has(a)&&parent.has(b)){const x=root(a),y=root(b);parent.set(Math.max(x,y),Math.min(x,y));}
 return new Set([...parent.keys()].map(root)).size;
}

/** Output-sensitive source-interval index. Independent monotone cursors
 * share the immutable ordering (section points and connecting slabs), not
 * mutable sweep state. Returned IDs preserve the original source order. */
function spanIndex(low:Float64Array,high:Float64Array):()=>((lo:number,hi:number)=>Generator<void,readonly number[]>){
 const starts=Uint32Array.from({length:low.length},(_,i)=>i).sort((a,b)=>low[a]!-low[b]!||a-b);
 const ends=Uint32Array.from({length:high.length},(_,i)=>i).sort((a,b)=>high[a]!-high[b]!||a-b);
 return ()=>{
  const active=new Set<number>();let first=0,last=0;
  return function*(lo:number,hi:number):Generator<void,readonly number[]>{
   // Exactly the original open-interval predicate: high > lo && low < hi.
   // Flat faces/edges and endpoints on a section keep their old treatment.
   while(first<starts.length&&low[starts[first]!]!<hi){const id=starts[first++]!;if(high[id]!>lo)active.add(id);yield;}
   while(last<ends.length&&high[ends[last]!]!<=lo){active.delete(ends[last++]!);yield;}
   return [...active].sort((a,b)=>a-b);
  };
 };
}

/** Source-space section graph, compiled once per immutable geometry revision.
 * Cross-section loops are GUIDES, never a row of circles to publish. Bands
 * retain split/merge incidence, source faces and nested holes, so articulation
 * must build the entire fork/attachment rather than cap each branch alone.
 * No camera, screen grid, glyph bank, asset name or fitted image participates.
 */
export function* meshPartWork(mesh:PartMesh,sourceWeld?:Uint32Array):Generator<void,MeshPartGraph>{
 const {positions:p,indices:ix,adjacency}=mesh,vertexCount=p.length/3,faceCount=ix.length/3;
 if(!Number.isInteger(vertexCount)||!Number.isInteger(faceCount))throw Error('Part source has an incomplete triangle extent');
 const point=(v:number):Point=>[p[v*3]!,p[v*3+1]!,p[v*3+2]!];
 const parent=Uint32Array.from({length:vertexCount},(_,i)=>i),used=new Set<number>();
 const root=(v:number):number=>{let r=v;while(parent[r]!==r)r=parent[r]!;while(parent[v]!==v){const next=parent[v]!;parent[v]=r;v=next;}return r;};
 const join=(a:number,b:number)=>{a=root(a);b=root(b);parent[Math.max(a,b)]=Math.min(a,b);};
 let area=0;const mean=[0,0,0],moment=new Float64Array(9);
 for(let f=0;f<faceCount;f++){
  const vertices=[ix[f*3]!,ix[f*3+1]!,ix[f*3+2]!];
  if(vertices.some(v=>v>=vertexCount))throw Error('Part source index is outside its vertex stream');
  vertices.forEach(v=>used.add(v));const q=vertices.map(point),weight=length(cross(sub(q[1]!,q[0]!),sub(q[2]!,q[0]!)))/2;
  if(weight){const sum=[0,1,2].map(k=>q[0]![k]!+q[1]![k]!+q[2]![k]!);area+=weight;
   for(let i=0;i<3;i++){mean[i]!+=weight*sum[i]!/3;for(let j=0;j<3;j++)moment[i*3+j]!+=weight*(sum[i]!*sum[j]!+q.reduce((n,v)=>n+v[i]!*v[j]!,0))/12;}}
  yield;
 }
 if(!area)throw Error('Part source has no surface area');
 const origin=mean.map(v=>v/area) as unknown as Point,cov=Array.from(moment,(v,k)=>v/area-origin[Math.floor(k/3)]!*origin[k%3]!);
 // A surface-area principal axis is independent of tessellation density and
 // camera pose. Canonical sign prevents a rigid turn from renaming branches.
 let axis:Point=[0,1,0];const major=[0,1,2].sort((a,b)=>cov[b*3+b]!-cov[a*3+a]!)[0]!;axis=[0,0,0].map((_,i)=>Number(i===major)) as unknown as Point;
 for(let step=0;step<32;step++){axis=unit([0,1,2].map(i=>cov[i*3]!*axis[0]+cov[i*3+1]!*axis[1]+cov[i*3+2]!*axis[2]) as unknown as Point);yield;}
 const sign=[0,1,2].sort((a,b)=>Math.abs(axis[b]!)-Math.abs(axis[a]!))[0]!;if(axis[sign]!<0)axis=axis.map(v=>-v) as unknown as Point;
 // P0's separately indexed triangles retain their published face relation.
 // Match CORNERS inside an already related pair sharing a whole edge. The
 // rest-position weld below joins the remaining position-equal corners.
 if(adjacency){
  if(adjacency.offsets[0]!==0||adjacency.offsets.at(-1)!==adjacency.faces.length)throw Error('Part source adjacency extent is invalid');
  for(let e=0;e+1<adjacency.offsets.length;e++){
   const first=adjacency.offsets[e]!,last=adjacency.offsets[e+1]!;
   if(last<first||last>adjacency.faces.length)throw Error('Part source adjacency range is invalid');
   for(let a=first;a<last;a++)for(let b=a+1;b<last;b++){
    const fa=adjacency.faces[a]!,fb=adjacency.faces[b]!;if(fa>=faceCount||fb>=faceCount)throw Error('Part source adjacency names an absent face');
    const matches:[number,number][]=[];
    for(let i=0;i<3;i++)for(let j=0;j<3;j++){const x=ix[fa*3+i]!,y=ix[fb*3+j]!;if([0,1,2].every(k=>Math.abs(p[x*3+k]!-p[y*3+k]!)<=1e-6))matches.push([x,y]);}
    if(matches.length>=2)for(const [x,y]of matches)join(x,y);
    yield;
   }
  }
 }
 // Position-equal corners are one vertex of the surface (the same welded
 // identity the source complex compiles). Without it a mesh exported as
 // triangle soup had no shared vertex at all: every triangle was its own
 // open section and band, and no limb, fork or end could be found.
 // The registration owner already compiled this exact rest-position weld.
 // Published adjacency above still contributes its distinct authorized joins.
 const weld=sourceWeld??weldSourceVertices(p);
 for(let v=0;v<vertexCount;v++){if(weld[v]!==v)join(v,weld[v]!);yield;}
 const representatives=Uint32Array.from({length:vertexCount},(_,i)=>root(i)),height=new Float64Array(vertexCount);
 let low=Infinity,high=-Infinity;for(const v of used){height[v]=dot(sub(point(v),origin),axis);low=Math.min(low,height[v]!);high=Math.max(high,height[v]!);yield;}
 const epsilon=Math.max(1,Math.abs(low),Math.abs(high))*Number.EPSILON*64;
 const links=new Map<number,{neighbors:Set<number>;edges:[number,number][]}>(),edges=new Map<string,{a:number;b:number;faces:number[]}>();
 const triangles:readonly number[][]=Array.from({length:faceCount},(_,f)=>[representatives[ix[f*3]!]!,representatives[ix[f*3+1]!]!,representatives[ix[f*3+2]!]!]);
 const faceLow=new Float64Array(faceCount),faceHigh=new Float64Array(faceCount);
 for(let f=0;f<faceCount;f++){
  const tri=triangles[f]!;faceLow[f]=Math.min(...tri.map(v=>height[v]!));faceHigh[f]=Math.max(...tri.map(v=>height[v]!));
  for(let k=0;k<3;k++){
   const v=tri[k]!,a=tri[(k+1)%3]!,b=tri[(k+2)%3]!,link=links.get(v)??{neighbors:new Set<number>(),edges:[]};link.neighbors.add(a);link.neighbors.add(b);link.edges.push([a,b]);links.set(v,link);
   const x=Math.min(v,a),y=Math.max(v,a),key=`${x}/${y}`,edge=edges.get(key)??{a:x,b:y,faces:[]};edge.faces.push(f);edges.set(key,edge);
  }
  yield;
 }
 const faceParent=Uint32Array.from({length:faceCount},(_,i)=>i);
 const faceRoot=(f:number):number=>{let r=f;while(faceParent[r]!==r)r=faceParent[r]!;while(faceParent[f]!==f){const n=faceParent[f]!;faceParent[f]=r;f=n;}return r;};
 for(const edge of edges.values()){for(let i=1;i<edge.faces.length;i++){const a=faceRoot(edge.faces[0]!),b=faceRoot(edge.faces[i]!);faceParent[Math.max(a,b)]=Math.min(a,b);}yield;}
 const critical=new Map<number,number[]>();
 for(const [v,link]of links){
  const h=height[v]!,below=linkComponents(link.neighbors,link.edges,n=>height[n]!<h-epsilon),above=linkComponents(link.neighbors,link.edges,n=>height[n]!>h+epsilon);
  if(below!==1||above!==1){const group=critical.get(h)??[];group.push(v);critical.set(h,group);}yield;
 }
 const levels=[...new Set([low,...critical.keys(),high])].sort((a,b)=>a-b).filter((h,i,a)=>i===0||h-a[i-1]!>epsilon);
 const sections:PartSection[]=[],stations:PartSection[][]=[],sectionStation:number[]=[];
 const basis=unit(cross(axis,Math.abs(axis[0])<.8?[1,0,0]:[0,0,1])),vertical=cross(axis,basis);
 const faceSpans=spanIndex(faceLow,faceHigh),sectionFaces=faceSpans();
 for(let station=0;station+1<levels.length;station++){
  // A non-vertex regular value avoids making a flat source face into a
  // fictitious contour. It is intrinsic, not a screen-space sampling grid.
  const level=levels[station]!+(levels[station+1]!-levels[station]!)*(Math.SQRT2-1);
  const samples:PartSample[]=[],sampleKeys=new Map<string,number>(),segments:[number,number,number][]=[],incidence=new Map<number,number[]>();
  for(const f of yield* sectionFaces(level,level)){
   if(faceLow[f]!<level&&faceHigh[f]!>level){
    const hits:number[]=[];
    for(let k=0;k<3;k++){
     const a=triangles[f]![k]!,b=triangles[f]![(k+1)%3]!,ha=height[a]!,hb=height[b]!;
     if((ha<level)===(hb<level)||ha===hb)continue;
     const x=Math.min(a,b),y=Math.max(a,b),key=`${x}/${y}`;let id=sampleKeys.get(key);
     if(id===undefined){id=samples.length;sampleKeys.set(key,id);samples.push({a:x,b:y,t:(level-height[x]!)/(height[y]!-height[x]!)});}hits.push(id);
    }
    if(hits.length===2&&hits[0]!==hits[1]){const id=segments.length;segments.push([hits[0]!,hits[1]!,f]);for(const hit of hits){const list=incidence.get(hit)??[];list.push(id);incidence.set(hit,list);}}
   }yield;
  }
  const seen=new Set<number>(),current:PartSection[]=[];
  for(let first=0;first<segments.length;first++){
   if(seen.has(first))continue;
   const pending=[first],members:number[]=[],vertices=new Set<number>();seen.add(first);
   while(pending.length){const at=pending.pop()!,segment=segments[at]!;members.push(at);for(const v of segment.slice(0,2)){vertices.add(v);for(const next of incidence.get(v)??[])if(!seen.has(next)){seen.add(next);pending.push(next);}}yield;}
   const closed=[...vertices].every(v=>incidence.get(v)!.length===2),order:number[]=[];
   if(closed){let at=segments[first]![0],before=-1;do{order.push(at);const next=incidence.get(at)!.find(e=>e!==before)!;const edge=segments[next]!;at=edge[0]===at?edge[1]:edge[0];before=next;}while(at!==order[0]&&order.length<=vertices.size);}
   else order.push(...vertices);
   const local=new Map(order.map((v,i)=>[v,i])),lineage=[...new Set(members.map(i=>segments[i]![2]))].sort((a,b)=>a-b),center=[0,0,0];let total=0;
   for(const i of members){const s=segments[i]!,a=partPoint(mesh,samples[s[0]]!),b=partPoint(mesh,samples[s[1]]!),w=length(sub(a,b));total+=w;for(let k=0;k<3;k++)center[k]!+=(a[k]!+b[k]!)*w/2;}
   const section:PartSection={id:sections.length,component:faceRoot(lineage[0]!),level,samples:order.map(v=>samples[v]!),segments:members.map(i=>{const s=segments[i]!;return [local.get(s[0])!,local.get(s[1])!,s[2]] as const;}),faces:lineage,closed,center:center.map(v=>v/(total||1)) as unknown as Point};
   sections.push(section);sectionStation.push(station);current.push(section);yield;
  }
  // Preserve annular/through-hole boundaries. Nesting classifies sections,
  // not source components as extra solid limbs.
  const chart=(s:PartSection)=>s.samples.map(p=>{const q=partPoint(mesh,p);return [dot(q,basis),dot(q,vertical)] as const;});
  const polygons=new Map(current.filter(s=>s.closed).map(s=>[s.id,chart(s)]));
  const contains=(poly:readonly (readonly [number,number])[],q:readonly [number,number])=>{let yes=false;for(let i=0,j=poly.length-1;i<poly.length;j=i++){const a=poly[i]!,b=poly[j]!;if((a[1]>q[1])!==(b[1]>q[1])&&q[0]<(b[0]-a[0])*(q[1]-a[1])/(b[1]-a[1])+a[0])yes=!yes;}return yes;};
  const area=(poly:readonly (readonly [number,number])[])=>Math.abs(poly.reduce((sum,p,i)=>{const q=poly[(i+1)%poly.length]!;return sum+p[0]*q[1]-q[0]*p[1];},0));
  for(let i=0;i<current.length;i++){const s=current[i]!,poly=polygons.get(s.id);if(poly?.length){const outer=current.filter(t=>t!==s&&t.component===s.component&&polygons.has(t.id)&&contains(polygons.get(t.id)!,poly[0]!)).sort((a,b)=>area(polygons.get(a.id)!)-area(polygons.get(b.id)!))[0];if(outer){const nested={...s,inside:outer.id};sections[s.id]=nested;current[i]=nested;}}yield;}
  stations.push(current);
 }
 const bands:PartBand[]=[];
 const edgeList=[...edges.values()],edgeLow=new Float64Array(edgeList.length),edgeHigh=new Float64Array(edgeList.length);
 for(const [i,edge]of edgeList.entries()){edgeLow[i]=Math.min(height[edge.a]!,height[edge.b]!);edgeHigh[i]=Math.max(height[edge.a]!,height[edge.b]!);yield;}
 const slabFaces=faceSpans(),slabEdges=spanIndex(edgeLow,edgeHigh)();
 const slabParent=new Int32Array(faceCount).fill(-1);let previousFaces:readonly number[]=[];
 // Each slab connects sections by ORIGINAL surface incidence. A projected
 // crossing or two nearby section centres cannot manufacture a fork.
 for(let station=0;station+1<stations.length;station++){
  const lower=stations[station]!,upper=stations[station+1]!,lo=lower[0]?.level,hi=upper[0]?.level;if(lo===undefined||hi===undefined)continue;
  const activeFaces=yield* slabFaces(lo,hi);
  for(const f of previousFaces)slabParent[f]=-1;
  for(const f of activeFaces)slabParent[f]=f;
  previousFaces=activeFaces;
  const root=(f:number):number=>{let r=f;while(slabParent[r]!==r)r=slabParent[r]!;while(slabParent[f]!==f){const n=slabParent[f]!;slabParent[f]=r;f=n;}return r;};
  for(const index of yield* slabEdges(lo,hi)){
   const edge=edgeList[index]!;
   if(Math.max(height[edge.a]!,height[edge.b]!)>lo&&Math.min(height[edge.a]!,height[edge.b]!)<hi){const active=edge.faces.filter(f=>slabParent[f]!>=0);for(let i=1;i<active.length;i++){const a=root(active[0]!),b=root(active[i]!);slabParent[Math.max(a,b)]=Math.min(a,b);}}yield;
  }
  const groups=new Map<number,{lower:Set<number>;upper:Set<number>;faces:number[];vertices:Set<number>}>();
  for(const f of activeFaces){if(slabParent[f]!>=0){const id=root(f),group=groups.get(id)??{lower:new Set(),upper:new Set(),faces:[],vertices:new Set()};group.faces.push(f);for(const v of triangles[f]!)if(critical.has(height[v]!)&&height[v]!>lo&&height[v]!<hi)group.vertices.add(v);groups.set(id,group);}yield;}
  for(const [side,list]of [['lower',lower],['upper',upper]] as const)for(const s of list)for(const f of s.faces)if(slabParent[f]!>=0)groups.get(root(f))![side].add(s.id);
  for(const group of groups.values()){if(group.lower.size||group.upper.size)bands.push({id:bands.length,lower:[...group.lower],upper:[...group.upper],faces:group.faces,criticalVertices:[...group.vertices].sort((a,b)=>a-b)});yield;}
 }
 // Part labels: which chains of sections are limbs, where they fork, and which
 // free ends are roots or caps. Decided once from the section graph and rest
 // geometry; no camera, name or glyph takes part.
 const above:number[][]=sections.map(()=>[]),below:number[][]=sections.map(()=>[]);
 for(const band of bands){for(const s of band.lower)above[s]!.push(band.id);for(const s of band.upper)below[s]!.push(band.id);}
 const degree=(band:PartBand)=>band.lower.length+band.upper.length;
 const chainParent=Uint32Array.from({length:sections.length},(_,i)=>i);
 const chainRoot=(s:number):number=>{let r=s;while(chainParent[r]!==r)r=chainParent[r]!;while(chainParent[s]!==s){const n=chainParent[s]!;chainParent[s]=r;s=n;}return r;};
 for(const band of bands)if(band.lower.length===1&&band.upper.length===1){const a=chainRoot(band.lower[0]!),b=chainRoot(band.upper[0]!);chainParent[Math.max(a,b)]=Math.min(a,b);}
 // The support plane is the lowest source Y, the up the support test reads.
 let floor=Infinity,ceiling=-Infinity;for(const v of used){floor=Math.min(floor,p[v*3+1]!);ceiling=Math.max(ceiling,p[v*3+1]!);}
 const support=floor+Math.max(1e-6,(ceiling-floor)*PART_ROOT_PLANE);
 const neighbors:number[][]=Array.from({length:faceCount},()=>[]);
 for(const edge of edgeList){for(const a of edge.faces)for(const b of edge.faces)if(a!==b)neighbors[a]!.push(b);yield;}
 // A free end owns the surface beyond its section: the faces wholly on that
 // side that connect to it (the tip of a branch, the foot of a trunk).
 const reachesSupport=function*(section:PartSection,side:-1|1):Generator<void,boolean>{
  const seen=new Set(section.faces),pending=[...section.faces];
  while(pending.length){
   const face=pending.pop()!;
   // Only corners on this end's side of the section count: a short post's
   // one section spans both of its ends.
   for(let k=0;k<3;k++){const v=ix[face*3+k]!,h=height[representatives[v]!]!;if((side<0?h<=section.level:h>=section.level)&&p[v*3+1]!<=support)return true;}
   for(const next of neighbors[face]!)if(!seen.has(next)&&(side<0?faceHigh[next]!<=section.level:faceLow[next]!>=section.level)){seen.add(next);pending.push(next);}
   yield;
  }
  return false;
 };
 const ends=new Uint8Array(sections.length),bandEnds=new Uint8Array(bands.length);
 for(const section of sections){
  for(const [side,list]of [[-1,below[section.id]!],[1,above[section.id]!]] as const){
   if(!list.every(id=>degree(bands[id]!)===1))continue;
   const label=(yield* reachesSupport(section,side))?PART_ROOT:PART_CAP;ends[section.id]!|=label;for(const id of list)bandEnds[id]!|=label;
  }
 }
 const members=new Map<number,number[]>();for(const section of sections){const key=chainRoot(section.id),list=members.get(key);if(list)list.push(section.id);else members.set(key,[section.id]);}
 const chainOf=new Int32Array(sections.length).fill(-1),chains:PartChain[]=[];
 for(const list of members.values()){
  const id=chains.length,ordered=[...list].sort((a,b)=>sections[a]!.level-sections[b]!.level||a-b);let axial=0,path=0,width=0,tube=true,endBits=0;
  for(const [i,s]of ordered.entries()){
   const section=sections[s]!,station=sectionStation[s]!;chainOf[s]=id;endBits|=ends[s]!;
   axial+=levels[station+1]!-levels[station]!;if(i)path+=length(sub(section.center,sections[ordered[i-1]!]!.center));
   tube&&=section.closed&&section.inside===undefined;
   let radius=0;for(const sample of section.samples)radius+=length(sub(partPoint(mesh,sample),section.center));width+=section.samples.length?2*radius/section.samples.length:0;
   yield;
  }
  width/=ordered.length;const extent=Math.max(axial,path),limb=tube&&width>0&&extent>=PART_LIMB_ELONGATION*width;
  chains.push({id,sections:ordered,bands:[],labels:(limb?PART_LIMB:0)|endBits,length:extent,width});
 }
 const chainWord=(chain:number)=>(((chain+1)<<PART_CHAIN_SHIFT)|(chains[chain]!.labels&PART_LIMB))>>>0;
 for(const section of sections)sections[section.id]={...section,labels:(chainWord(chainOf[section.id]!)|ends[section.id]!)>>>0};
 for(const band of bands){
  // A fork belongs to no single chain; a link or a tip belongs to the chain
  // of its one section.
  const owner=degree(band)>=3?-1:band.lower.length===1&&band.upper.length<=1?chainOf[band.lower[0]!]!:band.upper.length===1&&!band.lower.length?chainOf[band.upper[0]!]!:-1;
  if(owner>=0)(chains[owner]!.bands as number[]).push(band.id);
  bands[band.id]={...band,labels:((degree(band)>=3?PART_FORK:0)|(owner>=0?chainWord(owner):0)|bandEnds[band.id]!)>>>0};
 }
 return {axis,origin,sections,bands,levels,vertexRepresentatives:representatives,chains};
}

/** Static source work cooperates with the host. It never holds a render tick
 * until a whole complex statue is compiled, and never substitutes a new pose.
 * Callers own the geometry-revision cache and expose the result only when the
 * complete graph is ready. This is not a second frame publisher or registry. */
export async function compileMeshParts(mesh:PartMesh,signal?:AbortSignal):Promise<MeshPartGraph>{
 const work=meshPartWork(mesh);
 for(;;){const until=performance.now()+4;let operations=0;
  if(signal?.aborted)throw Error('Source part compilation retired');
  do{const next=work.next();if(next.done)return next.value;operations++;}while(operations<4096&&performance.now()<until);
  await new Promise<void>(resolve=>setTimeout(resolve,0));
 }
}
