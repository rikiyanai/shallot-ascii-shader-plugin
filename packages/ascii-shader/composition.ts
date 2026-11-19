import {compileSourceMaterialClass,compileSourceStyleWord,type SourceMaterialKey} from './source-material.ts';
import {selectFeatureGraph,selectContourGraph} from './feature-graph.ts';
import {relationPoint,type PosedGraph,type PosedRelation,type PosedOwner} from './posed-graph.ts';
import type {DrawingFieldView} from './field-readback.ts';
import type {MeshPartGraph} from './mesh-part-graph.ts';
import {compileSourceSheet,isTerrainSheet,planeRegionLoops,type SourceSheet} from './source-complex.ts';
import type {CompiledSourceRunProgram,CpuCompiledSourceRunProgram} from './compiled-construction-topology.ts';
import type {SourceConstruction} from './source-construction.ts';

export interface CompositionGeometry {
 readonly index?:number;readonly primary?:boolean;readonly owner:string;readonly positions:Float32Array;readonly indices:Uint32Array;readonly revision:number;
 readonly worldFromSource?:Float32Array;
 /** Immutable mesh definition for a GPU-owned instance. Its current pose is
  * in PosedGraph, NOT an implicit identity CPU transform. */
 readonly sourceOnly?:true;
 /** Adapter fact: the actual vertex stage is exactly this source/model
  * transform, without another GPU deformation. Never inferred from a name. */
 readonly poseAuthority?:'source-model';
 /** Optional original per-corner normals, xyz + padding (four floats). */
 readonly normals?:Float32Array;
 /** Immutable producer fact: a generated heightfield keeps its terrain
  * construction on steep/flat cells; liquid takes water bands. Neither is
  * inferred from a mesh name or radiance colour. */
 readonly surface?:'liquid'|'heightfield';
 /** Immutable authored asset/material identity; a change is a new source revision. */
 readonly materialSource?:SourceMaterialKey;
 /** Producer-authored part structure, preserved with this source revision. */
 readonly construction?:SourceConstruction;
 readonly lineage?:{readonly source:string;readonly faces:Uint32Array;readonly splits?:Uint32Array;readonly materials?:Uint32Array;readonly adjacency?:{readonly offsets:Uint32Array;readonly faces:Uint32Array}};
}
type V=readonly number[];
interface SupportPart {g:CompositionGeometry;points:number[][];low:number[];high:number[];index:number}
interface SupportNode {low:number[];high:number[];left?:SupportNode;right?:SupportNode;parts?:readonly SupportPart[]}
interface SupportFaces {low:number[];high:number[];left?:SupportFaces;right?:SupportFaces;faces?:Uint32Array}
/** The triangle index is immutable with its source revision. A receiver is
 * not its highest point: a foot can rest anywhere on its actual surface. */
function supportFaces(g:{positions:ArrayLike<number>;indices:Uint32Array}):SupportFaces{
 const bounds=new Float64Array(g.indices.length/3*6),p=g.positions;
 for(let face=0;face<g.indices.length/3;face++)for(let axis=0;axis<3;axis++){
  const a=p[g.indices[face*3]!*3+axis]!,b=p[g.indices[face*3+1]!*3+axis]!,c=p[g.indices[face*3+2]!*3+axis]!;
  bounds[face*6+axis]=Math.min(a,b,c);bounds[face*6+3+axis]=Math.max(a,b,c);
 }
 const build=(faces:Uint32Array):SupportFaces=>{
  const low=[Infinity,Infinity,Infinity],high=[-Infinity,-Infinity,-Infinity];
  for(const face of faces)for(let k=0;k<3;k++){low[k]=Math.min(low[k]!,bounds[face*6+k]!);high[k]=Math.max(high[k]!,bounds[face*6+3+k]!);}
  if(faces.length<=8)return {low,high,faces};
  let axis=0;for(let k=1;k<3;k++)if(high[k]!-low[k]!>high[axis]!-low[axis]!)axis=k;
  faces.sort((a,b)=>bounds[a*6+axis]!+bounds[a*6+3+axis]!-bounds[b*6+axis]!-bounds[b*6+3+axis]!||a-b);
  const middle=faces.length>>>1;return {low,high,left:build(faces.subarray(0,middle)),right:build(faces.subarray(middle))};
 };
 return build(Uint32Array.from({length:g.indices.length/3},(_,i)=>i));
}
function receivingFaces(node:SupportFaces,point:V,epsilon:V,found:number[]):void{
 for(let k=0;k<3;k++)if(point[k]!<node.low[k]!-epsilon[k]!||point[k]!>node.high[k]!+epsilon[k]!)return;
 if(node.faces){for(const face of node.faces)found.push(face);return;}
 receivingFaces(node.left!,point,epsilon,found);receivingFaces(node.right!,point,epsilon,found);
}
function supportTree(parts:readonly SupportPart[]):SupportNode{
 const low=[Infinity,Infinity,Infinity],high=[-Infinity,-Infinity,-Infinity];
 for(const part of parts)for(let k=0;k<3;k++){low[k]=Math.min(low[k]!,part.low[k]!);high[k]=Math.max(high[k]!,part.high[k]!);}
 if(parts.length<=8)return {low,high,parts};
 let axis=0;for(let k=1;k<3;k++)if(high[k]!-low[k]!>high[axis]!-low[axis]!)axis=k;
 const ordered=[...parts].sort((a,b)=>(a.low[axis]!+a.high[axis]!)-(b.low[axis]!+b.high[axis]!)||a.index-b.index),middle=ordered.length>>1;
 return {low,high,left:supportTree(ordered.slice(0,middle)),right:supportTree(ordered.slice(middle))};
}
function supportCandidates(node:SupportNode,part:SupportPart,epsilon:number,found:SupportPart[]):void{
 if(part.low[1]!<node.low[1]!-epsilon||part.low[1]!>node.high[1]!+epsilon||part.low[0]!>node.high[0]!+epsilon||part.high[0]!<node.low[0]!-epsilon||part.low[2]!>node.high[2]!+epsilon||part.high[2]!<node.low[2]!-epsilon)return;
 if(node.parts){found.push(...node.parts);return;}
 supportCandidates(node.left!,part,epsilon,found);supportCandidates(node.right!,part,epsilon,found);
}
/** `'terrain'` is a height surface (rule at Composition.analyze; thresholds
 * TERRAIN_RULE in source-complex.ts). */
export type ConstructionType='path-led'|'planar'|'rounded'|'far-strip'|'paper'|'terrain';
export interface CompiledConstructionGraph {readonly styleWord?:number;readonly type:ConstructionType;readonly structure?:MeshPartGraph;readonly planeRegions?:readonly SourcePlaneRegion[];
 /** Worker-compiled runtime input. Legacy CPU construction does not need it. */
 readonly runProgram?:CompiledSourceRunProgram;
 /** Unit face normals (3 per face, rest pose). A source edge is a hard line
  * when its faces meet at a crease or it is an open boundary, whatever the
  * whole mesh's class; compiled once at residency. */
 readonly faceNormals?:Float32Array;
 /** A foliage crown (foliageMass): its shade takes the FOLIAGE tone words
  * (tone-field.ts foliageOf). Decided once per source revision from the
  * source's own geometry, never per frame or per screen row. */
 readonly foliage?:boolean;
 /** An organic surface that is not a crown (a boulder, a rock: foliageMass):
  * its shade takes the round's rim band, not plane bands (tone-field.ts
  * organicOf); it has no plane-region tone table. */
 readonly organic?:boolean;
 /** The plane regions' dark-boundary table for region tone (compileToneRegions). */
 readonly toneRegions?:ToneRegions;
 /** Wall faces of a built form (compileWallFaces), packed into the run
  * topology slot as the face owner's edge table. Empty for every other source. */
 readonly faces?:readonly SourceFace[]}
/** A WALL FACE (face construction slice 1, README "Whole-part construction for
 * planar subjects", 2026-11-17): plane regions within FACE_PLANE_DEGREES and
 * FACE_OFFSET of one vertical plane (a facade: the wall, its timbers, trims and
 * door frame), holding at least FACE_LEAST_SHARE of the source's surface,
 * whose largest region spans at least two member widths both ways (a fence
 * rail is no wall).
 * `regions` are indices into planeRegions, `outline` the regions whose loops
 * are the face's outline and openings, `members` its thin strips (smaller
 * in-plane extent under FACE_OFFSET), `corners` the welded source vertices
 * where the main region's outline turns 20 degrees or more (its long edges
 * joined at cos .985). Faces are numbered by their least plane-region id, so
 * the id is stable for a source revision. Compiled once, in the worker. */
export interface SourceFace {readonly id:number;readonly regions:readonly number[];readonly main:number;readonly members:readonly number[];readonly corners:readonly number[];readonly openings:number;readonly share:number;
 /** The member width of the source (FACE_OFFSET of its radius, source units). */
 readonly width:number;
 /** SLICE 1b (perf-impl-81): the face's edge decisions, compiled once.
  * `drawn`: source halfedges the face construction draws (straight in the
  * face plane: level or upright within FACE_AXIS_DEGREES). `hidden`: face
  * halfedges compiled out, drawn by nothing (neither the face run nor a
  * per-edge run): the second long side and the ends of a strip (one timber,
  * one line), the outer sides of a frame's members (the frame is its inner
  * loop), an opening side a jamb carries, and every edge of a return (a
  * timber's or reveal's depth side inside the facade slab). A slanted edge of
  * a face (a gable slope) is in neither list and keeps today's per-edge run. */
 readonly drawn:readonly number[];readonly hidden:readonly number[];
 /** Closed frames: a door (an opening loop of the main region) or a recess
  * bounded by two upright and two level timbers (a window). cell = the inner
  * loop [u0,u1,v0,v1] in the face plane; sides = the region drawing its left,
  * right, bottom and top side (-1: the opening loop's own edge). */
 readonly frames:readonly FaceFrame[];
 /** Regions outside every face lying inside this face's slab: the depth sides of its timbers and reveals. */
 readonly returns:readonly number[]}
export interface FaceFrame {readonly cell:readonly number[];readonly sides:readonly number[];readonly opening:boolean}
export const FACE_AXIS_DEGREES=6;
export const FACE_PLANE_DEGREES=8,FACE_OFFSET=.11,FACE_LEAST_SHARE=.03,FACE_WALL_NORMAL_Y=.2;
/** THE CLASS GATE OF THE FACE OWNER (slice 1). A source reaches the face owner
 * when it is a built form, not by its plane count: its class stays what the
 * fixed class rule below decides (so terrain, voxel landscapes, rounds and
 * planar volumes classify exactly as before), and only a path-led source that
 * is neither foliage nor organic, has more plane regions than the box owner
 * takes (24), and compiles at least one wall face gets a face table. A face
 * is a facade, not a region: the house's 150 regions are 11 major faces. */
export function compileWallFaces(positions:Float32Array,indices:Uint32Array,weld:Uint32Array,faceNormals:Float32Array,planeRegions:readonly SourcePlaneRegion[],radius:number):SourceFace[]{
 const P=(v:number)=>[positions[v*3]!,positions[v*3+1]!,positions[v*3+2]!];
 const areaOf=(f:number)=>{const a=P(indices[f*3]!),b=P(indices[f*3+1]!),c=P(indices[f*3+2]!);return Math.hypot(...cross(sub(b,a),sub(c,a)))/2;};
 let total=0;for(let f=0;f<indices.length/3;f++)total+=areaOf(f);if(!(total>0))return [];
 const w=FACE_OFFSET*radius,planeCos=Math.cos(FACE_PLANE_DEGREES*Math.PI/180);
 const info=planeRegions.map(region=>{const f0=region.faces[0]!,n=[faceNormals[f0*3]!,faceNormals[f0*3+1]!,faceNormals[f0*3+2]!];
  return {n,d:dot(n,P(indices[f0*3]!)),area:region.faces.reduce((s,f)=>s+areaOf(f),0),wall:Math.abs(n[1]!)<FACE_WALL_NORMAL_Y};});
 const parent=info.map((_,i)=>i),find=(i:number):number=>parent[i]===i?i:(parent[i]=find(parent[i]!));
 for(let a=0;a<info.length;a++)if(info[a]!.wall)for(let b=a+1;b<info.length;b++){const A=info[a]!,B=info[b]!;
  if(!B.wall||dot(A.n,B.n)<planeCos||Math.abs(A.d-B.d)>w)continue;const ra=find(a),rb=find(b);if(ra!==rb)parent[Math.max(ra,rb)]=Math.min(ra,rb);}
 const groups=new Map<number,number[]>();info.forEach((r,i)=>{if(!r.wall)return;const k=find(i),list=groups.get(k);if(list)list.push(i);else groups.set(k,[i]);});
 type Pending=Omit<SourceFace,'id'|'drawn'|'hidden'|'frames'|'returns'>&{axes:number[][]};
 const faces:Pending[]=[];
 for(const members of groups.values()){
  const area=members.reduce((s,i)=>s+info[i]!.area,0);if(area/total<FACE_LEAST_SHARE)continue;
  const main=[...members].sort((a,b)=>info[b]!.area-info[a]!.area||planeRegions[a]!.id-planeRegions[b]!.id)[0]!;
  // In-plane extent of a region's outline on the face's own axes.
  const n=info[main]!.n,u=unit(cross(Math.abs(n[1]!)<.9?[0,1,0]:[1,0,0],n)),v=cross(n,u);
  const narrow=(i:number)=>{const loop=planeRegions[i]!.boundary??[];if(!loop.length)return 0;let lu=Infinity,hu=-Infinity,lv=Infinity,hv=-Infinity;
   for(const h of loop){const q=P(indices[h]!);lu=Math.min(lu,dot(q,u));hu=Math.max(hu,dot(q,u));lv=Math.min(lv,dot(q,v));hv=Math.max(hv,dot(q,v));}
   return Math.min(hu-lu,hv-lv);};
  const thin=(i:number)=>{const loop=planeRegions[i]!.boundary??[];return loop.length>0&&narrow(i)<w;};
  // Corners of the main region's outline: collinear halfedges joined into
  // long edges (cos .985, the perf-74 threshold), a corner where two meet
  // turning 20 degrees or more, keyed by the welded source vertex.
  const loop=planeRegions[main]!.boundary??[],pts=loop.map(h=>({v:weld[indices[h]!]!,p:P(indices[h]!)}));
  const edges:{a:number;b:number;d:number[]}[]=[];
  for(let i=0;i<pts.length;i++){const a=pts[i]!,b=pts[(i+1)%pts.length]!,d=sub(b.p,a.p),l=Math.hypot(...d);if(l<1e-9)continue;const e=unit(d),last=edges[edges.length-1];
   if(last&&dot(last.d,e)>=.985){last.b=b.v;last.d=unit(sub(b.p,P(last.a)));}else edges.push({a:a.v,b:b.v,d:e});}
  if(edges.length>1&&dot(edges[0]!.d,edges[edges.length-1]!.d)>=.985){const first=edges.shift()!;edges[edges.length-1]!.b=first.b;}
  // A face is a wall, not a rail or a plank: its main region spans at least
  // two member widths both ways (a fence rail, .6 of R 4.78, does not).
  if(narrow(main)<2*w)continue;
  const corners=edges.length>1?edges.flatMap((e,k)=>dot(e.d,edges[(k+1)%edges.length]!.d)<=Math.cos(20*Math.PI/180)?[e.b]:[]):[];
  // The face's own axes (slice 1b): its longest outline edge is level or
  // upright, so a house standing rolled on a slope keeps its jambs upright
  // and its lintels level in its own plane (a bank instance carries its pose).
  let longest=edges[0],length=0;for(const e of edges){const l=Math.hypot(...sub(P(e.b),P(e.a)));if(l>length){length=l;longest=e;}}
  let fu=u,fv=v;
  if(longest){const e=unit(sub(P(longest.b),P(longest.a)));
   if(Math.abs(dot(e,v))>=Math.abs(dot(e,u))){fv=dot(e,v)<0?e.map(x=>-x):e;fu=unit(cross(fv,n));}
   else{fu=dot(e,u)<0?e.map(x=>-x):e;fv=cross(n,fu);}}
  faces.push({regions:[...members].sort((a,b)=>a-b),main,members:members.filter(thin).sort((a,b)=>a-b),corners,openings:planeRegions[main]!.openings?.length??0,share:area/total,width:w,axes:[n,fu,fv]});
 }
 faces.sort((a,b)=>Math.min(...a.regions.map(i=>planeRegions[i]!.id))-Math.min(...b.regions.map(i=>planeRegions[i]!.id)));
 const inFace=new Set<number>();for(const f of faces)for(const i of f.regions)inFace.add(i);
 const regionNormal=(i:number)=>{const f0=planeRegions[i]!.faces[0]!;return [faceNormals[f0*3]!,faceNormals[f0*3+1]!,faceNormals[f0*3+2]!];};
 const loopVertices=(i:number)=>{const R=planeRegions[i]!;return [...(R.boundary??[]),...(R.openings??[]).flat()].map(h=>P(indices[h]!));};
 const claimed=new Set<number>();
 return faces.map(({axes,...face},id)=>{
  const [n,u,v]=axes as [number[],number[],number[]];
  const decisions=compileFaceDecisions(face,n,u,v);
  // RETURNS: a region outside every face, standing across this face's plane
  // (|n.nf| < FACE_WALL_NORMAL_Y), no deeper than one member width along the
  // face normal, inside the face's slab and extent: a post's or a beam's
  // depth side, a door or window reveal. Its edges are compiled out.
  const verts=face.regions.flatMap(i=>loopVertices(i));
  const depth=verts.map(q=>dot(q,n)),us=verts.map(q=>dot(q,u)),vs=verts.map(q=>dot(q,v));
  const d0=Math.min(...depth)-w,d1=Math.max(...depth)+w,u0=Math.min(...us)-w,u1=Math.max(...us)+w,v0=Math.min(...vs)-w,v1=Math.max(...vs)+w;
  const returns:number[]=[];
  for(let i=0;i<planeRegions.length;i++){
   if(inFace.has(i)||claimed.has(i)||Math.abs(dot(regionNormal(i),n))>=FACE_WALL_NORMAL_Y)continue;
   const q=loopVertices(i);if(!q.length)continue;const qd=q.map(x=>dot(x,n));
   if(Math.max(...qd)-Math.min(...qd)>w||q.some((x,k)=>qd[k]!<d0||qd[k]!>d1||dot(x,u)<u0||dot(x,u)>u1||dot(x,v)<v0||dot(x,v)>v1))continue;
   returns.push(i);claimed.add(i);
  }
  const drawn=new Set(decisions.drawn),hidden=new Set<number>();
  for(const h of decisions.hidden)if(!drawn.has(h))hidden.add(h);
  for(const i of returns)for(const h of [...(planeRegions[i]!.boundary??[]),...(planeRegions[i]!.openings??[]).flat()])if(!drawn.has(h))hidden.add(h);
  return {id,...face,drawn:[...drawn].sort((a,b)=>a-b),hidden:[...hidden].sort((a,b)=>a-b),frames:decisions.frames,returns};
 });
 /** One face's edge decisions in its own plane (u level, v up). Fixed rules,
  * nothing scored or tried: a TIMBER is a region (not the main one) narrower
  * than two member widths; an upright timber is taller than wide.
  * FRAMES: (a) each opening loop of the main region takes the timber whose
  * span holds each of its four sides (a jamb, the lintel, the sill); a side
  * with no timber is the opening's own edge. (b) two neighbouring upright
  * timbers at least one member width apart, with two level timbers spanning
  * the gap between them, bound a recess (a window): consecutive level
  * timbers (by height) with a member width of wall between them.
  * A frame member draws only its side facing the cell; a thin strip outside
  * every frame draws one line (a level strip its lower side, an upright strip
  * its side away from the face centre); every other region draws all its
  * straight edges. Slanted edges stay per-edge runs. */
 function compileFaceDecisions(face:Omit<Pending,'axes'>,n:number[],u:number[],v:number[]){
  const axis=Math.tan(FACE_AXIS_DEGREES*Math.PI/180);
  type Rect={lu:number;hu:number;lv:number;hv:number};
  const rect=(loop:readonly number[]):Rect=>{let lu=Infinity,hu=-Infinity,lv=Infinity,hv=-Infinity;
   for(const h of loop){const q=P(indices[h]!);lu=Math.min(lu,dot(q,u));hu=Math.max(hu,dot(q,u));lv=Math.min(lv,dot(q,v));hv=Math.max(hv,dot(q,v));}return {lu,hu,lv,hv};};
  const R=new Map<number,Rect>();for(const i of face.regions)R.set(i,rect(planeRegions[i]!.boundary??[]));
  const r=(i:number)=>R.get(i)!;
  const timber=(i:number)=>i!==face.main&&Math.min(r(i).hu-r(i).lu,r(i).hv-r(i).lv)<2*w;
  const upright=(i:number)=>r(i).hv-r(i).lv>r(i).hu-r(i).lu;
  const verticals=face.regions.filter(i=>timber(i)&&upright(i)).sort((a,b)=>r(a).lu-r(b).lu||a-b);
  const levels=face.regions.filter(i=>timber(i)&&!upright(i));
  const overlap=(a0:number,a1:number,b0:number,b1:number)=>Math.min(a1,b1)-Math.max(a0,b0);
  const holds=(lo:number,hi:number,x:number)=>x>=lo-.25*w&&x<=hi+.25*w;
  const pick=(list:number[],ok:(i:number)=>boolean,centre:(i:number)=>number,x:number)=>list.filter(ok).sort((a,b)=>Math.abs(centre(a)-x)-Math.abs(centre(b)-x)||a-b)[0]??-1;
  const frames:FaceFrame[]=[];const openingSides=new Map<number,string>();
  const main=planeRegions[face.main]!;
  for(const loop of main.openings??[]){const o=rect(loop);
   const cu=(i:number)=>(r(i).lu+r(i).hu)/2,cv=(i:number)=>(r(i).lv+r(i).hv)/2;
   const vOk=(i:number)=>overlap(r(i).lv,r(i).hv,o.lv,o.hv)>=.5*(o.hv-o.lv),uOk=(i:number)=>overlap(r(i).lu,r(i).hu,o.lu,o.hu)>=.5*(o.hu-o.lu);
   const L=pick(verticals,i=>vOk(i)&&holds(r(i).lu,r(i).hu,o.lu),cu,o.lu),Rt=pick(verticals,i=>i!==L&&vOk(i)&&holds(r(i).lu,r(i).hu,o.hu),cu,o.hu);
   const B=pick(levels,i=>uOk(i)&&holds(r(i).lv,r(i).hv,o.lv),cv,o.lv),T=pick(levels,i=>i!==B&&uOk(i)&&holds(r(i).lv,r(i).hv,o.hv),cv,o.hv);
   frames.push({cell:[L>=0?r(L).hu:o.lu,Rt>=0?r(Rt).lu:o.hu,B>=0?r(B).hv:o.lv,T>=0?r(T).lv:o.hv],sides:[L,Rt,B,T],opening:true});
   // The opening loop's own side is drawn only where no timber carries it.
   for(const h of loop){const s=sideOf(h,o);if(s)openingSides.set(h,s);}
   for(const [h,s] of [...openingSides])if(loop.includes(h)&&((s==='L'&&L>=0)||(s==='R'&&Rt>=0)||(s==='B'&&B>=0)||(s==='T'&&T>=0)))openingSides.set(h,'hidden');
  }
  for(let k=0;k<verticals.length;k++){const L=verticals[k]!;
   // The neighbour: the next upright timber to the right sharing half its height.
   const Rt=verticals.find(i=>i!==L&&r(i).lu>=r(L).hu-.25*w&&overlap(r(i).lv,r(i).hv,r(L).lv,r(L).hv)>=.5*Math.min(r(i).hv-r(i).lv,r(L).hv-r(L).lv));
   // A recess is a window: two member widths each way. A narrower wall strip
   // (between a corner post and a jamb) is wall, not a frame.
   if(Rt===undefined||r(Rt).lu-r(L).hu<2*w)continue;
   const a=r(L).hu,b=r(Rt).lu,lo=Math.max(r(L).lv,r(Rt).lv),hi=Math.min(r(L).hv,r(Rt).hv);
   const spans=levels.filter(i=>r(i).lu<=a+w&&r(i).hu>=b-w&&(r(i).lv+r(i).hv)/2>=lo-w&&(r(i).lv+r(i).hv)/2<=hi+w).sort((x,y)=>(r(x).lv+r(x).hv)-(r(y).lv+r(y).hv)||x-y);
   for(let j=0;j+1<spans.length;j++){const B=spans[j]!,T=spans[j+1]!;const c=[a,b,r(B).hv,r(T).lv];
    if(c[3]!-c[2]!<2*w||Math.min(hi,c[3]!)-Math.max(lo,c[2]!)<.8*(c[3]!-c[2]!))continue;
    if(frames.some(f=>overlap(f.cell[0]!,f.cell[1]!,c[0]!,c[1]!)>.5*(c[1]!-c[0]!)&&overlap(f.cell[2]!,f.cell[3]!,c[2]!,c[3]!)>.5*(c[3]!-c[2]!)))continue;
    frames.push({cell:c,sides:[L,Rt,B,T],opening:false});
   }
  }
  // Side of a halfedge on a rectangle: L/R upright, B/T level, '' slanted.
  function sideOf(h:number,box:Rect):string{
   const a=P(indices[h]!),b=P(indices[Math.floor(h/3)*3+(h%3+1)%3]!);
   const du=dot(b,u)-dot(a,u),dv=dot(b,v)-dot(a,v);
   if(Math.abs(du)<=axis*Math.abs(dv)){const m=(dot(a,u)+dot(b,u))/2;return Math.abs(m-box.lu)<=Math.abs(m-box.hu)?'L':'R';}
   if(Math.abs(dv)<=axis*Math.abs(du)){const m=(dot(a,v)+dot(b,v))/2;return Math.abs(m-box.lv)<=Math.abs(m-box.hv)?'B':'T';}
   return '';
  }
  const facing=new Map<number,Set<string>>();
  for(const f of frames){const [L,Rt,B,T]=f.sides as number[];
   for(const [i,s] of [[L,'R'],[Rt,'L'],[B,'T'],[T,'B']] as [number,string][])if(i>=0){const set=facing.get(i)??new Set<string>();set.add(s);facing.set(i,set);}}
  const thin=new Set(face.members);const cx=(r(face.main).lu+r(face.main).hu)/2;
  // The face's outer extent: a side lying on it is the face outline (a
  // corner post's outer side, the base, the wall top) and is always drawn.
  const all=[...R.values()],ext={lu:Math.min(...all.map(b=>b.lu)),hu:Math.max(...all.map(b=>b.hu)),lv:Math.min(...all.map(b=>b.lv)),hv:Math.max(...all.map(b=>b.hv))};
  const onOutline=(box:Rect,s:string)=>s==='L'?box.lu<=ext.lu+.25*w:s==='R'?box.hu>=ext.hu-.25*w:s==='B'?box.lv<=ext.lv+.25*w:box.hv>=ext.hv-.25*w;
  const drawn:number[]=[],hidden:number[]=[];
  for(const i of face.regions){const box=r(i),region=planeRegions[i]!;
   const frame=facing.get(i);
   const strip=!frame&&thin.has(i)?(upright(i)?((box.lu+box.hu)/2<cx?'L':'R'):'B'):'';
   for(const h of region.boundary??[]){const s=sideOf(h,box);if(!s)continue;
    if(onOutline(box,s)||(frame?frame.has(s):strip?s===strip:true))drawn.push(h);else hidden.push(h);}
   if(i===face.main)for(const [h,s] of openingSides){if(s==='hidden')hidden.push(h);else drawn.push(h);}
   else for(const h of (region.openings??[]).flat()){if(sideOf(h,rect(region.openings!.find(l=>l.includes(h))!)))drawn.push(h);}
  }
  return {drawn,hidden,frames};
 }
}
/** THE PLANE REGIONS' BOUNDARIES FOR TONE, compiled once per source revision
 * in the source frame (tone-field.ts darkEdgeDistance reads them on the GPU).
 * regions: 8 floats per plane region: its plane (unit normal xyz, offset),
 * its first edge and edge count, its radius (centroid of its boundary to its
 * farthest boundary vertex), 0. edges: 12 floats per boundary or opening
 * edge: end a (xyz, 0), end b (xyz, 0), the normal of the face across the
 * edge outside the region (xyz) and 1 when no face is there (an open edge).
 * Only turn edges are listed (the across face turned 30 degrees or more, or
 * none), at most TONE_REGION_EDGES per region; the radius counts every
 * boundary vertex. */
export interface ToneRegions {readonly regions:Float32Array;readonly edges:Float32Array}
export const TONE_REGION_LIMIT=512,TONE_REGION_EDGES=24;
function compileToneRegions(positions:Float32Array,indices:Uint32Array,weld:Uint32Array,faceNormals:Float32Array,planeRegions:readonly SourcePlaneRegion[]):ToneRegions|undefined{
 if(!planeRegions.length)return undefined;
 const vertexCount=Math.floor(positions.length/3),key=(a:number,b:number)=>Math.min(a,b)*vertexCount+Math.max(a,b);
 const across=new Map<number,number[]>();
 for(let face=0;face<indices.length/3;face++)for(let k=0;k<3;k++){const a=weld[indices[face*3+k]!]!,b=weld[indices[face*3+(k+1)%3]!]!;if(a===b)continue;const list=across.get(key(a,b));if(list)list.push(face);else across.set(key(a,b),[face]);}
 const regions:number[]=[],edges:number[]=[];
 for(const region of planeRegions){
  const own=new Set(region.faces),face0=region.faces[0]!,n=[faceNormals[face0*3]!,faceNormals[face0*3+1]!,faceNormals[face0*3+2]!];
  const v0=indices[face0*3]!*3,d=n[0]!*positions[v0]!+n[1]!*positions[v0+1]!+n[2]!*positions[v0+2]!;
  const halfedges=[...(region.boundary??[]),...(region.openings??[]).flat()];
  const first=edges.length/12;let cx=0,cy=0,cz=0,count=0;
  for(const h of halfedges){const a=indices[h]!;cx+=positions[a*3]!;cy+=positions[a*3+1]!;cz+=positions[a*3+2]!;}
  if(halfedges.length)cx/=halfedges.length,cy/=halfedges.length,cz/=halfedges.length;
  let radius=0;for(const h of halfedges){const a=indices[h]!;radius=Math.max(radius,Math.hypot(positions[a*3]!-cx,positions[a*3+1]!-cy,positions[a*3+2]!-cz));}
  for(const h of halfedges){
   if(count>=TONE_REGION_EDGES)break;
   const face=Math.floor(h/3),a=indices[h]!,b=indices[face*3+(h+1)%3]!;
   const other=across.get(key(weld[a]!,weld[b]!))?.find(f=>!own.has(f));
   // Only a real turn of the form is a boundary (the across face turned 30
   // degrees or more, or no face): tone-field.ts darkEdgeDistance never
   // counts the rest, so they are not listed.
   if(other!==undefined&&faceNormals[other*3]!*n[0]!+faceNormals[other*3+1]!*n[1]!+faceNormals[other*3+2]!*n[2]!>=.87)continue;
   edges.push(positions[a*3]!,positions[a*3+1]!,positions[a*3+2]!,0,positions[b*3]!,positions[b*3+1]!,positions[b*3+2]!,0,
    other===undefined?0:faceNormals[other*3]!,other===undefined?0:faceNormals[other*3+1]!,other===undefined?0:faceNormals[other*3+2]!,other===undefined?1:0);
   count++;
  }
  regions.push(n[0]!,n[1]!,n[2]!,d,first,count,radius,0);
 }
 return {regions:Float32Array.from(regions),edges:Float32Array.from(edges)};
}
/** FOLIAGE MASS, a geometry class of a path-led source, in this fixed rule:
 * the surface is ORGANIC (the six largest normal directions, on a quarter
 * grid, hold under half its area: a built form's faces fall into a few
 * directions, a crown's leaf masses face every way) and it HANGS (faces
 * turned down by more than about 17 degrees, standing above the lowest third
 * of its height, hold at least .15 of its area: the underside of a crown
 * over its trunk). A rock is organic but its underside is its foot; a house
 * hangs eaves but is built. A face HANGS only over open space: its centre
 * lies outside 1.5 times the radius of the source's own foot (the vertices of
 * its lowest .15 of height, about their centre). A crown over its trunk
 * overhangs a narrow foot; a boulder's down-turned bulge stands over its own
 * wide resting base, so a boulder is not a crown. The half, the third, .15,
 * the .15 foot slice and 1.5 are this rule's own choices, not measured
 * reference values. Source frame, +Y up (as the terrain rule reads it).
 * Returns the ORGANIC bit too: an organic surface that is not a crown (a
 * boulder, a rock) takes the round's rim band (tone-field.ts organicOf). */
function foliageMass(positions:Float32Array,indices:Uint32Array,faceNormals:Float32Array,low:number,high:number):{organic:boolean;foliage:boolean}{
 const bins=new Map<number,number>(),floor=low+(high-low)/3,footTop=low+(high-low)*.15;let total=0,hung=0;
 let fx=0,fz=0,fn=0;
 for(let i=0;i<positions.length;i+=3)if(positions[i+1]!<=footTop){fx+=positions[i]!;fz+=positions[i+2]!;fn++;}
 if(fn)fx/=fn,fz/=fn;
 let foot=0;for(let i=0;i<positions.length;i+=3)if(positions[i+1]!<=footTop)foot=Math.max(foot,Math.hypot(positions[i]!-fx,positions[i+2]!-fz));
 for(let face=0;face<indices.length/3;face++){
  const a=indices[face*3]!*3,b=indices[face*3+1]!*3,c=indices[face*3+2]!*3;
  const u=[positions[b]!-positions[a]!,positions[b+1]!-positions[a+1]!,positions[b+2]!-positions[a+2]!],v=[positions[c]!-positions[a]!,positions[c+1]!-positions[a+1]!,positions[c+2]!-positions[a+2]!];
  const area=Math.hypot(u[1]!*v[2]!-u[2]!*v[1]!,u[2]!*v[0]!-u[0]!*v[2]!,u[0]!*v[1]!-u[1]!*v[0]!)/2;
  const nx=faceNormals[face*3]!,ny=faceNormals[face*3+1]!,nz=faceNormals[face*3+2]!;
  if(!(area>0)||nx*nx+ny*ny+nz*nz<.25)continue;
  total+=area;
  const key=(Math.round(nx*4)+4)*81+(Math.round(ny*4)+4)*9+(Math.round(nz*4)+4);bins.set(key,(bins.get(key)??0)+area);
  if(ny< -.3&&(positions[a+1]!+positions[b+1]!+positions[c+1]!)/3>floor&&Math.hypot((positions[a]!+positions[b]!+positions[c]!)/3-fx,(positions[a+2]!+positions[b+2]!+positions[c+2]!)/3-fz)>1.5*foot)hung+=area;
 }
 if(!(total>0))return {organic:false,foliage:false};
 const top=[...bins.values()].sort((x,y)=>y-x).slice(0,6).reduce((n,x)=>n+x,0);
 const organic=top/total<.5;
 return {organic,foliage:organic&&hung/total>=.15};
}
export interface SourceSupportContact {
 readonly sourceVertex:number;readonly receiver:string;readonly receiverFace:number;
 readonly receiverBarycentric:readonly [number,number,number];readonly point:readonly [number,number,number];
 readonly posed:boolean;
}
export interface CompositionSubject {
 readonly id:string;readonly owner:string;readonly type:ConstructionType;readonly purpose:string;
 readonly bounds:readonly [number,number,number,number];readonly depth:number;readonly relations:readonly PosedRelation[];
 readonly openings:readonly string[];readonly contacts:readonly string[];readonly far:boolean;
 /** Source-to-receiver attachment is not exclusive to planar solids. */
 readonly supports?:readonly string[];
 /** Exact source/model incidence. Articulation also verifies the actual
  * material-posed source port before using this as a native phase anchor. */
 readonly supportContacts?:readonly SourceSupportContact[];
 readonly planar?:PlanarAdmission;
 /** Intrinsic source stations/forks/holes. These are construction guides,
  * never an instruction to spell every section or mesh edge. */
 readonly structure?:MeshPartGraph;readonly structureError?:string;
 /** Coplanar source regions remain parts of a mixed mesh; an entire house,
  * figure or branch complex is not forced into one global glyph family. */
 readonly planeRegions?:readonly SourcePlaneRegion[];
}
/** Connected coplanar faces (through welded edges); `id` is its lowest face.
 * `boundary` is the outline and `openings` the interior boundary cycles, as
 * original halfedge addresses (face * 3 + corner) in walking order. */
export interface SourcePlaneRegion {readonly id:number;readonly faces:readonly number[];readonly boundary?:readonly number[];readonly openings?:readonly (readonly number[])[]}
/** Geometry/context admission, not a font or topology-count dispatch. */
export interface PlanarAdmission {
 readonly feature:string;readonly revision:number;readonly purpose:'convex-volume'|'open-sheet'|'unresolved';
 readonly hierarchy:'primary'|'support';readonly faces:readonly {id:number;normal:readonly number[];origin:readonly number[];regions:readonly number[]}[];
 readonly supports:readonly string[];readonly openings:readonly string[];
 readonly authority:'independent-P1';readonly unresolved?:string;
}
interface Plane {id:number;key:string;area:number;normal:number[];origin:number[];u:number[];v:number[];faces:number[]}
export interface SourceAnalysis {readonly styleWord:number;planes:Plane[];planeRegions:readonly SourcePlaneRegion[];faceNormals:Float32Array;type:ConstructionType;radius:number;center:number[];convex:boolean;volume:boolean;foliage:boolean;organic:boolean;toneRegions?:ToneRegions;
 /** Wall faces (compileWallFaces); absent unless the face gate admits the source. */
 faces?:readonly SourceFace[];
 runProgram?:CpuCompiledSourceRunProgram;
 /** Welded-surface measurements the class was decided from. */
 sheet:SourceSheet;
 supportFaces?:SupportFaces;
 structure?:MeshPartGraph;structureQueued?:boolean;structureError?:string;workId?:number;workPriority?:number}
type Analysis=SourceAnalysis;
export interface ComposedPlane {
 readonly id:string;readonly owner:string;readonly view:DrawingFieldView;
 readonly geometry:CompositionGeometry;readonly plane:Plane;readonly inverse?:Float32Array;
 readonly uv:readonly [number,number,number,number];readonly screen:readonly [number,number,number,number];readonly samples:number;
 readonly spacing:number;readonly sites:readonly {id:string;point:readonly [number,number];source:readonly number[]}[];
}
export interface CompositionFrame {
 readonly subjects:readonly CompositionSubject[];readonly planes:readonly ComposedPlane[];
 readonly farStrip:readonly {camera:string;contributors:readonly string[];envelope:readonly {x:number;y:number;owner:string}[]}[];
 readonly owners:ReadonlyMap<string,PosedOwner>;readonly cpuMs:number;
}
const dot=(a:V,b:V)=>a.reduce((n,v,i)=>n+v*b[i]!,0);
const sub=(a:V,b:V)=>a.map((v,i)=>v-b[i]!);
const cross=(a:V,b:V)=>[a[1]!*b[2]!-a[2]!*b[1]!,a[2]!*b[0]!-a[0]!*b[2]!,a[0]!*b[1]!-a[1]!*b[0]!];
const unit=(a:V)=>{const l=Math.hypot(...a);return a.map(v=>v/(l||1));};
export function transformPoint(m:ArrayLike<number>,p:ArrayLike<number>):number[]{const q=[0,1,2,3].map(r=>m[r]!*p[0]!+m[4+r]!*p[1]!+m[8+r]!*p[2]!+m[12+r]!);return q.slice(0,3).map(v=>v/q[3]!);}
const IDENTITY=new Float32Array([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]);
export function inverseMatrix(m:ArrayLike<number>):Float32Array{
 const a=Array.from({length:4},(_,r)=>[...Array.from({length:4},(_,c)=>m[c*4+r]!),...Array.from({length:4},(_,c)=>Number(c===r))]);
 for(let c=0;c<4;c++){let best=c;for(let r=c+1;r<4;r++)if(Math.abs(a[r]![c]!)>Math.abs(a[best]![c]!))best=r;
  [a[c],a[best]]=[a[best]!,a[c]!];const k=a[c]![c]!;if(Math.abs(k)<1e-20)throw Error('Noninvertible composition source transform');a[c]=a[c]!.map(v=>v/k);
  for(let r=0;r<4;r++)if(r!==c){const f=a[r]![c]!;a[r]=a[r]!.map((v,i)=>v-f*a[c]![i]!);}}
 return Float32Array.from({length:16},(_,i)=>a[i%4]![4+Math.floor(i/4)]!);
}
function modelInverse(m:ArrayLike<number>):Float32Array|undefined{
 try{return inverseMatrix(m);}catch(error){if(error instanceof Error&&error.message==='Noninvertible composition source transform')return;throw error;}
}
export function fieldWorld(view:DrawingFieldView,x:number,y:number,depth:number):number[]{return transformPoint(view.field.source.worldFromClip,[(x-view.rect.x)/view.rect.width*2-1,1-(y-view.rect.y)/view.rect.height*2,depth]);}
const clips=new WeakMap<object,Float32Array>();
function project(view:DrawingFieldView,world:V):number[]{let clip=clips.get(view);if(!clip){clip=inverseMatrix(view.field.source.worldFromClip);clips.set(view,clip);}const p=transformPoint(clip,world);return [view.rect.x+(p[0]!+1)*view.rect.width/2,view.rect.y+(1-p[1]!)*view.rect.height/2,p[2]!];}
function bounds(points:readonly V[]):[number,number,number,number]{
 const result:[number,number,number,number]=[Infinity,Infinity,-Infinity,-Infinity];
 for(const p of points){result[0]=Math.min(result[0],p[0]!);result[1]=Math.min(result[1],p[1]!);result[2]=Math.max(result[2],p[0]!);result[3]=Math.max(result[3],p[1]!);}
 return result;
}
export function fieldOwner(view:DrawingFieldView,sample:{source:number;owner:number;kind:string}):string|undefined{
 const reference=sample.kind==='entity'?view.owners.get(sample.owner):sample.owner;
 return reference===undefined?undefined:`${view.camera}/${sample.source}/${sample.kind}/${reference}`;
}
export function compileSourceAnalysis(g:CompositionGeometry,materialClasses?:Uint32Array,registration?:{readonly weld?:Uint32Array;readonly retention?:'full'|'summary'}):SourceAnalysis{
  // One numeric source remains authoritative. Do not manufacture a retained
  // nested coordinate array (plus a TypedArray slice) for every source vertex.
  const p=(i:number)=>[g.positions[i*3]!,g.positions[i*3+1]!,g.positions[i*3+2]!];
  const low=[Infinity,Infinity,Infinity],high=[-Infinity,-Infinity,-Infinity];
  for(let at=0;at<g.positions.length;at+=3)for(let k=0;k<3;k++){low[k]=Math.min(low[k]!,g.positions[at+k]!);high[k]=Math.max(high[k]!,g.positions[at+k]!);}
  const center=low.map((v,k)=>(v+high[k]!)/2),radii=high.map((v,k)=>(v-low[k]!)/2);
  const radius=Math.max(...radii,1e-6);
  let round=g.positions.length/3>8&&radii.every(r=>r>radius*.05);
  for(let at=0;round&&at<g.positions.length;at+=3)round=Math.abs(Math.hypot((g.positions[at]!-center[0]!)/radii[0]!,
   (g.positions[at+1]!-center[1]!)/radii[1]!, (g.positions[at+2]!-center[2]!)/radii[2]!)-1)<.2;
  const planes:Plane[]=[],groups=new Map<string,Plane>(),faceNormals=new Float32Array(g.indices.length),facePlane=new Int32Array(g.indices.length/3).fill(-1),planeIndex=new Map<Plane,number>();
  for(let face=0;face<g.indices.length/3;face++){
   const a=p(g.indices[face*3]!),b=p(g.indices[face*3+1]!),c=p(g.indices[face*3+2]!),normal=unit(cross(sub(b,a),sub(c,a)));if(Math.hypot(...normal)<.5)continue;
   faceNormals.set(normal,face*3);
   const d=dot(normal,a),key=[...normal.map(v=>Math.round(v*10000)),Math.round(d/radius*10000)].join('/');
   let plane=groups.get(key);if(!plane){const u=unit(cross(Math.abs(normal[0]!)<.9?[1,0,0]:[0,0,1],normal));plane={id:face,key:[...normal.map(v=>Math.round(v*10000)),Math.round(d*10000)].join('/'),normal,origin:normal.map(v=>v*d),u,v:cross(normal,u),faces:[],area:0};groups.set(key,plane);planeIndex.set(plane,planes.length);planes.push(plane);}plane.faces.push(face);plane.area+=Math.hypot(...cross(sub(b,a),sub(c,a)))/2;facePlane[face]=planeIndex.get(plane)!;
  }
  // The surface is read through welded rest positions, so a mesh exported as
  // triangle soup has the same sheet, rim and plane regions as its indexed
  // twin. Compiled once per source revision; nothing here runs per frame.
  const sheet=compileSourceSheet(g.positions,g.indices,faceNormals,facePlane,registration?.weld);
  // Elongation describes a part's gesture, not its surface construction.
  // A long box, wall or rail still has the same joined planar faces as a
  // cube. Dispatching it to path-led solely by its aspect ratio disconnected
  // the plane producer exactly where its long sides mattered most.
  // The class is decided from geometry alone, in this fixed order:
  //  rounded: every vertex within .2 of the bounding ellipsoid (more than 8);
  //  planar: at most 24 quantised planes (box, room, block);
  //  terrain: a height surface. One connected open sheet whose faces look to
  //   one side, wide for its relief: at least 25 planes, an open rim, one
  //   welded component with >= .9 of the area, >= .9 of the area facing the
  //   mean normal, that normal within 60 degrees of +Y, relief <= .5 of the
  //   lateral size. Thresholds and reasons: TERRAIN_RULE, source-complex.ts.
  //   It takes sources only out of what was path-led: ground is ridges, slope
  //   turns and contact lines (D3), not a limb skeleton.
  //  path-led: everything else (limbs, crowns, figures).
  // A producer-owned heightfield remains a terrain surface on steep cliffs
  // and flat plateaux too. The geometric heuristic is for untyped meshes,
  // not permission to reinterpret a known grid as a limb skeleton.
  const type:ConstructionType=g.surface==='heightfield'?'terrain':round?'rounded':planes.length<=24?'planar':isTerrainSheet(sheet,planes.length)?'terrain':'path-led';
  const convex=type==='planar'&&planes.every(plane=>{let positive=false,negative=false;for(let at=0;at<g.positions.length;at+=3){let distance=0;for(let k=0;k<3;k++)distance+=plane.normal[k]!*(g.positions[at+k]!-plane.origin[k]!);positive||=distance>radius*1e-5;negative||=distance< -radius*1e-5;if(positive&&negative)return false;}return true;});
  const volume=radii.every(r=>r>radius*1e-5)&&planes.length>=4;
  // A plane region is CONNECTED coplanar faces: faces of one quantised plane
  // that reach each other through welded edges. Two coplanar patches that do
  // not touch (two table tops, window sills on one wall line) are two regions
  // with their own outline, shared edges and openings. `planes` keeps every
  // face of the quantised plane for the consumers that read area and extent.
  const planeRegions:SourcePlaneRegion[]=[];
  for(const plane of planes){
   if(plane.faces.length<2)continue;
   const parts=new Map<number,number[]>();for(const face of plane.faces){const root=sheet.planeRoot[face]!,part=parts.get(root);if(part)part.push(face);else parts.set(root,[face]);}
   for(const faces of parts.values())if(faces.length>=2){
    // Terrain playback consumes plane membership/edge classes, never plane
    // boundary cycles or plane-region tone. Preserve every region and face;
    // full inspection still compiles all outlines/openings. Do not build a
    // discarded loop graph for every coplanar patch of a streamed heightfield.
    if(registration?.retention==='summary'&&type==='terrain'){planeRegions.push({id:faces[0]!,faces});continue;}
    const loops=planeRegionLoops(g.positions,g.indices,sheet.weld,faces,plane.u,plane.v);planeRegions.push({id:faces[0]!,faces,boundary:loops.boundary,openings:loops.openings});
   }
  }
  const form=type==='path-led'?foliageMass(g.positions,g.indices,faceNormals,low[1]!,high[1]!):{organic:false,foliage:false};
  const foliage=form.foliage,organic=form.organic&&!foliage;
  const toneRegions=type!=='terrain'&&type!=='rounded'&&!foliage&&!organic&&planeRegions.length<=TONE_REGION_LIMIT?compileToneRegions(g.positions,g.indices,sheet.weld,faceNormals,planeRegions):undefined;
  const styleWord=compileSourceStyleWord({type,surface:g.surface,foliage,organic},compileSourceMaterialClass(g),materialClasses);
  // The face gate (compileWallFaces): the class above is unchanged; a built
  // path-led form with more regions than the box owner takes gets its faces.
  const built=type==='path-led'&&!foliage&&!organic&&planeRegions.length>24&&planeRegions.length<=TONE_REGION_LIMIT;
  const wallFaces=built?compileWallFaces(g.positions,g.indices,sheet.weld,faceNormals,planeRegions,radius):[];
  const faces=wallFaces.length?wallFaces:undefined;
  return {planes,planeRegions,faceNormals,type,radius,center,convex,volume,sheet,foliage,organic,toneRegions,styleWord,...(faces?{faces}:{})};
}
/** The single glyph-free composition owner. Geometry, read hierarchy, current
 * visibility and protected relationships are decided here; no font or role bank
 * is imported. Unsupported articulation is paper, never a segment speller. */
export class Composition {
 private readonly cache=new WeakMap<Float32Array,WeakMap<Uint32Array,{revision:number;analysis:Analysis}>>();
 private readonly far=new Map<string,boolean>();private readonly selected=new Set<string>();private readonly levels=new Map<string,number>();
 private supportSnapshot:{geometry:readonly CompositionGeometry[];matrices:readonly number[][];dependencies:ReadonlyMap<string,readonly string[]>;contacts:ReadonlyMap<string,readonly SourceSupportContact[]>}|undefined;
 private readonly supportParts=new Map<string,{part:SupportPart;matrix:readonly number[];inverse?:Float32Array;worldFaces?:SupportFaces}>();
 private readonly partWork=new Map<number,Analysis>();private partWorker:Worker|undefined;private nextPartWork=0;private disposed=false;
 dispose():void{this.disposed=true;this.partWorker?.terminate();this.partWorker=undefined;this.partWork.clear();this.supportParts.clear();this.supportSnapshot=undefined;}
 private queueParts(g:CompositionGeometry,analysis:Analysis,priority:number):void{
  // A terrain source is ridges, slope turns and contact lines read from its
  // source complex; it has no sections or bands to compile. Its chunks are
  // the largest sources of a streamed world, and their part graphs arriving
  // one by one each rebuilt the resident topology packet (loop 351 view 02).
  if(this.disposed||analysis.structure||analysis.structureError||analysis.type==='terrain')return;
  if(analysis.structureQueued){if(analysis.workPriority!==priority){analysis.workPriority=priority;this.partWorker!.postMessage({id:analysis.workId,priority});}return;}
  if(!this.partWorker){
   this.partWorker=new Worker(new URL('./mesh-part-worker.ts',import.meta.url),{type:'module'});
   this.partWorker.onmessage=({data}:{data:{id:number;graph?:MeshPartGraph;error?:string}})=>{
    const target=this.partWork.get(data.id);if(!target)return;this.partWork.delete(data.id);
    if(data.graph)target.structure=data.graph;else target.structureError=data.error??'Source part compiler returned no graph';
   };
   this.partWorker.onerror=event=>{for(const target of this.partWork.values())target.structureError=event.message||'Source part compiler failed';this.partWork.clear();};
  }
  analysis.structureQueued=true;analysis.workId=++this.nextPartWork;analysis.workPriority=priority;
  // The existing geometry-revision cache is the only authority. This queue
  // contains resumable static work, not a second feature registry/publisher.
  // Snapshot once: the next source revision cannot mutate an unfinished job.
  const adjacency=g.lineage?.adjacency;
  const mesh={positions:g.positions.slice(),indices:g.indices.slice(),adjacency:adjacency?{offsets:adjacency.offsets.slice(),faces:adjacency.faces.slice()}:undefined};
  this.partWork.set(analysis.workId,analysis);
  this.partWorker.postMessage({id:analysis.workId,priority,mesh},[mesh.positions.buffer,mesh.indices.buffer,...(mesh.adjacency?[mesh.adjacency.offsets.buffer,mesh.adjacency.faces.buffer]:[])]);
 }
 retire(isLive:(owner:string)=>boolean):void{for(const map of [this.far,this.levels])for(const key of map.keys())if(!isLive(key.split('/plane-')[0]!))map.delete(key);for(const key of this.selected)if(!isLive(key))this.selected.delete(key);}
 /** The GPU articulation adapter shares this geometry-revision compiler;
  * it must not build a second mesh graph or a new per-frame part registry. */
 compiledGraphs(geometry:readonly CompositionGeometry[]):ReadonlyMap<string,CompiledConstructionGraph>{
  const parts=new Map<string,CompiledConstructionGraph>();
  for(const source of geometry){const analysis=this.analyze(source);parts.set(source.owner,{type:analysis.type,structure:analysis.structure,planeRegions:analysis.planeRegions,faceNormals:analysis.faceNormals,runProgram:analysis.runProgram});if(!analysis.structure)this.queueParts(source,analysis,1);}
  return parts;
 }
 /** Residency lookup, shared by CPU incumbent and retained GPU construction.
  * analyze and the worker are keyed by the immutable source revision. */
 compiledSource(source:CompositionGeometry):CompiledConstructionGraph{
  const analysis=this.analyze(source);if(!analysis.structure)this.queueParts(source,analysis,1);
  return {styleWord:analysis.styleWord,type:analysis.type,structure:analysis.structure,planeRegions:analysis.planeRegions,faceNormals:analysis.faceNormals,foliage:analysis.foliage,organic:analysis.organic,toneRegions:analysis.toneRegions,runProgram:analysis.runProgram};
 }
 /** Install the worker-owned source revision once; legacy analysis uses the
  * same pure compiler, never an alternative geometry classifier. */
 installSource(source:CompositionGeometry,analysis:SourceAnalysis):void{
  let indexed=this.cache.get(source.positions);if(!indexed){indexed=new WeakMap();this.cache.set(source.positions,indexed);}
  indexed.set(source.indices,{revision:source.revision,analysis});
 }
 /** Complete source inventory, before current-view GPU ranking/visibility.
  * It is not a read-hierarchy quota: every input owner and every possible
  * support receiver are already members. Re-projecting bounds, sorting them
  * and building posed CPU contacts cannot add an owner to this set. Mesh
  * compilation stays in compiledGraphs; legacy select evaluates contacts
  * when it actually uses them. Neither producer is disconnected here. */
 chooseSources(geometry:readonly CompositionGeometry[]):ReadonlySet<string>{
  return new Set(geometry.map(source=>source.owner));
 }
 private analyze(g:CompositionGeometry):Analysis{
  let indexed=this.cache.get(g.positions);if(!indexed){indexed=new WeakMap();this.cache.set(g.positions,indexed);}
  const previous=indexed.get(g.indices);if(previous?.revision===g.revision)return previous.analysis;
  if(previous?.analysis.workId&&this.partWork.has(previous.analysis.workId)){
   this.partWorker?.postMessage({id:previous.analysis.workId,cancel:true});this.partWork.delete(previous.analysis.workId);
  }
  const analysis=compileSourceAnalysis(g);indexed.set(g.indices,{revision:g.revision,analysis});return analysis;
 }
 /** Directional physical support closure. A screen intersection or bounding
  * box alone is never a contact: an actual lowest source vertex must lie on
  * an actual receiving triangle. Static posed inputs reuse this calculation. */
 private supports(geometry:readonly CompositionGeometry[]):ReadonlyMap<string,readonly string[]>{
  geometry=geometry.filter(source=>!source.sourceOnly);
  const before=this.supportSnapshot;
  if(before&&geometry.length===before.geometry.length&&geometry.every((g,i)=>{const old=before.geometry[i]!;return g.owner===old.owner&&g.positions===old.positions&&g.indices===old.indices&&g.revision===old.revision&&g.poseAuthority===old.poseAuthority&&Array.from(g.worldFromSource??IDENTITY).every((v,k)=>v===before.matrices[i]![k]);}))return before.dependencies;
  const parts=geometry.map((g,index)=>{const m=g.worldFromSource??IDENTITY,cached=this.supportParts.get(g.owner);
   if(cached&&cached.part.g.positions===g.positions&&cached.part.g.indices===g.indices&&cached.part.g.revision===g.revision&&Array.from(m).every((v,k)=>v===cached.matrix[k])){cached.part={...cached.part,g,index};return cached.part;}
   const points:number[][]=[];for(let i=0;i<g.positions.length;i+=3)points.push(transformPoint(m,g.positions.subarray(i,i+3)));
   const low=[0,1,2].map(k=>points.reduce((n,p)=>Math.min(n,p[k]!),Infinity)),high=[0,1,2].map(k=>points.reduce((n,p)=>Math.max(n,p[k]!),-Infinity));
   const part={g,points,low,high,index};this.supportParts.set(g.owner,{part,matrix:Array.from(m),inverse:modelInverse(m)});return part;});
  const live=new Set(geometry.map(g=>g.owner));for(const owner of this.supportParts.keys())if(!live.has(owner))this.supportParts.delete(owner);
  const cameras=new Map<string,SupportPart[]>();for(const part of parts){const camera=part.g.owner.split('/')[0]!,list=cameras.get(camera)??[];list.push(part);cameras.set(camera,list);}
  const receivers=new Map([...cameras].map(([camera,list])=>[camera,supportTree(list)]));
  const dependencies=new Map<string,string[]>(),contacts=new Map<string,SourceSupportContact[]>();
  for(const a of parts){const epsilon=Math.max(1e-6,this.analyze(a.g).radius*1e-5);if(a.high[1]!-a.low[1]!<=epsilon)continue;
   const feet=a.points.flatMap((point,index)=>point[1]!-a.low[1]!<=epsilon?[{point,index}]:[]);
   const candidates:SupportPart[]=[];supportCandidates(receivers.get(a.g.owner.split('/')[0]!)!,a,epsilon,candidates);candidates.sort((a,b)=>a.index-b.index);
   for(const b of candidates){if(a===b||a.low[1]!<b.low[1]!-epsilon||a.low[1]!>b.high[1]!+epsilon||a.low[0]!>b.high[0]!+epsilon||a.high[0]!<b.low[0]!-epsilon||a.low[2]!>b.high[2]!+epsilon||a.high[2]!<b.low[2]!-epsilon)continue;
    let contact=false;
    const analysis=this.analyze(b.g),posed=this.supportParts.get(b.g.owner)!,inverse=posed.inverse;
    const tree=inverse?(analysis.supportFaces??=supportFaces(b.g)):(posed.worldFaces??=supportFaces({positions:Float64Array.from(b.points.flat()),indices:b.g.indices}));
    // Transform the world tolerance conservatively for non-uniform scale.
    // Final contact still uses the exact posed WORLD triangle, not this box.
    const tolerance=inverse?[0,1,2].map(k=>epsilon*Math.hypot(inverse[k]!,inverse[4+k]!,inverse[8+k]!)):[epsilon,epsilon,epsilon];
    for(const {point:foot,index:sourceVertex}of feet){
     const faces:number[]=[];receivingFaces(tree,inverse?transformPoint(inverse,foot):foot,tolerance,faces);faces.sort((a,b)=>a-b);
     for(const face of faces){const i=face*3,p=b.points[b.g.indices[i]!]!,q=b.points[b.g.indices[i+1]!]!,r=b.points[b.g.indices[i+2]!]!,u=sub(q,p),v=sub(r,p),n=cross(u,v),nn=dot(n,n);if(nn<1e-20)continue;
      const uu=dot(u,u),uv=dot(u,v),vv=dot(v,v),det=uu*vv-uv*uv;if(det<=1e-20)continue;
      const d=sub(foot,p);if(Math.abs(dot(d,n))>epsilon*Math.sqrt(nn))continue;const du=dot(d,u),dv=dot(d,v),s=(du*vv-dv*uv)/det,t=(dv*uu-du*uv)/det;
      if(s>=-1e-6&&t>=-1e-6&&s+t<=1+1e-6){
       contact=true;const list=contacts.get(a.g.owner)??[];
       list.push({sourceVertex,receiver:b.g.owner,receiverFace:face,receiverBarycentric:[1-s-t,s,t],point:[foot[0]!,foot[1]!,foot[2]!],posed:a.g.poseAuthority==='source-model'&&b.g.poseAuthority==='source-model'});contacts.set(a.g.owner,list);
      }
     }
    }
    if(contact){const list=dependencies.get(a.g.owner)??[];list.push(b.g.owner);dependencies.set(a.g.owner,list);}
   }
  }
  this.supportSnapshot={geometry:[...geometry],matrices:geometry.map(g=>Array.from(g.worldFromSource??IDENTITY)),dependencies,contacts};return dependencies;
 }
 select(graphs:readonly PosedGraph[],views:readonly DrawingFieldView[],geometry:readonly CompositionGeometry[]):CompositionFrame{
  const start=performance.now(),byOwner=new Map<string,PosedRelation[]>(),owners=new Map<string,PosedOwner>(),sources=new Map(geometry.map(g=>[g.owner,g]));
  for(const graph of graphs){for(const [id,owner]of graph.owners??[])owners.set(id,owner);for(const relation of graph.relations){const list=byOwner.get(relation.owner)??[];list.push(relation);byOwner.set(relation.owner,list);}}
  const observations=new Map<string,{view:DrawingFieldView;points:number[][];depth:number;count:number}>();
  // This pass observes geometry ownership/normals, never color or pixel darkness.
  for(const view of views)for(let y=2;y<view.field.height;y+=4)for(let x=2;x<view.field.width;x+=4){const s=view.field.sample(x,y);if(!s||s.depth<=0||s.radiance[3]<.5)continue;const owner=fieldOwner(view,s);if(!owner)continue;
   const sx=view.rect.x+(x+.5)/view.field.width*view.rect.width,sy=view.rect.y+(y+.5)/view.field.height*view.rect.height;
   let observed=observations.get(owner);if(!observed){observed={view,points:[],depth:0,count:0};observations.set(owner,observed);}observed.points.push([sx,sy]);observed.depth+=s.depth;observed.count++;
   if(!owners.has(owner)){const reference=s.kind==='entity'?view.owners.get(s.owner)!:s.owner;owners.set(owner,{camera:view.camera,domain:s.source,kind:s.kind,index:s.owner,lifetime:reference});}}
  for(const [owner,relations]of byOwner)if(!observations.has(owner)){const view=views.find(v=>owner.startsWith(v.camera+'/'));if(view){const points=relations.flatMap(r=>r.intervals.filter(i=>i.cause==='visible').flatMap(i=>[relationPoint(r,i.lo),relationPoint(r,i.hi)])).filter(p=>p.every(Number.isFinite));if(points.length)observations.set(owner,{view,points:points.map(p=>[p[0],p[1]]),depth:points.reduce((n,p)=>n+p[2],0),count:points.length});}}
  const grounds=new Map<string,{owner:string;height:number;area:number}>(),outdoor=new Set<string>(),dependencies=this.supports(geometry);
  // Geometry ownership at the top of the view supplies an exterior context;
  // a room floor by itself cannot move the enclosing walls into a skyline.
  for(const view of views){let sky=0,total=0;for(let x=2;x<view.field.width;x+=8){total++;const s=view.field.sample(x,2);if(!s||s.depth<=0||s.radiance[3]<.5)sky++;}if(total&&sky/total>.5)outdoor.add(view.camera);}
  for(const g of geometry){if(g.sourceOnly)continue;const view=views.find(v=>g.owner.startsWith(v.camera+'/'));if(!view?.pose?.perspective)continue;const m=g.worldFromSource??IDENTITY,analysis=this.analyze(g);
   // Upward-facing tessellation on a rounded volume is not an exterior
   // receiver. Treating sphere-cap triangles as a ground plane sent entire
   // floating scenes to a nonexistent landscape strip before articulation.
   if(analysis.type==='rounded')continue;
   for(const plane of analysis.planes){const origin=transformPoint(m,plane.origin),u=transformPoint(m,plane.origin.map((v,k)=>v+plane.u[k]!)),v=transformPoint(m,plane.origin.map((q,k)=>q+plane.v[k]!));const n=unit(cross(sub(u,origin),sub(v,origin)));
    const height=(dot(n,origin)-n[0]!*view.pose.eye[0]!-n[2]!*view.pose.eye[2]!)/n[1]!;
    if(n[1]!<.95||height>=view.pose.eye[1]!||plane.faces.length<2)continue;
    const area=plane.area*Math.hypot(...cross(sub(u,origin),sub(v,origin))),prior=grounds.get(view.camera);
    if(!prior||area>prior.area*1.01||Math.abs(area-prior.area)<prior.area*.01&&height>prior.height)grounds.set(view.camera,{owner:g.owner,height,area});}}
  const receiverConnected=(owner:string,receiver:string)=>{
   const seen=new Set<string>(),pending=[owner];
   for(let at=0;at<pending.length;at++){const id=pending[at]!;if(id===receiver)return true;if(seen.has(id))continue;seen.add(id);for(const support of dependencies.get(id)??[])if(!seen.has(support))pending.push(support);}
   return false;
  };
  const candidates:CompositionSubject[]=[];
  for(const [owner,o]of observations){const g=sources.get(owner);let relations=byOwner.get(owner)??[],b=bounds(o.points);const analysis=g&&this.analyze(g),pose=o.view.pose;let protectedOpenings:string[]=[];
   if(analysis?.type==='rounded'&&relations.length){
    // Do not splice geometric silhouettes and normal-zero candidates into
    // one alleged contour. They have different ports and may intersect.
    // Preserve an independently closed smooth construction; actual mesh
    // boundaries add the missing closed-source alternative, rather than
    // splicing new ports into (or bypassing) that serving smooth family.
    const boundarySource={relations:relations.filter(r=>r.kind==='boundary'),ports:new Map()},independent=selectFeatureGraph(boundarySource).walks.filter(w=>w.closed);
    // A genuine source vertex can carry multiple front-surface fans. Degree
    // four there is not four disconnected outlines, nor a physical four-way
    // junction. Preserve serving independent cycles; otherwise recover the
    // complete oriented boundary from its actual post-material face support.
    // No projected proximity weld or glyph decision participates.
    const boundary=independent.length?independent:selectContourGraph(boundarySource,r=>{
     const c=r.frontFacePoint;if(!c)return 0;const a=relationPoint(r,0),b=relationPoint(r,1);
     return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);
    }).walks.filter(w=>w.closed);
    const smooth=selectFeatureGraph({relations:relations.filter(r=>r.kind==='smooth-contour'),ports:new Map()}).walks.filter(w=>w.closed);
    const walks=smooth.length?smooth:boundary;
    walks.sort((a,b)=>{const area=(w:typeof a)=>{const p=w.steps.map(s=>relationPoint(s.relation,s.reverse?1:0));return Math.abs(p.reduce((n,q,i)=>{const r=p[(i+1)%p.length]!;return n+q[0]*r[1]-r[0]*q[1];},0));};return area(b)-area(a)||a.id.localeCompare(b.id);});
    if(walks[0]){relations=walks[0].steps.map(s=>s.relation);b=bounds(relations.flatMap(r=>[relationPoint(r,0),relationPoint(r,1)]));protectedOpenings=walks.slice(1).filter(w=>{const q=bounds(w.steps.flatMap(s=>[relationPoint(s.relation,0),relationPoint(s.relation,1)]));return q[2]-q[0]>3&&q[3]-q[1]>3&&q[0]>b[0]&&q[1]>b[1]&&q[2]<b[2]&&q[3]<b[3];}).map(w=>w.id);}
   }
   if(analysis?.type==='planar'){
    const cycles=selectFeatureGraph({ports:new Map(),relations:relations.filter(r=>r.kind==='boundary')}).walks.filter(w=>w.closed).map(w=>({id:w.id,bounds:bounds(w.steps.map(s=>relationPoint(s.relation,s.reverse?1:0)))}));
    protectedOpenings=cycles.filter(hole=>cycles.some(outer=>outer!==hole&&hole.bounds[0]>outer.bounds[0]&&hole.bounds[1]>outer.bounds[1]&&hole.bounds[2]<outer.bounds[2]&&hole.bounds[3]<outer.bounds[3])).map(hole=>hole.id);
   }
   let far=false;
   if(g&&pose?.perspective&&grounds.has(o.view.camera)&&outdoor.has(o.view.camera)&&receiverConnected(owner,grounds.get(o.view.camera)!.owner)){
    const center=transformPoint(g.worldFromSource??IDENTITY,analysis!.center),h=pose.eye[1]!-grounds.get(o.view.camera)!.height,horizontal=unit([pose.forward[0]!,0,pose.forward[2]!]),d=dot(sub(center,pose.eye),horizontal);
    const depression=Math.atan2(h,d),entry=Math.atan(3*17/pose.focalPixels),exit=Math.atan(5*17/pose.focalPixels),previous=this.far.get(owner)??false;
    far=h>0&&d>0&&(depression<entry||previous&&depression<=exit);this.far.set(owner,far);
   }
   const ports=new Map<string,number>();for(const r of relations)for(const p of r.ports)ports.set(p,(ports.get(p)??0)+1);
   const type=far?'far-strip':analysis?.type??(relations.some(r=>r.kind==='smooth-contour')?'rounded':relations.length?'planar':'paper');
   if(type==='path-led'&&g&&analysis){const area=Math.max(0,b[2]-b[0])*Math.max(0,b[3]-b[1]);this.queueParts(g,analysis,area*(area>o.view.rect.width*o.view.rect.height*.8?.03:1));}
   const planar:PlanarAdmission|undefined=type==='planar'&&g&&analysis?{feature:owner,revision:g.revision,purpose:analysis.convex?(analysis.volume?'convex-volume':'open-sheet'):'unresolved',hierarchy:g.primary===false?'support':'primary',faces:analysis.planes.map(p=>({id:p.id,normal:p.normal,origin:p.origin,regions:p.faces})),supports:dependencies.get(owner)??[],openings:protectedOpenings,authority:'independent-P1',unresolved:analysis.convex?undefined:'concave plane complex requires an opening/part construction'}:undefined;
   candidates.push({id:owner,owner,type,purpose:far?'distant mass and skyline':type==='path-led'?'connected main part, entries and support':type==='planar'?'plane separation, openings and contacts':type==='rounded'?'closed volume and its paper':'unsupported source',bounds:b,depth:o.depth/o.count,relations,contacts:[...ports].filter(([,n])=>n>2).map(([id])=>id),openings:protectedOpenings,far,supports:dependencies.get(owner)??[],supportContacts:this.supportSnapshot?.contacts.get(owner)??[],planar,planeRegions:analysis?.planeRegions,structure:analysis?.structure,structureError:analysis?.structureError});
  }
  // Ranking proposes read hierarchy; all contact/opening obligations of an
  // admitted owner remain attached, rather than being independently culled.
  const importance=(s:CompositionSubject)=>{const [a,b,c,d]=s.bounds;return Math.max(0,c-a)*Math.max(0,d-b)*(1+Math.min(1,s.depth))*(s.contacts.length?1.2:1)*(this.selected.has(s.id)?1.2:1);};
  const near=views.flatMap(view=>candidates.filter(s=>!s.far&&s.owner.startsWith(view.camera+'/')&&sources.get(s.owner)?.primary!==false).sort((a,b)=>importance(b)-importance(a)||a.id.localeCompare(b.id)));
  const required=new Set(near.flatMap(s=>s.supports??[]));
  for(const s of candidates)if(required.has(s.owner)&&!near.includes(s))near.push({...s,far:false,type:s.type==='far-strip'?'planar':s.type});
  this.selected.clear();near.forEach(s=>this.selected.add(s.id));
  const farStrip:CompositionFrame['farStrip'][number][]=[];
  for(const view of views){const subjects=candidates.filter(s=>s.far&&s.owner.startsWith(view.camera+'/'));if(!subjects.length)continue;
   const bins=new Map<number,{x:number;y:number;owner:string}>();for(const s of subjects)for(const r of s.relations)for(const part of r.intervals.filter(p=>p.cause==='visible'))for(const t of [part.lo,part.hi]){const p=relationPoint(r,t),x=Math.round(p[0]/11);const old=bins.get(x);if(!old||p[1]<old.y)bins.set(x,{x:p[0],y:p[1],owner:s.owner});}
   farStrip.push({camera:view.camera,contributors:subjects.map(s=>s.id).sort(),envelope:[...bins.values()].sort((a,b)=>a.x-b.x)});}
  const planes:ComposedPlane[]=[];
  type PlaneCell={plane:Plane;uv:number[][];screen:number[][];value:number};
  type PosedPlane={plane:Plane;origin:number[];u:number[];v:number[];normal:number[];uu:number;uv:number;vv:number;det:number};
  const receivers=new Map<string,{s:CompositionSubject;g:CompositionGeometry;analysis:Analysis;m:Float32Array;inv?:Float32Array;charts:PosedPlane[];view:DrawingFieldView;cells:Map<number,PlaneCell>}>();
  for(const s of near){
   if(s.type!=='planar')continue;
   const g=sources.get(s.owner),o=observations.get(s.owner);if(!g||!o||g.sourceOnly)continue;
   const m=g.worldFromSource??IDENTITY;
   const analysis=this.analyze(g),inv=modelInverse(m),charts:PosedPlane[]=[];
   // A zero model axis can leave a real visible plane. Its two-dimensional
   // source chart still exists even when a three-dimensional inverse does not.
   if(!inv)for(const plane of analysis.planes){
    const origin=transformPoint(m,plane.origin),u=sub(transformPoint(m,plane.origin.map((q,k)=>q+plane.u[k]!)),origin),v=sub(transformPoint(m,plane.origin.map((q,k)=>q+plane.v[k]!)),origin),normal=unit(cross(u,v));
    const uu=dot(u,u),uv=dot(u,v),vv=dot(v,v),det=uu*vv-uv*uv;if(det<=1e-20)continue;
    charts.push({plane,origin,u,v,normal,uu,uv,vv,det});
   }
   receivers.set(s.owner,{s,g,analysis,m,inv,charts,view:o.view,cells:new Map()});
  }
  // Ownership partitions this field. Visit its unchanged source sample grid
  // ONCE, rather than walking the entire image again for every plane owner.
  // Each receiver still sees exactly its original row-major sample sequence;
  // face equations, charts, illumination and native construction are unchanged.
  for(const view of views)for(let y=2;y<view.field.height;y+=6)for(let x=2;x<view.field.width;x+=6){
   const sample=view.field.sample(x,y);if(!sample)continue;
   const owner=fieldOwner(view,sample),receiver=owner&&receivers.get(owner);if(!receiver||receiver.view!==view)continue;
   const {analysis,inv,charts,cells}=receiver,sx=view.rect.x+(x+.5)/view.field.width*view.rect.width,sy=view.rect.y+(y+.5)/view.field.height*view.rect.height,world=fieldWorld(view,sx,sy,sample.depth),p=inv?transformPoint(inv,world):undefined;
   for(const plane of analysis.planes){
    let coordinates:number[];
    if(p){if(Math.abs(dot(plane.normal,sub(p,plane.origin)))>analysis.radius*.0005)continue;const d=sub(p,plane.origin);coordinates=[dot(d,plane.u),dot(d,plane.v)];}
    else{
     const chart=charts.find(chart=>chart.plane===plane);if(!chart)continue;const d=sub(world,chart.origin);
     if(Math.abs(dot(d,chart.normal))>analysis.radius*.0005*Math.max(Math.sqrt(chart.uu),Math.sqrt(chart.vv)))continue;
     const du=dot(d,chart.u),dv=dot(d,chart.v);coordinates=[(du*chart.vv-dv*chart.uv)/chart.det,(dv*chart.uu-du*chart.uv)/chart.det];
    }
    let cell=cells.get(plane.id);if(!cell){cell={plane,uv:[],screen:[],value:0};cells.set(plane.id,cell);}
    cell.uv.push(coordinates);cell.screen.push([sx,sy]);
    const light=sample.appearance??sample.radiance;cell.value+=light[0]*.2126+light[1]*.7152+light[2]*.0722;break;
   }
  }
  for(const {s,g,m,inv,view,cells}of receivers.values()){
   for(const cell of [...cells.values()].sort((a,b)=>b.uv.length-a.uv.length||a.plane.id-b.plane.id)){
    if(cell.uv.length<12)continue;const plane=cell.plane,id=`${s.owner}/plane-${g.lineage?plane.key:plane.id}`,uv=bounds(cell.uv),screen=bounds(cell.screen),center=[(uv[0]+uv[2])/2,(uv[1]+uv[3])/2];
    const world=(u:number,v:number)=>transformPoint(m,plane.origin.map((q,k)=>q+u*plane.u[k]!+v*plane.v[k]!));
    const p=project(view,world(center[0]!,center[1]!)),u=project(view,world(center[0]!+1,center[1]!)),v=project(view,world(center[0]!,center[1]!+1));
    const scale=Math.max(Math.hypot(u[0]!-p[0]!,u[1]!-p[1]!),Math.hypot(v[0]!-p[0]!,v[1]!-p[1]!));if(!Number.isFinite(scale)||scale<=0)continue;
    const targetSpacing=cell.value/cell.uv.length>.55?40:24;
    let level=this.levels.get(id)??Math.ceil(Math.log2(targetSpacing/scale));if(scale*2**level<targetSpacing*.75||scale*2**level>targetSpacing*1.5)level=Math.ceil(Math.log2(targetSpacing/scale));this.levels.set(id,level);const spacing=2**level;
    const sites:ComposedPlane['sites'][number][]=[];let visited=0;
    for(let j=Math.ceil(uv[1]/spacing);j<=Math.floor(uv[3]/spacing)&&visited<1000;j++)for(let i=Math.ceil(uv[0]/spacing);i<=Math.floor(uv[2]/spacing)&&visited++<1000;i++){
     const source=world(i*spacing,j*spacing),point=project(view,source);if(point[2]!<=0||point[2]!>1||!point.every(Number.isFinite))continue;
     const sample=view.field.sample((point[0]!-view.rect.x)/view.rect.width*view.field.width,(point[1]!-view.rect.y)/view.rect.height*view.field.height);if(!sample||fieldOwner(view,sample)!==s.owner||Math.abs(sample.depth-point[2]!)>Math.max(1e-5,point[2]!*.005))continue;
     sites.push({id:`${id}/${i*spacing}/${j*spacing}`,point:[point[0]!,point[1]!],source});}
    planes.push({id,owner:s.owner,view,geometry:g,plane,inverse:inv,uv,screen,samples:cell.uv.length,spacing,sites});
   }
  }
  return {subjects:near,planes,farStrip,owners,cpuMs:performance.now()-start};
 }
}
