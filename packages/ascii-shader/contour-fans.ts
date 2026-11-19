import {transformPoint,type CompositionGeometry} from './composition.ts';
import {inverseMatrix} from './composition.ts';
import {selectContourGraph,type FeatureGraph} from './feature-graph.ts';
import type {MeshPartGraph} from './mesh-part-graph.ts';
import type {DrawingFieldView} from './field-readback.ts';
import {relationPoint,type PosedGraph,type PosedPort,type PosedRelation} from './posed-graph.ts';

const clips=new WeakMap<object,Float32Array>();
/** Rotation of actual projected front-surface boundaries. Original adjacent
 * faces supply the inside side; source vertex incidence supplies contacts.
 * No silhouette bounding box, pixel tracing or glyph angle fit is involved. */
export function sourceContourGraph(graph:PosedGraph,geometry:CompositionGeometry,view:DrawingFieldView):FeatureGraph{
 let clip=clips.get(view);if(!clip){clip=inverseMatrix(view.field.source.worldFromClip);clips.set(view,clip);}
 const centers=new Map<number,readonly number[]>();
 const center=(face:number)=>{
  let projected=centers.get(face);if(projected)return projected;
  const p=[0,0,0];for(let k=0;k<3;k++){const vertex=geometry.indices[face*3+k]!;for(let axis=0;axis<3;axis++)p[axis]!+=geometry.positions[vertex*3+axis]!/3;}
  const q=transformPoint(clip!,geometry.worldFromSource?transformPoint(geometry.worldFromSource,p):p);
  projected=[view.rect.x+(q[0]!+1)*view.rect.width/2,view.rect.y+(1-q[1]!)*view.rect.height/2];centers.set(face,projected);return projected;
 };
 return selectContourGraph({...graph,relations:graph.relations.filter(relation=>relation.kind==='boundary')},relation=>{
  const face=Number(relation.regions[0].slice(relation.regions[0].lastIndexOf('/face-')+6));if(face<0||face*3+2>=geometry.indices.length)return 0;
  const a=relationPoint(relation,0),b=relationPoint(relation,1),c=relation.frontFacePoint??(geometry.sourceOnly?undefined:center(face));
  return c?(b[0]-a[0])*(c[1]!-a[1])-(b[1]-a[1])*(c[0]!-a[0]):0;
 });
}

type Point=ArrayLike<number>;
interface Ring {faces:number[];edges:Map<string,number[]>}
const rings=new WeakMap<MeshPartGraph,Map<number,Ring>>();
const subtract=(a:Point,b:Point)=>[0,1,2].map(k=>a[k]!-b[k]!);
const cross=(a:Point,b:Point)=>[a[1]!*b[2]!-a[2]!*b[1]!,a[2]!*b[0]!-a[0]!*b[2]!,a[0]!*b[1]!-a[1]!*b[0]!];
const dot=(a:Point,b:Point)=>a[0]!*b[0]!+a[1]!*b[1]!+a[2]!*b[2]!;

/** Resolve a projected contour's rotation system at ambiguous SOURCE
 * vertices. A raw degree-four vertex is often TWO front-surface fans, not a
 * four-way line junction. Follow the original triangle/edge fan, never the
 * nearest screen segment. Source addresses and physical part incidence stay
 * intact; this suffix names only the current-view contour continuation.
 */
export function contourFans(graph:PosedGraph,geometry:CompositionGeometry,structure:MeshPartGraph,view:DrawingFieldView):PosedGraph{
 let topology=rings.get(structure);
 const ix=geometry.indices,representatives=structure.vertexRepresentatives;
 if(!topology){
  topology=new Map();
  for(let f=0;f<ix.length/3;f++)for(let k=0;k<3;k++){
   const v=representatives[ix[f*3+k]!]!,ring=topology.get(v)??{faces:[],edges:new Map<string,number[]>()};ring.faces.push(f);
   for(const at of [(k+1)%3,(k+2)%3]){const other=representatives[ix[f*3+at]!]!,key=`${Math.min(v,other)}/${Math.max(v,other)}`,list=ring.edges.get(key)??[];list.push(f);ring.edges.set(key,list);}
   topology.set(v,ring);
  }
  rings.set(structure,topology);
 }
 const model=geometry.worldFromSource,p=geometry.positions,n=geometry.normals;
 const point=(vertex:number)=>{const q=p.subarray(vertex*3,vertex*3+3);return model?transformPoint(model,q):[...q];};
 const normal=(vertex:number)=>{
  const q=n!.subarray(vertex*4,vertex*4+3);if(!model)return [...q];
  const a=cross(model.subarray(4,7),model.subarray(8,11)),b=cross(model.subarray(8,11),model.subarray(0,3)),c=cross(model.subarray(0,3),model.subarray(4,7));
  const result=[0,1,2].map(k=>q[0]!*a[k]!+q[1]!*b[k]!+q[2]!*c[k]!),length=Math.hypot(...result);return result.map(v=>v/(length||1));
 };
 const facing=new Map<number,boolean>();
 const front=(face:number)=>{
  let value=facing.get(face);if(value!==undefined)return value;
  const vertices=[ix[face*3]!,ix[face*3+1]!,ix[face*3+2]!],points=vertices.map(point);let faceNormal=cross(subtract(points[1]!,points[0]!),subtract(points[2]!,points[0]!));
  if(n){const sum=[0,0,0];for(const vertex of vertices){const v=normal(vertex);for(let k=0;k<3;k++)sum[k]!+=v[k]!;}if(dot(faceNormal,sum)<0)faceNormal=faceNormal.map(v=>-v);}
  const toward=view.pose?.perspective?points[0]!.map((_,k)=>view.pose!.eye[k]!-(points[0]![k]!+points[1]![k]!+points[2]![k]!)/3):(view.pose?.forward??[0,0,1]).map(v=>-v);
  value=dot(faceNormal,toward)>0;facing.set(face,value);return value;
 };
 const incidence=new Map<string,PosedRelation[]>();
 for(const relation of graph.relations)if(relation.kind!=='smooth-contour')for(const port of relation.ports){const list=incidence.get(port)??[];list.push(relation);incidence.set(port,list);}
 const aliases=new Map<string,Map<number,string>>();
 for(const [port,relations]of incidence){
  if(relations.length<=2)continue;
  const match=/\/port-(\d+)$/.exec(port),vertex=match?Number(match[1]):-1;
  if(vertex<0||vertex>=representatives.length)continue;
  const ring=topology.get(representatives[vertex]!);if(!ring)continue;
  const parent=new Map(ring.faces.map(face=>[face,face]));
  const root=(face:number)=>{let at=face;while(parent.get(at)!==at)at=parent.get(at)!;return at;};
  for(const faces of ring.edges.values())for(let i=0;i<faces.length;i++)for(let j=i+1;j<faces.length;j++){
   const a=faces[i]!,b=faces[j]!;if(front(a)!==front(b))continue;
   const x=root(a),y=root(b);parent.set(Math.max(x,y),Math.min(x,y));
  }
  const parts=new Set(ring.faces.map(root));if(parts.size<=1)continue;
  const labels=new Map<number,string>();for(const face of ring.faces)labels.set(face,`${port}/fan-${root(face)}`);aliases.set(port,labels);
 }
 if(!aliases.size)return graph;
 const relations=graph.relations.map(relation=>{
  if(relation.kind==='smooth-contour')return relation;
  const face=Number(relation.regions[0].slice(relation.regions[0].lastIndexOf('/face-')+6));
  return {...relation,ports:relation.ports.map(port=>aliases.get(port)?.get(face)??port) as [string,string]};
 });
 const ports=new Map<string,PosedPort>(),originalRelations=new Map(graph.relations.map(relation=>[relation.id,relation]));
 for(const relation of relations)for(const [index,id]of relation.ports.entries()){
  const prior=ports.get(id),original=graph.ports.get(originalRelations.get(relation.id)!.ports[index]!);
  ports.set(id,{id,homogeneous:original?.homogeneous??(index===0?relation.a:relation.b),members:[...(prior?.members??[]),relation.id]});
 }
 return {...graph,relations,ports};
}
