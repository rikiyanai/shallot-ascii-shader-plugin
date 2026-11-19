import type {MeshSourceComplex,SourceEdge,SourceHalfedge} from './source-complex.ts';

/** Compiled relation consumers need a count, not an inspectable CPU topology.
 * This is a different result contract, never an empty MeshSourceComplex. */
export interface SourceResidencySummary {
 readonly schema:'shallot.source-residency-summary.v1';
 readonly edgeCount:number;
 readonly vertexCount?:number;readonly faceCount?:number;
}
export function sourceResidencyEdgeCount(source:{readonly summary:SourceResidencySummary}|{readonly complex:MeshSourceComplex}|{readonly gpu:{readonly incidence:{readonly edges:number}}}):number {
 if('gpu' in source)return source.gpu.incidence.edges;
  return 'summary' in source?source.summary.edgeCount:sourceComplexEdgeCount(source.complex);
}
export function sourceResidencyVertexCount(source:{readonly summary?:SourceResidencySummary;readonly complex?:MeshSourceComplex;readonly geometry?:{readonly positions:Float32Array};readonly gpu?:{readonly topology:{readonly vertexCapacity:number}}}):number{
 if(source.gpu)return source.gpu.topology.vertexCapacity;
 if(source.summary?.vertexCount!==undefined)return source.summary.vertexCount;
 if(source.geometry)return source.geometry.positions.length/3;
 if(source.complex){let count=0;for(const h of source.complex.halfedges)count=Math.max(count,h.from+1,h.to+1);return count;}
 throw Error('GPU source has no authoritative vertex extent');
}
export function sourceResidencyFaceCount(source:{readonly summary?:SourceResidencySummary;readonly complex?:MeshSourceComplex;readonly geometry?:{readonly indices:Uint32Array};readonly gpu?:{readonly incidence:{readonly layout:{readonly corners:number}}}}):number{
 if(source.gpu)return source.gpu.incidence.layout.corners/3;
 if(source.summary?.faceCount!==undefined)return source.summary.faceCount;
 if(source.geometry)return source.geometry.indices.length/3;
 if(source.complex)return source.complex.faceComponents.length;
 throw Error('GPU source has no authoritative face extent');
}

/** Worker transport, not a GPU packet/topology owner. Decisions are already
 * compiled. Transfer numeric columns instead of cloning millions of objects.
 * Explicit CPU inspection still receives ordinary arrays and plain records. */
export interface SourceComplexColumns {
 readonly schema:'shallot.source-complex-columns.v1';
 readonly data:Omit<MeshSourceComplex,'halfedges'|'edges'|'boundaries'|'contacts'|'vertexContacts'>;
 readonly halfedges:Uint32Array;readonly edges:Uint32Array;readonly members:Uint32Array;
 readonly boundaries:readonly {data:Omit<MeshSourceComplex['boundaries'][number],'halfedges'>;at:number;count:number}[];
 readonly contacts:readonly {kind:MeshSourceComplex['contacts'][number]['kind'];at:number;count:number}[];
 readonly contactMembers:Uint32Array;readonly vertexContacts?:Uint32Array;readonly hasVertexContacts:boolean;
}
const HALF_WORDS=7,EDGE_WORDS=8,NONE=0xffffffff;
const edgeCounts=new WeakMap<MeshSourceComplex,number>();
export function sourceComplexEdgeCount(complex:MeshSourceComplex):number {
 return edgeCounts.get(complex)??complex.edges.length;
}
export function encodeSourceComplex(complex:MeshSourceComplex):SourceComplexColumns {
 const {halfedges,edges,boundaries,contacts,vertexContacts,...data}=complex;
 const half=new Uint32Array(halfedges.length*HALF_WORDS),edge=new Uint32Array(edges.length*EDGE_WORDS);
 const members:number[]=[],lists=new Map<readonly number[],readonly [number,number]>();
 const list=(values:readonly number[]):readonly [number,number]=>{
  const old=lists.get(values);if(old)return old;
  const range=[members.length,values.length] as const;
  for(const value of values)members.push(value);lists.set(values,range);return range;
 };
 for(let i=0;i<halfedges.length;i++){
  const h=halfedges[i]!;half.set([h.id,h.from,h.to,h.face,h.next,h.edge,h.twin??NONE],i*HALF_WORDS);
 }
 for(let i=0;i<edges.length;i++){
  const e=edges[i]!,inc=list(e.incidences),weld=e.welded?list(e.welded):[NONE,0];
  edge.set([e.id,e.vertices[0],e.vertices[1],inc[0]!,inc[1]!,weld[0]!,weld[1]!,Number(Object.hasOwn(e,'welded'))],i*EDGE_WORDS);
 }
 const boundary=boundaries.map(({halfedges,...data})=>{const [at,count]=list(halfedges);return {data,at,count};});
 const contactMembers:number[]=[],contact=contacts.map(c=>{
  const at=contactMembers.length;for(const m of c.members)contactMembers.push(m.edge,Number(m.forward));
  return {kind:c.kind,at,count:c.members.length};
 });
 const vertices=vertexContacts?new Uint32Array(vertexContacts.length*2):undefined;
 if(vertices)vertexContacts!.forEach((group,i)=>vertices.set(list(group),i*2));
 return {schema:'shallot.source-complex-columns.v1',data,halfedges:half,edges:edge,members:Uint32Array.from(members),
  boundaries:boundary,contacts:contact,contactMembers:Uint32Array.from(contactMembers),vertexContacts:vertices,
  hasVertexContacts:Object.hasOwn(complex,'vertexContacts')};
}
export function decodeSourceComplex(wire:SourceComplexColumns):MeshSourceComplex {
 if(wire.schema!=='shallot.source-complex-columns.v1'||wire.halfedges.length%HALF_WORDS||wire.edges.length%EDGE_WORDS||wire.vertexContacts&&wire.vertexContacts.length%2)
  throw Error('Invalid source complex transfer columns');
 const lists=new Map<string,readonly number[]>();
 const list=(at:number,count:number):readonly number[]=>{
  const key=`${at}/${count}`;let value=lists.get(key);
  if(!value){if(at+count>wire.members.length)throw Error('Source transfer member range is invalid');value=Array.from(wire.members.subarray(at,at+count));lists.set(key,value);}
  return value;
 };
 let half:SourceHalfedge[]|undefined,edges:SourceEdge[]|undefined,boundaries:MeshSourceComplex['boundaries']|undefined,
  contacts:MeshSourceComplex['contacts']|undefined,vertices:MeshSourceComplex['vertexContacts'];
 const complex:MeshSourceComplex={...wire.data,
  get halfedges(){return half??=Array.from({length:wire.halfedges.length/HALF_WORDS},(_,i)=>{
   const h=wire.halfedges,k=i*HALF_WORDS;
   return {id:h[k]!,from:h[k+1]!,to:h[k+2]!,face:h[k+3]!,next:h[k+4]!,edge:h[k+5]!,twin:h[k+6]===NONE?undefined:h[k+6]};
  });},
  get edges(){return edges??=Array.from({length:wire.edges.length/EDGE_WORDS},(_,i)=>{
   const e=wire.edges,k=i*EDGE_WORDS;
   return {id:e[k]!,vertices:[e[k+1]!,e[k+2]!] as const,incidences:list(e[k+3]!,e[k+4]!),
    ...(e[k+7]?{welded:e[k+5]===NONE?undefined:list(e[k+5]!,e[k+6]!)}:{})};
  });},
  get boundaries(){return boundaries??=wire.boundaries.map(b=>({...b.data,halfedges:list(b.at,b.count)}));},
  get contacts(){return contacts??=wire.contacts.map(c=>({kind:c.kind,members:Array.from({length:c.count},(_,i)=>({edge:wire.contactMembers[c.at+i*2]!,forward:wire.contactMembers[c.at+i*2+1]!==0}))}));},
 };
 if(wire.hasVertexContacts)Object.defineProperty(complex,'vertexContacts',{enumerable:true,configurable:true,get(){
  if(!wire.vertexContacts)return undefined;
  return vertices??=Array.from({length:wire.vertexContacts.length/2},(_,i)=>list(wire.vertexContacts![i*2]!,wire.vertexContacts![i*2+1]!));
 }});
 edgeCounts.set(complex,wire.edges.length/EDGE_WORDS);return complex;
}
