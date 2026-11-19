/** Authored part/assembly input, not a classification inferred from mesh names,
 * colour or the current camera. Glyph construction remains a separate consumer.
 * A primitive range names the ORIGINAL index stream of one immutable revision. */
export const SOURCE_CONSTRUCTION_SCHEMA='shallot.source-construction.v1' as const;
export const SOURCE_PART_ROLES=['blade','stem','head','trunk','crown','solid'] as const;
export type SourcePartRole=typeof SOURCE_PART_ROLES[number];
/** Addresses in the immutable ORIGINAL source, never current screen samples. */
export type SourcePathSite=
 | {readonly kind:'vertex';readonly vertex:number}
 | {readonly kind:'face';readonly face:number;readonly barycentric:readonly [number,number,number]};
export interface SourcePathFeature {
 readonly id:number;
 readonly sites:readonly SourcePathSite[];
 /** Stable start/end ports. Shared port IDs explicitly declare a join. */
 readonly ports:readonly [number,number];
 readonly rank:number;
}
export interface SourcePathPair {
 readonly id:number;
 readonly paths:readonly [number,number];
 /** Each row joins one endpoint of path A to its corresponding endpoint of B.
  * Reversing B is explicit; source iteration order does not decide the pairing. */
 readonly endPorts:readonly [readonly [number,number],readonly [number,number]];
 /** Source-attached protected gap polygon, closed implicitly. */
 readonly gap:readonly SourcePathSite[];
}
export interface SourceConstruction {
 readonly schema:typeof SOURCE_CONSTRUCTION_SCHEMA;
 readonly role:SourcePartRole;
 readonly components:readonly {
  readonly primitive:'box';
  readonly center:readonly [number,number,number];
  readonly size:readonly [number,number,number];
  readonly firstIndex:number;
  readonly indexCount:number;
 }[];
 readonly paths?:readonly SourcePathFeature[];
 readonly pathPairs?:readonly SourcePathPair[];
}
/** Same-row companion to an existing GPU producer's placements. Row is ONLY
 * an address: records[row*6..] = signed parent X/Y/Z bits, opaque authored
 * family, part ordinal, sibling count. Namespace + the
 * first four words owns parent identity; atomic allocation order never does.
 * The original engine packed-instance row must address this SAME column. */
export const SOURCE_ASSEMBLY_WORDS=6 as const;
export interface SourceAssemblyColumn {
 readonly schema:'shallot.source-assembly.v1';
 readonly namespace:string;
 readonly records:GPUBufferBinding;
 readonly stride:24;
}
const sourceId=(value:number)=>Number.isSafeInteger(value)&&value>=0&&value<=0xffffffff;
function validatePathSite(site:SourcePathSite,indexCount:number,vertexCount:number):void{
 if(site.kind==='vertex'){
  if(!sourceId(site.vertex)||site.vertex>=vertexCount)throw Error('Authored path vertex is outside the original source');
 }else if(site.kind==='face'){
  const weights=site.barycentric;
  if(!sourceId(site.face)||site.face>=indexCount/3||weights.length!==3||!weights.every(n=>Number.isFinite(n)&&n>=0&&n<=1)||Math.abs(weights[0]+weights[1]+weights[2]-1)>1e-6)
   throw Error('Invalid authored path face site');
 }else throw Error('Unsupported authored path site');
}
export function validateSourceConstruction(value:SourceConstruction,indexCount:number,vertexCount?:number):void{
 if(value.schema!==SOURCE_CONSTRUCTION_SCHEMA||!SOURCE_PART_ROLES.includes(value.role)||!Number.isSafeInteger(indexCount)||indexCount<0)
  throw Error('Invalid authored source construction');
 let end=0;
 for(const component of value.components){
  if(component.primitive!=='box'||component.center.length!==3||component.size.length!==3||!component.center.every(Number.isFinite)||
   !component.size.every(n=>Number.isFinite(n)&&n>0)||component.firstIndex!==end||!Number.isSafeInteger(component.indexCount)||component.indexCount<=0||component.indexCount%3)
   throw Error('Invalid authored construction component range');
  end+=component.indexCount;
 }
 // Path annotations may supplement arbitrary source geometry without
 // pretending it is a collection of boxes. Existing box coverage stays exact.
 if((value.components.length>0||!value.paths?.length)&&end!==indexCount)throw Error('Authored construction does not cover original indices');
 const paths=value.paths??[],pairs=value.pathPairs??[];
 if(paths.length||pairs.length){
  if(!Number.isSafeInteger(vertexCount)||vertexCount!<1||indexCount%3)throw Error('Authored paths require original source extents');
  const byId=new Map<number,SourcePathFeature>();
  for(const path of paths){
   if(!sourceId(path.id)||byId.has(path.id)||path.sites.length<2||path.ports.length!==2||!path.ports.every(sourceId)||!Number.isFinite(path.rank)||path.rank<0)
    throw Error('Invalid authored path feature or identity');
   for(const site of path.sites)validatePathSite(site,indexCount,vertexCount!);
   byId.set(path.id,path);
  }
  const pairIds=new Set<number>();
  for(const pair of pairs){
   if(!sourceId(pair.id)||pairIds.has(pair.id)||pair.paths.length!==2||pair.paths[0]===pair.paths[1]||pair.endPorts.length!==2||pair.endPorts.some(ports=>ports.length!==2))
    throw Error('Invalid authored path pair identity');
   const a=byId.get(pair.paths[0]),b=byId.get(pair.paths[1]);
   if(!a||!b)throw Error('Authored path pair references an absent feature');
   for(const [column,path]of [a,b].entries()){
    const actual=pair.endPorts.map(ports=>ports[column]);
    if(actual[0]===actual[1]||!path.ports.every(port=>actual.includes(port))||!actual.every(port=>path.ports.includes(port!)))
     throw Error('Authored path pair does not map both source endpoints');
   }
   if(pair.gap.length<3||new Set(pair.gap.map(site=>JSON.stringify(site))).size<3)throw Error('Authored path pair requires its protected source gap');
   for(const site of pair.gap)validatePathSite(site,indexCount,vertexCount!);
   pairIds.add(pair.id);
  }
 }
}
/** Load/change-time resolution for compilation. Runtime pose consumers keep
 * these original addresses rather than reattaching to visible fragments. */
export function sourcePathPoint(site:SourcePathSite,positions:Float32Array,indices:Uint32Array):readonly [number,number,number]{
 const point=(vertex:number)=>[positions[vertex*3]!,positions[vertex*3+1]!,positions[vertex*3+2]!] as const;
 if(site.kind==='vertex')return point(site.vertex);
 const at=site.face*3,a=point(indices[at]!),b=point(indices[at+1]!),c=point(indices[at+2]!),w=site.barycentric;
 return [a[0]*w[0]+b[0]*w[1]+c[0]*w[2],a[1]*w[0]+b[1]*w[1]+c[1]*w[2],a[2]*w[0]+b[2]*w[1]+c[2]*w[2]];
}
