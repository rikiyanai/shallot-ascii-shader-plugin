import type {MeshPartGraph,PartBand} from './mesh-part-graph.ts';

/** Band-to-section vertex incidence, owned by the source (crawl brief item F,
 * T12). For one band of a MeshPartGraph, labels the band's face vertices
 * with the section id whose contour they continue: for each lower section
 * the band-side endpoint of every sample is the endpoint further along the
 * graph axis (larger dot), for each upper section the endpoint less far
 * along it (smaller dot). Only representatives that are band face vertices
 * are kept. A representative claimed by two different sections is ambiguous
 * and omitted, never guessed; a sample whose endpoints tie on the axis is
 * skipped. Iteration is deterministic (lower then upper in array order,
 * samples in order); a section id with no section is ignored. No camera,
 * screen grid, glyph bank or fitted image participates.
 *
 * @param graph mesh part graph that owns the band and sections.
 * @param band band whose lower/upper section incidences to resolve.
 * @param indices triangle index stream the band faces address.
 * @param positions vertex position stream (xyz per vertex).
 * @returns representative vertex -> incident section id.
 */
export function sourceBandIncidence(graph:MeshPartGraph,band:PartBand,indices:ArrayLike<number>,positions:ArrayLike<number>):ReadonlyMap<number,number>{
 const axis=graph.axis;
 const along=(v:number):number=>positions[v*3]!*axis[0]!+positions[v*3+1]!*axis[1]!+positions[v*3+2]!*axis[2]!;
 const rep=(v:number):number=>graph.vertexRepresentatives[v]??v;
 const faceVertices=new Set<number>();
 for(const face of band.faces)for(let k=0;k<3;k++)faceVertices.add(rep(indices[face*3+k]!));
 const byId=new Map<number,MeshPartGraph['sections'][number]>();
 for(const section of graph.sections)byId.set(section.id,section);
 const labels=new Map<number,number>(),ambiguous=new Set<number>();
 const claim=(vertex:number,section:number):void=>{
  if(ambiguous.has(vertex))return;
  const prior=labels.get(vertex);
  if(prior===undefined){labels.set(vertex,section);return;}
  if(prior!==section){labels.delete(vertex);ambiguous.add(vertex);}
 };
 for(let side=0;side<2;side++){
  const lower=side===0;
  for(const id of (lower?band.lower:band.upper)){
   const section=byId.get(id);
   if(!section)continue;
   for(const sample of section.samples){
    const da=along(sample.a),db=along(sample.b);
    if(da===db)continue;
    const end=lower?(da>db?sample.a:sample.b):(da<db?sample.a:sample.b);
    const vertex=rep(end);
    if(!faceVertices.has(vertex))continue;
    claim(vertex,id);
   }
  }
 }
 return labels;
}
