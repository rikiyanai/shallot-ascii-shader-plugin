import type {CompiledConstructionGraph,SourcePlaneRegion} from './composition.ts';
import {PART_CAP,PART_CHAIN_SHIFT,PART_FORK,PART_LIMB,PART_ROOT,partPoint,type MeshPartGraph,type PartSection} from './mesh-part-graph.ts';
import type {MeshSourceComplex} from './source-complex.ts';

/**
 * GPU-resident representation of the source construction graph.
 *
 * This is deliberately a word-addressed packet rather than a collection of
 * JavaScript objects.  Each source is one SLOT: its header (HEADER_WORDS) at
 * slot word 0, then its records. Every range in a header or record is a word
 * offset RELATIVE TO THE SLOT'S OWN BASE, the address of its header (since
 * 2026-11-12; before, ranges were absolute in one whole packet and every
 * repack relocated them). A reader adds the header address it found the
 * source at. A slot is immutable for a source revision; placing it anywhere,
 * or moving it, rewrites none of its words, and a change to one source
 * replaces that source's slot only (GPU residency: gpu-construction-runs.ts).
 *
 * No camera, projected coordinate, glyph candidate, pen position or image is
 * included.  The packet is source topology used by current-view GPU
 * articulation.
 */
export const TOPOLOGY_HEADER_WORDS=44;
/** Compact run packets use the otherwise-unused origin.z header word. This
 * NaN bit pattern cannot alias the finite origin in a full inspection packet. */
export const PACKED_RUN_CONTEXT=0x7fc70f01;
const CONTEXT_INDEX_MASK=0x07ffffff,CONTEXT_TAG_MASK=0x60000000;
export const TOPOLOGY_SECTION_WORDS=16;
export const TOPOLOGY_BAND_WORDS=10;
export const TOPOLOGY_PLANE_WORDS=8;
export const TOPOLOGY_INCIDENCE_WORDS=3;
/** Packet revision. 2 (2026-11-09) fills words that revision 1 left zero; no
 * record stride and no existing field offset changed:
 *  - section record word TOPOLOGY_SECTION_LABEL_WORD and band record word
 *    TOPOLOGY_BAND_LABEL_WORD hold the part label word of mesh-part-graph.ts
 *    (bit 0 PART_LIMB, bit 1 PART_FORK, bit 2 PART_ROOT, bit 3 PART_CAP,
 *    bits 8..31 chain id + 1, 0 = no chain);
 *  - plane record words TOPOLOGY_PLANE_LOOPS_WORD (offset of the
 *    region's loop table: one [halfedge list offset, halfedge count] pair per
 *    boundary cycle, outline first), TOPOLOGY_PLANE_LOOP_COUNT_WORD (cycles
 *    in that table) and TOPOLOGY_PLANE_OPENING_COUNT_WORD (how many of them
 *    are openings: the cycles after the outline);
 *  - header word 1 may hold TOPOLOGY_TYPE_TERRAIN (5).
 * 3 (2026-11-12): every offset named above is relative to the source's slot
 * base (its header address), no longer absolute in a whole packet.
 * 4 (2026-11-17, face construction slice 1): a RUN slot of a source with wall
 * faces (composition.ts compileWallFaces) holds the face table. Header word
 * TOPOLOGY_FACE_MAGIC_WORD is TOPOLOGY_FACE_MAGIC, word +1 the offset of the
 * face edge words (one per source edge address, as planeEdges; 0 = no face,
 * else face index + 1 in bits 0-7 and the edge's role in bits 8-9: 1 outline,
 * 2 opening, 3 compiled out; bit 10 FACE_HIDDEN: the edge is drawn by nothing,
 * slice 1b), +2 their count, +3 the face count, +4 the offset of
 * the face records (TOPOLOGY_FACE_WORDS each: main plane-region id, region
 * count, corner list offset, corner count, opening count, member count, f32
 * area share, frame count; the corner list holds welded source vertex ids), +5 the
 * member width (f32, source units: SourceFace.width). A slot without faces
 * leaves the six words zero. No other word or stride changed. */
export const TOPOLOGY_PACKET_REVISION=5;
export const TOPOLOGY_FACE_MAGIC_WORD=2,TOPOLOGY_FACE_MAGIC=0x7ace0f01,TOPOLOGY_FACE_WORDS=8;
export const FACE_ROLE_OUTLINE=1,FACE_ROLE_OPENING=2,FACE_ROLE_MEMBER=3;
/** Slice 1b: bit 10 of a face edge word: the edge is compiled out (SourceFace.hidden). */
export const FACE_HIDDEN=0x400;
export const TOPOLOGY_SECTION_LABEL_WORD=15;
export const TOPOLOGY_BAND_LABEL_WORD=9;
export const TOPOLOGY_PLANE_LOOPS_WORD=5;
export const TOPOLOGY_PLANE_LOOP_COUNT_WORD=6;
export const TOPOLOGY_PLANE_OPENING_COUNT_WORD=7;
export const TOPOLOGY_TYPE_TERRAIN=5;











export const TOPOLOGY_LIMB_MAGIC_WORD=8,TOPOLOGY_LIMB_MAGIC=0x11b0c4a1;

















export const TOPOLOGY_PART_MAGIC_WORD=11,TOPOLOGY_PART_MAGIC=0x50415254,TOPOLOGY_PART_PACKET_MAGIC=0x50505232,TOPOLOGY_PART_PROGRAM_WORDS=12;

export interface ResidentTopologySource {
 readonly domain:number;
 readonly type:CompiledConstructionGraph['type'];
 readonly headerWord:number;
 readonly structure?:MeshPartGraph;
 readonly planeRegions:readonly SourcePlaneRegion[];
 readonly complex?:MeshSourceComplex;
}
export interface PackedConstructionTopology {
 readonly words:Uint32Array;
 readonly sources:readonly ResidentTopologySource[];
 readonly fingerprint:string;
}

const TYPE_CODE:Readonly<Record<CompiledConstructionGraph['type'],number>>={
 'paper':0,'path-led':1,'planar':2,'rounded':3,'far-strip':4,'terrain':TOPOLOGY_TYPE_TERRAIN,
};
const f32bits=(value:number):number=>new Uint32Array(new Float32Array([value]).buffer)[0]!;
// A plain loop: Uint32Array.from with a mapper walks the iterator protocol and
// allocated about 1 GB of garbage per packet on a 700 000 triangle world.
const words=(values:readonly number[]):Uint32Array=>{const out=new Uint32Array(values.length);for(let i=0;i<values.length;i++)out[i]=values[i]!>>>0;return out;};
const range=(at:number,count:number):[number,number]=>[at,count];

function sourceGraph(item:CompiledConstructionGraph):MeshPartGraph|undefined{return item.structure;}

function appendList(target:number[],values:ArrayLike<number>):[number,number]{
 const at=target.length;for(let i=0;i<values.length;i++)target.push(values[i]!>>>0);return range(at,values.length);
}

function appendSections(target:number[],graph:MeshPartGraph):[number,number]{
 const at=target.length;
 const records=target.length;target.length+=graph.sections.length*TOPOLOGY_SECTION_WORDS;
 for(const section of graph.sections){
  const sample=appendList(target,section.samples.flatMap(point=>[point.a,point.b,f32bits(point.t)]));
  const segments=appendList(target,section.segments.flatMap(segment=>segment));
  const faces=appendList(target,section.faces);
  const record=records+section.id*TOPOLOGY_SECTION_WORDS;
  target[record]=section.id;target[record+1]=section.component;target[record+2]=f32bits(section.level);
  target[record+3]=(section.closed?1:0)|(section.inside===undefined?0:2);
  target[record+4]=f32bits(section.center[0]);target[record+5]=f32bits(section.center[1]);target[record+6]=f32bits(section.center[2]);
  target[record+7]=sample[0];target[record+8]=sample[1]/3;target[record+9]=segments[0];target[record+10]=segments[1]/3;target[record+11]=faces[0];target[record+12]=faces[1];
  target[record+13]=section.inside===undefined?0xffffffff:section.inside;target[record+14]=section.id;target[record+TOPOLOGY_SECTION_LABEL_WORD]=(section.labels??0)>>>0;
 }
 return range(at,graph.sections.length);
}

function appendBands(target:number[],graph:MeshPartGraph):[number,number]{
 const at=target.length;
 const records=target.length;target.length+=graph.bands.length*TOPOLOGY_BAND_WORDS;
 for(const band of graph.bands){
  const lower=appendList(target,band.lower),upper=appendList(target,band.upper),faces=appendList(target,band.faces),critical=appendList(target,band.criticalVertices);
  const record=records+band.id*TOPOLOGY_BAND_WORDS;
  target[record]=band.id;target[record+1]=lower[0];target[record+2]=lower[1];target[record+3]=upper[0];target[record+4]=upper[1];target[record+5]=faces[0];target[record+6]=faces[1];target[record+7]=critical[0];target[record+8]=critical[1];target[record+TOPOLOGY_BAND_LABEL_WORD]=(band.labels??0)>>>0;
 }
 return range(at,graph.bands.length);
}

function appendPlanes(target:number[],regions:readonly SourcePlaneRegion[],complex?:MeshSourceComplex):{planes:[number,number];incidences:[number,number]}{
 const planesAt=target.length;target.length+=regions.length*TOPOLOGY_PLANE_WORDS;
 const regionFaceLists=regions.map(region=>appendList(target,region.faces));
 for(const [index,region]of regions.entries()){
  const list=regionFaceLists[index]!,record=planesAt+index*TOPOLOGY_PLANE_WORDS;
  target[record]=region.id;target[record+1]=list[0];target[record+2]=list[1];target[record+3]=0;target[record+4]=0;target[record+5]=0;target[record+6]=0;target[record+7]=0;
 }
 const memberships=new Map<number,number[]>();
 for(const [index,region]of regions.entries())for(const face of region.faces){const list=memberships.get(face)??[];list.push(index);memberships.set(face,list);}
 // Plane adjacency is shared SOURCE EDGE incidence, not two overlapping
 // face lists. Preserve every incident region, including nonmanifold edges.
 // The incidence is the WELDED one: on a triangle soup the two faces of a
 // shared edge name it with different indices, so index incidence found no
 // shared edge at all. One record per welded edge, at its canonical address
 // (its lowest halfedge, the id an indexed mesh already gave it).
 const joins:{edge:number;regions:number[]}[]=[];
 for(const edge of complex?.edges??[]){
  const welded=edge.welded??edge.incidences;
  if(edge.welded&&edge.welded[0]!==edge.id)continue;
  const memberIndices=new Set<number>();
  for(const halfedge of welded)for(const region of memberships.get(complex!.halfedges[halfedge]!.face)??[])memberIndices.add(region);
  if(memberIndices.size>1)joins.push({edge:edge.id,regions:[...memberIndices].sort((a,b)=>a-b)});
 }
 const incidenceAt=target.length;target.length+=joins.length*TOPOLOGY_INCIDENCE_WORDS;
 const byPlane=regions.map(()=>[] as number[]);
 for(const [index,join]of joins.entries()){
  const list=appendList(target,join.regions.map(region=>regions[region]!.id)),at=incidenceAt+index*TOPOLOGY_INCIDENCE_WORDS;
  target[at]=join.edge;target[at+1]=list[0];target[at+2]=list[1];
  for(const region of join.regions)byPlane[region]!.push(index);
 }
 byPlane.forEach((indices,index)=>{const list=appendList(target,indices),at=planesAt+index*TOPOLOGY_PLANE_WORDS;target[at+3]=list[0];target[at+4]=list[1];});
 // Outline and openings of each region, compiled with the region (welded
 // boundary cycles as original halfedge addresses). A region compiled
 // without them keeps the three words zero.
 for(const [index,region]of regions.entries()){
  const loops=[...(region.boundary?.length?[region.boundary]:[]),...(region.openings??[])];if(!loops.length)continue;
  const lists=loops.map(loop=>appendList(target,loop)),table=appendList(target,lists.flat()),at=planesAt+index*TOPOLOGY_PLANE_WORDS;
  target[at+TOPOLOGY_PLANE_LOOPS_WORD]=table[0];target[at+TOPOLOGY_PLANE_LOOP_COUNT_WORD]=loops.length;target[at+TOPOLOGY_PLANE_OPENING_COUNT_WORD]=region.openings?.length??0;
 }
 return {planes:range(planesAt,regions.length),incidences:range(incidenceAt,joins.length)};
}

function appendComplex(target:number[],complex:MeshSourceComplex|undefined):{halfedges:[number,number];edges:[number,number];boundaries:[number,number];contacts:[number,number];vertexContacts:[number,number];faceComponents:[number,number]}{
 const empty:[number,number]=[target.length,0];if(!complex)return {halfedges:empty,edges:empty,boundaries:empty,contacts:empty,vertexContacts:empty,faceComponents:empty};
 const halfedgesAt=target.length;const halfedgeStride=8;target.length+=complex.halfedges.length*halfedgeStride;
 for(const edge of complex.halfedges){const at=halfedgesAt+edge.id*halfedgeStride;target[at]=edge.id;target[at+1]=edge.from;target[at+2]=edge.to;target[at+3]=edge.face;target[at+4]=edge.next;target[at+5]=edge.edge;target[at+6]=edge.twin??0xffffffff;target[at+7]=complex.faceComponents[edge.face]??0;}
 const edgesAt=target.length;const edgeStride=6;target.length+=complex.edges.length*edgeStride;
 // The edge record points at a compact incidence list, avoiding a
 // two-incidence assumption while keeping the record fixed-size.
 for(const [index,edge]of complex.edges.entries()){const list=appendList(target,edge.incidences),at=edgesAt+index*edgeStride;target[at]=edge.id;target[at+1]=edge.vertices[0];target[at+2]=edge.vertices[1];target[at+3]=list[0];target[at+4]=list[1];target[at+5]=0;}
 const boundariesAt=target.length;const boundaryStride=4;target.length+=complex.boundaries.length*boundaryStride;
 for(const [index,boundary]of complex.boundaries.entries()){const list=appendList(target,boundary.halfedges),at=boundariesAt+index*boundaryStride;target[at]=boundary.component;target[at+1]=list[0];target[at+2]=list[1];target[at+3]=Number(boundary.closed);}
 const contactsAt=target.length;const contactStride=4;target.length+=complex.contacts.length*contactStride;
 for(const [index,contact]of complex.contacts.entries()){const list=appendList(target,contact.members.flatMap(member=>[member.edge,Number(member.forward)])),at=contactsAt+index*contactStride;target[at]=list[0];target[at+1]=list[1]/2;target[at+2]=contact.kind==='published-source-incidence'?1:0;target[at+3]=0;}
 const vertexContactsAt=target.length;const groups=complex.vertexContacts??[];target.length+=groups.length*2;
 for(const [index,group]of groups.entries()){const list=appendList(target,group),at=vertexContactsAt+index*2;target[at]=list[0];target[at+1]=list[1];}
 const faceComponents=appendList(target,Array.from(complex.faceComponents));
 return {halfedges:range(halfedgesAt,complex.halfedges.length),edges:range(edgesAt,complex.edges.length),boundaries:range(boundariesAt,complex.boundaries.length),contacts:range(contactsAt,complex.contacts.length),vertexContacts:range(vertexContactsAt,groups.length),faceComponents};
}

function appendGraph(target:number[],item:CompiledConstructionGraph,complex?:MeshSourceComplex):{sections:[number,number];bands:[number,number];planes:[number,number];incidences:[number,number];vertices:[number,number];levels:[number,number];halfedges:[number,number];edges:[number,number];boundaries:[number,number];contacts:[number,number];vertexContacts:[number,number];faceComponents:[number,number]}{
 const graph=sourceGraph(item);
 if(!graph){const empty:[number,number]=[target.length,0],region=appendPlanes(target,item.planeRegions??[],complex),source=appendComplex(target,complex);return {sections:empty,bands:empty,planes:region.planes,incidences:region.incidences,vertices:empty,levels:empty,...source};}
 const sections=appendSections(target,graph),bands=appendBands(target,graph),region=appendPlanes(target,item.planeRegions??[],complex);
 const vertices=appendList(target,graph.vertexRepresentatives),levels=appendList(target,graph.levels.map(f32bits));
 return {sections,bands,planes:region.planes,incidences:region.incidences,vertices,levels,...appendComplex(target,complex)};
}

/** Compile direct source-address context lookups. A frame must not walk every
 * section/contact list for every visible relation. These keys describe source
 * context only; a contact candidate is NOT a posed physical weld. */
function appendContextIndices(target:number[],item:CompiledConstructionGraph,complex?:MeshSourceComplex,compact=false):{edges:[number,number];vertices:[number,number];planeEdges:[number,number];packed:boolean}{
 // SourceEdge.id and surface relation key.w are original halfedge addresses,
 // not dense edge-array indices. Halfedge.edge is the dense array index.
 const edgeGroups=new Uint32Array(complex?.halfedges.length??0),vertices=new Uint32Array(item.structure?.vertexRepresentatives.length??0);
 for(const [index,contact]of (complex?.contacts??[]).entries())for(const member of contact.members)if(!edgeGroups[member.edge])edgeGroups[member.edge]=(0x40000000|index)>>>0;
 for(const [index,boundary]of (complex?.boundaries??[]).entries())for(const halfedge of boundary.halfedges){const edge=complex!.edges[complex!.halfedges[halfedge]!.edge]!.id;if(!edgeGroups[edge])edgeGroups[edge]=(0x20000000|index)>>>0;}
 for(const section of item.structure?.sections??[])for(const sample of section.samples)for(const vertex of [sample.a,sample.b])if(!vertices[vertex])vertices[vertex]=(0x10000000|section.id)>>>0;
 for(const band of item.structure?.bands??[])for(const vertex of band.criticalVertices)if(!vertices[vertex])vertices[vertex]=(0x80000000|band.id)>>>0;
 for(let vertex=0;vertex<vertices.length;vertex++){const representative=item.structure!.vertexRepresentatives[vertex]!;if(!vertices[vertex])vertices[vertex]=vertices[representative]||representative;}
 // A mixed mesh may contain architectural planes without the WHOLE mesh
 // being a planar primitive. Compile the plane's boundary/interface edges,
 // not its triangulation diagonals. Current-view relation visibility still
 // decides which of these edges exists in the drawing.
 const facePlane=new Map<number,number>();
 for(const region of item.planeRegions??[])for(const face of region.faces)facePlane.set(face,region.id);
 // A hard line is a property of the source edge, not of the whole mesh's
 // class: an open boundary edge, or two faces meeting at a crease (the same
 // 45° the relation pass uses), is drawn with the straight-line units on a
 // building, a terrain diamond or a hexagon alike. Faces turning by less
 // than that keep the soft contour words.
 // On a volume (at most 24 plane regions, the count the run shader's
 // planarVolume reads from header word 16) the border between two plane
 // regions is a hard edge whatever its angle. A source with more regions is a
 // surface (terrain, heightfield): there the border between two coplanar
 // patches is tessellation, and only the dihedral or an open boundary below
 // makes the edge hard.
 // The table holds the edge's class. 1 is an authored feature edge: an open
 // boundary, a region border of a volume, or a crease with a plane of two or
 // more coplanar faces on at least one side (plane regions hold only such
 // planes), and every crease of a planar mesh. 2 is a facet crease: both
 // sides are single triangles of a triangle soup (canopy, rock, terrain).
 // The run shader draws a facet crease as a soft articulated run.
 // The faces at an edge are read through the compiled contact groups, not
 // through index incidence alone. A source whose triangles are separately
 // indexed (every imported world mesh: 30 faces, 90 edges) has ONE index
 // incidence on every edge; read by index, each triangle side was an open
 // boundary, so every edge of every such mesh was class 1 and every crown,
 // rock and slope was spelled edge by edge with the straight-line units. The
 // contact group is the same incidence the relation pass uses for the crease.
 const joined=new Map<number,readonly number[]>();
 for(const contact of complex?.contacts??[]){
  const faces=[...new Set(contact.members.flatMap(member=>complex!.edges[complex!.halfedges[member.edge]!.edge]!.incidences.map(index=>complex!.halfedges[index]!.face)))];
  for(const member of contact.members)if(!joined.has(member.edge))joined.set(member.edge,faces);
 }
 // Since the source compiler welds rest positions, the welded incidence
 // names both faces of a soup crease directly. The contact group is still
 // read and the two are united: lattice contact keys can join corners the
 // position weld does not, and a hand-built complex has no welded list.
 const planeEdges=new Uint32Array(complex?.halfedges.length??0),normals=item.faceNormals,surface=(item.planeRegions?.length??0)>24,planar=item.type==='planar';
 // One reused face list per edge (no per-edge Set/array: this loop runs over
 // every edge of every arriving source, the largest garbage of a compile).
 const faces:number[]=[];
 const addFace=(face:number):void=>{if(!faces.includes(face))faces.push(face);};
 // A FACE IN A PLANE (line repair, 2026-11-12): a welded neighbour across one
 // of its edges lies within COPLANAR_COS of its own plane. The plane regions
 // above group faces by a quantised plane key; an authored low-poly volume
 // leaves the two triangles of one box side a fraction of a degree apart (the
 // Y8 stair blocks, bridge-1.akm: 361 quantised planes for 381 faces, 17
 // regions), so its box edges read as facet creases and were drawn as soft
 // runs, then dropped. Three degrees is less than one text cell can show.
 // Read once per face: 0 unread, 1 in a plane, 2 not.
 const COPLANAR_COS=0.99862953,sided=new Uint8Array(complex?.faceComponents.length??0);
 const inPlane=(face:number):boolean=>{
  if(sided[face])return sided[face]===1;let found=false;
  for(let k=0;k<3&&!found&&normals;k++){
   const edge=complex!.edges[complex!.halfedges[face*3+k]!.edge]!;
   for(const index of edge.welded??edge.incidences){const other=complex!.halfedges[index]!.face;if(other===face)continue;
    if(normals[face*3]!*normals[other*3]!+normals[face*3+1]!*normals[other*3+1]!+normals[face*3+2]!*normals[other*3+2]!>=COPLANAR_COS){found=true;break;}}
  }
  sided[face]=found?1:2;return found;
 };
 for(const edge of complex?.edges??[]){
  faces.length=0;
  for(const index of edge.welded??edge.incidences)addFace(complex!.halfedges[index]!.face);
  const extra=joined.get(edge.id);if(extra)for(const face of extra)addFace(face);
  let all=faces.length>0,any=false,distinct=false,first:number|undefined;
  for(let i=0;i<faces.length;i++){const plane=facePlane.get(faces[i]!);if(plane===undefined)all=false;else any=true;if(i===0)first=plane;else if(plane!==first)distinct=true;}
  if(all&&(faces.length===1||(!surface&&distinct))){planeEdges[edge.id]=1;continue;}
  if(!normals||!faces.length)continue;
  // One face is an open boundary; three or more is a fin or a branching sheet.
  if(faces.length!==2){planeEdges[edge.id]=1;continue;}
  const a=faces[0]!*3,b=faces[1]!*3,cosine=normals[a]!*normals[b]!+normals[a+1]!*normals[b+1]!+normals[a+2]!*normals[b+2]!;
  if(cosine<0.70710678)planeEdges[edge.id]=planar||any||(inPlane(faces[0]!)&&inPlane(faces[1]!))?1:2;
 }
 // Edge groups use two tag bits and a source-local index. Their class is
 // only 0/1/2, so both fit one word without changing either decoded value.
 // Oversized indices retain the old encoding rather than lose address bits.
 const packed=compact&&edgeGroups.every(group=>(group&0x80000000)===0&&(group&0x1fffffff)<=CONTEXT_INDEX_MASK)&&planeEdges.every(value=>value<=3);
 if(packed){
  for(let i=0;i<edgeGroups.length;i++)edgeGroups[i]=((edgeGroups[i]!&CONTEXT_TAG_MASK)|((edgeGroups[i]!&CONTEXT_INDEX_MASK)<<2)|planeEdges[i]!)>>>0;
  const edges=appendList(target,edgeGroups);
  return {edges,vertices:appendList(target,vertices),planeEdges:edges,packed:true};
 }
 return {edges:appendList(target,edgeGroups),vertices:appendList(target,vertices),planeEdges:appendList(target,planeEdges),packed:false};
}

export type TopologySourceItem={domain:number}&CompiledConstructionGraph&{complex?:MeshSourceComplex;sourcePositions?:Float32Array};

/** Writes one source's header at `header` and appends its records to
 * `target`. Every range is target.length at the time it is written. */
function writeSource(target:number[],header:number,item:TopologySourceItem):void{
 const data=appendGraph(target,item,item.complex);
 target[header]=item.domain;target[header+1]=TYPE_CODE[item.type];
 target[header+2]=data.sections[0];target[header+3]=data.sections[1];
 target[header+4]=data.bands[0];target[header+5]=data.bands[1];
 target[header+6]=data.planes[0];target[header+7]=data.planes[1];
 target[header+8]=data.incidences[0];target[header+9]=data.incidences[1];
 target[header+10]=data.vertices[0];target[header+11]=data.vertices[1];
 target[header+12]=data.levels[0];target[header+13]=data.levels[1];
 // The graph ranges below are the source-side construction groups used by
 // the GPU.  They are redundant only in the empty case; retaining them in
 // the header makes the packet self-describing for future consumers.
 target[header+14]=data.sections[1]+data.bands[1];
 target[header+15]=data.planes[1]+data.incidences[1];
 target[header+16]=item.planeRegions?.length??0;
 target[header+17]=item.structure?.sections.length??0;
 target[header+18]=item.structure?.bands.length??0;
 target[header+19]=item.structure?.vertexRepresentatives.length??0;
 target[header+20]=data.halfedges[0];target[header+21]=data.halfedges[1];target[header+22]=data.edges[0];target[header+23]=data.edges[1];
 target[header+24]=data.boundaries[0];target[header+25]=data.boundaries[1];target[header+26]=data.contacts[0];target[header+27]=data.contacts[1];
 target[header+28]=data.vertexContacts[0];target[header+29]=data.vertexContacts[1];target[header+30]=data.faceComponents[0];target[header+31]=data.faceComponents[1];
 for(let axis=0;axis<3;axis++){target[header+32+axis]=f32bits(item.structure?.axis[axis]??0);target[header+35+axis]=f32bits(item.structure?.origin[axis]??0);}
 const context=appendContextIndices(target,item,item.complex);
 target[header+38]=context.edges[0];target[header+39]=context.edges[1];target[header+40]=context.vertices[0];target[header+41]=context.vertices[1];
 target[header+42]=context.planeEdges[0];target[header+43]=context.planeEdges[1];
}

/** Complete inspection/export slot: header at word 0, records after it,
 * every range relative to word 0. It can be placed at any address unchanged.
 * The renderer uses compileRunTopologySlot instead: complete graph incidence
 * is already resident with the source, not serialized again for run linking. */
export function compileTopologySlot(item:TopologySourceItem):Uint32Array{
 const target:number[]=new Array(TOPOLOGY_HEADER_WORDS).fill(0);
 writeSource(target,0,item);return words(target);
}

/** The run builder's resident program, NOT a second serialization of the
 * source mesh. SurfaceRelations already owns the complete source incidence;
 * the run shader consumes planarVolume, edgeClass and topologyGroup only:
 * plane count and direct edge/vertex context columns. Keep their existing
 * header addresses and slot-relative ranges so no shader decision changes.
 *
 * Full sections, bands, samples, plane loops, halfedges and contact records
 * stay with their source/compiler owner. Packing them here made every
 * arriving source allocate, marshal and upload records no runtime reader
 * used. compileTopologySlot remains the complete inspection/export packet;
 * it is deliberately not called on the rendering path. */
export function compileRunTopologySlot(item:TopologySourceItem):Uint32Array{
 const target:number[]=new Array(TOPOLOGY_HEADER_WORDS).fill(0);
 target[0]=item.domain;target[1]=TYPE_CODE[item.type];
 target[16]=item.planeRegions?.length??0;
 const context=appendContextIndices(target,item,item.complex,true);
 target[37]=context.packed?PACKED_RUN_CONTEXT:0;
 target[38]=context.edges[0];target[39]=context.edges[1];
 target[40]=context.vertices[0];target[41]=context.vertices[1];
 target[42]=context.planeEdges[0];target[43]=context.planeEdges[1];
 appendFaceTable(target,item);
 appendLimbTable(target,item);
 appendPartProgramTable(target,item);
 return words(target);
}

/** The limb-vertex table (TOPOLOGY_LIMB_MAGIC above), per source revision.
 * Every vertex of a LIMB section's samples and of a LIMB band's faces takes
 * that part's chain label; a vertex on no limb part takes its weld
 * representative's word. Undefined when the source has no limb. */
export function compileLimbVertices(item:Pick<TopologySourceItem,'structure'|'complex'>):Uint32Array|undefined{
 const graph=item.structure,complex=item.complex;if(!graph)return undefined;
 const limb=(labels:number|undefined)=>labels!==undefined&&(labels&PART_LIMB)!==0&&(labels>>>PART_CHAIN_SHIFT)!==0;
 if(!graph.sections.some(s=>limb(s.labels))&&!graph.bands.some(b=>limb(b.labels)))return undefined;
 let count=graph.vertexRepresentatives.length;
 if(complex)for(const h of complex.halfedges)count=Math.max(count,h.from+1);
 const table=new Uint32Array(count);
 const word=(labels:number)=>((labels&~0xff)|PART_LIMB)>>>0;
 for(const section of graph.sections){if(!limb(section.labels))continue;const w=word(section.labels!);
  for(const sample of section.samples)for(const v of [sample.a,sample.b])if(v<count&&!table[v])table[v]=w;}
 if(complex)for(const band of graph.bands){if(!limb(band.labels))continue;const w=word(band.labels!);
  for(const face of band.faces)for(let k=0;k<3;k++){const v=complex.halfedges[face*3+k]?.from;if(v!==undefined&&v<count&&!table[v])table[v]=w;}}
 for(let v=0;v<graph.vertexRepresentatives.length;v++){const r=graph.vertexRepresentatives[v]!;if(!table[v]&&r<count)table[v]=table[r]!;}
 return table;
}
function appendLimbTable(target:number[],item:TopologySourceItem):void{
 const table=compileLimbVertices(item);if(!table)return;
 const list=appendList(target,table),w=TOPOLOGY_LIMB_MAGIC_WORD;
 target[w]=TOPOLOGY_LIMB_MAGIC;target[w+1]=list[0];target[w+2]=list[1];
}

export interface SourcePartProgramRecord {
 readonly feature:number;
 readonly role:0|1|2|3;
 readonly d5:boolean;
 readonly direction?:number;
 readonly labels:number;
 /** Original section/band ID; kind is stored separately in the next word. */
 readonly sourceId:number;
 readonly kind:1|2;
}

/**
 * A source-owned context token for the bounded whole-part consumer.  The
 * upper three bits name the retained-entry context family.  Tangent ports
 * are deliberately not baked into this token: their native entry direction
 * is selected from the posed source endpoints by the GPU consumer.  Zero is
 * intentionally the unsupported/legacy value: correspondence still
 * publishes for those parts, but playback keeps its incumbent construction.
 */
export const SOURCE_PART_CONTEXT_BITS=9;
export const SOURCE_PART_CONTEXT_MASK=(1<<SOURCE_PART_CONTEXT_BITS)-1;
export const SOURCE_PART_FEATURE_ORDINAL_BITS=18;
export const SOURCE_PART_FEATURE_ORDINAL_MASK=(1<<SOURCE_PART_FEATURE_ORDINAL_BITS)-1;
export const SOURCE_PART_CONTEXT_CLASS_SHIFT=6;
export const SOURCE_PART_CONTEXT_CLASS_MASK=7;





export type SourcePartContextClass=1|2|3|4|5|6|7;
export type SourcePartContext={readonly incoming:number;readonly outgoing:number;readonly contextClass:SourcePartContextClass;readonly curvature:'convex';readonly inside:'below'|'above'|'left'|'right';readonly token:number};
export interface SourcePartGeometryContext {readonly points:readonly (readonly [number,number,number])[];readonly center:readonly [number,number,number];}
export interface SourcePartContextGeometry {readonly context:SourcePartContext;readonly points:readonly (readonly [number,number,number])[];}
type SourceSectionChart={readonly axis:readonly [number,number,number];readonly u:readonly [number,number,number];readonly v:readonly [number,number,number];};
const vectorBetween=(a:readonly number[],b:readonly number[])=>[b[0]!-a[0]!,b[1]!-a[1]!,b[2]!-a[2]!] as const;
const dot3=(a:readonly number[],b:readonly number[])=>a[0]!*b[0]!+a[1]!*b[1]!+a[2]!*b[2]!;
const cross3=(a:readonly number[],b:readonly number[])=>[a[1]!*b[2]!-a[2]!*b[1]!,a[2]!*b[0]!-a[0]!*b[2]!,a[0]!*b[1]!-a[1]!*b[0]!] as const;
const unit3=(value:readonly number[])=>{const length=Math.hypot(...value);return length>1e-12?[value[0]!/length,value[1]!/length,value[2]!/length] as const:undefined;};
/**
 * Build the intrinsic 2-D chart of one source section.  `axis` is the
 * compiled principal axis; `u` is the canonical longest centre-to-boundary
 * vector in its perpendicular plane and `v` completes the right-handed
 * chart.  No world-up test belongs here: a horizontal tube around world Y
 * is still a section with a usable below/above side.
 */
const sectionChart=(axis:readonly number[],points:readonly (readonly [number,number,number])[],center:readonly number[]):SourceSectionChart|undefined=>{
 const normal=unit3(axis);if(!normal)return;
 let best:readonly [number,number,number]|undefined,bestLength=-1;
 for(const point of points){
  const delta=vectorBetween(center,point),projection=[delta[0]!-dot3(delta,normal)*normal[0]!,delta[1]!-dot3(delta,normal)*normal[1]!,delta[2]!-dot3(delta,normal)*normal[2]!];
  const length=dot3(projection,projection);if(length<1e-12)continue;
  const candidate=unit3(projection)!;
  // Longest first makes the chart independent of tessellation density. Exact
  // ties are resolved by source coordinates, then the dominant component is
  // made positive so a reversed authored loop gets the same chart.
  const greater=length>bestLength+1e-9||(Math.abs(length-bestLength)<=1e-9&&(!best||candidate[0]!>best[0]!+1e-9||Math.abs(candidate[0]!-best[0]!)<=1e-9&&(candidate[1]!>best[1]!+1e-9||Math.abs(candidate[1]!-best[1]!)<=1e-9&&candidate[2]!>best[2]!)));
  if(greater){best=candidate;bestLength=length;}
 }
 if(!best)return;
 const dominant=[0,1,2].sort((a,b)=>Math.abs(best![b]!)-Math.abs(best![a]!)||a-b)[0]!;
 const u=best[dominant]!<0?[-best[0]!, -best[1]!, -best[2]!] as const:best;
 const v=unit3(cross3(normal,u));if(!v)return;
 return {axis:normal,u,v};
};
/** Direction code in the same E,SE,... order consumed by the resident role
 * table. It is measured in the source section chart, never in world XY. */
const sourceAxisDirection=(chart:SourceSectionChart,tangent:readonly number[]):number=>{
 const angle=Math.atan2(dot3(tangent,chart.v),dot3(tangent,chart.u));
 return Math.round((angle/(2*Math.PI)+1)*8)&7;
};
const sourceSide=(chart:SourceSectionChart,a:readonly number[],b:readonly number[],center:readonly number[]):'above'|'below'|'left'|'right'|'enclosed'=>{
 const midpoint=[(a[0]!+b[0]!)/2,(a[1]!+b[1]!)/2,(a[2]!+b[2]!)/2];
 const tangent=vectorBetween(a,b),toCenter=vectorBetween(midpoint,center);
 const across=dot3(toCenter,chart.v),along=dot3(toCenter,chart.u),epsilon=1e-8*Math.max(1,Math.hypot(...tangent));
 if(Math.abs(across)>epsilon&&Math.abs(across)>=Math.abs(along))return across>=0?'below':'above';
 // LIVE rule (incumbent, unchanged): left/right relative to travel. It feeds
 // the packet's section classification and every live flank token.
 const signed=dot3(cross3(tangent,toCenter),chart.axis);if(Math.abs(signed)<=epsilon)return 'enclosed';
 return signed>=0?'left':'right';
};








const sourceSidePage=(chart:SourceSectionChart,a:readonly number[],b:readonly number[],center:readonly number[]):'above'|'below'|'left'|'right'|'enclosed'=>{
 const midpoint=[(a[0]!+b[0]!)/2,(a[1]!+b[1]!)/2,(a[2]!+b[2]!)/2];
 const tangent=vectorBetween(a,b),toCenter=vectorBetween(midpoint,center);
 const across=dot3(toCenter,chart.v),along=dot3(toCenter,chart.u),epsilon=1e-8*Math.max(1,Math.hypot(...tangent));
 if(Math.abs(across)>epsilon&&Math.abs(across)>=Math.abs(along))return across>=0?'below':'above';
 if(Math.abs(along)<=epsilon)return 'enclosed';
 return along>0?'right':'left';
};
/** Which side rule a classifier uses: 'live' (packet/incumbent) or 'page'
 * (parent producer admission). */
export type SourcePartSideRule='live'|'page';
const sourceCurvature=(axis:readonly number[],points:readonly (readonly [number,number,number])[]):'convex'|'unknown'=>{
 if(points.length<3)return 'unknown';let sign=0,turns=0;
 for(let i=0;i<points.length;i++){
  const a=points[(i+points.length-1)%points.length]!,b=points[i]!,c=points[(i+1)%points.length]!,u=vectorBetween(a,b),v=vectorBetween(b,c);
  const cross=[u[1]! * v[2]!-u[2]! * v[1]!,u[2]! * v[0]!-u[0]! * v[2]!,u[0]! * v[1]!-u[1]! * v[0]!],value=cross[0]! * axis[0]!+cross[1]! * axis[1]!+cross[2]! * axis[2]!;
  if(Math.abs(value)<1e-7)continue;const next=value<0?-1:1;if(sign&&sign!==next)return 'unknown';sign=next;turns++;
 }
 return turns>=3?'convex':'unknown';
};
const sourcePartContextOne=(axis:readonly number[],role:0|1|2|3,closed:boolean,geometry:SourcePartGeometryContext,side:SourcePartSideRule='live'):SourcePartContext|undefined=>{
 if(!closed||geometry.points.length<3)return undefined;
 const curvature=sourceCurvature(axis,geometry.points);if(curvature!=='convex')return undefined;
 const chart=sectionChart(axis,geometry.points,geometry.center);if(!chart)return undefined;
 const first=geometry.points[0]!,second=geometry.points[1]!,last=geometry.points[geometry.points.length-1]!,before=geometry.points[geometry.points.length-2]!;
 // Native ports are outward-facing: incoming points away from the selected
 // source anchor, while outgoing follows the final segment toward it. This
 // is the same convention as sourcePartContextKey in gpu-articulation.ts.
 const incoming=sourceAxisDirection(chart,vectorBetween(second,first)),outgoing=sourceAxisDirection(chart,vectorBetween(before,last));
 // PartSection.inside is a nested-section relation, not a paper-side label;
 // the retained context side is derived from the actual section chart.
 const inside=(side==='page'?sourceSidePage:sourceSide)(chart,first,second,geometry.center);if(inside==='enclosed')return undefined;
 const contextClass=passageContextClass(role,inside);
 if(contextClass===undefined)return undefined;
 // Incoming/outgoing are retained in the immutable source analysis for table
 // admission and diagnostics. They are not object-world directions in the
 // resident token; playback reprojects the source-owned endpoints per pose.
 const token=(contextClass<<SOURCE_PART_CONTEXT_CLASS_SHIFT)&SOURCE_PART_CONTEXT_MASK;
 return {incoming,outgoing,contextClass,curvature,inside,token};
};
/** Return every legal source contour orientation in deterministic order. The
 * selected points are part of the result so a caller cannot admit one anchor
 * while emitting a different tangent pair. */
export function sourcePartContextCandidates(axis:readonly number[],role:0|1|2|3,closed:boolean,geometry?:SourcePartGeometryContext,side:SourcePartSideRule='live'):readonly SourcePartContextGeometry[]{
 if(!closed||!geometry||geometry.points.length<3)return [];
 const result:SourcePartContextGeometry[]=[];
 for(const winding of [geometry.points,[...geometry.points].reverse()])for(let offset=0;offset<winding.length;offset++){
  const points=[...winding.slice(offset),...winding.slice(0,offset)];
  const context=sourcePartContextOne(axis,role,closed,{points,center:geometry.center},side);if(context)result.push({context,points});
 }

 // already had a below (class 2) rotation keeps it first, so the packet token
 // (`sectionContext` takes candidate 0) and the producer's contour order are
 // unchanged for every previously admitted section and stay independent of
 // which side rotation the exporter's first sample happens to be. Stable.
 return result.map((item,index)=>({item,index})).sort((a,b)=>(a.item.context.contextClass>=6?1:0)-(b.item.context.contextClass>=6?1:0)||a.index-b.index).map(value=>value.item);
}




export interface SourcePartPassage extends SourcePartContextGeometry {readonly from:number;readonly to:number;}
/** Labelled end anchors for a passage that must join two different source
 * incidences (a fork band's branch sections). Label 0 is never an anchor. */
export interface SourcePartPassageAnchor {readonly index:number;readonly label:number;}
/** The one role x measured-inside -> context-class rule, shared by closed
 * contours and passages. Mirrors compiled-role-grammar.ts `contextInside`. A
 * cap is only ever below; a return only above; a flank only beside; a
 * shoulder below (2) or beside (6 left / 7 right). Nothing else maps. */
function passageContextClass(role:0|1|2|3,inside:'above'|'below'|'left'|'right'|'enclosed'):SourcePartContextClass|undefined{
 if(role===0)return inside==='below'?1:undefined;
 if(role===1)return inside==='below'?2:inside==='left'?6:inside==='right'?7:undefined;
 if(role===2)return inside==='left'?3:inside==='right'?4:undefined;
 return inside==='above'?5:undefined;
}






















export function sourcePartPassageCandidates(axis:readonly number[],role:0|1|2|3,geometry?:SourcePartGeometryContext,anchors?:readonly SourcePartPassageAnchor[]):readonly SourcePartPassage[]{
 if(!geometry||geometry.points.length<3)return [];
 const points=geometry.points,n=points.length;
 if(sourceCurvature(axis,points)!=='convex')return [];
 const chart=sectionChart(axis,points,geometry.center);if(!chart)return [];
 const scale=Math.max(1e-12,...points.map(point=>Math.hypot(...vectorBetween(geometry.center,point))));
 const degenerate=(i:number)=>Math.hypot(...vectorBetween(points[i]!,points[(i+1)%n]!))<=1e-9*scale;
 // The positive winding is the one whose turn is positive about the chart axis.
 let area=0;for(let i=0;i<n;i++)area+=dot3(cross3(vectorBetween(geometry.center,points[i]!),vectorBetween(geometry.center,points[(i+1)%n]!)),chart.axis);
 if(Math.abs(area)<=1e-12*scale*scale)return [];
 const result:{passage:SourcePartPassage;span:number;winding:number;rank:number}[]=[];
 for(const winding of [1,-1] as const){
  // order[k] = contour index of the k-th point in this winding.
  const order=Array.from({length:n},(_,k)=>winding===1?k:(n-k)%n);
  const at=(k:number)=>order[((k%n)+n)%n]!;
  const segmentDirection=(k:number)=>degenerate(winding===1?at(k):at(k+1))?-1:sourceAxisDirection(chart,vectorBetween(points[at(k)]!,points[at(k+1)]!));
  const lengthOf=(k:number)=>Math.hypot(...vectorBetween(points[at(k)]!,points[at(k+1)]!));
  // Anchor ranks (positions k in this winding) and their passage pairs.
  const pairs:{start:number;end:number;rank:number}[]=[];
  if(anchors){
   const rankOf=new Map(order.map((index,k)=>[index,k]));
   const ranked=anchors.filter(anchor=>anchor.label!==0&&rankOf.has(anchor.index)).map(anchor=>({k:rankOf.get(anchor.index)!,label:anchor.label})).sort((a,b)=>a.k-b.k);
   if(ranked.length<2)continue;

   // form one incidence run (dense anchors: every incident ring vertex on a
   // junction hull). A passage runs from the first anchor of one run to the
   // first anchor of the next run, so it covers one whole incidence and joins
   // it to the next. Sparse anchors are runs of length one (unchanged).
   const starts:number[]=[];for(let i=0;i<ranked.length;i++)if(ranked[i]!.label!==ranked[(i+ranked.length-1)%ranked.length]!.label)starts.push(i);
   if(starts.length<2)continue;
   const lowest=starts.reduce((best,item,index)=>ranked[item]!.label<ranked[starts[best]!]!.label?index:best,0);
   for(let i=0;i<starts.length;i++){
    const a=ranked[starts[i]!]!,b=ranked[starts[(i+1)%starts.length]!]!;
    const end=b.k>a.k?b.k:b.k+n;if(end-a.k<2)continue;
    pairs.push({start:a.k,end,rank:(i-lowest+starts.length)%starts.length});
   }
  }else{
   const directions=Array.from({length:n},(_,k)=>segmentDirection(k));
   // Degenerate segments join the run before them.
   for(let k=0;k<n;k++)if(directions[k]===-1)directions[k]=directions[(k+n-1)%n]!;
   if(directions.some(value=>value===-1))continue;
   const boundaries:number[]=[];for(let k=0;k<n;k++)if(directions[k]!==directions[(k+n-1)%n])boundaries.push(k);
   const runs=boundaries.length;
   // A convex loop's quantized tangent is monotone: each code is one run.
   if(runs<2||runs>8||new Set(boundaries.map(k=>directions[k])).size!==runs)continue;
   for(let i=0;i<runs;i++)for(let span=1;span<runs;span++){
    const start=boundaries[i]!,last=boundaries[(i+span)%runs]!,end=last>start?last:last+n;
    pairs.push({start,end,rank:directions[start]!});
   }
  }
  for(const pair of pairs){
   const passagePoints=Array.from({length:pair.end-pair.start+1},(_,offset)=>points[at(pair.start+offset)]!);
   if(passagePoints.length<3)continue;
   const first=passagePoints[0]!,second=passagePoints[1]!,before=passagePoints[passagePoints.length-2]!,last=passagePoints[passagePoints.length-1]!;
   const incoming=sourceAxisDirection(chart,vectorBetween(second,first)),outgoing=sourceAxisDirection(chart,vectorBetween(before,last));
   let total=0;for(let k=pair.start;k<pair.end;k++)total+=lengthOf(k);if(!(total>0))continue;
   let middle=pair.start,walked=0;for(let k=pair.start;k<pair.end;k++){const length=lengthOf(k);if(walked+length>=total/2){middle=k;break;}walked+=length;}

   const inside=sourceSidePage(chart,points[at(middle)]!,points[at(middle+1)]!,geometry.center);if(inside==='enclosed')continue;
   const contextClass=passageContextClass(role,inside);if(contextClass===undefined)continue;
   const token=(contextClass<<SOURCE_PART_CONTEXT_CLASS_SHIFT)&SOURCE_PART_CONTEXT_MASK;
   result.push({passage:{context:{incoming,outgoing,contextClass,curvature:'convex',inside,token},points:passagePoints,from:at(pair.start),to:at(pair.end)},
    span:pair.end-pair.start,winding:Math.sign(area)*winding>0?0:1,rank:pair.rank});
  }
 }
 return result.sort((a,b)=>b.span-a.span||a.winding-b.winding||a.rank-b.rank).map(item=>item.passage);
}
/** Source-space context only. A caller must provide actual contour points;
 * axis/role alone is intentionally insufficient and returns no eligibility. */
export function sourcePartContext(axis:readonly number[],role:0|1|2|3,closed:boolean,geometry?:SourcePartGeometryContext):SourcePartContext|undefined{
 return sourcePartContextCandidates(axis,role,closed,geometry)[0]?.context;
}

const partRole=(labels:number,closed:boolean):0|1|2|3=>{
 if((labels&PART_ROOT)!==0)return 3;
 if((labels&PART_CAP)!==0)return 0;
 if((labels&PART_LIMB)!==0)return 2;
 if((labels&PART_FORK)!==0)return 1;
 return closed?1:2;
};
const sectionContext=(graph:MeshPartGraph,section:PartSection,role:0|1|2|3,positions:Float32Array|undefined):SourcePartContext|undefined=>{
 if(!positions||section.samples.length<3)return undefined;
 const points=section.samples.map(sample=>partPoint({positions},sample));
 return sourcePartContextCandidates(graph.axis,role,section.closed,{points,center:section.center})[0]?.context;
};

/** Compile one source-owned program per section/band plus a one-word membership
 * table indexed by original halfedge address. The source feature is the
 * section/band identity, not a projected fragment; a split run therefore keeps
 * the same program while visibility merely clips its original units. Every source
 * feature stays in the table at full u32 width in the original-part word, with
 * kind separate. Compact run tokens use the collision-free program ordinal;
 * the original source ID is never narrowed into that consumer key. */
export function compileSourcePartPrograms(item:Pick<TopologySourceItem,'structure'|'complex'|'sourcePositions'>):Uint32Array|undefined{
 const graph=item.structure,complex=item.complex;if(!graph||!complex?.halfedges.length)return undefined;
 const byFace=new Map<number,number>(),programs:SourcePartProgramRecord[]=[];
 const byPart=new Map<string,number>();
 const add=(id:number,labels:number,closed:boolean,kind:1|2,section?:PartSection):number=>{
  const key=`${kind}:${id}`,old=byPart.get(key);if(old!==undefined)return old;
  const index=programs.length;
  // The compact consumer key is the packet-local program ordinal, not the
  // section/band ID. This keeps section 0 and band 0 distinct while the full
  // sourceId/kind words preserve source identity for non-compact consumers.
  const role=partRole(labels,closed);
  const context=kind===1&&section?sectionContext(graph,section,role,item.sourcePositions):undefined;
  // The compact token keeps the full context beside an 18-bit packet-local
  // ordinal.  The full source ID and kind remain separate words, so this
  // packing never aliases section/band identity or narrows source IDs.
  const ordinal=index+1;
  const feature=ordinal<=SOURCE_PART_FEATURE_ORDINAL_MASK
   ? (((context?.token??0)<<SOURCE_PART_FEATURE_ORDINAL_BITS)|ordinal)>>>0:0;
  const record={feature,role,d5:(labels&(PART_LIMB|PART_FORK))!==0,direction:context?.token,labels,sourceId:id,kind};
  programs.push(record);byPart.set(key,index);return index;
 };
 // A source graph can contain overlapping sections/bands. First ownership
 // still wins, but a shadowed part has no consumer and must not allocate a
 // resident program. Numabe's eager allocation serialized over a million
 // unreferenced records. Admit lazily from actual original source faces;
 // preserve the selected parts' order and full semantic identities.
 const unclaimed=new Set<number>();for(const halfedge of complex.halfedges)unclaimed.add(halfedge.face);
 for(const section of graph.sections){let index:number|undefined;for(const face of section.faces)if(unclaimed.delete(face)){index??=add(section.id,section.labels??0,section.closed,1,section);byFace.set(face,index);}}
 for(const band of graph.bands){let index:number|undefined;for(const face of band.faces)if(unclaimed.delete(face)){index??=add(band.id,band.labels??0,false,2);byFace.set(face,index);}}
 const members=new Uint32Array(complex.halfedges.length);members.fill(0xffffffff);
 let any=false;
 for(const halfedge of complex.halfedges){const part=byFace.get(halfedge.face);if(part===undefined||halfedge.id>=members.length)continue;members[halfedge.id]=part;any=true;}
 if(!any)return undefined;
 // Source-declared anchors (crawl brief item C, packet half): per program,
 // the halfedge id of the admitted member whose origin vertex is extreme in
 // rest space along -x,+x,-y,+y,-z,+z. Strict comparison wins; exact ties
 // take the lowest halfedge id. Members without three finite rest coords
 // (missing positions, out-of-range vertex, NaN) contribute nothing, so a
 // part with no member positions keeps 0xffffffff for all six.
 const positions=item.sourcePositions;
 const anchors=programs.map(()=>({ids:Uint32Array.from([0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff]),values:new Float64Array(6),have:new Uint8Array(6)}));
 if(positions)for(const halfedge of complex.halfedges){
  const part=byFace.get(halfedge.face);if(part===undefined)continue;
  const v=halfedge.from*3;if(halfedge.from<0||v+2>=positions.length)continue;
  const coords=[positions[v]!,positions[v+1]!,positions[v+2]!];
  if(coords.some(c=>!Number.isFinite(c)))continue;
  const slot=anchors[part]!;
  for(let axis=0;axis<3;axis++){
   const c=coords[axis]!,lo=axis*2,hi=axis*2+1;
   if(slot.have[lo]===0||c<slot.values[lo]!||(c===slot.values[lo]!&&halfedge.id<slot.ids[lo]!)){slot.have[lo]=1;slot.values[lo]=c;slot.ids[lo]=halfedge.id>>>0;}
   if(slot.have[hi]===0||c>slot.values[hi]!||(c===slot.values[hi]!&&halfedge.id<slot.ids[hi]!)){slot.have[hi]=1;slot.values[hi]=c;slot.ids[hi]=halfedge.id>>>0;}
  }
 }
 const programOffset=5,memberOffset=programOffset+programs.length*TOPOLOGY_PART_PROGRAM_WORDS;
 const out=new Uint32Array(memberOffset+members.length);out[0]=TOPOLOGY_PART_PACKET_MAGIC;out[1]=programOffset;out[2]=programs.length;out[3]=memberOffset;out[4]=members.length;
 for(const [index,source]of programs.entries()){
  const at=programOffset+index*TOPOLOGY_PART_PROGRAM_WORDS;out[at]=source.feature;
  out[at+1]=source.role|(source.d5?1<<8:0);
  if(source.direction!==undefined)out[at+1]|=1<<9|((source.direction&SOURCE_PART_CONTEXT_MASK)<<10);
  out[at+2]=source.direction??0;out[at+3]=source.labels>>>0;out[at+4]=source.sourceId>>>0;out[at+5]=source.kind;
  out.set(anchors[index]!.ids,at+6);
 }
 out.set(members,memberOffset);return out;
}

/** Adapter from T6 anchor halfedges to 137 anchor vertices (crawl brief item
 * C): six origin vertex ids per part program in packet order, for
 * `SurfaceRelations.registerPartAnchors`. Each anchor halfedge id resolves to
 * that halfedge's `from` vertex; 0xffffffff stays 0xffffffff, and an id with
 * no matching halfedge becomes 0xffffffff. Returns undefined unless the
 * packet carries TOPOLOGY_PART_PACKET_MAGIC with a well-formed header
 * (program offset/count safe integers, offset>=5, count>=1, and the full
 * program records inside the packet at TOPOLOGY_PART_PROGRAM_WORDS stride),
 * so an old-magic or truncated packet is rejected, never misread. The
 * producer assigns unique halfedge ids; if a hand-built array repeated one,
 * the first entry wins. */
export function sourcePartAnchorVertices(packet:Uint32Array,halfedges:readonly {readonly id:number;readonly from:number}[]|Uint32Array):Uint32Array|undefined{
 if(packet[0]!==TOPOLOGY_PART_PACKET_MAGIC)return undefined;
 const offset=packet[1]!,count=packet[2]!;
 if(!Number.isSafeInteger(offset)||!Number.isSafeInteger(count)||offset<5||count<1||offset+count*TOPOLOGY_PART_PROGRAM_WORDS>packet.length)return undefined;
 // Native indexed halfedge ids are original index addresses. Baked/numeric
 // sources already retain those indices; resolve only six anchors per part,
 // without reconstructing or mapping the complete halfedge graph on a phone.
 const indexed=halfedges instanceof Uint32Array;
 const origin=new Map<number,number>();
 if(!indexed)for(const halfedge of halfedges)if(!origin.has(halfedge.id))origin.set(halfedge.id,halfedge.from);
 const out=new Uint32Array(count*6);
 for(let index=0;index<count;index++){
  const at=offset+index*TOPOLOGY_PART_PROGRAM_WORDS;
  for(let k=0;k<6;k++){
   const id=packet[at+6+k]!;
   if(id===0xffffffff){out[index*6+k]=0xffffffff;continue;}
   const from=indexed?(id<halfedges.length?halfedges[id]:undefined):origin.get(id);
   out[index*6+k]=from===undefined?0xffffffff:from>>>0;
  }
 }
 return out;
}

/** View the already-compiled packet, not the later unrelated topology tail.
 * Worker metadata and semantic assets share these exact source decisions. */
export function sourcePartProgramPacket(topology:Uint32Array):Uint32Array|undefined{
 const w=TOPOLOGY_PART_MAGIC_WORD;if(topology[w]!==TOPOLOGY_PART_MAGIC)return;
 const base=topology[w+1]!-5,count=topology[w+2]!,members=topology[w+3]!,length=topology[w+4]!;
 if(base<0||base+5>topology.length||members<base+5||members+length>topology.length)return;
 if(topology[base]!==TOPOLOGY_PART_PACKET_MAGIC||topology[base+1]!==5||topology[base+2]!==count||
  topology[base+3]!==members-base||topology[base+4]!==length||members!==base+5+count*TOPOLOGY_PART_PROGRAM_WORDS)return;
 return topology.subarray(base,members+length);
}

function appendPartProgramTable(target:number[],item:TopologySourceItem):void{
 const table=compileSourcePartPrograms(item);if(!table)return;
 const list=appendList(target,table),w=TOPOLOGY_PART_MAGIC_WORD,programOffset=table[1]!,programCount=table[2]!,memberOffset=table[3]!,memberCount=table[4]!;
 // Header words 11..15 are the five compact-slot words. Word 16 is the
 // resident plane count consumed by planarVolume and must remain untouched.
 target[w]=TOPOLOGY_PART_MAGIC;target[w+1]=list[0]+programOffset;target[w+2]=programCount;target[w+3]=list[0]+memberOffset;target[w+4]=memberCount;
}

/** The face owner's edge table and face records (revision 4 above). An edge
 * bordering regions of several faces takes the least face index and the first
 * role in the order outline, opening, member. */
export function compileFaceEdges(item:TopologySourceItem):Uint32Array|undefined{
 const complex=item.complex,faces=item.faces,regions=item.planeRegions??[];if(!complex||!faces?.length)return undefined;
 const table=new Uint32Array(complex.halfedges.length);

 // A drawn halfedge owns its edge (the least face wins; outline before
 // opening); a hidden one marks the edge FACE_HIDDEN only where no face draws
 // it. Every other edge (a slanted gable side, a roof edge) keeps word 0.
 const edgeOf=(halfedge:number)=>{const record=complex.halfedges[halfedge];if(!record)return -1;const edge=complex.edges[record.edge]!.id;return edge<table.length?edge:-1;};
 for(const face of faces){const openings=new Set((regions[face.main]?.openings??[]).flat());
  for(const h of face.drawn){const edge=edgeOf(h);if(edge<0)continue;const role=openings.has(h)?FACE_ROLE_OPENING:FACE_ROLE_OUTLINE;
   const old=table[edge]!;if(old&&(old&FACE_HIDDEN)===0){const oldFace=(old&255)-1,oldRole=(old>>>8)&3;if(oldFace<face.id||(oldFace===face.id&&oldRole<=role))continue;}
   table[edge]=((face.id+1)|(role<<8))>>>0;}}
 for(const face of faces)for(const h of face.hidden){const edge=edgeOf(h);if(edge<0||table[edge])continue;table[edge]=((face.id+1)|(FACE_ROLE_MEMBER<<8)|FACE_HIDDEN)>>>0;}
 return table;
}
function appendFaceTable(target:number[],item:TopologySourceItem):void{
 const table=compileFaceEdges(item),faces=item.faces??[];if(!table)return;
 const edges=appendList(target,table);
 const recordsAt=target.length;target.length+=faces.length*TOPOLOGY_FACE_WORDS;
 for(const [index,face]of faces.entries()){
  const corners=appendList(target,face.corners),at=recordsAt+index*TOPOLOGY_FACE_WORDS;
  target[at]=item.planeRegions?.[face.main]?.id??0;target[at+1]=face.regions.length;target[at+2]=corners[0];target[at+3]=corners[1];
  target[at+4]=face.openings;target[at+5]=face.members.length;target[at+6]=f32bits(face.share);target[at+7]=face.frames.length;
 }
 const w=TOPOLOGY_FACE_MAGIC_WORD;
 target[w]=TOPOLOGY_FACE_MAGIC;target[w+1]=edges[0];target[w+2]=edges[1];target[w+3]=faces.length;target[w+4]=recordsAt;target[w+5]=f32bits(faces[0]!.width);
}

/** Immutable, domain-neutral runtime input. Compile with the source revision,
 * not when a camera first asks the GPU arena to draw it. The resident table
 * binds domain in header word0 without walking/repacking source topology. */
export interface CpuCompiledSourceRunProgram {
 readonly topology:Uint32Array;
 /** Sorted [source vertex, critical/fork flags] records. */
 readonly criticalVertices:Uint32Array;
}
export interface GpuCompiledSourceRunProgram {
 readonly gpuTopology:{readonly binding:GPUBufferBinding;readonly words:number};
 readonly criticalVertices:Uint32Array;
}
export type CompiledSourceRunProgram=CpuCompiledSourceRunProgram|GpuCompiledSourceRunProgram;
export function compileSourceRunProgram(item:CompiledConstructionGraph&{complex?:MeshSourceComplex;sourcePositions?:Float32Array}):CpuCompiledSourceRunProgram{
 const vertices=new Map<number,number>();
 for(const band of item.structure?.bands??[])for(const vertex of band.criticalVertices)vertices.set(vertex,(vertices.get(vertex)??0)|1|((band.lower.length>1||band.upper.length>1)?2:0));
 const sorted=[...vertices].sort((a,b)=>a[0]-b[0]),criticalVertices=new Uint32Array(sorted.length*2);
 sorted.forEach(([vertex,flags],index)=>{criticalVertices[index*2]=vertex;criticalVertices[index*2+1]=flags;});
 return {topology:compileRunTopologySlot({...item,domain:0}),criticalVertices};
}

/** Every source's slot, in domain order, one after another (CPU view of the
 * resident topology: tests and inspection). `sources[i].headerWord` is the
 * slot base; ranges in a slot are relative to it. */
export function packConstructionTopology(classes:readonly TopologySourceItem[]):PackedConstructionTopology{
 const ordered=[...classes].sort((a,b)=>a.domain-b.domain),slots=ordered.map(compileTopologySlot);
 const result=new Uint32Array(slots.reduce((n,slot)=>n+slot.length,0));
 const sources:ResidentTopologySource[]=[];
 let base=0;
 for(const [index,item]of ordered.entries()){
  result.set(slots[index]!,base);
  sources.push({domain:item.domain,type:item.type,headerWord:base,structure:item.structure,planeRegions:item.planeRegions??[],complex:item.complex});
  base+=slots[index]!.length;
 }
 // A byte-stable fingerprint makes plane-region changes invalidate residency
 // without rebuilding a source graph on every frame. This is intentionally
 // a packet identity, not an artistic score.
 let hash=2166136261>>>0;for(let i=0;i<result.length;i++){hash^=result[i]!;hash=Math.imul(hash,16777619)>>>0;}
 return {words:result,sources,fingerprint:hash.toString(16).padStart(8,'0')};
}

/** The pre-2026-11-12 whole packet: all headers first, ranges absolute in the
 * packet, which the run shader read as topologyBase + range. Kept only so the
 * test can prove that a slot read at its own base shows the shader the same
 * words the whole packet did. */
export function packConstructionTopologyWhole(classes:readonly TopologySourceItem[]):Uint32Array{
 const ordered=[...classes].sort((a,b)=>a.domain-b.domain);
 const target:number[]=new Array(ordered.length*TOPOLOGY_HEADER_WORDS).fill(0);
 for(const [index,item]of ordered.entries())writeSource(target,index*TOPOLOGY_HEADER_WORDS,item);
 return words(target);
}

export function topologyTypeCode(type:CompiledConstructionGraph['type']):number{return TYPE_CODE[type];}



































export const EDGE_LEVEL_UNIT=8;
/** Bit 15 of a code: the edge is class 1 DETAIL (a short authored feature edge
 * of a plane: window, panel). Bits 0-14: the chain ratio in EDGE_LEVEL_UNIT.
 * A detail edge is limited as a crease AND as a silhouette (its silhouettes
 * are the recess sides of the same detail); a facet crease (class 2) only as a
 * crease, so a far rock or crown keeps its whole tessellated outline (D9). */
export const EDGE_LEVEL_DETAIL=0x8000;
/** Code of a PRINCIPAL crease that is a member of a compiled L1 chain
 * (compilePrincipalChains): produced as itself at 近景 (L0); at 中景 (L1) its
 * chain's ONE construction edge is produced instead. levelDrops never drops
 * it (the ratio bits are all set), and no facet ratio takes this value. */
export const EDGE_LEVEL_CHAINED=0x7fff;


export const FACET_CREASE_LEAST_ROWS=1;
const CHAIN_TURN_COS=0.70710678;
/** A class 1 crease chain at least this share of its source's rest extent is
 * a principal edge of the form (roof, lid, corner) and keeps code 0. */
export const DETAIL_FRACTION=.25;
export function compileEdgeLevels(topology:Uint32Array,complex:MeshSourceComplex,positions:ArrayLike<number>):Uint32Array{
 const classesAt=topology[42]!,count=topology[43]!,packed=topology[37]===PACKED_RUN_CONTEXT,H=complex.halfedges.length;
 const klass=(h:number):number=>{if(h>=count)return 0;const v=topology[classesAt+h]!;return packed?v&3:v;};
 const weld=complex.weld,vertex=(v:number):number=>weld?weld[v]!:v;
 const lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];
 for(let i=0;i+2<positions.length;i+=3)for(let k=0;k<3;k++){const p=positions[i+k]!;if(p<lo[k]!)lo[k]=p;if(p>hi[k]!)hi[k]=p;}
 // The box's MIDDLE side: a long low building's corners and a pole's ends are
 // its own edges, not detail of its longest side.
 const extent=[hi[0]!-lo[0]!,hi[1]!-lo[1]!,hi[2]!-lo[2]!].sort((a,b)=>a-b)[1]!;
 const out=new Uint32Array(Math.ceil(H/2));let any=false;
 for(const level of [1,2]){
  // One record per WELDED crease of this class (index edges of one welded edge share it).
  const groupOf=new Map<number,number>(),ends:number[]=[],lengths:number[]=[],members:number[][]=[];
  for(const edge of complex.edges){
   if(klass(edge.id)!==level)continue;
   const key=edge.welded?.[0]??edge.id;let g=groupOf.get(key);
   if(g===undefined){
    g=lengths.length;groupOf.set(key,g);const a=vertex(edge.vertices[0]),b=vertex(edge.vertices[1]);ends.push(a,b);
    lengths.push(Math.hypot(positions[b*3]!-positions[a*3]!,positions[b*3+1]!-positions[a*3+1]!,positions[b*3+2]!-positions[a*3+2]!));members.push([]);
   }
   members[g]!.push(edge.id);
  }
  if(!lengths.length)continue;
  const parent=Int32Array.from(lengths,(_,i)=>i);
  const root=(i:number):number=>{while(parent[i]!==i){parent[i]=parent[parent[i]!]!;i=parent[i]!;}return i;};
  const at=new Map<number,number[]>();
  for(let g=0;g<lengths.length;g++)for(const v of [ends[g*2]!,ends[g*2+1]!]){const list=at.get(v);if(list)list.push(g);else at.set(v,[g]);}
  const away=(g:number,v:number):[number,number,number]=>{const o=ends[g*2]===v?ends[g*2+1]!:ends[g*2]!;return [positions[o*3]!-positions[v*3]!,positions[o*3+1]!-positions[v*3+1]!,positions[o*3+2]!-positions[v*3+2]!];};
  for(const [v,list]of at)for(let i=0;i<list.length;i++)for(let j=i+1;j<list.length;j++){
   const x=away(list[i]!,v),y=away(list[j]!,v),lx=Math.hypot(...x),ly=Math.hypot(...y);
   if(lx>0&&ly>0&&-(x[0]*y[0]+x[1]*y[1]+x[2]*y[2])>=CHAIN_TURN_COS*lx*ly){const a=root(list[i]!),b=root(list[j]!);if(a!==b)parent[a]=b;}
  }
  const chain=new Float64Array(lengths.length);for(let g=0;g<lengths.length;g++)chain[root(g)]+=lengths[g]!;
  for(let g=0;g<lengths.length;g++){
   const own=lengths[g]!,span=chain[root(g)]!;if(!(own>0))continue; // a zero-length connector keeps its incidence (no level)
   if(level===1&&!(span<DETAIL_FRACTION*extent))continue; // a principal edge of the form: every tier
   // EDGE_LEVEL_CHAINED (all ratio bits) is reserved for L1 chain members.
   const code=(Math.min(EDGE_LEVEL_CHAINED-1,Math.max(EDGE_LEVEL_UNIT,Math.ceil(EDGE_LEVEL_UNIT*span/own)))|(level===1?EDGE_LEVEL_DETAIL:0))>>>0;any=true;
   for(const h of members[g]!)if(h<H)out[h>>>1]=(out[h>>>1]!|(code<<((h&1)*16)))>>>0;
  }
 }
 return any?out:new Uint32Array(0);
}





















export const OUTLINE_HULL_MAGIC=0x48554c4c;
export const OUTLINE_HULL_WORDS=5;
/** Least share of its hull's volume a closed source fills to take the hull as
 * its L2 outline (compileOutlineHull adequacy). */
export const OUTLINE_HULL_FILL=.7;
/** The 26 directions of a cube's faces, edges and corners. */
export const OUTLINE_DIRECTIONS:readonly (readonly [number,number,number])[]=(()=>{
 const out:[number,number,number][]=[];
 for(let x=-1;x<=1;x++)for(let y=-1;y<=1;y++)for(let z=-1;z<=1;z++)if(x||y||z){const l=Math.hypot(x,y,z);out.push([x/l,y/l,z/l]);}
 return out;
})();
export function compileOutlineHull(complex:MeshSourceComplex,positions:ArrayLike<number>):Uint32Array|undefined{
 // Vertices that bound a face, with one face each (the draw range owner).
 const faceAt=new Map<number,number>();
 for(const h of complex.halfedges)if(!faceAt.has(h.from))faceAt.set(h.from,h.face);
 if(faceAt.size<4)return undefined;
 const P=(v:number):[number,number,number]=>[positions[v*3]!,positions[v*3+1]!,positions[v*3+2]!];
 let scale=0;
 const lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];
 for(const v of faceAt.keys()){const p=P(v);for(let k=0;k<3;k++){lo[k]=Math.min(lo[k]!,p[k]);hi[k]=Math.max(hi[k]!,p[k]);}}
 scale=Math.hypot(hi[0]!-lo[0]!,hi[1]!-lo[1]!,hi[2]!-lo[2]!);if(!(scale>0))return undefined;
 const eps=scale*1e-6;
 // Support vertices: per direction the most extreme vertex; ties (within eps)
 // keep the least vertex id, the same id the vertex port names.
 const vertices=[...faceAt.keys()].sort((a,b)=>a-b),support:number[]=[];
 for(const [dx,dy,dz]of OUTLINE_DIRECTIONS){
  let best=-1,value=-Infinity;
  for(const v of vertices){const p=P(v),s=p[0]*dx+p[1]*dy+p[2]*dz;if(s>value+eps){value=s;best=v;}}
  if(best<0)continue;
  // Coincident copies of one corner are one anchor.
  const q=P(best);if(!support.some(u=>{const r=P(u);return Math.hypot(r[0]-q[0],r[1]-q[1],r[2]-q[2])<=eps;}))support.push(best);
 }
 support.sort((a,b)=>a-b);
 const faces=convexHull(support.map(P),eps);if(!faces)return undefined;

 // source (boundary edges: a sheet, an opening) or a CONCAVE one (its closed
 // volume under OUTLINE_HULL_FILL of the hull's: an L or U block, a court, an
 // arch, a figure) has no L2 construction; it stays at the principal level.
 if(complex.boundaries.length)return undefined;
 const volume=(tris:Iterable<readonly [number,number,number]>,at:(i:number)=>readonly number[])=>{let v=0;for(const [i,j,k]of tris){const a=at(i),b=at(j),c=at(k);
  v+=a[0]!*(b[1]!*c[2]!-b[2]!*c[1]!)-a[1]!*(b[0]!*c[2]!-b[2]!*c[0]!)+a[2]!*(b[0]!*c[1]!-b[1]!*c[0]!);}return Math.abs(v)/6;};
 const hullVolume=volume(faces,i=>P(support[i]!));if(!(hullVolume>0))return undefined;
 const meshTris:[number,number,number][]=[];
 // Halfedge h is original corner h (face h/3), so each face is read once.
 const H=complex.halfedges;for(let h=0;h+2<H.length;h+=3)meshTris.push([H[h]!.from,H[h+1]!.from,H[h+2]!.from]);
 if(volume(meshTris,P)<OUTLINE_HULL_FILL*hullVolume)return undefined;
 // Undirected edges with their two faces' third vertices.
 const third=new Map<string,number>();
 for(const [i,j,k]of faces){third.set(`${i},${j}`,k);third.set(`${j},${k}`,i);third.set(`${k},${i}`,j);}
 const records:number[]=[];
 for(const [key,c]of third){
  const [i,j]=key.split(',').map(Number) as [number,number];if(!(i<j))continue;
  const d=third.get(`${j},${i}`);if(d===undefined)return undefined; // not closed: no outline construction
  const a=support[i]!,b=support[j]!;records.push(a,b,support[c]!,support[d]!,faceAt.get(a)!);
 }
 // Hull edge k is drawn by the relation thread of source edge k, so it may
 // not have more edges than its source (an open sheet's few edges).
 if(records.length/OUTLINE_HULL_WORDS>complex.edges.length)return undefined;
 const out=new Uint32Array(2+records.length);out[0]=OUTLINE_HULL_MAGIC;out[1]=records.length/OUTLINE_HULL_WORDS;out.set(records,2);
 return out;
}
/** Incremental convex hull of a few points (at most 26): outward triangles as
 * point-index triples, or undefined when the points span no volume. */
export function convexHull(points:readonly (readonly [number,number,number])[],eps:number):[number,number,number][]|undefined{
 const n=points.length;if(n<4)return undefined;
 const sub=(a:readonly number[],b:readonly number[])=>[a[0]!-b[0]!,a[1]!-b[1]!,a[2]!-b[2]!];
 const cross=(a:number[],b:number[])=>[a[1]!*b[2]!-a[2]!*b[1]!,a[2]!*b[0]!-a[0]!*b[2]!,a[0]!*b[1]!-a[1]!*b[0]!];
 const dot=(a:number[],b:number[])=>a[0]!*b[0]!+a[1]!*b[1]!+a[2]!*b[2]!;
 const normal=(f:readonly number[])=>cross(sub(points[f[1]!]!,points[f[0]!]!),sub(points[f[2]!]!,points[f[0]!]!));
 const height=(f:readonly number[],p:number)=>{const m=normal(f),l=Math.hypot(...m);return l>0?dot(m,sub(points[p]!,points[f[0]!]!))/l:0;};
 // Initial tetrahedron from the extreme spread.
 let i1=0;for(let i=1;i<n;i++)if(Math.hypot(...sub(points[i]!,points[0]!))>Math.hypot(...sub(points[i1]!,points[0]!)))i1=i;
 let i2=-1,best=eps;for(let i=0;i<n;i++){const l=Math.hypot(...cross(sub(points[i1]!,points[0]!),sub(points[i]!,points[0]!)))/Math.max(eps,Math.hypot(...sub(points[i1]!,points[0]!)));if(l>best){best=l;i2=i;}}
 if(i2<0||i1===0)return undefined;
 let i3=-1;best=eps;for(let i=0;i<n;i++){const h=Math.abs(height([0,i1,i2],i));if(h>best){best=h;i3=i;}}
 if(i3<0)return undefined;
 let faces:[number,number,number][]=height([0,i1,i2],i3)>0?[[0,i2,i1],[0,i1,i3],[i1,i2,i3],[i2,0,i3]]:[[0,i1,i2],[0,i3,i1],[i1,i3,i2],[i2,i3,0]];
 for(let p=0;p<n;p++){
  if(p===0||p===i1||p===i2||p===i3)continue;
  const visible=faces.filter(f=>height(f,p)>eps);if(!visible.length)continue;
  const directed=new Set<string>();for(const [a,b,c]of visible){directed.add(`${a},${b}`);directed.add(`${b},${c}`);directed.add(`${c},${a}`);}
  const horizon:[number,number][]=[];
  for(const [a,b,c]of visible)for(const [x,y]of [[a,b],[b,c],[c,a]] as const)if(!directed.has(`${y},${x}`))horizon.push([x,y]);
  faces=faces.filter(f=>!visible.includes(f));
  for(const [x,y]of horizon)faces.push([x,y,p]);
 }
 return faces;
}
/** The compiled level table of one source: the L1 codes (compileEdgeLevels,
 * ceil(H/2) words, zeros when nothing is limited) followed by the L2 hull
 * section. Terrain and rounded sources keep their own whole-form producers:
 * codes only (terrain none). The relation stage finds the hull after the
 * codes (sourceHullAt) and checks its magic. */





















export const PROFILE_LATTICE=4,PROFILE_FINE=16;
export function compileMeshProfileSafe(positions:ArrayLike<number>,indices:ArrayLike<number>):boolean{
 const n=Math.floor(positions.length/3);if(n<3||indices.length<3)return false;
 const lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];
 for(let v=0;v<n;v++)for(let k=0;k<3;k++){const x=positions[v*3+k]!;if(!Number.isFinite(x))return false;lo[k]=Math.min(lo[k]!,x);hi[k]=Math.max(hi[k]!,x);}
 const L=PROFILE_FINE,size=[0,1,2].map(k=>hi[k]!-lo[k]!);if(!(Math.max(...size)>0))return false;
 // views: [u axis, v axis]; occupancy L x L, cell (u,v) with v=0 the low side.
 const views=[[0,1],[2,1],[0,2]] as const,grids=views.map(()=>new Uint8Array(L*L));
 const cell=(k:number,x:number)=>size[k]!>0?Math.min(L-1,Math.max(0,Math.floor((x-lo[k]!)/size[k]!*L))):0;
 const P=(v:number)=>[positions[v*3]!,positions[v*3+1]!,positions[v*3+2]!];
 for(let t=0;t+2<indices.length;t+=3){
  const a=P(indices[t]!),b=P(indices[t+1]!),c=P(indices[t+2]!);
  // Samples dense enough that consecutive ones are under half a cell apart in every axis.
  let span=1;for(let k=0;k<3;k++)if(size[k]!>0)span=Math.max(span,Math.ceil(Math.max(Math.abs(b[k]!-a[k]!),Math.abs(c[k]!-a[k]!),Math.abs(c[k]!-b[k]!))/size[k]!*L*2));
  span=Math.min(span,4*L);
  for(let i=0;i<=span;i++)for(let j=0;i+j<=span;j++){
   const s=i/span,r=j/span,q=[0,1,2].map(k=>a[k]!+s*(b[k]!-a[k]!)+r*(c[k]!-a[k]!));
   views.forEach(([u,w],g)=>{grids[g]![cell(w,q[w]!)*L+cell(u,q[u]!)]=1;});
  }
 }
 // An opening counts when it spans at least half a glyph cell of the largest
 // parent in both directions (PROFILE_FINE/PROFILE_LATTICE/2 fine cells).
 const least=Math.max(1,PROFILE_FINE/PROFILE_LATTICE/2);
 return !latticeOpening(grids[0]!,L,false,L,least)&&!latticeOpening(grids[1]!,L,false,L,least)&&!latticeOpening(grids[2]!,L,true,L,least);
}
/** An OPENING on an occupancy lattice (row-major, row 0 the low side): an
 * empty cell covered on both sides in its row and above it in its column (an
 * arch, a passage, a gap between piers under a lintel); `enclosed` also asks
 * for cover below (a court seen from above). Space under an overhang or beside
 * a narrowing form is open to one side and is not an opening: nothing is seen
 * through it. Shared by compileMeshProfileSafe and source-lod parentAdequate. */
// `least`: the opening cells must form a 4-connected region at least this
// many cells across in BOTH directions (1: any single cell counts).
export function latticeOpening(grid:ArrayLike<number>,columns:number,enclosed:boolean,rows=columns,least=1):boolean{
 const on=(u:number,w:number)=>!!grid[w*columns+u];
 const opening=new Uint8Array(columns*rows);
 for(let w=0;w<rows;w++)for(let u=1;u+1<columns;u++){
  if(on(u,w))continue;
  let left=false,right=false,above=false,below=!enclosed;
  for(let k=0;k<u&&!left;k++)left=on(k,w);
  for(let k=u+1;k<columns&&!right;k++)right=on(k,w);
  for(let k=w+1;k<rows&&!above;k++)above=on(u,k);
  for(let k=0;k<w&&!below;k++)below=on(u,k);
  if(left&&right&&above&&below){if(least<=1)return true;opening[w*columns+u]=1;}
 }
 if(least<=1)return false;
 const seen=new Uint8Array(columns*rows);
 for(let start=0;start<opening.length;start++){
  if(!opening[start]||seen[start])continue;
  let u0=columns,u1=-1,w0=rows,w1=-1;const stack=[start];seen[start]=1;
  while(stack.length){const at=stack.pop()!,u=at%columns,w=(at-u)/columns;u0=Math.min(u0,u);u1=Math.max(u1,u);w0=Math.min(w0,w);w1=Math.max(w1,w);
   for(const [du,dw]of [[1,0],[-1,0],[0,1],[0,-1]] as const){const x=u+du,y=w+dw;if(x<0||y<0||x>=columns||y>=rows)continue;const n=y*columns+x;if(opening[n]&&!seen[n]){seen[n]=1;stack.push(n);}}}
  if(u1-u0+1>=least&&w1-w0+1>=least)return true;
 }
 return false;
}



export function compileMeshSolid(positions:ArrayLike<number>,indices:ArrayLike<number>):boolean{
 const n=Math.floor(positions.length/3);if(n<4||indices.length<12||indices.length%3)return false;
 const lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];
 for(let v=0;v<n;v++)for(let k=0;k<3;k++){const x=positions[v*3+k]!;if(!Number.isFinite(x))return false;lo[k]=Math.min(lo[k]!,x);hi[k]=Math.max(hi[k]!,x);}
 const scale=Math.hypot(hi[0]!-lo[0]!,hi[1]!-lo[1]!,hi[2]!-lo[2]!);if(!(scale>0))return false;
 const eps=scale*1e-6,q=(x:number)=>Math.round(x/eps);
 const weld=new Map<string,number>(),id=new Int32Array(n),points:[number,number,number][]=[];
 for(let v=0;v<n;v++){const key=`${q(positions[v*3]!)},${q(positions[v*3+1]!)},${q(positions[v*3+2]!)}`;let w=weld.get(key);
  if(w===undefined){w=points.length;weld.set(key,w);points.push([positions[v*3]!,positions[v*3+1]!,positions[v*3+2]!]);}id[v]=w;}
 const uses=new Map<number,number>(),tris:[number,number,number][]=[];const W=points.length;
 for(let t=0;t+2<indices.length;t+=3){
  const a=id[indices[t]!]!,b=id[indices[t+1]!]!,c=id[indices[t+2]!]!;if(a===b||b===c||c===a)continue;tris.push([a,b,c]);
  for(const [i,j]of [[a,b],[b,c],[c,a]] as const){const key=Math.min(i,j)*W+Math.max(i,j);uses.set(key,(uses.get(key)??0)+1);}
 }
 for(const count of uses.values())if(count!==2)return false;
 const support:number[]=[];
 for(const [dx,dy,dz]of OUTLINE_DIRECTIONS){let best=-1,value=-Infinity;for(let i=0;i<W;i++){const p=points[i]!,s=p[0]*dx+p[1]*dy+p[2]*dz;if(s>value+eps){value=s;best=i;}}if(best>=0&&!support.includes(best))support.push(best);}
 const faces=convexHull(support.map(i=>points[i]!),eps);if(!faces)return false;
 const volume=(list:Iterable<readonly [number,number,number]>,at:(i:number)=>readonly number[])=>{let v=0;for(const [i,j,k]of list){const a=at(i),b=at(j),c=at(k);
  v+=a[0]!*(b[1]!*c[2]!-b[2]!*c[1]!)-a[1]!*(b[0]!*c[2]!-b[2]!*c[0]!)+a[2]!*(b[0]!*c[1]!-b[1]!*c[0]!);}return Math.abs(v)/6;};
 const hullVolume=volume(faces,i=>points[support[i]!]!);if(!(hullVolume>0))return false;
 return volume(tris,i=>points[i]!)>=OUTLINE_HULL_FILL*hullVolume;
}
export function compileConstructionLevels(topology:Uint32Array,complex:MeshSourceComplex,positions:ArrayLike<number>,type:string):Uint32Array|undefined{
 if(type==='terrain')return undefined;
 const H=complex.halfedges.length,words=Math.ceil(H/2);
 const compiled=compileEdgeLevels(topology,complex,positions);
 const codes=compiled.length?compiled:new Uint32Array(words);
 const hull=type==='rounded'||type==='paper'?undefined:compileOutlineHull(complex,positions);
 const chains=type==='rounded'||type==='paper'?undefined:compilePrincipalChains(topology,complex,positions,codes);
 if(!hull&&!chains)return compiled.length?compiled:undefined;
 // [codes][HULL_MAGIC, n, hull records][CHAIN_MAGIC, m, chain records]: the
 // hull header is written (n = 0) whenever chains follow it.
 const hullWords=hull??Uint32Array.of(OUTLINE_HULL_MAGIC,0),out=new Uint32Array(words+hullWords.length+(chains?.length??0));
 out.set(codes);out.set(hullWords,words);if(chains)out.set(chains,words+hullWords.length);return out;
}














export const PRINCIPAL_CHAIN_MAGIC=0x4348414e;
export const PRINCIPAL_CHAIN_WORDS=5;
const CHAIN_STRAIGHT_COS=.99985;
export function compilePrincipalChains(topology:Uint32Array,complex:MeshSourceComplex,positions:ArrayLike<number>,codes:Uint32Array):Uint32Array|undefined{
 const classesAt=topology[42]!,count=topology[43]!,packed=topology[37]===PACKED_RUN_CONTEXT;
 const klass=(h:number):number=>{if(h>=count)return 0;const v=topology[classesAt+h]!;return packed?v&3:v;};
 const code=(h:number):number=>(codes[h>>>1]!>>>((h&1)*16))&0xffff;
 const weld=complex.weld,vertex=(v:number):number=>weld?weld[v]!:v;
 // One record per welded principal crease with exactly two faces.
 const seen=new Set<number>(),edges:{h:number;members:number[];a:number;b:number;faces:[number,number]}[]=[];
 for(const edge of complex.edges){
  const key=edge.welded?.[0]??edge.id;if(seen.has(key))continue;seen.add(key);
  if(klass(edge.id)!==1||code(edge.id)!==0)continue;
  const members=[...(edge.welded??edge.incidences)];
  const faces=[...new Set(members.map(h=>complex.halfedges[h]!.face))];if(faces.length!==2)continue;
  edges.push({h:key,members,a:edge.vertices[0],b:edge.vertices[1],faces:[faces[0]!,faces[1]!]});
 }
 if(!edges.length)return undefined;
 const at=new Map<number,number[]>();
 edges.forEach((e,i)=>{for(const v of [vertex(e.a),vertex(e.b)]){const list=at.get(v);if(list)list.push(i);else at.set(v,[i]);}});
 const P=(v:number)=>[positions[v*3]!,positions[v*3+1]!,positions[v*3+2]!];
 const away=(i:number,w:number)=>{const e=edges[i]!,from=vertex(e.a)===w?e.a:e.b,to=from===e.a?e.b:e.a,p=P(from),q=P(to);return [q[0]!-p[0]!,q[1]!-p[1]!,q[2]!-p[2]!];};
 // The straight continuation through welded vertex w, if exactly one.
 const through=(i:number,w:number):number|undefined=>{
  const list=at.get(w)!;if(list.length!==2)return;const j=list[0]===i?list[1]!:list[0]!;
  const x=away(i,w),y=away(j,w),lx=Math.hypot(...x),ly=Math.hypot(...y);
  return lx>0&&ly>0&&-(x[0]!*y[0]!+x[1]!*y[1]!+x[2]!*y[2]!)>=CHAIN_STRAIGHT_COS*lx*ly?j:undefined;
 };
 const used=new Uint8Array(edges.length),records:number[]=[];
 for(let start=0;start<edges.length;start++){
  if(used[start])continue;
  // Walk to one end, then collect the chain from that end.
  let i=start,w=vertex(edges[i]!.a),guard=0;
  for(;;){const j=through(i,w);if(j===undefined||j===start||guard++>edges.length)break;const e=edges[j]!;w=vertex(e.a)===w?vertex(e.b):vertex(e.a);i=j;}
  const end=w;const chain:number[]=[];let cur=i,from=end;guard=0;
  for(;;){if(used[cur])break;used[cur]=1;chain.push(cur);const e=edges[cur]!;const to=vertex(e.a)===from?vertex(e.b):vertex(e.a);from=to;
   const j=through(cur,to);if(j===undefined||used[j]||guard++>edges.length)break;cur=j;}
  if(chain.length<2)continue;
  const first=edges[chain[0]!]!,last=edges[chain.at(-1)!]!;
  const a=vertex(first.a)===end?first.a:first.b,b=vertex(last.a)===from?last.a:last.b;
  records.push(a,b,first.faces[0],first.faces[1],first.h);
  for(const index of chain)for(const h of edges[index]!.members)if(h<complex.halfedges.length)codes[h>>>1]=((codes[h>>>1]!&~(0xffff<<((h&1)*16)))|(EDGE_LEVEL_CHAINED<<((h&1)*16)))>>>0;
 }
 if(!records.length)return undefined;
 // Chain k is drawn by the relation thread of source edge k.
 if(records.length/PRINCIPAL_CHAIN_WORDS>complex.edges.length)return undefined;
 const out=new Uint32Array(2+records.length);out[0]=PRINCIPAL_CHAIN_MAGIC;out[1]=records.length/PRINCIPAL_CHAIN_WORDS;out.set(records,2);
 return out;
}
