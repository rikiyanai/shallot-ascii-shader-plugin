import {SOURCE_PACKET_VERSION} from './source-packet.ts';
/** Source topology exists before projection, visibility and typography.
 * Coincident but separately indexed surfaces keep their own addresses and
 * their contact candidates (relations to evaluate on the current posed
 * geometry); the compiled adjacency (twins, components, boundary cycles) reads
 * them through the welded rest-position identity. */
export interface MeshSource {
  readonly id:string;
  readonly revision:number;
  readonly positions:Float32Array;
  readonly indices:Uint32Array;
  /** Exact native position-lattice codes, when the engine quantizes a source.
   * They generate contact candidates only; posing must still validate them. */
  readonly contactKeys?:Uint32Array;
  /** Published face incidence, e.g. separately indexed P0 triangles. This
   * authorizes candidate correspondence, never a permanent posed weld. */
  readonly adjacency?:{readonly offsets:Uint32Array;readonly faces:Uint32Array};
}
export interface SourceHalfedge {
  readonly id:number;
  readonly from:number;
  readonly to:number;
  readonly face:number;
  readonly next:number;
  readonly edge:number;
  /** Only an unambiguous, oppositely oriented SOURCE incidence is a twin. */
  readonly twin:number|undefined;
}
export interface SourceEdge {
  readonly id:number;
  readonly vertices:readonly [number,number];
  /** INDEX incidence: the halfedges that name this exact vertex-index pair.
   * The relation shader pairs it with the contact groups, so it stays as it
   * was (one halfedge per edge on a triangle soup). */
  readonly incidences:readonly number[];
  /** WELDED incidence: every halfedge on this position-equal edge, ascending;
   * `welded[0]` is the welded edge's canonical address. Index edges of one
   * welded edge share the same list. A crease of a triangle soup has two. */
  readonly welded?:readonly number[];
}
export interface SourceBoundary {
  readonly id:string;
  readonly component:number;
  readonly halfedges:readonly number[];
  readonly closed:boolean;
}
export interface SourceContactCandidate {
  readonly members:readonly {readonly edge:number;readonly forward:boolean}[];
  /** Current 3D coincidence must still hold after material deformation.
   * This relation never asserts a permanent physical attachment. */
  readonly kind:'coincident-source-boundaries'|'published-source-incidence';
}
export interface MeshSourceComplex {
  readonly id:string;
  readonly revision:number;
  readonly halfedges:readonly SourceHalfedge[];
  readonly edges:readonly SourceEdge[];
  readonly faceComponents:Uint32Array;
  readonly boundaries:readonly SourceBoundary[];
  readonly contacts:readonly SourceContactCandidate[];
  readonly vertexContacts?:readonly (readonly number[])[];
  /** Welded vertex identity: vertex -> lowest original index at that rest
   * position (SOURCE_WELD_TOLERANCE). Original indices stay the addresses. */
  readonly weld?:Uint32Array;
  readonly issues:readonly {readonly kind:'nonmanifold-edge'|'inconsistent-winding'|'open-boundary';readonly source:number}[];
}
export interface SourceInterval {
  readonly source:string;
  readonly halfedge:number;
  readonly parameter:readonly [number,number];
  readonly visibility:'visible'|'occluded'|'outside-view';
  readonly cause?:string;
}
export interface SurfaceContourEndpoint {
  readonly source:string;
  readonly edge:number;
  readonly parameter:number;
}

/** Rest positions closer than this on every axis are one welded vertex. It is
 * the same bound the contact groups and the relation shader use (`same`). */
export const SOURCE_WELD_TOLERANCE=1e-6;

/** Welded vertex identity: vertex -> the lowest original index whose rest
 * position matches within the tolerance on every axis. Original indices stay
 * the addresses; this only says which of them are the same point, so a mesh
 * exported as triangle soup (three private indices per triangle) regains the
 * adjacency its positions describe. A hash grid of representatives, expected
 * O(n); deterministic (index order, lowest match). Compile time only. */
export function weldSourceVertices(positions:ArrayLike<number>,tolerance=SOURCE_WELD_TOLERANCE):Uint32Array{
  const count=Math.floor(positions.length/3),weld=new Uint32Array(count),chain=new Int32Array(count).fill(-1),heads=new Map<number,number>();
  const inverse=1/(tolerance*2);
  // Cells are two tolerances wide, so a match lies in at most two cells per
  // axis. The hash only buckets; equality is always the position test below.
  const hash=(x:number,y:number,z:number)=>Math.imul(x|0,73856093)^Math.imul(y|0,19349663)^Math.imul(z|0,83492791);
  for(let v=0;v<count;v++){
    const x=positions[v*3]!,y=positions[v*3+1]!,z=positions[v*3+2]!;
    const x0=Math.floor((x-tolerance)*inverse),x1=Math.floor((x+tolerance)*inverse),y0=Math.floor((y-tolerance)*inverse),y1=Math.floor((y+tolerance)*inverse),z0=Math.floor((z-tolerance)*inverse),z1=Math.floor((z+tolerance)*inverse);
    let found=-1;
    for(let i=x0;i<=x1;i++)for(let j=y0;j<=y1;j++)for(let k=z0;k<=z1;k++)
      for(let at=heads.get(hash(i,j,k))??-1;at>=0;at=chain[at]!)
        if((found<0||at<found)&&Math.abs(positions[at*3]!-x)<=tolerance&&Math.abs(positions[at*3+1]!-y)<=tolerance&&Math.abs(positions[at*3+2]!-z)<=tolerance)found=at;
    if(found>=0){weld[v]=found;continue;}
    weld[v]=v;const key=hash(Math.floor(x*inverse),Math.floor(y*inverse),Math.floor(z*inverse));
    chain[v]=heads.get(key)??-1;heads.set(key,v);
  }
  return weld;
}

/** Compile one mesh's immutable incidence. Camera changes do not rebuild this
 * structure. Halfedges, all boundary cycles and nonmanifold incidences
 * survive; projected crossings cannot manufacture an adjacency.
 *
 * Twins, face components, boundary cycles and `SourceEdge.welded` come from
 * the WELDED vertex identity (weldSourceVertices): position-equal corners are
 * one vertex of the surface, whatever index each triangle used. Read by index
 * alone, a triangle soup had no twin, one closed "boundary" per triangle and
 * one component per triangle, so nothing compiled from it knew a crease from
 * an open edge or a sheet from a solid. Addresses do not change: halfedge
 * from/to, edge ids, `SourceEdge.incidences` (index incidence, which the
 * relation shader pairs with the contact groups) and the contact groups
 * themselves are exactly as before; a contact is still a candidate that the
 * posed geometry must confirm. */
export function compileMeshSource(source:MeshSource):MeshSourceComplex{
  const {positions,indices}=source;
  if(!source.id||!Number.isSafeInteger(source.revision)||source.revision<0||positions.length%3||indices.length%3||source.contactKeys&&source.contactKeys.length!==positions.length/3*2)throw Error('Invalid triangle source extent or revision');
  for(const value of positions)if(!Number.isFinite(value))throw Error('Nonfinite source position');
  const vertexCount=positions.length/3,faceCount=indices.length/3;
  for(const index of indices)if(index>=vertexCount)throw Error('Source index is outside its vertex stream');
  const weld=weldSourceVertices(positions);
  const halfedges:{id:number;from:number;to:number;face:number;next:number;edge:number;twin:number|undefined}[]=[];
  const edges:{id:number;vertices:[number,number];incidences:number[]}[]=[];
  const edgeByVertices=new Map<string,number>();
  const issues:MeshSourceComplex['issues'][number][]=[];
  // One group per welded edge, in first-halfedge order; members ascend, so
  // group[0] is the address an indexed mesh already used for that edge. A
  // halfedge whose two ends weld to one point (a sliver) joins no group.
  const weldedGroups:number[][]=[],weldedByVertices=new Map<number,number>(),weldedOf=new Int32Array(indices.length).fill(-1);
  for(let at=0;at<indices.length;at++){
    const face=Math.floor(at/3),next=face*3+(at+1)%3,from=indices[at]!,to=indices[next]!;
    if(from===to)throw Error('Degenerate source edge repeats a vertex');
    const a=Math.min(from,to),b=Math.max(from,to),key=`${a}:${b}`;
    let edge=edgeByVertices.get(key);
    if(edge===undefined){edge=edges.length;edgeByVertices.set(key,edge);edges.push({id:at,vertices:[a,b],incidences:[]});}
    edges[edge]!.incidences.push(at);halfedges.push({id:at,from,to,face,next,edge,twin:undefined});
    const wa=weld[from]!,wb=weld[to]!;if(wa===wb)continue;
    const weldedKey=Math.min(wa,wb)*vertexCount+Math.max(wa,wb);let group=weldedByVertices.get(weldedKey);
    if(group===undefined){group=weldedGroups.length;weldedByVertices.set(weldedKey,group);weldedGroups.push([]);}
    weldedGroups[group]!.push(at);weldedOf[at]=group;
  }
  const parent=Uint32Array.from({length:faceCount},(_,i)=>i);
  const root=(id:number)=>{let r=id;while(parent[r]!==r)r=parent[r]!;while(parent[id]!==id){const next=parent[id]!;parent[id]=r;id=next;}return r;};
  for(const group of weldedGroups){
    for(let i=1;i<group.length;i++){
      const a=root(halfedges[group[0]!]!.face),b=root(halfedges[group[i]!]!.face);
      parent[Math.max(a,b)]=Math.min(a,b);
    }
    if(group.length>2){issues.push({kind:'nonmanifold-edge',source:group[0]!});continue;}
    if(group.length===2){
      const a=halfedges[group[0]!]!,b=halfedges[group[1]!]!;
      if(weld[a.from]!==weld[b.to]||weld[a.to]!==weld[b.from]){issues.push({kind:'inconsistent-winding',source:group[0]!});continue;}
      a.twin=b.id;b.twin=a.id;
    }
  }
  const faceComponents=Uint32Array.from({length:faceCount},(_,i)=>root(i));
  // A boundary is a welded edge with one face. The index boundary below still
  // feeds the contact groups, which keep their published form.
  const openEdge=(halfedge:number)=>weldedOf[halfedge]!>=0&&weldedGroups[weldedOf[halfedge]!]!.length===1;
  const boundaryEdges=halfedges.filter(h=>openEdge(h.id)),indexBoundaryEdges=halfedges.filter(h=>edges[h.edge]!.incidences.length===1);
  const visited=new Set<number>(),boundaries:SourceBoundary[]=[];
  for(const start of boundaryEdges){
    if(visited.has(start.id))continue;
    const cycle:number[]=[];let at:number|undefined=start.id,closed=false;
    while(at!==undefined&&!visited.has(at)){
      const edge:SourceHalfedge=halfedges[at]!;visited.add(at);cycle.push(at);
      let next:number=edge.next,steps=0;
      while(halfedges[next]!.twin!==undefined&&steps++<=halfedges.length)next=halfedges[halfedges[next]!.twin!]!.next;
      if(steps>halfedges.length||!openEdge(next)){at=undefined;break;}
      if(next===start.id){closed=true;break;}
      at=next;
    }
    const first=cycle.reduce((a,b)=>Math.min(a,b),Infinity);
    boundaries.push({id:`${source.id}/boundary-${first}`,component:faceComponents[start.face]!,halfedges:Object.freeze(cycle),closed});
    if(!closed)issues.push({kind:'open-boundary',source:first});
  }
  let vertexContacts:readonly (readonly number[])[]|undefined;
  const contacts:SourceContactCandidate[]=[];
  if(source.adjacency){
    const {offsets,faces}=source.adjacency;
    if(offsets[0]!==0||offsets.at(-1)!==faces.length)throw Error('Invalid published source incidence');
    const vertices=Uint32Array.from({length:vertexCount},(_,i)=>i),edgeParents=Uint32Array.from({length:edges.length},(_,i)=>i);
    const find=(parent:Uint32Array,value:number)=>{let at=value;while(parent[at]!==at)at=parent[at]!;while(parent[value]!==value){const next=parent[value]!;parent[value]=at;value=next;}return at;};
    const join=(parent:Uint32Array,a:number,b:number)=>{a=find(parent,a);b=find(parent,b);parent[Math.max(a,b)]=Math.min(a,b);};
    const same=(a:number,b:number)=>[0,1,2].every(axis=>Math.abs(positions[a*3+axis]!-positions[b*3+axis]!)<=1e-6);
    for(let group=0;group+1<offsets.length;group++){
      const first=offsets[group]!,last=offsets[group+1]!;
      if(last<first||last>faces.length)throw Error('Invalid published source incidence range');
      for(let a=first;a<last;a++)for(let b=a+1;b<last;b++){
        const fa=faces[a]!,fb=faces[b]!;if(fa>=faceCount||fb>=faceCount)throw Error('Published incidence names an absent source face');
        for(let i=0;i<3;i++)for(let j=0;j<3;j++){
          const x=halfedges[fa*3+i]!,y=halfedges[fb*3+j]!;
          const forward=same(x.from,y.from)&&same(x.to,y.to),reverse=same(x.from,y.to)&&same(x.to,y.from);
          if(!forward&&!reverse)continue;
          join(edgeParents,x.edge,y.edge);join(vertices,x.from,forward?y.from:y.to);join(vertices,x.to,forward?y.to:y.from);
          const ra=root(fa),rb=root(fb);parent[Math.max(ra,rb)]=Math.min(ra,rb);
        }
      }
    }
    // Keep ORIGINAL indices/addresses. SurfaceRelations revalidates current
    // positions before these candidates can share a port/edge. Equal-position
    // corners outside the published neighborhood are deliberately not added.
    const groups=new Map<number,number[]>();
    for(let v=0;v<vertexCount;v++){const key=find(vertices,v),group=groups.get(key)??[];group.push(v);groups.set(key,group);}
    vertexContacts=[...groups.values()];
    const edgeGroups=new Map<number,SourceEdge[]>();
    for(let edge=0;edge<edges.length;edge++){const key=find(edgeParents,edge),group=edgeGroups.get(key)??[];group.push(edges[edge]!);edgeGroups.set(key,group);}
    for(const group of edgeGroups.values())if(group.length>1){
      const first=halfedges[group[0]!.id]!,origin=find(vertices,first.from);
      contacts.push({kind:'published-source-incidence',members:group.map(edge=>({edge:edge.id,forward:find(vertices,halfedges[edge.id]!.from)===origin}))});
    }
    for(let face=0;face<faceCount;face++)faceComponents[face]=root(face);
  }
  // Exact rest-source coincidence creates a candidate contact, not a weld.
  // Preserve both source addresses and their orientation, including UV seams
  // and independently indexed coplanar faces.
  const bits=new Uint32Array(positions.buffer,positions.byteOffset,positions.length);
  const vertexKey=(vertex:number)=>source.contactKeys?`${source.contactKeys[vertex*2]}:${source.contactKeys[vertex*2+1]}`:[0,1,2].map(axis=>positions[vertex*3+axis]===0?0:bits[vertex*3+axis]).join(':');
  const groups=new Map<string,{edge:SourceEdge;forward:boolean}[]>();
  for(const halfedge of source.adjacency?[]:indexBoundaryEdges){
    const edge=edges[halfedge.edge]!,a=vertexKey(halfedge.from),b=vertexKey(halfedge.to),forward=a<b;
    const key=forward?`${a}/${b}`:`${b}/${a}`;
    const group=groups.get(key)??[];group.push({edge,forward});groups.set(key,group);
  }
  for(const group of groups.values())if(group.length>1)contacts.push({members:Object.freeze(group.map(member=>Object.freeze({edge:member.edge.id,forward:member.forward}))),kind:'coincident-source-boundaries'});
  for(const group of weldedGroups)Object.freeze(group);
  return Object.freeze({id:source.id,revision:source.revision,halfedges:Object.freeze(halfedges.map(halfedge=>Object.freeze(halfedge))),edges:Object.freeze(edges.map(edge=>Object.freeze({...edge,vertices:Object.freeze(edge.vertices),incidences:Object.freeze(edge.incidences),welded:weldedOf[edge.id]!>=0?weldedGroups[weldedOf[edge.id]!]!:Object.freeze([...edge.incidences])}))),faceComponents,boundaries:Object.freeze(boundaries.map(boundary=>Object.freeze({...boundary,component:faceComponents[halfedges[boundary.halfedges[0]!]!.face]!}))),contacts:Object.freeze(contacts),vertexContacts,weld,issues:Object.freeze(issues)});
}

/** Visibility annotates a source interval. It never manufactures a new path
 * identity or changes a closed source region into an unrelated open contour. */
export function sourceInterval(source:string,halfedge:number,parameter:readonly [number,number],visibility:SourceInterval['visibility'],cause?:string):SourceInterval{
  if(!source||!Number.isSafeInteger(halfedge)||halfedge<0||!parameter.every(Number.isFinite)||parameter[0]<0||parameter[1]>1||parameter[0]>parameter[1])throw Error('Invalid source visibility interval');
  return Object.freeze({source,halfedge,parameter:Object.freeze([...parameter]) as readonly [number,number],visibility,cause});
}

/** GPU word image of the same source graph. Offsets are u32 words, never byte
 * offsets. No source incidence is discarded to fit a two-face edge slot. */
export function packMeshSource(source:MeshSource,complex:MeshSourceComplex,attributes:{readonly materialClasses?:Uint32Array}={}):Uint32Array{
  if(source.id!==complex.id||source.revision!==complex.revision||source.indices.length!==complex.halfedges.length)throw Error('Source graph revision mismatch');
  const words:number[]=new Array<number>(16).fill(0);
  const reserve=(count:number)=>{for(let i=0;i<count;i++)words.push(0);};
  const set=(at:number,values:readonly number[])=>{for(let i=0;i<values.length;i++)words[at+i]=values[i]!;};
  const halfedges=words.length;
  for(const edge of complex.halfedges)words.push(edge.from,edge.to,edge.next,edge.twin??0xffffffff,edge.edge,edge.face,complex.faceComponents[edge.face]!,edge.id);
  const edgeRecords=words.length;reserve(complex.edges.length*4);
  complex.edges.forEach((edge,i)=>{
    const at=words.length;for(const incidence of edge.incidences)words.push(incidence);
    set(edgeRecords+i*4,[edge.id,at,edge.incidences.length,0]);
  });
  const boundaries=words.length;reserve(complex.boundaries.length*4);
  complex.boundaries.forEach((boundary,i)=>{
    const at=words.length;for(const edge of boundary.halfedges)words.push(edge);
    set(boundaries+i*4,[boundary.component,at,boundary.halfedges.length,Number(boundary.closed)]);
  });
  const contacts=words.length;reserve(complex.contacts.length*2);
  const edgeIndices=new Map(complex.edges.map((edge,index)=>[edge.id,index]));
  complex.contacts.forEach((contact,i)=>{
    const at=words.length;for(const member of contact.members)words.push(member.edge,Number(member.forward));
    set(contacts+i*2,[at,contact.members.length]);
    for(const member of contact.members)words[edgeRecords+edgeIndices.get(member.edge)!*4+3]=i+1;
  });
  const positions=words.length;
  const bits=new Uint32Array(source.positions.buffer,source.positions.byteOffset,source.positions.length);for(const value of bits)words.push(value);
  const vertexGroups=new Map<string,number[]>();
  for(let vertex=0;!complex.vertexContacts&&vertex<source.positions.length/3;vertex++){
    const key=source.contactKeys?`${source.contactKeys[vertex*2]}:${source.contactKeys[vertex*2+1]}`:[0,1,2].map(axis=>source.positions[vertex*3+axis]===0?0:bits[vertex*3+axis]).join(':');
    const group=vertexGroups.get(key)??[];group.push(vertex);vertexGroups.set(key,group);
  }
  const vertexIncidences:number[][]=Array.from({length:source.positions.length/3},()=>[]);
  for(const halfedge of complex.halfedges)vertexIncidences[halfedge.from]!.push(halfedge.id);
  const vertexContacts=words.length;reserve(source.positions.length/3*4);
  for(const group of complex.vertexContacts??vertexGroups.values()){
    const at=words.length;for(const vertex of group)words.push(vertex);
    for(const vertex of group){
      const incidencesAt=words.length;for(const halfedge of vertexIncidences[vertex]!)words.push(halfedge);
      set(vertexContacts+vertex*4,[at,group.length,incidencesAt,vertexIncidences[vertex]!.length]);
    }
  }
  // Header12 no longer duplicates complex.issues.length; that complete
  // CPU diagnostic list stays on complex. V2 uses it only as a face-class offset.
  let faceClasses=0;const classes=attributes.materialClasses,faces=source.indices.length/3;
  if(classes){
    if(classes.length!==Math.ceil(faces/8)||(faces%8!==0&&(classes.at(-1)!>>>((faces%8)*4))!==0))throw Error('Material classes exceed source face extent');
    for(let f=0;f<faces;f++)if(((classes[f>>>3]!>>>((f&7)*4))&15)>8)throw Error('Invalid source face material class');
    if(classes.some(value=>value!==0)){faceClasses=words.length;for(const value of classes)words.push(value);}
  }
  set(0,[0x4d41494e,SOURCE_PACKET_VERSION,source.positions.length/3,faces,halfedges,edgeRecords,complex.edges.length,boundaries,complex.boundaries.length,contacts,complex.contacts.length,positions,faceClasses,source.revision,vertexContacts,0]);
  return new Uint32Array(words);
}

/** What the welded surface of one source is, measured once per revision from
 * rest geometry alone: how many connected pieces, whether it is open, which
 * way it faces and how flat it lies. `planeRoot` is the connected-coplanar
 * region of every face: faces of one quantised plane (`facePlane`) that reach
 * each other through welded edges, named by their lowest face. */
export interface SourceSheet {
  readonly weld:Uint32Array;
  readonly planeRoot:Uint32Array;
  /** Connected face components through welded edges. */
  readonly components:number;
  /** Largest component's share of the surface area, 0..1. */
  readonly sheetShare:number;
  /** Welded edges with exactly one face: the open boundary. */
  readonly openEdges:number;
  /** Unit area-weighted mean face normal ([0,1,0] when the normals cancel). */
  readonly up:readonly [number,number,number];
  /** Share of the area whose normal lies on `up`'s side (n·up > TERRAIN_RULE.facingCosine). */
  readonly facing:number;
  /** Extent of the indexed vertices along `up`. */
  readonly relief:number;
  /** Square root of the area projected on the plane across `up`: the lateral
   * size, independent of how the sheet is turned about `up`. */
  readonly lateral:number;
}
/** Registration may supply the weld already compiled for these same source
 * positions. Sheet components/plane joins remain this analysis's own work;
 * published contact incidence is not substituted for position identity. */
export function compileSourceSheet(positions:ArrayLike<number>,indices:ArrayLike<number>,faceNormals:ArrayLike<number>,facePlane:ArrayLike<number>,sourceWeld?:Uint32Array):SourceSheet{
  const vertexCount=Math.floor(positions.length/3),faceCount=Math.floor(indices.length/3),weld=sourceWeld??weldSourceVertices(positions);
  const component=Uint32Array.from({length:faceCount},(_,i)=>i),planeRoot=Uint32Array.from({length:faceCount},(_,i)=>i);
  const find=(parent:Uint32Array,id:number)=>{let r=id;while(parent[r]!==r)r=parent[r]!;while(parent[id]!==id){const next=parent[id]!;parent[id]=r;id=next;}return r;};
  const join=(parent:Uint32Array,a:number,b:number)=>{a=find(parent,a);b=find(parent,b);parent[Math.max(a,b)]=Math.min(a,b);};
  // Welded edge -> first face * 4 + incidence count (held at 3).
  const edges=new Map<number,number>(),area=new Float64Array(faceCount),mean=[0,0,0];let total=0;
  for(let face=0;face<faceCount;face++){
    const i=[indices[face*3]!,indices[face*3+1]!,indices[face*3+2]!];
    const ax=positions[i[1]!*3]!-positions[i[0]!*3]!,ay=positions[i[1]!*3+1]!-positions[i[0]!*3+1]!,az=positions[i[1]!*3+2]!-positions[i[0]!*3+2]!;
    const bx=positions[i[2]!*3]!-positions[i[0]!*3]!,by=positions[i[2]!*3+1]!-positions[i[0]!*3+1]!,bz=positions[i[2]!*3+2]!-positions[i[0]!*3+2]!;
    const weight=Math.hypot(ay*bz-az*by,az*bx-ax*bz,ax*by-ay*bx)/2;area[face]=weight;total+=weight;
    for(let k=0;k<3;k++)mean[k]!+=weight*faceNormals[face*3+k]!;
    for(let k=0;k<3;k++){
      const a=weld[i[k]!]!,b=weld[i[(k+1)%3]!]!;if(a===b)continue;
      const key=Math.min(a,b)*vertexCount+Math.max(a,b),seen=edges.get(key);
      if(seen===undefined){edges.set(key,face*4+1);continue;}
      const first=Math.floor(seen/4);join(component,first,face);
      if(facePlane[first]!>=0&&facePlane[first]===facePlane[face])join(planeRoot,first,face);
      if(seen%4<3)edges.set(key,seen+1);
    }
  }
  let openEdges=0;for(const seen of edges.values())if(seen%4===1)openEdges++;
  const length=Math.hypot(mean[0]!,mean[1]!,mean[2]!),up:[number,number,number]=length>total*1e-9?[mean[0]!/length,mean[1]!/length,mean[2]!/length]:[0,1,0];
  const shares=new Map<number,number>();let facing=0,projected=0,low=Infinity,high=-Infinity;
  for(let face=0;face<faceCount;face++){
    const root=find(component,face);planeRoot[face]=find(planeRoot,face);shares.set(root,(shares.get(root)??0)+area[face]!);
    const cosine=faceNormals[face*3]!*up[0]+faceNormals[face*3+1]!*up[1]+faceNormals[face*3+2]!*up[2];
    if(cosine>TERRAIN_RULE.facingCosine){facing+=area[face]!;projected+=area[face]!*cosine;}
    for(let k=0;k<3;k++){const v=indices[face*3+k]!*3,height=positions[v]!*up[0]+positions[v+1]!*up[1]+positions[v+2]!*up[2];low=Math.min(low,height);high=Math.max(high,height);}
  }
  let largest=0;for(const share of shares.values())largest=Math.max(largest,share);
  return {weld,planeRoot,components:shares.size,sheetShare:total?largest/total:0,openEdges,up,facing:total?facing/total:0,relief:faceCount?high-low:0,lateral:Math.sqrt(projected)};
}

/** The height-surface rule (class `'terrain'`), geometry only; thresholds:
 *  - minPlanes 25: more than the 24 quantised planes that still class a source
 *    as planar, so no box, room or voxel block changes class; terrain is taken
 *    only out of what was path-led.
 *  - sheetShare .9: one connected welded sheet (or one sheet plus scraps)
 *    holds nine tenths of the area; a scatter of separate pieces is not a
 *    ground.
 *  - openEdges > 0: a ground sheet has an open rim; a closed bumpy solid
 *    (rock, crown) has none.
 *  - facing .9 at facingCosine .05: nine tenths of the area faces the mean
 *    normal's side by more than about 3 degrees, i.e. the sheet is a height
 *    function over one plane; walls and overhangs do not count.
 *  - up .5: that mean normal is within 60 degrees of source +Y, the same up
 *    the support test uses; a banner or a wall panel is not ground.
 *  - relief .5: height range along the mean normal is at most half the lateral
 *    size (square root of the projected area). A hemisphere dome measures
 *    about .56 and stays out; rolling ground and cliffs below that stay in. */
export const TERRAIN_RULE=Object.freeze({minPlanes:25,sheetShare:.9,facing:.9,facingCosine:.05,up:.5,relief:.5});
export function isTerrainSheet(sheet:SourceSheet,planeCount:number):boolean{
  return planeCount>=TERRAIN_RULE.minPlanes&&sheet.openEdges>0&&sheet.sheetShare>=TERRAIN_RULE.sheetShare&&sheet.facing>=TERRAIN_RULE.facing&&sheet.up[1]>=TERRAIN_RULE.up&&sheet.lateral>0&&sheet.relief<=TERRAIN_RULE.relief*sheet.lateral;
}

/** Boundary cycles of one connected plane region, as ORIGINAL halfedge
 * addresses (face * 3 + corner) in walking order. A region edge is on the
 * boundary when only one of the region's faces has it (welded identity), so a
 * soup region has the same cycles as an indexed one. The cycle enclosing the
 * largest area in the plane (u, v) is the outline; every other cycle is an
 * opening. Two cycles pinched at one vertex may be walked as one. */
export function planeRegionLoops(positions:ArrayLike<number>,indices:ArrayLike<number>,weld:Uint32Array,faces:readonly number[],u:ArrayLike<number>,v:ArrayLike<number>):{boundary:readonly number[];openings:readonly (readonly number[])[]}{
  const vertexCount=Math.floor(positions.length/3),counts=new Map<number,number>();
  const ends=(halfedge:number):[number,number]=>{const face=Math.floor(halfedge/3);return [weld[indices[halfedge]!]!,weld[indices[face*3+(halfedge+1)%3]!]!];};
  const key=(a:number,b:number)=>Math.min(a,b)*vertexCount+Math.max(a,b);
  for(const face of faces)for(let k=0;k<3;k++){const [a,b]=ends(face*3+k);if(a!==b)counts.set(key(a,b),(counts.get(key(a,b))??0)+1);}
  const leaving=new Map<number,number[]>(),open:number[]=[];
  for(const face of faces)for(let k=0;k<3;k++){const halfedge=face*3+k,[a,b]=ends(halfedge);if(a===b||counts.get(key(a,b))!==1)continue;open.push(halfedge);const list=leaving.get(a)??[];list.push(halfedge);leaving.set(a,list);}
  const used=new Set<number>(),loops:{halfedges:number[];area:number}[]=[];
  for(const start of open){
    if(used.has(start))continue;
    const halfedges:number[]=[];let area=0;
    for(let at:number|undefined=start;at!==undefined;){
      used.add(at);halfedges.push(at);
      const face=Math.floor(at/3),p=indices[at]!*3,q=indices[face*3+(at+1)%3]!*3;
      const px=positions[p]!*u[0]!+positions[p+1]!*u[1]!+positions[p+2]!*u[2]!,py=positions[p]!*v[0]!+positions[p+1]!*v[1]!+positions[p+2]!*v[2]!;
      const qx=positions[q]!*u[0]!+positions[q+1]!*u[1]!+positions[q+2]!*u[2]!,qy=positions[q]!*v[0]!+positions[q+1]!*v[1]!+positions[q+2]!*v[2]!;
      area+=px*qy-qx*py;
      at=leaving.get(ends(at)[1])?.find(next=>!used.has(next));
    }
    loops.push({halfedges,area:Math.abs(area)/2});
  }
  let outer=0;for(let i=1;i<loops.length;i++)if(loops[i]!.area>loops[outer]!.area)outer=i;
  return {boundary:loops[outer]?.halfedges??[],openings:loops.filter((_,i)=>i!==outer).map(loop=>loop.halfedges)};
}
