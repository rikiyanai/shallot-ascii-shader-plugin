import {compileSourceFaceMaterialClasses} from './source-material.ts';
import {compileMeshSource,packMeshSource,type MeshSource,type MeshSourceComplex} from './source-complex.ts';
import {compileSourceAnalysis,type SourceAnalysis,type CompositionGeometry} from './composition.ts';
import {meshPartWork,partPoint,PART_CAP,PART_FORK,PART_ROOT,type MeshPartGraph,type PartSample,type PartSection} from './mesh-part-graph.ts';
import {compileSourceRunProgram,compileConstructionLevels,compilePlanarLevels,compileMeshProfileSafe,compileSourcePartPrograms,compileSourcePlanarBoundaryPassages,sourcePartProgramPacket,sourcePartContextCandidates,sourcePartPassageCandidates,sourcePartOpenPassage,sourcePartAnchorVertices,sourcePartJunctions,type PlanarLevels,type SourcePlanarBoundaryPassage,type SourcePartJunctionRecord,type SourcePartContextGeometry,type SourcePartPassageAnchor,type SourcePartProgramRecord,TOPOLOGY_PART_PACKET_MAGIC,TOPOLOGY_PART_PROGRAM_WORDS,TOPOLOGY_PART_CANONICAL_MAGIC,TOPOLOGY_PART_CANONICAL_VERSION,TOPOLOGY_PART_CANONICAL_HEADER_WORDS,TOPOLOGY_PART_CANONICAL_RECORD_WORDS,TOPOLOGY_PART_CANONICAL_SAMPLE_WORDS,TOPOLOGY_PART_CANONICAL_FLAG_CLOSED,TOPOLOGY_PART_CANONICAL_FLAG_COMPLETE} from './compiled-construction-topology.ts';
import {compileSourceIncidencePlan,type CompiledSourceIncidencePlan} from './source-incidence-residual.ts';
import {sourceBandIncidence} from './source-band-incidence.ts';
import {compileSourceSmallParent,compileSourceRelationships,sourceContactRequirements,type SourceSmallParent} from './source-small-parent.ts';
import type {SourceResidencySummary} from './source-transfer.ts';
import type {CompiledSourceRunProgram} from './compiled-construction-topology.ts';
import {validateSourceConstruction} from './source-construction.ts';
import {compileSourceCornerIndex,type SourceCornerIndexData} from './source-contact-corners.ts';
import type {SourceWholeConstructionTemplate,SourceWholeOpeningTemplate,SourceWholePartRole,SourceWholePartTemplate} from './source-lod.ts';
import {GRAMMAR_ROLES,type GrammarRoleEntry} from './role-grammar-format.ts';
import {ROLE_ENTRIES} from './grammar/role-bank.ts';
import roundSupplement from './grammar/role-rounds-supplement.json' with {type:'json'};

/** Immutable source-revision work. Projection, lights, pens and glyph choice
 * are deliberately absent. Both source consumers receive one compilation. */
interface SourceProgramData {
 readonly packet:{readonly kind:'legacy-v2';readonly words:Uint32Array}|{readonly kind:'gpu-csr-v1';readonly plan:CompiledSourceIncidencePlan};
 /** The same immutable revision snapshot used by compilation. The adapter
  * must not publish its producer's subsequently mutable source wrapper. */
 readonly geometry:CompositionGeometry&Pick<MeshSource,'contactKeys'>;
 readonly physical?:CompiledPhysicalSource;
 /** Source-revision-owned contact correspondence; absent on legacy packets. */
 readonly cornerIndex?:SourceCornerIndexData;
 /** Set only when the caller explicitly requested contact corners. */
 readonly cornerIndexState?:'ready'|'empty';
 /** Static parent-profile verdict; absent means an older cached program. */
 readonly profileSafe?:boolean;
 /** Source-owned connected parent program compiled from the same graph as the
  * resident topology. It is a template until an adapter binds (mesh,eid). */
 readonly wholeConstruction?:SourceWholeConstructionTemplate;
 /** Original source vertex ids, compiled once; never visibility fragments. */
 readonly partAnchorVertices?:Uint32Array;




 readonly smallParent?:SourceSmallParent;
}
export type SourceComplexRetention='full'|'summary';
/** Everything the retained GPU drawing actually reads. Source planes, sheet
 * analysis, sections and rest normals have already compiled into runProgram;
 * retaining their object graphs as well made streamed terrain a CPU heap. */
export type PlaybackSourceAnalysis=Pick<SourceAnalysis,'type'|'styleWord'|'foliage'|'organic'|'toneRegions'>&{
 readonly runProgram:CompiledSourceRunProgram;
};
export interface FullCompiledSourceProgram extends SourceProgramData {
 readonly complexMode:'full';readonly complex:MeshSourceComplex;readonly analysis:SourceAnalysis;
}
export interface SummaryCompiledSourceProgram extends SourceProgramData {
 readonly complexMode:'summary';readonly summary:SourceResidencySummary;readonly analysis:PlaybackSourceAnalysis;
 readonly complex?:never;
}
export type CompiledSourceProgram=FullCompiledSourceProgram|SummaryCompiledSourceProgram;
export interface SourceProgramInput extends MeshSource {
 readonly incidence?:'gpu-csr';
 /** Full remains the inspection/physical contract. Compiled examples retain
  * only the metadata their runtime consumers read after compilation. */
 readonly complexMode?:SourceComplexRetention;
 /** Explicit host opt-in; false/absent keeps ordinary source residency lean. */
 readonly contactCorners?:boolean;
 /** Part-graph lineage is not automatically source-incidence authority.
  * Preserve the original adapter's distinction while compiling together. */
 readonly partAdjacency?:MeshSource['adjacency'];
 /** Retained source metadata, not new incidence/classification authority. */
 readonly definition?:Pick<CompositionGeometry,'normals'|'lineage'|'surface'|'materialSource'|'construction'>;
 readonly physical?:Pick<CompositionGeometry,'lineage'|'surface'|'materialSource'|'construction'>;
}

/** Decode the resident part packet that was produced by
 * `compileSourcePartPrograms`. This is deliberately a view of the production
 * packet, not a second role selector: source ids, context tokens, roles and
 * packet ordinals all come from the compiler's actual records. */
function sourcePartRecords(table:Uint32Array|undefined):readonly SourcePartProgramRecord[]{
 if(!table||table[0]!==TOPOLOGY_PART_PACKET_MAGIC)return [];
 const offset=table[1]!,count=table[2]!;
 if(!Number.isSafeInteger(offset)||!Number.isSafeInteger(count)||offset<5||offset+count*TOPOLOGY_PART_PROGRAM_WORDS>table.length)return [];
 const records:SourcePartProgramRecord[]=[];
 for(let index=0;index<count;index++){
  const at=offset+index*TOPOLOGY_PART_PROGRAM_WORDS,flags=table[at+1]!,role=flags&3;
  if(role>3)continue;
  records.push({feature:table[at]!,role:role as 0|1|2|3,d5:(flags&0x100)!==0,direction:(flags&0x200)!==0?table[at+2]! :undefined,labels:table[at+3]!,sourceId:table[at+4]!,kind:table[at+5]===2?2:1});
 }
 return records;
}





interface SourcePartCanonicalPath {
 readonly feature:number;readonly sourceId:number;readonly kind:1|2;
 readonly closed:boolean;readonly complete:boolean;
 readonly samples:readonly {readonly a:number;readonly b:number;readonly t:number}[];
 readonly startPort:number;readonly endPort:number;
}
/** Index-aligned with `sourcePartRecords` (both are packet record order).
 * An absent or malformed side table yields no canonical paths, never a guess. */
function sourcePartCanonicalPaths(packet:Uint32Array|undefined):readonly (SourcePartCanonicalPath|undefined)[]{
 if(!packet||packet[0]!==TOPOLOGY_PART_PACKET_MAGIC)return [];
 const count=packet[2]!,side=packet[3]!+packet[4]!;
 if(!Number.isSafeInteger(side)||side+TOPOLOGY_PART_CANONICAL_HEADER_WORDS>packet.length||packet[side]!==TOPOLOGY_PART_CANONICAL_MAGIC||packet[side+1]!==TOPOLOGY_PART_CANONICAL_VERSION||
  packet[side+3]!==count||packet[side+6]!==TOPOLOGY_PART_CANONICAL_RECORD_WORDS||packet[side+7]!==TOPOLOGY_PART_CANONICAL_SAMPLE_WORDS)return [];
 const recordAt=packet[side+2]!,sampleAt=packet[side+4]!,sampleWords=packet[side+5]!,sampleEnd=sampleAt+sampleWords;
 if(recordAt<side+TOPOLOGY_PART_CANONICAL_HEADER_WORDS||recordAt+count*TOPOLOGY_PART_CANONICAL_RECORD_WORDS>packet.length||sampleAt<recordAt+count*TOPOLOGY_PART_CANONICAL_RECORD_WORDS||
  sampleWords%TOPOLOGY_PART_CANONICAL_SAMPLE_WORDS!==0||sampleEnd>packet.length)return [];
 const floats=new Float32Array(packet.buffer,packet.byteOffset,packet.length);
 return Array.from({length:count},(_,index):SourcePartCanonicalPath|undefined=>{
  const at=recordAt+index*TOPOLOGY_PART_CANONICAL_RECORD_WORDS,flags=packet[at+3]!,first=packet[at+4]!,n=packet[at+5]!;
  if(n===0||first<sampleAt||(first-sampleAt)%TOPOLOGY_PART_CANONICAL_SAMPLE_WORDS!==0||first+n*TOPOLOGY_PART_CANONICAL_SAMPLE_WORDS>sampleEnd)return undefined;
  const samples=Array.from({length:n},(_,k)=>{const word=first+k*TOPOLOGY_PART_CANONICAL_SAMPLE_WORDS;return {a:packet[word]!,b:packet[word+1]!,t:floats[word+2]!};});
  return {feature:packet[at]!,sourceId:packet[at+1]!,kind:packet[at+2]===2?2:1,closed:(flags&TOPOLOGY_PART_CANONICAL_FLAG_CLOSED)!==0,complete:(flags&TOPOLOGY_PART_CANONICAL_FLAG_COMPLETE)!==0,samples,startPort:packet[at+10]!,endPort:packet[at+11]!};
 });
}








// the section level along the graph axis (ties: `a`). "Nearer endpoint" put
// every sample of a cap section next to a fan apex on the apex (T13: section
// 6, 32 candidates, 0 distinct ports). The lower endpoint is the same side
// rule `sourceBandIncidence` uses for an upper section's band-side vertex.
const sampleVertex=(sample:PartSample,axis:readonly number[],positions:ArrayLike<number>):number=>{
 const along=(v:number)=>positions[v*3]!*axis[0]!+positions[v*3+1]!*axis[1]!+positions[v*3+2]!*axis[2]!;
 return (along(sample.b)<along(sample.a)?sample.b:sample.a)>>>0;
};


const sourceBandIncidenceOwner:SourceBandIncidence=sourceBandIncidence;


// source-program-compiler.ts validateWholeConstruction): a passage feature is a 1-BASED identity,
// so 0 stays the conservative/unknown value (the same convention as the resident entry ref). Both
// callers pass a 0-based ordinal (raw contact candidate index; plane region/loop ordinal), and the
// first required contact or the first plane opening used to serialize feature 0, which the worker
// validator rightly rejects. Out-of-range ordinals throw here; they are never wrapped by >>>0.
function openingTemplate(ordinal:number,ports:readonly number[],sourceContributors:readonly number[],options:{readonly geometry?:readonly (readonly [number,number,number])[];readonly closed?:boolean;readonly entryRef?:number;readonly context?:number;readonly role?:SourceWholePartRole}={}):SourceWholeOpeningTemplate|undefined{
 if(ports.length<2||sourceContributors.length===0)return;
 if(!Number.isSafeInteger(ordinal)||ordinal<0||ordinal>=0xffffffff)throw Error(`Source whole construction passage ordinal out of range: ${ordinal}`);
 return {feature:ordinal+1,ports:[...ports].map(value=>value>>>0),sourceContributors:[...sourceContributors].map(value=>value>>>0),...(options.geometry?{geometry:options.geometry}:{}),...(options.closed!==undefined?{closed:options.closed}:{}),...(options.entryRef!==undefined?{entryRef:options.entryRef}:{}),...(options.context!==undefined?{context:options.context}:{}),...(options.role!==undefined?{role:options.role}: {})};
}
/** The resident grammar stores its retained entries in this exact order. A
 * source parent carries an index into that resident table (1-based so zero
 * remains the conservative/unknown value), never the ordinal of its source
 * packet. The runtime table still resolves the posed tangent pair through its
 * context token; this index is the retained native identity used by the far
 * consumer and validation. */
/** Keep this admission predicate byte-for-byte aligned with
 * compiled-role-grammar.ts.  The parent's entryRef is an index into the
 * resident `partEntries` table, so omitting contact/branch/whole-circle
 * entries here shifts every later native reference even though the source
 * context itself is valid. */
const isResidentPartEntry=(entry:GrammarRoleEntry):boolean=>
 GRAMMAR_ROLES.some(role=>entry.role===role||entry.also_roles?.includes(role))||
 entry.role==='contact_junction'||entry.role==='branch_entry'||entry.role==='whole_circle';
const RESIDENT_PART_ENTRIES:readonly GrammarRoleEntry[]=[...ROLE_ENTRIES,...(roundSupplement.entries as unknown as readonly GrammarRoleEntry[])].filter(isResidentPartEntry);
const NATIVE_ROLE:readonly string[]=['rim_cap','shoulder','flank','return_base'];
const SOURCE_DIRECTIONS=['E','SE','S','SW','W','NW','N','NE'] as const;
const dot3=(a:readonly number[],b:readonly number[])=>a[0]!*b[0]!+a[1]!*b[1]!+a[2]!*b[2]!;
const cross3=(a:readonly number[],b:readonly number[])=>[a[1]!*b[2]!-a[2]!*b[1]!,a[2]!*b[0]!-a[0]!*b[2]!,a[0]!*b[1]!-a[1]!*b[0]!];
/** Return the actual source boundary of a band. Band records are not glyph
 * fragments: their faces and welded source vertices are the immutable split /
 * merge passage which the parent program must retain. A projected convex hull
 * removes only repeated interior source vertices; it never replaces the band
 * with a box or a camera sample. */
function bandSourceBoundary(record:SourcePartProgramRecord,graph:MeshPartGraph,complex:MeshSourceComplex,positions:Float32Array,incident?:ReadonlyMap<number,number>):{readonly points:readonly (readonly [number,number,number])[];readonly vertices:readonly number[]}|undefined{
 const band=graph.bands.find(value=>value.id===record.sourceId);if(!band)return;
 const representatives=new Set<number>();

 // of the INCIDENT SECTIONS' band-side vertices, as this producer declares it.
 // Hulling every band face vertex kept only interior band vertices on a real
 // closed Y (hull 20, labelled 0), so no passage could join two branches.
 if(incident)for(const vertex of incident.keys())representatives.add(vertex);
 else for(const face of band.faces)for(let k=0;k<3;k++){
  const vertex=complex.halfedges[face*3+k]?.from;if(vertex===undefined)continue;
  representatives.add(graph.vertexRepresentatives[vertex]??vertex);
 }
 const raw=[...representatives].map(vertex=>({vertex,point:[positions[vertex*3]!,positions[vertex*3+1]!,positions[vertex*3+2]!] as const}));
 if(raw.length<3)return;
 const axis=graph.axis,seed=Math.abs(axis[0]!)<.8?[1,0,0]:[0,0,1],basis=cross3(axis,seed),basisLength=Math.hypot(...basis);if(!(basisLength>0))return;
 const u=basis.map(value=>value/basisLength),v=cross3(axis,u);
 // The source vertex id rides with its hull point: a band passage's ports are

 const projected=raw.map(item=>({...item,x:dot3(item.point,u),y:dot3(item.point,v)})).sort((a,b)=>a.x-b.x||a.y-b.y||a.vertex-b.vertex);
 const turn=(a:typeof projected[number],b:typeof projected[number],c:typeof projected[number])=>(b.x-a.x)*(c.y-a.y)-(b.y-a.y)*(c.x-a.x);
 const lower:typeof projected=[];for(const point of projected){while(lower.length>=2&&turn(lower[lower.length-2]!,lower.at(-1)!,point)<=0)lower.pop();lower.push(point);}
 const upper:typeof projected=[];for(const point of [...projected].reverse()){while(upper.length>=2&&turn(upper[upper.length-2]!,upper.at(-1)!,point)<=0)upper.pop();upper.push(point);}
 const hull=[...lower.slice(0,-1),...upper.slice(0,-1)];return hull.length>=3?{points:hull.map(value=>value.point),vertices:hull.map(value=>value.vertex)}:undefined;
}
/** Context admission is deterministic source work. Try each cyclic anchor of
 * the same compiled contour so a valid retained entry is not lost merely
 * because a mesh exporter chose a different first edge. No tangent/glyph
 * search occurs: the contour, role and resident grammar still decide it. */
/** One admitted source passage: the exact emitted geometry, its resident
 * entry, and the two source vertex ports at the ends of THAT geometry. */
export interface SourcePartAdmission extends SourcePartContextGeometry {
 readonly entryRef:number;readonly ports:readonly [number,number];readonly closed:boolean;


 readonly span:'contour'|'passage'|'path';
}




















const BAND_HULL_ADMISSION=false as boolean;
export type SourcePartNonAdmissionReason='no-source'|'open-path-absent'|'open-path-incomplete'|'open-path-mismatch'|'band-unsupported'|'band-path-absent'|'band-path-incomplete'|'band-path-mismatch'|'no-candidate'|'no-resident-entry'|'collapsed-ports';


export type SourceBandIncidence=(graph:MeshPartGraph,band:MeshPartGraph['bands'][number],indices:ArrayLike<number>,positions:ArrayLike<number>)=>ReadonlyMap<number,number>;






























function sourcePartContextFor(record:SourcePartProgramRecord,graph:MeshPartGraph,complex:MeshSourceComplex,source:Pick<SourceProgramInput,'positions'|'indices'>,incidence?:SourceBandIncidence,canonical?:SourcePartCanonicalPath):SourcePartAdmission|{readonly reason:SourcePartNonAdmissionReason}{
 const positions=source.positions,rep=(vertex:number)=>(graph.vertexRepresentatives[vertex]??vertex)>>>0;
 let tried=false,resolved=false;
 const refuse=(reason:SourcePartNonAdmissionReason)=>({reason});
 const unadmitted=()=>refuse(!tried?'no-candidate':resolved?'collapsed-ports':'no-resident-entry');
 const admit=(candidate:SourcePartContextGeometry,ports:readonly [number,number]|undefined,closed:boolean,span:SourcePartAdmission['span']):SourcePartAdmission|undefined=>{
  tried=true;
  const entryRef=residentNativeEntryRef(record,candidate.context);if(entryRef===undefined)return;
  resolved=true;
  if(!ports||rep(ports[0])===rep(ports[1]))return;
  return {...candidate,entryRef,ports,closed,span};
 };
 if(record.kind===1){
  const section=graph.sections.find(item=>item.id===record.sourceId);
  if(!section||section.samples.length<3)return refuse('no-source');
  if(!section.closed){
   if(!canonical||canonical.kind!==1||canonical.sourceId!==record.sourceId||canonical.feature!==record.feature)return refuse('open-path-absent');
   if(canonical.closed||!canonical.complete||canonical.samples.length<3)return refuse('open-path-incomplete');
   const byEdge=new Map<string,number[]>();
   for(const [index,sample]of section.samples.entries()){const key=`${sample.a}/${sample.b}`;const list=byEdge.get(key);if(list)list.push(index);else byEdge.set(key,[index]);}
   const order:number[]=[];
   for(const sample of canonical.samples){
    const list=byEdge.get(`${sample.a}/${sample.b}`),at=list?.findIndex(index=>Math.fround(section.samples[index]!.t)===sample.t)??-1;
    if(!list||at<0)return refuse('open-path-mismatch');
    order.push(list.splice(at,1)[0]!);
   }
   const points=order.map(index=>partPoint({positions},section.samples[index]!));
   const path=sourcePartOpenPassage(graph.axis,record.role,{points,center:section.center});
   if(!path||path.from!==0||path.to!==points.length-1)return refuse('open-path-mismatch');
   const ports:[number,number]=[sampleVertex(section.samples[order[0]!]!,graph.axis,positions),sampleVertex(section.samples[order.at(-1)!]!,graph.axis,positions)];
   if(ports[0]!==canonical.startPort||ports[1]!==canonical.endPort)return refuse('open-path-mismatch');
   return admit(path,ports,false,'path')??unadmitted();
  }
  const base=section.samples.map(sample=>partPoint({positions},sample)),center=section.center;
  // Emitted points are references into `base`; recover their sample index.
  const portsOf=(points:readonly (readonly [number,number,number])[]):readonly [number,number]|undefined=>{
   const a=base.indexOf(points[0] as typeof base[number]),b=base.indexOf(points[points.length-1] as typeof base[number]);
   return a<0||b<0?undefined:[sampleVertex(section.samples[a]!,graph.axis,positions),sampleVertex(section.samples[b]!,graph.axis,positions)];
  };

  // classification (`sectionContext`) keeps the incumbent rule.
  for(const candidate of sourcePartContextCandidates(graph.axis,record.role,true,{points:base,center},'page')){
   const admitted=admit(candidate,portsOf(candidate.points),true,'contour');if(admitted)return admitted;
  }
  for(const passage of sourcePartPassageCandidates(graph.axis,record.role,{points:base,center})){
   const admitted=admit(passage,[sampleVertex(section.samples[passage.from]!,graph.axis,positions),sampleVertex(section.samples[passage.to]!,graph.axis,positions)],false,'passage');if(admitted)return admitted;
  }
  return unadmitted();
 }

 // is published by the topology producer from the exact source-edge passage
 // in the canonical side table; it is consumed only when that same packet
 // record is complete and still owns a turn role.
 if(record.role!==1||(record.labels&PART_FORK)===0)return refuse('band-unsupported');
 if(!canonical||canonical.kind!==2||canonical.sourceId!==record.sourceId||canonical.feature!==record.feature)return refuse('band-path-absent');
 if(canonical.closed||!canonical.complete||canonical.samples.length<3)return refuse('band-path-incomplete');
 const forkPoints=canonical.samples.map(sample=>partPoint({positions},sample));
 const center=forkPoints.reduce<[number,number,number]>((sum,point)=>[sum[0]+point[0]!/forkPoints.length,sum[1]+point[1]!/forkPoints.length,sum[2]+point[2]!/forkPoints.length],[0,0,0]);
 const forkPath=sourcePartOpenPassage(graph.axis,record.role,{points:forkPoints,center});
 if(!forkPath||forkPath.from!==0||forkPath.to!==forkPoints.length-1)return refuse('band-path-mismatch');
 const forkPorts:[number,number]=[canonical.startPort,canonical.endPort];
 return admit(forkPath,forkPorts,false,'path')??unadmitted();
}
/** Keep byte-for-byte aligned with compiled-role-grammar.ts `pathPorts` /
 * `pathContext` (the SOFT PART table playback resolves). An entry with a
 * region/tone/open port is not a path passage; referencing it here named an
 * identity the runtime table never resolves for that context. */
const PATH_PORTS=new Set(['paper','side','flank','shoulder','rim_cap','return_base','fold_crease','arch_band','contact_junction']);
const residentPathContext=(entry:GrammarRoleEntry):boolean=>{
 const ports=(entry as GrammarRoleEntry&{readonly ports?:Readonly<Record<string,string|null>>}).ports;
 return !!ports&&Object.values(ports).every(port=>port!==null&&PATH_PORTS.has(port));
};
/** Test/inspection view of the producer's native entry rule (1-based
 * resident index, undefined when the runtime table has no such context). */
export function sourcePartResidentEntry(role:0|1|2|3,context:SourcePartContextGeometry['context']):{readonly entryRef:number;readonly id:string}|undefined{
 const entryRef=residentNativeEntryRef({feature:1,role,d5:false,labels:0,sourceId:0,kind:1},context);
 return entryRef===undefined?undefined:{entryRef,id:RESIDENT_PART_ENTRIES[entryRef-1]!.id};
}









interface SourcePlanarPartAdmission extends SourcePartAdmission {
 readonly component:number;
 readonly passage:number;
 readonly role:0|1|2|3;
}
// Native packet features use the lower context namespace (at most context
// class 7 plus the packet ordinal). Keep planar component identities in a
// disjoint high namespace and retain enough source-component bits that a
// large multipart source does not alias two boundaries.
const SOURCE_PLANAR_FEATURE_BASE=0x07f00000;
const SOURCE_PLANAR_PASSAGE_BITS=6;
const SOURCE_PLANAR_PASSAGE_MASK=(1<<SOURCE_PLANAR_PASSAGE_BITS)-1;
const SOURCE_PLANAR_COMPONENT_MASK=(1<<(20-SOURCE_PLANAR_PASSAGE_BITS))-1;
const sourcePlanarFeature=(component:number,passage:number):number|undefined=>Number.isSafeInteger(component)&&component>=0&&component<=SOURCE_PLANAR_COMPONENT_MASK&&Number.isSafeInteger(passage)&&passage>=0&&passage<=SOURCE_PLANAR_PASSAGE_MASK?(SOURCE_PLANAR_FEATURE_BASE|(component<<SOURCE_PLANAR_PASSAGE_BITS)|passage)>>>0:undefined;
const sourcePlanarRoleCandidates=(labels:readonly number[]):readonly (0|2|3)[]=>{
 const hasRoot=labels.some(value=>(value&PART_ROOT)!==0),hasCap=labels.some(value=>(value&PART_CAP)!==0);
 if(hasRoot&&hasCap)return [];
 const primary=hasRoot?3:hasCap?0:2;
 // A flat panel with no free-end marker is a side/flank construction.  The
 // role is the incumbent graph default for an open section, not a new scene
 // recipe or a camera-derived classification.
 // A cap/return label describes the source-owned end family. Do not silently
 // reclassify an uncertifiable cap/return cycle as a flank: it retains its
 // incumbent producer until its own context is legal.
 // An unlabelled flat boundary has no source end marker, so flank remains the
 // first legal family.  If that family has no resident whole entry for the
 // boundary's actual plane normal, the fixed retained base family is the next
 // legal construction; this is still source-level admission, never a runtime
 // glyph search or a camera choice.
 return hasRoot||hasCap?[primary]:[primary,3];
};
const sourcePlanarOpenRoleCandidates=(labels:readonly number[]):readonly (0|1|2|3)[]=>{
 const hasRoot=labels.some(value=>(value&PART_ROOT)!==0),hasCap=labels.some(value=>(value&PART_CAP)!==0);
 if(hasRoot&&hasCap)return [];
 const primary:0|1|2|3=hasRoot?3:hasCap?0:2;
 // An open edge passage is a side unit, not a closed cap/return contour. Its
 // source-side context is still certified by sourcePartOpenPassage and the
 // resident entry; this fixed order does not search or fit a new role.
 return [...new Set<0|1|2|3>([primary,2,1])];
};
function compileSourcePlanarPartAdmissions(graph:MeshPartGraph,complex:MeshSourceComplex,positions:Float32Array,planarLevels?:PlanarLevels,boundaries?:readonly SourcePlanarBoundaryPassage[]):readonly SourcePlanarPartAdmission[]{
 const boundaryPassages=boundaries??compileSourcePlanarBoundaryPassages(complex,positions,planarLevels),labelsByFace=new Map<number,number[]>();
 for(const section of graph.sections)for(const face of section.faces){const list=labelsByFace.get(face);if(list)list.push(section.labels??0);else labelsByFace.set(face,[section.labels??0]);}
 const representative=(vertex:number)=>(graph.vertexRepresentatives[vertex]??vertex)>>>0,admissions:SourcePlanarPartAdmission[]=[];
 for(const boundary of boundaryPassages){
  const labels=boundary.faces.flatMap(face=>labelsByFace.get(face)??[]),roles=boundary.closed?sourcePlanarRoleCandidates(labels):sourcePlanarOpenRoleCandidates(labels);if(!roles.length)continue;
  const base=boundary.vertices.map(vertex=>[positions[vertex*3]!,positions[vertex*3+1]!,positions[vertex*3+2]!] as const);
  if(base.length<3||base.some(point=>point.some(value=>!Number.isFinite(value))))continue;
  const center=base.reduce<[number,number,number]>((sum,point)=>[sum[0]+point[0]!/base.length,sum[1]+point[1]!/base.length,sum[2]+point[2]!/base.length],[0,0,0]);
  // A flat panel's part graph axis can lie in the sheet plane (for example a
  // source with no closed cross-section), which makes every native tangent
  // candidate empty.  The planar boundary itself already carries the exact
  // source winding; derive its normal from those ordered samples and use it
  // only for this planar admission.  Degenerate loops retain the graph axis
  // and are therefore still conservative.
  const subtract=(a:readonly number[],b:readonly number[]):readonly [number,number,number]=>[a[0]!-b[0]!,a[1]!-b[1]!,a[2]!-b[2]!];
  const cross=(a:readonly number[],b:readonly number[]):readonly [number,number,number]=>[a[1]!*b[2]!-a[2]!*b[1]!,a[2]!*b[0]!-a[0]!*b[2]!,a[0]!*b[1]!-a[1]!*b[0]!];
  const normal=base.reduce<readonly [number,number,number]|undefined>((found,_,index)=>{if(found)return found;const n=cross(subtract(base[(index+1)%base.length]!,base[index]!),subtract(base[(index+2)%base.length]!,base[(index+1)%base.length]!));const length=Math.hypot(...n);return length>1e-8?[n[0]!/length,n[1]!/length,n[2]!/length] as const:undefined;},undefined)??graph.axis;
  let admitted=false;
  for(const role of roles){
   const candidates=boundary.closed?sourcePartContextCandidates(normal,role,true,{points:base,center},'page'):(()=>{const openCandidate=sourcePartOpenPassage(normal,role,{points:base,center});return openCandidate?[openCandidate]:[];})();
   for(const candidate of candidates){
   // Context candidates preserve the source samples but may rotate the closed
   // cycle; array identity is therefore not an endpoint contract. Resolve the
   // candidate's first/last samples by exact source coordinates and retain the
   // original vertex IDs for stable ports.
   const sourcePoint=(point:readonly number[])=>base.findIndex(item=>item.length===point.length&&item.every((value,index)=>value===point[index]));
   const first=sourcePoint(candidate.points[0]!),last=sourcePoint(candidate.points.at(-1)!);
   if(first<0||last<0)continue;
   const ports:[number,number]=[boundary.vertices[first]!,boundary.vertices[last]!];
   if(representative(ports[0])===representative(ports[1]))continue;
   const native=sourcePartResidentEntry(role,candidate.context);if(!native)continue;
   admissions.push({component:boundary.component,passage:boundary.passage,role,context:candidate.context,points:candidate.points,entryRef:native.entryRef,ports,closed:boundary.closed,span:boundary.closed?'contour':'passage'});
    admitted=true;break;
   }
   if(admitted)break;
  }
  if(admitted)continue;
 }
 return admissions;
}


















const RESIDENT_NATIVE_INDEX:ReadonlyMap<string,number>=(()=>{
 const index=new Map<string,number>();
 for(const [at,entry]of RESIDENT_PART_ENTRIES.entries()){
  if(!(residentPathContext(entry)&&entry.context.curvature==='convex'&&entry.context.paper.length>0))continue;
  const key=`${entry.role}|${entry.size_band.band}|${entry.context.inside}`;
  if(!index.has(key))index.set(key,at);
 }
 return index;
})();
function residentNativeEntryRef(record:SourcePartProgramRecord,context:SourcePartContextGeometry['context']):number|undefined{
 const role=NATIVE_ROLE[record.role],inside=context.inside;if(!role||!inside)return;
 // Pick the first retained entry of this family in the fixed size-band order:
 // a real resident grammar identity, not a packet-local ordinal. The posed
 // tangent pair selects the drawn entry at runtime from the same family.
 for(const band of ['s','m','l']){
  const index=RESIDENT_NATIVE_INDEX.get(`${role}|${band}|${inside}`);
  if(index!==undefined)return index+1;
 }
 return undefined;
}
/** Per-record admission report, before the four-role whole-template gate.
 * The worker and baked routes both call this through
 * compileWholeConstructionTemplate; tests read it to see WHICH roles exist. */
export interface SourcePartPassageReport {
 readonly parts:readonly (SourcePartAdmission&{readonly record:SourcePartProgramRecord})[];


 readonly unadmitted:readonly {readonly record:SourcePartProgramRecord;readonly reason:SourcePartNonAdmissionReason}[];
 readonly declaredRoles:readonly SourceWholePartRole[];
 readonly admittedRoles:readonly SourceWholePartRole[];
 /** Source-owned fork/junction graph side table. This is metadata for the
  * junction consumer; it does not admit a native role or alter the incumbent
  * canonical path. */
 readonly junctions:readonly SourcePartJunctionRecord[];
}
const WHOLE_ROLE:readonly SourceWholePartRole[]=['cap','turn','flank','return'];
export function compileSourcePartPassages(source:Pick<SourceProgramInput,'positions'|'indices'>,analysis:SourceAnalysis,complex:MeshSourceComplex,incidence:SourceBandIncidence|undefined=sourceBandIncidenceOwner):SourcePartPassageReport|undefined{
 const graph=analysis.structure;if(!graph)return;
 const packet=analysis.runProgram?sourcePartProgramPacket(analysis.runProgram.topology):compileSourcePartPrograms({structure:graph,complex,sourcePositions:source.positions});
 const junctions=sourcePartJunctions(packet);
 // Records and canonical paths are both in packet record order.
 const canonical=sourcePartCanonicalPaths(packet);
 const entries=sourcePartRecords(packet).map((record,index)=>({record,canonical:canonical[index]})).filter(entry=>entry.record.feature!==0);if(!entries.length)return;
 const parts:(SourcePartAdmission&{readonly record:SourcePartProgramRecord})[]=[];
 const unadmitted:{readonly record:SourcePartProgramRecord;readonly reason:SourcePartNonAdmissionReason}[]=[];
 for(const {record,canonical:path}of entries){
  const result=sourcePartContextFor(record,graph,complex,source,incidence,path);
  if('reason' in result)unadmitted.push({record,reason:result.reason});else parts.push({...result,record});
 }
 const ordered=(roles:Iterable<number>)=>WHOLE_ROLE.filter((_,index)=>new Set(roles).has(index));
 return {parts,unadmitted,declaredRoles:ordered(entries.map(entry=>entry.record.role)),admittedRoles:ordered(parts.map(part=>part.record.role)),junctions};
}
/** Build the source-owned parent representation from the same immutable
 * structure/complex that feeds the resident topology. A legacy source lacking
 * one of the four connected passages, valid context, or stable source ports
 * stays unknown; the one bounded planar compound exception uses its complete
 * outer passages plus explicit opening/contact products instead. No fake role
 * or box is emitted. */









export function compileWholeTemplateContacts(complex:MeshSourceComplex,positions?:ArrayLike<number>,relation?:{readonly entryRef?:number;readonly context?:number;readonly role?:SourceWholePartRole}):SourceWholeOpeningTemplate[]{
 const contacts:SourceWholeOpeningTemplate[]=[];
 const requiredContacts=sourceContactRequirements(complex);
 for(const [contactIndex,contact]of complex.contacts.entries()){
  if(!requiredContacts[contactIndex])continue;
  // Each member is a source halfedge address.  Preserve the complete ordered
  // seam, including every bend, instead of reducing a curved contact to the
  // first endpoint of its first edge and the last endpoint of its last edge.
  // `forward` is the contact's source orientation; the halfedge itself keeps
  // the original indexed face orientation, so invert it when required.
  const point=(vertex:number):readonly [number,number,number]|undefined=>positions&&vertex*3+2<positions.length?[positions[vertex*3]!,positions[vertex*3+1]!,positions[vertex*3+2]!] as const:undefined;
  const vertices:number[]=[];
  for(const member of contact.members){
   const halfedge=complex.halfedges[member.edge];if(!halfedge)continue;
   const ordered=member.forward?[halfedge.from,halfedge.to]:[halfedge.to,halfedge.from];
   if(vertices.at(-1)!==ordered[0])vertices.push(ordered[0]!);vertices.push(ordered[1]!);
  }
  const sampled=vertices.map(point);
  const points=sampled.every((value):value is readonly [number,number,number]=>!!value)?sampled:[];
  // A metadata-only caller may omit positions (the relationship census uses
  // this path). Preserve the candidate identity there, but only attach native
  // geometry when the source coordinates were actually supplied; the parent
  // renderer remains conservative until that geometry is present.
  const opening=vertices.length>=2?openingTemplate(contactIndex,[vertices[0]!,vertices.at(-1)!],contact.members.map(member=>member.edge),{...(points.length===vertices.length?{geometry:points}:{}),closed:false,...relation}):undefined;
  if(opening)contacts.push(opening);
 }
 return contacts;
}
export function compileWholeConstructionTemplate(source:SourceProgramInput,analysis:SourceAnalysis,complex:MeshSourceComplex,planarLevels?:PlanarLevels):SourceWholeConstructionTemplate|undefined{
 const graph=analysis.structure;if(!graph)return;
 const report=compileSourcePartPassages(source,analysis,complex);if(!report)return;
 // A planar multipart source has no closed cross-section sections: its
 // cap/flank/return units come from the complete original boundary cycles
 // produced by compilePlanarLevels. Keep the existing section/band producer
 // as the turn source and require the same native context predicate for every
 // planar unit. The whole template is allowed to replace children only when
 // every source-owned planar passage has an admitted replacement; otherwise
 // the caller keeps the incumbent source path instead of hiding an omitted
 // component behind a valid four-role count.
 const planarPassages=compileSourcePlanarBoundaryPassages(complex,source.positions,planarLevels);
 const planar=compileSourcePlanarPartAdmissions(graph,complex,source.positions,planarLevels,planarPassages);
 const coveredPlanar=new Set(planar.map(part=>`${part.component}:${part.passage}`));
 if(planarPassages.some(passage=>!coveredPlanar.has(`${passage.component}:${passage.passage}`)))return;
 const admittedParts:(SourcePartAdmission&{readonly sourceId:number;readonly feature:number;readonly role:0|1|2|3})[]=[
  ...report.parts.map(part=>({...part,sourceId:part.record.sourceId,feature:part.record.feature,role:part.record.role})),
  ...planar.flatMap(part=>{const feature=sourcePlanarFeature(part.component,part.passage);return feature===undefined?[]:[{...part,sourceId:part.component,feature,role:part.role}]}),
 ];
 const admittedRoleSet=new Set(admittedParts.map(part=>WHOLE_ROLE[part.role]!));
 const completeLegacyRoles=WHOLE_ROLE.every(role=>admittedRoleSet.has(role));
 // Emit exactly the admitted geometry: its anchor/winding (or passage ends)
 // produced the tangent context and the two ports, so the runtime
 // correspondence, the resident entry and the ports cannot disagree. The
 // far consumer copies context into the played
 // run, so it is the ADMITTED candidate's token. The packet's record token is
 // its FIRST legal closed candidate, which for a flank may be the other side

 const parts:SourceWholePartTemplate[]=admittedParts.map(part=>({feature:part.feature,entryRef:part.entryRef,
  context:part.context.token,role:WHOLE_ROLE[part.role]!,
  ports:part.ports,sourceContributors:[part.sourceId],geometry:part.points,closed:part.closed}));
 // Relations may borrow the turn *family* only when the compiler actually
 // admitted a turn part.  There is no fallback to an arbitrary first part:
 // missing native relation construction keeps the children as the incumbent.
 // A planar compound may legitimately expose only the native flank/return
 // family for its outer sheet.  Prefer the old turn family when present, but
 // otherwise use the first admitted outer passage rather than fabricating an
 // absent role.  The product validator below keeps this fallback confined to
 // the explicit planar-opening/contact class.
 const relationSeed=parts.find(part=>part.role==='turn')??parts[0];
 if(!relationSeed)return;
 // Relation boundaries use the turn/shoulder native family.  Its context class
 // is derived from the ordered boundary when the loop is a valid convex ring;
 // an open contact keeps the fixed below-side turn class.  Direction bits are
 // deliberately not baked here: the GPU consumer recomputes them from these
 // same posed source samples, while retaining this source-owned class.
 const relationAxis=(geometry:readonly (readonly [number,number,number])[]):readonly [number,number,number]=>{
  for(let index=0;index+2<geometry.length;index++){
   const a=geometry[index]!,b=geometry[index+1]!,c=geometry[index+2]!,u=[b[0]!-a[0]!,b[1]!-a[1]!,b[2]!-a[2]!],v=[c[0]!-b[0]!,c[1]!-b[1]!,c[2]!-b[2]!],n=[u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]],length=Math.hypot(...n);if(length>1e-8)return [n[0]/length,n[1]/length,n[2]/length];
  }
  return analysis.structure!.axis as readonly [number,number,number];
 };
 const relationContext=(geometry:readonly (readonly [number,number,number])[],closed:boolean):number=>{
  const center=geometry.reduce((sum,point)=>[sum[0]+point[0],sum[1]+point[1],sum[2]+point[2]] as [number,number,number],[0,0,0] as [number,number,number]).map(value=>value/Math.max(geometry.length,1)) as [number,number,number];
  const relationRole=WHOLE_ROLE.indexOf(relationSeed.role) as 0|1|2|3;
  const candidate=closed?sourcePartContextCandidates(relationAxis(geometry),relationRole,true,{points:geometry,center},'page')[0]?.context.token:undefined;
  return candidate??(2<<6);
 };
 const relation={entryRef:relationSeed.entryRef,role:relationSeed.role};
 const openings:SourceWholeOpeningTemplate[]=[];
 for(const region of analysis.planeRegions??[]){
  for(const [loopIndex,loop]of (region.openings??[]).entries()){
   const first=complex.halfedges[loop[0]!],last=complex.halfedges[loop[loop.length-1]!];
   const geometry=loop.map(halfedge=>complex.halfedges[halfedge]!.from).map(vertex=>[source.positions[vertex*3]!,source.positions[vertex*3+1]!,source.positions[vertex*3+2]!] as const);
   const opening=first&&last?openingTemplate(region.id*1024+loopIndex,[first.from,last.to],[region.id],{geometry,closed:true,context:relationContext(geometry,true),...relation}):undefined;
   if(opening)openings.push(opening);
  }
 }
 const contacts=compileWholeTemplateContacts(complex,source.positions,{...relation,context:2<<6});
 // Plane openings are both a visible aperture and protected paper. Keeping
 // the two lists separate lets the native consumer distinguish their roles.
 // Publish one bounded compound product only when the same source compile
 // supplied every outer planar passage, every opening loop and at least one
 // required contact with stable endpoint anchors.  The feature lists are the
 // exact source-program identities consumed by the selector/product cache;
 // unsupported relation classes remain complete child constructions.
 const outerFeatures=planar.flatMap(part=>{const feature=sourcePlanarFeature(part.component,part.passage);return feature===undefined?[]:[feature];});
 const openingFeatures=openings.map(item=>item.feature),contactFeatures=contacts.map(item=>item.feature),anchorVertices=contacts.flatMap(item=>item.ports);
 const compound=outerFeatures.length&&openingFeatures.length&&contactFeatures.length&&anchorVertices.length?{
  kind:'planar-openings-contact' as const,
  outerFeatures:[...new Set(outerFeatures)],
  openingFeatures:[...new Set(openingFeatures)],
  contactFeatures:[...new Set(contactFeatures)],
  anchorVertices:[...new Set(anchorVertices)],
 }:undefined;
 if(!completeLegacyRoles&&!compound)return;
 return {revision:source.revision,parts,openings,contacts,protectedVoids:openings.slice(),paper:openings.slice(),...(compound?{compound}:{})};
}
export function* sourceProgramWork(source:SourceProgramInput):Generator<void,CompiledSourceProgram>{
 if(source.complexMode==='summary'&&source.incidence!=='gpu-csr')throw Error('Summary source requires compiled GPU incidence');
 const construction=(source.physical??source.definition)?.construction;
 if(construction)validateSourceConstruction(construction,source.indices.length,source.positions.length/3);
 // Build the immutable source-corner index in this worker only when the host
 // has opted this mesh into solver-corner attachment. Posing/contact ticks
 // never rebuild or scan it. An empty source is recorded as `empty` below.
 const cornerIndex=source.contactCorners===true?compileSourceCornerIndex(source.revision,source.positions,source.indices):undefined;
 const cornerIndexState=source.contactCorners===true?(cornerIndex?'ready':'empty'):undefined;yield;
 const complex=compileMeshSource(source);yield;
 const materialClasses=compileSourceFaceMaterialClasses(source.physical??source.definition??{},source.indices.length/3);
 yield;
 const physical=source.physical?physicalSource(source):undefined;yield;
 const analysis=compileSourceAnalysis(physical?.geometry??{...source,owner:source.id,...source.definition},materialClasses,{weld:complex.weld,retention:source.complexMode??'full'});yield;
 // Profile adequacy is source geometry, not a camera/render decision. Terrain
 // deliberately stays conservative; every other class gets one worker-side
 // verdict carried through both compiled residency contracts.
 const profileSafe=analysis.type==='terrain'?false:compileMeshProfileSafe(source.positions,source.indices);yield;
 if(analysis.type!=='terrain'){
  try{analysis.structure=yield* meshPartWork({...source,adjacency:source.partAdjacency??source.adjacency},complex.weld);}
  catch(error){analysis.structureError=error instanceof Error?error.message:String(error);}
 }
 const runProgram=compileSourceRunProgram({...analysis,complex,sourcePositions:source.positions});analysis.runProgram=runProgram;yield;
 // appendPartProgramTable embeds the complete packet before its membership
 // range. Reuse that packet instead of recompiling all source part membership.
 const partPacket=sourcePartProgramPacket(runProgram.topology);
 const partAnchorVertices=partPacket?sourcePartAnchorVertices(partPacket,source.indices):undefined;
 // Planar levels are one immutable per-source result. The whole-source
 // producer and the residency packet consume this same result; neither path
 // reruns the expensive boundary/contact scan.
 const planarLevels=analysis.type==='terrain'||analysis.type==='rounded'||analysis.type==='paper'?undefined:compilePlanarLevels(complex,source.positions);
 const wholeConstruction=compileWholeConstructionTemplate(source,analysis,complex,planarLevels);yield;

 // Its relationship counts come from this same compilation: plane-region
 // openings and source contacts, so eligibility can refuse a generic outline.

 // seams of separately indexed faces are not depicted contacts.
 const smallParent=analysis.type==='terrain'?undefined:compileSourceSmallParent(analysis.structure,source.positions,{
  openings:(analysis.planeRegions??[]).reduce((sum,region)=>sum+(region.openings?.length??0),0),...compileSourceRelationships(complex)});

 // from its own edge classes, carried in the source packet the relation
 // stage already binds. A terrain source and the legacy packet keep all.


 // the same level table (SOURCE_PATH section); the JSON copy stays as well.
 const edgeLevels=compileConstructionLevels(runProgram.topology,complex,source.positions,analysis.type,construction,planarLevels);
 const packet:CompiledSourceProgram['packet']=source.incidence==='gpu-csr'?{kind:'gpu-csr-v1',plan:compileSourceIncidencePlan(source,complex,{materialClasses,edgeLevels})}:{kind:'legacy-v2',words:packMeshSource(source,complex,{materialClasses})};yield;
 const geometry=physical?.geometry??{owner:source.id,revision:source.revision,positions:source.positions,indices:source.indices,contactKeys:source.contactKeys,...source.definition};
 // Physical vertex inputs are rendering data, not CPU topology inspection.
 // Compile with the complete complex above, then omit only its expanded CPU
 // transfer columns. Incidence, classifications and run programs stay intact.
 if(source.complexMode==='summary'){
  const {type,styleWord,foliage,organic,toneRegions,runProgram}=analysis;
  return {complexMode:'summary',summary:{schema:'shallot.source-residency-summary.v1',edgeCount:complex.edges.length},packet,
   analysis:{type,styleWord,foliage,organic,toneRegions,runProgram},geometry,...(physical?{physical}:{}),...(cornerIndex?{cornerIndex}:{}),...(cornerIndexState?{cornerIndexState}:{}),...(wholeConstruction?{wholeConstruction}:{}),...(partAnchorVertices?{partAnchorVertices}:{}),...(smallParent?{smallParent}:{}),profileSafe};
 }
 return {complexMode:'full',complex,packet,analysis,geometry,physical,...(cornerIndex?{cornerIndex}:{}),...(cornerIndexState?{cornerIndexState}:{}),...(wholeConstruction?{wholeConstruction}:{}),...(partAnchorVertices?{partAnchorVertices}:{}),...(smallParent?{smallParent}:{}),profileSafe};
}

/** Static physical vertex inputs, compiled once with their incidence. */
export interface CompiledPhysicalSource {
 readonly geometry:CompositionGeometry;
 readonly positions:Float32Array;
 readonly bounds:Float32Array;
}
function physicalSource(source:SourceProgramInput):CompiledPhysicalSource{
 const geometry:CompositionGeometry={owner:source.id,revision:source.revision,positions:source.positions,indices:source.indices,...source.physical};
 if(geometry.lineage&&[geometry.lineage.faces,geometry.lineage.splits,geometry.lineage.materials].some(values=>values&&values.length!==source.indices.length/3))throw Error('Physical source lineage does not cover its complete triangle extent');
 const normals=sourceNormals(geometry),positions=new Float32Array(source.positions.length/3*4);
 const bounds=new Float32Array([Infinity,Infinity,Infinity,0,-Infinity,-Infinity,-Infinity,0]);
 for(let i=0;i<source.positions.length/3;i++){
  positions.set(source.positions.subarray(i*3,i*3+3),i*4);positions[i*4+3]=1;
  for(let k=0;k<3;k++){bounds[k]=Math.min(bounds[k]!,source.positions[i*3+k]!);bounds[k+4]=Math.max(bounds[k+4]!,source.positions[i*3+k]!);}
 }
 return {geometry:{...geometry,normals},positions,bounds};
}
/** Estimate normals inside original indexed incidence or published P0 feature
 * incidence. Equal positions are only a CORNER correspondence within an
 * already related face pair. They never weld vertices or create adjacency. */
function sourceNormals(g:CompositionGeometry):Float32Array{
 const vertices=g.positions.length/3,faces=g.indices.length/3,result=new Float32Array(vertices*4),edges=new Map<string,number[]>();
 const cross=new Float64Array(faces*3),unit=new Float64Array(faces*3),incident:number[][]=Array.from({length:vertices},()=>[]),neighbors:Set<number>[]=Array.from({length:faces},()=>new Set());
 const same=(a:number,b:number)=>[0,1,2].every(k=>g.positions[a*3+k]===g.positions[b*3+k]);
 for(let f=0;f<faces;f++){
  const [a,b,c]=Array.from(g.indices.subarray(f*3,f*3+3)),p=g.positions;
  const u=[0,1,2].map(k=>p[b!*3+k]!-p[a!*3+k]!),v=[0,1,2].map(k=>p[c!*3+k]!-p[a!*3+k]!);
  const n=[u[1]!*v[2]!-u[2]!*v[1]!,u[2]!*v[0]!-u[0]!*v[2]!,u[0]!*v[1]!-u[1]!*v[0]!],length=Math.hypot(...n);
  for(let k=0;k<3;k++){
   cross[f*3+k]=n[k]!;unit[f*3+k]=n[k]!/(length||1);incident[g.indices[f*3+k]!]!.push(f);
   const a=g.indices[f*3+k]!,b=g.indices[f*3+(k+1)%3]!,key=`${Math.min(a,b)}/${Math.max(a,b)}`,members=edges.get(key)??[];members.push(f);edges.set(key,members);
  }
 }
 const connect=(members:readonly number[])=>{for(const a of members)for(const b of members)if(a!==b)neighbors[a]!.add(b);};
 // Shared indexed edges are actual source incidence. P0 triangle lists have
 // separate indices and supply the imported feature→face relation instead.
 for(const list of edges.values())connect(list);
 const adjacency=g.lineage?.adjacency;
 if(adjacency){
  if(!adjacency.offsets.length||adjacency.offsets[0]!==0||adjacency.offsets.at(-1)!==adjacency.faces.length)throw Error('Invalid physical source adjacency extent');
  for(let edge=0;edge+1<adjacency.offsets.length;edge++){
   const first=adjacency.offsets[edge]!,last=adjacency.offsets[edge+1]!;if(last<first||last>adjacency.faces.length)throw Error('Invalid physical source incidence range');
   const members=Array.from(adjacency.faces.subarray(first,last));if(members.some(f=>f>=faces))throw Error('Physical source adjacency names an absent face');connect(members);
  }
 }
 const dot=(a:number,b:number)=>[0,1,2].reduce((n,k)=>n+unit[a*3+k]!*unit[b*3+k]!,0),compatible=(a:number,b:number)=>dot(a,b)>=Math.SQRT1_2&&(!g.lineage?.materials||g.lineage.materials[a]===g.lineage.materials[b]);
 for(let vertex=0;vertex<vertices;vertex++){
  const seed=incident[vertex]![0];if(seed===undefined)continue;
  const visited=new Set<number>([seed]),queue=[seed],normal=[0,0,0];
  for(let at=0;at<queue.length;at++){
   const face=queue[at]!;for(let k=0;k<3;k++)normal[k]!+=cross[face*3+k]!;
   for(const next of neighbors[face]!)if(!visited.has(next)&&compatible(face,next)&&compatible(seed,next)&&[0,1,2].some(k=>same(vertex,g.indices[next*3+k]!))){visited.add(next);queue.push(next);}
  }
  // One shared index cannot express two authored normal lobes. Keep its
  // seed face normal in that case, rather than smooth a crease by position.
  if(incident[vertex]!.some(f=>!compatible(seed,f)))for(let k=0;k<3;k++)normal[k]=cross[seed*3+k]!;
  const length=Math.hypot(...normal);for(let k=0;k<3;k++)result[vertex*4+k]=normal[k]!/(length||1);
 }
 return result;
}

// The public asset build uses the SAME physical normal rules as dynamic
// registration; no topology/program archive or client-side recompile is needed
// merely to recover these immutable source attributes.
export {sourceNormals as compilePhysicalSourceNormals};


