import {compileSourceFaceMaterialClasses} from './source-material.ts';
import {compileMeshSource,packMeshSource,type MeshSource,type MeshSourceComplex} from './source-complex.ts';
import {compileSourceAnalysis,type SourceAnalysis,type CompositionGeometry} from './composition.ts';
import {meshPartWork,partPoint,PART_FORK,type MeshPartGraph,type PartSample,type PartSection} from './mesh-part-graph.ts';
import {compileSourceRunProgram,compileConstructionLevels,compileMeshProfileSafe,compileSourcePartPrograms,sourcePartProgramPacket,sourcePartContextCandidates,sourcePartPassageCandidates,sourcePartAnchorVertices,type SourcePartContextGeometry,type SourcePartPassageAnchor,type SourcePartProgramRecord,TOPOLOGY_PART_PACKET_MAGIC,TOPOLOGY_PART_PROGRAM_WORDS} from './compiled-construction-topology.ts';
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








// the section level along the graph axis (ties: `a`). "Nearer endpoint" put
// every sample of a cap section next to a fan apex on the apex (T13: section
// 6, 32 candidates, 0 distinct ports). The lower endpoint is the same side
// rule `sourceBandIncidence` uses for an upper section's band-side vertex.
const sampleVertex=(sample:PartSample,axis:readonly number[],positions:ArrayLike<number>):number=>{
 const along=(v:number)=>positions[v*3]!*axis[0]!+positions[v*3+1]!*axis[1]!+positions[v*3+2]!*axis[2]!;
 return (along(sample.b)<along(sample.a)?sample.b:sample.a)>>>0;
};


const sourceBandIncidenceOwner:SourceBandIncidence=sourceBandIncidence;

function openingTemplate(feature:number,ports:readonly number[],sourceContributors:readonly number[]):SourceWholeOpeningTemplate|undefined{
 if(ports.length<2||sourceContributors.length===0)return;
 return {feature:feature>>>0,ports:[...ports].map(value=>value>>>0),sourceContributors:[...sourceContributors].map(value=>value>>>0)};
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
 /** 'contour' = the whole closed source loop; 'passage' = an anchor-bounded arc. */
 readonly span:'contour'|'passage';
}


export type SourceBandIncidence=(graph:MeshPartGraph,band:MeshPartGraph['bands'][number],indices:ArrayLike<number>,positions:ArrayLike<number>)=>ReadonlyMap<number,number>;
















function sourcePartContextFor(record:SourcePartProgramRecord,graph:MeshPartGraph,complex:MeshSourceComplex,source:Pick<SourceProgramInput,'positions'|'indices'>,incidence?:SourceBandIncidence):SourcePartAdmission|undefined{
 const positions=source.positions,rep=(vertex:number)=>(graph.vertexRepresentatives[vertex]??vertex)>>>0;
 const admit=(candidate:SourcePartContextGeometry,ports:readonly [number,number]|undefined,closed:boolean,span:SourcePartAdmission['span']):SourcePartAdmission|undefined=>{
  if(!ports||rep(ports[0])===rep(ports[1]))return;
  const entryRef=residentNativeEntryRef(record,candidate.context);if(entryRef===undefined)return;
  return {...candidate,entryRef,ports,closed,span};
 };
 if(record.kind===1){
  const section=graph.sections.find(item=>item.id===record.sourceId);
  if(!section?.closed||section.samples.length<3)return;
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
  return;
 }
 const forkTurn=record.role===1&&(record.labels&PART_FORK)!==0;if(!forkTurn||!incidence)return;
 const band=graph.bands.find(item=>item.id===record.sourceId);if(!band||!band.lower.length||!band.upper.length)return;
 const labels=incidence(graph,band,source.indices,positions);
 const hull=bandSourceBoundary(record,graph,complex,positions,labels);if(!hull)return;
 // Section ids may be 0; anchor label 0 means "no anchor", so offset by one.
 const anchors:SourcePartPassageAnchor[]=hull.vertices.flatMap((vertex,index)=>{const section=labels.get(vertex);return section===undefined?[]:[{index,label:section+1}];});
 if(new Set(anchors.map(anchor=>anchor.label)).size<2)return;
 const center=hull.points.reduce<[number,number,number]>((sum,point)=>sum.map((item,index)=>item+point[index]!/hull.points.length) as [number,number,number],[0,0,0]);
 for(const passage of sourcePartPassageCandidates(graph.axis,record.role,{points:hull.points,center},anchors)){
  const admitted=admit(passage,[hull.vertices[passage.from]!>>>0,hull.vertices[passage.to]!>>>0],false,'passage');if(admitted)return admitted;
 }
 return;
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
function residentNativeEntryRef(record:SourcePartProgramRecord,context:SourcePartContextGeometry['context']):number|undefined{
 const role=NATIVE_ROLE[record.role],inside=context.inside;if(!role||!inside)return;
 const incoming=SOURCE_DIRECTIONS[context.incoming],outgoing=SOURCE_DIRECTIONS[context.outgoing];
 // The source context is intrinsic and convex. Pick the first retained entry
 // in the fixed size-band order; it is a real resident grammar identity, not a
 // packet-local source-part ordinal. Runtime LOD may resolve the corresponding
 // retained band from the same context table.
 for(const band of ['s','m','l']){
  const index=RESIDENT_PART_ENTRIES.findIndex(entry=>entry.role===role&&entry.size_band.band===band&&residentPathContext(entry)&&entry.context.curvature==='convex'&&entry.context.inside===inside&&entry.context.tangent_in===incoming&&entry.context.tangent_out===outgoing&&entry.context.paper.length>0);
  if(index>=0)return index+1;
 }
 return undefined;
}
/** Per-record admission report, before the four-role whole-template gate.
 * The worker and baked routes both call this through
 * compileWholeConstructionTemplate; tests read it to see WHICH roles exist. */
export interface SourcePartPassageReport {
 readonly parts:readonly (SourcePartAdmission&{readonly record:SourcePartProgramRecord})[];
 readonly declaredRoles:readonly SourceWholePartRole[];
 readonly admittedRoles:readonly SourceWholePartRole[];
}
const WHOLE_ROLE:readonly SourceWholePartRole[]=['cap','turn','flank','return'];
export function compileSourcePartPassages(source:Pick<SourceProgramInput,'positions'|'indices'>,analysis:SourceAnalysis,complex:MeshSourceComplex,incidence:SourceBandIncidence|undefined=sourceBandIncidenceOwner):SourcePartPassageReport|undefined{
 const graph=analysis.structure;if(!graph)return;
 const packet=analysis.runProgram?sourcePartProgramPacket(analysis.runProgram.topology):compileSourcePartPrograms({structure:graph,complex,sourcePositions:source.positions});
 const records=sourcePartRecords(packet).filter(record=>record.feature!==0);if(!records.length)return;
 const parts:(SourcePartAdmission&{readonly record:SourcePartProgramRecord})[]=[];
 for(const record of records){const admitted=sourcePartContextFor(record,graph,complex,source,incidence);if(admitted)parts.push({...admitted,record});}
 const ordered=(roles:Iterable<number>)=>WHOLE_ROLE.filter((_,index)=>new Set(roles).has(index));
 return {parts,declaredRoles:ordered(records.map(record=>record.role)),admittedRoles:ordered(parts.map(part=>part.record.role))};
}
/** Build the source-owned parent representation from the same immutable
 * structure/complex that feeds the resident topology. A source lacking one
 * of the four connected passages, valid context, or stable source ports stays
 * unknown and therefore retains the incumbent path; no fake role or box is
 * emitted. */








export function compileWholeTemplateContacts(complex:MeshSourceComplex):SourceWholeOpeningTemplate[]{
 const contacts:SourceWholeOpeningTemplate[]=[];
 const requiredContacts=sourceContactRequirements(complex);
 for(const [contactIndex,contact]of complex.contacts.entries()){
  if(!requiredContacts[contactIndex])continue;
  const first=contact.members[0],last=contact.members[contact.members.length-1];

  // index address `SourceEdge.id`), not an index into the compact `edges`
  // array. Resolve halfedge -> edge index first; indexing `edges` directly
  // dropped every required contact whose halfedge id exceeded the edge count
  // (shared-index source: 18 required contacts became 5). The contributors
  // keep the member halfedge ids unchanged (source identity).
  const edgeOf=(halfedge:number|undefined)=>{const edge=halfedge===undefined?undefined:complex.halfedges[halfedge]?.edge;return edge===undefined?undefined:complex.edges[edge];};
  const a=edgeOf(first?.edge)?.vertices,b=edgeOf(last?.edge)?.vertices;
  const opening=a&&b?openingTemplate(contactIndex,[a[0]!,b[1]!],contact.members.map(member=>member.edge)):undefined;
  if(opening)contacts.push(opening);
 }
 return contacts;
}
export function compileWholeConstructionTemplate(source:SourceProgramInput,analysis:SourceAnalysis,complex:MeshSourceComplex):SourceWholeConstructionTemplate|undefined{
 const graph=analysis.structure;if(!graph)return;
 const packet=analysis.runProgram?sourcePartProgramPacket(analysis.runProgram.topology):compileSourcePartPrograms({structure:graph,complex,sourcePositions:source.positions});
 // This producer requires all four roles after native admission. A missing
 // declared role cannot become present through context selection; reject the
 // same unsupported template before contour/passage work.
 if(new Set(sourcePartRecords(packet).filter(record=>record.feature!==0).map(record=>record.role)).size<4)return;
 const report=compileSourcePartPassages(source,analysis,complex);if(!report)return;
 if(WHOLE_ROLE.some(role=>!report.admittedRoles.includes(role)))return;
 // Emit exactly the admitted geometry: its anchor/winding (or passage ends)
 // produced the tangent context and the two ports, so the runtime
 // correspondence, the resident entry and the ports cannot disagree. The
 // far consumer (gpu-source-far-forms.ts) copies `context` into the played
 // run, so it is the ADMITTED candidate's token. The packet's record token is
 // its FIRST legal closed candidate, which for a flank may be the other side

 const parts:SourceWholePartTemplate[]=report.parts.map(part=>({feature:part.record.feature,entryRef:part.entryRef,
  context:part.context.token,role:WHOLE_ROLE[part.record.role]!,
  ports:part.ports,sourceContributors:[part.record.sourceId],geometry:part.points,closed:part.closed}));
 const openings:SourceWholeOpeningTemplate[]=[];
 for(const region of analysis.planeRegions??[]){
  for(const [loopIndex,loop]of (region.openings??[]).entries()){
   const first=complex.halfedges[loop[0]!],last=complex.halfedges[loop[loop.length-1]!];
   const opening=first&&last?openingTemplate((region.id*1024+loopIndex)>>>0,[first.from,last.to],[region.id]):undefined;
   if(opening)openings.push(opening);
  }
 }
 const contacts=compileWholeTemplateContacts(complex);
 // Plane openings are both a visible aperture and protected paper. Keeping
 // the two lists separate lets the native consumer distinguish their roles.
 return {revision:source.revision,parts,openings,contacts,protectedVoids:openings.slice(),paper:openings.slice()};
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
 const wholeConstruction=compileWholeConstructionTemplate(source,analysis,complex);yield;

 // Its relationship counts come from this same compilation: plane-region
 // openings and source contacts, so eligibility can refuse a generic outline.

 // seams of separately indexed faces are not depicted contacts.
 const smallParent=analysis.type==='terrain'?undefined:compileSourceSmallParent(analysis.structure,source.positions,{
  openings:(analysis.planeRegions??[]).reduce((sum,region)=>sum+(region.openings?.length??0),0),...compileSourceRelationships(complex)});

 // from its own edge classes, carried in the source packet the relation
 // stage already binds. A terrain source and the legacy packet keep all.

 const edgeLevels=compileConstructionLevels(runProgram.topology,complex,source.positions,analysis.type);
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


