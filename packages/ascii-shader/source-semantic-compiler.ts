import {compileMeshSource,type MeshSourceComplex} from './source-complex.ts';
import {compileSourceFaceMaterialClasses} from './source-material.ts';
import {compileSourceAnalysis} from './composition.ts';
import {meshPartWork} from './mesh-part-graph.ts';
import {compileSourceRunProgram,compileConstructionLevels,compileMeshProfileSafe,compileLimbVertices,sourcePartProgramPacket,PACKED_RUN_CONTEXT} from './compiled-construction-topology.ts';
import {compileWholeConstructionTemplate,type SourceProgramInput} from './source-program.ts';
import {encodeSourceSemanticResidual,type SourcePublishedAuthority,type SourceSemanticResidual} from './source-semantic-residual.ts';
import {compileSourceRelationships,compileSourceSmallParent} from './source-small-parent.ts';
import {validateSourceConstruction} from './source-construction.ts';

/** Copy the source-complex published correspondence into the compact semantic
 * asset. This is deliberately sourced from the published groups, rather than
 * rebuilding equal-position adjacency: P0's independent indices, member
 * orientation and original halfedge addresses are part of its authority. */
export function compilePublishedAuthority(complex:MeshSourceComplex):SourcePublishedAuthority|undefined{
 if(!complex.vertexContacts)return;
 let vertices=0;for(const group of complex.vertexContacts)for(const vertex of group)vertices=Math.max(vertices,vertex+1);
 const vertexCanon=new Uint32Array(vertices),vertexNext=new Uint32Array(vertices).fill(0xffffffff),vertexCounts=new Uint32Array(vertices),seen=new Uint8Array(vertices);
 for(const group of complex.vertexContacts){
  if(!group.length)throw Error('Published source contains an empty vertex group');
  const root=group[0]!;if(root>=vertices)throw Error('Published source vertex group is outside its source');
  vertexCounts[root]=group.length;
  for(const [index,vertex]of group.entries()){
   if(vertex>=vertices||seen[vertex])throw Error('Published source vertex groups overlap');
   seen[vertex]=1;vertexCanon[vertex]=root;vertexNext[vertex]=index+1<group.length?group[index+1]!:0xffffffff;
  }
 }
 if(seen.some(value=>value===0))throw Error('Published source vertex groups do not cover the source');
 const contactRecords=new Uint32Array(complex.contacts.length*2),edgeContacts=new Uint32Array(complex.edges.length),contactMembers=new Uint32Array(complex.contacts.reduce((sum,contact)=>sum+contact.members.length,0)*2);
 let memberStart=0;
 for(const [contactIndex,contact]of complex.contacts.entries()){
  contactRecords[contactIndex*2]=memberStart;contactRecords[contactIndex*2+1]=contact.members.length;
  for(const member of contact.members){
   const halfedge=complex.halfedges[member.edge],edge=halfedge?.edge;
   if(!halfedge||edge===undefined||edge>=edgeContacts.length)throw Error('Published source contact names an absent halfedge');
   if(edgeContacts[edge]!==0)throw Error('Published source edge belongs to multiple contacts');
   edgeContacts[edge]=contactIndex+1;contactMembers[memberStart*2]=member.edge;contactMembers[memberStart*2+1]=Number(member.forward);memberStart++;
  }
 }
 return {vertexCanon,vertexNext,vertexCounts,contactRecords,edgeContacts,contactMembers};
}

/** BUILD-TIME semantic adapter. It uses the existing pure decisions, but never
 * calls sourceProgramWork, packs incidence, expands physical vertex streams or
 * returns a MeshSourceComplex. Dynamic runtime compilation stays with its
 * existing worker until the paired GPU numeric producer is connected.
 *
 * Transient source analysis here is build work, not phone first-page work.
 * Only the explicit small result survives; no positions/indices/full analysis
 * graph can accidentally become another expanded program archive. */
export function* sourceSemanticResidualWork(source:SourceProgramInput):Generator<void,SourceSemanticResidual>{
 const complex=compileMeshSource(source);yield;
 const definition=source.physical??source.definition??{};
 const materialClasses=compileSourceFaceMaterialClasses(definition,source.indices.length/3);
 const analysis=compileSourceAnalysis({...source,owner:source.id,...definition},materialClasses,{weld:complex.weld,retention:'summary'});yield;
 if(analysis.type!=='terrain'){
  try{analysis.structure=yield* meshPartWork({...source,adjacency:source.partAdjacency??source.adjacency},complex.weld);}
  catch(error){analysis.structureError=error instanceof Error?error.message:String(error);}
 }
 const program=compileSourceRunProgram({...analysis,complex,sourcePositions:source.positions});analysis.runProgram=program;yield;
 const topology=program.topology,classesAt=topology[42]!,packed=topology[37]===PACKED_RUN_CONTEXT;
 const edgeClasses=new Uint32Array(Math.ceil(source.indices.length/16)),boundaryContexts:number[]=[];
 const groupsAt=topology[38]!;
 for(let h=0;h<source.indices.length;h++){
  const value=topology[classesAt+h]!,klass=packed?value&3:value;
  edgeClasses[h>>>4]=edgeClasses[h>>>4]!|(klass<<((h&15)*2));
  let group=topology[groupsAt+h]!;if(packed)group=((group&0x60000000)|((group>>>2)&0x07ffffff))>>>0;
  if((group&0x60000000)===0x20000000)boundaryContexts.push(h,group);
 }
 const verticesAt=topology[40]!,vertexContexts=topology.slice(verticesAt,verticesAt+topology[41]!);



 // validated exactly as the worker route validates them (sourceProgramWork).
 if(definition.construction)validateSourceConstruction(definition.construction,source.indices.length,source.positions.length/3);
 const edgeLevels=compileConstructionLevels(topology,complex,source.positions,analysis.type,definition.construction);
 const positions=source.positions;

 const profileSafe=analysis.type==='terrain'?false:compileMeshProfileSafe(positions,source.indices);

 const limbVertices=compileLimbVertices({structure:analysis.structure,complex});

 // baked/numeric route; no second pairing or projected-fragment program.
 const partPrograms=sourcePartProgramPacket(topology);
 // The baked path shares the worker's source producer. Reuse this build's
 // existing analysis/complex; never run sourceProgramWork or reconstruct a
 // source graph in the numeric resident or a displayed camera frame.
 const wholeConstruction=compileWholeConstructionTemplate(source,analysis,complex);
 const publishedAuthority=source.adjacency?compilePublishedAuthority(complex):undefined;
 // Keep the small-parent decision on the same immutable graph/relationship
 // inputs as sourceProgramWork. It is a compact source fact, not a displayed
 // frame compilation and not a second source-program worker.
 const smallParent=analysis.type==='terrain'?undefined:compileSourceSmallParent(analysis.structure,source.positions,{openings:(analysis.planeRegions??[]).reduce((sum,region)=>sum+(region.openings?.length??0),0),...compileSourceRelationships(complex)});
 return {profileSafe,...(limbVertices?.length?{limbVertices}:{}),...(partPrograms?.length?{partPrograms}:{}),...(wholeConstruction?{wholeConstruction}:{}),...(smallParent?{smallParent}:{}),vertices:positions.length/3,corners:source.indices.length,indexedEdges:complex.edges.length,
  type:analysis.type,styleWord:analysis.styleWord??0,foliage:!!analysis.foliage,organic:!!analysis.organic,publishedAdjacency:!!source.adjacency,
  planeRegions:analysis.planeRegions.length,boundaries:complex.boundaries.length,contacts:complex.contacts.length,
  contactMembers:complex.contacts.reduce((n,contact)=>n+contact.members.length,0),
  edgeClasses,vertexContexts,criticalVertices:program.criticalVertices,boundaryContexts:Uint32Array.from(boundaryContexts),faceMaterials:materialClasses??new Uint32Array(),toneRegions:analysis.toneRegions,...(edgeLevels?.length?{edgeLevels}:{}),...(publishedAuthority?{publishedAuthority}:{})};
}

/** Node asset-build entry. The generator remains available for cooperative
 * build hosts; neither entry is imported by the displayed-frame renderer. */
export function compileSourceSemanticAsset(source:SourceProgramInput,identity:Uint32Array):Uint8Array{
 const work=sourceSemanticResidualWork(source);let next=work.next();while(!next.done)next=work.next();
 return encodeSourceSemanticResidual(next.value,identity);
}
