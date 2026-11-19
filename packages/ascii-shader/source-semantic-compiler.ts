import {compileMeshSource} from './source-complex.ts';
import {compileSourceFaceMaterialClasses} from './source-material.ts';
import {compileSourceAnalysis} from './composition.ts';
import {meshPartWork} from './mesh-part-graph.ts';
import {compileSourceRunProgram,compileConstructionLevels,compileMeshProfileSafe,compileLimbVertices,sourcePartProgramPacket,PACKED_RUN_CONTEXT} from './compiled-construction-topology.ts';
import {compileWholeConstructionTemplate,type SourceProgramInput} from './source-program.ts';
import {encodeSourceSemanticResidual,type SourceSemanticResidual} from './source-semantic-residual.ts';
import {compileSourceRelationships,compileSourceSmallParent} from './source-small-parent.ts';

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
 // Compiled construction levels (perf-impl-93), with the same revision's
 // classes; perf-impl-106 appends the compiled L2 outline hull.
 const edgeLevels=compileConstructionLevels(topology,complex,source.positions,analysis.type);
 const positions=source.positions;
 // perf-impl-108: the parent member adequacy, with this revision (source-change work).
 const profileSafe=compileMeshProfileSafe(positions,source.indices);
 // perf-impl-109: the limb-vertex table for the GPU-derived run program (seedRuntime).
 const limbVertices=compileLimbVertices({structure:analysis.structure,complex});
 // perf-impl-110: the same source-owned part records are copied into the
 // baked/numeric route; no second pairing or projected-fragment program.
 const partPrograms=sourcePartProgramPacket(topology);
 // The baked path shares the worker's source producer. Reuse this build's
 // existing analysis/complex; never run sourceProgramWork or reconstruct a
 // source graph in the numeric resident or a displayed camera frame.
 const wholeConstruction=compileWholeConstructionTemplate(source,analysis,complex);
 // Keep the small-parent decision on the same immutable graph/relationship
 // inputs as sourceProgramWork. It is a compact source fact, not a displayed
 // frame compilation and not a second source-program worker.
 const smallParent=analysis.type==='terrain'?undefined:compileSourceSmallParent(analysis.structure,source.positions,{openings:(analysis.planeRegions??[]).reduce((sum,region)=>sum+(region.openings?.length??0),0),...compileSourceRelationships(complex)});
 return {profileSafe,...(limbVertices?.length?{limbVertices}:{}),...(partPrograms?.length?{partPrograms}:{}),...(wholeConstruction?{wholeConstruction}:{}),...(smallParent?{smallParent}:{}),vertices:positions.length/3,corners:source.indices.length,indexedEdges:complex.edges.length,
  type:analysis.type,styleWord:analysis.styleWord??0,foliage:!!analysis.foliage,organic:!!analysis.organic,publishedAdjacency:!!source.adjacency,
  planeRegions:analysis.planeRegions.length,boundaries:complex.boundaries.length,contacts:complex.contacts.length,
  contactMembers:complex.contacts.reduce((n,contact)=>n+contact.members.length,0),
  edgeClasses,vertexContexts,criticalVertices:program.criticalVertices,boundaryContexts:Uint32Array.from(boundaryContexts),faceMaterials:materialClasses??new Uint32Array(),toneRegions:analysis.toneRegions,...(edgeLevels?.length?{edgeLevels}:{})};
}

/** Node asset-build entry. The generator remains available for cooperative
 * build hosts; neither entry is imported by the displayed-frame renderer. */
export function compileSourceSemanticAsset(source:SourceProgramInput,identity:Uint32Array):Uint8Array{
 const work=sourceSemanticResidualWork(source);let next=work.next();while(!next.done)next=work.next();
 return encodeSourceSemanticResidual(next.value,identity);
}
