                                                                        
//
                                                                           
// emits every PART_FORK band as an explicit source incidence graph: each arm
// section with its ports, every band halfedge/face/source edge with the arm
// section that labels each end, and the section's exact interpolated samples.
// That graph is resident in the source's topology words; the native anchor pass
// (gpu-construction-runs.ts `anchors`, junctionTable/junctionJoint/
// junctionArmOf) reads it directly. This module is the same rule on the CPU:
// the reference the shader is tested against, and the report of what a source
// fork offers the native consumer.
//
// Association is by SOURCE ID only:
//  - an edge belongs to the arm whose section labels it (one arm label, or the
//    same arm at both ends); an edge whose two ends carry two DIFFERENT arms is
//    a joint edge between them; an edge with no arm label is interior
//    triangulation and belongs to no arm (it is never drawn as a construction);
//  - the JOINT is a band vertex at which every arm of the fork meets (its own
//    label plus the labels across its band edges cover all arm sections);
//  - the composition at a joint accounts for EVERY arm: the record's last arm
//    (the producer lists lower sections, then upper ones) is the trunk and
//    each other arm, in record order, is one continued branch, so a k-arm
//    fork is k-1 cooperating retained branch_entry parts at one source anchor;
//  - a relation reaches an arm through its own SOURCE EDGE id and halfedge
//    orientation (sourceJunctionEndOf), never by comparing a relation's port
//    with a packet representative;
//  - records sharing a joint compose all their branches in source order,
//    up to SOURCE_JUNCTION_UNION_LIMIT; an incomplete/oversized union keeps
//    the complete incumbent.
// Nothing here is a nearest-fragment choice, a hull or a per-edge drawing, and
// nothing runs per frame. An incomplete record has no joint (incumbent).
import {
 type SourcePartJunctionRecord,
 TOPOLOGY_PART_JUNCTION_FLAG_COMPLETE,TOPOLOGY_PART_JUNCTION_FLAG_MISSING_ARM,TOPOLOGY_PART_JUNCTION_ARM_COMPLETE,TOPOLOGY_PART_JUNCTION_NONE,
} from './compiled-construction-topology.ts';

/** The shader scans at most this many junction records and arms per source. */
export const SOURCE_JUNCTION_RECORD_LIMIT=64,SOURCE_JUNCTION_ARM_LIMIT=32;

export type SourceJunctionRefusal='incomplete'|'missing-arm'|'arm-limit'|'no-joint'|'shared-joint';
export interface SourceJunctionAssociation {
 readonly sourceId:number;
 /** Arm sections in record order and the original source edge ids each owns. */
 readonly arms:readonly {readonly sectionId:number;readonly ports:readonly number[];readonly sourceEdges:readonly number[]}[];
 /** Original source edges joining two different arms: [sourceEdge, armA, armB]. */
 readonly jointEdges:readonly (readonly [number,number,number])[];
 /** Interior triangulation edges no arm labels (never drawn as a construction). */
 readonly interiorEdges:readonly number[];
 /** Vertices where every arm meets. */
 readonly joints:readonly number[];
 /** The trunk arm (entering every retained part) and the continued branches. */
 readonly trunk:number;readonly branches:readonly number[];
}

const usable=(record:SourcePartJunctionRecord):SourceJunctionRefusal|undefined=>{
 if((record.flags&TOPOLOGY_PART_JUNCTION_FLAG_COMPLETE)===0)return 'incomplete';
 if((record.flags&TOPOLOGY_PART_JUNCTION_FLAG_MISSING_ARM)!==0||record.arms.length<2||record.arms.some(arm=>(arm.flags&TOPOLOGY_PART_JUNCTION_ARM_COMPLETE)===0))return 'missing-arm';
 if(record.arms.length>SOURCE_JUNCTION_ARM_LIMIT)return 'arm-limit';
 return undefined;
};

/** Every arm section met at `vertex` across this record's band edges. */
export function sourceJunctionArmsAt(record:SourcePartJunctionRecord,vertex:number):Set<number>{
 const arms=new Set(record.arms.map(arm=>arm.sectionId)),met=new Set<number>();
 for(const edge of record.edges){
  if(edge.a===vertex||edge.b===vertex)for(const label of [edge.labelA,edge.labelB])if(label!==TOPOLOGY_PART_JUNCTION_NONE&&arms.has(label))met.add(label);
 }
 return met;
}

export function sourceJunctionAssociation(record:SourcePartJunctionRecord):SourceJunctionAssociation|{readonly refusal:SourceJunctionRefusal}{
 const refusal=usable(record);if(refusal)return {refusal};
 const armIds=new Set(record.arms.map(arm=>arm.sectionId));const label=(value:number)=>armIds.has(value)?value:undefined;
 const owned=new Map<number,Set<number>>(record.arms.map(arm=>[arm.sectionId,new Set<number>()]));
 const joint=new Map<number,[number,number,number]>(),interior=new Set<number>();
 for(const edge of record.edges){
  const a=label(edge.labelA),b=label(edge.labelB);
  if(a!==undefined&&b!==undefined&&a!==b)joint.set(edge.sourceEdge,[edge.sourceEdge,Math.min(a,b),Math.max(a,b)]);
  else if(a!==undefined||b!==undefined)owned.get((a??b)!)!.add(edge.sourceEdge);
  else interior.add(edge.sourceEdge);
 }
 // A source edge listed by both of its halfedges keeps ONE owner: joint wins
 // over arm, arm over interior.
 for(const id of joint.keys()){for(const set of owned.values())set.delete(id);interior.delete(id);}
 for(const set of owned.values())for(const id of set)interior.delete(id);
 const vertices=[...new Set(record.edges.flatMap(edge=>[edge.a,edge.b]))].sort((x,y)=>x-y);
 const joints=vertices.filter(vertex=>sourceJunctionArmsAt(record,vertex).size===record.arms.length);
 if(!joints.length)return {refusal:'no-joint'};
 return {sourceId:record.sourceId,
  arms:record.arms.map(arm=>({sectionId:arm.sectionId,ports:arm.ports,sourceEdges:[...owned.get(arm.sectionId)!].sort((x,y)=>x-y)})),
  jointEdges:[...joint.values()].sort((x,y)=>x[0]-y[0]),interiorEdges:[...interior].sort((x,y)=>x-y),joints,
  trunk:record.arms.at(-1)!.sectionId,branches:record.arms.slice(0,-1).map(arm=>arm.sectionId)};
}

/** At most this many fork records compose at one shared joint (the shader's
 * FORK_UNION); a larger union keeps the incumbent. */
export const SOURCE_JUNCTION_UNION_LIMIT=4;
export interface SourceJunctionPart {readonly fork:number;readonly trunk:number;readonly branch:number}
/** The complete retained composition at one packet joint, as the shader
 * emits it: every usable record whose arms all meet there (table order), each
 * contributing one part per branch entering from its trunk. */
export function sourceJunctionParts(records:readonly SourcePartJunctionRecord[],joint:number):readonly SourceJunctionPart[]|{readonly refusal:SourceJunctionRefusal}{
 const meeting=records.filter(record=>usable(record)===undefined&&sourceJunctionArmsAt(record,joint).size===record.arms.length);
 if(!meeting.length)return {refusal:'no-joint'};
 if(meeting.length>SOURCE_JUNCTION_UNION_LIMIT)return {refusal:'shared-joint'};
 return meeting.flatMap(record=>record.arms.slice(0,-1).map(arm=>({fork:record.sourceId,trunk:record.arms.at(-1)!.sectionId,branch:arm.sectionId})));
}
/** The arm an endpoint at the joint reaches (the shader's forkArm): the far
 * end's arm, else the joint's own arm when the edge is owned by it. */
export function sourceJunctionArmReached(record:SourcePartJunctionRecord,end:readonly [number,number,number,number]):number|undefined{
 const arms=new Set(record.arms.map(arm=>arm.sectionId));
 return arms.has(end[2])?end[2]:arms.has(end[3])?end[3]:undefined;
}

/** The packet vertex at a band relation's endpoint, exactly as the shader's
 * forkEnd reads it: the relation names its index edge id (its lowest
 * incidence halfedge) and orders its ends by that halfedge; the packet edge
 * with that halfedge has the same orientation, a lone opposite incidence the
 * reverse. Returns [vertex, far vertex, far label, own label] or undefined. */
export function sourceJunctionEndOf(record:SourcePartJunctionRecord,edgeId:number,end:0|1):readonly [number,number,number,number]|undefined{
 const entries=record.edges.filter(edge=>edge.sourceEdge===edgeId);
 const own=entries.find(edge=>edge.halfedge===edgeId);
 if(own)return end===0?[own.a,own.b,own.labelB,own.labelA]:[own.b,own.a,own.labelA,own.labelB];
 if(entries.length!==1)return undefined;
 const twin=entries[0]!;return end===0?[twin.b,twin.a,twin.labelA,twin.labelB]:[twin.a,twin.b,twin.labelB,twin.labelA];
}
