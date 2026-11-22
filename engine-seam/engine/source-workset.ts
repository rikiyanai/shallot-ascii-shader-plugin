/** Native source broad phase. Rendering owns both its bounds and its change
 * stream; MAIN does not scan/reconstruct the World to guess what may show. */
import {Meshes,MeshInstance,type MeshHandle} from '../../core/mesh';
import {GlobalTransform,TransformRuntime} from '../../core/transform';
import {GlobalTransformHistory,globalTransformTable} from '../../core/rendering/global-transform';
import {RenderContext} from '../../core/rendering/render';
import {CULL_FRUSTUM,CULL_VOLUME_FLOATS} from '../../core/rendering/frustum';
import type {World} from '../../engine';
import {meshInstanceTable} from './preprocess';
import * as d from 'typegpu/data';
import {EntityMeshInstanceInput,MeshMaterial} from './material-data';
import {MaterialTypes} from './material-type';
import {SpatialIndex,type SpatialBounds} from './main-spatial-index';
import {resolvePosedBound,type PosedBoundAuthority} from './posed-bound-producer';
const CULL_BOUNDS_OFFSET=d.memoryLayoutOf(EntityMeshInstanceInput,(row:any)=>row.cullBounds).offset;
/** Column-major world transform for one settled native pose.  This is the
 * same quaternion/scale convention used by posedBox and by the physical
 * source's GPU Root matrix; cache it with the pose, never rebuild it per
 * camera or per parent query. */
export function sourcePoseMatrix(p:Float32Array):Float32Array|undefined{
 if(p.length<12||![0,1,2,4,5,6,7,8,9,10].every(k=>Number.isFinite(p[k]!)))return;
 const x=p[4]!,y=p[5]!,z=p[6]!,w=p[7]!,sx=p[8]!,sy=p[9]!,sz=p[10]!;
 if(![sx,sy,sz].every(Number.isFinite))return;
 const xx=x*x,yy=y*y,zz=z*z,xy=x*y,xz=x*z,yz=y*z,wx=w*x,wy=w*y,wz=w*z;
 return new Float32Array([
  (1-2*(yy+zz))*sx, 2*(xy+wz)*sx, 2*(xz-wy)*sx, 0,
  2*(xy-wz)*sy, (1-2*(xx+zz))*sy, 2*(yz+wx)*sy, 0,
  2*(xz+wy)*sz, 2*(yz-wx)*sz, (1-2*(xx+yy))*sz, 0,
  p[0]!,p[1]!,p[2]!,1,
 ]);
}
function posedBox(box:SpatialBounds,p:Float32Array):SpatialBounds|undefined{
 const bounds:[number,number,number,number,number,number]=[Infinity,Infinity,Infinity,-Infinity,-Infinity,-Infinity];
 for(let corner=0;corner<8;corner++){
  const x=box[corner&1?3:0]!*p[8]!,y=box[corner&2?4:1]!*p[9]!,z=box[corner&4?5:2]!*p[10]!,qx=p[4]!,qy=p[5]!,qz=p[6]!,qw=p[7]!;
  const tx=2*(qy*z-qz*y),ty=2*(qz*x-qx*z),tz=2*(qx*y-qy*x);
  const point=[p[0]!+x+qw*tx+qy*tz-qz*ty,p[1]!+y+qw*ty+qz*tx-qx*tz,p[2]!+z+qw*tz+qx*ty-qy*tx];
  const margin=1e-5*(1+Math.abs(p[0]!)+Math.abs(p[1]!)+Math.abs(p[2]!)+Math.abs(x)+Math.abs(y)+Math.abs(z)+(Math.abs(qw)+Math.abs(qx)+Math.abs(qy)+Math.abs(qz))*(Math.abs(tx)+Math.abs(ty)+Math.abs(tz)));
  for(let k=0;k<3;k++){bounds[k]=Math.min(bounds[k]!,point[k]!-margin);bounds[k+3]=Math.max(bounds[k+3]!,point[k]!+margin);}
 }
 return bounds.every(Number.isFinite)?bounds:undefined;
}

type Member={eid:number;mesh:number;meshRow:number;poseRow:number;materialType:number;materialRow:number;pose:Float32Array;worldFromSource?:Float32Array;seeded:boolean;settleTick:number;initialBoundsFrame:number;bounds?:SpatialBounds;formBounds?:SpatialBounds;boundsAuthority?:PosedBoundAuthority};
type Slot={version:number;volume:Float32Array;meshes:Map<number,number>;members:Member[]};
export interface SourceViewInstance {readonly eid:number;readonly mesh:number;readonly bounds?:SpatialBounds;readonly boundsAuthority?:PosedBoundAuthority;readonly settling:boolean;readonly boundsPending:boolean;readonly worldFromSource?:Readonly<Float32Array>}

export interface SourceViewParent {readonly bounds:SpatialBounds;readonly instances:readonly SourceViewInstance[];readonly key:number}
export interface SourceViewWorkset {readonly instances:readonly SourceViewInstance[];readonly parents?:readonly SourceViewParent[];readonly exterior:boolean}
class SourceWorkset{
 private readonly world:World;
 private readonly index=new SpatialIndex<number,Member>();
 private readonly members=new Map<number,Member>();
 private readonly meshRows=new Map<number,Member>();
 private readonly poseRows=new Map<number,Member>();
 private readonly meshMembers=new Map<number,Set<number>>();
 private readonly dirty=new Set<number>();
 private readonly settling=new Set<number>();
 private readonly uncertain=new Set<number>();
 private readonly materialSources=new Set<number>();
 /** Reverse index for changed material rows. A source table notification is
  * already keyed by (type,row); do not turn it back into a whole-world scan. */
 private readonly materialRows=new Map<number,Map<number,Set<number>>>();
 private enclosure?:{version:number;bounds:SpatialBounds};
 private readonly slots=new Map<number,Slot>();
 private readonly unsubscribe:(()=>void)[]=[];
 private version=0;private frame=-1;
 constructor(world:World){
  this.world=world;
  globalTransformTable(world);
  const poses=world.resource(GlobalTransformHistory).current!;
  this.unsubscribe.push(world.resource(Meshes).observe(handle=>{for(const eid of this.meshMembers.get(handle)??[])this.dirty.add(eid);}),world.observeMembership(MeshInstance,(eid,present)=>{
   if(present)this.dirty.add(eid);else this.remove(eid);
  }),world.observeMembership(GlobalTransform,(eid)=>this.dirty.add(eid)),
  meshInstanceTable(world).observeRows((row)=>{const member=this.meshRows.get(row);if(member)this.dirty.add(member.eid);}),
  poses.observeRows((row,present)=>{
   const member=this.poseRows.get(row);if(!member)return;
   // These notifications include each changed field and explicit range
   // marks. An unchanged range mark is not motion. Changed rows stay
   // conservative until BOTH native history endpoints have settled; motion
   // is not a permanent exemption from the hierarchy after it stops.
   if(member.seeded&&present){
    const data=poses.view,base=row*poses.rowBytes;let changed=false;
    for(const k of [0,1,2,4,5,6,7,8,9,10])if(data.getFloat32(base+k*4,true)!==member.pose[k]){changed=true;break;}
    if(!changed)return;
    this.moved(member);
   }
   if(!present){member.seeded=false;this.moved(member);}this.dirty.add(member.eid);
  }));
  for(const eid of world.query([MeshInstance]))this.dirty.add(eid);
 }
 private observeMaterialSources():void{
  const sources=this.world.resource(MaterialTypes).sources;
  for(let type=0;type<sources.length;type++){
   const source=sources[type];if(!source||this.materialSources.has(type))continue;
   this.materialSources.add(type);
   this.unsubscribe.push(source.table.observeRows((row)=>{
    const members=this.materialRows.get(type)?.get(row);if(!members)return;
    for(const eid of members)if(this.members.has(eid))this.dirty.add(eid);
   }));
  }
 }
 private indexMaterialMember(member:Member):void{
  let rows=this.materialRows.get(member.materialType);if(!rows){rows=new Map();this.materialRows.set(member.materialType,rows);}
  let members=rows.get(member.materialRow);if(!members){members=new Set();rows.set(member.materialRow,members);}
  members.add(member.eid);
 }
 private unindexMaterialMember(member:Member):void{
  const rows=this.materialRows.get(member.materialType),members=rows?.get(member.materialRow);if(!members)return;
  members.delete(member.eid);if(!members.size)rows!.delete(member.materialRow);if(!rows!.size)this.materialRows.delete(member.materialType);
 }
 private moved(member:Member):void{member.settleTick=this.world.time.fixedTick+2;this.settling.add(member.eid);}
 private remove(eid:number):void{
  const member=this.members.get(eid);if(!member)return;
  this.unindexMaterialMember(member);
  this.meshRows.delete(member.meshRow);this.poseRows.delete(member.poseRow);
  const meshMembers=this.meshMembers.get(member.mesh);meshMembers?.delete(eid);if(!meshMembers?.size)this.meshMembers.delete(member.mesh);
  this.members.delete(eid);this.index.delete(eid);this.dirty.delete(eid);this.settling.delete(eid);this.uncertain.delete(eid);this.version++;
 }
 private sync():void{
  const world=this.world;if(this.frame===world.gpu.frame)return;this.frame=world.gpu.frame;this.observeMaterialSources();
  const meshes=world.resource(Meshes),meshTable=meshInstanceTable(world),history=world.resource(GlobalTransformHistory),poses=history.current!;
  const discontinuities=world.resource(TransformRuntime).discontinuities;
  for(const eid of this.settling)this.dirty.add(eid);
  for(const eid of this.dirty){
   if(!world.has(eid,MeshInstance)){this.remove(eid);continue;}
   const mesh=world.storage(MeshInstance).mesh.get(eid),meshRow=meshTable.rowIndex(eid),poseRow=poses.rowIndex(eid);
   let member=this.members.get(eid),newMember=false,materialChanged=false;
   const materialType=world.has(eid,MeshMaterial)?world.storage(MeshMaterial).type.get(eid):0;
   const materialRow=world.has(eid,MeshMaterial)?world.storage(MeshMaterial).material.get(eid):0;
   if(!member){member={eid,mesh,meshRow,poseRow,materialType,materialRow,pose:new Float32Array(12),seeded:false,settleTick:world.time.fixedTick+2,initialBoundsFrame:world.gpu.frame+8};this.members.set(eid,member);this.settling.add(eid);newMember=true;}
   else if(member.materialType!==materialType||member.materialRow!==materialRow){this.unindexMaterialMember(member);materialChanged=true;}
   const meshChanged=member.mesh!==mesh;
   if(meshChanged){const previous=this.meshMembers.get(member.mesh);previous?.delete(eid);if(!previous?.size)this.meshMembers.delete(member.mesh);}
   let meshMembers=this.meshMembers.get(mesh);if(!meshMembers){meshMembers=new Set();this.meshMembers.set(mesh,meshMembers);}meshMembers.add(eid);
   if(member.meshRow!==meshRow)this.meshRows.delete(member.meshRow);
   let poseChanged=!member.seeded;
   if(member.poseRow!==poseRow){this.poseRows.delete(member.poseRow);member.seeded=false;poseChanged=true;this.moved(member);}
   member.mesh=mesh;member.meshRow=meshRow;member.poseRow=poseRow;member.materialType=materialType;member.materialRow=materialRow;
   if(newMember||materialChanged)this.indexMaterialMember(member);
   if(meshRow>=0)this.meshRows.set(meshRow,member);if(poseRow>=0)this.poseRows.set(poseRow,member);
   let bounds:SpatialBounds|undefined,formBounds:SpatialBounds|undefined,boundsAuthority:PosedBoundAuthority|undefined;
   if(poseRow>=0&&world.has(eid,GlobalTransform)){
    const data=poses.view,base=poseRow*poses.rowBytes;
    if(member.seeded)for(const k of [0,1,2,4,5,6,7,8,9,10])if(data.getFloat32(base+k*4,true)!==member.pose[k]){this.moved(member);poseChanged=true;break;}
    for(let k=0;k<12;k++)member.pose[k]=data.getFloat32(base+k*4,true);
    // The source transform is an immutable per-pose cache.  Parent selection
    // and far emission may run for several cameras, so deriving this matrix
    // in instance() would repeat pose work and risk camera-dependent state.
    if(poseChanged)member.worldFromSource=sourcePoseMatrix(member.pose);member.seeded=true;
    // New/changed/remapped rows remain conservative until previous/current
    // GPU history has crossed two unchanged fixed ticks. Only then may the
    // exact current pose bound both endpoints; no guessed interpolation.
    // The native history producer explicitly copies CURRENT into PREVIOUS
    // for a reset/new row before this draw. Those interpolants are identical
    // now; waiting two more ticks leaves a continuously streamed scene
    // permanently uncertain and launches its full-detail source queue.
    const reset=history.historyDiscarded||(((history.historyDiscontinuitiesOnSubmit[eid>>>5]??0)|(discontinuities[eid>>>5]??0))&(1<<(eid&31)))!==0;
    if(reset)member.settleTick=world.time.fixedTick;
    if(world.time.fixedTick>=member.settleTick){
     this.settling.delete(eid);
     const materialState=world.resource(MaterialTypes),material=materialState.types[materialType],source=materialState.sources[materialType];
     const native=meshes.get(mesh as MeshHandle);
     const override=meshRow>=0?[0,1,2,3].map(k=>meshTable.view.getFloat32(meshRow*meshTable.rowBytes+CULL_BOUNDS_OFFSET+k*4,true)):undefined;
     const resolved=resolvePosedBound({material,source,materialRow,override,nativeSphere:native?.bounds,pose:member.pose});
     bounds=resolved.bounds;boundsAuthority=resolved.authority;
     // positionBounds is a tighter affine certificate only. Never use it for
     // a custom vertex stage, even when that stage happens to have no source.
     if(boundsAuthority==='native-affine'&&native?.positionBounds)formBounds=posedBox(native.positionBounds,member.pose);
    }
   }
   // Mesh membership changes the cached candidate counts even when its
   // mutable Member retains an identical sphere and object identity.
   // Candidate COUNTS must still bound the native GPU SPHERE culler, even
   // where that culler includes geometry outside the frustum. Tightening
   // those counts to the form box under-reserves its real indirect stream.
   const formChanged=member.formBounds===undefined?formBounds!==undefined:formBounds===undefined||member.formBounds.some((value,k)=>value!==formBounds![k]);
   member.bounds=bounds;member.formBounds=formBounds;member.boundsAuthority=boundsAuthority;
   if(bounds)this.uncertain.delete(eid);else this.uncertain.add(eid);
   if(this.index.set(eid,bounds,member)||meshChanged||formChanged)this.version++;
  }
  this.dirty.clear();
 }
 visible(slot:number):ReadonlyMap<number,number>|undefined{
  this.sync();const context=this.world.resource(RenderContext),base=slot*CULL_VOLUME_FLOATS,data=context.cullVolumeStaging;
  if(slot<0||slot>=context.viewCount||!data||data[base]!==CULL_FRUSTUM)return;
  for(let i=4;i<CULL_VOLUME_FLOATS;i++)if(!Number.isFinite(data[base+i]))return;
  let held=this.slots.get(slot);
  if(!held){held={version:-1,volume:new Float32Array(CULL_VOLUME_FLOATS),meshes:new Map(),members:[]};this.slots.set(slot,held);}
  let same=held.version===this.version;for(let i=0;i<CULL_VOLUME_FLOATS&&same;i++)same=held.volume[i]===data[base+i];
  if(!same){held.meshes.clear();held.members.length=0;this.index.query(data,member=>{held!.meshes.set(member.mesh,(held!.meshes.get(member.mesh)??0)+1);held!.members.push(member);},base+4);held.volume.set(data.subarray(base,base+CULL_VOLUME_FLOATS));held.version=this.version;}
  return held.meshes;
 }
 view(slot:number,eye:ArrayLike<number>,adequate?:(bounds:SpatialBounds)=>boolean):SourceViewWorkset|undefined{
  if(!this.visible(slot))return;
  const held=this.slots.get(slot)!;
  if(adequate){

   // a node with a settling member is never one (its members stay single).
   const data=this.world.resource(RenderContext).cullVolumeStaging!,base=slot*CULL_VOLUME_FLOATS;
   const instances:SourceViewInstance[]=[],parents:SourceViewParent[]=[];
   this.index.select(data,adequate,(bounds,members,key)=>{if(members.some(member=>this.settling.has(member.eid)))instances.push(...members.map(member=>this.instance(member)));else parents.push({bounds,key,instances:members.map(member=>this.instance(member))});},member=>instances.push(this.instance(member)),base+4);
   return {instances,parents,exterior:this.exterior(eye)};
  }
  return {instances:held.members.map(member=>this.instance(member)),exterior:this.exterior(eye)};
 }
 private instance(member:Member):SourceViewInstance{
  return {eid:member.eid,mesh:member.mesh,bounds:member.formBounds??member.bounds,boundsAuthority:member.boundsAuthority,settling:this.settling.has(member.eid),worldFromSource:member.worldFromSource,
   // A paused fixed clock or continuous motion must never defer forever.
   // Eight drawing ticks bound only this initial wait, not source coverage.
   boundsPending:this.settling.has(member.eid)&&this.world.gpu.frame<member.initialBoundsFrame};
 }
 private exterior(eye:ArrayLike<number>):boolean{
  // A window/open doorway does not make an interior camera exterior. Only
  // being OUTSIDE a conservative bound of the complete known scene certifies
  // this first tier owner. Ground-level/unknown cases keep the full producer.
  if(this.dirty.size||this.uncertain.size||!this.members.size||![eye[0],eye[1],eye[2]].every(Number.isFinite))return false;
  if(this.enclosure?.version!==this.version){
   const bounds:[number,number,number,number,number,number]=[Infinity,Infinity,Infinity,-Infinity,-Infinity,-Infinity];
   for(const member of this.members.values()){const b=(member.formBounds??member.bounds)!;for(let k=0;k<3;k++){bounds[k]=Math.min(bounds[k]!,b[k]!);bounds[k+3]=Math.max(bounds[k+3]!,b[k+3]!);}}
   this.enclosure={version:this.version,bounds};
  }
  const bounds=this.enclosure.bounds;
  for(let k=0;k<3;k++)if(eye[k]!<bounds[k]!||eye[k]!>bounds[k+3]!)return true;
  return false;
 }
  dispose():void{for(const stop of this.unsubscribe)stop();this.index.clear();this.members.clear();this.meshRows.clear();this.poseRows.clear();this.meshMembers.clear();this.materialRows.clear();this.dirty.clear();this.settling.clear();this.uncertain.clear();this.materialSources.clear();this.enclosure=undefined;this.slots.clear();}
}
const key={create(world:World){const state=new SourceWorkset(world);world.onDispose(()=>state.dispose());return state;}};
export function initializeSourceWorkset(world:World):void{world.resource(key);}
/** Candidate counts bound the native packed instance stream before material
 * extraction. Unknown/moving instances remain candidates; the native GPU
 * culler and extraction overflow check still own exact visibility/extents. */
export function visibleSourceMeshes(world:World,slot:number):ReadonlyMap<number,number>|undefined{return world.resource(key).visible(slot);}
export function sourceViewWorkset(world:World,slot:number,eye:ArrayLike<number>,adequate?:(bounds:SpatialBounds)=>boolean):SourceViewWorkset|undefined{return world.resource(key).view(slot,eye,adequate);}
