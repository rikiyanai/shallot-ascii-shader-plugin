type SpatialBounds=[number,number,number,number,number,number];

/** A producer certificate is required before a custom vertex stage may use a
 * per-instance bound for early representation selection.  The native engine
 * documents both MeshInstance.cullBounds and MaterialSource.cullBounds as
 * conservative local-space spheres; this helper is the one transform seam
 * shared by those two producers. */
export const ENGINE_POSED_BOUND_CERTIFICATE='engine-deformed-cull-sphere-v1' as const;
export type PosedBoundAuthority='native-affine'|'entity-cull-sphere'|'engine-deformed-cull-sphere'|'material-source-cull-sphere';
export interface PosedBoundMaterial {readonly vertex?:unknown;readonly mainPosedBounds?:unknown}
export interface PosedBoundSource {readonly cullBounds?: (row:number,out:Float32Array,nativeSphere?:ArrayLike<number>)=>void}
export interface PosedBoundInput {
 readonly material?:PosedBoundMaterial;
 readonly source?:PosedBoundSource;
 readonly materialRow:number;
 readonly override?:ArrayLike<number>;
 readonly nativeSphere?:ArrayLike<number>;
 readonly pose:ArrayLike<number>;
}
export interface PosedBoundResult {readonly bounds?:SpatialBounds;readonly authority?:PosedBoundAuthority}

function sphere(value:ArrayLike<number>|undefined):[number,number,number,number]|undefined{
 if(!value||value.length<4)return;
 const result=[value[0],value[1],value[2],value[3]] as [number,number,number,number];
 return result.every(Number.isFinite)&&result[3]>=0?result:undefined;
}

/** Transform a complete local conservative sphere into a world AABB.  The
 * radius uses the largest absolute scale, exactly as native culling does. */
export function posedSphereBounds(value:ArrayLike<number>|undefined,pose:ArrayLike<number>):SpatialBounds|undefined{
 const input=sphere(value);
 if(!input||pose.length<11)return;
 const [cx,cy,cz,r]=input,x=cx*pose[8]!,y=cy*pose[9]!,z=cz*pose[10]!;
 const qx=pose[4]!,qy=pose[5]!,qz=pose[6]!,qw=pose[7]!;
 if(![pose[0],pose[1],pose[2],qx,qy,qz,qw,pose[8],pose[9],pose[10]].every(Number.isFinite))return;
 const tx=2*(qy*z-qz*y),ty=2*(qz*x-qx*z),tz=2*(qx*y-qy*x);
 const center=[pose[0]!+x+qw*tx+qy*tz-qz*ty,pose[1]!+y+qw*ty+qz*tx-qx*tz,pose[2]!+z+qw*tz+qx*ty-qy*tx];
 const roundoff=1e-5*(1+Math.abs(pose[0]!)+Math.abs(pose[1]!)+Math.abs(pose[2]!)+Math.abs(x)+Math.abs(y)+Math.abs(z)+
  (Math.abs(qw)+Math.abs(qx)+Math.abs(qy)+Math.abs(qz))*(Math.abs(tx)+Math.abs(ty)+Math.abs(tz)));
 const radius=r*Math.max(Math.abs(pose[8]!),Math.abs(pose[9]!),Math.abs(pose[10]!))+roundoff;
 const bounds=[center[0]!-radius,center[1]!-radius,center[2]!-radius,center[0]!+radius,center[1]!+radius,center[2]!+radius] as SpatialBounds;
 return bounds.every(Number.isFinite)?bounds:undefined;
}

/** Resolve one instance's only safe early bound.  A source callback wins over
 * an entity override, because that is also the native forward-cull order.  A
 * custom vertex stage never falls back to the rest mesh sphere. */
export function resolvePosedBound(input:PosedBoundInput):PosedBoundResult{
 if(input.source?.cullBounds){
  // A callback is allowed to write the output in place.  Start invalid so a
  // no-op or partial producer cannot accidentally certify the origin with a
  // zero-radius sphere.  The producer must author all four values.
  const value=new Float32Array([NaN,NaN,NaN,-1]);
  const base=sphere(input.override)??sphere(input.nativeSphere);
  try{input.source.cullBounds(input.materialRow,value,base);}catch{return {};} // an invalid producer keeps the incumbent
  const bounds=posedSphereBounds(value,input.pose);
  return bounds?{bounds,authority:'material-source-cull-sphere'}:{};
 }
 const override=sphere(input.override);
 if(override){
  const bounds=posedSphereBounds(override,input.pose);
  if(bounds)return {bounds,authority:input.material?.mainPosedBounds===ENGINE_POSED_BOUND_CERTIFICATE?'engine-deformed-cull-sphere':'entity-cull-sphere'};
  return {};
 }
 // No producer-issued posed sphere exists for an arbitrary custom vertex
 // function.  In particular, do not substitute its rest/native AABB.
 if(input.material?.vertex)return {};
 const bounds=posedSphereBounds(input.nativeSphere,input.pose);
 return bounds?{bounds,authority:'native-affine'}:{};
}
