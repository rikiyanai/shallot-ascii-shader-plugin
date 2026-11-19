export type ContactVector=readonly [number,number,number];
export interface ContactRotation{readonly v:{readonly x:number;readonly y:number;readonly z:number};readonly s:number}
/** Inverse body rotation: solver anchors are world-space vectors from COM,
 * whereas the retained incidence must follow the body's displayed pose. */
export function contactBodyVector(v:ContactVector,q:ContactRotation):ContactVector{
 const x=-q.v.x,y=-q.v.y,z=-q.v.z,w=q.s;
 const tx=2*(y*v[2]-z*v[1]),ty=2*(z*v[0]-x*v[2]),tz=2*(x*v[1]-y*v[0]);
 return [v[0]+w*tx+y*tz-z*ty,v[1]+w*ty+z*tx-x*tz,v[2]+w*tz+x*ty-y*tx];
}
export function contactBodyPoint(anchor:ContactVector,center:ContactVector,q:ContactRotation):ContactVector{
 const v=contactBodyVector(anchor,q);return [center[0]+v[0],center[1]+v[1],center[2]+v[2]];
}
/** Shape meshes are in body-local meters with Shape placement baked, but
 * without the body's Transform.scale (Shallot shapeMesh contract). Bound
 * shapes inherit the body's complete GlobalTransform. Undo that scale only;
 * never sample a fixed-step world point as if it were the interpolated pose. */
export function contactSourcePoint(point:ContactVector,scale:ContactVector):ContactVector|undefined{
 if(!point.every(Number.isFinite)||!scale.every(v=>Number.isFinite(v)&&Math.abs(v)>=.01))return;
 return [point[0]/scale[0],point[1]/scale[1],point[2]/scale[2]];
}

