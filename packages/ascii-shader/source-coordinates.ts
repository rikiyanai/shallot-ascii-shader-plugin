/** Private GPU record interface shared by relation production, projection,
 * native playback and optional inspection. Source charts survive clipping;
 * they never replace the edge ports or the exported chain parameter. A Point
 * keeps the original feature address in visible.w and its canonical source
 * parameter in position.w; visibility intervals are kept in the relation
 * partition stream and never overwrite either field. */
export const SOURCE_CHART_TAG=0x53434831;
export const RELATION_WORDS=24;
export const CONSTRUCTION_POINT_WORDS=8;
/** Stable source correspondence fields in the unchanged Point ABI. The
 * feature address is the original relation/topology address (key.w), not a
 * packed edgeOrdinal+t value; position.w is the source parameter used by the
 * native passage and may be interpolated perspective-correctly. */
export const POINT_FEATURE_ID_WORD='visible.w';
export const POINT_SOURCE_PARAMETER_WORD='position.w';
/** A quiet-NaN source-chart word is outside sourceChartFlags' finite
 * [0,1]-phase output and therefore disjoint from every ordinary Point. It
 * marks the paired, non-drawable whole-edge descriptor in visible.x; the
 * exact full-u32 feature address remains in visible.w for every point. */
export const WHOLE_EDGE_DESCRIPTOR=0x7fc00000;
/** A zero-point Run naming a contributing terrain source. Point.visible.z
 * is its tagged same-frame handle AFTER incidence/assembly; original vertex
 * ports still own joining. This is not a persistent feature identity. */
export const SPAN_SOURCE_TAG=0x53504331;
/** Native output word flag: words22/23 carry the contributor domain/owner,
 * while words16/17 retain the joined construction's root identity. */
export const NATIVE_SPAN_SOURCE=0x01000000;
/** Relation.sourceChart: xy=intrinsic phase, z=authored face class+1
 * (0=no face table), w=integer flags (bit0 phase present, bit1 held far part,
 * bit2 the owner's L2 connected-outline level: surface-relations outlineLevel,
 * packed whole by gpu-construction-runs outlineRelation).
 * The face role survives projection
 * independently of phase. No additional relation or point words. */
export const RELATION_RECORD_WGSL=/* wgsl */`
struct Relation {a:vec4f,b:vec4f,key:vec4u,ports:vec4u,coordinates:vec4f,sourceChart:vec4f}
`;
export const CONSTRUCTION_POINT_WGSL=/* wgsl */`
struct Point {position:vec4f,visible:vec4u}
`;
/** Exact vertex order emitted by examples/shallot/voxel/src/mesher.ts::quad.
 * The certificate is required before a consumer may replace cell-frame
 * inversion with the direct rest-corner lookup below. */
export const VOXEL_REST_CORNER_ORDER='shallot-voxel-quad-v1' as const;
/** Direct rest-lattice corner lookup for the certified Shallot voxel quad.
 * Face codes are the mesher's 0 +X, 1 -X, 2 +Y, 3 -Y, 4 +Z, 5 -Z. */
export const VOXEL_REST_CORNER_WGSL=/* wgsl */`
fn voxelRestLatticeCorner(cell:vec3u,face:u32,corner:u32)->vec3i{
 if(corner>=4u){return vec3i(-2147483647);}
 let u=i32(select(0,1,corner==1u||corner==2u));
 let v=i32(select(0,1,corner>=2u));
 let c=vec3i(cell);
 if(face==0u){return vec3i(c.x+1,c.y+u,c.z+v);}
 if(face==1u){return vec3i(c.x,c.y+v,c.z+u);}
 if(face==2u){return vec3i(c.x+v,c.y+1,c.z+u);}
 if(face==3u){return vec3i(c.x+u,c.y,c.z+v);}
 if(face==4u){return vec3i(c.x+u,c.y+v,c.z+1);}
 if(face==5u){return vec3i(c.x+v,c.y+u,c.z);}
 return vec3i(-2147483647);
}
`;
export function voxelRestLatticeCorner(cell:readonly [number,number,number],face:number,corner:number):readonly [number,number,number]|undefined{
 if(!Number.isInteger(face)||face<0||face>5||!Number.isInteger(corner)||corner<0||corner>3)return;
 const u=corner===1||corner===2?1:0,v=corner>=2?1:0,[x,y,z]=cell;
 switch(face){
  case 0:return [x+1,y+u,z+v];
  case 1:return [x,y+v,z+u];
  case 2:return [x+v,y+1,z+u];
  case 3:return [x+u,y,z+v];
  case 4:return [x+u,y+v,z+1];
  case 5:return [x+v,y+u,z];
 }
}
/** Native whole-word phase on the REST source, not world height, view height,
 * current run length, or the number of screen rows crossed. */
export const REST_SOURCE_CHART_WGSL=/* wgsl */`
fn sourceHeightChart(position:vec3f,low:vec3f,high:vec3f)->f32{
 return clamp((high.y-position.y)/max(high.y-low.y,1e-20),0.0,1.0);
}
`;
export const POINT_SOURCE_CHART_WGSL=/* wgsl */`
// A [0,1] f32 uses only its low30 bits. Charted points carry those bits
// EXACTLY, with bit31=chart present and bit30=visible. Uncharted points keep
// the old0/1 word. No quantization, extra point buffer, or changed edge port.
fn pointVisible(point:Point)->bool{
 if(pointWholeEdgeDescriptor(point)){return false;}
 let flags=point.visible.x;
 if((flags&0x80000000u)!=0u){return (flags&0x40000000u)!=0u;}
 return flags!=0u;
}
fn sourceChartFlags(visible:bool,phase:f32,present:bool)->u32{
 if(!present){return select(0u,1u,visible);}
 return 0x80000000u|select(0u,0x40000000u,visible)|bitcast<u32>(clamp(phase,0.0,1.0));
}
fn sourceChartAt(a:Point,b:Point,t:f32)->vec2f{
 let weights=vec2f(1.0-t,t)*vec2f(bitcast<f32>(a.visible.y),bitcast<f32>(b.visible.y));
 let denominator=weights.x+weights.y;
 if((a.visible.x&b.visible.x&0x80000000u)==0u||denominator<=0.0){return vec2f(0.0);}
 return vec2f(dot(weights,vec2f(bitcast<f32>(a.visible.x&0x3fffffffu),bitcast<f32>(b.visible.x&0x3fffffffu)))/denominator,1.0);
}
// Whole-edge correspondence is explicit in the existing resident packet.
// These helpers make it impossible for a visibility fragment to become the
// feature identity merely by changing its endpoint or local ordinal.
fn pointWholeEdgeDescriptor(point:Point)->bool{return point.visible.x==${WHOLE_EDGE_DESCRIPTOR}u;}
fn pointFeatureId(point:Point)->u32{return point.visible.w;}
fn pointSourceParameter(point:Point)->f32{return point.position.w;}
fn foliageChartBand(chart:f32)->u32{return min(3u,u32(clamp(chart,0.0,1.0)*4.0));}
`;
