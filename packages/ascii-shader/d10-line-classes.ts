/** The existing D10 field classifier, shared by underdrawing and selection.
 * This preserves its current n·v-valley implementation; it does not claim
 * that image-space valleys establish an exact surface-curvature derivative.
 * Callers bind depthField, normalField and ownerField from the SAME pose. */
export const D10_LINE_CLASSES=/* wgsl */`
struct D10Sample {d:f32,n:vec3f,o:vec2u,ndv:f32,on:bool}
fn d10Unproject(q:vec2f,d:f32,extent:vec2f,worldFromClip:mat4x4f)->vec3f{
 let p=worldFromClip*vec4f(q/extent*vec2f(2,-2)+vec2f(-1,1),d,1);return p.xyz/p.w;
}
fn d10At(q:vec2i,extent:vec2f,worldFromClip:mat4x4f)->D10Sample{
 if(any(q<vec2i(0))||any(q>=vec2i(extent))){return D10Sample(0.0,vec3f(0),vec2u(0),0.0,false);}
 let o=textureLoad(ownerField,q,0).xy;let d=textureLoad(depthField,q,0).x;
 if((o.x|o.y)==0u||d<=0.0){return D10Sample(0.0,vec3f(0),vec2u(0),0.0,false);}
 let e=textureLoad(normalField,q,0).xy;var n=vec3f(e,1.0-abs(e.x)-abs(e.y));
 if(n.z<0.0){n=vec3f((vec2f(1.0)-abs(n.yx))*select(vec2f(-1.0),vec2f(1.0),n.xy>=vec2f(0)),n.z);}
 n=normalize(n);let c=vec2f(q)+vec2f(.5);
 let v=normalize(d10Unproject(c,1.0,extent,worldFromClip)-d10Unproject(c,d,extent,worldFromClip));
 let s=dot(n,v);return D10Sample(d,select(n,-n,s<0.0),o,abs(s),true);
}
fn d10Same(a:D10Sample,b:D10Sample)->bool{return b.on&&all(a.o==b.o)&&abs(a.d-b.d)<=.03*max(a.d,b.d);}
// Bits: silhouette/depth step, occluding contour, contact/opening, crease,
// suggestive contour. Ranking and display consume the SAME classifications.
fn d10Classes(q:vec2i,extent:vec2f,worldFromClip:mat4x4f)->u32{
 let c=d10At(q,extent,worldFromClip);if(!c.on){return 0u;}
 var axes=array<vec2i,4>(vec2i(1,0),vec2i(0,1),vec2i(1,1),vec2i(1,-1));var classes=0u;
 for(var k=0u;k<2u;k++){
  let a=d10At(q+axes[k],extent,worldFromClip);let b=d10At(q-axes[k],extent,worldFromClip);
  if(!a.on||!b.on){classes|=1u;continue;}
  if(2.0*c.d-a.d-b.d>.03*c.d){classes|=1u;}
  if(!all(a.o==c.o)&&abs(a.d-c.d)<=.03*c.d){classes|=4u;}
  if(d10Same(c,a)&&dot(c.n,a.n)<.766){classes|=8u;}
 }
 if(c.ndv<.12){for(var k=0u;k<2u;k++){
  let a=d10At(q+axes[k],extent,worldFromClip);let b=d10At(q-axes[k],extent,worldFromClip);
  if((d10Same(c,a)&&a.ndv>=.12)||(d10Same(c,b)&&b.ndv>=.12)){classes|=2u;}
 }}
 if((classes&3u)==0u&&c.ndv<.9){for(var k=0u;k<4u;k++){
  let a1=d10At(q+axes[k],extent,worldFromClip);let b1=d10At(q-axes[k],extent,worldFromClip);
  let a3=d10At(q+3*axes[k],extent,worldFromClip);let b3=d10At(q-3*axes[k],extent,worldFromClip);
  if(d10Same(c,a1)&&d10Same(c,b1)&&d10Same(c,a3)&&d10Same(c,b3)
   &&c.ndv<=a1.ndv&&c.ndv<=b1.ndv&&c.ndv<min(a3.ndv,b3.ndv)-.044){classes|=16u;}
 }}
 return classes;
}
`;
