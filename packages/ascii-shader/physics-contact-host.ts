import {GlobalTransform,MeshInstance,Shape,type World} from '@dylanebert/shallot';
import {contactSourcePoint,type ContactVector} from './contact-attachment.ts';
import {SourcePhysicsContacts} from './source-physics-contacts.ts';
import {querySourceCorner,type SourceCornerIndexData} from './source-contact-corners.ts';
import {SourceContactPatchTable,type SourceContactCorner,type SourceContactLink} from './source-contact-patches.ts';

interface ContactSource{readonly domain:number;readonly revision:number;readonly geometry:{readonly positions:Float32Array};readonly cornerIndex?:SourceCornerIndexData}
interface Binding{readonly source:ContactSource;readonly eid:number;readonly lifetime:number;readonly scale:ContactVector}
/** Matches Box3D's physical LINEAR_SLOP (meters), not pixels or glyph cells. */
const CONTACT_CORNER_TOLERANCE=.005;
/** Cached solver anchors may be inside the contact's speculative normal
 * slab (Box3D SPECULATIVE_DISTANCE = 4 * LINEAR_SLOP), not on the mesh plane.
 * Only that normal direction gets the slab; tangential address tolerance
 * remains LINEAR_SLOP, so nearby unrelated corners are not merged. */
const CONTACT_NORMAL_SLAB=4*CONTACT_CORNER_TOLERANCE;

/** Native solver incidence -> retained original-source corners. No rendered
 * all-pairs scan, geometry compilation or world/screen weld runs here. */
export class PhysicsContactHost {
 private readonly contacts=new SourcePhysicsContacts();
 private readonly patches=new SourceContactPatchTable();
 private signature='';
 get value(){return this.patches.value;}
 update(world:World,sourceForMesh:(mesh:number)=>ContactSource|undefined){
  const changed=this.contacts.update(world),bindings=new Map<number,Binding|undefined>();
  const binding=(shape:number,body:number):Binding|undefined=>{
   if(bindings.has(shape))return bindings.get(shape);
   const eid=world.resolve(shape as Parameters<World['resolve']>[0]);let result:Binding|undefined;
   if(eid&&world.has(eid,Shape)&&world.has(eid,MeshInstance)&&world.has(eid,GlobalTransform)){
    const attached=world.storage(Shape).body.get(eid)||eid;
    if(world.resolve(body as Parameters<World['resolve']>[0])===attached){
     const source=sourceForMesh(world.storage(MeshInstance).mesh.get(eid));
     if(source?.cornerIndex&&source.cornerIndex.revision===source.revision){
      const s=world.storage(GlobalTransform).scale,scale=[s.x.get(eid),s.y.get(eid),s.z.get(eid)] as const;
      if(contactSourcePoint([0,0,0],scale))result={source,eid,lifetime:shape,scale};
     }
    }
   }
   bindings.set(shape,result);return result;
  };
  for(const pair of this.contacts.pairs){binding(pair.shapeA,pair.bodyA);binding(pair.shapeB,pair.bodyB);}
  // A source arriving/revising or an owner retiring also changes incidence,
  // even if the solver has not completed another step since the last frame.
  const signature=JSON.stringify([...bindings].map(([ref,b])=>[ref,b?.source.domain,b?.source.revision,b?.scale]));
  if(!changed&&signature===this.signature)return this.value;
  this.signature=signature;
  const corner=(binding:Binding,point:ContactVector,normal:ContactVector):SourceContactCorner|undefined=>{
   const local=contactSourcePoint(point,binding.scale);if(!local)return;
   const tolerance=Math.hypot(CONTACT_CORNER_TOLERANCE,CONTACT_NORMAL_SLAB)/Math.min(...binding.scale.map(Math.abs));
   const vertex=querySourceCorner(binding.source.cornerIndex,local,tolerance);
   if(vertex===undefined)return;
   // The KD radius is conservative under nonuniform scale; require the
   // actual candidate to satisfy the physical-meter tolerance on every axis.
   const p=binding.source.geometry.positions;
   const delta=binding.scale.map((s,k)=>(p[vertex*3+k]!-local[k]!)*s),along=delta.reduce((sum,d,k)=>sum+d*normal[k]!,0);
   if(Math.abs(along)>CONTACT_NORMAL_SLAB||Math.hypot(...delta.map((d,k)=>d-along*normal[k]!))>CONTACT_CORNER_TOLERANCE)return;
   return {domain:binding.source.domain,eid:binding.eid,lifetime:binding.lifetime,vertex};
  };
  const links:SourceContactLink[]=[];
  for(const pair of this.contacts.pairs){
   const a=bindings.get(pair.shapeA),b=bindings.get(pair.shapeB);if(!a||!b)continue;
   for(const manifold of pair.manifolds)for(const point of manifold.points){
    const ca=corner(a,point.a,manifold.normalA),cb=corner(b,point.b,manifold.normalB);if(ca&&cb)links.push({a:ca,b:cb});
   }
  }
  return this.patches.update(links);
 }
}
