import type {World} from '@dylanebert/shallot';
import * as Physics from '@dylanebert/shallot/standard/physics';
import {getBodyTransform,getMassData,type PhysicsContactData} from '@dylanebert/shallot/standard/physics';
import {contactBodyPoint,contactBodyVector,type ContactVector,type ContactRotation} from './contact-attachment.ts';
export interface PhysicsContactObservationStamp{readonly worldId:number;readonly generation:number;readonly stepIndex:number}
type ContactObserver=(world:World,previous:PhysicsContactObservationStamp|undefined,visit:(data:PhysicsContactData,a:number,b:number)=>void)=>PhysicsContactObservationStamp|undefined;
// The isolated example engine supplies the reproducible visitor patch. Other
// compatible hosts retain their incumbent incidence when the extension is absent.
const observePhysicsContacts=Reflect.get(Physics,'observePhysicsContacts') as ContactObserver|undefined;

export interface SourceContactPoint{
 readonly a:ContactVector;readonly b:ContactVector;
 readonly separation:number;readonly feature:number;
}
export interface SourceContactManifold{
 /** Directions local to each body, not stale screen/world-space normals. */
 readonly normalA:ContactVector;readonly normalB:ContactVector;readonly points:readonly SourceContactPoint[];
}
export interface SourceContactPair{
 /** All four references retain the owning World's generation bits. */
 readonly shapeA:number;readonly shapeB:number;readonly bodyA:number;readonly bodyB:number;
 readonly childIndex:number;readonly manifolds:readonly SourceContactManifold[];
}
interface BodyAttachment{readonly center:ContactVector;readonly rotation:ContactRotation}

/** Native Shallot solver incidence, shared by every scene with physics.
 * This observes actual touching pairs, not begin-event opt-ins, an all-pairs
 * geometry test or a screen weld. Records are copied once per completed
 * solver step; display jobs map references to their own current source owners.
 * Points remain BODY LOCAL so consumers use the same interpolated source/body
 * pose as the drawing, never frozen completed-step world-space positions.
 */
export class SourcePhysicsContacts{
 private stamp?:PhysicsContactObservationStamp;
 private current:readonly SourceContactPair[]=[];
 get pairs():readonly SourceContactPair[]{return this.current;}
 get step():PhysicsContactObservationStamp|undefined{return this.stamp;}

 update(world:World):boolean{
  if(!observePhysicsContacts)return false;
  let next:SourceContactPair[]|undefined,bodies:Map<number,BodyAttachment|undefined>|undefined;
  const attachment=(ref:number):BodyAttachment|undefined=>{
   bodies??=new Map();
   if(bodies.has(ref))return bodies.get(ref);
   const eid=world.resolve(ref as Parameters<World['resolve']>[0]);
   if(!eid){bodies.set(ref,undefined);return;}
   // Both engine calls return records with limited lifetimes. Copy before
   // another body's read can overwrite the borrowed transform.
   const pose=getBodyTransform(world,eid),mass=getMassData(world,eid);
   if(!pose||!mass){bodies.set(ref,undefined);return;}
   const result:BodyAttachment={center:[mass.center.x,mass.center.y,mass.center.z],rotation:{v:{x:pose.q.v.x,y:pose.q.v.y,z:pose.q.v.z},s:pose.q.s}};
   bodies.set(ref,result);return result;
  };
  const stamp=observePhysicsContacts(world,this.stamp,(data,bodyA,bodyB)=>{
   const a=attachment(bodyA),b=attachment(bodyB);if(!a||!b)return;
   const manifolds:SourceContactManifold[]=[];
   for(let m=0;m<data.manifoldCount;m++){
    const manifold=data.manifolds[m]!,points:SourceContactPoint[]=[];
    for(let p=0;p<manifold.pointCount;p++){
     const point=manifold.points[p]!;
     points.push({a:contactBodyPoint([point.anchorA.x,point.anchorA.y,point.anchorA.z],a.center,a.rotation),
      b:contactBodyPoint([point.anchorB.x,point.anchorB.y,point.anchorB.z],b.center,b.rotation),
      separation:point.separation,feature:point.featureId});
    }
   if(points.length){const normal=[manifold.normal.x,manifold.normal.y,manifold.normal.z] as const;manifolds.push({normalA:contactBodyVector(normal,a.rotation),normalB:contactBodyVector(normal,b.rotation),points});}
   }
   if(manifolds.length)(next??=[]).push({shapeA:data.shapeA,shapeB:data.shapeB,bodyA,bodyB,childIndex:data.childIndex,manifolds});
  });
  // The engine may return a fresh stamp record while skipping traversal.
  // Compare its identity, not JS object equality, or held frames erase pairs.
  const prior=this.stamp;
  if((!stamp&&!prior)||(stamp&&prior&&stamp.worldId===prior.worldId&&stamp.generation===prior.generation&&stamp.stepIndex===prior.stepIndex))return false;
  this.stamp=stamp;this.current=next??[];return true;
 }
}

