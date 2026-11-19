import type {FormLodScope} from './form-lod-state.ts';

let nextWorld=1;
const UINT32_MAX=0xffffffff;
function nextId(value:number,label:string):number{
 if(!Number.isSafeInteger(value)||value<1||value>UINT32_MAX)throw Error(`Form LOD ${label} identity exhausted`);
 return value;
}

/** Host lifetime authority, independent of frame/field-ring slots.
 * Source domains are generation-owned and never recycled by the source cache.
 * Entity indices can be recycled: retain their last actual World.ref so a new
 * lifetime invalidates history rather than inheriting an old object's level.
 */
export class FormLodLifetimes{
 readonly worldId=nextId(nextWorld++,'world');
 private nextCamera=1;
 private epoch=1;
 private readonly cameras=new Map<unknown,{lifetime:unknown;id:number}>();
 private readonly entities=new Map<number,number>();
 private snapshot?:{frame:number;owners:Map<number,number>};

 /** The primary material phase and later construction phase must see one
  * owner epoch. Resolve it before either selects a representation, reusing
  * the host's existing per-frame entity scan rather than scanning per view. */
 ownersForFrame(frame:number,read:()=>Map<number,number>):Map<number,number>{
  if(this.snapshot?.frame===frame)return this.snapshot.owners;
  const owners=read();this.syncEntities(owners);this.snapshot={frame,owners};
  return owners;
 }

 syncEntities(owners:ReadonlyMap<number,number>):void{
  let reused=false;
  for(const [eid,lifetime]of owners){
   const old=this.entities.get(eid);
   if(old!==undefined&&old!==lifetime)reused=true;
   this.entities.set(eid,lifetime);
  }
  // Do not reset history merely because an unrelated new entity was created,
  // or because an owner is temporarily outside the camera.
  if(reused)this.epoch=nextId(this.epoch+1,'lifetime epoch');
 }

 scope(camera:unknown,lifetime:unknown,rowPixels:number,allowCompact=true):FormLodScope{
  let record=this.cameras.get(camera);
  if(!record||record.lifetime!==lifetime){
   record={lifetime,id:nextId(this.nextCamera++,'camera')};
   this.cameras.set(camera,record);
  }
  return {worldId:this.worldId,cameraId:record.id,epoch:this.epoch,rowPixels,allowCompact};
 }

 /** Explicit producer-lifetime reset when a host reuses its source/row IDs. */
 invalidateOwners():void{this.epoch=nextId(this.epoch+1,'lifetime epoch');}
}
