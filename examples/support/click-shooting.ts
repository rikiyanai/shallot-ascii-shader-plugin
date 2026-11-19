/** Optional gameplay extension, not a renderer branch. Original first-person
 * composition/control sources remain unchanged. Physics owns the hit; the
 * existing authored musket bank owns every explosion glyph and its timing.
 * Ordinary engine Text supplies the same effect to ASCII on AND off. */
import {Body,BodyType,Camera,CameraMode,Character,GlobalTransform,Shape,ShapeKind,Transform,Text,TextPlugin,type Plugin} from '@dylanebert/shallot';
import {getAngularVelocity,PhysicsEvents} from '@dylanebert/shallot/standard/physics';
import {composeGlobalTransform} from '@dylanebert/shallot/transform';
import {Billboard,Views,viewportToWorld} from '@dylanebert/shallot/rendering';
import {internText,registerFont} from '@dylanebert/shallot/extras';
import font from '../../packages/ascii-shader/native-glyphs.json';
import {nativeSpaceCodes} from '../../packages/ascii-shader/text/native-whitespace.ts';
import {impactBank,type ImpactBank,type ImpactAtom,type MusketBankJson} from '../../apps/game/musket-fire-bank.ts';

const advance=new Map(font.glyphs.map(g=>[g.text,g.advance]));
const URLS=[
 new URL('../../assets/animations/effects/musket-fire/v20261112turn01/small/musket-fire-small.json',import.meta.url),
 new URL('../../assets/animations/effects/musket-fire/v20261112turn01/medium/musket-fire-medium.json',import.meta.url),
 new URL('../../assets/animations/effects/musket-fire/v20261112turn01/large/musket-fire-large.json',import.meta.url),
];
interface Passage {text:string;dx:number;dy:number}
/** Join already-authored atoms only where native whitespace reproduces their
 * exact pens. An unrepresentable gap starts another anchored passage. */
export function impactPassages(atoms:readonly ImpactAtom[]):Passage[]{
 const result:Passage[]=[];let last:Passage|undefined,end=0;
 for(const atom of atoms){
  const gap=last&&last.dy===atom.dy?nativeSpaceCodes(Math.round(atom.dx-end),0,false,false,'preformatted'):undefined;
  if(last&&gap){last.text+=String.fromCodePoint(...gap)+atom.text;}
  else{last={...atom};result.push(last);}
  end=atom.dx+advance.get(atom.text)!;
 }
 return result;
}
interface Bank {bank:ImpactBank;frames:readonly (readonly Passage[])[]}
interface Effect {position:number[];direction:number[];born:number;scale:number;bank:Bank;entities:number[];frame:number}
interface PointerSample {x:number;y:number;started:number;moved:boolean}
interface Projectile {start:number[];target:number[];direction:number[];born:number;entity:number}
interface Impact {point:number[];normal:number[];approachSpeed:number;projectile:number;other:number;otherType:number;otherLinearVelocity:number[]|undefined;otherAngularVelocity:number[]|undefined;otherRotation:number[]|undefined}
/** The same authored orange used by the physics collision presentation. This
 * is a Text foreground, so ASCII-on and ASCII-off use one effect source. */
export const EXPLOSION_FOREGROUND=0xff8135;
/** Match the upstream continuous bullet-versus-stack density scale. */
export const CANNONBALL_RADIUS=0.25;
export const CANNONBALL_DENSITY=10000;
export const TAP_MAX_DISTANCE=10;
export const TAP_MAX_DURATION_MS=750;
export function isTap(start:PointerSample,endX:number,endY:number,ended:number):boolean{
 return !start.moved&&Math.hypot(endX-start.x,endY-start.y)<=TAP_MAX_DISTANCE&&ended-start.started<=TAP_MAX_DURATION_MS;
}
const stateKey={create:()=>({banks:[] as Bank[],effects:[] as Effect[],projectiles:[] as Projectile[],fire:0,firePoint:undefined as {x:number;y:number}|undefined,last:-Infinity,font:0,shots:0,hits:0,lastImpact:undefined as Impact|undefined,canvas:undefined as HTMLCanvasElement|undefined})};

export interface ClickShootingOptions {
 /** A composition may own gestures while reusing the same projectile/effect. */
 interactive?:boolean;
 onImpact?:(world:import('@dylanebert/shallot').World,hit:{target:number;point:readonly number[];direction:readonly number[]})=>void;
}
export function requestShot(world:import('@dylanebert/shallot').World,point?:{x:number;y:number}):void {
 if(world.time.paused)return;
 const state=world.resource(stateKey);state.fire++;state.firePoint=point;
}
export function clickShooting(options:ClickShootingOptions={}):Plugin{return {
 name:'AuthoredClickShooting',dependencies:[TextPlugin],
 async initialize(world){
  const state=world.resource(stateKey),canvas=document.querySelector('canvas');if(!canvas)return;state.canvas=canvas;
  state.font=registerFont(world,new URL('../../assets/font/Saitamaar.ttf',import.meta.url).href);
  state.banks=await Promise.all(URLS.map(async url=>{
   const response=await fetch(url);if(!response.ok)throw Error(`Explosion bank HTTP ${response.status}`);
   const bank=impactBank(await response.json() as MusketBankJson,text=>advance.get(text));
   if(bank.census.missing)throw Error('Explosion bank has unavailable native glyphs');
   return {bank,frames:bank.frames.flatMap(yaw=>yaw.map(impactPassages))};
  }));
  const abort=new AbortController(),pointers=new Map<number,PointerSample>(),fire=(point?:{x:number;y:number})=>{if(options.interactive!==false)requestShot(world,point);};
  // Keep the engine's existing pointer controls untouched: observe the same
  // stream and classify a short, stationary release as a shot. A drag remains
  // camera/character input and never fires.
  canvas.addEventListener('pointerdown',event=>{
   if(event.button!==0)return;
   pointers.set(event.pointerId,{x:event.clientX,y:event.clientY,started:performance.now(),moved:false});
  },{signal:abort.signal});
  canvas.addEventListener('pointermove',event=>{
   const start=pointers.get(event.pointerId);if(!start)return;
   if(Math.hypot(event.clientX-start.x,event.clientY-start.y)>TAP_MAX_DISTANCE)start.moved=true;
  },{signal:abort.signal});
  canvas.addEventListener('pointerup',event=>{
   const start=pointers.get(event.pointerId);pointers.delete(event.pointerId);if(!start||!isTap(start,event.clientX,event.clientY,performance.now()))return;
   const rect=canvas.getBoundingClientRect();fire({x:event.clientX-rect.left,y:event.clientY-rect.top});
  },{signal:abort.signal});
  canvas.addEventListener('pointercancel',event=>{pointers.delete(event.pointerId);},{signal:abort.signal});
  const button=document.createElement('button');button.textContent='Fire';button.setAttribute('aria-label','Fire authored explosion');
  button.style.cssText='position:fixed;right:24px;bottom:104px;z-index:1000;padding:12px;touch-action:none';
  button.addEventListener('pointerdown',event=>{event.preventDefault();fire();},{signal:abort.signal});if(options.interactive!==false)document.body.append(button);
  // Existing diagnostics may inspect gameplay state; no renderer state or
  // private display path is created here.
  Object.assign(globalThis,{__shallotShooting:state});
  world.onDispose(()=>{abort.abort();button.remove();state.canvas=undefined;if((globalThis as any).__shallotShooting===state)delete (globalThis as any).__shallotShooting;});
 },
 systems:[{name:'AuthoredClickShooting/update',group:'simulation',last:true,update(world){
  const state=world.resource(stateKey),canvas=state.canvas;if(!state.banks.length||!canvas)return;
  // Shadow/depth cameras can precede the player's camera in ECS order.
  // The shot belongs to the presenting view, never the first Camera entity.
  // `present` is frame scratch, cleared before simulation. Canvas membership
  // is the retained display binding and remains valid in this system group.
  const camera=[...world.resource(Views)].find(([,view])=>view.canvas)?.[0];if(camera===undefined)return;
  const matrix=new Float32Array(16);composeGlobalTransform(world,camera,matrix);
  const eye=[matrix[12]!,matrix[13]!,matrix[14]!],forward=[-matrix[8]!,-matrix[9]!,-matrix[10]!],right=[matrix[0]!,matrix[1]!,matrix[2]!],up=[matrix[4]!,matrix[5]!,matrix[6]!];
  const now=world.time.elapsed;
  // Physics events are published after the preceding fixed step. The sphere
  // owns the impact: a ray is used only to choose the tap's direction, never
  // to manufacture a hit or an explosion.
  const events=world.resource(PhysicsEvents).contactHit;
  for(let i=0;i<events.count;i++){
   const event=events.records[i]!,a=world.resolve(event.bodyA),b=world.resolve(event.bodyB);
   const projectile=state.projectiles.find(p=>p.entity===a||p.entity===b);
   if(!projectile)continue;
   const other=a===projectile.entity?b:a;
   if(other&&world.has(other,Character))continue;
   const point=[event.point.x,event.point.y,event.point.z],normal=[event.normal.x,event.normal.y,event.normal.z],otherType=other?world.storage(Body).type.get(other):-1;
   const global=world.storage(GlobalTransform),angular=otherType===BodyType.Dynamic?getAngularVelocity(world,other):null;
   state.hits++;state.lastImpact={point,normal,approachSpeed:event.approachSpeed,projectile:projectile.entity,other,otherType,otherLinearVelocity:otherType===BodyType.Dynamic?[global.linearVelocity.x.get(other),global.linearVelocity.y.get(other),global.linearVelocity.z.get(other)]:undefined,otherAngularVelocity:angular?[angular.x,angular.y,angular.z]:undefined,otherRotation:otherType===BodyType.Dynamic?[global.rotation.x.get(other),global.rotation.y.get(other),global.rotation.z.get(other),global.rotation.w.get(other)]:undefined};
   options.onImpact?.(world,{target:other,point,direction:projectile.direction});
   world.destroy(projectile.entity);state.projectiles.splice(state.projectiles.indexOf(projectile),1);
   const distance=Math.max(.1,Math.hypot(...point.map((v,k)=>v-eye[k]!))),view=world.resource(Views).get(camera),fov=world.storage(Camera).fov.get(camera)*Math.PI/180,scale=(view?.height??canvasHeight())/(2*Math.tan(fov/2)*distance);
   const bank=state.banks.reduce((a,b)=>Math.abs(Math.log(b.bank.pxPerUnit/scale))<Math.abs(Math.log(a.bank.pxPerUnit/scale))?b:a);
   state.effects.push({position:point,direction:projectile.direction,born:now,scale,bank,entities:[],frame:-1});
  }
  if(state.fire){const point=state.firePoint;state.fire=0;state.firePoint=undefined;if(now-state.last>=.4){
   state.last=now;state.shots++;
   const rect=canvas.getBoundingClientRect(),ray=viewportToWorld(world,camera,point?.x??rect.width/2,point?.y??rect.height/2),shotDirection=ray?.dir??forward;
   const origin=ray&&world.storage(Camera).mode.get(camera)===CameraMode.Orthographic?ray.origin:eye;
   const start=origin.map((v,k)=>v+shotDirection[k]!*0.8) as [number,number,number],target=origin.map((v,k)=>v+shotDirection[k]! * 80),projectile=world.create();
   world.add(projectile,Transform,{translation:start});
   world.add(projectile,Body,{type:BodyType.Dynamic,linearVelocity:[shotDirection[0]!*60,shotDirection[1]!*60,shotDirection[2]!*60],gravityScale:0,isBullet:1,enableSleep:0});
   world.add(projectile,Shape,{kind:ShapeKind.Sphere,radius:CANNONBALL_RADIUS,density:CANNONBALL_DENSITY,restitution:.25,friction:.2,enableHitEvents:1});
   world.add(projectile,Text,{font:state.font,billboard:Billboard.Screen,color:EXPLOSION_FOREGROUND});
   world.storage(Text).content.set(projectile,internText(world,'●'));
   state.projectiles.push({start,target,direction:[...shotDirection],born:now,entity:projectile});
  }}
  for(let n=state.projectiles.length-1;n>=0;n--){
   const projectile=state.projectiles[n]!,pose=world.storage(Transform).translation,position=[pose.x.get(projectile.entity),pose.y.get(projectile.entity),pose.z.get(projectile.entity)] as [number,number,number],progress=Math.min(1,(now-projectile.born)/4);
   const view=world.resource(Views).get(camera),fov=world.storage(Camera).fov.get(camera)*Math.PI/180,fromEye=Math.max(.1,Math.hypot(...position.map((v,i)=>v-eye[i]!))),scale=(view?.height??canvasHeight())/(2*Math.tan(fov/2)*fromEye);
   world.storage(Text).fontSize.set(projectile.entity,8/scale);
   if(progress<1)continue;
   world.destroy(projectile.entity);state.projectiles.splice(n,1);
  }
  for(let n=state.effects.length-1;n>=0;n--){
   const effect=state.effects[n]!,index=Math.floor((now-effect.born)*1000/effect.bank.bank.frameMs);
   if(index>=effect.bank.bank.frames[0]!.length){for(const eid of effect.entities)world.destroy(eid);state.effects.splice(n,1);continue;}
   const away=effect.direction.reduce((sum,v,k)=>sum+v*forward[k]!,0),side=effect.direction.reduce((sum,v,k)=>sum+v*right[k]!,0);
   const yaw=((Math.round(Math.atan2(away,side)/(Math.PI/8))%16)+16)%16;
   const frame=yaw*effect.bank.bank.frames[0]!.length+index,passages=effect.bank.frames[frame]!;
   while(effect.entities.length<passages.length){const eid=world.create();world.add(eid,Transform);world.add(eid,Text,{font:state.font,billboard:Billboard.Screen,color:EXPLOSION_FOREGROUND});effect.entities.push(eid);}
   for(let i=0;i<effect.entities.length;i++){
    const eid=effect.entities[i]!,passage=passages[i];world.storage(Text).opacity.set(eid,passage?1:0);if(!passage)continue;
    if(effect.frame!==frame)world.storage(Text).content.set(eid,internText(world,passage.text));
    world.storage(Text).fontSize.set(eid,16/effect.scale);
    // Text's anchor is its baseline/bottom. Its native renderer uses the
    // same one-row offset, so the authored bank's dx/dy stay intact.
    world.storage(Transform).translation.set(eid,...effect.position.map((v,k)=>v+(right[k]!*passage.dx-up[k]!*(passage.dy+17))/effect.scale) as [number,number,number]);
   }
   effect.frame=frame;
  }
 }}],
};}
function canvasHeight():number{return document.querySelector('canvas')?.height??476;}
