import {Body,BodyType,Camera,CameraMode,DirectionalLight,eulerToQuat,GlobalAmbientLight,GlobalTransform,Hulls,lookAtRotation,Materials,Meshes,MeshInstance,MeshMaterial,Shape,ShapeKind,SphericalJoint,StandardMaterial,StandardRenderer,Transform,type MeshHandle,type Plugin,type World} from '@dylanebert/shallot';
import {Orbit,OrbitMode,OrbitPlugin} from '@dylanebert/shallot/extras';
import {GltfPlugin} from '@dylanebert/shallot/gltf';
import {viewportToWorld} from '@dylanebert/shallot/rendering';
import {applyLinearImpulseToCenter,castRayClosest,emptyQueryHit,setAngularVelocity,setBodyType,setLinearVelocity,setTargetTransform,setTransform,StandardPhysicsPlugin,StepPhysicsSystem,SyncPhysicsConstraintsSystem} from '@dylanebert/shallot/standard/physics';
import type {MaterialHandle} from '@dylanebert/shallot/standard/rendering';
import {clickShooting,requestShot} from '../../click-shooting';
import {mountTrainSetControls,type Item,type Pick} from './controls';
import {trackGeometry,trackNetworkGeometry,trainGeometry,type Geometry} from './geometry';
import {cellKey,Railway,TILE,TRAIN_RAIL_LIFT,type Cell,type Point,type Train,type TrainKind} from './railway';
import {Locomotives} from './locomotive';
import {vehicleShape,tenderShape,steamCouplers} from './vehicle-shape';
import type {RailPose} from './train-trail';

type TileBodies={ground?:number;rail?:number;mesh?:MeshHandle;shapes?:number[];drawing?:Geometry};
type Owner={kind:'terrain'|'track';cell:Cell}|{kind:'train';id:number};
interface State {
    railway:Railway;camera:number;world:World;tiles:Map<string,TileBodies>;owners:Map<number,Owner>;
    vehicles:Map<number,number>;meshes:Map<TrainKind,MeshHandle>;
    attachments:Map<number,{tender:number;joint:number}>;tenderHull:number;
    materials:MaterialHandle[];unitHull:number;trainHulls:Map<TrainKind,number>;locomotives:Locomotives;railDrawing?:{entity:number;mesh:MeshHandle};
    removeControls?:()=>void;hits:number;status?:HTMLOutputElement;statusText?:string;
}
const TrainSet={create:()=>({} as State)};

function entity(state:State,mesh:MeshHandle|undefined,position:Point,scale:Point,material:number,type:number=BodyType.Static,hull=state.unitHull):number {
    const {world}=state,eid=world.create();
    world.add(eid,Transform,{translation:position,scale});
    world.add(eid,MeshInstance,mesh===undefined?{}:{mesh});
    world.add(eid,MeshMaterial,state.materials[material]!);
    world.add(eid,Body,{type,enableSleep:type===BodyType.Static?1:0});
    world.add(eid,Shape,{kind:ShapeKind.Hull,geometry:hull,density:400,friction:.6,restitution:.15,enableHitEvents:1});
    return eid;
}
function destroy(state:State,eid:number|undefined):void {if(eid!==undefined){
    const attachment=state.attachments?.get(eid);
    if(attachment){state.world.destroy(attachment.joint);state.owners.delete(attachment.tender);state.world.destroy(attachment.tender);state.attachments.delete(eid);}
    state.locomotives?.remove(eid);state.owners.delete(eid);state.world.destroy(eid);
}}
function syncTiles(state:State):void {
    const {railway,world}=state;let railsChanged=false;
    for(const key of railway.dirty){
        const [x,z]=key.split(',').map(Number) as [number,number];if(!railway.contains(x,z))continue;
        const old=state.tiles.get(key);if(old?.drawing)railsChanged=true;if(old){for(const shape of old.shapes??[])destroy(state,shape);destroy(state,old.ground);destroy(state,old.rail);if(old.mesh!==undefined)world.resource(Meshes).remove(old.mesh);}
        const bodies:TileBodies={},height=railway.height(x,z);
        if(height>=0){
            bodies.ground=entity(state,undefined,[x*TILE,(height-1)/2,z*TILE],[TILE,height+1,TILE],0);
            state.owners.set(bodies.ground,{kind:'terrain',cell:[x,z]});
        }
        const track=railway.tracks.get(key);
        if(track){
            const {colliders,...geometry}=trackGeometry(railway,track);
            if(track.kind==='turntable'){
                const {construction:pendingConstruction,...native}=geometry;
                bodies.mesh=world.resource(Meshes).add({name:'Turntable',...native});
            }
            else {bodies.drawing=geometry;railsChanged=true;}
            bodies.rail=entity(state,bodies.mesh,[x*TILE,height,z*TILE],[1,1,1],1);
            // An empty MeshInstance selects the engine's default cube. Static
            // rail bodies are physics-only; the welded network draws them.
            if(bodies.mesh===undefined){world.remove(bodies.rail,MeshInstance);world.remove(bodies.rail,MeshMaterial);}
            // The same authored segments supply visual geometry and physical
            // support, including curve and slope pitch. No flat proxy ramp.
            world.remove(bodies.rail,Shape);bodies.shapes=[];
            for(const part of colliders){
                const shape=world.create();world.add(shape,Shape,{body:bodies.rail,kind:ShapeKind.Hull,geometry:state.unitHull,
                    scale:part.size,hullPosition:part.center,hullRotation:eulerToQuat(part.pitch,part.yaw,0),friction:.6,enableHitEvents:1});
                bodies.shapes.push(shape);
            }
            state.owners.set(bodies.rail,{kind:'track',cell:[x,z]});
        }
        state.tiles.set(key,bodies);
    }
    railway.dirty.clear();
    if(railsChanged){
        if(state.railDrawing){world.destroy(state.railDrawing.entity);world.resource(Meshes).remove(state.railDrawing.mesh);state.railDrawing=undefined;}
        const pieces:{geometry:Geometry;position:Point}[]=[];
        for(const [key,tile] of state.tiles)if(tile.drawing){
            const [x,z]=key.split(',').map(Number) as [number,number];
            pieces.push({geometry:tile.drawing,position:[x*TILE,railway.height(x,z),z*TILE]});
        }
        if(pieces.length){
            // PATH/PAIR authoring is retained on the geometry. Do not replace
            // native mesh fallback until the generic consumer draws these
            // declarations, including all non-path station/switch geometry.
            const {construction:pendingConstruction,...network}=trackNetworkGeometry(pieces);
            const mesh=world.resource(Meshes).add({name:'Rail network',...network}),drawing=world.create();
            world.add(drawing,Transform);world.add(drawing,MeshInstance,{mesh});world.add(drawing,MeshMaterial,state.materials[1]!);
            state.railDrawing={entity:drawing,mesh};
        }
    }
}
function spawnTrain(state:State,kind:TrainKind,cell:Cell):void {
    const train=state.railway.addTrain(kind,...cell);if(!train)return;
    const p=train.position.slice() as Point;p[1]+=TRAIN_RAIL_LIFT;
    const eid=entity(state,state.meshes.get(kind),p,[1,1,1],kind==='steam'?2:kind==='passenger'?3:4,BodyType.Kinematic,state.trainHulls.get(kind)!);
    state.world.storage(Transform).rotation.set(eid,...rotation(train));
    if(kind==='steam'){
        state.world.remove(eid,MeshInstance);state.world.remove(eid,MeshMaterial);
        const tail=train.trailer!,tp=tail.position.slice() as Point;tp[1]+=TRAIN_RAIL_LIFT;
        const tender=entity(state,undefined,tp,[1,1,1],2,BodyType.Kinematic,state.tenderHull);
        state.world.remove(tender,MeshInstance);state.world.remove(tender,MeshMaterial);
        state.world.storage(Transform).rotation.set(tender,...rotation(tail));
        const joint=state.world.create();state.world.add(joint,SphericalJoint,{a:eid,b:tender,localAnchorA:steamCouplers.rear,localAnchorB:steamCouplers.front,collideConnected:0,drawScale:0});
        state.attachments.set(eid,{tender,joint});state.owners.set(tender,{kind:'train',id:train.id});state.locomotives.add(eid,tender);
    }
    state.vehicles.set(train.id,eid);state.owners.set(eid,{kind:'train',id:train.id});
}
function rotation(train:RailPose):[number,number,number,number] {
    return eulerToQuat(-Math.atan2(train.tangent[1],Math.hypot(train.tangent[0],train.tangent[2])),Math.atan2(train.tangent[0],train.tangent[2]),0) as [number,number,number,number];
}
function eachBody(state:State,train:Train,visit:(eid:number,pose:RailPose)=>void):void{
    const eid=state.vehicles.get(train.id);if(eid===undefined)return;visit(eid,train);
    const attachment=state.attachments.get(eid);if(attachment&&train.trailer)visit(attachment.tender,train.trailer);
}
function releaseTrain(state:State,train:Train,impulse?:Point,hitBody?:number):void {
    const eid=state.vehicles.get(train.id);if(eid===undefined)return;
    state.railway.derail(train);eachBody(state,train,(body,pose)=>{
        setBodyType(state.world,body,BodyType.Dynamic);
        setLinearVelocity(state.world,body,{x:pose.tangent[0]*train.speed,y:pose.tangent[1]*train.speed,z:pose.tangent[2]*train.speed});
    });
    if(impulse){const target=hitBody??eid;applyLinearImpulseToCenter(state.world,target,{x:impulse[0],y:impulse[1],z:impulse[2]});setAngularVelocity(state.world,target,{x:impulse[2]*.03,y:.5,z:-impulse[0]*.03});}
}
function ray(state:State,x:number,y:number){
    const canvas=document.querySelector('canvas')!,r=canvas.getBoundingClientRect();
    return viewportToWorld(state.world,state.camera,x-r.left,y-r.top);
}
function plane(state:State,x:number,y:number,height:number):Point|undefined {
    const r=ray(state,x,y);if(!r||Math.abs(r.dir[1])<1e-6)return;
    const t=(height-r.origin[1])/r.dir[1];if(t<0)return;
    return r.origin.map((v,i)=>v+r.dir[i]!*t) as Point;
}
function pick(state:State,x:number,y:number):Pick|undefined {
    const r=ray(state,x,y);if(!r)return;
    const out=emptyQueryHit();
    if(castRayClosest(state.world,{x:r.origin[0],y:r.origin[1],z:r.origin[2]},{x:r.dir[0]*300,y:r.dir[1]*300,z:r.dir[2]*300},out)){
        const owner=state.owners.get(out.body);
        const point:Point=[out.point.x,out.point.y,out.point.z];
        if(owner?.kind==='train')return {kind:'train',id:owner.id,cell:[Math.round(point[0]/TILE),Math.round(point[2]/TILE)],point};
        if(owner)return {kind:owner.kind,cell:owner.cell,point,trackKind:owner.kind==='track'?state.railway.tracks.get(cellKey(...owner.cell))?.kind:undefined};
    }
    const point=plane(state,x,y,0);if(!point)return;
    return {kind:'empty',cell:[Math.round(point[0]/TILE),Math.round(point[2]/TILE)],point};
}
function paint(state:State,item:Item,cell:Cell):void {
    const r=state.railway;if(!r.contains(...cell))return;
    if(item==='terrain')r.setHeight(...cell,r.height(...cell)+1);
    else if(item==='lower-terrain')r.setHeight(...cell,r.height(...cell)-1);
    else if(['steam','passenger','metro'].includes(item))spawnTrain(state,item as TrainKind,cell);
    else r.placeTrack(...cell,item as 'track'|'switch'|'station'|'turntable');
}
function reset(state:State):void {
    for(const eid of state.vehicles.values())destroy(state,eid);state.vehicles.clear();
    state.railway.reset();syncTiles(state);
    const o=state.world.storage(Orbit),canvas=document.querySelector<HTMLCanvasElement>('canvas')!;
    o.pan.set(state.camera,0,0,0);o.yaw.set(state.camera,Math.PI/4);o.pitch.set(state.camera,Math.atan(1/Math.sqrt(2)));
    o.size.set(state.camera,Math.max(19,19/Math.min(1,canvas.clientWidth/canvas.clientHeight)));
    spawnTrain(state,'steam',[0,-3]);
}
function shoot(state:State,x:number,y:number):void {
    const canvas=document.querySelector('canvas')!,rect=canvas.getBoundingClientRect();
    requestShot(state.world,{x:x-rect.left,y:y-rect.top});
}

const shooting=clickShooting({interactive:false,onImpact(world,hit){
    const state=world.resource(TrainSet),owner=state.owners.get(hit.target);if(!owner)return;state.hits++;
    if(owner.kind==='train'){const train=state.railway.trains.get(owner.id);if(train)releaseTrain(state,train,hit.direction.map(v=>v*750) as Point,hit.target);}
    else if(owner.kind==='track')state.railway.removeTrack(...owner.cell);
    else state.railway.setHeight(...owner.cell,state.railway.height(...owner.cell)-1);
}});

export const AppPlugin:Plugin={
    name:'Train set',dependencies:[OrbitPlugin,StandardPhysicsPlugin,GltfPlugin,shooting],
    async initialize(world,progress){
        const state=world.resource(TrainSet);Object.assign(state,{world,railway:new Railway(),tiles:new Map(),owners:new Map(),vehicles:new Map(),attachments:new Map(),meshes:new Map(),hits:0});
        state.materials=[[.32,.42,.22,1],[.3,.28,.25,1],[.16,.19,.21,1],[.58,.2,.12,1],[.12,.38,.49,1],[1,.32,.045,1]].map(baseColor=>world.resource(Materials).add(StandardMaterial({baseColor:baseColor as [number,number,number,number],perceptualRoughness:.75})));
        state.unitHull=world.resource(Hulls).add({name:'Tile',halfExtents:[.5,.5,.5]});
        state.trainHulls=new Map();
        for(const kind of ['steam','passenger','metro'] as TrainKind[]){
            const shape=vehicleShape(kind);
            state.trainHulls.set(kind,world.resource(Hulls).add({name:kind,halfExtents:[shape.halfWidth,shape.halfHeight,shape.halfLength],transform:{position:[0,shape.offset,0],rotation:[0,0,0,1]}}));
            if(kind!=='steam')state.meshes.set(kind,world.resource(Meshes).add({name:kind,...trainGeometry(kind)}));
        }
        state.tenderHull=world.resource(Hulls).add({name:'Tender',halfExtents:[tenderShape.halfWidth,tenderShape.halfHeight,tenderShape.halfLength],transform:{position:[0,tenderShape.offset,0],rotation:[0,0,0,1]}});
        state.locomotives=await Locomotives.load(world,progress);
        Object.assign(world.resource(GlobalAmbientLight),{color:0xd8e4ff,brightness:400});
        const sun=world.create();world.add(sun,Transform,{rotation:lookAtRotation(0,0,0,1,-2,-.35)});world.add(sun,DirectionalLight,{color:0xffe8cc,illuminance:6000,shadowMapsEnabled:1,maximumDistance:100});
        const canvas=document.querySelector<HTMLCanvasElement>('canvas')!;
        const initialSize=Math.max(19,19/Math.min(1,canvas.clientWidth/canvas.clientHeight));
        state.camera=world.create();world.add(state.camera,Transform);world.add(state.camera,Camera,{mode:CameraMode.Orthographic,near:.1,far:200,size:initialSize,clearColor:0xf9f6ec});world.add(state.camera,StandardRenderer);
        world.add(state.camera,Orbit,{mode:OrbitMode.Manual,distance:60,yaw:Math.PI/4,pitch:Math.atan(1/Math.sqrt(2)),size:initialSize,minSize:5,maxSize:80,minPitch:25*Math.PI/180,maxPitch:70*Math.PI/180});
        reset(state);
        state.removeControls=mountTrainSetControls(canvas,{
            pick:(x,y)=>pick(state,x,y),plane:(x,y,h)=>plane(state,x,y,h),paint:(item,cell)=>paint(state,item,cell),erase:(cell,id)=>{
                const eid=id===undefined?undefined:state.vehicles.get(id);
                // A selected vehicle may have fallen out of the world while
                // its menu was open. Its stale id must never erase the ground.
                if(id!==undefined){destroy(state,eid);state.vehicles.delete(id);state.railway.removeTrain(id);}
                else state.railway.erase(...cell);
            },
            shoot:(x,y)=>shoot(state,x,y),operate:cell=>{state.railway.toggle(...cell);},
            orbit:(dx,dy,zoom)=>{const o=world.storage(Orbit);o.yaw.set(state.camera,o.yaw.get(state.camera)-dx*.006);o.pitch.set(state.camera,Math.max(.436,Math.min(1.22,o.pitch.get(state.camera)+dy*.006)));o.size.set(state.camera,Math.max(5,Math.min(80,o.size.get(state.camera)*zoom)));},
            pan:(dx,dy)=>{
                const o=world.storage(Orbit),yaw=o.yaw.get(state.camera),pitch=o.pitch.get(state.camera);
                const scale=2*o.size.get(state.camera)/Math.max(1,canvas.clientHeight),depth=dy/Math.max(.1,Math.sin(pitch));
                o.pan.x.set(state.camera,o.pan.x.get(state.camera)-scale*(dx*Math.cos(yaw)+depth*Math.sin(yaw)));
                o.pan.z.set(state.camera,o.pan.z.get(state.camera)+scale*(dx*Math.sin(yaw)-depth*Math.cos(yaw)));
            },
            reset:()=>reset(state),pause:()=>state.railway.running=!state.railway.running,
            move:(hit,point,velocity)=>{
                const cell:Cell=[Math.round(point[0]/TILE),Math.round(point[2]/TILE)];
                if(hit.kind==='train'&&hit.id!==undefined){
                    const train=state.railway.trains.get(hit.id),eid=state.vehicles.get(hit.id);if(!train||eid===undefined)return;
                    state.railway.derail(train);
                    const offset=point.map((v,k)=>v-hit.point[k]!),g=world.storage(GlobalTransform);
                    eachBody(state,train,body=>{const at:Point=[g.translation.x.get(body)+offset[0]!,g.translation.y.get(body)+offset[1]!,g.translation.z.get(body)+offset[2]!];const q:[number,number,number,number]=[g.rotation.x.get(body),g.rotation.y.get(body),g.rotation.z.get(body),g.rotation.w.get(body)];setBodyType(world,body,BodyType.Kinematic);setTransform(world,body,at,q);});
                    if(!state.railway.contains(...cell)){destroy(state,eid);state.vehicles.delete(train.id);state.railway.removeTrain(train.id);}
                    else if(state.railway.rerail(train,...cell))eachBody(state,train,(body,pose)=>{const at=pose.position.slice() as Point;at[1]+=TRAIN_RAIL_LIFT;setBodyType(world,body,BodyType.Kinematic);setTransform(world,body,at,rotation(pose));});
                    else {releaseTrain(state,train);eachBody(state,train,body=>setLinearVelocity(world,body,{x:Math.max(-25,Math.min(25,velocity[0])),y:velocity[1],z:Math.max(-25,Math.min(25,velocity[2]))}));}
                }else if(state.railway.contains(...cell)&&cellKey(...cell)!==cellKey(...hit.cell)){
                    if(hit.kind==='track')state.railway.moveTrack(hit.cell,cell);
                    else if(hit.kind==='terrain'&&state.railway.height(...cell)<4&&state.railway.height(...hit.cell)>=0){state.railway.setHeight(...cell,state.railway.height(...cell)+1);state.railway.setHeight(...hit.cell,state.railway.height(...hit.cell)-1);}
                }
            },
        });
        state.status=document.querySelector<HTMLOutputElement>('#train-set-status')!;
        world.onDispose(()=>state.removeControls?.());
    },
    systems:[{
        name:'Train set edits',group:'simulation',update(world){
            const state=world.resource(TrainSet);syncTiles(state);
            const guided=[...state.railway.trains.values()].filter(t=>t.guided).length;
            const stopped=[...state.railway.trains.values()].filter(t=>t.guided&&t.blocked).length;
            const text=`${state.railway.tracks.size} track tiles · ${state.vehicles.size} trains · ${state.vehicles.size-guided} derailed${stopped?' · '+stopped+' stopped':''}`;
            if(state.status&&text!==state.statusText){state.status.textContent=text;state.statusText=text;}
        },
    },{
        name:'Train set guidance',group:'fixed',after:[SyncPhysicsConstraintsSystem],before:[StepPhysicsSystem],update(world){
            const state=world.resource(TrainSet),before=new Set([...state.railway.trains.values()].filter(t=>t.guided).map(t=>t.id));
            state.railway.step(1/60);
            for(const track of state.railway.tracks.values())if(track.kind==='turntable'){
                const eid=state.tiles.get(cellKey(track.x,track.z))?.rail;
                if(eid!==undefined)setTransform(world,eid,[track.x*TILE,state.railway.height(track.x,track.z),track.z*TILE],eulerToQuat(0,track.turn*Math.PI/2,0));
            }
            for(const train of state.railway.trains.values()){
                const eid=state.vehicles.get(train.id);if(eid===undefined)continue;
                if(!train.guided){if(before.has(train.id))releaseTrain(state,train);continue;}
                eachBody(state,train,(body,pose)=>{const position=pose.position.slice() as Point;position[1]+=TRAIN_RAIL_LIFT;setTargetTransform(world,body,position,rotation(pose),1/60);});
            }
        },
    },{
        name:'Train set collisions',group:'fixed',after:[StepPhysicsSystem],update(world){
            const state=world.resource(TrainSet);
            for(const [id,eid] of state.vehicles){
                const p=world.storage(GlobalTransform).translation;
                if(p.y.get(eid)<-25){destroy(state,eid);state.vehicles.delete(id);state.railway.removeTrain(id);continue;}
                const train=state.railway.trains.get(id);
                if(train?.kind==='steam')state.locomotives.pose(eid,train.guided&&!train.blocked&&state.railway.running?1/60:0);
            }
        },
    }],
};
