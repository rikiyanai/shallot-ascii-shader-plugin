import {GlobalTransform,Transform,type World} from '@dylanebert/shallot';
import {loadGltfScene,poseGltfScene,spawnGltfScene,type Gltf,type GltfSceneInstance} from '@dylanebert/shallot/gltf';
import {bodyPlacement,primitivePlacement} from './body-placement';
import asset from './locomotive-asset.json';

/** Scene asset/animation owner. Rendering stays in Shallot + the shared MAIN plugin. */
export class Locomotives {
    private readonly instances=new Map<number,{scene:GltfSceneInstance;time:number;tender:number;body:Float32Array;placements:[Float32Array,Float32Array];scratch:Float32Array}>();
    private readonly membership:number[];
    private readonly duration:number;
    private constructor(private readonly world:World,private readonly model:Gltf){
        this.membership=model.primitives.map(p=>{
            const tender=asset.bodies.tender.nodes.includes(p.node),head=asset.bodies.locomotive.nodes.includes(p.node);
            if(tender===head)throw Error('BR52 primitive must belong to exactly one assembly: '+p.node);
            return tender?1:0;
        });
        let duration=0;for(const sampler of model.animations[0]!.samplers)duration=Math.max(duration,sampler.input[sampler.input.length-1]!);
        this.duration=duration;
    }
    static async load(world:World,progress?:(value:number)=>void):Promise<Locomotives>{
        const model=await loadGltfScene(world,new URL('models/br52.glb',document.baseURI),progress);
        if(model.animations.length!==1)throw Error('BR52 requires its combined wheel-and-rod clip');
        return new Locomotives(world,model);
    }
    add(body:number,tender:number):void {this.instances.set(body,{scene:spawnGltfScene(this.world,this.model),time:0,tender,body:new Float32Array([0,0,0,0,0,0,1,1,1,1]),placements:[new Float32Array(16),new Float32Array(16)],scratch:new Float32Array(16)});this.pose(body,0,true);}
    remove(body:number):void {
        const instance=this.instances.get(body);if(!instance)return;
        for(const eid of [...instance.scene.entities,...instance.scene.lightEntities])this.world.destroy(eid);
        this.instances.delete(body);
    }
    pose(body:number,delta:number,initial=false):void {
        const instance=this.instances.get(body);if(!instance)return;
        const g=initial?this.world.storage(Transform):this.world.storage(GlobalTransform),p=g.translation,q=g.rotation;
        for(const [index,eid] of [body,instance.tender].entries()){
            const t=instance.body;t[0]=p.x.get(eid);t[1]=p.y.get(eid);t[2]=p.z.get(eid);
            t[3]=q.x.get(eid);t[4]=q.y.get(eid);t[5]=q.z.get(eid);t[6]=q.w.get(eid);
            bodyPlacement(t,index===0?asset.bodies.locomotive.pivot:asset.bodies.tender.pivot,instance.placements[index]!);
        }
        // poseGltfScene owns joint/morph playback. Keep base placement identity;
        // skinned primitive TRS is caller-owned and must be installed before it
        // writes the kernel pose. Rigid primitives compose globals below.
        for(let i=0;i<this.model.primitives.length;i++)if(this.model.primitives[i]!.skin>=0)
            primitivePlacement(instance.placements[this.membership[i]!]!,instance.scene.globals[0]!,true,instance.scene.transforms[i]!,instance.scratch);
        const animation=this.model.animations[0]!;
        instance.time=this.duration>0?(instance.time+delta)%this.duration:0;
        poseGltfScene(this.world,instance.scene,animation,instance.time);
        const storage=this.world.storage(Transform);
        for(let i=0;i<this.model.primitives.length;i++){
            const primitive=this.model.primitives[i]!;if(primitive.skin>=0)continue;
            const t=instance.scene.transforms[i]!,eid=instance.scene.entities[i]!;
            primitivePlacement(instance.placements[this.membership[i]!]!,instance.scene.globals[primitive.nodeIndex]!,false,t,instance.scratch);
            storage.translation.set(eid,t[0]!,t[1]!,t[2]!);storage.rotation.set(eid,t[3]!,t[4]!,t[5]!,t[6]!);storage.scale.set(eid,t[7]!,t[8]!,t[9]!);
        }
    }
}
