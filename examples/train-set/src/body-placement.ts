import {composeMat4,multiplyMat4,decomposeMat4} from '@dylanebert/shallot';
import type {GltfSceneInstance} from '@dylanebert/shallot/gltf';

/** glTF's rigid nodes compose placement in poseGltfScene; skinned nodes use
 * their entity TRS directly. Update BOTH so rods never stay at the spawn point. */
export function placeBody(instance:GltfSceneInstance,body:Float32Array):void {
    composeMat4(body[0]!,body[1]!,body[2]!,body[3]!,body[4]!,body[5]!,body[6]!,1,1,1,instance.placement);
    for(let i=0;i<instance.gltf.primitives.length;i++)if(instance.gltf.primitives[i]!.skin>=0)instance.transforms[i]!.set(body);
}

/** Asset-space pivot to body-space placement. The unchanged scene clip stays
 * shared; each primitive belongs to exactly one physical assembly. */
export function bodyPlacement(body:Float32Array,pivot:readonly number[],out:Float32Array):void{
    composeMat4(body[0]!,body[1]!,body[2]!,body[3]!,body[4]!,body[5]!,body[6]!,1,1,1,out);
    for(let row=0;row<3;row++)out[12+row]-=out[row]!*pivot[0]!+out[4+row]!*pivot[1]!+out[8+row]!*pivot[2]!;
}
export function primitivePlacement(placement:Float32Array,global:Float32Array,skinned:boolean,out:Float32Array,scratch:Float32Array):void{
    if(skinned)scratch.set(placement);else multiplyMat4(placement,global,scratch);
    decomposeMat4(scratch,out);
}
