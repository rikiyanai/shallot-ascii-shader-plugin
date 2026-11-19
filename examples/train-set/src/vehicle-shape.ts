import locomotive from './locomotive-asset.json';

export interface VehicleShape {halfWidth:number;halfHeight:number;halfLength:number;offset:number}
const toy:VehicleShape={halfWidth:.5,halfHeight:.6,halfLength:.9,offset:.4};
const shape=(body:{halfExtents:number[];hullOffset:number}):VehicleShape=>({halfWidth:body.halfExtents[0]!,halfHeight:body.halfExtents[1]!,halfLength:body.halfExtents[2]!,offset:body.hullOffset});
const steam:VehicleShape=shape(locomotive.bodies.locomotive);
export const tenderShape:VehicleShape=shape(locomotive.bodies.tender);
export const steamCouplers={rear:locomotive.bodies.locomotive.coupler,front:locomotive.bodies.tender.coupler};
/** Guidance and physics share EACH authored rigid body's footprint. */
export const vehicleShape=(kind:string):VehicleShape=>kind==='steam'?steam:toy;
