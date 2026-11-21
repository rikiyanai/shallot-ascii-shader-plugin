import locomotive from './locomotive-asset.json';

export interface VehicleShape {halfWidth:number;halfHeight:number;halfLength:number;offset:number}
const toy:VehicleShape={halfWidth:.5,halfHeight:.6,halfLength:.9,offset:.4};
/** Compact the two assemblies together; keep the authored .6 wheel gauge. */
export const steamModelScale=[1,.8,.8] as const;
export const steamGroundY=-.3;
const groundedY=(value:number)=>steamGroundY+(value-steamGroundY)*steamModelScale[1];
const shape=(body:{halfExtents:number[];hullOffset:number}):VehicleShape=>({halfWidth:body.halfExtents[0]!*steamModelScale[0],halfHeight:body.halfExtents[1]!*steamModelScale[1],halfLength:body.halfExtents[2]!*steamModelScale[2],offset:groundedY(body.hullOffset)});
const steam:VehicleShape=shape(locomotive.bodies.locomotive);
export const tenderShape:VehicleShape=shape(locomotive.bodies.tender);
const coupler=(point:number[])=>[point[0]!*steamModelScale[0],groundedY(point[1]!),point[2]!*steamModelScale[2]];
export const steamCouplers={rear:coupler(locomotive.bodies.locomotive.coupler),front:coupler(locomotive.bodies.tender.coupler)};
/** Guidance and physics share EACH authored rigid body's footprint. */
export const vehicleShape=(kind:string):VehicleShape=>kind==='steam'?steam:toy;
