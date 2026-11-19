export interface InspectionState {at:number;frames:number;glyphs:number;omissions:string[];pending?:string[];errors:string[];state:string}
export interface InspectionController {readonly frames:ReadonlyMap<number,{readonly chars:readonly (readonly number[])[];readonly construction?:{readonly omitted:readonly string[];readonly pending?:readonly string[]}}> ;readonly residentFrames?:ReadonlyMap<number,{readonly glyphCount:number;readonly construction:{readonly omitted:readonly string[];readonly pending?:readonly string[]}}> ;readonly errors:readonly string[]}
export function inspectionKey(url:string):string;
export function frameState(controller:InspectionController):InspectionState;
export function stateText(state:InspectionState|null):string;
export function publishFrameState(controller:InspectionController,node:HTMLElement):InspectionState;
export function readFrameState(url:string):InspectionState|null;
