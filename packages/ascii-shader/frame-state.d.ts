export interface InspectionState {at:number;frames:number;glyphs:number;omissions:string[];pending:string[];unresolvedWords?:number;pendingStage?:string;pendingElapsedMs?:number;errors:string[];state:string}
export interface InspectionController {
 readonly ascii?:boolean;
 readonly pending?:boolean;readonly pendingStage?:string;readonly pendingSince?:number;
 readonly frames:ReadonlyMap<number,{readonly chars:readonly (readonly number[])[];readonly construction?:{readonly omitted:readonly string[];readonly pending?:readonly string[];readonly unresolvedWords?:number}}>;
 readonly residentFrames?:ReadonlyMap<number,{readonly glyphCount:number;readonly construction:{readonly omitted:readonly string[];readonly pending?:readonly string[];readonly unresolvedWords?:number}}>;
 readonly errors:readonly string[];
}
export function inspectionKey(url:string):string;
export function frameState(controller:InspectionController,now?:number):InspectionState;
export function stateText(state:InspectionState|null):string;
export function handoffBootstrap(owner?:HTMLElement):void;
export function createFrameStateNotice(parent:HTMLElement,appearance?:(notice:HTMLElement)=>void):HTMLElement;
export function disposeFrameState(notice:HTMLElement|undefined):void;
export function publishFrameStateNotice(notice:HTMLElement,state:{state:'PENDING'|'ERROR';stage?:string;startedAt?:number;error?:unknown},now?:number):string;
export function publishFrameState(controller:InspectionController,node:HTMLElement,notice?:HTMLElement):InspectionState;
export function readFrameState(url:string):InspectionState|null;
