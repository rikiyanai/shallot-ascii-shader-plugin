import font from './native-glyphs.json' with {type:'json'};
import {nativeScalarFont} from './native-font.ts';
import {NativePage} from './native-page.ts';
import {NativePresentation,type PresentedNativeFrame,type NativeSupport} from './native-presentation.ts';
import type {ArticulationResult} from './gpu-articulation.ts';
import type {ConstructionMetrics} from './native-construction.ts';
import type {PresentationMode} from './text/text-frame.ts';

/** Immutable display snapshot. Reading it is an operator export operation;
 * normal drawing uses results[].resident.tape and never calls this module. */
export interface ResidentNativeSnapshot {
 readonly results:readonly ArticulationResult[];
 readonly width:number;readonly height:number;readonly serial:number;
 readonly engineFrame:number;readonly mode:PresentationMode;readonly cpuEncodeMs:number;
 /** In-view source owners still compiling when this frame's job started: not
  * in this drawing, reported in its status, joining a later frame. */
 readonly pending?:readonly string[];
}
export interface ResidentFrameState {
 readonly frame:number;readonly glyphCount:number;
 readonly construction:{readonly sourceLines:number;readonly omitted:readonly string[];readonly pending:readonly string[];readonly unresolvedWords:number};
}
export function residentFrameState(snapshot:ResidentNativeSnapshot):ResidentFrameState {
 return {frame:snapshot.serial,glyphCount:snapshot.results.reduce((n,result)=>n+(result.resident?.glyphCount??0),0),
  construction:{sourceLines:snapshot.results.reduce((n,result)=>n+(result.sourceLines??0),0),omitted:snapshot.results.flatMap(result=>result.unresolved),
   pending:snapshot.pending??[],unresolvedWords:snapshot.results.reduce((n,result)=>n+(result.unresolvedWords??0),0)}};
}
export interface ResidentExportFrame extends PresentedNativeFrame {
 readonly construction:ResidentFrameState['construction']&{readonly metrics:ConstructionMetrics};
}
/** Decode source annotations on demand, with strict validation. Unlike the
 * withdrawn frame-time commitPlayback loop, export cannot discard a rejected
 * group, change a pen, or silently serialize a different drawing. */
export async function exportResidentFrame(snapshot:ResidentNativeSnapshot):Promise<ResidentExportFrame> {
 const started=performance.now();
 // Start all reads before yielding: each read pins that displayed tape even
 // if a new frame replaces it while the host encoder services this request.
 const reads=snapshot.results.map(result=>{
  if(!result.resident)throw Error('Displayed native snapshot has no resident tape');
  return result.resident.read();
 });
 const results=await Promise.all(reads),groups=results.flatMap(result=>result.groups);
 const glyphs=nativeScalarFont(font.glyphs);
 const page=new NativePage(glyphs,'preformatted').commit(groups.map(item=>item.group),snapshot.width,snapshot.height,snapshot.serial);
 const state=residentFrameState(snapshot);
 if(page.occurrences.length!==state.glyphCount)throw Error(`Native export differs from displayed tape: ${page.occurrences.length} vs ${state.glyphCount} glyphs`);
 const supports=new Map<string,NativeSupport>();
 for(const item of groups){
  if(!item.support)throw Error(`Native export lost source support ${item.group.id}`);
  supports.set(item.group.id,item.support);
 }
 // Fresh serialization has no presentation-history displacement. All pens
 // are the absolute native pens already painted by the GPU publication.
 const frame=new NativePresentation(glyphs,font.fontSha256).present(page,supports,snapshot.serial,snapshot.engineFrame,snapshot.mode);
 const served=new Set(groups.flatMap(item=>item.owners));
 return {...frame,construction:{...state.construction,metrics:{compositionMs:0,realizationMs:performance.now()-started,
  cpuFrameMs:snapshot.cpuEncodeMs,cpuFrameScope:'host source encoding only; native serialization is on-demand export, not frame work',
  subjects:served.size,sourceSubjects:served.size,farStripCount:0,publishedFarStripCount:0,layoutDroppedShare:null,
  dotShare:page.occurrences.length?page.occurrences.filter(atom=>/[.;:,]/u.test(atom.text)).length/page.occurrences.length:0,
  repeatedRampTokens:page.rows.reduce((n,row)=>n+(row.match(/;{3,}|:{3,}|\.{3,}/g)?.length??0),0),wholeEntries:['resident-compiled-gpu-tape']}}};
}
