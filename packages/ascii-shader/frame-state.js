/** The inspection surface reports observations, never visual acceptance.
 * This does not participate in construction or choose a scene's drawing. */
// Older observations mixed legacy and compiled frames in the same slot.
// Do not migrate them: their producer cannot be recovered reliably.
// v2 mixed local scene paths with the public front-door fallback map key.
// Keep it intact for old observations, but never let those observations be
// read as a current route.
const PREFIX='shallot-main-frame/v3/';
const pendingSince=new WeakMap();
const MAX_NOTICE_ERROR_LENGTH=4096;
export function inspectionKey(url){
  const base=typeof location==='object'&&typeof location.href==='string'?location.href:'http://localhost/';
  const u=new URL(url,base);
  // Keep the actual route identity, including a publication base and commit.
  // Do not infer game_map_y8 for arbitrary plugin/example pages that have no
  // map query; the front door may inspect those pages independently.
  const pathname=u.pathname.replace(/\/+$/,'')||'/';
  const grammar=u.searchParams.get('grammar')==='legacy'?'legacy':'compiled';
  const requested=Number(u.searchParams.get('rows'));
  const rows=Number.isInteger(requested)&&requested>0?String(requested):'auto';
  // A reduced text matrix is not an observation of the default matrix,
  // just as the old CPU producer is not an observation of GPU playback.
  const map=u.searchParams.has('map')?`&map=${encodeURIComponent(u.searchParams.get('map')??'')}`:'';
  return `${pathname}?grammar=${grammar}&rows=${rows}${map}`;
}
function pendingElapsed(controller,now,pending){
  if(!pending){pendingSince.delete(controller);return undefined;}
  const ownerSince=Number(controller.pendingSince);
  if(Number.isFinite(ownerSince)){pendingSince.delete(controller);return Math.max(0,now-ownerSince);}
  let since=pendingSince.get(controller);
  if(since===undefined){since=now;pendingSince.set(controller,since);}
  return Math.max(0,now-since);
}
function elapsedText(milliseconds){return `${(Math.max(0,milliseconds??0)/1000).toFixed(1)}s`;}
function noticeErrorText(value){
 try{
  const message=value instanceof Error?value.message:String(value);
  if(!message)return 'Native text failed';
  return message.length>MAX_NOTICE_ERROR_LENGTH?message.slice(0,MAX_NOTICE_ERROR_LENGTH):message;
 }catch{return 'Native text failed';}
}
function noticeText(s){
  if(s.state==='PENDING')return `Loading native text…\n${s.pendingStage??'Waiting for first text frame'} · elapsed ${elapsedText(s.pendingElapsedMs)}`;
  if(s.state==='NO FRAME')return 'No text frame was published';
  if(s.state==='BLANK')return s.pendingStage?`No text drawn yet\n${s.pendingStage} · elapsed ${elapsedText(s.pendingElapsedMs)}`:'No text was drawn';
  if(s.state==='ERROR')return `MAIN · ERROR\n${noticeErrorText(s.errors.at(-1))}`;
  return '';
}
export function frameState(controller,now=Date.now()){
  // Resident GPU publications are observable without reading their glyph
  // payload. CPU TextFrames exist only after an explicit export request.
  const resident=controller.residentFrames??new Map();
  const metadata=[...resident.values()];
  const frames=[...controller.frames].filter(([id])=>!resident.has(id)).map(([,frame])=>frame);
  const omissions=[...metadata,...frames].flatMap(f=>f.construction?.omitted??[]);
  // In-view sources still compiling: absent from the shown drawing until ready.
  const pending=[...metadata,...frames].flatMap(f=>f.construction?.pending??[]);
  const losses=[...metadata,...frames].map(f=>f.construction?.unresolvedWords);

  const unresolvedWords=losses.length&&losses.every(n=>Number.isFinite(n))?losses.reduce((n,value)=>n+value,0):undefined;
  const errors=[...controller.errors];
  const glyphs=metadata.reduce((n,f)=>n+f.glyphCount,0)+frames.reduce((n,f)=>n+f.chars.filter(c=>c[0]!==32&&c[0]!==0x3000).length,0),count=frames.length+metadata.length;
  // A host must explicitly own the pending lifecycle (or publish pending
  // source metadata). Do not infer "loading" from an empty camera: that keeps
  // a known no-frame/empty-world state distinct from a genuine startup wait.
  const waiting=Boolean(controller.pending)||pending.length>0;
  const pendingElapsedMs=pendingElapsed(controller,now,waiting);
  const ownerStage=typeof controller.pendingStage==='string'&&controller.pendingStage.trim()?controller.pendingStage.trim():undefined;
  const pendingStage=waiting?(ownerStage??(!count?'Waiting for first text frame':'Waiting for updated text frame')):undefined;
  return {at:now,frames:count,glyphs,omissions,pending,unresolvedWords,errors,pendingStage,pendingElapsedMs,
    state:errors.length?'ERROR':waiting&&!count?'PENDING':!count?'NO FRAME':!glyphs?'BLANK':omissions.length||pending.length?'INCOMPLETE':'DRAWN — unreviewed'};
}
export function stateText(s){
  if(!s)return 'MAIN · not inspected';
  const losses=s.unresolvedWords===undefined?'word-placement losses unreported':`${s.unresolvedWords} word-placement losses`;
  const pending=s.pending?.length?` · ${s.pending.length} sources pending`:'';
  const stage=s.pendingStage?` · ${s.pendingStage}`:'';
  const elapsed=s.pendingElapsedMs===undefined?'':` · elapsed ${elapsedText(s.pendingElapsedMs)}`;
  return `MAIN · ${s.state} · ${s.glyphs} glyphs · ${s.omissions.length} omissions · ${losses}${pending}${stage}${elapsed}${s.errors.length?` · ${noticeErrorText(s.errors.at(-1))}`:''}`;
}
/** Yield the pre-module bootstrap surface to the real MAIN notice owner. */
export function handoffBootstrap(owner){
  try{globalThis.__shallotBootstrap?.handoff?.(owner);}catch{/* A stale static page cannot block the real owner. */}
}
export function createFrameStateNotice(parent,appearance){
  const notice=parent.ownerDocument.createElement('div');
  notice.setAttribute('role','status');notice.setAttribute('aria-live','polite');notice.hidden=true;
  notice.style.cssText='position:fixed;left:max(8px,env(safe-area-inset-left));right:max(8px,env(safe-area-inset-right));top:max(8px,env(safe-area-inset-top));z-index:10001;box-sizing:border-box;padding:10px 12px;border:2px solid #000;background:#fff;color:#000;font-size:clamp(18px,4vw,28px);line-height:1.35;max-width:min(calc(100vw - 16px),640px);white-space:pre-line;overflow-wrap:anywhere;pointer-events:none';
  appearance?.(notice);
  // The shared Xanh appearance sets a 14px font shorthand. Restore the
  // loading notice's readable size AFTER that shorthand, in this one owner.
  notice.style.fontSize='clamp(18px,4vw,28px)';notice.style.lineHeight='1.35';
  parent.append(notice);return notice;
}
export function disposeFrameState(notice){notice?.remove();}
function setNotice(notice,text,error=false,hidden=!text){
 notice.textContent=text;notice.hidden=hidden;notice.title=text?text:'';
 // Pending/drawn notices stay a pass-through overlay. A bounded error may
 // contain compiler locations; make that same exact text selectable and
 // scrollable on a short phone viewport without adding normal UI clutter.
 notice.style.pointerEvents=error?'auto':'none';
 notice.style.maxHeight=error?'calc(100dvh - 16px)':'';
 notice.style.overflow=error?'auto':'';
 if(!hidden)handoffBootstrap(notice);
}
/** Publish the same external notice before a renderer controller exists.
 * The stage is supplied by the awaited preparation owner; elapsed is wall time
 * from the caller's startup boundary, never an inferred percentage or ETA. */
export function publishFrameStateNotice(notice,{state,stage,startedAt,error},now=Date.now()){
 if(!notice)return '';
 const elapsed=startedAt===undefined?undefined:Math.max(0,now-startedAt);
 const text=state==='ERROR'?`MAIN · ERROR\n${noticeErrorText(error)}`:`Loading native text…\n${stage??'Preparing MAIN GPU programs'} · elapsed ${elapsedText(elapsed)}`;
 setNotice(notice,text,state==='ERROR',false);return text;
}
export function publishFrameState(controller,node,notice){
  const s=frameState(controller);node.textContent=stateText(s);
  node.title=[stateText(s),...s.errors.map(noticeErrorText),...s.omissions,...s.pending.map(owner=>`pending: ${owner}`)].join('\n');
  if(notice){const text=controller.ascii===false&&s.state!=='ERROR'?'':noticeText(s);setNotice(notice,text,s.state==='ERROR'&&!!text,!text);if(text)notice.title=stateText(s);}
  try{localStorage.setItem(PREFIX+inspectionKey(location.href),JSON.stringify(s));}catch{/* Storage denial cannot stop the game. */}
  return s;
}
export function readFrameState(url){try{return JSON.parse(localStorage.getItem(PREFIX+inspectionKey(url))??'null');}catch{return null;}}
