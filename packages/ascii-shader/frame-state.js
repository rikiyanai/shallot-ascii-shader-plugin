/** The inspection surface reports observations, never visual acceptance.
 * This does not participate in construction or choose a scene's drawing. */
// Older observations mixed legacy and compiled frames in the same slot.
// Do not migrate them: their producer cannot be recovered reliably.
const PREFIX='shallot-main-frame/v2/';
export function inspectionKey(url){
  const u=new URL(url,location.href);
  const scene=u.pathname.startsWith('/examples/')?u.pathname.replace(/\/$/,''):`map/${u.searchParams.get('map')??'game_map_y8'}`;
  const grammar=u.searchParams.get('grammar')==='legacy'?'legacy':'compiled';
  const requested=Number(u.searchParams.get('rows'));
  const rows=Number.isInteger(requested)&&requested>0?String(requested):'auto';
  // A reduced text matrix is not an observation of the default matrix,
  // just as the old CPU producer is not an observation of GPU playback.
  return `${scene}?grammar=${grammar}&rows=${rows}`;
}
export function frameState(controller){
  // Resident GPU publications are observable without reading their glyph
  // payload. CPU TextFrames exist only after an explicit export request.
  const resident=controller.residentFrames??new Map();
  const metadata=[...resident.values()];
  const frames=[...controller.frames].filter(([id])=>!resident.has(id)).map(([,frame])=>frame);
  const omissions=[...metadata,...frames].flatMap(f=>f.construction?.omitted??[]);
  // In-view sources still compiling: absent from the shown drawing until ready.
  const pending=[...metadata,...frames].flatMap(f=>f.construction?.pending??[]);
  const losses=[...metadata,...frames].map(f=>f.construction?.unresolvedWords);
  // Unknown legacy/stored observations are not evidence of zero loss.
  const unresolvedWords=losses.length&&losses.every(n=>Number.isFinite(n))?losses.reduce((n,value)=>n+value,0):undefined;
  const errors=[...controller.errors];
  const glyphs=metadata.reduce((n,f)=>n+f.glyphCount,0)+frames.reduce((n,f)=>n+f.chars.filter(c=>c[0]!==32&&c[0]!==0x3000).length,0),count=frames.length+metadata.length;
  return {at:Date.now(),frames:count,glyphs,omissions,pending,unresolvedWords,errors,
    state:errors.length?'ERROR':!count?'NO FRAME':!glyphs?'BLANK':omissions.length||pending.length?'INCOMPLETE':'DRAWN — unreviewed'};
}
export function stateText(s){
  if(!s)return 'MAIN · not inspected';
  const losses=s.unresolvedWords===undefined?'word-placement losses unreported':`${s.unresolvedWords} word-placement losses`;
  const pending=s.pending?.length?` · ${s.pending.length} sources pending`:'';
  return `MAIN · ${s.state} · ${s.glyphs} glyphs · ${s.omissions.length} omissions · ${losses}${pending}${s.errors.length?` · ${s.errors.at(-1)}`:''}`;
}
export function publishFrameState(controller,node){
  const s=frameState(controller);node.textContent=stateText(s);
  node.title=[...s.errors,...s.omissions,...s.pending.map(owner=>`pending: ${owner}`)].join('\n');
  try{localStorage.setItem(PREFIX+inspectionKey(location.href),JSON.stringify(s));}catch{/* Storage denial cannot stop the game. */}
  return s;
}
export function readFrameState(url){try{return JSON.parse(localStorage.getItem(PREFIX+inspectionKey(url))??'null');}catch{return null;}}
