import {
 SMALL_CONSTRUCTION_TABLE_AT,SMALL_CONSTRUCTION_TABLE_VERSION,SMALL_CONSTRUCTION_TABLE_HEADER_WORDS,
 SMALL_CONSTRUCTION_TABLE_RECORD_WORDS,SMALL_CONSTRUCTION_TABLE_WIDTH_LIMIT,SMALL_CONSTRUCTION_TABLE_KIND_COUNT,
 SMALL_CONSTRUCTION_TABLE_END_COUNT,SMALL_CONSTRUCTION_TABLE_ROW_COUNT,SMALL_CONSTRUCTION_TABLE_KEY_COUNT,
 SMALL_CONSTRUCTION_TABLE_INDEX_AT,SMALL_CONSTRUCTION_TABLE_INDEX_WORDS,SMALL_CONSTRUCTION_TABLE_RECORDS_AT,
 SMALL_CONSTRUCTION_TABLE_NONE,
} from './role-grammar-format.ts';
export {SMALL_CONSTRUCTION_TABLE_AT,SMALL_CONSTRUCTION_TABLE_VERSION,SMALL_CONSTRUCTION_TABLE_HEADER_WORDS,
 SMALL_CONSTRUCTION_TABLE_RECORD_WORDS,SMALL_CONSTRUCTION_TABLE_WIDTH_LIMIT,SMALL_CONSTRUCTION_TABLE_KIND_COUNT,
 SMALL_CONSTRUCTION_TABLE_END_COUNT,SMALL_CONSTRUCTION_TABLE_ROW_COUNT,SMALL_CONSTRUCTION_TABLE_KEY_COUNT,
 SMALL_CONSTRUCTION_TABLE_INDEX_AT,SMALL_CONSTRUCTION_TABLE_INDEX_WORDS,SMALL_CONSTRUCTION_TABLE_RECORDS_AT,
 SMALL_CONSTRUCTION_TABLE_NONE};

/**
 * GPU playback ABI for the compiled smallest-tier construction bank.
 *
 * The bank is a suffix of the resident far-strip grammar.  Its eight-word
 * header is `[version, stride, count, recordsBase, indexBase, widthLimit,
 * kindCount, rowCount]`, all offsets relative to the suffix header.  The
 * index is dense over `(kind/ends, rows, integer width)`; each cell stores a
 * record ordinal.  Records remain eight words and their atom/paper offsets
 * address MAIN's existing partAtoms/partPaper arrays.  Playback therefore
 * performs one fixed table lookup, not a glyph selector or a search.
 */
export const SMALL_CONSTRUCTION_VERSION=SMALL_CONSTRUCTION_TABLE_VERSION;
export const SMALL_CONSTRUCTION_RECORD_WORDS=SMALL_CONSTRUCTION_TABLE_RECORD_WORDS;
export const SMALL_CONSTRUCTION_HEADER_WORDS=SMALL_CONSTRUCTION_TABLE_HEADER_WORDS;
export const SMALL_CONSTRUCTION_KIND_KEYS=SMALL_CONSTRUCTION_TABLE_KEY_COUNT;
export const SMALL_CONSTRUCTION_ROWS=SMALL_CONSTRUCTION_TABLE_ROW_COUNT;
export const SMALL_CONSTRUCTION_WIDTHS=SMALL_CONSTRUCTION_TABLE_WIDTH_LIMIT+1;
export const SMALL_CONSTRUCTION_DENSE_RECORDS=SMALL_CONSTRUCTION_TABLE_INDEX_WORDS;
export const SMALL_CONSTRUCTION_MARKER=0x534d414c;
export const SMALL_CONSTRUCTION_NONE=0xffffffff;

/** The three source classes and three end labels are packed by the semantic
 * source record: kind bits 0..1, positive end bits 2..3, negative end bits
 * 4..5.  This function is also the CPU adapter's ABI oracle. */
export function smallConstructionKindEndKey(kindEnds:number):number|undefined{
 const kind=kindEnds&3,positive=(kindEnds>>>2)&3,negative=(kindEnds>>>4)&3;
 if((kindEnds&~0x3f)!==0||kind>=3||positive>=3||negative>=3)return;
 return kind*9+positive*3+negative;
}

export interface SmallConstructionRecordRef{
 readonly address:number;
 readonly kindEnds:number;
 readonly rows:number;
 readonly width:number;
 readonly units:number;
 readonly firstAtom:number;
 readonly atomCount:number;
 readonly firstPaper:number;
 readonly paperCount:number;
}

/** Read one record through the same dense direct lookup used by WGSL.  This is
 * a test adapter, not a runtime fallback: callers use it to prove the packed
 * ABI and the GPU producer's unsupported-record behavior. */
export function smallConstructionRecordCpu(bank:ArrayLike<number>,tableBase:number,kindEnds:number,rows:number,requestedWidth:number):SmallConstructionRecordRef|undefined{
 const key=smallConstructionKindEndKey(kindEnds);
 if(key===undefined||!Number.isInteger(tableBase)||tableBase<0||rows<1||rows>SMALL_CONSTRUCTION_ROWS||requestedWidth<0||!Number.isFinite(requestedWidth))return;
 const headerEnd=tableBase+SMALL_CONSTRUCTION_HEADER_WORDS;
 if(headerEnd>bank.length||bank[tableBase]!==SMALL_CONSTRUCTION_VERSION||bank[tableBase+1]!==SMALL_CONSTRUCTION_RECORD_WORDS)return;
 const stride=bank[tableBase+1]!;
 const count=bank[tableBase+2]!,recordsRelativeAt=bank[tableBase+3]!,indexRelativeAt=bank[tableBase+4]!,widthLimit=bank[tableBase+5]!,kindCount=bank[tableBase+6]!,rowCount=bank[tableBase+7]!;
 if(headerEnd<tableBase+8||headerEnd>bank.length||!Number.isInteger(count)||!Number.isInteger(recordsRelativeAt)||!Number.isInteger(indexRelativeAt)||recordsRelativeAt<SMALL_CONSTRUCTION_HEADER_WORDS||indexRelativeAt<SMALL_CONSTRUCTION_HEADER_WORDS||widthLimit!==SMALL_CONSTRUCTION_TABLE_WIDTH_LIMIT||kindCount!==SMALL_CONSTRUCTION_TABLE_KIND_COUNT||rowCount!==SMALL_CONSTRUCTION_TABLE_ROW_COUNT)return;
 const width=Math.min(SMALL_CONSTRUCTION_WIDTHS-1,Math.floor(requestedWidth));
 const denseIndex=(key*SMALL_CONSTRUCTION_ROWS+(rows-1))*SMALL_CONSTRUCTION_WIDTHS+width;
 if(denseIndex>=SMALL_CONSTRUCTION_TABLE_INDEX_WORDS)return;
 const indexAddress=tableBase+indexRelativeAt+denseIndex;
 if(indexAddress<0||indexAddress>=bank.length)return;
 const ordinal=bank[indexAddress]!;if(ordinal===SMALL_CONSTRUCTION_NONE||ordinal>=count)return;
 const address=tableBase+recordsRelativeAt+ordinal*stride;
 if(address<0||address+SMALL_CONSTRUCTION_RECORD_WORDS>bank.length)return;
 const recordKind=bank[address]!,recordRows=bank[address+1]!,recordWidth=bank[address+2]!,units=bank[address+3]!;
 const firstAtom=bank[address+4]!,atomCount=bank[address+5]!,firstPaper=bank[address+6]!,paperCount=bank[address+7]!;
 // A zero atom count is the explicit unsupported/blank cell.  A valid record
 // may have no protected paper, but can never have zero width or rows.
 if(recordKind!==kindEnds||recordRows!==rows||recordWidth===0||recordWidth>width||atomCount===0)return;
 return {address,kindEnds:recordKind,rows:recordRows,width:recordWidth,units,firstAtom,atomCount,firstPaper,paperCount};
}

export interface SmallConstructionEndpoint{readonly x:number;readonly y:number;readonly depth?:number;readonly parameter?:number;readonly reciprocalW?:number}
export interface SmallConstructionRun{
 readonly marker:number;
 readonly kindEnds:number;
 /** Source-pair index relative to the run's point range. */
 readonly sourcePair:number;
 readonly originX:number;readonly topRow:number;readonly width:number;readonly rows:number;
}
export interface SmallConstructionGlyphEmission{
 readonly wordIndex:number;readonly x:number;readonly row:number;readonly advance:number;
 readonly sourcePhase:number;readonly sourcePair:number;
}
export interface SmallConstructionPaperEmission{readonly x:number;readonly row:number;readonly width:number}
export interface SmallConstructionPlaybackResult{
 readonly record:SmallConstructionRecordRef;
 readonly originX:number;
 readonly glyphs:readonly SmallConstructionGlyphEmission[];
 readonly paper:readonly SmallConstructionPaperEmission[];
}

/** CPU adapter for the exact GPU playback contract.  It deliberately has no
 * visibility or font selection logic: native word indices, paper and source
 * phase are all copied from the selected compiled record. */
export function smallConstructionPlaybackCpu(bank:ArrayLike<number>,partAtoms:ArrayLike<number>,partPaper:ArrayLike<number>,tableBase:number,run:SmallConstructionRun,endpoints:readonly SmallConstructionEndpoint[],page:{readonly width:number;readonly rows:number}):SmallConstructionPlaybackResult|undefined{
 if(run.marker!==SMALL_CONSTRUCTION_MARKER||run.sourcePair<0||run.sourcePair*2+1>=endpoints.length||run.rows<1||run.rows>SMALL_CONSTRUCTION_ROWS||run.width<1||run.topRow<0||run.topRow+run.rows>page.rows)return;
 const finite=(point:SmallConstructionEndpoint)=>Number.isFinite(point.x)&&Number.isFinite(point.y)&&Number.isFinite(point.depth??0)&&Number.isFinite(point.parameter??0)&&Number.isFinite(point.reciprocalW??1);
 if(!finite(endpoints[run.sourcePair*2]!)||!finite(endpoints[run.sourcePair*2+1]!))return;
 const record=smallConstructionRecordCpu(bank,tableBase,run.kindEnds,run.rows,run.width);if(!record)return;
 const slack=run.width-record.width,origin=run.originX+Math.floor(slack/2);
 if(origin<0||origin+record.width>page.width)return;
 const atomBase=record.firstAtom*4,paperBase=record.firstPaper*4;
 if(atomBase<0||atomBase+record.atomCount*4>partAtoms.length||paperBase<0||paperBase+record.paperCount*4>partPaper.length)return;
 const glyphs:SmallConstructionGlyphEmission[]=[];
 for(let i=0;i<record.atomCount;i++){
  const at=atomBase+i*4,word=partAtoms[at]!,x=partAtoms[at+1]!,row=partAtoms[at+2]!,advance=partAtoms[at+3]!;
  if(row>=run.rows||advance===0||x+advance>record.width||origin+x<0||origin+x+advance>page.width)return;
  const phase=run.rows===1?0:row/(run.rows-1);
  glyphs.push({wordIndex:word,x:origin+x,row:run.topRow+row,advance,sourcePhase:phase,sourcePair:run.sourcePair});
 }
 const paper:SmallConstructionPaperEmission[]=[];
 for(let i=0;i<record.paperCount;i++){
  const at=paperBase+i*4,x=partPaper[at]!,row=partPaper[at+1]!,width=partPaper[at+2]!;
  if(row>=run.rows||width===0||x+width>record.width||origin+x<0||origin+x+width>page.width)return;
  paper.push({x:origin+x,row:run.topRow+row,width});
 }
 return {record,originX:origin,glyphs,paper};
}

/** WGSL fragment inserted into a production articulation/far stage.  It
 * assumes the canonical MAIN declarations (`Run`, `Point`, `p`, `runs`,
 * `points`, `grammar`, `counts`, `NONE`, `rowCount`, `spanPartWord` and
 * `interpolatePoint`) are already present.  No new storage binding is used. */
export function smallConstructionWgsl(tableBaseExpression=`grammar[9u]+${SMALL_CONSTRUCTION_TABLE_AT}u`):string{
 return /* wgsl */`
const SMALL_CONSTRUCTION_VERSION=${SMALL_CONSTRUCTION_VERSION}u;
const SMALL_CONSTRUCTION_STRIDE=${SMALL_CONSTRUCTION_RECORD_WORDS}u;
const SMALL_CONSTRUCTION_HEADER_WORDS=${SMALL_CONSTRUCTION_HEADER_WORDS}u;
const SMALL_CONSTRUCTION_INDEX_WORDS=${SMALL_CONSTRUCTION_TABLE_INDEX_WORDS}u;
const SMALL_CONSTRUCTION_ROWS=${SMALL_CONSTRUCTION_ROWS}u;
const SMALL_CONSTRUCTION_WIDTHS=${SMALL_CONSTRUCTION_WIDTHS}u;
const SMALL_CONSTRUCTION_MARKER=0x534d414cu;
fn smallConstructionKindEndKey(kindEnds:u32)->u32{
 let kind=kindEnds&3u;let positive=(kindEnds>>2u)&3u;let negative=(kindEnds>>4u)&3u;
 if(kind>=3u||positive>=3u||negative>=3u||(kindEnds&0xffffffc0u)!=0u){return NONE;}
 return kind*9u+positive*3u+negative;
}
fn smallConstructionBase()->u32{return ${tableBaseExpression};}
fn smallConstructionTableReady(base:u32)->bool{
 return base+7u<arrayLength(&grammar)&&grammar[base]==SMALL_CONSTRUCTION_VERSION&&grammar[base+1u]==SMALL_CONSTRUCTION_STRIDE&&grammar[base+2u]>0u&&grammar[base+3u]>=SMALL_CONSTRUCTION_HEADER_WORDS&&grammar[base+4u]>=SMALL_CONSTRUCTION_HEADER_WORDS&&grammar[base+5u]==SMALL_CONSTRUCTION_WIDTHS-1u&&grammar[base+6u]==${SMALL_CONSTRUCTION_TABLE_KIND_COUNT}u&&grammar[base+7u]==SMALL_CONSTRUCTION_ROWS;
}
// The compiler has already filled every integer-width cell with the widest
// complete record that fits that width.  This is one direct lookup; it never
// searches glyphs, visibility fragments or another kind/end family.
fn smallConstructionRecord(kindEnds:u32,rows:u32,width:u32)->u32{
 let key=smallConstructionKindEndKey(kindEnds);if(key==NONE||rows<1u||rows>SMALL_CONSTRUCTION_ROWS){return NONE;}
 let base=smallConstructionBase();if(!smallConstructionTableReady(base)){return NONE;}
 let requested=min(width,SMALL_CONSTRUCTION_WIDTHS-1u);
 let dense=(key*SMALL_CONSTRUCTION_ROWS+(rows-1u))*SMALL_CONSTRUCTION_WIDTHS+requested;
 if(dense>=SMALL_CONSTRUCTION_INDEX_WORDS){return NONE;}
 let indexAt=base+grammar[base+4u]+dense;if(indexAt>=arrayLength(&grammar)){return NONE;}
 let ordinal=grammar[indexAt];if(ordinal==${SMALL_CONSTRUCTION_TABLE_NONE}u||ordinal>=grammar[base+2u]){return NONE;}
 let at=base+grammar[base+3u]+ordinal*SMALL_CONSTRUCTION_STRIDE;
 if(at+7u>=arrayLength(&grammar)){return NONE;}
 let recordKind=grammar[at];let recordRows=grammar[at+1u];let recordWidth=grammar[at+2u];let atomCount=grammar[at+5u];
 if(recordKind!=kindEnds||recordRows!=rows||recordWidth==0u||recordWidth>width||atomCount==0u){return NONE;}
 return at;
}
fn smallConstructionMarkPaper(record:u32,origin:i32,topRow:i32,rows:u32){
 let first=grammar[record+6u];let count=grammar[record+7u];
 for(var i=0u;i<count;i++){
  let at=grammar[6]+(first+i)*4u;let row=grammar[at+1u];let width=grammar[at+2u];
  if(row>=rows||width==0u){continue;}
  let pageRow=topRow+i32(row);let start=origin+i32(grammar[at]);let end=start+i32(width);
  if(pageRow<0||pageRow>=i32(rowCount())){continue;}
  for(var x=max(0,start);x<min(i32(p.extent.x),end);x++){
   atomicOr(&counts[paperBase()+u32(pageRow)*paperStride()+u32(x)/32u],1u<<(u32(x)%32u));
  }
 }
}
// Complete source-span playback. The caller must leave the original profile
// range in place and append the canonical pair after it; run.part.z names that
// pair relative to run.range.x. A false result emits no proposal and lets the
// incumbent profile stay authoritative.
fn smallConstructionPlayback(runID:u32,run:Run,priority:u32)->bool{
 if(run.correspondence.w!=0x534d414cu){return false;}
 let sourcePair=run.part.z;
 let origin=bitcast<i32>(run.paper.x);let topRow=bitcast<i32>(run.paper.y);let width=run.paper.z;let rows=run.paper.w;
 if(width==0u||rows==0u||rows>SMALL_CONSTRUCTION_ROWS||topRow<0||topRow+i32(rows)>i32(rowCount())){return false;}
 let record=smallConstructionRecord(run.part.x,rows,width);if(record==NONE){return false;}
 let base=smallConstructionBase();let recordWidth=grammar[record+2u];let slack=width-recordWidth;let candidateOrigin=origin+i32(slack/2u);
 if(candidateOrigin<0||candidateOrigin+i32(recordWidth)>i32(p.extent.x)){return false;}
 let first=grammar[record+4u];let atomCount=grammar[record+5u];let paperFirst=grammar[record+6u];let paperCount=grammar[record+7u];
 if(grammar[1]+(first+atomCount)*4u>arrayLength(&grammar)||grammar[6]+(paperFirst+paperCount)*4u>arrayLength(&grammar)){return false;}
 let pairAt=run.range.x+sourcePair*2u;
 // Source points are producer-owned finite endpoints. Reject malformed
 // appended pairs before any proposal, rather than mixing a partial small
 // drawing with the incumbent profile.
 if(pairAt+1u>=arrayLength(&points)){return false;}
 let sourceA=points[pairAt];let sourceB=points[pairAt+1u];
 if(sourceA.visible.y==0u||sourceB.visible.y==0u||any(sourceA.position!=sourceA.position)||any(sourceB.position!=sourceB.position)){return false;}
 for(var i=0u;i<atomCount;i++){
  let at=grammar[1]+(first+i)*4u;let word=grammar[at];let x=grammar[at+1u];let row=grammar[at+2u];let advance=grammar[at+3u];
  if(word==0u||row>=rows||advance==0u||x+advance>recordWidth||candidateOrigin+i32(x)<0||candidateOrigin+i32(x+advance)>i32(p.extent.x)){return false;}
 }
 for(var i=0u;i<paperCount;i++){
  let at=grammar[6]+(paperFirst+i)*4u;let x=grammar[at];let row=grammar[at+1u];let paperWidth=grammar[at+2u];
  if(row>=rows||paperWidth==0u||x+paperWidth>recordWidth||candidateOrigin+i32(x)<0||candidateOrigin+i32(x+paperWidth)>i32(p.extent.x)){return false;}
 }
 var emitted=false;
 var previousRow=NONE;var rowMember=0u;
 for(var i=0u;i<atomCount;i++){
  let at=grammar[1]+(first+i)*4u;let row=grammar[at+2u];let phase=f32(row)/max(f32(rows-1u),1.0);var q=interpolatePoint(sourceA,sourceB,phase);q.w=phase;
  if(row!=previousRow){previousRow=row;rowMember=0u;}
  var lastInRow=i+1u==atomCount;if(!lastInRow){lastInRow=grammar[grammar[1]+(first+i+1u)*4u+2u]!=row;}
  let memberRow=select(row,0x3ffu,row+1u==rows&&row>0u);
  let member=select(rowMember,0xffffu,lastInRow&&rowMember>0u);
  spanPartWord(runID,(memberRow<<16u)|member,q,candidateOrigin+i32(grammar[at+1u]),topRow+i32(row),grammar[at],1u,grammar[at+3u],priority,sourcePair,1u);emitted=true;rowMember++;
 }
 smallConstructionMarkPaper(record,candidateOrigin,topRow,rows);return emitted;
}
`;
}

export const SMALL_CONSTRUCTION_WGSL=smallConstructionWgsl();
/** Tiny named bridge for far-stage generators: retaining this function makes
 * the helper live in an isolated production standing shader without adding a
 * storage binding or duplicating the playback body. */
export const SMALL_CONSTRUCTION_FAR_BRIDGE_WGSL=/* wgsl */`fn smallConstructionFarBridge(runID:u32,run:Run,priority:u32)->bool{return smallConstructionPlayback(runID,run,priority);}`;
