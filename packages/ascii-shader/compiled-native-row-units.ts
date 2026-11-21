import {nativeSpaceCodes,nativeSpacePlan,type NativeWhitespacePolicy} from './text/native-whitespace.ts';
import type {NativeGlyph} from './native-page.ts';

/**
 * The input is a retained native construction, rather than a scene or role
 * entry. No source id is copied into the compiled programme: local member
 * numbers below are the correspondence consumed by playback.
 */
export interface NativeRowSourceContext {
 readonly tangentIn:string; readonly tangentOut:string; readonly curvature:string;
 readonly inside:string; readonly paper?:string;
}
export interface NativeRowSource {
 readonly role:string; readonly context:NativeRowSourceContext;
 readonly morphology?:string; readonly rows:readonly string[];
 readonly rowPenX:readonly number[];
}
export interface NativeRowUnitAtom {
 readonly scalar:string; readonly glyph:number; readonly x:number; readonly advance:number;
}
export type NativeRowUnitKind='anchor'|'arc'|'tone'|'ink'|'paper';
export type NativeRowSide='left'|'right'|'seam';
export interface NativeRowUnit {
 readonly member:number; readonly kind:NativeRowUnitKind; readonly text:string;
 readonly x:number; readonly width:number; readonly atoms:readonly NativeRowUnitAtom[];
 readonly repeatable:boolean; readonly side:NativeRowSide;
}
export interface NativePaperPlan {
 readonly width:number; readonly full:number; readonly half:number;
}
/** Expanded only by the CPU reference playback; compiled descriptors stay
 * bounded to the half/full counts suitable for direct GPU lookup. */
export interface NativePaperPlaybackPlan extends NativePaperPlan {readonly codes:readonly number[];}
export interface NativeRowGrowthRecord {
 readonly width:number; readonly delta:number; readonly repeatCount:number;
 /** Width of one complete repeated word plus its authored inter-word paper. */
 readonly repeatUnitWidth:number; readonly repeatGap:number;
 readonly paperDelta:number; readonly paperGap:number; readonly paper:NativePaperPlan;
}
export type NativeRowGrowthMode='gap'|'left-fixed'|'right-shift'|'baseline';
export interface NativeRowResiduePlan {
 readonly residue:number;
 /** Number of complete repeat units skipped to reach a legal paper plan. */
 readonly skip:number;
 readonly paperDelta:number;
 readonly paper:NativePaperPlan;
}
/** Compact direct lookup. `paperPrefix` is bounded by the whitespace
 * planner's finite invalid prefix; the tail is evaluated by its fixed plan
 * formula, never by a search or a width-sized object table. */
export interface NativeRowGrowthTable {
 readonly maxDelta:number; readonly mode:NativeRowGrowthMode; readonly seam:number;
 readonly growthGapIndex:number; readonly baseGap:number;
 readonly repeatMember?:number; readonly repeatBefore:boolean;
 readonly repeatUnitWidth:number; readonly repeatGap:number; readonly requireRepeat:boolean;
 readonly residues:readonly (NativeRowResiduePlan|undefined)[];
 readonly paperPrefix:readonly (NativePaperPlan|undefined)[];
 readonly paperTailStart:number;
}
export interface NativeRowPlacement {
 readonly member:number; readonly kind:NativeRowUnitKind; readonly text:string;
 /** Stable authored member carried by inserted whole-unit occurrences. */
 readonly sourceMember:number;
 readonly x:number; readonly width:number; readonly atoms:readonly NativeRowUnitAtom[];
 readonly inserted:boolean;
}
export interface NativeRowPlayback {
 readonly row:number; readonly width:number; readonly delta:number;
 readonly placements:readonly NativeRowPlacement[];
 readonly paper:readonly {readonly member:number;readonly x:number;readonly width:number;readonly plan:NativePaperPlaybackPlan}[];
}
export interface NativeRowUnsupported {
 readonly row:number;
 readonly reason:'no-legal-authored-gap'|'no-legal-growth-width'|'seam-crosses-indivisible-unit';
 readonly preservedWidth:number;
}
export interface CompiledNativeRow {
 readonly row:number; readonly authoredWidth:number; readonly sourceRow:number;
 readonly sourceOrigin:number; readonly leftInset:number;
 readonly policy:NativeWhitespacePolicy;
 readonly units:readonly NativeRowUnit[]; readonly growthSlot:number;
 readonly growth:NativeRowGrowthTable;
 readonly unsupported:readonly NativeRowUnsupported[];
}
export interface CompiledNativeRowProgram {
 readonly role:string; readonly context:NativeRowSourceContext; readonly morphology?:string;
 readonly policy:NativeWhitespacePolicy; readonly rows:readonly CompiledNativeRow[];
 readonly sourceOrigin:number; readonly constructionWidth:number;
 readonly seam:number;
 readonly sourceWidth:{readonly min:number;readonly max:number}; readonly maxGrowthPx:number;
 /**
  * Complete authored middle-row units used when the same construction is
  * posed taller than its retained page.  The sequence contains source rows,
  * not re-selected glyphs: playback repeats the whole sequence between the
  * enduring first and final anchors.
  */
 readonly height:NativeRowHeightProgram;
}
export interface CompileNativeRowUnitsOptions {readonly policy?:NativeWhitespacePolicy; readonly maxGrowthPx?:number;}

export interface NativeRowHeightUnit {
 /** Source row whose complete native units are replayed. */
 readonly sourceRow:number;
 /** Stable source-row identity; inserted cycles retain this identity. */
 readonly member:number;
 readonly leftAnchorX:number; readonly rightAnchorX:number;
}
export interface NativeRowHeightProgram {
 readonly authoredRows:number; readonly minRows:number; readonly maxRows:number;
 readonly firstAnchorMember:number; readonly finalAnchorMember:number;
 readonly memberStride:number;
 readonly sequence:readonly NativeRowHeightUnit[];
 /** Per-cycle continuation of the two enduring side branches. */
 readonly leftCycleShift:number; readonly rightCycleShift:number;
 /** Common source-space seam used to keep inner units on their authored side. */
 readonly pivot:number;
}
export interface NativeRowHeightPlayback extends NativeRowPlayback {
 readonly sourceRow:number; readonly cycle:number; readonly rowMember:number;
 readonly leftShift:number; readonly rightShift:number; readonly terminal:boolean;
}

/** Existing native-page width; the default table covers the complete page,
 * not a small display-specific quota. */
const NATIVE_PAGE_WIDTH=8192;
const NATIVE_PAGE_ROWS=Math.ceil(NATIVE_PAGE_WIDTH/17);
const FINAL_ANCHOR_MEMBER=0x3ff;

interface Token extends NativeRowUnitAtom {readonly word:number;}
interface Word {readonly index:number; readonly atoms:readonly Token[]; readonly x:number; readonly width:number; readonly text:string; readonly kind:NativeRowUnitKind;}
interface Gap {readonly before:number; readonly x:number; readonly width:number;}
interface ParsedRow {readonly words:Word[]; readonly gaps:Gap[]; readonly left:number; readonly right:number;}

const FULL_SPACE=11;
const HALF_SPACE=5;
const CURVE_SCALARS=new Set([...'/\\／＼｀ヽ、､⌒＜＞（）(){}「」']);
const TONE_SCALARS=new Set([...'.:,;･・']);
const LINE_SCALARS=new Set([...'-_＿￣─ｰ―‐']);
const scalarChars=(text:string):string[]=>[...text];
const decoded=(text:string)=>text.replace(/\\u([a-fA-F0-9]{4})/g,(_,hex:string)=>String.fromCharCode(parseInt(hex,16)));
const advanceOf=(glyph:NativeGlyph|undefined,scalar:string):number=>scalar===' '?HALF_SPACE:scalar==='　'?FULL_SPACE:glyph?.advance??0;
const unitKind=(text:string):NativeRowUnitKind=>{
 const chars=scalarChars(text);
 if(chars.length&&chars.every(char=>TONE_SCALARS.has(char)))return 'tone';
 if(chars.some(char=>CURVE_SCALARS.has(char)))return 'arc';
 return 'ink';
};
const legalPaper=(width:number,policy:NativeWhitespacePolicy):NativePaperPlan|undefined=>{
 if(!Number.isSafeInteger(width)||width<0)return;
 const plan=nativeSpacePlan(width,0,false,false,policy);
 if(!plan)return;
 return Object.freeze({width,full:plan.full,half:plan.half});
};

/** Parse authored words without changing scalar spelling or glyph order. */
const parseRow=(row:string,rowPen:number,font:ReadonlyMap<string,{glyph:NativeGlyph;index:number}>):ParsedRow=>{
 const words:Word[]=[]; const gaps:Gap[]=[]; let pen=rowPen; let current:Token[]=[]; let gapStart=pen; let inGap=false;
 const finish=()=>{
  if(!current.length)return;
  const x=current[0]!.x; const width=current.reduce((sum,atom)=>sum+atom.advance,0);
  const text=current.map(atom=>atom.scalar).join('');
  words.push(Object.freeze({index:words.length,atoms:Object.freeze([...current]),x,width,text,kind:unitKind(text)})); current=[];
 };
 for(const scalar of scalarChars(decoded(row))){
  const glyph=font.get(scalar); const advance=advanceOf(glyph?.glyph,scalar);
  if(!advance)throw Error(`Native row contains an unavailable scalar ${JSON.stringify(scalar)}`);
  if(scalar===' '||scalar==='　'){
   finish();
   if(words.length&&!inGap){gapStart=pen; inGap=true;}
   pen+=advance;
   continue;
  }
  if(inGap){gaps.push(Object.freeze({before:words.length-1,x:gapStart,width:pen-gapStart})); inGap=false;}
  if(!glyph)throw Error(`Native row contains an unavailable scalar ${JSON.stringify(scalar)}`);
  current.push({scalar,glyph:glyph.index,x:pen,advance,word:words.length}); pen+=advance;
 }
 finish();
 const ink=words.flatMap(word=>word.atoms.map(atom=>[atom.x,atom.x+atom.advance] as const));
 if(!ink.length)throw Error('Native row contains no ink');
 return {words,gaps,left:Math.min(...ink.map(pair=>pair[0])),right:Math.max(...ink.map(pair=>pair[1]))};
};
const glyphAtoms=(word:Word):readonly NativeRowUnitAtom[]=>Object.freeze(word.atoms.map(atom=>Object.freeze({scalar:atom.scalar,glyph:atom.glyph,x:atom.x,advance:atom.advance})));
const memberFor=(word:number)=>word+1;
const sideOf=(x:number,width:number,seam:number):NativeRowSide=>x+width<=seam?'left':x>=seam?'right':'seam';

const isLineWord=(word:Word|undefined)=>!!word&&scalarChars(word.text).length>0&&scalarChars(word.text).every(char=>LINE_SCALARS.has(char));
interface RepeatChoice {readonly word:Word; readonly before:boolean;}
/** Only a complete authored line word may be repeated. Arc, tone and opening
 * words remain indivisible. The two-word flat cap case repeats the terminal
 * line word before its held anchor, rather than widening its paper hole. */
const repeatableWord=(source:NativeRowSource,words:readonly Word[],growthSlot:number):RepeatChoice|undefined=>{
 if(!['rim_cap','return_base','cap','return'].includes(source.role))return;
 const candidate=words[growthSlot-1];
 if(candidate&&candidate.index>0&&candidate.index<words.length-1&&isLineWord(candidate))return {word:candidate,before:false};
 if(words.length===2&&growthSlot===1&&isLineWord(words[1]))return {word:words[1],before:true};
 return;
};

const growthTable=(mode:NativeRowGrowthMode,seam:number,growthGapIndex:number,baseGap:number,repeat:RepeatChoice|undefined,policy:NativeWhitespacePolicy,maxGrowthPx:number):NativeRowGrowthTable=>{
 const repeatUnitWidth=repeat?repeat.word.width+baseGap:0;
 const residues:Array<NativeRowResiduePlan|undefined>=repeat?Array.from({length:repeatUnitWidth}):[];
 const paperPrefix:Array<NativePaperPlan|undefined>=[];
 let paperTailStart=0;
 if(mode==='gap'&&repeat){
  for(let residue=0;residue<repeatUnitWidth;residue++){
   let skip=0; let paper:NativePaperPlan|undefined;
   while(baseGap+residue+skip*repeatUnitWidth<=baseGap+maxGrowthPx+repeatUnitWidth*2&&!paper){
    paper=legalPaper(baseGap+residue+skip*repeatUnitWidth,policy);
    if(!paper)skip++;
   }
   if(paper)residues[residue]=Object.freeze({residue,skip,paperDelta:residue+skip*repeatUnitWidth,paper});
  }
 }else if(mode==='gap'){
  let lastIllegal=-1;
  for(let delta=0;delta<=maxGrowthPx;delta++)if(!legalPaper(baseGap+delta,policy))lastIllegal=delta;
  paperTailStart=lastIllegal+1;
  for(let delta=0;delta<Math.min(paperTailStart,maxGrowthPx+1);delta++)paperPrefix.push(legalPaper(baseGap+delta,policy));
 }
 return Object.freeze({maxDelta:maxGrowthPx,mode,seam,growthGapIndex,baseGap,repeatMember:repeat?memberFor(repeat.word.index):undefined,repeatBefore:repeat?.before??false,repeatUnitWidth,repeatGap:repeat?baseGap:0,requireRepeat:!!repeat,residues:Object.freeze(residues),paperPrefix:Object.freeze(paperPrefix),paperTailStart});
};
/** Choose one construction seam for every row. Prefer a coordinate admitted
 * by the most rows' authored gaps, then avoid crossing indivisible arc/tone
 * words, then retain the strongest shared paper span. */
const sharedSeam=(rows:readonly ParsedRow[],sourceOrigin:number,constructionWidth:number):number=>{
 const candidates=[...rows.flatMap(row=>row.gaps.map(gap=>gap.x+gap.width/2)),sourceOrigin+constructionWidth/2];
 const centre=sourceOrigin+constructionWidth/2;
 const score=(candidate:number)=>{
  let covered=0,crossed=0,mismatch=0,gapWidth=0;
  for(const row of rows){
   const gap=row.gaps.find(value=>candidate>=value.x&&candidate<=value.x+value.width);
   if(gap){covered++;gapWidth+=gap.width;continue;}
   const word=row.words.find(value=>candidate>value.x&&candidate<value.x+value.width);
   if(word&&!isLineWord(word))crossed++;
  }
  for(let row=0;row+1<rows.length;row++)for(const atom of rows[row]!.words.flatMap(word=>word.atoms)){
   const nearest=rows[row+1]!.words.flatMap(word=>word.atoms).reduce<Token|undefined>((best,value)=>!best||Math.abs(value.x-atom.x)<Math.abs(best.x-atom.x)?value:best,undefined);
   if(nearest&&((atom.x+atom.advance/2)>=candidate)!==((nearest.x+nearest.advance/2)>=candidate))mismatch++;
  }
  return [covered,-crossed,-mismatch,gapWidth,-Math.abs(candidate-centre)] as const;
 };
 return candidates.reduce((best,candidate)=>{
  const next=score(candidate),current=score(best);
  for(let index=0;index<next.length;index++)if(next[index]!==current[index])return next[index]>current[index]?candidate:best;
  return best;
 },candidates[0]!)-sourceOrigin;
};

function compileRow(source:NativeRowSource,row:number,parsed:ParsedRow,sourceOrigin:number,seam:number,policy:NativeWhitespacePolicy,maxGrowthPx:number):CompiledNativeRow{
 const words=parsed.words.map(word=>Object.freeze({...word,x:word.x-sourceOrigin,atoms:Object.freeze(word.atoms.map(atom=>Object.freeze({...atom,x:atom.x-sourceOrigin})))}));
 const gaps=parsed.gaps.map(gap=>Object.freeze({...gap,x:gap.x-sourceOrigin}));
 const authoredWidth=parsed.right-parsed.left; const leftInset=parsed.left-sourceOrigin; const terminalIndex=words.length-1;
 let seamGapIndex=gaps.findIndex(gap=>seam>=gap.x&&seam<=gap.x+gap.width);
 let crossingIndex=words.findIndex(word=>seam>word.x&&seam<word.x+word.width);
 // If the shared seam falls through a whole line unit, grow at the next
 // authored gap by inserting that complete unit. Never split an arc/tone
 // word merely because the seam crossed its ink.
 if(seamGapIndex<0&&crossingIndex>=0&&isLineWord(words[crossingIndex])){
  if(crossingIndex<words.length-1&&gaps[crossingIndex])seamGapIndex=crossingIndex;
  else if(crossingIndex>0&&gaps[crossingIndex-1])seamGapIndex=crossingIndex-1;
 }
 const growthGap=seamGapIndex>=0?gaps[seamGapIndex]:undefined;
 const growthSlot=growthGap?growthGap.before+1:0;
 let repeat=repeatableWord(source,words,growthSlot);
 if(!repeat&&growthGap&&crossingIndex>=0&&isLineWord(words[crossingIndex])&&['rim_cap','return_base','cap','return'].includes(source.role)){
  repeat={word:words[crossingIndex]!,before:crossingIndex===growthSlot};
 }
 const units:NativeRowUnit[]=[];
 for(const word of words)units.push(Object.freeze({member:memberFor(word.index),kind:word.index===0||word.index===terminalIndex?'anchor':word.kind,text:word.text,x:word.x,width:word.width,atoms:glyphAtoms(word),repeatable:repeat?.word.index===word.index,side:sideOf(word.x,word.width,seam)}));
 for(const [index,gap]of gaps.entries())units.push(Object.freeze({member:0x10000+index,kind:'paper',text:'',x:gap.x,width:gap.width,atoms:Object.freeze([]),repeatable:false,side:sideOf(gap.x,gap.width,seam)}));
 let mode:NativeRowGrowthMode='baseline'; let gapIndex=-1; let baseGap=0; const unsupported:NativeRowUnsupported[]=[];
 if(growthGap){
  mode='gap'; gapIndex=seamGapIndex; baseGap=growthGap.width;
  if(!legalPaper(baseGap,policy)){mode='baseline'; unsupported.push(Object.freeze({row,reason:'no-legal-authored-gap',preservedWidth:authoredWidth}));}
 }else if(parsed.right-sourceOrigin<=seam){mode='left-fixed';
 }else if(parsed.left-sourceOrigin>=seam){mode='right-shift';
 }else{
  mode='baseline'; unsupported.push(Object.freeze({row,reason:'seam-crosses-indivisible-unit',preservedWidth:authoredWidth}));
 }
 const growth=growthTable(mode,seam,gapIndex,baseGap,mode==='gap'?repeat:undefined,policy,maxGrowthPx);
 return Object.freeze({row,sourceRow:row,authoredWidth,sourceOrigin,leftInset,policy,units:Object.freeze(units),growthSlot,growth,unsupported:Object.freeze(unsupported)});
}

const rowAnchorBounds=(row:CompiledNativeRow):{left:number;right:number}=>{
 const words=row.units.filter(unit=>unit.kind!=='paper');
 const first=words[0],last=words.at(-1);
 if(!first||!last)return {left:0,right:0};
 return {left:first.x,right:last.x+last.width};
};

/**
 * Compile the height programme from the same retained rows as width growth.
 * Every inserted row is a complete authored middle row.  A cycle is allowed
 * only while the construction still has both named anchors; a contraction is
 * deliberately unsupported so the incumbent can keep the complete source.
 */
const compileHeightProgram=(rows:readonly CompiledNativeRow[],seam:number):NativeRowHeightProgram=>{
 const authoredRows=rows.length;
 const sequence=rows.slice(1,-1).map(row=>{
  const bounds=rowAnchorBounds(row);
  return Object.freeze({sourceRow:row.sourceRow,member:row.sourceRow,leftAnchorX:bounds.left,rightAnchorX:bounds.right});
 });
 const firstInterior=sequence[0],terminal=rows.at(-1),terminalBounds=terminal?rowAnchorBounds(terminal):{left:0,right:0};
 const firstBounds=firstInterior?{left:firstInterior.leftAnchorX,right:firstInterior.rightAnchorX}:terminalBounds;
 let memberStride=1;while(memberStride<Math.max(1,authoredRows))memberStride*=2;
 const maxCycle=Math.max(0,Math.floor((FINAL_ANCHOR_MEMBER-1-Math.max(0,authoredRows-2))/memberStride));
 const pageLimit=Math.max(authoredRows,NATIVE_PAGE_ROWS);
 // The native page ceiling is not generally a multiple of the authored
 // middle-row sequence.  Round it down to a complete cycle; exposing the
 // raw ceiling would make the selector advertise a partial programme that
 // cannot preserve the terminal join.
 const cycleLimit=sequence.length?authoredRows+Math.floor((pageLimit-authoredRows)/sequence.length)*sequence.length:authoredRows;
 const maxRows=sequence.length?Math.min(cycleLimit,authoredRows+maxCycle*sequence.length):authoredRows;
 return Object.freeze({
  authoredRows,minRows:authoredRows,maxRows,firstAnchorMember:0,finalAnchorMember:FINAL_ANCHOR_MEMBER,memberStride,
  sequence:Object.freeze(sequence),
  leftCycleShift:terminalBounds.left-firstBounds.left,
  rightCycleShift:terminalBounds.right-firstBounds.right,
  pivot:seam,
 });
};

/** Compile authored rows into immutable direct-lookup programmes. */
export function compileNativeRowUnits(font:readonly NativeGlyph[],source:NativeRowSource,options:CompileNativeRowUnitsOptions={}):CompiledNativeRowProgram{
 const policy=options.policy??'constrained';
 if(options.maxGrowthPx!==undefined&&(!Number.isSafeInteger(options.maxGrowthPx)||options.maxGrowthPx<0))throw Error('maxGrowthPx must be a non-negative safe integer');
 if(source.rows.length!==source.rowPenX.length||!source.rows.length)throw Error('Native row source has mismatched or empty rows');
 const glyphMap=new Map<string,{glyph:NativeGlyph;index:number}>(); font.forEach((glyph,index)=>glyphMap.set(glyph.text,{glyph,index}));
 const parsedRows=source.rows.map((row,index)=>parseRow(row,source.rowPenX[index]!,glyphMap));
 const sourceOrigin=Math.min(...parsedRows.map(parsed=>parsed.left));
 const constructionRight=Math.max(...parsedRows.map(parsed=>parsed.right));
 const constructionWidth=constructionRight-sourceOrigin;
 const maxGrowthPx=options.maxGrowthPx??Math.max(0,NATIVE_PAGE_WIDTH-constructionWidth);
 const seam=sharedSeam(parsedRows,sourceOrigin,constructionWidth);
 const rows=parsedRows.map((parsed,row)=>compileRow(source,row,parsed,sourceOrigin,seam,policy,maxGrowthPx));
 const height=compileHeightProgram(rows,seam);
 return Object.freeze({role:source.role,context:Object.freeze({...source.context}),morphology:source.morphology,policy,rows:Object.freeze(rows),sourceOrigin,constructionWidth,seam,sourceWidth:Object.freeze({min:Math.min(...rows.map(row=>row.authoredWidth)),max:Math.max(...rows.map(row=>row.authoredWidth))}),maxGrowthPx,height});
}

const placement=(unit:NativeRowUnit,x:number,inserted:boolean,sourceMember=unit.member):NativeRowPlacement=>Object.freeze({member:unit.member,sourceMember,kind:unit.kind,text:unit.text,x,width:unit.width,atoms:Object.freeze(unit.atoms.map(atom=>Object.freeze({...atom,x:atom.x+(x-unit.x)}))),inserted});
const playbackPlan=(plan:NativePaperPlan,policy:NativeWhitespacePolicy):NativePaperPlaybackPlan=>{
 const codes=nativeSpaceCodes(plan.width,0,false,false,policy);
 if(!codes)throw Error(`Compiled native paper plan ${plan.width} is not serializable`);
 return Object.freeze({...plan,codes:Object.freeze([...codes])});
};

const emptyPaperPlan=(policy:NativeWhitespacePolicy):NativePaperPlan=>legalPaper(0,policy)!;
const growthRecord=(row:CompiledNativeRow,delta:number):NativeRowGrowthRecord|undefined=>{
 const table=row.growth;
 if(delta<0||delta>table.maxDelta)return;
 if(table.requireRepeat&&delta>0&&delta<table.repeatUnitWidth)return;
 // A seam-crossing indivisible unit or an illegal authored gap is an
 // incumbent fallback, not permission to claim arbitrary grown widths.
 if(table.mode==='baseline'&&delta>0)return;
 const empty=emptyPaperPlan(row.policy);
 if(table.mode!=='gap')return Object.freeze({width:row.authoredWidth+delta,delta,repeatCount:0,repeatUnitWidth:0,repeatGap:0,paperDelta:0,paperGap:0,paper:empty});
 if(table.repeatUnitWidth>0){
  const q=Math.floor(delta/table.repeatUnitWidth),residue=delta%table.repeatUnitWidth,entry=table.residues[residue];
  if(!entry||q<entry.skip)return;
  const repeatCount=q-entry.skip,paperDelta=entry.paperDelta,paperGap=table.baseGap+paperDelta;
  return Object.freeze({width:row.authoredWidth+delta,delta,repeatCount,repeatUnitWidth:table.repeatUnitWidth,repeatGap:table.repeatGap,paperDelta,paperGap,paper:entry.paper});
 }
 const paper=delta<table.paperTailStart?table.paperPrefix[delta]:legalPaper(table.baseGap+delta,row.policy);
 if(!paper)return;
 return Object.freeze({width:row.authoredWidth+delta,delta,repeatCount:0,repeatUnitWidth:0,repeatGap:0,paperDelta:delta,paperGap:table.baseGap+delta,paper});
};
/** Compact direct lookup exposed for the CPU reference and ABI tests. */
export function nativeRowGrowthAt(row:CompiledNativeRow,delta:number):NativeRowGrowthRecord|undefined{return growthRecord(row,delta);}

/** Direct compact-table playback. A missing width is an explicit unsupported
 * result; callers retain the authored row rather than inventing a replacement. */
export function playNativeRow(program:CompiledNativeRow,rowWidth:number):NativeRowPlayback|undefined{
 if(!Number.isSafeInteger(rowWidth)||rowWidth<program.authoredWidth)return;
 const delta=rowWidth-program.authoredWidth,record=nativeRowGrowthAt(program,delta); if(!record)return;
 const table=program.growth,repeat=table.repeatMember===undefined?undefined:program.units.find(unit=>unit.member===table.repeatMember);
 const wordUnits=program.units.filter(unit=>unit.kind!=='paper'); const terminal=wordUnits.at(-1); const placements:NativeRowPlacement[]=[]; const paper:Array<NativeRowPlayback['paper'][number]>=[];
 const repeatShift=record.repeatCount*record.repeatUnitWidth;
 const wholeShift=table.mode==='right-shift'?delta:0;
 for(const unit of wordUnits){
  let x=unit.x+wholeShift;
  if(table.mode==='gap'){
   const afterGrowth=unit.member>=memberFor(program.growthSlot);
   const afterRepeat=repeat!==undefined&&unit.member>repeat.member&&!afterGrowth;
   x=unit.x+(afterGrowth?record.delta:afterRepeat?repeatShift:0);
  }
  placements.push(placement(unit,x,false));
  if(repeat&&unit.member===repeat.member){
   for(let copy=0;copy<record.repeatCount;copy++){
    const copyX=table.repeatBefore?repeat.x+record.repeatUnitWidth*copy:repeat.x+record.repeatUnitWidth*(copy+1);
    const copyUnit={...repeat,member:(repeat.member<<16)|(copy+1)};
    placements.push(placement(copyUnit,copyX,true,repeat.member));
    const repeatPaper=legalPaper(record.repeatGap,program.policy);
    if(repeatPaper)paper.push(Object.freeze({member:0x30000|(repeat.member<<8)|(copy+1),x:copyX-repeatPaper.width,width:repeatPaper.width,plan:playbackPlan(repeatPaper,program.policy)}));
   }
  }
 }
 for(const unit of program.units.filter(candidate=>candidate.kind==='paper')){
  const gapIndex=unit.member-0x10000;
  if(table.mode==='gap'&&gapIndex===table.growthGapIndex){
   const growthAnchor=wordUnits.find(candidate=>candidate.member===memberFor(program.growthSlot));
   const anchorX=growthAnchor?.x??terminal?.x??program.authoredWidth;
   paper.push(Object.freeze({member:unit.member,x:anchorX+record.delta-record.paperGap,width:record.paperGap,plan:playbackPlan(record.paper,program.policy)}));
   continue;
  }
  const shifted=table.mode==='right-shift'||table.mode==='gap'&&gapIndex+1>=program.growthSlot;
  const x=unit.x+(shifted?delta:0);
  const plan=legalPaper(unit.width,program.policy);
  if(plan)paper.push(Object.freeze({member:unit.member,x,width:unit.width,plan:playbackPlan(plan,program.policy)}));
 }
 return Object.freeze({row:program.row,width:rowWidth,delta,placements:Object.freeze(placements),paper:Object.freeze(paper)});
}
export function playNativeRowProgram(program:CompiledNativeRowProgram,row:number,rowWidth:number):NativeRowPlayback|undefined{
 const delta=rowWidth-program.constructionWidth;
 if(!Number.isSafeInteger(delta)||delta<0||delta>program.maxGrowthPx)return;
 // A construction is one programme: every authored row must have a direct
 // compact rule for the same delta before any row is played.
 if(program.rows.some(compiled=>!nativeRowGrowthAt(compiled,delta)))return;
 const compiled=program.rows[row];
 return compiled?playNativeRow(compiled,compiled.authoredWidth+delta):undefined;
}

/** Direct height lookup. No row search or glyph choice occurs at playback. */
export function nativeRowHeightAt(program:CompiledNativeRowProgram,pageRows:number,pageRow:number):{
 readonly sourceRow:number; readonly cycle:number; readonly rowMember:number;
 readonly leftShift:number; readonly rightShift:number; readonly terminal:boolean;
}|undefined{
 const height=program.height;
 if(!Number.isSafeInteger(pageRows)||!Number.isSafeInteger(pageRow)||pageRows<height.minRows||pageRows>height.maxRows||pageRow<0||pageRow>=pageRows)return;
 if(pageRows===height.authoredRows){
  const terminal=pageRow===height.authoredRows-1;
  return Object.freeze({sourceRow:pageRow,cycle:0,rowMember:terminal?height.finalAnchorMember:pageRow,leftShift:0,rightShift:0,terminal});
 }
 if(height.sequence.length===0)return;
 const interiorCount=pageRows-2,extra=interiorCount-height.sequence.length;
 if(extra<0||extra%height.sequence.length!==0)return;
 const cycles=extra/height.sequence.length;
 if(pageRow===0)return Object.freeze({sourceRow:0,cycle:0,rowMember:height.firstAnchorMember,leftShift:0,rightShift:0,terminal:false});
 if(pageRow===pageRows-1)return Object.freeze({sourceRow:height.authoredRows-1,cycle:cycles,rowMember:height.finalAnchorMember,leftShift:cycles*height.leftCycleShift,rightShift:cycles*height.rightCycleShift,terminal:true});
 const index=pageRow-1;
 if(index<height.sequence.length){
  const unit=height.sequence[index]!;
  return Object.freeze({sourceRow:unit.sourceRow,cycle:0,rowMember:unit.member,leftShift:0,rightShift:0,terminal:false});
 }
 const repeated=index-height.sequence.length,cycle=1+Math.floor(repeated/height.sequence.length),unit=height.sequence[repeated%height.sequence.length]!;
 return Object.freeze({sourceRow:unit.sourceRow,cycle,rowMember:cycle*height.memberStride+unit.member,leftShift:cycle*height.leftCycleShift,rightShift:cycle*height.rightCycleShift,terminal:false});
}

/**
 * CPU reference for the future direct GPU height consumer. It reuses the
 * already-compiled width programme, translating complete row units by the
 * precompiled side continuation; it never stretches or re-picks a glyph.
 */
export function playNativeRowProgramAtHeight(program:CompiledNativeRowProgram,pageRows:number,pageRow:number,rowWidth:number):NativeRowHeightPlayback|undefined{
 const mapping=nativeRowHeightAt(program,pageRows,pageRow);if(!mapping)return;
 const delta=rowWidth-program.constructionWidth;if(!Number.isSafeInteger(delta)||delta<0||delta>program.maxGrowthPx)return;
 if(program.rows.some(row=>!nativeRowGrowthAt(row,delta)))return;
 const row=program.rows[mapping.sourceRow],played=playNativeRow(row,row.authoredWidth+delta);if(!played)return;
 const sourceRow=program.rows[mapping.sourceRow]!;
 const shiftFor=(side:NativeRowSide)=>side==='left'?mapping.leftShift:side==='right'?mapping.rightShift:(mapping.leftShift+mapping.rightShift)/2;
 const sideForPlacement=(unit:NativeRowPlacement):NativeRowSide=>sourceRow.units.find(candidate=>candidate.member===unit.sourceMember)?.side??sourceRow.units.find(candidate=>candidate.member===sourceRow.growth.repeatMember)?.side??'seam';
 const placements=played.placements.map(unit=>{
  const shift=shiftFor(sideForPlacement(unit));
  return Object.freeze({...unit,x:unit.x+shift,atoms:Object.freeze(unit.atoms.map(atom=>Object.freeze({...atom,x:atom.x+shift})))});
 });
 const paper=played.paper.map(interval=>{
  const side=sourceRow.units.find(candidate=>candidate.member===interval.member)?.side??sourceRow.units.find(candidate=>candidate.member===sourceRow.growth.repeatMember)?.side??'seam';
  const shift=shiftFor(side);
  return Object.freeze({...interval,x:interval.x+shift});
 });
 return Object.freeze({row:pageRow,width:rowWidth,delta,placements:Object.freeze(placements),paper:Object.freeze(paper),sourceRow:mapping.sourceRow,cycle:mapping.cycle,rowMember:mapping.rowMember,leftShift:mapping.leftShift,rightShift:mapping.rightShift,terminal:mapping.terminal});
}
