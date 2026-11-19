import {ROLE_ENTRIES,nativeRoleText} from './grammar/role-bank.ts';
import roundSupplement from './grammar/role-rounds-supplement.json' with {type:'json'};
import type {SourceSmallParentEnd,SourceSmallParentKind} from './source-small-parent.ts';
import type {NativeGlyph} from './native-page.ts';
import {PARENT_MAX_COLUMNS} from './source-lod.ts';

/** COMPILED SMALLEST-TIER PARENT CONSTRUCTIONS (perf-impl-156; D9, petit AA 07).
 *
 * A far parent held at 1-5 page rows plays one genuine connected construction
 * at ITS OWN size: no 9-row shoulder or 10-row flank squeezed into five rows,
 * no per-edge wireframe, no tone, gaps or dots. Every record is built only
 * from retained whole units:
 *  - round: the retained compact whole circles / domes, byte-identical to the
 *    bank rows (their own size; width variants are the different entries);
 *  - limb: one role per row: the compact circle's cap units (γ, ⌒ repeated as
 *    whole units, ヽ), straight side rows | ... | (a straight source side keeps
 *    an upright, D5), then its return units (ゝ, _ repeated, ノ) or, on a
 *    support contact, the dome's flat-base units ({, ＿ repeated, });
 *  - box: one row is 口 / ロ / ﾛ by width; more rows are the classic connected
 *    box: lid run ＿, sides |, base |＿＿| (D9: horizontals and verticals).
 * Text is authoritative: atoms are the native layout of the text rows (each
 * glyph's own advance), so exported text equals the placed construction.
 * Compiled once; the far consumer selects (kind, ends, rows, width) per pose. */
export interface SmallConstructionGlyph {readonly text:string;readonly advance:number}
export interface SmallConstructionRecord {
 readonly kind:SourceSmallParentKind;
 /** End constructions this record spells at the +axis (top row) and -axis
  * (bottom row) ends. A round record is closed at both ends. */
 readonly positiveEnd:SourceSmallParentEnd;readonly negativeEnd:SourceSmallParentEnd;
 readonly rows:number;
 /** Whole repeat units across (⌒ / ＿ count); 0 for a fixed retained entry. */
 readonly units:number;
 /** Provenance: retained entry ids and/or named whole units used. */
 readonly provenance:readonly string[];
 readonly text:readonly string[];
 /** [glyph index in the font, pen x px, row] for every ink glyph. */
 readonly atoms:readonly (readonly [number,number,number])[];
 /** Protected paper inside the outline: [row, start px, end px]. */
 readonly paper:readonly (readonly [number,number,number])[];
 readonly width:number;
}
const SPACE=' ',IDEOGRAPHIC='　';
/** One far page column (gpu-source-far-forms.ts: page x floored by 16 px). */
export const SMALL_CONSTRUCTION_COLUMN_PX=16;
/** A parent spans at most this many page columns (source-lod PARENT_MAX_COLUMNS). */
export const SMALL_CONSTRUCTION_MAX_COLUMNS=PARENT_MAX_COLUMNS;
/** Widest compiled small record: the whole parent profile width. */
export const SMALL_CONSTRUCTION_MAX_PX=SMALL_CONSTRUCTION_MAX_COLUMNS*SMALL_CONSTRUCTION_COLUMN_PX;
const entry=(id:string)=>[...ROLE_ENTRIES,...(roundSupplement.entries as unknown as typeof ROLE_ENTRIES)].find(value=>value.id===id);
/** Exact native whitespace for a gap of `px`: fewest glyphs of 11 (　) and 5 ( ). */
function gapText(px:number,font:ReadonlyMap<string,SmallConstructionGlyph>):string|undefined{
 const wide=font.get(IDEOGRAPHIC)?.advance,narrow=font.get(SPACE)?.advance;if(!wide||!narrow)return;
 for(let count=Math.floor(px/wide);count>=0;count--){const rest=px-count*wide;if(rest%narrow===0)return IDEOGRAPHIC.repeat(count)+SPACE.repeat(rest/narrow);}
 return undefined;
}
/** Nearest representable gap to `px` with its signed error (negative tucks
 * inside). `floor` takes the largest gap at or below `px`: a between-sides
 * gap then ends inside the cap extent, keeping the outline closed. `nearest`
 * takes the smallest absolute error (ties go below): for a leading offset
 * such as the box lid. 0 is always representable (empty string). */
function gapNear(px:number,font:ReadonlyMap<string,SmallConstructionGlyph>,mode:'floor'|'nearest'):{text:string;error:number}|undefined{
 if(!Number.isInteger(px))return;
 if(mode==='floor'){
  for(let want=px;want>=0;want--){const text=gapText(want,font);if(text!==undefined)return {text,error:want-px};}
  return;
 }
 for(let distance=0;;distance++){
  const below=gapText(px-distance,font);if(below!==undefined)return {text:below,error:-distance};
  const above=gapText(px+distance,font);if(above!==undefined)return {text:above,error:distance};
 }
}
function layout(text:readonly string[],font:ReadonlyMap<string,SmallConstructionGlyph>,index:ReadonlyMap<string,number>){
 const atoms:[number,number,number][]=[],paper:[number,number,number][]=[];let width=0;
 for(const [row,line]of text.entries()){
  let x=0,first=Infinity,last=-Infinity;const ink:[number,number][]=[];
  for(const ch of line){
   const glyph=font.get(ch);if(!glyph)throw Error(`Small construction glyph absent from the native font: ${JSON.stringify(ch)}`);
   if(ch!==SPACE&&ch!==IDEOGRAPHIC){atoms.push([index.get(ch)!,x,row]);ink.push([x,x+glyph.advance]);first=Math.min(first,x);last=Math.max(last,x+glyph.advance);}
   x+=glyph.advance;
  }
  width=Math.max(width,x);
  // Paper = whitespace strictly between ink inside the row's outline extent.
  for(let i=0;i+1<ink.length;i++)if(ink[i+1]![0]>ink[i]![1])paper.push([row,ink[i]![1],ink[i+1]![0]]);
 }
 return {atoms,paper,width};
}
/** The compiled records one source parent owns (compile-time family; the far
 * consumer picks rows/width among them per pose). Kind is the source's class;
 * the bottom row spells a support contact when the source's lower end is a
 * ROOT contact, otherwise a free return. An 'open' end has NO family (never
 * silently closed; see below). Formerly: Ends 'open' were closed as a free
 * return/cap: a small outline is closed (D9). Returns indices into records. */
export function smallConstructionsFor(parent:{readonly kind:SourceSmallParentKind;readonly positiveEnd:SourceSmallParentEnd;readonly negativeEnd:SourceSmallParentEnd},records:readonly SmallConstructionRecord[]):readonly number[]{
 // perf-impl-158: an OPEN end (a cut tube, an unclosed source end) has no
 // small vocabulary yet. It is never silently closed as a cap: the family is
 // empty, the incumbent keeps the form, and the gap stays an explicit
 // unfinished representation obligation.
 if(parent.positiveEnd==='open'||parent.negativeEnd==='open')return [];
 const contact=parent.negativeEnd==='contact'||parent.positiveEnd==='contact';
 const wanted=contact?'contact':'cap';
 const own=records.flatMap((record,index)=>record.kind===parent.kind?[index]:[]);
 const exact=own.filter(index=>records[index]!.negativeEnd===wanted);
 // D1 fixed order: the exact end records first. A CONTACT source may then
 // fall back to the same kind's closed cap records, but ONLY for row counts
 // with no exact record. A FREE (cap) source NEVER receives a contact
 // record: that would invent a ground line the source does not have.
 // Never across kinds.
 if(!contact)return [...exact];
 const covered=new Set(exact.map(index=>records[index]!.rows));
 return [...exact,...own.filter(index=>!exact.includes(index)&&!covered.has(records[index]!.rows)&&records[index]!.negativeEnd==='cap')];
}
export function compileSmallConstructions(glyphs:readonly NativeGlyph[]):readonly SmallConstructionRecord[]{
 const font=new Map(glyphs.map(glyph=>[glyph.text,glyph] as const)),index=new Map(glyphs.map((glyph,i)=>[glyph.text,i] as const));
 const advance=(text:string)=>[...text].reduce((sum,ch)=>{const glyph=font.get(ch);if(!glyph)throw Error(`Small construction glyph absent from the native font: ${JSON.stringify(ch)}`);return sum+glyph.advance;},0);
 // Native ink bounds of one glyph, for run lengths that match ink extents.
 const ink=(ch:string):readonly [number,number]=>{const glyph=font.get(ch);if(!glyph)throw Error(`Small construction glyph absent from the native font: ${JSON.stringify(ch)}`);let lo=Infinity,hi=-Infinity;for(const [x]of glyph.points){lo=Math.min(lo,x);hi=Math.max(hi,x+1);}return [lo,hi];};
 const inkOf=(text:string):readonly [number,number]=>{let pen=0,lo=Infinity,hi=-Infinity;for(const ch of text){const [a,b]=ink(ch);if(a<=b){lo=Math.min(lo,pen+a);hi=Math.max(hi,pen+b);}pen+=font.get(ch)!.advance;}return [lo,hi];};
 const records:SmallConstructionRecord[]=[];
 const push=(kind:SourceSmallParentKind,positiveEnd:SourceSmallParentEnd,negativeEnd:SourceSmallParentEnd,units:number,provenance:readonly string[],text:readonly string[])=>{
  const placed=layout(text,font,index);records.push({kind,positiveEnd,negativeEnd,rows:text.length,units,provenance,text,...placed});
 };
 // ROUND: retained compact wholes at their own size (rows 2/3/4). A support
 // contact takes the retained dome / grounded compact circle.
 for(const [id,ends]of [
  ['sh-lib67-circle-2row-tiny',['cap','cap']],['sh-lib66-circle-2row',['cap','cap']],
  ['sh-lib65-circle-3row-brace',['cap','cap']],['sh-lib64-circle-4row',['cap','cap']],
  ['sh-mlt05-dome-flat-base',['cap','contact']],['sh-mlt03-compact-circle',['cap','contact']],
 ] as const){
  const value=entry(id);if(!value)continue;
  push('round',ends[0],ends[1],0,[id],value.native.rows.map(nativeRoleText));
 }
 // LIMB and BOX: one role per row from named whole units.
 // WIDTH GROWTH (perf-impl-158): one whole ⌒ / ＿ unit is 16 px = one far
 // page column (gpu-source-far-forms.ts floors page x / 16), and a parent
 // spans at most PARENT_MAX_COLUMNS columns. Units therefore run until the
 // record would exceed SMALL_CONSTRUCTION_MAX_PX, never a fixed 1..4 table:
 // a 150 px beam gets its own ~9-unit record, not a 22 px box.
 const capUnits=entry('sh-lib66-circle-2row'),baseUnits=entry('sh-mlt05-dome-flat-base');
 const fits=(text:readonly string[])=>text.every(line=>advance(line)<=SMALL_CONSTRUCTION_MAX_PX);
 for(let units=1;units<=SMALL_CONSTRUCTION_MAX_COLUMNS;units++)for(let rows=2;rows<=5;rows++){
  if(capUnits&&advance('γ'+'⌒'.repeat(units)+'ヽ')<=SMALL_CONSTRUCTION_MAX_PX){
   const cap='γ'+'⌒'.repeat(units)+'ヽ',capWidth=advance(cap),sides=rows-2;
   // Sides: | under γ's left edge and | ending at ヽ's right edge. An
   // unreachable exact gap tucks inside (floor) rather than dropping rows.
   const gap=gapNear(capWidth-2*advance('|'),font,'floor')?.text;
   const [,capInkEnd]=inkOf(cap);
   for(const negativeEnd of ['cap','contact'] as const){
    let bottom:string|undefined,bottomUnit:string;
    // The return's ink end meets the cap's ink end: pen-width rounding alone
    // overshoots ink by 6px at 3 units (ノ runs long, ヽ trails paper).
    if(negativeEnd==='cap'){const run=Math.max(1,Math.round((capInkEnd-advance('ゝ')-ink('ノ')[1])/advance('_')));bottom='ゝ'+'_'.repeat(run)+'ノ';bottomUnit=`sh-lib66-circle-2row return units ゝ _x${run} ノ`;}
    else{if(!baseUnits)continue;const run=Math.max(1,Math.round((capWidth-2*advance('{'))/advance('＿')));bottom='{'+'＿'.repeat(run)+'}';bottomUnit=`sh-mlt05-dome-flat-base base units { ＿x${run} }`;}
    if(gap===undefined&&sides>0)continue;
    const text=[cap,...Array.from({length:sides},()=>'|'+gap+'|'),bottom!];if(!fits(text))continue;
    push('limb','cap',negativeEnd,units,[`sh-lib66-circle-2row cap units γ ⌒x${units} ヽ`,...(sides?['straight side unit |']:[]),bottomUnit],text);
    // A ROUND wider than every retained whole grows only through the 2-row
    // compact circle's / dome's OWN repeat units (⌒, _, ＿), at 2 rows: the
    // same entry widened by whole units, never a stretched glyph or a new row.
    if(rows===2&&units>=2)push('round','cap',negativeEnd,units,[`${negativeEnd==='cap'?'sh-lib66-circle-2row':'sh-mlt05-dome-flat-base'} widened by own repeat units (⌒x${units})`],text);
   }
  }
  // BOX rows >= 2: lid ＿ run sitting between the sides, base |＿...|.
  // The ideal 4px lid inset is unrepresentable in 11/5 whitespace; the
  // nearest (5px) keeps the right side touching the lid's ink.
  const lid=gapNear(advance('|'),font,'nearest')?.text??SPACE,inner=gapNear(units*advance('＿'),font,'floor')?.text;
  const boxText=[lid+'＿'.repeat(units),...Array.from({length:rows-2},()=>'|'+(inner??'')+'|'),'|'+'＿'.repeat(units)+'|'];
  if(inner!==undefined&&fits(boxText))push('box','cap','contact',units,['box lid unit ＿','straight side unit |','box base unit |＿|'],boxText);
 }
 // BOX one row: the one-glyph boxes by width (the compiled hard-word set).
 for(const [units,glyph]of [[1,'ﾛ'],[2,'ロ'],[3,'口']] as const)if(font.has(glyph))push('box','cap','contact',units,[`one-glyph box ${glyph}`],[glyph]);
 // 3/4-ROW ROUND FAMILIES (perf-impl-160; README "Codex continuation,
 // 2026-11-19"). Appended AFTER every earlier record, so existing record
 // indices and order are unchanged.
 //  - WIDER: the retained 3-row brace circle and 4-row circle grow only through
 //    their OWN repeat units, one 16 px page column per step: lib65 by its cap
 //    `―` (16 px) + interior paper + return `＿` (16 px); lib64 by two of its
 //    own `-` (8 px) in cap and bottom + interior paper in both flank rows.
 //    k=1 is the retained entry itself and is not re-emitted.
 //  - GROUNDED: a support contact is spelled as sh-mlt03-compact-circle does
 //    (`＿乂__ノ＿＿`): whole ground `＿` units touching the return glyph on
 //    both sides of the LAST row. Every row shifts by the same whitespace so
 //    the left ground run ends exactly at the return's first glyph; the round
 //    itself keeps its retained rows (no stretched glyph, no new form row).
 //  D5: these rounds never put an upright on one side in consecutive rows
 //  (their side glyphs are a single brace row / comma pairs).
 const leadPx=(line:string)=>{let px=0;for(const ch of line){if(ch!==SPACE&&ch!==IDEOGRAPHIC)break;px+=advance(ch);}return px;};
 const stripLead=(line:string)=>line.replace(/^[ 　]+/u,''),stripTrail=(line:string)=>line.replace(/[ 　]+$/u,'');
 // Widen the single interior whitespace run between a row's first and last ink.
 const widenPaper=(line:string,extra:number):string|undefined=>{
  const match=/^([ 　]*\S+)([ 　]+)(\S+[ 　]*)$/u.exec(line);if(!match)return undefined;
  const gap=gapText(advance(match[2]!)+extra,font);return gap===undefined?undefined:match[1]!+gap+match[3]!;
 };
 const grounded=(text:readonly string[]):{text:string[];units:number}|undefined=>{
  const last=text.at(-1)!,lead=leadPx(last),body=stripTrail(stripLead(last)),column=advance('＿');
  for(let units=1;units<=4;units++){
   const shift=units*column-lead;if(shift<0)continue;
   const prefix=shift===0?'':gapText(shift,font);if(prefix===undefined)continue;
   return {units,text:[...text.slice(0,-1).map(row=>prefix+row),'＿'.repeat(units)+body+'＿'.repeat(units)]};
  }
  return undefined;
 };
 const roundFamilies:{id:string;widen:(rows:readonly string[],k:number)=>string[]|undefined;unit:string}[]=[
  {id:'sh-lib65-circle-3row-brace',unit:'― / paper / ＿',widen:(rows,k)=>{
   const cap=rows[0]!.replace('―','―'.repeat(k)),flank=widenPaper(rows[1]!,(k-1)*advance('―')),base=rows[2]!.replace('＿','＿'.repeat(k));
   return flank===undefined?undefined:[cap,flank,base];
  }},
  {id:'sh-lib64-circle-4row',unit:'-- / paper / --',widen:(rows,k)=>{
   const cap=rows[0]!.replace('--','-'.repeat(2*k)),bottom=rows[3]!.replace('---','-'.repeat(2*k+1));
   const left=widenPaper(rows[1]!,(k-1)*2*advance('-')),right=widenPaper(rows[2]!,(k-1)*2*advance('-'));
   return left===undefined||right===undefined?undefined:[cap,left,right,bottom];
  }},
 ];
 for(const family of roundFamilies){
  const value=entry(family.id);if(!value)continue;
  const rows=value.native.rows.map(nativeRoleText);
  for(let k=1;k<=SMALL_CONSTRUCTION_MAX_COLUMNS;k++){
   const text=k===1?rows:family.widen(rows,k);if(!text||!fits(text))break;
   if(k>1)push('round','cap','cap',k,[`${family.id} widened by own repeat units (${family.unit}, k=${k})`],text);
   const ground=grounded(text);
   if(ground&&fits(ground.text))push('round','cap','contact',k,[`${family.id} ${k>1?`widened by own repeat units (${family.unit}, k=${k}) `:''}grounded by sh-mlt03-compact-circle ground units ＿x${ground.units}`],ground.text);
  }
 }
 // FREE BOX (perf-impl-162 companion): cap/cap records from the SAME retained
 // closed box units as above (lid ＿, sides |, base |＿...|, one-glyph boxes),
 // appended after every existing record. A free box's base is the box's own
 // closed bottom edge, never an outside ground run. Same units range and
 // 1..5 rows; existing cap/contact records stay available and unchanged.
 for(let units=1;units<=SMALL_CONSTRUCTION_MAX_COLUMNS;units++)for(let rows=2;rows<=5;rows++){
  const lid=gapNear(advance('|'),font,'nearest')?.text??SPACE,inner=gapNear(units*advance('＿'),font,'floor')?.text;
  const boxText=[lid+'＿'.repeat(units),...Array.from({length:rows-2},()=>'|'+(inner??'')+'|'),'|'+'＿'.repeat(units)+'|'];
  if(inner!==undefined&&fits(boxText))push('box','cap','cap',units,['box lid unit ＿','straight side unit |','box base unit |＿|'],boxText);
 }
 for(const [units,glyph]of [[1,'ﾛ'],[2,'ロ'],[3,'口']] as const)if(font.has(glyph))push('box','cap','cap',units,[`one-glyph box ${glyph}`],[glyph]);
 return records;
}
