// Unmodified build input from explorer 3f5b6a1 (includes 3493185).
// Pin the bytes in MAIN rather than silently changing an already-built bank.
import library from '../../../assets/grammar/aahub-role-lookup-3493185.v1.json' with {type:'json'};
import roundSupplement from './role-rounds-supplement.json' with {type:'json'};
import type {CompiledNativeTable,CompiledNativeWord} from './native-table.ts';
import {nativeSpacePlan} from '../text/native-whitespace.ts';
import base from '../../../assets/grammar/glyphs.json' with {type:'json'};
import local from './local-glyphs.json' with {type:'json'};

export interface RoleEntry {
  id:string;role:string;also_roles?:string[];size_band:{rows:number;px_wide:number;band:string};
  context:{tangent_in:string;tangent_out:string;curvature:string;inside:string;paper:string};
  native:{rows:string[];row_numbers:number[];row_pen_x:number[]};source:{src:string};status:{state:string};
}
export const ROLE_ENTRIES=(library.entries as unknown as RoleEntry[]).filter(e=>e.status.state==='retained');
export const OUTLINE_ROLES=['rim_cap','shoulder','flank','return_base','whole_circle','closed_ring','contact_junction','ellipse_rim','side'] as const;
export const nativeRoleText=(s:string)=>s.replace(/\\u([a-fA-F0-9]{4})/g,(_,n:string)=>String.fromCharCode(parseInt(n,16)));
const font=new Map([...base,...local].map(g=>[g.text,g]));
const advance=(text:string)=>[...text].reduce((n,ch)=>n+(ch===' '?5:ch==='　'?11:font.get(ch)?.advance??0),0);
const cupTone=ROLE_ENTRIES.find(e=>e.id==='sk-cup176-cylinder-tone')!;
/** Restore the previously visible rear source fields without imposing the
 * foreground small-circle contour-only context on those separate subjects. */
export const ROLE_REAR_TONE=Array.from({length:3},(_,level)=>cupTone.native.rows.slice(0,3).map(row=>
  /\.{3}[:;]*/u.exec(nativeRoleText(row))![0].slice(0,[3,6,9][level])));
/** Compact continuous source subfields, not a rescaled cup or isolated
 * punctuation fill. Preserve dots → colons → semicolons and lit-side paper. */
export const ROLE_COMPACT_TONE=Array.from({length:3},(_,level)=>cupTone.native.rows.slice(0,3).map((row,r)=>{
  const field=/\.{3}[:;]*/u.exec(nativeRoleText(row))![0];
  // A compact row is ONE complete native word, not independently admitted
  // chunks whose temporary gaps can reject the otherwise legal whole row.
  // Core r9/r10 keep their contiguous denser source segment under r8 dots.
  return level===2&&r>0?field.slice(3,9):field.slice(0,level===0?3:6);
}));
/** Exact contiguous ink-and-paper subfields of cup rows8–10. No spelling
 * mutation, repeated bundle fill, or independently thresholded source dots. */
export const ROLE_TONE_PASSAGES=['tone_fringe','tone_mid','tone_core'].map((role,level)=>({entry:cupTone,role,
  rows:cupTone.native.rows.slice(0,3).map((row,i)=>{
    const text=nativeRoleText(row),match=/[.:;][ .:;]*[.:;]/u.exec(text)!;
    const end=level===0?match[0].search(/[:;]/):level===1?match[0].indexOf(';'):-1;
    const spelling=(end<0?match[0]:match[0].slice(0,end)).trimEnd();
    return{text:spelling,x:cupTone.native.row_pen_x[i]!+advance(text.slice(0,match.index)),sourceRow:cupTone.native.row_numbers[0]!+i};
  })}));

/** A whole source token with both its native pen and its offset from the
 * selected entry's row origin.  These rows remain source-bound: consumers
 * may repeat a complete native unit, but must not stretch or split it. */
export interface NamedNativePassageRow {
  readonly text:string;
  readonly x:number;
  readonly relativeX:number;
  readonly advance:number;
  readonly sourceRow:number;
}
export interface NamedNativePassage {
  readonly entry:RoleEntry;
  readonly role:string;
  readonly source:string;
  readonly nativeWidth:number;
  readonly originX:number;
  readonly rows:readonly NamedNativePassageRow[];
  readonly relationship:string;
}
const sourceRowText=(entry:RoleEntry,row:number)=>{
  const first=entry.native.row_numbers[0]!;
  return nativeRoleText(entry.native.rows[row-first]!);
};
const sourceToken=(entry:RoleEntry,row:number,text:string):NamedNativePassageRow=>{
  const first=entry.native.row_numbers[0]!,index=row-first,source=sourceRowText(entry,row);
  const at=source.indexOf(text);
  if(at<0)throw new Error(`missing retained source token ${entry.id} row ${row} ${JSON.stringify(text)}`);
  const origin=entry.native.row_pen_x[index]!;
  const relativeX=advance(source.slice(0,at));
  return{text,x:origin+relativeX,relativeX,advance:advance(text),sourceRow:row};
};
const wallBrickBand=ROLE_ENTRIES.find(entry=>entry.id==='sk-church99-wall-brick-band')!;
const altarPanelFrame=ROLE_ENTRIES.find(entry=>entry.id==='sk-church104-altar-panel-frame')!;
/** C99's retained two-row periodic wall unit.  Odd rows and even rows keep
 * their original phase and native token widths; this is not a 170px spelling
 * requirement and is not a compact-box side/corner admission. */
export const ROLE_PLANAR_PERIODIC_PASSAGES:readonly NamedNativePassage[]=[
  {entry:wallBrickBand,role:'periodic_band',source:wallBrickBand.source.src,nativeWidth:37,originX:wallBrickBand.native.row_pen_x[2]!,
    rows:[sourceToken(wallBrickBand,23,'＿i;;::;:'),sourceToken(wallBrickBand,25,'＿i;;::;:')],
    relationship:'C99 rows23/25: complete ＿i;;::;: unit, native 37px, odd-row phase preserved.'},
  {entry:wallBrickBand,role:'periodic_band',source:wallBrickBand.source.src,nativeWidth:34,originX:wallBrickBand.native.row_pen_x[3]!,
    rows:[sourceToken(wallBrickBand,24,';;::;:＿'),sourceToken(wallBrickBand,26,';;::;:＿')],
    relationship:'C99 rows24/26: complete ;;::;:＿ unit, native 34px, even-row phase preserved.'},
] as const;
/** C104's exact bottom mixed-role words.  They preserve the same native
 * post/sill column; they support a post/sill phrase only, not a full compact
 * box side or missing corner role. */
export const ROLE_SUPPORT_JOINT_WORDS:readonly NamedNativePassage[]=[
  {entry:altarPanelFrame,role:altarPanelFrame.role,source:altarPanelFrame.source.src,nativeWidth:9,originX:altarPanelFrame.native.row_pen_x[6]!,
    rows:[sourceToken(altarPanelFrame,21,'_|')],
    relationship:'C104 row21 left joint: _| at native offset 0, 9px whole word under the left post.'},
  {entry:altarPanelFrame,role:altarPanelFrame.role,source:altarPanelFrame.source.src,nativeWidth:20,originX:altarPanelFrame.native.row_pen_x[6]!,
    rows:[sourceToken(altarPanelFrame,21,'|＿')],
    relationship:'C104 row21 right joint: |＿ at native offset 217, 20px whole word under the right post.'},
] as const;
const fogTail=ROLE_ENTRIES.find(e=>e.id==='sa-church126-wedge-thinning-tails')!;
/** Neighboring C126 rows25–28 supply the shadow stream's punctuation rhythm.
 * These are source subfields, not separate two-row brushes in the volume. */
export const ROLE_FOG_ROWS=fogTail.native.rows.slice(2,6).map((row,i)=>({entry:fogTail,
  text:/^[ .:;]+/u.exec(nativeRoleText(row))![0].trim(),sourceRow:25+i}));
export const ROLE_FEATURE_WORDS=ROLE_ENTRIES.flatMap(entry=>{
  // Tone roles can carry secondary edge/context tags; those tags do not turn
  // their bounded interior marks into contour strokes. Tone playback has its
  // own source-bound passage bank below.
  if(entry.role.startsWith('tone_'))return[];
  const roles=[entry.role,...entry.also_roles??[]].filter(role=>(OUTLINE_ROLES as readonly string[]).includes(role));
  if(!roles.length)return[];
  return entry.native.rows.flatMap((row,r)=>nativeRoleText(row).split(/[\s　]+/u).filter(Boolean).flatMap(run=>{
    const chars=[...run],words=[];
    for(let i=0;i<chars.length;i+=6){const text=chars.slice(i,i+6).join('');if([...text].every(ch=>font.get(ch)?.points.length))words.push({text,entry,roles,row:r});}
    return words;
  }));
});

/** Build-time compilation of COMPLETE retained closed constructions. Exact
 * glyphs and row correspondence survive; only canonical scalar whitespace is
 * adapted within the existing 3px native conversion allowance. No runtime fit. */
export function compileRoleCircles():CompiledNativeTable {
  // These complete constructions carry the inspected cap → shoulders,
  // paired flanks, and return at the sizes used by this scene. The dotted
  // sh-lib62 entry remains available to ordinary role lookup, but its repeated
  // lost contour spans are not a sphere-envelope size substitute.
  // The personally paired aa64 four-row block bridges the previous 3→8 row
  // jump. Its mirrored comma/quote sides are a complete source construction,
  // not isolated punctuation licensed as a generic flank.
  const wholeRoundedIds=new Set(['sh-lib57-circle-8row-whole','sh-lib60-circle-3row','sh-lib64-circle-4row','sh-lib66-circle-2row','rx-lib99-plain-round-7row']);
  // Keep the pinned instruction word order intact. Additional COMPLETE
  // constructions enter only this table, not the existing feature-word IDs.
  const entries=[...ROLE_ENTRIES,...roundSupplement.entries as RoleEntry[]].filter(e=>wholeRoundedIds.has(e.id)&&e.role==='whole_circle'&&e.source.src.startsWith('lib:shapes#'));
  const samples=entries.flatMap(entry=>{
    const rows=entry.native.rows.map(nativeRoleText),origin=Math.min(...entry.native.row_pen_x),words:CompiledNativeWord[]=[];
    for(const[row,text]of rows.entries()){
      // Shared full-space prefix keeps the original short leading half-space
      // offsets representable without changing any relationship in the block.
      let sourcePen=55+entry.native.row_pen_x[row]!-origin;
      let states:{pen:number;cost:number;words:CompiledNativeWord[]}[]=[{pen:0,cost:0,words:[]}];
      for(const run of text.matchAll(/[\s　]+|[^\s　]+/gu)){
        if(/^[\s　]+$/u.test(run[0])){for(const ch of run[0])sourcePen+=ch==='　'?11:5;continue;}
        if(![...run[0]].every(ch=>font.get(ch)?.points.length))return[];
        const advance=[...run[0]].reduce((n,ch)=>n+font.get(ch)!.advance,0);
        const next=new Map<number,typeof states[number]>();
        for(const state of states)for(const dx of[0,-1,1,-2,2,-3,3]){
          const x=sourcePen+dx;if(x<0||!nativeSpacePlan(x-state.pen,0,state.words.length===0))continue;
          const pen=x+advance,cost=state.cost+dx*dx;if((next.get(pen)?.cost??Infinity)<=cost)continue;
          next.set(pen,{pen,cost,words:[...state.words,{x,row,text:run[0],advance,feature:entry.id,role:entry.role}]});
        }
        states=[...next.values()];if(!states.length)return[];sourcePen+=advance;
      }
      words.push(...states.reduce((a,b)=>a.cost<=b.cost?a:b).words);
    }
    const right=Math.max(...words.map(w=>w.x+w.advance)),left=Math.min(...words.map(w=>w.x)),height=rows.length*17;
    return[{view:0,radius:height/2,anchor:[(left+right)/2,height/2] as const,extent:[right,height] as const,words,
      source:`unicode-glyph-morphology-explorer/assets/glyphs/roles/role_lookup.v1.json#${entry.id}`}];
  }).sort((a,b)=>a.radius-b.radius||a.source.localeCompare(b.source));
  // Native size is the projected HEIGHT, not width of a possibly elliptical
  // spelling. One complete source remains per height; no per-frame competition.
  const unique=samples.filter((s,i)=>i===0||s.radius!==samples[i-1]!.radius);
  return{schema:'trueff.native_table.v1',fontPx:16,rowPitch:17,samples:unique,views:[0],radii:unique.map(s=>s.radius),
    viewLookup:Array(256).fill(0),radiusThresholds:unique.slice(1).map(s=>s.radius),index:[unique.map((_,i)=>i)],
    provenance:{compiler:'packages/trueff/render/construct/role-bank.ts',sourceSha256:{'unicode-glyph-morphology-explorer/assets/glyphs/roles/role_lookup.v1.json':'a7b19015afa28463f382830ea24efb2bf626939f2daed47ed8d9d6d79782e64d'},maxWordShift:3}};
}

/** One row of whole native runs at their intended pens. Scalar gaps are made
 * legal by moving a whole run at most 5px (never splitting or stretching it);
 * the cheapest legal row is kept. Returns undefined when a glyph is missing or
 * no legal row exists, so a caller can never publish a partial envelope. */
function layoutEnvelopeRow(runs:readonly {text:string;x:number;role:string}[],row:number,feature:string):CompiledNativeWord[]|undefined{
  let states:{pen:number;cost:number;words:CompiledNativeWord[]}[]=[{pen:0,cost:0,words:[]}];
  for(const run of [...runs].sort((a,b)=>a.x-b.x)){
    if(![...run.text].every(ch=>font.get(ch)?.points.length))return undefined;
    const width=advance(run.text),next=new Map<number,typeof states[number]>();
    for(const state of states)for(const dx of[0,-1,1,-2,2,-3,3,-4,4,-5,5]){
      const x=Math.round(run.x)+dx;if(x<0||!nativeSpacePlan(x-state.pen,0,state.words.length===0))continue;
      const pen=x+width,cost=state.cost+dx*dx;if((next.get(pen)?.cost??Infinity)<=cost)continue;
      next.set(pen,{pen,cost,words:[...state.words,{x,row,text:run.text,advance:width,feature,role:run.role}]});
    }
    states=[...next.values()];if(!states.length)return undefined;
  }
  return states.reduce((a,b)=>a.cost<=b.cost?a:b).words;
}

/**
 * Sphere envelopes at EVERY projected row count, one construction family.
 *
 * Source: shapes aa[57] (sh-lib57-circle-8row-whole, `,.　-─-　 、` cap →
 * `／ ＼` shoulders → `/ ｀、` upper flank → `l l` flanks → `､ /` lower flank →
 * `＼ ／` lower shoulders → `｀ ｰ- -‐´` return, enclosed paper).  2/3/4-row
 * sizes keep the exact AAHub aa[66]/aa[60]/aa[64] entries.  5/6/7/9 rows are
 * construction-preserving reductions/extensions of aa57: the SAME role order,
 * whole native glyphs and enclosed paper; only repeated flank rows and interior
 * paper units are removed/added so the ink follows the projected circle chord.
 *
 * Tone: the cup176 / C126 graded wedge (dots → colons → semicolons) sits on the
 * shaded lower-right, against the right flank, growing toward the return and
 * thinning to protected paper toward the lit upper-left.  It is authored native
 * text per size (hand-authored target frame 000 rows 22–30), not a runtime fill.
 *
 * Selection is by projected silhouette HEIGHT (nearest row count), see
 * `selection` below and the native consumer in feature.ts.
 */
const F='　';
const SPHERE_ENVELOPE_ROWS:Readonly<Record<number,{rows:readonly string[];tone:readonly (readonly [number,string,number])[]}>>={
  5:{rows:[',.-─-、','／'+F.repeat(4)+'＼','l'+F.repeat(7)+'l','＼'+F.repeat(4)+'／','｀ｰ- -‐´'],
    tone:[[2,'.:',11],[3,'.:;:',5]]},
  6:{rows:[',. -─- 、','／'+F.repeat(5)+'＼','/'+F.repeat(7)+'｀、','､'+F.repeat(7)+' /','＼'+F.repeat(5)+'／','｀ ｰ- -‐´'],
    tone:[[3,'.:',11],[4,'.:;:;',5]]},
  7:{rows:[',.　-─- 、','／'+F.repeat(6)+'＼','/'+F.repeat(8)+'｀、','l'+F.repeat(10)+'l','､'+F.repeat(8)+' /','＼'+F.repeat(6)+'／','｀ ｰ-　-‐´'],
    tone:[[3,'.:',11],[4,':.:;:',5],[5,'.:;:;:',5]]},
  9:{rows:[',.　-─-　、','／'+F.repeat(8)+'＼','/'+F.repeat(10)+'｀、','l'+F.repeat(13)+'l','l'+F.repeat(13)+'l','l'+F.repeat(13)+'l','､'+F.repeat(11)+' /','＼'+F.repeat(8)+'／','｀ ｰ-　　-‐´'],
    tone:[[4,'.:',11],[5,':.:;:',5],[6,'.:;:;:',5],[7,':;:;:;',11]]},
};
const AA57_TONE:readonly (readonly [number,string,number])[]=[[3,'.:',11],[4,':.:;:',5],[5,'.:;:;:',5],[6,':;:;:;',11]];

/**
 * Loop 211: fog pendant lamp as ONE authored native construction (stem, shade
 * rim, underside return, bulb) instead of twelve box edges per stem/shade/bulb
 * spelled independently by the rounded role grammar (210: stacked 'ﾍﾍ', 'ヽi ､'
 * fragments with stray marks).
 *
 * Construction: hand-authored fog target frame 001 rows 5-10 ('|' suspension,
 * '＿___＿' shade rim, '｀ｰ-‐´' underside, '-' bulb); vessel rim over return
 * from AAHub Japanese-room aa[55] (circle profile 37) rows 21-23
 * ('{｀¨¨¨¨´}' over '乂;;;,,　ノ': a rim run over a returning underside with
 * paper between) and the shared sphere return spelling '｀ ｰ- -‐´'.  Three
 * related sizes keep the same role order; only stem rows and rim/underside
 * extents grow.  Runtime only indexes a size by projected radius.
 */
const PENDANT_SIZES:readonly {radius:number;stem:number;shade:string;under:string;bulb?:string}[]=[
  {radius:18,stem:2,shade:'＿_＿',under:'｀-‐´'},
  {radius:27,stem:3,shade:'＿___＿',under:'｀ｰ-‐´',bulb:'-'},
  {radius:42,stem:4,shade:'＿＿___＿＿',under:'｀ ｰ-　-‐´',bulb:'i'},
];
export function compilePendantConstruction():CompiledNativeTable {
  const samples:CompiledNativeTable['samples'][number][]=[];
  for(const size of PENDANT_SIZES){
    const axis=55+66,rows:string[]=[...Array(size.stem).fill('|'),size.shade,size.under,...(size.bulb?[size.bulb]:[])];
    const words:CompiledNativeWord[]=[];
    rows.forEach((text,row)=>{
      const runs:{text:string;x:number;role:string}[]=[];let pen=0;
      for(const run of text.matchAll(/[\s　]+|[^\s　]+/gu)){
        if(/^[\s　]+$/u.test(run[0])){for(const ch of run[0])pen+=ch==='　'?11:5;continue;}
        runs.push({text:run[0],x:pen,role:row<size.stem?'pendant-suspension':row===size.stem?'pendant-shade-rim':row===size.stem+1?'pendant-shade-return':'pendant-bulb'});
        pen+=advance(run[0]);
      }
      const lo=runs[0]!.x,hi=runs.at(-1)!.x+advance(runs.at(-1)!.text);
      const laid=layoutEnvelopeRow(runs.map(run=>({...run,x:run.x+axis-(lo+hi)/2})),row,'pendant-construction');
      if(!laid)throw Error(`Pendant construction row ${row} has no legal native spelling`);
      words.push(...laid);
    });
    const right=Math.max(...words.map(w=>w.x+w.advance));
    samples.push({view:0,radius:size.radius,anchor:[axis,(size.stem+.5)*17] as const,extent:[right,rows.length*17] as const,words,
      source:'hand-authored-targets-20261029/fog/rendered/001.txt rows 6-11 (rear pendant) + AAHub Japanese-room aa[55] rows 21-23 (vessel rim over return)'});
  }
  const radii=samples.map(s=>s.radius);
  return{schema:'trueff.native_table.v1',fontPx:16,rowPitch:17,samples,views:[0],radii,viewLookup:Array(256).fill(0),
    radiusThresholds:radii.slice(1).map((r,i)=>(r+radii[i]!)/2),index:[samples.map((_,i)=>i)],tonePassages:[[],[],[]],
    provenance:{compiler:'packages/trueff/render/construct/role-bank.ts#compilePendantConstruction',sourceSha256:{},maxWordShift:5}};
}
export const PENDANT_GLYPHS=[...new Set(PENDANT_SIZES.flatMap(s=>[...'|'+s.shade+s.under+(s.bulb??'')].filter(ch=>ch.trim())))];

export function compileSphereEnvelopes():CompiledNativeTable {
  const library=[...ROLE_ENTRIES,...roundSupplement.entries as RoleEntry[]];
  const entry=(id:string)=>library.find(e=>e.id===id&&e.role==='whole_circle'&&e.source.src.startsWith('lib:shapes#'));
  const samples:CompiledNativeTable['samples'][number][]=[];
  const finish=(rows:number,feature:string,source:string,structural:readonly {text:string;x:number}[][],tone:readonly (readonly [number,string,number])[])=>{
    const words:CompiledNativeWord[]=[];
    for(let row=0;row<rows;row++){
      const runs=structural[row]!.map(run=>({...run,role:'whole_circle'}));
      const wedge=tone.find(([r])=>r===row);
      if(wedge&&runs.length>=2){
        // Right-flank aligned: the wedge ends `gap` px before the right flank.
        const right=runs.reduce((a,b)=>a.x>b.x?a:b);
        for(const run of runsOf(wedge[1],right.x-wedge[2]-advance(wedge[1])))runs.push({...run,role:'tone'});
      }
      // Loop 218: the ground contact belongs to the resting sphere's own return
      // (tonemapping target frame 001 row 28 '｀ ｰ- .. -‐´', rows 27/29 '..'
      // in each front sphere's bottom-return gap), not to separate ground
      // specks a row below (207/213). '..' sits in the return's widest
      // interior gap when that gap can hold it with legal spaces.
      if(row===rows-1){
        const sorted=[...runs].sort((a,b)=>a.x-b.x);let best=-1,width=0;
        for(let i=0;i+1<sorted.length;i++){const gap=sorted[i+1]!.x-(sorted[i]!.x+advance(sorted[i]!.text));if(gap>width){width=gap;best=i;}}
        if(best>=0&&width>=11){const start=sorted[best]!.x+advance(sorted[best]!.text);
          runs.push({text:'..',x:start+Math.max(5,Math.round((width-advance('..'))/2)),role:'contact'});}
      }
      const laid=layoutEnvelopeRow(runs,row,feature);
      if(!laid)throw Error(`Sphere envelope ${feature} row ${row} has no legal native spelling`);
      words.push(...laid);
    }
    const right=Math.max(...words.map(w=>w.x+w.advance)),left=Math.min(...words.map(w=>w.x)),height=rows*17;
    samples.push({view:0,radius:height/2,anchor:[(left+right)/2,height/2] as const,extent:[right,height] as const,words,source});
  };
  const runsOf=(text:string,pen:number)=>{
    const runs:{text:string;x:number}[]=[];
    for(const run of text.matchAll(/[\s　]+|[^\s　]+/gu)){
      if(/^[\s　]+$/u.test(run[0])){for(const ch of run[0])pen+=ch==='　'?11:5;continue;}
      runs.push({text:run[0],x:pen});pen+=advance(run[0]);
    }
    return runs;
  };
  // Exact AAHub source rows at their original row pens (2/3/4/8 rows).
  for(const [id,tone] of [['sh-lib66-circle-2row',[]],['sh-lib60-circle-3row',[]],['sh-lib64-circle-4row',[]],['sh-lib57-circle-8row-whole',AA57_TONE]] as const){
    const e=entry(id);if(!e)throw Error(`Missing retained AAHub envelope ${id}`);
    const rows=e.native.rows.map(nativeRoleText),origin=Math.min(...e.native.row_pen_x);
    finish(rows.length,id,`unicode-glyph-morphology-explorer/assets/glyphs/roles/role_lookup.v1.json#${id}`,
      rows.map((text,r)=>runsOf(text,55+e.native.row_pen_x[r]!-origin)),tone);
  }
  // aa57-family sizes: every row centered on one axis by its ink extent.
  for(const [n,spec] of Object.entries(SPHERE_ENVELOPE_ROWS)){
    const rows=Number(n),axis=55+90;
    finish(rows,`sh-lib57-envelope-${rows}row`,`role_lookup.v1.json#sh-lib57-circle-8row-whole (construction-preserving ${rows}-row envelope)`,
      spec.rows.map(text=>{const runs=runsOf(text,0),lo=runs[0]!.x,hi=runs.at(-1)!.x+advance(runs.at(-1)!.text);
        return runs.map(run=>({...run,x:run.x+axis-(lo+hi)/2}));}),spec.tone);
  }
  samples.sort((a,b)=>a.radius-b.radius);
  const radii=samples.map(s=>s.radius);
  return{schema:'trueff.native_table.v1',fontPx:16,rowPitch:17,samples,views:[0],radii,viewLookup:Array(256).fill(0),
    // Nearest row count: a size is admitted at the midpoint between heights.
    radiusThresholds:radii.slice(1).map((r,i)=>(r+radii[i]!)/2),index:[samples.map((_,i)=>i)],
    selection:'projected-silhouette-height',
    provenance:{compiler:'packages/trueff/render/construct/role-bank.ts#compileSphereEnvelopes',sourceSha256:{'unicode-glyph-morphology-explorer/assets/glyphs/roles/role_lookup.v1.json':'a7b19015afa28463f382830ea24efb2bf626939f2daed47ed8d9d6d79782e64d'},maxWordShift:5}};
}
