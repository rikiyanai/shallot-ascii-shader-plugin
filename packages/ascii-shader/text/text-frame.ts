/** FL-4918: completed P5 rows are the shared paint/export interface. */
import {nativeSpacePlan,nativeSpaceCodes,type NativeWhitespacePolicy}from'./native-whitespace.ts';
export const ROW_PITCH = 17 as const;
export const FONT_PX = 16 as const;
export const FULL_SPACE = 0x3000 as const;
export const HALF_SPACE = 0x20 as const;
/** Operator-selectable placement; scene, structure and vocabulary stay shared. */
export type PresentationMode = 'text' | 'floating';

/** Character flag 16: a glyph the composition visibility decision occluded
 * behind a nearer run in a 'floating' presentation. Its record stays exact;
 * it is not displayed. Occlusion is decided by the shared P5 composition
 * pipeline (membership/visibility), never by a separate suppression path.
 * Consumers measuring visible glyphs (overlap, coverage) must skip it. */
export const OCCLUDED_FLAG = 16 as const;
/** Character flag 32: a glyph the realizer left out of this frame's layout
 * because the latest surface readback had it clearly hidden. It is shown
 * only if the current frame's visibility test finds it visible AND its ink
 * meets no laid-out ink (realize-gpu.ts, before raster); otherwise it is not
 * displayed (OCCLUDED in the verdict). No glyph can overlap another. */
export const PROVISIONAL_FLAG = 32 as const;
export type TextCharacter = readonly [
  codepoint: number, x: number, y: number, advance: number,
  flags: number, foreground: number, background: number,
];
export interface TextFontExtension {
  kind: 'godot_grass'; sha256: string; source_sha256: string; url: string;
  codepoints: readonly number[]; font_px: 16; advance_px: 8;
}
/** A composed native drawing, not a source mark or solver partition.
 * Members reference the current candidate page slots. drawing, when present,
 * owns their retained spelling/spacing; (keyframe,id) owns lifetime identity. */
export interface TextComposition {
  id: number;
  origin_px: readonly [number, number];
  translation_px: readonly [number, number];
  members: readonly number[];
  /** Birth frame plus id is the persistent identity, not the current row slot. */
  keyframe?: number;
  /** Retained native characters, ordered exactly like members. No glyph has
   * its own transform: translation_px applies to this entire drawing. */
  drawing?: readonly TextCharacter[];
  /** 'reconciled' (FL-4918 attempt 328): the retained drawing and identity
   * survived a local membership change (members joined or left). */
  transition?: 'retained'|'new'|'membership'|'deformation'|'page_boundary'|'collision'|'settled'|'rekeyed'|'reconciled';
  /** FL-4918 attempt 332: on a 'membership' re-key, the reconciliation clause
   * that failed. A failed reconciliation is an unresolved correspondence case,
   * not an authored keyframe; it is exported so it is never silent. */
  reconcileFailure?: 'member_limit'|'row_tolerance'|'page'|'advance_overlap'|'illegal_gap'|'half_space_run'|'no_continuing_sample';
  /** Historical schema (lane-A harness era): a floating run whose glyphs
   * carry OCCLUDED_FLAG named the occluding run (group id) and the clause
   * that decided the pair. No producer writes it after the Phase 2
   * replacement: occlusion is expressed by composition membership, and a
   * layout collision that cannot be solved legally is exported as a
   * 'collision' transition with its partner, never as a hidden suppression.
   * Retained for export-schema compatibility; never inferred when absent. */
  suppressedBy?: { id: number; clause: 'page'|'solid_nearer'|'incumbent'|'nearer'|'source_key'|'slot' };
  /** FL-4918 finding 2: on a True Text 'collision' re-key, what the colliding
   * partner was this frame: another translated proposal, or a component that
   * itself re-keyed (named by its own transition). Recorded at the decision. */
  collisionPartner?: 'proposal'|'new'|'membership'|'deformation'|'page_boundary'|'collision'|'settled'|'rekeyed'|'reconciled';
  /** 'actual': decided against the partners' actual representations;
   * 'conservative': only the closing round's partner-native test decided it. */
  collisionRound?: 'actual'|'conservative';
}
export interface TextFrame {
  schema: 'true_ff.text_frame.v1';
  frame: number;
  engine_frame: number;
  row_pitch_px: typeof ROW_PITCH;
  page_shift_px: readonly [number, number];
  font_sha256: string;
  /** Explicit replay contract. Preformatted text preserves ordinary spaces;
   * absence retains the existing constrained whitespace behavior. */
  whitespace?:NativeWhitespacePolicy;
  /** Explicit operator-approved glyph extension; never attributed to Saitamaar. */
  font_extensions?: readonly TextFontExtension[];
  /** Current legal candidate page. In positioned_runs this is NOT the final
   * drawing: consumers must use compositionCharacters/displayCharacters. */
  chars: readonly TextCharacter[];
  composition?: {
    kind: 'keyframe' | 'retained';
    keyframe: number;
    groups: readonly TextComposition[];
  };
  raster: { origin_px: readonly [number, number]; scale: 1; size_px: readonly [number, number] };
  page: {
    extent_px: readonly [number, number]; full_advance_px: 11; glyph_only: boolean;
    /** rows[r] = [end of last character, RGBA paper cells past that end].
     * Each cell spans 11 px, clipped at page width. These are NOT codepoints. */
    rows: readonly (readonly [number, readonly number[]])[];
  };
  export?: { dropped_frames: number; source_revision: string;
    /** Display verdicts of the frame's records (realize-gpu.ts): shown, hidden by the scene, provisional glyphs blocked by laid-out ink. */
    display_verdicts?: { shown: number; scene_hidden: number; provisional_blocked: number; crossing_break?: number } };
  /** Display honesty. Governing rule: operator goal V4 (~/Desktop/goals/
   * goal-v4.txt, rulings 2026-10-02 and 2026-10-04): in motion, intact runs
   * translate continuously in both axes with their features; no glyph moves
   * apart from its run; no fragments or overlaps; stationary compositions
   * (or selected keyframes) are exact text. Retained compositions preserve
   * their native drawing and declare the shared translation explicitly. */
  presentation?: { kind: 'exact_text' | 'positioned_runs' | 'floating' };
}

/** Validation never inserts positions/spaces to conceal a non-text row. */
export function validateTextFrame(frame: TextFrame, advances: ReadonlyMap<number, number>): void {
  const [width, height] = frame.page.extent_px;
  if (frame.schema !== 'true_ff.text_frame.v1' || frame.row_pitch_px !== ROW_PITCH ||
      !Number.isSafeInteger(width) || !Number.isSafeInteger(height) || width <= 0 || height <= 0 ||
      !/^[a-f0-9]{64}$/.test(frame.font_sha256)) throw new Error('Invalid native text frame header');
  const extensions=new Set<number>();
  for(const extension of frame.font_extensions??[]){
    if(extension.kind!=='godot_grass'||extension.font_px!==16||extension.advance_px!==8||
       !/^[a-f0-9]{64}$/.test(extension.sha256)||!/^[a-f0-9]{64}$/.test(extension.source_sha256)||
       !extension.url||extension.codepoints.length!==18)throw new Error('Invalid explicit font extension');
    for(const cp of extension.codepoints){if(!Number.isInteger(cp)||cp<0||cp>0x10ffff||(cp>=0xd800&&cp<=0xdfff)||extensions.has(cp)||advances.get(cp)!==8)throw new Error('Invalid extension scalar/advance');extensions.add(cp);}
  }
  let row = -1;
  let pen = 0;
  let halfRun = 0;
  const ends=new Map<number,number>();
  for (const [cp, x, y, advance, flags, fg, bg] of frame.chars) {
    if (![cp, x, y, advance, flags, fg, bg].every(Number.isSafeInteger) ||
        cp < 0 || cp > 0x10ffff || (cp >= 0xd800 && cp <= 0xdfff) ||
        y < row || y < 0 || y % ROW_PITCH !== 0 || y + ROW_PITCH > height ||
        x < 0 || x + advance > width || advance <= 0 || advances.get(cp) !== advance) {
      throw new Error(`Invalid native text character U+${cp.toString(16)} at ${x},${y}`);
    }
    if (y !== row) { row = y; pen = 0; halfRun = 0; }
    if (x !== pen) throw new Error(`Unserialized gap or overlap at ${x},${y}; expected ${pen}`);
    if (cp === HALF_SPACE) {
      if (frame.whitespace!=='preformatted'&&(pen === 0 || ++halfRun > 2)) throw new Error(`Invalid half-space grammar at ${x},${y}`);
    } else halfRun = 0;
    pen += advance;
    ends.set(y/ROW_PITCH,pen);
  }
  if(frame.raster.scale!==1||frame.page.full_advance_px!==11||
      frame.page.rows.length!==Math.ceil(height/ROW_PITCH)||
      frame.page_shift_px.length!==2||!frame.page_shift_px.every(Number.isFinite))throw new Error('Invalid text-page transport');
  frame.page.rows.forEach(([end,paper],index)=>{
    if(end!==(ends.get(index)??0)||paper.length!==Math.ceil((width-end)/11)||
        paper.some(color=>!Number.isSafeInteger(color)||color<0||color>0xffffffff))throw new Error(`Invalid paper tail at row ${index}`);
  });
  if(frame.composition){
    if(!['keyframe','retained'].includes(frame.composition.kind)||frame.composition.keyframe!==frame.frame)throw new Error('Invalid text composition frame');
    const ids=new Set<string>(),members=new Set<number>();
    for(const group of frame.composition.groups){
      const identity=`${group.keyframe??frame.frame}:${group.id}`;
      if(!Number.isSafeInteger(group.id)||group.id<0||ids.has(identity)||group.members.length===0||
          group.origin_px.length!==2||!group.origin_px.every(Number.isSafeInteger)||group.origin_px[1]%ROW_PITCH!==0||
          group.translation_px.length!==2||!group.translation_px.every(Number.isFinite))throw new Error(`Invalid native composition: group ${group.id} keyframe ${group.keyframe} duplicate=${ids.has(identity)} members=${group.members.length} origin=${JSON.stringify(group.origin_px)} translation=${JSON.stringify(group.translation_px)}`);
      if(frame.composition.kind==='keyframe'&&group.translation_px.some(v=>v!==0))throw new Error('Translated keyframe');
      if(frame.composition.kind==='retained'&&(!Number.isSafeInteger(group.keyframe)||group.keyframe!<1||group.keyframe!>frame.frame||group.drawing?.length!==group.members.length))throw new Error('Missing retained drawing/provenance');
      ids.add(identity);let previous=-1;
      for(const index of group.members){
        if(!Number.isSafeInteger(index)||index<=previous||index>=frame.chars.length||members.has(index)||
            (frame.chars[index]![4]&4)!==0)throw new Error('Duplicate, unordered or non-world composition member');
        members.add(index);previous=index;
      }
      if(group.drawing){
        let y=-1,end=0,half=0;
        for(const [cp,x,top,advance,flags,fg,bg] of [...group.drawing].sort((a,b)=>a[2]-b[2]||a[1]-b[1])){
          if(![cp,x,top,advance,flags,fg,bg].every(Number.isSafeInteger)||advances.get(cp)!==advance||
              top<y||top%ROW_PITCH!==0||x<0||top<0||x+advance>width||top+ROW_PITCH>height||(flags&4)!==0)throw new Error('Invalid retained native character');
          if(top===y){const gap=x-end,codes=nativeSpaceCodes(gap,half,false,cp===HALF_SPACE,frame.whitespace);if(!codes)throw new Error(`Retained drawing has an unspellable internal gap: group=${group.id} keyframe=${group.keyframe??frame.frame} transition=${group.transition} top=${top} gap=${gap}px before U+${cp.toString(16)} at x=${x} members=${group.members.length}`);for(const code of codes)half=code===HALF_SPACE?half+1:0;}
          else half=0;
          if(cp===HALF_SPACE){if(++half>2&&frame.whitespace!=='preformatted')throw new Error('Retained drawing violates half-space grammar');}else half=0;
          y=top;end=x+advance;
        }
        nativeCompositionRows(group,frame.whitespace);
      }
    }
  }
  const displayed=displayCharacters(frame);
  const exact=displayed.every((char,i)=>char.every((v,k)=>v===frame.chars[i]![k]));
  if((frame.presentation?.kind??'exact_text')==='exact_text'&&!exact)throw new Error('Positioned drawing falsely declared exact text');
}

export function plainText(frame: TextFrame, nativeRows?: readonly string[]): string {
  if(frame.presentation && frame.presentation.kind!=='exact_text')throw new Error('This frame requires positioning metadata and is not an exact plain-text page; export JSON');
  // Legacy exporters retain their byte contract. MAIN supplies the native
  // row manifest, including the final partial paper row of a2252px page.
  if(nativeRows&&nativeRows.length!==frame.page.rows.length)throw new Error('Native rows do not cover the displayed page');
  const rows = Array.from({ length: nativeRows?.length??Math.floor(frame.page.extent_px[1] / ROW_PITCH) }, () => '');
  for (const [cp, , y] of frame.chars) rows[y / ROW_PITCH] += String.fromCodePoint(cp);
  if(nativeRows&&rows.some((row,i)=>row!==nativeRows[i]))throw new Error('Native rows differ from the displayed characters');
  return rows.join('\n');
}

/** Native drawing before the ONE shared translation for each composition. */
export function compositionCharacters(frame:TextFrame):TextCharacter[]{
  const chars=[...frame.chars];
  for(const group of frame.composition?.groups??[])if(group.drawing)group.members.forEach((index,i)=>{chars[index]=group.drawing![i]!;});
  return chars;
}
/** Exact display/export reconstruction, including the GPU's float32 addition.
 * Page shift is separate and shared by paper, HUD and all compositions. */
export function displayCharacters(frame:TextFrame):TextCharacter[]{
  const chars=compositionCharacters(frame);
  for(const group of frame.composition?.groups??[])for(const index of group.members){const c=chars[index]!;chars[index]=[c[0],Math.fround(c[1]+group.translation_px[0]),Math.fround(c[2]+group.translation_px[1]),c[3],c[4],c[5],c[6]];}
  return chars;
}
/** One shared presentation decision for both production modes (FL-4918 Phase
 * 2): the label is a function of the frame, not a second pipeline. A frame
 * whose display reconstruction equals its native rows is exact text — even
 * in 'floating' mode at rest (a stationary composition is exact text; stopping
 * input is not an articulation event and never causes snapback). A translated
 * frame is 'positioned_runs' in True Text (exact-text keyframe plus intact-run
 * in-betweens) and 'floating' in Floating presentation (continuous positioned
 * runs). The export always carries the positioning metadata; paint reads the
 * same records, so export == rendered holds in every label. */
export function presentationKindFor(mode:PresentationMode,frame:TextFrame):NonNullable<TextFrame['presentation']>['kind']{
  const displayed=displayCharacters(frame);
  const exact=displayed.every((char,i)=>char.every((v,k)=>v===frame.chars[i]![k]));
  if(exact)return 'exact_text';
  return mode==='floating'?'floating':'positioned_runs';
}
/** Paint == export (FL-4918 Phase 2 export contract): the P6 painted records
 * (display positions after positioned-run offsets, in export order) must
 * agree with displayCharacters(frame) glyph for glyph. Records carrying
 * OCCLUDED_FLAG are not displayed and are skipped on both sides; everything
 * else compares exactly (positions are float32 on both sides). Throws on the
 * first mismatch; production-path tests call this on real constructions. */
export function assertPaintEqualsExport(frame:TextFrame,painted:readonly TextCharacter[]):void{
  const displayed=displayCharacters(frame).filter(c=>(c[4]&OCCLUDED_FLAG)===0);
  const visible=painted.filter(c=>(c[4]&OCCLUDED_FLAG)===0);
  if(visible.length!==displayed.length)throw new Error(`Paint/export count mismatch: painted ${visible.length} visible records, export ${displayed.length} visible characters`);
  for(let i=0;i<displayed.length;i++){
    const p=visible[i]!,e=displayed[i]!;
    if(p.some((v,k)=>v!==e[k]))throw new Error(`Paint/export mismatch at visible glyph ${i}/${displayed.length}: painted [${p.join(',')}] export [${e.join(',')}]`);
  }
}
/** An ordinary proportional-text serialization of ONE complete composition.
 * Its only positioning metadata is the returned origin plus the composition's
 * shared translation. Row offsets are actual legal whitespace, not CSS gaps. */
export function nativeCompositionRows(group:TextComposition,policy:NativeWhitespacePolicy='constrained'):{origin_px:readonly [number,number];rows:readonly (readonly TextCharacter[])[]}{
  if(!group.drawing?.length)throw new Error('Composition has no retained native drawing');
  const sorted=[...group.drawing].sort((a,b)=>a[2]-b[2]||a[1]-b[1]);
  const firsts=sorted.filter((c,i)=>i===0||c[2]!==sorted[i-1]![2]);
  const minimum=Math.min(...firsts.map(c=>c[1]));let originX=minimum;
  // Every leading gap >=95px is spellable before a non-half scalar; before
  // a half scalar the exact bound is106px. Seek the nearest common legal
  // origin, preserving all multiline relationships. These are complete
  // grammar bounds, not an iteration quota or a relaxed placement tolerance.
  const leadingBound=firsts.some(c=>c[0]===HALF_SPACE)?106:95;
  while(firsts.some(c=>!nativeSpacePlan(c[1]-originX,0,true,c[0]===HALF_SPACE,policy))){originX--;if(minimum-originX>leadingBound)throw new Error('Native composition indentation is not representable');}
  const originY=sorted[0]![2];const rows:TextCharacter[][]=[];
  for(let y=originY;y<=sorted.at(-1)![2];y+=ROW_PITCH){let pen=originX,halfRun=0;const row:TextCharacter[]=[];
    for(const char of sorted.filter(c=>c[2]===y)){
      const codes=nativeSpaceCodes(char[1]-pen,halfRun,row.length===0,char[0]===HALF_SPACE,policy);
      if(!codes)throw new Error('Native composition internal spacing is not representable');
      for(const cp of codes){const advance=cp===FULL_SPACE?11:5;row.push([cp,pen,y,advance,0,char[5],char[6]]);pen+=advance;halfRun=cp===HALF_SPACE?halfRun+1:0;}
      halfRun=char[0]===HALF_SPACE?halfRun+1:0;
      row.push(char);pen+=char[3];
    }rows.push(row);
  }
  return {origin_px:[originX,originY],rows};
}

export function rgbaCss(value: number): string {
  return `rgb(${value & 255} ${(value >>> 8) & 255} ${(value >>> 16) & 255} / ${((value >>> 24) & 255) / 255})`;
}
