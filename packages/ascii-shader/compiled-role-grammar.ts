import grammar from '../../assets/grammar/aahub-role-grammar-3493185.v1.json' with {type:'json'};
import {ROLE_FEATURE_WORDS,ROLE_ENTRIES,nativeRoleText} from './grammar/role-bank.ts';
import roundSupplement from './grammar/role-rounds-supplement.json' with {type:'json'};
import {nativeSpacePlan,type NativeWhitespacePolicy} from './text/native-whitespace.ts';
import type {NativeGlyph} from './native-page.ts';
import {TONE_PASSAGES} from './compiled-tone-passages.ts';
import {compileSmallConstructions} from './compiled-small-construction.ts';
export {WORD_STRIDE,GRAMMAR_RECORD_WORDS,GRAMMAR_DIRECTIONS,GRAMMAR_PHASES,GRAMMAR_ROLES,
  HARD_SLOPE_PHASES,PAIRED_FLANK_LEANS,TERRAIN_INSTRUCTION_WORDS,FOLIAGE_EDGES_AT,
  JUNCTION_DIRECTIONS,JUNCTION_TWO,JUNCTION_THREE,JUNCTION_RECORDS,
  SMALL_CONSTRUCTION_TABLE_AT,SMALL_CONSTRUCTION_TABLE_VERSION,SMALL_CONSTRUCTION_TABLE_HEADER_WORDS,
  SMALL_CONSTRUCTION_TABLE_RECORD_WORDS,SMALL_CONSTRUCTION_TABLE_WIDTH_LIMIT,SMALL_CONSTRUCTION_TABLE_KIND_COUNT,
  SMALL_CONSTRUCTION_TABLE_ROW_COUNT,SMALL_CONSTRUCTION_TABLE_INDEX_AT,SMALL_CONSTRUCTION_TABLE_INDEX_WORDS,
  SMALL_CONSTRUCTION_TABLE_RECORDS_AT,SMALL_CONSTRUCTION_TABLE_NONE} from './role-grammar-format.ts';
export type {GrammarRole,GrammarRoleEntry,CompiledRolePart,PackedRoleGrammar,PackedSmallConstructionRecord,PackedSmallConstructionTable} from './role-grammar-format.ts';
import {WORD_STRIDE,GRAMMAR_RECORD_WORDS,GRAMMAR_DIRECTIONS,GRAMMAR_PHASES,GRAMMAR_ROLES,
  HARD_SLOPE_PHASES,PAIRED_FLANK_LEANS,TERRAIN_INSTRUCTION_WORDS,FOLIAGE_EDGES_AT,
  JUNCTION_DIRECTIONS,JUNCTION_TWO,JUNCTION_THREE,JUNCTION_RECORDS,
  SOFT_PART_TABLE_VERSION,SOFT_PART_CONTEXTS,SOFT_PART_CONTEXT_IN_MASK,SOFT_PART_CONTEXT_OUT_SHIFT,SOFT_PART_CONTEXT_CLASS_SHIFT,SOFT_PART_CONTEXT_CLASS_MASK,
  SOFT_PART_TABLE_RECORD_WORDS,SOFT_PART_TABLE_RECORDS,SOFT_PART_TABLE_WORDS,SOFT_PART_TABLE_AT,
  SOFT_PHASE_TABLE_AT,SOFT_PHASE_TABLE_VERSION,SOFT_PHASE_TABLE_TABLES,SOFT_PHASE_TABLE_COLUMNS,SOFT_PHASE_TABLE_PHASES,
  SOFT_PHASE_TABLE_NONE,SOFT_PHASE_TABLE_WORDS_PER_TABLE,SOFT_PHASE_TABLE_HEADER_WORDS,SOFT_PHASE_TABLE_WORDS,
  SOFT_D5_TABLE_AT,SOFT_D5_TABLE_VERSION,SOFT_D5_TABLE_TABLES,SOFT_D5_TABLE_COLUMNS,SOFT_D5_TABLE_PHASES,
  SOFT_D5_TABLE_VARIANTS,SOFT_D5_TABLE_ENTRY_WORDS,SOFT_D5_TABLE_HEADER_WORDS,SOFT_D5_TABLE_NONE,SOFT_D5_TABLE_HARD,
  SOFT_D5_TABLE_UPRIGHT,SOFT_D5_TABLE_WORDS,
  SMALL_CONSTRUCTION_TABLE_AT,SMALL_CONSTRUCTION_TABLE_VERSION,SMALL_CONSTRUCTION_TABLE_HEADER_WORDS,
  SMALL_CONSTRUCTION_TABLE_RECORD_WORDS,SMALL_CONSTRUCTION_TABLE_WIDTH_LIMIT,SMALL_CONSTRUCTION_TABLE_KIND_COUNT,
  SMALL_CONSTRUCTION_TABLE_ROW_COUNT,SMALL_CONSTRUCTION_TABLE_INDEX_AT,SMALL_CONSTRUCTION_TABLE_INDEX_WORDS,
  SMALL_CONSTRUCTION_TABLE_RECORDS_AT,SMALL_CONSTRUCTION_TABLE_NONE,packSmallConstructionKindAndEnds,smallConstructionIndexOffset,
  type PackedRoleGrammar, type GrammarRole, type CompiledRolePart} from './role-grammar-format.ts';
/** Line classes by the row portion's slope |dy|/|dx| (page px). Under
 * TERRAIN_FLAT_SLOPE a terrain line is level ground (one level unit by
 * height); under TERRAIN_STEP_SLOPE (about 12 degrees) a hard edge is a level
 * ＿ / ￣. To TERRAIN_SLANT_SLOPE (17/24: a whole 17 px row crossed inside one
 * and a half full-width units, the playback's one-unit-per-row condition) the
 * row is the graded run; from there one best-fitting unit per row (SLOPE FIT
 * in packFeatureGrammar). */
export const TERRAIN_FLAT_SLOPE=.06;
export const TERRAIN_STEP_SLOPE=.2125,TERRAIN_SLANT_SLOPE=17/24;
/** Row phases (path height in the 17 px row) of a stepped shelf row: at or
 * above the first the path is high (￣), at or below the second low (＿). */
export const TERRAIN_STEP_PHASES=[5,11] as const;
/** Paired flank lean bounds, signed page px of horizontal travel per row going
 * down the page: ／ | / | upright | \\ | ＼. The same word spells both sides. */
/** A hard edge within this lean of vertical (|cos| of its page direction;
 * .26 is 15 degrees, the two direction bins either side of upright) is a run
 * of uprights whose pen follows the edge, one shift every few rows. A post or
 * a door jamb leans a few degrees in perspective; spelled with the 45 degree
 * slash on every row at nearly one pen it read as hatching (loop 351 view 02,
 * operator call-out). The bound is this rule's own choice. */
export const HARD_UPRIGHT_LEAN=.26;
/** A hard edge sloping between TERRAIN_STEP_SLOPE (about 12 degrees) and 45
 * degrees is the gentle-slope run of the fog target (frame 000: the floor
 * edge _,.-‐´ rising, ｀ヽ､､. falling) and the skill's slope vocabulary: one
 * unit by where the path stands in its row, so the run climbs through
 * _ , ‐ ´ ￣ (or falls through ￣ ｀ ‐ ､ _) instead of jumping between the two
 * cell-edge lines. Phase bounds (path height in the 17 px row, 0 = top) are
 * each unit's ink height in the native font. Level edges keep ＿ ￣. */
export const HARD_SLOPE_RISING=['￣','´','‐',',','_'] as const,HARD_SLOPE_FALLING=['￣','｀','‐','､','_'] as const;
/** Whole shallow-row passages, ordered by the unsigned 64-direction bin.
 * These are the graded-line construction, not five independent phase picks.
 * The fog target supplies _,.-‐´ / ｀ヽ､､.; the railway reference review
 * (ascii-art-archive 644a6de09, tetsudou-haikei/resK-137, lower track rows)
 * shows the same connected grade extended with whole horizontal members.
 * Related widths add/remove intervening whole units, never scale the font.
 * The reference is a construction witness, not an operator approval or a
 * runtime image. Bins 3..7 are the existing hard table's graded interval.
 */
export const HARD_SLOPE_PASSAGES={
 rising:['_,,.-‐´￣','_,.-‐‐´','_,.-‐´','_,.-´','_,‐´'],
 falling:['￣｀‐‐､__','￣｀‐､_','｀ヽ､､.','｀‐､_','｀ヽ_'],
} as const;

/** Compile the native ports and placement of one COMPLETE row passage.
 * Ink measurement places the already-authored unit; it never elects glyphs.
 * Advance alone is not a port: an underscore's 5px advance has only 4px of
 * ink on y15 and cannot stand in for a full 17px rise.
 */
export function hardSlopePassage(font:readonly NativeGlyph[],dir:number):{
 text:string;centre:number;entry:readonly [number,number];exit:readonly [number,number];
}|undefined{
 const angle=dir*Math.PI/32,dx=Math.cos(angle),dy=Math.sin(angle);
 const bin=Math.round(Math.atan2(Math.abs(dy),Math.abs(dx))*32/Math.PI);
 if(bin<3||bin>7)return undefined;
 const text=(dx*dy<0?HARD_SLOPE_PASSAGES.rising:HARD_SLOPE_PASSAGES.falling)[bin-3]!;
 const glyphs=new Map(font.map(g=>[g.text,g]));let pen=0;const ink:[number,number][]=[];
 for(const ch of text){const glyph=glyphs.get(ch);if(!glyph)throw Error(`Hard passage requires native scalar ${ch}`);
  for(const [x,y]of glyph.points)ink.push([pen+x!,y!]);pen+=glyph.advance;}
 const left=Math.min(...ink.map(p=>p[0])),right=Math.max(...ink.map(p=>p[0]));
 const edge=(x:number):readonly [number,number]=>{const ys=ink.filter(p=>p[0]===x).map(p=>p[1]);return [x,ys.reduce((a,b)=>a+b,0)/ys.length];};
 const travel=17*dx/dy,centre=ink.reduce((sum,[x,y])=>sum+x+.5-travel*(y+.5-8.5)/17,0)/ink.length;
 return {text,centre,entry:edge(left),exit:edge(right)};
}
/** CONIFER SIDE UNITS (tone lane 2026-11-13): AAHub 年中行事 nenchu-gyoji/resK-41
 * (クリスマスツリー, a straight-sided cone): rising side /ﾘ (line 10), ノ彡 (12),
 * ﾉリ (14), 彡ノ (15); falling side iヽ (10), ﾙﾊ (11), ゞ､ (12), ルヘ (16). */
export const FOLIAGE_EDGE_UNITS=['/ﾘ','ノ彡','ﾉリ','彡ノ','iヽ','ﾙﾊ','ゞ､','ルヘ'] as const;
/** The terrain instruction table's size (roles x sizes x directions x phases x record words). */
/** SLOPE UNITS: every AAHub editor palette part (ascii-art-archive c35ec01d6,
 * collections/aahub-editor-parts/parts.json, categories 斜線 点・チョン ノ・フ
 * 彡・イ へ・ハ リ・レ 組合せ１-３ 基本 く・Ｘ 十・Ｔ・ト 横線) that the native
 * font holds as 1-3 scalars without spaces, box-drawing glyphs excluded (the
 * palette report lists them as a contour conflict). A line row takes the unit
 * whose ink lies closest along that row's slope (fitSlopeUnit), measured once
 * here on the native font; the frame reads the result by address. Standalone
 * digits and Latin letters (1 7 y Z k T A X x Y and full-width forms) are not
 * candidates: a stack of them reads as text (council codex-20261111-cliff-crown:
 * "kl reads as letters"); l ｌ are strokes and stay. */
export const SLOPE_UNITS=["＼","丶","ヽ","ヾ","ゝ","ゞ","´","¨","｀","ﾞ","ﾟ","彡","∠","⌒","乂","亅","亠","从","仆","儿","兀","冂","冖","冫","几","凵","勹","匕","匚","匸","卅","卞","厂","叮","叭","叨","夂","宀","幵","广","廾","弋","_","￣","-","‐","ｰ","￤","|","∥","〕","［","］","「","」","=","＝","二","≡","三","ｌ","∈","∋","⊆","⊇","⊂","⊃","∪","∩","／","/","〆","⊿","％","‰","♂","≠","〃","∧","∨","キ","ｷ","ﾔ","∀","〈","〉","゛","'","′","〟","〝","、","､","（","(","巛","<","＜","《","へ","ヘ","ﾍ","ミ","ﾐ","丿","ノ","ﾉ","ソ","ｿ","）",")",">","＞","》","∟","く","戈","弌","弍","弐","式","戔","戉","戊","戍","戌","成","戒","廴","辷","込","×","※","米","＊","*","メ","ﾒ","ツ","ﾂ","ン","ﾝ","フ","ﾌ","ス","ｽ","ヌ","ﾇ","又","叉","双","ヲ","ｦ","ラ","ﾗ","ワ","ﾜ","ヮ","仄","√","尸","尺","戸","丘","反","氏","民","十","＋","+","卉","孑","±","丁","汀","行","庁","六","亢","山","屮","丱","艸","出","卜","ト","ﾄ","モ","ﾓ","下","寸","忖","可","斗","才","癶","ハ","ﾊ","八","公","兮","只","穴","价","ネ","ﾈ","之","乏","疋","片","爿","淵","乙","人","入","父","艾","爻","个","仝","今","夲","天","大","太","犬","夫","夭","夾","來","丈","史","介","リ","ﾘ","ル","ﾙ","レ","ﾚ","㍉","尢","尤","无","旡","牙","矛","朮","ホ","ﾎ","凡","丹","舟","咒","冗","瓜","爪","斤","斥","九","丸","彳","イ","ィ","ｨ","ｲ","〒","テ","ﾃ","チ","ﾁ","形","杉","尨","豸","豕","亥","毛","毟","气","￣｀","丶､","｀ヽ","｀丶","ヽ､","｀､","',","´￣",",.'",",'","しへ","しﾍ","ﾚへ","ﾚﾍ","レへ","レﾍ","ﾚ'7","ノ7","lﾝ","iﾉ","ﾚ'","レ'","メ、","へ、","_,ノ",",ノ",",ィ",",ｨ",",ｲ",",ﾉ","ト､","ﾄ､","｀Y´","ｔ'","f´￣","⌒て","i'´","`￣’","┬く","r―","rく","rｹ","--イ","l!","i!","|!","l|","ll","i}","{!","{i","||","|l","|i","!i","!l","!|","!}","|,リ","l,ﾘ","l,!","ｌル'","ｌﾙ'","k.l","ヽ!","ﾄ､!","|八|","|儿!","ｌハﾉ","!ﾊﾉ","ハ川","ﾞl刈"] as const;
/** HOOD EDGE COMBINATIONS (operator 2026-11-12: "IF DONE CORRECTLY, IT SHOULD
 * BE A COMBINATION, SEE THE 'HOODED FIGURE' AAHUB REFERENCE LIST"). AAHub
 * 人間・モブ11（フード） (ascii-art-archive collections/aahub/ningen-mobu-11,
 * 373 pieces): the first and last ink token of every row is where the hood or
 * robe outline crosses it; every 2-3 scalar token seen there 6 times or more,
 * without interior tone (: ;), box drawing or letters. A slope is spelled by
 * a stroke with its dot or comma (./ .／ ,／ /, .ﾊ .∨ ﾍ', 〃', ｀ヽ ヽ､ ヽ.)
 * or a doubled stroke (／／ ＼＼), not one bare glyph; the fit chooses among
 * these and SLOPE_UNITS together. Extraction:
 * scratchpad hood_edges.py 6 | hood_units.py, 2026-11-12. */
export const HOOD_EDGE_UNITS=["./","|l",".|",".ﾊ",".／","／／","＼＼",".∨","｀ヽ","＞､","ﾞ|","卜、_","',!","ゝ{","／＿","ヽ、",".′","_|／","∨.","〃',","/,",",′",",／","/|","ﾍ',","∨/","ヽ､","_..","/／","/!",".＼","ヽ.","／￣","'，",".!","|/","/ヽ","／|","／/","ﾞ,","l|","ﾘﾊj","_,.",",.ｨ","＼/","｝＼","'´","｀、","∨/∧","/l","＿／","'､",",!"] as const;
/** PRACTITIONER SLOPE RECIPES (skill references/aa-shokunin-tutorials.md,
 * posts 01 and 05): between the clean-line angles a slope is a thin glyph
 * reinforced by a small one (ｷ 2 dots a row, ,〈 / 〉, 11, .､≪ / ≫､. 27,
 * ､＜ / ＞､ 36) or a join glyph (ﾉ ｲ 厶 丿 人 八 〉､). */
export const PRACTITIONER_SLOPE_UNITS=["ｷ",",〈","〉,",".､≪","≫､.","､＜","＞､","厶","ｲ","丿","人","八","〉､","丶"] as const;
/** Upright pairs: a slanted line never takes them on any form (D5). */
export const UPRIGHT_STACKS=new Set(['∥','||','ll','|l','l|','!|','|!','!l','l!','|i','!i','i!']);
/** Lone strokes a natural form (terrain, rock, cliff) never takes alone
 * (operator 2026-11-10 ledger), plus the upright pairs. */
export const NATURAL_LONE_STROKES=new Set(['|','/','_','＿','／','＼','￣','-','‐','ｰ','￤',...UPRIGHT_STACKS]);
/** A unit holding a Latin letter other than the narrow strokes l i j ｌ ｉ reads
 * as text (skill 4.3, 15.6; council codex-20261111-cliff-crown "kl reads as
 * letters"): never a line, slope or junction unit. */
export function readsAsLetter(text:string):boolean{
 return [...text].some(ch=>/[A-Za-zＡ-Ｚａ-ｚ0-9０-９]/.test(ch)&&!'lijｌｉ'.includes(ch));
}
/** True when the glyph holding most of a unit's ink is an upright: the
 * principal axis of that glyph's own ink lies within 10 degrees of vertical
 * (a tick or dot beside it, ﾞ| ,! .|, does not make it a slant). */
export function uprightUnit(font:readonly NativeGlyph[],address:(text:string)=>number,text:string):boolean{
 let main:readonly (readonly number[])[]=[];
 for(const ch of text){const points=font[address(ch)]!.points;if(points.length>main.length)main=points;}
 const n=main.length;if(n<2)return false;
 let mx=0,my=0;for(const q of main){mx+=q[0]!;my+=q[1]!;}mx/=n;my/=n;
 let sxx=0,syy=0,sxy=0;for(const q of main){sxx+=(q[0]!-mx)**2;syy+=(q[1]!-my)**2;sxy+=(q[0]!-mx)*(q[1]!-my);}
 const axis=.5*Math.atan2(2*sxy,sxx-syy);return Math.abs(Math.abs(axis)-Math.PI/2)<=10*Math.PI/180;
}
/** LINES3 STROKE: a glyph carries a stroke when its largest 8-connected ink
 * component holds 5+ points spanning 5+ px along its own axis, or it is a
 * level rule 4+ px long (_ ￣ ‐). A dot, tick or comma (' , . ‘ ´ ｀ ､ 、 ﾞ),
 * a pair of them (: ; ¨ " ″) and a ring (｡ ﾟ) carry none: a line unit made of
 * them alone draws a dotted trail (D1 dots-only, D9). */
export function strokeGlyph(font:readonly NativeGlyph[],address:(text:string)=>number,ch:string):boolean{
 const points=font[address(ch)]?.points??[];if(points.length<4)return false;
 const xs=points.map(q=>q[0]!),ys=points.map(q=>q[1]!);
 if(Math.max(...ys)===Math.min(...ys)&&Math.max(...xs)-Math.min(...xs)+1>=4)return true;
 const seen=new Set<number>();let largest:(readonly number[])[]=[];
 for(let i=0;i<points.length;i++){if(seen.has(i))continue;const component=[i];seen.add(i);
  for(let k=0;k<component.length;k++){const a=points[component[k]!]!;for(let j=0;j<points.length;j++){if(seen.has(j))continue;const b=points[j]!;if(Math.abs(a[0]!-b[0]!)<=1&&Math.abs(a[1]!-b[1]!)<=1){seen.add(j);component.push(j);}}}
  if(component.length>largest.length)largest=component.map(c=>points[c]!);}
 if(largest.length<5)return false;
 const n=largest.length;let mx=0,my=0;for(const q of largest){mx+=q[0]!;my+=q[1]!;}mx/=n;my/=n;
 let sxx=0,syy=0,sxy=0;for(const q of largest){sxx+=(q[0]!-mx)**2;syy+=(q[1]!-my)**2;sxy+=(q[0]!-mx)*(q[1]!-my);}
 const axis=.5*Math.atan2(2*sxy,sxx-syy),along=largest.map(q=>(q[0]!-mx)*Math.cos(axis)+(q[1]!-my)*Math.sin(axis));
 return Math.max(...along)-Math.min(...along)+1>=5;
}
/** How well a unit's ink carries a straight line crossing one 17 px row with
 * `travel` page px of horizontal travel (positive: falls to the right going
 * down). The unit is centred where its ink sits on the line; the score adds
 * the ink's RMS distance from the line (px), a quarter px for each of the
 * row's 17 pixel rows the line crosses with no ink within 1.8 px (the line
 * not carried), and 4 x the share of ink more than 2 px off the line (無駄線,
 * D8). Lower is better. Returns the score and the ink x at mid-row. */
export function fitSlopeUnit(font:readonly NativeGlyph[],address:(text:string)=>number,text:string,travel:number):{score:number;centre:number}{
 let pen=0;const ink:[number,number][]=[];
 for(const ch of text){const glyph=font[address(ch)]!;for(const q of glyph.points)ink.push([pen+q[0]!+.5,q[1]!+.5]);pen+=glyph.advance;}
 if(ink.length<3)return {score:Infinity,centre:0};
 const line=(y:number)=>travel*(y-8.5)/17;
 const centre=ink.reduce((sum,[x,y])=>sum+x-line(y),0)/ink.length;
 const across=17/Math.hypot(travel,17);
 const off=ink.map(([x,y])=>Math.abs(x-centre-line(y))*across);
 const rms=Math.sqrt(off.reduce((sum,d)=>sum+d*d,0)/off.length);
 let gaps=0;for(let row=0;row<17;row++)if(!ink.some(([x,y],i)=>Math.abs(y-row-.5)<=1&&off[i]!<=1.8))gaps++;
 const stray=off.filter(d=>d>2).length/off.length;
 return {score:rms+.25*gaps+4*stray,centre};
}
/** JUNCTION UNITS (operator 2026-11-12: "IT SHOULD BE AUTOMATICALLY CHOOSING
 * JUNCTION GLYPHS AND PATTERNS THAT BEST FIT VIA COMPILED TABLE LOOKUP"). The
 * candidates for the ONE glyph where two or three runs meet: every AAHub
 * editor palette part (ascii-art-archive c35ec01d6,
 * collections/aahub-editor-parts/parts.json, categories 十・Ｔ・ト juu, く・Ｘ
 * ku-x, 点・チョン ten, 横線・冖 yokosen, ノ・フ・厂 no-fu, リ・レ・ホ ri-re,
 * へ・ハ・人 he-ha, 彡・イ mi-i, 組合せ１-３ kumiawase1-3) of 1-3 scalars, and
 * each scalar of a longer part (the 組合せ２ glyph lists 厂「｢丁…), that the
 * native font holds with ink, without spaces; box drawing (─ ┬ ┼ …, a contour
 * conflict in the palette report) and standalone Latin letters and digits
 * (T L Y k X ７ 1 …: a stack of them reads as text, council
 * codex-20261111-cliff-crown) are not candidates; l and ｌ are strokes and
 * stay. Extraction: scratchpad junction/candidates.ts, 2026-11-12. The
 * former hand lists (HARD_JUNCTIONS ∧ ヘ V < > ＜ ＞ 〈 〉 「 L 」 ７ ∠ ﾚ レ ﾉ
 * ノ, NATURAL_JUNCTIONS 廴 ﾉ ノ ｒ ヽ ﾍ ∧ ∨) are replaced by the measurement
 * below over these units plus HOOD_EDGE_UNITS and PRACTITIONER_SLOPE_UNITS
 * (the skill's joins ﾉ ｲ 厶 丿 人 八 〉､, aa-shokunin-tutorials post 05). */
export const JUNCTION_UNITS=["_","￣","-","‐","ｰ","冖","￤","亅","|","∥","〕","［","］","「","」","=","＝","二","≡","三","∈","∋","⊆","⊇","⊂","⊃","∪","∩","＼","｀","´","゛","ﾞ","'","′","〟","〝","¨","、","､","丶","ヽ","ヾ","ゝ","ゞ","（","(","⌒","巛","<","〈","＜","《","へ","ヘ","ﾍ","ミ","ﾐ","／","/","〃","丿","ノ","ﾉ","ソ","ｿ","）",")",">","〉","＞","》","〆","⊿","彡","∠","∧","∨","从","∟","く","弋","戈","弌","弍","弐","式","戔","戉","戊","戍","戌","成","戒","廴","辷","込","冫","×","※","米","＊","*","メ","ﾒ","ツ","ﾂ","ン","ﾝ","フ","ﾌ","ス","ｽ","ヌ","ﾇ","又","叉","双","ヲ","ｦ","ラ","ﾗ","ワ","ﾜ","ヮ","厂","仄","√","广","尸","尺","戸","丘","反","氏","民","十","＋","+","卉","卞","孑","匕","±","丁","汀","行","庁","亠","六","亢","山","屮","丱","艸","出","卜","ト","ﾄ","モ","ﾓ","下","寸","忖","可","斗","才","癶","ハ","ﾊ","八","公","兮","只","穴","价","ネ","ﾈ","之","乏","疋","片","爿","淵","乙","人","入","乂","父","艾","爻","个","仝","今","夲","天","大","太","犬","夫","夭","夾","來","丈","史","介","リ","ﾘ","ル","ﾙ","レ","ﾚ","㍉","儿","兀","尢","尤","无","旡","牙","矛","朮","ホ","ﾎ","几","凡","丹","舟","咒","冗","瓜","爪","斤","斥","九","丸","彳","イ","ィ","ｨ","ｲ","〒","テ","ﾃ","チ","ﾁ","形","杉","尨","豸","豕","亥","毛","毟","气","￣｀","丶､","｀ヽ","｀丶","ヽ､","｀､","',","´￣",",.'",",'","しへ","しﾍ","ﾚへ","ﾚﾍ","レへ","レﾍ","ﾚ'7","ノ7","lﾝ","iﾉ","ﾚ'","レ'","メ、","へ、","_,ノ",",ノ",",ィ",",ｨ",",ｲ",",ﾉ","ト､","ﾄ､","｀Y´","ｔ'","f´￣","⌒て","i'´","`￣’","r―","rく","rｹ","--イ","＿",",",".","l!","i!","|!","l|","ll","i}","{!","{i","||","|l","|i","!i","!l","!|","!}","l","ｌ","｣","!","ﾆ","ニ","｢","土","士","|,リ","l,ﾘ","l,!","ｌル'","ｌﾙ'","k.l","ヽ!","ﾄ､!","|八|","|儿!","ｌハﾉ","!ﾊﾉ","仆","ハ川","ﾞl刈"] as const;
/** Acceptance of a measured junction: every leg carried over at least this
 * share of its length out to the unit's edge, at most this share of the
 * unit's ink off the legs (無駄線, D8), and a score (measureJunction) of at most
 * JUNCTION_MOST_SCORE: two legs each a quarter short, or one leg a third short
 * with a little stray ink. A leg set whose best unit misses any bound keeps
 * "the runs simply meet" (record word 0). */
export const JUNCTION_LEAST_COVER=.7,JUNCTION_MOST_STRAY=.125,JUNCTION_STRAY_WEIGHT=3,JUNCTION_MOST_SCORE=.6;
/** A leg of a junction unit is the ray from the vertex in the leg's direction
 * to the edge of the unit's own box (its advance by the 17 px row); a leg
 * shorter than JUNCTION_LEAST_LEG px inside the box carries nothing. The
 * measurement for one unit, shared by fitJunctionUnit and the table compiler:
 * the unit's ink (pixel centres), and per candidate vertex (each ink pixel)
 * and per sixteenth the share of the ray (from 2 px out) passing within one
 * pixel of ink, and the ray's length. */
export const JUNCTION_LEAST_LEG=4,JUNCTION_LEAST_RISE=6;
/** A level leg (east, west) may be as short as a half-width glyph's half
 * (」: the lid unit beside it carries the rest); any leg that climbs or falls
 * needs JUNCTION_LEAST_RISE px, so a mark in a row's corner (〝, the ﾞ of ﾞ|)
 * is not a leg. */
const leastLeg=(d:number)=>d%8===0?JUNCTION_LEAST_LEG:JUNCTION_LEAST_RISE;
export interface JunctionGauge {
 readonly text:string;readonly ink:readonly (readonly [number,number])[];
 /** [vertex*16+direction]: share of the leg carried, its length (px). */
 readonly cover:Float32Array;readonly reach:Float32Array;
}
export function junctionGauge(font:readonly NativeGlyph[],address:(text:string)=>number,text:string):JunctionGauge{
 let pen=0;const ink:[number,number][]=[];
 for(const ch of text){const glyph=font[address(ch)]!;for(const q of glyph.points)ink.push([pen+q[0]!+.5,q[1]!+.5]);pen+=glyph.advance;}
 const width=Math.max(1,pen),raster=new Uint8Array(width*17);
 for(const [x,y]of ink)raster[Math.floor(y)*width+Math.floor(x)]=1;
 const inked=(x:number,y:number)=>{const px=Math.floor(x),py=Math.floor(y);return px>=0&&px<width&&py>=0&&py<17&&raster[py*width+px]===1;};
 const cover=new Float32Array(ink.length*16),reach=new Float32Array(ink.length*16);
 for(const [v,[vx,vy]]of ink.entries()){for(let d=0;d<16;d++){
  const ux=Math.cos(d*Math.PI/8),uy=Math.sin(d*Math.PI/8);
  const tx=Math.abs(ux)<1e-9?Infinity:(ux>0?width-vx:vx)/Math.abs(ux),ty=Math.abs(uy)<1e-9?Infinity:(uy>0?17-vy:vy)/Math.abs(uy);
  const length=Math.min(tx,ty);reach[v*16+d]=length;if(length<JUNCTION_LEAST_LEG)continue;
  // The leg is carried from the vertex outward by ONE connected stroke along
  // it: a sample is inked when ink lies on it or beside it ACROSS the leg
  // (never along it): one pixel for a level or upright leg, two for an
  // oblique one (a glyph's slant lies up to half a sixteenth, 11 degrees,
  // off the leg); the share runs to the last inked sample before the first
  // gap of two samples, so a separate tick or dot (ﾞ beside |, the . of .|,
  // the two marks of 〝) carries nothing.
  const side=d%4===0?1.25:2;
  let samples=0,carried=0,gap=0,joined=true;
  for(let s=1;s<=length;s++){samples++;const x=vx+s*ux,y=vy+s*uy;let hit=false;
   for(let py=Math.floor(y)-2;py<=Math.floor(y)+2&&!hit;py++)for(let px=Math.floor(x)-2;px<=Math.floor(x)+2&&!hit;px++){
    if(!inked(px,py))continue;const wx=px+.5-x,wy=py+.5-y;hit=Math.abs(wx*ux+wy*uy)<=.75&&Math.abs(wx*uy-wy*ux)<=side;}
   if(hit){if(joined)carried=samples;gap=0;}else if(++gap>=2)joined=false;}
  cover[v*16+d]=carried/samples;
 }}
 return {text,ink,cover,reach};
}
/** How well a measured unit spells a junction whose legs leave its vertex in
 * `legs` (sixteenths). Over every ink pixel as the vertex: every leg must be
 * leastLeg px or longer inside the unit; the score adds the mean and
 * the worst share of a leg NOT carried by ink out to the unit's edge, and
 * JUNCTION_STRAY_WEIGHT x the share of the unit's ink lying off every leg
 * (more than 1.5 px from the leg's ray and from the vertex: 無駄線, D8). Lower
 * is better. Returns the best vertex (ink x, y in the unit, y 0 at the row
 * top), the per-leg coverage and the stray share; score Infinity when no
 * vertex carries every leg. Exact bounds, no approximation: a vertex is
 * skipped once its coverage terms (or its stray ink, counted until it
 * exceeds the remaining margin) cannot beat `bound` or the best vertex so
 * far; with `accepted`, also once it cannot meet the acceptance bounds. */
export interface JunctionFit {readonly score:number;readonly x:number;readonly y:number;readonly coverage:readonly number[];readonly stray:number}
export function measureJunction(gauge:JunctionGauge,legs:readonly number[],bound=Infinity,accepted=false):JunctionFit{
 const {ink,cover,reach}=gauge;let best:JunctionFit={score:Infinity,x:0,y:0,coverage:legs.map(()=>0),stray:1};let limit=accepted?Math.min(bound,JUNCTION_MOST_SCORE+1e-9):bound;
 const rays=legs.map(d=>[Math.cos(d*Math.PI/8),Math.sin(d*Math.PI/8)] as const);
 for(let v=0;v<ink.length;v++){
  let total=0,worst=0,short=false;
  for(const d of legs){if(reach[v*16+d]!<leastLeg(d)){short=true;break;}const miss=1-cover[v*16+d]!;total+=miss;if(miss>worst)worst=miss;}
  if(short||(accepted&&worst>1-JUNCTION_LEAST_COVER))continue;
  const lines=total/legs.length+worst;if(lines>=limit)continue;
  // The most stray ink that can still beat the best so far (or be accepted).
  const most=Math.min(ink.length*(limit-lines)/JUNCTION_STRAY_WEIGHT,accepted?ink.length*JUNCTION_MOST_STRAY:Infinity);
  const [vx,vy]=ink[v]!;let off=0;
  for(const [qx,qy]of ink){
   const wx=qx-vx,wy=qy-vy;if(wx*wx+wy*wy<=2.25)continue;let on=false;
   for(const [i,[ux,uy]]of rays.entries()){const along=wx*ux+wy*uy;if(along>=-1&&along<=reach[v*16+legs[i]!]!+1&&Math.abs(wx*uy-wy*ux)<=1.5){on=true;break;}}
   if(!on&&++off>most)break;
  }
  if(off>most)continue;
  const stray=off/ink.length,score=lines+JUNCTION_STRAY_WEIGHT*stray;
  if(score<limit){limit=score;best={score,x:vx,y:vy,coverage:legs.map(d=>cover[v*16+d]!),stray};}
 }
 return best;
}
export function fitJunctionUnit(font:readonly NativeGlyph[],address:(text:string)=>number,text:string,legs:readonly number[]):JunctionFit{
 return measureJunction(junctionGauge(font,address,text),legs);
}
/** The leg sets the table holds: every two legs from two to six sixteenths
 * apart (45 to 135 degrees: a peak, a corner, a side join; within one
 * sixteenth of straight or of folding back, a glyph's stroke cannot tell the
 * two legs apart and the runs simply meet), and every three legs at least two
 * sixteenths (45 degrees) apart from one another (a T, a side join, a Y, an
 * arrow). Four-leg crossings are not compiled: the run builder joins a
 * straight crossing as two through runs. */
export function junctionLegSets():{two:number[][];three:number[][]}{
 const two:number[][]=[],three:number[][]=[];const apart=(a:number,b:number)=>Math.min((a-b+16)%16,(b-a+16)%16);
 for(let a=0;a<16;a++)for(let b=a+1;b<16;b++){if(apart(a,b)>=2&&apart(a,b)<=6)two.push([a,b]);for(let c=b+1;c<16;c++)if(apart(a,b)>=2&&apart(b,c)>=2&&apart(a,c)>=2)three.push([a,b,c]);}
 return {two,three};
}
/** The compiled junction table and its readable record (evidence table). */
export interface CompiledJunction {readonly legs:readonly number[];readonly natural:boolean;
 /** The accepted unit, '' when the runs simply meet; best is the best-measured unit either way. */
 readonly text:string;readonly best:string;readonly fit:JunctionFit|undefined}
export function compileJunctions(font:readonly NativeGlyph[],address:(text:string)=>number):CompiledJunction[]{
 const units=[...new Set<string>([...JUNCTION_UNITS,...HOOD_EDGE_UNITS,...PRACTITIONER_SLOPE_UNITS])].filter(text=>!readsAsLetter(text));
 const gauges=units.map(text=>junctionGauge(font,address,text));
 const {two,three}=junctionLegSets(),out:CompiledJunction[]=[];
 // Built forms (box, wall, post) never take doubled uprights (the operator's
 // || call-out); natural forms (terrain, rock, cliff) no lone | / _ ＿ ／ and no
 // doubled uprights (operator 2026-11-10 ledger). A slanted leg is carried
 // only by ink along it, so an upright cannot take one (D5).
 // One measurement per unit and leg set serves both vocabularies; the first
 // unit in candidate order wins a tie.
 const sets=[...two,...three],built:CompiledJunction[]=[],natural:CompiledJunction[]=[];
 for(const legs of sets){
  const best:({text:string;fit:JunctionFit}|undefined)[]=[undefined,undefined];
  for(const gauge of gauges){
   const allowed=[!UPRIGHT_STACKS.has(gauge.text),!NATURAL_LONE_STROKES.has(gauge.text)];
   let bound=-Infinity;for(const k of [0,1])if(allowed[k])bound=Math.max(bound,best[k]?.fit.score??Infinity);
   if(bound===-Infinity)continue;
   const fit=measureJunction(gauge,legs,bound,true);
   for(const k of [0,1])if(allowed[k]&&fit.score<(best[k]?.fit.score??Infinity))best[k]={text:gauge.text,fit};
  }
  for(const k of [0,1]){
   const b=best[k],good=!!b&&b.fit.score<=JUNCTION_MOST_SCORE&&Math.min(...b.fit.coverage)>=JUNCTION_LEAST_COVER&&b.fit.stray<=JUNCTION_MOST_STRAY;
   (k?natural:built).push({legs,natural:!!k,text:good?b!.text:'',best:b?.text??'',fit:b?.fit});
  }
 }
 out.push(...built,...natural);
 return out;
}

/** Residency half of old MAIN's packFeatureGrammar. The pinned instruction
 * index and original word-order prefix remain unchanged. Circle-library
 * whole units are appended without renumbering pinned words. Font addresses and native
 * scalar-space positions and practitioner straight-run records are packed
 * here; there is no live fit, score, candidate election or compilation.
 *
 * Unlike the old ten-bit words, font addresses have a full u32 each. Display
 * and export therefore do not depend on a private, truncated glyph bank. */
export function packFeatureGrammar(font:readonly NativeGlyph[],width:number,policy:NativeWhitespacePolicy='preformatted'):PackedRoleGrammar {
 if(grammar.schema!=='trueff.role_grammar.v1'||grammar.recordWords!==GRAMMAR_RECORD_WORDS||grammar.directions!==GRAMMAR_DIRECTIONS||grammar.phases!==GRAMMAR_PHASES)
  throw Error('Unsupported pinned AAHub role grammar');
 if(!Number.isSafeInteger(width)||width<1)throw Error('Invalid native grammar page width');
 const addresses=new Map(font.map((glyph,index)=>[glyph.text,index])),metrics:number[]=[],points:number[]=[];
 for(const glyph of font){metrics.push(glyph.text.codePointAt(0)!,glyph.advance,points.length/2,glyph.points.length);for(const p of glyph.points)points.push(p[0]!,p[1]!);}
 const address=(text:string):number=>{const id=addresses.get(text);if(id===undefined)throw Error(`Compiled role grammar requires native scalar ${text}`);return id;};
 const circle=roundSupplement.entries.find(entry=>entry.id==='rx-lib99-plain-round-7row')!;
 if(!circle||circle.native.rows.length!==7)throw Error('Unavailable retained circle construction: shapes aa[99]');
 // The largest literal paper interval divides each source row's paired
 // sides. Their smaller INTERNAL spaces remain in the whole native words.
 // This is source-program compilation, not a font fit or a live choice.
 const circleRows=circle.native.rows.map(encoded=>{
  const text=nativeRoleText(encoded).trim();let start=-1,end=-1,width=0;
  for(const gap of text.matchAll(/[\s　]+/gu)){const advance=[...gap[0]].reduce((n,ch)=>n+font[address(ch)]!.advance,0);if(advance>width){width=advance;start=gap.index!;end=start+gap[0].length;}}
  return start<0?[text,'']:[text.slice(0,start),text.slice(end)];
 });
 // Expanded role composition: retain aa[99]'s cap, shoulder, single centre
 // and return parts, but do not repeat its asymmetric / versus ‘， row down
 // an arbitrarily tall sphere. The operator-specified paired native curves
 // belong to the WHOLE height program, not independent tangent selections.
 // aa[99]'s sparse ticks intentionally lose its shoulders. The connected
 // expanded construction instead joins the retained flat caps to its native
 // curved sides with whole diagonal shoulder/return units, on BOTH rows.
 // A large round reads from its turns, never from a flat lid or a scalloped
 // run. The cap and base are the role bank's own graded rows (sh-lib57
 // cap-shoulders / return-base): ,.　-─-　、 rising through the apex over
 // ／ ＼, and ｀ ｰ-　-‐´ under ＼ ／ with the bottom edge left light. Width
 // inserts whole ─ units in the cap; the base keeps its paper. aa[99]'s flat
 // ＿＿/￣￣ rows read as a drum at 7+ rows and repeated ⌒ reads as scallops.
 // Source rows follow the hand-authored tonemapping target and shapes aa[57]
 // (hand-authored-targets-20261029/tonemapping/authoringnotes.txt): cap,
 // ／ ＼ shoulders, the / ｀、 entry, the l l flank, the ､ / return entry,
 // ＼ ／ returns, then the separated ｀ ｰ- -‐´ base. The shader picks each
 // interior row's unit from the projected chord's slope; no row is repeated.
 const expandedCircleRows=circleRows.map((row,index)=>index===0?[',.　-','-　、','─']:index===1?['／','＼']:index===2?['/','ヽ']:index===3?['l','l']:index===4?['ヽ','/']:index===5?['＼','／']:index===6?['｀ ｰ-','-‐´']:row);
 const recipes=ROLE_FEATURE_WORDS.map(word=>word.text),circleWords=expandedCircleRows.map(row=>row.map(text=>{if(!text)return 0;recipes.push(text);return recipes.length;}));
 // Practitioner clean-line units, kept separate from soft/rounded words.
 // A hard edge is one whole repeat unit, never a tone fragment or a pair
 // of slash columns harvested from an unrelated multirow part.
 const hardWords=new Map<string,number>();
 // The stepped units of a receding plane edge are whole native words, as the
 // fog target spells them: ＼､ per row on the near floor edge, ｀ヽ､､. per row on
 // the far shadowed room edge, and the rising _,.-‐´ of a floor edge coming
 // toward the viewer (hand-authored-targets-20261029/fog, frame 000).
 // The smallest tier of a box is one connected outline glyph (D9), never
 // dots or an upright: 口 ロ ﾛ by projected width, like the compact circles.
 for(const text of ['|','＿','￣','―','ー','/','／','＼','ノ','ヽ','∠','_','＼､','｀ヽ､､.','_,.-‐´','口','ロ','ﾛ']){recipes.push(text);hardWords.set(text,recipes.length);}
 // The half-width falling slash of a steep terrain row and of a paired flank
 // leaning right. Appended after every word a pinned index or a compiled
 // offset names (the far strip, junction and tone words follow by lookup).
 const classWords=new Map<string,number>();
 let wordsPacked=false;
 const classWord=(text:string):number=>{let word=hardWords.get(text)??classWords.get(text);if(word===undefined){if(wordsPacked)throw Error(`Class word ${text} was not registered before the word table was packed`);recipes.push(text);word=recipes.length;classWords.set(text,word);}return word;};
 // Region tone is spelled by the retained hand-authored passages as whole
 // words (D6: one constant pattern per region), never by a per-cell scalar.
 // They are contiguous so a passage index addresses its word by offset.
 const toneWords=recipes.length+1;
 for(const passage of TONE_PASSAGES.slice(1))recipes.push(passage.text);
 // 遠景 far strip vocabulary (operator PDF "EXAMPLE SUPER FAR LOD"): the peak
 // and upright-mark words of the one-row skyline. Appended after every pinned
 // and tone word, so no earlier word index moves. The native font has no ↑.
 const farWords=new Map<string,number>();
 // 人: the low standing mark of the super-far example. ヽ_ and _ノ: the lobe
 // base turns, and ,, the base flicker, of the skill's 15.5 idiom table.
 for(const text of ['∧','^','ヘ','、','y','个','⌒','人','ヽ_','_ノ',',,',':;:','l','ﾉ']){recipes.push(text);farWords.set(text,recipes.length);}
 // JUNCTION TABLE (compileJunctions): for every leg set, the junction unit
 // whose ink best carries every leg out to its edge with the least ink off
 // the legs, measured once here on the native font; a frame reads the record
 // by the legs' address and never ranks it. Words reuse a word that already
 // exists and append the rest after every earlier word. A unit with a
 // horizontal leg takes its row from the cell edge the lid units draw on (￣ at
 // the top of a cell, ＿ at the bottom); any other from its vertex.
 const hardJunctions=new Uint32Array(JUNCTION_RECORDS*4);
 const junctionWords=new Map<string,number>();
 for(const {legs,natural,text,fit}of compileJunctions(font,address)){
  if(!text||!fit)continue;
  let word=hardWords.get(text)??farWords.get(text)??junctionWords.get(text);
  if(word===undefined){recipes.push(text);word=recipes.length;junctionWords.set(text,word);}
  const advance=[...text].reduce((sum,ch)=>sum+font[address(ch)]!.advance,0);
  const lid=legs.some(d=>d%8===0),rowOffset=lid?(fit.y<8.5?-17:17):Math.round(2*fit.y)-17;
  const orders=legs.length===2?[[legs[0]!,legs[1]!],[legs[1]!,legs[0]!]]:[[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]].map(o=>o.map(i=>legs[i]!));
  for(const order of orders){
   const at=legs.length===2?JUNCTION_TWO+(natural?256:0)+order[0]!*16+order[1]!:JUNCTION_THREE+(natural?4096:0)+order[0]!*256+order[1]!*16+order[2]!;
   hardJunctions.set([word,advance,Math.floor(fit.x),rowOffset>>>0],at*4);
  }
 }
 // The graded run units of shallow lines, the level unit ‐, and ＼ (the
 // paired flank column's steep fall).
 for(const text of ['＼',...HARD_SLOPE_RISING,...HARD_SLOPE_FALLING,'‐'])classWord(text);
 for(const text of [...HARD_SLOPE_PASSAGES.rising,...HARD_SLOPE_PASSAGES.falling])classWord(text);
 // SLOPE FIT: the best-fitting unit of each of the 64 directions (one row of
 // travel 17 dx/dy), measured once on the font's ink over SLOPE_UNITS
 // (fitSlopeUnit). Two vocabularies: a natural form (terrain, rock, cliff, a
 // curved chain) without NATURAL_LONE_STROKES, a built form (box, wall,
 // post) without UPRIGHT_STACKS. Directions within a hair of level take none.
 // A unit whose ink's principal axis lies within 10 degrees of vertical is an
 // upright. A slanted direction never takes one (D5, AGENTS finding 3: the
 // fit rewards ink that covers every pixel row, and without this rule every
 // fold steeper than 70 degrees became a stack of ｌ / 亅, "verticals
 // dominating"); only the two exactly vertical bins (16, 48) may.
 const upright=(text:string)=>uprightUnit(font,address,text);
 // A small mark: a dot, a scalar with 4 ink pixels or fewer (. , ' ､ ﾞ).
 // At 8 the far tier also lost the short strokes (ヽ is 6) and spelled
 // falling folds with the peak ﾍ and the hook ゝ.
 const smallMark=(ch:string)=>font[address(ch)]!.points.length<=4;
 // LOD VOCABULARIES by size band (2 near, 1 middle, 0 far; tierBand):
 //  near   every candidate; a ずらし run may carry 抜き gaps (gap period);
 //  middle every candidate, always connected;
 //  far    no unit holding a small mark: the smallest tier is a connected
 //         outline, no dotted pieces and no 抜き (AAしょくにん 07, D9).
 // A slanted direction (70 degrees and steeper, not vertical) that no
 // non-upright unit carries (best score above SHIFT_RUN_SCORE) is a ずらし
 // run of the upright stroke, shifted with the line each row (AAしょくにん
 // 01: | one dot a row); D5 keeps uprights off curved and round sides, which
 // never read this table's steep rows. Operator 2026-11-12: "combine Best fit
 // as measured and Shifted upright run ... with implied continuation".
 const SHIFT_RUN_SCORE=2,SHIFT_RUN_GAP=3;
 const slopeFit=(exclude:ReadonlySet<string>,stroke:string,size:number)=>Array.from({length:64},(_,dir)=>{
  const angle=dir*Math.PI/32,dx=Math.cos(angle),dy=Math.sin(angle);
  if(Math.abs(dy)<.05)return undefined;
  const travel=17*dx/dy,vertical=dir===16||dir===48,steepness=Math.atan2(Math.abs(dy),Math.abs(dx))*180/Math.PI;
  let best={text:'',score:Infinity,centre:0,gap:0};
  for(const text of [...SLOPE_UNITS,...HOOD_EDGE_UNITS,...PRACTITIONER_SLOPE_UNITS]){
   // A unit of dots only (every scalar a small mark: ', '､ '，) draws a line
   // as a dotted chain (dots-only, D1/D9): the nearest rock's 62-68 degree
   // edge at spot-frames read as `',` dots. A line unit carries a stroke.
   if([...text].every(smallMark))continue;
   if(exclude.has(text)||readsAsLetter(text)||(!vertical&&upright(text))||(size===0&&[...text].some(smallMark)))continue;
   const fit=fitSlopeUnit(font,address,text,travel);if(fit.score<best.score)best={text,score:fit.score,centre:fit.centre,gap:0};}
  if(!vertical&&steepness>=70&&best.score>SHIFT_RUN_SCORE){const fit=fitSlopeUnit(font,address,stroke,travel);best={text:stroke,score:fit.score,centre:fit.centre,gap:size===2?SHIFT_RUN_GAP:0};}
  classWord(best.text);return best;
 });
 // The natural ずらし stroke is the vertical bar, not the thin l (line repair
 // 2026-11-12): in this font l inks 11 of the 17 px row, | 16, so a run of l a
 // row read as separated dashes down every steep cliff fold (view 06). AAしょく
 // にん 01 (skill aa-shokunin-tutorials.md 45-47): "Vertical bar: 1 dot per
 // row (thin l softens the steps but weakens the join)"; AAHub 地形（海・砂浜
 // ・島） 3c33aa91 aa[31] rows 2-5 (operator-kept label, swipe audit
 // verdicts.jsonl:1267): a leaning rock side is | shifted per row. The run is
 // still one unit plus the line's own shift (never a lone |).
 const naturalSlope=[0,1,2].map(size=>slopeFit(NATURAL_LONE_STROKES,'|',size)),builtSlope=[0,1,2].map(size=>slopeFit(UPRIGHT_STACKS,'|',size));
 // The conifer side units (FOLIAGE_EDGE_UNITS), registered LAST before the
 // table is packed, so no earlier word index moves.
 for(const text of FOLIAGE_EDGE_UNITS)classWord(text);
 // LINES3 STROKE SLOPE: per size band and direction, the best-fitting unit
 // that carries a stroke (strokeGlyph) among the slope candidates, never a
 // bracket, a lone natural stroke or an upright off vertical, and at the
 // far band no small mark (D9). Registered after every earlier word.
 const strokeBracket=/[{}｛｝()（）〈〉《》「」『』［］〔〕<>＜＞｣｢]/u;
 const strokeSlope=[0,1,2].map(size=>Array.from({length:64},(_,dir)=>{
  const angle=dir*Math.PI/32,dx=Math.cos(angle),dy=Math.sin(angle);if(Math.abs(dy)<.05)return undefined;
  const travel=17*dx/dy,vertical=dir===16||dir===48;let best={text:'',score:Infinity,centre:0};
  for(const text of [...SLOPE_UNITS,...HOOD_EDGE_UNITS,...PRACTITIONER_SLOPE_UNITS]){
   if(NATURAL_LONE_STROKES.has(text)||readsAsLetter(text)||strokeBracket.test(text)||![...text].some(ch=>strokeGlyph(font,address,ch)))continue;
   if((!vertical&&upright(text))||(size===0&&[...text].some(smallMark)))continue;
   // A vertical row takes an upright the playback knows as one (uprightWord:
   // | ｜ l ｌ !), so D5 still parts two consecutive rows on a curved run.
   if(vertical&&![...text].every(ch=>'|｜lｌ!'.includes(ch)))continue;
   const fit=fitSlopeUnit(font,address,text,travel);if(fit.score<best.score)best={text,score:fit.score,centre:fit.centre};}
  if(!best.text)return undefined;classWord(best.text);return best;
 }));
 // Small constructions are compiled at residency. Register every atom's
 // single-scalar canonical word before `words` is finalized; the retained
 // partAtoms ABI still stores the native font address, but this guarantees a
 // small atom uses the same canonical vocabulary as the word playback.
 const smallConstructionRecords=compileSmallConstructions(font);
 const smallWordFor=(text:string):number=>{
  const existing=recipes.findIndex(candidate=>candidate===text&&[...candidate].length===1);
  return existing>=0?existing+1:classWord(text);
 };
 for(const record of smallConstructionRecords)for(const [glyphIndex]of record.atoms){
  const text=font[glyphIndex]?.text;if(text===undefined)throw Error(`Small construction atom has no native glyph ${glyphIndex}`);
  smallWordFor(text);
 }
 wordsPacked=true;
 const words=new Uint32Array((recipes.length+1)*WORD_STRIDE);
 recipes.forEach((text,index)=>{
  const chars=[...text];if(chars.length>WORD_STRIDE-2)throw Error(`Invalid compiled whole word ${text}`);
  const at=(index+1)*WORD_STRIDE;words[at]=chars.length;
  for(const [i,ch]of chars.entries()){const id=address(ch);words[at+2+i]=id;words[at+1]!+=font[id]!.advance;}
 });
 const instructions=Uint32Array.from(grammar.instructions);
 for(let at=0;at<instructions.length;at+=GRAMMAR_RECORD_WORDS){
  const word=instructions[at]!;if(!word)continue;
  if(word>ROLE_FEATURE_WORDS.length||words[word*WORD_STRIDE]!==instructions[at+1]||words[word*WORD_STRIDE+1]!==instructions[at+2])throw Error('Pinned role instruction no longer matches its native word');
 }
 // This is a construction-class rule, NOT vocabulary pruning. Straight
 // vertical features and tone retain i/l/|. Compound retained words and
 // whole anchor parts retain all their cooperating strokes. Only a bare
 // repeated upright/parenthesis unit is unavailable to curved/slanted runs.
 // D1's same-role/size-band alternatives are resolved once from PINNED
 // records at the SAME direction; no live glyph/angle/ink fitting occurs.
 const articulatedInstructions=instructions.slice();
 const bare=(id:number)=>id!==0&&/^[|li!ｌ｜ｉ！()（）]+$/u.test(ROLE_FEATURE_WORDS[id-1]!.text);
 for(let role=0;role<4;role++)for(let size=0;size<3;size++)for(let dir=0;dir<64;dir++)for(let phase=0;phase<17;phase++){
  const at=(((role*3+size)*64+dir)*17+phase)*12;if(!bare(instructions[at]!))continue;
  let replacement:number|undefined;
  for(const band of [size,Math.max(0,size-1),Math.min(2,size+1)]){
   // The compiled phase order is a source-word transition, never a trial
   // against a displayed pixel or a currently occupied pen.
   for(let shift=0;shift<17;shift++){const other=(((role*3+band)*64+dir)*17+(phase+shift)%17)*12,id=instructions[other]!;
    if(id&&!bare(id)){replacement=other;break;}}
   if(replacement!==undefined)break;
  }
  if(replacement===undefined)articulatedInstructions.fill(0,at,at+12);
  else articulatedInstructions.set(instructions.subarray(replacement,replacement+12),at);
 }
 // Practitioner run rule: a steep part is ONE unit repeated once per native
 // row and moved by the source's shift. The pinned phase column grades a
 // direction by where the path enters the row, so most phases hold a
 // partial-height mark (､ 、 , ') that cannot chain row to row. Compile, once
 // per (role, size, direction), which record of the column is its whole-row
 // unit: the one whose ink travels furthest down the row, the first from the
 // entry edge on a tie. A bracket alone ({ } ( ) 〉 《) is a paired round
 // side, not a line unit; nor is a word of ring marks (ﾟ ｡ °): repeated down
 // the rows it is a chain of beads, which D9 forbids for an outline and
 // which spelled every steep right flank of a conifer as dots. Record word 11 of every phase carries that phase + 1
 // (0: the column has no line unit), so a frame plays the unit back by
 // address and never ranks the column.
 const brace=(id:number)=>/^[{}｛｝()（）〈〉《》]+$|[ﾟ｡°。゜]/u.test(recipes[id-1]!);
 const packedY=(value:number)=>((value>>8)&255)-128;
 const wholeRowUnits=(table:Uint32Array,only?:ReadonlySet<number>)=>{for(let context=0;context<12;context++)for(let dir=0;dir<64;dir++){
  const column=(context*64+dir)*17*12;if(only&&!only.has(column))continue;let unit=0,travel=-1;
  for(let step=0;step<17;step++){
   const phase=dir<32?step:16-step,at=column+phase*12,id=table[at]!;if(!id||brace(id))continue;
   const reach=Math.abs(packedY(table[at+7]!)-packedY(table[at+6]!));
   if(reach>travel){travel=reach;unit=phase+1;}
  }
  for(let phase=0;phase<17;phase++)table[column+phase*12+11]=unit;
 }};
 for(const table of [instructions,articulatedInstructions])wholeRowUnits(table);
 // The practitioner straight-line units are a construction context, not
 // a vocabulary ban. Curves, tone and retained parts keep their own tables.
 // Compile their fixed direction/phase records once from the practitioner's
 // clean-line units; no frame fits ink, tries pens or chooses another glyph.
 const hardLineInstructions=new Uint32Array(instructions.length);
 const packedPoint=(x:number,y:number)=>((Math.round(x)+128)|((Math.round(y)+128)<<8))>>>0;
 const hardPassages=Array.from({length:64},(_,dir)=>hardSlopePassage(font,dir));
 for(let role=0;role<4;role++)for(let size=0;size<3;size++)for(let dir=0;dir<64;dir++)for(let phase=0;phase<17;phase++){
  const at=(((role*3+size)*64+dir)*17+phase)*12;
  const angle=dir*Math.PI/32,dx=Math.cos(angle),dy=Math.sin(angle),vertical=Math.abs(dx)<HARD_UPRIGHT_LEAN,horizontal=dir===0||dir===32,shallow=Math.abs(dy)<Math.SQRT1_2;
  // The hand-authored targets draw every horizontal and shallow hard edge
  // with ＿ or ￣ on the cell edge nearest the source line, never a mid-row ―,
  // so uprights and lids meet at the corner (|＿＿＿|, ∠＿＿／|); a post or
  // jamb within HARD_UPRIGHT_LEAN of vertical is the upright run |. Between
  // level and 45° the edge is the graded run (_ , ‐ ´ ￣ by row phase). From 45°
  // to upright each direction takes its best-fitting built unit (builtSlope),
  // centred where its ink meets the line.
  // A shallow row takes a COMPLETE graded passage, independently of the
  // first source point's phase. Local height scalars stay in terrain's own
  // table; repeating one here cannot carry a neighbouring-row continuation.
  const low=phase>=8.5,slope=Math.abs(dy)/Math.max(Math.abs(dx),1e-9),passage=hardPassages[dir],graded=(dx*dy<0?HARD_SLOPE_RISING:HARD_SLOPE_FALLING)[HARD_SLOPE_PHASES.findIndex(bound=>phase<bound)]!;
  const fit=builtSlope[size]![dir],steep=!vertical&&!horizontal&&!shallow&&slope>=TERRAIN_STEP_SLOPE&&fit!==undefined;
  const text=steep?fit.text:vertical?'|':horizontal||slope<TERRAIN_STEP_SLOPE?(low?'＿':'￣'):graded;
  const chars=[...text],advance=chars.reduce((sum,ch)=>sum+font[address(ch)]!.advance,0),word=classWord(text);
  // The source point addresses the UNIT'S ink centre, regardless of which
  // end of the source edge the walk traverses first. Opposite traversals
  // must not offset the same corner to opposite sides by one full advance.
  const glyph=font[address(chars[0]!)]!;
  const offset=steep?-Math.round(fit.centre):-Math.round(glyph.points.reduce((sum,q)=>sum+q[0]!,0)/glyph.points.length);
  // The ＿/￣ choice already puts the ink on this row's nearest cell edge.
  const shift=0;
  // The local-height instruction remains available to the EXISTING planar
  // and soft fallback readers. It is not a repeatable row construction.
  // The same record declares the whole-row program explicitly: word 4 is
  // its whole word, 5 its signed ink-centre offset, 9 its playback kind and
  // 10/11 its real ink ports. featurecurves consumes that program ONCE per
  // crossed row. No other reader mistakes the new word for a height sample.
  hardLineInstructions.set([word,chars.length,advance,shift>>>0,passage?classWord(passage.text):0,passage?(-Math.round(passage.centre))>>>0:0,
   packedPoint(0,0),packedPoint(advance-1,16),offset>>>0,passage?1:0,
   passage?packedPoint(passage.entry[0],passage.entry[1]):0,passage?packedPoint(passage.exit[0],passage.exit[1]):0],at);
 }
 // TERRAIN LINE TABLE (source form 3: ridge and skyline silhouettes, shelf
 // edges, contact lines of a heightfield). One record per (role, size,
 // direction, row phase), compiled here once from the clean-line units; a
 // frame reads it by address in the soft-run row playback and never ranks
 // it. The row portion's own direction picks the column, so a slope changes
 // unit only where its direction changes class. Conditions, all stated:
 //  flat    (slope < TERRAIN_STEP_SLOPE): the horizontal unit whose ink line
 //          is nearest the path's height in the row. A ridge or skyline
 //          (shoulder, flank) reads ￣ ― ＿; a shelf edge or a foot line
 //          (rim_cap, return_base) only the two cell-edge lines ￣ ＿, so the
 //          band under a shelf keeps its row.
 //  graded  (to TERRAIN_SLANT_SLOPE): the run unit by the line's height in
 //          the row, _ , ‐ ´ ￣ rising, ￣ ｀ ‐ ､ _ falling.
 //  steep   (TERRAIN_SLANT_SLOPE and beyond, uprights included): one unit per
 //          row, the best-fitting natural unit of the direction (naturalSlope).
 // Word 11 is the phase + 1 of the column's one-per-row unit, as in the
 // pinned tables; words 6 and 7 carry that unit's ink centre.
 const terrainInstructions=new Uint32Array(instructions.length);
 const inkOf=(text:string)=>font[address(text)]!.points;
 const inkCentre=(text:string)=>Math.round(inkOf(text).reduce((sum,q)=>sum+q[0]!,0)/inkOf(text).length);
 const inkLine=(text:string)=>inkOf(text).reduce((sum,q)=>sum+q[1]!,0)/inkOf(text).length;
 const nearestLine=(units:readonly string[],phase:number)=>units.reduce((best,text)=>Math.abs(inkLine(text)-phase)<Math.abs(inkLine(best)-phase)?text:best);
 for(let role=0;role<4;role++)for(let size=0;size<3;size++)for(let dir=0;dir<64;dir++)for(let phase=0;phase<17;phase++){
  const at=(((role*3+size)*64+dir)*17+phase)*12,angle=dir*Math.PI/32,dx=Math.cos(angle),dy=Math.sin(angle);
  const slope=Math.abs(dy)/Math.max(Math.abs(dx),1e-9),rising=dx*dy<0;
  // NO LONE | / _ ＿ ／ ON TERRAIN (operator order 2026-11-10, GOAL-EXECUTOR
  // ledger). A slope up to TERRAIN_SLANT_SLOPE is a RUN UNIT spelled glyph by
  // glyph along the row at the line's own height in it, so a rising line
  // reads _ , ‐ ´ ￣ and a falling one ￣ ｀ ‐ ､ _ (AAHub _,.-‐''´ and
  // ｀ヽ､_; the hand target's _,ﾍ､_,ﾍ､). Level ground under TERRAIN_FLAT_SLOPE
  // is the level unit at the line's height. A row the line crosses within one
  // and a half units (TERRAIN_SLANT_SLOPE and steeper, uprights included)
  // takes the best-fitting natural unit of its direction (naturalSlope),
  // centred where its ink sits on the line; it replaced fixed rungs
  // (l / ﾉ ヽ / ノ ヽ､) that spelled every angle of a class with one glyph.
  let text:string,unit=1,fitted:number|undefined,gap=0;
  const fit=naturalSlope[size]![dir];
  if(slope>=TERRAIN_SLANT_SLOPE&&fit){text=fit.text;fitted=Math.round(fit.centre);gap=fit.gap;}
  else if(slope>=TERRAIN_FLAT_SLOPE){unit=9;text=(rising?HARD_SLOPE_RISING:HARD_SLOPE_FALLING)[HARD_SLOPE_PHASES.findIndex(bound=>phase<bound)]!;}
  else{unit=9;text=nearestLine(['￣','‐','＿'],phase);}
  // A unit of several glyphs (ヽ､): its advance is their sum; a fitted unit's
  // centre is where its ink meets the line, otherwise the first glyph's ink.
  const chars=[...text],glyph={advance:chars.reduce((sum,ch)=>sum+font[address(ch)]!.advance,0)},centre=fitted??inkCentre(chars[0]!);
  // Word 9: the 抜き gap period of a near-tier ずらし run (0 = connected).
  terrainInstructions.set([classWord(text),chars.length,glyph.advance,0,0,0,packedPoint(centre,0),packedPoint(centre,16),0,gap,0,unit],at);
 }
 // LINES3 NO DOTTED LINE UNITS. A record of the soft-run tables whose word has
 // no stroke glyph (the steep falling and vertical columns hold only ', ‘, :
 // ′, the shallow columns ´ '´ ,. ¨ " at most phases) takes the line record of
 // its direction and phase by the terrain table's rules: the level unit, the
 // graded run, beyond TERRAIN_SLANT_SLOPE the direction's stroke unit
 // (strokeSlope) centred on the line. Then the whole-row units are recompiled.
 const dottedWord=(id:number)=>{const n=words[id*WORD_STRIDE]!;for(let i=0;i<n;i++){const ch=font[words[id*WORD_STRIDE+2+i]!]!.text;if(strokeGlyph(font,address,ch))return false;}return true;};
 for(const table of [instructions,articulatedInstructions]){const recompile=new Set<number>();for(let role=0;role<4;role++)for(let size=0;size<3;size++)for(let dir=0;dir<64;dir++)for(let phase=0;phase<17;phase++){
  const at=(((role*3+size)*64+dir)*17+phase)*12,id=table[at]!;if(!id||!dottedWord(id))continue;
  // Only a column whose whole-row unit was a dotted word (or that had none)
  // takes a new one; a column with a stroke unit keeps it.
  const column=at-phase*12,unit=table[column+11]!;if(unit===0||unit-1===phase)recompile.add(column);
  const angle=dir*Math.PI/32,dx=Math.cos(angle),dy=Math.sin(angle),slope=Math.abs(dy)/Math.max(Math.abs(dx),1e-9),rising=dx*dy<0;
  /*LINES3-BISECT steep*/if(slope<TERRAIN_SLANT_SLOPE)continue;const fit=strokeSlope[size]![dir];let text:string,fitted:number|undefined;
  if(slope>=TERRAIN_SLANT_SLOPE&&fit){text=fit.text;fitted=Math.round(fit.centre);}
  else if(slope>=TERRAIN_FLAT_SLOPE){text=(rising?HARD_SLOPE_RISING:HARD_SLOPE_FALLING)[HARD_SLOPE_PHASES.findIndex(bound=>phase<bound)]!;}
  else text=nearestLine(['￣','‐','＿'],phase);
  const chars=[...text],advance=chars.reduce((sum,ch)=>sum+font[address(ch)]!.advance,0),centre=fitted??inkCentre(chars[0]!);
  const word=hardWords.get(text)??classWords.get(text);if(word===undefined)throw Error(`Line unit ${text} was not registered before the word table was packed`);
  table.set([word,chars.length,advance,0,0,0,packedPoint(centre,0),packedPoint(centre,16),0,0,0,table[at+11]!],at);
 }wholeRowUnits(table,recompile);}
 // PAIRED FLANK COLUMN (a path-led limb whose two silhouettes are one pair:
 // trunk, branch, post). One record by the row's lean; the SAME word spells
 // the left and the right flank of that row, so the two sides cooperate and
 // corresponding parts share glyph identity (practitioner pair rule). The
 // half-width backslash is never a unit: the native font draws the yen sign
 // at that code point (loop 351 view 06).
 const pairedFlanks=new Uint32Array((PAIRED_FLANK_LEANS.length+1)*4);
 for(const [index,text]of ['／','/','|','＼','＼'].entries())pairedFlanks.set([classWord(text),font[address(text)]!.advance,inkCentre(text),0],index*4);
 // CONIFER SIDE UNITS: each word's ink centre across its glyphs.
 const wordInkCentre=(text:string)=>{let pen=0,sum=0,n=0;for(const ch of text){const glyph=font[address(ch)]!;for(const q of glyph.points){sum+=pen+q[0]!;n++;}pen+=glyph.advance;}return Math.round(n?sum/n:pen/2);};
 const foliageEdges=new Uint32Array(FOLIAGE_EDGE_UNITS.length*4);
 for(const [index,text]of FOLIAGE_EDGE_UNITS.entries())foliageEdges.set([classWord(text),[...text].reduce((sum,ch)=>sum+font[address(ch)]!.advance,0),wordInkCentre(text),0],index*4);
 if(terrainInstructions.length!==TERRAIN_INSTRUCTION_WORDS||pairedFlanks.length!==(PAIRED_FLANK_LEANS.length+1)*4)throw Error('Grammar table sizes changed: FOLIAGE_EDGES_AT is stale');
 const parts:CompiledRolePart[]=[],partAtoms:number[]=[],partPaper:number[]=[],partEntries:typeof ROLE_ENTRIES[number][]=[];
 for(const entry of [...ROLE_ENTRIES,...roundSupplement.entries as unknown as typeof ROLE_ENTRIES]){
  if(!GRAMMAR_ROLES.some(role=>entry.role===role||entry.also_roles?.includes(role))&&entry.role!=='contact_junction'&&entry.role!=='branch_entry'&&entry.role!=='whole_circle')continue;
  const origin=Math.min(...entry.native.row_pen_x),first=partAtoms.length/4,paperFirst=partPaper.length/4;let left=Infinity,right=-Infinity;
  for(const [row,encoded]of entry.native.rows.entries()){
   let x=entry.native.row_pen_x[row]!-origin,previousEnd:number|undefined;
   for(const ch of nativeRoleText(encoded)){
    const id=address(ch),glyph=font[id]!;
    if(glyph.points.length){
     if(previousEnd!==undefined&&x>previousEnd)partPaper.push(previousEnd,row,x-previousEnd,0);
     partAtoms.push(id,x>>>0,row,glyph.advance);previousEnd=x+glyph.advance;
     left=Math.min(left,x);right=Math.max(right,previousEnd);
    }
    x+=glyph.advance;
   }
  }
  // Source crops can carry a common leading margin and trailing paper.
  // They are not part of the construction's extent: including them moves
  // the entire native part away from its projected anchor and can reject
  // a fitting part as wider than its owner. Translate ALL rows and their
  // internal paper together; never trim/centre rows independently.
  if(!Number.isFinite(left))throw Error(`Retained native part has no ink: ${entry.id}`);
  for(let at=first*4;at<partAtoms.length;at+=4)partAtoms[at+1]=(partAtoms[at+1]!-left)>>>0;
  for(let at=paperFirst*4;at<partPaper.length;at+=4)partPaper[at]=partPaper[at]!-left;
  parts.push({id:entry.id,role:entry.role,band:entry.size_band.band,rows:entry.native.rows.length,first,count:partAtoms.length/4-first,width:right-left,paperFirst,paperCount:partPaper.length/4-paperFirst});partEntries.push(entry);
 }
 const packedSmallRecords=smallConstructionRecords.map(record=>{
  const firstAtom=partAtoms.length/4,firstPaper=partPaper.length/4;
  for(const [glyphIndex,x,row]of record.atoms){const glyph=font[glyphIndex];if(!glyph)throw Error(`Small construction atom has no native glyph ${glyphIndex}`);partAtoms.push(glyphIndex,x>>>0,row,glyph.advance);}
  for(const [row,start,end]of record.paper)partPaper.push(start>>>0,row>>>0,(end-start)>>>0,0);
  return {kind:record.kind,positiveEnd:record.positiveEnd,negativeEnd:record.negativeEnd,kindAndEnds:packSmallConstructionKindAndEnds(record.kind,record.positiveEnd,record.negativeEnd),
   rows:record.rows,width:record.width>>>0,units:record.units>>>0,firstAtom,atomCount:record.atoms.length,firstPaper,paperCount:record.paper.length};
 });
 // Twelve words of compact circles, then seven 8-word rows: left word, right
 // word, left ink bound, right ink bound, fill word between the pair. Then
 // one 12-word record for a planar form's edge units: | ￣ ＿ ∠ ／ ＼ / _
 // (uprights, lids on the cell edge, the ∠ lid/depth-edge corner, rising
 // and falling depth edges, the narrow slash beside an upright, the
 // half-width lid remainder), then the stepped units of a receding edge
 // ＼､ ｀ヽ､､. _,.-‐´ as the hand-authored targets spell them, then the
 // smallest-tier whole box glyphs 口 ロ ﾛ (16 words in all).
 const circleProgram=new Uint32Array(12+7*8+16);
 const anchorRoles=['rim_cap','return_base','branch_entry','contact_junction'],bands=['s','m','l'],directions=['E','SE','S','SW','W','NW','N','NE'];
 const anchorContexts=new Uint32Array(anchorRoles.length*3*8*8*8);
 // A contact_junction anchor is the foot of a path-led form standing on a
 // surface (gpu-construction-runs.ts contactAnchor). Its retained entry is
 // admitted by its own declared context, not by the path-port rule below: the
 // path rises out of the contact (tangent_in N), the part's upper port is the
 // standing form (side or its tone) and its lower port is the ground (open or
 // a grass patch). The hand-web, cluster and machine junctions of this role
 // declare other ports and stay unconsumed here. The contact row is the row
 // of the splayed join glyph (从 人), else the last row; it lands on the
 // native row of the projected contact, rows above it on the standing form.
 const groundContact=(entry:typeof partEntries[number])=>{
  const ports=(entry as typeof entry&{ports?:Readonly<Record<string,string|null>>}).ports;
  return !!ports&&(ports.up==='side'||ports.up==='tone_mid')&&(ports.down===null||ports.down==='grass_patch');
 };
 const contactRow=(entry:typeof partEntries[number])=>{
  const rows=entry.native.rows.map(nativeRoleText),join=rows.findIndex(row=>/[从人]/u.test(row));
  return join<0?rows.length-1:join;
 };
 // This producer supplies PATH junctions, not neighboring foliage, tone,
 // openings or shadow regions. Keep all retained parts in the bank, but do
 // not admit a mixed path/region passage on tangent labels alone. E.g. T#7's
 // branch entry requires its declared tone_fringe/leaf boundary: pasting the
 // entire passage at a bare fork draws that missing region into the sky.
 // These are authored context requirements, not codepoint/frequency pruning.
 const pathPorts=new Set(['paper','side','flank','shoulder','rim_cap','return_base','fold_crease','arch_band','contact_junction']);
 const pathContext=(entry:typeof partEntries[number])=>{
  const ports=(entry as typeof entry&{ports?:Readonly<Record<string,string|null>>}).ports;
  return !!ports&&Object.values(ports).every(port=>port!==null&&pathPorts.has(port));
 };
 // SOURCE-PART SOFT PROGRAMS (perf-impl-111/112). The source graph owns the
 // feature, role and context token. This residency table resolves one exact
 // retained entry for role, requested size, both source tangent ports and the
 // source context class. The table is indexed before playback; no glyph or
 // row fragment is searched in a displayed frame. A requested band inherits
 // the nearest retained variant in the fixed D1 order, and its selected band
 // is carried in the record for deterministic LOD inheritance.
 const softPartTable=new Uint32Array(SOFT_PART_TABLE_WORDS);
 softPartTable[0]=SOFT_PART_TABLE_VERSION;softPartTable[1]=SOFT_PART_TABLE_RECORD_WORDS;
 softPartTable[2]=SOFT_PART_TABLE_RECORDS;softPartTable[3]=0;
 const directionCode=(name:string)=>{const index=directions.indexOf(name);return index<0?0:index;};
 const contextInside=(role:number,contextClass:number)=>{
  if(role===0&&contextClass===1)return 'below';
  if(role===1&&contextClass===2)return 'below';
  // perf-impl-154: a side turn is a genuine shoulder (retained
  // sk-church101-rib-shoulder-flank: shoulder, convex, inside right, NE->S,
  // path ports). Same class rule as compiled-construction-topology.ts.
  if(role===1&&contextClass===6)return 'left';
  if(role===1&&contextClass===7)return 'right';
  if(role===2&&contextClass===3)return 'left';
  if(role===2&&contextClass===4)return 'right';
  if(role===3&&contextClass===5)return 'above';
  return undefined;
 };
 const contextCompatible=(entry:typeof partEntries[number],role:number,contextClass:number,incoming:number,outgoing:number)=>{
  const inside=contextInside(role,contextClass);if(!inside)return false;
  return entry.role===GRAMMAR_ROLES[role]&&pathContext(entry)&&entry.context.curvature==='convex'&&entry.context.inside===inside&&
   entry.context.tangent_in===directions[incoming]&&entry.context.tangent_out===directions[outgoing]&&entry.context.paper.length>0;
 };
 for(let role=0;role<GRAMMAR_ROLES.length;role++)for(let requested=0;requested<3;requested++)for(let contextClass=1;contextClass<=7;contextClass++)for(let incoming=0;incoming<8;incoming++)for(let outgoing=0;outgoing<8;outgoing++){
  let entry:typeof partEntries[number]|undefined;let selectedBand=requested;
  for(const band of [requested,Math.max(0,requested-1),Math.min(2,requested+1)]){
   const candidate=partEntries.find(item=>item.size_band.band===bands[band]&&contextCompatible(item,role,contextClass,incoming,outgoing));
   if(candidate){entry=candidate;selectedBand=band;break;}
  }
  if(!entry)continue;
  const part=parts.find(item=>item.id===entry!.id);if(!part||part.width<0||part.width>=65536)continue;
  const context=(incoming&SOFT_PART_CONTEXT_IN_MASK)|((outgoing&SOFT_PART_CONTEXT_IN_MASK)<<SOFT_PART_CONTEXT_OUT_SHIFT)|(contextClass<<SOFT_PART_CONTEXT_CLASS_SHIFT);
  const at=4+(role*3*SOFT_PART_CONTEXTS+requested*SOFT_PART_CONTEXTS+context)*SOFT_PART_TABLE_RECORD_WORDS;
  // [atoms, paper, width/band, row count, context class, retained entry]
  softPartTable.set([part.first,part.count,part.paperFirst,part.paperCount,Math.round(part.width)>>>0|(selectedBand<<16)|(incoming<<18)|(outgoing<<21),part.rows,contextClass,partEntries.indexOf(entry)],at);
 }
 for(const [role,name]of anchorRoles.entries())for(let size=0;size<3;size++)for(let incoming=0;incoming<8;incoming++)for(let outgoing=0;outgoing<8;outgoing++){
  // Same D1 band order as the incumbent ConstructionRuns compiler. A
  // missing exact-size entry must not disconnect a retained anchor whose
  // declared role and ports exist in the adjacent size band. Resolve once,
  // never enumerate glyphs or alternative pens in a displayed frame.
  let entry:typeof partEntries[number]|undefined;
  for(const band of [bands[size],bands[Math.max(0,size-1)],bands[Math.min(2,size+1)]]){
   entry=partEntries.find(entry=>entry.role===name&&entry.size_band.band===band&&entry.context.tangent_in===directions[incoming]&&entry.context.tangent_out===directions[outgoing]&&(name==='contact_junction'?groundContact(entry):pathContext(entry)));
   if(entry)break;
  }
  if(!entry)continue;const part=parts.find(part=>part.id===entry.id)!;
  anchorContexts.set([part.first,part.count,Math.round(-part.width/2)>>>0,(name==='return_base'?1-part.rows:name==='contact_junction'?-contactRow(entry):0)>>>0,part.paperFirst,part.paperCount,part.rows,part.width],((role*3+size)*64+incoming*8+outgoing)*8);
 }
 for(const [i,id]of ['sh-lib66-circle-2row','sh-lib65-circle-3row-brace','sh-lib64-circle-4row'].entries()){
  const part=parts.find(part=>part.id===id)!;if(!part)throw Error(`Unavailable compact circle ${id}`);
  circleProgram.set([part.first,part.count,part.width,part.rows],i*4);
 }
 for(const [row,pair]of circleWords.entries()){
  const bounds=(word:number)=>{let x=0,lo=Infinity,hi=-Infinity;for(let i=0;i<words[word*WORD_STRIDE]!;i++){const glyph=font[words[word*WORD_STRIDE+2+i]!]!;for(const point of glyph.points){lo=Math.min(lo,x+point[0]!);hi=Math.max(hi,x+point[0]!+1);}x+=glyph.advance;}return [lo,hi];};
  circleProgram.set([pair[0]!,pair[1]!,bounds(pair[0]!)[0]!>>>0,pair[1]?bounds(pair[1]!)[1]!>>>0:0,pair[2]??0,pair[1]?bounds(pair[1]!)[0]!>>>0:0,bounds(pair[0]!)[1]!>>>0,0],12+row*8);
 }
 circleProgram.set(['|','￣','＿','∠','／','＼','/','_','＼､','｀ヽ､､.','_,.-‐´','口','ロ','ﾛ'].map(text=>hardWords.get(text)!),12+7*8);
 // 遠景 far strip (D3 exterior far field, D9). Fixed tables, read by address
 // in the frame; nothing here is tried against a drawing.
 //  [0,67)  unused (the per-run baseline and peak records of the producers
 //          the merged skyline replaced; kept zero so later offsets stand).
 //  [67,99) upright mark: per quarter-row height of the form, the mark word
 //          and the x of its foot: 、 under a quarter row, y to three
 //          quarters, 人 to a row, 个 above (the single marks standing on the
 //          base line of the operator's "EXAMPLE SUPER FAR LOD":
 //          apps/stroke-matcher/references/review-untitled-24.png); from a
 //          row and a half up, 个 over a | stem, one stem unit per row the
 //          form stands up to the far band's three rows (farSkyline; a trunk
 //          is a straight upright, D5).
 const farInk=(text:string)=>font[address(text)]!.points;
 const farEdge=(text:string,top:boolean)=>{
  const ink=farInk(text),y=ink.reduce((v,q)=>top?Math.min(v,q[1]!):Math.max(v,q[1]!),top?Infinity:-Infinity),line=ink.filter(q=>q[1]===y);
  return Math.round(line.reduce((sum,q)=>sum+q[0]!,0)/line.length);
 };
 // SOFT NEAREST-PHASE INDEX (crawl brief item F, T11). For every soft
 // direction column and phase, the phase index the T9 runtime rule selects:
 // the phase itself when its record holds a non-empty own-row word, else the
 // nearest such phase in fixed order phase-1, phase+1, ... (lower first), or
 // 0xff when none qualifies. One byte per (table, column, phase) over the
 // three tables softPortion reads (instructions, articulated, terrain); the
 // frame reads it by address instead of scanning 17 slots. Empty slots stay
 // empty: this table only records the selection, it never fills a record.
 // Little-endian byte order, matching the WGSL (word>>((byte&3)*8))&0xff read.
 const softPhaseTables=[instructions,articulatedInstructions,terrainInstructions];
 const softPhaseTable=new Uint32Array(SOFT_PHASE_TABLE_WORDS);
 softPhaseTable[0]=SOFT_PHASE_TABLE_VERSION;softPhaseTable[1]=SOFT_PHASE_TABLE_WORDS_PER_TABLE;
 softPhaseTable[2]=SOFT_PHASE_TABLE_TABLES;softPhaseTable[3]=SOFT_PHASE_TABLE_PHASES;
 const softPhaseBytes=new Uint8Array(softPhaseTable.buffer,SOFT_PHASE_TABLE_HEADER_WORDS*4);
 for(let sel=0;sel<SOFT_PHASE_TABLE_TABLES;sel++){const table=softPhaseTables[sel]!;
  for(let col=0;col<SOFT_PHASE_TABLE_COLUMNS;col++)for(let phase=0;phase<SOFT_PHASE_TABLE_PHASES;phase++){
   const at=(col*SOFT_PHASE_TABLE_PHASES+phase)*GRAMMAR_RECORD_WORDS;
   let want=SOFT_PHASE_TABLE_NONE;
   if(table[at]!==0&&table[at+3]===0)want=phase;
   else for(let d=1;d<SOFT_PHASE_TABLE_PHASES;d++){
    const lo=phase-d,hi=phase+d;
    if(lo>=0){const r=(col*SOFT_PHASE_TABLE_PHASES+lo)*GRAMMAR_RECORD_WORDS;if(table[r]!==0&&table[r+3]===0){want=lo;break;}}
    if(hi<SOFT_PHASE_TABLE_PHASES){const r=(col*SOFT_PHASE_TABLE_PHASES+hi)*GRAMMAR_RECORD_WORDS;if(table[r]!==0&&table[r+3]===0){want=hi;break;}}
   }
   softPhaseBytes[(sel*SOFT_PHASE_TABLE_COLUMNS+col)*SOFT_PHASE_TABLE_PHASES+phase]=want;
  }}
 // SOFT D5 PROGRESSION (crawl brief item F). The playback's old d5Swap path
 // classified the selected word, computed slashNeighbour, then tried the
 // neighbouring soft column and finally the clean hard-line record. Resolve
 // that exact progression once for every table/role/band/direction/phase.
 // The two variants cover the only direction whose slash neighbour also
 // depends on the sign of the projected chord (the exact vertical bins).
 const softD5Tables=[instructions,articulatedInstructions,terrainInstructions];
 const softD5Table=new Uint32Array(SOFT_D5_TABLE_WORDS);
 softD5Table.set([SOFT_D5_TABLE_VERSION,SOFT_D5_TABLE_ENTRY_WORDS,SOFT_D5_TABLE_TABLES,SOFT_D5_TABLE_COLUMNS,SOFT_D5_TABLE_PHASES,SOFT_D5_TABLE_VARIANTS]);
 const softD5Column=(table:Uint32Array,role:number,size:number,dir:number)=>{
  const column=((role*3+size)*64+dir)*17*12;
  return table[column+11]!==0?column:((role*3+size)*64+(dir<32?48:16))*17*12;
 };
 // This is intentionally the playback predicate, not the compile-time slope
 // gauge: d5Swap's incumbent `uprightWord` classifies a word by its inked
 // codepoints, including bracket strokes, and the current-bank table must be
 // byte-for-byte equivalent to that predicate.
 const softD5Upright=(word:number)=>{
  if(word===0)return false;let inked=0;for(const ch of recipes[word-1]??''){
   const glyph=font[address(ch)]!;if(glyph.points.length===0)continue;inked++;
   if(!'|｜lｌ!()（）'.includes(ch))return false;
  }return inked>0;
 };
 const softD5Slash=(dir:number,negative:boolean)=>{
  const lean=dir%32,downward=dir<32;
  if(lean===16){
   const positive=downward?(negative?13:19):(negative?19:13);
   return dir-lean+positive;
  }
  return dir-lean+(lean<16?Math.max(lean,19):Math.min(lean,13));
 };
 const softD5Record=(table:Uint32Array,hard:Uint32Array,role:number,size:number,dir:number,phase:number,sel:number)=>{
  const column=softD5Column(table,role,size,dir),at=column+phase*12;
  let record=at;
  if(table[record]===0){
   let near=-1;
   for(let d=1;d<17;d++){
    const lo=phase-d,hi=phase+d;
    if(lo>=0){const r=column+lo*12;if(table[r]!==0&&table[r+3]===0){near=r;break;}}
    if(hi<17){const r=column+hi*12;if(table[r]!==0&&table[r+3]===0){near=r;break;}}
   }
   if(near>=0)record=near;
   else {
    const whole=column+(Math.max(table[column+11]??0,1)-1)*12;
    if(table[whole]!==0&&table[whole+3]===0)record=whole;
    else {
     let clean=dir,lean=dir%32;
     if(sel!==0&&lean>=14&&lean<=18)clean=dir-lean+(lean<=16?19:13);
     record=((role*3+size)*64+clean)*17*12+phase*12;
     return {record,hard:true};
    }
   }
  }
  return {record,hard:false};
 };
 const softD5Candidate=(table:Uint32Array,hard:Uint32Array,role:number,size:number,dir:number,phase:number,sel:number,negative:boolean,whole:boolean)=>{
  const lean=softD5Slash(dir,negative),column=softD5Column(table,role,size,dir);
  const resolved=softD5Record(table,hard,role,size,dir,phase,sel);
  const current=whole?column+(Math.max(table[column+11]??0,1)-1)*12:resolved.record;
  const upright=softD5Upright((whole?table[current]:resolved.hard?hard[current]:table[current])??0);
  const other=softD5Column(table,role,size,lean);
  const swapped=whole?other+(Math.max(table[other+11]??0,1)-1)*12:other+phase*12;
  if(table[swapped]!==0&&(whole||table[swapped+3]===0)&&!softD5Upright(table[swapped]!))return {record:swapped,hard:false,upright};
  const builtPhase=whole?8:phase;
  const built=((role*3+size)*64+lean)*17*12+builtPhase*12;
  if(hard[built]!==0&&!softD5Upright(hard[built]!))return {record:built,hard:true,upright};
  return {record:SOFT_D5_TABLE_NONE,hard:false,upright};
 };
 const encodeD5=(result:{record:number;hard:boolean;upright:boolean})=>{
  const flags=result.upright?SOFT_D5_TABLE_UPRIGHT:0;
  if(result.record===SOFT_D5_TABLE_NONE)return (flags|SOFT_D5_TABLE_NONE)>>>0;
  return (flags|(result.hard?SOFT_D5_TABLE_HARD:0)|result.record)>>>0;
 };
 const d5At=(variant:number,sel:number,column:number,phase:number,slot:number)=>SOFT_D5_TABLE_HEADER_WORDS+
  ((((variant*SOFT_D5_TABLE_TABLES+sel)*SOFT_D5_TABLE_COLUMNS+column)*SOFT_D5_TABLE_PHASES+phase)*SOFT_D5_TABLE_ENTRY_WORDS+slot);
 for(let variant=0;variant<SOFT_D5_TABLE_VARIANTS;variant++)for(let sel=0;sel<SOFT_D5_TABLE_TABLES;sel++){
  const table=softD5Tables[sel]!;
  for(let role=0;role<4;role++)for(let size=0;size<3;size++)for(let dir=0;dir<64;dir++)for(let phase=0;phase<17;phase++){
   const column=(role*3+size)*64+dir,negative=variant===1&&(dir===16||dir===48);
   const shallow=softD5Candidate(table,hardLineInstructions,role,size,dir,phase,sel,negative,false);
   const whole=softD5Candidate(table,hardLineInstructions,role,size,dir,phase,sel,negative,true);
   softD5Table[d5At(variant,sel,column,phase,0)]=encodeD5(shallow);
   softD5Table[d5At(variant,sel,column,phase,1)]=encodeD5(whole);
  }
 }
 // The small table's index is compiled by descending native width. A lookup
 // cell therefore already contains the widest record that fits its requested
 // integer width; playback never scans the family or re-ranks a record.
 const smallIndex=new Uint32Array(SMALL_CONSTRUCTION_TABLE_INDEX_WORDS);smallIndex.fill(SMALL_CONSTRUCTION_TABLE_NONE);
 for(const kind of ['round','limb','box'] as const)for(const positiveEnd of ['cap','contact','open'] as const)for(const negativeEnd of ['cap','contact','open'] as const)for(let rows=1;rows<=SMALL_CONSTRUCTION_TABLE_ROW_COUNT;rows++){
  const candidates=packedSmallRecords.map((record,index)=>({record,index})).filter(value=>value.record.kind===kind&&value.record.positiveEnd===positiveEnd&&value.record.negativeEnd===negativeEnd&&value.record.rows===rows)
   .sort((a,b)=>b.record.width-a.record.width||b.record.units-a.record.units||a.index-b.index);
  let nextWidth=SMALL_CONSTRUCTION_TABLE_WIDTH_LIMIT+1;
  for(const candidate of candidates){
   const low=Math.max(0,candidate.record.width),high=Math.min(SMALL_CONSTRUCTION_TABLE_WIDTH_LIMIT,nextWidth-1);
   for(let requestedWidth=low;requestedWidth<=high;requestedWidth++){
    const offset=smallConstructionIndexOffset(kind,positiveEnd,negativeEnd,rows,requestedWidth);if(offset===undefined)throw Error('Small construction index dimensions changed');
    smallIndex[offset]=candidate.index;
   }
   nextWidth=low;
  }
 }
 const farStrip=new Uint32Array(SMALL_CONSTRUCTION_TABLE_AT+SMALL_CONSTRUCTION_TABLE_RECORDS_AT+packedSmallRecords.length*SMALL_CONSTRUCTION_TABLE_RECORD_WORDS);
 farStrip.set(softPartTable,SOFT_PART_TABLE_AT);farStrip.set(softPhaseTable,SOFT_PHASE_TABLE_AT);farStrip.set(softD5Table,SOFT_D5_TABLE_AT);
 for(const [bin,text]of ['、','y','y','人','个','个','个','个'].entries()){
  const stem=bin>=6;
  farStrip.set([farWords.get(text)!,farEdge(text,false)>>>0,stem?hardWords.get('|')!:0,stem?farEdge('|',false)>>>0:0],67+bin*4);
 }
 //  [99,163) MERGED SKYLINE units (gpu-articulation.ts farSkyline), eight
 //          records of eight words addressed by contributor class (0 far
 //          terrain silhouette, 1 standing soft mass) and by which neighbour
 //          columns of the shared profile are lower (0 neither: a flat run,
 //          1 the left: a rise, 2 the right: a fall, 3 both: a peak):
 //          cap word, x of the cap's apex ink (peak) or 0, then per side the
 //          flank word of the rows under the cap, its pen from the cap's pen
 //          (the cap's foot ink on that side to the flank's top ink) and its
 //          shift per row (the flank's own lean: a run is one unit and its
 //          shift, practitioner rule 1). The line level is a row's TOP edge:
 //          ￣ and ⌒ draw on it, ／ ＼ ∧ ヘ rise to it from the row's bottom
 //          edge, so every unit of one level shares one row and a mark stands
 //          on the line from the row above. Terrain: ￣ ／ ＼ ヘ over | (D9:
 //          the smallest tier simplifies steep diagonals to verticals; the
 //          operator's super-far strip is one base line with ヘ ∧ peaks).
 //          Mass: a straight-shouldered arch (the skill's 15.5 complete arch
 //          ／￣＼; the operator markup's far rocks, hills and houses are flat
 //          caps on slanted shoulders with ∧ peaks). The puff lobe ⌒ヽ is the
 //          smoke idiom and scalloped every far rock (capture self-7 view 06).
 //          Caps ￣ on the run, ／ where it rises, ＼ where it falls, ∧ on a
 //          one-column top; flank words ／ ＼ on ONE row under the cap, only
 //          where that side falls three rows or more (farSkyline).
 //  [163,168) the lobe's base turns, drawn on the last row of a side that
 //          falls two rows or more: left word ヽ_ and its pen from the column,
 //          right word _ノ and its pen from the column (ending on the column's
 //          right edge), both from the skill's 15.5 "base or turn of a lobe";
 //          then the tuft word ,, (15.5 base flicker) of the far base line.
 const farEnd=(text:string,top:boolean,right:boolean):readonly [number,number]=>{
  const ink=farInk(text),xs=ink.map(q=>q[0]!),edge=right?Math.max(...xs):Math.min(...xs),side=ink.filter(q=>Math.abs(q[0]!-edge)<=2);
  const y=side.reduce((v,q)=>top?Math.min(v,q[1]!):Math.max(v,q[1]!),top?Infinity:-Infinity),row=side.filter(q=>q[1]===y);
  return [right?Math.max(...row.map(q=>q[0]!)):Math.min(...row.map(q=>q[0]!)),y];
 };
 const farTip=(text:string,top:boolean):number=>{
  const ink=farInk(text),y=ink.reduce((v,q)=>top?Math.min(v,q[1]!):Math.max(v,q[1]!),top?Infinity:-Infinity),row=ink.filter(q=>q[1]===y);
  return Math.round(row.reduce((sum,q)=>sum+q[0]!,0)/row.length);
 };
 const farUnit=(text:string):number=>{const word=hardWords.get(text)??farWords.get(text);if(!word)throw Error(`Far skyline unit ${text} is not a compiled word`);return word;};
 const farAdvance=(text:string):number=>[...text].reduce((sum,ch)=>sum+font[address(ch)]!.advance,0);
 farStrip.set([farUnit('ヽ_'),0,farUnit('_ノ'),(16-farAdvance('_ノ'))>>>0,farUnit(',,')],163);
 for(const [cls,[caps,flanks]]of ([[['￣','ノ','ヽ','ヘ'],['ﾉ','ヽ']],[['￣','ノ','ヽ','∧'],['ノ','ヽ']]] as const).entries())for(const [shape,cap]of caps.entries()){
  const record=[farUnit(cap),shape===3?farEdge(cap,true):0,0,0,0,0,0,0];
  for(const [side,flank]of flanks.entries()){
   if(!(shape&(1<<side)))continue;
   // The flank continues the cap: its top ink starts under the cap's foot on that side and it leans by its own ink.
   const foot=farEnd(cap,false,side===1)[0],top=farTip(flank,true),bottom=farTip(flank,false);
   record.splice(2+side*3,3,farUnit(flank),(foot-top)>>>0,(bottom-top)>>>0);
  }
  farStrip.set(record,99+(cls*4+shape)*8);
 }
 //  [168,176) THE SIDE OF A PROFILE, one unit per row, read by how far the
 //          form's own edge moves across that row: | (it does not), / (under a
 //          full-width slant), ／ (a full slant, rising), ＼ (falling). Each
 //          record is the word and the x of the middle of its stroke, so the
 //          unit is set on the edge itself. [176,178): the two edge movements
 //          (page px per row) that separate | from / and / from ／, halfway
 //          between the leans of the glyphs' own ink. A slanted side never
 //          takes an upright and a straight side keeps it (D5).
 const sideLean=(text:string)=>farTip(text,false)-farTip(text,true);
 // No lone | / ／ on far forms (operator order 2026-11-10): l upright, ﾉ steep
 // rise, ノ rise, ヽ fall; thresholds from the same glyphs' own leans.
 for(const [k,text]of ['l','ﾉ','ノ','ヽ'].entries())farStrip.set([farUnit(text),Math.round((farTip(text,true)+farTip(text,false))/2)>>>0],168+k*2);
 farStrip.set([Math.round((Math.abs(sideLean('l'))+Math.abs(sideLean('ﾉ')))/2),Math.round((Math.abs(sideLean('ﾉ'))+Math.abs(sideLean('ノ')))/2)],176);
 //  [178,180) THE SHADE BAND of a profile: the unit :;: (the hand target's
 //          far and middle rocks: `| :;: ヽ`, `/:;:;:;`) and its advance. One
 //          constant unit repeated across the shaded cells of the form (D6).
 farStrip.set([farUnit(':;:'),farAdvance(':;:')],178);
 //  [180,182) THE TWO LEVEL UNITS ＿ and ￣ (typesetting reads them): ＿ at
 //          the foot of one row and ￣ at the head of the next draw the same
 //          page height three pixels apart, so two edges there are one line.
 farStrip.set([farUnit('＿'),farUnit('￣')],180);
 // Optional smallest-tier suffix. Its base is after the complete incumbent
 // D5 table, so every legacy far-strip address above remains byte-for-byte
 // stable. Both the direct index and record offsets are relative to suffix.
 const smallSuffixAt=SMALL_CONSTRUCTION_TABLE_AT;
 farStrip.set([SMALL_CONSTRUCTION_TABLE_VERSION,SMALL_CONSTRUCTION_TABLE_RECORD_WORDS,packedSmallRecords.length,
  SMALL_CONSTRUCTION_TABLE_RECORDS_AT,SMALL_CONSTRUCTION_TABLE_INDEX_AT,SMALL_CONSTRUCTION_TABLE_WIDTH_LIMIT,
  SMALL_CONSTRUCTION_TABLE_KIND_COUNT,SMALL_CONSTRUCTION_TABLE_ROW_COUNT],smallSuffixAt);
 farStrip.set(smallIndex,smallSuffixAt+SMALL_CONSTRUCTION_TABLE_INDEX_AT);
 for(const [index,record]of packedSmallRecords.entries())farStrip.set([record.kindAndEnds,record.rows,record.width,record.units,record.firstAtom,record.atomCount,record.firstPaper,record.paperCount],
  smallSuffixAt+SMALL_CONSTRUCTION_TABLE_RECORDS_AT+index*SMALL_CONSTRUCTION_TABLE_RECORD_WORDS);
 // The legacy packer's cursor lookup is compiled once at residency, not
 // searched in a displayed frame. NativePage remains the serializer.
 const cursorThreshold=policy==='preformatted'?40:95,side=cursorThreshold*2+1;
 // Above this conductor every integral scalar gap is representable. Check
 // a full period once at residency, rather than relying on a live pen loop.
 for(let gap=cursorThreshold;gap<cursorThreshold+55;gap++)for(const leading of [false,true])if(!nativeSpacePlan(gap,0,leading,false,policy))throw Error('Native cursor conductor does not cover this whitespace policy');
 const gaps=new Uint32Array((width+1)*2+2*side*side);
 for(let leading=0;leading<2;leading++)for(let wanted=0;wanted<=width;wanted++){
  let legal=wanted;
  while(!nativeSpacePlan(legal,0,!!leading,false,policy))legal++;
  gaps[leading*(width+1)+wanted]=legal;
 }
 // BOTH cursor gaps participate in the one transition. The old forward-only
 // table rejected a whole word when its legal native pen was one dot LEFT.
 // Resolve distance/ties here, once; no displayed frame tries pens. States
 // >= 2*conductor are equivalent for shifts within one conductor: even after
 // the largest shift, that side remains in the all-integral legal tail.
 for(let leading=0;leading<2;leading++)for(let left=0;left<side;left++)for(let right=0;right<side;right++){
  const at=(width+1)*2+(leading*side+left)*side+right;
  outer:for(let distance=0;distance<=cursorThreshold;distance++)for(const shift of distance?[distance,-distance]:[0]){
   if(nativeSpacePlan(left+shift,0,!!leading,false,policy)&&nativeSpacePlan(right-shift,0,false,false,policy)){
    gaps[at]=shift+cursorThreshold+1;break outer;
   }
  }
 }
 return {instructions,articulatedInstructions,hardLineInstructions,words,toneWords,glyphs:Uint32Array.from(metrics),points:Int32Array.from(points),partAtoms:Uint32Array.from(partAtoms),anchorContexts,partPaper:Uint32Array.from(partPaper),circleProgram,parts,partEntries,gaps,gapWidth:width,cursorThreshold,farStrip,hardJunctions,terrainInstructions,pairedFlanks,foliageEdges,font,sourceSha256:grammar.sourceSha256};
}
