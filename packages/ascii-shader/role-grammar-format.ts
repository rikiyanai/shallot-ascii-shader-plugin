import type {NativeGlyph} from './native-page.ts';
import type {ROLE_ENTRIES} from './grammar/role-bank.ts';

/** Runtime ABI shared by the build-time grammar compiler and the playback loader. */
export const WORD_STRIDE=12;
export const GRAMMAR_RECORD_WORDS=12,GRAMMAR_DIRECTIONS=64,GRAMMAR_PHASES=17;
export const GRAMMAR_ROLES=['rim_cap','shoulder','flank','return_base'] as const;
/**
 * The source-part soft consumer is deliberately packed into the reserved
 * prefix of the far-strip bank.  Keeping this ABI in the existing bank means
 * old baked grammar artifacts remain loadable: a zero version means the
 * programmed path keeps its incumbent construction.
 */
export const SOFT_PART_TABLE_VERSION=2;
/** 3 bits for each source tangent and 3 bits for the retained context class. */
export const SOFT_PART_CONTEXTS=1<<9;
export const SOFT_PART_CONTEXT_IN_MASK=7;
export const SOFT_PART_CONTEXT_OUT_SHIFT=3;
export const SOFT_PART_CONTEXT_CLASS_SHIFT=6;
export const SOFT_PART_CONTEXT_CLASS_MASK=7;
/**
 * The suffix is deliberately outside the fixed far-strip records.  Existing
 * far playback therefore keeps its old addresses while an old baked bank
 * naturally fails the version/record-count check and retains its incumbent.
 */
export const SOFT_PART_TABLE_AT=182;
export const SOFT_PART_TABLE_RECORD_WORDS=8;
export const SOFT_PART_TABLE_RECORDS=GRAMMAR_ROLES.length*3*SOFT_PART_CONTEXTS;
export const SOFT_PART_TABLE_WORDS=4+SOFT_PART_TABLE_RECORDS*SOFT_PART_TABLE_RECORD_WORDS;
/**
 * The soft nearest-phase index (crawl brief item F, T11) rides in the far
 * strip right after the soft-part table, with its own version word: all 12
 * bank header words are allocated, so no new header slot exists. One byte
 * per (table, column, phase) over the three tables softPortion reads
 * (instructions, articulated, terrain): the phase index the T9 rule selects,
 * or SOFT_PHASE_TABLE_NONE when no phase qualifies. Header words are
 * [version, words per table, tables, phases]. An artifact that predates the
 * table fails the version/bounds gate and the frame keeps the T9 scan.
 */
export const SOFT_PHASE_TABLE_AT=SOFT_PART_TABLE_AT+SOFT_PART_TABLE_WORDS;
export const SOFT_PHASE_TABLE_VERSION=1;
export const SOFT_PHASE_TABLE_TABLES=3;
export const SOFT_PHASE_TABLE_COLUMNS=GRAMMAR_ROLES.length*3*GRAMMAR_DIRECTIONS;
export const SOFT_PHASE_TABLE_PHASES=GRAMMAR_PHASES;
export const SOFT_PHASE_TABLE_NONE=0xff;
export const SOFT_PHASE_TABLE_BYTES_PER_TABLE=SOFT_PHASE_TABLE_COLUMNS*SOFT_PHASE_TABLE_PHASES;
export const SOFT_PHASE_TABLE_WORDS_PER_TABLE=SOFT_PHASE_TABLE_BYTES_PER_TABLE/4;
export const SOFT_PHASE_TABLE_HEADER_WORDS=4;
export const SOFT_PHASE_TABLE_WORDS=SOFT_PHASE_TABLE_HEADER_WORDS+SOFT_PHASE_TABLE_TABLES*SOFT_PHASE_TABLE_WORDS_PER_TABLE;
/**
 * The D5 progression table follows the nearest-phase table in farStrip.  Each
 * context has two packed entries: the shallow lattice result and the whole-row
 * result.  The low 30 bits are an offset in either the selected soft table or
 * the hard-line table; bit 30 selects the hard table, bit 31 preserves the
 * source record's pre-swap upright classification.  A versioned suffix keeps
 * old banks loadable: they retain the incumbent runtime D5 path.
 */
export const SOFT_D5_TABLE_AT=SOFT_PHASE_TABLE_AT+SOFT_PHASE_TABLE_WORDS;
export const SOFT_D5_TABLE_VERSION=1;
export const SOFT_D5_TABLE_TABLES=3;
export const SOFT_D5_TABLE_COLUMNS=SOFT_PHASE_TABLE_COLUMNS;
export const SOFT_D5_TABLE_PHASES=GRAMMAR_PHASES;
export const SOFT_D5_TABLE_VARIANTS=2;
export const SOFT_D5_TABLE_ENTRY_WORDS=2;
export const SOFT_D5_TABLE_HEADER_WORDS=6;
export const SOFT_D5_TABLE_NONE=0x3fffffff;
export const SOFT_D5_TABLE_HARD=0x40000000;
export const SOFT_D5_TABLE_UPRIGHT=0x80000000;
export const SOFT_D5_TABLE_WORDS=SOFT_D5_TABLE_HEADER_WORDS+
  SOFT_D5_TABLE_VARIANTS*SOFT_D5_TABLE_TABLES*SOFT_D5_TABLE_COLUMNS*SOFT_D5_TABLE_PHASES*SOFT_D5_TABLE_ENTRY_WORDS;
/**
 * Small parent constructions are a versioned suffix, not a replacement for
 * any incumbent far-strip record. The suffix has a direct lookup index so a
 * frame never scans a variable-width record family. Header words are
 * [version, stride, count, recordsBaseRelToSuffix, indexBaseRelToSuffix,
 * widthLimit, kindCount, rowCount]. The index stores a record ordinal or
 * SMALL_CONSTRUCTION_TABLE_NONE for each (kind/end key, rows, width) cell.
 * Record offsets are in existing four-word partAtoms/partPaper records (not
 * scalar-byte offsets): [kindAndEnds, rows, width, units, firstAtom,
 * atomCount, firstPaper, paperCount]. The atom first word remains the
 * incumbent native-font glyph address used by retained-part playback.
 */
export const SMALL_CONSTRUCTION_TABLE_AT=SOFT_D5_TABLE_AT+SOFT_D5_TABLE_WORDS;
export const SMALL_CONSTRUCTION_TABLE_VERSION=1;
export const SMALL_CONSTRUCTION_TABLE_HEADER_WORDS=8;
export const SMALL_CONSTRUCTION_TABLE_RECORD_WORDS=8;
export const SMALL_CONSTRUCTION_TABLE_WIDTH_LIMIT=1024;
export const SMALL_CONSTRUCTION_TABLE_KIND_COUNT=3;
export const SMALL_CONSTRUCTION_TABLE_END_COUNT=3;
export const SMALL_CONSTRUCTION_TABLE_ROW_COUNT=5;
export const SMALL_CONSTRUCTION_TABLE_KEY_COUNT=SMALL_CONSTRUCTION_TABLE_KIND_COUNT*SMALL_CONSTRUCTION_TABLE_END_COUNT*SMALL_CONSTRUCTION_TABLE_END_COUNT;
export const SMALL_CONSTRUCTION_TABLE_INDEX_AT=SMALL_CONSTRUCTION_TABLE_HEADER_WORDS;
export const SMALL_CONSTRUCTION_TABLE_INDEX_WORDS=SMALL_CONSTRUCTION_TABLE_KEY_COUNT*SMALL_CONSTRUCTION_TABLE_ROW_COUNT*(SMALL_CONSTRUCTION_TABLE_WIDTH_LIMIT+1);
export const SMALL_CONSTRUCTION_TABLE_RECORDS_AT=SMALL_CONSTRUCTION_TABLE_INDEX_AT+SMALL_CONSTRUCTION_TABLE_INDEX_WORDS;
export const SMALL_CONSTRUCTION_TABLE_KIND_MASK=3;
export const SMALL_CONSTRUCTION_TABLE_POSITIVE_SHIFT=2;
export const SMALL_CONSTRUCTION_TABLE_NEGATIVE_SHIFT=4;
export const SMALL_CONSTRUCTION_TABLE_END_MASK=3;
export const SMALL_CONSTRUCTION_TABLE_NONE=0xffffffff;
export type SmallConstructionKind='round'|'limb'|'box';
export type SmallConstructionEnd='cap'|'contact'|'open';
export interface PackedSmallConstructionRecord {
 readonly kind:SmallConstructionKind;readonly positiveEnd:SmallConstructionEnd;
 readonly negativeEnd:SmallConstructionEnd;readonly rows:number;readonly width:number;
 readonly units:number;readonly firstAtom:number;readonly atomCount:number;
 readonly firstPaper:number;readonly paperCount:number;
}
export interface PackedSmallConstructionTable {
 readonly version:number;readonly stride:number;readonly count:number;
 readonly recordsBaseRelToSuffix:number;readonly indexBaseRelToSuffix:number;
 readonly widthLimit:number;readonly kindCount:number;readonly rowCount:number;
 readonly index:Uint32Array;readonly records:readonly PackedSmallConstructionRecord[];
}
const SMALL_KIND_CODES:readonly SmallConstructionKind[]=['round','limb','box'];
const SMALL_END_CODES:readonly SmallConstructionEnd[]=['cap','contact','open'];
const smallCode=(values:readonly string[],value:string,label:string):number=>{
 const code=values.indexOf(value);if(code<0)throw Error(`Unknown small construction ${label}: ${value}`);return code;
};
export function packSmallConstructionKindAndEnds(kind:SmallConstructionKind,positiveEnd:SmallConstructionEnd,negativeEnd:SmallConstructionEnd):number{
 return (smallCode(SMALL_KIND_CODES,kind,'kind')|
   (smallCode(SMALL_END_CODES,positiveEnd,'positive end')<<SMALL_CONSTRUCTION_TABLE_POSITIVE_SHIFT)|
   (smallCode(SMALL_END_CODES,negativeEnd,'negative end')<<SMALL_CONSTRUCTION_TABLE_NEGATIVE_SHIFT))>>>0;
}
export function unpackSmallConstructionKindAndEnds(value:number):Pick<PackedSmallConstructionRecord,'kind'|'positiveEnd'|'negativeEnd'>|undefined{
 const kind=SMALL_KIND_CODES[value&SMALL_CONSTRUCTION_TABLE_KIND_MASK];
 const positiveEnd=SMALL_END_CODES[(value>>SMALL_CONSTRUCTION_TABLE_POSITIVE_SHIFT)&SMALL_CONSTRUCTION_TABLE_END_MASK];
 const negativeEnd=SMALL_END_CODES[(value>>SMALL_CONSTRUCTION_TABLE_NEGATIVE_SHIFT)&SMALL_CONSTRUCTION_TABLE_END_MASK];
 return kind&&positiveEnd&&negativeEnd?{kind,positiveEnd,negativeEnd}:undefined;
}
export function smallConstructionIndexOffset(kind:SmallConstructionKind,positiveEnd:SmallConstructionEnd,negativeEnd:SmallConstructionEnd,rows:number,width:number):number|undefined{
 const kindCode=SMALL_KIND_CODES.indexOf(kind),positiveCode=SMALL_END_CODES.indexOf(positiveEnd),negativeCode=SMALL_END_CODES.indexOf(negativeEnd);
 if(kindCode<0||positiveCode<0||negativeCode<0||!Number.isInteger(rows)||!Number.isInteger(width)||rows<1||rows>SMALL_CONSTRUCTION_TABLE_ROW_COUNT||width<0||width>SMALL_CONSTRUCTION_TABLE_WIDTH_LIMIT)return undefined;
 const key=(kindCode*SMALL_CONSTRUCTION_TABLE_END_COUNT+positiveCode)*SMALL_CONSTRUCTION_TABLE_END_COUNT+negativeCode;
 return (key*SMALL_CONSTRUCTION_TABLE_ROW_COUNT+(rows-1))*(SMALL_CONSTRUCTION_TABLE_WIDTH_LIMIT+1)+width;
}
/** Read the optional suffix; old banks intentionally return undefined. */
export function decodeSmallConstructionTable(farStrip:Uint32Array):PackedSmallConstructionTable|undefined{
 const at=SMALL_CONSTRUCTION_TABLE_AT;
 if(farStrip.length<at+SMALL_CONSTRUCTION_TABLE_HEADER_WORDS)return undefined;
 const version=farStrip[at]!,stride=farStrip[at+1]!,count=farStrip[at+2]!,recordsBase=farStrip[at+3]!,indexBase=farStrip[at+4]!,widthLimit=farStrip[at+5]!,kindCount=farStrip[at+6]!,rowCount=farStrip[at+7]!;
 if(version!==SMALL_CONSTRUCTION_TABLE_VERSION||stride!==SMALL_CONSTRUCTION_TABLE_RECORD_WORDS||recordsBase<SMALL_CONSTRUCTION_TABLE_HEADER_WORDS||indexBase<SMALL_CONSTRUCTION_TABLE_HEADER_WORDS||widthLimit!==SMALL_CONSTRUCTION_TABLE_WIDTH_LIMIT||kindCount!==SMALL_CONSTRUCTION_TABLE_KIND_COUNT||rowCount!==SMALL_CONSTRUCTION_TABLE_ROW_COUNT)return undefined;
 const indexAt=at+indexBase,recordsAt=at+recordsBase,indexWords=kindCount*SMALL_CONSTRUCTION_TABLE_END_COUNT*SMALL_CONSTRUCTION_TABLE_END_COUNT*rowCount*(widthLimit+1);
 if(indexAt>farStrip.length||indexWords>farStrip.length-indexAt||recordsAt>farStrip.length||count>(farStrip.length-recordsAt)/stride)return undefined;
 const records:PackedSmallConstructionRecord[]=[];
 for(let index=0;index<count;index++){
  const recordAt=recordsAt+index*stride,tag=unpackSmallConstructionKindAndEnds(farStrip[recordAt]!);
  if(!tag)return undefined;
  records.push({...tag,rows:farStrip[recordAt+1]!,width:farStrip[recordAt+2]!,units:farStrip[recordAt+3]!,
   firstAtom:farStrip[recordAt+4]!,atomCount:farStrip[recordAt+5]!,firstPaper:farStrip[recordAt+6]!,paperCount:farStrip[recordAt+7]!});
 }
 return {version,stride,count,recordsBaseRelToSuffix:recordsBase,indexBaseRelToSuffix:indexBase,widthLimit,kindCount,rowCount,
  index:farStrip.slice(indexAt,indexAt+indexWords),records};
}
/** Pure CPU counterpart of the GPU's direct table address. */
export function smallConstructionRecordIndex(table:PackedSmallConstructionTable,kind:SmallConstructionKind,positiveEnd:SmallConstructionEnd,negativeEnd:SmallConstructionEnd,rows:number,width:number):number|undefined{
 const boundedWidth=Math.max(0,Math.min(table.widthLimit,Math.floor(width))),offset=smallConstructionIndexOffset(kind,positiveEnd,negativeEnd,rows,boundedWidth);
 if(offset===undefined)return undefined;
 const index=table.index[offset]!;return index===SMALL_CONSTRUCTION_TABLE_NONE?undefined:index;
}
export function lookupSmallConstructionRecord(table:PackedSmallConstructionTable,kind:SmallConstructionKind,positiveEnd:SmallConstructionEnd,negativeEnd:SmallConstructionEnd,rows:number,width:number):PackedSmallConstructionRecord|undefined{
 const index=smallConstructionRecordIndex(table,kind,positiveEnd,negativeEnd,rows,width);return index===undefined?undefined:table.records[index];
}
export type GrammarRole=typeof GRAMMAR_ROLES[number];
export type GrammarRoleEntry=(typeof ROLE_ENTRIES)[number];

export interface CompiledRolePart {
 readonly id:string;readonly role:string;readonly band:string;readonly rows:number;
 readonly first:number;readonly count:number;readonly width:number;
 readonly paperFirst:number;readonly paperCount:number;
}

export interface PackedRoleGrammar {
 readonly instructions:Uint32Array;
 readonly articulatedInstructions:Uint32Array;
 readonly hardLineInstructions:Uint32Array;
 readonly words:Uint32Array;
 readonly toneWords:number;
 readonly glyphs:Uint32Array;readonly points:Int32Array;
 readonly partAtoms:Uint32Array;readonly parts:readonly CompiledRolePart[];
 readonly anchorContexts:Uint32Array;
 readonly partPaper:Uint32Array;
 readonly circleProgram:Uint32Array;
 readonly partEntries:readonly GrammarRoleEntry[];
 readonly gaps:Uint32Array;readonly gapWidth:number;
 readonly cursorThreshold:number;
 readonly farStrip:Uint32Array;
 readonly hardJunctions:Uint32Array;
 readonly terrainInstructions:Uint32Array;
 readonly pairedFlanks:Uint32Array;
 readonly foliageEdges:Uint32Array;
 readonly font:readonly NativeGlyph[];readonly sourceSha256:string;
}

/* These constants are part of the GPU grammar ABI, not fitting decisions. */
export const HARD_SLOPE_PHASES=[3,6,11,14,18] as const;
export const PAIRED_FLANK_LEANS=[-12,-2.5,2.5,12] as const;
export const TERRAIN_INSTRUCTION_WORDS=4*3*64*17*12;
export const FOLIAGE_EDGES_AT=(PAIRED_FLANK_LEANS.length+1)*4+TERRAIN_INSTRUCTION_WORDS;
export const JUNCTION_DIRECTIONS=16;
export const JUNCTION_TWO=0,JUNCTION_THREE=512,JUNCTION_RECORDS=512+2*4096;
