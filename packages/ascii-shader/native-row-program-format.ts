import {NATIVE_WHITESPACE_WGSL,nativeSpacePlan,type NativeWhitespacePolicy} from './text/native-whitespace.ts';
import type {CompiledNativeRow,CompiledNativeRowProgram,NativePaperPlan,NativeRowGrowthMode,NativeRowHeightProgram} from './compiled-native-row-units.ts';

/** Versioned suffix appended after the existing far-strip products. */
export const NATIVE_ROW_PROGRAM_VERSION=1;
export const NATIVE_ROW_PROGRAM_NONE=0xffffffff;
export const NATIVE_ROW_PROGRAM_HEADER_WORDS=24;
export const NATIVE_ROW_PROGRAM_STRIDE=12;
export const NATIVE_ROW_STRIDE=32;
export const NATIVE_ROW_UNIT_STRIDE=8;
export const NATIVE_ROW_ATOM_STRIDE=7;
export const NATIVE_ROW_PAPER_STRIDE=5;
export const NATIVE_ROW_RESIDUE_STRIDE=4;
export const NATIVE_ROW_PREFIX_STRIDE=2;
/** Optional trailer after the v1 row sections; old banks remain valid. */
export const NATIVE_ROW_HEIGHT_MAGIC=0x48544748;
export const NATIVE_ROW_HEIGHT_VERSION=1;
export const NATIVE_ROW_HEIGHT_HEADER_WORDS=9;
export const NATIVE_ROW_HEIGHT_RECORD_STRIDE=10;
export const NATIVE_ROW_HEIGHT_UNIT_STRIDE=4;
/** Height pivots are stored in exact half-pixel fixed point. */
export const NATIVE_ROW_HEIGHT_PIVOT_SCALE=2;

export const NATIVE_ROW_MODE_BASELINE=0;
export const NATIVE_ROW_MODE_LEFT_FIXED=1;
export const NATIVE_ROW_MODE_RIGHT_SHIFT=2;
export const NATIVE_ROW_MODE_GAP=3;

const POLICY_CODE={constrained:0,preformatted:1} as const;
const MODE_CODE:Readonly<Record<NativeRowGrowthMode,number>>={baseline:NATIVE_ROW_MODE_BASELINE,'left-fixed':NATIVE_ROW_MODE_LEFT_FIXED,'right-shift':NATIVE_ROW_MODE_RIGHT_SHIFT,gap:NATIVE_ROW_MODE_GAP};
const KIND_CODE:Readonly<Record<string,number>>={anchor:0,arc:1,tone:2,ink:3};
const repeatableBit=0x80000000;

/** Header indexes. Every offset is relative to the suffix base. */
const H={version:0,headerWords:1,totalWords:2,programCount:3,rowCount:4,unitCount:5,atomCount:6,paperCount:7,residueCount:8,prefixCount:9,programAt:10,rowAt:11,unitAt:12,atomAt:13,paperAt:14,residueAt:15,prefixAt:16,programStride:17,rowStride:18,unitStride:19,atomStride:20,paperStride:21,residueStride:22,prefixStride:23} as const;
/** Program record indexes. */
const P={rowFirst:0,rowCount:1,sourceOrigin:2,constructionWidth:3,seam:4,sourceWidthMin:5,sourceWidthMax:6,maxGrowthPx:7,policy:8,roleHash:9,contextHash:10,morphologyHash:11} as const;
/** Row record indexes. */
const R={row:0,sourceRow:1,authoredWidth:2,sourceOrigin:3,leftInset:4,unitFirst:5,unitCount:6,wordCount:7,paperFirst:8,paperCount:9,atomFirst:10,atomCount:11,mode:12,maxDelta:13,growthGapIndex:14,baseGap:15,growthAnchorX:16,growthAnchorMember:17,repeatMember:18,repeatBefore:19,repeatX:20,repeatUnitWidth:21,repeatGap:22,paperTailStart:23,residueFirst:24,residueCount:25,prefixFirst:26,prefixCount:27,growthSlot:28,repeatFull:29,repeatHalf:30,policy:31} as const;
/** Unit record indexes. */
const U={member:0,kind:1,x:2,width:3,atomFirst:4,atomCount:5,wordStart:6,wordEnd:7} as const;
/** Atom record indexes. */
const A={member:0,ordinal:1,x:2,advance:3,glyph:4,wordStart:5,wordEnd:6} as const;
/** Paper record indexes. */
const W={owner:0,x:1,width:2,full:3,half:4} as const;

export interface PackedNativeRowProgramTable {
 readonly storage:Uint32Array; readonly base:number; readonly totalWords:number;
 readonly programCount:number; readonly rowCount:number; readonly unitCount:number;
 readonly atomCount:number; readonly paperCount:number; readonly residueCount:number; readonly prefixCount:number;
 readonly offsets:Readonly<{program:number;row:number;unit:number;atom:number;paper:number;residue:number;prefix:number}>;
 readonly height?:PackedNativeRowHeightTable;
}
export interface PackedNativeRowHeightTable {
 readonly storage:Uint32Array; readonly base:number; readonly totalWords:number;
 readonly programCount:number; readonly recordStride:number; readonly unitStride:number;
 readonly recordsAt:number; readonly unitsAt:number;
}
export interface PackedNativeRowGrowth {readonly repeatCount:number;readonly repeatWidth:number;readonly paperDelta:number;readonly valid:true;}
export interface PackedNativeRowAtom {readonly member:number;readonly ordinal:number;readonly x:number;readonly advance:number;readonly glyph:number;readonly wordStart:number;readonly wordEnd:number;}
export interface PackedNativeRowPaper {readonly owner:number;readonly x:number;readonly width:number;readonly full:number;readonly half:number;}

const u32=(value:number,label:string):number=>{
 if(!Number.isSafeInteger(value)||value<0||value>0xffffffff)throw Error(`Native row ${label} is not a u32: ${value}`);
 return value>>>0;
};
const i32=(value:number,label:string):number=>{
 if(!Number.isSafeInteger(value)||value<-0x80000000||value>0x7fffffff)throw Error(`Native row ${label} is not an i32: ${value}`);
 return value|0;
};
const signed=(value:number):number=>value>>>0;
const readI32=(storage:Uint32Array,at:number):number=>storage[at]!|0;
const floatBits=(value:number):number=>{if(!Number.isFinite(value))throw Error(`Native row float is not finite: ${value}`);const f=new Float32Array([value]),u=new Uint32Array(f.buffer);return u[0]!;};
const hash=(value:string):number=>{
 let result=2166136261;
 for(const scalar of value){result^=scalar.codePointAt(0)!;result=Math.imul(result,16777619)>>>0;}
 return result>>>0;
};
const stableContextHash=(value:Readonly<Record<string,unknown>>):number=>hash(Object.keys(value).sort().map(key=>`${key}:${String(value[key])}`).join('|'));
const policyCode=(policy:NativeWhitespacePolicy):number=>POLICY_CODE[policy];
const modeCode=(mode:NativeRowGrowthMode):number=>MODE_CODE[mode];
const planWords=(plan:NativePaperPlan|undefined):[number,number]=>plan?[u32(plan.full,'paper full'),u32(plan.half,'paper half')]:[NATIVE_ROW_PROGRAM_NONE,NATIVE_ROW_PROGRAM_NONE];
const codeForKind=(kind:string):number=>{
 const code=KIND_CODE[kind];if(code===undefined)throw Error(`Unknown native row unit kind ${kind}`);return code;
};
const repeatMemberOf=(row:CompiledNativeRow):number=>row.growth.repeatMember===undefined?NATIVE_ROW_PROGRAM_NONE:u32(row.growth.repeatMember,'repeat member');
const growthAnchor=(row:CompiledNativeRow):{x:number;member:number}=>{
 const member=row.growthSlot>0?row.growthSlot+1:NATIVE_ROW_PROGRAM_NONE;
 const unit=row.units.find(candidate=>candidate.member===member);
 const words=row.units.filter(candidate=>candidate.kind!=='paper');
 const terminal=words.at(-1);
 return {x:i32(unit?.x??terminal?.x??row.authoredWidth,'growth anchor x'),member:u32(member,'growth anchor member')};
};
interface Flattened {programs:number[];rows:number[];units:number[];atoms:number[];papers:number[];residues:number[];prefixes:number[];heightRecords:number[];heightUnits:number[];}

const packHeight=(flat:Flattened,height:NativeRowHeightProgram|undefined):void=>{
 if(!height){flat.heightRecords.push(0,0,0,0,0,0,0,0,0,0);return;}
 const unitFirst=flat.heightUnits.length/NATIVE_ROW_HEIGHT_UNIT_STRIDE;
 for(const unit of height.sequence)flat.heightUnits.push(u32(unit.sourceRow,'height source row'),u32(unit.member,'height member'),signed(i32(unit.leftAnchorX,'height left anchor')),signed(i32(unit.rightAnchorX,'height right anchor')));
 flat.heightRecords.push(u32(height.authoredRows,'height authored rows'),u32(height.minRows,'height minimum rows'),u32(height.maxRows,'height maximum rows'),u32(unitFirst,'height sequence first'),u32(height.sequence.length,'height sequence count'),signed(i32(height.leftCycleShift,'height left cycle shift')),signed(i32(height.rightCycleShift,'height right cycle shift')),signed(i32(Math.round(height.pivot*NATIVE_ROW_HEIGHT_PIVOT_SCALE),'height pivot half-pixels')),u32(height.memberStride,'height member stride'),u32(height.finalAnchorMember,'height final anchor'));
};

const packProgram=(flat:Flattened,program:CompiledNativeRowProgram|undefined):void=>{
 if(!program){flat.programs.push(0,0,0,0,0,0,0,0,0,0,0,0);packHeight(flat,undefined);return;}
 if(!Number.isSafeInteger(program.sourceOrigin)||!Number.isSafeInteger(program.constructionWidth)||program.constructionWidth<0)throw Error('Native row programme has invalid common extent');
 if(!Number.isSafeInteger(program.maxGrowthPx)||program.maxGrowthPx<0)throw Error('Native row programme has invalid growth extent');
 const rowFirst=flat.rows.length/NATIVE_ROW_STRIDE;
 for(const row of program.rows){
  if(row.sourceOrigin!==program.sourceOrigin)throw Error('Native row programme rows do not share source origin');
  if(!Number.isSafeInteger(row.authoredWidth)||row.authoredWidth<0)throw Error('Native row has invalid authored width');
  const unitFirst=flat.units.length/NATIVE_ROW_UNIT_STRIDE,atomFirst=flat.atoms.length/NATIVE_ROW_ATOM_STRIDE,paperFirst=flat.papers.length/NATIVE_ROW_PAPER_STRIDE;
  const words=row.units.filter(unit=>unit.kind!=='paper'),paperUnits=row.units.filter(unit=>unit.kind==='paper');
  for(const unit of words){
   const atoms=unit.atoms,localAtomFirst=flat.atoms.length/NATIVE_ROW_ATOM_STRIDE;
   for(const [ordinal,atom] of atoms.entries())flat.atoms.push(u32(unit.member,'atom member'),u32(ordinal,'atom ordinal'),signed(i32(atom.x,'atom x')),u32(atom.advance,'atom advance'),u32(atom.glyph,'atom glyph'),u32(localAtomFirst,'atom word start'),u32(localAtomFirst+atoms.length,'atom word end'));
   flat.units.push(u32(unit.member,'unit member'),(codeForKind(unit.kind)|(unit.repeatable?repeatableBit:0))>>>0,signed(i32(unit.x,'unit x')),u32(unit.width,'unit width'),u32(localAtomFirst,'unit atom first'),u32(atoms.length,'unit atom count'),u32(localAtomFirst,'unit word start'),u32(localAtomFirst+atoms.length,'unit word end'));
  }
  for(const unit of paperUnits){
   const plan=nativeSpacePlan(unit.width,0,false,false,row.policy),full=plan?u32(plan.full,'paper full'):NATIVE_ROW_PROGRAM_NONE,half=plan?u32(plan.half,'paper half'):NATIVE_ROW_PROGRAM_NONE;
   flat.papers.push(u32(unit.member,'paper owner'),signed(i32(unit.x,'paper x')),u32(unit.width,'paper width'),full,half);
  }
  const growth=row.growth,residueFirst=flat.residues.length/NATIVE_ROW_RESIDUE_STRIDE,prefixFirst=flat.prefixes.length/NATIVE_ROW_PREFIX_STRIDE;
  for(const residue of growth.residues){
   if(!residue){flat.residues.push(NATIVE_ROW_PROGRAM_NONE,NATIVE_ROW_PROGRAM_NONE,NATIVE_ROW_PROGRAM_NONE,NATIVE_ROW_PROGRAM_NONE);continue;}
   const [full,half]=planWords(residue.paper);flat.residues.push(u32(residue.skip,'residue skip'),u32(residue.paperDelta,'residue paper delta'),full,half);
  }
  for(const prefix of growth.paperPrefix){const [full,half]=planWords(prefix);flat.prefixes.push(full,half);}
  const anchor=growthAnchor(row),repeatMember=repeatMemberOf(row),repeatUnitSource=row.units.find(unit=>unit.member===growth.repeatMember),repeatPlan=growth.repeatGap>0?nativeSpacePlan(growth.repeatGap,0,false,false,row.policy):undefined;
  const repeatFull=repeatPlan?u32(repeatPlan.full,'repeat paper full'):NATIVE_ROW_PROGRAM_NONE,repeatHalf=repeatPlan?u32(repeatPlan.half,'repeat paper half'):NATIVE_ROW_PROGRAM_NONE;
  const rowRecord=[
   u32(row.row,'row'),u32(row.sourceRow,'source row'),u32(row.authoredWidth,'authored width'),signed(i32(row.sourceOrigin,'row source origin')),signed(i32(row.leftInset,'left inset')),
   u32(unitFirst,'unit first'),u32(words.length,'unit count'),u32(words.length,'word count'),u32(paperFirst,'paper first'),u32(paperUnits.length,'paper count'),u32(atomFirst,'atom first'),u32(flat.atoms.length/NATIVE_ROW_ATOM_STRIDE-atomFirst,'atom count'),
   u32(modeCode(growth.mode)|(growth.requireRepeat?0x100:0)|(paperUnits.some(unit=>!nativeSpacePlan(unit.width,0,false,false,row.policy))?0x200:0),'growth mode'),u32(growth.maxDelta,'growth max delta'),growth.growthGapIndex<0?NATIVE_ROW_PROGRAM_NONE:u32(growth.growthGapIndex,'growth gap index'),u32(growth.baseGap,'growth base gap'),signed(anchor.x),anchor.member,repeatMember,growth.repeatBefore?1:0,signed(i32(repeatUnitSource?.x??0,'repeat x')),u32(growth.repeatUnitWidth,'repeat unit width'),u32(growth.repeatGap,'repeat gap'),u32(growth.paperTailStart,'paper tail start'),u32(residueFirst,'residue first'),u32(growth.residues.length,'residue count'),u32(prefixFirst,'prefix first'),u32(growth.paperPrefix.length,'prefix count'),u32(row.growthSlot,'growth slot'),repeatFull,repeatHalf,policyCode(row.policy)
  ];
  if(rowRecord.length!==NATIVE_ROW_STRIDE)throw Error('Native row internal stride mismatch');flat.rows.push(...rowRecord);
 }
 packHeight(flat,program.height);
 flat.programs.push(u32(rowFirst,'program row first'),u32(program.rows.length,'program row count'),signed(i32(program.sourceOrigin,'program source origin')),u32(program.constructionWidth,'construction width'),floatBits(program.seam),u32(program.sourceWidth.min,'source width min'),u32(program.sourceWidth.max,'source width max'),u32(program.maxGrowthPx,'program max growth'),policyCode(program.policy),hash(program.role),stableContextHash(program.context as unknown as Record<string,unknown>),hash(program.morphology??''));
};

/** Pack retained programs without shifting missing entry ordinals. */
export function packNativeRowPrograms(programs:readonly (CompiledNativeRowProgram|undefined)[]):Uint32Array{
 const flat:Flattened={programs:[],rows:[],units:[],atoms:[],papers:[],residues:[],prefixes:[],heightRecords:[],heightUnits:[]};
 for(const program of programs)packProgram(flat,program);
 const sections=[flat.programs,flat.rows,flat.units,flat.atoms,flat.papers,flat.residues,flat.prefixes],offsets:number[]=[];let cursor=NATIVE_ROW_PROGRAM_HEADER_WORDS;
 for(const section of sections){offsets.push(cursor);cursor+=section.length;}
 const heightAt=cursor,recordsAt=heightAt+NATIVE_ROW_HEIGHT_HEADER_WORDS,unitsAt=recordsAt+flat.heightRecords.length, totalWords=unitsAt+flat.heightUnits.length;
 const out=new Uint32Array(totalWords),header=[NATIVE_ROW_PROGRAM_VERSION,NATIVE_ROW_PROGRAM_HEADER_WORDS,totalWords,programs.length,flat.rows.length/NATIVE_ROW_STRIDE,flat.units.length/NATIVE_ROW_UNIT_STRIDE,flat.atoms.length/NATIVE_ROW_ATOM_STRIDE,flat.papers.length/NATIVE_ROW_PAPER_STRIDE,flat.residues.length/NATIVE_ROW_RESIDUE_STRIDE,flat.prefixes.length/NATIVE_ROW_PREFIX_STRIDE,...offsets,NATIVE_ROW_PROGRAM_STRIDE,NATIVE_ROW_STRIDE,NATIVE_ROW_UNIT_STRIDE,NATIVE_ROW_ATOM_STRIDE,NATIVE_ROW_PAPER_STRIDE,NATIVE_ROW_RESIDUE_STRIDE,NATIVE_ROW_PREFIX_STRIDE];
 if(header.length!==NATIVE_ROW_PROGRAM_HEADER_WORDS)throw Error('Native row header stride mismatch');out.set(header);for(const [index,section]of sections.entries())out.set(section,offsets[index]!);
 out.set([NATIVE_ROW_HEIGHT_MAGIC,NATIVE_ROW_HEIGHT_VERSION,NATIVE_ROW_HEIGHT_HEADER_WORDS,programs.length,NATIVE_ROW_HEIGHT_RECORD_STRIDE,NATIVE_ROW_HEIGHT_UNIT_STRIDE,recordsAt,unitsAt,totalWords],heightAt);
 out.set(flat.heightRecords,recordsAt);out.set(flat.heightUnits,unitsAt);return out;
}

const sectionFits=(base:number,total:number,offset:number,count:number,stride:number,storage:number):boolean=>Number.isSafeInteger(offset)&&Number.isSafeInteger(count)&&offset>=NATIVE_ROW_PROGRAM_HEADER_WORDS&&offset<=total&&count<=Math.floor((total-offset)/stride)&&base+offset+count*stride<=storage;
const addressIn=(table:PackedNativeRowProgramTable,offset:number,count:number,stride:number,address:number):boolean=>address>=table.base+offset&&address<table.base+offset+count*stride&&((address-(table.base+offset))%stride===0);

/** Decode and validate a suffix. Missing, old, truncated and corrupt data is rejected. */
export function decodeNativeRowPrograms(storage:Uint32Array,base=0):PackedNativeRowProgramTable|undefined{
 if(!(storage instanceof Uint32Array)||!Number.isSafeInteger(base)||base<0||base+NATIVE_ROW_PROGRAM_HEADER_WORDS>storage.length)return;
 const h=base;if(storage[h+H.version]!==NATIVE_ROW_PROGRAM_VERSION||storage[h+H.headerWords]!==NATIVE_ROW_PROGRAM_HEADER_WORDS)return;
 const total=storage[h+H.totalWords]!,counts=[storage[h+H.programCount]!,storage[h+H.rowCount]!,storage[h+H.unitCount]!,storage[h+H.atomCount]!,storage[h+H.paperCount]!,storage[h+H.residueCount]!,storage[h+H.prefixCount]!];
 if(total<NATIVE_ROW_PROGRAM_HEADER_WORDS||base+total>storage.length||counts.some(value=>!Number.isSafeInteger(value)))return;
 const strides=[storage[h+H.programStride]!,storage[h+H.rowStride]!,storage[h+H.unitStride]!,storage[h+H.atomStride]!,storage[h+H.paperStride]!,storage[h+H.residueStride]!,storage[h+H.prefixStride]!];
 if(strides.join(',')!==[NATIVE_ROW_PROGRAM_STRIDE,NATIVE_ROW_STRIDE,NATIVE_ROW_UNIT_STRIDE,NATIVE_ROW_ATOM_STRIDE,NATIVE_ROW_PAPER_STRIDE,NATIVE_ROW_RESIDUE_STRIDE,NATIVE_ROW_PREFIX_STRIDE].join(','))return;
 const rawOffsets=[storage[h+H.programAt]!,storage[h+H.rowAt]!,storage[h+H.unitAt]!,storage[h+H.atomAt]!,storage[h+H.paperAt]!,storage[h+H.residueAt]!,storage[h+H.prefixAt]!];
 if(rawOffsets.some((offset,index)=>!sectionFits(base,total,offset,counts[index]!,strides[index]!,storage.length)))return;
 let cursor=NATIVE_ROW_PROGRAM_HEADER_WORDS;
 for(let index=0;index<rawOffsets.length;index++){
  if(rawOffsets[index]!==cursor)return;
  cursor+=counts[index]!*strides[index]!;
 }
 if(cursor>total)return;
 let height:PackedNativeRowHeightTable|undefined;
 if(cursor<total){
  if(total-cursor<NATIVE_ROW_HEIGHT_HEADER_WORDS)return;
  const extension=base+cursor;
  if(storage[extension]!==NATIVE_ROW_HEIGHT_MAGIC||storage[extension+1]!==NATIVE_ROW_HEIGHT_VERSION||storage[extension+2]!==NATIVE_ROW_HEIGHT_HEADER_WORDS||storage[extension+3]!==counts[0]||storage[extension+4]!==NATIVE_ROW_HEIGHT_RECORD_STRIDE||storage[extension+5]!==NATIVE_ROW_HEIGHT_UNIT_STRIDE)return;
  const recordsAt=storage[extension+6]!,unitsAt=storage[extension+7]!,extensionTotal=storage[extension+8]!,recordWords=counts[0]!*NATIVE_ROW_HEIGHT_RECORD_STRIDE;
  if(extensionTotal!==total||recordsAt!==cursor+NATIVE_ROW_HEIGHT_HEADER_WORDS||unitsAt!==recordsAt+recordWords||unitsAt>total)return;
  const unitWords=total-unitsAt;
  if(unitWords%NATIVE_ROW_HEIGHT_UNIT_STRIDE!==0)return;
  height=Object.freeze({storage,base,totalWords:total,programCount:counts[0]!,recordStride:NATIVE_ROW_HEIGHT_RECORD_STRIDE,unitStride:NATIVE_ROW_HEIGHT_UNIT_STRIDE,recordsAt,unitsAt});
  for(let entry=0;entry<height.programCount;entry++){
   const at=base+recordsAt+entry*NATIVE_ROW_HEIGHT_RECORD_STRIDE,authored=storage[at]!,minRows=storage[at+1]!,maxRows=storage[at+2]!,sequenceFirst=storage[at+3]!,sequenceCount=storage[at+4]!;
   const expectedSequenceCount=authored>=2?authored-2:0,memberStride=storage[at+8]!,finalAnchor=storage[at+9]!;
   if(authored===0){
    if(minRows!==0||maxRows!==0||sequenceFirst!==0||sequenceCount!==0||memberStride!==0||finalAnchor!==0)return;
    continue;
   }
   const maxInteriorMember=Math.max(0,authored-2),powerOfTwo=memberStride>0&&(memberStride&(memberStride-1))===0;
   if(minRows!==authored||maxRows<minRows||sequenceFirst>unitWords/NATIVE_ROW_HEIGHT_UNIT_STRIDE||sequenceCount>unitWords/NATIVE_ROW_HEIGHT_UNIT_STRIDE-sequenceFirst||sequenceCount!==expectedSequenceCount||!powerOfTwo||finalAnchor<=maxInteriorMember||finalAnchor!==0x3ff||sequenceCount===0&&maxRows!==authored||sequenceCount>0&&(maxRows-authored)%sequenceCount!==0)return;
   const cycles=sequenceCount===0?0:(maxRows-authored)/sequenceCount;
   if(cycles*memberStride+maxInteriorMember>=finalAnchor)return;
  }
 }
 const table:PackedNativeRowProgramTable={storage,base,totalWords:total,programCount:counts[0]!,rowCount:counts[1]!,unitCount:counts[2]!,atomCount:counts[3]!,paperCount:counts[4]!,residueCount:counts[5]!,prefixCount:counts[6]!,offsets:Object.freeze({program:rawOffsets[0]!,row:rawOffsets[1]!,unit:rawOffsets[2]!,atom:rawOffsets[3]!,paper:rawOffsets[4]!,residue:rawOffsets[5]!,prefix:rawOffsets[6]!}),height};
 for(let entry=0;entry<table.programCount;entry++){
  const at=table.base+table.offsets.program+entry*NATIVE_ROW_PROGRAM_STRIDE,rowFirst=storage[at+P.rowFirst]!,rowCount=storage[at+P.rowCount]!,policy=storage[at+P.policy]!;
  if(policy>1||rowFirst>table.rowCount||rowCount>table.rowCount-rowFirst)return;
  for(let rowIndex=0;rowIndex<rowCount;rowIndex++){
   const rowAt=table.base+table.offsets.row+(rowFirst+rowIndex)*NATIVE_ROW_STRIDE,unitFirst=storage[rowAt+R.unitFirst]!,unitCount=storage[rowAt+R.unitCount]!,atomFirst=storage[rowAt+R.atomFirst]!,atomCount=storage[rowAt+R.atomCount]!,paperFirst=storage[rowAt+R.paperFirst]!,paperCount=storage[rowAt+R.paperCount]!,residueFirst=storage[rowAt+R.residueFirst]!,residueCount=storage[rowAt+R.residueCount]!,prefixFirst=storage[rowAt+R.prefixFirst]!,prefixCount=storage[rowAt+R.prefixCount]!,mode=storage[rowAt+R.mode]!&0xff;
   if(mode>NATIVE_ROW_MODE_GAP||storage[rowAt+R.policy]!>1||unitFirst>table.unitCount||unitCount>table.unitCount-unitFirst||atomFirst>table.atomCount||atomCount>table.atomCount-atomFirst||paperFirst>table.paperCount||paperCount>table.paperCount-paperFirst||residueFirst>table.residueCount||residueCount>table.residueCount-residueFirst||prefixFirst>table.prefixCount||prefixCount>table.prefixCount-prefixFirst)return;
  }
 }
 return table;
}

const rowAtFor=(table:PackedNativeRowProgramTable,entry:number,row:number):number|undefined=>{
 if(!Number.isSafeInteger(entry)||!Number.isSafeInteger(row)||entry<0||row<0||entry>=table.programCount)return;
 const program=table.base+table.offsets.program+entry*NATIVE_ROW_PROGRAM_STRIDE,first=table.storage[program+P.rowFirst]!,count=table.storage[program+P.rowCount]!;
 if(row>=count)return;return table.base+table.offsets.row+(first+row)*NATIVE_ROW_STRIDE;
};
/** CPU counterpart of the generated WGSL direct address lookup. */
export function nativeRowAddressCpu(table:PackedNativeRowProgramTable,entry:number,row:number):number|undefined{return rowAtFor(table,entry,row);}
export interface PackedNativeRowHeightLookup {
 readonly sourceRow:number; readonly cycle:number; readonly rowMember:number;
 readonly leftShift:number; readonly rightShift:number; readonly terminal:boolean;
}
/** CPU reader for the optional packed height trailer. */
export function nativeRowHeightCpu(table:PackedNativeRowProgramTable,entry:number,pageRows:number,pageRow:number):PackedNativeRowHeightLookup|undefined{
 const extension=table.height;
 if(!extension||!Number.isSafeInteger(entry)||!Number.isSafeInteger(pageRows)||!Number.isSafeInteger(pageRow)||entry<0||entry>=extension.programCount||pageRows<0||pageRow<0)return;
 const record=table.base+extension.recordsAt+entry*extension.recordStride,s=table.storage,authored=s[record]!,minRows=s[record+1]!,maxRows=s[record+2]!,sequenceFirst=s[record+3]!,sequenceCount=s[record+4]!;
 if(authored===0||pageRows<minRows||pageRows>maxRows||pageRow>=pageRows)return;
 const leftShift=(cycle:number)=>cycle*readI32(s,record+5),rightShift=(cycle:number)=>cycle*readI32(s,record+6),memberStride=s[record+8]!,final=s[record+9]!;
 if(pageRows===authored){const terminal=pageRow===authored-1;return {sourceRow:pageRow,cycle:0,rowMember:terminal?final:pageRow,leftShift:0,rightShift:0,terminal};}
 if(sequenceCount===0)return;
 // Height playback is a whole-programme operation.  Reject a page whose
 // interior contains only part of the authored middle-row sequence before
 // looking up either anchor; otherwise row zero/final would appear valid
 // while the intervening construction was incomplete.
 const extra=(pageRows-2)-sequenceCount;if(extra<0||extra%sequenceCount!==0)return;const cycles=extra/sequenceCount;
 if(pageRow===0)return {sourceRow:0,cycle:0,rowMember:0,leftShift:0,rightShift:0,terminal:false};
 if(pageRow===pageRows-1)return {sourceRow:authored-1,cycle:cycles,rowMember:final,leftShift:leftShift(cycles),rightShift:rightShift(cycles),terminal:true};
 const index=pageRow-1;
 if(index<sequenceCount){const unit=table.base+extension.unitsAt+ (sequenceFirst+index)*extension.unitStride;return {sourceRow:s[unit]!,cycle:0,rowMember:s[unit+1]!,leftShift:0,rightShift:0,terminal:false};}
 const repeated=index-sequenceCount,cycle=1+Math.floor(repeated/sequenceCount),unit=table.base+extension.unitsAt+(sequenceFirst+(repeated%sequenceCount))*extension.unitStride;
 return {sourceRow:s[unit]!,cycle,rowMember:cycle*memberStride+s[unit+1]!,leftShift:leftShift(cycle),rightShift:rightShift(cycle),terminal:false};
}
const growthAt=(table:PackedNativeRowProgramTable,rowAddress:number,extra:number):PackedNativeRowGrowth|undefined=>{
 if(!addressIn(table,table.offsets.row,table.rowCount,NATIVE_ROW_STRIDE,rowAddress)||!Number.isSafeInteger(extra)||extra<0)return;
 const s=table.storage,flags=s[rowAddress+R.mode]!,mode=flags&0xff,max=s[rowAddress+R.maxDelta]!;if(extra>max||(flags&0x200)!==0)return;if(mode===NATIVE_ROW_MODE_BASELINE&&extra!==0)return;
 if(mode!==NATIVE_ROW_MODE_GAP)return {repeatCount:0,repeatWidth:0,paperDelta:0,valid:true};
 const repeatWidth=s[rowAddress+R.repeatUnitWidth]!;
 if(repeatWidth>0){if((s[rowAddress+R.mode]!&0x100)!==0&&extra>0&&extra<repeatWidth)return;const residue=extra%repeatWidth,q=Math.floor(extra/repeatWidth),residueCount=s[rowAddress+R.residueCount]!;if(residue>=residueCount)return;const at=table.base+table.offsets.residue+(s[rowAddress+R.residueFirst]!+residue)*NATIVE_ROW_RESIDUE_STRIDE,skip=s[at]!,paperDelta=s[at+1]!;if(skip===NATIVE_ROW_PROGRAM_NONE||paperDelta===NATIVE_ROW_PROGRAM_NONE||q<skip)return;return {repeatCount:q-skip,repeatWidth,paperDelta,valid:true};}
 const prefixCount=s[rowAddress+R.prefixCount]!,tail=s[rowAddress+R.paperTailStart]!;if(extra<tail){if(extra>=prefixCount)return;const at=table.base+table.offsets.prefix+(s[rowAddress+R.prefixFirst]!+extra)*NATIVE_ROW_PREFIX_STRIDE;if(s[at]===NATIVE_ROW_PROGRAM_NONE||s[at+1]===NATIVE_ROW_PROGRAM_NONE)return;}return {repeatCount:0,repeatWidth:0,paperDelta:extra,valid:true};
};
export function nativeRowGrowthCpu(table:PackedNativeRowProgramTable,rowAddress:number,extra:number):PackedNativeRowGrowth|undefined{return growthAt(table,rowAddress,extra);}

const atomAtFor=(table:PackedNativeRowProgramTable,atomAddress:number):number|undefined=>addressIn(table,table.offsets.atom,table.atomCount,NATIVE_ROW_ATOM_STRIDE,atomAddress)?atomAddress:undefined;
/** Return an atom's x after its precompiled growth rule and whole-word copy. */
export function nativeRowAtomXCpu(table:PackedNativeRowProgramTable,rowAddress:number,atomAddress:number,extra:number,copy=0):number|undefined{
 const growth=growthAt(table,rowAddress,extra),atomAt=atomAtFor(table,atomAddress);if(!growth||atomAt===undefined||!Number.isSafeInteger(copy)||copy<0||copy>growth.repeatCount)return;
 const s=table.storage,member=s[atomAt+A.member]!,mode=s[rowAddress+R.mode]!&0xff;let x=readI32(s,atomAt+A.x);
 const repeated=member===s[rowAddress+R.repeatMember]!;
 if(copy>0){if(!repeated)return;return x+s[rowAddress+R.repeatUnitWidth]!*(s[rowAddress+R.repeatBefore]!?copy-1:copy);}
 if(mode===NATIVE_ROW_MODE_RIGHT_SHIFT)x+=extra;
 else if(mode===NATIVE_ROW_MODE_GAP){
  if(member>=s[rowAddress+R.growthAnchorMember]!)x+=extra;
  else if(member>s[rowAddress+R.repeatMember]!)x+=growth.repeatCount*growth.repeatWidth;
 }
 return x;
}
export function nativeRowAtomMemberCpu(table:PackedNativeRowProgramTable,atomAddress:number,copy=0):number|undefined{
 const at=atomAtFor(table,atomAddress);if(at===undefined||!Number.isSafeInteger(copy)||copy<0||copy>0xffff)return;const member=table.storage[at+A.member]!;return copy===0?member:((member<<16)|(copy&0xffff))>>>0;
}
export function nativeRowAtomRepeatCountCpu(table:PackedNativeRowProgramTable,rowAddress:number,extra:number):number|undefined{return growthAt(table,rowAddress,extra)?.repeatCount;}
export function nativeRowAtomRecordCpu(table:PackedNativeRowProgramTable,atomAddress:number):PackedNativeRowAtom|undefined{
 const at=atomAtFor(table,atomAddress);if(at===undefined)return;const s=table.storage;return {member:s[at+A.member]!,ordinal:s[at+A.ordinal]!,x:readI32(s,at+A.x),advance:s[at+A.advance]!,glyph:s[at+A.glyph]!,wordStart:s[at+A.wordStart]!,wordEnd:s[at+A.wordEnd]!};
}

const paperPlanFor=(table:PackedNativeRowProgramTable,rowAddress:number,extra:number,growth:PackedNativeRowGrowth):[number,number]|undefined=>{
 const s=table.storage,repeatWidth=s[rowAddress+R.repeatUnitWidth]!;
 if(repeatWidth>0){const residue=extra%repeatWidth,at=table.base+table.offsets.residue+(s[rowAddress+R.residueFirst]!+residue)*NATIVE_ROW_RESIDUE_STRIDE;return [s[at+2]!,s[at+3]!];}
 const prefixCount=s[rowAddress+R.prefixCount]!,tail=s[rowAddress+R.paperTailStart]!;if(extra<tail){if(extra>=prefixCount)return;const at=table.base+table.offsets.prefix+(s[rowAddress+R.prefixFirst]!+extra)*NATIVE_ROW_PREFIX_STRIDE;return [s[at]!,s[at+1]!];}
 const gap=s[rowAddress+R.baseGap]!+growth.paperDelta,max=Math.floor(gap/11),residue=gap%5;if(max<residue)return;const full=max-(max-residue)%5,half=(gap-full*11)/5;return [full,half];
};
/** Read static and dynamically resized paper spans without a width-sized table. */
export function nativeRowPaperSpansCpu(table:PackedNativeRowProgramTable,rowAddress:number,extra:number):readonly PackedNativeRowPaper[]|undefined{
 const growth=growthAt(table,rowAddress,extra);if(!growth)return;const s=table.storage,result:PackedNativeRowPaper[]=[];
 const paperFirst=s[rowAddress+R.paperFirst]!,paperCount=s[rowAddress+R.paperCount]!,gapIndex=s[rowAddress+R.growthGapIndex]!,mode=s[rowAddress+R.mode]!&0xff,slot=s[rowAddress+R.growthSlot]!;
 for(let index=0;index<paperCount;index++){
  const at=table.base+table.offsets.paper+(paperFirst+index)*NATIVE_ROW_PAPER_STRIDE,owner=s[at+W.owner]!,indexFromOwner=owner>=0x10000?owner-0x10000:index;let x=readI32(s,at+W.x),width=s[at+W.width]!,full=s[at+W.full]!,half=s[at+W.half]!;
  if(mode===NATIVE_ROW_MODE_GAP&&indexFromOwner===gapIndex){const plan=paperPlanFor(table,rowAddress,extra,growth);if(!plan)return;[full,half]=plan;width=s[rowAddress+R.baseGap]!+growth.paperDelta;x=readI32(s,rowAddress+R.growthAnchorX)+extra-width;}
  else if(mode===NATIVE_ROW_MODE_RIGHT_SHIFT||mode===NATIVE_ROW_MODE_GAP&&indexFromOwner+1>=slot)x+=extra;
  result.push({owner,x,width,full,half});
 }
 const repeatMember=s[rowAddress+R.repeatMember]!,repeatCount=growth.repeatCount,repeatWidth=s[rowAddress+R.repeatUnitWidth]!,repeatGap=s[rowAddress+R.repeatGap]!,repeatFull=s[rowAddress+R.repeatFull]!,repeatHalf=s[rowAddress+R.repeatHalf]!;
 if(repeatMember!==NATIVE_ROW_PROGRAM_NONE&&repeatCount>0)for(let copy=1;copy<=repeatCount;copy++)result.push({owner:(0x30000|(repeatMember<<8)|copy)>>>0,x:readI32(s,rowAddress+R.repeatX)+repeatWidth*(s[rowAddress+R.repeatBefore]!?copy-1:copy)-repeatGap,width:repeatGap,full:repeatFull,half:repeatHalf});
 return Object.freeze(result);
}

/**
 * Direct GPU reader for the optional height trailer. This fragment is kept
 * separate from the v1 row reader so existing consumers can add the height
 * call at their row-mapping owner without changing the width ABI.
 * `nativeRowHeight` returns source row, cycle, left shift and right shift;
 * `nativeRowHeightMember` preserves the first/final anchor names and
 * `nativeRowHeightPivotHalf` exposes the exact half-pixel side seam.
 */
export function nativeRowHeightReaderWgsl(storage='grammar'):string{
 const s=(expression:string)=>`${storage}[${expression}]`;
 return `
const NATIVE_ROW_HEIGHT_NONE:u32=${NATIVE_ROW_PROGRAM_NONE}u;
fn nativeRowHeightSuffix(base:u32)->vec2u{
 let end=nativeRowEnd(base);if(end==0u){return vec2u(NATIVE_ROW_HEIGHT_NONE,NATIVE_ROW_HEIGHT_NONE);}
 let trailer=base+${s('base+16u')}+${s('base+9u')}*${NATIVE_ROW_PREFIX_STRIDE}u;
 // Trailer offsets are relative to the suffix base.  trailer and end are
 // absolute addresses because this reader is used inside a shared bank.
 if(trailer>end||${NATIVE_ROW_HEIGHT_HEADER_WORDS}u>end-trailer){return vec2u(NATIVE_ROW_HEIGHT_NONE,NATIVE_ROW_HEIGHT_NONE);}
 let recordsAt=${s('trailer+6u')};let unitsAt=${s('trailer+7u')};
 if(${s('trailer')}!=${NATIVE_ROW_HEIGHT_MAGIC}u||${s('trailer+1u')}!=${NATIVE_ROW_HEIGHT_VERSION}u||${s('trailer+2u')}!=${NATIVE_ROW_HEIGHT_HEADER_WORDS}u||${s('trailer+3u')}!=${s('base+3u')}||${s('trailer+4u')}!=${NATIVE_ROW_HEIGHT_RECORD_STRIDE}u||${s('trailer+5u')}!=${NATIVE_ROW_HEIGHT_UNIT_STRIDE}u||${s('trailer+8u')}!=${s('base+2u')}||recordsAt!=(trailer-base)+${NATIVE_ROW_HEIGHT_HEADER_WORDS}u||unitsAt!=recordsAt+${s('base+3u')}*${NATIVE_ROW_HEIGHT_RECORD_STRIDE}u||base+unitsAt>end||((end-(base+unitsAt))%${NATIVE_ROW_HEIGHT_UNIT_STRIDE}u)!=0u){return vec2u(NATIVE_ROW_HEIGHT_NONE,NATIVE_ROW_HEIGHT_NONE);}
 return vec2u(end,trailer);
}
fn nativeRowHeightTrailer(base:u32)->u32{
 let suffix=nativeRowHeightSuffix(base);return select(NATIVE_ROW_HEIGHT_NONE,suffix.y,suffix.x!=NATIVE_ROW_HEIGHT_NONE);
}
fn nativeRowHeightDescriptor(base:u32,entry:u32)->vec3u{
 let suffix=nativeRowHeightSuffix(base);if(suffix.x==NATIVE_ROW_HEIGHT_NONE||entry>=${s('suffix.y+3u')}){return vec3u(NATIVE_ROW_HEIGHT_NONE,NATIVE_ROW_HEIGHT_NONE,NATIVE_ROW_HEIGHT_NONE);}
 let end=suffix.x;let trailer=suffix.y;
 let at=base+${s('trailer+6u')}+entry*${NATIVE_ROW_HEIGHT_RECORD_STRIDE}u;
 if(at+${NATIVE_ROW_HEIGHT_RECORD_STRIDE}u>end){return vec3u(NATIVE_ROW_HEIGHT_NONE,NATIVE_ROW_HEIGHT_NONE,NATIVE_ROW_HEIGHT_NONE);}
 let authored=${s('at')};let sequenceFirst=${s('at+3u')};let sequenceCount=${s('at+4u')};let memberStride=${s('at+8u')};let finalAnchor=${s('at+9u')};let expected=select(0u,authored-2u,authored>=2u);let powerOfTwo=memberStride>0u&&((memberStride&(memberStride-1u))==0u);
 let unitWords=(end-(base+${s('trailer+7u')}))/${NATIVE_ROW_HEIGHT_UNIT_STRIDE}u;var valid=sequenceFirst<=unitWords&&sequenceCount<=unitWords-sequenceFirst&&${s('at+1u')}==authored&&${s('at+2u')}>=authored&&sequenceCount==expected&&powerOfTwo&&finalAnchor>max(0u,authored-2u)&&finalAnchor==1023u&&((sequenceCount==0u&&${s('at+2u')}==authored)||(sequenceCount>0u&&(${s('at+2u')}-authored)%sequenceCount==0u&&(${s('at+2u')}-authored)/sequenceCount*memberStride+max(0u,authored-2u)<finalAnchor));
 if(authored==0u){valid=${s('at+1u')}==0u&&${s('at+2u')}==0u&&${s('at+3u')}==0u&&sequenceCount==0u&&memberStride==0u&&finalAnchor==0u;}
 if(!valid){return vec3u(NATIVE_ROW_HEIGHT_NONE,NATIVE_ROW_HEIGHT_NONE,NATIVE_ROW_HEIGHT_NONE);}
 return vec3u(end,trailer,at);
}
fn nativeRowHeightRecord(base:u32,entry:u32)->u32{
 let descriptor=nativeRowHeightDescriptor(base,entry);return select(NATIVE_ROW_HEIGHT_NONE,descriptor.z,descriptor.x!=NATIVE_ROW_HEIGHT_NONE);
}
fn nativeRowHeight(base:u32,entry:u32,pageRows:u32,pageRow:u32)->vec4i{
 let descriptor=nativeRowHeightDescriptor(base,entry);if(descriptor.x==NATIVE_ROW_HEIGHT_NONE){return vec4i(-1,-1,0,0);}let record=descriptor.z;let trailer=descriptor.y;
 let authored=${s('record')};let minimum=${s('record+1u')};let maximum=${s('record+2u')};let sequenceFirst=${s('record+3u')};let sequenceCount=${s('record+4u')};
 if(authored==0u||pageRows<minimum||pageRows>maximum||pageRow>=pageRows){return vec4i(-1,-1,0,0);}
 if(pageRows==authored){return vec4i(i32(pageRow),0,0,0);}
 if(sequenceCount==0u||pageRows<2u){return vec4i(-1,-1,0,0);}
 let leftCycle=bitcast<i32>(${s('record+5u')});let rightCycle=bitcast<i32>(${s('record+6u')});let extra=pageRows-2u-sequenceCount;if(extra%sequenceCount!=0u){return vec4i(-1,-1,0,0);}let cycles=extra/sequenceCount;
 if(pageRow==0u){return vec4i(0,0,0,0);}
 if(pageRow+1u==pageRows){return vec4i(i32(authored-1u),i32(cycles),leftCycle*i32(cycles),rightCycle*i32(cycles));}
 let index=pageRow-1u;var cycle=0u;let sequenceBase=base+${s('trailer+7u')};var unitBase=sequenceBase;
 if(index<sequenceCount){unitBase=sequenceBase+(${s('record+3u')}+index)*${NATIVE_ROW_HEIGHT_UNIT_STRIDE}u;}
 else{let repeated=index-sequenceCount;cycle=1u+repeated/sequenceCount;unitBase=sequenceBase+(${s('record+3u')}+(repeated%sequenceCount))*${NATIVE_ROW_HEIGHT_UNIT_STRIDE}u;}
 let source=${s('unitBase')};return vec4i(i32(source),i32(cycle),leftCycle*i32(cycle),rightCycle*i32(cycle));
}
fn nativeRowHeightMember(base:u32,entry:u32,pageRows:u32,pageRow:u32)->u32{
 let descriptor=nativeRowHeightDescriptor(base,entry);if(descriptor.x==NATIVE_ROW_HEIGHT_NONE){return NATIVE_ROW_HEIGHT_NONE;}let record=descriptor.z;let trailer=descriptor.y;
 let authored=${s('record')};if(pageRows<${s('record+1u')}||pageRows>${s('record+2u')}||pageRow>=pageRows){return NATIVE_ROW_HEIGHT_NONE;}
 let count=${s('record+4u')};if(count==0u&&pageRows!=authored){return NATIVE_ROW_HEIGHT_NONE;}if(pageRows!=authored&&((pageRows-2u-count)%count)!=0u){return NATIVE_ROW_HEIGHT_NONE;}
 if(pageRow==0u){return 0u;}if(pageRow+1u==pageRows){return ${s('record+9u')};}
 if(pageRows==authored){return pageRow;}
 let index=pageRow-1u;let cycle=select(0u,1u+((index-count)/count),index>=count);let ordinal=select(index,index%count,index>=count);let unit=base+${s('trailer+7u')}+(${s('record+3u')}+ordinal)*${NATIVE_ROW_HEIGHT_UNIT_STRIDE}u;return ${s('unit+1u')}+cycle*${s('record+8u')};
}
fn nativeRowHeightPivotHalf(base:u32,entry:u32)->i32{
 let descriptor=nativeRowHeightDescriptor(base,entry);if(descriptor.x==NATIVE_ROW_HEIGHT_NONE){return 0;}let record=descriptor.z;
 return bitcast<i32>(${s('record+7u')});
}
`;
}

/**
 * Generate the direct GPU reader. `base` is the absolute index of this
 * suffix in the existing grammar storage (normally the far-strip base plus
 * the compiler's dynamic vertical-growth length). The generated reader
 * validates every address against the suffix's own totalWords field; it does
 * not walk into the next grammar section.
 */
export function nativeRowProgramWgsl(storage='grammar',includeWhitespace=true,sectionEnd=`min(${storage}[10],arrayLength(&${storage}))`):string{
 const s=(expression:string)=>`${storage}[${expression}]`;
 return `${includeWhitespace?NATIVE_WHITESPACE_WGSL:''}
const NATIVE_ROW_NONE:u32=${NATIVE_ROW_PROGRAM_NONE}u;
fn nativeRowEnd(base:u32)->u32{
 let limit=${sectionEnd};
 if(base==NATIVE_ROW_NONE||base>limit||limit-base<${NATIVE_ROW_PROGRAM_HEADER_WORDS}u){return 0u;}
 if(${s('base')}!=${NATIVE_ROW_PROGRAM_VERSION}u||${s('base+1u')}!=${NATIVE_ROW_PROGRAM_HEADER_WORDS}u){return 0u;}
 let total=${s('base+2u')};if(total<${NATIVE_ROW_PROGRAM_HEADER_WORDS}u||total>limit-base){return 0u;}
 let strides=array<u32,7>(${NATIVE_ROW_PROGRAM_STRIDE}u,${NATIVE_ROW_STRIDE}u,${NATIVE_ROW_UNIT_STRIDE}u,${NATIVE_ROW_ATOM_STRIDE}u,${NATIVE_ROW_PAPER_STRIDE}u,${NATIVE_ROW_RESIDUE_STRIDE}u,${NATIVE_ROW_PREFIX_STRIDE}u);
 var cursor=${NATIVE_ROW_PROGRAM_HEADER_WORDS}u;
 for(var section=0u;section<7u;section++){
  let stride=strides[section];let count=${s('base+3u+section')};
  if(${s('base+17u+section')}!=stride||${s('base+10u+section')}!=cursor||count>(total-cursor)/stride){return 0u;}
  cursor+=count*stride;
 }
 var extensionOk=cursor==total;
 if(!extensionOk&&total-cursor>=${NATIVE_ROW_HEIGHT_HEADER_WORDS}u){
  let extension=base+cursor;let recordAt=${s('extension+6u')};let unitsAt=${s('extension+7u')};
  extensionOk=${s('extension')}==${NATIVE_ROW_HEIGHT_MAGIC}u&&${s('extension+1u')}==${NATIVE_ROW_HEIGHT_VERSION}u&&${s('extension+2u')}==${NATIVE_ROW_HEIGHT_HEADER_WORDS}u&&${s('extension+3u')}==${s('base+3u')}&&${s('extension+4u')}==${NATIVE_ROW_HEIGHT_RECORD_STRIDE}u&&${s('extension+5u')}==${NATIVE_ROW_HEIGHT_UNIT_STRIDE}u&&${s('extension+8u')}==total&&recordAt==cursor+${NATIVE_ROW_HEIGHT_HEADER_WORDS}u&&unitsAt==recordAt+${s('base+3u')}*${NATIVE_ROW_HEIGHT_RECORD_STRIDE}u&&unitsAt<=total&&((total-unitsAt)%${NATIVE_ROW_HEIGHT_UNIT_STRIDE}u)==0u;
 }
 return select(0u,base+total,extensionOk);
}
fn nativeRowProgramAddressChecked(base:u32,entry:u32,end:u32)->u32{
 if(end==0u||entry>=${s('base+3u')}){return NATIVE_ROW_NONE;}
 let at=base+${s('base+10u')}+entry*${NATIVE_ROW_PROGRAM_STRIDE}u;
 return select(NATIVE_ROW_NONE,at,at+${NATIVE_ROW_PROGRAM_STRIDE}u<=end);
}
fn nativeRowProgramAddress(base:u32,entry:u32)->u32{
 let end=nativeRowEnd(base);return nativeRowProgramAddressChecked(base,entry,end);
}
fn nativeRowAddressChecked(base:u32,entry:u32,row:u32,end:u32)->u32{
 let program=nativeRowProgramAddressChecked(base,entry,end);if(program==NATIVE_ROW_NONE||row>=${s('program+1u')}){return NATIVE_ROW_NONE;}
 let first=${s('program')};let count=${s('base+4u')};if(first>=count||row>=count-first){return NATIVE_ROW_NONE;}
 let at=base+${s('base+11u')}+(${s('program+0u')}+row)*${NATIVE_ROW_STRIDE}u;
 return select(NATIVE_ROW_NONE,at,at+${NATIVE_ROW_STRIDE}u<=end);
}
fn nativeRowAddress(base:u32,entry:u32,row:u32)->u32{
 let end=nativeRowEnd(base);return nativeRowAddressChecked(base,entry,row,end);
}
fn nativeRowGrowth(base:u32,rowAddr:u32,extra:u32)->vec4u{
 if(rowAddr==NATIVE_ROW_NONE){return vec4u(0u,0u,0u,NATIVE_ROW_NONE);}
 let flags=${s('rowAddr+12u')};let mode=flags&255u;let maxDelta=${s('rowAddr+13u')};if(extra>maxDelta||(flags&512u)!=0u||mode>${NATIVE_ROW_MODE_GAP}u||(mode==${NATIVE_ROW_MODE_BASELINE}u&&extra!=0u)){return vec4u(0u,0u,0u,NATIVE_ROW_NONE);}
 if(mode!=${NATIVE_ROW_MODE_GAP}u){return vec4u(0u,0u,0u,1u);}
 let repeatWidth=${s('rowAddr+21u')};
 if(repeatWidth>0u){if((${s('rowAddr+12u')}&256u)!=0u&&extra>0u&&extra<repeatWidth){return vec4u(0u,0u,0u,NATIVE_ROW_NONE);}let residue=extra%repeatWidth;let q=extra/repeatWidth;let count=${s('rowAddr+25u')};if(residue>=count){return vec4u(0u,0u,0u,NATIVE_ROW_NONE);}let at=base+${s('base+15u')}+(${s('rowAddr+24u')}+residue)*${NATIVE_ROW_RESIDUE_STRIDE}u;let skip=${s('at+0u')};let paperDelta=${s('at+1u')};if(skip==NATIVE_ROW_NONE||paperDelta==NATIVE_ROW_NONE||q<skip){return vec4u(0u,0u,0u,NATIVE_ROW_NONE);}return vec4u(q-skip,repeatWidth,paperDelta,1u);}
 let tail=${s('rowAddr+23u')};let prefixCount=${s('rowAddr+27u')};if(extra<tail){if(extra>=prefixCount){return vec4u(0u,0u,0u,NATIVE_ROW_NONE);}let at=base+${s('base+16u')}+(${s('rowAddr+26u')}+extra)*${NATIVE_ROW_PREFIX_STRIDE}u; if(${s('at+0u')}==NATIVE_ROW_NONE||${s('at+1u')}==NATIVE_ROW_NONE){return vec4u(0u,0u,0u,NATIVE_ROW_NONE);}}return vec4u(0u,0u,extra,1u);
}
fn nativeRowPaperPlan(base:u32,rowAddr:u32,extra:u32,growth:vec4u)->vec2i{
 let repeatWidth=${s('rowAddr+21u')};if(repeatWidth>0u){let residue=extra%repeatWidth;let at=base+${s('base+15u')}+(${s('rowAddr+24u')}+residue)*${NATIVE_ROW_RESIDUE_STRIDE}u;return vec2i(bitcast<i32>(${s('at+2u')}),bitcast<i32>(${s('at+3u')}));}
 let tail=${s('rowAddr+23u')};let prefixCount=${s('rowAddr+27u')};if(extra<tail&&extra<prefixCount){let at=base+${s('base+16u')}+(${s('rowAddr+26u')}+extra)*${NATIVE_ROW_PREFIX_STRIDE}u;return vec2i(bitcast<i32>(${s('at+0u')}),bitcast<i32>(${s('at+1u')}));}
 let gap=i32(${s('rowAddr+15u')}+growth.z);let maximum=gap/11;let residue=gap%5;if(${s('rowAddr+31u')}==1u){if(maximum<residue){return vec2<i32>(-1);}let full=maximum-(maximum-residue)%5;return vec2<i32>(i32(full),(gap-full*11)/5);}return nativeSpacePlan(gap,0,false,false);
}
fn nativeRowAtomX(base:u32,rowAddr:u32,atomAddr:u32,extra:u32,copy:u32)->i32{
 let growth=nativeRowGrowth(base,rowAddr,extra);if(growth.w==NATIVE_ROW_NONE||copy>growth.x){return -1;}
 let member=${s('atomAddr+0u')};var x=bitcast<i32>(${s('atomAddr+2u')});
 if(copy>0u){if(member!=${s('rowAddr+18u')}){return -1;}let copyIndex=select(copy,copy-1u,${s('rowAddr+19u')}!=0u);return x+i32(${s('rowAddr+21u')}*copyIndex);}
 let mode=${s('rowAddr+12u')}&255u;
 if(mode==${NATIVE_ROW_MODE_RIGHT_SHIFT}u){x+=i32(extra);}
 else if(mode==${NATIVE_ROW_MODE_GAP}u){
  if(member>=${s('rowAddr+17u')}){x+=i32(extra);}
  else if(member>${s('rowAddr+18u')}){x+=i32(growth.x*growth.y);}
 }
 return x;
}
fn nativeRowAtomMember(atomAddr:u32,copy:u32)->u32{let member=${s('atomAddr+0u')};return select(member,(member<<16u)|(copy&65535u),copy>0u);}
fn nativeRowAtomRepeatCount(base:u32,rowAddr:u32,extra:u32)->u32{let growth=nativeRowGrowth(base,rowAddr,extra);return select(NATIVE_ROW_NONE,growth.x,growth.w!=NATIVE_ROW_NONE);}
fn nativeRowAtomWordStart(atomAddr:u32)->u32{return ${s('atomAddr+5u')};}
fn nativeRowAtomWordEnd(atomAddr:u32)->u32{return ${s('atomAddr+6u')};}
fn nativeRowPaperOwner(paperAddr:u32)->u32{return ${s('paperAddr+0u')};}
fn nativeRowPaperSpan(base:u32,rowAddr:u32,paperAddr:u32,extra:u32)->vec4i{
 let growth=nativeRowGrowth(base,rowAddr,extra);if(growth.w==NATIVE_ROW_NONE){return vec4i(0,-1,0,0);}
 let owner=${s('paperAddr+0u')};var x=bitcast<i32>(${s('paperAddr+1u')});var width=i32(${s('paperAddr+2u')});var full=${s('paperAddr+3u')};var half=${s('paperAddr+4u')};
 let gapIndex=select(0u,owner-65536u,owner>=65536u);if((${s('rowAddr+12u')}&255u)==${NATIVE_ROW_MODE_GAP}u&&gapIndex==${s('rowAddr+14u')}){let plan=nativeRowPaperPlan(base,rowAddr,extra,growth);if(plan.x<0){return vec4i(0,-1,0,0);}width=i32(${s('rowAddr+15u')}+growth.z);x=bitcast<i32>(${s('rowAddr+16u')})+i32(extra)-width;full=u32(plan.x);half=u32(plan.y);}else if((${s('rowAddr+12u')}&255u)==${NATIVE_ROW_MODE_RIGHT_SHIFT}u||((${s('rowAddr+12u')}&255u)==${NATIVE_ROW_MODE_GAP}u&&gapIndex+1u>=${s('rowAddr+28u')})){x+=i32(extra);}return vec4i(x,width,i32(full),i32(half));
}
fn nativeRowRepeatPaperSpan(base:u32,rowAddr:u32,extra:u32,copy:u32)->vec4i{
 let growth=nativeRowGrowth(base,rowAddr,extra);if(growth.w==NATIVE_ROW_NONE||copy==0u||copy>growth.x){return vec4i(0,-1,0,0);}let width=i32(${s('rowAddr+22u')});let copyIndex=select(copy,copy-1u,${s('rowAddr+19u')}!=0u);let shift=i32(${s('rowAddr+21u')}*copyIndex);let copyX=bitcast<i32>(${s('rowAddr+20u')})+shift;return vec4i(copyX-width,width,i32(${s('rowAddr+29u')}),i32(${s('rowAddr+30u')}));
}
`;
}

