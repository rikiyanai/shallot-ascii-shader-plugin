import type {
 SourceWholeConstructionTemplate,
 SourceWholeOpeningTemplate,
 SourceWholeCompound,
 SourceWholePartRole,
 SourceWholePartTemplate,
} from './source-lod.ts';

/** Compact, bounded source-owned parent packet. Relation boundaries carry
 * source geometry and pose; opaque relation IDs remain non-renderable. */
export const SOURCE_WHOLE_CONSTRUCTION_MAGIC=0x57435450;
export const SOURCE_WHOLE_CONSTRUCTION_VERSION=4;
const LEGACY_SOURCE_WHOLE_CONSTRUCTION_VERSION=1;
const PREVIOUS_SOURCE_WHOLE_CONSTRUCTION_VERSION=2;
const COMPOUND_SOURCE_WHOLE_CONSTRUCTION_VERSION=3;
const NONE=0xffffffff;
const MAX_LIST=1<<20;
const MAX_GEOMETRY_POINTS=1<<22;
const ROLE_TO_WORD:Readonly<Record<SourceWholePartRole,number>>={cap:0,turn:1,flank:2,return:3};
const WORD_TO_ROLE:readonly SourceWholePartRole[]=['cap','turn','flank','return'];
const COMPOUND_TO_WORD:Readonly<Record<SourceWholeCompound['kind'],number>>={'planar-openings-contact':1,'planar-contact':2};
const WORD_TO_COMPOUND:readonly (SourceWholeCompound['kind']|undefined)[]=[undefined,'planar-openings-contact','planar-contact'];

const assertU32=(value:number,label:string):number=>{
 if(!Number.isSafeInteger(value)||value<0||value>0xffffffff)throw new RangeError(`Invalid whole construction ${label}`);
 return value;
};
const assertCount=(value:number,label:string,max=MAX_LIST):number=>{
 if(!Number.isSafeInteger(value)||value<0||value>max)throw new RangeError(`Invalid whole construction ${label} count`);
 return value;
};
const assertPoint=(point:readonly number[],label:string):void=>{
 if(point.length!==3||!point.every(Number.isFinite))throw new RangeError(`Invalid whole construction ${label} point`);
};
const roleWord=(role:SourceWholePartRole):number=>ROLE_TO_WORD[role];

class WordWriter {
 private readonly values:number[]=[];
 word(value:number):void{this.values.push(assertU32(value,'word'));}
 words(values:readonly number[]):void{for(const value of values)this.word(value);}
 float(value:number,label:string):void{
  assertPoint([value,0,0],label);
  const f=new Float32Array(1);const u=new Uint32Array(f.buffer);f[0]=value;this.word(u[0]!);
 }
 finish():Uint32Array{return Uint32Array.from(this.values);}
}
class WordReader {
 private at=0;
 constructor(private readonly values:Uint32Array){}
 word(label:string):number{
  if(this.at>=this.values.length)throw Error(`Truncated whole construction ${label}`);
  return this.values[this.at++]!;
 }
 count(label:string,max=MAX_LIST):number{return assertCount(this.word(label),label,max);}
 float(label:string):number{
  const f=new Float32Array(1);const u=new Uint32Array(f.buffer);u[0]=this.word(label);return f[0]!;
 }
 done():boolean{return this.at===this.values.length;}
}

const encodeOpening=(writer:WordWriter,value:SourceWholeOpeningTemplate,label:string):void=>{
 writer.word(assertU32(value.feature,`${label}.feature`));
 const ports=assertCount(value.ports.length,`${label}.ports`),contributors=assertCount(value.sourceContributors.length,`${label}.contributors`);
 writer.word(ports);writer.word(contributors);writer.words(value.ports.map(port=>assertU32(port,`${label}.port`)));
 writer.words(value.sourceContributors.map(source=>assertU32(source,`${label}.sourceContributor`)));
 const hasGeometry=value.geometry!==undefined,hasClosed=value.closed!==undefined,hasEntry=value.entryRef!==undefined,
  hasContext=value.context!==undefined,hasRole=value.role!==undefined,hasPose=value.worldFromSource!==undefined;
 writer.word(Number(hasGeometry)|(Number(hasClosed)<<1)|(Number(value.closed===true)<<2)|(Number(hasEntry)<<3)|(Number(hasContext)<<4)|(Number(hasRole)<<5)|(Number(hasPose)<<6));
 if(hasGeometry){
  const geometry=value.geometry!;assertCount(geometry.length,`${label}.geometry`,MAX_GEOMETRY_POINTS);writer.word(geometry.length);
  for(let index=0;index<geometry.length;index++){
   const point=geometry[index]!,itemLabel=label+'.geometry['+index+']';assertPoint(point,itemLabel);
   writer.float(point[0]!,itemLabel+'.x');writer.float(point[1]!,itemLabel+'.y');writer.float(point[2]!,itemLabel+'.z');
  }
 }
 if(hasEntry)writer.word(assertU32(value.entryRef!,`${label}.entryRef`));
 if(hasContext)writer.word(assertU32(value.context!,`${label}.context`));
 if(hasRole){const role=roleWord(value.role!);if(role===undefined)throw new RangeError(`Invalid whole construction ${label}.role`);writer.word(role);}
 if(hasPose){
  const pose=value.worldFromSource!;
  if(pose.length!==16||Array.from(pose).some(item=>!Number.isFinite(item)))throw new RangeError(`Invalid whole construction ${label}.pose`);
  for(let index=0;index<16;index++)writer.float(pose[index]!,`${label}.pose[${index}]`);
 }
};
const encodeOpeningList=(writer:WordWriter,values:readonly SourceWholeOpeningTemplate[]|undefined,label:string):void=>{
 if(values===undefined){writer.word(NONE);return;}
 writer.word(assertCount(values.length,label));
 for(let index=0;index<values.length;index++)encodeOpening(writer,values[index]!,`${label}[${index}]`);
};

const decodeOpening=(reader:WordReader,label:string,version:number):SourceWholeOpeningTemplate=>{
 const feature=reader.word(`${label}.feature`),portsCount=reader.count(`${label}.ports`),contributorsCount=reader.count(`${label}.contributors`);
 const ports=Array.from({length:portsCount},()=>reader.word(`${label}.port`));
 const sourceContributors=Array.from({length:contributorsCount},()=>reader.word(`${label}.sourceContributor`));
 if(version===LEGACY_SOURCE_WHOLE_CONSTRUCTION_VERSION)return {feature,ports,sourceContributors};
 const flags=reader.word(`${label}.flags`);if(flags&0xffffff80)throw Error(`Invalid whole construction ${label}.flags`);
 let geometry:readonly [number,number,number][]|undefined;
 if(flags&1){
  const count=reader.count(`${label}.geometry`,MAX_GEOMETRY_POINTS),points:[number,number,number][]=[];
  for(let index=0;index<count;index++){
   const itemLabel=`${label}.geometry[${index}]`,point:[number,number,number]=[reader.float(`${itemLabel}.x`),reader.float(`${itemLabel}.y`),reader.float(`${itemLabel}.z`)];
   if(!point.every(Number.isFinite))throw Error(`Invalid whole construction ${itemLabel}`);points.push(point);
  }
  geometry=points;
 }
 const entryRef=flags&8?reader.word(`${label}.entryRef`):undefined,context=flags&16?reader.word(`${label}.context`):undefined,role=flags&32?WORD_TO_ROLE[reader.word(`${label}.role`)!]:undefined;
 if((flags&32)&&!role)throw Error(`Invalid whole construction ${label}.role`);
 let worldFromSource:ArrayLike<number>|undefined;
 if(flags&64){const pose=new Float32Array(16);for(let index=0;index<16;index++)pose[index]=reader.float(`${label}.pose[${index}]`);worldFromSource=pose;}
 return {feature,ports,sourceContributors,...(geometry?{geometry}:{}),...(flags&2?{closed:(flags&4)!==0}:{}),...(entryRef!==undefined?{entryRef}:{}),...(context!==undefined?{context}:{}),...(role?{role}:{}),...(worldFromSource?{worldFromSource}: {})};
};
const decodeOpeningList=(reader:WordReader,label:string,version:number):SourceWholeOpeningTemplate[]|undefined=>{
 const count=reader.word(`${label}.count`);if(count===NONE)return;
 assertCount(count,label);return Array.from({length:count},(_,index)=>decodeOpening(reader,`${label}[${index}]`,version));
};

export function encodeSourceWholeConstructionTemplate(source:SourceWholeConstructionTemplate):Uint32Array{
 const writer=new WordWriter();
 writer.word(SOURCE_WHOLE_CONSTRUCTION_MAGIC);writer.word(SOURCE_WHOLE_CONSTRUCTION_VERSION);writer.word(assertU32(source.revision,'revision'));
 writer.word(assertCount(source.parts.length,'parts'));
 for(let index=0;index<source.parts.length;index++){
  const part=source.parts[index]!,label=`parts[${index}]`,role=ROLE_TO_WORD[part.role];
  if(role===undefined)throw new RangeError(`Invalid whole construction ${label}.role`);
  writer.word(assertU32(part.feature,`${label}.feature`));writer.word(assertU32(part.entryRef,`${label}.entryRef`));writer.word(assertU32(part.context,`${label}.context`));writer.word(role);
  writer.word(assertU32(part.ports[0]!,`${label}.port[0]`));writer.word(assertU32(part.ports[1]!,`${label}.port[1]`));
  const hasGeometry=part.geometry!==undefined,hasClosed=part.closed!==undefined,hasPaper=part.paper!==undefined;
  writer.word(Number(hasGeometry)|(Number(hasClosed)<<1)|(Number(part.closed===true)<<2)|(Number(hasPaper)<<3));
  if(hasGeometry){
   const geometry=part.geometry!;assertCount(geometry.length,label+'.geometry',MAX_GEOMETRY_POINTS);writer.word(geometry.length);
   for(let point=0;point<geometry.length;point++){const item=geometry[point]!,itemLabel=label+'.geometry['+point+']';assertPoint(item,itemLabel);writer.float(item[0]!,itemLabel+'.x');writer.float(item[1]!,itemLabel+'.y');writer.float(item[2]!,itemLabel+'.z');}
  }
  writer.word(assertCount(part.sourceContributors.length,label+'.contributors'));
  writer.words(part.sourceContributors.map(value=>assertU32(value,`${label}.sourceContributor`)));
  if(hasPaper)encodeOpeningList(writer,part.paper,label+'.paper');
 }
 encodeOpeningList(writer,source.openings,'openings');encodeOpeningList(writer,source.contacts,'contacts');encodeOpeningList(writer,source.protectedVoids,'protectedVoids');encodeOpeningList(writer,source.paper,'paper');
 writer.word(source.compound?1:0);
 if(source.compound){
  const kind=COMPOUND_TO_WORD[source.compound.kind];if(kind===undefined)throw new RangeError('Invalid whole construction compound kind');
  writer.word(kind);writer.word(assertCount(source.compound.outerFeatures.length,'compound.outerFeatures'));writer.words(source.compound.outerFeatures.map(value=>assertU32(value,'compound.outerFeature')));
  writer.word(assertCount(source.compound.openingFeatures.length,'compound.openingFeatures'));writer.words(source.compound.openingFeatures.map(value=>assertU32(value,'compound.openingFeature')));
  writer.word(assertCount(source.compound.contactFeatures.length,'compound.contactFeatures'));writer.words(source.compound.contactFeatures.map(value=>assertU32(value,'compound.contactFeature')));
  writer.word(assertCount(source.compound.anchorVertices.length,'compound.anchorVertices'));writer.words(source.compound.anchorVertices.map(value=>assertU32(value,'compound.anchorVertex')));
 }
 writer.word(Number(source.principalComplete===true));
 return writer.finish();
}

export function decodeSourceWholeConstructionTemplate(values:Uint32Array):SourceWholeConstructionTemplate{
 const reader=new WordReader(values);
 if(reader.word('magic')!==SOURCE_WHOLE_CONSTRUCTION_MAGIC)throw Error('Unsupported whole construction packet');
 const version=reader.word('version');if(version!==SOURCE_WHOLE_CONSTRUCTION_VERSION&&version!==COMPOUND_SOURCE_WHOLE_CONSTRUCTION_VERSION&&version!==PREVIOUS_SOURCE_WHOLE_CONSTRUCTION_VERSION&&version!==LEGACY_SOURCE_WHOLE_CONSTRUCTION_VERSION)throw Error('Unsupported whole construction packet');
 const revision=reader.word('revision'),partsCount=reader.count('parts');
 const parts:SourceWholePartTemplate[]=[];
 for(let index=0;index<partsCount;index++){
  const label=`parts[${index}]`,feature=reader.word(`${label}.feature`),entryRef=reader.word(`${label}.entryRef`),context=reader.word(`${label}.context`),roleWordValue=reader.word(`${label}.role`),role=WORD_TO_ROLE[roleWordValue];
  if(!role)throw Error(`Invalid whole construction ${label}.role`);
  const ports:[number,number]=[reader.word(`${label}.port[0]`),reader.word(`${label}.port[1]`)],flags=reader.word(`${label}.flags`);
  if(flags&0xfffffff0)throw Error(`Invalid whole construction ${label}.flags`);
  let geometry:readonly [number,number,number][]|undefined;
  if(flags&1){
   const count=reader.count(`${label}.geometry`,MAX_GEOMETRY_POINTS),points:[number,number,number][]=[];
   for(let point=0;point<count;point++){const item:[number,number,number]=[reader.float(`${label}.geometry[${point}].x`),reader.float(`${label}.geometry[${point}].y`),reader.float(`${label}.geometry[${point}].z`)];if(!item.every(Number.isFinite))throw Error(`Invalid whole construction ${label}.geometry`);points.push(item);}geometry=points;
  }
  const contributorCount=reader.count(`${label}.contributors`),sourceContributors=Array.from({length:contributorCount},()=>reader.word(`${label}.sourceContributor`)),paper=flags&8?decodeOpeningList(reader,`${label}.paper`,version):undefined;
  if((flags&8)&&paper===undefined)throw Error(`Invalid whole construction ${label}.paper`);
  parts.push({feature,entryRef,context,role,ports,sourceContributors,...(geometry!==undefined?{geometry}:{}),...(flags&2?{closed:(flags&4)!==0}:{}),...(paper!==undefined?{paper}: {})});
 }
 const openings=decodeOpeningList(reader,'openings',version),contacts=decodeOpeningList(reader,'contacts',version),protectedVoids=decodeOpeningList(reader,'protectedVoids',version),paper=decodeOpeningList(reader,'paper',version);
 if(!openings||!contacts||!protectedVoids)throw Error('Invalid whole construction required list');
 let compound:SourceWholeCompound|undefined;
 if(version>=COMPOUND_SOURCE_WHOLE_CONSTRUCTION_VERSION){
  const present=reader.word('compound.present');if(present>1)throw Error('Invalid whole construction compound marker');
  if(present){
   const kind=WORD_TO_COMPOUND[reader.word('compound.kind')!];if(!kind)throw Error('Invalid whole construction compound kind');
   const readList=(label:string)=>{const count=reader.count(label),values:number[]=[];for(let index=0;index<count;index++)values.push(reader.word(`${label}[${index}]`));return values;};
   compound={kind,outerFeatures:readList('compound.outerFeatures'),openingFeatures:readList('compound.openingFeatures'),contactFeatures:readList('compound.contactFeatures'),anchorVertices:readList('compound.anchorVertices')};
  }
 }
 const principal=version>=4?reader.word('principal.complete'):0;
 if(principal>1)throw Error('Invalid whole construction principal coverage marker');
 if(!reader.done())throw Error('Trailing whole construction words');
 return {revision,...(principal?{principalComplete:true as const}:{}),parts,openings,contacts,protectedVoids,...(paper!==undefined?{paper}: {}),...(compound?{compound}: {})};
}
