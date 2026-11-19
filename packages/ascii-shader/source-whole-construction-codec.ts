import type {
 SourceWholeConstructionTemplate,
 SourceWholeOpeningTemplate,
 SourceWholePartRole,
 SourceWholePartTemplate,
} from './source-lod.ts';

/**
 * Compact source-revision metadata for the baked far/parent consumer.
 *
 * This is deliberately a flat bounded packet.  It carries the source-owned
 * construction vocabulary, not a halfedge graph, mesh incidence table,
 * projected vertices, or camera state.  The packet is optional in the outer
 * semantic residual; an absent section therefore retains the incumbent.
 */
export const SOURCE_WHOLE_CONSTRUCTION_MAGIC=0x57435450;
export const SOURCE_WHOLE_CONSTRUCTION_VERSION=1;
const NONE=0xffffffff;
const MAX_LIST=1<<20;
const MAX_GEOMETRY_POINTS=1<<22;
const ROLE_TO_WORD:Readonly<Record<SourceWholePartRole,number>>={cap:0,turn:1,flank:2,return:3};
const WORD_TO_ROLE:readonly SourceWholePartRole[]=['cap','turn','flank','return'];

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
 get offset():number{return this.at;}
 get remaining():number{return this.values.length-this.at;}
 word(label:string):number{
  if(this.at>=this.values.length)throw Error(`Truncated whole construction ${label}`);
  return this.values[this.at++]!;
 }
 count(label:string,max=MAX_LIST):number{
  const value=this.word(label);return assertCount(value,label,max);
 }
 float(label:string):number{
  const f=new Float32Array(1);const u=new Uint32Array(f.buffer);u[0]=this.word(label);return f[0]!;
 }
 done():boolean{return this.at===this.values.length;}
}

const encodeOpening=(writer:WordWriter,value:SourceWholeOpeningTemplate,label:string):void=>{
 writer.word(assertU32(value.feature,`${label}.feature`));
 const ports=assertCount(value.ports.length,`${label}.ports`),contributors=assertCount(value.sourceContributors.length,`${label}.contributors`);
 writer.word(ports);writer.word(contributors);writer.words(value.ports.map((port)=>assertU32(port,`${label}.port`)));
 writer.words(value.sourceContributors.map((source)=>assertU32(source,`${label}.sourceContributor`)));
};

const encodeOpeningList=(writer:WordWriter,values:readonly SourceWholeOpeningTemplate[]|undefined,label:string):void=>{
 if(values===undefined){writer.word(NONE);return;}
 writer.word(assertCount(values.length,label));
 for(let index=0;index<values.length;index++)encodeOpening(writer,values[index]!,`${label}[${index}]`);
};

const decodeOpening=(reader:WordReader,label:string):SourceWholeOpeningTemplate=>{
 const feature=reader.word(`${label}.feature`),portsCount=reader.count(`${label}.ports`),contributorsCount=reader.count(`${label}.contributors`);
 const ports=Array.from({length:portsCount},()=>reader.word(`${label}.port`));
 const sourceContributors=Array.from({length:contributorsCount},()=>reader.word(`${label}.sourceContributor`));
 return {feature,ports,sourceContributors};
};

const decodeOpeningList=(reader:WordReader,label:string):SourceWholeOpeningTemplate[]|undefined=>{
 const count=reader.word(`${label}.count`);if(count===NONE)return;
 assertCount(count,label);
 return Array.from({length:count},(_,index)=>decodeOpening(reader,`${label}[${index}]`));
};

/** Encode one complete source-owned parent template. */
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
   for(let point=0;point<geometry.length;point++){const value=geometry[point]!;assertPoint(value,`${label}.geometry[${point}]`);writer.float(value[0]!,`${label}.geometry[${point}].x`);writer.float(value[1]!,`${label}.geometry[${point}].y`);writer.float(value[2]!,`${label}.geometry[${point}].z`);}
  }
  writer.word(assertCount(part.sourceContributors.length,label+'.contributors'));
  writer.words(part.sourceContributors.map((value)=>assertU32(value,`${label}.sourceContributor`)));
  if(hasPaper)encodeOpeningList(writer,part.paper,label+'.paper');
 }
 encodeOpeningList(writer,source.openings,'openings');encodeOpeningList(writer,source.contacts,'contacts');encodeOpeningList(writer,source.protectedVoids,'protectedVoids');encodeOpeningList(writer,source.paper,'paper');
 return writer.finish();
}

/** Decode and validate one complete source-owned parent template. */
export function decodeSourceWholeConstructionTemplate(values:Uint32Array):SourceWholeConstructionTemplate{
 const reader=new WordReader(values);
 if(reader.word('magic')!==SOURCE_WHOLE_CONSTRUCTION_MAGIC||reader.word('version')!==SOURCE_WHOLE_CONSTRUCTION_VERSION)throw Error('Unsupported whole construction packet');
 const revision=reader.word('revision'),partsCount=reader.count('parts');
 const parts:SourceWholePartTemplate[]=[];
 for(let index=0;index<partsCount;index++){
  const label=`parts[${index}]`,feature=reader.word(`${label}.feature`),entryRef=reader.word(`${label}.entryRef`),context=reader.word(`${label}.context`),roleWord=reader.word(`${label}.role`);
  const role=WORD_TO_ROLE[roleWord];if(!role)throw Error(`Invalid whole construction ${label}.role`);
  const ports:[number,number]=[reader.word(`${label}.port[0]`),reader.word(`${label}.port[1]`)],flags=reader.word(`${label}.flags`);
  if(flags&0xfffffff0)throw Error(`Invalid whole construction ${label}.flags`);
  let geometry:readonly [number,number,number][]|undefined;
  if(flags&1){const geometryCount=reader.count(`${label}.geometry`,MAX_GEOMETRY_POINTS),points:[number,number,number][]=[];for(let point=0;point<geometryCount;point++){const value:[number,number,number]=[reader.float(`${label}.geometry[${point}].x`),reader.float(`${label}.geometry[${point}].y`),reader.float(`${label}.geometry[${point}].z`)];if(!value.every(Number.isFinite))throw Error(`Invalid whole construction ${label}.geometry`);points.push(value);}geometry=points;}
  const contributorCount=reader.count(`${label}.contributors`),sourceContributors=Array.from({length:contributorCount},()=>reader.word(`${label}.sourceContributor`));
  const paper=flags&8?decodeOpeningList(reader,`${label}.paper`):undefined;
  if((flags&8)&&paper===undefined)throw Error(`Invalid whole construction ${label}.paper`);
  parts.push({feature,entryRef,context,role,ports,sourceContributors,...(geometry!==undefined?{geometry}:{}),...(flags&2?{closed:(flags&4)!==0}:{}),...(paper!==undefined?{paper}: {})});
 }
 const openings=decodeOpeningList(reader,'openings'),contacts=decodeOpeningList(reader,'contacts'),protectedVoids=decodeOpeningList(reader,'protectedVoids'),paper=decodeOpeningList(reader,'paper');
 if(!openings||!contacts||!protectedVoids)throw Error('Invalid whole construction required list');
 if(!reader.done())throw Error('Trailing whole construction words');
 return {revision,parts,openings,contacts,protectedVoids,...(paper!==undefined?{paper}: {})};
}
