import {isolateComputeEntry} from './gpu-compute-program.ts';

export const PLAYBACK_FAMILIES=['path','planar','round'] as const;
export type PlaybackFamily=typeof PLAYBACK_FAMILIES[number];
/** Authored paired paths use the existing PATH queue, but not its generic
 * playback program. Keeping this key separate closes the pair callgraph
 * without adding a queue, binding or publication record. */
export const PAIR_PLAYBACK='pair' as const;
/** Hard/anchor paths also use the existing PATH queue. This is a prepared
 * pipeline key only: the active queue, indirect argument record and
 * publication ABI remain unchanged. */
export const PATH_HARD_PLAYBACK='pathHard' as const;
/** Programmed soft paths share the PATH queue but close over the canonical
 * source-part producer instead of the legacy row-pair producer. */
export const PATH_PROGRAM_PLAYBACK='pathProgram' as const;
export type PathPlaybackClass='soft'|'softProgram'|'hard'|'anchor'|'pair'|'boundary';

/**
 * Prepared canonical native-row lookup.  The row bitset is deliberately
 * separate from the edge payload: a non-monotone source path can touch the
 * same 32-row block through several different edges, while playback still
 * needs one stable edge ordinal per touched row.  Prefixes contain the number
 * of set rows before each block, so a lookup is one block read, one bit count,
 * and one payload read; it never walks source intervals at draw time.
 */
export const SOURCE_PART_ROW_TABLE_MAGIC=0x53505254;
export const SOURCE_PART_ROW_TABLE_VERSION=1;
export const SOURCE_PART_ROW_TABLE_HEADER_WORDS=12;
export const SOURCE_PART_ROW_TABLE_GROUP_WORDS=8;
export interface SourcePartRowTableLayout {
 readonly runCount:number;
 readonly groupCount:number;
 readonly rowCount:number;
 readonly blocks:number;
 readonly touchedRows:number;
 readonly words:number;
 readonly bytes:number;
 readonly mapBase:number;
 readonly groupBase:number;
 readonly bitsetBase:number;
 readonly prefixBase:number;
 readonly payloadBase:number;
}
export function sourcePartRowTableLayout(runCount:number,groupCount:number,rowCount:number,touchedRows:number):SourcePartRowTableLayout{
 for(const value of [runCount,groupCount,rowCount,touchedRows])if(!Number.isSafeInteger(value)||value<0)throw Error('Invalid source-part row table extent');
 const blocks=Math.ceil(rowCount/32),mapBase=SOURCE_PART_ROW_TABLE_HEADER_WORDS,groupBase=mapBase+runCount;
 const bitsetBase=groupBase+groupCount*SOURCE_PART_ROW_TABLE_GROUP_WORDS;
 const prefixBase=bitsetBase+groupCount*blocks,payloadBase=prefixBase+groupCount*blocks;
 const words=payloadBase+touchedRows;
 return {runCount,groupCount,rowCount,blocks,touchedRows,words,bytes:words*4,mapBase,groupBase,bitsetBase,prefixBase,payloadBase};
}
export interface SourcePartRowTableData {readonly layout:SourcePartRowTableLayout;readonly words:Uint32Array}
/** CPU reference for the packed producer/consumer ABI.  `group` is the
 * compact group ordinal, and `row` is the native page row. */
export function sourcePartRowLookup(table:SourcePartRowTableData,group:number,row:number):number|undefined{
 const {layout,words}=table;if(group<0||group>=layout.groupCount||row<0||row>=layout.rowCount)return undefined;
 const descriptor=layout.groupBase+group*SOURCE_PART_ROW_TABLE_GROUP_WORDS;
 if(words[descriptor]!==1)return undefined;
 const block=row>>>5,bit=1<<(row&31),mask=words[layout.bitsetBase+group*layout.blocks+block]!;
 if((mask&bit)===0)return undefined;
 const before=words[layout.prefixBase+group*layout.blocks+block]!;
 const prior=popcount32(mask&(bit-1));
 const payload=words[descriptor+3]!+before+prior;
 return payload<layout.touchedRows?words[layout.payloadBase+payload]:undefined;
}
function popcount32(value:number):number{let n=value>>>0;let count=0;while(n){n=(n&(n-1))>>>0;count++;}return count;}
/** WGSL mirror used by the canonical playback consumer.  The caller supplies
 * the existing counts scratch and passes the compact group ordinal obtained
 * from its stable source-part record. */
export const SOURCE_PART_ROW_LOOKUP_WGSL=/* wgsl */`
const SOURCE_PART_ROW_TABLE_MAGIC=${SOURCE_PART_ROW_TABLE_MAGIC}u;
const SOURCE_PART_ROW_TABLE_VERSION=${SOURCE_PART_ROW_TABLE_VERSION}u;
const SOURCE_PART_ROW_TABLE_HEADER_WORDS=${SOURCE_PART_ROW_TABLE_HEADER_WORDS}u;
const SOURCE_PART_ROW_TABLE_GROUP_WORDS=${SOURCE_PART_ROW_TABLE_GROUP_WORDS}u;
fn sourcePartRowBase()->u32{return p.partAnchors.x;}
fn sourcePartRowWord(index:u32)->u32{return atomicLoad(&counts[sourcePartRowBase()+index]);}
fn sourcePartRowStore(index:u32,value:u32){atomicStore(&counts[sourcePartRowBase()+index],value);}
fn sourcePartRowReady()->bool{
 let base=sourcePartRowBase();let words=sourcePartRowWord(0u);
 return base>0u&&base+SOURCE_PART_ROW_TABLE_HEADER_WORDS<=arrayLength(&counts)&&
  sourcePartRowWord(0u)==SOURCE_PART_ROW_TABLE_MAGIC&&sourcePartRowWord(1u)==SOURCE_PART_ROW_TABLE_VERSION&&
  sourcePartRowWord(2u)>0u&&sourcePartRowWord(3u)>0u&&sourcePartRowWord(4u)>0u&&
  sourcePartRowWord(5u)==SOURCE_PART_ROW_TABLE_HEADER_WORDS&&sourcePartRowWord(6u)>=sourcePartRowWord(5u)+sourcePartRowWord(11u)&&
  sourcePartRowWord(7u)>=sourcePartRowWord(6u)+sourcePartRowWord(2u)*SOURCE_PART_ROW_TABLE_GROUP_WORDS&&
  sourcePartRowWord(8u)>=sourcePartRowWord(7u)+sourcePartRowWord(2u)*sourcePartRowWord(4u)&&
  sourcePartRowWord(9u)>=sourcePartRowWord(8u)+sourcePartRowWord(2u)*sourcePartRowWord(4u)&&
  sourcePartRowWord(9u)+sourcePartRowWord(10u)<=arrayLength(&counts)-base&&words>sourcePartRowWord(9u);
}
fn sourcePartRowGroupForRun(runID:u32)->u32{
 if(!sourcePartRowReady()||runID>=sourcePartRowWord(11u)){return NONE;}
 return sourcePartRowWord(sourcePartRowWord(5u)+runID);
}
fn sourcePartRowEdge(group:u32,row:i32)->u32{
 if(!sourcePartRowReady()||group>=sourcePartRowWord(2u)||row<0||u32(row)>=sourcePartRowWord(3u)){return NONE;}
 let block=u32(row)>>5u;let bit=1u<<(u32(row)&31u);
 let descriptor=sourcePartRowWord(sourcePartRowWord(6u)+group*SOURCE_PART_ROW_TABLE_GROUP_WORDS);
 if(descriptor!=1u){return NONE;}
 let mask=sourcePartRowWord(sourcePartRowWord(7u)+group*sourcePartRowWord(4u)+block);if((mask&bit)==0u){return NONE;}
 let before=sourcePartRowWord(sourcePartRowWord(8u)+group*sourcePartRowWord(4u)+block);
 let prior=countOneBits(mask&(bit-1u));let payload=sourcePartRowWord(sourcePartRowWord(6u)+group*SOURCE_PART_ROW_TABLE_GROUP_WORDS+3u)+before+prior;
 if(payload>=sourcePartRowWord(10u)){return NONE;}
 return sourcePartRowWord(sourcePartRowWord(9u)+payload);
}
fn sourcePartRowGroupReady(group:u32)->bool{
 return sourcePartRowReady()&&group<sourcePartRowWord(2u)&&sourcePartRowWord(sourcePartRowWord(6u)+group*SOURCE_PART_ROW_TABLE_GROUP_WORDS)==1u;
}
`;
export interface PathPointSample {readonly x:number;readonly y:number;readonly wholeEdgeDescriptor?:boolean}
function cpuPathBent(points:readonly PathPointSample[]):boolean{
 const finite=points.filter(point=>!point.wholeEdgeDescriptor);if(finite.length===0)return false;
 const first=finite[0]!,last=finite[finite.length-1]!,dx=last.x-first.x,dy=last.y-first.y;
 const threshold=.25*Math.max(1,Math.hypot(dx,dy));let bent=false;
 for(const point of finite)bent=bent||Math.abs((point.x-first.x)*dy-(point.y-first.y)*dx)>threshold;
 return bent;
}
/** CPU mirror of the WGSL source-and-geometry classifier. It preserves the
 * production rule exactly: a hard tag is hard only when the path is straight,
 * while an odd point count takes the incumbent repeat consumer. */
export function classifyPathRun(flags:number,points:readonly PathPointSample[],programFeature=0):PathPlaybackClass{
 if(!Number.isInteger(flags)||flags<0)throw Error('Invalid PATH run flags');
 if((flags&0x1000000)!==0)return 'pair';
 if((flags&0x10000)!==0)return 'anchor';
 if((flags&0x4120000)!==0)return 'boundary';
 const hardLine=(flags&0x40000)!==0&&!cpuPathBent(points);
 if(hardLine||points.length%2!==0)return 'hard';
 return programFeature!==0?'softProgram':'soft';
}
export function classifyPathRunFlags(flags:number):PathPlaybackClass{return classifyPathRun(flags,[]);}
const PATH_GEOMETRY_WGSL=`
struct PathGeometry {bent:bool,hardLine:bool,soft:bool}
fn pathGeometry(run:Run,first:vec2f,last:vec2f)->PathGeometry{
 let delta=last-first;var bent=false;
 for(var i=0u;i<run.range.y;i++){
  let point=points[run.range.x+i];if(pointWholeEdgeDescriptor(point)){continue;}
  let q=point.position.xy;bent=bent||abs((q.x-first.x)*delta.y-(q.y-first.y)*delta.x)>.25*max(1.0,length(delta));
 }
 let hardLine=(run.range.z&0x40000u)!=0u&&!bent;
 return PathGeometry(bent,hardLine,!hardLine&&(run.range.y&1u)==0u);
}
`;
const PATH_CLASS_WGSL=`
const PATH_SOFT=0u;const PATH_HARD=1u;const PATH_ANCHOR=2u;const PATH_PAIR=3u;const PATH_BOUNDARY=4u;const PATH_SOFT_PROGRAM=5u;
// One source-tag owner for all PATH consumers. Boundary forms are admitted
// by the planar/round queue, while paired forms intentionally stay on PATH.
fn pathTagClass(run:Run)->u32{
 let flags=run.range.z;
 if((flags&AUTHORED_PAIR_FLAG)!=0u){return PATH_PAIR;}
 if((flags&0x10000u)!=0u){return PATH_ANCHOR;}
 if((flags&0x4120000u)!=0u){return PATH_BOUNDARY;}
 return PATH_SOFT;
}
fn pathPlaybackClass(run:Run,geometry:PathGeometry)->u32{
 let tagged=pathTagClass(run);if(tagged!=PATH_SOFT){return tagged;}
 if(!geometry.soft){return PATH_HARD;}
 return select(PATH_SOFT,PATH_SOFT_PROGRAM,runProgramFeature(run)!=0u);
}
`;
const PLANAR=` if((boundary&0x100000u)!=0u){hardBoundaryConstruction(runID,run,priority,true);return;}
 if((boundary&0x4000000u)!=0u){hardBoundaryConstruction(runID,run,priority,false);return;}
`;
const ROUND=` if((boundary&0x20000u)!=0u){circleConstruction(runID,run,priority);return;}
`;
const FAR=` if(away!=0u&&!anchorOnly){
  let span=skySpan(run);let far=farKind(run,away,span);
  if((far==1u||far==3u)&&farSkyline(runID,run,far,span,priority)){return;}
 }
`;
const BOX_HOLD_READ=` for(var k=0u;k<boxHolds;k++){let held=boxHold[k];let centre=2*x+i32(advance);if(held.x<held.y&&(row==held.z||row==held.w)&&centre>=held.x&&centre<=held.y){return;}}
`;
const PAIR_BRANCH=` if((run.range.z&AUTHORED_PAIR_FLAG)!=0u){
  lineTier=heldTier(runID,run);
  authoredPairConstruction(runID,run,min(rank(run,first.xy,last.xy),0x7fffffffu));return;
 }
`;
function replace(source:string,before:string,after:string):string{
 if(source.split(before).length!==2)throw Error('Playback compiler anchor changed: '+before.slice(0,80));
 return source.replace(before,after);
}
function replaceComputeEntry(source:string,entry:string,replacement:string):string{
 const marker=`@compute @workgroup_size(64) fn ${entry}(`,start=source.indexOf(marker);
 if(start<0)throw Error(`Playback compiler entry missing: ${entry}`);
 const open=source.indexOf('{',start);if(open<0)throw Error(`Playback compiler body missing: ${entry}`);
 let depth=1;
 for(let i=open+1;i<source.length;i++){
  if(source[i]==='/'&&source[i+1]==='/'){const end=source.indexOf('\n',i+2);i=end<0?source.length:end;continue;}
  if(source[i]==='/'&&source[i+1]==='*'){const end=source.indexOf('*/',i+2);if(end<0)throw Error(`Playback compiler comment missing: ${entry}`);i=end+1;continue;}
  if(source[i]==='{')depth++;
  else if(source[i]==='}'&&--depth===0)return source.slice(0,start)+replacement+source.slice(i+1);
 }
 throw Error(`Playback compiler body unbalanced: ${entry}`);
}
function rewriteComputeEntry(source:string,entry:string,edit:(body:string)=>string):string{
 const marker=`@compute @workgroup_size(64) fn ${entry}(`,start=source.indexOf(marker);
 if(start<0)throw Error(`Playback compiler entry missing: ${entry}`);
 const open=source.indexOf('{',start);if(open<0)throw Error(`Playback compiler body missing: ${entry}`);
 let depth=1,end=-1;
 for(let i=open+1;i<source.length;i++){
  if(source[i]==='/'&&source[i+1]==='/'){const line=source.indexOf('\n',i+2);i=line<0?source.length:line;continue;}
  if(source[i]==='/'&&source[i+1]==='*'){const comment=source.indexOf('*/',i+2);if(comment<0)throw Error(`Playback compiler comment missing: ${entry}`);i=comment+1;continue;}
  if(source[i]==='{')depth++;
  else if(source[i]==='}'&&--depth===0){end=i;break;}
 }
 if(end<0)throw Error(`Playback compiler body unbalanced: ${entry}`);
 return source.slice(0,open+1)+edit(source.slice(open+1,end))+source.slice(end);
}
function rewriteFunction(source:string,name:string,edit:(body:string)=>string):string{
 const marker=`fn ${name}(`,start=source.indexOf(marker);
 if(start<0)throw Error(`Playback compiler function missing: ${name}`);
 const open=source.indexOf('{',start);if(open<0)throw Error(`Playback compiler function body missing: ${name}`);
 let depth=1,end=-1;
 for(let i=open+1;i<source.length;i++){
  if(source[i]==='/'&&source[i+1]==='/'){const line=source.indexOf('\n',i+2);i=line<0?source.length:line;continue;}
  if(source[i]==='/'&&source[i+1]==='*'){const comment=source.indexOf('*/',i+2);if(comment<0)throw Error(`Playback compiler comment missing: ${name}`);i=comment+1;continue;}
  if(source[i]==='{')depth++;
  else if(source[i]==='}'&&--depth===0){end=i;break;}
 }
 if(end<0)throw Error(`Playback compiler function body unbalanced: ${name}`);
 return source.slice(0,open+1)+edit(source.slice(open+1,end))+source.slice(end);
}
function removeUntil(source:string,startMarker:string,endMarker:string):string{
 const start=source.indexOf(startMarker),end=start<0?-1:source.indexOf(endMarker,start);
 if(start<0||end<0)throw Error(`Playback compiler section changed: ${startMarker.slice(0,80)}`);
 return source.slice(0,start)+source.slice(end);
}
function truncateBody(source:string,marker:string):string{
 const at=source.indexOf(marker);if(at<0)throw Error(`Playback compiler body marker changed: ${marker.slice(0,80)}`);
 return source.slice(0,at)+' return;\n';
}
function lowerSharedPathGeometry(body:string):string{
 const original=` var straightVertical=true;var bent=false;var monotone=true;var havePrevious=false;var previousY=first.y;
 for(var i=0u;i<run.range.y;i++){
  let point=points[run.range.x+i];if(pointWholeEdgeDescriptor(point)){continue;}
  let q=point.position.xy;straightVertical=straightVertical&&abs(q.x-first.x)<.5;
  bent=bent||abs((q.x-first.x)*delta.y-(q.y-first.y)*delta.x)>.25*max(1.0,length(delta));
`;
 const lowered=` var straightVertical=true;var bent=geometry.bent;var monotone=true;var havePrevious=false;var previousY=first.y;
 for(var i=0u;i<run.range.y;i++){
  let point=points[run.range.x+i];if(pointWholeEdgeDescriptor(point)){continue;}
  let q=point.position.xy;straightVertical=straightVertical&&abs(q.x-first.x)<.5;
`;
 if(!body.includes(original))throw Error('Playback compiler path geometry body changed');
 body=body.replace(original,lowered);
 body=body.replace(' let hardLine=(run.range.z&0x40000u)!=0u&&!bent;',' let hardLine=geometry.hardLine;');
 body=body.replace(' let soft=!hardLine&&(run.range.y&1u)==0u;',' let soft=geometry.soft;');
 return body;
}
function specializeSoftProducer(code:string,kind:'legacy'|'program'):string{
 if(kind==='legacy'){
  code=rewriteFunction(code,'softRunPhaseOrigin',()=>`\n return finiteRunStart(run);\n`);
  code=rewriteFunction(code,'softRunPlayback',body=>{
   // This queue contains ONLY unprogrammed sources. Remove its canonical
   // prologue by its interface, not by a frozen copy of the programmed
   // producer's condition. Junction/context changes belong to that producer.
   const start=body.indexOf(' let canonicalReady=sourcePartCanonicalPlayable(run);');
   const end=body.indexOf(' let base=run.range.x;let pairs=run.range.y/2u;',start);
   const prologue=body.slice(start,end);
   if(start<0||end<0||!prologue.includes('programmedWholeSourcePart(')||!prologue.includes('softRunAnchor='))
    throw Error('Playback compiler legacy soft producer interface changed');
   return (body.slice(0,start)+' softRunAnchor=finiteRunStart(run);\n'+body.slice(end)).replaceAll(',canonicalReady);',',false);');
  });
  code=rewriteFunction(code,'softPortion',body=>{
   const start=' let base=run.range.x;let programFeature=runProgramFeature(run);var role=runRole(run);';
   const end=' let enter=interpolatePoint(points[base+startPair*2u],points[base+startPair*2u+1u],startT);';
   const at=body.indexOf(start),to=body.indexOf(end,at);if(at<0||to<0)throw Error('Playback compiler legacy soft source block changed');
   body=body.slice(0,at)+` let base=run.range.x;var role=runRole(run);
 var sourcePair=softProgramRowPair(runID,run,row);let sourceQ=softSourceAt(run,sourcePair,row);
`+body.slice(to);
   body=body.replace(` var sourceA=canonical.a;var sourceB=canonical.b;
 if(!canonicalReady||canonical.valid==0u){sourceA=softWholeA(run,sourcePair);sourceB=softWholeB(run,sourcePair);}`,' var sourceA=softWholeA(run,sourcePair);var sourceB=softWholeB(run,sourcePair);');
   body=removeUntil(body,` if(programFeature!=0u&&sourcePartPrepared(run)!=NONE&&programmedWholePart(runID,run,role,size,row,portion,seen,sourcePair,sourceQ,priority,canonicalReady,canonical)){return;}
`,' // ONE UNIT PER RUN');
   body=body.replace(' var dir=select(softNativeDirection(chord),runDirection,programFeature!=0u&&!canonicalReady);',' var dir=softNativeDirection(chord);');
   body=body.replaceAll('(programFeature!=0u&&runProgramD5(run))||(programFeature==0u&&curvedRun&&abs(row-lastUprightRow)==1&&upright)','(curvedRun&&abs(row-lastUprightRow)==1&&upright)');
   body=body.replaceAll('(programFeature!=0u&&runProgramD5(run))||(programFeature==0u&&d5Swap(grammar[unit],row))','d5Swap(grammar[unit],row)');
   body=body.replaceAll('(programFeature!=0u&&runProgramD5(run))||(programFeature==0u&&d5Swap(grammar[record],row))','d5Swap(grammar[record],row)');
   body=body.replace('if(programFeature==0u&&abs(chord.y)>=9.0)','if(abs(chord.y)>=9.0)');
   body=body.replace('terrain&&programFeature==0u','terrain').replaceAll('else if(programFeature==0u&&','else if(');
   body=body.replace(' if(programFeature==0u){sourcePair=cursor;sourceA=softWholeA(run,cursor);sourceB=softWholeB(run,cursor);}',' sourcePair=cursor;sourceA=softWholeA(run,cursor);sourceB=softWholeB(run,cursor);');
   if(body.includes('programFeature')||body.includes('canonicalReady')||body.includes('sourcePartCanonical')||body.includes('programmedWhole'))throw Error('Legacy soft closure retained programmed source authority: '+(body.match(/[^\\n]*(?:programFeature|canonicalReady|sourcePartCanonical|programmedWhole)[^\\n]*/g)||[]).join(' || '));
   return body;
  });
  return code;
 }
 code=rewriteFunction(code,'softRunPlayback',body=>body.replace('if(runProgramFeature(run)==0u){directionSegments(run);}',''));
 code=rewriteFunction(code,'softPortion',body=>{
  const start=' let base=run.range.x;let programFeature=runProgramFeature(run);var role=runRole(run);';
  const end=' let enter=interpolatePoint(points[base+startPair*2u],points[base+startPair*2u+1u],startT);';
  const at=body.indexOf(start),to=body.indexOf(end,at);if(at<0||to<0)throw Error('Playback compiler programmed soft source block changed');
  // Retain the actual programmed row reader (including source-owned junction
  // assignment). Rebuilding this block here silently restored an older reader
  // after the source owner changed it. Only the impossible legacy branch goes.
  let prologue=body.slice(at,to).replace(start,' let base=run.range.x;var role=runRole(run);var terrain=terrainRun(run);');
  prologue=removeUntil(prologue,' if(programFeature==0u){',' else if(canonicalReady){');
  prologue=replace(prologue,' else if(canonicalReady){',' if(canonicalReady){');
  body=body.slice(0,at)+prologue+body.slice(to);
  body=removeUntil(body,` // The PART of the form this row of the outline lies on, by the run pass's
`,' // Programmed source parts consume');
  body=body.replace(' if(programFeature!=0u&&sourcePartPrepared(run)!=NONE&&programmedWholePart(runID,run,role,size,row,portion,seen,sourcePair,sourceQ,priority,canonicalReady,canonical)){return;}',' if(sourcePartPrepared(run)!=NONE&&programmedWholePart(runID,run,role,size,row,portion,seen,sourcePair,sourceQ,priority,canonicalReady,canonical)){return;}');
  body=body.replace(' var dir=select(softNativeDirection(chord),runDirection,programFeature!=0u&&!canonicalReady);',' var dir=select(softNativeDirection(chord),runDirection,!canonicalReady);');
  body=removeUntil(body,' if(programFeature==0u&&abs(chord.y)>=9.0){',' let column=softColumn');
  body=body.replaceAll('(programFeature!=0u&&runProgramD5(run))||(programFeature==0u&&curvedRun&&abs(row-lastUprightRow)==1&&upright)','runProgramD5(run)');
  body=body.replaceAll('(programFeature!=0u&&runProgramD5(run))||(programFeature==0u&&d5Swap(grammar[unit],row))','runProgramD5(run)');
  body=body.replaceAll('(programFeature!=0u&&runProgramD5(run))||(programFeature==0u&&d5Swap(grammar[record],row))','runProgramD5(run)');
  body=body.replace(' if(programFeature==0u){sourcePair=cursor;sourceA=softWholeA(run,cursor);sourceB=softWholeB(run,cursor);}','');
  if(body.includes('programFeature==0u')||body.includes('directionSegments(run)'))throw Error('Programmed soft closure retained legacy row authority: '+(body.match(/[^\\n]*(?:programFeature==0u|directionSegments\\(run\\))[^\\n]*/g)||[]).join(' || '));
  return body;
 });
 return code;
}
// A separate per-source word immediately before the existing selection grids.
// Loss/copy record offsets and publication ABI are unchanged.
export const FAR_STATUS_WGSL=`fn farStatus(runID:u32)->u32{return drawnBase()+p.playback.z+p.extent.z*2u+runID;}
`;
function genericFamilyPlayback(source:string,family:PlaybackFamily):string{
 let code=source+'\n'+FAR_STATUS_WGSL;
 const queueFamily=PLAYBACK_FAMILIES.indexOf(family)+1;
 code=replace(code,'@compute @workgroup_size(64) fn featurecurves(@builtin(global_invocation_id) id:vec3u){\n let runID=scheduledRun(id);',`@compute @workgroup_size(64) fn featurecurves(@builtin(global_invocation_id) id:vec3u){\n let runID=scheduledRunFamily(id,${queueFamily}u);`);
 code=replace(code,' let boundary=wholeBoundaryFlags(run);\n',` let boundary=wholeBoundaryFlags(run);
 let family=select(select(0u,2u,(boundary&0x20000u)!=0u),1u,(boundary&0x4100000u)!=0u);
 if(family!=${PLAYBACK_FAMILIES.indexOf(family)}u){return;}
`);
 code=replace(code,PAIR_BRANCH,' if((run.range.z&AUTHORED_PAIR_FLAG)!=0u){return;}\n');
 code=replace(code,FAR,' if(atomicLoad(&counts[farStatus(runID)])!=0u){return;}\n');
 if(family!=='planar')code=replace(code,PLANAR,'');
 // No writer of boxHolds is reachable in these programs: its specified
 // per-invocation zero value made this loop empty in the combined kernel.
 if(family!=='planar')code=replace(code,BOX_HOLD_READ,'');
 if(family!=='round')code=replace(code,ROUND,'');
 if(family!=='path'){
  const from=code.indexOf(' if(anchorOnly&&pointVisible(finiteRunStart(run))){for(var i=0u;i<run.paper.y;i++){');
  const to=code.indexOf('\nvar<workgroup> twinBits:',from);
  if(from<0||to<0)throw Error('Missing path program boundaries');
  code=code.slice(0,from)+' return;\n}\n'+code.slice(to);
 }
 return isolateComputeEntry(code,'featurecurves');
}
function pathPlaybackBase(source:string):string{
 let code=source+'\n'+FAR_STATUS_WGSL+PATH_GEOMETRY_WGSL+PATH_CLASS_WGSL;
 code=replace(code,'@compute @workgroup_size(64) fn featurecurves(@builtin(global_invocation_id) id:vec3u){\n let runID=scheduledRun(id);',`@compute @workgroup_size(64) fn featurecurves(@builtin(global_invocation_id) id:vec3u){\n let runID=scheduledRunFamily(id,1u);`);
 code=replace(code,` let first=finiteRunStart(run).position;let last=finiteRunEnd(run).position;`,` let first=finiteRunStart(run).position;let last=finiteRunEnd(run).position;let geometry=pathGeometry(run,first.xy,last.xy);if(pathPlaybackClass(run,geometry)!=PATH_SOFT){return;}\n`);
 code=replace(code,PAIR_BRANCH,'');
 code=replace(code,FAR,' if(atomicLoad(&counts[farStatus(runID)])!=0u){return;}\n');
 // PATH owns source runs only. Planar and round forms are admitted to their
 // own queue/consumer; keeping these calls reachable here defeats closure.
 code=replace(code,PLANAR,'');code=replace(code,ROUND,'');
 return code;
}
function compileSoftPlayback(source:string,kind:'legacy'|'program'):string{
 let code=pathPlaybackBase(source);
 const expected=kind==='legacy'?'PATH_SOFT':'PATH_SOFT_PROGRAM';
 code=code.replace('if(pathPlaybackClass(run,geometry)!=PATH_SOFT){return;}',`if(pathPlaybackClass(run,geometry)!=${expected}){return;}`);
 code=specializeSoftProducer(code,kind);
 code=rewriteComputeEntry(code,'featurecurves',body=>{
  body=removeUntil(body,' if(anchorOnly){\n  // Whole anchors keep their native spelling as the camera approaches.',' // Selection by distance tier');
  body=removeUntil(body,' if(anchorOnly&&pointVisible(finiteRunStart(run))){for(var i=0u;i<run.paper.y;i++){',' if(run.range.y<2u){return;}');
  body=truncateBody(body,' let count=grammar[instruction+1u];let advance=grammar[instruction+2u];');
  body=body.replace(' let anchorOnly=(run.range.z&0x10000u)!=0u;\n','');
  body=body.replace('let priority=select(baseRank,0x80000000u|baseRank,anchorOnly);','let priority=baseRank;');
  body=body.replace(' if(anchorOnly&&away!=0u&&(farUnderGlyph(run)||(away==2u&&farStands(run)<85.0))){return;}\n','');
  body=body.replace(' if(!anchorOnly&&!tierSelects(run,tier)){return;}',' if(!tierSelects(run,tier)){return;}');
  body=body.replace(' if(!anchorOnly&&rippleRun(run)&&rippleYields(run)){return;}',' if(rippleRun(run)&&rippleYields(run)){return;}');
  body=body.replace(' if(!anchorOnly&&terrainRun(run)){probeAdd(35u+min(tier,2u)*5u+min((run.range.z>>8u)&255u,4u));}',' if(terrainRun(run)){probeAdd(35u+min(tier,2u)*5u+min((run.range.z>>8u)&255u,4u));}');
  body=body.replace(' if(!anchorOnly&&!terrainRun(run)&&(run.range.z&0x40000u)!=0u&&wholeBoundaryFlags(run)==0u){probeAdd(52u+min(tier,2u)*5u+min((run.range.z>>8u)&255u,4u));}\n','');
  body=body.replace(' if(away!=0u&&!anchorOnly){',' if(away!=0u){');
  body=body.replace(' if(!anchorOnly&&boundary==0u&&run.range.y>=2u&&((run.range.z>>8u)&255u)<3u){gridWrite(run,baseRank);}',' if(boundary==0u&&run.range.y>=2u&&((run.range.z>>8u)&255u)<3u){gridWrite(run,baseRank);}');
  body=body.replace(' if(!anchorOnly){hardJunction(runID,run,baseRank);}',' hardJunction(runID,run,baseRank);');
  body=lowerSharedPathGeometry(body);
  return body;
 });
 return foldUnwrittenBoxHolds(isolateComputeEntry(code,'featurecurves'),'featurecurves');
}
function pathSoftPlayback(source:string):string{return compileSoftPlayback(source,'legacy');}
export function pathProgramPlayback(source:string):string{return compileSoftPlayback(source,'program');}
export function pathHardPlayback(source:string):string{
 let code=pathPlaybackBase(source);
 code=code.replace('if(pathPlaybackClass(run,geometry)!=PATH_SOFT){return;}','if(pathPlaybackClass(run,geometry)!=PATH_HARD&&pathPlaybackClass(run,geometry)!=PATH_ANCHOR){return;}');
 code=rewriteComputeEntry(code,'featurecurves',body=>{
  const softStart=' if(soft){\n';
  const hardStart=' let count=grammar[instruction+1u];let advance=grammar[instruction+2u];';
  body=removeUntil(body,softStart,hardStart);
  body=body.replace(' if(soft){softPhaseAnchor=softRunPhaseOrigin(run).position.y;}','');
  body=body.replace('  if(soft){atomicStore(&counts[memo+6u],bitcast<u32>(softPhaseAnchor-floor(softPhaseAnchor/17.0)*17.0));atomicStore(&counts[memo+7u],bitcast<u32>(angle));}','');
  body=lowerSharedPathGeometry(body);
  return body;
 });
 return foldUnwrittenBoxHolds(isolateComputeEntry(code,'featurecurves'),'featurecurves');
}
export function familyPlayback(source:string,family:PlaybackFamily):string{
 return family==='path'?pathSoftPlayback(source):genericFamilyPlayback(source,family);
}
/** Compile-closed consumer for the authored paired construction. It shares
 * the PATH active queue with the ordinary entry, but its mutually exclusive
 * guard means each queued source is emitted by exactly one entry. Replacing
 * the complete kernel body before isolation is important: leaving the old
 * early branch in the combined shader keeps the generic soft/hard/round/tone
 * callgraph alive even when the branch returns at runtime. */
export function pairPlayback(source:string):string{
 let code=source+'\n'+FAR_STATUS_WGSL+PATH_GEOMETRY_WGSL+PATH_CLASS_WGSL;
 const entry=`@compute @workgroup_size(64) fn featurecurves(@builtin(global_invocation_id) id:vec3u){
 let runID=scheduledRunFamily(id,1u);
 if(runID==0xffffffffu||runID>=p.extent.z){return;}
 var run=runs[runID];if(run.range.y==0u||pathTagClass(run)!=PATH_PAIR){return;}
 gridKeyBase=0u;collapseDir=NONE;
 let first=finiteRunStart(run).position;let last=finiteRunEnd(run).position;
 for(var i=0u;i<run.range.y;i++){if(pointVisible(points[run.range.x+i])){atomicOr(&counts[drawnSlot(run)],0x80000000u);break;}}
 lineTier=heldTier(runID,run);
 authoredPairConstruction(runID,run,min(rank(run,first.xy,last.xy),0x7fffffffu));
}`;
 code=replaceComputeEntry(code,'featurecurves',entry);
 const closed=isolateComputeEntry(code,'featurecurves');
 return foldUnwrittenJunction(foldUnwrittenBoxHolds(closed,'featurecurves'),'featurecurves');
}
// Fold only the closed module's implicit-zero arrays, never a real hold
// producer. The paired entry does not call box/junction preparation, so
// retaining those per-word loops only expands driver compiler input.
function foldUnwrittenBoxHolds(closed:string,entry:string):string{
 const code=closed.replace(/\/\*[\s\S]*?\*\//g,'').replace(/\/\/[^\n]*/g,'');
 if(!code.includes('var<private> boxHolds:u32;')||!code.includes('var<private> boxHold:array<vec4i,BOX_HOLDS>;')||
  [...code.matchAll(/\bboxHolds\b/g)].length!==2||[...code.matchAll(/\bboxHold\b/g)].length!==2||closed.split(BOX_HOLD_READ).length!==2)return closed;
 return isolateComputeEntry(closed.replace(BOX_HOLD_READ,'\n'),entry);
}
// Closed far playback never writes junctionHold. Its zero private array
// cannot reject a word, but retaining its loop expands every inlined emit.
// A future writer/escape/initializer makes this fold inapplicable, not a
// missing drawing: keep that complete original program unchanged.
const ZERO_FAR_JUNCTION_READ=" for(var side=0u;side<JUNCTION_HOLDS;side++){let held=junctionHold[side];let centre=2*x+i32(advance);if(held.x<held.y&&(row==held.z||row==held.w)&&centre>=held.x&&centre<=held.y){return;}}\n";
function foldUnwrittenJunction(closed:string,entry:string):string{
 const code=closed.replace(/\/\*[\s\S]*?\*\//g,'').replace(/\/\/[^\n]*/g,'');
 if(!code.includes('var<private> junctionHold:array<vec4i,JUNCTION_HOLDS>;')||[...code.matchAll(/\bjunctionHold\b/g)].length!==2||closed.split(ZERO_FAR_JUNCTION_READ).length!==2)return closed;
 return isolateComputeEntry(closed.replace(ZERO_FAR_JUNCTION_READ,'\n'),entry);
}
export function farPlayback(source:string):string{
 const closed=isolateComputeEntry(replace(source,BOX_HOLD_READ,'')+'\n'+FAR_STATUS_WGSL+`
@compute @workgroup_size(64) fn featurefar(@builtin(global_invocation_id) id:vec3u){
 let runID=scheduledRunFamily(id,4u);if(runID==0xffffffffu||runID>=p.extent.z){return;}
 atomicStore(&counts[farStatus(runID)],0u);
 let run=runs[runID];if(run.range.y<2u||(run.range.z&0x10000u)!=0u){return;}
 if(wholeBoundaryFlags(run)==0u&&atomicLoad(&counts[memoMapBase()+runID])==NONE){return;}
 gridKeyBase=0u;collapseDir=NONE;
 let away=heldTier(runID,run);if(away==0u||!tierSelects(run,away)){return;}
 if(rippleRun(run)&&rippleYields(run)){return;}
 lineTier=away;
 let first=finiteRunStart(run).position;let last=finiteRunEnd(run).position;
 let priority=min(rank(run,first.xy,last.xy),0x7fffffffu);
 let span=skySpan(run);let far=farKind(run,away,span);
 if((far==1u||far==3u)&&farSkyline(runID,run,far,span,priority)){atomicStore(&counts[farStatus(runID)],1u);}
}
`,'featurefar');
 return foldUnwrittenJunction(closed,'featurefar');
}
