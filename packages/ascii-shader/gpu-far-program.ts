import {farPlayback} from './gpu-playback-program.ts';
import {isolateComputeEntry} from './gpu-compute-program.ts';
import {PARENT_LEAVE_ROWS} from './source-lod.ts';
import {SMALL_CONSTRUCTION_WGSL} from './gpu-small-construction.ts';

/** Per-run far-kind execution split. The plan is eight u32 words in the
 * existing counts buffer at p.farLoops.w: kind, away, priority, span.xyzw,
 * and packed canonical profile-loop limits. Drawing functions are extracted from
 * the current canonical far playback source before specialization. */
export const FAR_PLAN_WORDS=8;
export const FAR_PLAN_BASE='p.farLoops.w';
export const FAR_PLAN_LAYOUT=['kind','away','priority','span.x','span.y','span.z','span.w','profileLoops'] as const;
export const FAR_KIND_ENTRIES=['featurefarPrepare','featurefarTerrain','featurefarStanding'] as const;
/** The parent selector's held leave band is the only row bound for this
 * representation.  Include one boundary row because a projected box may
 * straddle two native row edges; a source geometry beyond this bound declines
 * the parent and keeps the incumbent representation. */
const FAR_PARENT_ROW_PLAN_MAX=Math.ceil(PARENT_LEAVE_ROWS)+1;

const bits=(value:number)=>new Uint32Array(new Float32Array([value]).buffer)[0]!;
const float=(value:number)=>new Float32Array(new Uint32Array([value>>>0]).buffer)[0]!;

export type FarPlan={kind:number;away:number;priority:number;span:[number,number,number,number];profileLoops:number};
export function packFarPlan(plan:FarPlan):Uint32Array{
 const out=new Uint32Array(FAR_PLAN_WORDS);out[0]=plan.kind>>>0;out[1]=plan.away>>>0;out[2]=plan.priority>>>0;
 for(let i=0;i<4;i++)out[3+i]=bits(plan.span[i]!);
 out[7]=plan.profileLoops>>>0;
 return out;
}
export function unpackFarPlan(words:ArrayLike<number>):FarPlan{
 if(words.length<FAR_PLAN_WORDS)throw Error('Far plan is truncated');
 return {kind:words[0]!,away:words[1]!,priority:words[2]!,span:[float(words[3]!),float(words[4]!),float(words[5]!),float(words[6]!)],profileLoops:words[7]!};
}

/** Source scanner used for all canonical WGSL extraction. Braces in comments
 * never affect a function or conditional range; a changed/missing anchor is a
 * hard generator error rather than an approximation. */
function functionRange(source:string,name:string):[number,number]{
 const code=codeMask(source),start=code.search(new RegExp(`\\bfn ${name}\\s*\\(`));if(start<0)throw Error(`Missing WGSL function ${name}`);
 const open=code.indexOf('{',start);if(open<0)throw Error(`Missing WGSL body ${name}`);
 return [start,blockEnd(code,open)];
}
function codeMask(source:string):string{
 let result='',i=0;
 while(i<source.length){
  if(source.slice(i,i+2)==='//'){while(i<source.length&&source[i]!=='\n'){result+=' ';i++;}}
  else if(source.slice(i,i+2)==='/*'){
   let depth=1;result+='  ';i+=2;
   while(i<source.length&&depth){
    const pair=source.slice(i,i+2);
    if(pair==='/*'){depth++;result+='  ';i+=2;}
    else if(pair==='*/'){depth--;result+='  ';i+=2;}
    else{result+=source[i]==='\n'?'\n':' ';i++;}
   }
   if(depth)throw Error('Unterminated WGSL comment');
  }else result+=source[i++];
 }
 return result;
}
function blockEnd(source:string,open:number):number{
 let depth=0;
 for(let i=open;i<source.length;i++){
  if(source[i]==='{')depth++;else if(source[i]==='}'&&--depth===0)return i+1;
 }
 throw Error('Unbalanced WGSL block');
}
function functionText(source:string,name:string):string{const [start,end]=functionRange(source,name);return source.slice(start,end);}
function replaceFunction(source:string,name:string,replacement:string):string{const [start,end]=functionRange(source,name);return source.slice(0,start)+replacement+source.slice(end);}
/** Keep a small closure of canonical soft-playback helpers when the far
 * profile is first isolated.  The far profile never calls these helpers, so
 * `farPlayback` quite correctly prunes them; the compiled-part bridge is the
 * second consumer that needs the same resident grammar/placement path. */
function retainCanonicalFunctions(source:string,code:string,roots:readonly string[]):string{
 // Mask each module once, not once per function: this runs during renderer
 // startup, including phones, and the canonical shader has hundreds of bodies.
 const available=new Set<string>();for(const match of codeMask(code).matchAll(/\bfn\s+(\w+)\s*\(/g))available.add(match[1]!);
 const masked=codeMask(source),functions=new Map<string,{body:string;code:string}>();
 for(const match of masked.matchAll(/\bfn\s+(\w+)\s*\(/g)){
  const name=match[1]!,start=match.index!,open=masked.indexOf('{',start);
  if(open<0)throw Error('Missing canonical WGSL body '+name);
  const end=blockEnd(masked,open);functions.set(name,{body:source.slice(start,end),code:masked.slice(start,end)});
 }
 const pending=[...roots],kept:string[]=[],names=new Set<string>();
 while(pending.length){const name=pending.pop()!;if(available.has(name)||names.has(name))continue;const fn=functions.get(name);if(!fn)continue;names.add(name);kept.push(fn.body);for(const id of fn.code.matchAll(/\b[A-Za-z_]\w*\b/g)){if(functions.has(id[0]!)&&!available.has(id[0]!))pending.push(id[0]!);}}
 return kept.join('\n');
}
function once(source:string,before:string,after:string):string{
 if(source.split(before).length!==2)throw Error(`Far kind anchor changed: ${before.slice(0,120)}`);
 return source.replace(before,after);
}

type IfRange={start:number;end:number;body:string;elseBody?:string};
function conditionalRange(source:string,condition:string,hint?:string):IfRange{
 const code=codeMask(source),anchor=`if(${condition}){`;let from=0;
 while(true){
  const start=code.indexOf(anchor,from);if(start<0)throw Error(`Missing conditional anchor: ${anchor}`);
  const open=start+anchor.length-1,close=blockEnd(code,open);
  let cursor=close;while(/\s/.test(code[cursor]??''))cursor++;
  let end=close,elseBody:string|undefined;
  if(code.startsWith('else{',cursor)){const elseOpen=cursor+4,elseEnd=blockEnd(code,elseOpen);elseBody=source.slice(elseOpen+1,elseEnd-1);end=elseEnd;}
  const body=source.slice(open+1,close-1);
  if(!hint||body.includes(hint)||elseBody?.includes(hint))return {start,end,body,elseBody};
  from=end;
 }
}
function replaceConditional(source:string,condition:string,choice:'body'|'else'|'remove',hint?:string):string{
 const range=conditionalRange(source,condition,hint);
 if(choice==='else'&&!range.elseBody)throw Error(`Missing else branch for ${condition}`);
 return source.slice(0,range.start)+(choice==='body'?range.body:choice==='else'?range.elseBody!:'')+source.slice(range.end);
}
function removeFalseConjunctions(source:string,predicate:string):string{
 const prefix=`if(${predicate}&&`;
 for(;;){
  const at=source.indexOf(prefix);if(at<0)return source;
  const open=source.indexOf('{',at),condition=source.slice(at+3,open).trim();
  if(open<0||!condition.endsWith(')')||condition.includes('||'))throw Error('Unsupported constant conjunction');
  const expression=condition.slice(0,-1),selected=conditionalRange(source,expression);
  source=replaceConditional(source,expression,selected.elseBody===undefined?'remove':'else');
 }
}

const FAR_PLAN_WGSL=/* wgsl */`
fn farPlanBase(runID:u32)->u32{return p.farLoops.w+runID*${FAR_PLAN_WORDS}u;}
fn farPlanWrite(runID:u32,kind:u32,away:u32,priority:u32,span:vec4f){let at=farPlanBase(runID);atomicStore(&counts[at],kind);atomicStore(&counts[at+1u],away);atomicStore(&counts[at+2u],priority);atomicStore(&counts[at+3u],bitcast<u32>(span.x));atomicStore(&counts[at+4u],bitcast<u32>(span.y));atomicStore(&counts[at+5u],bitcast<u32>(span.z));atomicStore(&counts[at+6u],bitcast<u32>(span.w));atomicStore(&counts[at+7u],0u);}
`;
const WORKGROUP_SCHEDULER_WGSL=/* wgsl */`
fn scheduledRunWorkgroup(group:vec3u,family:u32)->u32{
 let queued=select(p.extent.z,queueCount(family),p.queue.x!=0u);
 let rows=max(1u,(queued+p.proposals.w-1u)/p.proposals.w);
 let columns=max(1u,(queued+rows-1u)/rows);
 let ordinal=group.x+group.y*columns;if(ordinal>=queued){return 0xffffffffu;}
 if(p.queue.x==0u){return ordinal;}
 return atomicLoad(&counts[queueBase(family)+ordinal]);
}
`;

function planPrepare(source:string):string{
 const combined=farPlayback(source),incumbent=functionText(combined,'featurefar');
 const boundary=incumbent.indexOf(' lineTier=away;');if(boundary<0)throw Error('Missing incumbent far eligibility boundary');
 const open=incumbent.indexOf('{'),body=incumbent.slice(open+1,boundary);
 const tail=`let first=finiteRunStart(run).position;let last=finiteRunEnd(run).position;
 let priority=min(rank(run,first.xy,last.xy),0x7fffffffu);
 let span=skySpan(run);let far=farKind(run,away,span);let kind=select(0u,far,far==1u||far==3u);
 farPlanWrite(runID,kind,away,priority,span);`;
 const planClear=' let runID=scheduledRunFamily(id,4u);if(runID==0xffffffffu||runID>=p.extent.z){return;}\n atomicStore(&counts[farStatus(runID)],0u);';
 const plannedBody=once(body,planClear,' let runID=scheduledRunFamily(id,4u);if(runID==0xffffffffu||runID>=p.extent.z){return;}\n atomicStore(&counts[farStatus(runID)],0u);farPlanWrite(runID,0u,0u,0u,vec4f(0));');
 const entry=`fn featurefarPrepare(@builtin(global_invocation_id) id:vec3u){${plannedBody}${tail}\n}`;
 return isolateComputeEntry(FAR_PLAN_WGSL+'\n'+replaceFunction(combined,'featurefar',entry),'featurefarPrepare');
}

function specializeCell(combined:string,kind:1|3):string{
 let fn=functionText(combined,'farCell');
 const chosen=kind===3?replaceConditional(fn,'kind==3u','body'):replaceConditional(fn,'kind==3u','remove');
 return chosen.replace('fn farCell(','fn farCell'+(kind===1?'Terrain':'Standing')+'(').replace(',kind:u32','');
}
function specializeColumns(combined:string,kind:1|3):string{
 let fn=functionText(combined,'farColumns');
 fn=kind===3?replaceConditional(fn,'kind==3u','body'):replaceConditional(fn,'kind==3u','remove');
 return fn.replace('fn farColumns(','fn farColumns'+(kind===1?'Terrain':'Standing')+'(').replace(',kind:u32','');
}
function specializeUnits(combined:string,kind:1|3):string{
 let fn=functionText(combined,'skyUnits');
 if(kind===3){const selected=conditionalRange(fn,'cls==1u','profileUnits');fn=fn.slice(0,fn.indexOf('{')+1)+selected.body+'\n}';}
 else{fn=replaceConditional(fn,'cls==1u','remove','profileUnits');fn=replaceConditional(fn,'lower&&cls==1u&&shape==0u','remove');}
 return fn.replace('fn skyUnits(','fn skyUnits'+(kind===1?'Terrain':'Standing')+'(');
}

function specializeSkyline(combined:string,kind:1|3):string{
 let fn=functionText(combined,'farSkyline');
 fn=fn.replace('fn farSkyline(','fn farSkyline'+(kind===1?'Terrain':'Standing')+'(').replace(',kind:u32','');
 fn=once(fn,'let cls=select(0u,1u,kind==3u);',`let cls=${kind===1?'0u':'1u'};`);
 fn=fn.replaceAll('farColumns(run,kind,span)',`farColumns${kind===1?'Terrain':'Standing'}(run,span)`);
 fn=fn.replaceAll('farCell(run,kind,',`farCell${kind===1?'Terrain':'Standing'}(run,`);
 fn=fn.replaceAll('skyUnits(id,run,',`skyUnits${kind===1?'Terrain':'Standing'}(id,run,`);
 if(kind===1){
  fn=fn.replaceAll('if(cls==0u&&terrainRun(run))','if(terrainRun(run))');
  fn=replaceConditional(fn,'kind==3u','remove');
  fn=replaceConditional(fn,'cls==1u','remove','foot=groupFoot');
  fn=replaceConditional(fn,'cls==0u','body');
  fn=replaceConditional(fn,'top.on&&cls==1u&&(top.id>>9u)==(skyID(run)>>9u)','remove');
  fn=replaceConditional(fn,'cls==1u','remove','left=skyGround');
  fn=removeFalseConjunctions(fn,'cls==1u');
  fn=replaceConditional(fn,'cls==1u','remove','shadeCell');
  fn=once(fn,'if(cls==0u&&terrainRun(run)&&left==level&&right==level&&(c&3)==1)','if(terrainRun(run)&&left==level&&right==level&&(c&3)==1)');
 }else{
  fn=replaceConditional(fn,'kind==3u','body');
  fn=replaceConditional(fn,'cls==0u&&terrainRun(run)','remove','var held');
  fn=replaceConditional(fn,'cls==0u','else');
  fn=replaceConditional(fn,'cls==1u','body','foot=groupFoot');
  fn=replaceConditional(fn,'cls==0u&&terrainRun(run)','remove','continue');
  fn=fn.replace('top.on&&cls==1u&&(top.id>>9u)==(skyID(run)>>9u)','top.on&&(top.id>>9u)==(skyID(run)>>9u)');
  fn=replaceConditional(fn,'cls==1u','body','left=skyGround');
  // Only the class test is known. The foot/base condition remains a real
  // drawing decision and must not be folded along with it.
  fn=fn.replaceAll('if(cls==1u&&','if(');
  fn=replaceConditional(fn,'cls==1u','body','shadeCell');
  fn=replaceConditional(fn,'cls==0u&&terrainRun(run)&&left==level&&right==level&&(c&3)==1','remove','let below');
 }
 return fn;
}

function planEntry(name:'featurefarTerrain'|'featurefarStanding',kind:1|3):string{
 const skyline=name==='featurefarTerrain'?'farSkylineTerrain':'farSkylineStanding';
 return `fn ${name}(@builtin(global_invocation_id) id:vec3u){
 let runID=scheduledRunFamily(id,${kind===1?5:6}u);if(runID==0xffffffffu||runID>=p.extent.z){return;}
 let at=p.farLoops.w+runID*${FAR_PLAN_WORDS}u;if(atomicLoad(&counts[at])!=${kind}u){return;}
 let away=atomicLoad(&counts[at+1u]);let priority=atomicLoad(&counts[at+2u]);let span=vec4f(bitcast<f32>(atomicLoad(&counts[at+3u])),bitcast<f32>(atomicLoad(&counts[at+4u])),bitcast<f32>(atomicLoad(&counts[at+5u])),bitcast<f32>(atomicLoad(&counts[at+6u])));
 let run=runs[runID];gridKeyBase=0u;collapseDir=NONE;lineTier=away;
 if(${skyline}(runID,run,span,priority)){atomicStore(&counts[farStatus(runID)],1u);}
}
`;
}

function emitter(source:string,name:'featurefarTerrain'|'featurefarStanding',kind:1|3):string{
 const combined=farPlayback(source);
 const skyline=specializeSkyline(combined,kind);
 const code=replaceFunction(replaceFunction(combined,'farSkyline',skyline),'featurefar',planEntry(name,kind))+
  '\n'+specializeUnits(combined,kind)+'\n'+specializeColumns(combined,kind)+'\n'+specializeCell(combined,kind);
 // The standing entry is lowered again by cooperativeStanding. Keep its
 // workgroup-ordinal helper reachable until that rewrite; isolating here
 // would prune scheduledRunWorkgroup before the final entry calls it.
 return name==='featurefarStanding'?code:isolateComputeEntry(code,name);
}

export type FarKindStages={prepare:string;terrain:string;standing:string;entries:readonly[{entryPoint:'featurefarPrepare';code:string},{entryPoint:'featurefarTerrain';code:string},{entryPoint:'featurefarStanding';code:string}]};
export function buildFarKindStages(source:string):FarKindStages{
 const rawStanding=cooperativeStanding(emitter(source,'featurefarStanding',3),source),bounds=deriveFarProfileLoopBounds(rawStanding);
 const prepare=isolateComputeEntry(lowerPrepareProfileLoops(planPrepare(source),bounds),'featurefarPrepare');
 const terrain=emitter(source,'featurefarTerrain',1),standing=isolateComputeEntry(lowerStandingProfileLoops(rawStanding,bounds),'featurefarStanding');
 return {prepare,terrain,standing,entries:[{entryPoint:'featurefarPrepare',code:prepare},{entryPoint:'featurefarTerrain',code:terrain},{entryPoint:'featurefarStanding',code:standing}]};
}

export const FAR_PROFILE_LOOP_FIELDS=['edgeForward','edgeBackward','footReach','profileSides'] as const;
export type FarProfileLoopBounds=Readonly<Record<typeof FAR_PROFILE_LOOP_FIELDS[number],number>>;
function profileByte(value:number,name:string):number{
 if(!Number.isInteger(value)||value<1||value>255)throw Error(`Far loop ${name} is outside one-byte plan storage: ${value}`);
 return value;
}
/** Limits are authored by the canonical loops, not a second runtime rule.
 * Moving their identical values through the existing plan avoids pathological
 * native pipeline compilation of nested fixed profile probes on the phone. */
export function deriveFarProfileLoopBounds(source:string):FarProfileLoopBounds{
 const edge=[...functionText(source,'formEdge').matchAll(/for\(var k=1;k<=([0-9]+);k\+\+\)/g)];
 const foot=[...functionText(source,'groupFoot').matchAll(/for\(var k=1;k<=([0-9]+)&&gap<=2;k\+\+\)/g)];
 const profile=[...functionText(source,'profileUnits').matchAll(/for\(var side=0u;side<([0-9]+)u;side\+\+\)/g)];
 if(edge.length!==2)throw Error(`Expected two formEdge probe loops, got ${edge.length}`);
 if(foot.length!==1)throw Error(`Expected one groupFoot reach loop, got ${foot.length}`);
 if(profile.length!==1)throw Error(`Expected one profileUnits side loop, got ${profile.length}`);
 return {edgeForward:profileByte(Number(edge[0]![1]),'edgeForward'),edgeBackward:profileByte(Number(edge[1]![1]),'edgeBackward'),footReach:profileByte(Number(foot[0]![1]),'footReach'),profileSides:profileByte(Number(profile[0]![1]),'profileSides')};
}
export function packFarProfileLoopBounds(bounds:FarProfileLoopBounds):number{
 const v=FAR_PROFILE_LOOP_FIELDS.map(name=>profileByte(bounds[name],name));
 return (v[0]!|(v[1]!<<8)|(v[2]!<<16)|(v[3]!<<24))>>>0;
}
export function unpackFarProfileLoopBounds(word:number):FarProfileLoopBounds{
 return {edgeForward:word&255,edgeBackward:(word>>>8)&255,footReach:(word>>>16)&255,profileSides:(word>>>24)&255};
}
function lowerPrepareProfileLoops(prepare:string,bounds:FarProfileLoopBounds):string{
 let out=`const FAR_PROFILE_LOOP_PACK=${packFarProfileLoopBounds(bounds)}u;\n`+prepare;
 let plan=functionText(out,'farPlanWrite');
 plan=once(plan,'span:vec4f)','span:vec4f,loopPack:u32)');
 plan=once(plan,'atomicStore(&counts[at+7u],0u);','atomicStore(&counts[at+7u],loopPack);');
 out=replaceFunction(out,'farPlanWrite',plan);
 out=once(out,'farPlanWrite(runID,0u,0u,0u,vec4f(0));','farPlanWrite(runID,0u,0u,0u,vec4f(0),0u);');
 return once(out,'farPlanWrite(runID,kind,away,priority,span);','farPlanWrite(runID,kind,away,priority,span,FAR_PROFILE_LOOP_PACK);');
}
function lowerStandingProfileLoops(standing:string,bounds:FarProfileLoopBounds):string{
 let edge=functionText(standing,'formEdge');
 edge=once(edge,`for(var k=1;k<=${bounds.edgeForward};k++)`,'for(var k=1;k<=i32(farLoopBounds.x);k++)');
 edge=once(edge,`for(var k=1;k<=${bounds.edgeBackward};k++)`,'for(var k=1;k<=i32(farLoopBounds.y);k++)');
 let out=replaceFunction(standing,'formEdge',edge);
 out=replaceFunction(out,'groupFoot',once(functionText(out,'groupFoot'),`for(var k=1;k<=${bounds.footReach}&&gap<=2;k++)`,'for(var k=1;k<=i32(farLoopBounds.z)&&gap<=2;k++)'));
 out=replaceFunction(out,'profileUnits',once(functionText(out,'profileUnits'),`for(var side=0u;side<${bounds.profileSides}u;side++)`,'for(var side=0u;side<farLoopBounds.w;side++)'));
 let entry=functionText(out,'featurefarStanding');
 const decode=' farLoopBounds=vec4u(0u);if(valid){let packed=atomicLoad(&counts[at+7u]);farLoopBounds=vec4u(packed&255u,(packed>>8u)&255u,(packed>>16u)&255u,(packed>>24u)&255u);}\n';
 const atAnchor=' let safeRunID=select(0u,runID,valid);let at=p.farLoops.w+safeRunID*8u;\n';
 entry=once(entry,atAnchor,atAnchor+decode);
 out=replaceFunction(out,'featurefarStanding',entry);
 return once(out,'var<private> farCacheBase:i32;','var<private> farCacheBase:i32;\nvar<private> farLoopBounds:vec4u;');
}

export function standingDispatch(runCount:number,limit:number):readonly[number,number]{
 if(!Number.isSafeInteger(runCount)||runCount<1||!Number.isSafeInteger(limit)||limit<1)throw Error('Invalid far workgroup extent');
 const rows=Math.ceil(runCount/limit),columns=Math.ceil(runCount/rows);
 if(rows>limit)throw Error('Far workgroup extent exceeds device dispatch dimensions');
 return [columns,rows];
}

/** Cooperative execution compiler, not a copied drawing producer. All field
 * values and emission decisions come from the existing source functions. */
function cooperativeStanding(standing:string,canonical:string):string{
 const max=Number(/const SKY_COLUMNS=(\d+)u/.exec(standing)?.[1]);if(!Number.isSafeInteger(max)||max<1)throw Error('Missing skyline column extent');
 const cells=max+2;
 const [fa,fb]=functionRange(canonical,'farForm'),form=canonical.slice(fa,fb),open=form.indexOf('{');
 const until=form.indexOf(' let height=base-top;');if(until<0)throw Error('Far form base dependency changed');
 const base=`fn farBase(run:Run,span:vec4f)->f32{${form.slice(open+1,until)}return base;}`;
 // The canonical field expression is used, not re-authored base arithmetic.
 standing=once(standing,'farForm(run,span).base','farBase(run,span)');
 const [ca,cb]=functionRange(standing,'farCellStanding');
 const compute=standing.slice(ca,cb).replace('fn farCellStanding(','fn farCellStandingCompute(');
 standing=replaceFunction(standing,'farCellStanding',`fn farCellStanding(run:Run,c:i32)->SkyCell{return farCachedCells[u32(c-farCacheBase)];}`);
 // Compiled source-part parents have real projected source geometry in their
 // point span.  Keep their native role passage on the same proposal path as
 // ordinary soft playback instead of feeding the part through the class
 // neutral skyline profile.  The bridge deliberately asks the resident soft
 // table for the role/context record; `entryRef` is an opaque source token and
 // is never treated as an index into the native grammar bank.
 const nativeBridge=`
fn farPartDirection(delta:vec2f)->u32{return u32(round(fract(atan2(delta.y,delta.x)/6.28318530718+1.0)*8.0))&7u;}
fn farPartContext(run:Run)->u32{
 let contextClass=(run.part.y>>6u)&7u;if(contextClass==0u||run.range.y<2u){return 0u;}
 let base=run.range.x;let count=run.range.y/2u;
 let first=points[base].position.xy;let firstEnd=points[base+1u].position.xy;
 let last=points[base+(count-1u)*2u].position.xy;let lastEnd=points[base+(count-1u)*2u+1u].position.xy;
 return farPartDirection(first-firstEnd)|(farPartDirection(lastEnd-last)<<3u)|(contextClass<<6u);
}
fn farPartRowsFit(record:u32,sourceRows:i32,sourceWidth:u32,origin:i32)->bool{
 let width=grammar[record+4u]&0xffffu;let authoredRows=i32(grammar[record+5u]);if(sourceWidth<width||sourceRows<authoredRows||sourceRows<=0){return false;}
 if(sourceRows>authoredRows){if(nativeWholeRow(record,0)==NONE){return false;}return wholePartHeightRowsFit(record,vec4i(origin,0,i32(sourceWidth),sourceRows));}
 if(nativeWholeRow(record,0)!=NONE){return nativeWholeRowsFit(record,vec4i(origin,0,i32(sourceWidth),sourceRows));}
 let first=grammar[record];let atoms=grammar[record+1u];
 for(var row=0;row<sourceRows;row++){
  let growth=wholePartRowGrowth(record,row,sourceWidth-width);if(growth.w==NONE||growth.x==NONE){return false;}
  for(var atom=0u;atom<atoms;atom++){
   let at=grammar[1]+(first+atom)*4u;if(bitcast<i32>(grammar[at+2u])!=row){continue;}let x=grammar[at+1u];
   let repeats=select(growth.w+1u,1u,x==growth.x||x==growth.y);
   for(var repeat=0u;repeat<repeats;repeat++){let pageX=origin+i32(wholePartAtomX(x,growth,repeat));if(pageX<0||pageX+i32(grammar[at+3u])>i32(p.extent.x)){return false;}}
  }
 }
 return true;
}
// The projected whole passage chooses among its three compiled related bands,
// not always band zero. A rejected variant cannot leave partial proposals.
fn farPartRecord(role:u32,context:u32,sourceRows:i32,sourceWidth:u32,origin:i32)->u32{
 for(var band=2;band>=0;band--){
  let record=softPartRecord(role,u32(band),context);if(record==NONE){continue;}
  if(farPartRowsFit(record,sourceRows,sourceWidth,origin)){return record;}
 }
 return NONE;
}
// Return true only after the complete retained row passage is queued.  Any
// unsupported context, bad native row, or too-narrow source span returns
// false so the caller keeps the incumbent connected far skyline.
// Relation and relation-parent runs are native boundary passages, never a
// skyline/profile paper fill. Their marker is emitted in the existing role
// word, so this helper keeps the canonical three-argument consumer ABI while
// still preventing a global paper mark from erasing an actor through a hole.
fn farNativePart(runID:u32,run:Run,priority:u32)->bool{
 let boundaryOnly=(run.paper.w&0xa0000000u)!=0u;
 let feature=run.range.w>>5u;if(feature==0u||run.range.y<2u||run.paper.y==0u){return false;}
 let context=farPartContext(run);if(context==0u){return false;}
 let role=run.paper.w&3u;
 let base=run.range.x;let count=run.range.y/2u;var minY=1e30;var maxY=-1e30;var minX=1e30;var maxX=-1e30;
 for(var pair=0u;pair<count;pair++){let a=points[base+pair*2u].position;let b=points[base+pair*2u+1u].position;minY=min(minY,min(a.y,b.y));maxY=max(maxY,max(a.y,b.y));minX=min(minX,min(a.x,b.x));maxX=max(maxX,max(a.x,b.x));}
 if(!(minY<maxY)||!(minX<maxX)){return false;}
 let firstRow=max(0,i32(floor(minY/17.0)));let lastRow=min(i32(rowCount())-1,i32(floor(maxY/17.0)));let sourceRows=lastRow-firstRow+1;
 if(sourceRows<2||sourceRows>${FAR_PARENT_ROW_PLAN_MAX}){return false;}
 let sourceWidth=u32(max(0.0,round(maxX-minX)));
 let record=farPartRecord(role,context,sourceRows,sourceWidth,i32(round(minX)));if(record==NONE){return false;}
 let width=grammar[record+4u]&0xffffu;
 let first=grammar[record];let atoms=grammar[record+1u];
 var rowPlan:array<vec4f,${FAR_PARENT_ROW_PLAN_MAX}>;
 for(var init=0u;init<${FAR_PARENT_ROW_PLAN_MAX}u;init++){rowPlan[init]=vec4f(1e30,-1e30,0.0,0.0);}
 // Attach each projected source segment to the rows it actually crosses,
 // preserving source segment order.  No nearest-visible-fragment score is
 // recomputed per row, so correspondence survives visibility changes.
 for(var pair=0u;pair<count;pair++){
  let a=points[base+pair*2u].position;let b=points[base+pair*2u+1u].position;let low=max(firstRow,i32(floor(min(a.y,b.y)/17.0)));let high=min(lastRow,i32(floor(max(a.y,b.y)/17.0)));
  for(var pageRow=low;pageRow<=high;pageRow++){
   let index=u32(pageRow-firstRow);let y=f32(pageRow)*17.0+8.5;let t=select(.5,clamp((y-a.y)/(b.y-a.y),0.0,1.0),abs(b.y-a.y)>.00001);let x=mix(a.x,b.x,t);var entry=rowPlan[index];entry.x=min(entry.x,x);entry.y=max(entry.y,x);if(entry.w==0.0){entry.z=f32(pair+1u);entry.w=1.0;}rowPlan[index]=entry;
  }
 }
 for(var plan=0u;plan<${FAR_PARENT_ROW_PLAN_MAX}u;plan++){if(plan>=u32(sourceRows)){break;}if(rowPlan[plan].w==0.0||rowPlan[plan].x>rowPlan[plan].y){return false;}}
 var emitted=false;
 for(var sourceRow=0u;sourceRow<${FAR_PARENT_ROW_PLAN_MAX}u;sourceRow++){
  if(sourceRow>=u32(sourceRows)){break;}
  let pageRow=firstRow+i32(sourceRow);let nativeRow=i32(sourceRow);let growth=wholePartRowGrowth(record,nativeRow,sourceWidth-width);
  let extent=rowPlan[sourceRow];let pair=min(u32(max(0.0,extent.z-1.0)),count-1u);var q=interpolatePoint(points[base+pair*2u],points[base+pair*2u+1u],.5);q.w=points[base+pair*2u].position.w;
  if(nativeWholeRow(record,0)!=NONE){
   let whole=vec4i(i32(round(minX)),firstRow,i32(sourceWidth),sourceRows);
   let words=nativeWholeEmit(record,whole,i32(sourceRow),pageRow,runID,q,priority,points[base+pair*2u],points[base+pair*2u+1u],!boundaryOnly);
   emitted=emitted||words>0u;continue;
  }
  for(var atom=0u;atom<atoms;atom++){
   let at=grammar[1]+(first+atom)*4u;if(bitcast<i32>(grammar[at+2u])!=nativeRow){continue;}let x=grammar[at+1u];
   let repeats=select(growth.w+1u,1u,x==growth.x||x==growth.y);
   for(var repeat=0u;repeat<repeats;repeat++){
    let pageX=i32(round(minX))+i32(wholePartAtomX(x,growth,repeat));
    // Match the near source-part consumer: the last anchor keeps its member
    // identity when whole middle rows are inserted above it.
    let member=wholePartMember(sourceRow,u32(sourceRows),x,growth,repeat,atom,atoms);
    spanPartWord(runID,member,q,pageX,pageRow,grammar[at],1u,grammar[at+3u],priority,pair,1u);emitted=true;
   }
  }
  if(!boundaryOnly){markSoftPartPaper(record,i32(round(minX)),pageRow,nativeRow,growth);}
 }
 return emitted;
}
fn farRelationPart(runID:u32,run:Run,priority:u32)->bool{
 let relation=(run.paper.w&0x80000000u)!=0u;if(!relation||run.range.y<2u){return false;}
 // Contacts and protected openings share this boundary-only native path. The
 // source program has already retained the ordered relation geometry; no
 // skyline fallback is legal when native lowering fails.
 return farNativePart(runID,run,priority);
}
// A compound parent is one coherent source product: ordinary outer passages
// carry the parent-paper marker, while opening/contact passages carry their
// relation marker.  Route both through the native parent consumer before any
// skyline fallback.  This is deliberately a consumer-side cutover, not a
// second selector or a per-piece glyph choice; an unsupported product returns
// false so the source selector's complete child representation remains the
// legal incumbent.
fn farCompoundParent(runID:u32,run:Run,priority:u32)->bool{
 let compound=(run.paper.w&0x10000000u)!=0u;if(!compound){return false;}
 if((run.paper.w&0x80000000u)!=0u){return farRelationPart(runID,run,priority);}
 return farNativePart(runID,run,priority);
}
`;
 const entry=`fn featurefarStanding(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let runID=scheduledRunWorkgroup(group,6u);
 let valid=runID!=0xffffffffu&&runID<p.extent.z;
 let safeRunID=select(0u,runID,valid);let at=p.farLoops.w+safeRunID*8u;
 // Every invocation reaches both barriers. Padded indirect groups and plans
 // of another far kind publish an inactive marker instead of returning before
 // the cache barrier, which Tint requires to be uniform control flow.
 if(lane==0u){farWorkgroupKind=0u;if(valid){farWorkgroupKind=atomicLoad(&counts[at]);}}
 workgroupBarrier();
 let isStanding=workgroupUniformLoad(&farWorkgroupKind)==3u;
 var run=Run(vec4u(0u),vec4u(0u),vec4u(0u),vec4u(0u),vec4u(0u));
 var span=vec4f(0.0);var priority=0u;
 if(isStanding){
  let away=atomicLoad(&counts[at+1u]);priority=atomicLoad(&counts[at+2u]);
  span=vec4f(bitcast<f32>(atomicLoad(&counts[at+3u])),bitcast<f32>(atomicLoad(&counts[at+4u])),bitcast<f32>(atomicLoad(&counts[at+5u])),bitcast<f32>(atomicLoad(&counts[at+6u])));
  run=runs[safeRunID];gridKeyBase=0u;collapseDir=NONE;lineTier=away;
  let columns=farColumnsStanding(run,span);farCacheBase=columns.x-1;
  let n=u32(max(0,columns.y-columns.x+3));
  // Keep the loop bound uniform so the following barrier remains in uniform
  // control flow. The lane selects its cell inside the fixed-stride loop.
  for(var i=0u;i<n;i+=64u){let idx=i+lane;if(idx<n){farCachedCells[idx]=farCellStandingCompute(run,farCacheBase+i32(idx));}}
 }
 workgroupBarrier();
 if(!isStanding||lane!=0u){return;}
 if(run.correspondence.w==0x534d414cu){
  let sourceAt=run.range.x+run.part.z*2u;
  if(sourceAt+1u<arrayLength(&points)){
   let sourceLow=points[sourceAt].position.xy;var smallRun=run;
   // A local view supplies native layout without overwriting the original
   // profile's packed top/bottom or its drawable point range.
   smallRun.paper=vec4u(bitcast<u32>(i32(round(sourceLow.x))),bitcast<u32>(i32(round(sourceLow.y/17.0))),run.part.y,run.part.w);
   if(smallConstructionPlayback(safeRunID,smallRun,priority)){atomicStore(&counts[farStatus(safeRunID)],1u);return;}
  }
 }
 if((run.paper.w&0x10000000u)!=0u){if(farCompoundParent(safeRunID,run,priority)){atomicStore(&counts[farStatus(safeRunID)],1u);}return;}
 // Preserve the incumbent relation-only path for older products. The explicit
 // compound bit above is the only branch that suppresses skyline fallback for
 // ordinary outer parts; relation-free and legacy relation parents retain the
 // existing consumer behavior.
 if((run.paper.w&0x80000000u)!=0u){if(farRelationPart(safeRunID,run,priority)){atomicStore(&counts[farStatus(safeRunID)],1u);}return;}
 let relationParent=(run.paper.w&0x20000000u)!=0u;
 if(farNativePart(safeRunID,run,priority)){atomicStore(&counts[farStatus(safeRunID)],1u);return;}
 if(farSkylineStanding(safeRunID,run,span,priority)){atomicStore(&counts[farStatus(safeRunID)],1u);}
}`;
 standing=replaceFunction(standing,'featurefarStanding',entry);
 const wordState=[
  ['wholeGrowthKey','var<private> wholeGrowthKey:u32=NONE;'],
  ['wholeGrowthCached','var<private> wholeGrowthCached:u32=NONE;'],
  ['wholeWordBase','var<private> wholeWordBase:u32=NONE;'],
  ['wholeWordChecked','var<private> wholeWordChecked:bool=false;'],
  ['wholeHeightCacheKey','var<private> wholeHeightCacheKey:vec3u=vec3u(NONE,NONE,NONE);'],
  ['wholeHeightCache','var<private> wholeHeightCache:vec4i=vec4i(-1,-1,0,0);'],
  ['wholeHeightCacheMember','var<private> wholeHeightCacheMember:u32=NONE;'],
  ['wholeHeightCachePivot','var<private> wholeHeightCachePivot:i32=0;'],
 ].filter(([name])=>!standing.includes('var<private> '+name+':')).map(([,declaration])=>declaration).join('\n');
 const wordConstant=standing.includes('const NATIVE_ROW_NONE:')?'':'\nconst NATIVE_ROW_NONE:u32=0xffffffffu;\n';
 const declarations=`var<workgroup> farCachedCells:array<SkyCell,${cells}>;\nvar<workgroup> farWorkgroupKind:u32;\nvar<private> farCacheBase:i32;\n`+wordState+wordConstant;
 const canonicalFunctions=retainCanonicalFunctions(canonical,standing,['softPartRecord','markSoftPartPaper','wholePartRowGrowth','wholePartAtomX','wholePartMember','spanPartWord','spanPartSourceWord','interpolatePoint',
  'wholePartHeightRowsFit','nativeWholeRowsFit','nativeWholeEmit','nativeWholeRow']);
 return isolateComputeEntry(standing+'\n'+nativeBridge+'\n'+SMALL_CONSTRUCTION_WGSL+'\n'+canonicalFunctions+'\n'+base+'\n'+compute+'\n'+declarations+'\n'+WORKGROUP_SCHEDULER_WGSL,'featurefarStanding');
}
