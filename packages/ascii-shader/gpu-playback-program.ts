import {isolateComputeEntry} from './gpu-compute-program.ts';

export const PLAYBACK_FAMILIES=['path','planar','round'] as const;
export type PlaybackFamily=typeof PLAYBACK_FAMILIES[number];
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
function replace(source:string,before:string,after:string):string{
 if(source.split(before).length!==2)throw Error('Playback compiler anchor changed: '+before.slice(0,80));
 return source.replace(before,after);
}
// A separate per-source word immediately before the existing selection grids.
// Loss/copy record offsets and publication ABI are unchanged.
export const FAR_STATUS_WGSL=`fn farStatus(runID:u32)->u32{return drawnBase()+p.playback.z+p.extent.z*2u+runID;}
`;
export function familyPlayback(source:string,family:PlaybackFamily):string{
 let code=source+'\n'+FAR_STATUS_WGSL;
 const queueFamily=PLAYBACK_FAMILIES.indexOf(family)+1;
 code=replace(code,'@compute @workgroup_size(64) fn featurecurves(@builtin(global_invocation_id) id:vec3u){\n let runID=scheduledRun(id);',`@compute @workgroup_size(64) fn featurecurves(@builtin(global_invocation_id) id:vec3u){\n let runID=scheduledRunFamily(id,${queueFamily}u);`);
 code=replace(code,' let boundary=wholeBoundaryFlags(run);\n',` let boundary=wholeBoundaryFlags(run);
 let family=select(select(0u,2u,(boundary&0x20000u)!=0u),1u,(boundary&0x4100000u)!=0u);
 if(family!=${PLAYBACK_FAMILIES.indexOf(family)}u){return;}
`);
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
// Closed far playback never writes junctionHold. Its zero private array
// cannot reject a word, but retaining its loop expands every inlined emit.
// A future writer/escape/initializer makes this fold inapplicable, not a
// missing drawing: keep that complete original program unchanged.
const ZERO_FAR_JUNCTION_READ=" for(var side=0u;side<JUNCTION_HOLDS;side++){let held=junctionHold[side];let centre=2*x+i32(advance);if(held.x<held.y&&(row==held.z||row==held.w)&&centre>=held.x&&centre<=held.y){return;}}\n";
function foldUnwrittenFarJunction(closed:string):string{
 const code=closed.replace(/\/\*[\s\S]*?\*\//g,'').replace(/\/\/[^\n]*/g,'');
 if(!code.includes('var<private> junctionHold:array<vec4i,JUNCTION_HOLDS>;')||[...code.matchAll(/\bjunctionHold\b/g)].length!==2||closed.split(ZERO_FAR_JUNCTION_READ).length!==2)return closed;
 return isolateComputeEntry(closed.replace(ZERO_FAR_JUNCTION_READ,'\n'),'featurefar');
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
 return foldUnwrittenFarJunction(closed);
}
