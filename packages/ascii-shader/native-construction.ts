import roles from '../../assets/grammar/aahub-role-lookup-3493185.v1.json' with {type:'json'};
import {nativeScalarFont} from './native-font.ts';
import {NativePage,NativeCommitError,nativePaperContains,nativeWholeInk,type NativeGlyph,type NativePaper,type NativeNeighborhood,type NativeOccurrence,type NativeFrame as NativePageFrame} from './native-page.ts';
import {nativeSpacePlan,type NativeWhitespacePolicy} from './text/native-whitespace.ts';
import {Composition,fieldOwner,type CompositionGeometry,type CompositionSubject} from './composition.ts';
import {relationPoint,relationParameter,posedOwnerId,type PosedRelation,type PosedGraph,type PosedOwner} from './posed-graph.ts';
import type {DrawingFieldView} from './field-readback.ts';
import {constructNativeLabels,type NativeLabel} from './semantic-text.ts';
import type {NativeSupport} from './native-presentation.ts';
import {PlanarConstruction} from './planar-construction.ts';
import {PlaneCapFan,type FanState} from './plane-cap-fan.ts';
import {nativeProgramPhase,type NativeCursorConstraint,type NativePattern} from './native-layout.ts';
import {ConstructionVisibility} from './construction-visibility.ts';
import {RoundConstruction} from './round-construction.ts';
import {LimbConstruction} from './limb-construction.ts';
import {NativeSurfaceVisibility} from './native-surface.ts';
import {terminalAdmission,type TerminalRequest} from './terminal-admission.ts';
import {RetainedProducers} from './retained-producers.ts';
import {NativePaperIndex} from './native-paper-index.ts';
import {RegionTone} from './region-tone.ts';
import {placeRegionTone,TONE_LADDER,type TonedFieldView} from './tone-field.ts';
import {ConstructionRuns} from './construction-runs.ts';
import type {ArticulationResult,ArticulationRun,ArticulationLiteral} from './gpu-articulation.ts';
export type {NativeGlyph} from './native-page.ts';
export interface GraphLine {readonly a:readonly number[];readonly b:readonly number[];readonly key:readonly number[];readonly ports:readonly [string,string];readonly owner:string}
export interface NativeWord {readonly x:number;readonly row:number;readonly text:string;readonly advance:number;readonly features:readonly string[];readonly role:string}
export interface ConstructionMetrics {
 readonly compositionMs:number;readonly realizationMs:number;readonly cpuFrameMs:number;readonly cpuFrameScope:string;
 readonly subjects:number;readonly sourceSubjects:number;readonly farStripCount:number;readonly publishedFarStripCount:number;
 readonly layoutDroppedShare:number|null;readonly dotShare:number;readonly repeatedRampTokens:number;readonly wholeEntries:readonly string[];
}
export interface NativeFrame extends NativePageFrame {
 readonly words:readonly NativeWord[];readonly sourceLines:number;readonly omitted:readonly string[];
 readonly supportOwners:ReadonlyMap<string,readonly string[]>;readonly supportRelations:ReadonlyMap<string,readonly string[]>;
 readonly directSupports:ReadonlyMap<string,NativeSupport>;readonly metrics:ConstructionMetrics;
}
interface Entry {id:string;native:{rows:readonly string[];row_pen_x:readonly (number|null)[]};status:{state:string}}
interface Block {id:string;paper:readonly NativePaper[];atoms:readonly {text:string;x:number;row:number;scalar:number}[];bounds:readonly [number,number,number,number]}
const ROUND_IDS=['sh-lib66-circle-2row','sh-lib60-circle-3row','sh-lib64-circle-4row','sh-lib67-circle-2row-tiny','sh-lib65-circle-3row-brace','sh-lib63-circle-3row-soft'] as const;
/** Whole native constructions are realized only AFTER Composition. NativePage
 * seals their common scalar tape; a missing/illegal construction is a visible
 * failure, never a successful blank or permission to hide a source form. */
export class NativeConstruction {
 private readonly composition=new Composition();private readonly font:Map<string,NativeGlyph>;private readonly page:NativePage;
 private readonly blocks=new Map<string,Block>();private readonly owners=new Map<string,PosedOwner>();
 private readonly whitespace:NativeWhitespacePolicy;
 private readonly planar:PlanarConstruction;private readonly planarPhases=new Map<string,number>();
 private readonly planePrograms:PlaneCapFan;private readonly planeStates=new Map<string,FanState>();
 private readonly planarVisibility:ConstructionVisibility;
 private readonly roundPrograms:RoundConstruction;
 private readonly limbPrograms:LimbConstruction;
 private readonly retained:RetainedProducers;
 private readonly tone:RegionTone;
 private readonly playback=new ConstructionRuns();
 constructor(glyphs:readonly NativeGlyph[],whitespace:NativeWhitespacePolicy='preformatted'){
  glyphs=nativeScalarFont(glyphs);this.font=new Map(glyphs.map(g=>[g.text,g]));this.page=new NativePage(glyphs,whitespace);this.whitespace=whitespace;
  this.planar=new PlanarConstruction(glyphs,whitespace);
  this.planePrograms=new PlaneCapFan(glyphs);
  this.planarVisibility=new ConstructionVisibility(glyphs);
  this.roundPrograms=new RoundConstruction(glyphs);
  this.limbPrograms=new LimbConstruction(glyphs);
  this.retained=new RetainedProducers(glyphs,whitespace);
  this.tone=new RegionTone(glyphs,whitespace);
  for(const id of ROUND_IDS){const entry=(roles.entries as unknown as Entry[]).find(e=>e.id===id);if(!entry||entry.status.state!=='retained')throw Error(`Unavailable whole retained construction ${id}`);
   const atoms:Block['atoms'][number][]=[],points:number[][]=[];const base=Math.min(...entry.native.row_pen_x.filter((p):p is number=>p!==null));
   for(const [row,encoded]of entry.native.rows.entries()){let x=(entry.native.row_pen_x[row]??base)-base;const text=encoded.replace(/\\u([0-9a-f]{4})/gi,(_,cp:string)=>String.fromCharCode(parseInt(cp,16)));
    for(const [scalar,ch]of [...text].entries()){const g=this.font.get(ch);if(!g)throw Error(`Unavailable native scalar ${ch} in ${id}`);if(g.points.length){atoms.push({text:ch,x,row,scalar});for(const p of g.points)points.push([x+p[0]!,row*17+p[1]!]);}x+=g.advance;}}
   const paper:NativePaper[]=[];
   for(let row=0;row<entry.native.rows.length;row++){
    const members=atoms.filter(a=>a.row===row);for(let i=1;i<members.length;i++){const left=members[i-1]!,right=members[i]!,a=left.x+Math.max(...this.font.get(left.text)!.points.map(p=>p[0]!))+1,b=right.x+Math.min(...this.font.get(right.text)!.points.map(p=>p[0]!));if(b-a>=5)paper.push({id:`${id}/paper/${row}/${i}`,rect:[a,row*17,b-a,17],sources:[id]});}}
   this.blocks.set(id,{id,atoms,paper,bounds:[Math.min(...points.map(p=>p[0]!)),Math.min(...points.map(p=>p[1]!)),Math.max(...points.map(p=>p[0]!))+1,Math.max(...points.map(p=>p[1]!))+1]});
  }
 }
 selectSources(geometry:readonly CompositionGeometry[]):ReadonlySet<string>{return this.composition.chooseSources(geometry);}
 compiledSource(source:CompositionGeometry){return this.composition.compiledSource(source);}
 installSource(source:CompositionGeometry,analysis:Parameters<Composition['installSource']>[1]):void{this.composition.installSource(source,analysis);}
 playbackRuns(graphs:readonly PosedGraph[],geometry:readonly CompositionGeometry[]):readonly ArticulationRun[]{for(const graph of graphs)for(const [id,owner]of graph.owners??[])this.owners.set(id,owner);return this.playback.compile(graphs,this.composition.compiledGraphs(geometry));}
 playbackLiterals(labels:readonly NativeLabel[],fields:readonly DrawingFieldView[],graphs:readonly PosedGraph[]):readonly ArticulationLiteral[]{
  // Authored strings still reflow in their native metrics. Their visibility
  // is evaluated from the same-pose GPU fields by featureplace; compiled
  // playback never asks the CPU for a surface pixel.
  const text=constructNativeLabels(labels,[...this.font.values()],[]),result:ArticulationLiteral[]=[];
  for(const pattern of text.patterns){const label=labels.find(label=>label.id===pattern.id)!;
   result.push({camera:label.camera,depth:label.depthBias>=0?label.anchor[2]:undefined,support:text.supports.get(pattern.id)!,group:{...pattern,occurrences:pattern.occurrences.map(atom=>({...atom,x:atom.x+pattern.origin[0],row:atom.row+pattern.origin[1]}))}});
  }
  const owners=new Map(graphs.flatMap(graph=>[...graph.owners??[]]));
  for(const view of fields as readonly TonedFieldView[]){
   const byOwner=new Map<string,NativeOccurrence[]>();
   for(const member of [...view.tone??[]].sort((a,b)=>(a.key??'').localeCompare(b.key??''))){
    // The current tone field has receiver provenance, not a bounded air
    // volume/effect owner. Never turn an old/background atmosphere member
    // into a camera-wide tone group. Surface fog/exposure still uses its
    // actual receiver below; backdrop stays paper.
    if(member.atmosphere)continue;
    const word=TONE_LADDER[member.level],glyph=word&&this.font.get(word);if(!glyph||!member.source)continue;
    const [encoded,index]=member.source,kind=encoded&0x80000000?'producer-row':'entity',domain=encoded&0x5fffffff;
    const reference=kind==='entity'?view.owners.get(index):index;
    if(reference===undefined)continue;
    const owner=posedOwnerId({camera:view.camera,domain,kind,lifetime:reference});
    // The same whole-form selection that supplies its paths owns the
    // smallest tier. Do not add surface dots to its outline-only drawing;
    // no line/anchor producer or glyph vocabulary is removed by this rule.
    if(!this.playback.allowsTone(owner))continue;
    // A member centred near the view edge must not leave the page: one such
    // atom makes page.commit reject the whole frame (0 glyphs).
    const x=Math.round(member.x-glyph.advance/2),{rect}=view;
    if(x<Math.max(0,Math.ceil(rect.x))||x+glyph.advance>Math.floor(rect.x+rect.width)||member.row*17<rect.y||member.row*17+17>rect.y+rect.height)continue;
    const source=`${owner}/tone`,atoms=byOwner.get(owner)??[];byOwner.set(owner,atoms);
    atoms.push({id:`${source}/${member.key}`,text:word,x,row:member.row,...(member.foreground===undefined?{}:{foreground:member.foreground}),contributions:[{source,kind:'region',pixels:glyph.points.map((_,i)=>i)}]});
    if(!owners.has(owner))owners.set(owner,{camera:view.camera,domain,kind,index,lifetime:reference});
   }
   for(const [owner,occurrences]of [...byOwner].sort(([a],[b])=>a.localeCompare(b))){const source=owners.get(owner),id=`${owner}/tone`;
    const support:NativeSupport={configuration:id,points:new Map(occurrences.map(atom=>[atom.id,[atom.x,atom.row*17] as const])),owners:source?[{source:source.domain,reference:source.lifetime,kind:source.kind}]:[]};
    result.push({camera:view.camera,tone:true,owners:source?[owner]:[],sources:[],support,group:{id,occurrences,contacts:[],paper:[],requiredSources:[id]}});
   }
  }
  return result;
 }
 dispose():void{this.composition.dispose();}
 retire(isLive:(owner:PosedOwner)=>boolean):void{for(const [id,owner]of this.owners)if(!isLive(owner)){this.owners.delete(id);this.planarPhases.delete(`${id}/P1`);}this.composition.retire(id=>this.owners.has(id));this.retained.retire(id=>this.owners.has(id));this.playback.retire(id=>this.owners.has(id));}
 construct(lines:readonly GraphLine[]|number,width:number,height:number,serial:number,graphs:readonly PosedGraph[]=[],fields:readonly DrawingFieldView[]=[],labels:readonly NativeLabel[]=[],geometry:readonly CompositionGeometry[]=[],playback?:ArticulationResult):NativeFrame{
  if(playback)return this.commitPlayback(lines,width,height,serial,graphs,playback);
  const started=performance.now(),composition=this.composition.select(graphs,fields,geometry);for(const [id,o]of composition.owners)this.owners.set(id,o);
  this.tone.begin(fields);
  const groups:NativeNeighborhood[]=[],supports=new Map<string,NativeSupport>(),supportOwners=new Map<string,readonly string[]>(),supportRelations=new Map<string,readonly string[]>(),omitted:string[]=[],entries:string[]=[];
  const intents=new Map<string,{pattern:NativePattern;owners:readonly string[];visibleAt?:ReadonlyMap<string,(pen:number)=>boolean>}>();
  const deferred=new Map<string,{group:NativeNeighborhood;support:NativeSupport;owners:readonly string[];sources:readonly string[]}>();
  const selectVisible=(...args:Parameters<ConstructionVisibility['select']>)=>{
   const result=this.planarVisibility.select(...args);
   for(const pattern of result.patterns)intents.set(pattern.id,{pattern,owners:args[1].get(pattern.id)??[]});
   for(const pattern of result.terminalPatterns)intents.set(pattern.id,{pattern,owners:args[1].get(pattern.id)??[],visibleAt:result.terminalVisibility});
   return result;
  };
  // Native legality is decided at this single, direct placement. Accepted
  // groups are immutable: publication cannot choose another phase or trim ink.
  let proposedGlyphs=0,rejectedGlyphs=0;
  const rows=new Map<number,NativeOccurrence[]>(),ink=new Set<string>();
  const paperIndex=new NativePaperIndex(width,height);
  let staged:{group:NativeNeighborhood;rows:Map<number,NativeOccurrence[]>}|undefined;
  const stage=(group:NativeNeighborhood):Map<number,NativeOccurrence[]>|undefined=>{
   if(staged?.group===group)return staged.rows;
   const merged=new Map<number,NativeOccurrence[]>();
   for(const a of group.occurrences){const g=this.font.get(a.text)!;if(a.x<0||a.row<0||a.x+g.advance>width||(a.row+1)*17>height)return;
    const list=merged.get(a.row)??[...(rows.get(a.row)??[])];
    // NativePage merges exact shared occurrences into one intact scalar with
    // both contributions. Admission must use that same drawing unit rather
    // than reject its shared ink or count two copies on the scalar tape.
    const shared=list.find(b=>b.x===a.x&&b.text===a.text&&(b.foreground??0xff000000)===(a.foreground??0xff000000));
    if(!shared){for(const p of g.points){const px=a.x+p[0]!,py=a.row*17+p[1]!,key=`${px}/${py}`;if(ink.has(key)||paperIndex.contains(px+.5,py+.5))return;}list.push(a);}
    merged.set(a.row,list);}
   for(const [row,list]of merged){list.sort((a,b)=>a.x-b.x||a.id.localeCompare(b.id));let x=0;for(const a of list){if(!nativeSpacePlan(a.x-x,0,x===0,false,this.whitespace))return;x=a.x+this.font.get(a.text)!.advance;}
    // Incumbent ink already satisfied incumbent paper at admission. Only
    // this new group's paper must be checked against the merged row here;
    // its new ink was checked against indexed incumbent paper above.
    for(const paper of group.paper)if(list.some(a=>this.font.get(a.text)!.points.some(p=>nativePaperContains(paper,a.x+p[0]!+.5,row*17+p[1]!+.5))))return;}
   // New protected paper can cover a row where current visibility removed
   // ALL of this program's glyphs. That row is absent from `merged`, but may
   // still contain an incumbent neighbor. Check those real rows before any
   // mutation: otherwise NativePage throws AFTER other valid forms were built.
   // This is not a D1 substitute: an unplaceable new part remains a reported
   // construction failure while the serving neighborhoods stay publishable.
   for(const paper of group.paper){const [,y,,h]=paper.rect;
    for(const [row,atoms]of rows)if(row>=Math.floor(y/17)&&row*17<y+h&&!merged.has(row)){
     if(atoms.some(a=>this.font.get(a.text)!.points.some(p=>nativePaperContains(paper,a.x+p[0]!+.5,row*17+p[1]!+.5))))return;
    }
   }
   staged={group,rows:merged};return merged;
  };
  const canAdmit=(group:NativeNeighborhood)=>stage(group)!==undefined;
  const admit=(group:NativeNeighborhood,support:NativeSupport,owner:readonly string[],sources:readonly string[]):boolean=>{
   proposedGlyphs+=group.occurrences.length;const merged=stage(group);
   if(!merged){deferred.set(group.id,{group,support,owners:owner,sources});rejectedGlyphs+=group.occurrences.length;return false;}
   for(const [row,list]of merged)rows.set(row,list);for(const a of group.occurrences)for(const p of this.font.get(a.text)!.points)ink.add(`${a.x+p[0]!}/${a.row*17+p[1]!}`);
   for(const paper of group.paper)paperIndex.add(paper);
   staged=undefined;groups.push(group);supports.set(group.id,support);supportOwners.set(group.id,owner);supportRelations.set(group.id,sources);return true;
  };
  const makeSupport=(id:string,owner:string,configuration:string,points:ReadonlyMap<string,readonly [number,number]>,native:NativeNeighborhood):NativeSupport=>{
   const o=composition.owners.get(owner);if(!o)throw Error(`Composition lost source owner ${owner}`);
   const first=native.occurrences[0],x=first?.x??0,y=(first?.row??0)*17;
   // A size-band name alone cannot certify retained text: legal pen choices,
   // visibility, shared ink and protected paper may have changed WITHIN that
   // band. Compare the actual complete native construction, modulo its one
   // origin, before NativePresentation is allowed to transport it unchanged.
   const shape=[native.occurrences.map(a=>[a.id,a.text,a.x-x,a.row*17-y,a.contributions]),
    native.paper.map(p=>[p.id,p.rect[0]-x,p.rect[1]-y,p.rect[2],p.rect[3],p.sources]),native.requiredSources];
   return {configuration:configuration+'\n'+JSON.stringify(shape),points,owners:[{reference:o.lifetime,source:o.domain,kind:o.kind}]};
  };
  const text=constructNativeLabels(labels,[...this.font.values()],fields);for(const reason of text.unresolved)omitted.push(reason);
  for(const p of text.patterns){const group={...p,occurrences:p.occurrences.map(a=>({...a,x:a.x+p.origin[0],row:a.row+p.origin[1]}))};if(!admit(group,text.supports.get(p.id)!,[],[]))omitted.push(`${p.id}:whole authored label has no legal native placement`);}
  const makePlanar=(s:CompositionSubject)=>{
   const incumbent=this.planar.construct(s,composition.planes);
   const extended=incumbent.ok?undefined:this.planePrograms.grow(s.relations,this.planeStates.get(s.id),undefined,s.id);
   const made=incumbent.ok?incumbent:extended?.ok?{ok:true as const,pattern:extended.pattern,configuration:`whole-plane/${extended.state.width}/${extended.state.depthRows}/${extended.state.wallRows}/${extended.state.context??'fan'}`}:{ok:false as const,reason:`${incumbent.reason}; ${extended&&!extended.ok?extended.obligation:'no complete plane program'}`};
   return {incumbent,extended,made};
  };
  const planes=new Map(composition.subjects.filter(s=>s.type==='planar').map(s=>[s.id,makePlanar(s)]));
  const rounds=new Map(composition.subjects.filter(s=>s.type==='rounded').map(s=>[s.id,this.roundPrograms.construct(s)]));
  const completePlanes=[...planes.values()].flatMap(({made})=>made.ok?[made.pattern]:[]);
  const completeRounds=[...rounds.values()].flatMap(made=>made.ok?[made.pattern]:[]);
  // Visibility belongs to the complete constructed scene, not the current
  // object's spelling. In particular a background return must see the actual
  // foreground native body BEFORE cursor placement; otherwise a later cursor
  // collision incorrectly discards that entire background box.
  const planeOwners=new Map([...planes].flatMap(([owner,{made}])=>made.ok?[[made.pattern.id,[owner]] as const]:[]));
  for(const [owner,made]of rounds)if(made.ok)planeOwners.set(made.pattern.id,[owner]);
  const planeVisibility=selectVisible([...completePlanes,...completeRounds],planeOwners,graphs,fields,width,height);
  const visiblePlanes=new Map(planeVisibility.patterns.map(pattern=>[pattern.id,pattern]));
  const drawnRounds=new Set<string>();
  const addGeneralRound=(s:CompositionSubject,priorReason:string)=>{
   const made=rounds.get(s.id)!;
   if(!made.ok){omitted.push(`${s.id}:${priorReason}; ${made.reason}`);return;}
   const pattern=visiblePlanes.get(made.pattern.id);
   if(!pattern){omitted.push(`${s.id}:round has no visible complete native units`);return;}
   for(const reason of planeVisibility.unresolved)if(reason.startsWith(made.pattern.id+'/'))omitted.push(reason);
   const placed=this.placePattern(pattern,rows,undefined,true,canAdmit);
   if(!placed.ok){omitted.push(`${s.id}:round:${placed.reason}`);return;}
   const points=new Map<string,readonly [number,number]>(s.relations.flatMap(relation=>[0,.5,1].map(t=>{const p=relationPoint(relation,t);return [`${relation.id}/${t}`,[p[0],p[1]]] as const;})));
   if(admit(placed.group,makeSupport(pattern.id,s.owner,`${made.configuration}/${placed.phase}`,points,placed.group),[s.owner],s.relations.map(r=>r.id))){entries.push('source-grown-round');drawnRounds.add(s.id);}
   else omitted.push(`${s.id}:round complete neighborhood needs its next legal native part variant`);
  };
  // Complete large rounds have no serving small-entry incumbent. Admit their
  // entire neighborhood before background planes; otherwise a frozen
  // background cursor can reject the visible foreground form. Only actually
  // admitted round bodies may subsequently occlude planar construction.
  for(const s of composition.subjects)if(s.type==='rounded'&&s.bounds[3]-s.bounds[1]>=62)addGeneralRound(s,'whole round exceeds the small-entry domain');
  const admittedRoundIDs=new Set([...drawnRounds].flatMap(owner=>{const made=rounds.get(owner)!;return made.ok?[made.pattern.id]:[];}));
  const admittedRounds=completeRounds.filter(pattern=>admittedRoundIDs.has(pattern.id));
  const planarNeighbors=new NativeSurfaceVisibility([...completePlanes,...admittedRounds],fields,[...this.font.values()],graphs);
  const planarScene=selectVisible([...completePlanes,...admittedRounds],planeOwners,graphs,fields,width,height,planarNeighbors);
  const completePlaneIDs=new Set(completePlanes.map(plane=>plane.id));
  for(const p of planarScene.patterns)if(completePlaneIDs.has(p.id))visiblePlanes.set(p.id,p);
  for(const reason of planarScene.unresolved)if(!admittedRounds.some(pattern=>reason.startsWith(pattern.id+'/')))omitted.push(reason);
  const nextPlane=(s:CompositionSubject,preferredID:string):boolean=>{
   // A successfully built P1 is not necessarily a legal current placement.
   // Its refused pen must reach the SAME source's complete plane family,
   // just as a build-time domain miss does. Fixed D1 order, no fit/ranking.
   if(!planes.get(s.id)!.incumbent.ok)return false;
   const next=this.planePrograms.grow(s.relations,this.planeStates.get(s.id),undefined,s.id);
   if(!next.ok||next.pattern.id===preferredID)return false;
   planeOwners.set(next.pattern.id,[s.owner]);
   // Same-owner bodies cannot occlude this alternative. Reuse the shared
   // current-scene index rather than rebuilding EVERY neighboring body and
   // graph for every refused plane. Append each newly serving alternative in
   // the same order as completePlanes below.
   const visible=selectVisible([next.pattern],planeOwners,graphs,fields,width,height,planarNeighbors),p=visible.patterns[0];
   if(!p)return false;
   // Start with the complete incident program: its contacts, charts and
   // repeat ordinals translate together, never separate rail repairs.
   const placed=this.placePattern(p,rows,this.planarPhases.get(p.id),false,canAdmit);if(!placed.ok)return false;
   const {group,phase}=placed,points=new Map<string,readonly [number,number]>(s.relations.flatMap(relation=>[0,.5,1].map(t=>{const a=relationPoint(relation,t);return [`${relation.id}/${t}`,[a[0],a[1]]] as const;})));
   const configuration=`whole-plane/${next.state.width}/${next.state.depthRows}/${next.state.wallRows}/${next.state.context??'fan'}/${phase}`;
   if(!admit(group,makeSupport(p.id,s.owner,configuration,points,group),[s.owner],s.relations.map(r=>r.id)))return false;
   this.planeStates.set(s.id,next.state);this.planarPhases.set(p.id,phase);entries.push('whole-plane/D1');
   // Only this published alternative supplies its native body to subsequent
   // source parts. A discarded preferred spelling is not an occluder.
   completePlanes.push({...p,origin:[p.origin[0]+phase,p.origin[1]]});
   planarNeighbors.append([completePlanes.at(-1)!]);
   for(let i=omitted.length-1;i>=0;i--)if(omitted[i]!.startsWith(preferredID+'/'))omitted.splice(i,1);
   for(const reason of visible.unresolved)omitted.push(reason);return true;
  };
  for(const s of composition.subjects){
   if(drawnRounds.has(s.id))continue;
   if(s.type==='planar'){
    const {incumbent,extended,made}=planes.get(s.id)!;
    // Additive restoration at the existing owner: a valid incumbent still
    // serves unchanged. Unsupported incidence takes the complete planar
    // family, never a second renderer or an independently spelled edge.
    if(!made.ok){omitted.push(`${s.id}:planar:${made.reason}`);continue;}
    // Current scene visibility acts on the already constructed neighborhood.
    // It never restarts its phase or crops glyph ink. A cut/refit obligation
    // remains reported while the other genuinely visible whole units survive.
    const p=visiblePlanes.get(made.pattern.id);if(!p){if(!nextPlane(s,made.pattern.id))omitted.push(`${s.id}:P1 has no current visible whole units`);continue;}
    const placed=this.placePattern(p,rows,incumbent.ok?this.planarPhases.get(p.id):undefined,!incumbent.ok,canAdmit);
    if(!placed.ok){if(!nextPlane(s,made.pattern.id))omitted.push(`${s.id}:planar:${placed.reason}`);continue;}
    const {group,phase}=placed;
    const x=p.origin[0]+phase,y=p.origin[1]*17;
    const points=new Map<string,readonly [number,number]>(s.relations.flatMap(relation=>[0,.5,1].map(t=>{const a=relationPoint(relation,t);return [`${relation.id}/${t}`,[a[0],a[1]]] as const;})));
    const support=makeSupport(p.id,s.owner,`${made.configuration}/${(x-s.bounds[0]).toFixed(4)}/${(y-s.bounds[1]).toFixed(4)}/${s.relations.map(r=>r.intervals.map(p=>`${p.lo}:${p.hi}:${p.cause}`).join(';')).join('|')}`,points,group);
    if(admit(group,support,[s.owner],s.relations.map(r=>r.id))){entries.push(incumbent.ok?'independent-P1':'whole-plane');this.planarPhases.set(p.id,phase);if(extended?.ok)this.planeStates.set(s.id,extended.state);}else if(!nextPlane(s,made.pattern.id))omitted.push(`${s.id}:planar:whole neighborhood has no legal current placement`);
    continue;
   }
   if(s.type==='path-led')continue;
   if(s.type!=='rounded'){omitted.push(`${s.id}:no admitted whole ${s.type} construction`);continue;}
   const w=s.bounds[2]-s.bounds[0],h=s.bounds[3]-s.bounds[1];
   // Whole-entry INK footprints differ from their source row counts. These
   // disjoint declared domains dispatch complete entries, never row pieces.
   const id=h<30?ROUND_IDS[3]:h<38?ROUND_IDS[0]:h<47?ROUND_IDS[4]:w<50?ROUND_IDS[5]:w<55?ROUND_IDS[1]:h<62?ROUND_IDS[2]:undefined,block=id&&this.blocks.get(id);
   if(!block||!this.roundContext(s)){addGeneralRound(s,'whole round context or native size unsupported');continue;}
   const [a,b,c,d]=block.bounds,target=s.bounds;if(Math.abs(c-a-(target[2]-target[0]))>16||Math.abs(d-b-(target[3]-target[1]))>17||(target[2]-target[0])<(c-a)*.7||(target[3]-target[1])<(d-b)*.7){addGeneralRound(s,'whole round footprint outside admitted native band');continue;}
   const x=Math.round((target[0]+target[2]-a-c)/2),r=Math.round((target[1]+target[3]-b-d)/34),source=`${s.id}/whole/${block.id}`;
   const occurrences=block.atoms.map(atom=>({id:`${source}/${atom.row}/${atom.scalar}`,text:atom.text,x:x+atom.x,row:r+atom.row,contributions:[{source,kind:'path' as const,pixels:this.font.get(atom.text)!.points.map((_,i)=>i)}]}));
   const view=fields.find(v=>s.owner.startsWith(v.camera+'/'));
   // The native construction is not an image-sized glyph stencil. Its one
   // whole shape chart maps ink contributions to the selected source envelope;
   // the intact rendered font remains at native size. Current foreground still
   // prohibits hidden background ink at the actual native output positions.
   if(view&&!occurrences.every(atom=>this.font.get(atom.text)!.points.every(p=>{
    const px=atom.x+p[0]!+.5,py=atom.row*17+p[1]!+.5;
    if(px<view.rect.x||py<view.rect.y||px>=view.rect.x+view.rect.width||py>=view.rect.y+view.rect.height)return false;
    const hit=this.envelopePoint(s,(px-x-(a+c)/2)/((c-a)/2),(py-r*17-(b+d)/2)/((d-b)/2));
    if(!hit||!hit.relation.intervals.some(part=>part.cause==='visible'&&part.lo-1e-5<=hit.t&&part.hi+1e-5>=hit.t)||!this.onOwner(view,s.owner,hit.point[0],hit.point[1],3))return false;
    const foreground=view.field.sample((px-view.rect.x)/view.rect.width*view.field.width,(py-view.rect.y)/view.rect.height*view.field.height);
    return !foreground||fieldOwner(view,foreground)===s.owner||foreground.depth<=hit.point[2]+1e-6;
   }))){addGeneralRound(s,'whole round contribution crosses current visibility or foreground');continue;}
   const paper=block.paper.map(p=>({...p,id:`${source}/${p.id}`,sources:[source],rect:[x+p.rect[0],r*17+p.rect[1],p.rect[2],p.rect[3]] as const}));
   const group={id:source,occurrences,contacts:[],paper,requiredSources:[source]},points=new Map<string,readonly [number,number]>(s.relations.flatMap(relation=>[0,.5,1].map(t=>{const p=relationPoint(relation,t);return [`${relation.id}/${t}`,[p[0],p[1]]] as const;})));
   if(admit(group,makeSupport(source,s.owner,`${block.id}/${(x-(target[0]+target[2])/2).toFixed(4)}/${(r*17-(target[1]+target[3])/2).toFixed(4)}/${s.relations.map(r=>r.intervals.map(p=>`${p.lo}:${p.hi}:${p.cause}`).join(';')).join('|')}`,points,group),[s.owner],s.relations.map(r=>r.id)))entries.push(block.id);else addGeneralRound(s,'whole round has no direct legal placement');
  }
  // Add the new source-part domain only after serving neighborhoods have
  // retained their exact tape/ports/paper. An unadmitted LP body cannot erase
  // incumbent geometry. LP's own whole boundary/void neighborhood is then
  // evaluated against those real neighbors before the SAME NativePage.
  const admitted=new Set(groups.map(group=>group.id)),bodies=[...completePlanes,...completeRounds].filter(pattern=>admitted.has(pattern.id));
  const partScene=new NativeSurfaceVisibility(bodies,fields,[...this.font.values()],graphs);
  const sources=new Map(geometry.map(source=>[source.owner,source]));
  for(const subject of composition.subjects.filter(subject=>subject.type==='path-led')){
   const view=fields.find(view=>subject.owner.startsWith(view.camera+'/'));
   const made=this.limbPrograms.construct(subject,sources.get(subject.owner),view);
   if(!made.ok){omitted.push(`${subject.id}:path-led:${made.reason}`);continue;}
   planeOwners.set(made.pattern.id,[subject.owner]);
   const visible=selectVisible([made.pattern],planeOwners,graphs,fields,width,height,partScene),pattern=visible.patterns[0];
   for(const reason of visible.unresolved)if(reason.startsWith(made.pattern.id+'/'))omitted.push(reason);
   if(!pattern){omitted.push(`${subject.id}:path-led:complete source part has no visible native units`);continue;}
   let points:Map<string,readonly [number,number]>|undefined;
   const placePart=(p:NativePattern,factored=false):string|undefined=>{
    const placed=this.placePattern(p,rows,undefined,true,canAdmit);if(!placed.ok)return placed.reason;
    points??=new Map(subject.relations.flatMap(relation=>[0,.5,1].map(t=>{const q=relationPoint(relation,t);return [`${relation.id}/${t}`,[q[0],q[1]]] as const;})));
    if(!admit(placed.group,makeSupport(p.id,subject.owner,`${made.configuration}/${placed.phase}`,points,placed.group),[subject.owner],subject.relations.map(relation=>relation.id)))return 'complete source-part neighborhood needs its next legal D1 alternative';
    entries.push(factored?'source-part-LP/component':'source-part-LP');
    const origin:readonly [number,number]=[p.origin[0]+placed.phase,p.origin[1]];
    partScene.append([{...p,origin,occurrences:placed.group.occurrences.map(atom=>({...atom,x:atom.x-origin[0],row:atom.row-origin[1]}))}]);return;
   };
   const refusal=placePart(pattern);
   if(refusal){
    const components=this.sourceComponents(pattern);
    if(components.length===1)omitted.push(`${subject.id}:path-led:${refusal}`);
    else for(const component of components){const reason=placePart(component,true);if(reason)omitted.push(`${component.id}:path-led:${reason}`);}
   }
  }
  // Restore the disconnected source programs BEFORE generic class fallback.
  // They feed the SAME visibility/admission/page owner. A current program
  // which already drew the owner remains intact; a failed placement is not
  // a reason to leave its historical producer disconnected.
  const alreadyServed=new Set([...supportOwners.values()].flat());
  const farContributors=new Set(composition.farStrip.flatMap(strip=>strip.contributors));
  const restoreGraphs=graphs.map(graph=>({...graph,relations:graph.relations.filter(relation=>!alreadyServed.has(relation.owner)&&!farContributors.has(relation.owner))}));
  const restored=this.retained.propose(restoreGraphs,fields);
  const restoredPatterns=restored.patterns.filter(pattern=>(restored.owners.get(pattern.id)??[]).every(owner=>!alreadyServed.has(owner)));
  const restoredScene=selectVisible(restoredPatterns,restored.owners,graphs,fields,width,height,partScene);
  const restoredRelations=new Map<string,PosedRelation>();
  if(restoredScene.patterns.length)for(const graph of graphs)for(const relation of graph.relations)restoredRelations.set(relation.id,relation);
  for(const pattern of restoredScene.patterns){
   const ownerIDs=restored.owners.get(pattern.id)??[],owner=ownerIDs[0];if(!owner||!composition.owners.has(owner))continue;
   const placed=this.placePattern(pattern,rows,undefined,true,canAdmit);if(!placed.ok)continue;
   const relationIDs=pattern.paths?.flatMap(path=>path.relations)??[];
   const points=new Map<string,readonly [number,number]>();
   for(const relationID of relationIDs){const relation=restoredRelations.get(relationID);if(!relation)continue;for(const t of [0,.5,1]){const p=relationPoint(relation,t);points.set(`${relation.id}/${t}`,[p[0],p[1]]);}}
   if(admit(placed.group,makeSupport(pattern.id,owner,'restored/'+pattern.id+'/'+placed.phase,points,placed.group),ownerIDs,relationIDs))entries.push('restored/'+pattern.id);
  }
  for(const strip of composition.farStrip)omitted.push(`${strip.camera}/far-strip:merged ${strip.contributors.length} contributors; no admitted whole skyline entry in this context`);
  // Build-time context misses must reach the same D1 owner as pen conflicts.
  // A worker still compiling, an open sheet, an opposite cube orientation or
  // raw GPU topology cannot make a visible source vanish before admission.
  // This generic class construction reads the complete posed graph; it is
  // never an icon at the object's bounding box or a scene-name exception.
  const candidateOwners=new Set([...intents.values()].flatMap(intent=>intent.owners));
  const servedOwners=new Set([...supportOwners.values()].flat());
  const subjectsByOwner=new Map(composition.subjects.map(subject=>[subject.owner,subject]));
  for(const subject of composition.subjects){
   if(candidateOwners.has(subject.owner)||servedOwners.has(subject.owner))continue;
   const view=fields.find(view=>subject.owner.startsWith(view.camera+'/'));if(!view)continue;
   const pattern=this.terminalPattern(subject,sources.get(subject.owner),view);if(!pattern)continue;
   const owners=new Map([[pattern.id,[subject.owner]]]);selectVisible([pattern],owners,graphs,fields,width,height,partScene);
  }
  // Every producer reaches ONE terminal admission. A preferred placement's
  // refusal is not its output: its still-visible intact units remain intents.
  // Do not add a second spelling when a different complete family already
  // serves the owner (e.g. a retained compact circle rather than smoothClosed).
  // Serving one part of an owner does not serve its other parts. A wall,
  // aperture and rail can share one source owner without sharing a native
  // construction. Suppress duplicate alternatives only when their actual
  // source relations are already represented, never merely by mesh identity.
  const selectedRelations=new Set([...supportRelations.values()].flat());
  // Source programs and their independently connected components are indexed
  // once. Re-scanning every subject/group for every terminal intent made a
  // dense producer scene quadratic before its first visible publication.
  const servingByPattern=new Map<string,NativeNeighborhood[]>();
  for(const group of groups){
   const keys=[group.id];let at=group.id.indexOf('/component/');
   while(at>=0){keys.push(group.id.slice(0,at));at=group.id.indexOf('/component/',at+1);}
   for(const key of keys){const members=servingByPattern.get(key)??[];members.push(group);servingByPattern.set(key,members);}
  }
  const requests:TerminalRequest[]=groups.map(group=>({group,pens:this.runPens(group)}));
  const present=new Set(groups.flatMap(group=>group.occurrences.map(atom=>atom.id)));
  for(const {pattern,owners,visibleAt}of intents.values()){
   const serving=servingByPattern.get(pattern.id)??[];
   const relationIDs=pattern.paths?.flatMap(path=>path.relations)??[];
   if(!serving.length&&relationIDs.length&&relationIDs.every(id=>selectedRelations.has(id)))continue;
   const missing=pattern.occurrences.filter(atom=>!present.has(atom.id));if(!missing.length)continue;
   const dx=pattern.origin[0],dy=pattern.origin[1]*17;
   const group:NativeNeighborhood={...pattern,id:`${pattern.id}/terminal`,
    occurrences:missing.map(atom=>({...atom,x:atom.x+dx,row:atom.row+pattern.origin[1]})),
    contacts:pattern.contacts.map(contact=>({...contact,fromAnchor:[contact.fromAnchor[0]+dx,contact.fromAnchor[1]+dy],toAnchor:[contact.toAnchor[0]+dx,contact.toAnchor[1]+dy]})),
    paper:pattern.paper.map(paper=>({...paper,rect:[paper.rect[0]+dx,paper.rect[1]+dy,paper.rect[2],paper.rect[3]]})),
    phaseWith:serving.map(group=>group.id),requiredSources:[...new Set(missing.flatMap(atom=>atom.contributions.map(c=>c.source)))]};
   requests.push({group,pens:this.runPens(group,pattern),visibleAt});missing.forEach(atom=>present.add(atom.id));
   relationIDs.forEach(id=>selectedRelations.add(id));
   const owner=owners[0],subject=owner===undefined?undefined:subjectsByOwner.get(owner);
   if(subject){const points=new Map<string,readonly [number,number]>(subject.relations.flatMap(relation=>[0,.5,1].map(t=>{const p=relationPoint(relation,t);return [`${relation.id}/${t}`,[p[0],p[1]]] as const;})));
    supports.set(group.id,makeSupport(group.id,subject.owner,'terminal/'+pattern.id,points,group));supportOwners.set(group.id,owners);supportRelations.set(group.id,subject.relations.map(relation=>relation.id));}
  }
  const requestedIDs=new Set(requests.map(request=>request.group.id));
  for(const item of deferred.values()){
   if(requestedIDs.has(item.group.id)||item.group.occurrences.every(atom=>present.has(atom.id))||item.sources.length&&item.sources.every(source=>selectedRelations.has(source)))continue;
   requests.push({group:item.group,pens:this.runPens(item.group)});supports.set(item.group.id,item.support);supportOwners.set(item.group.id,item.owners);supportRelations.set(item.group.id,item.sources);
   item.sources.forEach(source=>selectedRelations.add(source));item.group.occurrences.forEach(atom=>present.add(atom.id));
   requestedIDs.add(item.group.id);
  }
  const terminal=terminalAdmission(requests,[...this.font.values()],width,height,this.whitespace);
  const realized=new Set(terminal.groups.flatMap(group=>group.occurrences.map(atom=>atom.id)));
  const cutObligation=':visible source span needs its terminal run at the cut';
  // Retire this exact pending obligation only when its intact scalar really
  // reached the shared page at a majority-visible legal pen. An unplaced
  // scalar/joint remains a failure; preferred-program errors are not broadly
  // erased merely because some other part of the mesh drew.
  for(let i=omitted.length-1;i>=0;i--)if(omitted[i]!.endsWith(cutObligation)&&realized.has(omitted[i]!.slice(0,-cutObligation.length)))omitted.splice(i,1);
  // A large scene can exceed the engine's function-argument limit here.
  // Keep EVERY unresolved scalar/joint in order, without spreading a frame-
  // sized list onto the JavaScript call stack (voxel's actual RangeError).
  for(const reason of terminal.unresolved)omitted.push(reason);
  for(const group of terminal.groups)if(terminal.changed.has(group.id)){
   const support=supports.get(group.id);if(support)supports.set(group.id,{...support,configuration:support.configuration+'\nterminal/'+JSON.stringify(group.occurrences.map(atom=>[atom.id,atom.x,atom.row]))});
  }
  // D6 region tone classified on the GPU (tone-field.ts): placed only on
  // surfaces whose construction was admitted, cleared around its strokes.
  // Each owner's tone shares the support of the construction serving it, so
  // it moves and invalidates with that construction.
  const servingTone=new Map<string,string>();
  for(const group of terminal.groups)for(const owner of supportOwners.get(group.id)??[])if(!servingTone.has(owner)&&supports.has(group.id))servingTone.set(owner,group.id);
  const toneGroups=placeRegionTone(fields,terminal.groups,new Set(servingTone.keys()),this.font,width,height,this.whitespace).map(({owner,group})=>{
   const serving=servingTone.get(owner),source=serving?supports.get(serving):undefined;
   const first=group.occurrences[0],shape=group.occurrences.map(atom=>[atom.id,atom.text,atom.x-(first?.x??0),atom.row-(first?.row??0)]);
   // Atmosphere is owned by the camera/backdrop, not a fabricated mesh.
   // Surface members retain the actual source lifetime AND their lattice
   // identities; changed shadow membership cannot retain obsolete text.
   supports.set(group.id,{owners:source?.owners??[],configuration:(source?.configuration??owner)+'\ntone/'+JSON.stringify(shape),
    points:new Map([...(source?.points??[]),...group.occurrences.map(atom=>[atom.id,[atom.x,atom.row*17] as const] as const)])});
   supportOwners.set(group.id,source?[owner]:[]);supportRelations.set(group.id,serving?supportRelations.get(serving)??[]:[]);
   return group;
  });
  const page=this.page.commit([...terminal.groups,...toneGroups],width,height,serial),words=page.occurrences.map(a=>({text:a.text,x:a.x,row:a.row,advance:this.font.get(a.text)!.advance,features:a.contributions.map(c=>c.source),role:'composed-whole-construction'}));
  const count=page.occurrences.length,dots=page.occurrences.filter(a=>/[.;:,]/u.test(a.text)).length;
  const metrics:ConstructionMetrics={cpuFrameScope:'native composition + realization CPU; GPU/readback/engine excluded',compositionMs:composition.cpuMs,realizationMs:performance.now()-started-composition.cpuMs,cpuFrameMs:performance.now()-started,subjects:composition.subjects.length,sourceSubjects:composition.owners.size,farStripCount:composition.farStrip.length,publishedFarStripCount:0,layoutDroppedShare:proposedGlyphs?rejectedGlyphs/proposedGlyphs:0,dotShare:count?dots/count:0,repeatedRampTokens:page.rows.reduce((n,row)=>n+(row.match(/;{3,}|:{3,}|\.{3,}/g)?.length??0),0),wholeEntries:entries};
  return Object.freeze({...page,words,sourceLines:typeof lines==='number'?lines:lines.length,omitted,supportOwners,supportRelations,directSupports:supports,metrics});
 }
 /** The compiled backend has NO route through the retired fitter, row DP,
  * trial phases, terminal pen loops or CPU tone probes. NativePage verifies
  * the GPU's one tape; it cannot choose a replacement or move a character. */
 /** Serialize the already-selected GPU tape. This is not a construction or
  * visibility pass; retained GPU supports make posed graph decoding optional. */
 commitPlayback(lines:readonly GraphLine[]|number,width:number,height:number,serial:number,graphs:readonly PosedGraph[],playback:ArticulationResult):NativeFrame{
  const started=performance.now(),owners=new Map(graphs.flatMap(graph=>[...graph.owners??[]]));
  const relations=new Map<string,PosedRelation>();for(const graph of graphs)for(const relation of graph.relations)relations.set(relation.id,relation);
  const supportOwners=new Map<string,readonly string[]>(),supportRelations=new Map<string,readonly string[]>(),directSupports=new Map<string,NativeSupport>();
  for(const item of playback.groups){
   const points=new Map<string,readonly [number,number]>(),lifetimes:NativeSupport['owners'][number][]=[];
   for(const owner of item.owners){const source=owners.get(owner);if(source)lifetimes.push({reference:source.lifetime,source:source.domain,kind:source.kind});}
   for(const id of item.sources){const relation=relations.get(id);if(!relation)continue;for(const t of [0,.5,1]){const point=relationPoint(relation,t);points.set(`${id}/${t}`,[point[0],point[1]]);}}
   const first=item.group.occurrences[0],shape=item.group.occurrences.map(atom=>[atom.id,atom.text,atom.x-(first?.x??0),atom.row-(first?.row??0),atom.foreground,atom.contributions.map(part=>[part.source,part.kind,part.pixels===nativeWholeInk(this.font.get(atom.text)!)?'whole':part.pixels,part.parameter,part.support])]);
   const source=item.support??{configuration:item.group.id,points,owners:lifetimes};
   directSupports.set(item.group.id,{...source,configuration:source.configuration+'\ncompiled/'+JSON.stringify(shape)});
   supportOwners.set(item.group.id,item.owners);supportRelations.set(item.group.id,item.sources);
  }
  // D1: a group the page cannot verify (e.g. an illegal cursor gap against
  // another group) is reported and left out; it never blanks the frame.
  let committed=playback.groups;const rejected:string[]=[];let page:ReturnType<NativePage['commit']>|undefined;
  while(!page){
   try{page=this.page.commit(committed.map(item=>item.group),width,height,serial);}
   catch(error){
    if(!(error instanceof NativeCommitError)||!committed.some(item=>item.group.id===error.neighborhood))throw error;
    rejected.push(`${error.neighborhood}:${error.obligation}`);committed=committed.filter(item=>item.group.id!==error.neighborhood);
   }
  }
  const served=new Set(committed.flatMap(item=>item.owners)),omitted=[...playback.unresolved,...rejected];
  const visibleOwners=new Set<string>();for(const graph of graphs)for(const relation of graph.relations)if(relation.intervals.some(part=>part.cause==='visible'))visibleOwners.add(relation.owner);
  for(const owner of visibleOwners)if(!served.has(owner))omitted.push(`${owner}:compiled construction has no admitted visible run`);
  const words=page.occurrences.map(atom=>({text:atom.text,x:atom.x,row:atom.row,advance:this.font.get(atom.text)!.advance,features:atom.contributions.map(part=>part.source),role:'compiled-gpu-whole-word'}));
  const metrics:ConstructionMetrics={cpuFrameScope:'compiled native tape verification/publication CPU; GPU work and waits excluded',compositionMs:0,realizationMs:performance.now()-started,cpuFrameMs:performance.now()-started,subjects:served.size,sourceSubjects:owners.size,farStripCount:0,publishedFarStripCount:0,layoutDroppedShare:null,dotShare:page.occurrences.length?page.occurrences.filter(atom=>/[.;:,]/u.test(atom.text)).length/page.occurrences.length:0,repeatedRampTokens:page.rows.reduce((n,row)=>n+(row.match(/;{3,}|:{3,}|\.{3,}/g)?.length??0),0),wholeEntries:['compiled-gpu-playback']};
  return Object.freeze({...page,words,sourceLines:typeof lines==='number'?lines:lines.length,omitted,supportOwners,supportRelations,directSupports,metrics});
 }
 private terminalPattern(subject:CompositionSubject,geometry:CompositionGeometry|undefined,view:DrawingFieldView):NativePattern|undefined{
  // Compact rounds retain a COMPLETE library construction by row height,
  // including when a contact made the isolated-round incumbent ineligible.
  // Its visibility is resolved by the same source chart as every other part.
  const height=subject.bounds[3]-subject.bounds[1];
  if(subject.type==='rounded'&&height<4*17){
   const id=height<2*17?ROUND_IDS[3]:height<3*17?ROUND_IDS[4]:ROUND_IDS[2],block=this.blocks.get(id)!;
   const [a,b,c,d]=block.bounds,[left,top,right,bottom]=subject.bounds,source=`${subject.id}/class-round/${id}`;
   const origin:[number,number]=[Math.round((left+right-a-c)/2),Math.round((top+bottom-b-d)/34)];
   const native=[[a,b],[c,b],[c,d],[a,d]] as const,posed=[[left,top],[right,top],[right,bottom],[left,bottom]] as const;
   return {id:source,origin,occurrences:block.atoms.map(atom=>({id:`${source}/${atom.row}/${atom.scalar}`,text:atom.text,x:atom.x,row:atom.row,contributions:[{source,kind:'path',pixels:this.font.get(atom.text)!.points.map((_,i)=>i)}]})),
    contacts:[],paper:block.paper.map(paper=>({...paper,id:`${source}/${paper.id}`,sources:[source]})),requiredSources:[source],paths:[{id:source,relations:subject.relations.map(relation=>relation.id)}],
    surfaces:[{id:`${source}/body`,owner:subject.owner,coverage:'source-receiver',outline:native,
     charts:[[0,1,2],[0,2,3]].map(indices=>({native:indices.map(i=>native[i]!) as unknown as readonly [typeof native[0],typeof native[0],typeof native[0]],source:indices.map(i=>posed[i]!) as unknown as readonly [typeof posed[0],typeof posed[0],typeof posed[0]]}))}]};
  }
  const made=this.limbPrograms.terminal(subject,geometry,view);return made.ok?made.pattern:undefined;
 }
 /** A native word can move only on its selected run. Same-row repeats share
  * the source span; a flank/turn has its own row's local native-unit span.
  * This is a pen domain, not an image/glyph fit window or a font search. */
 private runPens(group:NativeNeighborhood,pattern?:NativePattern):ReadonlyMap<string,readonly [number,number]>{
  const pens=new Map<string,readonly [number,number]>(),byID=new Map(group.occurrences.map(atom=>[atom.id,atom]));
  for(const atom of group.occurrences){const unit=this.font.get(atom.text)!.advance;pens.set(atom.id,[atom.x-unit,atom.x+unit]);}
  for(const run of pattern?.repeatRuns??[]){
   const rows=new Map<number,NativeOccurrence[]>();for(const id of run.members){const atom=byID.get(id);if(atom){const members=rows.get(atom.row)??[];members.push(atom);rows.set(atom.row,members);}}
   for(const members of rows.values()){let left=Infinity,right=-Infinity;for(const atom of members){left=Math.min(left,atom.x);right=Math.max(right,atom.x+this.font.get(atom.text)!.advance);}
    for(const atom of members){const unit=this.font.get(atom.text)!.advance;pens.set(atom.id,[left-unit,right]);}}
  }
  return pens;
 }
 /** Factor only genuinely independent source constructions, never individual
  * glyphs or clipped visible fragments. A shared source edge, intact scalar,
  * join, repeat run or protected-paper relationship closes its participants
  * into ONE component. A refusal in another component cannot erase it.
  * Full path records retain hidden incidence, so visibility cannot split a
  * cooperating cap/flank/return into unrelated placement transactions. */
 private sourceComponents(pattern:NativePattern):NativePattern[]{
  if(!pattern.paths?.length)return [pattern];
  const parents=new Map<string,string>();
  const root=(id:string):string=>{if(!parents.has(id))parents.set(id,id);let at=id;while(parents.get(at)!==at)at=parents.get(at)!;while(parents.get(id)!==id){const next=parents.get(id)!;parents.set(id,at);id=next;}return at;};
  const join=(sources:readonly string[])=>{if(!sources.length)return;let first=root(sources[0]!);for(const source of sources.slice(1)){const next=root(source);parents.set(first<next?next:first,first<next?first:next);first=root(first);}};
  const incidences=new Map<string,string>(),byID=new Map(pattern.occurrences.map(atom=>[atom.id,atom]));
  for(const path of pattern.paths){root(path.id);for(const relation of path.relations){const other=incidences.get(relation);if(other)join([path.id,other]);else incidences.set(relation,path.id);}}
  for(const part of pattern.sourceParts??[])join(part.paths);
  for(const atom of pattern.occurrences)join(atom.contributions.map(part=>part.source));
  const atomSources=(id:string)=>byID.get(id)?.contributions.map(part=>part.source)??[];
  for(const contact of pattern.contacts)join([...atomSources(contact.from),...atomSources(contact.to),contact.fromSource,contact.toSource]);
  for(const link of pattern.continuations??[])join([...atomSources(link.from),...atomSources(link.to)]);
  for(const run of pattern.repeatRuns??[])join(run.members.flatMap(atomSources));
  for(const paper of pattern.paper)join(paper.sources);
  const components=new Map<string,NativeOccurrence[]>();
  for(const atom of pattern.occurrences){const key=root(atom.contributions[0]!.source),members=components.get(key)??[];members.push(atom);components.set(key,members);}
  if(components.size<2)return [pattern];
  const keys=[...components.keys()].sort(),first=keys[0]!;
  // Paper belonging to a fully hidden source still constrains the parent.
  // Keep that constraint with its first surviving component, rather than
  // dropping it or publishing a paper-only pseudo-form.
  const assigned=(source:string)=>{const key=root(source);return components.has(key)?key:first;};
  return keys.map(key=>{
   const occurrences=components.get(key)!,ids=new Set(occurrences.map(atom=>atom.id));
   const sourceParts=pattern.sourceParts?.filter(part=>part.paths.length&&assigned(part.paths[0]!)===key);
   return {...pattern,id:`${pattern.id}/component/${key}`,occurrences,
    sourceParts,surfaces:sourceParts?.length?sourceParts.map(part=>part.surface):pattern.surfaces,
    paths:pattern.paths!.filter(path=>assigned(path.id)===key),
    requiredSources:pattern.requiredSources.filter(source=>assigned(source)===key),
    contacts:pattern.contacts.filter(contact=>ids.has(contact.from)&&ids.has(contact.to)),
    continuations:pattern.continuations?.filter(link=>ids.has(link.from)&&ids.has(link.to)),
    repeatRuns:pattern.repeatRuns?.filter(run=>run.members.some(id=>ids.has(id))),
    paper:pattern.paper.filter(paper=>assigned(paper.sources[0]!)===key)};
  });
 }
 private placePattern(p:NativePattern,rows:ReadonlyMap<number,readonly NativeOccurrence[]>,previous?:number,parts=true,legal?:(group:NativeNeighborhood)=>boolean,toned=true):{ok:true;group:NativeNeighborhood;phase:number}|{ok:false;reason:string}{
  const original=p;let withTone=false;
  const fail=(reason:string):{ok:true;group:NativeNeighborhood;phase:number}|{ok:false;reason:string}=>{
   // D6 tone is chosen with the construction, never at its expense: when the
   // toned construction has no legal pen, the same construction is placed
   // without it (it never draws less than before tone existed).
   if(withTone)return this.placePattern(original,rows,previous,parts,legal,false);
   // A constrained run cannot prove that its COMPLETE incident neighborhood
   // has no legal pen. The whole program may translate while every joint,
   // chart and paper interval stays together. Try that same-entry D1
   // alternative before rejecting the form. Keep serving placements first;
   // this branch only restores a construction previously refused here.
   if(parts&&p.repeatRuns?.length){const rigid=this.placePattern(p,rows,previous,false,legal);if(rigid.ok)return rigid;}
   return {ok:false,reason};
  };
  let absolute=p.occurrences.map(a=>({...a,x:a.x+p.origin[0],row:a.row+p.origin[1]}));
  if(parts&&p.repeatRuns?.length){const placement=this.placeWholeParts(p,absolute,rows);if(!placement.ok)return fail(placement.reason);absolute=placement.occurrences;}
  if(toned){const tone=this.tone.members(p,absolute);if(tone){absolute=[...absolute,...tone.atoms];p={...p,paper:tone.paper};withTone=true;}}
  const constraints:NativeCursorConstraint[]=[],sharedConstraints:NativeCursorConstraint[]=[],lineRows=new Set(absolute.map(a=>a.row));
  for(const row of lineRows){
   const all=absolute.filter(a=>a.row===row),members=all.filter((a,i)=>!all.slice(0,i).some(b=>b.x===a.x&&b.text===a.text&&(b.foreground??0xff000000)===(a.foreground??0xff000000))).sort((a,b)=>a.x-b.x),current=new Set(members.map(a=>a.id));
   for(let i=1;i<members.length;i++)if(!nativeSpacePlan(members[i]!.x-members[i-1]!.x-this.font.get(members[i-1]!.text)!.advance,0,false,false,this.whitespace))return fail('complete neighborhood has no common legal native scalar tape');
   const existing=(rows.get(row)??[]).filter(b=>{
    if(!members.some(a=>a.x===b.x&&a.text===b.text&&(a.foreground??0xff000000)===(b.foreground??0xff000000)))return true;
    // Both contributors name this same pen; neither may transport a copy
    // independently while still claiming to share the original occurrence.
    sharedConstraints.push({gap:0,leading:false,direction:1},{gap:0,leading:false,direction:-1});return false;
   });
   for(const [neighbors,into]of [[rows.get(row)??[],constraints],[existing,sharedConstraints]] as const){
    const merged=[...neighbors,...members].sort((a,b)=>a.x-b.x||a.id.localeCompare(b.id));
    for(let i=0;i<merged.length;i++){const a=merged[i]!,before=merged[i-1],mine=current.has(a.id),prior=!!before&&current.has(before.id);if(!before){if(mine)into.push({gap:a.x,leading:true,direction:1});continue;}if(mine===prior)continue;into.push({gap:a.x-before.x-this.font.get(before.text)!.advance,leading:false,direction:mine?1:-1});}
   }
  }
  // Preserve every serving independent placement. Exact shared ink is an
  // additional whole-entry alternative, never a new constraint that can
  // withdraw a formerly legal separate pen.
  const at=(phase:number):NativeNeighborhood=>{const x=p.origin[0]+phase,y=p.origin[1]*17;return {id:p.id,occurrences:absolute.map(a=>({...a,x:a.x+phase})),requiredSources:p.requiredSources,
   contacts:p.contacts.map(c=>({...c,fromAnchor:[c.fromAnchor[0]+x,c.fromAnchor[1]+y],toAnchor:[c.toAnchor[0]+x,c.toAnchor[1]+y]})),
   paper:p.paper.map(a=>({...a,rect:[a.rect[0]+x,a.rect[1]+y,a.rect[2],a.rect[3]]}))};};
  // D1's nearest-legal pen is a placement of this SAME intact construction.
  // Whitespace, occupied ink and protected paper must agree on that pen;
  // selecting a text cursor first and refusing the whole form afterwards
  // skips legal placements. No glyph, source fit or quality score is tried.
  let group:NativeNeighborhood|undefined;const refused=new Set<number>();
  const allowed=(phase:number)=>{if(refused.has(phase))return false;const candidate=at(phase);if(legal&&!legal(candidate)){refused.add(phase);return false;}group=candidate;return true;};
  const phase=nativeProgramPhase(constraints,previous,this.whitespace,allowed)??nativeProgramPhase(sharedConstraints,previous,this.whitespace,allowed);
  if(phase===undefined)return fail('complete neighborhood needs the next shared native pen variant');
  return {ok:true,phase,group:group!};
 }
 /** D1 nearest-legal placement belongs to construction, after current source
  * visibility is known and before NativePage. A part is an entire declared
  * run, not a glyph/cell. Neither the scalar tape nor its repeat ordinals can
  * change here. Joints and every other already placed part constrain it. */
 private placeWholeParts(pattern:NativePattern,input:NativeOccurrence[],placed:ReadonlyMap<number,readonly NativeOccurrence[]>):{ok:true;occurrences:NativeOccurrence[]}|{ok:false;reason:string}{
  let occurrences=input;
  for(const part of pattern.repeatRuns??[]){
   const members=new Set(part.members),constraints:NativeCursorConstraint[]=[],sharedConstraints:NativeCursorConstraint[]=[],affected=new Set(occurrences.filter(atom=>members.has(atom.id)).map(atom=>atom.row));
   if(!affected.size)continue;
   for(const row of affected){
    const all=[...(placed.get(row)??[]),...occurrences.filter(atom=>atom.row===row)],atoms:NativeOccurrence[]=[];
    for(const atom of all){const same=atoms.find(a=>a.x===atom.x&&a.text===atom.text&&(a.foreground??0xff000000)===(atom.foreground??0xff000000));
     if(!same){atoms.push(atom);continue;}
     if(members.has(atom.id)!==members.has(same.id))sharedConstraints.push({gap:0,leading:false,direction:1},{gap:0,leading:false,direction:-1});
    }
    for(const [list,into]of [[all,constraints],[atoms,sharedConstraints]] as const){
     list.sort((a,b)=>a.x-b.x||a.id.localeCompare(b.id));
     for(let i=0;i<list.length;i++){
      const atom=list[i]!,before=list[i-1],mine=members.has(atom.id),prior=!!before&&members.has(before.id);
      if(!before){if(mine)into.push({gap:atom.x,leading:true,direction:1});continue;}
      if(mine===prior)continue;
      into.push({gap:atom.x-before.x-this.font.get(before.text)!.advance,leading:false,direction:mine?1:-1});
     }
    }
   }
   const phase=nativeProgramPhase(constraints,undefined,this.whitespace)??nativeProgramPhase(sharedConstraints,undefined,this.whitespace);
   if(phase===undefined)return {ok:false,reason:`whole part ${part.id} needs the next complete D1 variant`};
   if(phase)occurrences=occurrences.map(atom=>members.has(atom.id)?{...atom,x:atom.x+phase}:atom);
  }
  return {ok:true,occurrences};
 }
 private envelopePoint(s:CompositionSubject,nx:number,ny:number):{relation:PosedRelation;t:number;point:readonly [number,number,number]}|undefined{
  const cx=(s.bounds[0]+s.bounds[2])/2,cy=(s.bounds[1]+s.bounds[3])/2,dx=nx*(s.bounds[2]-s.bounds[0])/2,dy=ny*(s.bounds[3]-s.bounds[1])/2;
  let hit:{relation:PosedRelation;t:number;point:readonly [number,number,number];distance:number}|undefined;
  for(const relation of s.relations){const a=relationPoint(relation,0),b=relationPoint(relation,1),ex=b[0]-a[0],ey=b[1]-a[1],den=dx*ey-dy*ex;if(Math.abs(den)<1e-10)continue;
   const ax=a[0]-cx,ay=a[1]-cy,distance=(ax*ey-ay*ex)/den,along=(ax*dy-ay*dx)/den;if(distance<0||along< -1e-5||along>1+1e-5)continue;
   const t=relationParameter(relation,Math.min(1,Math.max(0,along)));if(!hit||distance>hit.distance)hit={relation,t,point:relationPoint(relation,t),distance};}
  return hit;
 }
 private roundContext(s:CompositionSubject):boolean{
  const edges=s.relations.filter(r=>r.kind!=='crease');if(s.openings.length||!edges.length||edges.some(r=>r.intervals.some(p=>p.cause!=='visible'&&Math.hypot(...relationPoint(r,p.hi).slice(0,2).map((v,k)=>v-relationPoint(r,p.lo)[k]!))>3))||s.contacts.length)return false;
  const degree=new Map<string,number>();for(const e of edges)for(const p of e.ports)degree.set(p,(degree.get(p)??0)+1);if([...degree.values()].some(n=>n!==2))return false;
  const cx=(s.bounds[0]+s.bounds[2])/2,cy=(s.bounds[1]+s.bounds[3])/2,rx=(s.bounds[2]-s.bounds[0])/2,ry=(s.bounds[3]-s.bounds[1])/2;
  return edges.every(e=>[0,.5,1].every(t=>{const p=relationPoint(e,t);return Math.abs(Math.hypot((p[0]-cx)/rx,(p[1]-cy)/ry)-1)<.2;}));
 }
 private onOwner(view:DrawingFieldView,owner:string,x:number,y:number,radius:number):boolean{
  const fx=(x-view.rect.x)/view.rect.width*view.field.width,fy=(y-view.rect.y)/view.rect.height*view.field.height;
  for(const [dx,dy]of [[0,0],[-radius,0],[radius,0],[0,-radius],[0,radius]]){const s=view.field.sample(fx+dx!,fy+dy!);if(s&&fieldOwner(view,s)===owner)return true;}return false;
 }
}
