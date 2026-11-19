import {CurveProgramBank} from './curve-programs.ts';
import {PlaneCapFan,type FanState,type FanResult} from './plane-cap-fan.ts';
import {returnRailPlans} from './return-rail.ts';
import {SurfaceTone} from './surface-tone.ts';
import {SourceNeighborhoods} from './source-neighborhoods.ts';
import {NativeSurfaceVisibility} from './native-surface.ts';
import type {PosedGraph} from './posed-graph.ts';
import type {DrawingFieldView} from './field-readback.ts';
import type {NativePattern} from './native-layout.ts';
import type {NativeGlyph} from './native-page.ts';
import type {NativeWhitespacePolicy} from './text/native-whitespace.ts';

/** The producers disconnected in 51e46a87f9, restored at the existing
 * construction seam. This proposes patterns, NEVER publishes a second page
 * or reinstates the old layout selector. Visibility and admission still
 * belong to NativeConstruction. These historical programs retain their
 * known limitations; reconnecting them is not visual acceptance. */
export class RetainedProducers {
 private readonly neighborhoods=new SourceNeighborhoods();
 private readonly curves:CurveProgramBank;
 private readonly planes:PlaneCapFan;
 private readonly tone:SurfaceTone;
 private readonly states=new Map<string,FanState>();
 private readonly owners=new Map<string,string>();
 constructor(private readonly glyphs:readonly NativeGlyph[],whitespace:NativeWhitespacePolicy){
  this.curves=new CurveProgramBank(glyphs);this.planes=new PlaneCapFan(glyphs);this.tone=new SurfaceTone(glyphs,whitespace);
 }
 retire(isLive:(owner:string)=>boolean):void{
  for(const [id,owner]of this.owners)if(!isLive(owner)){this.states.delete(id);this.owners.delete(id);}
  for(const owner of this.liveOwners)if(!isLive(owner)){this.neighborhoods.retire(owner);this.curves.retire(owner);this.tone.retire(owner);this.liveOwners.delete(owner);}
 }
 private readonly liveOwners=new Set<string>();
 propose(graphs:readonly PosedGraph[],fields:readonly DrawingFieldView[]):{
  patterns:readonly NativePattern[];owners:ReadonlyMap<string,readonly string[]>;
 }{
  const neighborhoods=this.neighborhoods.select(graphs);
  for(const part of neighborhoods){this.liveOwners.add(part.owner);this.owners.set(part.id,part.owner);}
  let curves=this.curves.compile(neighborhoods,new Map(),fields);
  const planes:Extract<FanResult,{ok:true}>[]=[];
  for(const {id,graph:{relations}}of neighborhoods){
   if(relations.every(relation=>relation.kind==='smooth-contour'))continue;
   const made=this.planes.grow(relations,this.states.get(id),undefined,id);
   if(made.ok)planes.push(made);
  }
  const joins=returnRailPlans(planes,curves,graphs,fields,this.planes,this.glyphs);
  if(joins.terminals.size)curves=this.curves.compile(neighborhoods,joins.terminals,fields);
  const patterns=[...curves.patterns,...joins.planes.map(plane=>plane.pattern)],owners=new Map(curves.patternOwners);
  for(const plane of joins.planes){owners.set(plane.pattern.id,[plane.fullEdges[0]!.owner]);this.states.set(plane.neighborhood,plane.state);}
  if(!patterns.length)return {patterns,owners};
  const surfaces=new NativeSurfaceVisibility(patterns,fields,this.glyphs,graphs);
  return {patterns:this.tone.apply(patterns,owners,fields,surfaces),owners};
 }
}
