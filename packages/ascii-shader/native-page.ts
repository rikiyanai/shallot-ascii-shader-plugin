import {nativeSpaceCodes,type NativeWhitespacePolicy} from './text/native-whitespace.ts';
import {nativeInterfacePoints,nativeInterfacesMeet} from './text/native-interface.ts';

export interface NativeGlyph {readonly text:string;readonly advance:number;readonly points:readonly (readonly number[])[]}
const wholeInk=new WeakMap<NativeGlyph['points'],readonly number[]>();
/** One immutable whole-mask membership per native glyph. Repeated intact
 * occurrences share it; no frame recreates a pixel-index array per scalar. */
export function nativeWholeInk(glyph:NativeGlyph):readonly number[]{let result=wholeInk.get(glyph.points);if(!result){result=Object.freeze(glyph.points.map((_,index)=>index));wholeInk.set(glyph.points,result);}return result;}
export interface InkContribution {readonly source:string;readonly kind:'path'|'region'|'particle'|'light'|'text';readonly pixels:readonly number[];
  /** The complete source-edge interval owned by this intact native unit.
   * Visibility never re-indexes its correspondence from clipped fragments. */
  readonly parameter?:readonly [number,number];
  readonly support?:{readonly id:string;readonly point:readonly [number,number];readonly configuration:string};}
export interface NativeOccurrence {
  readonly id:string;readonly text:string;readonly x:number;readonly row:number;
  readonly foreground?:number;
  /** One intact occurrence may contribute different ink to several sources. */
  readonly contributions:readonly InkContribution[];
}
export interface NativeContact {
  readonly id:string;readonly from:string;readonly to:string;
  /** Named source participants constrain ink membership. Shared glyph ink
   * belonging only to another path cannot donate this contact's interface. */
  readonly fromSource:string;readonly toSource:string;
  readonly fromPort:string;readonly toPort:string;
  readonly fromAnchor:readonly [number,number];readonly toAnchor:readonly [number,number];
  readonly relation:string;
}
export interface NativePaper {
  readonly id:string;readonly rect:readonly [number,number,number,number];readonly sources:readonly string[];
  /** Current-source visibility of this empty-space constraint, in coordinates
   * relative to rect. Paper is queried where intact ink would violate it;
   * it is not a viewport-sized image to rasterize before drawing anything. */
  readonly visibleAt?:(x:number,y:number)=>boolean;
}
export function nativePaperContains(paper:NativePaper,x:number,y:number):boolean{
  const [left,top,width,height]=paper.rect,u=x-left,v=y-top;
  return u>=0&&u<width&&v>=0&&v<height&&(!paper.visibleAt||paper.visibleAt(u,v));
}
export interface NativeNeighborhood {
  readonly id:string;readonly occurrences:readonly NativeOccurrence[];
  readonly contacts:readonly NativeContact[];readonly paper:readonly NativePaper[];
  readonly requiredSources:readonly string[];
  /** Native construction relationship, including apertures and shared
   * returns. This does not assert physical attachment of the sources. */
  readonly phaseWith?:readonly string[];
}
export interface NativeTextPage {
  readonly serial:number;readonly width:number;readonly height:number;readonly rows:readonly string[];
  readonly whitespace?:NativeWhitespacePolicy;
  /** Common native-page window transform, independent of moving groups. */
  readonly pageShift?:readonly [number,number];
}
export interface NativeFrame extends NativeTextPage {
  /** Annotations index the scalar tape; they never own independent display pens. */
  readonly occurrences:readonly (NativeOccurrence&{readonly scalar:number})[];
  readonly neighborhoods:readonly string[];
  readonly occurrenceAliases:readonly {readonly source:string;readonly occurrence:string}[];
  /** Original admission memberships survive scalar merging. Shared occurrences,
   * contacts and coupled paper close these into presentation dependencies. */
  readonly constructionGroups:readonly {readonly id:string;readonly occurrences:readonly string[];
    readonly contacts:readonly NativeContact[];readonly paper:readonly NativePaper[];readonly requiredSources:readonly string[];readonly phaseWith?:readonly string[]}[];
}
export class NativeCommitError extends Error {
  readonly neighborhood:string;readonly obligation:string;
  constructor(neighborhood:string,obligation:string){super(`${neighborhood}: ${obligation}`);this.name='NativeCommitError';this.neighborhood=neighborhood;this.obligation=obligation;}
}

/** A transactional native page. Selection must supply the complete cooperating
 * neighborhood at legal native cursors. This module cannot move or discard a
 * member to make a partial drawing look successful. Display and export consume
 * the same scalar tape; provenance/contact/paper are annotations on that tape. */
export class NativePage {
  private readonly font:Map<string,NativeGlyph>;
  private readonly whitespace:NativeWhitespacePolicy;
  constructor(glyphs:readonly NativeGlyph[],whitespace:NativeWhitespacePolicy='constrained'){
    this.whitespace=whitespace;
    this.font=new Map(glyphs.map(g=>[g.text,g]));
    for(const glyph of glyphs)if([...glyph.text].length!==1||!Number.isInteger(glyph.advance)||glyph.advance<=0)throw Error('Native font has an invalid scalar or advance');
  }
  commit(neighborhoods:readonly NativeNeighborhood[],width:number,height:number,serial:number):NativeFrame{
    if(![width,height,serial].every(Number.isSafeInteger)||width<=0||height<=0||serial<0)throw Error('Invalid native page extent or serial');
    const occurrences=new Map<string,{occurrence:NativeOccurrence;neighborhood:string}>();
    const positions=new Map<string,NativeOccurrence>();
    const aliases=new Map<string,string>();
    const identities=new Map<string,NativeOccurrence>();
    const error=(group:string,message:string):never=>{throw new NativeCommitError(group,message);};
    for(const neighborhood of neighborhoods){
      const covered=new Set<string>();
      for(const candidate of neighborhood.occurrences){
        const glyph=this.font.get(candidate.text);
        if(!glyph||[...candidate.text].length!==1||!glyph.points.length)error(neighborhood.id,`unavailable intact occurrence ${candidate.id}`);
        if(!Number.isSafeInteger(candidate.x)||!Number.isSafeInteger(candidate.row)||candidate.x<0||candidate.row<0||candidate.x+glyph!.advance>width||candidate.row*17+17>height)error(neighborhood.id,`occurrence outside native page ${candidate.id}`);
        const claimed=new Set<number>();let completeInk=false;
        for(const contribution of candidate.contributions){
          if(!contribution.source||!contribution.pixels.length)error(neighborhood.id,`empty ink contribution ${candidate.id}`);
          covered.add(contribution.source);
          if(contribution.pixels===nativeWholeInk(glyph!)){completeInk=true;continue;}
          for(const pixel of contribution.pixels){
            if(!Number.isInteger(pixel)||pixel<0||pixel>=glyph!.points.length)error(neighborhood.id,`invalid whole-mask contribution ${candidate.id}`);
            claimed.add(pixel);
          }
        }
        if(!completeInk&&claimed.size!==glyph!.points.length)error(neighborhood.id,`unowned ink in ${candidate.id}`);
        const existing=identities.get(candidate.id);
        if(existing&&(existing.x!==candidate.x||existing.row!==candidate.row||existing.text!==candidate.text))error(neighborhood.id,`conflicting occurrence identity ${candidate.id}`);
        identities.set(candidate.id,candidate);
        const key=`${candidate.row}:${candidate.x}`,shared=positions.get(key);
        if(shared){
          if(shared.text!==candidate.text)error(neighborhood.id,`incompatible intact glyphs at ${key}`);
          if((shared.foreground??0xff000000)!==(candidate.foreground??0xff000000))error(neighborhood.id,`incompatible source colors at ${key}`);
          const merged={...shared,contributions:[...shared.contributions,...candidate.contributions]};
          positions.set(key,merged);occurrences.set(shared.id,{occurrence:merged,neighborhood:neighborhood.id});aliases.set(candidate.id,shared.id);
        }else{positions.set(key,candidate);occurrences.set(candidate.id,{occurrence:candidate,neighborhood:neighborhood.id});aliases.set(candidate.id,candidate.id);}
      }
      for(const source of neighborhood.requiredSources)if(!covered.has(source))error(neighborhood.id,`unfulfilled source ${source}`);
    }
    const get=(id:string)=>occurrences.get(aliases.get(id)??id)?.occurrence;
    const ink=(occurrence:NativeOccurrence,source?:string):readonly (readonly [number,number])[]=>{
      const allowed=source===undefined?undefined:new Set(occurrence.contributions.filter(c=>c.source===source).flatMap(c=>c.pixels));
      return this.font.get(occurrence.text)!.points.flatMap((p,index)=>allowed&&!allowed.has(index)?[]:[[occurrence.x+p[0]!+.5,occurrence.row*17+p[1]!+.5] as const]);
    };
    // Index the SAME native pixel centres once. Protected paper must not
    // allocate a fresh whole-page ink image for every constraint rectangle.
    const inkRows=new Map<number,number[]>();
    // Paper-free constructions still verify every contribution, named
    // contact and scalar cursor below. They have no paper query to serve,
    // so do not raster-index and sort their entire drawing for nothing.
    if(neighborhoods.some(neighborhood=>neighborhood.paper.length)){
      for(const {occurrence}of occurrences.values())for(const point of this.font.get(occurrence.text)!.points){
        const y=occurrence.row*17+point[1]!,row=inkRows.get(y)??[];row.push(occurrence.x+point[0]!+.5);inkRows.set(y,row);
      }
      for(const row of inkRows.values())row.sort((a,b)=>a-b);
    }
    for(const neighborhood of neighborhoods){
      for(const contact of neighborhood.contacts){
        const from=get(contact.from),to=get(contact.to);if(!from||!to)error(neighborhood.id,`missing contact member ${contact.id}`);
        const a=nativeInterfacePoints(contact.fromAnchor,ink(from!,contact.fromSource)),b=nativeInterfacePoints(contact.toAnchor,ink(to!,contact.toSource));
        if(!contact.relation||!contact.fromSource||!contact.toSource||!contact.fromPort||!contact.toPort||!nativeInterfacesMeet(a,b))error(neighborhood.id,`broken named contact ${contact.id}`);
      }
      for(const paper of neighborhood.paper){
        const [x,y,w,h]=paper.rect;
        if(!paper.sources.length||![x,y,w,h].every(Number.isInteger)||w<=0||h<=0)error(neighborhood.id,`invalid protected paper ${paper.id}`);
        for(let py=Math.max(0,Math.floor(y));py<Math.min(height,y+h);py++){
          const row=inkRows.get(py);if(!row)continue;let lo=0,hi=row.length;
          while(lo<hi){const mid=(lo+hi)>>>1;if(row[mid]!<x)lo=mid+1;else hi=mid;}
          for(let i=lo;i<row.length&&row[i]!<x+w;i++)if(nativePaperContains(paper,row[i]!,py+.5))error(neighborhood.id,`ink occupies protected paper ${paper.id}`);
        }
      }
    }
    const rows:string[]=[],committed:(NativeOccurrence&{scalar:number})[]=[];
    const ordered=[...positions.values()].sort((a,b)=>a.row-b.row||a.x-b.x||a.id.localeCompare(b.id));
    let index=0;
    for(let row=0;row<Math.ceil(height/17);row++){
      let x=0,scalar=0,text='';
      while(index<ordered.length&&ordered[index]!.row===row){
        const occurrence=ordered[index++]!,owner=occurrences.get(occurrence.id)!.neighborhood;
        const spaces=nativeSpaceCodes(occurrence.x-x,0,x===0,false,this.whitespace);
        if(!spaces)error(owner,`illegal scalar cursor before ${occurrence.id}`);
        text+=String.fromCodePoint(...spaces!);scalar+=spaces!.length;
        committed.push({...occurrence,scalar});text+=occurrence.text;scalar++;x=occurrence.x+this.font.get(occurrence.text)!.advance;
      }
      rows.push(text);
    }
    const constructionGroups=neighborhoods.map(group=>({id:group.id,occurrences:[...new Set(group.occurrences.map(atom=>aliases.get(atom.id)!))],contacts:group.contacts,paper:group.paper,requiredSources:group.requiredSources,phaseWith:group.phaseWith}));
    return Object.freeze({serial,width,height,whitespace:this.whitespace,rows:Object.freeze(rows),occurrences:Object.freeze(committed),neighborhoods:Object.freeze(neighborhoods.map(n=>n.id)),
      occurrenceAliases:Object.freeze([...aliases].map(([source,occurrence])=>({source,occurrence}))),constructionGroups:Object.freeze(constructionGroups)});
  }
}
