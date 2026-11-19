import {nativeSpacePlan,type NativeWhitespacePolicy} from './text/native-whitespace.ts';
import {nativeInterfacePoints,nativeInterfacesMeet} from './text/native-interface.ts';
import {type NativeGlyph,type NativeNeighborhood,type NativeOccurrence} from './native-page.ts';
import {NativePaperIndex} from './native-paper-index.ts';

export interface TerminalRequest {
 readonly group:NativeNeighborhood;
 /** Legal pen span of this already selected source run, not a glyph-fit
  * window. A conflict may move its scalar along this span, never respell it. */
 readonly pens:ReadonlyMap<string,readonly [number,number]>;
 /** Same-pose source visibility at a cut. Only pen legality varies; the
  * producer's selected intact scalar and source run never change here. */
 readonly visibleAt?:ReadonlyMap<string,(pen:number)=>boolean>;
}
export interface TerminalResult {readonly groups:readonly NativeNeighborhood[];readonly unresolved:readonly string[];readonly changed:ReadonlySet<string>}

/** D1 terminal admission, shared by every source producer. The vocabulary and
 * source runs are ALREADY selected. This module only preserves a legal pen or
 * takes the next nearest legal pen for the conflicting intact scalar. There
 * is no image, glyph candidate bank, fit objective or whole-form rejection.
 * All protected paper is present before any placement, including paper from
 * a new neighbor that requires an existing conflicting scalar to move. */
export function terminalAdmission(requests:readonly TerminalRequest[],glyphs:readonly NativeGlyph[],width:number,height:number,whitespace:NativeWhitespacePolicy):TerminalResult{
 const font=new Map(glyphs.map(g=>[g.text,g])),rows=new Map<number,NativeOccurrence[]>(),ink=new Map<number,Set<number>>();
 const papers=new NativePaperIndex(width,height),positions=new Map<string,NativeOccurrence>(),placed=new Map<string,NativeOccurrence>();
 const unresolved:string[]=[],changed=new Set<string>();
 const originals=new Map(requests.flatMap(({group})=>group.occurrences.map(atom=>[atom.id,atom] as const)));
 for(const {group}of requests)for(const paper of group.paper)papers.add(paper);
 const legal=(atom:NativeOccurrence,x:number):boolean=>{
  const glyph=font.get(atom.text)!;if(x<0||x+glyph.advance>width||atom.row<0||(atom.row+1)*17>height)return false;
  const line=rows.get(atom.row)??[];let lo=0,hi=line.length;
  while(lo<hi){const mid=(lo+hi)>>>1;if(line[mid]!.x<x)lo=mid+1;else hi=mid;}
  const same=line[lo]?.x===x&&line[lo]!.text===atom.text&&(line[lo]!.foreground??0xff000000)===(atom.foreground??0xff000000);
  if(!same){
   const before=line[lo-1],after=line[lo];
   if(!nativeSpacePlan(x-(before?before.x+font.get(before.text)!.advance:0),0,!before,false,whitespace))return false;
   if(after&&!nativeSpacePlan(after.x-x-glyph.advance,0,false,false,whitespace))return false;
  }
  for(const q of glyph.points){const px=x+q[0]!,py=atom.row*17+q[1]!;
   if(!same&&ink.get(py)?.has(px))return false;
   if(papers.contains(px+.5,py+.5))return false;
  }
  return true;
 };
 for(const {group,pens,visibleAt}of requests)for(const atom of group.occurrences){
  const prior=placed.get(atom.id);if(prior){if(prior.text!==atom.text||prior.row!==atom.row)throw Error(`Conflicting terminal source occurrence ${atom.id}`);continue;}
  const glyph=font.get(atom.text);if(!glyph)throw Error(`Unavailable terminal scalar ${atom.text}`);
  const span=pens.get(atom.id)??[atom.x,atom.x],min=Math.max(0,Math.ceil(span[0])),max=Math.min(width-glyph.advance,Math.floor(span[1]));
  let left=Math.min(max,Math.floor(atom.x)),right=Math.max(min,left+1),pen:number|undefined;
  if(atom.x<min){left=min-1;right=min;}else if(atom.x>max){left=max;right=max+1;}
  while(left>=min||right<=max){
   const takeLeft=left>=min&&(right>max||Math.abs(left-atom.x)<=Math.abs(right-atom.x)),x=takeLeft?left--:right++;
   // Cursor/ink/paper conflicts are cheap and already disqualify this pen.
   // Only a remaining legal pen needs the source-visibility query. Dense
   // rows otherwise re-evaluate an entire source path at every occupied
   // pixel for every pending scalar, blocking the first frame for minutes.
   if(legal(atom,x)&&(visibleAt?.get(atom.id)?.(x)??true)){pen=x;break;}
  }
  if(pen===undefined){unresolved.push(`${atom.id}:terminal scalar has no legal pen on its visible source run`);changed.add(group.id);continue;}
  const next={...atom,x:pen};placed.set(atom.id,next);if(pen!==atom.x)changed.add(group.id);
  const key=`${next.row}/${next.x}`,shared=positions.get(key);
  if(!shared){
   positions.set(key,next);const line=rows.get(next.row)??[];let i=0;while(i<line.length&&line[i]!.x<next.x)i++;line.splice(i,0,next);rows.set(next.row,line);
   for(const q of glyph.points){const y=next.row*17+q[1]!,pixels=ink.get(y)??new Set<number>();pixels.add(next.x+q[0]!);ink.set(y,pixels);}
  }
 }
 const groups=requests.map(({group})=>{
  const occurrences=group.occurrences.flatMap(atom=>placed.has(atom.id)?[placed.get(atom.id)!]:[]),covered=new Set(occurrences.flatMap(atom=>atom.contributions.map(c=>c.source)));
  const contacts=group.contacts.flatMap(contact=>{
   const from=placed.get(contact.from),to=placed.get(contact.to);if(!from||!to){changed.add(group.id);return [];}
   const oldFrom=originals.get(contact.from),oldTo=originals.get(contact.to);
   const fromAnchor:[number,number]=[contact.fromAnchor[0]+from.x-(oldFrom?.x??from.x),contact.fromAnchor[1]],toAnchor:[number,number]=[contact.toAnchor[0]+to.x-(oldTo?.x??to.x),contact.toAnchor[1]];
   const points=(atom:NativeOccurrence,source:string)=>{const glyph=font.get(atom.text)!;return atom.contributions.filter(c=>c.source===source).flatMap(c=>c.pixels.map(i=>[atom.x+glyph.points[i]![0]!+.5,atom.row*17+glyph.points[i]![1]!+.5] as const));};
   if(!nativeInterfacesMeet(nativeInterfacePoints(fromAnchor,points(from,contact.fromSource)),nativeInterfacePoints(toAnchor,points(to,contact.toSource)))){
    unresolved.push(`${contact.id}:terminal construction still needs its named joint`);changed.add(group.id);return [];
   }
   return [{...contact,fromAnchor,toAnchor}];
  });
  return {...group,occurrences,contacts,requiredSources:group.requiredSources.filter(source=>covered.has(source))};
 });
 return {groups,unresolved,changed};
}
