import {displayCharacters,presentationKindFor,validateTextFrame,nativeCompositionRows,type TextFrame,type TextCharacter,type TextComposition,type PresentationMode} from './text/text-frame.ts';
import {nativeSpacePlan} from './text/native-whitespace.ts';
import {nativePaperContains,type NativeFrame,type NativeGlyph,type NativePaper} from './native-page.ts';

/** Producer-owned support, evaluated from the CURRENT geometry/fields. A
 * configuration change includes relevant visibility, contacts, paper and field
 * changes. Point IDs are source addresses, not cells or visible enumeration. */
export interface NativeSupport {
 readonly configuration:string;
 readonly points:ReadonlyMap<string,readonly [number,number]>;
 readonly owners:readonly {readonly reference:number;readonly source:number;readonly kind?:'entity'|'producer-row'}[];
}
export interface PresentedNativeFrame extends TextFrame {
 readonly nativeRows:readonly string[];
 readonly source:{readonly serial:number;readonly engineFrame:number};
}
interface Group {
 key:string;id:number;keyframe:number;members:number[];occurrences:string[];admissions:string[];
 drawing:TextCharacter[];support:NativeSupport;paper:NativePaper[];
 translation:readonly [number,number];
 transition:TextComposition['transition'];
}
interface GlyphInk {
 readonly glyph:NativeGlyph;readonly rows:ReadonlyMap<number,ReadonlySet<number>>;
 readonly left:number;readonly top:number;readonly right:number;readonly bottom:number;
}
interface PlacedInk {
 readonly ink:GlyphInk;readonly group:string;readonly x:number;readonly y:number;
 readonly left:number;readonly top:number;readonly right:number;readonly bottom:number;
}
function displacement(before:NativeSupport,after:NativeSupport):readonly [number,number]|undefined{
 if(before.configuration!==after.configuration||JSON.stringify(before.owners)!==JSON.stringify(after.owners)||before.points.size!==after.points.size||!before.points.size)return;
 let delta:readonly [number,number]|undefined;
 for(const [id,a]of before.points){
  const b=after.points.get(id);if(!b)return;const dx=b[0]-a[0],dy=b[1]-a[1];
  if(!Number.isFinite(dx)||!Number.isFinite(dy))return;
  if(!delta)delta=[dx,dy];else if(Math.abs(dx-delta[0])>.0001||Math.abs(dy-delta[1])>.0001)return;
 }
 return delta;
}
function combineSupport(ids:readonly string[],supports:ReadonlyMap<string,NativeSupport>):NativeSupport{
 const points=new Map<string,readonly [number,number]>(),configuration:string[]=[],owners=new Map<string,NativeSupport['owners'][number]>();
 for(const id of ids){
  const support=supports.get(id);if(!support)throw Error(`Construction ${id} has no current source support`);
  configuration.push(`${id}:${support.configuration}`);
  for(const owner of support.owners)owners.set(`${owner.kind}/${owner.reference}/${owner.source}`,{...owner});
  for(const [key,point]of support.points){const address=`${id}/${key}`;if(!point.every(Number.isFinite))throw Error(`Nonfinite construction support ${address}`);points.set(address,[point[0],point[1]]);}
 }
 return {configuration:configuration.join('\n'),points,owners:[...owners.values()].sort((a,b)=>a.reference-b.reference||a.source-b.source)};
}

/** Exact native admission and moving presentation are different responsibilities.
 * This adapter retains WHOLE dependency groups, verifies common translation,
 * then produces the existing TextFrame consumed by both paint and export.
 * Input stopping and presentation-mode changes never cause a new keyframe. */
export class NativePresentation {
 private readonly advances:Map<number,number>;private readonly font:Map<number,NativeGlyph>;
 private readonly ink=new Map<number,GlyphInk>();private integralInk=true;
 private readonly fontSha256:string;
 private groups=new Map<string,Group>();private nextId=0;
 private active=new Set<string>();
 constructor(glyphs:readonly NativeGlyph[],fontSha256:string){
  this.fontSha256=fontSha256;
  this.advances=new Map(glyphs.map(g=>[g.text.codePointAt(0)!,g.advance]));this.font=new Map(glyphs.map(g=>[g.text.codePointAt(0)!,g]));
  // Native font masks are immutable. Build their bounds/membership once,
  // not a screen-sized string dictionary of every translated pixel/frame.
  for(const glyph of glyphs){
   if(!glyph.points.length)continue;
   let left=Infinity,top=Infinity,right=-Infinity,bottom=-Infinity;
   const rows=new Map<number,Set<number>>();
   for(const [x,y]of glyph.points){
    this.integralInk&&=Number.isInteger(x)&&Number.isInteger(y);
    left=Math.min(left,x!);top=Math.min(top,y!);right=Math.max(right,x!+1);bottom=Math.max(bottom,y!+1);
    const row=rows.get(y!)??new Set<number>();row.add(x!);rows.set(y!,row);
   }
   this.ink.set(glyph.text.codePointAt(0)!,{glyph,rows,left,top,right,bottom});
  }
 }
 reset():void{this.groups.clear();this.active.clear();this.extent=undefined;}
 /** Visibility loss is not source retirement. Keep offscreen correspondence
  * until the source lifetime/revision actually ends. */
 retire(isLive:(owner:NativeSupport['owners'][number])=>boolean):void{
  for(const [key,group]of this.groups)if(group.support.owners.some(owner=>!isLive(owner))){this.groups.delete(key);this.active.delete(key);}
 }
 /** No camera-only transport shortcut: all declared source supports, including
  * deforming middles and coupled paper, must share one verified displacement. */
 private extent:readonly [number,number]|undefined;
 canRetain(supports:ReadonlyMap<string,NativeSupport>,width:number,height:number):boolean{
  if(this.extent?.[0]!==width||this.extent?.[1]!==height)return false;
  const visible=[...this.active].map(key=>this.groups.get(key)!);
  // A source-visible paper constraint belongs to THIS posed source snapshot.
  // Reuse the glyph construction when valid, but obtain current constraints
  // rather than transporting a closure over an old foreground/depth field.
  if(visible.some(group=>group.paper.some(paper=>paper.visibleAt)))return false;
  const expected=new Set(visible.flatMap(group=>group.admissions));
  return expected.size===supports.size&&[...supports.keys()].every(key=>expected.has(key))&&visible.length>0&&visible.every(group=>{
   try{return displacement(group.support,combineSupport(group.admissions,supports))!==undefined;}catch{return false;}
  });
 }
 present(page:NativeFrame,supports:ReadonlyMap<string,NativeSupport>,serial:number,engineFrame:number,mode:PresentationMode):PresentedNativeFrame{
  // A new native matrix is a new coordinate space, not a translation of
  // the previous one. Its admitted page must seed the retained drawings.
  // Otherwise a Rows/resize edit can keep old out-of-page characters and
  // fail validation forever instead of publishing the newly built page.
  if(this.extent&&(this.extent[0]!==page.width||this.extent[1]!==page.height))this.reset();
  const whitespace=page.whitespace??'constrained';
  const chars:TextCharacter[]=[],rowOffsets:number[]=[],ends:number[]=[];
  for(const [row,text]of page.rows.entries()){
   rowOffsets[row]=chars.length;let x=0;
   for(const scalar of text){const cp=scalar.codePointAt(0)!,advance=this.advances.get(cp);if(advance===undefined)throw Error(`Unknown native presentation scalar ${scalar}`);
    chars.push([cp,x,row*17,advance,0,0xff000000,0xffffffff]);x+=advance;
   }ends[row]=x;
  }
  const occurrenceIndex=new Map(page.occurrences.map(atom=>[atom.id,rowOffsets[atom.row]!+atom.scalar]));
  // Cursor dependencies ask for the owner at a scalar index. Build that
  // inverse once, instead of rescanning every occurrence for every donor.
  const occurrenceAt=new Map<number,NativeFrame['occurrences'][number]>();
  for(const atom of page.occurrences){const index=occurrenceIndex.get(atom.id)!;if(!occurrenceAt.has(index))occurrenceAt.set(index,atom);}
  for(const atom of page.occurrences)if(atom.foreground!==undefined){const index=occurrenceIndex.get(atom.id)!,char=chars[index]!;chars[index]=[char[0],char[1],char[2],char[3],char[4],atom.foreground,char[6]];}
  const groups=page.constructionGroups,parent=groups.map((_,i)=>i);
  const root=(i:number):number=>{while(parent[i]!==i){parent[i]=parent[parent[i]!]!;i=parent[i]!;}return i;};
  const union=(a:number,b:number)=>{a=root(a);b=root(b);if(a===b)return false;parent[Math.max(a,b)]=Math.min(a,b);return true;};
  const groupIndices=new Map(groups.map((group,index)=>[group.id,index]));
  groups.forEach((group,index)=>{for(const id of group.phaseWith??[]){const other=groupIndices.get(id);if(other!==undefined)union(index,other);}});
  const members=new Map<string,number[]>();
  groups.forEach((group,index)=>{for(const id of group.occurrences){const owners=members.get(id)??[];for(const owner of owners)union(index,owner);owners.push(index);members.set(id,owners);}});
  const aliases=new Map(page.occurrenceAliases.map(alias=>[alias.source,alias.occurrence]));
  const sourceGroups=new Map<string,number[]>();groups.forEach((group,index)=>{for(const source of group.requiredSources){const owners=sourceGroups.get(source)??[];owners.push(index);sourceGroups.set(source,owners);}});
  groups.forEach((group,index)=>{
   for(const contact of group.contacts)for(const id of [contact.from,contact.to])for(const owner of members.get(aliases.get(id)??id)??[])union(index,owner);
   for(const paper of group.paper)for(const source of paper.sources)for(const owner of sourceGroups.get(source)??[])union(index,owner);
  });
  // If another construction supplies a necessary scalar cursor between two
  // members, it is a genuine layout dependency. Do not borrow its glyph while
  // transporting it independently or falsely claim the run is native text.
  let changed=true;
  while(changed){
   changed=false;const partitions=new Map<number,Set<number>>();
   groups.forEach((group,index)=>{const r=root(index),set=partitions.get(r)??new Set<number>();for(const id of group.occurrences)set.add(occurrenceIndex.get(id)!);partitions.set(r,set);});
   for(const [r,set]of partitions){
    const ordered=[...set].sort((a,b)=>a-b);
    for(let i=1;i<ordered.length;i++){
     const a=chars[ordered[i-1]!]!,b=chars[ordered[i]!]!;
     if(a[2]!==b[2]||nativeSpacePlan(b[1]-a[1]-a[3],0,false,false,whitespace))continue;
     for(let index=ordered[i-1]!+1;index<ordered[i]!;index++){
      const cp=chars[index]![0];if(cp===32||cp===0x3000)continue;
      const occurrence=occurrenceAt.get(index);if(!occurrence)throw Error('A native cursor donor has no construction owner');
      for(const owner of members.get(occurrence.id)??[])changed=union(r,owner)||changed;
     }
    }
   }
  }
  const partitions=new Map<number,number[]>();groups.forEach((_,index)=>{const r=root(index),list=partitions.get(r)??[];list.push(index);partitions.set(r,list);});
  const next=new Map<string,Group>();
  for(const indices of partitions.values()){
   const admissions=indices.map(index=>groups[index]!.id).sort(),key=admissions.join('\n');
   const occurrences=[...new Set(indices.flatMap(index=>groups[index]!.occurrences))].sort((a,b)=>occurrenceIndex.get(a)!-occurrenceIndex.get(b)!);
   const current=combineSupport(admissions,supports),previous=this.groups.get(key);
   const delta=previous&&previous.occurrences.length===occurrences.length&&previous.occurrences.every((id,i)=>id===occurrences[i])?displacement(previous.support,current):undefined;
   const members=occurrences.map(id=>occurrenceIndex.get(id)!);
   const retained=previous&&delta!==undefined;
   if(!retained&&page.serial!==serial)throw Error('Changed source support requires a newly admitted native keyframe');
   const group:Group={key,id:previous?.id??++this.nextId,keyframe:retained?previous.keyframe:serial,members,occurrences,admissions,
    drawing:retained?previous.drawing:members.map(index=>chars[index]!),support:retained?previous.support:current,
    paper:retained&&page.serial!==serial?previous.paper:indices.flatMap(index=>groups[index]!.paper).map(paper=>retained?{...paper,
     rect:[paper.rect[0]-delta[0],paper.rect[1]-delta[1],paper.rect[2],paper.rect[3]] as const}:paper),
    translation:retained?delta:[0,0],transition:retained?'retained':previous?'deformation':'new'};
   next.set(key,group);
  }
  // Compare actual transported native masks and paper. A candidate-page check
  // cannot authorize a collision between retained drawings. Reset the entire
  // affected dependency group to this admitted keyframe, never hide a member.
  const conflicting=this.collisions([...next.values()]);
  if(conflicting.size&&page.serial!==serial)throw Error('Transported constructions require a joint native neighborhood refit');
  for(const key of conflicting){const group=next.get(key)!;group.keyframe=serial;group.drawing=group.members.map(index=>chars[index]!);group.support=combineSupport(group.admissions,supports);group.paper=groups.filter(g=>group.admissions.includes(g.id)).flatMap(g=>g.paper);group.translation=[0,0];group.transition='collision';}
  // No conflict means the first pass changed nothing: repeating the same
  // native-mask/paper check cannot produce a different answer. Recheck only
  // after resetting a conflicting dependency group to its admitted keyframe.
  if(conflicting.size&&this.collisions([...next.values()]).size)throw Error('Native presentation requires a joint neighborhood refit; admitted candidate drawings still conflict');
  // A paper-only dependency still participates in collision/correspondence
  // above, but has no displayed scalar composition. Never publish an invalid
  // zero-member text group or erase its protected-paper constraint.
  const compositions:TextComposition[]=[...next.values()].filter(group=>group.members.length).map(group=>({id:group.id,keyframe:group.keyframe,origin_px:[0,0],translation_px:group.translation,members:group.members,drawing:group.drawing,
   transition:group.transition}));
  const frame:PresentedNativeFrame={schema:'true_ff.text_frame.v1',frame:serial,engine_frame:engineFrame,row_pitch_px:17,page_shift_px:page.pageShift??[0,0],font_sha256:this.fontSha256,chars,
   composition:{kind:'retained',keyframe:serial,groups:compositions},raster:{origin_px:[0,0],scale:1,size_px:[page.width,page.height]},
   whitespace,page:{extent_px:[page.width,page.height],full_advance_px:11,glyph_only:true,rows:Array.from({length:Math.ceil(page.height/17)},(_,row)=>[ends[row]??0,Array(Math.ceil((page.width-(ends[row]??0))/11)).fill(0xffffffff)])},
   nativeRows:page.rows,source:{serial,engineFrame}};
  frame.presentation={kind:presentationKindFor(mode,frame)};validateTextFrame(frame,this.advances);
  // This also verifies each dependency group's own literal native spelling.
  for(const group of compositions)nativeCompositionRows(group,whitespace);
  // Invisible groups retain correspondence, not obsolete full-frame field
  // snapshots captured by a former paper constraint. Reappearance supplies
  // fresh paper through NativeConstruction before it can be published.
  for(const [key,group]of this.groups)if(!next.has(key)&&group.paper.some(paper=>paper.visibleAt))this.groups.set(key,{...group,paper:group.paper.filter(paper=>!paper.visibleAt)});
  for(const [key,group]of next)this.groups.set(key,group);this.active=new Set(next.keys());this.extent=[page.width,page.height];return frame;
 }
 private collisions(groups:readonly Group[]):Set<string>{
  // Preserve the general fractional-mask contract; installed native raster
  // glyphs use integral pixels and take the retained-mask broad phase below.
  if(!this.integralInk)return this.fractionalCollisions(groups);
  const conflict=new Set<string>(),bins=new Map<number,PlacedInk[]>();
  for(const group of groups)for(const char of group.drawing){
   const ink=this.ink.get(char[0]);if(!ink)continue;
   // ceil(pen + INTEGER_PIXEL - .5) is exactly this integer translation,
   // including negative and fractional retained-group displacements.
   const x=Math.ceil(Math.fround(char[1]+group.translation[0])-.5),y=Math.ceil(Math.fround(char[2]+group.translation[1])-.5);
   const placed:PlacedInk={ink,group:group.key,x,y,left:x+ink.left,top:y+ink.top,right:x+ink.right,bottom:y+ink.bottom};
   for(let bin=Math.floor(placed.top/17);bin<=Math.floor((placed.bottom-1)/17);bin++){
    const row=bins.get(bin)??[];row.push(placed);bins.set(bin,row);
   }
  }
  for(const [bin,row]of bins){
   row.sort((a,b)=>a.left-b.left);
   for(let i=0;i<row.length;i++){
    const a=row[i]!;
    for(let j=i+1;j<row.length&&row[j]!.left<a.right;j++){
     const b=row[j]!;
     if(a.group===b.group||a.top>=b.bottom||b.top>=a.bottom||bin!==Math.floor(Math.max(a.top,b.top)/17))continue;
     const [small,large]=a.ink.glyph.points.length<=b.ink.glyph.points.length?[a,b]:[b,a];
     if(small.ink.glyph.points.some(([x,y])=>large.ink.rows.get(small.y+y!-large.y)?.has(small.x+x!-large.x))){conflict.add(a.group);conflict.add(b.group);}
    }
   }
  }
  for(const group of groups)for(const paper of group.paper){
   const [dx,dy]=group.translation,[x,y,w,h]=paper.rect,left=x+dx,top=y+dy,right=left+w,bottom=top+h;
   if(w<=0||h<=0)continue;
   for(let bin=Math.floor(top/17);bin<=Math.ceil(bottom/17)-1;bin++){
    const row=bins.get(bin);if(!row)continue;
    for(const glyph of row){
     if(glyph.left>=right)break;
     if(glyph.group===group.key||glyph.right<=left||glyph.top>=bottom||glyph.bottom<=top||bin!==Math.floor(Math.max(top,glyph.top)/17))continue;
     if(glyph.ink.glyph.points.some(([px,py])=>nativePaperContains(paper,glyph.x+px!+.5-dx,glyph.y+py!+.5-dy))){conflict.add(group.key);conflict.add(glyph.group);}
    }
   }
  }
  return conflict;
 }
 private fractionalCollisions(groups:readonly Group[]):Set<string>{
  const conflict=new Set<string>(),occupied=new Map<string,string>(),points=new Map<number,{x:number;group:string}[]>();
  const hasPaper=groups.some(group=>group.paper.length);
  for(const group of groups)for(const char of group.drawing){
   const glyph=this.font.get(char[0])!;const pen=[Math.fround(char[1]+group.translation[0]),Math.fround(char[2]+group.translation[1])];
   for(const p of glyph.points){
    const x=Math.ceil(pen[0]!+p[0]!-.5),y=Math.ceil(pen[1]!+p[1]!-.5),key=`${x},${y}`,prior=occupied.get(key);
    if(prior&&prior!==group.key){conflict.add(prior);conflict.add(group.key);}occupied.set(key,group.key);
    if(hasPaper){const row=points.get(y)??[];row.push({x:x+.5,group:group.key});points.set(y,row);}
   }
  }
  let firstRow=Infinity,lastRow=-Infinity;for(const [y,row]of points){firstRow=Math.min(firstRow,y);lastRow=Math.max(lastRow,y);row.sort((a,b)=>a.x-b.x);}
  for(const group of groups)for(const paper of group.paper){const [dx,dy]=group.translation,[x,y,w,h]=paper.rect;
   for(let py=Math.max(firstRow,Math.floor(y+dy));py<=Math.min(lastRow,Math.ceil(y+dy+h)-1);py++){
    const row=points.get(py);if(!row)continue;let lo=0,hi=row.length;
    while(lo<hi){const mid=(lo+hi)>>>1;if(row[mid]!.x<x+dx)lo=mid+1;else hi=mid;}
    for(let i=lo;i<row.length&&row[i]!.x<x+dx+w;i++){const point=row[i]!;if(point.group!==group.key&&nativePaperContains(paper,point.x-dx,py+.5-dy)){conflict.add(group.key);conflict.add(point.group);}}
   }
  }
  return conflict;
 }
}
