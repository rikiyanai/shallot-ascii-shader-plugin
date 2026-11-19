import type {DrawingFieldView,FieldSample} from './field-readback.ts';
import type {NativeGlyph,NativeOccurrence,NativePaper} from './native-page.ts';
import type {NativePattern,NativeSurface} from './native-layout.ts';
import {sourceSurfacePoint} from './native-surface.ts';
import {nativeSpaceCodes,type NativeWhitespacePolicy} from './text/native-whitespace.ts';

type Point=readonly [number,number];
const ROW=17,CLEAR=11;
/** One constant repeated pattern (D6): a member every STEP native px, rows
 * staggered by half a step. With the 3 px ladder glyphs the gap between
 * members is 21 px, one full and two half spaces, legal in both whitespace
 * policies. */
const STEP=24;
/** D6 ladder, graded only by substituting the glyph. Level 1 ("spaced")
 * keeps every other row of the same pattern. */
const LADDER=['',':',':','i'] as const;
/** Same fixed n·L thresholds as the D10 underdrawing bands. */
const bandLevel=(nl:number)=>nl>=.7?0:nl>=.4?1:nl>=.15?2:3;
const luminance=(c:readonly number[])=>c[0]!*.2126+c[1]!*.7152+c[2]!*.0722;
/** Column-major 4x4 times (x,y,z,w). */
const apply=(m:ArrayLike<number>,v:readonly number[])=>[0,1,2,3].map(i=>m[i]!*v[0]!+m[4+i]!*v[1]!+m[8+i]!*v[2]!+m[12+i]!*v[3]!);
const parity=(n:number)=>((n%2)+2)%2;

interface Probe {sample:FieldSample;normal:readonly number[];toCamera:readonly number[];shown?:number}

/** D6 region tone from the D10 light bands, chosen with the construction.
 *
 * A construction's surface is toned where n·L toward the scene's key light
 * (or the camera when the scene has none) falls below the lit band, or where
 * the displayed value shows a cast shadow on it. The pattern is fixed to the
 * construction's own origin, so it moves exactly as the construction's
 * outline does, and its density never changes with the band: only the glyph
 * does. Members go only into row gaps bounded on both sides by the
 * construction's own ink, one full space clear of it, on the owner's visible
 * pixels. The lit paper the construction declared is cut only around each
 * member; a toned construction that has no legal pen falls back to the same
 * construction without tone (NativeConstruction.placePattern). */
export class RegionTone {
 private readonly font:Map<string,NativeGlyph>;
 private views:DrawingFieldView[]=[];
 private lit=new Map<string,number>();
 constructor(glyphs:readonly NativeGlyph[],private readonly whitespace:NativeWhitespacePolicy){
  this.font=new Map(glyphs.map(glyph=>[glyph.text,glyph]));
  for(const text of new Set(LADDER.slice(1)))if(!this.font.get(text)?.points.length)throw Error(`Region tone glyph ${text} is not in the native font`);
 }
 begin(views:readonly DrawingFieldView[]):void{this.views=[...views];this.lit=new Map();}

 /** Tone members for one construction. `absolute` are its atoms at page
  * rows/x (after whole-part placement, before the shared phase). */
 members(p:NativePattern,absolute:readonly NativeOccurrence[]):{atoms:NativeOccurrence[];paper:NativePaper[]}|undefined{
  const surfaces=p.surfaces?.filter(surface=>surface.kind!=='opening'&&surface.charts.length);
  if(!surfaces?.length||!this.views.length||!absolute.length)return;
  const [ox,oy]=p.origin;
  const rows=new Map<number,{x:number;end:number}[]>();
  for(const atom of absolute){const row=atom.row-oy,list=rows.get(row)??[];list.push({x:atom.x-ox,end:atom.x-ox+this.font.get(atom.text)!.advance});rows.set(row,list);}
  // D9: the smallest tier is outline only.
  if(rows.size<4)return;
  for(const list of rows.values())list.sort((a,b)=>a.x-b.x);
  const atoms:NativeOccurrence[]=[];const cuts:[number,number,number,number][]=[];
  for(const surface of surfaces){
   const camera=surface.owner.slice(0,surface.owner.indexOf('/'));
   const views=this.views.filter(view=>view.camera===camera);if(!views.length)continue;
   const sites:{row:number;column:number;x:number;lo:number;hi:number;probe:Probe;view:DrawingFieldView}[]=[];
   for(const [row,list]of rows)for(let i=1;i<list.length;i++){
    const lo=list[i-1]!.end+CLEAR,hi=list[i]!.x-CLEAR,offset=parity(row)*STEP/2;
    for(let column=Math.ceil((lo-offset)/STEP);column*STEP+offset+3<=hi;column++){
     const x=column*STEP+offset,found=this.probe([x+1.5,row*ROW+ROW/2],surface,views);
     if(found)sites.push({row,column,x,lo,hi,...found});
    }
   }
   if(sites.length<3)continue;
   // One lit reference per surface per frame, so every placement attempt of
   // this construction spells the same members.
   const key=`${p.id}|${surface.id}`;
   let lit=this.lit.get(key);if(lit===undefined){lit=Math.max(...sites.map(site=>site.probe.shown??0));this.lit.set(key,lit);}
   for(const site of sites){
    const {probe,view}=site,L=view.light?.length===3?view.light:probe.toCamera;
    let level=bandLevel(Math.max(0,probe.normal[0]!*L[0]!+probe.normal[1]!*L[1]!+probe.normal[2]!*L[2]!));
    // Cast shadow: the displayed value on this receiver falls well below the
    // receiver's own lit value (fixed ratios across frames).
    if(probe.shown!==undefined&&lit>.05){const ratio=probe.shown/lit;if(ratio<.3)level=3;else if(ratio<.55)level=Math.max(level,2);}
    if(!level||level===1&&parity(site.row))continue;
    const text=LADDER[level]!,glyph=this.font.get(text)!;
    const placed=this.place(site.row,site.x,glyph,site.lo,site.hi,rows.get(site.row)!,atoms,p.origin);
    if(placed===undefined)continue;
    if(!glyph.points.every(q=>this.probe([placed+q[0]!+.5,site.row*ROW+q[1]!+.5],surface,[view])))continue;
    atoms.push({id:`${p.id}/tone/${surface.id}/${site.row+oy}/${site.column}/${text}`,text,x:placed+ox,row:site.row+oy,
     contributions:[{source:`${p.id}/tone`,kind:'region',pixels:glyph.points.map((_,index)=>index)}]});
    cuts.push([placed-1,site.row*ROW,glyph.advance+2,ROW]);
   }
  }
  if(!atoms.length)return;
  let paper=[...p.paper];for(const cut of cuts)paper=paper.flatMap(item=>subtract(item,cut));
  return {atoms,paper};
 }

 /** The member's pen inside its bounded gap: legal native whitespace on both
  * sides, one full space clear of line ink and of other members. */
 private place(row:number,x:number,glyph:NativeGlyph,lo:number,hi:number,line:readonly {x:number;end:number}[],members:readonly NativeOccurrence[],origin:readonly [number,number]):number|undefined{
  const ink=[...line.map(item=>[item.x,item.end] as const),
   ...members.filter(atom=>atom.row===row+origin[1]).map(atom=>[atom.x-origin[0],atom.x-origin[0]+this.font.get(atom.text)!.advance] as const)];
  for(const d of [0,1,-1,2,-2,3,-3,4,-4,5,-5]){
   const at=x+d,end=at+glyph.advance;if(at<lo||end>hi)continue;
   let left=-Infinity,right=Infinity,overlap=false;
   for(const [a,b]of ink){if(b<=at)left=Math.max(left,b);else if(a>=end)right=Math.min(right,a);else{overlap=true;break;}}
   if(overlap||!Number.isFinite(left)||!Number.isFinite(right)||at-left<CLEAR||right-end<CLEAR)continue;
   if(!nativeSpaceCodes(at-left,0,false,false,this.whitespace)||!nativeSpaceCodes(right-end,0,false,false,this.whitespace))continue;
   return at;
  }
 }

 private probe(point:Point,surface:NativeSurface,views:readonly DrawingFieldView[]):{probe:Probe;view:DrawingFieldView}|undefined{
  const source=sourceSurfacePoint(point,surface);if(!source)return;
  for(const view of views){
   const {field,rect,owners,camera}=view;
   const fx=(source[0]-rect.x)/rect.width*field.width,fy=(source[1]-rect.y)/rect.height*field.height;
   if(fx<0||fy<0||fx>=field.width||fy>=field.height)continue;
   const sample=field.sample(fx,fy);if(!sample||sample.depth<=0)return;
   const reference=sample.kind==='entity'?owners.get(sample.owner):sample.owner;
   if(`${camera}/${sample.source}/${sample.kind}/${reference}`!==surface.owner)return;
   // Toward the camera from the receiver (reverse-Z: depth 1 is the near plane).
   const M=field.source.worldFromClip,ndc=[fx/field.width*2-1,1-fy/field.height*2];
   const unproject=(d:number)=>{const q=apply(M,[ndc[0]!,ndc[1]!,d,1]);return [q[0]!/q[3]!,q[1]!/q[3]!,q[2]!/q[3]!];};
   const world=unproject(sample.depth),near=unproject(1),v=[near[0]!-world[0]!,near[1]!-world[1]!,near[2]!-world[2]!];
   const length=Math.hypot(v[0]!,v[1]!,v[2]!);if(!Number.isFinite(length)||!length)return;
   const toCamera=v.map(c=>c/length),raw=sample.normal,flip=raw[0]*toCamera[0]!+raw[1]*toCamera[1]!+raw[2]*toCamera[2]!<0?-1:1;
   return {view,probe:{sample,normal:raw.map(c=>c*flip),toCamera,shown:sample.appearance?luminance(sample.appearance):undefined}};
  }
 }
}

/** Cut one member's box out of declared lit paper. Visibility predicates are
 * relative to their rect, so each piece keeps the original coordinates. */
function subtract(paper:NativePaper,cut:readonly [number,number,number,number]):NativePaper[]{
 const [x,y,w,h]=paper.rect,[a,b,c,d]=cut,left=Math.max(x,a),right=Math.min(x+w,a+c),top=Math.max(y,b),bottom=Math.min(y+h,b+d);
 if(left>=right||top>=bottom)return [paper];
 const rects:[number,number,number,number][]=[[x,y,w,top-y],[x,bottom,w,y+h-bottom],[x,top,left-x,bottom-top],[right,top,x+w-right,bottom-top]];
 const visibleAt=paper.visibleAt;
 return rects.filter(([,,width,height])=>width>0&&height>0).map((rect,i)=>({...paper,id:`${paper.id}/tone-${i}`,rect,
  visibleAt:visibleAt&&((u:number,v:number)=>visibleAt(u+rect[0]-x,v+rect[1]-y))}));
}
