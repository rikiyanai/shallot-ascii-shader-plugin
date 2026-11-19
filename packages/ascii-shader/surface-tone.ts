import library from '../../assets/grammar/aahub-role-lookup-3493185.v1.json' with {type:'json'};
import {nativeProgramPhase,type NativeCursorConstraint} from './native-layout.ts';
import type {NativeGlyph,NativeOccurrence,NativePaper} from './native-page.ts';
import type {NativePattern,NativeSurface} from './native-layout.ts';
import {nativeSurfacePoint,sourceSurfacePoint,NativeSurfaceVisibility} from './native-surface.ts';
import type {DrawingFieldView,FieldSample} from './field-readback.ts';
import type {NativeWhitespacePolicy} from './text/native-whitespace.ts';

interface Cell {view:DrawingFieldView;sample:FieldSample;x:number;y:number;distance:number}
interface Region {owner:string;maximum:number;minimum:number;cells:Map<number,Cell>;count:number}
interface BandAtom {text:string;x:number;row:number;sourceRow:number;sourceScalar:number}
const light=(sample:FieldSample)=>{const color=sample.appearance??sample.radiance;return color[0]*.2126+color[1]*.7152+color[2]*.0722;};
const shade=(sample:FieldSample,maximum:number)=>Math.min(1,Math.max(0,(1-light(sample))*.65+(1-light(sample)/Math.max(maximum,1e-9))*.35));
const uvSites=[.125,.375,.625,.875] as const;

/** A bounded native surface-turn function, compiled from adjacent source rows.
 * The source UV chart owns its phase; camera pixels never seed a tone lattice.
 * Its complete three-row passage is selected with the contour's scalar cuts
 * BEFORE publication. No independent punctuation overlay exists.
 */
export class SurfaceTone {
 private readonly font:Map<string,NativeGlyph>;private readonly rows:BandAtom[][]=[];
 private readonly whitespace:NativeWhitespacePolicy;
 private readonly sites=new Map<string,number>();private readonly phases=new Map<string,number>();
 retire(owner:string):void{
  for(const map of [this.sites,this.phases])for(const key of map.keys())if(key===owner||key.startsWith(`${owner}/`))map.delete(key);
 }
 constructor(glyphs:readonly NativeGlyph[],whitespace:NativeWhitespacePolicy='constrained'){
  this.whitespace=whitespace;
  this.font=new Map(glyphs.map(g=>[g.text,g]));
  const source=library.entries.find(e=>e.id==='sk-cup176-cylinder-tone')!;
  for(let row=0;row<3;row++){
   let x=source.native.row_pen_x[row]!;const atoms:BandAtom[]=[];
   for(const [scalar,text]of [...source.native.rows[row]!.replace(/\\u([0-9a-fA-F]{4})/g,(_,value:string)=>String.fromCharCode(parseInt(value,16)))].entries()){
    if('.:;'.includes(text))atoms.push({text,x,row,sourceRow:source.native.row_numbers[0]!+row,sourceScalar:scalar});
    const glyph=this.font.get(text);if(!glyph)throw Error(`Surface-turn source is missing ${text}`);x+=glyph.advance;
   }
   this.rows.push(atoms.slice(-6));
  }
 }
 private regions(views:readonly DrawingFieldView[]):Map<string,Region>{
  const regions=new Map<string,Region>();
  for(const view of views){const field=view.field;
   for(let y=2;y<field.height;y+=4)for(let x=2;x<field.width;x+=4){
    const sample=field.sample(x,y);if(!sample||!sample.depth||sample.radiance[3]<.75)continue;
    if(!sample.hasUV)continue;
    const reference=sample.kind==='entity'?view.owners.get(sample.owner):sample.owner;if(reference===undefined)continue;
    const owner=`${view.camera}/${sample.source}/${sample.kind}/${reference}`;
    let region=regions.get(owner);if(!region){region={owner,maximum:0,minimum:Infinity,cells:new Map(),count:0};regions.set(owner,region);}
    const value=light(sample);if(!Number.isFinite(value)||!sample.uv.every(Number.isFinite))continue;
    region.maximum=Math.max(region.maximum,value);region.minimum=Math.min(region.minimum,value);region.count++;
    for(const [id,u]of uvSites.entries()){
     const distance=(sample.uv[0]-u)**2+(sample.uv[1]-.5)**2,previous=region.cells.get(id);
     if(!previous||distance<previous.distance)region.cells.set(id,{view,sample,x:view.rect.x+(x+.5)/field.width*view.rect.width,y:view.rect.y+(y+.5)/field.height*view.rect.height,distance});
    }
   }
  }
  return regions;
 }
 apply(patterns:readonly NativePattern[],patternOwners:ReadonlyMap<string,readonly string[]>,views:readonly DrawingFieldView[],visibility=new NativeSurfaceVisibility(patterns,views)):readonly NativePattern[]{
  if(!views.length)return patterns;
  const regions=this.regions(views),live=new Set<string>();
  const result=patterns.map(pattern=>{
   if(pattern.surfaces?.every(surface=>surface.kind==='opening'))return pattern;
   const surfaces=[...new Set(pattern.surfaces?.map(surface=>surface.owner)??[])],owners=surfaces.length===1?surfaces:patternOwners.get(pattern.id);if(owners?.length!==1)return pattern;
   const region=regions.get(owners[0]!);if(!region||region.count<80||region.maximum-region.minimum<.1&&region.minimum>.6)return pattern;
   live.add(pattern.id);
   const eligible=[...region.cells].filter(([,cell])=>cell.distance<1/256&&shade(cell.sample,region.maximum)>.25);
   const previous=this.sites.get(pattern.id);
   // Each alternative is a COMPLETE source passage on a fixed receiver UV
   // landmark. Retain its landmark while the passage remains supportable;
   // never move separate punctuation members to make them fit.
   eligible.sort(([a],[b])=>Number(b===previous)-Number(a===previous));
   for(const [site,coarse]of eligible){
   const cell=this.anchor(coarse,uvSites[site]!);if(!cell)continue;
   const surface=pattern.surfaces?.find(surface=>surface.owner===region.owner),native=surface&&nativeSurfacePoint([cell.x,cell.y],surface);if(!surface||!native)continue;
   const key=`${pattern.id}/surface-turn/uv-${site}`,anchorRow=pattern.origin[1]+Math.round((native[1]-8)/17),origin=pattern.origin[0]+Math.round(native[0]-9);
   const interiorFirst=pattern.origin[1]+Math.min(...pattern.occurrences.map(atom=>atom.row))+1,interiorLast=pattern.origin[1]+Math.max(...pattern.occurrences.map(atom=>atom.row))-1;
   if(anchorRow<interiorFirst||anchorRow>interiorLast)continue;
   // Native parameters are chosen by exact scalar reachability, never by
   // a pixel score or independently nudging the passage's row members.
   const darkness=shade(cell.sample,region.maximum),count=darkness>.75?6:darkness>.55?5:darkness>.4?4:3;
   const available=this.rows.filter((_,row)=>anchorRow+row-1>=interiorFirst&&anchorRow+row-1<=interiorLast),minimum=Math.min(...this.rows.flat().map(atom=>atom.x));
   // Height changes remove whole adjacent source rows between the band's
   // retained anchor and its limits. A narrower/occluded receiver cannot
   // admit a three-row passage merely because its centre is on the surface.
   const variants=[available,...(available.length===3?[available.slice(0,2),available.slice(1)]:[]),this.rows.slice(1,2)]
    .filter((rows,index,all)=>rows.length&&all.findIndex(other=>other.length===rows.length&&other[0]===rows[0])===index);
   for(const selected of variants){
   const selectedRows=selected.map(row=>row.slice(-count));
   const support={id:key,point:[cell.x,cell.y] as const,configuration:`source-uv/${site}/${count}`};
   const base:NativeOccurrence[]=selectedRows.flatMap(source=>source.map(atom=>({id:`${key}/r${atom.sourceRow}/c${atom.sourceScalar}`,text:atom.text,x:origin+atom.x-minimum,row:anchorRow+atom.row-1,
    contributions:[{source:key,kind:'region' as const,support,pixels:this.font.get(atom.text)!.points.map((_,i)=>i)}]})));
   const constraints=this.cursors(pattern,base),phase=constraints&&nativeProgramPhase(constraints,this.phases.get(key),this.whitespace);
   const atoms=phase===undefined?undefined:base.map(atom=>({...atom,x:atom.x+phase}));
   if(!atoms||!this.receiver(atoms,cell,region.maximum,surface,pattern.origin,visibility))continue;
   this.sites.set(pattern.id,site);this.phases.set(key,phase!);
   const x0=Math.min(...atoms.map(a=>a.x)),y0=Math.min(...atoms.map(a=>a.row*17)),x1=Math.max(...atoms.map(a=>a.x+this.font.get(a.text)!.advance)),y1=Math.max(...atoms.map(a=>a.row*17+17));
   const local=atoms.map(atom=>({...atom,x:atom.x-pattern.origin[0],row:atom.row-pattern.origin[1]}));
   const cut=[x0-pattern.origin[0],y0-pattern.origin[1]*17,x1-x0,y1-y0] as const;
   return {...pattern,occurrences:[...pattern.occurrences,...local],requiredSources:[...pattern.requiredSources,key],paper:pattern.paper.flatMap(paper=>subtract(paper,cut))};
   }
   }
   return pattern;
  });
  for(const owner of this.sites.keys())if(!live.has(owner))this.sites.delete(owner);
  return result;
 }
 private anchor(cell:Cell,u:number):Cell|undefined{
  const {field,rect}=cell.view,x=Math.floor((cell.x-rect.x)/rect.width*field.width),y=Math.floor((cell.y-rect.y)/rect.height*field.height);
  const right=field.sample(x+1,y),below=field.sample(x,y+1),same=(sample:FieldSample|undefined)=>sample?.source===cell.sample.source&&sample.owner===cell.sample.owner;
  if(!same(right)||!same(below))return;
  const a=right!.uv[0]-cell.sample.uv[0],b=below!.uv[0]-cell.sample.uv[0],c=right!.uv[1]-cell.sample.uv[1],d=below!.uv[1]-cell.sample.uv[1],det=a*d-b*c;
  if(Math.abs(det)<1e-12)return;
  const du=u-cell.sample.uv[0],dv=.5-cell.sample.uv[1],dx=(du*d-b*dv)/det,dy=(a*dv-c*du)/det;
  if(Math.abs(dx)>8||Math.abs(dy)>8)return;
  return {...cell,x:cell.x+dx/field.width*rect.width,y:cell.y+dy/field.height*rect.height};
 }
 private cursors(pattern:NativePattern,band:readonly NativeOccurrence[]):NativeCursorConstraint[]|undefined{
  const moving=new Set(band.map(atom=>atom.id));
  const existing=pattern.occurrences.map(atom=>({...atom,x:atom.x+pattern.origin[0],row:atom.row+pattern.origin[1]}));
  const rows=new Map<number,NativeOccurrence[]>();for(const atom of [...existing,...band]){const row=rows.get(atom.row)??[];row.push(atom);rows.set(atom.row,row);}
  const constraints:NativeCursorConstraint[]=[];
  for(const row of new Set(band.map(atom=>atom.row))){
   const atoms=rows.get(row)!.sort((a,b)=>a.x-b.x);let pen=0,prior:NativeOccurrence|undefined;
   for(const atom of atoms){
    const left=!!prior&&moving.has(prior.id),right=moving.has(atom.id);
    if(left!==right)constraints.push({gap:atom.x-pen,leading:pen===0,direction:right?1:-1});
    pen=atom.x+this.font.get(atom.text)!.advance;prior=atom;
   }
  }
  return constraints;
 }
 private receiver(atoms:readonly NativeOccurrence[],cell:Cell,maximum:number,surface:NativeSurface,origin:readonly [number,number],visibility:NativeSurfaceVisibility):boolean{
  const {field,rect}=cell.view;
  for(const atom of atoms)for(const point of this.font.get(atom.text)!.points){
   if(visibility.opening(atom.x+point[0]!+.5,atom.row*17+point[1]!+.5,surface.owner))return false;
   const source=sourceSurfacePoint([atom.x+point[0]!+.5-origin[0],(atom.row-origin[1])*17+point[1]!+.5],surface);if(!source)return false;
   const sample=field.sample((source[0]-rect.x)/rect.width*field.width,(source[1]-rect.y)/rect.height*field.height);
   if(!sample||sample.source!==cell.sample.source||sample.owner!==cell.sample.owner||sample.radiance[3]<.75||shade(sample,maximum)<.1)return false;
   if(visibility.occluder(atom.x+point[0]!+.5,atom.row*17+point[1]!+.5,surface.owner,sample.depth)!==undefined)return false;
  }
  return true;
 }
}

/** The new tone-bearing construction declares its lit paper around the WHOLE
 * admitted band. This happens in selection; publication cannot punch holes. */
function subtract(paper:NativePaper,cut:readonly [number,number,number,number]):NativePaper[]{
 const [x,y,w,h]=paper.rect,[a,b,c,d]=cut,left=Math.max(x,a),right=Math.min(x+w,a+c),top=Math.max(y,b),bottom=Math.min(y+h,b+d);
 if(left>=right||top>=bottom)return [paper];
 const rects:NativePaper['rect'][]=[[x,y,w,top-y],[x,bottom,w,y+h-bottom],[x,top,left-x,bottom-top],[right,top,x+w-right,bottom-top]];
 return rects.filter(([, ,width,height])=>width>0&&height>0).map((rect,i)=>({...paper,id:`${paper.id}/lit-${i}`,rect}));
}
