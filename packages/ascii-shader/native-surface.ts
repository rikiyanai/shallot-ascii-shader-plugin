import type {NativeSurface,NativePattern} from './native-layout.ts';
import type {DrawingFieldView} from './field-readback.ts';
import {SourceSupportIndex} from './source-support.ts';
import type {NativeGlyph} from './native-page.ts';
import {relationPoint,type PosedGraph,type PosedRelation} from './posed-graph.ts';
type Point=readonly [number,number];
function inside(point:Point,surface:NativeSurface):boolean{
 let result=false;
 for(let i=0,j=surface.outline.length-1;i<surface.outline.length;j=i++){
  const a=surface.outline[i]!,b=surface.outline[j]!;
  if((a[1]>point[1])!==(b[1]>point[1])&&point[0]<(b[0]-a[0])*(point[1]-a[1])/(b[1]-a[1])+a[0])result=!result;
 }
 return result;
}
/** Visibility and region-led passages share the same native surface chart.
 * Original camera pixels do not become a substitute surface identity. */
function map(point:Point,surface:NativeSurface,toSource:boolean):Point|undefined{
 if(toSource&&!inside(point,surface))return;
 for(const triangle of surface.charts){
  const from=toSource?triangle.native:triangle.source,to=toSource?triangle.source:triangle.native;
  const [a,b,c]=from,dx=point[0]-a[0],dy=point[1]-a[1],ux=b[0]-a[0],uy=b[1]-a[1],vx=c[0]-a[0],vy=c[1]-a[1],det=ux*vy-vx*uy;
  if(Math.abs(det)<1e-9)continue;
  const u=(dx*vy-vx*dy)/det,v=(ux*dy-dx*uy)/det;if(u< -1e-6||v< -1e-6||u+v>1+1e-6)continue;
  const [p,q,r]=to,result:Point=[p[0]+u*(q[0]-p[0])+v*(r[0]-p[0]),p[1]+u*(q[1]-p[1])+v*(r[1]-p[1])];
  if(toSource||inside(result,surface))return result;
 }
}
export const sourceSurfacePoint=(point:Point,surface:NativeSurface):Point|undefined=>map(point,surface,true);
export const nativeSurfacePoint=(point:Point,surface:NativeSurface):Point|undefined=>map(point,surface,false);

interface Body {readonly pattern:NativePattern;readonly surface:NativeSurface;readonly camera:string}
interface Cursor {id:string;owner:string;start:number;end:number;depth:number;order:number}
/** Native bodies own occlusion in the constructed drawing. A local index
 * avoids testing every object against every glyph pixel. Tone selection and
 * path visibility share this decision BEFORE the native page is committed. */
export class NativeSurfaceVisibility {
 private readonly cells=new Map<string,Body[]>();private readonly views=new Map<string,DrawingFieldView[]>();
 private readonly cursors=new Map<number,Cursor[]>();
 private readonly ownerCursors=new Map<number,Map<string,Cursor[]>>();
 private readonly occlusions=new Map<string,Set<string>>();
 private readonly font=new Map<string,NativeGlyph>();private readonly relations=new Map<string,PosedRelation>();
 private readonly paths=new Map<string,Map<string,PosedRelation[]>>();private bounds:readonly [number,number,number,number]|undefined;
 private readonly support=new SourceSupportIndex();
 private readonly sourceBound:boolean;
 constructor(patterns:readonly NativePattern[],views:readonly DrawingFieldView[],glyphs:readonly NativeGlyph[]=[],graphs:readonly PosedGraph[]=[]){
  this.sourceBound=graphs.length>0;
  for(const glyph of glyphs)this.font.set(glyph.text,glyph);
  for(const graph of graphs)for(const relation of graph.relations)this.relations.set(relation.id,relation);
  if(!views.length)return;
  for(const view of views){const list=this.views.get(view.camera)??[];list.push(view);this.views.set(view.camera,list);}
  const minX=Math.min(...views.map(v=>v.rect.x)),minY=Math.min(...views.map(v=>v.rect.y));
  const maxX=Math.max(...views.map(v=>v.rect.x+v.rect.width)),maxY=Math.max(...views.map(v=>v.rect.y+v.rect.height));
  this.bounds=[minX,minY,maxX,maxY];const relations=this.relations;
  for(const relation of relations.values())for(const interval of relation.intervals){
   if(interval.cause!=='occluded'&&!interval.foreground)continue;
   const point=relationPoint(relation,(interval.lo+interval.hi)/2);
   for(const view of this.views.get(relation.owner.split('/')[0]!)??[]){
    const sample=view.field.sample((point[0]-view.rect.x)/view.rect.width*view.field.width,(point[1]-view.rect.y)/view.rect.height*view.field.height);
    if(!sample||sample.depth<=point[2]+1e-7)continue;
    const reference=sample.kind==='entity'?view.owners.get(sample.owner):sample.owner,foreground=`${view.camera}/${sample.source}/${sample.kind}/${reference}`;
    if(foreground===relation.owner)continue;
    const owners=this.occlusions.get(relation.id)??new Set<string>();owners.add(foreground);this.occlusions.set(relation.id,owners);
   }
  }
  this.append(patterns);
 }
 /** Add only a committed native neighborhood. Source visibility is decoded
  * once for the scene, not rebuilt for every candidate source part. */
 append(patterns:readonly NativePattern[]):void{
  if(!this.bounds)return;
  const [minX,minY,maxX,maxY]=this.bounds,font=this.font,relations=this.relations,paths=this.paths;
  for(const pattern of patterns)for(const path of pattern.paths??[]){const owners=new Map<string,PosedRelation[]>();for(const id of path.relations){const relation=relations.get(id);if(!relation)continue;const list=owners.get(relation.owner)??[];list.push(relation);owners.set(relation.owner,list);}paths.set(path.id,owners);}
  for(const pattern of patterns)for(const surface of pattern.surfaces??[]){
   const x0=Math.floor(Math.max(minX,pattern.origin[0]+Math.min(...surface.outline.map(p=>p[0])))/32),x1=Math.floor(Math.min(maxX,pattern.origin[0]+Math.max(...surface.outline.map(p=>p[0])))/32);
   const y0=Math.floor(Math.max(minY,pattern.origin[1]*17+Math.min(...surface.outline.map(p=>p[1])))/34),y1=Math.floor(Math.min(maxY,pattern.origin[1]*17+Math.max(...surface.outline.map(p=>p[1])))/34);
   const body:Body={pattern,surface,camera:surface.owner.split('/')[0]!};
   for(let y=y0;y<=y1;y++)for(let x=x0;x<=x1;x++){const key=`${x}/${y}`,cell=this.cells.get(key)??[];cell.push(body);this.cells.set(key,cell);}
   const center:Point=[surface.outline.reduce((sum,p)=>sum+p[0],0)/surface.outline.length,surface.outline.reduce((sum,p)=>sum+p[1],0)/surface.outline.length];
   for(const atom of pattern.occurrences){
    const glyph=font.get(atom.text);if(!glyph?.points.length)continue;
    const ink:Point=[atom.x+glyph.points.reduce((sum,p)=>sum+p[0]!,0)/glyph.points.length,atom.row*17+glyph.points.reduce((sum,p)=>sum+p[1]!,0)/glyph.points.length];
    // A contour scalar owns its paper too. Sample one pixel toward its OWN
    // constructed body to evaluate the source receiver, never a random
    // nearby screen pixel. The complete scalar span stays indivisible.
    const length=Math.hypot(center[0]-ink[0],center[1]-ink[1]),amount=length?Math.min(1,length)/length:0;
    const at=sourceSurfacePoint([ink[0]+(center[0]-ink[0])*amount,ink[1]+(center[1]-ink[1])*amount],surface);
    const depths:number[]=[];
    for(const contribution of atom.contributions){
     const exact=relations.get(contribution.source);
     if(exact?.owner===surface.owner&&contribution.parameter){
      const [lo,hi]=contribution.parameter,parts=exact.intervals.filter(part=>part.hi>=lo-1e-6&&part.lo<=hi+1e-6);
      if(parts.length&&parts.every(part=>part.cause==='visible'))depths.push(relationPoint(exact,(lo+hi)/2)[2]);
     }else if(at&&paths.has(contribution.source)){
      const nearest=this.support.nearest(at[0],at[1],paths.get(contribution.source)!.get(surface.owner)??[]);
      if(nearest?.relation.intervals.some(part=>part.cause==='visible'&&nearest.parameter>=part.lo&&nearest.parameter<=part.hi))depths.push(relationPoint(nearest.relation,nearest.parameter)[2]);
     }
    }
    if(depths.length){
     this.addCursor(pattern.origin[1]+atom.row,{id:pattern.id,owner:surface.owner,start:pattern.origin[0]+atom.x,end:pattern.origin[0]+atom.x+glyph.advance,depth:Math.max(...depths)});continue;
    }
    if(!at)continue;
    for(const view of this.views.get(body.camera)??[]){
     const sample=view.field.sample((at[0]-view.rect.x)/view.rect.width*view.field.width,(at[1]-view.rect.y)/view.rect.height*view.field.height);
     if(!sample||sample.radiance[3]<=.01)continue;
     const reference=sample.kind==='entity'?view.owners.get(sample.owner):sample.owner;
     if(`${view.camera}/${sample.source}/${sample.kind}/${reference}`!==surface.owner)continue;
     this.addCursor(pattern.origin[1]+atom.row,{id:pattern.id,owner:surface.owner,start:pattern.origin[0]+atom.x,end:pattern.origin[0]+atom.x+glyph.advance,depth:sample.depth});break;
    }
   }
  }
 }
 private addCursor(row:number,span:Omit<Cursor,'order'>):void{
  const all=this.cursors.get(row)??[],cursor={...span,order:all.length};all.push(cursor);this.cursors.set(row,all);
  const owners=this.ownerCursors.get(row)??new Map<string,Cursor[]>(),members=owners.get(span.owner)??[];
  members.push(cursor);owners.set(span.owner,members);this.ownerCursors.set(row,owners);
 }
 cursorOccluder(start:number,end:number,row:number,owner:string,depth:number,relation?:string):string|undefined{
  const camera=owner.split('/')[0];
  if(!this.sourceBound)return this.cursors.get(row)?.find(cursor=>cursor.owner!==owner&&cursor.owner.split('/')[0]===camera&&cursor.depth>depth+1e-7&&cursor.start<end&&cursor.end>start)?.id;
  // The original source interval already names the ONLY possible foreground
  // owners. Do not scan thousands of unrelated page cursors for every pixel
  // of an intact glyph. No glyph choice or source visibility changes here.
  const foreground=relation===undefined?undefined:this.occlusions.get(relation),owners=this.ownerCursors.get(row);
  if(!foreground?.size||!owners)return;
  let first:Cursor|undefined;
  for(const candidate of foreground){
   if(candidate===owner||candidate.split('/')[0]!==camera)continue;
   const cursor=owners.get(candidate)?.find(cursor=>cursor.depth>depth+1e-7&&cursor.start<end&&cursor.end>start);
   if(cursor&&(!first||cursor.order<first.order))first=cursor;
  }
  return first?.id;
 }
 /** An opening removes only its own source surface. Geometry visible through
  * it still owns its drawing; the aperture is not unconditional blank paper. */
 opening(x:number,y:number,owner:string):boolean{
  return (this.cells.get(`${Math.floor(x/32)}/${Math.floor(y/34)}`)??[]).some(({pattern,surface})=>
   surface.owner===owner&&surface.kind==='opening'&&inside([x-pattern.origin[0],y-pattern.origin[1]*17],surface));
 }
 occluder(x:number,y:number,owner:string,depth:number,relation?:string):string|undefined{
  for(const {pattern,surface,camera}of this.cells.get(`${Math.floor(x/32)}/${Math.floor(y/34)}`)??[]){
   if(surface.kind==='opening')continue;
   if(camera!==owner.split('/')[0])continue;
   if(surface.owner===owner)continue;
   if(this.opening(x,y,surface.owner))continue;
   if(this.sourceBound&&relation!==undefined&&!this.occlusions.get(relation)?.has(surface.owner))continue;
   const at=sourceSurfacePoint([x-pattern.origin[0],y-pattern.origin[1]*17],surface);if(!at)continue;
   for(const view of this.views.get(camera)??[]){
    const sample=view.field.sample((at[0]-view.rect.x)/view.rect.width*view.field.width,(at[1]-view.rect.y)/view.rect.height*view.field.height);
    if(!sample||sample.radiance[3]<.99||sample.depth<=depth+1e-7)continue;
    const reference=sample.kind==='entity'?view.owners.get(sample.owner):sample.owner;
    if(`${view.camera}/${sample.source}/${sample.kind}/${reference}`===surface.owner)return pattern.id;
   }
  }
 }
}
