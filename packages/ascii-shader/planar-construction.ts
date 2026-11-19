import type {CompositionSubject,ComposedPlane} from './composition.ts';
import {fieldOwner} from './composition.ts';
import {relationPoint,relationRange,type PosedRelation} from './posed-graph.ts';
import type {NativeGlyph,NativeOccurrence,NativePaper,NativeContact} from './native-page.ts';
import type {NativePattern,NativeSurface} from './native-layout.ts';
import {nativeSurfacePoint,sourceSurfacePoint} from './native-surface.ts';
import {nativeSpacePlan,type NativeWhitespacePolicy} from './text/native-whitespace.ts';

type Point=readonly [number,number];
interface Arm {edge:PosedRelation;from:string}
interface Corner {points:Map<string,Point>;center:string;left:string;upper:string;foot:string;rearLeft:string;rearFoot:string;leftFoot:string;arms:Record<string,Arm>}
export type PlanarResult={ok:true;pattern:NativePattern;configuration:string}|{ok:false;reason:string};

/** Independently authored P1, not a transferred morphology entry. Complete
 * B1–B4 tonemapping passages establish the cap/face/post/return relationship;
 * their scene anchors, arbitrary Unicode spacers and frame cuts are NOT used.
 * One coupled set of integer axes grows the complete native neighborhood.
 */
export class PlanarConstruction {
 private readonly font:Map<string,NativeGlyph>;
 constructor(glyphs:readonly NativeGlyph[],private readonly whitespace:NativeWhitespacePolicy){this.font=new Map(glyphs.map(g=>[g.text,g]));}
 private corner(s:CompositionSubject):Corner|undefined{
  if(s.relations.some(r=>r.kind==='smooth-contour'))return;
  const adjacent=new Map<string,string[]>(),points=new Map<string,Point>();
  for(const edge of s.relations)for(const [i,port]of edge.ports.entries()){const p=relationPoint(edge,i);if(!p.every(Number.isFinite))return;points.set(port,[p[0],p[1]]);const list=adjacent.get(port)??[];list.push(edge.ports[1-i]!);adjacent.set(port,list);}
  const centers=[...adjacent].filter(([,p])=>p.length===3).map(([p])=>p).filter(p=>[...adjacent].every(([q,n])=>p===q||n.filter(r=>r!==p).length===2));
  if(centers.length!==1)return;const center=centers[0]!,ordered=[...adjacent.get(center)!].sort((a,b)=>points.get(a)![1]-points.get(b)![1]),[upper,left,foot]=ordered as [string,string,string],c=points.get(center)!;
  if(!(points.get(left)![0]<c[0]&&points.get(upper)![0]>c[0]&&points.get(upper)![1]<c[1]&&points.get(foot)![1]>c[1]))return;
  const between=(a:string,b:string)=>adjacent.get(a)!.filter(p=>p!==center&&adjacent.get(b)!.includes(p));
  const a=between(upper,left),b=between(upper,foot),d=between(left,foot);if(a.length!==1||b.length!==1||d.length!==1||points.size!==7||s.relations.length!==9)return;
  const [rearLeft,rearFoot,leftFoot]=[a[0]!,b[0]!,d[0]!],pairs:Record<string,readonly [string,string]>={back:[rearLeft,upper],leftSlope:[rearLeft,left],front:[left,center],topSlope:[center,upper],backPost:[upper,rearFoot],leftPost:[left,leftFoot],frontPost:[center,foot],sideReturn:[foot,rearFoot],base:[leftFoot,foot]};
  const arms:Record<string,Arm>={};for(const [role,[from,to]]of Object.entries(pairs)){const matches=s.relations.filter(e=>e.ports.includes(from)&&e.ports.includes(to));if(matches.length!==1)return;arms[role]={edge:matches[0]!,from};}
  return {points,center,left,upper,foot,rearLeft,rearFoot,leftFoot,arms};
 }
 construct(s:CompositionSubject,planes:readonly ComposedPlane[]):PlanarResult{
  const fail=(reason:string):PlanarResult=>({ok:false,reason}),admission=s.planar;
  if(!admission||admission.authority!=='independent-P1'||admission.purpose!=='convex-volume'||admission.unresolved)return fail(admission?.unresolved??'P1 requires an admitted convex volume, not a sheet or aperture');
  if(admission.openings.length||s.openings.length)return fail('P1 cannot replace an opening with a filled face');
  const f=this.corner(s);if(!f)return fail('P1 needs its declared three-face orientation and incidence');
  const facePorts=[[f.rearLeft,f.upper,f.center,f.left],[f.left,f.center,f.foot,f.leftFoot],[f.center,f.upper,f.rearFoot,f.foot]];
  for(const face of facePorts){
   let common: Set<number>|undefined;
   const xyz:number[][]=[];
   for(let i=0;i<face.length;i++){const from=face[i]!,to=face[(i+1)%face.length]!,edge=s.relations.find(r=>r.ports.includes(from)&&r.ports.includes(to));if(!edge)return fail('P1 lost an incident source face edge');
    const regions=new Set(edge.regions.flatMap(region=>{const triangle=Number(region?.split('/face-').at(-1));return Number.isInteger(triangle)?admission.faces.filter(p=>p.regions.includes(triangle)).map(p=>p.id):[];}));common=common?new Set([...common].filter(id=>regions.has(id))):regions;
    xyz.push([...relationPoint(edge,edge.ports[0]===from?0:1)]);
   }
   if(!common?.size)return fail('P1 incident source regions are not one coplanar face');
   const [a,b,c,q]=xyz as [number[],number[],number[],number[]],ux=b[0]!-a[0]!,uy=b[1]!-a[1]!,vx=c[0]!-a[0]!,vy=c[1]!-a[1]!,det=ux*vy-vx*uy;
   if(Math.abs(det)<1e-8)return fail('P1 source face is edge-on');
   const u=((q[0]!-a[0]!)*vy-vx*(q[1]!-a[1]!))/det,v=(ux*(q[1]!-a[1]!)-(q[0]!-a[0]!)*uy)/det;
   if(Math.abs(q[2]!-a[2]!-u*(b[2]!-a[2]!)-v*(c[2]!-a[2]!))>1e-5)return fail('P1 posed face is no longer planar');
  }
  const p=f.points,l=p.get(f.left)!,c=p.get(f.center)!,u=p.get(f.upper)!,foot=p.get(f.foot)!;
  const W=c[0]-l[0],D=c[1]-u[1],H=foot[1]-c[1];
  // One semilinear family at every size: front20+16n, back40+16n,
  // return32+16n. All three gain ONE intact16px unit together. There is no
  // compact/regular phase reset or independent rail-width decision.
  const n=Math.max(0,Math.round((W-34)/16));
  const d=Math.max(1,Math.round(D/17)),h=Math.max(1,Math.round(H/17)),w=34+16*n;
  const x=Math.round(l[0]-2),row=Math.round((p.get(f.rearLeft)![1]-15)/17),S=w+2,extra=16*(d-1),baseRow=d+h,id=`${s.id}/P1`;
  const vertices=new Map<string,Point>([[f.rearLeft,[12+extra,15]],[f.upper,[w+16+extra,15]],[f.left,[2,17*d+12]],[f.center,[S,17*d+14]],[f.rearFoot,[w+19+extra,17*(h+1)]],[f.leftFoot,[2,17*baseRow+15]],[f.foot,[w+4,17*baseRow+15]]]);
  // Native16x17 quantization is an admitted context, not a fit objective.
  // Opposite hand, steep/tapered rails and other contexts need their own
  // complete authored alternative. No residual score can override this gate.
  if([...vertices].some(([port,q])=>Math.hypot(x+q[0]-p.get(port)![0],row*17+q[1]-p.get(port)![1])>17))return fail('P1 source anchors exceed the declared whole-axis native band');
  const occurrences:NativeOccurrence[]=[],contacts:NativeContact[]=[],paper:NativePaper[]=[];
  const add=(text:string,at:number,r:number,key:string,role:string,range:readonly [number,number],select?:(p:readonly number[])=>boolean)=>{
   const glyph=this.font.get(text)!;const pixels=glyph.points.flatMap((p,i)=>!select||select(p)?[i]:[]),arm=f.arms[role]!,contribution={source:arm.edge.id,kind:'path' as const,pixels,parameter:relationRange(arm.edge,arm.from,range)},old=occurrences.find(a=>a.id===`${id}/${key}`);
   if(old)occurrences[occurrences.indexOf(old)]={...old,contributions:[...old.contributions,contribution]};else occurrences.push({id:`${id}/${key}`,text,x:at,row:r,contributions:[contribution]});
  };
  const rail=(word:string,at:number,r:number,role:string,offset:number,total:number)=>{let pen=at;for(const [i,ch]of [...word].entries()){const glyph=this.font.get(ch)!,xs=glyph.points.map(p=>p[0]!);add(ch,pen,r,`${role}/unit-${i}`,role,[(offset+pen-at+Math.min(...xs))/total,(offset+pen-at+Math.max(...xs))/total]);pen+=glyph.advance;}};
  rail('_'.repeat(8)+'＿'.repeat(n),12+extra,0,'back',0,w+4);
  for(let r=1;r<d;r++)add('／',16*(d-r),r,`depth/left-${r}`,'leftSlope',[(r-1)/d,r/d]);
  add('∠',0,d,'corner/cap','leftSlope',[(d-1)/d,1],q=>q[1]!==12||q[0]===2);
  add('∠',0,d,'corner/cap','front',[0,14/w],q=>q[1]===12);
  rail('_'.repeat(4)+'＿'.repeat(n),16,d,'front',14,w);
  for(let r=1;r<=d;r++)add('／',S+16*(d-r),r,`depth/right-${r}`,'topSlope',[1-r/d,1-(r-1)/d]);
  for(let r=1;r<=h;r++)add('|',S+16+extra,r,`post/rear-${r}`,'backPost',[(r-1)/h,r/h]);
  for(let k=1;k<=h;k++){
   add('|',1,d+k,k===h?'corner/left-foot':`post/left-${k}`,'leftPost',[(k-1)/h,k/h]);
   add('|',S+1,d+k,k===h?'corner/front-foot':`post/front-${k}`,'frontPost',[(k-1)/h,k/h]);
  }
  add('|',1,baseRow,'corner/left-foot','base',[0,3/(w+2)],q=>q[1]===15);
  rail('＿'.repeat(n+2),5,baseRow,'base',3,w+2);
  add('|',S+1,baseRow,'corner/front-foot','base',[1-3/(w+2),1],q=>q[1]===15);
  for(let k=1;k<=d;k++)add('／',S+5+16*(d-k),h+k,`return/side-${k}`,'sideReturn',[1-k/d,1-(k-1)/d]);
  const surfaces:NativeSurface[]=facePorts.map((face,i)=>({id:`${id}/face-${i}`,owner:s.owner,outline:face.map(port=>vertices.get(port)!),charts:[[0,1,2],[0,2,3]].map(indices=>({native:indices.map(k=>vertices.get(face[k]!)!) as [Point,Point,Point],source:indices.map(k=>p.get(face[k]!)!) as [Point,Point,Point]}))}));
  // Lit top paper is an authored face-domain, not every incidental space.
  // Use inset rectangles inside that face; the fixed boundary is never cut.
  for(let r=1;r<=d;r++){const left=16*(d-r)+18,right=S-12+16*(d-r);if(right>left)paper.push({id:`${id}/lit-top-${r}`,rect:[left,17*r+1,right-left,9],sources:[f.arms.back!.edge.id,f.arms.front!.edge.id]});}
  // Only actual native interfaces become physical-contact annotations.
  // All incidents are already tied to the same complete source neighborhood.
  const ink=(a:NativeOccurrence,source:string)=>a.contributions.filter(c=>c.source===source).flatMap(c=>c.pixels.map(i=>{const q=this.font.get(a.text)!.points[i]!;return [a.x+q[0]!+.5,a.row*17+q[1]!+.5] as Point;}));
  for(const port of vertices.keys()){const incident=s.relations.filter(e=>e.ports.includes(port));for(let i=1;i<incident.length;i++){const a=incident[0]!,b=incident[i]!;let joint:{a:NativeOccurrence;b:NativeOccurrence;p:Point;q:Point;distance:number}|undefined;
   const end=(edge:PosedRelation)=>{const t=edge.ports[0]===port?0:1;return occurrences.filter(o=>o.contributions.some(c=>c.source===edge.id&&c.parameter&&c.parameter[0]<=t+1e-6&&c.parameter[1]>=t-1e-6));};
   for(const aa of end(a))for(const bb of end(b))for(const pa of ink(aa,a.id))for(const pb of ink(bb,b.id)){const distance=Math.hypot(pa[0]-pb[0],pa[1]-pb[1]);if(!joint||distance<joint.distance)joint={a:aa,b:bb,p:pa,q:pb,distance};}
   if(joint&&joint.distance<=3)contacts.push({id:`${id}/joint/${port}/${i}`,from:joint.a.id,to:joint.b.id,fromSource:a.id,toSource:b.id,fromPort:port,toPort:port,fromAnchor:joint.p,toAnchor:joint.q,relation:'independent-P1/whole-plane-corner'});
  }}
  // Tone sites are authored into the SAME native tape before admission.
  // Receiver coordinates propose sites; the complete face chart and protected
  // paper decide whether they belong, not a separate screen-grid overlay.
  for(const plane of planes.filter(p=>p.owner===s.owner))for(const site of plane.sites){let local:Point|undefined,surface:NativeSurface|undefined;
   for(const face of surfaces.slice(1)){const q=nativeSurfacePoint(site.point,face);if(q){local=q;surface=face;break;}}if(!local||!surface)continue;
   const r=Math.round((local[1]-14)/17),members=occurrences.filter(a=>a.row===r).sort((a,b)=>a.x-b.x);let previous=0,next=Infinity;
   for(const member of members){if(member.x<=local[0])previous=Math.max(previous,member.x+this.font.get(member.text)!.advance);else{next=member.x;break;}}
   const at=previous+5*Math.max(1,Math.round((local[0]-previous)/5)),glyph=this.font.get('.')!;
   if(at+glyph.advance>=next||Math.abs(at-local[0])>5||!nativeSpacePlan(at-previous,0,previous===0,false,this.whitespace)||Number.isFinite(next)&&!nativeSpacePlan(next-at-glyph.advance,0,false,false,this.whitespace))continue;
   if(occurrences.some(a=>a.x===at&&a.row===r))continue;
   if(!glyph.points.every(q=>{const a:Point=[at+q[0]!+.5,r*17+q[1]!+.5];if(paper.some(p=>a[0]>=p.rect[0]&&a[0]<p.rect[0]+p.rect[2]&&a[1]>=p.rect[1]&&a[1]<p.rect[1]+p.rect[3]))return false;const source=sourceSurfacePoint(a,surface!);if(!source)return false;const hit=plane.view.field.sample((source[0]-plane.view.rect.x)/plane.view.rect.width*plane.view.field.width,(source[1]-plane.view.rect.y)/plane.view.rect.height*plane.view.field.height);return !!hit&&fieldOwner(plane.view,hit)===s.owner;}))continue;
   occurrences.push({id:`${id}/tone/${site.id}`,text:'.',x:at,row:r,contributions:[{source:plane.id,kind:'region',pixels:glyph.points.map((_,i)=>i),support:{id:site.id,point:site.point,configuration:`P1/constant-plane/${plane.spacing}`}}]});
  }
  // The caller applies the shared source-interval visibility stage to this
  // COMPLETE phase-retained program. Never reject every face because one
  // unit meets an occluder, and never cut that unit's native ink to hide it.
  const pattern:NativePattern={id,origin:[x,row],occurrences,contacts,paper,requiredSources:s.relations.map(r=>r.id),surfaces,edgeCharts:s.relations.map(edge=>({source:edge.id,native:[vertices.get(edge.ports[0])!,vertices.get(edge.ports[1])!]}))};
  return {ok:true,pattern,configuration:`P1/${admission.revision}/${n}/${d}/${h}/${admission.supports.join(',')}`};
 }
}
