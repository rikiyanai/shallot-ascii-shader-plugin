import {partPoint,PART_CAP,PART_ROOT,PART_LIMB,PART_FORK,type MeshPartGraph} from './mesh-part-graph.ts';

/** SMALLEST-TIER PARENT CLASS (perf-impl-156, D9 + petit AA "one role per row").
 *
 * A far parent held at 2-5 page rows cannot spell the 9-row bank shoulder or the
 * 10-row bank flank. Its smallest construction is a different, connected
 * vocabulary of whole small units (README "Far-part scale consumer and
 * unresolved smallest representation"):
 *  - `round`: a compact form; the compact whole-circle / dome entries (2-4 rows,
 *    `γ⌒ヽ` over `ゝ__ノ`) at their own size.
 *  - `limb`: an elongated round tube; cap row, straight side rows (a straight
 *    source side may keep an upright, D5), return row; height grows by side
 *    rows, width by whole repeat units.
 *  - `box`: rectangular sections; lid run `＿`, straight sides, base run `￣`,
 *    or the one-glyph box (口 ロ ﾛ) at a single row.
 * This is source compilation only: it decides WHICH small construction the
 * source owns, from its own section geometry and part labels. Rows, width and
 * glyphs are chosen by the compiled small table and the far consumer per pose.
 * No camera, screen probe or glyph search occurs here. */
export type SourceSmallParentKind='round'|'limb'|'box';
export type SourceSmallParentEnd='cap'|'contact'|'open';
export interface SourceSmallParent {
 readonly kind:SourceSmallParentKind;
 /** Unit principal axis in source space; `along`/`across` are extents. */
 readonly axis:readonly [number,number,number];
 readonly along:number;readonly across:number;
 /** End construction at the +axis and -axis ends: a free closed end ('cap',
  * including a geometric PART_ROOT end), a KNOWN support contact ('contact',
  * only from an authored support), or 'open'. */
 readonly positiveEnd:SourceSmallParentEnd;readonly negativeEnd:SourceSmallParentEnd;
 /** Measured evidence (counts), kept for review; not consumed by playback. */
 readonly sections:number;readonly rectangular:number;
 /** Required source relationships a generic small outline cannot spell
  * (README "Small-parent integration ownership"): fork bands of the part
  * graph, plane-region openings and source contacts. */
 readonly relations:{readonly forks:number;readonly openings:number;readonly contacts:number;readonly seams:number;readonly components:number;readonly defects:number};
 /** True only when the source has none of those relationships, so its whole
  * form may be replaced by one small outline. False means runtime parent
  * eligibility must keep the incumbent/children; the class never grants it. */
 readonly replaceable:boolean;
}
/** Relationship counts the caller already owns (analysis / complex). */
export interface SourceSmallParentRelations {
 readonly openings?:number;
 /** REQUIRED contacts only (see compileSourceRelationships), never raw groups. */
 readonly contacts?:number;
 /** Proven ordinary seams of one closed welded shell (informational). */
 readonly seams?:number;
 /** Welded face components; more than one is a multi-part obligation. */
 readonly components?:number;
 /** Nonmanifold / inconsistent-winding / open-boundary source issues. */
 readonly defects?:number;
 /** A genuinely KNOWN external support (authored by the caller, e.g. a
  * placement that declares the source stands on a surface). 'lowest' marks
  * the source's lowest rest end as a support contact. Absent = no known
  * support: a geometric lowest vertex alone never requires a ground line. */
 readonly support?:'lowest';
}
/** The minimal MeshSourceComplex view this predicate reads. */
export interface SourceRelationshipComplex {
 readonly halfedges:readonly {readonly face:number;readonly twin?:number}[];
 readonly faceComponents:ArrayLike<number>;
 readonly contacts:readonly {readonly members:readonly {readonly edge:number;readonly forward:boolean}[];readonly kind:string}[];
 readonly issues:readonly {readonly kind:string;readonly source:number}[];
 /** Welded boundary cycles (one-face welded edges). A component with ANY
  * boundary is an open shell, even when its boundary loop is complete
  * (`compileMeshSource` raises `open-boundary` only for an unclosed walk). */
 readonly boundaries:readonly {readonly component:number}[];
}
/** RELATIONSHIP PRODUCER (perf-impl-159; README "Relationship producer
 * correction for Claude"). `SourceContactCandidate` is not synonymous with an
 * external depicted contact: on a separately indexed or flat-shaded mesh every
 * ordinary seam is an index-boundary pair at one position, so its
 * coincident-source-boundaries / published-source-incidence group carries the
 * welded seam incidence that d607bdf515 already reads for edge classes.
 *
 * A group is a PROVEN closed-shell seam only when ALL hold: exactly two
 * members; their halfedges are welded twins of each other (so the weld paired
 * them with consistent winding and the welded edge is manifold); both faces lie
 * in one welded component; and that component is a CLOSED shell: no welded
 * boundary cycle (an open sheet's complete boundary loop counts) and no
 * nonmanifold, inconsistent-winding or open-boundary issue. Every other group stays a
 * REQUIRED contact (three or more members, unpaired, cross-component, or on an
 * open/defective shell). Multi-component sources and defects are reported
 * separately; nothing is zeroed globally and the relation contract is
 * unchanged. Worker and baked compilation call this same function. */
export function compileSourceRelationships(complex:SourceRelationshipComplex):{readonly contacts:number;readonly seams:number;readonly components:number;readonly defects:number}{
 const required=sourceContactRequirements(complex);
 let contacts=0;for(const value of required)contacts+=value;
 const components=new Set<number>();for(let face=0;face<complex.faceComponents.length;face++)components.add(complex.faceComponents[face]!);
 return {contacts,seams:required.length-contacts,components:components.size,defects:complex.issues.length};
}
/** The ONE per-candidate classification (perf-impl-161): entry i is 1 when
 * `complex.contacts[i]` is a REQUIRED contact and 0 when it is a proven
 * ordinary seam of one closed welded shell (rule above). Compiled once per
 * source revision; `compileSourceRelationships` (counts) and the whole-
 * template producer (which candidates to serialize) both read it, so the two
 * cannot disagree and no consumer re-derives topology per edge or frame. */
export function sourceContactRequirements(complex:SourceRelationshipComplex):Uint8Array{
 const componentOf=(halfedge:number)=>{const face=complex.halfedges[halfedge]?.face;return face===undefined?undefined:complex.faceComponents[face];};
 const defective=new Set<number>();
 for(const issue of complex.issues){const component=componentOf(issue.source);if(component!==undefined)defective.add(component);}
 // An open shell (any welded boundary) is not a closed shell.
 for(const boundary of complex.boundaries)defective.add(boundary.component);
 const required=new Uint8Array(complex.contacts.length);
 for(const [index,contact]of complex.contacts.entries()){
  const [a,b]=contact.members;
  const seam=contact.members.length===2&&a!==undefined&&b!==undefined&&a.forward!==b.forward&&
   complex.halfedges[a.edge]?.twin===b.edge&&complex.halfedges[b.edge]?.twin===a.edge&&
   componentOf(a.edge)!==undefined&&componentOf(a.edge)===componentOf(b.edge)&&!defective.has(componentOf(a.edge)!);
  required[index]=seam?0:1;
 }
 return required;
}
/** Elongation at which a round form becomes a limb: the part graph's own
 * PART_LIMB_ELONGATION is 2 (a tube twice as long as wide). */
const LIMB_ASPECT=2;
const dot3=(a:readonly number[],b:readonly number[])=>a[0]!*b[0]!+a[1]!*b[1]!+a[2]!*b[2]!;
const cross3=(a:readonly number[],b:readonly number[])=>[a[1]!*b[2]!-a[2]!*b[1]!,a[2]!*b[0]!-a[0]!*b[2]!,a[0]!*b[1]!-a[1]!*b[0]!];
/** A section is rectangular in its own plane when its quantized segment
 * directions use only the four cardinal codes of a chart aligned to its
 * longest edge (rail/frame/box cross-sections; br52 sections 1 and 34). */
function rectangularSection(points:readonly (readonly [number,number,number])[],axis:readonly number[]):boolean{
 const n=points.length;if(n<4)return false;
 let longest=-1,u:number[]|undefined;
 for(let i=0;i<n;i++){const a=points[i]!,b=points[(i+1)%n]!,t=[b[0]-a[0],b[1]-a[1],b[2]-a[2]],k=dot3(t,axis),p=[t[0]!-k*axis[0]!,t[1]!-k*axis[1]!,t[2]!-k*axis[2]!],l=Math.hypot(...p);if(l>longest){longest=l;u=p.map(x=>x/l);}}
 if(!u||!(longest>0))return false;
 const v=cross3(axis,u);let total=0,offAxis=0;
 for(let i=0;i<n;i++){
  const a=points[i]!,b=points[(i+1)%n]!,t=[b[0]-a[0],b[1]-a[1],b[2]-a[2]],x=dot3(t,u),y=dot3(t,v),l=Math.hypot(x,y);if(!(l>0))continue;
  total+=l;
  // Length not within 22.5 deg of a chart axis is a rounded or slanted turn.
  if(Math.min(Math.abs(x),Math.abs(y))>l*Math.sin(Math.PI/8))offAxis+=l;
 }
 return total>0&&offAxis<=total*.05;
}
/** Compile the smallest-tier parent class of one source part graph. Returns
 * undefined only when the source has no part graph / no closed section; the
 * incumbent then keeps its producer (never blank, D1). */
export function compileSourceSmallParent(graph:MeshPartGraph|undefined,positions:ArrayLike<number>,relationInput:SourceSmallParentRelations={}):SourceSmallParent|undefined{
 if(!graph)return;
 const axisLength=Math.hypot(...graph.axis);if(!(axisLength>0))return;
 const axis=graph.axis.map(value=>value/axisLength) as unknown as [number,number,number];
 const closed=graph.sections.filter(section=>section.closed&&section.samples.length>=3);if(!closed.length)return;
 let lo=Infinity,hi=-Infinity,across=0,rectangular=0;
 const ends={positive:'open' as SourceSmallParentEnd,negative:'open' as SourceSmallParentEnd};
 let topLevel=-Infinity,bottomLevel=Infinity,topLabels=0,bottomLabels=0;
 for(const section of closed){
  const points=section.samples.map(sample=>partPoint({positions:positions as Float32Array},sample));
  const level=dot3(section.center,axis);
  for(const point of points){
   const along=dot3(point,axis);lo=Math.min(lo,along);hi=Math.max(hi,along);
   const radial=[point[0]-section.center[0],point[1]-section.center[1],point[2]-section.center[2]],k=dot3(radial,axis);
   across=Math.max(across,2*Math.hypot(radial[0]!-k*axis[0]!,radial[1]!-k*axis[1]!,radial[2]!-k*axis[2]!));
  }
  if(rectangularSection(points,axis))rectangular++;
  if(level>topLevel){topLevel=level;topLabels=section.labels??0;}
  if(level<bottomLevel){bottomLevel=level;bottomLabels=section.labels??0;}
 }
 // Sections are level cuts, so include the source vertices beyond the end
 // levels (a cap dome above its last section) in the along-axis extent.
 for(let v=0;v<positions.length/3;v++){const along=positions[v*3]!*axis[0]+positions[v*3+1]!*axis[1]+positions[v*3+2]!*axis[2];lo=Math.min(lo,along);hi=Math.max(hi,along);}
 // perf-impl-162 (README "Next source-end correction"): PART_ROOT is the free
 // chain end reaching the source's LOWEST source-Y plane (mesh-part-graph.ts
 // 8-10, 244-265), computed with no world, placement or other surface. It is
 // geometry, not proof of an external support. A ROOT end, like a CAP end, is
 // therefore a free CLOSED end ('cap'). The ROOT label itself is untouched for
 // its base/chain meaning; 'contact' is spelled only from a genuinely KNOWN
 // support (`relationInput.support`, authored by the caller), never invented
 // by a world query here. An end with neither bit stays 'open'.
 const end=(labels:number):SourceSmallParentEnd=>(labels&(PART_ROOT|PART_CAP))!==0?'cap':'open';
 ends.positive=end(topLabels);ends.negative=end(bottomLabels);
 // A known support names the end it touches: the lowest rest end of the source.
 if(relationInput.support==='lowest'){
  const lowestIsNegative=axis[1]>=0;
  if(lowestIsNegative)ends.negative='contact';else ends.positive='contact';
 }
 const along=hi-lo;if(!(along>0)||!(across>0))return;
 const kind:SourceSmallParentKind=rectangular*2>closed.length?'box':
  (along>=LIMB_ASPECT*across||closed.some(section=>((section.labels??0)&PART_LIMB)!==0))?'limb':'round';
 const relations={forks:graph.bands.filter(band=>((band.labels??0)&PART_FORK)!==0).length,openings:relationInput.openings??0,contacts:relationInput.contacts??0,
  seams:relationInput.seams??0,components:relationInput.components??1,defects:relationInput.defects??0};
 // Proven closed-shell seams are represented by the one whole outline; every
 // other relationship (fork, opening, required contact, extra component,
 // source defect) is an obligation the generic small outline cannot keep.
 const replaceable=relations.forks===0&&relations.openings===0&&relations.contacts===0&&relations.components===1&&relations.defects===0;
 return {kind,axis,along,across,positiveEnd:ends.positive,negativeEnd:ends.negative,sections:closed.length,rectangular,relations,replaceable};
}
