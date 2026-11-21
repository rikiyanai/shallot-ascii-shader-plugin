import {DeviceComputePrograms,type ComputePipelinePlan,type ComputePipelineProgress} from './gpu-pipeline-preparation.ts';
import {isolateComputeEntry} from './gpu-compute-program.ts';
import type {SurfaceField} from './surface-fields.ts';
import {formDepths,FAR_STRIP_BELOW_ROWS} from './form-depths.ts';
import type {DrawingFieldView} from './field-readback.ts';
import {inverseMatrix,TONE_REGION_EDGES,type ToneRegions} from './composition.ts';
import {type World,MeshInstance,GlobalTransform} from '@dylanebert/shallot';
import {composeGlobalTransform} from '@dylanebert/shallot/transform';
import type {NativeGlyph,NativeNeighborhood,NativeOccurrence,NativePaper} from './native-page.ts';
import nativeFont from './native-glyphs.json' with {type:'json'};
import {compileTonePassages,TONE_PASSAGES,TONE_PREFERRED,TEXTURE_EDGE,TEXTURE_EDGE_MEASURE} from './compiled-tone-passages.ts';
// The line producer's distance tiers: tone thins on the same measure.
import {DISTANCE_TIER_WGSL,MATERIAL_INK_WGSL,materialInkTexels} from './gpu-articulation.ts';
/** Same rule as the page's protected-paper check. */
function nativePaperContains(paper:NativePaper,x:number,y:number):boolean{
 const [left,top,width,height]=paper.rect,u=x-left,v=y-top,visibleAt=(paper as {visibleAt?:(x:number,y:number)=>boolean}).visibleAt;
 return u>=0&&u<width&&v>=0&&v<height&&(!visibleAt||visibleAt(u,v));
}
import {nativeSpaceCodes,type NativeWhitespacePolicy} from './text/native-whitespace.ts';
import {sourceToneStyleWord} from './source-material.ts';


/** D6 region tone from the D10 light bands, played back on the GPU.
 *
 * THE NON-SEARCH TONE (agreed design, GOAL-EXECUTOR "THE AGREED NON-SEARCH
 * DESIGN"; ledger 2026-11-11 "texture patches must not flip"). A displayed
 * frame holds no trial, fit or search. Every tone word is ONE LOOKUP in a
 * compiled table (the word table below, WGSL "THE WORD TABLE"), keyed only by
 * quantities attached to the surface:
 *   the unit's construction class (the source's compiled class: planar
 *     volume, round, facet-built form, foliage crown, terrain; composition.ts,
 *     encoded once per source in the styles table),
 *   its level, from its own facet's n·L and cast visibility in fixed bands
 *     (lightBand) and the air before it (transmittance), fixed thresholds,
 *   its node on the surface lattice (latticeAt / latticeKey), which carries
 *     the row stagger and the fixed erased members of the sparse levels.
 * One GPU thread per page pixel of a native row: the pixel that holds a
 * lattice node (the node's page X rounds to it) places that node's unit;
 * every other thread returns. A unit is admitted by one fixed clearance test
 * at its one pen (unitClear: its ends on its own surface, no drawn line
 * within a half space), never by trying other pens. No region table, no
 * ranking of this frame's visible regions, no screen-side light test and no
 * per-row crown mark decides a word. The only output is a compact member
 * list; compiled playback consumes it on the GPU (gpu-articulation seedTone).
 * The legacy entry (`main`, CPU RegionTone) is unchanged under D2.
 * Display and export retain the same native page. */
/** The legacy allocation remains unchanged under D2. Compiled playback uses
 * the actual cell bound instead; this constant is not its member quota. */
export const TONE_ROW=17,TONE_CELL=11,TONE_SPACING=34,TONE_CLEAR=11,TONE_CAPACITY=16384;
/** Slots of the legacy receiver-plane table. */
export const TONE_REGION_SLOTS=16384;
/** Band level -> glyph; the same ladder as the CPU RegionTone (D6). */
export const TONE_LADDER=['',':',':','i'] as const;
// Compile the admitted native units' real ink once (legacy toneFits).
const toneUnits=[':','i'].map(text=>{
 const glyph=nativeFont.glyphs.find(glyph=>glyph.text===text);
 if(!glyph)throw Error(`Missing native tone unit ${text}`);
 return glyph;
});
const toneInk=toneUnits.flatMap(glyph=>glyph.points);
const passages=compileTonePassages(nativeFont.glyphs);
// The wall field's retained words, by their text in the bank (all hand-target
// rows of fog/rendered/000.txt): the two dense row patterns, the spaced
// fringe and the spaced tail.
const wallWord=(text:string):number=>{const index=TONE_PASSAGES.findIndex(passage=>passage.context==='wall'&&passage.text===text);if(index<0)throw Error(`Missing wall tone passage ${text}`);return index;};
const DENSE_A=wallWord(':;:;:'),DENSE_B=wallWord(':;:;:;'),FRINGE=wallWord('. : .'),TAIL=wallWord('. .');
// The ground's retained words (hand-target ground passages): the even field
// member `:`, a slope's dense stacked band `:::` and its staggered thinning

// medium-far colon grid; the blue side-plane band), the sea's dash member.
const groundWord=(text:string):number=>{const index=TONE_PASSAGES.findIndex(passage=>passage.context==='ground'&&passage.text===text);if(index<0)throw Error(`Missing ground tone passage ${text}`);return index;};
const GROUND_FIELD=groundWord(':'),SLOPE_DENSE=groundWord(':::'),SLOPE_SPACED=groundWord('.:.:.:'),WATER_FIELD=groundWord('-');
// tone2: the shadow run's alternate-row unit (church aa[126] floor band, bank ground ':.:.:.').
const SHADE_B=groundWord(':.:.:.');
// MATERIAL BANDS (compiled-tone-passages.ts MATERIAL_PASSAGES, each a verbatim
// AAHub row): a terrain cliff takes ROCK, a terrain slope whose own material
// reads as turf takes GRASS, a foliage crown takes FOLIAGE (never its deep
// 爻爻爻 block: labelled bad, council 20261111-cliff-crown).
const ROCK=TONE_PREFERRED.rock,GRASS=TONE_PREFERRED.grass,FOLIAGE=TONE_PREFERRED.foliage,ROUND=TONE_PREFERRED.round;
const advanceOf=(index:number):number=>passages.headers[index*8+1]!;
const widest=(words:readonly number[]):number=>Math.max(...words.map(advanceOf));
// SLOTS: a lattice cell (between one and two slots on the page) holds its
// class's widest word and two half spaces, so no word reaches the next node.
const GAP=passages.gap;
// THE BAND'S TWO MEMBERS (skill section 8 rule 1): `:` (the ground member) and
// `;` (compiled-tone-passages.ts band-semicolon), alternating member by
// member. A band cell holds members at one even pitch (BAND_SLOT).
const BAND_COLON=GROUND_FIELD,BAND_SEMI=TONE_PASSAGES.findIndex(passage=>passage.id==='band-semicolon');
if(BAND_SEMI<0||advanceOf(BAND_COLON)!==advanceOf(BAND_SEMI))throw Error('Band members must be one advance');
const BAND_SLOT=24;
const FOLIAGE_SLOT=widest([FOLIAGE.fringe,FOLIAGE.mid,FOLIAGE.core])+2*GAP;
const SLOPE_SLOT=widest([SLOPE_DENSE,SLOPE_SPACED,ROCK.fringe,ROCK.mid,ROCK.core,ROCK.deep,GRASS.fringe,GRASS.mid,GRASS.core,GRASS.deep])+2*GAP;
// THE REGION FIELD'S WORDS (tone lane 2026-11-12), the wall context's own
// value ladder as the fog hand target spells it
// (hand-authored-targets-20261029/fog/rendered/000.txt rows 4-24): the light
// band's spaced `. : .` (toward the light passage), the mid band's continuous
// `.:.:.:`, the dark band's continuous `:;:;:;` (against the dark side; church
// aa[126] rows 16-36 and AAHub 背景 resK-70 thin the same way). A field cell of
// the surface lattice holds about four words (FIELD_UNITS), spread evenly over
// the cell, so the gaps between words stay under a half space.
const WALL_MID=wallWord('.:.:.:');
const FIELD_UNITS=4;
// The row units (compiled-tone-passages.ts FIELD_ROW_PASSAGES), a and b per
// level; b is taken on alternate lattice rows (the reference's own row
// alternation, ずらし).
const ROW_DARK_A=wallWord(':;:;:.:;:;'),ROW_DARK_B=wallWord('::;::;:;:.'),ROW_MID_A=wallWord('.:.:.:.:.:'),ROW_MID_B=wallWord(':.:.:.:.:.'),ROW_LIGHT_A=wallWord('. : . . :'),ROW_LIGHT_B=wallWord(': . . : .');
// The crown's row units (compiled-tone-passages.ts CROWN_ROW_PASSAGES).
const crownRowIndex=(id:string):number=>{const index=TONE_PASSAGES.findIndex(passage=>passage.id===id);if(index<0)throw Error(`Missing crown row ${id}`);return index;};
const CANOPY_LEFT_OUT=crownRowIndex('canopy-left-out'),CANOPY_RIGHT_IN=crownRowIndex('canopy-right-in');
// The sea's row units (compiled-tone-passages.ts SEA_ROW_PASSAGES); a sea
// cell holds two far units.
const SEA_CREST=crownRowIndex('sea-crest-level-0');
const SEA_FAR=crownRowIndex('sea-row-far'),SEA_MID=crownRowIndex('sea-row-mid'),SEA_NEAR=crownRowIndex('sea-row-near');
// tone2 g1: turf's one pattern (compiled-tone-passages.ts GRASS_PATCH_PASSAGES) and its paper, one full space.
const GRASS_PATCH=crownRowIndex('grass-patch'),GRASS_PAPER=11;
// tone2: the particle word (compiled-tone-passages.ts PARTICLE_PASSAGES) and its fixed page cell.
const PARTICLE=crownRowIndex('particle-spark'),PARTICLE_CELL=16;
const SEA_SLOT=2*(widest([SEA_FAR])+GAP);
const CROWN_DARK_A=crownRowIndex('crown-row-dark-a'),CROWN_DARK_B=crownRowIndex('crown-row-dark-b'),CROWN_MID_A=crownRowIndex('crown-row-mid-a'),CROWN_MID_B=crownRowIndex('crown-row-mid-b'),CROWN_LIGHT_A=crownRowIndex('crown-row-light-a'),CROWN_LIGHT_B=crownRowIndex('crown-row-light-b');
const FIELD_SLOT=FIELD_UNITS*(widest([ROW_DARK_A,ROW_DARK_B,ROW_MID_A,ROW_MID_B,ROW_LIGHT_A,ROW_LIGHT_B])+GAP);
const SLOPE_FIELD_SLOT=FIELD_UNITS*(widest([FRINGE,WALL_MID,DENSE_B,ROCK.fringe,ROCK.mid,ROCK.core,ROCK.deep,GRASS.fringe,GRASS.mid,GRASS.core,GRASS.deep])+GAP);
// THE ROUND'S ROW UNITS (tone2, 2026-11-13): the tonemapping hand target's rows
// inside each sphere's shadow quarter (hand-authored-targets-20261029/tonemapping/
// author.py S57, S63: '. :', ':.:;:', '.:;:;:;', '.:;:;:;:;', longer toward the
// rim), already in the bank as context round, source target:tonemapping/000.
const roundWord=(text:string):number=>{const found=TONE_PASSAGES.flatMap((passage,index)=>passage.context==='round'&&passage.text===text&&passage.source==='target:tonemapping/000'?[index]:[]);if(found.length!==1)throw Error(`Round tone passage ${text}: ${found.length} matches`);return found[0]!;};
const ROUND_LIGHT=roundWord('. :'),ROUND_MID=roundWord(':.:;:'),ROUND_CORE=roundWord('.:;:;:;'),ROUND_DEEP=roundWord('.:;:;:;:;');
// The paper kept inside the rim (the target's rows end 10-13 px inside the outline).
const ROUND_RIM_GAP=2*GAP+2;
/** THE REGION LOOKUP (WGSL regionOf, ToneField.planeTable): octahedral normal
 * bins per side, the slots a node reads, and the reach of a region's normal
 * tolerance (its exact test accepts about 5.7 degrees) in octahedral units. */
const REGION_OCT=16,REGION_PROBES=8,REGION_OCT_TOLERANCE=.1;
/** A plane region holds a turn band only when its world radius spans this
 * many lattice cells (the band itself reaches at most three). */
const REGION_LEAST_CELLS=4;
function octEncode(x:number,y:number,z:number):[number,number]{
 const s=Math.max(Math.abs(x)+Math.abs(y)+Math.abs(z),1e-20),ax=x/s,ay=y/s,az=z/s;
 if(az>=0)return [ax,ay];
 return [(1-Math.abs(ay))*(ax>=0?1:-1),(1-Math.abs(ax))*(ay>=0?1:-1)];
}
/** Same hash as the WGSL regionKey. */
function regionKey(domain:number,bx:number,by:number,offset:number):number{
 let h=(Math.imul(domain|0,0x9e3779b9)^Math.imul((bx+by*REGION_OCT)|0,0x85ebca6b)^Math.imul(offset|0,0xc2b2ae35))>>>0;
 h=(h^(h>>>15))>>>0;h=Math.imul(h,0x2c1b3c6d)>>>0;h=(h^(h>>>12))>>>0;
 return (h|1)>>>0;
}










export const TONE_ORDER_SHADER=/* wgsl */`
struct Q {rows:u32,capacity:u32,pad0:u32,pad1:u32}
@group(0) @binding(0) var<uniform> q:Q;
@group(0) @binding(1) var<storage,read> src:array<u32>;
@group(0) @binding(2) var<storage,read_write> rowWords:array<atomic<u32>>;
@group(0) @binding(3) var<storage,read_write> scattered:array<u32>;
@group(0) @binding(4) var<storage,read_write> sorted:array<u32>;
fn members()->u32{return min(src[0],q.capacity);}
@compute @workgroup_size(64) fn countRows(@builtin(global_invocation_id) id:vec3u){
 let i=id.x;if(i>=members()){return;}let row=src[4u+i*11u+1u];if(row<q.rows){atomicAdd(&rowWords[row],1u);}
}
@compute @workgroup_size(1) fn startRows(){
 var s=0u;for(var r=0u;r<q.rows;r++){atomicStore(&rowWords[q.rows+r],s);s+=atomicLoad(&rowWords[r]);}
 for(var w=0u;w<4u;w++){sorted[w]=src[w];}
}
@compute @workgroup_size(64) fn scatterRows(@builtin(global_invocation_id) id:vec3u){
 let i=id.x;if(i>=members()){return;}let row=src[4u+i*11u+1u];if(row>=q.rows){return;}
 let slot=atomicLoad(&rowWords[q.rows+row])+atomicAdd(&rowWords[2u*q.rows+row],1u);
 for(var w=0u;w<11u;w++){scattered[slot*11u+w]=src[4u+i*11u+w];}
}
// Key order: words 4,5,6,7,8,9,2,0,3 of the record, compared as u32 (the
// two f32 words are non-negative page X and reverse-Z depth: their bits
// order as the values do).
fn before(a:u32,b:u32)->bool{
 var order=array<u32,9>(4u,5u,6u,7u,8u,9u,2u,0u,3u);
 for(var k=0u;k<9u;k++){let x=scattered[a*11u+order[k]];let y=scattered[b*11u+order[k]];if(x!=y){return x<y;}}
 return false;
}
@compute @workgroup_size(64) fn rankRows(@builtin(global_invocation_id) id:vec3u){
 let i=id.x;if(i>=members()){return;}let row=scattered[i*11u+1u];if(row>=q.rows){return;}
 let start=atomicLoad(&rowWords[q.rows+row]);let end=start+atomicLoad(&rowWords[row]);
 // Equal records are identical words: either order writes the same list.
 var rank=0u;for(var j=start;j<end;j++){if(j!=i&&(before(j,i)||(j<i&&!before(i,j)))){rank++;}}
 for(var w=0u;w<11u;w++){sorted[4u+(start+rank)*11u+w]=scattered[i*11u+w];}
}
`;
/** Native row words of the per-row texture-edge table (rowPrelude). */
const ROW_STATE=2+64*2;

export const TONE_SHADER=/* wgsl */`
// fog: the scene's own medium (extinction density, height base, height
// falloff) — the atmosphere the drawing thins with (D3); zero when clear.
struct P {worldFromClip:mat4x4f,clipFromWorld:mat4x4f,rect:vec4f,field:vec4f,light:vec4f,grid:vec4u,ownership:vec4u,eye:vec4f,forward:vec4f,fog:vec4f}
struct Attachment {worldFromSource:mat4x4f,sourceFromWorld:mat4x4f,source:vec4u}
struct Out {count:atomic<u32>,flags:atomic<u32>,format:u32,capacity:u32,data:array<u32>}
@group(0) @binding(0) var<uniform> p:P;
@group(0) @binding(1) var depthField:texture_2d<f32>;
@group(0) @binding(2) var normalField:texture_2d<f32>;
@group(0) @binding(3) var ownerField:texture_2d<u32>;
@group(0) @binding(4) var<storage,read_write> out:Out;
@group(0) @binding(5) var appearance:texture_2d<f32>;
@group(0) @binding(6) var radiance:texture_2d<f32>;
// One table, two users. Legacy (litReference/main): sample, maximum,
// response. Compiled playback (regions/rank/passages) keeps the frame's
// REGION TABLE here; see "THE REGION TABLE" below for its fields.
struct Reference {sample:atomic<u32>,maximum:atomic<u32>,response:atomic<u32>,count:atomic<u32>,dark:atomic<u32>,left:atomic<u32>,right:atomic<u32>,top:atomic<u32>,bottom:atomic<u32>,best:atomic<u32>,ownerX:atomic<u32>,ownerY:atomic<u32>}
@group(0) @binding(7) var<storage,read_write> references:array<Reference>;
@group(0) @binding(8) var<storage,read> attachments:array<Attachment>;
// Engine SunShadow/Cascade layout (four cascades, 96-byte stride). These
// are the actual caster map and receiver bias, not a bright-material proxy.
struct Cascade {lightViewProj:mat4x4f,rect:vec4f,far:f32,texelWorld:f32}
struct SunShadow {cascades:array<Cascade,4>,lightDirection:vec4f,count:f32,overlap:f32,depthBias:f32,enabled:f32,normalBias:f32}
@group(0) @binding(9) var<uniform> sun:SunShadow;
@group(0) @binding(10) var sunMap:texture_depth_2d;
@group(0) @binding(11) var sunSampler:sampler_comparison;
// Same-pose D10/source construction, not a later live overlay. Its line
// pixels have alpha 1; value-band pixels have alpha .5 and paper has 0.
@group(0) @binding(12) var construction:texture_2d<f32>;
// Compiled source construction classes (domain, form: 1 planar, 2 rounded,
// 0 field), sorted by domain. The region decision is construction-owned:
// a face of a planar volume takes the wall/plane passages, a round its band
// passages. The producer never invents a region from a dark pixel.
@group(0) @binding(13) var<storage,read> styles:array<vec2u>;
// The field's per-form table (form-depths.ts), read-only (toneTier).
@group(0) @binding(14) var formDepths:texture_2d<u32>;
// The face's authored material class (physical-frame.ts writes class / 255
// into the field's location-2 target; source-material.ts classes 0-8).
@group(0) @binding(16) var classField:texture_2d<f32>;
 // Primary appearance keeps native alpha. Legacy/physical producers retain
 // the old alpha marker until they adopt this same optional coverage input.
 @group(0) @binding(17) var<storage,read> particleCoverage:array<u32>;
 fn particleAt(i:vec2i)->bool{
  if(particleCoverage[0]==0u){return textureLoad(radiance,i,0).a>=3.5;}
  let bit=u32(i.y)*u32(p.field.x)+u32(i.x);let word=1u+(bit>>5u);
  return word<arrayLength(&particleCoverage)&&(particleCoverage[word]&(1u<<(bit&31u)))!=0u;
 }
fn materialAt(q:vec2f,o:vec2u)->u32{
 if((o.x&0xc0000000u)==0x80000000u){
  let i=vec2i(floor(q));
  if(all(i>=vec2i(0))&&all(i<vec2i(textureDimensions(classField)))){let c=u32(round(textureLoad(classField,i,0).r*255.0))&15u;if(c!=0u){return c;}}
 }
 return (styleWord(o.x)>>6u)&15u;
}
// Keep these at module scope (origin 6b94869970 / f73289f830). Function-local
// copies caused multi-minute Metal compilation and failures on Intel GPUs.
const TONE_INK=array<vec2i,${toneInk.length+passages.ink.length/2}>(${toneInk.map(([x,y])=>`vec2i(${x},${y})`).concat(Array.from({length:passages.ink.length/2},(_,i)=>`vec2i(${passages.ink[i*2]},${passages.ink[i*2+1]})`)).join(',')});
const TONE_HEADERS=array<vec3u,${TONE_PASSAGES.length}>(${TONE_PASSAGES.map((_,i)=>`vec3u(${toneInk.length+passages.headers[i*8+3]!}u,${passages.headers[i*8+4]}u,${passages.headers[i*8+1]}u)`).join(',')});
fn styleWord(source:u32)->u32{
 let domain=source&0x5fffffffu;var lo=0u;var hi=p.ownership.z;
 while(lo<hi){let mid=(lo+hi)/2u;if(styles[mid].x<domain){lo=mid+1u;}else{hi=mid;}}
 if(lo<p.ownership.z&&styles[lo].x==domain){return styles[lo].y;}
 return 0u;
}
fn styleOf(source:u32)->u32{return styleWord(source)&7u;}
fn liquidOf(source:u32)->bool{return (styleWord(source)&8u)!=0u;}
fn lightDirection()->vec3f{
 if(p.ownership.y!=0u&&sun.enabled!=0.0){return sun.lightDirection.xyz;}
 return p.light.xyz;
}
// A level face takes no field from its own turn, lit or not: a floor, lid or
// open ground carries only the cast shadows that lie on it and, when larger
// than the page shows, sparse receding intervals (levelUnits); a ceiling is
// paper, as the fog and tonemapping hand targets draw every floor
// (hand-authored-targets-20261029/fog/rendered/000.txt rows 23-28: the
// ;:;:;:. . / :;:. . / :. . tails under each block; the ceiling is blank).
// A planar volume's face is level within 45 degrees; a facet-built form's
// facet only under about 26 degrees of slope (a steeper facet is a slope).
// These two bounds are geometry classes of this rule, not light values.
fn levelFace(n:vec3f,style:u32)->bool{return select(n.y>.9,abs(n.y)>.7,style==1u);}
// FOLIAGE (styles bit 4): the source's compiled geometry class says it is a
// crown (composition.ts foliage rule, decided once per source revision, never
// per screen row). Its facet-built shade takes the FOLIAGE words.
fn foliageOf(source:u32)->bool{return (styleWord(source)&16u)!=0u;}
// ORGANIC (styles bit 5): a facet-built surface whose faces turn every way and
// that is not a crown (composition.ts foliageMass: a boulder, a rock). Its
// facets are flat but its form is round: it takes the round's band at its
// shaded rim (organicLevel), never a band per facet.
fn organicOf(source:u32)->bool{return (styleWord(source)&32u)!=0u;}
fn cascadeShadow(index:u32,P:vec3f,n:vec3f)->f32{
 let c=sun.cascades[index];
 let receiver=P+normalize(sun.lightDirection.xyz)*sun.depthBias+n*(sun.normalBias*1.4142136*c.texelWorld);
 let clip=c.lightViewProj*vec4f(receiver,1);let q=clip.xyz/clip.w;
 if(any(abs(q.xy)>vec2f(1))||q.z<0.0||q.z>1.0){return 1.0;}
 let halfTexel=vec2f(.5)/vec2f(textureDimensions(sunMap));
 let uv=clamp(c.rect.xy+vec2f(q.x*.5+.5,.5-q.y*.5)*c.rect.zw,c.rect.xy+halfTexel,c.rect.xy+c.rect.zw-halfTexel);
 return textureSampleCompareLevel(sunMap,sunSampler,uv,q.z);
}
fn castVisibility(P:vec3f,n:vec3f)->f32{
 if(p.ownership.y==0u||sun.enabled==0.0){return 1.0;}
 let z=dot(P-p.eye.xyz,p.forward.xyz);let count=min(4u,u32(sun.count));
 for(var i=0u;i<count;i++){
  if(z>=sun.cascades[i].far){continue;}
  var value=cascadeShadow(i,P,n);
  if(i+1u<count){let far=sun.cascades[i].far;let near=(1.0-sun.overlap)*far;
   if(z>=near){value=mix(value,cascadeShadow(i+1u,P,n),clamp((z-near)/max(far-near,.00001),0.0,1.0));}
  }
  return value;
 }
 return 1.0;
}
fn lightBand(n:vec3f,v:vec3f,P:vec3f)->u32{
 // Only a scene-supplied light owns a surface band. Never substitute the
 // camera, beauty brightness, a material's dark colour or the backdrop.
 let direction=lightDirection();
 if(dot(direction,direction)<.000001){return 0u;}
 let nl=dot(n,normalize(direction));
 // A caster region is distinct from surface-turn tone. It does not license
 // punctuation across the receiver plane: authored ground has contact
 // intervals, not an area fill. Keep the class so contact construction can
 // consume it without confusing it with an object's interior value band.
 if(nl>0.0&&castVisibility(P,n)<.4){return 1u;}
 if(nl>=.7){return 0u;}
 return select(select(4u,3u,nl>=.15),2u,nl>=.4);
}
struct S {d:f32,n:vec3f,o:vec2u,v:vec3f,on:bool}
fn attachmentFor(owner:vec2u)->u32{
 let key=vec2u(owner.x&0xdfffffffu,owner.y);var lo=0u;var hi=p.ownership.x;
 // Exact source address lookup only. No drawing, glyph, pen or fit is chosen.
 while(lo<hi){let mid=(lo+hi)/2u;let source=attachments[mid].source.xy;
  if(source.x<key.x||(source.x==key.x&&source.y<key.y)){lo=mid+1u;}else{hi=mid;}
 }
 if(lo<p.ownership.x){if(all(attachments[lo].source.xy==key)){return lo;}}
 return 0xffffffffu;
}
// The lattice's frame: only an attachment that poses the rendered surface
// (the physical producer's certified source pose). A REGION POSE (source.z 1:
// an entity's own transform, ToneField.encode) addresses the source's plane
// regions (darkEdgeDistance verifies the point lies on the region's plane)
// but never moves the lattice.
fn latticeAttachment(owner:vec2u)->u32{
 let a=attachmentFor(owner);
 if(a!=0xffffffffu&&attachments[a].source.z!=0u){return 0xffffffffu;}
 return a;
}
fn octDecode(e:vec2f)->vec3f{
 var n=vec3f(e,1.0-abs(e.x)-abs(e.y));
 if(n.z<0.0){n=vec3f((vec2f(1.0)-abs(n.yx))*select(vec2f(-1.0),vec2f(1.0),n.xy>=vec2f(0.0)),n.z);}
 return normalize(n);
}
fn unproject(q:vec2f,d:f32)->vec3f{let c=p.worldFromClip*vec4f(q/p.field.xy*vec2f(2,-2)+vec2f(-1,1),d,1);return c.xyz/c.w;}
fn toField(page:vec2f)->vec2f{return (page-p.rect.xy)*p.field.xy/p.rect.zw;}
// One native member receives the authored material's already-lit radiance at
// its own accepted lattice node. This transports source material colour; it
// is not a beauty-buffer read, does not choose glyphs, and never creates a
// second render target or a second scene render.
// The frame's material inks (gpu-articulation.ts MATERIAL INK), the same
// table the line words read: one form, one hue in lines and tone.
@group(0) @binding(18) var materialInks:texture_2d<u32>;
${MATERIAL_INK_WGSL}
fn materialForeground(page:vec2f,owner:vec2u)->u32{
 // A form with a chromatic material keeps its material hue (examples-89,

 // the incumbent transport unchanged.
 let ink=materialInk(owner);if(ink!=0u){return ink;}
 let dimensions=vec2i(textureDimensions(radiance));
 let q=clamp(vec2i(floor(toField(page))),vec2i(0),dimensions-vec2i(1));
 let rgb=clamp(textureLoad(radiance,q,0).rgb,vec3f(0.0),vec3f(1.0));
 return pack4x8unorm(vec4f(rgb,1.0));
}
// CELL-RESOLUTION FIELD (2026-11-12). The field and the construction (D10)
// image may be the glyph grid, not the page: the image is read in its own
// texels, page bounds come from the page extent (p.field.zw), and a walk along
// a row steps one field column. With a page-sized field all three are what
// they were (colStep 1, identity mapping).
// A construction texel of the glyph-grid image (3 x 5.7 page px) stands for a
// line about one page pixel wide somewhere inside it. Read whole, every line
// would clear tone across the full texel: on Y8 view-02 the tone units fell
// by about a quarter. A probe reads the texel only at the page pixel that
// holds the texel's centre, so lines block tone at about their true width; a
// page-sized image (every pixel a centre) reads as before.
fn markAt(page:vec2i)->vec4f{
 let size=vec2f(textureDimensions(construction));let t=(vec2f(page)+vec2f(.5))*size/p.field.zw;let texel=floor(t);
 let centre=(texel+vec2f(.5))*p.field.zw/size;
 if(any(abs(centre-(vec2f(page)+vec2f(.5)))>vec2f(.5))){return vec4f(0.0);}
 return textureLoad(construction,vec2i(texel),0);
}
fn colStep()->i32{return max(1,i32(round(p.field.z/p.field.x)));}
fn at(q:vec2f)->S{
 let size=vec2i(p.field.xy);let i=vec2i(floor(q));
 if(any(i<vec2i(0))||any(i>=size)){return S(0.0,vec3f(0),vec2u(0),vec3f(0),false);}
 let o=textureLoad(ownerField,i,0).xy;let d=textureLoad(depthField,i,0).x;
 if((o.x|o.y)==0u||d<=0.0){return S(0.0,vec3f(0),vec2u(0),vec3f(0),false);}
 let c=vec2f(i)+vec2f(.5);let v=normalize(unproject(c,1.0)-unproject(c,d));
 let raw=octDecode(textureLoad(normalField,i,0).xy);
 return S(d,select(raw,-raw,dot(raw,v)<0.0),o,v,true);
}
// The stand-in's cheap read: the same surface as a (owner, no depth step, the
// normal within 40 degrees without its view sign), from the raw fields only.
// planar: also the normal within 40 degrees (a box's faces are separate
// surfaces); otherwise owner and depth only, as the field's clearance rule
// (unitClear) reads a facet-built form, a crown or a slope.
fn sameRaw(a:S,q:vec2f,planar:bool)->bool{
 let i=vec2i(floor(q));if(any(i<vec2i(0))||any(i>=vec2i(p.field.xy))){return false;}
 let o=textureLoad(ownerField,i,0).xy;let d=textureLoad(depthField,i,0).x;
 if(any(o!=a.o)||d<=0.0||abs(a.d-d)>select(.08,.03,p.grid.w!=0u)*max(a.d,d)){return false;}
 if(!planar){return true;}
 return abs(dot(a.n,octDecode(textureLoad(normalField,i,0).xy)))>.766;
}
fn same(a:S,b:S)->bool{return b.on&&all(a.o==b.o)&&abs(a.d-b.d)<=select(.08,.03,p.grid.w!=0u)*max(a.d,b.d)&&dot(a.n,b.n)>.766;}
// A construction-line pixel that is a seam between facets of ONE surface: the
// D10 crease class (underdrawing.ts: a normal turn over 40 degrees inside one
// owner with no depth step, drawn .95,.45,0) or a compiled mesh relation
// interval (0,.62,.9). On a facet-built form these are the triangle seams of
// its mesh; they bound no value region. Silhouette, occluding, contact and
// suggestive lines keep their own class colour and stay strokes.
fn facetSeam(mark:vec4f)->bool{
 let crease=mark.r>.8&&mark.g>.3&&mark.g<.6&&mark.b<.2;
 let relation=mark.r<.2&&mark.g>.5&&mark.b>.75;
 return crease||relation;
}
// THE ONE TIER READ OF THIS FILE: distanceTierOf, the line producer's own
// read (gpu-articulation.ts "THE TIER IS THE ROWS THE DRAWING HAS"), over
// the SAME form table of this field (formDepths, binding 14; form-depths.ts
// FormDepths). The compact-form rule, the per-point rule for terrain and
// spanning owners (a ground or water plane, a floor), the reference length,
// and the parallel and enclosed rules are stated there, once. This producer
// supplies only its terrain test (construction class 3) and the cell it
// asks about, so every cell of one form holds that form's one tier.
//
// TONE BY TIER. Distance does not stop tone: past the red line of the

// "EXAMPLE OF MEDIUM FAR LOD" and "EXAMPLE SUPER FAR LOD") the dot content
// carries most of the drawing and the lines thin instead. The word table is
// the same at every tier (passages, wordOf); the tier, the form's own size on
// the page, changes only how dense a unit may be and how many members stay:
//  near   (0) every level of the word table.
//  middle (1) level at most 2 (a side turned from the light keeps its band,
//             never the densest word).
//  far    (2) the same, with every other lattice node erased (skill section
//             8 rule 2: a band is lightened by erasing members, never by a
//             lighter glyph), so the far band keeps its dotted ground and
//             shaded sides under the strip's marks.
// Level ground is graded by its own scale instead (levelUnits).
${DISTANCE_TIER_WGSL}
fn toneTier(q:vec2f,d:f32,o:vec2u)->u32{
 // A selected parent owns this member's complete smallest-tier depiction.
 // The same table flag gates its fine surface/texture producers below.
 if(formParentSelected(o)){return 2u;}
 return distanceTierOf(o,styleOf(o.x)==3u,floor(q)+vec2f(.5),d,p.field.xy,${TONE_ROW}.0*p.field.y/p.rect.w,p.worldFromClip);
}
// GROUND: a level face that looks up (a terrain's level part, a ground or
// water plane, a floor), of any construction class but a round. The lid of a
// compact box holds the box's one tier (toneTier). A ground that is read per
// point (terrain, a ground plane spanning depth) in a receding view is
// GRADED, never switched at a tier line: groundRows is the same per-point row
// measure the tier bands (gpu-articulation.ts pointRowsOf), and the dot rows
// thin toward the camera by erasing lattice members (skill section 8 rule 2),
// at a step that is a power of two of that measure (groundGrade), so a node
// only appears or disappears as the camera moves, never slides. A room's
// floor (an enclosed view) and a parallel view are not graded: no rows (D6);
// an open room's floor holds 24 rows or more of L everywhere, which erases
// every member (groundGrade 4), so it stays paper too (D7).
fn groundFace(n:vec3f,style:u32)->bool{return n.y>0.0&&levelFace(n,style);}
// OPEN GROUND holds the even field: a height surface (terrain is outdoors),
// or a level face read per point or cut by the page that runs up to open
// background somewhere in view (its horizon). A room's floor ends at its
// walls and stays paper with its contacts (D6, D7); a lid the page holds
// whole is its form's top.
fn openGround(q:vec2f,d:f32,o:vec2u,style:u32)->bool{
 if(!viewRecedes(p.worldFromClip)){return false;}
 // Open ground is the outdoors by the source's own class: a height surface
 // (terrain) or a liquid surface (the adapter's fact: the sea). Any other
 // level face (a room's floor, a scene's ground plane, a lid) is a floor:
 // paper with the cast shadows on it (D6 bans ground drawn as a dot row; the
 // tonemapping and fog targets keep their floors nearly empty).
 return style==3u||liquidOf(o.x);
}
// THE GROUND'S OWN SCALE: the rows the camera's height above the point
// covers at the point. On level ground that is the point's distance below its
// horizon on the page, so the field is densest under the horizon and thins to
// paper toward the viewer (skill section 8 rule 2: lighten by erasing members;

// hand-authored tonemapping target: a floor seen from close is paper). No
// scene length is voted or guessed: the camera and the surface give it.
fn groundRows(q:vec2f,d:f32)->f32{
 let P=unproject(q,d);let across=length(unproject(q+vec2f(1.0,0.0),d)-P);
 // The eye is the point every view ray leaves: the clip point with w = 0.
 let e=p.worldFromClip*vec4f(0.0,0.0,1.0,0.0);if(abs(e.w)<1e-9){return 0.0;}
 return abs(e.y/e.w-P.y)/max(across,1e-6)*p.rect.w/p.field.y/${TONE_ROW}.0;
}
// Erase step by rows L holds at the point, one halving per doubling of the
// rows from the far band (FAR_STRIP_BELOW_ROWS):
//  0 under 3 rows: every node of every lattice row (densest, the far band);
//  1 under 6: every 2nd node;  2 under 12: every 4th node;
//  3 under 24: every 4th node of every 2nd lattice row;
//  4 from 24 rows (where a typical form would stand 24 rows tall, beside the
//    camera or across a room's floor): the last members erased, paper.
fn groundGrade(rows:f32)->u32{return u32(clamp(floor(log2(max(rows,1e-6)/${FAR_STRIP_BELOW_ROWS}.0))+1.0,0.0,4.0));}
fn toneFits(page:vec2f,row:u32,level:u32,source:S,band:u32)->bool{
 let dark=level==3u;
 let first=select(0u,${toneUnits[0]!.points.length}u,dark);
 let count=select(${toneUnits[0]!.points.length}u,${toneUnits[1]!.points.length}u,dark);
 let advance=select(${toneUnits[0]!.advance}.0,${toneUnits[1]!.advance}.0,dark);
 let pen=i32(round(page.x-advance*.5));let size=vec2i(p.field.zw);
 for(var i=first;i<first+count;i++){
  // Native cursor admission can move a tone unit by at most two pixels.
  // Certify that entire bounded displacement here, not only its seed cell.
  for(var shift=-2;shift<=2;shift++){
   let pixel=vec2i(pen+shift,i32(row)*${TONE_ROW})+TONE_INK[i];
   if(any(pixel<vec2i(0))||any(pixel>=size)){return false;}
   let q=toField(vec2f(pixel)+vec2f(.5));let sample=at(q);
   if(!same(source,sample)||markAt(pixel).a>.75){return false;}
   if(textureLoad(radiance,vec2i(q),0).a<.99){return false;}
   if(lightBand(sample.n,sample.v,unproject(floor(q)+vec2f(.5),sample.d))!=band){return false;}
  }
 }
 return true;
}
// Passage header: ink offset in the ink table, ink count, native advance.
fn passageHeader(passage:u32)->vec3u{
 return TONE_HEADERS[passage];
}
// THE SURFACE LATTICE OF A REGION'S MEMBERS (agreed design, Tone and
// Correspondence: the pattern is anchored to the surface, never the screen).
//
// The text row stays the medium: a member sits on its native row. WHICH
// surface points carry members is fixed on the surface. Along the row, the
// surface (the local plane of the pixel's own depth and normal) meets the
// row's view plane in one line. Its lattice coordinate u is the source axis
// (the owner's certified source frame when the producer attached one, else
// world) that changes most along that line. Members sit where u = node * s,
// s the power of two with the node interval at least one unit slot wide on
// the page (target px; between one and two slots). A node's page X is the
// exact projection of (surface plane, row plane, lattice plane u = node * s):
// three planes, one point, no trial and no scan. The pen is that X rounded to
// the native pixel, plus one native half space on alternate lattice cells
// across the row (ずらし, read from the surface's second axis at about one
// row per cell, never from the page row's number).
// Translation carries every node with its surface. A scale change keeps the
// nodes of the coarser lattice (they are nodes of the finer one too).
struct Lattice {x0:f32,x1:f32,node:i32,level:i32,axis:u32,vcell:i32,vlevel:i32,wcell:i32,odd:bool,ok:bool}
fn clipRow(r:u32)->vec4f{var m=p.clipFromWorld;return vec4f(m[0][r],m[1][r],m[2][r],m[3][r]);}
// The view planes of a page row's centre line and of a page column.
fn rowPlane(y:f32)->vec4f{return clipRow(1u)-(1.0-2.0*(y-p.rect.y)/p.rect.w)*clipRow(3u);}
fn colPlane(x:f32)->vec4f{return clipRow(0u)-(2.0*(x-p.rect.x)/p.rect.z-1.0)*clipRow(3u);}
// The point of three planes (xyz normal, w offset), solved about a nearby
// origin so a large world coordinate costs no precision; w 0 when they do
// not meet in one point.
fn meet(o:vec3f,a:vec4f,b:vec4f,c:vec4f)->vec4f{
 let wa=dot(a.xyz,o)+a.w;let wb=dot(b.xyz,o)+b.w;let wc=dot(c.xyz,o)+c.w;
 let bc=cross(b.xyz,c.xyz);let det=dot(a.xyz,bc);
 if(abs(det)<=1e-6*length(a.xyz)*length(b.xyz)*length(c.xyz)){return vec4f(0.0);}
 return vec4f(o-(wa*bc+wb*cross(c.xyz,a.xyz)+wc*cross(a.xyz,b.xyz))/det,1.0);
}
// A lattice axis as a plane family: source coordinate = dot(xyz, world) + w.
fn latticeRow(attachment:u32,axis:u32)->vec4f{
 if(attachment==0xffffffffu){return vec4f(f32(axis==0u),f32(axis==1u),f32(axis==2u),0.0);}
 var m=attachments[attachment].sourceFromWorld;
 return vec4f(m[0][axis],m[1][axis],m[2][axis],m[3][axis]);
}
// The lattice cell that holds page position (x, y) on the surface plane
// through P with normal n: its two bounding nodes' page X (x0 < x1), the
// signed node index of x0 and the cell addresses on the other two axes.
fn latticeAt(x:f32,y:f32,n:vec3f,P:vec3f,attachment:u32,slotPx:f32)->Lattice{
 var L:Lattice;L.ok=false;
 let plane=vec4f(n,-dot(n,P));let row=rowPlane(y);
 let A=meet(P,plane,row,colPlane(x));let B=meet(P,plane,row,colPlane(x+8.0));
 if(A.w==0.0||B.w==0.0){return L;}
 let D=B.xyz-A.xyz;
 var axis=0u;var du=0.0;
 for(var a=0u;a<3u;a++){let d=dot(latticeRow(attachment,a).xyz,D);if(abs(d)>abs(du)){du=d;axis=a;}}
 let perPx=abs(du)/8.0;
 if(!(perPx>1e-9)||!(perPx<1e9)){return L;}
 let level=i32(ceil(log2(slotPx*perPx)));
 if(level<-40||level>40){return L;}
 let s=exp2(f32(level));let dir=select(-1.0,1.0,du>0.0);
 let r=latticeRow(attachment,axis);
 let m=floor((dot(r.xyz,A.xyz)+r.w)*dir/s);
 if(!(abs(m)<8388608.0)){return L;}
 let n0=meet(P,plane,row,vec4f(r.xyz,r.w-m*s*dir));let n1=meet(P,plane,row,vec4f(r.xyz,r.w-(m+1.0)*s*dir));
 if(n0.w==0.0||n1.w==0.0){return L;}
 let c0=p.clipFromWorld*vec4f(n0.xyz,1.0);let c1=p.clipFromWorld*vec4f(n1.xyz,1.0);
 if(c0.w<=0.0||c1.w<=0.0){return L;}
 L.x0=p.rect.x+(c0.x/c0.w*.5+.5)*p.rect.z;L.x1=p.rect.x+(c1.x/c1.w*.5+.5)*p.rect.z;
 if(!(L.x1>L.x0)||!(L.x1-L.x0<4.0*slotPx+16.0)){return L;}
 // The other two axes: v runs across the row on the surface, w is the rest.
 let E=cross(n,D);let b=(axis+1u)%3u;let c=(axis+2u)%3u;
 let rb=latticeRow(attachment,b);let rc=latticeRow(attachment,c);
 let across=abs(dot(rb.xyz,E))>=abs(dot(rc.xyz,E));
 var rv=rc;var rw=rb;if(across){rv=rb;rw=rc;}
 let v=dot(rv.xyz,n0.xyz)+rv.w;
 let below=meet(P,plane,rowPlane(y+${TONE_ROW}.0),colPlane(x));
 var perRow=s;
 if(below.w!=0.0){let dv=abs(dot(rv.xyz,below.xyz-A.xyz));if(dv>1e-9&&dv<1e9){perRow=dv;}}
 // Cell address: at most one row per cell, so two rows never share one.
 L.vlevel=clamp(i32(floor(log2(perRow))),-32,31);
 L.vcell=i32(floor(v/exp2(f32(L.vlevel))));
 // Stagger: about one row per cell, so neighbouring rows alternate.
 L.odd=(i32(floor(v/exp2(round(log2(perRow)))))&1)!=0;
 // A third of a cell off the dyadic planes: a face that lies on one (a wall
 // at a whole coordinate) does not flip its address with depth noise.
 L.wcell=i32(floor((dot(rw.xyz,n0.xyz)+rw.w)/s+.3333));
 L.node=i32(m*dir);L.level=level;L.axis=axis;L.ok=true;
 return L;
}
// A member's correspondence key (record words 4..7): the lattice node, not a
// position on the page. Word 4 is the node index on the u axis reduced to its
// odd part (a node that survives a lattice scale change keeps it), word 5 the
// cell across the row, word 6 the cell on the third axis (which parallel face
// of one owner), word 7 packs band (bits 0-2), passage (3-10), the unit's
// ordinal after its node (11-16), the u axis (17-18), the reduced lattice
// exponent + 64 (19-25) and the cross-row exponent + 32 (26-31). Word 7 is
// never zero for a lattice key: zero there marks a bound-anchored unit.
fn latticeKey(L:Lattice,j:u32,passage:u32,band:u32)->vec4u{
 var a=L.node;var e=L.level;
 if(a==0){e=0;}else{while((a&1)==0&&e<60){a/=2;e++;}}
 return vec4u(bitcast<u32>(a),bitcast<u32>(L.vcell),bitcast<u32>(L.wcell),
  (band&7u)|((passage&255u)<<3u)|(min(j,63u)<<11u)|(L.axis<<17u)|((u32(e+64)&127u)<<19u)|((u32(L.vlevel+32)&63u)<<26u));
}
// One placed unit's record (layout: ToneField "whole-passages-v1" below).
fn emitPassage(row:u32,y:f32,pen:f32,passage:u32,band:u32,owner:vec2u,key:vec4u,depth:f32){
 if(formParentSelected(owner)){return;}
 // KEY QUANTA: page X to 1/64 px and depth with its low 8 mantissa bits
 // cleared (monotone, so the order is kept), so the two float sort-key words
 // tie identically on every GPU.
 let advance=f32(passageHeader(passage).z);let centre=round((pen+advance*.5)*64.0)/64.0;
 let slot=atomicAdd(&out.count,1u);if(slot>=p.grid.z){atomicOr(&out.flags,1u);return;}
 out.data[slot*11u]=bitcast<u32>(centre);out.data[slot*11u+1u]=row;out.data[slot*11u+2u]=passage;out.data[slot*11u+3u]=bitcast<u32>(max(depth,0.0))&0xffffff00u;
 out.data[slot*11u+4u]=key.x;out.data[slot*11u+5u]=key.y;out.data[slot*11u+6u]=key.z;out.data[slot*11u+7u]=key.w;
 out.data[slot*11u+8u]=owner.x;out.data[slot*11u+9u]=owner.y;out.data[slot*11u+10u]=materialForeground(vec2f(centre,f32(row*${TONE_ROW}+8)),owner);
}
// The scene's own atmosphere between the eye and a unit (aerial perspective:
// the fog target's shadow wall thins into the hall). A clear scene is 1.
fn transmittance(q:vec2f,P:vec3f)->f32{
 if(p.fog.x<=0.0){return 1.0;}
 let density=p.fog.x*exp(-p.fog.z*(P.y-p.fog.y));
 return exp(-density*length(P-unproject(floor(q)+vec2f(.5),1.0)));
}
// TURF: the surface's own native radiance (material lighting, before any
// post effect) is green-dominant, its green channel a tenth over red and over
// blue. The tenth is this rule's own choice, not a measured reference value.
fn turf(q:vec2f)->bool{let c=textureLoad(radiance,vec2i(q),0).rgb;return c.g>c.r*1.1&&c.g>c.b*1.1;}
// D10's suggestive contour colour (underdrawing.ts, magenta .85,.05,.55).
fn suggestiveLine(mark:vec4f)->bool{return mark.r>.7&&mark.g<.2&&mark.b>.4&&mark.b<.7;}

// THE UNIT CLASSES (one per pixel's surface, from the compiled source class):
const KIND_PLANE=1u;   // a non-level face of a planar volume (wall, block side)
const KIND_FACET=2u;   // a non-level facet of a facet-built form (rock, house, trunk, crown)
const KIND_ROUND=3u;   // a rounded form
const KIND_SLOPE=4u;   // a terrain slope
const KIND_GROUND=5u;  // open level ground (terrain level, a spanning ground or water plane)
const KIND_FLOOR=6u;   // any other level face (a floor, a lid): paper unless a cast shadow lies on it
// Clearance modes: which drawn lines bound a unit (lineBlocks).
const MODE_PLANE=0u;const MODE_ROUND=1u;const MODE_FACET=2u;const MODE_TERRAIN=5u;
// A LINE BOUNDS TONE ONLY WHERE THE LINE IS DRAWN. A facet-built form's mesh
// seams (facetSeam) are not lines; in the middle and far views the line
// producer draws no interior suggestive curve (gpu-articulation.ts
// TIER_SELECTION), so it bounds nothing there; a terrain's candidate lines are
// mostly not drawn (tierSelects), and the typesetter sets every drawn line
// word before a tone word, so terrain units are bounded by their surface only.
fn lineBlocks(mark:vec4f,mode:u32,far:bool)->bool{
 if(mark.a<=.75||mode==MODE_TERRAIN){return false;}
 if(mode==MODE_PLANE){return true;}
 // A round's and a facet-built form's mesh seams are not lines.
 if(facetSeam(mark)){return false;}
 return !(far&&suggestiveLine(mark));
}
// THE CLEARANCE RULE (fixed; one test at the unit's one pen, no other pen is
// ever tried): the unit and one native half space either side of it lie on
// the node's own surface (same owner, no depth step; a planar face also the
// same face), opaque, and no drawn line pixel lies in that box at three
// heights of the row. A unit that fails is not placed (it is counted in
// out.flags >> 8); its neighbours keep their nodes.
fn unitClear(pen:f32,row:u32,y:f32,advance:f32,s:S,mode:u32,far:bool)->bool{
 let size=vec2i(p.field.zw);
 let a=i32(round(pen))-${passages.gap};let b=i32(round(pen+advance))+${passages.gap};
 if(a<0||b>=size.x){return false;}
 for(var k=0;k<3;k++){
  let x=mix(f32(a),f32(b),f32(k)*.5);let q=toField(vec2f(x+.5,y));let t=at(q);
  if(!t.on||any(t.o!=s.o)){return false;}
  if(abs(t.d-s.d)>.08*max(t.d,s.d)){
   // tone2 c1: on the holder's own plane (its depth line along the row) within 3%.
   if(!(clearRef.w>0.0&&clearRef.y==s.d)){return false;}
   let tx=p.rect.x+(floor(q.x)+.5)*p.rect.z/p.field.x;let pd=clearRef.w+clearRef.z*(tx-clearRef.x);
   if(!(pd>0.0)||abs(t.d-pd)>.03*max(t.d,pd)){return false;}
  }
  if(mode==MODE_PLANE&&dot(t.n,s.n)<.766){return false;}
  if(textureLoad(radiance,vec2i(q),0).a<.99){return false;}
 }
 for(var dy=3;dy<${TONE_ROW};dy+=5){
  for(var x=a;x<=b;x+=2){
   if(lineBlocks(markAt(vec2i(x,i32(row)*${TONE_ROW}+dy)),mode,far)){return false;}
  }
 }
 return true;
}
// THE UNIT'S LEVEL. Tone is ORGANIZED by the form and the light (skill
// section 8, mark-purpose test: reading, organization, owner); most of a
// surface stays paper. Every quantity below is attached to the surface point:
// its facet's n·L, the caster's shadow on it, its place on its own plane
// region, the air before it. Fixed thresholds; nothing is ranked or tried.
//
// AIR (aerial perspective): one level less under transmittance .75, another
// under .45, paper under .2.
fn airLevel(level:i32,air:f32)->i32{
 var l=level;if(air<.75){l-=1;}if(air<.45){l-=1;}if(air<.2){l=0;}
 return max(l,0);
}
// A CURVED SURFACE (a round, a facet-built rock, trunk or crown) carries its
// band at the terminator: the side turning away from the light, densest
// where it has turned furthest, paper where it faces the light (the
// tonemapping target's wedge inside each sphere hugging its shadow rim:
// ". :" toward the middle, ".:;:;:;" against the rim; C#126's thinning beside
// the lit paper). n·L under -.1: 3; under .15: 2; under .35: 1; else paper. A
// cast shadow keeps at least level 1.
// AN ORGANIC FORM (a boulder: organicOf) carries the round's band only where
// its shaded side turns to the silhouette (n·L under .15 and the surface
// grazing the view, |n·v| under .45: the rim wedge, abutted in passages); its
// shaded core facing the viewer is paper (a field there fills a boulder
// whole). Its facets are flat, so their own n·L never fills one facet whole.
// A ROUND (rounded class) carries ONE WEDGE on its shaded side, against its
// rim, as the tonemapping target frame-000 draws each sphere: sparse ". :"
// toward the middle densifying to ".:;:;:;" against the inner rim at the
// shadow side, the rest of the round paper. Shaded (n·L under .15) and
// grazing the view (|n·v| under .45): the rim wedge (level 3, abutted in
// passages); shaded and turning toward the rim (|n·v| under .65): the
// sparsest level (every 4th member); else paper. The dark side's middle,
// facing the viewer, is paper: a field there filled every boulder and every
// sphere's dark half (tone-bands-d).
fn roundLevel(nl:f32,nv:f32)->i32{
 if(nl>=.15){return 0;}
 if(nv<.45){return 3;}
 return select(0,1,nv<.65);
}
fn organicLevel(nl:f32,nv:f32)->i32{return select(0,2,nl<.15&&nv<.45);}
fn curvedLevel(nl:f32,band:u32)->i32{
 var level=0;
 if(nl< -.1){level=3;}else if(nl<.15){level=2;}else if(nl<.35){level=1;}
 if(band==1u){level=max(level,1);}
 return level;
}
// A PLANE REGION'S DARK-BOUNDARY BAND. The source's plane regions and their
// boundary edges are compiled once per source revision (composition.ts
// compileToneRegions, binding 15) in the source frame. The node's own region
// is the one whose plane holds the node's source point (an exact address
// lookup by plane, no choice). Its DARK edges, each a real turn of the form
// (the face across the edge turned 30 degrees or more from the plane, or no
// face: a mesh's small bends are one surface, not a boundary), are:
//  (a) where the plane meets a face turned further from the light (a box's
//      shadow side, the soffit under an eave);
//  (b) its LEE edges: edges on the side of the plane away from the light,
//      where the form turns away (the across face's normal, laid in the
//      plane, points against the light laid in the plane). This is the
//      plane's own shadow edge when no neighbour is darker: the darkest face
//      of a box is densest along its lee edge (the wall's foot under a high
//      light, the fog target's left wall dense at the floor), as a round is
//      densest at its shadow-side rim (curvedLevel);
//  (c) its open edges below the node (its foot, its contact with what it
//      stands on).
// The band is counted in LATTICE CELLS from the dark edge (the node interval
// of the surface lattice, measured in the world on the node's own row:
// cellStep), never in a fraction of the plane: a plane only slightly turned
// from the light (n·L .15-.45) keeps one cell along its dark edge, a plane
// turned away (n·L under .15) two, one turned well away (n·L under -.1)
// three; the first cell holds the plane's level and each cell further one
// level less, members erased (skill section 8 rules 1-2: a band of two
// alternating glyphs, lightened by erasing members). Past that the plane is
// paper, however large it is. A cast shadow on a face toward the light counts
// as n·L 0 for its turn band (a band along the shadow side) and adds its own
// band along the shadow's edge (shadowEdgeLevel), never a field.
// darkEdgeDistance returns the world distance to the nearest dark edge, 1e9
// when the region has none, -1 when the source has no region table or no
// region holds the point (then planeFallback below).
//
// THE NODE'S REGION IN O(1). compileRegionTable (ToneField.planeTable) hashes
// every region once per source revision under (source, the octahedral bin of
// its plane normal on a 16 by 16 grid, the bin of its plane offset at the
// source's own offset step), into every bin its exact-match tolerance reaches,
// under both signs of its plane. A node computes its own key from its source
// point and normal and reads at most REGION_PROBES slots of that table; the
// exact plane test (below) confirms each candidate. Static cost per unit:
// one source binary search (log2 of the resident sources), REGION_PROBES (8)
// slot reads and their exact plane tests, then one segment distance per
// listed turn edge of its region (TONE_REGION_EDGES, 24, at most; edges whose
// across face turns under 30 degrees are not listed: they are never a
// boundary). Only a node whose plane can carry tone asks (planeCarries).
@group(0) @binding(15) var<storage,read> planes:array<vec4f>;
// Source entry: first region, region count, the source's offset bin step.
fn planeSource(domain:u32)->vec3u{
 let count=bitcast<u32>(planes[0].x);var lo=0u;var hi=count;
 // Exact source address lookup (sorted by domain); nothing is chosen.
 while(lo<hi){let mid=(lo+hi)/2u;if(bitcast<u32>(planes[1u+mid].x)<domain){lo=mid+1u;}else{hi=mid;}}
 if(lo<count){let e=planes[1u+lo];if(bitcast<u32>(e.x)==domain){return vec3u(bitcast<u32>(e.y),bitcast<u32>(e.z),bitcast<u32>(e.w));}}
 return vec3u(0u);
}
// The octahedral bin of a unit normal (16 by 16), as ToneField.planeTable.
fn octBin(n:vec3f)->vec2i{
 let a=n/max(abs(n.x)+abs(n.y)+abs(n.z),1e-20);
 var e=a.xy;if(a.z<0.0){e=(vec2f(1.0)-abs(a.yx))*select(vec2f(-1.0),vec2f(1.0),a.xy>=vec2f(0.0));}
 return clamp(vec2i(floor((e*.5+vec2f(.5))*${REGION_OCT}.0)),vec2i(0),vec2i(${REGION_OCT-1}));
}
fn regionKey(domain:u32,bin:vec2i,offset:i32)->u32{
 var h=(domain*0x9e3779b9u)^(bitcast<u32>(bin.x+bin.y*${REGION_OCT})*0x85ebca6bu)^(bitcast<u32>(offset)*0xc2b2ae35u);
 h^=h>>15u;h*=0x2c1b3c6du;h^=h>>12u;
 return h|1u;
}
// The region holding source point ps with source normal ns: O(1) probes.
fn regionOf(domain:u32,src:vec3u,ps:vec3f,ns:vec3f)->u32{
 let tableBase=bitcast<u32>(planes[0].w);let tableSize=bitcast<u32>(planes[tableBase].x);
 if(tableSize==0u){return 0xffffffffu;}
 let step=bitcast<f32>(src.z);if(!(step>0.0)){return 0xffffffffu;}
 let key=regionKey(domain,octBin(ns),i32(floor(dot(ns,ps)/step)));
 let regionBase=bitcast<u32>(planes[0].y);
 for(var k=0u;k<${REGION_PROBES}u;k++){
  // Two exact (key, region) slots per vec4; no padding per hash entry.
  let slot=(key+k)&(tableSize-1u);let pair=(slot&1u)*2u;
  let e=bitcast<vec4u>(planes[tableBase+1u+slot/2u]);
  let stored=e[pair];if(stored==0u){return 0xffffffffu;}
  if(stored!=key){continue;}
  let r=e[pair+1u];
  let h=planes[regionBase+r*2u];let g=planes[regionBase+r*2u+1u];
  if(abs(dot(h.xyz,ns))>=.995&&abs(dot(h.xyz,ps)-h.w)<=.02*g.z+1e-4){return r;}
 }
 return 0xffffffffu;
}
fn darkEdgeDistance(P:vec3f,n:vec3f,o:vec2u,nl:f32,toLight:vec3f)->vec2f{
 let domain=o.x&0x5fffffffu;
 let src=planeSource(domain);if(src.y==0u){return vec2f(-1.0);}
 var M=mat4x4f(1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1);var S=M;
 let attachment=attachmentFor(o);
 if(attachment!=0xffffffffu){M=attachments[attachment].worldFromSource;S=attachments[attachment].sourceFromWorld;}
 let ps4=S*vec4f(P,1.0);let ps=ps4.xyz/ps4.w;
 let ns=normalize((transpose(M)*vec4f(n,0.0)).xyz);
 // The light laid in the plane: the direction across the plane toward it.
 let lp=toLight-n*dot(toLight,n);let lpl=length(lp);
 let regionBase=bitcast<u32>(planes[0].y);let edgeBase=bitcast<u32>(planes[0].z);
 let r=regionOf(domain,src,ps,ns);
 if(r==0xffffffffu){return vec2f(-1.0);}
 if(true){
  let h=planes[regionBase+r*2u];let g=planes[regionBase+r*2u+1u];
  var best=1e9;let first=u32(g.x);let count=min(u32(g.y),${TONE_REGION_EDGES}u);
  for(var e=first;e<first+count;e++){
   let k=edgeBase+e*3u;
   let a4=M*vec4f(planes[k].xyz,1.0);let b4=M*vec4f(planes[k+1u].xyz,1.0);let a=a4.xyz/a4.w;let b=b4.xyz/b4.w;
   let m=planes[k+2u];
   var dark=false;
   // An open edge is a foot only on a standing plane (a wall on what it
   // stands on); a roof's open eave is not a dark boundary.
   if(m.w>.5){dark=abs(n.y)<.3&&max(a.y,b.y)<P.y;}
   else{
    let across=normalize((transpose(S)*vec4f(m.xyz,0.0)).xyz);
    // A real turn of the form only (30 degrees or more).
    if(dot(across,n)<.87){
     // (a) a face turned further from the light;
     dark=dot(across,toLight)<nl-.05;
     // (b) a lee edge: the turn lies on the side away from the light.
     let side=across-n*dot(across,n);
     if(!dark){dark=lpl<.2||dot(normalize(side),lp/lpl)< -.3;}
    }
   }
   if(!dark){continue;}
   let ab=b-a;let t=clamp(dot(P-a,ab)/max(dot(ab,ab),1e-12),0.0,1.0);
   best=min(best,length(P-(a+ab*t)));
  }
  // The region's radius in the world, in the plane's own scale (two in-plane
  // source axes: a box scaled 2.5 by 5 measures its face by those two).
  let u=normalize(cross(h.xyz,select(vec3f(1,0,0),vec3f(0,1,0),abs(h.x)>.7)));let v=cross(h.xyz,u);
  let scale=sqrt(length((M*vec4f(u,0.0)).xyz)*length((M*vec4f(v,0.0)).xyz));
  return vec2f(best,g.z*scale);
 }
 return vec2f(-1.0);
}
// The plane can carry tone at all: turned from the light (n·L under .45), or
// a face inside a cast shadow (band 1). A lit face is paper and never asks for
// its region.
fn planeCarries(n:vec3f,nl:f32,band:u32)->bool{return nl<.45||band==1u;}
// One lattice cell of the node's row, measured in the world on the surface
// (the node interval x0..x1 met with the node's plane).
fn cellStep(P:vec3f,n:vec3f,y:f32,L:Lattice)->f32{
 let plane=vec4f(n,-dot(n,P));
 let A=meet(P,plane,rowPlane(y),colPlane(L.x0));let B=meet(P,plane,rowPlane(y),colPlane(L.x1));
 if(A.w==0.0||B.w==0.0){return 0.0;}
 return length(B.xyz-A.xyz);
}
// THE REGION FIELD (tone lane 2026-11-12/13; replaces the turn band that
// kept only one to three cells along the dark edge and left the rest of a

// field over its whole region. Its n·L in the fixed bands gives the band's
// level: n·L .15 to .45 level 1 (spaced ". : ."), -.1 to .15 level 2
// (".:.:.:"), under -.1 level 3 (":;:;:;"); a face inside a cast shadow counts
// as n·L .1. The field GRADES FROM ITS DARK EDGE, counted in field cells of the
// surface lattice (cells): within one cell one level denser (capped at 3),
// within three cells the band's level, beyond that one level lighter (never
// under 1): densest toward the shadow edge, thinning toward the light (agreed
// design, Tone; the fog target's wall ":;:;:" at the dark side to ". : ."
// toward the light passage). A lit face (n·L .45 and over) is paper.
fn planeLevel(cells:f32,nl:f32,band:u32)->i32{
 let lit=select(nl,min(nl,.1),band==1u);
 if(lit>=.45||!(cells>=0.0)){return 0;}
 let base=select(select(1,2,lit<.15),3,lit< -.1);
 if(cells<1.0){return min(base+1,3);}
 if(cells<3.0){return base;}
 return max(base-1,1);
}
// A CAST SHADOW ON A FACE TOWARD THE LIGHT: a band along the shadow's edge on
// the receiving face (shadowEdgeLevel: the caster's map read at fixed points
// on the face's own plane one and two cells away), never a field; a face
// wholly in shadow is paper.
fn shadowBand(P:vec3f,n:vec3f,nl:f32,band:u32,y:f32,L:Lattice)->i32{
 if(band!=1u||nl<.15){return 0;}
 return shadowEdgeLevel(P,n,y,L);
}
// A plane with no region table (no source frame for it, or a point no
// compiled region holds) has no dark boundary to band along, so it carries
// no turn band at all: never a filled face, whatever its n·L. Only the edge
// of a cast shadow on it is drawn (shadowBand).
// A LEVEL FACE IN A CAST SHADOW carries a band only along the shadow's edge:
// the caster's own map is read at fixed points on the face's own plane, one
// and two lattice cells away (world points fixed on the surface, not page
// probes). Lit within one cell: level 2; within two: level 1; a face
// wholly in shadow (a roofed room's floor) stays paper. The fog target's
// tails under each block; the tonemapping target's nearly empty ground.
fn shadowEdgeLevel(P:vec3f,n:vec3f,y:f32,L:Lattice)->i32{
 if(p.ownership.y==0u||sun.enabled==0.0){return 0;}
 let plane=vec4f(n,-dot(n,P));
 let A=meet(P,plane,rowPlane(y),colPlane(L.x0));let B=meet(P,plane,rowPlane(y),colPlane(L.x1));
 if(A.w==0.0||B.w==0.0){return 0;}
 let step=length(B.xyz-A.xyz);if(!(step>0.0)){return 0;}
 let t1=normalize(B.xyz-A.xyz);let t2=normalize(cross(n,t1));
 for(var k=1;k<=2;k++){
  for(var d=0;d<4;d++){
   let dir=select(select(select(-t2,t2,d==2),-t1,d==1),t1,d==0);
   if(castVisibility(P+dir*step*f32(k),n)>.6){return 3-k;}
  }
 }
 return 0;
}
// THE WORD TABLE for the classes that keep whole words (one lookup per unit):
//   foliage    ; ; ;  (level 1, every 2nd node)   ゞゞゞ (2)   彡::彡 (3)   (爻爻爻 never: labelled bad)
//   slope rock : : :  (1)  ,:,:,: (2)  ;:;:;: (3)  ;;:;; where n·L < -.25
//   slope turf ﾞ;ﾞ;ﾞ; (1, 2)  ヾ;:ﾐ: (3)  ﾐ;ﾐ;ﾐ; where n·L < -.25
//   slope bare .:.:.: (1, 2)  ::: (3)
// Planes, rounds, other facet-built forms and floors take THE BAND (bandUnits).
fn wordOf(kind:u32,level:i32,foliage:bool,nl:f32,cliff:bool,grass:bool)->u32{
 if(level<=0){return 0u;}
 if(foliage){return select(select(${FOLIAGE.core}u,${FOLIAGE.mid}u,level==2),${FOLIAGE.fringe}u,level==1);}
 let deep=level==3&&nl< -.25;
 if(cliff){
  if(deep){return ${ROCK.deep}u;}
  return select(select(${ROCK.core}u,${ROCK.mid}u,level==2),${ROCK.fringe}u,level==1);
 }
 if(grass){
  if(deep){return ${GRASS.deep}u;}
  return select(${GRASS.mid}u,${GRASS.core}u,level==3);
 }
 return fieldWord(level);
}
fn slotOf(kind:u32,foliage:bool,organic:bool)->f32{
 if(kind==KIND_SLOPE){return ${SLOPE_FIELD_SLOT}.0;}
 if(foliage){return ${FIELD_SLOT}.0;}
 if(kind==KIND_PLANE||(kind==KIND_FACET&&!organic)){return ${FIELD_SLOT}.0;}
 return ${BAND_SLOT}.0;
}
// THE BAND (skill section 8 rule 1: two glyphs alternating, ":" and ";";
// rule 2: lightened by erasing members, never by a lighter glyph). The
// node's cell holds members at one even pitch, one member and a half space
// (8 px): every member at level 3, every 2nd at level 2, every 4th at level
// 1. Only a round's rim wedge at level 3 abuts its members (3 px, the
// tonemapping target's ".:;:;:;" rows against the rim); a flat face never
// carries an abutted block (it reads as a stamp). The far view erases every
// 2nd kept member again. A member's glyph and its erasure follow its fixed
// index on the lattice (node, member, lattice row), so the band stays on the
// surface. Each member is admitted by the fixed clearance rule.
fn bandUnits(row:u32,y:f32,s:S,L:Lattice,level:i32,band:u32,tier:u32,mode:u32,facets:bool,abut:bool){
 let pitch=select(8.0,3.0,level>=3&&abut);
 let cell=L.x1-L.x0;let count=max(floor(cell/pitch),1.0);let step=cell/count;let members=i32(count);
 var keep=1;if(level==2&&pitch>3.0){keep=2;}if(level<=1){keep=select(2,4,pitch>3.0);}if(tier==2u){keep*=2;}
 // Alternate lattice rows shift the kept members by half the erase step
 // (ずらし): a brick bond, never members stacked in columns.
 let index=L.node*members+select(0,keep/2,L.odd);
 let next=round(L.x1);let far=tier>=1u;
 if(facets){loadRow(row);}
 for(var j=0;j<members&&j<48;j++){
  let g=index+j;
  if((g%keep+keep)%keep!=0){continue;}
  let pen=round(L.x0+step*f32(j));
  if(pen+${advanceOf(BAND_COLON)}.0>next){break;}
  if(facets&&edgeTaken(pen,pen+${advanceOf(BAND_COLON)}.0)){continue;}
  let passage=select(${BAND_COLON}u,${BAND_SEMI}u,((j+i32(L.odd))&1)!=0);
  if(unitClear(pen,row,y,${advanceOf(BAND_COLON)}.0,s,mode,far)){emitPassage(row,y,pen,passage,band,s.o,latticeKey(L,u32(j),passage,band),s.d);}
  else{atomicAdd(&out.flags,256u);}
 }
}
// THE FIELD'S WORD BY LEVEL (one lookup; see FIELD_SLOT): the short word
// (a remainder or a fallback) and the row unit, a or b by lattice row.
fn fieldWord(level:i32)->u32{
 if(level>=3){return ${DENSE_B}u;}
 if(level==2){return ${WALL_MID}u;}
 return ${FRINGE}u;
}
// A crown's row unit by level, a or b by lattice row (CROWN_ROW_PASSAGES).
fn crownRow(level:i32,odd:bool)->u32{
 if(level>=3){return select(${CROWN_DARK_A}u,${CROWN_DARK_B}u,odd);}
 if(level==2){return select(${CROWN_MID_A}u,${CROWN_MID_B}u,odd);}
 return select(${CROWN_LIGHT_A}u,${CROWN_LIGHT_B}u,odd);
}
fn fieldRow(level:i32,odd:bool)->u32{
 if(level>=3){return select(${ROW_DARK_A}u,${ROW_DARK_B}u,odd);}
 if(level==2){return select(${ROW_MID_A}u,${ROW_MID_B}u,odd);}
 return select(${ROW_LIGHT_A}u,${ROW_LIGHT_B}u,odd);
}
// The band's single members over [x, end) at fixed pens (abutted; every 8 px
// at level 1), ":" and ";" alternating, each through the clearance rule.
// tone2 c2: MEMBER_COUNT[kind*161+W], compiled from the member advance (3 px), the
// member pitch (kind 0: abutted 3 px; kind 1: 8 px after a half-space lead) and
// the legal native gap set (sums of 5 and 11 px; every gap from 40): for W, the
// width from the previous ink's end to the next pen, the largest member count
// that fits and leaves a legal gap to that pen (toneplace drops the word after
// an unspellable gap), else 0. One read; W over 160 keeps all that fit.
const MEMBER_COUNT=array<u32,322>(0u,0u,0u,1u,0u,0u,2u,0u,1u,3u,0u,2u,4u,1u,3u,5u,2u,4u,6u,3u,5u,7u,4u,6u,8u,5u,7u,9u,6u,8u,10u,7u,9u,11u,8u,10u,12u,9u,11u,8u,10u,12u,9u,11u,11u,10u,12u,12u,11u,11u,10u,12u,12u,11u,11u,11u,12u,12u,12u,11u,11u,12u,12u,12u,11u,11u,12u,12u,12u,12u,11u,12u,12u,12u,12u,11u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,1u,0u,0u,0u,0u,1u,1u,0u,2u,0u,1u,1u,0u,2u,2u,1u,3u,1u,2u,2u,1u,3u,3u,2u,4u,2u,3u,3u,2u,4u,4u,3u,5u,3u,4u,4u,3u,5u,5u,4u,6u,4u,5u,5u,4u,6u,6u,5u,7u,5u,6u,6u,5u,7u,7u,6u,8u,6u,7u,7u,6u,8u,8u,7u,9u,7u,8u,8u,7u,9u,9u,8u,10u,8u,9u,9u,8u,10u,10u,9u,11u,9u,10u,10u,9u,11u,11u,10u,12u,10u,11u,11u,10u,12u,12u,11u,11u,11u,12u,12u,11u,11u,11u,12u,12u,12u,11u,11u,12u,12u,12u,11u,11u,12u,12u,12u,12u,11u,12u,12u,12u,12u,11u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u,12u);
fn fieldMembers(row:u32,y:f32,s:S,L:Lattice,x0:f32,end:f32,level:i32,band:u32,mode:u32,far:bool,ordinal:u32){
 let mp=select(${advanceOf(BAND_COLON)}.0,8.0,level<=1);
 let lead=select(0.0,${passages.gap}.0,level<=1);let xi=round(x0);
 let W=i32(round(end+lead))-i32(xi-lead);
 var count=0u;
 if(W>160){count=12u;}else if(W>0){count=MEMBER_COUNT[u32(select(0,161,level<=1)+W)];}
 for(var m=0;m<12;m++){
  let mpen=xi+f32(m)*mp;if(u32(m)>=count||mpen+${advanceOf(BAND_COLON)}.0>end){break;}
  let member=select(${BAND_COLON}u,${BAND_SEMI}u,((m+i32(L.odd))&1)!=0);
  if(unitClear(mpen,row,y,${advanceOf(BAND_COLON)}.0,s,mode,far)){emitPassage(row,y,mpen,member,band,s.o,latticeKey(L,32u+(ordinal*12u+u32(m))%32u,member,band),s.d);}
  else{atomicAdd(&out.flags,256u);}
 }
}
// The short word over [x0, end) at its pitch, each failing word split into
// members (D1 alternative 4).
// tone2 c3: MEMBER_LEGAL[kind*161+W]: 1 when some member count (none
// included) leaves a legal native gap over width W (same tables as c2).
const MEMBER_LEGAL=array<u32,322>(1u,0u,0u,1u,0u,1u,1u,0u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,0u,0u,0u,0u,1u,0u,0u,0u,0u,1u,1u,0u,1u,0u,1u,1u,0u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u);
fn fieldShorts(row:u32,y:f32,s:S,L:Lattice,x0:f32,end:f32,sw:u32,level:i32,band:u32,mode:u32,far:bool,ordinal:u32){
 let advance=f32(passageHeader(sw).z);let pitch=advance+select(0.0,${passages.gap}.0,level<=1);
 // tone2 c3: the most shorts that fit, less any whose remaining width the
 // members cannot close with a legal gap to the next pen (MEMBER_LEGAL).
 let lead=select(0.0,${passages.gap}.0,level<=1);let xi=round(x0);let next=i32(round(end+lead));
 var fit=0u;for(var k=0u;k<4u;k++){if(xi+f32(k)*pitch+advance>end){break;}fit=k+1u;}
 var shorts=fit;
 for(var t=0u;t<5u;t++){
  let W=next-i32(xi+f32(shorts)*pitch-lead);
  if(W>160||(W>=0&&MEMBER_LEGAL[u32(select(0,161,level<=1)+W)]!=0u)){break;}
  if(shorts==0u){shorts=fit;break;}
  shorts--;
 }
 var x=x0;
 for(var k=0u;k<4u;k++){
  let pen=round(x);if(k>=shorts||pen+advance>end){break;}
  if(unitClear(pen,row,y,advance,s,mode,far)){emitPassage(row,y,pen,sw,band,s.o,latticeKey(L,16u+(ordinal*4u+k)%16u,sw,band),s.d);}
  else{fieldMembers(row,y,s,L,pen,pen+advance,level,band,mode,far,ordinal*4u+k);}
  x+=pitch;
 }
 fieldMembers(row,y,s,L,x,end,level,band,mode,far,ordinal*4u+7u);
}
// THE REGION FIELD'S UNITS: the region's row unit repeated abutted from the
// node's own X at one pitch (its advance; plus one half space at level 1, the
// spaced light band), one phase for the whole cell. The cell's remainder (under
// one unit) takes the short word, then single members, at fixed pens, so the
// row runs on to the next node without a channel. Alternate lattice rows take
// the level's second unit (ずらし, from the surface's cross-row cell, never the
// page row). The far tier erases every second unit (skill section 8 rule 2).
// Each unit passes the fixed clearance rule at its one pen; a unit that cannot
// be placed whole takes, in fixed order, the short words over its span, then
// the members (D1 alternative 4). No other pen is ever tried. A slope passes
// its material word as both unit and short word (row 0).
// grade: x, y the dark-edge distance in field cells at the node and at the
// cell's far end, z n·L, w the air; w < 0: one level for the whole cell. The far
// tier caps the level at 2 and erases nothing (no sparse tone).
// THE WORKGROUP JOB QUEUE (tone cost, 2026-11-13). A holder decides; the
// row segment's 64 threads execute. Job kind 0: one unit (a = pen, b = its
// advance); kind 1: the cell's remainder (a = start, b = end).
struct ToneHolder {row:u32,y:f32,s:S,L:Lattice,band:u32,mode:u32,far:u32,depthLine:vec4f}
// tone2 c1: the holder's plane depth line along its row (x: the holder's page X,
// y: its depth (identifies the holder's surface), z: depth per page px, w: the
// line's depth at x; w 0 when none). unitClear accepts a sample on that line.
var<private> clearRef:vec4f;
struct ToneJob {a:f32,b:f32,unit:u32,sw:u32,lv:i32,ordinal:u32,holder:u32,kind:u32}
const TONE_HOLDERS=16u;const TONE_JOBS=192u;
var<workgroup> toneHolders:array<ToneHolder,16>;
var<workgroup> toneHolderCount:atomic<u32>;
var<workgroup> toneJobs:array<ToneJob,192>;
var<workgroup> toneJobCount:atomic<u32>;
fn holderSlot(row:u32,y:f32,s:S,L:Lattice,band:u32,mode:u32,far:bool)->u32{
 let h=atomicAdd(&toneHolderCount,1u);
 if(h<TONE_HOLDERS){toneHolders[h]=ToneHolder(row,y,s,L,band,mode,u32(far),clearRef);}
 return h;
}
// One unit, exactly the former inline body: the clearance rule at its one pen,
// else the short words over its span, else the members (D1 alternative 4).
fn doUnit(row:u32,y:f32,s:S,L:Lattice,pen:f32,advance:f32,unit:u32,sw:u32,lv:i32,band:u32,mode:u32,far:bool,j:u32){
 if(unitClear(pen,row,y,advance,s,mode,far)){emitPassage(row,y,pen,unit,band,s.o,latticeKey(L,j,unit,band),s.d);return;}
 if(unit!=sw){fieldShorts(row,y,s,L,pen,pen+advance,sw,lv,band,mode,far,j);}
 else{fieldMembers(row,y,s,L,pen,pen+advance,lv,band,mode,far,j);}
}
fn queueJob(h:u32,job:ToneJob)->bool{
 if(h>=TONE_HOLDERS){return false;}
 let k=atomicAdd(&toneJobCount,1u);if(k>=TONE_JOBS){return false;}
 toneJobs[k]=job;return true;
}
fn runToneJobs(li:u32){
 let n=min(atomicLoad(&toneJobCount),TONE_JOBS);
 for(var k=li;k<n;k+=64u){
  let J=toneJobs[k];let H=toneHolders[J.holder];clearRef=H.depthLine;
  if(J.kind==0u){doUnit(H.row,H.y,H.s,H.L,J.a,J.b,J.unit,J.sw,J.lv,H.band,H.mode,H.far!=0u,J.ordinal);}
  else{fieldShorts(H.row,H.y,H.s,H.L,J.a,J.b,J.sw,J.lv,H.band,H.mode,H.far!=0u,J.ordinal);}
 }
}
fn fieldUnits(row:u32,y:f32,s:S,L:Lattice,sw0:u32,level0:i32,band:u32,tier:u32,mode:u32,facets:bool,unit0:u32,fromX:f32,grade:vec4f,shade:i32){
 if(sw0==0u){return;}
 let far=tier>=1u;
 if(facets){loadRow(row);}
 let h=holderSlot(row,y,s,L,band,mode,far);
 var x=L.x0;var sw=sw0;var lv=level0;
 for(var j=0u;j<16u;j++){
  if(grade.w>=0.0){
   let f=clamp((x-L.x0)/max(L.x1-L.x0,1.0),0.0,1.0);
   lv=airLevel(max(planeLevel(mix(grade.x,grade.y,f),grade.z,band),shade),grade.w);
   if(tier>=1u){lv=min(lv,2);}
   if(lv<=0){x+=${advanceOf(ROW_DARK_A)}.0;continue;}
   sw=fieldWord(lv);
  }
  // Graded planes take the wall row of each unit's own level; an ungraded
  // field (a crown) keeps the unit it was given.
  let unit=select(sw,select(unit0,fieldRow(lv,L.odd),grade.w>=0.0),unit0!=0u);
  let advance=f32(passageHeader(unit).z);
  let pitch=advance+select(0.0,${passages.gap}.0,lv<=1);
  if(x+advance>L.x1){break;}
  let pen=round(x);x+=pitch;
  if(pen<fromX||(facets&&edgeTaken(pen,pen+advance))){continue;}
  if(!queueJob(h,ToneJob(pen,advance,unit,sw,lv,j,h,0u))){doUnit(row,y,s,L,pen,advance,unit,sw,lv,band,mode,far,j);}
 }
 if(lv<=0){return;}
 let rest=max(x,fromX);let end=round(L.x1)-select(0.0,${passages.gap}.0,lv<=1);
 if(rest<end&&!(facets&&edgeTaken(rest,end))){if(!queueJob(h,ToneJob(rest,end,0u,sw,lv,15u,h,1u))){fieldShorts(row,y,s,L,rest,end,sw,lv,band,mode,far,15u);}}
}
// THE ROUND'S WEDGE (tone2, 2026-11-13; replaces for the rounded class the
// lattice wedge of roundLevel and bandUnits, which kept a thin dotted rim and
// left the shaded half paper). The tonemapping hand target draws, inside each
// sphere's shadow quarter, ONE abutted run per row, laid from the shadow-side
// rim inward and longer toward the lower rim: ". :" / ":.:;:" / ".:;:;:;" /
// ".:;:;:;:;" (author.py S57, S63), with paper kept inside the outline. Here the
// rim pixel of the round's span on a native row holds that row's run: from
// ROUND_RIM_GAP inside the rim, units abut toward the light, each unit's word
// by the n.L of its own surface point (roundFieldLevel): grazing and shaded
// ".:;:;:;:;", under -.25 ".:;:;:;", under 0 ":.:;:", under .15 ". :". The run
// ends where the surface turns lit, where it leaves the form, or past the form's
// middle (the unit's normal turned more than 90 degrees from the rim's), so two
// rims never lay over each other. A row's shaded part shorter than its unit takes
// the shortest word that fits; a unit a drawn line blocks splits into its band
// members (D1 alternative 4). The middle tier keeps at most ":.:;:".
// r3: the levels follow the scene light's own bands (lightBand: .7, .4, .15),
// so the wedge covers the shaded crescent the mesh shows, not only its rim.
fn roundFieldLevel(nl:f32,nv:f32,band:u32)->i32{
 var level=0;
 // r6: the wedge starts near the terminator (n.L .3), not across the lit side.
 if(nl<0.0){level=3;}else if(nl<.15){level=2;}else if(nl<.3){level=1;}
 if(nl< -.1&&nv<.45){level=4;}
 if(band==1u){level=max(level,2);}
 return level;
}
// A surface's own value as a band where the scene has no key light: the
// luminance its material shows (base colour under ambient and point lights,
// plus emissive). The darkest value takes the core word, never the deep one.
fn valueLevel(value:f32)->i32{
 if(value<.15){return 3;}
 if(value<.35){return 2;}
 if(value<.6){return 1;}
 return 0;
}
fn roundRow(level:i32)->u32{
 if(level>=4){return ${ROUND_DEEP}u;}
 if(level==3){return ${ROUND_CORE}u;}
 if(level==2){return ${ROUND_MID}u;}
 return ${ROUND_LIGHT}u;
}
fn rimKey(x:i32,row:u32,k:u32,side:u32,passage:u32,band:u32)->vec4u{
 return vec4u(bitcast<u32>(x),row,k|(side<<8u),0x80000000u|((passage&255u)<<3u)|(band&7u));
}
fn roundRim(row:u32,y:f32,x:i32,q:vec2f,s:S,P:vec3f){
 let step=colStep();
 let right=!sameRaw(s,toField(vec2f(f32(x+step)+.5,y)),false);
 let left=!sameRaw(s,toField(vec2f(f32(x-step)+.5,y)),false);
 if(right==left){return;}
 // NO KEY LIGHT (tone2 m2): the band comes from the surface's own value
 // (valueLevel); normalize(0) is selected away and never decides a band.
 let direction=lightDirection();let keyed=dot(direction,direction)>=.000001;
 let toLight=select(vec3f(0.0,1.0,0.0),normalize(select(vec3f(0.0,1.0,0.0),direction,keyed)),keyed);
 let air=transmittance(q,P);if(air<.1){return;}
 let tier=toneTier(q,s.d,s.o);let far=tier>=1u;
 let dir=select(1.0,-1.0,right);let side=select(0u,1u,right);
 // The rim pixel's edge: the far side of its column on the rim side.
 let rim=select(f32(x),f32(x+step),right);
 // r4: THE RIM'S NORMAL. At the silhouette the field's normal grazes the view
 // and at() may turn it to face the viewer the wrong way (r3 stopped every run
 // after one unit and started runs on lit rims). The rim normal is read a gap
 // inside the rim, where the surface faces the viewer, with its view component
 // removed: the silhouette's own outward normal.
 let inner=at(toField(vec2f(rim+dir*${ROUND_RIM_GAP}.0,y)));
 if(!inner.on||any(inner.o!=s.o)){return;}
 let lateral=inner.n-inner.v*dot(inner.n,inner.v);if(dot(lateral,lateral)<1e-6){return;}
 let rimNormal=normalize(lateral);
 // r7: a round's wedge reads n.L only, never the sun map's cast-shadow band:
 // on a large round the map's own self-shadow past the terminator marked the
 // lit lower half as cast shadow and filled it (tone2-h4 tonemapping sphere).
 let band=0u;
 // r3: only the SHADOW-SIDE rim holds a run (its surface turned from the light,
 // n.L under -.1, or a cast shadow on it); the lit rim stays paper.
 if(keyed&&dot(rimNormal,toLight)>= -.1&&band!=1u){return;}
 // dir is the walk (toward the light); at0 is the next unit's rim-side end.
 var at0=rim+dir*${ROUND_RIM_GAP}.0;
 for(var k=0u;k<48u;k++){
  let probe=at(toField(vec2f(at0+dir*3.0,y)));
  if(!probe.on||any(probe.o!=s.o)||dot(probe.n,rimNormal)<0.0){return;}
  var lv=0;
  if(keyed){lv=roundFieldLevel(dot(probe.n,toLight),abs(dot(probe.n,probe.v)),band);}
  else{lv=valueLevel(shown(toField(vec2f(at0+dir*3.0,y))));}
  lv=airLevel(lv,air);
  if(far){lv=min(lv,2);}
  if(lv<=0){return;}
  var unit=roundRow(lv);var advance=f32(passageHeader(unit).z);
  // The shaded part ahead (the next unit's far end still shaded and on the
  // form), else the shortest word: a short row keeps its one word.
  let tail=at(toField(vec2f(at0+dir*(advance-1.0),y)));
  if(!tail.on||any(tail.o!=s.o)){unit=${ROUND_LIGHT}u;advance=f32(passageHeader(unit).z);}
  let pen=round(select(at0,at0-advance,dir<0.0));
  // r5: clearance against the unit's own surface point (probe): a sphere's
  // depth changes across the run, the rim's depth would refuse every inner unit.
  if(unitClear(pen,row,y,advance,probe,MODE_ROUND,far)){emitPassage(row,y,pen,unit,band,s.o,rimKey(x,row,k,side,unit,band),probe.d);}
  else{
   for(var m=0;m<12;m++){
    let mpen=pen+f32(m)*${advanceOf(BAND_COLON)}.0;if(mpen+${advanceOf(BAND_COLON)}.0>pen+advance){break;}
    let member=select(${BAND_COLON}u,${BAND_SEMI}u,(m&1)!=0);
    if(unitClear(mpen,row,y,${advanceOf(BAND_COLON)}.0,probe,MODE_ROUND,far)){emitPassage(row,y,mpen,member,band,s.o,rimKey(x,row,k|((u32(m)+1u)<<4u),side,member,band),probe.d);}
    else{atomicAdd(&out.flags,256u);}
   }
  }
  at0+=dir*advance;
 }
}
// TURF (tone2 g1): ONE constant pattern for the region, the T#3 distant-grass
// patch followed by its paper (one full space), laid from the lattice node; the
// value lightens by erasing patches of that pattern, never by a denser glyph
// (skill section 8 rule 2, D6): shaded (level 3) keeps every patch, level 2
// every 2nd, level 1 every 4th, by the patch's fixed index on the lattice (node
// and ordinal), so a kept patch stays on its ground. Alternate lattice rows
// begin half a pitch on (zurashi). A patch the clearance rule refuses is paper.
// tone2 g2: a turf patch's fixed erasure, by the lattice address hash of
// intervalKept plus the patch ordinal (no diagonal from a modular index).
fn patchKept(L:Lattice,j:u32,n:u32)->bool{
 let u=canonical(L.node,L.level);let v=canonical(L.vcell,L.vlevel);
 var h=(bitcast<u32>(u.x)*0x9e3779b9u)^(bitcast<u32>(u.y+64)*0x85ebca6bu)^(bitcast<u32>(v.x)*0xc2b2ae35u)^(bitcast<u32>(v.y+64)*0x27d4eb2fu)^((L.axis+1u)*0x165667b1u)^(bitcast<u32>(L.wcell)*0x61c88647u)^((j+1u)*0x7feb352du);
 h^=h>>15u;h*=0x2c1b3c6du;h^=h>>12u;h*=0x297a2d39u;h^=h>>15u;
 return h%max(n,1u)==0u;
}
fn grassUnits(row:u32,y:f32,s:S,L:Lattice,level:i32,band:u32,tier:u32,fromX:f32){
 let far=tier>=1u;var lv=level;if(far){lv=min(lv,2);}
 let keep=select(select(4,2,lv==2),1,lv>=3);
 let advance=f32(passageHeader(${GRASS_PATCH}u).z);let pitch=advance+${GRASS_PAPER}.0;
 var x=L.x0+select(0.0,round(pitch*.5),L.odd);
 for(var j=0;j<16;j++){
  if(x+advance>L.x1){break;}
  let pen=round(x);x+=pitch;
  if(pen<fromX||!patchKept(L,u32(j),u32(keep))){continue;}
  if(unitClear(pen,row,y,advance,s,MODE_TERRAIN,far)){emitPassage(row,y,pen,${GRASS_PATCH}u,band,s.o,latticeKey(L,u32(j),${GRASS_PATCH}u,band),s.d);}
  else{atomicAdd(&out.flags,256u);}
 }
}
// THE SEA FIELD (aa-0024 line 13): the unit by the ground's own row measure
// (groundGrade, the same measure the level ground thins by), densest at the
// horizon and opening toward the viewer; the nearest band keeps one '-' per
// cell (never blank). Units at one pitch (advance and a half space) from the
// node, alternate lattice rows half a pitch on (ずらし). Drawn lines and other
// surfaces bound each unit (the clearance rule at its one pen).
fn waterUnits(row:u32,y:f32,q:vec2f,s:S,L:Lattice,band:u32){
 let grade=groundGrade(groundRows(floor(q)+vec2f(.5),s.d));
 var unit=${SEA_FAR}u;if(grade>=1u){unit=${SEA_MID}u;}
 // Near bands thin by whole rows of the surface (its cross-row cell): every
 // 2nd row at grade 3, every 4th at grade 4; a kept row is a whole dashed row.
 let rows=select(select(1,2,grade>=3u),4,grade>=4u);
 if(((L.vcell%rows)+rows)%rows!=0){return;}
 let advance=f32(passageHeader(unit).z);let pitch=advance+${passages.gap}.0;
 let shift=select(0.0,pitch*.5,L.odd);
 var x=L.x0+shift;
 for(var j=0u;j<8u;j++){
  if(x+advance>L.x1+shift){break;}
  let pen=round(x);x+=pitch;
  if(unitClear(pen,row,y,advance,s,MODE_FACET,true)){emitPassage(row,y,pen,unit,band,s.o,latticeKey(L,j,unit,band),s.d);}
  else{atomicAdd(&out.flags,256u);}
 }
}
// LEVEL GROUND AND FLOORS (open ground, a terrain's level part or lit gentle
// slope, a ground or water plane, a room's floor, a lid): NEVER A FIELD (D6
// bans ground drawn as a dot row; the tonemapping target's ground holds three
// sparse ＿ intervals and its contacts, the fog target's floor the tails under
// each block and a few . . : . patches). What marks a level face:
//  CAST SHADOWS ON IT: a band along the shadow's edge, inside it
//   (shadowEdgeLevel: the caster's own map read at fixed points on the face's
//   plane), as the band members; a shadow's interior is no field (it keeps
//   only the intervals below, like the lit face);
//  SPARSE RECEDING INTERVALS: one short interval word (". ."; the sea's "-";
//   lit turf its fringe word) at a FEW lattice nodes, chosen by a fixed hash of
//   the node's canonical surface address (the same world node at every lattice
//   scale: its odd part and exponent on both axes), so a mark is fixed to its
//   ground point, appears or disappears as the camera moves, and never slides.
//   One node in 6 << grade keeps its interval (groundGrade: the rows the
//   camera's height covers there), so intervals close up toward the horizon
//   and thin to paper toward the viewer (skill section 8 rule 2: lighten by
//   erasing members);
//  CONTACTS: the line producer's contact lines, not tone.
// The 6 and its doubling are this rule's own choices, not measured values.
fn canonical(a0:i32,e0:i32)->vec2i{var a=a0;var e=e0;if(a==0){return vec2i(0);}while((a&1)==0&&e<60){a/=2;e++;}return vec2i(a,e);}
fn intervalKept(L:Lattice,n:u32)->bool{
 let u=canonical(L.node,L.level);let v=canonical(L.vcell,L.vlevel);
 var h=(bitcast<u32>(u.x)*0x9e3779b9u)^(bitcast<u32>(u.y+64)*0x85ebca6bu)^(bitcast<u32>(v.x)*0xc2b2ae35u)^(bitcast<u32>(v.y+64)*0x27d4eb2fu)^((L.axis+1u)*0x165667b1u)^(bitcast<u32>(L.wcell)*0x61c88647u);
 h^=h>>15u;h*=0x2c1b3c6du;h^=h>>12u;h*=0x297a2d39u;h^=h>>15u;
 return h%n==0u;
}
// Intervals belong to a level face larger than the page shows: a terrain or
// water surface, or an owner the page cuts or that spans depth (a floor, a
// ground plane). A lid the page holds whole is its form's top: no intervals.
fn broadLevel(o:vec2u)->bool{
 let style=styleOf(o.x);
 if(style==3u||liquidOf(o.x)){return true;}
 let form=formExtent(o);
 if(!form.found){return true;}
 return formCut(form.box,p.field.xy)||formSpansDepth(form.near,form.far,p.field.xy,p.worldFromClip);
}
// A CAST SHADOW'S REGION RUN (tone2, 2026-11-13; the shadow on a level face was
// only a band along its edge, the interior paper, so the mesh's cast shadows read
// as faint traces). The shadow on a level face is one region with one constant
// pattern (D6): the floor band of church aa[126] line 34 as the bank's ground
// units ".:.:.:" and, on alternate lattice rows, ":.:.:." (zurashi). Each native
// row of the region is ONE abutted run of whole units, held by the column where
// the shadow begins on the row (the column before it is lit or another surface)
// and laid toward the shadow's end: a unit goes down when the sun's map holds its
// far end in shadow on the same surface (the surface's own point there); a unit a
// drawn line blocks is skipped and the run goes on; the run ends where the
// shadow or the surface ends, so the region ends cleanly within one unit. Air
// lightens it; the far tier takes the light row a half space apart.
fn shadedOn(q:vec2f,t:S,s:S)->bool{
 if(!t.on||any(t.o!=s.o)||t.n.y<=0.0){return false;}
 return castVisibility(unproject(floor(q)+vec2f(.5),t.d),t.n)<.4;
}
fn shadowRun(row:u32,y:f32,x:i32,q:vec2f,s:S,P:vec3f,L:Lattice,mode:u32){
 if(p.ownership.y==0u||sun.enabled==0.0){return;}
 if(castVisibility(P,s.n)>=.4){return;}
 let before=toField(vec2f(f32(x-colStep())+.5,y));
 if(shadedOn(before,at(before),s)){return;}
 var lv=airLevel(2,transmittance(q,P));if(lv<=0){return;}
 let tier=toneTier(q,s.d,s.o);if(tier>=2u){lv=1;}
 let far=tier>=1u;let band=1u;
 let unit=select(select(${SLOPE_SPACED}u,${SHADE_B}u,L.odd),select(${ROW_LIGHT_A}u,${ROW_LIGHT_B}u,L.odd),lv<=1);
 let advance=f32(passageHeader(unit).z);let pitch=advance+select(0.0,${GAP}.0,lv<=1);
 let end=f32(rowBounds().y);
 var pen=f32(x);
 for(var k=0u;k<64u;k++){
  if(pen+advance>end){return;}
  let tq=toField(vec2f(pen+advance-1.5,y));let t=at(tq);
  if(!shadedOn(tq,t,s)){return;}
  if(unitClear(pen,row,y,advance,t,mode,far)){emitPassage(row,y,pen,unit,band,s.o,vec4u(bitcast<u32>(x),row,k,0x40000000u|((unit&255u)<<3u)|band),s.d);}
  else{atomicAdd(&out.flags,256u);}
  pen+=pitch;
 }
}
fn levelUnits(row:u32,y:f32,q:vec2f,s:S,L:Lattice,band:u32,P:vec3f,mode:u32,grass:bool){
 // tone2: inside a cast shadow the region run (shadowRun) draws the face.
 if(band==1u&&p.ownership.y!=0u&&sun.enabled!=0.0){return;}
 if(band==1u){
  let level=airLevel(shadowEdgeLevel(P,s.n,y,L),transmittance(q,P));
  if(level>0){bandUnits(row,y,s,L,level,band,0u,mode,false,false);return;}
 }
 // Inside a shadow and on lit ground alike: the surface's own intervals (a
 // roofed room's floor lies wholly in shadow and still recedes).
 if(!broadLevel(s.o)){return;}
 let grade=groundGrade(groundRows(floor(q)+vec2f(.5),s.d));
 if(grade>=4u||!intervalKept(L,6u<<grade)){return;}
 let water=liquidOf(s.o.x);
 var passage=select(${TAIL}u,${WATER_FIELD}u,water);
 if(grass&&!water){passage=${GRASS_PATCH}u;}
 let advance=f32(passageHeader(passage).z);
 let pen=round(L.x0)+select(0.0,${passages.gap}.0,L.odd);
 if(pen+advance>round(L.x1)){return;}
 if(unitClear(pen,row,y,advance,s,MODE_TERRAIN,true)){emitPassage(row,y,pen,passage,band,s.o,latticeKey(L,0u,passage,band),s.d);}
 else{atomicAdd(&out.flags,256u);}
}
// TEXTURE EDGE OF A REGION-LED FORM (a compiled field source whose boundary
// TEXTURE_EDGE_MEASURE reads as finer than a glyph and hanging: a canopy).
// One row of one form surface (same owner, no depth step) is a span. Its
// boundary on this row is spelled by one constant word per side (D6, the
// practitioner's unit rule), placed against the bound and repeated row after
// row, so the envelope is the texture meeting the sky and no outline is
// spelled (the line producer's tierSelects yields the same runs on the same
// measure: replace, never both):
//   left bound        canopy-left, two pixels inside the bound
//   right bound       canopy-right, ending two pixels inside the bound
//   top (the row above is open)        canopy-top, one half space apart
//   underside (the row below is open)  near: canopy-underside abutting, the
//                     defined underside line; middle: canopy-tip, one half
//                     space apart
//   a span too narrow for a bound word  canopy-tip
// Distance (toneTier): near also takes the bounds where the form steps in
// front of its own farther lobes; the middle view takes the outer envelope
// only; the far view takes none (the line producer keeps its connected
// outline, D9).
${TEXTURE_EDGE_MEASURE}
var<private> edgeSpans:array<vec2f,64>;
var<private> edgeCount:u32=0u;
fn edgeTaken(lo:f32,hi:f32)->bool{
 for(var i=0u;i<edgeCount;i++){if(lo<edgeSpans[i].y+${passages.gap}.0&&hi>edgeSpans[i].x-${passages.gap}.0){return true;}}
 return false;
}
// A texture-edge word's fixed clearance at its one pen (no other pen is
// tried): across the word and its half spaces, at three heights of the row,
// nothing nearer than the form or touching it (the form's own envelope: its
// ink lies on the form or on open ground beyond its silhouette, or on a
// surface a depth step behind), and no drawn line but the silhouette's own
// black D10 line, which the word replaces; facet seams are not lines.
fn edgeClear(pen:f32,row:u32,advance:f32,owner:vec2u,depth:f32)->bool{
 let size=vec2i(p.field.zw);
 let a=i32(round(pen));let b=i32(round(pen+advance));
 if(a<0||b>=size.x){return false;}
 for(var dy=3;dy<${TONE_ROW};dy+=5){
  for(var x=a;x<=b;x+=2){
   let pixel=vec2i(x,i32(row)*${TONE_ROW}+dy);
   let sample=at(toField(vec2f(pixel)+vec2f(.5)));
   let own=sample.on&&(sample.o.x&0xdfffffffu)==(owner.x&0xdfffffffu)&&sample.o.y==owner.y;
   if(sample.on&&!own&&sample.d>depth*.92){return false;}
   let mark=markAt(pixel);
   if(mark.a>.75&&!facetSeam(mark)&&max(mark.r,max(mark.g,mark.b))>.2){return false;}
  }
 }
 return true;
}
// Words are emitted left to right along the row, so a word that follows the
// last one within a unit's clearance extends its interval.
fn emitEdge(row:u32,y:f32,pen:f32,passage:u32,side:u32,k:u32)->bool{
 let advance=f32(passageHeader(passage).z);let centre=pen+advance*.5;let q=toField(vec2f(centre,y));let s=at(q);
 if(!s.on){return false;}
 if(!edgeClear(pen,row,advance,s.o,s.d)){atomicAdd(&out.flags,256u);return false;}
 if(edgeCount>0u&&pen>=edgeSpans[edgeCount-1u].x&&pen<=edgeSpans[edgeCount-1u].y+11.0){edgeSpans[edgeCount-1u].y=max(edgeSpans[edgeCount-1u].y,pen+advance);}
 else{if(edgeCount>=64u){return false;}edgeSpans[edgeCount]=vec2f(pen,pen+advance);edgeCount++;}
 let P=unproject(floor(q)+vec2f(.5),s.d);
 emitPassage(row,y,pen,passage,side,s.o,vec4u(bitcast<u32>(i32(round(dot(s.n,P)*1024.0))),k,passage,side),s.d);
 return true;
}
// What lies beyond a bound of the form: 0 the view's end, a nearer surface or
// a surface it touches; 1 nothing or another surface a depth step behind;
// 2 the form's own farther lobe.
fn edgeBeyond(q:vec2f,owner:vec2u,depth:f32)->u32{
 if(any(q<vec2f(0.0))||any(q>=p.field.xy)){return 0u;}
 let h=leafHit(q);
 if((h.x|h.y)==0u){return 1u;}
 if(bitcast<f32>(h.z)>=depth*.92){return 0u;}
 return select(1u,2u,h.x==(owner.x&0xdfffffffu)&&h.y==owner.y);
}
fn tileEdge(row:u32,y:f32,a:f32,b:f32,lo:f32,hi:f32,state:u32,tier:u32,owner:vec2u,depth:f32,k:u32)->u32{
 let line=state==2u&&tier==0u;
 let passage=select(select(${TEXTURE_EDGE.tip}u,${TEXTURE_EDGE.underside}u,line),${TEXTURE_EDGE.top}u,state==1u);
 let advance=f32(passageHeader(passage).z);let head=max(a,lo);let tail=min(b,hi);
 if(tail-head<advance||!textureEdge(toField(vec2f((a+b)*.5,y)),owner,depth)){return k;}
 let step=advance+select(${passages.gap}.0,0.0,line);
 var pen=head;var n=k;
 while(pen+advance<=tail){if(emitEdge(row,y,pen,passage,6u+state,n)){n++;}pen+=step;}
 return n;
}
fn edgeSpan(row:u32,y:f32,left:i32,right:i32,owner:vec2u,leftDepth:f32,rightDepth:f32,tier:u32){
 let up=vec2f(0.0,${TONE_ROW}.0*p.field.y/p.rect.w);
 var leafLeft=false;var leafRight=false;
 let beyondLeft=edgeBeyond(toField(vec2f(f32(left)-.5,y)),owner,leftDepth);
 if(beyondLeft==1u||(beyondLeft==2u&&tier==0u)){leafLeft=textureEdge(toField(vec2f(f32(left)+1.5,y)),owner,leftDepth);}
 let beyondRight=edgeBeyond(toField(vec2f(f32(right)+.5,y)),owner,rightDepth);
 if(beyondRight==1u||(beyondRight==2u&&tier==0u)){leafRight=textureEdge(toField(vec2f(f32(right)-1.5,y)),owner,rightDepth);}
 var lo=f32(left)+2.0;var hi=f32(right)-2.0;var k=0u;
 // THE SIDE'S OWN DIRECTION (resK-483): the left bound stepping out on the
 // next row (the form there 6 px left of this row's bound) takes ",ｲ"; the
 // right bound returning (no form 6 px inside this row's right bound on the
 // next row) takes ":ﾉ"; else the T#7 words. One fixed read per side.
 let outLeft=leafOn(leafHit(toField(vec2f(f32(left)-6.0,y))+up),owner,leftDepth);
 let inRight=!leafOn(leafHit(toField(vec2f(f32(right)-6.0,y))+up),owner,rightDepth);
 let leftWord=select(${TEXTURE_EDGE.left}u,${CANOPY_LEFT_OUT}u,outLeft);let rightWord=select(${TEXTURE_EDGE.right}u,${CANOPY_RIGHT_IN}u,inRight);
 let leftAdvance=f32(passageHeader(leftWord).z);let rightAdvance=f32(passageHeader(rightWord).z);
 if(leafLeft&&hi-lo>=leftAdvance&&emitEdge(row,y,lo,leftWord,5u,k)){lo+=leftAdvance+${passages.gap}.0;k++;}
 var rightPen=-1.0;
 if(leafRight&&hi-lo>=rightAdvance){rightPen=hi-rightAdvance;hi=rightPen-${passages.gap}.0;}
 // Top and underside runs: a run is the stretch whose row above, or below,
 // is open; it is read every third pixel and measured once, at its middle.
 var state=0u;var begin=f32(left);var depth=leftDepth;
 for(var x=left;x<right+3;x+=3){
  var now=0u;
  if(x<right){
   let q=toField(vec2f(f32(x)+.5,y));let h=leafHit(q);
   if(leafOn(h,owner,depth)){depth=bitcast<f32>(h.z);}
   let above=edgeBeyond(q-up,owner,depth);let below=edgeBeyond(q+up,owner,depth);
   now=select(select(0u,1u,above==1u||(above==2u&&tier==0u)),2u,below==1u||(below==2u&&tier==0u));
  }
  if(now==state&&f32(x)-begin<96.0&&x<right){continue;}
  if(state!=0u){k=tileEdge(row,y,begin,f32(min(x,right)),lo,hi,state,tier,owner,depth,k);}
  state=now;begin=f32(x);
 }
 if(rightPen>=0.0&&emitEdge(row,y,rightPen,rightWord,6u,k)){k++;}
 // A lobe tip narrower than a bound word still takes its one mark.
 let tip=f32(passageHeader(${TEXTURE_EDGE.tip}u).z);
 if(k==0u&&(leafLeft||leafRight)&&f32(right-left)>=tip+4.0){
  let pen=(f32(left)+f32(right)-tip)*.5;
  if(emitEdge(row,y,pen,${TEXTURE_EDGE.tip}u,5u,0u)){k++;}
 }
}
fn textureEdges(row:u32,y:f32,x0:i32,x1:i32){
 var x=x0;var known=0xffffffffu;var style=1u;
 while(x<x1){
  let q=toField(vec2f(f32(x)+.5,y));let h=leafHit(q);
  if((h.x|h.y)==0u||formParentSelected(h.xy)){x+=colStep();continue;}
  if(h.x!=known){known=h.x;style=styleOf(h.x);}
  if(style!=0u){x+=colStep();continue;}
  let leftDepth=bitcast<f32>(h.z);var depth=leftDepth;var end=x+colStep();
  while(end<x1){
   let next=leafHit(toField(vec2f(f32(end)+.5,y)));
   if(!leafOn(next,h.xy,depth)){break;}
   depth=bitcast<f32>(next.z);end+=colStep();
  }
  if(end-x>=8){
   let tier=toneTier(q,leftDepth,h.xy);
   if(tier<2u){edgeSpan(row,y,x,end,h.xy,leftDepth,depth,tier);}
  }
  x=end;
 }
}
// THE ROW'S TEXTURE-EDGE TABLE. The canopy envelope words (texture edges,
// shared with the line producer's tierSelects: replace, never both) are made
// first, one thread per native row (rowPrelude), and their spans kept in
// rowState; a facet-built unit on the same row keeps one half space clear of
// them (edgeTaken). The table holds nothing else: no crown mark, no extent.
const ROW_STATE=${ROW_STATE}u;
@group(1) @binding(2) var<storage,read_write> rowState:array<u32>;
fn saveRow(row:u32){
 let b=row*ROW_STATE;rowState[b]=edgeCount;
 for(var i=0u;i<edgeCount;i++){let a=b+2u+i*2u;rowState[a]=bitcast<u32>(edgeSpans[i].x);rowState[a+1u]=bitcast<u32>(edgeSpans[i].y);}
}
fn loadRow(row:u32){
 let b=row*ROW_STATE;edgeCount=min(rowState[b],64u);
 for(var i=0u;i<edgeCount;i++){let a=b+2u+i*2u;edgeSpans[i]=vec2f(bitcast<f32>(rowState[a]),bitcast<f32>(rowState[a+1u]));}
}
fn rowBounds()->vec2i{return vec2i(i32(ceil(p.rect.x)),i32(min(floor(p.rect.x+p.rect.z),p.field.z)));}
// THE ROW WALK, ONE WORKGROUP PER ROW (frame cost, 2026-11-12). One thread
// per row read every column of its row in turn (19 ms at 3488x2252 with the
// glyph-grid field). The row's 64 threads now read the columns in parallel:
// whether a column can start a texture-edge span (a hit whose source class is
// 0) and whether it continues the column before it (leafOn against that
// column, which carries the span's owner). One thread then walks the spans
// from those bits exactly as textureEdges does and runs edgeSpan for each, in
// the same order: the row's words and their order are unchanged.
const WALK_WORDS=256u;
var<workgroup> walkStarts:array<atomic<u32>,256>;
var<workgroup> walkContinues:array<atomic<u32>,256>;
fn startsSpan(c:u32)->bool{return (atomicLoad(&walkStarts[c/32u])&(1u<<(c%32u)))!=0u;}
fn continuesSpan(c:u32)->bool{return (atomicLoad(&walkContinues[c/32u])&(1u<<(c%32u)))!=0u;}
@compute @workgroup_size(64) fn rowWalk(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) li:u32){
 let row=group.x;
 if(row==0u&&li==0u){out.format=1u;out.capacity=p.grid.z;}
 if(row>=p.grid.y){return;}
 let y=f32(row)*${TONE_ROW}.0+${TONE_ROW/2};
 if(y<p.rect.y||y>=p.rect.y+p.rect.w){if(li==0u){rowState[row*ROW_STATE]=0u;}return;}
 let bounds=rowBounds();let step=colStep();
 let columns=u32(max(0,(bounds.y-bounds.x+step-1)/step));
 if(columns>WALK_WORDS*32u){if(li==0u){textureEdges(row,y,bounds.x,bounds.y);saveRow(row);}return;}
 for(var w=li;w<WALK_WORDS;w+=64u){atomicStore(&walkStarts[w],0u);atomicStore(&walkContinues[w],0u);}
 workgroupBarrier();
 for(var c=li;c<columns;c+=64u){
  let x=bounds.x+i32(c)*step;let h=leafHit(toField(vec2f(f32(x)+.5,y)));
  if((h.x|h.y)==0u){continue;}
  if(styleOf(h.x)==0u){atomicOr(&walkStarts[c/32u],1u<<(c%32u));}
  if(c>0u){let before=leafHit(toField(vec2f(f32(x-step)+.5,y)));if((before.x|before.y)!=0u&&leafOn(h,before.xy,bitcast<f32>(before.z))){atomicOr(&walkContinues[c/32u],1u<<(c%32u));}}
 }
 workgroupBarrier();
 if(li!=0u){return;}
 var c=0u;
 while(c<columns){
  if(!startsSpan(c)){c++;continue;}
  let x=bounds.x+i32(c)*step;let q=toField(vec2f(f32(x)+.5,y));let h=leafHit(q);
  let leftDepth=bitcast<f32>(h.z);var end=c+1u;
  while(end<columns&&continuesSpan(end)){end++;}
  let endX=bounds.x+i32(end)*step;
  if(endX-x>=8){
   let last=leafHit(toField(vec2f(f32(bounds.x+i32(end-1u)*step)+.5,y)));
   let tier=toneTier(q,leftDepth,h.xy);
   if(tier<2u){edgeSpan(row,y,x,endX,h.xy,leftDepth,bitcast<f32>(last.z),tier);}
  }
  c=end;
 }
 saveRow(row);
}
@compute @workgroup_size(64) fn rowPrelude(@builtin(global_invocation_id) id:vec3u){
 let row=id.x;
 if(row==0u){out.format=1u;out.capacity=p.grid.z;}
 if(row>=p.grid.y){return;}
 let y=f32(row)*${TONE_ROW}.0+${TONE_ROW/2};
 if(y<p.rect.y||y>=p.rect.y+p.rect.w){
  rowState[row*ROW_STATE]=0u;return;
 }
 let bounds=rowBounds();
 textureEdges(row,y,bounds.x,bounds.y);
 saveRow(row);
}
// ONE THREAD PER LATTICE NODE. Every page pixel of a native row has a thread;
// the thread reads its own surface point, the class of its source and the
// surface lattice there (latticeAt, keyed by latticeKey: the pattern travels
// with the surface). Only the pixel that holds a node (the node's page X
// rounds to the pixel) goes on; it reads the node's level (THE UNIT'S LEVEL)
// and places its band members or looks its word up (bandUnits, wordOf), applies the fixed erasures and the fixed clearance rule at the
// node's one pen (the node's X plus one native half space on alternate
// lattice rows: ずらし, read from the surface, never from the page row) and
// writes the unit. Neighbouring threads never wait on each other and no span
// is walked.

// sea looks and behaves like the frozen 10-12 Y8 runtime's water,
// runtimes/y8-20261012 packages/trueff/render/ff/terrain.ts FL-4918 attempts
// 217-329, spelled as its layout.ts:1081-1089 spells WATER from
// style/materials.ts). The field law is compiled into this shader once:
// ttChop's two sines (1.1 scale, .62/.38, 1.27, phase .82) on 1 m CARRIER
// cells, a crest where |chop| crosses mix(.42,.54) of the 9 m world patch
// envelope (ttWaveMaskOpen), the chord fixed by the cell's corners (ttCrest,
// saddle part 1 as there).
// THE CARRIER FRAME: the game poses the water source by the swell drift
// (sea-wave.ts seaCarrierDrift = ttCarrierDrift), so in the water's source
// coordinates both phases cancel. A carrier cell, its chord and its address
// stay fixed while the pose carries them across the world (attempt 329: a
// crest keeps its identity and its spelling variant while it travels); the
// patch envelope stays world-fixed, so crests open and close as they pass it.
// ONE THREAD PER CARRIER CELL (seaCells): the window of SEA_SPAN x SEA_SPAN
// carrier cells round the eye. The cell projects its own chord; it emits its
// crest where the chord midpoint lies on visible water. Where cells project
// smaller than the page holds, a crest is kept by a nested share of its
// address hash against the cell's own projected area (one crest per about
// SEA_CELL_PX page px of water), so approach only adds crests and nothing
// depends on where a page column samples. Overlaps resolve by the seal in
// the canonical member order. No search, no CPU path.
// THE WORD (layout.ts:1082-1089, by table): near shore (a dry 4-neighbour of
// the world cell, ttCrestNearShore) the one-glyph ripple, or surf while the
// same front arrives onshore (ttCrestBreaks); else level / rising / falling
// by the chord's page lean (.24) and the sign of its page dy; the variant by
// the carrier address hash.
fn seaHash2(c:vec2i)->f32{var h=(bitcast<u32>(c.x)*0x8da6b343u)^(bitcast<u32>(c.y)*0xd8163841u);h=(h^(h>>13u))*0x5bd1e995u;h=h^(h>>15u);return f32(h&65535u)/65535.0;}
fn seaOpen(xz:vec2f)->f32{let g=xz/9.0;let c=vec2i(floor(g));var t=fract(g);t=t*t*(3.0-2.0*t);
 let n=mix(mix(seaHash2(c),seaHash2(c+vec2i(1,0)),t.x),mix(seaHash2(c+vec2i(0,1)),seaHash2(c+vec2i(1,1)),t.x),t.y);return smoothstep(.32,.58,n);}
// ttChop and ttChopRate in the carrier frame (world minus drift): static there.
fn seaChop(c:vec2f)->f32{return clamp(sin(c.x*1.1)*.62+sin(c.y*1.1*1.27)*.38,-1.0,1.0);}
fn seaRate(c:vec2f)->f32{return cos(c.x*1.1)*.62-cos(c.y*1.1*1.27)*.82*.38;}
fn seaField(c:vec2f,w:vec2f)->f32{return abs(seaChop(c))-mix(.42,.54,seaOpen(w));}
fn seaPage(X:vec3f)->vec3f{let c=p.clipFromWorld*vec4f(X,1.0);if(c.w<=1e-4){return vec3f(-1e9,-1e9,0.0);}
 return vec3f(p.rect.x+(c.x/c.w*.5+.5)*p.rect.z,p.rect.y+(.5-c.y/c.w*.5)*p.rect.w,c.w);}
fn seaOnPage(g:vec3f)->bool{return g.z>0.0&&g.x>=p.rect.x&&g.y>=p.rect.y&&g.x<p.rect.x+p.rect.z&&g.y<p.rect.y+p.rect.w;}
// The surface at a page point is a liquid one (the retained wet cell).
fn seaLiquidAt(g:vec3f)->bool{if(!seaOnPage(g)){return true;}let t=at(toField(g.xy));return t.on&&liquidOf(t.o.x);}
// Carrier addresses: the drift wraps after whole swell periods (sea-wave.ts
// SEA_CARRIER_WRAP, 7 x and 8 z periods, about 40 x 36 m), so an address is
// the carrier cell modulo 40 x 36 cells, unchanged across a wrap.
const SEA_WRAP=vec2i(40,36);
const SEA_SPAN=384u;
const SEA_CELL_PX=480.0;
var<workgroup> seaAttach:atomic<u32>;
@compute @workgroup_size(64) fn seaCells(@builtin(global_invocation_id) id:vec3u,@builtin(local_invocation_index) li:u32){
 clearRef=vec4f(0.0);
 // The water's certified source pose (every liquid chunk shares it): the
 // lowest liquid attachment, read once by the workgroup.
 if(li==0u){atomicStore(&seaAttach,0xffffffffu);}
 workgroupBarrier();
 for(var i=li;i<p.ownership.x;i+=64u){let src=attachments[i].source;if(src.z==0u&&liquidOf(src.x)){atomicMin(&seaAttach,i);}}
 workgroupBarrier();
 let a=atomicLoad(&seaAttach);
 if(a==0xffffffffu||id.x>=SEA_SPAN||id.y>=SEA_SPAN){return;}
 let toSource=attachments[a].sourceFromWorld;let toWorld=attachments[a].worldFromSource;
 // The pose carries the water level too (physical-world.ts: the sheet is
 // built at height 0 and posed up by sea level), so the level is its y.
 let level=toWorld[3].y;
 let eye=(toSource*vec4f(p.eye.xyz,1.0)).xz;
 let cell=vec2i(floor(eye))+vec2i(i32(id.x),i32(id.y))-vec2i(i32(SEA_SPAN/2u));
 let lo=vec2f(cell);
 var c=array<vec2f,4>(lo,lo+vec2f(1.0,0.0),lo+vec2f(1.0,1.0),lo+vec2f(0.0,1.0));var f:array<f32,4>;var w:array<vec3f,4>;
 for(var k=0;k<4;k++){w[k]=(toWorld*vec4f(c[k].x,0.0,c[k].y,1.0)).xyz;f[k]=seaField(c[k],w[k].xz);}
 var hit:array<vec2f,4>;var count=0u;
 for(var k=0u;k<4u;k++){let j=(k+1u)&3u;if((f[k]>0.0)==(f[j]>0.0)){continue;}let t=clamp(f[k]/(f[k]-f[j]),0.0,1.0);hit[count]=mix(c[k],c[j],t);count++;}
 if(count<2u){return;}
 // The cell's own projected area (page px^2): the nested share's measure.
 var q:array<vec3f,4>;for(var k=0;k<4;k++){q[k]=seaPage(w[k]);}
 if(q[0].z<=0.0||q[1].z<=0.0||q[2].z<=0.0||q[3].z<=0.0){return;}
 let area=.5*abs((q[2].x-q[0].x)*(q[3].y-q[1].y)-(q[3].x-q[1].x)*(q[2].y-q[0].y));
 let m=vec2u(u32(((cell.x%SEA_WRAP.x)+SEA_WRAP.x)%SEA_WRAP.x),u32(((cell.y%SEA_WRAP.y)+SEA_WRAP.y)%SEA_WRAP.y));
 let heading=normalize(vec2f(-1.0/1.1,.82/(1.1*1.27)));
 for(var part=0u;part<2u;part++){
  if(part==1u&&count!=4u){continue;}
  let address=2u*(m.y*u32(SEA_WRAP.x)+m.x)+part;
  var h=address*0x9e3779b1u;h^=h>>15u;h*=0x2c1b3c6du;h^=h>>12u;
  if(f32(h&65535u)/65535.0>=min(1.0,area/SEA_CELL_PX)){continue;}
  var ia=part*2u;var ib=ia+1u;
  if(count==4u){let mc=lo+vec2f(.5);if(seaField(mc,(toWorld*vec4f(mc.x,0.0,mc.y,1.0)).xz)>0.0){ia=select(1u,3u,part==0u);ib=select(2u,0u,part==0u);}}
  var A=(toWorld*vec4f(hit[ia].x,0.0,hit[ia].y,1.0)).xyz;A.y=level;var B=(toWorld*vec4f(hit[ib].x,0.0,hit[ib].y,1.0)).xyz;B.y=level;
  let M=(A+B)*.5;let g=seaPage(M);
  if(!seaOnPage(g)){continue;}
  let row=u32(floor(g.y/${TONE_ROW}.0));let y=f32(row)*${TONE_ROW}.0+${TONE_ROW/2};
  if(row>=p.grid.y||y<p.rect.y||y>=p.rect.y+p.rect.w){continue;}
  let s=at(toField(vec2f(g.x,y)));if(!s.on||!liquidOf(s.o.x)||s.n.y<=0.0){continue;}
  let pa=seaPage(A);let pb=seaPage(B);var d=pb.xy-pa.xy;if(d.x<0.0){d=-d;}
  let lean=abs(d.y)/max(d.x,1e-6);let variant=(h>>8u)&1u;
  let wc=floor(M.xz)+vec2f(.5);var n=vec2f(0.0);
  if(!seaLiquidAt(seaPage(vec3f(wc.x-1.0,level,wc.y)))){n+=vec2f(-1.0,0.0);}
  if(!seaLiquidAt(seaPage(vec3f(wc.x+1.0,level,wc.y)))){n+=vec2f(1.0,0.0);}
  if(!seaLiquidAt(seaPage(vec3f(wc.x,level,wc.y-1.0)))){n+=vec2f(0.0,-1.0);}
  if(!seaLiquidAt(seaPage(vec3f(wc.x,level,wc.y+1.0)))){n+=vec2f(0.0,1.0);}
  var passage=${SEA_CREST}u;
  if(dot(n,n)>1e-6){
   let cc=(toSource*vec4f(wc.x,level,wc.y,1.0)).xz;
   let breaks=dot(heading,normalize(n))>-.2&&seaChop(cc)*seaRate(cc)>0.0;
   passage=select(${SEA_CREST}u+6u+variant,${SEA_CREST}u+8u+((h>>9u)&3u),breaks);
  }else if(lean<.24){passage=${SEA_CREST}u+variant;}
  else{passage=select(${SEA_CREST}u+4u,${SEA_CREST}u+2u,d.y<0.0)+variant;}
  let advance=f32(passageHeader(passage).z);let pen=round(g.x-advance*.5);
  if(!unitClear(pen,row,y,advance,s,MODE_FACET,true)){atomicAdd(&out.flags,256u);continue;}
  emitPassage(row,y,pen,passage,lightBand(s.n,s.v,M),s.o,vec4u(0x5743u,m.x,m.y,0x80000000u|((passage&255u)<<3u)|(part<<11u)),s.d);
 }
}
@compute @workgroup_size(64) fn passages(@builtin(global_invocation_id) id:vec3u,@builtin(local_invocation_index) li:u32){
 passageNode(id);
 workgroupBarrier();
 runToneJobs(li);
}
fn passageNode(id:vec3u){
 clearRef=vec4f(0.0);
 let row=id.y;if(row>=p.grid.y){return;}
 let y=f32(row)*${TONE_ROW}.0+${TONE_ROW/2};
 if(y<p.rect.y||y>=p.rect.y+p.rect.w){return;}
 let bounds=rowBounds();let x=bounds.x+i32(id.x)*colStep();
 if(x>=bounds.y){return;}
 particleCell(row,y,x);
 let q=toField(vec2f(f32(x)+.5,y));let s=at(q);if(!s.on||formParentSelected(s.o)){return;}
 // A transparent surface is not an opaque receiver.
 if(textureLoad(radiance,vec2i(q),0).a<.99){return;}
 let style=styleOf(s.o.x);
 var kind=KIND_FACET;
 if(style==2u){kind=KIND_ROUND;}
 else if(levelFace(s.n,style)){kind=select(KIND_FLOOR,KIND_GROUND,groundFace(s.n,style)&&openGround(q,s.d,s.o,style));}
 else if(style==1u){kind=KIND_PLANE;}
 else if(style==3u){kind=KIND_SLOPE;}
 // A foliage FACE (authored class 6) of a plane, a facet-built form or a
 // slope is crown: a mixed source (a pine's needles) draws crown mass.
 let leafy=materialAt(q,s.o)==6u;
 if(leafy&&(kind==KIND_PLANE||kind==KIND_SLOPE)){kind=KIND_FACET;}
 let foliage=kind==KIND_FACET&&(foliageOf(s.o.x)||leafy);
 let organic=kind==KIND_FACET&&!foliage&&organicOf(s.o.x);
 let P=unproject(floor(q)+vec2f(.5),s.d);
 // tone2: a round's wedge is held by its rim pixels (roundRim), not by lattice nodes.
 if(kind==KIND_ROUND){roundRim(row,y,x,q,s,P);return;}
 // sea-100: the 10-12 sea (seaCells, one thread per carrier cell) replaces the dash field on a liquid ground (waterUnits kept, not called).
 if(kind==KIND_GROUND&&liquidOf(s.o.x)&&s.n.y>0.0){return;}
 let L=latticeAt(f32(x)+.5,y,s.n,P,latticeAttachment(s.o),select(slotOf(kind,foliage,organic),${SEA_SLOT}.0,kind==KIND_GROUND&&liquidOf(s.o.x)));
 // The node belongs to the first field column at or after its X: this
 // column's sample (x + 0.5) lies in the node's own cell, which starts at the
 // node, so the node is in (x - colStep, x]. One column per node.
 let nodeX=i32(round(L.x0));
 // THE CELL'S STAND-IN (fields only: planes, facet-built plane regions,
 // slopes). A face narrower than a field cell, whose cell node lies on another
 // surface, would have no thread. The first field column of the face inside
 // the cell (the column before it is another surface) stands in for the node
 // when the node's own X is another surface: it places the cell's units from
 // its own X on, at the cell's phase on this face's lattice. Two fixed reads,
 // no search; every other column returns.
 var fromX=-1e9;
 if(!L.ok){return;}
 // tone2: every column of a level face may begin a cast shadow's row run.
 if((kind==KIND_GROUND||kind==KIND_FLOOR)&&s.n.y>0.0&&!(kind==KIND_GROUND&&liquidOf(s.o.x))){shadowRun(row,y,x,q,s,P,L,select(MODE_FACET,MODE_TERRAIN,kind==KIND_GROUND||style==3u));}
 if(nodeX>x||nodeX<=x-colStep()){
  let field=kind==KIND_PLANE||kind==KIND_SLOPE||(kind==KIND_FACET&&!organic);
  if(!field){return;}
  if(sameRaw(s,toField(vec2f(f32(x-colStep())+.5,y)),kind==KIND_PLANE)){return;}
  if(sameRaw(s,toField(vec2f(L.x0+.5,y)),kind==KIND_PLANE)){return;}
  fromX=f32(x);
 }
 // This thread holds the node. Its quantities, all on the surface point:
 let air=transmittance(q,P);if(air<.1){return;}
 // tone2 c1: the holder's plane depth line along the row (two meets, no search).
 if(kind==KIND_PLANE||kind==KIND_SLOPE||(kind==KIND_FACET&&!foliage&&!organic)){
  let pl=vec4f(s.n,-dot(s.n,P));let hx=p.rect.x+(floor(q.x)+.5)*p.rect.z/p.field.x;
  let M0=meet(P,pl,rowPlane(y),colPlane(hx));let M1=meet(P,pl,rowPlane(y),colPlane(hx+8.0));
  if(M0.w!=0.0&&M1.w!=0.0){
   let c0=p.clipFromWorld*vec4f(M0.xyz,1.0);let c1=p.clipFromWorld*vec4f(M1.xyz,1.0);
   if(c0.w>0.0&&c1.w>0.0&&c0.z/c0.w>0.0){clearRef=vec4f(hx,s.d,(c1.z/c1.w-c0.z/c0.w)/8.0,c0.z/c0.w);}
  }
 }
 let band=lightBand(s.n,s.v,P);
 // tone2 m2: no key light: normalize(0) never decides a band (see valueLevel).
 let keyLit=dot(lightDirection(),lightDirection())>=.000001;
 let toLight=select(vec3f(0.0,1.0,0.0),normalize(select(vec3f(0.0,1.0,0.0),lightDirection(),keyLit)),keyLit);
 let nl=select(0.0,dot(s.n,toLight),keyLit);
 // A level face (open ground or a floor): shadow-edge bands and sparse
 // receding intervals only (levelUnits); a ceiling, a level face turned
 // from the light, stays paper.
 if(kind==KIND_GROUND||kind==KIND_FLOOR){
  if(s.n.y<=0.0){return;}
  if(kind==KIND_GROUND&&liquidOf(s.o.x)){waterUnits(row,y,q,s,L,band);return;}
  levelUnits(row,y,q,s,L,band,P,select(MODE_FACET,MODE_TERRAIN,kind==KIND_GROUND||style==3u),false);return;
 }
 let cliff=kind==KIND_SLOPE&&s.n.y<.5;
 // tone2 s4: turf is the face's authored material class (8, turf), never a colour probe.
 let grass=kind==KIND_SLOPE&&!cliff&&materialAt(q,s.o)==8u;
 let index=L.node+i32(L.odd);
 // The level, by class (see THE UNIT'S LEVEL):
 var level=0;var grade=vec4f(0.0,0.0,0.0,-1.0);var shade=0;
 if(kind==KIND_SLOPE){
  // A terrain slope's material band by its facet's own light band.
  // tone2 s4: lit (band 2) paper or sparse intervals; band 3 light; band 4 or a
  // cast shadow mid; turned well away core. No carpet on a lit face.
  if(band==3u){level=1;}else if(band==1u||band==4u){level=select(2,3,band==4u&&nl< -.25);}
 }else if(kind==KIND_ROUND){
  level=roundLevel(nl,abs(dot(s.n,s.v)));
 }else if(foliage){
  // A CROWN'S BAND: its words gather where the crown turns from the light
  // (the terminator band, n·L -.3 to .35) and at its rim (the texture edge
  // carries the envelope: textureEdges); the deep shaded core keeps only its
  // sparse fringe and the lit side is paper. Never a whole-body field.
  // THE CROWN'S FIELD (2026-11-13; replaces the sparse terminator band):
  // the crown mass is one constant pattern per crown per level, graded by
  // n·L: under -.1 level 3, under .15 level 2, else level 1 (the lit side
  // lighter, never a whole crown of paper); a cast shadow at least level 2.
  level=select(select(1,2,nl<.15),3,nl< -.1);
  if(band==1u){level=max(level,2);}
 }else if(organic){
  level=organicLevel(nl,abs(dot(s.n,s.v)));
  level=max(level,shadowBand(P,s.n,nl,band,y,L));
 }else{
  // A plane (planar volume face, or a facet-built form's plane region): its
  // dark-boundary band (darkEdgeDistance, planeLevel); a lit face is paper
  // and does not ask. A surface no plane region holds (a facet-built form's
  // lone facet, a source without a table) has no boundary to band along and
  // takes no turn band (planeFallback): its facets are flat, so a band by
  // n·L alone would fill each one whole.
  if(!planeCarries(s.n,nl,band)){return;}
  let t=darkEdgeDistance(P,s.n,s.o,nl,toLight);
  let step=cellStep(P,s.n,y,L);
  // A region too small to hold a band (radius under REGION_LEAST_CELLS
  // cells: a log, a shingle, a plank of a detailed wall or roof) is texture
  // of a larger surface, not a plane: no turn band of its own, or every small
  // region fills (spot-roofs walls and roofs, tone-bands-c).
  // The field cell is FIELD_UNITS band cells wide: the size guard keeps its
  // band-cell measure, and the dark-edge band is half a field cell.
  // A point no compiled region holds, or a region too small for a band of
  // its own (a plank, a shingle, a lone facet), is texture of the larger
  // surface: the field continues across it at its band's lighter level, so a
  // face never shows cell-wide paper channels.
  // PER-UNIT GRADE: the dark-edge distance (in field cells) at the node and
  // at the cell's far end on this face's plane; each unit reads the level of
  // the distance between them at its own pen (fieldUnits), so the dense band
  // follows the face's dark edge, not the lattice cell's sides.
  var c0=1e9;var c1=1e9;
  if(t.x>=0.0&&step>0.0&&t.y>=${(REGION_LEAST_CELLS*BAND_SLOT/FIELD_SLOT).toFixed(4)}*step){
   c0=t.x/step;c1=c0;
   let E=meet(P,vec4f(s.n,-dot(s.n,P)),rowPlane(y),colPlane(L.x1));
   if(E.w!=0.0){let t1=darkEdgeDistance(E.xyz,s.n,s.o,nl,toLight);if(t1.x>=0.0){c1=t1.x/step;}}
  }
  shade=shadowBand(P,s.n,nl,band,y,L);
  level=max(planeLevel(min(c0,c1),nl,band),shade);
  grade=vec4f(c0,c1,nl,air);
 }
 // tone2 m2: no key light: one level per cell from the node's own value.
 if(!keyLit){level=valueLevel(shown(q));grade=vec4f(0.0,0.0,0.0,-1.0);shade=0;}
 level=airLevel(level,air);
 if(kind==KIND_SLOPE&&level==0){
  // A lit steep face is paper (the markup's side plane: the band stops and
  // the white face carries the light). A gentle lit slope is the ground
  // running on: its sparse receding intervals (lit turf: its fringe word).
  if(cliff||fromX>-1e8){return;}
  levelUnits(row,y,q,s,L,band,P,MODE_TERRAIN,grass);
  return;
 }
 if(level==0){return;}
 // Distance: the form's own tier (toneTier: its size on the page).
 let tier=toneTier(q,s.d,s.o);
 if(tier>=1u){level=min(level,2);}
 let far=tier>=1u;
 var mode=MODE_FACET;
 if(kind==KIND_PLANE){mode=MODE_PLANE;}
 else if(kind==KIND_ROUND||organic){mode=MODE_ROUND;}
 else if(kind==KIND_SLOPE){mode=MODE_TERRAIN;}
 // A ROUND'S RIM WEDGE: where its shaded side (level 1 and over) turns to
 // the silhouette (the surface grazes the view, |n·v| under .45), the band
 // densifies to abutted members against the inner rim (the tonemapping
 // target's ".:;:;:;" rows at each sphere's lower right; skill section 8
 // rule 3: rim dots carry the curvature). Only there are members abutted.
 var rim=false;
 if((kind==KIND_ROUND||(organic&&nl<.15))&&level>=1&&abs(dot(s.n,s.v))<.45){rim=true;level=select(3,2,tier>=1u);}
 if(kind==KIND_ROUND||organic){bandUnits(row,y,s,L,level,band,tier,mode,kind==KIND_FACET,rim);return;}
 if(!foliage){
  var word=fieldWord(level);var rowUnit=fieldRow(level,L.odd);
  // A terrain slope takes one band darker within its material (light band
  // ,:,:,: not the spaced : : :), so a shaded slope is a field; lit steep
  // faces (band 0) stayed paper above.
  if(kind==KIND_SLOPE&&grass){grassUnits(row,y,s,L,level,band,tier,fromX);return;}
  if(kind==KIND_SLOPE){word=wordOf(kind,level,false,nl,cliff,grass);rowUnit=0u;}
  fieldUnits(row,y,s,L,word,level,band,tier,mode,kind==KIND_FACET,rowUnit,fromX,grade,shade);return;
 }
 {let crown=crownRow(level,L.odd);fieldUnits(row,y,s,L,crown,level,band,tier,mode,true,crown,fromX,vec4f(0.0,0.0,0.0,-1.0),0);return;}
 var every=1;if(level==1){every*=2;}if(tier==2u){every*=2;}
 if((index%every+every)%every!=0){return;}
 let passage=wordOf(kind,level,foliage,nl,cliff,grass);
 if(passage==0u){return;}
 let advance=f32(passageHeader(passage).z);
 let pen=round(L.x0)+select(0.0,${passages.gap}.0,L.odd);
 if(pen+advance>round(L.x1)){return;}
 // A crown's words keep one half space clear of its row's texture-edge
 // words (the envelope is spelled as texture there).
 if(kind==KIND_FACET){
  loadRow(row);
  if(edgeTaken(pen,pen+advance)){return;}
 }
 if(!unitClear(pen,row,y,advance,s,mode,far)){atomicAdd(&out.flags,256u);return;}
 emitPassage(row,y,pen,passage,band,s.o,latticeKey(L,0u,passage,band),s.d);
}
// PARTICLES (D3: effects on their own trajectories; tone2, 2026-11-13). An
// additive draw that writes no surface (an engine's motes) leaves its coverage
// as +4 on the field's radiance alpha (surface-fields.ts emissive pipelines;
// a surface's own alpha is at most 1). The row's fixed page cells
// (PARTICLE_CELL px: the motes move on their own paths, the cell does not) that
// hold coverage take the spark word, one per cell, at the cell's middle. The
// first field column of a cell asks; it reads the cell every 2 px across and at
// four heights of the row.
fn particleCell(row:u32,y:f32,x:i32){
 let c=x/${PARTICLE_CELL};if(x>0&&(x-colStep())/${PARTICLE_CELL}==c){return;}
 let x0=c*${PARTICLE_CELL};var hit=false;
 for(var dy=2;dy<${TONE_ROW}&&!hit;dy+=4){
  for(var dx=0;dx<${PARTICLE_CELL};dx+=2){
   let i=vec2i(floor(toField(vec2f(f32(x0+dx)+.5,f32(row)*${TONE_ROW}.0+f32(dy)+.5))));
   if(any(i<vec2i(0))||any(i>=vec2i(p.field.xy))){continue;}
   if(particleAt(i)){hit=true;break;}
  }
 }
 if(!hit){return;}
 let pen=f32(x0)+round((${PARTICLE_CELL}.0-${advanceOf(PARTICLE)}.0)*.5);
 // A particle word's source: the producer-row flag (bit 31) on domain 0 and
 // row 0, so the decode takes its lifetime directly (no entity owner map).
 emitPassage(row,y,pen,${PARTICLE}u,0u,vec2u(0x80000000u,0u),vec4u(bitcast<u32>(c),row,0u,(${PARTICLE}u&255u)<<3u),1.0);
}
fn luma(c:vec3f)->f32{return dot(c,vec3f(.2126,.7152,.0722));}
fn shown(q:vec2f)->f32{
 let page=p.rect.xy+q*p.rect.zw/p.field.xy;
 let xy=clamp(vec2i(page/p.field.zw*vec2f(textureDimensions(appearance))),vec2i(0),vec2i(textureDimensions(appearance))-vec2i(1));
 return luma(textureLoad(appearance,xy,0).rgb);
}
fn ratio(q:vec2f)->f32{return clamp(shown(q)/max(.0001,luma(textureLoad(radiance,vec2i(q),0).rgb)),0.0,64.0);}
fn region(q:vec2f,c:S)->vec4u{
 // Depth belongs to a texel CENTRE. Unprojecting the caller's fractional
 // cell centre with that depth invents different planes across one floor,
 // so shadow samples cannot find the lit reference of their own receiver.
 let P=unproject(select(q,floor(q)+vec2f(.5),p.grid.w!=0u),c.d);
 return vec4u(c.o,pack4x8snorm(vec4f(c.n,0)),bitcast<u32>(i32(round(dot(c.n,P)*1024.0))));
}
fn referenceSlot(q:vec2f,c:S,cell:u32,create:bool)->u32{
 let key=region(q,c);var hash=(key.x*0x9e3779b9u)^(key.y*0x85ebca6bu)^(key.z*0xc2b2ae35u)^key.w;
 hash=(hash^(hash>>16u))*0x7feb352du;let mask=arrayLength(&references)-1u;
 // Exact source/plane address resolution, not a glyph/pen decision.
 for(var probe=0u;probe<32u;probe++){
  let slot=(hash+probe)&mask;var address=atomicLoad(&references[slot].sample);
  if(address==0u&&create){let claim=atomicCompareExchangeWeak(&references[slot].sample,0u,cell+1u);address=select(claim.old_value,cell+1u,claim.exchanged);}
  if(address==0u){return 0xffffffffu;}
  let index=address-1u;let page=vec2f(f32(index%p.grid.x)*${TONE_CELL}.0+${TONE_CELL/2},f32(index/p.grid.x)*${TONE_ROW}.0+${TONE_ROW/2});
  let other=toField(page);let surface=at(other);
  if(surface.on&&all(region(other,surface)==key)){return slot;}
 }
 return 0xffffffffu;
}
@compute @workgroup_size(8,8) fn litReference(@builtin(global_invocation_id) id:vec3u){
 if(id.x>=p.grid.x||id.y>=p.grid.y){return;}
 let page=vec2f(f32(id.x*${TONE_CELL}u)+${TONE_CELL/2},f32(id.y*${TONE_ROW}u)+${TONE_ROW/2});
 if(any(page<p.rect.xy)||any(page>=p.rect.xy+p.rect.zw)){return;}
 let q=toField(page);let c=at(q);if(!c.on||formParentSelected(c.o)){return;}
 let slot=referenceSlot(q,c,id.y*p.grid.x+id.x,true);
 if(slot!=0xffffffffu){
  // Native radiance already contains material lighting, including shadows.
  // Dividing displayed by native alone would CANCEL those shadows. Retain
  // both the receiver's native lit value and its post-effect response.
  atomicMax(&references[slot].maximum,bitcast<u32>(max(0.0,luma(textureLoad(radiance,vec2i(q),0).rgb))));
  atomicMax(&references[slot].response,bitcast<u32>(ratio(q)));
 }
}
// Legacy per-cell scalar members (CPU RegionTone, D2); compiled playback
// uses the row-region passages entry above.
@compute @workgroup_size(8,8) fn main(@builtin(global_invocation_id) id:vec3u){
 if(all(id==vec3u(0))){out.format=0u;out.capacity=p.grid.z;}
 if(id.x>=p.grid.x||id.y>=p.grid.y){return;}
 let x0=f32(id.x*${TONE_CELL}u);let y0=f32(id.y*${TONE_ROW}u);
 let center=vec2f(x0+${TONE_CELL/2},y0+${TONE_ROW/2});
 if(any(center<p.rect.xy)||any(center>=p.rect.xy+p.rect.zw)){return;}
 let c=at(toField(center));
 // Absence of geometry is NOT an atmosphere owner. Backdrop brightness
 // cannot distinguish a clear dark background from fog, haze or exposure.
 // This field has no bounded volume/effect owner there, so it stays paper.
 // Actual fog/exposure response on identified receivers is retained below;
 // a future volume producer must supply its own bounds and effect field.
 if(!c.on||formParentSelected(c.o)){return;}
 // This producer describes an opaque receiver's CAST shadow. Transparent
 // surfaces need their transmission relationship, not an opaque tone fill.
 if(p.grid.w!=0u&&textureLoad(radiance,vec2i(toField(center)),0).a<.99){return;}
 // Legacy keeps its cell-corner contract. Compiled members below validate
 // their actual native ink, rather than discarding a mostly empty cell.
 var corners=array<vec2f,4>(vec2f(x0+1.0,y0+2.0),vec2f(x0+${TONE_CELL-1}.0,y0+2.0),vec2f(x0+1.0,y0+${TONE_ROW-2}.0),vec2f(x0+${TONE_CELL-1}.0,y0+${TONE_ROW-2}.0));
 if(p.grid.w==0u){for(var k=0;k<4;k++){let corner=toField(corners[k]);if(!same(c,at(corner))){return;}}}
 let q=select(toField(center),floor(toField(center))+vec2f(.5),p.grid.w!=0u);let P=unproject(q,c.d);
 let L=select(c.v,normalize(p.light.xyz),p.light.w>0.0);
 let nl=max(dot(c.n,L),0.0);
 let band=select(select(select(.45,.62,nl>=.15),.8,nl>=.4),1.0,nl>=.7);
 // Legacy retains its appearance blend under D2. Compiled playback below
 // uses actual n.L/caster bands, never absolute beauty-frame brightness.
 let value=select(mix(band,shown(q),.35),band,p.grid.w!=0u);
 var level=select(select(select(3u,2u,value>=.55),1u,value>=.75),0u,value>=.9);
 // Scene shadow is relative to this SAME receiver plane's lit value.
 // Exposure cancels in the ratio; thresholds do not change across frames.
 var reference=0xffffffffu;if(p.grid.w==0u){reference=referenceSlot(q,c,id.y*p.grid.x+id.x,false);}
 if(reference!=0xffffffffu){
  let lit=bitcast<f32>(atomicLoad(&references[reference].maximum));let response=bitcast<f32>(atomicLoad(&references[reference].response));
  // These are receiver-value regions, not a claim to recover the engine's
  // separate caster identity or distinguish a dark material texel from shadow.
  let shadow=max(0.0,luma(textureLoad(radiance,vec2i(q),0).rgb))/max(lit,.0001);
  let atmosphere=ratio(q)/max(response,.0001);let relative=min(shadow,atmosphere);
  if(relative<.3){level=3u;}else if(relative<.55){level=max(level,2u);}
 }
 if(p.grid.w!=0u){
  // Membership comes from scene n.L and actual caster visibility, never
  // absolute radiance or a receiver's brightest material texel.
  let regionBand=lightBand(c.n,c.v,P);if(regionBand==0u){return;}
  // The scalar lattice cannot express a contact passage. Do not turn the
  // entire cast-shadow footprint into colon wallpaper while that producer
  // is absent. This withdraws the area fill, not the geometry/contact lines.
  if(regionBand==1u){return;}
  // Object-interior value bands remain separate from cast-shadow coverage.
  level=select(select(1u,2u,regionBand==3u),3u,regionBand==4u);
 }
 if(level==0u){return;}
 // Source coordinates for certified physical posing; world coordinates for
 // the existing uncharted material path. Lighting/visibility above and below
 // remain in WORLD space. Only attachment and its correspondence key change.
 let attachment=attachmentFor(c.o);var surface=P;var surfaceNormal=c.n;
 var neighbor=unproject(q+vec2f(1,0),c.d);
 if(attachment!=0xffffffffu){
  let local=attachments[attachment].sourceFromWorld*vec4f(P,1);surface=local.xyz/local.w;
  let next=attachments[attachment].sourceFromWorld*vec4f(neighbor,1);neighbor=next.xyz/next.w;
  surfaceNormal=normalize((transpose(attachments[attachment].worldFromSource)*vec4f(c.n,0)).xyz);
 }
 // Dyadic source spacing, evaluated at the current projected scale.
 let perField=length(neighbor-surface);
 let perPage=perField*p.field.x/p.rect.z;
 let s=exp2(round(log2(max(${TONE_SPACING}.0*perPage,1e-6))));
 let a=abs(surfaceNormal);var k=2;if(a.x>=a.y&&a.x>=a.z){k=0;}else if(a.y>=a.z){k=1;}
 let i=(k+1)%3;let j=(k+2)%3;
 let ia=i32(round(surface[i]/s));let ib=i32(round(surface[j]/s));
 if(level==1u&&((ia+ib)&1)!=0){return;}
 var node=surface;node[i]=f32(ia)*s;node[j]=f32(ib)*s;node[k]=surface[k]-(surfaceNormal[i]*(node[i]-surface[i])+surfaceNormal[j]*(node[j]-surface[j]))/surfaceNormal[k];
 var worldNode=vec4f(node,1);if(attachment!=0xffffffffu){worldNode=attachments[attachment].worldFromSource*worldNode;}
 let clip=p.clipFromWorld*worldNode;if(clip.w<=0.0){return;}
 let ndc=clip.xy/clip.w;let fieldPx=vec2f((ndc.x+1.0)*.5,(1.0-ndc.y)*.5)*p.field.xy;
 let page=p.rect.xy+fieldPx*p.rect.zw/p.field.xy;
 if(page.x<x0||page.x>=x0+${TONE_CELL}.0||floor(page.y/${TONE_ROW}.0)!=f32(id.y)){return;}
 let nodeSurface=at(fieldPx);if(!same(c,nodeSurface)){return;}
 if(p.grid.w!=0u&&lightBand(nodeSurface.n,nodeSurface.v,unproject(floor(fieldPx)+vec2f(.5),nodeSurface.d))!=lightBand(c.n,c.v,P)){return;}
 if(p.grid.w!=0u&&!toneFits(page,id.y,level,nodeSurface,lightBand(c.n,c.v,P))){atomicAdd(&out.flags,256u);return;}
 let slot=atomicAdd(&out.count,1u);if(slot>=p.grid.z){atomicOr(&out.flags,1u);return;}
 out.data[slot*11u]=bitcast<u32>(page.x);out.data[slot*11u+1u]=id.y;out.data[slot*11u+2u]=level;out.data[slot*11u+3u]=select(0u,bitcast<u32>(nodeSurface.d),p.grid.w!=0u);
 // Canonical dyadic node address: nodes surviving a lattice scale change
 // keep their identity instead of all receiving the new exponent as a key.
 var keyA=ia;var keyB=ib;var exponent=i32(round(log2(s)));
 if(keyA==0&&keyB==0){exponent=0;}else{
  while((keyA&1)==0&&(keyB&1)==0&&exponent<127){keyA/=2;keyB/=2;exponent++;}
 }
 out.data[slot*11u+4u]=bitcast<u32>(keyA);out.data[slot*11u+5u]=bitcast<u32>(keyB);
 out.data[slot*11u+6u]=u32(k)|(u32(exponent+128)<<8u);
 out.data[slot*11u+7u]=bitcast<u32>(i32(round(dot(surfaceNormal,node)*1024.0)));
 out.data[slot*11u+8u]=c.o.x;out.data[slot*11u+9u]=c.o.y;out.data[slot*11u+10u]=materialForeground(page,c.o);
}
`;

export interface ToneMember {readonly x:number;readonly row:number;readonly level:number;readonly atmosphere?:boolean;readonly key?:string;readonly source?:readonly [number,number];readonly foreground?:number}
/** A submission's compiled tone jobs, from their GPU headers. */
export interface ToneRejections {readonly members:number;readonly rejected:number;readonly overflow:boolean}
export interface ToneSunShadow {readonly view:GPUTextureView;readonly sampler:GPUSampler;readonly params:GPUBuffer;readonly eye:readonly number[];readonly forward:readonly number[]}
/** A frame's GPU tone members, read back with that frame's other records. */
export interface ToneReadback {
 /** Already known without a GPU readback (notably the withdrawn producer).
  * This avoids an extra host tick for an empty optional tone input. */
 readonly immediate?:readonly ToneMember[];
 /** Already classified SAME-pose members. Compiled articulation consumes
  * these on its host encoder; only the final native tape reaches the CPU. */
 readonly gpu?:GpuToneMembers;
 collect():Promise<readonly ToneMember[]>;dispose():void;
}
/** Tone-seed consumer ABI (whole-passages-v1).
 *
 * Buffer header (4 u32): count, flags, format, capacity. format 0 =
 * legacy scalar (CPU RegionTone); 1 = whole passages. flags: bit 0
 * overflow; bits 8..31 count the units the fixed clearance rule did not
 * admit (unitClear / edgeClear; legacy toneFits), read by ToneField.rejected.
 * Each record is 11 u32 at 4 + member*11:
 *   0 centreX (f32 bits), 1 native row (17 px), 2 passage index
 *   (TONE_PASSAGES order, 1..7), 3 reverse-Z depth (f32 bits),
 *   4..7 correspondence key. A unit on the surface lattice: 4 node index
 *   on the lattice axis (odd part), 5 cell across the row, 6 cell on the
 *   third axis, 7 band | passage<<3 | ordinal<<11 | axis<<17 |
 *   (exponent+64)<<19 | (cross-row exponent+32)<<26 (WGSL latticeKey).
 *   A texture-edge word (bound-anchored): plane offset, ordinal, passage,
 *   side.
 *   8 source, 9 owner, 10 packed RGBA foreground sampled from the same
 *   source-material radiance at the accepted lattice node. Run identity.yz =
 *   [source & 0xdfffffff, owner].
 * The passage's whole word is grammar word toneWords+index-1 in the
 * compiled role grammar's word table (gpu-articulation seedTone binding 6);
 * pen X = round(centreX - advance/2). The seal places the whole word with
 * its internal paper and 11 px of clearance from neighbouring strokes.
 */
export interface GpuToneMembers {
 readonly buffer:GPUBuffer;readonly capacity:number;readonly field:SurfaceField;
 /** Same eleven-word member layout; column 2 is a whole passage index only in
  * this format. */
 readonly format?:'scalar-v1'|'whole-passages-v1';
 dispose():void;
}
/** The compiled construction class of one resident source, for the
 * construction-owned region decision (planar faces vs rounds vs fields). */
export interface SourceToneClass {readonly domain:number;readonly type:string;
 /** The source's compiled styles word (source-material.ts: bits 0-5 class,
  * liquid, foliage, organic; bits 6-9 the authored material class). */
 readonly styleWord?:number;
 /** The adapter's surface fact (composition.ts CompositionGeometry.surface). */
 readonly surface?:'liquid'|'heightfield';
 /** A foliage crown by the source's compiled geometry (composition.ts
  * CompiledConstructionGraph.foliage): its shade takes the FOLIAGE words. */
 readonly foliage?:boolean;
 /** An organic surface that is not a crown (composition.ts
  * CompiledConstructionGraph.organic): the round's rim band (organicOf). */
 readonly organic?:boolean;
 /** The source's plane regions and their boundary edges (composition.ts
  * compileToneRegions): what darkEdgeDistance reads. */
 readonly toneRegions?:ToneRegions;
 /** REGION POSES: each drawn instance of the source ([its field owner id,
  * worldFromSource]) when no certified source pose reaches the field (an
  * entity draw: its GlobalTransform). Used only to address the plane regions
  * (the shader checks the point lies on the region's plane, so a deformed
  * surface finds no region and is paper); the lattice keeps its frame. */
 readonly instances?:readonly {readonly owner:number;readonly worldFromSource:Float32Array}[]}

/** The tone classes of one camera's resident sources, with each source's
 * entity instances as region poses (SourceToneClass.instances). One query of
 * the drawn entities per job; no per-unit work. */
export function toneClasses(world:World,classes:readonly {domain:number;type:string;styleWord?:number;foliage?:boolean;organic?:boolean;toneRegions?:ToneRegions}[],resident:readonly {draw:{mesh?:unknown};source?:{domain:number}}[]):SourceToneClass[]{
 if(!classes.length)return [];
 const regional=new Set(classes.filter(item=>item.toneRegions).map(item=>item.domain));
 const domains=new Map<unknown,number>();
 for(const {draw,source}of resident)if(source&&draw.mesh!==undefined&&regional.has(source.domain))domains.set(draw.mesh,source.domain);
 const instances=new Map<number,{owner:number;worldFromSource:Float32Array}[]>();
 if(domains.size){
  const meshes=world.storage(MeshInstance).mesh;
  for(const entity of world.query([MeshInstance,GlobalTransform])){
   const domain=domains.get(meshes.get(entity));if(domain===undefined)continue;
   const worldFromSource=new Float32Array(16);composeGlobalTransform(world,entity,worldFromSource);
   const list=instances.get(domain)??[];list.push({owner:entity,worldFromSource});instances.set(domain,list);
  }
 }
 return classes.map(item=>({domain:item.domain,type:item.type,styleWord:item.styleWord,foliage:item.foliage,organic:item.organic,toneRegions:item.toneRegions,instances:item.toneRegions?instances.get(item.domain):undefined}));
}

function decodeToneMembers(words:Uint32Array,count:number,offset:number):readonly ToneMember[]{
 const floats=new Float32Array(words.buffer,words.byteOffset,words.length),members:ToneMember[]=[];
 for(let i=0;i<count;i++){const b=offset+i*11;members.push({x:floats[b]!,row:words[b+1]!,level:words[b+2]!,atmosphere:words[b+3]!==0,key:[words[b+4],words[b+5],words[b+6],words[b+7]].join('/'),source:[words[b+8]!,words[b+9]!],foreground:words[b+10]!});}
 return members;
}

/** Incumbent transfer stays intact until compiled playback passes D2. */
class BufferedToneReadback implements ToneReadback {
 constructor(private readonly buffer:GPUBuffer){}
 async collect():Promise<readonly ToneMember[]>{
  await this.buffer.mapAsync(GPUMapMode.READ);
  try{
   const words=new Uint32Array(this.buffer.getMappedRange());
   return decodeToneMembers(words,Math.min(words[0]!, (words.length-4)/11),4);
  }finally{this.buffer.unmap();this.buffer.destroy();}
 }
 dispose():void{if(this.buffer.mapState==='mapped')this.buffer.unmap();this.buffer.destroy();}
}

/** Retain this job's output on the GPU. Only its count header and emitted
 * prefix cross to the CPU, driven by normal host submissions. No rereading a
 * later live field, own queue submission, selection or glyph changes. */
class CountedToneReadback implements ToneReadback {
 private readonly header:GPUBuffer;private payload?:GPUBuffer;
 private result?:Promise<readonly ToneMember[]>;
 private count=0;private ready=false;private encoded=false;private mapping=false;private disposed=false;
 private resolve?:((members:readonly ToneMember[])=>void);private reject?:((error:unknown)=>void);
 constructor(private readonly device:GPUDevice,private readonly output:GPUBuffer,private readonly capacity:number,private readonly retire:()=>void){
  this.header=device.createBuffer({label:'MAIN tone count header',size:16,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
 }
 encodeHeader(encoder:GPUCommandEncoder):void{encoder.copyBufferToBuffer(this.output,0,this.header,0,16);}
 collect():Promise<readonly ToneMember[]>{return this.result??=this.read().finally(()=>this.dispose());}
 private async read():Promise<readonly ToneMember[]>{
  if(this.disposed)throw Error('Tone readback was retired');
  await this.header.mapAsync(GPUMapMode.READ);
  if(this.disposed)throw Error('Tone readback was retired');
  this.count=new Uint32Array(this.header.getMappedRange())[0]!;this.header.unmap();
  if(this.count>this.capacity)throw Error('Region tone exceeded its complete cell allocation');
  if(!this.count)return [];
  return new Promise((resolve,reject)=>{this.resolve=resolve;this.reject=reject;this.ready=true;});
 }
 encodeReadback(encoder:GPUCommandEncoder):void{
  if(this.disposed||!this.ready||this.encoded)return;
  const size=this.count*44;
  this.payload=this.device.createBuffer({label:'MAIN count-sized tone readback',size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  encoder.copyBufferToBuffer(this.output,16,this.payload,0,size);this.encoded=true;
 }
 submitted():void{
  // Start the header as soon as THIS host submission completes, overlapping
  // relation collection instead of adding a serial count fence after it.
  if(!this.result&&!this.disposed)void this.collect().catch(()=>{});
  if(this.disposed||!this.encoded||this.mapping)return;
  this.mapping=true;const payload=this.payload!;
  void (async()=>{
   try{
    await payload.mapAsync(GPUMapMode.READ);
    if(this.disposed)return;
    const members=decodeToneMembers(new Uint32Array(payload.getMappedRange()),this.count,0);
    payload.unmap();this.resolve?.(members);
   }catch(error){this.reject?.(error);}
  })();
 }
 dispose():void{
  if(this.disposed)return;this.disposed=true;
  this.reject?.(Error('Tone readback was retired'));
  for(const buffer of [this.header,this.payload,this.output])if(buffer){if(buffer.mapState==='mapped')buffer.unmap();buffer.destroy();}
  this.retire();
 }
}


const tonePrograms=new DeviceComputePrograms(device=>{
 const layout=device.createBindGroupLayout({label:'MAIN GPU region tone',entries:[
   {binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:'uniform'}},
   {binding:1,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:'unfilterable-float'}},
   {binding:2,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:'unfilterable-float'}},
   {binding:3,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:'uint'}},
   {binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:'storage'}},
   {binding:5,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:'unfilterable-float'}},
   {binding:6,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:'unfilterable-float'}},
   {binding:7,visibility:GPUShaderStage.COMPUTE,buffer:{type:'storage'}},
   {binding:8,visibility:GPUShaderStage.COMPUTE,buffer:{type:'read-only-storage'}},
   {binding:9,visibility:GPUShaderStage.COMPUTE,buffer:{type:'uniform'}},
   {binding:10,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:'depth'}},
   {binding:11,visibility:GPUShaderStage.COMPUTE,sampler:{type:'comparison'}},
   {binding:12,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:'unfilterable-float'}},
   {binding:13,visibility:GPUShaderStage.COMPUTE,buffer:{type:'read-only-storage'}},
   {binding:14,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:'uint'}},
   {binding:15,visibility:GPUShaderStage.COMPUTE,buffer:{type:'read-only-storage'}},
   {binding:16,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:'unfilterable-float'}},
   {binding:17,visibility:GPUShaderStage.COMPUTE,buffer:{type:'read-only-storage'}},
   {binding:18,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:'uint'}}]});
 const rowLayout=device.createBindGroupLayout({label:'MAIN tone row tables',entries:[{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:'storage'}}]});
 const orderLayout=device.createBindGroupLayout({label:'MAIN tone canonical member order',entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:'uniform'}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:'read-only-storage'}},
   ...[2,3,4].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:'storage' as const}}))]});
 const main=device.createPipelineLayout({bindGroupLayouts:[layout]});
 const row=device.createPipelineLayout({bindGroupLayouts:[layout,rowLayout]});
 const order=device.createPipelineLayout({bindGroupLayouts:[orderLayout]});
 const plans=new Map<string,ComputePipelinePlan>();
 for(const [entryPoint,pipeline,label]of [
  ['main',main,'MAIN GPU region tone'],['litReference',main,'MAIN receiver-plane lit references'],
  ['rowWalk',row,'MAIN tone row texture edges'],['passages',row,'MAIN tone lattice units'],['seaCells',row,'MAIN tone 10-12 sea carriers'],
 ] as const){plans.set(entryPoint,()=>{const module=device.createShaderModule({label,code:isolateComputeEntry(TONE_SHADER,entryPoint)});return {label,layout:pipeline,compute:{module,entryPoint}};});}
 for(const [entryPoint,label]of [
  ['countRows','MAIN tone member rows'],['startRows','MAIN tone member row starts'],
  ['scatterRows','MAIN tone members into rows'],['rankRows','MAIN tone members by surface key'],
 ] as const){plans.set(entryPoint,()=>{const module=device.createShaderModule({label,code:isolateComputeEntry(TONE_ORDER_SHADER,entryPoint)});return {label,layout:order,compute:{module,entryPoint}};});}
 return {resources:{layout,rowLayout,orderLayout},plans};
});
export async function prepareToneFieldPrograms(device:GPUDevice,progress?:ComputePipelineProgress):Promise<void>{await tonePrograms.prepare(device,progress);}

export class ToneField {
 private readonly pipeline:GPUComputePipeline;private readonly reference:GPUComputePipeline;private readonly passages:GPUComputePipeline;private readonly seaCells:GPUComputePipeline;private readonly rowPrelude:GPUComputePipeline;private readonly layout:GPUBindGroupLayout;
 private pending:GPUBuffer[]=[];
 private readonly readbacks=new Set<CountedToneReadback>();
 // Per-job inputs, kept across frames and grown only: slot n serves the nth
 // job encoded since the last submission.
 private readonly scratch=new Map<string,GPUBuffer[]>();private readonly scratchUsed=new Map<string,number>();
 // Sixteen-byte copies of compiled jobs' headers, mapped after submission.
 private readonly idleCounters:GPUBuffer[]=[];private encodedCounters:GPUBuffer[]=[];
 private counted?:ToneRejections;
 /** The compiled jobs of the latest submission whose headers have arrived:
  * placed members, units the fit check rejected (their ink left the region
  * or met a line) and whether the member allocation overflowed. Undefined
  * until the first header arrives. A GPU count, one or two frames late. */
 get rejected():ToneRejections|undefined{return this.counted;}
 private reuse(kind:string,label:string,size:number,usage:number,data?:ArrayBufferView):GPUBuffer{
  const list=this.scratch.get(kind)??[],index=this.scratchUsed.get(kind)??0;this.scratch.set(kind,list);this.scratchUsed.set(kind,index+1);
  let buffer=list[index];
  if(!buffer||buffer.size<size){
   if(buffer)this.pending.push(buffer);
   buffer=this.device.createBuffer({label,size:Math.ceil(size*(buffer?1.5:1)/4)*4,usage:usage|GPUBufferUsage.COPY_DST});list[index]=buffer;
  }
  if(data)this.device.queue.writeBuffer(buffer,0,data.buffer,data.byteOffset,data.byteLength);
  return buffer;
 }
 private readonly noParticles:GPUBuffer;
 private readonly noSun:GPUBuffer;private readonly noSunMap:GPUTexture;private readonly sunSampler:GPUSampler;
 private readonly rowLayout:GPUBindGroupLayout;
 private readonly orderLayout:GPUBindGroupLayout;private readonly order:Record<'countRows'|'startRows'|'scatterRows'|'rankRows',GPUComputePipeline>;
 // The per-row texture-edge tables (rowPrelude → passages), ROW_STATE words
 // per tone row, grown only; jobs encoded in order reuse them.
 private rows?:{buffer:GPUBuffer;count:number;group:GPUBindGroup};
 private rowTables(rows:number){
  const have=this.rows;if(have&&have.count>=rows)return have;
  if(have)this.retiredBuffers.push(have.buffer);
  const count=Math.max(rows,have?.count??0);
  const buffer=this.device.createBuffer({label:'MAIN tone row texture-edge tables',size:count*ROW_STATE*4,usage:GPUBufferUsage.STORAGE});
  this.rows={buffer,count,group:this.device.createBindGroup({layout:this.rowLayout,entries:[{binding:2,resource:{buffer}}]})};
  return this.rows;
 }
 private retiredBuffers:GPUBuffer[]=[];
 // The material-ink table texture (binding 18; gpu-articulation MATERIAL INK).
 // A replaced texture is kept (an encoded job may still bind it).
 private inkTexture?:GPUTexture;private inkLast?:Uint32Array;private readonly inkRetired:GPUTexture[]=[];
 private materialInkView(table:Uint32Array):GPUTextureView{
  const {width,height,data}=materialInkTexels(table);
  let fresh=false;
  if(!this.inkTexture||this.inkTexture.width!==width||this.inkTexture.height!==height){if(this.inkTexture)this.inkRetired.push(this.inkTexture);this.inkTexture=this.device.createTexture({label:'MAIN tone material inks',size:[width,height],format:'rg32uint',usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});fresh=true;}
  if(fresh||!this.inkLast||this.inkLast.length!==table.length||table.some((v,i)=>v!==this.inkLast![i])){this.device.queue.writeTexture({texture:this.inkTexture},data,{bytesPerRow:width*8},[width,height]);this.inkLast=table.slice();}
  return this.inkTexture.createView();
 }
 // THE PLANE REGION TABLE (WGSL binding 15, darkEdgeDistance): the resident
 // sources' compiled plane regions and boundary edges, packed only when the
 // set of sources or a source's compiled table changes (each table is
 // compiled once per source revision; nothing here is per frame).
 // Layout in vec4 words: [sourceCount, regionBase, edgeBase, tableBase], one
 // entry per source sorted by domain [domain, firstRegion, regionCount, its
 // offset bin step (f32)], then two words per region, three per edge
 // (composition.ts ToneRegions), then THE REGION LOOKUP (WGSL regionOf): one
 // word [slot count], then two [key, absolute region] slots per vec4,
 // open addressing at a load of at most a quarter. The old two padding
 // words per slot made Numabe's lookup alone consume a full 128 MiB binding.
 private planes?:{key:readonly (ToneRegions|undefined)[];domains:readonly number[];buffer:GPUBuffer;size:number};
 private planeTable(ordered:readonly SourceToneClass[]):{buffer:GPUBuffer;size:number}{
  const listed=ordered.filter(item=>item.toneRegions&&item.toneRegions.regions.length);
  const have=this.planes;
  if(have&&have.key.length===listed.length&&listed.every((item,i)=>have.key[i]===item.toneRegions&&have.domains[i]===item.domain))return have;
  const regionCount=listed.reduce((n,item)=>n+item.toneRegions!.regions.length/8,0),edgeCount=listed.reduce((n,item)=>n+item.toneRegions!.edges.length/12,0);
  // Every region's keys: each bin its exact-match tolerance reaches, under
  // both signs of its plane (the shader's normal faces the viewer).
  const entries:[number,number][]=[],steps:number[]=[];
  let region=0;
  listed.forEach(item=>{
   const table=item.toneRegions!,count=table.regions.length/8;
   let radius=0;for(let r=0;r<count;r++)radius=Math.max(radius,table.regions[r*8+6]!);
   const step=Math.fround(Math.max(radius/64,1e-4));steps.push(step);
   for(let r=0;r<count;r++){
    const n=[table.regions[r*8]!,table.regions[r*8+1]!,table.regions[r*8+2]!],d=table.regions[r*8+3]!,tolerance=.02*table.regions[r*8+6]!+1e-4;
    for(const sign of [1,-1]){
     const keys=new Set<number>();
     const [ex,ey]=octEncode(n[0]!*sign,n[1]!*sign,n[2]!*sign);
     for(const du of [-REGION_OCT_TOLERANCE,0,REGION_OCT_TOLERANCE])for(const dv of [-REGION_OCT_TOLERANCE,0,REGION_OCT_TOLERANCE]){
      const bx=Math.min(REGION_OCT-1,Math.max(0,Math.floor(((ex+du)*.5+.5)*REGION_OCT))),by=Math.min(REGION_OCT-1,Math.max(0,Math.floor(((ey+dv)*.5+.5)*REGION_OCT)));
      for(let o=Math.floor((d*sign-tolerance)/step);o<=Math.floor((d*sign+tolerance)/step);o++)keys.add(regionKey(item.domain>>>0,bx,by,o));
     }
     for(const key of keys)entries.push([key,region+r]);
    }
   }
   region+=count;
  });
  let slots=1;while(slots<entries.length*4)slots*=2;
  const slot=new Uint32Array(slots*2);
  for(const [key,index]of entries){
   for(let k=0;k<slots;k++){const at=((key+k)&(slots-1))*2;if(slot[at]===0){slot[at]=key;slot[at+1]=index;break;}if(slot[at]===key&&slot[at+1]===index)break;}
  }
  const regionBase=1+listed.length,edgeBase=regionBase+regionCount*2,tableBase=edgeBase+edgeCount*3,words=tableBase+1+Math.ceil(slots/2);
  const data=new Float32Array(words*4),bits=new Uint32Array(data.buffer);
  bits.set([listed.length,regionBase,edgeBase,tableBase],0);
  bits[tableBase*4]=slots;
  bits.set(slot,(tableBase+1)*4);
  region=0;let edge=0;
  listed.forEach((item,i)=>{
   const table=item.toneRegions!,count=table.regions.length/8;
   bits.set([item.domain>>>0,region,count],(1+i)*4);data[(1+i)*4+3]=steps[i]!;
   for(let r=0;r<count;r++){
    const at=(regionBase+(region+r)*2)*4;
    data.set(table.regions.subarray(r*8,r*8+4),at);
    // First edge and edge count, made absolute; the radius.
    data.set([table.regions[r*8+4]!+edge,table.regions[r*8+5]!,table.regions[r*8+6]!,0],at+4);
   }
   data.set(table.edges,(edgeBase+edge*3)*4);
   region+=count;edge+=table.edges.length/12;
  });
  if(have)this.retiredBuffers.push(have.buffer);
  const buffer=this.device.createBuffer({label:'MAIN tone plane regions',size:data.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
  this.device.queue.writeBuffer(buffer,0,data);
  this.planes={key:listed.map(item=>item.toneRegions),domains:listed.map(item=>item.domain),buffer,size:data.byteLength};
  return this.planes;
 }
 constructor(private readonly device:GPUDevice){
  const ready=tonePrograms.get(device),pipelines=ready.pipelines;
  this.noParticles=device.createBuffer({label:'MAIN legacy particle coverage format',size:4,usage:GPUBufferUsage.STORAGE});
  this.noSun=device.createBuffer({label:'MAIN no sun caster',size:432,usage:GPUBufferUsage.UNIFORM});
  this.noSunMap=device.createTexture({label:'MAIN no sun map',size:[1,1],format:'depth32float',usage:GPUTextureUsage.TEXTURE_BINDING});
  this.sunSampler=device.createSampler({compare:'greater-equal',minFilter:'linear',magFilter:'linear'});
  this.layout=ready.resources.layout;this.rowLayout=ready.resources.rowLayout;this.orderLayout=ready.resources.orderLayout;
  this.pipeline=pipelines.get('main')!;this.reference=pipelines.get('litReference')!;
  this.rowPrelude=pipelines.get('rowWalk')!;this.passages=pipelines.get('passages')!;this.seaCells=pipelines.get('seaCells')!;
  this.order={countRows:pipelines.get('countRows')!,startRows:pipelines.get('startRows')!,scatterRows:pipelines.get('scatterRows')!,rankRows:pipelines.get('rankRows')!};
 }
 /** Encode this view's tone; returns the member readback for publish.
  * Compiled playback takes the resident sources' construction classes and
  * emits whole passages at the surface lattice's nodes (whole-passages-v1). */
 encode(encoder:GPUCommandEncoder,field:SurfaceField,clipFromWorld:ArrayLike<number>,rect:{x:number;y:number;width:number;height:number},pageWidth:number,pageHeight:number,light?:readonly number[],appearance?:GPUTexture,compiled=false,shadow?:ToneSunShadow,construction?:GPUTexture,classes:readonly SourceToneClass[]=[],fog?:readonly number[],inks?:Uint32Array):ToneReadback{
  // The rejected whole-plane classifier stays withdrawn. Compiled playback
  // requires this pose's construction and an actual scene light/caster.
  if(compiled&&(!construction||(!shadow&&!light)))return {immediate:[],collect:async()=>[],dispose(){}};
  const cols=Math.ceil(pageWidth/TONE_CELL),rows=Math.ceil(pageHeight/TONE_ROW);
  // Complete geometric allocation bound, not an admission quota: a row holds
  // at most one passage per narrowest passage advance; legacy one per cell.
  const narrowest=Math.max(1,Math.min(...TONE_PASSAGES.slice(1).map((_,i)=>passages.headers[(i+1)*8+1]!)));
  const capacity=compiled?Math.max(1,rows*Math.ceil(pageWidth/narrowest)):TONE_CAPACITY,size=16+capacity*44;
  const ordered=[...classes].sort((a,b)=>a.domain-b.domain),styleData=new Uint32Array(Math.max(1,ordered.length)*2);
  // On request only (a review tool sets globalThis.__mainToneClassesRequest):
  // the tone class of every resident source, to list which are crowns.
  const probe=globalThis as {__mainToneClassesRequest?:unknown;__mainToneClasses?:unknown};
  if(probe.__mainToneClassesRequest){probe.__mainToneClassesRequest=undefined;probe.__mainToneClasses=ordered.map(item=>({domain:item.domain,type:item.type,styleWord:item.styleWord,foliage:!!item.foliage,organic:!!item.organic,id:(item as {complex?:{id?:string}}).complex?.id}));}
  // Bit 3 marks a liquid surface (the adapter's fact, liquidOf): its ground
  // field takes the water word (levelUnits); bit 4 a foliage crown (the
  // source's compiled geometry class, composition.ts foliage; foliageOf);
  // bit 5 an organic non-crown surface (organicOf). Its class (styleOf) is
  // unchanged by any.
  // One owner of the word: source-material.ts sourceToneStyleWord (bit 3 the
  // adapter's liquid fact, bit 4 foliage, bit 5 organic, bits 6-9 material).
  ordered.forEach((item,index)=>styleData.set([item.domain,sourceToneStyleWord(item)],index*2));
  if(size>Math.min(this.device.limits.maxStorageBufferBindingSize,this.device.limits.maxBufferSize))throw Error('Complete region tone needs device-sized page chunks');
  const styles=this.reuse('styles','MAIN tone source construction classes',styleData.byteLength,GPUBufferUsage.STORAGE,styleData);
  const planes=this.planeTable(ordered);
  // Certified source poses (the field's attachments) first; a source's entity
  // instances join as REGION POSES (word 34 = 1: the shader's
  // latticeAttachment never takes them as the lattice frame) where no
  // certified pose has that owner. The field owner of an entity draw is
  // [domain, entity] (field-stage.ts mainOwner).
  const certified=compiled?[...field.attachments??[]]:[];
  const owned=new Set(certified.map(item=>`${item.source[0]&0xdfffffff}/${item.source[1]}`));
  const posed:{source:readonly [number,number];worldFromSource:Float32Array;sourceFromWorld:Float32Array;region:boolean}[]=certified.map(item=>({...item,region:false}));
  if(compiled)for(const item of classes)for(const instance of item.instances??[]){
   const source=[item.domain>>>0,instance.owner>>>0] as const;if(owned.has(`${source[0]&0xdfffffff}/${source[1]}`))continue;
   let sourceFromWorld:Float32Array;try{sourceFromWorld=inverseMatrix(instance.worldFromSource);}catch{continue;}
   owned.add(`${source[0]&0xdfffffff}/${source[1]}`);posed.push({source,worldFromSource:instance.worldFromSource,sourceFromWorld,region:true});
  }
  // Sorted by source address for attachmentFor's binary search (no source
  // here carries the UV-contract bit 29 that its key clears).
  const attached=posed.sort((a,b)=>(a.source[0]>>>0)-(b.source[0]>>>0)||(a.source[1]>>>0)-(b.source[1]>>>0));
  const attachmentData=new Float32Array(Math.max(1,attached.length)*36),attachmentWords=new Uint32Array(attachmentData.buffer);
  for(const [index,attachment]of attached.entries()){
   attachmentData.set(attachment.worldFromSource,index*36);attachmentData.set(attachment.sourceFromWorld,index*36+16);attachmentWords.set(attachment.source,index*36+32);attachmentWords[index*36+34]=attachment.region?1:0;
  }
  // Read by the default path too: a unit's lattice is its owner's source
  // frame where the producer attached one (WGSL latticeRow).
  const attachments=this.reuse('attachments','MAIN certified tone attachment poses',attachmentData.byteLength,GPUBufferUsage.STORAGE,attachmentData);
  const f=new Float32Array(64);
  f.set(field.worldFromClip,0);f.set(Array.from(clipFromWorld).slice(0,16),16);
  f.set([rect.x,rect.y,rect.width,rect.height,field.width,field.height,pageWidth,pageHeight],32);
  f.set(light?[light[0]!,light[1]!,light[2]!,1]:[0,0,0,0],40);
  const u=new Uint32Array(f.buffer);u.set([cols,rows,capacity,Number(compiled)],44);u[48]=attached.length;u[49]=Number(!!shadow);u[50]=ordered.length;
  if(shadow){f.set(shadow.eye.slice(0,3),52);f.set(shadow.forward.slice(0,3),56);}
  f.set(fog?[fog[0]!,fog[1]!,fog[2]!,0]:[0,0,0,0],60);
  const params=this.reuse('params','MAIN GPU region tone view',256,GPUBufferUsage.UNIFORM,f);
  // The frame's material inks (MATERIAL INK), shared with the line words.
  const inkView=this.materialInkView(inks&&inks.length?inks:new Uint32Array(2));
  // The member list belongs to its consumer until that job retires, so it
  // is the one per-job allocation left (and the bind group, whose field
  // views are the job's own).
  const out=this.device.createBuffer({label:'MAIN GPU region tone members',size,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
  // The legacy receiver-plane table (litReference), a power of two of
  // 48-byte slots. Compiled playback reads no table: its binding is a small
  // placeholder, never cleared or written.
  const referenceSize=(compiled?64:65536)*48;
  const reference=this.reuse(compiled?'reference':'legacy-reference','MAIN receiver-plane lit values',referenceSize,GPUBufferUsage.STORAGE);if(!compiled)encoder.clearBuffer(reference,0,referenceSize);
  encoder.clearBuffer(out,0,16);
  const group=this.device.createBindGroup({layout:this.layout,entries:[{binding:0,resource:{buffer:params,size:256}},{binding:1,resource:field.textures[0]!.createView()},
   {binding:2,resource:field.textures[1]!.createView()},{binding:3,resource:field.textures[3]!.createView()},{binding:4,resource:{buffer:out}},
   {binding:5,resource:(appearance??field.textures[4]!).createView()},{binding:6,resource:field.textures[4]!.createView()},
   {binding:7,resource:{buffer:reference,size:referenceSize}},{binding:8,resource:{buffer:attachments,size:attachmentData.byteLength}},
   {binding:9,resource:{buffer:shadow?.params??this.noSun}},{binding:10,resource:shadow?.view??this.noSunMap.createView()},{binding:11,resource:shadow?.sampler??this.sunSampler},
   {binding:12,resource:(construction??field.textures[4]!).createView()},
   {binding:13,resource:{buffer:styles,size:styleData.byteLength}},
   {binding:14,resource:formDepths(this.device).view(field)},
   {binding:15,resource:{buffer:planes.buffer,size:planes.size}},
   {binding:16,resource:(field.textures[2]??field.textures[4]!).createView()},
   {binding:17,resource:{buffer:field.emissiveCoverage??this.noParticles}},
   {binding:18,resource:inkView}]});
  if(compiled){
   // The row texture-edge tables first (one thread per row), then one
   // thread per page pixel, of which the pixel holding a lattice node places
   // that node's unit. Two dispatches in one pass: the second sees the
   // first's writes.
   const tables=this.rowTables(rows);
   // Two passes (labelled apart for the frame profile): the serial row walk

   const prelude=encoder.beginComputePass({label:'MAIN tone row texture edges'});prelude.setBindGroup(0,group);prelude.setBindGroup(1,tables.group);
   // One workgroup per row (WGSL rowWalk).
   prelude.setPipeline(this.rowPrelude);prelude.dispatchWorkgroups(rows);prelude.end();
   const pass=encoder.beginComputePass({label:'MAIN tone lattice units'});pass.setBindGroup(0,group);pass.setBindGroup(1,tables.group);
   // One thread per field column (page pixels per column: WGSL colStep).
   const step=Math.max(1,Math.round(pageWidth/field.width));
   pass.setPipeline(this.passages);pass.dispatchWorkgroups(Math.ceil(Math.ceil(pageWidth)/step/64),rows);
   // sea-100: one thread per carrier cell of the 10-12 sea (WGSL seaCells, SEA_SPAN 384).
   pass.setPipeline(this.seaCells);pass.dispatchWorkgroups(384/64,384);pass.end();
   // The header (members, rejected units) follows this job's own commands.
   const counter=this.idleCounters.pop()??this.device.createBuffer({label:'MAIN tone rejected-unit count',size:16,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
   encoder.copyBufferToBuffer(out,0,counter,0,16);this.encodedCounters.push(counter);
   // CANONICAL MEMBER ORDER (TONE_ORDER_SHADER): the seed reads `sorted`,
   // rows in order and each row by surface key; `out` keeps the appended list.
   const sorted=this.device.createBuffer({label:'MAIN GPU region tone members by surface key',size,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
   const rowWords=this.reuse('order-rows','MAIN tone member row counts',Math.max(16,rows*12),GPUBufferUsage.STORAGE);encoder.clearBuffer(rowWords,0,Math.max(16,rows*12));
   const scattered=this.reuse('order-scattered','MAIN tone members grouped by row',Math.max(16,capacity*44),GPUBufferUsage.STORAGE);
   const orderUniform=this.reuse('order-uniform','MAIN tone member order view',16,GPUBufferUsage.UNIFORM,new Uint32Array([rows,capacity,0,0]));
   const orderGroup=this.device.createBindGroup({layout:this.orderLayout,entries:[orderUniform,out,rowWords,scattered,sorted].map((buffer,binding)=>({binding,resource:{buffer}}))});
   const memberGroups=Math.ceil(capacity/64);
   if(memberGroups>this.device.limits.maxComputeWorkgroupsPerDimension)throw Error('Canonical tone order needs member-count chunks');
   const ordering=encoder.beginComputePass({label:'MAIN tone canonical member order'});ordering.setBindGroup(0,orderGroup);
   ordering.setPipeline(this.order.countRows);ordering.dispatchWorkgroups(memberGroups);
   ordering.setPipeline(this.order.startRows);ordering.dispatchWorkgroups(1);
   ordering.setPipeline(this.order.scatterRows);ordering.dispatchWorkgroups(memberGroups);
   ordering.setPipeline(this.order.rankRows);ordering.dispatchWorkgroups(memberGroups);ordering.end();
   let disposed=false;const dispose=()=>{if(disposed)return;disposed=true;out.destroy();sorted.destroy();};
   return {immediate:[],gpu:{buffer:sorted,capacity,field,format:'whole-passages-v1',dispose},collect:async()=>[],dispose};
  }
  for(const pipeline of [this.reference,this.pipeline]){const pass=encoder.beginComputePass({label:'MAIN GPU region tone and shadow'});pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(cols/8),Math.ceil(rows/8));pass.end();}
  this.pending.push(out);
  const read=this.device.createBuffer({label:'MAIN GPU region tone readback',size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  encoder.copyBufferToBuffer(out,0,read,0,size);
  return new BufferedToneReadback(read);
 }
 encodeReadbacks(encoder:GPUCommandEncoder):void{for(const readback of this.readbacks)readback.encodeReadback(encoder);}
 submitted():void{
  for(const buffer of this.pending)buffer.destroy();this.pending=[];this.scratchUsed.clear();
  for(const buffer of this.retiredBuffers.splice(0))buffer.destroy();
  for(const readback of this.readbacks)readback.submitted();
  const counters=this.encodedCounters;this.encodedCounters=[];
  if(counters.length)void Promise.all(counters.map(async counter=>{
   await counter.mapAsync(GPUMapMode.READ);
   const words=new Uint32Array(counter.getMappedRange().slice(0));counter.unmap();this.idleCounters.push(counter);
   return words;
  })).then(headers=>{
   this.counted={members:headers.reduce((n,words)=>n+words[0]!,0),rejected:headers.reduce((n,words)=>n+(words[1]!>>>8),0),overflow:headers.some(words=>(words[1]!&1)!==0)};
  },()=>{for(const counter of counters)if(!this.idleCounters.includes(counter))counter.destroy();});
 }
}
const instances=new WeakMap<GPUDevice,ToneField>();
export function toneField(device:GPUDevice):ToneField{let t=instances.get(device);if(!t){t=new ToneField(device);instances.set(device,t);}return t;}

/** Field view carrying its frame's GPU tone members. */
export type TonedFieldView=DrawingFieldView&{readonly tone?:readonly ToneMember[]};

/** Place the GPU members into the page as one region-tone neighborhood:
 * only on surfaces whose construction was admitted (tone is bounded by
 * constructed lines), one full space clear of every stroke, never on
 * protected paper, with legal native whitespace on both sides. */
export function placeRegionTone(views:readonly DrawingFieldView[],groups:readonly NativeNeighborhood[],served:ReadonlySet<string>,
 font:ReadonlyMap<string,NativeGlyph>,width:number,height:number,whitespace:NativeWhitespacePolicy):{owner:string;group:NativeNeighborhood}[]{
 const rows=new Map<number,[number,number][]>(),papers:NativePaper[]=[];
 for(const group of groups){
  papers.push(...group.paper);
  for(const atom of group.occurrences){const g=font.get(atom.text);if(!g)continue;const list=rows.get(atom.row)??[];list.push([atom.x,atom.x+g.advance]);rows.set(atom.row,list);}
 }
 const byOwner=new Map<string,NativeOccurrence[]>();
 for(const view of views as readonly TonedFieldView[]){
  if(!view.tone?.length)continue;
  const {field,rect,owners,camera}=view;
  for(const member of [...view.tone].sort((a,b)=>a.row-b.row||a.x-b.x)){
   const text=TONE_LADDER[member.level],glyph=text&&font.get(text);if(!glyph||!glyph.points.length)continue;
   const fx=(member.x-rect.x)/rect.width*field.width,fy=((member.row+.5)*TONE_ROW-rect.y)/rect.height*field.height;
   const sample=member.atmosphere?undefined:field.sample(fx,fy);if(!member.atmosphere&&(!sample||sample.depth<=0))continue;
   const reference=sample&&(sample.kind==='entity'?owners.get(sample.owner):sample.owner);
   const owner=member.atmosphere?`${camera}/atmosphere`:`${camera}/${sample!.source}/${sample!.kind}/${reference}`;if(!member.atmosphere&&!served.has(owner))continue;
   const list=rows.get(member.row)??[];
   const base=Math.round(member.x-glyph.advance/2);let placed:number|undefined;
   for(const d of [0,1,-1,2,-2]){
    const x=base+d,end=x+glyph.advance;if(x<0||end>width||member.row*TONE_ROW+TONE_ROW>height)continue;
    let left=0,right=Infinity,clash=false;
    for(const [a,b]of list){if(b<=x)left=Math.max(left,b);else if(a>=end)right=Math.min(right,a);else{clash=true;break;}}
    if(clash||x-left<TONE_CLEAR&&left>0||right-end<TONE_CLEAR)continue;
    if(!nativeSpaceCodes(x-left,0,left===0,false,whitespace)||Number.isFinite(right)&&!nativeSpaceCodes(right-end,0,false,false,whitespace))continue;
    if(papers.some(paper=>glyph.points.some(q=>nativePaperContains(paper,x+q[0]!+.5,member.row*TONE_ROW+q[1]!+.5))))continue;
    placed=x;break;
   }
   if(placed===undefined)continue;
   list.push([placed,placed+glyph.advance]);rows.set(member.row,list);
   const occurrences=byOwner.get(owner)??[];byOwner.set(owner,occurrences);
   occurrences.push({id:`${owner}/gpu-tone/${member.key??`${member.row}/${placed}`}`,text,x:placed,row:member.row,
    contributions:[{source:`${owner}/tone`,kind:member.atmosphere?'light':'region',pixels:glyph.points.map((_,i)=>i)}]});
  }
 }
 return [...byOwner].map(([owner,occurrences])=>({owner,group:{id:`${owner}/gpu-tone`,occurrences,contacts:[],paper:[],requiredSources:[]}}));
}
