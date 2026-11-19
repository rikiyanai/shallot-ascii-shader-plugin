import type {NativeGlyph} from './native-page.ts';
import bank from '../../assets/grammar/aahub-tone-passages-3493185.v1.json' with {type:'json'};

/** Retained whole passages, reference-bound: compiled by
 * `tools/compile-main-tone-passages.ts` from the pinned role lookup's RETAINED
 * tone_core / tone_mid / tone_fringe / cast_shadow_edge witnesses (rejected
 * entries excluded by their verdict) and the hand-authored targets. Each entry
 * carries its source id, status and context (wall / round / ground; the
 * appended canopy edge words and rock / grass / foliage material bands below
 * cite their AAHub rows). Frequency is not a criterion. Index 0 is paper. */
export type ToneContext='wall'|'round'|'ground'|'canopy'|'rock'|'grass'|'foliage';
export interface TonePassage {readonly id:string;readonly role:'tone_core'|'tone_mid'|'tone_fringe'|'tone_deep'|'contact'|'texture_edge';readonly context:ToneContext;readonly text:string;readonly source:string;readonly status:string;readonly paper:string}
/** Texture-edge words: the envelope of a region-led form whose silhouette is
 * finer than a glyph (TEXTURE_EDGE_MEASURE) is spelled by these, one constant
 * word per boundary side, instead of a path-led outline. Hand-authored
 * originals, opened as picture and text: T#7 (tree-007, labelled "minimal
 * glyphs showing branches going up into leaf": crown rows `ヾ ; ;";;/" ; ;ヾ`,
 * `"ゞ ; ; ; ゞ ; ;ヾ ; ; ヾ ;ゞ`, top row `" " ヾ　; " ; " ; ; ヾ`, underside
 * `""`) and C#8 (labelled "a defined underside line": `ﾞ'''''';'ﾚ;;;`).
 * Appended after every bank passage, so no bank index moves. */
const TEXTURE_EDGE_PASSAGES:readonly TonePassage[]=[
 {id:'canopy-left',role:'texture_edge',context:'canopy',text:'ヾ ;',source:'T#7 9636c2401facd04b90778f4bfea720c3 aa[7] crown row 3',status:'retained: operator label yes, branch entry into foliage',paper:''},
 {id:'canopy-right',role:'texture_edge',context:'canopy',text:'; ゞ',source:'T#7 9636c2401facd04b90778f4bfea720c3 aa[7] crown row 2',status:'retained: operator label yes, branch entry into foliage',paper:''},
 {id:'canopy-top',role:'texture_edge',context:'canopy',text:'" ;',source:'T#7 9636c2401facd04b90778f4bfea720c3 aa[7] crown row 1',status:'retained: operator label yes, branch entry into foliage',paper:''},
 {id:'canopy-underside',role:'texture_edge',context:'canopy',text:"'''",source:'C#8 2f0353dc8e23bde120d1c050746f937c aa[8] canopy underside row',status:'retained: hand label CANOPY, a defined underside line',paper:''},
 {id:'canopy-tip',role:'texture_edge',context:'canopy',text:'"',source:'T#7 9636c2401facd04b90778f4bfea720c3 aa[7] crown underside',status:'retained: operator label yes, branch entry into foliage',paper:''},
];
/** The even ground field's single member. The operator markup
 * (apps/stroke-matcher/references/review-untitled-24.png, lower-right and
 * medium-far insets) fills open ground with one colon at a constant pitch on
 * every row, `: : : : :`; the bank's `: : :` word cannot hold that pitch
 * across a surface-lattice cell of any other width, so the member is placed
 * singly. Appended after every other passage, so no index moves. */
const GROUND_MEMBER_PASSAGES:readonly TonePassage[]=[
 {id:'ground-member',role:'tone_fringe',context:'ground',text:':',source:'operator markup review-untitled-24.png, lower-right inset ground rows and medium-far inset field',status:'retained: operator markup, EXAMPLE OF MEDIUM FAR LOD',paper:''},
];
/** MATERIAL PASSAGES: the region tone of the three materials the bank's
 * wall / round / ground witnesses do not draw, each band a word copied
 * verbatim from one row of an AAHub reference whose picture shows that
 * material there. Archive root
 * /Users/r/Projects/ascii-art-archive/collections/aahub-y8; a source names the
 * page folder, the piece file and its 1-based line. Within a context the
 * bands are ordered by ink per advance (checked in compileTonePassages):
 * fringe < mid < core < deep.
 *   rock    cliff and rock faces: the lit facet's spaced colons, the cliff
 *           top's comma-colon, the column face's semicolon-colon, the
 *           shadowed foot's hatching.
 *   grass   ground cover on a slope: the blade textures of the grassy hills
 *           (aa-0028's caption: "grassy hills ... drawn with ﾞ;ﾞ;ﾞ;ﾞ and
 *           ヾ;:ﾐ: blade textures") and the turf at a tree's foot.
 *   foliage dense crown mass: the leaf words of a crown and of a forest
 *           canopy seen from above.
 * Appended after every other passage, so no earlier index moves. */
const R='3c33aa91ad5a95d68ec09a85919643be',T='9636c2401facd04b90778f4bfea720c3',M='1823425068f1993743f2308e797c2c2c';
const MATERIAL_PASSAGES:readonly TonePassage[]=[
 {id:'rock-fringe',role:'tone_fringe',context:'rock',text:': : :',source:`AAHub ${R} aa-0035.txt line 15 (rock arch, the facet under the arch)`,status:'retained: verbatim reference row, lit rock facet',paper:'spaced members, one half space apart'},
 {id:'rock-mid',role:'tone_mid',context:'rock',text:',:,:,:',source:`AAHub ${R} aa-0108.txt line 1 (jagged coastal rocks, the cliff top)`,status:'retained: verbatim reference row, half-lit cliff face',paper:''},
 {id:'rock-core',role:'tone_core',context:'rock',text:';:;:;:',source:`AAHub ${R} aa-0036.txt line 8 (columnar cliff, the column faces |;:;:;:;:;:;:/)`,status:'retained: verbatim reference row, shaded cliff face',paper:''},
 {id:'rock-deep',role:'tone_deep',context:'rock',text:';;:;;',source:`AAHub ${R} aa-0035.txt line 15 (rock arch, the shaded facet ;;:;; under the arch, rows 15-17)`,status:'retained: verbatim reference row, rock face turned from the light; replaces /// (bar hatching, banned in tone and a lone slash on rock: operator 2026-11-10)',paper:''},
 {id:'grass-fringe',role:'tone_fringe',context:'grass',text:':ﾞ:ﾞ:ﾞ',source:`AAHub ${T} aa-0068.txt line 21 (石割桜, the turf at the tree's foot)`,status:'retained: verbatim reference row, lit turf',paper:''},
 {id:'grass-mid',role:'tone_mid',context:'grass',text:'ﾞ;ﾞ;ﾞ;',source:`AAHub ${R} aa-0028.txt line 3 (海岸の風景, the grassy hill)`,status:'retained: verbatim reference row, grass blade texture',paper:''},
 {id:'grass-core',role:'tone_core',context:'grass',text:'ヾ;:ﾐ:',source:`AAHub ${R} aa-0028.txt line 6 (海岸の風景, the grassy hill's shaded fold)`,status:'retained: verbatim reference row, grass blade cluster',paper:''},
 {id:'grass-deep',role:'tone_deep',context:'grass',text:'ﾐ;ﾐ;ﾐ;',source:`AAHub ${R} aa-0028.txt line 19 (海岸の風景, the hill's dark lower slope)`,status:'retained: verbatim reference row, shaded grass mass',paper:''},
 {id:'foliage-fringe',role:'tone_fringe',context:'foliage',text:'; ; ;',source:`AAHub ${T} aa-0007.txt line 2 (T#7 tree, the crown's open interior)`,status:'retained: operator label yes on T#7, branch entry into foliage',paper:'spaced members, one half space apart'},
 {id:'foliage-mid',role:'tone_mid',context:'foliage',text:'ゞゞゞ',source:`AAHub ${R} aa-0127.txt line 7 (the tree at the right, ..ゞゞゞ)`,status:'retained: verbatim reference row, leaf clusters',paper:''},
 {id:'foliage-core',role:'tone_core',context:'foliage',text:'彡::彡',source:`AAHub ${R} aa-0127.txt line 5 (the tree at the right, 彡::彡:::ミ)`,status:'retained: verbatim reference row, crown mass',paper:''},
 {id:'foliage-deep',role:'tone_deep',context:'foliage',text:'爻爻爻',source:`AAHub ${M} aa-0033.txt line 1 (forest canopy from above, the crowns)`,status:'retained: verbatim reference row, dense canopy mass',paper:''},
];
/** The water field's single member. A liquid surface (the adapter's fact,
 * CompositionGeometry.surface) seen across is rows of spaced dashes, not the
 * ground's colon field: AAHub aa-0024 (海岸) line 13, the sea's row
 * `------- - - - - - - - - - - - - -- - - - -`, and aa-0028 (海岸の風景) line 5,
 * the sea's `‐ ‐ ‐‐`. Placed singly at its own advance plus a half space, on
 * the ground field's surface lattice and distance grade (tone-field.ts
 * terrainUnit). Appended last, so no index moves. */
const WATER_MEMBER_PASSAGES:readonly TonePassage[]=[
 {id:'water-member',role:'tone_fringe',context:'ground',text:'-',source:`AAHub ${R} aa-0024.txt line 13 (海岸, the sea's dash row); aa-0028.txt line 5 (海岸の風景, the sea's ‐ ‐ ‐‐)`,status:'retained: verbatim reference rows, open water',paper:'one half space after each member'},
];
/** The band's second member. A dark face's band is two glyphs alternating,
 * `:` and `;` (skill section 8 rule 1; the fog target's left wall rows
 * `:;:;:.:;:;:;:.` / `::;::;:;:.`), placed member by member on the surface
 * lattice and lightened by erasing members, never by a lighter glyph
 * (tone-field.ts bandUnits). Its first member is the ground member `:`.
 * Appended last, so no index moves. */
const BAND_MEMBER_PASSAGES:readonly TonePassage[]=[
 {id:'band-semicolon',role:'tone_core',context:'wall',text:';',source:'hand-authored-targets-20261029/fog/rendered/000.txt rows 4-23 (the shadow wall\'s :;:;: band)',status:'retained: hand-authored target, band member',paper:''},
];
/** THE FIELD'S ROW UNITS (tone lane 2026-11-13, coordinator approval): a
 * shaded region's row is read as one unit repeated with ずらし (practitioner
 * rule "a run is a unit plus a shift"), so each level has two verbatim
 * ten-scalar units of a reference row, the second for alternate lattice rows
 * as the reference alternates its rows. Fewer, longer words also keep the
 * native placement's word count down. Paths are repository-relative; lines are
 * 1-based. Appended last, so no index moves. */
const F='evidence/ascii-authoring/hand-authored-targets-20261029/fog/rendered/000.txt',CH='evidence/ascii-authoring/main-review/178-studio-horse-trot/church126-original.txt (AAHub 2f0353dc8e23bde120d1c050746f937c aa[126])';
const FIELD_ROW_PASSAGES:readonly TonePassage[]=[
 {id:'wall-row-dark-a',role:'tone_core',context:'wall',text:':;:;:.:;:;',source:`${F} line 6 (the shadow wall's row :;:;:.:;:;:;:.)`,status:'retained: hand-authored target, dark wall row',paper:''},
 {id:'wall-row-dark-b',role:'tone_core',context:'wall',text:'::;::;:;:.',source:`${F} line 5 (the alternate row ::;::;:;:.)`,status:'retained: hand-authored target, dark wall alternate row',paper:''},
 {id:'wall-row-mid-a',role:'tone_mid',context:'wall',text:'.:.:.:.:.:',source:`${CH} line 34 (the floor band . . : : .:.:.:.:.:.:)`,status:'retained: AAHub reference row, mid tone',paper:''},
 {id:'wall-row-mid-b',role:'tone_mid',context:'wall',text:':.:.:.:.:.',source:`${CH} line 15 (the alternate phase :.:.:.:.:.)`,status:'retained: AAHub reference row, mid tone alternate row',paper:''},
 {id:'wall-row-light-a',role:'tone_fringe',context:'wall',text:'. : . . :',source:`${F} line 7 (the thinning toward the light passage . : . . :)`,status:'retained: hand-authored target, light band row',paper:'single half spaces inside the unit'},
 {id:'wall-row-light-b',role:'tone_fringe',context:'wall',text:': . . : .',source:`${F} line 8 (the alternate row : . . : .)`,status:'retained: hand-authored target, light band alternate row',paper:'single half spaces inside the unit'},
];
/** THE CROWN'S ROW UNITS (tone lane 2026-11-13, coordinator order "foliage
 * crowns onto the field"): the crown mass of AAHub その他植物 resK-483 (大樹, a
 * great tree whose crown is an irregular colon/semicolon/tick texture, constant
 * over the crown, its edge carried by ヾ ゞ ｲ ﾉ), ten-scalar verbatim units, two
 * per level for alternate lattice rows. Archive path
 * /Users/r/Projects/ascii-art-archive/collections/aahub/sonota-shokubutsu/resK-483.txt,
 * 1-based lines. Appended last, so no index moves. */
const TREE='ascii-art-archive collections/aahub/sonota-shokubutsu/resK-483.txt (AAHub その他植物, 大樹)';
const CROWN_ROW_PASSAGES:readonly TonePassage[]=[
 {id:'crown-row-dark-a',role:'tone_core',context:'foliage',text:":;: :;'':;",source:`${TREE} line 2`,status:'retained: verbatim reference row, shaded crown mass',paper:''},
 {id:'crown-row-dark-b',role:'tone_core',context:'foliage',text:": ;:;:'':;",source:`${TREE} line 10`,status:'retained: verbatim reference row, shaded crown mass alternate row',paper:''},
 {id:'crown-row-mid-a',role:'tone_mid',context:'foliage',text:': ;: ;: ;:',source:`${TREE} line 19`,status:'retained: verbatim reference row, crown mass',paper:''},
 {id:'crown-row-mid-b',role:'tone_mid',context:'foliage',text:': :;: :; :',source:`${TREE} line 6`,status:'retained: verbatim reference row, crown mass alternate row',paper:''},
 {id:'crown-row-light-a',role:'tone_fringe',context:'foliage',text:': : : :; ;',source:`${TREE} line 7`,status:'retained: verbatim reference row, lit crown',paper:''},
 {id:'crown-row-light-b',role:'tone_fringe',context:'foliage',text:'; :; : ; :',source:`${TREE} line 18`,status:'retained: verbatim reference row, lit crown alternate row',paper:''},
];
/** CROWN SIDE WORDS (tone lane 2026-11-13): the crown edge of resK-483
 * (大樹) steps out with ｲ on its left (lines 3 and 8, ':,ｲ' and ';,ｲ') and
 * returns with ﾉ on its right (line 6, '; :ﾉ/:;'). Texture-edge words chosen by
 * the side's own direction (tone-field.ts edgeSpan). Appended last, so no index
 * moves. */
const CANOPY_SIDE_PASSAGES:readonly TonePassage[]=[
 {id:'canopy-left-out',role:'texture_edge',context:'canopy',text:',ｲ',source:`${TREE} lines 3 and 8 (the crown's left edge stepping out)`,status:'retained: verbatim reference unit, crown edge that is its own contour',paper:''},
 {id:'canopy-right-in',role:'texture_edge',context:'canopy',text:':ﾉ',source:`${TREE} line 6 (the crown's right edge returning under a lobe)`,status:'retained: verbatim reference unit, crown edge that is its own contour',paper:''},
];
/** THE SEA'S ROW UNITS (tone lane 2026-11-13): AAHub 地形（海・砂浜・島）
 * 3c33aa91ad5a95d68ec09a85919643be aa-0024 (海岸) line 13, the sea row
 * '------- - - - - - - - - - - - - -- - - - -': the run at the horizon, the
 * spaced dashes nearer. Placed by groundGrade, densest at the horizon
 * (tone-field.ts waterUnits); the nearest band keeps the single '-' member
 * (WATER_MEMBER_PASSAGES). Appended last, so no index moves. */
const SEA='AAHub 3c33aa91ad5a95d68ec09a85919643be aa-0024.txt line 13 (海岸, the sea row)';
const SEA_ROW_PASSAGES:readonly TonePassage[]=[
 {id:'sea-row-far',role:'tone_core',context:'ground',text:'-------',source:SEA,status:'retained: verbatim reference unit, the sea at the horizon',paper:''},
 {id:'sea-row-mid',role:'tone_mid',context:'ground',text:'- - - -',source:SEA,status:'retained: verbatim reference unit, the sea nearer',paper:'single half spaces inside the unit'},
 {id:'sea-row-near',role:'tone_fringe',context:'ground',text:'- -',source:SEA,status:'retained: verbatim reference unit, the near sea',paper:'single half space inside the unit'},
];
/** THE PARTICLE WORD (tone2, 2026-11-13): an emissive effect's occupied page
 * cell (tone-field.ts particleCell) takes the spark of AAHub 爆発・煙
 * (bakuhatsu-kemuri) resK-106 lines 13-14, 'ﾟ *。' and "ﾟ'*。": sparks floating
 * apart from the blast. Appended last, so no index moves. */
const PARTICLE_PASSAGES:readonly TonePassage[]=[
 {id:'particle-spark',role:'tone_fringe',context:'ground',text:'ﾟ',source:'ascii-art-archive collections/aahub/bakuhatsu-kemuri/resK-106.txt lines 13-14 (AAHub 爆発・煙, the sparks)',status:'retained: verbatim reference glyph, a spark',paper:''},
];
/** THE GRASS PATCH (tone2 g1, 2026-11-13): one constant pattern for turf, the
 * distant-grass patch of AAHub 9636c240 (T) aa-0003.txt lines 8, 10 and 15 (T#3,
 * operator "the patches of distnace grass"), followed by its own paper (one
 * full space). Replaces the grass bands, which the operator labelled bad
 * (an-r28-mi-dense; T#68 r1 "minus the superfluous inner tonal dots").
 * Appended last, so no index moves. */
const GRASS_PATCH_PASSAGES:readonly TonePassage[]=[
 {id:'grass-patch',role:'tone_fringe',context:'grass',text:'ﾞ" ﾞ"',source:`AAHub ${T} aa-0003.txt lines 8, 10, 15 (T#3, distant grass patches ﾞ" ﾞ"　ﾞ" ﾞ")`,status:'retained: operator verdict 9636c240:3 yes, patches of distance grass',paper:'one full space after the patch'},
];
/** THE 10-12 SEA'S WORDS (sea-100 rebase of numabe-90, operator 2026-11-17):
 * the frozen 10-12 Y8 runtime's water vocabulary, runtimes/y8-20261012
 * packages/trueff/render/ff/style/materials.ts WATER (crestLevel, crestRising,
 * crestFalling, ripple, surf), spelled by its layout.ts:1081-1089. Contiguous,
 * in this order (the tone field addresses them from SEA_CREST). Appended last,
 * so no index moves. */
const SEA_1012='runtimes/y8-20261012 packages/trueff/render/ff/style/materials.ts WATER';
const SEA_CREST_PASSAGES:readonly TonePassage[]=[
 {id:'sea-crest-level-0',role:'texture_edge',context:'ground',text:'⌒ヽ_',source:`${SEA_1012}.crestLevel[0]`,status:'retained: 10-12 travelling crest, level',paper:''},
 {id:'sea-crest-level-1',role:'texture_edge',context:'ground',text:'⌒',source:`${SEA_1012}.crestLevel[1]`,status:'retained: 10-12 travelling crest, level',paper:''},
 {id:'sea-crest-rising-0',role:'texture_edge',context:'ground',text:'_ノ',source:`${SEA_1012}.crestRising[0]`,status:'retained: 10-12 travelling crest, rising',paper:''},
 {id:'sea-crest-rising-1',role:'texture_edge',context:'ground',text:'(⌒',source:`${SEA_1012}.crestRising[1]`,status:'retained: 10-12 travelling crest, rising',paper:''},
 {id:'sea-crest-falling-0',role:'texture_edge',context:'ground',text:'ヽ_',source:`${SEA_1012}.crestFalling[0]`,status:'retained: 10-12 travelling crest, falling',paper:''},
 {id:'sea-crest-falling-1',role:'texture_edge',context:'ground',text:'⌒ヽ',source:`${SEA_1012}.crestFalling[1]`,status:'retained: 10-12 travelling crest, falling',paper:''},
 {id:'sea-ripple-0',role:'tone_fringe',context:'ground',text:'‐',source:`${SEA_1012}.ripple[0]`,status:'retained: 10-12 near-shore ripple (W-10/W-13)',paper:''},
 {id:'sea-ripple-1',role:'tone_fringe',context:'ground',text:'ｰ',source:`${SEA_1012}.ripple[1]`,status:'retained: 10-12 near-shore ripple (W-10/W-13)',paper:''},
 {id:'sea-surf-0',role:'tone_fringe',context:'ground',text:'∵',source:`${SEA_1012}.surf[0]`,status:'retained: 10-12 shore break spray',paper:''},
 {id:'sea-surf-1',role:'tone_fringe',context:'ground',text:'ﾟ｡',source:`${SEA_1012}.surf[1]`,status:'retained: 10-12 shore break spray',paper:''},
 {id:'sea-surf-2',role:'tone_fringe',context:'ground',text:'∴',source:`${SEA_1012}.surf[2]`,status:'retained: 10-12 shore break spray',paper:''},
 {id:'sea-surf-3',role:'tone_fringe',context:'ground',text:'〃',source:`${SEA_1012}.surf[3]`,status:'retained: 10-12 shore break spray',paper:''},
];
export const TONE_PASSAGES:readonly TonePassage[]=[{id:'paper',role:'tone_fringe',context:'wall',text:'',source:'paper',status:'paper',paper:''},...(bank.passages as readonly TonePassage[]),...TEXTURE_EDGE_PASSAGES,...GROUND_MEMBER_PASSAGES,...MATERIAL_PASSAGES,...WATER_MEMBER_PASSAGES,...BAND_MEMBER_PASSAGES,...FIELD_ROW_PASSAGES,...CROWN_ROW_PASSAGES,...CANOPY_SIDE_PASSAGES,...SEA_ROW_PASSAGES,...PARTICLE_PASSAGES,...GRASS_PATCH_PASSAGES,...SEA_CREST_PASSAGES];
const edgePassage=(id:string):number=>{const index=TONE_PASSAGES.findIndex(passage=>passage.id===id);if(index<0)throw Error(`Missing texture-edge passage ${id}`);return index;};
/** Indices into TONE_PASSAGES of the texture-edge words by boundary side. */
export const TEXTURE_EDGE={left:edgePassage('canopy-left'),right:edgePassage('canopy-right'),top:edgePassage('canopy-top'),underside:edgePassage('canopy-underside'),tip:edgePassage('canopy-tip')} as const;
/** Ring radius of the measure in page px: one and a half full-width glyphs. */
export const TEXTURE_EDGE_RADIUS=24;
/** PATH-LED OR REGION-LED, DECIDED BY THE FRAME'S OWN FIELDS (operator ruling
 * 5: a rendering rule, not a priority order). Shared verbatim by the tone
 * producer (tone-field.ts) and the line producer's selection
 * (gpu-articulation.ts tierSelects), so both read one function of the same
 * owner and depth fields. Needs p.field, p.rect, ownerField and depthField.
 *
 * A boundary can be spelled as a path (cap, shoulder, flank, return) only
 * when it holds its direction for about a glyph. On a ring of 16 samples
 * TEXTURE_EDGE_RADIUS page px about a point of the form beside its boundary,
 * a straight, convex or smoothly curved boundary enters and leaves the form
 * once (2 crossings). A lobed or serrated boundary whose lobes are finer than
 * the ring crosses it 4 times or more: no glyph run can follow it, and its
 * envelope is drawn as texture. A limb thinner than a glyph also crosses 4
 * times but covers under a quarter of the ring; the form must cover 6 of 16.
 * The form must also hang: down the page from the point, or from one ring
 * radius to either side of it, its surface ends within 288 page px on nothing
 * or on a surface a depth step behind it, not on a surface it rests on (a
 * rock, a bush on its ground and a terrain skyline keep their path-led
 * outline). The radius, the 6 of 16 and the 288 px are this rule's own
 * choices, not measured reference values. */
export const TEXTURE_EDGE_MEASURE=/* wgsl */`
// Owner (without the UV-contract flag, bit 29) and depth bits; zero when off.
fn leafHit(q:vec2f)->vec3u{
 let c=vec2i(floor(q));
 if(any(c<vec2i(0))||any(c>=vec2i(p.field.xy))){return vec3u(0u);}
 let o=textureLoad(ownerField,c,0).xy;let d=textureLoad(depthField,c,0).x;
 if((o.x|o.y)==0u||d<=0.0){return vec3u(0u);}
 return vec3u(o.x&0xdfffffffu,o.y,bitcast<u32>(d));
}
// The same form surface: its owner, within one depth step of a known depth.
fn leafOn(h:vec3u,owner:vec2u,depth:f32)->bool{
 let d=bitcast<f32>(h.z);
 return (h.x|h.y)!=0u&&h.x==(owner.x&0xdfffffffu)&&h.y==owner.y&&abs(d-depth)<=.08*max(d,depth);
}
fn leafHangs(q:vec2f,owner:vec2u,depth:f32)->bool{
 let step=8.0*p.field.y/p.rect.w;var d=depth;
 if(!leafOn(leafHit(q),owner,depth)){return false;}
 for(var k=1;k<=36;k++){
  let f=q+vec2f(0.0,f32(k)*step);
  if(f.y>=p.field.y){return false;}
  let h=leafHit(f);
  if(leafOn(h,owner,d)){d=bitcast<f32>(h.z);continue;}
  return (h.x|h.y)==0u||bitcast<f32>(h.z)<d*.92;
 }
 return false;
}
fn textureEdge(q:vec2f,owner:vec2u,depth:f32)->bool{
 let scale=p.field.xy/p.rect.zw;
 var inside=0u;var turns=0u;var first=false;var last=false;
 for(var k=0u;k<16u;k++){
  let a=f32(k)*.39269908;
  let on=leafOn(leafHit(q+vec2f(cos(a),sin(a))*${TEXTURE_EDGE_RADIUS}.0*scale),owner,depth);
  if(k==0u){first=on;}else if(on!=last){turns++;}
  last=on;if(on){inside++;}
 }
 if(last!=first){turns++;}
 if(turns<4u||inside<6u){return false;}
 let aside=vec2f(${TEXTURE_EDGE_RADIUS}.0*scale.x,0.0);
 return leafHangs(q,owner,depth)||leafHangs(q-aside,owner,depth)||leafHangs(q+aside,owner,depth);
}
`;
/** The construction's passage per context and field part (indices into
 * TONE_PASSAGES; 0 = none): fringe (sparse, toward the light), mid, core,
 * deep (a round's terminator row), contact (a receiver's cast-shadow tail). */
type Bands=Readonly<Record<'fringe'|'mid'|'core'|'deep'|'contact',number>>;
const bankPreferred=bank.preferred as Readonly<Record<'wall'|'round'|'ground',Bands>>;
const passageIndex=(context:ToneContext,text:string):number=>{const index=TONE_PASSAGES.findIndex(passage=>passage.context===context&&passage.text===text);if(index<0)throw Error(`Missing ${context} tone passage ${text}`);return index;};
const materialBands=(context:'rock'|'grass'|'foliage'):Bands=>({fringe:edgePassage(`${context}-fringe`),mid:edgePassage(`${context}-mid`),core:edgePassage(`${context}-core`),deep:edgePassage(`${context}-deep`),contact:0});
/** The bank's wall `deep` is its first longest core passage, `.:.:.:` (ink
 * per advance .5), lighter than the wall core `:;:;:` (.93): a value
 * inversion. The wall's deep band is the fog target's denser row of the same
 * family, `:;:;:;` (1.0, hand-authored-targets-20261029/fog/rendered/000.txt),
 * already in the bank. */
const WALL_DEEP=passageIndex('wall',':;:;:;');
export const TONE_PREFERRED:Readonly<Record<'wall'|'round'|'ground'|'rock'|'grass'|'foliage',Bands>>={
 ...bankPreferred,wall:{...bankPreferred.wall,deep:WALL_DEEP},
 rock:materialBands('rock'),grass:materialBands('grass'),foliage:materialBands('foliage'),
};
/** Contexts whose band order is checked by ink per advance at compile time. */
export const TONE_VALUE_ORDERED=['wall','rock','grass','foliage'] as const;
/** A planar field's graded run, dense to sparse, five steps (indices into
 * TONE_PASSAGES; 0 = none): `;` core, `:;:` mid, the alternating two-glyph
 * band, the spaced fringe, the sparsest spaced tail. */
export const TONE_FIELD_RUN=bank.ladder as Readonly<Record<'wall',readonly number[]>>;
export const TONE_PASSAGE_FONT_SHA256=bank.font_sha256 as string;

export const TONE_PASSAGE_HEADER_WORDS=8;
export const TONE_PASSAGE_CLEARANCE=11;
export interface CompiledTonePassages {
 /** Header per passage: scalar count, advance, atom offset, ink offset,
  * ink count, paper offset, paper count, outer horizontal clearance in px.
  * Offsets count RECORDS, relative to their section.
  * Atoms: font address / pen offset / native advance / Unicode scalar.
  * Ink: x/y pairs. Paper: x/width pairs (one native row high). */
 readonly headers:Uint32Array;
 readonly atoms:Uint32Array;
 readonly ink:Int32Array;
 readonly paper:Uint32Array;
 /** The native half-width space advance: the paper interval between the
  * members of a sparse field (the fog target's `. : . . : . : . .`). */
 readonly gap:number;
}

/** Compile once for the native font owned by the publisher. No per-frame
 * fitting, scalar ranking, substring clipping or shortened fallback word.
 * The tone producer (tone-field.ts) verifies a passage's ink and paper with
 * these tables; the seal spells it as the grammar's whole word
 * (compiled-role-grammar.ts toneWords). */
export function compileTonePassages(font:readonly NativeGlyph[]):CompiledTonePassages{
 const addresses=new Map(font.map((glyph,index)=>[glyph.text,index]));
 const headers=new Uint32Array(TONE_PASSAGES.length*TONE_PASSAGE_HEADER_WORDS);
 const atoms:number[]=[],ink:number[]=[],paper:number[]=[];
 for(const [index,passage]of TONE_PASSAGES.entries()){
  const atomStart=atoms.length/4,inkStart=ink.length/2,paperStart=paper.length/2;
  let pen=0;
  for(const scalar of passage.text){
   const address=addresses.get(scalar);
   if(address===undefined)throw Error(`Native tone passage ${passage.id} lacks ${scalar}`);
   const glyph=font[address]!;
   atoms.push(address,pen,glyph.advance,scalar.codePointAt(0)!);
   if(!glyph.points.length)paper.push(pen,glyph.advance);
   for(const point of glyph.points)ink.push(pen+point[0]!,point[1]!);
   pen+=glyph.advance;
  }
  headers.set([atoms.length/4-atomStart,pen,atomStart,inkStart,ink.length/2-inkStart,paperStart,paper.length/2-paperStart,index===0?0:TONE_PASSAGE_CLEARANCE],index*TONE_PASSAGE_HEADER_WORDS);
 }
 // Value order: deep darker than core darker than mid darker than fringe,
 // measured as the word's ink pixels per pixel of advance in this font.
 const value=(index:number)=>headers[index*TONE_PASSAGE_HEADER_WORDS+4]!/Math.max(1,headers[index*TONE_PASSAGE_HEADER_WORDS+1]!);
 for(const context of TONE_VALUE_ORDERED){
  const bands=TONE_PREFERRED[context];const order=[bands.fringe,bands.mid,bands.core,bands.deep];
  for(let i=1;i<order.length;i++)if(!(value(order[i]!)>value(order[i-1]!)))throw Error(`Tone context ${context} band ${i} (${TONE_PASSAGES[order[i]!]!.text}) is not darker than band ${i-1} (${TONE_PASSAGES[order[i-1]!]!.text})`);
 }
 const space=addresses.get(' ');
 if(space===undefined)throw Error('Native font lacks the half-width space');
 return {headers,atoms:Uint32Array.from(atoms),ink:Int32Array.from(ink),paper:Uint32Array.from(paper),gap:font[space]!.advance};
}
