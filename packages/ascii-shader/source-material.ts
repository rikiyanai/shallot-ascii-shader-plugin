import authored from '../../assets/banks/material-classes.v1.json';
/** Authored source facts; never a colour, geometry or subject classifier. */
export const MATERIAL_CLASSES=['unknown','stone/masonry','brick','wood','plaster','metal','foliage','rock','turf'] as const;
export type MaterialClassName=typeof MATERIAL_CLASSES[number];
export type MaterialClass=0|1|2|3|4|5|6|7|8;
export const MATERIAL_STYLE_SHIFT=6,MATERIAL_STYLE_MASK=0x3c0;
/** Authored foliage anywhere on this source, including mixed-material trees. */
export const MATERIAL_FOLIAGE_FACES_BIT=0x400;
export type SourceMaterialKey=
 |{readonly kind:'bank-family';readonly family:string}
 |{readonly kind:'gltf-material';readonly asset:string;readonly material:number};
export interface AuthoredMaterialClasses {
 readonly schema:'shallot.source-material-classes.v1';
 readonly families:Readonly<Record<string,MaterialClassName|Readonly<Record<string,MaterialClassName>>>>;
 /** sha256:<document bytes>, independent of hosting URL; nested ORIGINAL glTF material index. */
 readonly gltfMaterials:Readonly<Record<string,Readonly<Record<string,MaterialClassName>>>>;
}
interface MaterialSource {readonly materialSource?:SourceMaterialKey;readonly lineage?:{readonly materials?:Uint32Array}}
const classId=(name:unknown):MaterialClass=>{const id=MATERIAL_CLASSES.indexOf(name as MaterialClassName);return id>=0?id as MaterialClass:0;};
/** Whole-source style. Authored per-material families use the face stream,
 * never a majority class. Mixed unauthored sources stay unknown. */
export function compileSourceMaterialClass(source:MaterialSource,table:AuthoredMaterialClasses=authored as AuthoredMaterialClasses):MaterialClass{
 const ids=source.lineage?.materials;
 if(ids?.length){const first=ids[0];for(let i=1;i<ids.length;i++)if(ids[i]!==first)return 0;}
 const key=source.materialSource;if(!key)return 0;
 let name:unknown;
 if(key.kind==='bank-family')name=Object.hasOwn(table.families,key.family)?table.families[key.family]:undefined;
 else if(key.kind==='gltf-material'&&Number.isSafeInteger(key.material)&&key.material>=0){
  const materials=Object.hasOwn(table.gltfMaterials,key.asset)?table.gltfMaterials[key.asset]:undefined;
  if(materials&&Object.hasOwn(materials,String(key.material)))name=materials[String(key.material)];
 }
 return classId(name);
}
/** Source face f is the same face as packed halfedge f*3. Eight class
 * nibbles per u32; zero suffix omitted. No position/colour inference. */
export function compileSourceFaceMaterialClasses(source:MaterialSource,faceCount:number,table:AuthoredMaterialClasses=authored as AuthoredMaterialClasses):Uint32Array|undefined{
 const key=source.materialSource;
 if(key?.kind!=='bank-family'||!Object.hasOwn(table.families,key.family))return;
 const mapping=table.families[key.family];
 if(!mapping||typeof mapping!=='object'||Array.isArray(mapping))return;
 const ids=source.lineage?.materials;if(!ids)return;
 if(!Number.isSafeInteger(faceCount)||faceCount<0||ids.length!==faceCount)throw Error('Material lineage does not cover source faces');
 const words=new Uint32Array(Math.ceil(faceCount/8));let used=false;
 for(let f=0;f<faceCount;f++){const id=String(ids[f]),value=classId(Object.hasOwn(mapping,id)?mapping[id]:undefined);words[f>>>3]!|=value<<((f&7)*4);used||=value!==0;}
 return used?words:undefined;
}
interface SourceStyle {readonly type:string;readonly surface?:'liquid'|'heightfield';readonly foliage?:boolean;readonly organic?:boolean;readonly styleWord?:number}
/** Existing bits0-9 unchanged; bit10 reports authored foliage faces.
 * Scanned once per source revision, never during tone encode. */
export function compileSourceStyleWord(source:SourceStyle,material:MaterialClass,materialClasses?:Uint32Array):number{
 const foliageFaces=material===6||!!materialClasses?.some(word=>{
  for(let shift=0;shift<32;shift+=4)if(((word>>>shift)&15)===6)return true;
  return false;
 });
 return ((source.type==='planar'?1:source.type==='rounded'?2:source.type==='terrain'?3:0)
  |(source.surface==='liquid'?8:0)|(source.foliage?16:0)|(source.organic?32:0)|(material<<MATERIAL_STYLE_SHIFT)
  |(foliageFaces?MATERIAL_FOLIAGE_FACES_BIT:0))>>>0;
}
/** Tone-owner accessor: registered inputs are one integer read. Legacy callers
 * retain their old flags, without authored-data lookup or material inference. */
export function sourceToneStyleWord(source:SourceStyle):number{
 return source.styleWord===undefined?compileSourceStyleWord(source,0):source.styleWord>>>0;
}
/** Only when Y8 creates retained source geometry. AKM stems are the operator's
 * explicit authored registry keys, never an automatic class-name heuristic. */
export function bankFamilyMaterialSource(facts:Readonly<Record<string,unknown>>):SourceMaterialKey|undefined{
 const names=[facts.authored_geometry_variant,facts.source_mesh_name].filter((v):v is string=>typeof v==='string'&&/\.akm$/i.test(v));
 const families=new Set(names.map(name=>name.replaceAll('\\','/').split('/').at(-1)!.replace(/\.akm$/i,'')));
 return families.size===1?{kind:'bank-family',family:[...families][0]!}:undefined;
}
