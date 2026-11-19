/** Registration/build adapter for explicit authored parts. No renderer,
 * material classifier, mesh-name test or glyph recipe lives here. */
import {readFile,writeFile} from 'node:fs/promises';
import {resolve} from 'node:path';
const LEGACY_CONSTRUCTION_TYPE=`{ readonly schema: "shallot.source-construction.v1"; readonly role: "blade" | "stem" | "head" | "trunk" | "crown" | "solid"; readonly components: readonly { readonly primitive: "box"; readonly center: readonly [number, number, number]; readonly size: readonly [number, number, number]; readonly firstIndex: number; readonly indexCount: number }[] }`;
const PATH_SITE_TYPE=`({ readonly kind: "vertex"; readonly vertex: number } | { readonly kind: "face"; readonly face: number; readonly barycentric: readonly [number, number, number] })`;
export const CONSTRUCTION_TYPE=LEGACY_CONSTRUCTION_TYPE.slice(0,-2)+`; readonly paths?: readonly { readonly id: number; readonly sites: readonly ${PATH_SITE_TYPE}[]; readonly ports: readonly [number, number]; readonly rank: number }[]; readonly pathPairs?: readonly { readonly id: number; readonly paths: readonly [number, number]; readonly endPorts: readonly [readonly [number, number], readonly [number, number]]; readonly gap: readonly ${PATH_SITE_TYPE}[] }[] }`;
export const ASSEMBLY_TYPE=`{ readonly schema: "shallot.source-assembly.v1"; readonly namespace: string; readonly records: GPUBufferBinding; readonly stride: 24 }`;
const MARKER='/* MAIN_AUTHORED_CONSTRUCTION_0103 */';
function once(text,from,to,label){
 if(text.split(from).length!==2)throw Error('MAIN authored construction anchor changed: '+label);
 return text.replace(from,to);
}
/** Pure function: fixtures can check the actual engine seam without touching
 * the installed package or initializing an engine/browser/GPU. */
export function transformAuthoredConstructionEngine({mesh,forward,producer}){
 // Upgrade the existing declared seam too. A setup rerun must not leave a
 // previously installed engine with a narrower box-only contract.
 mesh=mesh.replaceAll(LEGACY_CONSTRUCTION_TYPE,CONSTRUCTION_TYPE);
 forward=forward.replaceAll(LEGACY_CONSTRUCTION_TYPE,CONSTRUCTION_TYPE);
 if(!mesh.includes(MARKER)){
  for(const declaration of ['export interface MeshData {','interface PendingMesh {'])
   mesh=once(mesh,declaration,declaration+'\n    construction?: '+CONSTRUCTION_TYPE+';',declaration);
  const declarations=mesh.split('\n').filter(line=>line.includes('    sourceGeometry?:')&&line.includes('Float32Array'));
  if(declarations.length!==1)throw Error('MAIN authored construction source geometry declaration changed');
  mesh=once(mesh,declarations[0],declarations[0].replace(/;$/,` & { construction?: ${CONSTRUCTION_TYPE} };`),'packed source type');
  const anchor='    const sourceSemantics = new Map(staged.map((m) => [m.handle, m.semantic]));';
  mesh=once(mesh,anchor,anchor+'\n    '+MARKER+'\n    const sourceConstructions = new Map(staged.map((m) => [m.handle, m.construction]));','registration map');
  mesh=once(mesh,'semantic: sourceSemantics.get(s.handle) }','semantic: sourceSemantics.get(s.handle), construction: sourceConstructions.get(s.handle) }','revision publication');
 }
 if(!producer.includes(MARKER)){
  producer=once(producer,'export interface ProducerTable {','export interface ProducerTable {\n    '+MARKER+'\n    readonly sourceAssembly?: '+ASSEMBLY_TYPE+';','producer table');
  producer=once(producer,'        materials: readonly MaterialHandle[];','        materials: readonly MaterialHandle[];\n        sourceAssembly?: '+ASSEMBLY_TYPE+';','producer options');
  producer=once(producer,'    const table: ProducerTable = {','    const table: ProducerTable = {\n        sourceAssembly: options.sourceAssembly,','producer registration');
  const state='    const state = world.resource(producerTables);';
  producer=once(producer,state,`    if (options.sourceAssembly) {
        const a = options.sourceAssembly, offset = a.records.offset ?? 0;
        const bytes = a.records.size ?? a.records.buffer.size - offset;
        if (a.schema !== "shallot.source-assembly.v1" || a.stride !== 24 || !a.namespace ||
            !Number.isSafeInteger(offset) || offset < 0 || offset % 4 ||
            bytes < capacity * 24 || bytes > limits.maxStorageBufferBindingSize ||
            offset + bytes > a.records.buffer.size) fail("invalid same-row authored assembly column");
    }
`+state,'companion capacity');
 }
 if(!forward.includes(MARKER)){
  const declarations=forward.split('\n').filter(line=>line.includes('    sourceGeometry?:')&&line.includes('Float32Array'));
  if(declarations.length!==1)throw Error('MAIN authored construction visitor declaration changed');
  forward=once(forward,declarations[0],MARKER+'\n'+declarations[0].replace(/;$/,` & { construction?: ${CONSTRUCTION_TYPE} };`)+'\n    sourceAssembly?: '+ASSEMBLY_TYPE+';','visitor type');
  const anchor='            sourceGeometry: world.resource(Meshes).get(item.draw.mesh)!.sourceGeometry,';
  forward=once(forward,anchor,anchor+'\n            sourceAssembly: world.resource(producerTables).draws.get(item.draw.name)?.sourceAssembly,','same producer row column');
 }
 if(!mesh.includes('construction: sourceConstructions.get(s.handle)'))throw Error('MAIN authored construction revision publication is absent');
 return {mesh,forward,producer};
}
export async function applyAuthoredConstruction(engine){
 const paths=['src/core/mesh/mesh.ts','src/standard/rendering/forward.ts','src/standard/rendering/producer-table.ts'];
 const original=await Promise.all(paths.map(path=>readFile(resolve(engine,path),'utf8')));
 const transformed=transformAuthoredConstructionEngine({mesh:original[0],forward:original[1],producer:original[2]});
 for(const [i,text]of [transformed.mesh,transformed.forward,transformed.producer].entries())if(text!==original[i])await writeFile(resolve(engine,paths[i]),text);
}
export function transformVoxelDetailParts(source){
 if(source.includes('/* MAIN_AUTHORED_DETAIL_PARTS */'))return source;
 const type='type Box = [number, number, number, number, number, number];';
 source=once(source,type,type+'\n/* MAIN_AUTHORED_DETAIL_PARTS */\ntype AuthoredBox = { primitive: "box"; center: [number,number,number]; size: [number,number,number]; firstIndex: number; indexCount: number };','box declaration');
 source=once(source,'    for (const [x, y, z, sx, sy, sz] of parts) {','    const components: AuthoredBox[] = [];\n    for (const [x, y, z, sx, sy, sz] of parts) {','original components');
 source=once(source,'        const base = vertices.length / 8;','        const base = vertices.length / 8;\n        components.push({ primitive: "box", center: [x,y,z], size: [sx,sy,sz], firstIndex: indices.length, indexCount: box.indices.length });','original primitive range');
 const returned='    return { name, vertices: new Float32Array(vertices), indices: new Uint32Array(indices) };';
 source=once(source,returned,'    return { name, vertices: new Float32Array(vertices), indices: new Uint32Array(indices), components };','part definition');
 // Array ordinals are the original kind/part table immediately below, not
 // guesses from mesh.name or RGB. Amber/cyan intentionally share the stem.
 const map='    ].map((mesh) => meshes.add(mesh));';
 source=once(source,map,`    ].map(({components, ...mesh}, index) => meshes.add({...mesh, construction: {
        schema: "shallot.source-construction.v1", role: (["blade","stem","head","head","solid","trunk","crown"] as const)[index], components,
    }}));`,'authored mesh roles');
 return source;
}
export function transformVoxelDetailPlacement(source){
 if(source.includes('/* MAIN_AUTHORED_DETAIL_ASSEMBLIES */'))return source;
 const params='const Params = d.struct({ center: d.i32, remainder: d.f32 });';
 source=once(source,params,params+'\n/* MAIN_AUTHORED_DETAIL_ASSEMBLIES */','marker');
 const count='    count: { storage: d.atomic(d.u32), access: "mutable" },';
 source=once(source,count,count+'\n    sourceAssembly: { storage: d.arrayOf(d.u32), access: "mutable" },','companion column layout');
 const write='            layout.$.instances[row + p] = MeshInstanceInput({';
 source=once(source,write,`            // World column, NOT the floating-origin center or atomic row.
            const attachment = (row+p)*6;
            layout.$.sourceAssembly[attachment] = d.u32(chunk.x * d.i32(size) + d.i32(packed & 255));
            layout.$.sourceAssembly[attachment+1] = (packed >>> 8) & 255;
            layout.$.sourceAssembly[attachment+2] = d.u32(chunk.y * d.i32(size) + d.i32((packed >>> 16) & 255));
            layout.$.sourceAssembly[attachment+3] = kind;
            layout.$.sourceAssembly[attachment+4] = p;
            layout.$.sourceAssembly[attachment+5] = n;
`+write,'stable parent + original sibling slot');
 const register='    const table = registerProducerTable(world, {';
 source=once(source,register,'    const sourceAssembly = root.createBuffer(d.arrayOf(d.u32, slots * columns * 12)).$usage("storage");\n'+register,'companion allocation');
 source=once(source,'        name: "voxel:detail",','        name: "voxel:detail",\n        sourceAssembly: {schema: "shallot.source-assembly.v1", namespace: "voxel:detail", records: {buffer: root.unwrap(sourceAssembly)}, stride: 24},','registered companion');
 source=once(source,'            placements: table.placements,','            placements: table.placements,\n            sourceAssembly,','producer writes registered column');
 return source;
}
/** Original imported files remain byte-identical. Only the build's modules
 * receive the explicit authored producer adapter. Defaults elsewhere unchanged. */
export function authoredVoxelConstructionPlugin(root){
 const parts=resolve(root,'voxel/src/parts.ts'),placement=resolve(root,'voxel/src/placement.ts');
 return {name:'shallot-authored-construction',enforce:'pre',transform(source,id){
  const path=id.split('?')[0];
  if(path===parts)return {code:transformVoxelDetailParts(source),map:null};
  if(path===placement)return {code:transformVoxelDetailPlacement(source),map:null};
 }};
}
