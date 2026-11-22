/** Reproducible immutable material-identity seam for isolated Shallot0.10.5. */
import {readFile,writeFile} from 'node:fs/promises';
import {resolve} from 'node:path';
const key='{ kind: "gltf-material"; asset: string; material: number }';
const cullBoundMarker='MAIN_MATERIAL_CULL_BOUNDS_ABI_20261122';
function once(text,from,to,label){if(text.split(from).length!==2)throw Error('MAIN material metadata anchor changed: '+label);return text.replace(from,to);}
export async function applySourceMaterialMetadata(engine){
 const relative=['src/core/mesh/mesh.ts','src/extras/gltf/gltf.ts','src/extras/gltf/load.ts','src/standard/rendering/forward.ts','src/standard/rendering/material-type.ts'];
 const original=await Promise.all(relative.map(path=>readFile(resolve(engine,path),'utf8')));let [mesh,gltf,load,forward,materialType]=original;
 if(!mesh.includes('MAIN_SOURCE_MATERIAL_METADATA_0103')){
  const source='sourceGeometry?: { positions: Float32Array; indices: Uint32Array; contactKeys?: Uint32Array; revision: number }';
  mesh=once(mesh,source,source+' & { materialSource?: '+key+' }','packed source');
  for(const declaration of ['export interface MeshData {','interface PendingMesh {'])mesh=once(mesh,declaration,declaration+'\n    materialSource?: '+key+';',declaration);
  const bounds='    const bounds = new Map(staged.map((m) => [m.handle, m.bounds]));';
  mesh=once(mesh,bounds,bounds+'\n    /* MAIN_SOURCE_MATERIAL_METADATA_0103 */\n    const sourceMaterials = new Map(staged.map((m) => [m.handle, m.materialSource]));','source metadata map');
  const packed='sourceGeometry: { positions: sourcePositions, indices: sourceIndices, contactKeys: sourceContactKeys, revision: 0 }';
  mesh=once(mesh,packed,'sourceGeometry: { positions: sourcePositions, indices: sourceIndices, contactKeys: sourceContactKeys, revision: 0, materialSource: sourceMaterials.get(s.handle) }','source metadata publication');
 }
 if(!gltf.includes('materialSource?: '+key))gltf=once(gltf,'export interface GltfMesh {','export interface GltfMesh {\n    /** Original document identity + material index, assigned once at load. */\n    materialSource?: '+key+';','gltf mesh');
 if(!load.includes('MAIN_SOURCE_MATERIAL_METADATA_0103')){
  const anchor='    const scene = parseGltf(json, buffers);';
  load=once(load,anchor,anchor+'\n    /* MAIN_SOURCE_MATERIAL_METADATA_0103: stable across hosting URL changes. */\n    const digest = await crypto.subtle.digest("SHA-256", bytes);\n    const asset = "sha256:" + Array.from(new Uint8Array(digest), value => value.toString(16).padStart(2, "0")).join("");\n    for (const mesh of scene.meshes) mesh.materialSource = { kind: "gltf-material", asset, material: mesh.material };','gltf source identity');
 }
 // 0.10.5 moved document fetching into resolveGltf. Keep the original document
 // bytes in scope for ONE hash per load, without refetching or hashing a BIN
 // resource, decoded JSON, individual mesh or current hosting URL instead.
 if(load.includes('export async function resolveGltf(')&&!load.includes('MAIN_SOURCE_MATERIAL_DOCUMENT_BYTES_0105')){
  load=once(load,'    const { base, json, buffers } = await resolveGltf(url, read);',
   '    const { base, json, buffers, bytes } = await resolveGltf(url, read);','resolved document bytes');
  load=once(load,'    return { base, json, buffers };',
   '    /* MAIN_SOURCE_MATERIAL_DOCUMENT_BYTES_0105 */\n    return { base, json, buffers, bytes };','document bytes return');
 }
 // Main apply rewrites forward each run. This optional intersection keeps its
 // original sourceGeometry anchor intact and is restored idempotently.
 if(!forward.includes('materialSource?: '+key)){
  const source='sourceGeometry?: { positions: Float32Array; indices: Uint32Array; contactKeys?: Uint32Array; revision: number }';
  forward=once(forward,source,source+' & { materialSource?: '+key+' }','visitor source metadata');
 }
 // Immutable producer surface facts travel with the same source revision.
 // This is not inferred from a mesh name, material colour or scene identity.
 const surface='{ surface?: "heightfield" | "liquid" }';
 const source='sourceGeometry?: { positions: Float32Array; indices: Uint32Array; contactKeys?: Uint32Array; revision: number }';
 if(!mesh.includes('MAIN_SOURCE_SURFACE_METADATA_0103')){
  mesh=once(mesh,source,source+' & '+surface,'source surface type');
  for(const declaration of ['export interface MeshData {','interface PendingMesh {'])
   mesh=once(mesh,declaration,declaration+'\n    surface?: "heightfield" | "liquid";',declaration+' surface');
  const materials='    const sourceMaterials = new Map(staged.map((m) => [m.handle, m.materialSource]));';
  mesh=once(mesh,materials,materials+'\n    /* MAIN_SOURCE_SURFACE_METADATA_0103 */\n    const sourceSurfaces = new Map(staged.map((m) => [m.handle, m.surface]));','surface source map');
  mesh=once(mesh,'materialSource: sourceMaterials.get(s.handle) }','materialSource: sourceMaterials.get(s.handle), surface: sourceSurfaces.get(s.handle) }','surface publication');
 }
 if(!forward.includes(surface))forward=once(forward,source,source+' & '+surface,'visitor surface type');
 // Optional build-time semantic residual. Typed-array MeshData and its packed
 // revision carry the SAME bytes; this is generic immutable source metadata,
 // not a scene-name decoder or a second mesh/renderer implementation.
 const semantic='{ semantic?: Uint8Array }';
 if(!mesh.includes('MAIN_SOURCE_SEMANTIC_METADATA_0103')){
  mesh=once(mesh,source,source+' & '+semantic,'source semantic type');
  for(const declaration of ['export interface MeshData {','interface PendingMesh {'])
   mesh=once(mesh,declaration,declaration+'\n    semantic?: Uint8Array;',declaration+' semantic');
  const surfaces='    const sourceSurfaces = new Map(staged.map((m) => [m.handle, m.surface]));';
  mesh=once(mesh,surfaces,surfaces+'\n    /* MAIN_SOURCE_SEMANTIC_METADATA_0103 */\n    const sourceSemantics = new Map(staged.map((m) => [m.handle, m.semantic]));','source semantic map');
  mesh=once(mesh,'surface: sourceSurfaces.get(s.handle) }','surface: sourceSurfaces.get(s.handle), semantic: sourceSemantics.get(s.handle) }','source semantic publication');
 }
 if(!forward.includes(semantic))forward=once(forward,source,source+' & '+semantic,'visitor semantic type');
 // A generic material producer may expand the native local sphere after the
 // material row is selected. Publish that optional callback through the
 // actual MaterialType -> MaterialAssets -> MaterialSource chain. This is a
 // public engine seam, not a scene-specific material transform.
 if(!materialType.includes(cullBoundMarker)){
  materialType=once(materialType,'    cullBounds?(row: number, out: Float32Array): void;',
   '    cullBounds?(row: number, out: Float32Array, nativeSphere?: ArrayLike<number>): void;','material source ABI');
  materialType=once(materialType,'    readonly defaults?: d.InferInput<P>;\n',
   '    readonly defaults?: d.InferInput<P>;\n    readonly cullBounds?: (row: number, out: Float32Array, nativeSphere?: ArrayLike<number>) => void;\n','material type ABI');
  materialType=once(materialType,'    private _table: GpuTable<P> | undefined;\n',
   '    private _table: GpuTable<P> | undefined;\n    /** Published only when this material type owns a deformation bound. */\n    readonly cullBounds?: (row: number, out: Float32Array, nativeSphere?: ArrayLike<number>) => void;\n','material asset ABI');
  materialType=once(materialType,'        this._type = type;\n',
   '        this._type = type;\n        if (type.cullBounds) this.cullBounds = (row, out, nativeSphere) => type.cullBounds!(row, out, nativeSphere);\n','material asset publication');
  materialType+='\n/* '+cullBoundMarker+' */\n';
 }
 if(!forward.includes(cullBoundMarker)){
  const call='            source.cullBounds(materials.material.get(instanceEid), state.materialBounds);';
  forward=once(forward,'    materialBounds: Float32Array;\n','    materialBounds: Float32Array;\n    nativeMaterialBounds: Float32Array;\n','forward native sphere scratch declaration');
  forward=once(forward,'        materialBounds: new Float32Array(4),\n','        materialBounds: new Float32Array(4),\n        nativeMaterialBounds: new Float32Array(4),\n','forward native sphere scratch allocation');
  forward=once(forward,call,'            state.nativeMaterialBounds[0] = bx;\n            state.nativeMaterialBounds[1] = by;\n            state.nativeMaterialBounds[2] = bz;\n            state.nativeMaterialBounds[3] = radius;\n            source.cullBounds(materials.material.get(instanceEid), state.materialBounds, state.nativeMaterialBounds);\n        /* '+cullBoundMarker+' */','forward native sphere');
 }
 // The native quantizer already owns each typed mesh's exact decoded box.
 // Publish that small extent once; a ground patch's bounding sphere must not
 // pretend its horizontal radius is occupied height. Custom/GPU sources have
 // no such certificate and retain their original cull-bound contract.
 if(!mesh.includes('MAIN_NATIVE_POSITION_BOUNDS_0103')){
  const declaration='    bounds?: [number, number, number, number];';
  mesh=once(mesh,declaration,declaration+'\n    /** Decoded native position extent before a material vertex hook. */\n    positionBounds?: readonly [number, number, number, number, number, number];','native position bounds type');
  const anchor='        const sourcePositions = new Float32Array(s.vertexCount * 3);';
  mesh=once(mesh,anchor,`        /* MAIN_NATIVE_POSITION_BOUNDS_0103 */
        const sourceQuant = s.vertexCount ? (q.main[s.vertexBase * 4 + 1] >>> 16) * MESH_QUANT_FLOATS : 0;
        const sourcePositionBounds = Array.from({length: 6}, (_, i) => {
            if (!s.vertexCount) return 0;
            const axis = i % 3, low = q.quant[sourceQuant + axis], extent = q.quant[sourceQuant + axis + 4];
            const margin = 1e-5 * (1 + Math.abs(low) + Math.abs(extent));
            return i < 3 ? low - margin : low + extent + margin;
        }) as [number, number, number, number, number, number];
`+anchor,'native position bounds publication');
  mesh=once(mesh,'            bounds: bounds.get(s.handle),','            bounds: bounds.get(s.handle),\n            positionBounds: sourcePositionBounds,','native position bounds owner');
 }
 for(const [i,text]of [mesh,gltf,load,forward,materialType].entries())if(text!==original[i])await writeFile(resolve(engine,relative[i]),text);
}
