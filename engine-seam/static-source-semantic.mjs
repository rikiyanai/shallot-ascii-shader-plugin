

















import {readFile,writeFile} from 'node:fs/promises';
import {resolve} from 'node:path';

// Published gzip-compressed (size depends on the compiler revision); the
// runtime decompresses with DecompressionStream, or reads raw bytes when a host
// already removed the encoding.
export const STATIC_SOURCE_SEMANTIC_SUFFIX='.main-semantic.gz';
export const STATIC_SOURCE_SEMANTIC_MAGIC=0x3153534d;
export const STATIC_SOURCE_SEMANTIC_VERSION=1;
export const STATIC_SOURCE_SEMANTIC_MARKER='/* MAIN_STATIC_SOURCE_SEMANTIC_0101 */';

/** Build side. `assetDigest` is the 32-byte SHA-256 of the document bytes. */
export function encodeStaticSourceSemantics({assetDigest,meshes}){
 if(!(assetDigest instanceof Uint8Array)||assetDigest.byteLength!==32)throw Error('Static source semantics require a 32-byte asset digest');
 const header=3+8+meshes.length*3;let bytes=header*4;
 for(const mesh of meshes){if(!Number.isSafeInteger(mesh.vertexCount)||!Number.isSafeInteger(mesh.indexCount))throw Error('Static source semantics require integer mesh counts');bytes+=Math.ceil((mesh.semantic?.byteLength??0)/4)*4;}
 const out=new Uint8Array(bytes),words=new Uint32Array(out.buffer);
 words[0]=STATIC_SOURCE_SEMANTIC_MAGIC;words[1]=STATIC_SOURCE_SEMANTIC_VERSION;words[2]=meshes.length;
 out.set(assetDigest,12);
 let at=header*4;
 for(const [index,mesh]of meshes.entries()){
  const length=mesh.semantic?.byteLength??0;words[11+index*3]=mesh.vertexCount;words[12+index*3]=mesh.indexCount;words[13+index*3]=length;
  if(length)out.set(mesh.semantic,at);at+=Math.ceil(length/4)*4;
 }
 return out;
}

// The runtime half, injected verbatim into the engine's src/extras/gltf/load.ts
// (and executed as-is by tools/main-static-source-semantic.test.ts).
export const STATIC_SOURCE_SEMANTIC_RUNTIME=`${STATIC_SOURCE_SEMANTIC_MARKER}
/** Fetch the build-time semantic sidecar of this exact document, in parallel with it. */
function readStaticSourceSemantics(url: string | URL, read: typeof fetch): Promise<ArrayBuffer | undefined> {
    let target: URL;
    try {
        target = new URL(url);
    } catch {
        return Promise.resolve(undefined);
    }
    target.pathname += "${STATIC_SOURCE_SEMANTIC_SUFFIX}";
    target.search = "";
    target.hash = "";
    return Promise.resolve()
        .then(() => read(target))
        .then((response) => (response.ok ? response.arrayBuffer() : undefined))
        .then((bytes) => {
            if (!bytes || bytes.byteLength < 2) return undefined;
            const head = new Uint8Array(bytes, 0, 2);
            if (head[0] !== 0x1f || head[1] !== 0x8b) return bytes;
            if (typeof DecompressionStream === "undefined") return undefined;
            return new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"))).arrayBuffer();
        })
        .catch(() => undefined);
}
/** Attach each mesh's baked semantic residual only when the sidecar names this
 * exact document digest and every decoded mesh count; otherwise attach nothing. */
function attachStaticSourceSemantics(
    meshes: { vertices: Float32Array; indices: Uint32Array; semantic?: Uint8Array }[],
    digest: ArrayBuffer,
    sidecar: ArrayBuffer | undefined,
): number {
    if (!sidecar || sidecar.byteLength < 44 || sidecar.byteLength % 4) return 0;
    const words = new Uint32Array(sidecar);
    if (words[0] !== ${STATIC_SOURCE_SEMANTIC_MAGIC} || words[1] !== ${STATIC_SOURCE_SEMANTIC_VERSION} || words[2] !== meshes.length) return 0;
    const header = 11 + meshes.length * 3;
    if (sidecar.byteLength < header * 4) return 0;
    const expected = new Uint8Array(digest), named = new Uint8Array(sidecar, 12, 32);
    if (expected.byteLength !== 32) return 0;
    for (let i = 0; i < 32; i++) if (expected[i] !== named[i]) return 0;
    const ranges: [number, number][] = [];
    let at = header * 4;
    for (let index = 0; index < meshes.length; index++) {
        const mesh = meshes[index]!, length = words[13 + index * 3]!;
        if (words[11 + index * 3] !== mesh.vertices.length / 8 || words[12 + index * 3] !== mesh.indices.length) return 0;
        if (at + length > sidecar.byteLength) return 0;
        ranges.push([at, length]);
        at += Math.ceil(length / 4) * 4;
    }
    if (at !== sidecar.byteLength) return 0;
    let attached = 0;
    for (let index = 0; index < meshes.length; index++) {
        const [start, length] = ranges[index]!;
        if (!length) continue;
        meshes[index]!.semantic = new Uint8Array(sidecar.slice(start, start + length));
        attached++;
    }
    return attached;
}
`;

function once(text,from,to,label){if(text.split(from).length!==2)throw Error('MAIN static source semantic anchor changed: '+label);return text.replace(from,to);}
/** Pure transform of the engine's (material-metadata patched) load.ts. */
export function transformGltfLoadStaticSemantics(load){
 if(load.includes(STATIC_SOURCE_SEMANTIC_MARKER))return load;
 if(!load.includes('MAIN_SOURCE_MATERIAL_METADATA_0103'))throw Error('MAIN static source semantics require the material metadata patch first');
 const entry='export async function loadGltf(url: string | URL, read: typeof fetch = fetch): Promise<GltfScene> {\n    const { base, json, buffers, bytes } = await resolveGltf(url, read);';
 load=once(load,entry,'export async function loadGltf(url: string | URL, read: typeof fetch = fetch): Promise<GltfScene> {\n    const staticSemantics = readStaticSourceSemantics(url, read);\n    const { base, json, buffers, bytes } = await resolveGltf(url, read);','loadGltf entry');
 const assign='    for (const mesh of scene.meshes) mesh.materialSource = { kind: "gltf-material", asset, material: mesh.material };';
 load=once(load,assign,assign+'\n    attachStaticSourceSemantics(scene.meshes, digest, await staticSemantics);','material source assignment');
 return load+'\n'+STATIC_SOURCE_SEMANTIC_RUNTIME;
}
/** GltfMesh carries the attached residual into Meshes.add (MeshData.semantic). */
export function transformGltfMeshStaticSemantics(gltf){
 if(gltf.includes('MAIN_STATIC_SOURCE_SEMANTIC_0101'))return gltf;
 return once(gltf,'export interface GltfMesh {','export interface GltfMesh {\n    /* MAIN_STATIC_SOURCE_SEMANTIC_0101: build-time semantic residual of this exact document mesh. */\n    semantic?: Uint8Array;','gltf mesh');
}
export async function applyStaticSourceSemantic(engine){
 for(const [relative,transform]of [['src/extras/gltf/load.ts',transformGltfLoadStaticSemantics],['src/extras/gltf/gltf.ts',transformGltfMeshStaticSemantics]]){
  const path=resolve(engine,relative),before=await readFile(path,'utf8'),after=transform(before);
  if(after!==before)await writeFile(path,after);
 }
}
