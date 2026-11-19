/**
 * Generic GPU-indexed-source metadata seam for the isolated Shallot engine.
 *
 * This transform does not choose a scene, allocate a CPU mesh, or change an
 * imported producer. It only makes an explicitly attached producer contract
 * survive Mesh registration and the existing material-geometry visitor. Run
 * it after apply.mjs, because apply.mjs owns the visitor's main suffix.
 */
import {readFile,writeFile} from 'node:fs/promises';
import {resolve} from 'node:path';

export const GPU_INDEXED_SOURCE_PRODUCER_MARKER='/* TRUEFF_MAIN_GPU_INDEXED_SOURCE_PRODUCER_0103 */';
export const GPU_INDEXED_SOURCE_METADATA_TYPE=`{
        apiVersion: 1;
        sourceId: string;
        revision: number;
        generation: number;
        topologyKey: string;
        topology: {
            indexPattern?: "independent-quads";
            indexFormat: "uint32";
            indexCapacity: number;
            vertexCapacity: number;
            indices: GPUBufferBinding;
        };
        liveIndexCount: GPUBufferBinding;
        registrationIndexCount?: number;
        constructionKind?: "terrain" | "planar" | "rounded" | "path-led" | "far-strip" | "paper";
        posedNormalClass?: "face-constant";
        edgeContactClass?: "posed-coincident";
        parts?: { kind: "closed-cells"; extent: readonly [number, number, number]; cellCodes: GPUBufferBinding };
        restCellChart?: { kind: "voxel-rest-cell-chart"; sourceGeneration: number; cornerOrder: "shallot-voxel-quad-v1"; extent: readonly [number, number, number]; vertexCellCodes: GPUBufferBinding };
        poseContent?: {
            kind: "voxel-terrain-material";
            sourceGeneration: number;
            revision: number;
            vertexRevision: number;
            instanceTransform: readonly [number, number, number, number, number, number, number, number, number, number];
            materialParameters: readonly [number, number];
        };
    }`;
export const GPU_INDEXED_SOURCE_VOXEL_MARKER='/* TRUEFF_MAIN_GPU_INDEXED_SOURCE_VOXEL_0103 */';

function once(text,from,to,label){
 const matches=text.split(from).length-1;
 if(matches!==1)throw Error(`MAIN GPU indexed source anchor ${label} expected once, found ${matches}`);
 return text.replace(from,to);
}

/** Pure text transform used by setup and by the CPU fixture test. */
export function transformGpuIndexedSourceSources({mesh,forward}){
 const meshMarker=`${GPU_INDEXED_SOURCE_PRODUCER_MARKER} mesh`;
 const forwardMarker=`${GPU_INDEXED_SOURCE_PRODUCER_MARKER} forward`;
 if(!mesh.includes(meshMarker)){
  const anchor='    /** the 8 B/vertex position-only stream';
  mesh=once(mesh,anchor,`    ${meshMarker}\n    /** Optional producer-owned indexed topology. The producer's live count\n     * binding is authoritative; this field carries no CPU positions. */\n    gpuIndexedSource?: ${GPU_INDEXED_SOURCE_METADATA_TYPE};\n${anchor}`,'Mesh metadata');
 }
 if(!forward.includes(forwardMarker)){
  const declarations=forward.split('\n').filter(line=>line.includes('    sourceGeometry?:')&&line.includes('Float32Array'));
  if(declarations.length!==1)throw Error(`MAIN GPU indexed source visitor declaration expected once, found ${declarations.length}`);
  const declaration=declarations[0];
  forward=once(forward,declaration,`${forwardMarker}\n${declaration}\n    /** Generic GPU source metadata attached by the registered producer. */\n    gpuIndexedSource?: ${GPU_INDEXED_SOURCE_METADATA_TYPE};`,'visitor metadata');
  const publication='            sourceGeometry: world.resource(Meshes).get(item.draw.mesh)!.sourceGeometry,';
  forward=once(forward,publication,`${publication}\n            gpuIndexedSource: world.resource(Meshes).get(item.draw.mesh)!.gpuIndexedSource,`,'visitor publication');
 }
 // Existing installations already carry the metadata marker. Upgrade that
 // declaration too; setup must not depend on deleting the isolated engine.
 const kind='        constructionKind?: "terrain" | "planar" | "rounded" | "path-led" | "far-strip" | "paper";';
 if(!mesh.includes('posedNormalClass?:'))mesh=once(mesh,kind,kind+'\n        posedNormalClass?: "face-constant";','Mesh posed-normal fact');
 if(!forward.includes('posedNormalClass?:'))forward=once(forward,kind,kind+'\n        posedNormalClass?: "face-constant";','visitor posed-normal fact');
 if(!mesh.includes('edgeContactClass?:'))mesh=once(mesh,kind,kind+'\n        edgeContactClass?: "posed-coincident";','Mesh edge-contact authority');
 if(!forward.includes('edgeContactClass?:'))forward=once(forward,kind,kind+'\n        edgeContactClass?: "posed-coincident";','visitor edge-contact authority');
 const parts='\n        parts?: { kind: "closed-cells"; extent: readonly [number, number, number]; cellCodes: GPUBufferBinding };';
 if(!mesh.includes('parts?: { kind: "closed-cells"'))mesh=once(mesh,kind,kind+parts,'Mesh construction parts');
 if(!forward.includes('parts?: { kind: "closed-cells"'))forward=once(forward,kind,kind+parts,'visitor construction parts');
 const restCellChart='        restCellChart?: { kind: "voxel-rest-cell-chart"; sourceGeneration: number; cornerOrder: "shallot-voxel-quad-v1"; extent: readonly [number, number, number]; vertexCellCodes: GPUBufferBinding };';
 const legacyRestCellChart='        restCellChart?: { kind: "voxel-rest-cell-chart"; sourceGeneration: number; extent: readonly [number, number, number]; vertexCellCodes: GPUBufferBinding };';
 const upgradeRestCellChart=(text,label)=>{
  const declarations=text.split('\n').filter(line=>line.includes('restCellChart?:'));
  if(declarations.length===0)return once(text,parts,parts+'\n'+restCellChart,label);
  if(declarations.length!==1)throw Error(`MAIN GPU indexed source ${label} expected one rest-cell chart declaration, found ${declarations.length}`);
  const declaration=declarations[0];
  if(declaration===restCellChart)return text;
  if(declaration===legacyRestCellChart)return once(text,declaration,restCellChart,label+' legacy schema');
  throw Error(`MAIN GPU indexed source ${label} has an unexpected rest-cell chart declaration`);
 };
 mesh=upgradeRestCellChart(mesh,'Mesh rest-cell chart');
 forward=upgradeRestCellChart(forward,'visitor rest-cell chart');
 const poseContent='\n        poseContent?: { kind: "voxel-terrain-material"; sourceGeneration: number; revision: number; vertexRevision: number; instanceTransform: readonly [number, number, number, number, number, number, number, number, number, number]; materialParameters: readonly [number, number] };';
 if(!mesh.includes('poseContent?:'))mesh=once(mesh,parts,parts+poseContent,'Mesh pose-content certificate');
 if(!forward.includes('poseContent?:'))forward=once(forward,parts,parts+poseContent,'visitor pose-content certificate');
 return {mesh,forward};
}

/**
 * Build-only transform for the original voxel producer. The checked-in
 * imported terrain.ts is never rewritten: Vite (or another build caller)
 * passes its source through this function and bundles the returned module.
 * The transform uses the producer's actual slot x/z/allFaces and the exact
 * counts offset already assigned to geometry.mesh.indirectIndexCount.
 */
export function transformVoxelTerrainSource(source,{importSpecifier='../../../../packages/shallot-main/gpu-indexed-source-producer.ts'}={}){
 if(source.includes(GPU_INDEXED_SOURCE_VOXEL_MARKER))return source;
 const importAnchor='import { type MeshHandle, MeshPlugin, MeshQuant } from "@dylanebert/shallot/mesh";';
 source=once(source,importAnchor,`${importAnchor}\nimport { GpuIndexedSourceProducerRuntime, gpuIndexedSourceTopologyKey } from "${importSpecifier}";`,'voxel producer import');
 const resourceAnchor='    emitPass: GPUComputePassDescriptor;';
 source=once(source,resourceAnchor,`${resourceAnchor}\n    producer: GpuIndexedSourceProducerRuntime | null;`,'voxel producer resource type');
 const slotAnchor='    geometry: Geometry;\n    dirty: boolean;';
 source=once(source,slotAnchor,`    geometry: Geometry;\n    /** Counts source-buffer emission changes independently of topology generation. */\n    poseRevision: number;\n    dirty: boolean;\n    /** A wrapped chunk calls resetInterpolation; omit pose retention for that frame. */\n    poseDiscontinuous: boolean;`,'voxel pose revision');
 const terrainAnchor='export const Terrain: Resource<{';
 source=once(source,terrainAnchor,`/** Exact CPU-owned inputs of TerrainMaterial.vertex for one chunk draw.\n * The producer updates this certificate whenever emission, chunk Transform, or\n * the material's radial parameters can change. Unknown stages do not call it.\n * The transform is the same previous/current interpolation used by Shallot's\n * global-transform render table; a wrapped chunk fails closed for one frame. */\nfunction voxelPoseContent(slot: Pick<Slot, "x" | "z" | "poseRevision" | "poseDiscontinuous">, previousViewer: number, viewer: number, alpha: number, config: TerrainConfig) {\n    if (slot.poseDiscontinuous) return undefined;\n    const { inner, outer } = radii(config);\n    const currentX = chunkTranslation(slot.x, viewer, config);\n    const previousX = chunkTranslation(slot.x, previousViewer, config);\n    const x = previousX * (1 - alpha) + currentX * alpha;\n    return {\n        kind: "voxel-terrain-material" as const,\n        vertexRevision: slot.poseRevision,\n        instanceTransform: [\n            x, -config.height / 2, chunkTranslation(slot.z, 0, config),\n            0, 0, 0, 1, 1, 1, 1,\n        ] as const,\n        materialParameters: [inner, outer] as const,\n    };\n}\n${terrainAnchor}`,'voxel pose certificate helper');
 const createAnchor='        emitPass: { label: "voxel:emit" },';
 source=once(source,createAnchor,`${createAnchor}\n        producer: null,`,'voxel producer resource create');
 const initializeAnchor='            const { device, root } = world.gpu;';
 source=once(source,initializeAnchor,`${initializeAnchor}\n            s.producer = new GpuIndexedSourceProducerRuntime(device);\n            world.onDispose(() => {\n                s.producer?.dispose();\n                s.producer = null;\n            });`,'voxel producer initialization');
 const updateAnchor='                    const s = world.resource(Terrain);\n                    let pass: GPUComputePassEncoder | undefined;';
 source=once(source,updateAnchor,`                    const s = world.resource(Terrain);\n                    s.producer?.rethrowFrameSubmitError();\n                    let pass: GPUComputePassEncoder | undefined;`,'voxel producer frame error');
 const wrapAnchor='            slot.x = x;\n            slot.dirty = true;';
 source=once(source,wrapAnchor,`            slot.x = x;\n            slot.poseDiscontinuous = true;\n            slot.dirty = true;`,'voxel wrapped transform discontinuity');
 const addAnchor='                geometry.mesh.indirectIndexCount!.offset = i * 4;\n                const handle = world.resource(Meshes).add(geometry.mesh);';
 const addReplacement=`                geometry.mesh.indirectIndexCount!.offset = i * 4;
                const producerIndices = root.unwrap(geometry.mesh.indices);
                const producerVertices = root.unwrap(geometry.mesh.vertices);
                s.producer!.observeMesh(geometry.mesh, {
                    sourceId: \`voxel/chunk-slot/\${i}\`,
                    topologyKey: gpuIndexedSourceTopologyKey({ slot: i, x, z, allFaces: geometry.allFaces }),
                    indexCapacity: geometry.mesh.indexCount,
                    vertexCapacity: Math.floor(producerVertices.size / 16),
                    indices: { buffer: producerIndices, offset: geometry.mesh.indexBase * 4, size: geometry.mesh.indexCount * 4 },
                    liveIndexCount: { buffer: s.counts!, offset: i * 4, size: 4 },
                    constructionKind: "terrain",
                    indexPattern: "independent-quads",
                    posedNormalClass: "face-constant",
                    edgeContactClass: "posed-coincident",
                    poseContent: voxelPoseContent({ x, z, poseRevision: 0, poseDiscontinuous: false }, s.previousViewer, s.viewer, 0, config),
                    restCellChart: { kind: "voxel-rest-cell-chart", cornerOrder: "shallot-voxel-quad-v1", extent: [config.chunkSize, config.height, config.chunkSize], vertexCellCodes: { buffer: root.unwrap(geometry.mesh.attributes!.voxel!), size: Math.floor(producerVertices.size / 16) * 4 } },
                    parts: geometry.allFaces ? { kind: "closed-cells", extent: [config.chunkSize, config.height, config.chunkSize], cellCodes: { buffer: root.unwrap(geometry.mesh.attributes!.voxel!), size: Math.floor(producerVertices.size / 16) * 4 } } : undefined,
                });
                const handle = world.resource(Meshes).add(geometry.mesh);`;
 source=once(source,addAnchor,addReplacement,'voxel producer initial mesh attachment');
 const slotInitAnchor='                    geometry,\n                    dirty: true,';
 source=once(source,slotInitAnchor,`                    geometry,\n                    poseRevision: 0,\n                    poseDiscontinuous: false,\n                    dirty: true,`,'voxel pose revision initialization');
 const emitAnchor='                        world.gpu.device.queue.writeBuffer(geometry.uniform, 0, geometry.input);';
 const emitReplacement=`                        world.gpu.device.queue.writeBuffer(geometry.uniform, 0, geometry.input);
                        const producerIndices = world.gpu.root.unwrap(geometry.mesh.indices);
                        const producerVertices = world.gpu.root.unwrap(geometry.mesh.vertices);
                        s.producer!.observeMesh(geometry.mesh, {
                            sourceId: \`voxel/chunk-slot/\${i}\`,
                            topologyKey: gpuIndexedSourceTopologyKey({ slot: i, x: slot.x, z: slot.z, allFaces: geometry.allFaces }),
                            indexCapacity: geometry.mesh.indexCount,
                            vertexCapacity: Math.floor(producerVertices.size / 16),
                            indices: { buffer: producerIndices, offset: geometry.mesh.indexBase * 4, size: geometry.mesh.indexCount * 4 },
                            liveIndexCount: { buffer: s.counts!, offset: i * 4, size: 4 },
                            constructionKind: "terrain",
                            indexPattern: "independent-quads",
                            posedNormalClass: "face-constant",
                            edgeContactClass: "posed-coincident",
                            poseContent: voxelPoseContent(slot, s.previousViewer, s.viewer, world.time.fixedAlpha, config),
                            restCellChart: { kind: "voxel-rest-cell-chart", cornerOrder: "shallot-voxel-quad-v1", extent: [config.chunkSize, config.height, config.chunkSize], vertexCellCodes: { buffer: world.gpu.root.unwrap(geometry.mesh.attributes!.voxel!), size: Math.floor(producerVertices.size / 16) * 4 } },
                            parts: geometry.allFaces ? { kind: "closed-cells", extent: [config.chunkSize, config.height, config.chunkSize], cellCodes: { buffer: world.gpu.root.unwrap(geometry.mesh.attributes!.voxel!), size: Math.floor(producerVertices.size / 16) * 4 } } : undefined,
                        });
                        slot.poseDiscontinuous = false;`;
 source=once(source,emitAnchor,emitReplacement,'voxel producer dirty emission');
 const poseAnchor='                        const slot = s.slots[i];\n                        if (!slot.dirty) continue;';
 source=once(source,poseAnchor,`                        const slot = s.slots[i];\n                        if (slot.dirty) slot.poseRevision++;\n                        s.producer?.observePoseContent(\`voxel/chunk-slot/\${i}\`, voxelPoseContent(slot, s.previousViewer, s.viewer, world.time.fixedAlpha, config));\n                        if (!slot.dirty) continue;`,'voxel pose certificate update');
 const endAnchor='                    if (pass) endComputePass(pass);';
 source=once(source,endAnchor,`                    if (pass) {\n                        endComputePass(pass);\n                        // Copies are encoded after voxel:emit closes, then mapped by the host after submit.\n                        s.producer?.encodeCountCopies(world.frameEncoder()!);\n                        // World invokes this callback after its ordinary queue submission.\n                        world.onFrameSubmit(() => onTerrainGpuIndexedSourceFrameSubmit(world));\n                    }`,'voxel producer count copy');
 const exportAnchor='export const TerrainPlugin = terrainPlugin(CONFIG);';
 source=once(source,exportAnchor,`${exportAnchor}
${GPU_INDEXED_SOURCE_VOXEL_MARKER}
/** Internal host hook: World invokes it after its ordinary queue submission. */
function onTerrainGpuIndexedSourceFrameSubmit(world: World): void {
    const producer = world.resource(Terrain).producer;
    if (!producer) return;
    void producer.onFrameSubmit().catch((error) => producer.captureFrameSubmitError(error));
}`,'voxel producer host hook');
 return source;
}

/** Vite-facing build hook. It matches the imported producer by resolved file
 * identity, not by scene name or renderer state, and returns a virtual
 * transformed module while leaving the checked-in source untouched. */
export function gpuIndexedSourceVoxelVitePlugin({sourcePath,importSpecifier}={}){
 const expected=sourcePath?resolve(sourcePath):undefined;
 return {
  name:'shallot-main-gpu-indexed-source-voxel',
  enforce:'pre',
  transform(code,id){
   const clean=id.split('?')[0];
   if(expected&&resolve(clean)!==expected)return null;
   if(!expected&&!clean.endsWith('/examples/shallot/voxel/src/terrain.ts'))return null;
   return {code:transformVoxelTerrainSource(code,{importSpecifier}),map:null};
  },
 };
}




export async function applyGpuIndexedSourceProducer(engine){
 const version=JSON.parse(await readFile(resolve(engine,'package.json'),'utf8')).version;
 if(version!=='0.10.5')throw Error(`MAIN GPU indexed source producer requires Shallot 0.10.5, received ${version}`);
 const meshPath=resolve(engine,'src/core/mesh/mesh.ts');
 const forwardPath=resolve(engine,'src/standard/rendering/forward.ts');
 const [mesh,forward]=await Promise.all([readFile(meshPath,'utf8'),readFile(forwardPath,'utf8')]);
 const transformed=transformGpuIndexedSourceSources({mesh,forward});
 if(transformed.mesh!==mesh)await writeFile(meshPath,transformed.mesh);
 if(transformed.forward!==forward)await writeFile(forwardPath,transformed.forward);
 return transformed;
}
