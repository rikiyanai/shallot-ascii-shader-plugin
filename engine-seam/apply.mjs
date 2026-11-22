/** Durable, version-checked engine seam. Applies only to the isolated example
 * engine, never changes the imported example files. */
import {readFile,writeFile} from 'node:fs/promises';
import {resolve,dirname} from 'node:path';
import {fileURLToPath} from 'node:url';
const project=resolve(dirname(fileURLToPath(import.meta.url)),'..');
const engine=resolve(process.argv[2]??resolve(project,'node_modules/@dylanebert/shallot'));
const version=JSON.parse(await readFile(resolve(engine,'package.json'),'utf8')).version;
if(version!=='0.10.5')throw Error(`MAIN material geometry patch requires Shallot 0.10.5, received ${version}`);
const marker='/* TRUEFF_MAIN_GEOMETRY_0103 */';
const pipelinePath=resolve(engine,'src/standard/rendering/pipelines.ts');
const forwardPath=resolve(engine,'src/standard/rendering/forward.ts');
const preprocessPath=resolve(engine,'src/standard/rendering/preprocess.ts');
const barrelPath=resolve(engine,'src/standard/rendering/index.ts');
const instancePath=resolve(engine,'src/standard/rendering/instance-data.ts');
const framePath=resolve(engine,'src/core/rendering/frame-state.ts');
const substratePath=resolve(engine,'src/core/rendering/substrate.ts');
const viewPath=resolve(engine,'src/core/rendering/view.ts');
const coreBarrelPath=resolve(engine,'src/core/rendering/index.ts');
const meshPath=resolve(engine,'src/core/mesh/mesh.ts');
const primitivePath=resolve(engine,'src/core/mesh/primitives.ts');
const phasesPath=resolve(engine,'src/core/rendering/phases.ts');
const gpuPath=resolve(engine,'src/engine/runtime/gpu.ts');
const runtimeBarrelPath=resolve(engine,'src/engine/runtime/index.ts');
const physicsRuntimePath=resolve(engine,'src/standard/physics/runtime.ts');
const physicsBarrelPath=resolve(engine,'src/standard/physics/index.ts');
const loadingPath=resolve(engine,'src/standard/loading/index.ts');
const appPath=resolve(engine,'src/engine/app/index.ts');
const skinningPath=resolve(engine,'src/extras/gltf/skinning.ts');
let pipelines=await readFile(pipelinePath,'utf8');
let forward=await readFile(forwardPath,'utf8');
let preprocess=await readFile(preprocessPath,'utf8');
let barrel=await readFile(barrelPath,'utf8');
let instance=await readFile(instancePath,'utf8');
let frame=await readFile(framePath,'utf8');
let substrate=await readFile(substratePath,'utf8');
let view=await readFile(viewPath,'utf8');
let coreBarrel=await readFile(coreBarrelPath,'utf8');
let mesh=await readFile(meshPath,'utf8');
let primitives=await readFile(primitivePath,'utf8');
let phases=await readFile(phasesPath,'utf8');
let gpu=await readFile(gpuPath,'utf8');
let runtimeBarrel=await readFile(runtimeBarrelPath,'utf8');
let physicsRuntime=await readFile(physicsRuntimePath,'utf8');
let physicsBarrel=await readFile(physicsBarrelPath,'utf8');
let loading=await readFile(loadingPath,'utf8');
let app=await readFile(appPath,'utf8');
let skinning=await readFile(skinningPath,'utf8');
const {transformLoadingDiagnostics,transformLoadingErrorPriority,transformPluginHookCause}=await import('./loading-diagnostics.mjs');
// Validate and prepare the diagnostics seam before any other patch writes.
// The final writes stay with the rest of this script's declared write phase.
loading=transformLoadingDiagnostics(loading);
loading=transformLoadingErrorPriority(loading);
app=transformPluginHookCause(app);
const phaseSource=await readFile(resolve(project,'engine-seam/engine/phase-geometry.ts'),'utf8');
const presentationTicketSource=await readFile(resolve(project,'engine-seam/engine/presentation-ticket.ts'),'utf8');
const presentationTargetsSource=await readFile(resolve(project,'engine-seam/engine/presentation-targets.ts'),'utf8');
const pipelineSource=await readFile(resolve(project,'engine-seam/engine/pipeline-source.ts'),'utf8');
const physicsContactObservationSource=await readFile(resolve(project,'engine-seam/engine/physics-contact-observation.ts'),'utf8');
const instanceAnchor='export function instanceDataRoot(\n    world: World,\n    root: ReturnType<World["gpu"]["root"]["with"]>,';
const genericInstance='export function instanceDataRoot<T extends { with(slot: any, value: any): T }>(\n    world: World,\n    root: T,';
if(instance.includes(instanceAnchor))instance=instance.replace(instanceAnchor,genericInstance);
else if(!instance.includes(genericInstance))throw Error('MAIN geometry patch: instance-data slot configuration changed');
const physicsContactMarker='/* TRUEFF_MAIN_PHYSICS_CONTACT_OBSERVATION_0105 */';
if(physicsRuntime.includes(physicsContactMarker))physicsRuntime=physicsRuntime.slice(0,physicsRuntime.indexOf(physicsContactMarker));
if(physicsBarrel.includes(physicsContactMarker))physicsBarrel=physicsBarrel.slice(0,physicsBarrel.indexOf(physicsContactMarker));
const physicsContactImport='import { ContactField, ContactFlags, contactCapacity, contactField } from "./collision/contact";';
if(!physicsRuntime.includes(physicsContactImport)){
  const contactImport='import { ContactDataBuffer, type PhysicsContactData } from "./contact";';
  if(physicsRuntime.split(contactImport).length!==2)throw Error('MAIN physics contact observation: contact import changed');
  physicsRuntime=physicsRuntime.replace(contactImport,contactImport+'\n'+physicsContactImport);
}
if(!physicsRuntime.includes(physicsContactMarker))physicsRuntime+='\n'+physicsContactObservationSource+'\n';
if(!physicsBarrel.includes(physicsContactMarker))physicsBarrel+='\n'+physicsContactMarker+'\nexport { observePhysicsContacts, type PhysicsContactObservationStamp, type PhysicsContactVisitor } from "./runtime";\n';
// Replace our own appended implementation on rerun; the version guard and
// anchors below still refuse an unrelated engine revision.
if(pipelines.includes(marker))pipelines=pipelines.slice(0,pipelines.indexOf(marker));
if(forward.includes(marker))forward=forward.slice(0,forward.indexOf(marker));
if(!pipelines.includes(marker))pipelines+=`
${marker}
const mainGeometryStages = { create: () => new WeakMap<CompiledMaterial, { vertex: object; vertexCode: string; layouts: GPUBindGroupLayout[] }>() };
/** Resolve the exact color vertex variant with the same slot substitutions.
 * Consumers retain the real vertex-stage bindings and shader visibility. */
export function materialGeometryStage(world: World, compiled: CompiledMaterial) {
    const stages = world.resource(mainGeometryStages);
    const cached = stages.get(compiled);
    if (cached) return cached;
    const surface = compiled.owner;
    const resolved = tgpu.resolveWithContext([compiled.args.vertex], {
        config: (cfg) => {
            let configured = instanceDataRoot(world, cfg.with(indirectLightSlot, indirectLightSources(world)), surface.layout, surface.instanceData, meshInstanceTable(world));
            for (const [accessor, value] of surface.slots ?? []) configured = configured.with(accessor, value);
            if (surface.readsPrepass?.depth) configured = configured.with(prepassDepthSlot, prepassDepthRead);
            if (surface.readsPrepass?.normal) configured = configured.with(prepassNormalSlot, prepassNormalRead);
            return configured;
        },
    });
    if (resolved.catchall || resolved.usedImmediate || resolved.logResources) {
        throw new Error('MAIN geometry refuses implicit shader bindings or shader logging');
    }
    const stage = {
        vertex: compiled,
        vertexCode: resolved.code,
        layouts: [engineLayout, shadowLayout, compiled.owner.layout].map((layout) => world.gpu.root.unwrap(layout)),
    };
    stages.set(compiled, stage);
    return stage;
}
`;
if(!forward.includes(marker)){
  const anchor='    meshStreamMask,\n';
  if(forward.split(anchor).length!==2)throw Error('MAIN geometry patch: pipeline import changed');
  if(!forward.includes('    materialGeometryStage,\n'))forward=forward.replace(anchor,anchor+'    materialGeometryStage,\n');
  const stageAnchor='const ResolveDrawsSystem: System = {';
  if(!forward.includes(stageAnchor))throw Error('MAIN geometry patch: draw resolution ordering changed');
  if(!forward.includes('export '+stageAnchor))forward=forward.replace(stageAnchor,'export '+stageAnchor);
  forward+=`
${marker}
/** Visit current engine-resolved draws after ResolveDrawsSystem, in the same
 * World frame. The callback must encode its GPU consumer before returning.
 * Geometry is the actual material vertex result, including GPU skin/morph and
 * material deformation. Alpha coverage and fragment shading are separate. */
export function visitMaterialGeometry(world: World, slot: number, consume: (draw: {
    name: string;
    vertex: object;
    vertexCode: string;
    layouts: GPUBindGroupLayout[];
    groups: GPUBindGroup[];
    indices: GPUBuffer;
    vertexCapacity: number;
    instanceCapacity?: number;
    cullMode: number;
    vertexBase?: number;
    mesh: number;
    indexBase: number;
    sourceGeometry?: { positions: Float32Array; indices: Uint32Array; contactKeys?: Uint32Array; revision: number };
    indirect: GPUBuffer;
    indirectOffset: number;
    bindings: Readonly<Record<string, unknown>>;
    material: MaterialType<any, any, any>;
}) => void): void {
    if (!world.frameEncoder()) throw new Error('MAIN geometry requires the current frame encoder');
    const state = world.resource(standardRendererStateKey);
    for (let i = 0; i < state.frameCount; i++) {
        const item = state.frameDraws[i]!;
        const stage = materialGeometryStage(world, item.r.t);
        const args = item.draw.args;
        consume({
            name: item.draw.name,
            ...stage,
            groups: [
                engineGroup(world, item.r.g.engineCache, slot, item.r.g.quant),
                shadowGroup(world),
                item.r.g.color,
            ],
            indices: world.gpu.root.unwrap(item.r.index),
            vertexCapacity: world.resource(Meshes).get(item.draw.mesh)!.vertexRange?.count ?? Math.floor(world.gpu.root.unwrap(world.resource(Meshes).get(item.draw.mesh)!.vertices).size / 16),
            cullMode: item.draw.cullMode ?? CullModeCode.Back,
            instanceCapacity: world.resource(producerTables).draws.get(item.draw.name)?.capacity
                ?? (isMeshInstanceDraw(world, item.draw) ? Math.max(1, meshInstanceTable(world).count) : undefined),
            vertexBase: world.resource(Meshes).get(item.draw.mesh)!.vertexRange?.first ?? 0,
            mesh: item.draw.mesh,
            indexBase: world.resource(Meshes).get(item.draw.mesh)!.indexBase,
            sourceGeometry: world.resource(Meshes).get(item.draw.mesh)!.sourceGeometry,
            indirect: world.gpu.root.unwrap(args.indirect),
            indirectOffset: (args.offset ?? 0) + slot * (args.viewStride ?? 0),
            bindings: item.r.g.values,
            material: item.r.t.owner,
        });
    }
}
`;
}
// Upgrade an already patched installation as well as a fresh package. This
// association is owned by the engine's draw registry, not a renderer name rule.
if(!forward.includes('    instanceCapacity?: number;')){
  const declaration='    vertexCapacity: number;';
  const value='            cullMode: item.draw.cullMode ?? CullModeCode.Back,';
  if(forward.split(declaration).length!==2||forward.split(value).length!==2)throw Error('MAIN instance extent: visitor contract changed');
  forward=forward.replace(declaration,declaration+'\n    instanceCapacity?: number;');
  forward=forward.replace(value,value+'\n            instanceCapacity: world.resource(producerTables).draws.get(item.draw.name)?.capacity\n                ?? (isMeshInstanceDraw(world, item.draw) ? Math.max(1, meshInstanceTable(world).count) : undefined),');
}
if(!barrel.includes(marker))barrel+=`\n${marker}\nexport { ResolveDrawsSystem, visitMaterialGeometry } from "./forward";\n`;
const presentationMarker='/* TRUEFF_MAIN_FINAL_PRESENTATION_0103 */';
if(!frame.includes(presentationMarker)){
  const anchor='        encodeUpscaling(world, world.resource(renderFrameKey).presentingViews);';
  if(frame.split(anchor).length!==2)throw Error('MAIN final presentation patch: native composition ordering changed');
  frame=frame.replace(anchor,anchor+'\n        for (const present of world.resource(FinalPresentationPasses))\n            present(world, world.resource(renderFrameKey).presentingViews);');
  frame+=`\n${presentationMarker}
/** Final-target presentation after native camera effects, viewports and stacking,
 * before captures and submission. Typography can stay at native pixels even
 * when the underlying camera uses a different render resolution. */
export const FinalPresentationPasses = {
    create: () => [] as ((world: World, views: readonly View[]) => void)[],
};\n`;
}
if(!coreBarrel.includes(presentationMarker))coreBarrel+=`\n${presentationMarker}\nexport { FinalPresentationPasses } from "./frame-state";\n`;
// Canvas acquisition is an engine-owned permission, not a renderer-side
// early return. The game requests one ticket before World.step; BeginFrame
// consumes it before getCurrentTexture and leaves simulation/draw preparation
// alive when no new presentation is needed.
const presentationTicketMarker='/* TRUEFF_MAIN_PRESENTATION_TICKET_0105 */';
const ticketImport='import { takePresentationTicket, type PresentationTicket } from "./presentation-ticket";';
const ticketTypeImport='import type { PresentationTicket } from "./presentation-ticket";';
if(!view.includes(presentationTicketMarker)){
  const anchor='import { releaseViewTargets } from "./targets";';
  if(view.split(anchor).length!==2)throw Error('MAIN presentation ticket: view imports changed');
  view=view.replace(anchor,anchor+'\n'+ticketTypeImport);
  const declaration='    /** Current canvas swapchain texture, retained through the frame for pre-presentation readback. */';
  if(view.split(declaration).length!==2)throw Error('MAIN presentation ticket: View declaration changed');
  view=view.replace(declaration,'    /** Engine-owned permission that authorized this frame\'s canvas acquisition. */\n    presentationTicket?: PresentationTicket;\n'+declaration);
  view+=`\n${presentationTicketMarker}\n`;
}else if(view.includes(ticketImport))view=view.replace(ticketImport,ticketTypeImport);
if(!substrate.includes(presentationTicketMarker)){
  const importAnchor='import { ClearChangeMarksSystem, invertMat4, unpackColor } from "../../engine";';
  if(substrate.split(importAnchor).length!==2)throw Error('MAIN presentation ticket: substrate imports changed');
  substrate=substrate.replace(importAnchor,importAnchor+'\n'+ticketImport);
  const viewTypeAnchor='    type View as ViewSlot,\n';
  if(substrate.split(viewTypeAnchor).length!==2)throw Error('MAIN presentation ticket: substrate view imports changed');
  const clearAnchor='            view.canvasTexture = undefined;';
  if(substrate.split(clearAnchor).length!==2)throw Error('MAIN presentation ticket: BeginFrame clear changed');
  substrate=substrate.replace(clearAnchor,clearAnchor+'\n            view.presentationTicket = undefined;');
  const acquired='        let acquiredTexture: GPUTexture | undefined;';
  if(substrate.split(acquired).length!==2)throw Error('MAIN presentation ticket: acquisition state changed');
  substrate=substrate.replace(acquired,acquired+'\n        let acquiredTicket: PresentationTicket | undefined;');
  const contextAnchor='            if (view.context && view.context !== acquiredContext) {\n                acquiredContext = view.context;\n                acquiredTexture = acquireCanvasTexture(world, view);\n            }';
  if(substrate.split(contextAnchor).length!==2)throw Error('MAIN presentation ticket: canvas acquisition site changed');
  substrate=substrate.replace(contextAnchor,`            if (view.context && view.context !== acquiredContext) {
                acquiredContext = view.context;
                acquiredTicket = takePresentationTicket(world);
                // The ticket helper preserves automatic acquisition for
                // unmanaged worlds. Only explicitly managed game worlds can
                // return undefined and suppress this canvas acquisition.
                acquiredTexture = acquiredTicket ? acquireCanvasTexture(world, view) : undefined;
            }`);
  const textureAnchor='            if (!texture) continue;';
  if(substrate.split(textureAnchor).length!==2)throw Error('MAIN presentation ticket: target selection changed');
  substrate=substrate.replace(textureAnchor,'            if (!texture) continue;\n            if (view.context && acquiredTexture === texture) view.presentationTicket = acquiredTicket;');
  substrate+=`\n${presentationTicketMarker}\n`;
}else substrate=substrate.replace(
  '                // No ticket means no canvas acquisition. Offscreen texture\n                // views remain valid without a ticket; only a canvas context\n                // is gated here.',
  '                // The ticket helper preserves automatic acquisition for\n                // unmanaged worlds. Only explicitly managed game worlds can\n                // return undefined and suppress this canvas acquisition.',
);
if(!frame.includes(presentationTicketMarker)){
  const clear='    view.present = null;';
  if(frame.split(clear).length!==2)throw Error('MAIN presentation ticket: frame target clear changed');
  frame=frame.replace(clear,clear+'\n    view.presentationTicket = undefined;');
  frame+=`\n${presentationTicketMarker}\n`;
}
const ticketBarrelMarker='/* TRUEFF_MAIN_PRESENTATION_TICKET_EXPORT_0105 */';
if(!coreBarrel.includes(ticketBarrelMarker))coreBarrel+=`\n${ticketBarrelMarker}\nexport { requestPresentationTicket, takePresentationTicket } from "./presentation-ticket";\nexport type { PresentationTicket } from "./presentation-ticket";\n`;

// world that asks for them (engine/presentation-targets.ts). Without its
// ticket a canvas is neither acquired nor presented, and its views render into
// an engine-owned stand-in so cameras, scene data and new GPU jobs continue
// targetless. A world that never asks keeps the world-level ticket path above
// byte for byte (the game's managed world and every unmanaged world).
const canvasPresentationMarker='/* TRUEFF_MAIN_CANVAS_PRESENTATION_0120 */';
if(!substrate.includes(canvasPresentationMarker)){
  const once=(text,anchor,replacement,what)=>{if(text.split(anchor).length!==2)throw Error(`MAIN canvas presentation: ${what} changed`);return text.replace(anchor,replacement);};
  substrate=once(substrate,ticketImport,ticketImport+'\nimport { canvasPresentationManaged, takeCanvasPresentationTicket, targetlessPresentationTexture } from "./presentation-targets";','ticket import');
  substrate=once(substrate,'        let acquiredTicket: PresentationTicket | undefined;','        let acquiredTicket: PresentationTicket | undefined;\n        let acquiredTargetless = false;','acquisition state');
  substrate=once(substrate,'                acquiredTicket = takePresentationTicket(world);',
    '                const canvasManaged = canvasPresentationManaged(world);\n                acquiredTicket = canvasManaged ? takeCanvasPresentationTicket(world, view.context) : takePresentationTicket(world);','ticket take');
  substrate=once(substrate,'                acquiredTexture = acquiredTicket ? acquireCanvasTexture(world, view) : undefined;',
    '                acquiredTexture = acquiredTicket\n                    ? acquireCanvasTexture(world, view)\n                    : canvasManaged\n                      ? targetlessPresentationTexture(world, view)\n                      : undefined;\n                acquiredTargetless = !acquiredTicket && acquiredTexture !== undefined;','acquisition');
  substrate=once(substrate,'            if (view.context && acquiredTexture === texture) view.presentationTicket = acquiredTicket;',
    '            if (view.context && acquiredTexture === texture) view.presentationTicket = acquiredTicket;\n            if (view.context && acquiredTargetless && acquiredTexture === texture) {\n                view.canvasTexture = undefined;\n                view.presentationTargetless = true;\n            }','ticket attach');
  substrate=once(substrate,'            view.presentationTicket = undefined;','            view.presentationTicket = undefined;\n            view.presentationTargetless = undefined;','BeginFrame clear');
  substrate+=`\n${canvasPresentationMarker}\n`;
}
if(!view.includes(canvasPresentationMarker)){
  const anchor='    presentationTicket?: PresentationTicket;\n';
  if(view.split(anchor).length!==2)throw Error('MAIN canvas presentation: View ticket declaration changed');
  view=view.replace(anchor,anchor+'    /** This frame renders into the canvas stand-in: not acquired, not presented. */\n    presentationTargetless?: boolean;\n');
  view+=`\n${canvasPresentationMarker}\n`;
}
if(!frame.includes(canvasPresentationMarker)){
  const anchor='    view.presentationTicket = undefined;';
  if(frame.split(anchor).length!==2)throw Error('MAIN canvas presentation: frame target clear changed');
  frame=frame.replace(anchor,anchor+'\n    view.presentationTargetless = undefined;')+`\n${canvasPresentationMarker}\n`;
}
if(!coreBarrel.includes(canvasPresentationMarker))coreBarrel+=`\n${canvasPresentationMarker}\nexport { requestCanvasPresentationTicket, canvasPresentationManaged, takeCanvasPresentationTicket, targetlessPresentationTexture } from "./presentation-targets";\n`;
if(!mesh.includes('vertexRange?: { first: number; count: number }')){
  const declaration='    vertices: MeshStorage<d.Vec4u>;';
  const packed='            indexCount: s.indexCount,';
  if(mesh.split(declaration).length!==2||mesh.split(packed).length!==2)throw Error('MAIN packed vertex range: mesh declaration or packing changed');
  mesh=mesh.replace(declaration,declaration+'\n    /** Absolute vertex extent in a packed family; absent for an unsliced producer. */\n    vertexRange?: { first: number; count: number };');
  mesh=mesh.replace(packed,packed+'\n            vertexRange: { first: s.vertexBase, count: s.vertexCount },');
}
mesh=mesh.replace('sourceGeometry?: { positions: Float32Array; indices: Uint32Array; revision: number }','sourceGeometry?: { positions: Float32Array; indices: Uint32Array; contactKeys?: Uint32Array; revision: number }');
if(!mesh.includes('sourceGeometry?: { positions: Float32Array; indices: Uint32Array; contactKeys?: Uint32Array; revision: number }')){
  const declaration='    vertexRange?: { first: number; count: number };';
  const packed='    for (const s of packed.slices) {\n        resources.meshes.replace(s.handle, {';
  const range='            vertexRange: { first: s.vertexBase, count: s.vertexCount },';
  if(mesh.split(declaration).length!==2||mesh.split(packed).length!==2||mesh.split(range).length!==2)throw Error('MAIN source topology: packed source layout changed');
  mesh=mesh.replace(declaration,declaration+'\n    /** Immutable source incidence; material-deformed positions are evaluated separately. */\n    sourceGeometry?: { positions: Float32Array; indices: Uint32Array; contactKeys?: Uint32Array; revision: number };');
  mesh=mesh.replace(packed,`    for (const s of packed.slices) {
        const sourcePositions = new Float32Array(s.vertexCount * 3);
        for (let v = 0; v < s.vertexCount; v++) for (let axis = 0; axis < 3; axis++)
            sourcePositions[v * 3 + axis] = packed.vertices[(s.vertexBase + v) * VERTEX_FLOATS + axis];
        const sourceIndices = packed.indices.slice(s.indexBase, s.indexBase + s.indexCount);
        for (let i = 0; i < sourceIndices.length; i++) sourceIndices[i] -= s.vertexBase;
        resources.meshes.replace(s.handle, {`);
  mesh=mesh.replace(range,range+'\n            sourceGeometry: { positions: sourcePositions, indices: sourceIndices, revision: 0 },');
}
if(!mesh.includes('const sourceContactKeys =')){
  const anchor='        const sourcePositions = new Float32Array(s.vertexCount * 3);';
  if(mesh.split(anchor).length!==2)throw Error('MAIN source contact keys: native quantized layout changed');
  mesh=mesh.replace(anchor,anchor+`
        const sourceContactKeys = new Uint32Array(s.vertexCount * 2);
        for (let v = 0; v < s.vertexCount; v++) {
            const at = (s.vertexBase + v) * 4;
            sourceContactKeys[v * 2] = q.main[at];
            sourceContactKeys[v * 2 + 1] = q.main[at + 1] & 0xffff;
        }`);
  mesh=mesh.replace('sourceGeometry: { positions: sourcePositions, indices: sourceIndices, revision: 0 }','sourceGeometry: { positions: sourcePositions, indices: sourceIndices, contactKeys: sourceContactKeys, revision: 0 }');
}
const phaseMarker='/* TRUEFF_MAIN_PHASE_GEOMETRY_0103 */';
if(!phases.includes(phaseMarker)){
  const declaration='export interface PhaseRenderer {';
  if(phases.split(declaration).length!==2)throw Error('MAIN phase geometry: renderer declaration changed');
  const start=phases.indexOf('function recordMainPass('),end=phases.indexOf('export const PrepassSystem:');
  if(start<0||end<start)throw Error('MAIN phase geometry: native main pass changed');
  let main=phases.slice(start,end);
  const opaque='    for (let i = 0; i < renderers.length; i++) renderers[i].opaque?.(this, eid, view, pass);';
  const after='    for (let i = 0; i < renderers.length; i++) renderers[i].afterOpaque?.(this, eid, view, pass);';
  const transparent='        renderer.drawTransparent(this, eid, view, phase, index, end, pass, item.renderer);';
  if(!main.includes(opaque)||!main.includes(after)||!main.includes(transparent)||!main.includes('    const pass = beginRenderPass('))throw Error('MAIN phase geometry: native draw ordering changed');
  main=main.replace('    const pass = beginRenderPass(','    const nativePass = beginRenderPass(');
  main=main.replace(opaque,`    const geometry = captureGeometryPass(this, view, nativePass);\n    const pass = geometry.pass;\n    for (let i = 0; i < renderers.length; i++) {\n        geometry.source(renderers[i], 'opaque');\n        renderers[i].opaque?.(this, eid, view, pass);\n    }`);
  main=main.replace(after,`    for (let i = 0; i < renderers.length; i++) {\n        geometry.source(renderers[i], 'after-opaque');\n        renderers[i].afterOpaque?.(this, eid, view, pass);\n    }`);
  main=main.replace(transparent,`        geometry.source(renderer, 'transparent', item.stableId, item.stablePart);\n${transparent}`);
  main=main.replace('    endRenderPass(pass);','    endRenderPass(nativePass);');
  phases=phases.slice(0,start)+main+phases.slice(end);
  phases=phases.replace(declaration,declaration+'\n    /** Engine-owned geometry adapter for this phase; other phases are observed as native draws. */\n    readonly geometryAdapter?: "standard-material";');
  phases=`${phaseMarker}\nimport { captureGeometryPass } from './phase-geometry';\n`+phases;
}
// ASCII's primary material producer owns the colour phase. Original beauty
// remains reachable only when that producer is not selected by the plugin.
const primaryMarker='/* TRUEFF_MAIN_PRIMARY_APPEARANCE_0116 */';
if(!phases.includes(primaryMarker)){
 const start=phases.indexOf('function recordMainPass('),end=phases.indexOf('export const PrepassSystem:');
 if(start<0||end<start)throw Error('MAIN primary phase changed');
 let main=phases.slice(start,end);
 const a=main.indexOf('    const targets = colorTargets('),b=main.indexOf('    const geometry = captureGeometryPass(',a);
 if(a<0||b<0||!main.includes('    endRenderPass(nativePass);'))throw Error('MAIN primary phase anchors changed');
 main=main.slice(0,a)+  '    const capture = this.resource(PhaseGeometryCapture);\n'+
  '    const primary = capture.enabled ? capture.primary?.(this, view, eid) : undefined;\n'+
  '    const nativePass = primary ? undefined : beginRenderPass(encoder, colorPassDescriptor(this, eid, colorTargets(this, eid, view.width, view.height, this.storage(Camera).antialias.get(eid) !== 0), view.framebuffer, !view.sortedIndexForTarget));\n'+main.slice(b);
 main=main.replace('    endRenderPass(nativePass);','    if (nativePass) endRenderPass(nativePass);\n    primary?.();');
 phases=primaryMarker+'\n'+phases.slice(0,start)+main+phases.slice(end);
 phases=phases.replace("import { captureGeometryPass } from './phase-geometry';","import { captureGeometryPass, PhaseGeometryCapture } from './phase-geometry';");
 phases=phases.replace('readonly geometryAdapter?: "standard-material";','readonly geometryAdapter?: "standard-material" | "screen-effect";');
}

// post-main original-beauty image pass is scheduled: the MAIN page clears and
// covers the final target. Tonemapping's camera loop also drives the
// EffectPasses before/after chain (bloom, effect stack); outline encodes its
// own passes into the framebuffer. Depth, shadows, SSAO, fog and sky are kept:
// the replayed field shading (fieldInputs) reads them.
{
 const {transformPrimaryImageTonemapping,transformPrimaryImageOutline}=await import('./primary-image.mjs');
 for(const [relative,transform]of [['src/core/rendering/tonemapping-state.ts',transformPrimaryImageTonemapping],['src/extras/outline/index.ts',transformPrimaryImageOutline]]){
  const path=resolve(engine,relative),before=await readFile(path,'utf8'),after=transform(before);
  if(after!==before)await writeFile(path,after);
 }
}
// Fullscreen fog is an image composite, not a one-triangle surface. A small
// vertex count is not that certificate: a real sheet/line may also have six.
const fogPath=resolve(engine,'src/extras/fog/index.ts');
let fog=await readFile(fogPath,'utf8');
if(!fog.includes('geometryAdapter: "screen-effect"')){
 const anchor='        world.resource(RenderPhases).push({';
 if(fog.split(anchor).length!==2)throw Error('MAIN fog phase certificate changed');
 fog=fog.replace(anchor,anchor+'\n            geometryAdapter: "screen-effect",');
 await writeFile(fogPath,fog);
}
if(!forward.includes('geometryAdapter: "standard-material"')){
  const anchor='        world.resource(RenderPhases).push({';
  if(forward.split(anchor).length!==2)throw Error('MAIN phase geometry: standard registration changed');
  forward=forward.replace(anchor,anchor+'\n            geometryAdapter: "standard-material",');
}
const pipelineMarker='/* TRUEFF_MAIN_PIPELINE_SOURCE_0103 */';
if(!gpu.includes(pipelineMarker)){
  const start=gpu.indexOf('              const trackedRenderPipeline = ('),end=gpu.indexOf('              const queueMethods:',start);
  if(start<0||end<start)throw Error('MAIN pipeline source: native pipeline tracking changed');
  const original=gpu.slice(start,end);
  if(!original.includes('primeRenderBundle(')||!original.includes('createRenderPipelineAsync(descriptor)'))throw Error('MAIN pipeline source: pipeline warm ordering changed');
  const tracked=`              const trackedRenderPipeline = (descriptor: GPURenderPipelineDescriptor) => {
                  checkPipeline(descriptor);
                  warnLatePipeline(descriptor);
                  if (owner.resource?.(precompileState).drained)
                      return retainRenderPipelineSource(createRenderPipeline(descriptor), descriptor, shaderSources);
                  descriptor = materializeRenderTargets(descriptor);
                  return retainRenderPipelineSource(primeRenderBundle(
                      bundleDescriptor(descriptor), createRenderPipeline(descriptor),
                  ), descriptor, shaderSources);
              };
              const trackedRenderPipelineAsync = (descriptor: GPURenderPipelineDescriptor) => {
                  checkPipeline(descriptor);
                  if (owner.resource?.(precompileState).drained)
                      return createRenderPipelineAsync(descriptor).then(pipeline => retainRenderPipelineSource(pipeline, descriptor, shaderSources));
                  descriptor = materializeRenderTargets(descriptor);
                  const bundle = bundleDescriptor(descriptor);
                  return createRenderPipelineAsync(descriptor).then(pipeline =>
                      retainRenderPipelineSource(primeRenderBundle(bundle, pipeline), descriptor, shaderSources),
                  );
              };
`;
  gpu=gpu.slice(0,start)+tracked+gpu.slice(end);
  gpu=`${pipelineMarker}\nimport { retainRenderPipelineSource } from './pipeline-source';\n`+gpu;
}
// Upgrade both a fresh install and an earlier tracked-pipeline seam. The raw
// device compiles the color-free depth variant once, not inside a draw frame.
const depthOnlyMarker='/* TRUEFF_MAIN_DEPTH_ONLY_0103 */';
if(!gpu.includes(depthOnlyMarker)){
  const anchor='descriptor, shaderSources)';
  if(gpu.split(anchor).length!==5)throw Error('MAIN native depth-only pipeline registration changed');
  gpu=gpu.replaceAll(anchor,'descriptor, shaderSources, d)')+'\n'+depthOnlyMarker+'\n';
}
if(!coreBarrel.includes(phaseMarker))coreBarrel+=`\n${phaseMarker}\nexport { PhaseGeometryCapture } from './phase-geometry';\nexport type { PhaseGeometryDraw } from './phase-geometry';\n`;
if(!runtimeBarrel.includes(pipelineMarker))runtimeBarrel+=`\n${pipelineMarker}\nexport { renderPipelineSource } from './pipeline-source';\nexport type { RenderPipelineSource } from './pipeline-source';\n`;
const groupMarker='/* TRUEFF_MAIN_BIND_GROUP_SOURCE_0103 */';
if(!gpu.includes(groupMarker)){
  const original=`              const trackedBindGroupLayout = (descriptor: GPUBindGroupLayoutDescriptor) => {
                  const layout = createBindGroupLayout(descriptor);
                  bindGroupLayouts.set(layout, countPipelineStorageBindings([descriptor.entries]));
                  return layout;
              };`;
  if(gpu.split(original).length!==2)throw Error('MAIN binding source: native layout tracking changed');
  gpu=gpu.replace(original,`              ${groupMarker}
              const trackedBindGroupLayout = (descriptor: GPUBindGroupLayoutDescriptor) => {
                  const captured = {...descriptor, entries: Array.from(descriptor.entries)};
                  const layout = createBindGroupLayout(captured);
                  bindGroupLayouts.set(layout, countPipelineStorageBindings([captured.entries]));
                  return retainBindGroupLayoutSource(layout, captured);
              };
              const trackedBindGroup = (descriptor: GPUBindGroupDescriptor) => {
                  const captured = {...descriptor, entries: Array.from(descriptor.entries)};
                  return retainBindGroupSource(d.createBindGroup(captured), captured);
              };`);
  gpu=gpu.replace("import { retainRenderPipelineSource } from './pipeline-source';","import { retainRenderPipelineSource, retainBindGroupSource, retainBindGroupLayoutSource } from './pipeline-source';");
  const dispatch='                      if (key === "createBindGroupLayout") return trackedBindGroupLayout;';
  if(gpu.split(dispatch).length!==2)throw Error('MAIN binding source: device dispatch changed');
  gpu=gpu.replace(dispatch,dispatch+'\n                      if (key === "createBindGroup") return trackedBindGroup;');
}
if(!runtimeBarrel.includes(groupMarker))runtimeBarrel+=`\n${groupMarker}\nexport { bindGroupSource } from './pipeline-source';\n`;
const periodicMarker='/* TRUEFF_PERIODIC_SOURCE_CLOSURE_0103 */';
if(!primitives.includes(periodicMarker)){
  const phi='const phi = u * Math.PI * 2;';
  if(primitives.split(phi).length!==6)throw Error('Periodic source closure: primitive angular sampling changed');
  primitives=primitives.replaceAll(phi,'const [cosPhi, sinPhi] = periodicCircleSample(u);')
    .replace('const theta = v * Math.PI;','const [cosTheta, sinTheta] = periodicCircleSample(v * 0.5);')
    .replaceAll('const theta = (y / halfRings) * (Math.PI / 2);','const [cosTheta, sinTheta] = periodicCircleSample((y / halfRings) * 0.25);')
    .replaceAll('Math.sin(theta) * Math.cos(phi)','sinTheta * cosPhi')
    .replaceAll('Math.sin(theta) * Math.sin(phi)','sinTheta * sinPhi')
    .replaceAll('Math.cos(theta)','cosTheta').replaceAll('Math.cos(phi)','cosPhi').replaceAll('Math.sin(phi)','sinPhi');
  primitives+=`\n${periodicMarker}
/** Exact cardinal/periodic endpoints preserve one geometric seam while UVs
 * still run from0to1. Trigonometric roundoff at2π must not quantize two copies
 * of the same endpoint onto opposite sides of the unorm16 midpoint. */
function periodicCircleSample(turn: number): [number, number] {
    if (turn === 0 || turn === 1) return [1, 0];
    if (turn === 0.25) return [0, 1];
    if (turn === 0.5) return [-1, 0];
    if (turn === 0.75) return [0, -1];
    return [Math.cos(turn * Math.PI * 2), Math.sin(turn * Math.PI * 2)];
}\n`;
}
// Validate every anchor before writing any file.
const {transformInstanceCapacity}=await import('./instance-capacity.mjs');
({preprocess,forward}=transformInstanceCapacity({preprocess,forward}));
const {transformNativeInstanceFrontier}=await import('./native-instance-frontier.mjs');
forward=transformNativeInstanceFrontier(forward);
// Native spatial work selection observes the same producer rows and cull
// volumes as the GPU pack. It runs before MAIN registers/extracts sources.
const spatialMarker='/* TRUEFF_MAIN_SOURCE_WORKSET_0103 */';
if(!preprocess.includes(spatialMarker)){
  const anchor='export function initializeMeshPreprocess(world: World): void {';
  if(preprocess.split(anchor).length!==2)throw Error('MAIN source workset initialization owner changed');
  preprocess=preprocess.replace(anchor,anchor+'\n    initializeSourceWorkset(world);');
  preprocess='import { initializeSourceWorkset } from "./source-workset";\n'+preprocess+'\n'+spatialMarker+'\n';
}
if(!forward.includes(spatialMarker)){
  const anchor='    for (let i = 0; i < state.frameCount; i++) {\n        const item = state.frameDraws[i]!;\n        const stage = materialGeometryStage(world, item.r.t);';
  if(forward.split(anchor).length!==2)throw Error('MAIN source workset geometry visitor changed');
  forward=forward.replace(anchor,'    const visibleMeshes = visibleSourceMeshes(world, slot);\n'+anchor.replace('        const stage',
    '        if (visibleMeshes && isMeshInstanceDraw(world, item.draw) && !visibleMeshes.has(item.draw.mesh)) continue;\n        const stage'));
  if(!forward.includes('import { visibleSourceMeshes } from "./source-workset";'))forward='import { visibleSourceMeshes } from "./source-workset";\n'+forward;
  forward+='\n'+spatialMarker+'\n';
}
// The broad phase must bound instance extraction too, not just filter mesh
// registration. A repeated mesh with few visible instances otherwise still
// reserves/plays slices for every instance in the world. Certified producer
// capacities keep their own owner; an unknown native view keeps its old bound.
const visibleInstanceCapacity='(visibleMeshes?.get(item.draw.mesh) ?? meshInstanceDrawCapacity(world, item.draw))';
if(!forward.includes(visibleInstanceCapacity)){
  const anchor='?? meshInstanceDrawCapacity(world, item.draw),';
  if(forward.split(anchor).length!==2)throw Error('MAIN spatial instance extent owner changed');
  forward=forward.replace(anchor,'?? '+visibleInstanceCapacity+',');
}
const spatialSource=await readFile(resolve(project,'packages/ascii-shader/spatial-index.ts'),'utf8');
const worksetSource=await readFile(resolve(project,'engine-seam/engine/source-workset.ts'),'utf8');
const posedBoundSource=await readFile(resolve(project,'engine-seam/engine/posed-bound-producer.ts'),'utf8');
const {transformGltfPosedBoundProducer}=await import('./posed-bound-material.mjs');
skinning=transformGltfPosedBoundProducer(skinning);
if(!barrel.includes('export { sourceViewWorkset } from "./source-workset";'))barrel+='\nexport { sourceViewWorkset } from "./source-workset";\n';
const meshChangeMarker='/* TRUEFF_MAIN_MESH_CHANGE_STREAM_0103 */';
if(!mesh.includes(meshChangeMarker)){
  const declaration='interface MeshRegistry extends Iterable<Mesh> {';
  if(mesh.split(declaration).length!==2)throw Error('MAIN mesh change stream interface changed');
  mesh=mesh.replace(declaration,declaration+'\n    /** Native source replacement/membership notification; read after mesh publication. */\n    observe(observer: (handle: MeshHandle) => void): () => void;');
  const classAt=mesh.indexOf('class MeshRegistryImpl implements MeshRegistry {'),classEnd=mesh.indexOf('\nexport const Meshes:',classAt);
  if(classAt<0||classEnd<0)throw Error('MAIN mesh change stream owner changed');
  let registry=mesh.slice(classAt,classEnd);
  const add='        this.version++;\n        return handle;',replace='        this._meshesByIndex[this.index(handle)!] = mesh;\n        this.version++;',remove='        this.version++;\n        return mesh;',clear='    clear(): void {\n        this._meshesByIndex.fill(undefined);';
  for(const anchor of [add,replace,remove,clear])if(registry.split(anchor).length!==2)throw Error('MAIN mesh change publication changed');
  registry=registry.replace('    private readonly _world: World;',`    ${meshChangeMarker}
    private readonly _observers = new Set<(handle: MeshHandle) => void>();
    observe(observer: (handle: MeshHandle) => void): () => void {
        this._observers.add(observer);return () => { this._observers.delete(observer); };
    }
    private notifyMesh(handle: MeshHandle): void { for (const observer of this._observers) observer(handle); }
    private readonly _world: World;`)
    .replace(add,'        this.version++;\n        this.notifyMesh(handle);\n        return handle;')
    .replace(replace,replace+'\n        this.notifyMesh(handle);')
    .replace(remove,'        this.version++;\n        this.notifyMesh(handle);\n        return mesh;')
    .replace(clear,'    clear(): void {\n        for (const handle of this._order) this.notifyMesh(handle);\n        this._meshesByIndex.fill(undefined);');
  mesh=mesh.slice(0,classAt)+registry+mesh.slice(classEnd);
}
await writeFile(pipelinePath,pipelines);
await writeFile(forwardPath,forward);
await writeFile(preprocessPath,preprocess);
await writeFile(barrelPath,barrel);
await writeFile(instancePath,instance);
await writeFile(framePath,frame);
await writeFile(substratePath,substrate);
await writeFile(viewPath,view);
await writeFile(coreBarrelPath,coreBarrel);
await writeFile(meshPath,mesh);
await writeFile(primitivePath,primitives);
await writeFile(phasesPath,phases);
await writeFile(gpuPath,gpu);
await writeFile(runtimeBarrelPath,runtimeBarrel);
await writeFile(physicsRuntimePath,physicsRuntime);
await writeFile(physicsBarrelPath,physicsBarrel);
await writeFile(resolve(engine,'src/core/rendering/phase-geometry.ts'),phaseSource);
await writeFile(resolve(engine,'src/core/rendering/presentation-ticket.ts'),presentationTicketSource);
await writeFile(resolve(engine,'src/core/rendering/presentation-targets.ts'),presentationTargetsSource);
await writeFile(resolve(engine,'src/engine/runtime/pipeline-source.ts'),pipelineSource);
await writeFile(resolve(engine,'src/standard/rendering/main-spatial-index.ts'),spatialSource);
await writeFile(resolve(engine,'src/standard/rendering/posed-bound-producer.ts'),posedBoundSource);
await writeFile(resolve(engine,'src/standard/rendering/source-workset.ts'),worksetSource);
const {applySourceMaterialMetadata}=await import('./material-source.mjs');
await applySourceMaterialMetadata(engine);

const {applyStaticSourceSemantic}=await import('./static-source-semantic.mjs');
await applyStaticSourceSemantic(engine);
const {applyAuthoredConstruction}=await import('./authored-construction.mjs');
await applyAuthoredConstruction(engine);
const {applyGpuIndexedSourceProducer}=await import('./gpu-indexed-source-producer.mjs');
await applyGpuIndexedSourceProducer(engine);
const {applyMaterialRecipes}=await import('./material-recipes.mjs');
await applyMaterialRecipes(engine);
const {applyMobileOrbitPatch}=await import('./mobile-orbit.mjs');
await applyMobileOrbitPatch(engine);
const {applyTypeContracts}=await import('./type-contracts.mjs');
await applyTypeContracts(engine);
const {applyNativeInstanceFrontier}=await import('./native-instance-frontier.mjs');
await applyNativeInstanceFrontier(engine);
await writeFile(loadingPath,loading);
await writeFile(appPath,app);
await writeFile(skinningPath,skinning);
console.log(`Applied MAIN live material geometry seam to Shallot ${version}`);
