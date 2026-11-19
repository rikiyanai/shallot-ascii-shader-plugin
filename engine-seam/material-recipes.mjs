import {readFile,writeFile} from 'node:fs/promises';
import {resolve,dirname} from 'node:path';
import {createRequire} from 'node:module';
// Version-pinned material metadata/native-execution seam.
export const RECIPE_PASS='shallot.phase.render-recipe';
export const PREPARE_RECIPE='shallot.material.prepare-recipe';
const CORE_MARK='/* SHALLOT_CAPTURE_RECIPE_0128 */';
function once(source,before,after){
 if(source.split(before).length!==2)throw Error('TypeGPU 0.12.8 recipe anchor changed: '+before.slice(0,90));
 return source.replace(before,after);
}
export function captureCoreSource(source){
 if(source.includes(CORE_MARK))return source;
 source=once(source,'    #memo;\n','    #memo;\n    #resolvedProgram;\n    #nativeProgram;\n    #captureMemo;\n');
 source=once(source,'    resolveAndCreateShaderModule() {\n',`    resolveAndCreateShaderModule() {
        if (this.#nativeProgram !== undefined) return this.#nativeProgram;
        const r = this.resolveRenderRecipeProgram();
        const module = this.options.root.device.createShaderModule({
            label: String(r.descriptor.label ?? '<unnamed>') + ' - Shader',
            code: r.resolutionResult.code,
        });
        const descriptor = Object.freeze({ ...r.descriptor,
            vertex: Object.freeze({ ...r.descriptor.vertex, module }),
            fragment: r.descriptor.fragment ? Object.freeze({ ...r.descriptor.fragment, module }) : undefined,
        });
        return this.#nativeProgram = { ...r, descriptor };
    }
    resolveRenderRecipeProgram() {
        if (this.#resolvedProgram !== undefined) return this.#resolvedProgram;
`);
 source=once(source,`        const module = device.createShaderModule({
            label: \`\${getName(this) ?? '<unnamed>'} - Shader\`,
            code,
        });`, `        // Metadata is not an executable GPU shader. The native module is
        // created only by resolveAndCreateShaderModule() for a native consumer.
        const module = undefined;`);
 source=once(source,'        return { resolutionResult, descriptor, connectedTargets, connectedAttribs };',
  '        return this.#resolvedProgram = { resolutionResult, descriptor: snapshotRecipeDescriptor(descriptor), connectedTargets, connectedAttribs };');
 source=once(source,'    initAsync() {\n        if (this.#memo !== undefined) {',`    captureMemo() {
        // Restored/precompiled wrappers already own a real pipeline. Keep their
        // existing metadata path rather than inventing a new shader source.
        if (!this.options.descriptor) return this.unwrap();
        if (this.#captureMemo !== undefined) return this.#captureMemo;
        const resolved = this.resolveRenderRecipeProgram();
        const r = resolved.resolutionResult;
        const recipe = Object.freeze({
            descriptor: resolved.descriptor,
            vertexCode: r.code,
            fragmentCode: resolved.descriptor.fragment ? r.code : undefined,
            materialize: () => this.unwrap().pipeline,
        });
        return this.#captureMemo = {
            recipe,
            usedBindGroupLayouts: r.usedBindGroupLayouts,
            catchall: r.catchall,
            logResources: r.logResources,
            usedVertexLayouts: resolved.connectedAttribs.usedVertexLayouts,
            connectedTargets: resolved.connectedTargets,
            usedImmediate: r.usedImmediate,
        };
    }
    initAsync() {
        if (this.#memo !== undefined) {
`);
 source=once(source,'    #withPriors(patch) {\n',`    [Symbol.for('${PREPARE_RECIPE}')]() {
        return this[$internal].core.captureMemo().recipe;
    }
    #withPriors(patch) {
`);
 // Only descriptor dictionaries are copied; GPU objects/layout identity stay

 source+=`\n${CORE_MARK}
function snapshotRecipeDescriptor(d) {
 const freeze = Object.freeze;
 return freeze({ ...d,
  vertex: freeze({ ...d.vertex,
   constants: d.vertex.constants ? freeze({ ...d.vertex.constants }) : undefined,
   buffers: d.vertex.buffers ? freeze(Array.from(d.vertex.buffers, b => b ? freeze({ ...b, attributes: freeze(Array.from(b.attributes, a => freeze({ ...a }))) }) : null)) : undefined,
  }),
  fragment: d.fragment ? freeze({ ...d.fragment,
   constants: d.fragment.constants ? freeze({ ...d.fragment.constants }) : undefined,
   targets: freeze(Array.from(d.fragment.targets, t => t ? freeze({ ...t, blend: t.blend ? freeze({ color: freeze({ ...t.blend.color }), alpha: freeze({ ...t.blend.alpha }) }) : undefined }) : null)),
  }) : undefined,
  primitive: d.primitive ? freeze({ ...d.primitive }) : undefined,
  multisample: d.multisample ? freeze({ ...d.multisample }) : undefined,
  depthStencil: d.depthStencil ? freeze({ ...d.depthStencil,
   stencilFront: d.depthStencil.stencilFront ? freeze({ ...d.depthStencil.stencilFront }) : undefined,
   stencilBack: d.depthStencil.stencilBack ? freeze({ ...d.depthStencil.stencilBack }) : undefined,
  }) : undefined,
 });
}
`;
 return source;
}
export function captureDrawStateSource(source){
 if(source.includes(CORE_MARK))return source;
 source=once(source,`function applyRenderPipelineState(encoder, root, pipeline, passState) {
    const { core, priors } = pipeline[$internal];
    const memo = core.unwrap();
    encoder.setPipeline(memo.pipeline);`, `function applyRenderPipelineState(encoder, root, pipeline, passState, memo) {
    const { priors } = pipeline[$internal];
    if (memo.recipe) {
        const record = encoder[Symbol.for('${RECIPE_PASS}')];
        if (typeof record !== 'function') throw new Error('A render recipe cannot execute in a native pass');
        record.call(encoder, memo.recipe);
    } else {
        encoder.setPipeline(memo.pipeline);
    }`);
 source=once(source,`    const memo = core.unwrap();
    if (!ownsPass) {
        reportIgnoredPriors(core, passInternals.owner, !!priors.timestampWrites, memo.logResources, 'render',`, `    const capture = rawPass[Symbol.for('${RECIPE_PASS}')];
    const memo = typeof capture === 'function' ? core.captureMemo() : core.unwrap();
    if (!ownsPass) {
        reportIgnoredPriors(core, passInternals.owner, !!priors.timestampWrites, memo.logResources, 'render',`);
 source=once(source,'        applyRenderPipelineState(rawPass, root, pipeline, state);','        applyRenderPipelineState(rawPass, root, pipeline, state, memo);');
 return source+'\n'+CORE_MARK+'\n';
}

const ENGINE_MARK='/* SHALLOT_MATERIAL_RECIPES_0116 */';
function engineOnce(s,a,b){if(s.split(a).length!==2)throw Error('Shallot material recipe anchor changed: '+a.slice(0,90));return s.replace(a,b);}
export function bundleSource(s){
 if(s.includes(ENGINE_MARK))return s;
 s=engineOnce(s,'const SET_PIPELINE = 0;',`const SET_RECIPE = 5;
const SET_PIPELINE = 0;`);
 s=engineOnce(s,'    return {\n        setPipeline: (pipeline:',`    return {
        [Symbol.for('${RECIPE_PASS}')]: (recipe:RenderPipelineRecipe) => emit(pass,SET_RECIPE,recipe),
        setPipeline: (pipeline:`);
 s=engineOnce(s,'            case SET_PIPELINE:\n',`            case SET_RECIPE: {
                const recipe=a as RenderPipelineRecipe;
                const capture=(encoder as unknown as Record<PropertyKey,unknown>)[Symbol.for('${RECIPE_PASS}')];
                if(typeof capture==='function')capture.call(encoder,recipe);
                else setRenderPipeline(encoder,recipe.materialize());
                break;
            }
            case SET_PIPELINE:
`);
 return ENGINE_MARK+"\nimport type {RenderPipelineRecipe} from '../../engine/runtime/pipeline-source';\n"+s;
}
export function materialWarmSource(s){
 if(s.includes(ENGINE_MARK))return s;
 // Scope every edit to the existing warm owner, not other pipeline factories.
 const a=s.indexOf('export async function prepareMaterialTypes('),b=s.indexOf('\n/* TRUEFF_MAIN_GEOMETRY_',a);
 if(a<0||b<0)throw Error('Missing current material warm owner');
 let warm=s.slice(a,b);
 warm=engineOnce(warm,'    const pipelines = new Map<TgpuRenderPipeline<any>, string>();',`    const pipelines = new Map<TgpuRenderPipeline<any>, string>();
    const beauty=new Set<TgpuRenderPipeline<any>>(),nativeRequired=new Set<TgpuRenderPipeline<any>>();
    const primary=world.resource(PhaseGeometryCapture).isPrimary?.()===true;`);
 warm=engineOnce(warm,'                    if (pipeline) pipelines.set(pipeline, type.name);','                    if (pipeline) {pipelines.set(pipeline, type.name);nativeRequired.add(pipeline);}');
 warm=engineOnce(warm,'                for (const p of [\n                    compiled.color,',`                for(const p of [compiled.color,compiled.transparent,compiled.single?.color,compiled.single?.transparent,compiled.stacked])if(p)beauty.add(p);
                for(const p of [compiled.point,compiled.cascade,compiled.prepass,compiled.prepassSingle])if(p)nativeRequired.add(p);
                for (const p of [
                    compiled.color,`);
 warm=engineOnce(warm,'        pipelines.set(cb.single, bg.name);','        pipelines.set(cb.single, bg.name);\n        beauty.add(cb.color);beauty.add(cb.single);');
 warm=engineOnce(warm,'                        pending = pipeline.initAsync();',`                        pending = primary&&beauty.has(pipeline)&&!nativeRequired.has(pipeline)
                            ? Promise.resolve(prepareRenderPipelineRecipe(pipeline))
                            : pipeline.initAsync();`);
 return ENGINE_MARK+"\nimport {PhaseGeometryCapture} from '../../core/rendering/phase-geometry';\nimport {prepareRenderPipelineRecipe} from '../../engine/runtime/pipeline-source';\n"+s.slice(0,a)+warm+s.slice(b);
}

export async function applyMaterialRecipes(engine){
 const require=createRequire(resolve(engine,'package.json'));
 const typegpu=dirname(require.resolve('typegpu/package.json'));
 const version=JSON.parse(await readFile(resolve(typegpu,'package.json'),'utf8')).version;
 if(version!=='0.12.8')throw Error('MAIN material recipes require TypeGPU0.12.8, received '+version);
 for(const [path,transform] of [
  [resolve(typegpu,'core/pipeline/renderPipeline.js'),captureCoreSource],
  [resolve(typegpu,'core/pipeline/drawState.js'),captureDrawStateSource],
  [resolve(engine,'src/standard/rendering/bundle.ts'),bundleSource],
  [resolve(engine,'src/standard/rendering/pipelines.ts'),materialWarmSource],
 ]){
  const before=await readFile(path,'utf8'),after=transform(before);
  if(after!==before)await writeFile(path,after);
 }
 const path=resolve(engine,'src/engine/runtime/index.ts'),marker='/* SHALLOT_RENDER_RECIPE_EXPORT_0116 */';
 let barrel=await readFile(path,'utf8');
 if(!barrel.includes(marker)){
  barrel+='\n'+marker+'\nexport {prepareRenderPipelineRecipe} from \'./pipeline-source\';\nexport type {RenderPipelineRecipe,RenderPipelineSourceDescriptor} from \'./pipeline-source\';\n';
  await writeFile(path,barrel);
 }
}

