/** Type-only compatibility contracts for the pinned Shallot 0.10.5 seam.
 * These transforms preserve runtime behavior and keep the installed engine
 * source type-correct without editing node_modules by hand. */
import { readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

export const TYPE_CONTRACT_MARKER = "/* TRUEFF_MAIN_TYPE_CONTRACTS_0105 */";
export const MATERIAL_TYPE_MARKER = "/* TRUEFF_MAIN_MATERIAL_TYPE_CONTRACT_0105 */";

function once(text, from, to, label) {
    const count = text.split(from).length - 1;
    if (count !== 1) throw Error(`MAIN type contract anchor ${label} expected once, found ${count}`);
    return text.replace(from, to);
}

/** Pure transform used by the reproducible engine application and checks. */
export function transformMaterialTypeContract(pipelines) {
    const alias = "type AnyMaterialType = MaterialType<any, any, Record<string, AnyWgslData>>;";
    const oldTypedAlias = `${TYPE_CONTRACT_MARKER}
type AnyMaterialType = MaterialType<any, any, Record<string, AnyWgslData>> & {
    /** MAIN's glTF deformation producer certificate; runtime value is optional. */
    readonly mainPosedBounds?: "engine-deformed-cull-sphere-v1";
};`;
    // The first version of this seam narrowed the generic alias and caused
    // unrelated layout variance errors. Normalize that form before applying
    // the contract to the shared MaterialType interface below.
    if (pipelines.includes(oldTypedAlias)) pipelines = pipelines.replace(oldTypedAlias, alias);
    return pipelines;
}

export function transformMaterialTypeInterface(materialType) {
    if (materialType.includes(MATERIAL_TYPE_MARKER)) return materialType;
    const anchor = "> extends Resource<MaterialAssets<P>> {\n    readonly name: string;\n";
    return once(
        materialType,
        anchor,
        `${anchor}    ${MATERIAL_TYPE_MARKER}\n    /** MAIN's optional deformation-bound producer certificate. */\n    readonly mainPosedBounds?: "engine-deformed-cull-sphere-v1";\n`,
        "material type owner certificate",
    );
}

export async function applyTypeContracts(engine) {
    const pipelinesPath = resolve(engine, "src/standard/rendering/pipelines.ts");
    const materialTypePath = resolve(engine, "src/standard/rendering/material-type.ts");
    const pipelines = await readFile(pipelinesPath, "utf8");
    const materialType = await readFile(materialTypePath, "utf8");
    const transformedPipelines = transformMaterialTypeContract(pipelines);
    const transformedMaterialType = transformMaterialTypeInterface(materialType);
    if (transformedPipelines !== pipelines) await writeFile(pipelinesPath, transformedPipelines);
    if (transformedMaterialType !== materialType) await writeFile(materialTypePath, transformedMaterialType);
}
