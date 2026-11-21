export const POSED_BOUND_MATERIAL_MARKER='/* TRUEFF_MAIN_POSED_BOUND_PRODUCER_0103 */';

/** Add the explicit glTF deformation certificate without editing the pinned
 * package by hand.  The kernel's cull sphere remains the per-entity value;
 * this marker only certifies which material producer owns that value. */
export function transformGltfPosedBoundProducer(skinning){
 if(skinning.includes(POSED_BOUND_MATERIAL_MARKER))return skinning;
 const anchor='});\nconst UploadDeformationsSystem: System = {';
 const count=skinning.split(anchor).length-1;
 if(count!==1)throw Error(`MAIN posed bound producer: glTF skinning material owner changed (${count})`);
 return skinning.replace(anchor,`});\n${POSED_BOUND_MATERIAL_MARKER}\n// The glTF kernel writes the deformed local sphere to MeshInstance.cullBounds\n// before MeshPreprocessSystem. MAIN may therefore use that sphere for the\n// custom vertex stage, but only as an explicit producer certificate.\n(SkinnedMaterial as typeof SkinnedMaterial & {mainPosedBounds: "engine-deformed-cull-sphere-v1"}).mainPosedBounds="engine-deformed-cull-sphere-v1";\nconst UploadDeformationsSystem: System = {`);
}
