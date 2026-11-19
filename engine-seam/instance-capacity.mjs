/** The existing native mesh-pair bins own this bound. No world-sized
 * fallback, new membership cache, visibility guess, or instance quota. */
export const INSTANCE_CAPACITY_SOURCE=`
/* TRUEFF_MAIN_MESH_INSTANCE_CAPACITY_0103 */
export function meshInstanceDrawCapacity(world: World, draw: Draw): number | undefined {
    if (!isMeshInstanceDraw(world, draw)) return undefined;
    const index = world.resource(meshResourcesKey).meshes.index(draw.mesh);
    if (index === undefined) return undefined;
    let count = 0;
    // All material/alpha/cull bins for this mesh form a safe bound even
    // when a source updates its phase on the GPU after CPU preprocessing.
    for (const bin of world.resource(meshPairsKey).bins.values()) count += bin[index] ?? 0;
    // Reserved native recipes can have no rows; their real indirect
    // instanceCount remains zero. One guarded slice preserves that case.
    return Math.max(1, count);
}
`;
export function transformInstanceCapacity({preprocess,forward}){
 const old='(isMeshInstanceDraw(world, item.draw) ? Math.max(1, meshInstanceTable(world).count) : undefined)';
 if(!preprocess.includes('/* TRUEFF_MAIN_MESH_INSTANCE_CAPACITY_0103 */'))preprocess+=INSTANCE_CAPACITY_SOURCE;
 if(forward.includes(old)){
  if(forward.split(old).length!==2)throw Error('MAIN native instance bound anchor is ambiguous');
  forward=forward.replace(old,'meshInstanceDrawCapacity(world, item.draw)');
 }
 if(!forward.includes('meshInstanceDrawCapacity(world, item.draw)'))throw Error('MAIN native instance bound owner is missing');
 if(!forward.includes('    meshInstanceDrawCapacity,')){
  const anchor='    isMeshInstanceDraw,\n';
  if(forward.split(anchor).length!==2)throw Error('MAIN native preprocessing import is ambiguous');
  forward=forward.replace(anchor,anchor+'    meshInstanceDrawCapacity,\n');
 }
 return {preprocess,forward};
}
