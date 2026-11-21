/**
 * Publish the standard renderer's real GPU membership frontier at the
 * MaterialGeometryTransport boundary.  This is deliberately a producer
 * certificate: the packed list and indirect record are both owned by the
 * same native preprocess submission.  No CPU row census or first-vertex
 * ownership inference is allowed here.
 */
export const NATIVE_INSTANCE_FRONTIER_MARKER='/* TRUEFF_MAIN_NATIVE_INSTANCE_FRONTIER_0103 */';

function once(text,from,to,label){
 const count=text.split(from).length-1;
 if(count!==1)throw Error(`MAIN native instance frontier anchor ${label} expected once, found ${count}`);
 return text.replace(from,to);
}

export function transformNativeInstanceFrontier(forward){
 if(forward.includes(NATIVE_INSTANCE_FRONTIER_MARKER))return forward;
 const declarations=forward.split('\n').filter(line=>line.includes('    sourceGeometry?:')&&line.includes('Float32Array'));
 if(declarations.length!==1)throw Error(`MAIN native instance frontier visitor declaration expected once, found ${declarations.length}`);
 const declaration=declarations[0];
 const contract=`${NATIVE_INSTANCE_FRONTIER_MARKER}
${declaration}
    /** Native owner certificate; absent means retain the complete incumbent. */
    ownerAuthority?: { readonly kind: "standard-entity"; readonly mesh: number; readonly identity: "eid"; readonly posedBounds?: "engine-deformed-cull-sphere-v1" | "material-source-cull-sphere-v1" } | { readonly kind: "producer-row"; readonly namespace: string; readonly records: GPUBufferBinding; readonly stride: 24; readonly identity: "row" };
    /** Same-submit packed membership plus the exact indirect record that owns its live count/base. */
    nativeVisibleInstanceFrontier?: { readonly kind: "native-indirect"; readonly members: GPUBuffer; readonly membersStride: 16; readonly indirect: GPUBuffer; readonly indirectOffset: number; readonly owner: "entity" | "producer-row" };`;
 forward=once(forward,declaration,contract,'visitor contract');
 const args='        const args = item.draw.args;';
 forward=once(forward,args,`${args}
        const producer = world.resource(producerTables).draws.get(item.draw.name) ?? null;
        const packedEids = world.resource(MeshDrawBuffers).packedEids;
        const meshSource = world.resource(Meshes).get(item.draw.mesh);
        const materialState = world.resource(MaterialTypes);
        const materialTypeId = materialState?.ids?.get(item.r.t.owner);
        const materialSource = materialTypeId === undefined ? undefined : materialState?.sources?.[materialTypeId];
        // A custom vertex stage is only admitted when its producer publishes a
        // complete conservative posed sphere through the engine contract:
        // MaterialSource.cullBounds, or the glTF deformation producer's
        // explicit certificate.  The old undefined-vertex test remains the
        // affine default; arbitrary custom/GPU-indexed stages stay incumbent.
        const posedBounds = item.r.t.owner?.mainPosedBounds === "engine-deformed-cull-sphere-v1"
            ? "engine-deformed-cull-sphere-v1"
            : typeof materialSource?.cullBounds === "function"
              ? "material-source-cull-sphere-v1"
              : undefined;
        const entityOwner = isMeshInstanceDraw(world, item.draw) && !meshSource?.gpuIndexedSource &&
            (item.r.t.owner?.vertex === undefined || posedBounds !== undefined);
        // Only the standard entity table, a certified deformation/source
        // producer, and an explicitly authored producer assembly certify
        // ownership.  Uncertified custom/GPU-indexed draws remain on the
        // complete incumbent path rather than borrowing a row or vertex.
        const ownerAuthority = producer?.sourceAssembly ? {
            kind: "producer-row" as const,
            namespace: producer.sourceAssembly.namespace,
            records: producer.sourceAssembly.records,
            stride: producer.sourceAssembly.stride,
            identity: "row" as const,
        } : entityOwner ? {
            kind: "standard-entity" as const,
            mesh: item.draw.mesh,
            identity: "eid" as const,
            ...(posedBounds ? { posedBounds } : {}),
        } : undefined;
        const indirectOffset = (args.offset ?? 0) + slot * (args.viewStride ?? 0);
        const visibleInstanceFrontier = ownerAuthority && packedEids ? {
            kind: "native-indirect" as const,
            members: world.gpu.root.unwrap(packedEids),
            membersStride: 16 as const,
            indirect: world.gpu.root.unwrap(args.indirect),
            indirectOffset,
            owner: ownerAuthority.kind === "producer-row" ? "producer-row" as const : "entity" as const,
        } : undefined;`,'same-submit producer inputs');
 const source='            sourceGeometry: world.resource(Meshes).get(item.draw.mesh)!.sourceGeometry,';
 forward=once(forward,source,`${source}
            ownerAuthority,
            nativeVisibleInstanceFrontier: visibleInstanceFrontier,`,'producer payload');
 return forward;
}
