/* TRUEFF_MAIN_PHYSICS_CONTACT_OBSERVATION_0105 */

/** The completed-solver identity used to make a host observation idempotent. */
export interface PhysicsContactObservationStamp {
    readonly worldId: number;
    readonly generation: number;
    readonly stepIndex: number;
}

/**
 * Visits one touching solver contact. `contact` and its nested manifolds are borrowed from the
 * World's ContactDataBuffer: copy every value needed after this callback returns. `bodyA` and
 * `bodyB` are generation-aware ECS references in the same solver order as `contact.shapeA/B`.
 */
export type PhysicsContactVisitor = (
    contact: PhysicsContactData,
    bodyA: EntityRef,
    bodyB: EntityRef,
) => void;

const activePhysicsContactObservers = new WeakSet<World>();

/**
 * Visits the actual touching contacts in the completed solver directory once per stamp.
 * The directory is traversed in O(contact capacity), not by testing rendered shapes or contact
 * event records. Call only between physics steps; callback re-entry and locked solver access are
 * rejected. An uninitialized or currently absent physics state returns undefined and does not
 * create a physics world beyond the existing runtimeFor resource lookup semantics.
 */
export function observePhysicsContacts(
    world: World,
    previous: PhysicsContactObservationStamp | undefined,
    visit: PhysicsContactVisitor,
): PhysicsContactObservationStamp | undefined {
    const runtime = world.resource(physicsRuntimeKey);
    const state = runtime.physicsWorld;
    if (!runtime.initialized || !state || !stateIsValid(state)) return undefined;
    if (state.locked) throw new Error("physics: observePhysicsContacts refuses a locked solver");
    if (activePhysicsContactObservers.has(world))
        throw new Error("physics: observePhysicsContacts refuses callback reentrancy");

    const stamp: PhysicsContactObservationStamp = {
        worldId: state.worldId,
        generation: state.generation,
        stepIndex: state.stepIndex,
    };
    if (
        previous &&
        previous.worldId === stamp.worldId &&
        previous.generation === stamp.generation &&
        previous.stepIndex === stamp.stepIndex
    )
        return stamp;

    activePhysicsContactObservers.add(world);
    try {
        const k = kernel(world);
        const contacts = contactCapacity(state);
        for (let id = 0; id < contacts; id++) {
            if (contactField(state, id, ContactField.contactId) < 0) continue;
            const flags = contactField(state, id, ContactField.flags) >>> 0;
            // simTouching is the solver's current contact state; the low contact flag and
            // contactBegin/contactEnd records are broad-phase/event state and are insufficient.
            if ((flags & ContactFlags.simTouchingFlag) === 0) continue;
            const shapeA = contactField(state, id, ContactField.shapeIdA);
            const shapeB = contactField(state, id, ContactField.shapeIdB);
            const childIndex = contactField(state, id, ContactField.childIndex);
            if (shapeA < 0 || shapeB < 0 || childIndex < 0) continue;
            const pointer = k.contactQuery(state.worldId, shapeA, shapeB, childIndex) >>> 0;
            if (!pointer) continue;
            const contact = runtime.contactData.read(k, pointer, childIndex, state);
            if (!contact || contact.manifoldCount === 0) continue;
            // ContactDataBuffer.read has just copied the live generation-aware bindings into
            // these per-world scratch records; they match contact.shapeA/contact.shapeB.
            visit(contact, runtime.contactData.first.body, runtime.contactData.second.body);
        }
        return stamp;
    } finally {
        activePhysicsContactObservers.delete(world);
    }
}
