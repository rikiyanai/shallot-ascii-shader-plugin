import type { World } from "../../engine";

/**
 * One engine-owned permission to acquire a canvas presentation target.
 *
 * The ticket is deliberately opaque to the renderer.  It is created before
 * `World.step()`, consumed by BeginFrameSystem before canvas acquisition, and
 * attached to the presenting View only when acquisition actually succeeded.
 * A later request supersedes an older one. Unmanaged worlds receive the
 * automatic ticket below; only a world that has explicitly requested manual
 * tickets treats an absent ticket as permission to skip `getCurrentTexture()`.
 */
export interface PresentationTicket {
    readonly serial: number;
    readonly reason: string;
}

interface TicketState {
    next: number;
    pending?: PresentationTicket;
}

const states = new WeakMap<object, TicketState>();
const AUTOMATIC_PRESENTATION_TICKET: PresentationTicket = Object.freeze({
    serial: 0,
    reason: "automatic",
});

function state(world: object): TicketState {
    let value = states.get(world);
    if (!value) {
        value = { next: 0 };
        states.set(world, value);
    }
    return value;
}

/** Request the next canvas acquisition before entering `World.step()`. */
export function requestPresentationTicket(world: World, reason = "content-change"): PresentationTicket {
    const current = state(world);
    const ticket = Object.freeze({ serial: ++current.next, reason });
    current.pending = ticket;
    return ticket;
}

/** Consume the one ticket that authorizes this frame's canvas acquisition. */
export function takePresentationTicket(world: World): PresentationTicket | undefined {
    const current = states.get(world);
    // Existing Shallot worlds are unmanaged until a caller explicitly asks
    // for ticketed presentation. Preserve the engine's original automatic
    // canvas acquisition for those worlds, including multiple canvas views.
    if (!current) return AUTOMATIC_PRESENTATION_TICKET;
    const ticket = current.pending;
    current.pending = undefined;
    return ticket;
}

