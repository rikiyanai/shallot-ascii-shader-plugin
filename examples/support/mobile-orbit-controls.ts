/**
 * Generic mobile input adapter for the engine's existing Orbit fly branch.
 *
 * This module owns only normalized intent and the shared DOM control surface.
 * It never writes a camera pose, synthesizes a key/pointer event, or installs
 * the engine's Orbit plugin. A composition that does not already own a perspective Orbit
 * camera remains untouched (including existing first-person touch controls).
 */
import {
    Camera,
    CameraMode,
    type Plugin,
    type System,
    type World,
} from "@dylanebert/shallot";
import {
    Orbit,
    OrbitMobileInput,
    type OrbitMobileInputState,
} from "@dylanebert/shallot/extras";
import {
    mountMobileControls,
    supportsTouchControls,
    type MobileControlsHandle,
} from "../../packages/shallot-touch/controls";

export type MobileOrbitVerticalAction = "up" | "down";

function unit(value: number): number {
    if (!Number.isFinite(value)) return 0;
    return Math.max(-1, Math.min(1, value));
}

/** Apply one joystick sample without allocating or normalizing its magnitude away. */
export function setOrbitMobileMove(state: OrbitMobileInputState, x: number, y: number): void {
    state.strafe = unit(x);
    // DOM y grows downward; Orbit's forward axis is -Z, so up is forward.
    const forward = unit(y);
    state.forward = forward === 0 ? 0 : -forward;
    state.active = state.strafe !== 0 || state.forward !== 0 || state.up !== 0;
}

/** Update one held vertical action; simultaneous Up+Down cancels to zero. */
export function setOrbitMobileVertical(
    state: OrbitMobileInputState,
    held: Set<MobileOrbitVerticalAction>,
    action: MobileOrbitVerticalAction,
    pressed: boolean,
): void {
    if (pressed) held.add(action);
    else held.delete(action);
    state.up = Number(held.has("up")) - Number(held.has("down"));
    state.active = state.strafe !== 0 || state.forward !== 0 || state.up !== 0;
}

/** Neutralize all axes at lifecycle boundaries. */
export function resetOrbitMobileInput(
    state: OrbitMobileInputState,
    held?: Set<MobileOrbitVerticalAction>,
): void {
    state.active = false;
    state.strafe = 0;
    state.forward = 0;
    state.up = 0;
    held?.clear();
}

/** Resource lookup is lazy; this query is empty when a composition has no Orbit owner. */
export function findPerspectiveOrbit(world: World): number {
    let cameraStorage: { mode: { get(eid: number): number } } | undefined;
    for (const eid of world.query(ORBIT_CAMERAS)) {
        cameraStorage ??= world.storage(Camera);
        if (cameraStorage.mode.get(eid) === CameraMode.Perspective) return eid;
    }
    return -1;
}
const ORBIT_CAMERAS=[Camera,Orbit];

function hasCoarsePointer(): boolean {
    return supportsTouchControls();
}

interface MobileOrbitState {
    coarse:boolean;
    readonly axes: OrbitMobileInputState;
    readonly held: Set<MobileOrbitVerticalAction>;
    handle?: MobileControlsHandle;
}

const states = new WeakMap<World, MobileOrbitState>();

function release(world: World, state: MobileOrbitState): void {
    state.handle?.dispose();
    state.handle = undefined;
    resetOrbitMobileInput(state.axes, state.held);
    states.delete(world);
}

function mountForPerspectiveOrbit(world: World, state: MobileOrbitState): void {
    if (state.handle || !state.coarse) return;
    const camera=findPerspectiveOrbit(world);if(camera<0)return;
    state.axes.camera=camera;
    state.handle = mountMobileControls({
        // The adapter has already gated on coarse pointer; the shared module
        // still receives the explicit override for deterministic hosts.
        coarsePointer: true,
        // Existing Orbit compositions may own a bottom scene chooser and a
        // right-side fire button. Keep both unobstructed, without scene keys.
        bottomInset: 72,
        actionAlign: 'start',
        active: () => !world.disposed&&world.storage(Camera).mode.get(camera)===CameraMode.Perspective,
        onMove: (x, y) => setOrbitMobileMove(state.axes, x, y),
        actions: {
            up: {
                label: "Up",
                ariaLabel: "Fly up",
                onPress: () => setOrbitMobileVertical(state.axes, state.held, "up", true),
                onRelease: () => setOrbitMobileVertical(state.axes, state.held, "up", false),
                onCancel: () => setOrbitMobileVertical(state.axes, state.held, "up", false),
            },
            down: {
                label: "Down",
                ariaLabel: "Fly down",
                onPress: () => setOrbitMobileVertical(state.axes, state.held, "down", true),
                onRelease: () => setOrbitMobileVertical(state.axes, state.held, "down", false),
                onCancel: () => setOrbitMobileVertical(state.axes, state.held, "down", false),
            },
        },
    });
}

const mountSystem: System = {
    name: "Orbit mobile controls",
    group: "simulation",
    update(world) {
        const state = states.get(world);
        if (state) mountForPerspectiveOrbit(world, state);
    },
};

/** Install the adapter beside an existing Orbit composition; it adds no engine plugin dependency. */
export const OrbitMobileControlsPlugin: Plugin = {
    name: "Orbit mobile controls",
    systems: [mountSystem],
    initialize(world) {
        const state: MobileOrbitState = {
            coarse:hasCoarsePointer(),
            axes: world.resource(OrbitMobileInput),
            held: new Set(),
        };
        states.set(world, state);
        const media=typeof window!=='undefined'?window.matchMedia?.('(any-pointer: coarse)'):undefined;
        const changed=()=>{state.coarse=hasCoarsePointer();if(!state.coarse){state.handle?.dispose();state.handle=undefined;resetOrbitMobileInput(state.axes,state.held);}};
        media?.addEventListener('change',changed);
        world.onDispose(()=>media?.removeEventListener('change',changed));
        world.onDispose(() => release(world, state));
    },
    dispose(world) {
        const state = states.get(world);
        if (state) release(world, state);
    },
};
