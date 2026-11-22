import type { World } from "../../engine";
import type { View } from "./view";
import type { PresentationTicket } from "./presentation-ticket";

/**
 * Per-canvas presentation tickets for the isolated 0.10.5 engine.
 *
 * A world becomes canvas-managed by its first request. Each canvas then owns
 * its ticket, so separate canvases do not compete for one world ticket. A
 * canvas without a ticket still renders into an engine-owned stand-in target;
 * it is not acquired or presented until its next ticket is consumed.
 */
interface CanvasTicketState {
    next: number;
    pending: Map<GPUCanvasContext, PresentationTicket>;
}

const canvasStates = new WeakMap<object, CanvasTicketState>();

interface StandInEntry {
    texture: GPUTexture;
    device: GPUDevice;
    world: object;
}

const standIns = new WeakMap<GPUCanvasContext, StandInEntry>();
const worldStandIns = new WeakMap<object, Set<GPUCanvasContext>>();

type MaybeDisposableWorld = World & { onDispose?: (fn: () => void) => void };

export function requestCanvasPresentationTicket(
    world: World,
    context: GPUCanvasContext,
    reason = "content-change",
): PresentationTicket {
    let state = canvasStates.get(world);
    if (!state) {
        state = { next: 0, pending: new Map() };
        canvasStates.set(world, state);
    }
    const ticket = Object.freeze({ serial: ++state.next, reason });
    state.pending.set(context, ticket);
    return ticket;
}

export function canvasPresentationManaged(world: World): boolean {
    return canvasStates.has(world);
}

export function takeCanvasPresentationTicket(
    world: World,
    context: GPUCanvasContext,
): PresentationTicket | undefined {
    const state = canvasStates.get(world);
    if (!state) return undefined;
    const ticket = state.pending.get(context);
    state.pending.delete(context);
    return ticket;
}

export function targetlessPresentationTexture(world: World, view: View): GPUTexture | undefined {
    const context = view.context;
    const canvas = view.canvas;
    const device = world.gpu.device;
    if (!context || !canvas || !device) return undefined;
    const configuration = context.getConfiguration();
    if (!configuration || canvas.width === 0 || canvas.height === 0) return undefined;
    let owned = worldStandIns.get(world);
    if (!owned) {
        owned = new Set();
        worldStandIns.set(world, owned);
        const maybeDisposable = world as MaybeDisposableWorld;
        if (typeof maybeDisposable.onDispose === "function") {
            const contexts = owned;
            maybeDisposable.onDispose(() => {
                for (const cached of contexts) {
                    const entry = standIns.get(cached);
                    if (entry && entry.world === world) {
                        entry.texture.destroy();
                        standIns.delete(cached);
                    }
                }
                contexts.clear();
            });
        }
    }
    owned.add(context);
    let entry = standIns.get(context);
    if (entry && entry.world !== world) {
        worldStandIns.get(entry.world)?.delete(context);
        entry.world = world;
    }
    if (
        !entry ||
        entry.device !== device ||
        entry.texture.width !== canvas.width ||
        entry.texture.height !== canvas.height ||
        entry.texture.format !== configuration.format
    ) {
        entry?.texture.destroy();
        entry = {
            texture: device.createTexture({
                label: "targetless canvas stand-in",
                size: [canvas.width, canvas.height],
                format: configuration.format,
                usage:
                    (configuration.usage ?? GPUTextureUsage.RENDER_ATTACHMENT) |
                    GPUTextureUsage.RENDER_ATTACHMENT |
                    GPUTextureUsage.TEXTURE_BINDING |
                    GPUTextureUsage.COPY_SRC |
                    GPUTextureUsage.COPY_DST,
            }),
            device,
            world,
        };
        standIns.set(context, entry);
    }
    return entry.texture;
}
