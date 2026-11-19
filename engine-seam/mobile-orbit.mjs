import { readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

/** Exact-source transform for the pinned Shallot 0.10.5 Orbit owner. */
export const MOBILE_ORBIT_MARKER = "/* TRUEFF_MAIN_MOBILE_ORBIT_0103 */";
export const MOBILE_ORBIT_BARREL_MARKER = "/* TRUEFF_MAIN_MOBILE_ORBIT_BARREL_0103 */";
export const MOBILE_ORBIT_VERSION = "0.10.5";

function once(source, anchor, replacement, label) {
    const count = source.split(anchor).length - 1;
    if (count !== 1) {
        throw new Error(`MAIN mobile Orbit anchor ${label} expected once, found ${count}`);
    }
    return source.replace(anchor, replacement);
}

/**
 * Add a mutable, World-owned mobile intent resource to the existing Orbit
 * input/fly system.  The transform is deliberately source-pinned: it refuses
 * a different Orbit implementation instead of silently producing a partial
 * input path.  A second invocation is a no-op.
 */
export function transformMobileOrbitSource(source) {
    if (source.includes(MOBILE_ORBIT_MARKER)) {
        if(!source.includes('export const OrbitMode ='))throw Error('MAIN mobile Orbit resource declaration changed');
        return source;
    }

    const resourceAnchor =
        "/** `Free` consumes orbit, pan, zoom and fly input; `Locked` disables look; `Manual` consumes no device input and follows authored Orbit fields. */\nexport const OrbitMode = { Free: 0, Locked: 1, Manual: 2 } as const;";
    const resourceReplacement = `${resourceAnchor}

/** Numeric mobile fly intent. The adapter owns DOM input; Orbit owns pose integration. */
export interface OrbitMobileInputState {
    active: boolean;
    camera: number;
    /** strafe, forward and up are normalized to [-1, 1]. */
    strafe: number;
    forward: number;
    up: number;
}

export const OrbitMobileInput: import("../../engine").Resource<OrbitMobileInputState> = {
    create: () => ({ active: false, camera: -1, strafe: 0, forward: 0, up: 0 }),
};`;
    source = once(source, resourceAnchor, resourceReplacement, "resource declaration");

    const inputAnchor = "            const touchCount = manual ? 0 : input.touch.count;\n            // Touch capture also holds pointer.left; reading mouse buttons here would orbit during pinch/pan.\n            // Fly has no touch equivalent.\n            const orbitHeld =\n                !manual &&\n                (touchCount > 0\n                    ? touchCount === 1\n                    : isButton(input.pointer, orbit.orbitButton.get(eid)));\n            const panHeld =\n                !manual &&\n                (touchCount > 0\n                    ? touchCount >= 2\n                    : isButton(input.pointer, orbit.panButton.get(eid)));\n            const flyHeld =\n                !manual &&\n                (touchCount > 0 ? false : isButton(input.pointer, orbit.flyButton.get(eid)));\n            // Leave movement keys to gameplay unless the fly button is held; orbit and pan take priority.\n            const flying = !isOrtho && !orbitHeld && !panHeld && flyHeld;";
    const inputReplacement = `            const mobile = world.resource(OrbitMobileInput);
            // A thumb owns the fly branch only while it carries non-zero intent. This
            // lets a right-side canvas touch continue to feed the existing look delta,
            // while a neutral/released thumb returns touch orbit/pan to the native path.
            const mobileActive =
                !manual &&
                !isOrtho &&
                mobile.active &&
                mobile.camera === eid &&
                (mobile.strafe !== 0 || mobile.forward !== 0 || mobile.up !== 0);
            const touchCount = manual ? 0 : input.touch.count;
            // Touch capture also holds pointer.left; reading mouse buttons here would orbit during pinch/pan.
            // A mobile thumb is the one explicit touch equivalent of fly.
            const orbitHeld =
                !manual && !mobileActive &&
                (touchCount > 0
                    ? touchCount === 1
                    : isButton(input.pointer, orbit.orbitButton.get(eid)));
            const panHeld =
                !manual && !mobileActive &&
                (touchCount > 0
                    ? touchCount >= 2
                    : isButton(input.pointer, orbit.panButton.get(eid)));
            const flyHeld =
                !manual &&
                (mobileActive ||
                    (touchCount > 0 ? false : isButton(input.pointer, orbit.flyButton.get(eid))));
            // Leave movement keys to gameplay unless the fly button is held; orbit and pan take priority.
            const flying = !isOrtho && !orbitHeld && !panHeld && flyHeld;`;
    source = once(source, inputAnchor, inputReplacement, "input ownership");

    const movementAnchor = `                // normalize the world move so a diagonal (e.g. forward + up) travels at \`speed\`, not faster
                let wx = mz * sy * cp + mx * cy;
                let wy = my - mz * sp;
                let wz = mz * cy * cp - mx * sy;
                const len = Math.hypot(wx, wy, wz);
                if (len > 0) {
                    const k = speed / len;
                    wx *= k;
                    wy *= k;
                    wz *= k;
                }`;
    const movementReplacement = `                // Keep the original digital normalization untouched. Mobile axes are
                // already normalized by the DOM producer, so scaling them by speed
                // preserves a half-deflection as half-speed instead of saturating it.
                let wx = mz * sy * cp + mx * cy;
                let wy = my - mz * sp;
                let wz = mz * cy * cp - mx * sy;
                if (mobileActive) {
                    wx = -mobile.forward * sy * cp + mobile.strafe * cy;
                    wy = mobile.up + mobile.forward * sp;
                    wz = -mobile.forward * cy * cp - mobile.strafe * sy;
                    const k = speed / Math.max(1, Math.hypot(wx, wy, wz));
                    wx *= k;
                    wy *= k;
                    wz *= k;
                } else {
                    const len = Math.hypot(wx, wy, wz);
                    if (len > 0) {
                        const k = speed / len;
                        wx *= k;
                        wy *= k;
                        wz *= k;
                    }
                }`;
    source = once(source, movementAnchor, movementReplacement, "analog fly movement");

    return `${source}\n${MOBILE_ORBIT_MARKER}\n`;
}

/** Publish the resource through the package's existing extras barrel. */
export function transformMobileOrbitBarrelSource(source) {
    if (source.includes(MOBILE_ORBIT_BARREL_MARKER)) {
        if(!source.includes('    Orbit,'))throw Error('MAIN mobile Orbit extras barrel changed');
        return source;
    }
    const anchor = `export {\n    Orbit,\n    OrbitMode,\n    OrbitPlugin,\n} from "./orbit";`;
    const replacement = `export {\n    Orbit,\n    OrbitMode,\n    OrbitMobileInput,\n    type OrbitMobileInputState,\n    OrbitPlugin,\n} from "./orbit";\n${MOBILE_ORBIT_BARREL_MARKER}`;
    return once(source, anchor, replacement, "extras barrel");
}

/** Apply the transform to one isolated Shallot installation, with a version guard. */
export async function applyMobileOrbitPatch(engine) {
    const packagePath = resolve(engine, "package.json");
    const version = JSON.parse(await readFile(packagePath, "utf8")).version;
    if (version !== MOBILE_ORBIT_VERSION) {
        throw new Error(
            `MAIN mobile Orbit patch requires Shallot ${MOBILE_ORBIT_VERSION}, received ${version}`,
        );
    }
    const orbitPath = resolve(engine, "src/extras/orbit/index.ts");
    const barrelPath = resolve(engine, "src/extras/index.ts");
    const [before, barrelBefore] = await Promise.all([
        readFile(orbitPath, "utf8"),
        readFile(barrelPath, "utf8"),
    ]);
    const after = transformMobileOrbitSource(before);
    const barrelAfter = transformMobileOrbitBarrelSource(barrelBefore);
    if (after !== before) await writeFile(orbitPath, after);
    if (barrelAfter !== barrelBefore) await writeFile(barrelPath, barrelAfter);
    return { orbit: after, barrel: barrelAfter };
}
