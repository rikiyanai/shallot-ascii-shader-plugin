/**
 * a captured GPU-side log line: what a `console.log` inside a TGSL kernel printed, once its ring buffer
 * was read back. `errors` holds the `console.error` subset, which a browser driver fails on.
 */
export interface GpuLog {
    lines: string[];
    errors: string[];
}

// typegpu prefixes every GPU line with `%c<messagePrefix>%c ` and two CSS argument strings, then the
// deserialized values. The prefix is the only thing distinguishing a GPU line from any other page log,
// since both arrive through the same `console` methods.
const MARK = "%c GPU %c ";


// and declaring it globally would put `__gpuLog` in every consumer's ambient types.
const captured = (): GpuLog | undefined => (globalThis as { __gpuLog?: GpuLog }).__gpuLog;

let patched = false;
let drainTail: Promise<void> = Promise.resolve();
let drainPoison: Error | undefined;

function text(args: unknown[]): string {
    const head = String(args[0]).slice(MARK.length);
    const rest = args.slice(3).map((a) => (typeof a === "string" ? a : JSON.stringify(a)));
    return rest.length > 0 ? `${head} ${rest.join(" ")}` : head;
}









export function captureGpuLog(): void {
    const log = captured();
    if (!log || patched) return;
    patched = true;
    for (const level of ["log", "warn", "error"] as const) {
        const original = console[level];
        console[level] = (...args: unknown[]) => {
            if (typeof args[0] === "string" && args[0].startsWith(MARK)) {
                const line = text(args);
                log.lines.push(line);
                if (level === "error") log.errors.push(line);
            }
            original.apply(console, args as never);
        };
    }
}

async function drain(trigger: () => void, timeoutMs: number): Promise<string[]> {
    const log = captured();
    const at = log?.lines.length ?? 0;
    trigger();
    if (!log) return [];
    const deadline = performance.now() + timeoutMs;
    while (log.lines.length === at && performance.now() < deadline) {
        await new Promise((resolve) => setTimeout(resolve, 1));
    }
    if (log.lines.length === at) {
        drainPoison ??= new Error(
            `drainLog timed out after ${timeoutMs}ms; this capture session is poisoned`,
        );
        throw drainPoison;
    }
    return log.lines.slice(at);
}























export function drainLog(trigger: () => void, timeoutMs = 1000): Promise<string[]> {
    const batch = drainTail.then(() => {
        if (drainPoison) throw drainPoison;
        return drain(trigger, timeoutMs);
    });
    drainTail = batch.then(
        () => undefined,
        () => undefined,
    );
    return batch;
}
