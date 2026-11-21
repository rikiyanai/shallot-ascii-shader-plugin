/** Durable diagnostics transform for the pinned Shallot 0.10.5 engine.
 * The loading panel and its Copy details action must preserve the actual
 * message even when Safari supplies a stack containing only a bundle offset.
 * The regression tests execute this injected formatter rather than a copy. */
export const LOADING_DIAGNOSTICS_MARKER = "/* TRUEFF_MAIN_LOADING_DIAGNOSTICS_0105 */";
export const LOADING_DIAGNOSTICS_HOOK_MARKER = "/* TRUEFF_MAIN_PLUGIN_HOOK_CAUSE_0105 */";
export const LOADING_DIAGNOSTICS_ERROR_PRIORITY_MARKER = "/* TRUEFF_MAIN_LOADING_ERROR_PRIORITY_0105 */";

export function transformLoadingDiagnostics(source) {
    const before = `function diagnosticText(error: Error): string {
    const ua = typeof navigator !== "undefined" ? navigator.userAgent : "unknown";
    const stack = error.stack ?? \`${"${error.name}: ${error.message}"}\`;
    return \`shallot v\${pkg.version}\\n\${stack}\\n\\nUser agent: \${ua}\`;
}`;
    const detail = "    detail.textContent = `${error.name}: ${error.message}`;";
    const patchedDetail = "    detail.textContent = diagnosticErrorText(error);";
    const markerAt = source.indexOf(LOADING_DIAGNOSTICS_MARKER);
    const replacement = `${LOADING_DIAGNOSTICS_MARKER}
function safeDiagnosticString(value: unknown): string {
    try { return String(value); } catch { return "[unprintable diagnostic value]"; }
}

function formatDiagnosticError(error: unknown, seen = new Set<object>(), depth = 0): string {
    if (error === null || error === undefined) return safeDiagnosticString(error);
    if ((typeof error !== "object" && typeof error !== "function")) return safeDiagnosticString(error);
    if (seen.has(error)) return "[cause cycle]";
    if (depth >= 8) return "[cause depth limit]";
    seen.add(error);
    try {
        const value = error as { name?: unknown; message?: unknown; stack?: unknown; cause?: unknown; errors?: unknown };
        const name = typeof value.name === "string" && value.name ? value.name : "Error";
        const message = typeof value.message === "string" ? value.message : safeDiagnosticString(error);
        const lines = [\`${"${name}: ${message}"}\`];
        if (typeof value.stack === "string" && value.stack) lines.push(\`Stack:\\n\${value.stack}\`);
        if (Array.isArray(value.errors) && value.errors.length) {
            lines.push("Aggregate errors:");
            value.errors.forEach((item, index) => lines.push(\`[\${index}] \${formatDiagnosticError(item, seen, depth + 1)}\`));
        }
        if ("cause" in value && value.cause !== undefined) {
            lines.push(\`Cause:\\n\${formatDiagnosticError(value.cause, seen, depth + 1)}\`);
        }
        return lines.join("\\n");
    } catch (cause) {
        return \`[unreadable diagnostic value: \${safeDiagnosticString(cause)}]\`;
    } finally {
        seen.delete(error);
    }
}

function diagnosticErrorText(error: unknown): string {
    return formatDiagnosticError(error);
}

function diagnosticText(error: Error): string {
    const ua = typeof navigator !== "undefined" ? navigator.userAgent : "unknown";
    return \`shallot v\${pkg.version}\\n\${diagnosticErrorText(error)}\\n\\nUser agent: \${ua}\`;
}`;
    if (markerAt >= 0) {
        const markerEnd = source.indexOf("\nfunction createLink", markerAt);
        if (markerEnd < 0 || source.slice(markerAt, markerEnd).trimEnd() !== replacement ||
            source.split(patchedDetail).length !== 2 ||
            source.split("navigator.clipboard.writeText(diagnosticText(error))").length !== 2) {
            throw new Error("MAIN loading diagnostics: marked formatter changed");
        }
        return source;
    }
    if (source.split(before).length !== 2) throw new Error("MAIN loading diagnostics: diagnosticText anchor changed");
    source = source.replace(before, replacement);
    if (source.split(detail).length !== 2) throw new Error("MAIN loading diagnostics: visible detail anchor changed");
    source = source.replace(detail, patchedDetail);
    return source;
}

export function transformPluginHookCause(source) {
    const before = `function pluginHookError(plugin: Plugin, hook: "initialize" | "warm", error: unknown): Error {
    const detail = error instanceof Error ? error.message : String(error);
    return new Error(\`Plugin "\${plugin.name}" \${hook} threw: \${detail}\`);
}`;
    const after = `${LOADING_DIAGNOSTICS_HOOK_MARKER}
function pluginHookError(plugin: Plugin, hook: "initialize" | "warm", error: unknown): Error {
    const detail = error instanceof Error ? error.message : String(error);
    return new Error(\`Plugin "\${plugin.name}" \${hook} threw: \${detail}\`, { cause: error });
}`;
    if (source.includes(LOADING_DIAGNOSTICS_HOOK_MARKER)) {
        if (source.split(after).length === 2) return source;
        throw new Error("MAIN plugin hook diagnostics: marked wrapper changed");
    }
    if (source.split(before).length !== 2) throw new Error("MAIN plugin hook diagnostics: anchor changed");
    return source.replace(before, after);
}

/** Keep a terminal engine error above MAIN's pending external notice. Ordinary
 * loading/progress overlays retain their existing z-index and lifecycle. */
export function transformLoadingErrorPriority(source) {
    const before = `function renderEngineError(overlay: HTMLDivElement, error: Error, theme: Theme): void {
    overlay.replaceChildren();`;
    const after = `${LOADING_DIAGNOSTICS_ERROR_PRIORITY_MARKER}
function renderEngineError(overlay: HTMLDivElement, error: Error, theme: Theme): void {
    overlay.replaceChildren();
    overlay.dataset.shallotLoadingState = "terminal-error";
    overlay.style.zIndex = "10002";`;
    if (source.includes(LOADING_DIAGNOSTICS_ERROR_PRIORITY_MARKER)) {
        if (source.split(after).length === 2) return source;
        throw new Error("MAIN loading diagnostics: marked error priority changed");
    }
    if (source.split(before).length !== 2) throw new Error("MAIN loading diagnostics: renderEngineError anchor changed");
    return source.replace(before, after);
}
