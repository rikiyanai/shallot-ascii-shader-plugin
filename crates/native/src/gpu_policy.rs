fn parse_require_hardware(value: Option<&str>) -> bool {
    value != Some("1")
}

pub fn require_hardware() -> bool {
    parse_require_hardware(std::env::var("SHALLOT_ALLOW_SOFTWARE_GPU").ok().as_deref())
}

pub fn initialization_script() -> &'static str {
    script(require_hardware())
}

fn script(require_hardware: bool) -> &'static str {
    if require_hardware {
        "Object.defineProperty(globalThis, '__shallotRequireHardwareGpu', { value: true });"
    } else {
        "Object.defineProperty(globalThis, '__shallotRequireHardwareGpu', { value: false });"
    }
}

#[cfg(all(target_os = "linux", feature = "portable"))]
pub fn configure_linux(
    cmd: &mut cef::CommandLine,
    platform: Option<&str>,
    session: Option<&str>,
    hardware: bool,
) {
    use cef::{CefString, ImplCommandLine};
    // Chromium resolves an absent Ozone platform from the session, defaulting to X11.
    let platform = platform.unwrap_or(if session == Some("wayland") {
        "wayland"
    } else {
        "x11"
    });
    match platform {
        "wayland" => {
            // ANGLE Vulkan avoids Wayland stalls, but its X11 configs can lack Chromium's ARGB visual.
            cmd.append_switch_with_value(
                Some(&CefString::from("use-angle")),
                Some(&CefString::from("vulkan")),
            );
        }
        "x11" if hardware => {
            // Vulkan composition supplies the provider needed for hardware WebGPU without
            // ANGLE Vulkan's X11 visual mismatch.
            let key = CefString::from("enable-features");
            let value = CefString::from(&cmd.switch_value(Some(&key))).to_string();
            if !value.split(',').any(|entry| {
                let entry = entry.trim_ascii();
                let entry = entry.strip_prefix('*').unwrap_or(entry);
                entry.split(['<', ':']).next() == Some("Vulkan")
            }) {
                let value = if value.is_empty() {
                    "Vulkan".to_string()
                } else {
                    format!("{value},Vulkan")
                };
                cmd.append_switch_with_value(Some(&key), Some(&CefString::from(value.as_str())));
            }
        }
        _ => {}
    }
}

#[cfg(feature = "portable")]
mod cef_context {
    use cef::*;

    wrap_app! {
        pub struct PolicyApp;

        impl App {
            fn render_process_handler(&self) -> Option<RenderProcessHandler> {
                Some(PolicyHandler::new())
            }
        }
    }

    wrap_render_process_handler! {
        pub struct PolicyHandler;

        impl RenderProcessHandler {
            fn on_context_created(
                &self,
                _browser: Option<&mut Browser>,
                _frame: Option<&mut Frame>,
                context: Option<&mut V8Context>,
            ) {
                if let Some(context) = context {
                    let mut value = None;
                    let mut exception = None;
                    assert_ne!(context.eval(
                        Some(&CefString::from(super::initialization_script())),
                        None, 0, Some(&mut value), Some(&mut exception),
                    ), 0, "failed to publish native GPU policy");
                }
            }
        }
    }
}

#[cfg(all(target_os = "macos", feature = "portable"))]
pub use cef_context::PolicyApp;
#[cfg(feature = "portable")]
pub use cef_context::PolicyHandler;
