use cef::*;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::OnceLock;

use crate::{app_name, asset, cache_dir, content_type};

#[cfg(target_os = "linux")]
const VK_F11: i32 = 0x7A;
#[cfg(target_os = "linux")]
const VK_RETURN: i32 = 0x0D;
#[cfg(target_os = "linux")]
const VK_ESCAPE: i32 = 0x1B;
#[cfg(target_os = "linux")]
const EVENTFLAG_ALT: u32 = 8;

mod policy;

static ORIGIN: OnceLock<String> = OnceLock::new();

fn start_asset_server() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").expect("failed to bind asset server");
    let port = listener.local_addr().unwrap().port();
    let origin = format!("http://127.0.0.1:{}", port);
    eprintln!("[shallot] asset server on {}", origin);

    // Chromium opens speculative connections that may send nothing, so each connection is served
    // on its own thread: a serial loop would block every later request on that first read.
    std::thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            std::thread::spawn(move || serve(stream));
        }
    });

    origin
}

fn serve(mut stream: TcpStream) {
    let mut buf = [0u8; 4096];
    let n = match stream.read(&mut buf) {
        Ok(n) if n > 0 => n,
        _ => return,
    };
    let request = String::from_utf8_lossy(&buf[..n]);
    let path = request
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .unwrap_or("/");
    let path = if path == "/" { "/index.html" } else { path };

    match asset(&path[1..]) {
        Some(data) => {
            let mime = content_type(path);
            // Cross-origin isolation permits shared-memory physics workers.
            let header = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: {}\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\nCross-Origin-Opener-Policy: same-origin\r\nCross-Origin-Embedder-Policy: require-corp\r\nConnection: close\r\n\r\n",
                mime, data.len()
            );
            let _ = stream.write_all(header.as_bytes());
            let _ = stream.write_all(&data);
        }
        None => {
            let _ = stream.write_all(
                b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
            );
        }
    }
}

/// Switches every CEF process receives unconditionally.
fn fixed_switches() -> impl Iterator<Item = &'static str> {
    // enable-unsafe-webgpu lifts the WebGPU adapter blocklist; ignore-gpu-blocklist lifts the broader
    // GPU software-rendering list — without both, requestAdapter() returns null on otherwise-capable
    // GPUs. On macOS, Chromium keeps its cookie key in the login keychain under an ACL for the signature
    // that created it, so each ad-hoc-signed rebuild would block its first page load on a keychain
    // password prompt.
    ["enable-unsafe-webgpu", "ignore-gpu-blocklist"]
        .into_iter()
        .chain(cfg!(target_os = "macos").then_some("use-mock-keychain"))
}

wrap_app! {
    pub struct ShallotApp;

    impl App {
        fn render_process_handler(&self) -> Option<RenderProcessHandler> {
            Some(crate::gpu_policy::PolicyHandler::new())
        }

        fn on_before_command_line_processing(
            &self,
            _process_type: Option<&CefString>,
            command_line: Option<&mut CommandLine>,
        ) {
            if let Some(cmd) = command_line {
                for switch in fixed_switches() {
                    cmd.append_switch(Some(&CefString::from(switch)));
                }
                // allow_unsafe_apis exposes the unsafe Dawn features an engine plugin may require
                // (timestamp-query, which ProfilePlugin declares).
                let dawn = CefString::from("enable-dawn-features");
                let value = CefString::from(&cmd.switch_value(Some(&dawn))).to_string();
                if !value.split(',').any(|feature| feature == "allow_unsafe_apis") {
                    let value = if value.is_empty() {
                        "allow_unsafe_apis".to_string()
                    } else {
                        format!("{value},allow_unsafe_apis")
                    };
                    cmd.append_switch_with_value(Some(&dawn), Some(&CefString::from(value.as_str())));
                }
                #[cfg(target_os = "linux")]
                {
                    let ozone = CefString::from("ozone-platform");
                    let platform = (cmd.has_switch(Some(&ozone)) != 0)
                        .then(|| CefString::from(&cmd.switch_value(Some(&ozone))).to_string());
                    let session_type = std::env::var("XDG_SESSION_TYPE").ok();
                    crate::gpu_policy::configure_linux(
                        cmd, platform.as_deref(), session_type.as_deref(),
                        crate::gpu_policy::require_hardware(),
                    );
                }
                // Windows: run the GPU in-process. The re-exec'd GPU subprocess can't initialize GL in
                // this single-exe CEF build — gl::init::InitializeStaticGLBindingsOneOff fails, the GPU
                // process exits, Chromium falls back to use-gl=disabled, and requestAdapter() returns
                // null ("no compatible GPU"). In-process the discrete GPU is detected and hardware
                // WebGPU works (verified rendering on real hardware).
                #[cfg(target_os = "windows")]
                cmd.append_switch(Some(&CefString::from("in-process-gpu")));
                #[cfg(debug_assertions)]
                cmd.append_switch_with_value(
                    Some(&CefString::from("remote-debugging-port")),
                    Some(&CefString::from("0")),
                );
            }
        }

        fn browser_process_handler(&self) -> Option<BrowserProcessHandler> {
            Some(ShallotBrowserProcessHandler::new())
        }
    }
}

wrap_browser_process_handler! {
    struct ShallotBrowserProcessHandler;

    impl BrowserProcessHandler {
        fn on_context_initialized(&self) {
            let origin = ORIGIN.get().expect("asset server not started");
            let fallback = format!("{}/", origin);
            let launch_url = crate::launch_url(std::env::var("SHALLOT_DEV_URL").ok(), fallback);
            let url = CefString::from(launch_url.as_str());
            let settings = BrowserSettings::default();
            let mut client = ShallotClient::new();
            let mut delegate = ShallotBrowserViewDelegate::new();
            let view = browser_view_create(
                Some(&mut client), Some(&url), Some(&settings), None, None,
                Some(&mut delegate),
            ).expect("failed to create app browser view");
            let icons = window_icons(&crate::window_icon(), || image_create().expect("CEF could not create native icon image"));
            let mut window_delegate = ShallotWindowDelegate::new(std::cell::RefCell::new(Some(view)), Some(icons));
            window_create_top_level(Some(&mut window_delegate)).expect("failed to create app window");
        }
    }
}

wrap_browser_view_delegate! {
    struct ShallotBrowserViewDelegate {}
    impl ViewDelegate {}
    impl BrowserViewDelegate {
        fn browser_runtime_style(&self) -> RuntimeStyle { RuntimeStyle::CHROME }
        fn chrome_toolbar_type(&self, _view: Option<&mut BrowserView>) -> ChromeToolbarType {
            ChromeToolbarType::NONE
        }
    }
}

fn window_icons(png: &image::RgbaImage, mut create: impl FnMut() -> Image) -> (Image, Image) {
    let mut make = |size: u32| {
        let image = create();
        fill_icon(&image, png, size);
        image
    };
    (make(16), make(32))
}

fn fill_icon(image: &Image, png: &image::RgbaImage, size: u32) {
    for scale in [1_u32, 2] {
        let pixels = image::imageops::resize(
            png,
            size * scale,
            size * scale,
            image::imageops::FilterType::Lanczos3,
        );
        assert_ne!(
            image.add_bitmap(
                scale as f32,
                (size * scale) as i32,
                (size * scale) as i32,
                ColorType::RGBA_8888,
                AlphaType::POSTMULTIPLIED,
                Some(pixels.as_raw())
            ),
            0,
            "CEF refused native icon bitmap"
        );
    }
}

fn set_window_icons(window: &Window, mut small: Image, mut app: Image) {
    window.set_window_icon(Some(&mut small));
    window.set_window_app_icon(Some(&mut app));
}

wrap_window_delegate! {
    struct ShallotWindowDelegate {
        view: std::cell::RefCell<Option<BrowserView>>,
        icons: Option<(Image, Image)>,
    }
    impl ViewDelegate {
        fn preferred_size(&self, _view: Option<&mut View>) -> Size {
            Size { width: 800, height: 600 }
        }
    }
    impl PanelDelegate {}
    impl WindowDelegate {
        fn window_runtime_style(&self) -> RuntimeStyle { RuntimeStyle::CHROME }
        // Rust delegates default boolean callbacks to false, unlike CEF's C++ delegates.
        fn with_standard_window_buttons(&self, _window: Option<&mut Window>) -> i32 { 1 }
        fn can_resize(&self, _window: Option<&mut Window>) -> i32 { 1 }
        fn can_maximize(&self, _window: Option<&mut Window>) -> i32 { 1 }
        fn can_minimize(&self, _window: Option<&mut Window>) -> i32 { 1 }
        fn on_window_created(&self, window: Option<&mut Window>) {
            if let (Some(window), Some(view)) = (window, self.view.borrow().as_ref()) {
                window.set_title(Some(&CefString::from(app_name().as_str())));
                if let Some((small, app)) = &self.icons {
                    set_window_icons(window, small.clone(), app.clone());
                }
                window.set_to_fill_layout();
                window.add_child_view(Some(&mut view.into()));
                window.show();
                view.request_focus();
            }
        }
        fn can_close(&self, _window: Option<&mut Window>) -> i32 {
            self.view.borrow().as_ref().and_then(|view| view.browser())
                .and_then(|browser| browser.host()).map_or(1, |host| host.try_close_browser())
        }
        fn on_window_destroyed(&self, _window: Option<&mut Window>) {
            self.view.borrow_mut().take();
        }
    }
}

wrap_display_handler! {
    struct ShallotDisplayHandler;
    impl DisplayHandler {
        fn on_title_change(&self, browser: Option<&mut Browser>, title: Option<&CefString>) {
            if let Some(window) = browser_view_get_for_browser(browser).and_then(|view| view.window()) {
                let fallback = CefString::from(app_name().as_str());
                window.set_title(Some(title.filter(|title| !title.to_string().is_empty()).unwrap_or(&fallback)));
            }
        }
    }
}


#[cfg(target_os = "linux")]
wrap_keyboard_handler! {
    struct ShallotKeyboardHandler;

    impl KeyboardHandler {
        fn on_pre_key_event(
            &self,
            browser: Option<&mut Browser>,
            event: Option<&KeyEvent>,
            _os_event: Option<&mut sys::XEvent>,
            _is_keyboard_shortcut: Option<&mut ::std::os::raw::c_int>,
        ) -> ::std::os::raw::c_int {
            let Some(event) = event else { return 0 };
            if event.type_ != KeyEventType::RAWKEYDOWN {
                return 0;
            }

            let key = event.windows_key_code;
            let alt = event.modifiers & EVENTFLAG_ALT != 0;

            let toggle = key == VK_F11 || (key == VK_RETURN && alt);
            let escape = key == VK_ESCAPE;

            if toggle || escape {
                if let Some(browser) = browser {
                    if let Some(window) = browser_view_get_for_browser(Some(browser)).and_then(|view| view.window()) {
                        window.set_fullscreen(if escape { 0 } else { 1 - window.is_fullscreen() });
                    }
                }
                if toggle {
                    return 1;
                }
            }

            0
        }
    }
}

wrap_client! {
    struct ShallotClient;

    impl Client {
        fn command_handler(&self) -> Option<CommandHandler> {
            Some(policy::AppCommandHandler::new())
        }

        fn context_menu_handler(&self) -> Option<ContextMenuHandler> {
            Some(policy::AppContextMenuHandler::new())
        }

        fn request_handler(&self) -> Option<RequestHandler> {
            Some(policy::AppRequestHandler::new())
        }

        fn display_handler(&self) -> Option<DisplayHandler> {
            Some(ShallotDisplayHandler::new())
        }

        fn life_span_handler(&self) -> Option<LifeSpanHandler> {
            Some(ShallotLifeSpanHandler::new())
        }

        fn keyboard_handler(&self) -> Option<KeyboardHandler> {
            #[cfg(target_os = "linux")]
            {
                Some(ShallotKeyboardHandler::new())
            }
            #[cfg(not(target_os = "linux"))]
            {
                None
            }
        }
    }
}

wrap_life_span_handler! {
    struct ShallotLifeSpanHandler;

    impl LifeSpanHandler {
        fn on_before_popup(
            &self,
            _browser: Option<&mut Browser>,
            _frame: Option<&mut Frame>,
            _popup_id: i32,
            _target_url: Option<&CefString>,
            _target_frame_name: Option<&CefString>,
            _target_disposition: WindowOpenDisposition,
            _user_gesture: i32,
            _popup_features: Option<&PopupFeatures>,
            _window_info: Option<&mut WindowInfo>,
            _client: Option<&mut Option<Client>>,
            _settings: Option<&mut BrowserSettings>,
            _extra_info: Option<&mut Option<DictionaryValue>>,
            _no_javascript_access: Option<&mut i32>,
        ) -> i32 { 1 }

        fn on_before_dev_tools_popup(
            &self,
            _browser: Option<&mut Browser>,
            _window_info: Option<&mut WindowInfo>,
            client: Option<&mut Option<Client>>,
            _settings: Option<&mut BrowserSettings>,
            _extra_info: Option<&mut Option<DictionaryValue>>,
            use_default_window: Option<&mut i32>,
        ) {
            // DevTools must not inherit app command filtering or app exit lifetime.
            if let Some(client) = client { *client = Some(DevToolsClient::new()); }
            if let Some(use_default_window) = use_default_window { *use_default_window = 1; }
        }

        fn on_before_close(&self, _browser: Option<&mut Browser>) {
            quit_message_loop();
        }
    }
}

wrap_client! {
    struct DevToolsClient;
    impl Client {
        fn life_span_handler(&self) -> Option<LifeSpanHandler> {
            Some(DevToolsLifeSpanHandler::new())
        }
    }
}

wrap_life_span_handler! {
    struct DevToolsLifeSpanHandler;
    impl LifeSpanHandler {
        fn on_before_close(&self, _browser: Option<&mut Browser>) {}
    }
}

#[cfg(target_os = "linux")]
fn find_cef_dir() -> Option<std::path::PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let exe_dir = exe.parent()?;
    let cef_dir = exe_dir.join("cef");
    if cef_dir.join("libcef.so").exists() {
        return Some(cef_dir);
    }
    if exe_dir.join("libcef.so").exists() {
        return Some(exe_dir.to_path_buf());
    }
    None
}

#[cfg(target_os = "linux")]
pub fn run() {
    let _ = api_hash(sys::CEF_API_VERSION_LAST, 0);

    let args = args::Args::new();
    let main_args = args.as_main_args();

    let mut app = ShallotApp::new();

    // Single-binary subprocess model: CEF re-execs this exe for renderer/GPU/etc. (browser_subprocess_path
    // below). Pass the app so each subprocess also runs on_before_command_line_processing — the
    // WebGPU/GPU switches must reach the GPU process, not just the browser, or its adapter stays
    // blocklisted. execute_process returns >= 0 in those subprocesses; the browser process gets -1.
    let ret = execute_process(Some(main_args), Some(&mut app), std::ptr::null_mut());
    if ret >= 0 {
        std::process::exit(ret);
    }

    let origin = start_asset_server();
    ORIGIN.set(origin).expect("origin already set");

    let exe = std::env::current_exe().expect("failed to get exe path");
    let exe_str = exe.to_string_lossy().to_string();

    let name = app_name();
    let cef_cache = cache_dir(&name).join("cef");
    let cef_cache_str = cef_cache.to_string_lossy().to_string();

    let mut settings = Settings {
        no_sandbox: 1,
        cache_path: CefString::from(cef_cache_str.as_str()),
        browser_subprocess_path: CefString::from(exe_str.as_str()),
        ..Default::default()
    };

    if let Some(cef_dir) = find_cef_dir() {
        let dir_str = cef_dir.to_string_lossy().to_string();
        settings.resources_dir_path = CefString::from(dir_str.as_str());
        let locales = cef_dir.join("locales");
        let locales_str = locales.to_string_lossy().to_string();
        settings.locales_dir_path = CefString::from(locales_str.as_str());
    }

    assert_eq!(
        initialize(
            Some(main_args),
            Some(&settings),
            Some(&mut app),
            std::ptr::null_mut()
        ),
        1,
        "CEF initialization failed"
    );

    run_message_loop();
    shutdown();
}

#[cfg(target_os = "windows")]
fn find_cef_dir() -> Option<std::path::PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let exe_dir = exe.parent()?;
    if exe_dir.join("libcef.dll").exists() {
        return Some(exe_dir.to_path_buf());
    }
    let cef_dir = exe_dir.join("cef");
    if cef_dir.join("libcef.dll").exists() {
        return Some(cef_dir);
    }
    None
}

// Windows portable CEF: same single-binary subprocess model as Linux (the exe re-execs itself for
// renderer/GPU/etc.), reading libcef.dll + the resource paks beside it. WebGPU runs on the default
// Dawn backend (D3D12), so only `enable-unsafe-webgpu` is needed (set in the command-line handler).
// No fullscreen key handler here yet — the system-webview build owns that path.
#[cfg(target_os = "windows")]
pub fn run() {
    let _ = api_hash(sys::CEF_API_VERSION_LAST, 0);

    let args = args::Args::new();
    let main_args = args.as_main_args();

    let mut app = ShallotApp::new();

    // Pass the app so each re-exec'd subprocess (GPU, renderer) also runs
    // on_before_command_line_processing — the WebGPU/GPU switches must reach the GPU process, not
    // just the browser, or its adapter stays blocklisted. Subprocesses return >= 0 and exit here.
    let ret = execute_process(Some(main_args), Some(&mut app), std::ptr::null_mut());
    if ret >= 0 {
        std::process::exit(ret);
    }

    let origin = start_asset_server();
    ORIGIN.set(origin).expect("origin already set");

    let exe = std::env::current_exe().expect("failed to get exe path");
    let exe_str = exe.to_string_lossy().to_string();

    let name = app_name();
    let cef_cache = cache_dir(&name).join("cef");
    let cef_cache_str = cef_cache.to_string_lossy().to_string();

    let mut settings = Settings {
        no_sandbox: 1,
        cache_path: CefString::from(cef_cache_str.as_str()),
        browser_subprocess_path: CefString::from(exe_str.as_str()),
        ..Default::default()
    };

    if let Some(cef_dir) = find_cef_dir() {
        let dir_str = cef_dir.to_string_lossy().to_string();
        settings.resources_dir_path = CefString::from(dir_str.as_str());
        let locales = cef_dir.join("locales");
        let locales_str = locales.to_string_lossy().to_string();
        settings.locales_dir_path = CefString::from(locales_str.as_str());
    }

    assert_eq!(
        initialize(
            Some(main_args),
            Some(&settings),
            Some(&mut app),
            std::ptr::null_mut()
        ),
        1,
        "CEF initialization failed"
    );

    run_message_loop();
    shutdown();
}

#[cfg(target_os = "macos")]
pub fn run() {
    use cef::library_loader::LibraryLoader;

    // Load the framework from Contents/Frameworks before any CEF call, then install the
    // CEF-compatible NSApplication. Both must happen before initialize.
    let loader = LibraryLoader::new(
        &std::env::current_exe().expect("failed to get exe path"),
        false,
    );
    assert!(loader.load(), "failed to load CEF framework");

    let _ = api_hash(sys::CEF_API_VERSION_LAST, 0);

    crate::mac::setup_application();

    let args = args::Args::new();
    let main_args = args.as_main_args();

    // Subprocesses run the separate helper bundles, so the main bundle executable is always the
    // browser process here (-1). Helpers never re-enter this code.
    let ret = execute_process(Some(main_args), None, std::ptr::null_mut());
    assert_eq!(ret, -1, "unexpected subprocess in main bundle executable");

    let origin = start_asset_server();
    ORIGIN.set(origin).expect("origin already set");

    let name = app_name();
    let cef_cache = cache_dir(&name).join("cef");
    let cef_cache_str = cef_cache.to_string_lossy().to_string();

    // framework_dir_path, main_bundle_path, and the helper subprocess paths are auto-discovered from
    // the loaded framework and the standard .app layout (Contents/Frameworks/<name> Helper (*).app).
    let settings = Settings {
        no_sandbox: 1,
        cache_path: CefString::from(cef_cache_str.as_str()),
        ..Default::default()
    };

    let mut app = ShallotApp::new();

    assert_eq!(
        initialize(
            Some(main_args),
            Some(&settings),
            Some(&mut app),
            std::ptr::null_mut()
        ),
        1,
        "CEF initialization failed"
    );

    run_message_loop();
    shutdown();
    drop(loader);
}
