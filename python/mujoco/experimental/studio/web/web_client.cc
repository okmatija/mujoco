// NetImgui WASM Viewer — renders remote ImGui draw data in a browser.
// Bridges the NetImgui server protocol to a WebGL/SDL2/Emscripten rendering
// context.
//
// This file is the Emscripten glue: app/scene state, the main loop, and the
// wiring between the two links (ui_link.h for the streamed Studio UI,
// state_link.h for simulation state) and the Filament renderer.

#include <SDL.h>
#include <SDL_opengl.h>
#include <stdint.h>
#include <stdio.h>

#include <cinttypes>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <imgui.h>
#include <implot.h>
#include <mujoco/mujoco.h>
#include "NetImgui_Api.h"
#include "experimental/platform/hal/renderer.h"
#include "experimental/platform/hal/window.h"
#include "experimental/platform/sim/model_holder.h"
#include "experimental/platform/ux/interaction.h"
#include "google/logging.h"
#include "ui_link.h"
#include "render_state.h"
#include "state_link.h"

#include <emscripten.h>
#include <emscripten/fetch.h>

#if !defined(__EMSCRIPTEN__)
#error "web_client.cc is only supported for Emscripten builds"
#endif

// No-op wrapper for glGetError(), installed via -Wl,--wrap=glGetError.
// Filament's GL backend calls glGetError() hundreds of times per frame
// through GLUtils::checkGLError/assertGLError. In WebGL each call forces a
// synchronous GPU pipeline flush across the JS-WASM bridge (~0.2ms each).
// This stub eliminates that cost entirely.
extern "C" GLenum __wrap_glGetError(void) { return GL_NO_ERROR; }

using mujoco::studio::StateLink;
using mujoco::studio::StatePayloadView;
using mujoco::studio::UiLink;

struct AppState {
  std::unique_ptr<mujoco::platform::Window> window;
  std::unique_ptr<mujoco::platform::ModelHolder> model_holder;
  mujoco::platform::Renderer* renderer = nullptr;
  mjvPerturb perturb;
  mjvCamera camera;
  mjvOption vis_options;
  int camera_idx = 0;

  // True when another browser holds the driver slot: this page renders the
  // scene from the state broadcast but has no UI stream or input until the
  // user takes control.
  bool spectator = false;
  int ui_reject_count = 0;
  // Connected browser count, from the server's roster updates.
  int session_viewers = 0;

  // Backend state received from the Python simulation via WebSocket.
  std::vector<mjtNum> backend_state;
  int backend_state_sig = 0;
  bool backend_state_dirty = false;

  // User-injected geoms (Viewer.extra_geoms) received with the state payload.
  std::vector<mjvGeom> extra_geoms;
};
AppState g_app;

// Byte-rate telemetry shown in the Link window.
struct Telemetry {
  double last_rate_time = 0;
  uint64_t gui_bytes_per_sec = 0;
  uint64_t sim_bytes_per_sec = 0;
  bool expanded = false;
  bool disconnected_notice_logged = false;
};
Telemetry g_telemetry;

// How long the state stream (~60Hz while the Python side is alive) may go
// silent before the "server not reachable" notice appears. A model-change
// restart resumes traffic in about a second and should not flash the notice.
constexpr double kServerSilenceNoticeSec = 3.0;

// WebSocket base URL matching the page origin, e.g. "ws://host:8080" or
// "wss://tunnel.example.com" behind an HTTPS tunnel. All viewer WebSockets
// are served as paths (/ui, /state) on the same host and port as the page
// itself, so exposing or tunneling that single port exposes the whole
// viewer.
std::string GetWsBaseUrl() {
  char* base = emscripten_run_script_string(
      "(window.location.protocol === 'https:' ? 'wss://' : 'ws://') + "
      "window.location.host");
  std::string url = base != nullptr ? base : "";
  if (url == "ws://" || url == "wss://" || url.empty()) {
    // No host in the page URL (e.g. file://) — assume a local server.
    return "ws://localhost:8080";
  }
  return url;
}

// Stable per-page id sent with every WebSocket connect (?sid=...), letting
// the server tie this page's /ui and /state connections together.
// (crypto.randomUUID would need a secure context, which plain http on a LAN
// is not, hence the homegrown id; uniqueness is all that matters here.)
std::string GetSessionId() {
  static const std::string sid = emscripten_run_script_string(
      "Module.sessionId = Module.sessionId || "
      "(Date.now().toString(36) + Math.random().toString(36).slice(2))");
  return sid;
}

std::string WsUrl(const char* path) {
  return GetWsBaseUrl() + path + "?sid=" + GetSessionId();
}

// Returns true if the Filament rendering context is initialized and ready for
// GPU texture uploads. The Renderer object is created in main() and is always
// non-null, but the Filament context is only initialized when Renderer::Init()
// is called from SetupScene after the async model fetch completes.
bool IsFilamentReady() {
  return g_app.renderer && g_app.model_holder && g_app.model_holder->ok();
}

// Applies a parsed state payload to the app. Wired as g_state_link.on_payload;
// runs only after the identity/reload policy has accepted the payload.
void ApplyStatePayload(const StatePayloadView& view) {
  mjModel* model = g_app.model_holder->model();

  // Physics state. Guard against a size mismatch (e.g. a stale packet from
  // before a model change) — mj_setState with a wrong-sized vector would
  // corrupt mjData.
  if (view.physics != nullptr) {
    const size_t expected_bytes =
        mj_stateSize(model, view.physics_spec) * sizeof(mjtNum);
    if (view.physics_bytes == expected_bytes) {
      g_app.backend_state.resize(view.physics_bytes / sizeof(mjtNum));
      memcpy(g_app.backend_state.data(), view.physics, view.physics_bytes);
      g_app.backend_state_sig = view.physics_spec;
      g_app.backend_state_dirty = true;
    } else {
      LOG(Warning, "Physics state size mismatch (%zu != %zu); dropping",
          view.physics_bytes, expected_bytes);
    }
  }

  // Render state. The headless viewer owns the camera and perturbation state
  // (all input is forwarded to it and handled by the same code as the native
  // viewer); the browser just renders them.
  if (view.render_state != nullptr) {
    const char* vis_ptr = view.render_state;

    memcpy(&g_app.camera, vis_ptr, sizeof(mjvCamera));
    vis_ptr += sizeof(mjvCamera);

    memcpy(&g_app.perturb, vis_ptr, sizeof(mjvPerturb));
    vis_ptr += sizeof(mjvPerturb);

    memcpy(&g_app.vis_options, vis_ptr, sizeof(mjvOption));
    vis_ptr += sizeof(mjvOption);

    memcpy(&model->opt, vis_ptr, sizeof(mjOption));
    vis_ptr += sizeof(mjOption);

    memcpy(&model->vis, vis_ptr, sizeof(mjVisual));
    vis_ptr += sizeof(mjVisual);

    memcpy(&model->stat, vis_ptr, sizeof(mjStatistic));
    vis_ptr += sizeof(mjStatistic);

    // Apply render flags to the renderer's scene if available.
    if (g_app.renderer) {
      memcpy(g_app.renderer->GetRenderFlags(), vis_ptr, mjNRNDFLAG);
    }
  }

  // Extra geoms (Viewer.extra_geoms on the Python side). The payload data is
  // not guaranteed to be aligned; memcpy into the vector.
  g_app.extra_geoms.resize(view.extra_geom_count);
  if (view.extra_geom_count > 0) {
    memcpy(g_app.extra_geoms.data(), view.extra_geoms,
           view.extra_geom_count * sizeof(mjvGeom));
  }
}

// The two links to the Python side. Their connection-scoped state lives
// inside the classes; see ui_link.h and state_link.h.
UiLink g_ui_link(
    [](uintptr_t current, const std::byte* rgba, uint32_t width,
       uint32_t height) -> uintptr_t {
      return g_app.renderer->UploadImage(current, rgba, width, height,
                                         rgba ? 4 : 0);
    },
    IsFilamentReady);
StateLink g_state_link(
    [] { return g_app.model_holder && g_app.model_holder->ok(); },
    ApplyStatePayload,
    [](const char* text) {
      int viewers = 0;
      if (sscanf(text, "viewers=%d", &viewers) == 1) {
        if (viewers != g_app.session_viewers) {
          LOG(Info, "Session roster: %s", text);
        }
        g_app.session_viewers = viewers;
      }
    });

void BuildBrowserGui() {
  if (const int close_code = g_state_link.TerminalCloseCode()) {
    const ImGuiIO& io = ImGui::GetIO();
    ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x * 0.5f, 48.0f),
                            ImGuiCond_Always, ImVec2(0.5f, 0.0f));
    ImGui::Begin("##disconnected_by_server", nullptr,
                 ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                     ImGuiWindowFlags_NoMove |
                     ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f),
                       close_code == 4002
                           ? "Session is full: too many viewers connected."
                           : "Disconnected by the viewer.");
    ImGui::TextUnformatted("Reload this page to try again.");
    ImGui::End();
  }

  const double now = ImGui::GetTime();
  if (now - g_telemetry.last_rate_time >= 1.0) {
    g_telemetry.gui_bytes_per_sec = g_ui_link.ConsumeByteCount();
    g_telemetry.sim_bytes_per_sec = g_state_link.ConsumeByteCount();
    g_telemetry.last_rate_time = now;
  }

  // "Server not reachable" notice, keyed on state-stream staleness rather
  // than socket state: a killed, suspended (Ctrl+Z), or unreachable server
  // all go silent, but only a killed one closes its sockets. Clears itself
  // when traffic resumes. Suppressed before the first payload (initial
  // load), after a deliberate server-side close (its own notice), and
  // during model-swap reloads.
  const double last_msg = g_state_link.LastMessageTime();
  const bool link_stale =
      last_msg > 0 &&
      emscripten_get_now() / 1000.0 - last_msg > kServerSilenceNoticeSec &&
      g_state_link.TerminalCloseCode() == 0 && !g_state_link.ReloadPending();
  if (!link_stale) {
    g_telemetry.disconnected_notice_logged = false;
  } else {
    if (!g_telemetry.disconnected_notice_logged) {
      LOG(Info, "Showing server-not-reachable notice");
      g_telemetry.disconnected_notice_logged = true;
    }
    const ImGuiIO& io = ImGui::GetIO();
    ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x * 0.5f, 48.0f),
                            ImGuiCond_Always, ImVec2(0.5f, 0.0f));
    ImGui::Begin("##disconnected", nullptr,
                 ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                     ImGuiWindowFlags_NoMove |
                     ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f),
                       "Viewer server is not reachable.");
    ImGui::TextUnformatted(
        "The Python script may have stopped. This page reconnects\n"
        "automatically if the viewer comes back.");
    ImGui::End();
  }

  // Default to the top center of the canvas; draggable afterwards.
  ImGui::SetNextWindowPos(ImVec2(ImGui::GetIO().DisplaySize.x * 0.5f, 8.0f),
                          ImGuiCond_FirstUseEver, ImVec2(0.5f, 0.0f));

  if (!g_telemetry.expanded) {
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(6.0f, 6.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowMinSize, ImVec2(16.0f, 16.0f));

    ImGui::Begin("Link", nullptr,
                 ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                     ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::Text("Link");
    if (ImGui::IsItemHovered()) {
      if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
        g_telemetry.expanded = true;
      }
      ImGui::SetTooltip("Double-click toggle telemetry");
    }
    ImGui::End();

    ImGui::PopStyleVar(2);
  } else {
    bool p_open = true;
    bool should_collapse = false;

    ImGui::Begin(
        "Link", &p_open,
        ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoCollapse);
    if (ImGui::IsWindowHovered() &&
        ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
      should_collapse = true;
    }

    if (g_app.spectator) {
      if (g_app.session_viewers > 1) {
        ImGui::Text("Spectating: %d viewers connected.",
                    g_app.session_viewers);
      } else {
        ImGui::Text("Spectating: another browser is driving.");
      }
      ImGui::Text("Data Rate (Sim): %" PRIu64 " KiB/s",
                  static_cast<uint64_t>(g_telemetry.sim_bytes_per_sec / 1024));
      if (ImGui::Button("Take control")) {
        // One attempt; if the driver is still there this page goes straight
        // back to spectating.
        g_app.spectator = false;
        g_app.ui_reject_count = 2;
        g_ui_link.Connect(WsUrl("/ui"));
      }
    } else {
      ImGui::Text("Spectators: %d",
                  g_app.session_viewers > 1 ? g_app.session_viewers - 1 : 0);
      ImGui::Text("Connection: %s", g_ui_link.StatusString());
      ImGui::Text("Remote Frame: %s",
                  g_ui_link.RemoteDrawData() ? "Received" : "None");
      ImGui::Text("Data Rate (GUI): %" PRIu64 " KiB/s",
                  static_cast<uint64_t>(g_telemetry.gui_bytes_per_sec / 1024));
      ImGui::Text("Data Rate (Sim): %" PRIu64 " KiB/s",
                  static_cast<uint64_t>(g_telemetry.sim_bytes_per_sec / 1024));
      bool use_compression = g_ui_link.UseCompression();
      if (ImGui::Checkbox("Use Compression", &use_compression)) {
        g_ui_link.SetUseCompression(use_compression);
        NetImgui::SetCompressionMode(use_compression ? NetImgui::kForceEnable
                                                     : NetImgui::kForceDisable);
      }
      if (!g_ui_link.RemoteDrawData()) {
        ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f),
                           "Waiting for Draw Data...");
      }
    }
    ImGui::End();

    if (!p_open || should_collapse) {
      g_telemetry.expanded = false;
    }
  }
}

//=================================================================================================
// Main loop — called once per frame.
//=================================================================================================
void MainLoopImpl();

// An exception escaping the RAF callback kills the main loop silently (the
// canvas freezes on the last rendered frame and input capture stops, with
// only an opaque "Uncaught <ptr>" in the console). Catch, log, and stop
// explicitly instead.
void MainLoop() {
  try {
    MainLoopImpl();
  } catch (const std::exception& e) {
    LOG(Error, "FATAL: uncaught exception in MainLoop: %s", e.what());
    emscripten_cancel_main_loop();
  } catch (...) {
    LOG(Error, "FATAL: uncaught non-std exception in MainLoop");
    emscripten_cancel_main_loop();
  }
}

void MainLoopImpl() {
  static int sMainFrameCount = 0;
  sMainFrameCount++;

  // Reconnect the state WebSocket if it dropped — e.g. the Python side
  // restarted its servers after a model change. Receiving a payload with a
  // different model identity then triggers a page reload (StateLink).
  static int sLastStateWsRetryFrame = 0;
  if (!g_state_link.HasSocket() && !g_state_link.ReloadPending() &&
      g_state_link.TerminalCloseCode() == 0 && g_app.model_holder &&
      g_app.model_holder->ok() &&
      sMainFrameCount - sLastStateWsRetryFrame > 60) {
    sLastStateWsRetryFrame = sMainFrameCount;
    LOG(Info, "State WebSocket down; reconnecting...");
    g_state_link.Connect(WsUrl("/state"));
  }

  // Reconnect the NetImgui UI socket too — the proxy tears the bridge down
  // whenever its headless-side TCP connection cycles (server restart). A
  // close code 4001 means another browser holds the driver slot: after a
  // few retries (a page reload of the driver races its own slot briefly)
  // this page settles into spectating.
  static int sLastUiWsRetryFrame = 0;
  if (g_ui_link.HasSocket() && !g_app.spectator &&
      !g_state_link.ReloadPending() &&
      g_state_link.TerminalCloseCode() == 0 &&
      sMainFrameCount - sLastUiWsRetryFrame > 60) {
    const UiLink::ReadyState uiState = g_ui_link.ConnectionState();
    if (uiState == UiLink::ReadyState::kOpen) {
      g_app.ui_reject_count = 0;
    } else if (uiState == UiLink::ReadyState::kClosed ||
               uiState == UiLink::ReadyState::kError) {
      sLastUiWsRetryFrame = sMainFrameCount;
      if (g_ui_link.CloseCode() == 4001 && ++g_app.ui_reject_count >= 3) {
        LOG(Info, "Driver slot taken; spectating");
        g_app.spectator = true;
      } else {
        LOG(Info, "UI WebSocket closed; reconnecting...");
        g_ui_link.Connect(WsUrl("/ui"));
      }
    }
  }

  // Process incoming UI data BEFORE the ImGui frame: textures and draw
  // frames must be ready before we start the local ImGui frame and render.
  if (g_app.window) {
    g_ui_link.SetMaxClip(static_cast<float>(g_app.window->GetWidth()),
                         static_cast<float>(g_app.window->GetHeight()));
  }
  g_ui_link.ReceiveAndProcessCommands(sMainFrameCount);

  // Event loop and ImGui NewFrame via window abstraction.
  mujoco::platform::Window::Status status = g_app.window->NewFrame();
  if (status == mujoco::platform::Window::kQuitting) {
    // NewFrame() started an ImGui frame — end it before bailing out.
    ImGui::EndFrame();
    emscripten_cancel_main_loop();
    return;
  }

  // All scene interaction (camera orbit/zoom, perturbation, picking) is
  // handled by the headless viewer: CaptureAndSendInput() forwards this
  // frame's input over NetImgui, the headless ViewerApp runs the same event
  // handlers as the native viewer, and the resulting camera/perturb state
  // streams back over the state WebSocket (see ApplyStatePayload).
  g_ui_link.CaptureAndSendInput();

  BuildBrowserGui();

  // Finalize the local ImGui draw data. ImguiBridge::Update() (called inside
  // Filament's Render) will call ImGui::Render() again, but that is a no-op
  // once the frame has already been rendered.
  ImGui::Render();

  // Inject remote draw lists into the local ImDrawData so that Filament's
  // ImguiBridge renders them alongside the local UI. Remote lists are inserted
  // first (background) and local lists are re-added after (foreground), so the
  // local UI always renders on top of remote content.
  // ImGui::GetDrawData() is only valid after ImGui::Render() and until the next
  // call to ImGui::NewFrame().
  mujoco::studio::NetImguiImDrawData* remoteDrawData =
      g_ui_link.RemoteDrawData();
  if (remoteDrawData && remoteDrawData->Valid) {
    ImDrawData* localDrawData = ImGui::GetDrawData();
    if (localDrawData) {
      // Save local draw lists.
      ImVector<ImDrawList*> localLists;
      localLists.reserve(localDrawData->CmdListsCount);
      for (int i = 0; i < localDrawData->CmdListsCount; ++i) {
        localLists.push_back(localDrawData->CmdLists[i]);
      }

      // Clear and rebuild: remote first, then local.
      localDrawData->CmdLists.resize(0);
      localDrawData->CmdListsCount = 0;
      localDrawData->TotalVtxCount = 0;
      localDrawData->TotalIdxCount = 0;

      // Remote draw lists (background).
      for (int i = 0; i < remoteDrawData->CmdListsCount; ++i) {
        localDrawData->AddDrawList(remoteDrawData->CmdLists[i]);
      }

      // Local draw lists (foreground).
      for (int i = 0; i < localLists.Size; ++i) {
        localDrawData->AddDrawList(localLists[i]);
      }
    }
  }

  // Filament render — scene + all ImGui (local + remote). Filament manages
  // its own clear, framebuffer, and GL state.
  if (g_app.renderer && g_app.model_holder && g_app.model_holder->ok()) {
    // Apply backend state if available.
    if (g_app.backend_state_dirty) {
      mjModel* model = g_app.model_holder->model();
      mjData* data = g_app.model_holder->data();
      mj_setState(model, data, g_app.backend_state.data(),
                  g_app.backend_state_sig);
      mj_forward(model, data);
      g_app.backend_state_dirty = false;
    }

    int width =
        static_cast<int>(g_app.window->GetWidth() * g_app.window->GetScale());
    int height =
        static_cast<int>(g_app.window->GetHeight() * g_app.window->GetScale());
    if (width > 0 && height > 0) {
      g_app.renderer->Render(g_app.model_holder->model(),
                             g_app.model_holder->data(), &g_app.perturb,
                             &g_app.camera, &g_app.vis_options, width, height,
                             /*pixels=*/{}, std::span(g_app.extra_geoms));
    }
  }

  g_app.window->Present();
}

void SetupScene(const mjModel* m) {
  g_app.renderer->Init(m);

  // Upload any textures (e.g. the font atlas) that were buffered before the
  // Filament context was available.
  g_ui_link.FlushPendingTextures();

  mjv_defaultPerturb(&g_app.perturb);
  mjv_defaultCamera(&g_app.camera);
  mjv_defaultOption(&g_app.vis_options);

  const int model_cam = m->vis.global.cameraid;
  if (model_cam >= 0 && model_cam < m->ncam) {
    g_app.camera_idx = mujoco::platform::SetCamera(m, &g_app.camera, model_cam);
  } else {
    mjv_defaultFreeCamera(m, &g_app.camera);
  }
}

void OnFetchSuccess(emscripten_fetch_t* fetch) {
  LOG(Info, "Fetched model.mjb, size: %llu", fetch->numBytes);
  g_app.model_holder = mujoco::platform::ModelHolder::FromBuffer(
      std::span<const std::byte>(
          reinterpret_cast<const std::byte*>(fetch->data),
          static_cast<size_t>(fetch->numBytes)),
      "application/mjb", "model.mjb");
  if (g_app.model_holder && g_app.model_holder->ok()) {
    LOG(Info, "Model loaded successfully!");
    mj_forward(g_app.model_holder->model(), g_app.model_holder->data());
    if (g_app.window) {
      SetupScene(g_app.model_holder->model());
    }
    // Connect the state WebSocket to receive simulation state from Python.
    g_state_link.Connect(WsUrl("/state"));
  } else {
    LOG(Error, "Failed to load model: %s",
        g_app.model_holder ? g_app.model_holder->error().data()
                           : "Unknown error");
  }
  emscripten_fetch_close(fetch);
}

void OnFetchError(emscripten_fetch_t* fetch) {
  LOG(Error, "Failed to fetch model.mjb, status: %d", fetch->status);
  emscripten_fetch_close(fetch);
}

// Loads an asset from the Emscripten virtual filesystem. The Filament assets
// (materials, IBL) are bundled into web_client.data at link time via
// --preload-file and mounted at /assets.
static std::vector<std::byte> LoadAsset(std::string_view path) {
  std::string_view subpath = path.substr(path.find(':') + 1);
  std::string file_path = std::string("/assets/") + std::string(subpath);

  std::ifstream file(file_path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    return {};
  }
  auto file_size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<std::byte> buffer(file_size);
  if (!file.read(reinterpret_cast<char*>(buffer.data()), file_size)) {
    return {};
  }
  return buffer;
}

// Holds loaded resource data for the MuJoCo resource provider.
struct ResourceData {
  std::vector<std::byte> bytes;
};

// Registers a resource provider so that "filament:" asset requests from the
// renderer resolve to the preloaded /assets files.
// TODO(matijak): Near-identical copies of LoadAsset/ResourceData/registration
// live in native_viewer.cc, launcher.cc, and emscripten.cc (differing only in
// asset root and prefixes). Extract a shared helper into mujoco::platform.
static void RegisterAssetProviders() {
  mjpResourceProvider resource_provider;
  mjp_defaultResourceProvider(&resource_provider);

  resource_provider.open = [](mjResource* resource) {
    auto* data = new ResourceData();
    data->bytes = LoadAsset(resource->name);
    if (data->bytes.empty()) {
      delete data;
      return 0;
    }
    resource->data = data;
    return static_cast<int>(data->bytes.size());
  };
  resource_provider.read = [](mjResource* resource, const void** buffer) {
    auto* data = static_cast<ResourceData*>(resource->data);
    *buffer = data->bytes.data();
    return static_cast<int>(data->bytes.size());
  };
  resource_provider.close = [](mjResource* resource) {
    delete static_cast<ResourceData*>(resource->data);
    resource->data = nullptr;
  };
  resource_provider.prefix = "filament";
  mjp_registerResourceProvider(&resource_provider);
}

int main(int argc, char** argv) {
  RegisterAssetProviders();

  mujoco::platform::Window::Config config;
  config.gfx_mode = mujoco::platform::GraphicsMode::FilamentWebGl;
  config.load_fonts = false;

  g_app.window = std::make_unique<mujoco::platform::Window>("MuJoCo Web Viewer",
                                                            1400, 720, config);
  ImPlot::CreateContext();  // Needed if the server app uses ImPlot.

  g_app.renderer = new mujoco::platform::Renderer(
      g_app.window->GetNativeWindowHandle(), config.gfx_mode);

  NetImgui::Internal::Network::Startup();
  SDL_StartTextInput();

  // The UI stream is a path on the page's own host and port.
  g_ui_link.Connect(WsUrl("/ui"));

  emscripten_fetch_attr_t attr;
  emscripten_fetch_attr_init(&attr);
  strcpy(attr.requestMethod, "GET");
  attr.attributes = EMSCRIPTEN_FETCH_LOAD_TO_MEMORY;
  attr.onsuccess = OnFetchSuccess;
  attr.onerror = OnFetchError;
  // Relative URL — the browser resolves this against the page origin,
  // so it works regardless of hostname, protocol, or port.
  emscripten_fetch(&attr, "/model.mjb");

  emscripten_set_main_loop(MainLoop, 0, 1);

  // Cleanup (reached if emscripten loop exits).
  g_ui_link.Shutdown();
  NetImgui::Internal::Network::Shutdown();

  g_app.window.reset();

  return 0;
}
