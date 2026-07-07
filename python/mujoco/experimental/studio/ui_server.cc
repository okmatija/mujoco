// MuJoCo Link
//
// A Python module to stream a MuJoCo simulation and ImGui UI to the browser.
//
// This pybind11 module provides a lightweight Viewer class that:
// 1. Creates a headless ImGui context (no window, no renderer).
// 2. Connects as a netimgui client, streaming ImGui draw data to a remote
//    viewer.
// 3. Receives input events from the remote viewer and injects them into
//    the ImGui context.

#include <string.h>
#include <sys/time.h>

#include <chrono>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

#include "file/base/helpers.h"
#include "file/base/options.h"
#include "file/base/path.h"
#include "third_party/dear_imgui/imgui.h"
#include "third_party/implot/implot.h"
#include "third_party/mujoco/google/runfiles/runfiles.h"
#include "third_party/mujoco/include/mujoco.h"
#include "third_party/netimgui/Code/Client/NetImgui_Api.h"
#include "third_party/netimgui/google/logging.h"
#include "third_party/py/mujoco/experimental/studio/live_state.h"
#include "third_party/py/mujoco/structs.h"
#include "third_party/pybind11/include/pybind11/pybind11.h"
#include "third_party/pybind11/include/pybind11/stl.h"

namespace py = pybind11;

static std::vector<std::byte> LoadFontAsset(std::string_view filename) {
  constexpr char kFontPath[] =
      "third_party/mujoco/src/experimental/platform/data/";

  auto runfiles_dir = mujoco::GetRunfilesDir();
  if (!runfiles_dir.ok()) return {};

  std::string file_path =
      file::JoinPath(runfiles_dir.value(), "google3", kFontPath, filename);

  // Fonts are optional — if not found, ImGui will use its built-in font.
  std::string contents;
  if (!file::GetContents(file_path, &contents, file::Defaults()).ok()) {
    return {};
  }

  std::vector<std::byte> buffer(contents.size());
  memcpy(buffer.data(), contents.data(), contents.size());
  return buffer;
}

// Headless ImGui + netimgui viewer for MuJoCo Link.
//
// This class manages a headless ImGui context that streams its draw data
// via the netimgui protocol to a remote viewer. Unlike the pyviewer's Viewer
// class (viewer.cc), this does NOT create a window, initialize a renderer,
// or handle mouse/keyboard input directly. All rendering and input handling
// happens on the remote client (MuJoCo Live in the browser).
//
// The lifecycle follows the SampleNoBackend pattern:
//   Client_Startup()   — create context, load fonts, init NetImgui
//   Client_Connect()   — manage connection state (called each frame)
//   Client_Shutdown()  — release resources

// Initialize the Dear ImGui Context and the NetImgui library.
// Based on SampleNoBackend::Client_Startup() from
// //third_party/netimgui/Samples/SampleNoBackend/SampleNoBackend.cpp
static bool Client_Startup(ImGuiContext*& context) {
  IMGUI_CHECKVERSION();
  context = ImGui::CreateContext();
  ImGui::SetCurrentContext(context);
  ImPlot::CreateContext();

  ImGuiIO& io = ImGui::GetIO();
  io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
  io.BackendFlags |= ImGuiBackendFlags_HasGamepad;
  io.IniFilename = nullptr;
  io.ConfigDpiScaleFonts = true;
  io.ConfigDpiScaleViewports = true;
  io.DisplaySize = ImVec2(1400, 720);

  // Disable ImGui auto-repeat in headless mode since the browser handles key
  // repeat at the OS level, avoiding the frame amplification issue where many
  // frames with identical key-down states trigger unwanted repeats.
  io.KeyRepeatDelay = 9999.0f;

  // Initialize the main viewport's DPI scale. Without this, the headless
  // context leaves DpiScale at 0.0f (no platform backend sets it), which
  // triggers an assertion in ImGui::SetCurrentViewport() when
  // BeginMainMenuBar() is called.
  ImGuiViewport* main_vp = ImGui::GetMainViewport();
  if (main_vp) {
    main_vp->DpiScale = 1.0f;
  }

  ImGui::StyleColorsLight();

  auto main_font_data = LoadFontAsset("OpenSans-Regular.ttf");
  auto icon_font_data = LoadFontAsset("fontawesome-webfont.ttf");

  if (!main_font_data.empty()) {
    void* font_copy = ImGui::MemAlloc(main_font_data.size());
    memcpy(font_copy, main_font_data.data(), main_font_data.size());
    io.Fonts->AddFontFromMemoryTTF(font_copy, main_font_data.size(), 20.0f);
  }

  if (!icon_font_data.empty()) {
    ImFontConfig icon_cfg;
    icon_cfg.MergeMode = true;
    constexpr ImWchar icon_ranges[] = {0xf000, 0xf3ff, 0x000};
    void* icon_copy = ImGui::MemAlloc(icon_font_data.size());
    memcpy(icon_copy, icon_font_data.data(), icon_font_data.size());
    io.Fonts->AddFontFromMemoryTTF(icon_copy, icon_font_data.size(), 14.0f,
                                   &icon_cfg, icon_ranges);
  }

  // On ImGui 1.92+ NETIMGUI_IMGUI_TEXTURES_ENABLED is always set.
  // NetImgui::Startup() will set RendererHasTextures and the managed texture
  // system handles font atlas building and transfer automatically.
  // Do NOT call io.Fonts->Build() here — it conflicts with RendererHasTextures.

  if (!NetImgui::Startup()) {
    LOG(Error, "NetImgui::Startup() failed");
    return false;
  }

  return true;
}

// Release resources.
// Based on SampleNoBackend::Client_Shutdown() from
// //third_party/netimgui/Samples/SampleNoBackend/SampleNoBackend.cpp
static void Client_Shutdown(ImGuiContext*& context) {
  NetImgui::Shutdown();
  ImPlot::DestroyContext();
  if (context) {
    ImGui::DestroyContext(context);
    context = nullptr;
  }
}

// Manage connection to the netimgui proxy.
// Based on SampleNoBackend::Client_Connect() from
// //third_party/netimgui/Samples/SampleNoBackend/SampleNoBackend.cpp
static void Client_Connect(const char* title, int port) {
  bool connected = NetImgui::IsConnected();
  bool pending = NetImgui::IsConnectionPending();

  if (!connected && !pending) {
    static auto last_reconnect_time = std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();
    if (now - last_reconnect_time > std::chrono::seconds(1)) {
      last_reconnect_time = now;
      LOG(Info, "Retrying ConnectToApp...");
      NetImgui::ConnectToApp(title, "127.0.0.1", port);
    }
  }

  static bool last_connected = false;
  if (connected != last_connected) {
    LOG(Info, "Status change: Connected=%s", connected ? "true" : "false");
    last_connected = connected;
  }
}

class LinkServer {
 public:
  LinkServer(const std::string& title, int port) : title_(title), port_(port) {
    if (!Client_Startup(context_)) {
      return;
    }

    LOG(Info, "Calling ConnectToApp('%s', '127.0.0.1', %d)", title_.c_str(),
        port_);
    bool connect_result =
        NetImgui::ConnectToApp(title_.c_str(), "127.0.0.1", port_);
    LOG(Info, "ConnectToApp returned: %s", connect_result ? "true" : "false");
    LOG(Info, "IsConnected: %s, IsConnectionPending: %s",
        NetImgui::IsConnected() ? "true" : "false",
        NetImgui::IsConnectionPending() ? "true" : "false");
  }

  ~LinkServer() { Client_Shutdown(context_); }

  bool IsConnected() const { return NetImgui::IsConnected(); }

  bool NewFrame() {
    ImGui::SetCurrentContext(context_);
    static int frame_count = 0;
    frame_count++;

    // Block until connected and a frame is successfully started.
    // This ensures that every call to NewFrame() that returns true
    // guarantees an active ImGui frame, eliminating the need for
    // is_frame_active() in the Python API.
    while (true) {
      Client_Connect(title_.c_str(), port_);

      if (!NetImgui::IsConnected()) {
        // Not connected yet — wait and retry.
        is_drawing_remote_ = false;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        continue;
      }

      bool newFrameResult = NetImgui::NewFrame(false);
      is_drawing_remote_ = newFrameResult;
      if (!newFrameResult) {
        // Connected but NetImgui not ready for a draw — wait and retry.
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        continue;
      }

      break;
    }

    VLOG(1, "Frame %d: DrawingRemote=%s", frame_count,
         is_drawing_remote_ ? "Y" : "N");
    return true;
  }

  void EndFrame() {
    ImGui::SetCurrentContext(context_);
    static int end_frame_count = 0;
    end_frame_count++;
    if (!is_drawing_remote_) {
      // No frame was started — nothing to end.
      return;
    }
    VLOG(1, "EndFrame %d: sending remote draw data", end_frame_count);
    NetImgui::EndFrame();
  }

  // Serialize visualization state as raw bytes for the Live browser app.
  py::bytes GetVisState(const mujoco::python::MjvCameraWrapper& camera,
                        const mujoco::python::MjvOptionWrapper& vis_options,
                        const mujoco::python::MjModelWrapper& model,
                        const std::vector<uint8_t>& render_flags) {
    auto buffer = mujoco::link::SerializeLiveState(
        *camera.get(), *vis_options.get(), model.get()->opt, model.get()->vis,
        model.get()->stat, render_flags);
    return py::bytes(buffer.data(), buffer.size());
  }

  // Returns the fixed size of the visualization state in bytes.
  static int GetVisStateSize() { return mujoco::link::kLiveStateSize; }

  uint64_t GetContext() const { return reinterpret_cast<uint64_t>(context_); }

 private:
  std::string title_;
  int port_;
  ImGuiContext* context_ = nullptr;
  bool is_drawing_remote_ = false;
};

PYBIND11_MODULE(ui_server, m) {
  m.doc() = "MuJoCo Link Server: NetImgui UI server and state streaming";

  py::class_<LinkServer>(m, "LinkServer")
      .def(py::init<const std::string&, int>(), py::arg("title"),
           py::arg("port") = 8888)
      .def("is_connected", &LinkServer::IsConnected)
      .def("new_frame", &LinkServer::NewFrame)
      .def("end_frame", &LinkServer::EndFrame)
      .def("get_context", &LinkServer::GetContext)
      .def("get_vis_state", &LinkServer::GetVisState)
      .def_static("get_vis_state_size", &LinkServer::GetVisStateSize);
}
