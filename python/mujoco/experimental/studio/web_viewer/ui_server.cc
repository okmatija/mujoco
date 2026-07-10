// Copyright 2026 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// MuJoCo web viewer UI server.
//
// A Python module to stream a MuJoCo simulation and ImGui UI to the browser.
//
// This pybind11 module provides a lightweight UiServer class that:
// 1. Creates a headless ImGui context (no window, no renderer).
// 2. Connects as a netimgui client, streaming ImGui draw data to a remote
//    viewer.
// 3. Receives input events from the remote viewer and injects them into
//    the ImGui context.

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <imgui.h>
#include <implot.h>
#include <mujoco/mujoco.h>
#include "NetImgui_Api.h"
#include "google/logging.h"
#include "render_state.h"
#include "structs.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Loads a font file from the given assets directory. Returns an empty vector
// if the file cannot be read; fonts are optional and ImGui falls back to its
// built-in font.
static std::vector<std::byte> LoadFontAsset(const std::string& assets_dir,
                                            std::string_view filename) {
  if (assets_dir.empty()) return {};
  const std::string file_path =
      (std::filesystem::path(assets_dir) / filename).string();

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

// Headless ImGui + netimgui viewer for the MuJoCo web viewer.
//
// This class manages a headless ImGui context that streams its draw data
// via the netimgui protocol to a remote viewer. Unlike the NativeViewer's
// Viewer class (native_viewer.cc), this does NOT create a window, initialize
// a renderer, or handle mouse/keyboard input directly. All rendering and
// input handling happens on the remote client (web_client.cc in the browser).
//
// The lifecycle follows the SampleNoBackend pattern:
//   Client_Startup()   — create context, load fonts, init NetImgui
//   Client_Connect()   — manage connection state (called each frame)
//   Client_Shutdown()  — release resources

// Initialize the Dear ImGui Context and the NetImgui library.
// Based on SampleNoBackend::Client_Startup() from
// netimgui/Code/Sample/SampleNoBackend/SampleNoBackend.cpp
static bool Client_Startup(ImGuiContext*& context,
                           const std::string& assets_dir) {
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

  auto main_font_data =
      LoadFontAsset(assets_dir, "AtkinsonHyperlegibleNext[wght].ttf");
  auto icon_font_data = LoadFontAsset(assets_dir, "fontawesome-webfont.ttf");

  // Font sizes match the native viewer (platform/hal/window.cc) so the UI
  // has the same proportions in the browser as in native Studio.
  if (!main_font_data.empty()) {
    void* font_copy = ImGui::MemAlloc(main_font_data.size());
    memcpy(font_copy, main_font_data.data(), main_font_data.size());
    io.Fonts->AddFontFromMemoryTTF(font_copy, main_font_data.size(), 16.0f);
  }

  if (!icon_font_data.empty()) {
    ImFontConfig icon_cfg;
    icon_cfg.MergeMode = true;
    constexpr ImWchar icon_ranges[] = {0xf000, 0xf3ff, 0x000};
    void* icon_copy = ImGui::MemAlloc(icon_font_data.size());
    memcpy(icon_copy, icon_font_data.data(), icon_font_data.size());
    io.Fonts->AddFontFromMemoryTTF(icon_copy, icon_font_data.size(), 13.0f,
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
// netimgui/Code/Sample/SampleNoBackend/SampleNoBackend.cpp
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
// netimgui/Code/Sample/SampleNoBackend/SampleNoBackend.cpp
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

class UiServer {
 public:
  UiServer(const std::string& title, int port, const std::string& assets_dir)
      : title_(title), port_(port) {
    if (!Client_Startup(context_, assets_dir)) {
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

  ~UiServer() { Client_Shutdown(context_); }

  bool IsConnected() const { return NetImgui::IsConnected(); }

  bool NewFrame() {
    py::gil_scoped_release no_gil;
    ImGui::SetCurrentContext(context_);
    static int frame_count = 0;
    frame_count++;

    // Block until connected and a frame is successfully started.
    // This ensures that every call to NewFrame() that returns true
    // guarantees an active ImGui frame, eliminating the need for
    // is_frame_active() in the Python API.
    auto last_signal_check = std::chrono::steady_clock::now();
    while (true) {
      // While blocked (e.g. no browser connected), periodically check for
      // Python signals so Ctrl+C interrupts the wait instead of hanging.
      auto now = std::chrono::steady_clock::now();
      if (now - last_signal_check > std::chrono::milliseconds(200)) {
        last_signal_check = now;
        py::gil_scoped_acquire acquire;
        if (PyErr_CheckSignals() != 0) {
          throw py::error_already_set();
        }
      }

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
    py::gil_scoped_release no_gil;
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

  // Serialize the render state (see render_state.h) as raw bytes for the
  // browser app.
  py::bytes GetRenderState(const mujoco::python::MjvCameraWrapper& camera,
                           const mujoco::python::MjvPerturbWrapper& perturb,
                           const mujoco::python::MjvOptionWrapper& vis_options,
                           const mujoco::python::MjModelWrapper& model,
                           const std::vector<uint8_t>& render_flags) {
    auto buffer = mujoco::studio::SerializeRenderState(
        *camera.get(), *perturb.get(), *vis_options.get(), model.get()->opt,
        model.get()->vis, model.get()->stat, render_flags);
    return py::bytes(buffer.data(), buffer.size());
  }

  // Returns the fixed size of the render state in bytes.
  static int GetRenderStateSize() { return mujoco::studio::kRenderStateSize; }

  intptr_t GetContext() const { return reinterpret_cast<intptr_t>(context_); }

 private:
  std::string title_;
  int port_;
  ImGuiContext* context_ = nullptr;
  bool is_drawing_remote_ = false;
};

PYBIND11_MODULE(ui_server, m, pybind11::mod_gil_not_used()) {
  py::module_::import("mujoco._structs");
  m.doc() = "MuJoCo web viewer UI server: NetImgui client and state streaming";

  py::class_<UiServer>(m, "UiServer")
      .def(py::init<const std::string&, int, const std::string&>(),
           py::arg("title"), py::arg("port") = 8888,
           py::arg("assets_dir") = "")
      .def("is_connected", &UiServer::IsConnected)
      .def("new_frame", &UiServer::NewFrame)
      .def("end_frame", &UiServer::EndFrame)
      .def("get_context", &UiServer::GetContext)
      .def("get_render_state", &UiServer::GetRenderState)
      .def_static("get_render_state_size", &UiServer::GetRenderStateSize);
}
