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

#include <imgui.h>
#include <implot.h>
#include <mujoco/mujoco.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <atomic>
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

#include "NetImgui_Api.h"
#include "google/logging.h"
#include "state_payload.h"
#include "structs.h"

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
  const std::streamsize file_size = file.tellg();
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

  const std::vector<std::byte> main_font_data =
      LoadFontAsset(assets_dir, "AtkinsonHyperlegibleNext[wght].ttf");
  const std::vector<std::byte> icon_font_data =
      LoadFontAsset(assets_dir, "fontawesome-webfont.ttf");

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
    static std::chrono::steady_clock::time_point last_reconnect_time =
        std::chrono::steady_clock::now();
    const std::chrono::steady_clock::time_point now =
        std::chrono::steady_clock::now();
    if (now - last_reconnect_time > std::chrono::seconds(1)) {
      last_reconnect_time = now;
      VLOG(1, "Retrying ConnectToApp...");
      NetImgui::ConnectToApp(title, "127.0.0.1", port);
    }
  }

  static bool last_connected = false;
  if (connected != last_connected) {
    VLOG(1, "Status change: Connected=%s", connected ? "true" : "false");
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

    VLOG(1, "Calling ConnectToApp('%s', '127.0.0.1', %d)", title_.c_str(),
         port_);
    bool connect_result =
        NetImgui::ConnectToApp(title_.c_str(), "127.0.0.1", port_);
    VLOG(1, "ConnectToApp returned: %s", connect_result ? "true" : "false");
    VLOG(1, "IsConnected: %s, IsConnectionPending: %s",
         NetImgui::IsConnected() ? "true" : "false",
         NetImgui::IsConnectionPending() ? "true" : "false");
  }

  ~UiServer() { Client_Shutdown(context_); }

  // Makes a NewFrame() call return false instead of blocking. Thread safe.
  void RequestClose() { close_requested_ = true; }

  bool NewFrame() {
    py::gil_scoped_release no_gil;
    ImGui::SetCurrentContext(context_);
    static int frame_count = 0;
    frame_count++;

    // Block until connected and a frame is successfully started.
    // This ensures that every call to NewFrame() that returns true
    // guarantees an active ImGui frame, eliminating the need for
    // is_frame_active() in the Python API.
    std::chrono::steady_clock::time_point last_signal_check =
        std::chrono::steady_clock::now();
    while (true) {
      if (close_requested_) {
        is_drawing_remote_ = false;
        return false;
      }
      // While blocked (e.g. no browser connected), periodically check for
      // Python signals so Ctrl+C interrupts the wait instead of hanging.
      const std::chrono::steady_clock::time_point now =
          std::chrono::steady_clock::now();
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

  // Serialize the complete state WebSocket payload (see state_payload.h):
  // physics state, render state and extra geoms as tagged blocks.
  py::bytes SerializeStatePayload(
      uint32_t model_crc32, int physics_spec, const py::bytes& physics_state,
      const mujoco::python::MjvCameraWrapper& camera,
      const mujoco::python::MjvPerturbWrapper& perturb,
      const mujoco::python::MjvOptionWrapper& vis_options,
      const mujoco::python::MjModelWrapper& model,
      const std::vector<uint8_t>& render_flags,
      const std::vector<mujoco::python::MjvGeomWrapper>& extra_geoms) {
    std::vector<mjvGeom> geoms;
    geoms.reserve(extra_geoms.size());
    for (const mujoco::python::MjvGeomWrapper& geom_wrapper : extra_geoms) {
      if (geom_wrapper.get()) {
        geoms.push_back(*geom_wrapper.get());
      }
    }

    std::string physics = physics_state;
    const std::vector<char> buffer = mujoco::studio::SerializeStatePayload(
        model_crc32, physics_spec, physics.data(), physics.size(),
        *camera.get(), *perturb.get(), *vis_options.get(), model.get()->opt,
        model.get()->vis, model.get()->stat, render_flags, geoms.data(),
        geoms.size());
    return py::bytes(buffer.data(), buffer.size());
  }

  // Upper bound of a serialized payload for a model whose physics state is
  // `physics_bytes` long. Used to size the StateServer's shared memory.
  static size_t MaxStatePayloadSize(size_t physics_bytes) {
    return mujoco::studio::MaxStatePayloadSize(physics_bytes);
  }

  // Uploads an RGB/RGBA image to the browser over the NetImgui texture
  // channel so handlers can display it with imgui.Image(). Returns the
  // texture id to use (allocates one when tex_id == 0). This is the UI-link
  // counterpart of the native renderer's UploadImage.
  uintptr_t UploadImage(uintptr_t tex_id, const py::bytes& pixels, int width,
                        int height, int bpp) {
    if (tex_id == 0) {
      tex_id = next_tex_id_++;
    }
    std::string data = pixels;
    if (width <= 0 || height <= 0 ||
        data.size() < static_cast<size_t>(width) * height * bpp) {
      LOG(Error, "UploadImage: bad dimensions %dx%dx%d for %zu bytes", width,
          height, bpp, data.size());
      return tex_id;
    }

    // NetImgui transfers RGBA8; expand RGB if needed.
    std::vector<uint8_t> rgba;
    const void* upload_data = data.data();
    if (bpp == 3) {
      rgba.resize(static_cast<size_t>(width) * height * 4);
      for (size_t p = 0; p < static_cast<size_t>(width) * height; ++p) {
        rgba[p * 4 + 0] = data[p * 3 + 0];
        rgba[p * 4 + 1] = data[p * 3 + 1];
        rgba[p * 4 + 2] = data[p * 3 + 2];
        rgba[p * 4 + 3] = 255;
      }
      upload_data = rgba.data();
    } else if (bpp != 4) {
      LOG(Error, "UploadImage: unsupported bpp %d (expected 3 or 4)", bpp);
      return tex_id;
    }

    py::gil_scoped_release no_gil;
    ImGui::SetCurrentContext(context_);
    NetImgui::SendDataTexture(
        static_cast<ImTextureID>(tex_id), const_cast<void*>(upload_data),
        static_cast<uint16_t>(width), static_cast<uint16_t>(height),
        NetImgui::eTexFormat::kTexFmtRGBA8);
    return tex_id;
  }

  intptr_t GetContext() const { return reinterpret_cast<intptr_t>(context_); }

  // The ImPlot context created alongside the ImGui context. Python must pass
  // this to ux.set_implot_context: extension modules each hold their own copy
  // of the ImPlot globals, so the context pointer has to be shared explicitly
  // (same pattern as get_context/set_imgui_context).
  intptr_t GetImPlotContext() const {
    return reinterpret_cast<intptr_t>(ImPlot::GetCurrentContext());
  }

 private:
  std::string title_;
  int port_;
  ImGuiContext* context_ = nullptr;
  bool is_drawing_remote_ = false;
  std::atomic<bool> close_requested_{false};
  // User texture ids start well above the ids ImGui's managed texture system
  // (font atlas) hands out, so the two can never collide in the browser's
  // texture map.
  uintptr_t next_tex_id_ = 0x10000;
};

PYBIND11_MODULE(ui_server, m, pybind11::mod_gil_not_used()) {
  py::module_::import("mujoco._structs");
  m.doc() = "MuJoCo web viewer UI server: NetImgui client and state streaming";

  py::class_<UiServer>(m, "UiServer")
      .def(py::init<const std::string&, int, const std::string&>(),
           py::arg("title"), py::arg("port") = 8888, py::arg("assets_dir") = "")
      .def("new_frame", &UiServer::NewFrame,
           "Starts a headless ImGui frame, returning True once the frame is "
           "active. When a browser is viewing the page (via the URL printed "
           "at startup), this returns at the browser's requested frame rate. "
           "When no browser is viewing, it blocks until one connects. "
           "Returns False only if request_close() interrupts the wait.")
      .def("request_close", &UiServer::RequestClose,
           "Makes a blocked (or future) new_frame() call return False. "
           "Safe to call from any thread.")
      .def("end_frame", &UiServer::EndFrame)
      .def("get_context", &UiServer::GetContext)
      .def("get_implot_context", &UiServer::GetImPlotContext)
      .def("serialize_state_payload", &UiServer::SerializeStatePayload)
      .def("upload_image", &UiServer::UploadImage)
      .def_static("max_state_payload_size", &UiServer::MaxStatePayloadSize);
  m.attr("MAX_EXTRA_GEOMS") = mujoco::studio::kMaxExtraGeoms;
}
