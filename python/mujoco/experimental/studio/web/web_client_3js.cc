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

// UI overlay + protocol module for the three.js Web Viewer client
// (--gfx=web_3js).
//
// This module owns everything except 3D rendering: it parses the model the
// page fetched from /model, owns the /state session (payload application via
// mj_setState + mj_forward, roster, roles, heartbeats) and the /ui NetImgui
// stream, and renders the streamed Studio UI plus the local role window with
// the ImGui OpenGL3 backend on a transparent overlay canvas stacked above the
// three.js canvas. The page's JavaScript renders the scene with three.js by
// reading body poses and the camera through the embind bridge below, and
// forwards all input events into ImGui here (scene interaction reaches the
// headless viewer through NetImgui, exactly like the Filament web_client).

#include <GLES3/gl3.h>
#include <emscripten.h>
#include <emscripten/bind.h>
#include <emscripten/html5.h>
#include <stdint.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <imgui.h>
#include <backends/imgui_impl_opengl3.h>
#include <mujoco/mujoco.h>
#include <NetImgui_Api.h>
#include "google/logging.h"
#include "state_payload.h"
#include "web_client_local_ui.h"
#include "web_client_remote_ui.h"
#include "web_client_session.h"

#if !defined(__EMSCRIPTEN__)
#error "web_client_3js.cc is only supported for Emscripten builds"
#endif

using mujoco::studio::DisconnectNotice;
using mujoco::studio::RemoteUi;
using mujoco::studio::RemoteUiState;
using mujoco::studio::RoleWindow;
using mujoco::studio::Session;
using mujoco::studio::SessionRole;
using mujoco::studio::SessionView;
using mujoco::studio::StatePayloadView;

namespace {

// How a spectating page drives its camera (see web_client.cc).
enum SpectatorCamMode {
  kSpecCamTumble = 0,
  kSpecCamWasd,
  kSpecCamFollow,
};

struct Telemetry {
  double last_rate_time = 0;
  uint64_t gui_bytes_per_sec = 0;
  uint64_t sim_bytes_per_sec = 0;
};

class AppCallbacks final : public RemoteUi::Callbacks,
                           public Session::Callbacks {
 public:
  // RemoteUi::Callbacks
  uintptr_t UploadTexture(uintptr_t current, const std::byte* rgba,
                          uint32_t width, uint32_t height) override;
  bool GpuReady() override;

  // Session::Callbacks
  bool ReadyForPayload() override;
  void OnPayload(const StatePayloadView& view) override;
  void OnModelChanged() override;
  void ConnectRemoteUi() override;
  void ShutdownRemoteUi() override;
  void SetCameraMode(int mode) override;
};

struct App {
  EMSCRIPTEN_WEBGL_CONTEXT_HANDLE gl_ctx = 0;
  bool gl_ready = false;

  mjModel* model = nullptr;
  mjData* data = nullptr;

  mjvPerturb perturb;
  mjvCamera camera;
  mjvOption vis_options;

  int spectator_cam_mode = kSpecCamTumble;
  float spectator_cam_speed = 0.001f;

  int frame_count = 0;
  int last_state_retry_frame = 0;

  std::vector<mjtNum> backend_state;
  int backend_state_sig = 0;
  bool backend_state_dirty = false;
  bool state_seen = false;

  std::vector<mjvGeom> extra_geoms;

  bool is_downloading = true;
  size_t bytes_downloaded = 0;
  size_t total_bytes = 0;
  int retry_count = 0;

  Telemetry telemetry;

  // eye(3), target(3), fovy, valid — refreshed every frame for JS.
  double camera_view[8] = {0};

  AppCallbacks callbacks;
  RemoteUi remote_ui{callbacks};
  Session session{callbacks};
  DisconnectNotice disconnect_notice;
  RoleWindow role_window;
};
App g_app;

// --- URL helpers (same scheme as web_client.cc). ----------------------------

std::string GetWsBaseUrl() {
  char* base = emscripten_run_script_string(
      "(window.location.protocol === 'https:' ? 'wss://' : 'ws://') + "
      "window.location.host");
  std::string url = base != nullptr ? base : "";
  if (url == "ws://" || url == "wss://" || url.empty()) {
    return "ws://localhost:8080";
  }
  return url;
}

std::string GetSessionId() {
  // EM_ASM (module scope), not emscripten_run_script (global eval): with
  // MODULARIZE there is no global Module object.
  static const std::string sid = []() {
    char* p = static_cast<char*>(EM_ASM_PTR({
      var sid = sessionStorage.getItem('mjwv_sid') ||
          (Date.now().toString(36) + Math.random().toString(36).slice(2));
      sessionStorage.setItem('mjwv_sid', sid);
      return stringToNewUTF8(sid);
    }));
    std::string s = p != nullptr ? p : "nosid";
    free(p);
    return s;
  }();
  return sid;
}

std::string WsUrl(const char* path) {
  return GetWsBaseUrl() + path + "?sid=" + GetSessionId();
}

// Standard CRC-32 matching Python's zlib.crc32 (see web_client.cc).
uint32_t Crc32(const uint8_t* data, size_t len) {
  uint32_t crc = 0xFFFFFFFFu;
  for (size_t i = 0; i < len; ++i) {
    crc ^= data[i];
    for (int bit = 0; bit < 8; ++bit) {
      crc = (crc >> 1) ^ (0xEDB88320u & (~(crc & 1u) + 1u));
    }
  }
  return crc ^ 0xFFFFFFFFu;
}

// Minimal stand-in for mujoco::platform::SetCamera (interaction.cc), covering
// the modes the browser uses.
constexpr int kTumbleCameraIdx = -3;
constexpr int kFreeCameraIdx = -2;
void SetCameraLite(const mjModel* m, mjvCamera* camera, int request_idx) {
  const int ncam = m ? m->ncam : 0;
  const int idx = std::clamp(request_idx, kTumbleCameraIdx, ncam - 1);
  if (idx == kTumbleCameraIdx) {
    camera->type = mjCAMERA_FREE;
    camera->fixedcamid = -1;
  } else if (idx == kFreeCameraIdx) {
    camera->type = mjCAMERA_FREE;
    camera->distance = 2.0f;
    camera->fixedcamid = -1;
  } else if (idx >= 0) {
    camera->type = mjCAMERA_FIXED;
    camera->fixedcamid = idx;
  }
}

// --- State payload application (web_client.cc, minus the Filament parts). ---

void ApplyStatePayload(const StatePayloadView& view) {
  mjModel* model = g_app.model;

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

  if (view.render_state != nullptr) {
    mujoco::studio::RenderStateView rs;
    mujoco::studio::ParseRenderState(view.render_state, &rs);
    if (g_app.session.Role() != SessionRole::kSpectating ||
        g_app.spectator_cam_mode == kSpecCamFollow) {
      g_app.camera = rs.camera;
    }
    g_app.perturb = rs.perturb;
    g_app.vis_options = rs.vis_options;
    model->opt = rs.opt;
    model->vis = rs.vis;
    model->stat = rs.stat;
  }

  g_app.extra_geoms.resize(view.extra_geom_count);
  if (view.extra_geom_count > 0) {
    memcpy(g_app.extra_geoms.data(), view.extra_geoms,
           view.extra_geom_count * sizeof(mjvGeom));
  }
}

void SetSpectatorCameraMode(int mode) {
  if (mode == g_app.spectator_cam_mode) {
    return;
  }
  g_app.spectator_cam_mode = mode;
  if (!g_app.model) {
    return;
  }
  if (mode == kSpecCamTumble) {
    SetCameraLite(g_app.model, &g_app.camera, kTumbleCameraIdx);
  } else if (mode == kSpecCamWasd) {
    SetCameraLite(g_app.model, &g_app.camera, kFreeCameraIdx);
  }
  // kSpecCamFollow: the next state payload restores the controller's camera.
}

// Local camera control for spectating pages in a free camera mode; verbatim
// from web_client.cc (the controller's input goes to the headless viewer).
void HandleSpectatorCameraInput() {
  if (g_app.spectator_cam_mode == kSpecCamFollow || !g_app.model) {
    return;
  }
  const mjModel* model = g_app.model;
  ImGuiIO& io = ImGui::GetIO();
  const bool wasd = g_app.spectator_cam_mode == kSpecCamWasd;

  if (g_app.camera.type == mjCAMERA_FIXED) {
    mjv_defaultFreeCamera(model, &g_app.camera);
    if (wasd) {
      SetCameraLite(model, &g_app.camera, kFreeCameraIdx);
    }
  }

  if (!io.WantCaptureMouse && io.DisplaySize.x > 0 && io.DisplaySize.y > 0) {
    const float mouse_dx = io.MouseDelta.x / io.DisplaySize.x;
    const float mouse_dy = io.MouseDelta.y / io.DisplaySize.y;
    const bool is_mouse_dragging =
        (mouse_dx != 0.0f || mouse_dy != 0.0f) &&
        (ImGui::IsMouseDown(ImGuiMouseButton_Left) ||
         ImGui::IsMouseDown(ImGuiMouseButton_Right) ||
         ImGui::IsMouseDown(ImGuiMouseButton_Middle));
    if (is_mouse_dragging) {
      if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
        if (wasd) {
          mjv_moveCamera(model, mjMOUSE_TURN_H, mouse_dx, 0.f, &g_app.camera);
          mjv_moveCamera(model, mjMOUSE_TURN_V, 0.f, mouse_dy, &g_app.camera);
        } else {
          mjv_moveCamera(model, mjMOUSE_ROTATE_H, mouse_dx, 0.f, &g_app.camera);
          mjv_moveCamera(model, mjMOUSE_ROTATE_V, 0.f, mouse_dy, &g_app.camera);
        }
      } else if (ImGui::IsMouseDown(ImGuiMouseButton_Middle) && !wasd) {
        mjv_moveCamera(model, mjMOUSE_ZOOM, 0.f, mouse_dy, &g_app.camera);
      }
      if (ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
        mjv_moveCamera(model, io.KeyShift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V,
                       mouse_dx, mouse_dy, &g_app.camera);
      }
    }
    const float mouse_scroll = io.MouseWheel / 50.0f;
    if (mouse_scroll != 0.0f && !wasd) {
      mjv_moveCamera(model, mjMOUSE_ZOOM, 0.f, -mouse_scroll, &g_app.camera);
    }
  }

  if (wasd && !io.WantCaptureKeyboard) {
    bool moved = false;
    const float speed = g_app.spectator_cam_speed;
    if (ImGui::IsKeyDown(ImGuiKey_W)) {
      mjv_moveCamera(model, mjMOUSE_MOVE_H_REL, 0, speed, &g_app.camera);
      moved = true;
    } else if (ImGui::IsKeyDown(ImGuiKey_S)) {
      mjv_moveCamera(model, mjMOUSE_MOVE_H_REL, 0, -speed, &g_app.camera);
      moved = true;
    }
    if (ImGui::IsKeyDown(ImGuiKey_A)) {
      mjv_moveCamera(model, mjMOUSE_MOVE_H_REL, -speed, 0, &g_app.camera);
      moved = true;
    } else if (ImGui::IsKeyDown(ImGuiKey_D)) {
      mjv_moveCamera(model, mjMOUSE_MOVE_H_REL, speed, 0, &g_app.camera);
      moved = true;
    }
    if (ImGui::IsKeyDown(ImGuiKey_Q)) {
      mjv_moveCamera(model, mjMOUSE_MOVE_V_REL, 0, speed, &g_app.camera);
      moved = true;
    } else if (ImGui::IsKeyDown(ImGuiKey_E)) {
      mjv_moveCamera(model, mjMOUSE_MOVE_V_REL, 0, -speed, &g_app.camera);
      moved = true;
    }
    if (moved) {
      const float max_speed = io.KeyShift ? 0.1f : 0.01f;
      g_app.spectator_cam_speed =
          std::min(g_app.spectator_cam_speed + 0.001f, max_speed);
    } else {
      g_app.spectator_cam_speed = 0.001f;
    }
  }
}

// Publishes the camera as eye/target/fovy in MuJoCo world coordinates for the
// three.js side (which applies its own z-up -> y-up conversion).
void ComputeCameraView() {
  double* out = g_app.camera_view;
  out[7] = 0;
  if (!g_app.model || !g_app.data) {
    return;
  }
  const mjvCamera& cam = g_app.camera;
  if (cam.type == mjCAMERA_FIXED && cam.fixedcamid >= 0 &&
      cam.fixedcamid < g_app.model->ncam) {
    const int id = cam.fixedcamid;
    const mjtNum* pos = g_app.data->cam_xpos + 3 * id;
    const mjtNum* mat = g_app.data->cam_xmat + 9 * id;
    // The camera looks along -z of its frame.
    out[0] = pos[0];
    out[1] = pos[1];
    out[2] = pos[2];
    out[3] = pos[0] - mat[2];
    out[4] = pos[1] - mat[5];
    out[5] = pos[2] - mat[8];
    out[6] = g_app.model->cam_fovy[id];
    out[7] = 1;
    return;
  }

  double lookat[3] = {cam.lookat[0], cam.lookat[1], cam.lookat[2]};
  if (cam.type == mjCAMERA_TRACKING && cam.trackbodyid >= 0 &&
      cam.trackbodyid < g_app.model->nbody) {
    const mjtNum* xpos = g_app.data->xpos + 3 * cam.trackbodyid;
    lookat[0] = xpos[0];
    lookat[1] = xpos[1];
    lookat[2] = xpos[2];
  }
  const double az = cam.azimuth * M_PI / 180.0;
  const double el = cam.elevation * M_PI / 180.0;
  const double forward[3] = {cos(el) * cos(az), cos(el) * sin(az), sin(el)};
  out[0] = lookat[0] - forward[0] * cam.distance;
  out[1] = lookat[1] - forward[1] * cam.distance;
  out[2] = lookat[2] - forward[2] * cam.distance;
  out[3] = lookat[0];
  out[4] = lookat[1];
  out[5] = lookat[2];
  out[6] = g_app.model->vis.global.fovy;
  out[7] = 1;
}

// --- AppCallbacks bodies. ---------------------------------------------------

uintptr_t AppCallbacks::UploadTexture(uintptr_t current, const std::byte* rgba,
                                      uint32_t width, uint32_t height) {
  emscripten_webgl_make_context_current(g_app.gl_ctx);
  if (rgba == nullptr) {
    if (current != 0) {
      GLuint tex = static_cast<GLuint>(current);
      glDeleteTextures(1, &tex);
    }
    return 0;
  }
  GLuint tex = static_cast<GLuint>(current);
  if (tex == 0) {
    glGenTextures(1, &tex);
  }
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA,
               GL_UNSIGNED_BYTE, rgba);
  return tex;
}

bool AppCallbacks::GpuReady() { return g_app.gl_ready; }

bool AppCallbacks::ReadyForPayload() {
  return g_app.model != nullptr && !g_app.is_downloading;
}

void AppCallbacks::OnPayload(const StatePayloadView& view) {
  ApplyStatePayload(view);
}

void AppCallbacks::OnModelChanged() {
  g_app.is_downloading = true;
  // The page refetches /model and calls parseModelBuffer + rebuilds the
  // three.js scene.
  EM_ASM({
    if (Module.onModelChanged) Module.onModelChanged();
  });
}

void AppCallbacks::ConnectRemoteUi() { g_app.remote_ui.Connect(WsUrl("/ui")); }

void AppCallbacks::ShutdownRemoteUi() { g_app.remote_ui.Shutdown(); }

void AppCallbacks::SetCameraMode(int mode) { SetSpectatorCameraMode(mode); }

// --- Local UI. --------------------------------------------------------------

void BuildBrowserGui() {
  const double now = ImGui::GetTime();
  if (now - g_app.telemetry.last_rate_time >= 1.0) {
    g_app.telemetry.gui_bytes_per_sec = g_app.remote_ui.ConsumeByteCount();
    g_app.telemetry.sim_bytes_per_sec = g_app.session.ConsumeByteCount();
    g_app.telemetry.last_rate_time = now;
  }

  const double last_msg = g_app.session.LastMessageTime();
  const double stale_sec =
      last_msg > 0 ? emscripten_get_now() / 1000.0 - last_msg : -1.0;
  g_app.disconnect_notice.Draw(g_app.session.ServerCloseCode(), stale_sec,
                               g_app.is_downloading);

  SessionView view;
  g_app.session.FillView(&view);
  view.gui_bytes_per_sec = g_app.telemetry.gui_bytes_per_sec;
  view.sim_bytes_per_sec = g_app.telemetry.sim_bytes_per_sec;
  view.have_remote_frame = g_app.remote_ui.RemoteDrawData() != nullptr;
  view.camera_mode = g_app.spectator_cam_mode;
  view.is_downloading = g_app.is_downloading;
  view.bytes_downloaded = g_app.bytes_downloaded;
  view.total_bytes = g_app.total_bytes;
  view.retry_count = g_app.retry_count;
  g_app.role_window.Draw(view, g_app.session);
}

// --- Frame. -----------------------------------------------------------------

int FrameImpl(double width, double height, double dpr, double dt_sec) {
  g_app.frame_count++;
  ImGuiIO& io = ImGui::GetIO();
  io.DisplaySize = ImVec2(static_cast<float>(width), static_cast<float>(height));
  io.DisplayFramebufferScale =
      ImVec2(static_cast<float>(dpr), static_cast<float>(dpr));
  io.DeltaTime = dt_sec > 0 ? static_cast<float>(dt_sec) : 1.0f / 60.0f;

  g_app.session.Update();

  // Reconnect the state WebSocket if it dropped (see web_client.cc).
  const int state_retry_interval =
      g_app.session.ServerCloseCode() != 0 ? 300 : 60;
  if (!g_app.session.HasSocket() && g_app.model && !g_app.is_downloading &&
      g_app.frame_count - g_app.last_state_retry_frame > state_retry_interval) {
    g_app.last_state_retry_frame = g_app.frame_count;
    LOG(Info, "State WebSocket down; reconnecting...");
    g_app.session.Connect(WsUrl("/state"));
  }

  RemoteUiState ui_state = RemoteUiState::kNoSocket;
  if (g_app.remote_ui.HasSocket()) {
    switch (g_app.remote_ui.ConnectionState()) {
      case RemoteUi::ReadyState::kOpen:
        ui_state = RemoteUiState::kOpen;
        break;
      case RemoteUi::ReadyState::kClosed:
      case RemoteUi::ReadyState::kError:
        ui_state = RemoteUiState::kClosedOrError;
        break;
      default:
        ui_state = RemoteUiState::kConnecting;
        break;
    }
  }
  g_app.session.HandleRemoteUiState(ui_state, g_app.remote_ui.CloseCode());

  g_app.remote_ui.SetMaxClip(static_cast<float>(width),
                             static_cast<float>(height));
  g_app.remote_ui.ReceiveAndProcessCommands(g_app.frame_count);

  ImGui_ImplOpenGL3_NewFrame();
  ImGui::NewFrame();

  g_app.remote_ui.CaptureAndSendInput();
  if (g_app.session.Role() == SessionRole::kSpectating) {
    HandleSpectatorCameraInput();
  }

  BuildBrowserGui();

  ImGui::Render();

  // Inject remote draw lists under the local UI (see web_client.cc).
  ImDrawData* remote_draw_data = g_app.remote_ui.RemoteDrawData();
  if (remote_draw_data && remote_draw_data->Valid &&
      g_app.session.Role() == SessionRole::kControlling) {
    ImDrawData* local_draw_data = ImGui::GetDrawData();
    if (local_draw_data) {
      ImVector<ImDrawList*> local_lists;
      local_lists.reserve(local_draw_data->CmdListsCount);
      for (int i = 0; i < local_draw_data->CmdListsCount; ++i) {
        local_lists.push_back(local_draw_data->CmdLists[i]);
      }
      local_draw_data->CmdLists.resize(0);
      local_draw_data->CmdListsCount = 0;
      local_draw_data->TotalVtxCount = 0;
      local_draw_data->TotalIdxCount = 0;
      for (int i = 0; i < remote_draw_data->CmdListsCount; ++i) {
        local_draw_data->AddDrawList(remote_draw_data->CmdLists[i]);
      }
      for (int i = 0; i < local_lists.Size; ++i) {
        local_draw_data->AddDrawList(local_lists[i]);
      }
    }
  }

  // Apply the newest simulation state.
  int status = 0;
  if (g_app.model && g_app.data && g_app.backend_state_dirty &&
      !g_app.is_downloading) {
    mj_setState(g_app.model, g_app.data, g_app.backend_state.data(),
                g_app.backend_state_sig);
    mj_forward(g_app.model, g_app.data);
    g_app.backend_state_dirty = false;
    g_app.state_seen = true;
    status |= 1;  // poses updated this frame
  }

  ComputeCameraView();

  // Render the UI into the (transparent) overlay canvas.
  emscripten_webgl_make_context_current(g_app.gl_ctx);
  glViewport(0, 0, static_cast<int>(width * dpr),
             static_cast<int>(height * dpr));
  glClearColor(0, 0, 0, 0);
  glClear(GL_COLOR_BUFFER_BIT);
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
  return status;
}

int Frame(double width, double height, double dpr, double dt_sec) {
  // An exception escaping the rAF callback would silently kill the page's
  // loop; catch, log, and report instead.
  try {
    return FrameImpl(width, height, dpr, dt_sec);
  } catch (const std::exception& e) {
    LOG(Error, "FATAL: uncaught exception in Frame: %s", e.what());
    return -1;
  } catch (...) {
    LOG(Error, "FATAL: uncaught non-std exception in Frame");
    return -1;
  }
}

// --- Model loading. ---------------------------------------------------------

uintptr_t AllocModelBuffer(size_t size) {
  char* ptr = new (std::nothrow) char[size];
  if (!ptr) {
    LOG(Error, "Failed to allocate WASM model buffer of size %zu", size);
    return 0;
  }
  return reinterpret_cast<uintptr_t>(ptr);
}

void FreeModelBuffer(uintptr_t ptr_val) {
  if (ptr_val) {
    delete[] reinterpret_cast<char*>(ptr_val);
  }
}

bool ParseModelBuffer(uintptr_t ptr_val, size_t size) {
  const char* bytes = reinterpret_cast<char*>(ptr_val);
  mjVFS vfs;
  mj_defaultVFS(&vfs);
  mj_addBufferVFS(&vfs, "model.mjb", bytes, size);
  mjModel* model = mj_loadModel("model.mjb", &vfs);
  mj_deleteVFS(&vfs);
  if (!model) {
    LOG(Error, "Failed to load model from /model bytes (%zu bytes)", size);
    return false;
  }

  if (g_app.data) {
    mj_deleteData(g_app.data);
    g_app.data = nullptr;
  }
  if (g_app.model) {
    mj_deleteModel(g_app.model);
  }
  g_app.model = model;
  g_app.data = mj_makeData(model);
  mj_forward(g_app.model, g_app.data);

  g_app.session.SetModelCrc32(
      Crc32(reinterpret_cast<const uint8_t*>(bytes), size));
  g_app.backend_state_dirty = false;
  g_app.state_seen = false;
  g_app.is_downloading = false;

  mjv_defaultPerturb(&g_app.perturb);
  mjv_defaultCamera(&g_app.camera);
  mjv_defaultOption(&g_app.vis_options);
  const int model_cam = model->vis.global.cameraid;
  if (model_cam >= 0 && model_cam < model->ncam) {
    SetCameraLite(model, &g_app.camera, model_cam);
  } else {
    mjv_defaultFreeCamera(model, &g_app.camera);
  }

  if (!g_app.session.HasSocket()) {
    g_app.session.Connect(WsUrl("/state"));
  }
  LOG(Info, "Model loaded (%zu bytes)", size);
  return true;
}

void UpdateModelDownloadProgress(size_t bytes_downloaded, size_t total_bytes,
                                 int retry_count) {
  g_app.is_downloading = true;
  g_app.bytes_downloaded = bytes_downloaded;
  g_app.total_bytes = total_bytes;
  g_app.retry_count = retry_count;
}

// --- Input bridge. ----------------------------------------------------------

void OnMouseMove(double x, double y) { ImGui::GetIO().AddMousePosEvent(x, y); }

void OnMouseButton(int button, bool down) {
  if (button >= 0 && button < ImGuiMouseButton_COUNT) {
    ImGui::GetIO().AddMouseButtonEvent(button, down);
  }
}

void OnMouseWheel(double dx, double dy) {
  ImGui::GetIO().AddMouseWheelEvent(static_cast<float>(dx),
                                    static_cast<float>(dy));
}

void OnFocus(bool focused) { ImGui::GetIO().AddFocusEvent(focused); }

void OnKey(int imgui_key, bool down, bool ctrl, bool shift, bool alt,
           bool super) {
  ImGuiIO& io = ImGui::GetIO();
  io.AddKeyEvent(ImGuiMod_Ctrl, ctrl);
  io.AddKeyEvent(ImGuiMod_Shift, shift);
  io.AddKeyEvent(ImGuiMod_Alt, alt);
  io.AddKeyEvent(ImGuiMod_Super, super);
  if (imgui_key != ImGuiKey_None) {
    io.AddKeyEvent(static_cast<ImGuiKey>(imgui_key), down);
  }
}

void OnTextInput(unsigned int codepoint) {
  ImGui::GetIO().AddInputCharacter(codepoint);
}

// Maps a DOM KeyboardEvent.code to an ImGuiKey; the page caches the results.
int KeyFromDomCode(std::string code) {
  static const std::unordered_map<std::string, ImGuiKey> kMap = []() {
    std::unordered_map<std::string, ImGuiKey> m;
    for (int i = 0; i < 26; ++i) {
      m["Key" + std::string(1, static_cast<char>('A' + i))] =
          static_cast<ImGuiKey>(ImGuiKey_A + i);
    }
    for (int i = 0; i < 10; ++i) {
      m["Digit" + std::to_string(i)] =
          static_cast<ImGuiKey>(ImGuiKey_0 + i);
      m["Numpad" + std::to_string(i)] =
          static_cast<ImGuiKey>(ImGuiKey_Keypad0 + i);
    }
    for (int i = 0; i < 12; ++i) {
      m["F" + std::to_string(i + 1)] = static_cast<ImGuiKey>(ImGuiKey_F1 + i);
    }
    m["ArrowLeft"] = ImGuiKey_LeftArrow;
    m["ArrowRight"] = ImGuiKey_RightArrow;
    m["ArrowUp"] = ImGuiKey_UpArrow;
    m["ArrowDown"] = ImGuiKey_DownArrow;
    m["Tab"] = ImGuiKey_Tab;
    m["Enter"] = ImGuiKey_Enter;
    m["NumpadEnter"] = ImGuiKey_KeypadEnter;
    m["Escape"] = ImGuiKey_Escape;
    m["Space"] = ImGuiKey_Space;
    m["Backspace"] = ImGuiKey_Backspace;
    m["Delete"] = ImGuiKey_Delete;
    m["Insert"] = ImGuiKey_Insert;
    m["Home"] = ImGuiKey_Home;
    m["End"] = ImGuiKey_End;
    m["PageUp"] = ImGuiKey_PageUp;
    m["PageDown"] = ImGuiKey_PageDown;
    m["ShiftLeft"] = ImGuiKey_LeftShift;
    m["ShiftRight"] = ImGuiKey_RightShift;
    m["ControlLeft"] = ImGuiKey_LeftCtrl;
    m["ControlRight"] = ImGuiKey_RightCtrl;
    m["AltLeft"] = ImGuiKey_LeftAlt;
    m["AltRight"] = ImGuiKey_RightAlt;
    m["MetaLeft"] = ImGuiKey_LeftSuper;
    m["MetaRight"] = ImGuiKey_RightSuper;
    m["Minus"] = ImGuiKey_Minus;
    m["Equal"] = ImGuiKey_Equal;
    m["BracketLeft"] = ImGuiKey_LeftBracket;
    m["BracketRight"] = ImGuiKey_RightBracket;
    m["Backslash"] = ImGuiKey_Backslash;
    m["Semicolon"] = ImGuiKey_Semicolon;
    m["Quote"] = ImGuiKey_Apostrophe;
    m["Comma"] = ImGuiKey_Comma;
    m["Period"] = ImGuiKey_Period;
    m["Slash"] = ImGuiKey_Slash;
    m["Backquote"] = ImGuiKey_GraveAccent;
    m["CapsLock"] = ImGuiKey_CapsLock;
    return m;
  }();
  auto it = kMap.find(code);
  return it != kMap.end() ? static_cast<int>(it->second)
                          : static_cast<int>(ImGuiKey_None);
}

// --- JS-facing views. -------------------------------------------------------
// Views alias WASM memory and are invalidated by heap growth: the page must
// re-fetch them every frame rather than caching them.

emscripten::val XposView() {
  if (!g_app.data || !g_app.model) return emscripten::val::null();
  return emscripten::val(
      emscripten::typed_memory_view(3 * g_app.model->nbody, g_app.data->xpos));
}

emscripten::val XquatView() {
  if (!g_app.data || !g_app.model) return emscripten::val::null();
  return emscripten::val(
      emscripten::typed_memory_view(4 * g_app.model->nbody, g_app.data->xquat));
}

emscripten::val LightXposView() {
  if (!g_app.data || !g_app.model) return emscripten::val::null();
  return emscripten::val(emscripten::typed_memory_view(
      3 * g_app.model->nlight, g_app.data->light_xpos));
}

emscripten::val LightXdirView() {
  if (!g_app.data || !g_app.model) return emscripten::val::null();
  return emscripten::val(emscripten::typed_memory_view(
      3 * g_app.model->nlight, g_app.data->light_xdir));
}

emscripten::val CameraView() {
  return emscripten::val(emscripten::typed_memory_view(8, g_app.camera_view));
}

bool StateSeen() { return g_app.state_seen; }

int SessionRoleValue() { return static_cast<int>(g_app.session.Role()); }

// --- Startup. ---------------------------------------------------------------

void StartApp() {
  // Surface MuJoCo errors as C++ exceptions so Frame()'s catch can log them.
  mju_user_error = +[](const char* msg) {
    LOG(Error, "MuJoCo error: %s", msg);
    throw std::runtime_error(msg);
  };
  mju_user_warning = +[](const char* msg) {
    LOG(Warning, "MuJoCo warning: %s", msg);
  };

  EmscriptenWebGLContextAttributes attrs;
  emscripten_webgl_init_context_attributes(&attrs);
  attrs.majorVersion = 2;
  attrs.minorVersion = 0;
  attrs.alpha = EM_TRUE;
  attrs.premultipliedAlpha = EM_FALSE;
  attrs.depth = EM_FALSE;
  attrs.stencil = EM_FALSE;
  attrs.antialias = EM_FALSE;
  g_app.gl_ctx = emscripten_webgl_create_context("#ui-canvas", &attrs);
  if (g_app.gl_ctx <= 0) {
    LOG(Error, "Failed to create WebGL2 context on #ui-canvas (%d)",
        static_cast<int>(g_app.gl_ctx));
    return;
  }
  emscripten_webgl_make_context_current(g_app.gl_ctx);

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  io.IniFilename = nullptr;
  io.BackendFlags |= ImGuiBackendFlags_RendererHasVtxOffset;
  ImGui::StyleColorsDark();
  ImGui_ImplOpenGL3_Init("#version 300 es");
  g_app.gl_ready = true;

  NetImgui::Internal::Network::Startup();
  g_app.remote_ui.Connect(WsUrl("/ui"));
  LOG(Info, "web_client_ui started");
}

}  // namespace

EMSCRIPTEN_BINDINGS(web_client_3js_bindings) {
  emscripten::function("startApp", &StartApp);
  emscripten::function("frame", &Frame);
  emscripten::function("allocModelBuffer", &AllocModelBuffer);
  emscripten::function("freeModelBuffer", &FreeModelBuffer);
  emscripten::function("parseModelBuffer", &ParseModelBuffer);
  emscripten::function("updateModelDownloadProgress",
                       &UpdateModelDownloadProgress);
  emscripten::function("onMouseMove", &OnMouseMove);
  emscripten::function("onMouseButton", &OnMouseButton);
  emscripten::function("onMouseWheel", &OnMouseWheel);
  emscripten::function("onFocus", &OnFocus);
  emscripten::function("onKey", &OnKey);
  emscripten::function("onTextInput", &OnTextInput);
  emscripten::function("keyFromDomCode", &KeyFromDomCode);
  emscripten::function("xposView", &XposView);
  emscripten::function("xquatView", &XquatView);
  emscripten::function("lightXposView", &LightXposView);
  emscripten::function("lightXdirView", &LightXdirView);
  emscripten::function("cameraView", &CameraView);
  emscripten::function("stateSeen", &StateSeen);
  emscripten::function("sessionRole", &SessionRoleValue);
}
