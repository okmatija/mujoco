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

// NetImgui WASM Viewer — renders remote ImGui draw data in a browser.
// Bridges the NetImgui server protocol to a WebGL/SDL2/Emscripten rendering
// context.
//
// This file is the Emscripten glue: app/scene state, the main loop, and the
// wiring between the two links (web_client_ui_link.h for the streamed Studio
// UI, web_client_state_link.h for simulation state) and the Filament
// renderer.

#include <SDL.h>
#include <SDL_opengl.h>
#include <emscripten.h>
#include <emscripten/fetch.h>
#include <imgui.h>
#include <implot.h>
#include <mujoco/mujoco.h>
#include <stdint.h>
#include <stdio.h>

#include <algorithm>
#include <cfloat>
#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "NetImgui_Api.h"
#include "experimental/platform/hal/renderer.h"
#include "experimental/platform/hal/window.h"
#include "experimental/platform/sim/model_holder.h"
#include "experimental/platform/ux/interaction.h"
#include "google/logging.h"
#include "state_payload.h"
#include "web_client_state_link.h"
#include "web_client_ui_link.h"

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

// How a spectating page drives its camera. The free modes control the local
// camera directly; kSpecCamFollow mirrors the controller's camera from the
// state broadcast.
enum SpectatorCamMode {
  kSpecCamTumble = 0,  // Orbit around the lookat point with the mouse.
  kSpecCamWasd,        // Fly camera: WASD/QE moves, mouse drag turns.
  kSpecCamFollow,      // Follow the controller's camera.
};

// Where the Link window is anchored on the page. The window is always
// draggable; releasing a drag near an anchor snaps to it, anywhere else
// leaves the window Free at its dropped position.
enum LinkWindowPos {
  kLinkPosFree = 0,
  kLinkPosTopLeft,  // The anchors must stay contiguous from here on.
  kLinkPosTopMid,
  kLinkPosTopRight,
  kLinkPosBottomLeft,
  kLinkPosBottomMid,
  kLinkPosBottomRight,
};

// Byte-rate telemetry shown in the Link window.
struct Telemetry {
  double last_rate_time = 0;
  uint64_t gui_bytes_per_sec = 0;
  uint64_t sim_bytes_per_sec = 0;
};

struct AppState {
  std::unique_ptr<mujoco::platform::Window> window;
  std::unique_ptr<mujoco::platform::ModelHolder> model_holder;
  mujoco::platform::Renderer* renderer = nullptr;
  mjvPerturb perturb;
  mjvCamera camera;
  mjvOption vis_options;

  // True when another browser holds the controller slot: this page renders the
  // scene from the state broadcast but has no UI stream or input until the
  // user takes control.
  bool spectator = false;
  int ui_reject_count = 0;
  // Session roster, from the server's updates on the state channel.
  int session_viewers = 0;
  int queue_pos = 0;  // 1-based position in the control queue; 0 = not queued.
  int queue_len = 0;
  int max_spectators = 8;  // Runtime limit, from the roster.

  // Spectator camera (combo order matches SpectatorCamMode).
  int spectator_cam_mode = kSpecCamTumble;
  float spectator_cam_speed = 0.001f;  // WASD speed; accelerates while held.

  // Link window anchor; updated by the drag-snap logic in BuildBrowserGui.
  int link_window_pos = kLinkPosTopMid;
  // Link window hover-expand state: expanded form, last hovered time (for
  // the collapse grace period), and whether a drag started on the window.
  bool link_expanded = false;
  double link_hover_time = 0;
  bool link_dragging = false;
  // Max spectators edit grace: the local value wins over the roster until
  // max_spectators_edit_time is old enough (see the InputInt).
  int max_spectators_edit = -1;
  double max_spectators_edit_time = -1e9;
  // True while the server-not-reachable notice is up (log once per outage).
  bool disconnected_notice_logged = false;

  Telemetry telemetry;

  // Main loop frame counters (MainLoopImpl): heartbeat pacing and
  // reconnect pacing for the two WebSockets.
  int frame_count = 0;
  int last_heartbeat_frame = 0;
  int last_state_retry_frame = 0;
  int last_ui_retry_frame = 0;

  // Backend state received from the Python simulation via WebSocket.
  std::vector<mjtNum> backend_state;
  int backend_state_sig = 0;
  bool backend_state_dirty = false;

  // User-injected geoms (Viewer.extra_geoms) received with the state payload.
  std::vector<mjvGeom> extra_geoms;
};
AppState g_app;

// Session-channel messages, sent and received as text frames on the state
// WebSocket. Keep in sync with _SessionMessage / _GRANT_MESSAGE in
// web_server.py.
constexpr char kMsgRequestControl[] = "request_control";
constexpr char kMsgLeaveQueue[] = "leave_queue";
constexpr char kMsgForceControl[] = "force_control";
constexpr char kMsgHeartbeat[] = "heartbeat";
constexpr char kMsgGrant[] = "grant";
constexpr char kMsgMaxSpectatorsPrefix[] = "max_spectators=";

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

// Stable per-tab id sent with every WebSocket connect (?sid=...), letting
// the server tie this page's /ui and /state connections together. It lives
// in sessionStorage so it survives page reloads: on a model change every
// page reloads and the server restarts with the controller slot reserved
// for the previous controller's sid — a reload-stable id is what lets the
// controller keep control. (crypto.randomUUID would need a secure context,
// which plain http on a LAN is not, hence the homegrown id. Duplicating a
// tab copies sessionStorage, so twin tabs share a sid and the server sees
// them as one viewer — a known cosmetic edge case.)
std::string GetSessionId() {
  static const std::string sid = emscripten_run_script_string(
      "(function() {"
      "  Module.sessionId = Module.sessionId ||"
      "      sessionStorage.getItem('mjwv_sid') ||"
      "      (Date.now().toString(36) + Math.random().toString(36).slice(2));"
      "  sessionStorage.setItem('mjwv_sid', Module.sessionId);"
      "  return Module.sessionId;"
      "})()");
  return sid;
}

std::string WsUrl(const char* path) {
  return GetWsBaseUrl() + path + "?sid=" + GetSessionId();
}

// Updates the spectator flag, mirroring it into JS (Module.isSpectator) so
// the page script can gate controller-only actions like model drag-and-drop.
// The server enforces the same rule; the mirror only saves a pointless
// upload and gives immediate feedback.
void SetSpectator(bool spectator) {
  g_app.spectator = spectator;
  // The JS avoids operators that clang-format would reformat into invalid
  // JavaScript (it once split a `!==` into `!= =`, breaking the build).
  EM_ASM({ Module.isSpectator = !!$0; }, spectator ? 1 : 0);
}

// Returns true if the Filament rendering context is initialized and ready for
// GPU texture uploads. The Renderer object is created in main() and is always
// non-null, but the Filament context is only initialized when Renderer::Init()
// is called from SetupScene after the async model fetch completes.
bool IsFilamentReady() {
  return g_app.renderer && g_app.model_holder && g_app.model_holder->ok();
}

// Applies a parsed state payload to the app. Wired as g_state_link.on_payload;
// runs only after the model-change/reload policy has accepted the payload.
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

    // Spectators in a free camera mode keep their own local camera; everyone
    // else (the controller, and spectators in Follow Controller) mirrors the
    // controller's camera.
    if (!g_app.spectator || g_app.spectator_cam_mode == kSpecCamFollow) {
      memcpy(&g_app.camera, vis_ptr, sizeof(mjvCamera));
    }
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
// inside the classes; see web_client_ui_link.h and web_client_state_link.h.
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
      int viewers = 0, queue_pos = 0, queue_len = 0;
      char role[16] = {0};
      int max_spectators = 0;
      if (sscanf(text,
                 "viewers=%d;role=%15[^;];queue_pos=%d;queue_len=%d;"
                 "max_spectators=%d",
                 &viewers, role, &queue_pos, &queue_len,
                 &max_spectators) == 5) {
        if (viewers != g_app.session_viewers || queue_pos != g_app.queue_pos ||
            queue_len != g_app.queue_len) {
          LOG(Info, "Session roster: %s", text);
        }
        g_app.session_viewers = viewers;
        g_app.queue_pos = queue_pos;
        g_app.queue_len = queue_len;
        g_app.max_spectators = max_spectators;
      } else if (strcmp(text, kMsgGrant) == 0) {
        // Our turn: the controller slot is reserved for this page.
        LOG(Info, "Control granted; claiming the controller slot");
        SetSpectator(false);
        g_app.ui_reject_count = 0;
        g_ui_link.Connect(WsUrl("/ui"));
      }
    });

// Switches the spectator camera mode, re-seeding the local camera when
// entering a free mode so the view starts from the current pose.
void SetSpectatorCameraMode(int mode) {
  if (mode == g_app.spectator_cam_mode) {
    return;
  }
  g_app.spectator_cam_mode = mode;
  if (!g_app.model_holder || !g_app.model_holder->ok()) {
    return;
  }
  const mjModel* model = g_app.model_holder->model();
  if (mode == kSpecCamTumble) {
    mujoco::platform::SetCamera(model, &g_app.camera,
                                mujoco::platform::kTumbleCameraIdx);
  } else if (mode == kSpecCamWasd) {
    mujoco::platform::SetCamera(model, &g_app.camera,
                                mujoco::platform::kFreeCameraIdx);
  }
  // kSpecCamFollow: the next state payload restores the controller's camera.
}

// Local camera control for a spectating page in a free camera mode. The
// controller's input goes to the headless viewer instead (CaptureAndSendInput),
// which streams its camera back over the state WebSocket. Mirrors the
// mouse/WASD camera handling in src/experimental/studio/app.cc.
// TODO(matijak): Share the camera handling code with the studio app (e.g. in
// platform/ux/interaction.cc) instead of mirroring it here.
void HandleSpectatorCameraInput() {
  if (g_app.spectator_cam_mode == kSpecCamFollow) {
    return;
  }
  if (!g_app.model_holder || !g_app.model_holder->ok()) {
    return;
  }
  const mjModel* model = g_app.model_holder->model();
  ImGuiIO& io = ImGui::GetIO();
  const bool wasd = g_app.spectator_cam_mode == kSpecCamWasd;

  // The camera can be fixed on entry (SetupScene honours the model's
  // vis.global.cameraid, and Follow Controller mirrors whatever camera the
  // controller uses), and mjv_moveCamera ignores fixed cameras. Coerce to a
  // free camera so the free modes always respond to input.
  if (g_app.camera.type == mjCAMERA_FIXED) {
    mjv_defaultFreeCamera(model, &g_app.camera);
    if (wasd) {
      mujoco::platform::SetCamera(model, &g_app.camera,
                                  mujoco::platform::kFreeCameraIdx);
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
    // Mouse scroll zooms towards/away from the lookat point; ignored by the
    // user-centered WASD camera which has no lookat point.
    const float mouse_scroll = io.MouseWheel / 50.0f;
    if (mouse_scroll != 0.0f && !wasd) {
      mjv_moveCamera(model, mjMOUSE_ZOOM, 0.f, -mouse_scroll, &g_app.camera);
    }
  }

  // WASD/QE flying, with the same accelerating speed as the studio app.
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

void BuildBrowserGui() {
  // TODO(matijak): Move these centered-text helpers into
  // platform/ux/imgui_widgets.cc.
  const auto centered_line = [](const char* text, const ImVec4* color) {
    ImGui::SetCursorPosX(std::max(
        0.0f, (ImGui::GetWindowWidth() - ImGui::CalcTextSize(text).x) * 0.5f));
    if (color != nullptr) {
      ImGui::TextColored(*color, "%s", text);
    } else {
      ImGui::TextUnformatted(text);
    }
  };
  // Large centered banner text (SPECTATING / CONTROLLING / DISCONNECTED).
  const auto centered_banner = [](const char* text, const ImVec4& color) {
    ImGui::SetWindowFontScale(1.6f);
    const float text_width = ImGui::CalcTextSize(text).x;
    ImGui::SetCursorPosX(
        std::max(0.0f, (ImGui::GetWindowWidth() - text_width) * 0.5f));
    ImGui::TextColored(color, "%s", text);
    ImGui::SetWindowFontScale(1.0f);
  };

  // Screen-centered DISCONNECTED notice: a big red banner over white
  // explanation lines. Shared by the deliberate-close and server-silence
  // notices, which never show at the same time (a close code suppresses the
  // silence notice).
  const auto disconnected_notice = [&](const char* window_id,
                                       std::initializer_list<const char*>
                                           lines) {
    const ImGuiIO& io = ImGui::GetIO();
    ImGui::SetNextWindowPos(
        ImVec2(io.DisplaySize.x * 0.5f, io.DisplaySize.y * 0.5f),
        ImGuiCond_Always, ImVec2(0.5f, 0.5f));
    ImGui::Begin(window_id, nullptr,
                 ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                     ImGuiWindowFlags_NoMove |
                     ImGuiWindowFlags_AlwaysAutoResize);
    centered_banner("DISCONNECTED", ImVec4(1.0f, 0.3f, 0.3f, 1.0f));
    for (const char* line : lines) {
      centered_line(line, nullptr);
    }
    ImGui::End();
  };

  if (const int close_code = g_state_link.ServerCloseCode()) {
    const char* reason = "Disconnected by the viewer.";
    if (close_code == 4002) {
      reason = "Session is full: too many viewers connected.";
    } else if (close_code == 4003) {
      reason = "Disconnected after inactivity.";
    }
    disconnected_notice("##disconnected_by_server",
                        {reason, "Retrying; reconnects automatically."});
  }

  const double now = ImGui::GetTime();
  if (now - g_app.telemetry.last_rate_time >= 1.0) {
    g_app.telemetry.gui_bytes_per_sec = g_ui_link.ConsumeByteCount();
    g_app.telemetry.sim_bytes_per_sec = g_state_link.ConsumeByteCount();
    g_app.telemetry.last_rate_time = now;
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
      g_state_link.ServerCloseCode() == 0 && !g_state_link.ReloadPending();
  if (!link_stale) {
    g_app.disconnected_notice_logged = false;
  } else {
    if (!g_app.disconnected_notice_logged) {
      LOG(Info, "Showing server-not-reachable notice");
      g_app.disconnected_notice_logged = true;
    }
    disconnected_notice(
        "##disconnected",
        {"Viewer server is not reachable: the Python script may",
         "have stopped. This page reconnects automatically if the",
         "viewer comes back."});
  }

  // Anchor points along the canvas edges, each with the pivot that keeps
  // the whole window on-screen 10px off the anchoring edges. Indexed by
  // LinkWindowPos - kLinkPosTopLeft. While anchored, the window is re-pinned
  // every frame (so it follows canvas resizes) for both the collapsed pill
  // and the expanded window (same ImGui window, so one call covers
  // whichever Begin runs this frame) — except while it is being dragged.
  constexpr float kEdgeMargin = 10.0f;
  const ImVec2 canvas = ImGui::GetIO().DisplaySize;
  const ImVec2 anchor_pos[6] = {
      {kEdgeMargin, kEdgeMargin},
      {canvas.x * 0.5f, kEdgeMargin},
      {canvas.x - kEdgeMargin, kEdgeMargin},
      {kEdgeMargin, canvas.y - kEdgeMargin},
      {canvas.x * 0.5f, canvas.y - kEdgeMargin},
      {canvas.x - kEdgeMargin, canvas.y - kEdgeMargin},
  };
  const ImVec2 anchor_pivot[6] = {
      {0.0f, 0.0f}, {0.5f, 0.0f}, {1.0f, 0.0f},
      {0.0f, 1.0f}, {0.5f, 1.0f}, {1.0f, 1.0f},
  };

  if (g_app.link_window_pos != kLinkPosFree && !g_app.link_dragging) {
    const int anchor = g_app.link_window_pos - kLinkPosTopLeft;
    ImGui::SetNextWindowPos(anchor_pos[anchor], ImGuiCond_Always,
                            anchor_pivot[anchor]);
  }

  // Tracks a drag of the Link window and snaps on release: if the window
  // was dropped within max(20% of its size, 100px) of where an anchor
  // would place it, adopt that anchor; otherwise it floats Free where it
  // was dropped. Call between Begin and End of whichever form is visible.
  const auto update_window_snap = [&] {
    if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) &&
        ImGui::IsWindowHovered(
            ImGuiHoveredFlags_AllowWhenBlockedByActiveItem)) {
      g_app.link_dragging = true;
    } else if (g_app.link_dragging &&
               !ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
      g_app.link_dragging = false;
      const ImVec2 pos = ImGui::GetWindowPos();
      const ImVec2 size = ImGui::GetWindowSize();
      const float threshold = std::max(0.2f * std::max(size.x, size.y), 100.0f);
      int best = -1;
      float best_d2 = threshold * threshold;
      for (int i = 0; i < 6; ++i) {
        // Per-axis distance to where this anchor would place the window,
        // clamped to zero on the anchor's off-screen side: a window dropped
        // past an edge or corner (mostly off-screen) still snaps to that
        // anchor instead of being left stranded outside the canvas.
        float dx = pos.x - (anchor_pos[i].x - anchor_pivot[i].x * size.x);
        if (anchor_pivot[i].x == 0.0f) {
          dx = std::max(dx, 0.0f);  // Dropped past the left edge.
        } else if (anchor_pivot[i].x == 1.0f) {
          dx = std::min(dx, 0.0f);  // Dropped past the right edge.
        }
        float dy = pos.y - (anchor_pos[i].y - anchor_pivot[i].y * size.y);
        if (anchor_pivot[i].y == 0.0f) {
          dy = std::max(dy, 0.0f);  // Dropped past the top edge.
        } else {
          dy = std::min(dy, 0.0f);  // Dropped past the bottom edge.
        }
        const float d2 = dx * dx + dy * dy;
        if (d2 <= best_d2) {
          best_d2 = d2;
          best = i;
        }
      }
      g_app.link_window_pos = best >= 0 ? kLinkPosTopLeft + best : kLinkPosFree;
    }
  };

  const ImVec2 kFullWidth(-FLT_MIN, 0.0f);
  const ImVec4 kSpectatingColor(1.0f, 0.75f, 0.2f, 1.0f);
  const ImVec4 kControllingColor(0.3f, 0.9f, 0.4f, 1.0f);
  const ImVec4 kQueueColor(1.0f, 0.62f, 0.15f, 1.0f);

  // Data rates, shown to both roles (spectators receive no GUI stream, so
  // that line reads 0 for them). With more than one viewer, the number in
  // parentheses is the host's total outgoing sim bandwidth: every connected
  // browser receives the same sim stream.
  const auto data_rate_lines = [] {
    ImGui::Text(
        "GUI Data Rate: %" PRIu64 " KiB/s",
        static_cast<uint64_t>(g_app.telemetry.gui_bytes_per_sec / 1024));
    if (g_app.spectator) {
      ImGui::SetItemTooltip("The UI is not streamed to spectators.");
    } else if (!g_ui_link.RemoteDrawData()) {
      ImGui::SameLine();
      ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f),
                         "(Waiting...)");
    }
    ImGui::Text(
        "Sim Data Rate: %" PRIu64 " KiB/s",
        static_cast<uint64_t>(g_app.telemetry.sim_bytes_per_sec / 1024));
    if (g_app.session_viewers > 1) {
      ImGui::SameLine();
      ImGui::Text("(%" PRIu64 " KiB/s)",
                  static_cast<uint64_t>(g_app.telemetry.sim_bytes_per_sec *
                                        g_app.session_viewers / 1024));
      ImGui::SetItemTooltip("Total sim data sent across all viewers.");
    }
  };

  // While someone waits for control, the window background pulses toward a
  // dark orange (the queue size itself is shown in the expanded window).
  const bool pulse = !g_app.spectator && g_app.queue_len > 0;
  if (pulse) {
    const float phase =
        0.5f + 0.5f * sinf(static_cast<float>(ImGui::GetTime()) * 2.5f);
    ImVec4 bg = ImGui::GetStyleColorVec4(ImGuiCol_WindowBg);
    const ImVec4 orange(0.45f, 0.22f, 0.02f, bg.w);
    const float blend = 0.6f * phase;
    bg.x += (orange.x - bg.x) * blend;
    bg.y += (orange.y - bg.y) * blend;
    bg.z += (orange.z - bg.z) * blend;
    ImGui::PushStyleColor(ImGuiCol_WindowBg, bg);
  }

  if (!g_app.link_expanded) {
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(6.0f, 6.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowMinSize, ImVec2(16.0f, 16.0f));

    ImGui::Begin("Link", nullptr,
                 ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                     ImGuiWindowFlags_AlwaysAutoResize);
    if (g_app.spectator) {
      ImGui::TextColored(kSpectatingColor, "SPECTATING");
    } else {
      ImGui::TextColored(kControllingColor, "CONTROLLING");
    }
    if (ImGui::IsWindowHovered(
            ImGuiHoveredFlags_AllowWhenBlockedByActiveItem)) {
      g_app.link_expanded = true;
    }
    update_window_snap();
    ImGui::End();

    ImGui::PopStyleVar(2);
  } else {
    // No title bar; the window is still movable by dragging empty space
    // (ImGui's default when ConfigWindowsMoveFromTitleBarOnly is off).
    ImGui::Begin(
        "Link", nullptr,
        ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar);

    if (g_app.spectator) {
      // Control group.
      centered_banner("SPECTATING", kSpectatingColor);
      char queue_text[64];
      if (g_app.queue_pos > 0) {
        snprintf(queue_text, sizeof(queue_text),
                 "Control Queue: Position %d of %d", g_app.queue_pos,
                 g_app.queue_len);
      } else if (g_app.queue_len > 0) {
        snprintf(queue_text, sizeof(queue_text), "Control Queue: %d waiting",
                 g_app.queue_len);
      } else {
        snprintf(queue_text, sizeof(queue_text), "Control Queue: (empty)");
      }
      centered_line(queue_text, nullptr);
      if (g_app.queue_pos == 0) {
        if (ImGui::Button("Request control", kFullWidth)) {
          g_state_link.SendText(kMsgRequestControl);
        }
      } else {
        const float half_width = (ImGui::GetContentRegionAvail().x -
                                  ImGui::GetStyle().ItemSpacing.x) *
                                 0.5f;
        if (ImGui::Button("Leave Queue", ImVec2(half_width, 0.0f))) {
          g_state_link.SendText(kMsgLeaveQueue);
        }
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_Button,
                              ImVec4(0.65f, 0.15f, 0.15f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                              ImVec4(0.80f, 0.20f, 0.20f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive,
                              ImVec4(0.50f, 0.10f, 0.10f, 1.0f));
        if (ImGui::Button("Take Control", ImVec2(half_width, 0.0f))) {
          g_state_link.SendText(kMsgForceControl);
        }
        ImGui::PopStyleColor(3);
        ImGui::SetItemTooltip("Takes control immediately, jumping the queue.");
      }

      // Data rate group.
      ImGui::Separator();
      data_rate_lines();

      // Camera group.
      ImGui::Separator();
      // TODO(matijak): Use the studio camera-selection UI here in future, so
      // a spectator can also pick any camera defined in the model.
      // Sized to its longest option; the default width would stretch the
      // window past the CONTROLLING layout's size.
      ImGui::SetNextItemWidth(ImGui::CalcTextSize("Follow Controller").x +
                              ImGui::GetFrameHeight() * 2.0f);
      int cam_mode = g_app.spectator_cam_mode;
      if (ImGui::Combo("Camera", &cam_mode,
                       "Free: tumble\0Free: wasd\0Follow Controller\0")) {
        SetSpectatorCameraMode(cam_mode);
      }
    } else {
      // Control group.
      centered_banner("CONTROLLING", kControllingColor);
      if (g_app.queue_len > 0) {
        char queue_text[64];
        snprintf(queue_text, sizeof(queue_text),
                 ">> Control Queue: %d waiting <<", g_app.queue_len);
        centered_line(queue_text, &kQueueColor);
      } else {
        centered_line("Control Queue: (empty)", nullptr);
      }
      if (ImGui::Button("Release control", kFullWidth)) {
        // Become a spectator; the server grants the slot down the queue.
        g_ui_link.Shutdown();
        SetSpectator(true);
      }

      // Data rate group.
      ImGui::Separator();
      data_rate_lines();

      // Session settings group.
      ImGui::Separator();
      // Local edits win for a grace period (the roster round trip takes a
      // few frames and typing takes longer); afterwards the server's value
      // is the truth.
      if (g_app.max_spectators_edit < 0 ||
          ImGui::GetTime() - g_app.max_spectators_edit_time > 1.5) {
        g_app.max_spectators_edit = g_app.max_spectators;
      }
      ImGui::SetNextItemWidth(100.0f);
      if (ImGui::InputInt("Max Spectators", &g_app.max_spectators_edit)) {
        g_app.max_spectators_edit_time = ImGui::GetTime();
        g_app.max_spectators_edit =
            std::clamp(g_app.max_spectators_edit, 0, 32);
        char msg[48];
        snprintf(msg, sizeof(msg), "%s%d", kMsgMaxSpectatorsPrefix,
                 g_app.max_spectators_edit);
        g_state_link.SendText(msg);
      }
      if (ImGui::IsItemActive()) {
        g_app.max_spectators_edit_time = ImGui::GetTime();
      }
    }

    update_window_snap();

    // Collapse when the mouse leaves the window, with a short grace period.
    // A popup (e.g. the camera combo's dropdown) is a separate window, so
    // moving the mouse into it unhovers this one; keep the window expanded
    // while one of its popups is open, and give brief excursions time to
    // come back before snapping shut.
    constexpr double kCollapseGraceSec = 0.2;
    const bool popup_open = ImGui::IsPopupOpen(
        "", ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel);
    if (popup_open || ImGui::IsWindowHovered(
                          ImGuiHoveredFlags_AllowWhenBlockedByActiveItem)) {
      g_app.link_hover_time = ImGui::GetTime();
    } else if (ImGui::GetTime() - g_app.link_hover_time > kCollapseGraceSec) {
      g_app.link_expanded = false;
    }
    ImGui::End();
  }
  if (pulse) {
    ImGui::PopStyleColor();
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
  g_app.frame_count++;

  // Heartbeat on the session channel. A hidden tab's rendering loop stops,
  // so the heartbeat stops with it — the server kicks spectators (and
  // releases a controller with a waiting queue) on silence.
  if (g_app.frame_count - g_app.last_heartbeat_frame >= 30 * 60) {
    g_app.last_heartbeat_frame = g_app.frame_count;
    g_state_link.SendText(kMsgHeartbeat);
  }

  // Reconnect the state WebSocket if it dropped — e.g. the Python side
  // restarted its servers after a model change. Receiving a payload with a
  // different model identity then triggers a page reload (StateLink).
  // Deliberate server closes (session full, inactivity) are transient;
  // retry them too, just at a gentler pace.
  const int state_retry_interval =
      g_state_link.ServerCloseCode() != 0 ? 300 : 60;
  if (!g_state_link.HasSocket() && !g_state_link.ReloadPending() &&
      g_app.model_holder && g_app.model_holder->ok() &&
      g_app.frame_count - g_app.last_state_retry_frame > state_retry_interval) {
    g_app.last_state_retry_frame = g_app.frame_count;
    LOG(Info, "State WebSocket down; reconnecting...");
    g_state_link.Connect(WsUrl("/state"));
  }

  // Reconnect the NetImgui UI socket too — the proxy tears the bridge down
  // whenever its headless-side TCP connection cycles (server restart). A
  // close code 4001 means another browser holds the controller slot: after a
  // few retries (a page reload of the controller races its own slot briefly)
  // this page settles into spectating.
  if (g_ui_link.HasSocket() && !g_app.spectator &&
      !g_state_link.ReloadPending() && g_state_link.ServerCloseCode() == 0 &&
      g_app.frame_count - g_app.last_ui_retry_frame > 60) {
    const UiLink::ReadyState uiState = g_ui_link.ConnectionState();
    if (uiState == UiLink::ReadyState::kOpen) {
      g_app.ui_reject_count = 0;
    } else if (uiState == UiLink::ReadyState::kClosed ||
               uiState == UiLink::ReadyState::kError) {
      g_app.last_ui_retry_frame = g_app.frame_count;
      if (g_ui_link.CloseCode() == 4001 && ++g_app.ui_reject_count >= 3) {
        LOG(Info, "Controller slot taken; spectating");
        SetSpectator(true);
        // Also drops the last received UI frame: a page forced out of the
        // controller slot must not keep showing a frozen Studio UI.
        g_ui_link.Shutdown();
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
  g_ui_link.ReceiveAndProcessCommands(g_app.frame_count);

  // Event loop and ImGui NewFrame via window abstraction.
  mujoco::platform::Window::Status status = g_app.window->NewFrame();
  if (status == mujoco::platform::Window::kQuitting) {
    // NewFrame() started an ImGui frame — end it before bailing out.
    ImGui::EndFrame();
    emscripten_cancel_main_loop();
    return;
  }

  // For the controller, all scene interaction (camera orbit/zoom, perturbation,
  // picking) is handled by the headless viewer: CaptureAndSendInput()
  // forwards this frame's input over NetImgui, the headless ViewerApp runs
  // the same event handlers as the native viewer, and the resulting
  // camera/perturb state streams back over the state WebSocket (see
  // ApplyStatePayload). Spectators have no input channel; in a free camera
  // mode they drive their local camera directly.
  g_ui_link.CaptureAndSendInput();
  if (g_app.spectator) {
    HandleSpectatorCameraInput();
  }

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
  if (remoteDrawData && remoteDrawData->Valid && !g_app.spectator) {
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
    mujoco::platform::SetCamera(m, &g_app.camera, model_cam);
  } else {
    mjv_defaultFreeCamera(m, &g_app.camera);
  }
}

// Standard CRC-32 (ISO-HDLC, poly 0xEDB88320), matching Python's zlib.crc32
// so the fetched model's checksum can be compared against the crc the Python
// side stamps into each state payload.
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
    // Baseline the state link on the model we just fetched, so a model swap
    // that raced this fetch is caught by the very first payload.
    g_state_link.SetModelCrc32(
        Crc32(reinterpret_cast<const uint8_t*>(fetch->data),
              static_cast<size_t>(fetch->numBytes)));
    // Connect the state WebSocket to receive simulation state from Python.
    g_state_link.Connect(WsUrl("/state"));
  } else {
    LOG(Error, "Failed to load model: %s",
        g_app.model_holder ? g_app.model_holder->error().data()
                           : "Unknown error");
  }
  emscripten_fetch_close(fetch);
}

void StartModelFetch();

void OnFetchError(emscripten_fetch_t* fetch) {
  LOG(Error, "Failed to fetch model.mjb, status: %d; retrying", fetch->status);
  emscripten_fetch_close(fetch);
  // The most likely cause is the Python side restarting its server after a
  // model change, so retry rather than leaving the page permanently blank
  // (nothing else re-triggers the fetch — the state link is only connected
  // on fetch success).
  emscripten_async_call([](void*) { StartModelFetch(); }, nullptr, 1000);
}

void StartModelFetch() {
  emscripten_fetch_attr_t attr;
  emscripten_fetch_attr_init(&attr);
  strcpy(attr.requestMethod, "GET");
  attr.attributes = EMSCRIPTEN_FETCH_LOAD_TO_MEMORY;
  attr.onsuccess = OnFetchSuccess;
  attr.onerror = OnFetchError;
  // Relative URL — the browser resolves this against the page origin,
  // so it works regardless of hostname, protocol, or port.
  emscripten_fetch(&attr, "/model.mjb");
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

// Registers resource providers so that "filament:" (renderer materials/IBL)
// and "font:" (the local ImGui UI fonts) asset requests resolve to the
// preloaded /assets files. Both prefixes share the same loader — LoadAsset
// maps either to /assets/<name>.
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
  for (const char* prefix : {"filament", "font"}) {
    resource_provider.prefix = prefix;
    mjp_registerResourceProvider(&resource_provider);
  }
}

int main(int argc, char** argv) {
  RegisterAssetProviders();

  mujoco::platform::Window::Config config;
  config.gfx_mode = mujoco::platform::GraphicsMode::FilamentWebGl;
  // Load the Studio UI fonts for the browser's local ImGui (the Link window);
  // the "font:" resource provider above resolves them from the preloaded
  // /assets. (The streamed Studio UI carries its own font atlas separately.)
  config.load_fonts = true;

  g_app.window = std::make_unique<mujoco::platform::Window>("MuJoCo Web Viewer",
                                                            1400, 720, config);
  ImPlot::CreateContext();  // Needed if the server app uses ImPlot.

  g_app.renderer = new mujoco::platform::Renderer(
      g_app.window->GetNativeWindowHandle(), config.gfx_mode);

  NetImgui::Internal::Network::Startup();
  SDL_StartTextInput();

  // The UI stream is a path on the page's own host and port.
  g_ui_link.Connect(WsUrl("/ui"));

  StartModelFetch();

  emscripten_set_main_loop(MainLoop, 0, 1);

  // Cleanup (reached if emscripten loop exits).
  g_ui_link.Shutdown();
  NetImgui::Internal::Network::Shutdown();

  g_app.window.reset();

  return 0;
}
