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

// The web viewer's local (browser-drawn) UI: the role window and the
// DISCONNECTED notices. Deliberately decoupled from the app: it reads a
// SessionView snapshot and reports user intent through the SessionActions
// interface, so it depends only on ImGui and can in principle be exercised
// natively.

#ifndef MUJOCO_PYTHON_EXPERIMENTAL_STUDIO_WEB_WEB_CLIENT_LOCAL_UI_H_
#define MUJOCO_PYTHON_EXPERIMENTAL_STUDIO_WEB_WEB_CLIENT_LOCAL_UI_H_

#include <imgui.h>

#include <cstdint>

namespace mujoco::studio {

// The page's role in the collaborative session. Every page starts by
// claiming the controller slot; the claim either succeeds (kControlling)
// or the page settles into spectating. A control grant puts a spectator
// back into kClaiming while it reconnects to /ui.
enum class SessionRole {
  kClaiming = 0,  // /ui claim in flight; the role is not yet resolved.
  kControlling,   // This page holds the open /ui connection.
  kSpectating,    // Another page controls; scene + local role window only.
};

// Read-only snapshot of the session, passed to the UI each frame.
struct SessionView {
  SessionRole role = SessionRole::kClaiming;
  int viewers = 0;
  int queue_pos = 0;  // 1-based position in the control queue; 0 = unqueued.
  int queue_len = 0;
  int max_spectators = 0;
  uint64_t gui_bytes_per_sec = 0;
  uint64_t sim_bytes_per_sec = 0;
  // False until the first remote Studio UI frame arrived (controller only).
  bool have_remote_frame = false;
  int camera_mode = 0;  // A SpectatorCamMode value (see web_client.cc).
};

// User intent reported by the role window; the app implements this once.
// The interface is the complete list of effects the local UI can cause.
class SessionActions {
 public:
  virtual ~SessionActions() = default;
  virtual void RequestControl() = 0;
  virtual void LeaveQueue() = 0;
  virtual void StealControl() = 0;
  virtual void ReleaseControl() = 0;
  virtual void SetCameraMode(int mode) = 0;      // A SpectatorCamMode value.
  virtual void SetMaxSpectators(int count) = 0;  // Already clamped by the UI.
};

// The screen-centered DISCONNECTED notices: a big red banner over white
// explanation lines, shown for deliberate server closes (session full,
// inactivity) and for server silence. The two never show at the same time:
// a close code suppresses the silence notice.
class DisconnectNotice {
 public:
  // seconds_since_last_payload is negative before the first payload.
  void Draw(int server_close_code, double seconds_since_last_payload,
            bool reload_pending);

 private:
  // How long the state stream (~60Hz while the Python side is alive) may go
  // silent before the notice appears. A model-change restart resumes
  // traffic in about a second and should not flash the notice.
  static constexpr double kServerSilenceNoticeSec = 3.0;

  // True while the server-not-reachable notice is up (log once per outage).
  bool logged_ = false;
};

// The role window: a collapsed role pill that expands on hover into the
// session panel (role banner, control queue, data rates, per-role
// settings). Owns its presentation state — mode, intro/hover timers, drag
// and snap anchoring, edit grace.
class RoleWindow {
 public:
  void Draw(const SessionView& view, SessionActions& actions);

 private:
  // Presentation state machine: the window starts in the intro (expanded
  // for a few seconds so new users see it exists and learn it expands on
  // hover); the first hover ends the intro.
  enum class Mode {
    kIntro = 0,  // Expanded on page load, until the timer or first hover.
    kExpanded,   // Expanded while hovered (plus a short leave grace).
    kCollapsed,  // The role pill; expands on hover.
  };

  // Where the window is anchored on the page. It is always draggable;
  // releasing a drag near an anchor snaps to it, anywhere else leaves the
  // window Free at its dropped position.
  enum Anchor {
    kAnchorFree = 0,
    kAnchorTopLeft,  // The anchors must stay contiguous from here on.
    kAnchorTopMid,
    kAnchorTopRight,
    kAnchorBottomLeft,
    kAnchorBottomMid,
    kAnchorBottomRight,
  };

  static constexpr double kCollapseGraceSec = 0.2;
  static constexpr double kIntroExpandedSec = 5.0;
  static constexpr float kEdgeMargin = 10.0f;
  // Pivots keeping the whole window on-screen kEdgeMargin off the
  // anchoring edges; indexed by Anchor - kAnchorTopLeft, like anchor_pos_.
  static constexpr ImVec2 kAnchorPivot[6] = {
      {0.0f, 0.0f}, {0.5f, 0.0f}, {1.0f, 0.0f},
      {0.0f, 1.0f}, {0.5f, 1.0f}, {1.0f, 1.0f},
  };

  void ComputeAnchorPositions();
  void UpdateSnap();
  void UpdateCollapse();
  static void DataRateLines(const SessionView& view);
  void DrawCollapsed(const SessionView& view);
  void DrawExpanded(const SessionView& view, SessionActions& actions);
  void DrawSpectatorContents(const SessionView& view, SessionActions& actions);
  void DrawControllerContents(const SessionView& view, SessionActions& actions);

  Mode mode_ = Mode::kIntro;
  Anchor anchor_ = kAnchorTopMid;
  ImVec2 anchor_pos_[6];    // Recomputed each frame (canvas resizes).
  double intro_start_ = 0;  // 0 until the expanded window first draws.
  double hover_time_ = 0;   // Feeds the collapse grace period.
  bool dragging_ = false;
  // Max Spectators edit grace: the local value wins over the roster until
  // max_spectators_edit_time_ is old enough (see the InputInt).
  int max_spectators_edit_ = -1;
  double max_spectators_edit_time_ = -1e9;
};

}  // namespace mujoco::studio

#endif  // MUJOCO_PYTHON_EXPERIMENTAL_STUDIO_WEB_WEB_CLIENT_LOCAL_UI_H_
