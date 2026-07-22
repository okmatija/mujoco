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

// The session: this page's relationship with the Python-side viewer.
//
// Owns the /state WebSocket (simulation payloads in, control messages out),
// the session wire protocol (roster, grants, heartbeats, acks), the role
// state machine (claiming -> controlling / spectating), and the
// model-change / page-reload / close-code policies. Everything outside the
// session goes through the Callbacks interface: applying a payload to the
// scene, and driving the remote UI stream on role transitions. The session
// also implements SessionActions, so the role window's intents land here
// directly.

#ifndef MUJOCO_PYTHON_EXPERIMENTAL_STUDIO_WEB_WEB_CLIENT_SESSION_H_
#define MUJOCO_PYTHON_EXPERIMENTAL_STUDIO_WEB_WEB_CLIENT_SESSION_H_

#include <emscripten/websocket.h>

#include <cstdint>
#include <string>

#include "state_payload.h"

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

// Read-only snapshot of the session, passed to the local UI each frame.
// The session fills the role and roster fields (FillView); the byte rates,
// remote-frame flag and camera mode are the app's.
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

// User intent reported by the role window. Session implements this; the
// interface is the complete list of effects the local UI can cause.
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

// The roster: the server's membership broadcast, sent as a text frame on
// /state whenever the session changes (a viewer joins or leaves, queues
// for control, or control moves). It tells this page how many viewers are
// connected, which role the server currently assigns it, and where it
// stands in the control queue.
struct Roster {
  int viewers = 0;
  bool spectator = false;  // The server's view: true = not the controller.
  int queue_pos = 0;  // 1-based position in the control queue; 0 = unqueued.
  int queue_len = 0;
  // Runtime spectator limit. Starts at the UI default; every parsed
  // roster overwrites it.
  int max_spectators = 8;
};

// Parses a roster line; returns false when text is not a roster. (Keep
// in sync: _roster_line in web_server.py.)
bool ParseRoster(const char* text, Roster* roster);

// The remote UI stream's connection state, reported to the role state
// machine by the app once per frame (see Session::HandleRemoteUiState).
enum class RemoteUiState {
  kNoSocket = 0,   // No connection attempt exists.
  kConnecting,     // In flight (or closing); the machine waits.
  kOpen,           // The claim succeeded: this page controls.
  kClosedOrError,  // Rejected or dropped; the machine retries or settles.
};

class Session : public SessionActions {
 public:
  // Everything the session needs from the rest of the app; the app
  // implements this once.
  class Callbacks {
   public:
    virtual ~Callbacks() = default;
    // Payloads are dropped until this returns true (model loaded).
    virtual bool ReadyForPayload() = 0;
    // Applies a parsed payload to the app (physics + render state + geoms).
    virtual void OnPayload(const StatePayloadView& view) = 0;
    // Role transitions drive the remote UI stream through these two:
    // (re)claim the controller slot / drop the stream when settling into
    // spectating.
    virtual void ConnectRemoteUi() = 0;
    virtual void ShutdownRemoteUi() = 0;
    // The one role-window intent that is not session business.
    virtual void SetCameraMode(int mode) = 0;
  };

  explicit Session(Callbacks& callbacks) : callbacks_(callbacks) {}

  void Connect(const std::string& url);

  // Records the CRC32 of the model this page actually loaded (from the
  // fetched /model.mjb bytes), so the first payload already reveals a model
  // that changed between the fetch and the first /state frame. Without this
  // the baseline is adopted from the first payload and such a race is never
  // detected. Must match zlib.crc32 (web_viewer.py's model_crc32).
  void SetModelCrc32(uint32_t crc) {
    model_crc32_ = crc;
    have_model_crc32_ = true;
  }

  // True while a connect attempt exists (possibly still in flight); used to
  // pace reconnects. emscripten_websocket_new returns a handle immediately,
  // so this is NOT the same as Connected().
  bool HasSocket() const { return socket_ != 0; }

  // True only while the WebSocket is actually open.
  bool Connected() const { return open_; }

  // True once a payload with a new model has scheduled a page reload; all
  // traffic is dropped from then on.
  bool ReloadPending() const { return reload_pending_; }

  // The close code from the server deliberately ending this connection
  // (codes 4000-4999, e.g. 4002 = session full), else 0. Such conditions
  // are transient (a slot frees up, the user returns to the tab), so the
  // page shows a notice and retries slowly; the code clears when a
  // connection opens again.
  int ServerCloseCode() const { return server_close_code_; }

  // Returns the bytes received since the last call and resets the counter.
  uint64_t ConsumeByteCount() {
    uint64_t bytes = bytes_accum_;
    bytes_accum_ = 0;
    return bytes;
  }

  // Wall-clock seconds of the last received message, or 0 before the first
  // one. Payloads stream at ~60Hz while the Python side is alive, so
  // staleness here means the server is gone — even if the socket still
  // looks open (a suspended process keeps its sockets established).
  double LastMessageTime() const { return last_message_time_; }

  SessionRole Role() const { return role_; }

  // Fills the role and roster fields of the view; the rest is the app's.
  void FillView(SessionView* view) const;

  // Periodic session upkeep (the ~30s liveness heartbeat); call once per
  // frame.
  void Update();

  // Feeds the role state machine the remote UI stream's connection state;
  // call once per frame. Owns claim retry pacing, promotion to kControlling
  // when a claim opens, instant settling when an open stream closes with
  // 4001 (ousted by Steal Control), and the retries-then-settle rule for
  // rejected claims.
  void HandleRemoteUiState(RemoteUiState state, int close_code);

  // Parses one WebSocket message and applies the model-change/reload
  // policy; public for tests.
  void HandleMessage(const uint8_t* data, uint32_t num_bytes);

  // SessionActions (the role window reports intent straight into the
  // session; see the app's RoleWindow::Draw call).
  void RequestControl() override;
  void LeaveQueue() override;
  void StealControl() override;
  void ReleaseControl() override;
  void SetCameraMode(int mode) override;
  void SetMaxSpectators(int count) override;

 private:
  // Detaches callbacks and frees socket_ (if any), resetting to disconnected.
  void CloseSocket();

  // Sends a session message (control requests, acks, activity reports) to
  // the server as a text frame. Dropped silently while not connected.
  void SendText(const char* text);

  // Updates the role and mirrors it into JS (Module.isSpectator), which
  // gates controller-only page behavior (model drag-and-drop upload).
  void SetRole(SessionRole role);

  // Routes a session text frame: roster updates, control grants.
  void OnSessionText(const char* text);

  static EM_BOOL OnWsMessage(int event_type,
                             const EmscriptenWebSocketMessageEvent* event,
                             void* user_data);
  static EM_BOOL OnWsOpen(int event_type,
                          const EmscriptenWebSocketOpenEvent* event,
                          void* user_data);
  static EM_BOOL OnWsError(int event_type,
                           const EmscriptenWebSocketErrorEvent* event,
                           void* user_data);
  static EM_BOOL OnWsClose(int event_type,
                           const EmscriptenWebSocketCloseEvent* event,
                           void* user_data);

  Callbacks& callbacks_;

  EMSCRIPTEN_WEBSOCKET_T socket_ = 0;
  bool open_ = false;

  // CRC32 of the model this page loaded (adopted from the first payload).
  // When the payload's crc changes, the Python side has swapped models:
  // reload the page, which refetches /model.mjb and reconnects everything.
  uint32_t model_crc32_ = 0;
  bool have_model_crc32_ = false;
  bool reload_pending_ = false;

  int server_close_code_ = 0;

  uint64_t bytes_accum_ = 0;
  double last_message_time_ = 0;

  // Role state machine + roster (all from the session text channel and
  // HandleRemoteUiState; see web_server.py for the wire formats).
  SessionRole role_ = SessionRole::kClaiming;
  Roster roster_;
  // Consecutive rejected /ui claims; after enough, the page stops claiming
  // and settles into spectating.
  int ui_reject_count_ = 0;
  RemoteUiState remote_ui_state_ = RemoteUiState::kNoSocket;
  double last_ui_retry_time_ = 0;
  double last_heartbeat_time_ = 0;
};

}  // namespace mujoco::studio

#endif  // MUJOCO_PYTHON_EXPERIMENTAL_STUDIO_WEB_WEB_CLIENT_SESSION_H_
