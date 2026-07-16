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

// State link: receives simulation state payloads from the Python StateServer
// over the /state WebSocket and owns the model-identity / page-reload /
// supersede policy. Applying a payload to the app is delegated to the
// on_payload callback, so this file stays free of scene and renderer
// dependencies.

#ifndef MUJOCO_PYTHON_EXPERIMENTAL_STUDIO_WEB_STATE_LINK_H_
#define MUJOCO_PYTHON_EXPERIMENTAL_STUDIO_WEB_STATE_LINK_H_

#include <cstdint>
#include <functional>
#include <string>

#include <emscripten/websocket.h>

#include "render_state.h"

namespace mujoco::studio {

class StateLink {
 public:
  // Payloads are dropped until this returns true (model loaded).
  using ReadyFn = std::function<bool()>;
  // Applies a parsed payload to the app (physics + render state + geoms).
  using OnPayloadFn = std::function<void(const StatePayloadView&)>;

  StateLink(ReadyFn ready, OnPayloadFn on_payload)
      : ready_(std::move(ready)), on_payload_(std::move(on_payload)) {}

  void Connect(const std::string& url);

  // True while a connect attempt exists (possibly still in flight); used to
  // pace reconnects. emscripten_websocket_new returns a handle immediately,
  // so this is NOT the same as Connected().
  bool HasSocket() const { return socket_ != 0; }

  // True only while the WebSocket is actually open.
  bool Connected() const { return open_; }

  // True once a payload with a new model identity has scheduled a page
  // reload; all traffic is dropped from then on.
  bool ReloadPending() const { return reload_pending_; }

  // The close code when the server deliberately ended this connection
  // (codes 4000-4999, e.g. 4002 = session full), else 0. On such a close
  // both reconnect loops must stop and the page shows a notice instead.
  int TerminalCloseCode() const { return terminal_close_code_; }

  // Returns the bytes received since the last call and resets the counter.
  uint64_t ConsumeByteCount() {
    uint64_t bytes = bytes_accum_;
    bytes_accum_ = 0;
    return bytes;
  }

  // Wall-clock seconds (emscripten_get_now-based) of the last received
  // message, or 0 before the first one. Payloads stream at ~60Hz while the
  // Python side is alive, so staleness here means the server is gone —
  // even if the socket still looks open (a suspended process keeps its
  // sockets established).
  double LastMessageTime() const { return last_message_time_; }

  // Parses one WebSocket message and applies the identity/reload policy;
  // public for tests.
  void HandleMessage(const uint8_t* data, uint32_t num_bytes);

 private:
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

  ReadyFn ready_;
  OnPayloadFn on_payload_;

  EMSCRIPTEN_WEBSOCKET_T socket_ = 0;
  bool open_ = false;

  // Identity of the model this page loaded (adopted from the first payload).
  // When the payload's identity changes, the Python side has swapped models:
  // reload the page, which refetches /model.mjb and reconnects everything.
  uint32_t model_ident_ = 0;
  bool have_model_ident_ = false;
  bool reload_pending_ = false;

  int terminal_close_code_ = 0;

  uint64_t bytes_accum_ = 0;
  double last_message_time_ = 0;
};

}  // namespace mujoco::studio

#endif  // MUJOCO_PYTHON_EXPERIMENTAL_STUDIO_WEB_STATE_LINK_H_
