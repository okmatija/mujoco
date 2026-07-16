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

#include "state_link.h"

#include <emscripten.h>

#include "google/logging.h"

namespace mujoco::studio {

EM_BOOL StateLink::OnWsMessage(int event_type,
                               const EmscriptenWebSocketMessageEvent* event,
                               void* user_data) {
  auto* link = static_cast<StateLink*>(user_data);
  if (event->isText) {
    // Text frames carry session metadata; emscripten null-terminates them.
    if (link->on_session_message_) {
      link->on_session_message_(reinterpret_cast<const char*>(event->data));
    }
  } else {
    link->HandleMessage(event->data, event->numBytes);
  }
  return EM_TRUE;
}

EM_BOOL StateLink::OnWsOpen(int event_type,
                            const EmscriptenWebSocketOpenEvent* event,
                            void* user_data) {
  static_cast<StateLink*>(user_data)->open_ = true;
  LOG(Info, "State WebSocket connected");
  return EM_TRUE;
}

EM_BOOL StateLink::OnWsError(int event_type,
                             const EmscriptenWebSocketErrorEvent* event,
                             void* user_data) {
  LOG(Error, "State WebSocket error");
  return EM_TRUE;
}

EM_BOOL StateLink::OnWsClose(int event_type,
                             const EmscriptenWebSocketCloseEvent* event,
                             void* user_data) {
  auto* link = static_cast<StateLink*>(user_data);
  LOG(Info, "State WebSocket closed (code=%d)", event->code);
  link->socket_ = 0;
  link->open_ = false;
  // Codes 4000-4999 are deliberate server-side closes (e.g. 4002 =
  // session full): stop reconnecting and let the GUI show a notice.
  if (event->code >= 4000 && event->code <= 4999) {
    link->terminal_close_code_ = event->code;
    LOG(Info, "Server ended this connection (code=%d); not reconnecting.",
        event->code);
  }
  return EM_TRUE;
}

void StateLink::Connect(const std::string& url) {
  EmscriptenWebSocketCreateAttributes attr;
  emscripten_websocket_init_create_attributes(&attr);
  attr.url = url.c_str();
  attr.protocols = nullptr;
  attr.createOnMainThread = EM_TRUE;

  socket_ = emscripten_websocket_new(&attr);
  if (socket_ <= 0) {
    LOG(Error, "Failed to create state WebSocket");
    return;
  }
  emscripten_websocket_set_onopen_callback(socket_, this, OnWsOpen);
  emscripten_websocket_set_onmessage_callback(socket_, this, OnWsMessage);
  emscripten_websocket_set_onerror_callback(socket_, this, OnWsError);
  emscripten_websocket_set_onclose_callback(socket_, this, OnWsClose);
  LOG(Info, "State WebSocket connecting to %s", url.c_str());
}

void StateLink::HandleMessage(const uint8_t* data, uint32_t num_bytes) {
  last_message_time_ = emscripten_get_now() / 1000.0;
  if (reload_pending_ || (ready_ && !ready_())) {
    return;
  }
  bytes_accum_ += num_bytes;

  StatePayloadView view;
  if (!ParseStatePayload(data, num_bytes, &view)) {
    LOG(Error, "Malformed state payload (%u bytes); dropping", num_bytes);
    return;
  }

  if (!have_model_ident_) {
    model_ident_ = view.model_ident;
    have_model_ident_ = true;
  } else if (view.model_ident != model_ident_) {
    LOG(Info, "Model changed on the Python side (ident %u -> %u); reloading",
        model_ident_, view.model_ident);
    reload_pending_ = true;
    EM_ASM({ setTimeout(function() { location.reload(); }, 0); });
    return;
  }

  if (on_payload_) {
    on_payload_(view);
  }
}

}  // namespace mujoco::studio
