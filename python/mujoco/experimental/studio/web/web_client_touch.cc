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

#include "web_client_touch.h"

#include <algorithm>
#include <cmath>

#include <SDL.h>
#include <SDL_events.h>
#include <SDL_hints.h>
#include <imgui.h>

namespace mujoco::studio {

void TouchInput::DisableSdlTouchMouseEmulation() {
  SDL_SetHint(SDL_HINT_TOUCH_MOUSE_EVENTS, "0");
}

void TouchInput::PumpEvents() {
  SDL_PumpEvents();
  SDL_Event events[32];
  int count;
  // The [SDL_FINGERDOWN, SDL_FINGERMOTION] range covers down/up/motion;
  // everything else stays queued for the window's event pump.
  while ((count = SDL_PeepEvents(events, 32, SDL_GETEVENT, SDL_FINGERDOWN,
                                 SDL_FINGERMOTION)) > 0) {
    for (int i = 0; i < count; ++i) {
      HandleFingerEvent(events[i].tfinger);
    }
    if (count < 32) {
      break;
    }
  }
}

void TouchInput::Centroid(float* x, float* y) const {
  *x = (fingers_[0].x + fingers_[1].x) * 0.5f;
  *y = (fingers_[0].y + fingers_[1].y) * 0.5f;
}

float TouchInput::PinchDist() const {
  const float dx = fingers_[0].x - fingers_[1].x;
  const float dy = fingers_[0].y - fingers_[1].y;
  return std::sqrt(dx * dx + dy * dy);
}

void TouchInput::HandleFingerEvent(const SDL_TouchFingerEvent& event) {
  ImGuiIO& io = ImGui::GetIO();
  if (io.DisplaySize.x <= 0 || io.DisplaySize.y <= 0) {
    return;
  }

  // Finger coordinates arrive normalized to the window.
  const float px = event.x * io.DisplaySize.x;
  const float py = event.y * io.DisplaySize.y;
  if (event.type == SDL_FINGERDOWN) {
    fingers_.push_back({event.fingerId, px, py});
  } else {
    auto it = std::find_if(
        fingers_.begin(), fingers_.end(),
        [&](const Finger& finger) { return finger.id == event.fingerId; });
    if (it == fingers_.end()) {
      return;  // Motion/up for a finger that landed before this page's focus.
    }
    if (event.type == SDL_FINGERUP) {
      fingers_.erase(it);
    } else {
      it->x = px;
      it->y = py;
    }
  }
  const size_t count = fingers_.size();

  // Phase transitions on finger count changes, then per-phase motion.
  if (phase_ == Phase::kIdle && count == 1) {
    io.AddMousePosEvent(fingers_[0].x, fingers_[0].y);
    io.AddMouseButtonEvent(ImGuiMouseButton_Left, true);
    phase_ = Phase::kSingle;
  } else if (phase_ == Phase::kSingle && count >= 2) {
    io.AddMouseButtonEvent(ImGuiMouseButton_Left, false);
    float cx, cy;
    Centroid(&cx, &cy);
    io.AddMousePosEvent(cx, cy);
    io.AddMouseButtonEvent(ImGuiMouseButton_Right, true);
    last_pinch_dist_ = PinchDist();
    phase_ = Phase::kMulti;
  } else if (phase_ == Phase::kSingle && count == 0) {
    io.AddMouseButtonEvent(ImGuiMouseButton_Left, false);
    phase_ = Phase::kIdle;
  } else if (phase_ == Phase::kMulti && count < 2) {
    io.AddMouseButtonEvent(ImGuiMouseButton_Right, false);
    phase_ = count == 0 ? Phase::kIdle : Phase::kEnded;
  } else if (phase_ == Phase::kEnded && count == 0) {
    phase_ = Phase::kIdle;
  } else if (phase_ == Phase::kSingle && event.type == SDL_FINGERMOTION) {
    io.AddMousePosEvent(fingers_[0].x, fingers_[0].y);
  } else if (phase_ == Phase::kMulti && event.type == SDL_FINGERUP) {
    // A finger beyond the first two lifted, so the driving pair changed:
    // re-anchor without emitting pan or zoom deltas.
    float cx, cy;
    Centroid(&cx, &cy);
    io.AddMousePosEvent(cx, cy);
    last_pinch_dist_ = PinchDist();
  } else if (phase_ == Phase::kMulti && event.type == SDL_FINGERMOTION) {
    float cx, cy;
    Centroid(&cx, &cy);
    io.AddMousePosEvent(cx, cy);
    const float dist = PinchDist();
    const float wheel =
        (dist - last_pinch_dist_) / io.DisplaySize.y * kPinchWheelPerScreen;
    if (wheel != 0.0f) {
      io.AddMouseWheelEvent(0.0f, wheel);
    }
    last_pinch_dist_ = dist;
  }
}

}  // namespace mujoco::studio
