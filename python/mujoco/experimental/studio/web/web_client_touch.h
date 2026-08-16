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

#ifndef MUJOCO_PYTHON_EXPERIMENTAL_STUDIO_WEB_WEB_CLIENT_TOUCH_H_
#define MUJOCO_PYTHON_EXPERIMENTAL_STUDIO_WEB_WEB_CLIENT_TOUCH_H_

#include <SDL_events.h>

#include <cstdint>
#include <vector>

namespace mujoco::studio {

// Translates SDL touch (finger) events into the synthetic ImGui mouse events
// the rest of the viewer already understands, mirroring the three.js
// OrbitControls touch mapping, with no inertia:
//
//   one-finger drag  -> left-button drag   (orbit; perturb when on a body)
//   two-finger drag  -> right-button drag  (pan)
//   two-finger pinch -> mouse wheel        (zoom)
//
// The synthesized input flows through the normal input path: the local ImGui
// context consumes it for browser-drawn windows, and CaptureAndSendInput
// forwards it to the headless viewer, whose mobile input mode gives the
// gestures their touch-first meaning (see studio_app_events.py).
//
// SDL's own touch-to-mouse emulation must be disabled first (see
// DisableSdlTouchMouseEmulation) or every finger would double as a mouse.
class TouchInput {
 public:
  // Stops SDL from synthesizing mouse events for touches. Must run before
  // the SDL window is created.
  static void DisableSdlTouchMouseEmulation();

  // Drains all pending SDL finger events and injects the corresponding
  // ImGui mouse events. Call once per frame, before the window's own event
  // pump (which handles every non-finger event).
  void PumpEvents();

 private:
  // A gesture runs from the first finger down to the last finger up. Once a
  // second finger has landed, lifting back to one finger ends the gesture
  // (kEnded) instead of restarting a one-finger drag: finishing a pinch with
  // staggered lifts must not grab a body or kick the camera.
  enum class Phase {
    kIdle,    // No fingers down.
    kSingle,  // One finger: left-button drag.
    kMulti,   // Two or more fingers: pinch zoom + right-button pan.
    kEnded,   // Gesture over; waiting for all fingers to lift.
  };

  struct Finger {
    int64_t id;
    float x, y;  // Logical (CSS) pixels, the space ImGui mouse events use.
  };

  // Wheel notches for a pinch spanning the full screen height. The server
  // side maps one notch to the same zoom step as a desktop scroll tick.
  static constexpr float kPinchWheelPerScreen = 15.0f;

  void HandleFingerEvent(const SDL_TouchFingerEvent& event);
  void Centroid(float* x, float* y) const;  // Of the first two fingers.
  float PinchDist() const;                  // Between the first two fingers.

  Phase phase_ = Phase::kIdle;
  std::vector<Finger> fingers_;  // In touch order.
  float last_pinch_dist_ = 0.0f;
};

}  // namespace mujoco::studio

#endif  // MUJOCO_PYTHON_EXPERIMENTAL_STUDIO_WEB_WEB_CLIENT_TOUCH_H_
