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

// Serialization of visualization state sent to the web viewer browser app.
//
// The visualization state is a fixed-size block of plain C structs sent
// alongside physics state from the Python simulation to the browser viewer.
// Layout: [mjvCamera][mjvOption][mjOption][mjVisual][mjStatistic][render_flags]

#ifndef MUJOCO_PYTHON_EXPERIMENTAL_STUDIO_WEB_VIEWER_LIVE_STATE_H_
#define MUJOCO_PYTHON_EXPERIMENTAL_STUDIO_WEB_VIEWER_LIVE_STATE_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include <mujoco/mujoco.h>

namespace mujoco::studio {

// Fixed byte size of the visualization state block appended after physics
// state.  Both sides (Linux x86_64 Python and Emscripten WASM) use identical
// struct layouts, so the size is the same everywhere.
inline constexpr size_t kLiveStateSize = sizeof(mjvCamera) + sizeof(mjvOption) +
                                         sizeof(mjOption) + sizeof(mjVisual) +
                                         sizeof(mjStatistic) + mjNRNDFLAG;

// Serialize visualization state as raw bytes.
//
// These are plain C structs with identical layout on both sides (Linux x86_64
// Python and Emscripten WASM — both LP64-like with the same packing). The
// total size is fixed (~1.1 KB) and independent of the model, making the
// overhead negligible compared to the physics state.
inline std::vector<char> SerializeLiveState(
    const mjvCamera& camera, const mjvOption& vis_options, const mjOption& opt,
    const mjVisual& vis, const mjStatistic& stat,
    const std::vector<uint8_t>& render_flags) {
  std::vector<char> buffer(kLiveStateSize);
  char* ptr = buffer.data();

  memcpy(ptr, &camera, sizeof(mjvCamera));
  ptr += sizeof(mjvCamera);

  memcpy(ptr, &vis_options, sizeof(mjvOption));
  ptr += sizeof(mjvOption);

  memcpy(ptr, &opt, sizeof(mjOption));
  ptr += sizeof(mjOption);

  memcpy(ptr, &vis, sizeof(mjVisual));
  ptr += sizeof(mjVisual);

  memcpy(ptr, &stat, sizeof(mjStatistic));
  ptr += sizeof(mjStatistic);

  // Pack render flags (mjNRNDFLAG bytes).
  memset(ptr, 0, mjNRNDFLAG);
  for (size_t i = 0; i < mjNRNDFLAG && i < render_flags.size(); ++i) {
    ptr[i] = static_cast<char>(render_flags[i]);
  }

  return buffer;
}

}  // namespace mujoco::studio

#endif  // MUJOCO_PYTHON_EXPERIMENTAL_STUDIO_WEB_VIEWER_LIVE_STATE_H_
