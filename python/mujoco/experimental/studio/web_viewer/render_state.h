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

// Serialization of the render state sent to the web viewer browser app.
//
// The browser renders the scene with the same call the native viewer makes
// every frame:
//
//   Render(model, data, perturb, camera, vis_options, width, height)
//
// In the native viewer those arguments are passed in-process by pointer. In
// the web viewer they must cross a process boundary, so each crosses in its
// cheapest form:
//
//   * model    — fetched once over HTTP as /model.mjb; runtime-mutable parts
//                (opt/vis/stat) are re-sent in this block.
//   * data     — streamed as the physics state vector (mjSTATE_INTEGRATION);
//                the browser runs mj_setState + mj_forward to recompute all
//                derived quantities locally.
//   * the rest — this fixed-size block: everything else Render() reads.
//
// The StateServer WebSocket payload is therefore:
//   [4B sig][physics state (mjtNum[])][render state (this block)]
// with this block laid out as:
//   [mjvCamera][mjvPerturb][mjvOption][mjOption][mjVisual][mjStatistic]
//   [render_flags]

#ifndef MUJOCO_PYTHON_EXPERIMENTAL_STUDIO_WEB_VIEWER_RENDER_STATE_H_
#define MUJOCO_PYTHON_EXPERIMENTAL_STUDIO_WEB_VIEWER_RENDER_STATE_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include <mujoco/mujoco.h>

namespace mujoco::studio {

// Fixed byte size of the render state block appended after physics state.
// Both sides (Linux x86_64 Python and Emscripten WASM) use identical struct
// layouts — these structs contain no pointers and both ABIs align doubles to
// 8 bytes — so the size is the same everywhere.
inline constexpr size_t kRenderStateSize =
    sizeof(mjvCamera) + sizeof(mjvPerturb) + sizeof(mjvOption) +
    sizeof(mjOption) + sizeof(mjVisual) + sizeof(mjStatistic) + mjNRNDFLAG;

// Serialize the render state as raw bytes.
//
// These are plain C structs with identical layout on both sides (Linux x86_64
// Python and Emscripten WASM). The total size is fixed (~1.5 KB) and
// independent of the model, making the overhead negligible compared to the
// physics state.
inline std::vector<char> SerializeRenderState(
    const mjvCamera& camera, const mjvPerturb& perturb,
    const mjvOption& vis_options, const mjOption& opt, const mjVisual& vis,
    const mjStatistic& stat, const std::vector<uint8_t>& render_flags) {
  std::vector<char> buffer(kRenderStateSize);
  char* ptr = buffer.data();

  memcpy(ptr, &camera, sizeof(mjvCamera));
  ptr += sizeof(mjvCamera);

  memcpy(ptr, &perturb, sizeof(mjvPerturb));
  ptr += sizeof(mjvPerturb);

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

#endif  // MUJOCO_PYTHON_EXPERIMENTAL_STUDIO_WEB_VIEWER_RENDER_STATE_H_
