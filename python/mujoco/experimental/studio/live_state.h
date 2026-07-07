// Serialization of visualization state sent to the MuJoCo Live browser app.
//
// The visualization state is a fixed-size block of plain C structs sent
// alongside physics state from the Python simulation to the browser viewer.
// Layout: [mjvCamera][mjvOption][mjOption][mjVisual][mjStatistic][render_flags]

#ifndef THIRD_PARTY_MUJOCO_SRC_EXPERIMENTAL_LINK_LIVE_STATE_H_
#define THIRD_PARTY_MUJOCO_SRC_EXPERIMENTAL_LINK_LIVE_STATE_H_

#include <cstddef>
#include <cstring>
#include <vector>

#include "third_party/mujoco/include/mujoco.h"

namespace mujoco::link {

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
  for (int i = 0; i < mjNRNDFLAG && i < render_flags.size(); ++i) {
    ptr[i] = static_cast<char>(render_flags[i]);
  }

  return buffer;
}

}  // namespace mujoco::link

#endif  // THIRD_PARTY_MUJOCO_SRC_EXPERIMENTAL_LINK_LIVE_STATE_H_
