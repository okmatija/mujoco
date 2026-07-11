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
// The StateServer WebSocket payload is a sequence of tagged blocks (see
// SerializeStatePayload below):
//   [StatePayloadHeader][u32 tag][u32 size][payload]...
// Readers skip blocks with unknown tags, so new blocks can be added without
// breaking older clients. The fixed-size render state block is laid out as:
//   [mjvCamera][mjvPerturb][mjvOption][mjOption][mjVisual][mjStatistic]
//   [render_flags]
//
// In addition to physics and render state, the payload may include an
// optional `kTagExtraGeoms` block containing extra mjvGeom instances.
// This allows the headless side to send user-injected geoms without changing
// the fixed render state size.

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

// -----------------------------------------------------------------------------
// State payload: the full StateServer WebSocket message.
// -----------------------------------------------------------------------------

// "MJWS" as little-endian bytes. This magic constant identifies the
// StateServer WebSocket payload header and helps detect malformed or
// misrouted messages.
inline constexpr uint32_t kStatePayloadMagic = 0x53574A4Du;
inline constexpr uint16_t kStatePayloadVersion = 1;

// Maximum number of extra geoms serialized per frame. Bounds the shared
// memory buffer the StateServer allocates; WebViewer truncates longer lists.
inline constexpr uint32_t kMaxExtraGeoms = 1024;

// Block tags. Readers must skip unknown tags.
enum StateBlockTag : uint32_t {
  kTagPhysicsState = 1,  // [i32 mjtState spec][mjtNum values...]
  kTagRenderState = 2,   // fixed-size block of kRenderStateSize bytes
  kTagExtraGeoms = 3,    // n x mjvGeom (n = size / sizeof(mjvGeom))
};

struct StatePayloadHeader {
  uint32_t magic = kStatePayloadMagic;
  uint16_t version = kStatePayloadVersion;
  uint16_t nblocks = 0;
  // Identity of the model the payload refers to (CRC32 of the MJB bytes).
  // When this changes, the browser must refetch /model.mjb before applying
  // any further state.
  uint32_t model_ident = 0;
  uint32_t reserved = 0;
};
static_assert(sizeof(StatePayloadHeader) == 16);

struct StateBlockHeader {
  uint32_t tag = 0;
  uint32_t size = 0;
};
static_assert(sizeof(StateBlockHeader) == 8);

// Upper bound of a serialized payload, used to size the StateServer's shared
// memory buffer. `physics_bytes` is mj_stateSize(...) * sizeof(mjtNum).
inline size_t MaxStatePayloadSize(size_t physics_bytes) {
  return sizeof(StatePayloadHeader) + 3 * sizeof(StateBlockHeader) +
         (sizeof(int32_t) + physics_bytes) + kRenderStateSize +
         kMaxExtraGeoms * sizeof(mjvGeom);
}

inline void AppendStateBlock(std::vector<char>& buffer, uint32_t tag,
                             const void* data, size_t size) {
  StateBlockHeader block_header{tag, static_cast<uint32_t>(size)};
  const char* header_bytes = reinterpret_cast<const char*>(&block_header);
  buffer.insert(buffer.end(), header_bytes,
                header_bytes + sizeof(StateBlockHeader));
  const char* data_bytes = static_cast<const char*>(data);
  buffer.insert(buffer.end(), data_bytes, data_bytes + size);
}

// Serialize the complete state payload sent over the state WebSocket.
inline std::vector<char> SerializeStatePayload(
    uint32_t model_ident, int32_t physics_spec, const void* physics,
    size_t physics_bytes, const mjvCamera& camera, const mjvPerturb& perturb,
    const mjvOption& vis_options, const mjOption& opt, const mjVisual& vis,
    const mjStatistic& stat, const std::vector<uint8_t>& render_flags,
    const mjvGeom* extra_geoms, size_t extra_geom_count) {
  extra_geom_count = extra_geom_count > kMaxExtraGeoms ? kMaxExtraGeoms
                                                       : extra_geom_count;
  std::vector<char> buffer;
  buffer.reserve(MaxStatePayloadSize(physics_bytes));

  StatePayloadHeader header;
  header.nblocks = extra_geom_count > 0 ? 3 : 2;
  header.model_ident = model_ident;
  const char* header_bytes = reinterpret_cast<const char*>(&header);
  buffer.insert(buffer.end(), header_bytes,
                header_bytes + sizeof(StatePayloadHeader));

  // Physics state: [i32 spec][mjtNum values...].
  std::vector<char> physics_block(sizeof(int32_t) + physics_bytes);
  memcpy(physics_block.data(), &physics_spec, sizeof(int32_t));
  memcpy(physics_block.data() + sizeof(int32_t), physics, physics_bytes);
  AppendStateBlock(buffer, kTagPhysicsState, physics_block.data(),
                   physics_block.size());

  // Render state.
  std::vector<char> render_block = SerializeRenderState(
      camera, perturb, vis_options, opt, vis, stat, render_flags);
  AppendStateBlock(buffer, kTagRenderState, render_block.data(),
                   render_block.size());

  // Extra geoms (only when present).
  if (extra_geom_count > 0) {
    AppendStateBlock(buffer, kTagExtraGeoms, extra_geoms,
                     extra_geom_count * sizeof(mjvGeom));
  }

  return buffer;
}

// Parsed view into a serialized payload. Pointers alias the input buffer and
// are NOT guaranteed to be aligned — memcpy the data out before use.
struct StatePayloadView {
  uint32_t model_ident = 0;
  int32_t physics_spec = 0;
  const char* physics = nullptr;
  size_t physics_bytes = 0;
  const char* render_state = nullptr;  // kRenderStateSize bytes when non-null
  const char* extra_geoms = nullptr;   // extra_geom_count * sizeof(mjvGeom)
  size_t extra_geom_count = 0;
};

// Parses a payload produced by SerializeStatePayload. Returns false if the
// buffer is malformed (bad magic/version or out-of-bounds block). Blocks
// with unknown tags are skipped.
inline bool ParseStatePayload(const void* data, size_t size,
                              StatePayloadView* out) {
  const char* bytes = static_cast<const char*>(data);
  if (size < sizeof(StatePayloadHeader)) return false;

  StatePayloadHeader header;
  memcpy(&header, bytes, sizeof(header));
  if (header.magic != kStatePayloadMagic) return false;
  if (header.version != kStatePayloadVersion) return false;
  out->model_ident = header.model_ident;

  size_t offset = sizeof(StatePayloadHeader);
  for (uint16_t i = 0; i < header.nblocks; ++i) {
    if (offset + sizeof(StateBlockHeader) > size) return false;
    StateBlockHeader block;
    memcpy(&block, bytes + offset, sizeof(block));
    offset += sizeof(StateBlockHeader);
    if (offset + block.size > size) return false;
    const char* payload = bytes + offset;

    switch (block.tag) {
      case kTagPhysicsState:
        if (block.size < sizeof(int32_t)) return false;
        memcpy(&out->physics_spec, payload, sizeof(int32_t));
        out->physics = payload + sizeof(int32_t);
        out->physics_bytes = block.size - sizeof(int32_t);
        break;
      case kTagRenderState:
        if (block.size != kRenderStateSize) return false;
        out->render_state = payload;
        break;
      case kTagExtraGeoms:
        if (block.size % sizeof(mjvGeom) != 0) return false;
        out->extra_geoms = payload;
        out->extra_geom_count = block.size / sizeof(mjvGeom);
        break;
      default:
        break;  // Unknown tag: skip.
    }
    offset += block.size;
  }
  return true;
}

}  // namespace mujoco::studio

#endif  // MUJOCO_PYTHON_EXPERIMENTAL_STUDIO_WEB_VIEWER_RENDER_STATE_H_
