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

// The web viewer's state payload: the wire format of the /state
// WebSocket, serialized by the Python side (state_payload_py.cc) and parsed
// by the browser (web_client_session.cc).
//
// The browser renders with the same call the native viewer makes each frame:
//
// Render(model, data, perturb, camera, vis_options, width, height, extra_geoms)
//
// width/height are the browser's own canvas size. The other arguments come
// from the Python process, each sent in its cheapest form:
//
//   * model       : fetched once over HTTP as /model.mjb; its runtime-mutable
//                   parts (opt/vis/stat) re-sent in the render state block.
//   * data        : streamed as the physics state vector (mjSTATE_INTEGRATION);
//                   the browser recomputes the rest via mj_setState/mj_forward.
//   * extra_geoms : optional variable-size kTagExtraGeoms block.
//   * the rest    : the fixed-size render state block:
//                   [mjvCamera][mjvPerturb][mjvOption][mjOption][mjVisual]
//                   [mjStatistic][render_flags]
//
// The payload (see SerializeStatePayload below) is a sequence of tagged
// blocks,
//   [StatePayloadHeader][u32 tag][u32 size][payload]...
// and readers skip unknown tags, so new blocks don't break older clients.

#ifndef MUJOCO_PYTHON_EXPERIMENTAL_STUDIO_WEB_STATE_PAYLOAD_H_
#define MUJOCO_PYTHON_EXPERIMENTAL_STUDIO_WEB_STATE_PAYLOAD_H_

#include <mujoco/mujoco.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace mujoco::studio {

// The wire structs below are memcpy'd across platforms, so both sides must
// agree on the exact layout; the static_asserts turn a layout divergence
// (padding, packing) into a compile error on the offending platform — the
// only place it can be fixed.

// -----------------------------------------------------------------------------
// Payload.
// -----------------------------------------------------------------------------

// Identifies the StateServer WebSocket payload header ("MJWS" as little-endian
// bytes on the wire); helps detect malformed or misrouted messages. Built from
// single-character literals so the byte order is explicit and well-defined (a
// multicharacter literal like 'MJWS' would be implementation-defined, and on
// gcc/clang big-endian-packed — the wrong wire bytes).
inline constexpr uint32_t kStatePayloadMagic =
    'M' | ('J' << 8) | ('W' << 16) | ('S' << 24);
static_assert(kStatePayloadMagic == 0x53574A4Du);
inline constexpr uint16_t kStatePayloadVersion = 1;

struct StatePayloadHeader {
  uint32_t magic = kStatePayloadMagic;
  uint16_t version = kStatePayloadVersion;
  uint16_t nblocks = 0;
  // CRC32 of the model's MJB bytes. When this changes, the browser must
  // refetch /model.mjb before applying any further state.
  uint32_t model_crc32 = 0;
  // Padding to 16 bytes; reserved for future flags so they can be added
  // without a version bump (readers ignore it today).
  uint32_t reserved = 0;
};
static_assert(sizeof(StatePayloadHeader) == 16);

// -----------------------------------------------------------------------------
// Blocks.
// -----------------------------------------------------------------------------

// Block tags. Readers must skip unknown tags.
enum StateBlockTag : uint32_t {
  kTagPhysicsState = 1,  // [i32 mjtState spec][mjtNum values...]
  kTagRenderState = 2,   // fixed-size block of kRenderStateSize bytes
  kTagExtraGeoms = 3,    // n x mjvGeom (n = size / sizeof(mjvGeom))
};

struct StateBlockHeader {
  uint32_t tag = 0;
  uint32_t size = 0;
};
static_assert(sizeof(StateBlockHeader) == 8);

// Fixed byte size of the render state block appended after physics state.
// These are plain C structs of int/float/double members — no pointers and no
// types whose width varies by platform — and every ABI MuJoCo runs on
// (Linux/macOS/Windows on x86_64 and arm64, Emscripten wasm32) lays them out
// identically under natural alignment, so the size matches everywhere. If
// the two sides still disagree (e.g. built from different MuJoCo versions),
// ParseStatePayload rejects the block rather than misreading it.
inline constexpr size_t kRenderStateSize =
    sizeof(mjvCamera) + sizeof(mjvPerturb) + sizeof(mjvOption) +
    sizeof(mjOption) + sizeof(mjVisual) + sizeof(mjStatistic) + mjNRNDFLAG;

// Maximum number of extra geoms serialized per frame. Bounds the shared
// memory buffer the StateServer allocates; WebViewer truncates longer lists.
inline constexpr uint32_t kMaxExtraGeoms = 1024;

// -----------------------------------------------------------------------------
// Serialization API.
// -----------------------------------------------------------------------------

// Upper bound of a serialized payload, used to size the StateServer's shared
// memory buffer. `physics_bytes` is mj_stateSize(...) * sizeof(mjtNum).
size_t MaxStatePayloadSize(size_t physics_bytes);

// Serialize the complete state payload sent over the state WebSocket.
// TODO(matijak): Try shrinking the physics block: float32 (or quantized)
// values instead of doubles, and/or delta-encoding against the client's
// last-acked payload — the /state ack (web_server.py) tells the server
// exactly which snapshot each client last applied, which is the baseline
// game-style delta compression needs. At 100-humanoid scale the payload is
// ~181 KB of doubles and dominates slow links.
std::vector<std::byte> SerializeStatePayload(
    uint32_t model_crc32, int32_t physics_spec, const void* physics,
    size_t physics_bytes, const mjvCamera& camera, const mjvPerturb& perturb,
    const mjvOption& vis_options, const mjOption& opt, const mjVisual& vis,
    const mjStatistic& stat, const std::vector<uint8_t>& render_flags,
    const mjvGeom* extra_geoms, size_t extra_geom_count);

// Parsed view into a serialized payload. Pointers alias the input buffer and
// are NOT guaranteed to be aligned — memcpy the data out before use.
struct StatePayloadView {
  uint32_t model_crc32 = 0;
  int32_t physics_spec = 0;
  const std::byte* physics = nullptr;
  size_t physics_bytes = 0;
  // kRenderStateSize bytes when non-null.
  const std::byte* render_state = nullptr;
  // extra_geom_count * sizeof(mjvGeom) bytes.
  const std::byte* extra_geoms = nullptr;
  size_t extra_geom_count = 0;
};

// Parses a payload produced by SerializeStatePayload. Returns false if the
// buffer is malformed (bad magic/version or out-of-bounds block). Blocks
// with unknown tags are skipped.
bool ParseStatePayload(const void* data, size_t size, StatePayloadView* out);

}  // namespace mujoco::studio

#endif  // MUJOCO_PYTHON_EXPERIMENTAL_STUDIO_WEB_STATE_PAYLOAD_H_
