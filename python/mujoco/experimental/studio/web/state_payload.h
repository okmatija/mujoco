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

// This file defines the serialization format for the web viewer's browser
// client render payload containing the data needed so that the browser can
// render the scene using the following call:
//
// Render(model, data, perturb, camera, vis_options, width, height, extra_geoms)
//
// The arguments come from the Python process:
//
//   * model       : fetched once over HTTP as /model; its runtime-mutable
//                   parts (opt/vis/stat) are re-sent in the render state block.
//   * data        : streamed as the physics state vector (mjSTATE_INTEGRATION);
//                   the browser recomputes the rest via mj_setState/mj_forward.
//   * width/height: the browser canvas size.
//   * extra_geoms : optional variable-size kTagExtraGeoms block.
//   * ...         : the rest of the arguments are sent as a fixed-size block
//
// The payload (SerializeStatePayload) is a sequence of tagged blocks:
//
//   [StatePayloadHeader][u32 tag][u32 size][payload]...
//
// The payload is serialized by Python, sent over the /state WebSocket, and
// parsed by the browser.
//
// TODO(matijak): Try shrinking the physics block: float32 (or quantized) values
// instead of doubles, and/or delta-encoding against the client's last-acked
// payload. The /state ack (web_server.py) tells the server which snapshot each
// client last applied, which is the baseline that delta compression needs. For
// 100humanoids.xml the payload is ~181 KB of doubles and dominates slow links.

#ifndef MUJOCO_PYTHON_EXPERIMENTAL_STUDIO_WEB_STATE_PAYLOAD_H_
#define MUJOCO_PYTHON_EXPERIMENTAL_STUDIO_WEB_STATE_PAYLOAD_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include <mujoco/mujoco.h>

namespace mujoco::studio {

// "MJWS" as little-endian bytes. This magic constant identifies the
// StateServer WebSocket payload header and helps detect malformed or
// misrouted messages.
constexpr uint32_t kStatePayloadMagic =
    'M' | ('J' << 8) | ('W' << 16) | ('S' << 24);
constexpr uint16_t kStatePayloadVersion = 1;

struct StatePayloadHeader {
  uint32_t magic = kStatePayloadMagic;
  uint16_t version = kStatePayloadVersion;
  uint16_t nblocks = 0;
  // CRC32 of the model's MJB bytes. When this changes, the browser must
  // refetch /model before applying any further state.
  uint32_t model_crc32 = 0;
};
static_assert(sizeof(StatePayloadHeader) == 12);

// Block tags. Readers must skip unknown tags.
enum StateBlockTag : uint32_t {
  kTagPhysicsState = 1,   // [i32 mjtState spec signature][mjtNum values...]
  kTagRenderState = 2,    // fixed-size block of kRenderStateSize bytes
  kTagExtraGeoms = 3,     // n x mjvGeom (n = size / sizeof(mjvGeom))
  kTagSceneViewport = 4,  // 4 x float: x, y, w, h in logical px, top-left
                          // origin. The 3D scene is confined to this window
                          // rectangle; it fills the window when absent.
  kTagModelTable = 5,     // n x StateModelTableEntry: registry models the
                          // client must hold (fetch /model?id=<name>).
  kTagEntryState = 6,     // repeatable, one per registry entry:
                          // [StateEntryStateHeader][mjtNum values...]
  kTagClientViews = 7,    // n x StateClientView: viewports the client
                          // renders with the modular filament renderer.
};

// Maximum length of a registry entry / view name on the wire, including the
// NUL terminator.
constexpr size_t kMaxWireName = 32;

// One registry model the client must mirror (kTagModelTable).
struct StateModelTableEntry {
  char name[kMaxWireName];  // registry id (NUL-terminated)
  uint32_t crc32;           // CRC32 of the entry's MJB bytes
  uint32_t nbytes;          // MJB byte size
};

// Header of a per-entry state block (kTagEntryState); mjtNum values follow.
struct StateEntryStateHeader {
  char name[kMaxWireName];  // registry id (NUL-terminated)
  int32_t spec;             // mjtState signature of the values
};

// One client-rendered viewport (kTagClientViews). The camera is the plugin's
// mjvCamera; the client converts it against the entry's own model/data with
// mjv_camera2GLCamera at render time.
struct StateClientView {
  char name[kMaxWireName];      // view name
  char model_id[kMaxWireName];  // registry entry to render
  uint32_t tex_id;              // ImGui texture id showing the render target
  uint16_t width;               // render target width in pixels
  uint16_t height;              // render target height in pixels
  int32_t draw_mode;            // mjrDrawMode
  mjvCamera camera;             // camera to render from
};

struct StateBlockHeader {
  uint32_t tag = 0;
  uint32_t size = 0;
};
static_assert(sizeof(StateBlockHeader) == 8);

// Fixed byte size of the render state block appended after physics state.
// These are plain C structs of int/float/double members whose total size is
// fixed, independent of the model and generally negligible compared to the size
// of the physics state
constexpr size_t kRenderStateSize =
    sizeof(mjvCamera) + sizeof(mjvPerturb) + sizeof(mjvOption) +
    sizeof(mjOption) + sizeof(mjVisual) + sizeof(mjStatistic) + mjNRNDFLAG;

// Maximum number of extra geoms serialized per frame. Bounds the shared
// memory buffer the StateServer allocates; WebViewer truncates longer lists.
constexpr uint32_t kMaxExtraGeoms = 1024;

// Upper bound of a serialized payload, used to size the StateServer's shared
// memory buffer. `physics_bytes` is mj_stateSize(...) * sizeof(mjtNum).
size_t MaxStatePayloadSize(size_t physics_bytes);

// Serializer-side description of one per-entry state block (kTagEntryState).
struct EntryStateInput {
  char name[kMaxWireName];
  int32_t spec;
  const void* values;
  size_t nbytes;
};

// Serialize the complete state payload sent over the state WebSocket.
// `scene_viewport` is null, or 4 floats (x, y, w, h in logical px, top-left
// origin) confining the scene; a zero-sized rect is treated as absent.
// `model_table`, `entry_states` and `client_views` describe the registry
// models, their per-frame state and the client-rendered viewports; all three
// may be empty.
std::vector<std::byte> SerializeStatePayload(
    uint32_t model_crc32, int32_t physics_spec, const void* physics,
    size_t physics_bytes, const mjvCamera& camera, const mjvPerturb& perturb,
    const mjvOption& vis_options, const mjOption& opt, const mjVisual& vis,
    const mjStatistic& stat, const std::vector<uint8_t>& render_flags,
    const mjvGeom* extra_geoms, size_t extra_geom_count,
    const float* scene_viewport = nullptr,
    const std::vector<StateModelTableEntry>& model_table = {},
    const std::vector<EntryStateInput>& entry_states = {},
    const std::vector<StateClientView>& client_views = {});

// Parsed view into a serialized payload. Pointers alias the input buffer and
// are NOT guaranteed to be aligned; so you must memcpy the data out before use.
// Parsed view of one per-entry state block; `values` aliases the payload.
struct EntryStateView {
  char name[kMaxWireName];
  int32_t spec = 0;
  const std::byte* values = nullptr;
  size_t nbytes = 0;
};

struct StatePayloadView {
  uint32_t model_crc32 = 0;
  int32_t physics_spec = 0;
  const std::byte* physics = nullptr;
  size_t physics_bytes = 0;
  const std::byte* render_state = nullptr;  // kRenderStateSize bytes when non-null
  const std::byte* extra_geoms = nullptr;   // extra_geom_count * sizeof(mjvGeom)
  size_t extra_geom_count = 0;
  // Scene viewport (x, y, w, h in logical px, top-left origin); all zero when
  // the payload carries no kTagSceneViewport block (scene fills the window).
  float scene_viewport[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  // Registry models, per-entry states, and client-rendered viewports.
  std::vector<StateModelTableEntry> model_table;
  std::vector<EntryStateView> entry_states;
  std::vector<StateClientView> client_views;
};

// Parses a payload produced by SerializeStatePayload. Returns false if the
// buffer is malformed (bad magic/version or out-of-bounds block). Blocks
// with unknown tags are skipped.
bool ParseStatePayload(const void* data, size_t size, StatePayloadView* out);

// A decoded render state block.
struct RenderStateView {
  mjvCamera camera;
  mjvPerturb perturb;
  mjvOption vis_options;
  mjOption opt;
  mjVisual vis;
  mjStatistic stat;
  uint8_t render_flags[mjNRNDFLAG];
};

// Decodes a render state block produced by the serializer. `data` must hold
// kRenderStateSize bytes (e.g. StatePayloadView::render_state).
void ParseRenderState(const std::byte* data, RenderStateView* out);

}  // namespace mujoco::studio

#endif  // MUJOCO_PYTHON_EXPERIMENTAL_STUDIO_WEB_STATE_PAYLOAD_H_
