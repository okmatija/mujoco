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

// Wire-format implementation for the web viewer's state payload (see
// state_payload.h). Compiled into BOTH sides of the wire: the state_payload
// pybind module (serializer, via state_payload_py.cc) and the wasm
// web_client (parser, via web_client_session.cc). It must therefore stay
// free of python- or browser-specific dependencies.

#include "state_payload.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include <mujoco/mujoco.h>

namespace mujoco::studio {
namespace {

// Appends raw bytes to the payload buffer.
void AppendBytes(std::vector<std::byte>& buffer, const void* data,
                 size_t size) {
  const std::byte* bytes = static_cast<const std::byte*>(data);
  buffer.insert(buffer.end(), bytes, bytes + size);
}

// Appends a complete [u32 tag][u32 size][payload] block.
void AppendStateBlock(std::vector<std::byte>& buffer, uint32_t tag,
                      const void* data, size_t size) {
  StateBlockHeader block_header{tag, static_cast<uint32_t>(size)};
  AppendBytes(buffer, &block_header, sizeof(block_header));
  AppendBytes(buffer, data, size);
}

// Serializes the render state (exactly kRenderStateSize bytes) into `ptr`.
// Must copy the same fields in the same order as ParseRenderState.
void SerializeRenderStateInto(std::byte* ptr, const mjvCamera& camera,
                              const mjvPerturb& perturb,
                              const mjvOption& vis_options, const mjOption& opt,
                              const mjVisual& vis, const mjStatistic& stat,
                              const std::vector<uint8_t>& render_flags) {
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
    ptr[i] = static_cast<std::byte>(render_flags[i]);
  }
}

}  // namespace

// Parses a render state block produced by SerializeStatePayload.
// Must copy the same fields in the same order as SerializeRenderStateInto.
void ParseRenderState(const std::byte* data, RenderStateView* out) {
  const std::byte* ptr = data;

  memcpy(&out->camera, ptr, sizeof(mjvCamera));
  ptr += sizeof(mjvCamera);

  memcpy(&out->perturb, ptr, sizeof(mjvPerturb));
  ptr += sizeof(mjvPerturb);

  memcpy(&out->vis_options, ptr, sizeof(mjvOption));
  ptr += sizeof(mjvOption);

  memcpy(&out->opt, ptr, sizeof(mjOption));
  ptr += sizeof(mjOption);

  memcpy(&out->vis, ptr, sizeof(mjVisual));
  ptr += sizeof(mjVisual);

  memcpy(&out->stat, ptr, sizeof(mjStatistic));
  ptr += sizeof(mjStatistic);

  memcpy(out->render_flags, ptr, mjNRNDFLAG);
}

size_t MaxStatePayloadSize(size_t physics_bytes) {
  return sizeof(StatePayloadHeader) + 4 * sizeof(StateBlockHeader) +
         (sizeof(int32_t) + physics_bytes) + kRenderStateSize +
         kMaxExtraGeoms * sizeof(mjvGeom) + 4 * sizeof(float);
}

std::vector<std::byte> SerializeStatePayload(
    uint32_t model_crc32, int32_t physics_spec, const void* physics,
    size_t physics_bytes, const mjvCamera& camera, const mjvPerturb& perturb,
    const mjvOption& vis_options, const mjOption& opt, const mjVisual& vis,
    const mjStatistic& stat, const std::vector<uint8_t>& render_flags,
    const mjvGeom* extra_geoms, size_t extra_geom_count,
    const float* scene_viewport,
    const std::vector<StateModelTableEntry>& model_table,
    const std::vector<EntryStateInput>& entry_states,
    const std::vector<StateClientView>& client_views) {
  extra_geom_count =
      extra_geom_count > kMaxExtraGeoms ? kMaxExtraGeoms : extra_geom_count;
  const bool has_viewport = scene_viewport != nullptr &&
                            scene_viewport[2] > 0.0f && scene_viewport[3] > 0.0f;
  std::vector<std::byte> buffer;
  buffer.reserve(MaxStatePayloadSize(physics_bytes));

  StatePayloadHeader header;
  header.nblocks = 2 + (extra_geom_count > 0 ? 1 : 0) + (has_viewport ? 1 : 0) +
                   (model_table.empty() ? 0 : 1) +
                   static_cast<uint16_t>(entry_states.size()) +
                   (client_views.empty() ? 0 : 1);
  header.model_crc32 = model_crc32;
  AppendBytes(buffer, &header, sizeof(header));

  // Physics state: [i32 spec][mjtNum values...].
  StateBlockHeader physics_header{
      kTagPhysicsState, static_cast<uint32_t>(sizeof(int32_t) + physics_bytes)};
  AppendBytes(buffer, &physics_header, sizeof(physics_header));
  AppendBytes(buffer, &physics_spec, sizeof(int32_t));
  AppendBytes(buffer, physics, physics_bytes);

  // Render state, serialized into place.
  StateBlockHeader render_header{kTagRenderState,
                                 static_cast<uint32_t>(kRenderStateSize)};
  AppendBytes(buffer, &render_header, sizeof(render_header));
  const size_t render_offset = buffer.size();
  buffer.resize(render_offset + kRenderStateSize);
  SerializeRenderStateInto(buffer.data() + render_offset, camera, perturb,
                           vis_options, opt, vis, stat, render_flags);

  // Extra geoms (only when present).
  if (extra_geom_count > 0) {
    AppendStateBlock(buffer, kTagExtraGeoms, extra_geoms,
                     extra_geom_count * sizeof(mjvGeom));
  }

  // Scene viewport (only when the scene is confined to a window sub-rect).
  if (has_viewport) {
    AppendStateBlock(buffer, kTagSceneViewport, scene_viewport,
                     4 * sizeof(float));
  }

  // Registry model table, per-entry state, and client views (only when the
  // viewer uses the model registry beyond the simulated model).
  if (!model_table.empty()) {
    AppendStateBlock(buffer, kTagModelTable, model_table.data(),
                     model_table.size() * sizeof(StateModelTableEntry));
  }
  for (const EntryStateInput& entry : entry_states) {
    StateEntryStateHeader entry_header;
    memcpy(entry_header.name, entry.name, kMaxWireName);
    entry_header.name[kMaxWireName - 1] = '\0';
    entry_header.spec = entry.spec;
    StateBlockHeader block_header{
        kTagEntryState,
        static_cast<uint32_t>(sizeof(entry_header) + entry.nbytes)};
    AppendBytes(buffer, &block_header, sizeof(block_header));
    AppendBytes(buffer, &entry_header, sizeof(entry_header));
    AppendBytes(buffer, entry.values, entry.nbytes);
  }
  if (!client_views.empty()) {
    AppendStateBlock(buffer, kTagClientViews, client_views.data(),
                     client_views.size() * sizeof(StateClientView));
  }

  return buffer;
}

bool ParseStatePayload(const void* data, size_t size, StatePayloadView* out) {
  const std::byte* bytes = static_cast<const std::byte*>(data);
  if (size < sizeof(StatePayloadHeader)) return false;

  StatePayloadHeader header;
  memcpy(&header, bytes, sizeof(header));
  if (header.magic != kStatePayloadMagic) return false;
  if (header.version != kStatePayloadVersion) return false;
  out->model_crc32 = header.model_crc32;
  out->model_table.clear();
  out->entry_states.clear();
  out->client_views.clear();

  size_t offset = sizeof(StatePayloadHeader);
  for (uint16_t i = 0; i < header.nblocks; ++i) {
    if (offset + sizeof(StateBlockHeader) > size) return false;
    StateBlockHeader block;
    memcpy(&block, bytes + offset, sizeof(block));
    offset += sizeof(StateBlockHeader);
    if (offset + block.size > size) return false;
    const std::byte* payload = bytes + offset;

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
      case kTagSceneViewport:
        if (block.size != 4 * sizeof(float)) return false;
        memcpy(out->scene_viewport, payload, 4 * sizeof(float));
        break;
      case kTagModelTable: {
        if (block.size % sizeof(StateModelTableEntry) != 0) return false;
        const size_t n = block.size / sizeof(StateModelTableEntry);
        out->model_table.resize(n);
        memcpy(out->model_table.data(), payload, block.size);
        break;
      }
      case kTagEntryState: {
        if (block.size < sizeof(StateEntryStateHeader)) return false;
        StateEntryStateHeader entry_header;
        memcpy(&entry_header, payload, sizeof(entry_header));
        EntryStateView view;
        memcpy(view.name, entry_header.name, kMaxWireName);
        view.name[kMaxWireName - 1] = '\0';
        view.spec = entry_header.spec;
        view.values = payload + sizeof(entry_header);
        view.nbytes = block.size - sizeof(entry_header);
        out->entry_states.push_back(view);
        break;
      }
      case kTagClientViews: {
        if (block.size % sizeof(StateClientView) != 0) return false;
        const size_t n = block.size / sizeof(StateClientView);
        out->client_views.resize(n);
        memcpy(out->client_views.data(), payload, block.size);
        break;
      }
      default:
        break;  // Unknown tag: skip.
    }
    offset += block.size;
  }
  return true;
}

}  // namespace mujoco::studio
