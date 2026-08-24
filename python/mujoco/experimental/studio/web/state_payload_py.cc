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

// Python bindings for the state payload wire format (state_payload.h).
//
// WebViewer serializes the /state WebSocket payload with this module each
// frame; the browser parses it with the same header (web_client_session).

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <tuple>
#include <vector>

#include <mujoco/mujoco.h>
#include "state_payload.h"
#include "structs.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Serialize the complete state WebSocket payload (see state_payload.h):
// physics state, render state and extra geoms as tagged blocks.
static py::bytes SerializeStatePayload(
    uint32_t model_crc32, int physics_spec, const py::bytes& physics_state,
    const mujoco::python::MjvCameraWrapper& camera,
    const mujoco::python::MjvPerturbWrapper& perturb,
    const mujoco::python::MjvOptionWrapper& vis_options,
    const mujoco::python::MjModelWrapper& model,
    const std::vector<uint8_t>& render_flags,
    const std::vector<mujoco::python::MjvGeomWrapper>& extra_geoms,
    const std::vector<float>& scene_viewport,
    // Registry blocks: (name, crc32, nbytes) per model the client must hold;
    // (name, spec, values bytes) per entry state; and
    // (name, model_id, tex_id, width, height, draw_mode, camera) per client
    // view.
    const std::vector<std::tuple<std::string, uint32_t, uint32_t>>&
        model_table,
    const std::vector<std::tuple<std::string, int32_t, py::bytes>>&
        entry_states,
    const std::vector<std::tuple<std::string, std::string, uint32_t, int, int,
                                 int, const mujoco::python::MjvCameraWrapper*>>&
        client_views) {
  std::vector<mjvGeom> geoms;
  geoms.reserve(extra_geoms.size());
  for (const mujoco::python::MjvGeomWrapper& geom_wrapper : extra_geoms) {
    if (geom_wrapper.get()) {
      geoms.push_back(*geom_wrapper.get());
    }
  }
  const float* viewport =
      scene_viewport.size() == 4 ? scene_viewport.data() : nullptr;

  auto copy_name = [](char (&dst)[mujoco::studio::kMaxWireName],
                      const std::string& src) {
    std::snprintf(dst, mujoco::studio::kMaxWireName, "%s", src.c_str());
  };

  std::vector<mujoco::studio::StateModelTableEntry> table;
  table.reserve(model_table.size());
  for (const auto& [name, crc32, nbytes] : model_table) {
    mujoco::studio::StateModelTableEntry entry{};
    copy_name(entry.name, name);
    entry.crc32 = crc32;
    entry.nbytes = nbytes;
    table.push_back(entry);
  }

  // The bytes objects in entry_states stay alive for the duration of this
  // call, so the EntryStateInput pointers below remain valid.
  std::vector<std::string> entry_values;
  entry_values.reserve(entry_states.size());
  std::vector<mujoco::studio::EntryStateInput> entries;
  entries.reserve(entry_states.size());
  for (const auto& [name, spec, values] : entry_states) {
    entry_values.push_back(values);
    mujoco::studio::EntryStateInput input{};
    copy_name(input.name, name);
    input.spec = spec;
    input.values = entry_values.back().data();
    input.nbytes = entry_values.back().size();
    entries.push_back(input);
  }

  std::vector<mujoco::studio::StateClientView> views;
  views.reserve(client_views.size());
  for (const auto& [name, model_id, tex_id, width, height, draw_mode,
                    view_camera] : client_views) {
    if (view_camera == nullptr || view_camera->get() == nullptr) {
      continue;
    }
    mujoco::studio::StateClientView view{};
    copy_name(view.name, name);
    copy_name(view.model_id, model_id);
    view.tex_id = tex_id;
    view.width = static_cast<uint16_t>(width);
    view.height = static_cast<uint16_t>(height);
    view.draw_mode = draw_mode;
    view.camera = *view_camera->get();
    views.push_back(view);
  }

  std::string physics = physics_state;
  std::vector<std::byte> buffer;
  {
    py::gil_scoped_release no_gil;
    buffer = mujoco::studio::SerializeStatePayload(
        model_crc32, physics_spec, physics.data(), physics.size(),
        *camera.get(), *perturb.get(), *vis_options.get(), model.get()->opt,
        model.get()->vis, model.get()->stat, render_flags, geoms.data(),
        geoms.size(), viewport, table, entries, views);
  }
  return py::bytes(reinterpret_cast<const char*>(buffer.data()), buffer.size());
}

// Upper bound of a serialized payload for a model whose physics state is
// `physics_bytes` long. Used to size the StateServer's shared memory.
static size_t MaxStatePayloadSize(size_t physics_bytes) {
  return mujoco::studio::MaxStatePayloadSize(physics_bytes);
}

PYBIND11_MODULE(state_payload, m, pybind11::mod_gil_not_used()) {
  py::module_::import("mujoco._structs");
  m.doc() = "MuJoCo web viewer state payload serialization";

  // scene_viewport: [] for a full-window scene, or [x, y, w, h] in logical px
  // (top-left origin) confining the 3D scene to a window sub-rectangle.
  m.def("serialize_state_payload", &SerializeStatePayload);
  m.def("max_state_payload_size", &MaxStatePayloadSize);
  m.attr("MAX_EXTRA_GEOMS") = mujoco::studio::kMaxExtraGeoms;
}
