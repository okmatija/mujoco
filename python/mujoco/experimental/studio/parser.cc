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

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <mujoco/mujoco.h>
#include <mujoco/experimental/platform/sim/model_holder.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace mujoco::python {

// Loads, parses, and compiles a MuJoCo model with the platform loader and
// returns it serialized as MJB bytes. parser.py wraps the bytes back into
// python MjModel/MjData objects via mujoco's own bindings. Do NOT construct
// the python wrappers here: extension modules load RTLD_LOCAL, so wrappers
// built in this module register their raw pointers in a private copy of the
// bookkeeping maps and mujoco._structs aborts when they are destroyed.
py::bytes ParseToMjb(std::string_view filepath) {
  std::unique_ptr<platform::ModelHolder> holder;
  std::vector<uint8_t> buffer;
  {
    py::gil_scoped_release no_gil;
    holder = platform::ModelHolder::FromFile(filepath);
    if (holder->ok()) {
      mjModel* model = holder->ReleaseModel();
      mj_deleteData(holder->ReleaseData());
      buffer.resize(mj_sizeModel(model));
      mj_saveModel(model, nullptr, buffer.data(),
                   static_cast<int>(buffer.size()));
      mj_deleteModel(model);
    }
  }
  // ReleaseModel() empties the holder, so a successful parse is signalled by
  // the serialized buffer, not by holder->ok().
  if (buffer.empty()) {
    throw py::value_error(std::string("Failed to load model from '") +
                          std::string(filepath) +
                          "': " + std::string(holder->error()));
  }
  return py::bytes(reinterpret_cast<const char*>(buffer.data()),
                   buffer.size());
}

}  // namespace mujoco::python

PYBIND11_MODULE(parser_cc, m, pybind11::mod_gil_not_used()) {
  m.def("parse_to_mjb", &mujoco::python::ParseToMjb);
}
