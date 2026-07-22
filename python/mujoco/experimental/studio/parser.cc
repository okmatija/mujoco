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

#include <mujoco/mujoco.h>
#include <mujoco/experimental/platform/sim/model_holder.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace mujoco::python {

// Loads, parses, and compiles a MuJoCo model with the platform loader and
// returns the raw mjModel* as an integer address, transferring ownership to
// the caller. parser.py wraps it into a python MjModel via
// mujoco.MjModel._from_model_ptr, which builds the wrapper inside mujoco's
// own bindings. Do NOT construct the python wrappers here: extension modules
// load RTLD_LOCAL, so wrappers built in this module register their raw
// pointers in a private copy of the bookkeeping maps and mujoco._structs
// aborts when they are destroyed.
uintptr_t ParseToModelPtr(std::string_view filepath) {
  std::unique_ptr<platform::ModelHolder> holder;
  mjModel* model = nullptr;
  {
    py::gil_scoped_release no_gil;
    holder = platform::ModelHolder::FromFile(filepath);
    if (holder->ok()) {
      // Hand the model to the caller and drop the paired data; parser.py
      // wraps the model (taking ownership) and builds a fresh MjData.
      model = holder->ReleaseModel();
      mj_deleteData(holder->ReleaseData());
    }
  }
  if (model == nullptr) {
    throw py::value_error(std::string("Failed to load model from '") +
                          std::string(filepath) +
                          "': " + std::string(holder->error()));
  }
  return reinterpret_cast<uintptr_t>(model);
}

}  // namespace mujoco::python

PYBIND11_MODULE(parser_cc, m, pybind11::mod_gil_not_used()) {
  m.def("parse_to_model_ptr", &mujoco::python::ParseToModelPtr);
}
