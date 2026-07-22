# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Loads and compiles MuJoCo models with the platform parser.

The heavy lifting happens in the ``parser_cc`` extension module, which
returns the compiled model as a raw pointer. The pointer is wrapped into an
MjModel here through mujoco's own bindings (``MjModel._from_model_ptr``), so
the python wrapper is built in mujoco's core module rather than in the studio
extension (see parser.cc for why that matters).
"""

import mujoco
from mujoco.experimental.studio import parser_cc


def parse(filepath: str) -> mujoco.MjData:
  """Parses and compiles a model file; returns a fresh MjData.

  The compiled model is available as ``data.model``.

  Args:
    filepath: Path to the model file (MJCF, MJB, or anything else the
      platform loader understands).

  Raises:
    ValueError: If the file cannot be loaded or compiled.
  """
  # parse_to_model_ptr hands over ownership of the raw mjModel*;
  # _from_model_ptr wraps it (and frees it on destruction).
  model = mujoco.MjModel._from_model_ptr(parser_cc.parse_to_model_ptr(filepath))
  return mujoco.MjData(model)
