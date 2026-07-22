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
"""Loads and compiles MuJoCo models for the studio viewer.

This is a thin wrapper over mujoco's own bindings: MJB files load directly,
everything else parses through mujoco.MjSpec (mj_parse under the hood, the
same call the platform loader used) and compiles. Keeping the model
construction inside mujoco's module means the MjModel/MjData wrappers come
from there, with no separate extension involved.
"""

import mujoco


def parse(filepath: str) -> mujoco.MjData:
  """Parses and compiles a model file; returns a fresh MjData.

  The compiled model is available as ``data.model``.

  Args:
    filepath: Path to the model file (MJCF, URDF, MJB, or anything else
      mujoco's parser understands).

  Raises:
    ValueError: If the file cannot be loaded or compiled.
  """
  if filepath.endswith('.mjb'):
    model = mujoco.MjModel.from_binary_path(filepath)
  else:
    model = mujoco.MjSpec.from_file(filepath).compile()
  return mujoco.MjData(model)
