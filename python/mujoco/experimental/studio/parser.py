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
"""Loads and compiles MuJoCo models for the studio viewer."""

from typing import Callable

import mujoco


def parse(
    filepath: str,
    spec_edit: Callable[[mujoco.MjSpec], None] | None = None,
) -> mujoco.MjData:
  """Loads a model file and returns an MjData for it.

  Args:
    filepath: Path to a .xml, .mjb, or .mjz/.zip model file.
    spec_edit: Optional callback applied to the MjSpec before compilation
      (e.g. splat.embed). Not supported for .mjb files, which are already
      compiled.
  """
  if filepath.endswith('.mjb'):
    if spec_edit is not None:
      raise ValueError('spec_edit cannot be applied to a compiled .mjb file.')
    model = mujoco.MjModel.from_binary_path(filepath)
  elif filepath.endswith(('.mjz', '.zip')):
    # .mjz archives mount assets into a VFS during parsing; the same VFS must
    # be passed to compile() so the assets remain accessible.
    with mujoco.MjVfs() as vfs:
      spec = mujoco.MjSpec.from_file(filepath, vfs=vfs)
      if spec_edit is not None:
        spec_edit(spec)
      model = spec.compile(vfs=vfs)
  else:
    spec = mujoco.MjSpec.from_file(filepath)
    if spec_edit is not None:
      spec_edit(spec)
    model = spec.compile()
  return mujoco.MjData(model)
