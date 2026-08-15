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
"""Embeds a gaussian-splat environment into a MuJoCo model.

The splat file bytes and their placement are stored as ``<custom>`` elements,
so they ride inside the compiled model — and therefore inside the ``.mjb``
that the Web Viewer serves at ``/model``. The three.js browser client
(``--gfx=web_3js``) finds them there and renders the splat around the scene
with correct depth compositing; other viewers simply ignore the fields.

Conventions (shared with web/client_3js/src/splat.ts):

  * text custom ``studio/splat``: the splat file (.spz/.ply/.splat/.ksplat/
    .sog), base64-encoded because text customs are NUL-terminated strings.
  * numeric custom ``studio/splat/xform``: [scale, x, y, z, roll, pitch, yaw]
    with offsets in the splat's own frame and angles in degrees, applied on
    top of the COLMAP -> renderer base rotation. The semantics match mjswan
    (https://github.com/ttktjmt/mjswan), so calibration constants published
    for its demos carry over.

Typical use, from a simulation script::

    spec = mujoco.MjSpec.from_file('humanoid.xml')
    splat.embed(spec, 'street.spz', scale=3.275, z_offset=0.708, yaw=40)
    model = spec.compile()

or pass ``--splat=street.spz`` to viewer.py.
"""

import base64
import os

import mujoco

TEXT_NAME = 'studio/splat'
XFORM_NAME = 'studio/splat/xform'


def embed(
    spec: mujoco.MjSpec,
    splat: str | bytes,
    *,
    scale: float = 1.0,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
    z_offset: float = 0.0,
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
) -> None:
  """Adds a splat environment to a model spec (before ``spec.compile()``).

  Args:
    spec: The model spec to embed the splat into.
    splat: Path to a splat file, or its raw bytes.
    scale: Uniform scale applied to the splat.
    x_offset: Offset along the splat's own x axis (pre-scale units).
    y_offset: Offset along the splat's own y axis.
    z_offset: Offset along the splat's own z axis (typically lifts the splat
      ground onto the MuJoCo floor plane).
    roll: Rotation in degrees on top of the COLMAP -> renderer base rotation.
    pitch: See roll.
    yaw: See roll.
  """
  if isinstance(splat, bytes):
    data = splat
  else:
    with open(os.fspath(splat), 'rb') as f:
      data = f.read()

  spec.add_text(name=TEXT_NAME, data=base64.b64encode(data).decode('ascii'))
  spec.add_numeric(
      name=XFORM_NAME,
      data=[scale, x_offset, y_offset, z_offset, roll, pitch, yaw],
      size=7,
  )
