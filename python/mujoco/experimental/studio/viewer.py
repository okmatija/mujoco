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
"""Interactive Studio GUI viewer for MuJoCo."""

from absl import app as _app
from absl import flags as _flags
from mujoco.experimental.studio import launch_passive
from mujoco.experimental.studio import messages
from mujoco.experimental.studio import parser
from mujoco.experimental.studio import sim
from mujoco.experimental.studio import splat as _splat
from mujoco.experimental.studio import viewer_app
from mujoco.experimental.studio import viewer_protocol

vp = viewer_protocol

_MODEL = _flags.DEFINE_string('model', None, 'Path to model file.')
_GFX = _flags.DEFINE_enum(
    'gfx', None, vp.GFX_MODES, 'Graphics mode ("web" launches Web Viewer).'
)
_PORT = _flags.DEFINE_integer(
    'port', 0, 'Web Viewer port (0 picks first free port >= 8080).'
)
_WIDTH = _flags.DEFINE_integer('width', 1200, 'Width of the output image.')
_HEIGHT = _flags.DEFINE_integer('height', 800, 'Height of the output image')
_SPLAT = _flags.DEFINE_string(
    'splat',
    None,
    'Path to a gaussian splat file (.spz/.ply/.splat) embedded into the model'
    ' and rendered as the environment (--gfx=web_3js only).',
)
_SPLAT_SCALE = _flags.DEFINE_float('splat_scale', 1.0, 'Splat uniform scale.')
_SPLAT_OFFSET = _flags.DEFINE_list(
    'splat_offset', ['0', '0', '0'], 'Splat x,y,z offset in its own frame.'
)
_SPLAT_RPY = _flags.DEFINE_list(
    'splat_rpy', ['0', '0', '0'], 'Splat roll,pitch,yaw in degrees.'
)


def main(argv: list[str]) -> None:
  config = vp.ViewerConfig(
      width=_WIDTH.value,
      height=_HEIGHT.value,
      gfx=_GFX.value or '',
      http_port=_PORT.value,
  )

  # Resolve model path, if provided.
  model_path = _MODEL.value or (
      argv[1] if len(argv) > 1 and not argv[1].startswith('--') else None
  )

  # Embed the splat environment into the model spec, if requested.
  spec_edit = None
  if _SPLAT.value:
    x, y, z = (float(v) for v in _SPLAT_OFFSET.value)
    roll, pitch, yaw = (float(v) for v in _SPLAT_RPY.value)
    spec_edit = lambda spec: _splat.embed(
        spec,
        _SPLAT.value,
        scale=_SPLAT_SCALE.value,
        x_offset=x,
        y_offset=y,
        z_offset=z,
        roll=roll,
        pitch=pitch,
        yaw=yaw,
    )

  # Load model if path was provided.
  data, model = None, None
  if model_path and (data := parser.parse(model_path, spec_edit=spec_edit)):
    model = data.model

  with launch_passive.launch_passive(
      config,
      viewer_plugins=[viewer_app.ViewerApp()],
  ) as handle:
    # Send the model to the viewer, if we have a model.
    if model is not None:
      handle.send_to_viewer(messages.ModelEvent(model=model, path=model_path))  # pyrefly: ignore[bad-argument-type]

    # Run the simulation.
    step_control = sim.StepControl()
    try:
      while handle.is_running():
        step_control.advance(model, data)
        model, data, step_control = handle.sync(model, data, step_control)
    except KeyboardInterrupt:
      # Ctrl+C is the documented way to quit; exit cleanly, no traceback.
      print('\nShutting down.', flush=True)


_app.run(main)
