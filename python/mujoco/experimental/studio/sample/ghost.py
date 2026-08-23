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
"""Example to run studio with a time-delayed ghost overlay of the model.

This script runs a Studio viewer that renders a time-delayed semi-transparent
ghost of the model, implemented with the viewer's model registry: the ghost is
a second display-only model (sharing the simulated MjModel, with its own
MjData) posed at the delayed time and drawn with a tint. The viewer derives
the overlay geoms automatically — no per-geom bookkeeping in the plugin.
"""

import collections
import os
import sys

from absl import app as _app
from absl import flags as _flags
import mujoco
from mujoco.experimental.studio import launch_passive
from mujoco.experimental.studio import messages
from mujoco.experimental.studio import parser
from mujoco.experimental.studio import step_control
from mujoco.experimental.studio import studio_app_events
from mujoco.experimental.studio import ux
from mujoco.experimental.studio import viewer_protocol
from mujoco.experimental.studio import viewer_utils
import numpy as np

from mujoco.experimental.dear_imgui import dear_imgui as imgui

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


class GhostRenderer:
  """Plugin that displays a time-delayed semi-transparent model ghost."""

  def __init__(self) -> None:
    self._viewer: viewer_protocol.Viewer | None = None
    self._entry: viewer_protocol.ModelEntry | None = None
    self._delay: float = 0.5
    self._ghost_rgba: list[float] = [0.4, 0.5, 0.9, 0.5]
    # History of (time, qpos, mocap_pos, mocap_quat) snapshots.
    self._history: collections.deque[
        tuple[float, np.ndarray, np.ndarray, np.ndarray]
    ] = collections.deque()
    self._last_time: float | None = None

  def _register_ghost(self) -> None:
    """(Re)registers the ghost as a display model for the current model."""
    assert self._viewer is not None
    self._entry = self._viewer.add_model(
        'ghost',
        self._viewer.model,
        tint=tuple(self._ghost_rgba),
    )

  @messages.handler
  def on_viewer_init(self, event: viewer_protocol.ViewerInitEvent) -> None:
    """Caches the Viewer reference and registers the ghost model."""
    self._viewer = event.viewer
    self._register_ghost()

  @messages.handler
  def on_model(self, event: messages.ModelEvent) -> bool:
    """Re-registers the ghost when the model changes."""
    del event  # The Viewer already swapped its model (CRITICAL handler).
    self._history.clear()
    self._last_time = None
    if self._viewer is not None:
      self._register_ghost()
    return False

  @messages.handler
  def on_build_gui(self, _: messages.BuildGuiEvent) -> None:
    """Renders the ghost configuration GUI panel."""
    if imgui.Begin('Ghost', flags=int(imgui.WindowFlags.AlwaysAutoResize)):
      changed_delay, delay = imgui.InputFloat(
          'Delay (s)', self._delay, 0.05, 0.5, '%.2f'
      )
      if changed_delay:
        self._delay = max(0.0, delay)

      changed_color, rgba = imgui.ColorEdit4('Color', self._ghost_rgba)
      if changed_color:
        self._ghost_rgba = rgba
        if self._entry is not None:
          self._entry.tint = tuple(rgba)
    imgui.End()

  @messages.handler
  def on_update(self, _: messages.UpdateEvent) -> None:
    """Handles mouse events, applies perturbations, and updates ghost geoms."""
    assert self._viewer is not None
    model = self._viewer.model
    data = self._viewer.data

    studio_app_events.handle_mouse_events(
        model,
        data,
        self._viewer.camera,
        self._viewer.vis_options,
        self._viewer.perturb,
        ux.UxState(),
    )
    viewer_utils.apply_perturb(self._viewer, model, data)

    if self._last_time is None or data.time < self._last_time:
      self._history.clear()
    if not self._history or data.time > self._last_time:  # pyrefly: ignore[unsupported-operation]
      self._history.append((
          data.time,
          data.qpos.copy(),
          data.mocap_pos.copy(),
          data.mocap_quat.copy(),
      ))
      self._last_time = data.time

    target_time = data.time - self._delay
    while len(self._history) > 1 and self._history[1][0] <= target_time:
      self._history.popleft()

    # Pose the ghost model at the delayed configuration; the viewer derives
    # the overlay geoms from the registry entry.
    entry = self._entry
    if entry is None:
      return
    _, qpos, mocap_pos, mocap_quat = self._history[0]  # pyrefly: ignore[bad-assignment]
    if len(qpos) != entry.model.nq:
      return
    entry.data.qpos[:] = qpos
    if entry.model.nmocap > 0:
      entry.data.mocap_pos[:] = mocap_pos
      entry.data.mocap_quat[:] = mocap_quat
    mujoco.mj_forward(entry.model, entry.data)


def main(argv: list[str]) -> None:
  model_path = _MODEL.value or (
      argv[1] if len(argv) > 1 and not argv[1].startswith('--') else None
  )
  if not model_path:
    raise _app.UsageError(
        'Please provide a model path argument or --model flag.'
    )

  data = None
  try:
    if (data := parser.parse(model_path)) is None:
      raise ValueError('parser returned None')
  except Exception as ex:  # pylint: disable=broad-except
    print(f'Failed to load model from {model_path!r}: {ex}')
    sys.exit(1)
  model = data.model

  config = viewer_protocol.ViewerConfig(
      title=os.path.basename(sys.argv[0]),
      width=_WIDTH.value,
      height=_HEIGHT.value,
      gfx=_GFX.value or '',
      http_port=_PORT.value,
  )

  ghost_renderer = GhostRenderer()

  with launch_passive.launch_passive(
      config,
      viewer_plugins=[ghost_renderer],
      sim_plugins=[step_control.StepControl()],
  ) as handle:
    handle.send_to_viewer(messages.ModelEvent(model=model))

    try:
      while handle.is_running():
        model, data = handle.sync(model, data)
    except KeyboardInterrupt:
      # Ctrl+C is the documented way to quit; exit cleanly, no traceback.
      print('\nShutting down.', flush=True)


if __name__ == '__main__':
  _app.run(main)
