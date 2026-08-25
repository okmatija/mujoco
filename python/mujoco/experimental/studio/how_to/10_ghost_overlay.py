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
"""How to draw overlay geometry: a time-delayed ghost of the model.

  python 10_ghost_overlay.py humanoid.xml

This tutorial introduces EXTRA GEOMS, the viewer's mechanism for drawing
things that are not part of the model: ghosts, target poses, debug arrows,
motion trails. `viewer.extra_geoms` is a plain Python list of mjvGeom; the
renderer draws whatever it holds each frame, after the model's own geoms.
The pattern every overlay handler follows:

    on UpdateEvent:  extra_geoms.clear(); append this frame's geoms.

Here we render a semi-transparent copy of every moving geom, delayed by a
configurable interval -- a "ghost" trailing the motion. Ghosts are a
workhorse of robotics visualization: reference poses in motion tracking,
sim-vs-real comparison (tutorial 30), before/after keyframe editing.

Note WHERE this runs: entirely on the viewer side, off the sim thread. The
sim below is the unmodified canonical loop from tutorial 02 -- it does not
know ghosts exist. The handler keeps its pose history at render rate from
the state the viewer already receives, so the overlay costs the physics
nothing. This separation -- physics in your loop, visualization in
handlers -- is the architecture working as intended.
"""

import collections
import os
import sys

from absl import app as _app
import mujoco
from mujoco.experimental.studio import launch_passive
from mujoco.experimental.studio import messages
from mujoco.experimental.studio import parser
from mujoco.experimental.studio import sim
from mujoco.experimental.studio import viewer_app
from mujoco.experimental.studio import viewer_protocol
import numpy as np

from mujoco.experimental.dear_imgui import dear_imgui as imgui


class GhostRenderer:
  """Handler that renders time-delayed semi-transparent ghost geoms."""

  def __init__(self) -> None:
    self._viewer: viewer_protocol.Viewer | None = None
    self._delay: float = 0.5
    self._ghost_rgba: list[float] = [0.4, 0.5, 0.9, 0.5]
    # Pose history: (sim time, geom positions, geom orientations). A deque
    # because we trim from the front as time advances.
    self._history: collections.deque[tuple[float, np.ndarray, np.ndarray]] = (
        collections.deque()
    )
    self._last_time: float | None = None

  def _is_fixed_body(self, body_id: int) -> bool:
    """True for bodies welded to the world (their ghost would be invisible,
    exactly covered by the body itself), unless they are mocap-driven."""
    assert self._viewer is not None
    model = self._viewer.model
    is_weld = model.body_weldid[body_id] == 0
    root_id = model.body_rootid[body_id]
    return bool(is_weld and model.body_mocapid[root_id] < 0)

  @messages.handler
  def on_viewer_init(self, event: viewer_protocol.ViewerInitEvent) -> None:
    """The tutorial-04 idiom: cache the Viewer to reach model/data later."""
    self._viewer = event.viewer

  @messages.handler
  def on_model(self, event: messages.ModelEvent) -> bool:
    """A new model arrived (e.g. drag & drop): our history is meaningless.

    Note the framework already cleared viewer.extra_geoms for us -- stale
    overlay geoms referencing a dead model would crash the renderer.
    """
    del event  # Model/data are accessed via self._viewer.
    self._history.clear()
    self._last_time = None
    return False  # Not consumed: other handlers also want ModelEvent.

  @messages.handler
  def on_build_gui(self, _: messages.BuildGuiEvent) -> None:
    """A small panel to configure the ghost, alongside the Studio UI."""
    if imgui.Begin('Ghost', flags=int(imgui.WindowFlags.AlwaysAutoResize)):
      changed_delay, delay = imgui.InputFloat(
          'Delay (s)', self._delay, 0.05, 0.5, '%.2f'
      )
      if changed_delay:
        self._delay = max(0.0, delay)

      changed_color, rgba = imgui.ColorEdit4('Color', self._ghost_rgba)
      if changed_color:
        self._ghost_rgba = rgba
    imgui.End()

  @messages.handler
  def on_update(self, _: messages.UpdateEvent) -> None:
    """Once per frame: record the current pose, emit the delayed one."""
    assert self._viewer is not None
    model = self._viewer.model
    data = self._viewer.data

    # Rebuild the overlay from scratch each frame. Overlay geoms are cheap
    # value objects; clearing and re-appending is the intended usage, not a
    # performance compromise.
    self._viewer.extra_geoms.clear()

    # Record history keyed by SIM time (data.time), not wall time: if the
    # user pauses, the ghost freezes with the sim; if time runs backwards
    # (reset), the history is invalid and must be dropped.
    if self._last_time is None or data.time < self._last_time:
      self._history.clear()
    if not self._history or data.time > self._last_time:
      self._history.append((
          data.time,
          data.geom_xpos.copy(),  # .copy(): data mutates every frame.
          data.geom_xmat.copy(),
      ))
      self._last_time = data.time

    # Drop history older than the delay window; index 0 becomes the frame
    # closest to (now - delay).
    target_time = data.time - self._delay
    while len(self._history) > 1 and self._history[1][0] <= target_time:
      self._history.popleft()

    _, xpos, xmat = self._history[0]
    if len(xpos) != model.ngeom or len(xmat) != model.ngeom:
      return  # Stale history from before a model swap; skip this frame.

    # One overlay geom per visible, moving model geom, at the historical
    # pose. mjv_initGeom fills an mjvGeom from (type, size, pos, mat, rgba);
    # setting dataid is what makes mesh geoms render as the actual mesh.
    ghost_rgba = np.array(self._ghost_rgba, dtype=np.float32)
    for i in range(model.ngeom):
      if (
          self._is_fixed_body(model.geom_bodyid[i])
          or model.geom_rgba[i, 3] == 0  # Invisible geoms have no ghost.
          or model.geom_group[i] > 2  # Match the default visible groups.
      ):
        continue

      geom = mujoco.MjvGeom()
      mujoco.mjv_initGeom(
          geom,
          int(model.geom_type[i]),
          model.geom_size[i],
          xpos[i],
          xmat[i].flatten(),
          ghost_rgba,
      )
      geom.dataid = model.geom_dataid[i]
      geom.objtype = int(mujoco.mjtObj.mjOBJ_GEOM)
      geom.objid = i
      self._viewer.extra_geoms.append(geom)


def main(argv: list[str]) -> None:
  if len(argv) != 2:
    raise _app.UsageError('Please provide exactly one MJCF path argument.')

  data = None
  try:
    if (data := parser.parse(argv[1])) is None:
      raise ValueError('parser returned None')
  except Exception as ex:  # pylint: disable=broad-except
    print(f'Failed to load model from {argv[1]!r}: {ex}')
    sys.exit(1)
  model = data.model

  config = viewer_protocol.ViewerConfig(
      title=os.path.basename(sys.argv[0]),
  )

  # Handlers compose: the full Studio UI and our overlay, side by side.
  # Neither knows about the other; both subscribe to the same events.
  with launch_passive.launch_passive(
      config,
      viewer_handlers=[viewer_app.ViewerApp(), GhostRenderer()],
  ) as handle:
    handle.send_to_viewer(messages.ModelEvent(model=model, path=argv[1]))

    step_control = sim.StepControl()
    while handle.is_running():
      step_control.advance(model, data)
      model, data, step_control = handle.sync(model, data, step_control)


# -----------------------------------------------------------------------------
# Things to try:
#
#   - Ctrl+Right-drag a body around and watch the ghost chase it.
#   - Increase the delay to 2 s, then pause: sim time stops, and so does
#     the ghost -- history is keyed by data.time on purpose.
#   - Drag & drop a different model: on_model clears the history and the
#     ghost re-forms on the new model, no restart needed.
# -----------------------------------------------------------------------------
if __name__ == '__main__':
  _app.run(main)
