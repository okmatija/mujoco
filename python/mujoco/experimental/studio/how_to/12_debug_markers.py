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
"""How to draw debug markers: arrows, frames and spheres from sim data.

  python 12_debug_markers.py

Tutorial 10 drew overlay geoms at recorded poses. This one draws MARKERS
computed from live quantities -- the vocabulary robotics debug drawing is
built from:

  - an ARROW for each body's linear velocity,
  - a FRAME TRIAD (RGB = XYZ) on the swinging tip,
  - a SPHERE at a fixed world-space target.

RL frameworks grow a "debug visualizer" interface with exactly these
primitives (arrow, frame, sphere, ghost) for drawing command directions,
goal poses and reward-term geometry. Each is a few lines of mjvGeom setup,
so we build them as small free functions you can lift straight into your
own tools.

The one new API here is mjv_connector: given a geom that mjv_initGeom has
already filled with type and color, it poses the geom to CONNECT two world
points -- the natural way to express "arrow from A to B" without composing
rotation matrices by hand.

Like the ghost, all of this is viewer-side and rebuilt every UpdateEvent.
The sim loop is once again the untouched canonical one.
"""

from absl import app as _app
import mujoco
from mujoco.experimental.studio import launch_passive
from mujoco.experimental.studio import messages
from mujoco.experimental.studio import sim
from mujoco.experimental.studio import viewer_app
from mujoco.experimental.studio import viewer_protocol
import numpy as np

from mujoco.experimental.dear_imgui import dear_imgui as imgui

# A double pendulum: chaotic enough that velocity arrows stay interesting.
_XML = """
<mujoco>
  <option timestep=".002"/>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="2 2 .05" rgba=".8 .8 .85 1" pos="0 0 -1.2"/>
    <body name="upper" pos="0 0 0">
      <joint name="shoulder" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0 0 -.4" size=".03" rgba=".3 .5 .9 1"/>
      <body name="lower" pos="0 0 -.4">
        <joint name="elbow" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 -.4" size=".03" rgba=".9 .5 .3 1"/>
      </body>
    </body>
  </worldbody>
  <keyframe>
    <key qpos="2.8 0"/>
  </keyframe>
</mujoco>
"""

_RED = np.array([1, 0.2, 0.2, 1], dtype=np.float32)
_GREEN = np.array([0.2, 1, 0.2, 1], dtype=np.float32)
_BLUE = np.array([0.2, 0.4, 1, 1], dtype=np.float32)
_YELLOW = np.array([1, 0.9, 0.2, 0.8], dtype=np.float32)


# -----------------------------------------------------------------------------
# The marker vocabulary. Each function returns a ready-to-append mjvGeom.
# -----------------------------------------------------------------------------


def make_arrow(
    from_: np.ndarray, to: np.ndarray, rgba: np.ndarray, width: float = 0.012
) -> mujoco.MjvGeom:
  """An arrow connecting two world points."""
  geom = mujoco.MjvGeom()
  # initGeom fills in type, color and sane defaults; pos/mat/size are
  # placeholders that mjv_connector overwrites from the two endpoints.
  mujoco.mjv_initGeom(
      geom,
      int(mujoco.mjtGeom.mjGEOM_ARROW),
      np.zeros(3),
      np.zeros(3),
      np.eye(3).flatten(),
      rgba,
  )
  mujoco.mjv_connector(
      geom, int(mujoco.mjtGeom.mjGEOM_ARROW), width, from_, to
  )
  return geom


def make_sphere(
    pos: np.ndarray, radius: float, rgba: np.ndarray
) -> mujoco.MjvGeom:
  """A sphere marker at a world point."""
  geom = mujoco.MjvGeom()
  mujoco.mjv_initGeom(
      geom,
      int(mujoco.mjtGeom.mjGEOM_SPHERE),
      np.array([radius, 0, 0]),
      pos,
      np.eye(3).flatten(),
      rgba,
  )
  return geom


def make_frame(
    pos: np.ndarray, xmat: np.ndarray, scale: float = 0.15
) -> list[mujoco.MjvGeom]:
  """A coordinate frame triad: red/green/blue arrows along local X/Y/Z.

  xmat is a row-major 3x3 rotation whose COLUMNS are the frame axes in
  world coordinates, so axis i is xmat[:, i].
  """
  mat = xmat.reshape(3, 3)
  return [
      make_arrow(pos, pos + scale * mat[:, i], rgba, width=0.008)
      for i, rgba in enumerate((_RED, _GREEN, _BLUE))
  ]


class DebugMarkers:
  """Handler that draws velocity arrows, a tip frame, and a target sphere."""

  def __init__(self) -> None:
    self._viewer: viewer_protocol.Viewer | None = None
    self._show_velocity = True
    self._show_frame = True
    self._velocity_scale = 0.2  # Arrow length per m/s.
    self._target = np.array([0.0, 0.0, -0.8])

  @messages.handler
  def on_viewer_init(self, event: viewer_protocol.ViewerInitEvent) -> None:
    self._viewer = event.viewer

  @messages.handler
  def on_build_gui(self, _: messages.BuildGuiEvent) -> None:
    if imgui.Begin('Markers', flags=int(imgui.WindowFlags.AlwaysAutoResize)):
      _, self._show_velocity = imgui.Checkbox(
          'velocity arrows', self._show_velocity
      )
      _, self._show_frame = imgui.Checkbox('tip frame', self._show_frame)
      _, self._velocity_scale = imgui.SliderFloat(
          'arrow scale', self._velocity_scale, 0.05, 1.0
      )
    imgui.End()

  @messages.handler
  def on_update(self, _: messages.UpdateEvent) -> None:
    assert self._viewer is not None
    model = self._viewer.model
    data = self._viewer.data
    out = self._viewer.extra_geoms
    out.clear()

    if self._show_velocity:
      # mj_objectVelocity writes a 6-vector: angular velocity in [0:3],
      # linear velocity in [3:6], here in world coordinates (flg_local=0),
      # measured at the body frame origin.
      vel = np.zeros(6)
      for body_id in range(1, model.nbody):  # Skip the world body.
        mujoco.mj_objectVelocity(
            model, data, int(mujoco.mjtObj.mjOBJ_BODY), body_id, vel, 0
        )
        pos = data.xpos[body_id]
        tip = pos + self._velocity_scale * vel[3:6]
        out.append(make_arrow(pos, tip, _YELLOW))

    if self._show_frame:
      lower = model.body('lower').id
      # The frame is drawn at the END of the lower capsule: transform the
      # local offset (0, 0, -0.4) into world coordinates by hand -- a
      # pattern you will use constantly when drawing markers.
      offset_local = np.array([0.0, 0.0, -0.4])
      xmat = data.xmat[lower]
      pos = data.xpos[lower] + xmat.reshape(3, 3) @ offset_local
      out.extend(make_frame(pos, xmat))

    # A fixed target: markers do not have to derive from the model at all.
    out.append(make_sphere(self._target, 0.04, _RED))


def main(argv: list[str]) -> None:
  del argv
  model = mujoco.MjModel.from_xml_string(_XML)
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)  # Start swung up, then drop.

  config = viewer_protocol.ViewerConfig(title='12_debug_markers')

  with launch_passive.launch_passive(
      config,
      viewer_handlers=[viewer_app.ViewerApp(), DebugMarkers()],
  ) as handle:
    handle.send_to_viewer(messages.ModelEvent(model=model))

    step_control = sim.StepControl()
    while handle.is_running():
      step_control.advance(model, data)
      model, data, step_control = handle.sync(model, data, step_control)


# -----------------------------------------------------------------------------
# Things to try:
#
#   - Slow the sim to 10% speed (- key) -- the arrows make the velocity
#     field legible in a way the motion itself never is.
#   - Add an angular-velocity arrow using vel[0:3]. One line.
#   - Replace the fixed target with data.subtree_com[0] to mark the whole
#     mechanism's center of mass.
# -----------------------------------------------------------------------------
if __name__ == '__main__':
  _app.run(main)
