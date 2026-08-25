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
"""How to build a target-dragging tool: forwarding mocap poses to the sim.

  python 40_mocap_target_tool.py

The tool: a red mocap target that you can grab and move -- by mouse, or by
typing exact coordinates -- while a welded box chases it through live
physics. Drag the target across the floor, park it, dial in a precise
height. This is the seed of every pose-authoring workflow: end-effector
targets for IK, goal poses for RL environments, waypoint editing.

It is also a lesson in HOW the built-in interaction machinery decomposes,
because we must extend it. Follow the chain:

  - Ctrl+dragging a body is ViewerApp writing into viewer.perturb and
    applying it to the VIEWER's model/data copy. For dynamic bodies it
    then publishes the resulting applied forces (PerturbEvent carrying
    XFRC_APPLIED state) and the sim replays them. Forces cross; poses
    don't need to.

  - Mocap bodies are different: MuJoCo moves them by POSE, not force.
    The viewer-side drag updates the viewer copy's mocap_pos -- and
    nothing crosses to the sim. One frame later the sim's authoritative
    StateSnapshot arrives and snaps the target right back. The drag
    "doesn't stick".

  - The fix needs no new machinery, which is the real point. PerturbEvent
    is not a "forces" message -- it carries an arbitrary STATE SIGNATURE:
    a bitmask naming which mjData fields its payload contains, applied
    sim-side by a built-in handler via the generic mj_setState. We simply
    send a second PerturbEvent whose signature says MOCAP_POS|MOCAP_QUAT.
    State signatures are how MuJoCo names subsets of simulation state;
    learn them once and messages like this come for free.

One dispatch subtlety, promised in tutorial 05: our forwarding handler
must read the viewer's mocap pose AFTER ViewerApp's drag handling has
written it. ViewerApp handles UpdateEvent at INTERNAL priority (1);
default USER handlers (100) run BEFORE it. So this one handler declares
priority=0 -- lower than INTERNAL, therefore last. Priorities are not
bureaucracy; they are how you order yourself around the built-ins.
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

# The target is a mocap body: kinematic, posed directly, no physics of its
# own. The box is dynamic and welded to it softly (solref tunes how
# springy the leash is). contype/conaffinity 0 keeps the target ghostly:
# it never collides, so you can drag it through the floor or the box.
_XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="2 2 .05" rgba=".8 .8 .85 1"/>
    <body name="target" mocap="true" pos="0 0 .5">
      <geom type="sphere" size=".05" rgba="1 .3 .3 .5" contype="0" conaffinity="0"/>
    </body>
    <body name="box" pos="0 0 .5">
      <freejoint/>
      <geom type="box" size=".07 .07 .07" rgba=".3 .5 .9 1" mass=".5"/>
    </body>
  </worldbody>
  <equality>
    <weld body1="target" body2="box" solref=".08 1"/>
  </equality>
</mujoco>
"""

# The signature naming exactly the state we forward: all mocap poses.
_MOCAP_SIG = int(mujoco.mjtState.mjSTATE_MOCAP_POS) | int(
    mujoco.mjtState.mjSTATE_MOCAP_QUAT
)


class MocapTargetTool:
  """Forwards viewer-side mocap poses to the sim; adds a coordinate panel."""

  def __init__(self) -> None:
    self._viewer: viewer_protocol.Viewer | None = None
    self._panel_dirty = False

  @messages.handler
  def on_viewer_init(self, event: viewer_protocol.ViewerInitEvent) -> None:
    self._viewer = event.viewer

  def _is_dragging_mocap(self) -> bool:
    """True while the user is Ctrl+dragging a mocap body."""
    assert self._viewer is not None
    perturb = self._viewer.perturb
    if perturb.active == 0 or perturb.select <= 0:
      return False
    # body_mocapid maps body id -> mocap index, or -1 for ordinary bodies.
    return self._viewer.model.body_mocapid[perturb.select] >= 0

  def _send_mocap_state(self) -> None:
    """Extracts the viewer's mocap state and forwards it to the sim.

    We use mj_getState rather than hand-packing [pos, quat] arrays: the
    packing order inside a multi-field signature is mj_getState/setState's
    business, and going through the API keeps us layout-agnostic.
    """
    assert self._viewer is not None
    model, data = self._viewer.model, self._viewer.data
    size = mujoco.mj_stateSize(model, _MOCAP_SIG)
    state = np.empty(size, np.float64)
    mujoco.mj_getState(model, data, state, _MOCAP_SIG)
    self._viewer.send_to_sim(
        messages.PerturbEvent(state=state, state_sig=_MOCAP_SIG)
    )

  # priority=0: after ViewerApp (INTERNAL=1) has applied this frame's drag
  # to the viewer's copy -- see the module docstring.
  @messages.handler(priority=0)
  def on_update(self, _: messages.UpdateEvent) -> None:
    if self._viewer is None or self._viewer.model.nmocap == 0:
      return
    # Send only while there is user intent (drag in progress, or panel
    # edit this frame). An unconditional send would also work here, but it
    # would make the viewer fight any OTHER writer of mocap state -- like
    # the streaming bridge of tutorial 31.
    if self._is_dragging_mocap() or self._panel_dirty:
      self._send_mocap_state()
      self._panel_dirty = False

  @messages.handler
  def on_build_gui(self, _: messages.BuildGuiEvent) -> None:
    """The second input path: exact coordinates, same message."""
    if self._viewer is None or self._viewer.model.nmocap == 0:
      return
    data = self._viewer.data
    if imgui.Begin('Target', flags=int(imgui.WindowFlags.AlwaysAutoResize)):
      imgui.Text('Ctrl+drag the red target, or type:')
      changed, values = imgui.InputFloatN(
          'pos (m)', [float(v) for v in data.mocap_pos[0]], '%.3f'
      )
      if changed:
        # Write into the VIEWER's copy, then forward through the exact
        # same path as a mouse drag. One code path to the sim, however
        # many input devices feed it.
        data.mocap_pos[0] = values
        self._panel_dirty = True
    imgui.End()


def main(argv: list[str]) -> None:
  del argv
  model = mujoco.MjModel.from_xml_string(_XML)
  data = mujoco.MjData(model)

  config = viewer_protocol.ViewerConfig(title='40_mocap_target_tool')

  # Note there are no sim_handlers: the sim side of this tool is entirely
  # the built-in PerturbEvent handler. A tool can be viewer-side only.
  with launch_passive.launch_passive(
      config,
      viewer_handlers=[viewer_app.ViewerApp(), MocapTargetTool()],
  ) as handle:
    handle.send_to_viewer(messages.ModelEvent(model=model))

    step_control = sim.StepControl()
    while handle.is_running():
      step_control.advance(model, data)
      model, data, step_control = handle.sync(model, data, step_control)


# -----------------------------------------------------------------------------
# Things to try:
#
#   - Double-click the red target to select it, then Ctrl+Right-drag to
#     move it in the vertical plane (Ctrl+Shift+Right-drag for the
#     horizontal plane). The box gives chase; the weld's solref makes it
#     a soft leash, not a rigid teleport.
#   - Type a height of 1.5 in the panel: precise authoring, same message.
#   - Comment out the priority=0 (leaving default USER priority) and drag:
#     the forwarded pose is now one frame behind your mouse -- the scene
#     still works, but the target visibly rubber-bands under fast drags.
#     A one-integer lesson in dispatch ordering.
#   - Load tutorial 31 next to this file and note they share the model
#     and the mocap-writing idea: 31 feeds it from a stream, 40 from the
#     mouse. Combining them -- stream + interactive override -- is a real
#     teleop tool, and you already have all the parts.
# -----------------------------------------------------------------------------
if __name__ == '__main__':
  _app.run(main)
