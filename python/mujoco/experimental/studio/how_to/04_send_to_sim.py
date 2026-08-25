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
"""How to send messages from the viewer to the simulation.

  python 04_send_to_sim.py

Tutorial 03 sent messages sim -> viewer. This one closes the loop: a GUI
panel on the viewer side steers the simulation. A ball hovers under a
P-controller; the panel has

  - a "hover height" slider  -> HoverTargetSnapshot   (continuous state,
                                latest-wins: dragging the slider floods
                                snapshots and only the newest matters)
  - a "Nudge" button         -> NudgeEvent            (a discrete action:
                                two clicks must mean two nudges, so it must
                                be an Event -- never dropped, never merged)

Two new ideas appear here:

1. SIM-SIDE HANDLERS. launch_passive takes sim_handlers as well as
   viewer_handlers. Sim handlers are dispatched on YOUR thread, inside
   handle.sync() -- there is no hidden concurrency. The canonical sim-side
   handler is a MAILBOX: its handler methods only record what arrived, and
   your sim loop reads the mailbox and applies it to the physics. That
   keeps mutation of model/data in exactly one place: your loop.

2. THE ViewerInitEvent IDIOM. A viewer-side handler that wants to SEND
   messages needs the Viewer object (for send_to_sim). Handlers receive it
   by handling ViewerInitEvent, a lifecycle event dispatched once when the
   viewer starts, and caching the reference. You will see this idiom in
   every viewer-side handler that talks back to the sim.

Aside: why a hover-height slider and not, say, a gravity slider? Because
model options (mjOption) are already synced for you: ViewerApp's Physics
pane edits the viewer's copy and publishes an MjOptionSnapshot every frame,
which a built-in sim-side handler applies to the true model. Custom
messages are for state the framework does NOT already own -- like our
controller's setpoint.
"""

import dataclasses

from absl import app as _app
import mujoco
from mujoco.experimental.studio import launch_passive
from mujoco.experimental.studio import messages
from mujoco.experimental.studio import sim
from mujoco.experimental.studio import viewer_app
from mujoco.experimental.studio import viewer_protocol

from mujoco.experimental.dear_imgui import dear_imgui as imgui

_XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="2 2 .05" rgba=".8 .8 .85 1"/>
    <body name="ball" pos="0 0 1">
      <freejoint/>
      <geom type="sphere" size=".1" rgba=".3 .5 .9 1"/>
    </body>
  </worldbody>
</mujoco>
"""


# -----------------------------------------------------------------------------
# Messages. Same rules as tutorial 03: frozen dataclasses, self-contained
# payloads, Event for actions, Snapshot for state.
# -----------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class HoverTargetSnapshot(messages.Snapshot):
  """The controller setpoint. Latest-wins is exactly right for a slider."""

  height: float


@dataclasses.dataclass(frozen=True)
class NudgeEvent(messages.Event):
  """One button click, one nudge. Reliable delivery, exactly once."""

  direction: float  # +1 or -1, along x.


# -----------------------------------------------------------------------------
# Viewer side: a GUI panel that sends messages.
# -----------------------------------------------------------------------------


class HoverPanel:
  """Draws the control panel and publishes the user's intent to the sim."""

  def __init__(self) -> None:
    self._viewer: viewer_protocol.Viewer | None = None
    self._height = 1.0
    self._nudge_dir = 1.0

  @messages.handler
  def on_viewer_init(self, event: viewer_protocol.ViewerInitEvent) -> None:
    # The idiom: cache the Viewer once, use it to send messages later.
    self._viewer = event.viewer

  @messages.handler
  def on_build_gui(self, _: messages.BuildGuiEvent) -> None:
    if self._viewer is None:
      return
    if imgui.Begin('Hover', flags=int(imgui.WindowFlags.AlwaysAutoResize)):
      _, self._height = imgui.SliderFloat(
          'height (m)', self._height, 0.2, 2.0
      )
      if imgui.Button('Nudge'):
        # Events are sent at the moment of the action.
        self._viewer.send_to_sim(NudgeEvent(direction=self._nudge_dir))
        self._nudge_dir = -self._nudge_dir  # Alternate for visible effect.
    imgui.End()

    # Publish the setpoint every frame, like ViewerApp does with its own
    # snapshots. We could send only when the slider changes; with a
    # Snapshot both are fine, because the channel stores at most one.
    self._viewer.send_to_sim(HoverTargetSnapshot(height=self._height))


# -----------------------------------------------------------------------------
# Sim side: a mailbox handler.
#
# These methods run on the SIM thread, inside handle.sync(). They do not
# touch model or data -- they only record. The sim loop below is the sole
# consumer, which makes the data flow trivial to reason about:
#
#   GUI (viewer thread) --messages--> mailbox (sync) --> sim loop applies.
# -----------------------------------------------------------------------------


class HoverMailbox:
  """Collects the latest setpoint and any pending nudges."""

  def __init__(self, initial_height: float) -> None:
    self.height = initial_height
    self.pending_nudges: list[NudgeEvent] = []

  @messages.handler
  def on_target(self, snapshot: HoverTargetSnapshot) -> None:
    self.height = snapshot.height

  @messages.handler
  def on_nudge(self, event: NudgeEvent) -> None:
    # Queue, don't apply: sync() may run between physics steps, and applying
    # forces belongs to the loop. Because NudgeEvent is an Event, this list
    # sees every click, even several per sync.
    self.pending_nudges.append(event)


def main(argv: list[str]) -> None:
  del argv
  model = mujoco.MjModel.from_xml_string(_XML)
  data = mujoco.MjData(model)

  config = viewer_protocol.ViewerConfig(title='04_send_to_sim')
  mailbox = HoverMailbox(initial_height=1.0)

  with launch_passive.launch_passive(
      config,
      viewer_handlers=[viewer_app.ViewerApp(), HoverPanel()],
      sim_handlers=[mailbox],
  ) as handle:
    handle.send_to_viewer(messages.ModelEvent(model=model))

    step_control = sim.StepControl()
    while handle.is_running():
      # Consume the mailbox. qfrc_applied is ours to use: the built-in
      # perturbation machinery (Ctrl+drag) works through xfrc_applied, so
      # the two coexist -- you can nudge AND drag at the same time.
      for nudge in mailbox.pending_nudges:
        data.qvel[0] += 1.5 * nudge.direction
      mailbox.pending_nudges.clear()

      # A P-D controller pushing the ball toward the setpoint. Ordinary
      # sim-side code, driven by state that happens to come from the GUI.
      z, vz = float(data.qpos[2]), float(data.qvel[2])
      kp, kd = 50.0, 10.0
      data.qfrc_applied[2] = kp * (mailbox.height - z) - kd * vz

      step_control.advance(model, data)
      model, data, step_control = handle.sync(model, data, step_control)


# -----------------------------------------------------------------------------
# Things to try:
#
#   - Drag the slider fast: the ball tracks the newest value; the sim never
#     sees (or wastes time on) the hundreds of intermediate positions.
#   - Click Nudge rapidly N times while paused, then unpause: exactly N
#     nudges are applied. Try to imagine debugging that with latest-wins
#     semantics -- that is why buttons send Events.
#   - Ctrl+Right-drag the ball while it hovers: built-in perturbation
#     (xfrc_applied) and our controller (qfrc_applied) add up.
# -----------------------------------------------------------------------------
if __name__ == '__main__':
  _app.run(main)
