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
"""How message dispatch works: priorities, consumption, base classes.

  python 05_handlers.py

This tutorial runs a BARE viewer -- launch_passive without ViewerApp -- to
show that the Studio UI is not privileged. What remains is the essential
machine: a window that renders the model, and the message dispatcher. No
toolbar, no panels, no mouse camera. Everything you see beyond the model
itself is drawn by the one handler defined in this file.

The dispatcher's full rulebook fits in four sentences:

  1. SUBSCRIPTION IS BY TYPE, INCLUDING BASE CLASSES. A message is offered
     to handlers of its exact class and of every ancestor class. A handler
     annotated with `messages.Message` therefore sees every message -- which
     is how the MessageTap below implements a message monitor without
     listing message types.

  2. PRIORITY ORDERS HANDLERS. Higher numbers run first. The named levels
     are CRITICAL=1000 (the framework applying model/state -- runs first so
     that later handlers see consistent data), USER=100 (your code; the
     default), LIBRARY=10 (reusable extensions), INTERNAL=1 (built-in UI
     like ViewerApp -- runs last so your handlers can react, or intervene,
     first). Any int works; the names are conventions, not a closed set.

  3. RETURNING True CONSUMES THE MESSAGE. Dispatch stops; lower-priority
     handlers never see it. Returning None or False passes it along. This
     is how ViewerHandle's built-ins stop a ResetEvent from propagating
     after applying it -- and how the checkbox below freezes the scene by
     eating StateSnapshots before the viewer applies them.

  4. UNHANDLED MESSAGES ARE SIMPLY DROPPED. The sim below publishes a
     HeartbeatEvent that nothing subscribes to specifically; no error, no
     warning. Publishers do not need to know whether anyone is listening.
"""

import dataclasses

from absl import app as _app
import mujoco
from mujoco.experimental.studio import launch_passive
from mujoco.experimental.studio import messages
from mujoco.experimental.studio import sim
from mujoco.experimental.studio import viewer_protocol

from mujoco.experimental.dear_imgui import dear_imgui as imgui

_XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="2 2 .05" rgba=".8 .8 .85 1"/>
    <body name="ball" pos=".2 0 1.5">
      <freejoint/>
      <geom type="sphere" size=".1" rgba=".9 .6 .2 1" solref=".02 .15"/>
    </body>
  </worldbody>
</mujoco>
"""


@dataclasses.dataclass(frozen=True)
class HeartbeatEvent(messages.Event):
  """Published by the sim once per sim-second. Nobody handles it by name."""

  beat: int


class MessageTap:
  """Counts every message and can freeze the scene by consuming state.

  Priority 2000 puts this tap above CRITICAL (1000), so it runs before even
  the framework's own handlers -- a middleware position: it can observe, or
  intercept, everything that reaches the viewer.
  """

  def __init__(self) -> None:
    self._counts: dict[str, int] = {}
    self._freeze = False

  @messages.handler(priority=2000)
  def on_any_message(self, message: messages.Message) -> bool:
    # Rule 1 at work: annotating the base class subscribes to everything --
    # built-in events and snapshots, lifecycle events, our HeartbeatEvent.
    name = type(message).__name__
    self._counts[name] = self._counts.get(name, 0) + 1

    # Rule 3 at work: while the checkbox is ticked, eat StateSnapshots.
    # The viewer's CRITICAL handler never applies them, so the rendered
    # pose freezes -- while the sim keeps running. When unfrozen, the very
    # next snapshot is the NEWEST state (latest-wins!), so the scene jumps
    # to the present rather than replaying what it missed.
    if self._freeze and isinstance(message, messages.StateSnapshot):
      return True
    return False

  @messages.handler  # Default priority: USER = 100.
  def on_build_gui(self, _: messages.BuildGuiEvent) -> None:
    # Note this handler ALSO ran through on_any_message a moment ago: one
    # object may subscribe to overlapping types; each matching handler runs
    # once, ordered by priority.
    if imgui.Begin('Message tap', flags=int(imgui.WindowFlags.AlwaysAutoResize)):
      _, self._freeze = imgui.Checkbox('freeze (consume StateSnapshots)',
                                       self._freeze)
      imgui.Separator()
      for name in sorted(self._counts):
        imgui.Text(f'{name:24s} {self._counts[name]}')
    imgui.End()


def main(argv: list[str]) -> None:
  del argv
  model = mujoco.MjModel.from_xml_string(_XML)
  data = mujoco.MjData(model)

  config = viewer_protocol.ViewerConfig(title='05_handlers')

  # No ViewerApp in this list. Compare the message counts with and without
  # it: ViewerApp is the one who publishes StepControlSnapshot and
  # MjOptionSnapshot to the sim, so with a bare viewer those simply never
  # happen -- and pause/speed/perturbation stop working, because they were
  # never framework features, only ViewerApp behaviors built from messages.
  with launch_passive.launch_passive(
      config,
      viewer_handlers=[MessageTap()],
  ) as handle:
    handle.send_to_viewer(messages.ModelEvent(model=model))

    step_control = sim.StepControl()
    next_beat = 1.0
    beats = 0
    while handle.is_running():
      step_control.advance(model, data)
      if data.time >= next_beat:
        beats += 1
        next_beat += 1.0
        handle.send_to_viewer(HeartbeatEvent(beat=beats))
      model, data, step_control = handle.sync(model, data, step_control)


# -----------------------------------------------------------------------------
# What to look at while it runs:
#
#   - UpdateEvent and BuildGuiEvent tick once per rendered frame: these are
#     LIFECYCLE events, dispatched locally by the viewer loop itself, not
#     sent by the sim. The dispatcher makes no distinction -- a message is
#     a message whether it crossed a thread or not.
#   - StateSnapshot counts frames, not physics steps: latest-wins delivery
#     means at most one per frame regardless of the physics rate.
#   - HeartbeatEvent arrives once per sim-second: pause the sim... except
#     you cannot -- there is no ViewerApp to send step control. Close the
#     window (the one interaction a bare viewer still has) to exit; the
#     handle notices and your loop ends.
#   - Tick freeze: the scene stops dead, StateSnapshot stops counting,
#     but HeartbeatEvent keeps arriving -- consumption is per-type, and the
#     sim is oblivious.
# -----------------------------------------------------------------------------
if __name__ == '__main__':
  _app.run(main)
