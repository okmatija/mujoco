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
"""How to define your own messages and receive them with a handler.

  python 03_send_to_viewer.py

Tutorial 02 used only built-in messages. Here we define our own and send
them from the sim side to a handler we write on the viewer side. A ball
bounces on the floor; the sim reports two things to the viewer:

  - a BounceEvent each time the ball hits the floor. This is an Event:
    discrete, reliable, ordered, never dropped. Our viewer-side log of
    bounces is therefore guaranteed to be complete.

  - a SimStatsSnapshot with counters, sent EVERY physics step -- possibly
    thousands of times per second. This is a Snapshot: latest-wins. The
    viewer, rendering at ~60 Hz, reads only the newest one and the rest
    are silently dropped. Watch the step counter in the GUI jump by dozens
    per frame: those are the intermediate snapshots you never paid for.

The two message types side by side in one window is the whole Event vs
Snapshot decision made visible. If bounces were Snapshots, quick double
bounces would merge and the log would have holes. If stats were Events,
the viewer would receive (and queue, and dispatch) every single one.

Three rules for defining messages:

  1. Derive from messages.Event or messages.Snapshot and make it a frozen
     dataclass. Sending a message transfers ownership: after put(), the
     other thread reads it, so it must never be mutated. Freezing makes
     that mistake hard. If a field is a numpy array, send a copy.

  2. Keep payloads self-contained values (numbers, strings, small arrays),
     not live references into MjData -- the sim will keep mutating those
     while the viewer reads them.

  3. One message type per meaning. Handlers subscribe by type; the type
     annotation on the handler's second parameter is the subscription.
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

# A bouncy ball: solref's second number is a damping ratio; below 1.0 the
# contact is under-damped, i.e. the ball actually bounces.
_XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="2 2 .05" rgba=".8 .8 .85 1"/>
    <body name="ball" pos="0 0 1.5">
      <freejoint/>
      <geom type="sphere" size=".1" rgba=".9 .3 .3 1" solref=".02 .15"/>
    </body>
  </worldbody>
</mujoco>
"""


# -----------------------------------------------------------------------------
# Custom message types. By convention, name Events for the thing that
# happened and Snapshots for the state they carry.
# -----------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class BounceEvent(messages.Event):
  """The ball hit the floor. Reliable and ordered: no bounce is ever lost."""

  count: int  # 1-based bounce number.
  sim_time: float  # data.time at impact.
  impact_speed: float  # Downward speed just before impact, m/s.


@dataclasses.dataclass(frozen=True)
class SimStatsSnapshot(messages.Snapshot):
  """Continuously-refreshed sim counters. Latest-wins: only the newest
  unread snapshot is delivered; intermediate ones are dropped."""

  steps: int
  sim_time: float


# -----------------------------------------------------------------------------
# A viewer-side handler.
#
# A handler is any object with methods decorated by @messages.handler. The
# decorator reads the type annotation of the second parameter -- that type
# is the subscription. There is no registration table to maintain: pass the
# object in viewer_handlers and every decorated method is discovered.
#
# Everything below runs on the VIEWER thread. That is the framework's
# threading contract: viewer handlers run on the viewer thread, sim handlers
# (tutorial 04) run on the sim thread inside handle.sync(). Since messages
# are immutable and each side owns its own model/data, no locks are needed.
# -----------------------------------------------------------------------------


class BounceMonitor:
  """Displays the bounce log (Events) and live counters (Snapshot)."""

  def __init__(self) -> None:
    self._bounces: list[BounceEvent] = []
    self._stats: SimStatsSnapshot | None = None

  @messages.handler
  def on_bounce(self, event: BounceEvent) -> None:
    # Events of one type arrive in the order they were sent, none missing,
    # so this list is a faithful record of every impact.
    self._bounces.append(event)

  @messages.handler
  def on_stats(self, snapshot: SimStatsSnapshot) -> None:
    # At most one of these arrives per viewer frame, no matter how fast the
    # sim publishes them. Storing "the latest" is all a Snapshot handler
    # ever needs to do.
    self._stats = snapshot

  @messages.handler
  def on_model(self, event: messages.ModelEvent) -> None:
    # Built-in lifecycle message: a new model was loaded (e.g. drag & drop).
    # Handlers usually reset their per-model state here.
    del event
    self._bounces.clear()
    self._stats = None

  @messages.handler
  def on_build_gui(self, _: messages.BuildGuiEvent) -> None:
    # BuildGuiEvent is dispatched once per frame on the viewer thread; this
    # is where handlers emit ImGui code. Windows from different handlers
    # coexist: ViewerApp draws the Studio panels, we draw one more window.
    if imgui.Begin('Bounces'):
      if self._stats is not None:
        imgui.Text(f'physics steps: {self._stats.steps}')
        imgui.Text(f'sim time:      {self._stats.sim_time:.3f} s')
      imgui.Separator()
      imgui.Text(f'{len(self._bounces)} bounces (an exact record):')
      for b in self._bounces[-10:]:
        imgui.Text(f'  #{b.count}  t={b.sim_time:.3f}s  {b.impact_speed:.2f} m/s')
    imgui.End()


def main(argv: list[str]) -> None:
  del argv
  model = mujoco.MjModel.from_xml_string(_XML)
  data = mujoco.MjData(model)

  config = viewer_protocol.ViewerConfig(title='03_send_to_viewer')

  with launch_passive.launch_passive(
      config,
      # Handler order in this list does not determine dispatch order --
      # priorities do (tutorial 05). ViewerApp gives us the Studio UI;
      # BounceMonitor is ours.
      viewer_handlers=[viewer_app.ViewerApp(), BounceMonitor()],
  ) as handle:
    handle.send_to_viewer(messages.ModelEvent(model=model))

    step_control = sim.StepControl()
    steps = 0
    bounces = 0
    was_falling = False
    while handle.is_running():
      # Remember the vertical velocity before stepping so we can report the
      # impact speed (qvel[2] is the free joint's vertical velocity).
      vz_before = float(data.qvel[2])
      step_control.advance(model, data)
      steps += 1

      # A bounce is a falling -> rising transition. This is ordinary sim-side
      # code: you compute what you like, then publish it as a message.
      is_rising = float(data.qvel[2]) > 1e-3
      if was_falling and is_rising:
        bounces += 1
        handle.send_to_viewer(
            BounceEvent(
                count=bounces,
                sim_time=float(data.time),
                impact_speed=abs(vz_before),
            )
        )
      was_falling = float(data.qvel[2]) < -1e-3

      # Publish stats unconditionally, every iteration. This would be very
      # rude with an Event; with a Snapshot it costs one small allocation
      # and the channel keeps only the newest.
      handle.send_to_viewer(
          SimStatsSnapshot(steps=steps, sim_time=float(data.time))
      )

      model, data, step_control = handle.sync(model, data, step_control)


# -----------------------------------------------------------------------------
# Things to try:
#
#   - Pause with Space: stats freeze (nothing published), the log stays.
#   - Press Backspace (reset): the ball drops again and the log keeps
#     counting -- our sim-side counters are ours to reset or not.
#   - Change the speed (- / =): the step counter's per-frame jump changes
#     with it; the number of BounceEvents per bounce never does.
# -----------------------------------------------------------------------------
if __name__ == '__main__':
  _app.run(main)
