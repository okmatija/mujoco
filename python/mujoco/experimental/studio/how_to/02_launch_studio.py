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
"""How to launch the full Studio viewer around your own simulation loop.

Run with a model of your choice, or with no arguments to use a built-in one:

  python 02_launch_studio.py [model.xml]

This is the first example that uses Studio's messaging architecture, so let's
lay out the big picture before the code.

A Studio application is split into two sides:

  - The SIM side. This is your code: the main thread, the while loop below.
    It owns the "true" model and data, and decides how physics advances.

  - The VIEWER side. A render loop on another thread (launch_passive starts a
    daemon thread; other launchers could use another process or a browser).
    It owns the window, the camera, the GUI, and -- importantly -- its own
    deep COPY of the model and data, so that rendering never races with the
    physics you are stepping.

The two sides never touch each other's objects. Instead they exchange
MESSAGES over channels. There are exactly two kinds of message, and the
distinction is the heart of the whole API:

  - Event:    reliable, ordered, never dropped. Use for discrete actions
              that must not be lost ("reset", "model changed", "user
              clicked").

  - Snapshot: latest-wins. A new snapshot of the same type overwrites the
              previous, unread one. Use for continuously-refreshed state
              where only the newest value matters ("here is the physics
              state", "here is the current slider value").

Why does a viewer need this split? Because the two sides run at different
rates. Physics might step at 2000 Hz while the viewer renders at 60 Hz. If
the sim sent every state as an Event, the viewer would drown in a queue of
stale states; because state is a Snapshot, the viewer always renders the
newest one and intermediate states are silently dropped -- which is exactly
what you want. Conversely, if "reset" were a Snapshot, two quick resets
could collapse into one. Choosing Event vs Snapshot IS the design decision,
and later tutorials return to it again and again.

You will see this decomposition of responsibilities:

  launch_passive(...)   starts the viewer thread and returns a ViewerHandle.
  ViewerApp()           the whole Studio GUI (toolbar, options, inspector,
                        mouse perturbation, keyboard bindings). It is not
                        special: it is just a message HANDLER, plugged into
                        the viewer like the ones you will write in tutorial
                        03. Leave it out and you get an empty window that
                        still renders the model (tutorial 05 does this).
  handle.sync(...)      the sim side's one obligation: call it once per
                        loop iteration to exchange messages with the viewer.
"""

import sys

from absl import app as _app
import mujoco
from mujoco.experimental.studio import launch_passive
from mujoco.experimental.studio import messages
from mujoco.experimental.studio import parser
from mujoco.experimental.studio import sim
from mujoco.experimental.studio import viewer_app
from mujoco.experimental.studio import viewer_protocol

# A fallback model so the tutorial runs with no arguments. In your own tools
# you will usually load a file instead (see main() below).
_FALLBACK_XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="2 2 .05" rgba=".8 .8 .85 1"/>
    <body name="ball" pos="0 0 1">
      <freejoint/>
      <geom type="sphere" size=".1" rgba=".9 .3 .3 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def main(argv: list[str]) -> None:
  # ---------------------------------------------------------------------------
  # Load a model on the sim side.
  #
  # parser.parse() is Studio's native loader; it returns an MjData whose
  # .model attribute is the compiled MjModel. For inline XML we use the
  # standard MuJoCo Python API. Either way, these objects belong to the sim
  # side: the viewer will receive its own copy via a message.
  # ---------------------------------------------------------------------------
  model_path = argv[1] if len(argv) > 1 else ''
  if model_path:
    data = parser.parse(model_path)
    model = data.model
  else:
    model = mujoco.MjModel.from_xml_string(_FALLBACK_XML)
    data = mujoco.MjData(model)

  config = viewer_protocol.ViewerConfig(title='02_launch_studio')

  # ---------------------------------------------------------------------------
  # Launch the viewer.
  #
  # launch_passive() wires up four channels (events and snapshots, in each
  # direction), starts the viewer loop on a daemon thread, and returns a
  # ViewerHandle -- the sim side's endpoint. "Passive" means the viewer is
  # passive with respect to physics: it never steps your simulation, it only
  # observes it and sends requests back.
  #
  # viewer_handlers is the list of handler objects that live on the VIEWER
  # thread. ViewerApp is the full Studio UI; add your own handlers after it
  # (tutorial 03). There is also a sim_handlers argument for handlers that
  # live on THIS thread (tutorial 04).
  #
  # The handle is a context manager: on exit it tells the viewer to close.
  # ---------------------------------------------------------------------------
  with launch_passive.launch_passive(
      config,
      viewer_handlers=[viewer_app.ViewerApp()],
  ) as handle:
    # The viewer starts out empty. Sending a ModelEvent gives it the model;
    # the viewer deep-copies it on receipt, so after this call the two sides
    # each own an independent copy. A model is an Event, not a Snapshot: a
    # model change is a discrete fact that must never be dropped.
    handle.send_to_viewer(messages.ModelEvent(model=model, path=model_path))

    # StepControl paces the simulation: it steps physics as many times as
    # needed to track real time (scaled by the speed setting), it implements
    # pause / single-step / viscous pause, and it injects control noise when
    # asked. The Studio toolbar's play/pause/speed buttons work by sending
    # StepControlSnapshots that ViewerHandle applies to this object.
    step_control = sim.StepControl()

    # -------------------------------------------------------------------------
    # The sim loop. This is the canonical shape of every Studio tool:
    #
    #   while handle.is_running():        # until the window closes
    #     step_control.advance(...)       # advance physics (or do your own)
    #     ... = handle.sync(...)          # exchange messages with the viewer
    #
    # sync() does the sim side's half of the messaging protocol:
    #   1. Drains pending viewer messages and dispatches them to handlers.
    #      Built-in handlers apply perturbation forces (Ctrl+drag in the
    #      viewer), pause/speed changes, resets, and physics-option edits.
    #   2. Sends the current physics state to the viewer as a StateSnapshot
    #      (latest-wins: the viewer renders the newest state, never a queue
    #      of stale ones).
    #
    # IMPORTANT: sync() RETURNS (model, data, step_control), and you must
    # keep using the returned objects. If the user drags-and-drops a new
    # MJCF file onto the window, the viewer sends a ModelEvent and sync()
    # hands you a brand-new model and data. Your loop keeps working because
    # it rebinds the names each iteration. If you cache `model` elsewhere,
    # react to the swap with a sim-side ModelEvent handler (tutorial 04).
    # -------------------------------------------------------------------------
    while handle.is_running():
      step_control.advance(model, data)
      model, data, step_control = handle.sync(model, data, step_control)


# -----------------------------------------------------------------------------
# Things to try while it runs -- all of this is ViewerApp, for free:
#
#   - Space pauses; the toolbar has single-step buttons; -/= change speed.
#   - Double-click the ball to select it; Ctrl+Right-drag applies a force
#     (watch the sim side receive PerturbEvents through sync()).
#   - Drag-and-drop another .xml file onto the window; note the sim loop
#     seamlessly picks up the new model from sync()'s return value.
#   - The Physics pane edits mjOption on the VIEWER's copy of the model;
#     each frame ViewerApp sends an MjOptionSnapshot and a built-in sim-side
#     handler applies it to the true model here. Latest-wins is perfect for
#     this: dragging a slider floods snapshots, only the newest matters.
# -----------------------------------------------------------------------------
if __name__ == '__main__':
  _app.run(main)
