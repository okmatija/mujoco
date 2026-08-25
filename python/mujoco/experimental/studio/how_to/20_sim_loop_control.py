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
"""How to run your own controller in the sim loop.

  python 20_sim_loop_control.py

So far the sim loops have been passive: advance, sync, repeat. This
tutorial makes the sim side earn its keep -- an open-loop controller
drives an actuated pendulum -- and spells out the contract your loop signs
with the framework once it starts doing real work.

THE CONTRACT, in three clauses:

  1. You own model and data BETWEEN calls to sync(); the framework owns
     them DURING sync(). Write ctrl, apply forces, mutate state freely in
     your loop -- but do it before advance(), so the physics step sees it.

  2. What sync() returns is the truth. The user can drag & drop a new
     model at any moment; when that happens sync() returns a NEW model and
     data, and anything you cached against the old one (actuator ids,
     addresses, warm-start state) is garbage. Code that survives a model
     swap re-derives its bindings whenever the model object changes --
     the _rebind() pattern below.

  3. step_control decides IF physics steps; you decide WHAT each step
     does. advance() honors pause, single-step and the speed setting, and
     it may take zero, one or many mj_step()s per call to track real time.
     Write ctrl as a function of data.time -- as below -- and pausing,
     stepping and slow motion all behave correctly with no extra code.
     (If instead you count loop iterations, your controller will drift
     from sim time and misbehave when paused. Don't.)

There is no custom GUI here on purpose: Studio's Inspector already shows
actuators (Controls pane) and sensors (Sensors pane) live. Look at the
right-hand panel while it runs -- for many control experiments you need
nothing else.
"""

import math

from absl import app as _app
import mujoco
from mujoco.experimental.studio import launch_passive
from mujoco.experimental.studio import messages
from mujoco.experimental.studio import sim
from mujoco.experimental.studio import viewer_app
from mujoco.experimental.studio import viewer_protocol

# A pendulum with a weak motor: the sine drive below pumps energy into the
# swing the way you would push a playground swing -- watch it wind up.
# The sensors feed the Sensors pane of the Inspector.
_XML = """
<mujoco>
  <option timestep=".002"/>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <body name="pole" pos="0 0 1">
      <joint name="hinge" type="hinge" axis="0 1 0" damping=".05"/>
      <geom type="capsule" fromto="0 0 0 0 0 -.5" size=".03" rgba=".9 .4 .3 1"/>
      <body name="bob" pos="0 0 -.5">
        <geom type="sphere" size=".08" rgba=".3 .4 .9 1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="drive" joint="hinge" gear="1" ctrlrange="-2 2"/>
  </actuator>
  <sensor>
    <jointpos name="angle" joint="hinge"/>
    <jointvel name="speed" joint="hinge"/>
  </sensor>
</mujoco>
"""


class SineController:
  """An open-loop sine drive, robust to model swaps.

  Deliberately written as a class with explicit (re)binding, because that
  is the shape controllers take in real tools: they hold references INTO
  the model (here just an actuator id) that must be re-derived when the
  model changes.
  """

  def __init__(self, amplitude: float = 1.2, frequency_hz: float = 0.66):
    self._amplitude = amplitude
    self._frequency_hz = frequency_hz
    self._model: mujoco.MjModel | None = None
    self._actuator_id = -1

  def _rebind(self, model: mujoco.MjModel) -> None:
    """Re-derive model bindings. Cheap, so correctness beats cleverness."""
    self._model = model
    try:
      self._actuator_id = model.actuator('drive').id
    except KeyError:
      self._actuator_id = -1  # Swapped-in model has no such actuator.

  def apply(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
    # Clause 2 of the contract: if the model object changed under us,
    # rebind before touching anything id-addressed.
    if model is not self._model:
      self._rebind(model)
    if self._actuator_id < 0:
      return  # Politely do nothing on models we don't understand.

    # Clause 3: a function of sim time, not of iteration count.
    phase = 2.0 * math.pi * self._frequency_hz * data.time
    data.ctrl[self._actuator_id] = self._amplitude * math.sin(phase)


def main(argv: list[str]) -> None:
  del argv
  model = mujoco.MjModel.from_xml_string(_XML)
  data = mujoco.MjData(model)

  config = viewer_protocol.ViewerConfig(title='20_sim_loop_control')
  controller = SineController()

  with launch_passive.launch_passive(
      config,
      viewer_handlers=[viewer_app.ViewerApp()],
  ) as handle:
    handle.send_to_viewer(messages.ModelEvent(model=model))

    step_control = sim.StepControl()
    while handle.is_running():
      # Clause 1: mutate before advancing.
      controller.apply(model, data)
      step_control.advance(model, data)
      model, data, step_control = handle.sync(model, data, step_control)


# -----------------------------------------------------------------------------
# Things to try:
#
#   - Open the Controls pane (Inspector, right) and watch the ctrl slider
#     wave back and forth: that is your controller, visualized for free.
#     Fight it by dragging the slider -- next loop iteration wins.
#   - Pause, then single-step (toolbar button): the drive stays
#     phase-correct because it is a function of data.time.
#   - Set speed to 25%: slow motion, same trajectory. Then try replacing
#     data.time with a hand-counted `t += 0.002` and watch both of these
#     break -- the contract's third clause, demonstrated.
#   - Drag & drop any other model: the controller notices the swap and
#     stands down; everything else keeps working.
# -----------------------------------------------------------------------------
if __name__ == '__main__':
  _app.run(main)
