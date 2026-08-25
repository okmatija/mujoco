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
"""How to display a robot hardware data stream next to the simulation.

  python 30_robot_telemetry.py

The sim-to-real workflow in one picture: a physical robot streams its
joint positions over some transport (UDP, CAN, a ROS topic); you want to
see that "real" pose IN the viewer, as a ghost overlaid on the simulated
twin, and you want to notice immediately when the stream goes stale.

Here the "robot" is a background thread generating a trajectory -- exactly
where a socket-reading loop would sit -- and it deliberately drops out for
a second every few seconds so you can watch the staleness handling work.
Everything else is real: the message design, the threading, the display.

The message design is borrowed from ROS, whose sensor_msgs/JointState has
carried two decades of hardware streams. Three of its decisions are worth
copying verbatim:

  1. NAME-KEYED PARALLEL ARRAYS. The payload maps joint NAMES to values
     rather than assuming an index order. A driver can report any subset
     of joints (real drivers do: separate arms, missing encoders) and the
     consumer matches by name against ITS model. Renamed or reordered
     joints degrade gracefully instead of silently corrupting poses.

  2. PRODUCER-SIDE TIMESTAMPS. The driver stamps each sample when it is
     TAKEN, not when it arrives. Arrival time lies: transports jitter,
     batch and reorder. With a stamp the consumer can measure staleness,
     drop out-of-order samples, and plot against a true time axis. We use
     time.monotonic() -- wall clocks jump (NTP, DST); monotonic doesn't.
     (ROS uses integer {sec, nanosec} for long-uptime precision; float
     seconds are fine for a viewer tool.)

  3. TELEMETRY IS LATEST-WINS. A 500 Hz stream into a 60 Hz viewer must
     drop intermediate samples -- the newest pose is the only one worth
     rendering. That is precisely a Snapshot, so the channel does the
     dropping for us. (ROS spells the same decision "best-effort QoS,
     shallow queue"; its reliable+queued mode corresponds to our Event,
     which is what you would use for the rare, must-not-miss messages:
     e-stop engaged, calibration finished, error codes.)

Also demonstrated: handle.send_to_viewer() is thread-safe, so the driver
thread publishes directly -- no hand-rolled locking between the driver,
the sim loop and the viewer. Immutable messages over channels are what
make that composition safe.
"""

import dataclasses
import math
import threading
import time

from absl import app as _app
import mujoco
from mujoco.experimental.studio import launch_passive
from mujoco.experimental.studio import messages
from mujoco.experimental.studio import sim
from mujoco.experimental.studio import viewer_app
from mujoco.experimental.studio import viewer_handle
from mujoco.experimental.studio import viewer_protocol
import numpy as np

from mujoco.experimental.dear_imgui import dear_imgui as imgui

# A 3-joint arm. The sim's copy swings passively under gravity; the ghost
# shows what the "real" arm is doing. Note the driver below only reports
# shoulder and elbow -- the wrist has "no encoder", a realistically
# incomplete stream.
_XML = """
<mujoco>
  <option timestep=".002"/>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="1.5 1.5 .05" rgba=".8 .8 .85 1"/>
    <body name="base" pos="0 0 .6">
      <geom type="cylinder" size=".06 .04" rgba=".4 .4 .45 1"/>
      <body name="link1">
        <joint name="shoulder" type="hinge" axis="0 1 0" damping=".1"/>
        <geom type="capsule" fromto="0 0 0 .35 0 0" size=".03" rgba=".3 .5 .9 1"/>
        <body name="link2" pos=".35 0 0">
          <joint name="elbow" type="hinge" axis="0 1 0" damping=".1"/>
          <geom type="capsule" fromto="0 0 0 .3 0 0" size=".025" rgba=".9 .5 .3 1"/>
          <body name="link3" pos=".3 0 0">
            <joint name="wrist" type="hinge" axis="0 1 0" damping=".1"/>
            <geom type="capsule" fromto="0 0 0 .15 0 0" size=".02" rgba=".5 .9 .3 1"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

_STALE_AFTER_S = 0.35  # Older telemetry than this is flagged stale.


@dataclasses.dataclass(frozen=True)
class JointTelemetrySnapshot(messages.Snapshot):
  """Joint positions sampled from the robot. Modeled on sensor_msgs/JointState.

  Attributes:
    stamp: time.monotonic() at the moment of sampling, PRODUCER side.
    names: joint names, parallel to positions. May be any subset of the
      robot's joints, in any order.
    positions: joint positions in radians (or meters for sliders). SI
      units and radians everywhere -- unit chaos is a self-inflicted
      wound; pick the convention at the message boundary and stick to it.
  """

  stamp: float
  names: tuple[str, ...]
  positions: np.ndarray


# -----------------------------------------------------------------------------
# The "robot driver". In a real tool this function is your transport
# reader: recv() from a socket, poll a CAN bus, subscribe to a ROS topic.
# Its only obligations are the ones shown: sample, stamp, publish.
# -----------------------------------------------------------------------------


def fake_robot_driver(
    handle: viewer_handle.ViewerHandle, stop: threading.Event
) -> None:
  """Publishes a joint trajectory at 50 Hz, dropping out periodically."""
  rate_hz = 50.0
  t0 = time.monotonic()
  while not stop.is_set():
    now = time.monotonic()
    t = now - t0

    # Simulated connection loss: silent for 1.5 s out of every 6 s. Watch
    # the ghost freeze and turn grey during the gap -- and note that when
    # the stream resumes, the ghost SNAPS to the current pose. Latest-wins
    # never replays a backlog; a queued design would slew through 1.5 s of
    # dead history instead. For telemetry, snapping is correct.
    if t % 6.0 > 4.5:
      time.sleep(0.05)
      continue

    # The "real robot" trajectory. Only two of the three joints report.
    try:
      handle.send_to_viewer(
          JointTelemetrySnapshot(
              stamp=now,
              names=('shoulder', 'elbow'),
              positions=np.array([
                  0.9 * math.sin(0.8 * t),
                  0.7 * math.sin(1.3 * t + 1.0),
              ]),
          )
      )
    except RuntimeError:
      return  # The endpoint closed (window closing); stop publishing.
    time.sleep(1.0 / rate_hz)


# -----------------------------------------------------------------------------
# The viewer-side consumer: telemetry in, ghost + status panel out.
# -----------------------------------------------------------------------------


class TelemetryGhost:
  """Renders the latest telemetry as a ghost pose of the same model."""

  def __init__(self) -> None:
    self._viewer: viewer_protocol.Viewer | None = None
    self._ghost_data: mujoco.MjData | None = None
    self._latest: JointTelemetrySnapshot | None = None

  @messages.handler
  def on_viewer_init(self, event: viewer_protocol.ViewerInitEvent) -> None:
    self._viewer = event.viewer
    self._rebuild()

  @messages.handler
  def on_model(self, event: messages.ModelEvent) -> bool:
    del event
    self._rebuild()
    return False

  def _rebuild(self) -> None:
    """(Re)creates the scratch MjData used to pose the ghost.

    The ghost needs its own MjData: we write telemetry into its qpos and
    run mj_forward to get world-space geom poses, without disturbing the
    data being rendered. Same model, second data -- MjData is designed to
    be cheap to duplicate for exactly this kind of use.
    """
    assert self._viewer is not None
    self._ghost_data = mujoco.MjData(self._viewer.model)
    self._latest = None

  @messages.handler
  def on_telemetry(self, snapshot: JointTelemetrySnapshot) -> None:
    # Guard against out-of-order samples by comparing STAMPS, not arrival
    # order. Redundant for this in-process channel (which preserves order),
    # but the habit is what makes multi-source and networked transports
    # safe -- robot_state_publisher does the same thing.
    if self._latest is not None and snapshot.stamp <= self._latest.stamp:
      return
    self._latest = snapshot

  def _age(self) -> float | None:
    if self._latest is None:
      return None
    return time.monotonic() - self._latest.stamp

  @messages.handler
  def on_update(self, _: messages.UpdateEvent) -> None:
    assert self._viewer is not None
    model = self._viewer.model
    ghost = self._ghost_data
    out = self._viewer.extra_geoms
    out.clear()
    if ghost is None or self._latest is None or model.ngeom == 0:
      return

    # Write telemetry into the ghost's qpos BY NAME. Unknown names are
    # skipped (the stream might describe a different robot than the loaded
    # model); unreported joints keep their previous value -- for the wrist
    # that means "wherever it was", which is honest: we have no data.
    for name, position in zip(self._latest.names, self._latest.positions):
      try:
        adr = model.joint(name).qposadr[0]
      except KeyError:
        continue
      ghost.qpos[adr] = position
    mujoco.mj_forward(model, ghost)

    # Fresh telemetry renders green-ish; stale renders grey. A pose with
    # no color-coded age is a trap: it looks authoritative long after the
    # robot stopped talking.
    age = self._age()
    if age is not None and age > _STALE_AFTER_S:
      rgba = np.array([0.55, 0.55, 0.55, 0.35], dtype=np.float32)
    else:
      rgba = np.array([0.3, 0.9, 0.4, 0.45], dtype=np.float32)

    # Emit ghost geoms at the telemetry pose (the tutorial-10 pattern).
    for i in range(model.ngeom):
      body_id = model.geom_bodyid[i]
      if model.body_weldid[body_id] == 0:  # Skip world-fixed geoms.
        continue
      geom = mujoco.MjvGeom()
      mujoco.mjv_initGeom(
          geom,
          int(model.geom_type[i]),
          model.geom_size[i],
          ghost.geom_xpos[i],
          ghost.geom_xmat[i].flatten(),
          rgba,
      )
      geom.dataid = model.geom_dataid[i]
      out.append(geom)

  @messages.handler
  def on_build_gui(self, _: messages.BuildGuiEvent) -> None:
    if imgui.Begin('Telemetry', flags=int(imgui.WindowFlags.AlwaysAutoResize)):
      age = self._age()
      if age is None:
        imgui.Text('no data received')
      else:
        stale = age > _STALE_AFTER_S
        imgui.Text(f'{"STALE" if stale else "live "}   age: {age*1000:6.0f} ms')
        assert self._latest is not None
        for name, position in zip(self._latest.names, self._latest.positions):
          imgui.Text(f'  {name:10s} {position:+.3f} rad')
    imgui.End()


def main(argv: list[str]) -> None:
  del argv
  model = mujoco.MjModel.from_xml_string(_XML)
  data = mujoco.MjData(model)

  config = viewer_protocol.ViewerConfig(title='30_robot_telemetry')

  with launch_passive.launch_passive(
      config,
      viewer_handlers=[viewer_app.ViewerApp(), TelemetryGhost()],
  ) as handle:
    handle.send_to_viewer(messages.ModelEvent(model=model))

    # Start the driver AFTER the viewer is up, stop it before shutdown.
    # send_to_viewer is safe to call from this thread while the sim loop
    # calls sync() -- channels are the designated thread boundary.
    stop = threading.Event()
    driver = threading.Thread(
        target=fake_robot_driver, args=(handle, stop), daemon=True
    )
    driver.start()

    try:
      step_control = sim.StepControl()
      while handle.is_running():
        step_control.advance(model, data)
        model, data, step_control = handle.sync(model, data, step_control)
    finally:
      stop.set()
      driver.join(timeout=1.0)


# -----------------------------------------------------------------------------
# Things to try:
#
#   - Watch a dropout (every ~6 s): ghost freezes, greys out, the panel
#     counts the age up -- then snaps to the live pose on reconnect.
#   - Ctrl+Right-drag the SIM arm away from the ghost: this is the
#     sim-vs-real comparison view every hardware bring-up wants.
#   - Add `velocities` to the snapshot and draw tutorial-12 arrows on the
#     ghost: you now have a live hardware dashboard.
#   - Replace fake_robot_driver's trajectory with a real transport read;
#     nothing else in the file needs to change. That is the point.
# -----------------------------------------------------------------------------
if __name__ == '__main__':
  _app.run(main)
