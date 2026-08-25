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
"""How to drive the simulation from an external pose stream.

  python 31_mocap_stream.py

Tutorial 30 DISPLAYED a hardware stream. This one lets a stream DRIVE the
physics: poses from a "motion capture system" (a thread producing bursty,
irregularly-timed samples, like a real mocap bridge or teleop device)
steer a mocap body, and a weld constraint drags a physical box along.
This is the teleoperation / mocap-retargeting topology.

The new wrinkle is WHERE the stream enters. Telemetry in tutorial 30
targeted the viewer, so it rode the framework's channels. This stream
targets the SIM -- and between a thread you own and your own sim loop
there is no framework channel. You need the same latest-wins semantics,
so we build it ourselves:

    mailbox = collections.deque(maxlen=1)     # put: mailbox.append(msg)
                                              # get: mailbox.popleft()

A deque of length 1 with append/popleft IS a latest-wins channel: append
on a full deque discards the oldest element, atomically, thanks to the
GIL. Three lines, thread-safe, and precisely the semantics of the
framework's SnapshotChannel. The lesson of the whole tutorial series in
miniature: Event and Snapshot are DESIGN CONCEPTS; the framework ships
them for the sim<->viewer hop, and you can (and should) reproduce them
for any other hop in your system. (For the reliable/ordered analogue,
queue.Queue is your EventChannel.)

Also called out below, in flashing lights, because it has burned every
robotics lab at least once: MuJoCo quaternions are (w, x, y, z); ROS and
most mocap vendors send (x, y, z, w). The conversion is one line; the
debugging session when you forget it is not.
"""

import collections
import dataclasses
import math
import random
import threading
import time

from absl import app as _app
import mujoco
from mujoco.experimental.studio import launch_passive
from mujoco.experimental.studio import messages
from mujoco.experimental.studio import sim
from mujoco.experimental.studio import viewer_app
from mujoco.experimental.studio import viewer_protocol
import numpy as np

# A mocap body (the translucent red target) welded to a free blue box.
# Mocap bodies are kinematic: the stream sets their pose directly, and the
# weld constraint turns that pose into forces on the dynamic box -- soft
# tracking rather than teleportation, which is what you want from noisy
# input devices.
_XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="2 2 .05" rgba=".8 .8 .85 1"/>
    <body name="target" mocap="true" pos="0 0 .6">
      <geom type="sphere" size=".04" rgba="1 .3 .3 .4" contype="0" conaffinity="0"/>
    </body>
    <body name="box" pos="0 0 .6">
      <freejoint/>
      <geom type="box" size=".06 .06 .06" rgba=".3 .5 .9 1" mass=".5"/>
    </body>
  </worldbody>
  <equality>
    <weld body1="target" body2="box" solref=".05 1"/>
  </equality>
</mujoco>
"""


# Note: NOT a messages.Snapshot. This message never crosses the framework's
# channels -- it lives in our own mailbox -- so it does not need to (though
# deriving from Snapshot would be harmless and self-documenting).
@dataclasses.dataclass(frozen=True)
class PoseSample:
  """One mocap sample, modeled on geometry_msgs/PoseStamped.

  Attributes:
    stamp: producer-side time.monotonic() (see tutorial 30 on stamps).
    frame_id: the coordinate frame the pose is expressed in. Streams that
      do not name their frame get misinterpreted eventually; naming it
      costs one string. We only accept 'world' here and say so loudly.
    position: (3,) position in meters.
    quat_xyzw: (4,) orientation in the sender's convention -- x, y, z, w,
      as ROS and most mocap vendors emit. Conversion happens at exactly
      one place, in the consumer, where it can be seen and audited.
  """

  stamp: float
  frame_id: str
  position: np.ndarray
  quat_xyzw: np.ndarray


def fake_mocap_bridge(
    mailbox: collections.deque, stop: threading.Event
) -> None:
  """Streams a slow figure-eight, with realistic timing misbehavior."""
  t0 = time.monotonic()
  while not stop.is_set():
    now = time.monotonic()
    t = now - t0

    # A figure-eight with a slow spin about z.
    position = np.array([
        0.35 * math.sin(0.7 * t),
        0.25 * math.sin(1.4 * t),
        0.6 + 0.15 * math.sin(0.9 * t),
    ])
    half_yaw = 0.5 * (0.5 * t)
    # Built directly in xyzw order -- the sender's convention.
    quat_xyzw = np.array([0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw)])

    mailbox.append(  # Full mailbox? The old sample is silently replaced.
        PoseSample(
            stamp=now,
            frame_id='world',
            position=position,
            quat_xyzw=quat_xyzw,
        )
    )

    # Realistic transport timing: mostly ~120 Hz, sometimes a burst of
    # back-to-back samples, sometimes a 200 ms hiccup. The consumer is
    # insulated from all of it by the mailbox: it reads at most one fresh
    # sample per loop iteration, always the newest.
    r = random.random()
    if r < 0.05:
      time.sleep(0.2)  # Hiccup.
    elif r < 0.15:
      pass  # Burst: no sleep, next sample immediately.
    else:
      time.sleep(1.0 / 120.0)


def main(argv: list[str]) -> None:
  del argv
  model = mujoco.MjModel.from_xml_string(_XML)
  data = mujoco.MjData(model)

  # The mocap body's index into data.mocap_pos/mocap_quat. (body_mocapid
  # is -1 for ordinary bodies.)
  mocap_id = int(model.body('target').mocapid[0])

  config = viewer_protocol.ViewerConfig(title='31_mocap_stream')

  # Our hand-rolled latest-wins channel, and the bridge that fills it.
  mailbox: collections.deque[PoseSample] = collections.deque(maxlen=1)
  stop = threading.Event()
  bridge = threading.Thread(
      target=fake_mocap_bridge, args=(mailbox, stop), daemon=True
  )

  with launch_passive.launch_passive(
      config,
      viewer_handlers=[viewer_app.ViewerApp()],
  ) as handle:
    handle.send_to_viewer(messages.ModelEvent(model=model))
    bridge.start()

    try:
      step_control = sim.StepControl()
      while handle.is_running():
        # Drain the mailbox: zero samples (keep the last pose -- a mocap
        # body holds its pose until told otherwise) or one sample (the
        # newest; anything older was already discarded by the deque).
        try:
          sample = mailbox.popleft()
        except IndexError:
          sample = None

        if sample is not None and sample.frame_id == 'world':
          data.mocap_pos[mocap_id] = sample.position
          # THE quaternion line. xyzw (sender) -> wxyz (MuJoCo). If your
          # streamed body ever renders tumbled by 90-or-180 degrees in a
          # way that "almost" looks right, come straight back to this line.
          x, y, z, w = sample.quat_xyzw
          data.mocap_quat[mocap_id] = np.array([w, x, y, z])

        step_control.advance(model, data)
        model, data, step_control = handle.sync(model, data, step_control)
    finally:
      stop.set()
      bridge.join(timeout=1.0)


# -----------------------------------------------------------------------------
# Things to try:
#
#   - Watch the box chase the target through hiccups and bursts: timing
#     chaos in, smooth tracking out. Comment out the weld in the XML to
#     see what "kinematic only" looks like instead.
#   - Pause the sim: the target keeps moving (mocap poses are data, not
#     physics), the box freezes -- and catches up on unpause.
#   - Swap deque(maxlen=1) for queue.Queue() and drain it fully each
#     iteration during a burst: same result here, but now a slow consumer
#     builds a backlog and the box lags reality. Latest-wins is not an
#     optimization, it is a correctness choice for pose streams.
#   - Change the [w, x, y, z] line to pass quat_xyzw through unchanged and
#     watch the spin go wrong. Better to see it once here than in the lab.
# -----------------------------------------------------------------------------
if __name__ == '__main__':
  _app.run(main)
