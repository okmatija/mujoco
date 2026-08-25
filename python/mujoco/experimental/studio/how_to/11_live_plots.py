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
"""How to plot live simulation data with ImPlot: an "Inspect Body" tool.

  python 11_live_plots.py humanoid.xml

Live plots are the single most requested debugging aid when authoring RL
environments: which reward term is misbehaving, is the torso height stable,
what is the actuator really doing. This tutorial builds the general shape
of such tools -- select a thing in the 3D view, see its signals plotted
over time -- using ImPlot, the plotting library that ships alongside
Studio's ImGui bindings.

  - Double-click a body in the viewer to select it; an "Inspect Body"
    window appears with its centroid position and orientation (as Euler
    angles) over the last 100 frames.

What it demonstrates:

  - PICKING FROM A HANDLER. ViewerApp already implements double-click
    selection and stores the result in viewer.perturb.select. Handlers can
    simply read it -- interaction state is shared viewer-side state, not a
    message.

  - THE ViewerAppInitEvent IDIOM. Like ViewerInitEvent in tutorial 04, but
    delivers the ViewerApp instance -- for handlers that want the Studio
    application (model, data, AND ui state), not just the bare viewer.

  - RESPONSIVE GUI LAYOUT. The plot area adapts to the window: side by
    side when wide, stacked when tall, and axis decorations drop away as
    plots shrink. Immediate-mode GUIs make such rules one-liners; this
    matters more than it sounds once your tool shares screen space with
    the Studio panels.

  - HISTORY AT RENDER RATE. Like the ghost in tutorial 10, the ring buffer
    is filled once per BuildGuiEvent (per frame), viewer-side. For
    per-physics-step signals you would instead sample on the sim side and
    publish a Snapshot (tutorials 03 and 20).
"""

import math
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
from mujoco.experimental.implot import implot

_N_HISTORY = 100

# ImPlot is configured with flag bitmasks, like ImGui. We disable the
# interactions that make no sense for a live scrolling plot.
_PLOT_FLAGS = (
    implot.Flags.NoInputs.value  # Disable pan/zoom mouse interaction.
    | implot.Flags.NoMenus.value  # Disable right-click context menu.
    | implot.Flags.NoBoxSelect.value  # Disable drag-to-select regions.
)

_AXIS_FLAGS = (
    implot.AxisFlags.NoGridLines.value  # Hide background grid lines.
    | implot.AxisFlags.NoTickMarks.value  # Hide small tick marks on the axis.
)


# -----------------------------------------------------------------------------
# Responsive-layout helpers: progressively remove chrome as space shrinks.
# Each takes the plot size it will be drawn at and picks flags accordingly.
# -----------------------------------------------------------------------------


def _setup_plot_flags(plot_size: imgui.Vec2) -> int:
  flags = _PLOT_FLAGS
  if min(plot_size.x, plot_size.y) < 300:
    flags |= implot.Flags.NoTitle.value
  if min(plot_size.x, plot_size.y) < 200:
    flags |= implot.Flags.NoLegend.value
  return flags


def _setup_time_axis(plot_size: imgui.Vec2) -> None:
  flags = _AXIS_FLAGS
  if plot_size.x < 300:
    flags |= implot.AxisFlags.NoTickLabels.value
  implot.SetupAxis(implot.Axis.X1, '', flags)
  implot.SetupAxisLimits(implot.Axis.X1, 0, _N_HISTORY)


def _setup_xpos_axis(centroid: list[np.ndarray], plot_size: imgui.Vec2) -> None:
  """Y axis for positions: auto-fit the data with a small margin."""
  flags = _AXIS_FLAGS
  if plot_size.y < 300:
    flags |= implot.AxisFlags.NoTickLabels.value
  implot.SetupAxis(implot.Axis.Y1, '', flags)
  min_y = min(c[1] for c in centroid)
  max_y = max(c[1] for c in centroid)
  margin = max((max_y - min_y) * 0.1, 0.05)
  implot.SetupAxisLimits(
      implot.Axis.Y1,
      min_y - margin,
      max_y + margin,
      cond=implot.Cond.Always,
  )


def _setup_angle_axis(plot_size: imgui.Vec2) -> None:
  """Y axis for angles: fixed [-180, 180] range with meaningful ticks."""
  flags = _AXIS_FLAGS
  if plot_size.y < 300:
    flags |= implot.AxisFlags.NoTickLabels.value
  implot.SetupAxis(implot.Axis.Y1, '', flags)
  implot.SetupAxisLimits(implot.Axis.Y1, -185.0, 185.0)
  implot.SetupAxisTicks(
      implot.Axis.Y1,
      [-180.0, -90.0, 0.0, 90.0, 180.0],
      ['-180', '-90', '0', '90', '180'],
  )


class BodyInspector:
  """Handler that draws body-inspection plots using ImGui/ImPlot."""

  def __init__(self) -> None:
    self._app: viewer_app.ViewerApp | None = None
    # Fixed-length history, pre-filled with zeros: the plot always shows
    # _N_HISTORY samples, sliding left as new ones arrive.
    self._centroid: list[np.ndarray] = [np.zeros(3) for _ in range(_N_HISTORY)]
    self._euler: list[np.ndarray] = [np.zeros(3) for _ in range(_N_HISTORY)]
    self._body_id: int = -1

  @messages.handler
  def on_viewer_app_init(self, event: viewer_app.ViewerAppInitEvent) -> None:
    """Caches the ViewerApp reference on startup."""
    assert isinstance(event.viewer_app, viewer_app.ViewerApp)
    self._app = event.viewer_app

  @messages.handler
  def inspect_body(self, _: messages.BuildGuiEvent) -> None:
    """Renders the body-inspection charts in ImGui/ImPlot."""
    app = self._app
    if app is None:
      return

    # ViewerApp's double-click handling stores the picked body here; 0 means
    # "background". We latch the last real selection so the window survives
    # a deselecting click.
    if app.viewer.perturb.select > 0:
      self._body_id = app.viewer.perturb.select

    # Display selected body information.
    if self._body_id > 0:
      body_name = mujoco.mj_id2name(
          app.model, int(mujoco.mjtObj.mjOBJ_BODY), self._body_id
      )

      io = imgui.GetIO()
      imgui.SetNextWindowPos(
          imgui.Vec2(io.DisplaySize.x * 0.5, io.DisplaySize.y * 0.5),
          imgui.Cond.FirstUseEver,
          imgui.Vec2(0.5, 0.5),
      )
      imgui.SetNextWindowSize(imgui.Vec2(1200, 600), imgui.Cond.FirstUseEver)

      # Note: The window title uses the special "###" markup to ensure the
      # imgui ID for the window is constant for all body names. This is
      # needed for the window to retain its position/size across selections.
      window_title = (
          f'Inspect Body {body_name or "(???)"!r} ({self._body_id})###Plot'
      )
      if imgui.Begin(window_title):
        # The responsive rule: plots side by side in a wide window,
        # stacked in a tall one.
        avail = imgui.GetContentRegionAvail()
        wide = avail.x > avail.y

        # Add a small padding factor to prevent scrollbars.
        plot_size = imgui.Vec2(
            avail.x * 0.5 - 4 if wide else avail.x,
            avail.y if wide else avail.y * 0.5 - 4,
        )

        plot_flags = _setup_plot_flags(plot_size)
        if implot.BeginPlot('Centroid vs Time', plot_size, flags=plot_flags):
          _setup_time_axis(plot_size)
          _setup_xpos_axis(self._centroid, plot_size)
          implot.PlotLine(
              'x', range(_N_HISTORY), [c[0] for c in self._centroid]
          )
          implot.PlotLine(
              'y', range(_N_HISTORY), [c[1] for c in self._centroid]
          )
          implot.PlotLine(
              'z', range(_N_HISTORY), [c[2] for c in self._centroid]
          )
          implot.EndPlot()

        if wide:
          imgui.SameLine()

        if implot.BeginPlot('Euler Angle vs Time', plot_size, flags=plot_flags):
          _setup_time_axis(plot_size)
          _setup_angle_axis(plot_size)
          implot.PlotLine(
              'roll', range(_N_HISTORY), [e[0] for e in self._euler]
          )
          implot.PlotLine(
              'pitch', range(_N_HISTORY), [e[1] for e in self._euler]
          )
          implot.PlotLine('yaw', range(_N_HISTORY), [e[2] for e in self._euler])
          implot.EndPlot()
      imgui.End()

    # Advance the ring buffer, once per frame, whether or not the window is
    # visible -- so the history is already filled when a body is selected.
    self._centroid.pop(0)
    self._euler.pop(0)
    if self._body_id > 0 and self._body_id < app.model.nbody:
      self._centroid.append(app.data.xpos[self._body_id].copy())
      quat = app.data.xquat[self._body_id]
      mat = np.zeros(9)
      mujoco.mju_quat2Mat(mat, quat)
      # mat is row-major 3x3: R[i,j] = mat[3*i + j].
      roll = math.atan2(mat[7], mat[8])
      pitch = math.atan2(-mat[6], math.sqrt(mat[7] ** 2 + mat[8] ** 2))
      yaw = math.atan2(mat[3], mat[0])
      self._euler.append(np.degrees(np.array([roll, pitch, yaw])))
    else:
      self._centroid.append(np.zeros(3))
      self._euler.append(np.zeros(3))


def main(argv: list[str]) -> None:
  if len(argv) != 2:
    raise _app.UsageError('Please provide exactly one MJCF path argument.')

  if (data := parser.parse(argv[1])) is None:
    print(f'Error loading model from {argv[1]!r}')
    sys.exit(1)
  model = data.model

  config = viewer_protocol.ViewerConfig(
      title=os.path.basename(sys.argv[0]),
  )

  with launch_passive.launch_passive(
      config,
      viewer_handlers=[viewer_app.ViewerApp(), BodyInspector()],
  ) as handle:
    handle.send_to_viewer(messages.ModelEvent(model=model, path=argv[1]))

    step_control = sim.StepControl()
    while handle.is_running():
      step_control.advance(model, data)
      model, data, step_control = handle.sync(model, data, step_control)


# -----------------------------------------------------------------------------
# Things to try:
#
#   - Double-click different bodies: the window retitles but keeps its
#     placement (the "###" trick).
#   - Ctrl+Right-drag the selected body and watch the plots respond.
#   - Resize the inspect window from wide to narrow: plots restack and
#     shed their decorations. Try building that in a retained-mode UI.
# -----------------------------------------------------------------------------
if __name__ == '__main__':
  _app.run(main)
