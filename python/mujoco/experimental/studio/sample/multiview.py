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
"""Multi-view Studio: every model camera in its own live viewport.

Runs the full Studio viewer on a simulated model and adds one ImGui window
per fixed model camera (plus an extra orbiting free view), each declared as a
client view: on the web viewer the browser renders every viewport itself
with the modular filament renderer (mjrfilament), sharing the main scene —
cameras are evaluated client-side against the freshest state, so tracking
and egocentric views follow with no latency. The main 3D view stays the
interactive free camera in the docked central rect.

Run (from the repo root; humanoid.xml defines back/side/egocentric cameras):
  python -m mujoco.experimental.studio.sample.multiview --gfx=web
"""

import sys

from absl import app as _app
from absl import flags as _flags
import mujoco
from mujoco.experimental.studio import launch_passive
from mujoco.experimental.studio import messages
from mujoco.experimental.studio import parser
from mujoco.experimental.studio import step_control
from mujoco.experimental.studio import viewer_app
from mujoco.experimental.studio import viewer_protocol

from mujoco.experimental.dear_imgui import dear_imgui as imgui

vp = viewer_protocol

_MODEL = _flags.DEFINE_string(
    'model', 'model/humanoid/humanoid.xml', 'Path to model file.'
)
_GFX = _flags.DEFINE_enum(
    'gfx', 'web', vp.GFX_MODES, 'Graphics mode ("web" launches Web Viewer).'
)
_PORT = _flags.DEFINE_integer(
    'port', 0, 'Web Viewer port (0 picks first free port >= 8080).'
)

_VIEW_SIZE = (320, 240)


class CameraViews:
  """Viewer plugin adding a live client-rendered window per model camera."""

  def __init__(self) -> None:
    self._viewer: vp.Viewer | None = None
    self._views: list[vp.ClientView] = []

  def _rebuild_views(self) -> None:
    """(Re)declares one view per fixed model camera plus a free orbit view."""
    assert self._viewer is not None
    for view in self._views:
      self._viewer.remove_client_view(view.name)
    self._views = []
    model = self._viewer.model
    for cam_id in range(model.ncam):
      name = (
          mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_id)
          or f'camera {cam_id}'
      )
      camera = mujoco.MjvCamera()
      camera.type = int(mujoco.mjtCamera.mjCAMERA_FIXED)
      camera.fixedcamid = cam_id
      self._views.append(
          self._viewer.add_client_view(name, camera=camera, size=_VIEW_SIZE)
      )
    self._views.append(
        self._viewer.add_client_view('free orbit', size=_VIEW_SIZE)
    )

  @messages.handler
  def on_viewer_init(self, event: vp.ViewerInitEvent) -> None:
    self._viewer = event.viewer
    self._rebuild_views()

  @messages.handler
  def on_model(self, event: messages.ModelEvent) -> bool:
    if getattr(event, 'model_id', vp.DEFAULT_MODEL_ID) != vp.DEFAULT_MODEL_ID:
      return False
    if self._viewer is not None:
      self._rebuild_views()
    return False

  @messages.handler
  def on_build_gui(self, _: messages.BuildGuiEvent) -> None:
    assert self._viewer is not None
    for view in self._views:
      if imgui.Begin(view.name, flags=int(imgui.WindowFlags.AlwaysAutoResize)):
        if view.name == 'free orbit':
          view.camera.azimuth += 0.5  # Slow turntable.
        imgui.Image(
            view.tex_id,
            imgui.Vec2(float(view.size[0]), float(view.size[1])),
        )
      imgui.End()


def main(argv: list[str]) -> None:
  model_path = _MODEL.value or (
      argv[1] if len(argv) > 1 and not argv[1].startswith('--') else None
  )
  data = parser.parse(model_path)
  if data is None:
    print(f'Failed to load model from {model_path!r}')
    sys.exit(1)
  model = data.model

  config = vp.ViewerConfig(
      title='Multi-view Studio',
      gfx=_GFX.value or 'web',
      http_port=_PORT.value,
  )

  with launch_passive.launch_passive(
      config,
      viewer_plugins=[viewer_app.ViewerApp(), CameraViews()],
      sim_plugins=[step_control.StepControl()],
  ) as handle:
    handle.send_to_viewer(messages.ModelEvent(model=model, path=model_path))

    try:
      while handle.is_running():
        model, data = handle.sync(model, data)
    except KeyboardInterrupt:
      print('\nShutting down.', flush=True)


if __name__ == '__main__':
  _app.run(main)
