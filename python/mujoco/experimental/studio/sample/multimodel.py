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
"""Multi-model gallery: an asset-browser page of 3D model cards.

The sim side loads several MuJoCo models and sends each to the viewer as a
``ModelEvent`` with its own ``model_id``, which lands it in the viewer's
model registry. The Gallery plugin declares one client view per registry
entry: on the web viewer, the browser fetches each model from
/model?id=<name> and renders every card itself with the modular filament
renderer (mjrfilament) — no images are streamed; only the models (once) and
their small state vectors travel over the wire.

The hovered card comes alive: its entry simulates in real time viewer-side
(each card owns a StepControl; the posed state streams to the browser), the
mouse controls its camera (left-drag orbit, right-drag pan, scroll zoom) and
Backspace resets it. Unhovered cards freeze.

Run (from the repo root, so the default model paths resolve):
  python -m mujoco.experimental.studio.sample.multimodel --gfx=web
"""

import os
import time

from absl import app as _app
from absl import flags as _flags
import mujoco
from mujoco.experimental.studio import launch_passive
from mujoco.experimental.studio import messages
from mujoco.experimental.studio import parser
from mujoco.experimental.studio import sim
from mujoco.experimental.studio import studio_app_events
from mujoco.experimental.studio import ux
from mujoco.experimental.studio import viewer_protocol

from mujoco.experimental.dear_imgui import dear_imgui as imgui

vp = viewer_protocol

_DEFAULT_MODELS = ','.join([
    'model/cards/cards.xml',
    'model/humanoid/humanoid.xml',
    'model/car/car.xml',
    'model/mug/mug.xml',
])

_MODELS = _flags.DEFINE_string(
    'models', _DEFAULT_MODELS, 'Comma-separated model paths.'
)
_GFX = _flags.DEFINE_enum(
    'gfx', 'web', vp.GFX_MODES, 'Graphics mode ("web" launches Web Viewer).'
)
_PORT = _flags.DEFINE_integer(
    'port', 0, 'Web Viewer port (0 picks first free port >= 8080).'
)

_CARD_IMAGE_SIZE = (256, 256)


class Pacer:
  """Sim plugin that paces the loop by sleeping on each StepEvent.

  The gallery's sim side has no physics: nothing steps, but the sim loop
  still calls ``handle.sync`` to pump messages, so it needs a heartbeat to
  avoid busy-spinning (``sync`` never sleeps). Card physics runs viewer-side
  on the registry entries.
  """

  def __init__(self, hz: float = 60.0) -> None:
    self._period = 1.0 / hz

  @messages.handler
  def on_step(self, event: messages.StepEvent) -> bool:
    del event
    time.sleep(self._period)
    return False


class AssetCard:
  """One gallery card: a client-rendered view of a registry entry."""

  def __init__(
      self, name: str, entry: vp.ModelEntry, view: vp.ClientView, path: str
  ) -> None:
    self.name = name
    self.entry = entry
    self.view = view
    # Each card owns its stepping: real-time paced while the card is hovered.
    self.step_control = sim.StepControl()
    self.hovered = False
    model = entry.model
    try:
      size_kb = os.path.getsize(path) / 1024.0 if path else 0.0
    except OSError:
      size_kb = 0.0
    self.stats = [
        f'bodies {model.nbody}   geoms {model.ngeom}',
        f'dofs {model.nv}   actuators {model.nu}',
        f'meshes {model.nmesh}   file {size_kb:.0f} KB',
    ]


class Gallery:
  """Viewer plugin laying out registry entries as a browsable card grid."""

  def __init__(self) -> None:
    self._viewer: vp.Viewer | None = None
    self._cards: dict[str, AssetCard] = {}
    self._filter = ''

  @messages.handler
  def on_viewer_init(self, event: vp.ViewerInitEvent) -> None:
    self._viewer = event.viewer

  @messages.handler
  def on_model(self, event: messages.ModelEvent) -> bool:
    """Turns every non-default registry model into a gallery card."""
    model_id = getattr(event, 'model_id', vp.DEFAULT_MODEL_ID)
    if model_id == vp.DEFAULT_MODEL_ID or self._viewer is None:
      return False
    # The Viewer's CRITICAL handler already placed the entry in the registry.
    entry = self._viewer.models.get(model_id)
    if entry is None:
      return False
    view = self._viewer.add_client_view(
        model_id, model_id=model_id, size=_CARD_IMAGE_SIZE
    )
    self._cards[model_id] = AssetCard(model_id, entry, view, event.path)
    return False

  @messages.handler
  def on_build_gui(self, _: messages.BuildGuiEvent) -> None:
    assert self._viewer is not None
    io = imgui.GetIO()
    imgui.SetNextWindowPos(imgui.Vec2(0, 0))
    imgui.SetNextWindowSize(imgui.Vec2(io.DisplaySize.x, io.DisplaySize.y))
    flags = int(imgui.WindowFlags.NoTitleBar) | int(
        imgui.WindowFlags.NoResize
    ) | int(imgui.WindowFlags.NoMove) | int(
        imgui.WindowFlags.NoBringToFrontOnFocus
    )
    if imgui.Begin('Gallery', flags=flags):
      imgui.Text('Asset gallery')
      imgui.SameLine()
      _, self._filter = imgui.InputText('Filter', self._filter)
      imgui.Separator()

      cards = [
          c
          for name, c in sorted(self._cards.items())
          if self._filter.lower() in name.lower()
      ]
      card_w = _CARD_IMAGE_SIZE[0] + 16.0
      columns = max(1, int(imgui.GetContentRegionAvail().x // card_w))
      for i, card in enumerate(cards):
        if i % columns != 0:
          imgui.SameLine()
        self._draw_card(card)
    imgui.End()

  def _draw_card(self, card: AssetCard) -> None:
    card_size = imgui.Vec2(
        _CARD_IMAGE_SIZE[0] + 12.0, _CARD_IMAGE_SIZE[1] + 110.0
    )
    if imgui.BeginChild(
        f'##card_{card.name}',
        card_size,
        int(imgui.ChildFlags.Borders),
        int(imgui.WindowFlags.NoScrollWithMouse),
    ):
      imgui.Text(card.name)
      imgui.Separator()

      # The hovered card comes alive: real-time simulation, Backspace reset.
      # Unhovered cards freeze; the browser keeps showing their last pose.
      hovered = imgui.IsWindowHovered()
      entry = card.entry
      if hovered:
        if not card.hovered:
          # Hover start: re-sync pacing so the card doesn't try to catch up
          # the wall-clock time it spent frozen.
          card.step_control.force_sync()
        studio_app_events.handle_reset_keyboard_events(entry.model, entry.data)
        card.step_control.advance(entry.model, entry.data)
      card.hovered = hovered

      # The card image is the client-rendered view of the registry entry: the
      # browser renders it with the modular filament renderer and maps the
      # view's texture id to the render target.
      imgui.Image(
          card.view.tex_id,
          imgui.Vec2(float(_CARD_IMAGE_SIZE[0]), float(_CARD_IMAGE_SIZE[1])),
      )
      self._handle_card_mouse(card)

      imgui.Separator()
      for line in card.stats:
        imgui.TextDisabled(line)
      imgui.TextDisabled('hover: simulate + drag/scroll, Backspace: reset')
    imgui.EndChild()

  def _handle_card_mouse(self, card: AssetCard) -> None:
    """Camera controls on the card's 3D view (the last submitted item)."""
    if not imgui.IsItemHovered():
      return
    io = imgui.GetIO()
    dx = io.MouseDelta.x / _CARD_IMAGE_SIZE[0]
    dy = io.MouseDelta.y / _CARD_IMAGE_SIZE[1]
    camera = card.view.camera
    model, data = card.entry.model, card.entry.data
    if dx != 0.0 or dy != 0.0:
      if imgui.IsMouseDown(imgui.MouseButton.Left):
        ux.MoveCamera(model, data, camera, ux.CameraMotion.ORBIT, dx, dy)
      elif imgui.IsMouseDown(imgui.MouseButton.Right):
        motion = (
            ux.CameraMotion.PLANAR_MOVE_H
            if io.KeyShift
            else ux.CameraMotion.PLANAR_MOVE_V
        )
        ux.MoveCamera(model, data, camera, motion, dx, dy)
      elif imgui.IsMouseDown(imgui.MouseButton.Middle):
        ux.MoveCamera(model, data, camera, ux.CameraMotion.ZOOM, dx, dy)
    if io.MouseWheel != 0.0:
      ux.MoveCamera(
          model, data, camera, ux.CameraMotion.ZOOM, 0.0, -io.MouseWheel / 50.0
      )


def main(argv: list[str]) -> None:
  del argv
  # Load the asset models sim-side; each travels to the viewer's registry as
  # a ModelEvent with its own model_id.
  assets = []
  for path in _MODELS.value.split(','):
    path = path.strip()
    if not path:
      continue
    data = parser.parse(path)
    if data is None:
      print(f'Skipping {path!r}: failed to load')
      continue
    name = os.path.splitext(os.path.basename(path))[0]
    assets.append((name, data.model, path))
  if not assets:
    raise _app.UsageError('No models could be loaded.')

  config = vp.ViewerConfig(
      title='Asset gallery',
      gfx=_GFX.value or 'web',
      http_port=_PORT.value,
  )

  with launch_passive.launch_passive(
      config,
      viewer_plugins=[Gallery()],
      sim_plugins=[Pacer()],
  ) as handle:
    # The page has no simulation, but the viewer still needs a (trivial)
    # main model so the state stream stays alive for the browser.
    model = mujoco.MjModel.from_xml_string('<mujoco/>')
    data = mujoco.MjData(model)
    handle.send_to_viewer(messages.ModelEvent(model=model))
    for name, asset_model, path in assets:
      handle.send_to_viewer(
          messages.ModelEvent(model=asset_model, path=path, model_id=name)
      )

    try:
      while handle.is_running():
        model, data = handle.sync(model, data)
    except KeyboardInterrupt:
      print('\nShutting down.', flush=True)


if __name__ == '__main__':
  _app.run(main)
