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
"""Base class and configuration for any viewer."""

import abc
import copy
import dataclasses
import enum
from typing import Any
import mujoco
from mujoco.experimental.studio import endpoints
from mujoco.experimental.studio import messages
from mujoco.experimental.studio import plugin_registry
from mujoco.experimental.studio import ux
import numpy as np

GFX_MODES = (
    'classic',
    'classic_headless',
    'opengl',
    'opengl_headless',
    'opengl_software',
    'vulkan',
    'vulkan_software',
    'web',
    'webgl',
)


# -----------------------------------------------------------------------------
# Viewer configuration.
# -----------------------------------------------------------------------------


@dataclasses.dataclass
class ViewerConfig:
  """Common configuration for creating a viewer window."""

  title: str = ''
  width: int = 1200
  height: int = 800
  gfx: str = ''  # Graphics mode ('web' launches Web Viewer).
  http_port: int = 0  # Web Viewer port (0 picks first free port >= 8080).


# Legacy message types kept for backward compatibility.
# Will be removed when callers are migrated.


@dataclasses.dataclass
class SimToView:
  """A message sent from the simulation to the viewer."""

  model: mujoco.MjModel | None = None
  state: np.ndarray | None = None
  state_sig: int = 0
  user_data: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class ViewToSim:
  """A message sent from the viewer to the simulation."""

  state: np.ndarray | None = None
  state_sig: int = 0
  reset: bool = False
  send_rate: float = 60.0
  user_data: dict[str, Any] = dataclasses.field(default_factory=dict)


# -----------------------------------------------------------------------------
# Base class for any viewer.
# -----------------------------------------------------------------------------


# Reserved id of the simulated model in the viewer's model registry. It is
# fed by StateSnapshots from the sim side; ``viewer.model`` / ``viewer.data``
# alias this entry.
DEFAULT_MODEL_ID = 'default'


@dataclasses.dataclass
class ModelEntry:
  """A model registered with the viewer for display.

  The entry with id ``DEFAULT_MODEL_ID`` is the simulated model. Additional
  entries are viewer-local display models (ghosts, previews, props): the
  plugin that adds them owns their lifecycle and poses their ``data``.

  Attributes:
    model: The model.
    data: The data posed for display.
    source: 'sim' when state arrives from the sim side, 'local' otherwise.
    overlay: Whether the entry is drawn into the main 3D scene.
    visible: Toggles drawing without removing the entry.
    tint: Optional RGBA override applied to every overlay geom (e.g. a
      semi-transparent ghost color); None keeps the model's own colors.
    include_static: Whether overlay drawing includes geoms on static
      (world-welded) bodies.
  """

  model: mujoco.MjModel
  data: mujoco.MjData
  source: str = 'local'
  overlay: bool = True
  visible: bool = True
  tint: tuple[float, float, float, float] | None = None
  include_static: bool = False
  _scene: mujoco.MjvScene | None = dataclasses.field(default=None, repr=False)


@dataclasses.dataclass(frozen=True)
class ViewerInitEvent(messages.Event):
  """Lifecycle event dispatched once when the concrete Viewer is initialized.

  Plugins that need access to the Viewer should handle this event
  and cache the reference.
  """

  viewer: 'Viewer'


def _entry_overlay_geoms(entry: ModelEntry) -> list[mujoco.MjvGeom]:
  """Extracts display geoms from a registry entry for main-scene overlay.

  Runs mjv_updateScene on the entry's model/data into a per-entry scratch
  scene and returns views of its geoms, with the entry's tint applied. The
  views alias the scratch scene, which stays valid until the entry's next
  extraction — consume them within the frame.
  """
  needed = entry.model.ngeom + 64
  if entry._scene is None or entry._scene.maxgeom < needed:  # pylint: disable=protected-access
    entry._scene = mujoco.MjvScene(entry.model, maxgeom=needed)  # pylint: disable=protected-access
  scene = entry._scene  # pylint: disable=protected-access

  vopt = mujoco.MjvOption()
  camera = mujoco.MjvCamera()
  catmask = (
      mujoco.mjtCatBit.mjCAT_ALL
      if entry.include_static
      else mujoco.mjtCatBit.mjCAT_DYNAMIC
  )
  mujoco.mjv_updateScene(
      entry.model, entry.data, vopt, None, camera, int(catmask), scene
  )

  out = []
  for i in range(scene.ngeom):
    geom = scene.geoms[i]
    if geom.objtype != int(mujoco.mjtObj.mjOBJ_GEOM):
      continue  # Skip decor elements (frames, labels, skybox).
    if entry.tint is not None:
      geom.rgba[:] = entry.tint
    out.append(geom)
  return out


class Viewer(abc.ABC):
  """Base class for any viewer.

  Owns the communication endpoint, plugin registry and core visualization
  objects. The application is rendered by calling ``sync()``.
  """

  # Whether this viewer can confine the 3D scene to a window sub-rectangle
  # (set per concrete class). When True, the GUI may set ``scene_viewport``
  # and input handling maps mouse coordinates into that rectangle.
  supports_scene_viewport = False

  def __init__(
      self,
      config: ViewerConfig,
      endpoint: endpoints.ViewerEndpoint,
      *,
      model: mujoco.MjModel | None = None,
      model_path: str = '',
      plugins: list[Any] | None = None,
      camera: mujoco.MjvCamera | None = None,
      vis_options: mujoco.MjvOption | None = None,
      perturb: mujoco.MjvPerturb | None = None,
      render_flags: ux.RenderFlags | None = None,
      extra_geoms: list[mujoco.MjvGeom] | None = None,
  ) -> None:
    """Initializes the Viewer.

    Args:
      config: Viewer window configuration.
      endpoint: The viewer endpoint for communication with the sim side.
      model: Optional initial MjModel. If None, an empty model is created from
        an empty MjSpec. The Viewer deep-copies this model and creates its own
        MjData.
      model_path: Optional path to the model file.
      plugins: Optional list of plugin instances for viewer-side processing.
      camera: Camera parameters. Internal object is created if None.
      vis_options: Visualization options. Internal object is created if None.
      perturb: Perturbation parameters. Internal object is created if None.
      render_flags: Render flags. Internal object is created if None.
      extra_geoms: List of extra geoms. Internal list is created if None.
    """
    self.config = config
    self._endpoint = endpoint
    self._is_running = True
    self._closed = False

    # Viewer-owned model registry. The DEFAULT_MODEL_ID entry is the
    # simulated model; further entries are viewer-local display models.
    self.models: dict[str, ModelEntry] = {}
    if model is None:
      model = mujoco.MjSpec().compile()
    self.model_path: str = ''
    self.load_model(model, model_path)

    # Scene viewport (x, y, w, h in logical px, top-left origin), set by the
    # GUI each frame from the docking layout; None renders the scene across
    # the full window.
    self.scene_viewport: tuple[float, float, float, float] | None = None

    # Visual state.
    self.camera = camera or mujoco.MjvCamera()
    self.cam_speed = 0.001
    self.perturb = perturb or mujoco.MjvPerturb()
    self.vis_options = vis_options or mujoco.MjvOption()
    self.extra_geoms = extra_geoms or []
    if render_flags is not None:
      self.render_flags = render_flags
    else:
      self.render_flags = ux.RenderFlags()
      # Initted to match mujoco/src/engine/engine_vis_init.c
      self.render_flags.flags = [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1]

    # Plugin infrastructure.
    all_plugins = [self] + list(plugins or [])
    self.plugins = plugin_registry.PluginRegistry(all_plugins)

  def close(self) -> None:
    """Closes the viewer, sends an exit event and shuts down the endpoint."""
    if self._closed:
      return
    self._closed = True
    self._is_running = False
    try:
      self.send_to_sim(messages.ExitEvent())
    except Exception:  # pylint: disable=broad-exception-caught
      pass  # Ignore exceptions, the sim may have already closed.
    self._endpoint.close()

  @messages.handler(priority=messages.Priority.CRITICAL)
  def _on_exit(self, _: messages.ExitEvent) -> bool:
    """Stops the viewer loop when the sim side requests an exit."""
    self._is_running = False
    return False  # Do not consume; app handlers may want cleanup too.

  def is_running(self) -> bool:
    """Returns True while the viewer has not been closed."""
    return self._is_running

  def send_to_sim(self, message: messages.Message) -> None:
    """Sends a message to the simulation process."""
    self._endpoint.send_to_sim(message)

  def get_sim_events(self) -> list[messages.Event]:
    """Returns all pending events from the simulation."""
    return self._endpoint.get_sim_events()

  def get_sim_snapshots(self) -> list[messages.Snapshot]:
    """Returns all pending latest snapshots from the simulation, one per type."""
    return self._endpoint.get_sim_snapshots()

  def dispatch(self, message: messages.Message) -> None:
    """Dispatches a message to registered handlers in priority order."""
    self.plugins.dispatch(message)

  @property
  def model(self) -> mujoco.MjModel:
    """The simulated model (the DEFAULT_MODEL_ID registry entry)."""
    return self.models[DEFAULT_MODEL_ID].model

  @model.setter
  def model(self, value: mujoco.MjModel) -> None:
    self.models[DEFAULT_MODEL_ID].model = value

  @property
  def data(self) -> mujoco.MjData:
    """The simulated model's data (the DEFAULT_MODEL_ID registry entry)."""
    return self.models[DEFAULT_MODEL_ID].data

  @data.setter
  def data(self, value: mujoco.MjData) -> None:
    self.models[DEFAULT_MODEL_ID].data = value

  def add_model(
      self,
      name: str,
      model: mujoco.MjModel,
      data: mujoco.MjData | None = None,
      *,
      tint: tuple[float, float, float, float] | None = None,
      overlay: bool = True,
      include_static: bool = False,
  ) -> ModelEntry:
    """Registers a viewer-local display model and returns its entry.

    The caller owns the entry's lifecycle: pose ``entry.data`` (e.g. set qpos
    and call ``mj_forward``) and remove the entry when done. The model is not
    copied.

    Args:
      name: Registry id; replaces any existing entry with the same name.
      model: The model to display.
      data: Optional data; a fresh forwarded MjData is created if None.
      tint: Optional RGBA override for all overlay geoms.
      overlay: Whether to draw the entry into the main 3D scene.
      include_static: Whether overlay drawing includes static-body geoms.

    Returns:
      The registered ModelEntry.
    """
    if name == DEFAULT_MODEL_ID:
      raise ValueError(f'{DEFAULT_MODEL_ID!r} is reserved for the sim model')
    if data is None:
      data = mujoco.MjData(model)
      mujoco.mj_forward(model, data)
    entry = ModelEntry(
        model=model,
        data=data,
        source='local',
        overlay=overlay,
        tint=tint,
        include_static=include_static,
    )
    self.models[name] = entry
    return entry

  def remove_model(self, name: str) -> None:
    """Removes a viewer-local display model; missing names are ignored."""
    if name == DEFAULT_MODEL_ID:
      raise ValueError(f'cannot remove the {DEFAULT_MODEL_ID!r} entry')
    self.models.pop(name, None)

  def apply_state(self, model_id: str, state: Any, state_sig: int) -> bool:
    """Applies a state vector to a registry entry (the chokepoint for all
    incoming state, today always addressed to DEFAULT_MODEL_ID).

    Args:
      model_id: Registry id of the target entry.
      state: The state vector (as in ``mj_getState``).
      state_sig: The mjtState signature of the vector.

    Returns:
      True if the state was applied.
    """
    entry = self.models.get(model_id)
    if entry is None:
      return False
    state_size = mujoco.mj_stateSize(entry.model, state_sig)
    if len(state) != state_size:
      return False
    mujoco.mj_setState(entry.model, entry.data, state, state_sig)
    mujoco.mj_forward(entry.model, entry.data)
    return True

  def display_geoms(self) -> list[mujoco.MjvGeom]:
    """Returns extra_geoms plus the overlay geoms of local registry entries.

    This is what viewers draw on top of the simulated model: the user's
    ``extra_geoms`` list, then every visible local entry with ``overlay``
    set, converted to display geoms (with the entry's tint applied).
    """
    out = list(self.extra_geoms)
    for name, entry in self.models.items():
      if name == DEFAULT_MODEL_ID or not entry.overlay or not entry.visible:
        continue
      out.extend(_entry_overlay_geoms(entry))
    return out

  def load_model(self, model: mujoco.MjModel, model_path: str = '') -> None:
    """Deep-copies a model and creates fresh data for the viewer."""
    self.model_path = model_path
    model = copy.deepcopy(model)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    self.models[DEFAULT_MODEL_ID] = ModelEntry(
        model=model, data=data, source='sim'
    )

  @messages.handler(priority=messages.Priority.CRITICAL)
  def _on_model(self, event: messages.ModelEvent) -> bool:
    """Deep-copies the incoming model so the Viewer owns its data."""
    self.load_model(event.model, event.path)
    self.extra_geoms.clear()
    return False  # Do not consume; let other handlers see the event.

  @messages.handler(priority=messages.Priority.CRITICAL)
  def _on_state(self, event: messages.StateSnapshot) -> bool:
    """Applies incoming simulation state to the addressed registry entry."""
    model_id = getattr(event, 'model_id', DEFAULT_MODEL_ID)
    self.apply_state(model_id, event.state, event.state_sig)
    return False  # Do not consume; let other handlers see the event.

  @abc.abstractmethod
  def prepare_next_frame(self) -> bool:
    """Advances to the next frame; returns whether one is ready to render."""
    ...

  @abc.abstractmethod
  def sync(self) -> None:
    """Renders the scene using the viewer's current model and data."""
    ...

  @abc.abstractmethod
  def get_drop_file(self) -> str:
    ...

  @abc.abstractmethod
  def upload_image(
      self, tex_id: int, img: str | bytes, width: int, height: int, bpp: int
  ) -> int:
    ...


# -----------------------------------------------------------------------------
# Standalone viewer loop.
# -----------------------------------------------------------------------------


def run_viewer_loop(viewer: Viewer) -> None:
  """Minimal viewer loop: process sim messages, dispatch lifecycle events, sync.

  Runs until the viewer window is closed or an exit event is received.
  On exit, closes the viewer (which sends an ExitEvent to the sim side).

  Args:
    viewer: A Viewer that owns the endpoint and plugin registry.
  """
  while True:
    # Get the next frame; this also gets the frame's mouse/keyboard events.
    frame = viewer.prepare_next_frame()

    # Process incoming simulation events.
    for event in viewer.get_sim_events():
      viewer.dispatch(event)

    # Stop the loop if the viewer is not running (endpoint will be closed).
    if not viewer.is_running():
      break

    # Process incoming simulation snapshots.
    for snapshot in viewer.get_sim_snapshots():
      viewer.dispatch(snapshot)

    # Skip rendering when prepare_next_frame returned no active frame.
    # e.g., no browser is connected to the web viewer.
    if frame:
      # Dispatch lifecycle events.
      viewer.dispatch(messages.UpdateEvent())
      viewer.dispatch(messages.BuildGuiEvent())

      # Render the scene.
      viewer.sync()

  viewer.close()
