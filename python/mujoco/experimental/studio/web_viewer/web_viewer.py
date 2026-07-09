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
"""Simulation-agnostic web viewer for MuJoCo models.

The WebViewer streams the Studio UI and simulation state to a browser:

  * The Studio ImGui UI is built into a headless ImGui context (``ui_server``
    pybind module) and streamed to the browser with the NetImgui protocol
    through a WebSocket-to-TCP proxy. Input captured in the browser flows back
    over the same connection and is injected into the headless context, so all
    viewer-side handlers (e.g. ``ViewerApp``) work unmodified.
  * Physics state and visualization state (camera, options, render flags) are
    streamed to the browser at ~60Hz over a WebSocket with latest-wins
    (snapshot) semantics — see ``state_server.StateServer``.
  * The browser runs the ``web_client`` WASM app, which renders the MuJoCo
    scene with Filament and overlays the remote ImGui draw data.

See the documentation for studio_app.py for more details on the architecture
separating the viewer and simulation.
"""

import os
from typing import Any

import mujoco
from mujoco.experimental.studio import endpoints
from mujoco.experimental.studio import messages
from mujoco.experimental.studio import ux
from mujoco.experimental.studio import viewer_protocol
from mujoco.experimental.studio.web_viewer import state_server as state_server_module
from mujoco.experimental.studio.web_viewer import ui_server as ui_server_module
from mujoco.experimental.studio.web_viewer import web_server as web_server_module
import numpy as np

from mujoco.experimental.dear_imgui import dear_imgui as imgui


def _find_assets_dir() -> str:
  """Locates the Studio assets directory (fonts) inside the mujoco package."""
  env_dir = os.environ.get('MUJOCO_STUDIO_ASSETS_DIR')
  if env_dir and os.path.isdir(env_dir):
    return env_dir
  package_dir = os.path.dirname(os.path.abspath(mujoco.__file__))
  assets_dir = os.path.join(package_dir, 'experimental', 'studio', 'assets')
  if os.path.isdir(assets_dir):
    return assets_dir
  return ''


def _serialize_model(model: mujoco.MjModel) -> bytes:
  """Serializes a compiled model to MJB bytes (served as /model.mjb)."""
  buffer = np.empty(mujoco.mj_sizeModel(model), np.uint8)
  mujoco.mj_saveModel(model, None, buffer)
  return buffer.tobytes()


class WebViewer(viewer_protocol.Viewer):
  """Simulation-agnostic web viewer for MuJoCo models."""

  def __init__(
      self,
      config: viewer_protocol.ViewerConfig,
      endpoint: endpoints.ViewerEndpoint,
      *,
      model: mujoco.MjModel | None = None,
      model_path: str = '',
      handlers: list[Any] | None = None,
      camera: mujoco.MjvCamera | None = None,
      vis_options: mujoco.MjvOption | None = None,
      perturb: mujoco.MjvPerturb | None = None,
      render_flags: ux.RenderFlags | None = None,
      extra_geoms: list[mujoco.MjvGeom] | None = None,
      host: str = '0.0.0.0',
      http_port: int = 8080,
      ui_tcp_port: int = 8888,
      ui_ws_port: int = 8890,
      state_ws_port: int = 8891,
  ) -> None:
    """Initializes the WebViewer.

    Args:
      config: Viewer window configuration.
      endpoint: The viewer endpoint for communication with the sim side.
      model: Optional initial MjModel. Forwarded to the base Viewer.
      model_path: Optional path to the model file.
      handlers: Optional list of handler instances.
      camera: Camera parameters. Internal object is created if None.
      vis_options: Visualization options. Internal object is created if None.
      perturb: Perturbation parameters. Internal object is created if None.
      render_flags: Render flags. Internal object is created if None.
      extra_geoms: List of extra geoms. Internal list is created if None.
      host: Interface the HTTP/WebSocket servers bind to.
      http_port: Port serving the browser page, WASM and /model.mjb.
      ui_tcp_port: TCP port the headless NetImgui client connects to.
      ui_ws_port: WebSocket port the browser NetImgui side connects to.
      state_ws_port: WebSocket port streaming simulation state to the browser.
    """
    super().__init__(
        config,
        endpoint,
        model=model,
        model_path=model_path,
        handlers=handlers,
        camera=camera,
        vis_options=vis_options,
        perturb=perturb,
        render_flags=render_flags,
        extra_geoms=extra_geoms,
    )

    self._host = host
    self._http_port = http_port
    self._ui_tcp_port = ui_tcp_port
    self._ui_ws_port = ui_ws_port
    self._state_ws_port = state_ws_port

    # Headless ImGui context streaming UI draw data via NetImgui.
    self._ui_server = ui_server_module.UiServer(
        config.title or 'MuJoCo Web Viewer',
        ui_tcp_port,
        _find_assets_dir(),
    )

    # Point the Python Dear ImGui bindings at the headless context so that
    # viewer-side handlers (ViewerApp etc.) build their GUI into it.
    ctx = self._ui_server.get_context()
    imgui.SetCurrentContext(ctx)
    ux.set_imgui_context(ctx)

    # HTTP + NetImgui proxy + state streaming servers. They are (re)started
    # whenever the model changes, since the served MJB bytes and the state
    # buffer size depend on the model.
    self._live_server: web_server_module.LiveServer | None = None
    self._state_server: state_server_module.StateServer | None = None
    self._start_servers()

    # Dispatch lifecycle event so handlers can cache the viewer reference.
    self.dispatch(viewer_protocol.ViewerInitEvent(viewer=self))

  # ---------------------------------------------------------------------------
  # Server lifecycle.
  # ---------------------------------------------------------------------------

  def _state_signature_and_size(self) -> tuple[int, int]:
    """Returns the physics state signature and its size in doubles."""
    sig = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    return sig, mujoco.mj_stateSize(self.model, sig)

  def _start_servers(self) -> None:
    """Starts (or restarts) the HTTP, proxy and state servers."""
    self._stop_servers()

    sig, state_size = self._state_signature_and_size()
    payload_bytes = (
        state_size * np.float64().itemsize
        + ui_server_module.UiServer.get_vis_state_size()
    )

    self._state_server = state_server_module.StateServer(
        host=self._host,
        state_ws_port=self._state_ws_port,
        state_size=payload_bytes,
        state_sig=sig,
    )
    self._state_server.start()

    self._live_server = web_server_module.LiveServer(
        host=self._host,
        http_port=self._http_port,
        tcp_port=self._ui_tcp_port,
        ws_port=self._ui_ws_port,
        mjb_data=_serialize_model(self.model),
    )
    self._live_server.start()
    print(
        'MuJoCo web viewer running at '
        f'http://localhost:{self._http_port} (Ctrl+C to quit)'
    )

  def _stop_servers(self) -> None:
    if self._live_server is not None:
      self._live_server.stop()
      self._live_server = None
    if self._state_server is not None:
      self._state_server.stop()
      self._state_server = None

  # ---------------------------------------------------------------------------
  # Message handlers.
  # ---------------------------------------------------------------------------

  @messages.handler(priority=messages.Priority.INTERNAL)
  def _on_model_web(self, event: messages.ModelEvent) -> bool:
    """Restarts the servers to serve the new model.

    The base Viewer's CRITICAL-priority handler has already replaced
    self.model/self.data by the time this runs. The browser does not reconnect
    automatically — reload the page after a model change.
    """
    del event
    self._start_servers()
    print('Model changed: reload the browser page to pick it up.')
    return False  # Do not consume; let other handlers see the event.

  # ---------------------------------------------------------------------------
  # Viewer interface.
  # ---------------------------------------------------------------------------

  def is_running(self) -> bool:
    """Starts a new headless frame; blocks until a browser is connected."""
    if super().is_running():
      # Pumps browser input (received via NetImgui) into the ImGui context and
      # paces the loop to the browser's desired frame rate.
      self._ui_server.new_frame()
    return super().is_running()

  def sync(self) -> None:
    """Streams state to the browser and ends the headless ImGui frame."""
    if self._state_server is not None:
      sig, state_size = self._state_signature_and_size()
      state = np.empty(state_size, np.float64)
      mujoco.mj_getState(self.model, self.data, state, sig)
      vis_bytes = self._ui_server.get_vis_state(
          self.camera,
          self.vis_options,
          self.model,
          list(self.render_flags.flags),
      )
      self._state_server.update_state(state.tobytes() + vis_bytes)

    # Finish the ImGui frame; NetImgui sends the draw data to the browser.
    self._ui_server.end_frame()

  def close(self) -> None:
    self._stop_servers()
    super().close()

  def get_drop_file(self) -> str:
    """File drop is not supported in the web viewer."""
    return ''

  def upload_image(
      self, tex_id: int, img: str | bytes, width: int, height: int, bpp: int
  ) -> int:
    """Image upload is not supported in the web viewer (no local renderer)."""
    del img, width, height, bpp
    return tex_id
