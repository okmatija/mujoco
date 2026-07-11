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
  * Physics state and render state (camera, perturb, options, extra geoms) are
    streamed to the browser at ~60Hz over a WebSocket with latest-wins
    (snapshot) semantics — see ``web_server.WebServer``.
  * The browser runs the ``web_client`` WASM app, which renders the MuJoCo
    scene with Filament and overlays the remote ImGui draw data.

  Everything is served through ONE public port (default 8080): the page and
  model over HTTP, the UI stream at path /ui and the state stream at /state
  (see ``web_server._run_router``). Exposing or tunneling that single port
  exposes the whole viewer; the other ports are loopback-internal.

See the documentation for studio_app.py for more details on the architecture
separating the viewer and simulation.
"""

import os
from typing import Any
import zlib

import mujoco
from mujoco.experimental.studio import endpoints
from mujoco.experimental.studio import messages
from mujoco.experimental.studio import ux
from mujoco.experimental.studio import viewer_protocol
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
      host: Public interface the server binds to.
      http_port: The single public port: page, WASM, /model.mjb, and the
        /ui and /state WebSocket paths.
      ui_tcp_port: Loopback TCP port the headless NetImgui client connects to.
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

    # Headless ImGui context streaming UI draw data via NetImgui.
    self._ui_server = ui_server_module.UiServer(
        config.title or 'MuJoCo Web Viewer',
        ui_tcp_port,
        _find_assets_dir(),
    )

    # Point the Python Dear ImGui bindings at the headless context so that
    # viewer-side handlers (ViewerApp etc.) build their GUI into it. The
    # ImPlot context must be shared the same way — every extension module has
    # its own copy of the ImGui/ImPlot globals, and the plotting GUIs crash
    # on a null ImPlot context otherwise.
    ctx = self._ui_server.get_context()
    imgui.SetCurrentContext(ctx)
    ux.set_imgui_context(ctx)
    ux.set_implot_context(self._ui_server.get_implot_context())

    # The single-port server (HTTP + /ui + /state). It is restarted whenever
    # the model changes, since the served MJB bytes and the state payload
    # capacity depend on the model.
    self._web_server: web_server_module.WebServer | None = None
    self._model_ident = 0
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

    mjb_data = _serialize_model(self.model)
    # Identity of the served model, included in every state payload. When it
    # changes, the browser refetches /model.mjb by reloading the page.
    self._model_ident = zlib.crc32(mjb_data)

    _, state_size = self._state_signature_and_size()
    max_payload = ui_server_module.UiServer.max_state_payload_size(
        state_size * np.float64().itemsize
    )

    self._web_server = web_server_module.WebServer(
        host=self._host,
        http_port=self._http_port,
        tcp_port=self._ui_tcp_port,
        mjb_data=mjb_data,
        max_payload_size=max_payload,
    )
    self._web_server.start()
    print(
        'MuJoCo web viewer running at '
        f'http://localhost:{self._http_port} (Ctrl+C to quit)'
    )

  def _stop_servers(self) -> None:
    if self._web_server is not None:
      self._web_server.stop()
      self._web_server = None

  # ---------------------------------------------------------------------------
  # Message handlers.
  # ---------------------------------------------------------------------------

  @messages.handler(priority=messages.Priority.INTERNAL)
  def _on_model_web(self, event: messages.ModelEvent) -> bool:
    """Restarts the servers to serve the new model.

    The base Viewer's CRITICAL-priority handler has already replaced
    self.model/self.data by the time this runs. The browser reconnects to the
    new state server, notices the changed model identity in the payload, and
    reloads itself to fetch the new model.
    """
    del event
    self._start_servers()
    print('Model changed: the browser page reloads automatically.')
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
    if self._web_server is not None:
      sig, state_size = self._state_signature_and_size()
      state = np.empty(state_size, np.float64)
      mujoco.mj_getState(self.model, self.data, state, sig)
      payload = self._ui_server.serialize_state_payload(
          self._model_ident,
          sig,
          state.tobytes(),
          self.camera,
          self.perturb,
          self.vis_options,
          self.model,
          list(self.render_flags.flags),
          self.extra_geoms[: ui_server_module.MAX_EXTRA_GEOMS],
      )
      self._web_server.update_state(payload)

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
    """Uploads an image to the browser over the NetImgui texture channel."""
    if isinstance(img, str):
      img = img.encode('latin-1')
    return self._ui_server.upload_image(tex_id, img, width, height, bpp)
