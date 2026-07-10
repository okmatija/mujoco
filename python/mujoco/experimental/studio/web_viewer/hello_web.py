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
"""Minimal end-to-end test of the web viewer transport.

Streams a stepped MuJoCo simulation and a demo ImGui UI to the browser without
going through the Studio viewer/sim message-passing machinery. Useful to debug
the transport layers (NetImgui, proxy, state WebSocket) in isolation.

Usage:
  python -m mujoco.experimental.studio.web_viewer.hello_web [model.xml]

Then open http://localhost:8080 in a browser.
"""

import sys
import time

import mujoco
from mujoco.experimental.studio.web_viewer import state_server as state_server_module
from mujoco.experimental.studio.web_viewer import ui_server as ui_server_module
from mujoco.experimental.studio.web_viewer import web_server as web_server_module
from mujoco.experimental.studio.web_viewer import web_viewer as web_viewer_module
import numpy as np

from mujoco.experimental.dear_imgui import dear_imgui as imgui

_DEFAULT_MODEL_XML = """
<mujoco>
  <option timestep="0.005"/>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="2 2 0.1" rgba=".9 .9 .9 1"/>
    <body pos="0 0 1">
      <freejoint/>
      <geom type="box" size=".1 .1 .1" rgba=".9 .2 .2 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def main(argv: list[str]) -> None:
  if len(argv) > 1:
    model = mujoco.MjModel.from_xml_path(argv[1])
  else:
    model = mujoco.MjModel.from_xml_string(_DEFAULT_MODEL_XML)
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)

  camera = mujoco.MjvCamera()
  mujoco.mjv_defaultFreeCamera(model, camera)
  perturb = mujoco.MjvPerturb()
  vis_options = mujoco.MjvOption()
  render_flags = [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1]

  # Headless ImGui context streaming UI draw data via NetImgui.
  ui_server = ui_server_module.UiServer(
      'hello_web', 8888, web_viewer_module._find_assets_dir()
  )
  imgui.SetCurrentContext(ui_server.get_context())

  # State streaming server (latest-wins snapshots over WebSocket).
  sig = int(mujoco.mjtState.mjSTATE_INTEGRATION)
  state_size = mujoco.mj_stateSize(model, sig)
  max_payload = ui_server_module.UiServer.max_state_payload_size(
      state_size * np.float64().itemsize
  )
  state_server = state_server_module.StateServer(
      state_ws_port=8891, max_payload_size=max_payload
  )
  state_server.start()

  # HTTP server (page, WASM, /model.mjb) + NetImgui WS-to-TCP proxy.
  web_server = web_server_module.WebServer(
      http_port=8080,
      tcp_port=8888,
      ws_port=8890,
      mjb_data=web_viewer_module._serialize_model(model),
  )
  web_server.start()

  print('hello_web running at http://localhost:8080 (Ctrl+C to quit)')

  state = np.empty(state_size, np.float64)
  frame = 0
  try:
    while True:
      # Blocks until a browser is connected and NetImgui starts a frame.
      ui_server.new_frame()

      # Step physics in real time.
      step_start = data.time
      t0 = time.monotonic()
      while data.time - step_start < 1.0 / 60.0:
        mujoco.mj_step(model, data)

      # Build a small UI.
      imgui.Begin('hello_web')
      imgui.Text(f'Frame {frame}')
      imgui.Text(f'Sim time: {data.time:.2f}s')
      imgui.Text(f'Step wall time: {(time.monotonic() - t0) * 1e3:.2f}ms')
      if imgui.Button('Reset'):
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
      imgui.End()
      imgui.ShowDemoWindow()

      # Stream the physics + visualization state to the browser.
      mujoco.mj_getState(model, data, state, sig)
      payload = ui_server.serialize_state_payload(
          0, sig, state.tobytes(), camera, perturb, vis_options, model,
          render_flags, []
      )
      state_server.update_state(payload)

      # End the frame; NetImgui streams the UI draw data to the browser.
      ui_server.end_frame()
      frame += 1
  except KeyboardInterrupt:
    pass
  finally:
    web_server.stop()
    state_server.stop()


if __name__ == '__main__':
  main(sys.argv)
