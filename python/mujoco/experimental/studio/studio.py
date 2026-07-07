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
"""This script runs Studio from Python.

The native viewer is used by default, pass --web_viewer to stream the simulation
to the browser using the web viewer.
"""

import os
import sys

from absl import app as absl_app
from absl import flags as absl_flags
from mujoco.experimental.studio import native_viewer
from mujoco.experimental.studio import studio_app
from mujoco.experimental.studio import viewer_protocol

# copybara:strip_begin
from mujoco.google import assetdb_browser

from google3.third_party.mujoco.src.experimental.py import web_viewer

# copybara:strip_end

_GFX = absl_flags.DEFINE_enum(
    'gfx', None, viewer_protocol.GFX_MODES, 'Rendering graphics mode.'
)
_WIDTH = absl_flags.DEFINE_integer('width', 1200, 'Width of the output image.')
_HEIGHT = absl_flags.DEFINE_integer('height', 800, 'Height of the output image')
_WEB_VIEWER = absl_flags.DEFINE_bool('web_viewer', False, 'Use web viewer.')


def main(argv: list[str]) -> None:
  title = os.path.basename(sys.argv[0])

  app = studio_app.StudioApp.from_argv(argv)
  # copybara:strip_begin
  browser = assetdb_browser.AssetBrowser(argv[1] if len(argv) > 1 else '')
  # copybara:strip_end

  # Initialize the viewer.
  if _WEB_VIEWER.value:
    viewer = web_viewer.WebViewer(
        app.model,
        title=title,
    )
  else:
    viewer = native_viewer.NativeViewer(
        app.model,
        title=title,
        width=_WIDTH.value,
        height=_HEIGHT.value,
        gfx=_GFX.value,
    )

  # Main viewer loop.
  while viewer.is_running():
    if not app.update_from_viewer(viewer):
      break

    app.build_gui(viewer.camera, viewer.vis_options, viewer.render_flags)

    # copybara:strip_begin
    browser.build_gui(app, viewer.camera, viewer.vis_options)
    browser.update(app, viewer)
    # copybara:strip_end

    viewer.sync(app.model, app.data)

  viewer.stop()


if __name__ == '__main__':
  absl_app.run(main)
