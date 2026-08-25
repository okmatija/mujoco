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
"""How to load a model and render it to an image, with no viewer at all.

  python 01_render_image.py --model=humanoid.xml --output=out.png

This first tutorial involves no window, no GUI and no messaging -- it exists
to introduce the two lowest-level Studio building blocks that everything
else stands on:

  - parser:   loads and compiles a model file into an MjModel/MjData pair.
  - renderer: turns a model and data into pixels, headlessly.

Everything in later tutorials -- the interactive window, the messaging
between simulation and viewer, the GUI panels -- is layered on top of these
two pieces. If you only need pictures (dataset generation, thumbnails, CI
golden images), this is all of Studio you need.
"""

import os
import sys

from absl import app
from absl import flags
import mujoco
from mujoco.experimental.studio import parser
from mujoco.experimental.studio import renderer
from mujoco.experimental.studio import viewer_protocol
from PIL import Image

_MODEL = flags.DEFINE_string('model', '', 'Model file to load.')
_OUTPUT = flags.DEFINE_string('output', '', 'Output file to save.')

# Studio renders through a hardware abstraction layer with several graphics
# backends. The default (Filament over OpenGL) is right for almost everyone;
# the others matter on headless machines ('*_headless' modes render without
# a display server) or when debugging driver issues ('*_software').
_GFX = flags.DEFINE_enum(
    'gfx', None, viewer_protocol.GFX_MODES, 'Rendering graphics mode.'
)
_WIDTH = flags.DEFINE_integer('width', 320, 'Width of the output image.')
_HEIGHT = flags.DEFINE_integer('height', 240, 'Height of the output image.')
_STEPS = flags.DEFINE_integer('steps', 1, 'Number of steps before render.')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  if not _MODEL.value:
    raise ValueError('`model` flag is required.')
  if not _OUTPUT.value:
    raise ValueError('`output` flag is required.')

  # ---------------------------------------------------------------------------
  # Load. parser.parse() reads a model file, compiles it, and returns the
  # MjData; the compiled MjModel rides along as data.model. This is the same
  # loader the interactive viewer uses for drag-and-dropped files.
  # ---------------------------------------------------------------------------
  try:
    data = parser.parse(_MODEL.value)
    model = data.model
  except Exception as ex:  # pylint: disable=broad-except
    print(f'Error loading model from `{_MODEL.value}`: {ex}')
    sys.exit(-1)

  # Advance the physics a little so the scene is not at its initial pose.
  # Note this is plain MuJoCo -- Studio does not wrap the physics API.
  for _ in range(_STEPS.value):
    mujoco.mj_step(model, data)

  # ---------------------------------------------------------------------------
  # Render. The Renderer is initialized once per model (it uploads meshes and
  # textures to the GPU), then Render() can be called as often as you like.
  # The three None arguments are (perturb, camera, vis_options): passing None
  # uses defaults -- the model's free camera, no perturbation highlights,
  # default visualization flags. Later tutorials pass real objects here.
  #
  # Render() returns raw RGB bytes, width * height * 3 of them.
  # ---------------------------------------------------------------------------
  try:
    r = renderer.Renderer(_GFX.value or '')
    r.Init(model)
    pixels = r.Render(
        model, data, None, None, None, _WIDTH.value, _HEIGHT.value
    )
  except Exception as ex:  # pylint: disable=broad-except
    print(f'Error rendering model: {ex}')
    sys.exit(-2)

  # Encode with PIL; the format is inferred from the output file extension.
  try:
    img = Image.frombytes('RGB', (_WIDTH.value, _HEIGHT.value), pixels)
    img.save(_OUTPUT.value, format=os.path.splitext(_OUTPUT.value)[1][1:])
  except Exception as ex:  # pylint: disable=broad-except
    print(f'Error saving image to `{_OUTPUT.value}`: {ex}')
    sys.exit(-3)

  return 0


if __name__ == '__main__':
  app.run(main)
