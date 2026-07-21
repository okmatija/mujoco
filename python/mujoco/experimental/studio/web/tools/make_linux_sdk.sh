#!/bin/bash
# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Assembles a MuJoCo SDK directory that the Python bindings build can consume
# to build the Studio extension modules (parser, native_viewer_cc, ux, sim,
# web/headless_ui, ...) on Linux.
#
# The Python build (python/mujoco/CMakeLists.txt) expects, under MUJOCO_PATH:
#   lib/libmujoco.so.*          — the main library (cmake --install)
#   lib/libmujoco_platform.a    — the platform static library
#   lib/lib<dep>.a              — static archives of platform dependencies
#   include/mujoco/**           — public headers (cmake --install)
#   include/mujoco/experimental/** and include/mujoco/render/** — source-tree
#       headers; the extra include root include/mujoco makes internal
#       "experimental/..." and "render/..." includes resolve
#   include/imgui.h, implot.h, SDL2/, math/, ... — third-party headers
#   assets/                     — Studio fonts + Filament assets (packaged into
#       the wheel by setup.py's _copy_studio_assets)
#
# Usage:
#   make_linux_sdk.sh <build_dir> <sdk_prefix>
#
# where <build_dir> is a build of the main MuJoCo project configured with
# -DMUJOCO_BUILD_STUDIO=ON -DMUJOCO_USE_FILAMENT=ON.

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <build_dir> <sdk_prefix>" >&2
    exit 1
fi

BUILD_DIR=$(readlink -f "$1")
PREFIX=$(readlink -f "$2")
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../../../../.." && pwd)

echo "Repo:   ${REPO_ROOT}"
echo "Build:  ${BUILD_DIR}"
echo "Prefix: ${PREFIX}"

# ------------------------------------------------------------------------------
# 1. Standard install: libmujoco.so + public headers + models.
# ------------------------------------------------------------------------------
cmake --install "${BUILD_DIR}" --prefix "${PREFIX}" > /dev/null

# ------------------------------------------------------------------------------
# 2. Engine plugins (copied next to the SDK; setup.py packages them).
# ------------------------------------------------------------------------------
mkdir -p "${PREFIX}/mujoco_plugin"
for plugin in actuator elasticity sensor sdf_plugin; do
    src="${BUILD_DIR}/lib/lib${plugin}.so"
    [[ -f "${src}" ]] && cp "${src}" "${PREFIX}/mujoco_plugin/"
done

# ------------------------------------------------------------------------------
# 3. Static archives: mujoco_platform + every dependency archive in the build
#    tree. The Python build looks each one up by name with find_library().
# ------------------------------------------------------------------------------
mkdir -p "${PREFIX}/lib"
find "${BUILD_DIR}" -name "*.a" -exec cp -u {} "${PREFIX}/lib/" \;

# ------------------------------------------------------------------------------
# 4. Source-tree headers for platform / filament-compat / render.
# ------------------------------------------------------------------------------
rsync -a --include='*/' --include='*.h' --include='*.inl' --exclude='*' \
    "${REPO_ROOT}/src/experimental/" "${PREFIX}/include/mujoco/experimental/"
rsync -a --include='*/' --include='*.h' --include='*.inl' --exclude='*' \
    "${REPO_ROOT}/src/render/" "${PREFIX}/include/mujoco/render/"

# ------------------------------------------------------------------------------
# 5. Third-party headers.
# ------------------------------------------------------------------------------
DEPS="${BUILD_DIR}/_deps"

# Dear ImGui (flat at the include root, matching the internal SDK layout).
cp "${DEPS}/dear_imgui-src/"im*.h "${PREFIX}/include/"
mkdir -p "${PREFIX}/include/misc/cpp"
cp "${DEPS}/dear_imgui-src/misc/cpp/imgui_stdlib.h" "${PREFIX}/include/misc/cpp/"
mkdir -p "${PREFIX}/include/backends"
cp "${DEPS}/dear_imgui-src/backends/"imgui_impl_{sdl2,opengl3}.h \
    "${PREFIX}/include/backends/" 2>/dev/null || true

# ImPlot.
cp "${DEPS}/implot-src/"implot*.h "${PREFIX}/include/"

# SDL2.
mkdir -p "${PREFIX}/include/SDL2"
cp "${DEPS}/sdl2-src/include/"*.h "${PREFIX}/include/SDL2/"
# SDL_config.h is generated into the build tree.
cp -f "${DEPS}/sdl2-build/include/"*.h "${PREFIX}/include/SDL2/" 2>/dev/null || true
cp -f "${DEPS}/sdl2-build/include-config-"*/*.h "${PREFIX}/include/SDL2/" 2>/dev/null || true

# Filament support libraries (math/, utils/, filament/, backend/, ...).
for lib in math utils filament backend filabridge ibl; do
    src="${DEPS}/filament-src/libs/${lib}/include/"
    [[ -d "${src}" ]] && rsync -a "${src}" "${PREFIX}/include/"
done
rsync -a "${DEPS}/filament-src/filament/include/" "${PREFIX}/include/"
rsync -a "${DEPS}/filament-src/filament/backend/include/" "${PREFIX}/include/"

# ------------------------------------------------------------------------------
# 6. Studio assets (fonts + Filament materials) for the wheel.
# ------------------------------------------------------------------------------
mkdir -p "${PREFIX}/assets"
if [[ -d "${BUILD_DIR}/bin/assets" ]]; then
    cp -r "${BUILD_DIR}/bin/assets/." "${PREFIX}/assets/"
else
    echo "WARNING: ${BUILD_DIR}/bin/assets not found; Studio fonts will be missing." >&2
fi

echo "SDK assembled at ${PREFIX}"
echo
echo "Build the Python bindings with:"
echo "  MUJOCO_PATH=${PREFIX} \\"
echo "  MUJOCO_PLUGIN_PATH=${PREFIX}/mujoco_plugin \\"
echo "  pip install -e ${REPO_ROOT}/python"
