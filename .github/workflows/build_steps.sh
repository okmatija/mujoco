#!/bin/bash
# Copyright 2025 DeepMind Technologies Limited
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

# TODO(matijak): Make all cmake commands run from the top-level directory, and
# consider making the builds parallel.


# Wrap the compiler with ccache when it is available (set up by ccache-action in
# CI). This makes warm rebuilds - including the expensive, pinned Filament build -
# much faster. ccache is content-addressed on the full compiler invocation, so
# changing a flag or source forces a recompile: a stale object is never reused.
# Guarded by `command -v` so the script still works locally without ccache.
CCACHE_ARGS=""
if command -v ccache >/dev/null 2>&1; then
    CCACHE_ARGS="-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache"
fi


# Emit the build matrix for build.yml as a step output. On pull_request we run
# only the representative "core" compiler set; on push (e.g. to main) we run the
# full compiler sweep. Tiers are defined in build_matrix.json.
generate_matrix() {
    echo "Generating build matrix for event '${GITHUB_EVENT_NAME}'..."
    local file=".github/workflows/build_matrix.json"
    local matrix
    if [[ "${GITHUB_EVENT_NAME}" == "pull_request" ]]; then
        matrix="$(jq -c '{include: [.include[] | select(.tier == "core") | del(.tier)]}' "${file}")"
    else
        matrix="$(jq -c '{include: [.include[] | del(.tier)]}' "${file}")"
    fi
    echo "matrix=${matrix}" >> "${GITHUB_OUTPUT}"
    echo "${matrix}" | jq .
}


prepare_linux() {
    echo "Preparing Linux..."
    sudo apt-get update && sudo apt-get install \
        libgl1-mesa-dev \
        libwayland-dev \
        libxinerama-dev \
        libxcursor-dev \
        libxkbcommon-dev \
        libxrandr-dev \
        libxi-dev \
        ninja-build
}


prepare_python() {
    echo "Preparing Python..."
    repo="${PWD}"
    pushd "${TMPDIR}" > /dev/null
    python -m venv venv
    if [[ $RUNNER_OS == "Windows" ]]; then
    mkdir venv/bin
    fixpath="$(s="$(cat venv/Scripts/activate | grep VIRTUAL_ENV=)"; echo "${s:13:-1}")"
    sed -i "s#$(printf "%q" "${fixpath}")#$(cygpath "${fixpath}")#g" venv/Scripts/activate
    ln -s ../Scripts/activate venv/bin/activate
    fi
    source venv/bin/activate
    # Install build deps with uv when available (set up by setup-uv in CI on
    # POSIX) - much faster than pip. Fall back to pip otherwise (e.g. Windows,
    # local dev). The venv is still created by `python -m venv`, so pip stays
    # available for later steps (pip wheel / python -m build).
    if command -v uv > /dev/null 2>&1; then
        uv pip install --require-hashes -r "${repo}/python/build_requirements.txt"
        uv pip install --require-hashes -r "${repo}/python/build_requirements_usd.txt"
    else
        python -m pip install --upgrade --require-hashes -r "${repo}/python/build_requirements.txt"
        python -m pip install --upgrade --require-hashes -r "${repo}/python/build_requirements_usd.txt"
    fi
    popd > /dev/null
}


npm_ci() {
    echo "Installing NPM dependencies for WASM bindings..."
    pushd wasm
    npm ci
    popd
}


setup_emsdk() {
    echo "Setting up Emscripten..."
    git clone https://github.com/emscripten-core/emsdk.git
    ./emsdk/emsdk install 4.0.10
    ./emsdk/emsdk activate 4.0.10
    # Force installing emscripten's typescript dependencies. This is a
    # workaround for the github update to a newer typescript, which gives an
    # error on the deprecated `--outFile` flag.
    pushd emsdk/upstream/emscripten
    npm i
    popd
}


configure_mujoco() {
    echo "Configuring MuJoCo..."
    # Disable IPO/LTO to cut build time. Skip this on Windows: turning off MSVC's
    # whole-program optimization (/GL) exposes a latent heap corruption in
    # SetConstTest.SleepingNotAllowed (a real bug worth a separate investigation),
    # and Windows build time is not a CI bottleneck.
    local ipo_off="-DCMAKE_INTERPROCEDURAL_OPTIMIZATION:BOOL=OFF"
    if [[ "${RUNNER_OS}" == "Windows" ]]; then
        ipo_off=""
    fi
    mkdir build &&
    cd build &&
    cmake .. \
        -DCMAKE_BUILD_TYPE:STRING=Release \
        ${ipo_off} \
        -DCMAKE_INSTALL_PREFIX:STRING=${TMPDIR}/mujoco_install \
        -DMUJOCO_BUILD_EXAMPLES:BOOL=OFF \
        ${CCACHE_ARGS} \
        ${CMAKE_ARGS}
}


build_mujoco() {
    echo "Building MuJoCo..."
    cmake --build . --config=Release ${CMAKE_BUILD_ARGS}
}


test_mujoco() {
    echo "Testing MuJoCo..."
    # ctest defaults to serial. The suite is ~1650 independent tests that use
    # unique temp files (mkstemp / testing::TempDir) and declare no RUN_SERIAL /
    # RESOURCE_LOCK, so running them in parallel is safe and ~2x faster on POSIX.
    # Windows is kept serial conservatively: parallel-safety on the Windows file
    # system is unverified and its test time is not a CI bottleneck.
    if [[ "${RUNNER_OS}" == "Windows" ]]; then
        ctest -C Release --output-on-failure .
    else
        local ncpu
        ncpu="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo "${NUMBER_OF_PROCESSORS:-2}")"
        ctest -C Release --output-on-failure --parallel "${ncpu}" .
    fi
}


install_mujoco() {
    echo "Installing MuJoCo..."
    cmake --install .
}


copy_plugins_posix() {
    echo "Copying plugins..."
    mkdir -p ${TMPDIR}/mujoco_install/mujoco_plugin &&
    cp lib/libactuator.* ${TMPDIR}/mujoco_install/mujoco_plugin &&
    cp lib/libelasticity.* ${TMPDIR}/mujoco_install/mujoco_plugin &&
    cp lib/libsensor.* ${TMPDIR}/mujoco_install/mujoco_plugin &&
    cp lib/libsdf_plugin.* ${TMPDIR}/mujoco_install/mujoco_plugin
}


copy_plugins_window() {
    echo "Copying plugins..."
    mkdir -p ${TMPDIR}/mujoco_install/mujoco_plugin &&
    cp bin/Release/actuator.dll ${TMPDIR}/mujoco_install/mujoco_plugin &&
    cp bin/Release/elasticity.dll ${TMPDIR}/mujoco_install/mujoco_plugin &&
    cp bin/Release/sensor.dll ${TMPDIR}/mujoco_install/mujoco_plugin
}


configure_samples() {
    echo "Configuring samples..."
    # Samples are tiny, so they keep the default IPO/LTO: disabling it saves no
    # meaningful build time and would expose the same gcc -Werror false positives
    # that -O3-without-LTO triggers (see configure_mujoco).
    mkdir build &&
    cd build &&
    cmake .. \
        -DCMAKE_BUILD_TYPE:STRING=Release \
        -Dmujoco_ROOT:STRING=${TMPDIR}/mujoco_install \
        ${CCACHE_ARGS} \
        ${CMAKE_ARGS}
}


configure_simulate() {
    echo "Configuring simulate..."
    # See configure_samples: keep the default IPO/LTO for this small build.
    mkdir build &&
    cd build &&
    cmake .. \
        -DCMAKE_BUILD_TYPE:STRING=Release \
        -Dmujoco_ROOT:STRING=${TMPDIR}/mujoco_install \
        ${CCACHE_ARGS} \
        ${CMAKE_ARGS}
}


build_simulate() {
    echo "Building simulate..."
    cmake --build . --config=Release ${CMAKE_BUILD_ARGS}
}


configure_studio() {
    echo "Configuring Studio..."
    cmake -B build \
        -DCMAKE_BUILD_TYPE:STRING=Release \
        -DCMAKE_INTERPROCEDURAL_OPTIMIZATION:BOOL=OFF \
        -DUSE_STATIC_LIBCXX=OFF \
        -DBUILD_SHARED_LIBS=OFF \
        -DMUJOCO_BUILD_EXAMPLES=OFF \
        -DMUJOCO_BUILD_SIMULATE=OFF \
        -DMUJOCO_BUILD_STUDIO=ON \
        -DMUJOCO_BUILD_TESTS=OFF \
        -DMUJOCO_TEST_PYTHON_UTIL=OFF \
        -DMUJOCO_WITH_USD=OFF \
        -DMUJOCO_USE_FILAMENT=ON \
        ${CCACHE_ARGS} \
        ${CMAKE_ARGS}
    echo "Configuring Studio... DONE"
}


build_studio() {
    echo "Building Studio..."
    cmake --build build --config=Release --target mujoco_studio --parallel
    echo "Building Studio... DONE"
}


make_python_sdist() {
    echo "Making Python sdist..."
    source ${TMPDIR}/venv/bin/activate &&
    ./make_sdist.sh
}


build_python_bindings() {
    echo "Building Python bindings..."
    source ${TMPDIR}/venv/bin/activate
    # pip unpacks the sdist into a randomized temp dir every run, so the absolute
    # source/include paths differ each time and defeat ccache (0% hit, full
    # recompile). CCACHE_BASEDIR rewrites absolute paths under it to paths relative
    # to the (also-in-temp) build cwd, cancelling the random component so objects
    # hash identically across runs. CCACHE_SLOPPINESS ignores timestamp/path noise.
    #
    # Do NOT add system_headers here: CMake adds the imported mujoco target's
    # include dir (MUJOCO_PATH/include) as -isystem, so ccache would treat the
    # public MuJoCo headers as system headers and skip hashing them. A change that
    # lives only in those headers (a new mjData field, a new enum value) would then
    # go undetected and ccache would reuse an object compiled against the old struct
    # layout, producing an ABI-mismatched binding (wrong field offsets, stale
    # mjNENABLE, signature mismatch). The mtime/ctime flags are kept: they handle the
    # temp-dir churn without affecting header content detection.
    export CCACHE_BASEDIR="${TMPDIR}"
    export CCACHE_SLOPPINESS="time_macros,include_file_mtime,include_file_ctime,pch_defines,locale"
    MUJOCO_PATH="${TMPDIR}/mujoco_install" \
    MUJOCO_PLUGIN_PATH="${TMPDIR}/mujoco_install/mujoco_plugin" \
    MUJOCO_CMAKE_ARGS="-DCMAKE_INTERPROCEDURAL_OPTIMIZATION:BOOL=OFF ${CCACHE_ARGS} ${CMAKE_ARGS}" \
    pip wheel -v --no-deps mujoco-*.tar.gz
}


install_python_bindings() {
    echo "Installing Python bindings..."
    source ${TMPDIR}/venv/bin/activate &&
    pip install --no-index mujoco-*.whl
}


test_python_bindings() {
    echo "Testing Python bindings..."
    source ${TMPDIR}/venv/bin/activate &&
    pytest -v --pyargs mujoco
}


build_test_wasm() {
    echo "Building and testing WASM bindings..."
    source emsdk/emsdk_env.sh
    export PATH="$(pwd)/node_modules/.bin:$PATH"
    echo "Build MuJoCo with Emscripten (Multi-Threaded)..."
    emcmake cmake -B build_wasm_mt \
        -DCMAKE_INTERPROCEDURAL_OPTIMIZATION:BOOL=OFF \
        -DMUJOCO_WASM_THREADS=ON \
        ${CCACHE_ARGS} \
        $WASM_CMAKE_ARGS
    cmake --build build_wasm_mt --parallel $(nproc)

    echo "Run bindings tests for Multi-Threaded version..."
    npm run test --prefix ./wasm

    echo "Moving Multi-Thread version under mt subfolder..."
    mkdir -p wasm/dist/mt
    mv wasm/dist/mujoco.* wasm/dist/mt/

    echo "Build MuJoCo with Emscripten (Single-Threaded)..."
    emcmake cmake -B build_wasm_st \
        -DCMAKE_INTERPROCEDURAL_OPTIMIZATION:BOOL=OFF \
        -DMUJOCO_WASM_THREADS=OFF \
        ${CCACHE_ARGS} \
        $WASM_CMAKE_ARGS
    cmake --build build_wasm_st --parallel $(nproc)

    echo "Run bindings tests for Single-Threaded version..."
    npm run test --prefix ./wasm
}

package_wasm() {
    echo "Publishing WASM bindings..."
    cp wasm/package.npm.json wasm/dist/package.json
    cp wasm/README.md wasm/dist/README.md
    VERSION="${VERSION:-${GITHUB_REF#refs/tags/}}"
    npm --prefix wasm/dist version "${VERSION}" --no-git-tag-version
    npm pack --dry-run ./wasm/dist
    npm publish ./wasm/dist --access public --provenance
}


package_mjx() {
    echo "Packaging MJX..."
    source ${TMPDIR}/venv/bin/activate &&
    python -m build .
}


install_mjx() {
    echo "Installing MJX..."
    source ${TMPDIR}/venv/bin/activate
    # The MJX requirements (jax, jaxlib, scipy, ...) are a big install; use uv when
    # available. Keep pip for the local --no-index wheel.
    if command -v uv > /dev/null 2>&1; then
        uv pip install --require-hashes -r requirements.txt
    else
        pip install --require-hashes -r requirements.txt
    fi
    pip install --no-index dist/mujoco_mjx-*.whl
}


test_mjx() {
    echo "Testing MJX..."
    source ${TMPDIR}/venv/bin/activate &&
    pytest -n auto -v -k 'not IntegrationTest' --pyargs mujoco.mjx
}


notify_team_chat() {
    CHATMSG="$(cat <<-'EOF' | python3
import json
import os
env = lambda x: os.getenv(x, '')
data = dict(
    result=env('JOB_URL'),
    job=env('CHATMSG_JOB_ID'),
    commit=env('GITHUB_SHA')[:6],
    name=env('CHATMSG_AUTHOR_NAME').replace('```', ''),
    email=env('CHATMSG_AUTHOR_EMAIL'),
    msg=env('CHATMSG_COMMIT_MESSAGE').replace('```', '')
)
text = '<{result}|*FAILURE*>: job `{job}` commit `{commit}`\n```Author: {name} <{email}>\n\n{msg}```'.format(**data)
print(json.dumps({'text' : text}))
EOF
)" &&

    curl "$GCHAT_API_URL&threadKey=$GITHUB_SHA&messageReplyOption=REPLY_MESSAGE_FALLBACK_TO_NEW_THREAD" \
    -X POST \
    -H "Content-Type: application/json" \
    --data-raw "${CHATMSG}"
}


build_mujoco_live() {
    echo "Setting up Emscripten SDK..."
    source emsdk/emsdk_env.sh

    echo "Building Filament tools, targeting host platform..."
    cmake -S . -B build_host -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DUSE_STATIC_LIBCXX=OFF \
        -DMUJOCO_BUILD_STUDIO=ON \
        -DMUJOCO_USE_FILAMENT=ON \
        -DMUJOCO_BUILD_TESTS=OFF \
        -DMUJOCO_BUILD_EXAMPLES=OFF \
        -DMUJOCO_BUILD_SIMULATE=OFF
    cmake --build build_host --target matc resgen cmgen mujoco_filament_assets -j$(nproc)

    echo "Building WASM app..."
    emcmake cmake -S . -B build_wasm -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DMUJOCO_BUILD_STUDIO=ON \
        -DMUJOCO_USE_FILAMENT=ON \
        -DMUJOCO_BUILD_TESTS_WASM=OFF \
        -DMUJOCO_NATIVE_BUILD_DIR=$(pwd)/build_host
    cmake --build build_wasm --target mujoco_studio -j$(nproc)
}


# -----------------------------------------------------------------------------
# Web viewer wheel.
#
# These four steps build a self-contained Python wheel in which the Studio web
# viewer works out of the box: the compiled server-side modules (headless_ui,
# native_viewer_cc, ux, sim, ...) plus the pre-built browser client bundled at
# experimental/studio/web/dist (which web_server.py serves). Run them, in order,
# from the repository top level:
#
#   build_web_viewer_host          # native MuJoCo + Studio + Filament -> build_host
#   build_web_viewer_wasm          # browser client (Emscripten) -> .../web/dist
#   install_mujoco_for_web_viewer  # headers + libs the wheel compiles against
#   build_web_viewer_wheel         # the wheel -> python/dist
#
# The browser client is platform-independent, so a release process can build it
# once (build_web_viewer_wasm) and bundle the same web/dist into each per-platform
# wheel; only the native host build + wheel compile are per-platform.
#
# Platform status: Linux and macOS build here; Windows is being brought up (the
# NetImgui Winsock backend has landed). The per-OS branches below cover Linux and
# macOS; Windows-specific paths are marked "verify on Windows".
#
# TODO(kokoro): these functions are exercised by the web_viewer* jobs in build.yml
# for development/CI only — those jobs build and smoke-check the wheel but do not
# release it. The release wheels are built by kokoro, so port these steps into
# kokoro's per-platform wheel build once the web viewer ships, so released wheels
# actually contain it.
#
# TODO(robotics-simulation): once the web viewer ships to users, fold these
# modules + the bundled browser client into the DEFAULT `mujoco` wheel
# (build_python_bindings) so `pip install mujoco` includes the web viewer with no
# extra steps. Kept as a separate build for now to avoid pulling Filament + an
# Emscripten pre-step into every release build.
# -----------------------------------------------------------------------------

build_web_viewer_host() {
    echo "Building native MuJoCo + Studio + Filament (host)..."
    # A full native build. It provides two things the later steps consume:
    #  1. the Filament host tools (matc/resgen/cmgen) + baked assets that the
    #     Emscripten client build needs (MUJOCO_NATIVE_BUILD_DIR=build_host);
    #  2. libmujoco.so, the platform/dependency static archives, the engine
    #     plugins and the fetched third-party sources under build_host/_deps,
    #     which install_mujoco_for_web_viewer gathers into the compile inputs.
    cmake -S . -B build_host -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INTERPROCEDURAL_OPTIMIZATION:BOOL=OFF \
        -DUSE_STATIC_LIBCXX=OFF \
        -DMUJOCO_BUILD_STUDIO=ON \
        -DMUJOCO_USE_FILAMENT=ON \
        -DMUJOCO_BUILD_TESTS=OFF \
        -DMUJOCO_BUILD_EXAMPLES=OFF \
        -DMUJOCO_BUILD_SIMULATE=OFF \
        ${CCACHE_ARGS} \
        ${CMAKE_ARGS}
    # getconf _NPROCESSORS_ONLN works on both Linux and macOS (nproc is
    # GNU-only); fall back to Windows' env var, then a constant.
    cmake --build build_host -j"$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo "${NUMBER_OF_PROCESSORS:-4}")"
}


build_web_viewer_wasm() {
    echo "Building web viewer browser client (WASM)..."
    source emsdk/emsdk_env.sh
    # The browser client. Reuses the Filament host tools from build_host. The
    # web/ CMakeLists POST_BUILD stages web_client.js/.wasm + index.html +
    # assets into python/mujoco/experimental/studio/web/dist, which web_server.py
    # serves and setup.py bundles into the wheel.
    emcmake cmake -S . -B build_wasm -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DMUJOCO_BUILD_STUDIO=ON \
        -DMUJOCO_USE_FILAMENT=ON \
        -DMUJOCO_BUILD_TESTS_WASM=OFF \
        -DMUJOCO_NATIVE_BUILD_DIR=$(pwd)/build_host \
        ${CCACHE_ARGS}
    cmake --build build_wasm --target web_client -j"$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo "${NUMBER_OF_PROCESSORS:-4}")"
}


install_mujoco_for_web_viewer() {
    echo "Gathering the headers + libraries the web viewer wheel compiles against..."
    # A plain `cmake --install` only provides the public headers + libmujoco.
    # The web viewer modules also need the platform/dependency static archives,
    # the internal source-tree headers (src/experimental, src/render), the
    # imgui/implot/SDL2/Filament headers, the engine plugins and the Studio
    # assets. Gather all of it into build/mujoco_install; setup.py reads this
    # directory via MUJOCO_PATH (see build_web_viewer_wheel). This is the former
    # web/tools/make_linux_sdk.sh, run from the top level.
    local build_dir prefix deps plugin_ext plugin_dir
    # Resolve build_host to an absolute real path. Use `cd && pwd -P` rather than
    # `readlink -f` (macOS's readlink has no -f); pwd -P resolves symlinks so
    # `find` still descends when build_host is a symlink (e.g. a local build
    # routed onto a faster filesystem).
    build_dir="$(cd build_host && pwd -P)"
    mkdir -p build/mujoco_install
    prefix="$(cd build/mujoco_install && pwd)"

    # Per-OS engine-plugin layout.
    case "$(uname -s)" in
        Darwin) plugin_ext="dylib"; plugin_dir="${build_dir}/lib" ;;
        # TODO(matijak): verify on Windows. MSVC emits plugin DLLs under
        # bin/<config> (see copy_plugins_window), not lib/.
        MINGW*|MSYS*|CYGWIN*) plugin_ext="dll"; plugin_dir="${build_dir}/bin" ;;
        *) plugin_ext="so"; plugin_dir="${build_dir}/lib" ;;
    esac

    # Portable copy helpers. rsync is not available in Windows Git Bash, so use
    # find+cp everywhere (one codepath on every OS).
    _copy_headers() {  # $1=src dir, $2=dst dir — only *.h/*.inl, keep subdirs
        local s="${1%/}"
        (cd "${s}" && find . \( -name '*.h' -o -name '*.inl' \) -print0 |
            while IFS= read -r -d '' f; do
                mkdir -p "$2/${f%/*}" && cp "${f}" "$2/${f}"
            done)
    }
    _copy_tree() {  # $1=src dir, $2=dst dir — the whole subtree
        local s="${1%/}"
        mkdir -p "$2" && cp -r "${s}/." "$2/"
    }

    # 1. Standard install: libmujoco + public headers + models.
    cmake --install "${build_dir}" --prefix "${prefix}"

    # 2. Engine plugins (setup.py packages them from MUJOCO_PLUGIN_PATH). find
    #    matches lib<name>.<ext> (posix/macOS) and <name>.dll (Windows), and any
    #    config subdir MSVC nests under bin/.
    mkdir -p "${prefix}/mujoco_plugin"
    for plugin in actuator elasticity sensor sdf_plugin; do
        find "${plugin_dir}" -name "*${plugin}.${plugin_ext}" \
            -exec cp {} "${prefix}/mujoco_plugin/" \; 2>/dev/null || true
    done

    # 3. Static archives: mujoco_platform + every dependency archive; the Python
    #    build looks each one up by name with find_library(). posix/macOS use .a;
    #    MSVC uses .lib.
    mkdir -p "${prefix}/lib"
    find "${build_dir}" \( -name "*.a" -o -name "*.lib" \) -exec cp -u {} "${prefix}/lib/" \;

    # 4. Source-tree headers for platform / filament-compat / render.
    _copy_headers src/experimental "${prefix}/include/mujoco/experimental"
    _copy_headers src/render "${prefix}/include/mujoco/render"

    # 5. Third-party headers.
    deps="${build_dir}/_deps"
    # Dear ImGui (flat at the include root, matching the internal SDK layout).
    cp "${deps}/dear_imgui-src/"im*.h "${prefix}/include/"
    mkdir -p "${prefix}/include/misc/cpp"
    cp "${deps}/dear_imgui-src/misc/cpp/imgui_stdlib.h" "${prefix}/include/misc/cpp/"
    mkdir -p "${prefix}/include/backends"
    cp "${deps}/dear_imgui-src/backends/"imgui_impl_{sdl2,opengl3}.h \
        "${prefix}/include/backends/" 2>/dev/null || true
    # ImPlot.
    cp "${deps}/implot-src/"implot*.h "${prefix}/include/"
    # SDL2.
    mkdir -p "${prefix}/include/SDL2"
    cp "${deps}/sdl2-src/include/"*.h "${prefix}/include/SDL2/"
    cp -f "${deps}/sdl2-build/include/"*.h "${prefix}/include/SDL2/" 2>/dev/null || true
    cp -f "${deps}/sdl2-build/include-config-"*/*.h "${prefix}/include/SDL2/" 2>/dev/null || true
    # Filament support libraries (math/, utils/, filament/, backend/, ...).
    for lib in math utils filament backend filabridge ibl; do
        [[ -d "${deps}/filament-src/libs/${lib}/include/" ]] &&
            _copy_tree "${deps}/filament-src/libs/${lib}/include" "${prefix}/include"
    done
    _copy_tree "${deps}/filament-src/filament/include" "${prefix}/include"
    _copy_tree "${deps}/filament-src/filament/backend/include" "${prefix}/include"

    # 6. Studio assets (fonts + Filament materials) for the wheel.
    mkdir -p "${prefix}/assets"
    if [[ -d "${build_dir}/bin/assets" ]]; then
        cp -r "${build_dir}/bin/assets/." "${prefix}/assets/"
    else
        echo "WARNING: ${build_dir}/bin/assets not found; Studio fonts will be missing." >&2
    fi

    echo "Gathered web viewer compile inputs at ${prefix}"
}


build_web_viewer_wheel() {
    echo "Building the self-contained web viewer wheel..."
    # In CI the venv lives under ${TMPDIR}; for local dev, activate your own
    # virtualenv before calling this.
    if [[ -n "${TMPDIR:-}" && -f "${TMPDIR}/venv/bin/activate" ]]; then
        source "${TMPDIR}/venv/bin/activate"
    fi
    # Build the sdist first: it carries web/dist (see MANIFEST.in) so the wheel
    # bundles the browser client.
    (cd python && ./make_sdist.sh)
    local prefix
    prefix="$(cd build/mujoco_install && pwd)"
    # See build_python_bindings for why CCACHE_BASEDIR/SLOPPINESS are set.
    export CCACHE_BASEDIR="${TMPDIR:-$(pwd)}"
    export CCACHE_SLOPPINESS="time_macros,include_file_mtime,include_file_ctime,pch_defines,locale"
    MUJOCO_PATH="${prefix}" \
    MUJOCO_PLUGIN_PATH="${prefix}/mujoco_plugin" \
    MUJOCO_CMAKE_ARGS="-DCMAKE_INTERPROCEDURAL_OPTIMIZATION:BOOL=OFF ${CCACHE_ARGS} ${CMAKE_ARGS}" \
    pip wheel -v --no-deps -w python/dist python/dist/mujoco-*.tar.gz
}


build_web_viewer() {
    # Convenience wrapper: build the whole self-contained web viewer wheel in
    # order. Assumes the toolchain is ready — a virtualenv (prepare_python) and
    # Emscripten (setup_emsdk). This is both the single CI build step and the
    # one-liner for local development.
    build_web_viewer_host
    build_web_viewer_wasm
    install_mujoco_for_web_viewer
    build_web_viewer_wheel
}


# Discover functions defined in this script by finding identifiers followed by
# "()" and capturing the identifier as a valid function name.
VALID_FUNCTIONS=()
while IFS= read -r func_name; do
  VALID_FUNCTIONS+=("$func_name")
done < <(grep -E '^[[:alnum:]_]+\(\)' "$0" | sed 's/().*$//')

# Exit with an error if the requested function is not found.
if [[ ! " ${VALID_FUNCTIONS[*]} " =~ " ${1} " ]]; then
    echo "Usage: $0 {$(IFS='|'; echo "${VALID_FUNCTIONS[*]}")}, got '$1'"
    exit 1
fi

# Set options to print the commands being run, and cause the script to exit with
# an error code if any command fails. Note we do this just before executing
# the requested function to avoid cluttering the output with the above command
# discovery code.
set -xe

# Execute the requested function.
"$1"
