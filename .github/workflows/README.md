# GitHub Actions & build scripts

This folder holds MuJoCo's continuous-integration workflows and the shared build
script they call, [`build_steps.sh`](build_steps.sh). It doubles as the entry
point for building MuJoCo from source for local development: **every CI step is a
`build_steps.sh` function you can run yourself**, so a green CI run is
reproducible on your machine by calling the same functions in order.

> [!NOTE]
> These are build/CI notes for people developing MuJoCo on GitHub. End-user
> documentation — installing the wheel, using the Python / JavaScript APIs — lives
> at <https://mujoco.readthedocs.io>, and the build instructions collected here
> are intended to move there over time. Treat this file as the interim home while
> the build story stabilises.

## What's in this folder

| File | Purpose |
|------|---------|
| [`build.yml`](build.yml) | Main CI. The compiler matrix plus the `studio`, `wasm`, `web_viewer` and `mjx` jobs. |
| [`build_steps.sh`](build_steps.sh) | Every build/test step, as a shell function. CI calls these; so can you. |
| [`build_matrix.json`](build_matrix.json) | The compiler/OS matrix `build.yml` expands (a "core" subset on PRs, the full sweep on push). |
| [`lint.yml`](lint.yml) | Python/C++ linting. |
| [`live.yml`](live.yml) / [`update_live.yml`](update_live.yml) | Build + deploy the hosted browser Studio demo to GitHub Pages. |
| [`publish-wasm.yml`](publish-wasm.yml) | Publish the `@mujoco/mujoco` npm package. |

## The `build_steps.sh` convention

> [!IMPORTANT]
> Run every command below from the repository **top-level directory**. Build
> trees land in top-level folders (`build`, `build_host`, `build_wasm`) and
> packaging output in `python/dist`.

Run one step with:

```sh
bash .github/workflows/build_steps.sh <function_name>
```

Each CI job in `build.yml` is just an ordered sequence of these calls. The
sections below group the functions into the artifact they build.

<details>
<summary><b>Build the core Python bindings (the <code>mujoco</code> wheel)</b></summary>

This is the standard `pip install mujoco` build — the physics engine and its
Python bindings, **without** Studio or the web viewer. CI runs it across the full
compiler matrix; the ordered steps are:

```sh
bash .github/workflows/build_steps.sh configure_mujoco       # -> build/
bash .github/workflows/build_steps.sh build_mujoco
bash .github/workflows/build_steps.sh install_mujoco         # -> ${TMPDIR}/mujoco_install
bash .github/workflows/build_steps.sh copy_plugins_posix
bash .github/workflows/build_steps.sh make_python_sdist      # -> python/dist/*.tar.gz
bash .github/workflows/build_steps.sh build_python_bindings  # -> python/dist/*.whl
bash .github/workflows/build_steps.sh install_python_bindings
bash .github/workflows/build_steps.sh test_python_bindings
```

`build_python_bindings` / `make_python_sdist` expect a virtualenv; CI creates one
under `${TMPDIR}/venv` via `prepare_python`. Locally, activate your own venv (with
`python/build_requirements.txt` installed) first.
</details>

<details>
<summary><b>Build MuJoCo Studio (native desktop app)</b></summary>

[MuJoCo Studio](../../src/experimental/studio) is the next iteration of the
`simulate` application — the UI reimplemented with [Dear ImGui](https://github.com/ocornut/imgui)
and Filament as the default renderer. It is a WIP.

CI compiles it (a build check, no packaging) with:

```sh
bash .github/workflows/build_steps.sh configure_studio       # -> build/
bash .github/workflows/build_steps.sh build_studio           # target: mujoco_studio
```

> [!NOTE]
> Studio does not yet work under Wayland on Linux — use X11 instead. Python
> integration and further UI/UX work are tracked as future work.
</details>

<details>
<summary><b>Build the WASM / JavaScript bindings (<code>@mujoco/mujoco</code>)</b></summary>

These are the browser bindings that compile the MuJoCo engine to WebAssembly and
expose it to JavaScript/TypeScript. CI builds and tests both threading models
(single- and multi-threaded) with:

```sh
bash .github/workflows/build_steps.sh npm_ci          # installs wasm/ node deps
bash .github/workflows/build_steps.sh setup_emsdk      # installs Emscripten 4.0.10 into ./emsdk
bash .github/workflows/build_steps.sh build_test_wasm  # -> wasm/dist (mujoco.js/.wasm/.d.ts)
```

To build by hand, first `source ./emsdk/emsdk_env.sh`, then:

```sh
emcmake cmake -B build && cmake --build build           # single-threaded
emcmake cmake -B build -DMUJOCO_WASM_THREADS=ON && cmake --build build   # multi-threaded
```

> [!TIP]
> The full JavaScript **API reference and user guide** (named access, memory
> management, out-parameters, threading headers, …) lives in
> [`wasm/README.md`](../../wasm/README.md), which is also the README shipped with
> the npm package.
</details>

<details>
<summary><b>Build the self-contained web viewer wheel</b> (Linux)</summary>

The Studio **web viewer** streams a running simulation to a browser. A
self-contained wheel bundles both halves so it works straight after install:

- the **server side** — compiled Python modules (`headless_ui`, `native_viewer_cc`,
  `ux`, `sim`, …), which are native and therefore **per-platform** (Linux-only for
  now);
- the **browser client** — `web_client.wasm`/`.js` + Filament/font assets, staged
  into `…/studio/web/dist/`. This half is **platform-independent** (it runs in the
  browser), so a release process can build it once and bundle the same `web/dist`
  into every per-platform wheel.

With an active virtualenv, Emscripten (`setup_emsdk`) and Ninja in place, one
command builds the wheel:

```sh
bash .github/workflows/build_steps.sh build_web_viewer   # -> python/dist/*.whl
```

`build_web_viewer` is a wrapper that runs these four steps in order — call them
individually when debugging a single stage:

```sh
bash .github/workflows/build_steps.sh build_web_viewer_host          # native MuJoCo+Studio+Filament -> build_host/
bash .github/workflows/build_steps.sh build_web_viewer_wasm          # browser client -> .../web/dist/
bash .github/workflows/build_steps.sh install_mujoco_for_web_viewer  # headers+libs the wheel compiles against -> build/mujoco_install/
bash .github/workflows/build_steps.sh build_web_viewer_wheel         # -> python/dist/*.whl
```

Then install and run:

```sh
pip install python/dist/mujoco-*.whl
python -m mujoco.experimental.studio.web_viewer     # open the printed URL in a browser
```

CI runs exactly these steps in the `web_viewer` job of `build.yml` and uploads the
wheel as an artifact.

> [!NOTE]
> The web viewer modules are Linux-only today (the build skips them on Windows).
> Folding these modules into the default `mujoco` wheel — so `pip install mujoco`
> ships the web viewer everywhere — is tracked by `TODO(robotics-simulation)` in
> `build_steps.sh`.
</details>