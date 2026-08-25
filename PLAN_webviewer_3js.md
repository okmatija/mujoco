# Plan: three.js client for the Studio Web Viewer (`--gfx=web_3js`)

Branch `webviewer_3js` (worktree `/mnt/c/Dev/mujoco_3js`), forked from upstream/main @ 86e98601.

Goal: a second browser client for the existing Studio Web Viewer that renders the
scene with three.js instead of the Filament WASM app, selected with
`--gfx=web_3js` on viewer.py (and any script using `ViewerConfig.gfx`).
Flagship demo: `model/humanoid/humanoid.xml` simulated inside a Gaussian-splat
environment, with the splat bytes embedded in the model so they ship through the
existing `/model` (.mjb) endpoint.

## 1. What already exists on main (reuse all of it)

The server side needs almost nothing new:

- `web_server.py` serves everything on one port: `/` static files,
  `/model` = raw .mjb bytes (with chunked `?offset_bytes/size_bytes` fetch),
  `/state` WebSocket, `/ui` (NetImgui), `/drop` (model drag-drop).
- `/state` payload (`state_payload.h`, magic `MJWS`): tagged blocks —
  physics state vector (`mjSTATE_INTEGRATION` spec + mjtNum values),
  fixed-size render state (mjvCamera/mjvPerturb/mjvOption/mjOption/mjVisual/
  mjStatistic/render flags), optional extra `mjvGeom`s. Header carries
  `model_crc32`; when it changes the client must refetch `/model`.
  Flow control: client sends the text message `"state_ack"` after applying a
  payload (`_STATE_ACK_MESSAGE`, web_server.py:92).
- The current client already proves the "browser runs MuJoCo" pattern: it loads
  the .mjb, applies the state vector, runs `mj_forward`, renders. We keep that
  contract and only swap the renderer.
- Official JS/TS bindings live in `wasm/` (npm `@mujoco/mujoco`): mjModel array
  views (incl. `text_data`), `MjData`, `mj_setState`/`mj_forward`, `MjvScene`.
  (`wasm/demo_app` is a rough three.js demo — reference for the bindings API
  only, not a base to build on.)
- mjSpec custom text (`mjsText`) compiles into `mjModel.text_data` and therefore
  rides inside the .mjb. `text_data` is 0-terminated char data → binary-unsafe →
  embed splat bytes as base64.

## 2. Research results (2026-08-15)

- **mjswan** (github.com/ttktjmt/mjswan, Apache-2.0, very active, by Tatsuki
  Tsujimoto @ttktjmt) is the best source. Uses the official `@mujoco/mujoco`
  npm bindings + three ^0.181. Port candidates:
  - `src/mjswan/template/src/core/scene/scene.ts` — mjModel arrays → three.js
    (primitives, meshes w/ dedup + UVs, heightfields, infinite-plane shader
    replicating MuJoCo's GL_OBJECT_PLANE UVs, `MeshPhysicalMaterial` from
    `mat_*`), plus `textures.ts`, `lights.ts`, `tendons.ts`, `coordinate.ts`
    (z-up wxyz → y-up xyzw swizzles: pos `(x, z, -y)`).
  - per-frame sync in `core/engine/runtime.ts`: copy `xpos`/`xquat` into one
    `THREE.Group` per body (~10 lines).
  - `core/scene/splat.ts` (~87 lines) — the whole splat integration:
    `new SplatMesh({fileBytes, fileType: getSplatFileType(bytes)})` added to the
    same scene; base quaternion = 180° about X (COLMAP/OpenCV → three.js);
    offsets swizzled `(x*s, z*s, y*s)`; dispose must also remove Spark's
    auto-installed `SparkRenderer` (mid-async-sort persistence bug).
  - `src/mjswan/splat.py` — Python-side splat config schema (scale, xyz offset,
    rpy, optional invisible .glb collider); `examples/demo/splat.py` has working
    World Labs Marble CDN `.spz` URLs + calibration constants.
- **Splat library: Spark** (`@sparkjsdev/spark`, sparkjs.dev, MIT, World Labs;
  2.x active 2026). `SplatMesh` is a `THREE.Object3D` in the normal render
  pass: depthTest on / depthWrite off → correct occlusion both ways against
  opaque robot meshes, zero custom passes. WebGL2 (no WebGPU gate). Loads
  .spz/.sog/.ply/.splat/.ksplat from URL or ArrayBuffer. Cost: ~4.8 MB vendored
  ESM (embedded WASM). Second choice: three.js r186+ native `GaussianSplatMesh`
  (WebGPURenderer only — a renderer migration; revisit later).
  Format for embedding: **.spz** (~10–25 MB/room, open spec) or .sog (smallest);
  clean/compress scans with SuperSplat / `@playcanvas/splat-transform`.
- **kevinzakka**: `mjc_viewer` (archived; Brax-derived three.js, baked
  trajectories); `robopianist-demo` = zalo/mujoco_wasm pattern (same
  arrays→three.js conversion as mjswan, older); **mjviser** (active, viser-based)
  demonstrates the alternative server-side-GLB architecture — not chosen here,
  but its `conversions.py` is the reference for tricky texture/UV cases.
- **MuGS** (Renforce-Dynamics/MuGS, MIT): Python-native PyTorch+gsplat;
  composites MuJoCo *offscreen sensor renders* with splat backgrounds via
  segmentation-mask alpha blending for vision RL. Different problem (no shared
  depth buffer, not a browser). Concept validation + possible future
  server-side sensor counterpart; not client source material.

## 3. Design

### 3.1 Client (`python/mujoco/experimental/studio/web/client_3js/`)

Vite + TypeScript app; deps: `three`, `@sparkjsdev/spark`, `@mujoco/mujoco`
(see version-lock risk below). Build output → `web/dist_3js/`.

Boot sequence:
1. `GET /model` → ArrayBuffer → `MjModel` from mjb bytes via WASM bindings.
2. Build scene once: port of mjswan `scene.ts` (+ textures/lights/coordinate).
   One `THREE.Group` per body; geoms parented with local `geom_pos/quat`.
3. Splat: scan model text customs for `studio/splat` → base64-decode →
   Spark `SplatMesh({fileBytes})`; transform from `studio/splat/xform` numeric
   (pos 3, quat 4, scale 1). Port mjswan `splat.ts` conventions verbatim.
4. Open `/state?sid=...`: per binary payload — parse MJWS blocks,
   `mj_setState` with the payload's spec signature, `mj_forward`, copy
   `xpos/xquat` into body groups, reply `"state_ack"`. On `model_crc32` change,
   refetch `/model` and rebuild (this also makes `/drop` hot-swap work, splat
   included, for free).
5. Camera: client-side OrbitControls; initial framing from `mjStatistic`
   (extent/center) in the render-state block. Ignore `/ui` entirely in v1 —
   `web_3js` is a scene-only view.

Deliberately v2+: extra-geoms block, tendons/flex/decor (or an `MjvScene`-based
alternative path), perturb/drag back-channel, server camera mirroring, ImGui
overlay.

### 3.2 Server plumbing (tiny)

- `viewer_protocol.py`: add `'web_3js'` to `GFX_MODES`.
- `launch_passive.py:94`: `if config.gfx in ('web', 'webgl', 'web_3js')`.
- `web_viewer.py`: when `gfx == 'web_3js'`, resolve static dir to `dist_3js`
  (new `MUJOCO_WEB_VIEWER_DIST_3JS` env override for dev) and pass it as
  `WebServer(static_files_dir=...)`. No protocol changes. v1 leaves the
  headless ImGui/NetImgui machinery running but unconnected; a later cleanup
  can skip starting it in this mode.

Dev loop: `vite build --watch` into a dir + `MUJOCO_WEB_VIEWER_DIST_3JS` →
refresh browser. (Or vite dev-server with `/model`,`/state` proxied; start with
the simpler build-watch.)

### 3.3 Splat-in-.mjb convention

- New helper `python/mujoco/experimental/studio/splat.py`:
  `embed(spec, path_or_bytes, *, pos=(0,0,0), quat=(1,0,0,0), scale=1.0)` →
  `spec.add_text(name='studio/splat', data=base64)` +
  `spec.add_numeric(name='studio/splat/xform', data=[*pos, *quat, scale])`.
  Format auto-detected client-side (Spark `getSplatFileType`).
- `viewer.py` gains `--splat=path.spz` (+ optional `--splat_scale`,
  `--splat_pos`, `--splat_euler`): `parser.py` already does
  `MjSpec.from_file → compile`, so inject `splat.embed(spec, ...)` between the
  two (add an optional arg to `parser.parse`).
- Size: base64 costs +33%; a Marble .spz room ≈ 10–40 MB → .mjb grows by that.
  `/model` already supports parallel chunked fetch. If it hurts, later escape
  hatch: raw bytes behind a dedicated endpoint or a binary-safe custom field.

### 3.4 ImGui UI in the 3js client (high reuse)

The server side needs nothing: `headless_ui.cc` + the `/ui` NetImgui stream are
client-agnostic, so every Studio plugin's UI arrives as draw lists regardless
of which client renders them. On the client, the existing code is already
layered for this:

- `web_client_remote_ui.{h,cc}` (`RemoteUi`) — the whole NetImgui client:
  connects to `/ui`, assembles `ImDrawData` (`RemoteDrawData()`), captures and
  sends input back. Its renderer-facing `Callbacks` interface is just
  `UploadTexture` + `GpuReady`. Reused as is.
- `web_client_session.{h,cc}` (`Session`) — /state socket, acks, roster/role
  machine, model-change policy; talks outward through a small `Callbacks`
  surface (`OnPayload`, `OnModelChanged`, `ConnectRemoteUi`, ...). Reused as is.
- `web_client_local_ui.{h,cc}` — role window + DISCONNECTED notices, depends
  only on ImGui. Reused as is.
- `web_client.cc` — the only Filament-specific file; this is what gets replaced.

Plan: build a `web_client_3js` Emscripten module = session + remote UI +
local UI + ImGui with the stock `imgui_impl_opengl3` backend (WebGL2 under
Emscripten), rendering to a **transparent overlay canvas** stacked above the
three.js canvas. New code is thin: an `AppCallbacks` that implements
`UploadTexture`/`GpuReady` with plain GL and bridges `OnPayload`/
`OnModelChanged` to the JS scene via embind, plus input routing — the overlay
canvas owns pointer/keyboard events, feeds ImGui, and forwards them to the
three.js OrbitControls whenever `io.WantCaptureMouse/Keyboard` is false.
Two canvases → two GL contexts (fine; DOM-composited); sync DPR/resize and
call `SetMaxClip` each frame.

Bonus: once this module exists, link mujoco into it and move
`mj_setState`/`mj_forward` (and .mjb loading) there, exposing typed-array views
of `xpos`/`xquat`/model arrays to JS. That drops the npm `@mujoco/mujoco`
dependency and dissolves the .mjb version-lock risk — the module builds from
this tree. End state: the 3js client is the existing C++ client minus Filament,
with three.js + Spark as the renderer.

Rejected alternatives: reimplementing NetImgui in TypeScript (large, drifts
from the C++ protocol code) and an HTML/React UI (breaks the core Studio
property that server-side plugin UIs work unmodified).

### 3.5 Demo (acceptance target)

`python -m mujoco.experimental.studio.viewer --model=model/humanoid/humanoid.xml \
  --gfx=web_3js --splat=street.spz` → humanoid walking/ragdolling inside the
splat scene, correct occlusion. Asset: a World Labs Marble `.spz` (mjswan's
demo pulls from cdn.marble.worldlabs.ai; their Unitree-G1 "street" calibration
scale 3.275 / z 0.708 / yaw 40 is a reference). Floor: mjswan's infinite-plane
material uses `depthWrite:false` so the splat ground reads as the floor —
mirror that; alternatively hide the plane when a splat is present.

## 4. Milestones

Status 2026-08-15: M1–M4 implemented and committed; verified headlessly (node
+ WSL) end to end except browser pixels. Key verification results:

- npm `@mujoco/mujoco` 3.10.0 loads a 3.10.0 .mjb, steps the humanoid, and
  `mj_setState(model, data, number[], spec)` works. Only broken bindings
  fields: `light_active`, `light_castshadow` (bool memory views; client falls
  back to defaults).
- .mjb version check is EXACT (engine_io.c header compare) — server and
  client MuJoCo versions must match to the patch digit.
- Embed chain: humanoid.xml + street.spz (8.1 MB) -> splat.embed -> 11.9 MB
  .mjb -> WASM extraction -> byte-identical .spz, Spark sniffs 'spz'.
  Compile time with the 8 MB text custom: 0.07 s. No size limits hit.
- Full server run in WSL: python3.12 venv + the 3.11.1 Linux wheel built on
  the web_viewer branch (which DOES ship headless_ui/state_payload .so) +
  main-tree studio .py overlaid. `viewer.py --gfx=web_3js --splat=...` serves
  the three.js client at `/`, the 11.9 MB model at `/model`, and streams
  ~55 payloads/s on `/state`; the client's parseStatePayload validates them
  (crc match, spec 16383, humanoid ragdolling live). Venv recipe:
  `python3.12 -m venv v && pip install <wheel> websockets absl-py`, copy
  `studio/*.py` + `studio/web/*.py` from this tree over site-packages, set
  MUJOCO_WEB_VIEWER_DIST_3JS at web/dist_3js.
- In-browser LIVE rendering is blocked only by version pairing: dist_3js
  bundles npm 3.10.0, so it pairs with a 3.10.0 server (the Windows venv,
  once web natives exist there) but not the 3.11.1 WSL wheel. Next step for
  live pixels: build `wasm/` from this tree (3.11.1) with emsdk and point the
  client at that build. The serverless static demo (`npm run dev`,
  public/model built at 3.10.0) works with the released npm package today.

1. **M1 static render**: scaffold client_3js; humanoid.xml compiled to .mjb,
   fetched from the real server, rendered with orbit camera (scene.ts port).
2. **M2 live state**: MJWS parsing, mj_setState/mj_forward, acks, crc hot-swap.
3. **M3 flag plumbing**: `--gfx=web_3js` end-to-end from viewer.py.
4. **M4 splats**: `splat.embed` + `--splat` + Spark rendering + humanoid demo.
5. **M5 UI overlay**: `web_client_3js` WASM module (session + NetImgui +
   ImGui/`imgui_impl_opengl3` on a transparent overlay canvas, §3.4); then fold
   mujoco into it and drop the npm bindings.
6. **M6 polish**: extra geoms, perf (float32 state?), camera niceties, docs.

## 4.5 M5 as built (2026-08-15, second session)

The UI overlay became `web_client_ui` (web/web_client_3js.cc + the reused
session/remote_ui/local_ui/state_payload + imgui_widgets), built by the plain
`emcmake cmake` tree build (top-level adds `studio/web` under EMSCRIPTEN;
1.6 MB wasm, no Filament). It owns the model (mjVFS from /model bytes), the
/state session (mj_setState + mj_forward), the /ui NetImgui stream, and the
camera (incl. spectator cam modes); renders streamed + local UI with
imgui_impl_opengl3 on the transparent `#ui-canvas`; and exposes an embind
bridge (pose/light/camera typed views re-fetched per frame, input events in,
keyFromDomCode lookup, model load, onModelChanged callback out). The page
(uioverlay.ts) forwards all input to ImGui — scene interaction reaches the
headless viewer over NetImgui exactly like the Filament client, so
OrbitControls is disabled and the TS StateClient is bypassed whenever the
module is deployed (public/uimod/); without it the standalone TS path still
works. Build wiring ported from the web_viewer branch: web/CMakeLists.txt
(pybind headless_ui/state_payload in the python build + both wasm clients
under EMSCRIPTEN), studio/CMakeLists hunks, setup.py (web extensions added,
stale parser extension dropped), pyproject (websockets dep, web package
data). Version pairing is solved by building everything from this tree:
wasm bindings via `emcmake cmake -B ~/build/mj3js_wasm` (client_3js now
aliases @mujoco/mujoco to wasm/dist, tree API = MjVFS.addBuffer), server
wheel via host install + make_sdist + pip wheel (scratchpad
build_server_wheel.sh; ~/build/mj3js_{host,install,wheels}).
MUJOCO_WASM_THREADS=OFF for all wasm (avoids COOP/COEP).

## 5. Risks / open questions

- **.mjb version lock**: npm `@mujoco/mujoco` must match the server's MuJoCo
  version for the .mjb to load. Robust answer: build `wasm/` from this tree
  (its CMakeLists exists) and vendor the artifact; pragmatic M1: pin the npm
  version and check. Decide by M2.
- **text_data size**: verify the compiler is happy with a ~30 MB base64 text
  custom (no known hard cap; test early in M4 with a real .spz).
- **Splat asset licensing** for a checked-in demo asset; CDN URL in the sample
  script avoids committing binaries.
- **Attribution**: ported mjswan files keep Apache-2.0 headers + provenance
  note (same license as MuJoCo).
- Whether `dist_3js` build output is checked in (like `dist`?) or built in CI —
  follow whatever the Filament client ends up doing.
