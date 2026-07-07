# MuJoCo WebViewer — Ported from Google3

This directory contains the MuJoCo WebViewer prototype, exported from Google3
CL cl/900757204. The code streams a MuJoCo simulation running in Python to a
browser using NetImgui for the Dear ImGui UI and Filament for 3D rendering.

## What was copied

### `experimental/studio/` (this directory)

Files from `//third_party/py/mujoco/experimental/studio/` in the CL:

| File | Description |
|---|---|
| `index.html` | MuJoCo Live browser page (loads WASM, sets up canvas) |
| `live_state.h` | C++ header — serialization of visualization state (camera, options, render flags) sent alongside physics state |
| `state_server.py` | Python WebSocket server that streams simulation state to the browser at ~60Hz via shared memory |
| `studio.py` | Main entry point — updated with `--web_viewer` flag to choose between native and web viewer |
| `ui_server.cc` | pybind11 C++ module — headless ImGui context + NetImgui client that streams UI draw data to the browser |
| `ui_server_proxy.py` | Python WebSocket-to-TCP proxy bridging the C++ NetImgui server and the browser's WebSocket |
| `web_client.cc` | WASM binary — runs in the browser, receives NetImgui draw data + simulation state, renders via Filament + ImGui |
| `web_server.py` | HTTP server serving static files (HTML/WASM) + model MJB, plus the WS-to-TCP proxy |
| `BUILD` | Google3 Blaze build rules (for reference only) |

### `experimental/netimgui/`

The full NetImgui library from `//third_party/netimgui/`, which provides the
remote ImGui protocol used to stream Dear ImGui draw data over the network.

| Directory | Contents |
|---|---|
| `Code/Client/` | NetImgui client API (`NetImgui_Api.h`) and implementation |
| `Code/Client/Private/` | Internal client sources — networking, command packets, draw frame compression |
| `Code/ServerApp/Source/` | NetImgui server app sources (remote client management, UI, config) |
| `Code/Sample/` | Sample applications including `SampleNoBackend` (headless pattern used by `ui_server.cc`) |
| `google/` | **Google-specific additions**: `NetImgui_NetworkWASM.cpp` (WebSocket networking for Emscripten) and `logging.h` (logging adapter) |
| `LICENSE` | MIT license |

## Architecture

```
Python (Linux)                          Browser (WASM)
┌─────────────────────┐                ┌──────────────────────┐
│  studio.py          │                │  web_client.cc       │
│    ↓                │                │  (Emscripten WASM)   │
│  ui_server.cc       │   NetImgui     │                      │
│  (headless ImGui)   │───TCP/WS──────▶│  ImGui draw data     │
│    ↓                │   protocol     │  rendered via         │
│  state_server.py    │                │  Filament + ImGui    │
│  (physics state)    │───WebSocket───▶│                      │
└─────────────────────┘                └──────────────────────┘
        │                                       │
   ui_server_proxy.py                    State WebSocket
   (TCP↔WS bridge)                      (port 8891)
   web_server.py
   (HTTP + proxy)
```

## What needs porting

### `ui_server.cc` — Google-internal deps to replace
- `file/base/helpers.h`, `file/base/options.h`, `file/base/path.h` → use `<filesystem>` or equivalent
- `third_party/mujoco/google/runfiles/runfiles.h` → replace with a local font path lookup
- `third_party/py/mujoco/structs.h` → the pybind11 wrapper types (`MjvCameraWrapper`, etc.)
- `third_party/pybind11/` → standard pybind11

### `web_client.cc` — Google-internal deps to replace
- `third_party/mujoco/src/experimental/filament/` → MuJoCo's Filament rendering (already in the mujoco repo)
- `third_party/mujoco/src/experimental/platform/` → Window, Renderer, ModelHolder, interaction utilities
- `third_party/dear_imgui/`, `third_party/implot/`, `third_party/SDL2/` → standard versions
- NetImgui includes use short paths (`"NetImgui_Api.h"`) that assume `-I` flags from BUILD

### `studio.py` — Google-internal code to remove
- `copybara:strip_begin/end` blocks contain Google-internal imports (`assetdb_browser`, `web_viewer`)
- Remove or replace with open-source equivalents

### `netimgui/google/` — Google-specific networking
- `NetImgui_NetworkWASM.cpp` — Emscripten WebSocket-based networking (this is the key addition over upstream NetImgui which only has POSIX sockets)
- `logging.h` — Logging adapter (maps to `printf` / Google logging)

### Build system
- The `BUILD` file is Blaze/Bazel — you'll need CMake rules
- The WASM binary (`web_client.cc`) was built with `wasm_cc_binary` + Emscripten toolchain
- `ui_server.cc` was built as a `pybind_extension`
