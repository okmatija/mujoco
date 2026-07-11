load("//devtools/python/blaze:pytype.bzl", "pytype_binary", "pytype_library", "pytype_strict_binary")
load("//third_party/bazel_rules/rules_cc/cc:cc_binary.bzl", "cc_binary")
load("//third_party/bazel_rules/rules_cc/cc:cc_library.bzl", "cc_library")
load("//third_party/bazel_rules/rules_wasm/wasm:defs.bzl", "wasm_cc_binary")
load("//third_party/gpus/cuda:build_defs.bzl", "if_cuda")
load("//third_party/pybind11/google3_utils:build_defs.bzl", "pybind_extension")

package(
    default_applicable_licenses = ["//devtools/compliance/licenses:no_external_contributions"],
    default_visibility = ["//third_party/py/mujoco:visibility"],
    licenses = ["unencumbered"],  # Google-authored code
)

pybind_extension(
    name = "parser",
    srcs = ["parser.cc"],
    py_deps = ["//third_party/py/mujoco"],
    deps = [
        "//third_party/mujoco",  # buildcleaner: keep
        "//third_party/mujoco/src/experimental/platform/sim:model_holder",
        "//third_party/py/mujoco:structs_header",
    ],
)

pybind_extension(
    name = "native_viewer_cc",
    srcs = ["native_viewer.cc"],
    defines = if_cuda(["CUDA"]),
    # Disabled import_test as this target depends on :window, which hardcodes a local desktop
    # dynamic linker path -Wl,--dynamic-linker=/lib64/... incompatible with Forge remote execution.
    # Additionally there is no X11 support on Borg which precludes running window-dependent targets.
    import_test = False,
    py_deps = ["//third_party/py/mujoco"],
    deps = [
        "//file/base",
        "//file/base:options_cc",
        "//file/base:path",
        "//third_party/implot",
        "//third_party/mujoco",  # buildcleaner: keep
        "//third_party/mujoco/google/runfiles",
        "//third_party/mujoco/src/experimental/platform/hal:graphics_mode",
        "//third_party/mujoco/src/experimental/platform/hal:renderer",
        "//third_party/mujoco/src/experimental/platform/hal:window",
        "//third_party/py/mujoco:structs_header",
    ],
)

pybind_extension(
    name = "renderer",
    srcs = ["renderer.cc"],
    py_deps = ["//third_party/py/mujoco"],
    deps = [
        "//third_party/mujoco",  # buildcleaner: keep
        "//third_party/mujoco/src/experimental/platform/hal:graphics_mode",
        "//third_party/mujoco/src/experimental/platform/hal:renderer",  # buildcleaner: keep
        "//third_party/py/mujoco:structs_header",
    ],
)

pybind_extension(
    name = "ux",
    srcs = ["ux.cc"],
    py_deps = ["//third_party/py/mujoco"],
    deps = [
        "//third_party/dear_imgui",
        "//third_party/mujoco",  # buildcleaner: keep
        "//third_party/mujoco/src/experimental/platform:helpers",
        "//third_party/mujoco/src/experimental/platform/sim:step_control",
        "//third_party/mujoco/src/experimental/platform/ux:gui",
        "//third_party/mujoco/src/experimental/platform/ux:interaction",
        "//third_party/py/mujoco:structs_header",
    ],
)

pybind_extension(
    name = "sim",
    srcs = ["sim.cc"],
    py_deps = ["//third_party/py/mujoco"],
    deps = [
        "//third_party/mujoco",  # buildcleaner: keep
        "//third_party/mujoco/src/experimental/platform/sim:step_control",
        "//third_party/py/mujoco:structs_header",
    ],
)

pytype_library(
    name = "studio_app_events",
    srcs = ["studio_app_events.py"],
    deps = [
        ":sim",
        ":ux",
        "//third_party/dear_imgui/google/py:dear_imgui",
        "//third_party/py/mujoco",
        "//third_party/py/numpy",
    ],
)

pytype_library(
    name = "studio_app",
    srcs = ["studio_app.py"],
    deps = [
        ":parser",
        ":sim",
        ":studio_app_events",
        ":ux",
        ":viewer_protocol",
        "//third_party/dear_imgui/google/py:dear_imgui",
        "//third_party/py/mujoco",
        "//third_party/py/numpy",
    ],
)

pytype_library(
    name = "viewer_protocol",
    srcs = ["viewer_protocol.py"],
    deps = [
        ":ux",
        "//third_party/py/mujoco",
    ],
)

pytype_library(
    name = "native_viewer",
    srcs = ["native_viewer.py"],
    deps = [
        ":native_viewer_cc",
        ":ux",
        ":viewer_protocol",
        "//third_party/py/mujoco",
    ],
)

pytype_binary(
    name = "studio",
    srcs = ["studio.py"],
    strict_deps = False,
    deps = [
        ":native_viewer",
        ":studio_app",
        ":viewer_protocol",
        "//third_party/mujoco",
        "//third_party/mujoco/google/gfx:osmesa_deps",
        "//third_party/mujoco/google/rsdb",  # buildcleaner: keep
        "//third_party/mujoco/src/experimental/platform/hal:renderer",
        "//third_party/mujoco/src/experimental/py:web_viewer",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/mujoco/google:assetdb_browser",
    ],
)

# Headless ImGui context and NetImgui client. Manages the ImGui frame
# lifecycle and streams UI draw data to the browser via NetImgui.
pybind_extension(
    name = "ui_server",
    srcs = ["ui_server.cc"],
    copts = [
        "-Ithird_party/dear_imgui",
    ],
    data = [
        "//third_party/mujoco/src/experimental/platform:fonts",
    ],
    tags = ["notap"],
    deps = [
        ":render_state",
        "//file/base",
        "//file/base:options_cc",
        "//file/base:path",
        "//third_party/dear_imgui",
        "//third_party/implot",
        "//third_party/mujoco:libmujoco_hdrs",
        "//third_party/mujoco/google/runfiles",
        "//third_party/netimgui",
        "//third_party/netimgui/google:logging",
        "//third_party/py/mujoco:structs_header",
    ],
)

# Serves the web viewer page (HTML/WASM) and proxies the NetImgui
# protocol between the headless UiServer and the browser.
pytype_library(
    name = "ui_server_proxy",
    srcs = ["ui_server_proxy.py"],
)

# The web viewer's single-port server: page/WASM/model over HTTP, /ui
# bridging the NetImgui stream, /state broadcasting the state payload.
pytype_library(
    name = "web_server",
    srcs = ["web_server.py"],
    data = [
        "index.html",
        ":web_client_wasm",
        "//third_party/mujoco/src/experimental/filament/assets",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":ui_server",
        ":ui_server_proxy",
        ":ux",
        "//third_party/dear_imgui/google/py:dear_imgui",
        "//third_party/mujoco",
        "//third_party/py/mujoco",
        "//third_party/py/numpy",
        "//third_party/py/websockets",
    ],
)

cc_library(
    name = "render_state",
    hdrs = ["render_state.h"],
    deps = ["//third_party/mujoco:libmujoco_hdrs"],
)

exports_files(["index.html"])

# WASM binary — renders remote ImGui draw data + MuJoCo 3D scene in browser.
cc_binary(
    name = "web_client_cc",
    srcs = [
        "web_client.cc",
    ],
    copts = [
        "-Ithird_party/dear_imgui",
        "-Wno-macro-redefined",
        "-I.",
    ],
    data = [
        "//third_party/mujoco/src/experimental/platform:fonts",
    ],
    defines = [
        "NDEBUG",
    ],
    linkopts = [
        "-sALLOW_MEMORY_GROWTH=1",
        "-sFETCH=1",
        "-sGL_ASSERTIONS=0",
        "-sSTACK_SIZE=512mb",
        # Filament's GL backend calls glGetError() on every draw call via
        # GLUtils::checkGLError/assertGLError, even in non-debug builds (the
        # functions are compiled into the library unconditionally). In WebGL,
        # each glGetError() forces a synchronous GPU pipeline flush across the
        # JS-WASM bridge (~0.2ms each, hundreds per frame). Wrapping it at
        # link time replaces all call sites with a no-op.
        "-Wl,--wrap=glGetError",
    ],
    tags = ["nobuilder"],
    target_compatible_with = ["//third_party/bazel_platforms/cpu:wasm32"],
    deps = [
        ":render_state",
        "//third_party/SDL2",
        "//third_party/dear_imgui",
        "//third_party/dear_imgui:imgui_sdl2",
        "//third_party/emscripten:GL-webgl2-full_es3",
        "//third_party/filament/filament:filament_gles3",
        "//third_party/implot",
        "//third_party/mujoco",
        "//third_party/mujoco:libmujoco_hdrs",
        "//third_party/mujoco/src/experimental/filament",
        "//third_party/mujoco/src/experimental/platform/hal:renderer",
        "//third_party/mujoco/src/experimental/platform/hal:window",
        "//third_party/mujoco/src/experimental/platform/sim:model_holder",
        "//third_party/mujoco/src/experimental/platform/ux:interaction",
        "//third_party/netimgui",
        "//third_party/netimgui/google:logging",
        "//third_party/netimgui/google:netimgui_wasm",
    ],
)

wasm_cc_binary(
    name = "web_client_wasm",
    cc_target = ":web_client_cc",
    threads = "off",
)

pytype_strict_binary(
    name = "studio_link",
    srcs = ["studio_link.py"],
    deps = [
        ":studio_app",
        ":web_server",
        "//third_party/py/absl:app",
    ],
)
