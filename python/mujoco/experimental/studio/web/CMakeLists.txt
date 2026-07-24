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

# MuJoCo web viewer.

set(NETIMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../netimgui)

if(EMSCRIPTEN)
  # ---------------------------------------------------------------------------
  # Browser WASM client (built from the main MuJoCo tree).
  # ---------------------------------------------------------------------------
  # netimgui always builds under Emscripten; if its target is somehow missing
  # (a real misconfiguration) the target_link_libraries below fails loudly,
  # which is what we want.
  add_subdirectory(${NETIMGUI_DIR} ${CMAKE_CURRENT_BINARY_DIR}/netimgui)

  add_executable(web_client web_client.cc web_client_local_ui.cc
                 web_client_remote_ui.cc web_client_session.cc)

  target_include_directories(web_client
    PRIVATE
      ${PROJECT_SOURCE_DIR}/include
      ${PROJECT_SOURCE_DIR}/src
  )

  target_link_libraries(web_client
    PRIVATE
      dear_imgui
      implot
      netimgui
      mujoco::mujoco
      mujoco::platform
      mujoco::render_noop
      SDL2::SDL2-static
  )

  # Ensure resource decoder plugins are linked into the WASM binary.
  target_link_options(web_client PRIVATE
    -Wl,--whole-archive $<TARGET_FILE:mujoco> -Wl,--no-whole-archive
  )

  # Compile with exception support so the try/catch around MainLoop can catch
  # throws escaping the requestAnimationFrame callback. Emscripten disables
  # exception handling by default at compile time; the -fexceptions in the
  # link options below only covers linking.
  target_compile_options(web_client PRIVATE -fexceptions)

  # Filament's OpenGL backend requires WebGL2 / full ES3 bindings.
  target_link_options(web_client PRIVATE
    --bind
    -sUSE_WEBGL2=1
    -sFULL_ES3
    -sMIN_WEBGL_VERSION=2
    -sMAX_WEBGL_VERSION=2
    -sALLOW_MEMORY_GROWTH=1
    -sFETCH=1
    # embind (EMSCRIPTEN_BINDINGS registerAsset/startApp) needs eval enabled.
    -sDYNAMIC_EXECUTION=1
    -sGL_ASSERTIONS=0
    -sSTACK_SIZE=32mb
    -sINITIAL_MEMORY=1024mb
    -sMAXIMUM_MEMORY=4gb
    -fexceptions
    # Filament's GL backend calls glGetError() on every draw call via
    # GLUtils::checkGLError/assertGLError, even in non-debug builds (the
    # functions are compiled into the library unconditionally). In WebGL,
    # each glGetError() forces a synchronous GPU pipeline flush across the
    # JS-WASM bridge (~0.2ms each, hundreds per frame). Wrapping it at
    # link time replaces all call sites with a no-op.
    -Wl,--wrap=glGetError
  )

  # Stage the three Studio UI fonts into their own directory so the browser's
  # local ImGui (the Link window) can load them, matching the native viewer.
  # They are fetched into _deps by the studio font dependencies; copy them
  # here (the copy handles the [wght] brackets in the variable font's name).
  set(WEB_CLIENT_FONTS_DIR ${CMAKE_CURRENT_BINARY_DIR}/web_fonts)
  add_custom_target(web_client_fonts
    COMMAND ${CMAKE_COMMAND} -E make_directory ${WEB_CLIENT_FONTS_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy
      "${CMAKE_BINARY_DIR}/_deps/atkinson_hyperlegible_next-src/fonts/variable/AtkinsonHyperlegibleNext[wght].ttf"
      "${WEB_CLIENT_FONTS_DIR}/AtkinsonHyperlegibleNext[wght].ttf"
    COMMAND ${CMAKE_COMMAND} -E copy
      ${CMAKE_BINARY_DIR}/_deps/font_awesome-src/fonts/fontawesome-webfont.ttf
      ${WEB_CLIENT_FONTS_DIR}/fontawesome-webfont.ttf
    COMMAND ${CMAKE_COMMAND} -E copy
      "${CMAKE_BINARY_DIR}/_deps/atkinson_hyperlegible_mono-src/fonts/ttf/AtkinsonHyperlegibleMono-Regular.ttf"
      ${WEB_CLIENT_FONTS_DIR}/AtkinsonHyperlegibleMono-Regular.ttf
    COMMENT "Staging web viewer UI fonts"
  )

  # The Filament assets (materials, IBL) and UI fonts are served as plain files
  # under /assets (staged into dist/assets below). The page fetches them and
  # pushes them into the client via registerAsset() before startApp() runs (see
  # index.html and web_client.cc's AssetRegistry). No --preload-file is used, so
  # no web_client.data is produced.
  set(WEB_CLIENT_FILAMENT_ASSETS_DIR
      ${CMAKE_BINARY_DIR}/src/render/filament/assets
  )
  add_dependencies(web_client mujoco_filament_assets web_client_fonts)

  # Stage the browser artifacts into the source-tree dist/ folder, which
  # web_server.py serves by default (see _find_static_files_dir). Assets go in
  # dist/assets/ so the page can fetch them at /assets/<name>.
  set(WEB_CLIENT_DIST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dist)
  add_custom_command(
    TARGET web_client
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${WEB_CLIENT_DIST_DIR}/assets
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE_DIR:web_client>/web_client.js
        $<TARGET_FILE_DIR:web_client>/web_client.wasm
        ${CMAKE_CURRENT_SOURCE_DIR}/index.html
        ${CMAKE_CURRENT_SOURCE_DIR}/favicon.png
        ${WEB_CLIENT_DIST_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${WEB_CLIENT_FILAMENT_ASSETS_DIR} ${WEB_CLIENT_DIST_DIR}/assets
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${WEB_CLIENT_FONTS_DIR} ${WEB_CLIENT_DIST_DIR}/assets
    COMMENT "Staging web viewer artifacts + assets into ${WEB_CLIENT_DIST_DIR}"
  )
else()
  # ---------------------------------------------------------------------------
  # headless_ui and state_payload pybind modules (built from the Python
  # bindings build).
  # ---------------------------------------------------------------------------
  if(WIN32)
    message(WARNING "The web viewer modules are not yet supported on "
                    "Windows. Skipping.")
    return()
  endif()

  if(NOT TARGET mujoco::dep::implot)
    message(WARNING "mujoco::dep::implot target not found. The web viewer "
                    "modules will not be built.")
    return()
  endif()

  add_subdirectory(${NETIMGUI_DIR} ${CMAKE_CURRENT_BINARY_DIR}/netimgui)
  if(NOT TARGET netimgui)
    return()
  endif()

  mujoco_pybind11_module(headless_ui headless_ui.cc)
  target_link_libraries(headless_ui PRIVATE
    mujoco
    mujoco::dep::implot
    netimgui
  )

  mujoco_pybind11_module(state_payload state_payload.cc)
  target_link_libraries(state_payload PRIVATE
    mujoco
    structs_header
    structs_wrappers
  )

  set(MUJOCO_WEB_VIEWER_TARGETS
      headless_ui
      state_payload
      PARENT_SCOPE
  )
endif()
