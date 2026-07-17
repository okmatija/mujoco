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
  add_subdirectory(${NETIMGUI_DIR} ${CMAKE_CURRENT_BINARY_DIR}/netimgui)
  if(NOT TARGET netimgui)
    return()
  endif()

  add_executable(web_client web_client.cc web_client_ui_link.cc
                 web_client_state_link.cc)

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

  # Compile with exception support so the try/catch around MainLoop can
  # catch throws escaping the RAF callback (emscripten disables EH by
  # default at compile time; -fexceptions below only covers linking).
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

  # Bundle the Filament assets (materials, IBL) into web_client.data. They are
  # exposed at /assets in the Emscripten virtual filesystem, where the
  # "filament:" resource provider registered in web_client.cc reads them.
  set(WEB_CLIENT_FILAMENT_ASSETS_DIR
      ${CMAKE_BINARY_DIR}/src/render/filament/assets
  )
  target_link_options(web_client PRIVATE
    "SHELL:--preload-file ${WEB_CLIENT_FILAMENT_ASSETS_DIR}@/assets"
  )
  add_dependencies(web_client mujoco_filament_assets)

  # Stage the browser artifacts into the source-tree dist/ folder, which
  # web_server.py serves by default (see _find_static_files_dir).
  set(WEB_CLIENT_DIST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dist)
  add_custom_command(
    TARGET web_client
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${WEB_CLIENT_DIST_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE_DIR:web_client>/web_client.js
        $<TARGET_FILE_DIR:web_client>/web_client.wasm
        $<TARGET_FILE_DIR:web_client>/web_client.data
        ${CMAKE_CURRENT_SOURCE_DIR}/index.html
        ${WEB_CLIENT_DIST_DIR}
    COMMENT "Staging web viewer artifacts into ${WEB_CLIENT_DIST_DIR}"
  )
else()
  # ---------------------------------------------------------------------------
  # ui_server pybind module (built from the Python bindings build).
  # ---------------------------------------------------------------------------
  if(WIN32)
    message(WARNING "The web viewer ui_server module is not yet supported on "
                    "Windows. Skipping.")
    return()
  endif()

  if(NOT TARGET mujoco::dep::implot)
    message(WARNING "mujoco::dep::implot target not found. The web viewer "
                    "ui_server module will not be built.")
    return()
  endif()

  add_subdirectory(${NETIMGUI_DIR} ${CMAKE_CURRENT_BINARY_DIR}/netimgui)
  if(NOT TARGET netimgui)
    return()
  endif()

  mujoco_pybind11_module(ui_server ui_server.cc)
  target_link_libraries(ui_server PRIVATE
    mujoco
    mujoco::dep::implot
    netimgui
    structs_header
    structs_wrappers
  )

  set(MUJOCO_WEB_VIEWER_TARGETS
      ui_server
      PARENT_SCOPE
  )
endif()
