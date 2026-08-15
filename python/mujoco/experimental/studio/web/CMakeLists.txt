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

set(NETIMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../netimgui)

if(EMSCRIPTEN)
  include(third_party_deps/dear_imgui)
  include(third_party_deps/implot)
  add_subdirectory(${NETIMGUI_DIR} ${CMAKE_CURRENT_BINARY_DIR}/netimgui)
  if(NOT TARGET netimgui)
    return()
  endif()

  # ---------------------------------------------------------------------------
  # UI overlay module for the three.js client (--gfx=web_3js).
  #
  # Owns the model (parsed from /model bytes the page passes in), the /state
  # session, the NetImgui /ui stream, and the camera; renders the streamed
  # Studio UI plus the local role window with the ImGui OpenGL3 backend on a
  # transparent overlay canvas. The page's three.js code renders the 3D scene,
  # reading body poses and the camera through the embind bridge
  # (web_client_3js.cc). Builds in the plain wasm bindings build; no Filament.
  # ---------------------------------------------------------------------------
  add_executable(web_client_ui
    web_client_3js.cc
    web_client_local_ui.cc
    web_client_remote_ui.cc
    web_client_session.cc
    state_payload.cc
    ${PROJECT_SOURCE_DIR}/src/experimental/platform/ux/imgui_widgets.cc
  )
  target_include_directories(web_client_ui PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
  )
  target_link_libraries(web_client_ui PRIVATE
    dear_imgui
    dear_imgui_OpenGL3
    implot
    netimgui
    mujoco::mujoco
  )
  # Exceptions: MuJoCo errors are surfaced as C++ throws by mju_user_error in
  # web_client_3js.cc; catches around the frame loop keep the page alive.
  target_compile_options(web_client_ui PRIVATE -fexceptions)
  target_link_options(web_client_ui PRIVATE
    --bind
    -sUSE_WEBGL2=1
    -sMIN_WEBGL_VERSION=2
    -sMAX_WEBGL_VERSION=2
    -sALLOW_MEMORY_GROWTH=1
    -sMODULARIZE=1
    -sEXPORT_ES6=1
    -sEXPORT_NAME=createWebClientUi
    -sENVIRONMENT=web,worker
    "SHELL:-s EXPORTED_RUNTIME_METHODS=HEAPU8,stringToNewUTF8"
    -sDYNAMIC_EXECUTION=1
    -fexceptions
  )

  # Stage into the three.js client's public dir so vite serves the module at
  # /uimod/ in both dev and build.
  set(WEB_CLIENT_UI_DIST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/client_3js/public/uimod)
  add_custom_command(
    TARGET web_client_ui
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${WEB_CLIENT_UI_DIST_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE_DIR:web_client_ui>/web_client_ui.js
        $<TARGET_FILE_DIR:web_client_ui>/web_client_ui.wasm
        ${WEB_CLIENT_UI_DIST_DIR}
    COMMENT "Staging web_client_ui into client_3js/public/uimod"
  )

  # ---------------------------------------------------------------------------
  # Filament browser client (web_client), from the web_viewer branch. Builds
  # only in the Studio emscripten build (needs mujoco::platform and the
  # Filament assets from the native host build).
  # ---------------------------------------------------------------------------
  if(TARGET mujoco::platform)
    add_executable(web_client web_client.cc web_client_local_ui.cc
                   web_client_remote_ui.cc web_client_session.cc
                   state_payload.cc)

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
    # throws escaping the requestAnimationFrame callback.
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
      -sDYNAMIC_EXECUTION=1
      -sGL_ASSERTIONS=0
      -sSTACK_SIZE=32mb
      # Reserve a large contiguous virtual address block upfront so big models
      # do not fail dynamic memory.grow calls due to heap fragmentation.
      -sINITIAL_MEMORY=3072mb
      -sMAXIMUM_MEMORY=4gb
      -fexceptions
      # Filament's GL backend calls glGetError() on every draw call; in WebGL
      # each call forces a synchronous flush. Wrap it away.
      -Wl,--wrap=glGetError
    )

    # Stage the Studio fonts so the browser can load them.
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

    set(WEB_CLIENT_FILAMENT_ASSETS_DIR
        ${CMAKE_BINARY_DIR}/src/render/filament/assets
    )
    add_dependencies(web_client mujoco_filament_assets web_client_fonts)

    # Stage the browser artifacts into the source-tree dist/ folder, which
    # web_server.py serves by default (see find_static_files_dir).
    set(WEB_CLIENT_DIST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dist)
    add_custom_command(
      TARGET web_client
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E make_directory ${WEB_CLIENT_DIST_DIR}/assets
      COMMAND ${CMAKE_COMMAND} -E copy
          $<TARGET_FILE_DIR:web_client>/web_client.js
          $<TARGET_FILE_DIR:web_client>/web_client.wasm
          ${CMAKE_CURRENT_SOURCE_DIR}/index.html
          ${PROJECT_SOURCE_DIR}/src/experimental/studio/favicon.ico
          ${WEB_CLIENT_DIST_DIR}
      COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${WEB_CLIENT_FILAMENT_ASSETS_DIR} ${WEB_CLIENT_DIST_DIR}/assets
      COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${WEB_CLIENT_FONTS_DIR} ${WEB_CLIENT_DIST_DIR}/assets
      COMMENT "Staging web viewer artifacts + assets into ${WEB_CLIENT_DIST_DIR}"
    )
  endif()
else()
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
    mujoco::platform  # for the shared Studio font loading (ux/fonts)
    mujoco::dep::implot
    netimgui
  )

  mujoco_pybind11_module(state_payload state_payload_py.cc state_payload.cc)
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
