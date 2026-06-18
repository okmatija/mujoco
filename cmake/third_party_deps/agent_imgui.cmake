# Copyright 2025 DeepMind Technologies Limited
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

set(MUJOCO_DEP_VERSION_agent_imgui
    6d435b8b980062a2af441009c364675196889f17
    CACHE STRING "Tag/version of `agent_imgui` to be fetched."
)
mark_as_advanced(MUJOCO_DEP_VERSION_agent_imgui)

include(FindOrFetch)

# agent_imgui ships its own CMakeLists.txt (no CUSTOM_CMAKE needed) and expects
# the `dear_imgui` and `imgui_test_engine` targets to already be defined, so it
# must be included after those dependencies.
fetchpackage(
    PACKAGE_NAME  agent_imgui
    GIT_REPO      https://github.com/okmatija/agent_imgui.git
    GIT_TAG       ${MUJOCO_DEP_VERSION_agent_imgui}
)
