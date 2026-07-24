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

# Hermetic, self-verifying patch application for FetchContent source trees.
#
# Usage (as a FetchContent PATCH_COMMAND, which runs with the working
# directory set to the source tree being patched):
#
#   ${CMAKE_COMMAND} -DPATCH_FILE=<absolute path> -P ApplyPatch.cmake
#
# Rationale: `git apply` resolves patch paths against whatever git context is
# in effect, not the working directory. If the tree being patched is not
# itself a git repository (e.g. it was extracted from a tarball via
# MUJOCO_CMAKE_DEP_CACHE) and an enclosing repository is discovered — or
# GIT_DIR/GIT_WORK_TREE are inherited from the environment, which BYPASSES
# GIT_CEILING_DIRECTORIES entirely — then every path in the patch falls
# outside the working directory's prefix and `git apply` SKIPS ALL OF THEM
# AND EXITS 0. The tree is silently left unpatched and the build fails later
# with an unrelated-looking error. This script neutralizes the ambient git
# context, applies the patch, and then verifies that it actually applied,
# so a failure is loud and points at the patch step.

if(NOT DEFINED PATCH_FILE)
  message(FATAL_ERROR "ApplyPatch: PATCH_FILE must be defined.")
endif()
if(NOT EXISTS "${PATCH_FILE}")
  message(FATAL_ERROR "ApplyPatch: patch file does not exist: ${PATCH_FILE}")
endif()

find_program(GIT_EXECUTABLE git REQUIRED)

# Neutralize ambient git context. GIT_DIR / GIT_WORK_TREE / GIT_INDEX_FILE
# override repository discovery entirely (GIT_CEILING_DIRECTORIES does not
# protect against them), silently redirecting `git apply` to another
# repository.
foreach(_var
        GIT_DIR
        GIT_WORK_TREE
        GIT_INDEX_FILE
        GIT_COMMON_DIR
        GIT_OBJECT_DIRECTORY
        GIT_ALTERNATE_OBJECT_DIRECTORIES
        GIT_PREFIX)
  unset(ENV{${_var}})
endforeach()

# Stop repository discovery at the parent of the tree being patched, so an
# enclosing repository (e.g. the main project, when the tree lives in its
# build directory) can never hijack path resolution. In CMake script mode,
# CMAKE_BINARY_DIR is the working directory, i.e. the source tree to patch.
get_filename_component(_parent_dir "${CMAKE_BINARY_DIR}/.." ABSOLUTE)
set(ENV{GIT_CEILING_DIRECTORIES} "${_parent_dir}")

# If the patch is already applied (re-run of the patch step on an already
# populated tree), succeed without touching anything: a fully applied patch
# round-trips `git apply --check --reverse`.
execute_process(
  COMMAND "${GIT_EXECUTABLE}" apply --check --reverse --ignore-space-change
          "${PATCH_FILE}"
  RESULT_VARIABLE _already_applied
  OUTPUT_QUIET
  ERROR_QUIET
)
if(_already_applied EQUAL 0)
  message(STATUS "ApplyPatch: already applied, skipping: ${PATCH_FILE}")
  return()
endif()

execute_process(
  COMMAND "${GIT_EXECUTABLE}" apply --verbose --reject --whitespace=fix
          --ignore-space-change "${PATCH_FILE}"
  RESULT_VARIABLE _apply_result
  OUTPUT_VARIABLE _apply_output
  ERROR_VARIABLE _apply_output
)
if(NOT _apply_result EQUAL 0)
  message(
    FATAL_ERROR
      "ApplyPatch: git apply failed (exit ${_apply_result}) for ${PATCH_FILE} "
      "in ${CMAKE_BINARY_DIR}:\n${_apply_output}"
  )
endif()

# `git apply` can exit 0 without applying anything (see header comment), so
# an exit status of 0 is not evidence that the tree was patched. Verify.
execute_process(
  COMMAND "${GIT_EXECUTABLE}" apply --check --reverse --ignore-space-change
          "${PATCH_FILE}"
  RESULT_VARIABLE _verify_result
  OUTPUT_VARIABLE _verify_output
  ERROR_VARIABLE _verify_output
)
if(NOT _verify_result EQUAL 0)
  message(
    FATAL_ERROR
      "ApplyPatch: verification failed — git apply exited 0 but the patch is "
      "not present in the tree at ${CMAKE_BINARY_DIR}. This usually means git "
      "resolved paths against an enclosing repository and silently skipped "
      "every file in the patch.\n"
      "Patch: ${PATCH_FILE}\n"
      "Apply output:\n${_apply_output}\n"
      "Verify output:\n${_verify_output}"
  )
endif()

message(STATUS "ApplyPatch: applied and verified: ${PATCH_FILE}")
