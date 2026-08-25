// Copyright 2026 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MUJOCO_SRC_RENDER_FILAMENT_MJRFILAMENT_CPP_H_
#define MUJOCO_SRC_RENDER_FILAMENT_MJRFILAMENT_CPP_H_

#include <functional>
#include <memory>
#include <string>

#include <mujoco/mjrfilament.h>

namespace mujoco {

// A unique pointer to a mujoco object.
template <typename T>
using UniquePtr = std::unique_ptr<T, void (*)(T*)>;

inline UniquePtr<mjrfContext> CreateContext(const mjrfContextConfig& config) {
  mjrfContext* context = mjrf_createContext(&config);
  return UniquePtr<mjrfContext>(context, mjrf_destroyContext);
}

inline UniquePtr<mjrfTexture> CreateTexture(mjrfContext* ctx,
                                            const mjrfTextureConfig& config) {
  mjrfTexture* texture = mjrf_createTexture(ctx, &config);
  return UniquePtr<mjrfTexture>(texture, mjrf_destroyTexture);
}

inline UniquePtr<mjrfMesh> CreateMesh(mjrfContext* ctx,
                                      const mjrfMeshData& data) {
  mjrfMesh* mesh = mjrf_createMesh(ctx, &data);
  return UniquePtr<mjrfMesh>(mesh, mjrf_destroyMesh);
}

inline UniquePtr<mjrfScene> CreateScene(mjrfContext* ctx,
                                        const mjrfSceneParams& params) {
  mjrfScene* scene = mjrf_createScene(ctx, &params);
  return UniquePtr<mjrfScene>(scene, mjrf_destroyScene);
}

inline UniquePtr<mjrfLight> CreateLight(mjrfContext* ctx,
                                        const mjrfLightParams& params) {
  mjrfLight* light = mjrf_createLight(ctx, &params);
  return UniquePtr<mjrfLight>(light, mjrf_destroyLight);
}

inline UniquePtr<mjrfRenderable> CreateRenderable(
    mjrfContext* ctx, const mjrfRenderableParams& params) {
  mjrfRenderable* renderable = mjrf_createRenderable(ctx, &params);
  return UniquePtr<mjrfRenderable>(renderable, mjrf_destroyRenderable);
}

inline UniquePtr<mjrfRenderTarget> CreateRenderTarget(
    mjrfContext* ctx, const mjrfRenderTargetConfig& config) {
  mjrfRenderTarget* render_target = mjrf_createRenderTarget(ctx, &config);
  return UniquePtr<mjrfRenderTarget>(render_target, mjrf_destroyRenderTarget);
}

std::string ResolveFilamentAssetPath(const std::string& filename);

// Visits each option of the experimental Filament view editor (the data shown in
// the "Plugins -> Filament" window) as a flat list, so a host UI such as the
// command palette can surface the options individually rather than as one
// monolithic window. For every option `emit` is called with a hierarchical
// `path` ("Section > Subsection > Option"), a `draw` closure that renders just
// that option's value widget (and applies edits in place), an optional `reset`
// closure that reverts to the captured initial value, and whether the value is
// currently modified. Safe to call for any scene; does nothing if `scene` is
// null or not a Filament scene.
struct FilamentEditorEntrySink {
  std::function<void(const std::string& path, std::function<void()> draw,
                     std::function<void()> reset, bool modified)>
      emit;
};
void VisitFilamentEditorEntries(mjrfScene* scene,
                                const FilamentEditorEntrySink& sink);

}  // namespace mujoco

#endif  // MUJOCO_SRC_RENDER_FILAMENT_MJRFILAMENT_CPP_H_
