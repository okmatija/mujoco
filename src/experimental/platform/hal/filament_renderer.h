// Copyright 2025 DeepMind Technologies Limited
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

#ifndef MUJOCO_SRC_EXPERIMENTAL_PLATFORM_HAL_FILAMENT_RENDERER_H_
#define MUJOCO_SRC_EXPERIMENTAL_PLATFORM_HAL_FILAMENT_RENDERER_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include <mujoco/mjrfilament.h>
#include <mujoco/mujoco.h>
#include "experimental/platform/hal/graphics_mode.h"
#include "experimental/platform/hal/renderer.h"
#include "experimental/platform/ux/imgui_bridge.h"
#include "render/filament/mjrfilament_cpp.h"
#include "render/filament/support/model_decorations.h"
#include "render/filament/support/model_lights.h"
#include "render/filament/support/model_objects.h"
#include "render/filament/support/model_renderables.h"

namespace mujoco::platform {

// Renders the mujoco simulation and the imgui state using the Filament-based
// MuJoCo renderer.
//
// For OpenGL, two different Filament configurations are available: normal and
// headless. Normal rendering assumes that we will be rendering to an x11
// window and therefore will use a x11-based context. Headless assumes that we
// will be rendering to a texture and will use an EGL context. Vulkan and WebGL
// have no need for such a distinction. If rendering to a window surface (e.g.
// x11), it requires a pointer to the native window to do so.
class FilamentRenderer : public Renderer {
 public:
  FilamentRenderer(void* native_window, GraphicsMode gfx_mode);
  ~FilamentRenderer();

  FilamentRenderer(const FilamentRenderer&) = delete;
  FilamentRenderer& operator=(const FilamentRenderer&) = delete;

  // Initializes the renderer with the given mjModel.
  void Init(const mjModel* model) override;

  // Renders the simulation and ux state. Renders into `pixels` if provided,
  // otherwise renders to the `native_window` provided at construction.
  void Render(const mjModel* model, mjData* data, const mjvPerturb* perturb,
              mjvCamera* camera, const mjvOption* vis_option, int width,
              int height, std::span<std::byte> pixels = {},
              std::span<mjvGeom> extra_geoms = {}) override;

  // Populates the given output buffer with RGB888 pixel data. The size of the
  // output buffer must be at least width * height * 3.
  void RenderToTexture(const mjModel* model, mjData* data, mjvCamera* camera,
                       int width, int height, std::byte* output) override;

  // Uploads an image to the backend for GUI rendering, returning the texture
  // ID for the texture. The ID can be used in subsequent calls to update the
  // texture data. A nullptr pixels argument will free the texture if it exists.
  // A texture ID of 0 will create a new texture.
  int UploadImage(int texture_id, const std::byte* pixels, int width,
                  int height, int bpp) override;

  // Rendering flags.
  mjtByte* GetRenderFlags() override { return render_flags_; }

  // Confines the main 3D scene to a sub-rectangle of the window, in
  // framebuffer pixels with a bottom-left origin. A zero-sized rect (the
  // default) renders the scene across the full window. The UI pass always
  // covers the full window, so floating UI windows can straddle the boundary.
  void SetSceneViewport(const mjrRect& viewport) { scene_viewport_ = viewport; }

  // Auxiliary render requests (e.g. client viewports rendered into offscreen
  // targets with the mjrf API) submitted before the main scene and the UI in
  // every Render() call until replaced. The requests' scenes and targets must
  // stay valid while set.
  void SetAuxRenderRequests(std::vector<mjrfRenderRequest> requests) {
    aux_requests_ = std::move(requests);
  }

  // The renderer's mjrf context (null until Init) and main scene, for
  // composing additional mjrf objects (scenes, targets, model helpers).
  mjrfContext* GetMjrfContext() const { return filament_context_.get(); }
  mjrfScene* GetMainScene() const { return main_scene_.get(); }

  // Registers an externally-owned texture (e.g. a render target's color
  // texture) under an ImGui texture id, so streamed or local UI can display
  // it with ImGui::Image. Pass nullptr to remove; the caller keeps ownership.
  void SetUiExternalTexture(uintptr_t tex_id, mjrfTexture* texture);

  // Returns the current frame rate.
  double GetFps() override;

 private:
  // Resets the renderer; no rendering will occur until Init() is called again.
  void Deinit();

  void BuildMainRenderRequest(mjrfRenderRequest* request,
                              const mjrRect& viewport, const mjrCamera& camera);
  void BuildUxRenderRequest(mjrfRenderRequest* request,
                            const mjrRect& viewport);

  void* native_window_ = nullptr;
  GraphicsMode gfx_ = GraphicsMode::FilamentVulkan;
  UniquePtr<mjrfContext> filament_context_{nullptr, nullptr};
  UniquePtr<mjrfScene> main_scene_{nullptr, nullptr};
  UniquePtr<mjrfScene> ux_scene_{nullptr, nullptr};
  UniquePtr<mjrfRenderTarget> render_target_{nullptr, nullptr};
  std::unique_ptr<ImguiBridge> imgui_bridge_;
  std::unique_ptr<ModelObjects> model_objects_;
  std::unique_ptr<ModelLights> model_lights_;
  std::unique_ptr<ModelRenderables> model_renderables_;
  std::unique_ptr<ModelDecorations> model_decorations_;
  mjtByte render_flags_[mjNRNDFLAG];
  mjrRect scene_viewport_ = {0, 0, 0, 0};
  std::vector<mjrfRenderRequest> aux_requests_;
  int framebuffer_mode_ = 0;
  double fps_ = 0;
};

}  // namespace mujoco::platform

#endif  // MUJOCO_SRC_EXPERIMENTAL_PLATFORM_HAL_FILAMENT_RENDERER_H_
