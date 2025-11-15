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

#include <mujoco/mujoco.h>

extern "C" {

MJAPI void mjr_setAux(int index, const mjrContext* con);
MJAPI void mjr_restoreBuffer(const mjrContext* con);
MJAPI void mjr_rectangle(mjrRect viewport, float r, float g, float b, float a);
MJAPI void mjr_blitAux(int index, mjrRect src, int left, int bottom,
                 const mjrContext* con);
MJAPI void mjr_changeFont(int fontscale, mjrContext* con);
MJAPI void mjr_addAux(int index, int width, int height, int samples,
                mjrContext* con);
MJAPI void mjr_resizeOffscreen(int width, int height, mjrContext* con);
MJAPI void mjr_drawPixels(const unsigned char* rgb, const float* depth,
                    mjrRect viewport, const mjrContext* con);
MJAPI void mjr_blitBuffer(mjrRect src, mjrRect dst, int flg_color, int flg_depth,
                    const mjrContext* con);
MJAPI void mjr_text(int font, const char* txt, const mjrContext* con, float x,
              float y, float r, float g, float b);
MJAPI void mjr_overlay(int font, int gridpos, mjrRect viewport, const char* overlay,
                 const char* overlay2, const mjrContext* con);
MJAPI void mjr_label(mjrRect viewport, int font, const char* txt, float r, float g,
               float b, float a, float rt, float gt, float bt,
               const mjrContext* con);
MJAPI void mjr_figure(mjrRect viewport, mjvFigure* fig, const mjrContext* con);
MJAPI void mjr_finish();
MJAPI int mjr_getError();
MJAPI mjrRect mjr_maxViewport(const mjrContext* con);
MJAPI int mjr_findRect(int x, int y, int nrect, const mjrRect* rect);

}  // extern "C"
