// Copyright 2026 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MUJOCO_SRC_EXPERIMENTAL_PLATFORM_UX_FONTS_H_
#define MUJOCO_SRC_EXPERIMENTAL_PLATFORM_UX_FONTS_H_

#include <cstddef>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace mujoco::platform {

// The Studio UI font files. Every front end (the native window, the web
// viewer's headless UI and the browser client) loads this same set; each
// build stages the files next to its consumer (see studio/CMakeLists.txt
// and studio/web/CMakeLists.txt).
inline constexpr char kMainFontFile[] = "AtkinsonHyperlegibleNext[wght].ttf";
inline constexpr char kIconFontFile[] = "fontawesome-webfont.ttf";
inline constexpr char kMonoFontFile[] = "AtkinsonHyperlegibleMono-Regular.ttf";

// Maps a Studio font filename to its TTF bytes; empty means unavailable.
using FontLoader = std::function<std::vector<std::byte>(std::string_view)>;

// Reads `filename` from `assets_dir`. Returns an empty vector if the file
// cannot be read; fonts are optional and ImGui falls back to its built-in
// font. A ready-made FontLoader body for consumers that keep the fonts in a
// plain directory (e.g. the headless UI's Python-supplied assets dir).
std::vector<std::byte> LoadFontAsset(const std::string& assets_dir,
                                     std::string_view filename);

// A FontLoader that reads "font:<filename>" through the registered MuJoCo
// resource provider (RegisterResourceProviders, or the browser client's
// AssetRegistry-backed provider).
std::vector<std::byte> LoadFontResource(std::string_view filename);

// Adds the Studio fonts to the current ImGui context's atlas at the shared
// sizes (main 16, icons 13 merged into it, mono 14), fetching each file's
// bytes through `load`. Fonts whose bytes come back empty are skipped. The
// atlas owns a copy of the data, so it can rebuild at new DPI scales without
// the loader's buffers staying alive.
void AddStudioFonts(const FontLoader& load);

}  // namespace mujoco::platform

#endif  // MUJOCO_SRC_EXPERIMENTAL_PLATFORM_UX_FONTS_H_
