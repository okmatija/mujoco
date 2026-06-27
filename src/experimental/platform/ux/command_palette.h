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

// A VS Code-style command palette, written as a flat C API in the spirit of
// Dear ImGui (and of cimgui / dear_bindings): no C++ in the public surface, so
// it can be driven from C and bound to other languages (Jai, Odin, ...) by
// translating this header directly -- no generator required.
//
// The whole interface is plain data and function pointers:
//   * opaque handles            (ImCmdPalette, ImCmdList),
//   * a POD command record       (ImCmd),
//   * callbacks as fn-pointer + void* user_data (never std::function),
//   * strings as const char*     (the list copies what it stores),
//   * value bytes by pointer     (ImCmd::user / the field *ptr),
//   * no <imgui.h> dependency here (it is an implementation detail of the .cc;
//     callbacks that draw value widgets call ImGui from the host side).
//
// Usage each frame (host pseudocode):
//   ImCmdList_Clear(list);
//   ImCmd cmd = { ">Reset", "Reset the sim", 0, OnReset, 0,0,0, host, false };
//   ImCmdList_Add(list, &cmd);
//   ImCmdList_AddFieldF64(list, "mjModel.opt.timestep", &m->opt.timestep, 2e-3);
//   ImCmdPalette_Draw(pal, list, &desc);   // runs callbacks, draws if open
//
// Commands are namespaced by a leading character in their name -- by convention
// '>' UI actions, '.' / dotted model-data fields, '/' agent commands -- so
// typing that character narrows the fuzzy results to one context. Choosing a '/'
// entry submits its text (ImCmdDrawDesc::on_submit_plain); any other entry runs
// its `run` callback and closes the palette.

#ifndef MUJOCO_SRC_EXPERIMENTAL_PLATFORM_UX_COMMAND_PALETTE_H_
#define MUJOCO_SRC_EXPERIMENTAL_PLATFORM_UX_COMMAND_PALETTE_H_

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ----------------------------------------------------------------------------
// Plain-old-data types
// ----------------------------------------------------------------------------

// 2D vector / axis-aligned rectangle. Deliberately distinct from ImGui's ImVec2
// / ImVec4 so this header never has to include a C++ header.
typedef struct ImCmdVec2 { float x, y; } ImCmdVec2;
typedef struct ImCmdRect { float x, y, w, h; } ImCmdRect;

// How the typed text is matched against command names.
typedef enum ImCmdSearchMode {
  ImCmdSearchMode_Prefix,     // name starts with the query (classic autocomplete)
  ImCmdSearchMode_Substring,  // query appears anywhere in the name ("contains")
  ImCmdSearchMode_Fuzzy,      // query chars appear in order with gaps (subsequence)
} ImCmdSearchMode;

// Callback signatures. Every callback takes the command's `user` pointer (or, for
// the draw-config callbacks, the matching *_user pointer) as its first argument,
// the C analogue of a captured `this`.
typedef void (*ImCmdRunCallback)(void* user);                  // entry chosen
typedef void (*ImCmdCycleCallback)(void* user, int delta);     // Left/Right step
typedef void (*ImCmdValueCallback)(void* user);                // draw value widget
typedef void (*ImCmdResetCallback)(void* user);                // restore default
typedef void (*ImCmdRenderCallback)(void* user);               // host-drawn panel
typedef void (*ImCmdSubmitCallback)(void* user, const char* text);  // '/' submit
typedef bool (*ImCmdGetBoolCallback)(void* user);              // read a flag
typedef void (*ImCmdSetBoolCallback)(void* user, bool value);  // write a flag

// One palette entry. The host fills this for ImCmdList_Add; the field helpers
// below fill it for you. Strings are borrowed only for the duration of the Add
// call (the list copies them); `user` and any pointer it carries must stay valid
// until the next ImCmdPalette_Draw returns. All callbacks and the two trailing
// strings are optional (pass NULL / 0).
typedef struct ImCmd {
  const char* name;          // namespaced by a leading '>' '.' '/'; required
  const char* description;   // dimmed one-line hint (entries with no value widget)
  const char* default_text;  // shown on the revert tooltip's second line
  ImCmdRunCallback run;          // chosen via Enter / click
  ImCmdCycleCallback cycle;      // Left/Right adjust in place (flags, enums)
  ImCmdValueCallback draw_value; // draws an editable value widget in the column
  ImCmdResetCallback reset;      // restores the default (revert button)
  void* user;                    // passed to every callback above
  bool modified;                 // value differs from default (shows '*' + revert)
} ImCmd;

// Per-frame inputs to ImCmdPalette_Draw that are not the command list. Zero-
// initialize and set what you need; every callback is optional.
typedef struct ImCmdDrawDesc {
  ImCmdRect rect;  // host area: palette is centered on x..x+w, top edge at y

  // Drawn inside the palette window (e.g. an agent conversation) while in agent
  // context -- when the input starts with '/' or is empty.
  ImCmdRenderCallback render_below;
  void* render_below_user;

  // Receives the text of a chosen / typed '/' command so the host can route it.
  ImCmdSubmitCallback on_submit_plain;
  void* on_submit_plain_user;

  // Drawn inside the cog settings panel, after the palette's own options, so the
  // host (and its plugins) can add settings there.
  ImCmdRenderCallback render_settings;
  void* render_settings_user;
} ImCmdDrawDesc;

// Opaque handles. Create with the *_Create functions, free with *_Destroy.
typedef struct ImCmdPalette ImCmdPalette;  // persistent palette state
typedef struct ImCmdList ImCmdList;        // the per-frame command list (an arena)

// ----------------------------------------------------------------------------
// Palette
// ----------------------------------------------------------------------------

ImCmdPalette* ImCmdPalette_Create(void);
void ImCmdPalette_Destroy(ImCmdPalette* p);

void ImCmdPalette_Open(ImCmdPalette* p);
// Opens pre-filled with `text` (e.g. ">Physics"), cursor parked at the end.
void ImCmdPalette_OpenWith(ImCmdPalette* p, const char* text);
// Replaces the input text without opening/closing (e.g. to "type" a question).
void ImCmdPalette_SetText(ImCmdPalette* p, const char* text);
void ImCmdPalette_Close(ImCmdPalette* p);
void ImCmdPalette_Toggle(ImCmdPalette* p);
bool ImCmdPalette_IsOpen(const ImCmdPalette* p);
// Center of the palette window from the last Draw (for an external cursor).
ImCmdVec2 ImCmdPalette_WindowCenter(const ImCmdPalette* p);

// Matching options (also exposed as controls in the cog settings panel).
void ImCmdPalette_SetSearchMode(ImCmdPalette* p, ImCmdSearchMode mode);
ImCmdSearchMode ImCmdPalette_GetSearchMode(const ImCmdPalette* p);
void ImCmdPalette_SetCaseInsensitive(ImCmdPalette* p, bool enabled);
bool ImCmdPalette_GetCaseInsensitive(const ImCmdPalette* p);

// Draws the palette (if open) and drives it: filters `list` against the typed
// text, runs keyboard / mouse navigation, and when an entry is chosen either
// submits it (names starting with '/') via desc->on_submit_plain or runs its
// `run` callback and closes. `desc` must be non-NULL; `list` may be NULL/empty.
void ImCmdPalette_Draw(ImCmdPalette* p, const ImCmdList* list,
                       const ImCmdDrawDesc* desc);

// ----------------------------------------------------------------------------
// Command list (rebuilt every frame)
// ----------------------------------------------------------------------------

ImCmdList* ImCmdList_Create(void);
void ImCmdList_Destroy(ImCmdList* list);
// Drops every command and reclaims the copied strings / field bindings for
// reuse. Call once at the start of each rebuild.
void ImCmdList_Clear(ImCmdList* list);

// Appends one fully-formed command. `name`, `description` and `default_text` are
// copied into the list, so they may be temporaries; the callbacks and `user`
// are stored as-is.
void ImCmdList_Add(ImCmdList* list, const ImCmd* cmd);

// Convenience builders for editable fields. They render the value widget, the
// '*'-when-modified marker, and the revert-to-default button for you. `path` is
// copied and `def` is stored by value; `ptr` must stay valid through the next
// Draw (a pointer into live state is fine, and lets '*' track external changes).
void ImCmdList_AddFieldS32(ImCmdList* list, const char* path, int* ptr, int def);
void ImCmdList_AddFieldF32(ImCmdList* list, const char* path, float* ptr,
                           float def);
void ImCmdList_AddFieldF64(ImCmdList* list, const char* path, double* ptr,
                           double def);
// N-element variants (an N-wide numeric input bound to ptr[0..count)).
void ImCmdList_AddFieldArrayS32(ImCmdList* list, const char* path, int* ptr,
                                int count, const int* def);
void ImCmdList_AddFieldArrayF32(ImCmdList* list, const char* path, float* ptr,
                                int count, const float* def);
void ImCmdList_AddFieldArrayF64(ImCmdList* list, const char* path, double* ptr,
                                int count, const double* def);

// A boolean field reached through a getter/setter (the general case -- it covers
// bitfields and unsigned-char flags, not just a bool). Enter / Left-Right toggle.
void ImCmdList_AddFieldFlag(ImCmdList* list, const char* path,
                            ImCmdGetBoolCallback get, ImCmdSetBoolCallback set,
                            void* user, bool def);
// Convenience overload for a flag that really is a `bool` in memory.
void ImCmdList_AddFieldFlagPtr(ImCmdList* list, const char* path, bool* ptr,
                               bool def);

// An int-backed enum: a combo of `names` (count entries) in the value column,
// Enter advances / Left-Right cycle. `names` (the array and its strings) is
// copied. `ptr` may point at any int-sized enum.
void ImCmdList_AddFieldEnum(ImCmdList* list, const char* path, int* ptr,
                            const char* const* names, int count, int def);

// A field that is not a plain pointer (reached only through an accessor): supply
// the value widget and the revert action, the modified state, and the default
// shown in the revert tooltip. `draw` runs inside a stable ImGui ID scope keyed
// on `path`, so give its widgets a constant label (e.g. "##v").
void ImCmdList_AddFieldCustom(ImCmdList* list, const char* path,
                              ImCmdValueCallback draw, ImCmdResetCallback reset,
                              void* user, bool modified, const char* default_text);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // MUJOCO_SRC_EXPERIMENTAL_PLATFORM_UX_COMMAND_PALETTE_H_
