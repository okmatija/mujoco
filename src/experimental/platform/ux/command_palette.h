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

#ifndef MUJOCO_SRC_EXPERIMENTAL_PLATFORM_UX_COMMAND_PALETTE_H_
#define MUJOCO_SRC_EXPERIMENTAL_PLATFORM_UX_COMMAND_PALETTE_H_

#include <cstddef>
#include <cstdio>
#include <functional>
#include <string>
#include <type_traits>
#include <vector>

#include <imgui.h>

namespace mujoco::platform {

// A VS Code-style command palette. The owner opens it (e.g. on Ctrl+Shift+P) and
// passes one flat list of commands to Draw() each frame, fuzzy-searched against
// the input. Commands are namespaced by a leading character in their name -- by
// convention '>' UI actions, '.' model/data fields, '/' agent commands -- so
// typing that character narrows the fuzzy results to one context. Selecting an
// entry runs its callback; '/' entries instead submit their text (see Draw).
//
// This widget is intentionally self-contained and knows nothing about the app:
// a command is a {name, action, optional value-widget} record (see Command), so
// it is easy to reuse elsewhere.
class CommandPalette {
 public:
  struct Command {
    std::string name;
    // Runs when the entry is chosen (Enter / click) -- for '>' and '.' entries.
    std::function<void()> run;
    // Optional dimmed one-line hint shown in the value column, for entries that
    // have no value widget (e.g. '>' UI actions, '/' agent commands).
    std::string description;
    // Optional: makes the command's value adjustable in place. Once the user has
    // navigated into the list with Up/Down, Left/Right call this with delta -1/+1
    // (and the palette stays open) instead of just moving the text cursor. Use it
    // for stepped values -- a flag toggles, an enum advances. Entries without it
    // ignore Left/Right (numeric inputs are typed instead; see `draw_value`).
    std::function<void(int delta)> cycle;
    // Optional: draws an editable widget for the value (checkbox, input, ...) in
    // the value column. When set it supersedes `description` there; the row's
    // selectable allows it to overlap so the widget receives the clicks.
    std::function<void()> draw_value;
    // Optional: restores the value to its default. When set (and `modified`),
    // a revert button is shown in a column on the right.
    std::function<void()> reset;
    // Whether the value differs from its default (gates the revert button).
    bool modified = false;
    // The default value as text, shown on a second line of the revert tooltip.
    std::string default_text;
  };

  // How the typed text is matched against command names.
  //   kPrefix    -- name starts with the query ("prefix" / classic autocomplete)
  //   kSubstring -- query appears anywhere in the name ("contains")
  //   kFuzzy     -- query characters appear in order with gaps (subsequence)
  enum class SearchMode { kPrefix, kSubstring, kFuzzy };

  // Matching options (also exposed as controls in the cog settings panel).
  void set_search_mode(SearchMode mode) { search_mode_ = mode; }
  SearchMode search_mode() const { return search_mode_; }
  void set_case_insensitive(bool on) { case_insensitive_ = on; }
  bool case_insensitive() const { return case_insensitive_; }

  void Open();
  // Opens the palette pre-filled with `text` (e.g. ">Physics"); used by the
  // capture script to show command-mode interactions.
  void OpenWith(const std::string& text);
  // Replaces the input text without opening/closing (used by the capture script
  // to "type" a question one character at a time).
  void SetText(const std::string& text);
  void Close();
  void Toggle();
  bool is_open() const { return open_; }
  // Center of the palette window from the last Draw (for the capture cursor).
  ImVec2 window_center() const { return center_; }

  // Draws the palette (if open), horizontally centered near the top of `rect`
  // (x, y, width, height), capped at 80% of the viewport height (the list
  // scrolls past that). `commands` is the single list fuzzy-matched against the
  // whole input. Choosing an entry whose name starts with '/' submits its text
  // via `on_submit_plain` (so the agent / app can route it); any other entry
  // runs its `run` callback and closes.
  //
  // Typed text that matches no entry but starts with '/' is also submitted as-is
  // on Enter (so an argument-bearing command like "/model sonnet" or
  // "/prompt how do I..." works). `render_below` is invoked inside the palette
  // window -- to draw the agent conversation -- while in agent context (the
  // input starts with '/' or is empty). `render_settings` is invoked inside the
  // cog settings panel, after the palette's own options, so the host (and its
  // plugins) can add their own settings there. All callbacks are optional.
  void Draw(const std::vector<Command>& commands, const ImVec4& rect,
            const std::function<void()>& render_below = {},
            const std::function<void(const std::string&)>& on_submit_plain = {},
            const std::function<void()>& render_settings = {});

 private:
  bool open_ = false;
  bool focus_input_ = false;
  // One-shot after Open()/OpenWith(): on the frame the input gains focus, move
  // the cursor to the end and clear the selection so a pre-filled string (from
  // OpenWith) isn't select-all'd and wiped by the first keystroke.
  bool init_cursor_end_ = false;
  int selection_ = 0;
  // True once the user has pressed Up/Down to move into the completion list;
  // while set, Left/Right act on the highlighted command's value (cycle it, or
  // Right focuses a numeric input) instead of moving the query text cursor.
  // Reset whenever the query text changes.
  bool in_list_ = false;
  // One-shot: Right on a selected value-input row gives that widget keyboard
  // focus next render (so the value can be typed).
  bool focus_value_ = false;
  std::string last_query_;  // query from the previous Draw, to detect edits.
  char input_[256] = "";
  ImVec2 center_{0.0f, 0.0f};
  // Points at the command list during Draw so the Tab-completion callback can
  // see it (the callback runs inside InputText, before the list is filtered).
  const std::vector<Command>* completion_list_ = nullptr;
  // Matching options (default: fuzzy, case-insensitive).
  SearchMode search_mode_ = SearchMode::kFuzzy;
  bool case_insensitive_ = true;
  bool highlight_matches_ = true;  // bold the matched characters in the list.
  // When set (toggled by the cog button), the settings panel replaces the
  // completion list.
  bool show_settings_ = false;

  // Filters `list` by `query` (current search mode), runs Up/Down/Left/Right
  // navigation, and draws the rows. Returns the command chosen by Enter on the
  // selected row, or nullptr -- clicks move the list focus or edit values, they
  // don't choose. `entered` is the InputText's Enter result for this frame.
  const Command* DrawCompletionList(const std::vector<Command>& list,
                                    const std::string& query, bool entered);
  // Draws the settings panel (search mode, case sensitivity, match
  // highlighting), then the host's `render_settings`, in the list area.
  void DrawSettings(const std::function<void()>& render_settings);
  // Calls `on_submit_plain`, then clears and refocuses the box (the shared
  // "submit a line" path for '/' agent commands).
  void SubmitPlain(const std::string& text,
                   const std::function<void(const std::string&)>& on_submit_plain);

  static int InputTextCallback(ImGuiInputTextCallbackData* data);
};

// Free functions that append CommandPalette::Command records for editable
// fields, so any subsystem can surface its members in the palette's dotted-path
// field list without hand-writing the value widget, the '*'-when-modified
// marker, or the revert-to-default button. They build Commands from raw pointers
// (or a getter/setter pair) only, so they stay independent of any particular app
// -- the palette merely consumes the Commands. The command list is the first
// argument (these are bound-argument helpers, not state, so there is no class).
//
// `path` is the dotted code path of the field (e.g. "mjModel.opt.gravity"), so
// fuzzy/prefix matching on the dots reads like navigating the struct. Each call
// appends one command to `out`. A bound pointer must outlive the drawn command;
// since the command list is typically rebuilt every frame, a pointer into live
// state is fine (and lets '*' track changes made elsewhere).

namespace command_palette_detail {

// Appends "*" to a modified field's name so the change shows and typing "*"
// filters to all changed fields. The value widget's id (###path) stays unmarked,
// so editing isn't disrupted when the marker appears/disappears.
inline std::string Marked(const std::string& path, bool modified) {
  return modified ? path + "*" : path;
}
// Formats a default for the revert tooltip ("%g" for floats, plain for ints).
template <class T>
std::string NumStr(T v) {
  if constexpr (std::is_integral_v<T>) {
    return std::to_string(v);
  } else {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%g", static_cast<double>(v));
    return std::string(buf);
  }
}
template <class T>
constexpr ImGuiDataType DataTypeOf() {
  if constexpr (std::is_same_v<T, int>) {
    return ImGuiDataType_S32;
  } else if constexpr (std::is_same_v<T, float>) {
    return ImGuiDataType_Float;
  } else {
    return ImGuiDataType_Double;
  }
}

}  // namespace command_palette_detail

// A boolean field: a checkbox value, Enter / Left-Right toggling, and a revert
// button when the value differs from `dflt`.
void RegisterFlagField(std::vector<CommandPalette::Command>& out,
                       const std::string& path, std::function<bool()> get,
                       std::function<void(bool)> set, bool dflt);

// A scalar field (T = int / float / double): a numeric input bound to *ptr,
// revert to `def`.
template <class T>
void RegisterScalarField(std::vector<CommandPalette::Command>& out,
                         const std::string& path, T* ptr, T def) {
  const ImGuiDataType dt = command_palette_detail::DataTypeOf<T>();
  const bool modified = *ptr != def;
  const std::string id = "###" + path;
  auto draw = [id, ptr, dt] { ImGui::InputScalar(id.c_str(), dt, ptr); };
  auto reset = [ptr, def] { *ptr = def; };
  out.push_back({command_palette_detail::Marked(path, modified), {}, "", {},
                 draw, reset, modified, command_palette_detail::NumStr(def)});
}

// An n-element array field (T = int / float / double): an N-input bound to
// ptr[0..n), revert to def[0..n).
template <class T>
void RegisterArrayField(std::vector<CommandPalette::Command>& out,
                        const std::string& path, T* ptr, int n, const T* def) {
  const ImGuiDataType dt = command_palette_detail::DataTypeOf<T>();
  // Copy the defaults by value: callers often pass a pointer into a per-frame
  // local (e.g. mj_defaultOption's output), but reset runs later, on a click.
  const std::vector<T> defs(def, def + n);
  bool modified = false;
  std::string def_text;
  for (int k = 0; k < n; ++k) {
    modified |= (ptr[k] != defs[k]);
    def_text += (k ? ", " : "") + command_palette_detail::NumStr(defs[k]);
  }
  const std::string id = "###" + path;
  auto draw = [id, ptr, dt, n] { ImGui::InputScalarN(id.c_str(), dt, ptr, n); };
  auto reset = [ptr, defs] {
    for (std::size_t k = 0; k < defs.size(); ++k) ptr[k] = defs[k];
  };
  out.push_back({command_palette_detail::Marked(path, modified), {}, "", {},
                 draw, reset, modified, def_text});
}

// An enum field (T = an int-backed enum or int): a combo of `names` in the value
// column, Enter advances / Left-Right cycles, revert to `def`.
template <class T>
void RegisterEnumField(std::vector<CommandPalette::Command>& out,
                       const std::string& path, T* ptr,
                       std::vector<const char*> names, T def) {
  const int n = static_cast<int>(names.size());
  const int cur = static_cast<int>(*ptr);
  const int def_i = static_cast<int>(def);
  auto cyc = [ptr, n](int delta) {
    const int v = static_cast<int>(*ptr);
    *ptr = static_cast<T>(((v + delta) % n + n) % n);
  };
  const std::string id = "###" + path;
  auto combo = [id, ptr, names, n] {
    int v = static_cast<int>(*ptr);
    if (ImGui::Combo(id.c_str(), &v, names.data(), n)) {
      *ptr = static_cast<T>(v);
    }
  };
  auto reset = [ptr, def] { *ptr = def; };
  const std::string def_text =
      (def_i >= 0 && def_i < n) ? names[def_i] : std::string();
  out.push_back({command_palette_detail::Marked(path, cur != def_i),
                 [cyc] { cyc(1); }, "", cyc, combo, reset, cur != def_i,
                 def_text});
}

// A field that can't be expressed as a plain pointer (e.g. one reached only
// through an accessor pair): supply the value widget and the revert action, plus
// the modified state and the default shown in the revert tooltip.
void RegisterCustomField(std::vector<CommandPalette::Command>& out,
                         const std::string& path, std::function<void()> draw,
                         std::function<void()> reset, bool modified,
                         std::string default_text);

}  // namespace mujoco::platform

#endif  // MUJOCO_SRC_EXPERIMENTAL_PLATFORM_UX_COMMAND_PALETTE_H_
