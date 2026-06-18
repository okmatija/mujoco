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

#ifndef MUJOCO_SRC_EXPERIMENTAL_STUDIO_COMMAND_PALETTE_H_
#define MUJOCO_SRC_EXPERIMENTAL_STUDIO_COMMAND_PALETTE_H_

#include <functional>
#include <string>
#include <vector>

#include <imgui.h>

namespace mujoco::studio {

// A VS Code-style command palette. The owner opens it (e.g. on Ctrl+Shift+P) and
// passes one flat list of commands to Draw() each frame, fuzzy-searched against
// the input. Commands are namespaced by a leading character in their name -- by
// convention '>' UI actions, '.' model/data fields, '/' agent commands -- so
// typing that character narrows the fuzzy results to one context. Selecting an
// entry runs its callback; '/' entries instead submit their text (see Draw).
//
// This widget is intentionally self-contained and knows nothing about the app:
// commands are just {name, callback} pairs, so it is easy to reuse elsewhere.
class CommandPalette {
 public:
  struct Command {
    std::string name;
    std::function<void()> run;
    // Optional one-line hint shown dimmed after the name in the completion list.
    // The exact strings "on"/"off" are drawn green/red.
    std::string description;
    // Optional: makes the command's value adjustable in place. Once the user has
    // navigated into the list with Up/Down, Left/Right call this with delta -1/+1
    // (and the palette stays open) instead of just moving the text cursor. Use it
    // for commands backed by a value -- e.g. a flag toggle cycles on/off. Commands
    // without it ignore Left/Right.
    std::function<void(int delta)> cycle;
    // Optional: draws an editable widget for the value (checkbox, input, ...) in
    // the value column. When set it supersedes `description` there; the row's
    // selectable allows it to overlap so the widget receives the clicks.
    std::function<void()> draw_value;
  };

  // How the typed text is matched against command names.
  //   kPrefix    -- name starts with the query ("prefix" / classic autocomplete)
  //   kSubstring -- query appears anywhere in the name ("contains")
  //   kFuzzy     -- query characters appear in order with gaps (subsequence)
  enum class SearchMode { kPrefix, kSubstring, kFuzzy };

  // Matching options. No UI toggles these yet; set them programmatically.
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
  // One-shot after Open(): on the frame the input gains focus, move the cursor
  // to the end and clear the selection so the pre-filled ">" isn't select-all'd
  // (and thus wiped by the first keystroke).
  bool init_cursor_end_ = false;
  int selection_ = 0;
  // True once the user has pressed Up/Down to move into the completion list;
  // while set, Left/Right cycle the highlighted command's value (Command::cycle)
  // rather than only editing the query. Reset whenever the query text changes.
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

  // Filters `list` by `query`, runs Up/Down navigation, and draws the rows.
  // Returns the chosen command (clicked, or Enter on the highlighted row) or
  // nullptr. `entered` is the InputText's Enter result for this frame.
  const Command* DrawCompletionList(const std::vector<Command>& list,
                                    const std::string& query, bool entered);
  // Draws the settings panel (search mode, case sensitivity) in the list area.
  void DrawSettings(const std::function<void()>& render_settings);
  // Calls `on_submit_plain`, then clears and
  // refocuses the box (the shared "submit a line" path for ask and '/' modes).
  void SubmitPlain(const std::string& text,
                   const std::function<void(const std::string&)>& on_submit_plain);

  static int InputTextCallback(ImGuiInputTextCallbackData* data);
};

}  // namespace mujoco::studio

#endif  // MUJOCO_SRC_EXPERIMENTAL_STUDIO_COMMAND_PALETTE_H_
