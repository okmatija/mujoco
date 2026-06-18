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

#include "experimental/studio/command_palette.h"

#include <algorithm>
#include <cctype>
#include <cfloat>
#include <cstddef>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include <imgui.h>

namespace mujoco::studio {
namespace {

// Fuzzy (subsequence) match, case-insensitive: returns true if every character
// of `query` appears in `text` in order, not necessarily contiguously (so "rndsh"
// matches "Render flag: Shadow"). When it matches, *score is set so that better
// matches rank higher: contiguous runs and matches right after a word boundary
// (space, ':', '/', etc.) score more, and matches deep in the string are
// penalized. An empty query matches everything with score 0 (preserving the
// plain alphabetical list shown before the user types).
bool FuzzyMatch(const std::string& text, const std::string& query, int* score) {
  *score = 0;
  if (query.empty()) {
    return true;
  }
  auto lower = [](char c) {
    return static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  };

  constexpr int kAdjacencyBonus = 5;       // match right after another match.
  constexpr int kSeparatorBonus = 10;      // match at a word boundary.
  constexpr int kLeadingPenalty = -3;      // per char before the first match...
  constexpr int kMaxLeadingPenalty = -9;   // ...capped here.
  constexpr int kUnmatchedPenalty = -1;    // per non-matching char in text.

  int total = 0;
  std::size_t qi = 0;
  bool prev_matched = false;
  bool prev_separator = true;  // treat the start of the string as a boundary.
  int first_match = -1;

  for (std::size_t ti = 0; ti < text.size(); ++ti) {
    const bool is_separator = !std::isalnum(static_cast<unsigned char>(text[ti]));
    if (qi < query.size() && lower(text[ti]) == lower(query[qi])) {
      if (first_match < 0) {
        first_match = static_cast<int>(ti);
      }
      if (prev_matched) total += kAdjacencyBonus;
      if (prev_separator) total += kSeparatorBonus;
      ++qi;
      prev_matched = true;
    } else {
      total += kUnmatchedPenalty;
      prev_matched = false;
    }
    prev_separator = is_separator;
  }

  if (qi != query.size()) {
    return false;  // not every query character was consumed, in order.
  }
  total += std::max(kMaxLeadingPenalty, kLeadingPenalty * first_match);
  *score = total;
  return true;
}

// Draws one completion row: a full-width selectable whose label is the name,
// then (if the value differs from default) a '*' at `marker_x`, then the
// description at `desc_x` -- both window-local x's so they line up in columns.
// The description is dimmed, except the literal "on"/"off" which are green/red.
// Returns true if the row was clicked.
bool CompletionRow(const CommandPalette::Command& cmd, bool selected,
                   float marker_x, float desc_x) {
  const bool clicked = ImGui::Selectable(cmd.name.c_str(), selected);
  if (cmd.modified) {
    ImGui::SameLine(marker_x);
    ImGui::TextUnformatted("*");
  }
  if (!cmd.description.empty()) {
    ImGui::SameLine(desc_x);
    if (cmd.description == "on") {
      ImGui::TextColored(ImVec4(0.36f, 0.80f, 0.36f, 1.0f), "on");
    } else if (cmd.description == "off") {
      ImGui::TextColored(ImVec4(0.90f, 0.40f, 0.40f, 1.0f), "off");
    } else {
      ImGui::TextDisabled("%s", cmd.description.c_str());
    }
  }
  return clicked;
}

}  // namespace

int CommandPalette::InputTextCallback(ImGuiInputTextCallbackData* data) {
  auto* self = static_cast<CommandPalette*>(data->UserData);

  // CallbackAlways: one-shot after Open() to undo the select-all that focus
  // applies, so the pre-filled ">" stays put and the cursor sits after it. This
  // fires on the activation frame, after InputText's select-all, so it wins.
  // (Up/Down/Left/Right are read via IsKeyPressed in DrawCompletionList, so the
  // single-line input never needs to handle them itself.)
  if (data->EventFlag == ImGuiInputTextFlags_CallbackAlways) {
    if (self->init_cursor_end_) {
      data->CursorPos = data->BufTextLen;
      data->SelectionStart = data->SelectionEnd = data->CursorPos;
      self->init_cursor_end_ = false;
    }
  }
  return 0;
}

void CommandPalette::Open() {
  open_ = true;
  focus_input_ = true;
  selection_ = 0;
  in_list_ = false;
  last_query_.clear();
  // Start in '>' command mode so the tool/command list is shown by default; the
  // cursor is parked after the '>' (see init_cursor_end_) so typing filters it.
  input_[0] = '>';
  input_[1] = '\0';
  init_cursor_end_ = true;
}

void CommandPalette::OpenWith(const std::string& text) {
  Open();
  const std::size_t n = std::min(text.size(), sizeof(input_) - 1);
  std::memcpy(input_, text.data(), n);
  input_[n] = '\0';
}

void CommandPalette::SetText(const std::string& text) {
  const std::size_t n = std::min(text.size(), sizeof(input_) - 1);
  std::memcpy(input_, text.data(), n);
  input_[n] = '\0';
}

void CommandPalette::Close() { open_ = false; }

void CommandPalette::Toggle() {
  if (open_) {
    Close();
  } else {
    Open();
  }
}

const CommandPalette::Command* CommandPalette::DrawCompletionList(
    const std::vector<Command>& list, const std::string& query, bool entered) {
  // Fuzzy-match the name against the query, keeping each match's score.
  std::vector<std::pair<const Command*, int>> matches;
  for (const Command& command : list) {
    int score = 0;
    if (FuzzyMatch(command.name, query, &score)) {
      matches.push_back({&command, score});
    }
  }

  // Best fuzzy score first; ties broken alphabetically (case-insensitive). With
  // an empty query every score is 0, so this is the plain alphabetical list.
  std::sort(matches.begin(), matches.end(),
            [](const std::pair<const Command*, int>& a,
               const std::pair<const Command*, int>& b) {
              if (a.second != b.second) {
                return a.second > b.second;
              }
              return std::lexicographical_compare(
                  a.first->name.begin(), a.first->name.end(),
                  b.first->name.begin(), b.first->name.end(),
                  [](unsigned char x, unsigned char y) {
                    return std::tolower(x) < std::tolower(y);
                  });
            });

  // Editing the query drops out of list-navigation mode (so Left/Right go back
  // to moving the text cursor) and re-anchors the selection at the top match.
  const bool query_changed = (query != last_query_);
  if (query_changed) {
    in_list_ = false;
    selection_ = 0;
    last_query_ = query;
  }

  // Keyboard navigation through the filtered list. Up/Down move the selection
  // and enter "list mode"; `moved` keeps the highlighted row scrolled into view.
  bool moved = query_changed;
  if (ImGui::IsKeyPressed(ImGuiKey_DownArrow)) {
    ++selection_;
    in_list_ = true;
    moved = true;
  }
  if (ImGui::IsKeyPressed(ImGuiKey_UpArrow)) {
    --selection_;
    in_list_ = true;
    moved = true;
  }
  if (!matches.empty()) {
    const int n = static_cast<int>(matches.size());
    selection_ = (selection_ % n + n) % n;
  } else {
    selection_ = 0;
  }

  // In list mode, Left/Right cycle the highlighted command's value in place
  // (e.g. toggle a flag on/off) without running it or closing the palette. The
  // description, rebuilt by the caller each frame, reflects the new value.
  if (in_list_ && !matches.empty()) {
    int delta = 0;
    if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow)) {
      delta = -1;
    } else if (ImGui::IsKeyPressed(ImGuiKey_RightArrow)) {
      delta = 1;
    }
    if (delta != 0 && matches[selection_].first->cycle) {
      matches[selection_].first->cycle(delta);
    }
  }

  if (matches.empty()) {
    return nullptr;  // nothing to show (e.g. mid-typing "/prompt question").
  }

  // Column layout: '*' marker just past the widest name, descriptions past that.
  const float spacing = ImGui::GetStyle().ItemSpacing.x;
  float name_w = 0.0f;
  for (const auto& [command, score] : matches) {
    name_w = std::max(name_w, ImGui::CalcTextSize(command->name.c_str()).x);
  }
  const float marker_x = name_w + spacing * 2.0f;
  const float desc_x = marker_x + ImGui::CalcTextSize("*").x + spacing * 2.0f;

  ImGui::Separator();

  // The list scrolls inside a child so the whole window stays under the 80% cap
  // set in Draw(); leave room for the input row above.
  const float max_h = ImGui::GetMainViewport()->WorkSize.y * 0.8f -
                      ImGui::GetFrameHeightWithSpacing() * 2.0f;
  ImGui::SetNextWindowSizeConstraints(ImVec2(0, 0), ImVec2(FLT_MAX, max_h));
  const Command* chosen = nullptr;
  if (ImGui::BeginChild("##completions", ImVec2(0, 0),
                        ImGuiChildFlags_AutoResizeY)) {
    for (int i = 0; i < static_cast<int>(matches.size()); ++i) {
      const Command* command = matches[i].first;
      const bool is_selected = (i == selection_);
      if (CompletionRow(*command, is_selected, marker_x, desc_x) ||
          (is_selected && entered)) {
        chosen = command;
      }
      if (is_selected && moved) {
        ImGui::SetScrollHereY();
      }
    }
  }
  ImGui::EndChild();
  return chosen;
}

void CommandPalette::SubmitPlain(
    const std::string& text,
    const std::function<void(const std::string&)>& on_submit_plain) {
  if (text.empty()) {
    return;
  }
  if (on_submit_plain) {
    on_submit_plain(text);
  }
  input_[0] = '\0';
  focus_input_ = true;
}

void CommandPalette::Draw(
    const std::vector<Command>& commands, const ImVec4& rect,
    const std::function<void()>& render_below,
    const std::function<void(const std::string&)>& on_submit_plain) {
  if (!open_) {
    return;
  }

  constexpr float kWidth = 480.0f;
  // Centered horizontally; the caller supplies the top edge (rect.y).
  ImGui::SetNextWindowPos(ImVec2(rect.x + rect.z * 0.5f, rect.y),
                          ImGuiCond_Always, ImVec2(0.5f, 0.0f));
  // Fixed width; height grows with the list but is capped at 80% of the viewport
  // (the completion list scrolls within its own child past that).
  const float max_h = ImGui::GetMainViewport()->WorkSize.y * 0.8f;
  ImGui::SetNextWindowSizeConstraints(ImVec2(kWidth, 0), ImVec2(kWidth, max_h));
  // Translucent, matching the rail/scrubber/top overlays. The input box keeps
  // its own (opaque) frame colour; the completion list shows the scene through.
  ImGui::SetNextWindowBgAlpha(0.65f);

  const ImGuiWindowFlags flags =
      ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
      ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoDocking |
      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoSavedSettings |
      ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoNavInputs;

  if (ImGui::Begin("##CommandPalette", nullptr, flags)) {
    center_ = ImVec2(ImGui::GetWindowPos().x + ImGui::GetWindowSize().x * 0.5f,
                     ImGui::GetWindowPos().y + ImGui::GetWindowSize().y * 0.5f);
    if (focus_input_) {
      ImGui::SetKeyboardFocusHere();
      focus_input_ = false;
    }
    ImGui::SetNextItemWidth(-FLT_MIN);
    const bool entered = ImGui::InputTextWithHint(
        "##cmdinput", "Type to search  ( > UI   . model/data   / agent )",
        input_, sizeof(input_),
        ImGuiInputTextFlags_EnterReturnsTrue |
            ImGuiInputTextFlags_CallbackAlways,
        InputTextCallback, this);

    // One unified list, fuzzy-matched against the whole input. A leading
    // '>'/'.'/'/' only matches names with that prefix, so it narrows by context.
    if (input_[0] != '\0') {
      if (const Command* chosen =
              DrawCompletionList(commands, input_, entered)) {
        if (chosen->name[0] == '/') {
          // Agent command: submit its text so the caller can route it.
          SubmitPlain(chosen->name, on_submit_plain);
        } else {
          // '>' UI / '.' model-data (and future '$'): run the action.
          if (chosen->run) chosen->run();
          Close();
        }
      } else if (entered && input_[0] == '/') {
        // Typed agent text with arguments that matched no completion (e.g.
        // "/prompt how do I..." or "/model sonnet"): submit it as-is.
        SubmitPlain(input_, on_submit_plain);
      }
    }

    // The agent conversation renders in agent context: while typing a '/'
    // command, and once a submitted prompt has cleared the box (empty input), so
    // replies stay visible.
    if (render_below && (input_[0] == '/' || input_[0] == '\0')) {
      render_below();
    }

    if (ImGui::IsKeyPressed(ImGuiKey_Escape)) {
      Close();
    }
  }
  ImGui::End();
}

}  // namespace mujoco::studio
