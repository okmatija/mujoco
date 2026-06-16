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
#include <vector>

#include <imgui.h>

namespace mujoco::studio {
namespace {

// Case-insensitive substring test (empty needle matches everything).
bool ContainsCaseInsensitive(const std::string& haystack,
                             const std::string& needle) {
  if (needle.empty()) {
    return true;
  }
  auto eq = [](char a, char b) {
    return std::tolower(static_cast<unsigned char>(a)) ==
           std::tolower(static_cast<unsigned char>(b));
  };
  return std::search(haystack.begin(), haystack.end(), needle.begin(),
                     needle.end(), eq) != haystack.end();
}

// Draws one completion row: a full-width selectable whose label is the name (in
// the normal text colour), with the optional description drawn dimmed at the
// `desc_x` column on the same row. `desc_x` is a window-local x so descriptions
// line up in a column. Returns true if the row was clicked.
bool CompletionRow(const CommandPalette::Command& cmd, bool selected,
                   float desc_x) {
  const bool clicked = ImGui::Selectable(cmd.name.c_str(), selected);
  if (!cmd.description.empty()) {
    ImGui::SameLine(desc_x);
    ImGui::TextDisabled("%s", cmd.description.c_str());
  }
  return clicked;
}

}  // namespace

int CommandPalette::InputTextCallback(ImGuiInputTextCallbackData* data) {
  auto* self = static_cast<CommandPalette*>(data->UserData);

  // CallbackAlways: one-shot after Open() to undo the select-all that focus
  // applies, so the pre-filled ">" stays put and the cursor sits after it. This
  // fires on the activation frame, after InputText's select-all, so it wins.
  if (data->EventFlag == ImGuiInputTextFlags_CallbackAlways) {
    if (self->init_cursor_end_) {
      data->CursorPos = data->BufTextLen;
      data->SelectionStart = data->SelectionEnd = data->CursorPos;
      self->init_cursor_end_ = false;
    }
    return 0;
  }

  // In list modes ('>' commands, '/' slash commands) leave Up/Down for the
  // filtered list navigation below.
  if (data->BufTextLen > 0 && (data->Buf[0] == '>' || data->Buf[0] == '/')) {
    return 0;
  }

  const int n = static_cast<int>(self->prompt_history_.size());
  if (n == 0) return 0;

  const int prev = self->history_pos_;
  if (data->EventKey == ImGuiKey_UpArrow) {
    if (self->history_pos_ == -1) self->saved_input_ = std::string(data->Buf);
    if (self->history_pos_ + 1 < n) ++self->history_pos_;
  } else if (data->EventKey == ImGuiKey_DownArrow) {
    if (self->history_pos_ > -1) --self->history_pos_;
  }
  if (self->history_pos_ == prev) return 0;

  const std::string& text = (self->history_pos_ == -1)
                                ? self->saved_input_
                                : self->prompt_history_[n - 1 - self->history_pos_];
  data->DeleteChars(0, data->BufTextLen);
  data->InsertChars(0, text.c_str());
  data->CursorPos = data->BufTextLen;
  data->SelectionStart = data->SelectionEnd = data->CursorPos;
  return 0;
}

void CommandPalette::Open() {
  open_ = true;
  focus_input_ = true;
  selection_ = 0;
  history_pos_ = -1;
  saved_input_.clear();
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
  std::vector<const Command*> matches;
  for (const Command& command : list) {
    if (ContainsCaseInsensitive(command.name, query)) {
      matches.push_back(&command);
    }
  }

  // Present candidates alphabetically (case-insensitive), regardless of the
  // order the caller supplied them in.
  std::sort(matches.begin(), matches.end(),
            [](const Command* a, const Command* b) {
              return std::lexicographical_compare(
                  a->name.begin(), a->name.end(), b->name.begin(),
                  b->name.end(), [](unsigned char x, unsigned char y) {
                    return std::tolower(x) < std::tolower(y);
                  });
            });

  // Keyboard navigation through the filtered list.
  if (ImGui::IsKeyPressed(ImGuiKey_DownArrow)) {
    ++selection_;
  }
  if (ImGui::IsKeyPressed(ImGuiKey_UpArrow)) {
    --selection_;
  }
  if (!matches.empty()) {
    const int n = static_cast<int>(matches.size());
    selection_ = (selection_ % n + n) % n;
  } else {
    selection_ = 0;
  }

  // Column where descriptions start: past the widest name so they line up.
  float desc_x = 0.0f;
  for (const Command* command : matches) {
    desc_x = std::max(desc_x, ImGui::CalcTextSize(command->name.c_str()).x);
  }
  desc_x += ImGui::GetStyle().ItemSpacing.x * 3.0f;

  ImGui::Separator();
  const Command* chosen = nullptr;
  for (int i = 0; i < static_cast<int>(matches.size()); ++i) {
    const bool is_selected = (i == selection_);
    if (CompletionRow(*matches[i], is_selected, desc_x) ||
        (is_selected && entered)) {
      chosen = matches[i];
    }
  }
  return chosen;
}

void CommandPalette::SubmitPlain(
    const std::string& text,
    const std::function<void(const std::string&)>& on_submit_plain) {
  if (text.empty()) {
    return;
  }
  if (prompt_history_.empty() || prompt_history_.back() != text) {
    prompt_history_.push_back(text);
  }
  history_pos_ = -1;
  saved_input_.clear();
  if (on_submit_plain) {
    on_submit_plain(text);
  }
  input_[0] = '\0';
  focus_input_ = true;
}

void CommandPalette::Draw(
    const std::vector<Command>& commands,
    const std::vector<Command>& slash_commands, const ImVec4& rect,
    const std::function<void()>& render_below,
    const std::function<void(const std::string&)>& on_submit_plain) {
  if (!open_) {
    return;
  }

  constexpr float kWidth = 480.0f;
  // Centered horizontally; the caller supplies the top edge (rect.y).
  ImGui::SetNextWindowPos(ImVec2(rect.x + rect.z * 0.5f, rect.y),
                          ImGuiCond_Always, ImVec2(0.5f, 0.0f));
  // Fixed width, height grows with the autocomplete list.
  ImGui::SetNextWindowSizeConstraints(ImVec2(kWidth, 0),
                                      ImVec2(kWidth, FLT_MAX));
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
        "##cmdinput", "Use > for app commands or / for agent commands", input_,
        sizeof(input_),
        ImGuiInputTextFlags_EnterReturnsTrue |
            ImGuiInputTextFlags_CallbackHistory |
            ImGuiInputTextFlags_CallbackAlways,
        InputTextCallback, this);

    const bool command_mode = (input_[0] == '>');
    const bool slash_mode = (input_[0] == '/');

    if (command_mode) {
      // '>' command mode: choosing an entry runs its callback.
      if (const Command* chosen =
              DrawCompletionList(commands, input_ + 1, entered)) {
        chosen->run();
        Close();
      }
    } else if (slash_mode) {
      // '/' slash mode: shares the completion list. Choosing an entry submits
      // its name; otherwise Enter submits the typed text (so commands with
      // arguments like "/model sonnet", which match no completion, still work).
      const Command* chosen =
          DrawCompletionList(slash_commands, input_ + 1, entered);
      if (chosen) {
        SubmitPlain(chosen->name, on_submit_plain);
      } else if (entered) {
        SubmitPlain(input_, on_submit_plain);
      }
    } else {
      // Ask mode: plain text submitted with Enter goes to the LLM; the box
      // clears but stays open so the conversation keeps rendering below.
      if (entered && input_[0] != '\0') {
        SubmitPlain(input_, on_submit_plain);
      }
      if (render_below) {
        render_below();
      }
    }

    if (ImGui::IsKeyPressed(ImGuiKey_Escape)) {
      Close();
    }
  }
  ImGui::End();
}

}  // namespace mujoco::studio
