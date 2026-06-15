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

#include "experimental/studio/llm/llm_panel.h"

#include <algorithm>
#include <cfloat>
#include <string>
#include <vector>

#include <imgui.h>

#include "experimental/studio/llm/ui_agent.h"

namespace mujoco::studio {

namespace {
// Characters revealed per frame for the typewriter effect.
constexpr int kRevealSpeed = 3;
}  // namespace

void LlmPanel::Render(UiAgent& agent) {
  const std::vector<UiAgent::Turn>& history = agent.history();
  const int n = static_cast<int>(history.size());

  // Detect /clear (or any history reset) so the typewriter starts fresh, and a
  // newly appended turn so we scroll the transcript to the bottom once.
  if (n < last_turn_count_) {
    revealing_index_ = -1;
    reveal_chars_ = 0;
  }
  if (n != last_turn_count_) {
    scroll_to_bottom_ = true;
    last_turn_count_ = n;
  }

  if (n == 0 && !agent.busy()) {
    // Nothing to show yet; the input box hint guides the user.
    return;
  }

  // The most recent assistant turn gets the typewriter reveal; older ones show
  // in full.
  int last_assistant = -1;
  for (int i = 0; i < n; ++i) {
    if (history[i].role == "assistant") last_assistant = i;
  }

  ImGui::Separator();

  // Scrollable transcript: grows with content up to a cap, then scrolls so the
  // whole conversation stays reachable by scrolling up.
  const float max_h = ImGui::GetMainViewport()->WorkSize.y * 0.5f;
  ImGui::SetNextWindowSizeConstraints(ImVec2(0, 0), ImVec2(FLT_MAX, max_h));
  if (ImGui::BeginChild("##convo", ImVec2(0, 0), ImGuiChildFlags_AutoResizeY)) {
    for (int i = 0; i < n; ++i) {
      const UiAgent::Turn& turn = history[i];
      const bool is_user = (turn.role == "user");
      if (is_user) {
        ImGui::TextDisabled("You");
      } else if (!turn.model.empty()) {
        ImGui::TextDisabled("Agent (%s)", turn.model.c_str());
      } else {
        ImGui::TextDisabled("Agent");
      }

      // Collapsible extended-thinking section (closed by default), shown only
      // for assistant turns that actually produced reasoning.
      if (!is_user && !turn.thinking.empty()) {
        ImGui::PushID(i);
        if (ImGui::CollapsingHeader("Thoughts")) {
          ImGui::PushStyleColor(ImGuiCol_Text,
                                ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
          ImGui::TextWrapped("%s", turn.thinking.c_str());
          ImGui::PopStyleColor();
        }
        ImGui::PopID();
      }

      if (!is_user && i == last_assistant) {
        // Typewriter reveal of the latest reply.
        if (i != revealing_index_) {
          revealing_index_ = i;
          reveal_chars_ = 0;
        }
        reveal_chars_ = std::min(reveal_chars_ + kRevealSpeed,
                                 static_cast<int>(turn.text.size()));
        ImGui::TextWrapped("%.*s", reveal_chars_, turn.text.c_str());
      } else {
        ImGui::TextWrapped("%s", turn.text.c_str());
      }
      ImGui::Spacing();
    }

    if (agent.busy()) {
      const std::string model = agent.provider_model();
      if (model.empty()) {
        ImGui::TextDisabled("Agent");
      } else {
        ImGui::TextDisabled("Agent (%s)", model.c_str());
      }
      const int dots = 1 + (static_cast<int>(ImGui::GetTime() * 3.0) % 3);
      ImGui::TextDisabled("thinking%s", std::string(dots, '.').c_str());
      // Cancel pushed to the right edge of the conversation panel.
      const char* kCancel = "Cancel";
      const float button_w = ImGui::CalcTextSize(kCancel).x +
                             ImGui::GetStyle().FramePadding.x * 2.0f;
      ImGui::SameLine(ImGui::GetContentRegionMax().x - button_w);
      if (ImGui::SmallButton(kCancel)) {
        agent.Cancel();
      }
    }

    // Follow new content while pinned to the bottom; release when the user
    // scrolls up. A freshly appended turn forces a one-shot jump.
    const bool revealing =
        last_assistant >= 0 &&
        reveal_chars_ < static_cast<int>(history[last_assistant].text.size());
    if (scroll_to_bottom_) {
      ImGui::SetScrollHereY(1.0f);
      scroll_to_bottom_ = false;
    } else if ((agent.busy() || revealing) &&
               ImGui::GetScrollY() >= ImGui::GetScrollMaxY()) {
      ImGui::SetScrollHereY(1.0f);
    }
  }
  ImGui::EndChild();
}

}  // namespace mujoco::studio
