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

// FontAwesome 4 "fa-cog" (U+F013); in the 0xf000-0xf3ff range the studio merges.
constexpr char kCogIcon[] = "\xEF\x80\x93";

// Does `text` match `query` under the given search mode? An empty query always
// matches.
//   kPrefix    -- `text` starts with `query` (classic autocomplete).
//   kSubstring -- `query` appears anywhere in `text` (contiguous).
//   kFuzzy     -- `query`'s characters appear in `text` in order, with gaps.
// `case_insensitive` folds case on both sides.
bool MatchQuery(const std::string& text, const std::string& query,
                CommandPalette::SearchMode mode, bool case_insensitive) {
  if (query.empty()) {
    return true;
  }
  auto eq = [case_insensitive](char a, char b) {
    if (case_insensitive) {
      return std::tolower(static_cast<unsigned char>(a)) ==
             std::tolower(static_cast<unsigned char>(b));
    }
    return a == b;
  };

  switch (mode) {
    case CommandPalette::SearchMode::kPrefix:
      if (text.size() < query.size()) {
        return false;
      }
      for (std::size_t i = 0; i < query.size(); ++i) {
        if (!eq(text[i], query[i])) {
          return false;
        }
      }
      return true;

    case CommandPalette::SearchMode::kSubstring:
      return std::search(text.begin(), text.end(), query.begin(), query.end(),
                         eq) != text.end();

    case CommandPalette::SearchMode::kFuzzy: {
      std::size_t qi = 0;
      for (std::size_t ti = 0; ti < text.size() && qi < query.size(); ++ti) {
        if (eq(text[ti], query[qi])) {
          ++qi;
        }
      }
      return qi == query.size();
    }
  }
  return false;
}

// Which column of a completion row a click landed in.
enum class RowHit { kNone, kName, kValue };

// ===== Match highlighting ==================================================
// Renders the matched characters of a name in a bold font. This whole block is
// self-contained: deleting it (plus the `match_font` branch in CompletionRow and
// the highlight handling in DrawCompletionList) removes the feature and leaves
// the plain, regular-font path untouched.

// Marks which characters of `name` the current query matched, under `mode` (so
// they can be drawn bold). Assumes `name` already matched the query.
std::vector<bool> MatchedChars(const std::string& name, const std::string& query,
                               CommandPalette::SearchMode mode,
                               bool case_insensitive) {
  std::vector<bool> hit(name.size(), false);
  if (query.empty()) {
    return hit;
  }
  auto eq = [case_insensitive](char a, char b) {
    if (case_insensitive) {
      return std::tolower(static_cast<unsigned char>(a)) ==
             std::tolower(static_cast<unsigned char>(b));
    }
    return a == b;
  };
  switch (mode) {
    case CommandPalette::SearchMode::kPrefix:
      for (std::size_t i = 0; i < query.size() && i < name.size(); ++i) {
        hit[i] = true;
      }
      break;
    case CommandPalette::SearchMode::kSubstring:
      for (std::size_t s = 0; s + query.size() <= name.size(); ++s) {
        std::size_t j = 0;
        for (; j < query.size() && eq(name[s + j], query[j]); ++j) {
        }
        if (j == query.size()) {
          for (std::size_t k = 0; k < query.size(); ++k) hit[s + k] = true;
          break;  // bold only the first occurrence.
        }
      }
      break;
    case CommandPalette::SearchMode::kFuzzy: {
      std::size_t qi = 0;
      for (std::size_t i = 0; i < name.size() && qi < query.size(); ++i) {
        if (eq(name[i], query[qi])) {
          hit[i] = true;
          ++qi;
        }
      }
      break;
    }
  }
  return hit;
}

// Lays out `name` in runs of adjacent characters that share a font -- matched
// characters (per `hit`) use `bold`, the rest `regular`. Draws starting at `pos`
// when `dl` is non-null; pass dl=nullptr to only measure. Returns the total
// width (so the caller can size the name column for the mixed fonts). ImGui lays
// out glyphs without kerning, so concatenated run widths match its own layout.
float HighlightedName(ImDrawList* dl, ImVec2 pos, const std::string& name,
                      const std::vector<bool>& hit, ImFont* regular,
                      ImFont* bold, float size, ImU32 col) {
  const char* base = name.c_str();
  const std::size_t n = name.size();
  float x = pos.x;
  for (std::size_t i = 0; i < n;) {
    const bool emphasized = hit[i];
    std::size_t j = i + 1;
    while (j < n && hit[j] == emphasized) {
      ++j;
    }
    ImFont* f = emphasized ? bold : regular;
    if (dl != nullptr) {
      dl->AddText(f, size, ImVec2(x, pos.y), col, base + i, base + j);
    }
    x += f->CalcTextSizeA(size, FLT_MAX, 0.0f, base + i, base + j).x;
    i = j;
  }
  return x - pos.x;
}
// ===========================================================================

// Draws one completion row: a full-width selectable for the background /
// highlight / click, then the value at `desc_x` (a window-local x so values line
// up in a column) -- either `draw_value`'s widget or the `description` text
// (dimmed, except "on"/"off" which are green/red). If `match_font` is non-null
// the name is drawn with the matched characters in that bold font (the
// selectable then draws no label); otherwise the selectable draws the name
// plainly. Returns which column was clicked this frame (kNone if not clicked):
// the value column is everything from the value onward, the name column the rest.
RowHit CompletionRow(const CommandPalette::Command& cmd, bool selected,
                     float desc_x, const std::string& query,
                     CommandPalette::SearchMode mode, bool case_insensitive,
                     ImFont* match_font, bool focus_value) {
  // AllowOverlap so an editable value widget (draw_value) drawn on top of the
  // row's selectable still receives its clicks.
  const ImGuiSelectableFlags sel_flags = ImGuiSelectableFlags_AllowOverlap;
  bool clicked;
  if (match_font != nullptr) {
    // Match-highlighting path: an empty selectable provides the row background /
    // highlight / click, and the name is drawn on top so matched characters can
    // use the bold font.
    const ImVec2 name_pos = ImGui::GetCursorScreenPos();
    ImGui::PushID(cmd.name.c_str());
    clicked = ImGui::Selectable(
        "##row", selected, sel_flags,
        ImVec2(ImGui::GetContentRegionAvail().x, ImGui::GetTextLineHeight()));
    ImGui::PopID();
    const std::vector<bool> hit =
        MatchedChars(cmd.name, query, mode, case_insensitive);
    HighlightedName(ImGui::GetWindowDrawList(), name_pos, cmd.name, hit,
                    ImGui::GetFont(), match_font, ImGui::GetFontSize(),
                    ImGui::GetColorU32(ImGuiCol_Text));
  } else {
    // Plain path: the selectable draws the name as its own label.
    clicked = ImGui::Selectable(cmd.name.c_str(), selected, sel_flags);
  }

  float value_x = FLT_MAX;  // screen-x where the value column begins.
  if (cmd.draw_value) {
    // An editable widget (checkbox / numeric input) fills the value column.
    ImGui::SameLine(desc_x);
    value_x = ImGui::GetCursorScreenPos().x;
    ImGui::SetNextItemWidth(-FLT_MIN);
    if (focus_value) {
      ImGui::SetKeyboardFocusHere();  // Right entered the widget; focus it.
    }
    cmd.draw_value();
  } else if (!cmd.description.empty()) {
    ImGui::SameLine(desc_x);
    value_x = ImGui::GetCursorScreenPos().x;
    if (cmd.description == "on") {
      ImGui::TextColored(ImVec4(0.36f, 0.80f, 0.36f, 1.0f), "on");
    } else if (cmd.description == "off") {
      ImGui::TextColored(ImVec4(0.90f, 0.40f, 0.40f, 1.0f), "off");
    } else {
      ImGui::TextDisabled("%s", cmd.description.c_str());
    }
  }
  if (!clicked) {
    return RowHit::kNone;
  }
  return ImGui::GetMousePos().x >= value_x ? RowHit::kValue : RowHit::kName;
}

// Tab-completion: extend `input` to the next dotted "segment" shared by every
// command that has `input` as a (case-insensitive) prefix -- e.g. ".mjvOp" with
// the .mjvOption.* entries completes to ".mjvOption.". Returns `input` unchanged
// when there is nothing unambiguous to add (and fixes casing to the canonical
// command spelling along the way).
std::string SegmentComplete(const std::string& input,
                            const std::vector<CommandPalette::Command>& list) {
  auto lower = [](char c) {
    return static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  };
  std::string lcp;
  bool any = false;
  for (const CommandPalette::Command& c : list) {
    if (c.name.size() < input.size()) {
      continue;
    }
    bool is_prefix = true;
    for (std::size_t i = 0; i < input.size(); ++i) {
      if (lower(c.name[i]) != lower(input[i])) {
        is_prefix = false;
        break;
      }
    }
    if (!is_prefix) {
      continue;
    }
    if (!any) {
      lcp = c.name;
      any = true;
    } else {
      std::size_t n = 0;
      while (n < lcp.size() && n < c.name.size() &&
             lower(lcp[n]) == lower(c.name[n])) {
        ++n;
      }
      lcp.resize(n);
    }
  }
  if (!any) {
    return input;  // nothing has `input` as a prefix.
  }
  // Stop at the first '.' at or past the typed length, i.e. complete one segment.
  const std::size_t dot = lcp.find('.', input.size());
  if (dot != std::string::npos) {
    lcp.resize(dot + 1);
  }
  return lcp;
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
  } else if (data->EventFlag == ImGuiInputTextFlags_CallbackCompletion) {
    // Tab: complete the current dotted segment against the command list.
    if (self->completion_list_) {
      const std::string completed =
          SegmentComplete(data->Buf, *self->completion_list_);
      if (completed.size() > static_cast<std::size_t>(data->BufTextLen)) {
        data->DeleteChars(0, data->BufTextLen);
        data->InsertChars(0, completed.c_str());
      }
    }
  }
  return 0;
}

void CommandPalette::Open() {
  open_ = true;
  focus_input_ = true;
  selection_ = 0;
  in_list_ = false;
  show_settings_ = false;  // a fresh open shows the command list, not settings.
  last_query_.clear();
  // Start empty (type to search; the hint lists the contexts). OpenWith() may
  // pre-fill it afterwards; the cursor is then parked at the end (see
  // init_cursor_end_) so a pre-filled string isn't select-all'd and wiped.
  input_[0] = '\0';
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
  // Filter by the current search mode (membership only), then always present
  // alphabetically (case-insensitive) so the list stays stable as the user types.
  std::vector<const Command*> matches;
  for (const Command& command : list) {
    if (MatchQuery(command.name, query, search_mode_, case_insensitive_)) {
      matches.push_back(&command);
    }
  }
  std::sort(matches.begin(), matches.end(),
            [](const Command* a, const Command* b) {
              return std::lexicographical_compare(
                  a->name.begin(), a->name.end(), b->name.begin(),
                  b->name.end(), [](unsigned char x, unsigned char y) {
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

  // Keyboard navigation. The cursor stays in the input until Down first enters
  // the list (landing on the top row); after that Down/Up move the selection,
  // and Up from the top row returns focus to the input. `moved` keeps the
  // selected row scrolled into view.
  bool moved = query_changed;
  if (ImGui::IsKeyPressed(ImGuiKey_DownArrow)) {
    if (in_list_) {
      ++selection_;
    } else {
      in_list_ = true;
      selection_ = 0;
    }
    moved = true;
  }
  if (ImGui::IsKeyPressed(ImGuiKey_UpArrow) && in_list_) {
    if (selection_ == 0) {
      in_list_ = false;  // back to the input text
    } else {
      --selection_;
    }
    moved = true;
  }
  if (!matches.empty()) {
    const int n = static_cast<int>(matches.size());
    selection_ = (selection_ % n + n) % n;
  } else {
    selection_ = 0;
  }

  // In list mode, the highlighted command's value reacts to Left/Right without
  // running it or closing the palette: a `cycle` value (flag/enum) cycles in
  // place; otherwise (a numeric input) Right gives the input keyboard focus so
  // it can be typed. Values rebuilt by the caller each frame reflect the change.
  if (in_list_ && !matches.empty()) {
    const Command* sel = matches[selection_];
    if (sel->cycle) {
      if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow)) {
        sel->cycle(-1);
      } else if (ImGui::IsKeyPressed(ImGuiKey_RightArrow)) {
        sel->cycle(1);
      }
    } else if (sel->draw_value && ImGui::IsKeyPressed(ImGuiKey_RightArrow)) {
      focus_value_ = true;
    }
  }

  if (matches.empty()) {
    return nullptr;  // nothing to show (e.g. mid-typing "/prompt question").
  }

  // Bold weight for match highlighting (the font the app loads after the
  // default); null when highlighting is off, which selects the plain path. See
  // the "Match highlighting" block above.
  const ImVector<ImFont*>& fonts = ImGui::GetIO().Fonts->Fonts;
  ImFont* match_font =
      (highlight_matches_ && fonts.Size > 1) ? fonts[1] : nullptr;

  // Column layout: the value column begins just past the widest name.
  const float spacing = ImGui::GetStyle().ItemSpacing.x;
  float name_w = 0.0f;
  for (const Command* command : matches) {
    float w;
    if (match_font != nullptr) {
      // The name renders in mixed fonts, so measure it the same way.
      const std::vector<bool> hit =
          MatchedChars(command->name, query, search_mode_, case_insensitive_);
      w = HighlightedName(nullptr, ImVec2(0, 0), command->name, hit,
                          ImGui::GetFont(), match_font, ImGui::GetFontSize(), 0);
    } else {
      w = ImGui::CalcTextSize(command->name.c_str()).x;
    }
    name_w = std::max(name_w, w);
  }
  const float desc_x = name_w + spacing * 3.0f;

  ImGui::Separator();

  // The list scrolls inside a child so the whole window stays under the 80% cap
  // set in Draw(); leave room for the input row above.
  const float max_h = ImGui::GetMainViewport()->WorkSize.y * 0.8f -
                      ImGui::GetFrameHeightWithSpacing() * 2.0f;
  ImGui::SetNextWindowSizeConstraints(ImVec2(0, 0), ImVec2(FLT_MAX, max_h));
  const Command* chosen = nullptr;
  if (ImGui::BeginChild(
          "##completions", ImVec2(0, 0),
          ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysUseWindowPadding)) {
    // Make a held (pressed) row use the hover colour rather than the darker
    // "active" colour.
    ImGui::PushStyleColor(ImGuiCol_HeaderActive,
                          ImGui::GetStyleColorVec4(ImGuiCol_HeaderHovered));
    for (int i = 0; i < static_cast<int>(matches.size()); ++i) {
      const Command* command = matches[i];
      const bool at_selection = (i == selection_);
      const float row_min = ImGui::GetCursorPosY();
      // Only highlight once the user has entered the list (in_list_); before
      // that the selection is invisible but still the Enter target (top match).
      const RowHit hit = CompletionRow(
          *command, in_list_ && at_selection, desc_x, query, search_mode_,
          case_insensitive_, match_font, focus_value_ && at_selection);
      if (hit != RowHit::kNone) {
        // A click moves the list focus to this row (and refocuses the input so
        // typing keeps working); it never runs or closes the palette. A click on
        // the value column also cycles that value in place.
        selection_ = i;
        in_list_ = true;
        focus_input_ = true;
        if (hit == RowHit::kValue && command->cycle) {
          command->cycle(1);
        }
      }
      // Enter (from the input) runs/submits the selected row; clicks never do.
      if (at_selection && entered) {
        chosen = command;
      }
      // Keep the selection visible, but only scroll once it reaches an edge of
      // the view (not when it's somewhere in the middle).
      if (at_selection && moved) {
        const float row_max = row_min + ImGui::GetTextLineHeightWithSpacing();
        const float view_h = ImGui::GetWindowHeight();
        const float scroll = ImGui::GetScrollY();
        if (row_min < scroll) {
          ImGui::SetScrollY(row_min);
        } else if (row_max > scroll + view_h) {
          ImGui::SetScrollY(row_max - view_h);
        }
      }
    }
    ImGui::PopStyleColor();
  }
  ImGui::EndChild();
  focus_value_ = false;  // one-shot, consumed by the selected row above.
  return chosen;
}

void CommandPalette::DrawSettings(const std::function<void()>& render_settings) {
  ImGui::Separator();

  // The palette window is translucent; give the settings panel a solid
  // background (the window's own colour, opaque) so its controls read clearly.
  ImVec4 bg = ImGui::GetStyleColorVec4(ImGuiCol_WindowBg);
  bg.w = 1.0f;
  ImGui::PushStyleColor(ImGuiCol_ChildBg, bg);
  ImGui::BeginChild(
      "##settings", ImVec2(0, 0),
      ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysUseWindowPadding);

  if (ImGui::CollapsingHeader("Command palette settings",
                              ImGuiTreeNodeFlags_DefaultOpen)) {
    const char* kModes[] = {"Prefix", "Substring", "Fuzzy"};
    int mode = static_cast<int>(search_mode_);
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x * 0.5f);
    if (ImGui::Combo("Autocomplete Mode", &mode, kModes, IM_ARRAYSIZE(kModes))) {
      search_mode_ = static_cast<SearchMode>(mode);
    }
    ImGui::Checkbox("Case Insensitive", &case_insensitive_);
    ImGui::Checkbox("Highlight Matches", &highlight_matches_);
  }

  // Host- (and plugin-) provided settings go below the palette's own.
  if (render_settings) {
    ImGui::Spacing();
    render_settings();
  }

  ImGui::EndChild();
  ImGui::PopStyleColor();
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
    const std::function<void(const std::string&)>& on_submit_plain,
    const std::function<void()>& render_settings) {
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
    // Input box, leaving room on the right for the cog settings toggle.
    const float cog_w = ImGui::GetFrameHeight();  // square button
    ImGui::SetNextItemWidth(-(cog_w + ImGui::GetStyle().ItemSpacing.x));
    completion_list_ = &commands;  // for the Tab-completion callback.
    const bool entered = ImGui::InputTextWithHint(
        "##cmdinput", "Type to search  ( > UI   . model/data   / agent )",
        input_, sizeof(input_),
        ImGuiInputTextFlags_EnterReturnsTrue |
            ImGuiInputTextFlags_CallbackAlways |
            ImGuiInputTextFlags_CallbackCompletion,
        InputTextCallback, this);

    // Cog toggle: opens the settings panel in place of the completion list.
    // Capture the tint state before the button, since clicking it flips
    // show_settings_ (the push and pop must use the same value).
    ImGui::SameLine();
    const bool cog_active = show_settings_;
    if (cog_active) {
      ImGui::PushStyleColor(ImGuiCol_Button,
                            ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
    }
    if (ImGui::Button(kCogIcon, ImVec2(cog_w, cog_w))) {
      show_settings_ = !show_settings_;
      focus_input_ = true;  // keep typing working after toggling.
    }
    if (cog_active) {
      ImGui::PopStyleColor();
    }
    ImGui::SetItemTooltip("Command Palette Preferences");

    if (show_settings_) {
      DrawSettings(render_settings);
    } else {
      // One unified list, matched against the whole input. A leading
      // '>'/'.'/'/' only matches names with that prefix, narrowing by context.
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
      // command, and once a submitted prompt has cleared the box (empty input),
      // so replies stay visible.
      if (render_below && (input_[0] == '/' || input_[0] == '\0')) {
        render_below();
      }
    }

    if (ImGui::IsKeyPressed(ImGuiKey_Escape)) {
      // Escape backs out of settings first, then closes the palette.
      if (show_settings_) {
        show_settings_ = false;
        focus_input_ = true;
      } else {
        Close();
      }
    }
  }
  ImGui::End();
}

}  // namespace mujoco::studio
