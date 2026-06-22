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

#include "experimental/platform/ux/command_palette.h"

#include <algorithm>
#include <cctype>
#include <cfloat>
#include <cstddef>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include <imgui.h>
#include <mujoco/mjxmacro.h>
#include <mujoco/mujoco.h>

#include "experimental/platform/sim/step_control.h"

namespace mujoco::platform {
namespace {

// FontAwesome 4 "fa-cog" (U+F013); in the 0xf000-0xf3ff range the host merges.
constexpr char kCogIcon[] = "\xEF\x80\x93";
// FontAwesome 4 "fa-undo" (U+F0E2), the revert/restore-default button glyph.
constexpr char kRevertIcon[] = "\xEF\x83\xA2";

// Builds a constant-style identifier from a table name: "Convex Hull" ->
// CONVEXHULL (used for the flag path segments from the mj*STRING tables).
std::string Ident(const std::string& s) {
  std::string out;
  for (char c : s) {
    if (std::isalnum(static_cast<unsigned char>(c))) {
      out += static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    }
  }
  return out;
}

// Wraps a MuJoCo enum *STRING table (e.g. mjINTEGRATORSTRING) as the names
// vector RegisterEnumField takes, so the combo labels come straight from the
// headers instead of hand-written literals.
std::vector<const char*> EnumNames(const char* const* table, int n) {
  return std::vector<const char*>(table, table + n);
}

// Builds a value-column draw lambda for a boolean: a checkbox right-aligned at
// the right edge of the value column (the revert button, when shown, lives in
// its own column further right). `id` is the widget id ("###path").
std::function<void()> RightAlignedCheckbox(std::string id,
                                           std::function<bool()> get,
                                           std::function<void(bool)> set) {
  return [id = std::move(id), get = std::move(get), set = std::move(set)] {
    bool b = get();
    const float offset =
        ImGui::GetContentRegionAvail().x - ImGui::GetFrameHeight();
    if (offset > 0.0f) {
      ImGui::SetCursorPosX(ImGui::GetCursorPosX() + offset);
    }
    if (ImGui::Checkbox(id.c_str(), &b)) set(b);
  };
}

// Character equality for query matching, optionally folding case. Shared by
// MatchQuery (the filter predicate) and MatchedChars (match highlighting).
bool CharsEqual(char a, char b, bool case_insensitive) {
  if (case_insensitive) {
    return std::tolower(static_cast<unsigned char>(a)) ==
           std::tolower(static_cast<unsigned char>(b));
  }
  return a == b;
}

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
    return CharsEqual(a, b, case_insensitive);
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
enum class RowHit { kNone, kName, kValue, kActivated };

// ===== Match highlighting ==================================================
// Renders the matched characters of a name in a bold font. This whole block is
// self-contained: deleting it (plus the `match_font` branch in CompletionRow and
// the highlight handling in DrawCompletionList) removes the feature and leaves
// the plain, regular-font path untouched.

// Marks which characters of `name` the current query matched, under `mode` (so
// they can be drawn bold). Assumes `name` already matched the query.
std::vector<char> MatchedChars(const std::string& name, const std::string& query,
                               CommandPalette::SearchMode mode,
                               bool case_insensitive) {
  std::vector<char> hit(name.size(), false);
  if (query.empty()) {
    return hit;
  }
  auto eq = [case_insensitive](char a, char b) {
    return CharsEqual(a, b, case_insensitive);
  };
  switch (mode) {
    case CommandPalette::SearchMode::kPrefix:
      for (std::size_t i = 0; i < query.size() && i < name.size(); ++i) {
        hit[i] = eq(name[i], query[i]);
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
                      const std::vector<char>& hit, ImFont* regular,
                      ImFont* bold, float size, ImU32 col) {
  const char* base = name.c_str();
  const std::size_t n = name.size();
  float x = pos.x;
  for (std::size_t i = 0; i < n;) {
    const char emphasized = hit[i];
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

// Draws one completion row into the name | value | revert table set up by
// DrawCompletionList. A selectable spanning all columns provides the row
// background / highlight / click; the name is drawn over the first column (its
// matched characters in `match_font` when non-null, else the selectable renders
// the name itself). The caller must have started the row and selected column 0.
// Returns which column was clicked this frame (kNone if not): a click in the
// value or revert column counts as kValue (so the caller cycles the value),
// anything else as kName, and the Execute button as kActivated.
RowHit CompletionRow(const CommandPalette::Command& cmd, bool selected,
                     const std::string& query, CommandPalette::SearchMode mode,
                     bool case_insensitive, ImFont* match_font,
                     bool focus_value) {
  RowHit result = RowHit::kNone;

  // Name column. SpanAllColumns makes the highlight/click cover the whole row;
  // AllowOverlap lets the value widget drawn on top still receive its clicks.
  ImVec2 name_pos = ImGui::GetCursorScreenPos();
  // Center the name against the framed value widgets (GetFrameHeight tall);
  // without the FramePadding.y nudge the text sits at the top of the widget.
  name_pos.y += ImGui::GetStyle().FramePadding.y;
  ImGui::PushID(cmd.name.c_str());
  const ImGuiSelectableFlags sel_flags =
      ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowOverlap;
  // Empty label on the match path (the bold name is drawn below); the plain path
  // lets the selectable render the name itself.
  const bool clicked = ImGui::Selectable(
      match_font != nullptr ? "##row" : cmd.name.c_str(), selected, sel_flags,
      ImVec2(0, ImGui::GetFrameHeight()));
  ImGui::PopID();
  if (match_font != nullptr) {
    const std::vector<char> hit =
        MatchedChars(cmd.name, query, mode, case_insensitive);
    HighlightedName(ImGui::GetWindowDrawList(), name_pos, cmd.name, hit,
                    ImGui::GetFont(), match_font, ImGui::GetFontSize(),
                    ImGui::GetColorU32(ImGuiCol_Text));
  }

  // Value column: editable widget / Execute button / dimmed description.
  ImGui::TableNextColumn();
  if (cmd.draw_value) {
    // An editable widget (checkbox / numeric input / combo) fills the column.
    ImGui::SetNextItemWidth(-FLT_MIN);
    if (focus_value) {
      ImGui::SetKeyboardFocusHere();  // Right entered the widget; focus it.
    }
    cmd.draw_value();
  } else if (cmd.run) {
    // A pure action ('>' command): an "Execute" button filling the column.
    // Clicking it activates the command -- Draw runs `run` and closes, exactly
    // as Enter on the selected row does.
    ImGui::PushID(cmd.name.c_str());
    const bool pressed = ImGui::Button("Execute", ImVec2(-FLT_MIN, 0.0f));
    ImGui::PopID();
    if (pressed) {
      result = RowHit::kActivated;
    }
  } else if (!cmd.description.empty()) {
    // A dimmed one-line hint (entries without a value widget).
    ImGui::AlignTextToFramePadding();  // center against the frame-height row
    ImGui::TextDisabled("%s", cmd.description.c_str());
  }

  // Revert column: reset-to-default button, shown only when the value differs
  // from its default. Its own fixed column keeps every revert button aligned.
  ImGui::TableNextColumn();
  if (cmd.draw_value && cmd.modified && cmd.reset) {
    if (ImGui::Button(
            (std::string(kRevertIcon) + "##revert" + cmd.name).c_str())) {
      cmd.reset();
    }
    if (cmd.default_text.empty()) {
      ImGui::SetItemTooltip("Reset to default");
    } else {
      ImGui::SetItemTooltip("Reset to '%s'", cmd.default_text.c_str());
    }
  }

  if (result == RowHit::kActivated) {
    return result;
  }
  if (!clicked) {
    return RowHit::kNone;
  }
  // A click in the value or revert column cycles the value; a click on the name
  // just moves the selection.
  return ImGui::TableGetHoveredColumn() >= 1 ? RowHit::kValue : RowHit::kName;
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

  // Tab: complete the current dotted segment against the command list.
  // (Up/Down/Left/Right are read via IsKeyPressed in DrawCompletionList, so the
  // single-line input never handles them itself.)
  if (data->EventFlag == ImGuiInputTextFlags_CallbackCompletion) {
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
  input_[0] = '\0';  // start empty; type to search (the hint lists the contexts).
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
      const std::vector<char> hit =
          MatchedChars(command->name, query, search_mode_, case_insensitive_);
      w = HighlightedName(nullptr, ImVec2(0, 0), command->name, hit,
                          ImGui::GetFont(), match_font, ImGui::GetFontSize(), 0);
    } else {
      w = ImGui::CalcTextSize(command->name.c_str()).x;
    }
    name_w = std::max(name_w, w);
  }
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
    // Three columns line everything up: the name (fixed to the widest name), the
    // value (stretches to fill), and a fixed slot for the revert button so the
    // values all end at the same x whether or not a revert button is shown.
    const float revert_w = ImGui::CalcTextSize(kRevertIcon).x +
                           ImGui::GetStyle().FramePadding.x * 2.0f;
    const ImGuiTableFlags table_flags =
        ImGuiTableFlags_NoSavedSettings | ImGuiTableFlags_NoPadOuterX;
    if (ImGui::BeginTable("##rows", 3, table_flags)) {
      ImGui::TableSetupColumn("name", ImGuiTableColumnFlags_WidthFixed,
                              name_w + spacing * 2.0f);
      ImGui::TableSetupColumn("value", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableSetupColumn("revert", ImGuiTableColumnFlags_WidthFixed,
                              revert_w);
      for (int i = 0; i < static_cast<int>(matches.size()); ++i) {
        const Command* command = matches[i];
        const bool at_selection = (i == selection_);
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        const float row_min = ImGui::GetCursorPosY();
        // Only highlight once the user has entered the list (in_list_); before
        // that the selection is invisible but still the Enter target (top match).
        const RowHit hit = CompletionRow(
            *command, in_list_ && at_selection, query, search_mode_,
            case_insensitive_, match_font, focus_value_ && at_selection);
        if (hit == RowHit::kActivated) {
          // An action button was pressed: treat it as choosing this row (Draw
          // runs its `run` and closes).
          chosen = command;
        } else if (hit != RowHit::kNone) {
          // A click moves the list focus to this row (and refocuses the input so
          // typing keeps working); it never runs or closes the palette. A click
          // on the value column also cycles that value in place.
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
          const float row_max = row_min + ImGui::GetFrameHeightWithSpacing();
          const float view_h = ImGui::GetWindowHeight();
          const float scroll = ImGui::GetScrollY();
          if (row_min < scroll) {
            ImGui::SetScrollY(row_min);
          } else if (row_max > scroll + view_h) {
            ImGui::SetScrollY(row_max - view_h);
          }
        }
      }
      ImGui::EndTable();
    }
    ImGui::PopStyleColor();
  }
  ImGui::EndChild();
  focus_value_ = false;  // one-shot, consumed by the selected row above.
  return chosen;
}

void CommandPalette::DrawSettings() {
  ImGui::Separator();

  // The palette window is translucent; give the settings panel a solid
  // background (the window's own colour, opaque) so its controls read clearly.
  ImVec4 bg = ImGui::GetStyleColorVec4(ImGuiCol_WindowBg);
  bg.w = 1.0f;
  ImGui::PushStyleColor(ImGuiCol_ChildBg, bg);
  ImGui::BeginChild(
      "##settings", ImVec2(0, 0),
      ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysUseWindowPadding);

  if (ImGui::CollapsingHeader("Command Palette Settings",
                              ImGuiTreeNodeFlags_DefaultOpen)) {
    const char* kModes[] = {"Prefix", "Substring", "Fuzzy"};
    int mode = static_cast<int>(search_mode_);
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x * 0.5f);
    if (ImGui::Combo("Search Mode", &mode, kModes, IM_ARRAYSIZE(kModes))) {
      search_mode_ = static_cast<SearchMode>(mode);
    }
    ImGui::Checkbox("Case Insensitive", &case_insensitive_);
    ImGui::Checkbox("Highlight Matches", &highlight_matches_);
  }

  ImGui::EndChild();
  ImGui::PopStyleColor();
}

void CommandPalette::Draw(const std::vector<Command>& commands,
                          const ImVec4& rect) {
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
    if (focus_input_) {
      ImGui::SetKeyboardFocusHere();
      focus_input_ = false;
    }
    // Input box, leaving room on the right for the cog settings toggle.
    const float cog_w = ImGui::GetFrameHeight();  // square button
    ImGui::SetNextItemWidth(-(cog_w + ImGui::GetStyle().ItemSpacing.x));
    completion_list_ = &commands;  // for the Tab-completion callback.
    const bool entered = ImGui::InputTextWithHint(
        "##cmdinput", "Type to search",
        input_, sizeof(input_),
        ImGuiInputTextFlags_EnterReturnsTrue |
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

    // Only show the part below the input (the completion list or the settings
    // panel) while the palette is focused. Clicking outside it (e.g. into the
    // viewport) defocuses it, collapsing the window to just the input box, which
    // still shows -- and stays above the docked panels, since the palette is a
    // floating window -- so clicking back into it brings the list back.
    const bool focused =
        ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows);

    if (focused) {
      if (show_settings_) {
        DrawSettings();
      } else if (input_[0] != '\0') {
        // One unified list, matched against the whole input. A leading '>'/'.'
        // only matches names with that prefix, narrowing by context. Choosing an
        // entry runs its action and closes.
        if (const Command* chosen =
                DrawCompletionList(commands, input_, entered)) {
          if (chosen->run) chosen->run();
          Close();
        }
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

void RegisterFlagField(std::vector<CommandPalette::Command>& out,
                       const std::string& path, std::function<bool()> get,
                       std::function<void(bool)> set, bool dflt) {
  const bool modified = get() != dflt;
  auto toggle = [get, set] { set(!get()); };
  auto checkbox = RightAlignedCheckbox("###" + path, get, set);
  auto reset = [set, dflt] { set(dflt); };
  out.push_back({detail::DisplayName(path, modified), toggle, "",
                 [toggle](int) { toggle(); }, checkbox, reset, modified,
                 dflt ? "on" : "off"});
}

void RegisterChoice(std::vector<CommandPalette::Command>& out,
                    const std::string& name, std::function<int()> get,
                    std::function<void(int)> set,
                    std::vector<const char*> names) {
  const int n = static_cast<int>(names.size());
  const std::string id = "###" + name;
  auto cyc = [get, set, n](int delta) {
    const int v = get();
    set(((v + delta) % n + n) % n);
  };
  auto combo = [id, get, set, names, n] {
    int v = get();
    if (ImGui::Combo(id.c_str(), &v, names.data(), n)) set(v);
  };
  out.push_back({name, [cyc] { cyc(1); }, "", cyc, combo, {}, false, ""});
}

void RegisterCustomField(std::vector<CommandPalette::Command>& out,
                         const std::string& path, std::function<void()> draw,
                         std::function<void()> reset, bool modified,
                         std::string default_text) {
  out.push_back({detail::DisplayName(path, modified), {}, "", {},
                 std::move(draw), std::move(reset), modified,
                 std::move(default_text)});
}

void RegisterMjvOptionFields(std::vector<CommandPalette::Command>& out,
                             const std::string& prefix, mjvOption* opt) {
  // Element-visibility flags. Column [1] of each table is the default.
  for (int i = 0; i < mjNVISFLAG; ++i) {
    RegisterFlagField(
        out, prefix + ".flags." + Ident(mjVISSTRING[i][0]),
        [opt, i] { return opt->flags[i] != 0; },
        [opt, i](bool b) { opt->flags[i] = b; }, mjVISSTRING[i][1][0] == '1');
  }
  // Per-group visibility toggles. MuJoCo has no xmacro for mjvOption, so list
  // its group arrays with a small local one (stringifying each name with #NAME
  // so the path and member stay in sync). Each is mjtByte[mjNGROUP], and
  // mjv_defaultOption shows groups 0-2, hides 3-5.
#define MJV_GROUP_ARRAYS(X)                                  \
  X(geomgroup) X(sitegroup) X(jointgroup) X(tendongroup)     \
  X(actuatorgroup) X(flexgroup) X(skingroup)
  const struct {
    const char* name;
    mjtByte* arr;
  } kGroups[] = {
#define X(NAME) {#NAME, opt->NAME},
      MJV_GROUP_ARRAYS(X)
#undef X
  };
#undef MJV_GROUP_ARRAYS
  for (const auto& g : kGroups) {
    mjtByte* arr = g.arr;
    for (int i = 0; i < mjNGROUP; ++i) {
      RegisterFlagField(
          out, prefix + "." + g.name + "." + std::to_string(i),
          [arr, i] { return arr[i] != 0; },
          [arr, i](bool b) { arr[i] = static_cast<mjtByte>(b); }, i < 3);
    }
  }
}

void RegisterMjvSceneFields(std::vector<CommandPalette::Command>& out,
                            const std::string& prefix, mjvScene* scn) {
  // Render-effect flags. Column [1] of each table is the default.
  for (int i = 0; i < mjNRNDFLAG; ++i) {
    RegisterFlagField(
        out, prefix + ".flags." + Ident(mjRNDSTRING[i][0]),
        [scn, i] { return scn->flags[i] != 0; },
        [scn, i](bool b) { scn->flags[i] = b; }, mjRNDSTRING[i][1][0] == '1');
  }
}

void RegisterMjOptionFields(std::vector<CommandPalette::Command>& out,
                            const std::string& prefix, mjOption* opt) {
  mjOption def;
  mj_defaultOption(&def);

  for (int i = 0; i < mjNDISABLE; ++i) {
    RegisterFlagField(
        out, prefix + ".disableflags." + Ident(mjDISABLESTRING[i]),
        [opt, i] { return ((opt->disableflags >> i) & 1) != 0; },
        [opt, i](bool b) {
          if (b) {
            opt->disableflags |= (1 << i);
          } else {
            opt->disableflags &= ~(1 << i);
          }
        },
        ((def.disableflags >> i) & 1) != 0);
  }
  for (int i = 0; i < mjNENABLE; ++i) {
    RegisterFlagField(
        out, prefix + ".enableflags." + Ident(mjENABLESTRING[i]),
        [opt, i] { return ((opt->enableflags >> i) & 1) != 0; },
        [opt, i](bool b) {
          if (b) {
            opt->enableflags |= (1 << i);
          } else {
            opt->enableflags &= ~(1 << i);
          }
        },
        ((def.enableflags >> i) & 1) != 0);
  }
  // disableactuator is a bitfield over actuator groups (the Physics panel's
  // "Act Group N"), so expose one toggle per group bit, like the flags above.
  for (int i = 0; i < mjNGROUP; ++i) {
    RegisterFlagField(
        out, prefix + ".disableactuator." + std::to_string(i),
        [opt, i] { return (opt->disableactuator & (1 << i)) != 0; },
        [opt, i](bool b) {
          if (b) {
            opt->disableactuator |= (1 << i);
          } else {
            opt->disableactuator &= ~(1 << i);
          }
        },
        (def.disableactuator & (1 << i)) != 0);
  }

  // Solver/integrator enums: a combo in the value column (also Enter advances /
  // Left-Right cycles).
  RegisterEnumField(out, prefix + ".integrator", &opt->integrator,
                    EnumNames(mjINTEGRATORSTRING, mjNINTEGRATOR), def.integrator);
  RegisterEnumField(out, prefix + ".cone", &opt->cone,
                    EnumNames(mjCONESTRING, mjNCONE), def.cone);
  RegisterEnumField(out, prefix + ".jacobian", &opt->jacobian,
                    EnumNames(mjJACOBIANSTRING, mjNJACOBIAN), def.jacobian);
  RegisterEnumField(out, prefix + ".solver", &opt->solver,
                    EnumNames(mjSOLVERSTRING, mjNSOLVERS), def.solver);

  // The remaining scalar/vector fields, generated from the mjxmacro so the list
  // tracks the struct. The enum/bitfield ints handled above are skipped.
  auto special = [](const char* name) {
    for (const char* s : {"integrator", "cone", "jacobian", "solver",
                          "disableflags", "enableflags", "disableactuator"}) {
      if (std::strcmp(name, s) == 0) return true;
    }
    return false;
  };
#define X(TYPE, NAME, SZ) \
  if (!special(#NAME)) \
    RegisterScalarField(out, prefix + "." #NAME, &opt->NAME, def.NAME);
#define XVEC(TYPE, NAME, SZ) \
  RegisterArrayField(out, prefix + "." #NAME, opt->NAME, SZ, def.NAME);
  MJOPTION_FIELDS
#undef X
#undef XVEC
}

void RegisterMjVisualFields(std::vector<CommandPalette::Command>& out,
                            const std::string& prefix, mjVisual* vis) {
  // Every field of mjVisual's six sub-structs, generated from the mjxmacros
  // against the library default (mj_defaultVisual). Booleans here are plain ints
  // (e.g. global.orthographic), so they show as 0/1 inputs rather than
  // checkboxes. VIS_STR stringifies the current sub-struct token (#SUB would
  // yield "SUB"); MJVIS_SUB names it both as a path segment and a member.
  mjVisual visdef;
  mj_defaultVisual(&visdef);
#define VIS_STR2(x) #x
#define VIS_STR(x) VIS_STR2(x)
#define X(TYPE, NAME, SZ) \
  RegisterScalarField(out, prefix + "." VIS_STR(MJVIS_SUB) "." #NAME, \
                      &vis->MJVIS_SUB.NAME, visdef.MJVIS_SUB.NAME);
#define XVEC(TYPE, NAME, SZ) \
  RegisterArrayField(out, prefix + "." VIS_STR(MJVIS_SUB) "." #NAME, \
                     vis->MJVIS_SUB.NAME, SZ, visdef.MJVIS_SUB.NAME);
#define MJVIS_SUB global
  MJVISUAL_GLOBAL_FIELDS
#undef MJVIS_SUB
#define MJVIS_SUB quality
  MJVISUAL_QUALITY_FIELDS
#undef MJVIS_SUB
#define MJVIS_SUB headlight
  MJVISUAL_HEADLIGHT_FIELDS
#undef MJVIS_SUB
#define MJVIS_SUB map
  MJVISUAL_MAP_FIELDS
#undef MJVIS_SUB
#define MJVIS_SUB scale
  MJVISUAL_SCALE_FIELDS
#undef MJVIS_SUB
#define MJVIS_SUB rgba
  MJVISUAL_RGBA_FIELDS
#undef MJVIS_SUB
#undef X
#undef XVEC
#undef VIS_STR
#undef VIS_STR2
}

void RegisterMjStatisticFields(std::vector<CommandPalette::Command>& out,
                               const std::string& prefix, mjStatistic* stat,
                               const mjStatistic& stat_default) {
  // All fields are mjtNum, so the xmacro entries carry no type. mjStatistic has
  // no library default, so compare against `stat_default`.
#define X(NAME, SZ) \
  RegisterScalarField(out, prefix + "." #NAME, &stat->NAME, stat_default.NAME);
#define XVEC(NAME, SZ) \
  RegisterArrayField(out, prefix + "." #NAME, stat->NAME, SZ, stat_default.NAME);
  MJSTATISTIC_FIELDS
#undef X
#undef XVEC
}

void RegisterStepControlNoiseFields(std::vector<CommandPalette::Command>& out,
                                    const std::string& prefix,
                                    StepControl* step_control) {
  // The noise pair is reached only through Get/SetNoiseParameters (no plain
  // pointer to bind), so it uses RegisterCustomField. Default is 0.
  auto add = [&out, step_control](const std::string& path, bool is_scale) {
    float scale = 0, rate = 0;
    step_control->GetNoiseParameters(scale, rate);
    const float cur = is_scale ? scale : rate;
    const std::string id = "###" + path;
    auto draw = [step_control, id, is_scale] {
      float s = 0, r = 0;
      step_control->GetNoiseParameters(s, r);
      if (ImGui::InputScalar(id.c_str(), ImGuiDataType_Float,
                             is_scale ? &s : &r)) {
        step_control->SetNoiseParameters(s, r);
      }
    };
    auto reset = [step_control, is_scale] {
      float s = 0, r = 0;
      step_control->GetNoiseParameters(s, r);
      (is_scale ? s : r) = 0.0f;
      step_control->SetNoiseParameters(s, r);
    };
    RegisterCustomField(out, path, draw, reset, cur != 0.0f, "0");
  };
  add(prefix + ".scale", true);
  add(prefix + ".rate", false);
}

}  // namespace mujoco::platform
