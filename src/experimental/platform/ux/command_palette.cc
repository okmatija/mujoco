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

// Implementation of the command palette. The public surface (command_palette.h)
// is a flat C ABI; everything here is plain Dear ImGui-style C++ -- ImVector and
// const char* rather than the STL, no std::function and no exceptions -- kept
// entirely inside this translation unit so it never leaks across the ABI.

#include "experimental/platform/ux/command_palette.h"

#include <algorithm>
#include <cctype>
#include <cfloat>
#include <cstddef>
#include <cstring>

#include <imgui.h>

// FontAwesome 4 glyphs the host font merges in the 0xf000-0xf3ff range.
#define ICON_COG "\xEF\x80\x93"     // fa-cog (U+F013): the settings toggle.
#define ICON_REVERT "\xEF\x83\xA2"  // fa-undo (U+F0E2): the revert-to-default button.

// ----------------------------------------------------------------------------
// Arena + entry storage (members of the externally-visible opaque types, so they
// have file-scope linkage rather than living in the anonymous namespace).
// ----------------------------------------------------------------------------

// A bump allocator backing one ImCmdList. It hands out stable pointers for the
// strings and field bindings the list copies, and Reset() rewinds it for the
// next frame without freeing (so the chunks are reused, not re-malloc'd).
struct CmdArena {
  struct Chunk { char* data; int cap; };
  ImVector<Chunk> chunks;
  int cur = 0;   // index of the chunk we are filling
  int used = 0;  // bytes used in chunks[cur]
  static const int kChunk = 16 * 1024;

  ~CmdArena() { FreeAll(); }

  void Reset() { cur = 0; used = 0; }
  void FreeAll() {
    for (Chunk& c : chunks) IM_FREE(c.data);
    chunks.clear();
    cur = 0;
    used = 0;
  }

  void* Alloc(int size, int align) {
    if (size <= 0) return nullptr;
    for (;;) {
      if (cur >= chunks.Size) {
        const int cap = size > kChunk ? size : kChunk;
        Chunk c = {(char*)IM_ALLOC((size_t)cap), cap};
        chunks.push_back(c);
        used = 0;
      }
      Chunk& c = chunks[cur];
      const int off = (used + (align - 1)) & ~(align - 1);
      if (off + size <= c.cap) {
        used = off + size;
        return c.data + off;
      }
      ++cur;  // doesn't fit; move to (or create) the next chunk.
      used = 0;
    }
  }
  char* StrDup(const char* s) {
    if (s == nullptr) return nullptr;
    const int n = (int)std::strlen(s) + 1;
    char* p = (char*)Alloc(n, 1);
    std::memcpy(p, s, n);
    return p;
  }
  char* StrCat2(const char* a, const char* b) {
    const int na = (int)std::strlen(a), nb = (int)std::strlen(b);
    char* p = (char*)Alloc(na + nb + 1, 1);
    std::memcpy(p, a, na);
    std::memcpy(p + na, b, nb);
    p[na + nb] = '\0';
    return p;
  }
};

// One stored command: the POD record (with strings copied into the arena) plus a
// stable id used to scope the value widget's ImGui ID independently of the '*'
// marker, which comes and goes as the value is edited.
struct CmdEntry {
  const char* id;
  ImCmd cmd;
};

// ----------------------------------------------------------------------------
// Matching, highlighting, and the field-binding callbacks (TU-local).
// ----------------------------------------------------------------------------
namespace {

char Lower(char c) { return (char)std::tolower((unsigned char)c); }
bool CharEq(char a, char b, bool ci) {
  return ci ? Lower(a) == Lower(b) : a == b;
}
int CompareNoCase(const char* a, const char* b) {
  for (;; ++a, ++b) {
    const int ca = Lower(*a), cb = Lower(*b);
    if (ca != cb) return ca - cb;
    if (ca == 0) return 0;
  }
}
void CopyStr(char* dst, int cap, const char* src) {
  int i = 0;
  for (; src[i] != '\0' && i < cap - 1; ++i) dst[i] = src[i];
  dst[i] = '\0';
}

// Does `text` match `query` under `mode`? An empty query always matches.
bool MatchQuery(const char* text, const char* query, ImCmdSearchMode mode,
                bool ci) {
  if (query == nullptr || query[0] == '\0') return true;
  const int tn = (int)std::strlen(text), qn = (int)std::strlen(query);
  switch (mode) {
    case ImCmdSearchMode_Prefix:
      if (tn < qn) return false;
      for (int i = 0; i < qn; ++i) {
        if (!CharEq(text[i], query[i], ci)) return false;
      }
      return true;
    case ImCmdSearchMode_Substring:
      for (int s = 0; s + qn <= tn; ++s) {
        int j = 0;
        for (; j < qn && CharEq(text[s + j], query[j], ci); ++j) {
        }
        if (j == qn) return true;
      }
      return false;
    case ImCmdSearchMode_Fuzzy: {
      int qi = 0;
      for (int ti = 0; ti < tn && qi < qn; ++ti) {
        if (CharEq(text[ti], query[qi], ci)) ++qi;
      }
      return qi == qn;
    }
  }
  return false;
}

// Which column of a completion row a click landed in.
enum RowHit { RowHit_None, RowHit_Name, RowHit_Value };

// ===== Match highlighting ==================================================
// Marks which characters of `name` matched the query (so they can be drawn in a
// bold font). Self-contained: deleting this block plus its two call sites and the
// `match_font` branch in CompletionRow removes the feature.

void MatchedChars(const char* name, const char* query, ImCmdSearchMode mode,
                  bool ci, ImVector<bool>& out) {
  const int n = (int)std::strlen(name);
  out.resize(n);
  for (int i = 0; i < n; ++i) out[i] = false;
  if (query == nullptr || query[0] == '\0') return;
  const int qn = (int)std::strlen(query);
  switch (mode) {
    case ImCmdSearchMode_Prefix:
      for (int i = 0; i < qn && i < n; ++i) out[i] = true;
      break;
    case ImCmdSearchMode_Substring:
      for (int s = 0; s + qn <= n; ++s) {
        int j = 0;
        for (; j < qn && CharEq(name[s + j], query[j], ci); ++j) {
        }
        if (j == qn) {
          for (int k = 0; k < qn; ++k) out[s + k] = true;
          break;  // bold only the first occurrence.
        }
      }
      break;
    case ImCmdSearchMode_Fuzzy: {
      int qi = 0;
      for (int i = 0; i < n && qi < qn; ++i) {
        if (CharEq(name[i], query[qi], ci)) {
          out[i] = true;
          ++qi;
        }
      }
      break;
    }
  }
}

// Lays out `name` in runs of adjacent characters that share a font -- matched
// characters (per `hit`) use `bold`, the rest `regular`. Draws at `pos` when `dl`
// is non-null; pass dl=null to only measure. Returns the total width.
float HighlightedName(ImDrawList* dl, ImVec2 pos, const char* name,
                      const bool* hit, int n, ImFont* regular, ImFont* bold,
                      float size, ImU32 col) {
  float x = pos.x;
  for (int i = 0; i < n;) {
    const bool emphasized = hit[i];
    int j = i + 1;
    while (j < n && hit[j] == emphasized) ++j;
    ImFont* f = emphasized ? bold : regular;
    if (dl != nullptr) {
      dl->AddText(f, size, ImVec2(x, pos.y), col, name + i, name + j);
    }
    x += f->CalcTextSizeA(size, FLT_MAX, 0.0f, name + i, name + j).x;
    i = j;
  }
  return x - pos.x;
}
// ===========================================================================

// ----- Built-in field bindings ----------------------------------------------
// Each field helper stores one of these in the arena and points the command's
// callbacks at the matching static functions, with `user` = the binding. The
// value widgets use the constant label "##v": CompletionRow scopes them under a
// PushID keyed on the field's stable id, so that is unique and marker-stable.

// A numeric scalar/array (T = int/float/double) bound to live memory.
struct NumBinding {
  void* ptr;        // live field memory
  const void* def;  // arena copy of the default value(s)
  int count;        // 1 for a scalar, n for an array
  int size;         // sizeof(T)
  ImGuiDataType dt;
};
void NumDraw(void* u) {
  NumBinding* b = (NumBinding*)u;
  ImGui::InputScalarN("##v", b->dt, b->ptr, b->count);
}
void NumReset(void* u) {
  NumBinding* b = (NumBinding*)u;
  std::memcpy(b->ptr, b->def, (size_t)b->size * b->count);
}

// A boolean reached through a getter/setter (covers bitfields and byte flags).
struct FlagBinding {
  ImCmdGetBoolCallback get;
  ImCmdSetBoolCallback set;
  void* user;
  bool def;
};
void FlagRun(void* u) {
  FlagBinding* b = (FlagBinding*)u;
  b->set(b->user, !b->get(b->user));
}
void FlagCycle(void* u, int /*delta*/) { FlagRun(u); }
void FlagReset(void* u) {
  FlagBinding* b = (FlagBinding*)u;
  b->set(b->user, b->def);
}
void FlagDraw(void* u) {
  FlagBinding* b = (FlagBinding*)u;
  bool v = b->get(b->user);
  // Right-align the checkbox, leaving the revert-button slot on the far right (a
  // flag always has a revert action) so the box keeps its place whether or not
  // the '*' marker is showing.
  const ImGuiStyle& style = ImGui::GetStyle();
  const float reserve = ImGui::CalcTextSize(ICON_REVERT).x +
                        style.FramePadding.x * 2.0f + style.ItemSpacing.x;
  const float offset =
      ImGui::GetContentRegionAvail().x - reserve - ImGui::GetFrameHeight();
  if (offset > 0.0f) {
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + offset);
  }
  if (ImGui::Checkbox("##v", &v)) b->set(b->user, v);
}
bool BoolPtrGet(void* u) { return *(bool*)u; }
void BoolPtrSet(void* u, bool v) { *(bool*)u = v; }

// An int-backed enum: a combo whose entries also advance under Enter/Left-Right.
struct EnumBinding {
  int* ptr;
  const char* const* names;
  int count;
  int def;
};
void EnumDraw(void* u) {
  EnumBinding* b = (EnumBinding*)u;
  int v = *b->ptr;
  if (ImGui::Combo("##v", &v, b->names, b->count)) *b->ptr = v;
}
void EnumCycle(void* u, int delta) {
  EnumBinding* b = (EnumBinding*)u;
  const int n = b->count;
  *b->ptr = ((*b->ptr + delta) % n + n) % n;
}
void EnumRun(void* u) { EnumCycle(u, 1); }
void EnumReset(void* u) {
  EnumBinding* b = (EnumBinding*)u;
  *b->ptr = b->def;
}

// Draws one completion row: a full-width selectable for the background /
// highlight / click, then the value at `desc_x` (a window-local x so values line
// up in a column) -- either draw_value's editable widget (with a revert button
// past it when modified) or the dimmed `description`. When `match_font` is set
// the name is drawn with matched characters in that bold font. Returns which
// column was clicked (RowHit_None if not clicked).
RowHit CompletionRow(const CmdEntry& e, bool selected, float desc_x,
                     const char* query, ImCmdSearchMode mode, bool ci,
                     ImFont* match_font, bool focus_value,
                     ImVector<bool>& scratch) {
  const ImCmd& cmd = e.cmd;
  // AllowOverlap so an editable value widget drawn over the row's selectable
  // still receives its clicks.
  const ImGuiSelectableFlags sel_flags = ImGuiSelectableFlags_AllowOverlap;
  bool clicked;
  if (match_font != nullptr) {
    // An empty selectable provides the background / highlight / click; the name
    // is drawn on top so matched characters can use the bold font.
    const ImVec2 name_pos = ImGui::GetCursorScreenPos();
    ImGui::PushID(cmd.name);
    clicked = ImGui::Selectable(
        "##row", selected, sel_flags,
        ImVec2(ImGui::GetContentRegionAvail().x, ImGui::GetTextLineHeight()));
    ImGui::PopID();
    MatchedChars(cmd.name, query, mode, ci, scratch);
    HighlightedName(ImGui::GetWindowDrawList(), name_pos, cmd.name,
                    scratch.Data, scratch.Size, ImGui::GetFont(), match_font,
                    ImGui::GetFontSize(), ImGui::GetColorU32(ImGuiCol_Text));
  } else {
    clicked = ImGui::Selectable(cmd.name, selected, sel_flags);
  }

  float value_x = FLT_MAX;  // screen-x where the value column begins.
  if (cmd.draw_value != nullptr) {
    ImGui::SameLine(desc_x);
    value_x = ImGui::GetCursorScreenPos().x;
    // Scope the value widget + revert button on the field's stable id so editing
    // isn't disrupted when the '*' marker (part of the name) appears/disappears.
    ImGui::PushID(e.id);
    const float revert_w =
        cmd.reset ? ImGui::CalcTextSize(ICON_REVERT).x +
                        ImGui::GetStyle().FramePadding.x * 2.0f
                  : 0.0f;
    ImGui::SetNextItemWidth(revert_w > 0.0f
                                ? -(revert_w + ImGui::GetStyle().ItemSpacing.x)
                                : -FLT_MIN);
    if (focus_value) {
      ImGui::SetKeyboardFocusHere();  // Right entered the widget; focus it.
    }
    cmd.draw_value(cmd.user);
    // Revert-to-default button, right-aligned into a fixed last column so every
    // revert button lines up regardless of the value widget's width.
    if (cmd.modified && cmd.reset != nullptr) {
      ImGui::SameLine();
      const float avail = ImGui::GetContentRegionAvail().x;
      if (avail > revert_w) {
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + avail - revert_w);
      }
      if (ImGui::Button(ICON_REVERT "##revert")) cmd.reset(cmd.user);
      if (cmd.default_text != nullptr && cmd.default_text[0] != '\0') {
        ImGui::SetItemTooltip("Reset to '%s'", cmd.default_text);
      } else {
        ImGui::SetItemTooltip("Reset to default");
      }
    }
    ImGui::PopID();
  } else if (cmd.description != nullptr && cmd.description[0] != '\0') {
    ImGui::SameLine(desc_x);
    value_x = ImGui::GetCursorScreenPos().x;
    ImGui::TextDisabled("%s", cmd.description);
  }
  if (!clicked) return RowHit_None;
  return ImGui::GetMousePos().x >= value_x ? RowHit_Value : RowHit_Name;
}

}  // namespace

// ----------------------------------------------------------------------------
// The opaque list type, then the field builders that fill it (TU-local helpers
// that need the list definition, so they follow it).
// ----------------------------------------------------------------------------

struct ImCmdList {
  CmdArena arena;
  ImVector<CmdEntry> entries;
};

namespace {

template <class T>
ImGuiDataType NumDataType();
template <>
ImGuiDataType NumDataType<int>() { return ImGuiDataType_S32; }
template <>
ImGuiDataType NumDataType<float>() { return ImGuiDataType_Float; }
template <>
ImGuiDataType NumDataType<double>() { return ImGuiDataType_Double; }

template <class T>
void AppendNum(ImGuiTextBuffer& buf, T v);
template <>
void AppendNum<int>(ImGuiTextBuffer& buf, int v) { buf.appendf("%d", v); }
template <>
void AppendNum<float>(ImGuiTextBuffer& buf, float v) { buf.appendf("%g", v); }
template <>
void AppendNum<double>(ImGuiTextBuffer& buf, double v) { buf.appendf("%g", v); }

// Appends "*" to a modified field's path for display + matching (so typing "*"
// filters to changed fields). The widget's id stays keyed on the bare path.
const char* MarkedName(CmdArena& a, const char* path, bool modified) {
  return modified ? a.StrCat2(path, "*") : path;
}

template <class T>
void AddNumField(ImCmdList* list, const char* path, T* ptr, int count,
                 const T* def) {
  CmdArena& a = list->arena;
  const char* p = a.StrDup(path);
  // Copy the defaults by value: callers often pass a pointer into a per-frame
  // local, but reset runs later, on a click.
  T* defc = (T*)a.Alloc((int)sizeof(T) * count, (int)alignof(T));
  std::memcpy(defc, def, sizeof(T) * (size_t)count);

  bool modified = false;
  ImGuiTextBuffer def_text;
  for (int i = 0; i < count; ++i) {
    if (ptr[i] != def[i]) modified = true;
    if (i != 0) def_text.append(" ");
    AppendNum<T>(def_text, def[i]);
  }

  NumBinding* b = (NumBinding*)a.Alloc((int)sizeof(NumBinding),
                                       (int)alignof(NumBinding));
  b->ptr = ptr;
  b->def = defc;
  b->count = count;
  b->size = (int)sizeof(T);
  b->dt = NumDataType<T>();

  CmdEntry e = {};
  e.id = p;
  e.cmd.name = MarkedName(a, p, modified);
  e.cmd.default_text = a.StrDup(def_text.c_str());
  e.cmd.draw_value = NumDraw;
  e.cmd.reset = NumReset;
  e.cmd.user = b;
  e.cmd.modified = modified;
  list->entries.push_back(e);
}

}  // namespace

// ----------------------------------------------------------------------------
// The opaque palette type and its (TU-local) drawing methods.
// ----------------------------------------------------------------------------

struct ImCmdPalette {
  bool open = false;
  bool focus_input = false;
  // One-shot after Open()/OpenWith(): on the frame the input gains focus, move
  // the cursor to the end and clear the selection so a pre-filled string isn't
  // select-all'd and wiped by the first keystroke.
  bool init_cursor_end = false;
  int selection = 0;
  // True once the user has pressed Down to move into the list; while set,
  // Left/Right act on the highlighted command's value instead of the text
  // cursor. Reset whenever the query text changes.
  bool in_list = false;
  // One-shot: Right on a selected value-input row focuses that widget next frame.
  bool focus_value = false;
  char last_query[256] = "";  // query from the previous Draw, to detect edits.
  char input[256] = "";
  ImVec2 center = ImVec2(0.0f, 0.0f);
  // Points at the command list during Draw so the Tab-completion callback can
  // see it (it runs inside InputText, before the list is filtered).
  const ImCmdList* completion_list = nullptr;
  ImCmdSearchMode search_mode = ImCmdSearchMode_Fuzzy;
  bool case_insensitive = true;
  bool highlight_matches = true;  // bold the matched characters in the list.
  bool show_settings = false;     // cog panel replaces the completion list.
  ImVector<bool> hit;             // reused scratch for match highlighting.

  void Render(const ImCmdList* list, const ImCmdDrawDesc* desc);
  const CmdEntry* DrawCompletionList(const ImCmdList* list, const char* query,
                                     bool entered);
  void DrawSettings(const ImCmdDrawDesc* desc);
  void SubmitPlain(const char* text, const ImCmdDrawDesc* desc);
  static int InputTextCallback(ImGuiInputTextCallbackData* data);
};

namespace {

// Tab-completion: extend `input` to the next dotted segment shared by every
// command that has `input` as a (case-insensitive) prefix -- e.g. "mjModel.opt.g"
// against the mjModel.opt.* entries. Writes the result (canonical casing) into
// `out` and returns true when it extended `input`.
bool SegmentComplete(const char* input, const ImCmdList* list, char* out,
                     int out_cap) {
  const int in_len = (int)std::strlen(input);
  int lcp_len = -1;  // -1 until the first prefix match seeds `out`.
  for (const CmdEntry& e : list->entries) {
    const char* name = e.cmd.name;
    const int nlen = (int)std::strlen(name);
    if (nlen < in_len) continue;
    bool is_prefix = true;
    for (int i = 0; i < in_len; ++i) {
      if (Lower(name[i]) != Lower(input[i])) {
        is_prefix = false;
        break;
      }
    }
    if (!is_prefix) continue;
    if (lcp_len < 0) {
      int n = nlen < out_cap - 1 ? nlen : out_cap - 1;
      std::memcpy(out, name, n);
      out[n] = '\0';
      lcp_len = n;
    } else {
      int n = 0;
      while (n < lcp_len && n < nlen && Lower(out[n]) == Lower(name[n])) ++n;
      lcp_len = n;
      out[lcp_len] = '\0';
    }
  }
  if (lcp_len < 0) return false;  // nothing has `input` as a prefix.
  // Stop at the first '.' at or past the typed length (complete one segment).
  for (int i = in_len; i < lcp_len; ++i) {
    if (out[i] == '.') {
      lcp_len = i + 1;
      out[lcp_len] = '\0';
      break;
    }
  }
  return lcp_len > in_len;
}

}  // namespace

int ImCmdPalette::InputTextCallback(ImGuiInputTextCallbackData* data) {
  ImCmdPalette* self = (ImCmdPalette*)data->UserData;
  // CallbackAlways: one-shot after Open()/OpenWith() to undo the select-all that
  // focus applies, so a pre-filled string stays put with the cursor after it.
  if (data->EventFlag == ImGuiInputTextFlags_CallbackAlways) {
    if (self->init_cursor_end) {
      data->CursorPos = data->BufTextLen;
      data->SelectionStart = data->SelectionEnd = data->CursorPos;
      self->init_cursor_end = false;
    }
  } else if (data->EventFlag == ImGuiInputTextFlags_CallbackCompletion) {
    // Tab: complete the current dotted segment against the command list.
    if (self->completion_list != nullptr) {
      char completed[256];
      if (SegmentComplete(data->Buf, self->completion_list, completed,
                          IM_ARRAYSIZE(completed))) {
        data->DeleteChars(0, data->BufTextLen);
        data->InsertChars(0, completed);
      }
    }
  }
  return 0;
}

const CmdEntry* ImCmdPalette::DrawCompletionList(const ImCmdList* list,
                                                 const char* query,
                                                 bool entered) {
  // Filter by the current search mode, then present alphabetically (case-
  // insensitive) so the list stays stable as the user types.
  ImVector<const CmdEntry*> matches;
  if (list != nullptr) {
    for (const CmdEntry& e : list->entries) {
      if (MatchQuery(e.cmd.name, query, search_mode, case_insensitive)) {
        matches.push_back(&e);
      }
    }
  }
  if (matches.Size > 1) {
    std::sort(matches.Data, matches.Data + matches.Size,
              [](const CmdEntry* a, const CmdEntry* b) {
                return CompareNoCase(a->cmd.name, b->cmd.name) < 0;
              });
  }

  // Editing the query drops out of list-navigation mode (so Left/Right move the
  // text cursor again) and re-anchors the selection at the top match.
  const bool query_changed = std::strcmp(query, last_query) != 0;
  if (query_changed) {
    in_list = false;
    selection = 0;
    CopyStr(last_query, IM_ARRAYSIZE(last_query), query);
  }

  // Keyboard navigation. The cursor stays in the input until Down first enters
  // the list; after that Down/Up move the selection, and Up from the top row
  // returns focus to the input. `moved` keeps the selection scrolled into view.
  bool moved = query_changed;
  if (ImGui::IsKeyPressed(ImGuiKey_DownArrow)) {
    if (in_list) {
      ++selection;
    } else {
      in_list = true;
      selection = 0;
    }
    moved = true;
  }
  if (ImGui::IsKeyPressed(ImGuiKey_UpArrow) && in_list) {
    if (selection == 0) {
      in_list = false;  // back to the input text
    } else {
      --selection;
    }
    moved = true;
  }
  if (matches.Size > 0) {
    const int n = matches.Size;
    selection = (selection % n + n) % n;
  } else {
    selection = 0;
  }

  // In list mode, the highlighted command's value reacts to Left/Right without
  // running it or closing: a `cycle` value steps in place; otherwise Right gives
  // a numeric input keyboard focus so it can be typed.
  if (in_list && matches.Size > 0) {
    const ImCmd& sel = matches[selection]->cmd;
    if (sel.cycle != nullptr) {
      if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow)) {
        sel.cycle(sel.user, -1);
      } else if (ImGui::IsKeyPressed(ImGuiKey_RightArrow)) {
        sel.cycle(sel.user, 1);
      }
    } else if (sel.draw_value != nullptr &&
               ImGui::IsKeyPressed(ImGuiKey_RightArrow)) {
      focus_value = true;
    }
  }

  if (matches.Size == 0) {
    return nullptr;  // nothing to show (e.g. mid-typing "/prompt question").
  }

  // Bold weight for match highlighting (the font the app loads after the
  // default); null when off, which selects the plain path.
  const ImVector<ImFont*>& fonts = ImGui::GetIO().Fonts->Fonts;
  ImFont* match_font =
      (highlight_matches && fonts.Size > 1) ? fonts[1] : nullptr;

  // Column layout: the value column begins just past the widest name.
  const float spacing = ImGui::GetStyle().ItemSpacing.x;
  float name_w = 0.0f;
  for (const CmdEntry* e : matches) {
    float w;
    if (match_font != nullptr) {
      MatchedChars(e->cmd.name, query, search_mode, case_insensitive, hit);
      w = HighlightedName(nullptr, ImVec2(0, 0), e->cmd.name, hit.Data,
                          hit.Size, ImGui::GetFont(), match_font,
                          ImGui::GetFontSize(), 0);
    } else {
      w = ImGui::CalcTextSize(e->cmd.name).x;
    }
    name_w = std::max(name_w, w);
  }
  const float desc_x = name_w + spacing * 3.0f;

  ImGui::Separator();

  // The list scrolls inside a child so the whole window stays under the 80% cap
  // set in Render(); leave room for the input row above.
  const float max_h = ImGui::GetMainViewport()->WorkSize.y * 0.8f -
                      ImGui::GetFrameHeightWithSpacing() * 2.0f;
  ImGui::SetNextWindowSizeConstraints(ImVec2(0, 0), ImVec2(FLT_MAX, max_h));
  const CmdEntry* chosen = nullptr;
  if (ImGui::BeginChild(
          "##completions", ImVec2(0, 0),
          ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysUseWindowPadding)) {
    // Make a held (pressed) row use the hover colour rather than the darker
    // "active" colour.
    ImGui::PushStyleColor(ImGuiCol_HeaderActive,
                          ImGui::GetStyleColorVec4(ImGuiCol_HeaderHovered));
    for (int i = 0; i < matches.Size; ++i) {
      const CmdEntry* e = matches[i];
      const bool at_selection = (i == selection);
      const float row_min = ImGui::GetCursorPosY();
      // Only highlight once the user has entered the list; before that the
      // selection is invisible but still the Enter target (top match).
      const RowHit hit_col = CompletionRow(
          *e, in_list && at_selection, desc_x, query, search_mode,
          case_insensitive, match_font, focus_value && at_selection, hit);
      if (hit_col != RowHit_None) {
        // A click moves the list focus here (and refocuses the input); it never
        // runs or closes. A click on the value column also cycles it in place.
        selection = i;
        in_list = true;
        focus_input = true;
        if (hit_col == RowHit_Value && e->cmd.cycle != nullptr) {
          e->cmd.cycle(e->cmd.user, 1);
        }
      }
      // Enter (from the input) runs/submits the selected row; clicks never do.
      if (at_selection && entered) {
        chosen = e;
      }
      // Keep the selection visible, scrolling only once it reaches an edge.
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
  focus_value = false;  // one-shot, consumed by the selected row above.
  return chosen;
}

void ImCmdPalette::DrawSettings(const ImCmdDrawDesc* desc) {
  ImGui::Separator();

  // The palette window is translucent; give the settings panel a solid
  // background so its controls read clearly.
  ImVec4 bg = ImGui::GetStyleColorVec4(ImGuiCol_WindowBg);
  bg.w = 1.0f;
  ImGui::PushStyleColor(ImGuiCol_ChildBg, bg);
  ImGui::BeginChild(
      "##settings", ImVec2(0, 0),
      ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysUseWindowPadding);

  if (ImGui::CollapsingHeader("Command Palette Settings",
                              ImGuiTreeNodeFlags_DefaultOpen)) {
    const char* kModes[] = {"Prefix", "Substring", "Fuzzy"};
    int mode = (int)search_mode;
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x * 0.5f);
    if (ImGui::Combo("Search Mode", &mode, kModes, IM_ARRAYSIZE(kModes))) {
      search_mode = (ImCmdSearchMode)mode;
    }
    ImGui::Checkbox("Case Insensitive", &case_insensitive);
    ImGui::Checkbox("Highlight Matches", &highlight_matches);
  }

  // Host- (and plugin-) provided settings go below the palette's own.
  if (desc->render_settings != nullptr) {
    ImGui::Spacing();
    desc->render_settings(desc->render_settings_user);
  }

  ImGui::EndChild();
  ImGui::PopStyleColor();
}

void ImCmdPalette::SubmitPlain(const char* text, const ImCmdDrawDesc* desc) {
  if (text == nullptr || text[0] == '\0') return;
  if (desc->on_submit_plain != nullptr) {
    desc->on_submit_plain(desc->on_submit_plain_user, text);
  }
  input[0] = '\0';
  focus_input = true;
}

void ImCmdPalette::Render(const ImCmdList* list, const ImCmdDrawDesc* desc) {
  if (!open) return;

  const float kWidth = 480.0f;
  // Centered horizontally; the caller supplies the top edge (rect.y).
  ImGui::SetNextWindowPos(
      ImVec2(desc->rect.x + desc->rect.w * 0.5f, desc->rect.y),
      ImGuiCond_Always, ImVec2(0.5f, 0.0f));
  // Fixed width; height grows with the list but is capped at 80% of the viewport
  // (the list scrolls within its own child past that).
  const float max_h = ImGui::GetMainViewport()->WorkSize.y * 0.8f;
  ImGui::SetNextWindowSizeConstraints(ImVec2(kWidth, 0), ImVec2(kWidth, max_h));
  // Translucent, matching the other overlays. The input box keeps its own opaque
  // frame colour; the completion list shows the scene through.
  ImGui::SetNextWindowBgAlpha(0.65f);

  const ImGuiWindowFlags flags =
      ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
      ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoDocking |
      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoSavedSettings |
      ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoNavInputs;

  if (ImGui::Begin("##CommandPalette", nullptr, flags)) {
    center = ImVec2(ImGui::GetWindowPos().x + ImGui::GetWindowSize().x * 0.5f,
                    ImGui::GetWindowPos().y + ImGui::GetWindowSize().y * 0.5f);
    if (focus_input) {
      ImGui::SetKeyboardFocusHere();
      focus_input = false;
    }
    // Input box, leaving room on the right for the cog settings toggle.
    const float cog_w = ImGui::GetFrameHeight();  // square button
    ImGui::SetNextItemWidth(-(cog_w + ImGui::GetStyle().ItemSpacing.x));
    completion_list = list;  // for the Tab-completion callback.
    const bool entered = ImGui::InputTextWithHint(
        "##cmdinput", "Type to search  ( > UI   . model/data   / agent )",
        input, IM_ARRAYSIZE(input),
        ImGuiInputTextFlags_EnterReturnsTrue |
            ImGuiInputTextFlags_CallbackAlways |
            ImGuiInputTextFlags_CallbackCompletion,
        InputTextCallback, this);

    // Cog toggle: opens the settings panel in place of the completion list.
    // Capture the tint state before the button, since clicking it flips
    // show_settings (the push and pop must use the same value).
    ImGui::SameLine();
    const bool cog_active = show_settings;
    if (cog_active) {
      ImGui::PushStyleColor(ImGuiCol_Button,
                            ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
    }
    if (ImGui::Button(ICON_COG, ImVec2(cog_w, cog_w))) {
      show_settings = !show_settings;
      focus_input = true;  // keep typing working after toggling.
    }
    if (cog_active) {
      ImGui::PopStyleColor();
    }
    ImGui::SetItemTooltip("Command Palette Preferences");

    // Only show the autocomplete list / conversation while the palette is
    // focused. Clicking outside ImGui defocuses it, collapsing the window to
    // just the input box without losing what the user typed.
    const bool focused =
        ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows);

    if (show_settings) {
      DrawSettings(desc);
    } else if (focused) {
      // One unified list, matched against the whole input. A leading '>'/'.'/'/'
      // only matches names with that prefix, narrowing by context.
      if (input[0] != '\0') {
        if (const CmdEntry* chosen = DrawCompletionList(list, input, entered)) {
          if (chosen->cmd.name[0] == '/') {
            // Agent command: submit its text so the caller can route it.
            SubmitPlain(chosen->cmd.name, desc);
          } else {
            // '>' UI / '.' model-data: run the action.
            if (chosen->cmd.run != nullptr) chosen->cmd.run(chosen->cmd.user);
            open = false;
          }
        } else if (entered && input[0] == '/') {
          // Typed agent text with arguments that matched no completion (e.g.
          // "/prompt how do I..." or "/model sonnet"): submit it as-is.
          SubmitPlain(input, desc);
        }
      }

      // The agent conversation renders in agent context: while typing a '/'
      // command, and once a submitted prompt has cleared the box (empty input).
      if (desc->render_below != nullptr &&
          (input[0] == '/' || input[0] == '\0')) {
        desc->render_below(desc->render_below_user);
      }
    }

    if (ImGui::IsKeyPressed(ImGuiKey_Escape)) {
      // Escape backs out of settings first, then closes the palette.
      if (show_settings) {
        show_settings = false;
        focus_input = true;
      } else {
        open = false;
      }
    }
  }
  ImGui::End();
}

// ----------------------------------------------------------------------------
// Public C ABI.
// ----------------------------------------------------------------------------
extern "C" {

ImCmdPalette* ImCmdPalette_Create(void) { return IM_NEW(ImCmdPalette)(); }
void ImCmdPalette_Destroy(ImCmdPalette* p) { IM_DELETE(p); }

void ImCmdPalette_Open(ImCmdPalette* p) {
  p->open = true;
  p->focus_input = true;
  p->selection = 0;
  p->in_list = false;
  p->show_settings = false;  // a fresh open shows the command list.
  p->last_query[0] = '\0';
  // Start empty (type to search). OpenWith() may pre-fill afterwards; the cursor
  // is then parked at the end so a pre-filled string isn't select-all'd.
  p->input[0] = '\0';
  p->init_cursor_end = true;
}
void ImCmdPalette_OpenWith(ImCmdPalette* p, const char* text) {
  ImCmdPalette_Open(p);
  CopyStr(p->input, IM_ARRAYSIZE(p->input), text);
}
void ImCmdPalette_SetText(ImCmdPalette* p, const char* text) {
  CopyStr(p->input, IM_ARRAYSIZE(p->input), text);
}
void ImCmdPalette_Close(ImCmdPalette* p) { p->open = false; }
void ImCmdPalette_Toggle(ImCmdPalette* p) {
  if (p->open) {
    p->open = false;
  } else {
    ImCmdPalette_Open(p);
  }
}
bool ImCmdPalette_IsOpen(const ImCmdPalette* p) { return p->open; }
ImCmdVec2 ImCmdPalette_WindowCenter(const ImCmdPalette* p) {
  ImCmdVec2 v = {p->center.x, p->center.y};
  return v;
}

void ImCmdPalette_SetSearchMode(ImCmdPalette* p, ImCmdSearchMode mode) {
  p->search_mode = mode;
}
ImCmdSearchMode ImCmdPalette_GetSearchMode(const ImCmdPalette* p) {
  return p->search_mode;
}
void ImCmdPalette_SetCaseInsensitive(ImCmdPalette* p, bool enabled) {
  p->case_insensitive = enabled;
}
bool ImCmdPalette_GetCaseInsensitive(const ImCmdPalette* p) {
  return p->case_insensitive;
}

void ImCmdPalette_Draw(ImCmdPalette* p, const ImCmdList* list,
                       const ImCmdDrawDesc* desc) {
  p->Render(list, desc);
}

ImCmdList* ImCmdList_Create(void) { return IM_NEW(ImCmdList)(); }
void ImCmdList_Destroy(ImCmdList* list) { IM_DELETE(list); }
void ImCmdList_Clear(ImCmdList* list) {
  list->entries.clear();
  list->arena.Reset();
}

void ImCmdList_Add(ImCmdList* list, const ImCmd* cmd) {
  CmdArena& a = list->arena;
  CmdEntry e = {};
  e.cmd = *cmd;
  e.cmd.name = a.StrDup(cmd->name);
  e.cmd.description = a.StrDup(cmd->description);
  e.cmd.default_text = a.StrDup(cmd->default_text);
  e.id = e.cmd.name;  // host entries carry no '*' marker, so id == name.
  list->entries.push_back(e);
}

void ImCmdList_AddFieldS32(ImCmdList* list, const char* path, int* ptr,
                           int def) {
  AddNumField<int>(list, path, ptr, 1, &def);
}
void ImCmdList_AddFieldF32(ImCmdList* list, const char* path, float* ptr,
                           float def) {
  AddNumField<float>(list, path, ptr, 1, &def);
}
void ImCmdList_AddFieldF64(ImCmdList* list, const char* path, double* ptr,
                           double def) {
  AddNumField<double>(list, path, ptr, 1, &def);
}
void ImCmdList_AddFieldArrayS32(ImCmdList* list, const char* path, int* ptr,
                                int count, const int* def) {
  AddNumField<int>(list, path, ptr, count, def);
}
void ImCmdList_AddFieldArrayF32(ImCmdList* list, const char* path, float* ptr,
                                int count, const float* def) {
  AddNumField<float>(list, path, ptr, count, def);
}
void ImCmdList_AddFieldArrayF64(ImCmdList* list, const char* path, double* ptr,
                                int count, const double* def) {
  AddNumField<double>(list, path, ptr, count, def);
}

void ImCmdList_AddFieldFlag(ImCmdList* list, const char* path,
                            ImCmdGetBoolCallback get, ImCmdSetBoolCallback set,
                            void* user, bool def) {
  CmdArena& a = list->arena;
  const char* p = a.StrDup(path);
  const bool modified = get(user) != def;
  FlagBinding* b =
      (FlagBinding*)a.Alloc((int)sizeof(FlagBinding), (int)alignof(FlagBinding));
  b->get = get;
  b->set = set;
  b->user = user;
  b->def = def;

  CmdEntry e = {};
  e.id = p;
  e.cmd.name = MarkedName(a, p, modified);
  e.cmd.default_text = def ? "on" : "off";  // string literals; no copy needed.
  e.cmd.run = FlagRun;
  e.cmd.cycle = FlagCycle;
  e.cmd.draw_value = FlagDraw;
  e.cmd.reset = FlagReset;
  e.cmd.user = b;
  e.cmd.modified = modified;
  list->entries.push_back(e);
}
void ImCmdList_AddFieldFlagPtr(ImCmdList* list, const char* path, bool* ptr,
                               bool def) {
  ImCmdList_AddFieldFlag(list, path, BoolPtrGet, BoolPtrSet, ptr, def);
}

void ImCmdList_AddFieldEnum(ImCmdList* list, const char* path, int* ptr,
                            const char* const* names, int count, int def) {
  CmdArena& a = list->arena;
  const char* p = a.StrDup(path);
  // Copy both the pointer array and the strings, so callers may pass temporaries.
  const char** names_copy = (const char**)a.Alloc(
      (int)sizeof(char*) * count, (int)alignof(char*));
  for (int i = 0; i < count; ++i) names_copy[i] = a.StrDup(names[i]);
  const bool modified = *ptr != def;

  EnumBinding* b =
      (EnumBinding*)a.Alloc((int)sizeof(EnumBinding), (int)alignof(EnumBinding));
  b->ptr = ptr;
  b->names = names_copy;
  b->count = count;
  b->def = def;

  CmdEntry e = {};
  e.id = p;
  e.cmd.name = MarkedName(a, p, modified);
  e.cmd.default_text =
      (def >= 0 && def < count) ? names_copy[def] : "";
  e.cmd.run = EnumRun;
  e.cmd.cycle = EnumCycle;
  e.cmd.draw_value = EnumDraw;
  e.cmd.reset = EnumReset;
  e.cmd.user = b;
  e.cmd.modified = modified;
  list->entries.push_back(e);
}

void ImCmdList_AddFieldCustom(ImCmdList* list, const char* path,
                              ImCmdValueCallback draw, ImCmdResetCallback reset,
                              void* user, bool modified,
                              const char* default_text) {
  CmdArena& a = list->arena;
  const char* p = a.StrDup(path);
  CmdEntry e = {};
  e.id = p;
  e.cmd.name = MarkedName(a, p, modified);
  e.cmd.default_text = a.StrDup(default_text);
  e.cmd.draw_value = draw;
  e.cmd.reset = reset;
  e.cmd.user = user;
  e.cmd.modified = modified;
  list->entries.push_back(e);
}

}  // extern "C"
