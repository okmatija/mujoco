// Copyright 2026 DeepMind Technologies Limited
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

// A self-contained XML model editor, wired up entirely through the Studio
// plugin API (GuiPlugin + ModelPlugin) so it can live in (and be split out to)
// its own translation unit / PR. It deliberately knows nothing about the rest
// of Studio and is known by nothing in it:
//
//   * To change the model it stashes the edited XML and lets the app PULL it
//     via ModelPlugin::get_model_to_load -- the app polls the plugin; the
//     plugin never calls into the app.
//   * When any model loads (from a file, a drag/drop, or anything else) the app
//     notifies via ModelPlugin::post_model_loaded and the editor refreshes its
//     text from the model file.
//
// This same generic seam is how any other feature (e.g. the LLM agent) can edit
// the model without coupling to -- or even knowing about -- this plugin.

#include <cfloat>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

#include <imgui.h>
#include <misc/cpp/imgui_stdlib.h>
#include <mujoco/mujoco.h>
#include "experimental/platform/ux/plugin.h"

namespace mujoco::studio {
namespace {

std::string ReadFileText(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return "";
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

bool WriteFileText(const std::string& path, const std::string& text) {
  std::ofstream f(path, std::ios::binary);
  if (!f) return false;
  f << text;
  return f.good();
}

bool EndsWith(const std::string& s, const char* suffix) {
  const size_t n = std::strlen(suffix);
  return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0;
}

}  // namespace

class XmlEditor {
 public:
  // GuiPlugin: draws the editor window contents (the app provides the window).
  void UpdateGui() {
    // --- Toolbar ---
    ImGui::PushStyleColor(ImGuiCol_Button, ImColor(40, 180, 40, 255).Value);
    bool compile = ImGui::Button("Compile and Reload");
    ImGui::PopStyleColor();
    ImGui::SetItemTooltip("%s", "Parse the XML and load it as the active model");

    ImGui::SameLine();
    const bool revert = ImGui::Button("Revert");
    ImGui::SetItemTooltip("%s", "Discard edits and reload from the model file");

    ImGui::SameLine();
    ImGui::TextDisabled("%s%zu chars", dirty_ ? "modified \xE2\x80\xA2 " : "",
                        text_.size());

    // --- Parse/compile error, if any ---
    if (!error_.empty()) {
      ImGui::PushStyleColor(ImGuiCol_Text, ImColor(235, 90, 90, 255).Value);
      ImGui::TextWrapped("%s", error_.c_str());
      ImGui::PopStyleColor();
    }

    // --- The XML itself: fills the window, leaving room for the save row. ---
    const float footer = ImGui::GetFrameHeightWithSpacing();
    if (ImGui::InputTextMultiline("##xml", &text_, ImVec2(-FLT_MIN, -footer),
                                  ImGuiInputTextFlags_AllowTabInput)) {
      dirty_ = (text_ != loaded_text_);
    }

    // --- Save row ---
    ImGui::SetNextItemWidth(-60.0f);
    ImGui::InputTextWithHint("##savepath", "path to save (.xml)", &save_path_);
    ImGui::SameLine();
    if (ImGui::Button("Save", ImVec2(-FLT_MIN, 0.0f))) {
      if (save_path_.empty()) {
        error_ = "Enter a path to save to.";
      } else if (WriteFileText(save_path_, text_)) {
        loaded_text_ = text_;
        dirty_ = false;
        error_.clear();
      } else {
        error_ = "Failed to save to '" + save_path_ + "'.";
      }
    }
    ImGui::SetItemTooltip("%s", "Write the current text to the path on the left");

    if (revert) {
      text_ = (!path_.empty() && !EndsWith(path_, ".mjb")) ? ReadFileText(path_)
                                                           : loaded_text_;
      dirty_ = (text_ != loaded_text_);
      error_.clear();
    }
    if (compile) {
      RequestCompile();
    }
  }

  // ModelPlugin: if a reload was requested, hand the edited XML to the app
  // (which compiles and swaps it in). Returns nullptr when there is nothing to
  // load. The returned pointer stays valid (it is owned by `load_buffer_`).
  const char* GetModelToLoad(int* size, char* content_type, int content_type_sz,
                             char* model_name, int model_name_sz) {
    if (!reload_pending_) {
      return nullptr;
    }
    reload_pending_ = false;
    suppress_refresh_ = true;  // keep our text; don't re-read on the reload.
    load_buffer_ = text_;
    *size = static_cast<int>(load_buffer_.size());
    std::snprintf(content_type, content_type_sz, "%s", "text/xml");
    std::snprintf(model_name, model_name_sz, "%s",
                  path_.empty() ? "model.xml" : path_.c_str());
    return load_buffer_.c_str();
  }

  // ModelPlugin: a model was (re)loaded. Refresh the editor from the model file
  // unless this load is the one we just triggered ourselves.
  void OnModelLoaded(const char* model_path) {
    const std::string path = model_path ? model_path : "";
    if (suppress_refresh_) {
      suppress_refresh_ = false;
      path_ = path;
      loaded_text_ = text_;
      dirty_ = false;
      return;
    }
    path_ = path;
    save_path_ = path;
    error_.clear();
    // Only text sources can be shown; .mjb is binary.
    text_ = (!path.empty() && !EndsWith(path, ".mjb")) ? ReadFileText(path)
                                                       : std::string();
    loaded_text_ = text_;
    dirty_ = false;
  }

 private:
  // Parse the buffer to surface syntax/schema errors inline before asking the
  // app to load it (so a typo shows an error here instead of clearing the
  // scene). The app re-parses and compiles the same text on load.
  void RequestCompile() {
    mjVFS vfs;
    mj_defaultVFS(&vfs);
    char err[1024] = "";
    mjSpec* spec = mj_parseXMLString(text_.c_str(), &vfs, err, sizeof(err));
    mj_deleteVFS(&vfs);
    if (spec == nullptr) {
      error_ = err[0] ? err : "Failed to parse XML.";
      return;
    }
    mj_deleteSpec(spec);
    error_.clear();
    reload_pending_ = true;
  }

  std::string text_;         // editor buffer (the XML source)
  std::string loaded_text_;  // text as last loaded/saved (baseline for dirty)
  std::string load_buffer_;  // stable storage handed to get_model_to_load
  std::string path_;         // path of the current model
  std::string save_path_;    // editable save target
  std::string error_;        // last parse/save error (shown inline)
  bool reload_pending_ = false;
  bool suppress_refresh_ = false;
  bool dirty_ = false;
};

// Anchor symbol. Referencing this from the Studio app forces this translation
// unit (and so the mjPLUGIN_LIB_INIT registration below) to be linked in; see
// the comment in object_launcher_plugin.cc.
void LinkXmlEditorPlugin() {}

}  // namespace mujoco::studio

mjPLUGIN_LIB_INIT(xml_editor) {
  using mujoco::studio::XmlEditor;

  static XmlEditor editor;

  mujoco::platform::GuiPlugin gui;
  gui.data = &editor;
  gui.name = "XML Editor";
  gui.icon = "XML";  // the tool rail shows this text as the button.
  gui.update = [](mujoco::platform::GuiPlugin* self) {
    static_cast<XmlEditor*>(self->data)->UpdateGui();
  };
  mujoco::platform::RegisterPlugin(gui);

  mujoco::platform::ModelPlugin model;
  model.data = &editor;
  model.name = "XML Editor";
  model.get_model_to_load =
      [](mujoco::platform::ModelPlugin* self, int* size, char* content_type,
         int content_type_sz, char* model_name,
         int model_name_sz) -> const char* {
    return static_cast<XmlEditor*>(self->data)->GetModelToLoad(
        size, content_type, content_type_sz, model_name, model_name_sz);
  };
  model.post_model_loaded = [](mujoco::platform::ModelPlugin* self,
                               const char* model_path) {
    static_cast<XmlEditor*>(self->data)->OnModelLoaded(model_path);
  };
  mujoco::platform::RegisterPlugin(model);
}
