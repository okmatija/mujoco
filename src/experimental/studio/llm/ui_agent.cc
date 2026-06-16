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

#include "experimental/studio/llm/ui_agent.h"

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "experimental/studio/llm/llm_claude.h"
#include "experimental/studio/llm/llm_gemini.h"
#include "experimental/studio/llm/llm_mock.h"
#include "experimental/studio/llm/llm_provider.h"

#ifndef MUJOCO_STUDIO_SOURCE_DIR
#define MUJOCO_STUDIO_SOURCE_DIR ""
#endif

namespace mujoco::studio {
namespace {

std::string ReadFile(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return "";
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

// The system prompt lives in an editable file on disk so it can be tweaked
// without recompiling (and it hot-reloads each turn). Lookup order: the
// MUJOCO_STUDIO_SYSTEM_PROMPT env var, then llm/system_prompt.md in the source
// tree, then a built-in fallback.
std::string LoadSystemPrompt() {
  if (const char* env = std::getenv("MUJOCO_STUDIO_SYSTEM_PROMPT");
      env && *env) {
    if (std::string s = ReadFile(env); !s.empty()) return s;
  }
  const std::string root = MUJOCO_STUDIO_SOURCE_DIR;
  if (!root.empty()) {
    if (std::string s = ReadFile(root + "/studio/llm/system_prompt.md");
        !s.empty()) {
      return s;
    }
  }
  return
      "You are an AI assistant embedded in MuJoCo Studio. Act on the UI only by "
      "calling run_ui_program (it drives the real widgets via the ImGui Test "
      "Engine). Use model_info for joint/actuator/body names and grep_source for "
      "source-level ids; do not invent refs. Keep replies to one short sentence.";
}

}  // namespace

UiAgent::UiAgent() {
  system_ = LoadSystemPrompt();

  std::string key = ClaudeProvider::KeyFromEnv();
  if (!key.empty()) {
    provider_ = std::make_shared<ClaudeProvider>(std::move(key));
  } else {
    provider_ = std::make_shared<MockProvider>();
  }
  provider_name_ = provider_->name();
}

void UiAgent::set_provider(std::unique_ptr<LlmProvider> provider) {
  if (!provider) return;
  provider_ = std::move(provider);
  provider_name_ = provider_->name();
}

void UiAgent::set_tools(std::vector<ToolDef> tools, ToolExecutor exec) {
  tools_ = std::move(tools);
  executor_ = std::move(exec);
}

void UiAgent::Clear() {
  if (busy_) return;
  history_.clear();
}

void UiAgent::Cancel() {
  if (!busy_ || !pending_) {
    busy_ = false;
    return;
  }
  // Mark cancelled so the worker's guarded executor stops applying actions, and
  // grab the thinking accumulated so far to show in the cancelled turn.
  std::string partial;
  {
    std::lock_guard<std::mutex> lk(pending_->mu);
    pending_->cancelled.store(true);
    partial = pending_->partial_thinking;
  }
  // Drop our handle so Poll() ignores the eventual result (the worker keeps its
  // own shared_ptr, so it stays safe). Both Cancel() and Poll() run on the UI
  // thread, so there is no race between them.
  pending_.reset();
  busy_ = false;
  history_.push_back({"assistant", "[cancelled]", partial, provider_->Model()});
}

std::string UiAgent::SwitchModel(const std::string& arg) {
  if (busy_) return "Busy; cannot switch model right now.";
  std::string a;
  for (char c : arg) {
    a += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  const bool gemini = a.rfind("gemini", 0) == 0 || a == "flash" || a == "pro";
  const bool claude = a == "opus" || a == "sonnet" || a == "haiku" ||
                      a.rfind("claude-", 0) == 0;
  const char* want = gemini ? "Gemini" : (claude ? "Claude" : nullptr);
  if (want == nullptr) {
    return "Unknown model \"" + arg +
           "\". Try: opus, sonnet, haiku, gemini, flash, pro.";
  }
  // Ensure the right provider is active (constructing it from its env key the
  // first time), then set the model on it.
  if (std::string(provider_->name()) != want) {
    std::string key =
        gemini ? GeminiProvider::KeyFromEnv() : ClaudeProvider::KeyFromEnv();
    if (key.empty()) {
      return std::string(gemini ? "GEMINI_API_KEY (or GOOGLE_API_KEY)"
                                : "ANTHROPIC_API_KEY") +
             " is not set.";
    }
    if (gemini) {
      provider_ = std::make_shared<GeminiProvider>(key);
    } else {
      provider_ = std::make_shared<ClaudeProvider>(key);
    }
    provider_name_ = provider_->name();
  }
  const std::string id = provider_->SetModel(arg);
  return id.empty() ? ("Unknown " + std::string(want) + " model \"" + arg + "\".")
                    : ("Switched to " + id + " (" + std::string(want) + ").");
}

std::vector<std::pair<std::string, std::string>> UiAgent::AvailableModels()
    const {
  std::vector<std::pair<std::string, std::string>> out;
  if (!ClaudeProvider::KeyFromEnv().empty()) {
    for (const auto& m : ClaudeProvider::Models()) out.push_back(m);
  }
  if (!GeminiProvider::KeyFromEnv().empty()) {
    for (const auto& m : GeminiProvider::Models()) out.push_back(m);
  }
  return out;
}

void UiAgent::Ask(const std::string& question) {
  if (question.empty() || busy_) return;

  // Local commands handled here (not sent to the model).
  {
    const size_t b = question.find_first_not_of(" \t");
    const size_t e = question.find_last_not_of(" \t");
    const std::string trimmed =
        (b == std::string::npos) ? "" : question.substr(b, e - b + 1);
    // "/clear" wipes the conversation and starts fresh.
    if (trimmed == "/clear") {
      Clear();
      return;
    }
    // "/settings" toggles the settings window (via the injected handler).
    if (trimmed == "/settings") {
      if (settings_) settings_();
      return;
    }
    // "/copy" puts the whole transcript on the clipboard (via the injected
    // handler) and confirms in the conversation.
    if (trimmed == "/copy") {
      std::string transcript;
      for (const Turn& t : history_) {
        const std::string who =
            (t.role == "user")
                ? "You"
                : (t.model.empty() ? "Agent" : ("Agent (" + t.model + ")"));
        transcript += who + ": " + t.text + "\n\n";
      }
      std::string msg;
      if (transcript.empty()) {
        msg = "Nothing to copy yet.";
      } else if (copy_) {
        copy_(transcript);
        msg = "Copied the conversation to the clipboard.";
      } else {
        msg = "No clipboard handler is set.";
      }
      history_.push_back({"user", question});
      history_.push_back({"assistant", msg});
      return;
    }
  }

  // "/model <alias|id>" is a local command: switch the provider/model (or report
  // the current one) without calling the model. Claude aliases:
  // opus/sonnet/haiku; Gemini aliases: gemini/flash/pro; or a full id.
  if (question.rfind("/model", 0) == 0) {
    std::string arg = question.substr(6);
    const size_t b = arg.find_first_not_of(" \t");
    const size_t e = arg.find_last_not_of(" \t");
    arg = (b == std::string::npos) ? "" : arg.substr(b, e - b + 1);
    std::string msg;
    if (arg.empty()) {
      // List the models for each provider whose API key is set, marking the one
      // that is currently active.
      const std::string active_id = provider_->Model();
      const std::string active_provider = provider_->name();
      auto append = [&](const char* heading, const char* provider_name,
                        const std::vector<std::pair<std::string, std::string>>&
                            models) {
        msg += std::string(heading) + ":\n";
        for (const auto& [alias, id] : models) {
          const bool current =
              active_provider == provider_name && active_id == id;
          msg += "  " + alias + "  (" + id + ")" +
                 (current ? "  (current)" : "") + "\n";
        }
      };
      if (!ClaudeProvider::KeyFromEnv().empty()) {
        append("Anthropic", "Claude", ClaudeProvider::Models());
      }
      if (!GeminiProvider::KeyFromEnv().empty()) {
        append("Gemini", "Gemini", GeminiProvider::Models());
      }
      msg = msg.empty()
                ? std::string("No model API keys set "
                              "(ANTHROPIC_API_KEY / GEMINI_API_KEY).")
                : ("Available models (use /model <alias>):\n" + msg);
    } else {
      msg = SwitchModel(arg);
    }
    history_.push_back({"user", question});
    history_.push_back({"assistant", msg, "", provider_->Model()});
    return;
  }

  system_ = LoadSystemPrompt();  // hot-reload so edits apply without a restart
  history_.push_back({"user", question});
  if (on_ask_) on_ask_();  // reset per-turn budgets (e.g. grep count)
  busy_ = true;

  std::vector<LlmMessage> messages;
  messages.reserve(history_.size());
  for (const Turn& t : history_) messages.push_back({t.role, t.text});

  if (synchronous_) {
    LlmResult r = provider_->Send(system_, messages, tools_, executor_);
    history_.push_back({"assistant", r.ok ? r.text : ("[error] " + r.error),
                        r.thinking, provider_->Model()});
    busy_ = false;
    return;
  }

  pending_ = std::make_shared<Pending>();
  auto provider = provider_;          // shared: outlives the agent if needed.
  auto pending = pending_;
  std::string system = system_;
  std::vector<ToolDef> tools = tools_;
  ToolExecutor exec = executor_;
  // Guard the executor: once cancelled, drop any further tool actions so nothing
  // from this abandoned request touches the UI.
  ToolExecutor guarded = [exec, pending](const std::string& name,
                                         const std::string& args) -> std::string {
    if (pending->cancelled.load()) return "(request cancelled)";
    return exec ? exec(name, args) : std::string();
  };
  // Stream thinking-so-far into the shared state for Cancel() to surface.
  ProgressCallback on_thinking = [pending](const std::string& thinking) {
    std::lock_guard<std::mutex> lk(pending->mu);
    pending->partial_thinking = thinking;
  };
  std::thread([provider, pending, system, messages, tools, guarded, on_thinking] {
    LlmResult r = provider->Send(system, messages, tools, guarded, on_thinking);
    std::lock_guard<std::mutex> lk(pending->mu);
    pending->result = std::move(r);
    pending->done = true;
  }).detach();
}

void UiAgent::Poll() {
  if (!pending_) return;
  auto pending = pending_;  // keep alive across the unlock below.
  std::lock_guard<std::mutex> lk(pending->mu);
  if (!pending->done) return;
  const LlmResult& r = pending->result;
  history_.push_back({"assistant", r.ok ? r.text : ("[error] " + r.error),
                      r.thinking, provider_->Model()});
  busy_ = false;
  pending_.reset();
}

}  // namespace mujoco::studio
