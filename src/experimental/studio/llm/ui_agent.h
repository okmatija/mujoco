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

#ifndef MUJOCO_SRC_EXPERIMENTAL_STUDIO_LLM_UI_AGENT_H_
#define MUJOCO_SRC_EXPERIMENTAL_STUDIO_LLM_UI_AGENT_H_

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "experimental/studio/llm/llm_provider.h"

namespace mujoco::studio {

// Routes plain-English command-prompt input to an LLM and keeps the running
// conversation. Networking happens on a detached worker thread; the UI thread
// calls Poll() once per frame to fold a finished reply into the history. For
// deterministic headless capture, set_synchronous(true) makes Ask() block
// inline (used with the mock provider).
//
// This is the MVP seam from LLM_INTEGRATION_DESIGN.md: text in, text out, no
// tools yet. The provider abstraction and conversation model are what the Test
// Engine tool-use loop will build on.
class UiAgent {
 public:
  struct Turn {
    std::string role;      // "user" or "assistant"
    std::string text;
    std::string thinking;  // extended-thinking text (assistant turns only).
    std::string model;     // model id that produced the reply (assistant only).
  };

  UiAgent();  // ClaudeProvider if ANTHROPIC_API_KEY is set, else MockProvider.
  ~UiAgent() = default;

  UiAgent(const UiAgent&) = delete;
  UiAgent& operator=(const UiAgent&) = delete;

  // Submits a user question. No-op if empty or a request is already in flight.
  void Ask(const std::string& question);

  // Wipes the conversation history to start fresh. No-op while a request is in
  // flight (the in-flight reply would otherwise reappear on the next Poll).
  void Clear();

  // Interrupts the in-flight request: stops showing as busy and discards the
  // reply when it eventually arrives. The detached worker keeps running in the
  // background (the network call can't be aborted mid-flight) but its result is
  // ignored. No-op if not busy.
  void Cancel();

  // Folds any finished worker result into the history. Call once per frame.
  void Poll();

  bool busy() const { return busy_; }
  const std::vector<Turn>& history() const { return history_; }
  const std::string& provider_name() const { return provider_name_; }
  // Current model id (e.g. "claude-opus-4-8"), or "" if not applicable.
  std::string provider_model() const { return provider_ ? provider_->Model() : ""; }

  // Switches the provider/model to `arg` (an alias like "opus"/"sonnet" or a
  // full id). Returns a human-readable status message. No history side effects;
  // used by both the "/model" command and the settings UI.
  std::string SwitchModel(const std::string& arg);

  // The {alias, full id} models offered by each provider whose API key is set.
  std::vector<std::pair<std::string, std::string>> AvailableModels() const;

  void set_synchronous(bool s) { synchronous_ = s; }
  void set_provider(std::unique_ptr<LlmProvider> provider);

  // Registers the tools offered to the model and the executor that runs them.
  // The executor is invoked from the provider's tool-use loop (on the worker
  // thread in async mode), so it should be quick and only touch state that is
  // safe to mutate there (e.g. setting a window-open bool).
  void set_tools(std::vector<ToolDef> tools, ToolExecutor exec);

  // Callback run at the start of each Ask (UI thread), e.g. to reset per-turn
  // budgets like the grep-call count.
  void set_on_ask(std::function<void()> cb) { on_ask_ = std::move(cb); }

  // Handler used by the "/copy" command to put the transcript on the clipboard,
  // injected so the agent stays free of any UI/clipboard dependency.
  void set_copy_handler(std::function<void(const std::string&)> cb) {
    copy_ = std::move(cb);
  }

  // Handler used by the "/settings" command to toggle the settings window,
  // injected so the agent does not depend on the app's UI.
  void set_settings_handler(std::function<void()> cb) {
    settings_ = std::move(cb);
  }

 private:
  std::shared_ptr<LlmProvider> provider_;
  std::string provider_name_;
  std::string system_;
  std::vector<Turn> history_;
  std::vector<ToolDef> tools_;
  ToolExecutor executor_;
  std::function<void()> on_ask_;
  std::function<void(const std::string&)> copy_;
  std::function<void()> settings_;

  bool synchronous_ = false;
  std::atomic<bool> busy_{false};

  // Result hand-off from the worker thread. Held by shared_ptr so the worker is
  // safe even if the UiAgent is destroyed mid-request.
  struct Pending {
    std::mutex mu;
    bool done = false;
    LlmResult result;
  };
  std::shared_ptr<Pending> pending_;
};

}  // namespace mujoco::studio

#endif  // MUJOCO_SRC_EXPERIMENTAL_STUDIO_LLM_UI_AGENT_H_
