// Copyright 2026 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// UI link: the NetImgui client side of the web viewer. Receives the
// headless Studio UI (draw frames + textures) over the /ui WebSocket and
// forwards browser input back to it.
//
// The GPU is reached only through the injected upload_texture/gpu_ready
// hooks, so the protocol logic here has no renderer or Emscripten
// dependencies and can be exercised natively in tests by feeding synthetic
// CmdDrawFrame/CmdTexture buffers to the ProcessCmd* methods.

#ifndef MUJOCO_PYTHON_EXPERIMENTAL_STUDIO_WEB_WEB_CLIENT_REMOTE_UI_H_
#define MUJOCO_PYTHON_EXPERIMENTAL_STUDIO_WEB_WEB_CLIENT_REMOTE_UI_H_

#include <imgui.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "NetImgui_Api.h"
#include "NetImgui_CmdPackets.h"
#include "NetImgui_Network.h"
#include "google/network_status.h"

namespace mujoco::studio {

// ImDrawData wrapper for receiving remote draw data.
struct NetImguiImDrawData : ImDrawData {
  NetImguiImDrawData()
      : mCommandList(ImGui::GetCurrentContext() ? ImGui::GetDrawListSharedData()
                                                : nullptr) {
    CmdLists.push_back(&mCommandList);
    CmdListsCount = 1;
  }
  ~NetImguiImDrawData() {}
  ImDrawList mCommandList;
  uint64_t mFrameIndex = 0;
};

class RemoteUi {
 public:
  using SocketInfo = NetImgui::Internal::Network::SocketInfo;
  using ReadyState = NetImgui::Internal::Network::ReadyState;
  using ClientTextureID = NetImgui::Internal::ClientTextureID;
  // The renderer-facing callbacks of the link; the app implements them once.
  // The link reaches the GPU only through it, so the protocol logic stays
  // renderer-agnostic.
  class Callbacks {
   public:
    virtual ~Callbacks() = default;
    // Uploads a full RGBA texture and returns the GPU handle; nullptr
    // pixels destroy the texture backing `current`.
    virtual uintptr_t UploadTexture(uintptr_t current, const std::byte* rgba,
                                    uint32_t width, uint32_t height) = 0;
    // True once the GPU context can accept texture uploads.
    virtual bool GpuReady() = 0;
  };

  explicit RemoteUi(Callbacks& callbacks) : callbacks_(callbacks) {}

  // (Re)connects to the UI WebSocket. Disconnects any existing socket first.
  void Connect(const std::string& url);
  bool HasSocket() const { return socket_ != nullptr; }
  // Current socket state; browser WebSockets connect and close
  // asynchronously, so this can differ from HasSocket() (see
  // google/network_status.h).
  ReadyState ConnectionState() const;
  // The WebSocket close code once the socket has closed, else 0 (e.g.
  // 4001 = another browser holds the controller slot).
  int CloseCode() const;

  // Remote clip rects are clamped to this (logical pixels); set each frame
  // before ReceiveAndProcessCommands().
  void SetMaxClip(float width, float height) {
    max_clip_[0] = width;
    max_clip_[1] = height;
  }

  // Sends the CmdVersion handshake when the socket is newly open, then drains
  // all incoming commands (draw frames, textures) for this frame.
  void ReceiveAndProcessCommands(int frame);

  // Captures ImGui input and sends it to the remote client. Must run before
  // ImGui::EndFrame(), because EndFrame() clears the io.InputQueueCharacters
  // queue this reads (note that ImGui::Render() calls EndFrame() implicitly).
  // Keep in sync with CaptureImguiInput in the vendored
  // netimgui/Code/ServerApp/Source/NetImguiServer_RemoteClient.cpp
  void CaptureAndSendInput();

  // Uploads CPU-buffered textures (e.g. the font atlas) that arrived before
  // the GPU context became available.
  void FlushPendingTextures();

  void Shutdown();

  // The latest assembled remote draw data, or nullptr before the first frame.
  NetImguiImDrawData* RemoteDrawData() { return remote_draw_data_; }

  // Returns the bytes received since the last call and resets the counter.
  uint64_t ConsumeByteCount() {
    uint64_t bytes = bytes_accum_;
    bytes_accum_ = 0;
    return bytes;
  }

  // Called from ReceiveAndProcessCommands(); public for tests.
  void ProcessCmdDrawFrame(NetImgui::Internal::CmdDrawFrame* cmd);
  void ProcessCmdTexture(NetImgui::Internal::CmdTexture* cmd);

 private:
  Callbacks& callbacks_;

  float max_clip_[2] = {0.0f, 0.0f};

  // --- Connection-scoped state, reset on every (re)connect. ----------------

  SocketInfo* socket_ = nullptr;
  bool handshake_sent_ = false;
  bool was_connected_ = false;
  ReadyState last_state_ = ReadyState::kDisconnected;
  NetImgui::Internal::PendingCom pending_receive_;
  NetImgui::Internal::CmdPendingRead cmd_pending_read_;
  // When true, the next CmdInput asks the client to send one uncompressed
  // draw frame (CmdInput::mCompressionSkip). Set on (re)connect and whenever
  // a delta frame arrives whose reference frame we don't have — without it,
  // a browser joining mid-stream drops every delta-compressed frame forever
  // and the remote UI never appears. Mirrors mbCompressionSkipOncePending in
  // NetImguiServer RemoteClient.
  bool request_keyframe_ = true;
  NetImgui::Internal::CmdDrawFrame* last_uncompressed_frame_ = nullptr;

  // --- Session state. -------------------------------------------------------

  NetImguiImDrawData* remote_draw_data_ = nullptr;
  std::unordered_map<ClientTextureID, uintptr_t> texture_map_;

  // CPU-side mirror of each texture's RGBA pixel data. Filament doesn't
  // support sub-region uploads (no glTexSubImage2D equivalent), so partial
  // updates from NetImgui must be patched into this buffer before
  // re-uploading the full texture.
  struct TextureEntry {
    std::vector<uint8_t> pixels;  // Full RGBA pixel data
    uint32_t width = 0;
    uint32_t height = 0;
  };
  std::unordered_map<ClientTextureID, TextureEntry> texture_cpu_;

  // Persistent input state for character accumulation across frames.
  // Mirrors mPendingInputChars/mMouseWheelPos in NetImguiServer RemoteClient.
  std::vector<ImWchar> pending_input_chars_;
  float mouse_wheel_pos_[2] = {0.0f, 0.0f};
  uint16_t last_screen_size_[2] = {0, 0};

  // Delta-compress the GUI stream (relayed to the client via CmdInput).
  bool use_compression_ = true;

  // --- Telemetry. -----------------------------------------------------------

  uint64_t bytes_accum_ = 0;
  int total_cmds_received_ = 0;
  int draw_frames_received_ = 0;
  int textures_received_ = 0;
};

}  // namespace mujoco::studio

#endif  // MUJOCO_PYTHON_EXPERIMENTAL_STUDIO_WEB_WEB_CLIENT_REMOTE_UI_H_
