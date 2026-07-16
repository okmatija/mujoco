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

#include "ui_link.h"

#include <algorithm>
#include <cstring>
#include <string>

#include "google/logging.h"

namespace mujoco::studio {

using namespace NetImgui::Internal;

namespace {

void log_unmapped_texture(
    UiLink::ClientTextureID clientTexId, size_t mapSize, uint32_t drawIdx,
    const std::unordered_map<UiLink::ClientTextureID, uintptr_t>& texMap) {
  VLOG(1, "DrawFrame: UNMAPPED clientTexId=%lu, mapSize=%zu, draw#=%u",
       static_cast<unsigned long>(clientTexId), mapSize, drawIdx);
  std::string keys_str = "";
  for (auto& kv : texMap) {
    keys_str += " " + std::to_string(kv.first) +
                "(fil=" + std::to_string(kv.second) + ")";
  }
  VLOG(1, "  map keys:%s", keys_str.c_str());
}

void log_cmd_received(CmdHeader::eCommands cmdType, uint32_t cmdSize,
                      int drawFrames, int textures,
                      const PendingCom& pendingRcv) {
  if (cmdType == CmdHeader::eCommands::DrawFrame) {
    VLOG(2, "Received DrawFrame #%d (size=%u)", drawFrames, cmdSize);
  } else if (cmdType == CmdHeader::eCommands::Texture) {
    VLOG(2, "Received Texture #%d (size=%u)", textures, cmdSize);
  } else if (cmdType == CmdHeader::eCommands::Version) {
    auto* pVer = reinterpret_cast<const CmdVersion*>(pendingRcv.pCommand);
    LOG(Info,
        "Received CmdVersion from client: name='%s', version=%d, wcharSize=%d",
        pVer->mClientName, static_cast<int>(pVer->mVersion), pVer->mWCharSize);
  } else if (cmdType == CmdHeader::eCommands::Background) {
    VLOG(2, "Received Background cmd (size=%u)", cmdSize);
  } else if (cmdType != CmdHeader::eCommands::Count &&
             cmdType != CmdHeader::eCommands::Clipboard &&
             cmdType != CmdHeader::eCommands::Input) {
    VLOG(2, "Received UNKNOWN cmd: type=%d, size=%u", static_cast<int>(cmdType),
         cmdSize);
  }
}

}  // namespace

void UiLink::Connect(const std::string& url) {
  if (socket_) {
    Network::Disconnect(socket_);
    socket_ = nullptr;
  }
  // The port argument is unused when a full URL is passed.
  LOG(Info, "Connecting to WebSocket at %s", url.c_str());
  socket_ = Network::Connect(url.c_str(), 0);
  LOG(Info, "Network::Connect returned: socket=%p",
      static_cast<void*>(socket_));
}

UiLink::ReadyState UiLink::ConnectionState() const {
  return Network::GetReadyState(socket_);
}

int UiLink::CloseCode() const { return Network::GetCloseCode(socket_); }

UiLink::RemoteTransform UiLink::GetRemoteTransform() const {
  RemoteTransform t;
  if (!receive_only_ || driver_screen_[0] == 0 || driver_screen_[1] == 0 ||
      max_clip_[0] <= 0.0f || max_clip_[1] <= 0.0f) {
    return t;
  }
  const float sx = max_clip_[0] / driver_screen_[0];
  const float sy = max_clip_[1] / driver_screen_[1];
  t.scale = sx < sy ? sx : sy;
  t.offset_x = (max_clip_[0] - driver_screen_[0] * t.scale) * 0.5f;
  t.offset_y = (max_clip_[1] - driver_screen_[1] * t.scale) * 0.5f;
  return t;
}

bool UiLink::DriverMousePos(float* x, float* y) const {
  if (!driver_mouse_valid_) return false;
  const RemoteTransform t = GetRemoteTransform();
  *x = driver_mouse_[0] * t.scale + t.offset_x;
  *y = driver_mouse_[1] * t.scale + t.offset_y;
  return true;
}

const char* UiLink::StatusString() const {
  return Network::ReadyStateName(ConnectionState());
}

void UiLink::ReceiveAndProcessCommands(int frame) {
  if (!socket_) return;

  const ReadyState state = ConnectionState();
  if (last_state_ != state) {
    LOG(Info, "WebSocket status changed: '%s' -> '%s' (frame %d)",
        Network::ReadyStateName(last_state_), Network::ReadyStateName(state),
        frame);
    last_state_ = state;
  }
  VLOG(1,
       "Frame %d: status='%s', handshake=%s, cmds=%d, draws=%d, textures=%d, "
       "hasDrawData=%s",
       frame, Network::ReadyStateName(state),
       handshake_sent_ ? "sent" : "not_sent", total_cmds_received_,
       draw_frames_received_, textures_received_,
       remote_draw_data_ != nullptr ? "yes" : "no");

  if (state == ReadyState::kOpen) {
    if (!handshake_sent_ && !receive_only_) {
      CmdVersion cmdVersion;
      StringCopy(cmdVersion.mClientName, "MuJoCo Web Viewer");
      LOG(Info,
          "Sending CmdVersion handshake: size=%u, type=%d, version=%d, "
          "wcharSize=%d, name='%s'",
          cmdVersion.mSize, static_cast<int>(cmdVersion.mType),
          static_cast<int>(cmdVersion.mVersion), cmdVersion.mWCharSize,
          cmdVersion.mClientName);

      PendingCom pendingSend;
      pendingSend.pCommand = &cmdVersion;
      pendingSend.SizeCurrent = 0;

      int sendAttempts = 0;
      while (!pendingSend.IsDone() && !pendingSend.IsError()) {
        Network::DataSend(socket_, pendingSend);
        sendAttempts++;
      }
      LOG(Info,
          "CmdVersion send: done=%s, error=%s, attempts=%d, bytesSent=%zu",
          pendingSend.IsDone() ? "true" : "false",
          pendingSend.IsError() ? "true" : "false", sendAttempts,
          static_cast<size_t>(pendingSend.SizeCurrent));
      handshake_sent_ = true;
      // Fresh connection: the client may resume delta compression against
      // a frame from a previous session; ask for an uncompressed keyframe.
      request_keyframe_ = true;
    }
  } else {
    if (handshake_sent_) {
      LOG(Info, "Resetting handshake (status='%s')",
          Network::ReadyStateName(state));
    }
    handshake_sent_ = false;
  }

  // Network receive — drain all available data in one frame.
  const bool isConnected = (state == ReadyState::kOpen);

  // If the connection has closed, discard any buffered data immediately
  // rather than churning through stale commands for several seconds.
  if (was_connected_ && !isConnected) {
    LOG(Info, "Connection lost (status='%s'). Discarding buffered data.",
        Network::ReadyStateName(state));
    // Reset any in-progress receive.
    if (pending_rcv_.bAutoFree) netImguiDeleteSafe(pending_rcv_.pCommand);
    pending_rcv_ = PendingCom();
    was_connected_ = false;
  }
  if (isConnected) was_connected_ = true;

  int maxCommandsPerFrame = 64;
  int cmdsThisFrame = 0;
  bool hadPendingData = Network::DataReceivePending(socket_);
  if (hadPendingData) {
    VLOG(1, "Frame %d: data pending on socket", frame);
  }

  while (isConnected && maxCommandsPerFrame-- > 0) {
    if (pending_rcv_.IsReady()) {
      cmd_pending_read_ = CmdPendingRead();
      pending_rcv_.pCommand = &cmd_pending_read_;
      pending_rcv_.bAutoFree = false;
    }

    if (!Network::DataReceivePending(socket_)) break;

    Network::DataReceive(socket_, pending_rcv_);

    if (pending_rcv_.pCommand->mSize > sizeof(CmdPendingRead) &&
        pending_rcv_.pCommand == &cmd_pending_read_) {
      VLOG(2, "Allocating %u bytes for incoming cmd type=%d",
           pending_rcv_.pCommand->mSize,
           static_cast<int>(pending_rcv_.pCommand->mType));
      CmdPendingRead* pCmdHeader = reinterpret_cast<CmdPendingRead*>(
          netImguiSizedNew<uint8_t>(pending_rcv_.pCommand->mSize));
      *pCmdHeader = cmd_pending_read_;
      pending_rcv_.pCommand = pCmdHeader;
      pending_rcv_.bAutoFree = true;
    }

    if (!pending_rcv_.IsDone()) {
      if (pending_rcv_.IsError()) {
        LOG(Error, "Receive ERROR: cmdSize=%u, got=%zu, type=%d",
            pending_rcv_.pCommand->mSize,
            static_cast<size_t>(pending_rcv_.SizeCurrent),
            static_cast<int>(pending_rcv_.pCommand->mType));
        if (pending_rcv_.bAutoFree) netImguiDeleteSafe(pending_rcv_.pCommand);
        pending_rcv_ = PendingCom();
      }
      continue;
    }

    // Command fully received — dispatch.
    bytes_accum_ += pending_rcv_.pCommand->mSize;
    cmdsThisFrame++;
    total_cmds_received_++;
    CmdHeader::eCommands cmdType = pending_rcv_.pCommand->mType;
    log_cmd_received(cmdType, pending_rcv_.pCommand->mSize,
                     draw_frames_received_, textures_received_, pending_rcv_);

    if (cmdType == CmdHeader::eCommands::Count) {
      // CmdPendingRead sentinel — skip silently.
    } else if (cmdType == CmdHeader::eCommands::DrawFrame) {
      draw_frames_received_++;
      ProcessCmdDrawFrame(
          reinterpret_cast<CmdDrawFrame*>(pending_rcv_.pCommand));
    } else if (cmdType == CmdHeader::eCommands::Texture) {
      textures_received_++;
      ProcessCmdTexture(reinterpret_cast<CmdTexture*>(pending_rcv_.pCommand));
    } else if (cmdType == CmdHeader::eCommands::Input) {
      // Only receive-only links get input: the server mirrors the driver's
      // input commands to spectators for cursor rendering.
      const auto* input =
          reinterpret_cast<const CmdInput*>(pending_rcv_.pCommand);
      driver_mouse_[0] = input->mMousePos[0];
      driver_mouse_[1] = input->mMousePos[1];
      driver_screen_[0] = input->mScreenSize[0];
      driver_screen_[1] = input->mScreenSize[1];
      driver_mouse_valid_ = true;
    }
    // Version, Background, Clipboard — already logged by log_cmd_received.

    if (pending_rcv_.bAutoFree) netImguiDeleteSafe(pending_rcv_.pCommand);
    pending_rcv_ = PendingCom();
  }

  VLOG(2, "Frame %d: processed %d commands", frame, cmdsThisFrame);
}

void UiLink::FlushPendingTextures() {
  for (auto& [texID, entry] : texture_cpu_) {
    uintptr_t& localTex = texture_map_[texID];
    if (localTex == 0 && !entry.pixels.empty()) {
      localTex = upload_texture_(
          localTex, reinterpret_cast<const std::byte*>(entry.pixels.data()),
          entry.width, entry.height);
      LOG(Info, "FlushPendingTextures: uploaded texID=%lu -> filament=%lu",
          static_cast<unsigned long>(texID),
          static_cast<unsigned long>(localTex));
    }
  }
}

void UiLink::ProcessCmdTexture(CmdTexture* pCmdTexture) {
  if (!pCmdTexture) return;

  if (!pCmdTexture->mpTextureData.IsPointer()) {
    pCmdTexture->mpTextureData.ToPointer();
  }

  ClientTextureID texID = pCmdTexture->mTextureClientID;
  VLOG(1,
       "ProcessCmdTexture: clientTexID=%lu, status=%d, size=%ux%u, format=%d, "
       "offset=%u,%u, mapSize=%zu",
       static_cast<unsigned long>(texID),
       static_cast<int>(pCmdTexture->mStatus),
       static_cast<uint32_t>(pCmdTexture->mWidth),
       static_cast<uint32_t>(pCmdTexture->mHeight),
       static_cast<int>(pCmdTexture->mFormat),
       static_cast<uint32_t>(pCmdTexture->mOffsetX),
       static_cast<uint32_t>(pCmdTexture->mOffsetY), texture_map_.size());

  uintptr_t& localTex = texture_map_[texID];

  if (pCmdTexture->mStatus == CmdTexture::eType::Destroy) {
    if (localTex != 0 && gpu_ready_()) {
      // Pass nullptr pixels to destroy the GPU texture.
      upload_texture_(localTex, nullptr, 0, 0);
    }
    localTex = 0;
    texture_cpu_.erase(texID);
    return;
  }

  uint8_t* pPixels = pCmdTexture->mpTextureData.Get();
  if (!pPixels) return;

  // Detect actual format by data size, not format tag (which can be wrong).
  uint32_t patchW = pCmdTexture->mWidth;
  uint32_t patchH = pCmdTexture->mHeight;
  uint32_t pixelCount = patchW * patchH;
  uint32_t dataSize = pCmdTexture->mSize - sizeof(CmdTexture);
  uint32_t expectedRGBA = pixelCount * 4;
  uint32_t expectedA8 = pixelCount * 1;

  bool isA8 = (pCmdTexture->mFormat == 1) ||
              (dataSize == expectedA8 && dataSize != expectedRGBA);

  // Convert incoming pixels to RGBA (Filament requires RGBA).
  // For A8 font atlas data, expand each byte to (255, 255, 255, alpha).
  std::vector<uint8_t> rgbaPixels(pixelCount * 4);
  if (isA8) {
    for (uint32_t p = 0; p < pixelCount; ++p) {
      rgbaPixels[p * 4 + 0] = 255;
      rgbaPixels[p * 4 + 1] = 255;
      rgbaPixels[p * 4 + 2] = 255;
      rgbaPixels[p * 4 + 3] = pPixels[p];
    }
  } else {
    memcpy(rgbaPixels.data(), pPixels, pixelCount * 4);
  }

  TextureEntry& entry = texture_cpu_[texID];

  if (pCmdTexture->mStatus == CmdTexture::eType::Create) {
    // Full texture creation — store the CPU-side mirror.
    entry.width = patchW;
    entry.height = patchH;
    entry.pixels = std::move(rgbaPixels);
  } else {
    // Partial update — patch the sub-region into the existing CPU mirror.
    // If no CPU mirror exists (e.g. we missed the Create), skip.
    if (entry.pixels.empty()) {
      LOG(Warning,
          "ProcessCmdTexture: partial update for texID=%lu "
          "but no CPU mirror exists, skipping",
          static_cast<unsigned long>(texID));
      return;
    }
    uint32_t offX = pCmdTexture->mOffsetX;
    uint32_t offY = pCmdTexture->mOffsetY;
    for (uint32_t row = 0; row < patchH; ++row) {
      uint32_t dstOffset = ((offY + row) * entry.width + offX) * 4;
      uint32_t srcOffset = row * patchW * 4;
      memcpy(&entry.pixels[dstOffset], &rgbaPixels[srcOffset], patchW * 4);
    }
  }

  // Upload the full CPU-side texture to the GPU if the context is ready.
  // If it isn't yet (model still loading), the texture stays in texture_cpu_
  // and will be flushed by FlushPendingTextures() later.
  if (gpu_ready_()) {
    localTex = upload_texture_(
        localTex, reinterpret_cast<const std::byte*>(entry.pixels.data()),
        entry.width, entry.height);
    VLOG(1, "uploadGuiImage for clientTexID=%lu -> filament=%lu",
         static_cast<unsigned long>(texID),
         static_cast<unsigned long>(localTex));
  } else {
    VLOG(1, "ProcessCmdTexture: buffered texID=%lu, deferring GPU upload",
         static_cast<unsigned long>(texID));
  }
}

void UiLink::CaptureAndSendInput() {
  if (!socket_ || receive_only_) return;

  if (ConnectionState() != ReadyState::kOpen) return;

  // Capture input from Dear ImGui.
  const ImGuiIO& io = ImGui::GetIO();

  // When the local UI wants to capture mouse or keyboard, suppress the
  // corresponding input from being forwarded to the remote client. This
  // prevents clicks/keys meant for the local status overlay (or any other
  // local window) from leaking through to the UI server.
  const bool localWantsMouse = io.WantCaptureMouse;
  const bool localWantsKeyboard = io.WantCaptureKeyboard;

  {
    // Only accumulate characters when the local UI is NOT capturing keyboard.
    if (!localWantsKeyboard) {
      const size_t initialSize = pending_input_chars_.size();
      const size_t addedChar = io.InputQueueCharacters.size();
      if (addedChar) {
        pending_input_chars_.resize(initialSize + addedChar);
        memcpy(&pending_input_chars_[initialSize], io.InputQueueCharacters.Data,
               addedChar * sizeof(ImWchar));
      }
    }

    // Only accumulate scroll when the local UI is NOT capturing mouse.
    if (!localWantsMouse) {
      mouse_wheel_pos_[0] += io.MouseWheel;
      mouse_wheel_pos_[1] += io.MouseWheelH;
    }
  }

  CmdInput cmdInput;
  cmdInput.mScreenSize[0] = static_cast<uint16_t>(io.DisplaySize.x);
  cmdInput.mScreenSize[1] = static_cast<uint16_t>(io.DisplaySize.y);
  // An unstable screen size forces the remote UI to relayout every frame
  // (visible as UI flicker), so make changes loud.
  if (cmdInput.mScreenSize[0] != last_screen_size_[0] ||
      cmdInput.mScreenSize[1] != last_screen_size_[1]) {
    LOG(Info, "Screen size sent to client changed: %ux%u -> %ux%u",
        last_screen_size_[0], last_screen_size_[1], cmdInput.mScreenSize[0],
        cmdInput.mScreenSize[1]);
    last_screen_size_[0] = cmdInput.mScreenSize[0];
    last_screen_size_[1] = cmdInput.mScreenSize[1];
  }
  cmdInput.mFontDPIScaling = 1.f;
  cmdInput.mDesiredFps = 60.0f;
  cmdInput.mCompressionUse = use_compression_;
  cmdInput.mCompressionSkip = request_keyframe_;

  // Send mouse position and wheel only when the local UI is not capturing.
  if (!localWantsMouse) {
    cmdInput.mMousePos[0] = static_cast<int16_t>(io.MousePos.x);
    cmdInput.mMousePos[1] = static_cast<int16_t>(io.MousePos.y);
    cmdInput.mMouseWheelVert = mouse_wheel_pos_[0];
    cmdInput.mMouseWheelHoriz = mouse_wheel_pos_[1];
  } else {
    // Park the mouse off-screen so the remote side doesn't think we're
    // hovering over anything.
    cmdInput.mMousePos[0] = -1;
    cmdInput.mMousePos[1] = -1;
    cmdInput.mMouseWheelVert = 0;
    cmdInput.mMouseWheelHoriz = 0;
  }

  // Mouse Buttons Inputs
  // If Dear ImGui Update this enum, must also adjust our enum copy
  static_assert(
      static_cast<int>(CmdInput::NetImguiMouseButton::ImGuiMouseButton_COUNT) ==
          static_cast<int>(ImGuiMouseButton_::ImGuiMouseButton_COUNT),
      "Update the NetImgui enum to match the updated Dear ImGui enum");
  cmdInput.mMouseDownMask = 0;
  if (!localWantsMouse) {
    cmdInput.mMouseDownMask |=
        ImGui::IsMouseDown(ImGuiMouseButton_::ImGuiMouseButton_Left)
            ? 1 << CmdInput::ImGuiMouseButton_Left
            : 0;
    cmdInput.mMouseDownMask |=
        ImGui::IsMouseDown(ImGuiMouseButton_::ImGuiMouseButton_Right)
            ? 1 << CmdInput::ImGuiMouseButton_Right
            : 0;
    cmdInput.mMouseDownMask |=
        ImGui::IsMouseDown(ImGuiMouseButton_::ImGuiMouseButton_Middle)
            ? 1 << CmdInput::ImGuiMouseButton_Middle
            : 0;
    cmdInput.mMouseDownMask |=
        ImGui::IsMouseDown(3) ? 1 << CmdInput::ImGuiMouseButton_Extra1 : 0;
    cmdInput.mMouseDownMask |=
        ImGui::IsMouseDown(4) ? 1 << CmdInput::ImGuiMouseButton_Extra2 : 0;
  }

// Keyboard Inputs
// If Dear ImGui Update their enum, must also adjust our enum copy,
// so adding a few check to detect a change
#define EnumKeynameTest(KEYNAME)                                               \
  static_cast<int>(CmdInput::NetImguiKeys::KEYNAME) ==                         \
      static_cast<int>(ImGuiKey::KEYNAME - ImGuiKey::ImGuiKey_NamedKey_BEGIN), \
      "Update the NetImgui enum to match the updated Dear ImGui enum"
  static_assert(
      CmdInput::NetImguiKeys::ImGuiKey_COUNT ==
          (ImGuiKey_NamedKey_END - ImGuiKey_NamedKey_BEGIN),
      "Update the NetImgui enum to match the updated Dear ImGui enum");
  static_assert(EnumKeynameTest(ImGuiKey_Tab));
  static_assert(EnumKeynameTest(ImGuiKey_Escape));
  static_assert(EnumKeynameTest(ImGuiKey_RightSuper));
  static_assert(EnumKeynameTest(ImGuiKey_Apostrophe));
  static_assert(EnumKeynameTest(ImGuiKey_Keypad0));
  static_assert(EnumKeynameTest(ImGuiKey_CapsLock));
  static_assert(EnumKeynameTest(ImGuiKey_ReservedForModCtrl));
  static_assert(EnumKeynameTest(ImGuiKey_ReservedForModShift));
  static_assert(EnumKeynameTest(ImGuiKey_ReservedForModAlt));
  static_assert(EnumKeynameTest(ImGuiKey_ReservedForModSuper));
#undef EnumKeynameTest

  // Save every keydown status to out bitmask — only when local UI is not
  // capturing keyboard.
  if (!localWantsKeyboard) {
    uint64_t valueMask(0);
    for (uint32_t i(0); i < ImGuiKey::ImGuiKey_NamedKey_COUNT; ++i) {
      valueMask |=
          ImGui::IsKeyDown(static_cast<ImGuiKey>(ImGuiKey_NamedKey_BEGIN + i))
              ? 0x0000000000000001ull << (i % 64)
              : 0;
      if (((i % 64) == 63) || i == (ImGuiKey::ImGuiKey_NamedKey_COUNT - 1)) {
        cmdInput.mInputDownMask[i / 64] = valueMask;
        valueMask = 0;
      }
    }
  }
  // When local UI captures keyboard, mInputDownMask and mInputAnalog stay
  // zero-initialized from the CmdInput constructor.

  // Copy waiting characters inputs — only when local UI is not capturing
  // keyboard. When captured, pending chars are discarded to avoid buffering
  // stale input that would replay when focus returns to remote.
  if (!localWantsKeyboard) {
    size_t addedKeyCount = std::min<size_t>(
        ArrayCount(cmdInput.mKeyChars) - cmdInput.mKeyCharCount,
        pending_input_chars_.size());
    if (addedKeyCount) {
      memcpy(&cmdInput.mKeyChars[cmdInput.mKeyCharCount],
             &pending_input_chars_[0], addedKeyCount * sizeof(ImWchar));
      cmdInput.mKeyCharCount += static_cast<uint16_t>(addedKeyCount);
      size_t charRemainCount = pending_input_chars_.size() - addedKeyCount;
      if (charRemainCount > 0) {
        memcpy(&pending_input_chars_[0], &pending_input_chars_[addedKeyCount],
               charRemainCount * sizeof(ImWchar));
      }
      pending_input_chars_.resize(charRemainCount);
    }
    if (cmdInput.mKeyCharCount > 0) {
      VLOG(1, "[ui_link.cc] Queued %u characters to send\n",
           cmdInput.mKeyCharCount);
    }
  } else {
    // Discard any pending characters that were accumulated while local UI
    // had keyboard focus.
    pending_input_chars_.clear();
  }

  PendingCom pendingSend;
  pendingSend.pCommand = &cmdInput;
  pendingSend.SizeCurrent = 0;
  Network::DataSend(socket_, pendingSend);
  if (pendingSend.IsDone() && cmdInput.mCompressionSkip) {
    request_keyframe_ = false;
  }
}

// Keep in sync with ProcessCmdDrawFrame in the vendored
// netimgui/Code/ServerApp/Source/NetImguiServer_RemoteClient.cpp
void UiLink::ProcessCmdDrawFrame(CmdDrawFrame* pCmdDrawFrame) {
  if (!pCmdDrawFrame) return;

  // Take ownership to prevent pending_rcv_ from deleting it prematurely.
  pending_rcv_.bAutoFree = false;
  pCmdDrawFrame->ToPointers();

  if (pCmdDrawFrame->mCompressed) {
    if (last_uncompressed_frame_ != nullptr &&
        (last_uncompressed_frame_->mFrameIndex + 1) ==
            pCmdDrawFrame->mFrameIndex) {
      CmdDrawFrame* pUncompressedFrame =
          DecompressCmdDrawFrame(last_uncompressed_frame_, pCmdDrawFrame);
      netImguiDeleteSafe(pCmdDrawFrame);
      pCmdDrawFrame = pUncompressedFrame;
    } else {
      // Missing previous / reference frame data. Ignore this delta-encoded
      // drawframe and ask the client for a fresh uncompressed keyframe.
      request_keyframe_ = true;
      netImguiDeleteSafe(pCmdDrawFrame);
      return;
    }
  }

  // Release previous cached frame and store current for the next delta
  // decompression.
  netImguiDeleteSafe(last_uncompressed_frame_);
  last_uncompressed_frame_ = pCmdDrawFrame;

  pCmdDrawFrame->ToPointers();

  NetImguiImDrawData* pDrawData = netImguiNew<NetImguiImDrawData>();
  pDrawData->mFrameIndex = pCmdDrawFrame->mFrameIndex;
  pDrawData->Valid = true;
  pDrawData->TotalVtxCount =
      static_cast<int>(pCmdDrawFrame->mTotalVerticeCount);
  pDrawData->TotalIdxCount = static_cast<int>(pCmdDrawFrame->mTotalIndiceCount);

  pDrawData->DisplayPos.x = pCmdDrawFrame->mDisplayArea[0];
  pDrawData->DisplayPos.y = pCmdDrawFrame->mDisplayArea[1];
  pDrawData->DisplaySize.x =
      pCmdDrawFrame->mDisplayArea[2] - pCmdDrawFrame->mDisplayArea[0];
  pDrawData->DisplaySize.y =
      pCmdDrawFrame->mDisplayArea[3] - pCmdDrawFrame->mDisplayArea[1];
  pDrawData->FramebufferScale = ImGui::GetIO().DisplayFramebufferScale;

  ImDrawList* pCmdList = pDrawData->CmdLists[0];
  pCmdList->IdxBuffer.resize(pCmdDrawFrame->mTotalIndiceCount);
  pCmdList->VtxBuffer.resize(pCmdDrawFrame->mTotalVerticeCount);
  pCmdList->CmdBuffer.resize(pCmdDrawFrame->mTotalDrawCount);
  // ImVector::resize() doesn't call constructors. Zero-init to ensure
  // TexRef._TexData is NULL, not garbage.
  memset(pCmdList->CmdBuffer.Data, 0,
         pCmdList->CmdBuffer.Size * sizeof(ImDrawCmd));
  pCmdList->Flags =
      ImDrawListFlags_AllowVtxOffset | ImDrawListFlags_AntiAliasedLines |
      ImDrawListFlags_AntiAliasedFill | ImDrawListFlags_AntiAliasedLinesUseTex;

  constexpr float kPosRangeMin = static_cast<float>(ImguiVert::kPosRange_Min);
  constexpr float kPosRangeMax = static_cast<float>(ImguiVert::kPosRange_Max);
  constexpr float kUVRangeMin = static_cast<float>(ImguiVert::kUvRange_Min);
  constexpr float kUVRangeMax = static_cast<float>(ImguiVert::kUvRange_Max);

  if (pCmdDrawFrame->mTotalDrawCount != 0) {
    // WebGL 1.0/2.0 often uses uint16_t for ImDrawIdx. Check for potential
    // overflow if the total vertex count exceeds the limit for 16-bit indices.
    if (sizeof(ImDrawIdx) == 2 && pCmdDrawFrame->mTotalVerticeCount > 65536) {
      fprintf(stderr,
              "WARNING: NetImgui WASM Viewer received a draw frame with %u "
              "vertices. This exceeds the maximum of 65536 for 16-bit "
              "ImDrawIdx, potentially causing rendering artifacts due to index "
              "wrapping.\n",
              pCmdDrawFrame->mTotalVerticeCount);
    }
    uint32_t indexOffset(0), vertexOffset(0);
    ImDrawIdx* pIndexDst = &pCmdList->IdxBuffer[0];
    ImDrawVert* pVertexDst = &pCmdList->VtxBuffer[0];
    ImDrawCmd* pCommandDst = &pCmdList->CmdBuffer[0];
    // On spectators, fit the driver-sized layout to this window.
    const RemoteTransform xform = GetRemoteTransform();

    for (uint32_t i(0); i < pCmdDrawFrame->mDrawGroupCount; ++i) {
      const ImguiDrawGroup& drawGroup = pCmdDrawFrame->mpDrawGroups[i];

      // Indices
      const uint16_t* pIndices =
          reinterpret_cast<const uint16_t*>(drawGroup.mpIndices.Get());
      if (drawGroup.mBytePerIndex == sizeof(ImDrawIdx)) {
        memcpy(pIndexDst, pIndices, drawGroup.mIndiceCount * sizeof(ImDrawIdx));
      } else {
        for (uint32_t indexIdx(0); indexIdx < drawGroup.mIndiceCount;
             ++indexIdx) {
          pIndexDst[indexIdx] = static_cast<ImDrawIdx>(pIndices[indexIdx]);
        }
      }

      // Vertices — unpack quantized positions and UVs.
      const ImguiVert* pVertexSrc = drawGroup.mpVertices.Get();
      for (uint32_t vtxIdx(0); vtxIdx < drawGroup.mVerticeCount; ++vtxIdx) {
        const float px =
            (static_cast<float>(pVertexSrc[vtxIdx].mPos[0]) *
             (kPosRangeMax - kPosRangeMin)) /
                static_cast<float>(0xFFFF) +
            kPosRangeMin + drawGroup.mReferenceCoord[0];
        const float py =
            (static_cast<float>(pVertexSrc[vtxIdx].mPos[1]) *
             (kPosRangeMax - kPosRangeMin)) /
                static_cast<float>(0xFFFF) +
            kPosRangeMin + drawGroup.mReferenceCoord[1];
        pVertexDst[vtxIdx].pos.x = px * xform.scale + xform.offset_x;
        pVertexDst[vtxIdx].pos.y = py * xform.scale + xform.offset_y;
        pVertexDst[vtxIdx].uv.x =
            (static_cast<float>(pVertexSrc[vtxIdx].mUV[0]) *
             (kUVRangeMax - kUVRangeMin)) /
                static_cast<float>(0xFFFF) +
            kUVRangeMin;
        pVertexDst[vtxIdx].uv.y =
            (static_cast<float>(pVertexSrc[vtxIdx].mUV[1]) *
             (kUVRangeMax - kUVRangeMin)) /
                static_cast<float>(0xFFFF) +
            kUVRangeMin;
        pVertexDst[vtxIdx].col = pVertexSrc[vtxIdx].mColor;
      }

      // Draw commands.
      // NOTE: WebGL lacks glDrawElementsBaseVertex, so the backend's
      // glDrawElements ignores VtxOffset. We bake the offset directly
      // into the index values.
      const ImguiDraw* pDrawSrc = drawGroup.mpDraws.Get();
      for (uint32_t drawIdx(0); drawIdx < drawGroup.mDrawCount; ++drawIdx) {
        uint32_t vtxOff = pDrawSrc[drawIdx].mVtxOffset + vertexOffset;
        uint32_t idxOff = pDrawSrc[drawIdx].mIdxOffset + indexOffset;
        uint32_t elemCount = pDrawSrc[drawIdx].mIdxCount;

        // Bake vertex offset into index values.
        for (uint32_t ei = 0; ei < elemCount; ++ei) {
          pCmdList->IdxBuffer[idxOff + ei] += static_cast<ImDrawIdx>(vtxOff);
        }

        const float rx0 =
            pDrawSrc[drawIdx].mClipRect[0] * xform.scale + xform.offset_x;
        const float ry0 =
            pDrawSrc[drawIdx].mClipRect[1] * xform.scale + xform.offset_y;
        const float rx1 =
            pDrawSrc[drawIdx].mClipRect[2] * xform.scale + xform.offset_x;
        const float ry1 =
            pDrawSrc[drawIdx].mClipRect[3] * xform.scale + xform.offset_y;
        float cx = std::max(0.f, std::min(max_clip_[0], rx0));
        float cy = std::max(0.f, std::min(max_clip_[1], ry0));
        float cz = std::max(cx, std::min(max_clip_[0], rx1));
        float cw = std::max(cy, std::min(max_clip_[1], ry1));

        pCommandDst[drawIdx].ClipRect.x = cx;
        pCommandDst[drawIdx].ClipRect.y = cy;
        pCommandDst[drawIdx].ClipRect.z = cz;
        pCommandDst[drawIdx].ClipRect.w = cw;
        pCommandDst[drawIdx].VtxOffset = 0;  // Baked into indices
        pCommandDst[drawIdx].IdxOffset = idxOff;
        pCommandDst[drawIdx].ElemCount = elemCount;
        pCommandDst[drawIdx].UserCallback = nullptr;
        pCommandDst[drawIdx].UserCallbackData = nullptr;

        // Map remote ClientTextureID -> local GL handle.
        ClientTextureID clientTexId = pDrawSrc[drawIdx].mClientTexId;
        auto it = texture_map_.find(clientTexId);
        if (it != texture_map_.end() && it->second != 0) {
          pCommandDst[drawIdx].TexRef._TexData = nullptr;
          pCommandDst[drawIdx].TexRef._TexID =
              static_cast<ImTextureID>(it->second);
        } else {
          log_unmapped_texture(clientTexId, texture_map_.size(), drawIdx,
                               texture_map_);
          // Skip draw commands with unmapped textures.
          pCommandDst[drawIdx].ElemCount = 0;
        }
      }

      pIndexDst += drawGroup.mIndiceCount;
      pVertexDst += drawGroup.mVerticeCount;
      pCommandDst += drawGroup.mDrawCount;
      indexOffset += drawGroup.mIndiceCount;
      vertexOffset += drawGroup.mVerticeCount;
    }
  }

  // Update internal write pointers to satisfy ImGui's draw list sanity checks.
  // AddDrawListToDrawDataEx asserts that _VtxWritePtr/_IdxWritePtr point to the
  // end of their respective buffers. Since we populated them via
  // resize()+memcpy (bypassing ImGui's PrimReserve API), we must fix up these
  // pointers manually.
  pCmdList->_VtxWritePtr = pCmdList->VtxBuffer.Data + pCmdList->VtxBuffer.Size;
  pCmdList->_IdxWritePtr = pCmdList->IdxBuffer.Data + pCmdList->IdxBuffer.Size;
  pCmdList->_VtxCurrentIdx = pCmdList->VtxBuffer.Size;

  if (remote_draw_data_ == nullptr) {
    LOG(Info, "First remote draw frame applied (%d cmds, %d vtx)",
        pDrawData->CmdLists[0]->CmdBuffer.Size, pDrawData->TotalVtxCount);
  }
  if (remote_draw_data_) netImguiDelete(remote_draw_data_);
  remote_draw_data_ = pDrawData;
}

void UiLink::Shutdown() {
  if (remote_draw_data_) {
    netImguiDelete(remote_draw_data_);
    remote_draw_data_ = nullptr;
  }
  netImguiDeleteSafe(last_uncompressed_frame_);
  if (socket_) {
    Network::Disconnect(socket_);
    socket_ = nullptr;
  }
}

}  // namespace mujoco::studio
