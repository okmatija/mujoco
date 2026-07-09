// NetImgui WASM Viewer — renders remote ImGui draw data in a browser.
// Bridges the NetImgui server protocol to a WebGL/SDL2/Emscripten rendering
// context.

#include <SDL.h>
#include <SDL_opengl.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/time.h>
#include <time.h>

#include <algorithm>
#include <cinttypes>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <imgui.h>
#include <implot.h>
#include <mujoco/mujoco.h>
#include "NetImgui_Api.h"
#include "NetImgui_CmdPackets.h"
#include "NetImgui_Network.h"
#include "experimental/platform/hal/renderer.h"
#include "experimental/platform/hal/window.h"
#include "experimental/platform/sim/model_holder.h"
#include "experimental/platform/ux/interaction.h"
#include "google/logging.h"
#include "live_state.h"

#if defined(__EMSCRIPTEN__)
#include <emscripten.h>
#include <emscripten/fetch.h>
#include <emscripten/websocket.h>

// No-op wrapper for glGetError(), installed via -Wl,--wrap=glGetError.
// Filament's GL backend calls glGetError() hundreds of times per frame
// through GLUtils::checkGLError/assertGLError. In WebGL each call forces a
// synchronous GPU pipeline flush across the JS-WASM bridge (~0.2ms each).
// This stub eliminates that cost entirely.
extern "C" GLenum __wrap_glGetError(void) { return GL_NO_ERROR; }
#endif

using mujoco::studio::kLiveStateSize;

struct AppState {
  std::unique_ptr<mujoco::platform::Window> window;
  std::unique_ptr<mujoco::platform::ModelHolder> model_holder;
  mujoco::platform::Renderer* renderer = nullptr;
  mjvPerturb perturb;
  mjvCamera camera;
  mjvOption vis_options;
  int camera_idx = 0;

  // Backend state received from the Python simulation via WebSocket.
  std::vector<mjtNum> backend_state;
  int backend_state_sig = 0;
  bool backend_state_dirty = false;
};
AppState g_app;

//=================================================================================================
// State WebSocket — receives simulation state from the Python Link server.
//=================================================================================================
#if defined(__EMSCRIPTEN__)
EMSCRIPTEN_WEBSOCKET_T gStateSocket = 0;

static mjvCamera s_last_received_camera = {};
static bool s_received_first_camera = false;

// Data transfer rate tracking stats.
static double s_last_rate_time = 0;
static uint64_t s_gui_bytes_accum = 0;
static uint64_t s_gui_bytes_per_sec = 0;
static uint64_t s_sim_bytes_accum = 0;
static uint64_t s_sim_bytes_per_sec = 0;
static bool gUseCompression = true;

EM_BOOL OnStateWsMessage(int eventType,
                         const EmscriptenWebSocketMessageEvent* wsEvent,
                         void* userData) {
  if (!wsEvent->isText && wsEvent->numBytes > 4 && g_app.model_holder &&
      g_app.model_holder->ok()) {
    s_sim_bytes_accum += wsEvent->numBytes;
    const uint8_t* data = wsEvent->data;
    // Parse: [4 bytes sig (int32 LE)] [N*8 bytes physics state] [Live state]
    int32_t sig;
    memcpy(&sig, data, 4);

    int payload_bytes = wsEvent->numBytes - 4;

    // Determine how many bytes are physics state vs Live state.
    // If the payload includes Live state, subtract its fixed size.
    int physics_bytes = payload_bytes;
    bool has_live_state = false;
    if (payload_bytes > kLiveStateSize) {
      physics_bytes = payload_bytes - kLiveStateSize;
      has_live_state = true;
    }

    int state_count = physics_bytes / sizeof(mjtNum);
    g_app.backend_state.resize(state_count);
    memcpy(g_app.backend_state.data(), data + 4, state_count * sizeof(mjtNum));
    g_app.backend_state_sig = sig;
    g_app.backend_state_dirty = true;

    // Apply Live state if present.
    if (has_live_state) {
      const uint8_t* vis_ptr = data + 4 + physics_bytes;
      mjModel* model = g_app.model_holder->model();

      mjvCamera incoming_cam;
      memcpy(&incoming_cam, vis_ptr, sizeof(mjvCamera));
      if (!s_received_first_camera ||
          memcmp(&incoming_cam, &s_last_received_camera, sizeof(mjvCamera)) !=
              0) {
        memcpy(&g_app.camera, &incoming_cam, sizeof(mjvCamera));
        memcpy(&s_last_received_camera, &incoming_cam, sizeof(mjvCamera));
        s_received_first_camera = true;
      }
      vis_ptr += sizeof(mjvCamera);

      memcpy(&g_app.vis_options, vis_ptr, sizeof(mjvOption));
      vis_ptr += sizeof(mjvOption);

      memcpy(&model->opt, vis_ptr, sizeof(mjOption));
      vis_ptr += sizeof(mjOption);

      memcpy(&model->vis, vis_ptr, sizeof(mjVisual));
      vis_ptr += sizeof(mjVisual);

      memcpy(&model->stat, vis_ptr, sizeof(mjStatistic));
      vis_ptr += sizeof(mjStatistic);

      // Apply render flags to the renderer's scene if available.
      if (g_app.renderer) {
        memcpy(g_app.renderer->GetRenderFlags(), vis_ptr, mjNRNDFLAG);
      }
    }
  }
  return EM_TRUE;
}

EM_BOOL OnStateWsOpen(int eventType,
                      const EmscriptenWebSocketOpenEvent* wsEvent,
                      void* userData) {
  LOG(Info, "State WebSocket connected");
  return EM_TRUE;
}

EM_BOOL OnStateWsError(int eventType,
                       const EmscriptenWebSocketErrorEvent* wsEvent,
                       void* userData) {
  LOG(Error, "State WebSocket error");
  return EM_TRUE;
}

EM_BOOL OnStateWsClose(int eventType,
                       const EmscriptenWebSocketCloseEvent* wsEvent,
                       void* userData) {
  LOG(Info, "State WebSocket closed (code=%d)", wsEvent->code);
  gStateSocket = 0;
  return EM_TRUE;
}

void ConnectStateWebSocket() {
  // Build URL relative to the page origin, using port 8891.
  char url[256];
  snprintf(url, sizeof(url), "ws://%s:8891", "localhost");

  EmscriptenWebSocketCreateAttributes attr;
  emscripten_websocket_init_create_attributes(&attr);
  attr.url = url;
  attr.protocols = nullptr;
  attr.createOnMainThread = EM_TRUE;

  gStateSocket = emscripten_websocket_new(&attr);
  if (gStateSocket <= 0) {
    LOG(Error, "Failed to create state WebSocket");
    return;
  }
  emscripten_websocket_set_onopen_callback(gStateSocket, nullptr,
                                           OnStateWsOpen);
  emscripten_websocket_set_onmessage_callback(gStateSocket, nullptr,
                                              OnStateWsMessage);
  emscripten_websocket_set_onerror_callback(gStateSocket, nullptr,
                                            OnStateWsError);
  emscripten_websocket_set_onclose_callback(gStateSocket, nullptr,
                                            OnStateWsClose);
  LOG(Info, "State WebSocket connecting to %s", url);
}
#endif

using namespace NetImgui::Internal;

// Forward-declare Google-specific function implemented in
// NetImgui_NetworkWASM.cpp.
namespace NetImgui {
namespace Internal {
namespace Network {
const char* GetStatusString(SocketInfo* pClientSocket);
}
}  // namespace Internal
}  // namespace NetImgui

//=================================================================================================
// ImDrawData wrapper for receiving remote draw data.
// Based on NetImguiImDrawData in
// //third_party/netimgui/Code/ServerApp/Source/NetImguiServer_RemoteClient.cpp
//=================================================================================================
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

//=================================================================================================
// Globals
//=================================================================================================
Network::SocketInfo* gClientSocket = nullptr;
NetImguiImDrawData* gRemoteDrawData = nullptr;
PendingCom gPendingRcv;
CmdPendingRead gCmdPendingRead;

std::unordered_map<ClientTextureID, uintptr_t> gTextureMap;

// CPU-side mirror of each texture's RGBA pixel data. Filament doesn't support
// sub-region uploads (no glTexSubImage2D equivalent), so partial updates from
// NetImgui must be patched into this buffer before re-uploading the full
// texture.
struct TextureEntry {
  std::vector<uint8_t> pixels;  // Full RGBA pixel data
  uint32_t width = 0;
  uint32_t height = 0;
};
std::unordered_map<ClientTextureID, TextureEntry> gTextureCPU;

// Persistent input state for character accumulation across frames.
// Mirrors mPendingInputChars/mMouseWheelPos in NetImguiServer RemoteClient.
std::vector<ImWchar> gPendingInputChars;
float gMouseWheelPos[2] = {0, 0};

// When true, the next CmdInput asks the client to send one uncompressed
// draw frame (CmdInput::mCompressionSkip). Set on (re)connect and whenever
// a delta frame arrives whose reference frame we don't have — without it,
// a browser joining mid-stream drops every delta-compressed frame forever
// and the remote UI never appears. Mirrors mbCompressionSkipOncePending in
// NetImguiServer RemoteClient.
bool gRequestKeyframe = true;

//=================================================================================================
// Debug logging functions — all diagnostic output is centralized here.
// Each function encapsulates its own throttling logic so call sites stay clean.
//=================================================================================================

void debug_log_texture(ClientTextureID texID, int status, uint32_t w,
                       uint32_t h, int format, uint32_t offX, uint32_t offY,
                       size_t mapSize) {
  VLOG(1,
       "ProcessCmdTexture: clientTexID=%lu, status=%d, size=%ux%u, format=%d, "
       "offset=%u,%u, mapSize=%zu",
       static_cast<unsigned long>(texID), status, w, h, format, offX, offY,
       mapSize);
}

void debug_log_texture_create(ClientTextureID texID, uintptr_t filamentTex) {
  VLOG(1, "uploadGuiImage for clientTexID=%lu -> filament=%lu",
       static_cast<unsigned long>(texID),
       static_cast<unsigned long>(filamentTex));
}

void debug_log_unmapped_texture(
    ClientTextureID clientTexId, size_t mapSize, uint32_t drawIdx,
    const std::unordered_map<ClientTextureID, uintptr_t>& texMap) {
  VLOG(1, "DrawFrame: UNMAPPED clientTexId=%lu, mapSize=%zu, draw#=%u",
       static_cast<unsigned long>(clientTexId), mapSize, drawIdx);
  std::string keys_str = "";
  for (auto& kv : texMap) {
    keys_str += " " + std::to_string(kv.first) +
                "(fil=" + std::to_string(kv.second) + ")";
  }
  VLOG(1, "  map keys:%s", keys_str.c_str());
}

void debug_log_status_change(const char* oldStatus, const char* newStatus,
                             int frame) {
  LOG(Info, "WebSocket status changed: '%s' -> '%s' (frame %d)",
      oldStatus ? oldStatus : "null", newStatus, frame);
}

void debug_log_periodic_stats(int frame, const char* status, bool handshakeSent,
                              int cmds, int draws, int textures,
                              bool hasDrawData) {
  VLOG(1,
       "Frame %d: status='%s', handshake=%s, cmds=%d, draws=%d, textures=%d, "
       "hasDrawData=%s",
       frame, status, handshakeSent ? "sent" : "not_sent", cmds, draws,
       textures, hasDrawData ? "yes" : "no");
}

void debug_log_handshake_send(const CmdVersion& v) {
  LOG(Info,
      "Sending CmdVersion handshake: size=%u, type=%d, version=%d, "
      "wcharSize=%d, name='%s'",
      v.mSize, static_cast<int>(v.mType), static_cast<int>(v.mVersion),
      v.mWCharSize, v.mClientName);
}

void debug_log_handshake_result(bool done, bool error, int attempts,
                                size_t bytesSent) {
  LOG(Info, "CmdVersion send: done=%s, error=%s, attempts=%d, bytesSent=%zu",
      done ? "true" : "false", error ? "true" : "false", attempts, bytesSent);
}

void debug_log_handshake_reset(const char* status) {
  LOG(Info, "Resetting handshake (status='%s')", status);
}

void debug_log_connection_lost(const char* status) {
  LOG(Info, "Connection lost (status='%s'). Discarding buffered data.", status);
}

void debug_log_data_pending(int frame) {
  VLOG(1, "Frame %d: data pending on socket", frame);
}

void debug_log_cmd_allocate(uint32_t size, int type) {
  VLOG(2, "Allocating %u bytes for incoming cmd type=%d", size, type);
}

void debug_log_receive_error(uint32_t cmdSize, size_t got, int type) {
  LOG(Error, "Receive ERROR: cmdSize=%u, got=%zu, type=%d", cmdSize, got, type);
}

void debug_log_cmd_received(CmdHeader::eCommands cmdType, uint32_t cmdSize,
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

void debug_log_frame_summary(int frame, int cmds) {
  VLOG(2, "Frame %d: processed %d commands", frame, cmds);
}

void debug_log_connect(const char* host, int port) {
  LOG(Info, "Connecting to WebSocket at %s:%d", host, port);
}

void debug_log_connect_result(Network::SocketInfo* socket) {
  LOG(Info, "Network::Connect returned: socket=%p", static_cast<void*>(socket));
}

//=================================================================================================
// Returns true if the Filament rendering context is initialized and ready for
// GPU texture uploads. The Renderer object is created in main() and is always
// non-null, but the Filament context is only initialized when Renderer::Init()
// is called from SetupScene after the async model fetch completes.
//=================================================================================================
bool IsFilamentReady() {
  return g_app.renderer && g_app.model_holder && g_app.model_holder->ok();
}

//=================================================================================================
// Upload all CPU-buffered textures that don't yet have a GPU handle. Called
// from SetupScene once the Filament context becomes available, to flush
// textures (e.g. the font atlas) that arrived before the model was loaded.
//=================================================================================================
void FlushPendingTextures() {
  for (auto& [texID, entry] : gTextureCPU) {
    uintptr_t& localTex = gTextureMap[texID];
    if (localTex == 0 && !entry.pixels.empty()) {
      localTex = g_app.renderer->UploadImage(
          localTex, reinterpret_cast<const std::byte*>(entry.pixels.data()),
          entry.width, entry.height, 4);
      LOG(Info, "FlushPendingTextures: uploaded texID=%lu -> filament=%lu",
          static_cast<unsigned long>(texID),
          static_cast<unsigned long>(localTex));
    }
  }
}

//=================================================================================================
// Process texture commands from the remote client.
//=================================================================================================
void ProcessCmdTexture(CmdTexture* pCmdTexture) {
  if (!pCmdTexture) return;

  if (!pCmdTexture->mpTextureData.IsPointer()) {
    pCmdTexture->mpTextureData.ToPointer();
  }

  ClientTextureID texID = pCmdTexture->mTextureClientID;
  debug_log_texture(
      texID, static_cast<int>(pCmdTexture->mStatus), pCmdTexture->mWidth,
      pCmdTexture->mHeight, static_cast<int>(pCmdTexture->mFormat),
      pCmdTexture->mOffsetX, pCmdTexture->mOffsetY, gTextureMap.size());

  uintptr_t& localTex = gTextureMap[texID];

  if (pCmdTexture->mStatus == CmdTexture::eType::Destroy) {
    if (localTex != 0 && IsFilamentReady()) {
      // Pass nullptr pixels to destroy the Filament texture.
      g_app.renderer->UploadImage(localTex, nullptr, 0, 0, 0);
    }
    localTex = 0;
    gTextureCPU.erase(texID);
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

  TextureEntry& entry = gTextureCPU[texID];

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

  // Upload the full CPU-side texture to Filament if the context is ready.
  // If Filament isn't initialized yet (model still loading), the texture stays
  // in gTextureCPU and will be flushed by FlushPendingTextures() later.
  if (IsFilamentReady()) {
    localTex = g_app.renderer->UploadImage(
        localTex, reinterpret_cast<const std::byte*>(entry.pixels.data()),
        entry.width, entry.height, 4);
    debug_log_texture_create(texID, localTex);
  } else {
    VLOG(1, "ProcessCmdTexture: buffered texID=%lu, deferring GPU upload",
         static_cast<unsigned long>(texID));
  }
}

//=================================================================================================
// Capture current ImGui input and send it to the remote client.
// Must be called before ImGui::EndFrame() since it uses io.InputQueueCharacters
// which clears that data. Note that ImGui::Render() calls ImGui::EndFrame().
// Based on CaptureImguiInput in
// //third_party/netimgui/Code/ServerApp/Source/NetImguiServer_RemoteClient.cpp
//=================================================================================================
void CaptureAndSendInput() {
  if (!gClientSocket) return;

  const char* status = Network::GetStatusString(gClientSocket);
  if (strcmp(status, "Open") != 0) return;

  // Capture input from Dear ImGui.
  const ImGuiIO& io = ImGui::GetIO();

  // When the local UI wants to capture mouse or keyboard, suppress the
  // corresponding input from being forwarded to the remote client. This
  // prevents clicks/keys meant for the local status overlay (or any other
  // local window) from leaking through to the link server.
  const bool localWantsMouse = io.WantCaptureMouse;
  const bool localWantsKeyboard = io.WantCaptureKeyboard;

  {
    // Only accumulate characters when the local UI is NOT capturing keyboard.
    if (!localWantsKeyboard) {
      const size_t initialSize = gPendingInputChars.size();
      const size_t addedChar = io.InputQueueCharacters.size();
      if (addedChar) {
        gPendingInputChars.resize(initialSize + addedChar);
        memcpy(&gPendingInputChars[initialSize], io.InputQueueCharacters.Data,
               addedChar * sizeof(ImWchar));
      }
    }

    // Only accumulate scroll when the local UI is NOT capturing mouse.
    if (!localWantsMouse) {
      gMouseWheelPos[0] += io.MouseWheel;
      gMouseWheelPos[1] += io.MouseWheelH;
    }
  }

  CmdInput cmdInput;
  cmdInput.mScreenSize[0] = static_cast<uint16_t>(io.DisplaySize.x);
  cmdInput.mScreenSize[1] = static_cast<uint16_t>(io.DisplaySize.y);
  // An unstable screen size forces the remote UI to relayout every frame
  // (visible as UI flicker), so make changes loud.
  static uint16_t sLastScreenSize[2] = {0, 0};
  if (cmdInput.mScreenSize[0] != sLastScreenSize[0] ||
      cmdInput.mScreenSize[1] != sLastScreenSize[1]) {
    LOG(Info, "Screen size sent to client changed: %ux%u -> %ux%u",
        sLastScreenSize[0], sLastScreenSize[1], cmdInput.mScreenSize[0],
        cmdInput.mScreenSize[1]);
    sLastScreenSize[0] = cmdInput.mScreenSize[0];
    sLastScreenSize[1] = cmdInput.mScreenSize[1];
  }
  cmdInput.mFontDPIScaling = 1.f;
  cmdInput.mDesiredFps = 60.0f;
  cmdInput.mCompressionUse = gUseCompression;
  cmdInput.mCompressionSkip = gRequestKeyframe;

  // Send mouse position and wheel only when the local UI is not capturing.
  if (!localWantsMouse) {
    cmdInput.mMousePos[0] = static_cast<int16_t>(io.MousePos.x);
    cmdInput.mMousePos[1] = static_cast<int16_t>(io.MousePos.y);
    cmdInput.mMouseWheelVert = gMouseWheelPos[0];
    cmdInput.mMouseWheelHoriz = gMouseWheelPos[1];
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

// Keyboard / Gamepads Inputs
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
  static_assert(EnumKeynameTest(ImGuiKey_GamepadStart));
  static_assert(EnumKeynameTest(ImGuiKey_GamepadLStickUp));
  static_assert(EnumKeynameTest(ImGuiKey_ReservedForModCtrl));
  static_assert(EnumKeynameTest(ImGuiKey_ReservedForModShift));
  static_assert(EnumKeynameTest(ImGuiKey_ReservedForModAlt));
  static_assert(EnumKeynameTest(ImGuiKey_ReservedForModSuper));
  static_assert(EnumKeynameTest(ImGuiKey_GamepadStart));
  static_assert(EnumKeynameTest(ImGuiKey_GamepadR3));
  static_assert(EnumKeynameTest(ImGuiKey_GamepadLStickUp));
  static_assert(EnumKeynameTest(ImGuiKey_GamepadRStickRight));

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
    // Save analog keys (gamepad)
    for (uint32_t i(0); i < CmdInput::kAnalog_Count; ++i) {
      cmdInput.mInputAnalog[i] =
          ImGui::GetIO().KeysData[CmdInput::kAnalog_First + i].AnalogValue;
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
        gPendingInputChars.size());
    if (addedKeyCount) {
      memcpy(&cmdInput.mKeyChars[cmdInput.mKeyCharCount],
             &gPendingInputChars[0], addedKeyCount * sizeof(ImWchar));
      cmdInput.mKeyCharCount += static_cast<uint16_t>(addedKeyCount);
      size_t charRemainCount = gPendingInputChars.size() - addedKeyCount;
      if (charRemainCount > 0) {
        memcpy(&gPendingInputChars[0], &gPendingInputChars[addedKeyCount],
               charRemainCount * sizeof(ImWchar));
      }
      gPendingInputChars.resize(charRemainCount);
    }
    if (cmdInput.mKeyCharCount > 0) {
      VLOG(1, "[web_client.cc] Queued %u characters to send\n",
           cmdInput.mKeyCharCount);
    }
  } else {
    // Discard any pending characters that were accumulated while local UI
    // had keyboard focus.
    gPendingInputChars.clear();
  }

  PendingCom pendingSend;
  pendingSend.pCommand = &cmdInput;
  pendingSend.SizeCurrent = 0;
  Network::DataSend(gClientSocket, pendingSend);
  if (pendingSend.IsDone() && cmdInput.mCompressionSkip) {
    gRequestKeyframe = false;
  }
}

//=================================================================================================
// Process a draw frame command: unpack vertices, indices, and draw commands
// into ImDrawData for rendering by the Dear ImGui OpenGL3 backend.
// Based on ProcessCmdDrawFrame in
// //third_party/netimgui/Code/ServerApp/Source/NetImguiServer_RemoteClient.cpp
//=================================================================================================
static CmdDrawFrame* sLastUncompressedFrame = nullptr;

void ProcessCmdDrawFrame(CmdDrawFrame* pCmdDrawFrame) {
  if (!pCmdDrawFrame) return;

  // Take ownership to prevent gPendingRcv from deleting it prematurely.
  gPendingRcv.bAutoFree = false;
  pCmdDrawFrame->ToPointers();

  if (pCmdDrawFrame->mCompressed) {
    if (sLastUncompressedFrame != nullptr &&
        (sLastUncompressedFrame->mFrameIndex + 1) ==
            pCmdDrawFrame->mFrameIndex) {
      CmdDrawFrame* pUncompressedFrame =
          DecompressCmdDrawFrame(sLastUncompressedFrame, pCmdDrawFrame);
      netImguiDeleteSafe(pCmdDrawFrame);
      pCmdDrawFrame = pUncompressedFrame;
    } else {
      // Missing previous / reference frame data. Ignore this delta-encoded
      // drawframe and ask the client for a fresh uncompressed keyframe.
      gRequestKeyframe = true;
      netImguiDeleteSafe(pCmdDrawFrame);
      return;
    }
  }

  // Release previous cached frame and store current for the next delta
  // decompression.
  netImguiDeleteSafe(sLastUncompressedFrame);
  sLastUncompressedFrame = pCmdDrawFrame;

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
        pVertexDst[vtxIdx].pos.x =
            (static_cast<float>(pVertexSrc[vtxIdx].mPos[0]) *
             (kPosRangeMax - kPosRangeMin)) /
                static_cast<float>(0xFFFF) +
            kPosRangeMin + drawGroup.mReferenceCoord[0];
        pVertexDst[vtxIdx].pos.y =
            (static_cast<float>(pVertexSrc[vtxIdx].mPos[1]) *
             (kPosRangeMax - kPosRangeMin)) /
                static_cast<float>(0xFFFF) +
            kPosRangeMin + drawGroup.mReferenceCoord[1];
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

        float maxWidth = static_cast<float>(g_app.window->GetWidth());
        float maxHeight = static_cast<float>(g_app.window->GetHeight());
        float cx =
            std::max(0.f, std::min(maxWidth, pDrawSrc[drawIdx].mClipRect[0]));
        float cy =
            std::max(0.f, std::min(maxHeight, pDrawSrc[drawIdx].mClipRect[1]));
        float cz =
            std::max(cx, std::min(maxWidth, pDrawSrc[drawIdx].mClipRect[2]));
        float cw =
            std::max(cy, std::min(maxHeight, pDrawSrc[drawIdx].mClipRect[3]));

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
        auto it = gTextureMap.find(clientTexId);
        if (it != gTextureMap.end() && it->second != 0) {
          pCommandDst[drawIdx].TexRef._TexData = nullptr;
          pCommandDst[drawIdx].TexRef._TexID =
              static_cast<ImTextureID>(it->second);
        } else {
          debug_log_unmapped_texture(clientTexId, gTextureMap.size(), drawIdx,
                                     gTextureMap);
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

  if (gRemoteDrawData == nullptr) {
    LOG(Info, "First remote draw frame applied (%d cmds, %d vtx)",
        pDrawData->CmdLists[0]->CmdBuffer.Size, pDrawData->TotalVtxCount);
  }

  // Fingerprint the frame's clip rects: if the fingerprint alternates
  // between values while the UI is idle, the incoming draw data itself
  // alternates (remote/transport bug); if it stays constant while pixels
  // flicker, the local rendering of identical data is at fault.
  {
    uint64_t h = 1469598103934665603ull;
    auto mix = [&h](uint32_t v) {
      h ^= v;
      h *= 1099511628211ull;
    };
    const ImDrawList* dl = pDrawData->CmdLists[0];
    for (int i = 0; i < dl->CmdBuffer.Size; ++i) {
      const ImVec4& r = dl->CmdBuffer[i].ClipRect;
      mix(static_cast<uint32_t>(r.x * 8.f));
      mix(static_cast<uint32_t>(r.y * 8.f));
      mix(static_cast<uint32_t>(r.z * 8.f));
      mix(static_cast<uint32_t>(r.w * 8.f));
      mix(dl->CmdBuffer[i].ElemCount);
    }
    static uint64_t sLastHash = 0;
    static int sHashLogs = 0;
    static uint32_t sFramesSinceChange = 0;
    ++sFramesSinceChange;
    if (h != sLastHash && sHashLogs < 60) {
      LOG(Info,
          "[drawhash] %08x%08x cmds=%d vtx=%d (stable for %u frames)",
          static_cast<uint32_t>(h >> 32), static_cast<uint32_t>(h),
          dl->CmdBuffer.Size, pDrawData->TotalVtxCount, sFramesSinceChange);
      sLastHash = h;
      sHashLogs++;
      sFramesSinceChange = 0;
    }
  }

  if (gRemoteDrawData) netImguiDelete(gRemoteDrawData);
  gRemoteDrawData = pDrawData;
}

// BUG: This function fails to detect mouse over window frames (title bars,
// scrollbars) because those elements often use full-screen clipping rectangles
// in the remote draw data, which are ignored by the is_fullscreen check.
// Interactive elements inside windows work correctly.
bool IsMouseOverRemoteUI() {
  if (!gRemoteDrawData) return false;
  ImGuiIO& io = ImGui::GetIO();
  ImVec2 mouse_pos = io.MousePos;

  // Ignore invalid mouse positions
  if (mouse_pos.x < 0 || mouse_pos.y < 0) return false;

  for (int i = 0; i < gRemoteDrawData->CmdListsCount; ++i) {
    const ImDrawList* cmd_list = gRemoteDrawData->CmdLists[i];
    for (int j = 0; j < cmd_list->CmdBuffer.Size; ++j) {
      const ImDrawCmd& cmd = cmd_list->CmdBuffer[j];
      if (cmd.ElemCount > 0) {
        // Ignore clip rects that cover the entire display area (likely
        // background)
        bool is_fullscreen =
            (cmd.ClipRect.x <= gRemoteDrawData->DisplayPos.x &&
             cmd.ClipRect.y <= gRemoteDrawData->DisplayPos.y &&
             cmd.ClipRect.z >= gRemoteDrawData->DisplayPos.x +
                                   gRemoteDrawData->DisplaySize.x &&
             cmd.ClipRect.w >= gRemoteDrawData->DisplayPos.y +
                                   gRemoteDrawData->DisplaySize.y);

        // Relaxed check: ignore clip rects that cover more than 80% of the
        // screen. This prevents the Dockspace window from blocking camera
        // controls.
        float cmd_area = (cmd.ClipRect.z - cmd.ClipRect.x) *
                         (cmd.ClipRect.w - cmd.ClipRect.y);
        float screen_area =
            gRemoteDrawData->DisplaySize.x * gRemoteDrawData->DisplaySize.y;
        bool is_large_background = (cmd_area / screen_area) > 0.8f;

        if (!is_fullscreen && !is_large_background) {
          if (mouse_pos.x >= cmd.ClipRect.x && mouse_pos.x <= cmd.ClipRect.z &&
              mouse_pos.y >= cmd.ClipRect.y && mouse_pos.y <= cmd.ClipRect.w) {
            return true;
          }
        }
      }
    }
  }
  return false;
}

void UpdateAndShowTelemetryGUI() {
  const double now = ImGui::GetTime();
  if (now - s_last_rate_time >= 1.0) {
    s_gui_bytes_per_sec = s_gui_bytes_accum;
    s_sim_bytes_per_sec = s_sim_bytes_accum;
    s_gui_bytes_accum = 0;
    s_sim_bytes_accum = 0;
    s_last_rate_time = now;
  }

  static bool s_expanded = false;
  if (!s_expanded) {
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(6.0f, 6.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowMinSize, ImVec2(16.0f, 16.0f));

    ImGui::Begin("NetImgui", nullptr,
                 ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                     ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::Text("NetImGui");
    if (ImGui::IsItemHovered()) {
      if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
        s_expanded = true;
      }
      ImGui::SetTooltip("Double-click toggle telemetry");
    }
    ImGui::End();

    ImGui::PopStyleVar(2);
  } else {
    bool p_open = true;
    bool should_collapse = false;

    ImGui::Begin(
        "NetImgui", &p_open,
        ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoCollapse);
    if (ImGui::IsWindowHovered() &&
        ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
      should_collapse = true;
    }

    ImGui::Text("Connection: %s", Network::GetStatusString(gClientSocket));
    ImGui::Text("Remote Frame: %s", gRemoteDrawData ? "Received" : "None");
    ImGui::Text("Data Rate (GUI): %" PRIu64 " KiB/s",
                static_cast<uint64_t>(s_gui_bytes_per_sec / 1024));
    ImGui::Text("Data Rate (Sim): %" PRIu64 " KiB/s",
                static_cast<uint64_t>(s_sim_bytes_per_sec / 1024));
    if (ImGui::Checkbox("Use Compression", &gUseCompression)) {
      NetImgui::SetCompressionMode(gUseCompression ? NetImgui::kForceEnable
                                                   : NetImgui::kForceDisable);
    }
    if (!gRemoteDrawData) {
      ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f),
                         "Waiting for Draw Data...");
    }
    ImGui::End();

    if (!p_open || should_collapse) {
      s_expanded = false;
    }
  }
}

//=================================================================================================
// Main loop — called once per frame.
//=================================================================================================
void MainLoopImpl();

// An exception escaping the RAF callback kills the main loop silently (the
// canvas freezes on the last rendered frame and input capture stops, with
// only an opaque "Uncaught <ptr>" in the console). Catch, log, and stop
// explicitly instead.
void MainLoop() {
  try {
    MainLoopImpl();
  } catch (const std::exception& e) {
    LOG(Error, "FATAL: uncaught exception in MainLoop: %s", e.what());
#if defined(__EMSCRIPTEN__)
    emscripten_cancel_main_loop();
#endif
  } catch (...) {
    LOG(Error, "FATAL: uncaught non-std exception in MainLoop");
#if defined(__EMSCRIPTEN__)
    emscripten_cancel_main_loop();
#endif
  }
}

void MainLoopImpl() {
  static bool sHandshakeSent = false;
  static int sMainFrameCount = 0;
  static int sTotalCmdsReceived = 0;
  static int sDrawFramesReceived = 0;
  static int sTexturesReceived = 0;
  sMainFrameCount++;

  // =========================================================================
  // Phase 1: Network — process incoming data BEFORE the ImGui frame.
  // This is the equivalent of NetImguiServer::App::UpdateClientDraw() in the
  // GlfwGL3 server app. Textures and draw frames must be ready before we
  // start the local ImGui frame and render.
  // =========================================================================
  if (gClientSocket) {
    const char* status = Network::GetStatusString(gClientSocket);

    static const char* sLastStatus = nullptr;
    if (sLastStatus != status) {
      debug_log_status_change(sLastStatus, status, sMainFrameCount);
      sLastStatus = status;
    }
    debug_log_periodic_stats(sMainFrameCount, status, sHandshakeSent,
                             sTotalCmdsReceived, sDrawFramesReceived,
                             sTexturesReceived, gRemoteDrawData != nullptr);

    if (strcmp(status, "Open") == 0) {
      if (!sHandshakeSent) {
        CmdVersion cmdVersion;
        StringCopy(cmdVersion.mClientName, "MuJoCo Live");
        debug_log_handshake_send(cmdVersion);

        PendingCom pendingSend;
        pendingSend.pCommand = &cmdVersion;
        pendingSend.SizeCurrent = 0;

        int sendAttempts = 0;
        while (!pendingSend.IsDone() && !pendingSend.IsError()) {
          Network::DataSend(gClientSocket, pendingSend);
          sendAttempts++;
        }
        debug_log_handshake_result(pendingSend.IsDone(), pendingSend.IsError(),
                                   sendAttempts, pendingSend.SizeCurrent);
        sHandshakeSent = true;
        // Fresh connection: the client may resume delta compression against
        // a frame from a previous session; ask for an uncompressed keyframe.
        gRequestKeyframe = true;
      }

      // Input is sent after SDL events (Phase 2) so io.MouseWheel
      // contains the current frame's scroll delta. See below.

    } else {
      if (sHandshakeSent) {
        debug_log_handshake_reset(status);
      }
      sHandshakeSent = false;
    }
  }

  // Network Receive — drain all available data in one frame.
  if (gClientSocket) {
    const char* netStatus = Network::GetStatusString(gClientSocket);
    bool isConnected = (strcmp(netStatus, "Open") == 0);

    // If the connection has closed, discard any buffered data immediately
    // rather than churning through stale commands for several seconds.
    static bool sWasConnected = false;
    if (sWasConnected && !isConnected) {
      debug_log_connection_lost(netStatus);
      // Reset any in-progress receive.
      if (gPendingRcv.bAutoFree) netImguiDeleteSafe(gPendingRcv.pCommand);
      gPendingRcv = PendingCom();
      sWasConnected = false;
    }
    if (isConnected) sWasConnected = true;

    int maxCommandsPerFrame = 64;
    int cmdsThisFrame = 0;
    bool hadPendingData = Network::DataReceivePending(gClientSocket);
    if (hadPendingData) {
      debug_log_data_pending(sMainFrameCount);
    }

    while (isConnected && maxCommandsPerFrame-- > 0) {
      if (gPendingRcv.IsReady()) {
        gCmdPendingRead = CmdPendingRead();
        gPendingRcv.pCommand = &gCmdPendingRead;
        gPendingRcv.bAutoFree = false;
      }

      if (!Network::DataReceivePending(gClientSocket)) break;

      Network::DataReceive(gClientSocket, gPendingRcv);

      if (gPendingRcv.pCommand->mSize > sizeof(CmdPendingRead) &&
          gPendingRcv.pCommand == &gCmdPendingRead) {
        debug_log_cmd_allocate(gPendingRcv.pCommand->mSize,
                               static_cast<int>(gPendingRcv.pCommand->mType));
        CmdPendingRead* pCmdHeader = reinterpret_cast<CmdPendingRead*>(
            netImguiSizedNew<uint8_t>(gPendingRcv.pCommand->mSize));
        *pCmdHeader = gCmdPendingRead;
        gPendingRcv.pCommand = pCmdHeader;
        gPendingRcv.bAutoFree = true;
      }

      if (!gPendingRcv.IsDone()) {
        if (gPendingRcv.IsError()) {
          debug_log_receive_error(
              gPendingRcv.pCommand->mSize, gPendingRcv.SizeCurrent,
              static_cast<int>(gPendingRcv.pCommand->mType));
          if (gPendingRcv.bAutoFree) netImguiDeleteSafe(gPendingRcv.pCommand);
          gPendingRcv = PendingCom();
        }
        continue;
      }

      // Command fully received — dispatch.
      s_gui_bytes_accum += gPendingRcv.pCommand->mSize;
      cmdsThisFrame++;
      sTotalCmdsReceived++;
      CmdHeader::eCommands cmdType = gPendingRcv.pCommand->mType;
      debug_log_cmd_received(cmdType, gPendingRcv.pCommand->mSize,
                             sDrawFramesReceived, sTexturesReceived,
                             gPendingRcv);

      if (cmdType == CmdHeader::eCommands::Count) {
        // CmdPendingRead sentinel — skip silently.
      } else if (cmdType == CmdHeader::eCommands::DrawFrame) {
        sDrawFramesReceived++;
        ProcessCmdDrawFrame(
            reinterpret_cast<CmdDrawFrame*>(gPendingRcv.pCommand));
      } else if (cmdType == CmdHeader::eCommands::Texture) {
        sTexturesReceived++;
        ProcessCmdTexture(reinterpret_cast<CmdTexture*>(gPendingRcv.pCommand));
      }
      // Version, Background, Clipboard, Input — already logged by
      // debug_log_cmd_received.

      if (gPendingRcv.bAutoFree) netImguiDeleteSafe(gPendingRcv.pCommand);
      gPendingRcv = PendingCom();
    }

    debug_log_frame_summary(sMainFrameCount, cmdsThisFrame);
  }

  // =========================================================================
  // Phase 2 & 3: Event loop and ImGui NewFrame via window abstraction.
  // =========================================================================
  static bool sLeftButtonDown = false;
  static bool sRightButtonDown = false;
  static bool sDragStartedOnRemoteUI = false;

  mujoco::platform::Window::Status status = g_app.window->NewFrame();
  if (status == mujoco::platform::Window::kQuitting) {
#if defined(__EMSCRIPTEN__)
    // NewFrame() started an ImGui frame — end it before bailing out.
    ImGui::EndFrame();
    emscripten_cancel_main_loop();
#endif
    return;
  }

  ImGuiIO& io = ImGui::GetIO();

  const bool hoverRemote = IsMouseOverRemoteUI();

  if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
    sLeftButtonDown = true;
    sDragStartedOnRemoteUI = hoverRemote;
  } else if (ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
    sLeftButtonDown = false;
  }

  if (ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
    sRightButtonDown = true;
    sDragStartedOnRemoteUI = hoverRemote;
  } else if (ImGui::IsMouseReleased(ImGuiMouseButton_Right)) {
    sRightButtonDown = false;
  }

  if (g_app.model_holder && g_app.model_holder->ok()) {
    bool suppress_camera =
        io.WantCaptureMouse || sDragStartedOnRemoteUI ||
        (hoverRemote && !sLeftButtonDown && !sRightButtonDown);

    if (!suppress_camera) {
      int width = g_app.window->GetWidth();
      int height = g_app.window->GetHeight();

      if (sLeftButtonDown && (io.MouseDelta.x != 0 || io.MouseDelta.y != 0)) {
        mujoco::platform::MoveCamera(
            g_app.model_holder->model(), g_app.model_holder->data(),
            &g_app.camera, mujoco::platform::CameraMotion::ORBIT,
            static_cast<mjtNum>(io.MouseDelta.x) / width,
            static_cast<mjtNum>(io.MouseDelta.y) / height);
      } else if (sRightButtonDown &&
                 (io.MouseDelta.x != 0 || io.MouseDelta.y != 0)) {
        mujoco::platform::MoveCamera(
            g_app.model_holder->model(), g_app.model_holder->data(),
            &g_app.camera, mujoco::platform::CameraMotion::PLANAR_MOVE_H,
            static_cast<mjtNum>(io.MouseDelta.x) / width,
            static_cast<mjtNum>(io.MouseDelta.y) / height);
      }

      if (io.MouseWheel != 0) {
        mujoco::platform::MoveCamera(g_app.model_holder->model(),
                                     g_app.model_holder->data(), &g_app.camera,
                                     mujoco::platform::CameraMotion::ZOOM, 0,
                                     -static_cast<mjtNum>(io.MouseWheel));
      }
    }
  }

  CaptureAndSendInput();

  UpdateAndShowTelemetryGUI();

  // Finalize the local ImGui draw data. ImguiBridge::Update() (called inside
  // Filament's Render) will call ImGui::Render() again, but that is a no-op
  // once the frame has already been rendered.
  ImGui::Render();

  // Inject remote draw lists into the local ImDrawData so that Filament's
  // ImguiBridge renders them alongside the local UI. Remote lists are inserted
  // first (background) and local lists are re-added after (foreground), so the
  // local UI always renders on top of remote content.
  // ImGui::GetDrawData() is only valid after ImGui::Render() and until the next
  // call to ImGui::NewFrame().
  if (gRemoteDrawData && gRemoteDrawData->Valid) {
    ImDrawData* localDrawData = ImGui::GetDrawData();
    // One-shot coordinate-space dump: every scale/clip bug so far has been a
    // units mismatch between these quantities.
    static bool sDumpedCoords = false;
    if (!sDumpedCoords && localDrawData) {
      sDumpedCoords = true;
      const ImGuiIO& dio = ImGui::GetIO();
      LOG(Info,
          "[coords] local io.DisplaySize=%.1fx%.1f FramebufferScale=%.2fx%.2f"
          " window=%dx%d canvasScale=%.2f",
          dio.DisplaySize.x, dio.DisplaySize.y, dio.DisplayFramebufferScale.x,
          dio.DisplayFramebufferScale.y, g_app.window->GetWidth(),
          g_app.window->GetHeight(), g_app.window->GetScale());
      LOG(Info,
          "[coords] localDrawData DisplaySize=%.1fx%.1f Pos=%.1f,%.1f"
          " FbScale=%.2f | remote DisplaySize=%.1fx%.1f Pos=%.1f,%.1f"
          " FbScale=%.2f",
          localDrawData->DisplaySize.x, localDrawData->DisplaySize.y,
          localDrawData->DisplayPos.x, localDrawData->DisplayPos.y,
          localDrawData->FramebufferScale.x, gRemoteDrawData->DisplaySize.x,
          gRemoteDrawData->DisplaySize.y, gRemoteDrawData->DisplayPos.x,
          gRemoteDrawData->DisplayPos.y, gRemoteDrawData->FramebufferScale.x);
      auto dump_list = [](const char* tag, const ImDrawList* dl) {
        if (!dl || dl->VtxBuffer.Size == 0) return;
        float vx0 = 1e9f, vy0 = 1e9f, vx1 = -1e9f, vy1 = -1e9f;
        for (int v = 0; v < dl->VtxBuffer.Size; ++v) {
          const ImVec2& p = dl->VtxBuffer[v].pos;
          vx0 = std::min(vx0, p.x); vy0 = std::min(vy0, p.y);
          vx1 = std::max(vx1, p.x); vy1 = std::max(vy1, p.y);
        }
        float cx0 = 1e9f, cy0 = 1e9f, cx1 = -1e9f, cy1 = -1e9f;
        for (int c = 0; c < dl->CmdBuffer.Size; ++c) {
          const ImVec4& r = dl->CmdBuffer[c].ClipRect;
          cx0 = std::min(cx0, r.x); cy0 = std::min(cy0, r.y);
          cx1 = std::max(cx1, r.z); cy1 = std::max(cy1, r.w);
        }
        LOG(Info,
            "[coords] %s: %d cmds, vtx bounds (%.1f,%.1f)-(%.1f,%.1f),"
            " clip bounds (%.1f,%.1f)-(%.1f,%.1f)",
            tag, dl->CmdBuffer.Size, vx0, vy0, vx1, vy1, cx0, cy0, cx1, cy1);
      };
      dump_list("remote", gRemoteDrawData->CmdLists[0]);
      if (localDrawData->CmdListsCount > 0) {
        dump_list("local", localDrawData->CmdLists[0]);
      }
    }
    if (localDrawData) {
      // Save local draw lists.
      ImVector<ImDrawList*> localLists;
      localLists.reserve(localDrawData->CmdListsCount);
      for (int i = 0; i < localDrawData->CmdListsCount; ++i) {
        localLists.push_back(localDrawData->CmdLists[i]);
      }

      // Clear and rebuild: remote first, then local.
      localDrawData->CmdLists.resize(0);
      localDrawData->CmdListsCount = 0;
      localDrawData->TotalVtxCount = 0;
      localDrawData->TotalIdxCount = 0;

      // Remote draw lists (background).
      for (int i = 0; i < gRemoteDrawData->CmdListsCount; ++i) {
        localDrawData->AddDrawList(gRemoteDrawData->CmdLists[i]);
      }

      // Local draw lists (foreground).
      for (int i = 0; i < localLists.Size; ++i) {
        localDrawData->AddDrawList(localLists[i]);
      }
    }
  }

  // =========================================================================
  // Phase 4: Filament render — scene + all ImGui (local + remote).
  // Filament manages its own clear, framebuffer, and GL state.
  // =========================================================================
  if (g_app.renderer && g_app.model_holder && g_app.model_holder->ok()) {
    // Apply backend state if available.
    if (g_app.backend_state_dirty) {
      mjModel* model = g_app.model_holder->model();
      mjData* data = g_app.model_holder->data();
      mj_setState(model, data, g_app.backend_state.data(),
                  g_app.backend_state_sig);
      mj_forward(model, data);
      g_app.backend_state_dirty = false;
    }

    int width =
        static_cast<int>(g_app.window->GetWidth() * g_app.window->GetScale());
    int height =
        static_cast<int>(g_app.window->GetHeight() * g_app.window->GetScale());
    if (width > 0 && height > 0) {
      g_app.renderer->Render(g_app.model_holder->model(),
                             g_app.model_holder->data(), &g_app.perturb,
                             &g_app.camera, &g_app.vis_options, width, height);
    }
  }

  g_app.window->Present();
}

#if defined(__EMSCRIPTEN__)
void SetupScene(const mjModel* m) {
  g_app.renderer->Init(m);

  // Upload any textures (e.g. font atlas) that were buffered before the
  // Filament context was available.
  FlushPendingTextures();

  mjv_defaultPerturb(&g_app.perturb);
  mjv_defaultCamera(&g_app.camera);
  mjv_defaultOption(&g_app.vis_options);

  const int model_cam = m->vis.global.cameraid;
  if (model_cam >= 0 && model_cam < m->ncam) {
    g_app.camera_idx = mujoco::platform::SetCamera(m, &g_app.camera, model_cam);
  } else {
    mjv_defaultFreeCamera(m, &g_app.camera);
  }
}

void OnFetchSuccess(emscripten_fetch_t* fetch) {
  LOG(Info, "Fetched model.mjb, size: %llu", fetch->numBytes);
  g_app.model_holder = mujoco::platform::ModelHolder::FromBuffer(
      std::span<const std::byte>(
          reinterpret_cast<const std::byte*>(fetch->data),
          static_cast<size_t>(fetch->numBytes)),
      "application/mjb", "model.mjb");
  if (g_app.model_holder && g_app.model_holder->ok()) {
    LOG(Info, "Model loaded successfully!");
    mj_forward(g_app.model_holder->model(), g_app.model_holder->data());
    if (g_app.window) {
      SetupScene(g_app.model_holder->model());
    }
    // Connect the state WebSocket to receive simulation state from Python.
    ConnectStateWebSocket();
  } else {
    LOG(Error, "Failed to load model: %s",
        g_app.model_holder ? g_app.model_holder->error().data()
                           : "Unknown error");
  }
  emscripten_fetch_close(fetch);
}

void OnFetchError(emscripten_fetch_t* fetch) {
  LOG(Error, "Failed to fetch model.mjb, status: %d", fetch->status);
  emscripten_fetch_close(fetch);
}

#endif

// Loads an asset from the Emscripten virtual filesystem. The Filament assets
// (materials, IBL) are bundled into web_client.data at link time via
// --preload-file and mounted at /assets.
static std::vector<std::byte> LoadAsset(std::string_view path) {
  std::string_view subpath = path.substr(path.find(':') + 1);
  std::string file_path = std::string("/assets/") + std::string(subpath);

  std::ifstream file(file_path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    return {};
  }
  auto file_size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<std::byte> buffer(file_size);
  if (!file.read(reinterpret_cast<char*>(buffer.data()), file_size)) {
    return {};
  }
  return buffer;
}

// Holds loaded resource data for the MuJoCo resource provider.
struct ResourceData {
  std::vector<std::byte> bytes;
};

// Registers a resource provider so that "filament:" asset requests from the
// renderer resolve to the preloaded /assets files.
static void RegisterAssetProviders() {
  mjpResourceProvider resource_provider;
  mjp_defaultResourceProvider(&resource_provider);

  resource_provider.open = [](mjResource* resource) {
    auto* data = new ResourceData();
    data->bytes = LoadAsset(resource->name);
    if (data->bytes.empty()) {
      delete data;
      return 0;
    }
    resource->data = data;
    return static_cast<int>(data->bytes.size());
  };
  resource_provider.read = [](mjResource* resource, const void** buffer) {
    auto* data = static_cast<ResourceData*>(resource->data);
    *buffer = data->bytes.data();
    return static_cast<int>(data->bytes.size());
  };
  resource_provider.close = [](mjResource* resource) {
    delete static_cast<ResourceData*>(resource->data);
    resource->data = nullptr;
  };
  resource_provider.prefix = "filament";
  mjp_registerResourceProvider(&resource_provider);
}

int main(int argc, char** argv) {
#if defined(__EMSCRIPTEN__)
  // Diagnostic: ?nocompress=1 disables NetImgui delta compression so the
  // client streams full draw frames every frame.
  gUseCompression = !EM_ASM_INT({
    return new URLSearchParams(window.location.search).has("nocompress") ? 1
                                                                         : 0;
  });
#endif
  RegisterAssetProviders();

  mujoco::platform::Window::Config config;
  config.gfx_mode = mujoco::platform::GraphicsMode::FilamentWebGl;
  config.load_fonts = false;

  g_app.window = std::make_unique<mujoco::platform::Window>("MuJoCo Live", 1400,
                                                            720, config);
  ImPlot::CreateContext();  // Needed if the server app uses ImPlot.

  g_app.renderer = new mujoco::platform::Renderer(
      g_app.window->GetNativeWindowHandle(), config.gfx_mode);

  Network::Startup();
  SDL_StartTextInput();

#if defined(__EMSCRIPTEN__)
  const char* host = "localhost";
#else
  const char* host = "127.0.0.1";
  if (argc > 1) host = argv[1];
#endif
  debug_log_connect(host, 8890);
  gClientSocket = Network::Connect(host, 8890);
  debug_log_connect_result(gClientSocket);

#if defined(__EMSCRIPTEN__)
  emscripten_fetch_attr_t attr;
  emscripten_fetch_attr_init(&attr);
  strcpy(attr.requestMethod, "GET");
  attr.attributes = EMSCRIPTEN_FETCH_LOAD_TO_MEMORY;
  attr.onsuccess = OnFetchSuccess;
  attr.onerror = OnFetchError;
  // Relative URL — the browser resolves this against the page origin,
  // so it works regardless of hostname, protocol, or port.
  emscripten_fetch(&attr, "/model.mjb");

  emscripten_set_main_loop(MainLoop, 0, 1);
#else
  while (true) {
    MainLoop();
    SDL_Delay(16);
  }
#endif

  // Cleanup (reached in non-WASM builds or if emscripten loop exits).
  if (gRemoteDrawData) {
    netImguiDelete(gRemoteDrawData);
    gRemoteDrawData = nullptr;
  }
  if (gClientSocket) {
    Network::Disconnect(gClientSocket);
    gClientSocket = nullptr;
  }
  Network::Shutdown();

  g_app.window.reset();

  return 0;
}
