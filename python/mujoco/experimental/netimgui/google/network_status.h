// Connection-state query for the WASM WebSocket network backend.
//
// Stock NetImgui has no such query and does not need one: its Connect()
// implementations block until the connection is established, so a non-null
// SocketInfo is always usable and later failures surface as DataSend /
// DataReceive errors inside the client thread. In the browser,
// emscripten_websocket_new() returns a socket handle immediately while the
// connection completes (or fails) asynchronously, and no client state
// machine is watching it. Callers poll this state to hold back traffic
// until the socket is actually open and to detect closure for reconnecting.

#ifndef THIRD_PARTY_NETIMGUI_GOOGLE_NETWORK_STATUS_H_
#define THIRD_PARTY_NETIMGUI_GOOGLE_NETWORK_STATUS_H_

namespace NetImgui {
namespace Internal {
namespace Network {

struct SocketInfo;

enum class ReadyState {
  kDisconnected,  // Null socket.
  kConnecting,
  kOpen,
  kClosing,
  kClosed,
  kError,
};

// Implemented in NetImgui_NetworkWASM.cpp (Emscripten builds only).
ReadyState GetReadyState(SocketInfo* client_socket);

// Human-readable state name, for status overlays and logs.
inline const char* ReadyStateName(ReadyState state) {
  switch (state) {
    case ReadyState::kDisconnected:
      return "Disconnected";
    case ReadyState::kConnecting:
      return "Connecting";
    case ReadyState::kOpen:
      return "Open";
    case ReadyState::kClosing:
      return "Closing";
    case ReadyState::kClosed:
      return "Closed";
    case ReadyState::kError:
      return "Error";
  }
  return "Unknown";
}

}  // namespace Network
}  // namespace Internal
}  // namespace NetImgui

#endif  // THIRD_PARTY_NETIMGUI_GOOGLE_NETWORK_STATUS_H_
