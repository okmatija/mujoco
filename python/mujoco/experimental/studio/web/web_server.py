"""Single-port web server for the MuJoCo web viewer.

This server runs in a child process with one asyncio loop on a single public port:

  * Plain HTTP GET     serves static files (index.html, WASM, assets), /model.mjb.
  * WebSocket /ui      serves the bridge to the headless NetImgui client, which
                       connects over loopback TCP (see ui_server.cc).
  * WebSocket /state   serves the latest-wins state payload broadcast at ~60Hz
                       (payload format: see render_state.h) as binary frames,
                       and session roster updates as text frames.

Because everything is served through one port, a single firewall rule,
port-forward, or HTTPS tunnel exposes the whole viewer. The browser derives its
WebSocket URLs from the page origin, so no client configuration is needed.

One browser at a time drives the interactive session (it owns the /ui
bridge); later browsers are rejected from /ui with WebSocket close code 4001
and spectate instead: they receive the state broadcast and render the scene,
and can take the driver slot once it frees up (see web_client.cc). /state
connections beyond the spectator limit are closed with code 4002.

Development: Set MUJOCO_WEB_VIEWER_DIST to point to a custom Emscripten build
directory to serve a locally-built web_client without reinstalling the package.
"""

import asyncio
import ctypes
import datetime
import enum
import logging
import multiprocessing
import multiprocessing.queues
import multiprocessing.sharedctypes
import os
import signal
import socket
import struct
import sys
import threading
from typing import Awaitable, Callable, Optional

from websockets.asyncio.server import serve
from websockets.asyncio.server import ServerConnection
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request
from websockets.http11 import Response


class _WebServerFormatter(logging.Formatter):

  def format(self, record: logging.LogRecord) -> str:
    level_char = record.levelname[0]
    dt = datetime.datetime.fromtimestamp(record.created)
    date_str = dt.strftime("%m%d")
    time_str = dt.strftime("%H:%M:%S.%f")
    filename = os.path.basename(record.pathname)
    line = record.lineno
    msg = record.getMessage()
    return f"{level_char}{date_str} {time_str} {filename}:{line}] {msg}"


logger = logging.getLogger("WebServer")


class _HandshakeNoiseFilter(logging.Filter):
  """Drops handshake-failure logs from connections that simply went away.

  Two routine cases produce them: browsers speculatively open spare
  connections and close them unused, and a page reload races the
  model-change server restart, tearing connections down mid-handshake. The
  websockets library logs each as an opening-handshake failure — an
  ERROR-level record with a full traceback. Raising the log level would
  hide real handshake errors too, hence a filter matching exactly these
  signatures. (Filters do not inherit down the logger tree, so it must be
  attached to "websockets.server" itself.)
  """

  def filter(self, record: logging.LogRecord) -> bool:
    if record.getMessage() != "opening handshake failed":
      return True
    exc = record.exc_info[1] if record.exc_info else None
    if exc is None:
      return True
    if isinstance(exc, (ConnectionClosed, EOFError, OSError)):
      return False  # The connection died; nothing wrong with the request.
    return "did not receive a valid HTTP request" not in str(exc)


def _configure_logging() -> None:
  """Configures this module's logger and reins in the websockets library's.

  Called once at import time; a no-op if already configured (re-import).
  """
  if logger.handlers:
    return
  # Info is a curated, always-on channel (browser connect/disconnect);
  # connection plumbing is logged at DEBUG.
  logger.setLevel(logging.INFO)
  handler = logging.StreamHandler(sys.stdout)
  handler.setFormatter(_WebServerFormatter())
  logger.addHandler(handler)
  logger.propagate = False
  # The websockets library logs its own lifecycle chatter at INFO ("server
  # listening", "connection rejected (200 OK)" for every HTTP request); keep
  # it to warnings and errors.
  logging.getLogger("websockets").setLevel(logging.WARNING)
  logging.getLogger("websockets.server").addFilter(_HandshakeNoiseFilter())


_configure_logging()


# Deliberate WebSocket close codes. Codes in the 4xxx range tell the
# browser not to reconnect (see state_link.cc / web_client.cc).
WS_CLOSE_DRIVER_TAKEN = 4001  # /ui: another browser holds the driver slot.
WS_CLOSE_SESSION_FULL = 4002  # /state: the spectator limit is reached.
WS_CLOSE_INACTIVE = 4003  # /state: hidden tab kicked to free a viewer slot.
WS_CLOSE_NOT_DRIVER = 4004  # /drop: only the driver may load models.


class SessionMessage(enum.StrEnum):
  """Text messages browsers send on /state (keep in sync: web_client.cc)."""

  REQUEST_CONTROL = "request_control"
  LEAVE_QUEUE = "leave_queue"
  FORCE_CONTROL = "force_control"
  HEARTBEAT = "heartbeat"


# Sent to the page whose control claim the driver slot is reserved for
# (keep in sync: web_client.cc).
GRANT_MESSAGE = "grant"

# Driver-only message carrying the new spectator limit, e.g.
# "max_spectators=4" (keep in sync: web_client.cc).
MAX_SPECTATORS_PREFIX = "max_spectators="

# Upper bound on the runtime-editable spectator limit.
MAX_SPECTATOR_HARD_CAP = 32


# A granted control claim must arrive within this window, else the grant
# moves on down the queue.
_GRANT_EXPIRY_SEC = 5.0
# A driver whose tab stops running (hidden or closed without a clean
# disconnect) is released once someone is waiting for control. Detected by
# silence: a live driver tab sends input packets every frame.
_DRIVER_SILENT_RELEASE_SEC = 90.0
# Spectators are kicked after this much heartbeat silence (their tab is
# hidden or gone), freeing a viewer slot. Live tabs heartbeat every ~30s.
_SPECTATOR_SILENT_KICK_SEC = 300.0

# How many browsers may watch in addition to the driver; the driver can
# change it at runtime (up to MAX_SPECTATOR_HARD_CAP). The limit is about
# session behaviour — keeping the crowd and the control-request queue at a
# manageable size — not about resources: a state payload is a few KB at
# 60Hz, serialized once and sent per viewer, so even a full session costs
# only a few Mbit/s of upload. There is deliberately no config or CLI
# option (spectating exposes nothing that driving does not).
DEFAULT_MAX_SPECTATORS = 8

# Default public port, and how many consecutive ports to try when it is
# taken (e.g. by another running viewer).
DEFAULT_HTTP_PORT = 8080
_PORT_SCAN_COUNT = 20


def bind_public_socket(host: str, port: int = 0) -> socket.socket:
  """Binds (but does not listen on) the public HTTP listening socket.

  The socket is bound in the viewer process and inherited by each server
  child, so the port stays stable across server restarts (model changes) and
  bind conflicts surface here, synchronously, instead of inside the child.

  Args:
    host: Interface to bind. "::" (the default) binds a dual-stack socket
      that accepts both IPv6 and IPv4 connections, falling back to IPv4-only
      when the machine has no IPv6 support.
    port: A specific port, or 0 to take the first free port starting at
      DEFAULT_HTTP_PORT (so several viewers can run side by side).

  Raises:
    RuntimeError: If the requested port (or every scanned port) is taken.
  """
  if host == "::":
    try:
      socket.socket(socket.AF_INET6, socket.SOCK_STREAM).close()
    except OSError:
      host = "0.0.0.0"
  family = socket.AF_INET6 if ":" in host else socket.AF_INET

  candidates = (
      [port] if port else range(DEFAULT_HTTP_PORT,
                                DEFAULT_HTTP_PORT + _PORT_SCAN_COUNT)
  )
  for candidate in candidates:
    sock = socket.socket(family, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if family == socket.AF_INET6:
      # Dual-stack: also accept IPv4 connections (as IPv4-mapped addresses).
      sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
    try:
      sock.bind((host, candidate))
      return sock
    except OSError:
      sock.close()
  if port:
    raise RuntimeError(
        f"Port {port} is already in use — is another web viewer running? "
        "Pass a different --port, or 0 to pick one automatically."
    )
  raise RuntimeError(
      f"Ports {DEFAULT_HTTP_PORT}-{DEFAULT_HTTP_PORT + _PORT_SCAN_COUNT - 1} "
      "are all in use — are that many web viewers running?"
  )


def bind_loopback_socket(port: int = 0) -> socket.socket:
  """Binds the loopback socket the headless NetImgui client connects to.

  Defaults to an OS-assigned ephemeral port: both endpoints live in this
  process tree, so no fixed number is needed and collisions between viewer
  instances are impossible.
  """
  sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  sock.bind(("127.0.0.1", port))
  return sock

def _session_id(ws: ServerConnection) -> str:
  """The page's session id (?sid=...), tying its /ui and /state together."""
  path = ws.request.path if ws.request else ""
  query = path.split("?", 1)[1] if "?" in path else ""
  for part in query.split("&"):
    if part.startswith("sid="):
      return part[4:]
  return f"anon-{id(ws)}"


# NetImgui's CmdVersion handshake packet is always 120 bytes.
_CMD_VERSION_SIZE = 120

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript",
    ".wasm": "application/wasm",
    ".data": "application/octet-stream",
    ".json": "application/json",
    ".png": "image/png",
    ".ico": "image/x-icon",
}


def _terminate_process(
    proc: multiprocessing.process.BaseProcess, timeout: float = 2.0
) -> None:
  """Terminates a server process, escalating to SIGKILL if it hangs.

  A wedged child that outlives stop() keeps its sockets alive, which
  prevents the C++ NetImgui client from ever noticing the disconnect and
  reconnecting to the replacement server.
  """
  # Grace period first: the child normally exits on its own (lifeline EOF),
  # and signalling a process mid-exit makes it print noise on stderr.
  proc.join(timeout=timeout)
  if proc.is_alive():
    proc.terminate()
    proc.join(timeout=timeout)
  if proc.is_alive():
    logger.warning(f"[Http] Process {proc.pid} ignored SIGTERM; killing.")
    proc.kill()
    proc.join(timeout=timeout)


def _find_static_files_dir() -> Optional[str]:
  """Locate the web viewer static files (index.html, WASM).

  The `dist` directory next to this file is populated by the Emscripten build
  of the `web_client` target (web_client.js/.wasm/.data, index.html and the
  Filament assets). This is the default for packaged installations.

  Development option: Set MUJOCO_WEB_VIEWER_DIST to serve from a different
  directory (e.g. a local Emscripten build tree). This allows rapid iteration
  on web_client changes without rebuilding and reinstalling the package.
  """
  env_dir = os.environ.get("MUJOCO_WEB_VIEWER_DIST")
  if env_dir:
    if os.path.isdir(env_dir):
      return env_dir
    logger.warning(
        f"[Http] MUJOCO_WEB_VIEWER_DIST is not a directory: {env_dir}"
    )

  dist_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dist")
  if os.path.isdir(dist_dir):
    return dist_dir

  return None


def _run_cancellable(main_loop_func: Callable[[], Awaitable[None]]) -> None:
  """Runs an asyncio loop with SIGINT/SIGTERM structured task cancellation."""

  async def _wrapped() -> None:
    loop = asyncio.get_running_loop()
    main_task = asyncio.current_task()

    def _cancel() -> None:
      if main_task:
        main_task.cancel()

    loop.add_signal_handler(signal.SIGINT, _cancel)
    loop.add_signal_handler(signal.SIGTERM, _cancel)
    try:
      await main_loop_func()
    except asyncio.CancelledError:
      pass
    finally:
      # Detach the handlers (and asyncio's signal wakeup pipe) before the
      # loop closes: a signal landing afterwards would try to write to the
      # closed pipe and print "Exception ignored ... BrokenPipeError".
      loop.remove_signal_handler(signal.SIGINT)
      loop.remove_signal_handler(signal.SIGTERM)

  try:
    asyncio.run(_wrapped())
  except Exception as e:  # pylint: disable=broad-exception-caught
    logger.error(f"[{main_loop_func.__name__}] Unexpected error: {e}")


def _run_server(
    http_sock: socket.socket,
    tcp_sock: socket.socket,
    lifeline_r: int,
    lifeline_w: int,
    static_dir: Optional[str],
    mjb_data: Optional[bytes],
    shm_array: Optional[ctypes.Array],
    shm_capacity: int,
    generation: multiprocessing.sharedctypes.Synchronized,
    drop_queue: Optional[multiprocessing.queues.Queue],
) -> None:
  """The server process: HTTP + /ui + /state on one port, one event loop."""

  # Close the inherited write end of the lifeline pipe: the read end must see
  # EOF the moment the parent (the only writer) goes away — for ANY reason,
  # including SIGKILL, which no signal handler or getppid poll can cover.
  os.close(lifeline_w)

  static_root = os.path.realpath(static_dir) if static_dir else None

  def _http_headers(
      content_type: str, content_length: int, cacheable: bool
  ) -> Headers:
    headers = Headers()
    headers["Content-Type"] = content_type
    headers["Content-Length"] = str(content_length)
    # The WASM is a pthread build: SharedArrayBuffer requires cross-origin
    # isolation on every response.
    headers["Cross-Origin-Opener-Policy"] = "same-origin"
    headers["Cross-Origin-Embedder-Policy"] = "require-corp"
    if not cacheable:
      headers["Access-Control-Allow-Origin"] = "*"
      headers["Cache-Control"] = "no-store"
    return headers

  def _serve_http(path: str) -> Response:
    """Builds the HTTP response for a non-WebSocket GET request."""
    if path == "/model.mjb":
      if not mjb_data:
        return Response(404, "Not Found", Headers(), b"no model\n")
      # The model changes on hot-swap; never serve a cached copy.
      headers = _http_headers(
          "application/octet-stream", len(mjb_data), cacheable=False
      )
      return Response(200, "OK", headers, mjb_data)

    if static_root is None:
      return Response(503, "Service Unavailable", Headers(), b"no dist dir\n")

    rel = path.lstrip("/") or "index.html"
    full = os.path.realpath(os.path.join(static_root, rel))
    if full != static_root and not full.startswith(static_root + os.sep):
      return Response(403, "Forbidden", Headers(), b"forbidden\n")
    if not os.path.isfile(full):
      return Response(404, "Not Found", Headers(), b"not found\n")

    with open(full, "rb") as f:
      body = f.read()
    content_type = _CONTENT_TYPES.get(
        os.path.splitext(full)[1], "application/octet-stream"
    )
    return Response(
        200, "OK", _http_headers(content_type, len(body), cacheable=True), body
    )

  async def main_loop() -> None:
    # The NetImgui client connection (from the headless ui_server, loopback
    # TCP). The client retries every second, so after a teardown a fresh
    # connection shows up quickly.
    tcp_reader = None
    tcp_writer = None
    tcp_connected = asyncio.Event()

    # The browser driving the UI (single slot) and all browsers receiving
    # the state broadcast (driver + spectators), keyed by session id.
    active_ui_ws = None
    driver_sid: Optional[str] = None
    state_clients: dict[str, ServerConnection] = {}
    # Control handoff: spectators queue for the driver slot; when it frees,
    # the head of the queue is granted a short exclusive claim window.
    control_queue: list[str] = []
    pending_grant_sid: Optional[str] = None
    max_spectators = DEFAULT_MAX_SPECTATORS
    # Liveness by absence of traffic: a hidden tab's rendering loop stops,
    # so it cannot report anything — silence is the signal.
    last_heartbeat: dict[str, float] = {}
    last_driver_input = [0.0]
    loop_time = asyncio.get_event_loop().time

    async def broadcast_roster() -> None:
      """Sends every browser the counts, its role and its queue position."""
      count = len(state_clients)
      for sid, client in list(state_clients.items()):
        role = "driver" if sid == driver_sid else "spectator"
        try:
          pos = control_queue.index(sid) + 1
        except ValueError:
          pos = 0
        try:
          await client.send(
              f"viewers={count};role={role};queue_pos={pos};"
              f"queue_len={len(control_queue)};max_spectators={max_spectators}"
          )
        except (ConnectionClosed, ConnectionError):
          pass

    async def grant_next() -> None:
      """Offers the free driver slot to the pending or next queued page."""
      nonlocal pending_grant_sid
      if active_ui_ws is not None:
        return
      if pending_grant_sid is None:
        while control_queue:
          candidate = control_queue.pop(0)
          if candidate in state_clients:
            pending_grant_sid = candidate
            break
      if pending_grant_sid is None:
        return
      client = state_clients.get(pending_grant_sid)
      if client is None:
        pending_grant_sid = None
        await grant_next()
        return
      try:
        await client.send(GRANT_MESSAGE)
      except (ConnectionClosed, ConnectionError):
        pending_grant_sid = None
        await grant_next()
        return
      logger.info(f"[Session] Control granted to {pending_grant_sid}")
      await broadcast_roster()

      async def expire(granted: str) -> None:
        nonlocal pending_grant_sid
        await asyncio.sleep(_GRANT_EXPIRY_SEC)
        if pending_grant_sid == granted and active_ui_ws is None:
          pending_grant_sid = None
          await grant_next()

      asyncio.create_task(expire(pending_grant_sid))

    async def handle_session_message(sid: str, text: str) -> None:
      """A control or heartbeat message from one browser (/state text)."""
      nonlocal pending_grant_sid, max_spectators
      last_heartbeat[sid] = loop_time()
      if text == SessionMessage.REQUEST_CONTROL:
        if sid != driver_sid and sid not in control_queue:
          control_queue.append(sid)
          await grant_next()
          await broadcast_roster()
      elif text == SessionMessage.LEAVE_QUEUE:
        if sid in control_queue:
          control_queue.remove(sid)
          await broadcast_roster()
      elif text == SessionMessage.FORCE_CONTROL:
        if sid != driver_sid:
          logger.info(f"[Session] {sid} forces control")
          if sid in control_queue:
            control_queue.remove(sid)
          pending_grant_sid = sid
          if active_ui_ws is not None:
            # The bridge teardown frees the slot and grant_next honors
            # the pending claim; the ousted page settles into spectating.
            await active_ui_ws.close(WS_CLOSE_DRIVER_TAKEN, "control taken")
          else:
            await grant_next()
      elif text == SessionMessage.HEARTBEAT:
        pass  # last_heartbeat is updated for every message.
      elif text.startswith(MAX_SPECTATORS_PREFIX):
        if sid != driver_sid:
          return  # Only the driver sets the limit.
        try:
          value = int(text[len(MAX_SPECTATORS_PREFIX):])
        except ValueError:
          return
        max_spectators = max(0, min(MAX_SPECTATOR_HARD_CAP, value))
        logger.info(f"[Session] Spectator limit set to {max_spectators}")
        # Enforce the new limit immediately, kicking the newest spectators
        # first. A kicked page shows the session-full notice and retries,
        # so it walks back in when a slot frees up.
        excess = len(state_clients) - (max_spectators + 1)
        for kick_sid in reversed(list(state_clients)):
          if excess <= 0:
            break
          if kick_sid == driver_sid:
            continue
          client = state_clients.get(kick_sid)
          if client is not None:
            logger.info(f"[Session] Kicking spectator over the new limit")
            try:
              await client.close(WS_CLOSE_SESSION_FULL, "session full")
            except (ConnectionClosed, ConnectionError):
              pass
          excess -= 1
        await broadcast_roster()

    async def enforce_activity() -> None:
      """Releases a silent driver when someone waits; kicks silent tabs."""
      while True:
        await asyncio.sleep(5.0)
        now = loop_time()
        if (
            active_ui_ws is not None
            and last_driver_input[0] > 0
            and now - last_driver_input[0] > _DRIVER_SILENT_RELEASE_SEC
            and (control_queue or pending_grant_sid)
        ):
          logger.info("[Session] Releasing control from an inactive driver")
          await active_ui_ws.close(WS_CLOSE_DRIVER_TAKEN, "inactive")
        for sid, client in list(state_clients.items()):
          if sid == driver_sid:
            continue
          seen = last_heartbeat.get(sid, now)
          if now - seen > _SPECTATOR_SILENT_KICK_SEC:
            logger.info(f"[Session] Kicking inactive spectator {sid}")
            try:
              await client.close(WS_CLOSE_INACTIVE, "inactive")
            except (ConnectionClosed, ConnectionError):
              pass

    async def handle_tcp_client(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
      nonlocal tcp_reader, tcp_writer
      if tcp_writer is not None:
        logger.debug("[UiBridge] Replacing previous NetImgui TCP connection")
        tcp_writer.close()
      tcp_reader = reader
      tcp_writer = writer
      # Disable Nagle's algorithm so small packets (CmdInput ~80 bytes) are
      # sent immediately instead of being buffered for up to 200ms.
      sock = writer.get_extra_info("socket")
      if sock:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
      tcp_connected.set()
      logger.debug("[UiBridge] NetImgui TCP connection accepted")

    # --- /ui: bridge the browser to the NetImgui client ----------------------

    async def ui_handler(ws: ServerConnection) -> None:
      nonlocal active_ui_ws, driver_sid, pending_grant_sid
      nonlocal tcp_reader, tcp_writer
      if active_ui_ws is not None:
        # One driver at a time. The rejected browser spectates and may retry
        # via its "Take control" button once the driver leaves.
        logger.debug("[UiBridge] Driver slot taken; rejecting new browser")
        await ws.close(WS_CLOSE_DRIVER_TAKEN, "driver slot taken")
        return
      sid = _session_id(ws)
      if pending_grant_sid is not None and sid != pending_grant_sid:
        # The slot is reserved for a granted page (queue fairness).
        await ws.close(WS_CLOSE_DRIVER_TAKEN, "driver slot reserved")
        return
      if sid == pending_grant_sid:
        pending_grant_sid = None
      if sid in control_queue:
        control_queue.remove(sid)
      active_ui_ws = ws
      driver_sid = sid
      last_driver_input[0] = loop_time()
      await broadcast_roster()

      # Each browser needs a fresh NetImgui session (full handshake, textures
      # and draw state). Drop any connection used by a previous bridge and
      # wait for the client's automatic reconnect.
      if tcp_writer is not None:
        tcp_writer.close()
        tcp_reader = None
        tcp_writer = None
        tcp_connected.clear()
      logger.debug("[UiBridge] Waiting for NetImgui TCP connection...")
      # Loop: the connection can be replaced (and the event cleared) between
      # the event firing and this task waking up.
      while tcp_writer is None:
        await tcp_connected.wait()
      my_reader, my_writer = tcp_reader, tcp_writer

      try:
        # Handshake: browser CmdVersion -> client; client CmdVersion -> browser.
        browser_version = await ws.recv()
        my_writer.write(browser_version)
        await my_writer.drain()
        server_version = await my_reader.readexactly(_CMD_VERSION_SIZE)
        await ws.send(server_version)
        logger.debug("[UiBridge] Handshake complete. Bridging.")

        async def ws_to_tcp() -> None:
          async for message in ws:
            if isinstance(message, bytes):
              last_driver_input[0] = loop_time()
              my_writer.write(message)
              await my_writer.drain()

        async def tcp_to_ws() -> None:
          while True:
            data = await my_reader.read(65536)
            if not data:
              break
            await ws.send(data)

        done, pending = await asyncio.wait(
            [
                asyncio.create_task(ws_to_tcp()),
                asyncio.create_task(tcp_to_ws()),
            ],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
          task.cancel()
        for task in done:
          task.exception()  # Retrieve to avoid "exception never retrieved".
      except (ConnectionClosed, ConnectionError, asyncio.IncompleteReadError):
        pass
      finally:
        logger.debug("[UiBridge] Bridge closed.")
        # Tear down this bridge's TCP connection so the NetImgui client
        # reconnects and re-sends everything to the next browser.
        my_writer.close()
        if tcp_writer is my_writer:
          tcp_reader = None
          tcp_writer = None
          tcp_connected.clear()
        if active_ui_ws is ws:
          active_ui_ws = None
          driver_sid = None
          await grant_next()
          await broadcast_roster()

    # --- /state: latest-wins payload broadcast --------------------------------

    async def state_handler(ws: ServerConnection) -> None:
      if len(state_clients) > max_spectators:  # driver + spectators
        logger.info("[StateWS] Session full; rejecting browser")
        await ws.close(WS_CLOSE_SESSION_FULL, "session full")
        return
      sid = _session_id(ws)
      state_clients[sid] = ws
      last_heartbeat[sid] = loop_time()
      logger.info(f"[StateWS] Browser connected ({len(state_clients)} total)")
      await broadcast_roster()

      async def send_payloads() -> None:
        last_gen = 0
        while True:
          await asyncio.sleep(1.0 / 60.0)
          cur_gen = generation.value
          if cur_gen == last_gen:
            continue
          last_gen = cur_gen
          (used,) = struct.unpack("<I", bytes(shm_array[:4]))
          if used == 0 or used > shm_capacity:
            continue
          await ws.send(bytes(shm_array[4 : 4 + used]))

      async def recv_messages() -> None:
        async for message in ws:
          if isinstance(message, str):
            await handle_session_message(sid, message)

      try:
        done, pending = await asyncio.wait(
            [
                asyncio.create_task(send_payloads()),
                asyncio.create_task(recv_messages()),
            ],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
          task.cancel()
        for task in done:
          task.exception()  # Retrieve to avoid "exception never retrieved".
      except (ConnectionClosed, ConnectionError):
        pass
      finally:
        if state_clients.get(sid) is ws:
          del state_clients[sid]
        if sid in control_queue:
          control_queue.remove(sid)
        last_heartbeat.pop(sid, None)
        logger.info(
            f"[StateWS] Browser disconnected ({len(state_clients)} total)")
        await broadcast_roster()

    # --- Dispatch --------------------------------------------------------------

    def process_request(
        connection: ServerConnection, request: Request
    ) -> Optional[Response]:
      del connection
      path = request.path.split("?")[0]
      if path in ("/ui", "/state", "/drop"):
        return None  # Proceed with the WebSocket handshake.
      return _serve_http(path)

    # --- /drop: receive a file dropped onto the browser page ---------------

    async def drop_handler(ws: ServerConnection) -> None:
      """Receives dropped files and forwards them to the viewer process.

      The browser opens this socket on demand and sends one binary frame per
      file ([u32 path length][relative path utf-8][file bytes], see
      index.html), then an empty frame as an end marker; this server closes
      once it has everything. The files cross to the viewer process via
      drop_queue as a dict of relative path -> bytes;
      WebViewer.get_drop_file() writes them to a temporary directory for the
      regular drop-loading flow.

      Loading a model changes the session for every connected page, so only
      the driver may do it; drops from other pages are rejected.
      """
      if driver_sid is None or _session_id(ws) != driver_sid:
        logger.info("[Drop] Ignored a model drop from a non-driver page")
        await ws.close(WS_CLOSE_NOT_DRIVER, "not the driver")
        return
      files: dict[str, bytes] = {}
      try:
        async for message in ws:
          if not isinstance(message, bytes):
            continue
          if not message:
            break  # End-of-drop marker.
          if len(message) < 4:
            continue
          (name_len,) = struct.unpack_from("<I", message)
          if 4 + name_len > len(message):
            continue
          name = message[4 : 4 + name_len].decode("utf-8", "replace")
          files[name] = message[4 + name_len :]
      except (ConnectionClosed, ConnectionError):
        pass
      if files and drop_queue is not None:
        total = sum(len(data) for data in files.values())
        logger.info(f"[Drop] Received {len(files)} file(s), {total} bytes")
        drop_queue.put(files)
      await ws.close()

    async def ws_handler(ws: ServerConnection) -> None:
      path = ws.request.path.split("?")[0]
      if path == "/ui":
        await ui_handler(ws)
      elif path == "/state":
        await state_handler(ws)
      elif path == "/drop":
        await drop_handler(ws)
      else:
        await ws.close(1008, "unknown endpoint")

    # Block until the lifeline pipe hits EOF, which happens exactly when the
    # viewer process is gone (clean stop() close, crash, or SIGKILL). An
    # orphaned server would otherwise keep serving a stale model and fight
    # any replacement server for browsers.
    async def watch_lifeline() -> None:
      # A dedicated daemon thread rather than run_in_executor: executor
      # threads are non-daemonic and are joined at interpreter exit, so a
      # blocked os.read would keep this process alive for as long as the
      # viewer holds the lifeline open — e.g. when the viewer is still
      # tearing down after Ctrl+C hit both processes.
      loop = asyncio.get_running_loop()
      eof = asyncio.Event()

      def wait_for_eof() -> None:
        try:
          os.read(lifeline_r, 1)
          loop.call_soon_threadsafe(eof.set)
        except (OSError, RuntimeError):
          pass  # fd or loop already closed — the process is exiting anyway.

      threading.Thread(target=wait_for_eof, daemon=True).start()
      await eof.wait()
      logger.debug("[Http] Viewer process exited; shutting down.")

    tcp_server = await asyncio.start_server(handle_tcp_client, sock=tcp_sock)
    # NetImgui does its own delta compression and the state payload is small;
    # permessage-deflate would only add latency at 60Hz.
    ws_server = await serve(
        ws_handler,
        sock=http_sock,
        process_request=process_request,
        compression=None,
        close_timeout=1.0,
        # Big enough for model files uploaded via /drop.
        max_size=2**26,
    )
    http_host, http_port = http_sock.getsockname()[:2]
    tcp_port = tcp_sock.getsockname()[1]
    logger.debug(
        f"[Http] Serving on http://{http_host}:{http_port} "
        f"(/, /model.mjb, /ui, /state; NetImgui TCP on 127.0.0.1:{tcp_port})"
    )

    # Not `async with`: on Python >= 3.12 waiting for close would block on
    # open connections, so cancellation (SIGTERM from WebServer.stop) could
    # hang the process. Close the listeners explicitly instead; open sockets
    # die with the process.
    enforcer = asyncio.create_task(enforce_activity())
    try:
      await watch_lifeline()
    finally:
      enforcer.cancel()
      ws_server.close()
      tcp_server.close()

  _run_cancellable(main_loop)


class WebServer:
  """The web viewer's single-port server.

  Serves the browser page, WASM, and model over HTTP; bridges the NetImgui UI
  stream at /ui; and broadcasts the state payload at /state on a single public
  port from a child process. State payloads are handed over through shared
  memory with latest-wins semantics: update_state() may be called at any rate
  from the viewer thread; browsers only ever see the newest payload.

  Run the simulation with the web viewer, then visit http://localhost:8080.
  """

  def __init__(
      self,
      http_sock: socket.socket,
      tcp_sock: socket.socket,
      static_files_dir: Optional[str] = None,
      mjb_data: Optional[bytes] = None,
      max_payload_size: int = 0,
      drop_queue: Optional[multiprocessing.queues.Queue] = None,
  ) -> None:
    """Initializes the server around pre-bound listening sockets.

    The sockets are bound by the viewer (see bind_public_socket /
    bind_loopback_socket) and shared with the server child, so ports stay
    stable across restarts and bind failures can't occur here.
    """
    self.http_sock = http_sock
    self.tcp_sock = tcp_sock
    self.static_files_dir = static_files_dir or _find_static_files_dir()
    self.mjb_data = mjb_data
    # Owned by the viewer (it outlives server restarts); dropped-file bytes
    # travel from the server child back to the viewer process through it.
    self.drop_queue = drop_queue
    self._process = None
    self._lifeline_w = None

    # Shared memory for zero-copy payload transfer to the server process:
    # [u32 length][payload bytes], so payloads can vary in size up to
    # max_payload_size (see UiServer.max_state_payload_size).
    self._shm_capacity = max_payload_size
    self._shm_array = (
        multiprocessing.RawArray(ctypes.c_char, 4 + self._shm_capacity)
        if self._shm_capacity > 0
        else None
    )
    self._generation = multiprocessing.Value("Q", 0)  # uint64 counter

  def update_state(self, payload: bytes) -> None:
    """Publishes the latest state payload. Called from the viewer thread."""
    if self._shm_array is None:
      return
    if len(payload) > self._shm_capacity:
      logger.error(
          f"[StateWS] Payload of {len(payload)} bytes exceeds capacity"
          f" {self._shm_capacity}; dropping."
      )
      return
    ctypes.memmove(
        self._shm_array,
        struct.pack("<I", len(payload)) + payload,
        4 + len(payload),
    )
    self._generation.value += 1

  def start(self) -> None:
    """Starts the server in a background process."""
    if not self.static_files_dir:
      logger.warning("[Http] WARNING: Could not find web viewer static files.")
      return
    logger.debug(f"[Http] Serving web viewer from: {self.static_files_dir}")
    # Lifeline pipe: the child holds the read end and exits on EOF, which
    # happens when this process closes the write end (stop()) or dies for
    # any reason at all — no orphaned servers.
    lifeline_r, self._lifeline_w = os.pipe()
    # Sockets and pipe fds must be inherited, so fork explicitly (the default
    # start method is fork on Linux today, but is changing upstream).
    self._process = multiprocessing.get_context("fork").Process(
        target=_run_server,
        args=(
            self.http_sock,
            self.tcp_sock,
            lifeline_r,
            self._lifeline_w,
            self.static_files_dir,
            self.mjb_data,
            self._shm_array,
            self._shm_capacity,
            self._generation,
            self.drop_queue,
        ),
        daemon=True,
    )
    self._process.start()
    os.close(lifeline_r)

  def stop(self) -> None:
    """Stops the server, escalating to SIGKILL if it hangs."""
    if self._lifeline_w is not None:
      os.close(self._lifeline_w)  # EOF on the child's lifeline: clean exit.
      self._lifeline_w = None
    if self._process:
      _terminate_process(self._process)
      self._process = None
