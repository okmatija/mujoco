"""Single-port web server for the MuJoCo web viewer.

This server runs in a child process with one asyncio loop on a single public port:

  * Plain HTTP GET     serves static files (index.html, WASM, assets), /model.mjb.
  * WebSocket /ui      serves the bridge to the headless NetImgui client, which
                       connects over loopback TCP (see ui_server.cc).
  * WebSocket /state   serves the latest-wins state payload broadcast at ~60Hz
                       (payload format: see render_state.h).

Because everything is served through one port, a single firewall rule,
port-forward, or HTTPS tunnel exposes the whole viewer. The browser derives its
WebSocket URLs from the page origin, so no client configuration is needed.

Single-viewer semantics: one browser at a time owns the interactive session.
When a new browser connects, the previous one is closed with WebSocket code
4000 ("superseded"); the kicked page stops reconnecting and shows a
"taken over" notice (see web_client.cc).

Development: Set MUJOCO_WEB_VIEWER_DIST to point to a custom Emscripten build
directory to serve a locally-built web_client without reinstalling the package.
"""

import asyncio
import ctypes
import datetime
import logging
import multiprocessing
import os
import signal
import socket
import struct
import sys
from typing import Optional

from websockets.asyncio.server import serve
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Response


class _WebServerFormatter(logging.Formatter):

  def format(self, record):
    level_char = record.levelname[0]
    dt = datetime.datetime.fromtimestamp(record.created)
    date_str = dt.strftime("%m%d")
    time_str = dt.strftime("%H:%M:%S.%f")
    filename = os.path.basename(record.pathname)
    line = record.lineno
    msg = record.getMessage()
    return f"{level_char}{date_str} {time_str} {filename}:{line}] {msg}"


logger = logging.getLogger("WebServer")
# Quiet by default: the viewer prints its URL itself. Set MUJOCO_WEB_VERBOSE
# (e.g. via viewer.py --verbose) for connection/bridge chatter.
logger.setLevel(
    logging.INFO if os.environ.get("MUJOCO_WEB_VERBOSE") else logging.WARNING
)
if not logger.handlers:
  handler = logging.StreamHandler(sys.stdout)
  handler.setFormatter(_WebServerFormatter())
  logger.addHandler(handler)
  logger.propagate = False


# WebSocket close code telling a browser that another browser took over the
# single viewer slot. The client must stop reconnecting when it sees this
# (otherwise two tabs kick each other in an endless loop).
WS_CLOSE_SUPERSEDED = 4000

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
    host: Interface to bind.
    port: A specific port, or 0 to take the first free port starting at
      DEFAULT_HTTP_PORT (so several viewers can run side by side).

  Raises:
    RuntimeError: If the requested port (or every scanned port) is taken.
  """
  candidates = (
      [port] if port else range(DEFAULT_HTTP_PORT,
                                DEFAULT_HTTP_PORT + _PORT_SCAN_COUNT)
  )
  for candidate in candidates:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
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


def _terminate_process(proc, timeout: float = 2.0) -> None:
  """Terminates a server process, escalating to SIGKILL if it hangs.

  A wedged child that outlives stop() keeps its sockets alive, which
  prevents the C++ NetImgui client from ever noticing the disconnect and
  reconnecting to the replacement server.
  """
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


def _run_cancellable(main_loop_func):
  """Runs an asyncio loop with SIGINT/SIGTERM structured task cancellation."""

  async def _wrapped():
    loop = asyncio.get_running_loop()
    main_task = asyncio.current_task()

    def _cancel():
      if main_task:
        main_task.cancel()

    loop.add_signal_handler(signal.SIGINT, _cancel)
    loop.add_signal_handler(signal.SIGTERM, _cancel)
    try:
      await main_loop_func()
    except asyncio.CancelledError:
      pass

  try:
    asyncio.run(_wrapped())
  except Exception as e:  # pylint: disable=broad-exception-caught
    logger.error(f"[{main_loop_func.__name__}] Unexpected error: {e}")


def _run_server(
    http_sock,
    tcp_sock,
    lifeline_r,
    lifeline_w,
    static_dir,
    mjb_data,
    shm_array,
    shm_capacity,
    generation,
):
  """The server process: HTTP + /ui + /state on one port, one event loop."""

  # Close the inherited write end of the lifeline pipe: the read end must see
  # EOF the moment the parent (the only writer) goes away — for ANY reason,
  # including SIGKILL, which no signal handler or getppid poll can cover.
  os.close(lifeline_w)

  static_root = os.path.realpath(static_dir) if static_dir else None

  def _http_headers(content_type, content_length, cacheable):
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

  def _serve_http(path):
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

  async def main_loop():
    # The NetImgui client connection (from the headless ui_server, loopback
    # TCP). The client retries every second, so after a teardown a fresh
    # connection shows up quickly.
    tcp_reader = None
    tcp_writer = None
    tcp_connected = asyncio.Event()

    # Active browser connections (single-viewer slot).
    active_ui_ws = None
    active_state_ws = None

    async def handle_tcp_client(reader, writer):
      nonlocal tcp_reader, tcp_writer
      if tcp_writer is not None:
        logger.info("[UiBridge] Replacing previous NetImgui TCP connection")
        tcp_writer.close()
      tcp_reader = reader
      tcp_writer = writer
      # Disable Nagle's algorithm so small packets (CmdInput ~80 bytes) are
      # sent immediately instead of being buffered for up to 200ms.
      sock = writer.get_extra_info("socket")
      if sock:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
      tcp_connected.set()
      logger.info("[UiBridge] NetImgui TCP connection accepted")

    def supersede(ws):
      """Kicks a browser connection in the background (close code 4000)."""

      async def _kick():
        try:
          await ws.close(WS_CLOSE_SUPERSEDED, "superseded")
        except Exception:  # pylint: disable=broad-exception-caught
          pass

      asyncio.create_task(_kick())

    # --- /ui: bridge the browser to the NetImgui client ----------------------

    async def ui_handler(ws):
      nonlocal active_ui_ws, tcp_reader, tcp_writer
      if active_ui_ws is not None:
        logger.info("[UiBridge] New browser; superseding previous one")
        supersede(active_ui_ws)
      active_ui_ws = ws

      # Each browser needs a fresh NetImgui session (full handshake, textures
      # and draw state). Drop any connection used by a previous bridge and
      # wait for the client's automatic reconnect.
      if tcp_writer is not None:
        tcp_writer.close()
        tcp_reader = None
        tcp_writer = None
        tcp_connected.clear()
      logger.info("[UiBridge] Waiting for NetImgui TCP connection...")
      # Loop: a newer browser bridge can reset the connection (and clear the
      # event) between the event firing and this task waking up.
      while tcp_writer is None:
        await tcp_connected.wait()
      if active_ui_ws is not ws:
        # Superseded while waiting; the newer bridge owns the connection.
        return
      my_reader, my_writer = tcp_reader, tcp_writer

      try:
        # Handshake: browser CmdVersion -> client; client CmdVersion -> browser.
        browser_version = await ws.recv()
        my_writer.write(browser_version)
        await my_writer.drain()
        server_version = await my_reader.readexactly(_CMD_VERSION_SIZE)
        await ws.send(server_version)
        logger.info("[UiBridge] Handshake complete. Bridging.")

        async def ws_to_tcp():
          async for message in ws:
            if isinstance(message, bytes):
              my_writer.write(message)
              await my_writer.drain()

        async def tcp_to_ws():
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
        logger.info("[UiBridge] Bridge closed.")
        # Tear down this bridge's TCP connection so the NetImgui client
        # reconnects and re-sends everything to the next browser.
        my_writer.close()
        if tcp_writer is my_writer:
          tcp_reader = None
          tcp_writer = None
          tcp_connected.clear()
        if active_ui_ws is ws:
          active_ui_ws = None

    # --- /state: latest-wins payload broadcast --------------------------------

    async def state_handler(ws):
      nonlocal active_state_ws
      if active_state_ws is not None:
        logger.info("[StateWS] New browser; superseding previous one")
        supersede(active_state_ws)
      active_state_ws = ws
      logger.info("[StateWS] Browser connected")

      last_gen = 0
      try:
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
      except (ConnectionClosed, ConnectionError):
        logger.info("[StateWS] Browser disconnected")
      finally:
        if active_state_ws is ws:
          active_state_ws = None

    # --- Dispatch --------------------------------------------------------------

    def process_request(connection, request):
      del connection
      path = request.path.split("?")[0]
      if path in ("/ui", "/state"):
        return None  # Proceed with the WebSocket handshake.
      return _serve_http(path)

    async def ws_handler(ws):
      path = ws.request.path.split("?")[0]
      if path == "/ui":
        await ui_handler(ws)
      elif path == "/state":
        await state_handler(ws)
      else:
        await ws.close(1008, "unknown endpoint")

    # Block until the lifeline pipe hits EOF, which happens exactly when the
    # viewer process is gone (clean stop() close, crash, or SIGKILL). An
    # orphaned server would otherwise keep serving a stale model and fight
    # any replacement server for browsers.
    async def watch_lifeline():
      loop = asyncio.get_running_loop()
      await loop.run_in_executor(None, os.read, lifeline_r, 1)
      logger.info("[Http] Viewer process exited; shutting down.")

    tcp_server = await asyncio.start_server(handle_tcp_client, sock=tcp_sock)
    # NetImgui does its own delta compression and the state payload is small;
    # permessage-deflate would only add latency at 60Hz.
    ws_server = await serve(
        ws_handler,
        sock=http_sock,
        process_request=process_request,
        compression=None,
        close_timeout=1.0,
        max_size=2**24,
    )
    http_host, http_port = http_sock.getsockname()[:2]
    tcp_port = tcp_sock.getsockname()[1]
    logger.info(
        f"[Http] Serving on http://{http_host}:{http_port} "
        f"(/, /model.mjb, /ui, /state; NetImgui TCP on 127.0.0.1:{tcp_port})"
    )

    # Not `async with`: on Python >= 3.12 waiting for close would block on
    # open connections, so cancellation (SIGTERM from WebServer.stop) could
    # hang the process. Close the listeners explicitly instead; open sockets
    # die with the process.
    try:
      await watch_lifeline()
    finally:
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
  ):
    """Initializes the server around pre-bound listening sockets.

    The sockets are bound by the viewer (see bind_public_socket /
    bind_loopback_socket) and shared with the server child, so ports stay
    stable across restarts and bind failures can't occur here.
    """
    self.http_sock = http_sock
    self.tcp_sock = tcp_sock
    self.static_files_dir = static_files_dir or _find_static_files_dir()
    self.mjb_data = mjb_data
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
    logger.info(f"[Http] Serving web viewer from: {self.static_files_dir}")
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
