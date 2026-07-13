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
logger.setLevel(logging.INFO)
if not logger.handlers:
  handler = logging.StreamHandler(sys.stdout)
  handler.setFormatter(_WebServerFormatter())
  logger.addHandler(handler)
  logger.propagate = False


# WebSocket close code telling a browser that another browser took over the
# single viewer slot. The client must stop reconnecting when it sees this
# (otherwise two tabs kick each other in an endless loop).
WS_CLOSE_SUPERSEDED = 4000

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
    host,
    port,
    tcp_port,
    static_dir,
    mjb_data,
    shm_array,
    shm_capacity,
    generation,
):
  """The server process: HTTP + /ui + /state on one port, one event loop."""

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
      await tcp_connected.wait()
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

    # Exit if the viewer process died without cleaning up (e.g. SIGKILL);
    # an orphaned server would otherwise keep serving a stale model and
    # fight any replacement server for browsers.
    async def watch_parent():
      while True:
        await asyncio.sleep(2.0)
        if os.getppid() == 1:
          logger.warning("[Http] Viewer process died; shutting down.")
          os._exit(0)  # pylint: disable=protected-access

    tcp_server = await asyncio.start_server(
        handle_tcp_client, "127.0.0.1", tcp_port
    )
    # NetImgui does its own delta compression and the state payload is small;
    # permessage-deflate would only add latency at 60Hz.
    ws_server = await serve(
        ws_handler,
        host,
        port,
        process_request=process_request,
        compression=None,
        close_timeout=1.0,
        max_size=2**24,
    )
    logger.info(
        f"[Http] Serving on http://{host}:{port} "
        f"(/, /model.mjb, /ui, /state; NetImgui TCP on 127.0.0.1:{tcp_port})"
    )

    # Not `async with`: on Python >= 3.12 waiting for close would block on
    # open connections, so cancellation (SIGTERM from WebServer.stop) could
    # hang the process. Close the listeners explicitly instead; open sockets
    # die with the process.
    try:
      await watch_parent()
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
      host: str = "0.0.0.0",
      http_port: int = 8080,
      tcp_port: int = 8888,
      static_files_dir: Optional[str] = None,
      mjb_data: Optional[bytes] = None,
      max_payload_size: int = 0,
  ):
    self.host = host
    self.http_port = http_port
    self.tcp_port = tcp_port
    self.static_files_dir = static_files_dir or _find_static_files_dir()
    self.mjb_data = mjb_data
    self._process = None

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
    self._process = multiprocessing.Process(
        target=_run_server,
        args=(
            self.host,
            self.http_port,
            self.tcp_port,
            self.static_files_dir,
            self.mjb_data,
            self._shm_array,
            self._shm_capacity,
            self._generation,
        ),
        daemon=True,
    )
    self._process.start()

  def stop(self) -> None:
    """Stops the server, escalating to SIGKILL if it hangs."""
    if self._process:
      _terminate_process(self._process)
      self._process = None
