"""HTTP and proxy servers for serving MuJoCo Live to the browser.

This module provides the LiveServer class that:
1. Serves MuJoCo Live static files (index.html, WASM) over HTTP.
2. Serves the model MJB over HTTP (GET /model.mjb).
3. Runs a WebSocket-to-TCP proxy bridging the NetImgui protocol between
   the C++ LinkServer and the browser.
"""

import asyncio
import base64
import datetime
import functools
import hashlib
import http.server
import logging
import multiprocessing
import os
import signal
import socket
import socketserver
import struct
import sys
from typing import Optional


class _LiveServerFormatter(logging.Formatter):

  def format(self, record):
    level_char = record.levelname[0]
    dt = datetime.datetime.fromtimestamp(record.created)
    date_str = dt.strftime("%m%d")
    time_str = dt.strftime("%H:%M:%S.%f")
    filename = os.path.basename(record.pathname)
    line = record.lineno
    msg = record.getMessage()
    return f"{level_char}{date_str} {time_str} {filename}:{line}] {msg}"


logger = logging.getLogger("LiveServer")
logger.setLevel(logging.INFO)
if not logger.handlers:
  handler = logging.StreamHandler(sys.stdout)
  handler.setFormatter(_LiveServerFormatter())
  logger.addHandler(handler)
  logger.propagate = False


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


def _find_live_files_dir() -> Optional[str]:
  """Locate the MuJoCo Live static files (index.html, WASM) via runfiles."""
  # When run via `blaze run`, runfiles are available relative to the binary.
  # The live_wasm outputs are at the live_dir directory below
  runfiles_dir = os.environ.get("RUNFILES_DIR")
  if not runfiles_dir:
    # Try to find runfiles relative to the current binary.
    binary_path = os.path.abspath(__file__)
    # Walk up to find the .runfiles directory.
    parts = binary_path.split(os.sep)
    for i in range(len(parts) - 1, -1, -1):
      if parts[i].endswith(".runfiles"):
        runfiles_dir = os.sep.join(parts[: i + 1])
        break

  if not runfiles_dir:
    return None

  live_dir = os.path.join(
      runfiles_dir,
      "google3",
      "third_party",
      "mujoco",
      "src",
      "experimental",
      "py",
  )
  if os.path.isdir(live_dir):
    return live_dir

  return None


class _ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
  daemon_threads = True
  allow_reuse_address = True


class _LiveHTTPHandler(http.server.SimpleHTTPRequestHandler):
  """HTTP handler that serves MuJoCo Live files and the model MJB."""

  def __init__(self, *args, mjb_data=None, ws_port=8890, **kwargs):
    self._mjb_data = mjb_data
    self._ws_port = ws_port
    super().__init__(*args, **kwargs)

  def do_GET(self):  # pylint: disable=invalid-name
    """Handle GET requests."""
    if self.path == "/model.mjb" and self._mjb_data:
      self.send_response(200)
      self.send_header("Content-Type", "application/octet-stream")
      self.send_header("Content-Length", str(len(self._mjb_data)))
      self.send_header("Access-Control-Allow-Origin", "*")
      self.end_headers()
      self.wfile.write(self._mjb_data)
    else:
      super().do_GET()

  def end_headers(self):
    # Add CORS headers for WASM loading.
    self.send_header("Cross-Origin-Opener-Policy", "same-origin")
    self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
    super().end_headers()

  def log_message(self, format, *args):  # pylint: disable=redefined-builtin
    """Suppress default HTTP request logging."""
    pass


WS_MAGIC = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


async def _handle_ws_handshake(reader, writer):
  """Performs the WebSocket handshake with the browser."""
  header_data = b""
  while b"\r\n\r\n" not in header_data:
    chunk = await reader.read(1024)
    if not chunk:
      return False
    header_data += chunk

  headers = {}
  lines = header_data.decode("utf-8").split("\r\n")
  for line in lines[1:]:
    if ": " in line:
      k, v = line.split(": ", 1)
      headers[k.lower()] = v

  ws_key = headers.get("sec-websocket-key")
  if not ws_key:
    return False

  accept_key = base64.b64encode(
      hashlib.sha1((ws_key + WS_MAGIC).encode("utf-8")).digest()
  ).decode("utf-8")

  response = (
      "HTTP/1.1 101 Switching Protocols\r\n"
      "Upgrade: websocket\r\n"
      "Connection: Upgrade\r\n"
      f"Sec-WebSocket-Accept: {accept_key}\r\n"
      "\r\n"
  )
  writer.write(response.encode("utf-8"))
  await writer.drain()
  return True


async def _read_ws_binary_frame(reader):
  """Read a single WebSocket binary frame and return the unmasked payload."""
  header = await reader.readexactly(2)
  b1, b2 = header
  mask_bit = b2 >> 7
  payload_len = b2 & 0x7F

  if payload_len == 126:
    len_bytes = await reader.readexactly(2)
    payload_len = struct.unpack("!H", len_bytes)[0]
  elif payload_len == 127:
    len_bytes = await reader.readexactly(8)
    payload_len = struct.unpack("!Q", len_bytes)[0]

  mask_key = b""
  if mask_bit:
    mask_key = await reader.readexactly(4)

  payload = await reader.readexactly(payload_len)

  if mask_bit:
    unmasked = bytearray(payload_len)
    for i in range(payload_len):
      unmasked[i] = payload[i] ^ mask_key[i % 4]
    return bytes(unmasked)
  return bytes(payload)


def _write_ws_binary_frame(writer, data):
  """Write a single WebSocket binary frame (unmasked, server-to-client)."""
  header = bytearray([0x82])  # FIN + binary opcode
  if len(data) <= 125:
    header.append(len(data))
  elif len(data) <= 65535:
    header.append(126)
    header += struct.pack("!H", len(data))
  else:
    header.append(127)
    header += struct.pack("!Q", len(data))
  writer.write(header + data)


async def _ws_to_tcp(ws_reader, tcp_writer, ws_writer):
  """Read WebSocket frames from browser and forward raw payload to TCP."""
  msg_count = 0
  try:
    while True:
      header = await ws_reader.readexactly(2)
      if not header:
        break
      b1, b2 = header
      opcode = b1 & 0x0F
      mask_bit = b2 >> 7
      payload_len = b2 & 0x7F

      if payload_len == 126:
        len_bytes = await ws_reader.readexactly(2)
        payload_len = struct.unpack("!H", len_bytes)[0]
      elif payload_len == 127:
        len_bytes = await ws_reader.readexactly(8)
        payload_len = struct.unpack("!Q", len_bytes)[0]

      mask_key = b""
      if mask_bit:
        mask_key = await ws_reader.readexactly(4)

      payload = await ws_reader.readexactly(payload_len)

      if mask_bit:
        unmasked = bytearray(payload_len)
        for i in range(payload_len):
          unmasked[i] = payload[i] ^ mask_key[i % 4]
        payload = unmasked

      msg_count += 1
      if msg_count <= 5 or msg_count % 100 == 0:
        logger.info(
            f"[LinkProxy] ws_to_tcp: opcode={opcode}, len={len(payload)} (msg"
            f" #{msg_count})"
        )

      if opcode == 8:  # Close
        break
      elif opcode == 9:  # Ping
        pong_frame = bytearray([0x8A])
        if payload_len <= 125:
          pong_frame.append(payload_len)
        elif payload_len <= 65535:
          pong_frame.append(126)
          pong_frame += struct.pack("!H", payload_len)
        else:
          pong_frame.append(127)
          pong_frame += struct.pack("!Q", payload_len)
        ws_writer.write(pong_frame + payload)
        await ws_writer.drain()
      elif opcode == 0 or opcode == 2:  # Continuation or Binary
        tcp_writer.write(payload)
        await tcp_writer.drain()
      elif opcode == 10:  # Pong
        pass

  except asyncio.IncompleteReadError:
    logger.info("[LinkProxy] ws_to_tcp: browser disconnected (incomplete read)")
  except ConnectionResetError:
    logger.info("[LinkProxy] ws_to_tcp: browser connection reset")
  except Exception as e:  # pylint: disable=broad-exception-caught
    logger.error(f"[LinkProxy] ws_to_tcp error: {e}")


async def _tcp_to_ws(tcp_reader, ws_writer, stop_event):
  """Read raw bytes from TCP server and wrap in WebSocket frames to browser."""
  msg_count = 0
  try:
    while not stop_event.is_set():
      # Use a short timeout so we react quickly to stop_event while also
      # forwarding TCP data with minimal latency. The previous 100ms timeout
      # was adding up to 100ms per TCP segment, causing 0.5-1s UI lag.
      try:
        data = await asyncio.wait_for(tcp_reader.read(65536), timeout=0.005)
      except asyncio.TimeoutError:
        continue
      if not data:
        logger.info("[LinkProxy] tcp_to_ws: TCP connection closed by server")
        break

      len_data = len(data)
      header = bytearray([0x82])

      if len_data <= 125:
        header.append(len_data)
      elif len_data <= 65535:
        header.append(126)
        header += struct.pack("!H", len_data)
      else:
        header.append(127)
        header += struct.pack("!Q", len_data)

      msg_count += 1
      if msg_count <= 5 or msg_count % 100 == 0:
        logger.info(
            f"[LinkProxy] tcp_to_ws: len={len(data)} (msg #{msg_count})"
        )
      ws_writer.write(header + data)
      await ws_writer.drain()

  except ConnectionResetError:
    logger.info("[LinkProxy] tcp_to_ws: WS connection reset")
  except Exception as e:  # pylint: disable=broad-exception-caught
    logger.error(f"[LinkProxy] tcp_to_ws error: {e}")

  # Send a WebSocket close frame (opcode 0x08) so the browser fires onclose
  # immediately rather than waiting for the TCP FIN to propagate.
  try:
    # Close frame: opcode=0x88, payload=2 bytes status code 1000 (normal)
    close_frame = bytearray([0x88, 0x02, 0x03, 0xE8])  # 1000 = normal closure
    ws_writer.write(close_frame)
    await ws_writer.drain()
    logger.info("[LinkProxy] tcp_to_ws: Sent WebSocket close frame to browser")
  except Exception:  # pylint: disable=broad-exception-caught
    pass


def _run_proxy(host, tcp_port, ws_port):
  """Runs the proxy event loop.

  Architecture: single-viewer proxy. Only one browser can be connected at a
  time. When the browser disconnects (or a new browser connects), the proxy
  tears down both the WebSocket and TCP sides, forcing the C++ app to
  reconnect and re-send the full handshake, textures, and draw commands.

  When a second browser connects while one is already active, the first
  browser is kicked (its WebSocket is closed).
  """
  # CmdVersion is always 120 bytes. The header is 8 bytes (size + type).
  CMD_VERSION_SIZE = 120

  async def main_loop():
    ws_connections = asyncio.Queue()

    tcp_reader = None
    tcp_writer = None

    # Track the active WS writer so new connections can kick the old one.
    active_ws_writer = None
    # Event signalled when a new browser arrives while a bridge is active.
    kick_event = asyncio.Event()

    async def handle_tcp_client(reader, writer):
      nonlocal tcp_reader, tcp_writer
      if tcp_writer is not None:
        logger.info("[LinkProxy] Replacing previous TCP connection")
        tcp_writer.close()
        try:
          await tcp_writer.wait_closed()
        except Exception:
          pass
      tcp_reader = reader
      tcp_writer = writer
      # Disable Nagle's algorithm so small packets (CmdInput ~80 bytes) are
      # sent immediately instead of being buffered for up to 200ms.
      sock = writer.get_extra_info("socket")
      if sock:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
      logger.info("[LinkProxy] TCP connection accepted")

    async def handle_ws_connection(ws_reader, ws_writer):
      nonlocal active_ws_writer
      logger.info("[LinkProxy] WebSocket connection attempt")
      if await _handle_ws_handshake(ws_reader, ws_writer):
        logger.info("[LinkProxy] WebSocket connection accepted")
        # Disable Nagle on the WS socket too.
        sock = ws_writer.get_extra_info("socket")
        if sock:
          sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        # Kick the previous browser if one is active.
        if active_ws_writer is not None:
          logger.info("[LinkProxy] Kicking previous browser connection")
          active_ws_writer.close()
          kick_event.set()

        await ws_connections.put((ws_reader, ws_writer))
      else:
        logger.info("[LinkProxy] WebSocket handshake failed")
        ws_writer.close()

    tcp_server = await asyncio.start_server(
        handle_tcp_client,
        host,
        tcp_port,
    )
    ws_server = await asyncio.start_server(
        handle_ws_connection,
        host,
        ws_port,
    )

    logger.info(
        f"[LinkProxy] Proxy listening on TCP:{tcp_port} and WS:{ws_port}"
    )

    async with tcp_server, ws_server:
      while True:
        # Wait for TCP connection from C++ app (if not already connected)
        while tcp_reader is None:
          logger.info("[LinkProxy] Waiting for TCP connection...")
          await asyncio.sleep(0.1)

        # Wait for browser WebSocket connection
        ws_reader, ws_writer = await ws_connections.get()

        # Drain any stale queued WS connections (only use the latest)
        while not ws_connections.empty():
          old_r, old_w = ws_connections.get_nowait()
          logger.info("[LinkProxy] Discarding stale WS connection")
          old_w.close()

        active_ws_writer = ws_writer
        kick_event.clear()
        logger.info("[LinkProxy] Both connected. Starting bridge.")

        # --- Handshake phase ---
        # Read the browser's CmdVersion from the first WS binary frame.
        try:
          browser_version = await _read_ws_binary_frame(ws_reader)
          logger.info(
              f"[LinkProxy] Received browser CmdVersion: {len(browser_version)}"
              " bytes"
          )
        except Exception as e:
          logger.error(f"[LinkProxy] Failed to read browser CmdVersion: {e}")
          ws_writer.close()
          active_ws_writer = None
          continue

        # Always forward the full handshake to the C++ app.
        logger.info("[LinkProxy] Forwarding CmdVersion handshake")
        tcp_writer.write(browser_version)
        await tcp_writer.drain()

        # Read the server's CmdVersion reply from TCP.
        server_version = await tcp_reader.readexactly(CMD_VERSION_SIZE)
        logger.info(
            f"[LinkProxy] Received server CmdVersion: {len(server_version)}"
            " bytes"
        )

        # Forward it to the browser as a WS frame.
        _write_ws_binary_frame(ws_writer, server_version)
        await ws_writer.drain()

        logger.info("[LinkProxy] Handshake complete. Starting data bridge.")

        # --- Data bridge phase ---
        stop_event = asyncio.Event()

        ws_task = asyncio.create_task(
            _ws_to_tcp(ws_reader, tcp_writer, ws_writer)
        )
        tcp_task = asyncio.create_task(
            _tcp_to_ws(tcp_reader, ws_writer, stop_event)
        )

        try:
          done, pending = await asyncio.wait(
              [ws_task, tcp_task],
              return_when=asyncio.FIRST_COMPLETED,
          )
          stop_event.set()
          for task in pending:
            task.cancel()
            try:
              await task
            except asyncio.CancelledError:
              pass
        except Exception as e:
          logger.error(f"[LinkProxy] Bridge error: {e}")

        # Close the WS side.
        logger.info("[LinkProxy] Bridge closed. Closing WS connection.")
        ws_writer.close()
        try:
          await ws_writer.wait_closed()
        except Exception:
          pass
        active_ws_writer = None

        # Always tear down TCP so the C++ app reconnects and re-sends
        # all textures and draw commands to the next browser.
        logger.info("[LinkProxy] Tearing down TCP to force full reconnect.")
        if tcp_writer is not None:
          tcp_writer.close()
          try:
            await tcp_writer.wait_closed()
          except Exception:
            pass
        tcp_reader = None
        tcp_writer = None

  _run_cancellable(main_loop)


class LiveServer:
  """HTTP server and NetImgui proxy for MuJoCo Live.

  This class manages:
  1. An HTTP server that serves MuJoCo Live static files and the model MJB.
  2. A WebSocket-to-TCP proxy that bridges the NetImgui protocol between
     the C++ LinkServer and the browser.

  Externally: run example_link.py, visit http://localhost:8080.
  Internally (Google): Requires SSH port forwarding for remote access.
  """

  def __init__(
      self,
      host: str = "0.0.0.0",
      http_port: int = 8080,
      tcp_port: int = 8888,
      ws_port: int = 8890,
      static_files_dir: Optional[str] = None,
      mjb_data: Optional[bytes] = None,
  ):
    self.host = host
    self.http_port = http_port
    self.tcp_port = tcp_port
    self.ws_port = ws_port
    self.static_files_dir = static_files_dir or _find_live_files_dir()
    self.mjb_data = mjb_data
    self._running = False
    self._http_thread = None
    self._proxy_thread = None

  def start(self) -> None:
    """Start the HTTP server and proxy in background processes."""
    self._running = True
    self._start_http_server()
    self._start_proxy_server()

  def stop(self) -> None:
    """Stop the server."""
    self._running = False
    if self._http_thread:
      self._http_thread.terminate()
      self._http_thread.join(timeout=5)
    if self._proxy_thread:
      self._proxy_thread.terminate()
      self._proxy_thread.join(timeout=5)

  def _start_http_server(self) -> None:
    """Start the HTTP server that serves MuJoCo Live files."""
    if self.static_files_dir:
      logger.info(f"[Link] Serving MuJoCo Live from: {self.static_files_dir}")
    else:
      logger.warning("[Link] WARNING: Could not find MuJoCo Live static files.")
      logger.warning("[Link]   The Live WASM files were not found in runfiles.")
      return

    def run_server():
      signal.signal(signal.SIGINT, signal.SIG_IGN)
      signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(0))
      try:
        handler = functools.partial(
            _LiveHTTPHandler,
            directory=self.static_files_dir,
            mjb_data=self.mjb_data,
            ws_port=self.ws_port,
        )
        server = _ThreadingHTTPServer((self.host, self.http_port), handler)
        logger.info(
            f"[Link] HTTP server on http://{self.host}:{self.http_port}"
        )
        server.serve_forever()
      except (KeyboardInterrupt, SystemExit):
        pass

    self._http_thread = multiprocessing.Process(target=run_server, daemon=True)
    self._http_thread.start()

  def _start_proxy_server(self) -> None:
    """Start the WebSocket-to-TCP proxy."""
    self._proxy_thread = multiprocessing.Process(
        target=_run_proxy,
        args=(self.host, self.tcp_port, self.ws_port),
        daemon=True,
    )
    self._proxy_thread.start()
