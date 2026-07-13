"""WebSocket-to-TCP proxy bridging the Dear ImGui UI to the browser.

This module provides the UIServerProxy class that runs a proxy event loop
in a multiprocessing child process, bridging the C++ UIServer (TCP) and the
browser WebClient (WebSocket).
"""

import asyncio
import base64
import datetime
import hashlib
import logging
import multiprocessing
import os
import signal
import socket
import struct
import sys
from typing import Optional


class _UIServerProxyFormatter(logging.Formatter):

  def format(self, record):
    level_char = record.levelname[0]
    dt = datetime.datetime.fromtimestamp(record.created)
    date_str = dt.strftime("%m%d")
    time_str = dt.strftime("%H:%M:%S.%f")
    filename = os.path.basename(record.pathname)
    line = record.lineno
    msg = record.getMessage()
    return f"{level_char}{date_str} {time_str} {filename}:{line}] {msg}"


logger = logging.getLogger("UIServerProxy")
logger.setLevel(logging.INFO)
if not logger.handlers:
  handler = logging.StreamHandler(sys.stdout)
  handler.setFormatter(_UIServerProxyFormatter())
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
            f"[UIServerProxy] ws_to_tcp: opcode={opcode}, len={len(payload)} (msg"
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
    logger.info("[UIServerProxy] ws_to_tcp: browser disconnected (incomplete read)")
  except ConnectionResetError:
    logger.info("[UIServerProxy] ws_to_tcp: browser connection reset")
  except Exception as e:  # pylint: disable=broad-exception-caught
    logger.error(f"[UIServerProxy] ws_to_tcp error: {e}")


async def _tcp_to_ws(tcp_reader, ws_writer, stop_event):
  """Read raw bytes from TCP server and wrap in WebSocket frames to browser."""
  msg_count = 0
  try:
    while not stop_event.is_set():
      try:
        data = await asyncio.wait_for(tcp_reader.read(65536), timeout=0.005)
      except asyncio.TimeoutError:
        continue
      if not data:
        logger.info("[UIServerProxy] tcp_to_ws: TCP connection closed by server")
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
            f"[UIServerProxy] tcp_to_ws: len={len(data)} (msg #{msg_count})"
        )
      ws_writer.write(header + data)
      await ws_writer.drain()

  except ConnectionResetError:
    logger.info("[UIServerProxy] tcp_to_ws: WS connection reset")
  except Exception as e:  # pylint: disable=broad-exception-caught
    logger.error(f"[UIServerProxy] tcp_to_ws error: {e}")

  try:
    close_frame = bytearray([0x88, 0x02, 0x03, 0xE8])  # 1000 = normal closure
    ws_writer.write(close_frame)
    await ws_writer.drain()
    logger.info("[UIServerProxy] tcp_to_ws: Sent WebSocket close frame to browser")
  except Exception:  # pylint: disable=broad-exception-caught
    pass


def _run_proxy(host, tcp_port, ws_port):
  """Runs the proxy event loop."""
  CMD_VERSION_SIZE = 120

  async def main_loop():
    ws_connections = asyncio.Queue()

    tcp_reader = None
    tcp_writer = None

    active_ws_writer = None
    kick_event = asyncio.Event()

    async def handle_tcp_client(reader, writer):
      nonlocal tcp_reader, tcp_writer
      if tcp_writer is not None:
        logger.info("[UIServerProxy] Replacing previous TCP connection")
        tcp_writer.close()
        try:
          await tcp_writer.wait_closed()
        except Exception:
          pass
      tcp_reader = reader
      tcp_writer = writer
      sock = writer.get_extra_info("socket")
      if sock:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
      logger.info("[UIServerProxy] TCP connection accepted")

    async def handle_ws_connection(ws_reader, ws_writer):
      nonlocal active_ws_writer
      logger.info("[UIServerProxy] WebSocket connection attempt")
      if await _handle_ws_handshake(ws_reader, ws_writer):
        logger.info("[UIServerProxy] WebSocket connection accepted")
        sock = ws_writer.get_extra_info("socket")
        if sock:
          sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        if active_ws_writer is not None:
          logger.info("[UIServerProxy] Kicking previous browser connection")
          active_ws_writer.close()
          kick_event.set()

        await ws_connections.put((ws_reader, ws_writer))
      else:
        logger.info("[UIServerProxy] WebSocket handshake failed")
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
        f"[UIServerProxy] Proxy listening on TCP:{tcp_port} and WS:{ws_port}"
    )

    async with tcp_server, ws_server:
      while True:
        while tcp_reader is None:
          logger.info("[UIServerProxy] Waiting for TCP connection...")
          await asyncio.sleep(0.1)

        ws_reader, ws_writer = await ws_connections.get()

        while not ws_connections.empty():
          old_r, old_w = ws_connections.get_nowait()
          logger.info("[UIServerProxy] Discarding stale WS connection")
          old_w.close()

        active_ws_writer = ws_writer
        kick_event.clear()
        logger.info("[UIServerProxy] Both connected. Starting bridge.")

        try:
          browser_version = await _read_ws_binary_frame(ws_reader)
          logger.info(
              f"[UIServerProxy] Received browser CmdVersion: {len(browser_version)}"
              " bytes"
          )
        except Exception as e:
          logger.error(f"[UIServerProxy] Failed to read browser CmdVersion: {e}")
          ws_writer.close()
          active_ws_writer = None
          continue

        logger.info("[UIServerProxy] Forwarding CmdVersion handshake")
        tcp_writer.write(browser_version)
        await tcp_writer.drain()

        server_version = await tcp_reader.readexactly(CMD_VERSION_SIZE)
        logger.info(
            f"[UIServerProxy] Received server CmdVersion: {len(server_version)}"
            " bytes"
        )

        _write_ws_binary_frame(ws_writer, server_version)
        await ws_writer.drain()

        logger.info("[UIServerProxy] Handshake complete. Starting data bridge.")

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
          logger.error(f"[UIServerProxy] Bridge error: {e}")

        logger.info("[UIServerProxy] Bridge closed. Closing WS connection.")
        ws_writer.close()
        try:
          await ws_writer.wait_closed()
        except Exception:
          pass
        active_ws_writer = None

        logger.info("[UIServerProxy] Tearing down TCP to force full reconnect.")
        if tcp_writer is not None:
          tcp_writer.close()
          try:
            await tcp_writer.wait_closed()
          except Exception:
            pass
        tcp_reader = None
        tcp_writer = None

  _run_cancellable(main_loop)


class UIServerProxy:
  """WebSocket-to-TCP proxy bridging Dear ImGui UI to the browser."""

  def __init__(
      self,
      host: str = "0.0.0.0",
      tcp_port: int = 8888,
      ws_port: int = 8890,
  ):
    self.host = host
    self.tcp_port = tcp_port
    self.ws_port = ws_port
    self._proxy_thread = None

  def start(self) -> None:
    """Start the WebSocket-to-TCP proxy in a background process."""
    self._proxy_thread = multiprocessing.Process(
        target=_run_proxy,
        args=(self.host, self.tcp_port, self.ws_port),
        daemon=True,
    )
    self._proxy_thread.start()

  def stop(self) -> None:
    """Stop the proxy."""
    if self._proxy_thread:
      self._proxy_thread.terminate()
      self._proxy_thread.join(timeout=5)
