"""State server for streaming MuJoCo simulation state to the browser.

This module provides the StateServer class that:
1. Manages shared memory for zero-copy state transfer from the sim thread.
2. Runs a WebSocket server in a child process that broadcasts state at ~60Hz.

Each WebSocket message is one opaque, variable-length payload produced by
UiServer.serialize_state_payload (a sequence of tagged blocks; see
render_state.h for the format). The server treats it as latest-wins bytes.
"""

import asyncio
import base64
import ctypes
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


class _StateServerFormatter(logging.Formatter):

  def format(self, record):
    level_char = record.levelname[0]
    dt = datetime.datetime.fromtimestamp(record.created)
    date_str = dt.strftime("%m%d")
    time_str = dt.strftime("%H:%M:%S.%f")
    filename = os.path.basename(record.pathname)
    line = record.lineno
    msg = record.getMessage()
    return f"{level_char}{date_str} {time_str} {filename}:{line}] {msg}"


logger = logging.getLogger("StateServer")
logger.setLevel(logging.INFO)
if not logger.handlers:
  handler = logging.StreamHandler(sys.stdout)
  handler.setFormatter(_StateServerFormatter())
  logger.addHandler(handler)
  logger.propagate = False


WS_MAGIC = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


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


# WebSocket close code telling a browser that another browser took over the
# single viewer slot. The client must stop reconnecting when it sees this
# (otherwise two tabs kick each other in an endless loop).
WS_CLOSE_SUPERSEDED = 4000


def _write_ws_close_frame(writer, code, reason=b""):
  """Write a WebSocket close frame with the given status code."""
  payload = struct.pack("!H", code) + reason
  writer.write(bytearray([0x88, len(payload)]) + payload)


def _run_state_server(host, ws_port, shm_array, shm_capacity, generation):
  """Runs an asyncio state WebSocket server.

  Reads the latest payload from a shared multiprocessing.Array and broadcasts
  it to the browser. The shared array holds [u32 length][payload bytes] so
  payload sizes can vary per frame (e.g. with the number of extra geoms).
  """

  async def main_loop():
    active_writer = None
    last_gen = 0

    async def handle_ws_client(reader, writer):
      nonlocal active_writer
      logger.info(f"[StateWS] Connection attempt on port {ws_port}")
      if await _handle_ws_handshake(reader, writer):
        logger.info("[StateWS] Browser connected")
        sock = writer.get_extra_info("socket")
        if sock:
          sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        if active_writer is not None:
          logger.info("[StateWS] Kicking previous browser connection")
          try:
            _write_ws_close_frame(
                active_writer, WS_CLOSE_SUPERSEDED, b"superseded"
            )
          except (ConnectionResetError, BrokenPipeError, OSError):
            pass
          active_writer.close()
        active_writer = writer
      else:
        logger.info("[StateWS] Handshake failed")
        writer.close()

    server = await asyncio.start_server(handle_ws_client, host, ws_port)
    logger.info(f"[StateWS] Listening on ws://{host}:{ws_port}")

    # Not `async with`: on Python >= 3.12 Server.__aexit__ awaits
    # wait_closed(), which blocks until every accepted connection is closed,
    # so cancellation (SIGTERM from StateServer.stop) would hang the process
    # while a browser is connected. Close explicitly instead.
    try:
      while True:
        await asyncio.sleep(1.0 / 60.0)  # ~60Hz polling

        # Check if new state is available.
        cur_gen = generation.value
        if cur_gen == last_gen:
          continue
        last_gen = cur_gen

        if active_writer is None or active_writer.is_closing():
          active_writer = None
          continue

        # Read the latest payload from the shared array: [u32 length][bytes].
        (used,) = struct.unpack("<I", bytes(shm_array[:4]))
        if used == 0 or used > shm_capacity:
          continue
        payload = bytes(shm_array[4 : 4 + used])

        # Send as WebSocket binary frame.
        try:
          _write_ws_binary_frame(active_writer, payload)
          await active_writer.drain()
        except (ConnectionResetError, BrokenPipeError, OSError):
          logger.info("[StateWS] Browser disconnected")
          active_writer = None
    finally:
      if active_writer is not None:
        active_writer.close()
      server.close()

  _run_cancellable(main_loop)


class StateServer:
  """WebSocket server that streams simulation state to the browser.

  This class manages shared memory for zero-copy state transfer from the
  simulation thread, and spawns a child process that broadcasts state
  updates to connected browser clients over WebSocket.

  Binds loopback by default: browsers reach it through the WebServer's
  public router at path /state.
  """

  def __init__(
      self,
      host: str = "127.0.0.1",
      state_ws_port: int = 8891,
      max_payload_size: int = 0,
  ):
    self.host = host
    self.state_ws_port = state_ws_port
    self._state_thread = None

    # Shared memory for zero-copy payload transfer to the state server
    # process: [u32 length][payload bytes], so payloads can vary in size up
    # to max_payload_size (see UiServer.max_state_payload_size).
    self._shm_capacity = max_payload_size
    self._shm_array = (
        multiprocessing.RawArray(ctypes.c_char, 4 + self._shm_capacity)
        if self._shm_capacity > 0
        else None
    )
    self._generation = multiprocessing.Value("Q", 0)  # uint64 counter

  def update_state(self, payload: bytes) -> None:
    """Update the latest state payload. Called from the sim thread."""
    if self._shm_array is None:
      return
    if len(payload) > self._shm_capacity:
      logger.error(
          f"[StateWS] Payload of {len(payload)} bytes exceeds capacity"
          f" {self._shm_capacity}; dropping."
      )
      return
    ctypes.memmove(
        self._shm_array, struct.pack("<I", len(payload)) + payload,
        4 + len(payload),
    )
    self._generation.value += 1

  def start(self) -> None:
    """Start the state WebSocket server in a background process."""
    if self._shm_array is None:
      logger.info("[StateWS] State server not started (no state configured).")
      return
    self._state_thread = multiprocessing.Process(
        target=_run_state_server,
        args=(
            self.host,
            self.state_ws_port,
            self._shm_array,
            self._shm_capacity,
            self._generation,
        ),
        daemon=True,
    )
    self._state_thread.start()

  def stop(self) -> None:
    """Stop the state server, escalating to SIGKILL if it hangs."""
    if self._state_thread:
      self._state_thread.terminate()
      self._state_thread.join(timeout=2)
      if self._state_thread.is_alive():
        logger.warning(
            f"[StateWS] Process {self._state_thread.pid} ignored SIGTERM;"
            " killing."
        )
        self._state_thread.kill()
        self._state_thread.join(timeout=2)
