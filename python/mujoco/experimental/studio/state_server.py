"""State server for streaming MuJoCo simulation state to the browser.

This module provides the StateServer class that:
1. Manages shared memory for zero-copy state transfer from the sim thread.
2. Runs a WebSocket server in a child process that broadcasts state at ~60Hz.

The wire format is: [4 bytes: state_sig (int32 LE)] [N bytes: state]
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


def _run_state_server(
    host, ws_port, shm_array, shm_size, generation, state_sig
):
  """Runs an asyncio state WebSocket server.

  Reads state from a shared multiprocessing.Array and broadcasts it to the
  browser. The wire format is: [4 bytes: state_sig (int32 LE)] [N bytes: state]
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
          active_writer.close()
        active_writer = writer
      else:
        logger.info("[StateWS] Handshake failed")
        writer.close()

    server = await asyncio.start_server(handle_ws_client, host, ws_port)
    logger.info(f"[StateWS] Listening on ws://{host}:{ws_port}")

    async with server:
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

        # Read state from shared array.
        state_bytes = bytes(shm_array[:shm_size])

        # Build wire frame: [4B sig LE] [state bytes]
        payload = struct.pack("<i", state_sig) + state_bytes

        # Send as WebSocket binary frame.
        try:
          _write_ws_binary_frame(active_writer, payload)
          await active_writer.drain()
        except (ConnectionResetError, BrokenPipeError, OSError):
          logger.info("[StateWS] Browser disconnected")
          active_writer = None

  _run_cancellable(main_loop)


class StateServer:
  """WebSocket server that streams simulation state to the browser.

  This class manages shared memory for zero-copy state transfer from the
  simulation thread, and spawns a child process that broadcasts state
  updates to connected browser clients over WebSocket.
  """

  def __init__(
      self,
      host: str = "0.0.0.0",
      state_ws_port: int = 8891,
      state_size: int = 0,
      state_sig: int = 0,
  ):
    self.host = host
    self.state_ws_port = state_ws_port
    self.state_sig = state_sig
    self._state_thread = None

    # Shared memory for zero-copy state transfer to the state server process.
    # state_size is the total payload size in bytes (physics state + vis state).
    self._shm_byte_size = state_size
    self._shm_array = (
        multiprocessing.RawArray(ctypes.c_char, self._shm_byte_size)
        if self._shm_byte_size > 0
        else None
    )
    self._generation = multiprocessing.Value("Q", 0)  # uint64 counter

  def update_state(self, state_bytes: bytes) -> None:
    """Update the latest simulation state. Called from the sim thread."""
    if self._shm_array is None:
      return
    ctypes.memmove(self._shm_array, state_bytes, len(state_bytes))
    self._generation.value += 1

  def start(self) -> None:
    """Start the state WebSocket server in a background process."""
    if self._shm_array is None:
      logger.info("[Link] State server not started (no state configured).")
      return
    self._state_thread = multiprocessing.Process(
        target=_run_state_server,
        args=(
            self.host,
            self.state_ws_port,
            self._shm_array,
            self._shm_byte_size,
            self._generation,
            self.state_sig,
        ),
        daemon=True,
    )
    self._state_thread.start()

  def stop(self) -> None:
    """Stop the state server."""
    if self._state_thread:
      self._state_thread.terminate()
      self._state_thread.join(timeout=5)
