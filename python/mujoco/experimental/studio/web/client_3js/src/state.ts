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

// Client for the web viewer's /state WebSocket. Wire format: see
// state_payload.h (tagged binary blocks) and web_server.py (text control
// messages: "state_ack", "heartbeat", roster lines).

// "MJWS" read as a little-endian u32.
const STATE_PAYLOAD_MAGIC = 0x53574a4d;
const STATE_PAYLOAD_VERSION = 1;

const TAG_PHYSICS_STATE = 1;
const TAG_RENDER_STATE = 2;
const TAG_EXTRA_GEOMS = 3;

const HEARTBEAT_INTERVAL_MS = 25_000;
const RECONNECT_DELAY_MS = 2_000;

export interface StatePayload {
  modelCrc32: number;
  /** mjtState spec signature + state vector, when the payload carried one. */
  physics?: { spec: number; values: Float64Array };
  /** Raw render-state block (camera/options/visual/statistic), unparsed. */
  renderState?: Uint8Array;
  /** Raw extra mjvGeom bytes, unparsed. */
  extraGeoms?: Uint8Array;
}

export function parseStatePayload(buffer: ArrayBuffer): StatePayload | null {
  const view = new DataView(buffer);
  if (buffer.byteLength < 12) return null;
  if (view.getUint32(0, true) !== STATE_PAYLOAD_MAGIC) return null;
  if (view.getUint16(4, true) !== STATE_PAYLOAD_VERSION) return null;
  const nblocks = view.getUint16(6, true);
  const payload: StatePayload = { modelCrc32: view.getUint32(8, true) };

  let offset = 12;
  for (let i = 0; i < nblocks; i++) {
    if (offset + 8 > buffer.byteLength) return null;
    const tag = view.getUint32(offset, true);
    const size = view.getUint32(offset + 4, true);
    offset += 8;
    if (offset + size > buffer.byteLength) return null;

    switch (tag) {
      case TAG_PHYSICS_STATE: {
        const spec = view.getInt32(offset, true);
        // slice() copies, so the Float64Array is aligned regardless of offset.
        payload.physics = {
          spec,
          values: new Float64Array(buffer.slice(offset + 4, offset + size)),
        };
        break;
      }
      case TAG_RENDER_STATE:
        payload.renderState = new Uint8Array(buffer.slice(offset, offset + size));
        break;
      case TAG_EXTRA_GEOMS:
        payload.extraGeoms = new Uint8Array(buffer.slice(offset, offset + size));
        break;
      default:
        break; // readers must skip unknown tags
    }
    offset += size;
  }
  return payload;
}

export interface StateClientCallbacks {
  /** A parsed state payload arrived. Return value is ignored; the client acks. */
  onPayload: (payload: StatePayload) => void;
  /** A text frame arrived (roster line or other session broadcast). */
  onText?: (message: string) => void;
  onConnected?: () => void;
  /** closeCode is 0 when the socket dropped without a close frame. */
  onDisconnected?: (closeCode: number) => void;
}

// Connects to /state, forwards payloads, and speaks the session protocol
// (acks after every applied payload, heartbeats to stay un-kicked, reconnect
// with a fixed delay).
export class StateClient {
  private ws: WebSocket | null = null;
  private heartbeatTimer: number | null = null;
  private closed = false;
  readonly sid: string;

  constructor(private readonly callbacks: StateClientCallbacks) {
    this.sid = Math.random().toString(36).slice(2, 10);
  }

  connect(): void {
    this.closed = false;
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${proto}//${location.host}/state?sid=${this.sid}`;
    const ws = new WebSocket(url);
    ws.binaryType = 'arraybuffer';
    this.ws = ws;

    ws.onopen = () => {
      this.callbacks.onConnected?.();
      this.heartbeatTimer = window.setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) ws.send('heartbeat');
      }, HEARTBEAT_INTERVAL_MS);
    };

    ws.onmessage = (event: MessageEvent) => {
      if (typeof event.data === 'string') {
        this.callbacks.onText?.(event.data);
        return;
      }
      const payload = parseStatePayload(event.data as ArrayBuffer);
      if (!payload) {
        console.warn('Dropping malformed state payload');
        return;
      }
      this.callbacks.onPayload(payload);
      if (ws.readyState === WebSocket.OPEN) ws.send('state_ack');
    };

    ws.onclose = (event: CloseEvent) => {
      this.stopHeartbeat();
      this.ws = null;
      this.callbacks.onDisconnected?.(event.code);
      if (!this.closed) {
        window.setTimeout(() => this.connect(), RECONNECT_DELAY_MS);
      }
    };
    ws.onerror = () => ws.close();
  }

  close(): void {
    this.closed = true;
    this.stopHeartbeat();
    this.ws?.close();
    this.ws = null;
  }

  private stopHeartbeat(): void {
    if (this.heartbeatTimer !== null) {
      window.clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }
}
