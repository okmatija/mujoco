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

// Loader and input bridge for the web_client_ui WASM overlay module (built by
// the emscripten build into public/uimod/). When present, the module owns the
// model, the /state session, the /ui NetImgui stream, and the camera; this
// page renders the scene with three.js from the module's pose views and
// forwards every input event into ImGui (scene interaction reaches the
// headless viewer over NetImgui, so perturb/picking/camera all work like the
// native client). When absent, main.ts falls back to the standalone TS path.

// The embind surface of web_client_3js.cc.
export interface UiModule {
  startApp(): void;
  frame(width: number, height: number, dpr: number, dtSec: number): number;
  allocModelBuffer(size: number): number;
  freeModelBuffer(ptr: number): void;
  parseModelBuffer(ptr: number, size: number): boolean;
  updateModelDownloadProgress(bytes: number, total: number, retries: number): void;
  onMouseMove(x: number, y: number): void;
  onMouseButton(button: number, down: boolean): void;
  onMouseWheel(dx: number, dy: number): void;
  onFocus(focused: boolean): void;
  onKey(key: number, down: boolean, ctrl: boolean, shift: boolean, alt: boolean, superKey: boolean): void;
  onTextInput(codepoint: number): void;
  keyFromDomCode(code: string): number;
  xposView(): Float64Array | null;
  xquatView(): Float64Array | null;
  lightXposView(): Float64Array | null;
  lightXdirView(): Float64Array | null;
  cameraView(): Float64Array;
  stateSeen(): boolean;
  sessionRole(): number;
  HEAPU8: Uint8Array;
  onModelChanged?: () => void;
}

export const ROLE_NAMES = ['claiming…', 'controlling', 'spectating'];

// Attempts to load the overlay module; returns null if it is not deployed.
export async function tryLoadUiOverlay(): Promise<UiModule | null> {
  const url = new URL('uimod/web_client_ui.js', document.baseURI).href;
  try {
    const head = await fetch(url, { method: 'GET', cache: 'no-store' });
    if (!head.ok) return null;
  } catch {
    return null;
  }
  try {
    const factory = (await import(/* @vite-ignore */ url)).default;
    return (await factory()) as UiModule;
  } catch (error) {
    console.warn('[ui] overlay module failed to load:', error);
    return null;
  }
}

// Copies model bytes into the module heap and parses them there.
export function parseModelInModule(module: UiModule, bytes: Uint8Array): boolean {
  const ptr = module.allocModelBuffer(bytes.length);
  if (!ptr) return false;
  try {
    module.HEAPU8.set(bytes, ptr);
    return module.parseModelBuffer(ptr, bytes.length);
  } finally {
    module.freeModelBuffer(ptr);
  }
}

// DOM button order is left/middle/right; ImGui's is left/right/middle.
const DOM_TO_IMGUI_BUTTON = [0, 2, 1, 3, 4];

// Wires browser input on the overlay canvas into the module's ImGui context.
// The overlay canvas sits on top of the three.js canvas and owns all input.
export function wireOverlayInput(module: UiModule, canvas: HTMLCanvasElement): void {
  const keyCache = new Map<string, number>();
  const keyOf = (code: string): number => {
    let key = keyCache.get(code);
    if (key === undefined) {
      key = module.keyFromDomCode(code);
      keyCache.set(code, key);
    }
    return key;
  };

  canvas.addEventListener('pointermove', (e) => {
    module.onMouseMove(e.clientX, e.clientY);
  });
  canvas.addEventListener('pointerdown', (e) => {
    canvas.setPointerCapture(e.pointerId);
    canvas.focus();
    module.onMouseMove(e.clientX, e.clientY);
    module.onMouseButton(DOM_TO_IMGUI_BUTTON[e.button] ?? 0, true);
  });
  canvas.addEventListener('pointerup', (e) => {
    module.onMouseButton(DOM_TO_IMGUI_BUTTON[e.button] ?? 0, false);
  });
  canvas.addEventListener('pointerleave', () => {
    module.onMouseMove(-Number.MAX_VALUE, -Number.MAX_VALUE);
  });
  canvas.addEventListener(
    'wheel',
    (e) => {
      module.onMouseWheel(-e.deltaX / 100, -e.deltaY / 100);
      e.preventDefault();
    },
    { passive: false }
  );
  canvas.addEventListener('contextmenu', (e) => e.preventDefault());

  window.addEventListener('keydown', (e) => {
    module.onKey(keyOf(e.code), true, e.ctrlKey, e.shiftKey, e.altKey, e.metaKey);
    if (e.key.length === 1 && !e.ctrlKey && !e.metaKey) {
      module.onTextInput(e.key.codePointAt(0)!);
    }
    // Keep Tab and Space from moving browser focus / scrolling the page.
    if (e.code === 'Tab' || e.code === 'Space') e.preventDefault();
  });
  window.addEventListener('keyup', (e) => {
    module.onKey(keyOf(e.code), false, e.ctrlKey, e.shiftKey, e.altKey, e.metaKey);
  });
  window.addEventListener('blur', () => module.onFocus(false));
  window.addEventListener('focus', () => module.onFocus(true));
}

// Keeps the overlay canvas backing store matched to the window size and dpr.
export function resizeOverlayCanvas(canvas: HTMLCanvasElement): void {
  const dpr = window.devicePixelRatio;
  const w = Math.round(window.innerWidth * dpr);
  const h = Math.round(window.innerHeight * dpr);
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
}
