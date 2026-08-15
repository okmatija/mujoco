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

// three.js browser client for the Studio Web Viewer (--gfx=web_3js).
//
// Boot: fetch /model (.mjb) -> load it with the MuJoCo WASM bindings -> build
// a three.js scene (scene.ts) and the embedded gaussian splat, if any
// (splat.ts). Live: /state payloads carry the physics state vector; the client
// applies it with mj_setState + mj_forward and mirrors xpos/xquat onto the
// body groups. Without a /state server (static dev via `npm run dev`), the
// client steps the simulation locally instead.

import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import loadMujoco from '@mujoco/mujoco';
import type { MainModule } from '@mujoco/mujoco';

import { updateHeadlightFromCamera, updateLightsFromData } from './lights';
import { buildScene, disposeScene, syncBodyPoses } from './scene';
import type { SceneHandles } from './scene';
import { disposeSplat, loadSplatFromModel } from './splat';
import type { SplatMesh } from './splat';
import { StateClient } from './state';
import type { StatePayload } from './state';

const statusEl = document.getElementById('status')!;
function setStatus(text: string): void {
  statusEl.textContent = text;
}

// Max catch-up steps per frame for the local (serverless) simulation.
const MAX_LOCAL_STEPS_PER_FRAME = 50;
// If no live payload arrives this soon after boot, start simulating locally.
const LOCAL_SIM_FALLBACK_MS = 1500;

class App {
  private mujoco!: MainModule;
  private mj!: any;

  private readonly scene = new THREE.Scene();
  private renderer!: THREE.WebGLRenderer;
  private camera!: THREE.PerspectiveCamera;
  private controls!: OrbitControls;

  private model: any = null;
  private data: any = null;
  private handles: SceneHandles | null = null;
  private splat: SplatMesh | null = null;

  private stateClient!: StateClient;
  private modelCrc: number | null = null;
  private live = false;
  private localSim = false;
  private refetching = false;
  private lastFrameTime = 0;
  private rosterLine = '';

  async boot(): Promise<void> {
    setStatus('loading MuJoCo…');
    this.mujoco = await loadMujoco();
    this.mj = this.mujoco as any;
    try {
      this.mj.FS.mkdir('/working');
    } catch {
      // already exists
    }

    this.initRenderer();

    setStatus('fetching model…');
    await this.reloadModel();

    this.stateClient = new StateClient({
      onPayload: (payload) => this.onPayload(payload),
      onText: (message) => {
        this.rosterLine = message;
        this.updateStatus();
      },
      onConnected: () => this.updateStatus(),
      onDisconnected: () => this.updateStatus(),
    });
    this.stateClient.connect();

    window.setTimeout(() => {
      if (!this.live) {
        this.localSim = true;
        this.updateStatus();
      }
    }, LOCAL_SIM_FALLBACK_MS);

    this.lastFrameTime = performance.now();
    this.renderer.setAnimationLoop(() => this.frame());
    this.updateStatus();
  }

  private initRenderer(): void {
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    document.getElementById('app')!.appendChild(this.renderer.domElement);

    this.camera = new THREE.PerspectiveCamera(
      45,
      window.innerWidth / window.innerHeight,
      0.01,
      1000
    );
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;

    window.addEventListener('resize', () => {
      this.camera.aspect = window.innerWidth / window.innerHeight;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(window.innerWidth, window.innerHeight);
    });
  }

  // Fetches /model and (re)builds the mujoco objects and the three.js scene.
  private async reloadModel(): Promise<void> {
    const response = await fetch('model', { cache: 'no-store' });
    if (!response.ok) {
      setStatus(`failed to fetch /model: HTTP ${response.status}`);
      throw new Error(`/model fetch failed: ${response.status}`);
    }
    const bytes = new Uint8Array(await response.arrayBuffer());

    if (this.handles) {
      disposeScene(this.handles);
      this.handles = null;
    }
    if (this.splat) {
      disposeSplat(this.splat, this.scene);
      this.splat = null;
    }
    this.data?.delete?.();
    this.model?.delete?.();
    this.data = null;
    this.model = null;

    this.mj.FS.writeFile('/working/model.mjb', bytes);
    const vfs = new this.mj.MjVFS();
    try {
      this.model = this.mj.MjModel.mj_loadModel('/working/model.mjb', vfs);
    } finally {
      vfs.delete();
    }
    if (!this.model) {
      setStatus('failed to load model (.mjb version mismatch?)');
      throw new Error('mj_loadModel returned null');
    }
    this.data = new this.mj.MjData(this.model);
    this.mujoco.mj_forward(this.model, this.data);

    this.handles = buildScene(this.mujoco, this.model);
    this.scene.add(this.handles.root);
    this.scene.background = this.handles.skybox ?? new THREE.Color(0x26262c);
    this.splat = loadSplatFromModel(this.model, this.scene, this.renderer);
    syncBodyPoses(this.data, this.handles.bodies);
    this.frameCamera();
  }

  private frameCamera(): void {
    const box = new THREE.Box3();
    for (const [b, group] of this.handles?.bodies ?? []) {
      if (b === 0) continue; // skip world (infinite plane quad pollutes bounds)
      box.expandByObject(group);
    }
    const center = new THREE.Vector3();
    let radius = 1.5;
    if (!box.isEmpty()) {
      box.getCenter(center);
      radius = Math.max(0.25, box.getSize(new THREE.Vector3()).length() / 2);
    }
    const dir = new THREE.Vector3(1, 0.45, 1).normalize();
    this.camera.position.copy(center).addScaledVector(dir, radius * 2.6);
    this.camera.near = radius / 100;
    this.camera.far = Math.max(1000, radius * 100);
    this.camera.updateProjectionMatrix();
    this.controls.target.copy(center);
    this.controls.update();
  }

  private onPayload(payload: StatePayload): void {
    if (!this.model || !this.data) return;
    this.live = true;
    this.localSim = false;

    if (this.modelCrc !== null && payload.modelCrc32 !== this.modelCrc && !this.refetching) {
      // Model hot-swap: refetch /model before applying any further state.
      this.refetching = true;
      this.modelCrc = payload.modelCrc32;
      this.reloadModel()
        .catch((error) => console.error('Model refetch failed:', error))
        .finally(() => {
          this.refetching = false;
        });
      return;
    }
    this.modelCrc = payload.modelCrc32;
    if (this.refetching || !payload.physics) return;

    this.mujoco.mj_setState(
      this.model,
      this.data,
      Array.from(payload.physics.values),
      payload.physics.spec
    );
    this.mujoco.mj_forward(this.model, this.data);
    this.updateStatus();
  }

  private frame(): void {
    const now = performance.now();
    const dt = Math.min(0.25, (now - this.lastFrameTime) / 1000);
    this.lastFrameTime = now;

    if (this.localSim && this.model && this.data) {
      const target = this.data.time + dt;
      let steps = 0;
      while (this.data.time < target && steps < MAX_LOCAL_STEPS_PER_FRAME) {
        this.mujoco.mj_step(this.model, this.data);
        steps++;
      }
    }

    if (this.handles && this.data) {
      syncBodyPoses(this.data, this.handles.bodies);
      updateLightsFromData(this.mujoco, this.data, this.handles.lights);
      updateHeadlightFromCamera(this.camera, this.handles.lights);
    }

    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  private updateStatus(): void {
    const mode = this.live
      ? 'live'
      : this.localSim
        ? 'local sim (no /state server)'
        : 'connecting…';
    const splat = this.splat ? ' · splat' : '';
    const roster = this.rosterLine ? `\n${this.rosterLine}` : '';
    setStatus(`${mode}${splat}${roster}`);
  }
}

new App().boot().catch((error) => {
  console.error(error);
  setStatus(String(error));
});
