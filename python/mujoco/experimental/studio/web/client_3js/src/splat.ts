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

// Gaussian-splat environment rendering via Spark (@sparkjsdev/spark). The
// splat bytes and placement ride inside the model as <custom> elements written
// by the Python-side helper (studio splat.embed):
//   text    "studio/splat"       base64 of the .spz/.ply/.splat file
//   numeric "studio/splat/xform" [scale, x, y, z, roll°, pitch°, yaw°]
// Transform semantics follow mjswan so its published calibration constants
// carry over. Adapted from mjswan (https://github.com/ttktjmt/mjswan),
// Apache-2.0.

import * as THREE from 'three';
import { SplatMesh, SparkRenderer, getSplatFileType } from '@sparkjsdev/spark';
import { decodeBase64, findNumeric, findText } from './customs';

export type { SplatMesh };

export const SPLAT_TEXT_NAME = 'studio/splat';
export const SPLAT_XFORM_NAME = 'studio/splat/xform';

export interface SplatTransform {
  scale?: number;
  xOffset?: number;
  yOffset?: number;
  zOffset?: number;
  /** Degrees, applied on top of the COLMAP -> three.js base rotation. */
  roll?: number;
  pitch?: number;
  yaw?: number;
}

const DEG2RAD = Math.PI / 180;
const BASE_QUAT = new THREE.Quaternion().setFromEuler(new THREE.Euler(Math.PI, 0, 0));

export function applySplatTransform(splat: SplatMesh, transform: SplatTransform): void {
  const scale = transform.scale ?? 1.0;
  const roll = (transform.roll ?? 0.0) * DEG2RAD;
  const pitch = (transform.pitch ?? 0.0) * DEG2RAD;
  const yaw = (transform.yaw ?? 0.0) * DEG2RAD;

  splat.scale.setScalar(scale);

  // Splat scans use the COLMAP/OpenCV convention (Y-down, Z-into-scene);
  // rotating 180 degrees about X flips to three.js (Y-up, Z-towards-viewer).
  // User roll/pitch/yaw compose on top.
  const userQuat = new THREE.Quaternion().setFromEuler(new THREE.Euler(pitch, yaw, roll));
  splat.quaternion.copy(BASE_QUAT.clone().multiply(userQuat));

  splat.position.set(
    (transform.xOffset ?? 0) * scale,
    (transform.zOffset ?? 0) * scale,
    (transform.yOffset ?? 0) * scale
  );
}

// Spark 2.x renders splats through an explicit SparkRenderer object in the
// scene (0.1.x auto-installed one); without it a SplatMesh silently renders
// nothing. Reuse an existing one so model hot-swaps don't stack them.
function ensureSparkRenderer(
  scene: THREE.Scene,
  renderer: THREE.WebGLRenderer
): SparkRenderer {
  for (const child of scene.children) {
    if (child instanceof SparkRenderer) return child;
  }
  const spark = new SparkRenderer({ renderer });
  scene.add(spark);
  return spark;
}

// Builds a SplatMesh from raw splat file bytes and places it in the scene.
export function loadSplat(
  bytes: Uint8Array,
  transform: SplatTransform,
  scene: THREE.Scene,
  renderer: THREE.WebGLRenderer
): SplatMesh {
  ensureSparkRenderer(scene, renderer);
  const splat = new SplatMesh({
    fileBytes: bytes,
    fileType: getSplatFileType(bytes),
  });
  applySplatTransform(splat, transform);
  scene.add(splat);
  splat.initialized
    .then((mesh) => {
      console.log(`[splat] initialized: ${mesh.packedSplats?.numSplats} splats`);
      (window as any).__splatReady = true;
    })
    .catch((error: unknown) => {
      console.error('[splat] initialization failed:', error);
      (window as any).__splatError = String(error);
    });
  return splat;
}

// Loads the splat embedded in the model's custom fields, if present.
export function loadSplatFromModel(
  model: any,
  scene: THREE.Scene,
  renderer: THREE.WebGLRenderer
): SplatMesh | null {
  const b64 = findText(model, SPLAT_TEXT_NAME);
  if (!b64) return null;

  let bytes: Uint8Array;
  try {
    bytes = decodeBase64(b64);
  } catch (error) {
    console.warn('Embedded splat is not valid base64:', error);
    return null;
  }

  const xf = findNumeric(model, SPLAT_XFORM_NAME) ?? [];
  const transform: SplatTransform = {
    scale: xf[0] ?? 1.0,
    xOffset: xf[1] ?? 0.0,
    yOffset: xf[2] ?? 0.0,
    zOffset: xf[3] ?? 0.0,
    roll: xf[4] ?? 0.0,
    pitch: xf[5] ?? 0.0,
    yaw: xf[6] ?? 0.0,
  };
  return loadSplat(bytes, transform, scene, renderer);
}

export function disposeSplat(splat: SplatMesh, scene: THREE.Scene): void {
  scene.remove(splat);
  splat.dispose?.();
  // Spark keeps drawing its last async sort, which never completes if the mesh
  // leaves mid-sort, so scene.remove alone leaves the splat on screen for
  // good. Dropping the renderer too is safe: the next SplatMesh installs a
  // fresh one.
  for (const child of [...scene.children]) {
    if (child instanceof SparkRenderer) {
      scene.remove(child);
      child.dispose();
    }
  }
}
