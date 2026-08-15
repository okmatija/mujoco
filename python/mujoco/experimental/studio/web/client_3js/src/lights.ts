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

// mjModel lights -> three.js lights, plus per-frame pose updates.
// Adapted from mjswan (https://github.com/ttktjmt/mjswan), Apache-2.0.

import * as THREE from 'three';
import type { MainModule } from '@mujoco/mujoco';
import { mjcToThreeCoordinate } from './coordinate';

// mjtByte fields are exposed as bool memory views, which embind cannot
// marshal in the 3.10 bindings — reading the property throws. Fall back to
// MuJoCo's defaults (active/castshadow true) when that happens.
function readByteField(model: any, field: string): ArrayLike<number> | null {
  try {
    return model[field] ?? null;
  } catch {
    return null;
  }
}

export function createLights(
  mujoco: MainModule,
  model: any,
  mujocoRoot: THREE.Group
): THREE.Light[] {
  const mj = mujoco as any;
  const lights: THREE.Light[] = [];
  const ambientSum = new THREE.Color(0, 0, 0);

  const lightActive = readByteField(model, 'light_active');
  const lightCastshadow = readByteField(model, 'light_castshadow');

  for (let l = 0; l < model.nlight; l++) {
    if (lightActive && !lightActive[l]) continue;

    const lightType = Number(model.light_type[l]);
    let light: THREE.DirectionalLight | THREE.PointLight | THREE.SpotLight;

    switch (lightType) {
      case mj.mjtLightType.mjLIGHT_DIRECTIONAL.value:
        light = new THREE.DirectionalLight();
        mujocoRoot.add((light as THREE.DirectionalLight).target);
        break;
      case mj.mjtLightType.mjLIGHT_POINT.value:
        light = new THREE.PointLight();
        break;
      case mj.mjtLightType.mjLIGHT_SPOT.value:
        light = new THREE.SpotLight();
        mujocoRoot.add((light as THREE.SpotLight).target);
        break;
      default:
        console.warn(`Skipping unsupported light type ${lightType} (light ${l})`);
        continue;
    }

    light.userData.mjIndex = l;
    light.userData.mjType = lightType;

    const diffuse = new THREE.Color().fromArray(
      Array.from(model.light_diffuse.slice(l * 3, l * 3 + 3))
    );
    const specular = new THREE.Color().fromArray(
      Array.from(model.light_specular.slice(l * 3, l * 3 + 3))
    );
    const combined = diffuse.clone().add(specular);
    const luminance = Math.max(combined.r, combined.g, combined.b);
    light.color = luminance > 0
      ? combined.multiplyScalar(1 / luminance)
      : new THREE.Color(0, 0, 0);

    const intensityMultiplier = model.light_intensity?.[l] || 0.5;
    light.intensity = luminance * intensityMultiplier * Math.PI;

    ambientSum.add(
      new THREE.Color().fromArray(
        Array.from(model.light_ambient.slice(l * 3, l * 3 + 3))
      )
    );

    light.castShadow = lightCastshadow ? Boolean(lightCastshadow[l]) : true;
    if (light.castShadow) {
      light.shadow!.mapSize.width = 2048;
      light.shadow!.mapSize.height = 2048;
      light.shadow!.camera.near = 0.1;
      light.shadow!.camera.far = 30;
      light.shadow!.radius = (model.light_bulbradius?.[l] ?? 0.02) * 50;
    }

    const pos = mjcToThreeCoordinate(model.light_pos.slice(l * 3, l * 3 + 3));
    const dir = mjcToThreeCoordinate(model.light_dir.slice(l * 3, l * 3 + 3)).normalize();

    if (lightType === mj.mjtLightType.mjLIGHT_DIRECTIONAL.value) {
      const len = Math.max(1, model.light_range?.[l] || 10);
      const dl = light as THREE.DirectionalLight;
      dl.position.copy(pos).addScaledVector(dir, -len);
      dl.target.position.copy(pos);
    } else if (lightType === mj.mjtLightType.mjLIGHT_SPOT.value) {
      const sl = light as THREE.SpotLight;
      sl.position.copy(pos);
      sl.target.position.copy(pos).add(dir);
      sl.angle = (model.light_cutoff[l] * Math.PI) / 180;
      sl.penumbra = 1 / (1 + model.light_exponent[l]);
    } else {
      light.position.copy(pos);
    }

    if (lightType !== mj.mjtLightType.mjLIGHT_DIRECTIONAL.value) {
      const att = model.light_attenuation.slice(l * 3, l * 3 + 3);
      const pl = light as THREE.PointLight | THREE.SpotLight;
      pl.distance = model.light_range?.[l] ?? 0;
      pl.decay = att[2] > 0 ? 2 : att[1] > 0 ? 1 : 0;
    }

    mujocoRoot.add(light);
    lights.push(light);
  }

  // The headlight follows the camera (see updateHeadlightFromCamera).
  const headlight = readHeadlight(model);
  if (headlight.active) {
    ambientSum.add(headlight.ambient);
    const combined = headlight.diffuse.clone().add(headlight.specular);
    const luminance = Math.max(combined.r, combined.g, combined.b);
    const head = new THREE.DirectionalLight();
    head.color = luminance > 0
      ? combined.multiplyScalar(1 / luminance)
      : new THREE.Color(0, 0, 0);
    head.intensity = luminance * Math.PI * 0.7;
    head.castShadow = false;
    head.userData.isHeadlight = true;
    mujocoRoot.add(head.target);
    mujocoRoot.add(head);
    lights.push(head);
  }

  if (!ambientSum.equals(new THREE.Color(0, 0, 0))) {
    const ambient = new THREE.AmbientLight(ambientSum, 1.0);
    mujocoRoot.add(ambient);
    lights.push(ambient);
  }

  // A model with no light sources at all would render black; give it the
  // default MuJoCo-ish headlight instead.
  if (lights.length === 0) {
    const head = new THREE.DirectionalLight(0xffffff, 1.2);
    head.userData.isHeadlight = true;
    mujocoRoot.add(head.target);
    mujocoRoot.add(head);
    lights.push(head);
    const ambient = new THREE.AmbientLight(0xffffff, 0.35);
    mujocoRoot.add(ambient);
    lights.push(ambient);
  }

  return lights;
}

// mjVisual headlight, tolerant of the binding exposing it as `vis` or
// `visual` or not at all.
function readHeadlight(model: any): {
  active: boolean;
  ambient: THREE.Color;
  diffuse: THREE.Color;
  specular: THREE.Color;
} {
  const vis = model.vis ?? model.visual;
  const headlight = vis?.headlight;
  if (!headlight || !headlight.active) {
    return {
      active: false,
      ambient: new THREE.Color(),
      diffuse: new THREE.Color(),
      specular: new THREE.Color(),
    };
  }
  const toColor = (v: any) =>
    new THREE.Color().fromArray(Array.from(v ?? [0, 0, 0]).map(Number));
  return {
    active: true,
    ambient: toColor(headlight.ambient),
    diffuse: toColor(headlight.diffuse),
    specular: toColor(headlight.specular),
  };
}

// Moves body-mounted lights to their simulated poses.
export function updateLightsFromData(
  mujoco: MainModule,
  data: any,
  lights: THREE.Light[]
): void {
  const mj = mujoco as any;
  if (!data?.light_xpos || !data?.light_xdir) return;

  for (const light of lights) {
    const idx = light.userData.mjIndex;
    const type = light.userData.mjType;
    if (idx == null) continue;

    const pos = mjcToThreeCoordinate(data.light_xpos.slice(idx * 3, idx * 3 + 3));
    const dir = mjcToThreeCoordinate(data.light_xdir.slice(idx * 3, idx * 3 + 3)).normalize();

    if (type === mj.mjtLightType.mjLIGHT_DIRECTIONAL.value) {
      const dl = light as THREE.DirectionalLight;
      const len = Math.max(1, (dl.shadow?.camera?.far as number) || 10);
      dl.target.position.copy(pos);
      dl.position.copy(pos).addScaledVector(dir, -len);
    } else if (type === mj.mjtLightType.mjLIGHT_SPOT.value) {
      const sl = light as THREE.SpotLight;
      sl.position.copy(pos);
      sl.target.position.copy(pos.clone().add(dir));
    } else if (type === mj.mjtLightType.mjLIGHT_POINT.value) {
      light.position.copy(pos);
    }
  }
}

// Keeps the headlight aligned with the viewer camera.
export function updateHeadlightFromCamera(
  camera: THREE.Camera,
  lights: THREE.Light[]
): void {
  const dir = new THREE.Vector3();
  for (const light of lights) {
    if (!light.userData.isHeadlight) continue;
    const dl = light as THREE.DirectionalLight;
    camera.getWorldDirection(dir);
    dl.position.copy(camera.position);
    dl.target.position.copy(camera.position).add(dir);
  }
}
