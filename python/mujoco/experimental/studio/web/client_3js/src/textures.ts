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

// mjModel textures -> three.js textures.
// Adapted from mjswan (https://github.com/ttktjmt/mjswan), Apache-2.0.

import * as THREE from 'three';
import type { MainModule } from '@mujoco/mujoco';

function expandChannelsToRGBA(src: Uint8Array, dest: Uint8Array, nchannel: number): void {
  const pixelCount = dest.length / 4;

  switch (nchannel) {
    case 1:
      for (let p = 0; p < pixelCount; p++) {
        const l = src[p];
        dest[p * 4 + 0] = l;
        dest[p * 4 + 1] = l;
        dest[p * 4 + 2] = l;
        dest[p * 4 + 3] = 255;
      }
      break;
    case 2:
      for (let p = 0; p < pixelCount; p++) {
        const l = src[p * 2 + 0];
        const a = src[p * 2 + 1];
        dest[p * 4 + 0] = l;
        dest[p * 4 + 1] = l;
        dest[p * 4 + 2] = l;
        dest[p * 4 + 3] = a;
      }
      break;
    case 3:
      for (let p = 0; p < pixelCount; p++) {
        dest[p * 4 + 0] = src[p * 3 + 0];
        dest[p * 4 + 1] = src[p * 3 + 1];
        dest[p * 4 + 2] = src[p * 3 + 2];
        dest[p * 4 + 3] = 255;
      }
      break;
    case 4:
      dest.set(src);
      break;
    default:
      for (let p = 0; p < pixelCount; p++) {
        const l = p < src.length ? src[p] : 0;
        dest[p * 4 + 0] = l;
        dest[p * 4 + 1] = l;
        dest[p * 4 + 2] = l;
        dest[p * 4 + 3] = 255;
      }
  }
}

function create2DTexture(model: any, texId: number): THREE.DataTexture | null {
  const width = Number(model.tex_width[texId]);
  const height = Number(model.tex_height[texId]);
  if (!width || !height) return null;

  const texAdr = Number(model.tex_adr[texId]);
  const nchannel = Number(model.tex_nchannel[texId]);
  if (nchannel < 1 || nchannel > 4) {
    console.warn(`Invalid channel count ${nchannel} for texture ${texId}`);
    return null;
  }

  const pixelCount = width * height;
  const srcByteCount = pixelCount * nchannel;
  if (!model.tex_data || model.tex_data.length < texAdr + srcByteCount) {
    console.warn(`Insufficient texture data for texture ${texId}`);
    return null;
  }

  const src = model.tex_data.subarray(texAdr, texAdr + srcByteCount);
  const textureData = new Uint8Array(pixelCount * 4);
  expandChannelsToRGBA(src, textureData, nchannel);

  const texture = new THREE.DataTexture(
    textureData,
    width,
    height,
    THREE.RGBAFormat,
    THREE.UnsignedByteType
  );
  texture.needsUpdate = true;
  texture.flipY = false;
  texture.anisotropy = 4;
  texture.magFilter = THREE.LinearFilter;
  texture.minFilter = THREE.LinearMipmapLinearFilter;
  texture.generateMipmaps = true;
  texture.colorSpace = THREE.SRGBColorSpace;

  return texture;
}

function createCubeTexture(
  model: any,
  texId: number,
  faceOrder = [3, 2, 0, 1, 4, 5]
): THREE.CubeTexture | null {
  const width = Number(model.tex_width[texId]);
  const height = Number(model.tex_height[texId]);
  if (!width || !height) return null;

  const faceSize = width;
  let isRepeated: boolean;
  if (height === width) {
    isRepeated = true;
  } else if (height === 6 * width) {
    isRepeated = false;
  } else {
    console.warn(`Invalid dimensions for cube texture ${texId}: ${width}x${height}`);
    return null;
  }

  const texAdr = Number(model.tex_adr[texId]);
  const nchannel = Number(model.tex_nchannel[texId]);
  const facePixelCount = faceSize * faceSize;
  const faceSrcByteCount = facePixelCount * nchannel;

  const faces: Uint8Array[] = [];
  for (let faceIdx = 0; faceIdx < 6; faceIdx++) {
    const faceOffset = texAdr + (isRepeated ? 0 : faceIdx * faceSrcByteCount);
    if (!model.tex_data || model.tex_data.length < faceOffset + faceSrcByteCount) {
      console.warn(`Insufficient texture data for cube face ${faceIdx} in texture ${texId}`);
      return null;
    }
    const src = model.tex_data.subarray(faceOffset, faceOffset + faceSrcByteCount);
    const faceData = new Uint8Array(facePixelCount * 4);
    expandChannelsToRGBA(src, faceData, nchannel);
    faces.push(faceData);
  }

  const reorderedFaces = faceOrder.map((i) => faces[i]);
  const images: HTMLCanvasElement[] = [];
  for (let i = 0; i < reorderedFaces.length; i++) {
    const canvas = document.createElement('canvas');
    canvas.width = faceSize;
    canvas.height = faceSize;
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      console.warn(`Failed to acquire 2D context for cube face ${i} in texture ${texId}`);
      return null;
    }
    const imageData = ctx.createImageData(faceSize, faceSize);
    imageData.data.set(reorderedFaces[i]);
    ctx.putImageData(imageData, 0, 0);
    images.push(canvas);
  }

  const cubeTexture = new THREE.CubeTexture();
  cubeTexture.image = images as unknown as HTMLImageElement[];
  cubeTexture.needsUpdate = true;
  cubeTexture.format = THREE.RGBAFormat;
  cubeTexture.flipY = false;
  cubeTexture.magFilter = THREE.LinearFilter;
  cubeTexture.minFilter = THREE.LinearFilter;
  cubeTexture.generateMipmaps = false;
  cubeTexture.colorSpace = THREE.SRGBColorSpace;

  return cubeTexture;
}

// Builds the three.js texture for mjModel texture `texId` (2D or cube).
export function createTexture(
  mujoco: MainModule,
  model: any,
  texId: number
): THREE.Texture | null {
  if (!model || texId < 0) return null;

  const mj = mujoco as any;
  const type = Number(model.tex_type[texId]);
  if (type === mj.mjtTexture.mjTEXTURE_2D.value) {
    return create2DTexture(model, texId);
  }
  if (type === mj.mjtTexture.mjTEXTURE_CUBE.value) {
    return createCubeTexture(model, texId);
  }
  console.warn(`Unsupported texture type ${type} for texture ${texId}`);
  return null;
}

// Builds the scene-background cube texture from the model's skybox, if any.
export function createSkyboxTexture(
  mujoco: MainModule,
  model: any
): THREE.CubeTexture | null {
  const mj = mujoco as any;
  for (let i = 0; i < model.ntex; i++) {
    if (Number(model.tex_type[i]) !== mj.mjtTexture.mjTEXTURE_SKYBOX.value) continue;
    // Identity face order: MuJoCo stores skybox faces with +Z (index 2) up and
    // -Z (index 3) down, which maps directly onto three.js +Y/-Y slots.
    const cube = createCubeTexture(model, i, [0, 1, 2, 3, 4, 5]);
    if (cube) {
      // Flip faces horizontally: MuJoCo's cubemap is outside-in, but a
      // three.js skybox is viewed from inside.
      const faces = cube.image as unknown as HTMLCanvasElement[];
      for (let f = 0; f < faces.length; f++) {
        const src = faces[f];
        const flipped = document.createElement('canvas');
        flipped.width = src.width;
        flipped.height = src.height;
        const ctx = flipped.getContext('2d')!;
        ctx.translate(src.width, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(src, 0, 0);
        faces[f] = flipped;
      }
      cube.needsUpdate = true;
    }
    return cube;
  }
  return null;
}
