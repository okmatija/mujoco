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

// Builds a three.js scene graph from a compiled mjModel: one THREE.Group per
// body, geoms as local children, plus lights and the skybox. Per-frame pose
// sync copies mjData.xpos/xquat into the body groups (syncBodyPoses).
// Adapted from mjswan (https://github.com/ttktjmt/mjswan), Apache-2.0.

import * as THREE from 'three';
import type { MainModule } from '@mujoco/mujoco';
import { getPosition, getQuaternion } from './coordinate';
import { createLights } from './lights';
import { createSkyboxTexture, createTexture } from './textures';

export interface SceneHandles {
  root: THREE.Group;
  bodies: Map<number, THREE.Group>;
  lights: THREE.Light[];
  skybox: THREE.CubeTexture | null;
}

function reflectanceParams(
  model: any,
  matId: number
): Pick<
  THREE.MeshPhysicalMaterialParameters,
  'specularIntensity' | 'reflectivity' | 'roughness' | 'metalness'
> {
  const specular = matId !== -1 ? (model.mat_specular?.[matId] ?? 0.5) : 0.5;
  const shininess = matId !== -1 ? (model.mat_shininess?.[matId] ?? 0.5) : 0.5;
  const reflectance = matId !== -1 ? (model.mat_reflectance?.[matId] ?? 0) : 0;
  const metallic = matId !== -1 ? (model.mat_metallic?.[matId] ?? -1) : -1;
  const roughnessAttr = matId !== -1 ? (model.mat_roughness?.[matId] ?? -1) : -1;

  return {
    specularIntensity: specular,
    reflectivity: reflectance,
    roughness: roughnessAttr >= 0 ? roughnessAttr : 1.0 - shininess,
    metalness: metallic >= 0 ? metallic : specular,
  };
}

// MuJoCo renders infinite planes with GL_OBJECT_PLANE texture coordinates and
// no horizon; this full-screen shader ray-casts the plane per fragment to
// reproduce that. depthWrite is off so the plane never occludes splats or
// transparent objects; depthTest still hides it behind opaque geometry.
function createInfinitePlaneShaderMaterial(params: {
  color: THREE.Color;
  opacity: number;
  texture: THREE.Texture | null;
  uvScaleX: number;
  uvScaleZ: number;
  planeY: number;
  centerX: number;
  centerZ: number;
  infiniteX: boolean;
  infiniteZ: boolean;
  halfExtentX: number;
  halfExtentZ: number;
}): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    uniforms: {
      uColor: { value: params.color },
      uOpacity: { value: params.opacity },
      uTexture: { value: params.texture },
      uHasTexture: { value: params.texture !== null },
      uUVScale: { value: new THREE.Vector2(params.uvScaleX, params.uvScaleZ) },
      uPlaneY: { value: params.planeY },
      uCenterX: { value: params.centerX },
      uCenterZ: { value: params.centerZ },
      uInfiniteX: { value: params.infiniteX },
      uInfiniteZ: { value: params.infiniteZ },
      uHalfExtentX: { value: params.halfExtentX },
      uHalfExtentZ: { value: params.halfExtentZ },
      uProjInverse: { value: new THREE.Matrix4() },
      uCamWorldMatrix: { value: new THREE.Matrix4() },
    },
    vertexShader: /* glsl */ `
      varying vec2 vNDC;
      void main() {
        vNDC = position.xy;
        gl_Position = vec4(position.xy, 1.0, 1.0);
      }
    `,
    fragmentShader: /* glsl */ `
      precision highp float;
      varying vec2 vNDC;
      uniform vec3 uColor;
      uniform float uOpacity;
      uniform sampler2D uTexture;
      uniform bool uHasTexture;
      uniform vec2 uUVScale;
      uniform float uPlaneY;
      uniform float uCenterX;
      uniform float uCenterZ;
      uniform bool uInfiniteX;
      uniform bool uInfiniteZ;
      uniform float uHalfExtentX;
      uniform float uHalfExtentZ;
      uniform mat4 uProjInverse;
      uniform mat4 uCamWorldMatrix;

      void main() {
        vec4 viewRay = uProjInverse * vec4(vNDC, 1.0, 1.0);
        viewRay /= viewRay.w;
        vec3 worldDir = normalize((uCamWorldMatrix * vec4(viewRay.xyz, 0.0)).xyz);

        float denom = worldDir.y;
        if (abs(denom) < 1e-6) discard;
        float t = (uPlaneY - cameraPosition.y) / denom;
        if (t < 0.0) discard;
        vec3 worldPos = cameraPosition + t * worldDir;

        if (!uInfiniteX && abs(worldPos.x - uCenterX) > uHalfExtentX) discard;
        if (!uInfiniteZ && abs(worldPos.z - uCenterZ) > uHalfExtentZ) discard;

        if (uHasTexture) {
          // MuJoCo GL_OBJECT_PLANE: S = worldPos * 0.5 * texrepeat (infinite)
          // or * 0.5 * texrepeat / halfSize (finite); baked into uUVScale.
          vec2 uv = fract(worldPos.xz * uUVScale);
          gl_FragColor = texture2D(uTexture, uv) * vec4(uColor, uOpacity);
        } else {
          gl_FragColor = vec4(uColor, uOpacity);
        }
      }
    `,
    transparent: params.opacity < 1.0,
    depthTest: true,
    depthWrite: false,
    side: THREE.DoubleSide,
  });
}

function buildMeshGeometry(model: any, meshID: number): THREE.BufferGeometry {
  const geometry = new THREE.BufferGeometry();

  // Copy out of the WASM heap before swizzling: subarray would alias model
  // memory (mutating it breaks rebuilds, and the heap can move on growth).
  const vertexBuffer: Float32Array = model.mesh_vert.slice(
    model.mesh_vertadr[meshID] * 3,
    (model.mesh_vertadr[meshID] + model.mesh_vertnum[meshID]) * 3
  );
  for (let v = 0; v < vertexBuffer.length; v += 3) {
    const temp = vertexBuffer[v + 1];
    vertexBuffer[v + 1] = vertexBuffer[v + 2];
    vertexBuffer[v + 2] = -temp;
  }

  const normalBuffer: Float32Array = model.mesh_normal.slice(
    model.mesh_normaladr[meshID] * 3,
    (model.mesh_normaladr[meshID] + model.mesh_normalnum[meshID]) * 3
  );
  for (let v = 0; v < normalBuffer.length; v += 3) {
    const temp = normalBuffer[v + 1];
    normalBuffer[v + 1] = normalBuffer[v + 2];
    normalBuffer[v + 2] = -temp;
  }

  const uvBuffer: Float32Array = model.mesh_texcoord.slice(
    model.mesh_texcoordadr[meshID] * 2,
    (model.mesh_texcoordadr[meshID] + model.mesh_texcoordnum[meshID]) * 2
  );

  const faceToVertex = model.mesh_face.subarray(
    model.mesh_faceadr[meshID] * 3,
    (model.mesh_faceadr[meshID] + model.mesh_facenum[meshID]) * 3
  );
  const faceToUv = model.mesh_facetexcoord.subarray(
    model.mesh_faceadr[meshID] * 3,
    (model.mesh_faceadr[meshID] + model.mesh_facenum[meshID]) * 3
  );
  const faceToNormal = model.mesh_facenormal.subarray(
    model.mesh_faceadr[meshID] * 3,
    (model.mesh_faceadr[meshID] + model.mesh_facenum[meshID]) * 3
  );

  // MuJoCo stores per-face (vertex, normal, uv) index triples; three.js wants
  // a single index per corner, so deduplicate corners by their triple.
  const positions: number[] = [];
  const normals: number[] = [];
  const uvs: number[] = [];
  const indices: number[] = [];
  const tupleToIndex = new Map<string, number>();

  const faceCount = faceToVertex.length / 3;
  for (let t = 0; t < faceCount; t++) {
    for (let c = 0; c < 3; c++) {
      const vi = faceToVertex[t * 3 + c];
      const nvi = faceToNormal[t * 3 + c];
      const uvi = faceToUv[t * 3 + c];
      const key = `${vi}_${nvi}_${uvi}`;
      let outIndex = tupleToIndex.get(key);
      if (outIndex === undefined) {
        outIndex = positions.length / 3;
        tupleToIndex.set(key, outIndex);
        positions.push(
          vertexBuffer[vi * 3 + 0],
          vertexBuffer[vi * 3 + 1],
          vertexBuffer[vi * 3 + 2]
        );
        normals.push(
          normalBuffer[nvi * 3 + 0],
          normalBuffer[nvi * 3 + 1],
          normalBuffer[nvi * 3 + 2]
        );
        uvs.push(uvBuffer[uvi * 2 + 0] ?? 0, uvBuffer[uvi * 2 + 1] ?? 0);
      }
      indices.push(outIndex);
    }
  }

  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));
  geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  geometry.setIndex(indices);
  return geometry;
}

function createHFieldGeometry(model: any, geomId: number): THREE.BufferGeometry | undefined {
  const hfieldId = model.geom_dataid[geomId];
  if (hfieldId < 0) return undefined;

  const nrow = model.hfield_nrow[hfieldId];
  const ncol = model.hfield_ncol[hfieldId];
  if (nrow < 2 || ncol < 2) return undefined;

  const sx = model.hfield_size[hfieldId * 4 + 0];
  const sy = model.hfield_size[hfieldId * 4 + 1];
  const sz = model.hfield_size[hfieldId * 4 + 2];
  const adr = model.hfield_adr[hfieldId];
  const data = model.hfield_data.subarray(adr, adr + nrow * ncol);

  const positions = new Float32Array(nrow * ncol * 3);
  const uvs = new Float32Array(nrow * ncol * 2);
  const dx = (2 * sx) / (ncol - 1);
  const dy = (2 * sy) / (nrow - 1);

  let vertexOffset = 0;
  let uvOffset = 0;
  for (let r = 0; r < nrow; r++) {
    const yMj = dy * r - sy;
    for (let c = 0; c < ncol; c++) {
      const xMj = dx * c - sx;
      const zMj = data[r * ncol + c] * sz;
      positions[vertexOffset++] = xMj;
      positions[vertexOffset++] = zMj;
      positions[vertexOffset++] = -yMj;
      uvs[uvOffset++] = c / (ncol - 1);
      uvs[uvOffset++] = r / (nrow - 1);
    }
  }

  const indexCount = (nrow - 1) * (ncol - 1) * 6;
  const indices =
    indexCount > 65535 ? new Uint32Array(indexCount) : new Uint16Array(indexCount);
  let indexOffset = 0;
  for (let r = 0; r < nrow - 1; r++) {
    for (let c = 0; c < ncol - 1; c++) {
      const i0 = r * ncol + c;
      const i1 = i0 + 1;
      const i2 = i0 + ncol;
      const i3 = i2 + 1;
      indices[indexOffset++] = i0;
      indices[indexOffset++] = i1;
      indices[indexOffset++] = i3;
      indices[indexOffset++] = i0;
      indices[indexOffset++] = i3;
      indices[indexOffset++] = i2;
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  geometry.setIndex(new THREE.BufferAttribute(indices, 1));
  geometry.computeVertexNormals();
  return geometry;
}

export function buildScene(mujoco: MainModule, model: any): SceneHandles {
  const mj = mujoco as any;
  const textDecoder = new TextDecoder('utf-8');
  const namesArray = new Uint8Array(model.names);

  const root = new THREE.Group();
  root.name = 'MuJoCo Root';

  const bodies = new Map<number, THREE.Group>();
  const meshGeometries = new Map<number, THREE.BufferGeometry>();

  const bodyGroup = (b: number): THREE.Group => {
    let group = bodies.get(b);
    if (!group) {
      group = new THREE.Group();
      const startIdx = model.name_bodyadr[b];
      let endIdx = startIdx;
      while (endIdx < namesArray.length && namesArray[endIdx] !== 0) endIdx++;
      group.name = textDecoder.decode(namesArray.subarray(startIdx, endIdx));
      group.userData.bodyID = b;
      bodies.set(b, group);
    }
    return group;
  };

  for (let g = 0; g < model.ngeom; g++) {
    if (!(model.geom_group[g] < 3)) continue;

    const b = model.geom_bodyid[g];
    const type = Number(model.geom_type[g]);
    const size = [
      model.geom_size[g * 3 + 0],
      model.geom_size[g * 3 + 1],
      model.geom_size[g * 3 + 2],
    ];
    const group = bodyGroup(b);

    const isInfinitePlane =
      type === mj.mjtGeom.mjGEOM_PLANE.value && (size[0] === 0 || size[1] === 0);

    let geometry: THREE.BufferGeometry | undefined;
    switch (type) {
      case mj.mjtGeom.mjGEOM_PLANE.value:
        if (isInfinitePlane) {
          // Full-screen clip-space quad driven by the infinite-plane shader.
          geometry = new THREE.PlaneGeometry(2, 2);
        } else {
          geometry = new THREE.PlaneGeometry(size[0] * 2.0, size[1] * 2.0);
          geometry.rotateX(-Math.PI / 2);
        }
        break;
      case mj.mjtGeom.mjGEOM_HFIELD.value:
        geometry = createHFieldGeometry(model, g);
        break;
      case mj.mjtGeom.mjGEOM_SPHERE.value:
        geometry = new THREE.SphereGeometry(size[0]);
        break;
      case mj.mjtGeom.mjGEOM_CAPSULE.value:
        geometry = new THREE.CapsuleGeometry(size[0], size[1] * 2.0, 20, 20);
        break;
      case mj.mjtGeom.mjGEOM_ELLIPSOID.value:
        geometry = new THREE.SphereGeometry(1);
        break;
      case mj.mjtGeom.mjGEOM_CYLINDER.value:
        geometry = new THREE.CylinderGeometry(size[0], size[0], size[1] * 2.0);
        break;
      case mj.mjtGeom.mjGEOM_BOX.value:
        geometry = new THREE.BoxGeometry(size[0] * 2.0, size[2] * 2.0, size[1] * 2.0);
        break;
      case mj.mjtGeom.mjGEOM_MESH.value: {
        const meshID = model.geom_dataid[g];
        let cached = meshGeometries.get(meshID);
        if (!cached) {
          cached = buildMeshGeometry(model, meshID);
          meshGeometries.set(meshID, cached);
        }
        geometry = cached;
        break;
      }
      default:
        break;
    }

    if (!geometry) {
      console.warn(`Skipping geom ${g} (type ${type}): unsupported geometry`);
      continue;
    }

    let color = [
      model.geom_rgba[g * 4 + 0],
      model.geom_rgba[g * 4 + 1],
      model.geom_rgba[g * 4 + 2],
      model.geom_rgba[g * 4 + 3],
    ];
    const matId = model.geom_matid[g];
    let texture: THREE.Texture | null = null;
    if (matId !== -1) {
      color = [
        model.mat_rgba[matId * 4 + 0],
        model.mat_rgba[matId * 4 + 1],
        model.mat_rgba[matId * 4 + 2],
        model.mat_rgba[matId * 4 + 3],
      ];
      const role = mj.mjtTextureRole.mjTEXROLE_RGB.value;
      const texId = model.mat_texid[matId * mj.mjtTextureRole.mjNTEXROLE.value + role];
      if (texId !== -1) {
        texture = createTexture(mujoco, model, texId);
        if (texture) {
          texture.repeat.set(
            model.mat_texrepeat?.[matId * 2 + 0] ?? 1,
            model.mat_texrepeat?.[matId * 2 + 1] ?? 1
          );
          texture.wrapS = THREE.RepeatWrapping;
          texture.wrapT = THREE.RepeatWrapping;
        }
      }
    }

    let material: THREE.Material | THREE.Material[] = new THREE.MeshPhysicalMaterial({
      color: new THREE.Color(color[0], color[1], color[2]),
      transparent: color[3] < 1.0,
      opacity: color[3],
      ...reflectanceParams(model, matId),
    });

    if (texture) {
      if (!(texture instanceof THREE.CubeTexture)) {
        (material as THREE.MeshPhysicalMaterial).map = texture;
      } else if (
        type === mj.mjtGeom.mjGEOM_BOX.value &&
        Array.isArray(texture.image) &&
        texture.image.length === 6
      ) {
        // Cubemap-textured box: one material per face from the cube faces.
        const images = texture.image as unknown as HTMLCanvasElement[];
        if (geometry.groups && geometry.groups.length !== 6) geometry.clearGroups();
        material = images.map((canvas) => {
          const faceTex = new THREE.CanvasTexture(canvas);
          faceTex.flipY = false;
          faceTex.wrapS = THREE.ClampToEdgeWrapping;
          faceTex.wrapT = THREE.ClampToEdgeWrapping;
          faceTex.colorSpace = THREE.SRGBColorSpace;
          return new THREE.MeshPhysicalMaterial({
            color: new THREE.Color(color[0], color[1], color[2]),
            transparent: color[3] < 1.0,
            opacity: color[3],
            ...reflectanceParams(model, matId),
            map: faceTex,
          });
        });
      } else {
        const m = material as THREE.MeshPhysicalMaterial;
        m.envMap = texture;
        m.envMapIntensity = matId !== -1 ? model.mat_reflectance?.[matId] || 0.5 : 0.5;
      }
    }

    const mesh = new THREE.Mesh(geometry, material as THREE.Material);
    mesh.castShadow = type !== mj.mjtGeom.mjGEOM_PLANE.value;
    mesh.receiveShadow = true;
    mesh.userData.bodyID = b;
    group.add(mesh);

    getPosition(model.geom_pos, g, mesh.position);
    if (!isInfinitePlane) {
      getQuaternion(model.geom_quat, g, mesh.quaternion);
    }
    if (type === mj.mjtGeom.mjGEOM_ELLIPSOID.value) {
      mesh.scale.set(size[0], size[2], size[1]);
    }

    if (isInfinitePlane) {
      const baseMat = (
        Array.isArray(material) ? material[0] : material
      ) as THREE.MeshPhysicalMaterial;
      const tex = baseMat.map ?? null;
      const repeatX = tex?.repeat.x ?? 1;
      const repeatY = tex?.repeat.y ?? 1;
      const uvScaleX = size[0] > 0 ? (repeatX * 0.5) / size[0] : repeatX * 0.5;
      const uvScaleZ = size[1] > 0 ? (repeatY * 0.5) / size[1] : repeatY * 0.5;
      const shaderMat = createInfinitePlaneShaderMaterial({
        color: baseMat.color.clone(),
        opacity: baseMat.opacity,
        texture: tex,
        uvScaleX,
        uvScaleZ,
        planeY: mesh.position.y,
        centerX: mesh.position.x,
        centerZ: mesh.position.z,
        infiniteX: size[0] === 0,
        infiniteZ: size[1] === 0,
        halfExtentX: size[0],
        halfExtentZ: size[1],
      });
      mesh.material = shaderMat;
      mesh.castShadow = false;
      mesh.receiveShadow = false;
      mesh.frustumCulled = false;
      mesh.renderOrder = 1;
      mesh.onBeforeRender = (_renderer, _scene, camera) => {
        shaderMat.uniforms.uProjInverse.value.copy(camera.projectionMatrixInverse);
        shaderMat.uniforms.uCamWorldMatrix.value.copy(camera.matrixWorld);
      };
    }
  }

  const lights = createLights(mujoco, model, root);

  // Every body gets a group (even geomless ones) so pose sync can be a flat
  // loop; world poses go directly onto each group, so the graph stays flat.
  for (let b = 0; b < model.nbody; b++) {
    root.add(bodyGroup(b));
  }

  const skybox = createSkyboxTexture(mujoco, model);
  return { root, bodies, lights, skybox };
}

// Copies simulated world poses (mjData.xpos/xquat) onto the body groups.
export function syncBodyPoses(data: any, bodies: Map<number, THREE.Group>): void {
  for (const [b, group] of bodies) {
    if (b === 0) continue; // world body never moves
    getPosition(data.xpos, b, group.position);
    getQuaternion(data.xquat, b, group.quaternion);
  }
}

// Frees GPU resources of a built scene (model hot-swap).
export function disposeScene(handles: SceneHandles): void {
  handles.root.traverse((obj) => {
    const mesh = obj as THREE.Mesh;
    if (mesh.isMesh) {
      mesh.geometry?.dispose();
      const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
      for (const m of mats) {
        for (const value of Object.values(m)) {
          if (value instanceof THREE.Texture) value.dispose();
        }
        m.dispose();
      }
    }
  });
  handles.skybox?.dispose();
  handles.root.removeFromParent();
}
