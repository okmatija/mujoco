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

// MuJoCo (z-up, wxyz quats) <-> three.js (y-up, xyzw quats) conversions.
// Adapted from mjswan (https://github.com/ttktjmt/mjswan), Apache-2.0.

import * as THREE from 'three';

export function mjcToThreeCoordinate(v: ArrayLike<number>): THREE.Vector3 {
  return new THREE.Vector3(v[0], v[2], -v[1]);
}

export function threeToMjcCoordinate(v: THREE.Vector3): THREE.Vector3 {
  return new THREE.Vector3(v.x, -v.z, v.y);
}

// Reads entry `index` of an x/y/z array in MuJoCo's frame into a three.js
// Vector3 (y-up).
export function getPosition(
  buffer: ArrayLike<number>,
  index: number,
  target: THREE.Vector3
): THREE.Vector3 {
  return target.set(
    buffer[index * 3 + 0],
    buffer[index * 3 + 2],
    -buffer[index * 3 + 1]
  );
}

// Reads entry `index` of a wxyz quaternion array in MuJoCo's frame into a
// three.js Quaternion (xyzw, y-up).
export function getQuaternion(
  buffer: ArrayLike<number>,
  index: number,
  target: THREE.Quaternion
): THREE.Quaternion {
  return target.set(
    -buffer[index * 4 + 1],
    -buffer[index * 4 + 3],
    buffer[index * 4 + 2],
    -buffer[index * 4 + 0]
  );
}
