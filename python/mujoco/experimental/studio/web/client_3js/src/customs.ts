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

// Lookup of mjModel custom elements (<custom> text/numeric) by name. Used to
// pull viewer-side assets that ride inside the model, e.g. the embedded
// gaussian splat (see splat.ts and the Python-side studio splat helper).

const textDecoder = new TextDecoder('utf-8');

// Reads the 0-terminated string at `adr` in the model's names blob.
function nameAt(names: Uint8Array, adr: number): string {
  let end = adr;
  while (end < names.length && names[end] !== 0) end++;
  return textDecoder.decode(names.subarray(adr, end));
}

// mjModel fields used here; the binding exposes them as typed-array views.
interface ModelCustoms {
  names: ArrayLike<number>;
  ntext: number;
  name_textadr: ArrayLike<number>;
  text_adr: ArrayLike<number>;
  text_size: ArrayLike<number>;
  text_data: ArrayLike<number>;
  nnumeric: number;
  name_numericadr: ArrayLike<number>;
  numeric_adr: ArrayLike<number>;
  numeric_size: ArrayLike<number>;
  numeric_data: ArrayLike<number>;
}

// Returns the text custom named `name` (without its trailing NUL), or null.
export function findText(model: ModelCustoms, name: string): string | null {
  const names = new Uint8Array(model.names as Uint8Array);
  for (let i = 0; i < model.ntext; i++) {
    if (nameAt(names, model.name_textadr[i]) !== name) continue;
    const adr = model.text_adr[i];
    const size = model.text_size[i]; // strlen + 1
    const bytes = new Uint8Array(size - 1);
    for (let j = 0; j < size - 1; j++) bytes[j] = model.text_data[adr + j];
    return textDecoder.decode(bytes);
  }
  return null;
}

// Returns the numeric custom named `name` as a plain array, or null.
export function findNumeric(model: ModelCustoms, name: string): number[] | null {
  const names = new Uint8Array(model.names as Uint8Array);
  for (let i = 0; i < model.nnumeric; i++) {
    if (nameAt(names, model.name_numericadr[i]) !== name) continue;
    const adr = model.numeric_adr[i];
    const size = model.numeric_size[i];
    const out = new Array<number>(size);
    for (let j = 0; j < size; j++) out[j] = model.numeric_data[adr + j];
    return out;
  }
  return null;
}

// Decodes a base64 string into bytes without blowing the stack on large
// inputs (atob handles the size fine; the copy loop is chunk-free).
export function decodeBase64(b64: string): Uint8Array {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}
