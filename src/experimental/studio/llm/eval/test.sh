#!/usr/bin/env bash
# Eval harness for the MuJoCo Studio LLM agent.
#
# Prompts live in per-model subfolders: eval/<model>/NNN_input.md, where <model>
# is the model id with '-' written as '_' (e.g. claude_sonnet_4_6 maps to the
# model claude-sonnet-4-6). For each input the harness launches Studio headless,
# has THAT model's agent carry out the one prompt, and writes NNN_output.gif
# next to it. Review the GIFs by eye to judge whether a system-prompt change
# helped or hurt (the agent is non-deterministic, so treat them as samples, not
# pass/fail).
#
# COST: every input makes real Anthropic API calls (billed to the key file),
# typically a few tool-use round-trips each.
#
# Usage:
#   bash test.sh                          # every model folder, every input
#   bash test.sh claude_sonnet_4_6        # one model folder, all its inputs
#   bash test.sh claude_sonnet_4_6 000    # that folder, just input 000
#
# Knobs (env vars): MODEL_FILE, WIDTH, HEIGHT, FRAMES, STRIDE, DELAY.
#   FRAMES caps the capture length; bump it if a complex prompt gets cut off.
#   STRIDE keeps every Nth frame in the GIF; DELAY is the GIF frame delay (1/100s).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../../.." && pwd)"   # repo root, e.g. /mnt/c/Dev/mujoco
EXE_DIR="$ROOT/build/bin"                    # run from here so assets/ resolve
EXE="./Debug/mujoco_studio.exe"

MODEL_FILE="${MODEL_FILE:-C:/Dev/mujoco/model/humanoid/humanoid.xml}"
WIDTH="${WIDTH:-640}"; HEIGHT="${HEIGHT:-400}"
FRAMES="${FRAMES:-1500}"; STRIDE="${STRIDE:-5}"; DELAY="${DELAY:-7}"

# The API key lives one directory above the repo (see project setup).
KEYFILE="$(dirname "$ROOT")/anthropic.txt"
if [[ -f "$KEYFILE" ]]; then
  export ANTHROPIC_API_KEY="$(cat "$KEYFILE")"
  export WSLENV="ANTHROPIC_API_KEY:${WSLENV:-}"
else
  echo "warning: $KEYFILE not found; the agent falls back to the mock provider." >&2
fi

if [[ ! -f "$EXE_DIR/Debug/mujoco_studio.exe" ]]; then
  echo "error: $EXE_DIR/Debug/mujoco_studio.exe not found -- build studio first." >&2
  exit 1
fi
command -v convert >/dev/null || { echo "error: ImageMagick 'convert' not found." >&2; exit 1; }

# Optional first arg: a model folder to limit to. Remaining args: input ids.
models=()
if (( $# )) && [[ -d "$HERE/$1" ]]; then
  models=("$1"); shift
fi
ids=("$@")
if (( ${#models[@]} == 0 )); then
  for d in "$HERE"/*/; do
    [[ -d "$d" ]] && models+=("$(basename "$d")")
  done
fi

cd "$EXE_DIR"
for model_dir in "${models[@]}"; do
  dir="$HERE/$model_dir"
  model_id="${model_dir//_/-}"          # claude_sonnet_4_6 -> claude-sonnet-4-6

  # Which inputs: explicit ids, else every *_input.md in this model folder.
  these=()
  if (( ${#ids[@]} )); then
    these=("${ids[@]}")
  else
    for pf in "$dir"/*_input.md; do
      [[ -e "$pf" ]] && these+=("$(basename "$pf" _input.md)")
    done
  fi

  for id in "${these[@]}"; do
    pf="$dir/${id}_input.md"
    [[ -f "$pf" ]] || { echo "skip $model_dir/$id (no $pf)"; continue; }
    prompt="$(tr -d '\r\n' < "$pf")"
    frames="$dir/.frames_${id}"
    rm -rf "$frames"; mkdir -p "$frames"
    echo "[$model_dir/$id] ($model_id) $prompt"
    timeout 240 "$EXE" \
      --model_file="$MODEL_FILE" \
      --capture_gif="$(wslpath -m "$frames")" \
      --capture_script=llm \
      --capture_prompt="$prompt" \
      --capture_model="$model_id" \
      --capture_frames="$FRAMES" \
      --window_width="$WIDTH" --window_height="$HEIGHT" \
      >/dev/null 2>&1

    mapfile -t ppms < <(ls "$frames"/frame_*.ppm 2>/dev/null | awk "NR % $STRIDE == 1")
    if (( ${#ppms[@]} )); then
      convert -delay "$DELAY" -loop 0 "${ppms[@]}" -layers Optimize \
              "$dir/${id}_output.gif" 2>/dev/null \
        && echo "  -> $model_dir/${id}_output.gif (${#ppms[@]} frames)" \
        || echo "  ! convert failed for $model_dir/$id"
    else
      echo "  ! no frames captured for $model_dir/$id (agent error? bump FRAMES / check the key)"
    fi
    rm -rf "$frames"
  done
done
