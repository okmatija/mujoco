#!/usr/bin/env bash
# Eval harness for the MuJoCo Studio LLM agent.
#
# For every NNN_prompt.md in this folder it launches Studio headless, has the
# agent carry out that one prompt, and writes NNN_output.gif next to it. Review
# the GIFs by eye to judge whether a system-prompt change helped or hurt (the
# agent is non-deterministic, so treat them as samples, not pass/fail).
#
# COST: every prompt makes real Anthropic API calls (billed to the key file),
# typically a few tool-use round-trips each -- so a full run is ~10 short
# agentic conversations. Set MODEL=haiku for the cheapest runs.
#
# Usage:
#   bash test.sh                 # all prompts
#   MODEL=opus bash test.sh      # pin a model (default: sonnet)
#   bash test.sh 003 007         # only these ids
#
# Knobs (env vars): MODEL, MODEL_FILE, WIDTH, HEIGHT, FRAMES, STRIDE, DELAY.
#   FRAMES caps the capture length; bump it if a complex prompt gets cut off.
#   STRIDE keeps every Nth frame in the GIF; DELAY is the GIF frame delay (1/100s).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../../.." && pwd)"   # repo root, e.g. /mnt/c/Dev/mujoco
EXE_DIR="$ROOT/build/bin"                    # run from here so assets/ resolve
EXE="./Debug/mujoco_studio.exe"

MODEL="${MODEL:-sonnet}"
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

# Which ids: explicit args, else every *_prompt.md.
ids=()
if (( $# )); then
  ids=("$@")
else
  for pf in "$HERE"/*_prompt.md; do
    [[ -e "$pf" ]] && ids+=("$(basename "$pf" _prompt.md)")
  done
fi

cd "$EXE_DIR"
for id in "${ids[@]}"; do
  pf="$HERE/${id}_prompt.md"
  [[ -f "$pf" ]] || { echo "skip $id (no $pf)"; continue; }
  prompt="$(tr -d '\r\n' < "$pf")"
  frames="$HERE/.frames_${id}"
  rm -rf "$frames"; mkdir -p "$frames"
  echo "[$id] ($MODEL) $prompt"
  timeout 240 "$EXE" \
    --model_file="$MODEL_FILE" \
    --capture_gif="$(wslpath -m "$frames")" \
    --capture_script=llm \
    --capture_prompt="$prompt" \
    --capture_model="$MODEL" \
    --capture_frames="$FRAMES" \
    --window_width="$WIDTH" --window_height="$HEIGHT" \
    >/dev/null 2>&1

  mapfile -t ppms < <(ls "$frames"/frame_*.ppm 2>/dev/null | awk "NR % $STRIDE == 1")
  if (( ${#ppms[@]} )); then
    convert -delay "$DELAY" -loop 0 "${ppms[@]}" -layers Optimize \
            "$HERE/${id}_output.gif" 2>/dev/null \
      && echo "  -> ${id}_output.gif (${#ppms[@]} frames)" \
      || echo "  ! convert failed for $id"
  else
    echo "  ! no frames captured for $id (agent error? bump FRAMES / check the key)"
  fi
  rm -rf "$frames"
done
