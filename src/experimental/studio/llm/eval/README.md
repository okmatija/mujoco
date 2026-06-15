# Agent eval set

A small, eyeball-able eval for the Studio LLM agent and its system prompt
(`../system_prompt.md`). Each `NNN_prompt.md` is one user request; `test.sh`
drives the agent through it headless and writes `NNN_output.gif` next to it.

Use it to sanity-check a system-prompt change: run before and after, then watch
the GIFs to see whether the agent still reaches the right panel/widget and does
the task. The agent is non-deterministic, so these are samples, not pass/fail
assertions.

## Run

```bash
bash test.sh                 # all prompts, default model (sonnet)
MODEL=haiku bash test.sh     # cheapest
MODEL=opus  bash test.sh     # strongest
bash test.sh 003 007         # just these ids
```

Requires: a built `mujoco_studio.exe`, the `anthropic.txt` key file (one dir
above the repo), and ImageMagick (`convert`). Knobs: `MODEL`, `MODEL_FILE`,
`WIDTH`, `HEIGHT`, `FRAMES`, `STRIDE`, `DELAY` (see the script header).

**Each run makes real Anthropic API calls** (a few tool-use round-trips per
prompt), billed to the key file.

## The prompts

| id  | request                                   | exercises                          |
|-----|-------------------------------------------|------------------------------------|
| 000 | Turn on contact force visualization       | Rendering flag toggle              |
| 001 | Switch the renderer to wireframe          | Rendering flag toggle              |
| 002 | Enable shadows                            | Rendering flag toggle              |
| 003 | Open the Joints panel                     | rail panel toggle                  |
| 004 | Pause the simulation                      | transport (Space)                  |
| 005 | Step forward 3 frames                     | history scrubber (behind toggle)   |
| 006 | Reset the simulation                      | one-shot rail button               |
| 007 | Pin the sim stats in the status bar       | the relocated `###Stats` toggle    |
| 008 | Launch a box with the Object Launcher     | plugin window + button             |
| 009 | Show contact points                       | Model Elements flag                |

Add a case by dropping in the next `NNN_prompt.md`.

Generated `*_output.gif` files are git-ignored.
