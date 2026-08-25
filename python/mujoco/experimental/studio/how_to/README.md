# MuJoCo Studio: how to build simulation and viewer tools

These are sample programs that show how Studio's messaging API works. They
are meant to be read and run in numerical order, starting from
`01_render_image.py` and going upward. Concepts are explained in comments,
in the file where they first appear, and later files assume the earlier
ones. Gaps in the numbering are deliberate: they group related topics and
leave room for insertions.

Each file is standalone and runnable:

```sh
python 02_launch_studio.py                # self-contained files
python 10_ghost_overlay.py humanoid.xml   # files that view a model of yours
```

## What the API is, in one paragraph

A Studio application has two sides: the **sim side** (your loop, which owns
the physics) and the **viewer side** (a render loop on another thread, which
owns the window, GUI, and its own copy of the model). The sides communicate
only by exchanging typed, immutable **messages** over channels. There are
two delivery semantics — **Events** are reliable, ordered and never dropped
(discrete actions: reset, model change, button click); **Snapshots** are
latest-wins (continuous state: poses, sliders, telemetry) — and choosing
between them is the central design act when building a tool. Code receives
messages by declaring **handlers**: methods decorated with
`@messages.handler`, subscribed by their parameter's type annotation,
ordered by priority, dispatched on the receiving side's thread. The entire
Studio GUI (`ViewerApp`) is itself just a handler, which is why your tools
can extend, reorder around, or replace it.

## The files

| File | What it teaches |
| --- | --- |
| `01_render_image.py` | The lowest-level building blocks: `parser` and `renderer`, no viewer at all. |
| `02_launch_studio.py` | The canonical tool skeleton: `launch_passive` + `ViewerApp` + your sim loop; the `handle.sync()` contract. |
| `03_send_to_viewer.py` | Defining messages; Event vs Snapshot semantics made visible; your first handler and GUI window. |
| `04_send_to_sim.py` | The reverse direction: GUI → sim; sim-side mailbox handlers; the `ViewerInitEvent` caching idiom. |
| `05_handlers.py` | Dispatch rules: priorities, consumption, base-class subscription; running a bare viewer without `ViewerApp`. |
| `10_ghost_overlay.py` | Overlay drawing with `extra_geoms`: a time-delayed ghost. |
| `11_live_plots.py` | Live ImPlot charts; picking; responsive immediate-mode layout. |
| `12_debug_markers.py` | The debug-drawing vocabulary: arrows, frame triads, spheres from live sim quantities. |
| `20_sim_loop_control.py` | Running your own controller sim-side; surviving model swaps; pause/speed correctness. |
| `30_robot_telemetry.py` | Sim-to-real: displaying a robot hardware stream as a ghost, with timestamps and staleness. |
| `31_mocap_stream.py` | Driving the sim from an external pose stream; building latest-wins mailboxes for hops the framework doesn't cover. |
| `40_mocap_target_tool.py` | A pose-authoring tool: forwarding mocap drags to the sim with state signatures; dispatch-order control. |
| `99_roadmap.md` | Observations on the current architecture and proposed improvements. |

## Suggested background

The tutorials assume basic familiarity with MuJoCo's Python bindings
(`MjModel`, `MjData`, `mj_step`) and no prior knowledge of Studio, ImGui,
or the messaging API. Tutorials 03+ use [Dear ImGui](https://github.com/ocornut/imgui)
through the bundled `dear_imgui` bindings; immediate-mode GUI is easy to
pick up from the examples themselves.
