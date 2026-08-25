# Roadmap: observations and proposals

Notes accumulated while writing this tutorial series. The API is young and
these are proposals, not complaints — most exist precisely because the
current design is clean enough that its remaining rough edges stand out.
Ordered roughly by expected impact.

## Naming

"Message passing API" undersells the design's actual center of gravity,
which is the **two delivery semantics**. Suggestion: name the framework
after them — "Events & Snapshots" — and reserve "channel"/"endpoint" for
the plumbing docs. In the same spirit, `PerturbEvent` deserves a rename:
it carries an arbitrary state signature applied via generic `mj_setState`
(tutorial 40 forwards mocap poses with it; perturbation forces are just
its most common payload). `SetStateEvent` says what it does.

## Correctness and robustness

- **Mocap perturbation does not cross to the sim.** Ctrl+dragging a mocap
  body moves only the viewer's copy; the next `StateSnapshot` snaps it
  back. `viewer_utils.apply_perturb` forwards only `XFRC_APPLIED`.
  Tutorial 40 works around this in ~20 lines; the same lines arguably
  belong in `apply_perturb` itself (send `MOCAP_POS|MOCAP_QUAT` when the
  selected body is mocap). Mocap dragging is a workhorse interaction in
  robotics models, and its current silent failure is puzzling to users.

- **A faulty handler kills the loop.** `HandlerRegistry.dispatch` has no
  error isolation: one exception in a user handler unwinds the viewer
  loop (or `handle.sync`). For a tool platform this is harsh — mjlab's
  viewer catches per-step errors, pauses, and surfaces the traceback in
  the UI instead of dying. Proposal: catch per-handler, log with the
  handler's name, optionally auto-disable a repeatedly-failing handler;
  an opt-out flag for people who want a hard crash while debugging.

- **The viewer is authoritative over `mjOption`, silently.** `ViewerApp`
  publishes `MjOptionSnapshot` from its copy every frame, and the built-in
  sim handler overwrites the true model's `opt` wholesale. Any sim-side
  programmatic change to `opt` (curriculum gravity schedules, DR of solver
  settings) is reverted within a frame — the losing side has no way to
  even detect the fight (tutorial 04 had to design around this). Proposal:
  send deltas only when the user edits a field, or make the sim
  authoritative and treat viewer edits as request events, matching the
  direction of authority everywhere else in the framework.

- **`ViewerApp` calls a binding that does not exist.** `viewer_app.py`'s
  Stats window calls `ux.stats_gui(...)`, but `ux.cc` binds only
  `info_gui` — toggling Charts → Stats raises `AttributeError`. One-line
  fix either way (rename the call or the binding); mostly evidence for
  the missing-tests point below.

- **Undefined ordering within a priority level.** Documented as undefined
  in `Priority`, but "undefined" here means `dir()` order, i.e. stable
  until an unrelated rename reorders dispatch. Cheap fix: tiebreak by
  registration order (position in the `handlers=` list, then method
  definition order) and promise it.

- **Both endpoints claim to close all four channels.** `ViewerEndpoint.close`
  and `SimEndpoint.close` each close everything; harmless today because
  passive channels' `close()` is a no-op, but the ownership story will
  matter the moment a channel holds a real resource (socket, shared
  memory). Proposal: each endpoint closes only its outgoing channels
  (half-close), or a single owner object closes the pair.

## API ergonomics

- **Expose reusable channel implementations.** `_PassiveSnapshotChannel` /
  `_PassiveEventChannel` are private to `launch_passive`, so anyone wiring
  a non-viewer hop (tutorial 31's mocap bridge, any hardware ingest)
  hand-rolls the same mailbox. Proposal: a public `channels` module with
  the thread-safe implementations, ready for `make_endpoints`.

- **Third parties as first-class message sources.** The topology is
  hardwired to exactly two endpoints. The sim-to-real use cases (30, 31)
  want a third participant — a hardware bridge that publishes snapshots
  and receives events (e-stop). Today it must either borrow the sim's
  endpoint from the sim thread's context or bypass channels entirely.
  Proposal worth sketching: named endpoint pairs on a small bus, or
  simply an officially-supported "injector" handle (thread-safe `put`
  into a side's incoming channels — `send_to_viewer` already is this in
  practice for the sim process; bless and document it).

- **Per-type snapshot slots collide across sources.** Latest-wins is keyed
  by Python type, so two producers of one snapshot type (left arm / right
  arm drivers, tutorial 30's multi-source extension) overwrite each other.
  ROS solves this with topic names. Proposal: optional key on `put`
  (default `type(msg)`, opt-in `(type, key)`), keeping the common case
  untyped and simple.

- **`StudioApp` vs `ViewerApp`.** `studio_app.py` (single-threaded, owns
  model+data+UI) and `viewer_app.py` (messaging-era, handler-based)
  duplicate the entire Studio GUI (~200 lines each of near-identical
  `build_gui`). No sample uses `StudioApp` anymore. If the single-threaded
  mode should live on, extract the shared GUI into one place; otherwise
  delete `StudioApp` before external code grows roots into it. Relatedly,
  the "Legacy message types" (`SimToView`/`ViewToSim`) in
  `viewer_protocol.py` are already marked for removal.

- **Lifecycle events are messages that never cross a channel.**
  `UpdateEvent`/`BuildGuiEvent`/`ViewerInitEvent` are dispatched locally,
  yet derive from `Event`, whose docstring promises channel semantics
  ("queued in order... delivered exactly once"). Works fine; reads oddly
  the first time (tutorial 05 has to explain it away). A `LifecycleEvent`
  base class — or one docstring sentence in `Event` — would settle it.
  While in there: the message catalog is scattered across `messages.py`,
  `viewer_protocol.py` and `viewer_app.py`; one module (or one doc page)
  listing every built-in message with its direction and semantics would
  be the single highest-value piece of reference documentation.

## Performance (none of these are urgent)

- `HandlerRegistry.dispatch` walks the MRO, concatenates, and sorts on
  every message. A per-type cache of the sorted handler chain would make
  dispatch O(handlers) after first sight of each type — relevant once
  telemetry streams (tutorial 30) push thousands of messages per second.
- `ViewerHandle.sync` allocates and fills a full `INTEGRATION` state
  vector every loop iteration regardless of rate. Buffer reuse plus an
  optional send-rate cap would cut the steady-state cost.
- `ViewerApp` deep-copies `model.opt` into a snapshot every frame (see
  also the authority question above — fixing that fixes this).

## Missing tests

`messages.py` and `handler_registry.py` are pure Python with crisp
contracts (annotation-based subscription, priority order, consumption,
MRO dispatch, string-annotation resolution) and no tests here. These are
the easiest high-value tests in the codebase, and they'd pin down the
"undefined ordering" question above as a side effect.

## Future how-tos

Candidates that earned their place while researching what people build
with viewers (mjviser, mjlab, mjlab playground), roughly in order:

1. **Record & scrub** — ring buffer of `qpos/qvel`, timeline slider,
   play/pause/loop; replaying through `mj_forward` so contacts render.
2. **Reset roulette / env authoring** — a "drop from random pose" button
   plus uprightness plot: the getup-task authoring loop in miniature.
3. **Domain-randomization preview** — randomize friction/mass/colors,
   preview N samples; answers "did my DR actually apply?".
4. **Multi-source telemetry with health** — two partial joint streams
   merged by name, per-source liveness dots (extends 30).
5. **Custom channels** — the same tool over a multiprocessing or socket
   channel pair; the seed of remote/web viewing.
6. **Policy-in-the-loop** — an RL policy driving `ctrl` in the sim loop,
   with a checkpoint hot-reload button (the mjlab workflow).
