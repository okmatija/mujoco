.. _StArchitecture:

Architecture
============

.. WARNING:: MuJoCo Studio is currently experimental and subject to frequent change.

- Page scope: the component library, how applications are assembled from it, the simulation/viewer separation and
  message-passing design, threading models. For *using* the Python framework, see :doc:`python`.

.. _StDesign:

Design goals
------------

- Granular components, very little application glue; an "application" = thin assembly + frame loop.
- The numbers that make the point:

  - Desktop: ``App`` + ~40-line ``main.cc`` + ~80-line launcher; core is
    ``while (app.Update()) { app.BuildGui(); app.Render(); }``.
  - Studio Live: same ``App``; ``emscripten.cc`` exposes ``init()`` / ``renderFrame()`` to JS; mostly resource
    providers, not app logic.
  - Python: does not use ``App`` — re-assembles the same components via bindings; ``ViewerApp`` plays the role of
    ``App``.

- Ongoing goal: keep migrating functionality out of the applications into components.

.. _StPlatform:

The platform library
--------------------

- ``src/experimental/platform``, CMake target ``mujoco::platform``; three layers:

- **Hardware abstraction (**\ ``hal/``\ **)** — ``GraphicsMode``, the central switch: renderer (classic/Filament) ×
  API (OpenGL/Vulkan/WebGL) × regime (windowed/headless/software); components branch at runtime, no subclasses.
  ``Window``: SDL2-based, one class covering native window, headless-to-texture, and Emscripten/WebGL regimes.
  ``Renderer``: classic ``mjr`` path or Filament path rendering physics scene + UI scene in one call;
  ``ImguiBridge`` turns ImGui draw data into Filament renderables → no GL UI backend needed in the browser;
  offscreen/texture targets (picture-in-picture, offscreen rendering).

- **Simulation services (**\ ``sim/``\ **)** — depend only on MuJoCo. ``StepControl``: real-time pacing
  (``Advance`` → OK/PAUSED/AUTO_RESET/DIVERGED), speed, single-step, noise, pre/post-step callbacks — the one sim
  component shared by every Studio application. ``ModelHolder``: owns model + data (+ spec, VFS), from
  file/buffer/spec, error reporting. ``SimHistory``: bounded state history for the timeline; scrub-then-resume
  branches history. ``SimProfiler``: per-step timings and solver diagnostics, plotted with ImPlot.

- **User experience (**\ ``ux/``\ **)** — stateless panel functions; GUI state lives in the caller's MuJoCo structs
  → reusable from any app, C++ or Python. ``gui.h``: the panel library (physics, rendering, groups, joints,
  controls, sensors, state, watch, noise, convergence, profiler; themes; docking layout). ``gui_spec.h`` /
  ``spec_editor.h``: :ref:`mjSpec` tree view + property editors, compile, undo/redo. ``interaction.h``: camera,
  picking, perturbation; no ImGui dependency; wraps the :ref:`abstract visualization<Abstract>` API. ``fonts.h``:
  shared font set, loader injected as a function (disk on desktop, HTTP bytes in the browser). ``file_dialog.h``:
  native dialogs per OS, pure-ImGui fallback in the browser. ``plugin.h``: C++ plugin registry, see
  :ref:`below<StCustomization>`.

- Component availability (unbound components are expected to become available to Python over time):

.. list-table::
   :header-rows: 1

   * - Component
     - C++ applications
     - Python framework
   * - ``StepControl``
     - yes
     - yes (``sim`` module)
   * - ``ux`` panels, interaction, themes
     - yes
     - yes (``ux`` module)
   * - ``Window`` / ``Renderer``
     - yes
     - yes (used internally by the viewers)
   * - ``ModelHolder``
     - yes
     - internal (parsing is exposed via the ``parser`` module)
   * - ``SimHistory`` (timeline), ``SimProfiler``
     - yes
     - not yet
   * - ``SpecEditor`` / spec editing panels
     - yes
     - not yet
   * - Messages, handlers
     - not yet
     - yes

.. _StAssemblies:

Applications as assemblies
--------------------------

.. list-table::
   :header-rows: 1

   * - Application
     - Assembly
   * - Studio App (C++)
     - ``App`` composes ``Window``, ``Renderer``, ``ModelHolder``, ``StepControl``, ``SimHistory``, ``SimProfiler``,
       ``SpecEditor`` and the ``ux`` panels into the full-featured application, driven by a
       ``while (Update) { BuildGui; Render; }`` loop. Runs natively, or in the browser as Studio Live — the same
       ``App`` in the ``FilamentWebGl`` graphics mode, driven by ``requestAnimationFrame``, with file and network
       access through MuJoCo resource providers.
   * - Studio App (Python)
     - ``ViewerApp`` re-assembles the panels and interaction handling from the ``ux`` bindings, with
       ``Window``/``Renderer`` wrapped by the ``native_viewer`` and ``StepControl`` driving the user's own
       simulation loop. Sim and viewer run on separate threads connected by :ref:`messages<StMessages>`. With the
       ``web_viewer``, the UI is drawn headless, streamed to the browser, and the scene is rendered there by a small
       WebAssembly client itself assembled from ``Window``, ``Renderer``, ``ModelHolder`` and ``interaction``.
   * - Custom Python apps
     - User assemblies, easily composed from the same components: your own handlers and messages, panels from
       ``ux``, with or without ``ViewerApp`` — see :doc:`python`.

- The frame contract is the same everywhere: *update* → *build GUI* → *render*
  (``App::Update/BuildGui/Render`` ↔ viewer loop dispatching ``UpdateEvent``/``BuildGuiEvent`` ↔ ``renderFrame()``).

.. _StSeparation:

Simulation/viewer separation
----------------------------

- Two sides: the *sim side* (owns the physics — in Python, your loop) and the *viewer side* (owns window, GUI, and
  its own copy of the model).
- No shared mutable state: the viewer deep-copies the model; physics state crosses as immutable snapshots; there
  are no locks.
- What the separation buys: the sides can run in different threads or processes, and the viewer can be native or in
  the browser — the viewer is a *mirror* of the simulation, not a participant in it.
- Per-frame rendezvous: the sim side publishes state (``handle.sync``); the viewer consumes the latest state at its
  own rate.
- Status today: fully realized in the Python framework; the C++ Studio App still runs sim and viewer on one thread
  (see :ref:`Threading models<StThreading>`).

.. _StMessages:

Events and snapshots
--------------------

- Communication across the separation is by typed, immutable messages; the design is language-neutral — currently
  implemented in Python (:ref:`API<StPyMessages>`), may move into C++ later to be shared by all Studio
  applications.
- Two delivery semantics — the central design decision per message:

  - **Events**: reliable, ordered, never dropped. For discrete actions (reset, model change, button press).
  - **Snapshots**: latest-wins; intermediate values may be dropped. For continuously refreshed state (poses,
    sliders, telemetry); lets a fast physics loop feed a slower render loop without unbounded queues.

- Plumbing: **channels** (event + snapshot, each direction) bundled into per-side **endpoints**.
- Receiving: **handlers** — methods subscribed by message type, collected in a registry, dispatched on the receiving
  side's thread; priorities; a handler may consume a message.
- Composability: everything is a plugin, including the Studio App itself (``ViewerApp`` is just handlers) —
  extending or replacing it is a one-line change to the handler list.

.. _StThreading:

Threading models
----------------

.. list-table::
   :header-rows: 1

   * - Application
     - Physics runs on
     - Synchronization
   * - ``simulate`` (classic)
     - A dedicated physics thread
     - A mutex shared with the UI thread
   * - Studio App (C++), native or Live
     - The render thread
     - None needed (single-threaded)
   * - Studio App (Python)
     - The user's thread; the viewer runs on a separate thread
     - Messages only; no shared mutable state

- ``simulate``: physics thread + lock around every UI↔sim interaction.
- Studio C++: single-threaded — simpler, but couples max physics rate to render rate. (The *Threads* setting =
  MuJoCo's internal ``mju_threadpool``, within-step parallelism — a different axis.)
- Python framework: decoupled already — sim publishes snapshots at its own rate, viewer consumes latest; no locks.
- Direction: moving the message layer into C++ would give the desktop app the sim-thread/viewer-thread split —
  ``simulate``'s decoupled physics rate without its mutex.

.. _StCustomization:

Customization: plugins and handlers
-----------------------------------

- Two extension mechanisms today — C++ plugins and Python handlers; same extension points, different idioms;
  reconciliation planned.

.. list-table::
   :header-rows: 1

   * - Extension point
     - C++ (platform plugins)
     - Python (message handlers)
   * - Draw a custom GUI panel
     - ``GuiPlugin``
     - ``BuildGuiEvent`` handler
   * - Add geoms to the rendered scene
     - ``ScenePlugin``
     - ``UpdateEvent`` handler + ``extra_geoms``
   * - Observe / drive the simulation
     - ``ModelPlugin`` (model load, update, pre/post-step hooks)
     - Sim-side handlers + custom messages; ``ModelEvent``
   * - Respond to a hotkey
     - ``KeyHandlerPlugin``
     - ``UpdateEvent`` handler + ImGui key queries
   * - Extend the spec editor
     - ``SpecEditorPlugin``
     - not available

- C++ plugins: small structs of C function pointers in a global registry (``plugin.h``); iterated by the app at the
  matching frame-loop points; one object can register as several plugin types (example:
  ``object_launcher_plugin.cc``).
- Python handlers: :ref:`the Python guide<StPyMessages>`.
