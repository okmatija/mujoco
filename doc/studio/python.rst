.. _StPython:

Python framework
================

.. WARNING:: MuJoCo Studio is currently experimental and subject to frequent change. The APIs on this page live under
   ``mujoco.experimental`` and may change without notice.

- **The Studio Python API decouples simulation and visualization, and implements communication and customization
  with a unified, composable API.**

  - *Decoupled*: the sim and the viewer do not share state — so they can run in different threads or processes,
    and the viewer can be native or in the browser.
  - *Communication and customization* both happen by passing typed, immutable **Messages**: send ``Snapshot`` or
    ``Event`` messages depending on the semantics you need; handle the empty per-frame ``UpdateEvent`` and
    ``BuildGuiEvent`` to add custom logic and UI.
  - *Composable*: extending or replacing behavior is a one-line change — everything is a plugin, including the
    Studio App itself.

- The Studio counterpart of the classic :ref:`passive viewer<PyViewerPassive>`; your script owns the physics loop.
  Design details: :ref:`the architecture page<StMessages>`.

.. _StQuickStart:

Quick start
-----------

- The canonical tool — launch, send a model, run your loop:

.. code-block:: python

   import mujoco
   from mujoco.experimental.studio import launch_passive
   from mujoco.experimental.studio import messages
   from mujoco.experimental.studio import sim
   from mujoco.experimental.studio import viewer_app
   from mujoco.experimental.studio import viewer_protocol

   model = mujoco.MjModel.from_xml_path('humanoid.xml')
   data = mujoco.MjData(model)

   config = viewer_protocol.ViewerConfig(title='My tool')
   with launch_passive.launch_passive(
       config, viewer_plugins=[viewer_app.ViewerApp()]) as handle:
     handle.send_to_viewer(messages.ModelEvent(model=model, path='humanoid.xml'))

     step_control = sim.StepControl()
     while handle.is_running():
       step_control.advance(model, data)
       model, data, step_control = handle.sync(model, data, step_control)

- Points to make: ``launch_passive`` returns immediately (viewer on a daemon thread); it takes *no model* — the
  model travels as a ``ModelEvent`` like everything else; ``advance`` honors the GUI pause/speed controls;
  ``ViewerApp`` provides the whole Studio UI and is optional (:ref:`below<StViewerApp>`).
- CLI: ``python -m mujoco.experimental.studio.viewer --mjcf=...`` with ``--gfx``, ``--width``, ``--height``,
  ``--viewer={native,web}``, ``--port``.
- ``.. warning::`` — ``handle.sync`` *returns* model/data/step_control and the loop must rebind them (drag-and-drop
  loads a new model; ignoring the return value silently keeps simulating the old one).

.. _StViewing:

Viewing options
---------------

- ``ViewerConfig`` fields: ``title``, ``width``/``height`` (1200×800), ``gfx``, ``viewer_mode``
  (``NATIVE``/``WEB``), ``http_port`` (web only; 0 = first free port from 8080).
- ``viewer_mode`` is the single native↔web switch; handlers/messages/panels/sim loop identical in both.
- ``gfx`` modes: ``classic``, ``classic_headless``, ``opengl``, ``opengl_headless``, ``opengl_software``,
  ``vulkan``, ``vulkan_software``, ``webgl``; default auto-picks; software modes for GPU-less machines.

.. _StWebViewer:

The web viewer
~~~~~~~~~~~~~~

- No window; prints a banner with local + LAN URLs. Browser shows the same UI (streamed ImGui draw data; input
  flows back) and renders the scene client-side (WASM Studio renderer). Viewer-side handlers work unmodified.
- One port for everything (page, model, UI stream, state stream, drops) → one firewall rule / tunnel exposes the
  viewer.
- Sessions: first browser = controller; others join as spectators; control can be requested/granted.
- Drag-and-drop uploads model files to Python; the new model comes back through ``handle.sync``.
- Requires the ``websockets`` package.
- ``.. note::`` — custom messages exist only between the Python sides; the browser stream is fixed (UI draw data +
  state). Custom *UI* does reach the browser, since it is ImGui draw data.

.. _StSimSide:

The simulation side
-------------------

- Your script owns the loop; the framework never steps physics for you.
- ``StepControl``: ``advance(model, data)`` → ``StepStatus`` (``OK``/``PAUSED``/``AUTO_RESET``/``DIVERGED``), GIL
  released; speed get/set + measured; pause + ``request_single_step``; noise parameters; ``force_sync``.
- ``ViewerHandle``: ``sync(model, data, step_control)`` (drains viewer messages → dispatches to sim handlers →
  publishes state → returns possibly-replaced objects); ``send_to_viewer``; ``is_running`` (also detects a dead
  viewer thread); ``close``; context manager.
- ``sim_plugins=[...]``: run inside ``sync`` on your thread; typical use — react to custom messages from your GUI
  code.

.. _StPyMessages:

Messages and handlers
---------------------

- Messages = frozen dataclasses deriving ``messages.Event`` or ``messages.Snapshot``; semantics in
  :ref:`the architecture page<StMessages>`.
- Built-in messages:

.. list-table::
   :header-rows: 1

   * - Message
     - Kind
     - Meaning
   * - ``StateSnapshot``
     - Snapshot, sim → viewer
     - The physics state (as in ``mj_getState``); sent by every ``handle.sync``.
   * - ``ModelEvent``
     - Event, both directions
     - A new model. Sending it to the viewer loads the model; the viewer sends it to the sim after drag-and-drop.
   * - ``ResetEvent``
     - Event, viewer → sim
     - Reset the simulation (``mj_resetData``).
   * - ``PerturbEvent``
     - Event, viewer → sim
     - Applied perturbation forces from mouse dragging.
   * - ``StepControlSnapshot``
     - Snapshot, viewer → sim
     - Pause state, speed and noise settings from the GUI.
   * - ``MjOptionSnapshot``
     - Snapshot, viewer → sim
     - Physics options (``mjOption``) edited in the GUI.
   * - ``ExitEvent``
     - Event, both directions
     - Shut down the other side.

- Viewer-side lifecycle events, dispatched every frame: ``UpdateEvent`` (input/per-frame logic), ``BuildGuiEvent``
  (build ImGui panels); plus ``ViewerInitEvent`` once at startup (carries the viewer).
- ``@messages.handler``: second parameter's type annotation = the subscription; any object with handler methods is a
  plugin and goes in ``viewer_plugins`` / ``sim_plugins``.
- Worked example (short code block): define ``RewardSnapshot(messages.Snapshot)``; a ``RewardPanel`` class with an
  ``on_reward`` handler and an ``on_build_gui`` handler drawing an ImGui window; sim loop calls
  ``handle.send_to_viewer(RewardSnapshot(value=...))``.
- Dispatch rules: priority order (``messages.Priority``, higher first, default ``USER``); truthy return consumes;
  dispatch follows the class hierarchy (annotate ``messages.Event`` to get all events); discovery is by method name
  — same-name subclass methods replace base handlers.

.. _StViewerCustom:

Customizing the viewer
----------------------

- **Custom panels**: handle ``BuildGuiEvent``; use the bundled ``dear_imgui`` / ``implot`` bindings (shared ImGui
  context — panels join the Studio docking layout); the ``ux`` module exposes Studio's ready-made panels (physics,
  rendering, joints, sensors, …, themes, camera selection, docking helpers).
- **Custom overlays**: append ``mujoco.MjvGeom`` to ``viewer.extra_geoms`` from an ``UpdateEvent`` handler — the
  Studio counterpart of ``user_scn``; persists across frames, cleared on model change, works in native and web.
- **Interaction**: ``ux.move_camera`` / ``pick`` / ``init_perturb`` / ``move_perturb``; Studio's keyboard/mouse
  behavior is in ``studio_app_events`` free functions (transitional; will be replaced by a key-binding API).
- **Beyond panels**: subclass ``viewer_protocol.Viewer`` (the native and web viewers are the two reference
  implementations).

.. _StViewerApp:

The ViewerApp class
-------------------

- ``ViewerApp`` = the Studio UI packaged as a plugin: menu bar, toolbar, options/inspector panes, status bar,
  input handling, and the per-frame messages that connect GUI controls (pause/speed/options) to the sim.
- Custom tools are *additional* plugins listed alongside it:
  ``viewer_plugins=[viewer_app.ViewerApp(), RewardPanel()]``.
- Bootstraps via ``ViewerInitEvent``; then dispatches ``ViewerAppInitEvent`` carrying itself — handle it to keep a
  reference (e.g. to read the current selection from a panel).
- Optional: without it you get a bare viewer (rendering + message plumbing + lifecycle events, no built-in UI), and
  your handlers build whatever interface the tool needs from ``ux`` / ``studio_app_events``.

.. _StSamples:

Samples
-------

- ``python/mujoco/experimental/studio/sample`` [GitHub link]:

  - ``ghost.py`` — tool *without* ``ViewerApp``: custom model/GUI/update handlers; time-delayed ghost via
    ``extra_geoms``.
  - ``implot.py`` — custom ImPlot panel on top of the full Studio UI; ``ViewerAppInitEvent`` idiom; responsive
    layout.
  - ``render.py`` — offscreen rendering, no viewer.

.. _StMigration:

Migrating from ``mujoco.viewer``
--------------------------------

.. list-table::
   :header-rows: 1

   * - ``mujoco.viewer``
     - Studio framework
   * - ``launch_passive(model, data)``
     - ``launch_passive(config)``; the model is sent afterwards as a ``ModelEvent``
   * - ``handle.sync()``
     - ``handle.sync(model, data, step_control)`` — note the return value must be rebound
   * - ``handle.lock()``
     - Not needed: the viewer owns a copy of the model, and all communication is via immutable messages
   * - ``user_scn``
     - ``viewer.extra_geoms`` (viewer side)
   * - ``key_callback=``
     - An ``UpdateEvent`` handler querying keys via ImGui
   * - ``cam`` / ``opt`` / ``pert`` handle properties
     - Viewer-side state: ``viewer.camera``, ``viewer.vis_options``, ``viewer.perturb``, accessed from viewer-side
       handlers
   * - Timing your own loop with ``time.sleep``
     - ``sim.StepControl.advance``, which paces to real time and honors the GUI controls
