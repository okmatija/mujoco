.. _Studio:

Studio
======

.. toctree::
    :hidden:

    architecture
    python

.. WARNING:: MuJoCo Studio is currently experimental and subject to frequent change.

.. _StIntro:

Introduction
------------

- **Studio is a platform for building interactive visualization tools for MuJoCo.**

  - Industry-standard foundations: SDL windowing, Dear ImGui UI, :ref:`Filament<FilamentRendering>` rendering.
  - Reusable components implementing useful UI/UX widgets and MuJoCo features — see :doc:`architecture`.
  - Ships with the batteries-included **Studio App**, the successor to :ref:`simulate<saSimulate>`.
  - Runs on all desktop platforms and in the browser.

- The Studio App has two launchers, each runs native or in a browser:

  - **C++ launcher**: native :ref:`desktop application<StNativeCpp>`, or in the browser as
    :ref:`Studio Live<StLive>`.
  - **Python launcher**: part of a framework for building simulation tools (your script owns the physics loop);
    native via ``native_viewer``, browser via ``web_viewer`` — see :doc:`python`.

- Code locations: ``src/experimental/studio`` (C++ application), ``src/experimental/platform`` (component library),
  ``python/mujoco/experimental/studio`` (Python framework). [GitHub links]

.. _StCppApps:

C++ applications
----------------

.. _StNativeCpp:

Native: Studio App
~~~~~~~~~~~~~~~~~~

- Build & run via ``build.sh``; follow the Studio README. [links]
- Model loading: command line, file menu, drag-and-drop; ``.xml`` / ``.mjb`` / zipped archives.
- UI layout: central 3D view + dockable panels — toolbar (run/pause, speed, camera), *Options* pane, *Inspector*
  pane, status bar.
- Features beyond ``simulate``, flat inline list: :ref:`mjSpec` editing (*Explorer*/*Properties*, undo/redo),
  timeline scrubbing, per-step profiler, picture-in-picture cameras (color/depth/segmentation), themes, persistent
  settings.
- Keyboard/mouse interaction carries over from :ref:`simulate<saSimulate>` (Space, ``-``/``=``, Backspace,
  Esc/``[``/``]``, Ctrl+drag).

.. _StLive:

Web: Studio Live
~~~~~~~~~~~~~~~~

- The same Studio App compiled to WebAssembly; no server component — physics, rendering (Filament WebGL2) and UI
  all run client-side in the browser.
- Model loading: drag-and-drop; from URLs via ``http(s)`` :ref:`resource providers<exProvider>` (referenced
  meshes/textures fetched automatically); ``github:org/repo/branch/path`` shorthand.

.. _StPyApps:

Python applications
-------------------

.. _StNativePython:

Native: the ``native_viewer``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- The Studio App UI in a native window; physics loop owned by your script.
- CLI: ``python -m mujoco.experimental.studio.viewer --mjcf=...``.
- From a script: the foundation for custom tools (messages, panels, overlays) — see :doc:`python`.

.. _StWebPython:

Web: the ``web_viewer``
~~~~~~~~~~~~~~~~~~~~~~~

- Unlike Studio Live, the simulation and all tool code stay in Python: the UI is streamed to the page, and the 3D
  scene is rendered client-side by a WASM build of the Studio renderer.
- Everything on a single port; native ↔ web is a one-line config change.
- Details: :ref:`the web viewer section<StWebViewer>` of the Python guide.
