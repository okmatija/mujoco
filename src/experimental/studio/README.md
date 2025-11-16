# MuJoCo Studio

MuJoCo Studio is the next iteration of the [simulate](../../../simulate)
application. The UI has been reimplemented using [Dear ImGui](https://github.com/ocornut/imgui)
and the default renderer has been switched to Filament. The application is
still WIP, see the [Future Work](#future-work) section for details.

## Usage

Configure and build MuJoCo Studio by running this command from the top-level
directory.

```
cmake -B build -DUSE_STATIC_LIBCXX=OFF -DMUJOCO_BUILD_STUDIO=ON
```

Next build MuJoCo Studio using:

```
cmake --build build --config=Release --target mujoco_studio --parallel
```

## Development

The command above the section is intended to get you up and running quickly. You
can see the cmake invocation by reading the `dev_studio` function implementation
in the [build_steps.sh](../../../.github/workflows/build_steps.sh) file. There
is also a `dev_studio_debug` command to conveniently build an executable with
debug information.

If you intend to develop the application you will may want to work from an IDE.
If use [Clion](https://www.jetbrains.com/clion/) you should be able to set it up
to work with the cmake files we provide. If you use Visual Studio follow these
[instructions](https://learn.microsoft.com/en-us/cpp/build/cmake-projects-in-visual-studio?view=msvc-170).


## Filament Rendering

Studio uses legacy OpenGL rendering by default. There is an option to use
Filament instead by passing `-DMUJOCO_USE_FILAMENT=ON` during the cmake
configuration step. The Filament renderer has multiple rendering backends,
on Linux OpenGL is the default but Vulkan can be used by also providing
the `-DMUJOCO_USE_FILAMENT_VULKAN=ON` option.

Note that you will need to run the application from the folder containing
the executable so that the expected materials/assets can be found. Also note
that currently Filament rendering is only supported on Linux.

See the options in the top-level [CMakeLists.txt](../../../CMakeLists.txt) file
for more details.

## Known Bugs

* MuJoCo Studio does not yet work using Wayland on Linux, use X11 instead.

## Future Work

1. **Stability and Robustness**. We need user feedback to find and fix bugs.

1. **UI/UX improvements**. We have ported the simulate UI to make it easier to
users to switch. We will be making further changes to make use of the
flexibility offered by Dear ImGui ([examples](https://github.com/ocornut/imgui/issues/8942)).

1. **Python integration**. As with simulate, we would like to make Studio usable
  via Python.
