# MuJoCo Studio

MuJoCo Studio is the next iteration of the [simulate](simulate) application.
The UI has been reimplemented using [Dear ImGui](https://github.com/ocornut/imgui)
and the default renderer has been
switched to Filament.

> [!IMPORTANT]
> _This application is still WIP. See the [Future Work](#future-work) section.

## Usage

Configure and build MuJoCo Studio by running this command from the top-level
directory. Then follow the printed instructions to run the executable.

```
bash .github/workflows/build_steps.sh dev_studio
```
## Development

The command above the section is intended to get you up and running quickly. You
can see the cmake invocation by reading the `dev_studio` function implementation
in the [build_steps.sh](.github/workflows/build_steps.sh) file. There is also a
`dev_studio_debug` command to conveniently build an executable with debug
information.

If you intend to develop the application you will may want to work from an IDE.
If use [Clion](https://www.jetbrains.com/clion/) you should be able to set it up
to work with the cmake files we provide. If you use Visual Studio follow these
[instructions](https://learn.microsoft.com/en-us/cpp/build/cmake-projects-in-visual-studio?view=msvc-170).


## Configuration

By default Studio uses Filament for rendering, the OpenGL renderer used in
simulate can be used instead by passing the `-DMUJOCO_USE_FILAMENT=OFF` option
in the cmake configuration step.

The Filament renderer has multiple backends, on Linux OpenGL is the default but
Vulkan can be used by providing `-DMUJOCO_USE_FILAMENT=ON` and
`-DMUJOCO_USE_FILAMENT_VULKAN=ON`

See the options in the top-level [CMakeLists.txt](CMakeLists.txt) file for more
details.

## Future Work

1. **Stability and Robustness**. We need user feedback to find and fix bugs.

1. **UI/UX improvements**. We have ported the simulate UI to make it easier to
users to switch. We will be making further changes to make use of the
flexibility offered by Dear ImGui ([examples](https://github.com/ocornut/imgui/issues/8942)).

1. **Python integration**. As with simulate, we would like to make Studio usable
  via Python.

