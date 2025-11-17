#!/bin/bash

# Detect the operating system
OS="$(uname -s)"
case "${OS}" in
    Linux*)     export OS_NAME="linux" ;;
    Darwin*)    export OS_NAME="macos" ;;
    MINGW* | CYGWIN* | MSYS*) export OS_NAME="windows" ;;
    *)          echo "Unsupported OS: ${OS}" >&2; exit 1 ;;
esac

# Parse arguments
do_configure=false
do_build=false
build_type="Release"
explicit_action=false

for arg in "$@"; do
    case "$arg" in
        configure|config)
            do_configure=true
            explicit_action=true
            ;;
        build)
            do_build=true
            explicit_action=true
            ;;
        debug|dbg)
            build_type="Debug"
            ;;
    esac
done

# If no specific action (configure/build) was requested, do both by default.
if [[ "$explicit_action" == false ]]; then
    do_configure=true
    do_build=true
fi

cd "$(git rev-parse --show-toplevel)"

# Configure MuJoCo Studio
if [[ "$do_configure" == true ]]; then
    echo "Configuring MuJoCo Studio (${build_type})..."
    CMAKE_CONFIG_ARGS=(
        "-B build"
        "-G Ninja"
        "-DCMAKE_BUILD_TYPE=${build_type}"
        "-DUSE_STATIC_LIBCXX=OFF"
        "-DBUILD_SHARED_LIB=OFF"
        "-DMUJOCO_USE_FILAMENT=OFF"
        "-DMUJOCO_USE_FILAMENT=OFF"
        "-DMUJOCO_USE_FILAMENT_VULKAN=OFF"
        "-DMUJOCO_BUILD_EXAMPLES=OFF"
        "-DMUJOCO_BUILD_SIMULATE=OFF"
        "-DMUJOCO_BUILD_TESTS=OFF"
        "-DMUJOCO_TEST_PYTHON_UTIL=OFF"
        "-DMUJOCO_WITH_USD=OFF"
        "-DMUJOCO_BUILD_STUDIO=ON"
    )

    if [[ "${build_type}" == "Debug" ]]; then
        CMAKE_CONFIG_ARGS+=("-DCMAKE_CXX_FLAGS=\"-O0 -g3\"")
        CMAKE_CONFIG_ARGS+=("-DCMAKE_C_FLAGS=\"-O0 -g3\"")
    fi

    # Add user-defined CMAKE_ARGS at the end so they override other settings.
    if [[ -n "${CMAKE_ARGS}" ]]; then
        CMAKE_CONFIG_ARGS+=("${CMAKE_ARGS}")
    fi

    cmake ${CMAKE_CONFIG_ARGS[@]}

    echo "Configuring MuJoCo Studio (${build_type})... DONE"
fi

# Build MuJoCo Studio
if [[ "$do_build" == true ]]; then
    echo "Building MuJoCo Studio..."
    cmake --build build --target mujoco_studio --parallel
    echo "Building MuJoCo Studio... DONE"

    # Print the command to run the built MuJoCo Studio from the right directory.
    echo "Use the following command to run mujoco_studio"
    echo ""
    if [[ "${OS_NAME}" == "windows" ]]; then
        echo " cd $(git rev-parse --show-toplevel)/build/bin && ./mujoco_studio.exe "
    else
        echo " cd $(git rev-parse --show-toplevel)/build/bin && ./mujoco_studio "
    fi
fi
