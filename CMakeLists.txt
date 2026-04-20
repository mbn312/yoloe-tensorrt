cmake_minimum_required(VERSION 3.24)

project(yoloe_tensorrt_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(YOLOE_TRT_BUILD_NATIVE "Build the native TensorRT extension" ON)
option(YOLOE_TRT_BUILD_CUDA_PREPROCESS "Build the shared CUDA preprocess kernels when a CUDA compiler is available" ON)
option(YOLOE_TRT_BUILD_JETSON_CAMERA "Build the Jetson zero-copy camera backend when dependencies are available" ON)

if(YOLOE_TRT_BUILD_JETSON_CAMERA AND NOT YOLOE_TRT_BUILD_CUDA_PREPROCESS)
  message(STATUS "YOLOE_TRT_BUILD_JETSON_CAMERA requires YOLOE_TRT_BUILD_CUDA_PREPROCESS; forcing preprocess ON")
  set(YOLOE_TRT_BUILD_CUDA_PREPROCESS ON CACHE BOOL "Build the shared CUDA preprocess kernels when a CUDA compiler is available" FORCE)
endif()

if(NOT YOLOE_TRT_BUILD_NATIVE)
  message(STATUS "YOLOE_TRT_BUILD_NATIVE=OFF, skipping native extension build")
  return()
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
include(CheckLanguage)
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE pybind11_DISCOVERY_STATUS
  )
  if(pybind11_DISCOVERY_STATUS EQUAL 0 AND EXISTS "${pybind11_CMAKE_DIR}/pybind11Config.cmake")
    list(APPEND CMAKE_PREFIX_PATH "${pybind11_CMAKE_DIR}")
    find_package(pybind11 CONFIG REQUIRED)
  else()
    message(FATAL_ERROR
      "Could not find pybind11 CMake config. Install pybind11 in the active Python environment "
      "or set pybind11_DIR to a directory containing pybind11Config.cmake."
    )
  endif()
endif()
find_package(CUDAToolkit REQUIRED)
find_package(PkgConfig REQUIRED)

set(CUDA_HEADER_HINT_DIRS
  ${CUDAToolkit_INCLUDE_DIRS}
  ${CUDAToolkit_INCLUDE_DIR}
  $ENV{CUDA_PATH}
  $ENV{CUDA_HOME}
  /usr/local/cuda
  /usr/local/cuda-${CUDAToolkit_VERSION}
)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c
          "import importlib.util, pathlib, sysconfig; candidates = []; \
seen = set(); \
def add(path): \
    if not path: return; \
    resolved = pathlib.Path(path).resolve(); \
    text = str(resolved); \
    if resolved.exists() and text not in seen: \
        seen.add(text); candidates.append(text); \
for key in ('platlib', 'purelib'): \
    base = sysconfig.get_paths().get(key); \
    if not base: continue; \
    root = pathlib.Path(base); \
    for relative in ('nvidia/cuda_runtime/include', 'nvidia/cu13/include', 'triton/backends/nvidia/include'): \
        add(root / relative); \
for name in ('nvidia.cuda_runtime', 'nvidia.cu13', 'triton.backends.nvidia'): \
    spec = importlib.util.find_spec(name); \
    if spec is None: continue; \
    if spec.submodule_search_locations: \
        for location in spec.submodule_search_locations: \
            add(location); add(pathlib.Path(location) / 'include'); \
    elif spec.origin: \
        location = pathlib.Path(spec.origin).parent; \
        add(location); add(location / 'include'); \
print('\\n'.join(candidates))"
  OUTPUT_VARIABLE CUDA_PYTHON_HEADER_HINTS
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
  RESULT_VARIABLE CUDA_PYTHON_HEADER_HINTS_STATUS
)
if(CUDA_PYTHON_HEADER_HINTS_STATUS EQUAL 0 AND CUDA_PYTHON_HEADER_HINTS)
  string(REPLACE "\n" ";" CUDA_PYTHON_HEADER_HINTS "${CUDA_PYTHON_HEADER_HINTS}")
  list(APPEND CUDA_HEADER_HINT_DIRS ${CUDA_PYTHON_HEADER_HINTS})
endif()
list(REMOVE_DUPLICATES CUDA_HEADER_HINT_DIRS)

find_path(
  CUDA_RUNTIME_INCLUDE_DIR
  cuda_fp16.h
  HINTS
    ${CUDA_HEADER_HINT_DIRS}
  PATH_SUFFIXES
    .
    include
    include/${CMAKE_LIBRARY_ARCHITECTURE}
    targets/${CMAKE_LIBRARY_ARCHITECTURE}/include
)

if(NOT CUDA_RUNTIME_INCLUDE_DIR)
  message(FATAL_ERROR
    "Could not find CUDA runtime headers (missing cuda_fp16.h). "
    "Install the CUDA Toolkit headers or a compatible Python CUDA runtime package in the active environment."
  )
endif()

set(CUDA_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS})
list(APPEND CUDA_INCLUDE_DIRS "${CUDA_RUNTIME_INCLUDE_DIR}")
list(REMOVE_DUPLICATES CUDA_INCLUDE_DIRS)

set(CUDA_LIBRARY_HINT_DIRS
  ${CUDAToolkit_LIBRARY_DIR}
  ${CUDAToolkit_LIBRARY_ROOT}
  $ENV{CUDA_PATH}
  $ENV{CUDA_HOME}
  /usr
  /usr/local
  /usr/local/cuda
  /usr/local/cuda-${CUDAToolkit_VERSION}
)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c
          "import importlib.util, pathlib, sysconfig; candidates = []; \
seen = set(); \
def add(path): \
    if not path: return; \
    resolved = pathlib.Path(path).resolve(); \
    text = str(resolved); \
    if resolved.exists() and text not in seen: \
        seen.add(text); candidates.append(text); \
for key in ('platlib', 'purelib'): \
    base = sysconfig.get_paths().get(key); \
    if not base: continue; \
    root = pathlib.Path(base); \
    for relative in ('nvidia/cu13/lib', 'nvidia/cuda_runtime/lib', 'nvidia/cuda_runtime/lib64'): \
        add(root / relative); \
for name in ('nvidia.cu13', 'nvidia.cuda_runtime'): \
    spec = importlib.util.find_spec(name); \
    if spec is None: continue; \
    if spec.submodule_search_locations: \
        for location in spec.submodule_search_locations: \
            add(location); add(pathlib.Path(location) / 'lib'); add(pathlib.Path(location) / 'lib64'); \
    elif spec.origin: \
        location = pathlib.Path(spec.origin).parent; \
        add(location); add(location / 'lib'); add(location / 'lib64'); \
print('\\n'.join(candidates))"
  OUTPUT_VARIABLE CUDA_PYTHON_LIBRARY_HINTS
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
  RESULT_VARIABLE CUDA_PYTHON_LIBRARY_HINTS_STATUS
)
if(CUDA_PYTHON_LIBRARY_HINTS_STATUS EQUAL 0 AND CUDA_PYTHON_LIBRARY_HINTS)
  string(REPLACE "\n" ";" CUDA_PYTHON_LIBRARY_HINTS "${CUDA_PYTHON_LIBRARY_HINTS}")
  list(APPEND CUDA_LIBRARY_HINT_DIRS ${CUDA_PYTHON_LIBRARY_HINTS})
endif()
list(REMOVE_DUPLICATES CUDA_LIBRARY_HINT_DIRS)

find_library(
  CUDA_DEVICE_RUNTIME_LIBRARY
  cudadevrt
  HINTS
    ${CUDA_LIBRARY_HINT_DIRS}
  PATH_SUFFIXES
    .
    lib
    lib64
    lib/${CMAKE_LIBRARY_ARCHITECTURE}
    targets/${CMAKE_LIBRARY_ARCHITECTURE}/lib
)

find_library(
  CUDA_RUNTIME_SHARED_LIBRARY
  cudart
  HINTS
    ${CUDA_LIBRARY_HINT_DIRS}
  PATH_SUFFIXES
    .
    lib
    lib64
    lib/${CMAKE_LIBRARY_ARCHITECTURE}
    targets/${CMAKE_LIBRARY_ARCHITECTURE}/lib
)

set(CUDA_LINK_DIRECTORIES "")
foreach(cuda_runtime_lib IN ITEMS "${CUDA_DEVICE_RUNTIME_LIBRARY}" "${CUDA_RUNTIME_SHARED_LIBRARY}")
  if(cuda_runtime_lib)
    get_filename_component(cuda_runtime_lib_dir "${cuda_runtime_lib}" DIRECTORY)
    list(APPEND CUDA_LINK_DIRECTORIES "${cuda_runtime_lib_dir}")
  endif()
endforeach()
list(REMOVE_DUPLICATES CUDA_LINK_DIRECTORIES)

set(CUDA_LINK_OPTIONS "")
foreach(cuda_link_dir IN LISTS CUDA_LINK_DIRECTORIES)
  list(APPEND CUDA_LINK_OPTIONS "-L${cuda_link_dir}")
  list(APPEND CUDA_LINK_OPTIONS "-Wl,-rpath-link,${cuda_link_dir}")
endforeach()

set(TENSORRT_HINT_DIRS
  $ENV{TENSORRT_ROOT}
  $ENV{TensorRT_ROOT}
  /usr
  /usr/local
  /opt
  /opt/TensorRT
  /usr/src/tensorrt
)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c
          "import importlib.util, pathlib, site, sysconfig; candidates = []; \
modules = ('tensorrt', 'tensorrt_libs', 'tensorrt_lean', 'tensorrt_dispatch', 'tensorrt_bindings'); \
seen = set(); \
def add(path): \
    if not path: return; \
    text = str(pathlib.Path(path).resolve()); \
    if text not in seen and pathlib.Path(text).exists(): \
        seen.add(text); candidates.append(text); \
for key in ('platlib', 'purelib'): add(sysconfig.get_paths().get(key)); \
for path in site.getsitepackages(): add(path); \
add(site.getusersitepackages()); \
for name in modules: \
    spec = importlib.util.find_spec(name); \
    if spec is None: continue; \
    if spec.submodule_search_locations: \
        [add(location) for location in spec.submodule_search_locations]; \
    elif spec.origin: add(pathlib.Path(spec.origin).parent); \
print('\\n'.join(candidates))"
  OUTPUT_VARIABLE TENSORRT_PYTHON_HINTS
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
  RESULT_VARIABLE TENSORRT_PYTHON_HINTS_STATUS
)
if(TENSORRT_PYTHON_HINTS_STATUS EQUAL 0 AND TENSORRT_PYTHON_HINTS)
  string(REPLACE "\n" ";" TENSORRT_PYTHON_HINTS "${TENSORRT_PYTHON_HINTS}")
  list(APPEND TENSORRT_HINT_DIRS ${TENSORRT_PYTHON_HINTS})
endif()
list(REMOVE_DUPLICATES TENSORRT_HINT_DIRS)

set(HAVE_CUDA_COMPILER OFF)
if(YOLOE_TRT_BUILD_CUDA_PREPROCESS OR YOLOE_TRT_BUILD_JETSON_CAMERA)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
    enable_language(CUDA)
    set(HAVE_CUDA_COMPILER ON)
  endif()
endif()

find_package(OpenCV QUIET COMPONENTS core imgproc imgcodecs)
if(OpenCV_FOUND)
  message(STATUS "Found OpenCV via CMake: ${OpenCV_VERSION}")
else()
  pkg_check_modules(OPENCV REQUIRED IMPORTED_TARGET opencv4)
  set(OpenCV_INCLUDE_DIRS ${OPENCV_INCLUDE_DIRS})
  set(OpenCV_LIBS PkgConfig::OPENCV)
  message(STATUS "Found OpenCV via pkg-config: ${OPENCV_VERSION}")
endif()

find_library(
  TENSORRT_LIBRARY
  nvinfer
  HINTS
    ${TENSORRT_HINT_DIRS}
  PATH_SUFFIXES
    .
    lib
    libs
    lib64
    lib/${CMAKE_LIBRARY_ARCHITECTURE}
    targets/${CMAKE_LIBRARY_ARCHITECTURE}/lib
)

if(NOT TENSORRT_LIBRARY)
  message(FATAL_ERROR
    "Could not find the TensorRT C++ library 'nvinfer'. "
    "This project builds a native TensorRT extension and needs the native runtime library to be discoverable by CMake. "
    "Set TENSORRT_ROOT to your TensorRT install prefix if it is not under a standard system location. "
    "If you only installed TensorRT with `pip install tensorrt`, note that the Python package layout may differ from a system SDK install and may still be insufficient for native C++ builds. "
    "On Debian/Ubuntu with NVIDIA's TensorRT apt repository configured, a common fix is `sudo apt-get update && sudo apt-get install -y libnvinfer-dev` plus the matching runtime package for your TensorRT major version, for example `libnvinfer10` on TensorRT 10.x."
  )
endif()

get_filename_component(TENSORRT_LIBRARY_DIR "${TENSORRT_LIBRARY}" DIRECTORY)
get_filename_component(TENSORRT_LIBRARY_PREFIX "${TENSORRT_LIBRARY_DIR}" DIRECTORY)

find_path(
  TENSORRT_INCLUDE_DIR
  NvInfer.h
  HINTS
    ${TENSORRT_HINT_DIRS}
    "${TENSORRT_LIBRARY_DIR}"
    "${TENSORRT_LIBRARY_PREFIX}"
  PATH_SUFFIXES
    .
    include
    includes
    include/${CMAKE_LIBRARY_ARCHITECTURE}
    ${CMAKE_LIBRARY_ARCHITECTURE}
    targets/${CMAKE_LIBRARY_ARCHITECTURE}/include
)

if(NOT TENSORRT_INCLUDE_DIR)
  message(FATAL_ERROR
    "Could not find TensorRT C++ headers (missing NvInfer.h). "
    "This native extension needs the TensorRT development headers. NVIDIA's TensorRT pip installation does not include the C++ headers, so `pip install tensorrt` alone is not enough to build this package natively. "
    "Install the TensorRT development package/tarball or set TENSORRT_ROOT to a full TensorRT SDK install. "
    "On Debian/Ubuntu with NVIDIA's TensorRT apt repository configured, a common fix is `sudo apt-get update && sudo apt-get install -y libnvinfer-dev` plus the matching runtime package for your TensorRT major version, for example `libnvinfer10` on TensorRT 10.x. "
    "If you only need docs or lightweight non-native workflows, install with `--config-settings=cmake.define.YOLOE_TRT_BUILD_NATIVE=OFF`."
  )
endif()

message(STATUS "Found TensorRT include dir: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "Found TensorRT library: ${TENSORRT_LIBRARY}")
message(STATUS "Found CUDA include dirs: ${CUDA_INCLUDE_DIRS}")
message(STATUS "Found CUDA link dirs: ${CUDA_LINK_DIRECTORIES}")

set(NATIVE_SOURCES
  src/native/module.cpp
  src/native/main_runtime.cpp
  src/native/visual_runtime.cpp
  src/native/jetson_camera.cpp
)
set(NATIVE_EXTRA_LIBS "")
set(NATIVE_EXTRA_INCLUDE_DIRS "")
set(NATIVE_EXTRA_COMPILE_DEFINITIONS "")

if(YOLOE_TRT_BUILD_CUDA_PREPROCESS)
  if(HAVE_CUDA_COMPILER)
    list(APPEND NATIVE_SOURCES src/native/preprocess_kernels.cu)
    list(APPEND NATIVE_EXTRA_COMPILE_DEFINITIONS YOLOE_TRT_ENABLE_CUDA_PREPROCESS=1)
    set_source_files_properties(src/native/preprocess_kernels.cu PROPERTIES LANGUAGE CUDA)
    message(STATUS "Enabling shared CUDA preprocess kernels")
  else()
    message(WARNING "CUDA compiler not available; building native runtime with CPU/OpenCV preprocess fallback")
  endif()
endif()

if(YOLOE_TRT_BUILD_JETSON_CAMERA)
  if(HAVE_CUDA_COMPILER)
    pkg_check_modules(GSTREAMER QUIET IMPORTED_TARGET gstreamer-1.0)
    pkg_check_modules(GSTREAMER_APP QUIET IMPORTED_TARGET gstreamer-app-1.0)
    pkg_check_modules(GSTREAMER_ALLOCATORS QUIET IMPORTED_TARGET gstreamer-allocators-1.0)
    pkg_check_modules(GSTREAMER_VIDEO QUIET IMPORTED_TARGET gstreamer-video-1.0)
    pkg_check_modules(EGL QUIET IMPORTED_TARGET egl)

    find_path(
      NVBUFSURFACE_INCLUDE_DIR
      NvBufSurface.h
      HINTS
        /usr/src/jetson_multimedia_api/include
        /usr/include
        /usr/local/include
    )
    find_library(
      NVBUFSURFACE_LIBRARY
      nvbufsurface
      HINTS
        /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/nvidia
        /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}
        /lib/${CMAKE_LIBRARY_ARCHITECTURE}
        /usr/lib
        /lib
    )

    if(GSTREAMER_FOUND AND GSTREAMER_APP_FOUND AND GSTREAMER_ALLOCATORS_FOUND AND GSTREAMER_VIDEO_FOUND AND EGL_FOUND AND NVBUFSURFACE_INCLUDE_DIR AND NVBUFSURFACE_LIBRARY)
      list(APPEND NATIVE_EXTRA_LIBS
        PkgConfig::GSTREAMER
        PkgConfig::GSTREAMER_APP
        PkgConfig::GSTREAMER_ALLOCATORS
        PkgConfig::GSTREAMER_VIDEO
        PkgConfig::EGL
        ${NVBUFSURFACE_LIBRARY}
        CUDA::cuda_driver
      )
      list(APPEND NATIVE_EXTRA_INCLUDE_DIRS ${NVBUFSURFACE_INCLUDE_DIR})
      list(APPEND NATIVE_EXTRA_COMPILE_DEFINITIONS YOLOE_TRT_ENABLE_JETSON_CAMERA=1)
      message(STATUS "Enabling Jetson zero-copy camera backend")
    else()
      message(WARNING "Jetson zero-copy camera backend dependencies not found; building without camera backend")
    endif()
  else()
    message(WARNING "CUDA compiler not available; building without Jetson zero-copy camera backend")
  endif()
endif()

pybind11_add_module(_native MODULE ${NATIVE_SOURCES})

target_link_options(
  _native
  PRIVATE
    ${CUDA_LINK_OPTIONS}
)

target_include_directories(
  _native
  PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/native
    ${OpenCV_INCLUDE_DIRS}
    ${NATIVE_EXTRA_INCLUDE_DIRS}
)

target_link_libraries(
  _native
  PRIVATE
    ${TENSORRT_LIBRARY}
    CUDA::cudart
    ${OpenCV_LIBS}
    ${NATIVE_EXTRA_LIBS}
)

target_compile_definitions(_native PRIVATE PY_SSIZE_T_CLEAN=1)
target_compile_definitions(_native PRIVATE ${NATIVE_EXTRA_COMPILE_DEFINITIONS})

if(TARGET _native AND CMAKE_CUDA_COMPILER)
  set_target_properties(
    _native
    PROPERTIES
      CUDA_RUNTIME_LIBRARY Shared
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
  )
endif()

install(TARGETS _native DESTINATION yoloe_tensorrt)
