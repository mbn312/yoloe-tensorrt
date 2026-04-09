cmake_minimum_required(VERSION 3.24)

project(yoloe_tensorrt_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(YOLOE_TRT_BUILD_NATIVE "Build the native TensorRT extension" ON)
option(YOLOE_TRT_BUILD_JETSON_CAMERA "Build the Jetson zero-copy camera backend when dependencies are available" ON)

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
    $ENV{TENSORRT_ROOT}
    $ENV{TensorRT_ROOT}
    /usr
    /usr/local
    /opt
    /opt/TensorRT
    /usr/src/tensorrt
  PATH_SUFFIXES
    lib
    lib64
    lib/${CMAKE_LIBRARY_ARCHITECTURE}
  REQUIRED
)

get_filename_component(TENSORRT_LIBRARY_DIR "${TENSORRT_LIBRARY}" DIRECTORY)
get_filename_component(TENSORRT_LIBRARY_PREFIX "${TENSORRT_LIBRARY_DIR}" DIRECTORY)

find_path(
  TENSORRT_INCLUDE_DIR
  NvInfer.h
  HINTS
    $ENV{TENSORRT_ROOT}
    $ENV{TensorRT_ROOT}
    "${TENSORRT_LIBRARY_DIR}"
    "${TENSORRT_LIBRARY_PREFIX}"
    /usr
    /usr/local
    /opt
    /opt/TensorRT
    /usr/src/tensorrt
  PATH_SUFFIXES
    include
    include/${CMAKE_LIBRARY_ARCHITECTURE}
    ${CMAKE_LIBRARY_ARCHITECTURE}
  REQUIRED
)

message(STATUS "Found TensorRT include dir: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "Found TensorRT library: ${TENSORRT_LIBRARY}")

set(NATIVE_SOURCES src/native/main.cpp)
set(NATIVE_EXTRA_LIBS "")
set(NATIVE_EXTRA_INCLUDE_DIRS "")
set(NATIVE_EXTRA_COMPILE_DEFINITIONS "")

if(YOLOE_TRT_BUILD_JETSON_CAMERA)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)

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
      list(APPEND NATIVE_SOURCES src/native/jetson_camera_kernels.cu)
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
      set_source_files_properties(src/native/jetson_camera_kernels.cu PROPERTIES LANGUAGE CUDA)
      message(STATUS "Enabling Jetson zero-copy camera backend")
    else()
      message(WARNING "Jetson zero-copy camera backend dependencies not found; building without camera backend")
    endif()
  else()
    message(WARNING "CUDA compiler not available; building without Jetson zero-copy camera backend")
  endif()
endif()

pybind11_add_module(_native MODULE ${NATIVE_SOURCES})

target_include_directories(
  _native
  PRIVATE
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
  set_target_properties(_native PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
endif()

install(TARGETS _native DESTINATION yoloe_tensorrt)
