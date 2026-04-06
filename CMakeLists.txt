cmake_minimum_required(VERSION 3.24)

project(yoloe_tensorrt_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(YOLOE_TRT_BUILD_NATIVE "Build the native TensorRT extension" ON)

if(NOT YOLOE_TRT_BUILD_NATIVE)
  message(STATUS "YOLOE_TRT_BUILD_NATIVE=OFF, skipping native extension build")
  return()
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
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

pybind11_add_module(_native MODULE src/native/main.cpp)

target_include_directories(
  _native
  PRIVATE
    ${TENSORRT_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/native
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(
  _native
  PRIVATE
    ${TENSORRT_LIBRARY}
    CUDA::cudart
    ${OpenCV_LIBS}
)

target_compile_definitions(_native PRIVATE PY_SSIZE_T_CLEAN=1)

install(TARGETS _native DESTINATION yoloe_tensorrt)
