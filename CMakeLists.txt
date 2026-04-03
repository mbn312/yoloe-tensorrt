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
find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(PkgConfig REQUIRED)

pkg_check_modules(OPENCV REQUIRED opencv4)

find_path(
  OPENCV_INCLUDE_DIR
  opencv2/imgproc.hpp
  PATHS
    /usr/include/opencv4
    /usr/local/include/opencv4
  REQUIRED
)

find_path(
  TENSORRT_INCLUDE_DIR
  NvInfer.h
  PATHS
    /usr/include
    /usr/include/aarch64-linux-gnu
    /usr/include/x86_64-linux-gnu
  REQUIRED
)

find_library(
  TENSORRT_LIBRARY
  nvinfer
  PATHS
    /usr/lib
    /usr/lib/aarch64-linux-gnu
    /usr/lib/x86_64-linux-gnu
  REQUIRED
)

pybind11_add_module(_native MODULE src/native/main.cpp)

target_include_directories(
  _native
  PRIVATE
    ${TENSORRT_INCLUDE_DIR}
    ${OPENCV_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/native
)

target_link_libraries(
  _native
  PRIVATE
    ${TENSORRT_LIBRARY}
    CUDA::cudart
    ${OPENCV_LIBRARIES}
)

target_compile_definitions(_native PRIVATE PY_SSIZE_T_CLEAN=1)

install(TARGETS _native DESTINATION yoloe_tensorrt)
