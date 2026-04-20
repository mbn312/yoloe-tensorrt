#pragma once

#include "dlpack.h"

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <NvInferRuntime.h>
#include <opencv2/imgproc.hpp>
#include "preprocess_kernels.h"

#if defined(YOLOE_TRT_ENABLE_JETSON_CAMERA) && !defined(YOLOE_TRT_ENABLE_CUDA_PREPROCESS)
#error "Jetson zero-copy camera backend requires YOLOE_TRT_ENABLE_CUDA_PREPROCESS"
#endif

#ifdef YOLOE_TRT_ENABLE_JETSON_CAMERA

#include <EGL/egl.h>
#include <cuda.h>
#include <cudaEGL.h>
#include <gst/allocators/gstdmabuf.h>
#include <gst/app/gstappsink.h>
#include <gst/gst.h>
#include <gst/video/video.h>
#include <NvBufSurface.h>
#endif

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <array>
#include <deque>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace yoloe_native {

using Clock = std::chrono::steady_clock;

struct TrtLogger final : nvinfer1::ILogger {
    void log(Severity severity, char const* msg) noexcept override
    {
        if (severity <= Severity::kWARNING) {
            std::fprintf(stderr, "[TRT] %s\n", msg);
        }
    }
};

inline void throw_if_cuda_failed(cudaError_t status, char const* message)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(status));
    }
}

class CudaEventHandle {
public:
    CudaEventHandle() = default;

    ~CudaEventHandle()
    {
        if (event_ != nullptr) {
            cudaEventDestroy(event_);
            event_ = nullptr;
        }
    }

    CudaEventHandle(CudaEventHandle const&) = delete;
    CudaEventHandle& operator=(CudaEventHandle const&) = delete;

    void create(char const* create_message, unsigned int flags = cudaEventDefault)
    {
        if (event_ != nullptr) {
            cudaEventDestroy(event_);
            event_ = nullptr;
        }
        throw_if_cuda_failed(cudaEventCreateWithFlags(&event_, flags), create_message);
    }

    operator cudaEvent_t() const
    {
        return event_;
    }

    bool valid() const
    {
        return event_ != nullptr;
    }

private:
    cudaEvent_t event_ = nullptr;
};

struct TrtDeleter {
    template <typename T>
    void operator()(T* ptr) const
    {
        if (ptr != nullptr) {
            delete ptr;
        }
    }
};

class CudaBuffer {
public:
    CudaBuffer() = default;

    ~CudaBuffer()
    {
        release();
    }

    CudaBuffer(CudaBuffer const&) = delete;
    CudaBuffer& operator=(CudaBuffer const&) = delete;

    CudaBuffer(CudaBuffer&& other) noexcept
        : ptr_(other.ptr_)
        , bytes_(other.bytes_)
    {
        other.ptr_ = nullptr;
        other.bytes_ = 0;
    }

    CudaBuffer& operator=(CudaBuffer&& other) noexcept
    {
        if (this != &other) {
            release();
            ptr_ = other.ptr_;
            bytes_ = other.bytes_;
            other.ptr_ = nullptr;
            other.bytes_ = 0;
        }
        return *this;
    }

    void ensure(std::size_t bytes)
    {
        if (bytes <= bytes_ && ptr_ != nullptr) {
            return;
        }
        release();
        throw_if_cuda_failed(cudaMalloc(&ptr_, bytes), "cudaMalloc failed");
        bytes_ = bytes;
    }

    void release()
    {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
            ptr_ = nullptr;
            bytes_ = 0;
        }
    }

    void* data() const
    {
        return ptr_;
    }

    std::size_t bytes() const
    {
        return bytes_;
    }

private:
    void* ptr_ = nullptr;
    std::size_t bytes_ = 0;
};

inline std::size_t dtype_size(nvinfer1::DataType dtype)
{
    switch (dtype) {
    case nvinfer1::DataType::kFLOAT:
        return 4;
    case nvinfer1::DataType::kHALF:
        return 2;
    case nvinfer1::DataType::kINT32:
        return 4;
    case nvinfer1::DataType::kINT8:
        return 1;
    case nvinfer1::DataType::kBOOL:
        return 1;
    default:
        throw std::runtime_error("Unsupported TensorRT dtype");
    }
}

inline DLDataType to_dlpack_dtype(nvinfer1::DataType dtype)
{
    switch (dtype) {
    case nvinfer1::DataType::kFLOAT:
        return DLDataType{2, 32, 1};
    case nvinfer1::DataType::kHALF:
        return DLDataType{2, 16, 1};
    case nvinfer1::DataType::kINT32:
        return DLDataType{0, 32, 1};
    case nvinfer1::DataType::kINT8:
        return DLDataType{0, 8, 1};
    case nvinfer1::DataType::kBOOL:
        return DLDataType{1, 1, 1};
    default:
        throw std::runtime_error("Unsupported TensorRT dtype for DLPack");
    }
}

inline bool dims_has_dynamic(nvinfer1::Dims const& dims)
{
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] < 0) {
            return true;
        }
    }
    return false;
}

inline std::vector<std::int64_t> dims_to_shape(nvinfer1::Dims const& dims)
{
    std::vector<std::int64_t> shape;
    shape.reserve(static_cast<std::size_t>(dims.nbDims));
    for (int i = 0; i < dims.nbDims; ++i) {
        shape.push_back(static_cast<std::int64_t>(dims.d[i]));
    }
    return shape;
}

inline std::size_t element_count(nvinfer1::Dims const& dims)
{
    std::size_t count = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        count *= static_cast<std::size_t>(dims.d[i]);
    }
    return count;
}

struct BindingInfo {
    std::string name;
    int index = -1;
    bool is_input = false;
    nvinfer1::DataType dtype = nvinfer1::DataType::kFLOAT;
    nvinfer1::Dims dims{};
};

class NativeMainRuntime;
#ifdef YOLOE_TRT_ENABLE_JETSON_CAMERA
class NativeJetsonCameraSource;
#endif

} // namespace yoloe_native
