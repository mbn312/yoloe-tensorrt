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

namespace {

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

struct ManagedTensorContext {
    std::shared_ptr<void> owner;
    std::shared_ptr<class PendingPostprocessOutputs> pending_postprocess_outputs;
    std::vector<std::int64_t> shape;
    DLDataType dtype{};
    DLManagedTensor managed{};
};

struct NativePostprocessCandidate {
    std::array<float, 4> input_box{};
    std::array<float, 4> output_box{};
    float confidence = 0.0f;
    int cls = 0;
    std::vector<float> mask_coeffs;
};

struct NativePostprocessResult {
    std::vector<float> boxes;
    std::vector<std::uint8_t> masks;
    int mask_height = 0;
    int mask_width = 0;
    bool include_masks = false;
};

struct VisualPromptHostBatch {
    std::vector<float> tensor;
    int prompt_count = 0;
    int prompt_height = 0;
    int prompt_width = 0;
};

inline cudaStream_t parse_dlpack_consumer_stream(py::object const& stream)
{
    if (stream.is_none()) {
        return cudaStreamLegacy;
    }

    auto const value = stream.cast<std::int64_t>();
    if (value == 1) {
        return cudaStreamLegacy;
    }
    if (value == 2) {
        return cudaStreamPerThread;
    }
    if (value == 0 || value < 0) {
        throw std::runtime_error("Invalid CUDA DLPack consumer stream value");
    }
    return reinterpret_cast<cudaStream_t>(static_cast<std::uintptr_t>(value));
}

struct PendingPostprocessOutputs {
    CudaBuffer boxes_buffer;
    CudaBuffer masks_buffer;
    cudaEvent_t ready_event = nullptr;
    int device_id = 0;

    PendingPostprocessOutputs() = default;
    PendingPostprocessOutputs(PendingPostprocessOutputs const&) = delete;
    PendingPostprocessOutputs& operator=(PendingPostprocessOutputs const&) = delete;

    ~PendingPostprocessOutputs()
    {
        if (device_id >= 0) {
            cudaSetDevice(device_id);
        }
        if (ready_event != nullptr) {
            cudaEventDestroy(ready_event);
            ready_event = nullptr;
        }
    }

    void record_ready_event(cudaStream_t stream)
    {
        throw_if_cuda_failed(cudaSetDevice(device_id), "cudaSetDevice failed for postprocess output event");
        if (ready_event == nullptr) {
            throw_if_cuda_failed(
                cudaEventCreateWithFlags(&ready_event, cudaEventDisableTiming),
                "cudaEventCreateWithFlags failed for postprocess output event");
        }
        try {
            throw_if_cuda_failed(
                cudaEventRecord(ready_event, stream),
                "cudaEventRecord failed for postprocess output event");
        } catch (...) {
            throw;
        }
    }

    void wait_on_consumer_stream(cudaStream_t consumer_stream) const
    {
        if (ready_event == nullptr) {
            return;
        }
        throw_if_cuda_failed(cudaSetDevice(device_id), "cudaSetDevice failed for postprocess DLPack handoff");
        throw_if_cuda_failed(
            cudaStreamWaitEvent(consumer_stream, ready_event, 0),
            "cudaStreamWaitEvent failed for postprocess DLPack handoff");
    }

    bool is_ready() const
    {
        if (ready_event == nullptr) {
            return true;
        }
        auto const status = cudaEventQuery(ready_event);
        if (status == cudaSuccess) {
            return true;
        }
        if (status == cudaErrorNotReady) {
            cudaGetLastError();
            return false;
        }
        throw std::runtime_error(std::string("cudaEventQuery failed for postprocess output event: ") + cudaGetErrorString(status));
    }
};

struct NativePostprocessUpload {
    std::shared_ptr<PendingPostprocessOutputs> outputs;
    std::vector<std::int64_t> boxes_shape;
    std::vector<std::int64_t> masks_shape;
    bool include_masks = false;
    double postprocess_ms = 0.0;
};

#ifdef YOLOE_TRT_ENABLE_JETSON_CAMERA
struct PendingJetsonFrame {
    CudaBuffer output_buffer;
    GstSample* sample = nullptr;
    GstBuffer* mapped_buffer = nullptr;
    GstMapInfo mapped_buffer_info{};
    bool buffer_mapped = false;
    NvBufSurface* surface = nullptr;
    bool mapped_egl = false;
    CUgraphicsResource resource = nullptr;
    cudaEvent_t ready_event = nullptr;
    int device_id = 0;

    PendingJetsonFrame() = default;
    PendingJetsonFrame(PendingJetsonFrame const&) = delete;
    PendingJetsonFrame& operator=(PendingJetsonFrame const&) = delete;

    ~PendingJetsonFrame()
    {
        release_frame_resources();
    }

    void record_ready_event(cudaStream_t stream)
    {
        throw_if_cuda_failed(cudaSetDevice(device_id), "cudaSetDevice failed for zero-copy frame event");
        throw_if_cuda_failed(
            cudaEventCreateWithFlags(&ready_event, cudaEventDisableTiming),
            "cudaEventCreateWithFlags failed for zero-copy frame event");
        try {
            throw_if_cuda_failed(
                cudaEventRecord(ready_event, stream),
                "cudaEventRecord failed for zero-copy frame event");
        } catch (...) {
            cudaEventDestroy(ready_event);
            ready_event = nullptr;
            throw;
        }
    }

    void wait_on_consumer_stream(cudaStream_t consumer_stream) const
    {
        if (ready_event == nullptr) {
            return;
        }
        throw_if_cuda_failed(cudaSetDevice(device_id), "cudaSetDevice failed for zero-copy DLPack handoff");
        throw_if_cuda_failed(
            cudaStreamWaitEvent(consumer_stream, ready_event, 0),
            "cudaStreamWaitEvent failed for zero-copy DLPack handoff");
    }

    bool is_ready() const
    {
        if (ready_event == nullptr) {
            return true;
        }
        auto const status = cudaEventQuery(ready_event);
        if (status == cudaSuccess) {
            return true;
        }
        if (status == cudaErrorNotReady) {
            cudaGetLastError();
            return false;
        }
        throw std::runtime_error(std::string("cudaEventQuery failed for zero-copy frame event: ") + cudaGetErrorString(status));
    }

    void release_frame_resources() noexcept
    {
        if (device_id >= 0) {
            cudaSetDevice(device_id);
        }
        if (resource != nullptr) {
            cuGraphicsUnregisterResource(resource);
            resource = nullptr;
        }
        if (mapped_egl && surface != nullptr) {
            NvBufSurfaceUnMapEglImage(surface, 0);
            mapped_egl = false;
        }
        if (buffer_mapped && mapped_buffer != nullptr) {
            gst_buffer_unmap(mapped_buffer, &mapped_buffer_info);
            mapped_buffer = nullptr;
            buffer_mapped = false;
        }
        if (ready_event != nullptr) {
            cudaEventDestroy(ready_event);
            ready_event = nullptr;
        }
        if (sample != nullptr) {
            gst_sample_unref(sample);
            sample = nullptr;
        }
        surface = nullptr;
    }
};

struct ManagedCameraTensorContext {
    std::shared_ptr<NativeJetsonCameraSource> owner;
    std::shared_ptr<PendingJetsonFrame> frame;
    std::vector<std::int64_t> shape;
    DLDataType dtype{};
    DLManagedTensor managed{};
};
#endif

inline float clip_float(float value, float low, float high)
{
    return std::max(low, std::min(value, high));
}

inline void clip_box_inplace(std::array<float, 4>& box, int image_h, int image_w)
{
    box[0] = clip_float(box[0], 0.0f, static_cast<float>(image_w));
    box[1] = clip_float(box[1], 0.0f, static_cast<float>(image_h));
    box[2] = clip_float(box[2], 0.0f, static_cast<float>(image_w));
    box[3] = clip_float(box[3], 0.0f, static_cast<float>(image_h));
}

inline std::array<float, 4> scale_box_to_shape(std::array<float, 4> box, int input_h, int input_w, int output_h, int output_w)
{
    auto const geometry = compute_letterbox_geometry(output_w, output_h, input_w, input_h);
    float const gain = static_cast<float>(geometry.resized_width) / static_cast<float>(output_w);
    float const pad_x = static_cast<float>(geometry.pad_left);
    float const pad_y = static_cast<float>(geometry.pad_top);

    box[0] = (box[0] - pad_x) / gain;
    box[1] = (box[1] - pad_y) / gain;
    box[2] = (box[2] - pad_x) / gain;
    box[3] = (box[3] - pad_y) / gain;
    clip_box_inplace(box, output_h, output_w);
    return box;
}

inline float box_iou(std::array<float, 4> const& lhs, std::array<float, 4> const& rhs)
{
    float const inter_x1 = std::max(lhs[0], rhs[0]);
    float const inter_y1 = std::max(lhs[1], rhs[1]);
    float const inter_x2 = std::min(lhs[2], rhs[2]);
    float const inter_y2 = std::min(lhs[3], rhs[3]);
    float const inter_w = std::max(0.0f, inter_x2 - inter_x1);
    float const inter_h = std::max(0.0f, inter_y2 - inter_y1);
    float const intersection = inter_w * inter_h;

    float const lhs_area = std::max(0.0f, lhs[2] - lhs[0]) * std::max(0.0f, lhs[3] - lhs[1]);
    float const rhs_area = std::max(0.0f, rhs[2] - rhs[0]) * std::max(0.0f, rhs[3] - rhs[1]);
    float const denominator = lhs_area + rhs_area - intersection;
    if (denominator <= 0.0f) {
        return 0.0f;
    }
    return intersection / denominator;
}

inline cv::Mat make_float_mask_view(std::vector<float> const& data, int height, int width)
{
    return cv::Mat(height, width, CV_32FC1, const_cast<float*>(data.data()));
}

inline void crop_mask_inplace(std::vector<float>& mask, int height, int width, std::array<float, 4> const& box)
{
    int const left = std::max(0, std::min(width, static_cast<int>(std::ceil(box[0]))));
    int const top = std::max(0, std::min(height, static_cast<int>(std::ceil(box[1]))));
    int const right = std::max(0, std::min(width, static_cast<int>(std::ceil(box[2]))));
    int const bottom = std::max(0, std::min(height, static_cast<int>(std::ceil(box[3]))));

    if (left >= right || top >= bottom) {
        std::fill(mask.begin(), mask.end(), 0.0f);
        return;
    }

    for (int y = 0; y < top; ++y) {
        std::fill_n(mask.data() + static_cast<std::size_t>(y * width), width, 0.0f);
    }
    for (int y = bottom; y < height; ++y) {
        std::fill_n(mask.data() + static_cast<std::size_t>(y * width), width, 0.0f);
    }
    for (int y = top; y < bottom; ++y) {
        auto* row = mask.data() + static_cast<std::size_t>(y * width);
        std::fill(row, row + left, 0.0f);
        std::fill(row + right, row + width, 0.0f);
    }
}

inline std::vector<float> resize_mask(std::vector<float> const& mask, int src_h, int src_w, int dst_h, int dst_w)
{
    cv::Mat const src = make_float_mask_view(mask, src_h, src_w);
    cv::Mat resized;
    cv::resize(src, resized, cv::Size(dst_w, dst_h), 0.0, 0.0, cv::INTER_LINEAR);
    if (!resized.isContinuous()) {
        resized = resized.clone();
    }
    std::vector<float> result(static_cast<std::size_t>(dst_h) * static_cast<std::size_t>(dst_w));
    std::memcpy(result.data(), resized.ptr<float>(), result.size() * sizeof(float));
    return result;
}

inline std::vector<float> scale_mask_to_shape(std::vector<float> const& mask, int src_h, int src_w, int dst_h, int dst_w)
{
    auto const geometry = compute_letterbox_geometry(dst_w, dst_h, src_w, src_h);
    int const top = std::max(0, std::min(src_h, geometry.pad_top));
    int const left = std::max(0, std::min(src_w, geometry.pad_left));
    int const bottom = std::max(top + 1, std::min(src_h, top + geometry.resized_height));
    int const right = std::max(left + 1, std::min(src_w, left + geometry.resized_width));

    cv::Mat const src = make_float_mask_view(mask, src_h, src_w);
    cv::Mat cropped = src(cv::Range(top, bottom), cv::Range(left, right));
    cv::Mat resized;
    cv::resize(cropped, resized, cv::Size(dst_w, dst_h), 0.0, 0.0, cv::INTER_LINEAR);
    if (!resized.isContinuous()) {
        resized = resized.clone();
    }
    std::vector<float> result(static_cast<std::size_t>(dst_h) * static_cast<std::size_t>(dst_w));
    std::memcpy(result.data(), resized.ptr<float>(), result.size() * sizeof(float));
    return result;
}

inline bool threshold_mask(std::vector<float> const& src, std::vector<std::uint8_t>& dst)
{
    dst.resize(src.size());
    bool any = false;
    for (std::size_t i = 0; i < src.size(); ++i) {
        auto const value = static_cast<std::uint8_t>(src[i] > 0.0f ? 1U : 0U);
        dst[i] = value;
        any = any || value != 0U;
    }
    return any;
}

inline std::vector<int> parse_visual_categories(
    py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> const& category_py)
{
    auto const info = category_py.request();
    if (info.ndim != 1) {
        throw std::runtime_error("visual prompt categories must have shape (N,)");
    }
    auto const count = static_cast<std::size_t>(info.shape[0]);
    auto const* src = static_cast<std::int32_t const*>(info.ptr);
    std::vector<int> categories(count);
    int max_category = -1;
    for (std::size_t index = 0; index < count; ++index) {
        if (src[index] < 0) {
            throw std::runtime_error("visual prompt categories must be non-negative");
        }
        categories[index] = static_cast<int>(src[index]);
        max_category = std::max(max_category, categories[index]);
    }
    if (count == 0 || max_category < 0) {
        throw std::runtime_error("visual prompt categories must not be empty");
    }
    return categories;
}

inline void validate_visual_prompt_count(std::size_t prompt_count, std::vector<int> const& categories)
{
    if (prompt_count != categories.size()) {
        throw std::runtime_error("visual prompt categories must match the number of prompts");
    }
}

inline std::pair<float, float> compute_visual_prompt_letterbox_padding(
    int src_h,
    int src_w,
    int dst_h,
    int dst_w)
{
    float const gain = std::min(static_cast<float>(dst_h) / static_cast<float>(src_h), static_cast<float>(dst_w) / static_cast<float>(src_w));
    float const pad_x = std::round((static_cast<float>(dst_w) - std::round(static_cast<float>(src_w) * gain)) / 2.0f - 0.1f);
    float const pad_y = std::round((static_cast<float>(dst_h) - std::round(static_cast<float>(src_h) * gain)) / 2.0f - 0.1f);
    return { pad_x, pad_y };
}

inline VisualPromptHostBatch build_visual_prompt_batch_from_boxes(
    py::array_t<float, py::array::c_style | py::array::forcecast> const& boxes_py,
    std::vector<int> const& categories,
    int src_h,
    int src_w,
    int dst_h,
    int dst_w,
    int visual_stride)
{
    auto const info = boxes_py.request();
    if (info.ndim != 2 || info.shape[1] != 4) {
        throw std::runtime_error("visual prompt bboxes must have shape (N, 4)");
    }
    auto const prompt_count = static_cast<std::size_t>(info.shape[0]);
    validate_visual_prompt_count(prompt_count, categories);

    int const prompt_height = dst_h / visual_stride;
    int const prompt_width = dst_w / visual_stride;
    int const unique_prompt_count = *std::max_element(categories.begin(), categories.end()) + 1;

    VisualPromptHostBatch batch;
    batch.prompt_count = unique_prompt_count;
    batch.prompt_height = prompt_height;
    batch.prompt_width = prompt_width;
    batch.tensor.assign(
        static_cast<std::size_t>(unique_prompt_count) * static_cast<std::size_t>(prompt_height) * static_cast<std::size_t>(prompt_width),
        0.0f);

    float const gain = std::min(static_cast<float>(dst_h) / static_cast<float>(src_h), static_cast<float>(dst_w) / static_cast<float>(src_w));
    auto const [pad_x, pad_y] = compute_visual_prompt_letterbox_padding(src_h, src_w, dst_h, dst_w);
    auto const* boxes = static_cast<float const*>(info.ptr);

    for (std::size_t index = 0; index < prompt_count; ++index) {
        float const x1 = ((boxes[(index * 4) + 0] * gain) + pad_x) / static_cast<float>(visual_stride);
        float const y1 = ((boxes[(index * 4) + 1] * gain) + pad_y) / static_cast<float>(visual_stride);
        float const x2 = ((boxes[(index * 4) + 2] * gain) + pad_x) / static_cast<float>(visual_stride);
        float const y2 = ((boxes[(index * 4) + 3] * gain) + pad_y) / static_cast<float>(visual_stride);

        int const left = std::max(0, std::min(prompt_width, static_cast<int>(std::ceil(x1))));
        int const top = std::max(0, std::min(prompt_height, static_cast<int>(std::ceil(y1))));
        int const right = std::max(0, std::min(prompt_width, static_cast<int>(std::ceil(x2))));
        int const bottom = std::max(0, std::min(prompt_height, static_cast<int>(std::ceil(y2))));
        if (left >= right || top >= bottom) {
            continue;
        }

        auto const plane_offset =
            static_cast<std::size_t>(categories[index]) * static_cast<std::size_t>(prompt_height) * static_cast<std::size_t>(prompt_width);
        for (int y = top; y < bottom; ++y) {
            auto* row = batch.tensor.data() + plane_offset + static_cast<std::size_t>(y * prompt_width);
            std::fill(row + left, row + right, 1.0f);
        }
    }

    return batch;
}

inline cv::Mat letterbox_mask_nearest(cv::Mat const& src, int dst_h, int dst_w)
{
    float const gain = std::min(static_cast<float>(dst_h) / static_cast<float>(src.rows), static_cast<float>(dst_w) / static_cast<float>(src.cols));
    int const resized_w = std::max(1, static_cast<int>(std::round(static_cast<float>(src.cols) * gain)));
    int const resized_h = std::max(1, static_cast<int>(std::round(static_cast<float>(src.rows) * gain)));

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(resized_w, resized_h), 0.0, 0.0, cv::INTER_NEAREST);

    float const dw = static_cast<float>(dst_w - resized_w) / 2.0f;
    float const dh = static_cast<float>(dst_h - resized_h) / 2.0f;
    int const top = std::round(dh - 0.1f);
    int const bottom = std::round(dh + 0.1f);
    int const left = std::round(dw - 0.1f);
    int const right = std::round(dw + 0.1f);

    cv::Mat output;
    cv::copyMakeBorder(resized, output, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(0));
    return output;
}

inline VisualPromptHostBatch build_visual_prompt_batch_from_masks(
    py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> const& masks_py,
    std::vector<int> const& categories,
    int dst_h,
    int dst_w,
    int visual_stride)
{
    auto const info = masks_py.request();
    if (info.ndim != 3) {
        throw std::runtime_error("visual prompt masks must have shape (N, H, W)");
    }
    auto const prompt_count = static_cast<std::size_t>(info.shape[0]);
    validate_visual_prompt_count(prompt_count, categories);

    int const src_h = static_cast<int>(info.shape[1]);
    int const src_w = static_cast<int>(info.shape[2]);
    int const prompt_height = dst_h / visual_stride;
    int const prompt_width = dst_w / visual_stride;
    int const unique_prompt_count = *std::max_element(categories.begin(), categories.end()) + 1;

    VisualPromptHostBatch batch;
    batch.prompt_count = unique_prompt_count;
    batch.prompt_height = prompt_height;
    batch.prompt_width = prompt_width;
    batch.tensor.assign(
        static_cast<std::size_t>(unique_prompt_count) * static_cast<std::size_t>(prompt_height) * static_cast<std::size_t>(prompt_width),
        0.0f);

    auto const* src = static_cast<std::uint8_t const*>(info.ptr);
    auto const mask_plane = static_cast<std::size_t>(src_h) * static_cast<std::size_t>(src_w);
    for (std::size_t index = 0; index < prompt_count; ++index) {
        cv::Mat const mask(src_h, src_w, CV_8UC1, const_cast<std::uint8_t*>(src + (index * mask_plane)));
        cv::Mat const letterboxed = letterbox_mask_nearest(mask, dst_h, dst_w);
        cv::Mat resized;
        cv::resize(letterboxed, resized, cv::Size(prompt_width, prompt_height), 0.0, 0.0, cv::INTER_NEAREST);
        auto const plane_offset =
            static_cast<std::size_t>(categories[index]) * static_cast<std::size_t>(prompt_height) * static_cast<std::size_t>(prompt_width);
        for (int y = 0; y < prompt_height; ++y) {
            auto const* row = resized.ptr<std::uint8_t>(y);
            auto* dst_row = batch.tensor.data() + plane_offset + static_cast<std::size_t>(y * prompt_width);
            for (int x = 0; x < prompt_width; ++x) {
                dst_row[x] = (dst_row[x] != 0.0f || row[x] != 0U) ? 1.0f : 0.0f;
            }
        }
    }

    return batch;
}

class CudaTensorView {
public:
    CudaTensorView(
        std::shared_ptr<void> owner,
        std::shared_ptr<PendingPostprocessOutputs> pending_postprocess_outputs,
        void* ptr,
        std::vector<std::int64_t> shape,
        nvinfer1::DataType dtype,
        int device_id)
        : CudaTensorView(
              std::move(owner),
              std::move(pending_postprocess_outputs),
              ptr,
              std::move(shape),
              to_dlpack_dtype(dtype),
              device_id)
    {
    }

    CudaTensorView(
        std::shared_ptr<void> owner,
        std::shared_ptr<PendingPostprocessOutputs> pending_postprocess_outputs,
        void* ptr,
        std::vector<std::int64_t> shape,
        DLDataType dtype,
        int device_id)
        : owner_(std::move(owner))
        , pending_postprocess_outputs_(std::move(pending_postprocess_outputs))
        , ptr_(ptr)
        , shape_(std::move(shape))
        , dtype_(dtype)
        , device_id_(device_id)
    {
    }

    py::capsule dlpack(py::object stream) const
    {
        if (pending_postprocess_outputs_ != nullptr) {
            pending_postprocess_outputs_->wait_on_consumer_stream(parse_dlpack_consumer_stream(stream));
        }
        auto* ctx = new ManagedTensorContext();
        ctx->owner = owner_;
        ctx->pending_postprocess_outputs = pending_postprocess_outputs_;
        ctx->shape = shape_;
        ctx->dtype = dtype_;
        ctx->managed.manager_ctx = ctx;
        ctx->managed.deleter = [](DLManagedTensor* self) {
            auto* managed_ctx = static_cast<ManagedTensorContext*>(self->manager_ctx);
            delete managed_ctx;
        };
        ctx->managed.dl_tensor.data = ptr_;
        ctx->managed.dl_tensor.device = DLDevice{kDLCUDA, device_id_};
        ctx->managed.dl_tensor.ndim = static_cast<int>(ctx->shape.size());
        ctx->managed.dl_tensor.dtype = ctx->dtype;
        ctx->managed.dl_tensor.shape = ctx->shape.data();
        ctx->managed.dl_tensor.strides = nullptr;
        ctx->managed.dl_tensor.byte_offset = 0;

        py::capsule capsule(&ctx->managed, "dltensor", [](PyObject* obj) {
            if (PyCapsule_IsValid(obj, "dltensor")) {
                auto* managed = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(obj, "dltensor"));
                if (managed != nullptr && managed->deleter != nullptr) {
                    managed->deleter(managed);
                }
            }
        });
        return capsule;
    }

    py::tuple dlpack_device() const
    {
        return py::make_tuple(static_cast<int>(kDLCUDA), device_id_);
    }

private:
    std::shared_ptr<void> owner_;
    std::shared_ptr<PendingPostprocessOutputs> pending_postprocess_outputs_;
    void* ptr_ = nullptr;
    std::vector<std::int64_t> shape_;
    DLDataType dtype_{2, 32, 1};
    int device_id_ = 0;
};

#ifdef YOLOE_TRT_ENABLE_JETSON_CAMERA
class CameraCudaTensorView {
public:
    CameraCudaTensorView(
        std::shared_ptr<NativeJetsonCameraSource> owner,
        std::shared_ptr<PendingJetsonFrame> frame,
        void* ptr,
        std::vector<std::int64_t> shape,
        nvinfer1::DataType dtype,
        int device_id)
        : CameraCudaTensorView(std::move(owner), std::move(frame), ptr, std::move(shape), to_dlpack_dtype(dtype), device_id)
    {
    }

    CameraCudaTensorView(
        std::shared_ptr<NativeJetsonCameraSource> owner,
        std::shared_ptr<PendingJetsonFrame> frame,
        void* ptr,
        std::vector<std::int64_t> shape,
        DLDataType dtype,
        int device_id)
        : owner_(std::move(owner))
        , frame_(std::move(frame))
        , ptr_(ptr)
        , shape_(std::move(shape))
        , dtype_(dtype)
        , device_id_(device_id)
    {
    }

    py::capsule dlpack(py::object stream) const
    {
        if (frame_ != nullptr) {
            frame_->wait_on_consumer_stream(parse_dlpack_consumer_stream(stream));
        }
        auto* ctx = new ManagedCameraTensorContext();
        ctx->owner = owner_;
        ctx->frame = frame_;
        ctx->shape = shape_;
        ctx->dtype = dtype_;
        ctx->managed.manager_ctx = ctx;
        ctx->managed.deleter = [](DLManagedTensor* self) {
            auto* managed_ctx = static_cast<ManagedCameraTensorContext*>(self->manager_ctx);
            delete managed_ctx;
        };
        ctx->managed.dl_tensor.data = ptr_;
        ctx->managed.dl_tensor.device = DLDevice{kDLCUDA, device_id_};
        ctx->managed.dl_tensor.ndim = static_cast<int>(ctx->shape.size());
        ctx->managed.dl_tensor.dtype = ctx->dtype;
        ctx->managed.dl_tensor.shape = ctx->shape.data();
        ctx->managed.dl_tensor.strides = nullptr;
        ctx->managed.dl_tensor.byte_offset = 0;

        py::capsule capsule(&ctx->managed, "dltensor", [](PyObject* obj) {
            if (PyCapsule_IsValid(obj, "dltensor")) {
                auto* managed = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(obj, "dltensor"));
                if (managed != nullptr && managed->deleter != nullptr) {
                    managed->deleter(managed);
                }
            }
        });
        return capsule;
    }

    py::tuple dlpack_device() const
    {
        return py::make_tuple(static_cast<int>(kDLCUDA), device_id_);
    }

private:
    std::shared_ptr<NativeJetsonCameraSource> owner_;
    std::shared_ptr<PendingJetsonFrame> frame_;
    void* ptr_ = nullptr;
    std::vector<std::int64_t> shape_;
    DLDataType dtype_{2, 32, 1};
    int device_id_ = 0;
};
#endif

class NativeMainRuntime : public std::enable_shared_from_this<NativeMainRuntime> {
public:
    NativeMainRuntime(std::string engine_path, std::string image_input_name, std::string prompt_input_name, int device_id)
        : engine_path_(std::move(engine_path))
        , image_input_name_(std::move(image_input_name))
        , prompt_input_name_(std::move(prompt_input_name))
        , device_id_(device_id)
        , runtime_(nullptr)
        , engine_(nullptr)
        , context_(nullptr)
    {
        throw_if_cuda_failed(cudaSetDevice(device_id_), "cudaSetDevice failed");
        throw_if_cuda_failed(cudaStreamCreate(&stream_), "cudaStreamCreate failed");
        preprocess_start_event_.create("cudaEventCreate failed for preprocess start");
        preprocess_end_event_.create("cudaEventCreate failed for preprocess end");
        inference_start_event_.create("cudaEventCreate failed for inference start");
        inference_end_event_.create("cudaEventCreate failed for inference end");
        producer_handoff_event_.create("cudaEventCreate failed for prepared tensor handoff", cudaEventDisableTiming);

        py::gil_scoped_release release;

        std::FILE* file = std::fopen(engine_path_.c_str(), "rb");
        if (file == nullptr) {
            throw std::runtime_error("Failed to open TensorRT engine: " + engine_path_);
        }
        std::fseek(file, 0, SEEK_END);
        auto const size = static_cast<std::size_t>(std::ftell(file));
        std::rewind(file);
        std::vector<char> bytes(size);
        auto const read = std::fread(bytes.data(), 1, size, file);
        std::fclose(file);
        if (read != size) {
            throw std::runtime_error("Failed to read TensorRT engine: " + engine_path_);
        }

        runtime_.reset(nvinfer1::createInferRuntime(logger_));
        if (!runtime_) {
            throw std::runtime_error("Failed to create TensorRT runtime");
        }
        engine_.reset(runtime_->deserializeCudaEngine(bytes.data(), bytes.size()));
        if (!engine_) {
            throw std::runtime_error("Failed to deserialize TensorRT engine");
        }
        context_.reset(engine_->createExecutionContext());
        if (!context_) {
            throw std::runtime_error("Failed to create TensorRT execution context");
        }

        discover_bindings();
        image_input_index_ = find_binding_index(image_input_name_, true);
        prompt_input_index_ = find_binding_index(prompt_input_name_, true);

        image_binding_ = bindings_[image_input_index_];
        prompt_binding_ = bindings_[prompt_input_index_];
        fp16_ = image_binding_.dtype == nvinfer1::DataType::kHALF;
        prompt_embed_dim_ = prompt_binding_.dims.nbDims >= 3 ? prompt_binding_.dims.d[2] : 0;

        if (engine_->getNbOptimizationProfiles() > 0 && context_->getOptimizationProfile() != 0) {
            if (!context_->setOptimizationProfileAsync(0, stream_)) {
                throw std::runtime_error("Failed to select TensorRT optimization profile 0");
            }
            throw_if_cuda_failed(
                cudaStreamSynchronize(stream_),
                "cudaStreamSynchronize failed after selecting optimization profile");
        }
    }

    ~NativeMainRuntime() noexcept
    {
        if (device_id_ >= 0) {
            cudaSetDevice(device_id_);
        }
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
    }

    std::vector<std::string> output_names() const
    {
        return output_names_;
    }

    bool fp16() const
    {
        return fp16_;
    }

    bool prompt_fp16() const
    {
        return prompt_binding_.dtype == nvinfer1::DataType::kHALF;
    }

    void clear_prompt_embeddings()
    {
        has_prompt_embeddings_ = false;
        current_prompt_count_ = 0;
    }

    void set_prompt_embeddings(py::array prompt_embeddings_py)
    {
        activate_device("cudaSetDevice failed before prompt upload");
        auto prompt_embeddings = py::array_t<float, py::array::c_style | py::array::forcecast>(prompt_embeddings_py);
        auto const info = prompt_embeddings.request();
        if (info.ndim != 2 && info.ndim != 3) {
            throw std::runtime_error("prompt_embeddings must have shape (N, E) or (1, N, E)");
        }

        std::int64_t const batch = info.ndim == 3 ? info.shape[0] : 1;
        std::int64_t const prompt_count = info.ndim == 3 ? info.shape[1] : info.shape[0];
        std::int64_t const embed_dim = info.ndim == 3 ? info.shape[2] : info.shape[1];

        if (batch != 1) {
            throw std::runtime_error("prompt_embeddings batch dimension must be 1");
        }
        if (embed_dim != prompt_embed_dim_) {
            throw std::runtime_error("prompt_embeddings embed dim does not match engine profile");
        }

        auto const element_total = static_cast<std::size_t>(batch * prompt_count * embed_dim);
        auto const* src = static_cast<float const*>(info.ptr);

        if (prompt_binding_.dtype == nvinfer1::DataType::kHALF) {
            prompt_half_host_.resize(element_total);
            for (std::size_t i = 0; i < element_total; ++i) {
                prompt_half_host_[i] = __float2half(src[i]);
            }
            prompt_buffer_.ensure(prompt_half_host_.size() * sizeof(__half));
            throw_if_cuda_failed(
                cudaMemcpyAsync(
                    prompt_buffer_.data(),
                    prompt_half_host_.data(),
                    prompt_half_host_.size() * sizeof(__half),
                    cudaMemcpyHostToDevice,
                    stream_),
                "cudaMemcpyAsync for prompt embeddings failed");
        } else {
            prompt_buffer_.ensure(element_total * sizeof(float));
            throw_if_cuda_failed(
                cudaMemcpyAsync(
                    prompt_buffer_.data(),
                    src,
                    element_total * sizeof(float),
                    cudaMemcpyHostToDevice,
                    stream_),
                "cudaMemcpyAsync for prompt embeddings failed");
        }
        throw_if_cuda_failed(cudaStreamSynchronize(stream_), "cudaStreamSynchronize failed after prompt upload");

        has_prompt_embeddings_ = true;
        current_prompt_count_ = static_cast<int>(prompt_count);
    }

    void set_prompt_embeddings_device(
        std::uintptr_t prompt_embeddings_ptr_value,
        int prompt_count,
        int embed_dim,
        std::uintptr_t producer_stream_value = 0)
    {
        if (prompt_embeddings_ptr_value == 0) {
            throw std::runtime_error("prompt_embeddings pointer must not be null");
        }
        if (prompt_count < 0) {
            throw std::runtime_error("prompt_count must be non-negative");
        }
        if (embed_dim != prompt_embed_dim_) {
            throw std::runtime_error("prompt_embeddings embed dim does not match engine profile");
        }

        auto* prompt_embeddings_ptr = reinterpret_cast<void*>(prompt_embeddings_ptr_value);
        auto* producer_stream = reinterpret_cast<cudaStream_t>(producer_stream_value);
        activate_device("cudaSetDevice failed before device prompt upload");
        wait_for_producer_stream(producer_stream);

        auto const element_total = static_cast<std::size_t>(prompt_count) * static_cast<std::size_t>(embed_dim);
        auto const bytes = element_total * dtype_size(prompt_binding_.dtype);
        prompt_buffer_.ensure(std::max<std::size_t>(bytes, sizeof(float)));
        if (bytes > 0) {
            throw_if_cuda_failed(
                cudaMemcpyAsync(prompt_buffer_.data(), prompt_embeddings_ptr, bytes, cudaMemcpyDeviceToDevice, stream_),
                "cudaMemcpyAsync for device prompt embeddings failed");
        }
        throw_if_cuda_failed(cudaStreamSynchronize(stream_), "cudaStreamSynchronize failed after device prompt upload");

        has_prompt_embeddings_ = true;
        current_prompt_count_ = prompt_count;
    }

    py::dict infer_image(py::array image_py, int target_h, int target_w)
    {
        if (!has_prompt_embeddings_) {
            throw std::runtime_error("No prompt embeddings are active");
        }

        auto image = py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>(image_py);
        auto const info = image.request();
        if (info.ndim != 3 || info.shape[2] != 3) {
            throw std::runtime_error("image must have shape (H, W, 3)");
        }

        activate_device("cudaSetDevice failed before host image inference");
        preprocess_host_image(image, target_h, target_w);
        auto const inference_ms = execute(image_buffer_.data(), target_h, target_w, prompt_buffer_.data(), current_prompt_count_);
        auto const preprocess_ms = resolve_preprocess_ms();

        return build_output_dict(target_h, target_w, preprocess_ms, inference_ms);
    }

    py::dict infer_image_postprocessed(
        py::array image_py,
        int target_h,
        int target_w,
        int original_h,
        int original_w,
        float conf,
        float iou,
        int max_det,
        bool retina_masks)
    {
        if (!has_prompt_embeddings_) {
            throw std::runtime_error("No prompt embeddings are active");
        }

        auto image = py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>(image_py);
        auto const info = image.request();
        if (info.ndim != 3 || info.shape[2] != 3) {
            throw std::runtime_error("image must have shape (H, W, 3)");
        }

        activate_device("cudaSetDevice failed before native postprocessed image inference");
        preprocess_host_image(image, target_h, target_w);
        auto const inference_ms = execute(image_buffer_.data(), target_h, target_w, prompt_buffer_.data(), current_prompt_count_);
        auto const preprocess_ms = resolve_preprocess_ms();
        return finish_postprocessed_inference(
            target_h,
            target_w,
            original_h,
            original_w,
            conf,
            iou,
            max_det,
            retina_masks,
            preprocess_ms,
            inference_ms);
    }

    py::dict _preprocess_image_to_tensor(py::array image_py, int target_h, int target_w)
    {
        auto image = py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>(image_py);
        auto const info = image.request();
        if (info.ndim != 3 || info.shape[2] != 3) {
            throw std::runtime_error("image must have shape (H, W, 3)");
        }

        activate_device("cudaSetDevice failed before internal preprocess hook");
        preprocess_host_image(image, target_h, target_w);
        throw_if_cuda_failed(cudaStreamSynchronize(stream_), "cudaStreamSynchronize failed after internal preprocess hook");
        auto const preprocess_ms = resolve_preprocess_ms();

        py::dict outputs;
        outputs[py::str("tensor")] = py::cast(
            CudaTensorView(
                shared_from_this(),
                nullptr,
                image_buffer_.data(),
                std::vector<std::int64_t>{ 1, 3, target_h, target_w },
                image_binding_.dtype,
                device_id_));
        outputs[py::str("input_shape")] = py::make_tuple(target_h, target_w);
        outputs[py::str("preprocess_ms")] = preprocess_ms;
        return outputs;
    }

    py::dict infer_tensor(std::uintptr_t image_ptr_value, int target_h, int target_w, std::uintptr_t producer_stream_value = 0)
    {
        if (!has_prompt_embeddings_) {
            throw std::runtime_error("No prompt embeddings are active");
        }
        if (image_ptr_value == 0) {
            throw std::runtime_error("image tensor pointer must not be null");
        }

        auto* image_ptr = reinterpret_cast<void*>(image_ptr_value);
        auto* producer_stream = reinterpret_cast<cudaStream_t>(producer_stream_value);
        activate_device("cudaSetDevice failed before prepared tensor inference");
        wait_for_producer_stream(producer_stream);
        auto const inference_ms = execute(image_ptr, target_h, target_w, prompt_buffer_.data(), current_prompt_count_);

        return build_output_dict(target_h, target_w, 0.0, inference_ms);
    }

    py::dict infer_tensor_postprocessed(
        std::uintptr_t image_ptr_value,
        int target_h,
        int target_w,
        int original_h,
        int original_w,
        float conf,
        float iou,
        int max_det,
        bool retina_masks,
        std::uintptr_t producer_stream_value = 0)
    {
        if (!has_prompt_embeddings_) {
            throw std::runtime_error("No prompt embeddings are active");
        }
        if (image_ptr_value == 0) {
            throw std::runtime_error("image tensor pointer must not be null");
        }

        auto* image_ptr = reinterpret_cast<void*>(image_ptr_value);
        auto* producer_stream = reinterpret_cast<cudaStream_t>(producer_stream_value);
        activate_device("cudaSetDevice failed before native postprocessed tensor inference");
        wait_for_producer_stream(producer_stream);
        auto const inference_ms = execute(image_ptr, target_h, target_w, prompt_buffer_.data(), current_prompt_count_);
        return finish_postprocessed_inference(
            target_h,
            target_w,
            original_h,
            original_w,
            conf,
            iou,
            max_det,
            retina_masks,
            0.0,
            inference_ms);
    }

    void warmup(int target_h, int target_w, int prompt_count, int runs)
    {
        if (runs < 1) {
            runs = 1;
        }

        activate_device("cudaSetDevice failed before warmup");

        CudaBuffer warm_prompt_buffer;
        void* prompt_ptr = prompt_buffer_.data();
        int prompt_count_to_use = current_prompt_count_;

        if (!has_prompt_embeddings_ || current_prompt_count_ != prompt_count) {
            auto const prompt_elements = static_cast<std::size_t>(prompt_count) * static_cast<std::size_t>(prompt_embed_dim_);
            auto const prompt_bytes = prompt_elements * dtype_size(prompt_binding_.dtype);
            warm_prompt_buffer.ensure(prompt_bytes);
            throw_if_cuda_failed(cudaMemsetAsync(warm_prompt_buffer.data(), 0, prompt_bytes, stream_), "cudaMemsetAsync failed");
            prompt_ptr = warm_prompt_buffer.data();
            prompt_count_to_use = prompt_count;
        }

        auto const image_elements = static_cast<std::size_t>(3 * target_h * target_w);
        image_buffer_.ensure(image_elements * dtype_size(image_binding_.dtype));
        throw_if_cuda_failed(cudaMemsetAsync(image_buffer_.data(), 0, image_buffer_.bytes(), stream_), "cudaMemsetAsync failed");
        throw_if_cuda_failed(cudaStreamSynchronize(stream_), "cudaStreamSynchronize failed before warmup");

        for (int i = 0; i < runs; ++i) {
            static_cast<void>(execute(image_buffer_.data(), target_h, target_w, prompt_ptr, prompt_count_to_use));
        }
    }

private:
    void activate_device(char const* message)
    {
        throw_if_cuda_failed(cudaSetDevice(device_id_), message);
    }

    double elapsed_event_ms(cudaEvent_t start_event, cudaEvent_t end_event, char const* context) const
    {
        float elapsed_ms = 0.0f;
        throw_if_cuda_failed(
            cudaEventElapsedTime(&elapsed_ms, start_event, end_event),
            (std::string(context) + ": cudaEventElapsedTime failed").c_str());
        return static_cast<double>(elapsed_ms);
    }

    double resolve_preprocess_ms()
    {
#ifdef YOLOE_TRT_ENABLE_CUDA_PREPROCESS
        return elapsed_event_ms(preprocess_start_event_, preprocess_end_event_, "host image preprocess");
#else
        return last_preprocess_cpu_ms_ + elapsed_event_ms(preprocess_start_event_, preprocess_end_event_, "host image upload");
#endif
    }

    void wait_for_producer_stream(cudaStream_t producer_stream)
    {
        if (producer_stream == nullptr || producer_stream == stream_) {
            return;
        }

        throw_if_cuda_failed(
            cudaEventRecord(producer_handoff_event_, producer_stream),
            "cudaEventRecord failed for prepared tensor handoff");
        throw_if_cuda_failed(
            cudaStreamWaitEvent(stream_, producer_handoff_event_, 0),
            "cudaStreamWaitEvent failed for prepared tensor handoff");
    }

    py::dict build_output_dict(int target_h, int target_w, double preprocess_ms, double inference_ms)
    {
        py::dict outputs;
        for (std::size_t i = 0; i < output_indices_.size(); ++i) {
            auto const binding_index = output_indices_[i];
            auto const& binding = bindings_[binding_index];
            outputs[py::str(binding.name)] = py::cast(
                CudaTensorView(shared_from_this(), nullptr, output_buffers_[i].data(), output_shapes_[i], binding.dtype, device_id_));
        }
        outputs[py::str("input_shape")] = py::make_tuple(target_h, target_w);
        outputs[py::str("preprocess_ms")] = preprocess_ms;
        outputs[py::str("inference_ms")] = inference_ms;
        return outputs;
    }

    py::dict finish_postprocessed_inference(
        int target_h,
        int target_w,
        int original_h,
        int original_w,
        float conf_threshold,
        float iou_threshold,
        int max_det,
        bool retina_masks,
        double preprocess_ms,
        double inference_ms)
    {
        auto upload = postprocess_outputs(
            target_h,
            target_w,
            original_h,
            original_w,
            conf_threshold,
            iou_threshold,
            max_det,
            retina_masks);
        return build_postprocessed_output_dict(target_h, target_w, preprocess_ms, inference_ms, upload);
    }

    py::dict build_postprocessed_output_dict(
        int target_h,
        int target_w,
        double preprocess_ms,
        double inference_ms,
        NativePostprocessUpload const& upload)
    {
        py::dict outputs;
        outputs[py::str("boxes")] = py::cast(
            CudaTensorView(
                shared_from_this(),
                upload.outputs,
                upload.outputs->boxes_buffer.data(),
                upload.boxes_shape,
                DLDataType{2, 32, 1},
                device_id_));
        if (upload.include_masks) {
            outputs[py::str("masks")] = py::cast(
                CudaTensorView(
                    shared_from_this(),
                    upload.outputs,
                    upload.outputs->masks_buffer.data(),
                    upload.masks_shape,
                    DLDataType{1, 8, 1},
                    device_id_));
        }
        outputs[py::str("input_shape")] = py::make_tuple(target_h, target_w);
        outputs[py::str("preprocess_ms")] = preprocess_ms;
        outputs[py::str("inference_ms")] = inference_ms;
        outputs[py::str("postprocess_ms")] = upload.postprocess_ms;
        return outputs;
    }

    NativePostprocessUpload upload_postprocess_result(NativePostprocessResult const& result)
    {
        NativePostprocessUpload upload;
        upload.outputs = std::make_shared<PendingPostprocessOutputs>();
        upload.outputs->device_id = device_id_;
        upload.include_masks = result.include_masks;
        upload.boxes_shape = { static_cast<std::int64_t>(result.boxes.size() / 6), 6 };

        auto const box_bytes = std::max<std::size_t>(sizeof(float), result.boxes.size() * sizeof(float));
        upload.outputs->boxes_buffer.ensure(box_bytes);
        if (!result.boxes.empty()) {
            throw_if_cuda_failed(
                cudaMemcpyAsync(
                    upload.outputs->boxes_buffer.data(),
                    result.boxes.data(),
                    result.boxes.size() * sizeof(float),
                    cudaMemcpyHostToDevice,
                    stream_),
                "cudaMemcpyAsync failed for native postprocessed boxes");
        }

        if (result.include_masks) {
            upload.masks_shape = {
                static_cast<std::int64_t>(result.boxes.size() / 6),
                static_cast<std::int64_t>(result.mask_height),
                static_cast<std::int64_t>(result.mask_width),
            };
            auto const mask_bytes = std::max<std::size_t>(sizeof(std::uint8_t), result.masks.size() * sizeof(std::uint8_t));
            upload.outputs->masks_buffer.ensure(mask_bytes);
            if (!result.masks.empty()) {
                throw_if_cuda_failed(
                    cudaMemcpyAsync(
                        upload.outputs->masks_buffer.data(),
                        result.masks.data(),
                        result.masks.size() * sizeof(std::uint8_t),
                        cudaMemcpyHostToDevice,
                        stream_),
                    "cudaMemcpyAsync failed for native postprocessed masks");
            }
        }

        upload.outputs->record_ready_event(stream_);
        return upload;
    }

    std::vector<float> copy_output_buffer_to_float(std::size_t output_position)
    {
        auto const binding_index = output_indices_[output_position];
        auto const& binding = bindings_[binding_index];
        auto const count = std::accumulate(
            output_shapes_[output_position].begin(),
            output_shapes_[output_position].end(),
            static_cast<std::size_t>(1),
            [](std::size_t lhs, std::int64_t rhs) { return lhs * static_cast<std::size_t>(rhs); });

        std::vector<float> host_output(count);
        if (binding.dtype == nvinfer1::DataType::kFLOAT) {
            if (count > 0) {
                throw_if_cuda_failed(
                    cudaMemcpy(host_output.data(), output_buffers_[output_position].data(), count * sizeof(float), cudaMemcpyDeviceToHost),
                    "cudaMemcpy failed for TensorRT output");
            }
            return host_output;
        }
        if (binding.dtype == nvinfer1::DataType::kHALF) {
            std::vector<__half> host_half(count);
            if (count > 0) {
                throw_if_cuda_failed(
                    cudaMemcpy(host_half.data(), output_buffers_[output_position].data(), count * sizeof(__half), cudaMemcpyDeviceToHost),
                    "cudaMemcpy failed for TensorRT half output");
            }
            for (std::size_t i = 0; i < count; ++i) {
                host_output[i] = __half2float(host_half[i]);
            }
            return host_output;
        }
        throw std::runtime_error("Native postprocess only supports float/half TensorRT outputs");
    }

    bool task_is_segment() const
    {
        return output_shapes_.size() == 2;
    }

    bool output_layout_is_end2end() const
    {
        return output_shapes_.size() >= 1 && output_shapes_[0].size() == 3 && output_shapes_[0][1] > output_shapes_[0][2];
    }

    int mask_dim() const
    {
        if (!task_is_segment()) {
            return 0;
        }
        if (output_shapes_[1].size() < 4) {
            throw std::runtime_error("Segmentation proto output shape is invalid for native postprocess");
        }
        return static_cast<int>(output_shapes_[1][1]);
    }

    std::vector<NativePostprocessCandidate> decode_end2end_candidates(
        std::vector<float> const& predictions,
        float conf_threshold,
        int max_det)
    {
        if (max_det <= 0) {
            return {};
        }
        auto const& shape = output_shapes_[0];
        int const det_count = static_cast<int>(shape[1]);
        int const det_width = static_cast<int>(shape[2]);
        int const extra = std::max(0, det_width - 6);
        std::vector<NativePostprocessCandidate> detections;
        detections.reserve(static_cast<std::size_t>(std::min(det_count, max_det)));

        for (int index = 0; index < det_count; ++index) {
            auto const* row = predictions.data() + static_cast<std::size_t>(index * det_width);
            if (row[4] <= conf_threshold) {
                continue;
            }
            NativePostprocessCandidate detection;
            detection.input_box = { row[0], row[1], row[2], row[3] };
            detection.output_box = detection.input_box;
            detection.confidence = row[4];
            detection.cls = static_cast<int>(row[5]);
            if (extra > 0) {
                detection.mask_coeffs.assign(row + 6, row + 6 + extra);
            }
            detections.push_back(std::move(detection));
            if (static_cast<int>(detections.size()) >= max_det) {
                break;
            }
        }
        return detections;
    }

    std::vector<NativePostprocessCandidate> decode_raw_candidates(
        std::vector<float> const& predictions,
        float conf_threshold,
        float iou_threshold,
        int max_det)
    {
        if (max_det <= 0) {
            return {};
        }
        auto const& shape = output_shapes_[0];
        int const channels = static_cast<int>(shape[1]);
        int const box_count = static_cast<int>(shape[2]);
        int const extra = mask_dim();
        int const class_count = channels - 4 - extra;
        if (class_count <= 0) {
            throw std::runtime_error("TensorRT output does not expose a valid class dimension for native postprocess");
        }

        std::vector<NativePostprocessCandidate> candidates;
        candidates.reserve(static_cast<std::size_t>(box_count));
        for (int box_index = 0; box_index < box_count; ++box_index) {
            auto const offset = [box_count](int channel) {
                return static_cast<std::size_t>(channel * box_count);
            };
            float best_score = -1.0f;
            int best_class = -1;
            for (int cls_index = 0; cls_index < class_count; ++cls_index) {
                auto const score = predictions[offset(4 + cls_index) + static_cast<std::size_t>(box_index)];
                if (score > best_score) {
                    best_score = score;
                    best_class = cls_index;
                }
            }
            if (best_score <= conf_threshold) {
                continue;
            }

            float const cx = predictions[offset(0) + static_cast<std::size_t>(box_index)];
            float const cy = predictions[offset(1) + static_cast<std::size_t>(box_index)];
            float const width = predictions[offset(2) + static_cast<std::size_t>(box_index)];
            float const height = predictions[offset(3) + static_cast<std::size_t>(box_index)];

            NativePostprocessCandidate detection;
            detection.input_box = {
                cx - (width * 0.5f),
                cy - (height * 0.5f),
                cx + (width * 0.5f),
                cy + (height * 0.5f),
            };
            detection.output_box = detection.input_box;
            detection.confidence = best_score;
            detection.cls = best_class;
            if (extra > 0) {
                detection.mask_coeffs.resize(static_cast<std::size_t>(extra));
                for (int extra_index = 0; extra_index < extra; ++extra_index) {
                    detection.mask_coeffs[static_cast<std::size_t>(extra_index)] =
                        predictions[offset(4 + class_count + extra_index) + static_cast<std::size_t>(box_index)];
                }
            }
            candidates.push_back(std::move(detection));
        }

        std::stable_sort(
            candidates.begin(),
            candidates.end(),
            [](NativePostprocessCandidate const& lhs, NativePostprocessCandidate const& rhs) {
                return lhs.confidence > rhs.confidence;
            });

        constexpr std::size_t kMaxNativeNms = 30000;
        if (candidates.size() > kMaxNativeNms) {
            candidates.resize(kMaxNativeNms);
        }

        std::vector<NativePostprocessCandidate> kept;
        kept.reserve(static_cast<std::size_t>(max_det));
        for (auto const& candidate : candidates) {
            bool suppressed = false;
            for (auto const& selected : kept) {
                if (box_iou(candidate.input_box, selected.input_box) > iou_threshold) {
                    suppressed = true;
                    break;
                }
            }
            if (suppressed) {
                continue;
            }
            kept.push_back(candidate);
            if (static_cast<int>(kept.size()) >= max_det) {
                break;
            }
        }
        return kept;
    }

    std::vector<float> reconstruct_mask_logits(
        std::vector<float> const& proto,
        std::vector<float> const& coeffs,
        int proto_h,
        int proto_w) const
    {
        int const channels = static_cast<int>(coeffs.size());
        std::vector<float> mask(static_cast<std::size_t>(proto_h) * static_cast<std::size_t>(proto_w), 0.0f);
        for (int channel = 0; channel < channels; ++channel) {
            auto const coeff = coeffs[static_cast<std::size_t>(channel)];
            auto const* proto_channel = proto.data() + static_cast<std::size_t>(channel * proto_h * proto_w);
            for (std::size_t index = 0; index < mask.size(); ++index) {
                mask[index] += coeff * proto_channel[index];
            }
        }
        return mask;
    }

    std::vector<std::uint8_t> build_mask_for_detection(
        NativePostprocessCandidate const& detection,
        std::vector<float> const& proto,
        int input_h,
        int input_w,
        int original_h,
        int original_w,
        bool retina_masks) const
    {
        int const proto_channels = mask_dim();
        int const proto_h = static_cast<int>(output_shapes_[1][2]);
        int const proto_w = static_cast<int>(output_shapes_[1][3]);
        if (proto_channels != static_cast<int>(detection.mask_coeffs.size())) {
            throw std::runtime_error("Mask coefficient dimension does not match segmentation proto output");
        }

        auto mask = reconstruct_mask_logits(proto, detection.mask_coeffs, proto_h, proto_w);
        if (retina_masks) {
            auto scaled = scale_mask_to_shape(mask, proto_h, proto_w, original_h, original_w);
            crop_mask_inplace(scaled, original_h, original_w, detection.output_box);
            std::vector<std::uint8_t> binary;
            if (!threshold_mask(scaled, binary)) {
                binary.clear();
            }
            return binary;
        }

        std::array<float, 4> proto_box = {
            detection.input_box[0] * (static_cast<float>(proto_w) / static_cast<float>(input_w)),
            detection.input_box[1] * (static_cast<float>(proto_h) / static_cast<float>(input_h)),
            detection.input_box[2] * (static_cast<float>(proto_w) / static_cast<float>(input_w)),
            detection.input_box[3] * (static_cast<float>(proto_h) / static_cast<float>(input_h)),
        };
        crop_mask_inplace(mask, proto_h, proto_w, proto_box);
        auto upsampled = resize_mask(mask, proto_h, proto_w, input_h, input_w);
        std::vector<std::uint8_t> binary;
        if (!threshold_mask(upsampled, binary)) {
            binary.clear();
        }
        return binary;
    }

    NativePostprocessUpload postprocess_outputs(
        int input_h,
        int input_w,
        int original_h,
        int original_w,
        float conf_threshold,
        float iou_threshold,
        int max_det,
        bool retina_masks)
    {
        auto const start = Clock::now();
        if (max_det <= 0) {
            auto upload = upload_postprocess_result(NativePostprocessResult{});
            upload.postprocess_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
            return upload;
        }
        auto const predictions = copy_output_buffer_to_float(0);
        std::vector<NativePostprocessCandidate> detections =
            output_layout_is_end2end()
                ? decode_end2end_candidates(predictions, conf_threshold, max_det)
                : decode_raw_candidates(predictions, conf_threshold, iou_threshold, max_det);

        std::vector<float> proto;
        if (task_is_segment()) {
            proto = copy_output_buffer_to_float(1);
        }

        int const mask_h = retina_masks ? original_h : input_h;
        int const mask_w = retina_masks ? original_w : input_w;
        bool const had_mask_candidates = task_is_segment() && !detections.empty();
        NativePostprocessResult result;
        result.mask_height = mask_h;
        result.mask_width = mask_w;

        for (auto& detection : detections) {
            detection.output_box = scale_box_to_shape(detection.input_box, input_h, input_w, original_h, original_w);
        }

        std::vector<std::vector<std::uint8_t>> masks_by_detection;
        if (task_is_segment()) {
            masks_by_detection.reserve(detections.size());
        }

        for (auto const& detection : detections) {
            std::vector<std::uint8_t> binary_mask;
            if (task_is_segment()) {
                binary_mask = build_mask_for_detection(detection, proto, input_h, input_w, original_h, original_w, retina_masks);
                if (binary_mask.empty()) {
                    continue;
                }
                masks_by_detection.push_back(std::move(binary_mask));
            }

            result.boxes.insert(
                result.boxes.end(),
                {
                    detection.output_box[0],
                    detection.output_box[1],
                    detection.output_box[2],
                    detection.output_box[3],
                    detection.confidence,
                    static_cast<float>(detection.cls),
                });
        }

        if (task_is_segment() && (!masks_by_detection.empty() || had_mask_candidates)) {
            result.include_masks = true;
            result.masks.reserve(
                masks_by_detection.size() * static_cast<std::size_t>(mask_h) * static_cast<std::size_t>(mask_w));
            for (auto const& mask : masks_by_detection) {
                result.masks.insert(result.masks.end(), mask.begin(), mask.end());
            }
        }

        auto upload = upload_postprocess_result(result);
        upload.postprocess_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        return upload;
    }

    int find_binding_index(std::string const& name, bool is_input) const
    {
        for (auto const& binding : bindings_) {
            if (binding.name == name && binding.is_input == is_input) {
                return binding.index;
            }
        }
        throw std::runtime_error("Failed to locate TensorRT binding: " + name);
    }

    void discover_bindings()
    {
        auto const count = engine_->getNbIOTensors();
        bindings_.reserve(static_cast<std::size_t>(count));
        output_indices_.clear();
        output_names_.clear();

        for (int i = 0; i < count; ++i) {
            BindingInfo binding;
            binding.index = i;
            binding.name = engine_->getIOTensorName(i);
            binding.is_input = engine_->getTensorIOMode(binding.name.c_str()) == nvinfer1::TensorIOMode::kINPUT;
            binding.dtype = engine_->getTensorDataType(binding.name.c_str());
            binding.dims = engine_->getTensorShape(binding.name.c_str());
            if (!binding.is_input) {
                output_indices_.push_back(i);
                output_names_.push_back(binding.name);
            }
            bindings_.push_back(binding);
        }

        output_buffers_.resize(output_indices_.size());
        output_shapes_.resize(output_indices_.size());
    }

    void preprocess_host_image(py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> const& image, int target_h, int target_w)
    {
        auto const info = image.request();
        auto const orig_h = static_cast<int>(info.shape[0]);
        auto const orig_w = static_cast<int>(info.shape[1]);

#ifdef YOLOE_TRT_ENABLE_CUDA_PREPROCESS
        auto const input_bytes = static_cast<std::size_t>(orig_h) * static_cast<std::size_t>(info.strides[0]);
        auto const plane_elements = static_cast<std::size_t>(target_h) * static_cast<std::size_t>(target_w) * 3U;
        image_upload_buffer_.ensure(input_bytes);
        image_buffer_.ensure(plane_elements * dtype_size(image_binding_.dtype));
        throw_if_cuda_failed(
            cudaEventRecord(preprocess_start_event_, stream_),
            "cudaEventRecord failed for host image preprocess start");
        throw_if_cuda_failed(
            cudaMemcpyAsync(
                image_upload_buffer_.data(),
                info.ptr,
                input_bytes,
                cudaMemcpyHostToDevice,
                stream_),
            "cudaMemcpyAsync for raw image upload failed");
        launch_cuda_preprocess(
            image_upload_buffer_.data(),
            orig_w,
            orig_h,
            static_cast<int>(info.strides[0]),
            CudaPreprocessInputFormat::kBGR,
            target_w,
            target_h,
            fp16_,
            image_buffer_.data(),
            stream_);
        throw_if_cuda_failed(cudaGetLastError(), "CUDA preprocess kernel launch failed for host image");
        throw_if_cuda_failed(
            cudaEventRecord(preprocess_end_event_, stream_),
            "cudaEventRecord failed for host image preprocess end");
#else
        auto const preprocess_start = Clock::now();
        cv::Mat const src(orig_h, orig_w, CV_8UC3, const_cast<void*>(info.ptr));
        auto const geometry = compute_letterbox_geometry(orig_w, orig_h, target_w, target_h);
        auto const resized_h = geometry.resized_height;
        auto const resized_w = geometry.resized_width;
        auto const top = geometry.pad_top;
        auto const left = geometry.pad_left;

        cv::Mat resized;
        cv::resize(src, resized, cv::Size(resized_w, resized_h), 0.0, 0.0, cv::INTER_LINEAR);

        cv::Mat canvas(target_h, target_w, CV_8UC3, cv::Scalar(114, 114, 114));
        resized.copyTo(canvas(cv::Rect(left, top, resized_w, resized_h)));

        cv::Mat rgb;
        cv::cvtColor(canvas, rgb, cv::COLOR_BGR2RGB);

        cv::Mat rgb_f32;
        rgb.convertTo(rgb_f32, CV_32FC3, 1.0 / 255.0);

        std::vector<cv::Mat> channels;
        cv::split(rgb_f32, channels);

        auto const plane_elements = static_cast<std::size_t>(target_h * target_w);
        host_image_float_.resize(plane_elements * 3);
        for (int c = 0; c < 3; ++c) {
            std::memcpy(
                host_image_float_.data() + plane_elements * static_cast<std::size_t>(c),
                channels[static_cast<std::size_t>(c)].ptr<float>(),
                plane_elements * sizeof(float));
        }

        if (image_binding_.dtype == nvinfer1::DataType::kHALF) {
            host_image_half_.resize(host_image_float_.size());
            for (std::size_t i = 0; i < host_image_float_.size(); ++i) {
                host_image_half_[i] = __float2half(host_image_float_[i]);
            }
            image_buffer_.ensure(host_image_half_.size() * sizeof(__half));
            throw_if_cuda_failed(
                cudaEventRecord(preprocess_start_event_, stream_),
                "cudaEventRecord failed for host image upload start");
            throw_if_cuda_failed(
                cudaMemcpyAsync(
                    image_buffer_.data(),
                    host_image_half_.data(),
                    host_image_half_.size() * sizeof(__half),
                    cudaMemcpyHostToDevice,
                    stream_),
                "cudaMemcpyAsync for image upload failed");
        } else {
            image_buffer_.ensure(host_image_float_.size() * sizeof(float));
            throw_if_cuda_failed(
                cudaEventRecord(preprocess_start_event_, stream_),
                "cudaEventRecord failed for host image upload start");
            throw_if_cuda_failed(
                cudaMemcpyAsync(
                    image_buffer_.data(),
                    host_image_float_.data(),
                    host_image_float_.size() * sizeof(float),
                    cudaMemcpyHostToDevice,
                    stream_),
                "cudaMemcpyAsync for image upload failed");
        }
        throw_if_cuda_failed(
            cudaEventRecord(preprocess_end_event_, stream_),
            "cudaEventRecord failed for host image upload end");
        last_preprocess_cpu_ms_ = std::chrono::duration<double, std::milli>(Clock::now() - preprocess_start).count();
#endif
    }

    void set_input_dimensions(int target_h, int target_w, int prompt_count)
    {
        auto image_dims = image_binding_.dims;
        if (dims_has_dynamic(image_dims)) {
            image_dims.d[0] = 1;
            image_dims.d[2] = target_h;
            image_dims.d[3] = target_w;
            if (!context_->setInputShape(image_binding_.name.c_str(), image_dims)) {
                throw std::runtime_error("Failed to set dynamic image tensor shape");
            }
        }

        auto prompt_dims = prompt_binding_.dims;
        if (dims_has_dynamic(prompt_dims)) {
            prompt_dims.d[0] = 1;
            prompt_dims.d[1] = prompt_count;
            if (!context_->setInputShape(prompt_binding_.name.c_str(), prompt_dims)) {
                throw std::runtime_error("Failed to set dynamic prompt tensor shape");
            }
        }

        if (context_->inferShapes(0, nullptr) != 0) {
            throw std::runtime_error("TensorRT shape inference failed after setting input shapes");
        }
    }

    void ensure_output_buffers()
    {
        for (std::size_t i = 0; i < output_indices_.size(); ++i) {
            auto const binding_index = output_indices_[i];
            auto const& binding = bindings_[binding_index];
            auto const dims = context_->getTensorShape(binding.name.c_str());
            if (dims.nbDims < 0 || dims_has_dynamic(dims)) {
                throw std::runtime_error("TensorRT reported unresolved output shape for tensor: " + binding.name);
            }
            auto const bytes = element_count(dims) * dtype_size(binding.dtype);
            output_buffers_[i].ensure(bytes);
            output_shapes_[i] = dims_to_shape(dims);
        }
    }

    double execute(void* image_ptr, int target_h, int target_w, void* prompt_ptr, int prompt_count)
    {
        py::gil_scoped_release release;

        set_input_dimensions(target_h, target_w, prompt_count);
        ensure_output_buffers();

        if (!context_->setInputTensorAddress(image_binding_.name.c_str(), image_ptr)) {
            throw std::runtime_error("Failed to bind TensorRT image input tensor");
        }
        if (!context_->setInputTensorAddress(prompt_binding_.name.c_str(), prompt_ptr)) {
            throw std::runtime_error("Failed to bind TensorRT prompt input tensor");
        }
        for (std::size_t i = 0; i < output_indices_.size(); ++i) {
            auto const binding_index = output_indices_[i];
            auto const& binding = bindings_[binding_index];
            if (!context_->setTensorAddress(binding.name.c_str(), output_buffers_[i].data())) {
                throw std::runtime_error("Failed to bind TensorRT output tensor: " + binding.name);
            }
        }

        throw_if_cuda_failed(
            cudaEventRecord(inference_start_event_, stream_),
            "cudaEventRecord failed for TensorRT inference start");
        if (!context_->enqueueV3(stream_)) {
            throw std::runtime_error("TensorRT enqueueV3 failed");
        }
        throw_if_cuda_failed(
            cudaEventRecord(inference_end_event_, stream_),
            "cudaEventRecord failed for TensorRT inference end");
        throw_if_cuda_failed(cudaStreamSynchronize(stream_), "cudaStreamSynchronize failed after inference");
        return elapsed_event_ms(inference_start_event_, inference_end_event_, "TensorRT inference");
    }

    std::string engine_path_;
    std::string image_input_name_;
    std::string prompt_input_name_;
    int device_id_ = 0;
    cudaStream_t stream_ = nullptr;

    TrtLogger logger_;
    std::unique_ptr<nvinfer1::IRuntime, TrtDeleter> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine, TrtDeleter> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext, TrtDeleter> context_;

    std::vector<BindingInfo> bindings_;
    std::vector<int> output_indices_;
    std::vector<std::string> output_names_;

    int image_input_index_ = -1;
    int prompt_input_index_ = -1;
    BindingInfo image_binding_{};
    BindingInfo prompt_binding_{};

    bool fp16_ = false;
    int prompt_embed_dim_ = 0;
    bool has_prompt_embeddings_ = false;
    int current_prompt_count_ = 0;
    double last_preprocess_cpu_ms_ = 0.0;

    CudaBuffer image_buffer_;
    CudaBuffer image_upload_buffer_;
    CudaBuffer prompt_buffer_;
    std::vector<CudaBuffer> output_buffers_;
    std::vector<std::vector<std::int64_t>> output_shapes_;

#ifndef YOLOE_TRT_ENABLE_CUDA_PREPROCESS
    std::vector<float> host_image_float_;
    std::vector<__half> host_image_half_;
#endif
    std::vector<__half> prompt_half_host_;
    CudaEventHandle preprocess_start_event_;
    CudaEventHandle preprocess_end_event_;
    CudaEventHandle inference_start_event_;
    CudaEventHandle inference_end_event_;
    CudaEventHandle producer_handoff_event_;
};

class NativeVisualPromptRuntime : public std::enable_shared_from_this<NativeVisualPromptRuntime> {
public:
    NativeVisualPromptRuntime(
        std::string engine_path,
        std::string image_input_name,
        std::string visual_input_name,
        int visual_stride,
        int device_id)
        : engine_path_(std::move(engine_path))
        , image_input_name_(std::move(image_input_name))
        , visual_input_name_(std::move(visual_input_name))
        , visual_stride_(visual_stride)
        , device_id_(device_id)
    {
        try {
            throw_if_cuda_failed(cudaSetDevice(device_id_), "cudaSetDevice failed");
            throw_if_cuda_failed(cudaStreamCreate(&stream_), "cudaStreamCreate failed");
            preprocess_start_event_.create("cudaEventCreate failed for visual preprocess start");
            preprocess_end_event_.create("cudaEventCreate failed for visual preprocess end");
            inference_start_event_.create("cudaEventCreate failed for visual inference start");
            inference_end_event_.create("cudaEventCreate failed for visual inference end");

            py::gil_scoped_release release;

            std::FILE* file = std::fopen(engine_path_.c_str(), "rb");
            if (file == nullptr) {
                throw std::runtime_error("Failed to open TensorRT engine: " + engine_path_);
            }
            std::fseek(file, 0, SEEK_END);
            auto const size = static_cast<std::size_t>(std::ftell(file));
            std::rewind(file);
            std::vector<char> bytes(size);
            auto const read = std::fread(bytes.data(), 1, size, file);
            std::fclose(file);
            if (read != size) {
                throw std::runtime_error("Failed to read TensorRT engine: " + engine_path_);
            }

            runtime_.reset(nvinfer1::createInferRuntime(logger_));
            if (!runtime_) {
                throw std::runtime_error("Failed to create TensorRT runtime");
            }
            engine_.reset(runtime_->deserializeCudaEngine(bytes.data(), bytes.size()));
            if (!engine_) {
                throw std::runtime_error("Failed to deserialize TensorRT engine");
            }
            context_.reset(engine_->createExecutionContext());
            if (!context_) {
                throw std::runtime_error("Failed to create TensorRT execution context");
            }

            discover_bindings();
            image_input_index_ = find_binding_index(image_input_name_, true);
            visual_input_index_ = find_binding_index(visual_input_name_, true);

            image_binding_ = bindings_[static_cast<std::size_t>(image_input_index_)];
            visual_binding_ = bindings_[static_cast<std::size_t>(visual_input_index_)];

            if (image_binding_.dims.nbDims != 4) {
                throw std::runtime_error("Visual image input must be NCHW");
            }
            if (visual_binding_.dims.nbDims != 4) {
                throw std::runtime_error("Visual prompt input must have shape (1, N, H, W)");
            }
            if (output_indices_.empty()) {
                throw std::runtime_error("Visual prompt runtime has no outputs");
            }

            fp16_ = image_binding_.dtype == nvinfer1::DataType::kHALF;
        } catch (...) {
            destroy_stream_noexcept();
            throw;
        }
    }

    ~NativeVisualPromptRuntime() noexcept
    {
        destroy_stream_noexcept();
    }

    std::vector<std::string> output_names() const
    {
        return output_names_;
    }

    bool fp16() const
    {
        return fp16_;
    }

    py::dict infer_image(
        py::array image_py,
        int target_h,
        int target_w,
        py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> categories_py,
        py::object bboxes = py::none(),
        py::object masks = py::none())
    {
        auto image = py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>(image_py);
        auto const info = image.request();
        if (info.ndim != 3 || info.shape[2] != 3) {
            throw std::runtime_error("image must have shape (H, W, 3)");
        }

        activate_device("cudaSetDevice failed before native visual inference");
        auto const categories = parse_visual_categories(categories_py);
        auto const visual_batch = build_visual_prompt_batch(categories, bboxes, masks, static_cast<int>(info.shape[0]), static_cast<int>(info.shape[1]), target_h, target_w);
        preprocess_host_image(image, target_h, target_w);
        upload_visual_prompt_batch(visual_batch);
        auto const inference_ms = execute(image_buffer_.data(), visual_buffer_.data(), target_h, target_w, visual_batch.prompt_count);
        auto const preprocess_ms = resolve_preprocess_ms();
        return build_output_dict(target_h, target_w, preprocess_ms, inference_ms);
    }

    void warmup(int target_h, int target_w, int prompt_count, int runs = 2)
    {
        if (runs < 1) {
            throw std::runtime_error("runs must be >= 1");
        }
        activate_device("cudaSetDevice failed before visual warmup");

        py::array_t<std::uint8_t> image({ target_h, target_w, 3 });
        {
            auto mutable_image = image.mutable_unchecked<3>();
            for (int y = 0; y < target_h; ++y) {
                for (int x = 0; x < target_w; ++x) {
                    mutable_image(y, x, 0) = 0;
                    mutable_image(y, x, 1) = 0;
                    mutable_image(y, x, 2) = 0;
                }
            }
        }
        py::array_t<std::int32_t> categories({ prompt_count });
        {
            auto mutable_categories = categories.mutable_unchecked<1>();
            for (int index = 0; index < prompt_count; ++index) {
                mutable_categories(index) = index;
            }
        }
        py::array_t<float> bboxes({ prompt_count, 4 });
        {
            auto mutable_boxes = bboxes.mutable_unchecked<2>();
            for (int index = 0; index < prompt_count; ++index) {
                mutable_boxes(index, 0) = 0.0f;
                mutable_boxes(index, 1) = 0.0f;
                mutable_boxes(index, 2) = static_cast<float>(target_w);
                mutable_boxes(index, 3) = static_cast<float>(target_h);
            }
        }
        for (int run = 0; run < runs; ++run) {
            static_cast<void>(infer_image(image, target_h, target_w, categories, bboxes, py::none()));
        }
    }

private:
    void destroy_stream_noexcept() noexcept
    {
        if (stream_ != nullptr) {
            try {
                cudaSetDevice(device_id_);
                cudaStreamDestroy(stream_);
            } catch (...) {
            }
            stream_ = nullptr;
        }
    }

    void activate_device(char const* message) const
    {
        throw_if_cuda_failed(cudaSetDevice(device_id_), message);
    }

    double elapsed_event_ms(cudaEvent_t start_event, cudaEvent_t end_event, char const* context) const
    {
        float elapsed_ms = 0.0f;
        throw_if_cuda_failed(
            cudaEventElapsedTime(&elapsed_ms, start_event, end_event),
            (std::string(context) + ": cudaEventElapsedTime failed").c_str());
        return static_cast<double>(elapsed_ms);
    }

    VisualPromptHostBatch build_visual_prompt_batch(
        std::vector<int> const& categories,
        py::object const& bboxes_obj,
        py::object const& masks_obj,
        int src_h,
        int src_w,
        int dst_h,
        int dst_w) const
    {
        if (!bboxes_obj.is_none()) {
            auto const bboxes = py::array_t<float, py::array::c_style | py::array::forcecast>(bboxes_obj);
            return build_visual_prompt_batch_from_boxes(bboxes, categories, src_h, src_w, dst_h, dst_w, visual_stride_);
        }
        if (!masks_obj.is_none()) {
            auto const masks = py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>(masks_obj);
            return build_visual_prompt_batch_from_masks(masks, categories, dst_h, dst_w, visual_stride_);
        }
        throw std::runtime_error("Either bboxes or masks must be provided");
    }

    void upload_visual_prompt_batch(VisualPromptHostBatch const& batch)
    {
        auto const element_total =
            static_cast<std::size_t>(batch.prompt_count) * static_cast<std::size_t>(batch.prompt_height) * static_cast<std::size_t>(batch.prompt_width);
        if (visual_binding_.dtype == nvinfer1::DataType::kHALF) {
            visual_half_host_.resize(element_total);
            for (std::size_t index = 0; index < element_total; ++index) {
                visual_half_host_[index] = __float2half(batch.tensor[index]);
            }
            visual_buffer_.ensure(std::max<std::size_t>(visual_half_host_.size() * sizeof(__half), sizeof(__half)));
            if (!visual_half_host_.empty()) {
                throw_if_cuda_failed(
                    cudaMemcpyAsync(
                        visual_buffer_.data(),
                        visual_half_host_.data(),
                        visual_half_host_.size() * sizeof(__half),
                        cudaMemcpyHostToDevice,
                        stream_),
                    "cudaMemcpyAsync failed for visual prompt upload");
            }
        } else {
            visual_buffer_.ensure(std::max<std::size_t>(element_total * sizeof(float), sizeof(float)));
            if (!batch.tensor.empty()) {
                throw_if_cuda_failed(
                    cudaMemcpyAsync(
                        visual_buffer_.data(),
                        batch.tensor.data(),
                        batch.tensor.size() * sizeof(float),
                        cudaMemcpyHostToDevice,
                        stream_),
                    "cudaMemcpyAsync failed for visual prompt upload");
            }
        }
    }

    double resolve_preprocess_ms() const
    {
#ifdef YOLOE_TRT_ENABLE_CUDA_PREPROCESS
        return elapsed_event_ms(preprocess_start_event_, preprocess_end_event_, "visual host image preprocess");
#else
        return last_preprocess_cpu_ms_ + elapsed_event_ms(preprocess_start_event_, preprocess_end_event_, "visual host image upload");
#endif
    }

    py::dict build_output_dict(int target_h, int target_w, double preprocess_ms, double inference_ms)
    {
        py::dict outputs;
        for (std::size_t i = 0; i < output_indices_.size(); ++i) {
            auto const binding_index = output_indices_[i];
            auto const& binding = bindings_[binding_index];
            outputs[py::str(binding.name)] = py::cast(
                CudaTensorView(shared_from_this(), nullptr, output_buffers_[i].data(), output_shapes_[i], binding.dtype, device_id_));
        }
        outputs[py::str("input_shape")] = py::make_tuple(target_h, target_w);
        outputs[py::str("preprocess_ms")] = preprocess_ms;
        outputs[py::str("inference_ms")] = inference_ms;
        return outputs;
    }

    int find_binding_index(std::string const& name, bool is_input) const
    {
        for (auto const& binding : bindings_) {
            if (binding.name == name && binding.is_input == is_input) {
                return binding.index;
            }
        }
        throw std::runtime_error("Failed to locate TensorRT binding: " + name);
    }

    void discover_bindings()
    {
        auto const count = engine_->getNbIOTensors();
        bindings_.reserve(static_cast<std::size_t>(count));
        output_indices_.clear();
        output_names_.clear();

        for (int i = 0; i < count; ++i) {
            BindingInfo binding;
            binding.index = i;
            binding.name = engine_->getIOTensorName(i);
            binding.is_input = engine_->getTensorIOMode(binding.name.c_str()) == nvinfer1::TensorIOMode::kINPUT;
            binding.dtype = engine_->getTensorDataType(binding.name.c_str());
            binding.dims = engine_->getTensorShape(binding.name.c_str());
            if (!binding.is_input) {
                output_indices_.push_back(i);
                output_names_.push_back(binding.name);
            }
            bindings_.push_back(binding);
        }

        output_buffers_.resize(output_indices_.size());
        output_shapes_.resize(output_indices_.size());
    }

    void preprocess_host_image(py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> const& image, int target_h, int target_w)
    {
        auto const info = image.request();
        auto const orig_h = static_cast<int>(info.shape[0]);
        auto const orig_w = static_cast<int>(info.shape[1]);

#ifdef YOLOE_TRT_ENABLE_CUDA_PREPROCESS
        auto const input_bytes = static_cast<std::size_t>(orig_h) * static_cast<std::size_t>(info.strides[0]);
        auto const plane_elements = static_cast<std::size_t>(target_h) * static_cast<std::size_t>(target_w) * 3U;
        image_upload_buffer_.ensure(input_bytes);
        image_buffer_.ensure(plane_elements * dtype_size(image_binding_.dtype));
        throw_if_cuda_failed(cudaEventRecord(preprocess_start_event_, stream_), "cudaEventRecord failed for visual preprocess start");
        throw_if_cuda_failed(
            cudaMemcpyAsync(image_upload_buffer_.data(), info.ptr, input_bytes, cudaMemcpyHostToDevice, stream_),
            "cudaMemcpyAsync for visual raw image upload failed");
        launch_cuda_preprocess(
            image_upload_buffer_.data(),
            orig_w,
            orig_h,
            static_cast<int>(info.strides[0]),
            CudaPreprocessInputFormat::kBGR,
            target_w,
            target_h,
            fp16_,
            image_buffer_.data(),
            stream_);
        throw_if_cuda_failed(cudaGetLastError(), "CUDA preprocess kernel launch failed for visual image");
        throw_if_cuda_failed(cudaEventRecord(preprocess_end_event_, stream_), "cudaEventRecord failed for visual preprocess end");
#else
        auto const preprocess_start = Clock::now();
        cv::Mat const src(orig_h, orig_w, CV_8UC3, const_cast<void*>(info.ptr));
        auto const geometry = compute_letterbox_geometry(orig_w, orig_h, target_w, target_h);

        cv::Mat resized;
        cv::resize(src, resized, cv::Size(geometry.resized_width, geometry.resized_height), 0.0, 0.0, cv::INTER_LINEAR);
        cv::Mat canvas(target_h, target_w, CV_8UC3, cv::Scalar(114, 114, 114));
        resized.copyTo(canvas(cv::Rect(geometry.pad_left, geometry.pad_top, geometry.resized_width, geometry.resized_height)));

        cv::Mat rgb;
        cv::cvtColor(canvas, rgb, cv::COLOR_BGR2RGB);
        cv::Mat rgb_f32;
        rgb.convertTo(rgb_f32, CV_32FC3, 1.0 / 255.0);

        std::vector<cv::Mat> channels;
        cv::split(rgb_f32, channels);
        auto const plane_elements = static_cast<std::size_t>(target_h) * static_cast<std::size_t>(target_w);
        host_image_float_.resize(plane_elements * 3U);
        for (int channel = 0; channel < 3; ++channel) {
            std::memcpy(
                host_image_float_.data() + plane_elements * static_cast<std::size_t>(channel),
                channels[static_cast<std::size_t>(channel)].ptr<float>(),
                plane_elements * sizeof(float));
        }

        if (image_binding_.dtype == nvinfer1::DataType::kHALF) {
            host_image_half_.resize(host_image_float_.size());
            for (std::size_t index = 0; index < host_image_float_.size(); ++index) {
                host_image_half_[index] = __float2half(host_image_float_[index]);
            }
            image_buffer_.ensure(host_image_half_.size() * sizeof(__half));
            throw_if_cuda_failed(cudaEventRecord(preprocess_start_event_, stream_), "cudaEventRecord failed for visual upload start");
            throw_if_cuda_failed(
                cudaMemcpyAsync(
                    image_buffer_.data(),
                    host_image_half_.data(),
                    host_image_half_.size() * sizeof(__half),
                    cudaMemcpyHostToDevice,
                    stream_),
                "cudaMemcpyAsync for visual image upload failed");
        } else {
            image_buffer_.ensure(host_image_float_.size() * sizeof(float));
            throw_if_cuda_failed(cudaEventRecord(preprocess_start_event_, stream_), "cudaEventRecord failed for visual upload start");
            throw_if_cuda_failed(
                cudaMemcpyAsync(
                    image_buffer_.data(),
                    host_image_float_.data(),
                    host_image_float_.size() * sizeof(float),
                    cudaMemcpyHostToDevice,
                    stream_),
                "cudaMemcpyAsync for visual image upload failed");
        }
        throw_if_cuda_failed(cudaEventRecord(preprocess_end_event_, stream_), "cudaEventRecord failed for visual upload end");
        last_preprocess_cpu_ms_ = std::chrono::duration<double, std::milli>(Clock::now() - preprocess_start).count();
#endif
    }

    void set_input_dimensions(int target_h, int target_w, int prompt_count)
    {
        auto image_dims = image_binding_.dims;
        if (dims_has_dynamic(image_dims)) {
            image_dims.d[0] = 1;
            image_dims.d[2] = target_h;
            image_dims.d[3] = target_w;
            if (!context_->setInputShape(image_binding_.name.c_str(), image_dims)) {
                throw std::runtime_error("Failed to set dynamic visual image tensor shape");
            }
        }

        auto visual_dims = visual_binding_.dims;
        if (dims_has_dynamic(visual_dims)) {
            visual_dims.d[0] = 1;
            visual_dims.d[1] = prompt_count;
            visual_dims.d[2] = target_h / visual_stride_;
            visual_dims.d[3] = target_w / visual_stride_;
            if (!context_->setInputShape(visual_binding_.name.c_str(), visual_dims)) {
                throw std::runtime_error("Failed to set dynamic visual prompt tensor shape");
            }
        }

        if (context_->inferShapes(0, nullptr) != 0) {
            throw std::runtime_error("TensorRT shape inference failed after setting visual input shapes");
        }
    }

    void ensure_output_buffers()
    {
        for (std::size_t i = 0; i < output_indices_.size(); ++i) {
            auto const binding_index = output_indices_[i];
            auto const& binding = bindings_[binding_index];
            auto const dims = context_->getTensorShape(binding.name.c_str());
            if (dims.nbDims < 0 || dims_has_dynamic(dims)) {
                throw std::runtime_error("TensorRT reported unresolved output shape for visual tensor: " + binding.name);
            }
            auto const bytes = element_count(dims) * dtype_size(binding.dtype);
            output_buffers_[i].ensure(bytes);
            output_shapes_[i] = dims_to_shape(dims);
        }
    }

    double execute(void* image_ptr, void* visual_ptr, int target_h, int target_w, int prompt_count)
    {
        py::gil_scoped_release release;

        set_input_dimensions(target_h, target_w, prompt_count);
        ensure_output_buffers();

        if (!context_->setInputTensorAddress(image_binding_.name.c_str(), image_ptr)) {
            throw std::runtime_error("Failed to bind visual image tensor");
        }
        if (!context_->setInputTensorAddress(visual_binding_.name.c_str(), visual_ptr)) {
            throw std::runtime_error("Failed to bind visual prompt tensor");
        }
        for (std::size_t i = 0; i < output_indices_.size(); ++i) {
            auto const binding_index = output_indices_[i];
            auto const& binding = bindings_[binding_index];
            if (!context_->setTensorAddress(binding.name.c_str(), output_buffers_[i].data())) {
                throw std::runtime_error("Failed to bind visual output tensor: " + binding.name);
            }
        }

        throw_if_cuda_failed(cudaEventRecord(inference_start_event_, stream_), "cudaEventRecord failed for visual inference start");
        if (!context_->enqueueV3(stream_)) {
            throw std::runtime_error("Visual TensorRT enqueueV3 failed");
        }
        throw_if_cuda_failed(cudaEventRecord(inference_end_event_, stream_), "cudaEventRecord failed for visual inference end");
        throw_if_cuda_failed(cudaStreamSynchronize(stream_), "cudaStreamSynchronize failed after visual inference");
        return elapsed_event_ms(inference_start_event_, inference_end_event_, "visual TensorRT inference");
    }

    std::string engine_path_;
    std::string image_input_name_;
    std::string visual_input_name_;
    int visual_stride_ = 8;
    int device_id_ = 0;
    cudaStream_t stream_ = nullptr;

    TrtLogger logger_;
    std::unique_ptr<nvinfer1::IRuntime, TrtDeleter> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine, TrtDeleter> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext, TrtDeleter> context_;

    std::vector<BindingInfo> bindings_;
    std::vector<int> output_indices_;
    std::vector<std::string> output_names_;

    int image_input_index_ = -1;
    int visual_input_index_ = -1;
    BindingInfo image_binding_{};
    BindingInfo visual_binding_{};

    bool fp16_ = false;
    double last_preprocess_cpu_ms_ = 0.0;

    CudaBuffer image_buffer_;
    CudaBuffer image_upload_buffer_;
    CudaBuffer visual_buffer_;
    std::vector<CudaBuffer> output_buffers_;
    std::vector<std::vector<std::int64_t>> output_shapes_;

#ifndef YOLOE_TRT_ENABLE_CUDA_PREPROCESS
    std::vector<float> host_image_float_;
    std::vector<__half> host_image_half_;
#endif
    std::vector<__half> visual_half_host_;
    CudaEventHandle preprocess_start_event_;
    CudaEventHandle preprocess_end_event_;
    CudaEventHandle inference_start_event_;
    CudaEventHandle inference_end_event_;
};

#ifdef YOLOE_TRT_ENABLE_JETSON_CAMERA
void ensure_gstreamer_initialized()
{
    static std::once_flag init_flag;
    std::call_once(init_flag, []() {
        gst_init(nullptr, nullptr);
    });
}

class NativeJetsonCameraSource : public std::enable_shared_from_this<NativeJetsonCameraSource> {
public:
    NativeJetsonCameraSource(
        std::string pipeline,
        std::string prefix,
        double timeout_s,
        int target_h,
        int target_w,
        bool fp16,
        bool preview_cpu,
        int device_id)
        : pipeline_spec_(std::move(pipeline))
        , prefix_(std::move(prefix))
        , timeout_ns_(static_cast<GstClockTime>(timeout_s * 1'000'000'000.0))
        , target_h_(target_h)
        , target_w_(target_w)
        , fp16_(fp16)
        , preview_cpu_(preview_cpu)
        , device_id_(device_id)
        , output_dtype_(fp16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT)
    {
        throw_if_cuda_failed(cudaSetDevice(device_id_), "cudaSetDevice failed");
        throw_if_cuda_failed(cudaStreamCreate(&stream_), "cudaStreamCreate failed");
    }

    ~NativeJetsonCameraSource()
    {
        close();
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
    }

    py::object read_frame()
    {
        open_if_needed();
        cleanup_completed_frames();

        GstSample* sample = gst_app_sink_try_pull_sample(appsink_, timeout_ns_);
        if (sample == nullptr) {
            if (gst_app_sink_is_eos(appsink_)) {
                return py::none();
            }
            if (auto* message = gst_bus_timed_pop_filtered(bus_, 0, static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS | GST_MESSAGE_WARNING))) {
                if (GST_MESSAGE_TYPE(message) == GST_MESSAGE_ERROR) {
                    GError* error = nullptr;
                    gchar* debug = nullptr;
                    gst_message_parse_error(message, &error, &debug);
                    std::string message_text = error != nullptr ? error->message : "unknown";
                    std::string debug_text = debug != nullptr ? debug : "";
                    if (error != nullptr) {
                        g_error_free(error);
                    }
                    if (debug != nullptr) {
                        g_free(debug);
                    }
                    gst_message_unref(message);
                    throw std::runtime_error("GStreamer source error: " + message_text + "; debug=" + debug_text);
                }
                if (GST_MESSAGE_TYPE(message) == GST_MESSAGE_EOS) {
                    gst_message_unref(message);
                    return py::none();
                }
                gst_message_unref(message);
            }
            throw std::runtime_error("Timed out waiting for a frame from zero-copy GStreamer source");
        }

        return frame_from_sample(sample);
    }

    void close()
    {
        if (stream_ != nullptr) {
            cudaSetDevice(device_id_);
            cudaStreamSynchronize(stream_);
        }
        while (!pending_frames_.empty()) {
            pending_frames_.front()->release_frame_resources();
            pending_frames_.pop_front();
        }
        reusable_frames_.clear();
        if (pipeline_ != nullptr) {
            gst_element_set_state(pipeline_, GST_STATE_NULL);
            gst_element_get_state(pipeline_, nullptr, nullptr, timeout_ns_);
        }
        if (appsink_ != nullptr) {
            gst_object_unref(appsink_);
            appsink_ = nullptr;
        }
        if (bus_ != nullptr) {
            gst_object_unref(bus_);
            bus_ = nullptr;
        }
        if (pipeline_ != nullptr) {
            gst_object_unref(pipeline_);
            pipeline_ = nullptr;
        }
        frame_index_ = 0;
        started_ = false;
    }

private:
    std::shared_ptr<PendingJetsonFrame> acquire_frame_slot(std::size_t bytes)
    {
        throw_if_cuda_failed(cudaSetDevice(device_id_), "cudaSetDevice failed during zero-copy frame allocation");
        while (!reusable_frames_.empty()) {
            auto frame = reusable_frames_.front();
            reusable_frames_.pop_front();
            if (frame.use_count() != 1) {
                continue;
            }
            frame->output_buffer.ensure(bytes);
            return frame;
        }

        auto frame = std::make_shared<PendingJetsonFrame>();
        frame->device_id = device_id_;
        frame->output_buffer.ensure(bytes);
        return frame;
    }

    void cleanup_completed_frames()
    {
        throw_if_cuda_failed(cudaSetDevice(device_id_), "cudaSetDevice failed during zero-copy frame cleanup");
        while (!pending_frames_.empty()) {
            auto frame = pending_frames_.front();
            if (!frame->is_ready()) {
                break;
            }
            frame->release_frame_resources();
            pending_frames_.pop_front();
            if (frame.use_count() == 1) {
                reusable_frames_.push_back(std::move(frame));
            }
        }
    }

    void open_if_needed()
    {
        if (started_) {
            return;
        }

        ensure_gstreamer_initialized();

        GError* error = nullptr;
        pipeline_ = gst_parse_launch(pipeline_spec_.c_str(), &error);
        if (pipeline_ == nullptr) {
            std::string message = error != nullptr ? error->message : "unknown";
            if (error != nullptr) {
                g_error_free(error);
            }
            throw std::runtime_error("Unable to parse GStreamer pipeline: " + message);
        }

        auto* appsink_element = gst_bin_get_by_name(GST_BIN(pipeline_), "sink");
        if (appsink_element == nullptr) {
            close();
            throw std::runtime_error("GStreamer pipeline did not expose appsink 'sink': " + pipeline_spec_);
        }
        if (!GST_IS_APP_SINK(appsink_element)) {
            gst_object_unref(appsink_element);
            close();
            throw std::runtime_error("Pipeline element 'sink' is not a GstAppSink");
        }
        appsink_ = GST_APP_SINK(appsink_element);
        bus_ = gst_element_get_bus(pipeline_);

        if (gst_element_set_state(pipeline_, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
            close();
            throw std::runtime_error("Unable to start GStreamer pipeline: " + pipeline_spec_);
        }
        gst_element_get_state(pipeline_, nullptr, nullptr, timeout_ns_);
        started_ = true;
    }

    py::array preview_from_surface(NvBufSurface* surface, int width, int height, bool input_is_bgrx)
    {
        if (!preview_cpu_) {
            return py::array();
        }
        if (NvBufSurfaceMap(surface, 0, 0, NVBUF_MAP_READ) != 0) {
            throw std::runtime_error("NvBufSurfaceMap failed for preview frame");
        }
        if (NvBufSurfaceSyncForCpu(surface, 0, 0) != 0) {
            NvBufSurfaceUnMap(surface, 0, 0);
            throw std::runtime_error("NvBufSurfaceSyncForCpu failed for preview frame");
        }

        try {
            auto* mapped = static_cast<unsigned char*>(surface->surfaceList[0].mappedAddr.addr[0]);
            int const pitch = static_cast<int>(surface->surfaceList[0].pitch);
            if (mapped == nullptr || pitch <= 0) {
                throw std::runtime_error("Preview frame did not expose CPU-mappable image data");
            }
            cv::Mat const src(height, width, CV_8UC4, mapped, static_cast<std::size_t>(pitch));
            cv::Mat bgr;
            cv::cvtColor(src, bgr, input_is_bgrx ? cv::COLOR_BGRA2BGR : cv::COLOR_RGBA2BGR);
            py::array_t<std::uint8_t> array({ bgr.rows, bgr.cols, bgr.channels() });
            auto const info = array.request();
            std::memcpy(info.ptr, bgr.data, static_cast<std::size_t>(bgr.rows * bgr.cols * bgr.channels()));
            NvBufSurfaceUnMap(surface, 0, 0);
            return array;
        } catch (...) {
            NvBufSurfaceUnMap(surface, 0, 0);
            throw;
        }
    }

    py::dict frame_from_sample(GstSample* sample)
    {
        auto* caps = gst_sample_get_caps(sample);
        if (caps == nullptr) {
            gst_sample_unref(sample);
            throw std::runtime_error("GStreamer sample did not include caps");
        }
        auto* structure = gst_caps_get_structure(caps, 0);
        int width = 0;
        int height = 0;
        if (!gst_structure_get_int(structure, "width", &width) || !gst_structure_get_int(structure, "height", &height)) {
            gst_sample_unref(sample);
            throw std::runtime_error("Unable to read GStreamer frame dimensions");
        }
        char const* format_name = gst_structure_get_string(structure, "format");
        if (format_name == nullptr) {
            gst_sample_unref(sample);
            throw std::runtime_error("Unable to read GStreamer frame format");
        }
        bool const input_is_bgrx = std::string(format_name) == "BGRx";
        if (!input_is_bgrx && std::string(format_name) != "RGBA") {
            gst_sample_unref(sample);
            throw std::runtime_error("Zero-copy source expects BGRx or RGBA NVMM frames, got: " + std::string(format_name));
        }

        auto* buffer = gst_sample_get_buffer(sample);
        if (buffer == nullptr || gst_buffer_n_memory(buffer) <= 0) {
            gst_sample_unref(sample);
            throw std::runtime_error("GStreamer sample did not include a buffer");
        }
        NvBufSurface* surface = nullptr;
        GstBuffer* mapped_buffer = nullptr;
        GstMapInfo mapped_buffer_info{};
        bool buffer_mapped = false;
        auto* memory = gst_buffer_peek_memory(buffer, 0);
        if (memory != nullptr && gst_is_dmabuf_memory(memory)) {
            int const dmabuf_fd = gst_dmabuf_memory_get_fd(memory);
            if (dmabuf_fd < 0) {
                gst_sample_unref(sample);
                throw std::runtime_error("Unable to extract DMABUF fd from GStreamer sample");
            }
            if (NvBufSurfaceFromFd(dmabuf_fd, reinterpret_cast<void**>(&surface)) != 0 || surface == nullptr) {
                gst_sample_unref(sample);
                throw std::runtime_error("NvBufSurfaceFromFd failed for zero-copy frame");
            }
        } else {
            if (!gst_buffer_map(buffer, &mapped_buffer_info, GST_MAP_READ)) {
                gst_sample_unref(sample);
                throw std::runtime_error("Unable to map zero-copy GstBuffer");
            }
            mapped_buffer = buffer;
            buffer_mapped = true;
            if (mapped_buffer_info.size < sizeof(NvBufSurface)) {
                gst_buffer_unmap(mapped_buffer, &mapped_buffer_info);
                gst_sample_unref(sample);
                throw std::runtime_error("Zero-copy GstBuffer did not expose an NvBufSurface payload");
            }
            surface = reinterpret_cast<NvBufSurface*>(mapped_buffer_info.data);
            if (surface == nullptr) {
                gst_buffer_unmap(mapped_buffer, &mapped_buffer_info);
                gst_sample_unref(sample);
                throw std::runtime_error("Mapped zero-copy GstBuffer did not expose a valid NvBufSurface");
            }
        }

        py::object preview = py::none();
        if (preview_cpu_) {
            try {
                preview = preview_from_surface(surface, width, height, input_is_bgrx);
            } catch (...) {
                gst_sample_unref(sample);
                throw;
            }
        }

        bool mapped_egl = false;
        if (surface->surfaceList[0].mappedAddr.eglImage == nullptr) {
            if (NvBufSurfaceMapEglImage(surface, 0) != 0) {
                gst_sample_unref(sample);
                throw std::runtime_error("NvBufSurfaceMapEglImage failed for zero-copy frame");
            }
            mapped_egl = true;
        }

        auto egl_image = surface->surfaceList[0].mappedAddr.eglImage;
        if (egl_image == nullptr) {
            if (mapped_egl) {
                NvBufSurfaceUnMapEglImage(surface, 0);
            }
            gst_sample_unref(sample);
            throw std::runtime_error("EGL image was unavailable for zero-copy frame");
        }

        CUgraphicsResource resource = nullptr;
        CUeglFrame egl_frame{};
        cudaFree(0);
        if (cuGraphicsEGLRegisterImage(&resource, egl_image, CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE) != CUDA_SUCCESS) {
            if (mapped_egl) {
                NvBufSurfaceUnMapEglImage(surface, 0);
            }
            gst_sample_unref(sample);
            throw std::runtime_error("cuGraphicsEGLRegisterImage failed for zero-copy frame");
        }

        try {
            if (cuGraphicsResourceGetMappedEglFrame(&egl_frame, resource, 0, 0) != CUDA_SUCCESS) {
                throw std::runtime_error("cuGraphicsResourceGetMappedEglFrame failed for zero-copy frame");
            }
            if (egl_frame.frameType != CU_EGL_FRAME_TYPE_PITCH) {
                throw std::runtime_error("Zero-copy source only supports pitch-linear EGL frames");
            }

            auto const plane_elements = static_cast<std::size_t>(target_h_) * static_cast<std::size_t>(target_w_) * 3U;
            auto pending_frame = acquire_frame_slot(plane_elements * dtype_size(output_dtype_));
            launch_cuda_preprocess(
                egl_frame.frame.pPitch[0],
                width,
                height,
                static_cast<int>(egl_frame.pitch),
                input_is_bgrx ? CudaPreprocessInputFormat::kBGRX : CudaPreprocessInputFormat::kRGBA,
                target_w_,
                target_h_,
                fp16_,
                pending_frame->output_buffer.data(),
                stream_);
            throw_if_cuda_failed(cudaGetLastError(), "Jetson zero-copy preprocess kernel launch failed");

            pending_frame->record_ready_event(stream_);
            pending_frame->sample = sample;
            pending_frame->mapped_buffer = mapped_buffer;
            pending_frame->mapped_buffer_info = mapped_buffer_info;
            pending_frame->buffer_mapped = buffer_mapped;
            pending_frame->surface = surface;
            pending_frame->mapped_egl = mapped_egl;
            pending_frame->resource = resource;
            pending_frames_.push_back(pending_frame);
            sample = nullptr;
            mapped_buffer = nullptr;
            buffer_mapped = false;
            surface = nullptr;
            mapped_egl = false;
            resource = nullptr;

            py::dict result;
            result["tensor"] = CameraCudaTensorView(
                shared_from_this(),
                pending_frame,
                pending_frame->output_buffer.data(),
                { 1, 3, target_h_, target_w_ },
                output_dtype_,
                device_id_);
            result["path"] = prefix_ + "_frame" + [&]() {
                char buffer[32];
                std::snprintf(buffer, sizeof(buffer), "%06zu", frame_index_);
                return std::string(buffer);
            }();
            result["original_shape"] = py::make_tuple(height, width);
            result["producer_stream"] = reinterpret_cast<std::uintptr_t>(stream_);
            result["preview"] = preview;
            ++frame_index_;
            return result;
        } catch (...) {
            if (resource != nullptr) {
                cuGraphicsUnregisterResource(resource);
            }
            if (mapped_egl && surface != nullptr) {
                NvBufSurfaceUnMapEglImage(surface, 0);
            }
            if (buffer_mapped && mapped_buffer != nullptr) {
                gst_buffer_unmap(mapped_buffer, &mapped_buffer_info);
            }
            if (sample != nullptr) {
                gst_sample_unref(sample);
            }
            throw;
        }
    }

    std::string pipeline_spec_;
    std::string prefix_;
    GstClockTime timeout_ns_ = 0;
    int target_h_ = 0;
    int target_w_ = 0;
    bool fp16_ = false;
    bool preview_cpu_ = false;
    int device_id_ = 0;
    nvinfer1::DataType output_dtype_ = nvinfer1::DataType::kFLOAT;
    cudaStream_t stream_ = nullptr;
    GstElement* pipeline_ = nullptr;
    GstAppSink* appsink_ = nullptr;
    GstBus* bus_ = nullptr;
    bool started_ = false;
    std::size_t frame_index_ = 0;
    std::deque<std::shared_ptr<PendingJetsonFrame>> pending_frames_;
    std::deque<std::shared_ptr<PendingJetsonFrame>> reusable_frames_;
};
#endif

} // namespace

PYBIND11_MODULE(_native, m)
{
    m.doc() = "Native TensorRT runtime for yoloe_tensorrt";

    py::class_<CudaTensorView>(m, "CudaTensorView")
        .def("__dlpack__", [](CudaTensorView const& self, py::object stream) { return self.dlpack(std::move(stream)); }, py::arg("stream") = py::none())
        .def("__dlpack_device__", &CudaTensorView::dlpack_device);

#ifdef YOLOE_TRT_ENABLE_JETSON_CAMERA
    py::class_<CameraCudaTensorView>(m, "CameraCudaTensorView")
        .def("__dlpack__", [](CameraCudaTensorView const& self, py::object stream) { return self.dlpack(std::move(stream)); }, py::arg("stream") = py::none())
        .def("__dlpack_device__", &CameraCudaTensorView::dlpack_device);
#endif

    py::class_<NativeMainRuntime, std::shared_ptr<NativeMainRuntime>>(m, "NativeMainRuntime")
        .def(py::init<std::string, std::string, std::string, int>(), py::arg("engine_path"), py::arg("image_input_name"), py::arg("prompt_input_name"), py::arg("device_id") = 0)
        .def_property_readonly("output_names", &NativeMainRuntime::output_names)
        .def_property_readonly("fp16", &NativeMainRuntime::fp16)
        .def_property_readonly("prompt_fp16", &NativeMainRuntime::prompt_fp16)
        .def("clear_prompt_embeddings", &NativeMainRuntime::clear_prompt_embeddings)
        .def("set_prompt_embeddings", &NativeMainRuntime::set_prompt_embeddings, py::arg("prompt_embeddings"))
        .def(
            "set_prompt_embeddings_device",
            &NativeMainRuntime::set_prompt_embeddings_device,
            py::arg("prompt_embeddings_ptr"),
            py::arg("prompt_count"),
            py::arg("embed_dim"),
            py::arg("producer_stream") = 0)
        .def("infer_image", &NativeMainRuntime::infer_image, py::arg("image"), py::arg("target_h"), py::arg("target_w"))
        .def(
            "infer_image_postprocessed",
            &NativeMainRuntime::infer_image_postprocessed,
            py::arg("image"),
            py::arg("target_h"),
            py::arg("target_w"),
            py::arg("original_h"),
            py::arg("original_w"),
            py::arg("conf"),
            py::arg("iou"),
            py::arg("max_det"),
            py::arg("retina_masks"))
        .def(
            "_preprocess_image_to_tensor",
            &NativeMainRuntime::_preprocess_image_to_tensor,
            py::arg("image"),
            py::arg("target_h"),
            py::arg("target_w"))
        .def(
            "infer_tensor",
            &NativeMainRuntime::infer_tensor,
            py::arg("image_ptr"),
            py::arg("target_h"),
            py::arg("target_w"),
            py::arg("producer_stream") = 0)
        .def(
            "infer_tensor_postprocessed",
            &NativeMainRuntime::infer_tensor_postprocessed,
            py::arg("image_ptr"),
            py::arg("target_h"),
            py::arg("target_w"),
            py::arg("original_h"),
            py::arg("original_w"),
            py::arg("conf"),
            py::arg("iou"),
            py::arg("max_det"),
            py::arg("retina_masks"),
            py::arg("producer_stream") = 0)
        .def("warmup", &NativeMainRuntime::warmup, py::arg("target_h"), py::arg("target_w"), py::arg("prompt_count"), py::arg("runs") = 2);

    py::class_<NativeVisualPromptRuntime, std::shared_ptr<NativeVisualPromptRuntime>>(m, "NativeVisualPromptRuntime")
        .def(
            py::init<std::string, std::string, std::string, int, int>(),
            py::arg("engine_path"),
            py::arg("image_input_name"),
            py::arg("visual_input_name"),
            py::arg("visual_stride"),
            py::arg("device_id") = 0)
        .def_property_readonly("output_names", &NativeVisualPromptRuntime::output_names)
        .def_property_readonly("fp16", &NativeVisualPromptRuntime::fp16)
        .def(
            "infer_image",
            &NativeVisualPromptRuntime::infer_image,
            py::arg("image"),
            py::arg("target_h"),
            py::arg("target_w"),
            py::arg("categories"),
            py::arg("bboxes") = py::none(),
            py::arg("masks") = py::none())
        .def(
            "warmup",
            &NativeVisualPromptRuntime::warmup,
            py::arg("target_h"),
            py::arg("target_w"),
            py::arg("prompt_count"),
            py::arg("runs") = 2);

#ifdef YOLOE_TRT_ENABLE_JETSON_CAMERA
    py::class_<NativeJetsonCameraSource, std::shared_ptr<NativeJetsonCameraSource>>(m, "NativeJetsonCameraSource")
        .def(
            py::init<std::string, std::string, double, int, int, bool, bool, int>(),
            py::arg("pipeline"),
            py::arg("prefix"),
            py::arg("timeout_s"),
            py::arg("target_h"),
            py::arg("target_w"),
            py::arg("fp16"),
            py::arg("preview_cpu") = false,
            py::arg("device_id") = 0)
        .def("read_frame", &NativeJetsonCameraSource::read_frame)
        .def("close", &NativeJetsonCameraSource::close);
#endif
}
