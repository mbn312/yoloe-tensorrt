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
#include <deque>
#include <memory>
#include <mutex>
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
    std::shared_ptr<NativeMainRuntime> owner;
    std::vector<std::int64_t> shape;
    DLManagedTensor managed{};
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
    DLManagedTensor managed{};
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
#endif

class CudaTensorView {
public:
    CudaTensorView(
        std::shared_ptr<NativeMainRuntime> owner,
        void* ptr,
        std::vector<std::int64_t> shape,
        nvinfer1::DataType dtype,
        int device_id)
        : owner_(std::move(owner))
        , ptr_(ptr)
        , shape_(std::move(shape))
        , dtype_(dtype)
        , device_id_(device_id)
    {
    }

    py::capsule dlpack() const
    {
        auto* ctx = new ManagedTensorContext();
        ctx->owner = owner_;
        ctx->shape = shape_;
        ctx->managed.manager_ctx = ctx;
        ctx->managed.deleter = [](DLManagedTensor* self) {
            auto* managed_ctx = static_cast<ManagedTensorContext*>(self->manager_ctx);
            delete managed_ctx;
        };
        ctx->managed.dl_tensor.data = ptr_;
        ctx->managed.dl_tensor.device = DLDevice{kDLCUDA, device_id_};
        ctx->managed.dl_tensor.ndim = static_cast<int>(ctx->shape.size());
        ctx->managed.dl_tensor.dtype = to_dlpack_dtype(dtype_);
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
    std::shared_ptr<NativeMainRuntime> owner_;
    void* ptr_ = nullptr;
    std::vector<std::int64_t> shape_;
    nvinfer1::DataType dtype_ = nvinfer1::DataType::kFLOAT;
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
        ctx->managed.manager_ctx = ctx;
        ctx->managed.deleter = [](DLManagedTensor* self) {
            auto* managed_ctx = static_cast<ManagedCameraTensorContext*>(self->manager_ctx);
            delete managed_ctx;
        };
        ctx->managed.dl_tensor.data = ptr_;
        ctx->managed.dl_tensor.device = DLDevice{kDLCUDA, device_id_};
        ctx->managed.dl_tensor.ndim = static_cast<int>(ctx->shape.size());
        ctx->managed.dl_tensor.dtype = to_dlpack_dtype(dtype_);
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
    nvinfer1::DataType dtype_ = nvinfer1::DataType::kFLOAT;
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
                CudaTensorView(shared_from_this(), output_buffers_[i].data(), output_shapes_[i], binding.dtype, device_id_));
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
        .def("__dlpack__", [](CudaTensorView const& self, py::object) { return self.dlpack(); }, py::arg("stream") = py::none())
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
        .def("clear_prompt_embeddings", &NativeMainRuntime::clear_prompt_embeddings)
        .def("set_prompt_embeddings", &NativeMainRuntime::set_prompt_embeddings, py::arg("prompt_embeddings"))
        .def("infer_image", &NativeMainRuntime::infer_image, py::arg("image"), py::arg("target_h"), py::arg("target_w"))
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
        .def("warmup", &NativeMainRuntime::warmup, py::arg("target_h"), py::arg("target_w"), py::arg("prompt_count"), py::arg("runs") = 2);

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
