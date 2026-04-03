#include "dlpack.h"

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <NvInferRuntime.h>
#include <opencv2/imgproc.hpp>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <memory>
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

struct TrtDeleter {
    template <typename T>
    void operator()(T* ptr) const
    {
        if (ptr != nullptr) {
            ptr->destroy();
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

struct ManagedTensorContext {
    std::shared_ptr<NativeMainRuntime> owner;
    std::vector<std::int64_t> shape;
    DLManagedTensor managed{};
};

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

        if (engine_->getNbOptimizationProfiles() > 0) {
            context_->setOptimizationProfile(0);
        }
    }

    ~NativeMainRuntime()
    {
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

        auto const preprocess_start = Clock::now();
        preprocess_host_image(image, target_h, target_w);
        auto const preprocess_ms = std::chrono::duration<double, std::milli>(Clock::now() - preprocess_start).count();

        auto const inference_start = Clock::now();
        execute(image_buffer_.data(), target_h, target_w, prompt_buffer_.data(), current_prompt_count_);
        auto const inference_ms = std::chrono::duration<double, std::milli>(Clock::now() - inference_start).count();

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

    void warmup(int target_h, int target_w, int prompt_count, int runs)
    {
        if (runs < 1) {
            runs = 1;
        }

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
            execute(image_buffer_.data(), target_h, target_w, prompt_ptr, prompt_count_to_use);
        }
    }

private:
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
        auto const count = engine_->getNbBindings();
        bindings_.reserve(static_cast<std::size_t>(count));
        output_indices_.clear();
        output_names_.clear();

        for (int i = 0; i < count; ++i) {
            BindingInfo binding;
            binding.index = i;
            binding.name = engine_->getBindingName(i);
            binding.is_input = engine_->bindingIsInput(i);
            binding.dtype = engine_->getBindingDataType(i);
            binding.dims = engine_->getBindingDimensions(i);
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

        cv::Mat const src(orig_h, orig_w, CV_8UC3, const_cast<void*>(info.ptr));

        double const gain = std::min(static_cast<double>(target_h) / static_cast<double>(orig_h), static_cast<double>(target_w) / static_cast<double>(orig_w));
        auto const resized_h = std::max(1, static_cast<int>(std::round(static_cast<double>(orig_h) * gain)));
        auto const resized_w = std::max(1, static_cast<int>(std::round(static_cast<double>(orig_w) * gain)));
        auto const pad_h = target_h - resized_h;
        auto const pad_w = target_w - resized_w;
        auto const top = pad_h / 2;
        auto const left = pad_w / 2;

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
                cudaMemcpyAsync(
                    image_buffer_.data(),
                    host_image_float_.data(),
                    host_image_float_.size() * sizeof(float),
                    cudaMemcpyHostToDevice,
                    stream_),
                "cudaMemcpyAsync for image upload failed");
        }
    }

    void set_input_dimensions(int target_h, int target_w, int prompt_count)
    {
        auto image_dims = image_binding_.dims;
        if (dims_has_dynamic(image_dims)) {
            image_dims.d[0] = 1;
            image_dims.d[2] = target_h;
            image_dims.d[3] = target_w;
            if (!context_->setBindingDimensions(image_input_index_, image_dims)) {
                throw std::runtime_error("Failed to set dynamic image binding dimensions");
            }
        }

        auto prompt_dims = prompt_binding_.dims;
        if (dims_has_dynamic(prompt_dims)) {
            prompt_dims.d[0] = 1;
            prompt_dims.d[1] = prompt_count;
            if (!context_->setBindingDimensions(prompt_input_index_, prompt_dims)) {
                throw std::runtime_error("Failed to set dynamic prompt binding dimensions");
            }
        }

        if (!context_->allInputDimensionsSpecified()) {
            throw std::runtime_error("TensorRT input dimensions are not fully specified");
        }
    }

    void ensure_output_buffers()
    {
        for (std::size_t i = 0; i < output_indices_.size(); ++i) {
            auto const binding_index = output_indices_[i];
            auto const dims = context_->getBindingDimensions(binding_index);
            auto const& binding = bindings_[binding_index];
            auto const bytes = element_count(dims) * dtype_size(binding.dtype);
            output_buffers_[i].ensure(bytes);
            output_shapes_[i] = dims_to_shape(dims);
        }
    }

    void execute(void* image_ptr, int target_h, int target_w, void* prompt_ptr, int prompt_count)
    {
        py::gil_scoped_release release;

        throw_if_cuda_failed(cudaSetDevice(device_id_), "cudaSetDevice failed");
        set_input_dimensions(target_h, target_w, prompt_count);
        ensure_output_buffers();

        std::vector<void*> bindings(static_cast<std::size_t>(engine_->getNbBindings()), nullptr);
        bindings[static_cast<std::size_t>(image_input_index_)] = image_ptr;
        bindings[static_cast<std::size_t>(prompt_input_index_)] = prompt_ptr;
        for (std::size_t i = 0; i < output_indices_.size(); ++i) {
            bindings[static_cast<std::size_t>(output_indices_[i])] = output_buffers_[i].data();
        }

        if (!context_->enqueueV2(bindings.data(), stream_, nullptr)) {
            throw std::runtime_error("TensorRT enqueueV2 failed");
        }
        throw_if_cuda_failed(cudaStreamSynchronize(stream_), "cudaStreamSynchronize failed after inference");
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

    CudaBuffer image_buffer_;
    CudaBuffer prompt_buffer_;
    std::vector<CudaBuffer> output_buffers_;
    std::vector<std::vector<std::int64_t>> output_shapes_;

    std::vector<float> host_image_float_;
    std::vector<__half> host_image_half_;
    std::vector<__half> prompt_half_host_;
};

} // namespace

PYBIND11_MODULE(_native, m)
{
    m.doc() = "Native TensorRT runtime for yoloe_tensorrt";

    py::class_<CudaTensorView>(m, "CudaTensorView")
        .def("__dlpack__", [](CudaTensorView const& self, py::object) { return self.dlpack(); }, py::arg("stream") = py::none())
        .def("__dlpack_device__", &CudaTensorView::dlpack_device);

    py::class_<NativeMainRuntime, std::shared_ptr<NativeMainRuntime>>(m, "NativeMainRuntime")
        .def(py::init<std::string, std::string, std::string, int>(), py::arg("engine_path"), py::arg("image_input_name"), py::arg("prompt_input_name"), py::arg("device_id") = 0)
        .def_property_readonly("output_names", &NativeMainRuntime::output_names)
        .def_property_readonly("fp16", &NativeMainRuntime::fp16)
        .def("clear_prompt_embeddings", &NativeMainRuntime::clear_prompt_embeddings)
        .def("set_prompt_embeddings", &NativeMainRuntime::set_prompt_embeddings, py::arg("prompt_embeddings"))
        .def("infer_image", &NativeMainRuntime::infer_image, py::arg("image"), py::arg("target_h"), py::arg("target_w"))
        .def("warmup", &NativeMainRuntime::warmup, py::arg("target_h"), py::arg("target_w"), py::arg("prompt_count"), py::arg("runs") = 2);
}
