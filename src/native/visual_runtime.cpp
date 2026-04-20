#include "native_common.h"

namespace yoloe_native {

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

void bind_native_visual_prompt_runtime(py::module_& m)
{
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
}

} // namespace yoloe_native
