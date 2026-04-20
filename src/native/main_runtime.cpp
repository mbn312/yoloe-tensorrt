#include "native_common.h"

namespace yoloe_native {

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

void bind_native_main_runtime(py::module_& m)
{
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
}

} // namespace yoloe_native
