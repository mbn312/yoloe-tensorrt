#include "native_common.h"

namespace yoloe_native {

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

void bind_native_jetson_camera_source(py::module_& m)
{
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
}
#else
void bind_native_jetson_camera_source(py::module_&) {}
#endif

} // namespace yoloe_native
