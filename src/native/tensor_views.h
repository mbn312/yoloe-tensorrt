#pragma once

#include "postprocess_common.h"

namespace yoloe_native {

struct ManagedTensorContext {
    std::shared_ptr<void> owner;
    std::shared_ptr<class PendingPostprocessOutputs> pending_postprocess_outputs;
    std::vector<std::int64_t> shape;
    DLDataType dtype{};
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
        throw_if_cuda_failed(
            cudaEventRecord(ready_event, stream),
            "cudaEventRecord failed for postprocess output event");
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

} // namespace yoloe_native
