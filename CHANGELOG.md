# Changelog

## Unreleased

## 0.2.0

- Unified `predict(...)` and `track(...)` around a broad input surface with specialized routing for paths, arrays, raw tensors, prepared CUDA tensors, and live sources.
- Added `PreparedTensorInput`, prepared CUDA-tensor prediction/tracking paths, and tracker-session support for prepared tensor updates.
- Added first-class RTSP input support for plain `rtsp://...` and `rtsps://...` sources without requiring custom GStreamer pipelines.
- Added Jetson zero-copy camera ingest using NVMM/EGL/CUDA interop, prepared tensor frame handoff, and CPU appsink fallback when zero-copy is unavailable.
- Replaced native CPU/OpenCV preprocessing with shared CUDA preprocess kernels for resize, letterbox, color conversion, normalization, and tensor packing.
- Moved main-runtime decode, NMS, box rescale, and segmentation mask reconstruction into the native backend.
- Moved visual-prompt TensorRT execution into the native backend and syncs prompt embeddings on device to avoid GPU-to-CPU-to-GPU round trips.
- Reduced steady-state Python allocations in prediction and tracking hot paths through cached prompt names, cached native metadata, and leaner result construction.
- Added the `yoloe-benchmark` CLI with host, CUDA, camera, text-prompt, visual-prompt, tracking, allocation, JSON output, and baseline-regression comparison modes.
- Documented the full runtime benchmark matrix for CPU-memory input, CUDA-tensor input, Jetson zero-copy camera input, text-prompt update cost, and visual-prompt update cost.
- Improved the live camera GUI with source controls, editable labels, confidence controls, tracking controls, RTSP/videotest source handling, and Jetson zero-copy capture when available.
- Fixed prepared-tensor correctness issues around integer dtype validation, CUDA producer-stream handoff, unit-range float inputs, cross-stream inference, and artifact `max_det` defaults.
- Expanded unit and integration coverage for input routing, native fast paths, zero-copy camera ingest, RTSP routing, benchmark CLI behavior, tracking, GUI helpers, and release metadata.

## 0.1.3

- Implemented stateful object tracking using ByteTrack or BoT-SORT on top of YOLOE results

## 0.1.2

- Set logging level of YOLOE inference messages to debug

## 0.1.1

- Added in missing dependencies and setup information
- Added auto/legacy/dynamo ONNX export modes for better suppport across PyTorch versions
- Fixed some bugs with the native build config in CMakeList.txt
- Added `all` option for building with extras
- Updated native backend builds to be compatible with newer TensorRT versions

## 0.1.0

- Initial package release.
