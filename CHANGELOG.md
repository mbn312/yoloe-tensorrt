# Changelog

## 0.2.0

## Unreleased

- Unified `predict(...)` and `track(...)` around a single broad input surface with specialized internal routing for paths, arrays, raw tensors, prepared CUDA tensors, and live sources.
- Added `PreparedTensorInput`, a prepared CUDA-tensor fast path, and a `benchmark` CLI for comparing host-image and CUDA-tensor inference through the native TensorRT runtime.
- Extended tracker sessions to use the unified input router, support prepared tensor updates, and preserve the artifact default `max_det` behavior.
- Fixed prepared-tensor correctness issues around integer dtype validation, CUDA producer-stream handoff, and cross-stream prepared-tensor inference.
- Fixed unit-range float preprocessing so non-square padded inputs keep the correct scale instead of being divided by `255` a second time.
- Expanded runtime and unit coverage for unified routing, benchmark CLI help, tracker-session input handling, and prepared-tensor regressions.
- Added first-class RTSP input support so plain `rtsp://...` and `rtsps://...` sources can be used without writing a custom GStreamer pipeline.
- Added an optional Jetson zero-copy camera ingest path that maps NVMM frames through EGL/CUDA, emits prepared tensors for inference, and falls back to the CPU appsink path when zero-copy is unavailable.

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
