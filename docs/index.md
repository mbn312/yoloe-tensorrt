# yoloe-tensorrt

`yoloe-tensorrt` packages a YOLOE TensorRT runtime that keeps runtime-custom prompts available from Python while moving
the main inference hot path into a native backend.

## What it is for

- Jetson-first YOLOE deployments
- Fast runtime label changes without rebuilding the main engine
- Production use through prebuilt artifact bundles
- Live USB, RTSP, and Jetson zero-copy camera experimentation through a bundled GUI
- Runtime benchmark checks across input paths and prompt update costs
- Dataset config validation before YOLOE training and fine-tuning workflows

## What it is not

- A CPU-only inference package
- A Windows or macOS deployment target
- A replacement for validating TensorRT engines and camera pipelines on the target deployment hardware

## Core entry points

- `YOLOEEngine`
- `export_model(...)`
- `yoloe-export`
- `yoloe-camera-gui`
- `yoloe-benchmark`
- `validate_dataset_config(...)`
- `python -m yoloe_tensorrt`

See the rest of the docs for installation, artifact bundles, prompts, training data validation, benchmarks, camera input,
and deployment guidance.
