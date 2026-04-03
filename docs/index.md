# yoloe-tensorrt

`yoloe-tensorrt` packages a YOLOE TensorRT runtime that keeps runtime-custom prompts available from Python while moving the main inference hot path into a native backend.

## What it is for

- Jetson-first YOLOE deployments
- Fast runtime label changes without rebuilding the main engine
- Production use through prebuilt artifact bundles
- Live camera experimentation through a bundled GUI

## What it is not

- A CPU-only inference package
- A Windows or macOS deployment target
- A full native end-to-end Jetson zero-copy pipeline yet

## Core entry points

- `YOLOEEngine`
- `export_model(...)`
- `yoloe-export`
- `yoloe-camera-gui`
- `python -m yoloe_tensorrt`

See the rest of the docs for installation, artifact bundles, prompts, and deployment guidance.
