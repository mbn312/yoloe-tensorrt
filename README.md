# yoloe-tensorrt

`yoloe-tensorrt` is a Jetson-first TensorRT runtime for Ultralytics YOLOE models with runtime-custom text labels and visual prompts. It exposes a normal Python package API, ships a native C++ TensorRT backend for the main inference hot path, and includes a live camera GUI for interactive testing.

## Highlights

- Runtime text labels without rebuilding the main TensorRT engine
- Optional visual prompts for YOLOE detection and segmentation models
- Stateful object tracking with ByteTrack or BoT-SORT on top of YOLOE results
- Native C++ TensorRT main-engine runtime with GPU output handoff back to Python
- Unified `predict(...)` and `track(...)` APIs with a prepared-tensor fast path
- Python package API, export CLI, and live camera GUI
- Jetson-first deployment model with Linux x86_64 CUDA/TensorRT also supported

## Supported Environments

- Primary target: Jetson Orin / JetPack-class deployments
- Also supported: Linux x86_64 with CUDA, TensorRT, and OpenCV available
- Not currently supported as a production target:
  - CPU-only environments
  - Windows
  - macOS

## System Prerequisites

The Python package assumes these platform-level dependencies already exist on the target machine:

- CUDA toolkit and a working NVIDIA GPU runtime
- TensorRT runtime and development headers
- OpenCV development/runtime libraries
- GStreamer if you want live camera input

On Jetson, these are typically supplied by JetPack and system packages. On x86_64 Linux, install the matching CUDA/TensorRT/OpenCV stack for your environment before installing the Python package.

## Installation

Install from a public git repository:

```bash
pip install "yoloe-tensorrt[export,gui] @ git+https://github.com/mbn312/yoloe-tensorrt"
```

If you want the aggregate optional dependency set from package metadata, use:

```bash
pip install "yoloe-tensorrt[all] @ git+https://github.com/mbn312/yoloe-tensorrt"
```

If you need runtime text-label prompting, also install the CLIP tokenizer dependency that Ultralytics expects:

```bash
pip install git+https://github.com/ultralytics/CLIP.git
```

Install from PyPI (Not Implemented Yet):

```bash
pip install "yoloe-tensorrt[export,gui]"
```

If you need runtime text-label prompting from a PyPI install, also install:

```bash
pip install git+https://github.com/ultralytics/CLIP.git
```

Install from a local checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install from a local checkout for development:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

Notes:

- `tensorrt` and system OpenCV are intentionally not hard-pinned as mandatory pip dependencies because Jetson environments commonly provide them outside pip.
- `onnxscript` is part of the export dependency set because newer PyTorch ONNX export flows require it.
- `all` is available as an aggregate extra for package-managed optional dependencies.
- The CLIP tokenizer dependency is currently distributed as a VCS install, so it is listed in `requirements.txt` and `requirements-dev.txt` and shown explicitly above for git and PyPI installs.
- The native extension builds by default during installation. For docs-only or unsupported environments, you can skip it with:

```bash
python -m pip install . --config-settings=cmake.define.YOLOE_TRT_BUILD_NATIVE=OFF
```

## Quickstart

Export a bundle:

```bash
yoloe-export yoloe-26s-seg.pt --artifact-dir outputs/artifacts/yoloe26s --fixed --imgsz 640
```

On newer PyTorch builds, the default ONNX exporter mode is `auto`: it tries the newer dynamo exporter first and falls back to the legacy exporter if YOLOE tracing is incompatible. To force the stable legacy path explicitly:

```bash
yoloe-export yoloe-26s-seg.pt --artifact-dir outputs/artifacts/yoloe26s --exporter legacy
```

Run inference with runtime labels:

```python
from yoloe_tensorrt import YOLOEEngine

engine = YOLOEEngine.from_engine("outputs/artifacts/yoloe26s")
engine.set_classes(["bus", "person"])
results = engine.predict("tests/assets/images/bus.jpg")
print(results[0].boxes.data.shape[0], results[0].names)
```

Run the lowest-overhead headless path through the unified API:

```python
from yoloe_tensorrt import YOLOEEngine

engine = YOLOEEngine.from_engine("outputs/artifacts/yoloe26s")
engine.set_classes(["bus"])
prepared = engine.prepare_cuda_input("tests/assets/images/bus.jpg")
result = engine.predict(prepared)[0]
print(result.boxes.data.shape[0], result.speed)
```

If you already have a model-ready tensor, route it through the same API explicitly:

```python
result = engine.predict(
    your_tensor,
    input_hint="prepared",
    original_image="tests/assets/images/bus.jpg",
    path="bus.jpg",
)[0]
```

Track objects across frames:

```python
from yoloe_tensorrt import YOLOEEngine

engine = YOLOEEngine.from_engine("outputs/artifacts/yoloe26s")
engine.set_classes(["bus"])
results = list(engine.track(["tests/assets/images/bus.jpg"] * 2, stream=True))
print(results[0].boxes.id, results[1].boxes.id)
```

Launch the live camera GUI:

```bash
yoloe-camera-gui
```

Launch the GUI against a synthetic GStreamer source when no camera is attached:

```bash
yoloe-camera-gui --source videotest://ball
```

Launch the GUI against an RTSP stream:

```bash
yoloe-camera-gui --source rtsp://user:pass@camera.local:554/stream
```

Repo-local convenience launcher:

```bash
scripts/launch_camera_gui.sh
```

Benchmark host-image vs CUDA-tensor paths:

```bash
yoloe-benchmark outputs/artifacts/yoloe26s tests/assets/images/bus.jpg --label bus --mode both
```

## CI/CD

GitHub Actions is the supported automation path for this repository.

- CI runs Ruff, runner-safe unit tests, docs validation, source-distribution builds, and clean install smoke tests.
- Tag pushes like `v0.1.0` run a release scaffold that verifies `pyproject.toml` and `CHANGELOG.md`, then uploads release artifacts without publishing them.
- GPU and Jetson-specific validation are intentionally not part of required public CI yet.

## Model Assets and Caching

- The default example checkpoint `yoloe-26s-seg.pt` is resolved on demand and cached under `~/.cache/yoloe-tensorrt/assets/` unless `YOLOE_TRT_CACHE_DIR` is set.
- Exported bundles and test artifacts belong under `outputs/`.
- When a YOLOE text encoder asset such as `mobileclip2_b.ts` is missing locally, the package now attempts to download it into the package cache automatically before falling back.
- You can still override the auto-resolved path with `YOLOE_TRT_TEXT_ENCODER` or `text_encoder=...`.
- If the TorchScript asset cannot be found or downloaded, the runtime falls back to Apple MobileCLIP `b` for prompt compilation. That preserves functionality but can increase prompt-update latency.

## Runtime Notes

- The main inference engine path is native C++/TensorRT.
- Prompt compilation, visual prompt orchestration, and Ultralytics `Results` wrapping still live in Python.
- Tracking is stateful and currently runs in Python on top of the detection/segmentation results.
- `predict(...)` and `track(...)` choose the fastest supported internal path for the input representation they are given.
- The prepared-tensor fast path is reached by passing the object returned from `prepare_cuda_input(...)` or by using `input_hint="prepared"`.
- Plain file paths, PIL images, NumPy arrays, CPU tensors, and CUDA tensors are still accepted without requiring manual preprocessing.
- For production deployments, prefer `YOLOEEngine.from_engine(...)` and prebuilt bundles over `from_pt(...)`.
- Live USB camera input is available through a GStreamer appsink pipeline. Jetson zero-copy camera ingest is still on the roadmap.
- Direct RTSP URLs such as `rtsp://camera.local/stream` are supported and resolved to a GStreamer RTSP pipeline automatically.
- If you do not have a camera attached, use `videotest://<pattern>` such as `videotest://ball` or `videotest://smpte`.

## Repository Layout

- `yoloe_tensorrt/`: public Python package, runtime orchestration, export flow, GUI
- `src/native/`: native TensorRT backend
- `tests/`: unit and integration tests plus checked-in image fixtures
- `outputs/`: generated artifacts, caches, logs, and scratch output
- `docs/`: user and contributor documentation

## Documentation

- [Installation](docs/installation.md)
- [Quickstart](docs/quickstart.md)
- [Export and Artifact Bundles](docs/export.md)
- [Runtime Prompts](docs/prompts.md)
- [Camera GUI](docs/camera-gui.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Development](docs/development.md)

## Status

The project is usable today for Jetson-focused deployments, but it is still early-stage. The remaining performance roadmap is centered on:

- Jetson zero-copy NVMM/EGL/CUDA camera ingest
- CUDA preprocess instead of CPU/OpenCV preprocess
- Native decode/NMS/mask reconstruction
- Native visual-prompt execution

## License

MIT. See [LICENSE](LICENSE).
