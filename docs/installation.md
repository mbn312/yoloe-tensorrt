# Installation

## Supported platforms

- Jetson Orin / JetPack-class systems: primary target
- Linux x86_64 with CUDA and TensorRT: supported

## Required system components

Install these before the Python package:

- CUDA toolkit/runtime
- TensorRT runtime and development headers
- OpenCV
- GStreamer for live camera input

## Install from git

```bash
pip install "yoloe-tensorrt[export,gui] @ git+https://github.com/mbn312/yoloe-tensorrt"
```

If you need runtime text-label prompting, also install:

```bash
pip install git+https://github.com/ultralytics/CLIP.git
```

## Install from PyPI (Not Implemented Yet)

```bash
pip install "yoloe-tensorrt[export,gui]"
```

If you need runtime text-label prompting from a PyPI install, also install:

```bash
pip install git+https://github.com/ultralytics/CLIP.git
```

## Install from a local checkout

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Install from a local checkout for development

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

## Native build notes

The native TensorRT extension builds by default. If you only need docs or lightweight non-runtime checks:

```bash
python -m pip install . --config-settings=cmake.define.YOLOE_TRT_BUILD_NATIVE=OFF
```

That mode is not a production deployment mode. It exists for packaging, docs, and limited development workflows.

## Python dependency notes

- `onnxscript` is required for the export path on newer PyTorch ONNX exporters and is included in the export dependency set.
- The Ultralytics CLIP tokenizer dependency is currently installed from `git+https://github.com/ultralytics/CLIP.git`, so source-checkout installs use `requirements.txt` / `requirements-dev.txt` and git/PyPI installs need the extra command above when you use runtime text labels.
