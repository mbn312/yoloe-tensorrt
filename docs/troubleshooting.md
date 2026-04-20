# Troubleshooting

## Native build fails

Check that these are installed and discoverable:

- TensorRT headers and `libnvinfer`
- CUDA toolkit
- OpenCV headers/libs
- `pybind11`, `scikit-build-core`, and CMake

If editable installation fails during CMake configuration:

- make sure `python -m pybind11 --cmakedir` works in the same environment you are using for `pip install`
- if OpenCV is installed under `/usr/local`, make sure `pkg-config --cflags opencv4` works
- if TensorRT is installed in a non-standard prefix, set `TENSORRT_ROOT=/path/to/TensorRT` before installing
- `pip install tensorrt` alone is not enough for the native build because the native extension also needs the TensorRT
  C++ headers such as `NvInfer.h`
- on Debian/Ubuntu with NVIDIA's TensorRT apt repository configured, a common recovery path is:

```bash
sudo apt-get update
sudo apt-get install -y libnvinfer-dev libnvinfer10
```

  If your stack is on an older TensorRT major, the runtime package may be versioned differently, such as `libnvinfer8`.
  Use `apt-cache search libnvinfer` if needed.
- if you are using `--no-build-isolation`, install the build backend into that environment first:

```bash
python -m pip install scikit-build-core pybind11 cmake ninja
```

On non-Jetson x86_64 systems, this warning is expected and not the cause of the build failure:

```text
Jetson zero-copy camera backend dependencies not found; building without camera backend
```

The Jetson camera backend is optional and only builds when Jetson-specific multimedia dependencies are present.

## Runtime falls back to Python

Set:

```bash
YOLOE_TRT_LOG_LEVEL=DEBUG
```

Then confirm the runtime logs report `native=True` when the engine is created.

## Prompt compilation is slow

The runtime first tries to auto-download `mobileclip2_b.ts` into the package cache. If prompt compilation is still slow, make sure the downloaded asset is available in the artifact bundle or point the runtime at a local copy with `YOLOE_TRT_TEXT_ENCODER`. Otherwise the package falls back to Apple MobileCLIP `b`.

## ONNX export fails

The default exporter mode is `auto`, which retries with the legacy path if the dynamo exporter fails. To force the legacy path directly, use:

```bash
yoloe-export yoloe-26s-seg.pt --artifact-dir outputs/artifacts/yoloe26s --exporter legacy
```

To isolate dynamo exporter failures, use:

```bash
yoloe-export yoloe-26s-seg.pt --artifact-dir outputs/artifacts/yoloe26s --exporter dynamo
```

## GUI first launch takes a long time

The first GUI launch may need to export ONNX and build a TensorRT engine. Subsequent launches reuse `outputs/artifacts/camera_gui_<imgsz>/`.
