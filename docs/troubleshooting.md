# Troubleshooting

## Native build fails

Check that these are installed and discoverable:

- TensorRT headers and `libnvinfer`
- CUDA toolkit
- OpenCV headers/libs
- `pybind11`, `scikit-build-core`, and CMake

## Runtime falls back to Python

Set:

```bash
YOLOE_TRT_LOG_LEVEL=DEBUG
```

Then confirm the runtime logs report `native=True` when the engine is created.

## Prompt compilation is slow

The runtime first tries to auto-download `mobileclip2_b.ts` into the package cache. If prompt compilation is still slow, make sure the downloaded asset is available in the artifact bundle or point the runtime at a local copy with `YOLOE_TRT_TEXT_ENCODER`. Otherwise the package falls back to Apple MobileCLIP `b`.

## ONNX export fails on newer PyTorch

The default exporter mode is `auto`, which already retries with the legacy path if the newer dynamo exporter fails. If you want to force the older path directly, use:

```bash
yoloe-export yoloe-26s-seg.pt --artifact-dir outputs/artifacts/yoloe26s --exporter legacy
```

If you are debugging a newer torch environment, you can also try:

```bash
yoloe-export yoloe-26s-seg.pt --artifact-dir outputs/artifacts/yoloe26s --exporter dynamo
```

## GUI first launch takes a long time

The first GUI launch may need to export ONNX and build a TensorRT engine. Subsequent launches reuse `outputs/artifacts/camera_gui_<imgsz>/`.
