# Quickstart

## Export an artifact bundle

```bash
yoloe-export yoloe-26s-seg.pt --artifact-dir outputs/artifacts/yoloe26s --fixed --imgsz 640
```

The example checkpoint is resolved on demand and cached under `~/.cache/yoloe-tensorrt/assets/` unless `YOLOE_TRT_CACHE_DIR` is set.

## Run image inference

```python
from yoloe_tensorrt import YOLOEEngine

engine = YOLOEEngine.from_engine("outputs/artifacts/yoloe26s")
engine.set_classes(["bus", "person"])
results = engine.predict("tests/assets/images/bus.jpg")
print(results[0].names)
```

## Launch the camera GUI

```bash
yoloe-camera-gui
```

The GUI lets you change:

- the video source
- the active runtime labels
- the confidence threshold

All from the same window while the stream is running.
