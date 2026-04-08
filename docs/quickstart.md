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

## Run the prepared-tensor fast path

```python
from yoloe_tensorrt import YOLOEEngine

engine = YOLOEEngine.from_engine("outputs/artifacts/yoloe26s")
engine.set_classes(["bus"])
prepared = engine.prepare_cuda_input("tests/assets/images/bus.jpg")
result = engine.predict(prepared)[0]
print(result.speed)
```

If you already have a model-ready tensor, keep using the same public API and mark it explicitly:

```python
result = engine.predict(
    your_tensor,
    input_hint="prepared",
    original_image="tests/assets/images/bus.jpg",
    path="bus.jpg",
)[0]
```

## Track objects across frames

```python
from yoloe_tensorrt import YOLOEEngine

engine = YOLOEEngine.from_engine("outputs/artifacts/yoloe26s")
engine.set_classes(["bus"])
results = list(engine.track(["tests/assets/images/bus.jpg"] * 2, stream=True))
print(results[0].boxes.id, results[1].boxes.id)
```

For frame-by-frame application loops, keep tracking state in a session:

```python
tracker = engine.create_tracker(tracker="bytetrack", frame_rate=30)
first = tracker.update("tests/assets/images/bus.jpg")
second = tracker.update("tests/assets/images/bus.jpg")
print(first.boxes.id, second.boxes.id)
```

## Launch the camera GUI

```bash
yoloe-camera-gui
```

Direct RTSP input is also supported:

```bash
yoloe-camera-gui --source rtsp://user:pass@camera.local:554/stream
```

The GUI lets you change:

- the video source
- the active runtime labels
- the confidence threshold
- whether tracking is enabled
- the tracker backend (`bytetrack` or `botsort`)

All from the same window while the stream is running.

## Benchmark current paths

```bash
yoloe-benchmark outputs/artifacts/yoloe26s tests/assets/images/bus.jpg --label bus --mode both
```
