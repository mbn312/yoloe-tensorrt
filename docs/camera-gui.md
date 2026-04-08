# Camera GUI

## Launch

Installed command:

```bash
yoloe-camera-gui
```

Repo-local launcher:

```bash
scripts/launch_camera_gui.sh
```

## Behavior

The GUI opens a single window that contains:

- the live video stream
- source controls
- editable label controls
- a live confidence threshold control
- a tracking toggle
- a tracker backend selector (`bytetrack` or `botsort`)

Tracking is enabled by default. When the tracker is active, rendered overlays use instance coloring and display
track IDs when detections are matched across frames.

## Source handling

- `/dev/video*` device paths are wrapped in the package's GStreamer USB camera pipeline
- `rtsp://...` and `rtsps://...` URLs are wrapped in a package-managed GStreamer RTSP pipeline
- raw GStreamer pipeline strings are also accepted
- `videotest://<pattern>` creates a synthetic GStreamer source, for example `videotest://ball` or `videotest://smpte`

## Current limitation

The camera path still uses an appsink/CPU-memory ingest path. Jetson zero-copy NVMM/EGL/CUDA ingest is still roadmap work.
