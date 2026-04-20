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
- a source selector with detected camera sources and custom source entry
- editable label controls
- a live confidence threshold control
- a tracking toggle
- a tracker backend selector (`bytetrack` or `botsort`)

Tracking is enabled by default. When the tracker is active, rendered overlays use instance coloring and display track IDs
when detections are matched across frames. Applied source, model, confidence, labels, and tracker settings are displayed
below the video stream.

## Source handling

- `/dev/video*` device paths are wrapped in the package's GStreamer USB camera pipeline
- `rtsp://...` and `rtsps://...` URLs are wrapped in a package-managed GStreamer RTSP pipeline
- raw GStreamer pipeline strings are also accepted
- `videotest://<pattern>` creates a synthetic GStreamer source, for example `videotest://ball` or `videotest://smpte`

## Jetson zero-copy path

On Jetson builds that include the native camera backend, the GUI requests the zero-copy NVMM/EGL/CUDA ingest path
automatically and only materializes a CPU preview frame for rendering the window. If zero-copy is unavailable or the
source cannot negotiate the required NVMM pipeline, the GUI falls back to the CPU appsink path.
