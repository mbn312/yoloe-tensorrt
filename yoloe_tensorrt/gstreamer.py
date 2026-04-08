from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .logging_utils import get_logger
from .source import SourceItem, SourceStream

LOGGER = get_logger(__name__)

_GST = None
_GST_APP = None
_GST_VIDEO = None
_RTSP_SCHEMES = (
    "rtsp://",
    "rtsps://",
    "rtspu://",
    "rtspt://",
    "rtsph://",
    "rtsp-sdp://",
    "rtspsu://",
    "rtspst://",
    "rtspsh://",
)


def _require_gst():
    global _GST, _GST_APP, _GST_VIDEO
    if _GST is not None and _GST_APP is not None and _GST_VIDEO is not None:
        return _GST, _GST_APP, _GST_VIDEO

    try:
        import gi

        gi.require_version("Gst", "1.0")
        gi.require_version("GstApp", "1.0")
        gi.require_version("GstVideo", "1.0")
        from gi.repository import Gst, GstApp, GstVideo
    except Exception as exc:  # pragma: no cover - exercised in integration environments
        raise RuntimeError(
            "GStreamer Python bindings are required for GStreamer camera input. "
            "Install or expose `gi.repository.Gst` in the runtime environment."
        ) from exc

    Gst.init(None)
    _GST = Gst
    _GST_APP = GstApp
    _GST_VIDEO = GstVideo
    return Gst, GstApp, GstVideo


def is_rtsp_uri(source_value: str | Path) -> bool:
    source_text = str(source_value).strip().lower()
    return source_text.startswith(_RTSP_SCHEMES)


def _gst_quote(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def build_usb_camera_pipeline(
    device: str | Path = "/dev/video0",
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    prefer_mjpeg: bool = True,
    io_mode: str = "mmap",
    appsink_name: str = "sink",
) -> str:
    device_path = str(Path(device))
    framerate = f"{int(fps)}/1"
    source = f"v4l2src device={device_path} io-mode={io_mode} do-timestamp=true"
    if prefer_mjpeg:
        caps = f"image/jpeg,width={int(width)},height={int(height)},framerate={framerate}"
        decode = "jpegparse ! jpegdec"
    else:
        caps = f"video/x-raw,format=YUY2,width={int(width)},height={int(height)},framerate={framerate}"
        decode = ""

    pipeline = f"{source} ! {caps} ! "
    if decode:
        pipeline += f"{decode} ! "
    pipeline += (
        f"videoconvert ! video/x-raw,format=BGR ! "
        f"appsink name={appsink_name} emit-signals=false max-buffers=1 drop=true sync=false"
    )
    return pipeline


def build_rtsp_pipeline(
    uri: str,
    width: int | None = None,
    height: int | None = None,
    fps: int | None = None,
    appsink_name: str = "sink",
) -> str:
    pipeline = (
        f"uridecodebin uri={_gst_quote(str(uri).strip())} use-buffering=false ! "
        "queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream ! "
        "autovideoconvert ! video/x-raw ! "
    )

    caps_parts = ["format=BGR"]
    if width is not None or height is not None or fps is not None:
        pipeline += "videoscale ! videorate ! "
        if width is not None:
            caps_parts.append(f"width={int(width)}")
        if height is not None:
            caps_parts.append(f"height={int(height)}")
        if fps is not None:
            caps_parts.append(f"framerate={int(fps)}/1")

    caps = "video/x-raw," + ",".join(caps_parts)
    pipeline += (
        f"videoconvert ! {caps} ! appsink name={appsink_name} emit-signals=false max-buffers=1 drop=true sync=false"
    )
    return pipeline


def build_dummy_video_pipeline(
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    pattern: str = "ball",
    appsink_name: str = "sink",
) -> str:
    framerate = f"{int(fps)}/1"
    return (
        f"videotestsrc is-live=true do-timestamp=true pattern={pattern} ! "
        f"video/x-raw,width={int(width)},height={int(height)},framerate={framerate} ! "
        f"videoconvert ! video/x-raw,format=BGR ! "
        f"appsink name={appsink_name} emit-signals=false max-buffers=1 drop=true sync=false"
    )


def camera_source_from_spec(
    source_value: str | Path,
    *,
    width: int | None = None,
    height: int | None = None,
    fps: int | None = None,
    timeout_s: float = 5.0,
    prefix: str = "camera",
    max_frames: int | None = None,
) -> GStreamerSource:
    source_text = str(source_value).strip()
    lowered = source_text.lower()
    resolved_width = 640 if width is None else int(width)
    resolved_height = 480 if height is None else int(height)
    resolved_fps = 30 if fps is None else int(fps)
    if is_rtsp_uri(source_text):
        return GStreamerSource.rtsp(
            source_text,
            width=width,
            height=height,
            fps=fps,
            max_frames=max_frames,
            timeout_s=timeout_s,
            prefix=prefix,
        )
    if "!" in source_text:
        return GStreamerSource(
            pipeline=source_text,
            prefix=prefix,
            max_frames=max_frames,
            timeout_s=timeout_s,
        )
    if lowered in {"dummy", "videotest", "videotestsrc"} or lowered.startswith(("dummy://", "videotest://")):
        pattern = source_text.split("://", 1)[1].strip() if "://" in source_text else "ball"
        pattern = pattern or "ball"
        return GStreamerSource(
            pipeline=build_dummy_video_pipeline(
                width=resolved_width,
                height=resolved_height,
                fps=resolved_fps,
                pattern=pattern,
            ),
            prefix=prefix,
            max_frames=max_frames,
            timeout_s=timeout_s,
        )

    device_path = Path(source_text)
    if not device_path.exists():
        raise FileNotFoundError(f"Camera source device is missing: {device_path}")
    return GStreamerSource.usb_camera(
        device=device_path,
        width=resolved_width,
        height=resolved_height,
        fps=resolved_fps,
        max_frames=max_frames,
        timeout_s=timeout_s,
        prefix=prefix,
    )


@dataclass
class GStreamerSource(SourceStream):
    pipeline: str
    prefix: str = "gst"
    max_frames: int | None = None
    timeout_s: float = 5.0
    appsink_name: str = "sink"
    is_live_source: bool = True

    @classmethod
    def rtsp(
        cls,
        uri: str,
        width: int | None = None,
        height: int | None = None,
        fps: int | None = None,
        max_frames: int | None = None,
        timeout_s: float = 5.0,
        prefix: str | None = None,
    ) -> GStreamerSource:
        return cls(
            pipeline=build_rtsp_pipeline(
                uri,
                width=width,
                height=height,
                fps=fps,
            ),
            prefix=prefix or "rtsp",
            max_frames=max_frames,
            timeout_s=timeout_s,
        )

    @classmethod
    def usb_camera(
        cls,
        device: str | Path = "/dev/video0",
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        prefer_mjpeg: bool = True,
        io_mode: str = "mmap",
        max_frames: int | None = None,
        timeout_s: float = 5.0,
        prefix: str | None = None,
    ) -> GStreamerSource:
        device_path = Path(device)
        source_prefix = prefix or f"{device_path.stem or 'camera'}"
        return cls(
            pipeline=build_usb_camera_pipeline(
                device=device_path,
                width=width,
                height=height,
                fps=fps,
                prefer_mjpeg=prefer_mjpeg,
                io_mode=io_mode,
            ),
            prefix=source_prefix,
            max_frames=max_frames,
            timeout_s=timeout_s,
        )

    @property
    def timeout_ns(self) -> int:
        return int(self.timeout_s * 1_000_000_000)

    def __iter__(self):
        Gst, GstApp, GstVideo = _require_gst()
        LOGGER.info("Opening GStreamer source '%s' with pipeline: %s", self.prefix, self.pipeline)
        pipeline = Gst.parse_launch(self.pipeline)
        appsink = pipeline.get_by_name(self.appsink_name)
        if appsink is None:
            raise RuntimeError(f"GStreamer pipeline did not expose appsink '{self.appsink_name}': {self.pipeline}")
        if not isinstance(appsink, GstApp.AppSink):
            raise RuntimeError(f"Pipeline element '{self.appsink_name}' is not a GstApp.AppSink")

        bus = pipeline.get_bus()
        state_result = pipeline.set_state(Gst.State.PLAYING)
        if state_result == Gst.StateChangeReturn.FAILURE:
            pipeline.set_state(Gst.State.NULL)
            raise RuntimeError(f"Unable to start GStreamer pipeline: {self.pipeline}")

        try:
            pipeline.get_state(self.timeout_ns)
            frame_index = 0
            while self.max_frames is None or frame_index < self.max_frames:
                sample = appsink.try_pull_sample(self.timeout_ns)
                if sample is None:
                    if appsink.is_eos():
                        LOGGER.info("Received EOS from GStreamer source '%s'", self.prefix)
                        break
                    message = bus.timed_pop_filtered(
                        0,
                        Gst.MessageType.ERROR | Gst.MessageType.EOS | Gst.MessageType.WARNING,
                    )
                    if message is not None:
                        if message.type == Gst.MessageType.ERROR:
                            error, debug = message.parse_error()
                            raise RuntimeError(f"GStreamer source error: {error}; debug={debug}")
                        if message.type == Gst.MessageType.WARNING:
                            warning, debug = message.parse_warning()
                            LOGGER.warning("GStreamer warning from '%s': %s; debug=%s", self.prefix, warning, debug)
                    raise TimeoutError(
                        "Timed out waiting for a frame from GStreamer source "
                        f"'{self.prefix}' after {self.timeout_s:.1f}s"
                    )

                image = _sample_to_bgr(sample, GstVideo)
                sample = None
                path = f"{self.prefix}_frame{frame_index:06d}"
                LOGGER.debug("Captured GStreamer frame '%s' with shape=%s", path, image.shape)
                yield SourceItem(image=image, path=path)
                frame_index += 1
        finally:
            appsink = None
            bus = None
            pipeline.set_state(Gst.State.NULL)
            pipeline.get_state(self.timeout_ns)
            pipeline = None
            LOGGER.info("Closed GStreamer source '%s'", self.prefix)


def _sample_to_bgr(sample, GstVideo) -> np.ndarray:
    caps = sample.get_caps()
    if caps is None:
        raise RuntimeError("GStreamer sample did not include caps")
    structure = caps.get_structure(0)
    width = int(structure.get_value("width"))
    height = int(structure.get_value("height"))
    format_name = str(structure.get_value("format"))
    if format_name != "BGR":
        raise RuntimeError(f"Expected BGR appsink frames, got '{format_name}'")

    video_info = GstVideo.VideoInfo()
    if not video_info.from_caps(caps):
        raise RuntimeError("Unable to parse GStreamer video caps")
    stride = int(video_info.stride[0]) if getattr(video_info, "stride", None) else 0

    buffer = sample.get_buffer()
    if buffer is None:
        raise RuntimeError("GStreamer sample did not include a buffer")

    ok, map_info = buffer.map(_require_gst()[0].MapFlags.READ)
    if not ok:
        raise RuntimeError("Unable to map GStreamer sample buffer")

    try:
        view = np.frombuffer(map_info.data, dtype=np.uint8)
        if stride <= 0:
            if height <= 0 or view.size % height != 0:
                raise RuntimeError(
                    f"Unable to infer row stride from mapped buffer size={view.size} and height={height}"
                )
            stride = view.size // height
        if stride < width * 3:
            raise RuntimeError(f"Invalid row stride {stride} for width={width} and BGR format")
        frame = view.reshape((height, stride))[:, : width * 3].reshape((height, width, 3))
        return np.ascontiguousarray(frame)
    finally:
        buffer.unmap(map_info)
