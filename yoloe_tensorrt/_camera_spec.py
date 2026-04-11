from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

CameraSourceKind = Literal["rtsp", "pipeline", "dummy", "device"]
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


@dataclass(frozen=True)
class CameraSourceSpec:
    raw_value: str
    kind: CameraSourceKind
    device_path: Path | None = None
    dummy_pattern: str | None = None


def is_rtsp_uri(source_value: str | Path) -> bool:
    source_text = str(source_value).strip().lower()
    return source_text.startswith(_RTSP_SCHEMES)


def parse_camera_source_spec(source_value: str | Path) -> CameraSourceSpec:
    source_text = str(source_value).strip()
    lowered = source_text.lower()
    if is_rtsp_uri(source_text):
        return CameraSourceSpec(raw_value=source_text, kind="rtsp")
    if "!" in source_text:
        return CameraSourceSpec(raw_value=source_text, kind="pipeline")
    if lowered in {"dummy", "videotest", "videotestsrc"} or lowered.startswith(("dummy://", "videotest://")):
        pattern = source_text.split("://", 1)[1].strip() if "://" in source_text else "ball"
        return CameraSourceSpec(raw_value=source_text, kind="dummy", dummy_pattern=pattern or "ball")
    return CameraSourceSpec(raw_value=source_text, kind="device", device_path=Path(source_text))
