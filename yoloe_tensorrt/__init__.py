from ._version import __version__

__all__ = [
    "__version__",
    "YOLOEEngine",
    "export_model",
    "configure_logging",
    "GStreamerSource",
    "build_usb_camera_pipeline",
    "build_dummy_video_pipeline",
    "camera_source_from_spec",
    "run_camera_gui",
]


def __getattr__(name: str):
    if name == "__version__":
        return __version__
    if name == "YOLOEEngine":
        from .engine import YOLOEEngine

        return YOLOEEngine
    if name == "export_model":
        from .export import export_model

        return export_model
    if name == "configure_logging":
        from .logging_utils import configure_logging

        return configure_logging
    if name == "GStreamerSource":
        from .gstreamer import GStreamerSource

        return GStreamerSource
    if name == "build_usb_camera_pipeline":
        from .gstreamer import build_usb_camera_pipeline

        return build_usb_camera_pipeline
    if name == "build_dummy_video_pipeline":
        from .gstreamer import build_dummy_video_pipeline

        return build_dummy_video_pipeline
    if name == "camera_source_from_spec":
        from .gstreamer import camera_source_from_spec

        return camera_source_from_spec
    if name == "run_camera_gui":
        from .gui import run_camera_gui

        return run_camera_gui
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(__all__)
