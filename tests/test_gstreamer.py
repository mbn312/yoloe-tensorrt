from __future__ import annotations

from yoloe_tensorrt import build_dummy_video_pipeline, build_usb_camera_pipeline, camera_source_from_spec


def test_build_usb_camera_pipeline_defaults_to_mjpeg_bgr_appsink() -> None:
    pipeline = build_usb_camera_pipeline("/dev/video0", width=640, height=480, fps=30)

    assert "v4l2src device=/dev/video0" in pipeline
    assert "image/jpeg,width=640,height=480,framerate=30/1" in pipeline
    assert "jpegparse ! jpegdec" in pipeline
    assert "video/x-raw,format=BGR" in pipeline
    assert "appsink name=sink" in pipeline


def test_build_usb_camera_pipeline_supports_raw_yuy2_capture() -> None:
    pipeline = build_usb_camera_pipeline(
        "/dev/video0",
        width=1280,
        height=720,
        fps=15,
        prefer_mjpeg=False,
    )

    assert "video/x-raw,format=YUY2,width=1280,height=720,framerate=15/1" in pipeline
    assert "image/jpeg" not in pipeline
    assert "jpegdec" not in pipeline


def test_build_dummy_video_pipeline_uses_videotestsrc_bgr_appsink() -> None:
    pipeline = build_dummy_video_pipeline(width=800, height=600, fps=24, pattern="smpte")

    assert "videotestsrc is-live=true do-timestamp=true pattern=smpte" in pipeline
    assert "video/x-raw,width=800,height=600,framerate=24/1" in pipeline
    assert "video/x-raw,format=BGR" in pipeline
    assert "appsink name=sink" in pipeline


def test_camera_source_from_spec_supports_dummy_alias() -> None:
    source = camera_source_from_spec(
        "videotest://zone-plate",
        width=320,
        height=240,
        fps=15,
        prefix="dummy",
        max_frames=3,
    )

    assert source.prefix == "dummy"
    assert source.max_frames == 3
    assert "videotestsrc" in source.pipeline
    assert "pattern=zone-plate" in source.pipeline
    assert "width=320,height=240,framerate=15/1" in source.pipeline
