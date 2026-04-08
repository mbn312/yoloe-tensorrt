from __future__ import annotations

from yoloe_tensorrt import (
    build_dummy_video_pipeline,
    build_rtsp_pipeline,
    build_usb_camera_pipeline,
    camera_source_from_spec,
)


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


def test_build_rtsp_pipeline_uses_uridecodebin_bgr_appsink() -> None:
    pipeline = build_rtsp_pipeline("rtsp://camera.local:554/stream", width=1280, height=720, fps=15)

    assert 'uridecodebin uri="rtsp://camera.local:554/stream" use-buffering=false' in pipeline
    assert "queue max-size-buffers=1" in pipeline
    assert "autovideoconvert" in pipeline
    assert "video/x-raw !" in pipeline
    assert "videoconvert" in pipeline
    assert "videoscale" in pipeline
    assert "videorate" in pipeline
    assert "video/x-raw,format=BGR,width=1280,height=720,framerate=15/1" in pipeline
    assert "appsink name=sink" in pipeline


def test_build_rtsp_pipeline_preserves_credentials_and_query() -> None:
    uri = "rtsp://user:pass@example.com:8554/stream?profile=main&channel=1"
    pipeline = build_rtsp_pipeline(uri, width=None, height=None, fps=None)

    assert f'uridecodebin uri="{uri}"' in pipeline
    assert "autovideoconvert" in pipeline
    assert "videoscale" not in pipeline
    assert "videorate" not in pipeline
    assert "video/x-raw,format=BGR" in pipeline


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


def test_camera_source_from_spec_supports_rtsp_uri() -> None:
    source = camera_source_from_spec(
        "rtsp://camera.local/stream",
        width=800,
        height=450,
        fps=20,
        prefix="rtspcam",
        max_frames=2,
    )

    assert source.prefix == "rtspcam"
    assert source.max_frames == 2
    assert 'uridecodebin uri="rtsp://camera.local/stream"' in source.pipeline
    assert "video/x-raw,format=BGR,width=800,height=450,framerate=20/1" in source.pipeline


def test_camera_source_from_spec_preserves_rtsp_native_caps_when_unspecified() -> None:
    source = camera_source_from_spec(
        "rtsp://camera.local/stream",
        prefix="rtspnative",
    )

    assert source.prefix == "rtspnative"
    assert 'uridecodebin uri="rtsp://camera.local/stream"' in source.pipeline
    assert "videoscale" not in source.pipeline
    assert "videorate" not in source.pipeline
    assert "video/x-raw,format=BGR" in source.pipeline


def test_camera_source_from_spec_prioritizes_rtsp_uri_over_bang_character() -> None:
    uri = "rtsp://user:pa!ss@camera.local:8554/stream?profile=main&flag=1"

    source = camera_source_from_spec(uri, prefix="rtspbang")

    assert source.prefix == "rtspbang"
    assert source.pipeline != uri
    assert f'uridecodebin uri="{uri}"' in source.pipeline


def test_camera_source_from_spec_keeps_explicit_rtspsrc_pipeline_literal() -> None:
    pipeline = (
        'rtspsrc location="rtsp://camera.local/stream" latency=0 ! '
        "rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink name=sink"
    )

    source = camera_source_from_spec(pipeline, prefix="manual")

    assert source.prefix == "manual"
    assert source.pipeline == pipeline
