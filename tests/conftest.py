from __future__ import annotations

import os
import sys
import tkinter as tk
from pathlib import Path

import pytest
import torch
from yoloe_tensorrt import (
    GStreamerSource,
    YOLOEEngine,
    camera_source_from_spec,
    configure_logging,
    export_model,
)
from yoloe_tensorrt.assets import default_example_model_spec, resolve_model_checkpoint

TESTS_ROOT = Path(__file__).resolve().parent
TEST_MODEL_SPEC = os.environ.get("YOLOE_TRT_TEST_MODEL", default_example_model_spec())
TEST_IMAGES = {
    "bus": TESTS_ROOT / "assets" / "images" / "bus.jpg",
    "zidane": TESTS_ROOT / "assets" / "images" / "zidane.jpg",
    "dog": TESTS_ROOT / "assets" / "images" / "dog.jpg",
}
USB_CAMERA_DEVICE = Path(os.environ.get("YOLOE_TRT_TEST_CAMERA_DEVICE", "/dev/video0"))
DEFAULT_CAMERA_SOURCE = (
    os.environ.get("YOLOE_TRT_TEST_CAMERA_SOURCE") or str(USB_CAMERA_DEVICE)
    if USB_CAMERA_DEVICE.exists()
    else "videotest://ball"
)


configure_logging(level=os.environ.get("YOLOE_TRT_LOG_LEVEL", "INFO"), stream=sys.stdout)


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("yoloe-camera")
    group.addoption(
        "--camera-source",
        action="store",
        default=DEFAULT_CAMERA_SOURCE,
        help="Camera source for live GUI/integration tests. "
        "Defaults to /dev/video0 when present, otherwise videotest://ball. "
        "Device paths are wrapped in a USB-camera GStreamer pipeline; RTSP URLs, raw GStreamer pipelines, and "
        "dummy aliases like videotest://ball are also accepted.",
    )
    group.addoption(
        "--camera-labels",
        action="store",
        default=os.environ.get("YOLOE_TRT_TEST_CAMERA_LABELS", "pen"),
        help="Comma-separated runtime prompt labels for live camera tests.",
    )
    group.addoption(
        "--camera-max-frames",
        action="store",
        type=int,
        default=int(os.environ.get("YOLOE_TRT_TEST_CAMERA_MAX_FRAMES", "0")),
        help="Maximum frames for live GUI tests. 0 means run until the Tk window is closed.",
    )
    group.addoption(
        "--camera-window-name",
        action="store",
        default=os.environ.get("YOLOE_TRT_TEST_CAMERA_WINDOW", "YOLOE Camera"),
        help="Window title for the live GUI test.",
    )
    group.addoption(
        "--camera-wait-ms",
        action="store",
        type=int,
        default=int(os.environ.get("YOLOE_TRT_TEST_CAMERA_WAIT_MS", "1")),
        help="Per-frame GUI delay in milliseconds for the live GUI test.",
    )
    group.addoption(
        "--run-gui",
        action="store_true",
        default=os.environ.get("YOLOE_TRT_RUN_GUI", "0") == "1",
        help="Run GUI tests that open a live window.",
    )


@pytest.fixture(scope="session")
def test_images() -> dict[str, Path]:
    missing = [str(path) for path in TEST_IMAGES.values() if not path.is_file()]
    if missing:
        pytest.skip(f"Checked-in image fixtures are missing: {missing}")
    return TEST_IMAGES


@pytest.fixture(scope="session")
def integration_imgsz() -> int:
    return int(os.environ.get("YOLOE_TRT_TEST_IMGSZ", "640"))


@pytest.fixture(scope="session")
def model_checkpoint() -> Path:
    try:
        return resolve_model_checkpoint(TEST_MODEL_SPEC)
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


def _camera_labels_from_config(config: pytest.Config) -> list[str]:
    labels = [label.strip() for label in config.getoption("--camera-labels").split(",")]
    labels = [label for label in labels if label]
    if not labels:
        raise pytest.UsageError("--camera-labels must contain at least one non-empty label")
    return labels


def _camera_max_frames_from_config(config: pytest.Config) -> int | None:
    value = int(config.getoption("--camera-max-frames"))
    return None if value <= 0 else value


@pytest.fixture(scope="session")
def camera_labels(pytestconfig: pytest.Config) -> list[str]:
    return _camera_labels_from_config(pytestconfig)


@pytest.fixture(scope="session")
def camera_source_spec(pytestconfig: pytest.Config) -> str:
    return str(pytestconfig.getoption("--camera-source"))


@pytest.fixture(scope="session")
def camera_max_frames(pytestconfig: pytest.Config) -> int | None:
    return _camera_max_frames_from_config(pytestconfig)


@pytest.fixture(scope="session")
def make_camera_source():
    width = int(os.environ.get("YOLOE_TRT_TEST_CAMERA_WIDTH", "640"))
    height = int(os.environ.get("YOLOE_TRT_TEST_CAMERA_HEIGHT", "480"))
    fps = int(os.environ.get("YOLOE_TRT_TEST_CAMERA_FPS", "30"))
    timeout_s = float(os.environ.get("YOLOE_TRT_TEST_CAMERA_TIMEOUT", "5.0"))

    def _make(source_value: str, *, prefix: str, max_frames: int | None) -> GStreamerSource:
        try:
            return camera_source_from_spec(
                source_value,
                width=width,
                height=height,
                fps=fps,
                max_frames=max_frames,
                timeout_s=timeout_s,
                prefix=prefix,
            )
        except FileNotFoundError as exc:
            pytest.skip(str(exc))

    return _make


@pytest.fixture(scope="session")
def usb_camera_source(pytestconfig: pytest.Config, make_camera_source) -> GStreamerSource:
    try:
        return make_camera_source(
            str(pytestconfig.getoption("--camera-source")),
            prefix="camera",
            max_frames=int(os.environ.get("YOLOE_TRT_TEST_CAMERA_FRAMES", "2")),
        )
    except RuntimeError as exc:
        pytest.skip(str(exc))


@pytest.fixture(scope="session")
def camera_window_name(pytestconfig: pytest.Config) -> str:
    return str(pytestconfig.getoption("--camera-window-name"))


@pytest.fixture(scope="session")
def camera_wait_ms(pytestconfig: pytest.Config) -> int:
    return int(pytestconfig.getoption("--camera-wait-ms"))


@pytest.fixture(scope="session")
def run_gui(pytestconfig: pytest.Config) -> bool:
    return bool(pytestconfig.getoption("--run-gui"))


@pytest.fixture(scope="session")
def gui_available(run_gui: bool) -> None:
    if not run_gui:
        pytest.skip("GUI test not requested; pass --run-gui to open a live window")
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        pytest.skip("No DISPLAY or WAYLAND_DISPLAY is available for GUI rendering")
    try:
        root = tk.Tk()
        root.withdraw()
        root.update_idletasks()
        root.destroy()
    except tk.TclError as exc:
        pytest.skip(f"Tk GUI is not available in this environment: {exc}")


@pytest.fixture(scope="session")
def engine_artifact_dir(
    tmp_path_factory: pytest.TempPathFactory,
    integration_imgsz: int,
    model_checkpoint: Path,
) -> Path:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for TensorRT runtime testing")

    try:
        import tensorrt  # noqa: F401
    except ImportError:
        pytest.skip("TensorRT is not installed")

    prebuilt_dir = os.environ.get("YOLOE_TRT_TEST_ARTIFACT_DIR")
    if prebuilt_dir:
        artifact_dir = Path(prebuilt_dir)
        if not artifact_dir.is_dir():
            raise FileNotFoundError(f"YOLOE_TRT_TEST_ARTIFACT_DIR does not exist: {artifact_dir}")
        return artifact_dir

    artifact_dir = tmp_path_factory.mktemp("yoloe_tensorrt") / "artifacts"
    return export_model(
        model_checkpoint,
        artifact_dir=artifact_dir,
        formats=("onnx", "engine"),
        dynamic=False,
        fp16=True,
        imgsz=integration_imgsz,
    )


@pytest.fixture(scope="session")
def runtime_engine(engine_artifact_dir: Path) -> YOLOEEngine:
    engine = YOLOEEngine.from_engine(engine_artifact_dir)
    assert engine.native_main_runtime is not None
    return engine
