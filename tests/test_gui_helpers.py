from __future__ import annotations

import os
import subprocess
import tkinter as tk
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from yoloe_tensorrt.gui import (
    CUSTOM_GUI_SOURCE_OPTION,
    CameraGuiWindow,
    _display_model_name,
    _overlay_gui_text,
    discover_gui_camera_sources,
    parse_confidence_text,
    parse_label_text,
    resolve_gui_source_selection,
    run_camera_gui,
    split_gui_source_selection,
)


class _Var:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value) -> None:
        self.value = value


class _Widget:
    def __init__(self) -> None:
        self.visible = True
        self.config_calls: list[dict[str, object]] = []
        self.children: list[object] = []
        self.bind_calls: list[tuple[str, object, str | None]] = []

    def grid(self, *args, **kwargs) -> None:
        self.visible = True

    def grid_remove(self) -> None:
        self.visible = False

    def winfo_exists(self) -> bool:
        return True

    def winfo_children(self):
        return list(self.children)

    def configure(self, **kwargs) -> None:
        self.config_calls.append(kwargs)

    def bind(self, sequence: str, callback, add: str | None = None) -> None:
        self.bind_calls.append((sequence, callback, add))


class _CanvasWidget(_Widget):
    def __init__(self) -> None:
        super().__init__()
        self.itemconfigure_calls: list[tuple[object, dict[str, object]]] = []
        self.bbox_value = (0, 0, 100, 200)
        self.height = 180
        self.yview_scroll_calls: list[tuple[int, str]] = []
        self.yview_moveto_calls: list[float] = []

    def bbox(self, _item):
        return self.bbox_value

    def itemconfigure(self, item, **kwargs) -> None:
        self.itemconfigure_calls.append((item, kwargs))

    def winfo_height(self) -> int:
        return self.height

    def yview_scroll(self, number: int, what: str) -> None:
        self.yview_scroll_calls.append((number, what))

    def yview_moveto(self, fraction: float) -> None:
        self.yview_moveto_calls.append(fraction)


class _FakeGuiWindow:
    last_instance: _FakeGuiWindow | None = None

    def __init__(
        self,
        _title: str,
        source_spec: str,
        _source_options,
        _model_name: str,
        labels: list[str],
        confidence: float,
        tracking_enabled: bool,
        tracker_name: str,
    ) -> None:
        _FakeGuiWindow.last_instance = self
        self._closed = False
        self._confidence = float(confidence)
        self._tracking_enabled = bool(tracking_enabled)
        self._tracker_name = tracker_name
        self.status_messages: list[str] = []
        self.applied_messages: list[str | None] = []

    @property
    def is_open(self) -> bool:
        return False

    def pump(self) -> None:
        return

    def take_pending_update(self):
        return None

    def set_status(self, message: str) -> None:
        self.status_messages.append(message)

    def confidence(self) -> float:
        return self._confidence

    def tracking_enabled(self) -> bool:
        return self._tracking_enabled

    def tracker_name(self) -> str:
        return self._tracker_name

    def mark_applied(
        self,
        source: str,
        labels: list[str],
        tracking_enabled: bool,
        tracker_name: str,
        message: str | None = None,
    ) -> None:
        self._tracking_enabled = bool(tracking_enabled)
        self._tracker_name = tracker_name
        self.applied_messages.append(message)

    def show_frame(self, _frame_bgr) -> bool:
        return False

    def close(self) -> None:
        self._closed = True


def _make_window(
    *,
    active_source: str = "/dev/video0",
    selected_source: str = "/dev/video0",
    custom_source: str = "",
    labels: list[str] | None = None,
    tracking_enabled: bool = True,
    tracker_name: str = "bytetrack",
) -> CameraGuiWindow:
    window = object.__new__(CameraGuiWindow)
    window._source_options = ("/dev/video0", "videotest://ball", CUSTOM_GUI_SOURCE_OPTION)
    window._source_selection_var = _Var(selected_source)
    window._custom_source_var = _Var(custom_source)
    window._label_input_var = _Var("")
    window._confidence_var = _Var(0.1)
    window._tracking_enabled_var = _Var(tracking_enabled)
    window._tracker_var = _Var(tracker_name)
    window._status_var = _Var("")
    window._active_source_var = _Var(active_source)
    window._active_model_var = _Var("yoloe-26s-seg.pt")
    window._active_labels_var = _Var(", ".join(labels or ["pen"]))
    window._active_confidence_var = _Var("0.10")
    window._active_tracker_var = _Var(tracker_name)
    window._pending_update = None
    window._labels_frame = None
    window._label_states = [(label, _Var(False)) for label in (labels or ["pen"])]
    window._custom_source_label = _Widget()
    window._custom_source_row = _Widget()
    window._tracker_label_widget = _Widget()
    window._tracker_value_widget = _Widget()
    window._root = _Widget()
    window._video_label = _Widget()
    window._closed = False
    window._photo = None
    return window


class _FakeEngine:
    def __init__(self) -> None:
        self.metadata = SimpleNamespace(default_imgsz=320, max_det=10, model_name="yoloe-26s-seg.pt", model_path="")
        self.device = "cuda:0"
        self.main_fp16 = False
        self.set_classes_calls: list[list[str]] = []

    def clear_prompts(self) -> None:
        return None

    def set_classes(self, labels: list[str]) -> None:
        self.set_classes_calls.append(list(labels))

    def create_tracker(self, *args, **kwargs):
        raise AssertionError("tracking is disabled in this test")


def test_parse_label_text_supports_multiple_labels_and_deduplicates() -> None:
    assert parse_label_text("pen, marker\nbottle, pen") == ["pen", "marker", "bottle"]


def test_parse_label_text_ignores_empty_values() -> None:
    assert parse_label_text(" , \n ,pen,, ") == ["pen"]


def test_parse_confidence_text_clamps_and_falls_back() -> None:
    assert parse_confidence_text("0.75") == 0.75
    assert parse_confidence_text("2.0") == 1.0
    assert parse_confidence_text("-0.5") == 0.0
    assert parse_confidence_text("bad", fallback=0.33) == 0.33


def test_run_camera_gui_handles_expected_initial_source_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("yoloe_tensorrt.gui.CameraGuiWindow", _FakeGuiWindow)

    engine = _FakeEngine()

    displayed = run_camera_gui(
        engine,
        tracking=False,
        make_source=lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad source")),
    )

    window = _FakeGuiWindow.last_instance
    assert displayed == 0
    assert window is not None
    assert window.status_messages == ["Unable to open source '/dev/video0': bad source"]
    assert window._closed is True


def test_run_camera_gui_propagates_unexpected_initial_source_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("yoloe_tensorrt.gui.CameraGuiWindow", _FakeGuiWindow)

    engine = _FakeEngine()

    with pytest.raises(KeyError, match="bad source state"):
        run_camera_gui(
            engine,
            tracking=False,
            make_source=lambda *args, **kwargs: (_ for _ in ()).throw(KeyError("bad source state")),
        )


def test_display_model_name_removes_last_extension() -> None:
    assert _display_model_name("yoloe-26s-seg.pt") == "yoloe-26s-seg"
    assert _display_model_name("yoloe-26s-seg.onnx") == "yoloe-26s-seg"
    assert _display_model_name("yoloe-26s-seg.engine") == "yoloe-26s-seg"


def test_display_model_name_preserves_names_without_extension() -> None:
    assert _display_model_name("yoloe-26s-seg") == "yoloe-26s-seg"


def test_discover_gui_camera_sources_sorts_devices_and_appends_presets(tmp_path: Path) -> None:
    for name in ("video10", "video2", "video0"):
        (tmp_path / name).touch()

    sources = discover_gui_camera_sources(device_dir=tmp_path)

    assert sources == [
        str(tmp_path / "video0"),
        str(tmp_path / "video2"),
        str(tmp_path / "video10"),
        "videotest://ball",
        CUSTOM_GUI_SOURCE_OPTION,
    ]


def test_split_gui_source_selection_uses_custom_for_unknown_source() -> None:
    selection, custom = split_gui_source_selection(
        "rtsp://camera.local/stream",
        ["/dev/video0", "videotest://ball", CUSTOM_GUI_SOURCE_OPTION],
    )

    assert selection == CUSTOM_GUI_SOURCE_OPTION
    assert custom == "rtsp://camera.local/stream"


def test_resolve_gui_source_selection_returns_dropdown_selection_when_not_custom() -> None:
    resolved, selection, custom = resolve_gui_source_selection(
        "/dev/video0",
        "",
        ["/dev/video0", "videotest://ball", CUSTOM_GUI_SOURCE_OPTION],
    )

    assert resolved == "/dev/video0"
    assert selection == "/dev/video0"
    assert custom == ""


def test_resolve_gui_source_selection_collapses_matching_custom_value() -> None:
    resolved, selection, custom = resolve_gui_source_selection(
        CUSTOM_GUI_SOURCE_OPTION,
        "  videotest://ball  ",
        ["/dev/video0", "videotest://ball", CUSTOM_GUI_SOURCE_OPTION],
    )

    assert resolved == "videotest://ball"
    assert selection == "videotest://ball"
    assert custom == ""


def test_resolve_gui_source_selection_rejects_empty_custom_value() -> None:
    with pytest.raises(ValueError, match="Source cannot be empty"):
        resolve_gui_source_selection(
            CUSTOM_GUI_SOURCE_OPTION,
            "   ",
            ["/dev/video0", "videotest://ball", CUSTOM_GUI_SOURCE_OPTION],
        )


def test_source_dropdown_change_queues_live_update() -> None:
    window = _make_window(selected_source="videotest://ball")

    CameraGuiWindow._on_source_selection_change(window)

    assert window._pending_update == ("videotest://ball", ["pen"], True, "bytetrack")
    assert window._status_var.get() == "Switching source to 'videotest://ball'..."


def test_selecting_custom_source_only_shows_custom_input() -> None:
    window = _make_window(selected_source=CUSTOM_GUI_SOURCE_OPTION)

    CameraGuiWindow._on_source_selection_change(window)

    assert window._pending_update is None
    assert window._custom_source_label.visible is True
    assert window._custom_source_row.visible is True
    assert window._status_var.get() == "Enter a custom source and press Apply"


def test_apply_custom_source_collapses_back_to_known_source() -> None:
    window = _make_window(
        selected_source=CUSTOM_GUI_SOURCE_OPTION,
        custom_source="videotest://ball",
    )

    CameraGuiWindow._apply_custom_source(window)

    assert window._source_selection_var.get() == "videotest://ball"
    assert window._custom_source_var.get() == ""
    assert window._pending_update == ("videotest://ball", ["pen"], True, "bytetrack")


def test_add_labels_applies_immediately() -> None:
    window = _make_window()
    window._label_input_var.set("marker")

    CameraGuiWindow._add_labels_from_input(window)

    assert window._label_input_var.get() == ""
    assert window._pending_update == ("/dev/video0", ["pen", "marker"], True, "bytetrack")


def test_remove_selected_labels_rejects_removing_all() -> None:
    window = _make_window(labels=["pen"])
    window._label_states = [("pen", _Var(True))]

    CameraGuiWindow._remove_selected_labels(window)

    assert window._pending_update is None
    assert window._status_var.get() == "At least one label must remain active"


def test_remove_selected_labels_supports_multiple_checked_items() -> None:
    window = _make_window(labels=["pen", "marker", "bottle"])
    window._label_states = [
        ("pen", _Var(True)),
        ("marker", _Var(False)),
        ("bottle", _Var(True)),
    ]

    CameraGuiWindow._remove_selected_labels(window)

    assert window._pending_update == ("/dev/video0", ["marker"], True, "bytetrack")


def test_tracking_toggle_applies_immediately() -> None:
    window = _make_window(tracking_enabled=False)

    CameraGuiWindow._on_tracking_toggle(window)

    assert window._pending_update == ("/dev/video0", ["pen"], False, "bytetrack")


def test_update_tracker_visibility_hides_widgets_when_disabled() -> None:
    window = _make_window()

    CameraGuiWindow._update_tracker_visibility(window, False)

    assert window._tracker_label_widget.visible is False
    assert window._tracker_value_widget.visible is False


def test_labels_frame_configure_updates_scrollregion() -> None:
    window = _make_window()
    window._labels_canvas = _CanvasWidget()
    window._labels_frame = _Widget()
    window._labels_scrollbar = _Widget()

    CameraGuiWindow._on_labels_frame_configure(window)

    assert window._labels_canvas.config_calls == [{"scrollregion": (0, 0, 100, 200)}]


def test_labels_canvas_configure_updates_inner_frame_width() -> None:
    window = _make_window()
    window._labels_canvas = _CanvasWidget()
    window._labels_window_id = 17
    window._labels_scrollbar = _Widget()

    CameraGuiWindow._on_labels_canvas_configure(window, SimpleNamespace(width=320))

    assert window._labels_canvas.itemconfigure_calls == [(17, {"width": 320})]


def test_labels_scrollbar_stays_hidden_without_overflow() -> None:
    window = _make_window()
    window._labels_canvas = _CanvasWidget()
    window._labels_canvas.bbox_value = (0, 0, 100, 80)
    window._labels_canvas.height = 180
    window._labels_scrollbar = _Widget()

    CameraGuiWindow._update_labels_scrollbar_visibility(window)

    assert window._labels_scrollbar.visible is False
    assert window._labels_canvas.yview_moveto_calls == [0.0]


def test_labels_scrollbar_shows_when_labels_overflow() -> None:
    window = _make_window()
    window._labels_canvas = _CanvasWidget()
    window._labels_canvas.bbox_value = (0, 0, 100, 320)
    window._labels_canvas.height = 180
    window._labels_scrollbar = _Widget()
    window._labels_scrollbar.visible = False

    CameraGuiWindow._update_labels_scrollbar_visibility(window)

    assert window._labels_scrollbar.visible is True


def test_labels_mousewheel_scrolls_when_overflow() -> None:
    window = _make_window()
    window._labels_canvas = _CanvasWidget()
    window._labels_canvas.bbox_value = (0, 0, 100, 320)
    window._labels_canvas.height = 180

    result = CameraGuiWindow._on_labels_mousewheel(window, SimpleNamespace(delta=-120, num=0))

    assert result == "break"
    assert window._labels_canvas.yview_scroll_calls == [(1, "units")]


def test_labels_mousewheel_ignores_non_overflowing_content() -> None:
    window = _make_window()
    window._labels_canvas = _CanvasWidget()
    window._labels_canvas.bbox_value = (0, 0, 100, 80)
    window._labels_canvas.height = 180

    result = CameraGuiWindow._on_labels_mousewheel(window, SimpleNamespace(delta=-120, num=0))

    assert result is None
    assert window._labels_canvas.yview_scroll_calls == []


def test_mark_applied_preserves_staged_custom_source_for_non_source_updates() -> None:
    window = _make_window(
        selected_source=CUSTOM_GUI_SOURCE_OPTION,
        custom_source="rtsp://camera.local/stream",
    )
    window._active_labels_var = _Var("pen")
    window._active_confidence_var = _Var("0.10")
    window._active_tracker_var = _Var("bytetrack")

    CameraGuiWindow.mark_applied(window, "/dev/video0", ["pen"], True, "bytetrack")

    assert window._source_selection_var.get() == CUSTOM_GUI_SOURCE_OPTION
    assert window._custom_source_var.get() == "rtsp://camera.local/stream"


def test_mark_applied_uses_compact_status_message_by_default() -> None:
    window = _make_window()

    CameraGuiWindow.mark_applied(window, "/dev/video0", ["pen"], True, "bytetrack")

    assert window._status_var.get() == "Settings applied"


def test_overlay_gui_text_only_renders_source_and_inference_fps(monkeypatch: pytest.MonkeyPatch) -> None:
    drawn: list[str] = []

    def _fake_put_text(image, text, *args, **kwargs):
        drawn.append(text)
        return image

    monkeypatch.setattr("yoloe_tensorrt.gui.cv2.putText", _fake_put_text)

    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    result = SimpleNamespace(speed={"preprocess": 1.0, "inference": 25.0, "postprocess": 3.0})

    _overlay_gui_text(frame, "/dev/video0", result)

    lines = [text for index, text in enumerate(drawn) if index % 2 == 0]
    assert lines == ["source: /dev/video0", "inf fps: 40.0"]


def test_overlay_gui_text_omits_fps_when_inference_timing_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    drawn: list[str] = []

    def _fake_put_text(image, text, *args, **kwargs):
        drawn.append(text)
        return image

    monkeypatch.setattr("yoloe_tensorrt.gui.cv2.putText", _fake_put_text)

    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    result = SimpleNamespace(speed={"preprocess": 1.0, "postprocess": 3.0})

    _overlay_gui_text(frame, "/dev/video0", result)

    lines = [text for index, text in enumerate(drawn) if index % 2 == 0]
    assert lines == ["source: /dev/video0"]


def test_show_frame_handles_tclerror_during_shutdown(monkeypatch: pytest.MonkeyPatch) -> None:
    window = _make_window()

    def _raise_photo_image(*args, **kwargs):
        raise tk.TclError('can\'t invoke "image" command: application has been destroyed')

    monkeypatch.setattr("yoloe_tensorrt.gui.ImageTk.PhotoImage", _raise_photo_image)

    shown = CameraGuiWindow.show_frame(window, np.zeros((8, 8, 3), dtype=np.uint8))

    assert shown is False
    assert window._closed is True


def test_python_launcher_displays_help(
    run_python_help: Callable[[list[str]], subprocess.CompletedProcess[str]],
) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    launcher = repo_root / "scripts" / "launch_camera_gui.py"
    result = run_python_help([str(launcher)])
    assert result.returncode == 0
    assert "Launch the YOLOE TensorRT live camera GUI." in result.stdout
    assert "--track" in result.stdout
    assert "--tracker" in result.stdout


def test_shell_launcher_prefers_console_script_when_available(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    launcher = repo_root / "scripts" / "launch_camera_gui.sh"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_launcher = bin_dir / "yoloe-camera-gui"
    fake_launcher.write_text("#!/usr/bin/env bash\nprintf 'fake-launcher %s\\n' \"$*\"\n", encoding="utf-8")
    fake_launcher.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
    result = subprocess.run(
        ["bash", str(launcher), "--help"],
        check=False,
        capture_output=True,
        text=True,
        cwd=repo_root,
        env=env,
    )
    assert result.returncode == 0
    assert "fake-launcher --help" in result.stdout
