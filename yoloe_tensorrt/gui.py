from __future__ import annotations

import argparse
import os
import re
import time
import tkinter as tk
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, Sequence

import cv2
from PIL import Image, ImageTk

from ._shapes import normalize_imgsz
from .assets import default_example_model_spec, resolve_model_checkpoint
from .gstreamer import camera_source_from_spec
from .logging_utils import configure_logging, get_logger
from .tracking import AVAILABLE_TRACKERS, DEFAULT_TRACKER, normalize_tracker_name

if TYPE_CHECKING:
    from .engine import YOLOEEngine


LOGGER = get_logger(__name__)

DEFAULT_GUI_SOURCE = os.environ.get("YOLOE_TRT_GUI_SOURCE", "/dev/video0")
DEFAULT_GUI_LABELS = tuple(
    label for label in re.split(r"[\n,]", os.environ.get("YOLOE_TRT_GUI_LABELS", "pen")) if label.strip()
) or ("pen",)
DEFAULT_GUI_TITLE = os.environ.get("YOLOE_TRT_GUI_WINDOW", "YOLOE Camera")
DEFAULT_GUI_WAIT_MS = int(os.environ.get("YOLOE_TRT_GUI_WAIT_MS", "1"))
DEFAULT_GUI_IMGSZ = int(os.environ.get("YOLOE_TRT_GUI_IMGSZ", "320"))
DEFAULT_GUI_CAMERA_WIDTH = int(os.environ.get("YOLOE_TRT_GUI_CAMERA_WIDTH", "640"))
DEFAULT_GUI_CAMERA_HEIGHT = int(os.environ.get("YOLOE_TRT_GUI_CAMERA_HEIGHT", "480"))
DEFAULT_GUI_CAMERA_FPS = int(os.environ.get("YOLOE_TRT_GUI_CAMERA_FPS", "30"))
DEFAULT_GUI_CAMERA_TIMEOUT_S = float(os.environ.get("YOLOE_TRT_GUI_CAMERA_TIMEOUT", "5.0"))
DEFAULT_GUI_PREFIX = "camera"
DEFAULT_GUI_CONF = 0.1
DEFAULT_GUI_LABELS_VIEWPORT_HEIGHT = int(os.environ.get("YOLOE_TRT_GUI_LABELS_VIEWPORT_HEIGHT", "180"))
DEFAULT_GUI_WORKSPACE_BYTES = int(os.environ.get("YOLOE_TRT_GUI_WORKSPACE_BYTES", str(512 << 20)))
DEFAULT_GUI_TRACKING = os.environ.get("YOLOE_TRT_GUI_TRACKING", "1").lower() not in {"0", "false", "no"}
DEFAULT_GUI_TRACKER = normalize_tracker_name(os.environ.get("YOLOE_TRT_GUI_TRACKER", DEFAULT_TRACKER))
DEFAULT_GUI_SOURCE_PRESETS = ("videotest://ball",)
CUSTOM_GUI_SOURCE_OPTION = "Custom..."


def parse_label_text(text: str) -> list[str]:
    labels: list[str] = []
    for raw_label in re.split(r"[\n,]", text):
        label = raw_label.strip()
        if label and label not in labels:
            labels.append(label)
    return labels


def parse_confidence_text(text: str, fallback: float = DEFAULT_GUI_CONF) -> float:
    try:
        value = float(str(text).strip())
    except (TypeError, ValueError):
        return float(fallback)
    return max(0.0, min(1.0, value))


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_gui_model_spec() -> str:
    return os.environ.get("YOLOE_TRT_GUI_MODEL", default_example_model_spec())


def discover_gui_camera_sources(
    *,
    device_dir: str | Path = "/dev",
    presets: Sequence[str] = DEFAULT_GUI_SOURCE_PRESETS,
) -> list[str]:
    root = Path(device_dir)

    def _sort_key(path: Path) -> tuple[str, int, str]:
        match = re.search(r"(\d+)$", path.name)
        if match is None:
            return (path.name, -1, path.name)
        prefix = path.name[: match.start()]
        return (prefix, int(match.group(1)), path.name)

    sources = [str(path) for path in sorted(root.glob("video*"), key=_sort_key) if path.exists()]
    for preset in presets:
        value = str(preset).strip()
        if value and value not in sources:
            sources.append(value)
    sources.append(CUSTOM_GUI_SOURCE_OPTION)
    return sources


def split_gui_source_selection(source_spec: str, available_sources: Sequence[str]) -> tuple[str, str]:
    source = str(source_spec).strip()
    known_sources = [option for option in available_sources if option != CUSTOM_GUI_SOURCE_OPTION]
    if source in known_sources:
        return source, ""
    return CUSTOM_GUI_SOURCE_OPTION, source


def resolve_gui_source_selection(
    selected_source: str,
    custom_source: str,
    available_sources: Sequence[str],
) -> tuple[str, str, str]:
    known_sources = [option for option in available_sources if option != CUSTOM_GUI_SOURCE_OPTION]
    selected = str(selected_source).strip()
    if selected and selected != CUSTOM_GUI_SOURCE_OPTION:
        if selected not in known_sources:
            raise ValueError(f"Unsupported source selection: {selected}")
        return selected, selected, ""

    custom = str(custom_source).strip()
    if not custom:
        raise ValueError("Source cannot be empty")
    if custom in known_sources:
        return custom, custom, ""
    return custom, CUSTOM_GUI_SOURCE_OPTION, custom


def default_gui_artifact_dir(imgsz: int = DEFAULT_GUI_IMGSZ) -> Path:
    return _repo_root() / "outputs" / "artifacts" / f"camera_gui_{int(imgsz)}"


def _display_model_name(model_name: str | Path | None, model_path: str | Path | None = None) -> str:
    candidate = str(model_name or model_path or "").strip()
    if not candidate:
        return ""
    name = Path(candidate).name
    stem = Path(name).stem
    return stem or name


def _bundle_ready(artifact_dir: Path) -> bool:
    return all(
        (artifact_dir / filename).is_file() for filename in ("metadata.json", "main.engine", "prompt_projector.ts")
    )


def load_or_build_gui_engine(
    *,
    artifact_dir: str | Path | None = None,
    model_path: str | Path | None = None,
    imgsz: int = DEFAULT_GUI_IMGSZ,
    device: str = "cuda:0",
) -> YOLOEEngine:
    from .engine import YOLOEEngine
    from .export import export_model

    resolved_artifact_dir = Path(artifact_dir) if artifact_dir is not None else default_gui_artifact_dir(imgsz)
    model_spec = model_path if model_path is not None else default_gui_model_spec()

    if not _bundle_ready(resolved_artifact_dir):
        resolved_model_path = resolve_model_checkpoint(model_spec)
        partial_files = [path.name for path in resolved_artifact_dir.glob("*") if path.is_file()]
        if partial_files:
            LOGGER.info(
                "Resuming partial GUI artifact bundle at '%s' with existing files=%s",
                resolved_artifact_dir,
                sorted(partial_files),
            )
        else:
            LOGGER.info(
                "Building default GUI main-engine bundle at '%s' from '%s' (imgsz=%d)",
                resolved_artifact_dir,
                resolved_model_path,
                int(imgsz),
            )
        if "YOLOE_TRT_BUILDER_OPT_LEVEL" not in os.environ:
            os.environ["YOLOE_TRT_BUILDER_OPT_LEVEL"] = "0"
            LOGGER.info("Defaulting YOLOE_TRT_BUILDER_OPT_LEVEL=0 for GUI bundle build")
        if "YOLOE_TRT_AVG_TIMING_ITERATIONS" not in os.environ:
            os.environ["YOLOE_TRT_AVG_TIMING_ITERATIONS"] = "1"
            LOGGER.info("Defaulting YOLOE_TRT_AVG_TIMING_ITERATIONS=1 for GUI bundle build")
        LOGGER.info("Using GUI TensorRT workspace limit of %d bytes", DEFAULT_GUI_WORKSPACE_BYTES)
        export_model(
            resolved_model_path,
            artifact_dir=resolved_artifact_dir,
            formats=("onnx", "engine"),
            dynamic=False,
            build_visual_engine=False,
            fp16=True,
            imgsz=int(imgsz),
            overwrite=False,
            workspace_bytes=DEFAULT_GUI_WORKSPACE_BYTES,
        )
    else:
        LOGGER.info("Reusing existing GUI artifact bundle at '%s'", resolved_artifact_dir)

    return YOLOEEngine.from_engine(resolved_artifact_dir, device=device)


def _compact_overlay_source(source_spec: str, max_chars: int = 44) -> str:
    source = str(source_spec).strip()
    if len(source) <= max_chars:
        return source
    suffix_chars = min(16, max_chars // 3)
    prefix_chars = max_chars - suffix_chars - 3
    return f"{source[:prefix_chars]}...{source[-suffix_chars:]}"


def _overlay_gui_text(frame, source_spec: str, result) -> None:
    lines = [f"source: {_compact_overlay_source(source_spec)}"]
    if getattr(result, "speed", None):
        inference_ms = float(result.speed.get("inference", 0.0))
        if inference_ms > 0.0:
            lines.append(f"inf fps: {1000.0 / inference_ms:.1f}")

    y = 30
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        y += 24


class CameraGuiWindow:
    def __init__(
        self,
        title: str,
        source_spec: str,
        source_options: Sequence[str],
        model_name: str,
        labels: list[str],
        confidence: float,
        tracking_enabled: bool,
        tracker_name: str,
    ) -> None:
        self._root = tk.Tk()
        self._root.title(title)
        self._root.resizable(True, True)
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._source_options = tuple(source_options)
        selected_source, custom_source = split_gui_source_selection(source_spec, self._source_options)
        self._source_selection_var = tk.StringVar(value=selected_source)
        self._custom_source_var = tk.StringVar(value=custom_source)
        self._label_input_var = tk.StringVar(value="")
        self._confidence_var = tk.DoubleVar(value=float(confidence))
        self._confidence_text_var = tk.StringVar(value=f"{float(confidence):.2f}")
        self._tracking_enabled_var = tk.BooleanVar(value=bool(tracking_enabled))
        self._tracker_var = tk.StringVar(value=normalize_tracker_name(tracker_name))
        self._status_var = tk.StringVar(value="Source, labels, confidence, and tracking update live")
        self._active_source_var = tk.StringVar(value=source_spec)
        self._active_model_var = tk.StringVar(value=model_name)
        self._active_labels_var = tk.StringVar(value=", ".join(labels))
        self._active_confidence_var = tk.StringVar(value=f"{float(confidence):.2f}")
        self._active_tracker_var = tk.StringVar(value=normalize_tracker_name(tracker_name))
        self._pending_update: tuple[str, list[str], bool, str] | None = None
        self._closed = False
        self._photo: ImageTk.PhotoImage | None = None
        self._label_states: list[tuple[str, object]] = []

        root_frame = tk.Frame(self._root, padx=12, pady=12)
        root_frame.pack(fill="both", expand=True)
        root_frame.columnconfigure(0, weight=0)
        root_frame.columnconfigure(1, weight=1)
        root_frame.rowconfigure(0, weight=1)

        controls = tk.Frame(root_frame)
        controls.grid(row=0, column=0, sticky="nsw", padx=(0, 12))
        controls.columnconfigure(0, weight=1)

        tk.Label(controls, text="Source").grid(row=0, column=0, sticky="w")
        source_menu = tk.OptionMenu(
            controls,
            self._source_selection_var,
            *self._source_options,
            command=lambda _value: self._on_source_selection_change(),
        )
        source_menu.grid(row=1, column=0, sticky="ew", pady=(0, 6))

        self._custom_source_label = tk.Label(controls, text="Custom Source")
        self._custom_source_row = tk.Frame(controls)
        self._custom_source_row.columnconfigure(0, weight=1)
        self._custom_source_entry = tk.Entry(self._custom_source_row, textvariable=self._custom_source_var, width=40)
        self._custom_source_entry.grid(row=0, column=0, sticky="ew")
        self._custom_source_entry.bind("<Return>", lambda _event: self._apply_custom_source())
        self._custom_source_apply_button = tk.Button(
            self._custom_source_row,
            text="Apply",
            command=self._apply_custom_source,
        )
        self._custom_source_apply_button.grid(row=0, column=1, sticky="w", padx=(8, 0))
        self._custom_source_label.grid(row=2, column=0, sticky="w")
        self._custom_source_row.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        self._update_custom_source_visibility()

        tk.Label(controls, text="Current Labels").grid(row=4, column=0, sticky="w")
        self._labels_container = tk.Frame(controls, borderwidth=1, relief="sunken")
        self._labels_container.grid(row=5, column=0, sticky="ew")
        self._labels_container.rowconfigure(0, weight=1)
        self._labels_container.columnconfigure(0, weight=1)
        self._labels_canvas = tk.Canvas(
            self._labels_container,
            borderwidth=0,
            highlightthickness=0,
            height=DEFAULT_GUI_LABELS_VIEWPORT_HEIGHT,
        )
        self._labels_canvas.grid(row=0, column=0, sticky="nsew")
        self._labels_scrollbar = tk.Scrollbar(
            self._labels_container,
            orient="vertical",
            command=self._labels_canvas.yview,
        )
        self._labels_scrollbar.grid(row=0, column=1, sticky="ns")
        self._labels_scrollbar.grid_remove()
        self._labels_canvas.configure(yscrollcommand=self._labels_scrollbar.set)
        self._labels_frame = tk.Frame(self._labels_canvas)
        self._labels_frame.columnconfigure(0, weight=1)
        self._labels_window_id = self._labels_canvas.create_window((0, 0), window=self._labels_frame, anchor="nw")
        self._labels_frame.bind("<Configure>", self._on_labels_frame_configure)
        self._labels_canvas.bind("<Configure>", self._on_labels_canvas_configure)
        self._bind_labels_mousewheel(self._labels_container)
        self._bind_labels_mousewheel(self._labels_canvas)
        self._bind_labels_mousewheel(self._labels_frame)
        self._bind_labels_mousewheel(self._labels_scrollbar)
        self._set_label_list(labels)

        label_buttons = tk.Frame(controls)
        label_buttons.grid(row=6, column=0, sticky="ew", pady=(8, 10))
        label_buttons.columnconfigure(0, weight=1)

        tk.Button(label_buttons, text="Remove Selected", command=self._remove_selected_labels).grid(
            row=0, column=0, sticky="ew"
        )

        tk.Label(controls, text="Add Label(s)").grid(row=7, column=0, sticky="w")
        label_entry = tk.Entry(controls, textvariable=self._label_input_var, width=48)
        label_entry.grid(row=8, column=0, sticky="ew", pady=(0, 8))
        label_entry.bind("<Return>", lambda _event: self._add_labels_from_input())

        tk.Button(controls, text="Add", command=self._add_labels_from_input).grid(row=9, column=0, sticky="ew")

        tk.Label(controls, text="Confidence Threshold").grid(row=10, column=0, sticky="w")
        confidence_scale = tk.Scale(
            controls,
            variable=self._confidence_var,
            from_=0.0,
            to=1.0,
            resolution=0.01,
            orient="horizontal",
            length=260,
            showvalue=False,
            command=self._on_confidence_change,
        )
        confidence_scale.grid(row=11, column=0, sticky="ew", pady=(0, 2))

        confidence_row = tk.Frame(controls)
        confidence_row.grid(row=12, column=0, sticky="ew", pady=(0, 8))
        confidence_row.columnconfigure(0, weight=1)
        confidence_row.columnconfigure(1, weight=0)

        confidence_entry = tk.Entry(confidence_row, textvariable=self._confidence_text_var, width=8)
        confidence_entry.grid(row=0, column=0, sticky="w")
        confidence_entry.bind("<Return>", lambda _event: self._apply_confidence_text())
        tk.Button(confidence_row, text="Apply Confidence", command=self._apply_confidence_text).grid(
            row=0, column=1, sticky="w", padx=(8, 0)
        )

        tracking_row = tk.Frame(controls)
        tracking_row.grid(row=13, column=0, sticky="ew", pady=(8, 4))
        tracking_row.columnconfigure(0, weight=1)
        tracking_row.columnconfigure(1, weight=1)
        tk.Checkbutton(
            tracking_row,
            text="Enable Tracking",
            variable=self._tracking_enabled_var,
            command=self._on_tracking_toggle,
        ).grid(row=0, column=0, sticky="w")
        tk.OptionMenu(
            tracking_row,
            self._tracker_var,
            *AVAILABLE_TRACKERS,
            command=lambda _value: self._on_tracker_change(),
        ).grid(row=0, column=1, sticky="ew")

        tk.Label(
            controls,
            text="Use commas or new lines to add multiple labels at once.",
            anchor="w",
            justify="left",
        ).grid(row=14, column=0, sticky="ew", pady=(8, 0))
        tk.Label(controls, textvariable=self._status_var, anchor="w", justify="left", wraplength=320).grid(
            row=15, column=0, sticky="ew", pady=(8, 0)
        )

        display = tk.Frame(root_frame)
        display.grid(row=0, column=1, sticky="nsew")
        display.columnconfigure(0, weight=1)
        display.rowconfigure(0, weight=1)

        self._video_label = tk.Label(display, text="Waiting for frames...", anchor="center")
        self._video_label.grid(row=0, column=0, sticky="nsew")

        applied = tk.LabelFrame(display, text="Applied Settings", padx=10, pady=8)
        applied.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        applied.columnconfigure(1, weight=1)
        applied.columnconfigure(3, weight=1)

        tk.Label(applied, text="Source").grid(row=0, column=0, sticky="nw", padx=(0, 8))
        tk.Label(
            applied,
            textvariable=self._active_source_var,
            anchor="w",
            justify="left",
            wraplength=360,
        ).grid(row=0, column=1, sticky="ew")
        self._tracker_label_widget = tk.Label(applied, text="Tracker")
        self._tracker_label_widget.grid(row=0, column=2, sticky="nw", padx=(16, 8))
        self._tracker_value_widget = tk.Label(
            applied, textvariable=self._active_tracker_var, anchor="w", justify="left"
        )
        self._tracker_value_widget.grid(row=0, column=3, sticky="ew")

        tk.Label(applied, text="Model").grid(row=1, column=0, sticky="nw", padx=(0, 8), pady=(4, 0))
        tk.Label(
            applied,
            textvariable=self._active_model_var,
            anchor="w",
            justify="left",
            wraplength=360,
        ).grid(row=1, column=1, sticky="ew", pady=(4, 0))
        tk.Label(applied, text="Confidence").grid(row=1, column=2, sticky="nw", padx=(16, 8), pady=(4, 0))
        tk.Label(applied, textvariable=self._active_confidence_var, anchor="w", justify="left").grid(
            row=1, column=3, sticky="ew", pady=(4, 0)
        )
        tk.Label(applied, text="Labels").grid(row=2, column=0, sticky="nw", padx=(0, 8), pady=(4, 0))
        tk.Label(
            applied,
            textvariable=self._active_labels_var,
            anchor="w",
            justify="left",
            wraplength=720,
        ).grid(row=2, column=1, columnspan=3, sticky="ew", pady=(4, 0))
        self._update_tracker_visibility(bool(tracking_enabled))

    @property
    def is_open(self) -> bool:
        return not self._closed

    def _on_close(self) -> None:
        self._closed = True
        try:
            self._root.destroy()
        except tk.TclError:
            pass

    def _make_label_state(self, selected: bool = False):
        labels_frame = getattr(self, "_labels_frame", None)
        if labels_frame is not None and hasattr(labels_frame, "winfo_exists"):
            return tk.BooleanVar(master=self._root, value=bool(selected))

        class _BoolState:
            def __init__(self, value: bool) -> None:
                self._value = bool(value)

            def get(self) -> bool:
                return self._value

            def set(self, value: bool) -> None:
                self._value = bool(value)

        return _BoolState(selected)

    def _on_labels_frame_configure(self, _event=None) -> None:
        labels_canvas = getattr(self, "_labels_canvas", None)
        labels_frame = getattr(self, "_labels_frame", None)
        if labels_canvas is None or labels_frame is None:
            return
        try:
            labels_canvas.configure(scrollregion=labels_canvas.bbox("all"))
        except tk.TclError:
            return
        self._update_labels_scrollbar_visibility()

    def _on_labels_canvas_configure(self, event) -> None:
        labels_canvas = getattr(self, "_labels_canvas", None)
        labels_window_id = getattr(self, "_labels_window_id", None)
        if labels_canvas is None or labels_window_id is None:
            return
        try:
            labels_canvas.itemconfigure(labels_window_id, width=event.width)
        except tk.TclError:
            return
        self._update_labels_scrollbar_visibility()

    def _update_labels_scrollbar_visibility(self) -> None:
        labels_canvas = getattr(self, "_labels_canvas", None)
        labels_scrollbar = getattr(self, "_labels_scrollbar", None)
        if labels_canvas is None or labels_scrollbar is None:
            return
        try:
            bbox = labels_canvas.bbox("all")
            if bbox is None:
                labels_scrollbar.grid_remove()
                return
            viewport_height = int(labels_canvas.winfo_height())
        except tk.TclError:
            return
        content_height = int(bbox[3] - bbox[1])
        if content_height > max(viewport_height, 1):
            labels_scrollbar.grid()
        else:
            try:
                labels_canvas.yview_moveto(0.0)
            except tk.TclError:
                return
            labels_scrollbar.grid_remove()

    def _labels_can_scroll(self) -> bool:
        labels_canvas = getattr(self, "_labels_canvas", None)
        if labels_canvas is None:
            return False
        try:
            bbox = labels_canvas.bbox("all")
            if bbox is None:
                return False
            viewport_height = int(labels_canvas.winfo_height())
        except tk.TclError:
            return False
        return int(bbox[3] - bbox[1]) > max(viewport_height, 1)

    def _bind_labels_mousewheel(self, widget) -> None:
        if widget is None or not hasattr(widget, "bind"):
            return
        widget.bind("<MouseWheel>", self._on_labels_mousewheel, add="+")
        widget.bind("<Button-4>", self._on_labels_mousewheel, add="+")
        widget.bind("<Button-5>", self._on_labels_mousewheel, add="+")

    def _on_labels_mousewheel(self, event) -> str | None:
        labels_canvas = getattr(self, "_labels_canvas", None)
        if labels_canvas is None or not self._labels_can_scroll():
            return None

        step = 0
        delta = int(getattr(event, "delta", 0) or 0)
        if delta != 0:
            step = -1 if delta > 0 else 1
        else:
            event_num = int(getattr(event, "num", 0) or 0)
            if event_num == 4:
                step = -1
            elif event_num == 5:
                step = 1
        if step == 0:
            return None

        try:
            labels_canvas.yview_scroll(step, "units")
        except tk.TclError:
            return None
        return "break"

    def _set_label_list(self, labels: Iterable[str]) -> None:
        labels_list = [str(label) for label in labels]
        labels_frame = getattr(self, "_labels_frame", None)
        existing_states = {label: bool(state.get()) for label, state in getattr(self, "_label_states", [])}
        self._label_states = []

        if labels_frame is not None and hasattr(labels_frame, "winfo_children"):
            for child in labels_frame.winfo_children():
                try:
                    child.destroy()
                except tk.TclError:
                    pass

            for row_index, label in enumerate(labels_list):
                state = self._make_label_state(existing_states.get(label, False))
                checkbox = tk.Checkbutton(labels_frame, text=label, variable=state, anchor="w", justify="left")
                checkbox.grid(row=row_index, column=0, sticky="ew")
                self._bind_labels_mousewheel(checkbox)
                self._label_states.append((label, state))
            self._on_labels_frame_configure()
            return

        for label in labels_list:
            state = self._make_label_state(existing_states.get(label, False))
            self._label_states.append((label, state))

    def _update_tracker_visibility(self, tracking_enabled: bool) -> None:
        if tracking_enabled:
            self._tracker_label_widget.grid()
            self._tracker_value_widget.grid()
        else:
            self._tracker_label_widget.grid_remove()
            self._tracker_value_widget.grid_remove()

    def _update_custom_source_visibility(self) -> None:
        if self._source_selection_var.get() == CUSTOM_GUI_SOURCE_OPTION:
            self._custom_source_label.grid()
            self._custom_source_row.grid()
        else:
            self._custom_source_label.grid_remove()
            self._custom_source_row.grid_remove()

    def _queue_current_update(self, *, source_override: str | None = None, status_message: str | None = None) -> bool:
        labels = self._current_labels()
        if not labels:
            self._status_var.set("Add at least one label before applying")
            return False
        source = str(source_override if source_override is not None else self._active_source_var.get()).strip()
        if not source:
            self._status_var.set("Source cannot be empty")
            return False
        self._pending_update = (
            source,
            labels,
            bool(self._tracking_enabled_var.get()),
            normalize_tracker_name(self._tracker_var.get()),
        )
        if status_message:
            self._status_var.set(status_message)
        return True

    def _on_source_selection_change(self) -> None:
        self._update_custom_source_visibility()
        selected_source = self._source_selection_var.get()
        if selected_source == CUSTOM_GUI_SOURCE_OPTION:
            self._status_var.set("Enter a custom source and press Apply")
            return
        self._queue_current_update(
            source_override=selected_source,
            status_message=f"Switching source to '{selected_source}'...",
        )

    def _current_labels(self) -> list[str]:
        return [label for label, _state in self._label_states]

    def _selected_labels(self) -> list[str]:
        return [label for label, state in self._label_states if bool(state.get())]

    def _add_labels_from_input(self) -> None:
        new_labels = parse_label_text(self._label_input_var.get())
        if not new_labels:
            self._status_var.set("Enter at least one non-empty label to add")
            return

        labels = self._current_labels()
        added = 0
        for label in new_labels:
            if label not in labels:
                labels.append(label)
                added += 1
        self._set_label_list(labels)
        self._label_input_var.set("")
        if added == 0:
            self._status_var.set("Those labels are already in the list")
        else:
            self._queue_current_update(status_message=f"Applying {added} added label(s)...")

    def _remove_selected_labels(self) -> None:
        selected = self._selected_labels()
        if not selected:
            self._status_var.set("Select one or more labels to remove")
            return

        labels = self._current_labels()
        if len(selected) >= len(labels):
            self._status_var.set("At least one label must remain active")
            return
        selected_labels = set(selected)
        self._set_label_list([label for label in labels if label not in selected_labels])
        self._queue_current_update(status_message="Applying removed label(s)...")

    def _apply_custom_source(self) -> None:
        try:
            source, selected_source, custom_source = resolve_gui_source_selection(
                self._source_selection_var.get(),
                self._custom_source_var.get(),
                self._source_options,
            )
        except ValueError as exc:
            self._status_var.set(str(exc))
            return
        self._source_selection_var.set(selected_source)
        self._custom_source_var.set(custom_source)
        self._update_custom_source_visibility()
        self._queue_current_update(
            source_override=source,
            status_message=f"Switching source to '{source}'...",
        )

    def _on_confidence_change(self, value: str) -> None:
        confidence = parse_confidence_text(value, fallback=float(DEFAULT_GUI_CONF))
        self._confidence_text_var.set(f"{confidence:.2f}")
        self._active_confidence_var.set(f"{confidence:.2f}")

    def _apply_confidence_text(self) -> None:
        confidence = parse_confidence_text(self._confidence_text_var.get(), fallback=float(DEFAULT_GUI_CONF))
        self._confidence_var.set(confidence)
        self._confidence_text_var.set(f"{confidence:.2f}")
        self._active_confidence_var.set(f"{confidence:.2f}")
        self._status_var.set(f"Confidence threshold set to {confidence:.2f}; it applies on the next frame")

    def _on_tracking_toggle(self) -> None:
        self._queue_current_update(status_message="Applying tracking setting...")

    def _on_tracker_change(self) -> None:
        self._queue_current_update(status_message="Applying tracker backend...")

    def pump(self) -> None:
        if self._closed:
            return
        try:
            self._root.update_idletasks()
            self._root.update()
        except tk.TclError:
            self._closed = True

    def take_pending_update(self) -> tuple[str, list[str], bool, str] | None:
        update = self._pending_update
        self._pending_update = None
        return update

    def set_status(self, message: str) -> None:
        self._status_var.set(message)

    def confidence(self) -> float:
        return float(self._confidence_var.get())

    def tracking_enabled(self) -> bool:
        return bool(self._tracking_enabled_var.get())

    def tracker_name(self) -> str:
        return normalize_tracker_name(self._tracker_var.get())

    def mark_applied(
        self,
        source: str,
        labels: list[str],
        tracking_enabled: bool,
        tracker_name: str,
        message: str | None = None,
    ) -> None:
        preserve_custom_staging = (
            self._source_selection_var.get() == CUSTOM_GUI_SOURCE_OPTION
            and bool(str(self._custom_source_var.get()).strip())
            and str(source).strip() == str(self._active_source_var.get()).strip()
        )
        if not preserve_custom_staging:
            selected_source, custom_source = split_gui_source_selection(source, self._source_options)
            self._source_selection_var.set(selected_source)
            self._custom_source_var.set(custom_source)
            self._update_custom_source_visibility()
        self._set_label_list(labels)
        self._tracking_enabled_var.set(bool(tracking_enabled))
        self._tracker_var.set(normalize_tracker_name(tracker_name))
        self._active_source_var.set(source)
        self._active_labels_var.set(", ".join(labels))
        self._active_confidence_var.set(f"{self.confidence():.2f}")
        self._active_tracker_var.set(normalize_tracker_name(tracker_name))
        self._update_tracker_visibility(bool(tracking_enabled))
        self._status_var.set(message or "Settings applied")

    def show_frame(self, frame_bgr) -> bool:
        if self._closed:
            return False
        try:
            if not bool(self._root.winfo_exists()) or not bool(self._video_label.winfo_exists()):
                self._closed = True
                return False
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            self._photo = ImageTk.PhotoImage(image=image, master=self._root)
            self._video_label.configure(image=self._photo, text="")
            return True
        except tk.TclError:
            self._closed = True
            self._photo = None
            return False

    def close(self) -> None:
        self._on_close()


def run_camera_gui(
    engine: YOLOEEngine,
    *,
    source_spec: str = DEFAULT_GUI_SOURCE,
    labels: Iterable[str] = DEFAULT_GUI_LABELS,
    window_title: str = DEFAULT_GUI_TITLE,
    wait_ms: int = DEFAULT_GUI_WAIT_MS,
    max_frames: int | None = None,
    source_prefix: str = DEFAULT_GUI_PREFIX,
    width: int = DEFAULT_GUI_CAMERA_WIDTH,
    height: int = DEFAULT_GUI_CAMERA_HEIGHT,
    fps: int = DEFAULT_GUI_CAMERA_FPS,
    timeout_s: float = DEFAULT_GUI_CAMERA_TIMEOUT_S,
    imgsz: int | tuple[int, int] | list[int] | None = None,
    conf: float = DEFAULT_GUI_CONF,
    iou: float = 0.45,
    max_det: int | None = None,
    retina_masks: bool = False,
    tracking: bool = DEFAULT_GUI_TRACKING,
    tracker: str = DEFAULT_GUI_TRACKER,
    make_source: Callable[..., object] | None = None,
    on_result: Callable[[object, str, list[str]], None] | None = None,
) -> int:
    target_size = normalize_imgsz(imgsz or engine.metadata.default_imgsz)
    resolved_max_det = int(max_det or engine.metadata.max_det)
    model_name = _display_model_name(
        getattr(engine.metadata, "model_name", ""),
        getattr(engine.metadata, "model_path", ""),
    )
    active_source_spec = str(source_spec)
    active_labels = list(labels)
    active_tracking = bool(tracking)
    active_tracker = normalize_tracker_name(tracker)
    if not active_labels:
        raise ValueError("labels must contain at least one class name")

    if make_source is None:

        def _make_source(source_value: str, *, prefix: str, max_frames: int | None):
            return camera_source_from_spec(
                source_value,
                width=width,
                height=height,
                fps=fps,
                timeout_s=timeout_s,
                prefix=prefix,
                max_frames=max_frames,
                zero_copy=None,
                preview_cpu=True,
                target_imgsz=target_size,
                fp16=bool(engine.main_fp16),
                device=engine.device,
            )

        make_source = _make_source

    engine.clear_prompts()
    engine.set_classes(active_labels)
    source_options = discover_gui_camera_sources()
    gui_window = CameraGuiWindow(
        window_title,
        active_source_spec,
        source_options,
        model_name,
        active_labels,
        confidence=conf,
        tracking_enabled=active_tracking,
        tracker_name=active_tracker,
    )
    displayed_frames = 0
    current_source = None
    tracker_session = engine.create_tracker(tracker=active_tracker, frame_rate=fps) if active_tracking else None

    def _apply_update(
        new_source_spec: str,
        new_labels: list[str],
        new_tracking_enabled: bool,
        new_tracker_name: str,
    ) -> bool:
        nonlocal active_labels, active_source_spec, active_tracking, active_tracker, current_source, tracker_session

        labels_changed = new_labels != active_labels
        tracking_changed = bool(new_tracking_enabled) != active_tracking
        tracker_changed = normalize_tracker_name(new_tracker_name) != active_tracker
        if labels_changed:
            engine.clear_prompts()
            engine.set_classes(new_labels)
            active_labels = new_labels

        if tracking_changed or tracker_changed or labels_changed:
            active_tracking = bool(new_tracking_enabled)
            active_tracker = normalize_tracker_name(new_tracker_name)
            tracker_session = engine.create_tracker(tracker=active_tracker, frame_rate=fps) if active_tracking else None

        if new_source_spec != active_source_spec:
            try:
                next_source = make_source(
                    new_source_spec,
                    prefix=source_prefix,
                    max_frames=max_frames,
                )
            except Exception as exc:
                LOGGER.warning("Unable to switch GUI source to '%s': %s", new_source_spec, exc)
                message = f"Unable to open source '{new_source_spec}': {exc}"
                gui_window.mark_applied(
                    active_source_spec,
                    active_labels,
                    active_tracking,
                    active_tracker,
                    message=f"Applied labels; {message}" if labels_changed else message,
                )
                return False

            active_source_spec = new_source_spec
            current_source = next_source
            if tracker_session is not None:
                tracker_session.reset()
            gui_window.mark_applied(active_source_spec, active_labels, active_tracking, active_tracker)
            return True

        if labels_changed or tracking_changed or tracker_changed:
            gui_window.mark_applied(active_source_spec, active_labels, active_tracking, active_tracker)
        else:
            gui_window.set_status("No changes to apply")
        return False

    try:
        try:
            current_source = make_source(active_source_spec, prefix=source_prefix, max_frames=max_frames)
            gui_window.mark_applied(active_source_spec, active_labels, active_tracking, active_tracker)
        except Exception as exc:
            LOGGER.warning("Unable to open initial GUI source '%s': %s", active_source_spec, exc)
            gui_window.set_status(f"Unable to open source '{active_source_spec}': {exc}")

        while gui_window.is_open:
            gui_window.pump()
            update = gui_window.take_pending_update()
            if update is not None and _apply_update(*update):
                continue
            if current_source is None:
                if wait_ms > 0:
                    time.sleep(wait_ms / 1000.0)
                continue

            source_iter = iter(current_source)
            restart_source = False
            try:
                while gui_window.is_open:
                    gui_window.pump()
                    update = gui_window.take_pending_update()
                    if update is not None and _apply_update(*update):
                        restart_source = True
                        break

                    if restart_source:
                        break

                    try:
                        item = next(source_iter)
                    except StopIteration:
                        break

                    active_conf = gui_window.confidence()
                    if tracker_session is not None:
                        result = tracker_session.update(
                            item,
                            imgsz=target_size,
                            conf=active_conf,
                            iou=iou,
                            max_det=resolved_max_det,
                            retina_masks=retina_masks,
                        )
                    else:
                        result = engine.predict_item(
                            item=item,
                            imgsz=target_size,
                            conf=active_conf,
                            iou=iou,
                            max_det=resolved_max_det,
                            retina_masks=retina_masks,
                        )
                    if on_result is not None:
                        on_result(result, active_source_spec, list(active_labels))
                    frame = result.plot(color_mode="instance" if tracker_session is not None else "class")
                    _overlay_gui_text(frame, active_source_spec, result)
                    if not gui_window.show_frame(frame):
                        break
                    displayed_frames += 1

                    if wait_ms > 0:
                        time.sleep(wait_ms / 1000.0)
            finally:
                close_iter = getattr(source_iter, "close", None)
                if callable(close_iter):
                    close_iter()

            if not restart_source and getattr(current_source, "max_frames", None) is not None:
                break
    finally:
        gui_window.close()

    return displayed_frames


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the YOLOE TensorRT live camera GUI.")
    parser.add_argument(
        "--source",
        default=DEFAULT_GUI_SOURCE,
        help="Initial camera source. Accepts /dev/video*, rtsp://..., videotest://..., or a raw GStreamer pipeline. "
        "Defaults to /dev/video0.",
    )
    parser.add_argument(
        "--label",
        action="append",
        dest="labels",
        default=None,
        help="Initial runtime label. Repeat to add more than one. The GUI lets you edit labels live.",
    )
    parser.add_argument(
        "--artifact-dir",
        default=str(default_gui_artifact_dir()),
        help="Artifact bundle directory. Defaults to outputs/artifacts/camera_gui_320.",
    )
    parser.add_argument(
        "--model",
        default=default_gui_model_spec(),
        help="YOLOE checkpoint path or downloadable asset name to export if the artifact bundle is missing.",
    )
    parser.add_argument("--imgsz", type=int, default=DEFAULT_GUI_IMGSZ, help="Inference/export image size.")
    parser.add_argument("--wait-ms", type=int, default=DEFAULT_GUI_WAIT_MS, help="Per-frame GUI delay in milliseconds.")
    parser.add_argument(
        "--max-frames", type=int, default=0, help="Stop after N frames. 0 means run until the window closes."
    )
    parser.add_argument("--title", default=DEFAULT_GUI_TITLE, help="Tk window title.")
    parser.add_argument("--device", default="cuda:0", help="Torch/TensorRT device. Defaults to cuda:0.")
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_GUI_CONF,
        help="Initial detection confidence threshold. The GUI can change it live.",
    )
    parser.add_argument(
        "--track",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_GUI_TRACKING,
        help="Enable object tracking in the GUI. Enabled by default.",
    )
    parser.add_argument(
        "--tracker",
        choices=AVAILABLE_TRACKERS,
        default=DEFAULT_GUI_TRACKER,
        help="Tracker backend to use when tracking is enabled.",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("YOLOE_TRT_LOG_LEVEL", "INFO"),
        help="Logging level for the launcher.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    configure_logging(level=args.log_level, include_timestamps=True)

    labels = list(args.labels) if args.labels else list(DEFAULT_GUI_LABELS)
    confidence = parse_confidence_text(args.conf, fallback=float(DEFAULT_GUI_CONF))
    tracker_name = normalize_tracker_name(args.tracker)
    LOGGER.info(
        "Launching YOLOE camera GUI with source='%s', labels=%s, conf=%.2f, tracking=%s, tracker=%s, artifact_dir='%s'",
        args.source,
        labels,
        confidence,
        bool(args.track),
        tracker_name,
        args.artifact_dir,
    )
    engine = load_or_build_gui_engine(
        artifact_dir=args.artifact_dir,
        model_path=args.model,
        imgsz=int(args.imgsz),
        device=args.device,
    )
    displayed_frames = run_camera_gui(
        engine,
        source_spec=args.source,
        labels=labels,
        window_title=args.title,
        wait_ms=int(args.wait_ms),
        max_frames=None if int(args.max_frames) <= 0 else int(args.max_frames),
        imgsz=int(args.imgsz),
        conf=confidence,
        tracking=bool(args.track),
        tracker=tracker_name,
    )
    LOGGER.info("Camera GUI closed after displaying %d frame(s)", displayed_frames)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
