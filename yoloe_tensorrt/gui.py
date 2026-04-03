from __future__ import annotations

import argparse
import os
import re
import time
import tkinter as tk
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable

import cv2
from PIL import Image, ImageTk

from .assets import default_example_model_spec, resolve_model_checkpoint
from .gstreamer import camera_source_from_spec
from .logging_utils import configure_logging, get_logger
from .preprocess import normalize_imgsz

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
DEFAULT_GUI_WORKSPACE_BYTES = int(os.environ.get("YOLOE_TRT_GUI_WORKSPACE_BYTES", str(512 << 20)))


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


def default_gui_artifact_dir(imgsz: int = DEFAULT_GUI_IMGSZ) -> Path:
    return _repo_root() / "outputs" / "artifacts" / f"camera_gui_{int(imgsz)}"


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


def _overlay_gui_text(frame, source_spec: str, labels: list[str], confidence: float, result) -> None:
    lines = [
        f"source: {source_spec}",
        f"labels: {', '.join(labels)}",
        f"conf: {confidence:.2f}",
        "controls: edit source/labels on the left, close window to quit",
    ]
    if getattr(result, "speed", None):
        lines.append("speed ms: pre={preprocess:.1f} inf={inference:.1f} post={postprocess:.1f}".format(**result.speed))

    y = 30
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        y += 24


class CameraGuiWindow:
    def __init__(self, title: str, source_spec: str, labels: list[str], confidence: float) -> None:
        self._root = tk.Tk()
        self._root.title(title)
        self._root.resizable(True, True)
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._source_var = tk.StringVar(value=source_spec)
        self._label_input_var = tk.StringVar(value="")
        self._confidence_var = tk.DoubleVar(value=float(confidence))
        self._confidence_text_var = tk.StringVar(value=f"{float(confidence):.2f}")
        self._status_var = tk.StringVar(value="Edit the source and labels, then press Apply")
        self._pending_update: tuple[str, list[str]] | None = None
        self._closed = False
        self._photo: ImageTk.PhotoImage | None = None

        root_frame = tk.Frame(self._root, padx=12, pady=12)
        root_frame.pack(fill="both", expand=True)
        root_frame.columnconfigure(0, weight=0)
        root_frame.columnconfigure(1, weight=1)
        root_frame.rowconfigure(0, weight=1)

        controls = tk.Frame(root_frame)
        controls.grid(row=0, column=0, sticky="nsw", padx=(0, 12))
        controls.columnconfigure(0, weight=1)

        tk.Label(controls, text="Camera Source").grid(row=0, column=0, sticky="w")
        source_entry = tk.Entry(controls, textvariable=self._source_var, width=48)
        source_entry.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        source_entry.bind("<Return>", lambda _event: self._apply())

        tk.Label(controls, text="Current Labels").grid(row=2, column=0, sticky="w")
        self._labels_list = tk.Listbox(controls, selectmode="extended", exportselection=False, height=8, width=32)
        self._labels_list.grid(row=3, column=0, sticky="ew")
        self._set_label_list(labels)

        label_buttons = tk.Frame(controls)
        label_buttons.grid(row=4, column=0, sticky="ew", pady=(8, 10))
        label_buttons.columnconfigure(0, weight=1)
        label_buttons.columnconfigure(1, weight=1)
        label_buttons.columnconfigure(2, weight=1)

        tk.Button(label_buttons, text="Remove Selected", command=self._remove_selected_labels).grid(
            row=0, column=0, sticky="ew", padx=(0, 4)
        )
        tk.Button(label_buttons, text="Clear", command=self._clear_labels).grid(row=0, column=1, sticky="ew", padx=4)
        tk.Button(label_buttons, text="Apply", command=self._apply).grid(row=0, column=2, sticky="ew", padx=(4, 0))

        tk.Label(controls, text="Add Label(s)").grid(row=5, column=0, sticky="w")
        label_entry = tk.Entry(controls, textvariable=self._label_input_var, width=48)
        label_entry.grid(row=6, column=0, sticky="ew", pady=(0, 8))
        label_entry.bind("<Return>", lambda _event: self._add_labels_from_input())

        tk.Button(controls, text="Add", command=self._add_labels_from_input).grid(row=7, column=0, sticky="ew")

        tk.Label(controls, text="Confidence Threshold").grid(row=8, column=0, sticky="w")
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
        confidence_scale.grid(row=9, column=0, sticky="ew", pady=(0, 2))

        confidence_row = tk.Frame(controls)
        confidence_row.grid(row=10, column=0, sticky="ew", pady=(0, 8))
        confidence_row.columnconfigure(0, weight=1)
        confidence_row.columnconfigure(1, weight=0)

        confidence_entry = tk.Entry(confidence_row, textvariable=self._confidence_text_var, width=8)
        confidence_entry.grid(row=0, column=0, sticky="w")
        confidence_entry.bind("<Return>", lambda _event: self._apply_confidence_text())
        tk.Button(confidence_row, text="Apply Confidence", command=self._apply_confidence_text).grid(
            row=0, column=1, sticky="w", padx=(8, 0)
        )

        tk.Label(
            controls,
            text="Use commas or new lines to add multiple labels at once.",
            anchor="w",
            justify="left",
        ).grid(row=11, column=0, sticky="ew", pady=(8, 0))
        tk.Label(controls, textvariable=self._status_var, anchor="w", justify="left", wraplength=320).grid(
            row=12, column=0, sticky="ew", pady=(8, 0)
        )

        self._video_label = tk.Label(root_frame, text="Waiting for frames...", anchor="center")
        self._video_label.grid(row=0, column=1, sticky="nsew")

    @property
    def is_open(self) -> bool:
        return not self._closed

    def _on_close(self) -> None:
        self._closed = True
        try:
            self._root.destroy()
        except tk.TclError:
            pass

    def _set_label_list(self, labels: Iterable[str]) -> None:
        self._labels_list.delete(0, tk.END)
        for label in labels:
            self._labels_list.insert(tk.END, label)

    def _current_labels(self) -> list[str]:
        return [str(item) for item in self._labels_list.get(0, tk.END)]

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
            self._status_var.set(f"Added {added} label(s); press Apply to activate them")

    def _remove_selected_labels(self) -> None:
        selected = list(self._labels_list.curselection())
        if not selected:
            self._status_var.set("Select one or more labels to remove")
            return

        labels = self._current_labels()
        for index in reversed(selected):
            del labels[index]
        self._set_label_list(labels)
        self._status_var.set("Removed selected label(s); press Apply to activate the change")

    def _clear_labels(self) -> None:
        self._set_label_list([])
        self._status_var.set("Cleared staged labels; add at least one label before applying")

    def _on_confidence_change(self, value: str) -> None:
        confidence = parse_confidence_text(value, fallback=float(DEFAULT_GUI_CONF))
        self._confidence_text_var.set(f"{confidence:.2f}")

    def _apply_confidence_text(self) -> None:
        confidence = parse_confidence_text(self._confidence_text_var.get(), fallback=float(DEFAULT_GUI_CONF))
        self._confidence_var.set(confidence)
        self._confidence_text_var.set(f"{confidence:.2f}")
        self._status_var.set(f"Confidence threshold set to {confidence:.2f}; it applies on the next frame")

    def _apply(self) -> None:
        source = self._source_var.get().strip()
        labels = self._current_labels()
        if not source:
            self._status_var.set("Source cannot be empty")
            return
        if not labels:
            self._status_var.set("Add at least one label before applying")
            return
        self._pending_update = (source, labels)
        self._status_var.set("Queued source/label update; confidence changes already apply live")

    def pump(self) -> None:
        if self._closed:
            return
        try:
            self._root.update_idletasks()
            self._root.update()
        except tk.TclError:
            self._closed = True

    def take_pending_update(self) -> tuple[str, list[str]] | None:
        update = self._pending_update
        self._pending_update = None
        return update

    def set_status(self, message: str) -> None:
        self._status_var.set(message)

    def confidence(self) -> float:
        return float(self._confidence_var.get())

    def mark_applied(self, source: str, labels: list[str], message: str | None = None) -> None:
        self._source_var.set(source)
        self._set_label_list(labels)
        self._status_var.set(
            message or f"Active source updated; labels={', '.join(labels)} conf={self.confidence():.2f}"
        )

    def show_frame(self, frame_bgr) -> None:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        self._photo = ImageTk.PhotoImage(image=image, master=self._root)
        self._video_label.configure(image=self._photo, text="")

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
    make_source: Callable[..., object] | None = None,
    on_result: Callable[[object, str, list[str]], None] | None = None,
) -> int:
    target_size = normalize_imgsz(imgsz or engine.metadata.default_imgsz)
    resolved_max_det = int(max_det or engine.metadata.max_det)
    active_source_spec = str(source_spec)
    active_labels = list(labels)
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
            )

        make_source = _make_source

    engine.clear_prompts()
    engine.set_classes(active_labels)
    gui_window = CameraGuiWindow(window_title, active_source_spec, active_labels, confidence=conf)
    displayed_frames = 0
    current_source = None

    def _apply_update(new_source_spec: str, new_labels: list[str]) -> bool:
        nonlocal active_labels, active_source_spec, current_source

        labels_changed = new_labels != active_labels
        if labels_changed:
            engine.clear_prompts()
            engine.set_classes(new_labels)
            active_labels = new_labels

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
                if labels_changed:
                    gui_window.mark_applied(
                        active_source_spec,
                        active_labels,
                        message=f"Applied labels; {message}",
                    )
                else:
                    gui_window.set_status(message)
                return False

            active_source_spec = new_source_spec
            current_source = next_source
            gui_window.mark_applied(active_source_spec, active_labels)
            return True

        if labels_changed:
            gui_window.mark_applied(active_source_spec, active_labels)
        else:
            gui_window.set_status("No changes to apply")
        return False

    try:
        try:
            current_source = make_source(active_source_spec, prefix=source_prefix, max_frames=max_frames)
            gui_window.mark_applied(active_source_spec, active_labels)
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
                    frame = result.plot()
                    _overlay_gui_text(frame, active_source_spec, active_labels, active_conf, result)
                    gui_window.show_frame(frame)
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
    parser.add_argument("--source", default=DEFAULT_GUI_SOURCE, help="Initial camera source. Defaults to /dev/video0.")
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
    LOGGER.info(
        "Launching YOLOE camera GUI with source='%s', labels=%s, conf=%.2f, artifact_dir='%s'",
        args.source,
        labels,
        confidence,
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
    )
    LOGGER.info("Camera GUI closed after displaying %d frame(s)", displayed_frames)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
