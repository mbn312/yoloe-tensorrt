from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Protocol, TypeAlias, cast

import numpy as np
import torch
import yaml
from ultralytics.engine.results import Results
from ultralytics.utils import IterableSimpleNamespace

from ._inference_types import PreparedTensorInput, prepare_tensor_input
from ._shapes import ImageSizeLike
from .logging_utils import get_logger
from .source import SourceItem

LOGGER = get_logger(__name__)

DEFAULT_TRACKER = "bytetrack"
AVAILABLE_TRACKERS = ("bytetrack", "botsort")
TrackerConfigValue: TypeAlias = str | int | float | bool | None
TrackerConfig: TypeAlias = dict[str, TrackerConfigValue]


class TrackerEngine(Protocol):
    @property
    def prompt_generation(self) -> int: ...

    def predict_item(
        self,
        item: SourceItem | PreparedTensorInput,
        **kwargs,
    ) -> Results: ...


class _TrackerBackend(Protocol):
    def update(self, boxes: np.ndarray, image: np.ndarray) -> np.ndarray: ...


DEFAULT_TRACKER_CONFIGS: dict[str, TrackerConfig] = {
    "bytetrack": {
        "tracker_type": "bytetrack",
        "track_high_thresh": 0.25,
        "track_low_thresh": 0.1,
        "new_track_thresh": 0.25,
        "track_buffer": 30,
        "match_thresh": 0.8,
        "fuse_score": True,
    },
    "botsort": {
        "tracker_type": "botsort",
        "track_high_thresh": 0.25,
        "track_low_thresh": 0.1,
        "new_track_thresh": 0.25,
        "track_buffer": 30,
        "match_thresh": 0.8,
        "fuse_score": True,
        "gmc_method": "sparseOptFlow",
        "proximity_thresh": 0.5,
        "appearance_thresh": 0.8,
        "with_reid": False,
        "model": "auto",
    },
}


def normalize_tracker_name(tracker: str | None) -> str:
    value = str(tracker or DEFAULT_TRACKER).strip().lower().replace("-", "")
    aliases = {
        "byte": "bytetrack",
        "bytetrack": "bytetrack",
        "botsort": "botsort",
    }
    try:
        return aliases[value]
    except KeyError as exc:
        raise ValueError(f"Unsupported tracker '{tracker}'. Expected one of: {', '.join(AVAILABLE_TRACKERS)}") from exc


def resolve_tracker_config(
    tracker: str,
    tracker_config: str | Path | TrackerConfig | None = None,
) -> TrackerConfig:
    tracker_name = normalize_tracker_name(tracker)
    config = deepcopy(DEFAULT_TRACKER_CONFIGS[tracker_name])

    if tracker_config is None:
        overrides: TrackerConfig = {}
    elif isinstance(tracker_config, dict):
        overrides = dict(tracker_config)
    else:
        config_path = Path(tracker_config)
        if not config_path.is_file():
            raise FileNotFoundError(f"Tracker config file does not exist: {config_path}")
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if loaded is None:
            overrides = {}
        elif isinstance(loaded, dict):
            overrides = cast(TrackerConfig, loaded)
        else:
            raise ValueError(f"Tracker config file must contain a mapping, got: {type(loaded)!r}")

    configured_tracker = overrides.get("tracker_type")
    if configured_tracker is not None and normalize_tracker_name(configured_tracker) != tracker_name:
        raise ValueError(f"Tracker config expects '{configured_tracker}', but the active tracker is '{tracker_name}'")

    config.update(overrides)
    config["tracker_type"] = tracker_name
    if tracker_name == "botsort" and bool(config.get("with_reid")):
        raise ValueError("BoT-SORT ReID is not supported by yoloe_tensorrt yet")
    return config


def _build_tracker_backend(tracker: str, config: TrackerConfig, frame_rate: int) -> _TrackerBackend:
    from ultralytics.trackers.bot_sort import BOTSORT
    from ultralytics.trackers.byte_tracker import BYTETracker

    tracker_map = {
        "bytetrack": BYTETracker,
        "botsort": BOTSORT,
    }
    tracker_name = normalize_tracker_name(tracker)
    tracker_cls = tracker_map[tracker_name]
    return tracker_cls(args=IterableSimpleNamespace(**config), frame_rate=int(frame_rate))


def _apply_tracking_to_result(result: Results, tracker_backend: _TrackerBackend) -> Results:
    if result.boxes is None or len(result.boxes) == 0:
        return result

    tracks = tracker_backend.update(result.boxes.cpu().numpy(), result.orig_img)
    if len(tracks) == 0:
        return result

    keep_indices_np = tracks[:, -1].astype(np.int64, copy=False)
    keep_indices = keep_indices_np
    if torch.is_tensor(result.boxes.data) and result.boxes.data.device.type != "cpu":
        keep_indices = torch.as_tensor(keep_indices_np, device=result.boxes.data.device)
    tracked_result = result[keep_indices]
    dtype = result.boxes.data.dtype if torch.is_tensor(result.boxes.data) else torch.float32
    tracked_boxes = torch.as_tensor(tracks[:, :-1], dtype=dtype)
    tracked_result.update(boxes=tracked_boxes)
    return tracked_result


class YOLOETrackerSession:
    def __init__(
        self,
        engine: TrackerEngine,
        tracker: str = DEFAULT_TRACKER,
        tracker_config: str | Path | TrackerConfig | None = None,
        frame_rate: int = 30,
    ) -> None:
        self.engine = engine
        self.tracker_name = normalize_tracker_name(tracker)
        self.frame_rate = int(frame_rate)
        self._tracker_config = resolve_tracker_config(self.tracker_name, tracker_config)
        self._backend = _build_tracker_backend(self.tracker_name, self._tracker_config, self.frame_rate)
        self._prompt_generation = int(engine.prompt_generation)
        self._source_key: str | None = None
        LOGGER.info(
            "Initialized tracker session backend='%s' frame_rate=%d",
            self.tracker_name,
            self.frame_rate,
        )

    @property
    def config(self) -> TrackerConfig:
        return deepcopy(self._tracker_config)

    def reset(self) -> None:
        self._backend = _build_tracker_backend(self.tracker_name, self._tracker_config, self.frame_rate)
        self._prompt_generation = int(self.engine.prompt_generation)
        self._source_key = None
        LOGGER.info("Reset tracker session backend='%s'", self.tracker_name)

    def _reset_if_needed(self, source_key: str | None) -> None:
        if self._prompt_generation != int(self.engine.prompt_generation):
            LOGGER.info("Resetting tracker session because active prompts changed")
            self.reset()

        if source_key is None:
            return

        if self._source_key is None:
            self._source_key = source_key
            return
        if self._source_key != source_key:
            LOGGER.info(
                "Resetting tracker session because source changed from '%s' to '%s'", self._source_key, source_key
            )
            self.reset()
            self._source_key = source_key

    def update(
        self,
        frame: SourceItem | PreparedTensorInput | object,
        source_key: str | None = None,
        imgsz: ImageSizeLike | None = None,
        conf: float = 0.25,
        iou: float = 0.45,
        max_det: int | None = None,
        retina_masks: bool = False,
        cuda: bool | None = None,
        input_hint: str | None = None,
        original_image: SourceItem | object | None = None,
        path: str | None = None,
    ) -> Results:
        from .inputs import normalize_single_inference_source

        self._reset_if_needed(source_key)
        item = normalize_single_inference_source(
            frame,
            default_prefix="frame",
            input_hint=input_hint,
            cuda=cuda,
            original_image=original_image,
            path=path,
            multiple_error="YOLOETrackerSession.update expects a single frame input",
        )
        result = self.engine.predict_item(
            item,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            max_det=max_det,
            retina_masks=retina_masks,
        )
        return _apply_tracking_to_result(result, self._backend)

    def update_cuda(
        self,
        tensor: torch.Tensor | object,
        *,
        original_image: SourceItem | object | None = None,
        path: str | None = None,
        source_key: str | None = None,
        conf: float = 0.25,
        iou: float = 0.45,
        max_det: int | None = None,
        retina_masks: bool = False,
    ) -> Results:
        prepared = prepare_tensor_input(
            tensor,
            path=path or "tensor0",
            original_image=original_image,
        )
        return self.update(
            prepared,
            source_key=source_key,
            conf=conf,
            iou=iou,
            max_det=max_det,
            retina_masks=retina_masks,
            cuda=True,
            input_hint="prepared",
        )
