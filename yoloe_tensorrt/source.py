from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import cv2
import numpy as np
from PIL import Image

from .logging_utils import get_logger

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class SourceItem:
    image: np.ndarray
    path: str


class SourceStream(Iterable[SourceItem]):
    is_live_source: bool = False
    max_frames: int | None = None


def _pil_to_bgr(image: Image.Image) -> np.ndarray:
    array = np.asarray(image.convert("RGB"))
    return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)


def _normalize_array(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        return cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
    if array.ndim != 3:
        raise ValueError(f"Expected a 2D or 3D array, got shape {array.shape}")
    if array.shape[2] == 1:
        return cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
    if array.shape[2] == 3:
        return np.ascontiguousarray(array)
    if array.shape[2] == 4:
        return cv2.cvtColor(array, cv2.COLOR_BGRA2BGR)
    raise ValueError(f"Unsupported channel count {array.shape[2]}")


def _load_path(path: str | Path) -> SourceItem:
    resolved = str(Path(path))
    LOGGER.debug("Loading image source from '%s'", resolved)
    image = cv2.imread(resolved, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Unable to load image at '{resolved}'")
    return SourceItem(image=_normalize_array(image), path=resolved)


def _is_iterable_source(source: object) -> bool:
    return hasattr(source, "__iter__") and not isinstance(source, (str, bytes, Path, np.ndarray, Image.Image))


def is_live_source(source: object) -> bool:
    return isinstance(source, SourceStream) and bool(getattr(source, "is_live_source", False))


def is_finite_live_source(source: object) -> bool:
    return is_live_source(source) and getattr(source, "max_frames", None) is not None


def normalize_source(source: object, default_prefix: str = "image") -> list[SourceItem]:
    items = list(iter_sources(source, default_prefix=default_prefix))
    if not items:
        raise ValueError("No images were found in the provided source")
    LOGGER.debug("Normalized %d source item(s)", len(items))
    return items


def iter_sources(source: object, default_prefix: str = "image") -> Iterator[SourceItem]:
    if isinstance(source, SourceStream):
        yield from source
        return
    if isinstance(source, (str, Path)):
        yield _load_path(source)
        return
    if isinstance(source, Image.Image):
        yield SourceItem(image=_pil_to_bgr(source), path=f"{default_prefix}0")
        return
    if isinstance(source, np.ndarray):
        yield SourceItem(image=_normalize_array(source), path=f"{default_prefix}0")
        return
    if _is_iterable_source(source):
        for index, item in enumerate(source):
            if isinstance(item, SourceItem):
                yield item
            elif isinstance(item, (str, Path)):
                yield _load_path(item)
            elif isinstance(item, Image.Image):
                yield SourceItem(image=_pil_to_bgr(item), path=f"{default_prefix}{index}")
            elif isinstance(item, np.ndarray):
                yield SourceItem(image=_normalize_array(item), path=f"{default_prefix}{index}")
            else:
                raise TypeError(f"Unsupported source item type: {type(item)!r}")
        return
    raise TypeError(f"Unsupported source type: {type(source)!r}")


def stream_sources(source: object, default_prefix: str = "frame") -> Iterable[SourceItem]:
    return iter_sources(source, default_prefix=default_prefix)
