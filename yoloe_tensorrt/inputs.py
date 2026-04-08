from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

import cv2
import numpy as np
import torch
from PIL import Image

from .logging_utils import get_logger
from .source import SourceItem, SourceStream, _is_iterable_source, _load_path, _normalize_array, _pil_to_bgr

LOGGER = get_logger(__name__)

InputHint = Literal["auto", "raw", "prepared"]


@dataclass(frozen=True)
class PreparedTensorInput:
    tensor: torch.Tensor
    path: str
    original_image: SourceItem | object | None = None
    producer_stream: int | None = None

    def __post_init__(self) -> None:
        if self.producer_stream is None and self.tensor.device.type == "cuda":
            object.__setattr__(self, "producer_stream", int(torch.cuda.current_stream(self.tensor.device).cuda_stream))

    @property
    def original_item(self) -> SourceItem | None:
        if isinstance(self.original_image, SourceItem):
            return self.original_image
        return None

    def __iter__(self):
        yield self.tensor
        yield self.original_item


InferenceSourceItem = SourceItem | PreparedTensorInput


def normalize_input_hint(input_hint: str | None) -> InputHint:
    value = str(input_hint or "auto").strip().lower()
    if value not in {"auto", "raw", "prepared"}:
        raise ValueError("input_hint must be one of: auto, raw, prepared")
    return value  # type: ignore[return-value]


def normalize_inference_source(
    source: object,
    *,
    default_prefix: str = "image",
    input_hint: str | None = None,
    cuda: bool | None = None,
    original_image: SourceItem | object | None = None,
    path: str | None = None,
) -> list[InferenceSourceItem]:
    items = list(
        iter_inference_sources(
            source,
            default_prefix=default_prefix,
            input_hint=input_hint,
            cuda=cuda,
            original_image=original_image,
            path=path,
        )
    )
    if not items:
        raise ValueError("No images were found in the provided source")
    LOGGER.debug("Normalized %d inference source item(s)", len(items))
    return items


def iter_inference_sources(
    source: object,
    *,
    default_prefix: str = "image",
    input_hint: str | None = None,
    cuda: bool | None = None,
    original_image: SourceItem | object | None = None,
    path: str | None = None,
) -> Iterator[InferenceSourceItem]:
    hint = normalize_input_hint(input_hint)
    if isinstance(source, SourceStream):
        yield from source
        return
    if isinstance(source, (SourceItem, PreparedTensorInput)):
        yield source
        return
    if isinstance(source, torch.Tensor):
        yield from _iter_tensor_source(
            source,
            default_prefix=default_prefix,
            input_hint=hint,
            cuda=cuda,
            original_image=original_image,
            path=path,
        )
        return
    if isinstance(source, (str, Path)):
        yield _load_path(source)
        return
    if isinstance(source, Image.Image):
        yield SourceItem(image=_pil_to_bgr(source), path=path or f"{default_prefix}0")
        return
    if isinstance(source, np.ndarray):
        yield SourceItem(image=_normalize_array(source), path=path or f"{default_prefix}0")
        return
    if _is_iterable_source(source):
        for index, item in enumerate(source):
            item_path = f"{path}_{index}" if path is not None else None
            yield from iter_inference_sources(
                item,
                default_prefix=f"{default_prefix}{index}",
                input_hint=hint,
                cuda=cuda,
                original_image=None,
                path=item_path,
            )
        return
    raise TypeError(f"Unsupported source type: {type(source)!r}")


def _iter_tensor_source(
    tensor: torch.Tensor,
    *,
    default_prefix: str,
    input_hint: InputHint,
    cuda: bool | None,
    original_image: SourceItem | object | None,
    path: str | None,
) -> Iterator[InferenceSourceItem]:
    if tensor.ndim == 4 and int(tensor.shape[0]) > 1:
        for index, item in enumerate(tensor):
            item_path = f"{path}_{index}" if path is not None else f"{default_prefix}{index}"
            yield from _iter_tensor_source(
                item,
                default_prefix=f"{default_prefix}{index}",
                input_hint=input_hint,
                cuda=cuda,
                original_image=None,
                path=item_path,
            )
        return

    item_path = path or f"{default_prefix}0"
    if input_hint == "prepared":
        if cuda is False:
            raise ValueError("input_hint='prepared' cannot be combined with cuda=False")
        validate_prepared_tensor(tensor)
        yield PreparedTensorInput(
            tensor=tensor,
            path=item_path,
            original_image=original_image,
        )
        return

    if cuda is True and tensor.device.type == "cuda":
        raise ValueError(
            "Fast CUDA routing for plain tensors requires input_hint='prepared' or a PreparedTensorInput instance"
        )

    yield SourceItem(image=_tensor_to_bgr_numpy(tensor), path=item_path)


def validate_prepared_tensor(tensor: torch.Tensor) -> None:
    layout = _tensor_layout(tensor)
    if layout not in {"chw", "nchw"}:
        raise ValueError("Prepared tensor inputs must use CHW or NCHW layout")
    channel_dim = int(tensor.shape[0] if layout == "chw" else tensor.shape[1])
    if channel_dim != 3:
        raise ValueError("Prepared tensor inputs must have exactly 3 channels")
    if not tensor.dtype.is_floating_point:
        raise ValueError("Prepared tensor inputs must use a floating point dtype")


def _tensor_layout(tensor: torch.Tensor) -> str | None:
    if tensor.ndim == 3:
        if int(tensor.shape[-1]) in {1, 3, 4}:
            return "hwc"
        if int(tensor.shape[0]) in {1, 3, 4}:
            return "chw"
        return None
    if tensor.ndim == 4:
        if int(tensor.shape[-1]) in {1, 3, 4}:
            return "nhwc"
        if int(tensor.shape[1]) in {1, 3, 4}:
            return "nchw"
        return None
    return None


def _tensor_to_bgr_numpy(tensor: torch.Tensor) -> np.ndarray:
    layout = _tensor_layout(tensor)
    if layout is None:
        raise ValueError(
            "Tensor input must be HWC/NHWC or CHW/NCHW with 1, 3, or 4 channels when routed as a raw image"
        )

    if tensor.ndim == 4:
        if int(tensor.shape[0]) != 1:
            raise ValueError("Raw tensor conversion only supports single-image tensors; split batched tensors first")
        tensor = tensor[0]
        layout = _tensor_layout(tensor)
        if layout is None:
            raise ValueError("Unable to infer tensor layout after squeezing the batch dimension")

    host_tensor = tensor.detach()
    if host_tensor.device.type != "cpu":
        host_tensor = host_tensor.to("cpu")
    host_tensor = host_tensor.contiguous()

    if layout == "chw":
        host_tensor = host_tensor.permute(1, 2, 0).contiguous()

    array = host_tensor.numpy()
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D tensor image after layout normalization, got shape {array.shape}")

    channels = int(array.shape[2])
    if channels == 1:
        array = np.repeat(array, 3, axis=2)
    elif channels == 4:
        array = cv2.cvtColor(array, cv2.COLOR_RGBA2BGR)
    elif channels == 3:
        array = array[..., ::-1]
    else:
        raise ValueError(f"Unsupported channel count {channels}")

    if array.dtype not in {np.uint8, np.float16, np.float32, np.float64}:
        array = array.astype(np.float32, copy=False)
    if np.issubdtype(array.dtype, np.floating):
        max_value = float(array.max())
        min_value = float(array.min())
        if 0.0 <= min_value and max_value <= 1.0:
            array = array * 255.0

    return np.ascontiguousarray(array)
