from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from ultralytics.data.augment import LetterBox

from .logging_utils import get_logger
from .source import SourceItem
from .tensor_utils import torch_from_numpy_safe

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class PreprocessedSample:
    original: np.ndarray
    transformed: np.ndarray
    tensor: torch.Tensor
    path: str


def normalize_imgsz(imgsz: int | tuple[int, int] | list[int]) -> tuple[int, int]:
    if isinstance(imgsz, int):
        return (imgsz, imgsz)
    if len(imgsz) != 2:
        raise ValueError(f"Expected imgsz with 2 elements, got {imgsz}")
    return int(imgsz[0]), int(imgsz[1])


def preprocess_image(
    item: SourceItem,
    imgsz: int | tuple[int, int] | list[int],
    device: torch.device,
    fp16: bool,
    stride: int,
) -> PreprocessedSample:
    target_shape = normalize_imgsz(imgsz)
    LOGGER.debug("Preprocessing '%s' from shape=%s to target_shape=%s", item.path, item.image.shape[:2], target_shape)
    transformed = LetterBox(new_shape=target_shape, auto=False, stride=stride)(image=item.image)
    array = transformed
    if array.shape[-1] == 3:
        array = array[..., ::-1]
    array = np.ascontiguousarray(array.transpose((2, 0, 1)))
    tensor = torch_from_numpy_safe(array, device=device)
    tensor = tensor.half() if fp16 else tensor.float()
    tensor /= 255.0
    tensor = tensor.unsqueeze(0)
    LOGGER.debug(
        "Prepared tensor for '%s' with shape=%s dtype=%s", item.path, tuple(int(v) for v in tensor.shape), tensor.dtype
    )
    return PreprocessedSample(
        original=item.image,
        transformed=transformed,
        tensor=tensor,
        path=item.path,
    )
