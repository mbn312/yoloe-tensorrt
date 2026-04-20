from __future__ import annotations

from dataclasses import dataclass

import torch

from ._shapes import HWShape
from .source import OriginalImageReference, SourceItem


@dataclass(frozen=True)
class PreparedTensorInput:
    tensor: torch.Tensor
    path: str
    original_image: OriginalImageReference = None
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


@dataclass(frozen=True)
class PreparedExecutionInput:
    tensor: torch.Tensor
    input_shape: HWShape
    producer_stream: int


def prepare_tensor_input(
    tensor: torch.Tensor | object,
    *,
    path: str,
    original_image: OriginalImageReference = None,
    producer_stream: int | None = None,
) -> PreparedTensorInput:
    if isinstance(tensor, PreparedTensorInput):
        return tensor
    resolved_tensor = tensor if isinstance(tensor, torch.Tensor) else torch.as_tensor(tensor)
    return PreparedTensorInput(
        tensor=resolved_tensor,
        path=path,
        original_image=original_image,
        producer_stream=producer_stream,
    )
