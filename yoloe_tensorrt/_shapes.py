from __future__ import annotations

from typing import TypeAlias

HWShape: TypeAlias = tuple[int, int]
ImageSizeLike: TypeAlias = int | HWShape | list[int]
TensorShape: TypeAlias = tuple[int, ...]
TensorProfileShape: TypeAlias = tuple[TensorShape, TensorShape, TensorShape]


def normalize_hw_shape(shape: HWShape | list[int], *, name: str = "shape") -> HWShape:
    if len(shape) != 2:
        raise ValueError(f"Expected {name} with 2 elements, got {shape}")
    return int(shape[0]), int(shape[1])


def normalize_imgsz(imgsz: ImageSizeLike) -> HWShape:
    if isinstance(imgsz, int):
        return (imgsz, imgsz)
    return normalize_hw_shape(imgsz, name="imgsz")
