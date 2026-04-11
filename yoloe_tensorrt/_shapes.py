from __future__ import annotations


def normalize_imgsz(imgsz: int | tuple[int, int] | list[int]) -> tuple[int, int]:
    if isinstance(imgsz, int):
        return (imgsz, imgsz)
    if len(imgsz) != 2:
        raise ValueError(f"Expected imgsz with 2 elements, got {imgsz}")
    return int(imgsz[0]), int(imgsz[1])
