from __future__ import annotations

import pytest
from yoloe_tensorrt._shapes import normalize_hw_shape, normalize_imgsz


def test_normalize_hw_shape_casts_and_preserves_order() -> None:
    assert normalize_hw_shape([480.0, 640.0], name="original_shape") == (480, 640)


def test_normalize_hw_shape_requires_exactly_two_elements() -> None:
    with pytest.raises(ValueError, match="Expected default_imgsz with 2 elements"):
        normalize_hw_shape([640], name="default_imgsz")


def test_normalize_imgsz_uses_shared_hw_normalizer_for_pair_inputs() -> None:
    assert normalize_imgsz([320, 512]) == (320, 512)
