from __future__ import annotations

import numpy as np
import pytest
import torch
from yoloe_tensorrt.inputs import PreparedTensorInput, normalize_inference_source, validate_prepared_tensor
from yoloe_tensorrt.preprocess import preprocess_image
from yoloe_tensorrt.source import SourceItem


def test_prepared_tensor_input_supports_legacy_unpacking() -> None:
    original = SourceItem(image=np.zeros((8, 8, 3), dtype=np.uint8), path="frame0")
    prepared = PreparedTensorInput(
        tensor=torch.zeros((1, 3, 8, 8), dtype=torch.float32),
        path=original.path,
        original_image=original,
    )

    tensor, item = prepared

    assert isinstance(tensor, torch.Tensor)
    assert item == original


def test_plain_tensor_defaults_to_raw_routing_in_auto_mode() -> None:
    tensor = torch.zeros((3, 8, 8), dtype=torch.float32)

    item = normalize_inference_source(tensor)[0]

    assert isinstance(item, SourceItem)
    assert item.image.shape == (8, 8, 3)


def test_prepared_input_hint_routes_tensor_to_prepared_path() -> None:
    tensor = torch.zeros((1, 3, 8, 8), dtype=torch.float32)

    item = normalize_inference_source(tensor, input_hint="prepared")[0]

    assert isinstance(item, PreparedTensorInput)


def test_prepared_input_hint_rejects_cuda_false() -> None:
    tensor = torch.zeros((3, 8, 8), dtype=torch.float32, device="cpu")

    with pytest.raises(ValueError, match="cannot be combined with cuda=False"):
        normalize_inference_source(tensor, cuda=False, input_hint="prepared")


def test_validate_prepared_tensor_rejects_integer_dtype() -> None:
    tensor = torch.zeros((1, 3, 8, 8), dtype=torch.uint8)

    with pytest.raises(ValueError, match="floating point dtype"):
        validate_prepared_tensor(tensor)


def test_preprocess_image_preserves_unit_range_float_inputs() -> None:
    item = SourceItem(image=np.full((8, 8, 3), 0.5, dtype=np.float32), path="frame0")

    sample = preprocess_image(
        item,
        imgsz=(8, 8),
        device=torch.device("cpu"),
        fp16=False,
        stride=32,
    )

    assert sample.tensor.dtype == torch.float32
    assert float(sample.tensor.max().item()) == pytest.approx(0.5, abs=1e-5)


def test_preprocess_image_preserves_unit_range_float_inputs_with_padding() -> None:
    item = SourceItem(image=np.full((8, 16, 3), 0.5, dtype=np.float32), path="frame0")

    sample = preprocess_image(
        item,
        imgsz=(16, 16),
        device=torch.device("cpu"),
        fp16=False,
        stride=32,
    )

    assert sample.tensor.dtype == torch.float32
    assert float(sample.tensor[0, 0, 8, 8].item()) == pytest.approx(0.5, abs=1e-5)
    assert float(sample.tensor[0, 0, 0, 0].item()) == pytest.approx(114.0 / 255.0, abs=1e-5)
