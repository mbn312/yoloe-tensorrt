from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from yoloe_tensorrt.engine import YOLOEEngine
from yoloe_tensorrt.inputs import (
    PreparedTensorInput,
    iter_inference_sources,
    normalize_inference_source,
    normalize_single_inference_source,
    validate_prepared_tensor,
)
from yoloe_tensorrt.preprocess import preprocess_image
from yoloe_tensorrt.source import SourceItem, SourceStream


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


def test_normalize_inference_source_rejects_rtsp_urls() -> None:
    with pytest.raises(ValueError, match="Live sources require stream=True"):
        normalize_inference_source("rtsp://camera.local/stream")


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


def test_iter_inference_sources_routes_rtsp_urls_to_gstreamer_source(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeRtspSource(SourceStream):
        is_live_source = True
        max_frames = 1

        def __iter__(self):
            yield SourceItem(image=np.zeros((4, 4, 3), dtype=np.uint8), path="rtsp_frame000000")

    captured: dict[str, object] = {}

    def _fake_camera_source_from_spec(source_value: object, **kwargs) -> SourceStream:
        captured["source_value"] = source_value
        captured.update(kwargs)
        return _FakeRtspSource()

    monkeypatch.setattr("yoloe_tensorrt.inputs.camera_source_from_spec", _fake_camera_source_from_spec)
    items = list(iter_inference_sources("rtsp://camera.local/stream"))

    assert len(items) == 1
    assert isinstance(items[0], SourceItem)
    assert captured["source_value"] == "rtsp://camera.local/stream"
    assert captured["prefix"] == "rtsp"
    assert captured["width"] is None
    assert captured["height"] is None
    assert captured["fps"] is None


@pytest.mark.parametrize("method_name", ["predict", "track"])
def test_engine_rejects_rtsp_urls_without_stream(method_name: str) -> None:
    engine = object.__new__(YOLOEEngine)
    engine.metadata = SimpleNamespace(default_imgsz=(640, 640), max_det=100)
    engine._prompt_generation = 0

    method = getattr(engine, method_name)
    with pytest.raises(ValueError, match="Live sources require stream=True"):
        method("rtsp://camera.local/stream")


@pytest.mark.parametrize("method_name", ["predict", "track"])
def test_engine_rejects_iterable_rtsp_urls_without_stream(method_name: str) -> None:
    engine = object.__new__(YOLOEEngine)
    engine.metadata = SimpleNamespace(default_imgsz=(640, 640), max_det=100)
    engine._prompt_generation = 0

    method = getattr(engine, method_name)
    with pytest.raises(ValueError, match="Live sources require stream=True"):
        method(["rtsp://camera.local/stream"])


@pytest.mark.parametrize("method_name", ["predict", "track"])
def test_engine_rejects_generator_rtsp_urls_without_stream(method_name: str) -> None:
    engine = object.__new__(YOLOEEngine)
    engine.metadata = SimpleNamespace(default_imgsz=(640, 640), max_det=100)
    engine._prompt_generation = 0

    def _source():
        yield "rtsp://camera.local/stream"

    method = getattr(engine, method_name)
    with pytest.raises(ValueError, match="Live sources require stream=True"):
        method(_source())


@pytest.mark.parametrize(
    ("method_name", "iter_attr"),
    [("predict", "_predict_iter"), ("track", "_track_iter")],
)
def test_engine_does_not_eagerly_materialize_iterable_sources(
    method_name: str,
    iter_attr: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = object.__new__(YOLOEEngine)
    engine.metadata = SimpleNamespace(default_imgsz=(640, 640), max_det=100)
    consumed = {"count": 0}

    def _source():
        consumed["count"] += 1
        yield PreparedTensorInput(
            tensor=torch.zeros((1, 3, 8, 8), dtype=torch.float32),
            path="tensor0",
        )

    def _fake_iter(*args, **kwargs):
        yield "sentinel"

    monkeypatch.setattr(engine, iter_attr, _fake_iter)
    method = getattr(engine, method_name)

    results = method(_source())

    assert len(results) == 1
    assert results[0] == "sentinel"
    assert consumed["count"] == 0


def test_iter_inference_sources_allows_finite_live_sources_when_unbounded_live_disabled() -> None:
    class _FiniteLiveSource(SourceStream):
        is_live_source = True
        max_frames = 1

        def __iter__(self):
            yield SourceItem(image=np.zeros((4, 4, 3), dtype=np.uint8), path="frame0")

    items = list(iter_inference_sources([_FiniteLiveSource()], allow_unbounded_live=False))

    assert len(items) == 1
    assert isinstance(items[0], SourceItem)
    assert items[0].path == "frame0"


def test_normalize_single_inference_source_returns_one_item() -> None:
    item = normalize_single_inference_source(np.zeros((4, 4, 3), dtype=np.uint8))

    assert isinstance(item, SourceItem)
    assert item.path == "image0"


def test_normalize_single_inference_source_rejects_empty_iterables() -> None:
    with pytest.raises(ValueError, match="No images were found"):
        normalize_single_inference_source([])


def test_normalize_single_inference_source_rejects_multiple_items_lazily() -> None:
    consumed = {"count": 0}

    def _source():
        for _ in range(3):
            consumed["count"] += 1
            yield np.zeros((4, 4, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="single frame"):
        normalize_single_inference_source(_source(), multiple_error="single frame")

    assert consumed["count"] == 2
