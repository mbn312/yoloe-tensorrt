from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from ultralytics.engine.results import Results
from yoloe_tensorrt.engine import YOLOEEngine
from yoloe_tensorrt.inputs import PreparedTensorInput
from yoloe_tensorrt.source import PreparedFrameMetadata, SourceItem


def test_resolve_original_image_path_uses_prepared_frame_metadata_shape() -> None:
    engine = object.__new__(YOLOEEngine)
    metadata = PreparedFrameMetadata(original_shape=(720, 1280), path="camera_frame000001")

    original, path = engine._resolve_original_image_path(metadata, None, (320, 320))

    assert path == "camera_frame000001"
    assert original.shape == (720, 1280, 3)
    assert original.flags.writeable


def test_resolve_original_image_path_prefers_prepared_frame_preview() -> None:
    engine = object.__new__(YOLOEEngine)
    preview = torch.zeros((5, 7, 3), dtype=torch.uint8).numpy()
    metadata = PreparedFrameMetadata(original_shape=(5, 7), path="camera_frame000002", preview_image=preview)

    original, path = engine._resolve_original_image_path(metadata, None, (320, 320))

    assert path == "camera_frame000002"
    assert original is preview


def test_resolve_original_image_path_returns_fresh_mutable_blank_fallback_images() -> None:
    engine = object.__new__(YOLOEEngine)

    first, first_path = engine._resolve_original_image_path(None, None, (320, 320))
    second, second_path = engine._resolve_original_image_path(None, "tensor1", (320, 320))

    assert first_path == "tensor0"
    assert second_path == "tensor1"
    assert first is not second
    assert first.shape == (320, 320, 3)
    assert first.flags.writeable
    first[0, 0, 0] = 255
    assert second[0, 0, 0] == 0


def test_active_names_uses_prompt_name_cache_without_embedding_concat() -> None:
    engine = object.__new__(YOLOEEngine)
    engine._text_names = ["bus"]
    engine._visual_names = ["person"]
    engine._active_names_cache = ()
    engine._active_prompt_embeddings = lambda: pytest.fail("active_names should not concatenate embeddings")  # type: ignore[method-assign]

    assert engine.active_names == ["bus", "person"]
    assert engine._active_names_cache == ("bus", "person")


def test_main_output_names_are_cached_for_native_runtime() -> None:
    class _FakeNativeRuntime:
        def __init__(self) -> None:
            self.calls = 0

        @property
        def output_names(self) -> list[str]:
            self.calls += 1
            return ["predictions", "proto"]

    fake_runtime = _FakeNativeRuntime()
    engine = object.__new__(YOLOEEngine)
    engine.native_main_runtime = fake_runtime
    engine.main_runtime = None
    engine._main_output_names_cache = ()

    assert engine._main_output_names == ("predictions", "proto")
    assert engine._main_output_names == ("predictions", "proto")
    assert fake_runtime.calls == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for prepared-tensor stream handoff tests")
def test_prepared_fast_path_forwards_current_cuda_stream() -> None:
    class _FakeNativeRuntime:
        output_names = ["predictions"]
        fp16 = False

        def __init__(self) -> None:
            self.calls: list[tuple[int, int, int, int]] = []

        def infer_tensor(self, image_ptr: int, target_h: int, target_w: int, producer_stream: int):
            self.calls.append((image_ptr, target_h, target_w, producer_stream))
            return {
                "predictions": torch.zeros((1, 6, 1), device="cuda"),
                "preprocess_ms": 0.0,
                "inference_ms": 0.0,
            }

    fake_runtime = _FakeNativeRuntime()
    fake_runtime.fp16 = True
    engine = object.__new__(YOLOEEngine)
    engine.device = torch.device("cuda:0")
    engine.native_main_runtime = fake_runtime
    engine._native_infer_tensor = fake_runtime.infer_tensor
    engine._native_infer_tensor_postprocessed = None
    engine.main_runtime = None
    engine.metadata = SimpleNamespace()
    engine._text_prompt_embeddings = None
    engine._visual_prompt_embeddings = None
    engine._text_names = ["bus"]
    engine._visual_names = []
    engine._active_prompt_embeddings = lambda: (torch.zeros((1, 1, 512), device="cuda"), ["bus"])  # type: ignore[method-assign]
    engine._postprocess_predictions = lambda **kwargs: Results(  # type: ignore[method-assign]
        kwargs["original_image"],
        path=kwargs["path"],
        names={0: "bus"},
        boxes=torch.zeros((0, 6), device="cpu"),
        speed=kwargs["speed"],
    )
    engine._build_native_legacy_result = lambda **kwargs: Results(  # type: ignore[method-assign]
        kwargs["original_image"],
        path=kwargs["path"],
        names={0: "bus"},
        boxes=torch.zeros((0, 6), device="cpu"),
        speed={"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0},
    )

    source_item = SourceItem(image=torch.zeros((8, 8, 3), dtype=torch.uint8).numpy(), path="frame0")
    prepared = PreparedTensorInput(
        tensor=torch.zeros((1, 3, 8, 8), dtype=torch.float32, device="cuda"),
        path=source_item.path,
        original_image=source_item,
    )

    stream = torch.cuda.Stream(device=engine.device)
    with torch.cuda.stream(stream):
        engine._predict_prepared_tensor(prepared, conf=0.25, iou=0.45, max_det=10, retina_masks=False)

    assert fake_runtime.calls
    _, _, _, producer_stream = fake_runtime.calls[-1]
    assert producer_stream == int(stream.cuda_stream)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for prepared-tensor stream handoff tests")
def test_prepared_fast_path_uses_captured_producer_stream_across_stream_contexts() -> None:
    class _FakeNativeRuntime:
        output_names = ["predictions"]
        fp16 = False

        def __init__(self) -> None:
            self.calls: list[tuple[int, int, int, int]] = []

        def infer_tensor(self, image_ptr: int, target_h: int, target_w: int, producer_stream: int):
            self.calls.append((image_ptr, target_h, target_w, producer_stream))
            return {
                "predictions": torch.zeros((1, 6, 1), device="cuda"),
                "preprocess_ms": 0.0,
                "inference_ms": 0.0,
            }

    fake_runtime = _FakeNativeRuntime()
    engine = object.__new__(YOLOEEngine)
    engine.device = torch.device("cuda:0")
    engine.native_main_runtime = fake_runtime
    engine._native_infer_tensor = fake_runtime.infer_tensor
    engine._native_infer_tensor_postprocessed = None
    engine.main_runtime = None
    engine.metadata = SimpleNamespace()
    engine._text_prompt_embeddings = None
    engine._visual_prompt_embeddings = None
    engine._text_names = ["bus"]
    engine._visual_names = []
    engine._active_prompt_embeddings = lambda: (torch.zeros((1, 1, 512), device="cuda"), ["bus"])  # type: ignore[method-assign]
    engine._postprocess_predictions = lambda **kwargs: Results(  # type: ignore[method-assign]
        kwargs["original_image"],
        path=kwargs["path"],
        names={0: "bus"},
        boxes=torch.zeros((0, 6), device="cpu"),
        speed=kwargs["speed"],
    )

    source_item = SourceItem(image=torch.zeros((8, 8, 3), dtype=torch.uint8).numpy(), path="frame1")
    producer = torch.cuda.Stream(device=engine.device)
    consumer = torch.cuda.Stream(device=engine.device)
    assert int(producer.cuda_stream) != int(consumer.cuda_stream)

    with torch.cuda.stream(producer):
        prepared = PreparedTensorInput(
            tensor=torch.zeros((1, 3, 8, 8), dtype=torch.float32, device="cuda"),
            path=source_item.path,
            original_image=source_item,
        )

    with torch.cuda.stream(consumer):
        engine._predict_prepared_tensor(prepared, conf=0.25, iou=0.45, max_det=10, retina_masks=False)

    assert fake_runtime.calls
    _, _, _, producer_stream = fake_runtime.calls[-1]
    assert producer_stream == int(producer.cuda_stream)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for prepared-tensor stream handoff tests")
def test_prepared_fast_path_skips_current_stream_lookup_when_input_is_already_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeNativeRuntime:
        output_names = ["predictions"]
        fp16 = False

        def __init__(self) -> None:
            self.calls: list[tuple[int, int, int, int]] = []

        def infer_tensor(self, image_ptr: int, target_h: int, target_w: int, producer_stream: int):
            self.calls.append((image_ptr, target_h, target_w, producer_stream))
            return {
                "predictions": torch.zeros((1, 6, 1), device="cuda"),
                "preprocess_ms": 0.0,
                "inference_ms": 0.0,
            }

    fake_runtime = _FakeNativeRuntime()
    engine = object.__new__(YOLOEEngine)
    engine.device = torch.device("cuda:0")
    engine.native_main_runtime = fake_runtime
    engine._native_infer_tensor = fake_runtime.infer_tensor
    engine._native_infer_tensor_postprocessed = None
    engine.main_runtime = None
    engine.metadata = SimpleNamespace()
    engine._text_prompt_embeddings = None
    engine._visual_prompt_embeddings = None
    engine._text_names = ["bus"]
    engine._visual_names = []
    engine._active_names_cache = ()
    engine._active_prompt_embeddings = lambda: (torch.zeros((1, 1, 512), device="cuda"), ["bus"])  # type: ignore[method-assign]
    engine._postprocess_predictions = lambda **kwargs: Results(  # type: ignore[method-assign]
        kwargs["original_image"],
        path=kwargs["path"],
        names={0: "bus"},
        boxes=torch.zeros((0, 6), device="cpu"),
        speed=kwargs["speed"],
    )
    engine._build_native_legacy_result = lambda **kwargs: Results(  # type: ignore[method-assign]
        kwargs["original_image"],
        path=kwargs["path"],
        names={0: "bus"},
        boxes=torch.zeros((0, 6), device="cpu"),
        speed={"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0},
    )

    producer_stream = 12345
    prepared = PreparedTensorInput(
        tensor=torch.zeros((1, 3, 8, 8), dtype=torch.float32, device="cuda"),
        path="tensor0",
        original_image=SourceItem(image=torch.zeros((8, 8, 3), dtype=torch.uint8).numpy(), path="tensor0"),
        producer_stream=producer_stream,
    )
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda *args, **kwargs: pytest.fail("current_stream should not be needed for ready prepared tensors"),
    )

    engine._predict_prepared_tensor(prepared, conf=0.25, iou=0.45, max_det=10, retina_masks=False)

    assert fake_runtime.calls
    assert fake_runtime.calls[-1][3] == producer_stream
