from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from ultralytics.engine.results import Results
from yoloe_tensorrt.engine import YOLOEEngine
from yoloe_tensorrt.inputs import PreparedTensorInput
from yoloe_tensorrt.source import SourceItem


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
    engine.main_runtime = None
    engine.metadata = SimpleNamespace()
    engine._text_prompt_embeddings = None
    engine._visual_prompt_embeddings = None
    engine._text_names = ["bus"]
    engine._visual_names = []
    engine._active_prompt_embeddings = lambda: (torch.zeros((1, 1, 512), device="cuda"), ["bus"])  # type: ignore[method-assign]
    engine._resolve_original_sample = lambda original_image, path, input_shape: SimpleNamespace(  # type: ignore[method-assign]
        original=original_image.image, path=path or original_image.path
    )
    engine._postprocess_predictions = lambda **kwargs: Results(  # type: ignore[method-assign]
        kwargs["sample"].original,
        path=kwargs["sample"].path,
        names={0: "bus"},
        boxes=torch.zeros((0, 6), device="cpu"),
        speed=kwargs["speed"],
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
    engine.main_runtime = None
    engine.metadata = SimpleNamespace()
    engine._text_prompt_embeddings = None
    engine._visual_prompt_embeddings = None
    engine._text_names = ["bus"]
    engine._visual_names = []
    engine._active_prompt_embeddings = lambda: (torch.zeros((1, 1, 512), device="cuda"), ["bus"])  # type: ignore[method-assign]
    engine._resolve_original_sample = lambda original_image, path, input_shape: SimpleNamespace(  # type: ignore[method-assign]
        original=original_image.image, path=path or original_image.path
    )
    engine._postprocess_predictions = lambda **kwargs: Results(  # type: ignore[method-assign]
        kwargs["sample"].original,
        path=kwargs["sample"].path,
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
