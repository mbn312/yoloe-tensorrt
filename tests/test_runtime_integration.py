from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.dlpack import from_dlpack
from yoloe_tensorrt import GStreamerSource, PreparedTensorInput, YOLOEEngine
from yoloe_tensorrt.source import SourceItem, normalize_source

TEST_CONF = 0.1


def _expected_names(labels: list[str]) -> dict[int, str]:
    return {index: label for index, label in enumerate(labels)}


def _assert_result_contract(result, image_path: Path, expected_names: dict[int, str]) -> None:
    with Image.open(image_path) as image:
        expected_shape = (image.height, image.width)

    assert Path(result.path).name == image_path.name
    assert result.orig_shape == expected_shape
    assert result.names == expected_names
    assert result.boxes is not None
    assert result.boxes.data.ndim == 2
    assert result.boxes.data.shape[1] == 6
    if result.masks is not None:
        assert tuple(result.masks.orig_shape) == expected_shape
        assert result.masks.data.shape[0] == result.boxes.data.shape[0]


def _assert_live_result_contract(result, expected_names: dict[int, str], path_prefix: str) -> None:
    assert Path(result.path).name.startswith(path_prefix)
    assert result.orig_shape[0] > 0
    assert result.orig_shape[1] > 0
    assert result.names == expected_names
    assert result.boxes is not None
    assert result.boxes.data.ndim == 2
    assert result.boxes.data.shape[1] == 6
    if result.masks is not None:
        assert tuple(result.masks.orig_shape) == tuple(result.orig_shape)
        assert result.masks.data.shape[0] == result.boxes.data.shape[0]


def _sorted_result_tensors(result) -> tuple[torch.Tensor, torch.Tensor | None]:
    boxes = result.boxes.data.detach().cpu()
    order = sorted(
        range(int(boxes.shape[0])),
        key=lambda index: (
            int(round(float(boxes[index, 5].item()))),
            -float(boxes[index, 4].item()),
            float(boxes[index, 0].item()),
            float(boxes[index, 1].item()),
            float(boxes[index, 2].item()),
            float(boxes[index, 3].item()),
        ),
    )
    if order:
        boxes = boxes[order]
    masks = result.masks.data.detach().cpu() if result.masks is not None else None
    if masks is not None and order:
        masks = masks[order]
    return boxes, masks


def _assert_native_postprocess_matches_python_reference(actual, reference) -> None:
    assert actual.orig_shape == reference.orig_shape
    assert actual.names == reference.names

    actual_boxes, actual_masks = _sorted_result_tensors(actual)
    reference_boxes, reference_masks = _sorted_result_tensors(reference)
    assert actual_boxes.shape == reference_boxes.shape
    torch.testing.assert_close(actual_boxes[:, :5], reference_boxes[:, :5], atol=1e-2, rtol=1e-3)
    assert torch.equal(actual_boxes[:, 5].round().to(torch.int64), reference_boxes[:, 5].round().to(torch.int64))

    if reference_masks is None:
        assert actual_masks is None
        return

    assert actual_masks is not None
    assert actual_masks.shape == reference_masks.shape
    for index in range(int(actual_masks.shape[0])):
        actual_mask = actual_masks[index].to(torch.bool)
        reference_mask = reference_masks[index].to(torch.bool)
        union = torch.logical_or(actual_mask, reference_mask).sum().item()
        if union == 0:
            continue
        intersection = torch.logical_and(actual_mask, reference_mask).sum().item()
        assert (intersection / union) >= 0.98


def _python_reference_from_native_image_outputs(
    runtime_engine: YOLOEEngine,
    item: SourceItem,
    labels: list[str],
    imgsz: tuple[int, int],
    conf: float,
    iou: float,
    max_det: int,
    retina_masks: bool,
):
    native_runtime = runtime_engine.native_main_runtime
    assert native_runtime is not None
    native_outputs = native_runtime.infer_image(item.image, imgsz[0], imgsz[1])
    outputs = {name: from_dlpack(native_outputs[name]) for name in runtime_engine._main_output_names}
    speed = {
        "preprocess": float(native_outputs["preprocess_ms"]),
        "inference": float(native_outputs["inference_ms"]),
        "postprocess": 0.0,
    }
    return runtime_engine._postprocess_predictions(
        outputs=outputs,
        sample=SimpleNamespace(original=item.image, path=item.path),
        input_shape=tuple(int(v) for v in native_outputs["input_shape"]),
        names=labels,
        conf=conf,
        iou=iou,
        max_det=max_det,
        retina_masks=retina_masks,
        speed=speed,
    )


def _python_reference_from_native_tensor_outputs(
    runtime_engine: YOLOEEngine,
    prepared: PreparedTensorInput,
    labels: list[str],
    conf: float,
    iou: float,
    max_det: int,
    retina_masks: bool,
):
    native_runtime = runtime_engine.native_main_runtime
    assert native_runtime is not None
    input_shape = (int(prepared.tensor.shape[2]), int(prepared.tensor.shape[3]))
    producer_stream = (
        prepared.producer_stream
        if prepared.producer_stream is not None
        else int(torch.cuda.current_stream(runtime_engine.device).cuda_stream)
    )
    native_outputs = native_runtime.infer_tensor(
        int(prepared.tensor.data_ptr()),
        input_shape[0],
        input_shape[1],
        producer_stream,
    )
    outputs = {name: from_dlpack(native_outputs[name]) for name in runtime_engine._main_output_names}
    sample = runtime_engine._resolve_original_sample(prepared.original_image, prepared.path, input_shape)
    speed = {
        "preprocess": float(native_outputs["preprocess_ms"]),
        "inference": float(native_outputs["inference_ms"]),
        "postprocess": 0.0,
    }
    return runtime_engine._postprocess_predictions(
        outputs=outputs,
        sample=sample,
        input_shape=tuple(int(v) for v in native_outputs["input_shape"]),
        names=labels,
        conf=conf,
        iou=iou,
        max_det=max_det,
        retina_masks=retina_masks,
        speed=speed,
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    ("image_key", "labels"),
    [
        ("bus", ["bus"]),
        ("zidane", ["person"]),
        ("dog", ["dog"]),
    ],
)
def test_engine_predicts_test_images_with_text_prompts(
    runtime_engine: YOLOEEngine,
    test_images: dict[str, Path],
    image_key: str,
    labels: list[str],
) -> None:
    runtime_engine.clear_prompts()
    runtime_engine.set_classes(labels)

    image_path = test_images[image_key]
    results = runtime_engine.predict(image_path, conf=TEST_CONF)

    assert len(results) == 1
    _assert_result_contract(results[0], image_path, _expected_names(labels))
    assert results[0].boxes.data.shape[0] >= 1


@pytest.mark.integration
def test_engine_streams_test_images_in_input_order(
    runtime_engine: YOLOEEngine,
    test_images: dict[str, Path],
) -> None:
    runtime_engine.clear_prompts()
    labels = ["bus", "person", "dog"]
    runtime_engine.set_classes(labels)

    ordered_paths = [
        test_images["bus"],
        test_images["zidane"],
        test_images["dog"],
    ]
    results = list(runtime_engine.predict(ordered_paths, stream=True, conf=TEST_CONF))

    assert len(results) == len(ordered_paths)
    assert [Path(result.path).name for result in results] == [path.name for path in ordered_paths]
    for result, path in zip(results, ordered_paths):
        _assert_result_contract(result, path, _expected_names(labels))

    assert any(result.boxes.data.shape[0] >= 1 for result in results)


@pytest.mark.integration
def test_engine_predicts_bus_image_with_visual_prompt(
    runtime_engine: YOLOEEngine,
    test_images: dict[str, Path],
) -> None:
    runtime_engine.clear_prompts()
    image_path = test_images["bus"]

    runtime_engine.set_visual_prompts(
        image_path,
        bboxes=[[20, 230, 810, 1060]],
        classes=["bus"],
    )
    results = runtime_engine.predict(image_path, conf=TEST_CONF)

    assert len(results) == 1
    _assert_result_contract(results[0], image_path, {0: "bus"})
    assert results[0].boxes.data.shape[0] >= 1


@pytest.mark.integration
def test_engine_predict_uses_fast_prepared_tensor_path_by_default(
    runtime_engine: YOLOEEngine,
    test_images: dict[str, Path],
) -> None:
    runtime_engine.clear_prompts()
    labels = ["bus"]
    runtime_engine.set_classes(labels)

    image_path = test_images["bus"]
    prepared_input = runtime_engine.prepare_cuda_input(image_path)

    result = runtime_engine.predict(prepared_input, conf=TEST_CONF)[0]

    _assert_result_contract(result, image_path, _expected_names(labels))
    assert result.speed["preprocess"] == pytest.approx(0.0)
    assert result.boxes.data.shape[0] >= 1


@pytest.mark.integration
def test_native_preprocess_matches_python_reference_for_real_image(
    runtime_engine: YOLOEEngine,
    test_images: dict[str, Path],
) -> None:
    native_runtime = runtime_engine.native_main_runtime
    assert native_runtime is not None

    image_path = test_images["bus"]
    item = normalize_source(image_path, default_prefix="frame")[0]
    prepared = runtime_engine.prepare_cuda_input(item)
    native_outputs = native_runtime._preprocess_image_to_tensor(
        item.image,
        int(prepared.tensor.shape[2]),
        int(prepared.tensor.shape[3]),
    )
    native_tensor = from_dlpack(native_outputs["tensor"])

    assert native_tensor.shape == prepared.tensor.shape
    assert native_tensor.dtype == prepared.tensor.dtype
    assert torch.allclose(native_tensor.float(), prepared.tensor.float(), atol=5e-3, rtol=1e-3)


@pytest.mark.integration
def test_native_preprocess_matches_python_reference_for_padded_image(
    runtime_engine: YOLOEEngine,
) -> None:
    native_runtime = runtime_engine.native_main_runtime
    assert native_runtime is not None

    item = SourceItem(image=np.full((48, 96, 3), 127, dtype=np.uint8), path="synthetic.png")
    prepared = runtime_engine.prepare_cuda_input(item, imgsz=(320, 320))
    native_outputs = native_runtime._preprocess_image_to_tensor(item.image, 320, 320)
    native_tensor = from_dlpack(native_outputs["tensor"])

    assert native_tensor.shape == prepared.tensor.shape
    assert native_tensor.dtype == prepared.tensor.dtype
    assert torch.allclose(native_tensor.float(), prepared.tensor.float(), atol=5e-3, rtol=1e-3)
    assert float(native_tensor[0, 0, 0, 0].float().item()) == pytest.approx(114.0 / 255.0, abs=5e-3)


@pytest.mark.integration
@pytest.mark.parametrize("retina_masks", [False, True])
def test_native_postprocess_matches_python_reference_for_image(
    runtime_engine: YOLOEEngine,
    test_images: dict[str, Path],
    retina_masks: bool,
) -> None:
    runtime_engine.clear_prompts()
    labels = ["bus"]
    runtime_engine.set_classes(labels)

    image_path = test_images["bus"]
    item = normalize_source(image_path, default_prefix="image")[0]
    reference = _python_reference_from_native_image_outputs(
        runtime_engine,
        item,
        labels,
        imgsz=(320, 320),
        conf=TEST_CONF,
        iou=0.45,
        max_det=runtime_engine.metadata.max_det,
        retina_masks=retina_masks,
    )
    actual = runtime_engine.predict(image_path, conf=TEST_CONF, imgsz=320, retina_masks=retina_masks)[0]

    _assert_native_postprocess_matches_python_reference(actual, reference)


@pytest.mark.integration
@pytest.mark.parametrize("retina_masks", [False, True])
def test_native_postprocess_matches_python_reference_for_prepared_tensor(
    runtime_engine: YOLOEEngine,
    test_images: dict[str, Path],
    retina_masks: bool,
) -> None:
    runtime_engine.clear_prompts()
    labels = ["bus"]
    runtime_engine.set_classes(labels)

    image_path = test_images["bus"]
    prepared = runtime_engine.prepare_cuda_input(image_path, imgsz=320)
    reference = _python_reference_from_native_tensor_outputs(
        runtime_engine,
        prepared,
        labels,
        conf=TEST_CONF,
        iou=0.45,
        max_det=runtime_engine.metadata.max_det,
        retina_masks=retina_masks,
    )
    actual = runtime_engine.predict(prepared, conf=TEST_CONF, retina_masks=retina_masks)[0]

    _assert_native_postprocess_matches_python_reference(actual, reference)


@pytest.mark.integration
def test_native_postprocessed_results_do_not_alias_across_inferences(
    runtime_engine: YOLOEEngine,
    test_images: dict[str, Path],
) -> None:
    runtime_engine.clear_prompts()
    runtime_engine.set_classes(["bus"])

    first = runtime_engine.predict(test_images["bus"], conf=TEST_CONF, imgsz=320, retina_masks=True)[0]
    assert first.boxes is not None
    first_boxes = first.boxes.data.detach().cpu().clone()
    first_masks = first.masks.data.detach().cpu().clone() if first.masks is not None else None

    runtime_engine.clear_prompts()
    runtime_engine.set_classes(["dog"])
    _ = runtime_engine.predict(test_images["dog"], conf=TEST_CONF, imgsz=320, retina_masks=True)[0]

    torch.testing.assert_close(first.boxes.data.detach().cpu(), first_boxes)
    if first_masks is not None:
        assert first.masks is not None
        assert torch.equal(first.masks.data.detach().cpu(), first_masks)


@pytest.mark.integration
@pytest.mark.parametrize("max_det", [0, -1])
def test_native_postprocess_handles_nonpositive_max_det(
    runtime_engine: YOLOEEngine,
    test_images: dict[str, Path],
    max_det: int,
) -> None:
    runtime_engine.clear_prompts()
    runtime_engine.set_classes(["bus"])
    native_runtime = runtime_engine.native_main_runtime
    assert native_runtime is not None

    item = normalize_source(test_images["bus"], default_prefix="image")[0]
    native_outputs = native_runtime.infer_image_postprocessed(
        item.image,
        320,
        320,
        int(item.image.shape[0]),
        int(item.image.shape[1]),
        TEST_CONF,
        0.45,
        max_det,
        False,
    )
    boxes = from_dlpack(native_outputs["boxes"])

    assert tuple(boxes.shape) == (0, 6)
    if "masks" in native_outputs:
        masks = from_dlpack(native_outputs["masks"])
        assert int(masks.shape[0]) == 0


@pytest.mark.integration
def test_engine_prepared_fast_path_rejects_integer_tensors(
    runtime_engine: YOLOEEngine,
) -> None:
    runtime_engine.clear_prompts()
    runtime_engine.set_classes(["bus"])

    prepared = PreparedTensorInput(
        tensor=torch.zeros((1, 3, 32, 32), dtype=torch.uint8),
        path="tensor0",
    )

    with pytest.raises(ValueError, match="floating point dtype"):
        runtime_engine.predict(prepared)


@pytest.mark.integration
def test_engine_predict_accepts_raw_cpu_tensor_inputs(
    runtime_engine: YOLOEEngine,
    test_images: dict[str, Path],
) -> None:
    runtime_engine.clear_prompts()
    labels = ["bus"]
    runtime_engine.set_classes(labels)

    image_path = test_images["bus"]
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        tensor = torch.from_numpy(np.asarray(rgb).copy()).permute(2, 0, 1).float() / 255.0

    result = runtime_engine.predict(tensor, conf=TEST_CONF, path=image_path.name)[0]

    assert Path(result.path).name == image_path.name
    assert result.names == _expected_names(labels)
    assert result.boxes is not None
    assert result.boxes.data.shape[0] >= 1


@pytest.mark.integration
def test_engine_streams_live_camera_source_frames_with_runtime_camera_prompts(
    runtime_engine: YOLOEEngine,
    usb_camera_source: GStreamerSource,
    camera_labels: list[str],
) -> None:
    runtime_engine.clear_prompts()
    runtime_engine.set_classes(camera_labels)

    results = list(runtime_engine.predict(usb_camera_source, stream=True, conf=TEST_CONF))

    assert len(results) == int(usb_camera_source.max_frames or 0)
    for index, result in enumerate(results):
        _assert_live_result_contract(result, _expected_names(camera_labels), f"{usb_camera_source.prefix}_frame")
        assert Path(result.path).name == f"{usb_camera_source.prefix}_frame{index:06d}"


def _assert_tracked_result_contract(result, image_path: Path, expected_names: dict[int, str]) -> None:
    with Image.open(image_path) as image:
        expected_shape = (image.height, image.width)

    assert Path(result.path).name == image_path.name
    assert result.orig_shape == expected_shape
    assert result.names == expected_names
    assert result.boxes is not None
    assert result.boxes.data.ndim == 2
    assert result.boxes.data.shape[1] == 7
    assert result.boxes.is_track
    assert result.boxes.id is not None
    if result.masks is not None:
        assert tuple(result.masks.orig_shape) == expected_shape
        assert result.masks.data.shape[0] == result.boxes.data.shape[0]


@pytest.mark.integration
def test_engine_track_assigns_stable_ids_on_repeated_frames(
    runtime_engine: YOLOEEngine,
    test_images: dict[str, Path],
) -> None:
    runtime_engine.clear_prompts()
    image_path = test_images["bus"]
    labels = ["bus"]
    runtime_engine.set_classes(labels)

    results = list(runtime_engine.track([image_path, image_path], stream=True, conf=TEST_CONF, tracker="bytetrack"))

    assert len(results) == 2
    for result in results:
        _assert_tracked_result_contract(result, image_path, _expected_names(labels))
        assert result.boxes.data.shape[0] >= 1

    first_ids = {int(track_id) for track_id in results[0].boxes.id.tolist()}
    second_ids = {int(track_id) for track_id in results[1].boxes.id.tolist()}
    assert first_ids & second_ids


@pytest.mark.integration
def test_tracker_session_updates_frames_incrementally(
    runtime_engine: YOLOEEngine,
    test_images: dict[str, Path],
) -> None:
    runtime_engine.clear_prompts()
    image_path = test_images["bus"]
    labels = ["bus"]
    runtime_engine.set_classes(labels)
    session = runtime_engine.create_tracker(tracker="bytetrack", frame_rate=30)

    first = session.update(image_path, conf=TEST_CONF)
    second = session.update(image_path, conf=TEST_CONF)

    _assert_tracked_result_contract(first, image_path, _expected_names(labels))
    _assert_tracked_result_contract(second, image_path, _expected_names(labels))
    assert first.boxes.data.shape[0] >= 1
    assert second.boxes.data.shape[0] >= 1
    assert {int(track_id) for track_id in first.boxes.id.tolist()} & {
        int(track_id) for track_id in second.boxes.id.tolist()
    }


@pytest.mark.integration
def test_tracker_session_updates_gpu_frames_incrementally(
    runtime_engine: YOLOEEngine,
    test_images: dict[str, Path],
) -> None:
    runtime_engine.clear_prompts()
    image_path = test_images["bus"]
    labels = ["bus"]
    runtime_engine.set_classes(labels)
    session = runtime_engine.create_tracker(tracker="bytetrack", frame_rate=30)

    prepared_input = runtime_engine.prepare_cuda_input(image_path)

    first = session.update(prepared_input, conf=TEST_CONF)
    second = session.update(prepared_input, conf=TEST_CONF)

    _assert_tracked_result_contract(first, image_path, _expected_names(labels))
    _assert_tracked_result_contract(second, image_path, _expected_names(labels))
    assert {int(track_id) for track_id in first.boxes.id.tolist()} & {
        int(track_id) for track_id in second.boxes.id.tolist()
    }
