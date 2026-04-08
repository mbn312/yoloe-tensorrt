from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image
from yoloe_tensorrt import GStreamerSource, YOLOEEngine

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
