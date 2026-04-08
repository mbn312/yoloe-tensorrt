from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from ultralytics.engine.results import Results
from yoloe_tensorrt.source import SourceItem
from yoloe_tensorrt.tracking import (
    YOLOETrackerSession,
    _apply_tracking_to_result,
    resolve_tracker_config,
)


def _make_result() -> Results:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    boxes = torch.tensor(
        [
            [1, 2, 10, 11, 0.9, 0],
            [3, 4, 12, 13, 0.8, 1],
        ],
        dtype=torch.float32,
    )
    masks = torch.rand(2, 8, 8)
    return Results(image, path="frame0", names={0: "pen", 1: "marker"}, boxes=boxes, masks=masks)


class _FakeTrackerBackend:
    def __init__(self, track_outputs: list[np.ndarray]) -> None:
        self.track_outputs = list(track_outputs)
        self.update_calls = 0

    def update(self, _detections, _image):
        index = min(self.update_calls, len(self.track_outputs) - 1)
        self.update_calls += 1
        return self.track_outputs[index]


class _FakeEngine:
    def __init__(self, results: list[Results]) -> None:
        self._prompt_generation = 0
        self._results = list(results)
        self.predict_calls = 0

    def predict_item(self, **_kwargs) -> Results:
        index = min(self.predict_calls, len(self._results) - 1)
        self.predict_calls += 1
        return self._results[index]


def test_resolve_tracker_config_supports_dict_and_yaml_overrides(tmp_path: Path) -> None:
    config = resolve_tracker_config("bytetrack", {"track_buffer": 42})
    assert config["tracker_type"] == "bytetrack"
    assert config["track_buffer"] == 42

    config_path = tmp_path / "botsort.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "tracker_type": "botsort",
                "match_thresh": 0.7,
                "with_reid": False,
            }
        ),
        encoding="utf-8",
    )
    file_config = resolve_tracker_config("botsort", config_path)
    assert file_config["tracker_type"] == "botsort"
    assert file_config["match_thresh"] == pytest.approx(0.7)


def test_resolve_tracker_config_rejects_botsort_reid() -> None:
    with pytest.raises(ValueError, match="ReID is not supported"):
        resolve_tracker_config("botsort", {"with_reid": True})


def test_apply_tracking_to_result_preserves_masks_and_assigns_ids() -> None:
    result = _make_result()
    tracker = _FakeTrackerBackend(
        [
            np.asarray(
                [
                    [3.0, 4.0, 12.0, 13.0, 7.0, 0.8, 1.0, 1.0],
                ],
                dtype=np.float32,
            )
        ]
    )

    tracked = _apply_tracking_to_result(result, tracker)

    assert tracked.boxes is not None
    assert tracked.boxes.data.shape == (1, 7)
    assert tracked.boxes.is_track
    assert tracked.boxes.id is not None
    assert tracked.boxes.id.tolist() == [7.0]
    assert tracked.masks is not None
    assert tracked.masks.data.shape[0] == 1


def test_tracker_session_resets_on_prompt_generation_and_source_change(monkeypatch: pytest.MonkeyPatch) -> None:
    backends: list[_FakeTrackerBackend] = []

    def _fake_build_backend(_tracker: str, _config: dict, _frame_rate: int) -> _FakeTrackerBackend:
        backend = _FakeTrackerBackend(
            [
                np.asarray([[1.0, 2.0, 10.0, 11.0, 1.0, 0.9, 0.0, 0.0]], dtype=np.float32),
            ]
        )
        backends.append(backend)
        return backend

    monkeypatch.setattr("yoloe_tensorrt.tracking._build_tracker_backend", _fake_build_backend)
    engine = _FakeEngine([_make_result(), _make_result(), _make_result()])
    session = YOLOETrackerSession(engine)

    frame = SourceItem(image=np.zeros((32, 32, 3), dtype=np.uint8), path="frame0")
    first = session.update(frame, source_key="cam0")
    assert first.boxes is not None
    assert len(backends) == 1
    assert backends[0].update_calls == 1

    engine._prompt_generation += 1
    session.update(frame, source_key="cam0")
    assert len(backends) == 2
    assert backends[1].update_calls == 1

    session.update(frame, source_key="cam1")
    assert len(backends) == 3
    assert backends[2].update_calls == 1
