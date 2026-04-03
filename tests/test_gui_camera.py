from __future__ import annotations

from pathlib import Path

import pytest
from yoloe_tensorrt import YOLOEEngine, run_camera_gui

TEST_CONF = 0.1


def _expected_names(labels: list[str]) -> dict[int, str]:
    return {index: label for index, label in enumerate(labels)}


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
@pytest.mark.gui
def test_engine_gui_displays_live_camera_stream_with_overlays(
    gui_available: None,
    runtime_engine: YOLOEEngine,
    camera_labels: list[str],
    camera_source_spec: str,
    camera_window_name: str,
    camera_wait_ms: int,
    camera_max_frames: int | None,
    make_camera_source,
) -> None:
    def _on_result(result, _source_spec: str, labels: list[str]) -> None:
        _assert_live_result_contract(
            result,
            _expected_names(labels),
            "camera_frame",
        )

    displayed_frames = run_camera_gui(
        runtime_engine,
        source_spec=camera_source_spec,
        labels=camera_labels,
        window_title=camera_window_name,
        wait_ms=camera_wait_ms,
        max_frames=camera_max_frames,
        source_prefix="camera",
        make_source=make_camera_source,
        conf=TEST_CONF,
        on_result=_on_result,
    )

    assert displayed_frames >= 1
