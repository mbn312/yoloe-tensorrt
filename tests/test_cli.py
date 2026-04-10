from __future__ import annotations

import subprocess
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from yoloe_tensorrt import benchmark_cli
from yoloe_tensorrt.benchmark_cli import _benchmark_runner, _format_ms
from yoloe_tensorrt.source import SourceItem

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_module_entrypoint_displays_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "yoloe_tensorrt", "--help"],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0
    assert "CLI entry points for yoloe-tensorrt." in result.stdout
    assert "camera-gui" in result.stdout
    assert "export" in result.stdout
    assert "benchmark" in result.stdout


def test_export_cli_displays_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "yoloe_tensorrt", "export", "--help"],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0
    assert "Export a YOLOE checkpoint into a yoloe-tensorrt artifact bundle." in result.stdout


def test_camera_gui_cli_displays_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "yoloe_tensorrt", "camera-gui", "--help"],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0
    assert "Launch the YOLOE TensorRT live camera GUI." in result.stdout
    assert "--track" in result.stdout
    assert "--tracker" in result.stdout


def test_benchmark_cli_displays_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "yoloe_tensorrt", "benchmark", "--help"],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0
    assert "Benchmark host-image and CUDA-tensor inference" in result.stdout
    assert "--mode" in result.stdout
    assert "--visual-prompts" in result.stdout
    assert "--profile-allocations" in result.stdout
    assert "--track" in result.stdout


def test_benchmark_runner_profiles_allocations(capsys) -> None:
    def _runner():
        return SimpleNamespace(speed={"preprocess": 1.0, "inference": 2.0, "postprocess": 3.0})

    _benchmark_runner(
        "dummy",
        _runner,
        runs=1,
        warmup=0,
        profile_allocations=True,
        allocation_top=1,
    )

    output = capsys.readouterr().out
    assert "dummy: runs=1" in output
    assert "dummy Python retained allocations:" in output


def test_benchmark_runner_clears_final_result_before_allocation_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Snapshot:
        def compare_to(self, _before, _group_by):
            return []

    class _Result:
        __slots__ = ("speed", "__weakref__")

        def __init__(self) -> None:
            self.speed = {"preprocess": 1.0, "inference": 2.0, "postprocess": 3.0}

    result_refs: list[weakref.ReferenceType[_Result]] = []
    snapshot_result_released: list[bool] = []

    def _runner():
        result = _Result()
        result_refs.append(weakref.ref(result))
        return result

    def _take_snapshot():
        snapshot_result_released.append(not result_refs or result_refs[-1]() is None)
        return _Snapshot()

    monkeypatch.setattr(benchmark_cli.tracemalloc, "start", lambda: None)
    monkeypatch.setattr(benchmark_cli.tracemalloc, "get_traced_memory", lambda: (0, 0))
    monkeypatch.setattr(benchmark_cli.tracemalloc, "take_snapshot", _take_snapshot)
    monkeypatch.setattr(benchmark_cli.tracemalloc, "stop", lambda: None)

    _benchmark_runner(
        "dummy",
        _runner,
        runs=1,
        warmup=0,
        profile_allocations=True,
        allocation_top=1,
    )

    assert snapshot_result_released == [True, True]


def test_benchmark_latency_summary_uses_nearest_rank_tail_percentiles() -> None:
    summary = _format_ms([1.0, 2.0, 3.0, 4.0, 100.0])

    assert "p95=100.00ms" in summary
    assert "p99=100.00ms" in summary


def test_benchmark_host_only_does_not_prepare_cuda_input(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class _Engine:
        native_visual_runtime = None

        def clear_prompts(self) -> None:
            pass

        def set_classes(self, _labels: list[str]) -> None:
            pass

        def prepare_cuda_input(self, *_args, **_kwargs):
            raise AssertionError("host-only benchmark should not prepare CUDA input")

        def predict_item(self, *_args, **_kwargs):
            return SimpleNamespace(speed={"preprocess": 1.0, "inference": 2.0, "postprocess": 3.0})

    image = np.zeros((4, 4, 3), dtype=np.uint8)
    source_item = SourceItem(image=image, path="benchmark0")

    monkeypatch.setattr(benchmark_cli, "_build_engine", lambda *_args, **_kwargs: _Engine())
    monkeypatch.setattr(
        "yoloe_tensorrt.source.normalize_source",
        lambda *_args, **_kwargs: [source_item],
    )

    result = benchmark_cli.main(
        [
            str(tmp_path / "artifact"),
            str(tmp_path / "image.jpg"),
            "--mode",
            "host",
            "--runs",
            "1",
            "--warmup",
            "0",
            "--log-level",
            "WARNING",
        ]
    )

    assert result == 0
