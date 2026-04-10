from __future__ import annotations

import json
import subprocess
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from yoloe_tensorrt import benchmark_cli
from yoloe_tensorrt.benchmark_cli import (
    _benchmark_runner,
    _compare_benchmark_results,
    _cpu_utilization,
    _CpuSnapshot,
    _default_output_name,
    _format_ms,
    _make_created_at,
)
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
    assert "Benchmark YOLOE TensorRT runtime paths" in result.stdout
    assert "--mode" in result.stdout
    assert "--text-prompts" in result.stdout
    assert "--visual-prompts" in result.stdout
    assert "--profile-allocations" in result.stdout
    assert "--track" in result.stdout
    assert "--camera-source" in result.stdout
    assert "--camera-zero-copy" in result.stdout
    assert "--output-dir" in result.stdout
    assert "--compare-to" in result.stdout
    assert "invalid comparison" in result.stdout
    assert "not decoded for text-only benchmarks" in result.stdout


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


def test_benchmark_runner_reports_cpu_metrics(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    snapshots = iter(
        [
            _CpuSnapshot(total_ticks=1000, idle_ticks=500, process_ticks=100, monotonic_s=1.0),
            _CpuSnapshot(total_ticks=1100, idle_ticks=540, process_ticks=105, monotonic_s=1.5),
        ]
    )
    monkeypatch.setattr(benchmark_cli, "_take_cpu_snapshot", lambda: next(snapshots))
    monkeypatch.setattr(benchmark_cli, "_clock_ticks_per_second", lambda: 100)

    result = _benchmark_runner(
        "cpu",
        lambda: SimpleNamespace(speed={}),
        runs=1,
        warmup=0,
        profile_allocations=False,
        allocation_top=0,
    )

    assert result["cpu"]["process_percent"]["mean"] == pytest.approx(10.0)
    assert result["cpu"]["system_percent"]["mean"] == pytest.approx(60.0)
    assert "cpu CPU: process_mean=10.0%" in capsys.readouterr().out


def test_benchmark_runner_runs_setup_before_warmup_and_measured_iterations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    times = iter([1.0, 3.0])
    monkeypatch.setattr(benchmark_cli.time, "perf_counter", lambda: next(times))
    monkeypatch.setattr(benchmark_cli, "_take_cpu_snapshot", lambda: None)

    def _setup() -> None:
        calls.append("setup")

    def _runner():
        calls.append("runner")
        return SimpleNamespace(speed={})

    result = _benchmark_runner(
        "setup",
        _runner,
        runs=1,
        warmup=1,
        profile_allocations=False,
        allocation_top=0,
        setup=_setup,
    )

    assert calls == ["setup", "runner", "setup", "runner"]
    assert result["latency_ms"]["mean"] == pytest.approx(2000.0)


def test_cpu_utilization_rejects_invalid_deltas(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(benchmark_cli, "_clock_ticks_per_second", lambda: 100)

    assert (
        _cpu_utilization(
            _CpuSnapshot(total_ticks=100, idle_ticks=50, process_ticks=10, monotonic_s=1.0),
            _CpuSnapshot(total_ticks=100, idle_ticks=50, process_ticks=11, monotonic_s=1.5),
        )
        is None
    )


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


def test_default_benchmark_output_name_preserves_subsecond_uniqueness() -> None:
    created_at = _make_created_at()

    assert "." in created_at
    assert _default_output_name("2026-04-09T12:00:00.000001Z") != _default_output_name("2026-04-09T12:00:00.000002Z")


def test_benchmark_main_writes_json_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class _Engine:
        native_visual_runtime = None

        def clear_prompts(self) -> None:
            pass

        def set_classes(self, _labels: list[str]) -> None:
            pass

        def prepare_cuda_input(self, item: SourceItem, **_kwargs):
            return item

        def predict_item(self, *_args, **_kwargs):
            return SimpleNamespace(speed={"preprocess": 1.0, "inference": 2.0, "postprocess": 3.0})

        def predict(self, *_args, **_kwargs):
            return [SimpleNamespace(speed={"preprocess": 0.0, "inference": 2.0, "postprocess": 3.0})]

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
            "both",
            "--runs",
            "1",
            "--warmup",
            "0",
            "--output-dir",
            str(tmp_path / "benchmarks"),
            "--output-name",
            "result",
            "--log-level",
            "WARNING",
        ]
    )

    payload = json.loads((tmp_path / "benchmarks" / "result.json").read_text())
    assert result == 0
    assert payload["schema_version"] == 1
    assert payload["config"]["mode"] == "both"
    assert set(payload["results"]) == {"host", "cuda"}
    assert payload["results"]["host"]["latency_ms"]["count"] == 1


def test_benchmark_none_mode_requires_prompt_benchmark(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc_info:
        benchmark_cli.main(
            [
                str(tmp_path / "artifact"),
                str(tmp_path / "image.jpg"),
                "--mode",
                "none",
                "--runs",
                "1",
                "--warmup",
                "0",
                "--no-save",
            ]
        )

    assert exc_info.value.code == 2


def test_benchmark_text_prompts_writes_prompt_only_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _Engine:
        native_visual_runtime = None
        device = "cpu"

        def __init__(self) -> None:
            self.calls: list[tuple[str, tuple[str, ...]]] = []

        def clear_prompts(self) -> None:
            self.calls.append(("clear_prompts", ()))

        def set_classes(self, labels: list[str]) -> None:
            self.calls.append(("set_classes", tuple(labels)))

    engine = _Engine()
    sync_calls = {"count": 0}

    monkeypatch.setattr(benchmark_cli, "_build_engine", lambda *_args, **_kwargs: engine)
    monkeypatch.setattr(
        benchmark_cli,
        "_synchronize_engine_device",
        lambda _engine: sync_calls.__setitem__("count", sync_calls["count"] + 1),
    )
    monkeypatch.setattr(
        "yoloe_tensorrt.source.normalize_source",
        lambda *_args, **_kwargs: pytest.fail("text-only benchmark should not decode the image argument"),
    )

    result = benchmark_cli.main(
        [
            str(tmp_path / "artifact"),
            str(tmp_path / "image.jpg"),
            "--label",
            "bus",
            "--label",
            "person",
            "--mode",
            "none",
            "--text-prompts",
            "--runs",
            "1",
            "--warmup",
            "1",
            "--output-dir",
            str(tmp_path / "benchmarks"),
            "--output-name",
            "text-result",
            "--log-level",
            "WARNING",
        ]
    )

    payload = json.loads((tmp_path / "benchmarks" / "text-result.json").read_text())
    assert result == 0
    assert set(payload["results"]) == {"text-set-classes"}
    assert payload["config"]["text_prompts"] is True
    assert payload["config"]["mode"] == "none"
    assert payload["results"]["text-set-classes"]["latency_ms"]["count"] == 1
    assert engine.calls == [
        ("clear_prompts", ()),
        ("set_classes", ("bus", "person")),
        ("clear_prompts", ()),
        ("set_classes", ("bus", "person")),
        ("clear_prompts", ()),
        ("set_classes", ("bus", "person")),
    ]
    assert sync_calls["count"] == 3


def test_benchmark_none_mode_allows_visual_prompt_only_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _Engine:
        native_visual_runtime = object()
        visual_runtime = None
        device = "cpu"

        def __init__(self) -> None:
            self.visual_calls = 0
            self.text_calls = 0

        def clear_prompts(self) -> None:
            pass

        def set_classes(self, _labels: list[str]) -> None:
            self.text_calls += 1

        def set_visual_prompts(self, *_args, **_kwargs) -> None:
            self.visual_calls += 1

    image = np.zeros((4, 4, 3), dtype=np.uint8)
    engine = _Engine()

    monkeypatch.setattr(benchmark_cli, "_build_engine", lambda *_args, **_kwargs: engine)
    monkeypatch.setattr(benchmark_cli, "_synchronize_engine_device", lambda _engine: None)
    monkeypatch.setattr(
        "yoloe_tensorrt.source.normalize_source",
        lambda *_args, **_kwargs: [SourceItem(image=image, path="benchmark0")],
    )

    result = benchmark_cli.main(
        [
            str(tmp_path / "artifact"),
            str(tmp_path / "image.jpg"),
            "--mode",
            "none",
            "--visual-prompts",
            "bbox",
            "--visual-runtime",
            "native",
            "--runs",
            "1",
            "--warmup",
            "0",
            "--output-dir",
            str(tmp_path / "benchmarks"),
            "--output-name",
            "visual-result",
            "--log-level",
            "WARNING",
        ]
    )

    payload = json.loads((tmp_path / "benchmarks" / "visual-result.json").read_text())
    assert result == 0
    assert set(payload["results"]) == {"visual-native-bbox"}
    assert payload["config"]["mode"] == "none"
    assert payload["config"]["visual_prompts"] == "bbox"
    assert engine.visual_calls == 1
    assert engine.text_calls == 0


def test_benchmark_none_mode_rejects_visual_prompt_without_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _Engine:
        native_visual_runtime = None
        visual_runtime = None
        device = "cpu"

        def __init__(self) -> None:
            self.text_calls = 0

        def clear_prompts(self) -> None:
            pass

        def set_classes(self, _labels: list[str]) -> None:
            self.text_calls += 1

    engine = _Engine()

    monkeypatch.setattr(benchmark_cli, "_build_engine", lambda *_args, **_kwargs: engine)
    monkeypatch.setattr(benchmark_cli, "_synchronize_engine_device", lambda _engine: None)

    with pytest.raises(SystemExit) as exc_info:
        benchmark_cli.main(
            [
                str(tmp_path / "artifact"),
                str(tmp_path / "image.jpg"),
                "--mode",
                "none",
                "--visual-prompts",
                "bbox",
                "--visual-runtime",
                "native",
                "--runs",
                "1",
                "--warmup",
                "0",
                "--output-dir",
                str(tmp_path / "benchmarks"),
                "--output-name",
                "empty-result",
                "--log-level",
                "WARNING",
            ]
        )

    assert exc_info.value.code == 2
    assert not (tmp_path / "benchmarks" / "empty-result.json").exists()
    assert engine.text_calls == 0


def test_benchmark_all_mode_writes_host_cuda_and_camera_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _Engine:
        native_visual_runtime = None

        def __init__(self) -> None:
            self.predict_item_paths: list[str] = []
            self.prepared = False

        def clear_prompts(self) -> None:
            pass

        def set_classes(self, _labels: list[str]) -> None:
            pass

        @property
        def _main_fp16(self) -> bool:
            return False

        def prepare_cuda_input(self, item: SourceItem, **_kwargs):
            self.prepared = True
            return item

        def predict_item(self, item: SourceItem, *_args, **_kwargs):
            self.predict_item_paths.append(item.path)
            return SimpleNamespace(speed={})

        def predict(self, *_args, **_kwargs):
            return [SimpleNamespace(speed={})]

    image = np.zeros((4, 4, 3), dtype=np.uint8)
    source_item = SourceItem(image=image, path="benchmark0")
    camera_item = SourceItem(image=image, path="camera0")
    engine = _Engine()
    normalize_calls = {"count": 0}

    def _normalize_source(*_args, **_kwargs):
        normalize_calls["count"] += 1
        return [source_item]

    monkeypatch.setattr(benchmark_cli, "_build_engine", lambda *_args, **_kwargs: engine)
    monkeypatch.setattr(benchmark_cli, "_camera_source_from_spec", lambda *_args, **_kwargs: [camera_item])
    monkeypatch.setattr("yoloe_tensorrt.source.normalize_source", _normalize_source)

    result = benchmark_cli.main(
        [
            str(tmp_path / "artifact"),
            str(tmp_path / "image.jpg"),
            "--mode",
            "all",
            "--camera-source",
            "dummy://ball",
            "--runs",
            "1",
            "--warmup",
            "0",
            "--output-dir",
            str(tmp_path / "benchmarks"),
            "--output-name",
            "all-results",
            "--log-level",
            "WARNING",
        ]
    )

    payload = json.loads((tmp_path / "benchmarks" / "all-results.json").read_text())
    assert result == 0
    assert set(payload["results"]) == {"host", "cuda", "camera"}
    assert engine.prepared
    assert engine.predict_item_paths == ["benchmark0", "camera0"]
    assert normalize_calls["count"] == 1


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
            "--no-save",
            "--log-level",
            "WARNING",
        ]
    )

    assert result == 0


def test_benchmark_camera_mode_requires_camera_source(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc_info:
        benchmark_cli.main(
            [
                str(tmp_path / "artifact"),
                str(tmp_path / "image.jpg"),
                "--mode",
                "camera",
                "--runs",
                "1",
                "--warmup",
                "0",
                "--no-save",
            ]
        )

    assert exc_info.value.code == 2


def test_benchmark_camera_mode_uses_finite_source(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class _Engine:
        native_visual_runtime = None

        def __init__(self) -> None:
            self.calls = 0

        def clear_prompts(self) -> None:
            pass

        def set_classes(self, _labels: list[str]) -> None:
            pass

        @property
        def _main_fp16(self) -> bool:
            return True

        def predict_item(self, *_args, **_kwargs):
            self.calls += 1
            return SimpleNamespace(speed={"preprocess": 1.0, "inference": 2.0, "postprocess": 3.0})

    image = np.zeros((4, 4, 3), dtype=np.uint8)
    engine = _Engine()
    source_kwargs = {}

    def _camera_source_from_spec(*_args, **kwargs):
        source_kwargs.update(kwargs)
        return [
            SourceItem(image=image, path="camera0"),
            SourceItem(image=image, path="camera1"),
            SourceItem(image=image, path="camera2"),
        ]

    monkeypatch.setattr(benchmark_cli, "_build_engine", lambda *_args, **_kwargs: engine)
    monkeypatch.setattr(benchmark_cli, "_camera_source_from_spec", _camera_source_from_spec)
    monkeypatch.setattr(
        "yoloe_tensorrt.source.normalize_source",
        lambda *_args, **_kwargs: pytest.fail("camera-only benchmark should not decode the image argument"),
    )

    result = benchmark_cli.main(
        [
            str(tmp_path / "artifact"),
            str(tmp_path / "image.jpg"),
            "--mode",
            "camera",
            "--camera-source",
            "dummy://ball",
            "--camera-zero-copy",
            "auto",
            "--runs",
            "1",
            "--warmup",
            "1",
            "--no-save",
            "--log-level",
            "WARNING",
        ]
    )

    assert result == 0
    assert engine.calls == 2
    assert source_kwargs["max_frames"] == 2
    assert source_kwargs["target_imgsz"] == (640, 640)
    assert source_kwargs["zero_copy"] is None
    assert source_kwargs["fp16"] is True


def test_benchmark_camera_mode_closes_source_on_early_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _Engine:
        native_visual_runtime = None

        def clear_prompts(self) -> None:
            pass

        def set_classes(self, _labels: list[str]) -> None:
            pass

        @property
        def _main_fp16(self) -> bool:
            return False

        def predict_item(self, *_args, **_kwargs):
            return SimpleNamespace(speed={})

    class _CameraIterator:
        def __init__(self) -> None:
            self.closed = False
            self.frames = iter([SourceItem(image=np.zeros((4, 4, 3), dtype=np.uint8), path="camera0")])

        def __iter__(self):
            return self

        def __next__(self):
            return next(self.frames)

        def close(self) -> None:
            self.closed = True

    camera_source = _CameraIterator()

    monkeypatch.setattr(benchmark_cli, "_build_engine", lambda *_args, **_kwargs: _Engine())
    monkeypatch.setattr(benchmark_cli, "_camera_source_from_spec", lambda *_args, **_kwargs: camera_source)

    with pytest.raises(RuntimeError, match="Camera benchmark source ended"):
        benchmark_cli.main(
            [
                str(tmp_path / "artifact"),
                str(tmp_path / "image.jpg"),
                "--mode",
                "camera",
                "--camera-source",
                "dummy://ball",
                "--runs",
                "2",
                "--warmup",
                "0",
                "--no-save",
                "--log-level",
                "WARNING",
            ]
        )

    assert camera_source.closed


def test_benchmark_compare_detects_regressions() -> None:
    baseline = {
        "schema_version": 1,
        "results": {
            "host": {
                "latency_ms": {"median": 10.0, "p95": 20.0},
                "fps": 100.0,
                "cpu": {
                    "process_percent": {"mean": 20.0},
                    "system_percent": {"mean": 30.0},
                },
            }
        },
    }
    current = {
        "schema_version": 1,
        "results": {
            "host": {
                "latency_ms": {"median": 12.0, "p95": 25.0},
                "fps": 80.0,
                "cpu": {
                    "process_percent": {"mean": 40.0},
                    "system_percent": {"mean": 60.0},
                },
            }
        },
    }
    thresholds = {
        "median_latency_percent": 10.0,
        "p95_latency_percent": 10.0,
        "fps_percent": 10.0,
        "process_cpu_percentage_points": 15.0,
        "system_cpu_percentage_points": 15.0,
    }

    comparison = _compare_benchmark_results(current, baseline, thresholds)

    assert comparison["status"] == "regressed"
    assert comparison["regression_count"] == 5


def test_benchmark_compare_reports_no_overlap() -> None:
    thresholds = {
        "median_latency_percent": 10.0,
        "p95_latency_percent": 10.0,
        "fps_percent": 10.0,
        "process_cpu_percentage_points": 15.0,
        "system_cpu_percentage_points": 15.0,
    }

    comparison = _compare_benchmark_results(
        {"schema_version": 1, "results": {"host": {}}},
        {"schema_version": 1, "results": {"cuda": {}}},
        thresholds,
    )

    assert comparison["status"] == "no_overlap"
    assert comparison["current_modes"] == ["host"]
    assert comparison["baseline_modes"] == ["cuda"]


def test_benchmark_compare_reports_no_comparable_metrics() -> None:
    thresholds = {
        "median_latency_percent": 10.0,
        "p95_latency_percent": 10.0,
        "fps_percent": 10.0,
        "process_cpu_percentage_points": 15.0,
        "system_cpu_percentage_points": 15.0,
    }

    comparison = _compare_benchmark_results(
        {"schema_version": 1, "results": {"host": {}}},
        {"schema_version": 1, "results": {"host": {}}},
        thresholds,
    )

    assert comparison["status"] == "no_comparable_metrics"
    assert comparison["unchecked_modes"] == ["host"]


def test_benchmark_compare_only_fails_when_requested(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class _Engine:
        native_visual_runtime = None

        def clear_prompts(self) -> None:
            pass

        def set_classes(self, _labels: list[str]) -> None:
            pass

        def predict_item(self, *_args, **_kwargs):
            return SimpleNamespace(speed={})

    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "results": {
                    "host": {
                        "latency_ms": {"median": 10.0, "p95": 10.0},
                        "fps": 100.0,
                        "cpu": {
                            "process_percent": {"mean": 10.0},
                            "system_percent": {"mean": 10.0},
                        },
                    }
                },
            }
        )
    )
    current_result = {
        "runs": 1,
        "warmup": 0,
        "latency_ms": {"median": 20.0, "p95": 20.0},
        "fps": 50.0,
        "speed_ms": {},
        "cpu": {
            "process_percent": {"mean": 40.0},
            "system_percent": {"mean": 40.0},
        },
        "allocations": None,
    }
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    monkeypatch.setattr(benchmark_cli, "_build_engine", lambda *_args, **_kwargs: _Engine())
    monkeypatch.setattr(benchmark_cli, "_benchmark_runner", lambda *_args, **_kwargs: current_result)
    monkeypatch.setattr(
        "yoloe_tensorrt.source.normalize_source",
        lambda *_args, **_kwargs: [SourceItem(image=image, path="benchmark0")],
    )

    base_args = [
        str(tmp_path / "artifact"),
        str(tmp_path / "image.jpg"),
        "--mode",
        "host",
        "--runs",
        "1",
        "--warmup",
        "0",
        "--compare-to",
        str(baseline_path),
        "--no-save",
        "--log-level",
        "WARNING",
    ]

    assert benchmark_cli.main(base_args) == 0
    assert benchmark_cli.main([*base_args, "--fail-on-regression"]) == 1


def test_benchmark_compare_no_metrics_fails_regression_gate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _Engine:
        native_visual_runtime = None

        def clear_prompts(self) -> None:
            pass

        def set_classes(self, _labels: list[str]) -> None:
            pass

        def predict_item(self, *_args, **_kwargs):
            return SimpleNamespace(speed={})

    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps({"schema_version": 1, "results": {"host": {}}}))
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    monkeypatch.setattr(benchmark_cli, "_build_engine", lambda *_args, **_kwargs: _Engine())
    monkeypatch.setattr(
        benchmark_cli,
        "_benchmark_runner",
        lambda *_args, **_kwargs: {
            "runs": 1,
            "warmup": 0,
            "latency_ms": {"median": 10.0, "p95": 10.0},
            "fps": 100.0,
            "speed_ms": {},
            "cpu": {
                "process_percent": {"mean": 10.0},
                "system_percent": {"mean": 10.0},
            },
            "allocations": None,
        },
    )
    monkeypatch.setattr(
        "yoloe_tensorrt.source.normalize_source",
        lambda *_args, **_kwargs: [SourceItem(image=image, path="benchmark0")],
    )
    base_args = [
        str(tmp_path / "artifact"),
        str(tmp_path / "image.jpg"),
        "--mode",
        "host",
        "--runs",
        "1",
        "--warmup",
        "0",
        "--compare-to",
        str(baseline_path),
        "--no-save",
        "--log-level",
        "WARNING",
    ]

    assert benchmark_cli.main(base_args) == 0
    assert benchmark_cli.main([*base_args, "--fail-on-regression"]) == 1


def test_benchmark_compare_no_overlap_fails_regression_gate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _Engine:
        native_visual_runtime = None

        def clear_prompts(self) -> None:
            pass

        def set_classes(self, _labels: list[str]) -> None:
            pass

        def predict_item(self, *_args, **_kwargs):
            return SimpleNamespace(speed={})

    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "results": {
                    "cuda": {
                        "latency_ms": {"median": 10.0, "p95": 10.0},
                        "fps": 100.0,
                        "cpu": {
                            "process_percent": {"mean": 10.0},
                            "system_percent": {"mean": 10.0},
                        },
                    }
                },
            }
        )
    )
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    monkeypatch.setattr(benchmark_cli, "_build_engine", lambda *_args, **_kwargs: _Engine())
    monkeypatch.setattr(
        benchmark_cli,
        "_benchmark_runner",
        lambda *_args, **_kwargs: {
            "runs": 1,
            "warmup": 0,
            "latency_ms": {"median": 10.0, "p95": 10.0},
            "fps": 100.0,
            "speed_ms": {},
            "cpu": {
                "process_percent": {"mean": 10.0},
                "system_percent": {"mean": 10.0},
            },
            "allocations": None,
        },
    )
    monkeypatch.setattr(
        "yoloe_tensorrt.source.normalize_source",
        lambda *_args, **_kwargs: [SourceItem(image=image, path="benchmark0")],
    )
    base_args = [
        str(tmp_path / "artifact"),
        str(tmp_path / "image.jpg"),
        "--mode",
        "host",
        "--runs",
        "1",
        "--warmup",
        "0",
        "--compare-to",
        str(baseline_path),
        "--no-save",
        "--log-level",
        "WARNING",
    ]

    assert benchmark_cli.main(base_args) == 0
    assert benchmark_cli.main([*base_args, "--fail-on-regression"]) == 1


def test_benchmark_compare_rejects_malformed_baseline_before_building_engine(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps({"results": {}}))
    monkeypatch.setattr(
        benchmark_cli,
        "_build_engine",
        lambda *_args, **_kwargs: pytest.fail("malformed baseline should fail before engine build"),
    )

    with pytest.raises(SystemExit) as exc_info:
        benchmark_cli.main(
            [
                str(tmp_path / "artifact"),
                str(tmp_path / "image.jpg"),
                "--mode",
                "host",
                "--compare-to",
                str(baseline_path),
                "--no-save",
            ]
        )

    assert exc_info.value.code == 2
