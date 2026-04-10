from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import platform
import statistics
import sys
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ._version import __version__
from .logging_utils import configure_logging


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark YOLOE TensorRT runtime paths, camera ingest, prompt updates, and JSON regression output."
        )
    )
    parser.add_argument("artifact_dir", help="Artifact bundle directory to benchmark.")
    parser.add_argument(
        "image",
        help=(
            "Image path for inference and visual-prompt benchmarks; accepted but not decoded for text-only benchmarks."
        ),
    )
    parser.add_argument(
        "--label",
        action="append",
        dest="labels",
        default=None,
        help="Active runtime label. Repeat to add more than one. Defaults to 'bus'.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size. Defaults to 640.")
    parser.add_argument("--runs", type=int, default=20, help="Measured iterations per mode. Defaults to 20.")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations per mode. Defaults to 5.")
    parser.add_argument(
        "--mode",
        choices=("none", "host", "cuda", "camera", "both", "all"),
        default="both",
        help="Benchmark no inference path, host-image path, CUDA fast path, camera path, host+CUDA, or all paths.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold.")
    parser.add_argument("--device", default="cuda:0", help="Torch/TensorRT device. Defaults to cuda:0.")
    parser.add_argument(
        "--track",
        action="store_true",
        help="Benchmark tracker session update() instead of raw prediction.",
    )
    parser.add_argument(
        "--profile-allocations",
        action="store_true",
        help="Profile retained Python allocations and traced peak memory across the measured loop.",
    )
    parser.add_argument(
        "--allocation-top",
        type=int,
        default=5,
        help="Number of allocation hot spots to print when --profile-allocations is enabled.",
    )
    parser.add_argument("--output-dir", default="outputs/benchmarks", help="Directory for JSON benchmark output.")
    parser.add_argument("--output-name", default=None, help="Output JSON filename. Defaults to a timestamped name.")
    parser.add_argument("--no-save", action="store_true", help="Do not write a JSON benchmark result file.")
    parser.add_argument(
        "--compare-to", type=Path, default=None, help="Compare current results to a prior JSON benchmark."
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit non-zero when --compare-to detects a regression or invalid comparison.",
    )
    parser.add_argument(
        "--max-p95-latency-regression",
        type=float,
        default=10.0,
        help="Maximum allowed p95 latency regression percentage for --compare-to.",
    )
    parser.add_argument(
        "--max-median-latency-regression",
        type=float,
        default=10.0,
        help="Maximum allowed median latency regression percentage for --compare-to.",
    )
    parser.add_argument(
        "--max-process-cpu-regression",
        type=float,
        default=15.0,
        help="Maximum allowed process CPU regression in percentage points for --compare-to.",
    )
    parser.add_argument(
        "--max-system-cpu-regression",
        type=float,
        default=15.0,
        help="Maximum allowed system CPU regression in percentage points for --compare-to.",
    )
    parser.add_argument(
        "--max-fps-regression",
        type=float,
        default=10.0,
        help="Maximum allowed FPS regression percentage for --compare-to.",
    )
    parser.add_argument("--camera-source", default=None, help="Camera source for --mode camera or --mode all.")
    parser.add_argument("--camera-width", type=int, default=None, help="Requested camera width.")
    parser.add_argument("--camera-height", type=int, default=None, help="Requested camera height.")
    parser.add_argument("--camera-fps", type=int, default=None, help="Requested camera frame rate.")
    parser.add_argument("--camera-timeout", type=float, default=5.0, help="Camera frame timeout in seconds.")
    parser.add_argument(
        "--camera-zero-copy",
        choices=("auto", "required", "off"),
        default="auto",
        help="Camera zero-copy policy for Jetson sources.",
    )
    parser.add_argument(
        "--text-prompts",
        action="store_true",
        help="Also benchmark set_classes() text-prompt updates.",
    )
    parser.add_argument(
        "--visual-prompts",
        choices=("none", "bbox", "mask", "both"),
        default="none",
        help="Also benchmark set_visual_prompts() using bbox prompts, mask prompts, or both.",
    )
    parser.add_argument(
        "--visual-runtime",
        choices=("native", "python", "both"),
        default="both",
        help="Which visual-prompt runtime to benchmark when --visual-prompts is enabled.",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level for the benchmark command.")
    return parser


@dataclass(frozen=True)
class _CpuSnapshot:
    total_ticks: int
    idle_ticks: int
    process_ticks: int
    monotonic_s: float


_PROC_STAT_PATH = Path("/proc/stat")
_PROC_SELF_STAT_PATH = Path("/proc/self/stat")


def _format_ms(values: list[float]) -> str:
    sorted_values = sorted(values)
    stdev = statistics.pstdev(values) if len(values) > 1 else 0.0
    p95 = _nearest_rank_percentile(sorted_values, 0.95)
    p99 = _nearest_rank_percentile(sorted_values, 0.99)
    return (
        f"mean={statistics.fmean(values):.2f}ms "
        f"median={statistics.median(values):.2f}ms "
        f"p95={p95:.2f}ms "
        f"p99={p99:.2f}ms "
        f"stdev={stdev:.2f}ms"
    )


def _nearest_rank_percentile(sorted_values: list[float], percentile: float) -> float:
    index = min(len(sorted_values) - 1, max(0, math.ceil(len(sorted_values) * percentile) - 1))
    return sorted_values[index]


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "p99": None,
            "stdev": None,
            "min": None,
            "max": None,
        }
    sorted_values = sorted(values)
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": _nearest_rank_percentile(sorted_values, 0.95),
        "p99": _nearest_rank_percentile(sorted_values, 0.99),
        "stdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": sorted_values[0],
        "max": sorted_values[-1],
    }


def _summarize_result_speeds(
    name: str,
    preprocess: list[float],
    inference: list[float],
    postprocess: list[float],
) -> None:
    if not preprocess:
        return
    print(
        f"{name} speed breakdown: preprocess={statistics.fmean(preprocess):.2f}ms "
        f"inference={statistics.fmean(inference):.2f}ms postprocess={statistics.fmean(postprocess):.2f}ms"
    )


def _print_cpu_summary(name: str, process_cpu: list[float], system_cpu: list[float]) -> None:
    if not process_cpu or not system_cpu:
        print(f"{name} CPU: unavailable")
        return
    process_stats = _stats(process_cpu)
    system_stats = _stats(system_cpu)
    print(
        f"{name} CPU: process_mean={float(process_stats['mean']):.1f}% "
        f"process_median={float(process_stats['median']):.1f}% "
        f"process_p95={float(process_stats['p95']):.1f}% "
        f"process_max={float(process_stats['max']):.1f}% "
        f"system_mean={float(system_stats['mean']):.1f}% "
        f"system_median={float(system_stats['median']):.1f}% "
        f"system_p95={float(system_stats['p95']):.1f}% "
        f"system_max={float(system_stats['max']):.1f}%"
    )


def _format_bytes(value: int) -> str:
    if value < 1024:
        return f"{value}B"
    if value < 1024 * 1024:
        return f"{value / 1024.0:.1f}KiB"
    return f"{value / (1024.0 * 1024.0):.1f}MiB"


def _print_allocation_summary(
    name: str,
    before: tracemalloc.Snapshot,
    after: tracemalloc.Snapshot,
    peak_bytes: int,
    top_n: int,
) -> dict[str, Any]:
    stats = [stat for stat in after.compare_to(before, "lineno") if stat.size_diff > 0 or stat.count_diff > 0]
    retained_bytes = sum(max(0, stat.size_diff) for stat in stats)
    retained_count = sum(max(0, stat.count_diff) for stat in stats)
    print(
        f"{name} Python retained allocations: retained={_format_bytes(retained_bytes)} "
        f"count={retained_count} peak_traced={_format_bytes(peak_bytes)}"
    )
    hotspots = []
    for index, stat in enumerate(stats[: max(0, int(top_n))], start=1):
        frame = stat.traceback[0]
        hotspot = {
            "filename": frame.filename,
            "line": int(frame.lineno),
            "retained_bytes": max(0, int(stat.size_diff)),
            "retained_count": max(0, int(stat.count_diff)),
        }
        hotspots.append(hotspot)
        print(
            f"{name} retained allocation hotspot {index}: {frame.filename}:{frame.lineno} "
            f"size={_format_bytes(max(0, stat.size_diff))} count={max(0, stat.count_diff)}"
        )
    return {
        "retained_bytes": int(retained_bytes),
        "retained_count": int(retained_count),
        "peak_traced_bytes": int(peak_bytes),
        "hotspots": hotspots,
    }


def _read_system_cpu_ticks(path: Path = _PROC_STAT_PATH) -> tuple[int, int]:
    first_line = path.read_text().splitlines()[0]
    parts = first_line.split()
    if not parts or parts[0] != "cpu":
        raise ValueError(f"Unable to parse system CPU stats from {path}")
    ticks = [int(value) for value in parts[1:]]
    total_ticks = sum(ticks)
    idle_ticks = ticks[3] + (ticks[4] if len(ticks) > 4 else 0)
    return total_ticks, idle_ticks


def _read_process_cpu_ticks(path: Path = _PROC_SELF_STAT_PATH) -> int:
    text = path.read_text()
    suffix = text[text.rfind(")") + 2 :].split()
    if len(suffix) < 13:
        raise ValueError(f"Unable to parse process CPU stats from {path}")
    return int(suffix[11]) + int(suffix[12])


def _clock_ticks_per_second() -> int:
    try:
        return int(os.sysconf("SC_CLK_TCK"))
    except (AttributeError, OSError, ValueError):
        return 100


def _take_cpu_snapshot() -> _CpuSnapshot | None:
    try:
        total_ticks, idle_ticks = _read_system_cpu_ticks()
        process_ticks = _read_process_cpu_ticks()
    except (OSError, ValueError, IndexError):
        return None
    return _CpuSnapshot(
        total_ticks=total_ticks,
        idle_ticks=idle_ticks,
        process_ticks=process_ticks,
        monotonic_s=time.perf_counter(),
    )


def _cpu_utilization(before: _CpuSnapshot | None, after: _CpuSnapshot | None) -> tuple[float, float] | None:
    if before is None or after is None:
        return None
    elapsed_s = after.monotonic_s - before.monotonic_s
    total_delta = after.total_ticks - before.total_ticks
    idle_delta = after.idle_ticks - before.idle_ticks
    process_delta = after.process_ticks - before.process_ticks
    if elapsed_s <= 0.0 or total_delta <= 0 or process_delta < 0:
        return None
    process_cpu = (process_delta / _clock_ticks_per_second()) / elapsed_s * 100.0
    system_cpu = max(0.0, min(100.0, (total_delta - idle_delta) / total_delta * 100.0))
    return process_cpu, system_cpu


def _benchmark_runner(
    name: str,
    runner: Callable[[], object],
    *,
    runs: int,
    warmup: int,
    profile_allocations: bool,
    allocation_top: int,
    setup: Callable[[], None] | None = None,
) -> dict[str, Any]:
    for _ in range(warmup):
        if setup is not None:
            setup()
        runner()

    latencies_ms = [0.0] * runs
    preprocess = [0.0] * runs
    inference = [0.0] * runs
    postprocess = [0.0] * runs
    cpu_before = None
    cpu_after = None

    before_snapshot = None
    after_snapshot = None
    peak_bytes = 0
    gc_was_enabled = gc.isenabled()
    if profile_allocations:
        gc.collect()
        gc.disable()
        tracemalloc.start()
        before_snapshot = tracemalloc.take_snapshot()

    result = None
    speed = None
    try:
        cpu_before = _take_cpu_snapshot()
        for index in range(runs):
            if setup is not None:
                setup()
            start = time.perf_counter()
            result = runner()
            latencies_ms[index] = (time.perf_counter() - start) * 1000.0
            speed = getattr(result, "speed", {})
            preprocess[index] = float(speed.get("preprocess") or 0.0)
            inference[index] = float(speed.get("inference") or 0.0)
            postprocess[index] = float(speed.get("postprocess") or 0.0)
        cpu_after = _take_cpu_snapshot()
    finally:
        result = None
        speed = None
        if profile_allocations:
            _, peak_bytes = tracemalloc.get_traced_memory()
            after_snapshot = tracemalloc.take_snapshot()
            tracemalloc.stop()
            if gc_was_enabled:
                gc.enable()
            else:
                gc.disable()

    cpu_values = _cpu_utilization(cpu_before, cpu_after)
    process_cpu = [cpu_values[0]] if cpu_values is not None else []
    system_cpu = [cpu_values[1]] if cpu_values is not None else []
    fps = 1000.0 / statistics.fmean(latencies_ms)
    print(f"{name}: runs={runs} {_format_ms(latencies_ms)} fps={fps:.2f}")
    _summarize_result_speeds(name, preprocess, inference, postprocess)
    _print_cpu_summary(name, process_cpu, system_cpu)
    allocation_summary = None
    if profile_allocations:
        assert before_snapshot is not None
        assert after_snapshot is not None
        allocation_summary = _print_allocation_summary(
            name, before_snapshot, after_snapshot, peak_bytes, allocation_top
        )

    return {
        "runs": int(runs),
        "warmup": int(warmup),
        "latency_ms": _stats(latencies_ms),
        "fps": float(fps),
        "speed_ms": {
            "preprocess": _stats(preprocess),
            "inference": _stats(inference),
            "postprocess": _stats(postprocess),
        },
        "cpu": {
            "process_percent": _stats(process_cpu),
            "system_percent": _stats(system_cpu),
        },
        "allocations": allocation_summary,
    }


def _selected_modes(mode: str) -> list[str]:
    if mode == "none":
        return []
    if mode == "both":
        return ["host", "cuda"]
    if mode == "all":
        return ["host", "cuda", "camera"]
    return [mode]


def _camera_zero_copy_value(policy: str) -> bool | None:
    if policy == "required":
        return True
    if policy == "off":
        return False
    return None


def _make_created_at() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _default_output_name(created_at: str) -> str:
    stamp = created_at.replace("-", "").replace(":", "")
    return f"benchmark-{stamp}.json"


def _resolve_output_path(output_dir: str | Path, output_name: str | None, created_at: str) -> Path:
    filename = output_name or _default_output_name(created_at)
    path = Path(filename)
    if path.suffix != ".json":
        path = path.with_suffix(".json")
    if path.is_absolute():
        return path
    return Path(output_dir) / path


def _benchmark_metadata(args: argparse.Namespace, created_at: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at": created_at,
        "package_version": __version__,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
        },
        "config": {
            "artifact_dir": str(Path(args.artifact_dir)),
            "image": str(Path(args.image)),
            "labels": list(args.labels or ["bus"]),
            "imgsz": int(args.imgsz),
            "runs": int(args.runs),
            "warmup": int(args.warmup),
            "mode": str(args.mode),
            "track": bool(args.track),
            "conf": float(args.conf),
            "device": str(args.device),
            "text_prompts": bool(args.text_prompts),
            "visual_prompts": str(args.visual_prompts),
            "visual_runtime": str(args.visual_runtime),
            "camera": {
                "source": args.camera_source,
                "width": args.camera_width,
                "height": args.camera_height,
                "fps": args.camera_fps,
                "timeout_s": float(args.camera_timeout),
                "zero_copy": str(args.camera_zero_copy),
            },
        },
    }


def _write_benchmark_json(document: dict[str, Any], output_dir: str | Path, output_name: str | None) -> Path:
    output_path = _resolve_output_path(output_dir, output_name, str(document["created_at"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return output_path


def _load_benchmark_baseline(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid benchmark baseline JSON '{path}': {exc.msg}") from exc
    except OSError as exc:
        raise ValueError(f"Unable to read benchmark baseline '{path}': {exc}") from exc
    _validate_benchmark_document(document, path)
    return document


def _validate_benchmark_document(document: object, path: Path) -> None:
    if not isinstance(document, dict):
        raise ValueError(f"Benchmark baseline '{path}' must be a JSON object")
    if document.get("schema_version") != 1:
        raise ValueError(f"Benchmark baseline '{path}' must have schema_version=1")
    results = document.get("results")
    if not isinstance(results, dict) or not results:
        raise ValueError(f"Benchmark baseline '{path}' must contain a non-empty results object")
    for mode, result in results.items():
        if not isinstance(mode, str) or not isinstance(result, dict):
            raise ValueError(f"Benchmark baseline '{path}' has an invalid result entry")


def _close_if_available(value: object) -> None:
    close = getattr(value, "close", None)
    if callable(close):
        close()


def _nested_float(data: dict[str, Any], path: tuple[str, ...]) -> float | None:
    value: Any = data
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _latency_regression(
    name: str,
    current: dict[str, Any],
    baseline: dict[str, Any],
    path: tuple[str, ...],
    max_percent: float,
) -> dict[str, Any] | None:
    current_value = _nested_float(current, path)
    baseline_value = _nested_float(baseline, path)
    if current_value is None or baseline_value is None or baseline_value <= 0.0:
        return None
    delta_percent = (current_value - baseline_value) / baseline_value * 100.0
    return {
        "metric": name,
        "baseline": baseline_value,
        "current": current_value,
        "delta": delta_percent,
        "unit": "percent",
        "threshold": float(max_percent),
        "regressed": delta_percent > max_percent,
    }


def _fps_regression(current: dict[str, Any], baseline: dict[str, Any], max_percent: float) -> dict[str, Any] | None:
    current_value = _nested_float(current, ("fps",))
    baseline_value = _nested_float(baseline, ("fps",))
    if current_value is None or baseline_value is None or baseline_value <= 0.0:
        return None
    delta_percent = (baseline_value - current_value) / baseline_value * 100.0
    return {
        "metric": "fps",
        "baseline": baseline_value,
        "current": current_value,
        "delta": delta_percent,
        "unit": "percent",
        "threshold": float(max_percent),
        "regressed": delta_percent > max_percent,
    }


def _cpu_regression(
    name: str,
    current: dict[str, Any],
    baseline: dict[str, Any],
    path: tuple[str, ...],
    max_percentage_points: float,
) -> dict[str, Any] | None:
    current_value = _nested_float(current, path)
    baseline_value = _nested_float(baseline, path)
    if current_value is None or baseline_value is None:
        return None
    delta_points = current_value - baseline_value
    return {
        "metric": name,
        "baseline": baseline_value,
        "current": current_value,
        "delta": delta_points,
        "unit": "percentage_points",
        "threshold": float(max_percentage_points),
        "regressed": delta_points > max_percentage_points,
    }


def _compare_benchmark_results(
    current: dict[str, Any],
    baseline: dict[str, Any],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    current_results = current.get("results", {})
    baseline_results = baseline.get("results", {})
    if not isinstance(current_results, dict):
        current_results = {}
    if not isinstance(baseline_results, dict):
        baseline_results = {}
    current_mode_names = sorted(str(mode) for mode in current_results)
    baseline_mode_names = sorted(str(mode) for mode in baseline_results)
    overlapping_modes = sorted(set(current_results) & set(baseline_results))
    modes: dict[str, Any] = {}
    regression_count = 0
    unchecked_modes: list[str] = []
    if not overlapping_modes:
        return {
            "status": "no_overlap",
            "regression_count": 0,
            "modes": modes,
            "current_modes": current_mode_names,
            "baseline_modes": baseline_mode_names,
            "unchecked_modes": unchecked_modes,
        }
    for mode in overlapping_modes:
        current_mode = current_results[mode]
        baseline_mode = baseline_results[mode]
        checks = [
            _latency_regression(
                "latency_ms.median",
                current_mode,
                baseline_mode,
                ("latency_ms", "median"),
                thresholds["median_latency_percent"],
            ),
            _latency_regression(
                "latency_ms.p95",
                current_mode,
                baseline_mode,
                ("latency_ms", "p95"),
                thresholds["p95_latency_percent"],
            ),
            _fps_regression(current_mode, baseline_mode, thresholds["fps_percent"]),
            _cpu_regression(
                "cpu.process_percent.mean",
                current_mode,
                baseline_mode,
                ("cpu", "process_percent", "mean"),
                thresholds["process_cpu_percentage_points"],
            ),
            _cpu_regression(
                "cpu.system_percent.mean",
                current_mode,
                baseline_mode,
                ("cpu", "system_percent", "mean"),
                thresholds["system_cpu_percentage_points"],
            ),
        ]
        resolved_checks = [check for check in checks if check is not None]
        if not resolved_checks:
            unchecked_modes.append(str(mode))
            modes[mode] = {
                "status": "no_comparable_metrics",
                "checks": [],
            }
            continue
        mode_regressions = [check for check in resolved_checks if check["regressed"]]
        regression_count += len(mode_regressions)
        modes[mode] = {
            "status": "regressed" if mode_regressions else "ok",
            "checks": resolved_checks,
        }
    if regression_count:
        status = "regressed"
    elif unchecked_modes:
        status = "no_comparable_metrics"
    else:
        status = "ok"
    return {
        "status": status,
        "regression_count": regression_count,
        "modes": modes,
        "current_modes": current_mode_names,
        "baseline_modes": baseline_mode_names,
        "unchecked_modes": unchecked_modes,
    }


def _comparison_thresholds(args: argparse.Namespace) -> dict[str, float]:
    return {
        "p95_latency_percent": float(args.max_p95_latency_regression),
        "median_latency_percent": float(args.max_median_latency_regression),
        "process_cpu_percentage_points": float(args.max_process_cpu_regression),
        "system_cpu_percentage_points": float(args.max_system_cpu_regression),
        "fps_percent": float(args.max_fps_regression),
    }


def _print_comparison_summary(comparison: dict[str, Any], baseline_path: Path) -> None:
    if comparison["status"] == "no_overlap":
        current_modes = ", ".join(comparison.get("current_modes") or ["<none>"])
        baseline_modes = ", ".join(comparison.get("baseline_modes") or ["<none>"])
        print(
            f"comparison: no overlapping modes with baseline '{baseline_path}' "
            f"(current={current_modes}; baseline={baseline_modes})"
        )
        return
    if comparison.get("unchecked_modes"):
        unchecked_modes = ", ".join(comparison["unchecked_modes"])
        print(f"comparison: no comparable metrics for mode(s) {unchecked_modes} in baseline '{baseline_path}'")
        if comparison["status"] == "no_comparable_metrics":
            return
    if comparison["status"] == "ok":
        print(f"comparison: no regressions detected against '{baseline_path}'")
        return
    print(f"comparison: {comparison['regression_count']} regression(s) detected against '{baseline_path}'")
    for mode, mode_comparison in comparison["modes"].items():
        for check in mode_comparison["checks"]:
            if check["regressed"]:
                print(
                    f"comparison regression {mode} {check['metric']}: current={check['current']:.3f} "
                    f"baseline={check['baseline']:.3f} delta={check['delta']:.2f}{check['unit']} "
                    f"threshold={check['threshold']:.2f}"
                )


def _build_engine(artifact_dir: Path, device: str, *, disable_native_visual: bool = False):
    from . import engine as engine_module
    from .engine import YOLOEEngine

    if not disable_native_visual:
        return YOLOEEngine.from_engine(artifact_dir, device=device)

    original_builder = engine_module.build_native_visual_runtime
    engine_module.build_native_visual_runtime = lambda *args, **kwargs: None
    try:
        return YOLOEEngine.from_engine(artifact_dir, device=device)
    finally:
        engine_module.build_native_visual_runtime = original_builder


def _camera_source_from_spec(source_value: str, **kwargs):
    from .gstreamer import camera_source_from_spec

    return camera_source_from_spec(source_value, **kwargs)


def _build_visual_prompt_inputs(image: np.ndarray, label: str) -> dict[str, dict[str, object]]:
    image_h, image_w = image.shape[:2]
    x1 = int(round(image_w * 0.2))
    y1 = int(round(image_h * 0.2))
    x2 = int(round(image_w * 0.8))
    y2 = int(round(image_h * 0.8))
    mask = np.zeros((1, image_h, image_w), dtype=np.uint8)
    mask[0, y1:y2, x1:x2] = 1
    return {
        "bbox": {
            "bboxes": [[float(x1), float(y1), float(x2), float(y2)]],
            "classes": [label],
        },
        "mask": {
            "masks": mask,
            "classes": [label],
        },
    }


@contextmanager
def _suppress_info_logs_for_timing():
    previous_disable_level = logging.root.manager.disable
    logging.disable(logging.INFO)
    try:
        yield
    finally:
        logging.disable(previous_disable_level)


def _synchronize_engine_device(engine: object) -> None:
    import torch

    device = torch.device(getattr(engine, "device", "cpu"))
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    runs = int(args.runs)
    warmup = int(args.warmup)
    selected_modes = _selected_modes(str(args.mode))
    if runs <= 0:
        parser.error("--runs must be greater than 0")
    if warmup < 0:
        parser.error("--warmup must be greater than or equal to 0")
    if not selected_modes and not args.text_prompts and args.visual_prompts == "none":
        parser.error("--mode none requires --text-prompts or --visual-prompts")
    if "camera" in selected_modes and not args.camera_source:
        parser.error("--camera-source is required when --mode is 'camera' or 'all'")
    if args.compare_to is not None and not Path(args.compare_to).is_file():
        parser.error(f"--compare-to file does not exist: {args.compare_to}")
    baseline = None
    if args.compare_to is not None:
        try:
            baseline = _load_benchmark_baseline(Path(args.compare_to))
        except ValueError as exc:
            parser.error(str(exc))

    configure_logging(level=args.log_level, include_timestamps=True)

    from .preprocess import normalize_imgsz
    from .source import normalize_source

    labels = list(args.labels or ["bus"])
    target_size = normalize_imgsz(int(args.imgsz))
    created_at = _make_created_at()
    document = _benchmark_metadata(args, created_at)
    document["results"] = {}

    artifact_dir = Path(args.artifact_dir)
    engine = _build_engine(artifact_dir, args.device)
    engine.clear_prompts()
    if selected_modes or args.text_prompts:
        engine.set_classes(labels)
        _synchronize_engine_device(engine)

    item = None

    def _benchmark_item():
        nonlocal item
        if item is None:
            item = normalize_source(Path(args.image), default_prefix="benchmark")[0]
        return item

    if args.text_prompts:

        def _run_text_prompt():
            with _suppress_info_logs_for_timing():
                engine.set_classes(labels)
                _synchronize_engine_device(engine)
            return None

        def _setup_text_prompt():
            with _suppress_info_logs_for_timing():
                engine.clear_prompts()

        document["results"]["text-set-classes"] = _benchmark_runner(
            "text-set-classes",
            _run_text_prompt,
            runs=runs,
            warmup=warmup,
            profile_allocations=bool(args.profile_allocations),
            allocation_top=int(args.allocation_top),
            setup=_setup_text_prompt,
        )

    host_tracker = engine.create_tracker() if args.track and "host" in selected_modes else None

    def _run_host():
        item = _benchmark_item()
        if host_tracker is not None:
            return host_tracker.update(item, source_key="benchmark", imgsz=target_size, conf=float(args.conf))
        return engine.predict_item(item, imgsz=target_size, conf=float(args.conf))

    if "host" in selected_modes:
        name = "host-track" if args.track else "host"
        document["results"][name] = _benchmark_runner(
            name,
            _run_host,
            runs=runs,
            warmup=warmup,
            profile_allocations=bool(args.profile_allocations),
            allocation_top=int(args.allocation_top),
        )

    if "cuda" in selected_modes:
        prepared_input = engine.prepare_cuda_input(_benchmark_item(), imgsz=target_size)
        cuda_tracker = engine.create_tracker() if args.track else None

        def _run_cuda():
            if cuda_tracker is not None:
                return cuda_tracker.update(
                    prepared_input,
                    source_key="benchmark",
                    conf=float(args.conf),
                    input_hint="prepared",
                    cuda=True,
                )
            return engine.predict(prepared_input, conf=float(args.conf))[0]

        name = "cuda-track" if args.track else "cuda"
        document["results"][name] = _benchmark_runner(
            name,
            _run_cuda,
            runs=runs,
            warmup=warmup,
            profile_allocations=bool(args.profile_allocations),
            allocation_top=int(args.allocation_top),
        )

    if "camera" in selected_modes:
        try:
            camera_fp16 = bool(getattr(engine, "_main_fp16"))
        except Exception:
            camera_fp16 = False
        camera_source = _camera_source_from_spec(
            str(args.camera_source),
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
            timeout_s=float(args.camera_timeout),
            prefix="benchmark_camera",
            max_frames=warmup + runs,
            zero_copy=_camera_zero_copy_value(str(args.camera_zero_copy)),
            target_imgsz=target_size,
            fp16=camera_fp16,
            device=args.device,
        )
        camera_frames = iter(camera_source)
        camera_tracker = engine.create_tracker() if args.track else None

        def _run_camera():
            try:
                frame = next(camera_frames)
            except StopIteration as exc:
                raise RuntimeError(
                    f"Camera benchmark source ended before {warmup + runs} required frame(s) were available"
                ) from exc
            if camera_tracker is not None:
                return camera_tracker.update(
                    frame,
                    source_key="benchmark-camera",
                    imgsz=target_size,
                    conf=float(args.conf),
                )
            return engine.predict_item(frame, imgsz=target_size, conf=float(args.conf))

        name = "camera-track" if args.track else "camera"
        try:
            document["results"][name] = _benchmark_runner(
                name,
                _run_camera,
                runs=runs,
                warmup=warmup,
                profile_allocations=bool(args.profile_allocations),
                allocation_top=int(args.allocation_top),
            )
        finally:
            _close_if_available(camera_frames)
            if camera_frames is not camera_source:
                _close_if_available(camera_source)

    if args.visual_prompts != "none":
        visual_modes = ["bbox", "mask"] if args.visual_prompts == "both" else [str(args.visual_prompts)]
        visual_runtime_labels: list[tuple[str, object]] = []
        if args.visual_runtime in {"native", "both"}:
            if engine.native_visual_runtime is not None:
                visual_runtime_labels.append(("native", engine))
            else:
                print("visual-native: skipped (native visual runtime unavailable)")
        if args.visual_runtime in {"python", "both"}:
            python_visual_engine = _build_engine(artifact_dir, args.device, disable_native_visual=True)
            if python_visual_engine.visual_runtime is not None:
                visual_runtime_labels.append(("python", python_visual_engine))
            else:
                print("visual-python: skipped (Python visual runtime unavailable)")

        if visual_runtime_labels:
            item = _benchmark_item()
            visual_prompt_inputs = _build_visual_prompt_inputs(item.image, labels[0])
            for runtime_name, visual_engine in visual_runtime_labels:
                for visual_mode in visual_modes:
                    prompt_kwargs = visual_prompt_inputs[visual_mode]

                    def _run_visual_prompt():
                        with _suppress_info_logs_for_timing():
                            visual_engine.set_visual_prompts(item.image, imgsz=target_size, **prompt_kwargs)
                            _synchronize_engine_device(visual_engine)
                        return None

                    def _setup_visual_prompt():
                        with _suppress_info_logs_for_timing():
                            visual_engine.clear_prompts()

                    name = f"visual-{runtime_name}-{visual_mode}"
                    document["results"][name] = _benchmark_runner(
                        name,
                        _run_visual_prompt,
                        runs=runs,
                        warmup=warmup,
                        profile_allocations=False,
                        allocation_top=0,
                        setup=_setup_visual_prompt,
                    )

    if not document["results"]:
        parser.error("no benchmark results were produced; check --mode and prompt benchmark runtime availability")

    if args.compare_to is not None:
        assert baseline is not None
        comparison = _compare_benchmark_results(document, baseline, _comparison_thresholds(args))
        document["comparison"] = comparison
        document["comparison"]["baseline_path"] = str(args.compare_to)
        _print_comparison_summary(comparison, Path(args.compare_to))
    if not args.no_save:
        output_path = _write_benchmark_json(document, args.output_dir, args.output_name)
        print(f"benchmark JSON: {output_path}")

    comparison = document.get("comparison")
    if args.fail_on_regression and comparison is not None and comparison["status"] != "ok":
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
