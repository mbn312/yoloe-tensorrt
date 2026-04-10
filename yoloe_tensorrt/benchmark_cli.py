from __future__ import annotations

import argparse
import gc
import logging
import math
import statistics
import time
import tracemalloc
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

import numpy as np

from .logging_utils import configure_logging


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark host-image and CUDA-tensor inference paths for a YOLOE TensorRT artifact bundle, "
            "and optionally benchmark visual-prompt updates."
        )
    )
    parser.add_argument("artifact_dir", help="Artifact bundle directory to benchmark.")
    parser.add_argument("image", help="Image path to use for repeated benchmark runs.")
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
        choices=("host", "cuda", "both"),
        default="both",
        help="Benchmark the host-image path, CUDA fast path, or both.",
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
) -> None:
    stats = [stat for stat in after.compare_to(before, "lineno") if stat.size_diff > 0 or stat.count_diff > 0]
    retained_bytes = sum(max(0, stat.size_diff) for stat in stats)
    retained_count = sum(max(0, stat.count_diff) for stat in stats)
    print(
        f"{name} Python retained allocations: retained={_format_bytes(retained_bytes)} "
        f"count={retained_count} peak_traced={_format_bytes(peak_bytes)}"
    )
    for index, stat in enumerate(stats[: max(0, int(top_n))], start=1):
        frame = stat.traceback[0]
        print(
            f"{name} retained allocation hotspot {index}: {frame.filename}:{frame.lineno} "
            f"size={_format_bytes(max(0, stat.size_diff))} count={max(0, stat.count_diff)}"
        )


def _benchmark_runner(
    name: str,
    runner: Callable[[], object],
    *,
    runs: int,
    warmup: int,
    profile_allocations: bool,
    allocation_top: int,
) -> None:
    for _ in range(warmup):
        runner()

    latencies_ms = [0.0] * runs
    preprocess = [0.0] * runs
    inference = [0.0] * runs
    postprocess = [0.0] * runs

    before_snapshot = None
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
        for index in range(runs):
            start = time.perf_counter()
            result = runner()
            latencies_ms[index] = (time.perf_counter() - start) * 1000.0
            speed = getattr(result, "speed", {})
            preprocess[index] = float(speed.get("preprocess") or 0.0)
            inference[index] = float(speed.get("inference") or 0.0)
            postprocess[index] = float(speed.get("postprocess") or 0.0)
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

    fps = 1000.0 / statistics.fmean(latencies_ms)
    print(f"{name}: runs={runs} {_format_ms(latencies_ms)} fps={fps:.2f}")
    _summarize_result_speeds(name, preprocess, inference, postprocess)
    if profile_allocations:
        assert before_snapshot is not None
        _print_allocation_summary(name, before_snapshot, after_snapshot, peak_bytes, allocation_top)


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

    device = torch.device(getattr(engine, "device", "cuda:0"))
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    configure_logging(level=args.log_level, include_timestamps=True)

    from .preprocess import normalize_imgsz
    from .source import normalize_source

    labels = list(args.labels or ["bus"])
    target_size = normalize_imgsz(int(args.imgsz))

    artifact_dir = Path(args.artifact_dir)
    engine = _build_engine(artifact_dir, args.device)
    engine.clear_prompts()
    engine.set_classes(labels)

    item = normalize_source(Path(args.image), default_prefix="benchmark")[0]
    host_tracker = engine.create_tracker() if args.track and args.mode in {"host", "both"} else None

    def _run_host():
        if host_tracker is not None:
            return host_tracker.update(item, source_key="benchmark", imgsz=target_size, conf=float(args.conf))
        return engine.predict_item(item, imgsz=target_size, conf=float(args.conf))

    if args.mode in {"host", "both"}:
        _benchmark_runner(
            "host-track" if args.track else "host",
            _run_host,
            runs=int(args.runs),
            warmup=int(args.warmup),
            profile_allocations=bool(args.profile_allocations),
            allocation_top=int(args.allocation_top),
        )

    if args.mode in {"cuda", "both"}:
        prepared_input = engine.prepare_cuda_input(item, imgsz=target_size)
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

        _benchmark_runner(
            "cuda-track" if args.track else "cuda",
            _run_cuda,
            runs=int(args.runs),
            warmup=int(args.warmup),
            profile_allocations=bool(args.profile_allocations),
            allocation_top=int(args.allocation_top),
        )

    if args.visual_prompts != "none":
        visual_prompt_inputs = _build_visual_prompt_inputs(item.image, labels[0])
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

        for runtime_name, visual_engine in visual_runtime_labels:
            for visual_mode in visual_modes:
                prompt_kwargs = visual_prompt_inputs[visual_mode]
                for _ in range(int(args.warmup)):
                    with _suppress_info_logs_for_timing():
                        visual_engine.clear_prompts()
                        visual_engine.set_visual_prompts(item.image, imgsz=target_size, **prompt_kwargs)
                        _synchronize_engine_device(visual_engine)

                latencies_ms: list[float] = []
                for _ in range(int(args.runs)):
                    with _suppress_info_logs_for_timing():
                        visual_engine.clear_prompts()
                        start = time.perf_counter()
                        visual_engine.set_visual_prompts(item.image, imgsz=target_size, **prompt_kwargs)
                        _synchronize_engine_device(visual_engine)
                    latencies_ms.append((time.perf_counter() - start) * 1000.0)

                print(f"visual-{runtime_name}-{visual_mode}: runs={int(args.runs)} {_format_ms(latencies_ms)}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
