from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from .logging_utils import configure_logging


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark host-image and CUDA-tensor inference paths for a YOLOE TensorRT artifact bundle."
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
    parser.add_argument("--log-level", default="INFO", help="Logging level for the benchmark command.")
    return parser


def _format_ms(values: list[float]) -> str:
    return (
        f"mean={statistics.fmean(values):.2f}ms "
        f"median={statistics.median(values):.2f}ms "
        f"p95={sorted(values)[max(0, int(len(values) * 0.95) - 1)]:.2f}ms"
    )


def _summarize_result_speeds(name: str, speeds: list[dict[str, float]]) -> None:
    if not speeds:
        return
    preprocess = [speed["preprocess"] for speed in speeds]
    inference = [speed["inference"] for speed in speeds]
    postprocess = [speed["postprocess"] for speed in speeds]
    print(
        f"{name} speed breakdown: preprocess={statistics.fmean(preprocess):.2f}ms "
        f"inference={statistics.fmean(inference):.2f}ms postprocess={statistics.fmean(postprocess):.2f}ms"
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    configure_logging(level=args.log_level, include_timestamps=True)

    from .engine import YOLOEEngine
    from .preprocess import normalize_imgsz
    from .source import normalize_source

    labels = list(args.labels or ["bus"])
    target_size = normalize_imgsz(int(args.imgsz))

    engine = YOLOEEngine.from_engine(Path(args.artifact_dir), device=args.device)
    engine.clear_prompts()
    engine.set_classes(labels)

    item = normalize_source(Path(args.image), default_prefix="benchmark")[0]
    prepared_input = engine.prepare_cuda_input(item, imgsz=target_size)

    def _run_host():
        return engine.predict_item(item, imgsz=target_size, conf=float(args.conf))

    def _run_cuda():
        return engine.predict(prepared_input, conf=float(args.conf))[0]

    runners = []
    if args.mode in {"host", "both"}:
        runners.append(("host", _run_host))
    if args.mode in {"cuda", "both"}:
        runners.append(("cuda", _run_cuda))

    for name, runner in runners:
        for _ in range(int(args.warmup)):
            runner()

        latencies_ms: list[float] = []
        speeds: list[dict[str, float]] = []
        for _ in range(int(args.runs)):
            start = time.perf_counter()
            result = runner()
            latencies_ms.append((time.perf_counter() - start) * 1000.0)
            speeds.append(dict(result.speed))

        fps = 1000.0 / statistics.fmean(latencies_ms)
        print(f"{name}: runs={int(args.runs)} {_format_ms(latencies_ms)} fps={fps:.2f}")
        _summarize_result_speeds(name, speeds)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
