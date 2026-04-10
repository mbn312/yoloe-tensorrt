from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

from .datasets import DatasetValidationError
from .logging_utils import configure_logging
from .training import TrainingError, TrainingResult, train_model


class _CliArgumentError(ValueError):
    pass


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train or fine-tune a YOLOE checkpoint through Ultralytics.")
    parser.add_argument("model", help="Local checkpoint path, URL, or downloadable asset name.")
    parser.add_argument("data", help="Ultralytics-compatible dataset YAML/config path.")
    parser.add_argument("--task", choices=("detect", "segment"), default="detect", help="Training task.")
    parser.add_argument(
        "--imgsz",
        type=_parse_imgsz,
        default=640,
        metavar="N|HxW",
        help="Image size. Use one value for square training or HxW/H,W for rectangular training. Defaults to 640.",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs. Defaults to 100.")
    parser.add_argument(
        "--batch",
        type=_parse_batch,
        default=16,
        help="Batch size. YAML scalars are accepted, for example 16, 0.5, or auto. Defaults to 16.",
    )
    parser.add_argument("--device", default=None, help="Training device, for example cuda:0, 0, or cpu.")
    parser.add_argument("--output-dir", default="outputs/training", help="Directory for Ultralytics runs.")
    parser.add_argument("--name", default=None, help="Ultralytics run name.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow Ultralytics to reuse an existing run name.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training with Ultralytics resume=True.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        "--resume-from",
        dest="resume_checkpoint",
        default=None,
        metavar="CHECKPOINT",
        help="Resume training from a specific checkpoint path.",
    )
    parser.add_argument(
        "--ultralytics-arg",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Extra Ultralytics trainer override. Repeat for multiple args; values are parsed as YAML.",
    )
    parser.add_argument(
        "--no-validate-dataset",
        action="store_false",
        dest="validate_dataset",
        help="Skip yoloe-tensorrt dataset validation before launching Ultralytics.",
    )
    parser.set_defaults(validate_dataset=True)
    parser.add_argument("--log-level", default="INFO", help="Logging level for the training command.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    configure_logging(level=args.log_level, include_timestamps=True)

    try:
        overrides = _parse_ultralytics_args(args.ultralytics_arg)
        if args.resume and args.resume_checkpoint is not None:
            raise _CliArgumentError("--resume cannot be combined with --resume-checkpoint.")
        if args.resume or args.resume_checkpoint is not None:
            if "resume" in overrides:
                raise _CliArgumentError("--resume cannot be combined with --ultralytics-arg resume=...")
            overrides["resume"] = args.resume_checkpoint if args.resume_checkpoint is not None else True
    except _CliArgumentError as exc:
        parser.error(str(exc))

    try:
        result = train_model(
            args.model,
            Path(args.data),
            task=args.task,
            imgsz=args.imgsz,
            epochs=int(args.epochs),
            batch=args.batch,
            device=args.device,
            output_dir=Path(args.output_dir),
            name=args.name,
            overrides=overrides,
            validate_dataset=bool(args.validate_dataset),
            exist_ok=bool(args.exist_ok),
        )
    except ImportError as exc:
        print(
            "error: training dependencies are unavailable. Ensure ultralytics and torch are installed "
            f"for training: {exc}",
            file=sys.stderr,
        )
        return 1
    except (DatasetValidationError, TrainingError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    _print_result(result)
    return 0


def _parse_batch(value: str) -> int | float | str:
    parsed = _load_yaml_value(value)
    if isinstance(parsed, bool) or parsed is None or isinstance(parsed, list | dict):
        raise argparse.ArgumentTypeError("--batch must be an integer, float, or string such as auto.")
    if not isinstance(parsed, int | float | str):
        raise argparse.ArgumentTypeError("--batch must be an integer, float, or string such as auto.")
    return parsed


def _parse_imgsz(value: str) -> int | tuple[int, int]:
    raw_value = value.strip().lower()
    if not raw_value:
        raise argparse.ArgumentTypeError("--imgsz must not be empty.")

    for separator in ("x", ","):
        if separator in raw_value:
            parts = [part.strip() for part in raw_value.split(separator)]
            if len(parts) != 2 or not all(parts):
                raise argparse.ArgumentTypeError("--imgsz must be one integer or two integers as HxW or H,W.")
            height, width = (_parse_positive_int(part, "--imgsz") for part in parts)
            return (height, width)

    return _parse_positive_int(raw_value, "--imgsz")


def _parse_positive_int(value: str, option_name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{option_name} must contain integer value(s).") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"{option_name} must be greater than zero.")
    return parsed


def _parse_ultralytics_args(entries: list[str] | None) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for entry in entries or []:
        if "=" not in entry:
            raise _CliArgumentError("--ultralytics-arg must use KEY=VALUE syntax.")
        key, raw_value = entry.split("=", 1)
        key = key.strip()
        if not key:
            raise _CliArgumentError("--ultralytics-arg keys must not be empty.")
        if key in overrides:
            raise _CliArgumentError(f"--ultralytics-arg key {key!r} was provided more than once.")
        try:
            overrides[key] = "" if raw_value == "" else _load_yaml_value(raw_value)
        except argparse.ArgumentTypeError as exc:
            raise _CliArgumentError(str(exc)) from exc
    return overrides


def _load_yaml_value(value: str) -> Any:
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError as exc:
        raise argparse.ArgumentTypeError(f"could not parse YAML scalar {value!r}: {exc}") from exc


def _print_result(result: TrainingResult) -> None:
    print(f"checkpoint: {result.checkpoint_path}")
    print(f"run_dir: {result.run_dir}")
    if result.metrics_path is not None:
        print(f"metrics: {result.metrics_path}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
