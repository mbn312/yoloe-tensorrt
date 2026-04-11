from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from ._cli_utils import load_yaml_value, parse_imgsz, parse_key_value_args, parse_positive_int
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
        type=lambda value: parse_imgsz(value, "--imgsz"),
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
    export_group = parser.add_argument_group("Export After Training")
    export_group.add_argument(
        "--export",
        action="store_true",
        help="Export the selected trained checkpoint into a yoloe-tensorrt artifact bundle after training.",
    )
    export_group.add_argument(
        "--export-checkpoint",
        choices=("best", "last"),
        default=None,
        help="Which trained checkpoint to export. Defaults to best when --export is enabled.",
    )
    export_group.add_argument(
        "--export-artifact-dir",
        default=None,
        help="Artifact bundle output directory. Defaults to the standard export_model(...) location.",
    )
    export_group.add_argument(
        "--export-format",
        action="append",
        dest="export_formats",
        choices=("onnx", "engine"),
        default=None,
        help="Artifact format to generate after training. Repeat to request multiple formats. Defaults to both.",
    )
    export_group.add_argument(
        "--export-dynamic",
        action="store_true",
        dest="export_dynamic",
        help="Enable dynamic image shapes for post-training export.",
    )
    export_group.add_argument(
        "--export-fixed",
        action="store_false",
        dest="export_dynamic",
        help="Disable dynamic image shapes for post-training export.",
    )
    parser.set_defaults(export_dynamic=None)
    export_group.add_argument(
        "--export-no-visual-engine",
        action="store_true",
        help="Skip the visual-prompt TensorRT engine build during post-training export.",
    )
    export_group.add_argument(
        "--export-no-fp16",
        action="store_true",
        help="Disable FP16 TensorRT builds during post-training export.",
    )
    export_group.add_argument(
        "--export-imgsz",
        type=lambda value: parse_positive_int(value, "--export-imgsz"),
        default=None,
        help="Export image size. Defaults to the export_model(...) behavior when omitted.",
    )
    export_group.add_argument(
        "--export-max-det",
        type=lambda value: parse_positive_int(value, "--export-max-det"),
        default=None,
        help="Maximum detections baked into export metadata. Defaults to 300.",
    )
    export_group.add_argument(
        "--export-workspace-mb",
        type=lambda value: parse_positive_int(value, "--export-workspace-mb"),
        default=None,
        help="TensorRT builder workspace in MiB for post-training export. Defaults to 2048.",
    )
    export_group.add_argument(
        "--export-exporter",
        choices=("auto", "legacy", "dynamo"),
        default=None,
        help="ONNX exporter backend for post-training export. Defaults to auto.",
    )
    export_group.add_argument(
        "--export-opset-version",
        type=lambda value: parse_positive_int(value, "--export-opset-version"),
        default=None,
        help="Override the ONNX opset version used for post-training export.",
    )
    export_group.add_argument(
        "--export-overwrite",
        action="store_true",
        help="Overwrite existing artifacts during post-training export.",
    )
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
        if _has_export_options(args) and not args.export:
            raise _CliArgumentError("--export must be provided when using --export-* options.")
    except _CliArgumentError as exc:
        parser.error(str(exc))

    train_kwargs: dict[str, Any] = {
        "task": args.task,
        "imgsz": args.imgsz,
        "epochs": int(args.epochs),
        "batch": args.batch,
        "device": args.device,
        "output_dir": Path(args.output_dir),
        "name": args.name,
        "overrides": overrides,
        "validate_dataset": bool(args.validate_dataset),
        "exist_ok": bool(args.exist_ok),
    }
    if args.export:
        train_kwargs.update(
            export_artifact=True,
            export_checkpoint=str(args.export_checkpoint or "best"),
            export_artifact_dir=None if args.export_artifact_dir is None else Path(args.export_artifact_dir),
            export_formats=tuple(args.export_formats or ("onnx", "engine")),
            export_dynamic=True if args.export_dynamic is None else bool(args.export_dynamic),
            export_build_visual_engine=not args.export_no_visual_engine,
            export_fp16=not args.export_no_fp16,
            export_imgsz=None if args.export_imgsz is None else int(args.export_imgsz),
            export_max_det=300 if args.export_max_det is None else int(args.export_max_det),
            export_overwrite=bool(args.export_overwrite),
            export_workspace_bytes=(
                (2048 if args.export_workspace_mb is None else int(args.export_workspace_mb)) << 20
            ),
            export_onnx_exporter=str(args.export_exporter or "auto"),
            export_onnx_opset_version=None if args.export_opset_version is None else int(args.export_opset_version),
        )

    try:
        result = train_model(
            args.model,
            Path(args.data),
            **train_kwargs,
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
    parsed = load_yaml_value(value)
    if isinstance(parsed, bool) or parsed is None or isinstance(parsed, list | dict):
        raise argparse.ArgumentTypeError("--batch must be an integer, float, or string such as auto.")
    if not isinstance(parsed, int | float | str):
        raise argparse.ArgumentTypeError("--batch must be an integer, float, or string such as auto.")
    return parsed


def _has_export_options(args: argparse.Namespace) -> bool:
    return any(
        (
            args.export_checkpoint is not None,
            args.export_artifact_dir is not None,
            bool(args.export_formats),
            args.export_dynamic is not None,
            bool(args.export_no_visual_engine),
            bool(args.export_no_fp16),
            args.export_imgsz is not None,
            args.export_max_det is not None,
            args.export_workspace_mb is not None,
            args.export_exporter is not None,
            args.export_opset_version is not None,
            bool(args.export_overwrite),
        )
    )


def _parse_ultralytics_args(entries: list[str] | None) -> dict[str, Any]:
    return parse_key_value_args(entries, option_name="--ultralytics-arg", error_type=_CliArgumentError)


def _print_result(result: TrainingResult) -> None:
    print(f"checkpoint: {result.checkpoint_path}")
    print(f"run_dir: {result.run_dir}")
    if result.metrics_path is not None:
        print(f"metrics: {result.metrics_path}")
    if result.exported_checkpoint_path is not None:
        print(f"exported_checkpoint: {result.exported_checkpoint_path}")
    if result.artifact_dir is not None:
        print(f"artifact_dir: {result.artifact_dir}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
