from __future__ import annotations

import argparse
from pathlib import Path
from typing import TypeAlias

from ._cli_utils import parse_positive_int
from .logging_utils import configure_logging

_EXPORT_FORMATS = ("onnx", "engine")
_EXPORTER_CHOICES = ("auto", "legacy", "dynamo")
_DEFAULT_EXPORT_FORMATS = ("onnx", "engine")
_DEFAULT_EXPORT_IMGSZ = 640
_DEFAULT_EXPORT_MAX_DET = 300
_DEFAULT_EXPORT_WORKSPACE_MB = 2048
_DEFAULT_ONNX_EXPORTER = "auto"
ExportKwargsValue: TypeAlias = Path | tuple[str, ...] | bool | int | str | None
_POST_TRAINING_OVERRIDE_FIELDS = (
    "export_checkpoint",
    "export_artifact_dir",
    "export_formats",
    "export_dynamic",
    "export_no_visual_engine",
    "export_no_fp16",
    "export_imgsz",
    "export_max_det",
    "export_workspace_mb",
    "export_exporter",
    "export_opset_version",
    "export_overwrite",
)


def _option_name(prefix: str, name: str) -> str:
    return f"--{prefix}{name}"


def _dest_name(prefix: str, name: str) -> str:
    return f"{prefix}{name}"


def _add_positive_int_argument(
    parser: argparse._ActionsContainer,
    *,
    option_prefix: str,
    dest_prefix: str,
    name: str,
    default: int | None,
    help_text: str,
) -> None:
    option_name = _option_name(option_prefix, name)
    parser.add_argument(
        option_name,
        dest=_dest_name(dest_prefix, name.replace("-", "_")),
        type=lambda value, current_option=option_name: parse_positive_int(value, current_option),
        default=default,
        help=help_text,
    )


def _add_dynamic_shape_arguments(
    parser: argparse._ActionsContainer,
    *,
    option_prefix: str,
    dest_prefix: str,
    default: bool | None,
    enabled_help: str,
    disabled_help: str,
) -> None:
    dest = _dest_name(dest_prefix, "dynamic")
    parser.add_argument(_option_name(option_prefix, "dynamic"), action="store_true", dest=dest, help=enabled_help)
    parser.add_argument(_option_name(option_prefix, "fixed"), action="store_false", dest=dest, help=disabled_help)
    parser.set_defaults(**{dest: default})


def add_post_training_export_arguments(parser: argparse._ActionsContainer) -> None:
    parser.add_argument(
        "--export-artifact-dir",
        default=None,
        help="Artifact bundle output directory. Defaults to the standard export_model(...) location.",
    )
    parser.add_argument(
        "--export-format",
        action="append",
        dest="export_formats",
        choices=_EXPORT_FORMATS,
        default=None,
        help="Artifact format to generate after training. Repeat to request multiple formats. Defaults to both.",
    )
    _add_dynamic_shape_arguments(
        parser,
        option_prefix="export-",
        dest_prefix="export_",
        default=None,
        enabled_help="Enable dynamic image shapes for post-training export.",
        disabled_help="Disable dynamic image shapes for post-training export.",
    )
    parser.add_argument(
        "--export-no-visual-engine",
        action="store_true",
        help="Skip the visual-prompt TensorRT engine build during post-training export.",
    )
    parser.add_argument(
        "--export-no-fp16",
        action="store_true",
        help="Disable FP16 TensorRT builds during post-training export.",
    )
    _add_positive_int_argument(
        parser,
        option_prefix="export-",
        dest_prefix="export_",
        name="imgsz",
        default=None,
        help_text="Export image size. Defaults to the export_model(...) behavior when omitted.",
    )
    _add_positive_int_argument(
        parser,
        option_prefix="export-",
        dest_prefix="export_",
        name="max-det",
        default=None,
        help_text="Maximum detections baked into export metadata. Defaults to 300.",
    )
    _add_positive_int_argument(
        parser,
        option_prefix="export-",
        dest_prefix="export_",
        name="workspace-mb",
        default=None,
        help_text="TensorRT builder workspace in MiB for post-training export. Defaults to 2048.",
    )
    parser.add_argument(
        "--export-exporter",
        choices=_EXPORTER_CHOICES,
        default=None,
        help="ONNX exporter backend for post-training export. Defaults to auto.",
    )
    _add_positive_int_argument(
        parser,
        option_prefix="export-",
        dest_prefix="export_",
        name="opset-version",
        default=None,
        help_text="Override the ONNX opset version used for post-training export.",
    )
    parser.add_argument(
        "--export-overwrite",
        action="store_true",
        help="Overwrite existing artifacts during post-training export.",
    )


def _build_export_kwargs(
    args: argparse.Namespace,
    *,
    attr_prefix: str,
    dynamic_default: bool,
    imgsz_default: int | None,
    max_det_default: int,
    workspace_mb_default: int,
    exporter_default: str,
) -> dict[str, ExportKwargsValue]:
    dynamic = getattr(args, _dest_name(attr_prefix, "dynamic"))
    imgsz = getattr(args, _dest_name(attr_prefix, "imgsz"))
    max_det = getattr(args, _dest_name(attr_prefix, "max_det"))
    workspace_mb = getattr(args, _dest_name(attr_prefix, "workspace_mb"))
    exporter = getattr(args, _dest_name(attr_prefix, "exporter"))
    opset_version = getattr(args, _dest_name(attr_prefix, "opset_version"))
    return {
        "artifact_dir": None
        if getattr(args, _dest_name(attr_prefix, "artifact_dir")) is None
        else Path(getattr(args, _dest_name(attr_prefix, "artifact_dir"))),
        "formats": tuple(getattr(args, _dest_name(attr_prefix, "formats")) or _DEFAULT_EXPORT_FORMATS),
        "dynamic": dynamic_default if dynamic is None else bool(dynamic),
        "build_visual_engine": not getattr(args, _dest_name(attr_prefix, "no_visual_engine")),
        "fp16": not getattr(args, _dest_name(attr_prefix, "no_fp16")),
        "imgsz": imgsz_default if imgsz is None else int(imgsz),
        "max_det": max_det_default if max_det is None else int(max_det),
        "overwrite": bool(getattr(args, _dest_name(attr_prefix, "overwrite"))),
        "workspace_bytes": int(workspace_mb_default if workspace_mb is None else workspace_mb) << 20,
        "onnx_exporter": str(exporter_default if exporter is None else exporter),
        "onnx_opset_version": None if opset_version is None else int(opset_version),
    }


def build_export_kwargs(args: argparse.Namespace) -> dict[str, ExportKwargsValue]:
    return _build_export_kwargs(
        args,
        attr_prefix="",
        dynamic_default=True,
        imgsz_default=_DEFAULT_EXPORT_IMGSZ,
        max_det_default=_DEFAULT_EXPORT_MAX_DET,
        workspace_mb_default=_DEFAULT_EXPORT_WORKSPACE_MB,
        exporter_default=_DEFAULT_ONNX_EXPORTER,
    )


def build_post_training_export_kwargs(args: argparse.Namespace) -> dict[str, ExportKwargsValue]:
    resolved = _build_export_kwargs(
        args,
        attr_prefix="export_",
        dynamic_default=True,
        imgsz_default=None,
        max_det_default=_DEFAULT_EXPORT_MAX_DET,
        workspace_mb_default=_DEFAULT_EXPORT_WORKSPACE_MB,
        exporter_default=_DEFAULT_ONNX_EXPORTER,
    )
    return {f"export_{key}": value for key, value in resolved.items()}


def has_post_training_export_overrides(args: argparse.Namespace) -> bool:
    nullable_fields = {
        "export_checkpoint",
        "export_artifact_dir",
        "export_dynamic",
        "export_imgsz",
        "export_max_det",
        "export_workspace_mb",
        "export_exporter",
        "export_opset_version",
    }
    return any(
        value is not None if field in nullable_fields else bool(value)
        for field in _POST_TRAINING_OVERRIDE_FIELDS
        for value in (getattr(args, field),)
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a YOLOE checkpoint into a yoloe-tensorrt artifact bundle.")
    parser.add_argument("model", help="Local checkpoint path or downloadable asset name, for example yoloe-26s-seg.pt.")
    parser.add_argument(
        "--artifact-dir", help="Output artifact directory. Defaults to a sibling directory next to the model."
    )
    parser.add_argument(
        "--format",
        action="append",
        dest="formats",
        choices=_EXPORT_FORMATS,
        help="Artifact format to generate. Repeat to request multiple formats. Defaults to both.",
    )
    _add_positive_int_argument(
        parser,
        option_prefix="",
        dest_prefix="",
        name="imgsz",
        default=_DEFAULT_EXPORT_IMGSZ,
        help_text="Export image size. Defaults to 640.",
    )
    _add_dynamic_shape_arguments(
        parser,
        option_prefix="",
        dest_prefix="",
        default=True,
        enabled_help="Enable dynamic image shapes.",
        disabled_help="Disable dynamic image shapes.",
    )
    parser.add_argument("--no-visual-engine", action="store_true", help="Skip the visual-prompt TensorRT engine build.")
    parser.add_argument("--no-fp16", action="store_true", help="Disable FP16 TensorRT builds.")
    _add_positive_int_argument(
        parser,
        option_prefix="",
        dest_prefix="",
        name="max-det",
        default=_DEFAULT_EXPORT_MAX_DET,
        help_text="Maximum detections baked into export metadata.",
    )
    _add_positive_int_argument(
        parser,
        option_prefix="",
        dest_prefix="",
        name="workspace-mb",
        default=_DEFAULT_EXPORT_WORKSPACE_MB,
        help_text="TensorRT builder workspace in MiB.",
    )
    parser.add_argument(
        "--exporter",
        choices=_EXPORTER_CHOICES,
        default=_DEFAULT_ONNX_EXPORTER,
        help="ONNX exporter backend. 'auto' tries dynamo first and falls back to legacy.",
    )
    _add_positive_int_argument(
        parser,
        option_prefix="",
        dest_prefix="",
        name="opset-version",
        default=None,
        help_text="Override the ONNX opset version. Defaults to 18 for dynamo and 17 for legacy.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing artifacts instead of resuming.")
    parser.add_argument("--log-level", default="INFO", help="Logging level for the export command.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    configure_logging(level=args.log_level, include_timestamps=True)
    from .export import export_model

    artifact_dir = export_model(args.model, **build_export_kwargs(args))
    print(artifact_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
