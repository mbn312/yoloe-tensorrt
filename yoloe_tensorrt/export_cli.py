from __future__ import annotations

import argparse
from pathlib import Path

from .logging_utils import configure_logging


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
        choices=("onnx", "engine"),
        help="Artifact format to generate. Repeat to request multiple formats. Defaults to both.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Export image size. Defaults to 640.")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic image shapes.")
    parser.add_argument("--fixed", dest="dynamic", action="store_false", help="Disable dynamic image shapes.")
    parser.set_defaults(dynamic=True)
    parser.add_argument("--no-visual-engine", action="store_true", help="Skip the visual-prompt TensorRT engine build.")
    parser.add_argument("--no-fp16", action="store_true", help="Disable FP16 TensorRT builds.")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections baked into export metadata.")
    parser.add_argument("--workspace-mb", type=int, default=2048, help="TensorRT builder workspace in MiB.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing artifacts instead of resuming.")
    parser.add_argument("--log-level", default="INFO", help="Logging level for the export command.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    configure_logging(level=args.log_level, include_timestamps=True)
    from .export import export_model

    artifact_dir = export_model(
        args.model,
        artifact_dir=None if args.artifact_dir is None else Path(args.artifact_dir),
        formats=tuple(args.formats or ("onnx", "engine")),
        dynamic=bool(args.dynamic),
        build_visual_engine=not args.no_visual_engine,
        fp16=not args.no_fp16,
        imgsz=int(args.imgsz),
        max_det=int(args.max_det),
        overwrite=bool(args.overwrite),
        workspace_bytes=int(args.workspace_mb) << 20,
    )
    print(artifact_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
