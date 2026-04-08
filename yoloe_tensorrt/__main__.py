from __future__ import annotations

import argparse
import sys

from ._version import __version__


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m yoloe_tensorrt", description="CLI entry points for yoloe-tensorrt."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    camera_parser = subparsers.add_parser("camera-gui", help="Launch the live camera GUI.")
    camera_parser.set_defaults(handler="camera-gui")

    export_parser = subparsers.add_parser("export", help="Export a YOLOE checkpoint into an artifact bundle.")
    export_parser.set_defaults(handler="export")

    benchmark_parser = subparsers.add_parser("benchmark", help="Benchmark host-image and CUDA-tensor inference.")
    benchmark_parser.set_defaults(handler="benchmark")
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["camera-gui"]:
        from .gui import main as gui_main

        return gui_main(argv[1:])
    if argv[:1] == ["export"]:
        from .export_cli import main as export_main

        return export_main(argv[1:])
    if argv[:1] == ["benchmark"]:
        from .benchmark_cli import main as benchmark_main

        return benchmark_main(argv[1:])

    parser = build_arg_parser()
    parser.parse_args(argv)
    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
