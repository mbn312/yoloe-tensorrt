from __future__ import annotations

import argparse
import importlib
import sys
from dataclasses import dataclass

from ._version import __version__


@dataclass(frozen=True)
class _CommandSpec:
    name: str
    help: str
    module_path: str


_COMMANDS = (
    _CommandSpec("camera-gui", "Launch the live camera GUI.", "yoloe_tensorrt.gui"),
    _CommandSpec("export", "Export a YOLOE checkpoint into an artifact bundle.", "yoloe_tensorrt.export_cli"),
    _CommandSpec("benchmark", "Benchmark runtime performance.", "yoloe_tensorrt.benchmark_cli"),
    _CommandSpec("train", "Train or fine-tune a YOLOE checkpoint.", "yoloe_tensorrt.train_cli"),
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m yoloe_tensorrt", description="CLI entry points for yoloe-tensorrt."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")
    for command in _COMMANDS:
        subparsers.add_parser(command.name, help=command.help)
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1]:
        for command in _COMMANDS:
            if argv[0] == command.name:
                return importlib.import_module(command.module_path).main(argv[1:])

    parser = build_arg_parser()
    parser.parse_args(argv)
    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
