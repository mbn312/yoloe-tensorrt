from __future__ import annotations

import logging
import os
import sys
from typing import TextIO

_ROOT_LOGGER_NAME = "yoloe_tensorrt"
_HANDLER_NAME = "yoloe_tensorrt.stream"


def _coerce_level(level: int | str | None) -> int:
    if level is None:
        level = os.environ.get("YOLOE_TRT_LOG_LEVEL", "INFO")
    if isinstance(level, int):
        return level
    resolved = logging.getLevelName(str(level).upper())
    return resolved if isinstance(resolved, int) else logging.INFO


def configure_logging(
    level: int | str | None = None,
    stream: TextIO | None = None,
    include_timestamps: bool = False,
) -> logging.Logger:
    logger = logging.getLogger(_ROOT_LOGGER_NAME)
    resolved_level = _coerce_level(level)
    logger.setLevel(resolved_level)
    logger.propagate = False

    fmt = "%(levelname)s %(name)s: %(message)s"
    if include_timestamps:
        fmt = "%(asctime)s " + fmt
    formatter = logging.Formatter(fmt)

    handler = next((item for item in logger.handlers if item.get_name() == _HANDLER_NAME), None)
    if handler is None:
        handler = logging.StreamHandler(stream or sys.stderr)
        handler.set_name(_HANDLER_NAME)
        logger.addHandler(handler)
    elif stream is not None and isinstance(handler, logging.StreamHandler):
        handler.setStream(stream)

    handler.setLevel(resolved_level)
    handler.setFormatter(formatter)
    return logger


def get_logger(name: str) -> logging.Logger:
    root = configure_logging()
    if name == _ROOT_LOGGER_NAME:
        return root
    return root.getChild(name.removeprefix(_ROOT_LOGGER_NAME).removeprefix("."))
