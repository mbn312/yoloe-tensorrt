from __future__ import annotations

import os
from pathlib import Path

from .logging_utils import get_logger

LOGGER = get_logger(__name__)

try:
    from ._native import NativeMainRuntime

    NATIVE_AVAILABLE = os.environ.get("YOLOE_TRT_DISABLE_NATIVE", "").lower() not in {"1", "true", "yes"}
    NATIVE_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - import surface depends on local build/runtime libs
    NativeMainRuntime = None  # type: ignore[assignment]
    NATIVE_AVAILABLE = False
    NATIVE_IMPORT_ERROR = exc


def build_native_main_runtime(
    engine_path: str | Path,
    image_input_name: str,
    prompt_input_name: str,
    device: str | int = "cuda:0",
):
    if not NATIVE_AVAILABLE or NativeMainRuntime is None:
        if NATIVE_IMPORT_ERROR is not None:
            LOGGER.info("Native backend unavailable: %s", NATIVE_IMPORT_ERROR)
        return None

    if isinstance(device, int):
        device_id = device
    else:
        device_str = str(device)
        if not device_str.startswith("cuda"):
            LOGGER.info("Native backend disabled for non-CUDA device '%s'", device_str)
            return None
        if ":" in device_str:
            device_id = int(device_str.split(":", 1)[1])
        else:
            device_id = 0

    return NativeMainRuntime(str(engine_path), image_input_name, prompt_input_name, device_id)
