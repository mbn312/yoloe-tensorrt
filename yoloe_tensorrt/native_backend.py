from __future__ import annotations

import os
from pathlib import Path

from .logging_utils import get_logger

LOGGER = get_logger(__name__)

try:
    from . import _native as _native_module

    NativeMainRuntime = _native_module.NativeMainRuntime  # type: ignore[attr-defined]
    NativeVisualPromptRuntime = getattr(_native_module, "NativeVisualPromptRuntime", None)
    NativeJetsonCameraSource = getattr(_native_module, "NativeJetsonCameraSource", None)

    NATIVE_AVAILABLE = os.environ.get("YOLOE_TRT_DISABLE_NATIVE", "").lower() not in {"1", "true", "yes"}
    NATIVE_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - import surface depends on local build/runtime libs
    NativeMainRuntime = None  # type: ignore[assignment]
    NativeVisualPromptRuntime = None  # type: ignore[assignment]
    NativeJetsonCameraSource = None  # type: ignore[assignment]
    NATIVE_AVAILABLE = False
    NATIVE_IMPORT_ERROR = exc


JETSON_ZERO_COPY_AVAILABLE = bool(NATIVE_AVAILABLE and NativeJetsonCameraSource is not None)


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


def build_native_visual_runtime(
    engine_path: str | Path,
    image_input_name: str,
    visual_input_name: str,
    visual_stride: int,
    device: str | int = "cuda:0",
):
    if not NATIVE_AVAILABLE or NativeVisualPromptRuntime is None:
        if NATIVE_IMPORT_ERROR is not None:
            LOGGER.info("Native visual backend unavailable: %s", NATIVE_IMPORT_ERROR)
        return None

    if isinstance(device, int):
        device_id = device
    else:
        device_str = str(device)
        if not device_str.startswith("cuda"):
            LOGGER.info("Native visual backend disabled for non-CUDA device '%s'", device_str)
            return None
        if ":" in device_str:
            device_id = int(device_str.split(":", 1)[1])
        else:
            device_id = 0

    try:
        return NativeVisualPromptRuntime(
            str(engine_path),
            image_input_name,
            visual_input_name,
            int(visual_stride),
            device_id,
        )
    except Exception as exc:  # pragma: no cover - depends on local TensorRT/CUDA runtime state
        LOGGER.warning("Native visual backend unavailable: %s", exc)
        return None


def build_native_jetson_camera_source(
    pipeline: str,
    *,
    prefix: str,
    timeout_s: float,
    target_h: int,
    target_w: int,
    fp16: bool,
    preview_cpu: bool = False,
    device: str | int = "cuda:0",
):
    if not NATIVE_AVAILABLE or NativeJetsonCameraSource is None:
        if NATIVE_IMPORT_ERROR is not None:
            LOGGER.info("Native Jetson camera backend unavailable: %s", NATIVE_IMPORT_ERROR)
        return None

    if isinstance(device, int):
        device_id = device
    else:
        device_str = str(device)
        if not device_str.startswith("cuda"):
            LOGGER.info("Native Jetson camera backend disabled for non-CUDA device '%s'", device_str)
            return None
        if ":" in device_str:
            device_id = int(device_str.split(":", 1)[1])
        else:
            device_id = 0

    return NativeJetsonCameraSource(
        pipeline,
        prefix,
        float(timeout_s),
        int(target_h),
        int(target_w),
        bool(fp16),
        bool(preview_cpu),
        device_id,
    )
