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


def _log_native_unavailable(message: str) -> None:
    if NATIVE_IMPORT_ERROR is not None:
        LOGGER.info("%s: %s", message, NATIVE_IMPORT_ERROR)


def resolve_cuda_device_id(device: str | int, *, component_name: str) -> int | None:
    if isinstance(device, int):
        return device
    device_str = str(device)
    if not device_str.startswith("cuda"):
        LOGGER.info("%s disabled for non-CUDA device '%s'", component_name, device_str)
        return None
    if ":" in device_str:
        return int(device_str.split(":", 1)[1])
    return 0


def _build_native_component(
    native_type: object | None,
    *,
    unavailable_message: str,
    component_name: str,
    device: str | int,
    constructor,
    swallow_errors: bool = False,
):
    if not NATIVE_AVAILABLE or native_type is None:
        _log_native_unavailable(unavailable_message)
        return None

    device_id = resolve_cuda_device_id(device, component_name=component_name)
    if device_id is None:
        return None
    try:
        return constructor(device_id)
    except Exception as exc:  # pragma: no cover - depends on local TensorRT/CUDA runtime state
        if not swallow_errors:
            raise
        LOGGER.warning("%s unavailable: %s", component_name, exc)
        return None


def build_native_main_runtime(
    engine_path: str | Path,
    image_input_name: str,
    prompt_input_name: str,
    device: str | int = "cuda:0",
):
    def _construct(device_id: int):
        return NativeMainRuntime(
            str(engine_path),
            image_input_name,
            prompt_input_name,
            device_id,
        )

    return _build_native_component(
        NativeMainRuntime,
        unavailable_message="Native backend unavailable",
        component_name="Native backend",
        device=device,
        constructor=_construct,
    )


def build_native_visual_runtime(
    engine_path: str | Path,
    image_input_name: str,
    visual_input_name: str,
    visual_stride: int,
    device: str | int = "cuda:0",
):
    return _build_native_component(
        NativeVisualPromptRuntime,
        unavailable_message="Native visual backend unavailable",
        component_name="Native visual backend",
        device=device,
        constructor=lambda device_id: NativeVisualPromptRuntime(
            str(engine_path),
            image_input_name,
            visual_input_name,
            int(visual_stride),
            device_id,
        ),
        swallow_errors=True,
    )


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
    return _build_native_component(
        NativeJetsonCameraSource,
        unavailable_message="Native Jetson camera backend unavailable",
        component_name="Native Jetson camera backend",
        device=device,
        constructor=lambda device_id: NativeJetsonCameraSource(
            pipeline,
            prefix,
            float(timeout_s),
            int(target_h),
            int(target_w),
            bool(fp16),
            bool(preview_cpu),
            device_id,
        ),
    )
