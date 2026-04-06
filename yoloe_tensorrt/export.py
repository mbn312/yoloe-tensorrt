from __future__ import annotations

import gc
import inspect
import shutil
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
from ultralytics.nn.modules import Detect

from .artifacts import ArtifactMetadata, ShapeProfile, default_artifact_dir, save_metadata
from .assets import resolve_model_checkpoint, text_asset_search_roots
from .logging_utils import get_logger
from .preprocess import normalize_imgsz
from .prompts import default_text_asset_name, resolve_text_asset_path, save_prompt_projector
from .trt import build_engine_from_onnx

IMAGE_INPUT_NAME = "images"
PROMPT_INPUT_NAME = "prompt_embeddings"
VISUAL_INPUT_NAME = "visual_prompts"
MAIN_ONNX_FILENAME = "main.onnx"
MAIN_ENGINE_FILENAME = "main.engine"
VISUAL_ONNX_FILENAME = "visual_prompt.onnx"
VISUAL_ENGINE_FILENAME = "visual_prompt.engine"
PROMPT_PROJECTOR_FILENAME = "prompt_projector.ts"

LOGGER = get_logger(__name__)


def _load_yoloe_model(pt_path: str | Path) -> torch.nn.Module:
    LOGGER.info("Loading YOLOE checkpoint from '%s'", pt_path)
    checkpoint = torch.load(pt_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        model = checkpoint.get("model")
    else:
        model = checkpoint
    if model is None:
        raise RuntimeError(f"Unable to load YOLOE model from '{pt_path}'")
    model = model.float().eval()
    if hasattr(model, "fuse"):
        model = model.fuse()
    LOGGER.info("Loaded YOLOE model '%s'", getattr(model, "task", "unknown"))
    return model


def _configure_export_state(model: torch.nn.Module, imgsz: tuple[int, int], dynamic: bool, max_det: int) -> None:
    anchors = sum(int(imgsz[0] / stride) * int(imgsz[1] / stride) for stride in model.stride.tolist())
    for module in model.modules():
        if isinstance(module, Detect):
            module.dynamic = dynamic
            module.export = True
            module.format = "onnx"
            module.max_det = min(max_det, anchors)
            module.agnostic_nms = True
            module.shape = None


def _flatten_export_outputs(outputs: object) -> tuple[torch.Tensor, ...]:
    if isinstance(outputs, torch.Tensor):
        return (outputs,)
    if isinstance(outputs, (list, tuple)):
        flattened: list[torch.Tensor] = []
        for item in outputs:
            flattened.extend(_flatten_export_outputs(item))
        return tuple(flattened)
    raise TypeError(f"Unsupported export output type: {type(outputs)!r}")


class _MainPromptWrapper(nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor, prompt_embeddings: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return _flatten_export_outputs(self.model(images, tpe=prompt_embeddings))


class _VisualPromptWrapper(nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor, visual_prompts: torch.Tensor) -> torch.Tensor:
        return self.model(images, vpe=visual_prompts, return_vpe=True)


def _save_text_encoder_asset(text_model: str, pt_path: str | Path, artifact_dir: str | Path) -> str | None:
    asset_name = default_text_asset_name(text_model)
    if asset_name is None:
        return None
    resolved = resolve_text_asset_path(
        text_model,
        text_asset_search_roots(
            [
                Path(pt_path).parent / asset_name,
                Path(pt_path).parent,
            ]
        ),
    )
    if resolved is None:
        LOGGER.warning(
            "No local text encoder asset was found for '%s'; runtime may need to use a slower prompt-encoder fallback",
            text_model,
        )
        return None
    destination = Path(artifact_dir) / resolved.name
    if resolved.resolve() != destination.resolve():
        shutil.copy2(resolved, destination)
        LOGGER.info("Copied text encoder asset to '%s'", destination)
    else:
        LOGGER.info("Using local text encoder asset at '%s'", destination)
    return destination.name


def _resolved_model_path(pt_path: str | Path) -> Path:
    resolved = resolve_model_checkpoint(pt_path)
    LOGGER.debug("Resolved model checkpoint '%s' -> '%s'", pt_path, resolved)
    return resolved


def _make_image_profile(dynamic: bool, imgsz: tuple[int, int]) -> ShapeProfile:
    if dynamic:
        min_h = min(320, imgsz[0])
        min_w = min(320, imgsz[1])
        max_h = max(1280, imgsz[0])
        max_w = max(1280, imgsz[1])
        return ShapeProfile((1, 3, min_h, min_w), (1, 3, imgsz[0], imgsz[1]), (1, 3, max_h, max_w))
    return ShapeProfile((1, 3, imgsz[0], imgsz[1]), (1, 3, imgsz[0], imgsz[1]), (1, 3, imgsz[0], imgsz[1]))


def _make_prompt_profile(embed_dim: int) -> ShapeProfile:
    return ShapeProfile((1, 1, embed_dim), (1, 80, embed_dim), (1, 256, embed_dim))


def _make_visual_profile(
    dynamic: bool, prompt_profile: ShapeProfile, image_profile: ShapeProfile, visual_stride: int
) -> ShapeProfile:
    if dynamic:
        return ShapeProfile(
            (
                1,
                prompt_profile.minimum[1],
                image_profile.minimum[2] // visual_stride,
                image_profile.minimum[3] // visual_stride,
            ),
            (
                1,
                prompt_profile.optimum[1],
                image_profile.optimum[2] // visual_stride,
                image_profile.optimum[3] // visual_stride,
            ),
            (
                1,
                prompt_profile.maximum[1],
                image_profile.maximum[2] // visual_stride,
                image_profile.maximum[3] // visual_stride,
            ),
        )
    return ShapeProfile(
        (
            1,
            prompt_profile.minimum[1],
            image_profile.optimum[2] // visual_stride,
            image_profile.optimum[3] // visual_stride,
        ),
        (
            1,
            prompt_profile.optimum[1],
            image_profile.optimum[2] // visual_stride,
            image_profile.optimum[3] // visual_stride,
        ),
        (
            1,
            prompt_profile.maximum[1],
            image_profile.optimum[2] // visual_stride,
            image_profile.optimum[3] // visual_stride,
        ),
    )


def _export_onnx(
    module: nn.Module,
    args: tuple[torch.Tensor, ...],
    output_path: str | Path,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict[str, dict[int, str]],
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Exporting ONNX graph to '%s'", path)
    export_kwargs = {
        "opset_version": 17,
        "input_names": input_names,
        "output_names": output_names,
        "dynamic_axes": dynamic_axes or None,
    }
    try:
        export_signature = inspect.signature(torch.onnx.export)
    except (TypeError, ValueError):  # pragma: no cover - depends on torch runtime internals
        export_signature = None
    if export_signature is not None and "dynamo" in export_signature.parameters:
        # Newer torch.onnx defaults to the torch.export/dynamo-based exporter, but the Ultralytics YOLOE
        # graph mutates attributes during tracing and still exports more reliably through the legacy path.
        export_kwargs["dynamo"] = False
        LOGGER.info("Using legacy torch.onnx exporter path (dynamo=False) for YOLOE ONNX export compatibility")
    torch.onnx.export(module, args, str(path), **export_kwargs)
    return path


def export_model(
    pt_path: str | Path,
    artifact_dir: str | Path | None = None,
    formats: Iterable[str] = ("onnx", "engine"),
    dynamic: bool = True,
    build_visual_engine: bool = True,
    fp16: bool = True,
    imgsz: int | tuple[int, int] | list[int] | None = None,
    max_det: int = 300,
    overwrite: bool = True,
    workspace_bytes: int = 2 << 30,
) -> Path:
    model_path = _resolved_model_path(pt_path)
    artifact_root = Path(artifact_dir) if artifact_dir is not None else default_artifact_dir(model_path)
    artifact_root.mkdir(parents=True, exist_ok=True)
    requested_formats = {fmt.lower() for fmt in formats}
    main_onnx_path = artifact_root / MAIN_ONNX_FILENAME
    main_engine_path = artifact_root / MAIN_ENGINE_FILENAME
    visual_onnx_path = artifact_root / VISUAL_ONNX_FILENAME
    visual_engine_path = artifact_root / VISUAL_ENGINE_FILENAME
    projector_path = artifact_root / PROMPT_PROJECTOR_FILENAME
    LOGGER.info(
        "Starting export for '%s' -> '%s' (formats=%s, dynamic=%s, fp16=%s, build_visual_engine=%s)",
        model_path,
        artifact_root,
        tuple(requested_formats),
        dynamic,
        fp16,
        build_visual_engine,
    )

    if overwrite:
        for filename in (
            main_onnx_path.name,
            main_engine_path.name,
            visual_onnx_path.name,
            visual_engine_path.name,
            projector_path.name,
            "metadata.json",
        ):
            file_path = artifact_root / filename
            if file_path.exists():
                file_path.unlink()
                LOGGER.debug("Removed stale artifact '%s'", file_path)

    model = _load_yoloe_model(model_path)
    default_imgsz = normalize_imgsz(imgsz or getattr(model, "args", {}).get("imgsz", 640))
    _configure_export_state(model, default_imgsz, dynamic=dynamic, max_det=max_det)

    task = getattr(model, "task", "detect")
    end2end = bool(getattr(model, "end2end", False))
    stride = int(max(int(v) for v in model.stride.tolist()))
    visual_stride = 8
    head = model.model[-1]
    embed_dim = int(getattr(head, "embed", 512))
    mask_dim = int(getattr(head, "nm", 0)) if task == "segment" else None
    text_model = model.yaml.get("text_model", "mobileclip:blt")
    LOGGER.info(
        "Prepared export state: task=%s, imgsz=%s, embed_dim=%d, mask_dim=%s, text_model=%s",
        task,
        default_imgsz,
        embed_dim,
        mask_dim,
        text_model,
    )

    image_profile = _make_image_profile(dynamic=dynamic, imgsz=default_imgsz)
    prompt_profile = _make_prompt_profile(embed_dim=embed_dim)
    visual_profile = _make_visual_profile(
        dynamic=dynamic, prompt_profile=prompt_profile, image_profile=image_profile, visual_stride=visual_stride
    )

    text_encoder_filename = _save_text_encoder_asset(text_model, model_path, artifact_root)
    if overwrite or not projector_path.is_file():
        save_prompt_projector(head.reprta, projector_path, embed_dim=embed_dim)
        LOGGER.info("Saved prompt projector to '%s'", projector_path)
    else:
        LOGGER.info("Reusing existing prompt projector at '%s'", projector_path)

    image_example = torch.randn(image_profile.optimum, dtype=torch.float32)
    prompt_example = torch.randn(prompt_profile.optimum, dtype=torch.float32)
    visual_example = torch.randn(visual_profile.optimum, dtype=torch.float32)

    main_wrapper = _MainPromptWrapper(model).eval()
    main_output_names = ["output0", "output1"] if task == "segment" else ["output0"]
    main_dynamic_axes = {
        PROMPT_INPUT_NAME: {0: "batch", 1: "num_prompts"},
        "output0": {0: "batch"},
    }
    if dynamic:
        main_dynamic_axes[IMAGE_INPUT_NAME] = {0: "batch", 2: "height", 3: "width"}
    if task == "segment":
        main_dynamic_axes["output1"] = {0: "batch"}
        if dynamic:
            main_dynamic_axes["output1"].update({2: "mask_height", 3: "mask_width"})
    if overwrite or not main_onnx_path.is_file():
        _export_onnx(
            main_wrapper,
            (image_example, prompt_example),
            main_onnx_path,
            input_names=[IMAGE_INPUT_NAME, PROMPT_INPUT_NAME],
            output_names=main_output_names,
            dynamic_axes=main_dynamic_axes,
        )
    else:
        LOGGER.info("Reusing existing main ONNX graph at '%s'", main_onnx_path)

    visual_onnx_filename: str | None = None
    visual_engine_filename: str | None = None
    visual_wrapper: nn.Module | None = None
    if build_visual_engine:
        visual_wrapper = _VisualPromptWrapper(model).eval()
        visual_onnx_filename = VISUAL_ONNX_FILENAME
        if overwrite or not visual_onnx_path.is_file():
            _export_onnx(
                visual_wrapper,
                (image_example, visual_example),
                visual_onnx_path,
                input_names=[IMAGE_INPUT_NAME, VISUAL_INPUT_NAME],
                output_names=["prompt_embeddings"],
                dynamic_axes=(
                    {
                        IMAGE_INPUT_NAME: {0: "batch", 2: "height", 3: "width"},
                        VISUAL_INPUT_NAME: {0: "batch", 1: "num_prompts", 2: "prompt_height", 3: "prompt_width"},
                        "prompt_embeddings": {0: "batch", 1: "num_prompts"},
                    }
                    if dynamic
                    else {
                        VISUAL_INPUT_NAME: {0: "batch", 1: "num_prompts"},
                        "prompt_embeddings": {0: "batch", 1: "num_prompts"},
                    }
                ),
            )
            LOGGER.info("Exported visual prompt ONNX graph to '%s'", visual_onnx_path)
        else:
            LOGGER.info("Reusing existing visual prompt ONNX graph at '%s'", visual_onnx_path)

    model = None
    head = None
    main_wrapper = None
    visual_wrapper = None
    image_example = None
    prompt_example = None
    visual_example = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    main_engine_filename: str | None = None
    if "engine" in requested_formats:
        if overwrite or not main_engine_path.is_file():
            LOGGER.info("Building main TensorRT engine from '%s'", main_onnx_path)
            build_engine_from_onnx(
                main_onnx_path,
                main_engine_path,
                profiles={
                    IMAGE_INPUT_NAME: image_profile,
                    PROMPT_INPUT_NAME: prompt_profile,
                },
                fp16=fp16,
                workspace_bytes=workspace_bytes,
            )
            LOGGER.info("Built main TensorRT engine at '%s'", main_engine_path)
        else:
            LOGGER.info("Reusing existing main TensorRT engine at '%s'", main_engine_path)
        if main_engine_path.is_file():
            main_engine_filename = MAIN_ENGINE_FILENAME
        if build_visual_engine and visual_onnx_filename is not None:
            if overwrite or not visual_engine_path.is_file():
                LOGGER.info("Building visual prompt TensorRT engine from '%s'", visual_onnx_path)
                build_engine_from_onnx(
                    visual_onnx_path,
                    visual_engine_path,
                    profiles={
                        IMAGE_INPUT_NAME: image_profile,
                        VISUAL_INPUT_NAME: visual_profile,
                    },
                    fp16=fp16,
                    workspace_bytes=workspace_bytes,
                )
                LOGGER.info("Built visual prompt TensorRT engine at '%s'", visual_engine_path)
            else:
                LOGGER.info("Reusing existing visual prompt TensorRT engine at '%s'", visual_engine_path)
            if visual_engine_path.is_file():
                visual_engine_filename = VISUAL_ENGINE_FILENAME

    metadata = ArtifactMetadata(
        version=1,
        model_path=str(model_path),
        model_name=model_path.name,
        task=task,
        end2end=end2end,
        dynamic=dynamic,
        default_imgsz=default_imgsz,
        stride=stride,
        visual_stride=visual_stride,
        embed_dim=embed_dim,
        mask_dim=mask_dim,
        max_det=max_det,
        fp16=fp16,
        prompt_input_name=PROMPT_INPUT_NAME,
        visual_input_name=VISUAL_INPUT_NAME,
        image_input_name=IMAGE_INPUT_NAME,
        text_model=text_model,
        text_encoder_filename=text_encoder_filename,
        prompt_projector_filename=PROMPT_PROJECTOR_FILENAME,
        main_onnx_filename=MAIN_ONNX_FILENAME,
        main_engine_filename=main_engine_filename,
        visual_onnx_filename=visual_onnx_filename,
        visual_engine_filename=visual_engine_filename,
        image_profile=image_profile,
        prompt_profile=prompt_profile,
        visual_profile=visual_profile if build_visual_engine else None,
    )
    save_metadata(artifact_root, metadata)
    LOGGER.info("Saved artifact metadata to '%s'", artifact_root / "metadata.json")
    LOGGER.info("Export complete for '%s'", artifact_root)
    return artifact_root
