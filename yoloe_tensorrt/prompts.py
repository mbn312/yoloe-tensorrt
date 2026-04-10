from __future__ import annotations

import copy
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.data.augment import LetterBox, LoadVisualPrompt
from ultralytics.nn.text_model import MobileCLIP, MobileCLIPTS, build_text_model

from .assets import download_text_asset, text_asset_cache_dir, text_asset_search_roots
from .logging_utils import get_logger
from .tensor_utils import torch_from_numpy_safe

LOGGER = get_logger(__name__)


def default_text_asset_name(text_model: str) -> str | None:
    base, _ = text_model.split(":", 1)
    if base == "mobileclip":
        return "mobileclip_blt.ts"
    if base == "mobileclip2":
        return "mobileclip2_b.ts"
    return None


def resolve_text_asset_path(
    text_model: str,
    candidates: Iterable[str | Path],
    *,
    allow_download: bool = True,
) -> Path | None:
    asset_name = default_text_asset_name(text_model)
    if asset_name is None:
        return None
    for candidate in candidates:
        path = Path(candidate)
        if path.is_file():
            return path
        if path.is_dir():
            nested = path / asset_name
            if nested.is_file():
                return nested
    if allow_download:
        try:
            return download_text_asset(asset_name)
        except FileNotFoundError as exc:
            LOGGER.warning("Unable to auto-download text encoder asset '%s': %s", asset_name, exc)
    return None


@contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _build_mobileclip_fallback(size: str, device: torch.device):
    cache_dir = text_asset_cache_dir()
    LOGGER.info("Initializing Apple MobileCLIP fallback '%s' with asset cache '%s'", size, cache_dir)
    with _working_directory(cache_dir):
        return MobileCLIP(size=size, device=device)


class PromptProjector(nn.Module):
    def __init__(self, reprta: nn.Module) -> None:
        super().__init__()
        self.reprta = copy.deepcopy(reprta).eval()

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.reprta(embeddings), dim=-1, p=2)


def save_prompt_projector(reprta: nn.Module, output_path: str | Path, embed_dim: int) -> Path:
    projector = PromptProjector(reprta).eval()
    example = torch.randn(1, 4, embed_dim, dtype=torch.float32)
    traced = torch.jit.trace(projector, example)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(path))
    return path


def load_prompt_projector(projector_path: str | Path, device: torch.device) -> torch.jit.ScriptModule:
    projector = torch.jit.load(str(projector_path), map_location=device)
    projector.eval()
    return projector


class TextPromptEncoder:
    def __init__(self, variant: str, device: torch.device, weight_path: str | Path | None = None) -> None:
        parts = variant.split(":", 1)
        base = parts[0]
        size = parts[1] if len(parts) == 2 else ("b" if base == "mobileclip2" else "blt")
        LOGGER.info("Initializing text prompt encoder '%s' on device '%s'", variant, device)
        if base in {"mobileclip", "mobileclip2"}:
            if weight_path is not None:
                weight = str(weight_path)
                self.model = MobileCLIPTS(device=device, weight=weight)
                LOGGER.info("Using MobileCLIP TorchScript weights from '%s'", weight)
            else:
                resolved_weight = resolve_text_asset_path(variant, text_asset_search_roots(), allow_download=True)
                if resolved_weight is not None:
                    weight = str(resolved_weight)
                    self.model = MobileCLIPTS(device=device, weight=weight)
                    LOGGER.info("Using auto-resolved MobileCLIP TorchScript weights from '%s'", weight)
                elif base == "mobileclip2":
                    LOGGER.warning(
                        "No mobileclip2 TorchScript weights were found or downloadable; "
                        "falling back to Apple MobileCLIP 'b'. Prompt compilation may be slower on the first use."
                    )
                    self.model = _build_mobileclip_fallback("b", device=device)
                else:
                    LOGGER.warning(
                        "No MobileCLIP TorchScript weights were found or downloadable for '%s'; "
                        "falling back to Apple MobileCLIP '%s'. Prompt compilation may be slower on the first use.",
                        variant,
                        size,
                    )
                    self.model = _build_mobileclip_fallback(size, device=device)
        else:
            self.model = build_text_model(variant, device=device)
        self.device = device

    @torch.inference_mode()
    def encode(self, texts: list[str]) -> torch.Tensor:
        tokens = self.model.tokenize(texts)
        embeddings = self.model.encode_text(tokens).float()
        if embeddings.ndim == 2:
            embeddings = embeddings.unsqueeze(0)
        return embeddings.to(self.device)


def compile_text_embeddings(
    classes: list[str],
    encoder: TextPromptEncoder,
    projector: torch.jit.ScriptModule,
) -> torch.Tensor:
    if not classes:
        raise ValueError("classes must not be empty")
    LOGGER.info("Compiling text prompt embeddings for %d class(es): %s", len(classes), classes)
    raw_embeddings = encoder.encode(classes).detach().clone()
    projected = projector(raw_embeddings)
    LOGGER.info("Compiled text prompt embeddings with shape %s", tuple(int(v) for v in projected.shape))
    return projected.float()


@dataclass(frozen=True)
class VisualPromptBatch:
    tensor: torch.Tensor
    names: list[str]


def _preserve_unique_names(names: list[str]) -> tuple[np.ndarray, list[str]]:
    ids: list[int] = []
    resolved: list[str] = []
    lookup: dict[str, int] = {}
    for name in names:
        key = str(name)
        if key not in lookup:
            lookup[key] = len(lookup)
            resolved.append(key)
        ids.append(lookup[key])
    return np.asarray(ids, dtype=np.int32), resolved


def resolve_visual_prompt_categories(
    prompt_count: int,
    classes: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    if prompt_count <= 0:
        raise ValueError("visual prompts must contain at least one prompt")
    if classes is None:
        names = [f"object{i}" for i in range(prompt_count)]
        return np.arange(prompt_count, dtype=np.int32), names
    if len(classes) != prompt_count:
        raise ValueError("classes must match the number of visual prompts")
    return _preserve_unique_names(classes)


def normalize_visual_prompt_boxes(bboxes: np.ndarray | list[list[float]]) -> np.ndarray:
    boxes = np.asarray(bboxes, dtype=np.float32)
    if boxes.ndim == 1:
        boxes = boxes[None, :]
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"Expected bboxes with shape (N, 4), got {boxes.shape}")
    if boxes.shape[0] == 0:
        raise ValueError("bboxes must not be empty")
    return boxes


def normalize_visual_prompt_masks(
    masks: np.ndarray | list[np.ndarray] | torch.Tensor,
) -> np.ndarray:
    if isinstance(masks, torch.Tensor):
        masks = masks.detach().cpu().numpy()
    masks_array = np.asarray(masks)
    if masks_array.ndim == 2:
        masks_array = masks_array[None, ...]
    if masks_array.ndim != 3:
        raise ValueError(f"Expected masks with shape (N, H, W), got {masks_array.shape}")
    if masks_array.shape[0] == 0:
        raise ValueError("masks must not be empty")
    return masks_array


def _resize_masks(
    masks: np.ndarray | list[np.ndarray] | torch.Tensor,
    dst_shape: tuple[int, int],
) -> np.ndarray:
    masks_array = normalize_visual_prompt_masks(masks)
    letterbox = LetterBox(
        new_shape=dst_shape,
        auto=False,
        center=True,
        padding_value=0,
        interpolation=cv2.INTER_NEAREST,
    )
    resized = []
    for mask in masks_array:
        mask_array = np.asarray(mask, dtype=np.uint8)
        if mask_array.ndim == 2:
            mask_array = mask_array[..., None]
        output = letterbox(image=mask_array)
        if output.ndim == 3 and output.shape[-1] == 1:
            output = output[..., 0]
        resized.append(output)
    return np.stack(resized, axis=0)


def build_visual_prompt_batch(
    image: np.ndarray,
    dst_shape: tuple[int, int],
    visual_stride: int,
    bboxes: np.ndarray | list[list[float]] | None = None,
    masks: np.ndarray | list[np.ndarray] | torch.Tensor | None = None,
    classes: list[str] | None = None,
) -> VisualPromptBatch:
    if bboxes is None and masks is None:
        raise ValueError("Either bboxes or masks must be provided")
    LOGGER.info(
        "Building visual prompt batch for %d prompt(s) at dst_shape=%s",
        len(bboxes) if bboxes is not None else (len(masks) if masks is not None else 0),
        dst_shape,
    )

    if bboxes is not None:
        boxes = normalize_visual_prompt_boxes(bboxes)
        prompt_count = boxes.shape[0]
        category, names = resolve_visual_prompt_categories(prompt_count, classes)
        src_shape = image.shape[:2]
        gain = min(dst_shape[0] / src_shape[0], dst_shape[1] / src_shape[1])
        boxes = boxes.copy()
        boxes *= gain
        boxes[..., 0::2] += round((dst_shape[1] - round(src_shape[1] * gain)) / 2 - 0.1)
        boxes[..., 1::2] += round((dst_shape[0] - round(src_shape[0] * gain)) / 2 - 0.1)
        visuals = LoadVisualPrompt(scale_factor=1 / visual_stride).get_visuals(
            category,
            dst_shape,
            bboxes=torch_from_numpy_safe(boxes.astype(np.float32)),
        )
    else:
        resized_masks = _resize_masks(masks, dst_shape)
        prompt_count = resized_masks.shape[0]
        category, names = resolve_visual_prompt_categories(prompt_count, classes)
        visuals = LoadVisualPrompt(scale_factor=1 / visual_stride).get_visuals(
            category,
            dst_shape,
            masks=torch_from_numpy_safe(resized_masks.astype(np.float32)),
        )
    batch = VisualPromptBatch(tensor=visuals.unsqueeze(0).float(), names=names)
    LOGGER.info(
        "Built visual prompt batch with shape %s and names=%s", tuple(int(v) for v in batch.tensor.shape), batch.names
    )
    return batch


def concat_prompt_embeddings(
    text_embeddings: torch.Tensor | None,
    text_names: list[str] | None,
    visual_embeddings: torch.Tensor | None,
    visual_names: list[str] | None,
) -> tuple[torch.Tensor, list[str]]:
    tensors: list[torch.Tensor] = []
    names: list[str] = []
    if text_embeddings is not None:
        tensors.append(text_embeddings)
        names.extend(text_names or [])
    if visual_embeddings is not None:
        tensors.append(visual_embeddings)
        names.extend(visual_names or [])
    if not tensors:
        raise RuntimeError("No prompt embeddings are active. Call set_classes() or set_visual_prompts() first.")
    return torch.cat(tensors, dim=1), names
