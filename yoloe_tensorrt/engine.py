from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Sequence

import numpy as np
import torch
from torch.utils.dlpack import from_dlpack
from ultralytics.engine.results import Results
from ultralytics.utils import nms, ops

from ._inference_types import PreparedExecutionInput, PreparedTensorInput, prepare_tensor_input
from ._shapes import HWShape, ImageSizeLike, normalize_imgsz
from .artifacts import ArtifactMetadata, default_artifact_dir, load_metadata, resolve_artifact_file
from .assets import text_asset_search_roots
from .inputs import (
    InferenceSourceItem,
    iter_inference_sources,
    validate_prepared_tensor,
)
from .logging_utils import get_logger
from .native_backend import (
    build_native_main_runtime,
    build_native_visual_runtime,
)
from .preprocess import preprocess_image
from .source import PreparedFrameMetadata, SourceItem, normalize_source
from .trt import TensorRTRuntime

if TYPE_CHECKING:
    from .prompts import TextPromptEncoder
    from .tracking import YOLOETrackerSession

LOGGER = get_logger(__name__)

_DEFAULT_TRACKER = "bytetrack"


@lru_cache(maxsize=1)
def _export_module():
    from . import export as export_module

    return export_module


@lru_cache(maxsize=1)
def _prompt_module():
    from . import prompts as prompt_module

    return prompt_module


@lru_cache(maxsize=1)
def _tracking_module():
    from . import tracking as tracking_module

    return tracking_module


def export_model(*args, **kwargs):
    return _export_module().export_model(*args, **kwargs)


def load_prompt_projector(*args, **kwargs):
    return _prompt_module().load_prompt_projector(*args, **kwargs)


def resolve_text_asset_path(*args, **kwargs):
    return _prompt_module().resolve_text_asset_path(*args, **kwargs)


def concat_prompt_embeddings(*args, **kwargs):
    return _prompt_module().concat_prompt_embeddings(*args, **kwargs)


class YOLOEEngine:
    def __init__(
        self,
        artifact_dir: str | Path,
        metadata: ArtifactMetadata,
        main_runtime: TensorRTRuntime | None,
        visual_runtime: TensorRTRuntime | None,
        prompt_projector: torch.jit.ScriptModule,
        device: str | torch.device = "cuda:0",
        text_encoder_path: str | Path | None = None,
        native_main_runtime: object | None = None,
        native_visual_runtime: object | None = None,
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.metadata = metadata
        self.device = torch.device(device)
        self.main_runtime = main_runtime
        self.native_main_runtime = native_main_runtime
        self.visual_runtime = visual_runtime
        self.native_visual_runtime = native_visual_runtime
        self.prompt_projector = prompt_projector
        self._text_encoder_path = Path(text_encoder_path) if text_encoder_path else None
        self._text_encoder: TextPromptEncoder | None = None
        self._text_prompt_embeddings: torch.Tensor | None = None
        self._visual_prompt_embeddings: torch.Tensor | None = None
        self._text_names: list[str] = []
        self._visual_names: list[str] = []
        self._active_names_cache: tuple[str, ...] = ()
        self._prompt_generation = 0
        self._native_infer_image = getattr(native_main_runtime, "infer_image", None)
        self._native_infer_tensor = getattr(native_main_runtime, "infer_tensor", None)
        self._native_infer_image_postprocessed = getattr(native_main_runtime, "infer_image_postprocessed", None)
        self._native_infer_tensor_postprocessed = getattr(native_main_runtime, "infer_tensor_postprocessed", None)
        if native_main_runtime is not None:
            self._main_output_names_cache: tuple[str, ...] = tuple(native_main_runtime.output_names)
        elif main_runtime is not None:
            self._main_output_names_cache = tuple(main_runtime.output_names)
        else:
            self._main_output_names_cache = ()
        LOGGER.info(
            "Initialized YOLOEEngine with artifact_dir='%s', task=%s, device=%s native=%s",
            self.artifact_dir,
            self.metadata.task,
            self.device,
            bool(self.native_main_runtime is not None),
        )

    @classmethod
    def from_pt(
        cls,
        pt_path: str | Path,
        artifact_dir: str | Path | None = None,
        build: bool = True,
        dynamic: bool = True,
        fp16: bool = True,
        text_encoder: str | Path | None = None,
        device: str | torch.device = "cuda:0",
    ) -> YOLOEEngine:
        artifact_root = Path(artifact_dir) if artifact_dir is not None else default_artifact_dir(pt_path)
        LOGGER.info("Creating YOLOEEngine from checkpoint '%s'", pt_path)
        metadata_file = artifact_root / "metadata.json"
        if build and (
            not metadata_file.exists()
            or not (artifact_root / "main.engine").exists()
            or not (artifact_root / "prompt_projector.ts").exists()
        ):
            export_model(
                pt_path,
                artifact_dir=artifact_root,
                formats=("onnx", "engine"),
                dynamic=dynamic,
                fp16=fp16,
            )
        return cls.from_engine(artifact_root, text_encoder=text_encoder, device=device)

    @classmethod
    def from_engine(
        cls,
        artifact_dir: str | Path,
        text_encoder: str | Path | None = None,
        device: str | torch.device = "cuda:0",
    ) -> YOLOEEngine:
        artifact_root = Path(artifact_dir)
        LOGGER.info("Creating YOLOEEngine from artifact bundle '%s'", artifact_root)
        metadata = load_metadata(artifact_root)
        main_engine = resolve_artifact_file(artifact_root, metadata.main_engine_filename)
        if main_engine is None or not main_engine.is_file():
            raise FileNotFoundError(f"Main TensorRT engine not found in '{artifact_root}'")
        visual_engine = resolve_artifact_file(artifact_root, metadata.visual_engine_filename)
        projector_path = resolve_artifact_file(artifact_root, metadata.prompt_projector_filename)
        if projector_path is None or not projector_path.is_file():
            raise FileNotFoundError(f"Prompt projector not found in '{artifact_root}'")

        preferred_text_path = None
        if text_encoder is not None:
            preferred_text_path = Path(text_encoder)
        else:
            preferred_text_path = resolve_text_asset_path(
                metadata.text_model,
                text_asset_search_roots(
                    [
                        artifact_root,
                        artifact_root / (metadata.text_encoder_filename or ""),
                    ]
                ),
            )

        native_main_runtime = build_native_main_runtime(
            main_engine,
            image_input_name=metadata.image_input_name,
            prompt_input_name=metadata.prompt_input_name,
            device=device,
        )
        native_visual_runtime = None
        if visual_engine and visual_engine.is_file():
            native_visual_runtime = build_native_visual_runtime(
                visual_engine,
                image_input_name=metadata.image_input_name,
                visual_input_name=metadata.visual_input_name,
                visual_stride=metadata.visual_stride,
                device=device,
            )

        return cls(
            artifact_dir=artifact_root,
            metadata=metadata,
            main_runtime=None if native_main_runtime is not None else TensorRTRuntime(main_engine, device=device),
            visual_runtime=(
                None
                if native_visual_runtime is not None or not (visual_engine and visual_engine.is_file())
                else TensorRTRuntime(visual_engine, device=device)
            ),
            prompt_projector=load_prompt_projector(projector_path, device=torch.device(device)),
            device=device,
            text_encoder_path=preferred_text_path,
            native_main_runtime=native_main_runtime,
            native_visual_runtime=native_visual_runtime,
        )

    def _get_text_encoder(self) -> TextPromptEncoder:
        if self._text_encoder is None:
            LOGGER.info("Resolving text encoder for '%s'", self.metadata.text_model)
            self._text_encoder = _prompt_module().TextPromptEncoder(
                self.metadata.text_model,
                device=self.device,
                weight_path=self._text_encoder_path,
            )
        return self._text_encoder

    def _active_prompt_embeddings(self) -> tuple[torch.Tensor, list[str]]:
        return concat_prompt_embeddings(
            self._text_prompt_embeddings,
            self._text_names,
            self._visual_prompt_embeddings,
            self._visual_names,
        )

    def _active_prompt_names(self) -> tuple[str, ...]:
        names = getattr(self, "_active_names_cache", ())
        if not names:
            names = (*self._text_names, *self._visual_names)
            if not names:
                raise RuntimeError("No prompt embeddings are active. Call set_classes() or set_visual_prompts() first.")
            self._active_names_cache = names
        return names

    @property
    def _main_output_names(self) -> tuple[str, ...]:
        cached = getattr(self, "_main_output_names_cache", ())
        if cached:
            return cached
        if self.native_main_runtime is not None:
            cached = tuple(self.native_main_runtime.output_names)
            self._main_output_names_cache = cached
            return cached
        if self.main_runtime is None:
            raise RuntimeError("Main runtime is not initialized")
        cached = tuple(self.main_runtime.output_names)
        self._main_output_names_cache = cached
        return cached

    @property
    def main_fp16(self) -> bool:
        if self.native_main_runtime is not None:
            return bool(self.native_main_runtime.fp16)
        if self.main_runtime is None:
            raise RuntimeError("Main runtime is not initialized")
        return self.main_runtime.fp16

    @property
    def prompt_generation(self) -> int:
        return int(self._prompt_generation)

    def _sync_native_prompt_embeddings(self) -> None:
        if self.native_main_runtime is None:
            return
        if not (self._text_names or self._visual_names):
            self.native_main_runtime.clear_prompt_embeddings()
            return
        embeddings, _ = self._active_prompt_embeddings()
        tensor = embeddings.detach().contiguous()
        if hasattr(self.native_main_runtime, "set_prompt_embeddings_device"):
            target_dtype = (
                torch.float16 if bool(getattr(self.native_main_runtime, "prompt_fp16", False)) else torch.float32
            )
            if tensor.dtype != target_dtype:
                tensor = tensor.to(device=self.device, dtype=target_dtype)
            producer_stream = int(torch.cuda.current_stream(self.device).cuda_stream)
            self.native_main_runtime.set_prompt_embeddings_device(
                int(tensor.data_ptr()),
                int(tensor.shape[1]),
                int(tensor.shape[2]),
                producer_stream,
            )
            return
        self.native_main_runtime.set_prompt_embeddings(tensor.float().cpu().contiguous().numpy())

    def _bump_prompt_generation(self) -> None:
        self._prompt_generation += 1
        self._active_names_cache = ()

    def _describe_source(self, source: object) -> str:
        if isinstance(source, SourceItem):
            return str(source.path)
        if isinstance(source, np.ndarray):
            return f"ndarray(shape={tuple(int(v) for v in source.shape)}, dtype={source.dtype})"
        if isinstance(source, torch.Tensor):
            return f"tensor(shape={tuple(int(v) for v in source.shape)}, dtype={source.dtype}, device={source.device})"
        return str(source)

    def _native_visual_image(self, image: np.ndarray) -> np.ndarray:
        if image.dtype == np.uint8:
            return np.ascontiguousarray(image)
        if np.issubdtype(image.dtype, np.floating):
            min_value = float(image.min())
            max_value = float(image.max())
            if 0.0 <= min_value and max_value <= 1.0:
                return np.ascontiguousarray(np.rint(image * 255.0).clip(0.0, 255.0).astype(np.uint8))
            if 0.0 <= min_value and max_value <= 255.0:
                return np.ascontiguousarray(np.rint(image).clip(0.0, 255.0).astype(np.uint8))
        raise ValueError(
            "Native visual prompts require uint8 reference images or non-negative float images in 0..1 or 0..255 range"
        )

    def clear_prompts(self) -> None:
        self._text_prompt_embeddings = None
        self._visual_prompt_embeddings = None
        self._text_names = []
        self._visual_names = []
        self._bump_prompt_generation()
        self._sync_native_prompt_embeddings()
        LOGGER.info("Cleared active text and visual prompts")

    def set_classes(self, classes: list[str]) -> None:
        resolved = [str(name) for name in classes]
        if not resolved:
            raise ValueError("classes must not be empty")
        embeddings = _prompt_module().compile_text_embeddings(
            resolved,
            encoder=self._get_text_encoder(),
            projector=self.prompt_projector,
        )
        self._text_prompt_embeddings = embeddings.to(self.device)
        self._text_names = resolved
        self._bump_prompt_generation()
        self._sync_native_prompt_embeddings()
        LOGGER.info("Activated %d text prompt class(es): %s", len(resolved), resolved)

    def set_prompt_embeddings(
        self,
        embeddings: torch.Tensor | object,
        names: list[str],
    ) -> None:
        resolved = [str(name) for name in names]
        if not resolved:
            raise ValueError("names must not be empty")
        tensor = torch.as_tensor(embeddings, dtype=torch.float32, device=self.device)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 3 or tensor.shape[0] != 1:
            raise ValueError(
                f"Expected embeddings with shape (1, N, E) or (N, E), got {tuple(int(v) for v in tensor.shape)}"
            )
        if tensor.shape[1] != len(resolved):
            raise ValueError("names must match the number of prompt embeddings")
        self._text_prompt_embeddings = tensor.contiguous()
        self._visual_prompt_embeddings = None
        self._text_names = resolved
        self._visual_names = []
        self._bump_prompt_generation()
        self._sync_native_prompt_embeddings()
        LOGGER.info("Activated %d external prompt embedding(s): %s", len(resolved), resolved)

    def set_visual_prompts(
        self,
        refer_image: object,
        bboxes: list[list[float]] | None = None,
        masks: object | None = None,
        classes: list[str] | None = None,
        imgsz: ImageSizeLike | None = None,
    ) -> None:
        if self.native_visual_runtime is None and self.visual_runtime is None:
            raise RuntimeError("This artifact bundle does not include a visual-prompt engine")
        if bboxes is None and masks is None:
            raise ValueError("Either bboxes or masks must be provided")
        LOGGER.info("Setting visual prompts from '%s'", self._describe_source(refer_image))
        source_item = normalize_source(refer_image, default_prefix="refer")[0]
        target_size = normalize_imgsz(imgsz or self.metadata.default_imgsz)
        prompt_module = _prompt_module()
        boxes = prompt_module.normalize_visual_prompt_boxes(bboxes) if bboxes is not None else None
        masks_array = None
        if boxes is not None:
            prompt_count = int(boxes.shape[0])
        elif masks is not None:
            masks_array = prompt_module.normalize_visual_prompt_masks(masks)
            prompt_count = int(masks_array.shape[0])
        else:
            prompt_count = 0
        categories, names = prompt_module.resolve_visual_prompt_categories(prompt_count, classes)

        if self.native_visual_runtime is not None:
            native_outputs = self.native_visual_runtime.infer_image(
                self._native_visual_image(source_item.image),
                int(target_size[0]),
                int(target_size[1]),
                categories,
                boxes,
                masks_array,
            )
            prompt_output_name = self.native_visual_runtime.output_names[0]
            prompt_embeddings = from_dlpack(native_outputs[prompt_output_name]).float().contiguous().clone()
        else:
            assert self.visual_runtime is not None
            sample = preprocess_image(
                source_item,
                imgsz=target_size,
                device=self.device,
                fp16=self.visual_runtime.fp16,
                stride=self.metadata.stride,
            )
            prompt_batch = prompt_module.build_visual_prompt_batch(
                image=sample.original,
                dst_shape=sample.transformed.shape[:2],
                visual_stride=self.metadata.visual_stride,
                bboxes=boxes,
                masks=masks_array,
                classes=classes,
            )
            outputs = self.visual_runtime.infer(
                {
                    self.metadata.image_input_name: sample.tensor,
                    self.metadata.visual_input_name: prompt_batch.tensor.to(self.device),
                }
            )
            prompt_output_name = self.visual_runtime.output_names[0]
            prompt_embeddings = outputs[prompt_output_name].float()
            names = prompt_batch.names
        self._visual_prompt_embeddings = prompt_embeddings
        self._visual_names = names
        self._bump_prompt_generation()
        self._sync_native_prompt_embeddings()
        LOGGER.info(
            "Activated %d visual prompt class(es): %s",
            len(self._visual_names),
            self._visual_names,
        )

    def warmup(self, imgsz: ImageSizeLike | None = None) -> None:
        target = normalize_imgsz(imgsz or self.metadata.default_imgsz)
        LOGGER.info("Warming up YOLOEEngine at imgsz=%s", target)
        prompt_count = (
            int(self._active_prompt_embeddings()[0].shape[1])
            if (self._text_prompt_embeddings is not None or self._visual_prompt_embeddings is not None)
            else int(self.metadata.prompt_profile.optimum[1])
        )
        if self.native_main_runtime is not None:
            self.native_main_runtime.warmup(target[0], target[1], prompt_count, 2)
        elif self.main_runtime is not None:
            shapes = {
                self.metadata.image_input_name: (1, 3, target[0], target[1]),
                self.metadata.prompt_input_name: (1, prompt_count, self.metadata.embed_dim),
            }
            self.main_runtime.warmup(shapes=shapes)
        if self.metadata.visual_profile is not None:
            if self.native_visual_runtime is not None:
                self.native_visual_runtime.warmup(target[0], target[1], self.metadata.visual_profile.optimum[1], 2)
            elif self.visual_runtime is not None:
                self.visual_runtime.warmup(
                    shapes={
                        self.metadata.image_input_name: (1, 3, target[0], target[1]),
                        self.metadata.visual_input_name: (
                            1,
                            self.metadata.visual_profile.optimum[1],
                            target[0] // self.metadata.visual_stride,
                            target[1] // self.metadata.visual_stride,
                        ),
                    }
                )
        LOGGER.info("YOLOEEngine warmup complete")

    @property
    def active_names(self) -> list[str]:
        if not (self._text_names or self._visual_names):
            return []
        return list(self._active_prompt_names())

    def _postprocess_predictions(
        self,
        outputs: dict[str, torch.Tensor],
        original_image: np.ndarray,
        path: str,
        input_shape: HWShape,
        names: Sequence[str],
        conf: float,
        iou: float,
        max_det: int,
        retina_masks: bool,
        speed: dict[str, float],
    ) -> Results:
        names_map = {index: name for index, name in enumerate(names)}
        pred_tensor = outputs[self._main_output_names[0]].float()
        preds = nms.non_max_suppression(
            pred_tensor,
            conf_thres=conf,
            iou_thres=iou,
            agnostic=True,
            max_det=max_det,
            nc=len(names_map),
            end2end=self.metadata.end2end,
        )[0]

        if self.metadata.task == "segment":
            proto = outputs[self._main_output_names[1]][0].float()
            if preds.shape[0] == 0:
                masks = None
            elif retina_masks:
                preds[:, :4] = ops.scale_boxes(input_shape, preds[:, :4], original_image.shape)
                masks = ops.process_mask_native(proto, preds[:, 6:], preds[:, :4], original_image.shape[:2])
            else:
                masks = ops.process_mask(proto, preds[:, 6:], preds[:, :4], input_shape, upsample=True)
                preds[:, :4] = ops.scale_boxes(input_shape, preds[:, :4], original_image.shape)
            if masks is not None:
                keep = masks.amax((-2, -1)) > 0
                if not bool(torch.all(keep)):
                    preds = preds[keep]
                    masks = masks[keep]
            return Results(original_image, path=path, names=names_map, boxes=preds[:, :6], masks=masks, speed=speed)

        preds[:, :4] = ops.scale_boxes(input_shape, preds[:, :4], original_image.shape)
        return Results(original_image, path=path, names=names_map, boxes=preds[:, :6], speed=speed)

    def _predict_one(
        self,
        item: SourceItem,
        imgsz: HWShape,
        conf: float,
        iou: float,
        max_det: int,
        retina_masks: bool,
    ) -> Results:
        if self.native_main_runtime is not None:
            prompt_embeddings = None
            names = self._active_prompt_names()
        else:
            prompt_embeddings, names = self._active_prompt_embeddings()
        LOGGER.debug(
            "Running inference on '%s' with %d active prompt class(es) at imgsz=%s",
            item.path,
            len(names),
            imgsz,
        )
        if self.native_main_runtime is not None:
            native_infer_image_postprocessed = self._native_infer_image_postprocessed
            if native_infer_image_postprocessed is not None:
                native_outputs = native_infer_image_postprocessed(
                    item.image,
                    imgsz[0],
                    imgsz[1],
                    int(item.image.shape[0]),
                    int(item.image.shape[1]),
                    float(conf),
                    float(iou),
                    int(max_det),
                    bool(retina_masks),
                )
                result = self._build_native_postprocessed_result(
                    native_outputs=native_outputs,
                    original_image=item.image,
                    path=item.path,
                    names=names,
                    preprocess_ms=float(native_outputs["preprocess_ms"]),
                    inference_ms=float(native_outputs["inference_ms"]),
                )
            else:
                native_infer_image = self._native_infer_image
                if native_infer_image is None:
                    raise RuntimeError("Native main runtime does not support image inference")
                result = self._build_native_raw_result(
                    native_outputs=native_infer_image(item.image, imgsz[0], imgsz[1]),
                    original_image=item.image,
                    path=item.path,
                    names=names,
                    conf=conf,
                    iou=iou,
                    max_det=max_det,
                    retina_masks=retina_masks,
                )
        else:
            if self.main_runtime is None:
                raise RuntimeError("Main runtime is not initialized")
            preprocess_start = time.perf_counter()
            sample = preprocess_image(
                item,
                imgsz=imgsz,
                device=self.device,
                fp16=self.main_fp16,
                stride=self.metadata.stride,
            )
            preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0

            inference_start = time.perf_counter()
            outputs = self.main_runtime.infer(
                {
                    self.metadata.image_input_name: sample.tensor,
                    self.metadata.prompt_input_name: prompt_embeddings,
                }
            )
            torch.cuda.synchronize(self.device)
            inference_ms = (time.perf_counter() - inference_start) * 1000.0
            input_shape = tuple(int(v) for v in sample.tensor.shape[2:])
            postprocess_start = time.perf_counter()
            speed = {"preprocess": preprocess_ms, "inference": inference_ms, "postprocess": 0.0}
            result = self._postprocess_predictions(
                outputs=outputs,
                original_image=sample.original,
                path=sample.path,
                input_shape=input_shape,
                names=names,
                conf=conf,
                iou=iou,
                max_det=max_det,
                retina_masks=retina_masks,
                speed=speed,
            )
            speed["postprocess"] = (time.perf_counter() - postprocess_start) * 1000.0
            result.speed = speed
        detections = int(result.boxes.data.shape[0]) if result.boxes is not None else 0
        LOGGER.debug(
            "Completed inference on '%s': detections=%d preprocess=%.1fms inference=%.1fms postprocess=%.1fms",
            item.path,
            detections,
            result.speed["preprocess"],
            result.speed["inference"],
            result.speed["postprocess"],
        )
        return result

    def _resolve_original_image_path(
        self,
        original_image: SourceItem | PreparedFrameMetadata | object | None,
        path: str | None,
        input_shape: HWShape,
    ) -> tuple[np.ndarray, str]:
        if isinstance(original_image, SourceItem):
            return original_image.image, path or original_image.path
        if isinstance(original_image, PreparedFrameMetadata):
            resolved_path = path or original_image.path or "tensor0"
            if original_image.preview_image is not None:
                return original_image.preview_image, resolved_path
            return np.zeros((*original_image.original_shape, 3), dtype=np.uint8), resolved_path
        if original_image is not None:
            item = normalize_source(original_image, default_prefix="tensor")[0]
            return item.image, path or item.path

        fallback_path = path or "tensor0"
        return np.zeros((input_shape[0], input_shape[1], 3), dtype=np.uint8), fallback_path

    def _build_native_postprocessed_result(
        self,
        native_outputs: dict[str, object],
        original_image: np.ndarray,
        path: str,
        names: Sequence[str],
        preprocess_ms: float,
        inference_ms: float,
    ) -> Results:
        speed = {
            "preprocess": preprocess_ms,
            "inference": inference_ms,
            "postprocess": float(native_outputs["postprocess_ms"]),
        }
        names_map = {index: name for index, name in enumerate(names)}
        boxes = from_dlpack(native_outputs["boxes"])
        masks = from_dlpack(native_outputs["masks"]) if "masks" in native_outputs else None
        return Results(original_image, path=path, names=names_map, boxes=boxes, masks=masks, speed=speed)

    def _build_native_raw_result(
        self,
        native_outputs: dict[str, object],
        original_image: np.ndarray,
        path: str,
        names: Sequence[str],
        conf: float,
        iou: float,
        max_det: int,
        retina_masks: bool,
        input_shape: HWShape | None = None,
    ) -> Results:
        outputs = {name: from_dlpack(native_outputs[name]) for name in self._main_output_names}
        resolved_input_shape = (
            input_shape if input_shape is not None else tuple(int(v) for v in native_outputs["input_shape"])
        )
        speed = {
            "preprocess": float(native_outputs["preprocess_ms"]),
            "inference": float(native_outputs["inference_ms"]),
            "postprocess": 0.0,
        }
        postprocess_start = time.perf_counter()
        result = self._postprocess_predictions(
            outputs=outputs,
            original_image=original_image,
            path=path,
            input_shape=resolved_input_shape,
            names=names,
            conf=conf,
            iou=iou,
            max_det=max_det,
            retina_masks=retina_masks,
            speed=speed,
        )
        speed["postprocess"] = (time.perf_counter() - postprocess_start) * 1000.0
        result.speed = speed
        return result

    def _resolve_max_det(self, max_det: int | None) -> int:
        return int(max_det or self.metadata.max_det)

    def _prepare_execution_input(self, item: PreparedTensorInput) -> PreparedExecutionInput:
        image_tensor = item.tensor
        validate_prepared_tensor(image_tensor)
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)
        if image_tensor.ndim != 4 or int(image_tensor.shape[0]) != 1 or int(image_tensor.shape[1]) != 3:
            raise ValueError(
                "Prepared tensor inputs must have shape (3, H, W) or (1, 3, H, W) before they enter the fast path"
            )

        current_stream: int | None = None
        producer_stream = item.producer_stream
        if image_tensor.device.type != "cuda":
            image_tensor = image_tensor.to(device=self.device)
            current_stream = int(torch.cuda.current_stream(self.device).cuda_stream)
            producer_stream = current_stream
        elif image_tensor.device != self.device:
            image_tensor = image_tensor.to(device=self.device)
            current_stream = int(torch.cuda.current_stream(self.device).cuda_stream)
            producer_stream = current_stream
        target_dtype = torch.float16 if self.main_fp16 else torch.float32
        if image_tensor.dtype != target_dtype:
            image_tensor = image_tensor.to(dtype=target_dtype)
            if current_stream is None:
                current_stream = int(torch.cuda.current_stream(self.device).cuda_stream)
            producer_stream = current_stream
        if not image_tensor.is_contiguous():
            image_tensor = image_tensor.contiguous()
            if current_stream is None:
                current_stream = int(torch.cuda.current_stream(self.device).cuda_stream)
            producer_stream = current_stream
        if producer_stream is None:
            if current_stream is None:
                current_stream = int(torch.cuda.current_stream(self.device).cuda_stream)
            producer_stream = current_stream
        return PreparedExecutionInput(
            tensor=image_tensor,
            input_shape=(int(image_tensor.shape[2]), int(image_tensor.shape[3])),
            producer_stream=int(producer_stream),
        )

    def _predict_prepared_tensor(
        self,
        item: PreparedTensorInput,
        conf: float,
        iou: float,
        max_det: int,
        retina_masks: bool,
    ) -> Results:
        if self.native_main_runtime is not None:
            prompt_embeddings = None
            names = self._active_prompt_names()
        else:
            prompt_embeddings, names = self._active_prompt_embeddings()

        prepared = self._prepare_execution_input(item)
        image_tensor = prepared.tensor
        input_shape = prepared.input_shape
        original_image, result_path = self._resolve_original_image_path(item.original_image, item.path, input_shape)
        if self.native_main_runtime is not None:
            native_infer_tensor_postprocessed = self._native_infer_tensor_postprocessed
            if native_infer_tensor_postprocessed is not None:
                native_outputs = native_infer_tensor_postprocessed(
                    int(image_tensor.data_ptr()),
                    input_shape[0],
                    input_shape[1],
                    int(original_image.shape[0]),
                    int(original_image.shape[1]),
                    float(conf),
                    float(iou),
                    int(max_det),
                    bool(retina_masks),
                    prepared.producer_stream,
                )
                result = self._build_native_postprocessed_result(
                    native_outputs=native_outputs,
                    original_image=original_image,
                    path=result_path,
                    names=names,
                    preprocess_ms=float(native_outputs["preprocess_ms"]),
                    inference_ms=float(native_outputs["inference_ms"]),
                )
            else:
                native_infer_tensor = self._native_infer_tensor
                if native_infer_tensor is None:
                    raise RuntimeError("Native main runtime does not support tensor inference")
                result = self._build_native_raw_result(
                    native_outputs=native_infer_tensor(
                        int(image_tensor.data_ptr()),
                        input_shape[0],
                        input_shape[1],
                        prepared.producer_stream,
                    ),
                    original_image=original_image,
                    path=result_path,
                    names=names,
                    conf=conf,
                    iou=iou,
                    max_det=max_det,
                    retina_masks=retina_masks,
                    input_shape=input_shape,
                )
        else:
            if self.main_runtime is None:
                raise RuntimeError("Main runtime is not initialized")
            inference_start = time.perf_counter()
            outputs = self.main_runtime.infer(
                {
                    self.metadata.image_input_name: image_tensor,
                    self.metadata.prompt_input_name: prompt_embeddings,
                }
            )
            torch.cuda.synchronize(self.device)
            preprocess_ms = 0.0
            inference_ms = (time.perf_counter() - inference_start) * 1000.0
            postprocess_start = time.perf_counter()
            speed = {"preprocess": preprocess_ms, "inference": inference_ms, "postprocess": 0.0}
            result = self._postprocess_predictions(
                outputs=outputs,
                original_image=original_image,
                path=result_path,
                input_shape=input_shape,
                names=names,
                conf=conf,
                iou=iou,
                max_det=max_det,
                retina_masks=retina_masks,
                speed=speed,
            )
            speed["postprocess"] = (time.perf_counter() - postprocess_start) * 1000.0
            result.speed = speed
        LOGGER.debug(
            "Completed prepared-tensor inference on '%s': input_shape=%s detections=%d "
            "pre=%.1fms inf=%.1fms post=%.1fms",
            result_path,
            input_shape,
            int(result.boxes.data.shape[0]) if result.boxes is not None else 0,
            result.speed["preprocess"],
            result.speed["inference"],
            result.speed["postprocess"],
        )
        return result

    def _predict_input_entry(
        self,
        item: InferenceSourceItem,
        imgsz: ImageSizeLike | None,
        conf: float,
        iou: float,
        max_det: int,
        retina_masks: bool,
    ) -> Results:
        if isinstance(item, PreparedTensorInput):
            return self._predict_prepared_tensor(
                item=item,
                conf=conf,
                iou=iou,
                max_det=max_det,
                retina_masks=retina_masks,
            )
        target_size = normalize_imgsz(imgsz or self.metadata.default_imgsz)
        return self._predict_one(
            item=item,
            imgsz=target_size,
            conf=conf,
            iou=iou,
            max_det=max_det,
            retina_masks=retina_masks,
        )

    def predict_item(
        self,
        item: InferenceSourceItem,
        imgsz: ImageSizeLike | None = None,
        conf: float = 0.25,
        iou: float = 0.45,
        max_det: int | None = None,
        retina_masks: bool = False,
    ) -> Results:
        resolved_max_det = self._resolve_max_det(max_det)
        return self._predict_input_entry(
            item=item,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            max_det=resolved_max_det,
            retina_masks=retina_masks,
        )

    def prepare_cuda_input(
        self,
        source: SourceItem | object,
        imgsz: ImageSizeLike | None = None,
    ) -> PreparedTensorInput:
        item = source if isinstance(source, SourceItem) else normalize_source(source, default_prefix="tensor")[0]
        target_size = normalize_imgsz(imgsz or self.metadata.default_imgsz)
        sample = preprocess_image(
            item,
            imgsz=target_size,
            device=self.device,
            fp16=self.main_fp16,
            stride=self.metadata.stride,
        )
        return prepare_tensor_input(
            sample.tensor,
            path=item.path,
            original_image=item,
        )

    def predict_cuda(
        self,
        tensor: torch.Tensor | object,
        *,
        original_image: SourceItem | object | None = None,
        path: str | None = None,
        conf: float = 0.25,
        iou: float = 0.45,
        max_det: int | None = None,
        retina_masks: bool = False,
    ) -> Results:
        resolved_max_det = self._resolve_max_det(max_det)
        prepared = prepare_tensor_input(
            tensor,
            path=path or "tensor0",
            original_image=original_image,
        )
        return self._predict_prepared_tensor(
            item=prepared,
            conf=conf,
            iou=iou,
            max_det=resolved_max_det,
            retina_masks=retina_masks,
        )

    def _predict_iter(
        self,
        source: object,
        imgsz: HWShape,
        conf: float,
        iou: float,
        max_det: int,
        retina_masks: bool,
        cuda: bool | None,
        input_hint: str | None,
        original_image: SourceItem | object | None,
        path: str | None,
        allow_unbounded_live: bool = True,
    ) -> Iterator[Results]:
        for item in iter_inference_sources(
            source,
            default_prefix="frame",
            input_hint=input_hint,
            cuda=cuda,
            allow_unbounded_live=allow_unbounded_live,
            original_image=original_image,
            path=path,
        ):
            yield self._predict_input_entry(
                item=item,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                max_det=max_det,
                retina_masks=retina_masks,
            )

    def create_tracker(
        self,
        tracker: str = _DEFAULT_TRACKER,
        tracker_config: str | Path | dict | None = None,
        frame_rate: int = 30,
    ) -> YOLOETrackerSession:
        return _tracking_module().YOLOETrackerSession(
            self,
            tracker=tracker,
            tracker_config=tracker_config,
            frame_rate=frame_rate,
        )

    def _track_iter(
        self,
        source: object,
        imgsz: HWShape,
        conf: float,
        iou: float,
        max_det: int,
        retina_masks: bool,
        tracker: str,
        tracker_config: str | Path | dict | None,
        frame_rate: int,
        cuda: bool | None,
        input_hint: str | None,
        original_image: SourceItem | object | None,
        path: str | None,
        allow_unbounded_live: bool = True,
    ) -> Iterator[Results]:
        session = self.create_tracker(
            tracker=tracker,
            tracker_config=tracker_config,
            frame_rate=frame_rate,
        )
        for item in iter_inference_sources(
            source,
            default_prefix="frame",
            input_hint=input_hint,
            cuda=cuda,
            allow_unbounded_live=allow_unbounded_live,
            original_image=original_image,
            path=path,
        ):
            yield session.update(
                item,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                max_det=max_det,
                retina_masks=retina_masks,
            )

    def predict(
        self,
        source: object,
        stream: bool = False,
        imgsz: ImageSizeLike | None = None,
        conf: float = 0.25,
        iou: float = 0.45,
        max_det: int | None = None,
        retina_masks: bool = False,
        cuda: bool | None = None,
        input_hint: str | None = None,
        original_image: SourceItem | object | None = None,
        path: str | None = None,
    ) -> list[Results] | Iterator[Results]:
        target_size = normalize_imgsz(imgsz or self.metadata.default_imgsz)
        resolved_max_det = self._resolve_max_det(max_det)
        LOGGER.debug(
            "Starting predict(stream=%s, imgsz=%s, conf=%.3f, iou=%.3f, max_det=%d)",
            stream,
            target_size,
            conf,
            iou,
            resolved_max_det,
        )
        results = self._predict_iter(
            source=source,
            imgsz=target_size,
            conf=conf,
            iou=iou,
            max_det=resolved_max_det,
            retina_masks=retina_masks,
            cuda=cuda,
            input_hint=input_hint,
            original_image=original_image,
            path=path,
            allow_unbounded_live=stream,
        )
        if stream:
            return results
        return list(results)

    def track(
        self,
        source: object,
        stream: bool = False,
        imgsz: ImageSizeLike | None = None,
        conf: float = 0.25,
        iou: float = 0.45,
        max_det: int | None = None,
        retina_masks: bool = False,
        tracker: str = _DEFAULT_TRACKER,
        tracker_config: str | Path | dict | None = None,
        frame_rate: int = 30,
        cuda: bool | None = None,
        input_hint: str | None = None,
        original_image: SourceItem | object | None = None,
        path: str | None = None,
    ) -> list[Results] | Iterator[Results]:
        target_size = normalize_imgsz(imgsz or self.metadata.default_imgsz)
        resolved_max_det = self._resolve_max_det(max_det)
        LOGGER.debug(
            "Starting track(stream=%s, imgsz=%s, conf=%.3f, iou=%.3f, max_det=%d, tracker=%s)",
            stream,
            target_size,
            conf,
            iou,
            resolved_max_det,
            tracker,
        )
        results = self._track_iter(
            source=source,
            imgsz=target_size,
            conf=conf,
            iou=iou,
            max_det=resolved_max_det,
            retina_masks=retina_masks,
            tracker=tracker,
            tracker_config=tracker_config,
            frame_rate=frame_rate,
            cuda=cuda,
            input_hint=input_hint,
            original_image=original_image,
            path=path,
            allow_unbounded_live=stream,
        )
        if stream:
            return results
        return list(results)

    __call__ = predict

    @staticmethod
    def export(*args, **kwargs) -> Path:
        return export_model(*args, **kwargs)
