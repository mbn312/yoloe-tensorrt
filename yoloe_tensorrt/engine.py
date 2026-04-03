from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import torch
from torch.utils.dlpack import from_dlpack
from ultralytics.engine.results import Results
from ultralytics.utils import nms, ops

from .artifacts import ArtifactMetadata, default_artifact_dir, load_metadata, resolve_artifact_file
from .assets import text_asset_search_roots
from .export import export_model
from .logging_utils import get_logger
from .native_backend import build_native_main_runtime
from .preprocess import normalize_imgsz, preprocess_image
from .prompts import (
    TextPromptEncoder,
    build_visual_prompt_batch,
    compile_text_embeddings,
    concat_prompt_embeddings,
    load_prompt_projector,
    resolve_text_asset_path,
)
from .source import SourceItem, is_finite_live_source, is_live_source, normalize_source, stream_sources
from .trt import TensorRTRuntime

LOGGER = get_logger(__name__)


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
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.metadata = metadata
        self.device = torch.device(device)
        self.main_runtime = main_runtime
        self.native_main_runtime = native_main_runtime
        self.visual_runtime = visual_runtime
        self.prompt_projector = prompt_projector
        self._text_encoder_path = Path(text_encoder_path) if text_encoder_path else None
        self._text_encoder: TextPromptEncoder | None = None
        self._text_prompt_embeddings: torch.Tensor | None = None
        self._visual_prompt_embeddings: torch.Tensor | None = None
        self._text_names: list[str] = []
        self._visual_names: list[str] = []
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

        return cls(
            artifact_dir=artifact_root,
            metadata=metadata,
            main_runtime=None if native_main_runtime is not None else TensorRTRuntime(main_engine, device=device),
            visual_runtime=TensorRTRuntime(visual_engine, device=device)
            if visual_engine and visual_engine.is_file()
            else None,
            prompt_projector=load_prompt_projector(projector_path, device=torch.device(device)),
            device=device,
            text_encoder_path=preferred_text_path,
            native_main_runtime=native_main_runtime,
        )

    def _get_text_encoder(self) -> TextPromptEncoder:
        if self._text_encoder is None:
            LOGGER.info("Resolving text encoder for '%s'", self.metadata.text_model)
            self._text_encoder = TextPromptEncoder(
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

    @property
    def _main_output_names(self) -> list[str]:
        if self.native_main_runtime is not None:
            return list(self.native_main_runtime.output_names)
        if self.main_runtime is None:
            raise RuntimeError("Main runtime is not initialized")
        return self.main_runtime.output_names

    @property
    def _main_fp16(self) -> bool:
        if self.native_main_runtime is not None:
            return bool(self.native_main_runtime.fp16)
        if self.main_runtime is None:
            raise RuntimeError("Main runtime is not initialized")
        return self.main_runtime.fp16

    def _sync_native_prompt_embeddings(self) -> None:
        if self.native_main_runtime is None:
            return
        try:
            embeddings, _ = self._active_prompt_embeddings()
        except RuntimeError:
            self.native_main_runtime.clear_prompt_embeddings()
            return
        self.native_main_runtime.set_prompt_embeddings(embeddings.detach().float().cpu().contiguous().numpy())

    def clear_prompts(self) -> None:
        self._text_prompt_embeddings = None
        self._visual_prompt_embeddings = None
        self._text_names = []
        self._visual_names = []
        self._sync_native_prompt_embeddings()
        LOGGER.info("Cleared active text and visual prompts")

    def set_classes(self, classes: list[str]) -> None:
        resolved = [str(name) for name in classes]
        if not resolved:
            raise ValueError("classes must not be empty")
        embeddings = compile_text_embeddings(
            resolved,
            encoder=self._get_text_encoder(),
            projector=self.prompt_projector,
        )
        self._text_prompt_embeddings = embeddings.to(self.device)
        self._text_names = resolved
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
        self._sync_native_prompt_embeddings()
        LOGGER.info("Activated %d external prompt embedding(s): %s", len(resolved), resolved)

    def set_visual_prompts(
        self,
        refer_image: object,
        bboxes: list[list[float]] | None = None,
        masks: object | None = None,
        classes: list[str] | None = None,
        imgsz: int | tuple[int, int] | list[int] | None = None,
    ) -> None:
        if self.visual_runtime is None:
            raise RuntimeError("This artifact bundle does not include a visual-prompt engine")
        LOGGER.info("Setting visual prompts from '%s'", refer_image)
        source_item = normalize_source(refer_image, default_prefix="refer")[0]
        target_size = normalize_imgsz(imgsz or self.metadata.default_imgsz)
        sample = preprocess_image(
            source_item,
            imgsz=target_size,
            device=self.device,
            fp16=self.visual_runtime.fp16,
            stride=self.metadata.stride,
        )
        prompt_batch = build_visual_prompt_batch(
            image=sample.original,
            dst_shape=sample.transformed.shape[:2],
            visual_stride=self.metadata.visual_stride,
            bboxes=bboxes,
            masks=masks,
            classes=classes,
        )
        outputs = self.visual_runtime.infer(
            {
                self.metadata.image_input_name: sample.tensor,
                self.metadata.visual_input_name: prompt_batch.tensor.to(self.device),
            }
        )
        prompt_output_name = self.visual_runtime.output_names[0]
        self._visual_prompt_embeddings = outputs[prompt_output_name].float()
        self._visual_names = prompt_batch.names
        self._sync_native_prompt_embeddings()
        LOGGER.info(
            "Activated %d visual prompt class(es): %s",
            len(self._visual_names),
            self._visual_names,
        )

    def warmup(self, imgsz: int | tuple[int, int] | list[int] | None = None) -> None:
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
        if self.visual_runtime is not None and self.metadata.visual_profile is not None:
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
        try:
            _, names = self._active_prompt_embeddings()
            return names
        except RuntimeError:
            return []

    def _postprocess_predictions(
        self,
        outputs: dict[str, torch.Tensor],
        sample,
        input_shape: tuple[int, int],
        names: list[str],
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
                preds[:, :4] = ops.scale_boxes(input_shape, preds[:, :4], sample.original.shape)
                masks = ops.process_mask_native(proto, preds[:, 6:], preds[:, :4], sample.original.shape[:2])
            else:
                masks = ops.process_mask(proto, preds[:, 6:], preds[:, :4], input_shape, upsample=True)
                preds[:, :4] = ops.scale_boxes(input_shape, preds[:, :4], sample.original.shape)
            if masks is not None:
                keep = masks.amax((-2, -1)) > 0
                if not bool(torch.all(keep)):
                    preds = preds[keep]
                    masks = masks[keep]
            return Results(
                sample.original, path=sample.path, names=names_map, boxes=preds[:, :6], masks=masks, speed=speed
            )

        preds[:, :4] = ops.scale_boxes(input_shape, preds[:, :4], sample.original.shape)
        return Results(sample.original, path=sample.path, names=names_map, boxes=preds[:, :6], speed=speed)

    def _predict_one(
        self,
        item: SourceItem,
        imgsz: tuple[int, int],
        conf: float,
        iou: float,
        max_det: int,
        retina_masks: bool,
    ) -> Results:
        prompt_embeddings, names = self._active_prompt_embeddings()
        LOGGER.info(
            "Running inference on '%s' with %d active prompt class(es) at imgsz=%s",
            item.path,
            len(names),
            imgsz,
        )
        preprocess_start = time.perf_counter()
        if self.native_main_runtime is not None:
            native_outputs = self.native_main_runtime.infer_image(item.image, imgsz[0], imgsz[1])
            preprocess_ms = float(native_outputs["preprocess_ms"])
            inference_ms = float(native_outputs["inference_ms"])
            outputs = {name: from_dlpack(native_outputs[name]) for name in self._main_output_names}
            input_shape = tuple(int(v) for v in native_outputs["input_shape"])
            sample = SimpleNamespace(original=item.image, path=item.path)
        else:
            if self.main_runtime is None:
                raise RuntimeError("Main runtime is not initialized")
            sample = preprocess_image(
                item,
                imgsz=imgsz,
                device=self.device,
                fp16=self._main_fp16,
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
            sample=sample,
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
        LOGGER.info(
            "Completed inference on '%s': detections=%d preprocess=%.1fms inference=%.1fms postprocess=%.1fms",
            item.path,
            detections,
            speed["preprocess"],
            speed["inference"],
            speed["postprocess"],
        )
        return result

    def predict_item(
        self,
        item: SourceItem,
        imgsz: int | tuple[int, int] | list[int] | None = None,
        conf: float = 0.25,
        iou: float = 0.45,
        max_det: int | None = None,
        retina_masks: bool = False,
    ) -> Results:
        target_size = normalize_imgsz(imgsz or self.metadata.default_imgsz)
        resolved_max_det = int(max_det or self.metadata.max_det)
        return self._predict_one(
            item=item,
            imgsz=target_size,
            conf=conf,
            iou=iou,
            max_det=resolved_max_det,
            retina_masks=retina_masks,
        )

    def _predict_iter(
        self,
        source: object,
        imgsz: tuple[int, int],
        conf: float,
        iou: float,
        max_det: int,
        retina_masks: bool,
    ) -> Iterator[Results]:
        for item in stream_sources(source):
            yield self.predict_item(
                item=item,
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
        imgsz: int | tuple[int, int] | list[int] | None = None,
        conf: float = 0.25,
        iou: float = 0.45,
        max_det: int | None = None,
        retina_masks: bool = False,
    ) -> list[Results] | Iterator[Results]:
        target_size = normalize_imgsz(imgsz or self.metadata.default_imgsz)
        resolved_max_det = int(max_det or self.metadata.max_det)
        if is_live_source(source) and not stream and not is_finite_live_source(source):
            raise ValueError("Live sources require stream=True or an explicit max_frames limit")
        LOGGER.info(
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
        )
        if stream:
            return results
        return list(results)

    __call__ = predict

    @staticmethod
    def export(*args, **kwargs) -> Path:
        return export_model(*args, **kwargs)
