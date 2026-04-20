from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ._shapes import HWShape, TensorShape, normalize_hw_shape


def _json_compatible(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [_json_compatible(item) for item in value]
    return str(value)


@dataclass(frozen=True)
class ShapeProfile:
    minimum: TensorShape
    optimum: TensorShape
    maximum: TensorShape

    def to_dict(self) -> dict[str, list[int]]:
        return {
            "minimum": list(self.minimum),
            "optimum": list(self.optimum),
            "maximum": list(self.maximum),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ShapeProfile:
        return cls(
            minimum=tuple(int(v) for v in data["minimum"]),
            optimum=tuple(int(v) for v in data["optimum"]),
            maximum=tuple(int(v) for v in data["maximum"]),
        )


@dataclass(frozen=True)
class ArtifactMetadata:
    version: int
    model_path: str
    model_name: str
    task: str
    end2end: bool
    dynamic: bool
    default_imgsz: HWShape
    stride: int
    visual_stride: int
    embed_dim: int
    mask_dim: int | None
    max_det: int
    fp16: bool
    prompt_input_name: str
    visual_input_name: str
    image_input_name: str
    text_model: str
    text_encoder_filename: str | None
    prompt_projector_filename: str
    main_onnx_filename: str
    main_engine_filename: str | None
    visual_onnx_filename: str | None
    visual_engine_filename: str | None
    image_profile: ShapeProfile
    prompt_profile: ShapeProfile
    visual_profile: ShapeProfile | None
    training_metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["default_imgsz"] = list(self.default_imgsz)
        data["image_profile"] = self.image_profile.to_dict()
        data["prompt_profile"] = self.prompt_profile.to_dict()
        if self.visual_profile is not None:
            data["visual_profile"] = self.visual_profile.to_dict()
        if data["training_metadata"] is not None:
            data["training_metadata"] = _json_compatible(data["training_metadata"])
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArtifactMetadata:
        return cls(
            version=int(data["version"]),
            model_path=data["model_path"],
            model_name=data["model_name"],
            task=data["task"],
            end2end=bool(data["end2end"]),
            dynamic=bool(data["dynamic"]),
            default_imgsz=normalize_hw_shape(data["default_imgsz"], name="default_imgsz"),
            stride=int(data["stride"]),
            visual_stride=int(data["visual_stride"]),
            embed_dim=int(data["embed_dim"]),
            mask_dim=None if data["mask_dim"] is None else int(data["mask_dim"]),
            max_det=int(data["max_det"]),
            fp16=bool(data["fp16"]),
            prompt_input_name=data["prompt_input_name"],
            visual_input_name=data["visual_input_name"],
            image_input_name=data["image_input_name"],
            text_model=data["text_model"],
            text_encoder_filename=data["text_encoder_filename"],
            prompt_projector_filename=data["prompt_projector_filename"],
            main_onnx_filename=data["main_onnx_filename"],
            main_engine_filename=data["main_engine_filename"],
            visual_onnx_filename=data["visual_onnx_filename"],
            visual_engine_filename=data["visual_engine_filename"],
            image_profile=ShapeProfile.from_dict(data["image_profile"]),
            prompt_profile=ShapeProfile.from_dict(data["prompt_profile"]),
            visual_profile=None if data["visual_profile"] is None else ShapeProfile.from_dict(data["visual_profile"]),
            training_metadata=None if data.get("training_metadata") is None else dict(data["training_metadata"]),
        )


def default_artifact_dir(model_path: str | Path) -> Path:
    path = Path(model_path)
    return path.parent / f"{path.stem}_yoloe_tensorrt"


def metadata_path(artifact_dir: str | Path) -> Path:
    return Path(artifact_dir) / "metadata.json"


def save_metadata(artifact_dir: str | Path, metadata: ArtifactMetadata) -> Path:
    path = metadata_path(artifact_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata.to_dict(), indent=2, sort_keys=True))
    return path


def load_metadata(artifact_dir: str | Path) -> ArtifactMetadata:
    data = json.loads(metadata_path(artifact_dir).read_text())
    return ArtifactMetadata.from_dict(data)


def resolve_artifact_file(artifact_dir: str | Path, filename: str | None) -> Path | None:
    if not filename:
        return None
    return Path(artifact_dir) / filename
