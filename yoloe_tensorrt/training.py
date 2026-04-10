from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

from .datasets import DatasetConfig, DatasetTask, validate_dataset_config

_CONFLICTING_OVERRIDE_KEYS = frozenset(
    {
        "batch",
        "data",
        "device",
        "epochs",
        "exist_ok",
        "imgsz",
        "mode",
        "model",
        "name",
        "project",
        "task",
    }
)
_METRICS_FILENAMES = ("results.csv", "results.json", "metrics.json")
_SUPPORTED_TRAINING_TASKS = {"detect", "segment"}
_SUPPORTED_EXPORT_CHECKPOINTS = {"best", "last"}
ExportCheckpoint = Literal["best", "last"]
OnnxExporterMode = Literal["auto", "legacy", "dynamo"]


class TrainingError(RuntimeError):
    """Raised when Ultralytics training completes without expected run artifacts."""


@dataclass(frozen=True)
class TrainingResult:
    checkpoint_path: Path
    run_dir: Path
    metrics_path: Path | None
    metadata: dict[str, Any]
    best_checkpoint_path: Path | None = None
    last_checkpoint_path: Path | None = None
    artifact_dir: Path | None = None
    exported_checkpoint_path: Path | None = None


def train_model(
    model_checkpoint: str | Path,
    dataset_config: str | Path,
    *,
    task: DatasetTask = "detect",
    imgsz: int | tuple[int, int] = 640,
    epochs: int = 100,
    batch: int | float | str = 16,
    device: Any = None,
    output_dir: str | Path = "outputs/training",
    name: str | None = None,
    overrides: Mapping[str, Any] | None = None,
    validate_dataset: bool = True,
    exist_ok: bool = False,
    export_artifact: bool = False,
    export_checkpoint: ExportCheckpoint = "best",
    export_artifact_dir: str | Path | None = None,
    export_formats: Iterable[str] | str = ("onnx", "engine"),
    export_dynamic: bool = True,
    export_build_visual_engine: bool = True,
    export_fp16: bool = True,
    export_imgsz: int | tuple[int, int] | list[int] | None = None,
    export_max_det: int = 300,
    export_overwrite: bool = True,
    export_workspace_bytes: int = 2 << 30,
    export_onnx_exporter: OnnxExporterMode = "auto",
    export_onnx_opset_version: int | None = None,
) -> TrainingResult:
    """Train or fine-tune a YOLOE checkpoint through Ultralytics and return run artifacts."""

    if task not in _SUPPORTED_TRAINING_TASKS:
        raise ValueError(f"Unsupported training task {task!r}; expected 'detect' or 'segment'.")
    if export_checkpoint not in _SUPPORTED_EXPORT_CHECKPOINTS:
        raise ValueError(f"Unsupported export checkpoint {export_checkpoint!r}; expected 'best' or 'last'.")

    model_path = _normalize_model_checkpoint(model_checkpoint)
    dataset_path = _normalize_dataset_config_path(dataset_config)
    output_path = Path(output_dir).expanduser()
    trainer_overrides = _normalize_overrides(overrides)
    dataset = validate_dataset_config(dataset_path, task=task) if validate_dataset else None

    train_kwargs: dict[str, Any] = {
        "data": str(dataset_path),
        "task": task,
        "imgsz": imgsz,
        "epochs": epochs,
        "batch": batch,
        "project": str(output_path),
        "exist_ok": exist_ok,
        **trainer_overrides,
    }
    if device is not None:
        train_kwargs["device"] = device
    if name is not None:
        train_kwargs["name"] = name

    yoloe_cls = _load_yoloe_class()
    model = yoloe_cls(model_path, task=task)
    metrics = model.train(**train_kwargs)
    trainer = getattr(model, "trainer", None)
    if trainer is None:
        raise TrainingError("Ultralytics training did not expose a trainer with run artifacts.")

    run_dir = _resolve_run_dir(trainer)
    best_checkpoint_path = _resolve_named_checkpoint_path(trainer, run_dir, "best")
    last_checkpoint_path = _resolve_named_checkpoint_path(trainer, run_dir, "last")
    checkpoint_path = _resolve_primary_checkpoint_path(best_checkpoint_path, last_checkpoint_path, run_dir)
    metrics_path = _resolve_metrics_path(run_dir)
    metadata = _build_training_metadata(
        model_checkpoint=model_path,
        dataset_path=dataset_path,
        dataset=dataset,
        task=task,
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        device=device,
        output_path=output_path,
        name=name,
        exist_ok=exist_ok,
        overrides=trainer_overrides,
        metrics=metrics,
    )
    artifact_dir: Path | None = None
    exported_checkpoint_path: Path | None = None

    if export_artifact:
        from .export import export_model

        exported_checkpoint_path = _select_export_checkpoint_path(
            best_checkpoint_path=best_checkpoint_path,
            last_checkpoint_path=last_checkpoint_path,
            export_checkpoint=export_checkpoint,
            run_dir=run_dir,
        )
        try:
            artifact_dir = export_model(
                exported_checkpoint_path,
                artifact_dir=None if export_artifact_dir is None else Path(export_artifact_dir).expanduser(),
                formats=export_formats,
                dynamic=bool(export_dynamic),
                build_visual_engine=bool(export_build_visual_engine),
                fp16=bool(export_fp16),
                imgsz=export_imgsz,
                max_det=int(export_max_det),
                overwrite=bool(export_overwrite),
                workspace_bytes=int(export_workspace_bytes),
                onnx_exporter=export_onnx_exporter,
                onnx_opset_version=None if export_onnx_opset_version is None else int(export_onnx_opset_version),
                training_metadata=_build_export_training_metadata(
                    metadata=metadata,
                    run_dir=run_dir,
                    metrics_path=metrics_path,
                    checkpoint_path=checkpoint_path,
                    best_checkpoint_path=best_checkpoint_path,
                    last_checkpoint_path=last_checkpoint_path,
                    export_checkpoint=export_checkpoint,
                    exported_checkpoint_path=exported_checkpoint_path,
                ),
            )
        except Exception as exc:
            raise TrainingError(
                f"Training succeeded but export failed for checkpoint '{exported_checkpoint_path}': {exc}"
            ) from exc

    return TrainingResult(
        checkpoint_path=checkpoint_path,
        run_dir=run_dir,
        metrics_path=metrics_path,
        metadata=metadata,
        best_checkpoint_path=best_checkpoint_path,
        last_checkpoint_path=last_checkpoint_path,
        artifact_dir=artifact_dir,
        exported_checkpoint_path=exported_checkpoint_path,
    )


def _load_yoloe_class() -> Any:
    from ultralytics import YOLOE

    return YOLOE


def _normalize_model_checkpoint(model_checkpoint: str | Path) -> str:
    if not isinstance(model_checkpoint, str | Path):
        raise ValueError("train_model(...) requires model_checkpoint to be a checkpoint path or asset name.")
    raw_value = str(model_checkpoint).strip()
    if not raw_value:
        raise ValueError("train_model(...) requires model_checkpoint to be non-empty.")
    if isinstance(model_checkpoint, str) and "://" in raw_value:
        return raw_value
    return str(Path(raw_value).expanduser())


def _normalize_dataset_config_path(dataset_config: str | Path) -> Path:
    if not isinstance(dataset_config, str | Path):
        raise ValueError("train_model(...) requires dataset_config to be a YAML/config path.")
    if not str(dataset_config).strip():
        raise ValueError("train_model(...) requires dataset_config to be a non-empty YAML/config path.")
    return Path(dataset_config).expanduser()


def _normalize_overrides(overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    if overrides is None:
        return {}
    if not isinstance(overrides, Mapping):
        raise ValueError("train_model(...) overrides must be a mapping when provided.")

    invalid_keys = sorted(repr(key) for key in overrides if not isinstance(key, str) or not key)
    if invalid_keys:
        joined = ", ".join(invalid_keys)
        raise ValueError(f"Ultralytics override key(s) must be non-empty strings: {joined}")

    if "save" in overrides and not overrides["save"]:
        raise ValueError(
            "train_model(...) requires Ultralytics to save checkpoints; call Ultralytics directly for save=False runs."
        )

    conflicting = sorted(key for key in overrides if key in _CONFLICTING_OVERRIDE_KEYS)
    if conflicting:
        joined = ", ".join(conflicting)
        raise ValueError(f"Ultralytics override key(s) conflict with train_model(...) arguments: {joined}")
    return dict(overrides)


def _resolve_run_dir(trainer: Any) -> Path:
    raw_save_dir = getattr(trainer, "save_dir", None)
    if raw_save_dir is None:
        raise TrainingError("Ultralytics trainer did not expose a save_dir.")
    return Path(raw_save_dir).expanduser().resolve(strict=False)


def _resolve_named_checkpoint_path(trainer: Any, run_dir: Path, checkpoint_name: ExportCheckpoint) -> Path | None:
    path = _optional_path(getattr(trainer, checkpoint_name, None), base_dir=run_dir)
    if path is not None and path.exists():
        return path.resolve()

    candidate = run_dir / "weights" / f"{checkpoint_name}.pt"
    if candidate.exists():
        return candidate.resolve()
    return None


def _resolve_primary_checkpoint_path(
    best_checkpoint_path: Path | None,
    last_checkpoint_path: Path | None,
    run_dir: Path,
) -> Path:
    if best_checkpoint_path is not None:
        return best_checkpoint_path
    if last_checkpoint_path is not None:
        return last_checkpoint_path
    raise TrainingError(f"Could not find a trained checkpoint under Ultralytics run directory: {run_dir}")


def _select_export_checkpoint_path(
    *,
    best_checkpoint_path: Path | None,
    last_checkpoint_path: Path | None,
    export_checkpoint: ExportCheckpoint,
    run_dir: Path,
) -> Path:
    if export_checkpoint == "best":
        if best_checkpoint_path is not None:
            return best_checkpoint_path
        raise TrainingError(
            f"Could not export the best checkpoint because '{run_dir / 'weights' / 'best.pt'}' was not found."
        )
    if last_checkpoint_path is not None:
        return last_checkpoint_path
    raise TrainingError(
        f"Could not export the last checkpoint because '{run_dir / 'weights' / 'last.pt'}' was not found."
    )


def _resolve_metrics_path(run_dir: Path) -> Path | None:
    for filename in _METRICS_FILENAMES:
        path = run_dir / filename
        if path.exists():
            return path.resolve()
    return None


def _optional_path(value: Any, *, base_dir: Path) -> Path | None:
    if value is None or value is False or value == "":
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve(strict=False)


def _build_training_metadata(
    *,
    model_checkpoint: str,
    dataset_path: Path,
    dataset: DatasetConfig | None,
    task: DatasetTask,
    imgsz: int | tuple[int, int],
    epochs: int,
    batch: int | float | str,
    device: Any,
    output_path: Path,
    name: str | None,
    exist_ok: bool,
    overrides: Mapping[str, Any],
    metrics: Any,
) -> dict[str, Any]:
    return {
        "model_checkpoint": model_checkpoint,
        "dataset_config": str(dataset_path),
        "dataset": _dataset_metadata(dataset),
        "task": task,
        "imgsz": imgsz,
        "epochs": epochs,
        "batch": batch,
        "device": device,
        "output_dir": str(output_path),
        "name": name,
        "exist_ok": exist_ok,
        "overrides": dict(overrides),
        "metrics_type": None if metrics is None else type(metrics).__name__,
    }


def _build_export_training_metadata(
    *,
    metadata: Mapping[str, Any],
    run_dir: Path,
    metrics_path: Path | None,
    checkpoint_path: Path,
    best_checkpoint_path: Path | None,
    last_checkpoint_path: Path | None,
    export_checkpoint: ExportCheckpoint,
    exported_checkpoint_path: Path,
) -> dict[str, Any]:
    export_metadata = dict(metadata)
    export_metadata["run_dir"] = str(run_dir)
    export_metadata["metrics_path"] = None if metrics_path is None else str(metrics_path)
    export_metadata["checkpoint_path"] = str(checkpoint_path)
    export_metadata["best_checkpoint_path"] = None if best_checkpoint_path is None else str(best_checkpoint_path)
    export_metadata["last_checkpoint_path"] = None if last_checkpoint_path is None else str(last_checkpoint_path)
    export_metadata["export_checkpoint"] = export_checkpoint
    export_metadata["exported_checkpoint_path"] = str(exported_checkpoint_path)
    return export_metadata


def _dataset_metadata(dataset: DatasetConfig | None) -> dict[str, Any] | None:
    if dataset is None:
        return None
    return {
        "root": str(dataset.root),
        "names": dict(dataset.names),
        "train_image_count": dataset.train.image_count,
        "val_image_count": dataset.val.image_count,
        "test_image_count": None if dataset.test is None else dataset.test.image_count,
    }
