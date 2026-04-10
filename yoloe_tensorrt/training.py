from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

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


class TrainingError(RuntimeError):
    """Raised when Ultralytics training completes without expected run artifacts."""


@dataclass(frozen=True)
class TrainingResult:
    checkpoint_path: Path
    run_dir: Path
    metrics_path: Path | None
    metadata: dict[str, Any]


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
) -> TrainingResult:
    """Train or fine-tune a YOLOE checkpoint through Ultralytics and return run artifacts."""

    if task not in _SUPPORTED_TRAINING_TASKS:
        raise ValueError(f"Unsupported training task {task!r}; expected 'detect' or 'segment'.")

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
    checkpoint_path = _resolve_checkpoint_path(trainer, run_dir)
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

    return TrainingResult(
        checkpoint_path=checkpoint_path,
        run_dir=run_dir,
        metrics_path=metrics_path,
        metadata=metadata,
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


def _resolve_checkpoint_path(trainer: Any, run_dir: Path) -> Path:
    for attr in ("best", "last"):
        path = _optional_path(getattr(trainer, attr, None), base_dir=run_dir)
        if path is not None and path.exists():
            return path.resolve()

    for candidate in (run_dir / "weights" / "best.pt", run_dir / "weights" / "last.pt"):
        if candidate.exists():
            return candidate.resolve()

    raise TrainingError(f"Could not find a trained checkpoint under Ultralytics run directory: {run_dir}")


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
