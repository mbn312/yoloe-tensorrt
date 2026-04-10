from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
import yoloe_tensorrt
from yoloe_tensorrt import TrainingError, TrainingResult, train_model, training
from yoloe_tensorrt.datasets import DatasetConfig, DatasetSplit


class _FakeYOLOE:
    instances: list["_FakeYOLOE"] = []
    trainer: Any = None
    metrics: Any = {"map50": 0.5}

    def __init__(self, model: str, task: str | None = None) -> None:
        self.model = model
        self.task = task
        self.train_kwargs: dict[str, Any] | None = None
        type(self).instances.append(self)

    def train(self, **kwargs: Any) -> Any:
        self.train_kwargs = kwargs
        self.trainer = type(self).trainer
        return type(self).metrics


def _make_dataset_config(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    for split in ("train", "val"):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        (image_dir / "img0.jpg").write_bytes(b"")
        (label_dir / "img0.txt").write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")

    config_path = tmp_path / "data.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "path": "dataset",
                "train": "images/train",
                "val": "images/val",
                "names": ["person"],
            }
        ),
        encoding="utf-8",
    )
    return config_path


def _make_dataset_result(tmp_path: Path) -> DatasetConfig:
    split = DatasetSplit(
        name="train",
        images=(tmp_path / "dataset" / "images" / "train",),
        labels=(tmp_path / "dataset" / "labels" / "train",),
        image_count=2,
        label_count=2,
    )
    return DatasetConfig(
        yaml_path=tmp_path / "data.yaml",
        root=tmp_path / "dataset",
        task="detect",
        names={0: "person"},
        train=split,
        val=DatasetSplit(
            name="val",
            images=(tmp_path / "dataset" / "images" / "val",),
            labels=(tmp_path / "dataset" / "labels" / "val",),
            image_count=1,
            label_count=1,
        ),
    )


def _install_fake_yoloe(
    monkeypatch: pytest.MonkeyPatch,
    *,
    trainer: Any,
    metrics: Any = {"map50": 0.5},
) -> type[_FakeYOLOE]:
    _FakeYOLOE.instances.clear()
    _FakeYOLOE.trainer = trainer
    _FakeYOLOE.metrics = metrics
    monkeypatch.setattr(training, "_load_yoloe_class", lambda: _FakeYOLOE)
    return _FakeYOLOE


def test_train_model_validates_dataset_and_calls_ultralytics(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dataset_path = tmp_path / "data.yaml"
    model_path = tmp_path / "model.pt"
    output_dir = tmp_path / "runs"
    run_dir = output_dir / "custom"
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True)
    best_path = weights_dir / "best.pt"
    last_path = weights_dir / "last.pt"
    best_path.write_bytes(b"best")
    last_path.write_bytes(b"last")
    metrics_path = run_dir / "results.csv"
    metrics_path.write_text("metrics\n", encoding="utf-8")

    validation_calls: list[tuple[Path, str]] = []

    def _fake_validate(config: str | Path, *, task: str) -> DatasetConfig:
        validation_calls.append((Path(config), task))
        return _make_dataset_result(tmp_path)

    monkeypatch.setattr(training, "validate_dataset_config", _fake_validate)
    fake_yoloe = _install_fake_yoloe(
        monkeypatch,
        trainer=SimpleNamespace(save_dir=run_dir, best=best_path, last=last_path),
        metrics=SimpleNamespace(),
    )

    result = train_model(
        model_path,
        dataset_path,
        task="detect",
        imgsz=320,
        epochs=3,
        batch=4,
        device="cuda:0",
        output_dir=output_dir,
        name="custom",
        exist_ok=True,
    )

    assert isinstance(result, TrainingResult)
    assert validation_calls == [(dataset_path, "detect")]
    assert result.checkpoint_path == best_path.resolve()
    assert result.run_dir == run_dir.resolve()
    assert result.metrics_path == metrics_path.resolve()
    assert result.metadata["dataset"] == {
        "root": str(tmp_path / "dataset"),
        "names": {0: "person"},
        "train_image_count": 2,
        "val_image_count": 1,
        "test_image_count": None,
    }
    assert result.metadata["metrics_type"] == "SimpleNamespace"

    instance = fake_yoloe.instances[0]
    assert instance.model == str(model_path)
    assert instance.task == "detect"
    assert instance.train_kwargs == {
        "data": str(dataset_path),
        "task": "detect",
        "imgsz": 320,
        "epochs": 3,
        "batch": 4,
        "project": str(output_dir),
        "exist_ok": True,
        "device": "cuda:0",
        "name": "custom",
    }


def test_train_model_merges_non_conflicting_ultralytics_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset_path = _make_dataset_config(tmp_path)
    run_dir = tmp_path / "runs" / "train"
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True)
    (weights_dir / "best.pt").write_bytes(b"best")
    fake_yoloe = _install_fake_yoloe(
        monkeypatch,
        trainer=SimpleNamespace(save_dir=run_dir, best=weights_dir / "best.pt", last=weights_dir / "last.pt"),
    )

    train_model(
        "yoloe-26s-seg.pt",
        dataset_path,
        overrides={"workers": 2, "optimizer": "AdamW", "lr0": 0.001},
    )

    assert fake_yoloe.instances[0].train_kwargs["workers"] == 2
    assert fake_yoloe.instances[0].train_kwargs["optimizer"] == "AdamW"
    assert fake_yoloe.instances[0].train_kwargs["lr0"] == 0.001


@pytest.mark.parametrize("save_value", [False, 0])
def test_train_model_rejects_falsey_save_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    save_value: bool | int,
) -> None:
    dataset_path = _make_dataset_config(tmp_path)
    _install_fake_yoloe(monkeypatch, trainer=SimpleNamespace())

    with pytest.raises(ValueError, match="requires Ultralytics to save checkpoints"):
        train_model("model.pt", dataset_path, overrides={"save": save_value})

    assert _FakeYOLOE.instances == []


def test_train_model_rejects_invalid_override_keys(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dataset_path = _make_dataset_config(tmp_path)
    _install_fake_yoloe(monkeypatch, trainer=SimpleNamespace())

    with pytest.raises(ValueError, match="must be non-empty strings"):
        train_model("model.pt", dataset_path, overrides={1: "workers"})  # type: ignore[dict-item]

    assert _FakeYOLOE.instances == []


@pytest.mark.parametrize("conflicting_key", ["data", "task", "imgsz", "epochs", "batch", "device", "project", "name"])
def test_train_model_rejects_conflicting_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    conflicting_key: str,
) -> None:
    dataset_path = _make_dataset_config(tmp_path)
    _install_fake_yoloe(monkeypatch, trainer=SimpleNamespace())

    with pytest.raises(ValueError, match="conflict with train_model"):
        train_model("model.pt", dataset_path, overrides={conflicting_key: "override"})

    assert _FakeYOLOE.instances == []


def test_train_model_falls_back_to_last_checkpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dataset_path = _make_dataset_config(tmp_path)
    run_dir = tmp_path / "runs" / "train"
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True)
    last_path = weights_dir / "last.pt"
    last_path.write_bytes(b"last")
    _install_fake_yoloe(
        monkeypatch,
        trainer=SimpleNamespace(save_dir=run_dir, best=weights_dir / "missing_best.pt", last=last_path),
    )

    result = train_model("model.pt", dataset_path)

    assert result.checkpoint_path == last_path.resolve()
    assert result.metrics_path is None


def test_train_model_falls_back_to_default_checkpoint_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dataset_path = _make_dataset_config(tmp_path)
    run_dir = tmp_path / "runs" / "train"
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True)
    best_path = weights_dir / "best.pt"
    best_path.write_bytes(b"best")
    _install_fake_yoloe(monkeypatch, trainer=SimpleNamespace(save_dir=run_dir, best=None, last=None))

    result = train_model("model.pt", dataset_path)

    assert result.checkpoint_path == best_path.resolve()


def test_train_model_resolves_relative_trainer_checkpoint_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset_path = _make_dataset_config(tmp_path)
    run_dir = tmp_path / "runs" / "train"
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True)
    best_path = weights_dir / "best.pt"
    best_path.write_bytes(b"best")
    _install_fake_yoloe(monkeypatch, trainer=SimpleNamespace(save_dir=run_dir, best="weights/best.pt", last=None))

    result = train_model("model.pt", dataset_path)

    assert result.checkpoint_path == best_path.resolve()


def test_train_model_expands_user_model_checkpoint_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dataset_path = _make_dataset_config(tmp_path)
    run_dir = tmp_path / "runs" / "train"
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True)
    (weights_dir / "best.pt").write_bytes(b"best")
    fake_yoloe = _install_fake_yoloe(
        monkeypatch,
        trainer=SimpleNamespace(save_dir=run_dir, best=weights_dir / "best.pt", last=weights_dir / "last.pt"),
    )

    train_model("~/models/yoloe.pt", dataset_path)

    assert fake_yoloe.instances[0].model == str(Path("~/models/yoloe.pt").expanduser())


def test_train_model_preserves_url_model_checkpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dataset_path = _make_dataset_config(tmp_path)
    run_dir = tmp_path / "runs" / "train"
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True)
    (weights_dir / "best.pt").write_bytes(b"best")
    fake_yoloe = _install_fake_yoloe(
        monkeypatch,
        trainer=SimpleNamespace(save_dir=run_dir, best=weights_dir / "best.pt", last=weights_dir / "last.pt"),
    )

    checkpoint_url = "https://example.com/models/yoloe.pt"
    train_model(checkpoint_url, dataset_path)

    assert fake_yoloe.instances[0].model == checkpoint_url


def test_train_model_raises_when_checkpoint_is_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dataset_path = _make_dataset_config(tmp_path)
    run_dir = tmp_path / "runs" / "train"
    run_dir.mkdir(parents=True)
    _install_fake_yoloe(monkeypatch, trainer=SimpleNamespace(save_dir=run_dir, best=None, last=None))

    with pytest.raises(TrainingError, match="Could not find a trained checkpoint"):
        train_model("model.pt", dataset_path)


def test_train_model_raises_when_trainer_is_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dataset_path = _make_dataset_config(tmp_path)
    _install_fake_yoloe(monkeypatch, trainer=None)

    with pytest.raises(TrainingError, match="did not expose a trainer"):
        train_model("model.pt", dataset_path)


def test_train_model_can_skip_dataset_validation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dataset_path = tmp_path / "data.yaml"
    run_dir = tmp_path / "runs" / "train"
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True)
    (weights_dir / "best.pt").write_bytes(b"best")
    fake_yoloe = _install_fake_yoloe(
        monkeypatch,
        trainer=SimpleNamespace(save_dir=run_dir, best=weights_dir / "best.pt", last=weights_dir / "last.pt"),
    )
    monkeypatch.setattr(
        training,
        "validate_dataset_config",
        lambda *_args, **_kwargs: pytest.fail("dataset validation should have been skipped"),
    )

    result = train_model("model.pt", dataset_path, validate_dataset=False)

    assert result.metadata["dataset"] is None
    assert fake_yoloe.instances[0].train_kwargs["data"] == str(dataset_path)


def test_train_model_rejects_mapping_dataset_config() -> None:
    with pytest.raises(ValueError, match="requires dataset_config to be a YAML/config path"):
        train_model("model.pt", {"train": "images/train"})  # type: ignore[arg-type]


def test_train_model_rejects_empty_model_checkpoint() -> None:
    with pytest.raises(ValueError, match="model_checkpoint to be non-empty"):
        train_model("", "data.yaml")


def test_train_model_rejects_empty_dataset_config_path() -> None:
    with pytest.raises(ValueError, match="non-empty YAML/config path"):
        train_model("model.pt", "")


def test_train_model_rejects_unsupported_task_when_validation_is_disabled(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported training task"):
        train_model("model.pt", tmp_path / "data.yaml", task="pose", validate_dataset=False)  # type: ignore[arg-type]


def test_training_symbols_are_exported_lazily() -> None:
    assert yoloe_tensorrt.train_model is train_model
    assert yoloe_tensorrt.TrainingResult is TrainingResult
    assert yoloe_tensorrt.TrainingError is TrainingError
    assert "train_model" in dir(yoloe_tensorrt)
