from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from yoloe_tensorrt import train_cli
from yoloe_tensorrt.datasets import DatasetValidationError
from yoloe_tensorrt.training import TrainingError, TrainingResult


def test_train_cli_displays_help(
    run_python_help: Callable[[list[str]], subprocess.CompletedProcess[str]],
) -> None:
    result = run_python_help(["-m", "yoloe_tensorrt", "train"])

    assert result.returncode == 0
    assert "Train or fine-tune a YOLOE checkpoint through Ultralytics." in result.stdout
    assert "--imgsz" in result.stdout
    assert "--epochs" in result.stdout
    assert "--batch" in result.stdout
    assert "--resume" in result.stdout
    assert "--resume-checkpoint" in result.stdout
    assert "--ultralytics-arg" in result.stdout
    assert "--export" in result.stdout
    assert "--export-checkpoint" in result.stdout
    assert "--export-format" in result.stdout


def test_train_cli_passes_core_args_and_prints_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[dict[str, Any]] = []
    checkpoint = tmp_path / "runs" / "seg" / "weights" / "best.pt"
    run_dir = checkpoint.parent.parent
    metrics = run_dir / "results.csv"

    def _fake_train_model(model_checkpoint: str, dataset_config: Path, **kwargs: Any) -> TrainingResult:
        calls.append(
            {
                "model_checkpoint": model_checkpoint,
                "dataset_config": dataset_config,
                **kwargs,
            }
        )
        return TrainingResult(
            checkpoint_path=checkpoint,
            run_dir=run_dir,
            metrics_path=metrics,
            metadata={},
        )

    monkeypatch.setattr(train_cli, "train_model", _fake_train_model)

    exit_code = train_cli.main(
        [
            "yoloe-26s-seg.pt",
            "data.yaml",
            "--task",
            "segment",
            "--imgsz",
            "320x480",
            "--epochs",
            "3",
            "--batch",
            "0.5",
            "--device",
            "cuda:0",
            "--output-dir",
            str(tmp_path / "outputs" / "training"),
            "--name",
            "seg-run",
            "--resume",
            "--exist-ok",
            "--no-validate-dataset",
            "--ultralytics-arg",
            "workers=2",
            "--ultralytics-arg",
            "amp=false",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert output == f"checkpoint: {checkpoint}\nrun_dir: {run_dir}\nmetrics: {metrics}\n"
    assert calls == [
        {
            "model_checkpoint": "yoloe-26s-seg.pt",
            "dataset_config": Path("data.yaml"),
            "task": "segment",
            "imgsz": (320, 480),
            "epochs": 3,
            "batch": 0.5,
            "device": "cuda:0",
            "output_dir": tmp_path / "outputs" / "training",
            "name": "seg-run",
            "overrides": {"workers": 2, "amp": False, "resume": True},
            "validate_dataset": False,
            "exist_ok": True,
        }
    ]


def test_train_cli_passes_resume_checkpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    def _fake_train_model(model_checkpoint: str, dataset_config: Path, **kwargs: Any) -> TrainingResult:
        calls.append(kwargs)
        return TrainingResult(
            checkpoint_path=tmp_path / "best.pt",
            run_dir=tmp_path / "run",
            metrics_path=None,
            metadata={},
        )

    monkeypatch.setattr(train_cli, "train_model", _fake_train_model)

    exit_code = train_cli.main(["model.pt", "data.yaml", "--resume-checkpoint", "runs/train/weights/last.pt"])

    assert exit_code == 0
    assert calls[0]["overrides"]["resume"] == "runs/train/weights/last.pt"


def test_train_cli_allows_options_before_positionals(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    def _fake_train_model(model_checkpoint: str, dataset_config: Path, **kwargs: Any) -> TrainingResult:
        calls.append(
            {
                "model_checkpoint": model_checkpoint,
                "dataset_config": dataset_config,
                **kwargs,
            }
        )
        return TrainingResult(
            checkpoint_path=tmp_path / "best.pt",
            run_dir=tmp_path / "run",
            metrics_path=None,
            metadata={},
        )

    monkeypatch.setattr(train_cli, "train_model", _fake_train_model)

    exit_code = train_cli.main(["--imgsz", "640", "--resume", "model.pt", "data.yaml"])

    assert exit_code == 0
    assert calls[0]["model_checkpoint"] == "model.pt"
    assert calls[0]["dataset_config"] == Path("data.yaml")
    assert calls[0]["imgsz"] == 640
    assert calls[0]["overrides"]["resume"] is True


def test_train_cli_omits_metrics_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def _fake_train_model(*_args: Any, **_kwargs: Any) -> TrainingResult:
        return TrainingResult(
            checkpoint_path=tmp_path / "best.pt",
            run_dir=tmp_path / "run",
            metrics_path=None,
            metadata={},
        )

    monkeypatch.setattr(train_cli, "train_model", _fake_train_model)

    exit_code = train_cli.main(["model.pt", "data.yaml"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert output == f"checkpoint: {tmp_path / 'best.pt'}\nrun_dir: {tmp_path / 'run'}\n"


def test_train_cli_forwards_post_training_export_options_and_prints_artifact_info(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[dict[str, Any]] = []
    checkpoint = tmp_path / "runs" / "seg" / "weights" / "best.pt"
    run_dir = checkpoint.parent.parent
    artifact_dir = tmp_path / "artifacts" / "bundle"

    def _fake_train_model(model_checkpoint: str, dataset_config: Path, **kwargs: Any) -> TrainingResult:
        calls.append(
            {
                "model_checkpoint": model_checkpoint,
                "dataset_config": dataset_config,
                **kwargs,
            }
        )
        return TrainingResult(
            checkpoint_path=checkpoint,
            run_dir=run_dir,
            metrics_path=None,
            metadata={},
            best_checkpoint_path=checkpoint,
            artifact_dir=artifact_dir,
            exported_checkpoint_path=checkpoint,
        )

    monkeypatch.setattr(train_cli, "train_model", _fake_train_model)

    exit_code = train_cli.main(
        [
            "model.pt",
            "data.yaml",
            "--export",
            "--export-checkpoint",
            "last",
            "--export-artifact-dir",
            str(tmp_path / "exports"),
            "--export-format",
            "onnx",
            "--export-format",
            "engine",
            "--export-fixed",
            "--export-no-visual-engine",
            "--export-no-fp16",
            "--export-imgsz",
            "320",
            "--export-max-det",
            "17",
            "--export-workspace-mb",
            "64",
            "--export-exporter",
            "legacy",
            "--export-opset-version",
            "17",
            "--export-overwrite",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert output == (
        f"checkpoint: {checkpoint}\n"
        f"run_dir: {run_dir}\n"
        f"exported_checkpoint: {checkpoint}\n"
        f"artifact_dir: {artifact_dir}\n"
    )
    assert calls == [
        {
            "model_checkpoint": "model.pt",
            "dataset_config": Path("data.yaml"),
            "task": "detect",
            "imgsz": 640,
            "epochs": 100,
            "batch": 16,
            "device": None,
            "output_dir": Path("outputs/training"),
            "name": None,
            "overrides": {},
            "validate_dataset": True,
            "exist_ok": False,
            "export_artifact": True,
            "export_checkpoint": "last",
            "export_artifact_dir": tmp_path / "exports",
            "export_formats": ("onnx", "engine"),
            "export_dynamic": False,
            "export_build_visual_engine": False,
            "export_fp16": False,
            "export_imgsz": 320,
            "export_max_det": 17,
            "export_overwrite": True,
            "export_workspace_bytes": 64 << 20,
            "export_onnx_exporter": "legacy",
            "export_onnx_opset_version": 17,
        }
    ]


@pytest.mark.parametrize(
    "args",
    [
        ["model.pt", "data.yaml", "--ultralytics-arg", "workers"],
        ["model.pt", "data.yaml", "--ultralytics-arg", "=2"],
        ["model.pt", "data.yaml", "--ultralytics-arg", "workers=2", "--ultralytics-arg", "workers=4"],
        ["model.pt", "data.yaml", "--resume", "--ultralytics-arg", "resume=true"],
        ["model.pt", "data.yaml", "--resume", "--resume-checkpoint", "runs/train/weights/last.pt"],
        ["model.pt", "data.yaml", "--ultralytics-arg", "workers=["],
        ["model.pt", "data.yaml", "--imgsz", "320x"],
        ["model.pt", "data.yaml", "--imgsz", "320", "480", "640"],
        ["model.pt", "data.yaml", "--export-format", "engine"],
    ],
)
def test_train_cli_rejects_invalid_cli_args(
    monkeypatch: pytest.MonkeyPatch,
    args: list[str],
) -> None:
    monkeypatch.setattr(train_cli, "train_model", lambda *_args, **_kwargs: pytest.fail("should not train"))

    with pytest.raises(SystemExit) as exc_info:
        train_cli.main(args)

    assert exc_info.value.code == 2


@pytest.mark.parametrize(
    ("exception", "message"),
    [
        (DatasetValidationError("bad dataset"), "error: bad dataset"),
        (TrainingError("missing checkpoint"), "error: missing checkpoint"),
        (ValueError("bad override"), "error: bad override"),
    ],
)
def test_train_cli_reports_actionable_training_errors(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    exception: Exception,
    message: str,
) -> None:
    def _fake_train_model(*_args: Any, **_kwargs: Any) -> TrainingResult:
        raise exception

    monkeypatch.setattr(train_cli, "train_model", _fake_train_model)

    exit_code = train_cli.main(["model.pt", "data.yaml"])

    assert exit_code == 1
    assert message in capsys.readouterr().err


def test_train_cli_reports_unavailable_training_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def _fake_train_model(*_args: Any, **_kwargs: Any) -> TrainingResult:
        raise ModuleNotFoundError("No module named 'ultralytics'")

    monkeypatch.setattr(train_cli, "train_model", _fake_train_model)

    exit_code = train_cli.main(["model.pt", "data.yaml"])

    assert exit_code == 1
    error_output = capsys.readouterr().err
    assert "training dependencies are unavailable" in error_output
    assert "ultralytics" in error_output
