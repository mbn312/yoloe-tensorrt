from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_module_entrypoint_displays_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "yoloe_tensorrt", "--help"],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0
    assert "CLI entry points for yoloe-tensorrt." in result.stdout
    assert "camera-gui" in result.stdout
    assert "export" in result.stdout
    assert "benchmark" in result.stdout


def test_export_cli_displays_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "yoloe_tensorrt", "export", "--help"],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0
    assert "Export a YOLOE checkpoint into a yoloe-tensorrt artifact bundle." in result.stdout


def test_camera_gui_cli_displays_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "yoloe_tensorrt", "camera-gui", "--help"],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0
    assert "Launch the YOLOE TensorRT live camera GUI." in result.stdout
    assert "--track" in result.stdout
    assert "--tracker" in result.stdout


def test_benchmark_cli_displays_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "yoloe_tensorrt", "benchmark", "--help"],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0
    assert "Benchmark host-image and CUDA-tensor inference" in result.stdout
    assert "--mode" in result.stdout
    assert "--visual-prompts" in result.stdout
