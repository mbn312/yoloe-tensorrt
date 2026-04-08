from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from yoloe_tensorrt.gui import parse_confidence_text, parse_label_text


def test_parse_label_text_supports_multiple_labels_and_deduplicates() -> None:
    assert parse_label_text("pen, marker\nbottle, pen") == ["pen", "marker", "bottle"]


def test_parse_label_text_ignores_empty_values() -> None:
    assert parse_label_text(" , \n ,pen,, ") == ["pen"]


def test_parse_confidence_text_clamps_and_falls_back() -> None:
    assert parse_confidence_text("0.75") == 0.75
    assert parse_confidence_text("2.0") == 1.0
    assert parse_confidence_text("-0.5") == 0.0
    assert parse_confidence_text("bad", fallback=0.33) == 0.33


def test_python_launcher_displays_help() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    launcher = repo_root / "scripts" / "launch_camera_gui.py"
    result = subprocess.run(
        [sys.executable, str(launcher), "--help"],
        check=False,
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    assert result.returncode == 0
    assert "Launch the YOLOE TensorRT live camera GUI." in result.stdout
    assert "--track" in result.stdout
    assert "--tracker" in result.stdout


def test_shell_launcher_prefers_console_script_when_available(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    launcher = repo_root / "scripts" / "launch_camera_gui.sh"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "yoloe-camera-gui"
    stub.write_text("#!/usr/bin/env bash\nprintf 'stub-launcher %s\\n' \"$*\"\n", encoding="utf-8")
    stub.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
    result = subprocess.run(
        ["bash", str(launcher), "--help"],
        check=False,
        capture_output=True,
        text=True,
        cwd=repo_root,
        env=env,
    )
    assert result.returncode == 0
    assert "stub-launcher --help" in result.stdout
