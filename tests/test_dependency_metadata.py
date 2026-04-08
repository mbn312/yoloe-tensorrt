from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_pyproject() -> dict:
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_export_and_dev_dependencies_include_onnxscript() -> None:
    data = _load_pyproject()
    dependencies = data["project"]["dependencies"]
    optional = data["project"]["optional-dependencies"]

    export_reqs = optional["export"]
    dev_reqs = optional["dev"]
    all_reqs = optional["all"]

    assert "lap>=0.5.12" in dependencies
    assert any(req.startswith("onnx>=") for req in export_reqs)
    assert any(req.startswith("onnxscript>=") for req in export_reqs)
    assert any(req.startswith("onnxscript>=") for req in dev_reqs)
    assert any(req.startswith("onnxscript>=") for req in all_reqs)
    assert "pytest>=8.0" in all_reqs
    assert "mkdocs>=1.6" in all_reqs


def test_requirements_files_include_clip_dependency() -> None:
    requirements = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
    requirements_dev = (REPO_ROOT / "requirements-dev.txt").read_text(encoding="utf-8")

    clip_requirement = "git+https://github.com/ultralytics/CLIP.git"
    assert clip_requirement in requirements
    assert clip_requirement in requirements_dev
    assert "-e .[all]" in requirements_dev
