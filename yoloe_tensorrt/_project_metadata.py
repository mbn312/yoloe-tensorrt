from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


def load_pyproject(pyproject_path: str | Path) -> dict[str, Any]:
    path = Path(pyproject_path)
    return tomllib.loads(path.read_text(encoding="utf-8"))


def load_project_version(pyproject_path: str | Path) -> str:
    data = load_pyproject(pyproject_path)
    try:
        return str(data["project"]["version"])
    except KeyError as exc:
        raise ValueError(f"Unable to find a project version in '{pyproject_path}'") from exc
