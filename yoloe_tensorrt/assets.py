from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from .logging_utils import get_logger

LOGGER = get_logger(__name__)

DEFAULT_EXAMPLE_MODEL = "yoloe-26s-seg.pt"


def cache_root() -> Path:
    configured = os.environ.get("YOLOE_TRT_CACHE_DIR")
    if configured:
        return Path(configured).expanduser()
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        return Path(xdg_cache).expanduser() / "yoloe-tensorrt"
    return Path.home() / ".cache" / "yoloe-tensorrt"


def asset_cache_dir() -> Path:
    return cache_root() / "assets"


def text_asset_cache_dir() -> Path:
    return asset_cache_dir() / "text"


def default_example_model_spec() -> str:
    return os.environ.get("YOLOE_TRT_DEFAULT_MODEL", DEFAULT_EXAMPLE_MODEL)


def resolve_model_checkpoint(model: str | Path | None, *, allow_download: bool = True) -> Path:
    spec = default_example_model_spec() if model is None else str(model)
    if not str(spec).strip():
        raise ValueError("model must not be empty")

    path = Path(spec).expanduser()
    if path.is_file():
        return path.resolve()

    if path.parent != Path("."):
        raise FileNotFoundError(f"Model checkpoint does not exist: {path}")
    if not allow_download:
        raise FileNotFoundError(f"Model checkpoint does not exist: {path}")

    from ultralytics.utils.downloads import attempt_download_asset

    target = asset_cache_dir() / path.name
    target.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Resolving model checkpoint '%s' into cache '%s'", path.name, target)
    resolved = Path(attempt_download_asset(target))
    if resolved.is_file():
        return resolved.resolve()
    raise FileNotFoundError(
        f"Unable to resolve model checkpoint '{spec}'. Provide a local path or a downloadable asset name."
    )


def download_text_asset(asset_name: str | Path) -> Path:
    requested = Path(asset_name).expanduser()
    if requested.is_file():
        return requested.resolve()

    target = text_asset_cache_dir() / requested.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        return target.resolve()

    from ultralytics.utils.downloads import attempt_download_asset

    LOGGER.info("Resolving text encoder asset '%s' into cache '%s'", requested.name, target)
    resolved = Path(attempt_download_asset(target))
    if resolved.is_file():
        return resolved.resolve()
    raise FileNotFoundError(
        f"Unable to resolve text encoder asset '{requested.name}'. "
        "Provide a local path or ensure the Ultralytics asset is downloadable."
    )


def text_asset_search_roots(additional: Iterable[str | Path] | None = None) -> list[Path]:
    candidates: list[Path] = []
    env_file = os.environ.get("YOLOE_TRT_TEXT_ENCODER")
    if env_file:
        candidates.append(Path(env_file).expanduser())
    env_dir = os.environ.get("YOLOE_TRT_TEXT_ENCODER_DIR")
    if env_dir:
        candidates.append(Path(env_dir).expanduser())
    candidates.extend(
        [
            text_asset_cache_dir(),
            asset_cache_dir(),
            Path.cwd() / "outputs" / "assets",
            Path.cwd() / "outputs" / "assets-cache",
            Path.cwd() / "outputs" / "assets-cache" / "local",
            Path.cwd(),
        ]
    )
    if additional is not None:
        candidates.extend(Path(candidate).expanduser() for candidate in additional)

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique.append(candidate)
    return unique
