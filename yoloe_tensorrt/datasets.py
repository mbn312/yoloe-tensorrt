from __future__ import annotations

import math
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml

DatasetTask = Literal["detect", "segment"]

_CLASS_ID_RE = re.compile(r"^[+-]?\d+$")
_IMAGE_SUFFIXES = {
    ".avif",
    ".bmp",
    ".dng",
    ".heic",
    ".heif",
    ".jpeg",
    ".jpeg2000",
    ".jpg",
    ".jp2",
    ".mpo",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
_SUPPORTED_TASKS = {"detect", "segment"}


class DatasetValidationError(ValueError):
    """Raised when a YOLO dataset config cannot be used safely for training."""


@dataclass(frozen=True)
class DatasetSplit:
    name: str
    images: tuple[Path, ...]
    labels: tuple[Path, ...]
    image_count: int
    label_count: int


@dataclass(frozen=True)
class DatasetConfig:
    yaml_path: Path | None
    root: Path
    task: DatasetTask
    names: dict[int, str]
    train: DatasetSplit
    val: DatasetSplit
    test: DatasetSplit | None = None


@dataclass(frozen=True)
class _SplitValidation:
    label_sources: tuple[Path, ...]
    image_count: int
    label_count: int


def validate_dataset_config(
    config: str | Path | Mapping[str, Any],
    *,
    task: DatasetTask = "detect",
    require_test: bool = False,
    strict_label_check: bool = True,
) -> DatasetConfig:
    """Validate an Ultralytics-style YOLO dataset config before training."""

    if task not in _SUPPORTED_TASKS:
        raise DatasetValidationError(f"Unsupported dataset task {task!r}; expected 'detect' or 'segment'.")

    data, yaml_path, base_dir = _load_config(config)
    root = _resolve_dataset_root(data, base_dir)
    names = _normalize_names(data.get("names"), data.get("nc"))

    train = _validate_required_split(
        data,
        "train",
        root=root,
        task=task,
        class_count=len(names),
        strict_label_check=strict_label_check,
    )
    val = _validate_required_split(
        data,
        "val",
        root=root,
        task=task,
        class_count=len(names),
        strict_label_check=strict_label_check,
    )
    test = None
    if "test" in data or require_test:
        test = _validate_required_split(
            data,
            "test",
            root=root,
            task=task,
            class_count=len(names),
            strict_label_check=strict_label_check,
        )

    return DatasetConfig(
        yaml_path=yaml_path,
        root=root,
        task=task,
        names=names,
        train=train,
        val=val,
        test=test,
    )


def _load_config(config: str | Path | Mapping[str, Any]) -> tuple[dict[str, Any], Path | None, Path]:
    if isinstance(config, str | Path):
        config_path = Path(config).expanduser()
        if not config_path.exists():
            raise DatasetValidationError(f"Dataset config does not exist: {config_path}")
        if not config_path.is_file():
            raise DatasetValidationError(f"Dataset config is not a file: {config_path}")

        try:
            text = config_path.read_text(encoding="utf-8")
            loaded = yaml.safe_load(text)
        except UnicodeDecodeError as exc:
            raise DatasetValidationError(f"Dataset config is not valid UTF-8: {config_path}") from exc
        except yaml.YAMLError as exc:
            raise DatasetValidationError(f"Dataset config is malformed YAML: {config_path}: {exc}") from exc

        if not isinstance(loaded, Mapping):
            raise DatasetValidationError(f"Dataset config must be a YAML mapping: {config_path}")
        return dict(loaded), config_path.resolve(), config_path.parent.resolve()

    if isinstance(config, Mapping):
        return dict(config), None, Path.cwd().resolve()

    raise DatasetValidationError(f"Dataset config must be a path or mapping, got {type(config).__name__}.")


def _resolve_dataset_root(data: Mapping[str, Any], base_dir: Path) -> Path:
    raw_root = data.get("path")
    if raw_root is None:
        return base_dir
    if not isinstance(raw_root, str) or not raw_root.strip():
        raise DatasetValidationError("Dataset config 'path' must be a non-empty string when provided.")

    root = _resolve_path(raw_root, base_dir)
    if not root.exists():
        raise DatasetValidationError(f"Dataset root does not exist: {root}")
    if not root.is_dir():
        raise DatasetValidationError(f"Dataset root is not a directory: {root}")
    return root


def _normalize_names(raw_names: Any, raw_nc: Any) -> dict[int, str]:
    if raw_names is None:
        raise DatasetValidationError("Dataset config is missing required 'names' class metadata.")

    if isinstance(raw_names, list | tuple):
        names = {index: _validate_class_name(value, index) for index, value in enumerate(raw_names)}
    elif isinstance(raw_names, Mapping):
        names = {}
        for raw_key, raw_value in raw_names.items():
            key = _normalize_class_key(raw_key)
            if key in names:
                raise DatasetValidationError(f"Dataset class index {key} is duplicated in 'names'.")
            names[key] = _validate_class_name(raw_value, key)
    else:
        raise DatasetValidationError("Dataset config 'names' must be a list or mapping.")

    if not names:
        raise DatasetValidationError("Dataset config 'names' must contain at least one class.")

    expected_keys = list(range(len(names)))
    actual_keys = sorted(names)
    if actual_keys != expected_keys:
        raise DatasetValidationError(
            "Dataset config 'names' keys must be contiguous and zero-based; "
            f"expected {expected_keys}, got {actual_keys}."
        )

    duplicate_names = _duplicate_class_names(names)
    if duplicate_names:
        raise DatasetValidationError(f"Dataset config 'names' contains duplicate class name {duplicate_names[0]!r}.")

    if raw_nc is not None:
        if isinstance(raw_nc, bool) or not isinstance(raw_nc, int) or raw_nc <= 0:
            raise DatasetValidationError("Dataset config 'nc' must be a positive integer when provided.")
        if raw_nc != len(names):
            raise DatasetValidationError(f"Dataset config 'nc' is {raw_nc}, but 'names' defines {len(names)} classes.")

    return names


def _normalize_class_key(raw_key: Any) -> int:
    if isinstance(raw_key, bool):
        raise DatasetValidationError(f"Dataset class index must be an integer, got {raw_key!r}.")
    if isinstance(raw_key, int):
        key = raw_key
    elif isinstance(raw_key, str) and _CLASS_ID_RE.match(raw_key.strip()):
        key = int(raw_key)
    else:
        raise DatasetValidationError(f"Dataset class index must be an integer, got {raw_key!r}.")
    if key < 0:
        raise DatasetValidationError(f"Dataset class index must be non-negative, got {key}.")
    return key


def _validate_class_name(raw_value: Any, index: int) -> str:
    if not isinstance(raw_value, str):
        raise DatasetValidationError(f"Dataset class name at index {index} must be a string.")
    value = raw_value.strip()
    if not value:
        raise DatasetValidationError(f"Dataset class name at index {index} must not be empty.")
    return value


def _duplicate_class_names(names: Mapping[int, str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for name in names.values():
        if name in seen:
            duplicates.append(name)
        seen.add(name)
    return duplicates


def _validate_required_split(
    data: Mapping[str, Any],
    split_name: str,
    *,
    root: Path,
    task: DatasetTask,
    class_count: int,
    strict_label_check: bool,
) -> DatasetSplit:
    if split_name not in data:
        raise DatasetValidationError(f"Dataset config is missing required '{split_name}' split.")

    image_sources = _resolve_split_sources(data[split_name], split_name, root)
    label_sources: list[Path] = []
    image_count = 0
    label_count = 0

    for image_source in image_sources:
        validation = _validate_split_source(
            image_source,
            split_name=split_name,
            task=task,
            class_count=class_count,
            strict_label_check=strict_label_check,
        )
        label_sources.extend(validation.label_sources)
        image_count += validation.image_count
        label_count += validation.label_count

    return DatasetSplit(
        name=split_name,
        images=image_sources,
        labels=tuple(dict.fromkeys(label_sources)),
        image_count=image_count,
        label_count=label_count,
    )


def _resolve_split_sources(raw_split: Any, split_name: str, root: Path) -> tuple[Path, ...]:
    if isinstance(raw_split, str | Path):
        raw_sources = [raw_split]
    elif isinstance(raw_split, list | tuple):
        raw_sources = list(raw_split)
    else:
        raise DatasetValidationError(f"Dataset split '{split_name}' must be a path string or list of path strings.")

    if not raw_sources:
        raise DatasetValidationError(f"Dataset split '{split_name}' must not be empty.")

    sources: list[Path] = []
    for raw_source in raw_sources:
        if not isinstance(raw_source, str | Path) or not str(raw_source).strip():
            raise DatasetValidationError(f"Dataset split '{split_name}' contains an invalid path entry.")
        sources.append(_resolve_path(raw_source, root))
    return tuple(sources)


def _validate_split_source(
    image_source: Path,
    *,
    split_name: str,
    task: DatasetTask,
    class_count: int,
    strict_label_check: bool,
) -> _SplitValidation:
    if not image_source.exists():
        raise DatasetValidationError(f"Dataset split '{split_name}' image path does not exist: {image_source}")

    if image_source.is_dir():
        return _validate_image_directory(
            image_source,
            split_name=split_name,
            task=task,
            class_count=class_count,
            strict_label_check=strict_label_check,
        )

    if image_source.is_file():
        suffix = image_source.suffix.lower()
        if suffix == ".txt":
            return _validate_image_list(
                image_source,
                split_name=split_name,
                task=task,
                class_count=class_count,
                strict_label_check=strict_label_check,
            )
        if suffix in _IMAGE_SUFFIXES:
            return _validate_single_image(
                image_source,
                split_name=split_name,
                task=task,
                class_count=class_count,
                strict_label_check=strict_label_check,
            )

    raise DatasetValidationError(
        f"Dataset split '{split_name}' image path must be a directory, image-list .txt file, or image file: "
        f"{image_source}"
    )


def _validate_image_directory(
    image_dir: Path,
    *,
    split_name: str,
    task: DatasetTask,
    class_count: int,
    strict_label_check: bool,
) -> _SplitValidation:
    label_dir = _derive_label_path(image_dir, split_name=split_name, source_kind="image directory")
    if not label_dir.exists():
        raise DatasetValidationError(f"Dataset split '{split_name}' label directory does not exist: {label_dir}")
    if not label_dir.is_dir():
        raise DatasetValidationError(f"Dataset split '{split_name}' label path is not a directory: {label_dir}")

    validation = _validate_expected_labels(
        _iter_images(image_dir),
        split_name=split_name,
        task=task,
        class_count=class_count,
        strict_label_check=strict_label_check,
    )
    if validation.image_count == 0:
        raise DatasetValidationError(f"Dataset split '{split_name}' image directory contains no images: {image_dir}")
    return _SplitValidation(
        label_sources=(label_dir,),
        image_count=validation.image_count,
        label_count=validation.label_count,
    )


def _validate_image_list(
    list_path: Path,
    *,
    split_name: str,
    task: DatasetTask,
    class_count: int,
    strict_label_check: bool,
) -> _SplitValidation:
    validation = _validate_expected_labels(
        _iter_image_list(list_path, split_name),
        split_name=split_name,
        task=task,
        class_count=class_count,
        strict_label_check=strict_label_check,
        collect_label_sources="dirs",
    )
    if validation.image_count == 0:
        raise DatasetValidationError(f"Dataset split '{split_name}' image list contains no images: {list_path}")
    return validation


def _validate_single_image(
    image_path: Path,
    *,
    split_name: str,
    task: DatasetTask,
    class_count: int,
    strict_label_check: bool,
) -> _SplitValidation:
    _validate_image_file(image_path, split_name)
    return _validate_expected_labels(
        (image_path,),
        split_name=split_name,
        task=task,
        class_count=class_count,
        strict_label_check=strict_label_check,
        collect_label_sources="files",
    )


def _iter_images(image_dir: Path) -> Iterator[Path]:
    for path in image_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES:
            yield path


def _iter_image_list(list_path: Path, split_name: str) -> Iterator[Path]:
    try:
        with list_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                image_path = _resolve_path(line, list_path.parent)
                _validate_image_file(image_path, split_name, line_number=line_number)
                yield image_path
    except UnicodeDecodeError as exc:
        raise DatasetValidationError(
            f"Dataset split '{split_name}' image list is not valid UTF-8: {list_path}"
        ) from exc


def _validate_image_file(image_path: Path, split_name: str, *, line_number: int | None = None) -> None:
    location = f" at line {line_number}" if line_number is not None else ""
    if not image_path.exists():
        raise DatasetValidationError(f"Dataset split '{split_name}' image file{location} does not exist: {image_path}")
    if not image_path.is_file():
        raise DatasetValidationError(f"Dataset split '{split_name}' image path{location} is not a file: {image_path}")
    if image_path.suffix.lower() not in _IMAGE_SUFFIXES:
        raise DatasetValidationError(
            f"Dataset split '{split_name}' image file{location} has unsupported suffix: {image_path}"
        )


def _validate_expected_labels(
    image_paths: Iterable[Path],
    *,
    split_name: str,
    task: DatasetTask,
    class_count: int,
    strict_label_check: bool,
    collect_label_sources: Literal["none", "dirs", "files"] = "none",
) -> _SplitValidation:
    label_sources: list[Path] = []
    seen_label_sources: set[Path] = set()
    first_missing_label: Path | None = None
    missing_label_count = 0
    image_count = 0
    label_count = 0

    for image_path in image_paths:
        image_count += 1
        label_path = _derive_label_path(image_path, split_name=split_name, source_kind="image file").with_suffix(".txt")
        if collect_label_sources == "dirs":
            _append_unique_label_source(label_sources, seen_label_sources, label_path.parent)
        elif collect_label_sources == "files":
            _append_unique_label_source(label_sources, seen_label_sources, label_path)

        if not label_path.exists():
            if strict_label_check:
                missing_label_count += 1
                if first_missing_label is None:
                    first_missing_label = label_path
            continue
        if not label_path.is_file():
            raise DatasetValidationError(f"Dataset split '{split_name}' label path is not a file: {label_path}")

        _validate_label_file(label_path, split_name=split_name, task=task, class_count=class_count)
        label_count += 1

    if first_missing_label is not None:
        raise DatasetValidationError(
            f"Dataset split '{split_name}' is missing {missing_label_count} label file(s); "
            f"first missing: {first_missing_label}"
        )
    return _SplitValidation(
        label_sources=tuple(label_sources),
        image_count=image_count,
        label_count=label_count,
    )


def _append_unique_label_source(label_sources: list[Path], seen_label_sources: set[Path], label_source: Path) -> None:
    if label_source in seen_label_sources:
        return
    seen_label_sources.add(label_source)
    label_sources.append(label_source)


def _validate_label_file(label_path: Path, *, split_name: str, task: DatasetTask, class_count: int) -> None:
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise DatasetValidationError(
            f"Dataset split '{split_name}' label file is not valid UTF-8: {label_path}"
        ) from exc

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        _validate_label_row(
            line.split(),
            label_path=label_path,
            line_number=line_number,
            split_name=split_name,
            task=task,
            class_count=class_count,
        )


def _validate_label_row(
    tokens: list[str],
    *,
    label_path: Path,
    line_number: int,
    split_name: str,
    task: DatasetTask,
    class_count: int,
) -> None:
    if task == "detect" and len(tokens) != 5:
        raise DatasetValidationError(
            f"Dataset split '{split_name}' label {label_path}:{line_number} must contain exactly "
            "'class x_center y_center width height' for detection."
        )
    if task == "segment" and (len(tokens) < 7 or (len(tokens) - 1) % 2 != 0):
        raise DatasetValidationError(
            f"Dataset split '{split_name}' label {label_path}:{line_number} must contain a class id followed by "
            "at least three x/y polygon points for segmentation."
        )

    class_id = _parse_class_id(tokens[0], label_path=label_path, line_number=line_number, split_name=split_name)
    if class_id >= class_count:
        raise DatasetValidationError(
            f"Dataset split '{split_name}' label {label_path}:{line_number} uses class id {class_id}, "
            f"but the config defines {class_count} class(es)."
        )

    for token in tokens[1:]:
        value = _parse_coordinate(token, label_path=label_path, line_number=line_number, split_name=split_name)
        if value < 0.0 or value > 1.0:
            raise DatasetValidationError(
                f"Dataset split '{split_name}' label {label_path}:{line_number} has coordinate {value}, "
                "but YOLO training labels must be normalized to [0, 1]."
            )


def _parse_class_id(token: str, *, label_path: Path, line_number: int, split_name: str) -> int:
    if not _CLASS_ID_RE.match(token):
        raise DatasetValidationError(
            f"Dataset split '{split_name}' label {label_path}:{line_number} has non-integer class id {token!r}."
        )
    class_id = int(token)
    if class_id < 0:
        raise DatasetValidationError(
            f"Dataset split '{split_name}' label {label_path}:{line_number} has negative class id {class_id}."
        )
    return class_id


def _parse_coordinate(token: str, *, label_path: Path, line_number: int, split_name: str) -> float:
    try:
        value = float(token)
    except ValueError as exc:
        raise DatasetValidationError(
            f"Dataset split '{split_name}' label {label_path}:{line_number} has non-numeric coordinate {token!r}."
        ) from exc
    if not math.isfinite(value):
        raise DatasetValidationError(
            f"Dataset split '{split_name}' label {label_path}:{line_number} has non-finite coordinate {token!r}."
        )
    return value


def _derive_label_path(source_path: Path, *, split_name: str, source_kind: str) -> Path:
    parts = source_path.parts
    image_indexes = [index for index, part in enumerate(parts) if part.lower() == "images"]
    if not image_indexes:
        raise DatasetValidationError(
            f"Dataset split '{split_name}' cannot derive a label path for {source_kind}: {source_path}. "
            "Use a standard YOLO layout with paired 'images/' and 'labels/' path components."
        )

    index = image_indexes[-1]
    label_parts = list(parts)
    label_parts[index] = "labels"
    return Path(*label_parts)


def _resolve_path(raw_path: str | Path, base_dir: Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve(strict=False)
