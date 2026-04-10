from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml
from yoloe_tensorrt import DatasetConfig, DatasetSplit, DatasetValidationError, validate_dataset_config


def _make_dataset(
    tmp_path: Path,
    *,
    segment: bool = False,
    names: list[str] | None = None,
    image_suffix: str = ".jpg",
) -> tuple[Path, Path]:
    root = tmp_path / "dataset"
    for split in ("train", "val"):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        (image_dir / f"img0{image_suffix}").write_bytes(b"")
        label = "0 0.1 0.1 0.5 0.1 0.5 0.5\n" if segment else "0 0.5 0.5 0.25 0.25\n"
        (label_dir / "img0.txt").write_text(label, encoding="utf-8")

    config_path = _write_config(
        tmp_path,
        {
            "path": "dataset",
            "train": "images/train",
            "val": "images/val",
            "names": names or ["person"],
        },
    )
    return root, config_path


def _write_config(base_dir: Path, data: object, *, name: str = "data.yaml") -> Path:
    config_path = base_dir / name
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return config_path


def test_validate_dataset_config_accepts_valid_detection_yaml(tmp_path: Path) -> None:
    root, config_path = _make_dataset(tmp_path)

    config = validate_dataset_config(config_path, task="detect")

    assert isinstance(config, DatasetConfig)
    assert config.yaml_path == config_path.resolve()
    assert config.root == root.resolve()
    assert config.task == "detect"
    assert config.names == {0: "person"}
    assert isinstance(config.train, DatasetSplit)
    assert config.train.image_count == 1
    assert config.train.label_count == 1
    assert config.train.labels == (root.resolve() / "labels" / "train",)
    assert config.val.image_count == 1
    assert config.test is None


def test_validate_dataset_config_accepts_valid_segmentation_yaml(tmp_path: Path) -> None:
    _, config_path = _make_dataset(tmp_path, segment=True)

    config = validate_dataset_config(config_path, task="segment")

    assert config.task == "segment"
    assert config.train.label_count == 1


def test_validate_dataset_config_accepts_mapping_and_normalizes_names(tmp_path: Path) -> None:
    root, _ = _make_dataset(tmp_path, names=["person", "car"])
    (root / "labels" / "train" / "img0.txt").write_text("1 0.5 0.5 0.25 0.25\n", encoding="utf-8")
    (root / "labels" / "val" / "img0.txt").write_text("1 0.5 0.5 0.25 0.25\n", encoding="utf-8")

    config = validate_dataset_config(
        {
            "path": str(root),
            "train": "images/train",
            "val": "images/val",
            "names": {"0": "person", 1: "car"},
            "nc": 2,
        }
    )

    assert config.yaml_path is None
    assert config.names == {0: "person", 1: "car"}


def test_validate_dataset_config_accepts_image_list_splits(tmp_path: Path) -> None:
    root, config_path = _make_dataset(tmp_path)
    train_list = root / "train.txt"
    val_list = root / "val.txt"
    train_list.write_text("images/train/img0.jpg\n", encoding="utf-8")
    val_list.write_text("# comment\n\nimages/val/img0.jpg\n", encoding="utf-8")
    config_path = _write_config(
        tmp_path,
        {
            "path": "dataset",
            "train": "train.txt",
            "val": "val.txt",
            "names": ["person"],
        },
    )

    config = validate_dataset_config(config_path)

    assert config.train.images == (train_list.resolve(),)
    assert config.train.labels == (root.resolve() / "labels" / "train",)
    assert config.train.image_count == 1
    assert config.val.image_count == 1


def test_validate_dataset_config_accepts_ultralytics_image_suffixes(tmp_path: Path) -> None:
    _, config_path = _make_dataset(tmp_path, image_suffix=".heic")

    config = validate_dataset_config(config_path)

    assert config.train.image_count == 1


@pytest.mark.parametrize("missing_split", ["train", "val", "test"])
def test_validate_dataset_config_rejects_missing_required_splits(tmp_path: Path, missing_split: str) -> None:
    _, config_path = _make_dataset(tmp_path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data.pop(missing_split, None)
    config_path = _write_config(tmp_path, data)

    with pytest.raises(DatasetValidationError, match=f"missing required '{missing_split}' split"):
        validate_dataset_config(config_path, require_test=missing_split == "test")


def test_validate_dataset_config_rejects_missing_image_path(tmp_path: Path) -> None:
    _, config_path = _make_dataset(tmp_path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data["train"] = "images/missing"
    config_path = _write_config(tmp_path, data)

    with pytest.raises(DatasetValidationError, match="image path does not exist"):
        validate_dataset_config(config_path)


def test_validate_dataset_config_rejects_missing_label_directory(tmp_path: Path) -> None:
    root, config_path = _make_dataset(tmp_path)
    shutil.rmtree(root / "labels" / "train")

    with pytest.raises(DatasetValidationError, match="label directory does not exist"):
        validate_dataset_config(config_path)


def test_validate_dataset_config_rejects_missing_label_file(tmp_path: Path) -> None:
    root, config_path = _make_dataset(tmp_path)
    (root / "labels" / "train" / "img0.txt").unlink()

    with pytest.raises(DatasetValidationError, match="missing 1 label file"):
        validate_dataset_config(config_path)


def test_validate_dataset_config_rejects_paths_without_yolo_label_layout(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    (root / "train").mkdir(parents=True)
    (root / "val").mkdir()
    (root / "train" / "img0.jpg").write_bytes(b"")
    (root / "val" / "img0.jpg").write_bytes(b"")
    config_path = _write_config(
        tmp_path,
        {
            "path": "dataset",
            "train": "train",
            "val": "val",
            "names": ["person"],
        },
    )

    with pytest.raises(DatasetValidationError, match="standard YOLO layout"):
        validate_dataset_config(config_path)


@pytest.mark.parametrize(
    ("names", "message"),
    [
        ([], "at least one class"),
        (["person", "person"], "duplicate class name"),
        ([""], "must not be empty"),
        ([0], "must be a string"),
        ({1: "person"}, "contiguous and zero-based"),
    ],
)
def test_validate_dataset_config_rejects_invalid_class_names(
    tmp_path: Path,
    names: object,
    message: str,
) -> None:
    _, config_path = _make_dataset(tmp_path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data["names"] = names
    config_path = _write_config(tmp_path, data)

    with pytest.raises(DatasetValidationError, match=message):
        validate_dataset_config(config_path)


def test_validate_dataset_config_rejects_nc_mismatch(tmp_path: Path) -> None:
    _, config_path = _make_dataset(tmp_path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data["nc"] = 2
    config_path = _write_config(tmp_path, data)

    with pytest.raises(DatasetValidationError, match="'nc' is 2"):
        validate_dataset_config(config_path)


def test_validate_dataset_config_rejects_unsupported_task(tmp_path: Path) -> None:
    _, config_path = _make_dataset(tmp_path)

    with pytest.raises(DatasetValidationError, match="Unsupported dataset task"):
        validate_dataset_config(config_path, task="pose")  # type: ignore[arg-type]


def test_validate_dataset_config_rejects_malformed_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "data.yaml"
    config_path.write_text("names: [person\n", encoding="utf-8")

    with pytest.raises(DatasetValidationError, match="malformed YAML"):
        validate_dataset_config(config_path)


def test_validate_dataset_config_rejects_non_utf8_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "data.yaml"
    config_path.write_bytes(b"\xff\xfe\x00")

    with pytest.raises(DatasetValidationError, match="not valid UTF-8"):
        validate_dataset_config(config_path)


def test_validate_dataset_config_rejects_non_mapping_yaml(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, ["not", "a", "mapping"])

    with pytest.raises(DatasetValidationError, match="must be a YAML mapping"):
        validate_dataset_config(config_path)


def test_validate_dataset_config_rejects_detection_labels_for_segmentation(tmp_path: Path) -> None:
    _, config_path = _make_dataset(tmp_path)

    with pytest.raises(DatasetValidationError, match="at least three x/y polygon points"):
        validate_dataset_config(config_path, task="segment")


def test_validate_dataset_config_rejects_segmentation_labels_for_detection(tmp_path: Path) -> None:
    _, config_path = _make_dataset(tmp_path, segment=True)

    with pytest.raises(DatasetValidationError, match="exactly 'class x_center"):
        validate_dataset_config(config_path, task="detect")


@pytest.mark.parametrize(
    ("label", "message"),
    [
        ("1 0.5 0.5 0.25 0.25\n", "uses class id 1"),
        ("0 1.5 0.5 0.25 0.25\n", "must be normalized"),
        ("person 0.5 0.5 0.25 0.25\n", "non-integer class id"),
        ("0 0.5 nope 0.25 0.25\n", "non-numeric coordinate"),
    ],
)
def test_validate_dataset_config_rejects_malformed_label_rows(
    tmp_path: Path,
    label: str,
    message: str,
) -> None:
    root, config_path = _make_dataset(tmp_path)
    (root / "labels" / "train" / "img0.txt").write_text(label, encoding="utf-8")

    with pytest.raises(DatasetValidationError, match=message):
        validate_dataset_config(config_path)
