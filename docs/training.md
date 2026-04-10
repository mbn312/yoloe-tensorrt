# Training

`yoloe-tensorrt` delegates YOLOE training and fine-tuning to Ultralytics while providing a package-level API that
validates dataset configs before launching a run.

## Train or fine-tune

Use `train_model(...)` when you want the package to validate the dataset and call Ultralytics for you:

```python
from yoloe_tensorrt import train_model

result = train_model(
    "yoloe-26s-seg.pt",
    "data.yaml",
    task="segment",
    imgsz=640,
    epochs=50,
    batch=8,
    device="cuda:0",
)

print(result.checkpoint_path)
print(result.run_dir)
```

By default, run outputs are placed under `outputs/training`. The result includes the selected checkpoint path, the
Ultralytics run directory, an optional metrics file path when one exists, and basic metadata about the run.

Pass advanced Ultralytics trainer options with `overrides`:

```python
result = train_model(
    "yoloe-26s-seg.pt",
    "data.yaml",
    overrides={"workers": 4, "optimizer": "AdamW", "lr0": 0.001},
)
```

Ultralytics keys controlled by the wrapper are reserved in `overrides`. Pass `data` as `dataset_config`, `project` as
`output_dir`, and use the explicit `task`, `imgsz`, `epochs`, `batch`, `device`, and `name` arguments.

For advanced Ultralytics workflows that need a custom trainer, nonstandard save behavior, or distributed training
orchestration, call Ultralytics directly.

## Validate a dataset config

Use `validate_dataset_config(...)` directly when you only need a preflight check:

```python
from yoloe_tensorrt import validate_dataset_config

dataset = validate_dataset_config("data.yaml", task="detect")
print(dataset.names)
print(dataset.train.image_count, dataset.val.image_count)
```

For segmentation datasets, set `task="segment"`.

## Supported config shape

The training and validation helpers support the standard YOLO dataset fields:

```yaml
path: /data/my-dataset
train: images/train
val: images/val
names:
  0: person
  1: car
```

`train` and `val` are required. `test` is optional unless `require_test=True` is passed. Split values can be directories,
image-list `.txt` files, image files, or lists of those paths. Relative split paths resolve against `path` when it is
provided, otherwise against the YAML file directory.

## Validation rules

- Class names must be non-empty, unique, and indexed from zero without gaps.
- `nc`, when present, must match the number of class names.
- Image paths must exist and contain supported image files.
- Label paths are derived from the standard paired `images/` and `labels/` YOLO layout.
- Detection labels must use `class x_center y_center width height`.
- Segmentation labels must use `class x1 y1 x2 y2 ...` with at least three polygon points.
- Label class IDs must be in range and coordinates must be finite normalized values in `[0, 1]`.

Dataset validation alone does not create generated outputs.
