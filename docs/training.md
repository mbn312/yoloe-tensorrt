# Training Data

`yoloe-tensorrt` keeps training and fine-tuning execution delegated to Ultralytics, but it provides a preflight helper for
checking YOLOE dataset configs before training starts.

## Validate a dataset config

Use `validate_dataset_config(...)` with an Ultralytics-style dataset YAML:

```python
from yoloe_tensorrt import validate_dataset_config

dataset = validate_dataset_config("data.yaml", task="detect")
print(dataset.names)
print(dataset.train.image_count, dataset.val.image_count)
```

For segmentation datasets, set `task="segment"`:

```python
dataset = validate_dataset_config("data.yaml", task="segment")
```

## Supported config shape

The helper supports the standard YOLO dataset fields:

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

Validation does not create generated outputs. If future training helpers write reports or artifacts, they should place
them under `outputs/`.
