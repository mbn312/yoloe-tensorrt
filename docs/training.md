# Training

`yoloe-tensorrt` delegates training and fine-tuning to Ultralytics, then can export the resulting checkpoint directly
into the same prompt-capable TensorRT artifact format used for inference.

## End-to-end flow

The normal workflow is:

1. Prepare an Ultralytics-compatible dataset YAML and YOLO labels.
2. Train with `yoloe-train` or `train_model(...)`.
3. Optionally export the trained `best.pt` or `last.pt` checkpoint into an artifact bundle.
4. Load that bundle with `YOLOEEngine.from_engine(...)` and use the normal runtime prompt APIs.

## Dataset format

A minimal dataset layout looks like this:

```text
my-dataset/
  images/
    train/
      img0.jpg
    val/
      img1.jpg
  labels/
    train/
      img0.txt
    val/
      img1.txt
  data.yaml
```

A minimal Ultralytics-compatible dataset YAML:

```yaml
path: /data/my-dataset
train: images/train
val: images/val
names:
  0: person
  1: helmet
```

`train` and `val` are required. `test` is optional unless `require_test=True` is passed. Split values can be
directories, image-list `.txt` files, image files, or lists of those paths. Relative split paths resolve against
`path` when it is provided, otherwise against the YAML file directory.

Minimal detection label example:

```text
0 0.5 0.5 0.25 0.25
```

Minimal segmentation label example:

```text
0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75
```

Use detection labels with `task="detect"` and polygon labels with `task="segment"`. Do not mix label formats within
the same run. Label coordinates must be normalized into `[0, 1]`.

The package can validate a dataset config before training:

```python
from yoloe_tensorrt import validate_dataset_config

dataset = validate_dataset_config("data.yaml", task="detect")
print(dataset.names)
print(dataset.train.image_count, dataset.val.image_count)
```

## Train with the CLI

Use `yoloe-train` for the standard command-line workflow:

```bash
yoloe-train yoloe-26s-seg.pt data.yaml \
  --task segment \
  --imgsz 640 \
  --epochs 50 \
  --batch 8 \
  --device cuda:0 \
  --output-dir outputs/training \
  --name seg-run
```

The same command is available through the module entry point:

```bash
python -m yoloe_tensorrt train yoloe-26s-seg.pt data.yaml --task segment
```

Resume an interrupted run with `--resume`, or pass a specific checkpoint:

```bash
yoloe-train yoloe-26s-seg.pt data.yaml --resume
yoloe-train yoloe-26s-seg.pt data.yaml --resume-checkpoint outputs/training/seg-run/weights/last.pt
```

Pass advanced Ultralytics trainer options with repeatable `--ultralytics-arg KEY=VALUE` entries. Values are parsed as
YAML:

```bash
yoloe-train yoloe-26s-seg.pt data.yaml \
  --ultralytics-arg workers=4 \
  --ultralytics-arg optimizer=AdamW \
  --ultralytics-arg lr0=0.001
```

By default, CLI run outputs go under `outputs/training`. On success, the CLI prints the selected checkpoint path and
run directory, plus a metrics file when Ultralytics created one:

```text
checkpoint: outputs/training/seg-run/weights/best.pt
run_dir: outputs/training/seg-run
metrics: outputs/training/seg-run/results.csv
```

## Train with the Python API

Use `train_model(...)` when you want training and export to stay inside Python:

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
print(result.best_checkpoint_path)
print(result.last_checkpoint_path)
print(result.run_dir)
print(result.metrics_path)
```

The result includes:

- `checkpoint_path`: the primary trained checkpoint used by the wrapper, preferring `best.pt`
- `best_checkpoint_path` and `last_checkpoint_path` when those files exist
- `run_dir` for the Ultralytics run directory
- `metrics_path` when Ultralytics created `results.csv`, `results.json`, or `metrics.json`
- `artifact_dir` and `exported_checkpoint_path` when post-training export is enabled

Pass advanced Ultralytics trainer options with `overrides`. Ultralytics keys controlled by the wrapper are reserved in
`overrides`; use the explicit `task`, `imgsz`, `epochs`, `batch`, `device`, `output_dir`, and `name` arguments
instead.

For advanced Ultralytics workflows that need a custom trainer, nonstandard save behavior, or distributed training
orchestration, call Ultralytics directly.

## Export-to-runtime workflow

The simplest end-to-end CLI flow is to train and export in one command:

```bash
yoloe-train yoloe-26s-seg.pt data.yaml \
  --task segment \
  --device cuda:0 \
  --output-dir outputs/training \
  --name seg-run \
  --export \
  --export-artifact-dir outputs/artifacts/seg-run
```

Export the last checkpoint instead of the default best checkpoint:

```bash
yoloe-train yoloe-26s-seg.pt data.yaml \
  --resume \
  --export \
  --export-checkpoint last
```

The training CLI mirrors the export CLI through `--export-*` flags for formats, dynamic or fixed shapes, FP16,
visual-prompt engine generation, export image size, max detections, workspace size, ONNX exporter, opset, and
overwrite behavior.

When export runs successfully, the CLI also prints:

```text
exported_checkpoint: outputs/training/seg-run/weights/best.pt
artifact_dir: outputs/artifacts/seg-run
```

The same workflow is available from Python:

```python
from yoloe_tensorrt import YOLOEEngine, train_model

result = train_model(
    "yoloe-26s-seg.pt",
    "data.yaml",
    task="segment",
    export_artifact=True,
    export_checkpoint="best",
    export_artifact_dir="outputs/artifacts/seg-run",
)

engine = YOLOEEngine.from_engine(result.artifact_dir)
engine.set_classes(["person", "helmet"])
predictions = engine.predict("example.jpg")
print(predictions[0].names)
```

## Output layout

Typical training output layout:

```text
outputs/
  training/
    seg-run/
      weights/
        best.pt
        last.pt
      results.csv      # optional
      results.json     # optional
      metrics.json     # optional
```

Typical exported artifact layout:

```text
outputs/
  artifacts/
    seg-run/
      metadata.json
      prompt_projector.ts
      main.onnx
      main.engine              # when engine export is requested
      visual_prompt.onnx       # when visual export is enabled
      visual_prompt.engine     # when engine export is requested and visual export is enabled
      mobileclip2_b.ts         # optional bundled text encoder asset
```

`metadata.json` can also contain `training_metadata` when the bundle came from the training flow, which makes it
possible to trace the artifact back to the run directory and selected checkpoint.

## Runtime prompts after fine-tuning

Fine-tuning changes the model weights. It does not remove or replace the package's runtime prompt workflow.

- Exported fine-tuned bundles support `engine.set_classes(...)`.
- Exported fine-tuned bundles support `engine.set_visual_prompts(...)` when the visual engine is present.
- Training dataset class names are not automatically activated as runtime labels; your application chooses the
  active prompt set at inference time.

See [Runtime Prompts](prompts.md) for the prompt APIs themselves.

## Jetson guidance

Training runs through Ultralytics and PyTorch, not through the native TensorRT runtime. For Jetson deployments:

- Prefer full training or larger fine-tuning runs on a more capable CUDA system when possible.
- If you do train on Jetson, start with `imgsz=640` and a small batch size such as `1`, `2`, or `auto`.
- Export and build the TensorRT artifact on the target Jetson whenever possible, because TensorRT engine compatibility
  depends on the local JetPack, TensorRT, and CUDA stack.
- A conservative starting export recipe on Jetson is `--export --export-fixed --export-imgsz 640 --export-exporter legacy`.
- Leave visual-prompt export enabled unless the deployment will never use visual prompts.

These are recommended defaults, not hard requirements. Adjust them based on available memory, model size, and target
latency.

## Validation rules

- Class names must be non-empty, unique, and indexed from zero without gaps.
- `nc`, when present, must match the number of class names.
- Image paths must exist and contain supported image files.
- Label paths are derived from the standard paired `images/` and `labels/` YOLO layout.
- Detection labels must use `class x_center y_center width height`.
- Segmentation labels must use `class x1 y1 x2 y2 ...` with at least three polygon points.
- Label class IDs must be in range and coordinates must be finite normalized values in `[0, 1]`.

Dataset validation alone does not create generated outputs.
