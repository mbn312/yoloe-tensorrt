# Export and Artifact Bundles

## Export surfaces

- Python API: `export_model(...)`
- CLI: `yoloe-export`

## Typical export

```bash
yoloe-export yoloe-26s-seg.pt --artifact-dir outputs/artifacts/yoloe26s --fixed --imgsz 640
```

## Exporter modes

- `--exporter auto`: default. Tries the dynamo ONNX exporter first and falls back to the legacy exporter if YOLOE tracing fails.
- `--exporter legacy`: forces the `torch.onnx.export` path. This is the most conservative compatibility option.
- `--exporter dynamo`: forces the dynamo-based exporter. Use this for debugging or exporter compatibility checks.

The default opset is exporter-specific:

- legacy: opset 17
- dynamo: opset 18

You can override either path with `--opset-version`.

## Bundle contents

A typical bundle contains:

- `main.onnx`
- `main.engine`
- `prompt_projector.ts`
- `metadata.json`

If visual prompt export is enabled, it may also contain:

- `visual_prompt.onnx`
- `visual_prompt.engine`

`metadata.json` always stores the core runtime/export metadata. When the bundle was produced from the training flow, it
can also include a `training_metadata` block for traceability back to the training run and selected checkpoint.

## Production guidance

- Prefer prebuilt bundles and `YOLOEEngine.from_engine(...)`.
- Keep bundles outside the source tree in deployment environments.
- Rebuild TensorRT engines on the target environment if TensorRT or JetPack versions differ.
