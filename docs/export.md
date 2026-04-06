# Export and Artifact Bundles

## Export surfaces

- Python API: `export_model(...)`
- CLI: `yoloe-export`

## Typical export

```bash
yoloe-export yoloe-26s-seg.pt --artifact-dir outputs/artifacts/yoloe26s --fixed --imgsz 640
```

## Exporter modes

- `--exporter auto`: default. On newer torch builds, tries the dynamo ONNX exporter first and falls back to the legacy exporter if YOLOE tracing fails.
- `--exporter legacy`: forces the older `torch.onnx.export` path. This is the most conservative compatibility option.
- `--exporter dynamo`: forces the newer dynamo-based exporter. Use this for debugging or when validating a newer torch environment.

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

## Production guidance

- Prefer prebuilt bundles and `YOLOEEngine.from_engine(...)`.
- Keep bundles outside the source tree in deployment environments.
- Rebuild TensorRT engines on the target environment if TensorRT or JetPack versions differ.
