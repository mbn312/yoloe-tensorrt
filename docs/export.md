# Export and Artifact Bundles

## Export surfaces

- Python API: `export_model(...)`
- CLI: `yoloe-export`

## Typical export

```bash
yoloe-export yoloe-26s-seg.pt --artifact-dir outputs/artifacts/yoloe26s --fixed --imgsz 640
```

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
