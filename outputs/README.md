# outputs

Store generated, disposable project output here.

- `artifacts/`: exported ONNX/TensorRT bundles and smoke-test exports
- `assets-cache/`: local downloaded or manually staged large assets that should not live in the repo root
- `dist/`: package build outputs such as sdists
- `pytest_cache/`: pytest cache directory
- `site/`: generated MkDocs site output
- `logs/`: optional runtime or benchmark logs

Do not create new ad hoc output directories in the repository root.
