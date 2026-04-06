# Contributing

## Scope

This project targets the fastest practical YOLOE TensorRT runtime that still exposes a Python package API. Jetson is the primary optimization target. Changes that accidentally push work back to CPU or pure-Python hot paths should be treated as regressions.

## Local Environment

Create a standard virtual environment and install the contributor dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

The contributor requirements file also installs the current VCS-only Ultralytics CLIP dependency used by runtime text prompting.

If you do not have TensorRT headers/libs available and only need docs or lightweight unit tests, install with the native build disabled:

```bash
python -m pip install ".[dev,export,gui,docs]" --config-settings=cmake.define.YOLOE_TRT_BUILD_NATIVE=OFF
```

## Repository Conventions

- Keep source in `yoloe_tensorrt/` and `src/native/`.
- Keep tests and checked-in fixtures in `tests/`.
- Route generated artifacts, caches, and logs into `outputs/`.
- Do not add ad hoc root-level artifact directories.
- Prefer `YOLOEEngine.from_engine(...)` for demos and performance-sensitive tests.

## Verification

CI-required checks:

- `ruff check .`
- `ruff format --check .`
- runner-safe pytest subset
- `mkdocs build --strict`
- `python -m build --sdist`
- clean install smoke with `YOLOE_TRT_BUILD_NATIVE=OFF`

Fast unit pass:

```bash
PYTHONPYCACHEPREFIX=outputs/pycache python -m ruff check .
PYTHONPYCACHEPREFIX=outputs/pycache python -m ruff format --check .
```

```bash
PYTHONPYCACHEPREFIX=outputs/pycache python -m pytest -q \
  tests/test_assets.py \
  tests/test_cli.py \
  tests/test_gui_helpers.py \
  tests/test_logging.py \
  tests/test_prompts.py \
  tests/test_release_metadata.py \
  tests/test_text_encoder_fallback.py \
  tests/test_export.py
```

Native TensorRT integration pass:

```bash
PYTHONPYCACHEPREFIX=outputs/pycache \
YOLOE_TRT_LOG_LEVEL=INFO \
YOLOE_TRT_TEST_IMGSZ=320 \
YOLOE_TRT_BUILDER_OPT_LEVEL=0 \
YOLOE_TRT_AVG_TIMING_ITERATIONS=1 \
python -m pytest -s -o log_cli=true --log-cli-level=INFO \
  tests/test_runtime_integration.py -m integration
```

## Releases

- Bump the version in `pyproject.toml`.
- Add the matching released section to `CHANGELOG.md`.
- Push a tag in the form `vX.Y.Z`.
- The release workflow will verify metadata and upload non-publishing release artifacts.

## Pull Requests

- Keep changes scoped.
- Document verification that actually ran.
- Call out unsupported assumptions, hardware-specific behavior, or missing test coverage.
- Update `README.md`, `docs/`, and `AGENTS.md` when repo conventions or public entry points change.
