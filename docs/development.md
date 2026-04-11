# Development

## Local environment

Use a standard Python virtual environment unless your platform already has an equivalent workflow.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

`requirements-dev.txt` includes the repo's editable install plus the current VCS-only Ultralytics CLIP dependency that runtime text prompting needs.

## Native install notes

The editable install above is the normal contributor path. If you only need docs or lightweight checks and do not have TensorRT headers/libs available:

```bash
python -m pip install ".[dev,export,gui,docs]" --config-settings=cmake.define.YOLOE_TRT_BUILD_NATIVE=OFF
```

## Fast verification

```bash
PYTHONPYCACHEPREFIX=outputs/pycache python -m ruff check .
PYTHONPYCACHEPREFIX=outputs/pycache python -m ruff format --check .
```

```bash
PYTHONPYCACHEPREFIX=outputs/pycache python -m pytest -q \
  tests/test_assets.py \
  tests/test_cli.py \
  tests/test_gui_helpers.py \
  tests/test_inputs.py \
  tests/test_logging.py \
  tests/test_prompts.py \
  tests/test_release_metadata.py \
  tests/test_text_encoder_fallback.py \
  tests/test_export.py \
  tests/test_tracking.py
```

## Integration verification

```bash
PYTHONPYCACHEPREFIX=outputs/pycache \
YOLOE_TRT_LOG_LEVEL=INFO \
YOLOE_TRT_TEST_IMGSZ=320 \
YOLOE_TRT_BUILDER_OPT_LEVEL=0 \
YOLOE_TRT_AVG_TIMING_ITERATIONS=1 \
python -m pytest -s -o log_cli=true --log-cli-level=INFO \
  tests/test_runtime_integration.py -m integration
```

## Performance checks

The public API is unified around `YOLOEEngine.predict(...)` and `YOLOEEngine.track(...)`, but the lowest-overhead headless
route is still the prepared-tensor path returned by `YOLOEEngine.prepare_cuda_input(...)`. Benchmark host-image vs
prepared-tensor inference with JSON output under `outputs/benchmarks/`:

```bash
yoloe-benchmark outputs/artifacts/yoloe26s tests/assets/images/bus.jpg \
  --label bus --mode both --runs 200 --warmup 20
```

Jetson camera-path benchmarks use the same command and consume a finite live-source window of `warmup + runs` frames:

```bash
yoloe-benchmark outputs/artifacts/yoloe26s tests/assets/images/bus.jpg \
  --label bus --mode camera --camera-source /dev/video0 \
  --camera-zero-copy auto --runs 200 --warmup 20
```

Each result reports median/p95 latency, FPS, measured-loop process CPU, and measured-loop total system CPU. To compare
against a saved baseline without failing the command, use:

```bash
yoloe-benchmark outputs/artifacts/yoloe26s tests/assets/images/bus.jpg \
  --label bus --mode all --camera-source /dev/video0 \
  --compare-to outputs/benchmarks/<baseline>.json
```

Add `--fail-on-regression` to make the command exit non-zero when the configured latency, FPS, or CPU thresholds are
exceeded, or when the comparison is invalid because no modes or no metrics were compared.

The full benchmark matrix, including text-prompt and visual-prompt update costs, is documented in
[Benchmarks](benchmarks.md).

## CI and release automation

- Public CI runs Ruff, runner-safe unit tests, docs validation, sdist builds, and clean install smoke tests.
- Required public CI does not assume CUDA, TensorRT headers, or a self-hosted Jetson runner.
- Tag pushes like `vX.Y.Z` trigger the release scaffold, which verifies the version and changelog before uploading source artifacts.
- GPU and Jetson-specific validation remain a future optional self-hosted workflow.

## Output conventions

- Put artifacts, caches, logs, and scratch state under `outputs/`.
- Keep checked-in fixtures under `tests/`.
- Avoid reintroducing large checked-in runtime assets.
