# Benchmarks

`yoloe-benchmark` measures runtime inference paths and writes structured JSON output under `outputs/benchmarks/` by
default. It reports latency, FPS, measured-loop process CPU utilization, and measured-loop total system CPU utilization
for each measured path.

## Host and CUDA paths

```bash
yoloe-benchmark outputs/artifacts/yoloe26s tests/assets/images/bus.jpg \
  --label bus --mode both --runs 200 --warmup 20
```

`--mode both` preserves the historical behavior and runs only the host-memory image path plus the prepared CUDA tensor
path.

## Jetson camera path

```bash
yoloe-benchmark outputs/artifacts/yoloe26s tests/assets/images/bus.jpg \
  --label bus --mode camera --camera-source /dev/video0 \
  --camera-zero-copy auto --runs 200 --warmup 20
```

Camera mode opens the live source once and consumes `warmup + runs` frames. Use `--camera-zero-copy required` when the
benchmark must fail instead of falling back to the CPU appsink path.

## JSON output

Each JSON file contains:

- `schema_version`, `created_at`, `package_version`, and platform metadata
- `config` with artifact, labels, image size, run counts, tracking, and camera settings
- `results` keyed by mode, with `latency_ms`, `fps`, `speed_ms`, `cpu`, and optional `allocations`

Disable file output with `--no-save`, or choose the location with `--output-dir` and `--output-name`.

## Baseline comparison

```bash
yoloe-benchmark outputs/artifacts/yoloe26s tests/assets/images/bus.jpg \
  --label bus --mode all --camera-source /dev/video0 \
  --compare-to outputs/benchmarks/<baseline>.json
```

Comparison mode checks only modes present in both result files. It reports latency and CPU increases plus FPS decreases.
If no modes overlap, the comparison is reported as `no_overlap`; if overlapping modes have no comparable metrics, it is
reported as `no_comparable_metrics`. Add `--fail-on-regression` to exit non-zero when a configured threshold is exceeded
or when the comparison is invalid.
