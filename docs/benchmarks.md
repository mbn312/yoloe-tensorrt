# Benchmarks

`yoloe-benchmark` measures runtime inference paths and prompt update operations, then writes structured JSON output under
`outputs/benchmarks/` by default. It reports latency, FPS, measured-loop process CPU utilization, and measured-loop total
system CPU utilization for each measured path.

Use the same artifact, image, labels, `--imgsz`, `--runs`, and `--warmup` when comparing results across commits or
devices. The positional image is used by inference and visual-prompt benchmarks; text-only benchmarks accept it for CLI
consistency but do not decode it.

## Benchmark matrix

| Coverage | Command | Result key | Output |
| --- | --- | --- | --- |
| CPU-memory input path | `--mode host` | `host` | `outputs/benchmarks/matrix-host.json` |
| CUDA-tensor input path | `--mode cuda` | `cuda` | `outputs/benchmarks/matrix-cuda.json` |
| Jetson zero-copy camera path | `--mode camera --camera-zero-copy required` | `camera` | `outputs/benchmarks/matrix-camera.json` |
| Text-prompt update cost | `--mode none --text-prompts` | `text-set-classes` | `outputs/benchmarks/matrix-text-prompts.json` |
| Visual-prompt update cost | `--mode none --visual-prompts both --visual-runtime both` | `visual-native-bbox`, `visual-native-mask`, `visual-python-bbox`, `visual-python-mask` when available | `outputs/benchmarks/matrix-visual-prompts.json` |

## Commands

CPU-memory input path:

```bash
yoloe-benchmark outputs/artifacts/yoloe26s tests/assets/images/bus.jpg \
  --label bus --mode host --runs 200 --warmup 20 \
  --output-name matrix-host
```

CUDA-tensor input path:

```bash
yoloe-benchmark outputs/artifacts/yoloe26s tests/assets/images/bus.jpg \
  --label bus --mode cuda --runs 200 --warmup 20 \
  --output-name matrix-cuda
```

Jetson zero-copy camera path:

```bash
yoloe-benchmark outputs/artifacts/yoloe26s tests/assets/images/bus.jpg \
  --label bus --mode camera --camera-source /dev/video0 \
  --camera-zero-copy required --runs 200 --warmup 20 \
  --output-name matrix-camera
```

Text-prompt update cost:

```bash
yoloe-benchmark outputs/artifacts/yoloe26s tests/assets/images/bus.jpg \
  --label bus --label person --mode none --text-prompts \
  --runs 50 --warmup 5 --output-name matrix-text-prompts
```

Visual-prompt update cost:

```bash
yoloe-benchmark outputs/artifacts/yoloe26s tests/assets/images/bus.jpg \
  --label bus --mode none --visual-prompts both --visual-runtime both \
  --runs 50 --warmup 5 --output-name matrix-visual-prompts
```

`--mode both` runs the historical host plus CUDA pair. `--mode all` runs host, CUDA, and camera paths, and therefore
requires `--camera-source`. Use `--mode none` only with `--text-prompts` or `--visual-prompts` for prompt-update-only
benchmarks.

## Output and interpretation

Each JSON file contains:

- `schema_version`, `created_at`, `package_version`, and platform metadata
- `config` with artifact, labels, image size, run counts, tracking, prompt, and camera settings
- `results` keyed by mode, with `latency_ms`, `fps`, `speed_ms`, `cpu`, and optional `allocations`

For inference paths, compare `latency_ms.median`, `latency_ms.p95`, `fps`, `speed_ms.inference.mean`,
`cpu.process_percent.mean`, and `cpu.system_percent.mean`. For prompt-update rows, `latency_ms` is the update cost;
`speed_ms` is inference-specific and should not be used as the prompt-update metric.

Camera mode opens the live source once and consumes `warmup + runs` frames. Use `--camera-zero-copy required` for
regression baselines so an accidental CPU appsink fallback fails instead of polluting the zero-copy result.

Use `--profile-allocations` when checking Python steady-state allocation pressure. The `allocations` object reports
retained Python allocations and traced peak memory for that measured loop.

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
