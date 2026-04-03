#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-outputs/pycache}"

LAUNCHER="${ROOT_DIR}/scripts/launch_camera_gui.py"

if command -v yoloe-camera-gui >/dev/null 2>&1; then
  exec yoloe-camera-gui "$@"
fi

if command -v python3 >/dev/null 2>&1 && python3 - <<'PY' >/dev/null 2>&1
import importlib.util
import sys

required = ["tkinter", "cv2", "PIL", "torch", "tensorrt", "ultralytics"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
raise SystemExit(0 if not missing else 1)
PY
then
  exec python3 "${LAUNCHER}" "$@"
fi

if command -v python >/dev/null 2>&1 && python - <<'PY' >/dev/null 2>&1
import importlib.util
import sys

required = ["tkinter", "cv2", "PIL", "torch", "tensorrt", "ultralytics"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
raise SystemExit(0 if not missing else 1)
PY
then
  exec python "${LAUNCHER}" "$@"
fi

echo "Unable to launch the GUI from the current environment." >&2
echo "Install the package first, for example:" >&2
echo "  python -m pip install \"yoloe-tensorrt[gui]\"" >&2
echo "or from a local checkout:" >&2
echo "  python -m pip install -r requirements-dev.txt" >&2
exit 1
