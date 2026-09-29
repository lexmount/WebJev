#!/usr/bin/env bash
# Start training in the background:        ./train.sh [--run-id NAME]
# Check inputs and GPUs, start nothing:     ./train.sh --check
# Resume an interrupted run:                ./train.sh --resume runs/NAME
# Other config:                             ./train.sh --config configs/other.yaml
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${WEBJEV_PYTHON:-$HERE/.venv/bin/python}"
[[ -x "$PY" ]] || { echo "run ./setup.sh first (or set WEBJEV_PYTHON)" >&2; exit 2; }
exec "$PY" "$HERE/src/launch.py" "$@"
