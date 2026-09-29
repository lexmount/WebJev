#!/usr/bin/env bash
# One-time setup on the training machine (Linux, CUDA 12.8, 8 x A100 80GB):
#   1. the upstream Mapika/decider checkout at the pinned commit (training framework and prompt format);
#   2. .venv       the training environment (torch 2.10 + cu128);
#   3. .venv-data  the CPU environment for building the data mixture.
# Needs git, uv (https://docs.astral.sh/uv/) and a CUDA 12.8 toolkit (causal-conv1d may compile).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPSTREAM="${WEBJEV_UPSTREAM:-$HERE/third_party/decider}"
COMMIT=c4daaac28af9fea95d627015cffa2dd5a5926ee6

if [[ ! -d "$UPSTREAM/.git" ]]; then
  git clone -q https://github.com/Mapika/decider.git "$UPSTREAM"
fi
git -C "$UPSTREAM" fetch -q origin "$COMMIT" 2>/dev/null || true
git -C "$UPSTREAM" checkout -q --detach "$COMMIT"
[[ "$(git -C "$UPSTREAM" rev-parse HEAD)" == "$COMMIT" ]] || { echo "upstream is not at $COMMIT" >&2; exit 1; }

if [[ ! -x "$HERE/.venv/bin/python" ]]; then
  uv venv -q --python 3.11 "$HERE/.venv"
  uv pip install -q --python "$HERE/.venv/bin/python" "torch==2.10.0" --index-url https://download.pytorch.org/whl/cu128
  uv pip install -q --python "$HERE/.venv/bin/python" --no-build-isolation -r "$HERE/requirements.txt"
fi
if [[ ! -x "$HERE/.venv-data/bin/python" ]]; then
  uv venv -q --python 3.11 "$HERE/.venv-data"
  uv pip install -q --python "$HERE/.venv-data/bin/python" "torch==2.10.0" --index-url https://download.pytorch.org/whl/cpu
  uv pip install -q --python "$HERE/.venv-data/bin/python" -r "$HERE/data/requirements.txt"
fi
echo "ok: upstream@${COMMIT:0:8}, $HERE/.venv (training), $HERE/.venv-data (data)"
