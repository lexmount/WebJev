#!/usr/bin/env bash
# Build the general decision mixture (component 1) from the upstream task registry.
#
#   bash data/general/build_general.sh <base model dir>
#
# Optional: with TEACHER_BASE_URL and TEACHER_API_KEY set (any OpenAI-compatible endpoint), it also generates the
# teacher-written contrastive pairs; without them that one step is skipped (see data/README.md).
#
# Every step is resumable; rerun the script after an interruption. Output: $WEBJEV_WORK/packs/general.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DIR="$(cd "$HERE/../.." && pwd)"
PY="${WEBJEV_DATA_PYTHON:-$TRAIN_DIR/.venv-data/bin/python}"
BASE_MODEL="$(cd "${1:?pass the base model directory (for its tokenizer)}" && pwd)"
WORK="${WEBJEV_WORK:-$TRAIN_DIR/work}"
UPSTREAM="${WEBJEV_UPSTREAM:-$TRAIN_DIR/third_party/decider}"
COMMIT=c4daaac28af9fea95d627015cffa2dd5a5926ee6
export WEBJEV_WORK="$WORK" HF_HUB_DISABLE_IMPLICIT_TOKEN=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 SDL_VIDEODRIVER=dummy

# The builders write generated files (teacher data, caches) into a private working copy of the upstream checkout.
if [[ ! -d "$WORK/general/upstream" ]]; then
  mkdir -p "$WORK/general"
  git clone -q "$UPSTREAM" "$WORK/general/upstream"
fi
[[ "$(git -C "$WORK/general/upstream" rev-parse HEAD)" == "$COMMIT" ]] || git -C "$WORK/general/upstream" checkout -q "$COMMIT"

G="$WORK/general"
[[ -f "$G/cache/trec/verified.json" ]] || "$PY" "$HERE/sources.py" trec
[[ -f "$G/cache/mind2web/verified.json" ]] || "$PY" "$HERE/sources.py" mind2web
[[ -f "$G/upstream/data/mario.pkl" ]] || "$PY" "$HERE/build_mario.py"
"$PY" "$HERE/convert_tasks.py" --workers "${CONVERT_WORKERS:-6}"          # skips finished tasks
[[ -f "$G/base.json" ]] || "$PY" "$HERE/assemble_base.py"
if [[ -n "${TEACHER_BASE_URL:-}" && -n "${TEACHER_API_KEY:-}" ]]; then
  [[ -f "$G/contrastive.json" ]] || "$PY" "$HERE/generate_contrastive.py" --n 6000 --workers "${TEACHER_WORKERS:-24}"   # resumable
elif [[ ! -f "$G/contrastive.json" ]]; then
  echo "note: TEACHER_BASE_URL/TEACHER_API_KEY not set; skipping the teacher-written contrastive pairs (see data/README.md)" >&2
fi
[[ -f "$G/upstream/data/mixture_full.pkl" ]] || "$PY" "$HERE/build_mixture.py"
[[ -f "$G/raw/items.pkl.json" ]] || "$PY" "$HERE/tokenize_items.py" --tokenizer "$BASE_MODEL" --workers "${TOKENIZER_WORKERS:-16}"
[[ -f "$WORK/packs/general/manifest.json" ]] || "$PY" "$HERE/finalize.py"
