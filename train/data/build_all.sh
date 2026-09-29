#!/usr/bin/env bash
# Build the complete training mixture: four components, then one shuffled training file.
#
#   bash data/build_all.sh <base model dir> [benchmark dir]
#
# <base model dir>  the downloaded Qwen/Qwen3.5-35B-A3B-Base (its tokenizer is used; no weights are loaded)
# [benchmark dir]   optional: prepared evaluation benchmark JSONL files, excluded from Open-Jev by exact match
#
# Output: $WEBJEV_WORK/mixture/items.pkl (+ items.pkl.json), which configs/webjev-35b-a3b.yaml points at.
# Every step is resumable or skips finished outputs, so rerun after an interruption.
# Optional: set TEACHER_BASE_URL and TEACHER_API_KEY to also generate the teacher-written contrastive pairs
# (the only step that calls an LLM; see data/README.md). Without them that step is skipped.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DIR="$(cd "$HERE/.." && pwd)"
PY="${WEBJEV_DATA_PYTHON:-$TRAIN_DIR/.venv-data/bin/python}"
BASE_MODEL="$(cd "${1:?pass the base model directory}" && pwd)"
BENCH_DIR="${2:+$(cd "$2" && pwd)}"
export WEBJEV_WORK="${WEBJEV_WORK:-$TRAIN_DIR/work}"
W="$WEBJEV_WORK"

"$PY" "$HERE/download_sources.py" --out "$W/sources"                           # public sources, pinned and hashed
bash "$HERE/general/build_general.sh" "$BASE_MODEL"                             # 1. general decision mixture
[[ -f "$W/packs/web/manifest.json" ]] || "$PY" "$HERE/web.py" --tokenizer "$BASE_MODEL"   # 2. live-web decisions
[[ -f "$W/index/manifest.json" ]] || "$PY" "$HERE/index_components.py"
[[ -f "$W/packs/open_jev/manifest.json" ]] || \
  "$PY" "$HERE/open_jev.py" --sources "$W/sources" --tokenizer "$BASE_MODEL" ${BENCH_DIR:+--bench-dir "$BENCH_DIR"}   # 3. Open-Jev
[[ -f "$W/packs/knowledge_mcqa/manifest.json" ]] || \
  "$PY" "$HERE/knowledge_mcqa.py" --sources "$W/sources" --tokenizer "$BASE_MODEL"   # 4. knowledge MCQA + Nimble
[[ -f "$W/mixture/items.pkl.json" ]] || "$PY" "$HERE/assemble.py"
cat "$W/mixture/items.pkl.json"
