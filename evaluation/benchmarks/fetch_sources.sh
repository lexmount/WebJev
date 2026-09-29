#!/usr/bin/env bash
# Download the sources of the 8 benchmarks at the exact versions we evaluated, into $EVAL_ROOT/sources
# (default: ./work/sources). Nothing is redistributed by this repository; every file comes from its upstream.
#
#   ./fetch_sources.sh
#
# HF_ENDPOINT is honored for the Hugging Face download (e.g. a mirror).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${EVAL_ROOT:-$HERE/work}
SRC="$ROOT/sources"
mkdir -p "$SRC"

clone() {  # clone <github repo> <commit>
  local dir="$SRC/$(basename "$1")"
  [ -d "$dir/.git" ] || git clone -q --filter=blob:none "https://github.com/$1" "$dir"
  git -C "$dir" cat-file -e "$2^{commit}" 2>/dev/null || git -C "$dir" fetch -q origin
  git -C "$dir" checkout -q --detach "$2"
  echo "$1 @ $(git -C "$dir" rev-parse --short HEAD)"
}
clone fstandhartinger/jevbench fd51755eb0c0b546ca206d764faf3302feca913e   # JevBench public items (MIT)
clone bespokelabsai/nimble     f136b3f75721fda4ea961f73993cc50b08488835   # Nimble eval split
clone jaredpalmer/kev          e0bcf50153f1bda4ca6a8be5e12cbd5f9ebbce1c   # kev frozen suites: transfer-v4, decision-v7, SemIf, scienthoon, MMLU-Pro (Apache-2.0)

# typed-decisions test split (LocalLLaMA/typed-decisions, config "all", Apache-2.0), pinned by revision and checksum
REV=468b1461d59e01d6404e1cf37431da962acd43a0
SHA=4f294f218ea1da27f3efef936359389c62ea4d3973a41457732990f1d31b647c
OUT="$SRC/typed-decisions/test.parquet"
mkdir -p "$(dirname "$OUT")"
if [ ! -s "$OUT" ] || ! echo "$SHA  $OUT" | shasum -a 256 -c - > /dev/null 2>&1; then
  curl -sSL --retry 5 -o "$OUT" \
    "${HF_ENDPOINT:-https://huggingface.co}/datasets/LocalLLaMA/typed-decisions/resolve/$REV/all/test-00000-of-00001.parquet"
fi
echo "$SHA  $OUT" | shasum -a 256 -c - > /dev/null || { echo "typed-decisions checksum mismatch" >&2; exit 1; }
echo "LocalLLaMA/typed-decisions @ ${REV:0:7} (all/test)"
echo "sources ready in $SRC"
