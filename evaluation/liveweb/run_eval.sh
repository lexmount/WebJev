#!/usr/bin/env bash
# One command: every model in models.conf runs the 125 tasks in parallel (decision model alone, see harness/episode.py),
# then the deterministic verifier scores every episode, then the summary table is printed.
#
#   ./run_eval.sh                                          all models, all tasks
#   MODELS=WebJev-35B-A3B RUN_TAG=smoke ./run_eval.sh --tasks vts-021-apple-current-macbook-air-specs
#
# Extra flags go to harness/runner.py. Re-running with the same RUN_TAG skips tasks that already have a result;
# only tasks that ended in the infra bucket run again.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MODELS_CONF="${MODELS_CONF:-$HERE/models.conf}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d)}"
RUNS="$HERE/runs/$RUN_TAG"
[ -f "$HERE/.env" ] || { echo "copy .env.example to .env and fill it in" >&2; exit 1; }
set -a; . "$HERE/.env"; set +a
PY="${WEBJEV_AGENT_PYTHON:-python3}"
mkdir -p "$RUNS"

models() { awk '!/^#/ && NF >= 5 {print $1, $2, $3, $4, $5}' "$MODELS_CONF"; }
selected() { [ -z "${MODELS:-}" ] || [[ ",$MODELS," == *",$1,"* ]]; }

pids=()
while read -r name url decision_model key_env workers; do
  selected "$name" || continue
  if [ "$key_env" != "-" ] && [ -z "${!key_env:-}" ] && [[ "$url" != http://127.0.0.1* ]]; then
    echo "$key_env is not set in .env" >&2; exit 1
  fi
  echo "[$name] $decision_model at $url, up to $workers workers, log $RUNS/$name.log"
  "$PY" "$HERE/harness/runner.py" --results-dir "$RUNS/$name" --label "$name" --decision-url "$url" \
    --decision-model "$decision_model" --decision-key-env "$( [ "$key_env" = - ] || echo "$key_env" )" \
    --workers "$workers" --agent-python "$PY" "$@" > "$RUNS/$name.log" 2>&1 &
  pids+=($!)
done < <(models)
status=0
for pid in ${pids[@]+"${pids[@]}"}; do wait "$pid" || status=1; done

runs=()
while read -r name _; do
  selected "$name" || continue
  (cd "$HERE" && "$PY" -m verifier.judge "$RUNS/$name") | sed "s/^/[$name] /"
  runs+=(--run "$name=$RUNS/$name")
done < <(models)
"$PY" "$HERE/harness/summarize.py" "${runs[@]}" --out-dir "$RUNS/summary"
exit $status
