#!/usr/bin/env bash
# Serve WebJev-35B-A3B on one GPU: `vllm serve` (localhost only) plus the Jev-compatible adapter (adapter.py) in front.
#
#   MODEL_DIR=/path/to/WebJev-35B-A3B ./serve.sh [extra `vllm serve` args ...]
#
# Environment (all optional except MODEL_DIR):
#   GPU=0                 GPU index; the script refuses a GPU that already holds more than 1 GiB
#   PORT=8200             adapter port: this is the URL clients call (/api/alpha/decisions, /v1/systemone)
#   HOST=127.0.0.1        adapter bind address; any other address requires WEBJEV_API_KEY or WEBJEV_API_KEY_FILE
#   VLLM_PORT=8100        vLLM port, always bound to 127.0.0.1
#   SERVED_MODEL_NAME=webjev-35b-a3b   model id in every response (also vLLM's --served-model-name)
#   VENV=                 virtualenv with vllm + the adapter requirements (default: `vllm` and `python3` on PATH)
#   RUN_DIR=runs/<time>   logs, pid files and the exact vLLM command
#
# vLLM settings: bf16, max_model_len 34816, gpu_memory_utilization 0.9, max_num_seqs 64,
# --logprobs-mode processed_logits --max-logprobs 256 (the adapter reads option-label logits), and two serving choices:
#   - prefix caching ON (vLLM's default). The 1-4 prompts of one browser decision share the page state, so the
#     state prefix is prefilled once.
#   - --max-num-batched-tokens 16384. Decisions are prefill only; vLLM's default 2048-token step on A100 leaves the MoE
#     GEMMs underfed.
# Extra arguments (e.g. --enforce-eager) are passed to `vllm serve` unchanged.
set -euo pipefail
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR to the WebJev-35B-A3B model directory}
GPU=${GPU:-0}
PORT=${PORT:-8200}
HOST=${HOST:-127.0.0.1}
VLLM_PORT=${VLLM_PORT:-8100}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-webjev-35b-a3b}
HERE=$(cd "$(dirname "$0")" && pwd)
RUN_DIR=${RUN_DIR:-$HERE/runs/$(date +%Y%m%d-%H%M%S)}
if [[ -n "${VENV:-}" ]]; then VLLM="$VENV/bin/vllm"; PYTHON="$VENV/bin/python"; else VLLM=vllm; PYTHON=python3; fi

if [[ "$HOST" != 127.0.0.1 && "$HOST" != localhost && -z "${WEBJEV_API_KEY:-}${WEBJEV_API_KEY_FILE:-}" ]]; then
  echo "HOST=$HOST exposes the API beyond this machine: set WEBJEV_API_KEY or WEBJEV_API_KEY_FILE" >&2; exit 2
fi
USED=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)
if (( USED > 1024 )); then echo "GPU $GPU is in use (${USED} MiB); refusing" >&2; exit 2; fi
mkdir -p "$RUN_DIR"

export CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 PYTHONUNBUFFERED=1
VLLM_ARGS=(serve "$MODEL_DIR" --served-model-name "$SERVED_MODEL_NAME" --host 127.0.0.1 --port "$VLLM_PORT"
  --dtype bfloat16 --max-model-len 34816 --gpu-memory-utilization 0.9 --max-num-seqs 64 --max-num-batched-tokens 16384
  --logprobs-mode processed_logits --max-logprobs 256 --seed 0 "$@")
printf '%q ' "$VLLM" "${VLLM_ARGS[@]}" > "$RUN_DIR/vllm.cmd"; echo >> "$RUN_DIR/vllm.cmd"
nohup "$VLLM" "${VLLM_ARGS[@]}" > "$RUN_DIR/vllm.log" 2>&1 < /dev/null &
echo $! > "$RUN_DIR/vllm.pid"
T0=$(date +%s)
echo "starting vLLM on GPU $GPU (log: $RUN_DIR/vllm.log); the first start compiles and can take several minutes"
for _ in $(seq 1 360); do
  curl -sf "http://127.0.0.1:$VLLM_PORT/health" > /dev/null && break
  kill -0 "$(cat "$RUN_DIR/vllm.pid")" 2> /dev/null || { echo "vLLM exited" >&2; tail -40 "$RUN_DIR/vllm.log" >&2; exit 1; }
  sleep 5
done
curl -sf "http://127.0.0.1:$VLLM_PORT/health" > /dev/null || { echo "vLLM not healthy after 30 min" >&2; exit 1; }
echo "vllm_ready_seconds=$(( $(date +%s) - T0 ))" > "$RUN_DIR/ready.txt"

MODEL_DIR="$MODEL_DIR" VLLM_URL="http://127.0.0.1:$VLLM_PORT" SERVED_MODEL_NAME="$SERVED_MODEL_NAME" ADAPTER_TIMING_LOG="$RUN_DIR/adapter-timing.jsonl" \
  nohup "$PYTHON" -m uvicorn adapter:app --app-dir "$HERE" --host "$HOST" --port "$PORT" \
  --log-level warning > "$RUN_DIR/adapter.log" 2>&1 < /dev/null &
echo $! > "$RUN_DIR/adapter.pid"
for _ in $(seq 1 60); do
  curl -sf "http://127.0.0.1:$PORT/health" | grep -q '"ok":true' && break
  sleep 2
done
curl -sf "http://127.0.0.1:$PORT/health" | grep -q '"ok":true' || { echo "adapter not healthy" >&2; tail -40 "$RUN_DIR/adapter.log" >&2; exit 1; }
echo "adapter_ready_seconds=$(( $(date +%s) - T0 ))" >> "$RUN_DIR/ready.txt"
nvidia-smi -i "$GPU" --query-gpu=index,name,memory.used --format=csv,noheader >> "$RUN_DIR/ready.txt"
cat "$RUN_DIR/ready.txt"
echo "WebJev API: http://$HOST:$PORT  (stop: kill \$(cat $RUN_DIR/adapter.pid) \$(cat $RUN_DIR/vllm.pid))"
