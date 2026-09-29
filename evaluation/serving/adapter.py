"""Jev-compatible HTTP API for WebJev-35B-A3B, in front of a stock `vllm serve` OpenAI server.

Routes (request and response bodies have the same shape as Jev's):

    POST /v1/systemone          TypeSafe's System One route
    POST /api/alpha/decisions   OpenRouter's Decisions route
    GET  /v1/models             the served model id
    GET  /health                liveness of the adapter and of vLLM (never needs a key)

How one request is answered:

* Prompt. The upstream `decider` package (https://github.com/Mapika/decider, Apache-2.0) renders the state and every
  typed question (`systemone.render_state`, `render_question`, `plan_rows`), tokenizes each question with
  `prompt.build`, and assembles the answers (`systemone.assemble`). WebJev-35B-A3B was trained with exactly this
  prompt code. The package is imported from `$MODEL_DIR/decider` when the model directory ships it, otherwise from the
  installed package (`pip install --no-deps "decider @ git+https://github.com/Mapika/decider@c4daaac28af9fea95d627015cffa2dd5a5926ee6"`).
* Forward pass. Every question is one vLLM request whose prompt ends at the answer slot `Answer: (`. vLLM samples one
  token with the question's option-label tokens as the only allowed ids and returns their processed logits
  (`vllm serve --logprobs-mode processed_logits --max-logprobs 256`, see serve.sh). The adapter softmaxes those logits
  over the options at the model's temperature. Nothing is generated: a decision is one prefill.
* Like Jev, a `choice` question with exactly one option is answered with that option at probability 1 without calling
  the model (the prompt format needs 2 to 255 options; browser agents send one-option target questions often).
* `independent: false` would pack several answer slots into one prompt; vLLM only reads the last position, so such
  requests are rejected with HTTP 422.

Environment:

    MODEL_DIR                  model directory (weights, tokenizer, decider_config.json)       required
    VLLM_URL                   vLLM OpenAI server                                              http://127.0.0.1:8100
    SERVED_MODEL_NAME          model id returned in every response (and vLLM's served name)    webjev-35b-a3b
    VLLM_SERVED_MODEL          --served-model-name given to vLLM, if different                 $SERVED_MODEL_NAME
    WEBJEV_API_KEY             if set, every route except /health needs "Authorization: Bearer <key>"
    WEBJEV_API_KEY_FILE        read the key from this file instead
    WEBJEV_MAX_STATE_TOKENS    token budget of the rendered state                              max_state_tokens of the model config, else 32768
    WEBJEV_TEMPERATURE         softmax temperature over option logits                          temperature of the model config, else 1.0
    ADAPTER_TIMING_LOG         optional JSONL file: rows, prompt tokens, prepare and vLLM time per request

    MODEL_DIR=/path/to/WebJev-35B-A3B VLLM_URL=http://127.0.0.1:8100 \
        python -m uvicorn adapter:app --host 127.0.0.1 --port 8200
"""
import asyncio
import hmac
import json
import math
import os
import sys
import time
from dataclasses import dataclass

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

MODEL_DIR = os.environ["MODEL_DIR"]
VLLM_URL = os.environ.get("VLLM_URL", "http://127.0.0.1:8100").rstrip("/")
MODEL_NAME = os.environ.get("SERVED_MODEL_NAME", "webjev-35b-a3b")   # public id; never derived from the directory name
VLLM_SERVED_MODEL = os.environ.get("VLLM_SERVED_MODEL", MODEL_NAME)
TIMING_LOG = os.environ.get("ADAPTER_TIMING_LOG", "")

if os.path.isdir(os.path.join(MODEL_DIR, "decider")):
    sys.path.insert(0, MODEL_DIR)  # the model directory ships the prompt package it was trained with
from transformers import AutoTokenizer  # noqa: E402
from decider import systemone as S1  # noqa: E402  (upstream prompt package, Mapika/decider)
from decider.prompt import MAX_OPTIONS, build, label_table  # noqa: E402

# Model settings, in the upstream model-directory format.
CFG = json.load(open(os.path.join(MODEL_DIR, "decider_config.json")))
TEMP = float(os.environ.get("WEBJEV_TEMPERATURE", CFG.get("temperature", 1.0)))
MAX_STATE_TOKENS = int(os.environ.get("WEBJEV_MAX_STATE_TOKENS", CFG.get("max_state_tokens", 32768)))
ISOLATED = bool(CFG.get("isolated_levels", False))  # score questions: one yes/no row per level
TOK = AutoTokenizer.from_pretrained(MODEL_DIR)
_, LABEL_IDS, _ = label_table(TOK)

app = FastAPI(title="WebJev decisions API")
client = None
stats = dict(requests=0, rows=0, decisions=0)

API_KEY = os.environ.get("WEBJEV_API_KEY", "")
if not API_KEY and os.environ.get("WEBJEV_API_KEY_FILE"):
    with open(os.environ["WEBJEV_API_KEY_FILE"]) as f:
        API_KEY = f.read().strip()
if API_KEY:
    @app.middleware("http")
    async def require_api_key(request: Request, call_next):
        if request.url.path != "/health" and not hmac.compare_digest(
                request.headers.get("authorization", ""), "Bearer " + API_KEY):
            return JSONResponse({"error": {"message": "Invalid or missing API key", "code": 401}}, status_code=401)
        return await call_next(request)


@dataclass
class Q:                                   # the fields prompt.build reads
    text: str; options: list; gold: int = 0


@dataclass
class Example:
    context: str; qs: list; task: str = "infer"; image: bytes = None


class _KeepOrder:                          # keep the caller's option order (training shuffles, serving must not)
    def shuffle(self, x): pass
    def sample(self, xs, k): return xs[:k]


class DecisionRequest(BaseModel):
    state: object
    questions: dict
    model: str | None = None               # accepted for compatibility; this server has one model
    independent: bool = True
    layout: str | None = None              # accepted for compatibility; the trained layout is used


def _prepare(state, questions, independent):
    """State + typed questions -> scoring rows: one per question, one per level for isolated score questions."""
    ctx = S1.render_state(state)
    rqs = {k: S1.render_question(v) for k, v in questions.items()}
    flat, index = S1.plan_rows(rqs, ISOLATED and independent)
    rows = [[r] for r in flat] if independent else [flat]
    items = [build(Example(ctx, [Q(r["question"], list(r["options"]), 0) for r in row]), TOK, _KeepOrder(),
                   max_options=MAX_OPTIONS, max_ctx_tokens=MAX_STATE_TOKENS) for row in rows]
    return (rqs, index), items


@app.on_event("startup")
async def _start():
    global client
    client = httpx.AsyncClient(base_url=VLLM_URL, timeout=httpx.Timeout(900.0),
                               limits=httpx.Limits(max_connections=512, max_keepalive_connections=512))


async def _score_row(item):
    """One scoring row -> probabilities over its options."""
    ids, slots, n = item["ids"], item["slots"], item["nopts"][0]
    if len(slots) != 1 or slots[0] != len(ids) - 1:
        raise ValueError("vLLM reads the last position only; this row's answer slot is elsewhere")
    body = {"model": VLLM_SERVED_MODEL, "prompt": ids, "max_tokens": 1, "temperature": 1.0, "logprobs": n,
            "allowed_token_ids": LABEL_IDS[:n], "return_tokens_as_token_ids": True}
    for attempt in range(3):               # a pooled keep-alive connection can be closed under us: resend
        try:
            r = await client.post("/v1/completions", json=body)
            break
        except httpx.TransportError:
            if attempt == 2:
                raise
            stats["transport_retries"] = stats.get("transport_retries", 0) + 1
    if r.status_code != 200:
        raise RuntimeError(f"vLLM HTTP {r.status_code}: {r.text[:300]}")
    out = r.json()
    if out["usage"]["prompt_tokens"] != len(ids):
        raise RuntimeError(f"vLLM saw {out['usage']['prompt_tokens']} prompt tokens, sent {len(ids)}")
    # With --logprobs-mode processed_logits these values are logits, not log-probabilities.
    top = out["choices"][0]["logprobs"]["top_logprobs"][0]
    by_id = {int(k.split(":", 1)[1]): float(v) for k, v in top.items()}
    missing = [j for j in range(n) if LABEL_IDS[j] not in by_id]
    if missing:
        raise RuntimeError(f"vLLM returned no logit for options {missing[:5]}")
    logits = [by_id[LABEL_IDS[j]] for j in range(n)]
    m = max(logits)
    z = [math.exp((x - m) / TEMP) for x in logits]
    s = math.fsum(z)
    return [x / s for x in z]


def _log_timing(rec):
    if TIMING_LOG:
        with open(TIMING_LOG, "a") as f:
            f.write(json.dumps(rec) + "\n")


async def _model_answers(r: DecisionRequest):
    if not r.independent:
        raise HTTPException(422, "independent=false packs several answer slots into one prompt; vLLM reads only the last position")
    loop = asyncio.get_running_loop()
    t0 = time.perf_counter()
    try:
        (rqs, index), items = await loop.run_in_executor(None, _prepare, r.state, r.questions, r.independent)
    except ValueError as e:
        raise HTTPException(422, str(e))
    t1 = time.perf_counter()
    probs = await asyncio.gather(*[_score_row(it) for it in items])
    t2 = time.perf_counter()
    stats["requests"] += 1
    stats["rows"] += len(items)
    stats["decisions"] += len(rqs)
    _log_timing({"rows": len(items), "tokens": [len(it["ids"]) for it in items], "prepare_ms": round((t1 - t0) * 1000, 2),
                 "vllm_ms": round((t2 - t1) * 1000, 2), "t": time.time()})
    return {"model": MODEL_NAME, "answers": S1.assemble(rqs, index, probs),
            "usage": {"input_tokens": S1.unique_tokens(items), "output_tokens": 0}}


def _options(question):
    criteria = question.get("criteria", question.get("options"))
    return [str(c) for c in criteria] if isinstance(criteria, (list, tuple)) else list(criteria or {})


async def answer(r: DecisionRequest):
    """Both routes. A choice with exactly one option is answered by it at probability 1, as Jev does."""
    single = {k: _options(q)[0] for k, q in r.questions.items()
              if q.get("type", "choice") == "choice" and len(_options(q)) == 1}
    rest = {k: q for k, q in r.questions.items() if k not in single}
    if rest:
        out = await _model_answers(r.model_copy(update={"questions": rest}))
    else:
        out = {"model": MODEL_NAME, "answers": {}, "usage": {"input_tokens": 0, "output_tokens": 0}}
    answers = dict(out["answers"])
    for k, name in single.items():
        answers[k] = {"type": "choice", "choice": name, "confidence": 1.0, "certainty": 1.0, "probabilities": {name: 1.0}}
    return {**out, "answers": {k: answers[k] for k in r.questions}}


@app.post("/v1/systemone")
async def systemone(r: DecisionRequest):
    return await answer(r)


@app.post("/api/alpha/decisions")
async def decisions(r: DecisionRequest):
    return await answer(r)


@app.get("/v1/models")
async def models():
    return {"models": [{"name": MODEL_NAME, "description": "WebJev on vLLM: one-pass typed decisions"}]}


@app.get("/health")
async def health():
    try:
        ok = (await client.get("/health")).status_code == 200
    except httpx.HTTPError:
        ok = False
    return {"ok": ok, "model": MODEL_NAME, "engine": "vllm"}


@app.get("/stats")
async def get_stats():
    return stats
