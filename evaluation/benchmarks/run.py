#!/usr/bin/env python3
"""Send every benchmark item to a Jev-compatible endpoint and cache the answers.

    # WebJev behind ../serving (no key on localhost)
    python run.py --name webjev --url http://127.0.0.1:8200/v1/systemone
    # Jev through OpenRouter's Decisions API
    python run.py --name jev --url https://openrouter.ai/api/alpha/decisions \
        --model typesafe/jev-1.13-20260917 --key-env OPENROUTER_API_KEY --workers 6
    # Jev through TypeSafe's API
    python run.py --name jev --url https://api.typesafe.ai/v1/systemone --model jev-1.13.0 --key-env TYPESAFE_API_KEY --workers 6

The request body is exactly {"state", "questions"} of the item (plus "model" when --model is given). Answers are cached
per item in $EVAL_ROOT/pred/<name>/<benchmark>.jsonl as {"id", "ms", "model", "answers", "usage"}, or {"id", "error"}.
Rerunning keeps the answered items and only re-sends missing or failed ones. Per-benchmark wall time and median
latency go to $EVAL_ROOT/pred/<name>/_timing.json. Score with score.py.
"""
import argparse
import concurrent.futures
import json
import os
import time
import urllib.error
import urllib.request

from common import BENCHES, ROOT, load_bench

ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
ap.add_argument("--name", required=True, help="prediction directory under $EVAL_ROOT/pred")
ap.add_argument("--url", required=True, help="full endpoint URL (.../v1/systemone or .../api/alpha/decisions)")
ap.add_argument("--model", default="", help='"model" field of the request body (needed by OpenRouter and TypeSafe)')
ap.add_argument("--key-env", default="", help="environment variable holding the bearer key")
ap.add_argument("--workers", type=int, default=1, help="concurrent requests (1 = one item at a time)")
ap.add_argument("--timeout", type=int, default=900)
ap.add_argument("--retries", type=int, default=5, help="retries on transport errors and HTTP 408/429/5xx")
ap.add_argument("--warmup", type=int, default=0, help="untimed requests per benchmark before timing starts")
ap.add_argument("--limit", type=int, default=0, help="only the first N items of each benchmark (smoke test)")
ap.add_argument("benches", nargs="*", help=f"default: all 8 ({', '.join(BENCHES)})")
a = ap.parse_args()
KEY = os.environ[a.key_env] if a.key_env else ""


def post(item):
    body = {"state": item["state"], "questions": item["questions"]}
    if a.model:
        body["model"] = a.model
    data = json.dumps(body, ensure_ascii=False).encode()
    headers = {"Content-Type": "application/json", **({"Authorization": "Bearer " + KEY} if KEY else {})}
    for attempt in range(a.retries + 1):
        try:
            with urllib.request.urlopen(urllib.request.Request(a.url, data=data, headers=headers), timeout=a.timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            text = e.read().decode(errors="ignore")[:300]
            if e.code in (408, 429, 500, 502, 503, 504) and attempt < a.retries:
                time.sleep(2 * (attempt + 1))
                continue
            raise RuntimeError(f"HTTP {e.code}: {text}") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt < a.retries:
                time.sleep(2 * (attempt + 1))
                continue
            raise


def call(item):
    t = time.perf_counter()
    try:
        resp = post(item)
        answers = resp.get("answers") or {}
        missing = [q for q in item["questions"] if q not in answers]
        if missing:
            raise RuntimeError(f"no answer for {missing}")
        return {"id": item["id"], "ms": round((time.perf_counter() - t) * 1000, 1), "model": resp.get("model"),
                "answers": answers, "usage": resp.get("usage")}
    except Exception as e:  # noqa: BLE001 - recorded per item; a failed item counts as wrong until it is re-sent
        return {"id": item["id"], "error": f"{type(e).__name__}: {str(e)[:300]}"}


out_dir = f"{ROOT}/pred/{a.name}"
os.makedirs(out_dir, exist_ok=True)
timing_path = f"{out_dir}/_timing.json"
timing = json.load(open(timing_path)) if os.path.exists(timing_path) else {}
for bench in a.benches or BENCHES:
    items = load_bench(bench)[:a.limit or None]
    path = f"{out_dir}/{bench}.jsonl"
    done = {}
    if os.path.exists(path):
        for line in open(path):
            row = json.loads(line)
            if "error" not in row:
                done[row["id"]] = row
    todo = [it for it in items if it["id"] not in done]
    if a.warmup and todo:
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
            list(pool.map(call, items[-a.warmup:]))
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        new = dict(zip([it["id"] for it in todo], pool.map(call, todo)))
    wall = time.perf_counter() - t0
    rows = [done.get(it["id"]) or new[it["id"]] for it in items]       # benchmark order
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    errors = sum("error" in r for r in rows)
    ms = sorted(r["ms"] for r in new.values() if "ms" in r)
    if todo:
        timing[bench] = {"items": len(items), "sent": len(todo), "workers": a.workers, "wall_s": round(wall, 2),
                         "p50_ms": ms[len(ms) // 2] if ms else None, "errors": errors, "url": a.url}
        json.dump(timing, open(timing_path, "w"), indent=1)
    print(f"{a.name} {bench}: {len(items)} items, sent {len(todo)} in {wall:.1f}s, errors {errors}", flush=True)
print(f"answers in {out_dir}; score with: python score.py {a.name}")
