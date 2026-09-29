#!/usr/bin/env python3
"""One live-web task on the browser-agent runtime with the decision model alone, in its own process.

Calls the agent's own `engine.run_live()` (apps/browser-agent/backend of this repository, unmodified) with four
stubs, so that the decision model is the only thing being measured:
  - `semantic_plan` (the LLM planner): the start URL and the single stage come from the task;
  - `AdaptiveGuide`: no stall hints, no recovery after BLOCKED, no completion review;
  - `login_gate` (the pause for a human login): nobody is there to log in;
  - `LiveFrames` (the UI screencast): streams frames to the web page, never affects a decision.
and two hooks the engine offers for harnesses:
  - `browser_opened`: notes which browser this episode got, so it can be released should the process die;
  - `browser_closing`: called just before the agent closes its browser, with the page exactly as the agent left
    it -- the verifier captures its evidence here over its own CDP connection.
Everything else (decision endpoint, browser backend) comes from the environment the runner sets. The final answer is
the engine's own result summary, the text the web page shows.

    python harness/episode.py --task tasks/<id>.json --task-dir DIR [--max-seconds 900]   ->   DIR/episode.json
"""
import argparse
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIVEWEB = HERE.parent
REPO = LIVEWEB.parents[1]
AGENT_BACKEND = Path(os.environ.get("WEBJEV_AGENT_BACKEND") or REPO / "apps/browser-agent/backend")
sys.path[:0] = [str(AGENT_BACKEND), str(LIVEWEB), str(HERE)]

import engine  # noqa: E402  the agent's engine
from buckets import classify_detail, classify_stop  # noqa: E402
from cleanup import note_browser, release_browser  # noqa: E402
from verifier.evidence import apply_evidence, collect_evidence  # noqa: E402

EVIDENCE = {}
BROWSER = {}  # backend, session_id / browser_context_id of this episode's browser


def plan_from_task(request, model, runtime_context):
    return request["target_website"], [request["query"]]


class NoGuide:
    def __init__(self, *args, **kwargs):
        self.records = []

    def current(self, count):
        return None

    def update(self, *args, **kwargs):
        return False


class NoFrames:
    def __init__(self, *args, **kwargs):
        pass

    def target(self, target):
        pass

    def close(self):
        pass


def install(task, task_dir, record):
    engine.semantic_plan = plan_from_task
    engine.AdaptiveGuide = NoGuide
    engine.login_gate = lambda page: False
    engine.LiveFrames = NoFrames

    def browser_opened(endpoint):
        BROWSER.update(note_browser(task_dir, endpoint))

    def browser_closing(endpoint):
        # The page is still exactly as the agent left it; the engine closes the browser right after this returns.
        EVIDENCE["value"] = collect_evidence(endpoint["cdp_url"], task, task_dir, record=record,
                                             agent_target_id=endpoint.get("target_id"),
                                             browser_context_id=endpoint.get("browser_context_id"))

    engine.browser_opened = browser_opened
    engine.browser_closing = browser_closing
    original_load = engine.load_runtime

    def load_runtime(writer=None):
        adapter, agent_module, model = original_load(writer)
        if os.environ.get("DECISION_TIMEOUT_S"):  # opt-in: a slower endpoint than the runtime's 25 s client
            import httpx
            timeout = float(os.environ["DECISION_TIMEOUT_S"])
            model.CLIENT = httpx.Client(http2=True, timeout=timeout)
            model.BACKUP = httpx.Client(http2=True, timeout=timeout)
        return adapter, agent_module, model

    engine.load_runtime = load_runtime


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def answer_text(result):
    """The engine's result summary, with its table written out, is the final answer."""
    if not isinstance(result, dict) or result.get("status") != "summarized":
        return ""
    lines = [str(result.get("summary") or "").strip()]
    columns = result.get("columns") or []
    for row in result.get("rows") or []:
        cells = row.get("cells") or {}
        lines.append("; ".join(f"{c.get('label')}: {(cells.get(c.get('key')) or {}).get('value', cells.get(c.get('key')))}"
                               for c in columns))
    return "\n".join(line for line in lines if line).strip()


def failed_call_kind(folder):
    """`jev` (decision) or `generation` (text helper): which model call failed last, from the engine's call log."""
    kind = None
    if (folder / "model_calls.jsonl").exists():
        for line in (folder / "model_calls.jsonl").read_text().splitlines():
            call = json.loads(line) if line.strip() else {}
            if call.get("status") == "failed":
                kind = call.get("kind")
    return kind


def stop_reason(state, diagnostics, folder):
    """(finish_reason, bucket or None, message) from how run_live ended."""
    failure = state.get("failure")
    status = state.get("status")
    last = diagnostics[-1] if diagnostics else {}
    message = f"{last.get('exception', '')}: {last.get('message', '')}".strip(": ")
    rejected = re.search(r"Model provider returned HTTP (4\d\d)", message)
    if rejected and rejected.group(1) not in ("408", "429") and failed_call_kind(folder) == "jev":
        # The decision endpoint refused this page's request (e.g. a page with too many elements for its token
        # limit). Deterministic for the page, and the runtime ends the task the same way: an agent failure, not infra.
        return "decision_request_rejected", None, message
    if status == "failed":  # no browser: the browser could not be created or reached before the agent existed
        return "no_browser", "infra", message or "run_live failed before the browser existed"
    if not failure:
        return status, None, ""
    if failure == "execution_budget_reached":
        return "episode_timeout", None, ""
    if failure == "repeated_action_cycle":
        return "repeated_action_cycle", None, ""
    hit = classify_stop(str(last.get("message", "")))
    if hit:
        return hit[0], hit[1], message
    error = type(last.get("exception") or failure, (Exception,), {})(last.get("message", ""))
    bucket, _ = classify_detail(error)
    return failure, "infra" if bucket == "infra" else None, message


def build_record(task, folder, started):
    state = read_json(folder / "raw_state.json") or {}
    result = read_json(folder / "result.json") or {}
    diagnostics = []
    if (folder / "diagnostics.jsonl").exists():
        diagnostics = [json.loads(line) for line in (folder / "diagnostics.jsonl").read_text().splitlines() if line]
    decisions = state.get("decisions") or []
    history = [{k: v for k, v in h.items() if k != "execution"} for h in state.get("history") or []]
    calls = []
    for i, d in enumerate(decisions):
        request = (d.get("request") or {}).get("state") or {}
        calls.append({"i": i, "url": (request.get("page") or {}).get("url"), "n_elements": len(request.get("elements") or []),
                      "operation": d.get("operation"), "operation_confidence": d.get("confidence"),
                      "target": d.get("target"), "target_confidence": d.get("target_confidence"),
                      "choice": d.get("choice"), "model": d.get("model"), "latency_ms": d.get("latency_ms"),
                      "usage": d.get("usage") or {}, "fingerprint": str(d.get("fingerprint") or "")[:16],
                      "elapsed_ms": d.get("elapsed_ms")})
    finish_reason, bucket, message = stop_reason(state, diagnostics, folder)
    terminal = int(bool(decisions) and state.get("status") in ("done", "blocked")
                   and decisions[-1].get("operation") in ("DONE", "BLOCKED"))
    done_text = answer_text(result)
    record = {
        "task_id": task["task_id"], "status": state.get("status"), "failure": state.get("failure"),
        "finish_reason": finish_reason, "env_steps": len(history), "history": history, "decisions": calls,
        "stalled_ticks": max(0, len(decisions) - len(history) - terminal),
        "text_calls": state.get("text_calls") or [], "final_url": (state.get("page") or {}).get("url"),
        "done_text": done_text, "answer_source": "result_summary" if done_text else None,
        "result_status": result.get("status"), "observations": len(state.get("observations") or []),
        "session_timing": state.get("session_timing"), "stop_error": message or None,
        "runtime_sources": state.get("source_hashes"), "seconds": round(time.monotonic() - started, 1),
    }
    if bucket == "infra":
        record["infra_error"] = message or finish_reason
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, type=Path)
    ap.add_argument("--task-dir", required=True, type=Path)
    ap.add_argument("--max-seconds", type=int, default=900)
    a = ap.parse_args()
    task = json.loads(a.task.read_text())
    folder = a.task_dir / "apps"
    folder.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    record = {"task_id": task["task_id"]}
    try:
        install(task, a.task_dir, record)
        writer = engine.EventWriter(folder)
        request = {"id": task["task_id"], "query": task["task_name"], "target_website": task["website"],
                   "max_seconds": a.max_seconds}
        try:
            engine.run_live(request, writer)
        except Exception as exc:  # the runtime could not start the task: infra
            record["infra_error"] = f"run_live before the browser: {type(exc).__name__}: {str(exc)[:300]}"
        finally:
            writer.close()
        if not record.get("infra_error"):
            record.update(build_record(task, folder, started))
        if not record.get("infra_error"):
            record["bucket"], record["bucket_reason"] = classify_detail(finish_reason=record["finish_reason"],
                                                                        done_text=record["done_text"])
        apply_evidence(record, EVIDENCE.get("value"))  # anti-bot wall at evidence time -> infra
    except BaseException as exc:  # our glue failed; the runner files it as infra and retries
        record["infra_error"] = f"episode glue: {type(exc).__name__}: {str(exc)[:300]}"
        record["traceback"] = traceback.format_exc()[-2000:]
    finally:
        record["browser"] = dict(BROWSER, release=release_browser(BROWSER))
    (a.task_dir / "episode.json").write_text(json.dumps(record, ensure_ascii=False, default=str))
    print(json.dumps({k: record.get(k) for k in ("task_id", "bucket", "finish_reason", "env_steps", "infra_error")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
