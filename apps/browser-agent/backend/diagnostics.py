"""Owner-only model inputs, outputs and timings, with credentials redacted."""
from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path


def clean(value):
    if isinstance(value, dict):
        return {str(k): ("[REDACTED]" if re.search(r"api.?key|authorization|cookie|secret|password|access.?token|connect_url|cdp_url", str(k), re.I) else clean(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, str):
        for key, secret in os.environ.items():
            if re.search(r"KEY|TOKEN|SECRET|PASSWORD|CDP_URL", key, re.I) and len(secret) >= 8:
                value = value.replace(secret, "[REDACTED]")
        value = re.sub(r"(?i)\bBearer\s+[\w.\-+/=]+", "Bearer [REDACTED]", value)
        value = re.sub(r"(?i)([?&](?:token|key|api_key|access_token|secret|signature)=)[^\s&#\"']+", r"\1[REDACTED]", value)
        return value
    return value


def generation_stage(body):
    for message in reversed(body.get("messages", [])):
        if message.get("role") != "user":
            continue
        try:
            payload = json.loads(message["content"])
        except (ValueError, TypeError, KeyError):
            continue
        if not isinstance(payload, dict):
            continue
        if "field" in payload:
            return "field_text"
        if "completion_check" in payload:
            return "completion_check" if payload["completion_check"] else "guidance"
        if "observations" in payload:
            return "result_summary" if "plan" not in payload else "result_extraction"
        if "website" in payload or ("query" in payload and "runtime" in payload):
            return "semantic_plan"
    return "generation"


class ModelDiagnostics:
    def __init__(self, writer):
        self.writer = writer
        self.path = writer.folder / "model_calls.jsonl"
        self.lock = threading.Lock()
        self.sequence = 0

    def save(self, entry):
        # Diagnostics must never alter the model result or execute another call.
        try:
            with self.lock:
                fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
                with os.fdopen(fd, "a", encoding="utf-8") as stream:
                    stream.write(json.dumps(clean(entry), ensure_ascii=True) + "\n")
        except OSError:
            pass

    def call(self, kind, body, invoke):
        with self.lock:
            self.sequence += 1
            identifier = f"call-{self.sequence}"
        started = time.perf_counter()
        entry = {"id": identifier, "kind": kind,
                 "stage": "decision" if kind == "jev" else generation_stage(body),
                 "requested_model": body.get("model"), "response_model": None,
                 "started_elapsed_ms": self.writer.elapsed(), "duration_ms": None,
                 "status": "running", "input": body, "output": None, "source": "live"}
        self.save(entry)
        try:
            response = invoke()
            entry.update(status="succeeded", response_model=response.get("model"), usage=response.get("usage", {}))
            if kind == "jev":
                entry["output"] = response.get("answers", {})
                answers = response.get("answers", {})
                operation = answers.get("operation", {}).get("choice", "")
                target = answers.get(operation.lower() + "_target", {}).get("choice")
                candidate = body.get("questions", {}).get(operation.lower() + "_target", {}).get("criteria", {}).get(str(target), {})
                entry["action"] = operation
                entry["target"] = candidate.get("element", "") if isinstance(candidate, dict) else str(candidate)
            else:
                choice = (response.get("choices") or [{}])[0]
                entry["output"] = choice.get("message", {}).get("content")
                entry["finish_reason"] = choice.get("finish_reason")
            return response
        except BaseException as exc:
            entry.update(status="failed", error=type(exc).__name__)
            match = re.search(r"HTTP (\d{3})", str(exc))
            if match:
                entry["error"] += f" · HTTP {match[1]}"
            raise
        finally:
            entry["duration_ms"] = round((time.perf_counter() - started) * 1000)
            entry["finished_elapsed_ms"] = self.writer.elapsed()
            self.save(entry)


def _read_json(path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def read_calls(folder: Path, since=""):
    journal = folder / "model_calls.jsonl"
    paths = [journal] if journal.exists() else [folder / name for name in ("raw_state.json", "timings.jsonl", "plan.json", "guidance.json", "result.json")]
    revision = "-".join(f"{p.stat().st_mtime_ns}:{p.stat().st_size}" for p in paths if p.exists()) or "empty"
    if since == revision:
        return {"revision": revision, "calls": None}
    calls = {}
    if journal.exists():
        for line in journal.read_text().splitlines():
            try:
                entry = json.loads(line)
                calls[entry["id"]] = entry
            except (ValueError, KeyError, TypeError):
                continue
        return {"revision": revision, "calls": clean(sorted(calls.values(), key=lambda c: c["started_elapsed_ms"])), "legacy": False}

    # Older runs did not record every response. Display only what was saved.
    state = _read_json(folder / "raw_state.json", {})
    records = []
    for index, decision in enumerate(state.get("decisions", [])):
        duration = decision.get("latency_ms")
        end = decision.get("elapsed_ms")
        records.append({"id": f"legacy-jev-{index}", "kind": "jev", "stage": "decision",
                        "requested_model": decision.get("request", {}).get("model"),
                        "response_model": decision.get("model"), "duration_ms": duration,
                        "started_elapsed_ms": max(0, end - duration) if isinstance(end, (int, float)) and isinstance(duration, (int, float)) else None,
                        "finished_elapsed_ms": end, "status": "succeeded", "source": "saved",
                        "input": decision.get("request"),
                        "output": decision.get("raw_answers"), "action": decision.get("operation"),
                        "usage": decision.get("usage", {})})
    for index, call in enumerate(state.get("text_calls", [])):
        duration, end = call.get("latency_ms"), call.get("elapsed_ms")
        records.append({"id": f"legacy-generation-{index}", "kind": "generation", "stage": "field_text",
                        "requested_model": call.get("model"), "response_model": None,
                        "duration_ms": duration, "finished_elapsed_ms": end,
                        "started_elapsed_ms": max(0, end - duration) if isinstance(end, (int, float)) and isinstance(duration, (int, float)) else None,
                        "status": "succeeded", "source": "saved", "output": call.get("value"),
                        "usage": call.get("usage", {})})
    try:
        timings = [json.loads(line) for line in (folder / "timings.jsonl").read_text().splitlines()]
    except (OSError, ValueError):
        timings = []
    for index, timing in enumerate(timings):
        phase = timing.get("phase")
        if phase not in {"semantic_plan", "guidance", "guidance_completion", "result_summary"}:
            continue
        output = None
        if phase == "semantic_plan":
            output = _read_json(folder / "plan.json", None)
        if phase == "result_summary":
            result = _read_json(folder / "result.json", {})
            output = {k: result[k] for k in ("summary", "columns", "rows") if k in result} or None
        records.append({"id": f"legacy-stage-{index}", "kind": "generation",
                        "stage": "completion_check" if phase == "guidance_completion" else phase,
                        "requested_model": None, "response_model": None,
                        "started_elapsed_ms": timing.get("started_elapsed_ms"), "duration_ms": timing.get("duration_ms"),
                        "status": "succeeded" if timing.get("ok") else "failed", "source": "saved_stage",
                        "output": output})
    return {"revision": revision, "calls": clean(sorted(records, key=lambda c: c.get("started_elapsed_ms") or 0)), "legacy": True}
