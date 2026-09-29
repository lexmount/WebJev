"""Evidence capture and scoring.

    episode ends --> collect_evidence() --> evidence.json --> session deleted
                     (session still alive)     (on disk)         (page gone)
                                                  |
                            judge  ---------------+--> score_evidence() --> judge.json

The cut is a hard constraint: once the session is deleted, URL / cookies / DOM / localStorage are gone. It also gives
a useful property for free: scoring reads only evidence.json, so a check can be changed and re-scored without
re-running the episode.

What is stored:
  * getters that need a live session (URL / DOM / storage / API) run at capture time and are stored in evidence;
  * getters in POSTHOC_SAFE (const / agent_answer) are computed at scoring time from files and not stored.

Rules:
  * capture happens before the session is deleted;
  * capture never fails the episode -- every error is written into evidence.json instead of being raised;
  * capture is harness-only; the agent never sees it;
  * capture uses its own CDP connection (see cdp.py).
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from .cdp import CdpEvalContext, CdpTransport
from .getters import (GETTERS, POSTHOC_SAFE, GetterError, from_jsonable,
                      resolve, to_jsonable)
from .metrics import combine, run_metric
from .metrics.rules import RuleError

EVIDENCE_FILE = "evidence.json"
SPEC_VERSION = "1"

#: The health gate. Deliberately NOT overridable per task: if two tasks disagreed on "is this a wall", the infra
#: bucket -- the one excluded from the denominator -- would depend on the task.
DEFAULT_HEALTH_GATE: dict = {"type": "page_health"}


# -- evidence keys -----------------------------------------------------------------------------------------------

def evidence_key(config: dict) -> str:
    """A stable key for one getter configuration: a hash of its canonical JSON, not the check name, so the same
    configuration in two checks is captured once and renaming a check does not orphan stored evidence."""
    blob = json.dumps(config, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def needs_session(config: Any) -> bool:
    """Does this configuration need a live session? Non-getter values (constants, lists, dicts without type) do
    not."""
    if not (isinstance(config, dict) and isinstance(config.get("type"), str)):
        return False
    return config["type"] in GETTERS and config["type"] not in POSTHOC_SAFE


def session_configs(spec: dict) -> dict[str, dict]:
    """Every getter configuration of a task that needs a session -> {key: config} (from each check's `result` and
    `expected`)."""
    out: dict[str, dict] = {}
    for check in (spec.get("checks") or []):
        for slot in ("result", "expected"):
            config = check.get(slot)
            if needs_session(config):
                out[evidence_key(config)] = config
    return out


# -- post-hoc context --------------------------------------------------------------------------------------------

class ReplayContext:
    """Scoring-time context: session evidence comes from evidence.json, file-based values are computed now.

    `js` / `cdp` are None on purpose, so a session getter used at scoring time fails with a clear GetterError.
    """

    js = None
    cdp = None
    new_session = None

    def __init__(self, *, evidence: dict, record: dict, task: dict,
                 task_dir: Path):
        self.evidence = evidence or {}
        self.record = record or {}
        self.task = task or {}
        self.task_dir = task_dir
        self.network = (self.evidence.get("network") or [])

    def stored(self, config: dict) -> Any:
        """Read one captured value from evidence.json."""
        entry = (self.evidence.get("getters") or {}).get(evidence_key(config))
        if entry is None:
            # If capture itself failed, say so first: that needs a retry, while a check changed after the episode
            # needs a re-run of the episode.
            collect_error = self.evidence.get("error")
            if collect_error:
                raise GetterError(
                    f"no {config.get('type')!r} evidence was captured -- capture itself failed: {collect_error}")
            raise GetterError(
                f"evidence.json has no {config.get('type')!r} evidence -- capture never reached this check "
                f"(was the check changed after the episode? then the episode must be re-run)")
        if not entry.get("ok"):
            raise GetterError(str(entry.get("error") or "capture failed"))
        return from_jsonable(entry.get("value"))


def resolve_for_scoring(ctx: ReplayContext, config: Any) -> Any:
    """Resolve one result/expected at scoring time: session getters from evidence, the rest computed now."""
    if needs_session(config):
        return ctx.stored(config)
    return resolve(ctx, config)


# -- capture -----------------------------------------------------------------------------------------------------

def open_eval_context(cdp_url: str, *, task: dict, task_dir: Path,
                      record: dict | None = None,
                      agent_target_id: str | None = None,
                      browser_context_id: str | None = None) -> CdpEvalContext:
    """Open an evidence connection to a CDP websocket URL."""
    return CdpEvalContext(CdpTransport(cdp_url), task=task, task_dir=task_dir,
                          record=record or {}, agent_target_id=agent_target_id,
                          browser_context_id=browser_context_id)


def collect_evidence(cdp_url: Any, task: dict, task_dir: Path, *,
                     record: dict | None = None,
                     health_gate: dict | None = None,
                     agent_target_id: str | None = None,
                     browser_context_id: str | None = None,
                     context_factory: Any = None) -> dict:
    """Capture evidence after the episode and BEFORE the session is deleted; write `<task_dir>/evidence.json`.

    `cdp_url` is the browser's CDP websocket URL, or a callable returning it (so a failure to obtain the URL is also
    recorded instead of raised). `agent_target_id` is the agent's own idea of its tab, audit only.
    `browser_context_id` scopes the evidence to the run's browser context (runs sharing one local Chrome).

    Never raises. Failures are written into the file and become `judge_error` at scoring time.
    `context_factory` exists for tests only.

    Returns the dict it wrote.
    """
    spec = (task or {}).get("evaluator")
    evidence: dict = {
        "spec_version": SPEC_VERSION,
        "collected_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "phase": "in_session",
        "getters": {},
        "health": None,
        "target": None,
        "error": None,
    }
    ctx = None
    try:
        url = cdp_url() if callable(cdp_url) else cdp_url
        factory = context_factory or open_eval_context
        ctx = factory(url, task=task, task_dir=task_dir, record=record or {},
                      agent_target_id=agent_target_id, browser_context_id=browser_context_id)
        # The health gate first: it never raises, and it usually explains every later capture failure (on a wall
        # page all selectors fail).
        gate = health_gate or DEFAULT_HEALTH_GATE
        try:
            evidence["health"] = to_jsonable(resolve(ctx, gate))
        except Exception as exc:                               # noqa: BLE001
            evidence["health"] = {"ok": False,
                                  "error": f"{type(exc).__name__}: {exc}"}

        if not isinstance(spec, dict):
            evidence["error"] = "the task has no evaluator; only the health gate was captured"
        else:
            for key, config in session_configs(spec).items():
                entry: dict = {"config": config}
                try:
                    entry |= {"ok": True, "value": to_jsonable(resolve(ctx, config))}
                except GetterError as exc:
                    entry |= {"ok": False, "error": f"GetterError: {exc}"}
                except Exception as exc:                       # noqa: BLE001
                    entry |= {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                evidence["getters"][key] = entry
    except Exception as exc:                                   # noqa: BLE001
        evidence["error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
    finally:
        if ctx is not None:
            # Which page was read is stored: it is the first thing to look at when a task scores 0.
            evidence["target"] = getattr(ctx, "target_audit", None) or None
            try:
                ctx.close()
            except Exception:                                  # noqa: BLE001,S110
                pass

    try:
        (task_dir / EVIDENCE_FILE).write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
    except OSError as exc:
        evidence["error"] = f"writing evidence failed: {exc}"
    return evidence


def apply_evidence(record: dict, evidence: Any) -> dict:
    """Write an evidence SUMMARY into the episode record, and move the episode to the infra bucket when the health
    gate found a wall.

    The wall is only discovered at capture time (the agent itself finished normally). Moving the bucket here is what
    makes such an episode retried and not counted as done; scoring decides infra independently as well.
    """
    if not isinstance(evidence, dict):
        return record
    getters = evidence.get("getters") or {}
    health = evidence.get("health")
    blocked = bool(health.get("blocked")) if isinstance(health, dict) else None
    record["evidence"] = {
        "n": len(getters),
        "n_failed": sum(1 for g in getters.values() if not g.get("ok")),
        "health_blocked": blocked,
        "target_rule": (evidence.get("target") or {}).get("rule"),
        "error": evidence.get("error"),
    }
    if blocked and record.get("bucket") != "infra":
        signals = health.get("signals") if isinstance(health, dict) else None
        record["bucket_before_health"] = record.get("bucket")
        record.update(bucket="infra",
                      bucket_reason=f"health_gate: {signals}",
                      infra_error=f"page blocked at evidence time: {signals}")
    return record


# -- scoring -----------------------------------------------------------------------------------------------------

def _summarize(value: Any, limit: int = 300) -> Any:
    """A short, readable copy of the evidence for the verdict file."""
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "..."
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    blob = json.dumps(to_jsonable(value), ensure_ascii=False)[:limit]
    return blob


# `requires` -- scoping the canary.
#
# A `must_exist` canary means "the selector found nothing = capture is broken = judge_error". That is right when the
# agent stopped on the right page. But on the live web the agent decides where it stops: if it wandered off, the page
# naturally lacks the results-list selector -- an agent failure, not a rotten selector. Without this distinction most
# failures would become judge_error and leave the denominator, inflating the success rate.
#
# So a page-specific check can declare `"requires": ["on_search_page"]`: when a prerequisite (an EARLIER check) did
# not score 1.0, this check's GetterError is downgraded to a score of 0 (agent failure). If the prerequisite passed
# and capture still fails, that is real rot and stays judge_error. Only GetterError is downgraded; a misspelled metric
# is a task bug and always judge_error.


def _unmet(requires, seen: dict[str, float], name: str) -> list[str]:
    """Prerequisites that did not score 1.0. An unknown prerequisite name is a task bug (RuleError)."""
    if requires is None:
        return []
    names = [requires] if isinstance(requires, str) else list(requires)
    unknown = [r for r in names if r not in seen]
    if unknown:
        raise RuleError(
            f"{name}: requires {unknown}, which are not names of EARLIER checks "
            f"(scored so far: {sorted(seen)})")
    return [r for r in names if seen[r] < 1.0]


def _unmet_quiet(check: dict, seen: dict[str, float]) -> list[str]:
    """Like `_unmet`, for the except branch: unknown names count as no gate (the error stays judge_error)."""
    requires = check.get("requires")
    if requires is None:
        return []
    names = [requires] if isinstance(requires, str) else list(requires)
    return [r for r in names if seen.get(r, 1.0) < 1.0]


def _broken_prereqs(requires, broken_names: set[str]) -> list[str]:
    """Prerequisites that are themselves judge_error (broken capture / task bug)."""
    if requires is None or not broken_names:
        return []
    names = [requires] if isinstance(requires, str) else list(requires)
    return [r for r in names if r in broken_names]


def score_evidence(spec: dict, *, evidence: dict, record: dict, task: dict,
                   task_dir: Path) -> dict:
    """Score one episode from its evidence; returns the content of judge.json.

    Buckets:
      * health gate hit             -> `infra` (excluded from the denominator, retried)
      * any getter capture failure  -> `judge_error` (excluded from the denominator, not retried)
      * everything else (including 0) -> the episode's own bucket stands
    """
    ctx = ReplayContext(evidence=evidence, record=record, task=task,
                        task_dir=task_dir)
    checks_out: list[dict] = []
    scores: list[float] = []
    weights: list[float] = []
    broken: list[str] = []
    broken_names: set[str] = set()   # checks that could not be scored (for `requires`)
    seen: dict[str, float] = {}      # checks scored so far -> score (for `requires`)

    for check in (spec.get("checks") or []):
        name = check.get("name") or check.get("metric") or "?"
        row: dict = {"name": name, "metric": check.get("metric")}
        blocked_on = _broken_prereqs(check.get("requires"), broken_names)
        if blocked_on:
            msg = f"prerequisite checks {blocked_on} could not be scored (see their errors); this check cannot be scored"
            row |= {"score": None, "error": msg, "blocked_on": blocked_on}
            broken.append(f"{name}: {msg}")
            broken_names.add(name)
            checks_out.append(row)
            continue
        try:
            gated_by = _unmet(check.get("requires"), seen, name)
            result = resolve_for_scoring(ctx, check.get("result"))
            expected = resolve_for_scoring(ctx, check.get("expected"))
            score = run_metric(check["metric"], result, expected,
                               options=check.get("options"))
        except Exception as exc:                               # noqa: BLE001
            if isinstance(exc, GetterError) and _unmet_quiet(check, seen):
                # The prerequisite failed: the agent never reached this page, so a missing selector is expected.
                # Score 0 (agent failure).
                row |= {"score": 0.0, "gated_by": _unmet_quiet(check, seen),
                        "note": "prerequisite check failed; a missing canary counts as an agent failure",
                        "error": f"{type(exc).__name__}: {exc}"}
                checks_out.append(row)
                scores.append(0.0)
                weights.append(float(check.get("weight", 1)))
                seen[name] = 0.0
                continue
            # Broken capture or a task bug -- neither is the agent's failure.
            row |= {"score": None, "error": f"{type(exc).__name__}: {exc}"}
            broken.append(f"{name}: {type(exc).__name__}: {exc}")
            broken_names.add(name)
            checks_out.append(row)
            continue
        row |= {"score": score, "result": _summarize(result),
                "expected": _summarize(expected)}
        if gated_by:
            row["gated_by"] = gated_by
        checks_out.append(row)
        scores.append(score)
        weights.append(float(check.get("weight", 1)))
        seen[name] = score

    health = evidence.get("health")
    health_blocked = bool(isinstance(health, dict) and health.get("blocked"))

    verdict: dict = {
        "spec_version": SPEC_VERSION,
        "evaluator": "script",
        "judge_model": "script",       # explicitly not an LLM judge
        "checks": checks_out,
        "health": health,
        "evidence_ref": EVIDENCE_FILE,
    }

    if health_blocked:
        # Nothing on a wall page means anything: every selector fails, the score would come from the environment.
        signals = health.get("signals") if isinstance(health, dict) else None
        return verdict | {"reward": None, "success": False, "bucket": "infra",
                          "judge_status": "error",
                          "error": f"health gate hit: {signals}"}
    if broken:
        return verdict | {"reward": None, "success": False,
                          "bucket": "judge_error", "judge_status": "error",
                          "error": "; ".join(broken)[:600]}
    if not scores:
        return verdict | {"reward": None, "success": False,
                          "bucket": "judge_error", "judge_status": "error",
                          "error": "the task has no checks"}

    reward = combine(scores, conj=spec.get("conj", "and"), weights=weights)
    return verdict | {"reward": reward, "success": reward >= 1.0,
                      "judge_status": "ok"}
