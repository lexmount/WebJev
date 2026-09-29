"""How an episode ended -> bucket, and the infra retry loop.

Buckets (a closed vocabulary; it defines what is counted):
  agent_run    the episode ran to its end (right or wrong)
  infra        environment / cloud browser / gateway failure; excluded from the denominator and retried
  no_answer    ran, but produced no final answer
  timeout      the step or time budget ran out
  judge_error  written by the verifier only, never by `classify_detail`
"""

from __future__ import annotations

import random
import time
import traceback as _tb

BUCKETS = ("agent_run", "infra", "no_answer", "timeout", "judge_error")

# Failure signatures of the cloud browser / gateways. A hit means infra, not the agent. Matched against
# f"{class name} {message}".lower().
_INFRA_MARKERS = (
    "navigation timed out",
    "err_blocked_by_administrator",
    "websocket",
    "connection reset",
    "connection refused",
    "remote end closed",
    "502 bad gateway",
    "503 service unavailable",
    "cdp",
    # Transport-level timeouts: "the link died", not "our step or episode budget ran out".
    "navigatetimeout",
    "readtimeout",
    "connecttimeout",
    "apitimeout",
    "timeoutexpired",      # subprocess.TimeoutExpired
    "websockettimeout",
)

_NO_ANSWER_REASONS = {"truncated", "no_tool_output", "no_done_call"}


def _exc_text(exc: BaseException) -> str:
    try:
        return str(exc)
    except Exception:  # noqa: BLE001 - the classifier runs on error paths and must never raise itself
        return ""


def classify_detail(exc: BaseException | None = None, *,
                    finish_reason: str | None = None,
                    done_text: str | None = None) -> tuple[str, str]:
    """(bucket, reason) -- the reason says how the bucket was decided, for auditing."""
    if exc is not None:
        # An episode that already produced an answer is never downgraded by a later exception.
        if (done_text or "").strip():
            return "agent_run", "answered_despite_exception"

        name = type(exc).__name__.lower()
        blob = f"{name} {_exc_text(exc)}".lower()
        # Infra signatures must be tested before any timeout heuristic.
        if any(m in blob for m in _INFRA_MARKERS):
            return "infra", "infra_marker"
        if "infraerror" in name:
            return "infra", "infra_error_type"
        # A timeout that reaches this point is our own budget guard: a legitimate agent outcome.
        if isinstance(exc, TimeoutError) or "timeout" in name:
            return "timeout", "episode_timeout"
        # Unknown exceptions count as infra (better a smaller denominator than blaming the agent for the
        # environment), marked separately so they can be audited.
        return "infra", "unclassified_exception"

    if not (done_text or "").strip():
        return "no_answer", "empty_done_text"
    if (finish_reason or "") in _NO_ANSWER_REASONS:
        return "no_answer", "finish_reason"
    return "agent_run", "clean"


#: Stop messages of the agent runtime -> (finish_reason, forced bucket or None). The same exception types report very
#: different things (budget used up, page navigating, decision endpoint down), so the message decides whether it is
#: a legitimate agent outcome or an infrastructure failure.
STOP_REASONS: tuple[tuple[str, str, str | None], ...] = (
    # budget: a legitimate agent outcome
    ("-action demo budget", "max_steps", None),
    ("Reached the demo's model-call budget", "model_call_budget", None),
    # the page never settled: the cloud browser or the site itself, not the agent
    ("Page did not settle", "page_unsettled", "infra"),
    ("Document is navigating", "page_unsettled", "infra"),
    ("Document changed during evaluation", "page_churn", None),
    ("Invalid observed node", "invalid_observed_node", None),
    ("Dropdown execution", "select_unconfirmed", None),
    ("Target changed or is covered", "target_moved", None),
    # the decision endpoint: our side failed, the agent never got to act
    ("Invalid TypeSafe response", "decision_model_invalid_response", "infra"),
    ("Model connection failed", "decision_model_unreachable", "infra"),
    ("Model provider returned HTTP", "decision_model_http_error", "infra"),
    ("Model unavailable", "decision_model_unavailable", "infra"),
    ("needs TEXT_MODEL_API_KEY", "text_model_unconfigured", "infra"),
    # the text helper returned an invalid value: counted against the agent
    ("Text helper returned no valid field value", "text_helper_invalid", None),
)


def classify_stop(message: str) -> tuple[str, str | None] | None:
    """A recognized stop message -> (finish_reason, forced bucket); None when unrecognized."""
    for fragment, reason, bucket in STOP_REASONS:
        if fragment in message:
            return reason, bucket
    return None


# -- infra retries -----------------------------------------------------------------------------------------------

_CAUSE_FIELDS = ("bucket", "bucket_reason", "error", "infra_error", "teardown_error", "finish_reason")


def _cause(record: dict) -> dict:
    return {k: record[k] for k in _CAUSE_FIELDS if record.get(k) is not None}


def _attempt(fn) -> dict:
    """Run fn() once; always return a dict with a bucket."""
    try:
        record = fn()
    except Exception as exc:  # noqa: BLE001 - KeyboardInterrupt still propagates
        bucket, reason = classify_detail(exc)
        return {"bucket": bucket, "bucket_reason": reason,
                "error": f"{type(exc).__name__}: {_exc_text(exc)[:400]}",
                "traceback": _tb.format_exc()[-1500:]}
    if not isinstance(record, dict):
        return {"bucket": "infra", "bucket_reason": "non_dict_result",
                "error": f"episode returned {type(record).__name__}, expected dict"}
    return record


def run_with_retry(fn, *, retries: int = 2, backoff_s: float = 5.0) -> tuple[dict, int]:
    """Run fn() and retry ONLY the infra bucket. Returns (record, retries used).

    agent_run / no_answer / timeout are legitimate outcomes; retrying them would hand out extra chances and inflate
    the success rate.
    """
    record = _attempt(fn)
    attempts, prior = 0, []
    while record.get("bucket") == "infra" and attempts < retries:
        attempts += 1
        prior.append(_cause(record))
        if backoff_s > 0:
            time.sleep(backoff_s * random.uniform(0.5, 1.5))   # jitter: workers must not retry in lockstep
        record = _attempt(fn)
    if prior:
        record["prior_attempts"] = prior
    return record, attempts
