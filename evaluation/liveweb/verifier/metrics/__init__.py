"""Metrics -- the checking layer.

Convention:

    def <metric>(result, expected, **options) -> float   # 0.0 .. 1.0

* `result` is the evidence a getter captured; `expected` is a constant from the task, or the value another getter
  returned.
* Metrics return a float, not a bool, so partial credit can be expressed.
* Metrics know nothing about any site. All site knowledge lives in the task JSON.
* Standard library only.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable

from .answer import answer_match
from .rules import MISSING, RuleError, match_value, norm_text, resolve_path, to_number
from .structured import count_in_range, json_object_match, numeric_close, set_match
from .text import exact_match, include_exclude, literal_match
from .url import normalize_url, tabs_match, url_matches, url_pattern_match

__all__ = [
    "METRICS", "get_metric", "allowed_options", "run_metric", "combine",
    "MISSING", "RuleError", "match_value", "norm_text", "resolve_path", "to_number",
    "normalize_url",
]

#: Metric registry: `"metric": "<key>"` in a task looks up this table. A misspelled name must fail loudly instead of
#: silently scoring 0 on one task.
METRICS: dict[str, Callable[..., float]] = {
    # URL (most stable, preferred)
    "url_matches": url_matches,
    "url_pattern_match": url_pattern_match,
    "tabs_match": tabs_match,
    # text
    "include_exclude": include_exclude,
    "exact_match": exact_match,
    "literal_match": literal_match,
    # final answer (content, not wrapping)
    "answer_match": answer_match,
    # structured
    "json_object_match": json_object_match,
    "set_match": set_match,
    "count_in_range": count_in_range,
    "numeric_close": numeric_close,
}


def get_metric(name: str) -> Callable[..., float]:
    """Look up a metric by name; an unknown name raises RuleError."""
    try:
        return METRICS[name]
    except KeyError:
        pass
    raise RuleError(f"unknown metric: {name!r}; registered: {sorted(METRICS)}")


def allowed_options(name: str) -> set[str] | None:
    """Keyword options a metric accepts after (result, expected); None when it takes **kwargs (no restriction)."""
    params = list(inspect.signature(get_metric(name)).parameters.values())[2:]
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params):
        return None
    return {p.name for p in params
            if p.kind in (inspect.Parameter.KEYWORD_ONLY,
                          inspect.Parameter.POSITIONAL_OR_KEYWORD)}


def run_metric(name: str, result: Any, expected: Any,
               options: dict | None = None) -> float:
    """Run one metric and clamp its return value to [0, 1].

    Options are checked against the metric's signature first: a misspelled key or options passed to a metric that
    takes none raise RuleError (a task bug, filed as judge_error) instead of a TypeError that would look like broken
    evidence capture.
    """
    fn = get_metric(name)
    if options is not None and not isinstance(options, dict):
        raise RuleError(f"the options of metric {name!r} must be an object, got {type(options).__name__}")
    allowed = allowed_options(name)
    if allowed is not None:
        unknown = sorted(set(options or {}) - allowed)
        if unknown:
            raise RuleError(f"metric {name!r} does not accept options {unknown}; "
                            f"it accepts {sorted(allowed) or '(none)'}")
    score = fn(result, expected, **(options or {}))
    if not isinstance(score, (int, float)) or isinstance(score, bool):
        raise RuleError(f"metric {name!r} returned {type(score).__name__}, expected a float")
    return max(0.0, min(1.0, float(score)))


def combine(scores: list[float], conj: str = "and",
            weights: list[float] | None = None) -> float:
    """Combine the scores of several checks.

    and       any check at 0 -> 0; otherwise the mean
    or        any check at 1 -> 1; otherwise the maximum (for tasks with two acceptable readings)
    weighted  sum(w * s) / sum(w)
    """
    if not scores:
        raise RuleError("combine got an empty list of scores")
    if conj == "and":
        return 0.0 if any(s == 0 for s in scores) else sum(scores) / len(scores)
    if conj == "or":
        return 1.0 if any(s == 1 for s in scores) else max(scores)
    if conj == "weighted":
        ws = weights if weights is not None else [1.0] * len(scores)
        if len(ws) != len(scores):
            raise RuleError(f"weighted: {len(ws)} weights for {len(scores)} checks")
        total = sum(ws)
        if total <= 0:
            raise RuleError("weighted: the sum of weights must be > 0")
        return sum(w * s for w, s in zip(ws, scores)) / total
    raise RuleError(f"unknown conj: {conj!r} (supported: and/or/weighted)")
