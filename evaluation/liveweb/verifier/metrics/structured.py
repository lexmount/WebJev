"""Structured checks on dicts, lists and numbers.

`json_object_match` is the workhorse: URL parameters, control states and JSON returned by a site's own API all use it;
only the source of `result` differs.
"""

from __future__ import annotations

import re
from typing import Any

from .rules import MISSING, RuleError, match_value, resolve_path, to_number

__all__ = ["json_object_match", "set_match", "count_in_range", "numeric_close"]


def json_object_match(result: Any, expected: Any, *, partial: bool = False,
                      case_sensitive: bool = False,
                      expect_in_result: bool = True,
                      ignore_literal_whitespace: bool = False) -> float:
    """Check path by path; each value goes through the rule language.

    `expected` is {path: rule}. Paths support `a.b`, `items[0].size`, `items[*].size` and `*`. A multi-valued path
    means "any of them matches"; use {"all": [...]} for "every one of them".

    partial=False (default): 1.0 only if every path matches, else 0.0. partial=True: fraction of paths that match.
    expect_in_result=False flips the direction: every top-level key of `result` must have a rule in `expected` that
    holds ("nothing else may be on the page").
    ignore_literal_whitespace=True removes all whitespace on both sides, only for plain string expectations.
    """
    if not isinstance(ignore_literal_whitespace, bool):
        raise RuleError("ignore_literal_whitespace must be a bool")

    def matches(value: Any, rule: Any) -> bool:
        if ignore_literal_whitespace and isinstance(value, str) and isinstance(rule, str):
            value, rule = re.sub(r"\s+", "", value), re.sub(r"\s+", "", rule)
        return match_value(value, rule, case_sensitive=case_sensitive)

    if not isinstance(expected, dict):
        raise RuleError(f"the expected value of json_object_match must be a dict, "
                        f"got {type(expected).__name__}")
    if not expected:
        raise RuleError("the expected value of json_object_match must not be empty")

    if not expect_in_result:
        if not isinstance(result, dict):
            return 0.0
        checks = [(k, expected.get(k, MISSING)) for k in result]
        hits = [k in expected and matches(result[k], expected[k])
                for k, _ in checks]
    else:
        hits = []
        for path, rule in expected.items():
            found = resolve_path(result, str(path))
            if not found:
                hits.append(matches(MISSING, rule))
            else:
                hits.append(any(matches(v, rule)
                                for v in found))

    if not hits:
        return 0.0
    if partial:
        return sum(hits) / len(hits)
    return 1.0 if all(hits) else 0.0


def set_match(result: Any, expected: Any, *, mode: str = "equal",
              case_sensitive: bool = False) -> float:
    """Set comparison. mode: equal / subset (result is a subset of expected) / superset."""
    from .rules import norm_text
    if result is MISSING or result is None:
        return 0.0
    to_set = (lambda xs: {norm_text(x, case_sensitive=case_sensitive)
                          for x in (xs if isinstance(xs, (list, tuple, set)) else [xs])})
    got, want = to_set(result), to_set(expected)
    if mode == "equal":
        ok = got == want
    elif mode == "subset":
        ok = got <= want
    elif mode == "superset":
        ok = got >= want
    else:
        raise RuleError(f"set_match mode must be equal/subset/superset, got {mode!r}")
    return 1.0 if ok else 0.0


def count_in_range(result: Any, expected: Any) -> float:
    """A count falls in a range.

    result is a number (e.g. from `dom_count`) or something with a length; expected is {"min": n, "max": m} or a
    rule such as {"ge": 1}.
    """
    size = to_number(result)
    if size is None and hasattr(result, "__len__"):
        size = float(len(result))
    if size is None:
        return 0.0
    if isinstance(expected, dict) and ("min" in expected or "max" in expected):
        lo = to_number(expected.get("min", float("-inf")))
        hi = to_number(expected.get("max", float("inf")))
        if lo is None or hi is None:
            raise RuleError(f"count_in_range min/max must be numbers: {expected!r}")
        return 1.0 if lo <= size <= hi else 0.0
    return 1.0 if match_value(size, expected) else 0.0


def numeric_close(result: Any, expected: Any, *, tol: float = 1e-6,
                  relative: bool = False) -> float:
    """Numeric closeness; `expected` may be a scalar returned by another getter."""
    return 1.0 if match_value(result, {"approx": expected, "tol": tol,
                                       "relative": relative}) else 0.0
