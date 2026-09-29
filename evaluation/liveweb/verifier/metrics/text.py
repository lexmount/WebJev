"""Text checks on page text or the final answer.

Keep checks shallow: on live sites deep checks ("is every result really black?") break on ads, recommendations,
A/B tests and lazy loading. Check the words that must (or must not) be there.
"""

from __future__ import annotations

from typing import Any

from .rules import MISSING, RuleError, norm_text

__all__ = ["include_exclude", "exact_match", "literal_match"]


def _text_of(value: Any) -> str:
    if value is MISSING or value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def include_exclude(result: Any, expected: Any, *, case_sensitive: bool = False,
                    partial: bool = False) -> float:
    """Every `must_include` string is present and no `must_exclude` string is.

    expected: {"must_include": [...], "must_exclude": [...]} (both optional); a bare list means must_include.
    """
    if isinstance(expected, (list, tuple)):
        expected = {"must_include": list(expected)}
    if not isinstance(expected, dict):
        raise RuleError("the expected value of include_exclude must be a dict or a list")
    inc = expected.get("must_include", []) or []
    exc = expected.get("must_exclude", []) or []
    if not inc and not exc:
        raise RuleError("include_exclude needs must_include or must_exclude")

    hay = norm_text(_text_of(result), case_sensitive=case_sensitive)
    hits = [norm_text(n, case_sensitive=case_sensitive) in hay for n in inc]
    hits += [norm_text(n, case_sensitive=case_sensitive) not in hay for n in exc]
    if partial:
        return sum(hits) / len(hits)
    return 1.0 if all(hits) else 0.0


def exact_match(result: Any, expected: Any, *, case_sensitive: bool = False,
                collapse_ws: bool = True) -> float:
    """Equal after normalization (collapsed whitespace, case-insensitive by default)."""
    cands = expected if isinstance(expected, (list, tuple)) else [expected]
    got = norm_text(_text_of(result), case_sensitive=case_sensitive,
                    collapse_ws=collapse_ws)
    return 1.0 if any(got == norm_text(_text_of(c), case_sensitive=case_sensitive,
                                       collapse_ws=collapse_ws)
                      for c in cands) else 0.0


def literal_match(result: Any, expected: Any) -> float:
    """Byte-for-byte equality, no normalization. Only for checks that are about the format itself."""
    cands = expected if isinstance(expected, (list, tuple)) else [expected]
    return 1.0 if any(_text_of(result) == _text_of(c) for c in cands) else 0.0
