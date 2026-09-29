"""Rule language, path resolution and value normalization.

The metric layer uses only the standard library (re / urllib / math), so a verdict is bit-for-bit reproducible on any
machine. Only evidence capture (the getters) needs a browser.

Three building blocks:
  match_value(value, rule)   does one value satisfy one rule   <- the smallest unit of a check
  resolve_path(obj, path)    pick values out of nested data    <- "items[*].size"
  norm_text / to_number      normalization                     <- "$1,234.00" -> 1234.0
"""

from __future__ import annotations

import math
import re
from typing import Any

__all__ = [
    "MISSING", "RuleError", "match_value", "resolve_path",
    "norm_text", "to_number", "compile_flags",
]


class _Missing:
    """Sentinel for "the path resolved to nothing". Distinct from None, which can be a real value."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "<MISSING>"

    def __bool__(self) -> bool:
        return False


MISSING = _Missing()


class RuleError(ValueError):
    """The rule itself is malformed (not "the check failed").

    Callers must file it as `judge_error`, never as a 0: a broken check and a wrong agent are different things, and
    mixing them would report a verifier bug as a model regression.
    """


# -- normalization -----------------------------------------------------------------------------------------------

_WS = re.compile(r"\s+")
# Decorations allowed inside a number: currency symbols, thousands separators, percent, whitespace
# (including the no-break space).
_NUM_STRIP = re.compile("[,\\s\u00a0$\u00a5\u20ac\u00a3%]")
_NUM_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


def norm_text(value: Any, *, case_sensitive: bool = False,
              collapse_ws: bool = True) -> str:
    """Normalize text: collapse whitespace, strip, and (by default) casefold.

    Whitespace in page text is unstable (line breaks, no-break spaces, indentation), so it is collapsed by default.
    """
    text = value if isinstance(value, str) else str(value)
    text = text.replace("\u00a0", " ")
    if collapse_ws:
        text = _WS.sub(" ", text)
    text = text.strip()
    return text if case_sensitive else text.casefold()


def to_number(value: Any) -> float | None:
    """Best-effort number parsing; returns None when the value is not a number (never raises).

    `"$1,234.00"` -> 1234.0, `"2014"` -> 2014.0, `"25%"` -> 25.0.
    """
    if isinstance(value, bool):        # bool is a subclass of int but must not compare as a number
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    cleaned = _NUM_STRIP.sub("", value.strip())
    if not _NUM_RE.match(cleaned):
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


_FLAG_MAP = {"i": re.IGNORECASE, "m": re.MULTILINE, "s": re.DOTALL, "x": re.VERBOSE}


def compile_flags(flags: str | int | None) -> int:
    """`"IM"` -> re.IGNORECASE | re.MULTILINE. Empty / None -> 0."""
    if flags is None:
        return 0
    if isinstance(flags, int):
        return flags
    out = 0
    for ch in flags:
        if ch in (" ", "|", "-"):
            continue
        mapped = _FLAG_MAP.get(ch.lower())
        if mapped is None:
            raise RuleError(f"unknown regex flag: {ch!r} (supported: i/m/s/x)")
        out |= mapped
    return out


# -- path resolution ---------------------------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"""
    (?P<key>[^.\[\]]+)          # plain key (including the * wildcard)
  | \[(?P<index>-?\d+)\]        # index [0] [-1]
  | \[(?P<star>\*)\]            # every element [*]
""", re.VERBOSE)


def _tokenize(path: str) -> list[tuple[str, Any]]:
    tokens: list[tuple[str, Any]] = []
    pos = 0
    while pos < len(path):
        if path[pos] == ".":
            pos += 1
            continue
        m = _TOKEN_RE.match(path, pos)
        if not m:
            raise RuleError(f"path syntax error: {path!r} at character {pos}")
        if m.group("key") is not None:
            tokens.append(("key", m.group("key")))
        elif m.group("index") is not None:
            tokens.append(("index", int(m.group("index"))))
        else:
            tokens.append(("star", None))
        pos = m.end()
    return tokens


def resolve_path(obj: Any, path: str) -> list[Any]:
    """Every value a path selects in nested dicts/lists (possibly none).

    Supports `a.b`, `a[0].b`, `items[*].size` and `*` (any key). The result is a list on purpose: `items[*].size` is
    naturally multi-valued, and a check on it means "any of them matches".
    """
    if path in ("", "."):
        return [obj]
    frontier: list[Any] = [obj]
    for kind, arg in _tokenize(path):
        nxt: list[Any] = []
        for node in frontier:
            if kind == "key":
                if arg == "*":
                    if isinstance(node, dict):
                        nxt.extend(node.values())
                    elif isinstance(node, list):
                        nxt.extend(node)
                elif isinstance(node, dict) and arg in node:
                    nxt.append(node[arg])
            elif kind == "index":
                if isinstance(node, list) and -len(node) <= arg < len(node):
                    nxt.append(node[arg])
            else:  # star
                if isinstance(node, list):
                    nxt.extend(node)
                elif isinstance(node, dict):
                    nxt.extend(node.values())
        frontier = nxt
        if not frontier:
            return []
    return frontier


# -- rule language -----------------------------------------------------------------------------------------------

_OPS = {
    "eq", "ne", "lt", "le", "gt", "ge", "re", "not_re", "in", "not_in",
    "contains", "not_contains", "approx", "range", "exists", "len",
    "any", "all", "not",
}
# Not operators: parameters of operators.
_AUX = {"flags", "tol", "relative", "closed", "case_sensitive"}


def _as_text_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [v if isinstance(v, str) else str(v) for v in value]
    return [value if isinstance(value, str) else str(value)]


def _cmp(value: Any, other: Any, op: str) -> bool:
    a, b = to_number(value), to_number(other)
    if a is None or b is None:
        raise RuleError(f"{op} needs numbers, got {value!r} and {other!r}")
    return {"lt": a < b, "le": a <= b, "gt": a > b, "ge": a >= b}[op]


def match_value(value: Any, rule: Any, *, case_sensitive: bool = False) -> bool:
    """Does a value satisfy a rule?

    A rule is either
      * a bare scalar / list -- equality (text via norm_text, numbers via to_number), or
      * a dict -- operator mode; every operator must hold (AND).
    A malformed rule raises RuleError.
    """
    if isinstance(rule, dict):
        op_keys = set(rule) & _OPS
        unknown = set(rule) - _OPS - _AUX
        if op_keys and unknown:
            raise RuleError(f"rule mixes unknown keys {sorted(unknown)}; "
                            f"operators={sorted(_OPS)} parameters={sorted(_AUX)}")
        if not op_keys:
            # No operator at all: compare as a literal dict.
            return value == rule
        cs = bool(rule.get("case_sensitive", case_sensitive))
        return all(_apply_op(value, op, rule, cs) for op in sorted(op_keys))

    # Bare value: numbers first, then text.
    if value is MISSING:
        return False
    a, b = to_number(value), to_number(rule)
    if a is not None and b is not None:
        return math.isclose(a, b, rel_tol=0.0, abs_tol=1e-9)
    if isinstance(rule, (list, tuple, set)) and isinstance(value, (list, tuple, set)):
        return {norm_text(v, case_sensitive=case_sensitive) for v in value} == \
               {norm_text(v, case_sensitive=case_sensitive) for v in rule}
    return norm_text(value, case_sensitive=case_sensitive) == \
        norm_text(rule, case_sensitive=case_sensitive)


def _apply_op(value: Any, op: str, rule: dict, cs: bool) -> bool:  # noqa: C901
    arg = rule[op]

    if op == "exists":
        return (value is not MISSING) == bool(arg)
    # Combinators must let their child rules see missing values; otherwise any([exists:false, eq:""]) would reject an
    # absent optional query parameter.
    if op in ("any", "all"):
        if not isinstance(arg, list):
            raise RuleError(f"the argument of {op} must be a list of rules")
        matches = (match_value(value, r, case_sensitive=cs) for r in arg)
        return any(matches) if op == "any" else all(matches)
    if value is MISSING:
        return False        # apart from exists, nothing holds for a missing value

    if op == "eq":
        return match_value(value, arg, case_sensitive=cs)
    if op == "ne":
        return not match_value(value, arg, case_sensitive=cs)
    if op in ("lt", "le", "gt", "ge"):
        return _cmp(value, arg, op)

    if op in ("re", "not_re"):
        flags = compile_flags(rule.get("flags"))
        if not cs and not (flags & re.IGNORECASE):
            flags |= re.IGNORECASE
        text = value if isinstance(value, str) else str(value)
        hit = any(re.search(p, text, flags) for p in _as_text_list(arg))
        return hit if op == "re" else not hit

    if op in ("in", "not_in"):
        if not isinstance(arg, (list, tuple, set)):
            raise RuleError(f"the argument of {op} must be a list, got {type(arg).__name__}")
        hit = any(match_value(value, cand, case_sensitive=cs) for cand in arg)
        return hit if op == "in" else not hit

    if op in ("contains", "not_contains"):
        needles = _as_text_list(arg)
        if isinstance(value, (list, tuple, set)):
            hay = {norm_text(v, case_sensitive=cs) for v in value}
            hit = all(norm_text(n, case_sensitive=cs) in hay for n in needles)
        else:
            hay_text = norm_text(value, case_sensitive=cs)
            hit = all(norm_text(n, case_sensitive=cs) in hay_text for n in needles)
        return hit if op == "contains" else not hit

    if op == "approx":
        a, b = to_number(value), to_number(arg)
        if a is None or b is None:
            raise RuleError(f"approx needs numbers, got {value!r} and {arg!r}")
        tol = float(rule.get("tol", 1e-6))
        if rule.get("relative"):
            return math.isclose(a, b, rel_tol=tol, abs_tol=0.0)
        return abs(a - b) <= tol

    if op == "range":
        if not (isinstance(arg, (list, tuple)) and len(arg) == 2):
            raise RuleError("the argument of range must be [lo, hi]")
        num = to_number(value)
        lo, hi = to_number(arg[0]), to_number(arg[1])
        if num is None or lo is None or hi is None:
            raise RuleError(f"range needs numbers, got {value!r} and {arg!r}")
        closed = str(rule.get("closed", "cc"))
        if len(closed) != 2 or set(closed) - {"c", "o"}:
            raise RuleError("closed must be two characters, each c (closed) or o (open), e.g. 'co'")
        left = lo <= num if closed[0] == "c" else lo < num
        right = num <= hi if closed[1] == "c" else num < hi
        return left and right

    if op == "len":
        try:
            size = len(value)
        except TypeError as exc:
            raise RuleError(f"len applied to a value without a length: {value!r}") from exc
        return match_value(size, arg, case_sensitive=cs)

    if op == "not":
        return not match_value(value, arg, case_sensitive=cs)

    raise RuleError(f"operator not implemented: {op}")   # pragma: no cover
