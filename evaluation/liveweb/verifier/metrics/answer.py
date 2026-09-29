"""Final-answer check -- judge the content, not the wrapping.

What a web task asks for is facts: values, units, names, complete sets. The wrapping is not part of the answer. The
same correct answer with or without a ```json fence, with or without "Here is the result:", written as `1239722` or
`$1,239,722`, is the same answer.

So `answer_match` runs two channels in parallel for every {path: rule}; a hit in either counts:

  1. structured -- every JSON value found in the reply (bare, fenced, or embedded in prose) is parsed and the path is
     resolved and checked with the rule language. Only this channel can tell "answered too many items" in a set.
  2. text -- the whole reply is the haystack; does the expected value appear in some accepted spelling? Numbers
     accept thousands separators, currency symbols, percent signs and English number words (`1,239,722`,
     `$1,239,722`, `one`); text accepts case and whitespace differences.

The two channels are OR-ed, not a fallback chain. This is a deliberate relaxation: a reply that mentions the correct
value in passing also counts. The one exception is sets: when the expected value is a list and the agent submitted a
structured list at that path, the structured verdict is final.

Things this metric deliberately does not do (they would be guessing):
  * no synonyms: `Celsius` is not `C`, `yes` is not `true` -- accepted spellings are written into the task's rule;
  * no rescaling: `1.24 million` is not `1239722`;
  * no picking of a convenient JSON block out of a self-contradicting reply -- but prose around JSON is fine.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

from .rules import (MISSING, RuleError, compile_flags, match_value, norm_text,
                    resolve_path, to_number)

__all__ = ["answer_match", "json_candidates"]


# -- JSON out of a reply -----------------------------------------------------------------------------------------

def _reject_constant(value: str):
    raise ValueError(f"Non-finite JSON constant: {value}")


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("JSON number overflow")
    return number


def _finite_int(value: str) -> int:
    number = int(value)
    try:
        finite = math.isfinite(float(number))
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError("JSON integer exceeds numeric comparison range")
    return number


def _unique_object(pairs):
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


_DECODER = json.JSONDecoder(parse_constant=_reject_constant,
                            parse_float=_finite_float, parse_int=_finite_int,
                            object_pairs_hook=_unique_object)


def json_candidates(text: str) -> list[Any]:
    """Every top-level JSON object/array that parses out of a reply, in order of appearance.

    Fences, surrounding prose and several objects in one reply are all accepted -- scoring only asks whether one of
    them carries the right value. A block with duplicate keys, non-finite numbers or extreme nesting is skipped (as
    if it did not exist); it does not void the whole reply.
    """
    out: list[Any] = []
    index, size = 0, len(text)
    while index < size:
        if text[index] not in "{[":
            index += 1
            continue
        try:
            value, end = _DECODER.raw_decode(text, index)
        except (ValueError, RecursionError):
            index += 1
            continue
        out.append(value)
        index = max(end, index + 1)     # skip the whole block; nested values are reached through paths
    return out


# -- text channel ------------------------------------------------------------------------------------------------

#: Common spellings of a number: optional sign, currency symbol, thousands groups (comma or thin/no-break space),
#: decimals, percent. Groups must have three digits, otherwise `19 21` would be read as `1921`.
_NUMBER = re.compile(r"""
    [-+]?[$\u00a5\u20ac\u00a3]?\s?
    (?: \d{1,3} (?: [,\u00a0\u202f\u2009\x20] \d{3} )+ | \d+ )
    (?: \.\d+ )? %?
""", re.VERBOSE)

#: English words for small integers: tasks often say "one government-issued photo ID".
_WORD_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90, "hundred": 100,
}


def _has_number(text: str, number: float) -> bool:
    for match in _NUMBER.finditer(text):
        got = to_number(match.group(0))
        if got is not None and math.isclose(got, number, rel_tol=0.0,
                                            abs_tol=1e-9):
            return True
    return any(re.search(rf"\b{word}\b", text, re.IGNORECASE)
               for word, value in _WORD_NUMBERS.items() if value == number)


def _has_phrase(text: str, needle: str, *, case_sensitive: bool,
                squash_ws: bool) -> bool:
    if squash_ws:
        hay = re.sub(r"\s+", "", norm_text(text, case_sensitive=case_sensitive))
        pin = re.sub(r"\s+", "", norm_text(needle, case_sensitive=case_sensitive))
        return bool(pin) and pin in hay
    hay = norm_text(text, case_sensitive=case_sensitive)
    pin = norm_text(needle, case_sensitive=case_sensitive)
    if not pin:
        return False
    # Short alphanumeric needles (`C`, `IM`, `F`) need word boundaries, otherwise `C` hits inside `Calgary`. The left
    # side only blocks letters: a unit right after a number is normal (`18C`, `13km/h`). Long needles are not bounded:
    # plural and punctuation differences between page copy and answer are more common than accidental hits.
    if len(pin) <= 4 and pin.isalnum():
        return re.search(rf"(?<![A-Za-z_]){re.escape(pin)}(?![0-9A-Za-z_])",
                         hay) is not None
    return pin in hay


def _has_value(text: str, value: Any, *, case_sensitive: bool,
               squash_ws: bool) -> bool:
    """Does the expected value appear in some spelling in the reply?"""
    if isinstance(value, bool) or value is None or value is MISSING:
        return False        # booleans / null have no single spelling in prose (no synonyms, see module docstring)
    number = to_number(value)
    if number is not None:
        return _has_number(text, number)
    if isinstance(value, (list, tuple, set)):
        return bool(value) and all(
            _has_value(text, item, case_sensitive=case_sensitive,
                       squash_ws=squash_ws) for item in value)
    if isinstance(value, dict):
        return False        # nested objects only go through the structured channel
    return _has_phrase(text, str(value), case_sensitive=case_sensitive,
                       squash_ws=squash_ws)


#: Operators the text channel understands. Numeric comparisons (`ge`/`range`/`approx`, ...) are deliberately left
#: out: "is there a number >= 3 somewhere in the reply" has no checkable answer.
_TEXT_OPS = {"eq", "in", "any", "all", "re", "not_re", "contains",
             "not_contains", "not"}


def _text_rule(text: str, rule: Any, *, case_sensitive: bool,
               squash_ws: bool) -> bool:
    kw = {"case_sensitive": case_sensitive, "squash_ws": squash_ws}
    if not isinstance(rule, dict):
        return _has_value(text, rule, **kw)

    ops = set(rule) & _TEXT_OPS
    if not ops:
        return False        # numeric comparison / literal dict: nothing the text channel can decide
    for op in sorted(ops):
        arg = rule[op]
        if op == "eq":
            hit = _text_rule(text, arg, **kw)
        elif op in ("in", "any"):
            if not isinstance(arg, (list, tuple, set)):
                raise RuleError(f"the argument of {op} must be a list")
            hit = any(_text_rule(text, item, **kw) for item in arg)
        elif op == "all":
            if not isinstance(arg, list):
                raise RuleError("the argument of all must be a list of rules")
            hit = all(_text_rule(text, item, **kw) for item in arg)
        elif op in ("re", "not_re"):
            flags = compile_flags(rule.get("flags"))
            if not case_sensitive and not (flags & re.IGNORECASE):
                flags |= re.IGNORECASE
            patterns = arg if isinstance(arg, (list, tuple, set)) else [arg]
            found = any(re.search(str(p), text, flags) for p in patterns)
            hit = found if op == "re" else not found
        elif op in ("contains", "not_contains"):
            needles = arg if isinstance(arg, (list, tuple, set)) else [arg]
            found = all(_has_value(text, n, **kw) for n in needles)
            hit = found if op == "contains" else not found
        else:                                   # not
            hit = not _text_rule(text, arg, **kw)
        if not hit:
            return False
    return True


# -- structured channel ------------------------------------------------------------------------------------------

def _views(result: Any) -> tuple[str, list[Any]]:
    """Split a final answer into (text haystack, structured candidates)."""
    if result is MISSING or result is None:
        return "", []
    if isinstance(result, str):
        return result, json_candidates(result)
    try:
        text = json.dumps(result, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = str(result)
    return text, [result]


def answer_match(result: Any, expected: Any, *, partial: bool = False,
                 case_sensitive: bool = False,
                 ignore_literal_whitespace: bool = False) -> float:
    """Judge a final answer by content: structured and text channels in parallel, a hit in either counts.

    `expected` is {path: rule} (as in `json_object_match`); a bare rule is the same as {".": rule}, i.e. "the whole
    answer is this value / this value appears in the reply". `expected` may also be the dict a getter returned
    (e.g. `dom_extract`): whatever the page shows must be in the answer, so the check survives copy changes.

    partial=True gives partial credit (paths hit / paths), useful for diagnosis; the default is 1.0 only if every path
    is hit, else 0.0.
    """
    if not isinstance(ignore_literal_whitespace, bool):
        raise RuleError("ignore_literal_whitespace must be a bool")
    if isinstance(expected, dict) and not expected:
        raise RuleError("the expected value of answer_match must not be empty")
    rules = expected if isinstance(expected, dict) else {".": expected}

    text, structures = _views(result)
    squash = ignore_literal_whitespace

    def structured(value: Any, rule: Any) -> bool:
        if squash and isinstance(value, str) and isinstance(rule, str):
            value = re.sub(r"\s+", "", value)
            rule = re.sub(r"\s+", "", rule)
        return match_value(value, rule, case_sensitive=case_sensitive)

    hits: list[bool] = []
    for path, rule in rules.items():
        found = [v for structure in structures
                 for v in resolve_path(structure, str(path))]
        if found:
            hit = any(structured(value, rule) for value in found)
            if not hit and isinstance(rule, (list, tuple, set)):
                # The expected value is a set and the agent submitted a structured set at this path: that is its
                # answer. Do not fall back to the text channel, which can count missing items but not extra ones.
                hits.append(False)
                continue
        else:
            # The path resolved nowhere: rules such as {"exists": false} must still hold.
            hit = structured(MISSING, rule)
        hits.append(hit or _text_rule(text, rule, case_sensitive=case_sensitive,
                                      squash_ws=squash))

    if partial:
        return sum(hits) / len(hits)
    return 1.0 if all(hits) else 0.0
