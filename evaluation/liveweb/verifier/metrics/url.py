"""URL checks -- the most stable layer; prefer it whenever it can express the task.

A URL is not affected by ad slots, lazy loading, A/B experiments or regional differences; it only breaks when a site
changes its routes.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlsplit

from .rules import RuleError, compile_flags

__all__ = ["normalize_url", "url_matches", "url_pattern_match", "tabs_match"]

_DEFAULT_PORTS = {"http": "80", "https": "443"}


def normalize_url(url: str, *, ignore_scheme: bool = True, ignore_www: bool = True,
                  ignore_query: bool = True, ignore_fragment: bool = True,
                  ignore_trailing_slash: bool = True) -> str:
    """Normalize a URL with the standard library only: lowercase host, drop `www.` and default ports, optionally drop
    query / fragment / trailing slash.

    Public suffixes are NOT ignored: on the live web `.com` and `.co.uk` are genuinely different sites (currency,
    inventory, region).
    """
    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower()
    if ignore_www and host.startswith("www."):
        host = host[4:]
    netloc = host
    if parts.port and str(parts.port) != _DEFAULT_PORTS.get(scheme):
        netloc = f"{host}:{parts.port}"
    path = parts.path or "/"
    if ignore_trailing_slash and len(path) > 1:
        path = path.rstrip("/") or "/"
    out = netloc + path
    if not ignore_scheme:
        out = f"{scheme}://{out}"
    if not ignore_query and parts.query:
        out += "?" + parts.query
    if not ignore_fragment and parts.fragment:
        out += "#" + parts.fragment
    return out


def url_matches(result: Any, expected: Any, **options: Any) -> float:
    """The current URL equals the expected one after normalization. `expected` may be one string or a list.

    options: ignore_scheme / ignore_www / ignore_query / ignore_fragment / ignore_trailing_slash (all True by default).
    """
    if not isinstance(result, str):
        return 0.0
    got = normalize_url(result, **options)
    cands = expected if isinstance(expected, (list, tuple)) else [expected]
    return 1.0 if any(got == normalize_url(str(c), **options) for c in cands) else 0.0


def url_pattern_match(result: Any, expected: Any, *, flags: str | None = None,
                      mode: str = "all") -> float:
    """`re.search` with each pattern; by default ALL patterns must match (mode="any" relaxes that).

    Case-insensitive by default: URL case is not stable on real sites. Path boundaries are written into the pattern,
    e.g. '^https://site\\.com/cart(/|$|\\?)', so that /cart does not match /cartoon.
    """
    if not isinstance(result, str):
        return 0.0
    pats = expected if isinstance(expected, (list, tuple)) else [expected]
    if not pats:
        raise RuleError("the expected value of url_pattern_match must not be empty")
    f = compile_flags(flags)
    if not (f & re.IGNORECASE):
        f |= re.IGNORECASE
    hits = [bool(re.search(str(p), result, f)) for p in pats]
    if mode == "any":
        return 1.0 if any(hits) else 0.0
    if mode != "all":
        raise RuleError(f"url_pattern_match mode must be all/any, got {mode!r}")
    return 1.0 if all(hits) else 0.0


def tabs_match(result: Any, expected: Any, *, mode: str = "subset",
               **options: Any) -> float:
    """The set of open tabs.

    result: what `open_tabs` returns (list[{url, title}] or list[str]).
    mode: "subset" (every expected tab is open, default) / "equal" (same set).
    """
    if not isinstance(result, (list, tuple)):
        return 0.0
    got = {normalize_url(t.get("url", "") if isinstance(t, dict) else str(t), **options)
           for t in result}
    want = {normalize_url(str(e), **options)
            for e in (expected if isinstance(expected, (list, tuple)) else [expected])}
    if mode == "subset":
        return 1.0 if want <= got else 0.0
    if mode == "equal":
        return 1.0 if want == got else 0.0
    raise RuleError(f"tabs_match mode must be subset/equal, got {mode!r}")
