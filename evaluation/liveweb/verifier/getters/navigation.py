"""Navigation state -- the most stable evidence; prefer it.

A URL is not affected by ad slots, lazy loading, A/B experiments or regional differences.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import parse_qs, urlsplit

from ..metrics.rules import MISSING
from .base import GetterError, cdp_call, js_text

__all__ = ["get_final_url", "get_url_parse", "get_url_path_parse",
           "get_open_tabs", "get_nav_history"]


def get_final_url(ctx: Any, config: dict) -> str:
    """The URL the page ended on."""
    url = js_text(ctx, "location.href", what="final_url")
    if not url.strip():
        raise GetterError("final_url: empty URL (the page may not have navigated yet)")
    return url


def get_url_parse(ctx: Any, config: dict) -> dict:
    """The URL's query parameters.

    Without `parse_keys` every parameter is returned. For repeated keys the FIRST value is kept (what the address bar
    reads); `multi: true` keeps all of them.
    """
    url = config.get("url") or get_final_url(ctx, {})
    raw = parse_qs(urlsplit(url).query, keep_blank_values=True)
    multi = bool(config.get("multi"))
    params = {k: (v if multi else (v[0] if v else "")) for k, v in raw.items()}
    keys = config.get("parse_keys")
    if keys:
        return {k: params.get(k, MISSING) for k in keys}
    return params


def get_url_path_parse(ctx: Any, config: dict) -> Any:
    """The URL's path segments.

    Without `index` the whole list (empty segments removed); with `index` that segment, MISSING when out of range.
    `split_by` changes the separator.
    """
    url = config.get("url") or get_final_url(ctx, {})
    path = urlsplit(url).path
    sep = config.get("split_by", "/")
    parts = [p for p in path.split(sep) if p]
    index = config.get("index")
    if index is None:
        return parts
    try:
        return parts[int(index)]
    except (ValueError, TypeError) as exc:
        raise GetterError(f"url_path_parse: index must be an integer, got {index!r}") from exc
    except IndexError:
        return MISSING


def get_open_tabs(ctx: Any, config: dict) -> list:
    """Open tabs -- [{url, title}], real pages only (type == "page")."""
    out = cdp_call(ctx, "Target.getTargets", what="open_tabs")
    infos = out.get("targetInfos")
    if not isinstance(infos, list):
        raise GetterError("open_tabs: Target.getTargets returned no targetInfos")
    return [{"url": t.get("url", ""), "title": t.get("title", "")}
            for t in infos
            if isinstance(t, dict) and t.get("type") == "page"]


def get_nav_history(ctx: Any, config: dict) -> list:
    """URLs in the navigation history of the CURRENT tab only (CDP has no browser-wide history)."""
    out = cdp_call(ctx, "Page.getNavigationHistory", what="nav_history")
    entries = out.get("entries")
    if not isinstance(entries, list):
        raise GetterError("nav_history: Page.getNavigationHistory returned no entries")
    if config.get("up_to_current"):
        idx = out.get("currentIndex")
        if isinstance(idx, int):
            entries = entries[:idx + 1]
    return [e.get("url", "") for e in entries if isinstance(e, dict)]
