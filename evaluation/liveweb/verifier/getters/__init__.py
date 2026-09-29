"""Getters -- the evidence layer.

    get_<name>(ctx, config) -> evidence

A getter captures evidence and never decides right or wrong; its value goes to a metric unchanged.
"""

from __future__ import annotations

from typing import Any, Callable

from ..metrics.rules import MISSING
from .base import EvalContext, GetterError
from .health import get_page_health
from .navigation import (get_final_url, get_nav_history, get_open_tabs,
                         get_url_parse, get_url_path_parse)
from .page import (get_axtree_query, get_control_state, get_dom_count,
                   get_dom_extract, get_page_text, get_page_title,
                   get_visible_text)
from .posthoc import get_agent_answer, get_const
from .server import get_api_state, get_storage

__all__ = ["GETTERS", "POSTHOC_SAFE", "get_getter", "resolve", "collect",
           "to_jsonable", "from_jsonable", "GetterError", "EvalContext", "MISSING"]

#: How MISSING is stored in evidence.json. It cannot be null: null can be a real value ("the field exists and is
#: null"), which differs from "not on the page".
_MISSING_TAG = {"__missing__": True}

#: Getter registry: `{"type": "<key>"}` in a task looks up this table. A misspelled name must fail loudly instead of
#: silently scoring 0 on one task.
GETTERS: dict[str, Callable[[Any, dict], Any]] = {
    # read from files (no session needed)
    "const": get_const,
    "agent_answer": get_agent_answer,
    # navigation (most stable)
    "final_url": get_final_url,
    "url_parse": get_url_parse,
    "url_path_parse": get_url_path_parse,
    "open_tabs": get_open_tabs,
    "nav_history": get_nav_history,
    # page content and controls
    "page_text": get_page_text,
    "page_title": get_page_title,
    "dom_extract": get_dom_extract,
    "dom_count": get_dom_count,
    "control_state": get_control_state,
    "visible_text": get_visible_text,
    "axtree_query": get_axtree_query,
    # server side and storage
    "api_state": get_api_state,
    "storage": get_storage,
    # environment
    "page_health": get_page_health,
}

#: Getters that need no live session. At scoring time (post-hoc) only these run; every other getter must be captured
#: before the session is deleted and stored in evidence.json.
POSTHOC_SAFE = frozenset({"const", "agent_answer"})


def get_getter(name: str) -> Callable[[Any, dict], Any]:
    """Look up a getter by name; an unknown name raises GetterError."""
    try:
        return GETTERS[name]
    except KeyError:
        pass
    raise GetterError(f"unknown getter: {name!r}; registered: {sorted(GETTERS)}")


def resolve(ctx: Any, config: Any) -> Any:
    """Turn one `result` / `expected` configuration into evidence.

    * a dict with a `type` -> call that getter
    * anything else (bare scalar / list / dict without type) -> a literal constant
    """
    if isinstance(config, dict) and "type" in config:
        name = config["type"]
        if not isinstance(name, str):
            raise GetterError(f"a getter type must be a string, got {type(name).__name__}")
        return get_getter(name)(ctx, config)
    return config


def to_jsonable(value: Any) -> Any:
    """Make evidence storable (MISSING -> a tag dict)."""
    if value is MISSING:
        return dict(_MISSING_TAG)
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def from_jsonable(value: Any) -> Any:
    """Inverse of `to_jsonable`, applied when evidence.json is read back."""
    if isinstance(value, dict):
        if value.get("__missing__") is True and len(value) == 1:
            return MISSING
        return {k: from_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [from_jsonable(v) for v in value]
    return value


def collect(ctx: Any, specs: dict, *, phase: str = "in_session") -> dict:
    """Capture a table of {name: config}; one failing getter does not affect the others."""
    out: dict[str, dict] = {}
    for name, config in (specs or {}).items():
        kind = config.get("type") if isinstance(config, dict) else None
        if phase == "post_hoc" and kind is not None and kind not in POSTHOC_SAFE:
            out[name] = {"ok": False,
                         "error": f"GetterError: {kind!r} needs a live session and must be captured before the "
                                  f"session is deleted"}
            continue
        try:
            out[name] = {"ok": True, "value": to_jsonable(resolve(ctx, config))}
        except GetterError as exc:
            out[name] = {"ok": False, "error": f"GetterError: {exc}"}
        except Exception as exc:                               # noqa: BLE001
            out[name] = {"ok": False,
                         "error": f"{type(exc).__name__}: {exc}"}
    return out
