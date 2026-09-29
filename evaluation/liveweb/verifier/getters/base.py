"""Foundation of the evidence layer: the error type, the context protocol and JS/CDP helpers.

A getter captures evidence; it never decides right or wrong. Its value goes to a metric unchanged, so a check can be
changed without re-running the episode, and evidence capture can be changed without rewriting checks.

Three kinds of "did not get it" must stay apart:

    got a value, but the content is wrong     -> a normal value, the metric scores 0     -> agent failure
    valid path, nothing on the page           -> MISSING, the metric scores 0            -> agent failure
    the capture mechanism itself is broken    -> raise GetterError                       -> judge_error

Letting the third kind leak into the first two is the most dangerous mistake: the day a site is redesigned every task
would drop to 0 and look like a model regression.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..metrics.rules import MISSING

__all__ = ["GetterError", "EvalContext", "MISSING",
           "js_json", "js_text", "cdp_call", "quote", "first_str"]


class GetterError(RuntimeError):
    """Evidence capture itself failed: a rotten selector, an API 404, a CDP timeout, a canary that did not answer.

    Callers must file it as `judge_error`, never as a 0. It is the twin of the metric layer's RuleError (a bug in the
    task's check); neither is the agent's failure.
    """


@runtime_checkable
class EvalContext(Protocol):
    """Everything a getter may use.

    Two capture phases:
      in-session   after the episode, BEFORE the browser session is deleted. cdp/js are available.
      post-hoc     any time later, reading files. cdp/js are None.

    Once the session is deleted, URL / cookies / DOM / storage are gone.
    """

    task: dict            # task definition
    task_dir: Path        # the task's result directory
    record: dict          # the episode record
    network: list         # captured network events (empty unless recorded)

    def cdp(self, method: str, params: dict | None = None,
            *, timeout: float = 30, target: str | None = None) -> dict:
        """One CDP call; `target` addresses a specific tab (CDP target id), default the current page."""
        ...

    def js(self, expr: str, *, await_promise: bool = False,
           timeout: float = 60, target: str | None = None) -> Any:
        """Evaluate in the page and return the deserialized value (or a JSON string; both are accepted)."""
        ...


# -- session availability ----------------------------------------------------------------------------------------

def _need_session(ctx: Any, what: str) -> None:
    if getattr(ctx, "js", None) is None and getattr(ctx, "cdp", None) is None:
        raise GetterError(f"{what} needs a live session; this is the post-hoc phase")


def js_text(ctx: Any, expr: str, *, what: str, **kw: Any) -> str:
    """Evaluate and require a string."""
    _need_session(ctx, what)
    try:
        raw = ctx.js(expr, **kw)
    except GetterError:
        raise
    except Exception as exc:
        raise GetterError(f"{what}: JS evaluation failed: {exc}") from exc
    if raw is None:
        raise GetterError(f"{what}: JS returned None")
    return raw if isinstance(raw, str) else str(raw)


def js_json(ctx: Any, expr: str, *, what: str, **kw: Any) -> Any:
    """Evaluate and read the result as JSON.

    Page-side code always does `return JSON.stringify(...)` rather than relying on how the context serializes complex
    objects (`returnByValue` is inconsistent across realms and with DOM nodes). An already deserialized dict/list is
    accepted as well.
    """
    _need_session(ctx, what)
    try:
        raw = ctx.js(expr, **kw)
    except GetterError:
        raise
    except Exception as exc:
        raise GetterError(f"{what}: JS evaluation failed: {exc}") from exc
    if isinstance(raw, (dict, list)):
        return raw
    if raw is None:
        raise GetterError(f"{what}: JS returned None")
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except ValueError as exc:
        raise GetterError(f"{what}: JS did not return JSON: {raw[:200]!r}") from exc


def cdp_call(ctx: Any, method: str, params: dict | None = None,
             *, what: str, **kw: Any) -> dict:
    _need_session(ctx, what)
    try:
        out = ctx.cdp(method, params or {}, **kw)
    except GetterError:
        raise
    except Exception as exc:
        raise GetterError(f"{what}: CDP {method} failed: {exc}") from exc
    if not isinstance(out, dict):
        raise GetterError(f"{what}: CDP {method} returned {type(out).__name__}, expected a dict")
    if "error" in out:
        raise GetterError(f"{what}: CDP {method} error: {out['error']}")
    return out.get("result", out) if "result" in out else out


# -- small helpers -----------------------------------------------------------------------------------------------

def quote(value: Any) -> str:
    """Embed a Python value safely in JS source (JSON is a subset of JS)."""
    return json.dumps(value, ensure_ascii=False)


def first_str(record: dict, *names: str) -> Any:
    """The first non-empty string field among `names`, or MISSING."""
    for name in names:
        value = record.get(name)
        if isinstance(value, str) and value.strip():
            return value
    return MISSING
