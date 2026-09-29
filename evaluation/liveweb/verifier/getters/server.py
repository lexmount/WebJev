"""Server-side state and browser storage -- the hardest evidence.

"Was the item really added to the cart" is decided by what the server says, not by a badge rendered on the page.
"""

from __future__ import annotations

import json
import time
import urllib.parse
from typing import Any

from ..metrics.rules import MISSING
from .base import GetterError, cdp_call, js_json, quote

__all__ = ["get_api_state", "get_storage"]


# -- the site's own API, with the session's credentials ----------------------------------------------------------

_FETCH_JS = """
(async () => {
  const opt = {method: %(method)s, credentials: %(credentials)s};
  const h = %(headers)s; if (h) opt.headers = h;
  const b = %(body)s; if (b !== null) opt.body = (typeof b === 'string') ? b : JSON.stringify(b);
  try {
    const r = await fetch(%(url)s, opt);
    const t = await r.text();
    return JSON.stringify({ok: true, status: r.status, text: t});
  } catch (e) {
    return JSON.stringify({ok: false, error: String(e)});
  }
})()
"""


def get_api_state(ctx: Any, config: dict) -> Any:
    """Call the site's own endpoint from the page context, with the session's cookies, to read server-side truth.

    A credentialed `fetch()` in the page carries the httpOnly session cookies, and same-origin APIs answer with JSON.

    config: `url`, `method` (default GET), `headers`, `body`, `credentials` (default `include`), `ok_status`
    (default 2xx), `expect_json` (default true). A status outside `ok_status` raises GetterError: an API 404 means
    broken capture, not a wrong agent.
    """
    url = config.get("url")
    if not url:
        raise GetterError("api_state needs a url")
    expr = _FETCH_JS % {
        "url": quote(url),
        "method": quote(config.get("method", "GET")),
        "credentials": quote(config.get("credentials", "include")),
        "headers": quote(config["headers"]) if config.get("headers") else "null",
        "body": quote(config["body"]) if config.get("body") is not None else "null",
    }
    out = js_json(ctx, expr, what="api_state", await_promise=True)
    if not isinstance(out, dict):
        raise GetterError("api_state: JS did not return an object")
    if not out.get("ok"):
        raise GetterError(f"api_state: fetch failed: {out.get('error')}")

    status = int(out.get("status", 0))
    ok_status = config.get("ok_status")
    if ok_status is None:
        acceptable = 200 <= status < 300
    else:
        codes = ok_status if isinstance(ok_status, (list, tuple)) else [ok_status]
        acceptable = status in {int(c) for c in codes}
    if not acceptable:
        raise GetterError(f"api_state: {url} returned HTTP {status} -- broken capture, "
                          f"this is a judge_error, not a 0")

    text = out.get("text", "")
    if config.get("expect_json", True):
        try:
            return json.loads(text)
        except ValueError as exc:
            raise GetterError(
                f"api_state: {url} did not return JSON (first 200 chars: {text[:200]!r})") from exc
    return text


# -- local storage -----------------------------------------------------------------------------------------------

_STORAGE_JS = """
(() => {
  const store = (%(kind)s === 'session') ? sessionStorage : localStorage;
  const keys = %(keys)s;
  const out = {__origin__: location.origin};
  try {
    const names = keys || Object.keys(store);
    for (const k of names) out[k] = store.getItem(k);
  } catch (e) { return JSON.stringify({__error__: String(e)}); }
  return JSON.stringify(out);
})()
"""


def _target_on_origin(ctx: Any, want_origin: str) -> str | None:
    """An already open tab of the SAME browser on the wanted origin, or None.

    localStorage is per origin and per profile: any tab of that origin in this browser reads the same storage.
    """
    try:
        out = cdp_call(ctx, "Target.getTargets", what="storage")
    except GetterError:
        return None
    infos = out.get("targetInfos")
    if not isinstance(infos, list):
        return None
    want = str(want_origin).rstrip("/")
    for info in infos:
        if not isinstance(info, dict) or info.get("type") != "page":
            continue
        url = str(info.get("url") or "")
        parts = urllib.parse.urlsplit(url)
        if parts.scheme and parts.netloc and f"{parts.scheme}://{parts.netloc}" == want:
            return info.get("targetId")
    return None


def _open_target_on_origin(ctx: Any, want_origin: str) -> str | None:
    """Open a NEW tab on the wanted origin in the same browser and return its targetId (None on failure).

    The last resort, and the only branch of this getter with a side effect. localStorage belongs to the browser
    profile, not to a tab: after the agent navigates its last tab of that origin away, the state is still there,
    only no window shows it. The side effect is bounded: only for kind=local when the other two ways fail, the tab is
    closed right after reading, and it happens after the episode, so it cannot change what the agent saw. Opening a
    site's home page does not create the key a check looks for, so this cannot forge evidence.
    """
    url = str(want_origin).rstrip("/") + "/"
    try:
        out = cdp_call(ctx, "Target.createTarget", {"url": url}, what="storage")
    except GetterError:
        return None
    return out.get("targetId")


def _close_target(ctx: Any, target: str) -> None:
    try:
        cdp_call(ctx, "Target.closeTarget", {"targetId": target}, what="storage")
    except Exception:  # noqa: BLE001 - a failed cleanup must not discard evidence already captured
        pass


def _read_storage_from(ctx: Any, expr: str, target: str | None, want_origin: str,
                       *, tries: int = 1) -> dict:
    """Read storage once in a given tab and CHECK the origin; any other origin raises.

    `tries > 1` is for a freshly opened tab, whose `location.origin` is still about:blank until navigation finishes.
    """
    last = ""
    for attempt in range(tries):
        if attempt:
            time.sleep(1.5)
        out = js_json(ctx, expr, what="storage", target=target)
        if not isinstance(out, dict):
            raise GetterError("storage: the cross-tab read did not return an object")
        if "__error__" in out:
            last = f" could not read it ({out['__error__']})"
            continue
        origin = out.pop("__origin__", "")
        if str(want_origin).rstrip("/") == str(origin).rstrip("/"):
            return out
        last = f" read {origin!r}"
    raise GetterError(f"storage: expected origin {want_origin!r}, but{last}")


def get_storage(ctx: Any, config: dict) -> dict:
    """localStorage / sessionStorage -- for "set as home store" or "saved a preference", which page text cannot show.

    `origin` is a hard constraint and is never relaxed: silently reading another origin's storage is worse than not
    reading at all. What is relaxed is only WHICH TAB is read, and only for kind=local (sessionStorage belongs to a
    tab). Three levels:
      1. the current page;
      2. an already open tab of that origin in the same browser;
      3. a newly opened tab of that origin, closed after reading (the only level with a side effect).
    Only if all three fail does it raise GetterError.

    `json_values: true` parses values that are JSON strings (sites often store a whole preference object as one
    string).
    """
    kind = config.get("kind", "local")
    if kind not in ("local", "session"):
        raise GetterError(f"storage kind must be local/session, got {kind!r}")
    keys = config.get("keys")
    expr = _STORAGE_JS % {"kind": quote(kind),
                          "keys": quote(list(keys)) if keys else "null"}
    want_origin = config.get("origin")

    out = js_json(ctx, expr, what="storage")
    if not isinstance(out, dict):
        raise GetterError("storage: JS did not return an object")
    if "__error__" in out:
        raise GetterError(f"storage: could not read it ({out['__error__']}) -- "
                          f"the origin most likely disables storage")
    origin = out.pop("__origin__", "")

    if want_origin and str(want_origin).rstrip("/") != str(origin).rstrip("/"):
        if kind != "local":
            raise GetterError(
                f"storage: expected origin {want_origin!r}, the page is on {origin!r} -- sessionStorage is per tab "
                f"and cannot be read across origins")
        target = _target_on_origin(ctx, want_origin)
        if target is not None:                        # level 2: an open tab
            out = _read_storage_from(ctx, expr, target, want_origin)
            out["__read_from_tab__"] = target
        else:                                         # level 3: open one, read, close
            target = _open_target_on_origin(ctx, want_origin)
            if target is None:
                raise GetterError(
                    f"storage: expected origin {want_origin!r}, the page is on {origin!r}, and no tab of that "
                    f"origin is open or could be opened")
            try:
                out = _read_storage_from(ctx, expr, target, want_origin, tries=6)
            finally:
                _close_target(ctx, target)
            out["__read_from_opened_tab__"] = want_origin

    if config.get("json_values"):
        # A missing key (getItem -> null) is MISSING in both modes.
        parsed = {}
        for key, value in out.items():
            if value is None:
                parsed[key] = MISSING
                continue
            try:
                parsed[key] = json.loads(value) if isinstance(value, str) else value
            except ValueError:
                parsed[key] = value
        return parsed
    return {k: (MISSING if v is None else v) for k, v in out.items()}
