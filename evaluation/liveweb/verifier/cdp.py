"""The evidence channel -- an independent CDP websocket, unrelated to how the agent talks to the browser.

Evidence is never captured through the agent's own connection. If it were, the same page state could yield different
evidence depending on the agent's framework, and score differences could no longer be attributed to the agent. The
verifier is the constant, the agent is the variable; so evidence capture opens its own connection to the cloud
browser, given only the session's CDP websocket URL.

Two hard rules:
1. "The current page" is decided by browser facts, not by what the agent believes (see `choose_page_target`). The
   agent's own idea of its tab is recorded for audit only.
2. Never pretend success. Cannot connect, no page, command timeout -> GetterError, which scoring turns into
   `judge_error` (reward None), not 0. A 0 means "the agent was wrong"; broken capture means "we cannot tell".
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Callable

from .getters import GetterError

#: Connect timeout. The browser is already warm: this connection is opened after the episode.
CONNECT_TIMEOUT = 15.0
#: Default budget of one CDP command.
DEFAULT_TIMEOUT = 30.0
#: recv slice: a short socket timeout lets the loop wake up on its deadline.
_RECV_SLICE = 1.0

#: Targets that are not content pages.
_NOT_A_PAGE = ("devtools://", "chrome-extension://", "chrome://", "edge://")


# -- transport ---------------------------------------------------------------------------------------------------

class CdpTransport:
    """One flat-mode CDP connection to the cloud browser: send a command, wait for THAT command's response.

    Flat mode (`Target.attachToTarget {flatten: true}`) lets one socket talk to several targets; every command carries
    its `sessionId`.
    """

    def __init__(self, ws_url: str, *, connect_timeout: float = CONNECT_TIMEOUT):
        try:
            from websocket import create_connection
        except ImportError as exc:     # pragma: no cover - pinned in requirements.txt
            raise GetterError(f"the evidence channel needs websocket-client: {exc}") from exc
        try:
            # No Origin header: Chrome rejects DevTools websockets that carry one unless it was started with
            # --remote-allow-origins, and a CDP client needs none.
            self._ws = create_connection(ws_url, timeout=connect_timeout, suppress_origin=True)
        except Exception as exc:                               # noqa: BLE001
            raise GetterError(f"the evidence channel cannot reach the browser: "
                              f"{type(exc).__name__}: {exc}") from exc
        self._ws.settimeout(_RECV_SLICE)
        self._lock = threading.Lock()
        self._next_id = 0
        self._closed = False

    def send(self, method: str, params: dict | None = None, *,
             session_id: str | None = None,
             timeout: float = DEFAULT_TIMEOUT) -> dict:
        """Send one command and return its `result`. Errors and timeouts raise GetterError.

        Events (messages without `id`) and other commands' responses are discarded.
        """
        if self._closed:
            raise GetterError(f"the evidence channel is closed; cannot call {method}")
        with self._lock:
            self._next_id += 1
            cmd_id = self._next_id
            message: dict = {"id": cmd_id, "method": method,
                             "params": params or {}}
            if session_id:
                message["sessionId"] = session_id
            try:
                self._ws.send(json.dumps(message, ensure_ascii=False))
            except Exception as exc:                           # noqa: BLE001
                raise GetterError(f"{method}: send failed: "
                                  f"{type(exc).__name__}: {exc}") from exc
            deadline = time.monotonic() + max(1.0, float(timeout))
            while True:
                if time.monotonic() >= deadline:
                    # A timeout is not "nothing on the page"; it is capture that got no answer.
                    raise GetterError(f"{method}: no response within {timeout:g}s")
                try:
                    raw = self._ws.recv()
                except Exception as exc:                       # noqa: BLE001
                    if _is_timeout(exc):
                        continue        # nothing in this slice; check the deadline again
                    raise GetterError(f"{method}: connection lost: "
                                      f"{type(exc).__name__}: {exc}") from exc
                try:
                    payload = json.loads(raw)
                except (TypeError, ValueError):
                    continue
                if payload.get("id") != cmd_id:
                    continue            # an event or another command's response
                if "error" in payload:
                    raise GetterError(f"{method}: CDP error: {payload['error']}")
                return payload.get("result") or {}

    def close(self) -> None:
        self._closed = True
        try:
            self._ws.close()
        except Exception:                                      # noqa: BLE001,S110
            pass       # a failed close must not discard evidence already captured


def _is_timeout(exc: BaseException) -> bool:
    """websocket-client read timeouts, recognized by class name."""
    return type(exc).__name__ in ("WebSocketTimeoutException", "timeout",
                                  "TimeoutError")


# -- the one rule for "the current page" -------------------------------------------------------------------------

def choose_page_target(targets: list, *,
                       visible_of: Callable[[str], bool | None]) -> tuple[str, dict]:
    """Pick the page the agent ended on from `Target.getTargets`, using browser facts only:

      1. only `type == "page"` targets, excluding DevTools / extensions / chrome:// pages;
      2. exactly one -> that one;
      3. several -> ask each `document.visibilityState`: there is only one foreground tab;
      4. none visible (all in background / probing failed) -> the last page that is not about:blank
         (`Target.getTargets` returns roughly in creation order, so the last one is the most recently opened).

    Returns (target_id, audit); the audit is stored in evidence.json.
    """
    pages = [t for t in targets
             if isinstance(t, dict) and t.get("type") == "page"
             and isinstance(t.get("targetId"), str)
             and not str(t.get("url") or "").startswith(_NOT_A_PAGE)]
    audit: dict = {"n_page_targets": len(pages),
                   "urls": [str(t.get("url") or "")[:200] for t in pages]}
    if not pages:
        raise GetterError("the browser has no page target -- evidence must be captured before the session is deleted")
    if len(pages) == 1:
        return pages[0]["targetId"], audit | {"rule": "only_page"}

    visible = []
    probe_errors = 0
    for page in pages:
        try:
            if visible_of(page["targetId"]):
                visible.append(page)
        except Exception:                                      # noqa: BLE001
            probe_errors += 1
    audit["probe_errors"] = probe_errors
    if visible:
        chosen = visible[-1]           # several windows: the most recent foreground page
        return chosen["targetId"], audit | {"rule": "visible",
                                            "n_visible": len(visible)}

    named = [t for t in pages if str(t.get("url") or "") not in ("", "about:blank")]
    chosen = (named or pages)[-1]
    return chosen["targetId"], audit | {"rule": "last_non_blank" if named
                                        else "last_page"}


# -- evidence context --------------------------------------------------------------------------------------------

class CdpEvalContext:
    """The getters' context on a raw CDP connection.

    `browser_context_id` scopes the evidence to one browser context. A cloud session is a browser of its own, so no
    scope is needed there; runs on a local Chrome share one browser, each in its own context, and a run's evidence
    must not see another run's tabs or storage. With a scope, only the context's pages are candidates for "the
    current page", `Target.getTargets` results are filtered to the context, and `Target.createTarget` opens the tab
    in it.
    """

    def __init__(self, transport: CdpTransport, *, task: dict, task_dir: Path,
                 record: dict | None = None, network: list | None = None,
                 agent_target_id: str | None = None,
                 browser_context_id: str | None = None):
        self.transport = transport
        self.task = task or {}
        self.task_dir = task_dir
        self.record = record or {}
        self.network = network or []
        #: The agent's own idea of its tab. Audit only, never used for selection (rule 1).
        self.agent_target_id = agent_target_id
        self.browser_context_id = browser_context_id
        self._sessions: dict[str, str] = {}  # targetId -> flat sessionId
        self._target_id: str | None = None
        self.target_audit: dict = {}

    def _in_scope(self, infos: list) -> list:
        if not self.browser_context_id:
            return infos
        return [t for t in infos if isinstance(t, dict) and t.get("browserContextId") == self.browser_context_id]

    # -- targets --

    def _attach(self, target_id: str) -> str:
        cached = self._sessions.get(target_id)
        if cached:
            return cached
        out = self.transport.send("Target.attachToTarget",
                                  {"targetId": target_id, "flatten": True},
                                  timeout=DEFAULT_TIMEOUT)
        session_id = out.get("sessionId")
        if not isinstance(session_id, str) or not session_id:
            raise GetterError(f"attaching to {target_id} returned no sessionId: {out}")
        self._sessions[target_id] = session_id
        return session_id

    def _visible(self, target_id: str) -> bool | None:
        session_id = self._attach(target_id)
        out = self.transport.send(
            "Runtime.evaluate",
            {"expression": "document.visibilityState", "returnByValue": True},
            session_id=session_id, timeout=10)
        return (out.get("result") or {}).get("value") == "visible"

    def target_id(self) -> str:
        """The current page's targetId, chosen once on first use and then fixed, so that all getters of one capture
        read the same page."""
        if self._target_id is None:
            out = self.transport.send("Target.getTargets",
                                      timeout=DEFAULT_TIMEOUT)
            targets = self._in_scope(out.get("targetInfos") or [])
            self._target_id, audit = choose_page_target(
                targets, visible_of=self._visible)
            if self.browser_context_id:
                audit["browser_context_id"] = self.browser_context_id
            if self.agent_target_id:
                audit["agent_target_id"] = self.agent_target_id
                audit["agrees_with_agent"] = (
                    self.agent_target_id == self._target_id)
            self.target_audit = audit
        return self._target_id

    # -- channel --

    def cdp(self, method: str, params: dict | None = None,
            *, timeout: float = DEFAULT_TIMEOUT, target: str | None = None) -> dict:
        """One CDP call, returned as {"result": <CDP result>} (Runtime.evaluate's result itself has a `result` key,
        so the extra level keeps unwrapping unambiguous)."""
        params = dict(params or {})
        if method == "Target.createTarget" and self.browser_context_id:
            params.setdefault("browserContextId", self.browser_context_id)
        session_id = self._attach(target or self.target_id())
        out = self.transport.send(method, params, session_id=session_id,
                                  timeout=timeout)
        if method == "Target.getTargets" and self.browser_context_id:
            out = dict(out, targetInfos=self._in_scope(out.get("targetInfos") or []))
        return {"result": out}

    def js(self, expr: str, *, await_promise: bool = False,
           timeout: float = 60, target: str | None = None) -> object:
        """Evaluate in the page; with `target`, in THAT tab (localStorage is per origin, and the agent may have left
        the state in another tab)."""
        out = self.cdp("Runtime.evaluate",
                       {"expression": expr, "returnByValue": True,
                        "awaitPromise": bool(await_promise)},
                       timeout=timeout, target=target)
        payload = out.get("result", out) if isinstance(out, dict) else {}
        if not isinstance(payload, dict):
            raise GetterError(f"Runtime.evaluate returned {type(payload).__name__}")
        if payload.get("exceptionDetails"):
            detail = payload["exceptionDetails"]
            text = (detail.get("exception") or {}).get("description") \
                or detail.get("text") or str(detail)
            raise GetterError(f"JS threw: {str(text)[:300]}")
        remote = payload.get("result")
        if not isinstance(remote, dict):
            raise GetterError("Runtime.evaluate returned no result")
        if remote.get("type") == "undefined":
            return None
        return remote.get("value")

    # -- lifecycle --

    def close(self) -> None:
        """Close this evidence connection. The agent's session is deleted by the episode's teardown, not here."""
        self.transport.close()
