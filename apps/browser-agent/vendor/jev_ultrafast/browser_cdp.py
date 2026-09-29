"""Observed actions on a Chrome browser over CDP. One CDP socket, pipelined bursts.

The decision loop, snapshot.js, and every freshness guard are the upstream ones. What changes is
transport: the browser may be a network round trip away, so checks that upstream ran as separate
local calls are merged into one in-page evaluation, and input events are sent back to back with
the next observation. Freshness is compared inside the page and reported as short hashes.

The browser itself comes from a provider, chosen by BROWSER (see browser_backend):
  - LexmountBrowser: a Lexmount Browser cloud session (https://browser.lexmount.com), one per run;
  - LocalChrome: your own Chrome, started with --remote-debugging-port. Each run gets its own browser
    context and tab, disposed of at the end; the browser and its other tabs are never touched.
Both only hand over a CDP websocket URL and a page; everything after that is the same code.
"""

import hashlib
import json
import os
import re
import stat
import threading
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

from .cdp import CDP

LEXMOUNT_API = "https://api.lexmount.com"
LOCAL_CDP = "http://127.0.0.1:9222"


def browser_backend():
    """"lexmount" or "local": BROWSER when it names one of them, else lexmount when LEXMOUNT_API_KEY is set.

    Other BROWSER values are ignored, because many systems already use BROWSER for the default web browser.
    """
    choice = (os.environ.get("BROWSER") or "").strip().lower()
    if choice in {"lexmount", "local"}:
        return choice
    return "lexmount" if os.environ.get("LEXMOUNT_API_KEY") else "local"


class BrowserUnavailable(RuntimeError):
    """The configured browser cannot be reached. The message names only the local endpoint, so it is safe to show."""


# Atomically read visible content and controls, preserving actual DOM node identity. Verbatim upstream.
READ_STATE = Path(__file__).with_name("snapshot.js").read_text()

# cyrb53 plus length. Equal JSON gives equal hashes; the page compares, Python keeps the short string.
HASH = """(s => { let a=0xdeadbeef, b=0x41c6ce57;
  for (let i=0;i<s.length;i++) { const c=s.charCodeAt(i); a=Math.imul(a^c,2654435761); b=Math.imul(b^c,1597334677); }
  a=Math.imul(a^(a>>>16),2246822507)^Math.imul(b^(b>>>13),3266489909);
  b=Math.imul(b^(b>>>16),2246822507)^Math.imul(a^(a>>>13),3266489909);
  return (4294967296*(2097151&b)+(a>>>0)).toString(36)+'.'+s.length; })"""

# Upstream's post-input wait, verbatim: two animation frames or 50 ms; editable comboboxes wait for
# visible options, capped at 200 ms.
SETTLE = """(action => new Promise(resolve => {
  const field=window.__jevFast?.nodes.get(action.node);
  const autocomplete=action.kind==='fill' && field?.getAttribute('role')==='combobox';
  let frames=0, stopped=false;
  const finish=()=>{stopped=true;resolve()};
  setTimeout(finish,autocomplete ? 200 : 50);
  const ready=()=>{
    if (stopped) return;
    const ids=(field?.getAttribute('aria-controls')||field?.getAttribute('aria-owns')||'')
      .split(/\\s+/).filter(Boolean);
    const roots=ids.length ? ids.map(id=>document.getElementById(id)).filter(Boolean) : [document];
    const options=roots.flatMap(root=>[...root.querySelectorAll('[role="option"]')]);
    if (++frames>=2 && (!autocomplete || options.some(e=>{
      const r=e.getBoundingClientRect();
      return r.width && r.height && r.bottom>0 && r.top<innerHeight &&
        e.checkVisibility({checkOpacity:true,checkVisibilityCSS:true});
    }))) finish();
    else requestAnimationFrame(ready);
  };
  requestAnimationFrame(ready);
}))"""

# Upstream's target resolution, verbatim. Code-owned node IDs refer to actual observed elements,
# never model-generated selectors. Geometry and hit-testing are read again immediately before input.
RESOLVE = """(action => {
  const e=window.__jevFast?.nodes.get(action.node);
  if (!e?.isConnected || e.matches(':disabled') || e.closest('[aria-disabled="true"],[inert]') ||
      !e.checkVisibility({checkOpacity:true,checkVisibilityCSS:true})) return null;
  if (action.kind==='fill' && (e.readOnly || e.getAttribute('aria-readonly')==='true')) return null;
  const r=e.getBoundingClientRect(), x=r.x+r.width/2, y=r.y+r.height/2;
  if (!r.width || !r.height || x<0 || y<0 || x>=innerWidth || y>=innerHeight) return null;
  if (!e.contains(document.elementFromPoint(x,y))) return null;
  if (action.kind==='select') {
    if (e.tagName!=='SELECT' || ![...e.options].some(o=>o.value===action.value &&
        !o.disabled && !o.closest('optgroup[disabled]'))) return null;
    e.value=action.value;
    e.dispatchEvent(new Event('input',{bubbles:true}));
    e.dispatchEvent(new Event('change',{bubbles:true}));
  }
  return {x,y};
})"""

# Snapshot, then replace the large equality-only fields with hashes computed in the page.
SNAPSHOT = f"""(() => {{ const H={HASH}; const state={READ_STATE};
  if (!state) return null;
  state.marker_hash=H(JSON.stringify(state.marker));
  state.page_key_hash=H(JSON.stringify(state.page_key));
  state.guard_hashes={{}};
  for (const [node,guard] of Object.entries(state.guards)) state.guard_hashes[node]=H(JSON.stringify(guard));
  delete state.marker; delete state.page_key; delete state.guards; state.taken_at=Date.now();
  return state; }})()"""

# A wasted prediction costs far more here than locally (the model is an ocean away), so after upstream's
# wait the page is sampled until two consecutive markers agree, up to a cap. The last sample is the observation.
QUIET_MS = int(os.environ.get("JEV_QUIET_MS", "60"))
QUIET_CAP_MS = int(os.environ.get("JEV_QUIET_CAP_MS", "700"))
QUIET_LAST_CAP_MS = int(os.environ.get("JEV_QUIET_LAST_CAP_MS", "250"))  # after the network deadline has passed
QUIET_FUNCTION = f"""(cap => new Promise(resolve => {{ const H={HASH}; const started=performance.now(); let last=null;
  const finish=state => {{ if (!state) return resolve(null);
    state.marker_hash=H(JSON.stringify(state.marker));
    state.page_key_hash=H(JSON.stringify(state.page_key));
    state.guard_hashes={{}};
    for (const [node,guard] of Object.entries(state.guards)) state.guard_hashes[node]=H(JSON.stringify(guard));
    state.settle_ms=Math.round(performance.now()-started); state.taken_at=Date.now();
    delete state.marker; delete state.page_key; delete state.guards;
    resolve(state); }};
  // A menu or dialog that is still animating will change the page when it lands. Spinners never end: ignored.
  const animating=() => document.getAnimations().some(a => a.playState==='running' &&
    Number.isFinite(Number(a.effect?.getComputedTiming?.().endTime ?? Infinity)));
  const sample=() => {{ const state={READ_STATE};
    const hash=state && !animating() ? H(JSON.stringify(state.marker)) : null;
    if ((hash!==null && hash===last) || performance.now()-started>=cap) return finish(state);
    last=hash; setTimeout(sample,{QUIET_MS}); }};
  sample(); }}))"""
QUIET_SNAPSHOT = f"{QUIET_FUNCTION}({QUIET_CAP_MS})"

# Same comparisons as upstream Browser.fresh(), evaluated in the page.
FRESH = f"""((scoped,node,pageKey,guard,marker) => {{ const H={HASH};
  if (scoped) {{ const c=window.__jevFast; if (!c) return false;
    const current=c.guard(c.nodes.get(node));
    return H(JSON.stringify(c.pageKey()))===pageKey &&
      (guard===null ? current===null : H(JSON.stringify(current))===guard); }}
  const state={READ_STATE};
  return !!state && H(JSON.stringify(state.marker))===marker; }})"""


# Observing while the page still waits for its own data wastes a whole prediction: the answer arrives, the
# page re-renders, and the freshness gate rejects the decision. So an observation also waits for same-origin
# XHR/fetch/document requests to finish (the usual "network idle" readiness rule), within a cap.
NET_CAP_MS = int(os.environ.get("JEV_NET_CAP_MS", "2300"))
NET_GRACE_MS = int(os.environ.get("JEV_NET_GRACE_MS", "100"))  # time for the page to render a response
NET_STREAM_MS = 4000  # requests older than this are treated as long-lived streams, not page loads

class NetworkWatch:
    def __init__(self, cdp, session):
        self.session = session
        self.changed = threading.Condition()
        self.inflight = {}
        self.events = 0
        self.last_finish = 0.0  # local wall clock when the last relevant request was seen to finish
        cdp.handlers["Network.requestWillBeSent"] = self.started
        cdp.handlers["Network.loadingFinished"] = self.finished
        cdp.handlers["Network.loadingFailed"] = self.finished

    def started(self, message):
        p = message["params"]
        if message.get("sessionId") != self.session or p.get("type") not in {"XHR", "Fetch", "Document"}:
            return
        url, page = urlsplit(p["request"]["url"]), urlsplit(p.get("documentURL", ""))
        if p["type"] != "Document" and (url.scheme, url.netloc) != (page.scheme, page.netloc):
            return  # third-party beacons and logging never gate an observation
        with self.changed:
            self.inflight[p["requestId"]] = time.time()
            self.events += 1
            self.changed.notify_all()

    def finished(self, message):
        with self.changed:
            if self.inflight.pop(message["params"]["requestId"], None) is not None:
                self.events += 1
                self.last_finish = time.time()
                self.changed.notify_all()

    def busy(self):
        now = time.time()
        return any(now - started < NET_STREAM_MS / 1000 for started in self.inflight.values())

    def wait_idle(self, cap_ms):
        deadline = time.time() + cap_ms / 1000
        with self.changed:
            while self.busy() and time.time() < deadline:
                self.changed.wait(min(0.05, max(0.0, deadline - time.time())))
        return not self.busy()


class StalePage(ValueError):
    """A decision no longer refers to the observed page."""


# CDP errors that mean the document went away underneath a read. Reads may be retried; inputs never are.
NAVIGATING = ("context", "navigat", "target closed", "frame", "Promise was collected")


class LexmountBrowser:
    """A Lexmount Browser cloud session (https://browser.lexmount.com): created for the run, or adopted once from
    the app's warm pool. Needs LEXMOUNT_API_KEY and LEXMOUNT_PROJECT_ID."""

    name = "lexmount"

    def __init__(self):
        from lexmount import Lexmount

        self.client = Lexmount(base_url=LEXMOUNT_API, timeout=30)
        self.session = None

    def open(self, browser):
        lease_file = os.environ.pop("BROWSER_AGENT_SESSION_LEASE_FILE", None)
        if lease_file:
            started = time.perf_counter()
            try:
                self.session = self._consume_lease(lease_file)
            except Exception:
                browser.session_pool_failure = "lease_invalid"
            else:
                try:
                    self._connect(browser)
                    browser.session_source = "pool"
                except Exception:
                    browser.session_pool_failure = "pool_attach_failed"
                    self.release(browser)
            browser.session_pool_attempt_ms = round((time.perf_counter() - started) * 1000)
            if browser.session_source != "pool":
                browser.session_source = "cold_fallback"
        if self.session is None:
            started = time.perf_counter()
            self.session = self._create()
            browser.session_create_ms = round((time.perf_counter() - started) * 1000)
            self._connect(browser)

    def _connect(self, browser):
        browser.connect_url = self.session.connect_url
        browser.inspect_url = getattr(self.session, "inspect_url", "") or ""
        browser.attach()

    def page(self, cdp):
        """The session's first page, or a new one."""
        pages = [t for t in cdp.call("Target.getTargets")["targetInfos"] if t["type"] == "page"]
        return pages[0]["targetId"] if pages else cdp.call("Target.createTarget", url="about:blank")["targetId"]

    def _create(self):
        try:
            return self.client.sessions.create(poll_timeout_sec=90)
        except Exception as exc:
            # A create that times out may still start a session that nobody would close: delete the one it names.
            for session_id in set(re.findall(r"session_[0-9a-f-]{36}", str(exc))):
                try:
                    self.client.sessions.delete(session_id=session_id)
                except Exception:  # noqa: BLE001 - best effort; the original error is what matters
                    pass
            raise

    def _consume_lease(self, filename):
        """Consume server-created private metadata once, without a create/get request."""
        from lexmount import SessionInfo

        path = Path(filename)
        try:
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
                info = os.fstat(stream.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077 or info.st_size > 16_384:
                    raise ValueError("Invalid private session lease")
                lease = json.load(stream)
            if not isinstance(lease, dict):
                raise ValueError("Invalid session lease")
            for key, limit in (("id", 256), ("connect_url", 8192), ("inspect_url", 8192)):
                value = lease.get(key, "")
                if not isinstance(value, str) or len(value) > limit or any(ord(c) < 32 for c in value):
                    raise ValueError("Invalid session lease field")
            endpoint = urlsplit(lease["connect_url"])
            if not lease.get("id") or endpoint.scheme not in {"ws", "wss"} or not endpoint.hostname or endpoint.username:
                raise ValueError("Invalid session lease endpoint")
            return SessionInfo(
                id=lease["id"],
                ws=lease["connect_url"],
                inspect_url=lease.get("inspect_url", ""),
                created_at=str(lease.get("created_at", ""))[:128],
                client=self.client,
            )
        finally:
            path.unlink(missing_ok=True)

    def endpoint(self):
        return {"session_id": getattr(self.session, "id", None)}

    def release(self, browser):
        """Close the CDP socket and delete the session."""
        browser.disconnect()
        if self.session:
            try:
                self.session.close()
            finally:
                self.session = None

    def close(self, browser):
        try:
            self.release(browser)
        finally:
            self.client.close()


class LocalChrome:
    """Your own Chrome or Chromium, started with --remote-debugging-port (CHROME_CDP_URL, default
    http://127.0.0.1:9222). The run opens its own browser context (separate cookies and storage) with one tab and
    disposes of it at the end; the browser, its profile and its other tabs are never touched."""

    name = "local"

    def __init__(self):
        self.url = (os.environ.get("CHROME_CDP_URL") or LOCAL_CDP).strip()
        self.context = None

    def open(self, browser):
        browser.connect_url = self._websocket()
        started = time.perf_counter()
        browser.attach()
        browser.session_create_ms = round((time.perf_counter() - started) * 1000)

    def _websocket(self):
        """The browser's CDP websocket: CHROME_CDP_URL itself (ws://...), or asked from Chrome's HTTP endpoint."""
        if urlsplit(self.url).scheme in {"ws", "wss"}:
            return self.url
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # a local port: never via a proxy
        try:
            with opener.open(self.url.rstrip("/") + "/json/version", timeout=10) as response:
                return json.load(response)["webSocketDebuggerUrl"]
        except Exception:
            raise BrowserUnavailable(
                f"No Chrome with remote debugging answers at {self.url}. Start Chrome with "
                "--remote-debugging-port=9222 and a separate --user-data-dir, or set CHROME_CDP_URL."
            ) from None

    def page(self, cdp):
        """A new tab in a browser context of its own. The context also goes away if this socket drops."""
        self.context = cdp.call("Target.createBrowserContext", disposeOnDetach=True)["browserContextId"]
        return cdp.call("Target.createTarget", url="about:blank", browserContextId=self.context)["targetId"]

    def endpoint(self):
        return {"browser_context_id": self.context}

    def release(self, browser):
        """Dispose of the run's context (its tabs close with it), then close the CDP socket."""
        if self.context and browser.cdp:
            try:
                browser.cdp.call("Target.disposeBrowserContext", browserContextId=self.context, timeout=10)
            except Exception:  # noqa: BLE001 - disposeOnDetach still removes it when the socket closes
                pass
        self.context = None
        browser.disconnect()

    def close(self, browser):
        self.release(browser)


class Browser:
    REMOTE = True  # every call may be a network round trip; the loop overlaps what it safely can

    def __init__(self, url):
        self.backend = browser_backend()
        self.provider = LexmountBrowser() if self.backend == "lexmount" else LocalChrome()
        self.connect_url = None  # the browser's CDP websocket (live frames and evaluation tools connect here too)
        self.inspect_url = ""  # a live view of the session, when the provider has one
        self.session_source = "cold" if self.backend == "lexmount" else "local"
        self.session_create_ms = 0
        self.session_attach_ms = 0
        self.session_pool_attempt_ms = 0
        self.session_pool_failure = None
        self.cdp = None
        self.target = None
        self.pending = None
        self.input_at = None
        try:
            self.provider.open(self)
            # Only connection setup may fall back. Never retry navigation or
            # later browser actions in a different session.
            self.call("Page.navigate", url=url)
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                try:
                    if self.evaluate("document.readyState") == "complete":
                        break
                except (StalePage, RuntimeError):
                    pass
                time.sleep(0.05)
        except BaseException:
            self.close()
            raise

    def attach(self):
        """Open the CDP socket to connect_url and take over the provider's page."""
        started = time.perf_counter()
        self.cdp = CDP(self.connect_url)
        self.target = self.provider.page(self.cdp)
        self.session = self.cdp.call("Target.attachToTarget", targetId=self.target, flatten=True)["sessionId"]
        self.call("Emulation.setDeviceMetricsOverride", width=1120, height=780, deviceScaleFactor=1, mobile=False)
        self.call("Emulation.setFocusEmulationEnabled", enabled=True)
        self.network = NetworkWatch(self.cdp, self.session)
        self.call("Network.enable")
        self.clock_offset = self.measure_clock_offset()
        mac = str(self.evaluate("navigator.platform") or "").startswith("Mac")
        self.select_all_modifiers = 4 if mac else 2
        self.session_attach_ms = round((time.perf_counter() - started) * 1000)

    def endpoint(self):
        """Where this run's page lives, for tools that open their own CDP connection (evidence capture, cleanup):
        {"backend": "lexmount" | "local", "cdp_url": the browser's CDP websocket, "target_id": the page the agent
        is on, "browser_context_id": the run's context (local, else None), "session_id": the Lexmount session
        (lexmount, else None)}."""
        info = {"backend": self.backend, "cdp_url": self.connect_url, "target_id": self.target,
                "browser_context_id": None, "session_id": None}
        info.update(self.provider.endpoint())
        return info

    def call(self, method, **params):
        return self.cdp.call(method, session_id=self.session, **params)

    def send(self, method, **params):
        return self.cdp.send(method, session_id=self.session, **params)

    @staticmethod
    def value(response):
        if response.get("exceptionDetails"):
            raise StalePage("Document changed during evaluation")
        return response.get("result", {}).get("value")

    def evaluate(self, expression, await_promise=False):
        try:
            return self.value(
                self.call("Runtime.evaluate", expression=expression, returnByValue=True, awaitPromise=await_promise)
            )
        except RuntimeError as error:
            if any(word in str(error) for word in NAVIGATING):
                raise StalePage("Document is navigating") from None
            raise

    def measure_clock_offset(self):
        """Browser wall clock minus local wall clock, in seconds, from the quickest of several exchanges."""
        best = None
        for _ in range(7):
            sent = time.time()
            remote = self.call("Runtime.evaluate", expression="Date.now()", returnByValue=True)["result"]["value"]
            received = time.time()
            if best is None or received - sent < best[0]:
                best = (received - sent, remote / 1000 - (sent + received) / 2)
        return best[1]

    def settled(self, info):
        """Was this observation taken after the page had its data and time to render it?"""
        if info is None:
            return False
        if not NET_CAP_MS:
            return True  # network gating switched off: upstream's timing
        if self.network.busy():
            return False
        return info["taken_at"] / 1000 >= self.network.last_finish + self.clock_offset + NET_GRACE_MS / 1000

    def finish(self, info, screenshot):
        if info is None:
            raise StalePage("Document is navigating")
        info["fingerprint"] = fingerprint(info)
        if screenshot:
            info["screenshot"] = self.call("Page.captureScreenshot", format="jpeg", quality=72)["data"]
        return info

    def observe(self, screenshot=True):
        pending, self.pending = self.pending, None
        info = None
        if pending is not None:
            # act() already sent settle + snapshot right behind the input. Read-only; a navigation may interrupt it.
            try:
                info = self.value(pending.result(30))
            except (StalePage, RuntimeError):
                info = None
        return self.settle(info, screenshot)

    def settle(self, info=None, screenshot=False):
        """An observation of the page at rest: no finite animation running, no same-origin request in flight,
        and two consecutive equal markers. Bounded by NET_CAP_MS."""
        # The budget runs from the input itself, not from when its first snapshot came back.
        started, self.input_at = self.input_at or time.time(), None
        deadline = started + NET_CAP_MS / 1000
        for attempt in range(10):
            if info is not None and (self.settled(info) or time.time() >= deadline):
                return self.finish(info, screenshot)
            # The page is still fetching, or the snapshot predates its last response: wait, then look again.
            self.network.wait_idle(max(0, (deadline - time.time()) * 1000))
            render = self.network.last_finish + NET_GRACE_MS / 1000 - time.time()
            if 0 < render and time.time() + render < deadline:
                time.sleep(render)
            try:
                late = time.time() >= deadline  # out of budget: one short look, then the model decides
                snapshot = f"{QUIET_FUNCTION}({QUIET_LAST_CAP_MS if late else QUIET_CAP_MS})" if QUIET_MS else SNAPSHOT
                info = self.evaluate(snapshot, await_promise=True)
                if info is None:
                    raise StalePage("Document is navigating")
            except StalePage:
                if attempt == 9:
                    raise
                time.sleep(0.02)
        if info is None:
            raise StalePage("Page did not settle")
        return self.finish(info, screenshot)

    def fresh_expression(self, page, action=None):
        scoped = action is not None and action["kind"] in {"click", "select"}
        node = action["node"] if scoped else None
        if scoped and type(node) is not int:
            return "false"
        arguments = [scoped, node, page["page_key_hash"], page["guard_hashes"].get(str(node)), page["marker_hash"]]
        return f"{FRESH}(...{json.dumps(arguments)})"

    def fresh(self, page, action=None):
        return self.evaluate(self.fresh_expression(page, action)) is True

    def fresh_later(self, page, action=None):
        """Start a freshness check without waiting, so it can overlap a model call."""
        future = self.send("Runtime.evaluate", expression=self.fresh_expression(page, action), returnByValue=True)
        return lambda: self.value(future.result(30)) is True

    def act(self, action, page, text=None):
        kind = action["kind"]
        if kind == "wait":
            # Upstream: freshness check, sleep 100 ms, then a plain observation. Here: one evaluation.
            result = self.evaluate(
                f"(async () => {{ if (!{self.fresh_expression(page, action)}) return {{fresh:false}};"
                f" await new Promise(r=>setTimeout(r,100)); return {{fresh:true,state:{SNAPSHOT}}}; }})()",
                await_promise=True,
            )
            if not result or not result["fresh"]:
                raise StalePage("Page changed since this decision. Observe again.")
            self.pending = Done({"result": {"value": result["state"]}})
            return {"executed": action["id"]}
        if kind != "scroll" and type(action["node"]) is not int:
            raise ValueError("Invalid observed node")
        # One round trip: upstream's freshness comparison, then upstream's target resolution, in the page.
        resolve = "null" if kind == "scroll" else f"{RESOLVE}({json.dumps(action)})"
        gate = f"(() => {{ if (!{self.fresh_expression(page, action)}) return {{fresh:false}};"
        gate += f" return {{fresh:true,target:{resolve}}}; }})()"
        try:
            result = self.evaluate(gate)
        except StalePage:
            if kind == "select":
                raise RuntimeError("Dropdown execution was interrupted; inspect before retrying.") from None
            raise
        if not result or not result["fresh"]:
            raise StalePage("Page changed since this decision. Observe again.")
        target = result["target"]
        sent = []
        if kind == "scroll":
            sent.append(self.send("Input.dispatchMouseEvent", type="mouseWheel", x=550, y=650, deltaX=0,
                                  deltaY=action["delta"]))
        elif target is None:
            if kind == "select":
                raise RuntimeError("Dropdown execution was not confirmed; inspect before retrying.")
            raise StalePage("Target changed or is covered. Observe again.")
        elif kind != "select":
            x, y = target["x"], target["y"]
            for event in ("mousePressed", "mouseReleased"):
                sent.append(self.send("Input.dispatchMouseEvent", type=event, x=x, y=y, button="left", clickCount=1))
            if kind == "fill":
                keys = dict(key="a", code="KeyA", modifiers=self.select_all_modifiers)
                sent.append(self.send("Input.dispatchKeyEvent", type="keyDown", commands=["selectAll"], **keys))
                sent.append(self.send("Input.dispatchKeyEvent", type="keyUp", **keys))
                sent.append(self.send("Input.insertText", text=text))
        # The next observation rides behind the input on the same session, so it runs after it.
        snapshot = QUIET_SNAPSHOT if QUIET_MS else SNAPSHOT
        self.input_at = time.time()
        self.pending = self.send(
            "Runtime.evaluate",
            expression=f"{SETTLE}({json.dumps(action)}).then(() => {snapshot})",
            awaitPromise=True,
            returnByValue=True,
        )
        for future in sent:
            future.result(30)  # An input error stops the run. Browser mutations are never retried.
        return {"executed": action["id"]}

    def disconnect(self):
        if self.cdp:
            self.cdp.close()
            self.cdp = None

    def close(self):
        self.provider.close(self)


class Done:
    """An already-resolved stand-in for a CDP future."""

    def __init__(self, value):
        self.value = value

    def result(self, _timeout=None):
        return self.value


def fingerprint(state):
    content = {k: state[k] for k in ("url", "text", "actions", "scroll")}
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
