"""The browser layer the engine loads around the vendored jev-ultrafast runtime (engine.ADAPTER).

It adds browser Back and Enter controls, follows popup tabs, clicks through open shadow roots, and gives the
decision model extra element context (state classes, nearby text, row context, input constraints). There are no
task-specific action scripts or prepared field values.
"""

import copy
import json
import os
import re
import time
from pathlib import Path

# The vendored runtime (apps/browser-agent/vendor/jev_ultrafast); engine.load_runtime puts it on sys.path.
UPSTREAM = Path(__file__).resolve().parents[1] / "vendor"
# A focused text field, looking inside open shadow roots (document.activeElement stops at the host).
FOCUSED_TEXT_FIELD = (
    "(() => { let a=document.activeElement; while (a?.shadowRoot?.activeElement) a=a.shadowRoot.activeElement;"
    " return ['INPUT','TEXTAREA'].includes(a?.tagName); })()"
)


# DOM substring boundaries can split UTF-16 emoji; keep logs valid UTF-8.
def safe_dumps(value, **kwargs):
    return json.dumps(value, **kwargs).encode("utf-8", errors="backslashreplace").decode("utf-8")


def choose_with_response_retry(choose, page, goal, history):
    """Retry only a rejected model response before any browser action exists."""
    for attempt in range(3):
        try:
            return choose(page, goal, history)
        except ValueError as exc:
            if str(exc) != "Invalid TypeSafe response; no action executed." or attempt == 2:
                raise
            print(safe_dumps({"invalid_response_retry": attempt + 1}), flush=True)


class ExecutionTrace:
    """Passively retain observed geometry and the exact existing CDP input calls."""

    def __init__(self, browser, clock):
        self.clock = clock
        self.completed = []
        self.current = None
        self.original_act = browser.act
        self.original_evaluate = browser.evaluate
        self.original_call = browser.call
        self.original_send = browser.send
        browser.act = self.act
        browser.evaluate = self.evaluate
        browser.call = self.call
        browser.send = self.send

    def input(self, method, params):
        if self.current is not None and method.startswith("Input."):
            fields = ("type", "x", "y", "button", "clickCount", "deltaX", "deltaY", "key", "code")
            self.current["input_events"].append(
                {
                    "at_ms": self.clock(),
                    "method": method,
                    **{key: params[key] for key in fields if key in params},
                }
            )

    def call(self, method, **params):
        self.input(method, params)
        return self.original_call(method, **params)

    def send(self, method, **params):
        self.input(method, params)
        return self.original_send(method, **params)

    def evaluate(self, expression, **kwargs):
        result = self.original_evaluate(expression, **kwargs)
        if self.current is not None and isinstance(result, dict):
            target = result.get("target") if result.get("fresh") else None
            if isinstance(target, dict) and "x" in target and "y" in target:
                self.current["resolved_target"] = copy.deepcopy(target)
                self.current["resolved_at_ms"] = self.clock()
        return result

    def act(self, action, page, text=None):
        page_key = page.get("page_key", [])
        self.current = {
            "started_ms": self.clock(),
            "coordinate_space": "viewport_css_pixels",
            "action": copy.deepcopy(action),
            "before": {
                "url": page.get("url"),
                "title": page.get("title"),
                "taken_at": page.get("taken_at"),
                "page_key_hash": page.get("page_key_hash"),
                "viewport": {"width": page_key[4], "height": page_key[5]}
                if len(page_key) >= 6
                else None,
                "scroll": {"x": page_key[2], "y": page_key[3]}
                if len(page_key) >= 4
                else copy.deepcopy(page.get("scroll")),
            },
            "input_events": [],
        }
        try:
            result = self.original_act(action, page, text)
            self.current["ended_ms"] = self.clock()
            self.completed.append(self.current)
            return result
        finally:
            self.current = None

    def attach(self, history):
        for entry, trace in zip(history, self.completed, strict=False):
            if entry.get("choice") == trace["action"]["id"]:
                entry["execution"] = trace


class SemanticStages:
    """Change only the semantic goal between native Jev runs on one live agent."""

    CONSTRAINTS = (
        "\nComplete only the current stage and judge DONE from the currently visible evidence;"
        " do not carry out later tasks early. Keep conditions already in effect and do not repeat"
        " completed operations. Information retrieval only: no login, no purchase."
    )  # engine.run_live replaces this with the run's own constraints, including today's date.

    def __init__(self, agent, goals):
        self.agent, self.goals = agent, goals
        self.index = 0
        self.history = []
        agent.state["stage_history"] = self.history
        agent.state["plan"] = [goal + self.CONSTRAINTS for goal in goals]
        self.start()

    def start(self):
        state = self.agent.state
        state.update(
            goal=state["plan"][self.index],
            plan_index=self.index,
            execution_stage=self.index + 1,
            execution_stage_count=len(self.goals),
            status="ready",
            decision=None,
        )
        self.agent.pending_text = None
        self.history.append(
            {
                "stage": self.index + 1,
                "goal": self.goals[self.index],
                "execution_goal": state["goal"],
                "started_elapsed_ms": state["elapsed_ms"],
                "ended_elapsed_ms": None,
                "native_status": None,
            }
        )

    def finish(self, native_status, reason=None):
        current = self.history[-1]
        if current["ended_elapsed_ms"] is None:
            current.update(
                ended_elapsed_ms=self.agent.state["elapsed_ms"],
                native_status=native_status,
            )
            if reason:
                current["stop_reason"] = reason

    def advance(self):
        state = self.agent.state
        if state["status"] not in {"done", "blocked"}:
            return
        self.finish(state["status"])
        if state["status"] == "done":
            state["plan_index"] = self.index + 1
            if self.index + 1 < len(self.goals):
                self.index += 1
                self.start()


def install_adapter():
    from jev_ultrafast import agent as agent_module
    from jev_ultrafast import browser_cdp as browser_module
    from jev_ultrafast import model
    from jev_ultrafast.browser_cdp import Browser, NetworkWatch, StalePage, fingerprint

    # Match the generic policy's control guidance to this adapter's real input
    # capabilities. Keep all target and value choices with Jev.
    model.NEXT_ACTION = (
        model.NEXT_ACTION.replace(
            "For date pickers, CLICK the field, date, then confirmation.",
            "For editable date fields with declared format metadata, "
            "prefer TYPE_TEXT in that format. "
            "For readonly date pickers, CLICK the field, date, then confirmation.",
        )
        .replace(
            "Set every requested filter/control; a matching result alone does not prove "
            "a requested filter was set.",
            "Set every available requested filter/control; a matching result alone does not prove "
            "a filter was set. If no exact filter exists, inspect the sorted results in order, "
            "verify the requested attributes visibly for each result, exclude nonmatches, "
            "and collect the requested number of qualifying results.",
        )
        .replace(
            "If Search/Submit is visible and the required fields are ready, CLICK it immediately.",
            "A form is ready only after its text, category selection, matching mode, expansions "
            "and available filters agree with the goal. "
            "Nonempty defaults are not proof of readiness. "
            "For quoted technical phrases, use exact matching rather than fuzzy tokenization; "
            "disable expansions that broaden the requested topic. Review these existing controls "
            "before submitting. Then CLICK Search/Submit. Do not replace an already applied search "
            "with a broader one or paginate past unchecked higher-ranked candidates.",
        )
    )
    model.NEXT_ACTION += (
        "\nPreserve an applied interval: do not narrow a requested multi-year range "
        "by selecting a single-year facet. Once start and end dates have been entered "
        "and submitted, keep that inclusive range while sorting and reading results. "
        "Read results in rank order from the first row, including their visible metadata, "
        "before considering another results page. On a detail page, read the item's content "
        "and scroll for missing metadata, then go back to the results. Do not use the "
        "sitewide search box or its field selector to filter an already opened detail page."
    )

    # Hit-test through open shadow roots with the snapshot's own helpers (snapshot.js cache.deepAt and
    # cache.within), so a control inside a consent dialog or a web component can be clicked.
    hit = (
        "(c=>c?.within ? c.within(e,c.deepAt({x},{y})) : "
        "e.contains(document.elementFromPoint({x},{y})))(window.__jevFast)"
    )
    for old, new in (
        (
            "const r=e.getBoundingClientRect(), x=r.x+r.width/2, y=r.y+r.height/2;",
            "const r=e.getBoundingClientRect(); "
            "const point=[.5,.25,.75,.1,.9].map(f=>({x:r.x+r.width*f,y:r.y+r.height*.5}))"
            f".find(p=>p.x>=0&&p.y>=0&&p.x<innerWidth&&p.y<innerHeight&&{hit.format(x='p.x', y='p.y')});"
            "if (!point) return null; const {x,y}=point;",
        ),
        (
            "if (!e.contains(document.elementFromPoint(x,y))) return null;",
            f"if (!{hit.format(x='x', y='y')}) return null;",
        ),
    ):
        if old not in browser_module.RESOLVE:
            raise RuntimeError("jev_ultrafast RESOLVE changed; the click-point patch no longer applies")
        browser_module.RESOLVE = browser_module.RESOLVE.replace(old, new)

    original_space = model.action_space

    def contextual_space(actions):
        elements, targets, controls = original_space(actions)
        for group in targets.values():
            for index, action in group.items():
                element = elements[int(index.split(":")[0]) - 1]
                flags = re.findall(
                    r"[\w-]*(?:active|selected|checked|chosen|current|disabled)[\w-]*|\bcur\b",
                    action.get("dom_state", ""),
                    flags=re.I,
                )
                if flags:
                    element["state_classes"] = " ".join(flags)[:60]
                nearby = " ".join(action.get("nearby_text", "").split())[:70]
                if nearby and nearby != " ".join(action["label"].split()):
                    element["context"] = nearby
                if action["kind"] == "fill":
                    element["field_context"] = action.get("parent_class", "")[:140]
                for key in ("row_context", "input_constraints"):
                    if action.get(key):
                        element[key] = action[key]
                if action.get("visited"):
                    element["already_visited"] = True
                if action.get("position"):
                    element["position"] = action["position"]
        return elements, targets, controls

    model.action_space = agent_module.action_space = contextual_space
    original_field_context = model.field_context

    def contextual_field_context(goal, action, page, history):
        context = original_field_context(goal, action, page, history)
        context["field"].update(
            {
                key: action[key]
                for key in ("parent_class", "row_context", "input_constraints")
                if key in action
            }
        )
        return context

    model.field_context = agent_module.field_context = contextual_field_context

    class MultiPageBrowser(Browser):
        def __init__(self, url):
            super().__init__(url)
            self.call(
                "Emulation.setDeviceMetricsOverride",
                width=int(os.environ.get("JEV_VIEWPORT_WIDTH", "1120")),
                height=780,
                deviceScaleFactor=1,
                mobile=False,
            )
            self.parents = []
            self.owned = {self.target}
            self.known = {p["targetId"] for p in self.cdp.call("Target.getTargets")["targetInfos"]}
            self.on_target = None

        def switch(self, target):
            self.pending = None
            self.target = target
            self.session = self.cdp.call("Target.attachToTarget", targetId=target, flatten=True)[
                "sessionId"
            ]
            self.call(
                "Emulation.setDeviceMetricsOverride",
                width=int(os.environ.get("JEV_VIEWPORT_WIDTH", "1120")),
                height=780,
                deviceScaleFactor=1,
                mobile=False,
            )
            self.call("Emulation.setFocusEmulationEnabled", enabled=True)
            self.network = NetworkWatch(self.cdp, self.session)
            self.call("Network.enable")
            if self.on_target:
                self.on_target(target)

        def observe(self, screenshot=True):
            if hasattr(self, "known"):
                deadline = time.monotonic() + (2 if getattr(self, "expected_popup", False) else 0)
                while True:
                    pages = self.cdp.call("Target.getTargets")["targetInfos"]
                    fresh = [
                        p
                        for p in pages
                        if p["type"] == "page"
                        and p["targetId"] not in self.known
                        and p.get("openerId") in self.owned
                    ]
                    if fresh or time.monotonic() >= deadline:
                        break
                    time.sleep(0.04)
                self.expected_popup = False
                if fresh:
                    self.known.update(p["targetId"] for p in fresh)
                    self.owned.update(p["targetId"] for p in fresh)
                    self.parents.append(self.target)
                    self.switch(fresh[-1]["targetId"])
                    deadline = time.monotonic() + 4
                    while time.monotonic() < deadline:
                        try:
                            if self.evaluate(
                                "location.href !== 'about:blank' && "
                                "document.readyState !== 'loading'"
                            ):
                                break
                        except (RuntimeError, StalePage):
                            pass
                        time.sleep(0.05)
            page = super().observe(screenshot)
            if self.evaluate(FOCUSED_TEXT_FIELD):
                page["actions"].append(
                    {
                        "id": "press_enter",
                        "kind": "enter",
                        "label": "Press Enter to submit the focused search or input field",
                    }
                )
            nav = self.call("Page.getNavigationHistory")
            if any(
                e["url"] != "about:blank" for e in nav["entries"][: nav["currentIndex"]]
            ) or getattr(self, "parents", []):
                page["actions"].append(
                    {"id": "go_back", "kind": "back", "label": "Go back to the previous page to see other results"}
                )
            page["fingerprint"] = fingerprint(page)
            return page

        def act(self, action, page, text=None):
            if action["kind"] == "scroll_panel":
                gate = self.fresh_expression(page, {**action, "kind": "click"})
                target = self.evaluate(
                    f"(() => {{if (!{gate}) return null; "
                    f"return {browser_module.RESOLVE}({safe_dumps(action)}); }})()"
                )
                if target is None:
                    raise StalePage("Scroll region changed or is covered")
                self.call(
                    "Input.dispatchMouseEvent",
                    type="mouseWheel",
                    x=target["x"],
                    y=target["y"],
                    deltaX=0,
                    deltaY=action["delta"],
                )
                self.pending = None
                self.input_at = time.time()
                return {"executed": action["id"]}
            if action["kind"] not in {"enter", "back"}:
                try:
                    self.expected_popup = bool(action.get("opens_new_tab"))
                    return super().act(action, page, text)
                except StalePage as exc:
                    print(
                        safe_dumps(
                            {"stale_target": action["label"], "reason": str(exc)},
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    raise
            if not self.fresh(page):
                raise StalePage("Page changed before browser control")
            if action["kind"] == "enter":
                focused = self.evaluate(FOCUSED_TEXT_FIELD)
                if not focused:
                    raise StalePage("Enter requires a focused text input")
                for kind in ("keyDown", "keyUp"):
                    self.call(
                        "Input.dispatchKeyEvent",
                        type=kind,
                        key="Enter",
                        code="Enter",
                        windowsVirtualKeyCode=13,
                    )
            else:
                nav = self.call("Page.getNavigationHistory")
                previous = [
                    e for e in nav["entries"][: nav["currentIndex"]] if e["url"] != "about:blank"
                ]
                if previous:
                    self.call("Page.navigateToHistoryEntry", entryId=previous[-1]["id"])
                elif self.parents:
                    previous_target = self.target
                    self.switch(self.parents.pop())
                    self.cdp.call("Target.closeTarget", targetId=previous_target)
            self.pending = None
            self.input_at = time.time()
            return {"executed": action["id"]}

        def fresh_expression(self, page, action=None):
            if action is None and getattr(self, "terminal_readonly", False):
                # A staged DONE/BLOCKED only ends a semantic subtask; it sends no
                # browser input and does not certify the final result. Preserve
                # document, URL, viewport, scroll and every form value/state.
                # Passive tickers changing the full marker must not prevent it.
                return (
                    f"(() => {{ const c=window.__jevFast; const H={browser_module.HASH}; "
                    "return !!c && H(JSON.stringify(c.pageKey()))==="
                    f"{json.dumps(page['page_key_hash'])}; }})()"
                )
            if action is not None and action["kind"] == "fill":
                # Preserve document/form/target/nearby checks; unrelated rotating
                # adverts must not invalidate a generated value for this field.
                action = {**action, "kind": "click"}
            return super().fresh_expression(page, action)

        def fresh_later(self, page, action=None):
            return super().fresh_later(page, action or getattr(self, "decision_action", None))

    agent_module.Browser = MultiPageBrowser
