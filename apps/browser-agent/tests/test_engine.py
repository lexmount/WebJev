"""Offline loop-guard regressions; no browser or model network calls."""

import importlib.util
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH = Path(__file__).resolve().parents[1] / "backend" / "engine.py"
SPEC = importlib.util.spec_from_file_location("browser_engine_under_test", PATH)
engine = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(engine)


def planner_model(monkeypatch, replies):
    monkeypatch.setenv("TEXT_MODEL", "test-model")
    monkeypatch.setenv("TEXT_MODEL_BASE_URL", "https://model.example")
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "test-key")
    monkeypatch.setattr(engine.socket, "getaddrinfo", lambda *a, **kw: [
        (2, 1, 6, "", ("93.184.216.34", 443))
    ])
    calls = []

    def post(*args):
        calls.append(json.loads(json.dumps(args[2])))
        reply = replies[len(calls) - 1]
        return {"choices": [{"message": {"content": json.dumps(reply)}}]}

    return SimpleNamespace(post_json=post), calls


def test_plan_recovers_from_logged_empty_website_response(monkeypatch):
    model, calls = planner_model(monkeypatch, [
        {"website": "", "runtime": "today is 2026-09-21"},
        {"website": "https://bookshop.org/", "stages": ["Search books on artificial intelligence and apply the user's filters"]},
    ])
    start, goals = engine.semantic_plan({"query": "Search bookshop.org for books on artificial intelligence"}, model, "today")
    assert start == "https://bookshop.org/" and len(goals) == 1
    assert len(calls) == 2
    assert "website" not in json.loads(calls[0]["messages"][1]["content"])
    assert calls[1]["messages"][-1]["role"] == "user"


def test_plan_retry_is_bounded_and_reports_planning_failure(monkeypatch):
    model, calls = planner_model(monkeypatch, [{"website": ""}, {"website": ""}])
    with pytest.raises(engine.TaskPlanningError, match="Task planning failed"):
        engine.semantic_plan({"query": "Search for books"}, model, "today")
    assert len(calls) == 2


def test_valid_plan_preserves_explicit_target_without_retry(monkeypatch):
    model, calls = planner_model(monkeypatch, [
        {"website": "https://other.example/", "stages": ["Search for books"]},
    ])
    start, _ = engine.semantic_plan(
        {"query": "Search for books", "target_website": "https://bookshop.org/"}, model, "today"
    )
    assert start == "https://bookshop.org/" and len(calls) == 1


def test_plan_does_not_weaken_private_address_validation(monkeypatch):
    model, calls = planner_model(monkeypatch, [
        {"website": "https://127.0.0.1/", "stages": ["Search for books"]},
    ])
    with pytest.raises(engine.TaskPlanningError, match="address is not valid"):
        engine.semantic_plan({"query": "Search for books"}, model, "today")
    assert len(calls) == 1


def setup_guard():
    state = {"execution_stage": 1, "history": [], "page": {}}
    guard = engine.ExecutionStagnation()
    assert guard.check(state) is None
    return guard, state


def test_login_handoff_opens_observed_entry_but_never_submits_form():
    from types import SimpleNamespace
    seen = []
    login = {"url": "https://accounts.example.com/user/login", "actions": []}
    browser = SimpleNamespace(observe=lambda **kwargs: login)
    # "\u53bb\u767b\u5f55" is a Chinese site's "Go to sign-in" entry button.
    entry = {"id": "e51", "kind": "click", "label": "\u53bb\u767b\u5f55"}
    page = {"actions": [entry]}
    assert engine.open_login_page(browser, page, lambda *args: seen.append(args)) is login
    assert seen == [(entry, page)]
    form = {"actions": [{"kind": "click", "label": "\u767b \u5f55"}, {"kind": "fill", "label": "\u53bb\u767b\u5f55"}]}
    assert engine.open_login_page(browser, form, lambda *args: seen.append(args)) is form
    assert len(seen) == 1


def test_login_loading_or_iframe_does_not_prematurely_resume_agent():
    assert engine.login_gate({"url": "https://accounts.example.com/user/login?backurl=x", "text": ""})
    assert not engine.login_gate({"url": "https://example.org/?next=/login", "text": "ordinary results"})


def test_login_popup_follows_owned_opener_and_returns_after_close():
    from types import SimpleNamespace
    pages = [{"type": "page", "targetId": "main"},
             {"type": "page", "targetId": "unrelated", "openerId": "other"},
             {"type": "page", "targetId": "login", "openerId": "main"}]
    browser = SimpleNamespace(cdp=SimpleNamespace(call=lambda method: {"targetInfos": pages}),
                              known={"main"}, owned={"main"}, parents=[], target="main")
    browser.switch = lambda target: setattr(browser, "target", target)
    engine.sync_login_target(browser)
    assert browser.target == "login" and browser.parents == ["main"]
    assert "unrelated" not in browser.owned
    pages.pop()
    engine.sync_login_target(browser)
    assert browser.target == "main" and browser.parents == []


def execute(guard, state, action="toggle", text="same", kind="click", url="https://example.org/"):
    state["history"].append({"kind": kind, "action": action, "text": None})
    state["page"] = {"url": url, "title": "Page", "text": text, "actions": [], "scroll": {"y": 0}}
    return guard.check(state)


def test_same_state_recovers_once_then_stops():
    guard, state = setup_guard()
    assert execute(guard, state) is None
    assert execute(guard, state) is None
    assert execute(guard, state)["outcome"] == "recover"
    assert guard.check(state) is None  # observation/decision is not another executed action
    assert execute(guard, state)["outcome"] == "stop"


def test_two_step_state_cycle_recovers_once_then_stops():
    guard, state = setup_guard()
    notices = []
    for i in range(7):
        phase = i % 2
        notice = execute(guard, state, f"toggle {phase}", f"state {phase}")
        if notice:
            notices.append(notice)
    assert [(n["step"], n["outcome"], n["period"]) for n in notices] == [
        (5, "recover", 2),
        (7, "stop", 2),
    ]


def test_normal_scroll_through_new_content_is_not_stagnation():
    guard, state = setup_guard()
    for i in range(30):
        assert execute(guard, state, "Scroll down", f"new visible row {i}", "scroll") is None


def test_same_control_on_different_pages_is_not_stagnation():
    guard, state = setup_guard()
    for i in range(12):
        assert execute(guard, state, url=f"https://example.org/item/{i}") is None


def test_legitimate_back_to_list_between_distinct_items_is_not_stagnation():
    guard, state = setup_guard()
    for i in range(8):
        assert execute(guard, state, f"item {i}", f"item content {i}") is None
        assert execute(guard, state, "Back", "list", "back") is None


def test_stage_transition_resets_prior_cycles():
    guard, state = setup_guard()
    for _ in range(3):
        execute(guard, state)
    state["execution_stage"] = 2
    assert guard.check(state) is None
    assert execute(guard, state) is None
    assert execute(guard, state) is None
    assert execute(guard, state)["outcome"] == "recover"


def test_timing_contains_only_fixed_metadata(tmp_path):
    writer = engine.EventWriter(tmp_path)
    with engine.measured(writer, "decision"):
        pass
    writer.close()
    data = json.loads((tmp_path / "timings.jsonl").read_text())
    assert set(data) == {"phase", "started_elapsed_ms", "duration_ms", "ok"}
    assert data["phase"] == "decision" and data["ok"] is True
    assert data["duration_ms"] >= 0
    assert (tmp_path / "timings.jsonl").stat().st_mode & 0o777 == 0o600


def test_guide_identical_evidence_uses_cache(tmp_path, monkeypatch):
    calls = []
    for key, value in {
        "TEXT_MODEL": "test",
        "TEXT_MODEL_BASE_URL": "https://example.org",
        "TEXT_MODEL_API_KEY": "fake",
    }.items():
        monkeypatch.setenv(key, value)

    class Model:
        def post_json(self, *args):
            calls.append(args)
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "guidance": "Read the fields that are still collapsed.",
                                    "can_continue": True,
                                    "stage_missing": True,
                                }
                            )
                        }
                    }
                ]
            }

    writer = engine.EventWriter(tmp_path)
    guide = engine.AdaptiveGuide(Model(), {"query": "Read the fields"}, writer, [])
    state = {
        "goal": "Read the fields",
        "history": [],
        "page": {
            "url": "https://example.org",
            "title": "Page",
            "text": "folded fields",
            "actions": [],
        },
    }
    assert guide.update(state, 1, "blocked") is True
    assert guide.update(state, 1, "blocked") is True
    assert len(calls) == 1
    writer.close()


CJK = re.compile(r"[\u4e00-\u9fff]")


def test_planner_prompts_are_english(monkeypatch):
    model, calls = planner_model(monkeypatch, [
        {"website": "", "runtime": "Today is 2026-09-21"},
        {"website": "https://www.example.org/", "stages": ["Read the page title"]},
    ])
    engine.semantic_plan({"query": "Find the page title on example.org"}, model, "Today is 2026-09-21")
    retry = calls[1]["messages"]
    assert not CJK.search(retry[0]["content"])  # system prompt
    assert not CJK.search(retry[-1]["content"])  # retry request


def test_guide_prompt_is_english(tmp_path, monkeypatch):
    for key, value in {"TEXT_MODEL": "test", "TEXT_MODEL_BASE_URL": "https://example.org",
                       "TEXT_MODEL_API_KEY": "fake"}.items():
        monkeypatch.setenv(key, value)
    bodies = []

    class Model:
        def post_json(self, url, key, body):
            bodies.append(body)
            return {"choices": [{"message": {"content": json.dumps(
                {"guidance": "Read the collapsed fields.", "can_continue": True, "stage_missing": True})}}]}

    writer = engine.EventWriter(tmp_path)
    guide = engine.AdaptiveGuide(Model(), {"query": "Read the fields"}, writer, [])
    state = {"goal": "Read the fields", "history": [],
             "page": {"url": "https://example.org", "title": "Page", "text": "folded fields", "actions": []}}
    guide.update(state, 1, "The selector believes it cannot continue")
    writer.close()
    assert not CJK.search(bodies[0]["messages"][0]["content"])


def run_with_ticks(tmp_path, monkeypatch, ticks, hooks=None):
    """run_live over a fake browser whose ticks raise or set the status as scripted."""
    spec = importlib.util.spec_from_file_location("adapter_under_test", engine.ADAPTER)
    adapter_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter_module)
    page = {"url": "https://example.org/", "title": "Example", "text": "Example page", "actions": [],
            "scroll": {"y": 0}}
    script, agents, events = list(ticks), [], hooks if hooks is not None else []

    class Browser:
        connect_url = "wss://browser.example/devtools"
        inspect_url = ""
        target = "target-1"

        def observe(self, screenshot=True):
            return dict(page)

        def endpoint(self):
            return {"backend": "local", "cdp_url": self.connect_url, "target_id": self.target,
                    "browser_context_id": "context-1", "session_id": None}

    class Agent:
        def __init__(self, url, goal):
            self.browser, self.pending_text, self.ticks, self.first_goal = Browser(), None, 0, goal
            self.state = {"goal": goal, "page": dict(page), "history": [], "status": "ready", "decision": None,
                          "plan": [goal], "plan_index": 0, "decisions": [], "text_calls": [], "elapsed_ms": 0}
            agents.append(self)

        def snapshot(self):
            return dict(self.state)

        def command(self, name):
            self.ticks += 1
            outcome = script.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            self.state["status"] = outcome
            return self.snapshot()

        def close(self):
            events.append("close")

    class Trace:
        def __init__(self, browser, clock):
            self.current, self.original_act = None, None
            self.input = lambda method, params: None

        def attach(self, history):
            pass

    class NoGuide:  # the completion review would call a model
        def __init__(self, *args):
            self.records = []

        def current(self, count):
            return ""

        def update(self, *args, **kwargs):
            return False

    class NoFrames:
        def __init__(self, *args):
            pass

        def target(self, target):
            pass

        def close(self):
            pass

    adapter = SimpleNamespace(UPSTREAM=PATH.parents[1] / "vendor", SemanticStages=adapter_module.SemanticStages,
                              ExecutionTrace=Trace, safe_dumps=adapter_module.safe_dumps,
                              choose_with_response_retry=adapter_module.choose_with_response_retry)
    agent_module = SimpleNamespace(Agent=Agent, choose=None, field_context=lambda *a: {}, StalePage=RuntimeError)
    monkeypatch.setattr(engine, "load_runtime", lambda writer=None: (adapter, agent_module, SimpleNamespace()))
    monkeypatch.setattr(engine, "semantic_plan", lambda request, model, context: ("https://example.org/",
                                                                                  ["Read the page"]))
    monkeypatch.setattr(engine, "AdaptiveGuide", NoGuide)
    monkeypatch.setattr(engine, "LiveFrames", NoFrames)
    monkeypatch.setitem(sys.modules, "result", SimpleNamespace(build_summary=lambda query, observations, status: {
        "status": "summarized", "summary": status, "columns": [], "rows": []}))
    writer = engine.EventWriter(tmp_path)
    engine.run_live({"id": "run", "query": "Read the page", "max_seconds": 60}, writer)
    writer.close()
    state = json.loads((tmp_path / "raw_state.json").read_text())
    diagnostics = tmp_path / "diagnostics.jsonl"
    contexts = [json.loads(line)["context"] for line in diagnostics.read_text().splitlines()] \
        if diagnostics.exists() else []
    return agents[0], state, contexts


def test_browser_timeout_observes_again_and_continues(tmp_path, monkeypatch):
    agent, state, contexts = run_with_ticks(tmp_path, monkeypatch, [TimeoutError(), "done"])
    assert agent.ticks == 2
    assert state["status"] == "done" and state["failure"] is None
    assert contexts == ["browser_timeout_retry"]


def test_browser_timeouts_stop_the_run_after_five_in_a_row(tmp_path, monkeypatch):
    agent, state, contexts = run_with_ticks(tmp_path, monkeypatch, [TimeoutError()] * 5)
    assert agent.ticks == 5
    assert state["status"] == "blocked" and state["failure"] == "TimeoutError"
    assert contexts == ["browser_timeout_retry"] * 4 + ["execution"]


def test_a_successful_step_resets_the_timeout_count(tmp_path, monkeypatch):
    ticks = [TimeoutError()] * 4 + ["ready"] + [TimeoutError()] * 4 + ["done"]
    agent, state, contexts = run_with_ticks(tmp_path, monkeypatch, ticks)
    assert state["status"] == "done" and state["failure"] is None
    assert contexts.count("browser_timeout_retry") == 8


def test_goal_sent_to_the_decision_model_is_english(tmp_path, monkeypatch):
    agent, state, _ = run_with_ticks(tmp_path, monkeypatch, ["done"])
    assert not CJK.search(agent.first_goal) and not CJK.search(state["goal"])


def test_browser_hooks_get_the_endpoint_and_closing_runs_before_close(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(engine, "browser_opened", lambda endpoint: events.append(("opened", endpoint)))
    monkeypatch.setattr(engine, "browser_closing", lambda endpoint: events.append(("closing", endpoint)))
    run_with_ticks(tmp_path, monkeypatch, ["done"], hooks=events)
    assert [e if isinstance(e, str) else e[0] for e in events] == ["opened", "closing", "close"]
    endpoint = events[1][1]
    assert endpoint["cdp_url"] == "wss://browser.example/devtools" and endpoint["target_id"] == "target-1"
    assert endpoint["browser_context_id"] == "context-1"


def test_a_failing_hook_is_logged_and_the_browser_still_closes(tmp_path, monkeypatch):
    events = []

    def broken(endpoint):
        raise RuntimeError("evidence capture failed")

    monkeypatch.setattr(engine, "browser_closing", broken)
    _, state, contexts = run_with_ticks(tmp_path, monkeypatch, ["done"], hooks=events)
    assert "browser_hook" in contexts and events == ["close"]
    assert state["status"] == "done"
