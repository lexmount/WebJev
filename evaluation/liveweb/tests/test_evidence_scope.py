"""Evidence capture on a local Chrome shared by several runs must stay inside the run's own browser context; on a
cloud session (no context given) nothing is filtered."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from verifier.cdp import CdpEvalContext  # noqa: E402
from verifier.getters.navigation import get_open_tabs  # noqa: E402

TARGETS = [
    {"targetId": "user-tab", "type": "page", "url": "https://example.org/", "title": "other", "browserContextId": "DEFAULT"},
    {"targetId": "other-run", "type": "page", "url": "https://www.apple.com/", "title": "Apple", "browserContextId": "CTX-B"},
    {"targetId": "our-tab", "type": "page", "url": "https://www.wolframalpha.com/", "title": "Wolfram", "browserContextId": "CTX-A"},
]


class FakeTransport:
    def __init__(self):
        self.sent = []

    def send(self, method, params=None, *, session_id=None, timeout=30):
        self.sent.append((method, dict(params or {}), session_id))
        if method == "Target.getTargets":
            return {"targetInfos": [dict(t) for t in TARGETS]}
        if method == "Target.attachToTarget":
            return {"sessionId": "S-" + params["targetId"]}
        if method == "Target.createTarget":
            return {"targetId": "new-tab"}
        if method == "Runtime.evaluate":
            return {"result": {"type": "string", "value": "visible"}}
        return {}

    def close(self):
        pass


def _ctx(context_id):
    return CdpEvalContext(FakeTransport(), task={}, task_dir=ROOT, browser_context_id=context_id)


def test_scoped_context_picks_the_runs_page_and_hides_other_tabs():
    ctx = _ctx("CTX-A")
    assert ctx.target_id() == "our-tab"
    assert ctx.target_audit["rule"] == "only_page"
    assert get_open_tabs(ctx, {}) == [{"url": "https://www.wolframalpha.com/", "title": "Wolfram"}]


def test_scoped_context_opens_new_tabs_inside_the_context():
    ctx = _ctx("CTX-A")
    ctx.cdp("Target.createTarget", {"url": "https://www.wolframalpha.com/"})
    method, params, _ = ctx.transport.sent[-1]
    assert method == "Target.createTarget" and params["browserContextId"] == "CTX-A"


def test_unscoped_context_sees_the_whole_browser():
    ctx = _ctx(None)
    assert len(get_open_tabs(ctx, {})) == 3
    ctx.cdp("Target.createTarget", {"url": "about:blank"})
    assert "browserContextId" not in ctx.transport.sent[-1][1]
