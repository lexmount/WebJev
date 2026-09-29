import importlib
import json
import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
api = importlib.import_module("backend.app")


@pytest.fixture
def clients(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "DATA", tmp_path)
    api.runs.clear()
    api.submission_times.clear()
    async def fake_start(self):
        self.folder.mkdir(exist_ok=True)
        await self.publish("phase", {"phase": "running"})
    monkeypatch.setattr(api.Run, "start", fake_start)
    with TestClient(api.app) as one, TestClient(api.app) as two:
        one.get("/api/presets")
        two.get("/api/presets")
        yield one, two


def create(client, **overrides):
    body = {"query": api.PRESETS[0]["query"], "preset_id": api.PRESETS[0]["id"]}
    return client.post("/api/runs", json={**body, **overrides})


def test_owner_isolation_for_every_run_resource(clients):
    one, two = clients
    response = create(one)
    assert response.status_code == 201
    runid = response.json()["id"]
    assert one.get(f"/api/runs/{runid}").status_code == 200
    for suffix in ("", "/result", "/updates", "/events", "/diagnostics", "/frames/example.jpg"):
        assert two.get(f"/api/runs/{runid}{suffix}").status_code == 404
    for suffix in ("/cancel", "/resume"):
        assert two.post(f"/api/runs/{runid}{suffix}").status_code == 404
    assert two.post(f"/api/runs/{runid}/input", json={"type": "click", "x": 20, "y": 20}).status_code == 404


def test_private_target_and_cross_origin_rejected(clients):
    one, _ = clients
    for host in ("127.0.0.1", "localhost", "[::1]", "http://10.0.0.1", "a.local", "https://public.com:444", "file:///etc/passwd", "https://name:password@public.com"):
        assert create(one, target_website=host).status_code == 422
    assert one.post("/api/runs", json={"query": "Find today's weather in Boston"}, headers={"Origin": "https://attacker.test"}).status_code == 403


def test_edited_preset_becomes_custom_and_same_owner_can_start_another_task(clients):
    one, _ = clients
    response = create(one, query="Search amtrak.com for trains from Boston to New York tomorrow")
    runid = response.json()["id"]
    assert response.json()["preset_id"] is None
    assert api.runs[runid].payload["target_website"] is None
    assert create(one).status_code == 201


def test_missing_or_forged_cookie_cannot_read_run(clients):
    one, two = clients
    runid = create(one).json()["id"]
    token = one.cookies.get(api.COOKIE)
    two.cookies.set(api.COOKIE, token[:-1]+("0" if token[-1] != "0" else "1"))
    assert two.get(f"/api/runs/{runid}").status_code == 404


def test_summarized_run_is_terminal_and_allows_next_task(clients):
    one, _ = clients
    runid = create(one).json()["id"]
    api.runs[runid].status = "summarized"
    assert "summarized" in api.TERMINAL
    assert create(one).status_code == 201


def test_diagnostics_refreshes_pending_call_when_run_ends(clients):
    one, _ = clients
    identifier = create(one).json()["id"]
    run = api.runs[identifier]
    (run.folder / "model_calls.jsonl").write_text(json.dumps({
        "id": "call-1", "kind": "generation", "status": "running", "started_elapsed_ms": 20, "duration_ms": None,
    }) + "\n")
    first = one.get(f"/api/runs/{identifier}/diagnostics").json()
    assert first["calls"][0]["status"] == "running"
    run.status = "cancelled"
    second = one.get(f"/api/runs/{identifier}/diagnostics", params={"since": first["revision"]}).json()
    assert second["calls"][0]["status"] == "interrupted"
    assert second["calls"][0]["duration_ms"] is None


def test_updates_keep_actions_but_only_latest_frame(clients):
    one, _ = clients
    runid = create(one).json()["id"]
    run = api.runs[runid]
    run.events.extend([
        {"id": 2, "type": "frame", "data": {"url": "old.jpg"}},
        {"id": 3, "type": "step", "data": {"label": "Open the search page"}},
        {"id": 4, "type": "frame", "data": {"url": "new.jpg"}},
    ])
    run.sequence = 4
    response = one.get(f"/api/runs/{runid}/updates?after=1").json()
    assert [e["id"] for e in response["events"]] == [3,4]


def test_worker_error_log_never_exposed(clients):
    one, _ = clients
    runid = create(one).json()["id"]
    (api.runs[runid].folder / "worker.log").write_text("API_SECRET=not-for-web")
    assert one.get(f"/api/runs/{runid}/worker.log").status_code == 404
    assert one.get(f"/api/runs/{runid}/frames/worker.log").status_code == 404


def test_input_only_when_paused_and_never_writes_password(clients):
    one, _ = clients
    runid = create(one).json()["id"]
    run = api.runs[runid]
    assert one.post(f"/api/runs/{runid}/input", json={"type": "text", "text": "secret"}).status_code == 409
    assert one.post(f"/api/runs/{runid}/input", json={"type": "Runtime.evaluate", "text": "alert(1)"}).status_code == 422
    run.status = "needs_login"
    (run.folder / "browser_control.json").write_text("{}")
    received = []
    class Controller:
        def send(self, body):
            received.append(body)
    run.input_controller = Controller()
    assert one.post(f"/api/runs/{runid}/input", json={"type": "text", "text": "secret"}).status_code == 200
    assert received[0]["text"] == "secret"
    assert "secret" not in json.dumps(run.snapshot())
    assert "secret" not in "".join(p.read_text() for p in run.folder.glob("*.json"))
