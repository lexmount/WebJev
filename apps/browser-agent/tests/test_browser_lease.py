"""Private warm-session handoff tests; no SDK requests or browser connections."""

import importlib.util
import json
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

BASE = Path(__file__).resolve().parents[1]


@pytest.fixture
def transport(monkeypatch):
    sessions, clients, sockets = [], [], []
    rejected = set()
    calls = []

    class Session:
        def __init__(self, id, ws, client, inspect_url="", created_at="", **kwargs):
            self.id, self.connect_url, self._client = id, ws, client
            self.inspect_url, self.created_at = inspect_url, created_at
            self.closed = False
            sessions.append(self)

        def close(self):
            if not self.closed:
                self._client.sessions.delete(session_id=self.id)
                self.closed = True

    class Client:
        def __init__(self, **kwargs):
            self.created, self.deleted, self.closed = [], [], False
            self.sessions = SimpleNamespace(create=self.create, delete=self.delete)
            clients.append(self)

        def create(self, **kwargs):
            self.created.append(kwargs)
            return Session("cold-1", "wss://cloud.example/cold", self)

        def delete(self, session_id):
            self.deleted.append(session_id)

        def close(self):
            self.closed = True

    class CDP:
        def __init__(self, url):
            if url in rejected:
                raise RuntimeError("private session connection rejected")
            self.url, self.handlers, self.closed = url, {}, False
            sockets.append(self)

        def call(self, method, **params):
            calls.append((method, params))
            if method == "Target.getTargets":
                return {"targetInfos": [{"type": "page", "targetId": "page-1"}]}
            if method == "Target.attachToTarget":
                return {"sessionId": "cdp-1"}
            if method == "Runtime.evaluate":
                expression = params["expression"]
                value = (
                    time.time() * 1000
                    if expression == "Date.now()"
                    else "complete"
                    if expression == "document.readyState"
                    else "Linux"
                )
                return {"result": {"value": value}}
            return {}

        def close(self):
            self.closed = True

    package = ModuleType("lease_browser_test")
    package.__path__ = [str(BASE / "vendor/jev_ultrafast")]
    monkeypatch.setitem(sys.modules, "lease_browser_test", package)
    monkeypatch.setitem(
        sys.modules, "lexmount", SimpleNamespace(Lexmount=Client, SessionInfo=Session)
    )
    monkeypatch.setitem(sys.modules, "lease_browser_test.cdp", SimpleNamespace(CDP=CDP))
    spec = importlib.util.spec_from_file_location(
        "lease_browser_test.browser_cdp", BASE / "vendor/jev_ultrafast/browser_cdp.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.delenv("BROWSER_AGENT_SESSION_LEASE_FILE", raising=False)
    monkeypatch.setenv("BROWSER", "lexmount")
    return SimpleNamespace(
        module=module,
        clients=clients,
        sessions=sessions,
        sockets=sockets,
        rejected=rejected,
        calls=calls,
    )


def lease(tmp_path, monkeypatch):
    path = tmp_path / "session_lease.json"
    path.write_text(
        json.dumps(
            {
                "id": "warm-1",
                "connect_url": "wss://cloud.example/warm?token=private",
                "inspect_url": "https://cloud.example/view",
                "created_at": "2026-09-20T00:00:00Z",
            }
        )
    )
    path.chmod(0o600)
    monkeypatch.setenv("BROWSER_AGENT_SESSION_LEASE_FILE", str(path))
    return path


def test_warm_lease_consumed_once_without_create_and_closed(tmp_path, monkeypatch, transport):
    path = lease(tmp_path, monkeypatch)
    browser = transport.module.Browser("https://destination.example/")
    assert browser.session_source == "pool"
    assert browser.session_create_ms == 0
    assert browser.session_attach_ms >= 0
    assert transport.clients[0].created == []
    assert not path.exists()
    assert "BROWSER_AGENT_SESSION_LEASE_FILE" not in transport.module.os.environ
    assert [p["url"] for method, p in transport.calls if method == "Page.navigate"] == [
        "https://destination.example/"
    ]
    browser.close()
    browser.close()
    assert transport.clients[0].deleted == ["warm-1"]
    assert transport.clients[0].closed
    assert all(socket.closed for socket in transport.sockets)


def test_expired_connection_falls_back_once_and_closes_leased_session(
    tmp_path, monkeypatch, transport
):
    path = lease(tmp_path, monkeypatch)
    transport.rejected.add("wss://cloud.example/warm?token=private")
    browser = transport.module.Browser("https://destination.example/")
    assert browser.session_source == "cold_fallback"
    assert browser.session_pool_failure == "pool_attach_failed"
    assert transport.clients[0].created == [{"poll_timeout_sec": 90}]
    assert transport.clients[0].deleted == ["warm-1"]
    assert not path.exists()
    browser.close()
    assert transport.clients[0].deleted == ["warm-1", "cold-1"]


@pytest.mark.parametrize("invalid", ["permissions", "json", "endpoint"])
def test_invalid_private_lease_is_not_reused(tmp_path, monkeypatch, transport, invalid):
    path = lease(tmp_path, monkeypatch)
    if invalid == "permissions":
        path.chmod(0o644)
    elif invalid == "json":
        path.write_text("invalid-json")
    else:
        data = json.loads(path.read_text())
        data["connect_url"] = "https://not-a-cdp.example/"
        path.write_text(json.dumps(data))
    browser = transport.module.Browser("https://destination.example/")
    assert browser.session_source == "cold_fallback"
    assert browser.session_pool_failure == "lease_invalid"
    assert len(transport.clients[0].created) == 1
    assert not path.exists()
    browser.close()


def test_cold_without_lease_creates_exactly_once(transport):
    browser = transport.module.Browser("https://destination.example/")
    assert browser.session_source == "cold"
    assert browser.session_pool_failure is None
    assert len(transport.clients[0].created) == 1
    browser.close()


def test_failed_cold_fallback_cannot_loop_or_leave_client_open(tmp_path, monkeypatch, transport):
    lease(tmp_path, monkeypatch)
    transport.rejected.update(
        {"wss://cloud.example/warm?token=private", "wss://cloud.example/cold"}
    )
    with pytest.raises(RuntimeError):
        transport.module.Browser("https://destination.example/")
    assert len(transport.clients[0].created) == 1
    assert transport.clients[0].deleted == ["warm-1", "cold-1"]
    assert transport.clients[0].closed


def test_navigation_failure_never_replays_in_a_fresh_session(tmp_path, monkeypatch, transport):
    lease(tmp_path, monkeypatch)
    original = transport.module.Browser.call

    def call(self, method, **params):
        if method == "Page.navigate":
            raise RuntimeError("navigation failed")
        return original(self, method, **params)

    monkeypatch.setattr(transport.module.Browser, "call", call)
    with pytest.raises(RuntimeError, match="navigation failed"):
        transport.module.Browser("https://destination.example/")
    assert transport.clients[0].created == []
    assert transport.clients[0].deleted == ["warm-1"]


def test_engine_session_timing_never_serializes_credentials(tmp_path):
    spec = importlib.util.spec_from_file_location("lease_engine_test", BASE / "backend/engine.py")
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    browser = SimpleNamespace(
        session_source="pool",
        session_create_ms=0,
        session_attach_ms=123,
        session_pool_attempt_ms=124,
        session_pool_failure=None,
        connect_url="wss://secret/?key=private",
    )
    data = engine.session_timing(browser)
    assert data == {
        "session_source": "pool",
        "session_pool_failure": None,
        "session_create_ms": 0,
        "session_attach_ms": 123,
        "session_pool_attempt_ms": 124,
    }
    browser.session_source = "private-key"
    browser.session_pool_failure = "private-url"
    browser.session_create_ms = "private-key"
    safe = engine.session_timing(browser)
    assert "private" not in json.dumps(safe)
    engine.atomic_json(tmp_path / "session_timing.json", data)
    assert (tmp_path / "session_timing.json").stat().st_mode & 0o777 == 0o600
