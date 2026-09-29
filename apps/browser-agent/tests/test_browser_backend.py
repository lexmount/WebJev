"""Browser backends: selection, the local-Chrome provider over a fake CDP, and Lexmount cleanup. No network."""

import importlib
import importlib.util
import io
import json
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from fastapi.testclient import TestClient

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
api = importlib.import_module("backend.app")
from backend.browser_backend import browser_backend  # noqa: E402

WS = "ws://127.0.0.1:9222/devtools/browser/test"


@pytest.fixture
def transport(monkeypatch):
    calls, sockets, clients = [], [], []

    class CDP:
        def __init__(self, url):
            self.url, self.handlers, self.closed = url, {}, False
            sockets.append(self)

        def call(self, method, session_id=None, timeout=30, **params):
            calls.append((method, params))
            if method == "Target.createBrowserContext":
                return {"browserContextId": "context-1"}
            if method == "Target.createTarget":
                return {"targetId": "tab-1"}
            if method == "Target.getTargets":  # the user's own tab must never be taken over
                return {"targetInfos": [{"type": "page", "targetId": "users-tab"}]}
            if method == "Target.attachToTarget":
                return {"sessionId": "cdp-1"}
            if method == "Runtime.evaluate":
                expression = params["expression"]
                value = time.time() * 1000 if expression == "Date.now()" else (
                    "complete" if expression == "document.readyState" else "Linux")
                return {"result": {"value": value}}
            return {}

        def close(self):
            self.closed = True

    class Client:
        def __init__(self, **kwargs):
            self.kwargs, self.deleted, self.closed = kwargs, [], False
            self.sessions = SimpleNamespace(create=self.create, delete=self.delete)
            clients.append(self)

        def create(self, **kwargs):
            raise TimeoutError("session_0f0e0d0c-0b0a-4908-8706-050403020100 did not become active in 90 s")

        def delete(self, session_id):
            self.deleted.append(session_id)

        def close(self):
            self.closed = True

    package = ModuleType("backend_browser_test")
    package.__path__ = [str(BASE / "vendor/jev_ultrafast")]
    monkeypatch.setitem(sys.modules, "backend_browser_test", package)
    monkeypatch.setitem(sys.modules, "backend_browser_test.cdp", SimpleNamespace(CDP=CDP))
    monkeypatch.setitem(sys.modules, "lexmount", SimpleNamespace(Lexmount=Client))
    spec = importlib.util.spec_from_file_location(
        "backend_browser_test.browser_cdp", BASE / "vendor/jev_ultrafast/browser_cdp.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return SimpleNamespace(module=module, calls=calls, sockets=sockets, clients=clients)


def discovery(monkeypatch, module, answer=None, error=None):
    asked = []

    class Opener:
        def open(self, url, timeout=None):
            asked.append(url)
            if error:
                raise error
            return io.BytesIO(json.dumps(answer or {"webSocketDebuggerUrl": WS}).encode())

    monkeypatch.setattr(module.urllib.request, "build_opener", lambda *handlers: Opener())
    return asked


@pytest.mark.parametrize("env, expected", [
    ({}, "local"),
    ({"LEXMOUNT_API_KEY": "k"}, "lexmount"),
    ({"LEXMOUNT_API_KEY": "k", "BROWSER": "local"}, "local"),
    ({"BROWSER": "Lexmount"}, "lexmount"),
    ({"BROWSER": "/usr/bin/firefox"}, "local"),
    ({"BROWSER": "firefox", "LEXMOUNT_API_KEY": "k"}, "lexmount"),
])
def test_backend_rule_is_the_same_in_the_app_and_the_transport(transport, monkeypatch, env, expected):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert browser_backend() == expected
    assert transport.module.browser_backend() == expected


def test_local_chrome_uses_its_own_context_and_disposes_of_it(transport, monkeypatch):
    monkeypatch.setenv("BROWSER", "local")
    asked = discovery(monkeypatch, transport.module)
    browser = transport.module.Browser("https://destination.example/")
    assert asked == ["http://127.0.0.1:9222/json/version"]
    assert browser.connect_url == WS and browser.target == "tab-1" and browser.session_source == "local"
    methods = [method for method, _ in transport.calls]
    assert ("Target.createBrowserContext", {"disposeOnDetach": True}) in transport.calls
    assert ("Target.createTarget", {"url": "about:blank", "browserContextId": "context-1"}) in transport.calls
    assert "Target.getTargets" not in methods  # never picks one of the user's tabs
    assert browser.endpoint() == {"backend": "local", "cdp_url": WS, "target_id": "tab-1",
                                  "browser_context_id": "context-1", "session_id": None}
    browser.close()
    browser.close()
    methods = [method for method, _ in transport.calls]
    assert methods.count("Target.disposeBrowserContext") == 1
    assert ("Target.disposeBrowserContext", {"browserContextId": "context-1"}) in transport.calls
    assert not {"Browser.close", "Target.closeTarget"} & set(methods)
    assert all(socket.closed for socket in transport.sockets)


def test_a_websocket_url_is_used_directly(transport, monkeypatch):
    monkeypatch.setenv("BROWSER", "local")
    monkeypatch.setenv("CHROME_CDP_URL", "ws://127.0.0.1:9333/devtools/browser/direct")
    asked = discovery(monkeypatch, transport.module)
    browser = transport.module.Browser("https://destination.example/")
    assert asked == [] and browser.connect_url == "ws://127.0.0.1:9333/devtools/browser/direct"
    browser.close()


def test_unreachable_chrome_is_a_clear_error(transport, monkeypatch):
    monkeypatch.setenv("BROWSER", "local")
    monkeypatch.setenv("CHROME_CDP_URL", "http://127.0.0.1:9555")
    discovery(monkeypatch, transport.module, error=OSError("connection refused"))
    with pytest.raises(transport.module.BrowserUnavailable, match="127.0.0.1:9555"):
        transport.module.Browser("https://destination.example/")
    assert transport.sockets == []


def test_lexmount_uses_the_com_api_and_deletes_a_session_left_by_a_failed_create(transport, monkeypatch):
    monkeypatch.setenv("BROWSER", "lexmount")
    with pytest.raises(TimeoutError):
        transport.module.Browser("https://destination.example/")
    client = transport.clients[0]
    assert client.kwargs["base_url"] == "https://api.lexmount.com" and "region" not in client.kwargs
    assert client.deleted == ["session_0f0e0d0c-0b0a-4908-8706-050403020100"]
    assert client.closed


def test_config_reports_the_browser_backend(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "DATA", tmp_path)
    with TestClient(api.app) as client:
        assert client.get("/api/config").json()["browser"] == {"id": "local", "label": "Local Chrome"}
        monkeypatch.setenv("LEXMOUNT_API_KEY", "k")
        assert client.get("/api/config").json()["browser"] == {"id": "lexmount", "label": "Lexmount Browser"}
        monkeypatch.setenv("BROWSER", "local")
        assert client.get("/api/config").json()["browser"]["id"] == "local"
