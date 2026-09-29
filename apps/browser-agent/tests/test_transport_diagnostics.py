import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.transport_diagnostics import TransportDiagnostics, request_hash


URL = "https://openrouter.ai/api/alpha/decisions"


def recorder(folder):
    return TransportDiagnostics(SimpleNamespace(folder=folder, elapsed=lambda: 12))


def test_trace_preserves_request_response_and_records_no_secrets(tmp_path):
    response = SimpleNamespace(status_code=200, http_version="HTTP/2")
    headers = {"Authorization": "Bearer sensitive-key"}
    content = json.dumps({"state": "private browser text"}, separators=(",", ":"))
    seen = []

    def send(url, **kwargs):
        assert url == URL and kwargs["content"] == content and kwargs["headers"] is headers
        trace = kwargs["extensions"]["trace"]
        trace("connection.connect_tcp.started", {"headers": headers})
        trace("connection.connect_tcp.complete", {"private": "private browser text"})
        return response

    client = SimpleNamespace(post=send)
    recorder(tmp_path).instrument_client(client, "primary")
    assert client.post(URL, content=content, headers=headers,
                       extensions={"trace": lambda name, info: seen.append(name)}) is response
    path = tmp_path / "model_transport.jsonl"
    entries = [json.loads(line) for line in path.read_text().splitlines()]
    assert [e["status"] for e in entries] == ["running", "returned"]
    assert entries[1]["http_status"] == 200 and entries[1]["duration_ms"] >= 0
    assert entries[1]["request_sha256"] == request_hash(content)
    assert len(seen) == 2 and len(entries[1]["events"]) == 2
    assert "sensitive-key" not in path.read_text() and "private browser text" not in path.read_text()
    assert path.stat().st_mode & 0o777 == 0o600


def test_failure_is_rethrown_without_logging_exception_text(tmp_path):
    error = TimeoutError("private endpoint and key")

    def fail(*args, **kwargs):
        raise error

    client = SimpleNamespace(post=fail)
    recorder(tmp_path).instrument_client(client, "backup")
    with pytest.raises(TimeoutError) as caught:
        client.post(URL, content=b"{}")
    assert caught.value is error
    path = tmp_path / "model_transport.jsonl"
    entry = json.loads(path.read_text().splitlines()[-1])
    assert entry["error"] == "TimeoutError" and entry["lane"] == "backup"
    assert "private endpoint" not in path.read_text()


def test_generation_calls_are_not_wrapped_or_logged(tmp_path):
    seen = []
    client = SimpleNamespace(post=lambda *args, **kwargs: seen.append(kwargs))
    recorder(tmp_path).instrument_client(client, "primary")
    client.post("https://example.org/chat/completions", content=b"{}")
    assert seen == [{"content": b"{}"}]
    assert not (tmp_path / "model_transport.jsonl").exists()


def test_dispatch_keeps_future_semantics_and_correlates_request(tmp_path):
    client = SimpleNamespace(post=lambda *args, **kwargs: None)
    backup = SimpleNamespace(post=lambda *args, **kwargs: None)
    body = {"model": "test", "state": "page"}
    with ThreadPoolExecutor(1) as pool:
        model = SimpleNamespace(CLIENT=client, BACKUP=backup, POOL=pool)
        recorder(tmp_path).install(model)
        marker = object()
        future = pool.submit(lambda *args: marker, URL, "secret", body, backup)
        assert future.result(timeout=1) is marker
    entry = json.loads((tmp_path / "model_transport.jsonl").read_text())
    assert entry["type"] == "dispatch" and entry["lane"] == "backup"
    assert entry["queue_ms"] >= 0 and entry["request_sha256"] == request_hash(body)


def test_unwritable_log_does_not_fail_the_model_call(tmp_path):
    (tmp_path / "model_transport.jsonl").mkdir()
    response = SimpleNamespace(status_code=200, http_version="HTTP/2")
    client = SimpleNamespace(post=lambda *args, **kwargs: response)
    recorder(tmp_path).instrument_client(client, "primary")
    assert client.post(URL, content=b"{}") is response
