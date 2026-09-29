"""Decision-model selection: server configuration, run requests and decision routing (offline)."""

import importlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
api = importlib.import_module("backend.app")
SPEC = importlib.util.spec_from_file_location("decision_engine_under_test", ROOT / "backend" / "engine.py")
engine = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(engine)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "DATA", tmp_path)
    api.runs.clear()
    api.submission_times.clear()

    async def fake_start(self):
        self.folder.mkdir(exist_ok=True)
        await self.publish("phase", {"phase": "running"})

    monkeypatch.setattr(api.Run, "start", fake_start)
    with TestClient(api.app) as one:
        yield one


def test_config_lists_both_models_and_prefers_webjev(client, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    data = client.get("/api/config").json()
    assert [(m["id"], m["label"], m["available"]) for m in data["decision_models"]] == [
        ("webjev", "WebJev-35B-A3B", True), ("jev", "Jev 1.13", True)]
    assert data["default_decision_model"] == "webjev"
    assert "test-key" not in str(data) and "127.0.0.1" not in str(data)


def test_default_falls_back_to_the_configured_model(client, monkeypatch):
    monkeypatch.delenv("DECISION_URL")
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    assert client.get("/api/config").json()["default_decision_model"] == "jev"
    run = client.post("/api/runs", json={"query": "Find the opening hours of a public library"}).json()
    assert run["decision_model"] == "jev"


def test_run_records_the_chosen_model_and_rejects_an_unconfigured_one(client):
    run = client.post("/api/runs", json={"query": "Find the opening hours of a public library",
                                         "decision_model": "webjev"})
    assert run.status_code == 201 and run.json()["decision_model"] == "webjev"
    assert api.runs[run.json()["id"]].payload["decision_model"] == "webjev"
    refused = client.post("/api/runs", json={"query": "Find the opening hours of a public library",
                                             "decision_model": "jev"})
    assert refused.status_code == 422 and "Jev 1.13" in refused.json()["detail"]


def test_no_configured_model_is_a_clear_error(client, monkeypatch):
    monkeypatch.delenv("DECISION_URL")
    response = client.post("/api/runs", json={"query": "Find the opening hours of a public library"})
    assert response.status_code == 503


def test_missing_configuration_names_only_variables(monkeypatch):
    for key in ("TEXT_MODEL_API_KEY", "TEXT_MODEL_BASE_URL", "LEXMOUNT_API_KEY", "LEXMOUNT_PROJECT_ID"):
        monkeypatch.setenv(key, "x")
    assert engine.missing_configuration("webjev") == []
    assert engine.missing_configuration("jev") == ["TYPESAFE_API_KEY or OPENROUTER_API_KEY"]
    assert engine.missing_configuration("") == []
    monkeypatch.delenv("DECISION_URL")
    monkeypatch.delenv("LEXMOUNT_API_KEY")
    assert engine.missing_configuration("webjev") == ["DECISION_URL"]  # no Lexmount key: the local Chrome is used
    monkeypatch.setenv("BROWSER", "lexmount")
    assert engine.missing_configuration("webjev") == ["LEXMOUNT_API_KEY", "DECISION_URL"]


def test_webjev_decisions_go_to_the_self_hosted_endpoint(monkeypatch):
    monkeypatch.setenv("DECISION_URL", "http://127.0.0.1:18000/")
    monkeypatch.setenv("DECISION_API_KEY", "local-key")
    model = SimpleNamespace(jev_endpoint=lambda body: ("https://openrouter.ai/api/alpha/decisions", "k"))
    engine.route_decisions(model, "jev")
    assert model.jev_endpoint({"model": "jev-1.13"})[0].startswith("https://openrouter.ai")
    engine.route_decisions(model, "webjev")
    body = {"model": "jev-1.13", "state": {}, "questions": {}}
    assert model.jev_endpoint(body) == ("http://127.0.0.1:18000/api/alpha/decisions", "local-key")
    assert body["model"] == engine.DEFAULT_WEBJEV_MODEL
