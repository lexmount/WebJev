import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.diagnostics import ModelDiagnostics, generation_stage, read_calls


def test_plan_without_explicit_website_still_has_correct_stage():
    assert generation_stage({"messages": [{"role": "user", "content": json.dumps({
        "query": "Search bookshop.org for books on artificial intelligence", "runtime": "today"
    })}]}) == "semantic_plan"


class Writer:
    def __init__(self, folder):
        self.folder = folder
        self.tick = 100

    def elapsed(self):
        self.tick += 1
        return self.tick


def test_actual_input_output_and_models_are_recorded_without_reasoning(tmp_path, monkeypatch):
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "test-private-secret-123")
    recorder = ModelDiagnostics(Writer(tmp_path))
    body = {"model": "requested-model", "messages": [{"role": "user", "content": json.dumps({"field": "private input"})}]}
    response = {"model": "actual-model", "choices": [{"message": {"content": '{"text":"answer"}', "reasoning_content": "private reasoning"}}], "usage": {"total_tokens": 3}}

    def invoke():
        pending = read_calls(tmp_path)["calls"]
        assert pending[0]["status"] == "running"
        assert pending[0]["duration_ms"] is None
        assert pending[0]["input"] == body
        return response

    assert recorder.call("generation", body, invoke) is response
    result = read_calls(tmp_path)
    assert len(result["calls"]) == 1
    call = result["calls"][0]
    assert call["stage"] == "field_text" and call["response_model"] == "actual-model"
    assert call["output"] == '{"text":"answer"}' and call["duration_ms"] >= 0
    assert call["input"] == body
    assert read_calls(tmp_path, result["revision"])["calls"] is None
    serialized = (tmp_path / "model_calls.jsonl").read_text()
    assert "private input" in serialized and "private reasoning" not in serialized
    assert (tmp_path / "model_calls.jsonl").stat().st_mode & 0o777 == 0o600


def test_jev_output_includes_actual_decision_and_target(tmp_path):
    recorder = ModelDiagnostics(Writer(tmp_path))
    body = {"model": "jev", "questions": {"click_target": {"criteria": {"2": {"element": "[2] Search"}}}}}
    answers = {"operation": {"choice": "CLICK"}, "click_target": {"choice": "2"}}
    response = {"model": "jev-pinned", "answers": answers}
    recorder.call("jev", body, lambda: response)
    call = read_calls(tmp_path)["calls"][0]
    assert call["output"] == answers
    assert call["input"] == body
    assert call["target"] == "[2] Search" and call["action"] == "CLICK"


def test_failures_keep_timing_without_leaking_provider_error(tmp_path):
    recorder = ModelDiagnostics(Writer(tmp_path))

    def fail():
        raise RuntimeError("HTTP 429 private-endpoint private-key")

    with pytest.raises(RuntimeError):
        recorder.call("generation", {"model": "test"}, fail)
    call = read_calls(tmp_path)["calls"][0]
    assert call["status"] == "failed" and call["duration_ms"] >= 0
    assert call["error"] == "RuntimeError · HTTP 429"
    assert "private-" not in json.dumps(call)


def test_output_secrets_are_redacted_and_partial_journal_lines_are_ignored(tmp_path, monkeypatch):
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "test-private-secret-123")
    recorder = ModelDiagnostics(Writer(tmp_path))
    recorder.call("generation", {"model": "test"}, lambda: {"choices": [{"message": {"content": "test-private-secret-123 Bearer very-private-token"}}]})
    with (tmp_path / "model_calls.jsonl").open("a") as stream:
        stream.write('{"id":')
    call = read_calls(tmp_path)["calls"][0]
    assert "test-private" not in call["output"] and "very-private" not in call["output"]


def test_input_is_redacted_without_mutating_the_sent_request(tmp_path, monkeypatch):
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "test-private-secret-123")
    recorder = ModelDiagnostics(Writer(tmp_path))
    body = {"model": "test", "state": {"api_key": "hidden-value", "text": "test-private-secret-123"}}

    def invoke():
        assert body["state"]["api_key"] == "hidden-value"
        assert body["state"]["text"] == "test-private-secret-123"
        return {"answers": {}}

    recorder.call("jev", body, invoke)
    assert read_calls(tmp_path)["calls"][0]["input"]["state"] == {"api_key": "[REDACTED]", "text": "[REDACTED]"}
    assert "hidden-value" not in (tmp_path / "model_calls.jsonl").read_text()


def test_legacy_logs_use_recorded_values_and_leave_missing_timing_unknown(tmp_path):
    (tmp_path / "raw_state.json").write_text(json.dumps({
        "decisions": [{"latency_ms": 300, "elapsed_ms": 800, "model": "actual-jev", "request": {"model": "jev", "state": "saved page"}, "raw_answers": {"operation": {"choice": "DONE"}}}],
        "text_calls": [{"value": "search", "model": "configured-model"}],
    }))
    data = read_calls(tmp_path)
    assert data["legacy"] is True
    by_kind = {c["kind"]: c for c in data["calls"]}
    assert by_kind["jev"]["started_elapsed_ms"] == 500
    assert by_kind["jev"]["input"] == {"model": "jev", "state": "saved page"}
    assert by_kind["generation"].get("input") is None
    assert by_kind["generation"]["response_model"] is None
    assert by_kind["generation"]["duration_ms"] is None
