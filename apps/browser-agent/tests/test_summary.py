"""Direct final-LLM output: no extra requirement or evidence-grading call."""
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend import result


def test_summary_is_one_call_and_preserves_model_answer_without_grading():
    answer = {
        "summary": "Here is the result.",
        "columns": [{"key": "rating", "label": "Rating"}],
        "rows": [{"cells": {"rating": {"value": "8.4"}}}],
        "status": "partial", "notes": ["native browser status was blocked"],
    }
    with patch.object(result, "_call_model", return_value=answer) as call:
        value = result.build_summary("Find this movie's rating", [{"id": 1, "text": "Rating 8.4"}], "blocked")
    assert call.call_count == 1
    assert "browser_status" not in call.call_args.args[1]
    assert value["status"] == "summarized"
    assert value["summary"] == answer["summary"]
    assert value["rows"] == answer["rows"]
    assert not {"checks", "evidence_validation", "missing_fields", "notes"} & value.keys()


def test_missing_information_remains_in_the_model_answer():
    with patch.object(result, "_call_model", return_value={
        "summary": "The page does not provide this information.", "columns": [], "rows": [],
    }) as call:
        value = result.build_summary("Find the information on the page", [], "failed")
    assert call.call_count == 1
    assert value["summary"] == "The page does not provide this information."
    assert value["status"] == "summarized"


def test_summary_transport_failure_does_not_expose_provider_error():
    with patch.object(result, "_call_model", side_effect=RuntimeError("private-provider-error")):
        value = result.build_summary("Find the information on the page", [], "done")
    assert value["status"] == "failed"
    assert "private-provider-error" not in str(value)


def test_summary_redacts_secrets_before_display(monkeypatch):
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "private-summary-secret-123")
    with patch.object(result, "_call_model", return_value={
        "summary": "private-summary-secret-123", "columns": [], "rows": [],
    }):
        value = result.build_summary("Find the information on the page", [], "done")
    assert "private-summary-secret-123" not in str(value)


def test_summary_prompt_is_english_and_answers_in_the_task_language():
    import re

    with patch.object(result, "_call_model", return_value={"summary": "77", "columns": [], "rows": []}) as call:
        result.build_summary("How many lessons are free?", [{"id": 1, "text": "77 lessons"}], "done")
    system, payload = call.call_args.args
    assert not re.search(r"[\u4e00-\u9fff]", system)
    assert payload["answer_language"] == "English" and "answer_language" in system
    assert set(payload["runtime"]) == {"now", "timezone", "today", "tomorrow", "next_monday"}
    with patch.object(result, "_call_model", return_value={"summary": "7", "columns": [], "rows": []}) as call:
        # A Chinese task ("find this movie's rating") gets a Chinese answer.
        result.build_summary("\u67e5\u8be2\u8be5\u7535\u5f71\u7684\u8bc4\u5206", [{"id": 1, "text": "7.0"}], "done")
    assert call.call_args.args[1]["answer_language"] == "Chinese"
