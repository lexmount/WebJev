"""Offline tests of the summary module's input handling: no browser, recordings or live model."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1] / "backend" / "result.py"
SPEC = importlib.util.spec_from_file_location("browser_result_under_test", PATH)
result = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(result)


def test_native_aria_boolean_strings_are_normalized_without_inventing_selection():
    observed, truncated = result._observations(
        [
            {
                "id": 0,
                "text": "Filter",
                "visible_controls_verified": True,
                "visible_controls": [
                    {
                        "id": "yes",
                        "label": "Boston",
                        "visible": True,
                        "checked": "true",
                        "selected": "false",
                    }
                ],
            }
        ],
        (),
    )
    assert truncated is False
    assert observed[0]["visible_controls"][0]["checked"] is True
    assert observed[0]["visible_controls"][0]["selected"] is False


def test_extraction_has_its_own_bounded_timeout_and_closes_client(monkeypatch):
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "test-private-token")
    observed = {}

    def post(url, key, body, client):
        observed["client"] = client
        observed["timeout"] = client.timeout
        return {"choices": [{"finish_reason": "stop", "message": {"content": "{}"}}]}

    with patch.dict(
        sys.modules, {"jev_ultrafast": SimpleNamespace(model=SimpleNamespace(post_json=post))}
    ):
        assert result._call_model("extract", {}) == {}
    assert observed["timeout"].read == 120
    assert observed["timeout"].connect == 15
    assert observed["client"].is_closed


def test_stateless_navigation_controls_do_not_truncate_real_evidence():
    controls = [
        {
            "id": f"link{i}",
            "visible": True,
            "role": "button",
            "label": "navigation" * 40,
            "value": "0",
        }
        for i in range(220)
    ]
    controls += [
        {"id": "sort", "visible": True, "role": "button", "label": "Citations", "selected": True}
    ]
    controls += [
        {"id": "topic", "visible": True, "role": "textbox", "label": "Topic", "value": "quantum computing"}
    ]
    observations = [
        {
            "id": i,
            "text": f"Visible page {i}",
            "visible_controls": controls,
            "visible_controls_verified": True,
        }
        for i in range(36)
    ]
    value, trimmed = result._observations(observations, ())
    assert trimmed is False
    assert len(value) == 36
    assert [c["id"] for c in value[0]["visible_controls"]] == ["sort", "topic"]
