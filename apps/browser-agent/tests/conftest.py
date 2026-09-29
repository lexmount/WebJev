"""API unit tests must never start billable cloud browsers or call real models."""

import pytest


@pytest.fixture(autouse=True)
def disable_live_pool(monkeypatch):
    monkeypatch.setenv("BROWSER_AGENT_POOL_SIZE", "0")


@pytest.fixture(autouse=True)
def offline_configuration(monkeypatch):
    # A decision model is "configured" when its endpoint or key is set; nothing is called in these tests.
    monkeypatch.setenv("DECISION_URL", "http://127.0.0.1:9")
    for key in ("TYPESAFE_API_KEY", "OPENROUTER_API_KEY", "BROWSER_AGENT_DECISION", "BROWSER_AGENT_DEFAULT_DECISION",
                "BROWSER", "CHROME_CDP_URL", "LEXMOUNT_API_KEY", "LEXMOUNT_PROJECT_ID"):
        monkeypatch.delenv(key, raising=False)
