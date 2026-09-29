"""Structure of the 125 task files: every check refers to a registered metric and getter, options are accepted by the
metric, `requires` only names earlier checks, and the set has the documented composition."""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from verifier.getters import GETTERS  # noqa: E402
from verifier.metrics import METRICS, allowed_options  # noqa: E402

TASKS = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((ROOT / "tasks").glob("*.json"))]


def test_composition():
    assert len(TASKS) == 125
    assert len({t["task_id"] for t in TASKS}) == 125
    assert collections.Counter(t["source"]["benchmark"] for t in TASKS) == {
        "Online-Mind2Web": 75, "WebGym": 26, "WebVoyager": 24}
    assert len({(t["source"]["benchmark"], t["source"]["task_id"]) for t in TASKS}) == 125   # one task per item


def test_file_names_match_ids():
    for path in sorted((ROOT / "tasks").glob("*.json")):
        assert json.loads(path.read_text(encoding="utf-8"))["task_id"] == path.stem


def _getter_types(config):
    if isinstance(config, dict) and isinstance(config.get("type"), str):
        yield config["type"]


def test_checks_are_well_formed():
    for task in TASKS:
        spec = task["evaluator"]
        assert spec["conj"] in ("and", "or"), task["task_id"]
        assert "health_gate" not in spec, task["task_id"]      # the health gate is global, never per task
        seen = []
        for check in spec["checks"]:
            assert check["metric"] in METRICS, (task["task_id"], check["metric"])
            for slot in ("result", "expected"):
                for kind in _getter_types(check.get(slot)):
                    assert kind in GETTERS, (task["task_id"], kind)
            allowed = allowed_options(check["metric"])
            if allowed is not None:
                assert set(check.get("options") or {}) <= allowed, (task["task_id"], check.get("options"))
            requires = check.get("requires") or []
            requires = [requires] if isinstance(requires, str) else requires
            assert set(requires) <= set(seen), (task["task_id"], check["name"], requires)
            seen.append(check["name"])


def test_or_tasks_are_the_documented_two():
    assert sorted(t["task_id"] for t in TASKS if t["evaluator"]["conj"] == "or") == [
        "vts-108-onlinemind2web-1df24ec81137386d6476bcf343a79012",
        "vts-119-onlinemind2web-7680a920359cb1a508fbddb001b98167"]
