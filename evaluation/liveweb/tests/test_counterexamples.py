"""Counterexamples: synthetic wrong end states built on real evidence; the verifier must REJECT them.

A positive run proves little -- the verifier was written from it. What matters is what a verifier rejects, and false
positives are only caught by counterexamples. Each task under `counterexamples/` has a list of cases and a base
evidence file captured from a live session when the verifier was written. A case rewrites one or two captured values
(the URL, a DOM value, a control state, the final answer) and the result goes through `score_evidence()` -- offline,
no browser, no LLM.

`expect` is the score the verifier must give (1 = must pass, 0 = must fail). A case without `expect` asserts that
evidence capture is broken: the reward must be None (judge_error), not 0.0 -- 0.0 means "the agent was wrong", None
means "we cannot tell".

    python -m pytest tests -q
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from verifier.evidence import score_evidence, session_configs  # noqa: E402
from verifier.metrics.rules import MISSING  # noqa: E402

CASES = ROOT / "tests" / "counterexamples"
TASKS = ROOT / "tasks"
URL_FAMILY = {"final_url", "url_parse", "url_path_parse", "open_tabs"}


def _url_value(cfg: dict, url: str, base_title: str = ""):
    """What a URL-family getter would return for a given URL (mirrors verifier/getters/navigation.py)."""
    t = cfg["type"]
    if t == "final_url":
        return url
    if t == "url_parse":
        raw = parse_qs(urlsplit(url).query, keep_blank_values=True)
        multi = bool(cfg.get("multi"))
        params = {k: (v if multi else (v[0] if v else "")) for k, v in raw.items()}
        keys = cfg.get("parse_keys")
        return {k: params.get(k, MISSING) for k in keys} if keys else params
    if t == "url_path_parse":
        parts = [p for p in urlsplit(url).path.split(cfg.get("split_by", "/")) if p]
        idx = cfg.get("index")
        if idx is None:
            return parts
        try:
            return parts[int(idx)]
        except IndexError:
            return MISSING
    if t == "open_tabs":
        return [{"url": url, "title": base_title}]
    raise AssertionError(t)


def _apply_case(base: dict, spec: dict, case: dict) -> dict:
    ev = copy.deepcopy(base)
    getters = ev.setdefault("getters", {})
    title = ((ev.get("target") or {}).get("title")) or ""

    if case.get("url"):
        for key, cfg in session_configs(spec).items():
            if cfg.get("type") in URL_FAMILY:
                getters[key] = {"config": cfg, "ok": True,
                                "value": _url_value(cfg, case["url"], title)}
        ev.setdefault("target", {})["url"] = case["url"]

    # Replace one getter's value directly (page text, control state, DOM values), or make it fail.
    for ovr in case.get("getters") or []:
        for key, cfg in session_configs(spec).items():
            if all(cfg.get(k) == v for k, v in (ovr.get("match") or {}).items()):
                entry = {"config": cfg}
                if "error" in ovr:
                    entry.update(ok=False, error=ovr["error"])
                else:
                    entry.update(ok=True, value=ovr["value"])
                getters[key] = entry
    return ev


def _load(task_id: str):
    task = json.loads((TASKS / f"{task_id}.json").read_text(encoding="utf-8"))
    base = json.loads((CASES / "_evidence" / f"{task_id}.json").read_text(encoding="utf-8"))
    cases = json.loads((CASES / f"{task_id}.json").read_text(encoding="utf-8"))
    return task, base, cases


ALL = sorted(p.stem for p in CASES.glob("vts-*.json"))
FLAT = [(tid, i) for tid in ALL for i in range(len(json.loads(
    (CASES / f"{tid}.json").read_text(encoding="utf-8"))))]


@pytest.mark.parametrize("task_id,index", FLAT,
                         ids=[f"{t.split('-')[1]}#{i}" for t, i in FLAT])
def test_counterexample(task_id, index, tmp_path):
    task, base, cases = _load(task_id)
    case = cases[index]
    evidence = _apply_case(base, task["evaluator"], case)
    out = score_evidence(task["evaluator"],
                         evidence=evidence,
                         record={"done_text": case.get("answer", ""), "bucket": "agent_run"},
                         task=task, task_dir=tmp_path)
    reward, expect = out.get("reward"), case.get("expect")
    label = case.get("label", "")
    if expect is None:
        # Broken capture must leave the denominator instead of counting as a wrong agent.
        assert reward is None, f"{task_id} ({label}): broken capture scored {reward!r}, expected None"
        assert out.get("bucket") == "judge_error", f"{task_id} ({label}): bucket {out.get('bucket')!r}"
    else:
        assert reward == float(expect), f"{task_id} ({label}): scored {reward!r}, expected {float(expect)}"


def test_every_case_file_has_a_task_and_base_evidence():
    for tid in ALL:
        assert (TASKS / f"{tid}.json").is_file(), tid
        assert (CASES / "_evidence" / f"{tid}.json").is_file(), tid
