"""Shared by the preparer, the runner and the scorer: one item format, one way to read a predicted label.

Item (one line of $EVAL_ROOT/bench/<benchmark>.jsonl):
    {"bench", "subset", "id", "state", "questions": {qid: {"type", "instructions", "criteria"}}, "gold": {qid: label}}

`state` and `questions` are exactly the body of a Jev request (TypeSafe `POST /v1/systemone`, OpenRouter
`POST /api/alpha/decisions`), so the same item goes unchanged to Jev and to WebJev.
"""
import gzip
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = os.environ.get("EVAL_ROOT", str(HERE / "work"))      # sources/, bench/ and pred/ live here
BENCHES = [
    "jevbench_public", "nimble_eval324", "kev_ext_semif", "kev_ext_scienthoon", "kev_transfer_v4_dev",
    "kev_decision_v7_dev", "kev_ext_mmlupro", "typed_decisions_test",
]
TITLES = {
    "jevbench_public": "JevBench (public items)",
    "nimble_eval324": "Nimble (eval split)",
    "kev_ext_semif": "SemIf (external items)",
    "kev_ext_scienthoon": "scienthoon support tickets",
    "kev_transfer_v4_dev": "kev transfer-v4 (dev)",
    "kev_decision_v7_dev": "kev decision-v7 (dev)",
    "kev_ext_mmlupro": "MMLU-Pro (10-way)",
    "typed_decisions_test": "typed-decisions (test)",
}


def load_bench(name):
    return [json.loads(line) for line in open(f"{ROOT}/bench/{name}.jsonl")]


def read_jsonl(path):
    """Rows of a .jsonl or .jsonl.gz file."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


def n_options(q):
    """How many answers a question admits. noul is always yes/no."""
    if q["type"] == "noul":
        return 2
    c = q.get("criteria")
    return len(c) if isinstance(c, (dict, list)) else 0


def gold_label(q, g):
    """Normalize a benchmark's gold answer to the label space used for scoring."""
    t = q["type"]
    if t == "noul":
        if isinstance(g, bool):
            return "yes" if g else "no"
        return "yes" if str(g).strip().lower() in ("yes", "true", "1") else "no"
    if t == "score":
        return str(int(g)) if not isinstance(g, str) else g.strip()
    return str(g)


def pred_label(q, a):
    """The label an answer commits to: choice -> `choice`; noul -> yes iff p >= 0.5; score -> most probable level."""
    if not isinstance(a, dict):
        return None
    t = q["type"]
    if t == "choice":
        return a.get("choice")
    if t == "noul":
        p = a.get("noul")
        return None if p is None else ("yes" if p >= 0.5 else "no")
    probs = a.get("probabilities") or {}
    if probs:
        return str(max(probs, key=lambda k: probs[k]))
    return None if a.get("score") is None else str(int(round(a["score"])))
