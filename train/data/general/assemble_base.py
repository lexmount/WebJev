#!/usr/bin/env python3
"""Collect every converted task into the upstream base cache `data/tasks.pkl` (inside the upstream working copy).

Training rows come only from tasks that are not held out; the evaluation rows of every task are kept
separately. Run after convert_tasks.py has finished every registered task.
"""
from __future__ import annotations

import json
from pathlib import Path
import pickle
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import WORK, require, sha256_file, use_upstream  # noqa: E402

GENERAL = WORK / "general"
EMPTY_BY_DESIGN = {"abstain_probe", "offtopic_probe"}


def main() -> None:
    upstream = use_upstream(GENERAL / "upstream")
    from decider import data as D  # noqa: F401  (registers the example classes for unpickling)

    registry = json.loads((GENERAL / "task-registry.json").read_text())
    train, evals, rows, missing = [], {}, [], []
    for entry in registry:
        name = entry["task"]
        folder = GENERAL / "task-cache" / name
        status = json.loads((folder / "status.json").read_text()) if (folder / "status.json").exists() else {}
        if status.get("status") != "complete":
            missing.append(name)
            continue
        with (folder / "converted.pkl").open("rb") as stream:
            task_train, task_eval = pickle.load(stream)
        require(task_train or task_eval or name in EMPTY_BY_DESIGN, f"{name}: empty task")
        if not entry["heldout"]:
            train.extend(task_train)
        evals[name] = task_eval
        rows.append(status)
    require(not missing, f"tasks not converted yet: {missing}")
    out = upstream / "data" / "tasks.pkl"
    out.parent.mkdir(exist_ok=True)
    with out.open("wb") as stream:
        pickle.dump((train, evals), stream, protocol=5)
    report = {"tasks": len(rows), "heldout_tasks": sum(r["heldout"] for r in rows),
              "train_rows": len(train), "train_questions": sum(len(e.qs) for e in train),
              "eval_rows": sum(len(v) for v in evals.values()),
              "sha256": sha256_file(out)}
    (GENERAL / "base.json").write_text(json.dumps({**report, "per_task": rows}, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
