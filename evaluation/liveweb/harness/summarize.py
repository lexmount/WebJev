#!/usr/bin/env python3
"""Success rates from judged result directories: per model, per subset, and paired between two models.

    python harness/summarize.py --run WebJev-35B-A3B=runs/<tag>/webjev --run "Jev 1.13"=runs/<tag>/jev \
        --out-dir results/

A model may have several result directories (comma-separated), e.g. when its tasks were split across servers.

Outcome of one task: S success, F failure (both "gradable"), I infrastructure failure (browser or anti-bot wall),
J verifier error. Success rate = S / (S + F): environment failures and verifier errors are excluded from the
denominator and reported separately. Nothing is re-scored here; run `python -m verifier.judge` first.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIVEWEB = HERE.parent
sys.path.insert(0, str(LIVEWEB))

from verifier.judge import outcome, read_json, task_dirs  # noqa: E402

SUBSETS = [
    ("all", "All 125 tasks", lambda t: True),
    ("online_mind2web", "Online-Mind2Web", lambda t: t["source"]["benchmark"] == "Online-Mind2Web"),
    ("webgym", "WebGym", lambda t: t["source"]["benchmark"] == "WebGym"),
    ("webvoyager", "WebVoyager", lambda t: t["source"]["benchmark"] == "WebVoyager"),
]


def load_tasks(tasks_dir: Path) -> dict:
    return {t["task_id"]: t for t in (json.loads(p.read_text(encoding="utf-8"))
                                      for p in sorted(tasks_dir.glob("*.json")))}


def load_run(dirs: list[Path], tasks: dict) -> dict:
    """{task_id: {outcome, steps, seconds}} over one model's result directories."""
    out = {}
    for root in dirs:
        for task_dir in task_dirs(root):
            record = read_json(task_dir / "record.json") or {}
            tid = record.get("task_id") or task_dir.name
            if tid not in tasks:
                continue
            out[tid] = {"outcome": outcome(record, read_json(task_dir / "judge.json")),
                        "steps": record.get("env_steps"), "seconds": record.get("seconds")}
    return out


def rate(rows: list[dict]) -> dict:
    s = sum(r["outcome"] == "S" for r in rows)
    judged = sum(r["outcome"] in "SF" for r in rows)
    return {"tasks": len(rows), "gradable": judged, "success": s,
            "success_rate": round(s / judged, 4) if judged else None,
            "infra": sum(r["outcome"] == "I" for r in rows), "verifier_error": sum(r["outcome"] == "J" for r in rows),
            "not_run": sum(r["outcome"] == "U" for r in rows)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help="NAME=DIR[,DIR...]")
    ap.add_argument("--tasks-dir", type=Path, default=LIVEWEB / "tasks")
    ap.add_argument("--out-dir", type=Path, help="write per_task.csv, per_task.json and summary.json here")
    a = ap.parse_args()

    tasks = load_tasks(a.tasks_dir)
    runs = {}
    for spec in a.run:
        name, _, dirs = spec.partition("=")
        runs[name] = load_run([Path(d) for d in dirs.split(",") if d], tasks)
    missing = {"outcome": "U", "steps": None, "seconds": None}

    summary = {"tasks": len(tasks), "models": {}, "subsets": {k: label for k, label, _ in SUBSETS}}
    for name, res in runs.items():
        summary["models"][name] = {
            key: rate([res.get(tid, missing) for tid, t in tasks.items() if pred(t)]) for key, _, pred in SUBSETS}
    names = list(runs)
    if len(names) == 2:
        x, y = names
        both = [tid for tid in tasks if runs[x].get(tid, missing)["outcome"] in "SF"
                and runs[y].get(tid, missing)["outcome"] in "SF"]
        summary["paired"] = {
            "gradable_for_both": len(both),
            f"only_{x}": sum(runs[x][t]["outcome"] == "S" and runs[y][t]["outcome"] == "F" for t in both),
            f"only_{y}": sum(runs[y][t]["outcome"] == "S" and runs[x][t]["outcome"] == "F" for t in both),
            "both": sum(runs[x][t]["outcome"] == "S" and runs[y][t]["outcome"] == "S" for t in both),
            "neither": sum(runs[x][t]["outcome"] == "F" and runs[y][t]["outcome"] == "F" for t in both),
        }

    rows = []
    for tid, t in tasks.items():
        row = {"task_id": tid, "benchmark": t["source"]["benchmark"], "website": t["website"]}
        for name, res in runs.items():
            r = res.get(tid, missing)
            row.update({f"{name}_outcome": r["outcome"], f"{name}_steps": r["steps"], f"{name}_seconds": r["seconds"]})
        rows.append(row)

    if a.out_dir:
        a.out_dir.mkdir(parents=True, exist_ok=True)
        (a.out_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
        (a.out_dir / "per_task.json").write_text(json.dumps(rows, indent=1) + "\n")
        with open(a.out_dir / "per_task.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    head = "| Subset | " + " | ".join(names) + " |"
    print(head)
    print("|---|" + "---|" * len(names))
    for key, label, _ in SUBSETS:
        cells = []
        for name in names:
            r = summary["models"][name][key]
            cells.append(f"{r['success']}/{r['gradable']} = {100 * r['success_rate']:.2f}%"
                         if r["success_rate"] is not None else "-")
        print(f"| {label} | " + " | ".join(cells) + " |")
    if "paired" in summary:
        print(json.dumps(summary["paired"]))


if __name__ == "__main__":
    main()
