"""Scoring of finished episodes -- a deterministic verifier, no LLM judge.

Scoring reads only `record.json` + `evidence.json` of an episode and the task definition on disk; it never touches a
browser. The same evidence always yields the same verdict. After a change to a check, re-scoring is enough; after a
change to evidence capture (selectors, API paths), the episode must be re-run.

    python -m verifier.judge RESULTS_DIR [--tasks-dir tasks] [--force]

RESULTS_DIR is one model's result directory (one sub-directory per task) or a single task directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .evidence import EVIDENCE_FILE, score_evidence

HERE = Path(__file__).resolve().parent
TASKS_DIR = HERE.parent / "tasks"

#: Episode buckets that are scored. `infra` never is.
JUDGED_BUCKETS = frozenset({"agent_run", "no_answer", "timeout"})


def read_json(path: Path) -> dict | None:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return data if isinstance(data, dict) else None


def load_task(task_id: str, tasks_dir: Path = TASKS_DIR) -> tuple[dict | None, str | None]:
    """Read a task definition from disk: (task, error)."""
    path = Path(tasks_dir) / f"{task_id}.json"
    if not path.is_file():
        return None, f"task definition not found: {path.name}"
    task = read_json(path)
    if task is None:
        return None, f"task definition is not valid JSON: {path.name}"
    if not isinstance(task.get("evaluator"), dict):
        return None, f"{task_id}: the task has no evaluator"
    return task, None


def judge_task_dir(task_dir: Path, tasks_dir: Path = TASKS_DIR) -> dict:
    """Score one episode directory; returns the verdict (the content of judge.json)."""
    record = read_json(task_dir / "record.json")
    if record is None:
        return {"error": "record.json cannot be read", "judge_status": "error"}

    task_id = record.get("task_id") or task_dir.name
    task, why = load_task(str(task_id), tasks_dir)
    if task is None:
        # A missing task definition is our problem, not the agent's: judge_error, not 0.
        return {"evaluator": "script", "judge_model": "script",
                "reward": None, "success": False, "bucket": "judge_error",
                "judge_status": "error", "error": why}

    evidence = read_json(task_dir / EVIDENCE_FILE)
    if evidence is None:
        return {"evaluator": "script", "judge_model": "script",
                "reward": None, "success": False, "bucket": "judge_error",
                "judge_status": "error",
                "error": f"no {EVIDENCE_FILE}: evidence was never captured for this episode; re-run it"}

    return score_evidence(task["evaluator"], evidence=evidence, record=record,
                          task=task, task_dir=task_dir)


def judge_status(verdict: dict) -> str:
    """"ok" = a usable verdict; "error" = the verifier could not score (kept out of the denominator)."""
    status = verdict.get("judge_status")
    if status in ("ok", "error"):
        return status
    if verdict.get("error"):
        return "error"
    if verdict.get("reward") is None and verdict.get("verdict") is None:
        return "error"
    return "ok"


def is_success(verdict: dict) -> bool:
    """reward == 1.0 (a bool placeholder is not a score)."""
    reward = verdict.get("reward")
    if isinstance(reward, (int, float)) and not isinstance(reward, bool):
        if reward == 1.0:
            return True
    return str(verdict.get("verdict", "")).upper() == "SUCCESS"


def outcome(record: dict, verdict: dict | None) -> str:
    """S success / F failure / I infra (environment) / J verifier error / U unscored."""
    if verdict is None:
        return "I" if record.get("bucket") == "infra" else "U"
    if verdict.get("bucket") == "infra":
        return "I"                       # health gate: the page was an anti-bot wall
    if judge_status(verdict) != "ok":
        return "J"
    return "S" if is_success(verdict) else "F"


def needs_judge(task_dir: Path, *, force: bool) -> bool:
    record = read_json(task_dir / "record.json")
    if record is None or record.get("bucket") not in JUDGED_BUCKETS:
        return False
    return force or not (task_dir / "judge.json").exists()


def write_judge(task_dir: Path, verdict: dict) -> Path:
    out = dict(verdict)
    out.setdefault("judge_status", judge_status(verdict))
    path = task_dir / "judge.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def task_dirs(root: Path):
    """Episode directories under a model's result directory (or the directory itself if it is one)."""
    root = Path(root)
    if (root / "record.json").exists():
        yield root
        return
    for path in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith((".", "_"))):
        if (path / "record.json").exists():
            yield path


def judge_all(root: Path, *, tasks_dir: Path = TASKS_DIR, force: bool = False) -> dict:
    counts = {"S": 0, "F": 0, "I": 0, "J": 0, "U": 0}
    for task_dir in task_dirs(root):
        if needs_judge(task_dir, force=force):
            try:
                verdict = judge_task_dir(task_dir, tasks_dir)
            except Exception as exc:  # noqa: BLE001 - a verifier failure is judge_error, never a 0
                verdict = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
            verdict.setdefault("judge_model", "script")
            write_judge(task_dir, verdict)
        counts[outcome(read_json(task_dir / "record.json") or {}, read_json(task_dir / "judge.json"))] += 1
    judged = counts["S"] + counts["F"]
    return {"success": counts["S"], "judged": judged, "rate": round(counts["S"] / judged, 4) if judged else None,
            "outcomes": counts}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=Path)
    ap.add_argument("--tasks-dir", type=Path, default=TASKS_DIR)
    ap.add_argument("--force", action="store_true", help="re-score episodes that already have a judge.json")
    a = ap.parse_args(argv)
    print(json.dumps(judge_all(a.results, tasks_dir=a.tasks_dir, force=a.force)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
