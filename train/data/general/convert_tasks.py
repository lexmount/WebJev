#!/usr/bin/env python3
"""Convert every task registered by the upstream data package into the unified decision format.

Each task runs in its own process (`--worker NAME`) with the upstream loader, unchanged. The loader's
`datasets.load_dataset` calls are pinned to the revisions in source_revisions.json, and every source it
reads is recorded. Output per task (resumable; finished tasks are skipped):

    $WEBJEV_WORK/general/task-cache/<task>/converted.pkl   (train examples, eval examples)
    $WEBJEV_WORK/general/task-cache/<task>/status.json     row counts, held-out flag, SHA-256
    $WEBJEV_WORK/general/task-cache/<task>/sources.json    datasets, configs, splits and revisions read

    python general/convert_tasks.py [TASK ...] [--workers 6]

Run `general/sources.py trec`, `general/sources.py mind2web` and `general/build_mario.py` first: the upstream
`mario` loader reads the generated data/mario.pkl, and an empty task fails the conversion.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import WORK, use_upstream  # noqa: E402

GENERAL = WORK / "general"
CACHE = GENERAL / "task-cache"
REVISIONS = json.loads((HERE / "source_revisions.json").read_text())["datasets"]
EMPTY_BY_DESIGN = {"abstain_probe", "offtopic_probe"}   # placeholders; their rows are built by the mixture step
RETRYABLE = ("timeout", "ssl", "connection", "502", "503", "504", "429", "temporary", "retry", "disconnected", "eof")


def atomic_json(path: Path, value) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(path)


def pinned_loader(sources: list, record: Path):
    """Wrap datasets.load_dataset: pin revisions, route TREC and Mind2Web to verified local copies, record sources."""
    import datasets

    original = datasets.load_dataset

    def load(*args, **kwargs):
        repo = str(args[0] if args else kwargs["path"])
        entry = {"dataset": repo, "config": args[1] if len(args) > 1 else kwargs.get("name"),
                 "split": kwargs.get("split")}
        if repo == "CogComp/trec":
            verified = json.loads((GENERAL / "cache/trec/verified.json").read_text())
            if kwargs.get("split") != "test":
                raise ValueError("only the TREC test split is used by the upstream loaders")
            result = original("parquet", data_files={"test": verified["local_parquet"]}, split="test")
            entry.update(revision=verified["original_revision"], transport="verified local parquet")
        elif repo == "osunlp/Mind2Web":
            verified = json.loads((GENERAL / "cache/mind2web/verified.json").read_text())
            files = [f["local"] for f in verified["files"]]
            result = original("json", data_files={"train": files}, split=kwargs.get("split", "train"))
            entry.update(revision=verified["revision"], transport="verified local originals")
        else:
            if "/" in repo and not Path(repo).exists() and not repo.startswith(("http:", "https:")):
                kwargs.setdefault("revision", REVISIONS.get(repo))
                if kwargs["revision"] is None:
                    kwargs.pop("revision")
            entry["revision"] = kwargs.get("revision")
            for attempt in range(5):
                try:
                    result = original(*args, **kwargs)
                    break
                except Exception as exc:  # transient hub errors only
                    text = (type(exc).__name__ + " " + str(exc)).lower()
                    if attempt == 4 or not any(t in text for t in RETRYABLE):
                        raise
                    time.sleep(min(30, 2 ** attempt))
        entry["rows"] = getattr(result, "num_rows", None)
        sources.append(entry)
        atomic_json(record, sources)
        return result

    datasets.load_dataset = load


def worker(name: str) -> None:
    upstream = GENERAL / "upstream"
    use_upstream(upstream)
    os.chdir(upstream)
    folder = CACHE / name
    folder.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        sources: list = []
        pinned_loader(sources, folder / "sources.json")
        from decider import data as D
        train, evals = D.load_task(name)
        if not train and not evals and name not in EMPTY_BY_DESIGN:
            raise ValueError("registered task produced no rows")
        bad = [(split, i) for split, rows in (("train", train), ("eval", evals)) for i, e in enumerate(rows)
               if not isinstance(e.context, str) or not e.qs
               or any(not q.options or not isinstance(q.gold, int) or not 0 <= q.gold < len(q.options) for q in e.qs)]
        if bad:
            raise ValueError(f"invalid records: {bad[:8]} (count {len(bad)})")
        tmp = folder / "converted.tmp"
        with tmp.open("wb") as stream:
            pickle.dump((train, evals), stream, protocol=5)
        tmp.replace(folder / "converted.pkl")
        status = {"status": "complete", "task": name, "heldout": D.TASKS[name]["heldout"],
                  "train_rows": len(train), "eval_rows": len(evals),
                  "train_questions": sum(len(e.qs) for e in train), "eval_questions": sum(len(e.qs) for e in evals),
                  "seconds": round(time.time() - started, 1),
                  "sha256": hashlib.sha256((folder / "converted.pkl").read_bytes()).hexdigest()}
        atomic_json(folder / "status.json", status)
        print(json.dumps(status), flush=True)
    except Exception as exc:
        atomic_json(folder / "status.json", {"status": "failed", "task": name, "error": f"{type(exc).__name__}: {exc}"[:2000]})
        traceback.print_exc()
        sys.exit(1)


def registry() -> list[dict]:
    use_upstream(GENERAL / "upstream")
    from decider import data as D
    return [{"task": name, "heldout": spec["heldout"]} for name, spec in D.TASKS.items()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tasks", nargs="*")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.worker:
        return worker(a.worker)
    tasks = registry()
    (GENERAL / "logs").mkdir(parents=True, exist_ok=True)
    (GENERAL / "task-registry.json").write_text(json.dumps(tasks, indent=1) + "\n")
    names = [t["task"] for t in tasks if not a.tasks or t["task"] in a.tasks]

    def run(name: str):
        status = CACHE / name / "status.json"
        if status.exists() and json.loads(status.read_text()).get("status") == "complete":
            return name, "cached"
        with (GENERAL / "logs" / f"task-{name}.log").open("a") as log:
            try:
                code = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", name],
                                      stdout=log, stderr=subprocess.STDOUT, timeout=3 * 3600).returncode
            except subprocess.TimeoutExpired:
                code = 124
        print(json.dumps({"task": name, "exit": code}), flush=True)
        return name, code

    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        results = list(pool.map(run, names))
    failed = [name for name, code in results if code not in (0, "cached")]
    print(json.dumps({"tasks": len(results), "failed": failed}), flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
