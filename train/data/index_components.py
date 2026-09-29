#!/usr/bin/env python3
"""Index the general and web components so that the later components add only new decisions.

Keys (SQLite table seen(kind, key, source)):
  stem      normalized context (for multiple-choice data the context holds the question stem)
  full      normalized context + option set, per question
  decision  normalized context + question + option set, per question
  prompt    the exact tokenized model input and answer positions

The Open-Jev and knowledge-MCQA builders skip any candidate whose stem, full, decision or prompt key is
already in this index; `assemble.py` then removes exact prompt duplicates across all components.

    python data/index_components.py   ->   $WEBJEV_WORK/index/seen.sqlite (+ manifest.json)
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import pickle
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import WORK, decision_key, iter_jsonl, mcqa_keys, prompt_identity, require, sha256_file  # noqa: E402

GENERAL_PARQUET = WORK / "general" / "exports" / "train.parquet"


class Seen:
    """Read-only view used by the builders."""

    def __init__(self, path: Path):
        require(Path(path).is_file(), f"missing component index {path}; run data/index_components.py")
        self.conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        self.counts = dict(self.conn.execute("SELECT kind, COUNT(*) FROM seen GROUP BY kind"))

    def has(self, kind: str, key: str) -> bool:
        return self.conn.execute("SELECT 1 FROM seen WHERE kind=? AND key=? LIMIT 1", (kind, key)).fetchone() is not None

    def keys(self, kind: str) -> set:
        return {key for (key,) in self.conn.execute("SELECT DISTINCT key FROM seen WHERE kind=?", (kind,))}

    def close(self) -> None:
        self.conn.close()


def main() -> None:
    import pyarrow.parquet as pq
    out = WORK / "index"
    out.mkdir(parents=True, exist_ok=True)
    db = out / "seen.sqlite"
    require(not db.exists(), f"index already exists: {db}")
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE seen(kind TEXT NOT NULL, key TEXT NOT NULL, source TEXT NOT NULL, "
                 "PRIMARY KEY(kind, key, source)) WITHOUT ROWID")
    counts, pending = Counter(), []

    def flush():
        conn.executemany("INSERT OR IGNORE INTO seen(kind, key, source) VALUES (?, ?, ?)", pending)
        pending.clear()

    # General component: its portable export holds the untokenized rows.
    for batch in pq.ParquetFile(GENERAL_PARQUET).iter_batches(batch_size=2048, columns=["context", "questions"]):
        for row in batch.to_pylist():
            counts["general_rows"] += 1
            context = row.get("context")
            if not isinstance(context, str) or not context.strip():
                continue
            pending.append(("stem", mcqa_keys(context, [])[0], "general"))
            for q in row.get("questions") or []:
                options = q.get("options") or []
                if 2 <= len(options) <= 255 and all(isinstance(o, str) for o in options):
                    pending.append(("full", mcqa_keys(context, options)[1], "general"))
                    if isinstance(q.get("text"), str):
                        pending.append(("decision", decision_key(context, q["text"], options), "general"))
            if len(pending) >= 20000:
                flush()
    flush()
    for name in ("general", "web"):
        with (WORK / "packs" / name / "items.pkl").open("rb") as stream:
            items = pickle.load(stream)
        for item in items:
            pending.append(("prompt", prompt_identity(item), name))
            if len(pending) >= 20000:
                flush()
        counts[f"{name}_prompts"] = len(items)
        del items
        flush()
    # Web component: its pack origins carry the context and decision keys of each row.
    for _, origin in iter_jsonl(WORK / "packs" / "web" / "origins.jsonl.gz"):
        pending.append(("stem", origin["stem_key"], "web"))
        pending.append(("decision", origin["decision_key"], "web"))
        counts["web_rows"] += 1
        if len(pending) >= 20000:
            flush()
    flush()
    conn.commit()
    distinct = {f"{kind}:{source}": n for kind, source, n in
                conn.execute("SELECT kind, source, COUNT(*) FROM seen GROUP BY kind, source")}
    conn.close()
    require(counts["general_rows"] == counts["general_prompts"], "general parquet and items are not aligned")
    manifest = {"counts": dict(counts), "distinct_keys": distinct, "sqlite_sha256": sha256_file(db)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
