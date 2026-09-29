#!/usr/bin/env python3
"""Finish the general mixture: drop rows without questions, export portable files, write the component pack.

1. The upstream JSON-state builder can sample zero matching records for a row and emit a row with no
   question. Such rows carry no supervision; they are removed from the mixture and from the tokenized items
   (nothing else changes; their original indices are listed in excluded-empty-questions.jsonl).
2. Portable copies for reading without the upstream package (and for the leakage checks of the other
   components): exports/train.parquet, exports/eval/<task>.parquet, exports/probes/<probe>.parquet.
   One row: {"id", "task", "context", "questions": [{"text", "options", "gold"}]}; gold is zero-based.
3. The component pack $WEBJEV_WORK/packs/general (items.pkl, origins.jsonl.gz, manifest.json).
"""
from __future__ import annotations

import json
from pathlib import Path
import pickle
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import WORK, require, use_upstream, write_pack  # noqa: E402

GENERAL = WORK / "general"
EXPORTS = GENERAL / "exports"


def export_rows(rows, path: Path) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq
    schema = pa.schema([("id", pa.string()), ("task", pa.string()), ("context", pa.large_string()),
                        ("questions", pa.list_(pa.struct([("text", pa.string()), ("options", pa.list_(pa.string())),
                                                          ("gold", pa.int32())])))])
    path.parent.mkdir(parents=True, exist_ok=True)
    writer, batch, questions = pq.ParquetWriter(str(path), schema, compression="zstd"), [], 0
    for i, e in enumerate(rows):
        qs = [{"text": q.text, "options": list(q.options), "gold": int(q.gold)} for q in e.qs]
        questions += len(qs)
        batch.append({"id": f"{path.stem}:{i:09d}", "task": e.task, "context": e.context, "questions": qs})
        if len(batch) >= 5000:
            writer.write_table(pa.Table.from_pylist(batch, schema=schema))
            batch = []
    if batch:
        writer.write_table(pa.Table.from_pylist(batch, schema=schema))
    writer.close()
    return {"rows": len(rows), "questions": questions}


def main() -> None:
    upstream = use_upstream(GENERAL / "upstream")
    from decider import data as D
    train, evals = D.load_cache(str(upstream / "data" / "mixture_full.pkl"))
    _, probes = D.load_cache(str(upstream / "data" / "probes.pkl"))
    with (GENERAL / "raw" / "items.pkl").open("rb") as stream:
        items = pickle.load(stream)
    require(len(items) == len(train), "tokenized items and mixture rows are not aligned")

    empty = [i for i, e in enumerate(train) if not e.qs]
    for i in empty:
        require(not items[i]["slots"] and not items[i]["golds"], f"row {i} has no question but has answer slots")
    keep = [i for i, e in enumerate(train) if e.qs]
    EXPORTS.mkdir(parents=True, exist_ok=True)
    with (EXPORTS / "excluded-empty-questions.jsonl").open("w") as stream:
        for i in empty:
            stream.write(json.dumps({"source_index": i, "task": train[i].task, "context": train[i].context},
                                    ensure_ascii=False) + "\n")
    clean = [train[i] for i in keep]
    with (EXPORTS / "mixture_full.pkl").open("wb") as stream:
        pickle.dump((clean, evals), stream, protocol=5)

    report = {"removed_zero_question_rows": len(empty), "train": export_rows(clean, EXPORTS / "train.parquet"),
              "eval": {name: export_rows(rows, EXPORTS / "eval" / f"{name}.parquet") for name, rows in evals.items() if rows},
              "probes": {name: export_rows(rows, EXPORTS / "probes" / f"{name}.parquet") for name, rows in probes.items() if rows}}
    (EXPORTS / "export.json").write_text(json.dumps(report, indent=2) + "\n")

    kept_items, origins = [], []
    for i in keep:
        item = items[i]
        require(item["ex_id"] == i and item["slots"], f"item {i} is misaligned or unsupervised")
        kept_items.append(item)
        origins.append({"source": "general", "task": item["task"], "source_index": i})
    del items, train, clean
    manifest = write_pack(WORK / "packs" / "general", kept_items, origins,
                          {"component": "general", "upstream_mixture": "full, seed 6",
                           "tokenization": json.loads((GENERAL / "raw" / "items.pkl.json").read_text())["settings"],
                           "removed_zero_question_rows": len(empty)})
    print(json.dumps({k: manifest[k] for k in ("items", "questions", "tokens")}), flush=True)


if __name__ == "__main__":
    main()
