#!/usr/bin/env python3
"""Component 2: our live-web decision data, downloaded from Hugging Face and tokenized.

The rows are published as the dataset https://huggingface.co/datasets/Lexmount/WebJev (see its dataset card for how
they were collected, labeled and audited). Each row is one decision on a real web page: `context` (the page state
the browser agent observed), `question` (what to decide: the next operation, or which element, input field or
dropdown option to act on), `options`, the zero-based `gold` index, and `question_type`. The dataset has two
configurations, `action_prediction` (next operation) and `element_grounding` (target element).

Tokenization matches the rest of the mixture: the upstream prompt builder, state-first layout, at most 255
options, a 16,384-token context budget, and one RNG seeded 0. Rows are visited in a fixed order (the
action_prediction shards, then the element_grounding shards, each in file order). No row is changed or dropped.
The RNG only shuffles the options of each question, so a rebuild contains exactly the same decisions as our run;
only the option order within a question can differ.

    python data/web.py --tokenizer <base model dir> [--source <local copy of the dataset>]   ->   $WEBJEV_WORK/packs/web

Environment overrides for the download: WEBJEV_WEB_REPO (default Lexmount/WebJev), WEBJEV_WEB_REPO_TYPE
(default dataset), WEBJEV_WEB_PATH (default data).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (WORK, Tokenizer, decision_key, iter_jsonl, require, stem_key,  # noqa: E402
                    use_upstream, write_pack)

REPO = os.environ.get("WEBJEV_WEB_REPO", "Lexmount/WebJev")
REPO_TYPE = os.environ.get("WEBJEV_WEB_REPO_TYPE", "dataset")
SUBDIR = os.environ.get("WEBJEV_WEB_PATH", "data")
CORE = ("context", "question", "options", "gold", "question_type")
CONFIGS = ("action_prediction", "element_grounding")


def download() -> Path:
    from huggingface_hub import snapshot_download
    local = snapshot_download(repo_id=REPO, repo_type=REPO_TYPE, allow_patterns=[f"{SUBDIR}/**"],
                              local_dir=WORK / "web" / "download")
    return Path(local) / SUBDIR


def rows(source: Path):
    """Yield the rows in a fixed order: a JSONL(.gz) file as it is, or the parquet shards of each configuration."""
    if source.is_file():
        yield from (row for _, row in iter_jsonl(source))
        return
    import pyarrow.parquet as pq
    shards = [f for config in CONFIGS for f in sorted((source / config).glob("*.parquet"))]
    require(shards, f"no parquet shards under {source}/{{{','.join(CONFIGS)}}}")
    for shard in shards:
        yield from pq.read_table(shard, columns=list(CORE)).to_pylist()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", required=True, type=Path, help="base model directory or tokenizer.json")
    ap.add_argument("--source", type=Path, help="a local copy of the dataset (file or directory); default: download")
    a = ap.parse_args()
    use_upstream()
    from decider.data.core import Example, Q
    from decider.prompt import build

    source = a.source or download()
    tok, rng = Tokenizer(a.tokenizer), random.Random(0)
    items, origins, problems = [], [], {}
    for i, r in enumerate(rows(source)):
        options, gold = list(r["options"]), int(r["gold"])
        require(2 <= len(options) <= 255 and 0 <= gold < len(options), f"row {i}: invalid options or gold")
        item = build(Example(r["context"], [Q(r["question"], options, gold)], "web"), tok, rng,
                     max_options=255, max_ctx_tokens=16384, layout="state_first")
        if item["nopts"][0] != len(options) or item["golds"][0] < 0 or item["perms"][0][item["golds"][0]] != gold:
            problems[i] = "option sampling or gold lost"
            continue
        task = "web_" + (r.get("question_type") or r.get("task") or "decision")
        item.update(task=task, ex_id=i)
        items.append(item)
        origins.append({"source": "web", "row_index": i, "task": task, "stem_key": stem_key(r["context"]),
                        "decision_key": decision_key(r["context"], r["question"], options)})
    require(not problems, f"rows that do not build cleanly: {list(problems.items())[:5]}")
    manifest = write_pack(WORK / "packs" / "web", items, origins,
                          {"component": "web", "source": f"{REPO}/{SUBDIR}", "layout": "state_first",
                           "max_options": 255, "max_ctx_tokens": 16384, "rng": "random.Random(0); action_prediction then element_grounding, file order"})
    print(json.dumps({k: manifest[k] for k in ("items", "questions", "tokens")}), flush=True)


if __name__ == "__main__":
    main()
