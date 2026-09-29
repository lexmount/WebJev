#!/usr/bin/env python3
"""Tokenize the general mixture with the upstream recipe settings, in parallel, with the serial result.

Settings (upstream `make_items`): context budget 16,384 tokens, at most 255 options per question, an added
"none of the above" option with probability 0.1 where eligible, schema-first layout with probability 0.5,
one RNG seeded 0. Options are shuffled and the gold index follows them.

The RNG sequence is planned serially per 1,000-row chunk, so the parallel output equals a serial run; every
chunk's final RNG state is checked, and five chunks are compared field by field with upstream `make_items`.

    python general/tokenize_items.py --tokenizer <base model dir or tokenizer dir> [--workers 16]
      -> $WEBJEV_WORK/general/raw/items.pkl (+ items.pkl.json)
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path
import pickle
import random
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import WORK, require, sha256_file, use_upstream  # noqa: E402

GENERAL = WORK / "general"
SETTINGS = dict(max_ctx=16384, none_prob=0.1, max_options=255, schema_first_prob=0.5, seed=0)
TRAIN = TOK = LABEL_POOL = None


def plan(train, chunk=1000):
    from decider.data.augment import none_augment
    from decider.prompt import _select
    rng, jobs = random.Random(SETTINGS["seed"]), []
    for start in range(0, len(train), chunk):
        end, begin = min(len(train), start + chunk), rng.getstate()
        for e in train[start:end]:
            e = none_augment(e, rng, SETTINGS["none_prob"], LABEL_POOL)
            rng.random()  # the layout draw; both layouts consume the same _select sequence
            for q in e.qs:
                _select(q, rng, SETTINGS["max_options"])
        jobs.append((start, end, begin, rng.getstate()))
    return jobs


def build_chunk(job):
    from decider.data.augment import none_augment
    from decider.prompt import build
    start, end, begin, expected = job
    rng, out = random.Random(), []
    rng.setstate(begin)
    for i in range(start, end):
        e = TRAIN[i]
        item = build(none_augment(e, rng, SETTINGS["none_prob"], LABEL_POOL), TOK, rng,
                     max_options=SETTINGS["max_options"], max_ctx_tokens=SETTINGS["max_ctx"],
                     layout="schema_first" if rng.random() < SETTINGS["schema_first_prob"] else "state_first")
        item["task"], item["ex_id"] = e.task, i
        item["ids"] = np.asarray(item["ids"], dtype=np.int32)
        out.append(item)
    require(rng.getstate() == expected, f"RNG state mismatch in rows {start}-{end}")
    return out


def same(a, b) -> bool:
    return a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) if k == "ids" else a[k] == b[k] for k in a)


def compare_with_upstream(items, jobs) -> dict:
    import decider.train as upstream
    original = upstream.build_label_pool
    upstream.build_label_pool = lambda _: LABEL_POOL
    picks = sorted({0, len(jobs) // 4, len(jobs) // 2, 3 * len(jobs) // 4, len(jobs) - 1})
    checked = 0
    try:
        for j in picks:
            start, end, begin, expected = jobs[j]
            rng = random.Random()
            rng.setstate(begin)
            serial = upstream.make_items(TRAIN[start:end], TOK, rng, SETTINGS["max_ctx"], SETTINGS["none_prob"],
                                         SETTINGS["max_options"], SETTINGS["schema_first_prob"])
            require(rng.getstate() == expected, "serial RNG state differs")
            for k, item in enumerate(serial):
                item["ex_id"] += start
                item["ids"] = np.asarray(item["ids"], dtype=np.int32)
                require(same(items[start + k], item), f"parallel item {start + k} differs from upstream make_items")
            checked += len(serial)
    finally:
        upstream.build_label_pool = original
    return {"items_compared": checked, "chunks_compared": picks, "chunks": len(jobs)}


def main() -> None:
    global TRAIN, TOK, LABEL_POOL
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", required=True, help="directory with the base model tokenizer files")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    upstream = use_upstream(GENERAL / "upstream")
    from transformers import AutoTokenizer
    from decider import data as D
    from decider.data.augment import build_label_pool
    started = time.time()
    TRAIN, _ = D.load_cache(str(upstream / "data" / "mixture_full.pkl"))
    TOK = AutoTokenizer.from_pretrained(a.tokenizer)
    LABEL_POOL = build_label_pool(TRAIN)
    jobs = plan(TRAIN)
    items = []
    with mp.get_context("fork").Pool(a.workers) as pool:
        for result in pool.imap(build_chunk, jobs, chunksize=1):
            items.extend(result)
    require(len(items) == len(TRAIN), "tokenized item count differs from the mixture")
    check = compare_with_upstream(items, jobs)
    out = GENERAL / "raw" / "items.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as stream:
        pickle.dump(items, stream, protocol=5)
    meta = dict(items=len(items), questions=sum(len(i["slots"]) for i in items), tokens=int(sum(len(i["ids"]) for i in items)),
                sha256=sha256_file(out), settings=SETTINGS, equivalence=check,
                seconds=round(time.time() - started, 1))
    Path(str(out) + ".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
