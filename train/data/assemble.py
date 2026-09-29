#!/usr/bin/env python3
"""Mix all component packs into the single training file the trainer reads.

    python data/assemble.py [--components general web open_jev knowledge_mcqa] [--seed 20260927]
      -> $WEBJEV_WORK/mixture/items.pkl, items.pkl.json (counts and SHA-256), origins.jsonl.gz

Every item of the general and web components is kept as built (the upstream recipe repeats some rows on
purpose). An item of a later component is dropped when the same tokenized input already exists in the
mixture, and every copy of an input is dropped when the copies disagree on the answer. The result is shuffled
once with a fixed seed; the trainer then forms length-bucketed batches from it.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import pickle
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import WORK, canonical, prompt_identity, read_pack, require, sha256_file, valid_item  # noqa: E402

KEEP_ALL = {"general", "web"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--components", nargs="+", default=["general", "web", "open_jev", "knowledge_mcqa"])
    ap.add_argument("--seed", type=int, default=20260927)
    ap.add_argument("--out", type=Path, default=WORK / "mixture")
    a = ap.parse_args()
    require(not (a.out / "items.pkl").exists(), f"output already exists: {a.out}")
    combined, keys, stats = [], {}, Counter()
    conflicts: set[str] = set()
    inputs = {}
    for name in a.components:
        manifest, items, origins = read_pack(WORK / "packs" / name)
        inputs[name] = {k: manifest[k] for k in ("items", "questions", "tokens", "items_sha256")}
        for item, origin in zip(items, origins):
            origin["component"] = name
            if name in KEEP_ALL:
                keys.setdefault(prompt_identity(item), (len(combined), tuple(item["golds"])))
                combined.append((item, origin))
                continue
            key, label = prompt_identity(item), tuple(item["golds"])
            if key in conflicts:
                stats["conflicting_label_excluded"] += 1
            elif key not in keys:
                keys[key] = (len(combined), label)
                combined.append((item, origin))
            elif combined[keys[key][0]][1]["component"] in KEEP_ALL:
                stats[f"already_in_mixture_excluded:{name}"] += 1
            elif keys[key][1] == label:
                stats[f"duplicate_input_excluded:{name}"] += 1
            else:
                combined[keys.pop(key)[0]] = None
                conflicts.add(key)
                stats["conflicting_label_excluded"] += 2
        del items, origins
    combined = [entry for entry in combined if entry is not None]
    random.Random(a.seed).shuffle(combined)
    a.out.mkdir(parents=True, exist_ok=True)
    items, counts = [], {name: Counter() for name in a.components}
    with gzip.open(a.out / "origins.jsonl.gz", "wt", encoding="utf-8") as stream:
        for i, (item, origin) in enumerate(combined):
            require(valid_item(item), f"invalid item at {i}")
            item["ex_id"] = i
            origin["index"] = i
            items.append(item)
            c = counts[origin["component"]]
            c["items"] += 1
            c["questions"] += len(item["slots"])
            c["tokens"] += len(item["ids"])
            stream.write(canonical(origin) + "\n")
    with (a.out / "items.pkl").open("xb") as stream:
        pickle.dump(items, stream, protocol=5)
    total = {k: sum(c[k] for c in counts.values()) for k in ("items", "questions", "tokens")}
    manifest = {**total, "sha256": sha256_file(a.out / "items.pkl"), "seed": a.seed,
                "max_sequence": max(len(it["ids"]) for it in items),
                "components": {name: dict(c) for name, c in counts.items()}, "component_inputs": inputs,
                "cross_component_dedup": dict(stats), "origins_sha256": sha256_file(a.out / "origins.jsonl.gz")}
    (a.out / "items.pkl.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({**total, "components": manifest["components"]}), flush=True)


if __name__ == "__main__":
    main()
