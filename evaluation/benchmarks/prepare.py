#!/usr/bin/env python3
"""Convert the 8 benchmark sources (fetched by fetch_sources.sh) into one item format (see common.py).

    python prepare.py            ->  $EVAL_ROOT/bench/<benchmark>.jsonl  and  $EVAL_ROOT/bench/_manifest.json

A question with a single option is not evaluation data (every model gets it right), so it would be dropped for every
model alike; none of these 8 benchmarks has one. Each output file is compared with the SHA-256 of the input file our
reported results were computed on.
"""
import collections
import hashlib
import json
import os

import pandas as pd

from common import BENCHES, ROOT, gold_label, n_options

SRC = os.environ.get("BENCH_SOURCES", f"{ROOT}/sources")
KEV_EVALS = f"{SRC}/kev/evals"
TYPED = f"{SRC}/typed-decisions/test.parquet"

# SHA-256 of the prepared files behind the published results (items, questions)
EXPECTED = {
    "jevbench_public": ("6777163772ef5a999c6d1468bdb17aa31eb89812c63031e7cd3dbef496d56d87", 231, 231),
    "nimble_eval324": ("e48e0d8470a0e8d28d2f7808e6cac765492ce068b7ebd60fdd0e4e0358c69ec2", 324, 324),
    "kev_ext_semif": ("fec21babc44b4d68b2bb12f14e3f40929d188e315e24777104a3fcf3cb774d95", 252, 252),
    "kev_ext_scienthoon": ("c71c9f98c12d2ba1773c485a4f560fd616765f547f1efa84426d31bcc84bb218", 291, 873),
    "kev_transfer_v4_dev": ("cab2136a527cd26181175336202e9935ec830a4b8ad1118b25e719877376c3be", 764, 764),
    "kev_decision_v7_dev": ("9002e54b391520d36d11a61376425bbcc40069c2822cc86e4a1500c550d08e1e", 1204, 1468),
    "kev_ext_mmlupro": ("8562e586bc3fa792e74289a36cc21f5b940d14d5bca2b5c62f4cc099ac631f09", 1000, 1000),
    "typed_decisions_test": ("4d653296de1c607edb859cf5e02c85b46ae2bd51366372115a4d02b0c76b608b", 400, 2000),
}

out, dropped = {}, collections.Counter()


def add(bench, subset, id_, state, questions, gold):
    qs = {k: {kk: vv for kk, vv in q.items() if kk in ("type", "instructions", "criteria")} for k, q in questions.items()}
    keep = {k: g for k, g in gold.items() if n_options(qs[k]) >= 2}
    dropped[bench] += len(gold) - len(keep)
    if not keep:
        return
    qs = {k: qs[k] for k in keep}
    out.setdefault(bench, []).append({"bench": bench, "subset": subset, "id": str(id_), "state": state, "questions": qs,
                                      "gold": {k: gold_label(qs[k], g) for k, g in keep.items()}})


# 1 JevBench public items, as the benchmark's own TypeSafe adapter sends them (one question, key "decision")
for tier, fn in (("easy", "easy"), ("standard", "original"), ("hard", "hard")):
    for line in open(f"{SRC}/jevbench/datasets/public/{fn}.jsonl"):
        r = json.loads(line)
        add("jevbench_public", tier, r["id"], r["state"], {"decision": r["question"]}, {"decision": r["expected"]})
# 2 Nimble eval split
for line in open(f"{SRC}/nimble/data/eval.jsonl"):
    r = json.loads(line)
    q = r["input"]["questions"]
    k = next(iter(q))
    add("nimble_eval324", q[k]["type"], r["id"], r["input"]["state"], q, {k: r["reference"]["target"]})
# 3-7 kev frozen suites (the development splits are the ones with published Jev numbers)
for bench, path in (("kev_transfer_v4_dev", "v4/transfer-v4/development.jsonl"), ("kev_decision_v7_dev", "v7/decision-v7/development.jsonl"),
                    ("kev_ext_semif", "external/semif-v1/development.jsonl"), ("kev_ext_scienthoon", "external/scienthoon-v1/development.jsonl"),
                    ("kev_ext_mmlupro", "external/ekzhang-mmlupro-v1/records.jsonl")):
    for n, line in enumerate(open(f"{KEV_EVALS}/{path}")):
        r = json.loads(line)
        meta = r.get("_meta", {})
        add(bench, meta.get("source", "all"), meta.get("id", n), r["state"], r["questions"], {k: q["label"] for k, q in r["questions"].items()})
# 8 typed-decisions test split
for _, r in pd.read_parquet(TYPED).iterrows():
    qs, g = json.loads(r["questions"]), json.loads(r["gold"])
    add("typed_decisions_test", r["workflow"], r["id"], json.loads(r["state"]), qs, {k: g[k]["label"] for k in qs if k in g})

os.makedirs(f"{ROOT}/bench", exist_ok=True)
manifest, all_match = {}, True
for b in BENCHES:
    rows = out.get(b, [])
    path = f"{ROOT}/bench/{b}.jsonl"
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    types = collections.Counter(q["type"] for r in rows for q in r["questions"].values())
    questions = sum(len(r["gold"]) for r in rows)
    match = sha == EXPECTED[b][0]
    all_match &= match
    manifest[b] = {"items": len(rows), "questions": questions, "types": dict(types), "subsets": sorted({r["subset"] for r in rows}),
                   "dropped_single_option": dropped[b], "sha256": sha, "matches_evaluated_input": match}
    print(f"{b:22s} items {len(rows):5d} questions {questions:5d} types {dict(types)}  "
          f"{'identical to the evaluated input' if match else 'DIFFERS from the evaluated input (expected %d items, %d questions)' % EXPECTED[b][1:]}")
json.dump(manifest, open(f"{ROOT}/bench/_manifest.json", "w"), ensure_ascii=False, indent=1)
print("all 8 inputs identical to the evaluated ones" if all_match else "some inputs differ from the evaluated ones; scores are not comparable")
