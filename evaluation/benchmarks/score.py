#!/usr/bin/env python3
"""Score cached answers on the 8 benchmarks. The same rule for every model; no LLM judge.

    python score.py webjev jev                       # names under $EVAL_ROOT/pred
    python score.py results/webjev-35b-a3b           # or any directory of <benchmark>.jsonl[.gz]
    python score.py webjev jev --json scores.json

Rules:
- a question is correct when the committed label equals the gold label (exact match):
  choice -> the returned `choice`; noul -> "yes" iff the returned probability >= 0.5; score -> the most probable level;
- an unanswered or failed item counts as wrong for all its questions;
- a question with fewer than 2 options is not scored (none exist in these 8 benchmarks);
- accuracy = correct questions / questions; "mean of 8" = unweighted mean of the 8 accuracies.
With exactly two models, each benchmark also gets the paired comparison on the same questions (how many only one model
got right) and a two-sided exact McNemar p-value on those discordant pairs.
"""
import argparse
import json
import math
import os

from common import BENCHES, ROOT, TITLES, load_bench, n_options, pred_label, read_jsonl


def load_predictions(ref, bench):
    base = ref if os.path.isdir(ref) else f"{ROOT}/pred/{ref}"
    for suffix in (".jsonl", ".jsonl.gz"):
        path = f"{base}/{bench}{suffix}"
        if os.path.exists(path):
            return {r["id"]: r for r in read_jsonl(path) if "error" not in r and r.get("answers")}
    return {}


def mcnemar_p(k, n):
    """Two-sided exact binomial test of k successes in n trials at p = 0.5."""
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n)


ap = argparse.ArgumentParser()
ap.add_argument("models", nargs="+", help="prediction names (under $EVAL_ROOT/pred) or directories")
ap.add_argument("--json", default="", help="write the full scores here")
a = ap.parse_args()
labels = [os.path.basename(os.path.normpath(m)) for m in a.models]

scores = {label: {} for label in labels}
paired = {}
for bench in BENCHES:
    items = load_bench(bench)
    preds = [load_predictions(m, bench) for m in a.models]
    per_q = {label: [] for label in labels}
    for it in items:
        for qid, gold in it["gold"].items():
            q = it["questions"][qid]
            if n_options(q) < 2:
                continue
            for label, p in zip(labels, preds):
                row = p.get(it["id"])
                answer = (row["answers"] or {}).get(qid) if row else None
                per_q[label].append(str(pred_label(q, answer)) == str(gold) if row else False)
    for label, p in zip(labels, preds):
        ok = per_q[label]
        scores[label][bench] = {"correct": sum(ok), "n": len(ok), "accuracy": sum(ok) / len(ok) if ok else 0.0,
                                "unanswered_items": sum(1 for it in items if it["id"] not in p)}
    if len(labels) == 2:
        x, y = per_q[labels[0]], per_q[labels[1]]
        only_x = sum(1 for i, j in zip(x, y) if i and not j)
        only_y = sum(1 for i, j in zip(x, y) if j and not i)
        paired[bench] = {f"only_{labels[0]}": only_x, f"only_{labels[1]}": only_y, "p_mcnemar": mcnemar_p(only_x, only_x + only_y)}

for label in labels:
    s = scores[label]
    s["mean_of_8"] = sum(s[b]["accuracy"] for b in BENCHES) / len(BENCHES)
    s["micro"] = sum(s[b]["correct"] for b in BENCHES) / sum(s[b]["n"] for b in BENCHES)

width = max(12, *(len(label) for label in labels))
print(f"{'benchmark':28s} {'questions':>9s} " + " ".join(f"{label:>{width + 10}s}" for label in labels)
      + ("   paired: only-first / only-second, p" if paired else ""))
for bench in BENCHES:
    n = scores[labels[0]][bench]["n"]
    cells = " ".join(f"{scores[l][bench]['correct']:>{width - 1}d} ({100 * scores[l][bench]['accuracy']:6.2f}%)" for l in labels)
    extra = ""
    if paired:
        pb = paired[bench]
        extra = f"   {pb[f'only_{labels[0]}']:>4d} / {pb[f'only_{labels[1]}']:<4d} p={pb['p_mcnemar']:.2g}"
    print(f"{TITLES[bench]:28s} {n:>9,d} {cells}{extra}")
print(f"{'mean of 8':28s} {'':>9s} " + " ".join(f"{'':>{width - 1}s} ({100 * scores[l]['mean_of_8']:6.2f}%)" for l in labels))
missing = {l: sum(scores[l][b]["unanswered_items"] for b in BENCHES) for l in labels}
if any(missing.values()):
    print("unanswered items (scored as wrong): " + ", ".join(f"{l} {n}" for l, n in missing.items()))
if a.json:
    json.dump({"scores": scores, "paired": paired}, open(a.json, "w"), indent=1)
    print("scores ->", a.json)
