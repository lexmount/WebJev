#!/usr/bin/env python3
"""Component 3: typed decisions from the public Open-Jev datasets.

Sources (train splits only; pinned in download_sources.py):
  - ZefanCai/Open-Jev: eleven configurations (raw/<config>/train.jsonl.gz);
  - ZefanCai/Open-Jev-v1.1: community-hard-mix-v2 (WANLI decisions and community policy scenarios).

Admission:
  - Families: the thirteen *-control-v1 families (browser, reasoning, entity alignment, IR, phone, amount,
    email, context retention, citation, mailroom, customer, sponsor segment, silent failure), seven game and
    control families of the browser/drone expansion (drone control, painting geometry, snake, tic-tac-toe, tile
    platformer, T-Rex runner, ViZDoom basic), WANLI decisions and community-diversity-v2. workflow-controls-v1 is
    excluded: a single visible workflow question does not fully specify the action priority.
  - Hard one-hot labels only; soft-label rows are skipped, never coerced to one answer.
  - Labels re-derived independently before use: browser-control and reasoning-control from the visible rules
    (rule_audit.py, no source program is executed); community rows from their visible numbered rules (the
    dataset authors' checker, pinned); WANLI rows against the original human-labeled WANLI train file.
  - The source datasets' own held-out splits are respected: a train row that shares an ID, group, seed, premise,
    scenario family, semantic context or source instance with an Open-Jev held-out split (calibration,
    validation, test, OOD) or with the WANLI test set is skipped. Evaluation items are excluded by exact
    normalized match (eval_guard.py).
  - One training view per original question. A Score question becomes, by a fixed hash, either the original
    multi-level question, the yes/no row of its correct level, or the yes/no row of an adjacent wrong level.
  - Duplicates: repeated IDs, identical visible decisions (every copy is dropped if their labels disagree),
    identical tokenized prompts, and decisions already present in the general or web component.
  - Rendering: upstream prompt builder, state-first, all options, contexts up to 16,384 tokens.

    python data/open_jev.py --sources $WEBJEV_WORK/sources --tokenizer <base model dir> [--bench-dir DIR]
      -> $WEBJEV_WORK/packs/open_jev
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import random
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import (WORK, Tokenizer, context_key, decision_key, digest, normal,  # noqa: E402
                    prompt_identity, require, stem_key, use_upstream, write_pack)
from eval_guard import load_guard, summary  # noqa: E402
from index_components import Seen  # noqa: E402
from rule_audit import browser_gold, reasoning_gold  # noqa: E402

SEED = 20260927
ORIGINAL_CONFIGS = [
    "amount-extraction-control-v1", "browser-drone-expansion-v1-redistributable", "citation-control-v1",
    "context-retention-control-v1", "email-selection-control-v1", "entity-alignment-control-v1", "ir-control-v1",
    "mailroom-control-v1", "phone-extraction-control-v1", "silent-failure-control-v1", "sponsor-segment-control-v1",
]
V11_CONFIG = "community-hard-mix-v2-redistributable"
HELDOUT_SPLITS = ("calibration", "validation", "test", "ood")
GAME_FAMILIES = frozenset({
    "drone-control-v1", "painting-geometry-v1", "snake-v1", "tic_tac_toe-v1",
    "tile_platformer-v1", "trex_runner-v1", "vizdoom-basic-v1",
})
ADMITTED_FAMILIES = frozenset({
    "amount-extraction-control-v1", "browser-control-v1", "citation-control-v1",
    "context-retention-control-v1", "customer-control-v1", "email-selection-control-v1",
    "entity-alignment-control-v1", "ir-control-v1", "mailroom-control-v1",
    "phone-extraction-control-v1", "reasoning-control-v1", "silent-failure-control-v1",
    "sponsor-segment-control-v1", "community-diversity-v2", "wanli-decisions-v1",
}) | GAME_FAMILIES
WANLI_LABEL_TEXT = {"entailment": "The premise supports the claim.",
                    "contradiction": "The premise contradicts the claim.",
                    "neutral": "The premise leaves the claim undetermined."}


def onehot_gold(row: dict) -> int | None:
    target, options = row.get("target"), row.get("options")
    if not isinstance(target, list) or not isinstance(options, list):
        return None
    if not 2 <= len(options) <= 255 or len(target) != len(options):
        return None
    if not all(isinstance(v, str) and v for v in options) or len(set(options)) != len(options):
        return None
    if sorted(target) != [0.0] * (len(target) - 1) + [1.0]:
        return None
    return target.index(1.0)


def score_view(source_id: str, question: str, options: list[str], gold: int, s1):
    """One deterministic view per original Score question: original, correct-level yes/no, or adjacent-level yes/no."""
    mode = int(digest([SEED, source_id, "view"]), 16) % 3
    if mode == 0:
        return question, options, gold, "original"
    proposed = gold
    if mode == 2:
        adjacent = [n for n in (gold - 1, gold + 1) if 0 <= n < len(options)]
        proposed = adjacent[int(digest([SEED, source_id, "adjacent"]), 16) % len(adjacent)]
    q, opts = s1.isolated_rows(question, options)[proposed]
    return q, opts, int(mode == 1), "isolated_positive" if mode == 1 else "isolated_negative"


def render(row: dict, s1):
    gold = onehot_gold(row)
    require(gold is not None, "non-hard or invalid row")
    context = s1.render_state(row["state"])
    spec = {"type": row["kind"], "instructions": row["question"]}
    if row["kind"] == "choice":
        spec["criteria"] = row["options"]
    elif row["kind"] == "score":
        spec["criteria"] = [s1.strip_level_number(o) for o in row["options"]]
    elif row["kind"] != "noul" or row["options"] != ["no", "yes"]:
        raise ValueError("unsupported Open-Jev decision type")
    rendered = s1.render_question(spec)
    return context, rendered["question"], rendered["options"], gold


def gz_rows(path: Path):
    with gzip.open(path, "rt") as stream:
        for line in stream:
            yield json.loads(line)


def load_heldouts(src: Path):
    """Keys of the Open-Jev held-out splits for the WANLI, community and game families."""
    v11 = defaultdict(lambda: defaultdict(set))
    v11_premises = set()
    for split in HELDOUT_SPLITS:
        for row in gz_rows(src / "open-jev-v1.1" / "raw" / V11_CONFIG / f"{split}.jsonl.gz"):
            family = row["source"].split("/")[0]
            md = row["metadata"]
            if family == "wanli-decisions-v1":
                for key, value in (("raw_id", row["id"]), ("group_id", row["group_id"]),
                                   ("source_seed", md["source_seed_id"]), ("exact_premise", row["state"]["premise"])):
                    v11[family][key].add(value)
                v11_premises.add(normal(row["state"]["premise"]))
            elif family == "community-diversity-v2":
                for key, value in (("raw_id", row["id"]), ("group_id", row["group_id"]),
                                   ("source_family", md["scenario_family"]),
                                   ("semantic_context", md["semantic_context_sha256"]),
                                   ("source_instance", md["source_instance_id"])):
                    v11[family][key].add(value)
    games = defaultdict(lambda: defaultdict(set))
    for split in HELDOUT_SPLITS:
        for row in gz_rows(src / "open-jev" / "raw" / "browser-drone-expansion-v1-redistributable" / f"{split}.jsonl.gz"):
            family = row["source"].split("/")[0]
            if family in GAME_FAMILIES:
                games[family]["raw_id"].add(row["id"])
                games[family]["group_id"].add(row["group_id"])
                games[family]["state"].add(context_key(row["state"]))
    return v11, v11_premises, games


def check_v11_train_row(row: dict, v11) -> None:
    """The v1.1 train split must be disjoint from its held-out splits; an overlap is a source error."""
    family = row["source"].split("/")[0]
    if family not in ("wanli-decisions-v1", "community-diversity-v2"):
        return
    md = row["metadata"]
    checks = {"raw_id": row["id"], "group_id": row["group_id"]}
    if family == "wanli-decisions-v1":
        checks.update(source_seed=md["source_seed_id"], exact_premise=row["state"]["premise"])
    else:
        checks.update(source_family=md["scenario_family"], semantic_context=md["semantic_context_sha256"],
                      source_instance=md["source_instance_id"])
    for key, value in checks.items():
        require(value not in v11[family][key], f"Open-Jev v1.1 {family} train/held-out overlap on {key}: {row['id']}")


def load_wanli(src: Path):
    original = {}
    for line in (src / "wanli" / "train.jsonl").open():
        row = json.loads(line)
        require(str(row["id"]) not in original, "duplicate WANLI ID")
        original[str(row["id"])] = row
    test_seeds, test_premises = set(), set()
    for line in (src / "wanli" / "test.jsonl").open():
        row = json.loads(line)
        test_seeds.add(str(row["pairID"]))
        test_premises.add(normal(row["premise"]))
    require(len(test_seeds) > 0, "WANLI test split is empty")
    return original, test_seeds, test_premises


def wanli_heldout(row: dict, original: dict, test_seeds: set, test_premises: set, v11_premises: set) -> bool:
    """Verify the row against the original human label; True when its seed or premise is held out."""
    source = original.get(row["metadata"]["provenance"]["original_id"])
    require(source is not None, f"WANLI original ID missing: {row['id']}")
    require(row["state"]["premise"] == source["premise"] and row["state"]["claim"] == source["hypothesis"]
            and row["metadata"]["source_seed_id"] == str(source["pairID"]), f"WANLI text or seed mismatch: {row['id']}")
    require(row["options"][onehot_gold(row)] == WANLI_LABEL_TEXT[source["gold"]], f"WANLI label mismatch: {row['id']}")
    premise = normal(source["premise"])
    return str(source["pairID"]) in test_seeds or premise in test_premises or premise in v11_premises


def collect(src: Path, guard, seen: Seen, s1, community_target, community_audit):
    v11, v11_premises, games = load_heldouts(src)
    wanli, wanli_test_seeds, wanli_test_premises = load_wanli(src)
    seen_stems = seen.keys("stem")
    selected, owner, conflicted = {}, {}, set()
    counts, community_rows = Counter(), []

    def reject(reason: str) -> None:
        counts["excluded:" + reason] += 1

    sources = [(config, src / "open-jev" / "raw" / config / "train.jsonl.gz") for config in ORIGINAL_CONFIGS]
    sources.append(("open-jev-v1.1", src / "open-jev-v1.1" / "raw" / V11_CONFIG / "train.jsonl.gz"))
    for config, path in sources:
        for row in gz_rows(path):
            counts["read:" + config] += 1
            require(row.get("split") == "train", f"non-train row in {path}")
            family = row["source"].split("/")[0]
            if family not in ADMITTED_FAMILIES:
                reject("family_not_admitted")
                continue
            if config == "open-jev-v1.1":
                check_v11_train_row(row, v11)
            if onehot_gold(row) is None:
                reject("soft_or_invalid_label")
                continue
            if family in GAME_FAMILIES and (row["id"] in games[family]["raw_id"] or row["group_id"] in games[family]["group_id"]
                                            or context_key(row["state"]) in games[family]["state"]):
                reject("open_jev_heldout_overlap")
                continue
            if family == "wanli-decisions-v1":
                if wanli_heldout(row, wanli, wanli_test_seeds, wanli_test_premises, v11_premises):
                    reject("wanli_heldout_overlap")
                    continue
            elif family == "community-diversity-v2":
                require(row["target"] == community_target(row), f"community label disagrees with its rules: {row['id']}")
                community_rows.append(row)
            if family in ("browser-control-v1", "reasoning-control-v1"):
                derived = browser_gold(row) if family == "browser-control-v1" else reasoning_gold(row)
                require(derived == onehot_gold(row), f"independent label disagreement: {row['id']}")
                counts["labels_rederived:" + family] += 1
            if row["id"] in selected:
                reject("duplicate_id")
                continue
            context, question, options, gold = render(row, s1)
            source_context, rendered_context, stem = context_key(row["state"]), context_key(context), stem_key(context)
            if source_context in guard.contexts or rendered_context in guard.contexts or stem in guard.stems:
                reject("evaluation_context")
                continue
            if normal(question) in guard.questions:
                reject("evaluation_question")
                continue
            if stem in seen_stems:
                reject("context_already_in_mixture")
                continue
            if seen.has("decision", decision_key(context, question, options)):
                reject("decision_already_in_mixture")
                continue
            if row["kind"] == "score":
                question, options, gold, view = score_view(row["id"], question, options, gold, s1)
            else:
                view = "original"
            visible = decision_key(context, question, options)
            if seen.has("decision", visible):
                reject("decision_already_in_mixture")
                continue
            label = normal(options[gold])
            if visible in conflicted:
                reject("conflicting_visible_input")
                continue
            if visible in owner:
                first, old_label = owner[visible]
                if label != old_label:
                    conflicted.add(visible)
                    selected.pop(first, None)
                    counts["excluded:conflicting_visible_input"] += 2
                else:
                    reject("duplicate_visible_input")
                continue
            owner[visible] = (row["id"], label)
            selected[row["id"]] = {"source_id": row["id"], "config": config, "family": family, "context": context,
                                   "question": question, "options": options, "gold": gold, "view": view}
    audit = community_audit(community_rows)["checks"]
    require(all(audit.values()), f"community train audit failed: {audit}")
    rows = sorted(selected.values(), key=lambda r: digest([SEED, r["source_id"]]))
    counts["selected_before_encoding"] = len(rows)
    return rows, counts


def encode(rows: list[dict], tok, seen: Seen):
    from decider.data.core import Example, Q
    from decider.prompt import build
    items, origins, counts, prompts = [], [], Counter(), set()
    for row in rows:
        if len(tok.encode("Context:\n" + row["context"])) > 16384:
            counts["excluded:context_overlength"] += 1
            continue
        item = build(Example(row["context"], [Q(row["question"], row["options"], row["gold"])], "open_jev/" + row["family"]),
                     tok, random.Random(int(digest([SEED, row["source_id"], "permutation"]), 16)),
                     max_options=255, max_ctx_tokens=16384, layout="state_first")
        if len(item["ids"]) > 16384:
            counts["excluded:input_overlength"] += 1
            continue
        require(item["nopts"] == [len(row["options"])] and item["perms"][0][item["golds"][0]] == row["gold"],
                f"gold/option mapping lost: {row['source_id']}")
        identity = prompt_identity(item)
        if identity in prompts or seen.has("prompt", identity):
            counts["excluded:duplicate_prompt"] += 1
            continue
        prompts.add(identity)
        item.update(task="open_jev/" + row["family"])
        items.append(item)
        origins.append({"source": "open_jev", "source_id": row["source_id"], "config": row["config"],
                        "family": row["family"], "view": row["view"]})
        counts["items:" + row["family"]] += 1
    return items, origins, counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", type=Path, default=WORK / "sources")
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--index", type=Path, default=WORK / "index" / "seen.sqlite")
    ap.add_argument("--general-exports", type=Path, default=WORK / "general" / "exports")
    ap.add_argument("--bench-dir", type=Path, help="prepared evaluation benchmark JSONL files to hold out")
    a = ap.parse_args()
    use_upstream()
    from decider import systemone as s1
    sys.path.insert(0, str(a.sources / "open-jev-code"))
    from jev.community_diversity_v2 import audit_records, expected_target

    guard = load_guard(a.general_exports, a.bench_dir, a.sources / "mmlu_pro" / "test-00000-of-00001.parquet",
                       a.sources / "nimble" / "eval.jsonl")
    seen = Seen(a.index)
    rows, scan = collect(a.sources, guard, seen, s1, expected_target, audit_records)
    items, origins, encoding = encode(rows, Tokenizer(a.tokenizer), seen)
    seen.close()
    manifest = write_pack(WORK / "packs" / "open_jev", items, origins,
                          {"component": "open_jev", "seed": SEED, "admitted_families": sorted(ADMITTED_FAMILIES),
                           "scan": dict(scan), "encoding": dict(encoding), "evaluation_guard": summary(guard),
                           "by_family": dict(Counter(o["family"] for o in origins)),
                           "by_view": dict(Counter(o["view"] for o in origins))})
    print(json.dumps({k: manifest[k] for k in ("items", "questions", "tokens")}), flush=True)


if __name__ == "__main__":
    main()
