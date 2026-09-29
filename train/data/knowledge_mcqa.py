#!/usr/bin/env python3
"""Component 4: knowledge multiple-choice questions and Nimble evidence-checking decisions.

Sources (train splits only; see download_sources.sh for the pinned revisions):
  - nvidia/Nemotron-RL-knowledge-mcqa, all four train shards;
  - nvidia/OpenScienceReasoning-2, the train file;
  - bespokelabsai/nimble, data/train.jsonl.

Rules:
  - Only questions with a contiguous, labeled A/B/C/... option block (2-255 options) and one unambiguous gold
    option are used. The OpenScience reasoning text (`output`) is never read.
  - A question that appears with two different gold answers is dropped entirely; repeats are kept once.
  - Evaluation items are excluded by exact normalized match (eval_guard.py).
  - A question already present in the general or web component (same stem, stem + option set, decision or
    tokenized prompt; see index_components.py) is skipped.
  - Each Nimble Score row becomes one yes/no row per level (the upstream "isolated levels" view); Choice and
    yes/no rows are used directly.
  - Rendering: upstream prompt builder, state-first, all options, no truncation; items longer than 16,384
    tokens are dropped. Exact duplicate inputs are removed at the end.

    python data/knowledge_mcqa.py --sources $WEBJEV_WORK/sources --tokenizer <base model dir>
      -> $WEBJEV_WORK/packs/knowledge_mcqa
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import random
import re
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (WORK, Tokenizer, canonical, decision_key, digest, encode_single, expand,  # noqa: E402
                    iter_jsonl, iter_parquet, mcqa_keys, normal, prompt_identity, remove_duplicate_inputs,
                    require, stable_rank, use_upstream, write_pack)
from eval_guard import load_guard  # noqa: E402
from index_components import Seen  # noqa: E402

OPTION_LINE = re.compile(r"^\s*([A-Z])\s*[:.)]\s+(.+?)\s*$")
INSTRUCTION = re.compile(r"^(?:Answer the following multiple choice question\.|Solve the following problem\.)", re.I)
MCQA_QUESTION = "Which option correctly answers the question?"
TASKS = {"nemotron": "nemotron_knowledge_mcqa", "openscience": "openscience_mcqa", "nimble": "nimble"}


@dataclass(frozen=True)
class MCQA:
    source: str
    source_id: str
    source_index: int
    stem: str
    options: list
    gold: int


def labelled_stem_and_options(text: str):
    """Accept only a trailing, contiguous A/B/... option block; never infer labels."""
    if not isinstance(text, str):
        return None
    text = text.replace("\r\n", "\n").strip()
    if INSTRUCTION.match(text):
        pieces = text.split("\n\n", 1)
        if len(pieces) != 2:
            return None
        text = pieces[1].strip()
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if (m := OPTION_LINE.fullmatch(line)) and m.group(1) == "A"]
    for start in reversed(starts):
        stem = "\n".join(lines[:start]).strip()
        if not stem:
            continue
        values, labels = [], []
        for line in lines[start:]:
            m = OPTION_LINE.fullmatch(line)
            if m:
                labels.append(m.group(1))
                values.append(m.group(2).strip())
            elif values and line.strip():
                values[-1] += " " + line.strip()
            elif values and not line.strip():
                continue
            else:
                break
        if 2 <= len(values) <= 255 and labels == [chr(65 + i) for i in range(len(values))] and all(normal(x) for x in values):
            return stem, values
    return None


def strip_label(value: str, label: str) -> str:
    return re.sub(r"^\s*" + re.escape(label) + r"\s*[:.)]\s*", "", value).strip()


def nemotron_record(row: dict, index: int) -> MCQA | None:
    params = row.get("responses_create_params")
    messages = params.get("input") if isinstance(params, dict) else None
    if not isinstance(messages, list) or len(messages) != 1 or not isinstance(messages[0], dict):
        return None
    parsed = labelled_stem_and_options(messages[0].get("content"))
    raw_options = row.get("options")
    if parsed is None or not isinstance(raw_options, list):
        return None
    stem, parsed_options = parsed
    by_label = {}
    for entry in raw_options:
        live = [(k, v) for k, v in entry.items() if v is not None] if isinstance(entry, dict) else []
        if len(live) != 1 or not isinstance(live[0][1], str) or live[0][0] in by_label:
            return None
        by_label[live[0][0]] = live[0][1]
    labels = [chr(65 + i) for i in range(len(parsed_options))]
    if set(by_label) != set(labels):
        return None
    options = [strip_label(by_label[label], label) for label in labels]
    if any(normal(a) != normal(b) for a, b in zip(options, parsed_options)):
        return None
    answer, source_id = row.get("expected_answer"), row.get("uuid")
    if not isinstance(answer, str) or answer not in labels or not isinstance(source_id, str) or not source_id:
        return None
    return MCQA("nemotron", source_id, index, stem, options, labels.index(answer))


def openscience_record(row: dict, index: int, source_id: str) -> MCQA | None:
    parsed = labelled_stem_and_options(row.get("input"))
    answer = row.get("expected_answer")
    if parsed is None or not isinstance(answer, str):
        return None
    stem, options = parsed
    labels = [chr(65 + i) for i in range(len(options))]
    answer = answer.strip()
    # The answer is a letter, a boxed/text letter, or the exact option text; anything else is excluded.
    if answer in labels:
        gold = labels.index(answer)
    else:
        boxed = re.fullmatch(r"\\(?:text|mathrm|boxed)\{\s*([A-Z])\s*\}", answer)
        if boxed is not None and boxed.group(1) in labels:
            gold = labels.index(boxed.group(1))
        else:
            literal = " ".join(answer.casefold().split())
            matches = [i for i, o in enumerate(options) if literal == " ".join(o.casefold().split())]
            if len(matches) != 1:
                return None
            gold = matches[0]
    return MCQA("openscience", source_id, index, stem, options, gold)


def stage_mcqa(conn, nemotron: list[Path], openscience: list[Path], guard, seed: int, limit: int | None) -> dict:
    """Parse both sources into one table keyed by question; flag questions seen with two different answers."""
    conn.execute("CREATE TABLE mcqa (full_key TEXT PRIMARY KEY, question_key TEXT NOT NULL, source TEXT NOT NULL, "
                 "source_id TEXT NOT NULL, source_index INTEGER NOT NULL, rank TEXT NOT NULL, stem TEXT NOT NULL, "
                 "options TEXT NOT NULL, gold INTEGER NOT NULL, gold_text TEXT NOT NULL, conflict INTEGER NOT NULL DEFAULT 0)")
    conn.execute("CREATE UNIQUE INDEX mcqa_question ON mcqa(question_key)")
    conn.execute("CREATE INDEX mcqa_rank ON mcqa(source, rank)")
    stats = {"nemotron": Counter(), "openscience": Counter()}
    specs = (("nemotron", nemotron, ["responses_create_params", "expected_answer", "uuid", "options"]),
             ("openscience", openscience, ["input", "expected_answer"]))
    for source, paths, columns in specs:
        for i, path, row in iter_parquet(paths, columns):
            if limit is not None and i >= limit:
                break
            stats[source]["read"] += 1
            record = nemotron_record(row, i) if source == "nemotron" else openscience_record(row, i, f"{path.name}:{i}")
            if record is None:
                stats[source]["invalid_or_not_mcqa"] += 1
                continue
            qkey, fkey = mcqa_keys(record.stem, record.options)
            if fkey in guard.mcqa_full or qkey in guard.mcqa_stems:
                stats[source]["evaluation_item_excluded"] += 1
                continue
            value = (fkey, qkey, source, record.source_id, record.source_index,
                     stable_rank(seed, source, record.source_id), record.stem, canonical(record.options),
                     record.gold, normal(record.options[record.gold]))
            try:
                conn.execute("INSERT INTO mcqa(full_key, question_key, source, source_id, source_index, rank, stem, options, "
                             "gold, gold_text) VALUES (?,?,?,?,?,?,?,?,?,?)", value)
                stats[source]["accepted_unique"] += 1
            except sqlite3.IntegrityError:
                old = conn.execute("SELECT full_key, gold_text FROM mcqa WHERE question_key=?", (qkey,)).fetchone()
                if old[1] != value[-1]:
                    conn.execute("UPDATE mcqa SET conflict=1 WHERE question_key=?", (qkey,))
                    stats[source]["conflicting_label"] += 1
                else:
                    stats[source]["duplicate_question"] += 1
        conn.commit()
    return {k: dict(v) for k, v in stats.items()}


def nimble_views(row: dict, render_state, render_question, plan_rows):
    """All inference rows of one Nimble train example: (context, question, options, gold, type, view_index)."""
    if row.get("split") != "train" or not isinstance(row.get("id"), str):
        return None
    inp, ref = row.get("input"), row.get("reference")
    if not isinstance(inp, dict) or not isinstance(ref, dict):
        return None
    qs = inp.get("questions")
    if not isinstance(qs, dict) or len(qs) != 1 or not isinstance(qs.get("decision"), dict):
        return None
    if not isinstance(inp.get("state"), (str, dict, list)):
        return None
    try:
        context = render_state(inp["state"])
        rendered = render_question(qs["decision"])
    except (ValueError, TypeError, KeyError):
        return None
    target, typ, names = ref.get("target"), rendered["type"], rendered["names"]
    if (typ == "noul" and type(target) is not bool) or (typ == "score" and (type(target) is not int or target < 0)) \
            or (typ == "choice" and not isinstance(target, str)) or target not in names:
        return None
    if not context.strip() or not all(normal(o) for o in rendered["options"]):
        return None
    rows, mapping = plan_rows({"decision": rendered}, isolated=True)
    if not rows or len(mapping) != 1:
        return None
    views = []
    for view_index, view in enumerate(rows):
        question, options = view["question"].strip(), view["options"]
        if not question or not 2 <= len(options) <= 255:
            return None
        gold = int(view_index == target) if typ == "score" else names.index(target)
        views.append((context, question, options, gold, typ, view_index))
    if typ == "score" and len(views) != len(names):
        return None
    return views


def nimble_signature(row: dict) -> str | None:
    inp = row.get("input")
    q = (inp.get("questions") or {}).get("decision") if isinstance(inp, dict) else None
    if not isinstance(q, dict) or not isinstance(q.get("instructions"), str):
        return None
    state = inp.get("state")
    return digest([normal(state if isinstance(state, str) else canonical(state)), normal(q["instructions"])])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", type=Path, default=WORK / "sources", help="download_sources.sh output directory")
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--index", type=Path, default=WORK / "index" / "seen.sqlite")
    ap.add_argument("--seed", type=int, default=20260927)
    ap.add_argument("--smoke-rows", type=int, help="scan at most N rows per MCQA source (diagnostics only)")
    a = ap.parse_args()
    use_upstream()
    from decider.prompt import build
    from decider.systemone import plan_rows, render_question, render_state

    src = a.sources
    nemotron = expand([str(src / "nemotron" / "train-*.parquet")])
    openscience = expand([str(src / "openscience" / "OpenScienceReasoning-2.parquet")])
    require(len(nemotron) == 4, "all four Nemotron train shards are required")
    guard = load_guard(mcqa_holdout=src / "mmlu_pro" / "test-00000-of-00001.parquet")
    seen = Seen(a.index)
    tok = Tokenizer(a.tokenizer)
    stage = WORK / "tmp" / "mcqa.sqlite"
    stage.parent.mkdir(parents=True, exist_ok=True)
    stage.unlink(missing_ok=True)
    conn = sqlite3.connect(stage)
    candidates, stats = [], {"nemotron": Counter(), "openscience": Counter(), "nimble": Counter()}
    scan = stage_mcqa(conn, nemotron, openscience, guard, a.seed, a.smoke_rows)
    for source in ("nemotron", "openscience"):
        rows = conn.execute("SELECT source_id, source_index, stem, options, gold, question_key, full_key FROM mcqa "
                            "WHERE source=? AND conflict=0 ORDER BY rank", (source,))
        for source_id, source_index, stem, options_json, gold, qkey, fkey in rows:
            if seen.has("stem", qkey) or seen.has("full", fkey):
                stats[source]["already_in_mixture"] += 1
                continue
            rng = random.Random(int(stable_rank(a.seed, source, source_id), 16))
            item = encode_single(build, tok, TASKS[source], stem, MCQA_QUESTION, json.loads(options_json), gold, rng)
            if item is None:
                stats[source]["invalid_or_overlength"] += 1
                continue
            if seen.has("prompt", prompt_identity(item)):
                stats[source]["already_in_mixture"] += 1
                continue
            candidates.append((source, item, {"source": source, "source_id": source_id, "source_index": source_index}))
            stats[source]["selected"] += 1
    conn.close()
    stage.unlink()

    eval_ids, eval_families, eval_source_families, eval_signatures = set(), set(), set(), set()
    for _, row in iter_jsonl(src / "nimble" / "eval.jsonl"):
        for value, target in ((row.get("id"), eval_ids), (row.get("family"), eval_families),
                              (row.get("source_family"), eval_source_families), (nimble_signature(row), eval_signatures)):
            if isinstance(value, str):
                target.add(value)
    require(eval_ids and eval_families and eval_signatures, "the Nimble evaluation split could not be read")
    nimble, nimble_ids = [], set()
    for i, row in iter_jsonl(src / "nimble" / "train.jsonl"):
        stats["nimble"]["read"] += 1
        views = nimble_views(row, render_state, render_question, plan_rows)
        if views is None:
            stats["nimble"]["invalid"] += 1
            continue
        if (row["id"] in eval_ids or row.get("family") in eval_families
                or row.get("source_family") in eval_source_families or nimble_signature(row) in eval_signatures):
            stats["nimble"]["evaluation_item_excluded"] += 1
            continue
        if row["id"] in nimble_ids:
            stats["nimble"]["duplicate_id"] += 1
            continue
        nimble_ids.add(row["id"])
        nimble.append((stable_rank(a.seed, "nimble", row["id"]), row["id"], i, views))
    for _, sid, i, views in sorted(nimble):
        for context, question, options, gold, typ, view_index in views:
            view_id = f"{sid}:isolated:{view_index}" if typ == "score" else sid
            item = encode_single(build, tok, TASKS["nimble"], context, question, options, gold,
                                 random.Random(int(stable_rank(a.seed, "nimble", view_id), 16)))
            require(item is not None, f"Nimble row {sid} view {view_index} does not render")
            if seen.has("decision", decision_key(context, question, options)) or seen.has("prompt", prompt_identity(item)):
                stats["nimble"]["already_in_mixture"] += 1
                continue
            candidates.append(("nimble", item, {"source": "nimble", "source_id": sid, "source_index": i,
                                                "type": typ, "view_index": view_index}))
            stats["nimble"]["selected"] += 1
    seen.close()
    kept, dedup = remove_duplicate_inputs(candidates)
    manifest = write_pack(WORK / "packs" / "knowledge_mcqa", [item for _, item, _ in kept], [o for _, _, o in kept],
                          {"component": "knowledge_mcqa", "seed": a.seed, "smoke_only": a.smoke_rows is not None,
                           "scan": scan, "selection": {k: dict(v) for k, v in stats.items()}, "final_dedup": dedup,
                           "by_source": dict(Counter(o["source"] for _, _, o in kept))})
    print(json.dumps({k: manifest[k] for k in ("items", "questions", "tokens", "by_source")}), flush=True)


if __name__ == "__main__":
    main()
