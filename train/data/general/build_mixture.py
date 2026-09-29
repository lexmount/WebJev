#!/usr/bin/env python3
"""Run the upstream mixture builder, unchanged, on the converted tasks: mode `full`, seed 6.

The upstream builder adds the wide, padded, described, JSON, single-question, custom, routing, commands,
isolated, rules and contrastive parts on top of the converted tasks, and writes the probes. When it reloads a
task for a probe, this wrapper serves the frozen per-task cache instead of downloading it again.

    python general/build_mixture.py   ->   <upstream working copy>/data/mixture_full.pkl and data/probes.pkl
"""
from __future__ import annotations

import os
from pathlib import Path
import pickle
import runpy
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import WORK, use_upstream  # noqa: E402

GENERAL = WORK / "general"


def main() -> None:
    upstream = use_upstream(GENERAL / "upstream")
    if not (upstream / "teacher_data" / "contrastive_pairs.jsonl").is_file():
        # The upstream recipe treats these pairs as optional (load_contrastive() returns [] without the file).
        print("note: no teacher-written contrastive pairs; the mixture is built without them", flush=True)
    os.chdir(upstream)
    from decider import data as D

    def cached_task(name):
        folder = GENERAL / "task-cache" / name
        with (folder / "converted.pkl").open("rb") as stream:
            return pickle.load(stream)

    D.load_task = cached_task
    sys.argv = ["decider.data.mixture", "--base", "data/tasks.pkl", "--mode", "full", "--seed", "6",
                "--out", "data/mixture_full.pkl", "--probes", "data/probes.pkl"]
    runpy.run_module("decider.data.mixture", run_name="__main__")


if __name__ == "__main__":
    main()
