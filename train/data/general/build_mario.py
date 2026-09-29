#!/usr/bin/env python3
"""Generate the Super Mario Bros. decision task with the upstream generator.

Upstream `run_level` and its action balancing are used as is: 8 levels, 4 episodes per level, seed 0; the last
episode of level 1-1 is kept for evaluation. Writes <upstream copy>/data/mario.pkl, which the upstream `mario`
task loader reads. Needs the game dependencies in data/requirements.txt.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import pickle
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import WORK, use_upstream  # noqa: E402

EPISODES_PER_LEVEL = 4
EPSILONS = [0.0, 0.1, 0.2, 0.3, 0.15]


def main() -> None:
    upstream = use_upstream(WORK / "general" / "upstream")
    os.chdir(upstream)
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    from decider.games import mario_data as M

    started, rng, train, held = time.time(), random.Random(0), [], []
    for level in M.LEVELS:
        for episode in range(EPISODES_PER_LEVEL):
            sink = held if (level == "1-1" and episode == EPISODES_PER_LEVEL - 1) else train
            M.run_level(level, 1, EPSILONS[episode % len(EPSILONS)], rng, sink)
    jumps = [e for e in train if "jump" in e.qs[0].options[e.qs[0].gold]]
    lefts = [e for e in train if e.qs[0].options[e.qs[0].gold] == "step left"]
    rng.shuffle(lefts)
    rest = [e for e in train if e not in jumps and e not in lefts]
    train = rest + jumps * 4 + lefts[:len(jumps)]
    out = upstream / "data" / "mario.pkl"
    out.parent.mkdir(exist_ok=True)
    with out.open("wb") as stream:
        pickle.dump((train, {"mario": held[:1500]}), stream, protocol=5)
    report = {"train_rows": len(train), "eval_rows": len(held[:1500]), "episodes_per_level": EPISODES_PER_LEVEL,
              "seed": 0, "seconds": round(time.time() - started, 1)}
    (WORK / "general" / "mario.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
