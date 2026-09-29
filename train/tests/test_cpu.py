"""CPU-only checks of the batch schedule and the data pack helpers.

    python -m pytest tests/test_cpu.py -q
"""
from array import array
import json
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pytest

TRAIN_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(TRAIN_DIR / "src"), str(TRAIN_DIR / "data")]
from batch_schedule import assign_batches, checkpoint_due  # noqa: E402
import common  # noqa: E402


def item(rng, n=None, options=4, gold=None):
    n = n or rng.randint(20, 3000)
    gold = rng.randrange(options) if gold is None else gold
    return dict(ids=array("i", [rng.randrange(1000) for _ in range(n)]), slots=[n - 1], golds=[gold], nopts=[options],
                perms=[list(range(options))], task="t", ex_id=-1)


def test_assign_batches_covers_every_item_once_and_weights_sum_to_world():
    rng = random.Random(0)
    items = [item(rng) for _ in range(300)]
    group = [list(range(i, min(i + 7, 300))) for i in range(0, 300, 7)][:16]
    ranks, loads = assign_batches(group, items, world=8, micro_tokens=2048)
    seen = sorted(i for tasks in ranks for task in tasks for i in task["indices"])
    assert seen == sorted(i for batch in group for i in batch)
    # gradients are averaged over 8 ranks, so the weights of all micro-batches must add up to the world size
    assert abs(sum(task["loss_weight"] for tasks in ranks for task in tasks) - 8) < 1e-9
    assert len(loads) == 8 and all(load >= 0 for load in loads)


def test_checkpoint_due():
    assert checkpoint_due(2000, 4756, 2000) and checkpoint_due(4756, 4756, 2000) and not checkpoint_due(2001, 4756, 2000)


def test_prompt_identity_is_the_same_for_numpy_and_array_ids():
    rng = random.Random(1)
    a = item(rng, n=50)
    b = dict(a, ids=np.asarray(a["ids"], dtype=np.int32))
    assert common.prompt_identity(a) == common.prompt_identity(b)


def test_remove_duplicate_inputs_drops_repeats_and_conflicts():
    rng = random.Random(2)
    x, y = item(rng, n=30, gold=1), item(rng, n=31, gold=0)
    x_repeat, y_conflict = dict(x), dict(y, golds=[2])
    kept, stats = common.remove_duplicate_inputs([("a", x, {}), ("b", x_repeat, {}), ("a", y, {}), ("b", y_conflict, {})])
    assert [entry[1] is x for entry in kept] == [True]
    assert stats["duplicate_input_excluded"] == 1 and stats["conflicting_label_excluded"] == 2


def test_packs_and_assembly(tmp_path, monkeypatch):
    rng = random.Random(3)
    monkeypatch.setattr(common, "WORK", tmp_path)
    general = [item(rng) for _ in range(20)]
    extra = [dict(general[0]), item(rng), item(rng)]           # the first one repeats a general item
    common.write_pack(tmp_path / "packs" / "general", general, [{"source": "general"} for _ in general], {"component": "general"})
    common.write_pack(tmp_path / "packs" / "extra", extra, [{"source": "extra"} for _ in extra], {"component": "extra"})
    manifest, items, origins = common.read_pack(tmp_path / "packs" / "extra")
    assert manifest["items"] == 3 and len(items) == len(origins) == 3
    subprocess.run([sys.executable, str(TRAIN_DIR / "data" / "assemble.py"), "--components", "general", "extra"],
                   check=True, env={"WEBJEV_WORK": str(tmp_path), "PATH": "/usr/bin:/bin"})
    mixture = json.loads((tmp_path / "mixture" / "items.pkl.json").read_text())
    assert mixture["items"] == 22 and mixture["components"]["extra"]["items"] == 2
    assert mixture["cross_component_dedup"] == {"already_in_mixture_excluded:extra": 1}


def test_component_index(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    rng = random.Random(4)
    general = [item(rng) for _ in range(3)]
    common.write_pack(tmp_path / "packs" / "general", general, [{"source": "general"} for _ in general], {"component": "general"})
    web = [item(rng) for _ in range(2)]
    web_origins = [{"source": "web", "stem_key": common.stem_key(f"page {i}"),
                    "decision_key": common.decision_key(f"page {i}", "Next?", ["CLICK", "DONE"])} for i in range(2)]
    common.write_pack(tmp_path / "packs" / "web", web, web_origins, {"component": "web"})
    rows = [{"context": f"state {i}", "questions": [{"text": "Q?", "options": ["a", "b"], "gold": 0}]} for i in range(3)]
    (tmp_path / "general" / "exports").mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "general" / "exports" / "train.parquet")
    subprocess.run([sys.executable, str(TRAIN_DIR / "data" / "index_components.py")], check=True,
                   env={"WEBJEV_WORK": str(tmp_path), "PATH": "/usr/bin:/bin"})
    manifest = json.loads((tmp_path / "index" / "manifest.json").read_text())
    assert manifest["counts"] == {"general_rows": 3, "general_prompts": 3, "web_prompts": 2, "web_rows": 2}
    sys.path.insert(0, str(TRAIN_DIR / "data"))
    from index_components import Seen
    seen = Seen(tmp_path / "index" / "seen.sqlite")
    assert seen.has("stem", common.stem_key("state 1")) and seen.has("prompt", common.prompt_identity(web[0]))
    assert not seen.has("stem", common.stem_key("state 9"))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
