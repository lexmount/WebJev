"""Export a checkpoint as a complete Hugging Face model directory, the layout of the released WebJev-35B-A3B.

    python src/export.py --run RUN_DIR --base BASE_DIR --out MODEL_DIR [--step N]

The checkpoint's trainable weights (overlay.pt) replace the corresponding base-model weights, and the full model
is saved as safetensors with the tokenizer. The directory also gets the prompt package that the serving code
imports (`decider/`, copied from the pinned upstream checkout) and the model's decision settings in
`decider_config.json` (the upstream file name). Afterwards every routed-expert tensor is compared byte for byte
with the base model: they were frozen during training and must be unchanged. Runs on the CPU and holds the
whole model in bf16 (34.7B parameters, about 70 GB of RAM).
"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil
import struct
import time

from model_runtime import UPSTREAM, TrainingModel, is_routed_expert
import torch

MODEL_SETTINGS = dict(version="WebJev-35B-A3B", base="Qwen/Qwen3.5-35B-A3B-Base", temperature=1.0,
                      temperature_fitted=False, neutralize_none=False, max_options=255, max_state_tokens=32768,
                      schema_first=False, schema_first_trained=True, isolated_levels=True)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def tensor_table(directory):
    index = json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"]
    table = {}
    for name in sorted(set(index.values())):
        path = directory / name
        with path.open("rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        for key, entry in header.items():
            if key != "__metadata__":
                lo, hi = entry["data_offsets"]
                table[key] = dict(path=path, offset=n + 8 + lo, length=hi - lo, dtype=entry["dtype"], shape=entry["shape"])
    return table


def tensor_digest(info):
    h, remaining = hashlib.sha256(), info["length"]
    with info["path"].open("rb") as f:
        f.seek(info["offset"])
        while remaining:
            part = f.read(min(8 * 1024 * 1024, remaining))
            if not part:
                raise IOError("short tensor read")
            h.update(part)
            remaining -= len(part)
    return h.hexdigest()


def verify_frozen_experts(base: Path, out: Path) -> int:
    base_table, out_table = tensor_table(base), tensor_table(out)

    def compare(name):
        # the base checkpoint stores the language model under model.language_model.*
        candidate = name if name in base_table else "model.language_model." + name[len("model."):]
        old, new = base_table[candidate], out_table[name]
        if old["shape"] != new["shape"] or old["dtype"] != new["dtype"]:
            raise ValueError("frozen expert metadata changed: " + name)
        return tensor_digest(old) == tensor_digest(new)

    names = [n for n in out_table if is_routed_expert(n)]
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        unchanged = list(pool.map(compare, names))
    if not names or not all(unchanged):
        raise ValueError("routed-expert weights differ from the base model")
    return len(names)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--step", type=int, help="checkpoint step (default: the final step of a completed run)")
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"output already exists: {a.out}")
    step = a.step or json.loads((a.run / "training-complete.json").read_text())["steps"]
    checkpoint = a.run / "checkpoints" / f"step-{step:06d}"
    ready = json.loads((checkpoint / "ready.json").read_text())
    if sha256(checkpoint / "overlay.pt") != ready["files"]["overlay.pt"]["sha256"]:
        raise ValueError("overlay.pt does not match its recorded SHA-256")
    model = TrainingModel(str(a.base), grad_ckpt=False)
    overlay = torch.load(checkpoint / "overlay.pt", map_location="cpu", weights_only=True)
    if set(overlay) != {n for n, _ in model.trainable_named()}:
        raise ValueError("overlay parameters differ from the trainable parameters")
    result = model.lm.load_state_dict(overlay, strict=False)
    if result.unexpected_keys or set(result.missing_keys) != {n for n, p in model.lm.named_parameters() if not p.requires_grad}:
        raise ValueError("unexpected keys while loading the overlay")
    del overlay
    a.out.mkdir(parents=True)
    model.lm.config.experts_implementation = "grouped_mm"
    model.lm.save_pretrained(a.out, safe_serialization=True, max_shard_size="5GB")
    model.tok.save_pretrained(a.out)
    del model
    shutil.copy2(a.base / "LICENSE", a.out / "LICENSE")
    shutil.copytree(UPSTREAM / "decider", a.out / "decider", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (a.out / "decider_config.json").write_text(json.dumps(MODEL_SETTINGS, indent=1) + "\n")
    frozen = verify_frozen_experts(a.base, a.out)
    report = dict(step=step, frozen_expert_tensors_unchanged=frozen, seen=ready["seen"],
                  exported_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    (a.out / "training_export.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
