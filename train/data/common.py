"""Shared helpers for the data builders in this directory.

Every builder turns one public (or our own) source into a *component pack*:

    <pack>/items.pkl          tokenized training items, the format the trainer reads
    <pack>/origins.jsonl.gz   one provenance row per item, aligned by index
    <pack>/manifest.json      counts, settings, input hashes and output hashes

`assemble.py` concatenates the component packs into the final training file.
Prompts are rendered by the upstream prompt builder (`decider.prompt.build`) from the
pinned Mapika/decider checkout, so every component uses the exact input format the
model is trained and served with.
"""
from __future__ import annotations

from array import array
from collections import Counter
import glob
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import shutil
import sys
import unicodedata

TRAIN_DIR = Path(__file__).resolve().parents[1]
UPSTREAM = Path(os.environ.get("WEBJEV_UPSTREAM", TRAIN_DIR / "third_party" / "decider"))
WORK = Path(os.environ.get("WEBJEV_WORK", TRAIN_DIR / "work"))
UPSTREAM_COMMIT = "c4daaac28af9fea95d627015cffa2dd5a5926ee6"
MAX_TOTAL_TOKENS = 16384
MAX_OPTIONS = 255
NORMALIZE_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def use_upstream(path: Path | None = None) -> Path:
    """Make `import decider` resolve to the pinned upstream checkout (see ../setup.sh)."""
    root = Path(path or UPSTREAM)
    require((root / "decider" / "prompt.py").is_file(),
            f"upstream checkout not found at {root}; run train/setup.sh or set WEBJEV_UPSTREAM")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


# ---------------------------------------------------------------- hashing and normalization
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def stable_rank(seed: int, *parts: object) -> str:
    return digest([seed, *parts])


def normal(text: str) -> str:
    """Case-, width- and punctuation-insensitive text used by every duplicate and leakage check."""
    text = unicodedata.normalize("NFKC", text).casefold()
    text = NORMALIZE_PUNCT.sub(" ", text)
    return " ".join(text.split())


def context_key(value: object) -> str:
    """Whitespace/case-normalized identity of a context (JSON contexts are canonicalized first)."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            pass
    if not isinstance(value, str):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False)
    return digest(" ".join(unicodedata.normalize("NFKC", value).split()).casefold())


def stem_key(text: str) -> str:
    """Normalized question stem; a leading "Question" label is ignored."""
    return digest(re.sub(r"^question\s+", "", normal(text), count=1))


def mcqa_keys(stem: str, options: list[str]) -> tuple[str, str]:
    """(stem key, stem + option-set key) for multiple-choice questions."""
    return stem_key(stem), digest([re.sub(r"^question\s+", "", normal(stem), count=1),
                                   sorted(normal(x) for x in options)])


def decision_key(context: str, question: str, options: list[str]) -> str:
    """The visible decision, independent of option order."""
    return digest([normal(context), normal(question), sorted(normal(o) for o in options)])


# ---------------------------------------------------------------- tokenized items
def prompt_identity(item: dict) -> str:
    """The exact model input and answer positions, independent of any metadata."""
    ids = item["ids"]
    h = hashlib.sha256()
    # int32 numpy arrays and array("i") give the same bytes; the fast path avoids a Python-level copy
    h.update(ids.tobytes() if getattr(ids, "dtype", None) == "int32" else array("i", ids).tobytes())
    h.update(canonical([[int(x) for x in item["slots"]], [int(x) for x in item["nopts"]]]).encode())
    return h.hexdigest()


def item_core_hash(item: dict) -> str:
    fields = ("ids", "slots", "golds", "nopts", "perms", "prefix_len", "task")
    return hashlib.sha256(pickle.dumps([(k, item[k]) for k in fields if k in item], protocol=5)).hexdigest()


def invalid_item_reason(item: dict) -> str | None:
    try:
        n = len(item["ids"])
        slots, golds, nopts, perms = (item[k] for k in ("slots", "golds", "nopts", "perms"))
        if not 0 < n <= MAX_TOTAL_TOKENS:
            return "empty_or_overlength"
        if not slots or not len(slots) == len(golds) == len(nopts) == len(perms):
            return "slot_shape"
        if not all(0 <= slot < n for slot in slots):
            return "slot_position"
        if not any(0 <= gold < width for gold, width in zip(golds, nopts)):
            return "no_supervised_slot"
        for gold, width, perm in zip(golds, nopts, perms):
            if not 2 <= width <= MAX_OPTIONS:
                return "option_width"
            if not 0 <= gold < width:
                return "unsupervised_or_bad_gold"
            if len(perm) != width or len(set(perm)) != width or min(perm) < 0:
                return "option_permutation"
        return None
    except (KeyError, TypeError, ValueError):
        return "malformed_fields"


def valid_item(item: dict) -> bool:
    return invalid_item_reason(item) is None


class Tokenizer:
    """Fast tokenizer with the `encode` signature the upstream prompt builder calls.

    Given a model directory, the tokenizer is loaded the way the trainer and the exported model load it
    (transformers AutoTokenizer); its pre-tokenizer differs from the raw tokenizer.json of the base repository
    on text with combining marks, so do not read that file directly. A tokenizer.json path is used as is (pass
    the one of an exported or released WebJev-35B-A3B checkpoint)."""

    def __init__(self, path: Path):
        path = Path(path)
        if path.is_dir():
            from transformers import AutoTokenizer
            self.backend = AutoTokenizer.from_pretrained(str(path)).backend_tokenizer
        else:
            from tokenizers import Tokenizer as Backend
            self.backend = Backend.from_file(str(path))

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self.backend.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        return self.backend.decode(list(ids), skip_special_tokens=skip_special_tokens)


def encode_single(build, tok, task: str, context: str, question: str, options: list[str], gold: int,
                  rng, max_ctx_tokens: int = 10 ** 8) -> dict | None:
    """One single-question item, state-first, no option sampling and no truncation.

    Returns None when the rendered item is invalid or longer than MAX_TOTAL_TOKENS."""
    from decider.data.core import Example, Q
    item = build(Example(context, [Q(question, list(options), int(gold))], task), tok, rng,
                 max_options=MAX_OPTIONS, max_ctx_tokens=max_ctx_tokens, layout="state_first")
    item.update(task=task, ex_id=-1)
    if not valid_item(item) or item["nopts"] != [len(options)] or item["perms"][0][item["golds"][0]] != gold:
        return None
    if item["slots"] != [len(item["ids"]) - 1]:
        return None
    item["ids"] = array("i", item["ids"])
    return item


def remove_duplicate_inputs(candidates: list[tuple[str, dict, dict]]):
    """Drop repeated model inputs; drop every copy of an input that appears with different labels."""
    first: dict[str, tuple[int, tuple[int, ...]]] = {}
    drop: set[int] = set()
    conflicts: set[str] = set()
    stats: Counter = Counter()
    for i, (source, item, _) in enumerate(candidates):
        key = prompt_identity(item)
        target = tuple(int(x) for x in item["golds"])
        if key in conflicts:
            drop.add(i)
            stats["conflicting_label_excluded"] += 1
        elif key not in first:
            first[key] = (i, target)
        elif first[key][1] == target:
            drop.add(i)
            stats["duplicate_input_excluded"] += 1
            stats["duplicate_from_" + source] += 1
        else:
            drop.add(i)
            drop.add(first[key][0])
            conflicts.add(key)
            stats["conflicting_label_excluded"] += 2
    return [entry for i, entry in enumerate(candidates) if i not in drop], dict(stats)


# ---------------------------------------------------------------- readers
def iter_jsonl(path: Path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for i, line in enumerate(stream):
            if line.strip():
                yield i, json.loads(line)


def iter_parquet(paths: list[Path], columns: list[str]):
    import pyarrow.parquet as pq
    index = 0
    for path in paths:
        pf = pq.ParquetFile(path)
        missing = set(columns) - set(pf.schema_arrow.names)
        require(not missing, f"{path}: missing columns {sorted(missing)}")
        for batch in pf.iter_batches(batch_size=1024, columns=columns):
            for row in batch.to_pylist():
                yield index, path, row
                index += 1


def expand(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = [Path(p) for p in sorted(glob.glob(str(pattern)))]
        require(matches, f"no file matches {pattern}")
        paths.extend(matches)
    return sorted(set(p.resolve() for p in paths))


# ---------------------------------------------------------------- packs
def write_pack(out: Path, items: list[dict], origins: list[dict], manifest: dict) -> dict:
    """Write a component pack into a staging directory, re-read it, then publish it atomically."""
    out = Path(out)
    require(not out.exists(), f"output already exists: {out}")
    require(len(items) == len(origins) and items, "a pack needs aligned, non-empty items and origins")
    stage = out.with_name(out.name + ".building")
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    for i, (item, origin) in enumerate(zip(items, origins)):
        item["ex_id"] = i
        origin["index"] = i
        origin["core_sha256"] = item_core_hash(item)
        require(valid_item(item), f"invalid item at {i}: {invalid_item_reason(item)}")
    with (stage / "items.pkl").open("xb") as stream:
        pickle.dump(items, stream, protocol=5)
    with gzip.open(stage / "origins.jsonl.gz", "wt", encoding="utf-8") as stream:
        for origin in origins:
            stream.write(canonical(origin) + "\n")
    with (stage / "items.pkl").open("rb") as stream:
        check = pickle.load(stream)
    require(len(check) == len(origins), "serialized item count mismatch")
    for i, (item, origin) in enumerate(zip(check, origins)):
        require(item["ex_id"] == i and item_core_hash(item) == origin["core_sha256"], f"serialized item mismatch at {i}")
    manifest = dict(manifest)
    manifest.update(items=len(items), questions=sum(len(it["slots"]) for it in items),
                    tokens=sum(len(it["ids"]) for it in items), max_sequence=max(len(it["ids"]) for it in items),
                    items_sha256=sha256_file(stage / "items.pkl"),
                    origins_sha256=sha256_file(stage / "origins.jsonl.gz"))
    (stage / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    stage.rename(out)
    return manifest


def read_pack(directory: Path) -> tuple[dict, list[dict], list[dict]]:
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    require(sha256_file(directory / "items.pkl") == manifest["items_sha256"], f"items.pkl changed: {directory}")
    with (directory / "items.pkl").open("rb") as stream:
        items = pickle.load(stream)
    origins = [row for _, row in iter_jsonl(directory / "origins.jsonl.gz")]
    require(len(items) == len(origins) == manifest["items"], f"pack is incomplete: {directory}")
    return manifest, items, origins


def input_record(paths: list[Path]) -> dict:
    return {str(p): {"bytes": Path(p).stat().st_size, "sha256": sha256_file(p)} for p in paths}
