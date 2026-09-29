"""Evaluation exclusion shared by the Open-Jev and knowledge-MCQA builders.

Collects the texts of the evaluation items (the upstream task evaluation splits and probes exported by
general/finalize.py, the prepared benchmark files passed with --bench-dir, and the evaluation files fetched by
download_sources.py) so that the builders can drop any training candidate that matches one of them exactly
after normalization (case, width, punctuation and whitespace folded).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

from common import canonical, context_key, iter_jsonl, mcqa_keys, normal, require, sha256_file, stem_key


@dataclass
class Guard:
    contexts: set = field(default_factory=set)    # context_key of every evaluation context
    stems: set = field(default_factory=set)       # stem_key of every evaluation context and question
    questions: set = field(default_factory=set)   # normalized evaluation question texts
    mcqa_stems: set = field(default_factory=set)  # multiple-choice question stem keys
    mcqa_full: set = field(default_factory=set)   # multiple-choice question + option-set keys
    evidence: list = field(default_factory=list)

    def add_context(self, value) -> None:
        self.contexts.add(context_key(value))
        self.stems.add(stem_key(value if isinstance(value, str) else canonical(value)))


def _parquet_contexts(path: Path):
    import pyarrow.parquet as pq
    for batch in pq.ParquetFile(path).iter_batches(columns=["context"], batch_size=4096):
        yield from batch.column(0).to_pylist()


def load_guard(general_exports: Path | None = None, bench_dir: Path | None = None,
               mcqa_holdout: Path | None = None, state_holdout: Path | None = None) -> Guard:
    """mcqa_holdout: parquet with `question` and `options`; state_holdout: JSONL whose rows carry input.state."""
    guard = Guard()
    if general_exports:
        files = sorted(Path(general_exports).glob("eval/*.parquet")) + sorted(Path(general_exports).glob("probes/*.parquet"))
        require(files, f"no evaluation parquet files under {general_exports}")
        for path in files:
            n = 0
            for value in _parquet_contexts(path):
                guard.add_context(value)
                n += 1
            guard.evidence.append({"file": f"{path.parent.name}/{path.name}", "rows": n})
    if bench_dir:
        for path in sorted(Path(bench_dir).glob("*.jsonl")):
            n = 0
            for _, row in iter_jsonl(path):
                value = row.get("state", row.get("context"))
                if value is not None:
                    guard.add_context(value)
                    n += 1
            guard.evidence.append({"file": path.name, "rows": n, "sha256": sha256_file(path)})
    if mcqa_holdout:
        import pyarrow.parquet as pq
        n = 0
        for batch in pq.ParquetFile(mcqa_holdout).iter_batches(columns=["question", "options"], batch_size=4096):
            for row in batch.to_pylist():
                question, options = row.get("question"), row.get("options")
                if isinstance(question, str):
                    guard.questions.add(normal(question))
                    guard.stems.add(stem_key(question))
                    if isinstance(options, list) and 2 <= len(options) <= 255:
                        stem, full = mcqa_keys(question, [str(o) for o in options])
                        guard.mcqa_stems.add(stem)
                        guard.mcqa_full.add(full)
                n += 1
        require(n > 0, f"no questions read from {mcqa_holdout}")
        guard.evidence.append({"file": Path(mcqa_holdout).name, "rows": n, "sha256": sha256_file(mcqa_holdout)})
    if state_holdout:
        n = 0
        for _, row in iter_jsonl(state_holdout):
            guard.add_context(row["input"]["state"])
            n += 1
        require(n > 0, f"no rows read from {state_holdout}")
        guard.evidence.append({"file": Path(state_holdout).name, "rows": n, "sha256": sha256_file(state_holdout)})
    return guard


def summary(guard: Guard) -> dict:
    return {"sources": guard.evidence, "context_keys": len(guard.contexts), "stem_keys": len(guard.stems),
            "question_keys": len(guard.questions), "mcqa_keys": len(guard.mcqa_full)}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Print the size of the evaluation guard built from the given sources.")
    ap.add_argument("--general-exports", type=Path)
    ap.add_argument("--bench-dir", type=Path)
    ap.add_argument("--mcqa-holdout", type=Path)
    ap.add_argument("--state-holdout", type=Path)
    a = ap.parse_args()
    print(json.dumps(summary(load_guard(a.general_exports, a.bench_dir, a.mcqa_holdout, a.state_holdout)), indent=2))
