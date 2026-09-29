#!/usr/bin/env python3
"""Local copies of the two task sources that `datasets` can no longer load directly.

    python general/sources.py trec       # CogComp/trec test split, parquet verified row by row against the original file
    python general/sources.py mind2web   # the 11 original Mind2Web training JSON files, verified by their LFS SHA-256

`convert_tasks.py` loads these local copies instead of the remote datasets. Nothing about the task
conversion changes: the upstream loaders receive the same rows, labels and order.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import WORK, require, sha256_file  # noqa: E402

CACHE = WORK / "general" / "cache"
TREC_REPO = "CogComp/trec"
TREC_ORIGINAL_REVISION = "eb1e45c1ba990fecca7cf84b67ce845edbcf49bf"   # holds the original loading script and label names
TREC_PARQUET_REVISION = "65752bf53af25bc935a0dce92fb5b6c930728450"    # the Hub's parquet conversion of the same data
MIND2WEB_REPO = "osunlp/Mind2Web"
MIND2WEB_REVISION = "17ece8eb89862368edc0cc806acee6fca5163474"


def trec() -> dict:
    """Check the converted parquet against all 500 original TREC_10.label rows (text, both labels, order)."""
    import requests
    from datasets import Dataset
    from huggingface_hub import hf_hub_download

    script = Path(hf_hub_download(TREC_REPO, "trec.py", repo_type="dataset", revision=TREC_ORIGINAL_REVISION)).read_text()
    values = {}
    for node in ast.parse(script).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ("_FINE_LABELS", "_COARSE_LABELS", "_URLs"):
                    values[target.id] = ast.literal_eval(node.value)
    parquet = Path(hf_hub_download(TREC_REPO, "default/test/0000.parquet", repo_type="dataset",
                                   revision=TREC_PARQUET_REVISION))
    out = CACHE / "trec"
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "test.parquet"
    shutil.copyfile(parquet, dest)
    ds = Dataset.from_parquet(str(dest))
    require(len(ds) == 500, "TREC test must have 500 rows")
    require(ds.features["fine_label"].names == values["_FINE_LABELS"], "TREC fine label names differ")
    require(ds.features["coarse_label"].names == values["_COARSE_LABELS"], "TREC coarse label names differ")
    response = requests.get(values["_URLs"]["test"], timeout=(20, 60))
    response.raise_for_status()
    rows = []
    for line in response.content.splitlines():
        fine, _, text = line.replace(b"\xf0", b" ").strip().decode().partition(" ")
        rows.append({"text": text, "fine_label": values["_FINE_LABELS"].index(fine),
                     "coarse_label": values["_COARSE_LABELS"].index(fine.split(":")[0])})
    require(rows == ds.to_list(), "the parquet conversion does not match the original TREC test file")
    report = {"dataset": TREC_REPO, "original_revision": TREC_ORIGINAL_REVISION,
              "parquet_revision": TREC_PARQUET_REVISION, "local_parquet": str(dest),
              "parquet_sha256": sha256_file(dest), "original_url": values["_URLs"]["test"],
              "original_sha256": hashlib.sha256(response.content).hexdigest(), "verified_rows": len(rows)}
    (out / "verified.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def mind2web() -> dict:
    """Download the original training JSON files; hf_hub_download checks each file against its LFS SHA-256."""
    from huggingface_hub import HfApi, hf_hub_download

    out = CACHE / "mind2web"
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for entry in HfApi().list_repo_tree(MIND2WEB_REPO, repo_type="dataset", revision=MIND2WEB_REVISION, recursive=True):
        path = getattr(entry, "path", "")
        if path.startswith("data/train/") and path.endswith(".json"):
            require(entry.lfs is not None and len(entry.lfs.sha256) == 64, f"{path} has no LFS hash")
            files.append({"path": path, "bytes": entry.size, "sha256": entry.lfs.sha256})
    files.sort(key=lambda x: x["path"])
    require(len(files) == 11, f"expected 11 Mind2Web training files, found {len(files)}")
    for f in files:
        local = Path(hf_hub_download(MIND2WEB_REPO, f["path"], repo_type="dataset", revision=MIND2WEB_REVISION,
                                     local_dir=out))
        require(local.stat().st_size == f["bytes"], f"size mismatch: {f['path']}")
        require(sha256_file(local) == f["sha256"], f"SHA-256 mismatch: {f['path']}")
        f["local"] = str(local)
    report = {"dataset": MIND2WEB_REPO, "revision": MIND2WEB_REVISION, "files": files}
    (out / "verified.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", choices=["trec", "mind2web"])
    a = ap.parse_args()
    report = trec() if a.source == "trec" else mind2web()
    print(json.dumps({"source": a.source, "ok": True, "files": len(report.get("files", [report]))}), flush=True)


if __name__ == "__main__":
    main()
