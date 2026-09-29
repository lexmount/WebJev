#!/usr/bin/env python3
"""Download the public sources of the Open-Jev and knowledge-MCQA components at pinned revisions.

Every file is checked against its SHA-256 before use.

    python data/download_sources.py [--out $WEBJEV_WORK/sources]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import WORK, require, sha256_file  # noqa: E402

OPEN_JEV = ("ZefanCai/Open-Jev", "c67699e13d0ae25e35b77165a4b6b079bedc8aba")
OPEN_JEV_V11 = ("ZefanCai/Open-Jev-v1.1", "10ad6888333fa97f8c948192797bad3de3040802")
WANLI = ("alisawuffles/WANLI", "61c95318fd71c55b6ba355d76253254615f387ec")
NEMOTRON = ("nvidia/Nemotron-RL-knowledge-mcqa", "62a1eec1f952723eab2ee3832222f533b8138067")
OPENSCIENCE = ("nvidia/OpenScienceReasoning-2", "174b02c9cdf231f220765b2a1d5ece4550921894")
MMLU_PRO = ("TIGER-Lab/MMLU-Pro", "b189ec765aa7ed75c8acfea42df31fdae71f97be")
NIMBLE_GITHUB = ("bespokelabsai/nimble", "62076b4f2d365b5879dafcf7f6dd072a1fe76df7")
OPEN_JEV_GITHUB = ("Zefan-Cai/Open-Jev", "3308a15ccd7eea1df7a37d6ddc39b023b801ba16")

# (Hugging Face dataset, revision, file in the repo, local path, SHA-256)
HF_FILES = [
    *[(OPEN_JEV, f"raw/{c}/train.jsonl.gz", f"open-jev/raw/{c}/train.jsonl.gz", h) for c, h in [
        ("amount-extraction-control-v1", "fdbc56d0da9dd29a5432e22a345419018b86141769bf1eca2275fd7584d107db"),
        ("browser-drone-expansion-v1-redistributable", "4d1d7a99eec9703ec87c1fa50e82db5444aef559b4f85a92edad94f962005834"),
        ("citation-control-v1", "ba611b3a124460c8bed5e9f5b47e237c0b820b154a33ced4d1ed5371627a45cd"),
        ("context-retention-control-v1", "d40f4e4936d95cab9956ee8867bf8608054f0845a69a6fa6b78e223fe8f8d5d7"),
        ("email-selection-control-v1", "39030185f75f8b5fa079f410b7cb4687830a3f2376591212537547fea689084e"),
        ("entity-alignment-control-v1", "87c9d68ce3ffa5d1dd4a80b764a9e66f8dcf10c21f1958e147fcb649bfb881e4"),
        ("ir-control-v1", "b55297bd98bb187fb5fa71df92382ffb23fecbbd9234f1c185491b0eb4ff9dd3"),
        ("mailroom-control-v1", "ebbfae0b4024ac8dbbbc709f54b8cc02023d5953a0b712888a9d2ffbb5189811"),
        ("phone-extraction-control-v1", "a7d0092ed374ef69281097f6d5986f1bde53e5d9fcf71c5f7f848d464a3115d6"),
        ("silent-failure-control-v1", "083d158c2341a85b29d36bba5b2f0d80aa72ca66aa3e24a82ad97699b97d23e8"),
        ("sponsor-segment-control-v1", "9d0552334e5fec052efa99be0517e9878b9598e3022d7c8a03c134d7f6f63ddb"),
    ]],
    *[(OPEN_JEV, f"raw/browser-drone-expansion-v1-redistributable/{s}.jsonl.gz",
       f"open-jev/raw/browser-drone-expansion-v1-redistributable/{s}.jsonl.gz", h) for s, h in [
        ("calibration", "4f7ba6a48d1784d4152678353d6abee82ab37d5a6855e14871c0c85d62584636"),
        ("validation", "8302593d799e58d3e50bf07915e06508e29074d0b07177de121b8e23cc6cfc6c"),
        ("test", "7b6a91e1518e4df4614af4caa1980594d9f9332312936ec6fddd2c479d4860ee"),
        ("ood", "9fce350e617496e9b3c4437f7b4752d5e16755dbb61d689aa52028c618a469e3"),
    ]],
    *[(OPEN_JEV_V11, f"raw/community-hard-mix-v2-redistributable/{s}.jsonl.gz",
       f"open-jev-v1.1/raw/community-hard-mix-v2-redistributable/{s}.jsonl.gz", h) for s, h in [
        ("train", "35fc8560a3f3823e0e257c1507da7345388b1e72765e9fb612bcea07c2c3ed60"),
        ("calibration", "3f8c0d243a9a230cedf65ea58c70e8a99dbc8bc0eb3852a0a0e089b1fcd7c6b2"),
        ("validation", "f43b38d39c62a205adc3369ad0e305191382d308e01b614c8a800be52945eee5"),
        ("test", "fabadcae4ab1f2c1c000dff969aecd59ec68cc8ed6d18603f43ffe0ec693eec9"),
        ("ood", "0b960c7f9d88bb64f1d282dddc2b45f1b5ae2d35920dfe515422d3dfe4319fef"),
    ]],
    (WANLI, "train.jsonl", "wanli/train.jsonl", "85058cf017a911e89242dc29fa0a4ddaad3664cb923dc0a82145fdda14b694e5"),
    (WANLI, "test.jsonl", "wanli/test.jsonl", "4276e0af7fcdf657d1ab7beb54eaf025fda592a76c9ee86b63b7871953fc74fd"),
    *[(NEMOTRON, f"data/train-{i:05d}-of-00004.parquet", f"nemotron/train-{i:05d}-of-00004.parquet", h) for i, h in enumerate([
        "5eae1d6e1962a2201d873bc3acd802c9cdc6f3ebef01cbdda90740570cc533ee",
        "1b6e1bf9ef0e4a53f5bfefc7460b2cb929a73feee895189ace4b6649e34b56f8",
        "6f88ddbfe5cf50a2faa17f1602ca58488c4c7d4dc7be4f917ee27d23dd892341",
        "70e3bbbc7cb3409f38e5bb5c7dbe9d86f5ed795a9e3bc2f95a970b97db6b73b3",
    ])],
    (OPENSCIENCE, "train/OpenScienceReasoning-2.parquet", "openscience/OpenScienceReasoning-2.parquet",
     "e82e9c7de7ac4befc12734d1b7b42b62c9896ba5bdef4e5d17cb7e7e8bceaef2"),
    # evaluation file: used only to exclude evaluation items from training
    (MMLU_PRO, "data/test-00000-of-00001.parquet", "mmlu_pro/test-00000-of-00001.parquet",
     "0e24a191921c2f453518a537a8b2117bd137e7714d4ef1565e9ba06c1ecb9ad8"),
]
# (GitHub repository, commit, file in the repo, local path, SHA-256)
GITHUB_FILES = [
    (NIMBLE_GITHUB, "data/train.jsonl", "nimble/train.jsonl", "beadbb9b81837f7c339e090cd210ce91f65a55f2a3ada1ce9b623765b8d6fe2e"),
    # evaluation file: used only to exclude evaluation items from training
    (NIMBLE_GITHUB, "data/eval.jsonl", "nimble/eval.jsonl", "8e9e48b8de5206593912ae01ddc95bd77e40ad2ecf4c9292c1711290eca0d896"),
    (OPEN_JEV_GITHUB, "jev/__init__.py", "open-jev-code/jev/__init__.py", "039d6c25d5ebea9724711756aff885f44d2226abbeea4abb99dfbc7b0b35ce7e"),
    (OPEN_JEV_GITHUB, "jev/data.py", "open-jev-code/jev/data.py", "97c1764a5090f60476397baf889a864359e0d59b48d99ddf3b34335d3e1b138e"),
    (OPEN_JEV_GITHUB, "jev/community_diversity_v2.py", "open-jev-code/jev/community_diversity_v2.py",
     "2336b2e25f01f98e436924f7ff46a0057b05b772fa4d2e482a4db6af7f2a61c4"),
]


def fetch_hf(repo: tuple, name: str, dest: Path) -> Path:
    from huggingface_hub import hf_hub_download
    local = Path(hf_hub_download(repo[0], name, repo_type="dataset", revision=repo[1]))
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(local, dest)
    return dest


def fetch_github(repo: tuple, name: str, dest: Path) -> Path:
    import requests
    response = requests.get(f"https://raw.githubusercontent.com/{repo[0]}/{repo[1]}/{name}", timeout=(20, 300))
    response.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(response.content)
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=WORK / "sources")
    ap.add_argument("--only", nargs="*", help="local path prefixes to fetch (default: everything)")
    a = ap.parse_args()
    report = []
    jobs = [("hf", *entry) for entry in HF_FILES] + [("github", *entry) for entry in GITHUB_FILES]
    for kind, repo, name, local, digest in jobs:
        if a.only and not any(local.startswith(prefix) for prefix in a.only):
            continue
        dest = a.out / local
        if not (dest.is_file() and sha256_file(dest) == digest):
            (fetch_hf if kind == "hf" else fetch_github)(repo, name, dest)
        require(sha256_file(dest) == digest, f"SHA-256 mismatch for {local}")
        report.append({"file": local, "source": f"{repo[0]}@{repo[1]}:{name}", "sha256": digest})
        print(json.dumps({"ok": local}), flush=True)
    (a.out / "sources.json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
