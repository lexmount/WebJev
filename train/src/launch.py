#!/usr/bin/env python3
"""Check the inputs and start (or resume) a training run in the background.

    python src/launch.py [--config configs/webjev-35b-a3b.yaml] [--run-id NAME] [--check] [--foreground]
    python src/launch.py --resume runs/NAME

A new run gets runs/<run-id>/ with a snapshot of src/, the resolved configuration and all logs. The run keeps
going after you log out; resume it with --resume after an interruption.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

TRAIN_DIR = Path(__file__).resolve().parents[1]
UPSTREAM_COMMIT = "c4daaac28af9fea95d627015cffa2dd5a5926ee6"
BASE_FILES = ("config.json", "model.safetensors.index.json", "tokenizer.json", "LICENSE")


def load_config(path: Path) -> dict:
    import yaml
    config = yaml.safe_load(path.read_text())
    for key, value in config["paths"].items():
        p = Path(value).expanduser()
        config["paths"][key] = str(p if p.is_absolute() else (TRAIN_DIR / p).resolve())
    return config


def preflight(config: dict) -> dict:
    paths, problems = config["paths"], []
    upstream = Path(paths["upstream"])
    try:
        head = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        head = None
    if head != UPSTREAM_COMMIT:
        problems.append(f"upstream checkout {upstream} is at {head}, expected {UPSTREAM_COMMIT}; run setup.sh")
    for name in BASE_FILES:
        if not (Path(paths["base_model"]) / name).is_file():
            problems.append(f"base model file missing: {Path(paths['base_model']) / name}")
    items = Path(paths["items"])
    if not items.is_file() or not Path(str(items) + ".json").is_file():
        problems.append(f"training items missing: {items} (+ .json); run data/build_all.sh")
        data = None
    else:
        data = {k: v for k, v in json.loads(Path(str(items) + ".json").read_text()).items() if k in ("items", "questions", "tokens")}
    gpus, busy = [], []
    try:
        text = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total",
                                        "--format=csv,noheader,nounits"], text=True)
        for line in text.strip().splitlines():
            index, uuid, name, memory = [x.strip() for x in line.split(",")]
            gpus.append(dict(index=int(index), uuid=uuid, name=name, memory_mib=int(memory)))
        apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True)
        busy = [line.strip() for line in apps.splitlines() if line.strip()]
    except (subprocess.CalledProcessError, FileNotFoundError):
        problems.append("nvidia-smi is not available")
    need = config["resources"]
    usable = [g for g in gpus if g["memory_mib"] >= need["min_gpu_memory_mib"]]
    if len(usable) < need["gpus"]:
        problems.append(f"needs {need['gpus']} GPUs with >= {need['min_gpu_memory_mib']} MiB; found {len(usable)}")
    if busy:
        problems.append(f"GPUs already run compute processes: {busy}")
    return dict(ok=not problems, problems=problems, data=data, gpus=gpus)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=TRAIN_DIR / "configs" / "webjev-35b-a3b.yaml")
    ap.add_argument("--run-id", help="new run name (default: timestamp)")
    ap.add_argument("--resume", type=Path, help="resume an existing run directory")
    ap.add_argument("--check", action="store_true", help="only check inputs and GPUs; start nothing")
    ap.add_argument("--foreground", action="store_true", help="wait for the run instead of detaching")
    a = ap.parse_args()
    if a.resume:
        run = a.resume.resolve()
        config = json.loads((run / "resolved-config.json").read_text())
        if not list((run / "checkpoints").glob("step-*/ready.json")):
            print(f"note: {run} has no complete checkpoint yet; training restarts from the base model", flush=True)
    else:
        config = load_config(a.config)
    report = preflight(config)
    if a.check:
        print(json.dumps(dict(config=config, preflight=report), indent=2))
        sys.exit(0 if report["ok"] else 1)
    if not report["ok"]:
        raise SystemExit("preflight failed:\n  " + "\n  ".join(report["problems"]) + "\n(see --check)")
    if not a.resume:
        run_id = a.run_id or "webjev-35b-a3b-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,100}", run_id):
            raise SystemExit("invalid --run-id")
        run = Path(config["paths"]["runs"]) / run_id
        if run.exists():
            raise SystemExit(f"run directory already exists: {run} (use --resume)")
        run.mkdir(parents=True)
        shutil.copytree(TRAIN_DIR / "src", run / "code", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        config["run"] = str(run)
        config["model_out"] = str(Path(config["paths"]["exports"]) / run_id)
        (run / "resolved-config.json").write_text(json.dumps(config, indent=2) + "\n")
    env = dict(os.environ, **{k: str(v) for k, v in config["environment"].items()})
    env.update(WEBJEV_UPSTREAM=config["paths"]["upstream"], PYTHONPATH=str(run / "code"),
               WEBJEV_EXPERT_CHUNK_TOKENS=str(config["training"]["expert_chunk_tokens"]))
    with (run / "pipeline.log").open("a") as log:
        child = subprocess.Popen([sys.executable, str(run / "code" / "supervisor.py"), "--run", str(run)],
                                 cwd=run, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
    print(json.dumps(dict(run=str(run), supervisor_pid=child.pid, training_log=str(run / "training.log"),
                          metrics=str(run / "curve.jsonl"), tensorboard=str(run / "tensorboard"),
                          model_out=config["model_out"]), indent=2), flush=True)
    if a.foreground:
        sys.exit(child.wait())
    time.sleep(2)
    if child.poll() is not None:
        raise SystemExit(f"the run exited immediately ({child.returncode}); see {run / 'pipeline.log'}")


if __name__ == "__main__":
    main()
