"""Background supervisor of one run: train (resuming up to twice after a failure), then export the model.

Started by launch.py with the run directory; everything it runs logs into that directory. Create a file named
STOP in the run directory to prevent further retries (the current step is not interrupted).
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def atomic(path: Path, value) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    run = ap.parse_args().run
    config = json.loads((run / "resolved-config.json").read_text())
    code, t, paths = run / "code", config["training"], config["paths"]
    started = time.time()

    def execute(phase: str, command: list) -> int:
        atomic(run / "pipeline-status.json", dict(phase=phase, pid=os.getpid(), started=started, command=command))
        with (run / f"{phase}.log").open("a") as log:
            return subprocess.run(command, cwd=code, stdout=log, stderr=subprocess.STDOUT).returncode

    train = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={config['resources']['gpus']}",
             str(code / "train.py"), "--out", str(run), "--model", paths["base_model"], "--items", paths["items"],
             "--lr", str(t["learning_rate"]), "--schedule", t["lr_schedule"], "--warmup", str(t["warmup_steps"]),
             "--batch_tokens", str(t["batch_tokens"]), "--batches_per_step", str(t["batches_per_update"]),
             "--batch_seed", str(t["batch_seed"]), "--model_seed", str(t["model_seed"]),
             "--micro_tokens", str(t["micro_tokens"]), "--grad_clip", str(t["grad_clip"]),
             "--checkpoint_every", str(config["checkpoint"]["every_steps"]),
             "--keep_checkpoints", str(config["checkpoint"]["keep_last"]),
             "--log_every", str(config["logging"]["log_every_steps"]), "--max_steps", str(t["max_steps"]), "--resume"]
    try:
        if not (run / "training-complete.json").exists():
            for attempt in range(3):
                code_ = execute("training", train)
                if code_ == 0:
                    break
                if (run / "STOP").exists() or not list((run / "checkpoints").glob("step-*/ready.json")) or attempt == 2:
                    raise RuntimeError(f"training failed (exit {code_}); see {run / 'training.log'}")
                time.sleep(10)
        if (run / "training-complete.json").exists() and config["export"]["after_training"]:
            if not Path(config["model_out"]).exists() and execute(
                    "export", [sys.executable, str(code / "export.py"), "--run", str(run),
                               "--base", paths["base_model"], "--out", config["model_out"]]):
                raise RuntimeError(f"export failed; see {run / 'export.log'}")
        phase = "complete" if (run / "training-complete.json").exists() else "stopped_at_max_steps"
        atomic(run / "pipeline-status.json", dict(phase=phase, elapsed_seconds=round(time.time() - started)))
    except Exception as exc:
        atomic(run / "pipeline-status.json", dict(phase="failed", error=f"{type(exc).__name__}: {exc}",
                                                  elapsed_seconds=round(time.time() - started)))
        raise


if __name__ == "__main__":
    main()
