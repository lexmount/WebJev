"""One epoch of supervised fine-tuning over the training mixture, on one node of 8 GPUs (torchrun).

Batching follows the upstream recipe: items are grouped into length-bucketed batches of at most `batch_tokens`
padded tokens (upstream `batches_by_tokens`, seed `batch_seed`), and `batches_per_step` consecutive batches form
one optimizer update. The loss of an update is the mean of the per-batch mean losses. Whole items are spread
over the 8 ranks longest-first; each rank runs them as micro-batches of at most `micro_tokens` padded tokens,
weighted so that the gradient equals that mean. The final partial group is trained too, so every item is used
exactly once.

Checkpoints hold the trainable weights (overlay.pt) plus each rank's optimizer shard and RNG state; `--resume`
continues from the newest complete checkpoint of the same run directory.

    torchrun --standalone --nproc_per_node 8 src/train.py --out RUN_DIR --model BASE_DIR --items ITEMS [...]
"""
import argparse
from collections import deque
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import random
import time

from model_runtime import TrainingModel, collate, to_cuda
from sharded_recipe_optim import ShardedRecipeOptim
from batch_schedule import assign_batches, checkpoint_due, prune_complete_checkpoints
from decider.train import batches_by_tokens, loss_fn
import torch
import torch.distributed as dist


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="run directory (checkpoints, logs, metrics)")
    p.add_argument("--model", required=True, help="base model directory")
    p.add_argument("--items", required=True, help="items.pkl written by data/assemble.py (items.pkl.json next to it)")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--schedule", choices=["constant", "warmup_cosine"], default="constant")
    p.add_argument("--warmup", type=int, default=150, help="warm-up updates for warmup_cosine")
    p.add_argument("--batch_tokens", type=int, default=8192)
    p.add_argument("--batches_per_step", type=int, default=16)
    p.add_argument("--batch_seed", type=int, default=1)
    p.add_argument("--model_seed", type=int, default=0)
    p.add_argument("--micro_tokens", type=int, default=2048)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--checkpoint_every", type=int, default=2000)
    p.add_argument("--keep_checkpoints", type=int, default=2)
    p.add_argument("--log_every", type=int, default=5)
    p.add_argument("--max_steps", type=int, default=0, help="stop after this many updates (0: full epoch)")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.batches_per_step < 1 or args.checkpoint_every < 1 or args.keep_checkpoints < 1:
        raise ValueError("batch and checkpoint settings must be positive")
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", timeout=timedelta(hours=1), device_id=torch.device("cuda", local))
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 8:
        raise ValueError("the optimizer shard layout of this trainer expects 8 ranks")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device, started = torch.device("cuda", local), time.time()
    status = dict(phase="loading_items", world=world, pid=os.getpid(), args=vars(args))

    def record(**kw):
        if rank == 0:
            status.update(kw, updated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            atomic_json(out / "status.json", status)

    def event(value):
        if rank == 0:
            line = json.dumps(value)
            print(line, flush=True)
            with (out / "events.jsonl").open("a") as f:
                f.write(line + "\n")

    torch.manual_seed(args.model_seed)
    random.seed(args.model_seed)
    record()
    manifest = json.loads(Path(args.items + ".json").read_text())
    if rank == 0 and sha256(args.items) != manifest["sha256"]:
        raise ValueError("items.pkl does not match the SHA-256 in items.pkl.json")
    dist.barrier()
    with open(args.items, "rb") as f:
        items = pickle.load(f)
    batches = batches_by_tokens(items, args.batch_tokens, random.Random(args.batch_seed))
    total = math.ceil(len(batches) / args.batches_per_step)
    limit = min(total, args.max_steps) if args.max_steps else total
    data = dict(items=len(items), questions=sum(len(x["slots"]) for x in items), tokens=sum(len(x["ids"]) for x in items),
                original_batches=len(batches), total_steps=total, batch_tokens=args.batch_tokens,
                batch_seed=args.batch_seed, batches_per_step=args.batches_per_step,
                final_partial_group=len(batches) % args.batches_per_step)
    if (data["items"], data["questions"], data["tokens"]) != (manifest["items"], manifest["questions"], manifest["tokens"]):
        raise ValueError("items.pkl counts differ from items.pkl.json")
    record(phase="loading_model", data=data, total_steps=total, step=0)
    event(dict(event="data", **data))

    checkpoint = None
    if args.resume:
        ready = sorted((out / "checkpoints").glob("step-*/ready.json"))
        if ready:
            checkpoint = ready[-1].parent
    model = TrainingModel(args.model).to(device).train()
    seen = dict(items_seen=0, questions_seen=0, tokens_seen=0)
    start_step, prior_seconds = 0, 0.0
    if checkpoint:
        metadata = json.loads((checkpoint / "ready.json").read_text())
        if metadata["data"] != data:
            raise ValueError("resume: the data or schedule differs from the checkpoint")
        if sha256(checkpoint / "overlay.pt") != metadata["files"]["overlay.pt"]["sha256"]:
            raise ValueError("resume: overlay.pt is corrupted")
        overlay = torch.load(checkpoint / "overlay.pt", map_location="cpu", weights_only=True)
        if set(overlay) != {n for n, _ in model.trainable_named()}:
            raise ValueError("resume: trainable parameter names differ")
        model.lm.load_state_dict(overlay, strict=False)
        del overlay
        start_step, prior_seconds = metadata["step"], metadata["training_seconds"]
        seen.update(metadata["seen"])
    record(phase="initializing_optimizer", step=start_step, parameters=model.parameter_summary())
    optimizer = ShardedRecipeOptim(model.trainable_named(), lr=args.lr)
    optimizer.allocate_gradients()
    if checkpoint:
        path = checkpoint / f"optimizer-rank-{rank:02d}.pt"
        if sha256(path) != metadata["files"][path.name]["sha256"]:
            raise ValueError("resume: optimizer shard is corrupted")
        state = torch.load(path, map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"], device)
        random.setstate(state["python_rng"])
        del state
    summary = optimizer.summary()
    record(optimizer=summary)
    event(dict(event="optimizer", **summary))
    named = model.trainable_named()
    params = [p for _, p in named]
    if rank == 0:
        atomic_json(out / "run-config.json", dict(args=vars(args), data=data, parameters=model.parameter_summary(),
                                                  optimizer=summary, torch=torch.__version__,
                                                  initial_weights="checkpoint" if checkpoint else "base model"))
    writer = None
    if rank == 0:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(str(out / "tensorboard"))
        except ImportError:
            pass
    torch.cuda.empty_cache()
    dist.barrier()
    train_start, recent = time.time(), deque(maxlen=20)

    def lr_at(step):
        if args.schedule == "constant":
            return args.lr
        if step < args.warmup:
            return args.lr * step / args.warmup
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, (step - args.warmup) / max(1, total - args.warmup))))

    def save_checkpoint(step, training_seconds):
        record(phase="checkpointing", step=step)
        folder = out / "checkpoints" / f"step-{step:06d}"
        folder.mkdir(parents=True, exist_ok=True)
        state = dict(optimizer=optimizer.state_dict(), torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state(device), python_rng=random.getstate(), step=step)
        path = folder / f"optimizer-rank-{rank:02d}.pt"
        tmp = path.with_suffix(".tmp")
        torch.save(state, tmp)
        tmp.replace(path)
        del state
        files = {path.name: dict(bytes=path.stat().st_size, sha256=sha256(path))}
        if rank == 0:
            overlay = {n: p.detach().cpu() for n, p in named}
            path = folder / "overlay.pt"
            tmp = path.with_suffix(".tmp")
            torch.save(overlay, tmp)
            tmp.replace(path)
            del overlay
            files[path.name] = dict(bytes=path.stat().st_size, sha256=sha256(path))
        gathered = [None] * world
        dist.all_gather_object(gathered, files)
        if rank == 0:
            all_files = {name: info for shard in gathered for name, info in shard.items()}
            atomic_json(folder / "ready.json", dict(step=step, data=data, seen=seen, training_seconds=training_seconds,
                                                    files=all_files, world=world))
            prune_complete_checkpoints(out / "checkpoints", args.keep_checkpoints)
            event(dict(event="checkpoint", step=step, path=str(folder)))
        dist.barrier()

    for step in range(start_step + 1, limit + 1):
        t = time.time()
        optimizer.zero_grad()
        group = batches[(step - 1) * args.batches_per_step:step * args.batches_per_step]
        assignments, rank_loads = assign_batches(group, items, world, args.micro_tokens)
        weighted_ce, local_padded = torch.zeros((), device=device), 0
        record(phase="training", step=step - 1, active_step=step)
        for task in assignments[rank]:
            batch = to_cuda(collate([items[i] for i in task["indices"]], model.tok.pad_token_id), device)
            logits = model(batch)
            loss, ce = loss_fn(logits, batch["golds"], batch["nopts"])
            if int((batch["golds"] >= 0).sum().item()) != task["nvalid"]:
                raise ValueError("scheduled answer count differs from the collated batch")
            (loss * task["loss_weight"]).backward()
            weighted_ce += ce * task["loss_weight"]
            local_padded += batch["input_ids"].numel()
            del logits, loss, ce, batch
        optimizer.average_gradients()
        norm = torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        if not torch.isfinite(norm).item():
            raise RuntimeError(f"non-finite gradient norm at step {step}")
        stats = torch.tensor([float(weighted_ce), local_padded], device=device, dtype=torch.float64)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        ce_value = float(stats[0]) / world
        if not math.isfinite(ce_value):
            raise RuntimeError(f"non-finite loss at step {step}")
        optimizer.step(lr_at(step))
        torch.cuda.synchronize(device)
        elapsed = time.time() - t
        recent.append(elapsed)
        step_ids = [i for original in group for i in original]
        seen["items_seen"] += len(step_ids)
        seen["questions_seen"] += sum(len(items[i]["slots"]) for i in step_ids)
        tokens = sum(len(items[i]["ids"]) for i in step_ids)
        seen["tokens_seen"] += tokens
        training_seconds = prior_seconds + time.time() - train_start
        metrics = dict(step=step, total_steps=total, ce=ce_value, grad_norm=float(norm), lr=lr_at(step),
                       step_seconds=elapsed, tokens_per_second=tokens / elapsed,
                       peak_memory_GiB=torch.cuda.max_memory_allocated(device) / 2 ** 30,
                       training_seconds=training_seconds, eta_seconds=(total - step) * sum(recent) / len(recent), **seen)
        record(phase="training", **metrics)
        if rank == 0:
            with (out / "curve.jsonl").open("a") as f:
                f.write(json.dumps(metrics) + "\n")
            if writer:
                for key in ("ce", "grad_norm", "lr", "tokens_per_second", "peak_memory_GiB"):
                    writer.add_scalar(f"train/{key}", metrics[key], step)
        if step % args.log_every == 0 or step <= 3 or step == limit:
            event(dict(event="train", **metrics))
        if checkpoint_due(step, total, args.checkpoint_every) or step == limit:
            save_checkpoint(step, training_seconds)
        del weighted_ce, stats, norm
    if limit == total:
        assert seen["items_seen"] == data["items"] and seen["tokens_seen"] == data["tokens"]
        record(phase="training_complete", step=limit, **seen)
        if rank == 0:
            atomic_json(out / "training-complete.json", dict(status="complete", steps=total, data=data, seen=seen,
                                                             training_seconds=prior_seconds + time.time() - train_start))
    else:
        record(phase="stopped_at_max_steps", step=limit, **seen)
    if writer:
        writer.close()
    event(dict(event="done", step=limit, seconds=time.time() - started))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
