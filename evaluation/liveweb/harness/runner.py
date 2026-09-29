#!/usr/bin/env python3
"""Run the live-web tasks for ONE decision model: parallel episodes, resume, infra retries.

Every episode runs in its own worker process (harness/episode.py) because the agent engine patches modules at load
time and is not built to run several tasks in one process.

Result layout (one directory per task):
    <results-dir>/<task_id>/record.json      how the episode ended (bucket, steps, answer, decisions)
    <results-dir>/<task_id>/evidence.json    what the verifier captured from the live page
    <results-dir>/<task_id>/apps/            the agent's own event log, model calls and final state
    <results-dir>/<task_id>/worker.log

Rules:
  * a task with a record.json whose bucket is not `infra` is skipped (resume); `--force` re-runs everything;
  * only the `infra` bucket is retried (`--retry-infra`, default 2), each attempt in a clean directory;
  * the verifier runs afterwards (`python -m verifier.judge <results-dir>`), see run_eval.sh.

Browser (the same rule as the app): BROWSER=lexmount (Lexmount Browser, needs LEXMOUNT_API_KEY and
LEXMOUNT_PROJECT_ID; the reported runs used it) or BROWSER=local (your Chrome started with --remote-debugging-port,
CHROME_CDP_URL, default http://127.0.0.1:9222). Without BROWSER, lexmount is used when LEXMOUNT_API_KEY is set. Every
run on a local Chrome gets its own browser context in that one Chrome, so parallel episodes are capped
(LOCAL_CHROME_WORKERS, default 2).

Decision endpoint:
  * self-hosted WebJev:  --decision-url http://127.0.0.1:8200 [--decision-key-env WEBJEV_API_KEY]
  * Jev on OpenRouter:   --decision-url https://openrouter.ai/api/v1 --decision-model typesafe/jev-1.13-20260917
                         --decision-key-env OPENROUTER_API_KEY
  * Jev on TypeSafe:     --decision-url https://api.typesafe.ai --decision-model jev-1.13.0
                         --decision-key-env TYPESAFE_API_KEY
"""
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIVEWEB = HERE.parent
TASKS_DIR = LIVEWEB / "tasks"
sys.path.insert(0, str(HERE))

from buckets import classify_detail, run_with_retry  # noqa: E402
from cleanup import release_noted_browser  # noqa: E402

LOCAL_CHROME_WORKERS = 2


class InfraError(RuntimeError):
    """The environment failed, not the agent: the episode is filed as infra and retried."""


def load_tasks(selector: str | None) -> list[dict]:
    tasks = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(TASKS_DIR.glob("*.json"))]
    if not selector:
        return tasks
    if selector.startswith("@"):
        wanted = [line.strip() for line in Path(selector[1:]).read_text().splitlines() if line.strip()]
    else:
        wanted = [t.strip() for t in selector.split(",") if t.strip()]
    by_id = {t["task_id"]: t for t in tasks}
    missing = [t for t in wanted if t not in by_id]
    if missing:
        raise SystemExit(f"unknown task ids: {missing}")
    return [by_id[t] for t in wanted]


def has_result(task_dir: Path) -> bool:
    """A record.json that counts: readable and not in the infra bucket."""
    try:
        bucket = json.loads((task_dir / "record.json").read_text(encoding="utf-8")).get("bucket")
    except (OSError, ValueError):
        return False
    return bucket != "infra"


def browser_backend() -> str:
    """"lexmount" or "local", the app's rule: BROWSER when it names one, else lexmount when LEXMOUNT_API_KEY is set."""
    choice = (os.environ.get("BROWSER") or "").strip().lower()
    if choice in ("lexmount", "local"):
        return choice
    return "lexmount" if os.environ.get("LEXMOUNT_API_KEY") else "local"


def worker_env(args) -> dict:
    env = dict(os.environ)
    env["BROWSER"] = args.browser
    # Relative dates in the agent's prompts ("today is ...") are resolved in this zone; the reported runs used it.
    env.setdefault("BROWSER_AGENT_TIMEZONE", "Asia/Shanghai")
    env["TEXT_MODEL"] = args.text_model
    required = ["TEXT_MODEL_API_KEY", "TEXT_MODEL_BASE_URL"]
    if args.browser == "lexmount":
        required += ["LEXMOUNT_API_KEY", "LEXMOUNT_PROJECT_ID"]
    for name in required:
        if not env.get(name):
            raise SystemExit(f"{name} is not set (see .env.example)")
    key = env.get(args.decision_key_env, "") if args.decision_key_env else ""
    url = args.decision_url.rstrip("/")
    env.pop("TYPESAFE_API_KEY", None)   # set below only for the TypeSafe route; it switches the runtime's endpoint
    if "openrouter.ai" in url:          # Jev through OpenRouter (the runtime's default route)
        env.update(BROWSER_AGENT_DECISION="jev", OPENROUTER_API_KEY=key, TYPESAFE_MODEL=args.decision_model)
    elif "typesafe.ai" in url:          # Jev through TypeSafe's own API
        env.update(BROWSER_AGENT_DECISION="jev", TYPESAFE_API_KEY=key, TYPESAFE_MODEL=args.decision_model)
    else:                               # a self-hosted WebJev server with the same decisions route
        env.pop("OPENROUTER_API_KEY", None)
        env.update(BROWSER_AGENT_DECISION="webjev", DECISION_URL=url, DECISION_MODEL=args.decision_model,
                   DECISION_API_KEY=key,
                   # the self-hosted GPU is the bottleneck: a hedged duplicate request only lengthens its queue
                   JEV_HEDGE_MS="0")
    if args.decision_timeout:
        env["DECISION_TIMEOUT_S"] = str(args.decision_timeout)
    return env


def run_episode(task: dict, task_dir: Path, args) -> dict:
    started = time.monotonic()
    task_file = task_dir / "task.json"
    task_file.write_text(json.dumps(task, ensure_ascii=False))
    cmd = [args.agent_python, str(HERE / "episode.py"), "--task", str(task_file), "--task-dir", str(task_dir),
           "--max-seconds", str(args.max_seconds)]
    with open(task_dir / "worker.log", "w") as log:
        proc = subprocess.Popen(cmd, env=worker_env(args), stdout=log, stderr=subprocess.STDOUT)
        try:
            proc.wait(timeout=args.max_seconds + 600)
        except subprocess.TimeoutExpired:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                proc.kill()
            release_noted_browser(task_dir)  # the killed worker could not release its cloud session itself
            raise InfraError(f"episode worker still running {args.max_seconds + 600}s after start") from None
    out = task_dir / "episode.json"
    if not out.exists():
        tail = (task_dir / "worker.log").read_text()[-300:]
        raise InfraError(f"episode worker exited {proc.returncode} without a result: {tail}")
    record = json.loads(out.read_text())
    if record.get("infra_error") and record.get("bucket") != "infra":
        raise InfraError(record["infra_error"])
    record.update(model=args.label, decision_model=args.decision_model, text_model=args.text_model,
                  seconds=round(time.monotonic() - started, 1))
    return record


def commit(final: Path, partial: Path, record: dict) -> None:
    """Move a finished attempt into place: the old result is renamed away first, so an interruption at any point
    leaves at least one complete result on disk."""
    (partial / "record.json").write_text(json.dumps(record, ensure_ascii=False, indent=1, default=str))
    stale = None
    if final.exists():
        stale = final.with_name(f"{final.name}.stale-{int(time.time() * 1000)}")
        final.rename(stale)
    partial.rename(final)
    if stale is not None:
        shutil.rmtree(stale, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", required=True, type=Path, help="this model's result directory")
    ap.add_argument("--label", required=True, help="model name used in reports")
    ap.add_argument("--decision-url", required=True)
    ap.add_argument("--decision-model", default="webjev-35b-a3b",
                    help="sent as `model` in every decision request (self-hosted servers ignore it)")
    ap.add_argument("--decision-key-env", default="", help="environment variable holding the endpoint's key")
    ap.add_argument("--decision-timeout", type=float, default=0,
                    help="seconds for one decision call; 0 keeps the runtime's 25 s")
    ap.add_argument("--text-model", default=os.environ.get("TEXT_MODEL") or "deepseek-v4.1-flash",
                    help="OpenAI-compatible model that writes typed text and the final answer")
    ap.add_argument("--tasks", default="", help="comma-separated task ids or @file; default: all 125")
    ap.add_argument("--workers", type=int, default=10,
                    help="parallel episodes (= parallel browsers); capped for a local Chrome")
    ap.add_argument("--retry-infra", type=int, default=2)
    ap.add_argument("--max-seconds", type=int, default=900, help="per-task time budget")
    ap.add_argument("--force", action="store_true", help="re-run tasks that already have a result")
    ap.add_argument("--agent-python", default=os.environ.get("WEBJEV_AGENT_PYTHON") or sys.executable,
                    help="interpreter with the agent's requirements installed")
    args = ap.parse_args()

    args.browser = browser_backend()
    if args.browser == "local":
        cap = int(os.environ.get("LOCAL_CHROME_WORKERS") or LOCAL_CHROME_WORKERS)
        if args.workers > cap:
            print(f"[{args.label}] local Chrome: {args.workers} -> {cap} parallel episodes (one Chrome serves them "
                  f"all; set LOCAL_CHROME_WORKERS to change)", flush=True)
            args.workers = cap
    tasks = load_tasks(args.tasks)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    todo = [t for t in tasks if args.force or not has_result(args.results_dir / t["task_id"])]
    print(f"[{args.label}] {len(todo)} to run ({len(tasks) - len(todo)} already done), browser={args.browser}, "
          f"workers={args.workers} -> {args.results_dir}", flush=True)

    def one(task: dict) -> dict:
        final = args.results_dir / task["task_id"]
        partial = final.with_name(final.name + ".partial")
        t0 = time.monotonic()

        def attempt() -> dict:
            if partial.exists():        # every attempt starts in a clean directory
                shutil.rmtree(partial)
            partial.mkdir(parents=True)
            return run_episode(task, partial, args)

        rec, retried = run_with_retry(attempt, retries=args.retry_infra)
        rec["infra_retries"] = retried
        if not rec.get("bucket"):
            rec["bucket"], rec["bucket_reason"] = classify_detail(finish_reason=rec.get("finish_reason"),
                                                                  done_text=rec.get("done_text"))
        rec.setdefault("task_id", task["task_id"])
        rec.setdefault("model", args.label)
        rec.setdefault("seconds", round(time.monotonic() - t0, 1))
        partial.mkdir(parents=True, exist_ok=True)
        commit(final, partial, rec)
        cause = rec.get("infra_error") or rec.get("error") or ""
        print(f"[{args.label}] [{rec.get('bucket')}] {task['task_id']} steps={rec.get('env_steps')} "
              f"{str(cause)[:80]}", flush=True)
        return rec

    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(one, t): t["task_id"] for t in todo}
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001 - one task's bookkeeping must not stop the others
                failed.append(futures[fut])
                print(f"[{args.label}] [batch-error] {futures[fut]}: {type(exc).__name__}: {exc}", flush=True)
    if failed:
        print(f"[{args.label}] {len(failed)} tasks have no result: {','.join(failed)}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
