"""Isolated live worker: python worker.py --run-dir PATH."""

import argparse
import json
import os
import signal
from pathlib import Path

from engine import (
    ConfigurationError,
    EventWriter,
    TaskPlanningError,
    atomic_json,
    private_diagnostic,
    run_live,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    folder = args.run_dir.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    os.chmod(folder, 0o700)
    request = json.loads((folder / "request.json").read_text())
    if not isinstance(request.get("query"), str) or not request["query"].strip():
        parser.error("request.query must be nonempty")
    if request.get("decision_model") in {"webjev", "jev"}:
        # engine.load_runtime routes this run's decisions by it.
        os.environ["BROWSER_AGENT_DECISION"] = request["decision_model"]
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    writer = EventWriter(folder)
    try:
        run_live(request, writer)
    except BaseException as exc:
        private_diagnostic(folder, exc, writer.elapsed(), "worker")
        # Never send provider exceptions, URLs containing tokens or credentials to clients.
        # BrowserUnavailable (vendored transport) only names the local Chrome endpoint.
        safe = isinstance(exc, (TaskPlanningError, ConfigurationError)) or type(exc).__name__ == "BrowserUnavailable"
        result = {
            "status": "failed",
            "summary": str(exc) if safe else "The run could not connect. Please try again.",
            "columns": [],
            "rows": [],
            "missing_fields": [],
            "notes": [],
            "error_code": type(exc).__name__,
        }
        if not (folder / "raw_state.json").exists():
            atomic_json(
                folder / "raw_state.json",
                {"status": "failed", "history": [], "error_code": type(exc).__name__},
            )
        if not (folder / "observations.json").exists():
            atomic_json(folder / "observations.json", [])
        atomic_json(folder / "result.json", result)
        writer.emit("result", result)
        writer.phase("failed", result["summary"])
    finally:
        writer.close()


if __name__ == "__main__":
    main()
