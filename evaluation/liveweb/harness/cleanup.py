"""Make sure an episode's browser is released.

The agent releases its own browser at the end of a run: it deletes its Lexmount Browser session, or disposes of the
browser context it opened on a local Chrome. Its session provider also deletes a session named in a failed-create
error. What remains is a worker process that dies or is killed mid-run: its cloud session would keep running (and
count against the project's concurrent-session limit) until it expires. So the episode notes its browser in
`browser.json` as soon as the browser is ready, and this module deletes that session at the end of the episode, or
from the runner after it had to kill the worker. A local Chrome context needs nothing: it is created with
disposeOnDetach and disappears when the worker's connection drops.
"""

from __future__ import annotations

import json
from pathlib import Path

BROWSER_FILE = "browser.json"
LEXMOUNT_API = "https://api.lexmount.com"


def note_browser(task_dir: Path, endpoint: dict) -> dict:
    """Store what identifies this episode's browser (never the CDP URL)."""
    info = {k: endpoint.get(k) for k in ("backend", "session_id", "browser_context_id")}
    (Path(task_dir) / BROWSER_FILE).write_text(json.dumps(info))
    return info


def release_browser(info: dict | None) -> str:
    """Delete this episode's Lexmount Browser session if it is still there. Returns what was done."""
    if not info or info.get("backend") != "lexmount" or not info.get("session_id"):
        return "nothing to release"
    try:
        from lexmount import Lexmount

        client = Lexmount(base_url=LEXMOUNT_API, timeout=30)
    except Exception as exc:  # noqa: BLE001 - best effort
        return f"no client: {type(exc).__name__}"
    try:
        client.sessions.delete(session_id=info["session_id"])
        return "delete sent"
    except Exception:  # noqa: BLE001 - usually already deleted by the agent itself
        return "already closed"
    finally:
        try:
            client.close()
        except Exception:  # noqa: BLE001
            pass


def release_noted_browser(task_dir: Path) -> str:
    try:
        info = json.loads((Path(task_dir) / BROWSER_FILE).read_text())
    except (OSError, ValueError):
        return "nothing noted"
    return release_browser(info)
