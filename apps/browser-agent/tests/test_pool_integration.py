"""Pool handoff is private, one-use, and released even when a worker cannot start."""

import asyncio
import importlib
import json
from pathlib import Path
import stat
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
api = importlib.import_module("backend.app")


class Pool:
    def __init__(self, lease):
        self.lease = lease
        self.acquired = []
        self.released = []

    async def acquire(self, run_id):
        self.acquired.append(run_id)
        return self.lease

    async def release(self, run_id):
        self.released.append(run_id)


def setup_run(tmp_path, monkeypatch, lease):
    pool = Pool(lease)
    monkeypatch.setattr(api, "DATA", tmp_path)
    monkeypatch.setattr(api, "session_pool", pool)
    run = api.Run("12345678-1234-1234-1234-123456789012", "test-owner", {"query": "Look up information on a public page"})
    return run, pool


def test_lease_handoff_is_private_and_released_after_worker_exit(tmp_path, monkeypatch):
    lease = {"id": "session_test", "connect_url": "wss://private.invalid/secret", "inspect_url": "https://viewer.invalid/", "created_at": "2026-09-20T00:00:00Z"}
    run, pool = setup_run(tmp_path, monkeypatch, lease)
    captured = {}

    async def spawn(*args, **kwargs):
        lease_path = Path(kwargs["env"]["BROWSER_AGENT_SESSION_LEASE_FILE"])
        captured["lease"] = json.loads(lease_path.read_text())
        assert stat.S_IMODE(lease_path.stat().st_mode) == 0o600
        assert "connect_url" not in json.dumps(run.snapshot())
        assert "connect_url" not in (run.folder / "request.json").read_text()
        return type("Exited", (), {"returncode": 0})()

    monkeypatch.setattr(api.asyncio, "create_subprocess_exec", spawn)

    async def scenario():
        await run.start()
        await run.monitor_task

    asyncio.run(scenario())
    assert captured["lease"] == lease
    assert pool.acquired == pool.released == [run.id]
    assert not (run.folder / "session_lease.json").exists()
    assert json.loads((run.folder / "session_allocation.json").read_text())["source"] == "pool"


def test_spawn_failure_releases_lease_and_closes_log(tmp_path, monkeypatch):
    run, pool = setup_run(tmp_path, monkeypatch, {"id": "session_test"})

    async def spawn(*args, **kwargs):
        raise OSError("cannot spawn")

    monkeypatch.setattr(api.asyncio, "create_subprocess_exec", spawn)
    with pytest.raises(OSError):
        asyncio.run(run.start())
    assert pool.released == [run.id]
    assert not (run.folder / "session_lease.json").exists()
    assert run.log_handle.closed


def test_empty_pool_cold_fallback_does_not_inherit_another_lease(tmp_path, monkeypatch):
    run, pool = setup_run(tmp_path, monkeypatch, None)
    monkeypatch.setenv("BROWSER_AGENT_SESSION_LEASE_FILE", "/another/run/private.json")

    async def spawn(*args, **kwargs):
        assert "BROWSER_AGENT_SESSION_LEASE_FILE" not in kwargs["env"]
        return type("Exited", (), {"returncode": 0})()

    monkeypatch.setattr(api.asyncio, "create_subprocess_exec", spawn)

    async def scenario():
        await run.start()
        await run.monitor_task

    asyncio.run(scenario())
    assert json.loads((run.folder / "session_allocation.json").read_text())["source"] == "cold"
    assert pool.released == [run.id]
