"""Offline warm-pool lifecycle tests with an injected, non-network provider."""

import asyncio
import importlib.util
import json
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "backend" / "session_pool.py"
SPEC = importlib.util.spec_from_file_location("browser_pool_under_test", PATH)
pool_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pool_module)
SessionPool = pool_module.SessionPool


class FakeProvider:
    def __init__(self, health=True, gate=None, delete_failures=0):
        self.created = []
        self.deleted = []
        self.health = health
        self.gate = gate
        self.delete_failures = delete_failures
        self.inflight = 0
        self.max_inflight = 0
        self.closed = False

    async def create(self):
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        if self.gate:
            await self.gate.wait()
        await asyncio.sleep(0)
        session_id = f"owned-{len(self.created)}"
        self.created.append(session_id)
        self.inflight -= 1
        return {
            "id": session_id,
            "connect_url": f"wss://example.org/{session_id}?token=PRIVATE",
            "inspect_url": f"https://example.org/{session_id}",
            "created_at": "2026-09-20T00:00:00Z",
        }

    async def healthy(self, lease):
        return self.health

    async def delete(self, session_id):
        if self.delete_failures:
            self.delete_failures -= 1
            raise RuntimeError("private provider details")
        self.deleted.append(session_id)

    async def close(self):
        self.closed = True


async def until(predicate, timeout=2):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.005)


def test_ten_ready_atomic_unique_acquire_and_refill(tmp_path):
    async def run():
        provider = FakeProvider()
        pool = SessionPool(tmp_path, target=10, provider=provider)
        await pool.start()
        await until(lambda: pool.stats()["ready"] == 10)
        assert provider.max_inflight <= 3
        leases = await asyncio.gather(*(pool.acquire(f"run-{i}") for i in range(12)))
        taken = [lease for lease in leases if lease]
        assert len(taken) == 10
        assert len({lease["id"] for lease in taken}) == 10
        assert await pool.acquire("run-0") is None
        await until(lambda: pool.stats()["ready"] == 10)
        assert pool.stats()["active"] == 10
        await pool.release("run-0")
        await pool.release("run-0")
        assert provider.deleted.count(taken[0]["id"]) == 1
        assert pool.stats()["ready"] == 10  # used sessions never return to ready
        await pool.close()
        assert set(provider.deleted) == set(provider.created)
        assert provider.closed
        assert await pool.acquire("after-close") is None

    asyncio.run(run())


def test_health_failure_is_never_ready_and_backoff_applies(tmp_path):
    async def run():
        provider = FakeProvider(health=False)
        pool = SessionPool(tmp_path, target=1, provider=provider, retry_base=10)
        await pool.start()
        await until(lambda: bool(provider.deleted))
        assert await pool.acquire("run") is None
        assert len(provider.created) == 1
        assert pool.stats()["error_code"] == "session_warm_failed"
        await pool.close()

    asyncio.run(run())


def test_expired_ready_session_is_not_leased(tmp_path):
    async def run():
        provider = FakeProvider()
        pool = SessionPool(tmp_path, target=1, provider=provider)
        await pool.start()
        await until(lambda: pool.stats()["ready"] == 1)
        first = provider.created[0]
        pool._ready[first]["ready_at"] -= 1000
        assert await pool.acquire("run") is None
        await until(lambda: first in provider.deleted and pool.stats()["ready"] == 1)
        lease = await pool.acquire("run")
        assert lease["id"] != first
        await pool.close()

    asyncio.run(run())


def test_periodic_health_check_replaces_unhealthy_session(tmp_path):
    async def run():
        provider = FakeProvider()
        pool = SessionPool(tmp_path, target=1, provider=provider, retry_base=10)
        await pool.start()
        await until(lambda: pool.stats()["ready"] == 1)
        first = provider.created[0]
        provider.health = False
        pool._ready[first]["checked_at"] -= 1000
        pool._wake.set()
        await until(lambda: first in provider.deleted)
        assert pool.stats()["ready"] == 0
        assert pool.stats()["error_code"] == "session_health_failed"
        await pool.close()

    asyncio.run(run())


def test_close_waits_for_late_create_and_closes_only_owned_sessions(tmp_path):
    async def run():
        gate = asyncio.Event()
        provider = FakeProvider(gate=gate)
        pool = SessionPool(tmp_path, target=2, provider=provider)
        await pool.start()
        await until(lambda: provider.inflight == 2)
        close = asyncio.create_task(pool.close())
        await asyncio.sleep(0)
        assert not close.done()
        gate.set()
        await close
        assert set(provider.deleted) == set(provider.created)
        assert len(provider.deleted) == 2
        assert "other-user-session" not in provider.deleted

    asyncio.run(run())


def test_release_failure_keeps_ownership_for_shutdown_retry(tmp_path):
    async def run():
        provider = FakeProvider(delete_failures=1)
        pool = SessionPool(tmp_path, target=1, provider=provider)
        await pool.start()
        await until(lambda: pool.stats()["ready"] == 1)
        lease = await pool.acquire("run")
        await pool.release("run")
        assert lease["id"] in pool._owned
        assert pool.stats()["error_code"] == "session_close_failed"
        await pool.close()
        assert lease["id"] in provider.deleted
        assert not pool._owned

    asyncio.run(run())


def test_stats_never_disclose_connection_or_provider_details(tmp_path):
    async def run():
        provider = FakeProvider()
        pool = SessionPool(tmp_path, target=1, provider=provider)
        await pool.start()
        await until(lambda: pool.stats()["ready"] == 1)
        public = json.dumps(pool.stats())
        assert all(
            word not in public for word in ("PRIVATE", "connect_url", "inspect_url", "owned-")
        )
        await pool.close()

    asyncio.run(run())


def test_zero_target_never_initializes_real_provider(tmp_path, monkeypatch):
    def forbidden():
        raise AssertionError("disabled pool initialized provider")

    monkeypatch.setattr(pool_module, "LexmountProvider", forbidden)

    async def run():
        pool = SessionPool(tmp_path, target=0)
        await pool.start()
        await asyncio.sleep(0)
        assert await pool.acquire("run") is None
        assert pool.stats()["ready"] == 0
        await pool.close()

    asyncio.run(run())


def test_real_provider_uses_sdk_window_format_without_network(monkeypatch):
    import sys
    from types import SimpleNamespace

    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            id="test",
            connect_url="wss://example.org",
            inspect_url="https://example.org",
            created_at="now",
        )

    monkeypatch.setenv("LEXMOUNT_API_KEY", "fake-key")
    monkeypatch.setenv("LEXMOUNT_PROJECT_ID", "fake-project")
    client = SimpleNamespace(sessions=SimpleNamespace(create=create))
    monkeypatch.setitem(sys.modules, "lexmount", SimpleNamespace(Lexmount=lambda **kwargs: client))
    provider = pool_module.LexmountProvider()
    lease = asyncio.run(provider.create())
    assert captured == {"poll_timeout_sec": 90, "window_size": "1600,780"}
    assert lease["id"] == "test"


def test_private_diagnostic_keeps_type_but_not_provider_message(tmp_path):
    pool = SessionPool(tmp_path, target=0)
    pool._diagnose("warm", ValueError("wss://example.org?key=DO_NOT_LOG"))
    path = tmp_path / "session_pool_diagnostics.jsonl"
    entry = json.loads(path.read_text())
    assert entry["exception_type"] == "ValueError"
    assert "DO_NOT_LOG" not in path.read_text()
    assert "wss" not in path.read_text()
    assert path.stat().st_mode & 0o777 == 0o600
