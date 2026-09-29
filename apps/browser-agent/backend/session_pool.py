"""Private, single-use Lexmount warm sessions. No task pages or answers are preloaded."""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path


def configure_browser_env():
    """Lexmount Browser credentials come from the environment only (LEXMOUNT_API_KEY, LEXMOUNT_PROJECT_ID)."""
    required = ("LEXMOUNT_API_KEY", "LEXMOUNT_PROJECT_ID")
    if not all(os.environ.get(k) for k in required):
        raise RuntimeError("browser_credentials_missing")


class LexmountProvider:
    """Warm Lexmount Browser sessions, adopted once each by a run's worker (see browser_cdp.LexmountBrowser)."""

    def __init__(self):
        configure_browser_env()
        from lexmount import Lexmount

        self.client = Lexmount(base_url="https://api.lexmount.com", timeout=30)

    async def create(self):
        session = await asyncio.to_thread(
            self.client.sessions.create, poll_timeout_sec=90, window_size="1600,780"
        )
        return {
            "id": session.id,
            "connect_url": session.connect_url,
            "inspect_url": session.inspect_url,
            "created_at": getattr(session, "created_at", None) or datetime.now(UTC).isoformat(),
        }

    async def healthy(self, lease):
        from websockets.asyncio.client import connect

        async with connect(lease["connect_url"], open_timeout=10, close_timeout=2) as ws:
            for number, method in enumerate(("Browser.getVersion", "Target.getTargets"), 1):
                await ws.send(json.dumps({"id": number, "method": method}))
                async with asyncio.timeout(10):
                    while True:
                        response = json.loads(await ws.recv())
                        if response.get("id") == number:
                            if "error" in response or "result" not in response:
                                return False
                            break
        return True

    async def delete(self, session_id):
        try:
            await asyncio.to_thread(self.client.sessions.delete, session_id=session_id)
        except Exception as exc:
            if type(exc).__name__ != "SessionNotFoundError":
                raise

    async def close(self):
        await asyncio.to_thread(self.client.close)


class SessionPool:
    """Provider contract: async create()->lease, healthy(lease)->bool, delete(id), close()."""

    def __init__(
        self,
        data_dir,
        target=10,
        provider=None,
        *,
        max_creating=3,
        ttl_seconds=900,
        health_interval=60,
        retry_base=2,
    ):
        self.data_dir = Path(data_dir)
        self.target = max(0, int(target))
        self.provider = provider
        self.max_creating = max(1, int(max_creating))
        self.ttl_seconds = ttl_seconds
        self.health_interval = health_interval
        self.retry_base = retry_base
        self._lock = asyncio.Lock()
        self._wake = asyncio.Event()
        self._ready, self._active, self._owned = {}, {}, {}
        self._jobs = set()
        self._creating = self._checking = 0
        self._closing = False
        self._runner = None
        self._failures = 0
        self._retry_at = 0.0
        self._error = None

    def stats(self):
        return {
            "target": self.target,
            "ready": len(self._ready),
            "active": len(self._active),
            "creating": self._creating,
            "checking": self._checking,
            "healthy": not self._closing and self._error is None,
            "error_code": self._error,
        }

    async def start(self):
        if self._runner is not None or self._closing:
            return
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._runner = asyncio.create_task(self._maintain())

    def _spawn(self, coroutine):
        task = asyncio.create_task(coroutine)
        self._jobs.add(task)
        task.add_done_callback(self._jobs.discard)
        return task

    def _diagnose(self, operation, exc):
        # Deliberately exclude provider messages, URL strings and credential values.
        entry = {
            "operation": operation,
            "exception_type": type(exc).__name__,
            "at": datetime.now(UTC).isoformat(),
        }
        try:
            path = self.data_dir / "session_pool_diagnostics.jsonl"
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(entry) + "\n")
            path.chmod(0o600)
        except OSError:
            pass

    def _failed(self, code):
        self._error = code
        self._failures += 1
        self._retry_at = time.monotonic() + min(
            60, self.retry_base * 2 ** min(self._failures - 1, 5)
        )

    async def _delete(self, session_id):
        if session_id not in self._owned:
            return
        try:
            await self.provider.delete(session_id)
            self._owned.pop(session_id, None)
        except Exception as exc:
            self._diagnose("delete", exc)
            self._error = "session_close_failed"

    async def _create(self):
        lease = None
        try:
            lease = await self.provider.create()
            if not isinstance(lease, dict) or not all(lease.get(k) for k in ("id", "connect_url")):
                raise ValueError("invalid_session")
            self._owned[lease["id"]] = lease
            if self._closing or not await self.provider.healthy(lease):
                raise RuntimeError("session_not_ready")
            async with self._lock:
                if self._closing:
                    raise RuntimeError("pool_closed")
                now = time.monotonic()
                self._ready[lease["id"]] = {"lease": lease, "ready_at": now, "checked_at": now}
                self._failures = 0
                self._error = None
        except Exception as exc:
            self._diagnose("warm", exc)
            if not self._closing:
                self._failed("session_warm_failed")
            if lease and isinstance(lease, dict) and lease.get("id"):
                await self._delete(lease["id"])
        finally:
            self._creating -= 1
            self._wake.set()

    async def _check(self, entry):
        lease = entry["lease"]
        try:
            if not await self.provider.healthy(lease):
                raise RuntimeError("session_unhealthy")
            async with self._lock:
                if not self._closing and time.monotonic() - entry["ready_at"] < self.ttl_seconds:
                    entry["checked_at"] = time.monotonic()
                    self._ready[lease["id"]] = entry
                    retire = False
                else:
                    retire = True
            if retire:
                await self._delete(lease["id"])
        except Exception as exc:
            self._diagnose("health", exc)
            self._failed("session_health_failed")
            await self._delete(lease["id"])
        finally:
            self._checking -= 1
            self._wake.set()

    async def _maintain(self):
        while not self._closing:
            self._wake.clear()
            if self.target and self.provider is None and time.monotonic() >= self._retry_at:
                try:
                    self.provider = LexmountProvider()
                except Exception as exc:
                    self._diagnose("provider", exc)
                    self._failed("browser_provider_unavailable")
            if self.provider is not None:
                async with self._lock:
                    now = time.monotonic()
                    for session_id, entry in list(self._ready.items()):
                        if now - entry["ready_at"] >= self.ttl_seconds:
                            self._ready.pop(session_id)
                            self._spawn(self._delete(session_id))
                        elif (
                            now - entry["checked_at"] >= self.health_interval
                            and self._checking < self.max_creating
                        ):
                            self._ready.pop(session_id)
                            self._checking += 1
                            self._spawn(self._check(entry))
                    missing = self.target - len(self._ready) - self._creating - self._checking
                    if now >= self._retry_at:
                        for _ in range(max(0, min(missing, self.max_creating - self._creating))):
                            self._creating += 1
                            self._spawn(self._create())
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=0.5)
            except TimeoutError:
                pass

    async def acquire(self, run_id):
        async with self._lock:
            if self._closing or run_id in self._active:
                return None
            now = time.monotonic()
            for session_id, entry in list(self._ready.items()):
                if now - entry["ready_at"] >= self.ttl_seconds:
                    self._ready.pop(session_id)
                    self._spawn(self._delete(session_id))
                    continue
                if now - entry["checked_at"] > self.health_interval:
                    continue
                self._ready.pop(session_id)
                self._active[run_id] = entry["lease"]
                self._wake.set()
                return dict(entry["lease"])
            self._wake.set()
            return None

    async def release(self, run_id):
        async with self._lock:
            lease = self._active.pop(run_id, None)
        if lease:
            await self._delete(lease["id"])
        self._wake.set()

    async def close(self):
        if self._closing:
            return
        self._closing = True
        self._wake.set()
        if self._runner:
            await self._runner
        # Do not cancel SDK create threads: wait for their IDs and close any late sessions.
        while self._jobs:
            await asyncio.gather(*list(self._jobs), return_exceptions=True)
        if self.provider:
            for _ in range(2):
                await asyncio.gather(*(self._delete(i) for i in list(self._owned)))
            await self.provider.close()
        self._ready.clear()
        self._active.clear()
