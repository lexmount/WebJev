"""Run lifecycle regressions using inert workers and local ASGI requests only."""

import asyncio
import importlib
import json
import sys
import threading
import uuid
from collections import deque
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
api = importlib.import_module("backend.app")


class InertProcess:
    """Never starts a process, browser, model request, or billable resource."""

    returncode = None

    def terminate(self):
        self.returncode = -15


@pytest.fixture
def isolated_api(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "DATA", tmp_path)
    monkeypatch.setattr(api, "runs", {})
    monkeypatch.setattr(api, "submission_times", {})
    monkeypatch.setattr(api, "global_submission_times", deque())
    monkeypatch.setattr(api, "submit_lock", asyncio.Lock())
    monkeypatch.setattr(api, "MAX_ACTIVE", 1)
    starts = []

    async def fake_start(run):
        starts.append(run.id)
        run.process = InertProcess()
        await run.publish("phase", {"phase": "running", "message": "mock worker"})

    monkeypatch.setattr(api.Run, "start", fake_start)
    return starts


def client():
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api.app), base_url="http://testserver"
    )


async def create(client):
    await client.get("/api/presets")
    return await client.post("/api/runs", json={"query": "Look up example information on a public website"})


def test_cancelled_but_alive_worker_still_occupies_global_capacity(isolated_api):
    async def scenario():
        async with client() as owner, client() as stranger:
            response = await create(owner)
            assert response.status_code == 201
            run_id = response.json()["id"]
            run = api.runs[run_id]
            assert (await owner.post(f"/api/runs/{run_id}/cancel")).status_code == 200
            assert run.status == "cancelled"
            assert run.process.returncode is None
            assert (await create(owner)).status_code == 429
            assert (await create(stranger)).status_code == 429
            assert len(isolated_api) == 1
            run.process.returncode = 0
            assert (await create(stranger)).status_code == 201
            assert len(isolated_api) == 2

    asyncio.run(scenario())


def test_real_restore_with_empty_event_history_resets_sse_and_polling_cursor(isolated_api):
    async def scenario():
        async with client() as owner:
            response = await create(owner)
            run_id = response.json()["id"]
            run = api.runs[run_id]
            await run.publish("result", {"status": "partial", "summary": "Partial record kept"})
            api.atomic_json(
                run.folder / "owner.json",
                {
                    "owner": run.owner,
                    "created_at": run.created_at,
                },
            )
            api.atomic_json(run.folder / "request.json", run.payload)
            # Simulate the persisted running task being loaded after a restart.
            api.runs.clear()
            async with api.lifespan(api.app):
                restored = api.runs[run_id]
                assert restored.status == "failed"
                assert not restored.events
                assert restored.sequence == 0
                update = (await owner.get(f"/api/runs/{run_id}/updates?after=800")).json()
                assert update["snapshot"]["status"] == "failed"
                assert update["snapshot"]["result"]["summary"] == "Partial record kept"
                assert update["last_event_id"] == 0
                stream = await owner.get(
                    f"/api/runs/{run_id}/events",
                    headers={
                        "Last-Event-ID": "800",
                    },
                )
                assert stream.status_code == 200
                assert "event: result" in stream.text
                assert "Partial record kept" in stream.text
                assert "event: phase" in stream.text
                assert '"phase": "failed"' in stream.text
                assert "id: 0" in stream.text

    asyncio.run(scenario())


@pytest.mark.parametrize("cursor_kind", ["future", "evicted"])
def test_nonempty_history_also_resets_out_of_range_cursor(isolated_api, cursor_kind):
    async def scenario():
        async with client() as owner:
            run_id = (await create(owner)).json()["id"]
            run = api.runs[run_id]
            run.process.returncode = 0
            run.status = "completed"
            run.result = {"status": "completed", "summary": "Final result"}
            run.sequence = 200
            run.events = deque(
                [
                    {
                        "id": 200,
                        "type": "phase",
                        "data": {"phase": "completed"},
                    }
                ],
                maxlen=1600,
            )
            cursor = 201 if cursor_kind == "future" else 1
            update = (await owner.get(f"/api/runs/{run_id}/updates?after={cursor}")).json()
            assert update["snapshot"]["result"]["summary"] == "Final result"
            stream = await owner.get(f"/api/runs/{run_id}/events?after={cursor}")
            assert "event: result" in stream.text
            assert "Final result" in stream.text
            assert "id: 200" in stream.text

    asyncio.run(scenario())


@pytest.mark.parametrize("operation,final_status", [("resume", "running"), ("cancel", "cancelled")])
def test_resume_and_cancel_wait_for_inflight_input_then_reject_queued_input(
    isolated_api,
    operation,
    final_status,
):
    async def scenario():
        async with client() as owner:
            run_id = (await create(owner)).json()["id"]
            run = api.runs[run_id]
            run.status = "needs_login"
            (run.folder / "browser_control.json").write_text("{}")
            entered, release = threading.Event(), threading.Event()
            received = []

            class BlockingController:
                def send(self, body):
                    entered.set()
                    assert release.wait(timeout=3), "test did not release fake input"
                    received.append(body["text"])

            run.input_controller = BlockingController()
            first = asyncio.create_task(
                owner.post(
                    f"/api/runs/{run_id}/input",
                    json={
                        "type": "text",
                        "text": "first-test-password",
                    },
                )
            )
            assert await asyncio.to_thread(entered.wait, 1)
            transition = asyncio.create_task(owner.post(f"/api/runs/{run_id}/{operation}"))
            # Give the transition a turn to queue on the shared lock before the
            # second input request. No real worker or browser is involved.
            await asyncio.sleep(0.02)
            queued = asyncio.create_task(
                owner.post(
                    f"/api/runs/{run_id}/input",
                    json={
                        "type": "text",
                        "text": "late-test-password",
                    },
                )
            )
            await asyncio.sleep(0.02)
            try:
                assert not transition.done()
                assert not queued.done()
                assert run.status == "needs_login"
                assert not (run.folder / "control.json").exists()
            finally:
                release.set()
            first_response, transition_response, queued_response = await asyncio.gather(
                first,
                transition,
                queued,
            )
            assert first_response.status_code == 200
            assert transition_response.status_code == 200
            assert queued_response.status_code == 409
            assert received == ["first-test-password"]
            assert run.status == final_status
            expected_command = "resume" if operation == "resume" else "stop"
            assert (
                json.loads((run.folder / "control.json").read_text())["action"] == expected_command
            )
            assert "test-password" not in json.dumps(run.snapshot())
            assert "test-password" not in "".join(p.read_text() for p in run.folder.glob("*.json"))

    asyncio.run(scenario())


def test_wire_safe_preserves_valid_unicode_and_repairs_nested_surrogates():
    value = {
        "field\ud800": ["text", "🚀", "\ud83d\ude80", {"quote": "cut\ud83d"}],
        "count": 3,
        "empty": None,
        "passed": True,
    }
    clean = api.wire_safe(value)
    assert clean == {
        "field�": ["text", "🚀", "🚀", {"quote": "cut�"}],
        "count": 3,
        "empty": None,
        "passed": True,
    }
    json.dumps(clean, ensure_ascii=False).encode("utf-8")


def test_worker_surrogate_cannot_break_state_result_polling_or_sse(isolated_api):
    async def scenario():
        async with client() as owner:
            run_id = (await create(owner)).json()["id"]
            run = api.runs[run_id]
            await run.publish("step", {"step": 1, "label": "Open the page\ud800"})
            await run.publish(
                "result",
                {
                    "status": "partial",
                    "summary": "Kept😀\ud83d",
                    "columns": [{"key": "name", "label": "Name\udc00"}],
                    "rows": [{"cells": {"name": {"value": "Content\ud83d", "quote": "Content\ud83d"}}}],
                },
            )
            await run.publish("phase", {"phase": "partial", "message": "Check finished\ud800"})
            run.process.returncode = 0
            for suffix in ("", "/result", "/updates?after=9999"):
                response = await owner.get(f"/api/runs/{run_id}{suffix}")
                assert response.status_code == 200
                response.text.encode("utf-8")
                assert "�" in response.text
            stream = await owner.get(f"/api/runs/{run_id}/events?after=9999")
            assert stream.status_code == 200
            assert "Kept😀�" in stream.text
            assert "Name�" in stream.text
            persisted = json.loads((run.folder / "state.json").read_text())
            assert persisted["steps"][0]["label"] == "Open the page�"

    asyncio.run(scenario())


def test_restored_legacy_surrogate_state_is_also_wire_safe(isolated_api):
    async def scenario():
        async with client() as owner:
            run_id = (await create(owner)).json()["id"]
            run = api.runs[run_id]
            saved = run.snapshot()
            saved.update(
                status="failed",
                message="Old cut\ud800",
                steps=[{"step": 1, "label": "Old page\ud83d"}],
                result={"status": "partial", "summary": "Old result\udc00"},
            )
            api.atomic_json(run.folder / "state.json", saved)
            api.atomic_json(
                run.folder / "owner.json",
                {
                    "owner": run.owner,
                    "created_at": run.created_at,
                },
            )
            api.atomic_json(run.folder / "request.json", run.payload)
            api.runs.clear()
            async with api.lifespan(api.app):
                response = await owner.get(f"/api/runs/{run_id}")
                assert response.status_code == 200
                assert response.json()["steps"][0]["label"] == "Old page�"
                stream = await owner.get(f"/api/runs/{run_id}/events?after=800")
                assert stream.status_code == 200
                assert "Old result�" in stream.text

    asyncio.run(scenario())


def test_concurrent_same_request_id_starts_one_worker_and_retry_costs_no_slot(isolated_api):
    async def scenario():
        async with client() as owner:
            await owner.get("/api/presets")
            payload = {"query": "Look up example information on a public website", "client_request_id": str(uuid.uuid4())}
            responses = await asyncio.gather(
                *[owner.post("/api/runs", json=payload) for _ in range(4)]
            )
            assert all(response.status_code == 201 for response in responses)
            assert len({response.json()["id"] for response in responses}) == 1
            assert len(isolated_api) == 1
            assert len(api.global_submission_times) == 1
            run = api.runs[responses[0].json()["id"]]
            assert len(api.submission_times[run.owner]) == 1
            run.status, run.process.returncode = "completed", 0
            retry = await owner.post("/api/runs", json=payload)
            assert retry.status_code == 201
            assert retry.json()["id"] == run.id
            assert retry.json()["status"] == "completed"
            assert len(isolated_api) == 1

    asyncio.run(scenario())


def test_same_owner_tabs_can_run_concurrently_up_to_global_limit(isolated_api, monkeypatch):
    monkeypatch.setattr(api, "MAX_ACTIVE", 3)

    async def scenario():
        async with client() as owner, client() as stranger:
            await owner.get("/api/presets")
            payloads = [{"query": "Separate tasks from different browser tabs", "client_request_id": str(uuid.uuid4())}
                        for _ in range(4)]
            responses = await asyncio.gather(*[owner.post("/api/runs", json=p) for p in payloads[:3]])
            assert [r.status_code for r in responses] == [201, 201, 201]
            ids = {r.json()["id"] for r in responses}
            assert len(ids) == len(isolated_api) == 3
            assert len({api.runs[rid].owner for rid in ids}) == 1
            blocked = await owner.post("/api/runs", json=payloads[3])
            assert blocked.status_code == 429
            assert "at most 3 can run at once" in blocked.json()["detail"]
            # Retry is idempotent even when every slot is occupied.
            retry = await owner.post("/api/runs", json=payloads[0])
            assert retry.status_code == 201 and retry.json()["id"] == responses[0].json()["id"]
            assert len(isolated_api) == 3
            assert (await create(stranger)).status_code == 429
            assert (await stranger.get(f"/api/runs/{responses[0].json()['id']}")).status_code == 404
            first = api.runs[responses[0].json()["id"]]
            first.status, first.process.returncode = "completed", 0
            assert (await owner.post("/api/runs", json=payloads[3])).status_code == 201

    asyncio.run(scenario())


def test_recover_pending_submission_by_id_does_not_adopt_another_tab(isolated_api, monkeypatch):
    monkeypatch.setattr(api, "MAX_ACTIVE", 3)

    async def scenario():
        async with client() as owner, client() as stranger:
            await owner.get("/api/presets")
            nonce = str(uuid.uuid4())
            first = (await owner.post("/api/runs", json={"query": "The same query from two browser tabs", "client_request_id": nonce})).json()
            second = (await owner.post("/api/runs", json={"query": first["query"], "client_request_id": str(uuid.uuid4())})).json()
            assert first["id"] != second["id"]
            exact = await owner.get("/api/runs/latest", params={"client_request_id": nonce})
            assert exact.json()["run"]["id"] == first["id"]
            assert (await owner.get("/api/runs/latest", params={"client_request_id": str(uuid.uuid4())})).json() == {"run": None}
            assert (await stranger.get("/api/runs/latest", params={"client_request_id": nonce})).json() == {"run": None}

    asyncio.run(scenario())


@pytest.mark.parametrize("change", ["query", "target_website"])
def test_same_request_id_with_changed_task_is_rejected(isolated_api, change):
    async def scenario():
        async with client() as owner:
            await owner.get("/api/presets")
            payload = {
                "query": "Look up example information on a public website",
                "target_website": "https://example.com",
                "client_request_id": str(uuid.uuid4()),
            }
            first = await owner.post("/api/runs", json=payload)
            assert first.status_code == 201
            replacement = "Look up other information on a public website" if change == "query" else "https://example.org"
            second = await owner.post("/api/runs", json={**payload, change: replacement})
            assert second.status_code == 409
            assert len(isolated_api) == 1
            assert api.runs[first.json()["id"]].payload[change] == payload[change]

    asyncio.run(scenario())


def test_nonce_is_owner_scoped_and_latest_never_returns_another_owner(isolated_api, monkeypatch):
    monkeypatch.setattr(api, "MAX_ACTIVE", 3)

    async def scenario():
        async with client() as one, client() as two, client() as newcomer:
            for session in (one, two, newcomer):
                assert (await session.get("/api/runs/latest")).json() == {"run": None}
            nonce = str(uuid.uuid4())
            payload = {"query": "Look up example information on a public website", "client_request_id": nonce}
            first = (await one.post("/api/runs", json=payload)).json()
            other = (await two.post("/api/runs", json=payload)).json()
            assert first["id"] != other["id"]
            assert len(isolated_api) == 2
            first_run, other_run = api.runs[first["id"]], api.runs[other["id"]]
            first_run.created_at = "2026-01-01T00:00:00+00:00"
            other_run.created_at = "2026-12-31T23:59:59+00:00"
            assert (await one.get("/api/runs/latest")).json()["run"]["id"] == first["id"]
            assert (await two.get("/api/runs/latest")).json()["run"]["id"] == other["id"]
            assert (await newcomer.get("/api/runs/latest")).json() == {"run": None}
            first_run.status, first_run.process.returncode = "completed", 0
            fresh = await one.post(
                "/api/runs", json={**payload, "client_request_id": str(uuid.uuid4())}
            )
            assert fresh.status_code == 201
            assert (await one.get("/api/runs/latest")).json()["run"]["id"] == fresh.json()["id"]
            assert (await two.get("/api/runs/latest")).json()["run"]["id"] == other["id"]
            retry = await one.post("/api/runs", json=payload)
            assert retry.json()["id"] == first["id"]
            assert len(isolated_api) == 3

    asyncio.run(scenario())


def test_invalid_client_request_id_is_rejected_before_start(isolated_api):
    async def scenario():
        async with client() as owner:
            response = await owner.post(
                "/api/runs",
                json={
                    "query": "Look up example information on a public website",
                    "client_request_id": "not-a-uuid",
                },
            )
            assert response.status_code == 422
            assert isolated_api == []

    asyncio.run(scenario())
