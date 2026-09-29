"""WebJev browser-agent web app: owner-scoped live runs; browser execution happens in isolated worker processes."""
from __future__ import annotations

import asyncio
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import hashlib
import hmac
import ipaddress
import json
import os
from pathlib import Path
import re
import secrets
import signal
import sys
import time
from typing import Any
from typing import Literal
from urllib.parse import urlsplit
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .browser_backend import LABELS as BROWSER_LABELS, browser_backend
from .presets import PRESETS, PRESET_BY_ID
from .session_pool import SessionPool
from .diagnostics import read_calls

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DATA = Path(os.environ.get("BROWSER_AGENT_DATA", PROJECT / "runs")).resolve()
DATA.mkdir(parents=True, exist_ok=True)
COOKIE = "browser_agent_owner"
MAX_ACTIVE = int(os.environ.get("BROWSER_AGENT_MAX_ACTIVE", "3"))
MAX_SECONDS = int(os.environ.get("BROWSER_AGENT_MAX_SECONDS", "900"))
TERMINAL = {"completed", "partial", "summarized", "failed", "cancelled"}
DECISION_MODELS = {
    "webjev": {"label": "WebJev-35B-A3B",
               "description": "WebJev, self-hosted behind a Jev-compatible endpoint (DECISION_URL)"},
    "jev": {"label": "Jev 1.13", "description": "TypeSafe's Jev 1.13 through TypeSafe's API or OpenRouter"},
}
EVENT_TYPES = {"phase", "step", "frame", "browser", "pointer", "result"}


def available_decision_models() -> dict[str, bool]:
    """Which decision models this server can call; credentials never leave the server."""
    return {
        "webjev": bool(os.environ.get("DECISION_URL")),
        "jev": bool(os.environ.get("TYPESAFE_API_KEY") or os.environ.get("OPENROUTER_API_KEY")),
    }


def default_decision_model() -> str | None:
    available = available_decision_models()
    preferred = (os.environ.get("BROWSER_AGENT_DEFAULT_DECISION") or "webjev").strip().lower()
    if available.get(preferred):
        return preferred
    return next((name for name, ready in available.items() if ready), None)


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any):
    temporary = path.with_suffix(".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=True))
    temporary.replace(path)


def wire_safe(value):
    # DOM substring limits can split a UTF-16 emoji. Keep one malformed label
    # from terminating the event stream or making every state response fail.
    if isinstance(value, str):
        return value.encode("utf-16", "surrogatepass").decode("utf-16", "replace")
    if isinstance(value, dict):
        return {wire_safe(k): wire_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [wire_safe(v) for v in value]
    return value


def session_secret() -> bytes:
    supplied = os.environ.get("BROWSER_AGENT_SESSION_SECRET")
    if supplied:
        return supplied.encode()
    path = DATA / ".session-secret"
    if not path.exists():
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as stream:
                stream.write(secrets.token_hex(32))
        except FileExistsError:
            pass
    return path.read_bytes()


SECRET = session_secret()


def sign_owner(owner: str) -> str:
    digest = hmac.new(SECRET, owner.encode(), hashlib.sha256).hexdigest()
    return owner + "." + digest


def verify_owner(value: str | None) -> str | None:
    if not value or "." not in value:
        return None
    owner, signature = value.rsplit(".", 1)
    if not re.fullmatch(r"[0-9a-f]{32}", owner):
        return None
    return owner if hmac.compare_digest(sign_owner(owner), value) else None


def public_website(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip()
    if len(value) > 2048:
        raise HTTPException(422, "The website address is too long.")
    parsed = urlsplit(value if "://" in value else "https://" + value)
    hostname = parsed.hostname or ""
    if parsed.scheme not in {"https", "http"} or parsed.username or parsed.password:
        raise HTTPException(422, "Enter the HTTPS address of a public website.")
    if not hostname or "." not in hostname or hostname.endswith((".local", ".internal", ".localhost")):
        raise HTTPException(422, "Only public websites are supported.")
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        address = None
    if address is not None:
        raise HTTPException(422, "Use the domain name of a public website.")
    try:
        port = parsed.port
    except ValueError:
        raise HTTPException(422, "The website port is not valid.")
    if port not in {None, 80, 443}:
        raise HTTPException(422, "Only the standard ports of public websites are supported.")
    return value


class RunRequest(BaseModel):
    query: str = Field(min_length=5, max_length=6000)
    preset_id: str | None = None
    target_website: str | None = Field(default=None, max_length=2048)
    decision_model: Literal["webjev", "jev"] | None = None
    client_request_id: uuid.UUID | None = None


class InputRequest(BaseModel):
    type: Literal["click", "wheel", "text", "key"]
    x: float | None = Field(default=None, ge=0, le=10000)
    y: float | None = Field(default=None, ge=0, le=10000)
    delta_y: float = Field(default=0, ge=-1500, le=1500)
    text: str = Field(default="", max_length=1000)
    key: Literal["Enter", "Tab", "Backspace", "Delete", "Escape", "ArrowLeft", "ArrowUp", "ArrowRight", "ArrowDown", "Home", "End", "a"] = "Enter"
    modifiers: Literal[0, 1, 2, 4, 8] = 0


class Run:
    def __init__(self, identifier: str, owner: str, payload: dict):
        self.id, self.owner, self.payload = identifier, owner, payload
        self.folder = DATA / identifier
        self.folder.mkdir(mode=0o700, exist_ok=True)
        self.created_at = utcnow()
        self.status = "queued"
        self.message = "Preparing the browser"
        self.elapsed_ms = 0
        self.started = time.monotonic()
        self.process: asyncio.subprocess.Process | None = None
        self.events: deque = deque(maxlen=1600)
        self.sequence = 0
        self.steps: dict = {}
        self.frame = None
        self.browser = None
        self.result = None
        self.pointer = None
        self.changed = asyncio.Condition()
        self.monitor_task = None
        self.log_handle = None
        self.cancel_requested = False
        self.input_controller = None
        self.control_lock = asyncio.Lock()
        self.lease_pool = None

    def snapshot(self):
        return {
            "id": self.id, "query": self.payload["query"], "preset_id": self.payload.get("preset_id"),
            "decision_model": self.payload.get("decision_model"),
            "browser_backend": self.payload.get("browser_backend"),
            "client_request_id": self.payload.get("client_request_id"),
            "created_at": self.created_at, "status": self.status, "phase": self.status,
            "message": self.message, "elapsed_ms": self.elapsed_ms,
            "steps": list(self.steps.values()), "frame": self.frame,
            "browser": self.browser, "result": self.result, "pointer": self.pointer,
            "last_event_id": self.sequence,
        }

    async def publish(self, kind: str, data: dict):
        if kind not in EVENT_TYPES or not isinstance(data, dict):
            return
        data = wire_safe(data)
        if self.cancel_requested and kind == "phase" and data.get("phase") != "cancelled":
            return
        if self.cancel_requested and kind == "result":
            data = {**data, "status": "cancelled", "summary": "The run was stopped. This is what was collected before it stopped."}
        if kind == "phase":
            phase = data.get("phase")
            if phase:
                self.status = phase
            self.message = data.get("message", "")
        elif kind == "step":
            self.steps[str(data.get("step", len(self.steps)))] = data
        elif kind == "frame":
            # A worker can name a file, but cannot make the browser fetch external URLs.
            filename = Path(str(data.get("url", data.get("filename", "")))).name
            if not re.fullmatch(r"[a-zA-Z0-9_-]+\.(jpg|jpeg|png)", filename):
                return
            data["url"] = f"/api/runs/{self.id}/frames/{filename}"
            self.frame = data
        elif kind == "browser":
            self.browser = data
        elif kind == "result":
            self.result = data
        elif kind == "pointer":
            self.pointer = data
        self.elapsed_ms = max(self.elapsed_ms, int(data.get("elapsed_ms", 0) or 0))
        self.sequence += 1
        event = {"id": self.sequence, "type": kind, "data": data}
        async with self.changed:
            self.events.append(event)
            self.changed.notify_all()
        if kind != "frame" and kind != "pointer":
            atomic_json(self.folder / "state.json", self.snapshot())

    async def start(self):
        atomic_json(self.folder / "request.json", {**self.payload, "id": self.id, "max_seconds": MAX_SECONDS})
        atomic_json(self.folder / "owner.json", {"owner": self.owner, "created_at": self.created_at})
        (self.folder / "owner.json").chmod(0o600)
        atomic_json(self.folder / "state.json", self.snapshot())
        self.log_handle = (self.folder / "worker.log").open("ab")
        python = os.environ.get("BROWSER_AGENT_PYTHON") or sys.executable
        if not Path(python).exists():
            python = sys.executable
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        env.pop("BROWSER_AGENT_SESSION_LEASE_FILE", None)
        self.lease_pool = session_pool
        try:
            acquired_at = time.monotonic()
            lease = await self.lease_pool.acquire(self.id) if self.lease_pool else None
            atomic_json(self.folder / "session_allocation.json", {
                "source": "pool" if lease else "cold",
                "acquire_ms": round((time.monotonic() - acquired_at) * 1000, 2),
            })
            if lease:
                lease_path = self.folder / "session_lease.json"
                atomic_json(lease_path, lease)
                env["BROWSER_AGENT_SESSION_LEASE_FILE"] = str(lease_path)
            self.process = await asyncio.create_subprocess_exec(
                python, str(HERE / "worker.py"), "--run-dir", str(self.folder),
                cwd=str(PROJECT), stdout=self.log_handle, stderr=self.log_handle,
                env=env, start_new_session=True,
            )
        except BaseException:
            if self.lease_pool:
                await self.lease_pool.release(self.id)
            (self.folder / "session_lease.json").unlink(missing_ok=True)
            self.log_handle.close()
            raise
        self.monitor_task = asyncio.create_task(self.monitor())

    async def monitor(self):
        position = 0
        pending = b""
        path = self.folder / "events.jsonl"
        try:
            while True:
                if path.exists():
                    with path.open("rb") as stream:
                        stream.seek(position)
                        chunk = stream.read()
                        position = stream.tell()
                    pending += chunk
                    lines = pending.split(b"\n")
                    pending = lines.pop()
                    for line in lines:
                        try:
                            event = json.loads(line)
                            await self.publish(event["type"], event["data"])
                        except (ValueError, KeyError, TypeError):
                            continue
                if self.process and self.process.returncode is not None:
                    break
                if time.monotonic() - self.started > MAX_SECONDS + 1000:
                    await self.stop("The run exceeded its time budget and was stopped.")
                await asyncio.sleep(0.15)
            if self.status not in TERMINAL:
                status = "cancelled" if self.cancel_requested else "failed"
                await self.publish("phase", {"phase": status, "message": "The run was stopped." if self.cancel_requested else "The run lost its connection. Please try again; pages received so far are kept.", "elapsed_ms": round((time.monotonic() - self.started)*1000)})
            atomic_json(self.folder / "state.json", self.snapshot())
        finally:
            try:
                if self.log_handle:
                    self.log_handle.close()
                if self.input_controller:
                    self.input_controller.close()
            finally:
                try:
                    if self.lease_pool:
                        await self.lease_pool.release(self.id)
                finally:
                    (self.folder / "session_lease.json").unlink(missing_ok=True)

    async def stop(self, message="The run was stopped."):
        async with self.control_lock:
            if self.status in TERMINAL:
                return
            self.cancel_requested = True
            atomic_json(self.folder / "control.json", {"action": "stop", "at": utcnow()})
            await self.publish("phase", {"phase": "cancelled", "message": message, "elapsed_ms": round((time.monotonic()-self.started)*1000)})
        async def terminate_later():
            await asyncio.sleep(8)
            if self.process and self.process.returncode is None:
                self.process.send_signal(signal.SIGTERM)
                try:
                    await asyncio.wait_for(self.process.wait(), timeout=5)
                except asyncio.TimeoutError:
                    self.process.kill()
        asyncio.create_task(terminate_later())


runs: dict[str, Run] = {}
submission_times: dict[str, deque] = {}
global_submission_times: deque = deque()
submit_lock = asyncio.Lock()
session_pool: SessionPool | None = None


@asynccontextmanager
async def lifespan(app):
    global session_pool
    for state_path in DATA.glob("*/state.json"):
        try:
            folder = state_path.parent
            if str(uuid.UUID(folder.name)) != folder.name:
                continue
            identity = json.loads((folder / "owner.json").read_text())
            payload = wire_safe(json.loads((folder / "request.json").read_text()))
            saved = wire_safe(json.loads(state_path.read_text()))
            restored = Run(folder.name, identity["owner"], payload)
            restored.created_at = identity["created_at"]
            restored.status = saved.get("status", "failed")
            restored.message = saved.get("message", "")
            if restored.status not in TERMINAL:
                restored.status = "failed"
                restored.message = "The server restarted and interrupted this run. Review what was collected or run it again."
            restored.elapsed_ms = saved.get("elapsed_ms", 0)
            restored.steps = {str(s.get("step", i)): s for i,s in enumerate(saved.get("steps", []))}
            restored.frame, restored.browser = saved.get("frame"), saved.get("browser")
            restored.result = saved.get("result")
            runs[restored.id] = restored
        except (OSError, ValueError, TypeError, KeyError):
            continue
    # Warm sessions exist only for Lexmount Browser; a local Chrome needs none.
    pool_size = int(os.environ.get("BROWSER_AGENT_POOL_SIZE", "0")) if browser_backend() == "lexmount" else 0
    local_pool = SessionPool(DATA, target=max(0, pool_size))
    session_pool = local_pool
    await local_pool.start()
    try:
        yield
    finally:
        for run in runs.values():
            if run.process and run.process.returncode is None:
                atomic_json(run.folder / "control.json", {"action": "stop", "at": utcnow()})
                run.process.terminate()
        await local_pool.close()
        if session_pool is local_pool:
            session_pool = None


app = FastAPI(title="WebJev Browser Agent", lifespan=lifespan, docs_url=None, redoc_url=None)


@app.middleware("http")
async def owner_session(request: Request, call_next):
    owner = verify_owner(request.cookies.get(COOKIE))
    fresh = owner is None
    request.state.owner = owner or secrets.token_hex(16)
    if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
        origin = request.headers.get("origin")
        if origin and urlsplit(origin).netloc != request.headers.get("host"):
            return JSONResponse({"detail": "Please use this page to send the request."}, status_code=403)
        if request.headers.get("sec-fetch-site") == "cross-site":
            return JSONResponse({"detail": "Please use this page to send the request."}, status_code=403)
    response = await call_next(request)
    if fresh:
        response.set_cookie(COOKIE, sign_owner(request.state.owner), httponly=True, samesite="lax", secure=request.url.scheme == "https", max_age=30*24*3600)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self'; font-src 'self'; object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


def owned_run(identifier: str, request: Request) -> Run:
    run = runs.get(identifier)
    if not run or run.owner != request.state.owner:
        raise HTTPException(404, "Run not found.")
    return run


@app.get("/api/health")
async def health():
    return {"ok": True, "browser_pool": session_pool.stats() if session_pool else {"enabled": False}}


@app.get("/api/presets")
async def presets():
    return {"presets": PRESETS}


@app.get("/api/config")
async def config():
    available = available_decision_models()
    return {
        "decision_models": [
            {"id": name, "available": available[name], **info} for name, info in DECISION_MODELS.items()
        ],
        "default_decision_model": default_decision_model(),
        "browser": {"id": browser_backend(), "label": BROWSER_LABELS[browser_backend()]},
    }


@app.post("/api/runs", status_code=201)
async def create_run(body: RunRequest, request: Request):
    query = wire_safe(body.query.strip())
    if len(query) < 5:
        raise HTTPException(422, "Describe the web task to run.")
    preset = PRESET_BY_ID.get(body.preset_id or "")
    if body.preset_id and not preset:
        raise HTTPException(422, "Unknown example task.")
    # Edited suggestions are custom tasks: old presets cannot silently change the user's intent.
    preset_id = preset["id"] if preset and query == preset["query"] else None
    website = public_website(body.target_website or (preset["target_website"] if preset_id else None))
    decision = body.decision_model or default_decision_model()
    if decision is None:
        raise HTTPException(503, "No decision model is configured on this server.")
    if not available_decision_models()[decision]:
        raise HTTPException(422, f"{DECISION_MODELS[decision]['label']} is not configured on this server.")
    async with submit_lock:
        owner = request.state.owner
        if body.client_request_id:
            key = str(body.client_request_id)
            existing = next((r for r in runs.values() if r.owner == owner and r.payload.get("client_request_id") == key), None)
            if existing:
                if (existing.payload["query"] != query or existing.payload.get("target_website") != website
                        or existing.payload.get("preset_id") != preset_id
                        or existing.payload.get("decision_model") != decision):
                    raise HTTPException(409, "The task changed. Please submit it again.")
                return existing.snapshot()
        active = [r for r in runs.values() if r.status not in TERMINAL or (r.process is not None and r.process.returncode is None)]
        if len(active) >= MAX_ACTIVE:
            raise HTTPException(429, f"{len(active)} runs are in progress; at most {MAX_ACTIVE} can run at once. Please wait for one to finish.")
        times = submission_times.setdefault(owner, deque())
        now = time.monotonic()
        while global_submission_times and now-global_submission_times[0] > 3600:
            global_submission_times.popleft()
        if len(global_submission_times) >= int(os.environ.get("BROWSER_AGENT_HOURLY_LIMIT", "40")):
            raise HTTPException(429, "This server has reached its hourly run limit. Please try again later.")
        while times and now-times[0] > 3600:
            times.popleft()
        if len(times) >= 20:
            raise HTTPException(429, "You have started many runs this hour. Please try again later.")
        identifier = str(uuid.uuid4())
        run = Run(identifier, owner, {"query": query, "target_website": website, "preset_id": preset_id,
                                      "decision_model": decision, "browser_backend": browser_backend(),
                                      "client_request_id": str(body.client_request_id) if body.client_request_id else None})
        runs[identifier] = run
        times.append(now)
        global_submission_times.append(now)
        try:
            await run.start()
        except Exception:
            await run.publish("phase", {"phase": "failed", "message": "The browser service is unavailable right now. Please try again later."})
    return run.snapshot()


@app.get("/api/runs/latest")
async def latest_run(request: Request, client_request_id: uuid.UUID | None = None):
    owned = [run for run in runs.values() if run.owner == request.state.owner]
    if client_request_id is not None:
        owned = [run for run in owned if run.payload.get("client_request_id") == str(client_request_id)]
    return {"run": max(owned, key=lambda r:r.created_at).snapshot() if owned else None}


@app.get("/api/runs/{identifier}")
async def get_run(identifier: str, request: Request):
    return owned_run(identifier, request).snapshot()


@app.get("/api/runs/{identifier}/result")
async def get_result(identifier: str, request: Request):
    run = owned_run(identifier, request)
    return run.result or {"status": run.status, "summary": "The run is still in progress.", "columns": [], "rows": []}


@app.get("/api/runs/{identifier}/updates")
async def updates(identifier: str, request: Request, after: int = 0):
    run = owned_run(identifier, request)
    batch = [event for event in run.events if event["id"] > after]
    # Do not download a backlog of obsolete video frames after a slow connection.
    latest_frame = next((event for event in reversed(batch) if event["type"] == "frame"), None)
    latest_pointer = next((event for event in reversed(batch) if event["type"] == "pointer"), None)
    batch = [event for event in batch if event["type"] not in {"frame", "pointer"} or event is latest_frame or event is latest_pointer]
    reset = not run.events or after > run.sequence or after < run.events[0]["id"]-1
    return {"events": batch, "last_event_id": run.sequence, "status": run.status, "snapshot": run.snapshot() if reset else None}


@app.get("/api/runs/{identifier}/diagnostics")
async def diagnostics(identifier: str, request: Request, since: str = ""):
    run = owned_run(identifier, request)
    prior_status, _, prior_revision = since.partition("|")
    data = await asyncio.to_thread(read_calls, run.folder, prior_revision if prior_status == run.status else "")
    data["revision"] = f"{run.status}|{data['revision']}"
    # A process interrupted during a request has no response or end timestamp.
    # Keep its duration unknown instead of letting the UI timer run forever.
    if run.status in TERMINAL and data.get("calls"):
        data["calls"] = [
            {**call, "status": "interrupted"} if call.get("status") == "running" else call
            for call in data["calls"]
        ]
    return {**data, "run_status": run.status, "elapsed_ms": run.elapsed_ms}


@app.post("/api/runs/{identifier}/cancel")
async def cancel_run(identifier: str, request: Request):
    run = owned_run(identifier, request)
    await run.stop()
    return run.snapshot()


@app.post("/api/runs/{identifier}/resume")
async def resume_run(identifier: str, request: Request):
    run = owned_run(identifier, request)
    async with run.control_lock:
        if run.status != "needs_login":
            raise HTTPException(409, "This run is not waiting for a sign-in.")
        atomic_json(run.folder / "control.json", {"action": "resume", "at": utcnow()})
        await run.publish("phase", {"phase": "running", "message": "Continuing the task"})
    return run.snapshot()


@app.post("/api/runs/{identifier}/input")
async def browser_input(identifier: str, body: InputRequest, request: Request):
    run = owned_run(identifier, request)
    async with run.control_lock:
        if run.status != "needs_login":
            raise HTTPException(409, "Browser input is only accepted while the run waits for a sign-in.")
        if not (run.folder / "browser_control.json").is_file():
            raise HTTPException(409, "The browser is not ready yet. Please try again shortly.")
        if run.input_controller is None:
            from .input import BrowserInput
            run.input_controller = BrowserInput(run.folder)
        try:
            await asyncio.to_thread(run.input_controller.send, body.model_dump())
        except Exception:
            raise HTTPException(503, "The browser connection was interrupted. Please try again shortly.")
    return {"ok": True}


@app.get("/api/runs/{identifier}/frames/{filename}")
async def frame(identifier: str, filename: str, request: Request):
    run = owned_run(identifier, request)
    if not re.fullmatch(r"[a-zA-Z0-9_-]+\.(jpg|jpeg|png)", filename):
        raise HTTPException(404)
    path = run.folder / "frames" / filename
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path, media_type="image/png" if filename.endswith(".png") else "image/jpeg")


@app.get("/api/runs/{identifier}/events")
async def events(identifier: str, request: Request):
    run = owned_run(identifier, request)
    try:
        after = int(request.headers.get("last-event-id", request.query_params.get("after", "0")))
    except ValueError:
        after = 0

    async def stream():
        nonlocal after
        # Snapshots permit reconnect even after a long screencast evicts old event entries.
        if not run.events or after > run.sequence or after < run.events[0]["id"]-1:
            snapshot_events = [
                ("browser", run.browser), ("frame", run.frame),
                *[("step", s) for s in run.steps.values()], ("result", run.result),
                ("phase", {"phase": run.status, "message": run.message, "elapsed_ms": run.elapsed_ms}),
            ]
            for kind, data in snapshot_events:
                if data:
                    yield f"event: {kind}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
            after = run.sequence
            yield f"id: {after}\n: synchronized\n\n"
        while not await request.is_disconnected():
            batch = [event for event in run.events if event["id"] > after]
            for event in batch:
                after = event["id"]
                yield f"id: {after}\nevent: {event['type']}\ndata: {json.dumps(event['data'], ensure_ascii=False)}\n\n"
            if run.status in TERMINAL and not batch:
                return
            try:
                async with run.changed:
                    await asyncio.wait_for(run.changed.wait(), timeout=10)
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"})


FRONTEND = PROJECT / "frontend/dist"
if (FRONTEND / "assets").exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND / "assets"), name="assets")


@app.get("/{path:path}")
async def frontend(path: str):
    if path.startswith("api/"):
        raise HTTPException(404)
    file = FRONTEND / "index.html"
    if not file.exists():
        return JSONResponse({"detail": "The web page has not been built yet (npm --prefix frontend run build)."}, status_code=503)
    return FileResponse(file)
