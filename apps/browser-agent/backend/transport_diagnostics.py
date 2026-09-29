"""Private per-attempt Jev transport timings; never log headers or request bodies."""
import hashlib
import json
import os
import threading
import time
from urllib.parse import urlsplit


def request_hash(content):
    if isinstance(content, dict):
        content = json.dumps(content, separators=(",", ":"))
    if isinstance(content, str):
        content = content.encode()
    return hashlib.sha256(content).hexdigest()


def is_jev(url):
    parsed = urlsplit(str(url))
    return (parsed.hostname, parsed.path) in {
        ("openrouter.ai", "/api/alpha/decisions"),
        ("api.typesafe.ai", "/v1/systemone"),
    }


class TransportDiagnostics:
    def __init__(self, writer):
        self.writer = writer
        self.path = writer.folder / "model_transport.jsonl"
        self.lock = threading.Lock()
        self.sequence = 0

    def save(self, entry):
        try:
            with self.lock:
                fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
                with os.fdopen(fd, "a") as stream:
                    stream.write(json.dumps(entry) + "\n")
        except OSError:
            pass

    def instrument_client(self, client, lane):
        original = client.post

        def post(url, *args, **kwargs):
            if not is_jev(url):
                return original(url, *args, **kwargs)
            started = time.perf_counter()
            with self.lock:
                self.sequence += 1
                identifier = self.sequence
            entry = {"type": "http", "id": identifier, "lane": lane,
                     "request_sha256": request_hash(kwargs.get("content", b"")),
                     "started_elapsed_ms": self.writer.elapsed(), "status": "running"}
            self.save(entry)
            events = []
            extensions = dict(kwargs.get("extensions") or {})
            previous_trace = extensions.get("trace")

            def trace(name, info):
                # `info` may contain authorization headers; retain only event names/times.
                events.append({"event": name, "elapsed_ms": round((time.perf_counter() - started) * 1000, 2)})
                if previous_trace:
                    previous_trace(name, info)

            extensions["trace"] = trace
            kwargs["extensions"] = extensions
            try:
                response = original(url, *args, **kwargs)
                entry.update(status="returned", http_status=response.status_code,
                             http_version=response.http_version)
                return response
            except BaseException as exc:
                entry.update(status="failed", error=type(exc).__name__)
                raise
            finally:
                entry.update(duration_ms=round((time.perf_counter() - started) * 1000, 2),
                             finished_elapsed_ms=self.writer.elapsed(), events=events)
                self.save(entry)

        client.post = post

    def install(self, model):
        self.instrument_client(model.CLIENT, "primary")
        self.instrument_client(model.BACKUP, "backup")
        submit = model.POOL.submit

        def traced_submit(fn, *args, **kwargs):
            if len(args) < 3 or not is_jev(args[0]):
                return submit(fn, *args, **kwargs)
            queued = time.perf_counter()
            elapsed = self.writer.elapsed()
            digest = request_hash(args[2])

            def work():
                self.save({"type": "dispatch", "request_sha256": digest,
                           "queued_elapsed_ms": elapsed, "started_elapsed_ms": self.writer.elapsed(),
                           "queue_ms": round((time.perf_counter() - queued) * 1000, 2),
                           "lane": "backup" if len(args) > 3 and args[3] is model.BACKUP else "primary"})
                return fn(*args, **kwargs)

            return submit(work)

        model.POOL.submit = traced_submit
