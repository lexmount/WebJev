"""A tiny pipelined CDP client for a remote browser.

Browser Harness talks to a local Chrome one command at a time. A cloud browser is a network
round trip away, so this client can send several commands back to back and wait once.
Commands on one session still execute in the order they were sent.
"""

import itertools
import json
import queue
import threading
from concurrent.futures import Future

from websockets.sync.client import connect


class CDP:
    def __init__(self, ws_url, keep_events=()):
        self.ws = connect(ws_url, max_size=None, open_timeout=30)
        self.keep_events = set(keep_events)
        self.events = queue.SimpleQueue()
        self.handlers = {}  # CDP event name -> callable(message); runs on the reader thread, must be quick
        self.calls = 0
        self.bytes_in = 0
        self._ids = itertools.count(1)
        self._pending = {}
        self._lock = threading.Lock()
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def _read(self):
        try:
            for raw in self.ws:
                self.bytes_in += len(raw)
                message = json.loads(raw)
                if "id" in message:
                    entry = self._pending.pop(message["id"], None)
                    if not entry:
                        continue
                    method, future = entry
                    if "error" in message:
                        future.set_exception(RuntimeError(f"{method}: {message['error'].get('message')}"))
                    else:
                        future.set_result(message.get("result", {}))
                elif message.get("method") in self.handlers:
                    try:
                        self.handlers[message["method"]](message)
                    except Exception:  # noqa: BLE001 - a bookkeeping error must not kill the socket reader
                        pass
                elif message.get("method") in self.keep_events:
                    self.events.put(message)
        except Exception as error:  # noqa: BLE001 - every waiter must learn that the socket died.
            failure = RuntimeError(f"CDP connection lost: {error}")
        else:
            failure = RuntimeError("CDP connection closed")
        for _method, future in list(self._pending.values()):
            if not future.done():
                future.set_exception(failure)
        self._pending.clear()

    def send(self, method, session_id=None, **params):
        """Queue one command without waiting. Returns a Future for its result."""
        message_id = next(self._ids)
        future = Future()
        message = {"id": message_id, "method": method, "params": params}
        if session_id:
            message["sessionId"] = session_id
        with self._lock:
            self._pending[message_id] = (method, future)
            self.calls += 1
            self.ws.send(json.dumps(message))
        return future

    def call(self, method, session_id=None, timeout=30, **params):
        return self.send(method, session_id, **params).result(timeout)

    def close(self):
        try:
            self.ws.close()
        except Exception:  # noqa: BLE001
            pass
