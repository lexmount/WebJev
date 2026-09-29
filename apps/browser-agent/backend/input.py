"""Ephemeral, owner-authorized input while the worker is paused for login.

No text or keyboard values are written to disk or added to the agent history.
The API deliberately does not expose arbitrary CDP methods or JavaScript.
"""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import threading

_spec = importlib.util.spec_from_file_location("live_control_cdp", Path(__file__).resolve().parents[1] / "vendor/jev_ultrafast/cdp.py")
_cdp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cdp)


class BrowserInput:
    def __init__(self, folder: Path):
        self.folder = folder
        self.connection = None
        self.target = None
        self.session = None
        self.lock = threading.Lock()

    def send(self, body: dict):
        with self.lock:
            config = json.loads((self.folder / "browser_control.json").read_text())
            if self.connection is None:
                self.connection = _cdp.CDP(config["connect_url"])
            target = config["target_id"]
            if target != self.target:
                self.session = self.connection.call("Target.attachToTarget", targetId=target, flatten=True)["sessionId"]
                self.target = target
            viewport = config.get("viewport") or {"width": 1600, "height": 780}
            x = max(0, min(float(body.get("x") or 0), viewport.get("width", 1600)-1))
            y = max(0, min(float(body.get("y") or 0), viewport.get("height", 780)-1))
            kind = body["type"]
            def call(method, **params):
                return self.connection.call(method, session_id=self.session, **params)
            if kind == "click":
                call("Input.dispatchMouseEvent", type="mouseMoved", x=x, y=y)
                call("Input.dispatchMouseEvent", type="mousePressed", x=x, y=y, button="left", clickCount=1)
                call("Input.dispatchMouseEvent", type="mouseReleased", x=x, y=y, button="left", clickCount=1)
            elif kind == "wheel":
                call("Input.dispatchMouseEvent", type="mouseWheel", x=x, y=y, deltaX=0, deltaY=body.get("delta_y", 0))
            elif kind == "text":
                call("Input.insertText", text=body.get("text", ""))
            elif kind == "key":
                key = body["key"]
                codes = {"Enter": 13, "Tab": 9, "Backspace": 8, "Delete": 46, "Escape": 27, "ArrowLeft": 37, "ArrowUp": 38, "ArrowRight": 39, "ArrowDown": 40, "Home": 36, "End": 35, "a": 65}
                params = {"key": key, "code": "KeyA" if key == "a" else key, "windowsVirtualKeyCode": codes[key], "modifiers": body.get("modifiers", 0)}
                call("Input.dispatchKeyEvent", type="keyDown", **params)
                call("Input.dispatchKeyEvent", type="keyUp", **params)

    def close(self):
        if self.connection:
            self.connection.close()
            self.connection = None
