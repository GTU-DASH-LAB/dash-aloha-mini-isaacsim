#!/usr/bin/env python3
"""A small HTTP surface for watching the robot and changing its limits while it drives.

    curl localhost:8770/status
    curl -X POST localhost:8770/limits -d '{"cruise_mps": 0.2}'
    curl -X POST localhost:8770/enable  -d '{"enabled": false}'
    curl -X POST localhost:8770/stop

Stdlib `http.server` on purpose. Everything else in this repository that serves HTTP uses
FastAPI, and the reason not to here is that this runs on the ROBOT, which is where the
dependency list has to stay short enough to install on a Jetson over a phone hotspot. It
is a handful of endpoints with no schema evolution ahead of them.

RELATIONSHIP TO robot.yaml. This is not a second place limits live -- a limit in two
places is a limit that drifts, and this repository has already produced two complete,
plausible, wrong result tables that way. Every write here goes through `config.save()`
into the same `robot.yaml`, and the runner picks it up through the same mtime check it
uses for a hand edit. So the file is the state, `set_limit.py` and this server are two
ways of writing it, and `GET /limits` reads back what is actually in force.

BINDING AND WHO CAN REACH IT. Defaults to 127.0.0.1. There is no authentication, and
`POST /enable {"enabled": true}` makes a robot drive, so binding this to 0.0.0.0 hands
that button to the network you are on. `--bind` will do it, and prints a warning, because
"tune the speed from my laptop while it drives" is a real need on a real robot.
"""

from __future__ import annotations

import json
import threading
from dataclasses import fields, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import config as robot_config

MAX_BODY_BYTES = 64 * 1024


class _Handler(BaseHTTPRequestHandler):
    server_version = "qvla-robot/1.0"
    runner = None            # set by serve()
    limits_path: Path = robot_config.DEFAULT_PATH

    # --- plumbing ---------------------------------------------------------------------
    def log_message(self, fmt: str, *args) -> None:
        # Silenced: the control loop's own output is what the operator is watching, and a
        # status poll every second would scroll it off the screen.
        pass

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        if length > MAX_BODY_BYTES:
            raise ValueError(f"body of {length} bytes is too large")
        raw = self.rfile.read(length)
        parsed = json.loads(raw.decode())
        if not isinstance(parsed, dict):
            raise ValueError("body must be a JSON object")
        return parsed

    # --- routes -----------------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802 -- BaseHTTPRequestHandler's spelling
        try:
            if self.path.rstrip("/") in ("/status", ""):
                self._send(200, self._status())
            elif self.path.rstrip("/") == "/limits":
                lim = robot_config.load(self.limits_path)
                self._send(200, {f.name: getattr(lim, f.name) for f in fields(lim)})
            else:
                self._send(404, {"error": f"no route {self.path}",
                                 "routes": ["/status", "/limits", "/enable", "/stop"]})
        except Exception as exc:  # noqa: BLE001 -- a 500 beats killing the thread
            self._send(500, {"error": str(exc)})

    def do_POST(self) -> None:  # noqa: N802
        route = self.path.rstrip("/")
        try:
            body = self._body()
        except (ValueError, json.JSONDecodeError) as exc:
            self._send(400, {"error": f"bad request body: {exc}"})
            return

        try:
            if route == "/limits":
                self._send(200, self._write_limits(body))
            elif route == "/enable":
                if "enabled" not in body:
                    self._send(400, {"error": "send {\"enabled\": true|false}"})
                    return
                self._send(200, self._write_limits({"enabled": bool(body["enabled"])}))
            elif route == "/stop":
                # Two things, and both are wanted. `enabled: false` persists, so the
                # robot stays stopped across a restart of the runner and cannot be
                # un-stopped by a process dying; `runner.stop()` ends the current run.
                # A stop that only lived in memory would be undone by a crash-restart,
                # which is the moment you least want the robot to resume.
                out = self._write_limits({"enabled": False})
                if self.runner is not None:
                    self.runner.stop()
                self._send(200, {"stopped": True, "limits": out["limits"]})
            else:
                self._send(404, {"error": f"no route {self.path}"})
        except Exception as exc:  # noqa: BLE001
            self._send(500, {"error": str(exc)})

    # --- work -------------------------------------------------------------------------
    def _status(self) -> dict:
        if self.runner is None:
            return {"state": "no runner attached"}
        out = self.runner.telemetry.snapshot()
        out["limits_path"] = str(self.limits_path)
        return out

    def _write_limits(self, changes: dict) -> dict:
        known = {f.name for f in fields(robot_config.Limits)}
        unknown = sorted(set(changes) - known)
        if unknown:
            return {"error": f"unknown limit(s): {', '.join(unknown)}",
                    "known": sorted(known)}
        current = robot_config.load(self.limits_path)
        # `validated()` runs here as well as on load, so a value the clamp would reject
        # is reported in the response the operator is looking at rather than only in the
        # runner's stdout, which they may not have in front of them.
        merged = replace(current, **changes).validated()
        robot_config.save(merged, self.limits_path)
        applied = {k: getattr(merged, k) for k in changes}
        # Keyed by what was ASKED for, so the response says 9.9 -> 0.45 rather than
        # reporting the clamped value twice. An operator who sees their own number
        # echoed back has no way to know it was not honoured.
        clamped = {k: changes[k] for k, v in applied.items() if v != changes[k]}
        print(f"[control] {self.client_address[0]} set "
              + ", ".join(f"{k}={v}" for k, v in applied.items())
              + (f" (clamped from {clamped})" if clamped else ""), flush=True)
        return {"ok": True, "applied": applied, "clamped": clamped or None,
                "limits": {f.name: getattr(merged, f.name) for f in fields(merged)}}


def serve(runner=None, host: str = "127.0.0.1", port: int = 8770,
          limits_path: Path | str = robot_config.DEFAULT_PATH) -> ThreadingHTTPServer:
    """Start the control API on a daemon thread and return the server."""
    handler = type("_BoundHandler", (_Handler,),
                   {"runner": runner, "limits_path": Path(limits_path)})
    httpd = ThreadingHTTPServer((host, port), handler)
    if host not in ("127.0.0.1", "localhost", "::1"):
        print(f"[control] listening on {host}:{port} with NO authentication. Anyone who "
              f"can reach this port can start the robot.", flush=True)
    else:
        print(f"[control] http://{host}:{port}/status", flush=True)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd
