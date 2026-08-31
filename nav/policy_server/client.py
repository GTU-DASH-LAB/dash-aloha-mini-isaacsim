"""Thin client for the TIC-VLA policy server.

Deliberately **stdlib only** (urllib, json). This module is imported by the Isaac Sim
6.0.1 process, whose Python is a separate 3.12 interpreter from the policy server's
3.11 venv -- anything exotic here would have to be installed into Isaac Sim's bundled
environment, which is exactly the kind of cross-contamination ~/robotics/README.md
forbids. `requests` is not worth that.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any


class PolicyServerError(RuntimeError):
    pass


class PolicyClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 8765, timeout: float = 120.0):
        self.base = f"http://{host}:{port}"
        self.timeout = timeout

    # ---------------------------------------------------------------- internals
    def _post(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = json.dumps(payload or {}).encode()
        req = urllib.request.Request(
            f"{self.base}{path}", data=data,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            raise PolicyServerError(f"{path} -> HTTP {exc.code}: {exc.read()[:400]!r}") from exc
        except urllib.error.URLError as exc:
            raise PolicyServerError(
                f"cannot reach policy server at {self.base} ({exc.reason}). "
                "Start it with nav/policy_server/launch.sh"
            ) from exc

    def _get(self, path: str) -> dict[str, Any]:
        try:
            with urllib.request.urlopen(f"{self.base}{path}", timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.URLError as exc:
            raise PolicyServerError(
                f"cannot reach policy server at {self.base} ({exc.reason}). "
                "Start it with nav/policy_server/launch.sh"
            ) from exc

    # ------------------------------------------------------------------- public
    def health(self) -> dict[str, Any]:
        return self._get("/health")

    def wait_until_ready(self, timeout_s: float = 900.0, poll_s: float = 3.0) -> dict[str, Any]:
        """Block until the server reports a loaded model.

        Startup is slow on purpose -- InternVL3-1B plus a 1.9 GB checkpoint -- so the
        sim process should call this once rather than racing the first /predict.
        """
        deadline = time.time() + timeout_s
        last = None
        while time.time() < deadline:
            try:
                info = self.health()
                if info.get("ok"):
                    return info
                last = info
            except PolicyServerError as exc:
                last = {"error": str(exc)}
            time.sleep(poll_s)
        raise PolicyServerError(f"policy server not ready after {timeout_s}s; last={last}")

    def reset(self, run: str = "") -> dict[str, Any]:
        """Clear the policy's state between episodes.

        `run` names the episode that is about to start. `server.py` ignores it; the
        arc-menu server uses it to open a per-episode recording directory, and without a
        name it does not record at all -- the step counter restarts every episode, so a
        shared directory has the second one overwriting the first.
        """
        return self._post("/reset", {"run": run} if run else {})

    def replan(self) -> dict[str, Any]:
        """Throw away the cached plan MID-episode and leave everything else alone.

        Deliberately not `reset()`. The arc-menu server rebuilds its recording directory
        on every /reset, so a bare reset in the middle of a run sets `run_dir` to None and
        silently stops recording the menus and decisions -- losing the evidence for the
        very manoeuvre that prompted the call.

        Only the arc-menu server implements this. `server.py` does not, so a 404 comes
        back as `PolicyServerError`; callers that may talk to either should treat that as
        "nothing to clear" rather than as a failure, because for TIC-VLA it is true --
        its action expert re-runs on the current frame every tick and holds no plan to
        throw away.
        """
        return self._post("/replan")

    def predict(
        self,
        image_paths: list[str],
        instruction: str,
        robot_state: list[float] | None = None,
        current_step: int | None = None,
        time_delay: float = 0.0,
        previous_waypoints_text: str = "",
        delayed_image_paths: list[str] | None = None,
        robot_type: str = "wheeled robot",
        recovered: bool = False,
    ) -> dict[str, Any]:
        """Returns {waypoints, reasoning, num_waypoints, latency_s, kv_cache_available,
        vlm_generation_start_step}.

        Waypoints are body-frame FLU displacements, same convention DynaNav's Nova
        Carter behaviour consumes.

        `time_delay` and the dx,dy tail of `robot_state` are not decoration: the server
        runs `predict_async`, so the plan comes back built on a KV cache that is roughly
        one VLM generation old. These two fields are how the caller tells the action
        head how old, and how far the robot travelled meanwhile. Leaving them at zero
        does not make the staleness go away -- it just hides it from the model.

        `vlm_generation_start_step` is non-None only on calls that started a new
        background generation; the caller is expected to keep those and reference the
        second-to-last one. See run_navigation.py's `gen_starts`.

        `recovered` says the robot has just reversed out of a wedge. The arc-menu server
        uses it to tell the model something the picture cannot: that the obstacle filling
        the frame is the one that stopped it, seen from a metre further back. `server.py`
        ignores the field, so it is safe to send either way.
        """
        return self._post(
            "/predict",
            {
                "image_paths": image_paths,
                "instruction": instruction,
                "robot_state": robot_state if robot_state is not None else [0.0] * 6,
                "current_step": current_step,
                "time_delay": time_delay,
                "previous_waypoints_text": previous_waypoints_text,
                "delayed_image_paths": delayed_image_paths,
                "robot_type": robot_type,
                "recovered": recovered,
            },
        )
