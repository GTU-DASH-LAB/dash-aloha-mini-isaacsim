"""Thin client for the TIC-VLA policy server.

Deliberately **stdlib only** (urllib, json). This module is imported by the Isaac Sim
6.0.1 process, whose Python is a separate 3.12 interpreter from the policy server's
3.11 venv -- anything exotic here would have to be installed into Isaac Sim's bundled
environment, which is exactly the kind of cross-contamination ~/robotics/README.md
forbids. `requests` is not worth that.
"""

from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from typing import Any


class PolicyServerError(RuntimeError):
    pass


# The keys server_qwen's `camera` accepts (arc_menu.MENU_CAMERA_KEYS). Repeated here, not
# imported, because this module stays stdlib-only for Isaac Sim's interpreter; the server
# rejects any key it does not know with a 422, so a drift between the two copies is loud.
CAMERA_KEYS = (
    "name", "width", "height", "fx", "fy", "cx", "cy", "hfov_deg",
    "cam_h", "cam_x", "cam_y", "pitch_deg", "roll_deg", "yaw_deg",
    "arcs", "min_visible_m", "prompt", "robot",
)
POSE_KEYS = ("cam_h", "cam_x", "cam_y", "pitch_deg", "roll_deg", "yaw_deg")


def camera_from_spec(cam_spec: dict[str, Any], menu_args: dict[str, Any], *,
                     name: str | None = None, arcs: list[float] | None = None,
                     min_visible_m: float | None = None, prompt: bool | None = None,
                     robot: str | None = None) -> dict[str, Any]:
    """The `camera` payload for one frame: fixed optics from a camera spec, LIVE pose.

    `cam_spec` is an asset_spec-style camera entry (nav/results/g1/asset_spec.json's
    `nav_camera` / `head_camera_real`): `resolution_wh` and `intrinsics_px` {fx, fy, cx, cy}.
    `menu_args` is the pose AT FRAME TIME in nav/sim/g1/camera_geometry.py's convention --
    G1Locomotion.camera_pose(cam_spec)["menu_args"] read in the same update the frame was
    grabbed in. All six pose keys are required: a missing one raises here rather than
    letting the server draw a level menu on a pitched camera.
    """
    missing = [k for k in POSE_KEYS if menu_args.get(k) is None]
    if missing:
        raise ValueError(f"menu_args is missing {missing}: send the live camera pose")
    w, h = cam_spec["resolution_wh"]
    intr = cam_spec["intrinsics_px"]
    cam: dict[str, Any] = {
        "name": name if name is not None else str(cam_spec.get("label", "")),
        "width": int(w), "height": int(h),
        "fx": float(intr["fx"]), "fy": float(intr["fy"]),
        "cx": float(intr["cx"]), "cy": float(intr["cy"]),
        **{k: float(menu_args[k]) for k in POSE_KEYS},
    }
    if arcs is not None:
        cam["arcs"] = [float(k) for k in arcs]
    if min_visible_m is not None:
        cam["min_visible_m"] = float(min_visible_m)
    if prompt is not None:
        cam["prompt"] = bool(prompt)
    if robot is not None:
        cam["robot"] = str(robot)
    return cam


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

    def menu_speed(self, cruise_mps: float) -> dict[str, Any]:
        """Retune the cruise speed without reloading the weights.

        The arc menu has no speed channel: the model picks a DIRECTION and the pace comes
        from the server's configuration. So a caller that caps speed only on its own side
        leaves the server planning 3 m arcs at a pace the robot never drives, and
        `BrakingPursuitController` takes `min(v_max, plan_speed(waypoints))` against a
        number describing somebody else's robot. Real hardware is where that bites, since
        `robot.yaml` is edited mid-drive and a restart there costs a model load.

        Only the arc-menu server implements this; `server.py` answers 404 as a
        `PolicyServerError`. Between episodes, not during a scored one -- every decision
        records the speed it used, but `summarize_runs.py` cannot know a run changed
        pace halfway and will report it as one measurement.
        """
        return self._post("/menu_speed", {"cruise_mps": float(cruise_mps)})

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
        image_paths: list[str] | None = None,
        *,
        instruction: str,
        images: list[bytes] | None = None,
        robot_state: list[float] | None = None,
        current_step: int | None = None,
        time_delay: float = 0.0,
        previous_waypoints_text: str = "",
        delayed_image_paths: list[str] | None = None,
        robot_type: str = "wheeled robot",
        recovered: bool = False,
        recovery_kind: str = "",
        stalled_s: float = 0.0,
        wait_fresh: bool = False,
        wait_inflight: bool = False,
        scan_points: list[list[float]] | None = None,
        yaw_delta_rad: float = 0.0,
        pose_world: list[float] | None = None,
        sim_time: float | None = None,
        camera: dict[str, Any] | None = None,
        odometry: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Returns {waypoints, reasoning, num_waypoints, latency_s, kv_cache_available,
        vlm_generation_start_step}.

        Waypoints are body-frame FLU displacements, same convention DynaNav's Nova
        Carter behaviour consumes.

        FRAMES GO ONE OF TWO WAYS, and which one you can use is a fact about your
        deployment, not a preference. `image_paths` names files the SERVER opens on its
        OWN disk: correct and cheaper when the caller and the server share a filesystem,
        which is the simulator's situation. `images` carries the encoded frames
        themselves, for the case that made this parameter necessary -- the robot has the
        camera, the workstation has the GPU, and they share no disk. Pass exactly one.
        Either way the order is oldest-first and it is load-bearing: the policy reads
        position in the list as time.

        `time_delay` and the dx,dy tail of `robot_state` are not decoration: the server
        runs `predict_async`, so the plan comes back built on a KV cache that is roughly
        one VLM generation old. These two fields are how the caller tells the action
        head how old, and how far the robot travelled meanwhile. Leaving them at zero
        does not make the staleness go away -- it just hides it from the model.

        `vlm_generation_start_step` is non-None only on calls that started a new
        background generation; the caller is expected to keep those and reference the
        second-to-last one. See run_navigation.py's `gen_starts`.

        `recovered` says the robot has just reversed. The arc-menu server uses it to tell
        the model something the picture cannot: that the view in front of it is a view it
        has already failed at, seen from a metre further back. `recovery_kind` says which
        failure -- "wedge" (drove into something) or "balk" (stopped with clear floor
        ahead) -- and `stalled_s` is how long the robot has been making no progress, which
        no single frame can show.

        `wait_fresh` asks the server to block until it has a plan built on THESE images,
        instead of returning the cached one. The caller is expected to have stopped the
        robot first -- that is the whole point of it, and returning a fresh plan to a robot
        that kept driving would buy nothing. See run_navigation.py's NAV_PLAN_PERIOD_S.

        `wait_inflight` is the middle setting: return the plan built on the PREVIOUS
        call's images, waiting for it only if it is still decoding, and start the next
        generation on these. The robot keeps driving, but never on thinking more than one
        planning period old -- where plain async lets that age grow to a whole generation
        and therefore with the thinking budget. Send at most one of the two; `wait_fresh`
        wins if both arrive, since it is the strictly stronger guarantee.

        `scan_points` is one revolution of the 2D lidar as body-frame (x_forward, y_left)
        metres. The arc-menu server uses it to drop menu arcs the robot cannot actually
        drive; it is the one thing on this list the camera structurally cannot supply,
        since a 90 degree frame cannot measure the width of a gap it is looking at. Sent
        as points rather than ranges so the frame convention travels with the data.

        `pose_world` (x, y, yaw) and `sim_time` are the robot's own pose and clock at this
        call. With both, server_qwen reframes the plan it serves against THAT plan's own
        generation start instead of the anchor implied by `time_delay` / dx,dy /
        `yaw_delta_rad`, which in async mode is one generation too old on every call that
        picks up a freshly finished plan. The response's `plan_age_s` is the staleness the
        server actually applied.

        `camera` is the camera THIS frame was taken with -- intrinsics, mount and the live
        pitch/roll/yaw at frame time, plus optionally an arc subset (see
        `camera_from_spec`). server_qwen draws the arc menu with it instead of its own level
        camera_nav model, which is what lets one server drive a second body (the G1's head
        D435). None -- the default -- leaves the request byte-identical to what it was
        before the field existed, and the server draws exactly as it always has.

        All of them are ignored by `server.py`, so they are safe to send either way. That is
        not incidental: the runner talks to whichever server is listening, and a field that
        broke the TIC-VLA baseline would make the two policies un-comparable on the same
        ladder. `wait_fresh` is the one to watch there -- TIC-VLA's own loop is genuinely
        asynchronous by design, so sending it True does not make that baseline synchronous
        and a run must not be labelled as though it had.
        """
        if camera is not None:
            unknown = sorted(set(camera) - set(CAMERA_KEYS))
            if unknown:
                raise ValueError(f"camera has unknown keys {unknown}; allowed: {list(CAMERA_KEYS)}")
        if (image_paths is None) == (images is None):
            raise ValueError(
                "pass exactly one of image_paths= (server reads them off its own disk) "
                "or images= (raw encoded frames, for a robot that shares no filesystem "
                "with the server)")

        payload = {
            "image_paths": image_paths or [],
            # Encoded here rather than by the caller so the wire format stays this
            # module's business. `images` is a list of ENCODED frames -- the bytes of
            # a JPEG or PNG, not a raw pixel buffer -- oldest first, same order and
            # same meaning as image_paths.
            "images_b64": ([base64.b64encode(b).decode("ascii") for b in images]
                           if images is not None else None),
            "instruction": instruction,
            "robot_state": robot_state if robot_state is not None else [0.0] * 6,
            "current_step": current_step,
            "time_delay": time_delay,
            "previous_waypoints_text": previous_waypoints_text,
            "delayed_image_paths": delayed_image_paths,
            "robot_type": robot_type,
            "recovered": recovered,
            "recovery_kind": recovery_kind,
            "stalled_s": stalled_s,
            "wait_fresh": wait_fresh,
            "wait_inflight": wait_inflight,
            "scan_points": scan_points,
            # Heading change since the generation behind the plan, paired with the
            # dx,dy tail of robot_state. server_qwen re-serves a CACHED plan and must
            # rotate it as well as translate it into the current body frame; left
            # at 0.0 it translated only, so a turn already under way was asked for
            # again on every re-serve -- about 1.9x the chosen yaw on a repeated arc.
            # server.py (TIC-VLA) ignores the field: its action expert re-runs on
            # the current frame, so nothing it returns is stale in heading.
            "yaw_delta_rad": yaw_delta_rad,
            "pose_world": pose_world,
            "sim_time": sim_time,
        }
        # Only when given, so a caller without a camera sends exactly the bytes it sent
        # before this field existed.
        if camera is not None:
            payload["camera"] = camera
        # Same rule: `odometry` (sim/turn_odometer.py's payload) only when given. server_qwen
        # reads it under QVLA_ODOMETRY=1; server.py's pydantic model ignores it.
        if odometry is not None:
            payload["odometry"] = odometry
        return self._post("/predict", payload)
