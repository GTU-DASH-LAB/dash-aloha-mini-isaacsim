"""Prompt UI, served from inside the Isaac Sim process.

Runs uvicorn on a worker thread. It has to be a thread rather than a second process
because the UI needs live access to the running stage -- the camera frames and the
episode status come straight out of the NavigationRunner object. But Kit itself must
be driven from the MAIN thread, so this side never touches the simulation directly:
it only pushes jobs onto the runner's queue and reads its status snapshot. The main
loop in run_navigation.py picks jobs up.

Two things about uvicorn-in-a-thread that will bite otherwise:
  * signal handlers only install on the main thread, so they are disabled here;
  * uvicorn.run() would create and own an event loop, which fights Kit's. A
    manually configured Server started inside the thread keeps its loop local.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "nav" / "sim"))

from episode import env_name, load_episodes  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"

# A 1x1 grey JPEG, so the <img> has something valid to show before the first render
# instead of a broken-image icon.
_PLACEHOLDER = bytes.fromhex(
    "ffd8ffe000104a46494600010100000100010000ffdb004300ffffffffffffffffffffffffffff"
    "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
    "ffffffffffffffffffffffffffffffffffc00011080001000101011100ffc4001f000001050101"
    "0101010100000000000000000102030405060708090a0bffc400b5100002010303020403050504"
    "0400000177ffda000c03010002110311003f00bf800000000000ffd9"
)


class RunRequest(BaseModel):
    instruction: str
    controller: str | None = None
    # Which benchmark episode to run: sets the start pose and the goal the run is
    # scored against. Only episodes in the loaded environment are accepted; None keeps
    # whichever is current, which is what a hand-typed prompt wants.
    episode: str | None = None


class ChaseRequest(BaseModel):
    enabled: bool


class ResetRequest(BaseModel):
    episode: str | None = None


# Both request models MUST stay at module level. This file uses
# `from __future__ import annotations`, so every annotation is a string that FastAPI
# resolves with get_type_hints() against the function's *module* globals -- a model
# defined inside build_app() is invisible there. It does not raise: FastAPI silently
# falls back to treating the parameter as a query string, and the endpoint answers
# every POST with {"loc": ["query", "req"], "msg": "Field required"}. Cost one restart
# of a 3-minute Isaac Sim boot to find.


def build_app(runner) -> FastAPI:
    app = FastAPI(title="AlohaMini navigation")
    episodes = load_episodes()

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (STATIC / "index.html").read_text()

    @app.get("/api/episodes")
    def api_episodes() -> JSONResponse:
        """The whole DynaNav ladder, marked by whether this session can run it.

        `runnable` means the episode lives in the loaded ENVIRONMENT, so selecting it
        is a teleport to its start plus a new goal -- no rebuild, no restart. That is
        the payoff from building one stage per environment instead of per episode:
        six hospital episodes are six clicks in one session.

        The rest are still listed, with the command that would load them. Typing
        another environment's prompt into this one is also legitimate and
        informative, but it is not that episode -- it is this environment with a
        borrowed sentence, and the distance readout would be scored against the
        wrong goal. `runnable: false` is what the UI uses to keep those apart.
        """
        env = runner.environment
        return JSONResponse({
            "loaded": runner.episode.name,
            "environment": env,
            "episodes": [
                {
                    "name": name,
                    "instruction": ep.instruction,
                    "environment": env_name(ep.scene),
                    "runnable": env_name(ep.scene) == env,
                    "current": name == runner.episode.name,
                    "goal_distance_m": round(ep.straight_line_distance_m, 2),
                    "timeout_s": ep.timeout_s,
                }
                for name, ep in episodes.items()
            ],
        })

    @app.get("/api/status")
    def api_status() -> JSONResponse:
        return JSONResponse(runner.snapshot())

    @app.post("/api/run")
    def api_run(req: RunRequest) -> JSONResponse:
        instruction = req.instruction.strip()
        if not instruction:
            return JSONResponse({"ok": False, "error": "empty instruction"}, status_code=400)
        # Rejected here rather than on the sim thread, so the browser gets a real
        # error instead of a run that starts and then quietly does the wrong thing.
        if req.episode:
            reason = runner.check_episode(req.episode)
            if reason:
                return JSONResponse({"ok": False, "error": reason}, status_code=400)
        if not runner.submit_job(instruction, req.controller, req.episode):
            return JSONResponse(
                {"ok": False, "error": "a run is already in progress"}, status_code=409
            )
        return JSONResponse(
            {"ok": True, "instruction": instruction, "episode": req.episode}
        )

    @app.post("/api/stop")
    def api_stop() -> JSONResponse:
        runner.request_stop()
        return JSONResponse({"ok": True})

    @app.post("/api/reset")
    def api_reset(req: ResetRequest | None = None) -> JSONResponse:
        """Put the robot on a start line -- this episode's, or another one's.

        Returns immediately -- the teleport itself has to happen on the main thread
        (touching the articulation from here races PhysX), so this only raises a flag
        the sim loop picks up within a frame or two.

        Taking an episode here is what makes browsing an environment work: pick one,
        hit Reset, and both camera panels show that episode's opening view without
        committing to a run.
        """
        episode = req.episode if req else None
        if episode:
            reason = runner.check_episode(episode)
            if reason:
                return JSONResponse({"ok": False, "error": reason}, status_code=400)
        runner.request_reset(episode)
        return JSONResponse({"ok": True, "episode": episode})

    @app.post("/api/chase")
    def api_chase(req: ChaseRequest) -> JSONResponse:
        """Toggle the third-person view.

        Off by default and created lazily on the first enable: an Isaac Sim Camera
        allocates a render product that Kit renders every frame regardless of whether
        anyone reads it, so a headless run should not pay for a view nobody watches.
        """
        available = runner.set_chase_enabled(req.enabled)
        return JSONResponse({"ok": True, "enabled": req.enabled, "available": available})

    def _jpeg(data: bytes | None) -> Response:
        # no-store, or the browser serves the first frame forever.
        return Response(
            content=data or _PLACEHOLDER,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
        )

    @app.get("/frame.jpg")
    def frame() -> Response:
        return _jpeg(runner.latest_jpeg)

    @app.get("/chase.jpg")
    def chase_frame() -> Response:
        return _jpeg(runner.chase_jpeg)

    return app


def serve_ui(runner, port: int = 8080, host: str = "127.0.0.1") -> threading.Thread:
    """Start the UI on a daemon thread and return it."""
    import uvicorn

    config = uvicorn.Config(
        build_app(runner), host=host, port=port, log_level="warning",
    )
    server = uvicorn.Server(config)
    # Signal handlers can only be installed from the main thread; leaving this on
    # raises ValueError the moment the server starts here.
    server.install_signal_handlers = lambda: None

    thread = threading.Thread(target=server.run, daemon=True, name="nav-ui")
    thread.start()
    return thread
