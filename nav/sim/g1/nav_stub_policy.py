"""A stand-in for the arc-menu policy server, for testing the G1 runner without a model.

`nav/sim/run_navigation_g1.py --policy stub` talks to this instead of server_qwen on :8766. It
answers the same calls the runner makes (health, wait_until_ready, reset, replan, predict), with
SCRIPTED plans, so the walking G1, its controller, guard, recovery and cameras can be exercised
end to end while the 27B is busy with somebody else's ladder -- and so a failure in that chain
cannot be blamed on the model.

What it does per /predict, mirroring what server_qwen does with the `camera` field:

  * picks the plan of the script segment active at `sim_time` -- a curvature from the arc menu
    (plan_arc, the same "drive the 3 m arc, then carry on straight" shape as
    arc_menu.plan_from_kappa), a STOP (every waypoint at the origin, which is what the menu's
    STOP is), or a WORLD-anchored heading (latched at the segment's first call; the plan is
    re-expressed in the body frame on every call, the way server_qwen re-serves a cached plan).
    A heading more than 45 deg off the nose is what makes the G1 controller pivot in place;
  * draws the menu onto the newest frame with the LIVE camera pose the runner sent
    (camera_geometry.project_ground_live -- the pose of the frame it is drawn on, never module
    constants), the chosen arc highlighted, and writes it as menu_<step>.jpg into the run
    directory next to the runner's chase_<step>.jpg, plus one decisions.jsonl row;
  * returns waypoints at 0.1 s spacing over a 10 s horizon (100 points), like server_qwen at the
    medium think level, so plan_speed() reads the scripted speed back exactly.

The script is a list of segments {"until_s": t, "kind": "arc"|"stop"|"heading", ...}, ended by
sim time; built-ins in SCRIPTS, or a JSON file with that list.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import camera_geometry as CG  # noqa: E402

HORIZON_S = 10.0
DT = 0.1
N_WAYPOINTS = int(round(HORIZON_S / DT))
ARC_LEN_M = 3.0          # the drawn arc, arc_menu.DEFAULT_LENGTH_M

# Built-in scripts. "exercise" walks, turns both ways on the menu's +-0.35 arcs, STOPS (the
# hold), turns to a heading 135 deg to the left (> 45 deg: the controller pivots in place), then
# walks straight until something stops it (the guard; then the turn-in-place recovery and a
# replan). Times are sim seconds from the episode start.
SCRIPTS: dict[str, list[dict]] = {
    "exercise": [
        {"until_s": 6.0, "kind": "arc", "kappa": 0.0},
        {"until_s": 10.0, "kind": "arc", "kappa": 0.35},
        {"until_s": 14.0, "kind": "arc", "kappa": -0.35},
        {"until_s": 18.0, "kind": "stop"},
        {"until_s": 28.0, "kind": "heading", "rel_deg": 135.0},
        {"until_s": 1e9, "kind": "arc", "kappa": 0.0},
    ],
    # Straight on, forever: the guard and the recovery are the only things that turn it.
    "straight": [{"until_s": 1e9, "kind": "arc", "kappa": 0.0}],
}

_COLOURS = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
            (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212),
            (0, 128, 128)]


def plan_arc(kappa: float, speed: float, n: int = N_WAYPOINTS, dt: float = DT,
             arc_len_m: float = ARC_LEN_M) -> np.ndarray:
    """arc_menu.plan_from_kappa: the 3 m arc, then straight along the heading it ends on."""
    s = np.arange(1, n + 1) * dt * speed
    on_arc = np.minimum(s, arc_len_m)
    beyond = np.maximum(s - arc_len_m, 0.0)
    theta = kappa * on_arc
    if abs(kappa) < 1e-9:
        x, y = on_arc, np.zeros_like(on_arc)
    else:
        x, y = np.sin(theta) / kappa, (1.0 - np.cos(theta)) / kappa
    return np.stack([x + beyond * np.cos(theta), y + beyond * np.sin(theta)], axis=-1)


def plan_heading(rel_rad: float, speed: float, n: int = N_WAYPOINTS, dt: float = DT) -> np.ndarray:
    """A straight line `rel_rad` off the nose (+ = left), body frame."""
    s = np.arange(1, n + 1) * dt * speed
    return np.stack([s * math.cos(rel_rad), s * math.sin(rel_rad)], axis=-1)


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


class StubArcPolicy:
    """Duck-types nav/policy_server/client.PolicyClient for NavigationRunner.run_episode()."""

    base = "http://stub:0"

    def __init__(self, script: list[dict] | str = "exercise", run_root: Path | str | None = None,
                 camera_provider=None, speed_mps: float = 0.45, draw: bool = True,
                 menu_width: int = 1280):
        if isinstance(script, str):
            p = Path(script)
            self.script_name = script if script in SCRIPTS else p.name
            self.script = SCRIPTS[script] if script in SCRIPTS else json.loads(p.read_text())
        else:
            self.script_name = "custom"
            self.script = list(script)
        self.run_root = Path(run_root) if run_root else None
        self.camera_provider = camera_provider
        self.speed = float(speed_mps)
        self.draw = draw
        self.menu_width = int(menu_width)
        self.run_dir: Path | None = None
        self._latched: dict[int, float] = {}   # segment index -> world heading (rad)
        self.replans: list[float] = []
        self.calls = 0
        self.camera_missing = 0
        self.draw_errors = 0
        self._last_t = 0.0

    # ------------------------------------------------------------------ PolicyClient surface
    def health(self) -> dict:
        return {"ok": True, "model": "stub-arcs", "format": "stub", "mode": "stub",
                "horizon_s": HORIZON_S, "menu_speed_mps": self.speed, "script": self.script_name}

    def wait_until_ready(self, *a, **k) -> dict:
        return self.health()

    def reset(self, run: str = "") -> dict:
        self._latched.clear()
        self.replans.clear()
        self.run_dir = None
        if run and self.run_root is not None:
            self.run_dir = self.run_root / f"{datetime.now():%Y%m%d-%H%M%S}_{run}"
            self.run_dir.mkdir(parents=True, exist_ok=True)
        return {"ok": True, "run_dir": str(self.run_dir) if self.run_dir else None}

    def replan(self) -> dict:
        self.replans.append(round(self._last_t, 3))
        return {"ok": True}

    def menu_speed(self, cruise_mps: float) -> dict:
        self.speed = float(cruise_mps)
        return {"ok": True, "menu_speed_mps": self.speed}

    # ------------------------------------------------------------------ the decision
    def _segment(self, t: float) -> tuple[int, dict]:
        for i, seg in enumerate(self.script):
            if t < float(seg.get("until_s", 1e9)):
                return i, seg
        return len(self.script) - 1, self.script[-1]

    def predict(self, image_paths=None, *, instruction: str = "", sim_time: float | None = None,
                pose_world=None, current_step: int | None = None, camera: dict | None = None,
                **_kw) -> dict:
        t = float(sim_time or 0.0)
        self._last_t = t
        self.calls += 1
        i, seg = self._segment(t)
        kind = seg.get("kind", "arc")
        speed = float(seg.get("speed", self.speed))
        yaw = float(pose_world[2]) if pose_world else 0.0
        rel = None
        kappa = None
        if kind == "stop":
            plan = np.zeros((N_WAYPOINTS, 2))
        elif kind == "heading":
            if i not in self._latched:
                self._latched[i] = yaw + math.radians(float(seg.get("rel_deg", 0.0)))
            rel = _wrap(self._latched[i] - yaw)
            plan = plan_heading(rel, speed)
        else:
            kappa = float(seg.get("kappa", 0.0))
            plan = plan_arc(kappa, speed)
        if camera is None and self.camera_provider is not None:
            camera = self.camera_provider(sim_time)
        if camera is None:
            self.camera_missing += 1
        menu = None
        if self.draw and self.run_dir is not None and image_paths:
            try:
                menu = self._draw(image_paths[-1], current_step, t, seg, kappa, rel, camera)
            except Exception as exc:  # noqa: BLE001  (a drawing bug must not end the episode)
                self.draw_errors += 1
                print(f"[stub] menu draw failed at t={t:.2f}: {exc!r}", flush=True)
        if self.run_dir is not None:
            row = {"step": current_step, "t": round(t, 3), "menu": menu, "segment": i, "kind": kind,
                   "kappa": kappa, "stop": kind == "stop",
                   "rel_deg": None if rel is None else round(math.degrees(rel), 2), "speed": speed,
                   "frame": Path(image_paths[-1]).name if image_paths else None,
                   "camera": None if camera is None else {k: camera.get(k) for k in (
                       "label", "cam_h", "cam_x", "cam_y", "pitch_deg", "roll_deg", "yaw_deg", "frame_sim_t")},
                   "instruction": instruction}
            with open(self.run_dir / "decisions.jsonl", "a") as fh:
                fh.write(json.dumps(row) + "\n")
        reason = f"stub[{self.script_name}] segment {i}: {kind}" + (
            f" kappa={kappa:+.2f}" if kappa is not None else "") + (
            f" heading {math.degrees(rel):+.1f} deg off the nose" if rel is not None else "")
        return {"waypoints": plan.tolist(), "reasoning": reason, "num_waypoints": int(len(plan)),
                "latency_s": 0.0, "kv_cache_available": False,
                "vlm_generation_start_step": current_step, "pivot_rad": 0.0, "plan_age_s": 0.0}

    # ------------------------------------------------------------------ the drawn menu
    def _draw(self, frame_path: str, step, t: float, seg: dict, kappa, rel, camera) -> str:
        from PIL import Image, ImageDraw, ImageFont

        im = Image.open(frame_path).convert("RGB")
        w, h = im.size
        d = ImageDraw.Draw(im)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
        except OSError:
            font = ImageFont.load_default()
        optics = {}
        if camera:
            optics = {"w": int(camera.get("w", w)), "h": int(camera.get("h", h)),
                      "fov_deg": float(camera.get("hfov_deg", CG.HFOV_DEG))}

        def poly(points):
            if not camera:
                return []
            out = []
            for x, y in points:
                p = CG.project_ground_live(x, y, camera, **optics)
                if p is not None and -200 <= p[0] <= w + 200 and -200 <= p[1] <= h + 200:
                    out.append(p)
            return out

        s = np.linspace(0.05, ARC_LEN_M, 60)
        curv = list((camera or {}).get("curvatures") or [])
        for j, k in enumerate(curv):
            pts = [CG.arc_xy(k, float(si)) for si in s]
            px = poly(pts)
            if len(px) < 2:
                continue
            chosen = kappa is not None and abs(k - kappa) < 1e-6
            col = _COLOURS[j % len(_COLOURS)]
            d.line(px, fill=(20, 20, 20), width=(26 if chosen else 15), joint="curve")
            d.line(px, fill=(255, 255, 255) if chosen else col, width=(18 if chosen else 9), joint="curve")
            ux, uy = px[-1]
            d.ellipse([ux - 26, uy - 26, ux + 26, uy + 26], fill=(20, 20, 20), outline=col, width=4)
            d.text((ux, uy), str(j + 1), fill=col, font=font, anchor="mm")
        if rel is not None:
            px = poly([(float(si) * math.cos(rel), float(si) * math.sin(rel)) for si in s])
            if len(px) >= 2:
                d.line(px, fill=(20, 20, 20), width=26, joint="curve")
                d.line(px, fill=(255, 255, 255), width=18, joint="curve")
        label = (f"STOP" if seg.get("kind") == "stop" else
                 f"heading {math.degrees(rel):+.0f} deg" + (" (off-screen)" if abs(math.degrees(rel)) > 34.5 else "")
                 if rel is not None else f"arc kappa {kappa:+.2f}")
        hdr = f"STUB t={t:5.1f}s  {label}"
        if camera:
            hdr += (f"   cam h={camera['cam_h']:.3f} pitch={camera['pitch_deg']:.1f} "
                    f"roll={camera['roll_deg']:+.1f} yaw={camera['yaw_deg']:+.1f}")
        else:
            hdr += "   NO CAMERA POSE"
        d.rectangle([0, 0, w, 60], fill=(0, 0, 0))
        d.text((16, 30), hdr, fill=(255, 255, 255), font=font, anchor="lm")
        if self.menu_width and self.menu_width < w:
            im = im.resize((self.menu_width, int(round(h * self.menu_width / w))), Image.LANCZOS)
        name = f"menu_{(step if step is not None else 0):08d}.jpg"
        im.save(self.run_dir / name, quality=85)
        return name
