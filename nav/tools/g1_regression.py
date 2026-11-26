"""Regression test for the walking G1 (nav/sim/g1/locomotion.py), the way the nav runner will drive it:
G1Base in CALLBACK mode, one kit.update() per 1/60 s. Flat cube ground with NO material (PhysX default;
locomotion's robot material makes the effective friction the trained 0.8/0.6), spawn yaw 1.2 rad so
world != body frame. Every check is an assertion with a numeric verdict in <out>/report.json.

  parity    Isaac Lab parity schedule (stand 2 s / vx 0.5 6 s / vx 0.8 3 s / turn (0.3,0,0.6) 3 s /
            stand 2 s), body-frame COM velocity averaged after 1 s of each segment, against the Isaac
            Lab 2.2.1 GPU+CPU ground truth (scratchpad/g1/policy_notes.md) +- PARITY_TOL. "Stands and
            walks" cannot certify the glue (g1/check: an L/R ankle-roll swap, a shoulder swap and a
            policy at 100 or 25 Hz all walk); these numbers can. Plus an INDEPENDENT observation rebuild
            (own quaternion code on the raw tensor API) against the obs the policy consumed, the PD
            targets PhysX holds against default + 0.5*action, policy ticks per sim second, and
            pre/post physics callback counts.
  hold      zero command for 20 s with hold_on_stop off, then on; and a 0.6 m/s walk into a STOP.
            Graded on the MAXIMUM distance from the latched pose over the hold window (per physics
            sub-step, from the tool's own root-pose log, and G1Base's per-tick figure), not on the
            end point: round 2 quoted an end point (0.005 m) that only said where inside the 5 cm
            band the run ended (g1/check measured 0.051 m on the same code).
  reverse   vx -0.3 for 10 s with hold_on_stop on: the request is clamped to (0, 0, 0) and must HOLD
            (round 2 decided on the raw request and drifted 0.296 m backwards); a reverse request
            with a turn in it must keep the turn.
  rate      the policy-rate check must be able to FAIL: (a) loco.decimation = 2 for 2 s -> policy_hz
            ~100 and policy_hz_ok False (round 2 judged against loco.decimation and said OK), then
            restored -> OK again; (b) the physics scene's timeStepsPerSecond changed to 100 while
            playing -> step_dt 0.01 reaches the PRE_STEP callback -> dt_mismatch_steps > 0 and fault
            physics_dt_mismatch; restored to 200 -> no new mismatches.
  nofloor   reset_to() a start with no floor under it: fault "no_floor" at once, still there after
            reset_stats(), and fault "free_fall" within FREE_FALL_HOLD_S + 0.2 s of falling (round 2:
            fallen() fired after 4.75 s at pelvis z -110.9 m).
  ledge     walk off platforms: 0.135 m (the guard's step-down limit) and 0.20 m at 0.5/0.8/1.0 m/s
            must NOT raise free_fall (vz_min and the longest run below FREE_FALL_VZ_MPS are recorded);
            0.40 and 1.00 m at 0.5 m/s are recorded (free fall expected on the 1.00 m one).
  contact   torso contact counter: walk at 0.5 m/s into (a) a slab hanging at chest height (only the
            torso box can touch it) -> torso_contact_steps MUST be > 0; (b) a full-height wall (the
            feet reach it first) -> reported.
  singleton a second G1Base in callback mode must be refused.
  destroy   create an unrelated Camera, walk, destroy it, keep walking: the G1's physics view must
            survive (it did not before _ScopedArticulation).
  handle    the handle-lost path: drop the physics view by hand; physics_state() must report
            physics_handle_lost (an error, not a fall) and the callbacks must not raise. Runs last.
  stress    (NOT in the default set: --only stress, or --with-stress) the g1/prototype's randomised
            60 s command streams, regenerated bit for bit from the same seeded RNG (box_corners seed 2
            is the stream that fell at 14.25 s without a slew limit), through G1Base as shipped
            (DEFAULT_SLEW on). Pass = no falls. --stress-slew off replays them without the limit.

--gpu runs everything on GPU dynamics + GPU broadphase (readback on, numpy tensor API) instead of CPU
MBP: the parity numbers are then the acceptance test for moving a stage to GPU PhysX.

The parity segments run with the shipping G1Base defaults (slew limit ON, hold OFF -- the Isaac Lab
reference stands without any hold): the slew limit settles within 0.3 s and every segment average
starts 1 s in.

    nav/tools/isaac_slot.sh ~/robotics/isaacsim-6.0.1/python.sh nav/tools/g1_regression.py --out <dir>

report["status"] is "ok" only if every assertion passed. The process exit code is NOT the verdict:
nothing after SimulationApp.close() runs (kit.close() ends the process), so read the report.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
# nav/tools/profile.py shadows the stdlib `profile`, which Kit's warp imports.
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != TOOLS]
REPO = TOOLS.parent.parent
sys.path.insert(0, str(REPO / "nav" / "sim" / "g1"))
sys.path.insert(1, str(REPO / "nav" / "sim"))

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--yaw", type=float, default=1.2)
ap.add_argument("--only", default="", help="comma list of checks (default all but stress)")
ap.add_argument("--gpu", action="store_true", help="GPU dynamics + GPU broadphase instead of CPU MBP")
ap.add_argument("--with-stress", action="store_true", help="also run the stress check")
ap.add_argument("--stress-seeds", type=int, default=6, help="box_corners seeds 0..N-1 (random_box gets N//2)")
ap.add_argument("--stress-seconds", type=float, default=60.0)
ap.add_argument("--stress-slew", choices=["on", "off"], default="on")
args = ap.parse_args()
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
ONLY = set(args.only.split(",")) if args.only else None

# Isaac Lab 2.2.1 ground truth (GPU, CPU) for the parity schedule, body-frame mean after 1 s of each
# segment (policy_notes.md "Ground truth" table). Tolerance per the g1/check request.
PARITY_REF = {"stand": {"vx": (-0.031, -0.030)}, "fwd0.5": {"vx": (0.393, 0.395)}, "fwd0.8": {"vx": (0.670, 0.672)},
              "turn": {"vx": (0.234, 0.238), "wz": (0.623, 0.635)}, "stand2": {"vx": (-0.036, -0.035)}}
PARITY_TOL = 0.01

from isaacsim import SimulationApp  # noqa: E402

T0 = time.time()
kit = SimulationApp({"headless": True, "active_gpu": 0, "physics_gpu": 0, "multi_gpu": False})
report: dict = {"pid": os.getpid(), "args": vars(args), "checks": {}}


def save():
    (OUT / "report.json").write_text(json.dumps(report, indent=1, default=str))


class GpuPoll(threading.Thread):
    """GPU0/GPU1 memory of THIS pid, GPU0 utilisation and the number of compute processes on GPU0,
    once a second: a cost figure means nothing without the card's load next to it."""

    def __init__(self):
        super().__init__(daemon=True)
        self.samples, self.stop_flag = [], False

    def run(self):
        while not self.stop_flag:
            try:
                apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
                                       "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
                cards = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used",
                                        "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
                idx, util = {}, {}
                for line in cards.strip().splitlines():
                    i, u, ut, mu = [c.strip() for c in line.split(",")]
                    idx[u] = int(i)
                    util[int(i)] = (int(ut), int(mu))
                mine, n0 = {}, 0
                for line in apps.strip().splitlines():
                    pid, uuid, mib = [c.strip() for c in line.split(",")]
                    if int(pid) == os.getpid():
                        mine[idx.get(uuid)] = int(mib)
                    if idx.get(uuid) == 0:
                        n0 += 1
                self.samples.append({"t": round(time.time() - T0, 1), "mine_gpu0": mine.get(0, 0), "mine_gpu1": mine.get(1, 0),
                                     "gpu0_util": util[0][0], "gpu0_used": util[0][1], "procs_gpu0": n0,
                                     "load1": round(os.getloadavg()[0], 1)})
            except Exception as e:  # noqa: BLE001
                self.samples.append({"error": repr(e)})
            time.sleep(1.0)

    def window(self, t0: float, t1: float) -> dict:
        s = [x for x in self.samples if "gpu0_util" in x and t0 <= x["t"] <= t1]
        if not s:
            return {"n": 0}
        return {"n": len(s), "gpu0_util_mean": round(sum(x["gpu0_util"] for x in s) / len(s), 1),
                "procs_gpu0_min_max": [min(x["procs_gpu0"] for x in s), max(x["procs_gpu0"] for x in s)],
                "mine_gpu0_max_mib": max(x["mine_gpu0"] for x in s), "mine_gpu1_max_mib": max(x["mine_gpu1"] for x in s),
                "load1_mean": round(sum(x["load1"] for x in s) / len(s), 1)}


poll = GpuPoll()
poll.start()
failures: list[str] = []


def check(name: str, ok: bool, **evidence) -> bool:
    report["checks"][name] = {"pass": bool(ok), **evidence}
    if not ok:
        failures.append(name)
    print(f"[regression] {'PASS' if ok else 'FAIL'} {name}: {json.dumps(evidence, default=str)[:400]}", flush=True)
    save()
    return ok


def want(name: str) -> bool:
    if name == "stress":
        return (ONLY is not None and name in ONLY) or args.with_stress
    return ONLY is None or name in ONLY


rc = 1
try:
    import numpy as np
    import omni.timeline
    import omni.usd
    from pxr import Gf, UsdGeom, UsdLux, UsdPhysics

    import locomotion as L
    from isaacsim.core.simulation_manager import SimulationEvent, SimulationManager

    ctx = omni.usd.get_context()
    ctx.new_stage()
    for _ in range(5):
        kit.update()
    stage = ctx.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    def box(path, center, size):
        c = UsdGeom.Cube.Define(stage, path)
        c.CreateSizeAttr(1.0)
        xf = UsdGeom.Xformable(c)
        xf.AddTranslateOp().Set(Gf.Vec3d(*center))
        xf.AddScaleOp().Set(Gf.Vec3f(*size))
        UsdPhysics.CollisionAPI.Apply(c.GetPrim())
        return path

    box("/World/Ground", (0, 0, -0.5), (400, 400, 1))
    # contact course, far from the parity track: robot spawns 1.2 m in front of each obstacle face
    SLAB_X, WALL_X = 40.0, 70.0
    box("/World/ChestSlab", (SLAB_X + 1.2 + 0.1, 0.0, 1.0), (0.2, 2.0, 0.6))       # z 0.70-1.30, face at x+1.2
    box("/World/Wall", (WALL_X + 1.2 + 0.1, 0.0, 1.0), (0.2, 2.0, 2.0))            # z 0-2.0, face at x+1.2
    # ledges: 6 m x 2 m platforms whose +x edge is at LEDGE_X, one lane per height
    LEDGE_X, LEDGE_HEIGHTS = -100.0, (0.135, 0.20, 0.40, 1.00)
    LEDGE_Y = {h: 100.0 + 6.0 * i for i, h in enumerate(LEDGE_HEIGHTS)}
    for h_ in LEDGE_HEIGHTS:
        box(f"/World/Ledge_{int(round(h_ * 1000))}mm", (LEDGE_X - 3.0, LEDGE_Y[h_], h_ / 2.0), (6.0, 2.0, h_))
    UsdLux.DomeLight.Define(stage, "/World/Dome").CreateIntensityAttr(500.0)

    loco = L.G1Locomotion("/World/G1")
    report["physics_scene_changes"] = L.configure_physics_scene(stage, loco.physics_dt, cpu=not args.gpu)
    report["author"] = loco.author(stage, (0.0, 0.0, L.SPAWN_HEIGHT_M), args.yaw)
    for _ in range(60):
        kit.update()
    stage.Load()

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    kit.update()
    if SimulationManager.get_physics_sim_view() is None:
        SimulationManager.initialize_physics()
    report["init"] = loco.initialize()
    report["sim_device"] = str(SimulationManager.get_physics_sim_device())
    base = L.G1Base(loco, hold_on_stop=False)
    report["slew"] = loco.slew
    pv = loco.art._physics_view
    names = list(loco.art.dof_names)
    CTL = 1.0 / 60.0

    # per-sub-step log (tool's own post-step callback; G1Base's singleton rule is about G1Base only)
    LOG: list = []
    log_on = {"v": False}

    def on_post(step_dt, context=None):
        if not log_on["v"] or not loco.handle_ok():
            return
        rt = np.asarray(loco.art._physics_view.get_root_transforms()).reshape(-1)
        rv = np.asarray(loco.art._physics_view.get_root_velocities()).reshape(-1)
        LOG.append([loco.sim_t(), *rt[:7], *rv[:6]])
    tool_cb = SimulationManager.register_callback(on_post, event=SimulationEvent.PHYSICS_POST_STEP)

    # time spent in G1's post-step bookkeeping (the contact read added this round)
    post_timer = {"n": 0, "s": 0.0}
    _orig_post = loco.post_physics_step

    def timed_post():
        a = time.perf_counter()
        _orig_post()
        post_timer["n"] += 1
        post_timer["s"] += time.perf_counter() - a
    loco.post_physics_step = timed_post

    def reset(x=0.0, y=0.0, yaw=args.yaw, settle=3, z=0.0):
        log_on["v"] = False
        base.reset_to((x, y, z), yaw)
        base.reset_stats()
        loco.set_command(0, 0, 0)
        for _ in range(settle):
            kit.update()

    def run(schedule, stop_on_fall=True, on_update=None):
        """schedule: [(name, seconds, (vx, vy, wz))]. Returns [(name, i0, i1, cmd)] into LOG + fall + wall."""
        log_on["v"] = True
        segs, fell, wall0 = [], None, time.perf_counter()
        t = 0.0
        for name, secs, cmd in schedule:
            i0 = len(LOG)
            for _ in range(int(round(secs / CTL))):
                base.apply(*cmd, CTL)
                kit.update()
                ps = base.physics_state()
                t += CTL
                if on_update:
                    on_update(t, ps)
                if ps["fallen"] and fell is None:
                    fell = {"t": round(t, 3), "segment": name, "why": base.last_fault and base.last_fault.get("why")}
                    if stop_on_fall:
                        break
            segs.append((name, i0, len(LOG), cmd))
            if fell and stop_on_fall:
                break
        log_on["v"] = False
        return segs, fell, time.perf_counter() - wall0

    def seg_velocity(i0, i1, settle=1.0):
        A = np.array(LOG[i0:i1])
        a = int(round(settle / loco.physics_dt))
        A = A[a:]
        if len(A) < 50:
            return None
        q = A[:, 4:8]                                                     # xyzw
        yaw = np.arctan2(2 * (q[:, 3] * q[:, 2] + q[:, 0] * q[:, 1]), 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2))
        vx = np.cos(yaw) * A[:, 8] + np.sin(yaw) * A[:, 9]
        vy = -np.sin(yaw) * A[:, 8] + np.cos(yaw) * A[:, 9]
        return {"vx": round(float(vx.mean()), 4), "vy": round(float(vy.mean()), 4), "wz": round(float(A[:, 13].mean()), 4),
                "pelvis_z": [round(float(A[:, 3].min()), 3), round(float(A[:, 3].max()), 3)]}

    # ================================================================== parity (+ obs rebuild)
    if want("parity"):
        t_a = time.time() - T0

        def quat_xyzw_to_R(q):
            x, y, z, w = (float(c) for c in q)
            v = np.array([x, y, z])
            vx_ = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
            return (w * w - v @ v) * np.eye(3) + 2 * np.outer(v, v) + 2 * w * vx_

        pol_idx = {n: i for i, n in enumerate(loco.policy.joint_names)}
        default_sim = np.array([loco.policy.default_q[pol_idx[n]] for n in names])
        perm = np.array([names.index(n) for n in loco.policy.joint_names])
        captured = {}
        _orig_obs = loco.policy.observation

        def cap_obs(*a_, **kw):
            o = _orig_obs(*a_, **kw)
            captured["obs"] = o.copy()
            return o
        loco.policy.observation = cap_obs
        obs_chk = {"n": 0, "obs_maxabs": 0.0, "target_maxabs": 0.0}
        prev_a = {"a": np.zeros(len(names))}
        _orig_tick = loco.tick

        def checked_tick():
            rt = np.asarray(pv.get_root_transforms()).reshape(-1)
            rv = np.asarray(pv.get_root_velocities()).reshape(-1)
            R = quat_xyzw_to_R(rt[3:7])
            q = np.asarray(pv.get_dof_positions()).reshape(-1)
            dq = np.asarray(pv.get_dof_velocities()).reshape(-1)
            mine = np.concatenate([R.T @ rv[:3], R.T @ rv[3:6], R.T @ np.array([0, 0, -1.0]), loco.cmd,
                                   (q - default_sim)[perm], dq[perm], prev_a["a"][perm]])
            _orig_tick()
            obs_chk["obs_maxabs"] = max(obs_chk["obs_maxabs"], float(np.abs(captured["obs"] - mine).max()))
            a_sim = np.zeros(len(names))
            a_sim[perm] = loco.policy.last_action
            tgt = np.asarray(pv.get_dof_position_targets()).reshape(-1)
            obs_chk["target_maxabs"] = max(obs_chk["target_maxabs"], float(np.abs(tgt - (default_sim + 0.5 * a_sim)).max()))
            prev_a["a"] = a_sim
            obs_chk["n"] += 1
        reset()
        # the settle updates inside reset() already ticked the policy: start the rebuild from ITS last action
        prev_a["a"] = np.zeros(len(names))
        prev_a["a"][perm] = loco.policy.last_action
        loco.tick = checked_tick
        steps0, post0 = loco.sim_steps, loco.post_steps
        IL = [("stand", 2.0, (0.0, 0.0, 0.0)), ("fwd0.5", 6.0, (0.5, 0.0, 0.0)), ("fwd0.8", 3.0, (0.8, 0.0, 0.0)),
              ("turn", 3.0, (0.3, 0.0, 0.6)), ("stand2", 2.0, (0.0, 0.0, 0.0))]
        segs, fell, wall = run(IL)
        loco.tick = _orig_tick
        loco.policy.observation = _orig_obs
        res, ok_all = {}, fell is None
        for name, i0, i1, cmd in segs:
            v = seg_velocity(i0, i1)
            res[name] = {"cmd": cmd, **(v or {})}
            for key, (lo, hi) in PARITY_REF.get(name, {}).items():
                got = None if v is None else v[key]
                ok = got is not None and lo - PARITY_TOL <= got <= hi + PARITY_TOL
                res[name][f"{key}_ref"] = [lo, hi]
                res[name][f"{key}_ok"] = ok
                ok_all &= ok
        sim_s = (loco.sim_steps - steps0) * loco.physics_dt
        st = base.stats()
        check("parity_il_velocities", ok_all, segments=res, fell=fell, tol=PARITY_TOL)
        check("parity_obs_rebuild", obs_chk["obs_maxabs"] < 1e-4 and obs_chk["target_maxabs"] < 1e-5 and obs_chk["n"] > 700,
              **obs_chk)
        check("policy_hz", bool(st["policy_hz_ok"]), policy_hz=st["policy_hz"], expected=st["policy_hz_expected"],
              ticks_per_sim_s_raw=round(obs_chk["n"] / sim_s, 3))
        check("pre_post_callbacks_equal", (loco.sim_steps - steps0) == (loco.post_steps - post0) and
              base.physics_steps == base.post_steps,
              pre=loco.sim_steps - steps0, post=loco.post_steps - post0, base_pre=base.physics_steps, base_post=base.post_steps)
        # stats()["pre_post_steps"] after a reset_stats(): both counts must be since-reset (g1/check found
        # [since-reset pre, LIFETIME post] -> [1600, 4000] after a 2nd reset)
        check("stats_pre_post_since_reset", st["pre_post_steps"][0] == st["pre_post_steps"][1] > 0,
              pre_post_steps=st["pre_post_steps"])
        report["cost_parity_flat_no_camera"] = {"sim_s": round(sim_s, 2), "wall_s": round(wall, 2),
                                                "wall_per_sim_s": round(wall / sim_s, 3),
                                                "gpu_load": poll.window(t_a, time.time() - T0)}
        report["post_step_cost_ms"] = round(1000 * post_timer["s"] / max(post_timer["n"], 1), 4)
        save()

    # ================================================================== hold-on-stop
    if want("hold"):
        out = {}
        cmd_log: list = []

        def log_cmd(t, ps):
            cmd_log.append((t, *[float(c) for c in loco.cmd_target]))
        def band(A, ref_xy):
            """Distance of every logged sub-step from ref_xy: max, fraction outside HOLD_ENTER_M, and the
            displacement over each consecutive 2 s window (the 'creep' a short stop shows)."""
            dd = np.hypot(A[:, 1] - ref_xy[0], A[:, 2] - ref_xy[1])
            n2 = int(round(2.0 / loco.physics_dt))
            creep = [float(np.hypot(A[i + n2 - 1, 1] - A[i, 1], A[i + n2 - 1, 2] - A[i, 2]))
                     for i in range(0, len(A) - n2 + 1, n2)]
            return {"max_from_latch_m": round(float(dd.max()), 4), "end_from_latch_m": round(float(dd[-1]), 4),
                    "frac_outside_enter_band": round(float((dd > L.HOLD_ENTER_M).mean()), 3),
                    "creep_per_2s_window_m": [round(c, 3) for c in creep]}
        for label, hold in (("raw_zero_command", False), ("hold_on_stop", True)):
            reset(0.0, 20.0)
            base.hold_on_stop = hold
            x0, y0, _, yaw0 = loco.pose()
            cmd_log.clear()
            segs, fell, _ = run([("stop", 20.0, (0.0, 0.0, 0.0))], on_update=log_cmd)
            x1, y1, _, yaw1 = loco.pose()
            A = np.array(LOG[segs[0][1]:segs[0][2]])
            C = np.array(cmd_log)
            last5 = C[C[:, 0] > C[-1, 0] - 5.0]
            st_ = base.stats()
            out[label] = {"drift_m": round(math.hypot(x1 - x0, y1 - y0), 3),
                          "yaw_drift_deg": round(math.degrees(math.atan2(math.sin(yaw1 - yaw0), math.cos(yaw1 - yaw0))), 2),
                          "fell": fell, "hold_steps": base.hold_steps, "hold_active_ticks": base.hold_active_ticks,
                          **band(A, (x0, y0)),
                          "g1base_hold_max_from_latch_m": st_["hold_max_from_latch_m"], "hold_windows": st_["hold_windows"],
                          "nonzero_command_frac_last5s": round(float((np.abs(last5[:, 1:]).max(axis=1) > 1e-9).mean()), 3),
                          "command_absmax_last5s": [round(float(v), 4) for v in np.abs(last5[:, 1:]).max(axis=0)]}
        # walk at 0.6 then STOP with hold: how far past the stop point does it settle, and does it stay
        reset(0.0, 30.0)
        base.hold_on_stop = True
        stop_pt = {}

        def mark(t, ps):
            if "xy" not in stop_pt and t >= 4.0 - 1e-9:
                x_, y_, _, _ = loco.pose()
                stop_pt["xy"] = (x_, y_)
        cmd_log.clear()

        def mark_and_log(t, ps):
            mark(t, ps)
            log_cmd(t, ps)
        segs, fell, _ = run([("walk", 4.0, (0.6, 0.0, 0.0)), ("stop", 20.0, (0.0, 0.0, 0.0))], on_update=mark_and_log)
        A = np.array(LOG[segs[1][1]:segs[1][2]])
        sx, sy = stop_pt.get("xy", (A[0, 1], A[0, 2]))
        d = np.hypot(A[:, 1] - sx, A[:, 2] - sy)
        C = np.array(cmd_log)
        last5 = C[C[:, 0] > C[-1, 0] - 5.0]
        # back inside the ENTER band AFTER the braking overshoot (the stop starts at the latch, d ~ 0, so
        # "first sample inside the band" would be sample 0 and grade the overshoot itself)
        i_peak = int(np.argmax(d))
        back = np.nonzero(d[i_peak:] <= L.HOLD_ENTER_M)[0]
        i_back = i_peak + int(back[0]) if len(back) else None
        after = band(A[i_back:], (sx, sy)) if i_back is not None else None
        st_ = base.stats()
        out["walk_then_stop_hold"] = {"overshoot_max_m": round(float(d.max()), 3), "final_from_stop_point_m": round(float(d[-1]), 3),
                                      "last5s_motion_m": round(float(np.hypot(A[-1, 1] - A[-1000, 1], A[-1, 2] - A[-1000, 2])), 3),
                                      "back_inside_enter_band_after_s": (None if i_back is None else
                                                                         round(i_back * loco.physics_dt, 2)),
                                      "after_back_in_band": after,
                                      "creep_per_2s_window_from_stop_m": band(A, (sx, sy))["creep_per_2s_window_m"],
                                      "g1base_hold_max_from_latch_m": st_["hold_max_from_latch_m"],
                                      "fell": fell, "hold_active_ticks": base.hold_active_ticks,
                                      "nonzero_command_frac_last5s": round(float((np.abs(last5[:, 1:]).max(axis=1) > 1e-9).mean()), 3),
                                      "command_absmax_last5s": [round(float(v), 4) for v in np.abs(last5[:, 1:]).max(axis=0)]}
        base.hold_on_stop = False
        # graded on the BAND (max distance from the latch), not on an end point (module docstring)
        ok = (out["hold_on_stop"]["max_from_latch_m"] < 0.08 and out["hold_on_stop"]["fell"] is None
              and out["walk_then_stop_hold"]["fell"] is None
              and after is not None and after["max_from_latch_m"] < 0.08)
        # the round-1 hold never went quiet (~0.03 m/s commands forever, g1/check); now it must be
        # exactly zero most of the time once settled
        quiet = (out["hold_on_stop"]["nonzero_command_frac_last5s"] <= 0.5
                 and out["walk_then_stop_hold"]["nonzero_command_frac_last5s"] <= 0.5)
        out["quiet_ok"] = quiet
        ok = ok and quiet
        check("hold_on_stop", ok, **out)

    # ================================================================== reverse request -> hold
    if want("reverse"):
        out = {}
        for label, hold in (("hold_off", False), ("hold_on", True)):
            reset(0.0, 40.0)
            base.hold_on_stop = hold
            x0, y0, _, yaw0 = loco.pose()
            cmd_seen = []
            segs, fell, _ = run([("reverse", 10.0, (-0.3, 0.0, 0.0))],
                                on_update=lambda t, ps: cmd_seen.append([float(c) for c in loco.cmd_target]))
            x1, y1, _, _ = loco.pose()
            A = np.array(LOG[segs[0][1]:segs[0][2]])
            dd = np.hypot(A[:, 1] - x0, A[:, 2] - y0)
            fwd = math.cos(yaw0) * (x1 - x0) + math.sin(yaw0) * (y1 - y0)
            st_ = base.stats()
            C = np.array(cmd_seen)
            out[label] = {"drift_m": round(math.hypot(x1 - x0, y1 - y0), 3), "fwd_disp_m": round(fwd, 3),
                          "max_from_start_m": round(float(dd.max()), 4), "fell": fell,
                          "reverse_requests": st_["reverse_requests"], "clamped_commands": st_["clamped_commands"],
                          "reverse_requests_held": st_["reverse_requests_held"], "hold_steps": st_["hold_steps"],
                          "hold_max_from_latch_m": st_["hold_max_from_latch_m"],
                          "policy_cmd_vx_min_max": [round(float(C[:, 0].min()), 4), round(float(C[:, 0].max()), 4)]}
        # reverse + turn keeps the turn (clamped (0, 0, 0.5)), no hold
        reset(0.0, 50.0)
        base.hold_on_stop = True
        yaw0 = loco.pose()[3]
        segs, fell, _ = run([("reverse_turn", 3.0, (-0.3, 0.0, 0.5))])
        yaw1 = loco.pose()[3]
        st_ = base.stats()
        out["reverse_plus_turn"] = {"yaw_change_deg": round(math.degrees(math.atan2(math.sin(yaw1 - yaw0), math.cos(yaw1 - yaw0))), 1),
                                    "policy_cmd_target": [round(float(c), 4) for c in loco.cmd_target],
                                    "hold_steps": st_["hold_steps"], "fell": fell}
        base.hold_on_stop = False
        on = out["hold_on"]
        ok = (on["hold_steps"] > 0 and on["reverse_requests_held"] == on["reverse_requests"] == 600
              and on["max_from_start_m"] < 0.08 and on["fell"] is None and on["policy_cmd_vx_min_max"][0] >= 0.0
              and out["reverse_plus_turn"]["hold_steps"] == 0 and out["reverse_plus_turn"]["yaw_change_deg"] > 30
              and out["reverse_plus_turn"]["fell"] is None)
        check("reverse_request_holds", ok, **out)

    # ================================================================== torso contact
    if want("contact"):
        out = {}
        for label, x in (("chest_slab", SLAB_X), ("full_wall", WALL_X)):
            reset(x, 0.0, yaw=0.0)
            segs, fell, _ = run([("stand", 1.0, (0, 0, 0)), ("walk", 6.0, (0.5, 0.0, 0.0))], stop_on_fall=True)
            st = base.stats()
            px, py, pz, _ = loco.pose()
            out[label] = {"torso_contact_steps": st["torso_contact_steps"], "events": st["torso_contact_events"],
                          "first_t": st["torso_contact_first_t"], "max_n": st["torso_contact_max_n"],
                          "read_errors": st["torso_contact_read_errors"], "fell": fell,
                          "pelvis_x_end_minus_face": round(px - (x + 1.2), 3)}
        check("torso_contact_counts_chest_slab", out["chest_slab"]["torso_contact_steps"] > 0 and
              out["chest_slab"]["read_errors"] == 0, **out)

    # ================================================================== singleton
    if want("singleton"):
        try:
            b2 = L.G1Base(loco)
            b2.close()
            check("singleton_refuses_second_base", False, note="second G1Base was accepted")
        except RuntimeError as e:
            check("singleton_refuses_second_base", True, error=str(e)[:200])

    # ================================================================== unrelated camera destroy
    if want("destroy"):
        from isaacsim.sensors.camera import Camera

        cam = UsdGeom.Camera.Define(stage, "/World/UnrelatedCam")
        UsdGeom.Xformable(cam).AddTranslateOp().Set(Gf.Vec3d(0, -5, 2))
        reset(0.0, -10.0)
        c = Camera(prim_path="/World/UnrelatedCam", resolution=(640, 360))
        c.initialize()
        segs, fell1, _ = run([("walk_a", 3.0, (0.5, 0.0, 0.0))])
        p_mid = loco.pose()
        valid_before = loco.handle_ok()
        c.destroy()
        kit.update()
        valid_after = loco.handle_ok()
        segs, fell2, _ = run([("walk_b", 3.0, (0.5, 0.0, 0.0))])
        p_end = loco.pose() if loco.handle_ok() else p_mid
        moved = math.hypot(p_end[0] - p_mid[0], p_end[1] - p_mid[1])
        st = base.stats()
        check("unrelated_camera_destroy_keeps_g1", valid_before and valid_after and fell2 is None and moved > 0.6 and
              st["skipped_ticks"] == 0, valid_before=valid_before, valid_after=valid_after,
              moved_after_destroy_m=round(moved, 3), fell=fell2, skipped_ticks=st["skipped_ticks"],
              deletions_ignored=getattr(loco.art, "_g1_deletions_ignored", 0),
              deletions_forwarded=getattr(loco.art, "_g1_deletions_forwarded", 0))

    # ================================================================== no floor under the start
    if want("nofloor"):
        base.hold_on_stop = True
        log_on["v"] = False
        base.reset_to((300.0, 300.0, 0.0), 0.0)         # the ground is +-200 m: nothing below
        f_reset = dict(base.first_fault or {})
        spawn = dict(base.spawn)
        base.reset_stats()                                # the runner resets stats AFTER reset_to
        f_after = dict(base.first_fault or {})
        rows, t, ff_t, fall_t = [], 0.0, None, None
        for _ in range(int(round(2.0 / CTL))):
            base.apply(0.0, 0.0, 0.0, CTL)
            kit.update()
            t += CTL
            ps = base.physics_state()
            if ps["free_fall"] and ff_t is None:
                ff_t = round(t, 3)
            if ps["fallen"] and fall_t is None:
                fall_t = round(t, 3)
            if len(rows) < 200:
                rows.append([round(t, 3), round(ps["z"], 3), round(ps["vz_mps"], 3)])
        st_ = base.stats()
        out = {"spawn": spawn, "first_fault_at_reset": f_reset, "first_fault_after_reset_stats": f_after,
               "free_fall_t_s": ff_t, "fallen_t_s": fall_t, "terminal": ps["terminal"],
               "fault_kinds": [f["kind"] for f in st_["fault_log"]], "pelvis_z_at_free_fall": None,
               "trace_t_z_vz_first": rows[:30]}
        if ff_t is not None:
            out["pelvis_z_at_free_fall"] = next((r[1] for r in rows if r[0] >= ff_t), None)
        check("no_floor_start_refused_and_free_fall_detected",
              spawn.get("ground_z") is None and f_reset.get("kind") == "no_floor" and f_after.get("kind") == "no_floor"
              and ff_t is not None and ff_t <= L.FREE_FALL_HOLD_S + 0.35 and ps["terminal"], **out)
        base.hold_on_stop = False

    # ================================================================== ledges: free fall true/false positives
    if want("ledge"):
        runs_ = []
        plan = [(0.135, 0.5), (0.135, 0.8), (0.135, 1.0), (0.20, 0.5), (0.20, 0.8), (0.20, 1.0), (0.40, 0.5), (1.00, 0.5)]
        for h_, v_ in plan:
            # start 2.0 m before the edge, on the platform, facing +x
            reset(LEDGE_X - 2.0, LEDGE_Y[h_], yaw=0.0, z=h_)
            spawn = dict(base.spawn)
            t_run0 = base.sim_time()          # fault times are lifetime sim time; rows below are run-relative
            segs, fell, _ = run([("stand", 1.0, (0.0, 0.0, 0.0)), ("walk", 2.0 / max(v_ * 0.8, 0.1) + 3.0, (v_, 0.0, 0.0))],
                                stop_on_fall=True)
            A = np.array(LOG[segs[-1][1]:segs[-1][2]]) if segs else np.zeros((0, 14))
            st_ = base.stats()
            cross = np.nonzero(A[:, 1] > LEDGE_X)[0] if len(A) else []
            t_cross = None if len(cross) == 0 else round(float(A[cross[0], 0] - A[0, 0]) + 1.0, 3)
            vz_sub = float(A[:, 10].min()) if len(A) else None          # root linear vz per SUB-step
            below = (A[:, 10] < L.FREE_FALL_VZ_MPS) if len(A) else np.zeros(0, bool)
            longest = 0
            run_ = 0
            for b in below:
                run_ = run_ + 1 if b else 0
                longest = max(longest, run_)
            r_ = {"height_m": h_, "cmd_vx": v_, "spawn_ground_z": spawn.get("ground_z"), "fell": fell,
                  "free_fall": st_["free_fall"],
                  "free_fall_t": None if st_["free_fall_t"] is None else round(st_["free_fall_t"] - t_run0, 3),
                  "t_pelvis_crossed_edge": t_cross,
                  "vz_min_per_update_mps": st_["vz_min_mps"], "vz_min_per_substep_mps": None if vz_sub is None else round(vz_sub, 3),
                  "longest_substep_run_below_bound_s": round(longest * loco.physics_dt, 3),
                  "free_fall_longest_run_s_g1base": st_["free_fall_longest_run_s"],
                  "fault_kinds": [f["kind"] for f in st_["fault_log"]], "tilt_max": st_["tilt_max_deg"],
                  "end_pelvis_x_minus_edge": round(loco.pose()[0] - LEDGE_X, 3)}
            runs_.append(r_)
            print(f"[regression] ledge {h_} @ {v_}: {r_}", flush=True)
        negatives = [r for r in runs_ if r["height_m"] <= 0.20]
        ok = all(not r["free_fall"] for r in negatives) and all(r["t_pelvis_crossed_edge"] is not None for r in runs_)
        check("ledge_free_fall_no_false_positive_on_walkable_drops", ok,
              runs=runs_, note="0.40 / 1.00 m rows are recorded, not graded (a G1 may or may not fall there)")

    # ================================================================== policy-rate check can fail
    if want("rate"):
        out = {}
        # (a) decimation 2: the policy ticks at 100 Hz
        reset(0.0, 60.0)
        run([("stand", 1.0, (0.0, 0.0, 0.0))])
        base.reset_stats()
        loco.decimation = 2
        run([("stand_dec2", 2.0, (0.0, 0.0, 0.0))])
        s_bad = base.stats()
        loco.decimation = base.spec_decimation
        base.reset_stats()
        run([("stand_dec4", 2.0, (0.0, 0.0, 0.0))])
        s_good = base.stats()
        out["decimation_2"] = {k: s_bad[k] for k in ("policy_hz", "policy_hz_ok", "decimation_matches_spec", "dt_mismatch_steps")}
        out["decimation_restored"] = {k: s_good[k] for k in ("policy_hz", "policy_hz_ok", "decimation_matches_spec")}
        ok_a = (s_bad["policy_hz_ok"] is False and abs(s_bad["policy_hz"] - 100.0) < 1.0
                and s_good["policy_hz_ok"] is True and abs(s_good["policy_hz"] - 50.0) < 0.5)
        # (b) the stage's physics rate changed while playing (what an un-configured nav stage would do)
        from pxr import PhysxSchema, Usd

        scenes = L.physics_scene_prims(stage)
        attr_vals = {}
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            for p_ in scenes:
                PhysxSchema.PhysxSceneAPI(p_).GetTimeStepsPerSecondAttr().Set(100)
                attr_vals[str(p_.GetPath())] = 100
        base.reset_stats()
        run([("stand_100hz", 0.5, (0.0, 0.0, 0.0))])
        s_100 = base.stats()
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            for p_ in scenes:
                PhysxSchema.PhysxSceneAPI(p_).GetTimeStepsPerSecondAttr().Set(int(round(1.0 / base.spec_dt)))
        kit.update()
        kit.update()
        base.reset_stats()
        run([("stand_200hz", 1.5, (0.0, 0.0, 0.0))])
        s_200 = base.stats()
        out["timesteps_100"] = {k: s_100[k] for k in ("step_dt_min_max", "dt_mismatch_steps", "dt_mismatch_first", "policy_hz",
                                                      "policy_hz_ok", "first_fault", "sim_s_since_reset")}
        out["timesteps_restored_200"] = {k: s_200[k] for k in ("step_dt_min_max", "dt_mismatch_steps", "policy_hz", "policy_hz_ok",
                                                               "first_fault", "fell")}
        live_applied = s_100["step_dt_min_max"] is not None and s_100["step_dt_min_max"][1] > base.spec_dt + 1e-6
        out["live_timestep_change_reached_physx"] = bool(live_applied)
        ok_b = (not live_applied) or (s_100["dt_mismatch_steps"] > 0
                                      and (s_100["first_fault"] or {}).get("kind") == "physics_dt_mismatch")
        ok_b = ok_b and s_200["dt_mismatch_steps"] == 0
        check("policy_rate_check_can_fail", ok_a and ok_b, **out)

    # ================================================================== stress (opt-in)
    if want("stress"):
        sys.path.insert(1, str(REPO / "nav" / "sim"))
        from controllers import make_controller

        def arc_plan(kappa, v, n=30, dt=0.1):
            s_ = np.arange(1, n + 1) * v * dt
            if abs(kappa) < 1e-6:
                return np.stack([s_, np.zeros_like(s_)], 1)
            return np.stack([np.sin(kappa * s_) / kappa, (1 - np.cos(kappa * s_)) / kappa], 1)
        kappas = (-0.60, -0.35, -0.15, 0.0, 0.15, 0.35, 0.60)
        GENS = ("random_box", "box_corners", "pursuit_arcs")

        def stream(gen, seed, seconds):
            """Bit-for-bit the generator of nav/tools/try_g1_runner_style.py --mode stress."""
            rng = np.random.default_rng(1000 * (1 + GENS.index(gen)) + seed)
            seq, tt = [], 0.0
            ctrl = make_controller("braking", 1.0, 1.0)
            ctrl.reset()
            while tt < seconds + 2.0:
                hold = float(rng.uniform(0.1, 1.5)) if gen != "pursuit_arcs" else 0.5
                if gen == "random_box":
                    cmd = (float(rng.uniform(0, 1)), float(rng.uniform(-0.5, 0.5)), float(rng.uniform(-1, 1)))
                elif gen == "box_corners":
                    cmd = (float(rng.choice([0.0, 1.0])), float(rng.choice([-0.5, 0.0, 0.5])), float(rng.choice([-1.0, 0.0, 1.0])))
                else:
                    v = float(rng.uniform(0.3, 1.0))
                    c = ctrl(arc_plan(float(rng.choice(kappas)), v))
                    cmd = (c.vx, c.vy, c.omega)
                seq.append((tt, tt + hold, cmd))
                tt += hold
            return seq
        saved_slew = loco.slew
        loco.slew = dict(L.DEFAULT_SLEW) if args.stress_slew == "on" else None
        runs = []
        plan = [("box_corners", s_) for s_ in range(args.stress_seeds)] + \
               [("random_box", s_) for s_ in range(max(1, args.stress_seeds // 2))]
        for gen, seed in plan:
            seq = stream(gen, seed, args.stress_seconds)
            i_ = len(runs)                     # 90 m grid away from every other course (ground is +-200 m)
            reset((-150.0, -60.0, 60.0, 150.0)[i_ % 4], (-160.0, -90.0, 110.0, 170.0)[(i_ // 4) % 4])
            run([("stand", 1.5, (0.0, 0.0, 0.0))])
            idx = {"i": 0}
            tilt = {"max": 0.0}
            t_run = {"t": 0.0}
            fell = None
            wall0 = time.perf_counter()
            n = int(round(args.stress_seconds / CTL))
            for k in range(n):
                tt = k * CTL
                while idx["i"] < len(seq) - 1 and tt >= seq[idx["i"]][1]:
                    idx["i"] += 1
                base.apply(*seq[idx["i"]][2], CTL)
                kit.update()
                ps = base.physics_state()
                tilt["max"] = max(tilt["max"], ps["tilt_deg"])
                if ps["fallen"]:
                    fell = {"t": round(tt + CTL, 2), "why": (base.last_fault or {}).get("why")}
                    break
            st = base.stats()
            r = {"gen": gen, "seed": seed, "fell": fell, "tilt_max": round(tilt["max"], 2),
                 "slew_limited_ticks": st["slew_limited_ticks"], "n_cmd_changes": len(seq),
                 "wall_per_sim_s": round((time.perf_counter() - wall0) / ((k + 1) * CTL), 3)}
            runs.append(r)
            print(f"[regression] stress {gen} seed {seed}: {r}", flush=True)
            report["stress_runs"] = runs
            save()
        loco.slew = saved_slew
        check("stress_no_falls", all(r["fell"] is None for r in runs), slew=args.stress_slew,
              falls=[(r["gen"], r["seed"], r["fell"]) for r in runs if r["fell"]], runs=len(runs),
              tilt_max=max(r["tilt_max"] for r in runs))

    # ================================================================== handle-lost path (destructive: last)
    if want("handle"):
        reset(0.0, -20.0)
        run([("walk", 1.0, (0.5, 0.0, 0.0))])
        del loco.art._physics_view                       # what the upstream bug / a timeline stop does
        errs = []
        try:
            for _ in range(30):
                base.apply(0.5, 0.0, 0.0, CTL)
                kit.update()
                ps = base.physics_state()
        except Exception as e:  # noqa: BLE001
            errs.append(repr(e))
        st = base.stats()
        check("handle_lost_reported_as_error", not errs and ps.get("physics_handle_lost") is True and not ps.get("fallen")
              and (st["first_fault"] or {}).get("kind") == "physics_handle_lost" and st["skipped_ticks"] > 0,
              errors=errs, state=ps, first_fault=st["first_fault"], skipped_ticks=st["skipped_ticks"],
              handle_lost_at=st["physics_handle_lost_at"])

    SimulationManager.deregister_callback(tool_cb)
    base.close()
    report["gpu_whole_run"] = poll.window(0, time.time() - T0)
    report["failures"] = failures
    report["status"] = "ok" if not failures else f"FAILED: {failures}"
    rc = 0 if not failures else 1
except Exception:  # noqa: BLE001
    import traceback

    report["status"] = "error"
    report["error"] = traceback.format_exc()
    print(report["error"], flush=True)
    rc = 1
finally:
    poll.stop_flag = True
    report["wall_total_s"] = round(time.time() - T0, 1)
    save()
    print(f"[regression] status={report.get('status')}", flush=True)
    try:
        omni.timeline.get_timeline_interface().stop()
    except Exception:  # noqa: BLE001
        pass
    kit.close()
    os._exit(rc)
