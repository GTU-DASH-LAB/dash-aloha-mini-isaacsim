"""Session-layer colliders for stage geometry the robot can see but PhysX cannot.

WHY. Every obstacle sense this repo has goes through PhysX: the collision guard's fan, the
2D lidar, the arc menu's clearance filter and the kinematic contact layer all ask PhysX's
scene query, and the scene query only knows prims with a collider. A nav stage's
environment is somebody else's asset, and its authors decided what collides -- outdoor_small
decided almost nothing does: CollisionAPI on 4807 of 8862 drivable-surface gprims, 11 of
24733 props, 0 of 5590 vehicle gprims, 0 of 200 characters (scene/outdoor_small probe1,
2026-10-05). So the base drove through buildings, every guard/lidar/clearance number
outdoors measured only kerbs, and the camera showed a store interior on a trace the logs
called clear.

WHAT. Before the timeline plays, every Mesh under /World/Environment that is in the region
the robot can reach, in its height band, visible, and part of an asset that carries NO
collider at all is copied -- world-space triangles, clipped to the band -- into a new
invisible mesh under COLLIDER_ROOT in the SESSION layer, with a triangle-mesh collider
(MeshCollisionAPI approximation "none", i.e. exact). Nothing is written to the stage file;
the copies live as long as the process. Copy, not edit: most of these meshes are instance
proxies, and authoring to a proxy raises.

"An asset that carries no collider at all" is the detection, and it is per ASSET, not per
mesh, on purpose. An asset whose author gave it any collider -- a simplified collision
proxy beside its render meshes, a door sill, a deliberately disabled one -- was decided by
somebody, and a copy of its render meshes would duplicate or override that decision. An
asset with none was not decided at all. The asset is the nearest ancestor that is an
instance or carries references/payloads, below the environment's own root; a mesh inline in
the environment file is its own asset with its parent prim. Nothing here knows which stage
is outdoors: indoors the same rule finds whatever the indoor assets left out, and on a
stage where everything collides it authors nothing and costs one traversal.

WHEN. Before `timeline.play()` (run_navigation.NavigationRunner.setup()): PhysX parses the
colliders with the rest of the scene, nothing is ever authored around a robot that is
already standing there (a collider appearing around a body is a separation impulse), and
nothing is ever DELETED -- isaacsim.core's Articulation drops its physics view on any prim
deletion in the stage (scene/outdoor_small geom_hook, 2026-10-05), so these prims stay.

Constants are next to their single user, each with its measurement.
"""

from __future__ import annotations

import math
import time

ENV_ROOT = "/World/Environment"
COLLIDER_ROOT = "/World/SessionColliders"

# The region: the axis-aligned box of every start and goal the session's episodes name,
# grown by this much. The robot cannot be scored outside it and a policy that wanders
# further than this past its farthest start or goal has failed anyway; what matters is that
# every metre it drives near its route is real.
REGION_MARGIN_M = 20.0
# The height band, relative to the episodes' own start z (DynaNav's Nova Carter spawn
# height: indoors ~1 cm above the floor, outdoors 0.14-2.21 m above it). Below: the deepest
# start drop measured (2.21 m, pillars) plus the deepest relief under any recorded trace
# (1.26 m, outdoor_library's lower plaza), rounded up. Above: the robot's top (1.084 m
# above base_link, the column) plus a raised lift (0.6 m) plus margin, 2.5 m -- also low
# enough to leave indoor ceilings (3-4 m) and the upper floors of outdoor buildings alone.
BAND_BELOW_M = 4.0
BAND_ABOVE_M = 2.5
# An asset whose every candidate mesh lies within this of one WALKABLE plane (least-squares
# fit, slope <= kinematic_contact.MAX_WALKABLE_SLOPE_DEG) is a floor covering, not an
# obstacle: road markings and decals on a road that already collides, carpet runners on an
# office floor. A plain world-z extent cannot say this: a road marking on a sloped road spans
# decimetres of z (outdoor_small roadmarks_6/7: 122 meshes, asset boxes 0.42-0.59 m tall) and
# office's SM_FloorCarpetPath slabs are exactly 0.010 m thick, both copied by the first
# version (2026-10-06). The slope condition is what keeps a narrow kerb strip, which a plane
# tilted across its width would otherwise fit (a 12 x 15 cm kerb strip reads 0.060 m, a
# 1 cm slab 0.005, a decal on a 5% slope 0.001; a box of height h reads h/2). 2 cm, i.e. a
# slab under 4 cm thick, is under everything any sense here can see -- the contact layer's
# low box starts kinematic_contact.MAX_STEP_M (4.5 cm) above the floor, the guard and the
# lidar sit at 0.30 m -- so a copy would only add a second surface for the tyre spheres to
# bump over. Per ASSET, so a thin table top whose legs are in the
# same asset still collides. `flat_margin` in the plan's stats reports both sides of the cut.
FLAT_ASSET_M = 0.02
# A mesh larger than this is skipped and recorded rather than copied (a whole-terrain
# mesh with no collider is a stage problem to look at, not something to cook silently).
MAX_POINTS = 400_000


def _xform_rows(m) -> "np.ndarray":
    import numpy as np

    return np.array([[m[r][c] for c in range(4)] for r in range(4)], dtype=np.float64)


def env_episodes(env: str) -> list:
    """Every episode of environment `env` in EVERY table under nav/config/ (start, goal,
    name only). The region must not depend on which table a process loaded: the ladder's
    hospital_vending_machine and the full table's hospital_e01 are one episode, and the
    two runs must drive among the same colliders."""
    from types import SimpleNamespace

    from episode import all_episode_specs, env_name

    return [SimpleNamespace(name=n, start=tuple(s["start"]), goal=tuple(s["goal"]))
            for n, s in sorted(all_episode_specs().items()) if env_name(s["scene"]) == env]


def region_of(episodes, margin_m: float = REGION_MARGIN_M,
              below_m: float = BAND_BELOW_M, above_m: float = BAND_ABOVE_M) -> dict:
    """The box (xmin, ymin, xmax, ymax) and band (zlo, zhi) for a set of episodes."""
    xs = [float(p[0]) for e in episodes for p in (e.start, e.goal)]
    ys = [float(p[1]) for e in episodes for p in (e.start, e.goal)]
    zs = [float(e.start[2]) for e in episodes]
    return {"box": (min(xs) - margin_m, min(ys) - margin_m,
                    max(xs) + margin_m, max(ys) + margin_m),
            "band": (min(zs) - below_m, max(zs) + above_m),
            "margin_m": margin_m, "episodes": len(episodes)}


def _asset_of(prim, env_path, memo: dict):
    """The asset a gprim belongs to: nearest ancestor (or itself) below the environment
    root that is an instance or carries references/payloads; else its parent prim."""
    chain = []
    p = prim
    root = None
    while p and p.GetPath() != env_path:
        key = p.GetPath()
        if key in memo:
            root = memo[key]
            break
        chain.append(key)
        if p.IsInstance() or p.HasAuthoredReferences() or p.HasAuthoredPayloads():
            root = key
            break
        p = p.GetParent()
    if root is None:
        root = prim.GetParent().GetPath()      # inline in the environment file
        chain = chain[:1]                      # only the gprim itself maps to its parent
    for k in chain:
        memo[k] = root
    return root


def plan(stage, region: dict | None = None, *, include_skinned: bool = False) -> dict:
    """Detect. Returns {"candidates": [...], "stats": {...}}; authors nothing.

    A candidate is a visible, non-guide/proxy Mesh whose world box meets the region box and
    the band, with no CollisionAPI of its own, in an asset where no gprim has one.
    """
    import numpy as np
    from pxr import Usd, UsdGeom, UsdPhysics, UsdSkel

    t0 = time.perf_counter()
    env = stage.GetPrimAtPath(ENV_ROOT)
    out = {"candidates": [], "stats": {}}
    if not env:
        out["stats"] = {"error": f"no {ENV_ROOT}"}
        return out
    env_path = env.GetPath()
    bb = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_,
                                                    UsdGeom.Tokens.render],
                           useExtentsHint=True)
    memo: dict = {}
    asset_colliders: dict = {}
    rows = []
    n_gprims = n_coll = 0
    for prim in Usd.PrimRange(env, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Gprim):
            continue
        n_gprims += 1
        asset = _asset_of(prim, env_path, memo)
        has = prim.HasAPI(UsdPhysics.CollisionAPI)
        if has:
            n_coll += 1
            asset_colliders[asset] = asset_colliders.get(asset, 0) + 1
            continue
        asset_colliders.setdefault(asset, 0)
        rows.append((prim, asset))
    stats = {"gprims": n_gprims, "with_collider": n_coll, "assets": len(asset_colliders),
             "assets_without_collider": sum(1 for v in asset_colliders.values() if v == 0)}
    box = region["box"] if region else None
    band = region["band"] if region else None
    skip: dict[str, int] = {}

    def skipped(why: str) -> None:
        skip[why] = skip.get(why, 0) + 1

    for prim, asset in rows:
        if asset_colliders.get(asset, 0) > 0:
            skipped("asset_has_collider")
            continue
        img = UsdGeom.Imageable(prim)
        if img.ComputePurpose() in (UsdGeom.Tokens.guide, UsdGeom.Tokens.proxy):
            skipped("purpose_guide_or_proxy")
            continue
        if img.ComputeVisibility(Usd.TimeCode.Default()) == UsdGeom.Tokens.invisible:
            skipped("invisible")
            continue
        try:
            rng = bb.ComputeWorldBound(prim).ComputeAlignedRange()
        except Exception:  # noqa: BLE001
            skipped("no_bound")
            continue
        if rng.IsEmpty():
            skipped("empty_bound")
            continue
        lo, hi = rng.GetMin(), rng.GetMax()
        if not all(math.isfinite(v) for v in (*lo, *hi)):
            skipped("nonfinite_bound")
            continue
        if box and (hi[0] < box[0] or lo[0] > box[2] or hi[1] < box[1] or lo[1] > box[3]):
            skipped("outside_region")
            continue
        if band and (hi[2] < band[0] or lo[2] > band[1]):
            skipped("outside_band")
            continue
        if not prim.IsA(UsdGeom.Mesh):
            skipped(f"not_mesh:{prim.GetTypeName()}")
            continue
        if prim.HasAPI(UsdSkel.BindingAPI) and not include_skinned:
            # A skinned mesh's points are its BIND pose; what is rendered is the skeleton's
            # pose. Copied blindly, a posed character would collide as a T-pose.
            skipped("skinned")
            continue
        out["candidates"].append({"path": str(prim.GetPath()), "asset": str(asset),
                                  "lo": [float(v) for v in lo], "hi": [float(v) for v in hi]})
    # Floor coverings (FLAT_ASSET_M): each mesh's distance from its own best walkable plane,
    # and per asset the thickest mesh (a mesh whose plane is too steep counts as infinite).
    from kinematic_contact import MAX_WALKABLE_SLOPE_DEG

    xc = UsdGeom.XformCache()
    max_tan = math.tan(math.radians(MAX_WALKABLE_SLOPE_DEG))
    thick: dict[str, float] = {}
    for c in out["candidates"]:
        c["flat_m"] = _plane_thickness(UsdGeom.Mesh(stage.GetPrimAtPath(c["path"])), xc,
                                       max_tan)
        thick[c["asset"]] = max(thick.get(c["asset"], 0.0), c["flat_m"])
    kept = []
    for c in out["candidates"]:
        if thick[c["asset"]] < FLAT_ASSET_M:
            skipped("flat_asset")
        else:
            kept.append(c)
    out["candidates"] = kept
    flat = sorted((v, a) for a, v in thick.items() if v < FLAT_ASSET_M)
    solid = sorted((v, a) for a, v in thick.items() if v >= FLAT_ASSET_M)
    stats["flat_margin"] = {
        "flat_assets": len(flat), "thickest_flat": [[round(v, 4), a] for v, a in flat[-5:]],
        "thinnest_kept": [[round(v, 4) if math.isfinite(v) else "steep", a]
                          for v, a in solid[:5]]}
    stats["skipped"] = skip
    stats["candidates"] = len(out["candidates"])
    stats["plan_s"] = round(time.perf_counter() - t0, 2)
    out["stats"] = stats
    return out


def _plane_thickness(mesh, xc, max_tan: float) -> float:
    """Max distance (m, vertical) of a mesh's world points from their least-squares plane
    z = a x + b y + c; inf when that plane is steeper than walkable or the fit is degenerate
    (a vertical wall, a kerb strip). Subsampled to ~20k points: a bound, not a render."""
    import numpy as np

    pts = mesh.GetPointsAttr().Get()
    if not pts:
        return math.inf
    P = np.asarray(pts, dtype=np.float64)
    if len(P) > 20000:
        P = P[:: len(P) // 20000 + 1]
    M = _xform_rows(xc.GetLocalToWorldTransform(mesh.GetPrim()))
    W = P @ M[:3, :3] + M[3, :3]
    if len(W) < 3:
        return math.inf
    xy = W[:, :2] - W[:, :2].mean(0)
    A = np.column_stack((xy, np.ones(len(W))))
    coef, _, rank, _ = np.linalg.lstsq(A, W[:, 2], rcond=None)
    if rank < 3 or math.hypot(coef[0], coef[1]) > max_tan:
        return math.inf
    return float(np.abs(W[:, 2] - A @ coef).max())


def author(stage, candidates: list[dict], band: tuple[float, float] | None = None) -> dict:
    """Copy each candidate's world-space triangles (faces meeting the band) into an invisible
    triangle-mesh collider under COLLIDER_ROOT, in the session layer. Idempotent per path
    within a process: a candidate already copied is skipped."""
    import numpy as np
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, Vt

    t0 = time.perf_counter()
    xc = UsdGeom.XformCache()
    rep = {"authored": 0, "faces": 0, "points": 0, "skipped": {}, "map": {}}
    root = stage.GetPrimAtPath(COLLIDER_ROOT)
    done = set()
    if root:
        for child in root.GetChildren():
            src = child.GetCustomDataByKey("source")
            if src:
                done.add(str(src))

    def skipped(why: str) -> None:
        rep["skipped"][why] = rep["skipped"].get(why, 0) + 1

    # No Sdf.ChangeBlock: Define() reads composed state, which is not safe inside one.
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        UsdGeom.Scope.Define(stage, COLLIDER_ROOT)
        n = len(done)
        for c in candidates:
            if c["path"] in done:
                skipped("already_authored")
                continue
            prim = stage.GetPrimAtPath(c["path"])
            mesh = UsdGeom.Mesh(prim)
            pts = mesh.GetPointsAttr().Get()
            fvc = mesh.GetFaceVertexCountsAttr().Get()
            fvi = mesh.GetFaceVertexIndicesAttr().Get()
            if not pts or not fvc or not fvi:
                skipped("empty")
                continue
            if len(pts) > MAX_POINTS:
                skipped("too_large")
                continue
            P = np.asarray(pts, dtype=np.float64)
            M = _xform_rows(xc.GetLocalToWorldTransform(prim))
            W = P @ M[:3, :3] + M[3, :3]
            counts = np.asarray(fvc, dtype=np.int64)
            idx = np.asarray(fvi, dtype=np.int64)
            if counts.sum() != idx.size or (idx.size and (idx.min() < 0 or idx.max() >= len(W))):
                skipped("bad_topology")
                continue
            if band is not None and counts.size:
                starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
                z = W[idx, 2]
                fmin = np.minimum.reduceat(z, starts)
                fmax = np.maximum.reduceat(z, starts)
                keep = (fmax >= band[0]) & (fmin <= band[1]) & (counts >= 3)
                if not keep.any():
                    skipped("no_face_in_band")
                    continue
                if not keep.all():
                    vmask = np.repeat(keep, counts)
                    idx = idx[vmask]
                    counts = counts[keep]
            used = np.unique(idx)
            remap = np.full(len(W), -1, dtype=np.int64)
            remap[used] = np.arange(used.size)
            Wc = W[used].astype(np.float32)
            path = f"{COLLIDER_ROOT}/c{n}"
            n += 1
            dst = UsdGeom.Mesh.Define(stage, path)
            dst.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(Wc))
            dst.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(counts.astype(np.int32)))
            dst.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(remap[idx].astype(np.int32)))
            dst.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
            dst.GetDoubleSidedAttr().Set(True)
            lo, hi = Wc.min(0), Wc.max(0)
            dst.GetExtentAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*map(float, lo)),
                                                   Gf.Vec3f(*map(float, hi))]))
            dst.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
            dprim = dst.GetPrim()
            dprim.SetCustomDataByKey("source", c["path"])
            UsdPhysics.CollisionAPI.Apply(dprim)
            UsdPhysics.MeshCollisionAPI.Apply(dprim).CreateApproximationAttr("none")
            rep["map"][path] = c["path"]
            rep["authored"] += 1
            rep["faces"] += int(counts.size)
            rep["points"] += int(used.size)
    rep["author_s"] = round(time.perf_counter() - t0, 2)
    return rep


def add_session_colliders(stage, episodes, **region_kw) -> dict:
    """Detect and author, for the region of `episodes`. Call before timeline.play().

    Returns one record for the run log / EpisodeResult config: region, detection stats,
    what was authored and how long each half took. `map` (collider -> source mesh) is
    returned too, for tools that want to name what a ray hit."""
    region = region_of(episodes, **region_kw)
    p = plan(stage, region)
    a = author(stage, p["candidates"], band=region["band"]) if p["candidates"] else {
        "authored": 0, "faces": 0, "points": 0, "skipped": {}, "map": {}, "author_s": 0.0}
    by_asset: dict[str, int] = {}
    for c in p["candidates"]:
        by_asset[c["asset"]] = by_asset.get(c["asset"], 0) + 1
    return {"region": {"box": [round(v, 2) for v in region["box"]],
                       "band": [round(v, 2) for v in region["band"]],
                       "margin_m": region["margin_m"], "episodes": region["episodes"]},
            "detect": p["stats"], "authored": a["authored"], "faces": a["faces"],
            "points": a["points"], "author_skipped": a["skipped"], "author_s": a["author_s"],
            "assets": len(by_asset),
            "top_assets": sorted(by_asset.items(), key=lambda kv: -kv[1])[:12],
            "map": a["map"]}


def source_of(prim_path: str, report: dict | None) -> str:
    """The environment mesh a session collider stands in for (else the path itself)."""
    if report and prim_path in (report.get("map") or {}):
        return report["map"][prim_path]
    return prim_path
