"""PhysX scene settings every Kit entry point that drives AlohaMini should apply.

One function today, kept in its own module so that the runner, the validators and any
replay tool raise the same capacities to the same values: a capacity that is right in one
entry point and default in another makes the two disagree about which contacts exist,
and that shows up as a navigation difference, not as an error.

Call it on the OPENED stage, BEFORE `timeline.play()`: PhysX sizes its GPU buffers when
it parses the scene at play, and reads these attributes only then.
"""

from __future__ import annotations

# PxGpuDynamicsMemoryConfig capacities to raise, as {PhysxSceneAPI attribute: floor}.
#
# gpuFoundLostAggregatePairsCapacity is the only one PhysX has ever warned about here, and
# it warns in every outdoor run: "needs to increase ... foundLostAggregatePairsCapacity to
# N, otherwise, the simulation will miss interactions", N = 1615-4385 against the schema
# default of 1024, at play and at every reset teleport (the articulation is one aggregate
# of 17 links, and a teleport across the outdoor stage's thousands of trimesh colliders
# finds and loses that many pairs in one step). Measured over every run log in
# nav/results/logs: 0 warnings on all 13 indoor runs, 2-6848 per outdoor run, worst 4210;
# the instrumented replay saw 4385. 16384 is 3.7x the worst ever seen; an entry is a small
# pair record, so even at this size the buffer is well under a megabyte of GPU memory.
#
# Not raised: gpuTotalAggregatePairsCapacity (also 1024 by default) was never named in a
# warning. Raising the found/lost buffer could in principle let more pairs through to the
# total -- if PhysX ever names it, add it here with its own measured number.
GPU_CAPACITY_FLOORS: dict[str, int] = {
    "gpuFoundLostAggregatePairsCapacity": 16384,
}

# How deep raise_gpu_capacities looks for UsdPhysics.Scene prims (1 = children of the
# pseudo-root). See the comment in the function.
SCENE_SEARCH_DEPTH = 3


def raise_gpu_capacities(stage, floors: dict[str, int] | None = None) -> list[str]:
    """Raise each PhysX GPU capacity in `floors` to at least its floor, on every physics
    scene in the stage. Never lowers one. Returns a line per change, for the run log.

    Authored into the SESSION layer: the change lasts as long as this process and never
    dirties the stage file, so a tool that later saves the stage cannot bake a runtime
    setting into nav_<env>.usda behind the pipeline's back.
    """
    from pxr import PhysxSchema, Usd, UsdPhysics

    floors = GPU_CAPACITY_FLOORS if floors is None else floors
    # Physics scenes sit near the top of a stage: ours at /World/PhysicsScene (depth 2),
    # and an environment that brings its own sits at /World/<Env>/PhysicsScene (depth 3).
    # Three levels covers both without traversing an outdoor environment's hundreds of
    # thousands of prims -- level 3 is only the direct children of /World's children.
    # Every scene found is raised, not just the first: which one PhysX simulates the robot
    # in is PhysX's choice, and a raised scene it does not use costs nothing.
    scenes = []
    level = list(stage.GetPseudoRoot().GetChildren())
    for _depth in range(SCENE_SEARCH_DEPTH):
        nxt = []
        for prim in level:
            if prim.IsA(UsdPhysics.Scene):
                scenes.append(prim)
            nxt.extend(prim.GetChildren())
        level = nxt

    changes: list[str] = []
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for prim in scenes:
            api = PhysxSchema.PhysxSceneAPI.Apply(prim)
            for name, floor in floors.items():
                suffix = name[0].upper() + name[1:]
                attr = getattr(api, f"Get{suffix}Attr")()
                current = attr.Get() if attr and attr.IsValid() else None
                if current is not None and int(current) >= floor:
                    continue
                getattr(api, f"Create{suffix}Attr")().Set(int(floor))
                changes.append(f"{prim.GetPath()}.physxScene:{name} {current} -> {floor}")
    if not scenes:
        changes.append(f"no UsdPhysics.Scene found in the top {SCENE_SEARCH_DEPTH} levels; "
                       "nothing raised")
    return changes
