"""Offline (no Kit, no GPU) checks that nav/sim/people.py reproduces DynaNav's own run.

The numbers people.py hard-codes were read off DynaNav's OWN Isaac Sim 5.0 run of the hospital
(`probe_dynanav_people_5x.py`, copy in nav/results/people/dn5_dynanav_hospital_5x.json) and off
the local DynaNav checkout. This re-derives each of them from those raw sources instead of from
the design notes, so a typo in a table or a drifted generator fails here, not in a 3-minute
Kit run.

    python3 nav/tools/check_people_offline.py

Exit status 0 = every check passed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "nav" / "sim"))
import people  # noqa: E402

DN5 = REPO / "nav" / "results" / "people" / "dn5_dynanav_hospital_5x.json"
ok = True


def check(name: str, cond: bool, detail: str = "") -> None:
    global ok
    ok &= bool(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


dn5 = json.loads(DN5.read_text())

# 1. command lists: DynaNav's own generator, called our way, == the list DynaNav's run wrote
cmds = people.generate_character_commands("hospital", 10)
check("hospital 10-person command list == the one DynaNav's 5.0 run wrote", cmds == dn5["commands"],
      f"{len(cmds)} lines")
big = people.generate_character_commands("hospital", 40)
check("a smaller crowd is a prefix of a larger one (commands)",
      all(big[i * 10:(i + 1) * 10] == cmds[i * 10:(i + 1) * 10] for i in range(10)) and len(big) == 400)

# 2. spawn table == DynaNav's own spawn points (USD read at play, 3 decimals)
table = people.DYNANAV_SPAWN_TABLES["hospital"]
names = people.character_names(10)
want = [tuple(round(v, 3) for v in dn5["usd_at_play"][n][:2]) for n in names]
got = [(t[0], t[1]) for t in table]
check("hospital spawn table == dn5 usd_at_play (xy)", all(abs(a[0] - b[0]) < 1.5e-3 and abs(a[1] - b[1]) < 1.5e-3
                                                      for a, b in zip(got, want)), f"{len(table)} points")

# 3. asset order: the 10 assets DynaNav's run referenced, and the numpy shuffle for all 21
ref = [dn5["character_assets"][n][0].split("/Characters//")[1].split("/")[0] for n in names]
check("asset order, first 10 == the assets dn5 referenced", people.DYNANAV_ASSET_ORDER_666[:10] == ref)
import numpy as np  # noqa: E402

folders = sorted(people.CHARACTER_FOLDERS)
shuffled = [str(f) for f in np.random.RandomState(666).choice(folders, size=len(folders), replace=False)]
check("asset order, all 21 == np.random.seed(666) choice(replace=False) over the sorted folders",
      shuffled == people.DYNANAV_ASSET_ORDER_666, f"numpy {np.__version__}")
check("21 folders, each once", len(set(people.DYNANAV_ASSET_ORDER_666)) == 21 and
      set(people.DYNANAV_ASSET_ORDER_666) == set(people.CHARACTER_FOLDERS))

# 4. navmesh parameters == what DynaNav's 5.0 bake measured (hospital)
p = people.NAVMESH_PARAMS["hospital"]
meas = {"agentMinHeight": dn5["navmesh_get_agent_height"], "agentMinRadius": dn5["navmesh_get_agent_radius"],
        "agentMaxStepHeight": dn5["navmesh_get_agent_max_step_height"],
        "agentMaxFloorSlope": dn5["navmesh_get_agent_max_slope"]}
for k, v in meas.items():
    scale = 1.0 if k == "agentMaxFloorSlope" else 100.0
    check(f"hospital navmesh {k} == dn5 measured", abs(p[k] - v * scale) < 1e-3, f"{p[k]} vs {v * scale:.4f}")

# 5. no physics on DynaNav's people (the premise of the ghost filter)
check("DynaNav's characters carry 0 Collision / 0 RigidBody APIs",
      dn5["characters_physics_api_counts"] == {"collision": 0, "rigid": 0})

# 6. outdoor: 200 baked people, every asset name resolves to one of the 21 folders
base = json.loads((people.ASSET_DIR / "outdoor_small_baked.json").read_text())["characters"]
check("outdoor baked table has 200 people", len(base) == 200)
bad = sorted({c["asset"] for c in base
              if c["asset"].split("/")[0] not in people.CHARACTER_FOLDERS
              or c["asset"].split("/")[1] != people.CHARACTER_FOLDERS[c["asset"].split("/")[0]]})
check("outdoor baked assets are all known folder/file pairs (as on the 6.0 CDN)", not bad, repr(bad[:3]))

# 7. episodes_full.yaml counts
want_n = {"hospital_vending_machine": 0, "hospital_vending_machine2": 30, "hospital_down_hallway": 10,
          "hospital_down_hallway2": 40, "hospital_past_wheelchairs": 10, "hospital_forward_staircase": 10,
          "hospital_exit_room": 20, "office_nearest_elevator": 20, "office_hallway_turn2": 10,
          "office_hallway_turn": 30, "office_passing_hallway": 30, "outdoor_umbrellas": 200,
          "outdoor_pillars": 200, "outdoor_library": 200, "outdoor_upsway": 100, "outdoor_upsway_far": 200,
          "outdoor_ramp_fountain": 100, "warehouse": 0, "warehouse_aisle6": 0}
got_n = {k: people.episode_num_people(k) for k in want_n}
check("episodes_full.yaml num_people for all 19 of our episodes", got_n == want_n,
      "" if got_n == want_n else str({k: (got_n[k], want_n[k]) for k in want_n if got_n[k] != want_n[k]}))

# 8. names
check("character_names", people.character_names(3) == ["Character", "Character_01", "Character_02"])
sys.exit(0 if ok else 1)
