#!/usr/bin/env python3
"""Turn a pinned profile into shell exports, and check a live server against it.

    eval "$(nav/tools/profile.py env)"        # export the whole configuration
    nav/tools/profile.py env --sim            # ...including the sim-side variables
    nav/tools/profile.py check                # read /health back and diff it
    nav/tools/profile.py show                 # what this profile scored, and why

WHY THIS EXISTS.
Every setting in `nav/config/profiles/*.yaml` is an environment variable, so every one of
them can be silently wrong in somebody's shell. That is not hypothetical here: a ladder
labelled `very_high` was once run against a server still holding `medium`, and a whole
19-episode campaign meant to test the memory frame ran with `QVLA_MENU_FRAMES` at its
default of 1. Both produced complete, plausible, wrong tables, and neither was caught by
anything except reading the numbers weeks later.

`env` removes the typing. `check` removes the trust: it asks the running server what it
actually ended up with and compares that, because the failure mode is never "the export
was missing from the file" -- it is "the export was in the file and the server was
started before it".

`check` exits non-zero on any mismatch, so it is safe to put in front of a benchmark run.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import urllib.error
import urllib.request
from pathlib import Path

import yaml

PROFILE_DIR = Path(__file__).resolve().parent.parent / "config" / "profiles"


def load(name: str) -> dict:
    path = PROFILE_DIR / f"{name}.yaml"
    if not path.exists():
        have = sorted(p.stem for p in PROFILE_DIR.glob("*.yaml"))
        raise SystemExit(f"no profile {name!r} in {PROFILE_DIR} (have: {', '.join(have)})")
    return yaml.safe_load(path.read_text())


def _as_str(value) -> str:
    """YAML booleans come back as Python bools; the shell wants 1 and 0.

    Writing `QVLA_MENU_PIVOTS: false` in the profile and exporting the string "False"
    would set the variable to something the server reads as... false, by accident, since
    it tests against ("1", "true", "yes"). It would work, and it would stop working the
    day someone writes a flag that tests the other way round. Convert deliberately.
    """
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def cmd_env(prof: dict, include_sim: bool, soft: bool) -> int:
    env = dict(prof.get("env") or {})
    if include_sim:
        env.update(prof.get("sim_env") or {})
    for key, value in env.items():
        quoted = shlex.quote(_as_str(value))
        if soft:
            # `: "${K:=V}"` assigns only when K is unset or empty, and `export -- K` then
            # publishes whatever survived. This is the form `launch_qwen.sh` wants: that
            # script takes the thinking level as argument one and exports it BEFORE
            # loading the profile, and a plain `export` here would silently throw that
            # argument away -- the exact class of "ran the cheap level, labelled it the
            # expensive one" mistake the profile exists to prevent.
            print(f': "${{{key}:={quoted}}}"; export -- {key}')
        else:
            print(f"export {key}={quoted}")
    return 0


def cmd_check(prof: dict, host: str, port: int) -> int:
    url = f"http://{host}:{port}/health"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            live = json.loads(resp.read().decode())
    except (urllib.error.URLError, OSError) as exc:
        print(f"FAIL  cannot reach {url}: {exc}", file=sys.stderr)
        return 2

    want = prof.get("health") or {}
    bad: list[str] = []
    for key, expected in want.items():
        if key not in live:
            bad.append(f"  {key}: server does not report this field")
            continue
        got = live[key]
        # Floats via JSON survive exactly for the values in play here (0.45, 10.0), but
        # compare with a tolerance anyway rather than let a future 1/3 fail a check that
        # has nothing to do with the configuration being wrong.
        same = (abs(got - expected) < 1e-6
                if isinstance(expected, float) and isinstance(got, (int, float))
                else got == expected)
        if not same:
            bad.append(f"  {key}: want {expected!r}, server has {got!r}")

    label = f"{prof.get('profile', '?')} @ {host}:{port}"
    if bad:
        print(f"FAIL  {label} does not match the profile:", file=sys.stderr)
        print("\n".join(bad), file=sys.stderr)
        print("\nRestart the server with:  eval \"$(nav/tools/profile.py env)\"",
              file=sys.stderr)
        return 1

    if not live.get("ok"):
        # A server whose config matches but whose model has not finished loading is not
        # a pass -- it is a benchmark that will fail on its first /predict.
        print(f"FAIL  {label} matches the profile but the model is not loaded",
              file=sys.stderr)
        return 1

    print(f"OK    {label}: {len(want)} fields match, model loaded")
    return 0


def cmd_show(prof: dict) -> int:
    m = prof.get("measured") or {}
    print(f"{prof.get('profile')}: {(prof.get('description') or '').strip()}\n")
    if m:
        print(f"  measured {m.get('date')}  arm={m.get('arm')} "
              f"controller={m.get('controller')}")
        print(f"  passed   {m.get('passed')}/{m.get('episodes')}  "
              f"indoor {m.get('indoor')}  outdoor {m.get('outdoor')}")
        floor = m.get("noise_floor_indoor")
        if floor:
            print(f"  noise    prior clean ladders scored {', '.join(floor)} indoor -- "
                  f"read any small difference against this")
    rejected = prof.get("rejected") or []
    if rejected:
        print(f"\n  already tried and lost ({len(rejected)}):")
        for r in rejected:
            print(f"    {r.get('arm'):<10} {r.get('change')}")
            print(f"    {'':<10} -> {r.get('result')}")
    for c in prof.get("caveats") or []:
        print(f"\n  caveat: {' '.join(c.split())}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("env", "check", "show"))
    ap.add_argument("--profile", default="baseline")
    ap.add_argument("--sim", action="store_true",
                    help="env: also export the sim-side variables (NAV_LIDAR etc.)")
    ap.add_argument("--soft", action="store_true",
                    help="env: emit set-if-unset assignments so the environment wins")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=0,
                    help="default: the profile's own NAV_POLICY_PORT")
    args = ap.parse_args()

    prof = load(args.profile)
    if args.command == "env":
        return cmd_env(prof, args.sim, args.soft)
    if args.command == "show":
        return cmd_show(prof)

    port = args.port or int((prof.get("env") or {}).get("NAV_POLICY_PORT", 8766))
    return cmd_check(prof, args.host, port)


if __name__ == "__main__":
    raise SystemExit(main())
