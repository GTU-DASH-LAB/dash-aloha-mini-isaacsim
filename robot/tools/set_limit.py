#!/usr/bin/env python3
"""Change one limit from the shell, with or without the robot running.

    robot/tools/set_limit.py                    # show what is in force
    robot/tools/set_limit.py cruise_mps 0.2
    robot/tools/set_limit.py enabled false

Writes `robot/robot.yaml` through the same `config.save()` the control API uses, and the
runner picks it up on its next control tick. Nothing here talks to the running process --
the file IS the interface, which is why this works whether or not the robot is up, and
why a limit set here survives a restart.

Use this over `POST /limits` when you are on the robot itself. Use the control API when
you are on your laptop and the robot is across the room.
"""

from __future__ import annotations

import sys
from dataclasses import fields, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as robot_config  # noqa: E402

_BOOL = {"true": True, "false": False, "yes": True, "no": False,
         "1": True, "0": False, "on": True, "off": False}


def _coerce(name: str, raw: str):
    field = {f.name: f for f in fields(robot_config.Limits)}[name]
    if field.type in ("bool", bool):
        key = raw.strip().lower()
        if key not in _BOOL:
            raise SystemExit(f"{name} is a true/false setting; got {raw!r}")
        return _BOOL[key]
    try:
        return float(raw)
    except ValueError as exc:
        raise SystemExit(f"{name} wants a number; got {raw!r}") from exc


def main(argv: list[str]) -> int:
    path = robot_config.DEFAULT_PATH
    if argv and argv[0] == "--path":
        path, argv = Path(argv[1]), argv[2:]

    current = robot_config.load(path)
    if not argv:
        print(f"# {path}")
        for f in fields(current):
            print(f"{f.name:26s} {getattr(current, f.name)}")
        return 0

    if len(argv) != 2:
        print(__doc__.strip())
        return 2

    name, raw = argv
    known = {f.name for f in fields(robot_config.Limits)}
    if name not in known:
        print(f"unknown limit {name!r}. Known: {', '.join(sorted(known))}")
        return 2

    value = _coerce(name, raw)
    merged = replace(current, **{name: value}).validated()
    robot_config.save(merged, path)
    got = getattr(merged, name)
    # Reported explicitly, because a clamp that only prints inside the runner is a change
    # the operator believes they made and did not.
    if got != value:
        print(f"{name}: {getattr(current, name)} -> {got} (clamped from {value})")
    else:
        print(f"{name}: {getattr(current, name)} -> {got}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
