#!/usr/bin/env python3
"""Print the environment an argument names. Accepts an environment or an episode.

    $ nav/sim/resolve_env.py hospital_down_hallway
    hospital
    $ nav/sim/resolve_env.py hospital
    hospital

Exists as a file rather than an inline heredoc because the shell scripts that need it
already contain heredocs, and nesting them silently truncates the outer one at the
first line matching its delimiter. Also useful on its own: "which stage do I have to
build for this episode?" is now one command.

Deliberately runs on system python3, not Kit's -- it only reads YAML, and paying a
90 s SimulationApp boot to answer a naming question would be absurd.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from episode import env_name, episodes_by_env, load_episodes  # noqa: E402


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {Path(argv[0]).name} <environment|episode>", file=sys.stderr)
        return 2

    want = argv[1]
    environments = episodes_by_env()
    if want in environments:
        print(want)
        return 0

    episodes = load_episodes()
    if want in episodes:
        print(env_name(episodes[want].scene))
        return 0

    print(
        f"no such environment or episode: {want!r}\n"
        f"  environments: {', '.join(sorted(environments))}\n"
        f"  episodes    : {', '.join(sorted(episodes))}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
