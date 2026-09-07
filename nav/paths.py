#!/usr/bin/env python3
"""Where the things that are not in this repository live.

    $ python3 nav/paths.py            # print every resolved root and where it came from

Four external things this code needs and none of which git can carry: the TIC-VLA
checkout (a submodule), the model weights (tens of GB), the Isaac Sim install, and the
virtualenvs. Until now each was a `/home/gtu-dsa/...` default spelled out at the point of
use, which is fine on the one machine those paths are true on and is the first thing that
breaks for anyone who clones this.

THE RESOLUTION ORDER IS THE POINT. Environment variable first, then the submodule, then
this machine's historical location, then a readable error. The third step is what lets
the change be made without a flag day -- the workstation keeps running unmodified while
`third_party/TIC-VLA` sits uninitialised -- and the fourth is what stops a fresh clone
from failing somewhere deep in a scene loader with `FileNotFoundError: ''`.

Nothing here downloads anything or has an opinion about how you install it. It answers
"where is it", and says clearly what to do when the answer is nowhere.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# The historical locations on the GTU workstation. Kept as a LAST-RESORT fallback rather
# than deleted, so that this refactor cannot break a run already in flight, and so the
# machine that produced every number in nav/results keeps reproducing them. A clone
# elsewhere never finds these and falls through to the error, which is the intent.
_LEGACY = {
    "ticvla": Path("/home/gtu-dsa/robotics/TIC-VLA"),
    "models": Path("/home/gtu-dsa/robotics/models"),
    "isaacsim": Path("/home/gtu-dsa/robotics/isaacsim-6.0.1"),
}


class MissingDependency(RuntimeError):
    """Raised with instructions, not just a path."""


def _resolve(kind: str, env_var: str, submodule: Path | None,
             marker: str, how_to_get_it: str) -> Path:
    """First of: $env_var, the submodule, the legacy path. Otherwise explain.

    `marker` is a file that must exist inside the candidate. An empty directory is the
    normal state of an uninitialised submodule, so "the directory exists" is exactly the
    wrong test -- it would resolve to `third_party/TIC-VLA`, hand back a path containing
    nothing, and turn a clear error here into a confusing one several layers down.
    """
    override = os.environ.get(env_var, "").strip()
    if override:
        path = Path(override).expanduser()
        if not (path / marker).exists():
            # An override that is wrong is worse than no override: it was set on purpose,
            # so silently falling through to a default would defeat the setting AND hide
            # the typo. Fail on it instead.
            raise MissingDependency(
                f"{env_var}={override} does not look like a {kind} checkout "
                f"({marker} is missing there)")
        return path

    for candidate in (submodule, _LEGACY.get(kind)):
        if candidate is not None and (candidate / marker).exists():
            return candidate

    raise MissingDependency(
        f"cannot find {kind}.\n"
        f"  Looked at: ${env_var}"
        + (f", {submodule}" if submodule else "")
        + f", {_LEGACY.get(kind)}\n"
        f"  {how_to_get_it}\n"
        f"  Or point at an existing copy:  export {env_var}=/path/to/{kind}")


def ticvla_root() -> Path:
    """The TIC-VLA checkout. Submodule at third_party/TIC-VLA.

    Needed to REPRODUCE the simulated benchmark -- the DynaNav episode definitions and
    the four .usd scenes live in it. It is not needed to run the policy, and it is not
    needed on a real robot at all. Worth knowing before you fetch it: the checkout is
    several GB, most of it scene assets.
    """
    return _resolve(
        "ticvla", "TICVLA_ROOT", REPO / "third_party" / "TIC-VLA",
        marker="DynaNav",
        how_to_get_it="Fetch it with:  git submodule update --init third_party/TIC-VLA")


def dynanav_root() -> Path:
    """TIC-VLA's DynaNav subdirectory -- benchmark configs and scene assets.

    Its own variable because DynaNav's upstream config uses TICVLA_DYNANAV_ROOT, and
    honouring the name upstream chose costs nothing and saves a translation step.
    """
    override = os.environ.get("TICVLA_DYNANAV_ROOT", "").strip()
    if override:
        return Path(override).expanduser()
    return ticvla_root() / "DynaNav"


def model_root() -> Path:
    """Where downloaded weights live. Not a repository -- tens of GB of HF snapshots.

    Only the TIC-VLA comparison path needs this. The shipping policy names its model as
    a Hugging Face id (see nav/config/profiles/baseline.yaml), so it resolves through the
    HF cache and never comes through here.
    """
    return _resolve(
        "models", "QVLA_MODEL_ROOT", None,
        marker=".",
        how_to_get_it=("Any directory holding the downloaded weights will do; "
                       "see docs/ for what belongs in it."))


def isaacsim_python() -> Path:
    """Isaac Sim's own python.sh. Kit will not import into an outside interpreter."""
    override = os.environ.get("ISAACSIM_PYTHON", "").strip()
    if override:
        return Path(override).expanduser()
    return _resolve(
        "isaacsim", "ISAACSIM_ROOT", None,
        marker="python.sh",
        how_to_get_it=("Install from the standalone zip -- NOT pip. pypi.nvidia.com "
                       "runs at 31 KB/s from some networks and simply times out.")) \
        / "python.sh"


def _describe(name: str, fn) -> str:
    try:
        return f"  {name:<16} {fn()}"
    except MissingDependency as exc:
        return f"  {name:<16} MISSING -- {str(exc).splitlines()[0]}"


_SINGLE = {
    "--ticvla": ticvla_root,
    "--dynanav": dynanav_root,
    "--models": model_root,
    "--isaacsim-python": isaacsim_python,
}

if __name__ == "__main__":
    import sys

    # One flag prints one bare path and nothing else, so a shell script can capture it.
    # Errors keep going to stderr and set a non-zero status, which is what lets the
    # callers use a plain assignment and let `set -e` do its job.
    if len(sys.argv) == 2 and sys.argv[1] in _SINGLE:
        try:
            print(_SINGLE[sys.argv[1]]())
        except MissingDependency as exc:
            raise SystemExit(f"{Path(sys.argv[0]).name}: {exc}")
        raise SystemExit(0)
    if len(sys.argv) > 1:
        raise SystemExit(f"usage: paths.py [{' | '.join(_SINGLE)}]")

    print(f"  {'repo':<16} {REPO}")
    for label, func in (("ticvla_root", ticvla_root), ("dynanav_root", dynanav_root),
                        ("model_root", model_root), ("isaacsim_python", isaacsim_python)):
        print(_describe(label, func))
