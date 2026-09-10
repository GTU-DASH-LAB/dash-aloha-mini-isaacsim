"""Download the five survey candidates, smallest first, resumable.

Smallest first is not tidiness -- it is so the benchmark can start on candidate 1 while
candidate 5 is still arriving. The whole set is ~45 GB and the 8B alone is a third of it.

`hf_transfer` is deliberately NOT used and is asserted absent: it cannot resume. It
deletes the partial file and restarts from zero, which on this network has already cost
70.98 GB of re-downloading. Plain `snapshot_download` resumes from the incomplete blob.

Weights land in `paths.model_root` (`robotics/models/`), one directory per candidate,
alongside the Qwen baseline -- not in `~/.cache/huggingface`, so that `du` on one path
answers "what do the models cost" and a candidate can be deleted by removing a directory.

Usage:
    /home/gtu-dsa/envs/qvla/bin/python nav/tools/fetch_vlm_candidates.py
    /home/gtu-dsa/envs/qvla/bin/python nav/tools/fetch_vlm_candidates.py --only qwen3vl-4b
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths  # noqa: E402

# Keys are the short names used everywhere downstream: the server's --candidate, the
# results filenames, the comparison table. Order is ascending download size.
CANDIDATES: dict[str, str] = {
    "smolvlm2-2.2b": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    "qwen3vl-4b": "Qwen/Qwen3-VL-4B-Instruct",
    "gemma3-4b": "google/gemma-3-4b-it",
    "internvl3.5-4b": "OpenGVLab/InternVL3_5-4B-HF",
    "qwen3vl-8b": "Qwen/Qwen3-VL-8B-Instruct",
}

# Everything needed to load and run, and nothing else. Without this the 4B repos also
# drag down .pth consolidated copies and ONNX exports that no path in this repo reads.
ALLOW = ["*.json", "*.safetensors", "*.txt", "*.model", "*.py", "*.jinja"]


def human(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0
    return f"{n:.1f} GiB"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", action="append", choices=sorted(CANDIDATES),
                    help="Fetch just these (repeatable). Default: all five.")
    # `paths.model_root` is a FUNCTION, and stringifying it without calling it yields
    # "<function model_root at 0x...>" -- a perfectly valid directory name, which
    # `mkdir(parents=True)` then created and downloaded 261 MB into before anyone noticed.
    # No error, no warning, a plausible-looking path in the log.
    ap.add_argument("--dest", default=str(paths.model_root()))
    args = ap.parse_args()

    if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "0") not in ("0", "", "false"):
        print("refusing to run with HF_HUB_ENABLE_HF_TRANSFER set -- it cannot resume",
              file=sys.stderr)
        return 2

    from huggingface_hub import snapshot_download

    dest_root = Path(args.dest)
    dest_root.mkdir(parents=True, exist_ok=True)
    wanted = args.only or list(CANDIDATES)

    for name in wanted:
        repo = CANDIDATES[name]
        out = dest_root / name
        t0 = time.perf_counter()
        print(f"\n=== {name}  <-  {repo}\n    -> {out}", flush=True)
        snapshot_download(repo_id=repo, local_dir=str(out), allow_patterns=ALLOW,
                          max_workers=8)
        # `du -B1`, not `du -sb`: the second reports apparent size, which on a partially
        # written file is the size it WILL be, so a stalled download looks complete.
        size = sum(f.stat().st_blocks * 512 for f in out.rglob("*") if f.is_file())
        print(f"    done in {time.perf_counter() - t0:.0f} s, {human(size)} on disk",
              flush=True)

    print("\nall requested candidates present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
