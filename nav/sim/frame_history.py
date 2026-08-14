"""The last few seconds of camera frames, sampled the way TIC-VLA was trained to see.

TIC-VLA is not a single-image model. Its own system prompt says so, in the text the
training code emits verbatim (`ticvla/data/vlm_data.py:_build_messages`):

    "You are provided with a video consisting of visual observations,
     including historical and current frames."

DynaNav honours that by passing **four** frames per inference, sampled at 3-second
intervals — `[-9s, -6s, -3s, current]`, oldest first
(`nova_carter_test_ticvla.py:_get_sampled_image_paths`). We were passing one still.

Why that single still was enough to sink the run: with no view of its own past, the
model has no visual evidence that it has been moving, so nothing in the observation
ever says "you have come far enough, now turn". It answered "go straight" 141 times in
a row, and the trace shows exactly that — yaw held at ~87 degrees for 42 seconds while
the robot drove 31 m past its goal, closest approach 6.06 m, never once turning.

This is the visual half of the same omission as `waypoint_history.py`: that one tells
the model numerically what it has done, this one shows it. Both were defaulted away,
and both are load-bearing.

Sampling rule copied from DynaNav, including the edge cases:
  * an offset is only used if the history actually reaches that far back — early in an
    episode you legitimately get fewer than four frames,
  * offset 0 (the current frame) is always included,
  * duplicates are removed with order preserved, so a young history collapses to one
    frame rather than four copies of it,
  * oldest first, current LAST. The order is part of the format, not a detail.
"""

from __future__ import annotations

from pathlib import Path


class FrameHistory:
    """Recent nav frames, sampled at fixed intervals for the policy prompt."""

    def __init__(self, interval_s: float = 3.0, count: int = 4):
        self.interval_s = interval_s
        self.count = count
        self.reset()

    def reset(self) -> None:
        """Between episodes. Frames from the previous run are a different place."""
        self._frames: list[tuple[float, str]] = []

    def add(self, sim_time: float, path: Path | str) -> None:
        self._frames.append((float(sim_time), str(path)))

    def sample(self, now: float) -> list[str]:
        """Return up to `count` paths, oldest first, current last."""
        if not self._frames:
            return []

        oldest = self._frames[0][0]
        picked: list[str] = []
        for k in range(self.count - 1, -1, -1):  # e.g. 3, 2, 1, 0 -> -9s, -6s, -3s, now
            target = now - k * self.interval_s
            # Skip an offset the history cannot actually reach; k == 0 is always kept,
            # so there is never an inference with no image at all.
            if k > 0 and oldest > target:
                continue
            picked.append(self._nearest_at_or_before(target))

        # Dedupe, preserving order. A history younger than one interval maps several
        # offsets onto the same file, and sending the same JPEG four times is not a
        # video -- it is one frame plus a claim about motion that did not happen.
        seen: set[str] = set()
        unique = []
        for p in picked:
            if p not in seen:
                seen.add(p)
                unique.append(p)
        return unique

    def _nearest_at_or_before(self, target: float) -> str:
        best = self._frames[0][1]
        for t, p in self._frames:
            if t > target:
                break
            best = p
        return best
