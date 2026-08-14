"""Render the nav camera and hand the policy server a file path.

Frames cross the process boundary as paths on disk rather than as bytes in JSON.
Both processes are on this one machine and TICVLA.predict() already takes
`image_paths: list[str]`, so encoding a 640x480 render into JSON would be pure
overhead on the hot path -- a policy call already costs 1.0-1.5 s and runs every
0.5 s of sim time.

The one thing this couples is the scratch directory: the sim process writes it and
the policy process reads it, so they must agree. The policy server returns HTTP 400
naming the missing paths if they ever disagree, rather than failing somewhere deeper.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from alohamini1_specs import CAMERA_RESOLUTION, NAV_CAMERA_PRIM_PATH  # noqa: E402

DEFAULT_SCRATCH = Path("/tmp/alohamini-nav-frames")


class NavCameraSource:
    """The robot's forward-facing navigation camera.

    Note this is NAV_CAMERA_PRIM_PATH, not CAMERA_PRIM_PATHS["forward"]. The three
    LeRobot cameras all face the manipulation front (-Y) while the base drives +X; a
    navigation policy fed the "forward" camera would be reasoning about the view 90
    degrees off its direction of travel. See scripts/pipeline/add_cameras.py.
    """

    def __init__(
        self,
        scratch_dir: Path | str = DEFAULT_SCRATCH,
        prim_path: str = NAV_CAMERA_PRIM_PATH,
        resolution: tuple[int, int] = CAMERA_RESOLUTION,
        keep_frames: bool = True,
    ):
        from isaacsim.sensors.camera import Camera

        self.scratch_dir = Path(scratch_dir)
        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        self.keep_frames = keep_frames
        self._frame_index = 0

        self._camera = Camera(prim_path=prim_path, resolution=resolution)
        self._camera.initialize()

    def warmup(self, kit, steps: int = 30) -> None:
        """The first renders come back empty -- the sensor pipeline needs to spin up.

        Same warmup capture_cameras.py does. Calling predict() on an empty frame does
        not error, it just returns a plan computed from a black image, which is far
        harder to notice than a crash.
        """
        for _ in range(steps):
            kit.update()

    def grab(self) -> np.ndarray | None:
        rgba = self._camera.get_rgba()
        if rgba is None or rgba.size == 0:
            return None
        return rgba[:, :, :3].astype(np.uint8)

    def grab_to_file(self, tag: str = "") -> Path | None:
        """Render one frame and write it. Returns the path, or None if empty."""
        frame = self.grab()
        if frame is None:
            return None

        from PIL import Image

        name = f"nav_{self._frame_index:06d}{('_' + tag) if tag else ''}.jpg"
        path = self.scratch_dir / name
        # JPEG, quality 95: the policy resizes to 448x448 internally anyway, so PNG's
        # losslessness buys nothing and costs ~8x the write time on the hot path.
        Image.fromarray(frame).save(path, quality=95)

        if not self.keep_frames and self._frame_index > 0:
            prev = self.scratch_dir / f"nav_{self._frame_index - 1:06d}.jpg"
            prev.unlink(missing_ok=True)

        self._frame_index += 1
        return path

    @property
    def frames_written(self) -> int:
        return self._frame_index
