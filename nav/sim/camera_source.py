"""Render a robot camera and hand out the frame — as a file path, or as JPEG bytes.

Two cameras matter here and they serve different masters:

  nav    the frame the POLICY sees. Written to disk, because frames cross the process
         boundary as paths rather than bytes -- both processes are on this machine and
         TICVLA.predict() already takes `image_paths`, so encoding a render into JSON
         would be pure overhead on a hot path.
  chase  a third-person view for a HUMAN. Never sent to the policy. Only ever needs to
         reach the browser, so it stays in memory as JPEG bytes and never touches disk.

The chase camera is created LAZILY, on first request from the UI. That is not
premature optimisation: an Isaac Sim `Camera` allocates a render product, and Kit then
renders it every single frame whether or not anyone reads it. Someone running a
headless benchmark should not pay for a view nobody is watching.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from alohamini1_specs import (  # noqa: E402
    CAMERA_RESOLUTION,
    CHASE_CAMERA_PRIM_PATH,
    NAV_CAMERA_PRIM_PATH,
)

DEFAULT_SCRATCH = Path("/tmp/alohamini-nav-frames")


class _Source:
    """Thin wrapper over an Isaac Sim Camera. Not used directly."""

    def __init__(self, prim_path: str, resolution: tuple[int, int] = CAMERA_RESOLUTION):
        from isaacsim.sensors.camera import Camera

        self.prim_path = prim_path
        self._camera = Camera(prim_path=prim_path, resolution=resolution)
        self._camera.initialize()

    def grab(self) -> np.ndarray | None:
        rgba = self._camera.get_rgba()
        if rgba is None or rgba.size == 0:
            return None
        return rgba[:, :, :3].astype(np.uint8)

    def grab_jpeg(self, quality: int = 85) -> bytes | None:
        frame = self.grab()
        if frame is None:
            return None
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(frame).save(buf, format="JPEG", quality=quality)
        return buf.getvalue()


class NavCameraSource(_Source):
    """The robot's forward-facing navigation camera — what the policy is shown.

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
        super().__init__(prim_path, resolution)
        self.scratch_dir = Path(scratch_dir)
        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        self.keep_frames = keep_frames
        self._frame_index = 0

    def warmup(self, kit, steps: int = 30) -> None:
        """The first renders come back empty — the sensor pipeline needs to spin up.

        Calling predict() on an empty frame does not error, it just returns a plan
        computed from a black image, which is far harder to notice than a crash.
        """
        for _ in range(steps):
            kit.update()

    def grab_to_file(self, tag: str = "") -> Path | None:
        """Render one frame and write it. Returns the path, or None if empty."""
        frame = self.grab()
        if frame is None:
            return None

        from PIL import Image

        name = f"nav_{self._frame_index:06d}{('_' + tag) if tag else ''}.jpg"
        path = self.scratch_dir / name
        # quality 95: the policy resizes to 448x448 internally, so PNG's losslessness
        # buys nothing and costs ~8x the write time on the hot path.
        Image.fromarray(frame).save(path, quality=95)

        if not self.keep_frames and self._frame_index > 0:
            prev = self.scratch_dir / f"nav_{self._frame_index - 1:06d}.jpg"
            prev.unlink(missing_ok=True)

        self._frame_index += 1
        return path

    @property
    def frames_written(self) -> int:
        return self._frame_index


class ChaseCameraSource(_Source):
    """Third-person view, for watching. Never shown to the policy.

    Parented to base_link in USD, so it follows the robot's position and yaw for free
    -- no per-step repositioning, and it keeps facing the way the robot faces.
    """

    def __init__(
        self,
        prim_path: str = CHASE_CAMERA_PRIM_PATH,
        resolution: tuple[int, int] = CAMERA_RESOLUTION,
    ):
        super().__init__(prim_path, resolution)


def try_create_chase(prim_path: str = CHASE_CAMERA_PRIM_PATH) -> ChaseCameraSource | None:
    """Create the chase camera if the scene actually has one.

    Nav scenes built before the chase camera existed do not carry the prim, and the
    right answer there is a clear message telling the user to rebuild -- not a stack
    trace out of the render pipeline halfway through an episode.
    """
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    if stage is None or not stage.GetPrimAtPath(prim_path).IsValid():
        return None
    return ChaseCameraSource(prim_path)
