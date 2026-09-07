#!/usr/bin/env python3
"""A `CameraSource` over anything OpenCV can open: USB, CSI, RTSP, or a video file.

    cam = OpenCVCamera(0)                       # first USB camera
    cam = OpenCVCamera("rtsp://192.168.1.9/h264")
    cam = OpenCVCamera("clip.mp4", loop=True)   # bench testing with no robot

This is the one driver in `robot/drivers/` that will probably work on your hardware
unchanged, because OpenCV already sits on top of V4L2, GStreamer and FFmpeg. If your
camera has its own SDK -- a RealSense, a ZED, an Arducam on CSI -- write a class with
`grab()` and `close()` against that SDK instead. It does not need to import anything from
this repository.
"""

from __future__ import annotations

import threading
import time

try:
    import cv2
except ImportError as exc:  # pragma: no cover -- an install problem, not a runtime one
    raise SystemExit(
        "OpenCVCamera needs opencv: pip install opencv-python-headless\n"
        "(headless unless you also want cv2.imshow; the runner never displays anything)"
    ) from exc


class OpenCVCamera:
    """Grabs frames on a background thread and hands out the newest one, JPEG-encoded.

    THE THREAD IS NOT AN OPTIMISATION. A USB camera delivers into a driver-side queue,
    and `VideoCapture.read()` returns the OLDEST frame in it, not the newest. Reading at
    the decision rate -- once every 3 s -- therefore returns a frame from the start of
    the queue, and the lag grows until the buffer is full. The robot ends up steering on
    a view of where it used to be, smoothly, with correct-looking imagery. Draining
    continuously on a thread is what makes `grab()` mean "now".

    `CAP_PROP_BUFFERSIZE` addresses the same thing and is set below, but it is a request:
    plenty of V4L2 and every RTSP backend ignore it. The thread does not depend on it.
    """

    def __init__(
        self,
        source: int | str = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        jpeg_quality: int = 85,
        loop: bool = False,
        warmup_s: float = 2.0,
    ):
        self.source = source
        self.jpeg_quality = int(jpeg_quality)
        self.loop = loop

        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"could not open camera source {source!r}. For a USB camera try a "
                f"different index (0, 1, 2), check `v4l2-ctl --list-devices`, and check "
                f"your user is in the `video` group.")
        # Requests, all of them. A camera that does not support the mode silently keeps
        # its own, which is why the resolution is read back below rather than assumed.
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.actual_size = (int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.frames_read = 0
        self.encode_failures = 0

        self._latest: bytes | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

        # Most USB cameras hand back several black or badly exposed frames while auto
        # exposure settles. Those are a fine first policy input in the sense that nothing
        # crashes, and a bad one in the sense that the first decision of every run is
        # taken on a dark rectangle.
        deadline = time.monotonic() + warmup_s
        while time.monotonic() < deadline:
            with self._lock:
                if self._latest is not None:
                    break
            time.sleep(0.05)

    def _pump(self) -> None:
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok:
                if self.loop:
                    # A file that ran out. Rewind; a live camera never gets here.
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                # A live camera that stopped delivering: unplugged, or a USB reset. Do
                # not spin -- back off and keep trying, and let `grab()` return the last
                # good frame becoming stale rather than raising into the control loop.
                time.sleep(0.1)
                continue
            ok, buf = cv2.imencode(".jpg", frame,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
            if not ok:
                self.encode_failures += 1
                continue
            with self._lock:
                self._latest = buf.tobytes()
                self.frames_read += 1

    def grab(self) -> bytes | None:
        """Newest frame as JPEG bytes, or None if none has arrived yet."""
        with self._lock:
            return self._latest

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._cap.release()


if __name__ == "__main__":
    # Bring-up check: does the camera open, and are the frames a sane size?
    import argparse
    import pathlib

    ap = argparse.ArgumentParser(description="Grab one frame and report on it.")
    ap.add_argument("--source", default="0", help="index, device path, URL or file")
    ap.add_argument("--out", default="/tmp/qvla-camera-check.jpg")
    args = ap.parse_args()

    src: int | str = int(args.source) if args.source.isdigit() else args.source
    cam = OpenCVCamera(src)
    try:
        data = cam.grab()
        if data is None:
            raise SystemExit("opened, but produced no frame within the warmup window")
        pathlib.Path(args.out).write_bytes(data)
        print(f"{cam.actual_size[0]}x{cam.actual_size[1]}, {len(data)/1024:.1f} KiB JPEG "
              f"-> {args.out}")
        print("Open it and check the camera faces the direction the robot drives.")
    finally:
        cam.close()
