"""Reference implementations of the three interfaces in `robot/interfaces.py`.

These exist so that `robot/run_robot.py` runs on a laptop with a webcam and nothing else
-- which is the setup you want for the first hour, when the question is "does the policy
answer and does it point the right way" and a robot underneath would only add ways to be
wrong. None of them is a driver for YOUR hardware, and `opencv_camera.py` is the only one
likely to survive contact with it unchanged.

    opencv_camera.OpenCVCamera   a real camera: USB/CSI/RTSP via cv2. Usually enough.
    sim_base.PrintBase           prints what it would drive. Nothing moves.
    null_range.NullRangeSensor   an explicitly absent sensor, not an empty one.

Write your own next to these and pass it to `RobotRunner`; nothing subclasses anything.
"""
