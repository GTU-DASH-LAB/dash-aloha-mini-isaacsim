"""Run this on YOUR LOCAL machine (the one with the PS4 controller plugged in), not on
the Isaac Sim machine. Reads the controller with pygame and streams its state as
newline-delimited JSON over a TCP connection to
scripts/control_terminal.py --joystick-network running on the remote box.

Setup on your Mac:
    python3 -m pip install pygame
    python3 joystick_bridge_local.py --host <remote-ip-or-localhost> --port 9999

If the remote machine (10.1.18.165 on its LAN) isn't directly reachable from your Mac
(different networks -- likely, since you're going through AnyDesk), tunnel it over SSH
instead of trying a direct connection. From your Mac:
    ssh -L 9999:localhost:9999 <user>@<remote-host-or-ip>
Leave that running in its own terminal, then run this script with
--host localhost --port 9999 (the tunnel forwards it through). This works with plain
`ssh -L` with no extra tools because the bridge uses TCP, not UDP.

This script auto-reconnects if the TCP connection drops (e.g. the SSH tunnel restarts
or control_terminal.py is relaunched on the other end) -- no need to restart it.

Debug first: run with --debug to print raw button/axis events instead of sending
anything, so you can confirm this maps correctly to your specific controller before
trusting it. NOT tested against real PS4 hardware -- built against pygame's standard
joystick API and common (but not universally identical across OS/driver) button/axis
conventions. If --debug shows different indices than the DEFAULT_MAPPING below, edit
DEFAULT_MAPPING or pass --button-l1 etc. to override.

Controls: L1=right arm, L2=left arm, L1+L2=both arms mirrored, R2=base. Left stick and
right stick control different joints/axes depending on mode -- see
control_terminal.py's module docstring for the full mapping.
"""

import argparse
import json
import socket
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--host", default="127.0.0.1", help="Remote host running control_terminal.py --joystick-network")
parser.add_argument("--port", type=int, default=9999)
parser.add_argument("--rate", type=float, default=30.0, help="Packets per second to send")
parser.add_argument("--debug", action="store_true", help="Print raw events instead of sending packets")
parser.add_argument("--joystick-index", type=int, default=0, help="Which controller if you have more than one")
# Overrides in case --debug shows different indices than the defaults below.
parser.add_argument("--button-l1", type=int, default=None)
parser.add_argument("--button-r1", type=int, default=None)
parser.add_argument("--button-l2", type=int, default=None)
parser.add_argument("--button-r2", type=int, default=None)
parser.add_argument("--button-cross", type=int, default=None)
parser.add_argument("--button-circle", type=int, default=None)
parser.add_argument("--axis-l2", type=int, default=None, help="Analog L2 trigger axis, if your controller has one")
parser.add_argument("--axis-r2", type=int, default=None, help="Analog R2 trigger axis, if your controller has one")
args = parser.parse_args()

try:
    import pygame
except ImportError:
    print("pygame not installed. Run: python3 -m pip install pygame")
    sys.exit(1)

# Common pygame/SDL PS4 controller mapping. Varies by OS/driver -- verify with --debug.
DEFAULT_MAPPING = {
    "button_l1": 4,
    "button_r1": 5,
    "button_l2": 6,   # some drivers expose L2 only as a digital button at this index
    "button_r2": 7,   # ...ditto for R2
    "button_cross": 0,
    "button_circle": 1,
    "axis_lx": 0,
    "axis_ly": 1,
    "axis_rx": 2,
    "axis_ry": 3,
    "axis_l2": None,  # set via --axis-l2 if your controller has an analog trigger axis
    "axis_r2": None,
    "hat_index": 0,
}
for key, cli_val in (
    ("button_l1", args.button_l1), ("button_r1", args.button_r1),
    ("button_l2", args.button_l2), ("button_r2", args.button_r2),
    ("button_cross", args.button_cross), ("button_circle", args.button_circle),
    ("axis_l2", args.axis_l2), ("axis_r2", args.axis_r2),
):
    if cli_val is not None:
        DEFAULT_MAPPING[key] = cli_val

pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    print("No joystick/controller detected by pygame. Is it connected and paired (if Bluetooth)?")
    sys.exit(1)

joystick = pygame.joystick.Joystick(args.joystick_index)
joystick.init()
print(f"Using: {joystick.get_name()}  "
      f"(buttons={joystick.get_numbuttons()}, axes={joystick.get_numaxes()}, hats={joystick.get_numhats()})")

if args.debug:
    print("\n--debug mode: printing raw events. Press buttons / move sticks. Ctrl+C to stop.")
    print("Use this to find your controller's real button/axis indices if the defaults don't match.\n")
    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.JOYBUTTONDOWN:
                    print(f"Button DOWN: index={event.button}")
                elif event.type == pygame.JOYBUTTONUP:
                    print(f"Button UP:   index={event.button}")
                elif event.type == pygame.JOYAXISMOTION:
                    if abs(event.value) > 0.2:  # ignore small noise so output is readable
                        print(f"Axis moved:  index={event.axis} value={event.value:.3f}")
                elif event.type == pygame.JOYHATMOTION:
                    print(f"Hat moved:   index={event.hat} value={event.value}")
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\nStopped.")
    sys.exit(0)

def connect():
    while True:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((args.host, args.port))
            print(f"Connected to {args.host}:{args.port}")
            return s
        except (ConnectionRefusedError, OSError) as e:
            print(f"Couldn't connect to {args.host}:{args.port} ({e}) -- retrying in 2s. "
                  f"Is control_terminal.py --joystick-network running there (and the SSH "
                  f"tunnel up, if you're using one)?")
            time.sleep(2)


sock = connect()
print(f"Sending controller state to {args.host}:{args.port} at {args.rate} Hz. Ctrl+C to stop.")

period = 1.0 / args.rate
m = DEFAULT_MAPPING
try:
    while True:
        pygame.event.pump()  # refresh internal state without needing the event queue

        def get_axis_safe(idx):
            return joystick.get_axis(idx) if idx is not None and idx < joystick.get_numaxes() else 0.0

        def get_button_safe(idx):
            return bool(joystick.get_button(idx)) if idx is not None and idx < joystick.get_numbuttons() else False

        l2_analog = get_axis_safe(m["axis_l2"])
        r2_analog = get_axis_safe(m["axis_r2"])
        # Analog trigger axes on most controllers report -1 (released) to 1 (fully
        # pressed); treat > 0 as "held" if an analog axis is configured, else fall
        # back to the digital button.
        l2_held = (l2_analog > 0) if m["axis_l2"] is not None else get_button_safe(m["button_l2"])
        r2_held = (r2_analog > 0) if m["axis_r2"] is not None else get_button_safe(m["button_r2"])

        hat_y = 0
        if joystick.get_numhats() > 0:
            _, hat_y = joystick.get_hat(m["hat_index"])

        packet = {
            "lx": get_axis_safe(m["axis_lx"]),
            "ly": get_axis_safe(m["axis_ly"]),
            "rx": get_axis_safe(m["axis_rx"]),
            "ry": get_axis_safe(m["axis_ry"]),
            "l1": get_button_safe(m["button_l1"]),
            "l2": l2_held,
            "r2": r2_held,
            "hat_y": hat_y,
            "gripper_open": get_button_safe(m["button_cross"]),
            "gripper_close": get_button_safe(m["button_circle"]),
        }
        try:
            sock.sendall((json.dumps(packet) + "\n").encode("utf-8"))
        except (BrokenPipeError, ConnectionResetError, OSError):
            print("Connection lost -- reconnecting...")
            sock = connect()
        time.sleep(period)
except KeyboardInterrupt:
    print("\nStopped.")
