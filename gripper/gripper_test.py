import argparse
import json
import time

from gripper.gripper_client import GripperClient


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick smoke test for the remote RB-Y1 gripper.")
    parser.add_argument("--host", default="192.168.30.2", help="Gripper server host/IP")
    parser.add_argument("--port", type=int, default=5678, help="Gripper server TCP port")
    parser.add_argument(
        "--open-width",
        type=float,
        default=0.07,
        help="Target opening width (meters) for the first command",
    )
    parser.add_argument(
        "--close-width",
        type=float,
        default=0.01,
        help="Target opening width (meters) for the second command",
    )
    parser.add_argument(
        "--hold-sec",
        type=float,
        default=1.0,
        help="Seconds to wait between commands",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    with GripperClient(args.host, port=args.port) as gripper:
        print("Ping:", gripper.ping())
        status = gripper.status()
        print("Server status:", json.dumps(status, indent=2))
        print(f"Setting gripper to ({args.open_width:.3f}, {args.open_width:.3f}) m")
        gripper.set_target([args.open_width, args.open_width])
        time.sleep(args.hold_sec)
        print(f"Setting gripper to ({args.close_width:.3f}, {args.close_width:.3f}) m")
        gripper.set_target([args.close_width, args.close_width])


if __name__ == "__main__":
    main()
