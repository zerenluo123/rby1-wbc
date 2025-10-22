"""Minimal smoke test for the rby1 pybind module."""

import argparse
import sys

from pyparsing import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    
from rby1.control import RealtimeDriver, Config, debug_echo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", default="localhost:50051")
    parser.add_argument("--seconds", type=float, default=1.0)
    args = parser.parse_args()

    print("Calling debug_echo...")
    debug_echo("hello from python")

    cfg = Config()
    cfg.robot_address = args.address

    print(f"Constructing driver for {cfg.robot_address}...")
    driver = RealtimeDriver(cfg)
    print("Driver constructed successfully.")

    print("Starting control loop stub...")
    driver.start()
    driver.wait_until_ready(timeout_sec=args.seconds)
    print("Stop request...")
    driver.stop()
    print("Done.")


if __name__ == "__main__":
    main()
