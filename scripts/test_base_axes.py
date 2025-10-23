"""Send body-frame base twist commands to verify axis mapping.

Phase 1 (5s): [0.2, 0.0, 0.0]  => Forward along +X body
Phase 2 (5s): [0.0, 0.2, 0.0]  => Left along +Y body
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


# Ensure project root on sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rby1.control import Config as ControllerConfig, RealtimeDriver  # type: ignore


def run_test(address: str, rate_hz: float) -> None:
    cfg = ControllerConfig()
    cfg.robot_address = address
    driver = RealtimeDriver(cfg)
    driver.start()
    if not driver.wait_until_ready(timeout_sec=15.0):
        driver.stop()
        raise SystemExit("Controller did not become ready in time")

    print("Driver ready. Sending base twists...")
    period = 1.0 / rate_hz

    def _hold_twist(twist: np.ndarray, seconds: float) -> None:
        t_end = time.time() + seconds
        while time.time() < t_end:
            driver.set_base_twist_command(twist)
            time.sleep(period)

    try:
        # print("Phase 1: [0.2, 0.0, 0.0] for 5s (forward)")
        # _hold_twist(np.array([0.2, 0.0, 0.0], dtype=float), 5.0)

        print("Phase 2: [0.0, 0.2, 0.0] for 5s (left)")
        _hold_twist(np.array([0.0, 0.2, 0.0], dtype=float), 10.0)

        print("Phase 3: [0.0, 0.0, 0.2] for 5s (spin)")
        _hold_twist(np.array([0.0, 0.0, 0.2], dtype=float), 10.0)
    finally:
        driver.set_base_twist_command(np.array([0.0, 0.0, 0.0], dtype=float))
        driver.stop()
        print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test base axis mapping by sending body twists")
    parser.add_argument("--address", default="localhost:50051", help="Robot gRPC address")
    parser.add_argument("--rate", type=float, default=100.0, help="Command rate (Hz)")
    args = parser.parse_args()
    run_test(args.address, args.rate)


if __name__ == "__main__":
    main()


