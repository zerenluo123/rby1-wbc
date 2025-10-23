"""Poll rby1 pybind driver state at 100 Hz to reproduce segfaults.

Usage:
  python scripts/read_state_100hz.py --address localhost:50051 --seconds 30
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path


# Ensure project root is on sys.path regardless of cwd
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rby1.control import Config as ControllerConfig, RealtimeDriver  # type: ignore


def main() -> None:
    parser = argparse.ArgumentParser(description="Read robot state at 100 Hz")
    parser.add_argument("--address", default="localhost:50051")
    parser.add_argument("--seconds", type=float, default=30.0)
    args = parser.parse_args()

    # Graceful shutdown on Ctrl+C
    stop = False

    def _sigint(_sig, _frm):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _sigint)

    cfg = ControllerConfig()
    cfg.robot_address = args.address
    driver = RealtimeDriver(cfg)
    driver.start()
    if not driver.wait_until_ready(timeout_sec=15.0):
        driver.stop()
        raise SystemExit("Driver did not become ready in time")

    print("Driver ready. Polling at 100 Hz...")
    t_end = time.time() + float(args.seconds)

    period = 0.01  # 100 Hz
    next_t = time.time()
    num_loops = 0
    num_none = 0
    num_valid = 0
    last_heartbeat = time.time()

    try:
        while not stop and time.time() < t_end:
            snap = driver.get_latest_robot_state()
            if snap is None:
                num_none += 1
            else:
                if getattr(snap, "is_valid", False):
                    num_valid += 1
                else:
                    num_none += 1

            num_loops += 1

            now = time.time()
            if now - last_heartbeat >= 1.0:
                print(
                    f"t+{int(now - (t_end - args.seconds))}s loops={num_loops} valid={num_valid} none={num_none}"
                )
                last_heartbeat = now

            next_t += period
            sleep_dt = next_t - time.time()
            if sleep_dt > 0:
                time.sleep(sleep_dt)
            else:
                # Missed deadline; reset schedule to now
                next_t = time.time()
    finally:
        print(f"Done. loops={num_loops} valid={num_valid} none={num_none}")
        driver.stop()


if __name__ == "__main__":
    main()


