"""iPhone teleoperation frontend for the standalone RBY1 WBC thread."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

# Ensure project root is on sys.path regardless of current working directory.
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from control.rby1_wbc import RBY1WBC
from teleop.teleop_iphone import TeleopIphone

# Reuse the GUI/IK integration logic implemented for the VR teleop frontend.
from rby1_wbc_teleop_vr import RBY1WBCTeleopVR


class RBY1WBCTeleopIphone(RBY1WBCTeleopVR):
    """Thin wrapper to keep the type name descriptive for the iPhone frontend."""

    def __init__(self, wbc: RBY1WBC, teleop: TeleopIphone, headless: bool = False) -> None:
        super().__init__(wbc=wbc, teleop=teleop, headless=headless)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RBY1 whole-body teleop GUI driven by iPhone pose streaming"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host/IP address to bind the local pose server for the iPhone app.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5555,
        help="TCP port to bind the local pose server for the iPhone app.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Skip launching the MuJoCo viewer (useful for debugging controller only).",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Persist computed teleop trajectory as a dataset-style pickle under demo/.",
    )
    parser.add_argument(
        "--portrait",
        action="store_true",
        help="Apply portrait-mode rotation to incoming iPhone poses.",
    )
    args = parser.parse_args()

    if args.headless:
        os.environ.setdefault("MUJOCO_GL", "egl")

    wbc = RBY1WBC()
    wbc.start()
    teleop = TeleopIphone(
        wbc=wbc,
        host=args.host,
        port=args.port,
        save_trajectory=args.save,
        use_portrait_mode=args.portrait,
    )
    if not teleop.initialize():
        raise RuntimeError("Teleoperation can not be initialized!")
    teleop.start()

    gui: Optional[RBY1WBCTeleopIphone] = None
    try:
        gui = RBY1WBCTeleopIphone(wbc=wbc, teleop=teleop, headless=args.headless)
        gui.run()
    finally:
        if gui is not None:
            gui.close()
        wbc.stop()


if __name__ == "__main__":
    main()
