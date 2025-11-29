"""GUI frontend that drives targets for the standalone RBY1 WBC thread."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path regardless of current working directory.
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rby1.whole_body_control import RBY1WBC
from rby1.ee_targets import EETargets
from rby1.rby1_wbc_app import WBCStreamingApp

class RBY1WBCGui(WBCStreamingApp):
    def __init__(self, wbc: RBY1WBC):
        self.mocap_ids = None
        super().__init__(
            wbc=wbc,
            headless=False,
            model_path=PROJECT_ROOT + "/model/rby1/rby1_mocap.xml",
        )
        if self.visualizer is not None:
            self.mocap_ids = self.visualizer.init_mocap_targets()

    def get_target(self) -> EETargets | None:
        if self.visualizer is None:
            return None
        (
            left_pos,
            left_quat,
            right_pos,
            right_quat,
            head_pos,
            head_quat,
        ) = self.visualizer.get_mocap_targets(self.mocap_ids)
        
        return EETargets(
            left_pos=left_pos,
            left_quat=left_quat,
            right_pos=right_pos,
            right_quat=right_quat,
            left_width=None,
            right_width=None,
            head_pos=head_pos,
            head_quat=head_quat,
            duration=self.trajectory_rate.dt,
            timestamp=time.monotonic(),
        )

def main() -> None:
    parser = argparse.ArgumentParser(description="RBY1 whole-body IK GUI decoupled from WBC thread")
    args = parser.parse_args()

    wbc = RBY1WBC()
    wbc.start()
    gui = None
    try:
        gui = RBY1WBCGui(wbc=wbc)
        gui.run()
    finally:
        if gui is not None:
            gui.close()
        wbc.stop()

if __name__ == "__main__":
    main()
