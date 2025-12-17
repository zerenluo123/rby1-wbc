import os
import sys
from pathlib import Path
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR.parent))
import time

from gripper_client import RemoteGripper

gripper = RemoteGripper(host="192.168.30.2", port=5678)
# print(gripper.min_q)
# print(gripper.max_q)
gripper.set_target([0.07, 0.07])
print("Setting target to 0.07")
time.sleep(1)
gripper.set_target([0.01, 0.01])
print("Setting target to 0.01")
