import copy
import json
import logging
import pickle
import socket
import threading
import time
import subprocess
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from .vr_control_state import VRControlState


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)-8s - %(message)s"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "log"
DEMO_DIR = PROJECT_ROOT / "demo"

T_CONV = np.array(
    [
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=float,
)


LOCAL_PORT = 5005
META_QUEST_PORT = 6000


class TeleopLogger:
    """Thread-safe logger for controller samples and optional teleop trajectory dumps."""

    def __init__(self, save_trajectory: bool) -> None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        DEMO_DIR.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.session_name = f"teleop_{stamp}"
        self._controller_path = LOG_DIR / f"{self.session_name}.jsonl"
        self._timeout_path = LOG_DIR / f"{self.session_name}_timeouts.jsonl"
        self._trajectory_path = (
            DEMO_DIR / f"{self.session_name}.pkl" if save_trajectory else None
        )
        self._save_trajectory = save_trajectory
        self._closed = False
        self._lock = threading.Lock()
        self._controller_samples: list[dict] = []
        self._socket_timeouts: list[float] = []
        self._target_timestamps: list[float] = []
        self._target_data: dict[str, dict[str, list]] = {}

    @staticmethod
    def _sanitize_pose_entry(hand: dict) -> dict:
        return {
            "position": [float(x) for x in hand.get("position", [])],
            "rotation": [float(x) for x in hand.get("rotation", [])],
        }

    @staticmethod
    def _quat_wxyz_to_rotvec(quat: np.ndarray) -> np.ndarray:
        quat = np.asarray(quat, dtype=float)
        if quat.shape[-1] != 4 or np.linalg.norm(quat) < 1e-9:
            return np.zeros(3, dtype=float)
        quat_xyzw = np.array([quat[1], quat[2], quat[3], quat[0]], dtype=float)
        return R.from_quat(quat_xyzw).as_rotvec()

    def log_controller_state(self, controller_state: dict) -> None:
        timestamp = float(time.time())
        entry: dict[str, object] = {"timestamp": timestamp}
        hands = controller_state.get("hands", {})
        left = hands.get("left")
        right = hands.get("right")
        head = controller_state.get("head")
        if left:
            entry["left"] = self._sanitize_pose_entry(left)
        if right:
            entry["right"] = self._sanitize_pose_entry(right)
        if head:
            entry["head"] = self._sanitize_pose_entry(head)
        with self._lock:
            self._controller_samples.append(entry)

    def log_socket_timeout(self) -> None:
        with self._lock:
            self._socket_timeouts.append(time.time())

    def log_target(self, target: "TeleopTargets", timestamp: float) -> None:
        with self._lock:
            self._target_timestamps.append(timestamp)
            self._append_target("left", target.left_pos, target.left_quat, target.left_width)
            self._append_target("right", target.right_pos, target.right_quat, target.right_width)
            self._append_target("head", target.head_pos, target.head_quat, None)

    def _append_target(
        self,
        side: str,
        position: Optional[np.ndarray],
        quaternion_wxyz: Optional[np.ndarray],
        gripper_width: Optional[float],
    ) -> None:
        if position is None or quaternion_wxyz is None:
            return
        data = self._target_data.setdefault(side, {"tcp_pose": [], "gripper_width": []})
        rotvec = self._quat_wxyz_to_rotvec(quaternion_wxyz)
        pose_sample = list(np.asarray(position, dtype=float)) + list(rotvec)
        data["tcp_pose"].append(pose_sample)
        width_value = 0.0 if gripper_width is None else float(gripper_width)
        data["gripper_width"].append(width_value)

    def close(self) -> None:
        if self._closed:
            return
        with self._lock:
            controller_samples = list(self._controller_samples)
            socket_timeouts = list(self._socket_timeouts)
            target_timestamps = list(self._target_timestamps)
            target_data = {k: {kk: list(vv) for kk, vv in v.items()} for k, v in self._target_data.items()}

        timeout_entries = [{"timestamp": ts} for ts in socket_timeouts]
        self._write_jsonl(self._controller_path, controller_samples)
        self._write_jsonl(self._timeout_path, timeout_entries)
        self._write_targets_pickle(target_data, target_timestamps)
        self._closed = True

    def _write_jsonl(self, path: Path, entries: list[dict]) -> None:
        if not entries:
            return
        with path.open("w", encoding="utf-8") as fh:
            for entry in entries:
                fh.write(json.dumps(entry))
                fh.write("\n")

    def _write_targets_pickle(self, target_data: dict, target_timestamps: list[float]) -> None:
        if (
            not self._save_trajectory
            or not target_timestamps
            or self._trajectory_path is None
        ):
            return

        def build_gripper_payload(side: str) -> list[dict]:
            data = target_data.get(side)
            if not data or not data["tcp_pose"]:
                return []
            tcp_pose = np.asarray(data["tcp_pose"], dtype=np.float32)
            gripper_width = np.asarray(data["gripper_width"], dtype=np.float32)
            demo_start = np.repeat(tcp_pose[[0]], tcp_pose.shape[0], axis=0)
            demo_end = np.repeat(tcp_pose[[-1]], tcp_pose.shape[0], axis=0)
            return [
                {
                    "tcp_pose": tcp_pose,
                    "gripper_width": gripper_width,
                    "demo_start_pose": demo_start,
                    "demo_end_pose": demo_end,
                }
            ]

        episode = {
            "tasks": [],
            "episode_name": self.session_name,
            "grippers_left": build_gripper_payload("left"),
            "grippers_right": build_gripper_payload("right"),
            "grippers_head": build_gripper_payload("head"),
            "cameras_left": [],
            "cameras_right": [],
            "cameras_head": [],
            "target_timestamps": np.asarray(target_timestamps, dtype=np.float64),
        }

        with self._trajectory_path.open("wb") as fh:
            pickle.dump([episode], fh)

@dataclass
class TeleopTargets:
    left_pos: np.ndarray
    left_quat: np.ndarray
    right_pos: np.ndarray
    right_quat: np.ndarray
    left_width: Optional[float]
    right_width: Optional[float]
    head_pos: Optional[np.ndarray]
    head_quat: Optional[np.ndarray]


def _pose_to_matrix(position, rotation_quat):
    T = np.eye(4, dtype=float)
    T[:3, :3] = R.from_quat(rotation_quat).as_matrix()
    T[:3, 3] = np.asarray(position, dtype=float)
    return T


class TeleopVR:
    """Meta Quest streaming interface that converts VR poses into WBC targets."""

    def __init__(
        self,
        wbc,
        local_ip: str,
        meta_quest_ip: str,
        save_trajectory: bool = False,
    ):
        self.wbc = wbc
        self.local_ip = local_ip
        self.meta_quest_ip = meta_quest_ip

        self.vr_state = VRControlState()
        self._controller_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._mujoco_model = mujoco.MjModel.from_xml_path(self.wbc.model_path)
        self._mujoco_data = mujoco.MjData(self._mujoco_model)
        mujoco.mj_forward(self._mujoco_model, self._mujoco_data)
        self._mujoco_lock = threading.Lock()

        self._right_site_id = self._mujoco_model.site("end_effector_r").id
        self._left_site_id = self._mujoco_model.site("end_effector_l").id
        self._head_site_id = self._mujoco_model.site("head").id
        self._torso_body_id = self._mujoco_model.body("link_torso_5").id

        self._has_initial_pose = False
        self._last_left_width = 0.0
        self._last_right_width = 0.0
        self._logger = TeleopLogger(save_trajectory=save_trajectory)

    def initialize(self) -> bool:
        rv = False
        # Ping Meta Quest IP once
        param = "-n" if platform.system().lower() == "windows" else "-c"
        command = ["ping", param, "1", self.meta_quest_ip]

        if subprocess.call(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
            rv = True
            print(f"Meta Quest is connected to the same network: {self.meta_quest_ip}")
        else:
            print(f"Meta Quest is not connected to the same network: {self.meta_quest_ip}")

        if rv:
            payload = {"ip": self.local_ip, "port": LOCAL_PORT}
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                message = json.dumps(payload).encode("utf-8")
                sock.sendto(message, (self.meta_quest_ip, META_QUEST_PORT))
        return rv

    def start(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._teleop_loop, name="meta-quest-udp", daemon=True
            )
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self._logger.close()

    def _teleop_loop(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as server_sock:
            server_sock.bind((self.local_ip, LOCAL_PORT))
            server_sock.settimeout(1.0)
            while not self._stop_event.is_set():
                try:
                    data, _ = server_sock.recvfrom(8192)
                except socket.timeout:
                    self._logger.log_socket_timeout()
                    continue

                udp_msg = data.decode("utf-8")
                controller_state = json.loads(udp_msg)
                self._logger.log_controller_state(controller_state)

                with self._controller_lock:
                    self.vr_state.controller_state = controller_state
                    self._update_event_flags(controller_state)

    def _update_event_flags(self, controller_state: dict) -> None:
        hands = controller_state.get("hands", {})
        left = hands.get("left")
        right = hands.get("right")

        if left:
            buttons = left.get("buttons", {})
            self.vr_state.event_left_a_pressed |= bool(buttons.get("primaryButton"))
            self.vr_state.event_left_b_pressed |= bool(buttons.get("secondaryButton"))

        if right:
            buttons = right.get("buttons", {})
            self.vr_state.event_right_a_pressed |= bool(buttons.get("primaryButton"))
            self.vr_state.event_right_b_pressed |= bool(buttons.get("secondaryButton"))

    def compute_target(self) -> Optional[TeleopTargets]:
        """Compute the next teleoperation target for the WBC."""
        snapshot = self.wbc.get_latest_robot_state()
        if snapshot is None or not snapshot.is_valid:
            return None

        qpos = self.wbc.snapshot_to_qpos(snapshot)
        if qpos is None:
            return None

        with self._mujoco_lock:
            self._mujoco_data.qpos[:] = qpos
            mujoco.mj_forward(self._mujoco_model, self._mujoco_data)
            right_pose = self._site_transform(self._right_site_id)
            left_pose = self._site_transform(self._left_site_id)
            head_pose = self._site_transform(self._head_site_id)
            torso_pose = (
                self._body_transform(self._torso_body_id)
                if self._torso_body_id is not None
                else np.eye(4)
            )

        self.vr_state.joint_positions = qpos.copy()
        self.vr_state.right_ee_current_pose = right_pose
        self.vr_state.left_ee_current_pose = left_pose
        self.vr_state.head_ee_current_pose = head_pose
        self.vr_state.torso_current_pose = torso_pose
        if not self._has_initial_pose:
            self._initialize_locked_poses()

        controller_state = self._get_controller_state()
        if not controller_state:
            return None

        self._update_controller_poses(controller_state)
        self._handle_button_events()
        if not self.vr_state.is_initialized or self.vr_state.is_stopped:
            return None

        left_width = self._update_left_follow_state(controller_state)
        right_width = self._update_right_follow_state(controller_state)
        head_pose_target = self._update_head_follow_state(controller_state)

        left_target_pose = self.vr_state.left_hand_locked_pose
        right_target_pose = self.vr_state.right_hand_locked_pose
        head_target_pose = head_pose_target

        left_pos, left_quat = self._matrix_to_pose(left_target_pose)
        right_pos, right_quat = self._matrix_to_pose(right_target_pose)

        head_pos = head_quat = None
        if head_target_pose is not None:
            head_pos, head_quat = self._matrix_to_pose(head_target_pose)

        targets = TeleopTargets(
            left_pos=left_pos,
            left_quat=left_quat,
            right_pos=right_pos,
            right_quat=right_quat,
            left_width=left_width,
            right_width=right_width,
            head_pos=head_pos,
            head_quat=head_quat,
        )
        self._logger.log_target(targets, timestamp=time.time())
        return targets

    def _initialize_locked_poses(self) -> None:
        self.vr_state.right_hand_locked_pose = self.vr_state.right_ee_current_pose
        self.vr_state.left_hand_locked_pose = self.vr_state.left_ee_current_pose
        self.vr_state.head_locked_pose = self.vr_state.head_ee_current_pose
        self.vr_state.torso_locked_pose = self.vr_state.torso_current_pose
        self._has_initial_pose = True

    def _get_controller_state(self) -> dict:
        with self._controller_lock:
            if not self.vr_state.controller_state:
                return {}
            return copy.deepcopy(self.vr_state.controller_state)

    def _handle_button_events(self) -> None:
        if self.vr_state.event_right_a_pressed:
            logging.info("Right primary button pressed. Initializing teleop.")
            self.vr_state.is_initialized = True
            self.vr_state.is_stopped = False
            self._initialize_locked_poses()

        if self.vr_state.event_right_b_pressed:
            logging.info("Right secondary button pressed. Stopping teleop.")
            self.vr_state.is_stopped = True

        self.vr_state.event_right_a_pressed = False
        self.vr_state.event_right_b_pressed = False
        self.vr_state.event_left_a_pressed = False
        self.vr_state.event_left_b_pressed = False

    def _update_controller_poses(self, controller_state: dict) -> None:
        hands = controller_state.get("hands", {})
        head = controller_state.get("head")

        right = hands.get("right")
        if right:
            right_pose = (
                T_CONV.T
                @ _pose_to_matrix(right["position"], right["rotation"])
                @ T_CONV
            )
            self.vr_state.right_controller_current_pose = right_pose

        left = hands.get("left")
        if left:
            left_pose = (
                T_CONV.T @ _pose_to_matrix(left["position"], left["rotation"]) @ T_CONV
            )
            self.vr_state.left_controller_current_pose = left_pose

        if head:
            head_pose = (
                T_CONV.T @ _pose_to_matrix(head["position"], head["rotation"]) @ T_CONV
            )
            self.vr_state.head_controller_current_pose = head_pose

    def _update_right_follow_state(self, controller_state: dict) -> Optional[float]:
        right = controller_state.get("hands", {}).get("right")
        if not right:
            self.vr_state.is_right_following = False
            return self._last_right_width

        trigger_pressed = right.get("buttons", {}).get("grip", 0.0) > 0.8
        if self.vr_state.is_right_following and not trigger_pressed:
            self.vr_state.is_right_following = False
        if not self.vr_state.is_right_following and trigger_pressed:
            self.vr_state.right_controller_start_pose = (
                self.vr_state.right_controller_current_pose
            )
            self.vr_state.right_ee_start_pose = self.vr_state.right_hand_locked_pose = (
                self.vr_state.right_ee_current_pose
            )
            self.vr_state.is_right_following = True

        if self.vr_state.is_right_following:
            controller_start = self.vr_state.right_controller_start_pose
            controller_current = self.vr_state.right_controller_current_pose
            ee_start = self.vr_state.right_ee_start_pose
            rot_delta = controller_current[:3, :3] @ controller_start[:3, :3].T
            pos_delta = controller_current[:3, 3] - controller_start[:3, 3]
            right_T = ee_start.copy()
            right_T[:3, :3] = rot_delta @ ee_start[:3, :3]
            right_T[:3, 3] = ee_start[:3, 3] + pos_delta
            self.vr_state.right_hand_locked_pose = right_T

        width = (
            1 - float(right.get("buttons", {}).get("trigger", self._last_right_width))
        ) * 0.1
        self._last_right_width = width
        return width

    def _update_left_follow_state(self, controller_state: dict) -> Optional[float]:
        left = controller_state.get("hands", {}).get("left")
        if not left:
            self.vr_state.is_left_following = False
            return self._last_left_width

        trigger_pressed = left.get("buttons", {}).get("grip", 0.0) > 0.8
        if self.vr_state.is_left_following and not trigger_pressed:
            self.vr_state.is_left_following = False
        if not self.vr_state.is_left_following and trigger_pressed:
            self.vr_state.left_controller_start_pose = (
                self.vr_state.left_controller_current_pose
            )
            self.vr_state.left_ee_start_pose = self.vr_state.left_hand_locked_pose = (
                self.vr_state.left_ee_current_pose
            )
            self.vr_state.is_left_following = True

        if self.vr_state.is_left_following:
            controller_start = self.vr_state.left_controller_start_pose
            controller_current = self.vr_state.left_controller_current_pose
            ee_start = self.vr_state.left_ee_start_pose
            rot_delta = controller_current[:3, :3] @ controller_start[:3, :3].T
            pos_delta = controller_current[:3, 3] - controller_start[:3, 3]
            left_T = ee_start.copy()
            left_T[:3, :3] = rot_delta @ ee_start[:3, :3]
            left_T[:3, 3] = ee_start[:3, 3] + pos_delta
            self.vr_state.left_hand_locked_pose = left_T

        width = (
            1 - left.get("buttons", {}).get("trigger", self._last_left_width)
        ) * 0.1
        self._last_left_width = width
        return width

    def _update_head_follow_state(self, controller_state: dict) -> Optional[np.ndarray]:
        head = controller_state.get("head")
        following = self.vr_state.is_right_following and self.vr_state.is_left_following

        if self.vr_state.is_torso_following and not following:
            self.vr_state.is_torso_following = False

        if not self.vr_state.is_torso_following and following:
            self.vr_state.is_torso_following = True
            self.vr_state.head_controller_start_pose = (
                self.vr_state.head_controller_current_pose
            )
            self.vr_state.head_ee_start_pose = self.vr_state.head_locked_pose = (
                self.vr_state.head_ee_current_pose
            )

        if head and self.vr_state.is_torso_following:
            controller_start = self.vr_state.head_controller_start_pose
            controller_current = self.vr_state.head_controller_current_pose
            ee_start = self.vr_state.head_ee_start_pose
            rot_delta = controller_current[:3, :3] @ controller_start[:3, :3].T
            pos_delta = controller_current[:3, 3] - controller_start[:3, 3]
            head_T = ee_start.copy()
            head_T[:3, :3] = rot_delta @ ee_start[:3, :3]
            head_T[:3, 3] = ee_start[:3, 3] + pos_delta
            
            self.vr_state.head_locked_pose = head_T

        return self.vr_state.head_locked_pose

    def _site_transform(self, site_id: int) -> np.ndarray:
        pos = self._mujoco_data.site_xpos[site_id].copy()
        mat = self._mujoco_data.site_xmat[site_id].copy().reshape(3, 3)
        T = np.eye(4, dtype=float)
        T[:3, :3] = mat
        T[:3, 3] = pos
        return T

    def _body_transform(self, body_id: Optional[int]) -> np.ndarray:
        if body_id is None:
            return np.eye(4, dtype=float)
        pos = self._mujoco_data.xpos[body_id].copy()
        mat = self._mujoco_data.xmat[body_id].copy().reshape(3, 3)
        T = np.eye(4, dtype=float)
        T[:3, :3] = mat
        T[:3, 3] = pos
        return T

    @staticmethod
    def _matrix_to_pose(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pos = transform[:3, 3].astype(float)
        quat_xyzw = R.from_matrix(transform[:3, :3]).as_quat()
        quat_wxyz = np.array(
            [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=float
        )
        return pos, quat_wxyz
