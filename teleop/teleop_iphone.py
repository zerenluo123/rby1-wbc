"""iPhone pose streaming interface compatible with the WBC teleop API."""

from __future__ import annotations

import base64
import copy
import logging
import struct
import threading
import time
from typing import Dict, Optional, Tuple

import mujoco
import numpy as np
from flask import Flask
from flask_socketio import SocketIO
from scipy.spatial.transform import Rotation as R

from .teleop_vr import TeleopTargets, TeleopLogger


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)-8s - %(message)s"
)


ARKIT_TCP_ROT = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float64,
)


def _rpy_transform(roll: float, pitch: float, yaw: float, translation) -> np.ndarray:
    rot = R.from_euler("xyz", [roll, pitch, yaw]).as_matrix()
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rot
    transform[:3, 3] = np.asarray(translation, dtype=np.float64)
    return transform


def _invert_transform(T: np.ndarray) -> np.ndarray:
    inv = np.eye(4, dtype=np.float64)
    inv[:3, :3] = T[:3, :3].T
    inv[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return inv


TCP_TO_MODEL_FRAME: Dict[str, np.ndarray] = {
    "head": _invert_transform(
        _rpy_transform(
            roll=-np.pi / 2,
            pitch=0.0,
            yaw=-np.pi / 2,
            translation=[0.04, 0.0, 0.0601],
        )
    ),
    "left": _invert_transform(
        _rpy_transform(
            roll=np.pi,
            pitch=0.0,
            yaw=0.0,
            translation=[0.0, 0.0, -0.2],
        )
    ),
    "right": _invert_transform(
        _rpy_transform(
            roll=0.0,
            pitch=np.pi,
            yaw=0.0,
            translation=[0.0, 0.0, -0.2],
        )
    ),
}


def _iphone_tcp_transform(apply_offset: bool) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = ARKIT_TCP_ROT
    if apply_offset:
        transform[:3, 3] = np.array([0.0, -0.083, -0.231], dtype=np.float64)
    return transform


IPHONE_TCP_WITH_OFFSET = _iphone_tcp_transform(apply_offset=True)
IPHONE_TCP_WITHOUT_OFFSET = _iphone_tcp_transform(apply_offset=False)


def _matrix_to_pose(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos = transform[:3, 3].astype(np.float64)
    quat_xyzw = R.from_matrix(transform[:3, :3]).as_quat()
    quat_wxyz = np.array(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64
    )
    return pos, quat_wxyz


def _controller_entry_from_pose(transform: np.ndarray) -> dict:
    pos, quat = _matrix_to_pose(transform)
    return {
        "position": pos.tolist(),
        "rotation": quat.tolist(),
    }


class TeleopIphone:
    """Socket.IO server that feeds ARKit poses into the WBC teleop interface."""

    def __init__(
        self,
        wbc,
        host: str = "0.0.0.0",
        port: int = 5555,
        save_trajectory: bool = False,
    ):
        self.wbc = wbc
        self.host = host
        self.port = port

        self._app = Flask(__name__)
        self._socketio = SocketIO(
            self._app,
            async_mode="threading",
            cors_allowed_origins="*",
            logger=False,
            engineio_logger=False,
        )

        self._register_handlers()

        self._pose_lock = threading.Lock()
        self._latest_poses: Dict[str, Tuple[float, np.ndarray]] = {}
        self._alignment: Dict[str, np.ndarray] = {}
        self._controller_state = {"hands": {}}

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._mujoco_model = mujoco.MjModel.from_xml_path(self.wbc.model_path)
        self._mujoco_data = mujoco.MjData(self._mujoco_model)
        mujoco.mj_forward(self._mujoco_model, self._mujoco_data)
        self._mujoco_lock = threading.Lock()
        self._site_ids = {
            "left": self._mujoco_model.site("end_effector_l").id,
            "right": self._mujoco_model.site("end_effector_r").id,
            "head": self._mujoco_model.site("head").id,
        }

        self._logger = TeleopLogger(save_trajectory=save_trajectory)

    def _register_handlers(self) -> None:
        self._socketio.on_event("connect", self._on_connect)
        self._socketio.on_event("disconnect", self._on_disconnect)
        self._socketio.on_event("updateLeft", self._make_pose_handler("left"))
        self._socketio.on_event("updateRight", self._make_pose_handler("right"))
        self._socketio.on_event("updateHead", self._make_pose_handler("head"))

    def initialize(self) -> bool:
        logging.info("Ready to accept iPhone poses on %s:%d", self.host, self.port)
        return True

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_server, name="iphone-teleop-server", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        try:
            self._socketio.stop()
        except Exception:
            pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self._logger.close()

    def _run_server(self) -> None:
        logging.info("Starting iPhone teleop server on %s:%d", self.host, self.port)
        try:
            self._socketio.run(
                self._app,
                host=self.host,
                port=self.port,
                allow_unsafe_werkzeug=True,
            )
        except Exception as exc:  # pragma: no cover - best effort logging
            if not self._stop_event.is_set():
                logging.error("Socket server error: %s", exc)

    def _on_connect(self, auth=None) -> None:
        logging.info("iPhone client connected")

    def _on_disconnect(self) -> None:
        logging.info("iPhone client disconnected")

    def _make_pose_handler(self, label: str):
        def handler(data: str) -> None:
            self._handle_pose_update(label, data)

        return handler

    def _handle_pose_update(self, label: str, payload: str) -> None:
        decoded = self._decode_pose_payload(payload)
        if decoded is None:
            logging.warning("Failed to decode %s pose payload", label)
            return

        timestamp, pose_matrix = decoded
        apply_offset = label != "head"
        tcp_pose = self._iphone_to_tcp_pose(pose_matrix, apply_offset)
        final_pose = tcp_pose @ TCP_TO_MODEL_FRAME[label]

        with self._pose_lock:
            if label not in self._alignment:
                alignment = self._estimate_alignment(label, final_pose)
                if alignment is not None:
                    self._alignment[label] = alignment
                else:
                    logging.debug("Alignment for %s pending robot state", label)
                    return

            self._latest_poses[label] = (timestamp, final_pose)
            if label in ("left", "right"):
                self._controller_state.setdefault("hands", {})[label] = _controller_entry_from_pose(final_pose)
            else:
                self._controller_state["head"] = _controller_entry_from_pose(final_pose)
            controller_snapshot = self._serialize_controller_state_locked()

        if controller_snapshot.get("hands") or controller_snapshot.get("head"):
            self._logger.log_controller_state(controller_snapshot)

    def _serialize_controller_state_locked(self) -> dict:
        state = {"hands": {}}
        hands = self._controller_state.get("hands", {})
        for side, entry in hands.items():
            if entry:
                state["hands"][side] = copy.deepcopy(entry)
        head_entry = self._controller_state.get("head")
        if head_entry:
            state["head"] = copy.deepcopy(head_entry)
        return state

    def _iphone_to_tcp_pose(self, pose: np.ndarray, apply_offset: bool) -> np.ndarray:
        transform = pose @ (
            IPHONE_TCP_WITH_OFFSET if apply_offset else IPHONE_TCP_WITHOUT_OFFSET
        )
        return transform

    @staticmethod
    def _decode_pose_payload(payload: str) -> Optional[Tuple[float, np.ndarray]]:
        try:
            raw = base64.b64decode(payload)
            if len(raw) < 72:
                return None
            matrix = struct.unpack("<16f", raw[:64])
            timestamp = struct.unpack("<d", raw[64:72])[0]
            pose = np.array(matrix, dtype=np.float64).reshape(4, 4).T
            return timestamp, pose
        except Exception:
            return None

    def _estimate_alignment(self, label: str, pose: np.ndarray) -> Optional[np.ndarray]:
        robot_pose = self._current_site_pose(label)
        if robot_pose is None:
            return None
        return robot_pose @ np.linalg.inv(pose)

    def _current_site_pose(self, label: str) -> Optional[np.ndarray]:
        snapshot = self.wbc.get_latest_robot_state()
        if snapshot is None or not snapshot.is_valid:
            return None
        qpos = self.wbc.snapshot_to_qpos(snapshot)
        if qpos is None:
            return None
        with self._mujoco_lock:
            self._mujoco_data.qpos[:] = qpos
            mujoco.mj_forward(self._mujoco_model, self._mujoco_data)
            site_id = self._site_ids[label]
            return self._site_transform(site_id)

    def _site_transform(self, site_id: int) -> np.ndarray:
        pos = self._mujoco_data.site_xpos[site_id].copy()
        mat = self._mujoco_data.site_xmat[site_id].copy().reshape(3, 3)
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = mat
        transform[:3, 3] = pos
        return transform

    def compute_target(self) -> Optional[TeleopTargets]:
        with self._pose_lock:
            ready = all(
                label in self._latest_poses and label in self._alignment
                for label in ("left", "right")
            )
            latest = {k: (ts, pose.copy()) for k, (ts, pose) in self._latest_poses.items()}
            alignment = {k: mat.copy() for k, mat in self._alignment.items()}

        if not ready:
            return None

        left_transform = alignment["left"] @ latest["left"][1]
        right_transform = alignment["right"] @ latest["right"][1]

        head_transform = None
        if "head" in latest and "head" in alignment:
            head_transform = alignment["head"] @ latest["head"][1]
        else:
            head_transform = self._current_site_pose("head")

        left_pos, left_quat = _matrix_to_pose(left_transform)
        right_pos, right_quat = _matrix_to_pose(right_transform)
        head_pos = head_quat = None
        if head_transform is not None:
            head_pos, head_quat = _matrix_to_pose(head_transform)

        targets = TeleopTargets(
            left_pos=left_pos,
            left_quat=left_quat,
            right_pos=right_pos,
            right_quat=right_quat,
            left_width=None,
            right_width=None,
            head_pos=head_pos,
            head_quat=head_quat,
        )
        self._logger.log_target(targets, timestamp=time.time())
        return targets
