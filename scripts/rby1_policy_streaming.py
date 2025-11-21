"""Streaming bridge between the policy server and the realtime controller."""

from __future__ import annotations

import argparse
import heapq
import itertools
import threading
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import zmq

from control.camera_stream import AravisCameraStreamer
from control.policy_robot import RBY1PolicyRobot


CAMERA_SERIAL_TO_KEY = {
    "BFS_25037059": "camera_head_ultrawide_rgb",
    "BFS_25037058": "camera_head_main_rgb",
    "BFS_25037070": "camera_head_main_right_rgb",
    "BFS_24017452": "camera_left_main_rgb",
    "BFS_24293899": "camera_right_main_rgb",
}


@dataclass
class ScheduledAction:
    """Action scheduled for execution at a given timestamp."""

    timestamp: float
    duration: float
    payload: Dict[str, np.ndarray]


class ActionScheduler:
    """Thread-safe priority queue that orders actions by timestamp."""

    def __init__(self) -> None:
        self._heap: List[Tuple[float, int, ScheduledAction]] = []
        self._counter = itertools.count()
        self._condition = threading.Condition()

    def add_actions(self, actions: Iterable[ScheduledAction]) -> None:
        with self._condition:
            for action in actions:
                if not np.isfinite(action.timestamp):
                    continue
                entry = (float(action.timestamp), next(self._counter), action)
                heapq.heappush(self._heap, entry)
            if self._heap:
                self._condition.notify_all()

    def pop_ready(self, now: float) -> List[ScheduledAction]:
        ready: List[ScheduledAction] = []
        with self._condition:
            while self._heap and self._heap[0][0] <= now:
                _, _, action = heapq.heappop(self._heap)
                ready.append(action)
        return ready

    def peek_timestamp(self) -> Optional[float]:
        with self._condition:
            return float(self._heap[0][0]) if self._heap else None

    def wait(self, timeout: Optional[float]) -> None:
        with self._condition:
            self._condition.wait(timeout=timeout)


class PolicyClient:
    """Thin ZMQ wrapper that talks to the detached policy server."""

    def __init__(self, ip: str, port: int, timeout: float = 2.0) -> None:
        self._ctx = zmq.Context.instance()
        self._socket = self._ctx.socket(zmq.REQ)
        self._socket.connect(f"tcp://{ip}:{port}")
        self._socket.setsockopt(zmq.RCVTIMEO, int(max(timeout, 0.1) * 1000))
        self._socket.setsockopt(zmq.SNDTIMEO, int(max(timeout, 0.1) * 1000))

    def close(self) -> None:
        self._socket.close(0)

    def request_action_indexing(self) -> Dict[str, Tuple[int, int]]:
        while True:
            try:
                self._socket.send_string("get_action_indexing")
                reply = self._socket.recv_pyobj()
            except zmq.Again:
                continue
            if isinstance(reply, dict):
                return reply
            time.sleep(0.5)

    def infer(self, obs: Dict[str, np.ndarray]) -> Optional[Dict[str, np.ndarray]]:
        try:
            self._socket.send_pyobj(obs)
            return self._socket.recv_pyobj()
        except zmq.Again:
            return None


def decode_action_vector(
    action_vec: np.ndarray,
    indexing: Dict[str, Tuple[int, int]],
) -> Dict[str, np.ndarray]:
    """Convert a flat action vector to structured pose commands."""

    action_vec = np.asarray(action_vec, dtype=float)

    def _slice(name: str) -> np.ndarray:
        if name not in indexing:
            raise KeyError(f"Action indexing missing key '{name}'")
        start, end = indexing[name]
        return action_vec[start:end]

    action: Dict[str, np.ndarray] = {
        "left_pos": _slice("gripper_left_eef_pos"),
        "left_rot_axis_angle": _slice("gripper_left_eef_rot_axis_angle"),
        "right_pos": _slice("gripper_right_eef_pos"),
        "right_rot_axis_angle": _slice("gripper_right_eef_rot_axis_angle"),
    }
    if "head_eef_pos" in indexing and "head_eef_rot_axis_angle" in indexing:
        action["head_pos"] = _slice("head_eef_pos")
        action["head_rot_axis_angle"] = _slice("head_eef_rot_axis_angle")
    if "gripper_left_gripper_width" in indexing:
        action["left_gripper_width"] = float(_slice("gripper_left_gripper_width")[0])
    if "gripper_right_gripper_width" in indexing:
        action["right_gripper_width"] = float(_slice("gripper_right_gripper_width")[0])
    return action


def build_scheduled_actions(
    actions: np.ndarray,
    timestamps: np.ndarray,
    indexing: Dict[str, Tuple[int, int]],
    fallback_dt: float,
    now: float,
) -> List[ScheduledAction]:
    actions = np.asarray(actions, dtype=float)
    if actions.ndim == 1:
        actions = actions[None, :]

    scheduled: List[ScheduledAction] = []
    if actions.size == 0:
        return scheduled

    if timestamps.size == 0:
        timestamps = now + fallback_dt * (np.arange(len(actions), dtype=float) + 1.0)

    timestamps = np.asarray(timestamps, dtype=float)
    for idx, vector in enumerate(actions):
        try:
            payload = decode_action_vector(vector, indexing)
        except KeyError as exc:
            print(f"[policy] {exc}")
            continue

        timestamp = float(timestamps[min(idx, len(timestamps) - 1)])
        if not np.isfinite(timestamp):
            timestamp = now + fallback_dt * (idx + 1)
        if timestamp <= now:
            # Drop commands that are already stale.
            continue

        if idx + 1 < len(timestamps):
            duration = float(max(fallback_dt, timestamps[idx + 1] - timestamp))
        else:
            duration = float(fallback_dt)

        scheduled.append(ScheduledAction(timestamp=timestamp, duration=duration, payload=payload))

    return scheduled


def main() -> None:
    parser = argparse.ArgumentParser(description="RBY1 policy streaming bridge")
    parser.add_argument("--policy-ip", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=8766)
    parser.add_argument("--wbc-config", default=None, help="Optional path to WBC YAML config")
    parser.add_argument("--robot-horizon", type=int, default=2)
    parser.add_argument("--robot-stride", type=int, default=1)
    parser.add_argument("--camera-horizon", type=int, default=2)
    parser.add_argument("--camera-stride", type=int, default=1)
    parser.add_argument("--state-only", action="store_true", help="Skip camera streaming")
    parser.add_argument("--mock-cameras", action="store_true", help="Generate synthetic camera images")
    parser.add_argument("--control-dt", type=float, default=0.1)
    parser.add_argument("--policy-timeout", type=float, default=2.0, help="ZMQ request timeout in seconds")
    parser.add_argument("--executor-lookahead", type=float, default=0.02, help="Execution loop lookahead in seconds")
    args = parser.parse_args()

    control_dt = max(args.control_dt, 1e-3)
    robot = RBY1PolicyRobot(config_path=args.wbc_config)
    robot.start()
    camera_streamer = None
    stop_event = threading.Event()
    scheduler = ActionScheduler()
    threads: List[threading.Thread] = []
    policy_client: Optional[PolicyClient] = None

    try:
        robot.wait_until_ready()
        required_robot_samples = max(args.robot_horizon * args.robot_stride, 1)
        if not robot.wait_for_observations(required_robot_samples, timeout=2.0):
            raise TimeoutError("Timed out waiting for initial robot observations")

        if not args.state_only:
            camera_streamer = AravisCameraStreamer(
                CAMERA_SERIAL_TO_KEY,
                buffer_size=max(args.camera_horizon * args.camera_stride, 4),
                mock_mode=args.mock_cameras,
            )
            camera_streamer.start()
            try:
                camera_streamer.wait_until_ready(
                    min_frames=max(args.camera_horizon * args.camera_stride, 1),
                    timeout=2.0,
                )
            except TimeoutError as exc:
                print(f"[camera] {exc}")

        policy_client = PolicyClient(
            ip=args.policy_ip,
            port=args.policy_port,
            timeout=args.policy_timeout,
        )
        action_indexing = policy_client.request_action_indexing()

        def inference_worker() -> None:
            while not stop_event.is_set():
                start = time.monotonic()
                try:
                    robot_obs = robot.get_observation_window(
                        horizon=args.robot_horizon,
                        stride=args.robot_stride,
                    )
                except Exception as exc:  # pragma: no cover - runtime safeguard
                    print(f"[robot] Failed to gather observation: {exc}")
                    if stop_event.wait(timeout=control_dt):
                        break
                    continue

                obs_dict = {k: v for k, v in robot_obs.items() if k != "timestamp"}
                timestamps = robot_obs["timestamp"]

                if camera_streamer is not None:
                    try:
                        camera_obs = camera_streamer.get_observation_window(
                            horizon=args.camera_horizon,
                            stride=args.camera_stride,
                        )
                        obs_dict.update(camera_obs)
                    except RuntimeError as exc:
                        print(f"[camera] {exc}")

                obs_dict["timestamp"] = timestamps

                reply = policy_client.infer(obs_dict)
                if reply is None or "actions" not in reply:
                    print("[policy] Inference timeout or malformed reply")
                    if stop_event.wait(timeout=control_dt):
                        break
                    continue

                actions = np.asarray(reply.get("actions"))
                action_timestamps = np.asarray(reply.get("timestamps", []), dtype=float)

                scheduled_actions = build_scheduled_actions(
                    actions=actions,
                    timestamps=action_timestamps,
                    indexing=action_indexing,
                    fallback_dt=control_dt,
                    now=time.monotonic(),
                )
                if scheduled_actions:
                    scheduler.add_actions(scheduled_actions)

                elapsed = time.monotonic() - start
                wait_time = max(0.0, control_dt - elapsed)
                if stop_event.wait(timeout=wait_time):
                    break

        def executor_worker() -> None:
            lookahead = max(args.executor_lookahead, 0.0)
            while not stop_event.is_set():
                now = time.monotonic() + lookahead
                ready = scheduler.pop_ready(now)
                if ready:
                    for scheduled_action in ready:
                        robot.apply_action(
                            scheduled_action.payload,
                            duration=scheduled_action.duration,
                            timestamp=scheduled_action.timestamp,
                        )
                    continue

                next_ts = scheduler.peek_timestamp()
                if next_ts is None:
                    if stop_event.wait(timeout=control_dt):
                        break
                else:
                    wait_time = max(0.0, next_ts - (time.monotonic() + lookahead))
                    if stop_event.wait(timeout=wait_time):
                        break

        threads.append(threading.Thread(target=inference_worker, name="policy-inference", daemon=True))
        threads.append(threading.Thread(target=executor_worker, name="policy-executor", daemon=True))

        for thread in threads:
            thread.start()

        while all(thread.is_alive() for thread in threads):
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("[main] Interrupted, shutting down...")
    finally:
        stop_event.set()
        for thread in threads:
            thread.join(timeout=1.0)
        if camera_streamer is not None:
            camera_streamer.stop()
        if policy_client is not None:
            policy_client.close()
        robot.stop()


if __name__ == "__main__":
    main()
