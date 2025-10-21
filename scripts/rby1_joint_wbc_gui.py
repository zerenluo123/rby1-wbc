from pathlib import Path
from queue import SimpleQueue
import sys
import time
import math
import numpy as np
import mujoco
import mujoco.viewer

from loop_rate_limiters import RateLimiter
import os
# Ensure project root is on sys.path regardless of current working directory
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from ik.rby1_whole_body_ik import RBY1WholeBodyIK
import rby1_sdk
from rby1_sdk import *

import threading
from dataclasses import dataclass
import os

class RobotState:
    def __init__(self):
        self.q = SimpleQueue()
        self.latest = None

    def store(self, new_state):
        self.q.put_nowait(new_state)
        if self.q.qsize() > 1:
            self.q.get_nowait()

    def load(self):
        if not self.q.empty():
            self.latest = self.q.get_nowait()
        return self.latest

@dataclass
class SimpleRobotState:
    position: np.ndarray  # (26,)
    odom_T: np.ndarray    # (3,3) SE(2) matrix: [[R t],[0 0 1]]

@dataclass
class SharedTargets:
    """Thread-safe shared targets and current qpos snapshot for IK."""
    _lock: threading.Lock
    left_target_pos: np.ndarray | None = None   # shape (3,)
    left_target_quat: np.ndarray | None = None  # shape (4,) (w,x,y,z)
    right_target_pos: np.ndarray | None = None  # shape (3,)
    right_target_quat: np.ndarray | None = None # shape (4,)
    current_qpos: np.ndarray | None = None  # shape (nq,)

    def set_from_viewer(self, left_pos, left_quat, right_pos, right_quat, qpos):
        with self._lock:
            self.left_target_pos = np.asarray(left_pos).copy()
            self.left_target_quat = np.asarray(left_quat).copy()
            self.right_target_pos = np.asarray(right_pos).copy()
            self.right_target_quat = np.asarray(right_quat).copy()
            self.current_qpos = np.asarray(qpos).copy()

    def get_for_ik(self):
        with self._lock:
            lt_p = None if self.left_target_pos is None else self.left_target_pos.copy()
            lt_q = None if self.left_target_quat is None else self.left_target_quat.copy()
            rt_p = None if self.right_target_pos is None else self.right_target_pos.copy()
            rt_q = None if self.right_target_quat is None else self.right_target_quat.copy()
            q  = None if self.current_qpos is None else self.current_qpos.copy()
        return lt_p, lt_q, rt_p, rt_q, q

class SimGUI:
    def __init__(self, model_path: str, address: str ='localhost:50051', model_name="m"):
        # Robot Controller & State 
        self.robot_state = RobotState()
        self.robot_state_frequency = 200 
        self.robot  = self.init_robot(address, model_name)
        self.stream = self.robot.create_command_stream(self.robot_state_frequency)

        # Robot Viewer & Rate
        self.viewer = self.init_viewer(model_path)
        self.ik_rate = RateLimiter(frequency=200.0, warn=False)
        self.viewer_rate = RateLimiter(frequency=60.0, warn=False)
        
        # Shared targets between viewer(main) and controller(thread)
        self.shared = SharedTargets(_lock=threading.Lock())

        # threading control
        self._stop = threading.Event()
        self._controller_thread = None


    def init_robot(self, address, model_name="m", power=".*", servo=".*"):
        print("Attempting to connect to the robot...")
        robot = rby1_sdk.create_robot(address, model_name)
        if not robot.connect():
            print("Error: Unable to establish connection to the robot at")
            sys.exit(1)
        print("Successfully connected to the robot")

        print("Starting state update...")
        robot.start_state_update(self.robot_state_callback, self.robot_state_frequency)

        # robot.factory_reset_all_parameters()
        # robot.set_parameter("default.acceleration_limit_scaling", "1.0")
        # robot.set_parameter("joint_position_command.cutoff_frequency", "5")
        # robot.set_parameter("cartesian_command.cutoff_frequency", "5")
        # robot.set_parameter("default.linear_acceleration_limit", "20")
        # robot.set_parameter("default.angular_acceleration_limit", "10")
        # robot.set_parameter("manipulability_threshold", "1e4")
        # robot.set_time_scale(1.0)

        print("parameters setting is done")

        if not robot.is_connected():
            print("Robot is not connected")
            exit(1)

        if not robot.is_power_on(power):
            rv = robot.power_on(power)
            if not rv:
                print("Failed to power on")
                exit(1)

        print(servo)
        if not robot.is_servo_on(servo):
            rv = robot.servo_on(servo)
            if not rv:
                print("Fail to servo on")
                exit(1)

        control_manager_state = robot.get_control_manager_state()

        if (
            control_manager_state.state == rby1_sdk.ControlManagerState.State.MinorFault
            or control_manager_state.state == rby1_sdk.ControlManagerState.State.MajorFault
        ):

            if control_manager_state.state == rby1_sdk.ControlManagerState.State.MajorFault:
                print(
                    "Warning: Detected a Major Fault in the Control Manager!!!!!!!!!!!!!!!."
                )
            else:
                print(
                    "Warning: Detected a Minor Fault in the Control Manager@@@@@@@@@@@@@@@@."
                )

            print("Attempting to reset the fault...")
            if not robot.reset_fault_control_manager():
                print("Error: Unable to reset the fault in the Control Manager.")
                sys.exit(1)
            print("Fault reset successfully.")

        print("Control Manager state is normal. No faults detected.")

        print("Enabling the Control Manager...")
        if not robot.enable_control_manager(unlimited_mode_enabled=True):
            print("Error: Failed to enable the Control Manager.")
            sys.exit(1)
        print("Control Manager enabled successfully.")
        
        return robot

    def init_viewer(self, model_path):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.ik = RBY1WholeBodyIK()

        # Initialize from model default
        mujoco.mj_forward(self.model, self.data)
        self.current_qpos = self.data.qpos.copy()
        self.prev_qpos = self.current_qpos.copy()

        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            start = self.model.jnt_qposadr[i]
            jtype = self.model.jnt_type[i]
            dim = 7 if jtype == mujoco.mjtJoint.mjJNT_FREE else (4 if jtype == mujoco.mjtJoint.mjJNT_BALL else 1)
            print(f"{name:20s} : qpos[{start}:{start+dim}]")

        # Get mocap ids for target bodies and initialize them at current EE poses
        self.ee_l_mid = self.model.body("ee_l_target").mocapid[0]
        self.ee_r_mid = self.model.body("ee_r_target").mocapid[0]

        # Build joint-name mapping (SDK -> MuJoCo qpos adr) and cache base free joint address
        self._build_joint_mapping()

        # Cache base origin (free joint initial pose)
        self._extract_base_origin()

        # Wait for first robot state and apply it before initializing mocap to EE poses            applied = False
        for _ in range(200):  # ~2s
            rs = self.robot_state.load()
            if rs is not None:
                break
        if rs is None:
            print("[init_viewer] No robot state found")
            exit(1)
        # Apply base odom and joint mapping once
        T = rs.odom_T
        x = float(T[0, 2])
        y = float(T[1, 2])
        yaw = math.atan2(T[1, 0], T[0, 0])
        # position
        self.data.qpos[self._base_free_adr + 0] = self._base_origin_pos[0] + x
        self.data.qpos[self._base_free_adr + 1] = self._base_origin_pos[1] + y
        self.data.qpos[self._base_free_adr + 2] = self._base_origin_pos[2]
        # orientation (yaw)
        self.data.qpos[self._base_free_adr + 3 : self._base_free_adr + 7] = self._quat_from_yaw(yaw)

        if rs.position.shape[0] == len(self._sdk_joint_names):
            for sdk_idx, adr in enumerate(self._sdk_to_mj_qadr):
                if adr is not None:
                    self.data.qpos[adr] = rs.position[sdk_idx]

        mujoco.mj_forward(self.model, self.data)

        # Initialize mocap pos/quats to current EE poses
        left_nominal = self.site_pos("end_effector_l", self.data.qpos)
        right_nominal = self.site_pos("end_effector_r", self.data.qpos)
        self.data.mocap_pos[self.ee_l_mid] = left_nominal
        self.data.mocap_pos[self.ee_r_mid] = right_nominal
        # Use EE body orientations
        l_q = self.data.xquat[self.model.body("EE_BODY_L").id].copy()
        r_q = self.data.xquat[self.model.body("EE_BODY_R").id].copy()
        self.data.mocap_quat[self.ee_l_mid] = l_q
        self.data.mocap_quat[self.ee_r_mid] = r_q

        # Base velocity smoothing state
        self._se2_prev = np.zeros(3, dtype=float)

        viewer = mujoco.viewer.launch_passive(
            model=self.model, data=self.data, show_left_ui=False, show_right_ui=False
        )
        mujoco.mjv_defaultFreeCamera(self.model, viewer.cam)
        return viewer

    def robot_state_callback(self, rs):  
        # Store joint positions and odometry (SE2)
        try:
            pos = np.asarray(rs.position, dtype=float).copy()
        except Exception:
            pos = np.array([], dtype=float)
        try:
            odom = np.asarray(rs.odometry, dtype=float).copy()
        except Exception:
            odom = np.eye(3, dtype=float)
        self.robot_state.store(SimpleRobotState(position=pos, odom_T=odom))

    # Helper: FK using viewer model
    def site_pos(self, site_name: str, qpos: np.ndarray) -> np.ndarray:
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        return self.data.site_xpos[sid].copy()

    def controller_loop(self):
        """200 Hz: read shared targets + qpos snapshot, run IK, send command."""
        model_m = rby1_sdk.Model_M()
        body_idx = list(model_m.body_idx)
        while not self._stop.is_set():
            # get the latest targets (from viewer) and qpos snapshot
            left_pos, left_quat, right_pos, right_quat, qpos_for_ik = self.shared.get_for_ik()

            if qpos_for_ik is None or left_pos is None or right_pos is None:
                # wait until viewer publishes first samples
                self.ik_rate.sleep()
                continue

            # incremental whole-body IK
            sol_qpos, success, _info = self.ik.solve(
                left_target_pos=left_pos,
                left_target_quat=left_quat,
                right_target_pos=right_pos,
                right_target_quat=right_quat,
                current_qpos=qpos_for_ik,
                dt=self.ik_rate.dt,
            )

            # Build command: body joint positions (torso+arms) + base SE2 velocity
            try:
                # 1) Compute base velocities via P-control on pose error (reduces oscillations)
                x_err_w = (sol_qpos[0] - qpos_for_ik[0])
                y_err_w = (sol_qpos[1] - qpos_for_ik[1])
                yaw_now = self._yaw_from_quat(sol_qpos[3:7])
                yaw_prev = self._yaw_from_quat(qpos_for_ik[3:7])
                yaw_err = (yaw_now - yaw_prev + math.pi) % (2 * math.pi) - math.pi

                # rotate position error into base frame using current yaw
                cy, sy = math.cos(yaw_prev), math.sin(yaw_prev)
                x_err_b =  cy * x_err_w + sy * y_err_w
                y_err_b = -sy * x_err_w + cy * y_err_w
                x_err_b = x_err_w
                y_err_b = y_err_w

                vx_body = x_err_b / self.ik_rate.dt
                vy_body = y_err_b / self.ik_rate.dt
                wz = yaw_err / self.ik_rate.dt

                vx_body *= 0.1
                vy_body *= 0.1
                wz *= 0.1

                # clamp speeds
                max_lin = 1.5  # m/s
                max_ang = np.pi / 2  # rad/s
                vx_body = float(np.clip(vx_body, -max_lin, max_lin))
                vy_body = float(np.clip(vy_body, -max_lin, max_lin))
                wz = float(np.clip(wz, -max_ang, max_ang))

                # 2) Extract SDK-ordered joint targets for body indices
                # positions_sdk[i] = sol_qpos at the mapped MuJoCo qadr
                positions_sdk = np.zeros(len(self._sdk_joint_names), dtype=float)
                for i, adr in enumerate(self._sdk_to_mj_qadr):
                    if adr is not None:
                        positions_sdk[i] = sol_qpos[adr]

                body_targets = positions_sdk[body_idx]

                # 3) Construct builders
                header = rby1_sdk.CommandHeaderBuilder().set_control_hold_time(1e6)

                jp = (
                    rby1_sdk.JointPositionCommandBuilder()
                    .set_command_header(header)
                    .set_minimum_time(self.ik_rate.dt)
                    .set_position(body_targets)
                )

                se2 = (
                    rby1_sdk.SE2VelocityCommandBuilder()
                    .set_command_header(header)
                    .set_minimum_time(max(3.0 * self.ik_rate.dt, 0.03))
                    .set_velocity(np.array([vx_body, vy_body], dtype=float), float(wz))
                    # .set_acceleration_limit(np.array([0.5, 0.5], dtype=float), 1.0)
                )

                comp = (
                    rby1_sdk.ComponentBasedCommandBuilder()
                    .set_body_command(jp)
                    .set_mobility_command(se2)
                )

                rc = rby1_sdk.RobotCommandBuilder().set_command(comp)

                # Send through the stream
                self.stream.send_command(rc)
            except Exception as e:
                print(f"[controller] send_command error: {e}")

            self.ik_rate.sleep()


    def visualize_loop(self):
        """~60 Hz: pull latest robot state, update MuJoCo, publish mocap targets for controller, render."""
        rs = self.robot_state.load()
        if rs is not None and isinstance(rs, SimpleRobotState):
            # Apply odometry to free joint pose (x,y,yaw) while keeping base height constant
            try:
                T = rs.odom_T
                x = float(T[0, 2])
                y = float(T[1, 2])
                yaw = math.atan2(T[1, 0], T[0, 0])
                # position
                self.data.qpos[self._base_free_adr + 0] = self._base_origin_pos[0] + x
                self.data.qpos[self._base_free_adr + 1] = self._base_origin_pos[1] + y
                self.data.qpos[self._base_free_adr + 2] = self._base_origin_pos[2]
                # orientation (yaw about world Z)
                qw, qx, qy, qz = self._quat_from_yaw(yaw)
                self.data.qpos[self._base_free_adr + 3 : self._base_free_adr + 7] = [qw, qx, qy, qz]
            except Exception as e:
                print(f"[visualize] odometry apply error: {e}")

            # Map 26-DOF joint positions into MuJoCo qpos according to name mapping
            try:
                if rs.position.shape[0] == len(self._sdk_joint_names):
                    for sdk_idx, adr in enumerate(self._sdk_to_mj_qadr):
                        if adr is not None:
                            self.data.qpos[adr] = rs.position[sdk_idx]
                else:
                    print(
                        f"[visualize] Unexpected DOF from robot ({rs.position.shape[0]}) != expected ({len(self._sdk_joint_names)})"
                    )
            except Exception as e:
                print(f"[visualize] joint mapping apply error: {e}")

            mujoco.mj_forward(self.model, self.data)
            self.prev_qpos = self.data.qpos.copy()

        # read current mocap targets (safe in viewer thread)
        left_pos = self.data.mocap_pos[self.ee_l_mid].copy()
        right_pos = self.data.mocap_pos[self.ee_r_mid].copy()
        # read orientation from mocap quats so rotating mocap affects IK
        left_quat = self.data.mocap_quat[self.ee_l_mid].copy()
        right_quat = self.data.mocap_quat[self.ee_r_mid].copy()

        # publish targets + the qpos snapshot for IK
        self.shared.set_from_viewer(left_pos, left_quat, right_pos, right_quat, self.data.qpos)

        # light & render
        mujoco.mj_camlight(self.model, self.data)
        self.viewer.sync()
        self.viewer_rate.sleep()
                
    def _build_joint_mapping(self):
        """Build mapping from SDK joint names (Model_M) to MuJoCo qpos addresses for hinge joints."""
        # SDK model joint names (26 DOF for Model M)
        try:
            sdk_model = rby1_sdk.Model_M()
            self._sdk_joint_names = list(sdk_model.robot_joint_names)
        except Exception:
            # Fallback to hardcoded order if SDK model unavailable
            self._sdk_joint_names = [
                "wheel_fr", "wheel_fl", "wheel_rr", "wheel_rl",
                "torso_0", "torso_1", "torso_2", "torso_3", "torso_4", "torso_5",
                "right_arm_0", "right_arm_1", "right_arm_2", "right_arm_3", "right_arm_4", "right_arm_5", "right_arm_6",
                "left_arm_0", "left_arm_1", "left_arm_2", "left_arm_3", "left_arm_4", "left_arm_5", "left_arm_6",
                "head_0", "head_1",
            ]

        # Map MuJoCo joint name -> qpos address (skip free/ball)
        self._mj_joint_qadr = {}
        self._base_free_adr = None
        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            jtype = int(self.model.jnt_type[i])
            adr = int(self.model.jnt_qposadr[i])
            if jtype == mujoco.mjtJoint.mjJNT_FREE:
                if name == "world_j":
                    self._base_free_adr = adr
                continue
            if jtype == mujoco.mjtJoint.mjJNT_HINGE:
                self._mj_joint_qadr[name] = adr

        if self._base_free_adr is None:
            raise RuntimeError("Base free joint address not found (world_j)")

        # Build SDK index -> MuJoCo qpos address list
        self._sdk_to_mj_qadr = []
        missing = []
        for name in self._sdk_joint_names:
            adr = self._mj_joint_qadr.get(name)
            if adr is None:
                missing.append(name)
            self._sdk_to_mj_qadr.append(adr)
        if len(missing) > 0:
            print(f"[mapping] Missing joints in MuJoCo model: {missing}")

    def _extract_base_origin(self):
        """Cache initial base position (x0,y0,z0)."""
        self._base_origin_pos = self.data.qpos[self._base_free_adr : self._base_free_adr + 3].copy()

    @staticmethod
    def _quat_from_yaw(yaw):
        """Quaternion (w,x,y,z) from yaw angle about +Z."""
        half = 0.5 * yaw
        return math.cos(half), 0.0, 0.0, math.sin(half)

    @staticmethod
    def _yaw_from_quat(q):
        """Yaw from quaternion (w,x,y,z)."""
        w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        # ZYX yaw
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    def run(self):
        """Start controller in a background thread; run the viewer loop on main thread."""
        self._controller_thread = threading.Thread(
            target=self.controller_loop, name="controller", daemon=True
        )
        self._controller_thread.start()

        try:
            # main viewer loop (must own OpenGL context)
            while self.viewer.is_running() and not self._stop.is_set():
                self.visualize_loop()
        finally:
            # shutdown sequence
            self._stop.set()
            if self._controller_thread is not None:
                self._controller_thread.join(timeout=2.0)
            try:
                self.stream.close()
            except Exception:
                pass
            try:
                self.robot.stop_state_update()
            except Exception:
                pass
            try:
                self.viewer.close()
            except Exception:
                pass

def main():
    model_path = PROJECT_ROOT + "/model/rby1/rby1_mocap.xml"
    gui = SimGUI(model_path=model_path, address="localhost:50051")
    gui.run()

if __name__ == "__main__":
    main()