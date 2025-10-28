from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Union

import numpy as np
import rby1_sdk


def _ensure_array(values: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Return a 1-D float array copy when values are provided."""
    if values is None:
        return None
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr.copy()


@dataclass
class InitPosition:
    """Container for optional joint targets used by move_initial."""

    torso: Optional[np.ndarray] = None
    left_arm: Optional[np.ndarray] = None
    right_arm: Optional[np.ndarray] = None
    head: Optional[np.ndarray] = None
    grippers: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.torso = _ensure_array(self.torso)
        self.left_arm = _ensure_array(self.left_arm)
        self.right_arm = _ensure_array(self.right_arm)
        self.head = _ensure_array(self.head)
        self.grippers = _ensure_array(self.grippers)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, np.ndarray]) -> "InitPosition":
        """Build InitPosition from a dictionary keyed by component name."""
        return cls(
            torso=mapping.get("torso"),
            left_arm=mapping.get("left_arm"),
            right_arm=mapping.get("right_arm"),
            head=mapping.get("head"),
            grippers=mapping.get("grippers"),
        )

    def has_body_targets(self) -> bool:
        return any(component is not None for component in (self.torso, self.left_arm, self.right_arm))


def _joint_position_command(
    joint_values: np.ndarray,
    minimum_time: float,
    control_hold_time: Optional[float],
) -> rby1_sdk.JointPositionCommandBuilder:
    builder = (
        rby1_sdk.JointPositionCommandBuilder()
        .set_minimum_time(float(minimum_time))
        .set_position(joint_values)
    )
    if control_hold_time is not None:
        builder = builder.set_command_header(
            rby1_sdk.CommandHeaderBuilder().set_control_hold_time(float(control_hold_time))
        )
    return builder


def move_initial(
    address: str,
    init_positions: Union[InitPosition, Mapping[str, np.ndarray]],
    *,
    power_pattern: str = ".*",
    servo_pattern: str = ".*",
    minimum_time: float = 2.0,
    control_hold_time: Optional[float] = None,
    gripper_pose: Optional[np.ndarray] = None,
) -> bool:
    """
    Move the RB-Y1 Model M robot to the provided initial joint configuration.

    Parameters
    ----------
    address
        gRPC endpoint for the robot controller (e.g., "localhost:50051").
    init_positions
        Mapping or InitPosition describing body/head joint targets.
    power_pattern
        Regex pattern specifying which power groups to enable.
    servo_pattern
        Regex pattern specifying which actuators to servo on.
    minimum_time
        Minimum execution time passed to each joint command.
    control_hold_time
        Optional control hold time applied via the command header.
    gripper_pose
        Placeholder for future use. The current implementation ignores it.
    """

    if not isinstance(init_positions, InitPosition):
        init_positions = InitPosition.from_mapping(init_positions)

    robot: Optional[rby1_sdk.Robot_M] = None
    try:
        print(f"INFO: Connecting to RB-Y1 (Model M) at {address}")
        robot = rby1_sdk.create_robot_m(address)

        if not robot.connect():
            print(f"ERROR: Failed to connect to robot at {address}")
            return False

        if not robot.is_power_on(power_pattern) and not robot.power_on(power_pattern):
            print(f"ERROR: Failed to power on components matching {power_pattern}")
            return False

        if not robot.is_servo_on(servo_pattern) and not robot.servo_on(servo_pattern):
            print(f"ERROR: Failed to servo on components matching {servo_pattern}")
            return False

        control_state = robot.get_control_manager_state()
        fault_states = {
            rby1_sdk.ControlManagerState.State.MajorFault,
            rby1_sdk.ControlManagerState.State.MinorFault,
        }
        if control_state.state in fault_states:
            print(f"WARNING: Control manager in fault state ({control_state.state}); attempting reset")
            if not robot.reset_fault_control_manager():
                print("ERROR: Unable to reset control manager fault state")
                return False

        if not robot.enable_control_manager():
            print("ERROR: Failed to enable control manager")
            return False

        component_command = rby1_sdk.ComponentBasedCommandBuilder()

        if init_positions.has_body_targets():
            body_builder = rby1_sdk.BodyComponentBasedCommandBuilder()

            if init_positions.torso is not None:
                body_builder.set_torso_command(
                    _joint_position_command(init_positions.torso, minimum_time, control_hold_time)
                )

            if init_positions.right_arm is not None:
                body_builder.set_right_arm_command(
                    _joint_position_command(init_positions.right_arm, minimum_time, control_hold_time)
                )

            if init_positions.left_arm is not None:
                body_builder.set_left_arm_command(
                    _joint_position_command(init_positions.left_arm, minimum_time, control_hold_time)
                )

            component_command.set_body_command(body_builder)

        if init_positions.head is not None:
            component_command.set_head_command(
                _joint_position_command(init_positions.head, minimum_time, control_hold_time)
            )

        if not init_positions.has_body_targets() and init_positions.head is None:
            print("ERROR: No joint targets provided for torso, arms, or head; aborting.")
            return False

        command = rby1_sdk.RobotCommandBuilder().set_command(component_command)

        print(f"INFO: Sending initial pose command (minimum_time={minimum_time:.2f})")
        feedback = robot.send_command(command, 10).get()

        if feedback.finish_code != rby1_sdk.RobotCommandFeedback.FinishCode.Ok:
            print(f"ERROR: Initial pose command failed with finish code {feedback.finish_code}")
            return False

        print("INFO: Initial pose command executed successfully")
        return True
    except Exception as exc:
        print(f"ERROR: Failed to execute initial pose command: {exc}")
        return False
    finally:
        if robot is not None:
            try:
                robot.disconnect()
            except Exception as disconnect_exc:
                print(f"DEBUG: Error while disconnecting from robot: {disconnect_exc}")


def load_init_positions_from_file(path: Union[str, Path]) -> InitPosition:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Init position file not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    if not isinstance(payload, dict):
        raise ValueError(f"Init position file must contain a JSON object, got {type(payload).__name__}")

    normalized: dict[str, Optional[np.ndarray]] = {}
    for key, value in payload.items():
        if value is None:
            normalized[key] = None
            continue

        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        normalized[key] = arr

    return InitPosition.from_mapping(normalized)


DEFAULT_INIT_POSITION = InitPosition(
    torso=np.array([0.0, 0.7854, -1.5708, 0.7854, 0.0, 0.0], dtype=float),
    left_arm=np.array([0.0, 0.0873, 0.0, -2.0944, 0.0, 0.9599, -1.5708], dtype=float),
    right_arm=np.array([0.0, -0.0873, 0.0, -2.0944, 0.0, 0.9599, 1.5708], dtype=float),
    head=np.array([0.0, 0.6109], dtype=float),
    grippers=np.array([0.1, 0.1], dtype=float),
)

def main() -> int:
    parser = argparse.ArgumentParser(description="Move RB-Y1 Model M to an initial pose.")
    parser.add_argument("--address", default="localhost:50051", help="Robot controller address.")
    parser.add_argument("--minimum-time", type=float, default=2.0, help="Minimum execution time for joint commands.")
    parser.add_argument("--control-hold-time", type=float, default=None, help="Optional control hold time.")
    parser.add_argument("--power-pattern", default=".*", help="Regex for power groups to enable.")
    parser.add_argument("--servo-pattern", default=".*", help="Regex for servo groups to enable.")
    parser.add_argument("--init-file", default=None, help="Path to a JSON file with initial joint targets.")
    args = parser.parse_args()

    init_positions: Union[InitPosition, Mapping[str, np.ndarray]] = DEFAULT_INIT_POSITION
    if args.init_file is not None:
        init_positions = load_init_positions_from_file(args.init_file)

    success = move_initial(
        args.address,
        init_positions,
        power_pattern=args.power_pattern,
        servo_pattern=args.servo_pattern,
        minimum_time=args.minimum_time,
        control_hold_time=args.control_hold_time,
    )
    return 0 if success else 1

if __name__ == "__main__":
    main()
