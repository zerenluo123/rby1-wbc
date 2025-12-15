# Gripper Setup for RB-Y1

The RB-Y1 robot supports gripper control through locally emulated devices. To use this feature, your local machine must be connected to the robot's PC via an Ethernet cable.

## Setup Instructions

1. **Power on the robot.**
2. **SSH into the robot's PC:**
   ```bash
   ssh nvidia@192.168.30.2
   ```
3. **Start the gripper sharing service on the robot's PC:**
   ```bash
    sudo ~/gripper_share.sh start
   ```
4. **On your local PC, start the gripper client:**
    ```bash
    sudo ~/gripper_client.sh start
   ```

## Python TCP Gripper Server/Client

If you prefer a pure Python solution that lets a remote workstation stream gripper targets without creating virtual serial devices, use the provided TCP server and client:

1. **On the robot PC (Computer A)**
    ```bash
    cd /path/to/rby1
    python -m gripper.gripper_server --host 0.0.0.0 --port 5678
    ```
    This initializes, homes, and starts the local `Gripper` thread, then listens for JSON commands on TCP port 5678.

2. **On the remote workstation (Computer B)**
    ```bash
    cd /path/to/rby1
    python -m gripper.gripper_client --host <robot-pc-ip> --port 5678 set-target 0.05 0.05
    ```
    The CLI can also run `status`, `ping`, `start`, `stop`, `initialize`, or `homing`.

3. **Programmatic control**
    ```python
    from gripper.gripper_client import RemoteGripper

    gripper = RemoteGripper(host="192.168.30.2", port=5678)
    gripper.set_target([0.07, 0.07])
    ```
    `RemoteGripper` mirrors the `Gripper` API, so existing control code can substitute it directly when the hardware lives on another machine.
