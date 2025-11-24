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
    sudo apt install -y socat # first time
    sudo ~/gripper_client.sh start
   ```