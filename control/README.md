# RBY1 Realtime Controller (pybind)

This directory contains a trimmed-down version of the realtime controller used
to command the RBY1 robot at ~500 Hz. The controller is implemented entirely in
C++, exposes a small C++ executable (`rby1_realtime_control_main`), and ships a
`pybind11` module (`rby1_controller`) for direct consumption from Python.

## Building

```bash
mkdir -p build && cd build
cmake -DRBY1_SDK_ROOT=/path/to/rby1-sdk ..
cmake --build .
```

If your SDK install uses non-standard paths, set `RBY1_SDK_INCLUDE_DIR` and
`RBY1_SDK_LIBRARY` explicitly. The resulting shared object
(`rby1_controller.*.so`) can be placed on `PYTHONPATH`, and the executable can
be used to sanity-check connectivity.

## Python Usage

```python
from rby1.control import RealtimeDriver, Config

config = Config()
config.robot_address = "192.168.30.1:50051"

driver = RealtimeDriver(config)
driver.start()
driver.wait_until_ready(10.0)
driver.set_body_position_targets([...])
driver.stop()
```

See `rby1/scripts/rby1_pybind_joint_wbc_gui.py` for a complete example that
streams IK results from Python into the controller while visualising the robot
in MuJoCo.
