# RBY1 Realtime Controller (pybind)

This directory contains a trimmed-down version of the realtime controller used
to command the RBY1 robot at ~500 Hz. The controller is implemented entirely in
C++, exposes a small C++ executable (`rby1_realtime_control_main`), and ships a
`pybind11` module (`rby1_controller`) for direct consumption from Python.

## Install rby1-sdk
https://github.com/RainbowRobotics/rby1-sdk

refer to RBY1_SDK_BUILD.md

## Building
```bash
# Need to install rby1-sdk first
mkdir -p build && cd build
# You may need to adjust the directory with your address (/usr/local/) - if rby1-sdk is installed using sudo make install
cmake .. \
  -DCMAKE_BUILD_TYPE=Debug \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
  -DPython_EXECUTABLE="$(which python)" \
  -DRBY1_SDK_INCLUDE_DIR="$HOME/.local/include" \
  -DRBY1_SDK_LIBRARY="$HOME/.local/lib/librby1-sdk.so"
cmake --build .

cd ..
ln -s build/rby1_controller*.so rby1_controller.so
```

## Rebuilding
If you have made the source code changes
```bash
cd build
cmake --build . -j
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
