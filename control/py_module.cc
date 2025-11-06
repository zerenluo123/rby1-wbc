#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <iostream>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "realtime_driver.h"

namespace py = pybind11;
using rby1::control::RealtimeDriver;

namespace {

void BindSnapshot(py::module_& m) {
  py::class_<RealtimeDriver::RobotSnapshot>(m, "RobotSnapshot")
      .def_property_readonly(
          "timestamp_ns",
          [](const RealtimeDriver::RobotSnapshot& self) {
            return self.timestamp_ns;
          })
      .def_property_readonly(
          "joint_is_ready",
          [](const RealtimeDriver::RobotSnapshot& self) {
            return self.joint_is_ready;
          })
      .def_property_readonly(
          "joint_position",
          [](const RealtimeDriver::RobotSnapshot& self) {
            return self.joint_position;
          })
      .def_property_readonly(
          "joint_velocity",
          [](const RealtimeDriver::RobotSnapshot& self) {
            return self.joint_velocity;
          })
      .def_property_readonly(
          "joint_current",
          [](const RealtimeDriver::RobotSnapshot& self) {
            return self.joint_current;
          })
      .def_property_readonly(
          "joint_torque",
          [](const RealtimeDriver::RobotSnapshot& self) {
            return self.joint_torque;
          })
      .def_property_readonly(
          "joint_target_position",
          [](const RealtimeDriver::RobotSnapshot& self) {
            return self.joint_target_position;
          })
      .def_property_readonly(
          "joint_target_velocity",
          [](const RealtimeDriver::RobotSnapshot& self) {
            return self.joint_target_velocity;
          })
      .def_property_readonly(
          "joint_feedback_gain",
          [](const RealtimeDriver::RobotSnapshot& self) {
            return self.joint_feedback_gain;
          })
      .def_property_readonly(
          "joint_feedforward_torque",
          [](const RealtimeDriver::RobotSnapshot& self) {
            return self.joint_feedforward_torque;
          })
      .def_property_readonly(
          "odom_SE2",
          [](const RealtimeDriver::RobotSnapshot& self) {
            return self.odom_SE2;
})
      .def_property_readonly(
          "left_ee_wrench",
          [](const RealtimeDriver::RobotSnapshot& self) {
            return self.left_ee_wrench;
          })
      .def_property_readonly(
          "right_ee_wrench",
          [](const RealtimeDriver::RobotSnapshot& self) {
            return self.right_ee_wrench;
          })
      .def_property_readonly(
          "left_ft_valid",
          [](const RealtimeDriver::RobotSnapshot& self) {
            return self.left_ft_valid;
          })
      .def_property_readonly(
          "right_ft_valid",
          [](const RealtimeDriver::RobotSnapshot& self) {
            return self.right_ft_valid;
          })
      .def_readonly("is_valid", &RealtimeDriver::RobotSnapshot::is_valid);
}

void BindConfig(py::module_& m) {
  py::class_<RealtimeDriver::Config>(m, "Config")
      .def(py::init<>())
      .def_readwrite("robot_address", &RealtimeDriver::Config::robot_address)
      .def_readwrite("low_pass_freq_hz",
                     &RealtimeDriver::Config::low_pass_freq_hz)
      .def_readwrite("expect_wheel_velocity",
                     &RealtimeDriver::Config::expect_wheel_velocity)
      .def_readwrite("command_timeout_us",
                     &RealtimeDriver::Config::command_timeout_us);
}

void BindDriver(py::module_& m) {
  py::class_<RealtimeDriver>(m, "RealtimeDriver")
      .def(py::init<RealtimeDriver::Config>(),
           py::arg("config") = RealtimeDriver::Config(),
           "Create a realtime controller with the given configuration.")
      .def(
          "run",
          [](RealtimeDriver& self) {
            py::gil_scoped_release release;
            self.Run();
          },
          "Run the control loop in the current thread.")
      .def("start",
           [](RealtimeDriver& self) {
             py::gil_scoped_release release;
             self.Start();
           },
           "Start the control loop on a background thread.")
      .def("stop",
           [](RealtimeDriver& self) {
             py::gil_scoped_release release;
             self.Stop();
           },
           "Stop the control loop and wait for shutdown.")
      .def("is_running", &RealtimeDriver::IsRunning,
           "Return true while the control loop is active.")
      .def("wait_until_ready", &RealtimeDriver::WaitUntilReady,
           py::arg("timeout_sec"), py::call_guard<py::gil_scoped_release>(),
           "Block until all robot components report ready or timeout occurs.")
      .def("set_component_position_targets",
           &RealtimeDriver::SetComponentPositionTargets, py::arg("name"),
           py::arg("targets"),
           "Set joint position targets for a named component (e.g. 'LEFT').")
      .def("set_component_velocity_targets",
           &RealtimeDriver::SetComponentVelocityTargets, py::arg("name"),
           py::arg("targets"),
           "Set joint velocity targets for a named component.")
      .def("set_body_position_targets",
           &RealtimeDriver::SetBodyPositionTargets, py::arg("targets"),
           "Apply joint position targets across torso, left arm, right arm, "
           "and head in SDK order.")
      .def("set_base_twist_command", &RealtimeDriver::SetBaseTwistCommand,
           py::arg("twist_body"),
           "Command the base using body-frame velocities [vx, vy, wz].")
      .def("set_wheel_velocity_targets",
           &RealtimeDriver::SetWheelVelocityTargets, py::arg("targets"),
           "Direct wheel velocity targets when expect_wheel_velocity is true.")
      .def(
          "get_latest_robot_state",
          [](const RealtimeDriver& self) -> py::object {
            // Release the GIL only while invoking the C++ getter, then
            // reacquire it before interacting with Python objects.
            const auto snapshot = [&self]() {
              py::gil_scoped_release release;
              return self.GetLatestRobotState();
            }();
            if (!snapshot.has_value()) {
              return py::none();
            }
            return py::cast(snapshot.value());
          },
          "Return the most recent robot state snapshot, or None if unavailable.")
      .def("__enter__",
           [](RealtimeDriver& self) -> RealtimeDriver& {
             self.Start();
             return self;
           })
      .def("__exit__",
           [](RealtimeDriver& self, py::object, py::object, py::object) {
             self.Stop();
             return false;
           });
}

}  // namespace

PYBIND11_MODULE(rby1_controller, m) {
  m.doc() = "Pybind11 bindings for the RBY1 realtime controller";

  BindSnapshot(m);
  BindConfig(m);
  BindDriver(m);

  m.def(
      "debug_echo",
      [](const std::string& message) {
        std::cout << "[pybind debug] " << message << std::endl;
        return message;
      },
      py::arg("message"),
      "Print the supplied message from within the pybind module and return it.");
}
