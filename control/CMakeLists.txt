cmake_minimum_required(VERSION 3.20)
project(rby1_realtime_controller LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(Threads REQUIRED)

set(RBY1_SDK_ROOT "" CACHE PATH "Path to the root of the compiled RBY1 SDK")
set(RBY1_SDK_INCLUDE_DIR "" CACHE PATH "Override for RBY1 SDK include directory")
set(RBY1_SDK_LIBRARY "" CACHE FILEPATH "Path to the librby1-sdk library")

if (RBY1_SDK_INCLUDE_DIR STREQUAL "")
  if (NOT RBY1_SDK_ROOT STREQUAL "")
    set(RBY1_SDK_INCLUDE_DIR "${RBY1_SDK_ROOT}/include" CACHE PATH "" FORCE)
  else()
    message(FATAL_ERROR "Set RBY1_SDK_ROOT or RBY1_SDK_INCLUDE_DIR so headers like 'rby1-sdk/robot.h' can be found.")
  endif()
endif()

if (RBY1_SDK_LIBRARY STREQUAL "")
  if (NOT RBY1_SDK_ROOT STREQUAL "")
    find_library(RBY1_SDK_LIBRARY NAMES rby1-sdk
                 PATHS "${RBY1_SDK_ROOT}/lib" "${RBY1_SDK_ROOT}/build/src"
                 NO_DEFAULT_PATH)
  endif()
  if (NOT RBY1_SDK_LIBRARY)
    message(FATAL_ERROR "Failed to locate librby1-sdk. Set RBY1_SDK_LIBRARY to the full path of the library.")
  endif()
endif()

add_library(rby1_realtime_driver STATIC
  realtime_driver.cc
)
set_target_properties(rby1_realtime_driver PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)
target_include_directories(rby1_realtime_driver
  PUBLIC
    ${RBY1_SDK_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(rby1_realtime_driver
  PUBLIC
    Eigen3::Eigen
    Threads::Threads
    ${RBY1_SDK_LIBRARY}
)

pybind11_add_module(rby1_controller MODULE
  py_module.cc
)
target_link_libraries(rby1_controller PRIVATE rby1_realtime_driver)
target_include_directories(rby1_controller PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(rby1_realtime_control_main
  main.cc
)
target_link_libraries(rby1_realtime_control_main PRIVATE rby1_realtime_driver)

add_executable(stream_state
  stream_state.cc
)
target_link_libraries(stream_state PRIVATE rby1_realtime_driver)

add_executable(test_controller_loop
  test_controller_loop.cc
)
target_link_libraries(test_controller_loop PRIVATE rby1_realtime_driver)
