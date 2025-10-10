cmake_minimum_required(VERSION 3.8)
project(huro)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ROS2 Dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(rosbag2_cpp REQUIRED)

# Unitree ROS2 messages
find_package(unitree_go REQUIRED)
find_package(unitree_hg REQUIRED)
find_package(unitree_api REQUIRED)

# MuJoCo
find_package(mujoco REQUIRED)

set(
  DEPENDENCY_LIST
  unitree_go
  unitree_hg
  unitree_api
  rclcpp
  std_msgs
  sensor_msgs
  tf2
  tf2_ros
  rosbag2_cpp
)

set(
  SRCS
  src/humanoid_root_node.cc
  src/quadruped_root_node.cc
  src/motor_crc_hg.cc
  src/sim_node.cc
)

## TODO: Create a huro library and link it to the executables
# (thus removing the need to target_link_libraries() for all apps

## Compile Cpp Executables ##
add_executable(g1 apps/g1.cc ${SRCS})
target_include_directories(g1 PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>)
target_link_libraries(g1 mujoco)
ament_target_dependencies(g1 ${DEPENDENCY_LIST})
target_compile_options(g1 PUBLIC "-Wall;-Wextra;-O3;")
target_compile_features(g1 PUBLIC c_std_99 cxx_std_17)
install(TARGETS
  g1
  DESTINATION lib/${PROJECT_NAME}
)

add_executable(go2 apps/go2.cc ${SRCS})
target_include_directories(go2 PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>)
target_link_libraries(go2 mujoco)
ament_target_dependencies(go2 ${DEPENDENCY_LIST})
target_compile_options(go2 PUBLIC "-Wall;-Wextra;-O3;")
target_compile_features(go2 PUBLIC c_std_99 cxx_std_17)
install(TARGETS
  go2
  DESTINATION lib/${PROJECT_NAME}
)

add_executable(sim apps/sim.cc ${SRCS})
target_include_directories(sim PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>)
target_link_libraries(sim mujoco)
ament_target_dependencies(sim ${DEPENDENCY_LIST})
target_compile_features(sim PUBLIC c_std_99 cxx_std_17)
target_compile_options(sim PUBLIC "-Wall;-Wextra;-O3")
install(TARGETS sim
  DESTINATION lib/${PROJECT_NAME})

## Install Python Executables ##
# Install Python modules
ament_python_install_package(${PROJECT_NAME})
# Install Python executables
install(PROGRAMS
  python/move_example.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install Resources Folder ##
install(
  DIRECTORY resources
  DESTINATION share/${PROJECT_NAME}
)

## Install Launch Files ##
install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
