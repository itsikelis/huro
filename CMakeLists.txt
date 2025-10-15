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
  src/motor_crc_hg.cc
)

## TODO: Create a huro library and link it to the executables
# (thus removing the need to target_link_libraries() for all apps

## Compile Cpp Executables ##
add_executable(root_g1 apps/root_g1.cc ${SRCS})
target_include_directories(root_g1 PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>)
target_link_libraries(root_g1 mujoco)
ament_target_dependencies(root_g1 ${DEPENDENCY_LIST})
target_compile_options(root_g1 PUBLIC "-Wall;-Wextra;-O3;")
target_compile_features(root_g1 PUBLIC c_std_99 cxx_std_17)
install(
  TARGETS root_g1
  DESTINATION lib/${PROJECT_NAME}
)

add_executable(root_go2 apps/root_go2.cc ${SRCS})
target_include_directories(root_go2 PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>)
target_link_libraries(root_go2 mujoco)
ament_target_dependencies(root_go2 ${DEPENDENCY_LIST})
target_compile_options(root_go2 PUBLIC "-Wall;-Wextra;-O3;")
target_compile_features(root_go2 PUBLIC c_std_99 cxx_std_17)
install(
  TARGETS root_go2
  DESTINATION lib/${PROJECT_NAME}
)

add_executable(sim_g1 apps/sim_g1.cc ${SRCS})
target_include_directories(sim_g1 PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>)
target_link_libraries(sim_g1 mujoco)
ament_target_dependencies(sim_g1 ${DEPENDENCY_LIST})
target_compile_features(sim_g1 PUBLIC c_std_99 cxx_std_17)
target_compile_options(sim_g1 PUBLIC "-Wall;-Wextra;-O3")
install(
  TARGETS sim_g1
  DESTINATION lib/${PROJECT_NAME}
)

add_executable(sim_go2 apps/sim_go2.cc ${SRCS})
target_include_directories(sim_go2 PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>)
target_link_libraries(sim_go2 mujoco)
ament_target_dependencies(sim_go2 ${DEPENDENCY_LIST})
target_compile_features(sim_go2 PUBLIC c_std_99 cxx_std_17)
target_compile_options(sim_go2 PUBLIC "-Wall;-Wextra;-O3")
install(
  TARGETS sim_go2
  DESTINATION lib/${PROJECT_NAME}
)

## Install Python Executables ##
# Install Python modules
ament_python_install_package(${PROJECT_NAME})
# Install Python executables
install(
  PROGRAMS python/move_example.py
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
