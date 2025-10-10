// Copyright 2025 Ioannis Tsikelis

#include <huro/params.h>
#include <huro/quadruped_root_node.h>

#include <cstdio>
#include <memory>

#include <rclcpp/rclcpp.hpp>

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);

  auto params = huro::Params();
  params.high_freq = true;
  params.info_imu = false;
  params.info_motors = false;
  params.n_motors = 12;
  params.joint_names = {
      "FL_hip_joint",   "FL_thigh_joint", "FL_calf_joint",  "FR_hip_joint",
      "FR_thigh_joint", "FR_calf_joint",  "RL_hip_joint",   "RL_thigh_joint",
      "RL_calf_joint",  "RR_hip_joint",   "RR_thigh_joint", "RR_calf_joint",

  };

  auto node = std::make_shared<huro::QuadrupedRootNode>(params);
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
