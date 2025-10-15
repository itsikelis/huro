// Copyright 2025 Ioannis Tsikelis

#include <huro/sim_node.h>

#include <cstdio>
#include <memory>

#include <rclcpp/rclcpp.hpp>

#include <unitree_go/msg/low_cmd.hpp>
#include <unitree_go/msg/low_state.hpp>
#include <unitree_go/msg/motor_cmd.hpp>
#include <unitree_go/msg/sport_mode_state.hpp>

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);

  auto params = huro::Params();
  params.fix_base = false;
  params.info_imu = false;
  params.info_motors = false;
  params.n_motors = 12;
  params.xml_filename = "go2/go2.xml";
  params.sim_dt_ms = 2;
  params.lowstate_topic_name = "/lowstate";
  params.lowcmd_topic_name = "/lowcmd";
  params.odom_topic_name = "/sportmodestate";
  params.base_link_name = "base";
  params.sole_link_name = "RR_point";
  params.joint_names = {
      "FL_hip_joint",   "FL_thigh_joint", "FL_calf_joint",  "FR_hip_joint",
      "FR_thigh_joint", "FR_calf_joint",  "RL_hip_joint",   "RL_thigh_joint",
      "RL_calf_joint",  "RR_hip_joint",   "RR_thigh_joint", "RR_calf_joint",

  };

  auto node = std::make_shared<
      huro::SimNode<unitree_go::msg::LowCmd, unitree_go::msg::LowState,
                    unitree_go::msg::SportModeState, 12>>(params);
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
