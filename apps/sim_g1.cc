// Copyright 2025 Ioannis Tsikelis

#include <huro/sim_node.h>

#include <cstdio>
#include <memory>

#include <rclcpp/rclcpp.hpp>

#include <unitree_go/msg/sport_mode_state.hpp>
#include <unitree_hg/msg/low_cmd.hpp>
#include <unitree_hg/msg/low_state.hpp>
#include <unitree_hg/msg/motor_cmd.hpp>

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);

  auto params = huro::Params();
  params.fix_base = true;
  params.info_imu = false;
  params.info_motors = false;
  params.n_motors = 29;
  params.xml_filename = "g1/g1_29dof.xml";
  params.sim_dt_ms = 2;
  params.lowstate_topic_name = "/lowstate";
  params.lowcmd_topic_name = "/lowcmd";
  params.odom_topic_name = "/odommodestate";
  params.base_link_name = "pelvis";
  params.sole_link_name = "right_foot_point_contact";
  params.joint_names = {"left_hip_pitch_joint",
                        "left_hip_roll_joint",
                        "left_hip_yaw_joint",
                        "left_knee_joint",
                        "left_ankle_pitch_joint",
                        "left_ankle_roll_joint",
                        "right_hip_pitch_joint",
                        "right_hip_roll_joint",
                        "right_hip_yaw_joint",
                        "right_knee_joint",
                        "right_ankle_pitch_joint",
                        "right_ankle_roll_joint",
                        "waist_yaw_joint",
                        "waist_roll_joint",
                        "waist_pitch_joint",
                        "left_shoulder_pitch_joint",
                        "left_shoulder_roll_joint",
                        "left_shoulder_yaw_joint",
                        "left_elbow_joint",
                        "left_wrist_roll_joint",
                        "left_wrist_pitch_joint",
                        "left_wrist_yaw_joint",
                        "right_shoulder_pitch_joint",
                        "right_shoulder_roll_joint",
                        "right_shoulder_yaw_joint",
                        "right_elbow_joint",
                        "right_wrist_roll_joint",
                        "right_wrist_pitch_joint",
                        "right_wrist_yaw_joint"};

  auto node = std::make_shared<
      huro::SimNode<unitree_hg::msg::LowCmd, unitree_hg::msg::LowState,
                    unitree_go::msg::SportModeState, 29>>(params);
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
