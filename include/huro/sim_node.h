// Copyright 2025 Ioannis Tsikelis

#ifndef HURO_SIM_NODE_H_
#define HURO_SIM_NODE_H_

#include <mujoco/mujoco.h>

#include <array>
#include <memory>
#include <string>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <rclcpp/rclcpp.hpp>
#include <unitree_go/msg/sport_mode_state.hpp>
#include <unitree_hg/msg/low_cmd.hpp>
#include <unitree_hg/msg/low_state.hpp>
#include <unitree_hg/msg/motor_cmd.hpp>

namespace huro {
class SimNode : public rclcpp::Node {
public:
  using LowCmdMsg = unitree_hg::msg::LowCmd;
  using LowStateMsg = unitree_hg::msg::LowState;
  using OdometryMsg = unitree_go::msg::SportModeState;

public:
  SimNode();
  ~SimNode();

protected:
  void Step();
  void LowCmdHandler(LowCmdMsg::SharedPtr message);
  void FixModelBase();
  OdometryMsg GenerateOdometryMsg();
  LowStateMsg GenerateLowStateMsg();
  mjtNum GetZDistanceFromSoleToPelvis();

protected:
  const double control_dt_ = 0.002; // 2ms
  const int timer_dt_ = control_dt_ * 1000;
  double time_; // Running time count
  int mode_machine;

  rclcpp::Publisher<LowStateMsg>::SharedPtr lowstate_pub_;
  rclcpp::Publisher<OdometryMsg>::SharedPtr odom_pub_;
  rclcpp::Subscription<LowCmdMsg>::SharedPtr lowmcd_sub_;
  rclcpp::TimerBase::SharedPtr timer_;
  mjModel *mj_model_;
  mjData *mj_data_;

  std::array<double, 29> q_des_;
  std::array<double, 29> qdot_des_;
  std::array<double, 29> tau_ff_;
  std::array<double, 29> kp_;
  std::array<double, 29> kd_;
};
} // namespace huro
#endif // HURO_SIM_NODE_H_
