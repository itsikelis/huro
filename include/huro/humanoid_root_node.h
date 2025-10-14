// Copyright 2025 Ioannis Tsikelis

#ifndef HURO_HUMANOID_ROOT_NODE_H_
#define HURO_HUMANOID_ROOT_NODE_H_

#include <huro/humanoid_root_node.h>
#include <huro/params.h>

#include <tf2_ros/transform_broadcaster.h>

#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <unitree_go/msg/sport_mode_state.hpp>
#include <unitree_hg/msg/low_cmd.hpp>
#include <unitree_hg/msg/low_state.hpp>
#include <unitree_hg/msg/motor_cmd.hpp>

namespace huro {
class HumanoidRootNode : public rclcpp::Node {
public:
  using LowCmdMsg = unitree_hg::msg::LowCmd;
  using ImuStateMsg = unitree_hg::msg::IMUState;
  using MotorStateMsg = unitree_hg::msg::MotorState;
  using LowStateMsg = unitree_hg::msg::LowState;
  using OdometryMsg = unitree_go::msg::SportModeState;

  using JointStateMsg = sensor_msgs::msg::JointState;
  using TransformStamped = geometry_msgs::msg::TransformStamped;

  using TransformBroadcaster = tf2_ros::TransformBroadcaster;

  HumanoidRootNode(Params params);

protected:
  void LowStateHandler(LowStateMsg::SharedPtr message);
  void OdometryHandler(OdometryMsg::SharedPtr message);
  TransformStamped GenerateTransformMsg(OdometryMsg::SharedPtr message);

protected:
  Params params_;
  rclcpp::Publisher<JointStateMsg>::SharedPtr jointstate_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Subscription<LowStateMsg>::SharedPtr lowstate_sub_;
  rclcpp::Subscription<OdometryMsg>::SharedPtr odometry_sub_;
  std::shared_ptr<TransformBroadcaster> tf_broadcaster_;
};
} // namespace huro
#endif // HURO_HUMANOID_ROOT_NODE_H_
