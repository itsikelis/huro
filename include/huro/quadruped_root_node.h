// Copyright 2025 Ioannis Tsikelis

#ifndef HURO_QUADRUPED_ROOT_NODE_H_
#define HURO_QUADRUPED_ROOT_NODE_H_

#include <huro/params.h>
#include <tf2_ros/transform_broadcaster.h>

#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <unitree_go/msg/low_cmd.hpp>
#include <unitree_go/msg/low_state.hpp>
#include <unitree_go/msg/motor_cmd.hpp>
#include <unitree_go/msg/sport_mode_state.hpp>

namespace huro {
class QuadrupedRootNode : public rclcpp::Node {
public:
  using LowCmdMsg = unitree_go::msg::LowCmd;
  using ImuStateMsg = unitree_go::msg::IMUState;
  using MotorStateMsg = unitree_go::msg::MotorState;
  using LowStateMsg = unitree_go::msg::LowState;
  using OdometryMsg = unitree_go::msg::SportModeState;

  using JointStateMsg = sensor_msgs::msg::JointState;
  using TransformStamped = geometry_msgs::msg::TransformStamped;

  using TransformBroadcaster = tf2_ros::TransformBroadcaster;

  QuadrupedRootNode(Params params);

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
#endif // HURO_QUADRUPED_ROOT_NODE_H_
