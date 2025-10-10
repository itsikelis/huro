// Copyright 2025 Ioannis Tsikelis

#include <huro/params.h>
#include <huro/quadruped_root_node.h>

#include <memory>
#include <string>

namespace huro {

using ImuStateMsg = QuadrupedRootNode::ImuStateMsg;
using MotorStateMsg = QuadrupedRootNode::MotorStateMsg;
using LowStateMsg = QuadrupedRootNode::LowStateMsg;
using OdometryMsg = QuadrupedRootNode::OdometryMsg;

using JointStateMsg = QuadrupedRootNode::JointStateMsg;
using TransformStamped = QuadrupedRootNode::TransformStamped;

using TransformBroadcaster = QuadrupedRootNode::TransformBroadcaster;

QuadrupedRootNode::QuadrupedRootNode(Params params)
    : Node("quad_root_node"), params_(params) {
  // Update topic names conditionally
  std::string ls_topic = params_.high_freq ? "/lowstate" : "/lf/lowstate";
  std::string odom_topic =
      params_.high_freq ? "/sportmodestate" : "/lf/sportmodestate";

  // Initialize the transform broadcaster
  tf_broadcaster_ = std::make_unique<TransformBroadcaster>(*this);

  // Set up publishers
  jointstate_pub_ = this->create_publisher<JointStateMsg>("/joint_states", 10);

  // Set up subscribers
  lowstate_sub_ = this->create_subscription<LowStateMsg>(
      ls_topic, 10,
      std::bind(&QuadrupedRootNode::LowStateHandler, this,
                std::placeholders::_1));
  odometry_sub_ = this->create_subscription<OdometryMsg>(
      odom_topic, 10,
      std::bind(&QuadrupedRootNode::OdometryHandler, this,
                std::placeholders::_1));
}

void QuadrupedRootNode::LowStateHandler(LowStateMsg::SharedPtr message) {
  if (params_.info_imu) {
    ImuStateMsg imu = message->imu_state;
    RCLCPP_INFO(this->get_logger(),
                "Euler angle -- roll: %f; pitch: %f; yaw: %f", imu.rpy[0],
                imu.rpy[1], imu.rpy[2]);
    RCLCPP_INFO(this->get_logger(),
                "Quaternion -- qw: %f; qx: %f; qy: %f; qz: %f",
                imu.quaternion[0], imu.quaternion[1], imu.quaternion[2],
                imu.quaternion[3]);
    RCLCPP_INFO(this->get_logger(), "Gyroscope -- wx: %f; wy: %f; wz: %f",
                imu.gyroscope[0], imu.gyroscope[1], imu.gyroscope[2]);
    RCLCPP_INFO(this->get_logger(), "Accelerometer -- ax: %f; ay: %f; az: %f",
                imu.accelerometer[0], imu.accelerometer[1],
                imu.accelerometer[2]);
  }
  if (params_.info_motors) {
    for (size_t i = 0; i < params_.n_motors; ++i) {
      MotorStateMsg motor = message->motor_state[i];
      RCLCPP_INFO(this->get_logger(),
                  "Motor state -- num: %ld; q: %f; dq: %f; ddq: %f; tau: %f", i,
                  motor.q, motor.dq, motor.ddq, motor.tau_est);
    }
  }

  JointStateMsg jointstate_msg;
  jointstate_msg.header.stamp = this->now();
  for (size_t i = 0; i < params_.n_motors; ++i) {
    std::string joint_name = params_.joint_names[i];
    jointstate_msg.name.push_back(joint_name);
    jointstate_msg.position.push_back(message->motor_state[i].q);
    jointstate_msg.velocity.push_back(message->motor_state[i].dq);
    jointstate_msg.effort.push_back(message->motor_state[i].tau_est);
  }

  std::cout << jointstate_msg.position[0] << std::endl;

  jointstate_pub_->publish(jointstate_msg);
}

void QuadrupedRootNode::OdometryHandler(OdometryMsg::SharedPtr message) {
  TransformStamped tf = GenerateTransformMsg(message);
  tf_broadcaster_->sendTransform(tf);
}

TransformStamped
QuadrupedRootNode::GenerateTransformMsg(OdometryMsg::SharedPtr message) {
  TransformStamped tf;

  tf.header.stamp = this->get_clock()->now();
  tf.header.frame_id = "world";
  tf.child_frame_id = "base";

  tf.transform.translation.x = message->position[0];
  tf.transform.translation.y = message->position[1];
  tf.transform.translation.z = message->position[2];

  tf.transform.rotation.w = message->imu_state.quaternion[0];
  tf.transform.rotation.x = message->imu_state.quaternion[1];
  tf.transform.rotation.y = message->imu_state.quaternion[2];
  tf.transform.rotation.z = message->imu_state.quaternion[3];

  return tf;
}
} // namespace huro
