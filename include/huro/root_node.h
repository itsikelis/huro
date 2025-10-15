// Copyright 2025 Ioannis Tsikelis

#ifndef HURO_ROOT_NODE_H_
#define HURO_ROOT_NODE_H_

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
#include <unitree_hg/msg/low_cmd.hpp>
#include <unitree_hg/msg/low_state.hpp>
#include <unitree_hg/msg/motor_cmd.hpp>

namespace huro {
/**
 * @class RootNode
 * @brief Root node of HURo, used for both Unitree quadrupeds and humanoids
 *
 * This class handles incoming low-level state messages from the robot hardware
 * (e.g., IMU, motor states), and publishes the equivalent ROS2 standard
 * messages.
 *
 * It supports both high-frequency and low-frequency data sources, and
 * optionally logs IMU and motor information.
 *
 * @tparam LowCmdMsg Type of the low-level command message.
 * @tparam ImuStateMsg Type of the IMU state sub-message.
 * @tparam MotorStateMsg Type of the motor state sub-message.
 * @tparam LowStateMsg Type of the low-level state message from the robot.
 * @tparam OdometryMsg Type of the odometry message from the robot.
 */
template <typename LowCmdMsg, typename ImuStateMsg, typename MotorStateMsg,
          typename LowStateMsg, typename OdometryMsg>
class RootNode : public rclcpp::Node {
public:
  using JointStateMsg = sensor_msgs::msg::JointState;
  using TransformStamped = geometry_msgs::msg::TransformStamped;

  using TransformBroadcaster = tf2_ros::TransformBroadcaster;

public:
  /**
   * @brief Constructs the RootNode.
   *
   * Initializes publishers, subscribers, and the transform broadcaster based on
   * the provided parameters.
   *
   * @param params A Params struct specified in huro/params.h
   */
  RootNode(Params params) : Node("root_node"), params_(params) {
    // Update topic names conditionally
    // TODO: Add topic names in params
    std::string ls_topic = params_.lowstate_topic_name;
    std::string odom_topic = params_.odom_topic_name;

    // Initialize the transform broadcaster
    tf_broadcaster_ = std::make_unique<TransformBroadcaster>(*this);

    // Set up publishers
    jointstate_pub_ =
        this->create_publisher<JointStateMsg>("/joint_states", 10);

    // Set up subscribers
    lowstate_sub_ = this->create_subscription<LowStateMsg>(
        ls_topic, 10,
        std::bind(&RootNode::LowStateHandler, this, std::placeholders::_1));
    odometry_sub_ = this->create_subscription<OdometryMsg>(
        odom_topic, 10,
        std::bind(&RootNode::OdometryHandler, this, std::placeholders::_1));
  }

  ~RootNode() {}

protected:
  /**
   * @brief Callback for handling incoming LowState messages.
   *
   * @param message Shared pointer to the incoming LowStateMsg.
   */
  void LowStateHandler(std::shared_ptr<LowStateMsg> message) {
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
                    "Motor state -- num: %ld; q: %f; dq: %f; ddq: %f; tau: %f",
                    i, motor.q, motor.dq, motor.ddq, motor.tau_est);
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

    jointstate_pub_->publish(jointstate_msg);
  }

  /**
   * @brief Callback for handling incoming Odometry messages.
   *
   * Converts the odometry message into a TransformStamped and broadcasts it.
   *
   * @param message Shared pointer to the incoming OdometryMsg.
   */
  void OdometryHandler(std::shared_ptr<OdometryMsg> message) {
    TransformStamped tf = GenerateTransformMsg(message);
    tf_broadcaster_->sendTransform(tf);
  }

  /**
   * @brief Generates a TransformStamped message from odometry data.
   *
   * Uses IMU orientation and position from the odometry message to construct a
   * transform from world to base link.
   *
   * @param message Shared pointer to the odometry message.
   * @return TransformStamped The generated transform message.
   */
  TransformStamped GenerateTransformMsg(std::shared_ptr<OdometryMsg> message) {
    TransformStamped tf;

    tf.header.stamp = this->get_clock()->now();
    tf.header.frame_id = "world";
    tf.child_frame_id = params_.base_link_name;

    tf.transform.translation.x = message->position[0];
    tf.transform.translation.y = message->position[1];
    tf.transform.translation.z = message->position[2];

    tf.transform.rotation.w = message->imu_state.quaternion[0];
    tf.transform.rotation.x = message->imu_state.quaternion[1];
    tf.transform.rotation.y = message->imu_state.quaternion[2];
    tf.transform.rotation.z = message->imu_state.quaternion[3];

    return tf;
  }

protected:
  Params params_;
  std::shared_ptr<rclcpp::Publisher<JointStateMsg>> jointstate_pub_;
  std::shared_ptr<rclcpp::TimerBase> timer_;
  std::shared_ptr<rclcpp::Subscription<LowStateMsg>> lowstate_sub_;
  std::shared_ptr<rclcpp::Subscription<OdometryMsg>> odometry_sub_;
  std::shared_ptr<TransformBroadcaster> tf_broadcaster_;
};
} // namespace huro
#endif // HURO_ROOT_NODE_H_
