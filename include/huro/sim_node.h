// Copyright 2025 Ioannis Tsikelis

#ifndef HURO_SIM_NODE_H_
#define HURO_SIM_NODE_H_

#include <huro/params.h>

#include <mujoco/mujoco.h>

#include <array>
#include <memory>
#include <string>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <rclcpp/rclcpp.hpp>
#include <unitree_go/msg/low_cmd.hpp>
#include <unitree_go/msg/low_state.hpp>
#include <unitree_go/msg/motor_cmd.hpp>
#include <unitree_go/msg/sport_mode_state.hpp>

namespace huro {
template <typename LowCmdMsg, typename LowStateMsg, typename OdometryMsg,
          size_t N_MOTORS>
class SimNode : public rclcpp::Node {
public:
  SimNode(Params params) : Node("sim_node"), params_(params) {
    std::string ls_topic = params_.lowstate_topic_name;
    std::string odom_topic = params_.odom_topic_name;

    std::string xml_path =
        ament_index_cpp::get_package_share_directory("huro") +
        "/resources/description_files/xml/" + params_.xml_filename;

    mj_model_ = mj_loadXML(xml_path.c_str(), nullptr, nullptr, 0);
    if (!mj_model_) {
      std::string error_msg = "Mujoco XML Model Loading. The XML path is: \n";
      RCLCPP_ERROR_STREAM(this->get_logger(), error_msg << xml_path);
    }
    mj_data_ = mj_makeData(mj_model_);

    mjtNum z_dist = GetZDistanceFromSoleToBaseLink();
    mj_data_->qpos[0] = 0.;
    mj_data_->qpos[1] = 0.;
    mj_data_->qpos[2] = z_dist;
    mj_data_->qpos[3] = 1.;
    mj_data_->qpos[4] = 0.;
    mj_data_->qpos[5] = 0.;
    mj_data_->qpos[6] = 0.;

    mj_data_->qvel[0] = 0.;
    mj_data_->qvel[1] = 0.;
    mj_data_->qvel[2] = 0.;
    mj_data_->qvel[3] = 0.;
    mj_data_->qvel[4] = 0.;
    mj_data_->qvel[5] = 0.;

    // Initialize vectors
    for (size_t i = 0; i < N_MOTORS; ++i) {
      q_des_[i] = 0.0;
      qdot_des_[i] = 0.0;
      tau_ff_[i] = 0.0;
      kp_[i] = 0.0;
      kd_[i] = 0.0;
    }

    lowstate_pub_ = this->create_publisher<LowStateMsg>(ls_topic, 10);
    odom_pub_ = this->create_publisher<OdometryMsg>(odom_topic, 10);

    lowmcd_sub_ = this->create_subscription<LowCmdMsg>(
        params_.lowcmd_topic_name, 10,
        std::bind(&SimNode::LowCmdHandler, this, std::placeholders::_1));

    // 500Hz control loop
    timer_ =
        this->create_wall_timer(std::chrono::milliseconds(params_.sim_dt_ms),
                                std::bind(&SimNode::Step, this));

    // Running time count
    time_s_ = 0;
  }

  ~SimNode() {}

protected:
  void Step() {
    time_s_ += params_.sim_dt_ms / 1000;

    if (params_.fix_base) {
      FixModelBase();
    }

    // Calculate control
    for (size_t i = 0; i < N_MOTORS; ++i) {
      mjtNum q_e = q_des_[i] - mj_data_->qpos[7 + i];
      mjtNum qdot_e = qdot_des_[i] - mj_data_->qvel[6 + i];

      mj_data_->ctrl[i] = kp_[i] * q_e + kd_[i] * qdot_e + tau_ff_[i];
      // mj_data_->ctrl[i] = 0.0;
    }

    // Step the simulation
    mj_step(mj_model_, mj_data_);

    // Publish the new state
    // OdomMsg: world frame base position and linear velocity
    // LowStateMsg: bodyframe base orientation and angular velocity and
    // joint state
    OdometryMsg odom_msg = GenerateOdometryMsg();
    LowStateMsg lowstate_msg = GenerateLowStateMsg();

    odom_pub_->publish(odom_msg);
    lowstate_pub_->publish(lowstate_msg);
  }

  void LowCmdHandler(std::shared_ptr<LowCmdMsg> message) {
    // Not used in simulation, also breaks go2 api comaptibility
    // mode_machine = (int)message->mode_machine;

    for (size_t i = 0; i < N_MOTORS; ++i) {
      q_des_[i] = static_cast<mjtNum>(message->motor_cmd[i].q);
      qdot_des_[i] = static_cast<mjtNum>(message->motor_cmd[i].dq);
      tau_ff_[i] = static_cast<mjtNum>(message->motor_cmd[i].tau);
      kp_[i] = static_cast<mjtNum>(message->motor_cmd[i].kp);
      kd_[i] = static_cast<mjtNum>(message->motor_cmd[i].kd);
    }
  }

  void FixModelBase() {
    mj_data_->qpos[0] = 0.;
    mj_data_->qpos[1] = 0.;
    mj_data_->qpos[2] = 1.;
    mj_data_->qpos[3] = 1.;
    mj_data_->qpos[4] = 0.;
    mj_data_->qpos[5] = 0.;
    mj_data_->qpos[6] = 0.;

    mj_data_->qvel[0] = 0.;
    mj_data_->qvel[1] = 0.;
    mj_data_->qvel[2] = 0.;
    mj_data_->qvel[3] = 0.;
    mj_data_->qvel[4] = 0.;
    mj_data_->qvel[5] = 0.;
  }

  OdometryMsg GenerateOdometryMsg() const {
    OdometryMsg odom;

    odom.position[0] = static_cast<float>(mj_data_->qpos[0]);
    odom.position[1] = static_cast<float>(mj_data_->qpos[1]);
    odom.position[2] = static_cast<float>(mj_data_->qpos[2]);

    odom.velocity[0] = static_cast<float>(mj_data_->qvel[0]);
    odom.velocity[1] = static_cast<float>(mj_data_->qvel[1]);
    odom.velocity[2] = static_cast<float>(mj_data_->qvel[2]);

    float qw = static_cast<float>(mj_data_->qpos[3]);
    float qx = static_cast<float>(mj_data_->qpos[4]);
    float qy = static_cast<float>(mj_data_->qpos[5]);
    float qz = static_cast<float>(mj_data_->qpos[6]);
    odom.imu_state.quaternion[0] = qw;
    odom.imu_state.quaternion[1] = qx;
    odom.imu_state.quaternion[2] = qy;
    odom.imu_state.quaternion[3] = qz;

    // Angular velocity
    float omegax = static_cast<float>(mj_data_->qvel[3]);
    float omegay = static_cast<float>(mj_data_->qvel[4]);
    float omegaz = static_cast<float>(mj_data_->qvel[5]);
    odom.imu_state.gyroscope[0] = omegax;
    odom.imu_state.gyroscope[1] = omegay;
    odom.imu_state.gyroscope[2] = omegaz;

    return odom;
  }
  LowStateMsg GenerateLowStateMsg() const {
    LowStateMsg lowstate;

    // Rotation
    float qx = static_cast<float>(mj_data_->qpos[3]);
    float qy = static_cast<float>(mj_data_->qpos[4]);
    float qz = static_cast<float>(mj_data_->qpos[5]);
    float qw = static_cast<float>(mj_data_->qpos[6]);
    lowstate.imu_state.quaternion[0] = qx;
    lowstate.imu_state.quaternion[1] = qy;
    lowstate.imu_state.quaternion[2] = qz;
    lowstate.imu_state.quaternion[3] = qw;

    // angular velocity
    float omegax = static_cast<float>(mj_data_->qvel[3]);
    float omegay = static_cast<float>(mj_data_->qvel[4]);
    float omegaz = static_cast<float>(mj_data_->qvel[5]);
    lowstate.imu_state.gyroscope[0] = omegax;
    lowstate.imu_state.gyroscope[1] = omegay;
    lowstate.imu_state.gyroscope[2] = omegaz;

    // Motor States
    for (size_t i = 0; i < N_MOTORS; ++i) {
      float q = static_cast<float>(mj_data_->qpos[7 + i]);
      float qdot = static_cast<float>(mj_data_->qvel[6 + i]);
      float qddot = static_cast<float>(mj_data_->qacc[6 + i]);

      lowstate.motor_state[i].q = q;
      lowstate.motor_state[i].dq = qdot;
      lowstate.motor_state[i].ddq = qddot;
    }

    return lowstate;
  }

  mjtNum GetZDistanceFromSoleToBaseLink() {
    mj_fwdPosition(mj_model_, mj_data_);

    int pelvis_id =
        mj_name2id(mj_model_, mjOBJ_BODY, params_.base_link_name.c_str());
    int sole_id =
        mj_name2id(mj_model_, mjOBJ_BODY, params_.sole_link_name.c_str());

    if (pelvis_id == -1 || sole_id == -1) {
      std::string msg = "Invalid body name(s) during model z calculation";
      RCLCPP_ERROR(this->get_logger(), msg.c_str());
      return 0.0;
    }

    const mjtNum *pelvis_pos = mj_data_->xpos + 3 * pelvis_id;
    const mjtNum *sole_pos = mj_data_->xpos + 3 * sole_id;

    return pelvis_pos[2] - sole_pos[2];
  }

protected:
  Params params_;

  double time_s_; // Running time count (in seconds)
  int mode_machine;

  std::shared_ptr<rclcpp::Publisher<LowStateMsg>> lowstate_pub_;
  std::shared_ptr<rclcpp::Publisher<OdometryMsg>> odom_pub_;
  std::shared_ptr<rclcpp::Subscription<LowCmdMsg>> lowmcd_sub_;
  std::shared_ptr<rclcpp::TimerBase> timer_;

  mjModel *mj_model_;
  mjData *mj_data_;

  std::array<double, N_MOTORS> q_des_;
  std::array<double, N_MOTORS> qdot_des_;
  std::array<double, N_MOTORS> tau_ff_;
  std::array<double, N_MOTORS> kp_;
  std::array<double, N_MOTORS> kd_;
};
} // namespace huro
#endif // HURO_SIM_NODE_H_
