// Copyright 2025 Ioannis Tsikelis

#ifndef HURO_SIM_NODE_H_
#define HURO_SIM_NODE_H_

#include <mujoco/mujoco.h>

#include <array>
#include <memory>
#include <string>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rosgraph_msgs/msg/clock.hpp>
#include <unitree_go/msg/low_cmd.hpp>
#include <unitree_go/msg/low_state.hpp>
#include <unitree_go/msg/motor_cmd.hpp>
#include <unitree_go/msg/sport_mode_state.hpp>

namespace huro {
template <typename LowCmdMsg, typename LowStateMsg, typename OdometryMsg>
class SimNode : public rclcpp::Node {

public:
  struct SimNodeParams {
    std::string robot_name;
    std::string lowstate_topic_name;
    std::string lowcmd_topic_name;
    std::string odom_topic_name;
    std::string xml_filename;
    std::string base_link_name;
    std::string sole_link_name;
    std::vector<double> q_init;
    size_t sim_dt_ms;
    size_t n_motors;
    size_t n_gripper_motors;
  };

public:
  SimNode() : Node("sim_node") {

    // Declare parameters used
    this->declare_parameter("robot_name", rclcpp::PARAMETER_STRING);
    this->declare_parameter("lowstate_topic_name", rclcpp::PARAMETER_STRING);
    this->declare_parameter("lowcmd_topic_name", rclcpp::PARAMETER_STRING);
    this->declare_parameter("odom_topic_name", rclcpp::PARAMETER_STRING);
    this->declare_parameter("xml_filename", rclcpp::PARAMETER_STRING);
    this->declare_parameter("base_link_name", rclcpp::PARAMETER_STRING);
    this->declare_parameter("sole_link_name", rclcpp::PARAMETER_STRING);
    this->declare_parameter("q_init", rclcpp::PARAMETER_DOUBLE_ARRAY);
    this->declare_parameter("sim_dt_ms", rclcpp::PARAMETER_INTEGER);
    this->declare_parameter("n_motors", rclcpp::PARAMETER_INTEGER);
    this->declare_parameter("n_gripper_motors", rclcpp::PARAMETER_INTEGER);

    params_.robot_name = this->get_parameter("robot_name").as_string();
    params_.lowstate_topic_name =
        this->get_parameter("lowstate_topic_name").as_string();
    params_.lowcmd_topic_name =
        this->get_parameter("lowcmd_topic_name").as_string();
    params_.odom_topic_name =
        this->get_parameter("odom_topic_name").as_string();
    params_.xml_filename = this->get_parameter("xml_filename").as_string();
    params_.base_link_name = this->get_parameter("base_link_name").as_string();
    params_.sole_link_name = this->get_parameter("sole_link_name").as_string();
    params_.q_init = this->get_parameter("q_init").as_double_array();
    params_.sim_dt_ms =
        static_cast<size_t>(this->get_parameter("sim_dt_ms").as_int());
    params_.n_motors =
        static_cast<size_t>(this->get_parameter("n_motors").as_int());
    params_.n_gripper_motors =
        static_cast<size_t>(this->get_parameter("n_gripper_motors").as_int());

    // Initialize publishers and subscripbers
    lowstate_pub_ =
        this->create_publisher<LowStateMsg>(params_.lowstate_topic_name, 10);
    odom_pub_ =
        this->create_publisher<OdometryMsg>(params_.odom_topic_name, 10);

    lowmcd_sub_ = this->create_subscription<LowCmdMsg>(
        params_.lowcmd_topic_name, 10,
        std::bind(&SimNode::LowCmdHandler, this, std::placeholders::_1));

    // 500Hz control loop
    timer_ =
        this->create_wall_timer(std::chrono::milliseconds(params_.sim_dt_ms),
                                std::bind(&SimNode::Step, this));

    // Initialize robot
    auto xml_path = ament_index_cpp::get_package_share_directory("huro") +
                    "/resources/description_files/xml/" + params_.xml_filename;
    InitRobot(xml_path);

    // Flags and time keeping
    loop_count_ = 0;
  }

  ~SimNode() {}

protected:
  void Step() {
    if (low_cmd_ == nullptr) {
      loop_count_++;
      if (loop_count_ >= kLogInterval) {
        RCLCPP_INFO_STREAM(this->get_logger(), "Awaiting command");
        loop_count_ = 0;
      }
    } else {

      // Calculate control
      for (size_t i = 0; i < params_.n_motors; ++i) {
        RCLCPP_INFO_STREAM(this->get_logger(), "i: " << i);
        int motor_mode = static_cast<int>(low_cmd_->motor_cmd[i].mode);
        mjtNum q_des = static_cast<mjtNum>(low_cmd_->motor_cmd[i].q);
        mjtNum qdot_des = static_cast<mjtNum>(low_cmd_->motor_cmd[i].dq);
        mjtNum tau_ff = static_cast<mjtNum>(low_cmd_->motor_cmd[i].tau);
        mjtNum kp = static_cast<mjtNum>(low_cmd_->motor_cmd[i].kp);
        mjtNum kd = static_cast<mjtNum>(low_cmd_->motor_cmd[i].kd);

        mjtNum q_e = q_des - mj_data_->qpos[7 + i];
        mjtNum qdot_e = qdot_des - mj_data_->qvel[6 + i];

        mj_data_->ctrl[i] = motor_mode * (kp * q_e + kd * qdot_e + tau_ff);
        // mj_data_->ctrl[i] = 0.0;
        // RCLCPP_INFO(this->get_logger(), "u[%ld]: %f",i , mj_data_->ctrl[i]);
      }

      // Step the simulation
      mj_step(mj_model_, mj_data_);
    }

    // Publish the new state
    // OdomMsg: world frame base position and linear velocity
    // LowStateMsg: bodyframe base orientation and angular velocity and joint
    // state
    OdometryMsg odom_msg = GenerateOdometryMsg();
    LowStateMsg lowstate_msg = GenerateLowStateMsg();

    odom_pub_->publish(odom_msg);
    lowstate_pub_->publish(lowstate_msg);
  }

  void LowCmdHandler(std::shared_ptr<LowCmdMsg> message) {
    // Not used in simulation, also breaks go2 api comaptibility
    // mode_machine = (int)message->mode_machine;

    low_cmd_ = message;
  }

  void InitRobot(const std::string &xml_path) {
    // Load robot in MuJoCo
    mj_model_ = mj_loadXML(xml_path.c_str(), nullptr, nullptr, 0);
    if (!mj_model_) {
      std::string error_msg = "Mujoco XML Model Loading. The XML path is: \n";
      RCLCPP_ERROR_STREAM(this->get_logger(), error_msg << xml_path);
    }
    mj_data_ = mj_makeData(mj_model_);

    // Set joint positions
    for (size_t i = 0; i < params_.n_motors; ++i) {
      mj_data_->qpos[7 + i] = static_cast<mjtNum>(params_.q_init[i]);
      mj_data_->qvel[6 + i] = 0.0;
    }

    for (size_t i = 0; i < params_.n_gripper_motors; ++i) {
      mj_data_->qpos[7 + params_.n_motors + i] =
          static_cast<mjtNum>(params_.q_init[params_.n_motors + i]);
      mj_data_->qvel[6 + params_.n_motors + i] = 0.0;
    }

    // Place robot on the floor
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

    // Angular velocity
    float omegax = static_cast<float>(mj_data_->qvel[3]);
    float omegay = static_cast<float>(mj_data_->qvel[4]);
    float omegaz = static_cast<float>(mj_data_->qvel[5]);
    lowstate.imu_state.gyroscope[0] = omegax;
    lowstate.imu_state.gyroscope[1] = omegay;
    lowstate.imu_state.gyroscope[2] = omegaz;

    // Motor states
    for (size_t i = 0; i < params_.n_motors; ++i) {
      float q = static_cast<float>(mj_data_->qpos[7 + i]);
      float qdot = static_cast<float>(mj_data_->qvel[6 + i]);
      float qddot = static_cast<float>(mj_data_->qacc[6 + i]);

      lowstate.motor_state[i].q = q;
      lowstate.motor_state[i].dq = qdot;
      lowstate.motor_state[i].ddq = qddot;
    }

    // Foot contact forces from MuJoCo contact data for go2
    // if constexpr (params_.robot_name == "go_2") {
    //   for (size_t i = 0; i < 4; ++i) {
    //     lowstate.foot_force[i] = 0;
    //     lowstate.foot_force_est[i] = 0;
    //   }
    //
    //   // Get foot geom IDs for Go2 (collisions geometries)
    //   std::vector<std::string> foot_geom_names = {"FL", "FR", "RL", "RR"};
    //
    //   // Sum contact forces for each foot
    //   for (size_t i = 0; i < mj_data_->ncon; ++i) {
    //     const mjContact &con = mj_data_->contact[i];
    //
    //     // Check contact
    //     for (size_t foot_idx = 0; foot_idx < foot_geom_names.size();
    //          ++foot_idx) {
    //       int foot_geom_id = mj_name2id(mj_model_, mjOBJ_GEOM,
    //                                     foot_geom_names[foot_idx].c_str());
    //
    //       if (con.geom1 == foot_geom_id || con.geom2 == foot_geom_id) {
    //         // Get contact force in world frame
    //         mjtNum contact_force[6];
    //         mj_contactForce(mj_model_, mj_data_, i, contact_force);
    //
    //         // Sum normal force magnitude (contact_force[0] is normal force
    //         in
    //         // contact frame)
    //         float normal_force =
    //         static_cast<float>(std::abs(contact_force[0]));
    //         lowstate.foot_force[foot_idx] +=
    //         static_cast<int16_t>(normal_force);
    //         lowstate.foot_force_est[foot_idx] +=
    //             static_cast<int16_t>(normal_force);
    //       }
    //     }
    //   }
    // }

    return lowstate;
  }

protected:
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
  const size_t kLogInterval = 500; // Logging frequency

  SimNodeParams params_;
  size_t loop_count_; // Track loop count for logging
  int mode_machine;

  std::shared_ptr<rclcpp::Publisher<LowStateMsg>> lowstate_pub_;
  std::shared_ptr<rclcpp::Publisher<OdometryMsg>> odom_pub_;
  std::shared_ptr<rclcpp::Subscription<LowCmdMsg>> lowmcd_sub_;
  std::shared_ptr<rclcpp::TimerBase> timer_;

  mjModel *mj_model_;
  mjData *mj_data_;

  std::shared_ptr<LowCmdMsg> low_cmd_{nullptr};
};
} // namespace huro
#endif // HURO_SIM_NODE_H_
