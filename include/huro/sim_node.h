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
  SimNode() : Node("sim_node") {

    // Declare parameters used
    this->declare_parameter("robot_name", rclcpp::PARAMETER_STRING);
    this->declare_parameter("lowstate_topic", rclcpp::PARAMETER_STRING);
    this->declare_parameter("lowcmd_topic", rclcpp::PARAMETER_STRING);
    this->declare_parameter("odom_topic", rclcpp::PARAMETER_STRING);
    this->declare_parameter("xml_filename", rclcpp::PARAMETER_STRING);
    this->declare_parameter("base_link_name", rclcpp::PARAMETER_STRING);
    this->declare_parameter("sole_link_name", rclcpp::PARAMETER_STRING);
    this->declare_parameter("q_init", rclcpp::PARAMETER_DOUBLE_ARRAY);
    this->declare_parameter("sim_dt_ms", rclcpp::PARAMETER_INTEGER);
    this->declare_parameter("n_motors", rclcpp::PARAMETER_INTEGER);

    robot_name_ = this->get_parameter("robot_name").as_string();
    lowstate_topic_ = this->get_parameter("lowstate_topic").as_string();
    lowcmd_topic_ = this->get_parameter("lowcmd_topic").as_string();
    odom_topic_ = this->get_parameter("odom_topic").as_string();
    xml_filename_ = this->get_parameter("xml_filename").as_string();
    base_link_name_ = this->get_parameter("base_link_name").as_string();
    sole_link_name_ = this->get_parameter("sole_link_name").as_string();
    q_init_ = this->get_parameter("q_init").as_double_array();
    sim_dt_ms_ = static_cast<size_t>(this->get_parameter("sim_dt_ms").as_int());
    n_motors_ = static_cast<size_t>(this->get_parameter("n_motors").as_int());

    // Initialize publishers and subscripbers
    lowstate_pub_ = this->create_publisher<LowStateMsg>(lowstate_topic_, 10);
    odom_pub_ = this->create_publisher<OdometryMsg>(odom_topic_, 10);

    lowmcd_sub_ = this->create_subscription<LowCmdMsg>(
        lowcmd_topic_, 10,
        std::bind(&SimNode::LowCmdHandler, this, std::placeholders::_1));

    // 500Hz control loop
    timer_ = this->create_wall_timer(std::chrono::milliseconds(sim_dt_ms_),
                                     std::bind(&SimNode::Step, this));

    // Initialize robot
    auto xml_path = ament_index_cpp::get_package_share_directory("huro") +
                    "/resources/description_files/xml/" + xml_filename_;
    InitRobot(xml_path);

    // Flags and time keeping
    loop_count_ = 0;
    time_s_ = 0;
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
      time_s_ += static_cast<double>(sim_dt_ms_) / 1000.0;

      // Calculate control
      for (size_t i = 0; i < n_motors_; ++i) {
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
    for (size_t i = 0; i < n_motors_; ++i) {
      mj_data_->qpos[7 + i] = static_cast<mjtNum>(q_init_[i]);
      mj_data_->qvel[6 + i] = 0.0;
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
    for (size_t i = 0; i < n_motors_; ++i) {
      float q = static_cast<float>(mj_data_->qpos[7 + i]);
      float qdot = static_cast<float>(mj_data_->qvel[6 + i]);
      float qddot = static_cast<float>(mj_data_->qacc[6 + i]);

      lowstate.motor_state[i].q = q;
      lowstate.motor_state[i].dq = qdot;
      lowstate.motor_state[i].ddq = qddot;
    }

    // Foot contact forces from MuJoCo contact data for go2
    // if constexpr (robot_name_ == "go_2") {
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

    int pelvis_id = mj_name2id(mj_model_, mjOBJ_BODY, base_link_name_.c_str());
    int sole_id = mj_name2id(mj_model_, mjOBJ_BODY, sole_link_name_.c_str());

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

  std::string robot_name_;     // robot name (g1 or go2)
  std::string lowstate_topic_; // lostate topic name
  std::string lowcmd_topic_;   // lowcmd topic name
  std::string odom_topic_;     // Odometry topic name
  std::string xml_filename_;   // XML file description name
  std::string base_link_name_; // Base link name
  std::string sole_link_name_; // Sole link name (for z height calculation)
  std::vector<double> q_init_; // Initial joint position
  size_t sim_dt_ms_;           // Sim dt
  size_t n_motors_;            // Track loop count for logging
  double time_s_;              // Running time count (in seconds)
  size_t loop_count_;          // Track loop count for logging
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
