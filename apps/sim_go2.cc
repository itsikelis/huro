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

  auto node = std::make_shared<
      huro::SimNode<unitree_go::msg::LowCmd, unitree_go::msg::LowState,
                    unitree_go::msg::SportModeState>>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
