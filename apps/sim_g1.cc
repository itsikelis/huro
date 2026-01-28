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

  auto node = std::make_shared<
      huro::SimNode<unitree_hg::msg::LowCmd, unitree_hg::msg::LowState,
                    unitree_go::msg::SportModeState>>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
