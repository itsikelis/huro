// Copyright 2025 Ioannis Tsikelis

#include <huro/sim_node.h>

#include <cstdio>
#include <memory>

#include <rclcpp/rclcpp.hpp>

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<huro::SimNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
