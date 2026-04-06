#!/usr/bin/env python3

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose


class GoalPoseToNav2Relay(Node):
    def __init__(self) -> None:
        super().__init__("goal_pose_to_nav2_relay")
        self.declare_parameter("goal_pose_topic", "/goal_pose")
        self.declare_parameter("nav2_action_name", "/navigate_to_pose")

        goal_pose_topic = str(self.get_parameter("goal_pose_topic").value)
        nav2_action_name = str(self.get_parameter("nav2_action_name").value)

        self._client = ActionClient(self, NavigateToPose, nav2_action_name)
        self._sub = self.create_subscription(
            PoseStamped, goal_pose_topic, self._on_goal_pose, 10
        )

        self.get_logger().info(
            f"Goal relay ready: {goal_pose_topic} -> {nav2_action_name}"
        )

    def _on_goal_pose(self, msg: PoseStamped) -> None:
        if not self._client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn("NavigateToPose action server not available yet")
            return

        goal = NavigateToPose.Goal()
        goal.pose = msg

        send_future = self._client.send_goal_async(goal)
        send_future.add_done_callback(self._on_goal_response)

        self.get_logger().info(
            f"Forwarded goal: frame={msg.header.frame_id}, x={msg.pose.position.x:.2f}, y={msg.pose.position.y:.2f}"
        )

    def _on_goal_response(self, future) -> None:
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().warn("NavigateToPose goal rejected")
            return
        self.get_logger().info("NavigateToPose goal accepted")


def main() -> None:
    rclpy.init()
    node = GoalPoseToNav2Relay()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
