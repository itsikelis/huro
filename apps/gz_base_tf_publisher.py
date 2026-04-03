#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage


class GzBaseTfPublisher(Node):
    def __init__(self) -> None:
        super().__init__("gz_base_tf_publisher")

        self.declare_parameter("input_topic", "/gz_pose_tf")
        self.declare_parameter("world_frame", "world")
        self.declare_parameter("base_frame", "base")
        self.declare_parameter("model_name", "go2")

        self.input_topic = self.get_parameter("input_topic").value
        self.world_frame = self.get_parameter("world_frame").value
        self.base_frame = self.get_parameter("base_frame").value
        self.model_name = self.get_parameter("model_name").value

        self.tf_pub = self.create_publisher(TFMessage, "/tf", 50)
        self.tf_sub = self.create_subscription(
            TFMessage, self.input_topic, self.pose_cb, 50
        )

        self.get_logger().info(
            f"Bridge TF fix active: {self.input_topic} -> /tf ({self.world_frame}->{self.base_frame}, model={self.model_name})"
        )

    def pose_cb(self, msg: TFMessage) -> None:
        now_msg = self.get_clock().now().to_msg()
        chosen = None
        fallback = None

        for transform in msg.transforms:
            child_name = transform.child_frame_id
            if not child_name:
                continue
            child_leaf = child_name.split("::")[-1]

            if child_name == self.model_name or child_leaf == self.model_name:
                chosen = transform
                break
            if child_name == self.base_frame or child_leaf == self.base_frame:
                fallback = transform

        if chosen is None:
            chosen = fallback
        if chosen is None:
            return

        chosen.header.frame_id = self.world_frame
        chosen.header.stamp = now_msg
        chosen.child_frame_id = self.base_frame

        tf_out = TFMessage()
        tf_out.transforms = [chosen]
        self.tf_pub.publish(tf_out)


def main() -> None:
    rclpy.init()
    node = GzBaseTfPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
