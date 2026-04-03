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
        self.declare_parameter("publish_all_links", True)

        self.input_topic = self.get_parameter("input_topic").value
        self.world_frame = self.get_parameter("world_frame").value
        self.base_frame = self.get_parameter("base_frame").value
        self.model_name = self.get_parameter("model_name").value
        self.publish_all_links = self.get_parameter("publish_all_links").value

        self.tf_pub = self.create_publisher(TFMessage, "/tf", 50)
        self.tf_sub = self.create_subscription(
            TFMessage, self.input_topic, self.pose_cb, 50
        )

        self.get_logger().info(
            f"Bridge TF fix active: {self.input_topic} -> /tf ({self.world_frame}, model={self.model_name}, all_links={self.publish_all_links})"
        )

    def pose_cb(self, msg: TFMessage) -> None:
        # Keep only the latest transform per child frame.
        by_child = {}
        now_msg = self.get_clock().now().to_msg()

        for transform in msg.transforms:
            child_name = transform.child_frame_id
            if not child_name:
                continue

            child_leaf = child_name.split("::")[-1]

            # Ignore model-root frame (e.g., "go2") to avoid mixing world pose
            # with model-local link poses.
            if child_name == self.model_name or child_leaf == self.model_name:
                continue

            # Keep frame names usable by RViz and ignore world/root helper names.
            if child_leaf in ("world", self.world_frame, "", "__default__"):
                continue

            if not self.publish_all_links and child_leaf != self.base_frame:
                continue

            # Skip non-link helper frames that can pollute TF.
            if child_leaf.startswith("__") or child_leaf == "":
                continue

            # Gazebo Pose_V->TFMessage conversion may leave frame_id empty.
            # RViz discards such transforms, so enforce a valid world parent.
            transform.header.frame_id = self.world_frame
            transform.header.stamp = now_msg
            transform.child_frame_id = child_leaf
            by_child[child_leaf] = transform

        if by_child:
            tf_out = TFMessage()
            tf_out.transforms = list(by_child.values())
            self.tf_pub.publish(tf_out)


def main() -> None:
    rclpy.init()
    node = GzBaseTfPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
