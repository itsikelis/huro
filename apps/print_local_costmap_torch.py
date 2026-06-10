#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import torch

# Print as many values as possible and avoid line wrapping in stdout.
torch.set_printoptions(threshold=10_000_000, linewidth=10_000, edgeitems=1_000)


class LocalCostmapTorchPrinter(Node):
    def __init__(self) -> None:
        super().__init__("local_costmap_torch_printer")
        self.subscription = self.create_subscription(
            OccupancyGrid,
            "/local_costmap/costmap",
            self._callback,
            10,
        )

    def _callback(self, msg: OccupancyGrid) -> None:
        height = int(msg.info.height)
        width = int(msg.info.width)

        tensor = torch.tensor(msg.data, dtype=torch.float32).reshape(height, width)

        # Normalize to [0, 1] range, mapping -1 to a neutral value (e.g., 0.5)
        tensor[tensor == -1] = 50
        tensor /= 100.0

        self.get_logger().info(
            f"Received OccupancyGrid: {height}x{width}, resolution={msg.info.resolution:.3f}"
        )
        self.get_logger().info(f"Sum of probabilities: {torch.sum(tensor)}")

        # Apply softmax to show a distribution that sums to 1
        softmax_tensor = torch.nn.functional.softmax(tensor.flatten(), dim=0).reshape(
            height, width
        )
        self.get_logger().info(f"Sum after softmax: {torch.sum(softmax_tensor)}")

        print("Original (normalized):")
        print(tensor)
        print("Softmax:")
        print(softmax_tensor)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LocalCostmapTorchPrinter()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
