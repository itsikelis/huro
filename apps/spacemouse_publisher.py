#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

# from geometry_msgs.msg import Twist
from huro.msg import SpaceMouseState
import pyspacemouse
from pyspacemouse.pyspacemouse import SpaceNavigator, DeviceSpec


class SpaceMousePublisher(Node):
    def __init__(self):
        super().__init__("spacemouse_publisher")
        self.tf_publisher = self.create_publisher(
            SpaceMouseState, "spacemouse_state", 10
        )
        self.timer = self.create_timer(0.005, self.timer_callback)
        try:
            self.device = pyspacemouse.open()
        except:
            self.get_logger().error("Failed to open SpaceMouse device.")

    def timer_callback(self):
        state = pyspacemouse.read()
        if state:
            msg = SpaceMouseState()
            msg.twist.linear.x = float(state.x)
            msg.twist.linear.y = float(state.y)
            msg.twist.linear.z = float(state.z)
            msg.twist.angular.x = float(state.roll)
            msg.twist.angular.y = float(state.pitch)
            msg.twist.angular.z = float(state.yaw)
            msg.button_1_pressed = bool(state.buttons[0])
            msg.button_2_pressed = bool(state.buttons[1])
            self.tf_publisher.publish(msg)
        else:
            msg = SpaceMouseState()
            msg.twist.linear.x = float(0.0)
            msg.twist.linear.y = float(0.0)
            msg.twist.linear.z = float(0.0)
            msg.twist.angular.x = float(1.0)
            msg.twist.angular.y = float(1.0)
            msg.twist.angular.z = float(0.0)
            msg.button_1_pressed = bool(False)
            msg.button_2_pressed = bool(False)
            self.tf_publisher.publish(msg)
            


def main(args=None):
    rclpy.init(args=args)
    spacemouse_publisher = SpaceMousePublisher()
    try:
        rclpy.spin(spacemouse_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        spacemouse_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
