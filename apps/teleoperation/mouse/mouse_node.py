#!/usr/bin/env python3
import math
import time
from dataclasses import dataclass

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool

import pyspacemouse
import numpy as np
from scipy.spatial.transform import Rotation as R


@dataclass
class SpaceMouseState:
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float
    left_button: int
    right_button: int


class MouseController:
    def __init__(self, device_info: str, name="spacemouse", dead_zone=0.2):
        self.name = name
        self.dead_zone = float(dead_zone)
        self.spacemouse = None

        self.spacemouse = pyspacemouse.open(device=device_info)

    def read_data(self):
        if self.spacemouse is None:
            return None

        s = self.spacemouse.read()
        if s is None:
            return None

        def dz(v):
            return 0.0 if abs(v) < self.dead_zone else v

        return SpaceMouseState(
            x=round(dz(s.y), 3),
            y=-round(dz(s.x), 3),
            z=round(dz(s.z), 3),
            roll=-round(getattr(s, "roll", 0.0), 3),
            pitch=round(getattr(s, "pitch", 0.0), 3),
            yaw=-round(getattr(s, "yaw", 0.0), 3),
            left_button=(s.buttons[0] if s.buttons else 0),
            right_button=(s.buttons[1] if s.buttons and len(s.buttons) > 1 else 0),
        )


class SpaceMouseRightHandNode(Node):
    def __init__(self):
        super().__init__("spacemouse_right_hand_node")

        # ---------------- Params ----------------
        self.declare_parameter("device_name", "SpaceMouse Wireless")
        self.declare_parameter("dead_zone", 0.2)

        self.declare_parameter("pose_topic", "/right_hand/pose_ref")
        self.declare_parameter("frame_id", "pelvis")

        self.declare_parameter("rate_hz", 200.0) 

        self.declare_parameter("linear_scale_mps", 0.25)
        self.declare_parameter("angular_scale_rps", 0.8)

        self.declare_parameter("deadman_required", True)

        self.declare_parameter("publish_buttons", True)
        self.declare_parameter("left_button_topic", "/spacemouse/right/left_button")
        self.declare_parameter("right_button_topic", "/spacemouse/right/right_button")

        device_name = self.get_parameter("device_name").value
        dead_zone = self.get_parameter("dead_zone").value

        try:
            self.controller = MouseController(device_info=device_name, name="right_spacemouse", dead_zone=dead_zone)
        except Exception as e:
            self.get_logger().error(f"Could not open SpaceMouse '{device_name}': {e}")
            raise

        pose_topic = self.get_parameter("pose_topic").value
        self.pose_pub = self.create_publisher(PoseStamped, pose_topic, 10)

        self.publish_buttons = bool(self.get_parameter("publish_buttons").value)
        if self.publish_buttons:
            self.left_pub = self.create_publisher(Bool, self.get_parameter("left_button_topic").value, 10)
            self.right_pub = self.create_publisher(Bool, self.get_parameter("right_button_topic").value, 10)

        self.frame_id = self.get_parameter("frame_id").value
        self.p = np.array([0.0, 0.0, 0.0], dtype=float)
        self.q = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)

        self.home_p = self.p.copy()
        self.home_q = self.q.copy()

        self.prev_right_button = 0

        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.dt = 1.0 / max(self.rate_hz, 1.0)
        self.last_time = time.time()

        self.lin_scale = float(self.get_parameter("linear_scale_mps").value)
        self.ang_scale = float(self.get_parameter("angular_scale_rps").value)
        self.deadman_required = bool(self.get_parameter("deadman_required").value)

        self.get_logger().info(
            f"SpaceMouseRightHandNode running. device='{device_name}', topic='{pose_topic}', frame='{self.frame_id}', rate={self.rate_hz}Hz"
        )

        self.timer = self.create_timer(self.dt, self.loop)

    def loop(self):
        now = time.time()
        dt = now - self.last_time
        if dt <= 0.0:
            dt = self.dt
        self.last_time = now

        st = self.controller.read_data()
        if st is None:
            return

        if self.publish_buttons:
            self.left_pub.publish(Bool(data=bool(st.left_button)))
            self.right_pub.publish(Bool(data=bool(st.right_button)))

        if st.right_button == 1 and self.prev_right_button == 0:
            self.p = self.home_p.copy()
            self.q = self.home_q.copy()

        self.prev_right_button = st.right_button

        if self.deadman_required and st.left_button == 0:
            self.publish_pose()
            return
        
        v = np.array([st.x, st.y, st.z], dtype=float) * self.lin_scale
        w = np.array([st.roll, st.pitch, st.yaw], dtype=float) * self.ang_scale

        self.p = self.p + v * dt

        droll, dpitch, dyaw = (w * dt).tolist()
        dR = R.from_euler("xyz", [droll, dpitch, dyaw], degrees=False)

        R_curr = R.from_quat(self.q)
        R_new = R_curr * dR
        self.q = R_new.as_quat()

        self.publish_pose()

    def publish_pose(self):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id

        msg.pose.position.x = float(self.p[0])
        msg.pose.position.y = float(self.p[1])
        msg.pose.position.z = float(self.p[2])

        msg.pose.orientation.x = float(self.q[0])
        msg.pose.orientation.y = float(self.q[1])
        msg.pose.orientation.z = float(self.q[2])
        msg.pose.orientation.w = float(self.q[3])

        self.pose_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = SpaceMouseRightHandNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
