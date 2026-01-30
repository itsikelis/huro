#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.qos import QoSProfile
from rclpy.node import Node
from unitree_hg.msg import HandCmd, HandState, MotorCmd
from geometry_msgs.msg import PointStamped 

CLOSE_RIGHT_VALUES = [-0.10, 0.63, -1.74, 1.06, 0.95, 0.91, 1.22]
CLOSE_LEFT_VALUES  = [0.04,  -0.04,  1.51, -1.10, -1.47, -1.13, -1.23]
OPEN_VALUES        = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

HAND_NAMES = {
    "thumb_0": 0,
    "thumb_1": 1,
    "thumb_2": 2,
    "middle_0": 3,
    "middle_1": 4,
    "index_0": 5,
    "index_1": 6,
}

class DX3Controller(Node):
    def __init__(self):
        super().__init__('dx3_hand_controller')

        self.right_target = OPEN_VALUES
        self.left_target = OPEN_VALUES
        self.total_motors = 7
        self.send_commands = True

        self.right_gripper_publisher_cmd = self.create_publisher(
            HandCmd, '/dex3/right/cmd', 10)
        self.right_subscriber_state = self.create_subscription(
            HandState, '/lf/dex3/right/state', self.right_callback, QoSProfile(depth=10))
        self.right_gripper_command_subscriber = self.create_subscription(
            PointStamped,
            '/right_hand/dx3/action',
            self.right_gripper_command_callback,
            QoSProfile(depth=10)
        )
        
        self.left_gripper_publisher_cmd = self.create_publisher(
            HandCmd, '/dex3/left/cmd', 10)
        self.left_subscriber_state = self.create_subscription(
            HandState, '/lf/dex3/left/state', self.left_callback, QoSProfile(depth=10))
        self.left_gripper_command_subscriber = self.create_subscription(
            PointStamped,
            '/left_hand/dx3/action',
            self.left_gripper_command_callback,
            QoSProfile(depth=10)
        )

        self.create_timer(0.05, self.publish_commands)

    def right_gripper_command_callback(self, msg):
        if msg.point.x > 0.5:
            self.right_target = CLOSE_RIGHT_VALUES
        else:
            self.right_target = OPEN_VALUES

    def left_gripper_command_callback(self, msg):
        if msg.point.x > 0.5:
            self.left_target = CLOSE_LEFT_VALUES
        else:
            self.left_target = OPEN_VALUES

    def left_callback(self, msg):
        positions = []
        for i in range(len(msg.motor_state)):
            positions.append(float(msg.motor_state[i].q))
        if  self.send_commands:
            return
        self.get_logger().debug(f'Left hand positions: {positions}')

    def right_callback(self, msg):
        positions = []
        for i in range(len(msg.motor_state)):
            positions.append(float(msg.motor_state[i].q))
        if  self.send_commands:
            return
        self.get_logger().info(f'Right hand positions: {positions}')   

    def create_cmd(self, values):
        cmd = HandCmd()
        cmd.motor_cmd = []

        n = min(self.total_motors, len(values))
        for i in range(n):
            m = MotorCmd()
            m.mode = 0
            m.q = float(values[i])
            m.dq = 0.0
            m.tau = 0.0
            m.kp = 1.5
            m.kd = 0.2
            cmd.motor_cmd.append(m)

        return cmd
    
    def publish_commands(self):
        if not self.send_commands:
            return
        
        right_cmd = self.create_cmd(self.right_target)
        self.right_gripper_publisher_cmd.publish(right_cmd)
        # left_cmd = self.create_cmd(self.left_target)
        # self.left_gripper_publisher_cmd.publish(left_cmd)
        


def main(args=None):
    rclpy.init(args=args)
    node = DX3Controller()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()