#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.timer import Timer
import math

from unitree_api.msg import Request
from unitree_go.msg import LowCmd, LowState, IMUState, MotorState

from huro_py.crc_go import Crc

# Custom PRorAB enum or constant
# from hucebot_g1_ros.msg import MotorMode  # Assuming PRorAB is defined here

# from hucebot_g1_ros.config import GO2_NUM_MOTOR, Kp, Kd
# from hucebot_g1_ros.motor_crc_hg import get_crc  # Assuming Python version available

GO2_NUM_MOTOR = 12

Kp = [
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
]

Kd = [
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
]

q_init = [
    0.005,
    0.72,
    -1.4,
    -0.005,
    0.72,
    -1.4,
    -0.005,
    0.72,
    -1.4,
    0.005,
    0.72,
    -1.4,
]


class MoveExample(Node):
    def __init__(self):
        super().__init__("move_example")

        self.control_dt = 0.01  # 10ms
        self.timer_dt_ms = int(self.control_dt * 1000)
        self.time = 0.0
        self.init_duration_s = 5.0

        self.motors_on = 1

        self.imu = IMUState()
        self.motor = [MotorState() for _ in range(GO2_NUM_MOTOR)]

        self.topic_name = (
            "lowstate" if self.get_parameter_or("HIGH_FREQ", False) else "lf/lowstate"
        )

        self.lowcmd_pub = self.create_publisher(LowCmd, "/lowcmd", 10)
        self.lowstate_sub = self.create_subscription(
            LowState, self.topic_name, self.low_state_handler, 10
        )

        self.sport_pub = self.create_publisher(Request, "/api/sport/request", 10)
        ROBOT_SPORT_API_ID_STANDDOWN = 1005
        req = Request()
        req.header.identity.api_id = ROBOT_SPORT_API_ID_STANDDOWN
        self.sport_pub.publish(req)

        self.motion_pub = self.create_publisher(
            Request, "/api/motion_switcher/request", 10
        )
        ROBOT_MOTION_SWITCHER_API_RELEASEMODE = 1003
        req = Request()
        req.header.identity.api_id = ROBOT_MOTION_SWITCHER_API_RELEASEMODE
        self.motion_pub.publish(req)

        self.timer = self.create_timer(self.control_dt, self.control)

    def control(self):
        low_cmd = LowCmd()
        low_cmd.head[0] = 0xFE
        low_cmd.head[1] = 0xEF
        # low_cmd.levelFlag = 0xFF
        low_cmd.gpio = 0

        self.time += self.control_dt

        if self.time < self.init_duration_s:
            for i in range(GO2_NUM_MOTOR):
                ratio = self.clamp(self.time / self.init_duration_s, 0.0, 1.0)
                cmd = low_cmd.motor_cmd[i]
                cmd.mode = self.motors_on
                cmd.q = (1.0 - ratio) * self.motor[i].q + ratio * q_init[i]
                cmd.dq = 0.0
                cmd.tau = 0.0
                cmd.kp = Kp[i]
                cmd.kd = Kd[i]
        else:
            #####################
            # Control code here #
            #####################
            for i in range(GO2_NUM_MOTOR):
                cmd = low_cmd.motor_cmd[i]
                cmd.mode = self.motors_on
                cmd.q = q_init[i]
                cmd.dq = 0.0
                cmd.tau = 0.0
                cmd.kp = Kp[i]
                cmd.kd = Kd[i]

        low_cmd.crc = Crc(low_cmd)
        self.lowcmd_pub.publish(low_cmd)
        # self.get_logger().info("Published low_cmd")

    def low_state_handler(self, msg: LowState):
        # self.get_logger().info(str(self.motors_on))
        self.imu = msg.imu_state
        for i in range(GO2_NUM_MOTOR):
            self.motor[i] = msg.motor_state[i]

        ## Handle Controller Message
        self.controller_msg = msg.wireless_remote
        if self.controller_msg[3] == 1:
            self.motors_on = 0

    def clamp(self, value, low, high):
        if value < low:
            return low
        if value > high:
            return high
        return value


def main(args=None):
    rclpy.init(args=args)
    node = MoveExample()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
