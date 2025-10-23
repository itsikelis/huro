#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.timer import Timer
import math

from unitree_hg.msg import LowCmd, LowState, IMUState, MotorState

from huro.crc_hg import Crc

# Custom PRorAB enum or constant
# from hucebot_g1_ros.msg import MotorMode  # Assuming PRorAB is defined here

# from hucebot_g1_ros.config import G1_NUM_MOTOR, Kp, Kd
# from hucebot_g1_ros.motor_crc_hg import get_crc  # Assuming Python version available

G1_NUM_MOTOR = 29

Kp = [
    60.0,
    60.0,
    60.0,
    100.0,
    40.0,
    40.0,  # legs
    60.0,
    60.0,
    60.0,
    100.0,
    40.0,
    40.0,  # legs
    60.0,
    40.0,
    40.0,  # waist
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,  # arms
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,  # arms
]

Kd = [
    1.0,
    1.0,
    1.0,
    2.0,
    1.0,
    1.0,  # legs
    1.0,
    1.0,
    1.0,
    2.0,
    1.0,
    1.0,  # legs
    1.0,
    1.0,
    1.0,  # waist
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,  # arms
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,  # arms
]

q_init = [0.0 for _ in range(G1_NUM_MOTOR)]
q_init[0] = -0.6
q_init[3] = 1.2
q_init[4] = -0.6
q_init[6] = -0.6
q_init[9] = 1.2
q_init[10] = -0.6


class Mode:
    PR = 0  # Series Control for Pitch/Roll Joints
    AB = 1  # Parallel Control for A/B Joints


class MoveExample(Node):
    def __init__(self):
        super().__init__("move_example")

        self.control_dt = 0.01  # 10ms
        self.timer_dt_ms = int(self.control_dt * 1000)
        self.time = 0.0
        self.init_duration_s = 3.0

        self.mode_ = Mode.PR
        self.mode_machine = 0

        self.motors_on = 1

        self.imu = IMUState()
        self.motor = [MotorState() for _ in range(G1_NUM_MOTOR)]

        self.topic_name = (
            "lowstate" if self.get_parameter_or("HIGH_FREQ", False) else "lf/lowstate"
        )

        self.lowcmd_pub = self.create_publisher(LowCmd, "/lowcmd", 10)
        self.lowstate_sub = self.create_subscription(
            LowState, self.topic_name, self.low_state_handler, 10
        )

        self.timer = self.create_timer(self.control_dt, self.control)

    def control(self):
        low_cmd = LowCmd()
        self.time += self.control_dt

        low_cmd.mode_pr = self.mode_
        low_cmd.mode_machine = self.mode_machine

        if self.time < self.init_duration_s:
            for i in range(G1_NUM_MOTOR):
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
            for i in range(G1_NUM_MOTOR):
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
        self.mode_machine = msg.mode_machine
        self.imu = msg.imu_state
        for i in range(G1_NUM_MOTOR):
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
