#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.timer import Timer
import math
import numpy as np

from unitree_api.msg import Request
from unitree_go.msg import LowCmd, LowState, IMUState, MotorState
from sensor_msgs.msg import JointState, Joy

from huro_py.crc_go import Crc

# Custom PRorAB enum or constant
# from hucebot_g1_ros.msg import MotorMode  # Assuming PRorAB is defined here

# from hucebot_g1_ros.config import GO2_NUM_MOTOR, Kp, Kd
# from hucebot_g1_ros.motor_crc_hg import get_crc  # Assuming Python version available

GO2_NUM_MOTOR = 12

Kp = np.ones(GO2_NUM_MOTOR) * 50.

Kd =  np.ones(GO2_NUM_MOTOR) * 1.

q_init = [
    0.05,
    0.72,
    -1.4,
    -0.05,
    0.72,
    -1.4,
    -0.05,
    0.72,
    -1.4,
    0.05,
    0.72,
    -1.4,
]

class MoveExample(Node):
    def __init__(self):
        super().__init__("move_example")

        self.control_dt = 0.01  # 10ms
        self.timer_dt_ms = int(self.control_dt * 1000)
        self.time = 0.0
        self.init_duration_s = 3.0
        self.mpc_control_dt = 0.03
        self.q_start = None
        self.first_command = False

        self.motors_on = 1

        self.imu = IMUState()
        self.motor = [MotorState() for _ in range(GO2_NUM_MOTOR)]

        self.topic_name = (
            "lowstate" if self.get_parameter_or("HIGH_FREQ", True) else "lf/lowstate"
        )

        self.lowcmd_pub = self.create_publisher(LowCmd, "/lowcmd", 10)
        self.lowstate_sub = self.create_subscription(
            LowState, self.topic_name, self.low_state_handler, 10
        )

        self.command_sub = self.create_subscription(
            JointState, "/joint_commands", self.joint_commands_handler, 10
        )

        self.joy_subsriber = self.create_subscription(
            Joy,             # message type
            '/joy',      # topic name
            self.joy_callback,      # callback function
            10                       # QoS (queue size)
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


        if self.time < self.init_duration_s:
            for i in range(GO2_NUM_MOTOR):
                ratio = self.clamp(self.time / self.init_duration_s, 0.0, 1.0)
                cmd = low_cmd.motor_cmd[i]
                cmd.mode = self.motors_on
                cmd.q = (1.0 - ratio) * self.q_start[i] + ratio * q_init[i]
                cmd.dq = 0.0
                cmd.tau = 0.0
                cmd.kp = 200.
                cmd.kd = Kd[i]
        elif not self.first_command:
            for i in range(GO2_NUM_MOTOR):
                cmd = low_cmd.motor_cmd[i]
                cmd.mode = self.motors_on
                cmd.q = q_init[i]
                cmd.dq = 0.0
                cmd.tau = 0.0
                cmd.kp = 200.
                cmd.kd = Kd[i]
        else:
            if np.linalg.norm(np.array(self.last_joint_commands.velocity)-np.array(self.joint_commands.velocity)) >=15.:
                self.get_logger().info("Emergency STOP")
                self.motors_on = 0
            ratio = self.clamp((self.time-self.last_cmd_time) / self.mpc_control_dt, 0.0, 1.0)
            for i in range(GO2_NUM_MOTOR):
                cmd = low_cmd.motor_cmd[i]
                cmd.mode = self.motors_on
                cmd.q =  (1.0 - ratio) * self.last_joint_commands.position[i] + ratio * self.joint_commands.position[i]
                cmd.dq =  (1.0 - ratio) * self.last_joint_commands.velocity[i] + ratio * self.joint_commands.velocity[i]
                # cmd.tau = (1.0 - ratio) * self.last_joint_commands.effort[i] + ratio * self.joint_commands.effort[i]
                cmd.tau = self.joint_commands.effort[i]
                cmd.kp = Kp[i]
                cmd.kd = Kd[i]
        
        self.time += self.control_dt

        low_cmd.crc = Crc(low_cmd)
        self.lowcmd_pub.publish(low_cmd)
        # self.get_logger().info("Published low_cmd")

    def low_state_handler(self, msg: LowState):
        # self.get_logger().info("Readed Low_state")
        self.imu = msg.imu_state
        for i in range(GO2_NUM_MOTOR):
            self.motor[i] = msg.motor_state[i]

        if self.q_start is None:
            self.q_start = [self.motor[i].q for i in range(GO2_NUM_MOTOR)]
            self.get_logger().info(f"Initialized q_start")

            self.joint_commands = JointState()
            self.joint_commands.position = q_init
            self.joint_commands.velocity = [0.]*(GO2_NUM_MOTOR)
            self.joint_commands.effort = [0.]*(GO2_NUM_MOTOR)


        ## Handle Controller Message
        self.controller_msg = msg.wireless_remote
        if self.controller_msg[3] == 1:
            self.motors_on = 0

    def joint_commands_handler(self, msg: JointState):
        if not self.first_command:
            self.first_command = True
        
        self.last_cmd_time = self.time
        self.last_joint_commands = self.joint_commands
        self.joint_commands = msg

    def joy_callback(self, msg: Joy):
        if msg.buttons[10]==1:
            self.motors_on = 0
        if msg.buttons[10]==1 and msg.buttons[9]==1:
            self.motors_on = 1
            self.time=0.
            self.q_start = [self.motor[i].q for i in range(GO2_NUM_MOTOR)]
        


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
