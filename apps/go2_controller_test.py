#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import torch

from unitree_api.msg import Request
from unitree_go.msg import IMUState, LowCmd, LowState, MotorState

from huro_py.crc_go import Crc
from huro_py.utils import process_height_map_rotated


torch.set_printoptions(precision=2, sci_mode=False, linewidth=10_000, threshold=1_000_000)


GO2_NUM_MOTOR = 12
KP = [60.0] * GO2_NUM_MOTOR
KD = [5.0] * GO2_NUM_MOTOR

# Nominal standing pose used by existing examples.
STAND_Q = [
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


class Go2ControllerTest(Node):
    def __init__(self) -> None:
        super().__init__("go2_controller_test")

        self.control_dt = 0.01
        self.ramp_duration_s = 5.0
        self.pitch_period_s = 10.0

        self.time_s = 0.0
        self.last_log_s = -1.0

        self.motors_on = 1
        self.received_state = False
        self.captured_start_pose = False
        self.lowstate_msg = None

        self.imu = IMUState()
        self.motor = [MotorState() for _ in range(GO2_NUM_MOTOR)]
        self.start_q = [0.0] * GO2_NUM_MOTOR

        self.x_range = [1.0, -0.5]
        self.y_range = [-0.5, 0.5]
        self.res = 0.1
        self.height_map = torch.zeros((3, 15, 10), dtype=torch.float32)
        self.lidar_frame_count = 0
        self.lidar_print_stride = 5

        self.lowcmd_pub = self.create_publisher(LowCmd, "/lowcmd", 10)
        self.lowstate_sub = self.create_subscription(
            LowState, "/lowstate", self.low_state_handler, 10
        )
        self.lidar_sub = self.create_subscription(
            PointCloud2, "/utlidar/cloud", self.lidar_handler, 10
        )

        self.sport_pub = self.create_publisher(Request, "/api/sport/request", 10)
        self.motion_pub = self.create_publisher(
            Request, "/api/motion_switcher/request", 10
        )

        self._send_sport_stand_down()
        self._send_motion_release_mode()

        self.timer = self.create_timer(self.control_dt, self.control)

    def _send_sport_stand_down(self) -> None:
        req = Request()
        req.header.identity.api_id = 1005
        self.sport_pub.publish(req)

    def _send_motion_release_mode(self) -> None:
        req = Request()
        req.header.identity.api_id = 1003
        self.motion_pub.publish(req)

    def _pitch_pose(self, pitch_ratio: float):
        target = list(STAND_Q)

        # +pitch_ratio tilts body forward, -pitch_ratio tilts body backward.
        front_thigh_delta = 0.10 * pitch_ratio
        front_calf_delta = -0.10 * pitch_ratio
        rear_thigh_delta = -0.10 * pitch_ratio
        rear_calf_delta = 0.10 * pitch_ratio

        for idx in (1, 4):
            target[idx] += front_thigh_delta
        for idx in (2, 5):
            target[idx] += front_calf_delta
        for idx in (7, 10):
            target[idx] += rear_thigh_delta
        for idx in (8, 11):
            target[idx] += rear_calf_delta

        return target

    def control(self) -> None:
        if not self.received_state:
            return

        if not self.captured_start_pose:
            for i in range(GO2_NUM_MOTOR):
                self.start_q[i] = self.motor[i].q
            self.captured_start_pose = True
            self.get_logger().info("Captured start pose. Beginning stand-up ramp.")

        self.time_s += self.control_dt

        if self.time_s < self.ramp_duration_s:
            ratio = self.time_s / self.ramp_duration_s
            target_q = [
                (1.0 - ratio) * self.start_q[i] + ratio * STAND_Q[i]
                for i in range(GO2_NUM_MOTOR)
            ]
            pitch_ratio = 0.0
        else:
            t = self.time_s - self.ramp_duration_s
            pitch_ratio = math.sin(2.0 * math.pi * t / self.pitch_period_s)
            target_q = self._pitch_pose(pitch_ratio)

        if self.time_s - self.last_log_s > 1.0:
            self.get_logger().info(f"Pitch ratio: {pitch_ratio:+.2f}")
            self.last_log_s = self.time_s

        low_cmd = LowCmd()
        low_cmd.head[0] = 0xFE
        low_cmd.head[1] = 0xEF
        low_cmd.gpio = 0

        for i in range(GO2_NUM_MOTOR):
            cmd = low_cmd.motor_cmd[i]
            cmd.mode = self.motors_on
            cmd.q = target_q[i]
            cmd.dq = 0.0
            cmd.tau = 0.0
            cmd.kp = KP[i]
            cmd.kd = KD[i]

        low_cmd.crc = Crc(low_cmd)
        self.lowcmd_pub.publish(low_cmd)

    def low_state_handler(self, msg: LowState) -> None:
        self.lowstate_msg = msg
        self.imu = msg.imu_state
        for i in range(GO2_NUM_MOTOR):
            self.motor[i] = msg.motor_state[i]

        if not self.received_state:
            self.received_state = True
            self.get_logger().info("Received /lowstate, controller active.")

        if len(msg.wireless_remote) > 3 and msg.wireless_remote[3] == 1:
            self.motors_on = 0

    def lidar_handler(self, msg: PointCloud2) -> None:
        if self.lowstate_msg is None:
            return

        process_height_map_rotated(
            self.height_map,
            msg,
            self.lowstate_msg,
            self.x_range,
            self.y_range,
            self.res,
            delete_count=5,
        )

        self.lidar_frame_count += 1
        if self.lidar_frame_count % self.lidar_print_stride == 0:
            print(self.height_map[0])


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Go2ControllerTest()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
