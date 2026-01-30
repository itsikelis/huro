#!/usr/bin/env python3

import os
import numpy as np
import onnxruntime as ort

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from rclpy.timer import Timer
import math
import yaml

from sensor_msgs.msg import Joy
from unitree_api.msg import Request
from unitree_go.msg import LowCmd, LowState, IMUState, MotorState

from huro_py.crc_go import Crc

GO2_NUM_MOTOR = 12

JOINT_NAMES = [
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
]

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

q_start = [
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


def quat_rotate_inverse(q, v):
    q_w, q_x, q_y, q_z = q[0], q[1], q[2], q[3]
    q_conj = np.array([q_w, -q_x, -q_y, -q_z])
    t = 2.0 * np.cross(q_conj[1:], v)  # ✅ Correct: q_conj[1:] = [x, y, z]
    return v + q_conj[0] * t + np.cross(q_conj[1:], t)  # ✅ Complete formula


class Mode:
    PR = 0  # Series Control for Pitch/Roll Joints
    AB = 1  # Parallel Control for A/B Joints


class GO2PolicyRunner(Node):
    def __init__(self):
        super().__init__("go2_policy_runner")

        self.control_freq_hz = 50  # 10ms
        self.control_dt = 1.0 / self.control_freq_hz
        self.timer_dt_ms = int((self.control_dt) * 1000)
        self.time = 0.0
        self.init_duration_s = 3.0

        self.run_policy = False
        self.motors_on = 1

        share = get_package_share_directory("huro")
        # Load yaml config
        yaml_name = "action-info-Mjlab-Velocity-Flat-Unitree-Go2.yaml"
        yaml_path = os.path.join(share, "resources", "policies", "go2", yaml_name)
        with open(yaml_path, "r") as file:
            self.cfg = yaml.safe_load(file)

        # Load policy model params
        policy_name = "policy-Mjlab-Velocity-Flat-Unitree-Go2.onnx"
        policy_path = os.path.join(share, "resources", "policies", "go2", policy_name)
        if not os.path.exists(policy_path):
            raise FileNotFoundError(f"File not found at: {policy_path}")
        self._load_onnx_model(policy_path)
        self.get_logger().info("Policy loaded successfully")

        self.imu = IMUState()
        self.motor = [MotorState() for _ in range(len(JOINT_NAMES))]
        self.joystick = Joy()
        self.prev_actions = np.zeros(len(JOINT_NAMES))

        self.actions = np.zeros(GO2_NUM_MOTOR)

        # Publishers
        self.lowcmd_pub = self.create_publisher(LowCmd, "/lowcmd", 10)

        # Subscribers
        self.lowstate_sub = self.create_subscription(
            LowState, "/lowstate", self.low_state_handler, 10
        )

        self.joystick_sub = self.create_subscription(Joy, "/joy", self.joy_handler, 10)

        # To enter Go2 low-level control mode
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

        if not self.run_policy:
            for name in JOINT_NAMES:
                ratio = self.clamp(self.time / self.init_duration_s, 0.0, 1.0)
                idx = self.cfg[name]["index"]
                cmd = low_cmd.motor_cmd[idx]
                cmd.mode = self.motors_on
                cmd.q = (1.0 - ratio) * self.motor[idx].q + ratio * q_start[idx]
                # cmd.q = q_start[idx]
                cmd.dq = 0.0
                cmd.tau = 0.0
                cmd.kp = Kp[idx]
                cmd.kd = Kd[idx]
        else:  # If A (or cross) is pressed, run policy
            ## Get observations
            o = self._get_obs()
            ## Get action
            a = self._get_raw_action(o)
            ## Command robot
            for name in JOINT_NAMES:
                idx = self.cfg[name]["index"]
                q_init = self.cfg[name]["default_position"]
                action_scale = self.cfg[name]["action_scale"]
                cmd = low_cmd.motor_cmd[idx]
                cmd.mode = self.motors_on
                cmd.q = float(action_scale * a[idx] + q_init)
                cmd.dq = 0.0
                cmd.tau = 0.0
                cmd.kp = self.cfg[name]["stiffness"]
                cmd.kd = self.cfg[name]["damping"]

            self.actions = a.copy()

        low_cmd.crc = Crc(low_cmd)
        self.lowcmd_pub.publish(low_cmd)

    def _get_obs(self):
        """
        +------------------------------------------------------------+
        | Active Observation Terms in Group: 'policy' (shape: (45,)) |
        +-----------+----------------------------------+-------------+
        |   Index   | Name                             |    Shape    |
        +-----------+----------------------------------+-------------+
        |     0     | base_ang_vel                     |     (3,)    |
        |     1     | projected_gravity                |     (3,)    |
        |     2     | joint_pos                        |    (12,)    |
        |     3     | joint_vel                        |    (12,)    |
        |     4     | actions                          |    (12,)    |
        |     5     | command                          |     (3,)    |
        +-----------+----------------------------------+-------------+
        """

        base_ang_vel = np.array(
            [
                self.imu.gyroscope[0],
                self.imu.gyroscope[1],
                self.imu.gyroscope[2],
            ]
        )

        quat = np.array(
            [
                self.imu.quaternion[0],  # w
                self.imu.quaternion[1],  # x
                self.imu.quaternion[2],  # y
                self.imu.quaternion[3],  # z
            ]
        )
        gravity_world = np.array([0.0, 0.0, -1.0])
        projected_gravity = quat_rotate_inverse(quat, gravity_world)

        joint_pos_rel = np.zeros(GO2_NUM_MOTOR)
        joint_vel_rel = np.zeros(GO2_NUM_MOTOR)
        for name in JOINT_NAMES:
            idx = self.cfg[name]["index"]
            q_init = self.cfg[name]["default_position"]
            joint_pos_rel[idx] = self.motor[idx].q - q_init
            joint_vel_rel[idx] = self.motor[idx].dq - 0.0

        command = np.array(
            [
                self.joystick.axes[3] * 2,
                self.joystick.axes[2] * 1,
                self.joystick.axes[0] * 0.7,
            ]
        )

        return np.concatenate(
            [
                base_ang_vel,
                projected_gravity,
                joint_pos_rel,
                joint_vel_rel,
                self.actions,
                command,
            ],
            axis=0,
        )

    def low_state_handler(self, msg: LowState):
        self.imu = msg.imu_state
        for name in JOINT_NAMES:
            idx = self.cfg[name]["index"]
            self.motor[idx] = msg.motor_state[idx]

    def joy_handler(self, msg: Joy):
        self.joystick = msg
        if msg.buttons[1] == 1:
            self.run_policy = True
        if msg.buttons[0] == 1:
            self.motors_on = 0

    def clamp(self, value, low, high):
        if value < low:
            return low
        if value > high:
            return high
        return value

    def _load_onnx_model(self, model_path) -> None:
        # Choose execution providers (CPU by default)
        sess_options = ort.SessionOptions()
        self._ort_sess = ort.InferenceSession(
            model_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
        )
        # Check runtime I/O
        i = self._ort_sess.get_inputs()[0]
        assert i.name == "obs"
        self._obs_dim = i.shape[1]
        o = self._ort_sess.get_outputs()[0]
        assert o.shape[1] == GO2_NUM_MOTOR

    def _get_raw_action(self, obs: np.ndarray) -> np.ndarray:
        obs = obs.astype(np.float32, copy=False)
        inputs = {"obs": obs.reshape(1, -1)}
        # start = time.monotonic()
        outputs = self._ort_sess.run(None, inputs)
        # elapsed = time.monotonic() - start
        # print(f"Inference time: {elapsed * 1000:.2f} ms")
        return outputs[0][0]


def main(args=None):
    rclpy.init(args=args)
    node = GO2PolicyRunner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
