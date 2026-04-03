#!/usr/bin/env python3

import traceback
from typing import Dict, List

import rclpy
import math
from rclpy.node import Node
from rclpy.exceptions import ParameterAlreadyDeclaredException
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float64
from unitree_go.msg import LowCmd, LowState


class GzGo2StateAdapter(Node):
    """Minimal Gazebo adapter: lowstate + joint_states + lowcmd->cmd_pos."""

    JOINT_ORDER: List[str] = [
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

    DEFAULT_Q: List[float] = [
        0.1,
        0.8,
        -1.5,
        -0.1,
        0.8,
        -1.5,
        0.1,
        1.0,
        -1.5,
        -0.1,
        1.0,
        -1.5,
    ]

    def __init__(self) -> None:
        super().__init__("gz_go2_state_adapter")

        self._declare_param_if_needed("joint_state_topic", "/joint_states_gz")
        self._declare_param_if_needed("joint_state_out_topic", "/joint_states")
        self._declare_param_if_needed("imu_topic", "/imu")
        self._declare_param_if_needed("lowstate_topic", "/lowstate")
        self._declare_param_if_needed("lowcmd_topic", "/lowcmd")
        self._declare_param_if_needed("model_name", "go2")
        self._declare_param_if_needed("publish_rate_hz", 200.0)

        self.joint_state_topic = self.get_parameter("joint_state_topic").value
        self.joint_state_out_topic = self.get_parameter("joint_state_out_topic").value
        self.imu_topic = self.get_parameter("imu_topic").value
        self.lowcmd_topic = self.get_parameter("lowcmd_topic").value
        self.lowstate_topic = self.get_parameter("lowstate_topic").value
        self.model_name = self.get_parameter("model_name").value
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)

        self.joint_order = list(self.JOINT_ORDER)

        self._joint_pos: Dict[str, float] = {}
        self._joint_vel: Dict[str, float] = {}
        self._joint_eff: Dict[str, float] = {}

        self._got_imu = False
        self._imu_quat_xyzw = [0.0, 0.0, 0.0, 1.0]
        self._imu_gyro = [0.0, 0.0, 0.0]
        self._imu_acc = [0.0, 0.0, 0.0]

        self._got_joint = False
        self._got_lowcmd = False
        self._cmd_q = [0.0] * 12
        self._cmd_dq = [0.0] * 12
        self._cmd_tau = [0.0] * 12

        self.lowstate_pub = self.create_publisher(LowState, self.lowstate_topic, 10)
        self.jointstate_pub = self.create_publisher(
            JointState, self.joint_state_out_topic, 20
        )

        self.create_subscription(JointState, self.joint_state_topic, self._on_js, 50)
        self.create_subscription(LowCmd, self.lowcmd_topic, self._on_lowcmd, 50)
        self.create_subscription(Imu, self.imu_topic, self._on_imu, 50)

        self.cmd_pos_pubs = {}
        for jn in self.joint_order:
            t1 = f"/model/{self.model_name}/joint/{jn}/cmd_pos"
            self.cmd_pos_pubs[jn] = self.create_publisher(Float64, t1, 10)

        period = 1.0 / max(self.publish_rate_hz, 1.0)
        self.timer = self.create_timer(period, self._publish)

        self.get_logger().info(
            f"Gazebo adapter active: {self.joint_state_topic} + {self.imu_topic} -> {self.lowstate_topic}; {self.joint_state_out_topic} for RViz"
        )

    @staticmethod
    def _normalize_joint_name(name: str) -> str:
        # Accept common Gazebo/bridge naming forms and extract the bare joint name.
        n = (name or "").strip()
        if "::" in n:
            n = n.split("::")[-1]
        if "/" in n:
            n = n.split("/")[-1]
        return n

    def _on_js(self, msg: JointState) -> None:
        for i, name in enumerate(msg.name):
            canonical = self._normalize_joint_name(name)
            if canonical not in self.joint_order:
                continue
            self._joint_pos[canonical] = msg.position[i] if i < len(msg.position) else 0.0
            self._joint_vel[canonical] = msg.velocity[i] if i < len(msg.velocity) else 0.0
            self._joint_eff[canonical] = msg.effort[i] if i < len(msg.effort) else 0.0
        self._got_joint = True

    def _on_lowcmd(self, msg: LowCmd) -> None:
        for i in range(min(12, len(msg.motor_cmd))):
            self._cmd_q[i] = float(msg.motor_cmd[i].q)
            self._cmd_dq[i] = float(msg.motor_cmd[i].dq)
            self._cmd_tau[i] = float(msg.motor_cmd[i].tau)

        for i, jn in enumerate(self.joint_order):
            cmd = Float64()
            cmd.data = self._cmd_q[i]
            self.cmd_pos_pubs[jn].publish(cmd)

        self._got_lowcmd = True

    def _on_imu(self, msg: Imu) -> None:
        qx = float(msg.orientation.x)
        qy = float(msg.orientation.y)
        qz = float(msg.orientation.z)
        qw = float(msg.orientation.w)

        # Reject invalid quaternions and normalize valid ones to avoid gravity bias.
        valid_orientation = False
        if all(math.isfinite(v) for v in (qx, qy, qz, qw)):
            norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
            if norm > 1e-6:
                inv = 1.0 / norm
                self._imu_quat_xyzw = [qx * inv, qy * inv, qz * inv, qw * inv]
                valid_orientation = True
        self._imu_gyro = [
            float(msg.angular_velocity.x),
            float(msg.angular_velocity.y),
            float(msg.angular_velocity.z),
        ]
        self._imu_acc = [
            float(msg.linear_acceleration.x),
            float(msg.linear_acceleration.y),
            float(msg.linear_acceleration.z),
        ]
        self._got_imu = valid_orientation

    @staticmethod
    def _safe_set(arr, idx: int, value: float) -> None:
        try:
            arr[idx] = value
        except Exception:
            pass

    def _declare_param_if_needed(self, name: str, default_value) -> None:
        try:
            self.declare_parameter(name, default_value)
        except ParameterAlreadyDeclaredException:
            pass

    def _publish(self) -> None:
        low = LowState()
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = list(self.joint_order)
        js.position = []
        js.velocity = []
        js.effort = []

        if self._got_imu:
            qx, qy, qz, qw = self._imu_quat_xyzw
            gx, gy, gz = self._imu_gyro
            ax, ay, az = self._imu_acc
        else:
            qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
            gx, gy, gz = 0.0, 0.0, 0.0
            ax, ay, az = 0.0, 0.0, 0.0

        if hasattr(low, "imu_state"):
            self._safe_set(low.imu_state.quaternion, 0, qw)
            self._safe_set(low.imu_state.quaternion, 1, qx)
            self._safe_set(low.imu_state.quaternion, 2, qy)
            self._safe_set(low.imu_state.quaternion, 3, qz)

            self._safe_set(low.imu_state.gyroscope, 0, gx)
            self._safe_set(low.imu_state.gyroscope, 1, gy)
            self._safe_set(low.imu_state.gyroscope, 2, gz)
            self._safe_set(low.imu_state.accelerometer, 0, ax)
            self._safe_set(low.imu_state.accelerometer, 1, ay)
            self._safe_set(low.imu_state.accelerometer, 2, az)

        for i, jn in enumerate(self.joint_order):
            if self._got_joint:
                q = float(self._joint_pos.get(jn, 0.0))
                dq = float(self._joint_vel.get(jn, 0.0))
                tau = float(self._joint_eff.get(jn, 0.0))
            elif self._got_lowcmd:
                q = self._cmd_q[i]
                dq = self._cmd_dq[i]
                tau = self._cmd_tau[i]
            else:
                q = self.DEFAULT_Q[i]
                dq = 0.0
                tau = 0.0
            try:
                low.motor_state[i].q = q
                low.motor_state[i].dq = dq
                low.motor_state[i].ddq = 0.0
                if hasattr(low.motor_state[i], "tau_est"):
                    low.motor_state[i].tau_est = tau
            except Exception:
                pass

            js.position.append(q)
            js.velocity.append(dq)
            js.effort.append(tau)

        for i in range(4):
            self._safe_set(low.foot_force, i, 0)
            if hasattr(low, "foot_force_est"):
                self._safe_set(low.foot_force_est, i, 0)

        self.lowstate_pub.publish(low)
        self.jointstate_pub.publish(js)


def main() -> None:
    try:
        rclpy.init()
        node = GzGo2StateAdapter()
        rclpy.spin(node)
        node.destroy_node()
        rclpy.shutdown()
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
