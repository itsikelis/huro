#!/usr/bin/env python3

import os
import traceback
from typing import Dict, List

import rclpy
import math
from rclpy.node import Node
from rclpy.exceptions import ParameterAlreadyDeclaredException
import yaml
from ament_index_python.packages import get_package_share_directory
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float64
from tf2_msgs.msg import TFMessage
from unitree_go.msg import LowCmd, LowState

try:
    from unitree_go.msg import SportModeState

    HAS_SPORT_MODE_STATE = True
except Exception:
    SportModeState = None  # type: ignore
    HAS_SPORT_MODE_STATE = False


class GzGo2StateAdapter(Node):
    """Adapt Gazebo ROS topics to Unitree-style state topics used by RL nodes."""

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

    DEFAULT_MAPPING_REL_PATH = os.path.join(
        "resources", "mappings", "go2", "physx_to_mujoco_go2.yaml"
    )

    def __init__(self) -> None:
        super().__init__("gz_go2_state_adapter")

        # use_sim_time is often auto-declared by launch/system in ROS 2.
        # Declare parameters defensively to avoid startup crashes.
        self._declare_param_if_needed("joint_state_topic", "/joint_states")
        self._declare_param_if_needed("joint_state_out_topic", "/joint_states")
        self._declare_param_if_needed("lowcmd_topic", "/lowcmd")
        self._declare_param_if_needed("tf_topic", "/tf")
        self._declare_param_if_needed("lowstate_topic", "/lowstate")
        self._declare_param_if_needed("sportmode_topic", "/sportmodestate")
        self._declare_param_if_needed("publish_rate_hz", 200.0)
        self._declare_param_if_needed("world_frame", "world")
        self._declare_param_if_needed("base_frame", "base")
        self._declare_param_if_needed("model_name", "go2")
        self._declare_param_if_needed("imu_topic", "/imu")
        # get_obs expects Unitree lowstate quaternion in wxyz order.
        self._declare_param_if_needed("lowstate_quat_wxyz", True)
        self._declare_param_if_needed("mapping_yaml_path", "")

        self.joint_state_topic = self.get_parameter("joint_state_topic").value
        self.joint_state_out_topic = self.get_parameter("joint_state_out_topic").value
        self.lowcmd_topic = self.get_parameter("lowcmd_topic").value
        self.tf_topic = self.get_parameter("tf_topic").value
        self.lowstate_topic = self.get_parameter("lowstate_topic").value
        self.sportmode_topic = self.get_parameter("sportmode_topic").value
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.world_frame = self.get_parameter("world_frame").value
        self.base_frame = self.get_parameter("base_frame").value
        self.model_name = self.get_parameter("model_name").value
        self.imu_topic = self.get_parameter("imu_topic").value
        self.lowstate_quat_wxyz = bool(
            self.get_parameter("lowstate_quat_wxyz").value
        )
        self.mapping_yaml_path = str(self.get_parameter("mapping_yaml_path").value)

        self.joint_order = self._resolve_expected_joint_order()

        self._joint_pos: Dict[str, float] = {}
        self._joint_vel: Dict[str, float] = {}
        self._joint_eff: Dict[str, float] = {}

        self._base_pos = [0.0, 0.0, 0.0]
        self._base_quat_xyzw = [0.0, 0.0, 0.0, 1.0]
        self._base_lin_vel = [0.0, 0.0, 0.0]
        self._last_base_pos = None
        self._last_base_t = None
        self._got_imu = False
        self._imu_quat_xyzw = [0.0, 0.0, 0.0, 1.0]
        self._imu_gyro = [0.0, 0.0, 0.0]
        self._imu_acc = [0.0, 0.0, 0.0]

        self._got_joint = False
        self._got_base_tf = False
        self._got_lowcmd = False
        self._cmd_q = [0.0] * 12
        self._cmd_dq = [0.0] * 12
        self._cmd_tau = [0.0] * 12

        self.lowstate_pub = self.create_publisher(LowState, self.lowstate_topic, 10)
        self.jointstate_pub = self.create_publisher(
            JointState, self.joint_state_out_topic, 20
        )
        self.sportmode_pub = None
        if HAS_SPORT_MODE_STATE:
            self.sportmode_pub = self.create_publisher(
                SportModeState, self.sportmode_topic, 10
            )
        else:
            self.get_logger().warning(
                "unitree_go.msg.SportModeState not available; /sportmodestate bridge disabled"
            )

        self.create_subscription(JointState, self.joint_state_topic, self._on_js, 50)
        self.create_subscription(LowCmd, self.lowcmd_topic, self._on_lowcmd, 50)
        self.create_subscription(TFMessage, self.tf_topic, self._on_tf, 50)
        self.create_subscription(Imu, self.imu_topic, self._on_imu, 50)

        self.cmd_pos_pubs = {}
        for jn in self.joint_order:
            t1 = f"/model/{self.model_name}/joint/{jn}/cmd_pos"
            self.cmd_pos_pubs[jn] = self.create_publisher(Float64, t1, 10)

        period = 1.0 / max(self.publish_rate_hz, 1.0)
        self.timer = self.create_timer(period, self._publish)
        self.diag_timer = self.create_timer(2.0, self._log_health)

        self.get_logger().info(
            f"Gazebo adapter active: {self.joint_state_topic} -> {self.joint_state_out_topic} (+fallback {self.lowcmd_topic}) + {self.tf_topic} -> {self.lowstate_topic}, {self.sportmode_topic}"
        )
        self.get_logger().info(
            f"Joint order for lowstate/joint_states: {self.joint_order}"
        )

    def _resolve_expected_joint_order(self) -> List[str]:
        mapping_path = self.mapping_yaml_path
        if not mapping_path:
            try:
                share = get_package_share_directory("huro")
                mapping_path = os.path.join(share, self.DEFAULT_MAPPING_REL_PATH)
            except Exception:
                mapping_path = ""

        if mapping_path:
            try:
                with open(mapping_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                target_names = data.get("target_joint_names", [])
                if len(target_names) == 12 and all(isinstance(n, str) for n in target_names):
                    return list(target_names)
                self.get_logger().warning(
                    f"Invalid target_joint_names in mapping file ({mapping_path}); using built-in order"
                )
            except Exception as e:
                self.get_logger().warning(
                    f"Could not load mapping file ({mapping_path}): {e}. Using built-in order"
                )

        return list(self.JOINT_ORDER)

    @staticmethod
    def _normalize_joint_name(name: str) -> str:
        # Accept common Gazebo/bridge naming forms and extract the bare joint name.
        n = (name or "").strip()
        if "::" in n:
            n = n.split("::")[-1]
        if "/" in n:
            n = n.split("/")[-1]
        return n

    def _log_health(self) -> None:
        src = "joint_states_gz" if self._got_joint else ("lowcmd" if self._got_lowcmd else "default_q")
        imu_src = "imu_topic" if self._got_imu else "default_zero"
        q0 = self._joint_pos.get(self.joint_order[0], None)
        cmd0 = self._cmd_q[0]
        self.get_logger().info(
            f"health: state_src={src} imu_src={imu_src} got_tf={self._got_base_tf} got_joint={self._got_joint} got_lowcmd={self._got_lowcmd} got_imu={self._got_imu} q0_joint={q0} q0_cmd={cmd0:.3f}"
        )

    def _on_js(self, msg: JointState) -> None:
        for i, name in enumerate(msg.name):
            canonical = self._normalize_joint_name(name)
            if canonical not in self.joint_order:
                continue
            self._joint_pos[canonical] = msg.position[i] if i < len(msg.position) else 0.0
            self._joint_vel[canonical] = msg.velocity[i] if i < len(msg.velocity) else 0.0
            self._joint_eff[canonical] = msg.effort[i] if i < len(msg.effort) else 0.0
        self._got_joint = True

    def _on_tf(self, msg: TFMessage) -> None:
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        for tr in msg.transforms:
            if tr.child_frame_id != self.base_frame:
                continue
            if tr.header.frame_id not in ("", self.world_frame):
                continue

            pos = [
                float(tr.transform.translation.x),
                float(tr.transform.translation.y),
                float(tr.transform.translation.z),
            ]
            quat_xyzw = [
                float(tr.transform.rotation.x),
                float(tr.transform.rotation.y),
                float(tr.transform.rotation.z),
                float(tr.transform.rotation.w),
            ]

            if self._last_base_pos is not None and self._last_base_t is not None:
                dt = max(now_sec - self._last_base_t, 1e-6)
                self._base_lin_vel = [
                    (pos[0] - self._last_base_pos[0]) / dt,
                    (pos[1] - self._last_base_pos[1]) / dt,
                    (pos[2] - self._last_base_pos[2]) / dt,
                ]

            self._base_pos = pos
            self._base_quat_xyzw = quat_xyzw
            self._last_base_pos = pos
            self._last_base_t = now_sec
            self._got_base_tf = True

    def _on_lowcmd(self, msg: LowCmd) -> None:
        for i in range(min(12, len(msg.motor_cmd))):
            self._cmd_q[i] = float(msg.motor_cmd[i].q)
            self._cmd_dq[i] = float(msg.motor_cmd[i].dq)
            self._cmd_tau[i] = float(msg.motor_cmd[i].tau)

        # Mirror LowCmd into per-joint position command topics for Gazebo controllers.
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
        # Keep gyro/acc updates even when orientation is temporarily invalid.
        # Orientation validity is tracked via _got_imu and normalized quaternion above.
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

        # Use IMU topic values only; no TF-derived IMU fallback.
        if self._got_imu:
            qx, qy, qz, qw = self._imu_quat_xyzw
            gx, gy, gz = self._imu_gyro
            ax, ay, az = self._imu_acc
        else:
            qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
            gx, gy, gz = 0.0, 0.0, 0.0
            ax, ay, az = 0.0, 0.0, 0.0

        if hasattr(low, "imu_state"):
            if self.lowstate_quat_wxyz:
                self._safe_set(low.imu_state.quaternion, 0, qw)
                self._safe_set(low.imu_state.quaternion, 1, qx)
                self._safe_set(low.imu_state.quaternion, 2, qy)
                self._safe_set(low.imu_state.quaternion, 3, qz)
            else:
                self._safe_set(low.imu_state.quaternion, 0, qx)
                self._safe_set(low.imu_state.quaternion, 1, qy)
                self._safe_set(low.imu_state.quaternion, 2, qz)
                self._safe_set(low.imu_state.quaternion, 3, qw)

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

        # Optional odom-like state for nodes that use /sportmodestate.
        if self.sportmode_pub is not None and HAS_SPORT_MODE_STATE:
            sm = SportModeState()
            if hasattr(sm, "position"):
                self._safe_set(sm.position, 0, self._base_pos[0])
                self._safe_set(sm.position, 1, self._base_pos[1])
                self._safe_set(sm.position, 2, self._base_pos[2])
            if hasattr(sm, "velocity"):
                self._safe_set(sm.velocity, 0, self._base_lin_vel[0])
                self._safe_set(sm.velocity, 1, self._base_lin_vel[1])
                self._safe_set(sm.velocity, 2, self._base_lin_vel[2])
            if hasattr(sm, "imu_state"):
                self._safe_set(sm.imu_state.quaternion, 0, qw)
                self._safe_set(sm.imu_state.quaternion, 1, qx)
                self._safe_set(sm.imu_state.quaternion, 2, qy)
                self._safe_set(sm.imu_state.quaternion, 3, qz)
                self._safe_set(sm.imu_state.gyroscope, 0, 0.0)
                self._safe_set(sm.imu_state.gyroscope, 1, 0.0)
                self._safe_set(sm.imu_state.gyroscope, 2, 0.0)

            self.sportmode_pub.publish(sm)


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
