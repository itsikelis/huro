#!/usr/bin/env python3

import math
import traceback
from typing import Dict, List

import rclpy
from rclpy.node import Node
from rclpy.exceptions import ParameterAlreadyDeclaredException
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
        self._declare_param_if_needed("pose_tf_topic", "/gz_pose_tf")
        self._declare_param_if_needed("lowstate_quat_wxyz", False)
        self._declare_param_if_needed("estimate_gyro_from_pose_tf", False)

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
        self.pose_tf_topic = self.get_parameter("pose_tf_topic").value
        self.lowstate_quat_wxyz = bool(
            self.get_parameter("lowstate_quat_wxyz").value
        )
        self.estimate_gyro_from_pose_tf = bool(
            self.get_parameter("estimate_gyro_from_pose_tf").value
        )

        self._joint_pos: Dict[str, float] = {}
        self._joint_vel: Dict[str, float] = {}
        self._joint_eff: Dict[str, float] = {}

        self._base_pos = [0.0, 0.0, 0.0]
        self._base_quat_xyzw = [0.0, 0.0, 0.0, 1.0]
        self._base_lin_vel = [0.0, 0.0, 0.0]
        self._base_ang_vel = [0.0, 0.0, 0.0]
        self._last_base_pos = None
        self._last_base_t = None
        self._last_base_quat = None
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
        self.create_subscription(TFMessage, self.pose_tf_topic, self._on_pose_tf, 50)

        self.cmd_pos_pubs = {}
        for jn in self.JOINT_ORDER:
            t1 = f"/model/{self.model_name}/joint/{jn}/cmd_pos"
            self.cmd_pos_pubs[jn] = self.create_publisher(Float64, t1, 10)

        period = 1.0 / max(self.publish_rate_hz, 1.0)
        self.timer = self.create_timer(period, self._publish)
        self.diag_timer = self.create_timer(2.0, self._log_health)

        self.get_logger().info(
            f"Gazebo adapter active: {self.joint_state_topic} -> {self.joint_state_out_topic} (+fallback {self.lowcmd_topic}) + {self.tf_topic} -> {self.lowstate_topic}, {self.sportmode_topic}"
        )

    def _log_health(self) -> None:
        src = "joint_states_gz" if self._got_joint else ("lowcmd" if self._got_lowcmd else "default_q")
        imu_src = "imu_topic" if self._got_imu else "pose_tf_fallback"
        q0 = self._joint_pos.get(self.JOINT_ORDER[0], None)
        cmd0 = self._cmd_q[0]
        self.get_logger().info(
            f"health: state_src={src} imu_src={imu_src} got_tf={self._got_base_tf} got_joint={self._got_joint} got_lowcmd={self._got_lowcmd} got_imu={self._got_imu} q0_joint={q0} q0_cmd={cmd0:.3f}"
        )

    def _on_js(self, msg: JointState) -> None:
        for i, name in enumerate(msg.name):
            self._joint_pos[name] = msg.position[i] if i < len(msg.position) else 0.0
            self._joint_vel[name] = msg.velocity[i] if i < len(msg.velocity) else 0.0
            self._joint_eff[name] = msg.effort[i] if i < len(msg.effort) else 0.0
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

    @staticmethod
    def _quat_conj_xyzw(q):
        return [-q[0], -q[1], -q[2], q[3]]

    @staticmethod
    def _quat_mul_xyzw(a, b):
        ax, ay, az, aw = a
        bx, by, bz, bw = b
        return [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]

    def _update_base_pose(self, pos, quat_xyzw, now_sec):
        if self._last_base_pos is not None and self._last_base_t is not None:
            dt = max(now_sec - self._last_base_t, 1e-6)
            self._base_lin_vel = [
                (pos[0] - self._last_base_pos[0]) / dt,
                (pos[1] - self._last_base_pos[1]) / dt,
                (pos[2] - self._last_base_pos[2]) / dt,
            ]

            if self._last_base_quat is not None:
                dq = self._quat_mul_xyzw(
                    quat_xyzw, self._quat_conj_xyzw(self._last_base_quat)
                )
                vx, vy, vz, vw = dq
                vnorm = math.sqrt(vx * vx + vy * vy + vz * vz)
                if vnorm > 1e-9:
                    angle = 2.0 * math.atan2(vnorm, max(-1.0, min(1.0, vw)))
                    scale = angle / (dt * vnorm)
                    if self.estimate_gyro_from_pose_tf:
                        self._base_ang_vel = [vx * scale, vy * scale, vz * scale]
                    else:
                        self._base_ang_vel = [0.0, 0.0, 0.0]

        self._base_pos = pos
        self._base_quat_xyzw = quat_xyzw
        self._last_base_pos = pos
        self._last_base_quat = quat_xyzw
        self._last_base_t = now_sec
        self._got_base_tf = True

    def _on_pose_tf(self, msg: TFMessage) -> None:
        now_sec = self.get_clock().now().nanoseconds * 1e-9

        # Prefer model root pose from Gazebo stream, fallback to base.
        chosen = None
        fallback = None
        for tr in msg.transforms:
            child = tr.child_frame_id
            if child == self.model_name:
                chosen = tr
                break
            if child == self.base_frame:
                fallback = tr
        if chosen is None:
            chosen = fallback
        if chosen is None:
            return

        pos = [
            float(chosen.transform.translation.x),
            float(chosen.transform.translation.y),
            float(chosen.transform.translation.z),
        ]
        quat_xyzw = [
            float(chosen.transform.rotation.x),
            float(chosen.transform.rotation.y),
            float(chosen.transform.rotation.z),
            float(chosen.transform.rotation.w),
        ]
        self._update_base_pose(pos, quat_xyzw, now_sec)

    def _on_lowcmd(self, msg: LowCmd) -> None:
        for i in range(min(12, len(msg.motor_cmd))):
            self._cmd_q[i] = float(msg.motor_cmd[i].q)
            self._cmd_dq[i] = float(msg.motor_cmd[i].dq)
            self._cmd_tau[i] = float(msg.motor_cmd[i].tau)

        # Mirror LowCmd into per-joint position command topics for Gazebo controllers.
        for i, jn in enumerate(self.JOINT_ORDER):
            cmd = Float64()
            cmd.data = self._cmd_q[i]
            self.cmd_pos_pubs[jn].publish(cmd)

        self._got_lowcmd = True

    def _on_imu(self, msg: Imu) -> None:
        self._imu_quat_xyzw = [
            float(msg.orientation.x),
            float(msg.orientation.y),
            float(msg.orientation.z),
            float(msg.orientation.w),
        ]
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
        self._got_imu = True

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
        js.name = list(self.JOINT_ORDER)
        js.position = []
        js.velocity = []
        js.effort = []

        # Match existing MuJoCo lowstate convention in this repo:
        # For policy observations we default to [w, x, y, z].
        if self._got_imu:
            qx, qy, qz, qw = self._imu_quat_xyzw
            gx, gy, gz = self._imu_gyro
            ax, ay, az = self._imu_acc
        else:
            qx, qy, qz, qw = self._base_quat_xyzw
            gx, gy, gz = self._base_ang_vel
            ax, ay, az = 0.0, 0.0, 0.0

        # Keep quaternion normalized to avoid gravity projection drift in policy obs.
        qn = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if qn > 1e-9:
            qx, qy, qz, qw = qx / qn, qy / qn, qz / qn, qw / qn
        else:
            qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0

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

        for i, jn in enumerate(self.JOINT_ORDER):
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
