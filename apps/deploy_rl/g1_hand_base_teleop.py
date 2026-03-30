#!/usr/bin/env python3

from __future__ import annotations

import select
import sys
import termios
import tty

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import Float32
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import InteractiveMarker
from visualization_msgs.msg import InteractiveMarkerControl
from visualization_msgs.msg import Marker

BASE_CMD_TOPIC = "/g1_hand_base/cmd_vel"
HEIGHT_TOPIC = "/g1_hand_base/height"
LEFT_HAND_TOPIC = "/g1_hand_base/left_hand_target"
RIGHT_HAND_TOPIC = "/g1_hand_base/right_hand_target"

WORLD_FRAME = "world"
BASE_FRAME = "pelvis"
LEFT_HAND_FRAME = "left_wrist_yaw_link"
RIGHT_HAND_FRAME = "right_wrist_yaw_link"

PUBLISH_DT = 0.05
LINEAR_STEP_MPS = 0.1
ANGULAR_STEP_RADPS = 0.15
HEIGHT_STEP_M = 0.02
MAX_LINEAR_MPS = 1.0
MAX_ANGULAR_RADPS = 1.5
MIN_HEIGHT_M = 0.4
MAX_HEIGHT_M = 1.2
MARKER_SCALE = 0.18


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class G1HandBaseTeleop(Node):
    """Publish topic-driven Hand-Base references from keyboard + RViz markers."""

    def __init__(self) -> None:
        super().__init__("g1_hand_base_teleop")

        self.base_cmd_pub = self.create_publisher(Twist, BASE_CMD_TOPIC, 10)
        self.height_pub = self.create_publisher(Float32, HEIGHT_TOPIC, 10)
        self.left_hand_pub = self.create_publisher(PoseStamped, LEFT_HAND_TOPIC, 10)
        self.right_hand_pub = self.create_publisher(PoseStamped, RIGHT_HAND_TOPIC, 10)

        self.tf_buffer = Buffer(node=self)
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)
        self.marker_server = InteractiveMarkerServer(self, "g1_hand_base_markers")

        self.cmd_vx = 0.0
        self.cmd_vy = 0.0
        self.cmd_yaw = 0.0
        self.target_height_m: float | None = None
        self.left_pose_msg: PoseStamped | None = None
        self.right_pose_msg: PoseStamped | None = None
        self.targets_initialized = False
        self._last_wait_log_s = -1.0

        self._stdin_fd: int | None = None
        self._stdin_termios: list | None = None
        self._keyboard_enabled = False
        self._setup_keyboard()

        self.timer = self.create_timer(PUBLISH_DT, self._timer_callback)

        if self._keyboard_enabled:
            self.get_logger().info(
                "Keyboard ready: w/s forward, a/d lateral, q/e yaw, r/f height, "
                "SPACE zero base velocity, z reinitialize from TF."
            )
        else:
            self.get_logger().warn(
                "Keyboard control is disabled because stdin is not a TTY. "
                "Inside Docker, run this node from an interactive shell "
                "such as `docker exec -it ... bash`."
            )
        self.get_logger().info(
            "Waiting for TF to initialize pelvis and hand targets for the interactive markers."
        )

    # ------------------------------------------------------------------
    # Main loop.
    # ------------------------------------------------------------------

    def _timer_callback(self) -> None:
        self._poll_keyboard()

        if not self.targets_initialized:
            if not self._initialize_targets_from_tf():
                self._log_waiting_for_tf()
                return
            self.get_logger().info(
                "Initialized height and hand targets from live TF and created RViz markers."
            )

        self._publish_base_targets()
        self._publish_hand_targets()

    # ------------------------------------------------------------------
    # Initialization from TF.
    # ------------------------------------------------------------------

    def _initialize_targets_from_tf(self) -> bool:
        pelvis_tf = self._lookup_transform(WORLD_FRAME, BASE_FRAME)
        left_tf = self._lookup_transform(WORLD_FRAME, LEFT_HAND_FRAME)
        right_tf = self._lookup_transform(WORLD_FRAME, RIGHT_HAND_FRAME)
        if pelvis_tf is None or left_tf is None or right_tf is None:
            return False

        self.target_height_m = float(pelvis_tf.transform.translation.z)
        self.left_pose_msg = self._pose_stamped_from_transform(left_tf)
        self.right_pose_msg = self._pose_stamped_from_transform(right_tf)
        self._create_interactive_markers()
        self.targets_initialized = True
        return True

    def _lookup_transform(self, target_frame: str, source_frame: str):
        try:
            return self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                Time(),
                timeout=Duration(seconds=0.05),
            )
        except Exception:
            return None

    def _pose_stamped_from_transform(self, transform) -> PoseStamped:
        msg = PoseStamped()
        msg.header.frame_id = WORLD_FRAME
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(transform.transform.translation.x)
        msg.pose.position.y = float(transform.transform.translation.y)
        msg.pose.position.z = float(transform.transform.translation.z)
        msg.pose.orientation.x = float(transform.transform.rotation.x)
        msg.pose.orientation.y = float(transform.transform.rotation.y)
        msg.pose.orientation.z = float(transform.transform.rotation.z)
        msg.pose.orientation.w = float(transform.transform.rotation.w)
        return msg

    # ------------------------------------------------------------------
    # Interactive marker setup.
    # ------------------------------------------------------------------

    def _create_interactive_markers(self) -> None:
        assert self.left_pose_msg is not None
        assert self.right_pose_msg is not None

        self.marker_server.clear()
        self.marker_server.insert(
            self._make_hand_marker("left_hand_target", self.left_pose_msg, 0.2, 0.6, 1.0),
            feedback_callback=self._left_marker_feedback,
        )
        self.marker_server.insert(
            self._make_hand_marker("right_hand_target", self.right_pose_msg, 1.0, 0.4, 0.2),
            feedback_callback=self._right_marker_feedback,
        )
        self.marker_server.applyChanges()

    def _make_hand_marker(
        self,
        name: str,
        pose_msg: PoseStamped,
        r: float,
        g: float,
        b: float,
    ) -> InteractiveMarker:
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = WORLD_FRAME
        int_marker.name = name
        int_marker.description = name
        int_marker.scale = MARKER_SCALE
        int_marker.pose = pose_msg.pose

        visible_control = InteractiveMarkerControl()
        visible_control.always_visible = True
        visible_control.interaction_mode = InteractiveMarkerControl.MOVE_ROTATE_3D
        visible_control.markers.append(self._hand_marker_visual(r, g, b))
        int_marker.controls.append(visible_control)

        int_marker.controls.append(
            self._axis_control(
                "move_x", InteractiveMarkerControl.MOVE_AXIS, x=1.0, y=0.0, z=0.0
            )
        )
        int_marker.controls.append(
            self._axis_control(
                "move_y", InteractiveMarkerControl.MOVE_AXIS, x=0.0, y=1.0, z=0.0
            )
        )
        int_marker.controls.append(
            self._axis_control(
                "move_z", InteractiveMarkerControl.MOVE_AXIS, x=0.0, y=0.0, z=1.0
            )
        )
        int_marker.controls.append(
            self._axis_control(
                "rotate_x", InteractiveMarkerControl.ROTATE_AXIS, x=1.0, y=0.0, z=0.0
            )
        )
        int_marker.controls.append(
            self._axis_control(
                "rotate_y", InteractiveMarkerControl.ROTATE_AXIS, x=0.0, y=1.0, z=0.0
            )
        )
        int_marker.controls.append(
            self._axis_control(
                "rotate_z", InteractiveMarkerControl.ROTATE_AXIS, x=0.0, y=0.0, z=1.0
            )
        )
        return int_marker

    def _hand_marker_visual(self, r: float, g: float, b: float) -> Marker:
        marker = Marker()
        marker.type = Marker.SPHERE
        marker.scale.x = 0.06
        marker.scale.y = 0.06
        marker.scale.z = 0.06
        marker.color.r = float(r)
        marker.color.g = float(g)
        marker.color.b = float(b)
        marker.color.a = 0.85
        return marker

    def _axis_control(
        self,
        name: str,
        mode: int,
        *,
        x: float,
        y: float,
        z: float,
    ) -> InteractiveMarkerControl:
        control = InteractiveMarkerControl()
        control.name = name
        control.orientation.w = 1.0
        control.orientation.x = float(x)
        control.orientation.y = float(y)
        control.orientation.z = float(z)
        control.interaction_mode = mode
        return control

    def _left_marker_feedback(self, feedback) -> None:
        if self.left_pose_msg is None:
            return
        self.left_pose_msg.header.stamp = self.get_clock().now().to_msg()
        self.left_pose_msg.pose = feedback.pose
        self.left_hand_pub.publish(self.left_pose_msg)

    def _right_marker_feedback(self, feedback) -> None:
        if self.right_pose_msg is None:
            return
        self.right_pose_msg.header.stamp = self.get_clock().now().to_msg()
        self.right_pose_msg.pose = feedback.pose
        self.right_hand_pub.publish(self.right_pose_msg)

    # ------------------------------------------------------------------
    # Keyboard handling.
    # ------------------------------------------------------------------

    def _setup_keyboard(self) -> None:
        if not sys.stdin.isatty():
            return
        try:
            self._stdin_fd = sys.stdin.fileno()
            self._stdin_termios = termios.tcgetattr(self._stdin_fd)
            tty.setcbreak(self._stdin_fd)
            self._keyboard_enabled = True
        except (termios.error, ValueError, OSError) as exc:
            self.get_logger().warn(f"Failed to enable keyboard control: {exc}")
            self._stdin_fd = None
            self._stdin_termios = None
            self._keyboard_enabled = False

    def _restore_keyboard(self) -> None:
        if self._stdin_fd is None or self._stdin_termios is None:
            return
        try:
            termios.tcsetattr(self._stdin_fd, termios.TCSADRAIN, self._stdin_termios)
        except (termios.error, ValueError, OSError):
            pass
        self._stdin_fd = None
        self._stdin_termios = None
        self._keyboard_enabled = False

    def _poll_keyboard(self) -> None:
        if not self._keyboard_enabled or self._stdin_fd is None:
            return
        while True:
            ready, _, _ = select.select([self._stdin_fd], [], [], 0.0)
            if not ready:
                break
            key = sys.stdin.read(1)
            if not key:
                break
            self._handle_keypress(key)

    def _handle_keypress(self, key: str) -> None:
        if key == "w":
            self.cmd_vx = _clamp(self.cmd_vx + LINEAR_STEP_MPS, -MAX_LINEAR_MPS, MAX_LINEAR_MPS)
        elif key == "s":
            self.cmd_vx = _clamp(self.cmd_vx - LINEAR_STEP_MPS, -MAX_LINEAR_MPS, MAX_LINEAR_MPS)
        elif key == "a":
            self.cmd_vy = _clamp(self.cmd_vy + LINEAR_STEP_MPS, -MAX_LINEAR_MPS, MAX_LINEAR_MPS)
        elif key == "d":
            self.cmd_vy = _clamp(self.cmd_vy - LINEAR_STEP_MPS, -MAX_LINEAR_MPS, MAX_LINEAR_MPS)
        elif key == "q":
            self.cmd_yaw = _clamp(
                self.cmd_yaw + ANGULAR_STEP_RADPS, -MAX_ANGULAR_RADPS, MAX_ANGULAR_RADPS
            )
        elif key == "e":
            self.cmd_yaw = _clamp(
                self.cmd_yaw - ANGULAR_STEP_RADPS, -MAX_ANGULAR_RADPS, MAX_ANGULAR_RADPS
            )
        elif key == "r" and self.target_height_m is not None:
            self.target_height_m = _clamp(
                self.target_height_m + HEIGHT_STEP_M, MIN_HEIGHT_M, MAX_HEIGHT_M
            )
        elif key == "f" and self.target_height_m is not None:
            self.target_height_m = _clamp(
                self.target_height_m - HEIGHT_STEP_M, MIN_HEIGHT_M, MAX_HEIGHT_M
            )
        elif key in {" ", "\n", "\r"}:
            self.cmd_vx = 0.0
            self.cmd_vy = 0.0
            self.cmd_yaw = 0.0
        elif key in {"z", "Z"}:
            if self._initialize_targets_from_tf():
                self.get_logger().info("Reinitialized height and hand targets from TF.")
        else:
            return

        self.get_logger().info(
            f"Base targets: vx={self.cmd_vx:.2f} m/s, vy={self.cmd_vy:.2f} m/s, "
            f"yaw={self.cmd_yaw:.2f} rad/s, height={self.target_height_m:.2f} m"
            if self.target_height_m is not None
            else f"Base targets: vx={self.cmd_vx:.2f} m/s, vy={self.cmd_vy:.2f} m/s, "
            f"yaw={self.cmd_yaw:.2f} rad/s"
        )

    # ------------------------------------------------------------------
    # Topic publishing.
    # ------------------------------------------------------------------

    def _publish_base_targets(self) -> None:
        twist_msg = Twist()
        twist_msg.linear.x = float(self.cmd_vx)
        twist_msg.linear.y = float(self.cmd_vy)
        twist_msg.angular.z = float(self.cmd_yaw)
        self.base_cmd_pub.publish(twist_msg)

        if self.target_height_m is not None:
            height_msg = Float32()
            height_msg.data = float(self.target_height_m)
            self.height_pub.publish(height_msg)

    def _publish_hand_targets(self) -> None:
        if self.left_pose_msg is not None:
            self.left_pose_msg.header.stamp = self.get_clock().now().to_msg()
            self.left_hand_pub.publish(self.left_pose_msg)
        if self.right_pose_msg is not None:
            self.right_pose_msg.header.stamp = self.get_clock().now().to_msg()
            self.right_hand_pub.publish(self.right_pose_msg)

    # ------------------------------------------------------------------
    # Small helpers.
    # ------------------------------------------------------------------

    def _log_waiting_for_tf(self) -> None:
        now_s = self.get_clock().now().nanoseconds * 1.0e-9
        if now_s - self._last_wait_log_s < 1.0:
            return
        self._last_wait_log_s = now_s
        self.get_logger().warn(
            "Waiting for TF frames `world -> pelvis`, `world -> left_wrist_yaw_link`, "
            "and `world -> right_wrist_yaw_link` before creating markers."
        )

    def close(self) -> None:
        self.marker_server.shutdown()
        self._restore_keyboard()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = G1HandBaseTeleop()
        rclpy.spin(node)
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
