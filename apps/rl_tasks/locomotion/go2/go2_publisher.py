#!/usr/bin/env python3

"""
RL Policy Controller for Unitree Go2 Robot
Loads a PyTorch policy and controls the robot at 50Hz
"""

"""
TO RUN:

ros2 launch huro go2_rviz.launch.py
ros2 run huro spacemouse_publisher.py
ros2 run huro sim_go2
ros2 run huro go2_publisher.py


python run_policy.py \
    --vel-x -0.5 \
    --vel-y 0.1 \
    --vel-yaw 0.3 \
    --kd 0.5 \
    --kp 25.0

Or with all parameters:

python run_policy.py \
    --policy policy.pt \
    --vel-x 0.3 \
    --vel-y 0.0 \
    --vel-yaw 0.0 \
    --action-scale 0.5 \
    --kp 25.0 \
    --kd 0.5 \
    --control-freq 200
"""
import rclpy
from rclpy.node import Node
import numpy as np
from ament_index_python.packages import get_package_share_directory
import torch
import os
import time

from unitree_api.msg import Request
from unitree_go.msg import LowCmd, LowState, SportModeState
from huro.msg import SpaceMouseState


from huro_py.crc_go import Crc
from huro_py.get_obs import get_obs_high_state, get_obs_low_state
from huro_py.mapping import Mapper

np.set_printoptions(precision=3)


class Go2PolicyController(Node):
    """RL Policy controller for Unitree Go2 locomotion."""

    def __init__(
        self, policy_name = "policy.pt", high_state = False, training_type = "normal"
    ):
        """
        Initialize the policy controller.

        Args:
            policy_path: Path to the policy.pt file
            policy_freq: Policy inference frequency in Hz (default: 50Hz)
            control_freq: Motor command frequency in Hz (default: 500Hz, simulation dt=0.002)
            kp: Position gain/stiffness (default: 25.0) - MUST match training value!
            kd: Velocity gain/damping (default: 0.5)
            action_scale: Scale factor for policy actions (default: 0.25)
        """
        super().__init__("go2_policy_controller")

        self.step_dt = 1 / 50 # policy freq = 50Hz
        self.control_gait = 0.02 # the phase updated at 2Hz
        self.run_policy = False
        self.high_state = high_state

        # Emergency mode
        self.emergency_mode = False
        self.emergency_mode_start_time = None
        self.last_commanded_positions = None


        # Load policy model
        share = get_package_share_directory("huro")
        
        if policy_name is None:
            if not self.high_state:
                if training_type == "asymmetric":
                    policy_name = "policy_asymmetric4.pt"
                elif training_type == "student":
                    policy_name = "policy_student.pt"
                else:
                    policy_name = "policy_low_state.pt"
            else:
                policy_name = "policy_teacher.pt"
                
        policy_path = os.path.join(share, "resources", "models", policy_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[INFO] Loading policy from: {policy_name}")
        print(f"[INFO] Using device: {self.device}")

        if not os.path.exists(policy_path):
            raise FileNotFoundError(f"Policy file not found: {policy_path}")
        self.policy = torch.jit.load(policy_path, map_location=self.device)
        self.policy.eval()
        print("[INFO] Policy loaded successfully")

        # Initialize the mapper for the joints and the actions
        mapping_path = policy_path = os.path.join(
            share, "resources", "mappings", "physx_to_mujoco_go2.yaml"
        )
        self.mapper = Mapper(mapping_yaml_path=mapping_path)

        # Store latest action (for use between policy updates)
        self.current_action = np.zeros(12)

        # Store latest messages
        self.latest_low_state = None
        self.spacemouse_state = None
        if self.high_state:
            self.latest_high_state = None
            self.prev_vel = None

        self.kp = 35.0  # Position gain
        self.kd = 1.0  # Velocity gain
        self.action_scale = 0.5  # Scale policy output

        # Standing position (default joint positions but coud be different)
        self.stand_pos = np.array(
            [
                0.0,
                0.8,
                -1.5,  # FL
                0.0,
                0.8,
                -1.5,  # FR
                0.0,
                0.8,
                -1.5,  # RL
                0.0,
                0.8,
                -1.5,  # RR
            ],
            dtype=float,
        )
        self.time_to_stand = 3.0  # Time to reach the standing position

        # Statistics - initialize BEFORE callbacks
        self.tick_count = 0
        self.start_time = self.get_clock().now()

        # Initialize communication
        self.low_cmd_pub = self.create_publisher(LowCmd, "/lowcmd", 10)

        # To read data from spacemouse
        self.spacemouse_sub = self.create_subscription(
            SpaceMouseState, "/spacemouse_state", self.spacemouse_callback, 10
        )

        # Get low lovel data from robot
        self.low_state_sub = self.create_subscription(
            LowState, "/lowstate", self.low_state_callback, 10
        )

        # Get high lovel data from robot
        if self.high_state:
            self.high_state_sub = self.create_subscription(
                SportModeState, "/sportmodestate", self.high_state_callback, 10
            )

        self.motion_pub = self.create_publisher(
            Request, "/api/motion_switcher/request", 10
        )
        ROBOT_MOTION_SWITCHER_API_RELEASEMODE = 1003
        req = Request()
        req.header.identity.api_id = ROBOT_MOTION_SWITCHER_API_RELEASEMODE
        self.motion_pub.publish(req)

        time.sleep(1)

        self.timer = self.create_timer(self.step_dt, self.run)

        print(f"  Policy controller initialized")
        print(f"  Policy runs at: {1 / self.step_dt}Hz")

    def high_state_callback(self, msg: SportModeState):
        """Log high state message."""
        self.latest_high_state = msg

    def low_state_callback(self, msg: LowState):
        """Log low state message."""
        self.latest_low_state = msg

    def spacemouse_callback(self, msg: SpaceMouseState):
        """Log spacemouse state"""
        self.spacemouse_state = msg

    def emergency_mode_control(self):
        """Smoothly reduce gains and torque to zero over release_duration."""
        if self.latest_low_state is None:
            return

        # Calculate progress (0 to 1)
        release_duration = 2
        elapsed = (
            self.get_clock().now() - self.emergency_mode_start_time
        ).nanoseconds * 1e-9
        r = min(elapsed / release_duration, 1.0)

        alpha = 1.0 - (1.0 - r) ** 10

        # Gradually reduce gains from current values to zero
        current_kp = self.kp * (1.0 - alpha)
        current_kd = self.kd * (1.0 - alpha)

        cmd = LowCmd()

        for i in range(12):
            q = self.latest_low_state.motor_state[i].q
            dq = self.latest_low_state.motor_state[i].dq

            # Compute diminishing torque
            tau = current_kp * (self.last_commanded_positions[i] - q) - current_kd * dq

            cmd.motor_cmd[i].mode = 0x01
            cmd.motor_cmd[i].q = self.last_commanded_positions[i]
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].kp = current_kp
            cmd.motor_cmd[i].kd = current_kd
            cmd.motor_cmd[i].tau = tau

        cmd.crc = Crc(cmd)
        self.low_cmd_pub.publish(cmd)

    def stand_control(self):
        """PD control to standing position."""
        if self.latest_low_state is None:
            return
        cmd = LowCmd()
        r = min(
            (self.get_clock().now() - self.start_time).nanoseconds
            * 1e-9
            / self.time_to_stand,
            1,
        )
        alpha = 1.0 - (1.0 - r) ** 3

        for i in range(12):
            curr_kd = alpha * self.kd
            curr_kp = alpha * self.kp
            # tau = curr_kp * (self.stand_pos[i] - q) - curr_kd * dq
            cmd.motor_cmd[i].mode = 1
            cmd.motor_cmd[i].q = self.stand_pos[i]
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].kp = curr_kp
            cmd.motor_cmd[i].kd = curr_kd
            cmd.motor_cmd[i].tau = 0.0

        # Calculate CRC and publish
        cmd.crc = Crc(cmd)
        self.low_cmd_pub.publish(cmd)

    def send_motor_commands(self):
        """Send motor commands to the robot based on current action."""
        # Convert current action from policy order to SDK order
        actions_sdk_order = self.mapper.actions_policy_to_sdk(self.current_action)
        
        # Store last commanded positions for potential emergency mode
        self.last_commanded_positions = (
            self.mapper.default_pos_sdk
        )   + actions_sdk_order * self.action_scale

        cmd = LowCmd()
        cmd.head[0] = 0xFE
        cmd.head[1] = 0xEF
        cmd.level_flag = 0xFF
        cmd.gpio = 0
        # target_positions = self.mapper.default_pos_sdk + actions_sdk_order * self.action_scale
        # Set motor commands
        for i in range(12):
            cmd.motor_cmd[i].mode = 0x01  # PMSM mode
            cmd.motor_cmd[i].q = self.last_commanded_positions[i]
            cmd.motor_cmd[i].kp = self.kp
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].kd = self.kd
            cmd.motor_cmd[i].tau = 0.0

        # Calculate CRC and publish
        cmd.crc = Crc(cmd)
        self.low_cmd_pub.publish(cmd)

    def run(self):
        """Main control loop running at control_freq Hz."""

        # Robot in standing position for the begining

        try:
            # Process current state (callbacks update latest_low_state and latest_high_state)
            if self.high_state:
                cond = self.latest_low_state is not None and self.latest_high_state is not None and self.spacemouse_state is not None
            else:
                cond = self.latest_low_state is not None and self.spacemouse_state is not None
            if cond:
                self.process_control_step()
            else:
                print("Waiting for robot state...")
                self.start_time = self.get_clock().now()

        except KeyboardInterrupt:
            print("\n\n" + "=" * 60)
            print("Shutting down...")
            print(f"Total inferences: {self.tick_count}")
            print(f"Total ticks: {self.tick_count}")
            elapsed = time.time() - self.start_time
            print(f"Real time elapsed: {elapsed:.2f}s")
            print(
                f"Average policy frequency: {self.tick_count / elapsed:.1f}Hz (target: {1 / self.step_dt}Hz)"
            )
            print("=" * 60)

    def process_control_step(self):
        """Process one control step (called at control_freq Hz)."""
        self.tick_count += 1
        self.curr_time = self.get_clock().now()       

        if (
            self.spacemouse_state.button_1_pressed
            and self.spacemouse_state.button_2_pressed
            and self.run_policy
            or self.emergency_mode
        ):
            if not self.emergency_mode:
                self.emergency_mode_start_time = self.get_clock().now()
            self.emergency_mode = True
            self.emergency_mode_control()
        elif self.spacemouse_state.button_1_pressed:
            self.run_policy = True
        elif (
            self.curr_time - self.start_time
        ).nanoseconds * 1e-9 <= self.time_to_stand:
            print((self.curr_time - self.start_time).nanoseconds * 1e-9)
            self.stand_control()
        # Run policy
        elif (
            self.curr_time - self.start_time
        ).nanoseconds * 1e-9 >= self.time_to_stand:# and self.run_policy:
            self.policy_control()
            
    def policy_control(self):
        # Get observation
        phase = (self.phase + self.step_dt * self.control_gait) % 1
        if self.high_state:
            obs = get_obs_high_state(
                self.latest_low_state,
                self.latest_high_state,
                self.spacemouse_state,
                height=0.30,
                prev_actions=self.current_action,
                mapper=self.mapper,
                previous_vel= self.prev_vel
            )
            self.prev_vel = obs[0:3]
        else:
            obs = get_obs_low_state(
                self.latest_low_state,
                self.spacemouse_state,
                height=0.30,
                prev_actions=self.current_action,
                phase = phase
                mapper=self.mapper,
            )
        with torch.no_grad():
            obs_tensor = torch.tensor(
                obs, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            actions_tensor = self.policy(obs_tensor)
        actions_policy_order = actions_tensor.squeeze(0).cpu().numpy()
        self.current_action = actions_policy_order.copy()
        self.send_motor_commands()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Go2 RL Policy Controller")
    parser.add_argument(
        "--policy", type=str, default=None, help="Path to policy.pt file"
    )

    parser.add_argument(
        "--high_state", type=bool, default=False, help="Wether to use base_vel sent by sportsmode state or not"
    )
    
    parser.add_argument(
        "--training_type", type=str, default="normal", help="The type of training use (normal, asymmetric or student)"
    )

    args = parser.parse_args()

    # Initialize DDS communication
    rclpy.init(args=None)

    # Create controller
    node = Go2PolicyController(
        policy_name=args.policy,
        high_state= args.high_state,
        training_type = args.training_type
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("Shutting down...")
        print("=" * 60)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
