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

"""
import rclpy
from rclpy.node import Node
import numpy as np
from ament_index_python.packages import get_package_share_directory
import torch
import os
import time

from unitree_api.msg import Request
from unitree_go.msg import LowCmd, LowState
from huro.msg import SpaceMouseState


from huro_py.crc_go import Crc
from huro_py.get_obs import get_obs_low_state
from huro_py.utils import Mapper, MockController
from sensor_msgs.msg import Joy

np.set_printoptions(precision=3)


class Go2PolicyController(Node):
    """RL Policy controller for Unitree Go2 locomotion."""

    def __init__(
        self,
        policy_name="policy.pt",
        sim=True,
        vel=[0.0, 0.0, 0.0]
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
        params = []
        
        
        if sim:
            params.append(rclpy.parameter.Parameter("use_sim_time", value=True))
        super().__init__("go2_policy_controller", parameter_overrides=params)
        # Verify we're using sim time if the sim param is set to True
        use_sim_time = self.get_parameter("use_sim_time").value
        print(f"[INFO] use_sim_time: {use_sim_time}")

        self.step_dt = 1 / 50  # policy freq = 50Hz
        self.run_policy = True # set to false to rely on joy buttons to lauch the policy
        self.vel = [float(vel[0]), float(vel[1]), float(vel[2])]

        # Emergency mode
        self.emergency_mode = False
        self.emergency_mode_start_time = None
        self.last_commanded_positions = None

        # Load policy model
        share = get_package_share_directory("huro")

        if policy_name is None:
            policy_name = "policy_asymmetric9.pt"
   

        policy_path = os.path.join(share, "resources", "models", "go2", policy_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[INFO] Loading policy from: {policy_name}")
        print(f"[INFO] Using device: {self.device}")

        if not os.path.exists(policy_path):
            raise FileNotFoundError(f"Policy file not found: {policy_path}")
        self.policy = torch.jit.load(policy_path, map_location=self.device)
        self.policy.eval()
        print("[INFO] Policy loaded successfully")

        # Initialize the mapper for the joints and the actions
        mapping_file = "physx_to_mujoco_go2.yaml"
        if not sim:
            mapping_file = "sim_to_real.yaml"

        mapping_path = policy_path = os.path.join(
            share, "resources", "mappings", "go2", mapping_file
        )

        if not sim:
            self.default_pos_sdk = np.array(
                [
                    -0.1,
                    0.8,
                    -1.5,  # FR: hip, thigh, calf (actuators 0-2)
                    0.1,
                    0.8,
                    -1.5,  # FL: hip, thigh, calf (actuators 3-5)
                    -0.1,
                    1.0,
                    -1.5,  # RR: hip, thigh, calf (actuators 6-8)
                    0.1,
                    1.0,
                    -1.5,  # RL: hip, thigh, calf (actuators 9-11)
                ]
            )
        else:
            self.default_pos_sdk = np.array(
                [
                    0.1,
                    0.8,
                    -1.5,  # FL: hip, thigh, calf (actuators 0-2)
                    -0.1,
                    0.8,
                    -1.5,  # FR: hip, thigh, calf (actuators 3-5)
                    0.1,
                    1.0,
                    -1.5,  # RL: hip, thigh, calf (actuators 6-8)
                    -0.1,
                    1.0,
                    -1.5,  # RR: hip, thigh, calf (actuators 9-11)
                ]
            )
            
        self.mapper = Mapper(
            mapping_yaml_path=mapping_path, default_pos_sdk=self.default_pos_sdk
        )

        # Store latest action (for use between policy updates)
        self.current_action = np.zeros(12)

        # Store latest messages
        self.latest_low_state = None
        self.controller_state = None

        self.kp = 25.0  # Position gain
        self.kd = 0.5  # Velocity gain
        self.kp_p = 25.0  # Position gain
        self.kd_p = 0.5  # Velocity gain
        self.action_scale = 0.25  # Scale policy output

        # Standing position (default joint positions but coud be different)
        self.stand_pos = self.default_pos_sdk
        
        self.time_to_stand = 5.0  # Time to reach the standing position

        # Statistics - initialize BEFORE callbacks
        self.tick_count = 0

        # Initialize communication
        self.low_cmd_pub = self.create_publisher(LowCmd, "/lowcmd", 10)

        if sum(self.vel) != 0.0:
            self.controller_state = MockController(self.vel)
        else:
            self.joy_sub = self.create_subscription(Joy, "/joy", self.joy_callback, 10)

        # Get low lovel data from robot
        self.low_state_sub = self.create_subscription(
            LowState, "/lowstate", self.low_state_callback, 10
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

    def low_state_callback(self, msg: LowState):
        """Log low state message."""
        self.latest_low_state = msg

    def joy_callback(self, msg: SpaceMouseState):
        """Log spacemouse state"""
        self.controller_state = msg
        

    def emergency_mode_control(self):
        """Smoothly reduce gains and torque to zero over release_duration."""
        if self.latest_low_state is None:
            return

        # Calculate progress (0 to 1)
        release_duration = 5
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
        cmd.head[0] = 0xFE
        cmd.head[1] = 0xEF
        cmd.gpio = 0
        ratio = min((self.curr_time - self.start_time).nanoseconds * 1e-9 / self.time_to_stand, 1.0)
        self.last_commanded_positions = [(1.0 - ratio) * self.latest_low_state.motor_state[i].q + ratio * self.stand_pos[i] for i in range(12)]

        for i in range(12):
            motorcmd = cmd.motor_cmd[i]
            motorcmd.mode = 1
            motorcmd.q = self.last_commanded_positions[i]
            motorcmd.dq = 0.0
            motorcmd.tau = 0.0
            motorcmd.kp = self.kp
            motorcmd.kd = self.kd

        # Calculate CRC and publish
        cmd.crc = Crc(cmd)
        self.low_cmd_pub.publish(cmd)

    def send_motor_commands(self):
        """Send motor commands to the robot based on current action."""
        # Convert current action from policy order to SDK order
        actions_sdk_order = self.mapper.actions_policy_to_sdk(self.current_action)

        self.last_commanded_positions = (
            self.mapper.default_pos_sdk
        ) + actions_sdk_order * self.action_scale

        cmd = LowCmd()
        cmd.head[0] = 0xFE
        cmd.head[1] = 0xEF
        cmd.level_flag = 0xFF
        cmd.gpio = 0

        for i in range(12):
            cmd.motor_cmd[i].mode = 0x01  # PMSM mode
            cmd.motor_cmd[i].q = self.last_commanded_positions[i]
            cmd.motor_cmd[i].kp = self.kp_p
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].kd = self.kd_p
            cmd.motor_cmd[i].tau = 0.0

        # Calculate CRC and publish
        cmd.crc = Crc(cmd)
        self.low_cmd_pub.publish(cmd)

    def run(self):
        """Main control loop running at control_freq Hz."""
        print(self.latest_low_state is None)
        try:
            if self.latest_low_state is not None and self.controller_state is not None :
                if self.tick_count == 0:
                    self.start_time = self.get_clock().now()
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

        emergency_cond = (
            self.controller_state.buttons[8]
            and self.controller_state.buttons[9]
            and self.run_policy
            or self.emergency_mode
        )
        policy_run_cond = (
            self.controller_state.buttons[8] or self.controller_state.buttons[9]
        )

        if emergency_cond or self.emergency_mode:
            if not self.emergency_mode:
                self.emergency_mode_start_time = self.get_clock().now()
            self.emergency_mode = True
            self.emergency_mode_control()

        elif policy_run_cond:
            self.run_policy = True

        elif (self.curr_time - self.start_time).nanoseconds * 1e-9 <= self.time_to_stand or self.run_policy == False:
            self.stand_control()
        # Run policy
        elif (
            self.curr_time - self.start_time
        ).nanoseconds * 1e-9 >= self.time_to_stand and self.run_policy:
            self.policy_control()

    def policy_control(self):

        obs = get_obs_low_state(
            self.latest_low_state,
            self.controller_state,
            height=0.30,
            prev_actions=self.current_action,
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
        "--sim", type=bool, default=True, help="Wether to use simulation or real robot"
    )
    
    parser.add_argument(
        "--vx", type=float, default=0.0, help="Velocity along x axis"
    )
    
    parser.add_argument(
        "--vy", type=float, default=0.0, help="Velocity along x axis"
    )
    
    parser.add_argument(
        "--wz", type=float, default=0.0, help="Velocity along x axis"
    )

    # Parse only known args to allow ROS args to pass through
    args, unknown = parser.parse_known_args()

    # Initialize DDS communication - must be called before creating nodes
    rclpy.init()

    # Create controller
    node = Go2PolicyController(
        policy_name=args.policy,
        sim=args.sim,
        vel=[args.vx, args.vy, args.wz]
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
