#!/usr/bin/env python3
"""
RL Policy Controller for Unitree Go2 Robot (ROS 2 Version)
Loads a PyTorch policy and controls the robot using ROS 2
"""

import rclpy
from rclpy.node import Node
import numpy as np
import torch
import os

from unitree_api.msg import Request
from unitree_go.msg import LowCmd, LowState, IMUState, MotorState, SportModeState

from huro_py.crc_go import Crc
from get_obs2 import get_observation, ObservationBuffer

GO2_NUM_MOTOR = 12


class PolicyController(Node):
    """RL Policy controller for Unitree Go2 locomotion using ROS 2."""
    
    def __init__(self, policy_path="policy.pt", policy_freq=50, 
                 kp=25.0, kd=0.5, action_scale=0.5,
                 vel_x=0.0, vel_y=0.0, vel_yaw=0.0, height=0.3):
        super().__init__("policy_controller")
        
        # Control parameters
        self.policy_freq = policy_freq
        self.control_dt = 0.01  # 100Hz control loop (matching go2_example)
        self.policy_dt = 1.0 / policy_freq
        self.decimation = int((1.0 / self.control_dt) / policy_freq)
        
        # Time tracking
        self.time = 0.0
        self.last_policy_time = 0.0
        self.init_duration_s = 5.0  # 5 second initialization period
        
        # Control gains
        self.kp = [kp] * GO2_NUM_MOTOR
        self.kd = [kd] * GO2_NUM_MOTOR
        self.action_scale = action_scale
        
        # Motor state
        self.motors_on = 1
        self.imu = IMUState()
        self.motor = [MotorState() for _ in range(GO2_NUM_MOTOR)]
        
        # Load policy
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Loading policy from: {policy_path}")
        self.get_logger().info(f"Using device: {self.device}")
        
        if not os.path.exists(policy_path):
            raise FileNotFoundError(f"Policy file not found: {policy_path}")
        
        self.policy = torch.jit.load(policy_path, map_location=self.device)
        self.policy.eval()
        self.get_logger().info("✓ Policy loaded successfully")
        
        # Initialize observation buffer
        self.obs_buffer = ObservationBuffer()
        self.obs_buffer.set_commands(
            cmd_vel=[vel_x, vel_y, vel_yaw],
            height_cmd=height
        )
        
        # Current action (updated by policy)
        self.current_action = np.zeros(12)
        
        # Default standing position (will be set from first state message)
        self.q_init = None
        self.initialized = False
        
        # Statistics
        self.inference_count = 0
        self.tick_count = 0
        
        # ROS 2 setup
        self.topic_name = "/lowstate"  # Fixed topic name for simulation
        
        self.lowcmd_pub = self.create_publisher(LowCmd, "/lowcmd", 10)
        self.lowstate_sub = self.create_subscription(
            LowState, "/lowstate", self.low_state_handler, 10
        )
        
        self.highstate_sub = self.create_subscription(
            SportModeState, "/sportmodestate", self.high_state_handler, 10
        )
        
        # Send standdown command
        self.sport_pub = self.create_publisher(Request, "/api/sport/request", 10)
        ROBOT_SPORT_API_ID_STANDDOWN = 1005
        req = Request()
        req.header.identity.api_id = ROBOT_SPORT_API_ID_STANDDOWN
        self.sport_pub.publish(req)
        
        # Release motion control
        self.motion_pub = self.create_publisher(
            Request, "/api/motion_switcher/request", 10
        )
        ROBOT_MOTION_SWITCHER_API_RELEASEMODE = 1003
        req = Request()
        req.header.identity.api_id = ROBOT_MOTION_SWITCHER_API_RELEASEMODE
        self.motion_pub.publish(req)
        
        # Create control timer
        self.timer = self.create_timer(self.control_dt, self.control)
        
        self.get_logger().info("="*60)
        self.get_logger().info("Controller initialized")
        self.get_logger().info(f"  Control frequency: {1.0/self.control_dt:.0f}Hz (dt={self.control_dt}s)")
        self.get_logger().info(f"  Policy frequency: {self.policy_freq}Hz (dt={self.policy_dt}s)")
        self.get_logger().info(f"  Decimation: {self.decimation}")
        self.get_logger().info(f"  Commands: vx={vel_x:.2f}, vy={vel_y:.2f}, vyaw={vel_yaw:.2f}, h={height:.2f}")
        self.get_logger().info(f"  Listening on topic: {self.topic_name}")
        self.get_logger().info("="*60)
    
    def control(self):
        """Main control loop - called at 100Hz."""
        if not self.initialized:
            # Wait for first state message to set q_init
            if self.tick_count % 100 == 0:  # Print every second
                print(f"Waiting for state messages on topic: {self.topic_name}")
            self.tick_count += 1
            return
        
        low_cmd = LowCmd()
        low_cmd.head[0] = 0xFE
        low_cmd.head[1] = 0xEF
        low_cmd.gpio = 0
        
        self.time += self.control_dt
        self.tick_count += 1
        
        if self.time < self.init_duration_s:
            # Initialization phase: smoothly move to standing position
            ratio = self.clamp(self.time / self.init_duration_s, 0.0, 1.0)
            for i in range(GO2_NUM_MOTOR):
                cmd = low_cmd.motor_cmd[i]
                cmd.mode = self.motors_on
                cmd.q = (1.0 - ratio) * self.motor[i].q + ratio * self.q_init[i]
                cmd.dq = 0.0
                cmd.tau = 0.0
                cmd.kp = self.kp[i]
                cmd.kd = self.kd[i]
        else:
            # Policy control phase
            # Run policy at specified frequency
            if self.time - self.last_policy_time >= self.policy_dt:
                self.run_policy_inference()
                self.last_policy_time = self.time
            
            # Apply current action to motors
            self.apply_policy_action(low_cmd)
        
        low_cmd.crc = Crc(low_cmd)
        self.lowcmd_pub.publish(low_cmd)
    
    def run_policy_inference(self):
        """Run policy inference to get new action."""
        # Get observation from current state
        obs = get_observation(self.base_velocity, self.imu, self.motor, self.obs_buffer)
        
        # Run policy
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            actions_tensor = self.policy(obs_tensor)
            actions_policy_order = actions_tensor.squeeze(0).cpu().numpy()
        
        # Update current action
        self.current_action = actions_policy_order.copy()
        
        # Update action buffer (store in policy order)
        self.obs_buffer.update_actions(actions_policy_order)
        
        self.inference_count += 1
        # print(f"  Obs[0:3] (lin_vel): {obs[0:3]}")
        # print(f"  Obs[3:6] (ang_vel): {obs[3:6]}")
        # print(f"  Obs[6:9] (gravity_b): {obs[6:9]}")
        # print(f"  Obs[9:12] (cmd_vel): {obs[9:12]}")
        # print(f"  Obs[12] (cmd_height): {obs[12]}")
        # print(f"  Obs[13:25] (joint_pos): {obs[13:25]}")
        # print(f"  Obs[25:37] (joint_vel): {obs[25:37]}")
        # print(f"  Obs[37:49] (prev_actions): {obs[37:49]}")
    
        if self.inference_count % 50 == 0:  # Log every 50 inferences (~1 second at 50Hz)
            print(f"Inferences: {self.inference_count}, Time: {self.time:.2f}s")

            
    def apply_policy_action(self, low_cmd: LowCmd):
        """Apply current policy action to motor commands."""
        # Convert action from policy order to SDK order
        actions_sdk_order = self.obs_buffer.remap_actions_to_sdk(self.current_action)
        
        # Calculate target positions
        target_positions = self.obs_buffer.default_pos + actions_sdk_order * self.action_scale
        
        # Set motor commands
        for i in range(GO2_NUM_MOTOR):
            cmd = low_cmd.motor_cmd[i]
            cmd.mode = self.motors_on
            cmd.q = target_positions[i]
            cmd.dq = 0.0
            cmd.tau = 0.0
            cmd.kp = self.kp[i]
            cmd.kd = self.kd[i]
    
    def low_state_handler(self, msg: LowState):
        """Handle incoming state messages."""
        self.imu = msg.imu_state
        for i in range(GO2_NUM_MOTOR):
            self.motor[i] = msg.motor_state[i]
        
        # Initialize q_init from first message
        if not self.initialized:
            self.q_init = [self.motor[i].q for i in range(GO2_NUM_MOTOR)]
            self.initialized = True
            print(f"Initialized with standing position: {self.q_init[:3]}...")
        
        # Handle controller message (emergency stop)
        self.controller_msg = msg.wireless_remote
        if self.controller_msg[3] == 1:
            self.motors_on = 0
            print("Emergency stop activated!")
    
    def high_state_handler(self, msg: SportModeState):
        self.base_velocity = msg.velocity[:3]
    
    def clamp(self, value, low, high):
        """Clamp value between low and high."""
        if value < low:
            return low
        if value > high:
            return high
        return value


def main(args=None):
    import argparse
    
    # Parse command line arguments (before rclpy.init)
    parser = argparse.ArgumentParser(description="Go2 RL Policy Controller (ROS 2)")
    parser.add_argument("--policy", type=str, default="policy.pt",
                       help="Path to policy.pt file")
    parser.add_argument("--policy-freq", type=int, default=50,
                       help="Policy inference frequency in Hz (default: 50)")
    parser.add_argument("--kp", type=float, default=25.0,
                       help="Position gain/stiffness (default: 25.0)")
    parser.add_argument("--kd", type=float, default=0.5,
                       help="Velocity gain/damping (default: 0.5)")
    parser.add_argument("--action-scale", type=float, default=0.5,
                       help="Scale factor for policy actions (default: 0.5)")
    parser.add_argument("--vel-x", type=float, default=0.5,
                       help="Forward velocity command (m/s)")
    parser.add_argument("--vel-y", type=float, default=0.0,
                       help="Lateral velocity command (m/s)")
    parser.add_argument("--vel-yaw", type=float, default=0.0,
                       help="Yaw rate command (rad/s)")
    parser.add_argument("--height", type=float, default=0.3,
                       help="Height command (m)")
    
    parsed_args, ros_args = parser.parse_known_args()
    
    # Initialize ROS 2
    rclpy.init(args=None)
    
    # Create controller node
    node = PolicyController(
        policy_path=parsed_args.policy,
        policy_freq=parsed_args.policy_freq,
        kp=parsed_args.kp,
        kd=parsed_args.kd,
        action_scale=parsed_args.action_scale,
        vel_x=parsed_args.vel_x,
        vel_y=parsed_args.vel_y,
        vel_yaw=parsed_args.vel_yaw,
        height=parsed_args.height
    )
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Print statistics before destroying node (avoid rosout errors)
        print("\n" + "="*60)
        print("Shutting down...")
        print(f"Total inferences: {node.inference_count}")
        print(f"Total ticks: {node.tick_count}")
        print(f"Simulation time: {node.time:.2f}s")
        print("="*60)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
