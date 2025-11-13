#!/usr/bin/env python3

"""
RL Policy Controller for Unitree Go2 Robot
Loads a PyTorch policy and controls the robot at 50Hz
"""

"""
TO RUN:
First run the simulation, then:

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
import torch
import os
import time

from unitree_api.msg import Request
from unitree_go.msg import LowCmd, LowState, SportModeState
from huro.msg import SpaceMouseState


from huro_py.crc_go import Crc
from get_obs import get_observation_projectedgravity, get_observation_imuquat
from mapping import Mapper 
np.set_printoptions(precision=3)


class Go2PolicyController(Node):
    """RL Policy controller for Unitree Go2 locomotion."""
    
    def __init__(self, policy_path, policy_freq=50, kp=60.0, kd=5.0, action_scale=0.5, raw = False):
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
                
        self.step_dt = 1 / policy_freq
        
        # Simulation time tracking

        self.last_emergency_stop_time = 0.0
        
        # Load policy
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Loading policy from: {policy_path}")
        print(f"[INFO] Using device: {self.device}")
        
        self.get_obs = get_observation_projectedgravity
        if raw:
            policy_path = "policy_raw.pt"
            self.get_obs = get_observation_imuquat
            
        if not os.path.exists(policy_path):
            raise FileNotFoundError(f"Policy file not found: {policy_path}")
        
        self.policy = torch.jit.load(policy_path, map_location=self.device)
        self.policy.eval()
        print("[INFO] Policy loaded successfully")
        
        # Initialize observation buffer
        self.mapper = Mapper()
        
        # Store latest action (for use between policy updates)
        self.current_action = np.zeros(12)
        
        # Store latest messages
        self.latest_low_state = None
        self.latest_high_state = None
        self.latest_spacemouse_state = None
        
        # Control parameters (MUST match training values!)
        self.kp = kp  # Position gain
        self.kd = kd  # Velocity gain
        self.action_scale = action_scale  # Scale policy output
        
        # Standing position (default joint positions but coud be different)
        self.target_pos = np.array([
            0.0, 0.8, -1.5,  # FL
            0.0, 0.8, -1.5,  # FR
            0.0, 0.8, -1.5,  # RL
            0.0, 0.8, -1.5   # RR
        ], dtype=float)
        
        
        # Statistics - initialize BEFORE callbacks
        self.tick_count = 0
        self.start_time = None  # Will be set when run() is called
        
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
        self.high_state_sub = self.create_subscription(
            SportModeState, "/sportmodestate", self.high_state_callback, 10
        )
        
        # This part handles th release mode
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
        
        self.timer = self.create_timer(1/50, self.run)
        
        print(f"  Policy controller initialized")
        print(f"  Policy runs at: {1 / self.step_dt}Hz")

    
    def high_state_callback(self, msg: SportModeState):
        """Store high state message."""
        self.latest_high_state = msg
        
    
    def low_state_callback(self, msg: LowState):
        """Store low state message."""
        self.latest_low_state = msg
        
    def spacemouse_callback(self, msg: SpaceMouseState):
        self.latest_spacemouse_state = msg
        
    def stand_control(self):
        """PD control to standing position."""
        if self.latest_low_state is None:
            return
        
        cmd = LowCmd()
        
        for i in range(12):
            q = self.latest_low_state.motor_state[i].q
            dq = self.latest_low_state.motor_state[i].dq
            
            # PD control to standing position
            tau = self.kp * (self.target_pos[i] - q) - self.kd * dq
            
            cmd.motor_cmd[i].q = self.target_pos[i]
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].kp = self.kp
            cmd.motor_cmd[i].kd = self.kd
            cmd.motor_cmd[i].tau = tau
            
        # Calculate CRC and publish
        cmd.crc = Crc(cmd)
        self.low_cmd_pub.publish(cmd)
    
    def send_motor_commands(self):
        """Send motor commands to the robot based on current action."""
        # Convert current action from policy order to SDK order
        actions_sdk_order = self.mapper.actions_policy_to_sdk(self.current_action)
        
        cmd = LowCmd()
        cmd.head[0] = 0xFE
        cmd.head[1] = 0xEF
        cmd.level_flag = 0xFF
        cmd.gpio = 0
        target_positions = self.mapper.default_pos_sdk + actions_sdk_order * self.action_scale
        # Set motor commands
        for i in range(12):
            cmd.motor_cmd[i].mode = 0x01  # PMSM mode
            cmd.motor_cmd[i].q = target_positions[i]
            cmd.motor_cmd[i].kp = self.kp
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].kd = self.kd
            cmd.motor_cmd[i].tau = 0.0
        
        # Calculate CRC and publish
        cmd.crc = Crc(cmd)
        self.low_cmd_pub.publish(cmd)
    
    def run(self):
        """Main control loop running at control_freq Hz."""
        self.start_time = time.time()
        
        # Robot in standing position for the begining
        
        try:            
            # Process current state (callbacks update latest_low_state and latest_high_state)
            if self.latest_low_state is not None and self.latest_high_state is not None and self.latest_spacemouse_state is not None:
                self.process_control_step()
            else :
                print("Waiting for robot state...")
               
        except KeyboardInterrupt:
            print("\n\n" + "="*60)
            print("Shutting down...")
            print(f"Total inferences: {self.tick_count}")
            print(f"Total ticks: {self.tick_count}")
            elapsed = time.time() - self.start_time
            print(f"Real time elapsed: {elapsed:.2f}s")
            print(f"Average policy frequency: {self.tick_count / elapsed:.1f}Hz (target: {1 / self.step_dt}Hz)")
            print("="*60)
    
    def process_control_step(self):
        """Process one control step (called at control_freq Hz)."""
        self.tick_count += 1
                
        # Get observation
        obs = self.get_obs(
            self.latest_low_state, 
            self.latest_high_state, 
            self.latest_spacemouse_state,
            height= 0.3,
            prev_actions= self.current_action,
            mapper= self.mapper
        )        
        print(obs.size)
        
        # # Run policy at 50Hz based on simulation time
        # keyboard.read_key() # an important inclusion thanks to @wkl
        if self.latest_spacemouse_state.button_1_pressed and self.latest_spacemouse_state.button_2_pressed:
            self.last_emergency_stop_time = time.perf_counter()

        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            actions_tensor = self.policy(obs_tensor)
            actions_policy_order = actions_tensor.squeeze(0).cpu().numpy()
        
        # Update current action
        self.current_action = actions_policy_order.copy()
                
        # Send motor commands every tick (using latest action)
        if self.tick_count <= (3 + self.last_emergency_stop_time) * 1 / self.step_dt:
            self.stand_control()
        if self.tick_count >= (3 + self.last_emergency_stop_time) * 1 / self.step_dt:
            self.send_motor_commands()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Go2 RL Policy Controller")
    parser.add_argument("--policy", type=str, default="policy.pt",
                       help="Path to policy.pt file")
    parser.add_argument("--policy-freq", type=int, default=50,
                       help="Policy inference frequency in Hz (default: 50)")
    parser.add_argument("--kp", type=float, default=25.0,
                       help="Position gain/stiffness (default: 25.0 - lower for simulation stability)")
    parser.add_argument("--kd", type=float, default=0.5,
                       help="Velocity gain/damping (default: 0.5 - lower for simulation stability)")
    parser.add_argument("--action-scale", type=float, default=0.5,
                       help="Scale factor for policy actions (default: 0.5)")
    parser.add_argument("--raw", type=bool, default=False,
                       help="Wether to use raw IMU data or not")
    
    args = parser.parse_args()
    
    # Initialize DDS communication
    rclpy.init(args=None)
    
    # Create controller
    node = Go2PolicyController(
        policy_path=args.policy,
        policy_freq=args.policy_freq,
        kp=args.kp,
        kd=args.kd,
        action_scale=args.action_scale,
        raw = args.raw
    )
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n" + "="*60)
        print("Shutting down...")
        print("="*60)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
