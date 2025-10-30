#!/usr/bin/env python3
"""
Stand the Go2 robot first, THEN run the RL policy.
This ensures the robot starts from a stable state.
"""

import time
import numpy as np
import torch
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_, SportModeState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
import sys

# Add current directory to path
sys.path.append('.')
from get_obs import get_observation_with_high_state, ObservationBuffer

class Go2StandThenPolicy:
    def __init__(self, policy_path, stand_time=3.0, kp=25.0, kd=0.5, action_scale=0.5):
        """
        Stand the robot first, then run policy.
        
        Args:
            policy_path: Path to policy.pt file
            stand_time: How long to stand before starting policy (seconds)
            kp, kd, action_scale: PD control parameters
        """
        # Initialize DDS
        ChannelFactoryInitialize(1, "lo")
        
        # Load policy
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading policy from: {policy_path}")
        self.policy = torch.jit.load(policy_path, map_location=self.device)
        self.policy.eval()
        print("✓ Policy loaded")
        
        # Control parameters
        self.kp = kp
        self.kd = kd
        self.action_scale = action_scale
        self.stand_time = stand_time
        
        # Standing position (default joint positions)
        self.target_pos = np.array([
            0.0, 0.8, -1.5,  # FL
            0.0, 0.8, -1.5,  # FR
            0.0, 0.8, -1.5,  # RL
            0.0, 0.8, -1.5   # RR
        ], dtype=float)
        
        # State
        self.latest_low_state = None
        self.latest_high_state = None
        self.crc = CRC()
        self.cmd = unitree_go_msg_dds__LowCmd_()
        
        # Observation buffer
        self.obs_buffer = ObservationBuffer(use_joint_mapping=True)
        
        # Timing
        self.dt = 0.002  # 500Hz
        self.policy_dt = 0.02  # 50Hz
        self.time = 0.0
        self.last_policy_time = 0.0
        self.mode = "STANDING"  # STANDING -> POLICY
        
        # Communication
        self.low_cmd_pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.low_cmd_pub.Init()
        
        self.low_state_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.low_state_sub.Init(self.low_state_callback, 10)
        
        self.high_state_sub = ChannelSubscriber("rt/sportmodestate", SportModeState_)
        self.high_state_sub.Init(self.high_state_callback, 10)
        
        print(f"✓ Initialized")
        print(f"  Standing for {stand_time}s first")
        print(f"  Then running policy at 50Hz")
        print(f"  PD gains: kp={kp}, kd={kd}, action_scale={action_scale}")
    
    def low_state_callback(self, msg: LowState_):
        self.latest_low_state = msg
    
    def high_state_callback(self, msg: SportModeState_):
        self.latest_high_state = msg
    
    def set_commands(self, vel_x=0.0, vel_y=0.0, vel_yaw=0.0, height=0.3):
        """Set velocity commands for policy."""
        self.obs_buffer.set_commands(
            cmd_vel=[vel_x, vel_y, vel_yaw],
            height_cmd=height
        )
        print(f"✓ Commands set: vel_x={vel_x}, vel_y={vel_y}, vel_yaw={vel_yaw}, height={height}")
    
    def stand_control(self):
        """PD control to standing position."""
        if self.latest_low_state is None:
            return
        
        for i in range(12):
            q = self.latest_low_state.motor_state[i].q
            dq = self.latest_low_state.motor_state[i].dq
            
            # PD control to standing position
            tau = self.kp * (self.target_pos[i] - q) - self.kd * dq
            
            self.cmd.motor_cmd[i].q = self.target_pos[i]
            self.cmd.motor_cmd[i].dq = 0.0
            self.cmd.motor_cmd[i].kp = self.kp
            self.cmd.motor_cmd[i].kd = self.kd
            self.cmd.motor_cmd[i].tau = tau
    
    def policy_control(self):
        """Run policy inference and control."""
        if self.latest_low_state is None or self.latest_high_state is None:
            return
        
        # Get observation
        obs = get_observation_with_high_state(
            self.latest_low_state,
            self.latest_high_state,
            self.obs_buffer
        )
        
        # Run policy at 50Hz
        if self.time - self.last_policy_time >= self.policy_dt:
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                actions_tensor = self.policy(obs_tensor)
                actions = actions_tensor.cpu().numpy().squeeze()
            
            # Clip actions to [-1, 1]
            actions = np.clip(actions, -1.0, 1.0)
            
            # Update action buffer
            self.obs_buffer.update_actions(actions)
            
            self.last_policy_time = self.time
            
            # Debug output
            if self.time > self.stand_time + 1.0:  # After 1s of policy
                print(f"\n[{self.time:.2f}s] Policy running:")
                print(f"  Actions range: [{actions.min():.3f}, {actions.max():.3f}]")
                print(f"  Lin vel: [{obs[0]:.3f}, {obs[1]:.3f}, {obs[2]:.3f}]")
                print(f"  Ang vel: [{obs[3]:.3f}, {obs[4]:.3f}, {obs[5]:.3f}]")
                print(f"  Gravity: [{obs[6]:.3f}, {obs[7]:.3f}, {obs[8]:.3f}]")
        
        # Remap actions from policy order to SDK order
        actions_sdk = self.obs_buffer.remap_actions_to_sdk(self.obs_buffer.previous_actions)
        
        # Compute target positions
        target_positions = self.action_scale * actions_sdk + self.target_pos
        
        # Send motor commands
        for i in range(12):
            self.cmd.motor_cmd[i].q = target_positions[i]
            self.cmd.motor_cmd[i].dq = 0.0
            self.cmd.motor_cmd[i].kp = self.kp
            self.cmd.motor_cmd[i].kd = self.kd
            self.cmd.motor_cmd[i].tau = 0.0
    
    def control_loop(self):
        """Main control loop."""
        self.time += self.dt
        
        if self.time < self.stand_time:
            # Phase 1: Stand
            if self.mode != "STANDING":
                self.mode = "STANDING"
                print(f"\n[{self.time:.2f}s] STANDING phase...")
            self.stand_control()
        else:
            # Phase 2: Policy
            if self.mode != "POLICY":
                self.mode = "POLICY"
                print(f"\n[{self.time:.2f}s] POLICY phase started!")
                print("Robot should be stable now.\n")
            self.policy_control()
        
        # Set CRC and publish
        self.cmd.crc = self.crc.Crc(self.cmd)
        self.low_cmd_pub.Write(self.cmd)
    
    def run(self):
        """Run the controller."""
        print("\nStarting controller...")
        print("Press Ctrl+C to stop\n")
        
        # Start control thread
        control_thread = RecurrentThread(
            interval=self.dt,
            target=self.control_loop,
            name="control"
        )
        control_thread.Start()
        
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\nStopping...")
            control_thread.Stop()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Stand then run Go2 RL Policy")
    parser.add_argument("--policy", type=str, required=True, help="Path to policy.pt file")
    parser.add_argument("--stand-time", type=float, default=3.0,
                       help="Time to stand before starting policy (seconds, default: 3.0)")
    parser.add_argument("--kp", type=float, default=25.0,
                       help="Position gain (default: 25.0 from Isaac Lab)")
    parser.add_argument("--kd", type=float, default=0.5,
                       help="Velocity gain (default: 0.5 from Isaac Lab)")
    parser.add_argument("--action-scale", type=float, default=0.5,
                       help="Action scale (default: 0.5)")
    parser.add_argument("--vel-x", type=float, default=0.0,
                       help="Forward velocity command (m/s)")
    parser.add_argument("--vel-y", type=float, default=0.0,
                       help="Lateral velocity command (m/s)")
    parser.add_argument("--vel-yaw", type=float, default=0.0,
                       help="Yaw rate command (rad/s)")
    parser.add_argument("--height", type=float, default=0.3,
                       help="Height command (m)")
    
    args = parser.parse_args()
    
    # Create controller
    controller = Go2StandThenPolicy(
        policy_path=args.policy,
        stand_time=args.stand_time,
        kp=args.kp,
        kd=args.kd,
        action_scale=args.action_scale
    )
    
    # Set velocity commands
    controller.set_commands(
        vel_x=args.vel_x,
        vel_y=args.vel_y,
        vel_yaw=args.vel_yaw,
        height=args.height
    )
    
    # Run
    controller.run()


if __name__ == "__main__":
    main()
