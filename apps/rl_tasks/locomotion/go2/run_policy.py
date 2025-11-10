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
import keyboard

from unitree_api.msg import Request
from unitree_go.msg import LowCmd, LowState, SportModeState
from huro.msg import SpaceMouseState


from huro_py.crc_go import Crc
from get_obs import get_observation_with_high_state, ObservationBuffer
np.set_printoptions(precision=3)


class Go2PolicyController(Node):
    """RL Policy controller for Unitree Go2 locomotion."""
    
    def __init__(self, policy_path,vel_x = 0.5, vel_y = 0.0, yaw = 0.0, height = 0.3, policy_freq=50, control_freq=500, 
                 kp=60.0, kd=5.0, action_scale=0.5):
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
                
        self.obs = np.zeros(49)
        self.policy_freq = policy_freq
        self.control_freq = control_freq
        self.control_dt = 1 / self.control_freq
        self.policy_dt = 1.0 / policy_freq  # e.g., 0.02s for 50Hz
        self.control_dt = 1.0 / control_freq  # e.g., 0.002s for 500Hz
        self.decimation = int(control_freq / policy_freq)  # Auto-calculate decimation
        
        # Simulation time tracking
        self.sim_time = 0.0
        self.last_policy_time = 0.0
        self.last_keyboard_press_time = 0.0
        
        # Load policy
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading policy from: {policy_path}")
        print(f"Using device: {self.device}")
        
        if not os.path.exists(policy_path):
            raise FileNotFoundError(f"Policy file not found: {policy_path}")
        
        self.policy = torch.jit.load(policy_path, map_location=self.device)
        self.policy.eval()
        print("✓ Policy loaded successfully")
        
        # Initialize observation buffer
        self.obs_buffer = ObservationBuffer()
        
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
        self.vel_x = vel_x
        self.vel_y = vel_y
        self.yaw = yaw
        self.height = height
        
        # Standing position (default joint positions)
        self.target_pos = np.array([
            0.0, 0.8, -1.5,  # FL
            0.0, 0.8, -1.5,  # FR
            0.0, 0.8, -1.5,  # RL
            0.0, 0.8, -1.5   # RR
        ], dtype=float)
        
        
        # Statistics - initialize BEFORE callbacks
        self.inference_count = 0
        self.tick_count = 0
        self.start_time = None  # Will be set when run() is called
        
        # Initialize communication
        self.low_cmd_pub = self.create_publisher(LowCmd, "/lowcmd", 10)
        
        self.spacemouse_sub = self.create_subscription(
            SpaceMouseState, "/spacemouse_state", self.spacemouse_callback, 10
        )
        
        # Subscribe to state (for reading, not callback-driven)
        self.low_state_sub = self.create_subscription(
            LowState, "/lowstate", self.low_state_callback, 10
        )
        
        self.high_state_sub = self.create_subscription(
            SportModeState, "/sportmodestate", self.high_state_callback, 10
        )
        
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
        
        self.timer = self.create_timer(self.control_dt, self.run)
        
        print(f"  Controller initialized")
        print(f"  Control frequency: {self.control_freq}Hz (dt={self.control_dt}s)")
        print(f"  Policy frequency: {self.policy_freq}Hz (dt={self.policy_dt}s)")
        print(f"  Decimation: {self.decimation} (policy runs every {self.decimation} ticks)")
    
    def set_commands(self, vel_x=-0.5, vel_y=0.0, vel_yaw=0.0, height=0.3):
        """Set desired velocity commands for the policy."""
        self.obs_buffer.set_commands(
            cmd_vel=[vel_x, vel_y, vel_yaw],
            height_cmd=height
        )
    
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
        actions_sdk_order = self.obs_buffer.remap_actions_to_sdk(self.current_action)
        
        cmd = LowCmd()
        cmd.head[0] = 0xFE
        cmd.head[1] = 0xEF
        cmd.level_flag = 0xFF
        cmd.gpio = 0
        target_positions = self.obs_buffer.default_pos + actions_sdk_order * self.action_scale
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
            
        
            step_start = time.perf_counter()
            
            # Process current state (callbacks update latest_low_state and latest_high_state)
            if self.latest_low_state is not None and self.latest_high_state is not None:
                self.process_control_step()
            else :
                print("Waiting for robot state...")
            
            # Sleep to maintain control frequency
            time_until_next_step = self.control_dt - (time.perf_counter() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
                    
        except KeyboardInterrupt:
            print("\n\n" + "="*60)
            print("Shutting down...")
            if self.inference_count > 0:
                print(f"Total inferences: {self.inference_count}")
                print(f"Total ticks: {self.tick_count}")
                print(f"Simulation time: {self.sim_time:.2f}s")
                elapsed = time.time() - self.start_time
                print(f"Real time elapsed: {elapsed:.2f}s")
                print(f"Average policy frequency: {self.inference_count / elapsed:.1f}Hz (target: {self.policy_freq}Hz)")
                print(f"Average tick frequency: {self.tick_count / elapsed:.1f}Hz (target: {self.control_freq}Hz)")
            print("="*60)
    
    def process_control_step(self):
        """Process one control step (called at control_freq Hz)."""
        self.tick_count += 1
        
        # Increment simulation time
        self.sim_time += self.control_dt
        
        # Get observation
        obs = get_observation_with_high_state(
            self.latest_low_state, 
            self.latest_high_state, 
            self.latest_spacemouse_state,
            self.obs_buffer
        )
        self.obs = obs
        
        
        # # Run policy at 50Hz based on simulation time
        # keyboard.read_key() # an important inclusion thanks to @wkl
        if self.latest_spacemouse_state.button_1_pressed or self.latest_spacemouse_state.button_2_pressed:
            self.last_keyboard_press_time = time.perf_counter()
            
        if self.sim_time - self.last_policy_time >= self.policy_dt:
            # print(f"  Obs[0:3] (lin_vel): {self.obs[0:3]}")
            # print(f"  Obs[3:6] (ang_vel): {self.obs[3:6]}")
            # print(f"  Obs[6:9] (gravity_b): {self.obs[6:9]}")
            # print(f"  Obs[9:12] (cmd_vel): {self.obs[9:12]}")
            # print(f"  Obs[12] (cmd_height): {self.obs[12]}")
            # print(f"  Obs[13:25] (joint_pos): {self.obs[13:25]}")
            # print(f"  Obs[25:37] (joint_vel): {self.obs[25:37]}")
            # print(f"  Obs[37:49] (prev_actions): {self.obs[37:49]}")
            # Run policy inference
            self.set_commands(self.vel_x, self.vel_y, self.yaw, self.height)
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                actions_tensor = self.policy(obs_tensor)
                actions_policy_order = actions_tensor.squeeze(0).cpu().numpy()
            
            # Update current action
            self.current_action = actions_policy_order.copy()
            
            # Update action buffer (store in policy order)
            self.obs_buffer.update_actions(actions_policy_order)
            
            self.inference_count += 1
            self.last_policy_time = self.sim_time
        
        # Send motor commands every tick (using latest action)
        if self.sim_time <= 3 + self.last_keyboard_press_time:
            self.stand_control()
        if self.sim_time >= 3 + self.last_keyboard_press_time:
            self.send_motor_commands()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Go2 RL Policy Controller")
    parser.add_argument("--policy", type=str, default="policy.pt",
                       help="Path to policy.pt file")
    parser.add_argument("--domain_id", type=int, default=1,
                       help="DDS domain ID (default: 1 for simulation)")
    parser.add_argument("--interface", type=str, default="lo",
                       help="Network interface (default: lo for local)")
    parser.add_argument("--policy-freq", type=int, default=50,
                       help="Policy inference frequency in Hz (default: 50)")
    parser.add_argument("--control-freq", type=int, default=200,
                       help="Motor command frequency in Hz (default: 200, dt=0.005)")
    parser.add_argument("--kp", type=float, default=25.0,
                       help="Position gain/stiffness (default: 25.0 - lower for simulation stability)")
    parser.add_argument("--kd", type=float, default=0.5,
                       help="Velocity gain/damping (default: 0.5 - lower for simulation stability)")
    parser.add_argument("--action-scale", type=float, default=0.5,
                       help="Scale factor for policy actions (default: 0.5)")
    parser.add_argument("--vel-x", type=float, default=0.0,
                       help="Forward velocity command (m/s)")
    parser.add_argument("--vel-y", type=float, default=0.0,
                       help="Lateral velocity command (m/s)")
    parser.add_argument("--vel-yaw", type=float, default=0.0,
                       help="Yaw rate command (rad/s)")
    parser.add_argument("--height", type=float, default=0.3,
                       help="Height command (m)")
    
    args = parser.parse_args()
    
    # Initialize DDS communication
    print(f"Initializing DDS (domain_id={args.domain_id}, interface={args.interface})")
    rclpy.init(args=None)
    
    # Create controller
    node = Go2PolicyController(
        policy_path=args.policy,
        vel_x = args.vel_x,
        vel_y = args.vel_y,
        yaw = args.vel_yaw,
        height = args.height,
        policy_freq=args.policy_freq,
        control_freq=args.control_freq,
        kp=args.kp,
        kd=args.kd,
        action_scale=args.action_scale
    )
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Print statistics before destroying node (avoid rosout errors)
        print("\n" + "="*60)
        print("Shutting down...")
        # print(f"Total inferences: {node.inference_count}")
        # print(f"Total ticks: {node.tick_count}")
        # print(f"Simulation time: {node.time:.2f}s")
        print("="*60)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
