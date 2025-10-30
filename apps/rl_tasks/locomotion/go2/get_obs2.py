"""
Observation extraction for RL policy using ROS 2 messages.
Adapted from SDK version to work with unitree_go.msg types.
"""

from unitree_go.msg import LowState, IMUState, MotorState
import numpy as np
import yaml
import os
import torch


def load_joint_mapping(yaml_file_path):
    """
    Load joint mapping from YAML file for policy transfer.
    
    Args:
        yaml_file_path: Path to YAML file with source_joint_names and target_joint_names
        
    Returns:
        tuple: (target_to_source_indices, source_to_target_indices, source_joint_names, target_joint_names)
               - target_to_source_indices: Maps target (SDK) order to source (policy) order
               - source_to_target_indices: Maps source (policy) order to target (SDK) order
               - source_joint_names: List of joint names in policy order
               - target_joint_names: List of joint names in SDK order
    """
    try:
        with open(yaml_file_path) as file:
            config = yaml.safe_load(file)
    except Exception as e:
        raise RuntimeError(f"Failed to load joint mapping from {yaml_file_path}: {e}")
    
    source_joint_names = config["source_joint_names"]  # Policy order (Isaac Lab)
    target_joint_names = config["target_joint_names"]  # SDK order (Unitree)
    
    # Create target to source mapping (SDK -> Policy)
    # This maps from SDK joint order to policy joint order
    target_to_source = []
    for joint_name in target_joint_names:
        if joint_name in source_joint_names:
            target_to_source.append(source_joint_names.index(joint_name))
        else:
            raise ValueError(f"Joint '{joint_name}' not found in source joint names")
    
    # Create source to target mapping (Policy -> SDK)
    # This maps from policy joint order to SDK joint order
    source_to_target = []
    for joint_name in source_joint_names:
        if joint_name in target_joint_names:
            source_to_target.append(target_joint_names.index(joint_name))
        else:
            raise ValueError(f"Joint '{joint_name}' not found in target joint names")
    
    return target_to_source, source_to_target, source_joint_names, target_joint_names


def remap_joints_by_name(values_sdk, target_names, source_names, target_to_source):
    """
    Remap joint values from SDK order to Policy order using joint names.
    This ensures correct mapping even when joints are grouped differently.
    
    Args:
        values_sdk: Array of values in SDK/target order (12,)
        target_names: List of joint names in SDK order
        source_names: List of joint names in policy order
        target_to_source: Index mapping array
        
    Returns:
        values_policy: Array of values in policy order (12,)
    """
    values_policy = np.zeros(12)
    for policy_idx, policy_joint_name in enumerate(source_names):
        # Find where this joint appears in SDK order
        sdk_idx = target_names.index(policy_joint_name)
        values_policy[policy_idx] = values_sdk[sdk_idx]
    return values_policy


class ObservationBuffer:
    """
    Buffer to maintain state needed for observation construction.
    """
    def __init__(self, mapping_yaml_path=None):
        self.previous_actions = np.zeros(12)
        self.cmd_vel = np.array([1.0, 0.0, 0.0])  # [forward, lateral, yaw_rate]
        self.height_cmd = 0.3
        
        # Default position in MuJoCo actuator order (FR, FL, RR, RL)
        # Actuator order from MuJoCo: FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf, ...
        self.default_pos_sdk = np.array([
            0.0, 0.8, -1.5,  # FR: hip, thigh, calf (actuators 0-2)
            0.0, 0.8, -1.5,  # FL: hip, thigh, calf (actuators 3-5)
            0.0, 0.8, -1.5,  # RR: hip, thigh, calf (actuators 6-8)
            0.0, 0.8, -1.5   # RL: hip, thigh, calf (actuators 9-11)
        ])
        
        # Store base velocity estimate (from IMU integration or other source)
        # In ROS 2 version, we'll estimate this from IMU or use a velocity estimator
        self._base_velocity = np.zeros(3)
        
        # Joint mapping for policy transfer
        if mapping_yaml_path is None:
            # Use default path
            script_dir = os.path.dirname(os.path.abspath(__file__))
            mapping_yaml_path = os.path.join(script_dir, "physx_to_mujoco_go2.yaml")
        
        self.target_to_source, self.source_to_target, self.source_names, self.target_names = load_joint_mapping(mapping_yaml_path)
        
        # Pre-compute default_pos in policy order for observations
        self.default_pos_policy = remap_joints_by_name(
            self.default_pos_sdk, self.target_names, self.source_names, self.target_to_source
        )
        
        print(f"[INFO] Loaded joint mapping from: {mapping_yaml_path}")
        print(f"[INFO] Target to Source (SDK->Policy): {self.target_to_source}")
        print(f"[INFO] Source to Target (Policy->SDK): {self.source_to_target}")
        print(f"[INFO] Default pos SDK order: {self.default_pos_sdk}")
        print(f"[INFO] Default pos Policy order: {self.default_pos_policy}")

    
    @property
    def default_pos(self):
        """Return default_pos in SDK order (for motor commands)."""
        return self.default_pos_sdk
    
    def set_commands(self, cmd_vel=None, height_cmd=None):
        """Update command velocities and height."""
        if cmd_vel is not None:
            self.cmd_vel = np.array(cmd_vel)
        if height_cmd is not None:
            self.height_cmd = height_cmd
    
    def update_actions(self, actions):
        """
        Update the previous actions buffer.
        Actions should be in policy order and will be stored as-is.
        """
        self.previous_actions = np.array(actions)
    
    def remap_actions_to_sdk(self, actions_policy_order):
        """
        Convert actions from policy order to SDK order for motor commands.
        
        Args:
            actions_policy_order: Actions in policy order (12,)
            
        Returns:
            actions_sdk_order: Actions in SDK order (12,)
        """
        # Convert from policy order to SDK order using name-based mapping
        actions_sdk = np.zeros(12)
        for policy_idx, policy_joint_name in enumerate(self.source_names):
            sdk_idx = self.target_names.index(policy_joint_name)
            actions_sdk[sdk_idx] = actions_policy_order[policy_idx]
        return actions_sdk
    
    def update_base_velocity(self, velocity):
        """Update base velocity estimate."""
        self._base_velocity = np.array(velocity)


def get_observation(velocity: list, imu_state: IMUState, motor_states: list, obs_buffer: ObservationBuffer):
    """
    Extract observations from ROS 2 state messages for RL policy.
    
    Args:
        imu_state: IMUState message from LowState
        motor_states: List of MotorState messages (12 motors)
        obs_buffer: ObservationBuffer containing previous actions and commands
    
    Returns:
        obs: numpy array of shape (49,) with observation vector
        
    Observation structure (49 dimensions):
    - obs[0:3]   : Base linear velocity (estimated from IMU)
    - obs[3:6]   : Base angular velocity (from IMU gyroscope) 
    - obs[6:9]   : Gravity direction (from IMU quaternion)
    - obs[9:12]  : Command velocity (x, y, yaw)
    - obs[12]    : Height command
    - obs[13:25] : Joint positions relative to default (12 joints)
    - obs[25:37] : Joint velocities (12 joints)
    - obs[37:49] : Previous actions (12 values)
    """
    obs = np.zeros(49)
    
    obs_buffer.update_base_velocity(velocity)
    
    # Get joint positions and velocities in SDK order
    current_joint_pos_sdk = np.array([motor_states[i].q for i in range(12)])
    current_joint_vel_sdk = np.array([motor_states[i].dq for i in range(12)])
    
    # Convert SDK order to Policy order using name-based mapping
    current_joint_pos_policy = remap_joints_by_name(
        current_joint_pos_sdk, obs_buffer.target_names, obs_buffer.source_names, obs_buffer.target_to_source
    )
    current_joint_vel_policy = remap_joints_by_name(
        current_joint_vel_sdk, obs_buffer.target_names, obs_buffer.source_names, obs_buffer.target_to_source
    )
    default_pos_policy = obs_buffer.default_pos_policy

    # Fill joint positions (obs[13:25]) in policy order
    obs[13:25] = current_joint_pos_policy - default_pos_policy
    
    # Fill joint velocities (obs[25:37]) in policy order
    obs[25:37] = current_joint_vel_policy
    
    # Extract IMU quaternion from ROS 2 message
    # ROS 2 quaternion format: [w, x, y, z]
    quat = np.array([
        imu_state.quaternion[0],  # w
        imu_state.quaternion[1],  # x
        imu_state.quaternion[2],  # y
        imu_state.quaternion[3]   # z
    ])
    
    # Compute gravity direction in body frame from quaternion
    # Gravity in world frame is [0, 0, -1]
    # Rotate it to body frame using inverse quaternion rotation
    gravity_world = np.array([0.0, 0.0, -1.0])
    gravity_b = quat_rotate_inverse(quat, gravity_world)
    obs[6:9] = gravity_b
    
    # Base angular velocity (gyroscope) (obs[3:6])
    obs[3:6] = np.array([
        imu_state.gyroscope[0],
        imu_state.gyroscope[1],
        imu_state.gyroscope[2]
    ])
    
    # Base linear velocity (obs[0:3])
    # For ROS 2 version, use velocity estimate from buffer
    obs[0:3] = obs_buffer._base_velocity
    
    # Command velocity (obs[9:12])
    obs[9:12] = obs_buffer.cmd_vel
    
    # Height command (obs[12])
    obs[12] = obs_buffer.height_cmd
    
    # Previous actions (obs[37:49])
    obs[37:49] = obs_buffer.previous_actions
        
    return obs


def quat_rotate_inverse(q, v):
    """
    Rotate vector v by the inverse of quaternion q.
    q: quaternion [w, x, y, z]
    v: vector [x, y, z]
    Returns rotated vector
    """
    # Inverse rotation is equivalent to conjugate quaternion rotation
    q_w, q_x, q_y, q_z = q[0], q[1], q[2], q[3]
    
    # Conjugate quaternion (inverse for unit quaternions)
    q_conj = np.array([q_w, -q_x, -q_y, -q_z])
    
    # Quaternion-vector multiplication: q_conj * [0, v] * q
    # Simplified formula for rotating vector by quaternion
    t = 2.0 * np.cross(q_conj[1:], v)
    return v + q_conj[0] * t + np.cross(q_conj[1:], t)
