from unitree_go.msg import LowState, SportModeState
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
        self.cmd_vel = np.array([-0.5, 0.0, 0.0])  # [forward, lateral, yaw_rate]
        self.height_cmd = 0.3
        
        # Default position in MuJoCo actuator order (FR, FL, RR, RL)
        # Actuator order from MuJoCo: FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf, ...
        self.default_pos_sdk = np.array([
            0.0, 0.8, -1.5,  # FR: hip, thigh, calf (actuators 0-2)
            0.0, 0.8, -1.5,  # FL: hip, thigh, calf (actuators 3-5)
            0.0, 0.8, -1.5,  # RR: hip, thigh, calf (actuators 6-8)
            0.0, 0.8, -1.5   # RL: hip, thigh, calf (actuators 9-11)
        ])
        
        # Store latest high state velocity (from SportModeState)
        self._high_state_velocity = np.zeros(3)
        
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

    
    def update_high_state_velocity(self, velocity):
        """Update velocity from SportModeState (high state) message."""
        self._high_state_velocity = np.array(velocity)


def get_observation(msg: LowState, obs_buffer: ObservationBuffer):
    """
    Extract observations from LowState message for RL policy.
    
    Args:
        msg: LowState message from robot
        obs_buffer: ObservationBuffer containing previous actions and commands
    
    Returns:
        obs: numpy array of shape (49,) with observation vector
        
    Observation structure (49 dimensions):
    - obs[0:3]   : Base linear velocity (from IMU)
    - obs[3:6]   : Base angular velocity (from IMU) 
    - obs[6:9]   : Gravity direction (from IMU)
    - obs[9:12]  : Command velocity (x, y, yaw)
    - obs[12]    : Height command
    - obs[13:25] : Joint positions relative to default (12 joints)
    - obs[25:37] : Joint velocities (12 joints)
    - obs[37:49] : Previous actions (12 values)
    """
    obs = np.zeros(49)
    
    # Extract motor states (Go2 has 12 motors: indices 0-11)
    # NOTE: motor_states are in SDK/target order (FL, FR, RL, RR grouping)
    motor_states = msg.motor_state[:12]
    
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
    
    # Extract IMU quaternion (w, x, y, z format)
    quat = np.array([
        msg.imu_state.quaternion[0],  # w
        msg.imu_state.quaternion[1],  # x
        msg.imu_state.quaternion[2],  # y
        msg.imu_state.quaternion[3]   # z
    ])
    
    # Compute gravity direction in body frame from quaternion
    # Gravity in world frame is [0, 0, -1]
    # Rotate it to body frame using inverse quaternion rotation
    gravity_world = np.array([0.0, 0.0, -1.0])
    gravity_b = quat_rotate_inverse(quat, gravity_world)
    obs[6:9] = gravity_b
    
    # Base angular velocity (gyroscope) (obs[3:6])
    obs[3:6] = np.array([
        msg.imu_state.gyroscope[0],
        msg.imu_state.gyroscope[1],
        msg.imu_state.gyroscope[2]
    ])
    
    # Base linear velocity (obs[0:3])
    obs[0:3] = obs_buffer._high_state_velocity
    
    # Command velocity (obs[9:12]) - default to zero (forward, lateral, yaw rate)
    obs[9:12] = obs_buffer.cmd_vel
    
    # Height command (obs[12]) - default standing height
    obs[12] = obs_buffer.height_cmd
    
    # Previous actions (obs[37:49]) - default to zero
    obs[37:49] = obs_buffer.previous_actions
        
    return obs


def get_observation_with_high_state(low_state_msg: LowState, high_state_msg: SportModeState, obs_buffer: ObservationBuffer):
    """
    Extract observations using both LowState and SportModeState (RECOMMENDED).
    This provides more accurate velocity measurements from SportModeState.
    
    Args:
        low_state_msg: LowState message from robot (motor states, IMU)
        high_state_msg: SportModeState message (velocity, position)
        obs_buffer: ObservationBuffer containing previous actions and commands
    
    Returns:
        obs: numpy array of shape (49,) with observation vector
    """
    # Update velocity from high state
    obs_buffer.update_high_state_velocity(high_state_msg.velocity[:3])
    
    # Get observation using low state (which will use the updated velocity)
    return get_observation(low_state_msg, obs_buffer)


def LowStateHandler(msg: LowState):
    """
    Simple handler function for compatibility.
    For full functionality, use get_observation() with ObservationBuffer.
    """
    obs_buffer = ObservationBuffer()
    return get_observation(msg, obs_buffer)


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
    
