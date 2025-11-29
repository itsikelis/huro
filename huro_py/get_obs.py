#!/usr/bin/env python3

from unitree_go.msg import LowState, SportModeState
from huro.msg import SpaceMouseState

import numpy as np
import yaml
import os
from huro_py.mapping import Mapper
from huro_py.utils import rotate



def get_obs_low_state(lowstate_msg: LowState, spacemouse_msg: SpaceMouseState, height: float, prev_actions: np.array, phase: float, mapper: Mapper):
    """
    Extract observations from LowState message for RL policy.
    
    Args:
        msg: LowState message from robot
        obs_buffer: ObservationBuffer containing previous actions and commands
    
    Returns:
        obs: numpy array of shape (49,) with observation vector
        
    Observation structure (49 dimensions):
    - obs[0:3]   : Base angular velocity (from IMU) 
    - obs[3:7]   : Gravity direction (from IMU)
    - obs[7:10]  : Command velocity (x, y, yaw)
    - obs[10]    : Height command
    - obs[11:23] : Joint positions relative to default (12 joints)
    - obs[23:35] : Joint velocities (12 joints)
    - obs[35:47] : Previous actions (12 values)
    """
    
    
    # MAPPING ROBOT -> POLICY
    
    motor_states = lowstate_msg.motor_state[:12]
    
    current_joint_pos_sdk = np.array([motor_states[i].q for i in range(12)])
    current_joint_vel_sdk = np.array([motor_states[i].dq for i in range(12)])
    
    current_joint_pos_policy = mapper.remap_joints_by_name(
        current_joint_pos_sdk, mapper.target_names, mapper.source_names, mapper.target_to_source
    )
    current_joint_vel_policy = mapper.remap_joints_by_name(
        current_joint_vel_sdk, mapper.target_names, mapper.source_names, mapper.target_to_source
    )
    default_pos_policy = mapper.default_pos_policy

    # FILLING OBS VECTOR


    obs = np.zeros(48)
        
    # Base linear velocity (obs[0:3])
    
    # Base angular velocity (gyroscope) (obs[0:3])
    obs[0:3] = np.array([
        lowstate_msg.imu_state.gyroscope[0],
        lowstate_msg.imu_state.gyroscope[1],
        lowstate_msg.imu_state.gyroscope[2]
    ])
    # Computing projected gravity from IMU sensor
    quat = np.array([
        lowstate_msg.imu_state.quaternion[0],  # w
        lowstate_msg.imu_state.quaternion[1],  # x
        lowstate_msg.imu_state.quaternion[2],  # y
        lowstate_msg.imu_state.quaternion[3]   # z
    ])
    # Normalize quaternion to prevent drift
  
    
    gravity_world = np.array([0.0, 0.0, -1.0])

    gravity_b = rotate(quat,gravity_world)
    # gravity_b[0] *= 2.0
    # gravity_b[1] *= 2.0
    print(gravity_b)
    obs[3:6] = gravity_b
    # Command velocity (obs[9:12]) - default to zero (forward, lateral, yaw rate)
    obs[6:9] = [spacemouse_msg.twist.angular.y / 2, -spacemouse_msg.twist.angular.x / 2, spacemouse_msg.twist.angular.z / 2]
    # Height command (obs[12]) - default standing height
    obs[9] = height
    # Fill joint positions (obs[13:25]) in policy order
    obs[10:22] = current_joint_pos_policy - default_pos_policy
    # Fill joint velocities (obs[25:37]) in policy order
    obs[22:34] = current_joint_vel_policy
    # Previous actions (obs[37:49]) - default to zero
    obs[34:46] = prev_actions
    obs[46] = np.sin(2.0 * np.pi * phase)
    obs[47] = np.cos(2.0 * np.pi * phase)
        
    return obs

def get_obs_high_state(lowstate_msg: LowState, highstate_msg: SportModeState, spacemouse_msg: SpaceMouseState, height: float, prev_actions: np.array, mapper: Mapper, previous_vel = None):
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
    
    
    # MAPPING ROBOT -> POLICY
    
    motor_states = lowstate_msg.motor_state[:12]
    
    current_joint_pos_sdk = np.array([motor_states[i].q for i in range(12)])
    current_joint_vel_sdk = np.array([motor_states[i].dq for i in range(12)])
    
    current_joint_pos_policy = mapper.remap_joints_by_name(
        current_joint_pos_sdk, mapper.target_names, mapper.source_names, mapper.target_to_source
    )
    current_joint_vel_policy = mapper.remap_joints_by_name(
        current_joint_vel_sdk, mapper.target_names, mapper.source_names, mapper.target_to_source
    )
    default_pos_policy = mapper.default_pos_policy


    obs = np.zeros(49)
        
    # Base linear velocity (obs[0:3])
    obs[0:3] = highstate_msg.velocity[:3]
    # obs[0:3] = compute_base_lin_vel(lowstate_msg=lowstate_msg, prev_vel=previous_vel)
    
    # Base angular velocity (gyroscope) (obs[3:6])
    obs[3:6] = np.array([
        lowstate_msg.imu_state.gyroscope[0],
        lowstate_msg.imu_state.gyroscope[1],
        lowstate_msg.imu_state.gyroscope[2]
    ])
    
    # Computing projected gravity from IMU sensor
    quat = np.array([
        lowstate_msg.imu_state.quaternion[0],  # w
        lowstate_msg.imu_state.quaternion[1],  # x
        lowstate_msg.imu_state.quaternion[2],  # y
        lowstate_msg.imu_state.quaternion[3]   # z
    ])
    # Normalize quaternion to prevent drift    
    gravity_world = np.array([0.0, 0.0, -0.91])
    gravity_b = rotate(quat, gravity_world)
    obs[6:9] = np.array([0.0, 0.0, 0.0])
    # Command velocity (obs[9:12]) - default to zero (forward, lateral, yaw rate)
    obs[9:12] = [spacemouse_msg.twist.angular.y / 2, -spacemouse_msg.twist.angular.x / 2, spacemouse_msg.twist.angular.z / 2]
    # Height command (obs[12]) - default standing height
    obs[12] = height
    # Fill joint positions (obs[13:25]) in policy order
    obs[13:25] = current_joint_pos_policy - default_pos_policy
    # Fill joint velocities (obs[25:37]) in policy order
    obs[25:37] = current_joint_vel_policy
    # Previous actions (obs[37:49]) - default to zero
    obs[37:49] = prev_actions
        
    return obs


    

