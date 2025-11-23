#!/usr/bin/env python3

from unitree_go.msg import LowState, SportModeState
from huro.msg import SpaceMouseState

import numpy as np
import yaml
import os
import torch
from huro_py.mapping import Mapper


def get_obs_low_state(lowstate_msg: LowState, spacemouse_msg: SpaceMouseState, height: float, prev_actions: np.array, mapper: Mapper, raw = True):
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
    
    if raw:
        obs = np.zeros(47)
    else:
        obs = np.zeros(46)
        
    # Base linear velocity (obs[0:3])
    
    # Base angular velocity (gyroscope) (obs[3:6])
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
    if raw:
        obs[3:7] = quat
        # Command velocity (obs[9:12]) - default to zero (forward, lateral, yaw rate)
        obs[7:10] = [spacemouse_msg.twist.angular.y / 2, -spacemouse_msg.twist.angular.x / 2, spacemouse_msg.twist.angular.z / 2]
        # Height command (obs[12]) - default standing height
        obs[10] = height
        # Fill joint positions (obs[13:25]) in policy order
        obs[11:23] = current_joint_pos_policy - default_pos_policy
        # Fill joint velocities (obs[25:37]) in policy order
        obs[23:35] = current_joint_vel_policy
        # Previous actions (obs[37:49]) - default to zero
        obs[35:47] = prev_actions
    else:
        gravity_world = np.array([0.0, 0.0, -1.0])
        gravity_b = quat_rotate_inverse(quat, gravity_world)
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
        
    return obs

def get_obs_high_state(lowstate_msg: LowState, highstate_msg: SportModeState, spacemouse_msg: SpaceMouseState, height: float, prev_actions: np.array, mapper: Mapper, raw = True):
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

    # FILLING OBS VECTOR
    if raw:
        obs = np.zeros(50)
    else:
        obs = np.zeros(49)
        
    # Base linear velocity (obs[0:3])
    obs[0:3] = highstate_msg.velocity[:3]
    
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
    if raw:
        obs[6:10] = quat
        # Command velocity (obs[9:12]) - default to zero (forward, lateral, yaw rate)
        obs[10:13] = [spacemouse_msg.twist.angular.y / 2, -spacemouse_msg.twist.angular.x / 2, spacemouse_msg.twist.angular.z / 2]
        # Height command (obs[12]) - default standing height
        obs[13] = height
        # Fill joint positions (obs[13:25]) in policy order
        obs[14:26] = current_joint_pos_policy - default_pos_policy
        # Fill joint velocities (obs[25:37]) in policy order
        obs[26:38] = current_joint_vel_policy
        # Previous actions (obs[37:49]) - default to zero
        obs[38:50] = prev_actions
    else:
        gravity_world = np.array([0.0, 0.0, -1.0])
        gravity_b = quat_rotate_inverse(quat, gravity_world)
        obs[6:9] = gravity_b
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




def quat_rotate_inverse(q, v):
    """
    Rotate vector v by the inverse of quaternion q.
    q: quaternion [w, x, y, z]
    v: vector [x, y, z]
    Returns rotated vector
    """
    xyz = q[1:]
    t = np.cross(v, xyz) * 2
    return v - q[0] * t + np.cross(t, xyz)
    
