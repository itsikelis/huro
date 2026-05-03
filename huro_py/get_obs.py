#!/usr/bin/env python3

from unitree_go.msg import LowState, SportModeState
from huro.msg import SpaceMouseState
from sensor_msgs.msg import PointCloud2


import torch

from huro_py.utils import Mapper, quat_rotate_inverse


def get_obs(
    lowstate_msg: LowState,
    vel,
    height: float,
    prev_actions: torch.tensor,
    mapper: Mapper,
):
    """
    Extract observations from LowState message for RL policy.

    Args:
        msg: LowState message from robot
        obs_buffer: ObservationBuffer containing previous actions and commands

    Returns:
        obs: numpy array of shape (49,) with observation vector

    Observation structure (49 dimensions):
    - obs[0:3]   : Base angular velocity (from IMU)
    - obs[3:6]   : Gravity direction (from IMU)
    - obs[6:9]  : Command velocity (x, y, yaw)
    - obs[9]    : Height command
    - obs[10:22] : Joint positions relative to default (12 joints)
    - obs[22:34] : Joint velocities (12 joints)
    - obs[34:46] : Previous actions (12 values)
    - obs[46:50] : Foot contacts
    """

    # MAPPING ROBOT -> POLICY

    motor_states = lowstate_msg.motor_state[:12]

    current_joint_pos_sdk = torch.tensor([motor_states[i].q for i in range(12)])
    current_joint_vel_sdk = torch.tensor([motor_states[i].dq for i in range(12)])

    current_joint_pos_policy = mapper.remap_joints_by_name(
        current_joint_pos_sdk,
        mapper.target_names,
        mapper.source_names,
        mapper.target_to_source,
    )
    current_joint_vel_policy = mapper.remap_joints_by_name(
        current_joint_vel_sdk,
        mapper.target_names,
        mapper.source_names,
        mapper.target_to_source,
    )
    default_pos_policy = mapper.default_pos_policy

    # FILLING OBS VECTOR
    obs = torch.zeros(50)
    
    # Base linear velocity (obs[0:3])

    # Base angular velocity (gyroscope) (obs[0:3])
    obs[0:3] = torch.tensor(
        [
            lowstate_msg.imu_state.gyroscope[0],
            lowstate_msg.imu_state.gyroscope[1],
            lowstate_msg.imu_state.gyroscope[2],
        ]
    )
    # Computing projected gravity from IMU sensor
    quat = torch.tensor(
        [
            lowstate_msg.imu_state.quaternion[0],  # w
            lowstate_msg.imu_state.quaternion[1],  # x
            lowstate_msg.imu_state.quaternion[2],  # y
            lowstate_msg.imu_state.quaternion[3],  # z
        ]
    )

    gravity_world = torch.tensor([0.0, 0.0, -1.0])

    gravity_b = quat_rotate_inverse(quat, gravity_world)

    obs[3:6] = gravity_b

    obs[6:9] = torch.tensor([
        vel[0],  # forward velocity
        vel[1],  # lateral velocity (flip for correct direction)
        vel[2],  # yaw rate
    ])
    
    obs[9] = 0.3 # height
    
    # Fill joint positions (obs[13:25]) in policy order
    obs[10:22] = torch.tensor(current_joint_pos_policy - default_pos_policy)
    # Fill joint velocities (obs[25:37]) in policy order
    obs[22:34] = torch.tensor(current_joint_vel_policy)
    # Previous actions (obs[37:49]) - default to zero
    obs[34:46] = prev_actions
    # see the best threshold for real robot
    obs[46:50] = torch.tensor([
        float(lowstate_msg.foot_force[0]>10),
        float(lowstate_msg.foot_force[1]>10),
        float(lowstate_msg.foot_force[2]>10),
        float(lowstate_msg.foot_force[3]>10)
    ])

    return obs


def get_obs_lidar(
    lowstate_msg: LowState,
    height_map: torch.tensor,
    vel,
    height: float,
    prev_actions: torch.tensor,
    mapper: Mapper,
):
    """
    Extract observations from LowState message for RL policy.

    Args:
        msg: LowState message from robot
        obs_buffer: ObservationBuffer containing previous actions and commands

    Returns:
        obs: numpy array of shape (49,) with observation vector

    Observation structure (49 dimensions):
    - obs[0:3]   : Base angular velocity (from IMU)
    - obs[3:6]   : Gravity direction (from IMU)
    - obs[6:9]  : Command velocity (x, y, yaw)
    - obs[9]    : Height command
    - obs[10:22] : Joint positions relative to default (12 joints)
    - obs[22:34] : Joint velocities (12 joints)
    - obs[34:46] : Previous actions (12 values)
    - obs[46:50] : Foot contacts
    """

    # MAPPING ROBOT -> POLICY

    motor_states = lowstate_msg.motor_state[:12]

    current_joint_pos_sdk = torch.tensor([motor_states[i].q for i in range(12)])
    current_joint_vel_sdk = torch.tensor([motor_states[i].dq for i in range(12)])

    current_joint_pos_policy = mapper.remap_joints_by_name(
        current_joint_pos_sdk,
        mapper.target_names,
        mapper.source_names,
        mapper.target_to_source,
    )
    current_joint_vel_policy = mapper.remap_joints_by_name(
        current_joint_vel_sdk,
        mapper.target_names,
        mapper.source_names,
        mapper.target_to_source,
    )
    default_pos_policy = mapper.default_pos_policy

    # FILLING OBS VECTOR
    obs = torch.zeros(195)
    
    # Base linear velocity (obs[0:3])

    # Base angular velocity (gyroscope) (obs[0:3])
    obs[0:3] = torch.tensor(
        [
            lowstate_msg.imu_state.gyroscope[0],
            lowstate_msg.imu_state.gyroscope[1],
            lowstate_msg.imu_state.gyroscope[2],
        ]
    )
    # Computing projected gravity from IMU sensor
    quat = torch.tensor(
        [
            lowstate_msg.imu_state.quaternion[0],  # w
            lowstate_msg.imu_state.quaternion[1],  # x
            lowstate_msg.imu_state.quaternion[2],  # y
            lowstate_msg.imu_state.quaternion[3],  # z
        ]
    )
    

    gravity_world = torch.tensor([0.0, 0.0, -1.0])

    gravity_b = quat_rotate_inverse(quat, gravity_world)

    obs[3:6] = gravity_b

    # print(cmd_vel_msg)
    obs[6:9] = torch.tensor([
        vel[0],  # forward velocity
        vel[1],  # lateral velocity (flip for correct direction)
        vel[2],  # yaw rate
    ])

    # Fill joint positions (obs[13:25]) in policy order
    obs[9:21] = torch.tensor(current_joint_pos_policy - default_pos_policy)
    # Fill joint velocities (obs[25:37]) in policy order
    obs[21:33] = torch.tensor(current_joint_vel_policy)
    # height_data
    idx = 33+150
    height_map_copy = height_map[0, :, :].clone().flip(0, 1).reshape(150)
    obs[33:idx] = height_map_copy

    obs[idx:idx + 12] = prev_actions
    # Previous actions (obs[37:49]) - default to zero
    
    # CHANGE FOR DEBUG !! 

    return obs

def get_obs_lidar_cnn(
    lowstate_msg: LowState,
    height_map: torch.tensor,
    vel,
    height: float,
    prev_actions: torch.tensor,
    mapper: Mapper,
):
    """
    Extract observations from LowState message for RL policy.

    Args:
        msg: LowState message from robot
        obs_buffer: ObservationBuffer containing previous actions and commands

    Returns:
        obs: numpy array of shape (49,) with observation vector

    Observation structure (49 dimensions):
    - obs[0:3]   : Base angular velocity (from IMU)
    - obs[3:6]   : Gravity direction (from IMU)
    - obs[6:9]  : Command velocity (x, y, yaw)
    - obs[9]    : Height command
    - obs[10:22] : Joint positions relative to default (12 joints)
    - obs[22:34] : Joint velocities (12 joints)
    - obs[34:46] : Previous actions (12 values)
    - obs[46:50] : Foot contacts
    """

    # MAPPING ROBOT -> POLICY

    motor_states = lowstate_msg.motor_state[:12]

    current_joint_pos_sdk = torch.tensor([motor_states[i].q for i in range(12)])
    current_joint_vel_sdk = torch.tensor([motor_states[i].dq for i in range(12)])

    current_joint_pos_policy = mapper.remap_joints_by_name(
        current_joint_pos_sdk,
        mapper.target_names,
        mapper.source_names,
        mapper.target_to_source,
    )
    current_joint_vel_policy = mapper.remap_joints_by_name(
        current_joint_vel_sdk,
        mapper.target_names,
        mapper.source_names,
        mapper.target_to_source,
    )
    default_pos_policy = mapper.default_pos_policy

    # FILLING OBS VECTOR
    obs = torch.zeros(45)
    
    # Base linear velocity (obs[0:3])

    # Base angular velocity (gyroscope) (obs[0:3])
    obs[0:3] = torch.tensor(
        [
            lowstate_msg.imu_state.gyroscope[0],
            lowstate_msg.imu_state.gyroscope[1],
            lowstate_msg.imu_state.gyroscope[2],
        ]
    )
    # Computing projected gravity from IMU sensor
    quat = torch.tensor(
        [
            lowstate_msg.imu_state.quaternion[0],  # w
            lowstate_msg.imu_state.quaternion[1],  # x
            lowstate_msg.imu_state.quaternion[2],  # y
            lowstate_msg.imu_state.quaternion[3],  # z
        ]
    )
    

    gravity_world = torch.tensor([0.0, 0.0, -1.0])

    gravity_b = quat_rotate_inverse(quat, gravity_world)

    obs[3:6] = gravity_b

    # print(cmd_vel_msg)
    obs[6:9] = torch.tensor([
        vel[0],  # forward velocity
        vel[1],  # lateral velocity (flip for correct direction)
        vel[2],  # yaw rate
    ])

    # Fill joint positions (obs[13:25]) in policy order
    obs[9:21] = torch.tensor(current_joint_pos_policy - default_pos_policy)
    # Fill joint velocities (obs[25:37]) in policy order
    obs[21:33] = torch.tensor(current_joint_vel_policy)
    obs[33:45] = prev_actions
    # height_data
    height_map_copy = [height_map[0, :, :].clone().reshape(1,1,15,10) - 0.5]
    
    # print(height_map[0, :, :].clone().flip(0, 1).reshape(1,1,15,10))

    return obs, height_map_copy