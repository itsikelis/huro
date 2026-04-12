import numpy as np
import sys
import torch
import yaml
from types import SimpleNamespace
from sensor_msgs.msg import PointCloud2
from unitree_go.msg import LowState

import os
# from unitree_go.msg import PointCloud2

def quat_rotate_inverse(q, v):
    q_conj_xyz = -q[1:4]
    q_conj_w = q[0]
    t = 2.0 * torch.cross(q_conj_xyz, v, dim=0)
    rotated = v + q_conj_w * t + torch.cross(q_conj_xyz, t, dim=0)
    return rotated






class MockController:
    """Minimal controller-like object used by get_obs in headless runs."""

    def __init__(self, vel=None, height_axis=0.0):
        if vel is None:
            vel = [0.0, 0.0, 0.0]

        vx, vy, wz = float(vel[0]), float(vel[1]), float(vel[2])

        # Match get_obs_low_state convention:
        # forward = axes[1], lateral = axes[0], yaw = axes[2], height = axes[3]
        self.axes = [vy, vx, wz, float(height_axis)]

        # Keep emergency buttons always released.
        self.buttons = [0] * 10
        self.buttons[0] = 1
        self.buttons[1] = 1


class MockCmdVel:
    """Minimal cmd_vel-like object with vx, vy, wz at initialization."""

    def __init__(self, vx=0.0, vy=0.0, wz=0.0):
        vx = 0.0 if vx is None else vx
        vy = 0.0 if vy is None else vy
        wz = 0.0 if wz is None else wz
        self.vx = float(vx)
        self.vy = float(vy)
        self.wz = float(wz)

        # cmd_vel style fields
        self.linear = SimpleNamespace(x=self.vx, y=self.vy, z=0.0)
        self.angular = SimpleNamespace(x=0.0, y=0.0, z=self.wz)

        # Optional twist wrapper for APIs that expect msg.twist.linear / msg.twist.angular
        self.twist = SimpleNamespace(linear=self.linear, angular=self.angular)


class Mapper:
    """
    Buffer to maintain state needed for observation construction.
    """
    def __init__(self, mapping_yaml_path, default_pos_sdk):
        
        self.default_pos_sdk = default_pos_sdk
        
        
        self.target_to_source, self.source_to_target, self.source_names, self.target_names = self.load_joint_mapping(mapping_yaml_path)
        
        # Pre-compute default_pos in policy order for observations
        self.default_pos_policy = self.remap_joints_by_name(
            self.default_pos_sdk, self.target_names, self.source_names, self.target_to_source
        )
        
        print(f"[INFO] Loaded joint mapping from: {mapping_yaml_path}")
        print(f"[INFO] Source to Target (Policy->SDK): {self.source_to_target}")

    def load_joint_mapping(self, yaml_file_path):
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


    def remap_joints_by_name(self, values_sdk, target_names, source_names, target_to_source):
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
    
    
    def actions_policy_to_sdk(self, actions_policy_order):
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
        
    
#!/usr/bin/env python3
"""
Utility functions for processing LiDAR data and creating height maps.
"""


# np.set_printoptions(precision=2, threshold=sys.maxsize, linewidth=np.inf, edgeitems=100, suppress=True)


LIDAR_PITCH_DEG = -15.1
LIDAR_PITCH_RAD = np.deg2rad(LIDAR_PITCH_DEG)
COS_PITCH = np.cos(LIDAR_PITCH_RAD)
SIN_PITCH = np.sin(LIDAR_PITCH_RAD)
# replace with https://github.com/anybotics/grid_map
LIDAR_PITCH_DEG = torch.tensor(-15.09)
LIDAR_PITCH_RAD = torch.deg2rad(LIDAR_PITCH_DEG)
COS_PITCH_LIDAR = torch.cos(LIDAR_PITCH_RAD).item()
SIN_PITCH_LIDAR = torch.sin(LIDAR_PITCH_RAD).item()


def process_height_map(height_map: torch.tensor, lidar_msg: PointCloud2, lowstate_msg: LowState, x_range: list, y_range: list, res, delete_count: int = 100, min_x = 0, max_x = 0, min_z = 0, max_z = 0):
    if lidar_msg is None or lowstate_msg is None:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    qw = float(lowstate_msg.imu_state.quaternion[0])
    qx = float(lowstate_msg.imu_state.quaternion[1])
    qy = float(lowstate_msg.imu_state.quaternion[2])
    qz = float(lowstate_msg.imu_state.quaternion[3])

    grid_size_x = int(height_map.shape[1])
    grid_size_y = int(height_map.shape[2])

    hm_height = height_map[0].numpy()
    hm_age = height_map[1].numpy()

    old_cells = hm_age > delete_count
    hm_height[old_cells] = 0.0
    hm_age[old_cells] = 0.0
    hm_age += 1.0

    num_points = int(lidar_msg.width) * int(lidar_msg.height)
    point_step = int(lidar_msg.point_step)
    if num_points == 0 or point_step < 12:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    raw = np.frombuffer(lidar_msg.data, dtype=np.uint8)
    needed = num_points * point_step
    if raw.size < needed:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    xyz = np.ndarray(
        shape=(num_points, 3),
        dtype=np.float32,
        buffer=raw,
        offset=0,
        strides=(point_step, 4),
    )

    finite_xyz = np.isfinite(xyz).all(axis=1)
    if not np.any(finite_xyz):
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    xyz = xyz[finite_xyz]

    x = -xyz[:, 0]
    y = -xyz[:, 1]
    z = xyz[:, 2]
    
    # --- Extract roll and pitch from quaternion ---
    # Roll (rotation around X)
    sin_roll = 2.0 * (qw * qx + qy * qz)
    cos_roll = 1.0 - 2.0 * (qx * qx + qy * qy)

    # Pitch (rotation around Y)  
    sin_pitch = 2.0 * (qw * qy - qz * qx)
    cos_pitch = np.sqrt(max(1.0 - sin_pitch * sin_pitch, 0.0))
    pitch_deg = np.degrees(np.arcsin(np.clip(sin_pitch, -1, 1)))
    roll_deg  = np.degrees(np.arctan2(sin_roll, cos_roll))
    print(f"roll={roll_deg:.2f}°  pitch={pitch_deg:.2f}°")

    # Yaw intentionally ignored — we only want gravity alignment

    x1 = x * COS_PITCH_LIDAR - z * SIN_PITCH_LIDAR
    y1 = y
    z1 = x * SIN_PITCH_LIDAR + z * COS_PITCH_LIDAR

    # Compensate robot pitch (sign flipped)
    x2 =  cos_pitch * x1 - sin_pitch * z1
    y2 =  y1
    z2 =  sin_pitch * x1 + cos_pitch * z1

    # Compensate robot roll
    x_body = x2
    y_body =  cos_roll * y2 - sin_roll * z2
    z_body =  sin_roll * y2 + cos_roll * z2


    finite_mask = np.isfinite(x_body) & np.isfinite(y_body) & np.isfinite(z_body)
    if not np.any(finite_mask):
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    x_valid = x_body[finite_mask]
    y_valid = y_body[finite_mask]
    z_valid = z_body[finite_mask]

    grid_x_base = ((x_valid - x_range[1]) / res).astype(np.int32)
    grid_x = (grid_size_x - 1) - grid_x_base
    grid_y = ((y_valid - y_range[0]) / res).astype(np.int32)

    in_grid = (
        (grid_x >= 0)
        & (grid_x < grid_size_x)
        & (grid_y >= 0)
        & (grid_y < grid_size_y)
    )

    if np.any(in_grid):
        flat_idx = grid_x[in_grid] * grid_size_y + grid_y[in_grid]
        z_grid = z_valid[in_grid]
        max_heightmap = np.full(grid_size_x * grid_size_y, -np.inf, dtype=np.float32)
        np.maximum.at(max_heightmap, flat_idx, z_grid)

        max_heightmap = max_heightmap.reshape(grid_size_x, grid_size_y)
        valid_cells = np.isfinite(max_heightmap)
        # lidar_offset = -0.046825
        lidar_offset = -0.0
        hm_height[valid_cells] = max_heightmap[valid_cells] - 0.28 + lidar_offset
        hm_age[valid_cells] = 0.0


def process_height_map2(height_map: torch.tensor, lidar_msg: PointCloud2, lowstate_msg: LowState, x_range: list, y_range: list, res, delete_count: int = 100, min_x = 0, max_x = 0, min_z = 0, max_z = 0):
    if lidar_msg is None or lowstate_msg is None:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    qw = float(lowstate_msg.imu_state.quaternion[0])
    qx = float(lowstate_msg.imu_state.quaternion[1])
    qy = float(lowstate_msg.imu_state.quaternion[2])
    qz = float(lowstate_msg.imu_state.quaternion[3])

    grid_size_x = int(height_map.shape[1])
    grid_size_y = int(height_map.shape[2])

    hm_height = height_map[0].numpy()
    hm_age = height_map[1].numpy()

    old_cells = hm_age > delete_count
    hm_height[old_cells] = 0.0
    hm_age[old_cells] = 0.0
    hm_age += 1.0

    num_points = int(lidar_msg.width) * int(lidar_msg.height)
    point_step = int(lidar_msg.point_step)
    if num_points == 0 or point_step < 12:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    raw = np.frombuffer(lidar_msg.data, dtype=np.uint8)
    needed = num_points * point_step
    if raw.size < needed:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    xyz = np.ndarray(
        shape=(num_points, 3),
        dtype=np.float32,
        buffer=raw,
        offset=0,
        strides=(point_step, 4),
    )

    finite_xyz = np.isfinite(xyz).all(axis=1)
    if not np.any(finite_xyz):
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    xyz = xyz[finite_xyz]

    x = -xyz[:, 0]
    y = -xyz[:, 1]
    z = xyz[:, 2]
    
    x1 = x * COS_PITCH_LIDAR - z * SIN_PITCH_LIDAR
    y1 = y
    z1 = x * SIN_PITCH_LIDAR + z * COS_PITCH_LIDAR



    finite_mask = np.isfinite(x1) & np.isfinite(y1) & np.isfinite(z1)
    if not np.any(finite_mask):
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    x_valid = x1[finite_mask]
    y_valid = y1[finite_mask]
    z_valid = z1[finite_mask]

    grid_x_base = ((x_valid - x_range[1]) / res).astype(np.int32)
    grid_x = (grid_size_x - 1) - grid_x_base
    grid_y = ((y_valid - y_range[0]) / res).astype(np.int32)

    in_grid = (
        (grid_x >= 0)
        & (grid_x < grid_size_x)
        & (grid_y >= 0)
        & (grid_y < grid_size_y)
    )

    if np.any(in_grid):
        flat_idx = grid_x[in_grid] * grid_size_y + grid_y[in_grid]
        z_grid = z_valid[in_grid]
        max_heightmap = np.full(grid_size_x * grid_size_y, -np.inf, dtype=np.float32)
        np.maximum.at(max_heightmap, flat_idx, z_grid)

        max_heightmap = max_heightmap.reshape(grid_size_x, grid_size_y)
        valid_cells = np.isfinite(max_heightmap)
        # lidar_offset = -0.046825
        lidar_offset = -0.0
        hm_height[valid_cells] = max_heightmap[valid_cells] - 0.28 + lidar_offset
        hm_age[valid_cells] = 0.0



# z_world = sin_roll * y_pitch + cos_roll * z_pitch