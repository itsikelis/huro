import numpy as np

def quat_rotate_inverse(q, v):
    q_w, q_x, q_y, q_z = q[0], q[1], q[2], q[3]
    q_conj = np.array([q_w, -q_x, -q_y, -q_z])
    t = 2.0 * np.cross(q_conj[1:], v)  # ✅ Correct: q_conj[1:] = [x, y, z]
    return v + q_conj[0] * t + np.cross(q_conj[1:], t)  # ✅ Complete formula



import numpy as np
import yaml
import os



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

import numpy as np
from unitree_sdk2_python.unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_
import struct
import sys
np.set_printoptions(precision=2, threshold=sys.maxsize, linewidth=np.inf, edgeitems=100, suppress=True)


LIDAR_PITCH_DEG = 15.0
LIDAR_PITCH_RAD = np.deg2rad(LIDAR_PITCH_DEG)
COS_PITCH = np.cos(LIDAR_PITCH_RAD)
SIN_PITCH = np.sin(LIDAR_PITCH_RAD)

def process_height_map(height_map: np.array, lidar_msg: PointCloud2_, max_dist: float, delete_count: int = 100):
    grid_size = height_map.shape[0]
    map_range = 2.0 * max_dist
    cell_size = map_range / grid_size  # meters per cell
    
    # Parse pointcloud
    num_points = lidar_msg.width * lidar_msg.height
    point_step = lidar_msg.point_step
    data_bytes = bytes(lidar_msg.data)

    # Clear old data (cells not updated in delete_count frames)
    old_cells = height_map[:, :, 1] > delete_count
    height_map[old_cells, 0] = 1.0 # Reset height
    height_map[old_cells, 1] = 0    # Reset age
    
    # Increment age for all cells (vectorized)
    height_map[:, :, 1] += 1
    
    # Project points onto grid
    for i in range(num_points):
        offset = i * point_step
        x = struct.unpack_from('f', data_bytes, offset)[0]
        y = struct.unpack_from('f', data_bytes, offset + 4)[0]
        z = struct.unpack_from('f', data_bytes, offset + 8)[0]
        
        # Filter invalid points
        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
            continue
        
        # Correct for lidar pitch (rotation around Y-axis)
        x_corrected = x * COS_PITCH - z * SIN_PITCH
        z_corrected = x * SIN_PITCH + z * COS_PITCH
        x, z = x_corrected, z_corrected
        x = -x
        
        # Convert to grid coordinates (robot at center)
        # Flip x-axis: high x → row 0 (top), low x → row grid_size-1 (bottom)
        grid_x = grid_size - 1 - int((x + max_dist) / cell_size)
        # grid_x = int((x + max_dist) / cell_size)
        grid_y = int((y + max_dist) / cell_size)
        if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size and z > 0.0:
            if height_map[grid_x, grid_y, 0] != 1.0:
                height_map[grid_x, grid_y, 0] = max(height_map[grid_x, grid_y, 0], z)
            else:
                height_map[grid_x, grid_y, 0] = z
            height_map[grid_x, grid_y, 1] = 0
        # height_map[height_map[:,:,0]<0.25] = None
    
    
    