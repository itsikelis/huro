#!/usr/bin/env python3

import numpy as np
import yaml
import os



class Mapper:
    """
    Buffer to maintain state needed for observation construction.
    """
    def __init__(self, mapping_yaml_path):
        
        self.default_pos_sdk = np.array([
            0.0, 0.8, -1.5,  # FR: hip, thigh, calf (actuators 0-2)
            0.0, 0.8, -1.5,  # FL: hip, thigh, calf (actuators 3-5)
            0.0, 0.8, -1.5,  # RR: hip, thigh, calf (actuators 6-8)
            0.0, 0.8, -1.5   # RL: hip, thigh, calf (actuators 9-11)
        ])
        
        
        self.target_to_source, self.source_to_target, self.source_names, self.target_names = self.load_joint_mapping(mapping_yaml_path)
        
        # Pre-compute default_pos in policy order for observations
        self.default_pos_policy = self.remap_joints_by_name(
            self.default_pos_sdk, self.target_names, self.source_names, self.target_to_source
        )
        
        print(f"[INFO] Loaded joint mapping from: {mapping_yaml_path}")
        print(f"[INFO] Target to Source (SDK->Policy): {self.target_to_source}")
        print(f"[INFO] Source to Target (Policy->SDK): {self.source_to_target}")
        print(f"[INFO] Default pos SDK order: {self.default_pos_sdk}")
        print(f"[INFO] Default pos Policy order: {self.default_pos_policy}")

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
        
    
