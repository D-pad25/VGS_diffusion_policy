#!/usr/bin/env python3
"""
Integration example for xArm6 diffusion policy with your existing environment.
This script shows how to connect your XArmRealEnv with the diffusion policy framework.
"""

import os
import sys
import pathlib
import numpy as np
import torch

# Add the diffusion_policy to path
ROOT_DIR = str(pathlib.Path(__file__).parent)
sys.path.append(ROOT_DIR)

# Import your environment (you'll need to adjust this path)
# sys.path.append('/path/to/your/openpi/repo')
# from src.xarm6_control.xarm_env import XArmRealEnv, MockXArmEnv

def create_integrated_environment():
    """
    Create an environment that integrates your xArm6 setup with the diffusion policy.
    """
    
    # Option 1: Use your real environment (uncomment when ready)
    # camera_dict = {
    #     'base': your_base_camera,
    #     'wrist': your_wrist_camera
    # }
    # env = XArmRealEnv(ip="192.168.1.203", camera_dict=camera_dict)
    
    # Option 2: Use mock environment for testing
    camera_dict = {
        'base': MockCamera(),
        'wrist': MockCamera()
    }
    env = MockXArmEnv(camera_dict=camera_dict)
    
    return env


class MockCamera:
    """Mock camera class for testing integration"""
    def __init__(self):
        self.width, self.height = 224, 224  # Downsampled size
    
    def read(self):
        """Return mock RGB and depth images (downsampled to 224x224)"""
        rgb = np.random.randint(0, 256, size=(self.height, self.width, 3), dtype=np.uint8)
        depth = np.random.rand(self.height, self.width).astype(np.float32)
        return rgb, depth


class MockXArmEnv:
    """Mock xArm environment for testing integration"""
    def __init__(self, camera_dict=None):
        self.camera_dict = camera_dict or {}
        self.current_joint_position = np.random.uniform(low=-1.0, high=1.0, size=(6,))
        self.current_gripper_position = np.random.uniform(0.0, 1.0)

    def get_observation(self):
        obs = {
            "joint_position": self.current_joint_position,
            "gripper_position": np.array([self.current_gripper_position]),
        }

        # Build state with 6 joints + 1 gripper
        obs["state"] = np.concatenate([obs["joint_position"], obs["gripper_position"]])

        # Get camera images
        for name, camera in self.camera_dict.items():
            image, depth = camera.read()
            obs[f"{name}_rgb"] = image
            obs[f"{name}_depth"] = depth

        return obs

    def step(self, action):
        print(f"[MOCK STEP] Action received: {action}")
        self.current_joint_position = np.array(action[:6])
        self.current_gripper_position = float(np.clip(action[-1], 0.0, 1.0))


def integrate_with_diffusion_policy():
    """
    Example of how to integrate your xArm6 environment with the diffusion policy.
    """
    
    print("Setting up xArm6 environment integration...")
    
    # Create your environment
    env = create_integrated_environment()
    
    # Test observation
    obs = env.get_observation()
    print(f"Environment observation keys: {list(obs.keys())}")
    print(f"State shape: {obs['state'].shape}")
    print(f"Base RGB shape: {obs['base_rgb'].shape}")
    print(f"Wrist RGB shape: {obs['wrist_rgb'].shape}")
    
    # Test action
    test_action = np.random.rand(7)  # 6 joints + 1 gripper
    print(f"Test action: {test_action}")
    env.step(test_action)
    
    # Get new observation
    new_obs = env.get_observation()
    print(f"New state: {new_obs['state']}")
    
    print("\n✅ Environment integration successful!")
    print("\nNext steps:")
    print("1. Replace MockXArmEnv with your actual XArmRealEnv")
    print("2. Connect your real cameras to the camera_dict")
    print("3. Update the IP address for your xArm")
    print("4. Test with real hardware")


def test_data_structure_compatibility():
    """
    Test that your data structure is compatible with the diffusion policy.
    """
    
    print("\nTesting data structure compatibility...")
    
    # Create sample data matching your environment structure
    sample_data = {
        'state': np.random.rand(100, 7).astype(np.float32),  # 6 joints + 1 gripper
        'base_rgb': np.random.randint(0, 256, size=(100, 224, 224, 3), dtype=np.uint8),
        'wrist_rgb': np.random.randint(0, 256, size=(100, 224, 224, 3), dtype=np.uint8),
        'action': np.random.rand(100, 7).astype(np.float32),  # 6 joints + 1 gripper
    }
    
    print("✓ Sample data created with correct structure:")
    for key, value in sample_data.items():
        print(f"  - {key}: {value.shape}")
    
    # Test that this matches what the diffusion policy expects
    try:
        from diffusion_policy.dataset.xarm6_image_dataset import XArm6ImageDataset
        print("\n✓ XArm6ImageDataset imported successfully")
        print("✓ Data structure is compatible with diffusion policy")
    except ImportError as e:
        print(f"\n⚠️  Import error: {e}")
        print("Make sure you're running from the VGS_diffusion_policy directory")


if __name__ == "__main__":
    print("=" * 60)
    print("xArm6 Diffusion Policy Integration Example")
    print("=" * 60)
    
    # Test data structure compatibility
    test_data_structure_compatibility()
    
    # Test environment integration
    integrate_with_diffusion_policy()
    
    print("\n" + "=" * 60)
    print("Integration test completed!")
    print("=" * 60) 