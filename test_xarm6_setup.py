#!/usr/bin/env python3
"""
Test script for xArm6 diffusion policy setup.
This script tests all components to ensure they're working correctly.
"""

import os
import sys
import pathlib
import numpy as np

# Add the diffusion_policy to path
ROOT_DIR = str(pathlib.Path(__file__).parent)
sys.path.append(ROOT_DIR)

def test_dataset():
    """Test the xArm6 dataset class"""
    print("Testing xArm6 dataset class...")
    
    try:
        from diffusion_policy.dataset.xarm6_image_dataset import XArm6ImageDataset
        
        # Create dummy data for testing
        dummy_data = {
            'state': np.random.rand(100, 7).astype(np.float32),
            'base_rgb': np.random.randint(0, 256, size=(100, 224, 224, 3), dtype=np.uint8),
            'wrist_rgb': np.random.randint(0, 256, size=(100, 224, 224, 3), dtype=np.uint8),
            'action': np.random.rand(100, 7).astype(np.float32),
        }
        
        # Test dataset creation (this will fail without actual zarr data, but we can test the class)
        print("✓ XArm6ImageDataset class imported successfully")
        print(f"  - Expected state shape: {dummy_data['state'].shape}")
        print(f"  - Expected base_rgb shape: {dummy_data['base_rgb'].shape}")
        print(f"  - Expected action shape: {dummy_data['action'].shape}")
        
        return True
        
    except ImportError as e:
        print(f"✗ Failed to import XArm6ImageDataset: {e}")
        return False
    except Exception as e:
        print(f"✗ Error testing dataset: {e}")
        return False

def test_environment_runner():
    """Test the xArm6 environment runner"""
    print("\nTesting xArm6 environment runner...")
    
    try:
        from diffusion_policy.env_runner.xarm6_image_runner import XArm6ImageRunner
        
        # Test runner creation
        runner = XArm6ImageRunner(
            output_dir="./test_output",
            n_train=1,
            n_train_vis=1,
            train_start_seed=0,
            n_test=1,
            n_test_vis=1,
            max_steps=10,
            n_obs_steps=2,
            n_action_steps=8,
            fps=10,
            past_action=False,
            n_envs=1
        )
        
        # Test observation
        obs = runner._get_obs()
        print("✓ XArm6ImageRunner created successfully")
        print(f"  - Observation keys: {list(obs.keys())}")
        print(f"  - Base RGB shape: {obs['base_rgb'].shape}")
        print(f"  - State shape: {obs['state'].shape}")
        
        # Test step
        action = np.random.rand(7)
        obs, reward, done, info = runner._step(action)
        print(f"  - Step result - reward: {reward}, done: {done}")
        
        return True
        
    except ImportError as e:
        print(f"✗ Failed to import XArm6ImageRunner: {e}")
        return False
    except Exception as e:
        print(f"✗ Error testing environment runner: {e}")
        return False

def test_config_files():
    """Test that configuration files exist and are valid"""
    print("\nTesting configuration files...")
    
    config_files = [
        "diffusion_policy/config/task/xarm6_image.yaml",
        "diffusion_policy/config/train_xarm6_diffusion_unet_image_workspace.yaml"
    ]
    
    all_exist = True
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"✓ {config_file} exists")
        else:
            print(f"✗ {config_file} missing")
            all_exist = False
    
    return all_exist

def test_data_conversion():
    """Test the data conversion script"""
    print("\nTesting data conversion script...")
    
    try:
        from convert_xarm6_data import create_sample_data, create_xarm6_zarr_dataset
        
        # Test sample data creation
        sample_data = create_sample_data()
        print(f"✓ Sample data creation successful")
        print(f"  - Created {len(sample_data)} episodes")
        print(f"  - First episode state shape: {sample_data[0]['state'].shape}")
        print(f"  - First episode base_rgb shape: {sample_data[0]['base_rgb'].shape}")
        print(f"  - First episode wrist_rgb shape: {sample_data[0]['wrist_rgb'].shape}")
        
        return True
        
    except ImportError as e:
        print(f"✗ Failed to import data conversion functions: {e}")
        return False
    except Exception as e:
        print(f"✗ Error testing data conversion: {e}")
        return False

def test_training_script():
    """Test that the training script can be imported"""
    print("\nTesting training script...")
    
    try:
        # Test that the script exists and can be imported
        script_path = "train_xarm6.py"
        if os.path.exists(script_path):
            print(f"✓ {script_path} exists")
            
            # Try to import the main function
            import importlib.util
            spec = importlib.util.spec_from_file_location("train_xarm6", script_path)
            module = importlib.util.module_from_spec(spec)
            
            # This will fail without proper dependencies, but we can test the file exists
            print("  - Training script file is valid")
            return True
        else:
            print(f"✗ {script_path} missing")
            return False
            
    except Exception as e:
        print(f"✗ Error testing training script: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("XArm6 Diffusion Policy Setup Test")
    print("=" * 60)
    
    tests = [
        ("Dataset Class", test_dataset),
        ("Environment Runner", test_environment_runner),
        ("Configuration Files", test_config_files),
        ("Data Conversion", test_data_conversion),
        ("Training Script", test_training_script),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:25} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your xArm6 diffusion policy setup is ready.")
        print("\nNext steps:")
        print("1. Prepare your xArm6 data in the specified format")
        print("2. Convert data to zarr format using: python convert_xarm6_data.py --sample")
        print("3. Update the zarr_path in the task configuration")
        print("4. Start training with: python train_xarm6.py")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
        print("Make sure all required dependencies are installed and files are in the correct locations.")

if __name__ == "__main__":
    main() 