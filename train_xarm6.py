#!/usr/bin/env python3
"""
Training script for xArm6 diffusion policy.
This script follows the same data structure as your pi0 implementation.
"""

import os
import sys
import pathlib
import hydra
from omegaconf import OmegaConf

# Add the diffusion_policy to path
ROOT_DIR = str(pathlib.Path(__file__).parent)
sys.path.append(ROOT_DIR)

@hydra.main(config_path="diffusion_policy/config", config_name="train_xarm6_diffusion_unet_image_workspace")
def main(cfg: OmegaConf):
    """
    Main training function for xArm6 diffusion policy.
    
    This script will:
    1. Load the xArm6 dataset with your data structure
    2. Train a diffusion policy using the same format as pi0
    3. Save checkpoints and logs
    """
    print("Starting xArm6 diffusion policy training...")
    print(f"Configuration: {cfg}")
    
    # Instantiate and run the workspace
    workspace = hydra.utils.instantiate(cfg)
    
    print("Workspace instantiated successfully. Starting training...")
    workspace.run()
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main() 