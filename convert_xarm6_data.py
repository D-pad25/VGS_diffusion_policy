#!/usr/bin/env python3
"""
Data conversion script for xArm6 data to zarr format.
This script helps convert your xArm6 data to the format expected by the diffusion policy framework.
"""

import os
import sys
import numpy as np
import zarr
from pathlib import Path
import argparse
from typing import Dict, List, Any
import json

def create_xarm6_zarr_dataset(
    output_path: str,
    data_samples: List[Dict[str, Any]],
    episode_lengths: List[int] = None,
    chunk_size: int = 1000
):
    """
    Create a zarr dataset for xArm6 data.
    
    Args:
        output_path: Path to save the zarr dataset
        data_samples: List of data samples, each containing:
            - state: np.array of shape (T, 7) - 6 joints + 1 gripper
            - base_rgb: np.array of shape (T, 224, 224, 3) - Base camera images (downsampled)
            - wrist_rgb: np.array of shape (T, 224, 224, 3) - Wrist camera images (downsampled)
            - action: np.array of shape (T, 7) - 6 joints + 1 gripper
        episode_lengths: List of episode lengths (if not provided, will be inferred)
        chunk_size: Chunk size for zarr arrays
    """
    
    # Create output directory
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize zarr group
    root = zarr.open_group(str(output_path), mode='w')
    
    # Prepare data arrays
    all_states = []
    all_images = []
    all_wrist_images = []
    all_actions = []
    all_prompts = []
    
    if episode_lengths is None:
        episode_lengths = []
    
    for i, sample in enumerate(data_samples):
        # Validate sample
        required_keys = ['state', 'base_rgb', 'wrist_rgb', 'action']
        for key in required_keys:
            if key not in sample:
                raise ValueError(f"Sample {i} missing required key: {key}")
        
        # Get episode length
        episode_len = len(sample['state'])
        episode_lengths.append(episode_len)
        
        # Validate shapes
        if sample['state'].shape != (episode_len, 7):
            raise ValueError(f"Sample {i} state shape {sample['state'].shape} != (T, 7)")
        if sample['base_rgb'].shape != (episode_len, 224, 224, 3):
            raise ValueError(f"Sample {i} base_rgb shape {sample['base_rgb'].shape} != (T, 224, 224, 3)")
        if sample['wrist_rgb'].shape != (episode_len, 224, 224, 3):
            raise ValueError(f"Sample {i} wrist_rgb shape {sample['wrist_rgb'].shape} != (T, 224, 224, 3)")
        if sample['action'].shape != (episode_len, 7):
            raise ValueError(f"Sample {i} action shape {sample['action'].shape} != (T, 7)")
        
        # Append data
        all_states.append(sample['state'])
        all_images.append(sample['base_rgb'])
        all_wrist_images.append(sample['wrist_rgb'])
        all_actions.append(sample['action'])
    
    # Concatenate all episodes
    all_states = np.concatenate(all_states, axis=0)
    all_images = np.concatenate(all_images, axis=0)
    all_wrist_images = np.concatenate(all_wrist_images, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)
    
    # Create zarr arrays
    root.create_dataset('state', data=all_states, chunks=(chunk_size, 7), dtype=np.float32)
    root.create_dataset('base_rgb', data=all_images, chunks=(chunk_size, 224, 224, 3), dtype=np.uint8)
    root.create_dataset('wrist_rgb', data=all_wrist_images, chunks=(chunk_size, 224, 224, 3), dtype=np.uint8)
    root.create_dataset('action', data=all_actions, chunks=(chunk_size, 7), dtype=np.float32)
    
    # Store episode lengths and metadata
    root.attrs['episode_lengths'] = episode_lengths
    root.attrs['n_episodes'] = len(episode_lengths)
    root.attrs['total_steps'] = len(all_states)
    
    print(f"Created zarr dataset at {output_path}")
    print(f"Total episodes: {len(episode_lengths)}")
    print(f"Total steps: {len(all_states)}")
    print(f"State shape: {all_states.shape}")
    print(f"Image shape: {all_images.shape}")
    print(f"Action shape: {all_actions.shape}")
    
    return output_path


def create_sample_data():
    """
    Create sample data for testing the conversion script.
    """
    # Create sample episodes
    episodes = []
    
    for episode_idx in range(5):
        episode_len = np.random.randint(50, 100)
        
        # Create sample data
        state = np.random.rand(episode_len, 7).astype(np.float32)
        base_rgb = np.random.randint(0, 256, size=(episode_len, 224, 224, 3), dtype=np.uint8)
        wrist_rgb = np.random.randint(0, 256, size=(episode_len, 224, 224, 3), dtype=np.uint8)
        action = np.random.rand(episode_len, 7).astype(np.float32)
        
        episode = {
            'state': state,
            'base_rgb': base_rgb,
            'wrist_rgb': wrist_rgb,
            'action': action,
        }
        episodes.append(episode)
    
    return episodes


def main():
    parser = argparse.ArgumentParser(description='Convert xArm6 data to zarr format')
    parser.add_argument('--output', type=str, default='data/xarm6/xarm6_replay.zarr',
                       help='Output path for zarr dataset')
    parser.add_argument('--sample', action='store_true',
                       help='Create sample data for testing')
    parser.add_argument('--input_data', type=str, default=None,
                       help='Path to input data file (JSON, pickle, etc.)')
    
    args = parser.parse_args()
    
    if args.sample:
        print("Creating sample data...")
        data_samples = create_sample_data()
        output_path = create_xarm6_zarr_dataset(args.output, data_samples)
        print(f"Sample dataset created at: {output_path}")
        
    elif args.input_data:
        print(f"Loading data from {args.input_data}...")
        # You'll need to implement loading your specific data format here
        # For example:
        # if args.input_data.endswith('.json'):
        #     with open(args.input_data, 'r') as f:
        #         data_samples = json.load(f)
        # elif args.input_data.endswith('.pkl'):
        #     import pickle
        #     with open(args.input_data, 'rb') as f:
        #         data_samples = pickle.load(f)
        
        print("Data loading not implemented yet. Please implement based on your data format.")
        
    else:
        print("No action specified. Use --sample to create sample data or --input_data to convert existing data.")
        print("\nExample usage:")
        print("  python convert_xarm6_data.py --sample")
        print("  python convert_xarm6_data.py --input_data path/to/your/data.json --output data/xarm6/xarm6_replay.zarr")


if __name__ == "__main__":
    main() 