#!/usr/bin/env python3
import time
import torch
from torch.utils.data import DataLoader
import psutil
import os

from diffusion_policy.dataset.real_xarm6_image_dataset import RealXArm6ImageDataset, EXAMPLE_SHAPE_META

def get_mem_gb():
    """Return current process RSS memory in GB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024**3)

def main():
    dataset_path = "/home/d_pad25/Thesis/Data/diffusion_test/test_data"  # change to your dataset
    batch_size = 32
    num_workers = 4

    print("Loading dataset...")
    ds = RealXArm6ImageDataset(
        shape_meta=EXAMPLE_SHAPE_META,
        dataset_path=dataset_path,
        use_cache=True,
        horizon=16,
        n_obs_steps=2,
        val_ratio=0.0,
    )
    print(f"Dataset ready with {len(ds)} sequences")

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    print("Iterating 10 batches to measure speed + RAM...")
    start = time.time()
    mem_before = get_mem_gb()

    for i, batch in enumerate(dl):
        if i >= 10:
            break
        mem_now = get_mem_gb()
        print(
            f"Batch {i+1}: "
            f"wrist_rgb={batch['obs']['wrist_rgb'].shape}, "
            f"base_rgb={batch['obs']['base_rgb'].shape}, "
            f"robot_state={batch['obs']['robot_state'].shape}, "
            f"action={batch['action'].shape}, "
            f"mem={mem_now:.2f} GB"
        )

    duration = time.time() - start
    mem_after = get_mem_gb()
    print(f"\nAvg time per batch: {duration/10:.3f}s")
    print(f"Memory before: {mem_before:.2f} GB, after: {mem_after:.2f} GB")

if __name__ == "__main__":
    main()
