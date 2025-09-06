#!/usr/bin/env python3
"""
Smoke test for the low-memory xArm6 dataset + converter.

What it does:
  1) Instantiates the low-mem dataset (which will stream-build the zarr cache on first run).
  2) Prints basic info (episodes, seq count, cache path).
  3) Pulls a few samples directly.
  4) Runs a small DataLoader loop (optional) to ensure batching works.
  5) Builds the normalizer (streamed mean/std).

Run:
  python tools/smoke_test_real_xarm6_low_mem.py \
      --dataset_path /path/to/raw/episodes \
      --out_dir /path/to/cache_dir \
      --store dir \
      --horizon 16 --n_obs_steps 2 \
      --batch_size 2 --num_workers 0 --steps 3 \
      --delta_action

Tip: use --store dir for fastest iteration and lowest RAM;
     use --store zip if you prefer a single compressed cache file.
"""

import os
import json
import time
import random
import hashlib
import argparse
import numpy as np
import torch
import cv2
from omegaconf import OmegaConf

# Keep OpenCV + PyTorch threading modest (avoid surprise CPU/RAM spikes)
cv2.setNumThreads(1)
torch.set_num_threads(1)

from diffusion_policy.dataset.real_xarm6_image_dataset_low_mem import (
    RealXArm6ImageDataset, EXAMPLE_SHAPE_META
)

def _expected_cache_path(target_dir, shape_meta, camera_res, work_store_kind):
    """Reproduce the dataset's cache naming to print where the cache should live."""
    fp = {"shape_meta": OmegaConf.to_container(shape_meta), "camera_res": camera_res}
    h = hashlib.md5(json.dumps(fp, sort_keys=True).encode("utf-8")).hexdigest()
    return os.path.join(target_dir, f"{h}.zarr" + (".zip" if work_store_kind == "zip" else ""))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path with raw episodes (Episode*/step*.pkl).")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Where to write/open the zarr cache (defaults to dataset_path).")
    parser.add_argument("--store", type=str, choices=["dir", "zip"], default="dir",
                        help="Cache format: directory or zipped store.")
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--n_obs_steps", type=int, default=2)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--delta_action", action="store_true")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--steps", type=int, default=3, help="Batches to iterate in the loader.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Camera res here just for expected-path print. The dataset itself uses (224, 224) default.
    camera_res = (224, 224)
    target_dir = args.out_dir if args.out_dir is not None else args.dataset_path
    print(f"[smoke] Expected cache path: {_expected_cache_path(target_dir, EXAMPLE_SHAPE_META, camera_res, args.store)}")

    t0 = time.time()
    ds = RealXArm6ImageDataset(
        shape_meta=EXAMPLE_SHAPE_META,
        dataset_path=args.dataset_path,
        out_dir=args.out_dir,
        use_cache=True,
        work_store_kind=args.store,
        horizon=args.horizon,
        n_obs_steps=args.n_obs_steps,
        val_ratio=args.val_ratio,
        delta_action=args.delta_action
    )
    t1 = time.time()
    print(f"[smoke] Dataset instantiated in {t1 - t0:.2f}s")
    print(f"[smoke] Episodes: {ds.replay_buffer.n_episodes}")
    print(f"[smoke] Train sequences (len(ds)): {len(ds)}")
    print(f"[smoke] RGB keys: {ds.rgb_keys}")
    print(f"[smoke] Low-dim keys: {ds.lowdim_keys}")

    # --- Direct item pulls ---
    print("[smoke] Sampling a few items directly...")
    for i in range(min(3, len(ds))):
        sample = ds[i]
        obs = sample["obs"]
        act = sample["action"]
        for k in ds.rgb_keys:
            print(f"  obs[{k}]: {tuple(obs[k].shape)} dtype={obs[k].dtype} (min={obs[k].min():.3f}, max={obs[k].max():.3f})")
        for k in ds.lowdim_keys:
            print(f"  obs[{k}]: {tuple(obs[k].shape)} dtype={obs[k].dtype}")
        print(f"  action: {tuple(act.shape)} dtype={act.dtype}")

    # --- DataLoader loop (optional) ---
    if args.steps > 0:
        print(f"[smoke] Running a small DataLoader loop: batch_size={args.batch_size}, num_workers={args.num_workers}")
        from torch.utils.data import DataLoader
        loader = DataLoader(
            ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=False, drop_last=False
        )
        it = iter(loader)
        for step in range(args.steps):
            t2 = time.time()
            batch = next(it)
            t3 = time.time()
            print(f"  step {step}: load {t3 - t2:.3f}s")
            # show shapes
            for k in ds.rgb_keys:
                print(f"    batch.obs[{k}]: {tuple(batch['obs'][k].shape)}")
            for k in ds.lowdim_keys:
                print(f"    batch.obs[{k}]: {tuple(batch['obs'][k].shape)}")
            print(f"    batch.action: {tuple(batch['action'].shape)}")

    # --- Normalizer (streamed) ---
    print("[smoke] Fitting normalizer (streamed mean/std)...")
    t4 = time.time()
    norm = ds.get_normalizer()
    t5 = time.time()
    print(f"[smoke] Normalizer ready in {t5 - t4:.2f}s")
    # action mean/std preview (won't dump huge arrays—just shapes)
    amean = norm["action"].mean
    astd  = norm["action"].std
    try:
        print(f"  action mean shape: {tuple(amean.shape)}  std shape: {tuple(astd.shape)}")
    except Exception:
        print("  action normalizer present.")

    print("[smoke] ✅ Done.")

if __name__ == "__main__":
    main()
