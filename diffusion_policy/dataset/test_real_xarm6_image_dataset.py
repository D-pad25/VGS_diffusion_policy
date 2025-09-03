#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smoke test for RealXArm6ImageDataset using your Hydra training workspace.
- Instantiates the dataset via Hydra (same as train.py)
- Builds/loads the replay buffer (uses your conversion under the hood)
- Normalizes actions and plots a "velocity" histogram
- Prints sample tensor shapes

Usage (WSL):
  # optionally override dataset path via env var
  XARM6_DATASET_PATH=/home/you/Thesis/Data/diffusion_test/test_data \
  python -m diffusion_policy.dataset.test_real_xarm6_image_dataset
"""
import os
from pathlib import Path
import numpy as np
import matplotlib
# Use non-GUI backend if no display (e.g., WSL without X)
if os.environ.get("DISPLAY", "") == "":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

import hydra
from omegaconf import OmegaConf

def _compose_cfg(workspace_name: str, dataset_path_override: str | None):
    # Point Hydra at the repo's config/ directory robustly
    config_dir = (Path(__file__).resolve().parents[2] / "config").as_posix()
    OmegaConf.register_new_resolver("eval", eval, replace=True)

    overrides = []
    if dataset_path_override:
        overrides.append(f"task.dataset_path={dataset_path_override}")

    # initialize from an absolute config dir
    with hydra.initialize_config_dir(config_dir=config_dir, job_name="xarm6_test"):
        cfg = hydra.compose(config_name=workspace_name, overrides=overrides)
        OmegaConf.resolve(cfg)
    return cfg

def _diff_within_episodes(arr: np.ndarray, episode_ends: np.ndarray) -> np.ndarray:
    """Compute first differences within each episode (avoid boundary jumps)."""
    diffs = []
    start = 0
    for end in episode_ends:
        seg = arr[start:end]
        if len(seg) > 1:
            diffs.append(np.diff(seg, axis=0))
        start = end
    if len(diffs) == 0:
        return np.empty((0,) + arr.shape[1:], dtype=arr.dtype)
    return np.concatenate(diffs, axis=0)

def test():
    # Workspace to use (must exist in config/)
    workspace = os.environ.get(
        "XARM6_WORKSPACE",
        "train_xarm6_diffusion_unet_image_pretrained_workspace"
    )
    # Optional dataset path override (raw Episode*/step*.pkl root)
    ds_override = os.environ.get("XARM6_DATASET_PATH", None)

    cfg = _compose_cfg(workspace, ds_override)
    # Instantiate dataset from config (just like train.py)
    dataset = hydra.utils.instantiate(cfg.task.dataset)

    # --- quick shape smoke test (first sample) ---
    sample = dataset[0]
    print("wrist_rgb:", sample["obs"]["wrist_rgb"].shape)   # (2,3,224,224)
    print("base_rgb :", sample["obs"]["base_rgb"].shape)    # (2,3,224,224)
    print("robot_state:", sample["obs"]["robot_state"].shape)  # (2,7)
    print("action:", sample["action"].shape)                # (16,7)

    # --- action velocity histogram (normalized action diffs) ---
    normalizer = dataset.get_normalizer()
    actions = dataset.replay_buffer["action"][:]
    nactions = normalizer["action"].normalize(actions)

    # safer: diff within episode boundaries
    d_nactions = _diff_within_episodes(
        nactions, dataset.replay_buffer.episode_ends[:]
    )
    dists = np.linalg.norm(d_nactions, axis=-1)

    print(
        "action-velocity stats "
        f"(n={len(dists)}): mean={dists.mean():.4f}, std={dists.std():.4f}, "
        f"min={dists.min():.4f}, max={dists.max():.4f}"
    )

    plt.figure(figsize=(6,4))
    plt.hist(dists, bins=100)
    plt.title("xArm6 normalized action velocity")
    plt.xlabel("||Δa_normalized||")
    plt.ylabel("count")

    out_png = Path.cwd() / "xarm6_action_velocity_hist.png"
    try:
        # show if a display exists; otherwise just save
        if matplotlib.get_backend().lower() != "agg":
            plt.show()
        else:
            plt.savefig(out_png, dpi=120, bbox_inches="tight")
            print(f"Saved histogram to: {out_png}")
    finally:
        plt.close()

if __name__ == "__main__":
    test()
