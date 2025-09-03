"""
RealXArm6ImageDataset
---------------------
Drop-in dataset class adapted for Daniel's xArm6 setup with two RealSense cameras
(wrist & base) and 7-DoF action/state (6 joints + gripper). Keeps images at 224x224.

Assumptions
- Raw data is converted via `real_data_to_replay_buffer` into a ReplayBuffer with keys:
    - 'base_rgb'    : (T, H, W, 3) uint8
    - 'wrist_rgb'   : (T, H, W, 3) uint8
    - 'robot_state' : (T, 7) float32   # [j1..j6, gripper]
    - 'action'      : (T, 7) float32   # same convention as robot_state
    - 'episode_ends': (E,) int64       # cumulative T indices
- Units: keep whatever your logger produced (deg or rad), but be consistent for
  both state and action. The normalizer will scale appropriately.
- Image size: 224x224 (C,H,W) after dataset transforms.
"""

from typing import Dict, List, Tuple, Optional
import os
import json
import shutil
import hashlib
import copy

import numpy as np
import torch
import zarr
import cv2

from filelock import FileLock
from threadpoolctl import threadpool_limits
from omegaconf import OmegaConf

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import (
    LinearNormalizer, SingleFieldLinearNormalizer
)
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask
)
from diffusion_policy.real_world.real_xarm6_data_conversion import real_data_to_replay_buffer
from diffusion_policy.common.normalize_util import (
    get_image_range_normalizer
)


def _zarr_resize_index_last_dim(zarr_arr, idxs: List[int]):
    """In-place slice on last dimension of a zarr array and resize storage."""
    data = zarr_arr[:]
    data = data[..., idxs]
    zarr_arr.resize(zarr_arr.shape[:-1] + (len(idxs),))
    zarr_arr[:] = data
    return zarr_arr


def _get_replay_buffer(
    dataset_path: str,
    shape_meta: dict,
    store: zarr.storage.BaseStore,
    camera_res: Tuple[int, int] = (224, 224),
) -> ReplayBuffer:
    """Build a ReplayBuffer from real data, resized to camera_res, with keys from shape_meta."""
    # Parse shape_meta
    rgb_keys: List[str] = []
    lowdim_keys: List[str] = []
    out_resolutions = {}
    lowdim_shapes = {}

    obs_shape_meta = shape_meta["obs"]
    for key, attr in obs_shape_meta.items():
        typ = attr.get("type", "low_dim")
        shp = tuple(attr.get("shape"))
        if typ == "rgb":
            rgb_keys.append(key)
            # shape in shape_meta is (C,H,W); cv2 wants (W,H) as out size
            out_resolutions[key] = (camera_res[0], camera_res[1])
        elif typ == "low_dim":
            lowdim_keys.append(key)
            lowdim_shapes[key] = shp

    action_shape = tuple(shape_meta["action"]["shape"])
    assert action_shape in [(2,), (6,), (7,)], f"Unexpected action shape: {action_shape}"

    # Build replay buffer
    cv2.setNumThreads(1)
    with threadpool_limits(1):
        rb = real_data_to_replay_buffer(
            dataset_path=dataset_path,
            out_store=store,
            out_resolutions=out_resolutions,         # resize images here
            lowdim_keys=lowdim_keys + ["action"],     # include action as low-dim to ingest
            image_keys=rgb_keys
        )

    # If action/state are larger than expected, optionally slice them.
    if action_shape == (2,):
        _zarr_resize_index_last_dim(rb["action"], idxs=[0, 1])
    elif action_shape == (6,):
        _zarr_resize_index_last_dim(rb["action"], idxs=[0, 1, 2, 3, 4, 5])
    elif action_shape == (7,):
        pass  # full 6 joints + gripper

    # Example: if a 'robot_state' got logged with >7 dims, keep first 7
    if "robot_state" in rb and rb["robot_state"].shape[-1] > 7:
        _zarr_resize_index_last_dim(rb["robot_state"], idxs=list(range(7)))

    return rb


class RealXArm6ImageDataset(BaseImageDataset):
    """
    Real-world xArm6 dataset for Diffusion Policy, images at 224x224 and 7-DoF actions by default.
    Includes:
      - cache to zipped zarr
      - optional delta-action (with option to exclude gripper from delta)
      - train/val split with downsampling
      - image normalization to [0,1] and channel-first conversion
      - latency support
    """
    def __init__(
        self,
        shape_meta: dict,
        dataset_path: str,
        horizon: int = 16,
        pad_before: int = 0,
        pad_after: int = 0,
        n_obs_steps: Optional[int] = None,
        n_latency_steps: int = 0,
        use_cache: bool = True,
        seed: int = 42,
        val_ratio: float = 0.1,
        max_train_episodes: Optional[int] = None,
        delta_action: bool = False,
        delta_exclude_gripper: bool = True,
        camera_res: Tuple[int, int] = (224, 224),
    ) -> None:
        assert os.path.isdir(dataset_path), f"dataset_path not found: {dataset_path}"

        # Build or load cache
        replay_buffer: Optional[ReplayBuffer] = None
        if use_cache:
            # fingerprint shape_meta + camera_res (so cache refreshes if res changes)
            fp = {
                "shape_meta": OmegaConf.to_container(shape_meta),
                "camera_res": camera_res
            }
            shape_meta_json = json.dumps(fp, sort_keys=True)
            shape_meta_hash = hashlib.md5(shape_meta_json.encode("utf-8")).hexdigest()
            cache_zarr_path = os.path.join(dataset_path, f"{shape_meta_hash}.zarr.zip")
            cache_lock_path = cache_zarr_path + ".lock"

            print(f"[RealXArm6ImageDataset] Acquiring cache lock: {cache_lock_path}")
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    try:
                        print("[RealXArm6ImageDataset] Cache miss → building ReplayBuffer in memory...")
                        replay_buffer = _get_replay_buffer(
                            dataset_path=dataset_path,
                            shape_meta=shape_meta,
                            store=zarr.MemoryStore(),
                            camera_res=camera_res
                        )
                        print("[RealXArm6ImageDataset] Saving cache to disk...")
                        with zarr.ZipStore(cache_zarr_path, mode="w") as zip_store:
                            replay_buffer.save_to_store(store=zip_store)
                        print("[RealXArm6ImageDataset] Cache saved.")
                    except Exception as e:
                        # If a partial file exists, remove it
                        if os.path.exists(cache_zarr_path):
                            shutil.rmtree(cache_zarr_path, ignore_errors=True)
                        raise e
                else:
                    print("[RealXArm6ImageDataset] Cache hit → loading ReplayBuffer from disk...")
                    with zarr.ZipStore(cache_zarr_path, mode="r") as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store, store=zarr.MemoryStore()
                        )
                    print("[RealXArm6ImageDataset] Cache loaded.")
        else:
            replay_buffer = _get_replay_buffer(
                dataset_path=dataset_path,
                shape_meta=shape_meta,
                store=zarr.MemoryStore(),
                camera_res=camera_res
            )

        assert replay_buffer is not None

        # Optionally convert actions to deltas (joints only by default)
        if delta_action:
            actions = replay_buffer["action"][:]
            n_dim = actions.shape[1]
            episode_ends = replay_buffer.episode_ends[:]
            actions_diff = np.zeros_like(actions)

            # indices to delta; default exclude gripper (index 6 when n_dim==7)
            if delta_exclude_gripper and n_dim == 7:
                delta_idxs = list(range(6))  # 0..5 joints
            else:
                delta_idxs = list(range(n_dim))

            for epi in range(len(episode_ends)):
                start = 0 if epi == 0 else episode_ends[epi - 1]
                end = episode_ends[epi]
                # schedule delta for t as (a_t - a_{t-1}) applied at index t
                # leave the very first step as zeros
                seg = actions[start:end]
                d = np.diff(seg, axis=0, prepend=seg[0:1])
                # apply only on selected indices
                actions_diff[start:end, delta_idxs] = d[:, delta_idxs]

                # if we excluded gripper from deltas, keep it as absolute command
                if delta_exclude_gripper and n_dim == 7:
                    actions_diff[start:end, 6] = seg[:, 6]

            replay_buffer["action"][:] = actions_diff

        # Parse obs keys
        rgb_keys: List[str] = []
        lowdim_keys: List[str] = []
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            typ = attr.get("type", "low_dim")
            if typ == "rgb":
                rgb_keys.append(key)
            elif typ == "low_dim":
                lowdim_keys.append(key)

        key_first_k = {}
        if n_obs_steps is not None:
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps

        # Train/Val split (by episodes)
        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, max_n=max_train_episodes, seed=seed
        )

        sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=horizon + n_latency_steps,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k
        )

        # Expose
        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.n_obs_steps = n_obs_steps
        self.val_mask = val_mask
        self.horizon = horizon
        self.n_latency_steps = n_latency_steps
        self.pad_before = pad_before
        self.pad_after = pad_after

    # -------------------- Public API --------------------
    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon + self.n_latency_steps,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.val_mask
        )
        val_set.val_mask = ~self.val_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # action
        normalizer["action"] = SingleFieldLinearNormalizer.create_fit(
            self.replay_buffer["action"]
        )

        # low-dim obs
        for key in self.lowdim_keys:
            normalizer[key] = SingleFieldLinearNormalizer.create_fit(
                self.replay_buffer[key]
            )

        # images → [0,1]
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()

        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer["action"])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Limit threads for dataloader workers
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)

        # Only keep first n_obs_steps of obs (if set)
        T_slice = slice(self.n_obs_steps)

        obs_dict = {}
        for key in self.rgb_keys:
            # (T,H,W,C) uint8 → (T,C,H,W) float32 in [0,1]
            obs_dict[key] = np.moveaxis(data[key][T_slice], -1, 1).astype(np.float32) / 255.0
            del data[key]

        for key in self.lowdim_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            del data[key]

        action = data["action"].astype(np.float32)
        if self.n_latency_steps > 0:
            action = action[self.n_latency_steps:]

        torch_data = {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "action": torch.from_numpy(action)
        }
        return torch_data


# -------------------- Convenience: example shape_meta --------------------

EXAMPLE_SHAPE_META = {
    "obs": {
        "base_rgb":   {"shape": [3, 224, 224], "type": "rgb"},
        "wrist_rgb":  {"shape": [3, 224, 224], "type": "rgb"},
        "robot_state": {"shape": [7], "type": "low_dim"}  # [j1..j6, gripper]
    },
    "action": {"shape": [7]}  # [j1..j6, gripper]
}



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="Path with raw episodes for real_data_to_replay_buffer")
    parser.add_argument("--use_cache", action="store_true", help="Cache ReplayBuffer to zipped zarr")
    parser.add_argument("--delta_action", action="store_true", help="Use delta actions (joints-only by default)")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--n_obs_steps", type=int, default=2)
    args = parser.parse_args()

    ds = RealXArm6ImageDataset(
        shape_meta=EXAMPLE_SHAPE_META,
        dataset_path=args.dataset_path,
        use_cache=args.use_cache,
        delta_action=args.delta_action,
        val_ratio=args.val_ratio,
        horizon=args.horizon,
        n_obs_steps=args.n_obs_steps,
    )
    print("Dataset ready!")
    print(f"- #episodes: {ds.replay_buffer.n_episodes}")
    print(f"- #seq (train): {len(ds)}")
    print(f"- rgb keys: {ds.rgb_keys}")
    print(f"- low-dim keys: {ds.lowdim_keys}")

